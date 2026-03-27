try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch
import time
import copy

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

parser.add_argument(
    "--reach_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Reach partial pose",
    type=float,
    default=None,
)
parser.add_argument(
    "--hold_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Hold partial pose while moving to goal",
    type=float,
    default=None,
)

parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--physics_step",
    type=int,
    default=60,
)

args = parser.parse_args()

############################################################

# Third Party
# from omni.isaac.kit import SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
simulation_app._carb_settings.set("/omni/replicator/asyncRendering", False)
simulation_app._carb_settings.set("/app/asyncRendering", False)

# Standard Library
from typing import Dict, Optional, Tuple

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene

from isaacsim.core.api import World
from isaacsim.core.api.objects import cuboid, sphere, cylinder
from isaacsim.core.utils import extensions
from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim
from isaacsim.core.prims import SingleRigidPrim as RigidPrim
from isaacsim.core.prims import SingleXFormPrim as XFormPrim

from isaacsim.core.utils.prims import (
    delete_prim,
    get_prim_at_path,
    get_prim_children,
    get_prim_object_type,
)
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.physics import _sensor

try:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    pass

import omni
import asyncio
import yaml
import re

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.util.usd_helper import UsdHelper, get_prim_world_pose
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

from scipy.spatial.transform import Rotation

import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    collection_dir = os.path.join(root_dir, "source/data_collection")
    sys.path.append(collection_dir)
from source.data_collection.common.base_utils import transform_utils
from source.data_collection.server.motion_generator.mesh_utils import get_mesh_attrs, simplify_obstacles_from_stage
from source.data_collection.server.motion_generator.path_filters import (
    filter_paths_by_position_error,
    filter_paths_by_rotation_error,
    sort_by_difference_js,
)
from source.data_collection.server.controllers.kinematics_solver import KinematicsSolver
from source.data_collection.server.controllers.ruckig_move import RuckigController
from source.data_collection.server.robot import RobotCfg
from source.data_collection.server.command_enum import Command


CUROBO_BATCH_SIZE = 10
MAX_MESH_FACES = 1000

# ============================================================
# NOTE: Vertical adjustment constants for galbot.
# These need calibration with the actual robot URDF.
# The simple approach: keep torso vertical via constraint joint1 + joint3 ≈ joint2.
# See CLAUDE.md "vertical" task specification for details.
# ============================================================

# Default reference leg joint values (from retract_config in basic_test.yaml).
# Treated as the maximum height state - no torso increase beyond this.
GALBOT_LEG_DEFAULT = {
    "leg_joint1": 0.5236,   # ~30 deg, hip pitch
    "leg_joint2": 1.0821,   # ~62 deg, knee pitch
    "leg_joint3": 0.6109,   # ~35 deg, ankle pitch
}
# Constraint: joint1 + joint3 ≈ joint2 (keeps torso approximately vertical)

# Approximate arm workspace geometry relative to torso_base_link.
# CALIBRATION NOTE: Measure actual values in Isaac Sim by checking FK of arm at
# extreme poses. These are rough estimates for galbot one golf.
GALBOT_SHOULDER_Z_ABOVE_TORSO = 0.15   # meters, torso_base_link to shoulder Z offset
GALBOT_ARM_REACH_RADIUS = 0.70          # meters, approximate sphere radius of arm workspace

# Height decrease per unit of leg bending delta (meters per radian).
# Calibrated via calibrate_leg_height.py (2026-03-11):
#   baseline torso_z = 0.9174 m at default joints
#   avg m/rad for lowering (negative delta) ≈ 0.67
#   sign: NEGATIVE delta lowers torso (decreasing joints squats down)
# Formula: torso_z(delta) = torso_z_default + GALBOT_HEIGHT_PER_LEG_DELTA * delta
#          → to lower by h, use delta = -h / GALBOT_HEIGHT_PER_LEG_DELTA (negative)
GALBOT_HEIGHT_PER_LEG_DELTA = 0.67

# Maximum leg bending delta magnitude for lowering (radians, positive).
# Calibrated: tested safely to delta=-0.4 (torso drops ~0.28 m).
GALBOT_MAX_LEG_DELTA = 0.4

# Prim path to torso_base_link for world-frame height query.
GALBOT_TORSO_PRIM_PATH = "/galbot_one_golf/torso_base_link"

# Safety margin for sphere-radius reachability check (meters).
# Makes the model more conservative so IK actually succeeds when we decide to lower.
GALBOT_REACH_SAFETY_MARGIN = 0.10

# Steps for gradual leg joint trajectory (physics steps, ~1 s at 60 Hz).
GALBOT_LEG_TRAJ_STEPS = 60


class RobotUsdHelper(UsdHelper):
    def get_obstacles_from_stage(
        self, 
        only_paths = None, 
        ignore_paths = None, 
        only_substring = None, 
        ignore_substring = None, 
        reference_prim_path = None, 
        timecode = 0
    ) -> WorldConfig:
        obstacles = {
            "cuboid": None,
            "sphere": None,
            "mesh": None,
            "cylinder": None,
            "capsule": None,
        }

        r_T_w = None
        try:
            self._xform_cache.Clear()
            self._xform_cache.SetTime(timecode)
        except Exception:
            try:
                self._xform_cache = UsdGeom.XformCache(timecode)
            except Exception:
                pass

        if reference_prim_path is not None:
            reference_prim = self.stage.GetPrimAtPath(reference_prim_path)
            r_T_w, _ = get_prim_world_pose(self._xform_cache, reference_prim, inverse=True)
        
        all_items = self.stage.Traverse()
        for prim in all_items:
            prim_path = str(prim.GetPath())

            # only/ignore path filters
            if only_paths is not None and not any([prim_path.startswith(k) for k in only_paths]):
                continue
            if ignore_paths is not None and any([prim_path.startswith(k) for k in ignore_paths]):
                continue
            if only_substring is not None and not any([k in prim_path for k in only_substring]):
                continue
            if ignore_substring is not None and any([k in prim_path for k in ignore_substring]):
                continue

            # TODO
            try:
                collisionAPI = UsdPhysics.CollisionAPI.Get(self.stage, prim_path)
                if collisionAPI and not collisionAPI.GetCollisionEnabledAttr().Get():
                    continue
            except Exception:
                pass

            try:
                if prim.IsA(UsdGeom.Mesh):
                    if obstacles["mesh"] is None:
                        obstacles["mesh"] = []
                    m_data = get_mesh_attrs(prim, cache=self._xform_cache, transform=r_T_w)
                    if m_data is not None:
                        obstacles["mesh"].append(m_data)
            except Exception as e:
                print(f"Error extracting prim {prim_path}: {e}")
                continue

        world_model = WorldConfig(**obstacles)
        return world_model


class CuroboMotion:
    world_coll_checker = None
    cached_obstacle_info = {}

    def reset(self):
        self.motion_gen.clear_world_cache()
    
    def reset_link(self):
        self.link_names = self.motion_gen.kinematics.link_names
        self.ee_link_name = self.motion_gen.kinematics.ee_link
        for i in self.link_names:
            self.target_links[i] = XFormPrim(
                "/World/target_" + i,
                position=self.init_ee_pose[i]["position"],
                orientation=self.init_ee_pose[i]["orientation"],
            )

    def _get_curobo_kinematics(self):
        if getattr(self, "curobo_kinematics", None) is None and hasattr(self, "robot_cfg") and self.robot_cfg is not None:
            self.curobo_kinematics_robot_cfg = copy.deepcopy(self.robot_cfg)
            if self.curobo_kinematics_robot_cfg["kinematics"].get("link_names", None) is None:
                self.curobo_kinematics_robot_cfg["kinematics"]["link_names"] = []
            for link_name in self.curobo_kinematics_robot_cfg["kinematics"]["collision_link_names"]:
                if "arm" in link_name:
                    self.curobo_kinematics_robot_cfg["kinematics"]["link_names"].append(link_name)
            cuda_robot_model_config = CudaRobotModelConfig.from_data_dict(
                data_dict=self.curobo_kinematics_robot_cfg, tensor_args=self.tensor_args
            )
            self.curobo_kinematics = CudaRobotModel(cuda_robot_model_config)
        return getattr(self, "curobo_kinematics", None)

    def _extract_cached_obstacles(self, need_reset_cache=True):
        print(f"Extracting and caching obstacle geometries...")

        # Initialize ignore list
        self.ignore_substring_list = [
            self.robot_prim_path,
            "/World/target",
            "/World/Xform_01",
            "/World/GroundPlane",
            "/World/Environment_01",
            "/curobo",
            "/World/GroundPlane_01",
            "/World/Meshes",
            "/World/Root/Meshes",
            "/World/Objects/part",
            "/base_cube",
            "virtual_fixed_joint",
            "/World/background",
            "/World/Background",
        ]
        if need_reset_cache:
            initial_obstacles = self.usd_help.get_obstacles_from_stage(
                only_paths=None,
                ignore_substring=self.ignore_substring_list,
                reference_prim_path=None,
                timecode=0,
            )

            time0 = time.time()
            simplified_obstacles = simplify_obstacles_from_stage(initial_obstacles, max_faces=MAX_MESH_FACES)
            time1 = time.time()
            print(f"Simplified obstacles in {time1 - time0:.2f} seconds")

            CuroboMotion.cached_obstacle_info = {}

            if simplified_obstacles.mesh:
                for mesh in simplified_obstacles.mesh:
                    prim_path = mesh.name
                    CuroboMotion.cached_obstacle_info[prim_path] = {
                        "type": "mesh",
                        "geometry": mesh,
                        "original_pose": mesh.pose,
                    }
            print(f"Cache completed, extracted {len(CuroboMotion.cached_obstacle_info)} obstacle geometries")

        if self.robot_prim_path:
            reference_prim = self.usd_help.stage.GetPrimAtPath(self.robot_prim_path)
            self.robot_transform_cache = self.usd_help._xform_cache
            self.robot_transform_cache.Clear()
            self.robot_transform_cache.SetTime(0)
            self.robot_reference_prim = reference_prim
        else:
            self.robot_transform_cache = None
            self.robot_reference_prim = None
    

    def add_obstacle_from_prim_path(self, prim_path, usd_path):
        try:
            print(f"Adding obstacle from prim path: {prim_path}...")

            new_obstacles = self.usd_help.get_obstacles_from_stage(
                only_paths=[prim_path],
                ignore_substring=self.ignore_substring_list,
                reference_prim_path=usd_path,
                timecode=0,
            )

            time0 = time.time()
            simplified_obstacles = simplify_obstacles_from_stage(new_obstacles, usd_path=usd_path)
            time1 = time.time()
            print(f"Simplified obstacles in {time1 - time0:.2f} seconds")

            obstacle_types = [
                ("mesh", simplified_obstacles.mesh),
            ]

            for obstacle_type, obstacles_list in obstacle_types:
                if obstacles_list:
                    for obstacle in obstacles_list:
                        obstacle_prim_path = obstacle.name

                        if obstacle_prim_path not in CuroboMotion.cached_obstacle_info:
                            new_obstacles_count += 1
                            print(f"New obstacle: {obstacle_prim_path} (type: {obstacle_type})")
                        else:
                            print(f"Updated existing obstacle: {obstacle_prim_path} (type: {obstacle_type})")
                        
                        CuroboMotion.cached_obstacle_info[obstacle_prim_path] = {
                            "type": obstacle_type,
                            "geometry": obstacle,
                            "original_pose": obstacle.pose,
                        }

            if new_obstacles_count > 0:
                print(f"Successfully added {new_obstacles_count} new obstacles to cache")
            else:
                print(f"No new obstacles found under path {prim_path}")

            print(f"Current total cache count: {len(CuroboMotion.cached_obstacle_info)}")
            return new_obstacles_count

        except Exception as e:
            print(f"Error adding obstacles from prim path {prim_path}: {e}")
            return 0
        
    
    def add_obstacles_from_prim_paths(self, prim_paths):
        total_added = 0
        for prim_path in prim_paths:
            added_count = self.add_obstacle_from_prim_path(prim_path)
            total_added += added_count
        
        print(f"Batch addition completed, total {total_added} new obstacles added")
        return total_added


    def get_cached_obstacles_info(self):
        info = {}
        for prim_path, obstacle_info in CuroboMotion.cached_obstacle_info.items():
            info[prim_path] = {
                "type": obstacle_info["type"],
                "geometry": obstacle_info["geometry"],
            }
        return info
    
    def refresh_obstacle_cache(self):
        print("Re-extracting obstacle cache...")
        self._extract_cached_obstacles()

    def __init__(
        self,
        name: str,
        robot: Articulation,
        world: World,
        robot_cfg,
        robot_prim_path,
        robot_list,
        step=100,
        debug=False,
    ):
        self.name = name
        self.debug = debug
        self.usd_help = RobotUsdHelper()
        self.target_pose = None
        self.target_orientation = None
        self.past_pose = None
        self.past_orientation = None
        self.robot_list = robot_list
        tensor_args = TensorDeviceType()
        self.robot_prim_path = robot_prim_path
        n_obstacle_cuboids = 30
        self.init_ee_pose = {}
        n_obstacle_mesh = 30

        robot_cfg_path = get_robot_configs_path()
        self.robot_cfg = load_yaml(join_path(robot_cfg_path, robot_cfg))["robot_cfg"]
        # TODO
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = {
            "attached_object": 30,
            "left_attached_object": 30,
        }
        self.lock_joints = self.robot_cfg["kinematics"]["lock_joints"]
        self.lock_js_names = []
        if self.lock_joints:
            for key in self.lock_joints:
                self.lock_js_names.append(key)
        j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]
        self.collision_link_names = self.robot_cfg["kinematics"]["collision_link_names"]

        self.world_cfg = WorldConfig()
        motion_gen_config = MotionGen.load_from_robot_config(
            robot_cfg=self.robot_cfg,
            world_model=self.world_cfg,
            tensor_args=tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            num_trajopt_seeds=4,
            num_graph_seeds=1,    # 4
            num_ik_seeds=32,
            num_batch_ik_seeds=32,
            interpolation_dt=0.01,
            interpolation_steps=5000,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=step,
            num_trajopt_noisy_seeds=1,
            num_batch_trajopt_seeds=1,
            collision_activation_distance=0.01,
            world_coll_checker=CuroboMotion.world_coll_checker,
        )

        self.tensor_args = tensor_args
        self.motion_gen = MotionGen(motion_gen_config)
        if CuroboMotion.world_coll_checker is None:
            CuroboMotion.world_coll_checker = self.motion_gen.world_coll_checker
        self.motion_gen.warmup(parallel_finetune=True, batch=CUROBO_BATCH_SIZE)
        self.world_model = self.motion_gen.world_collision
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
            need_graph_success=True,
            enable_graph_attempt=5,
            max_attempts=40,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
            time_dilation_factor=1.0,
            ik_fail_return=5,
            success_ratio=0.5,
        )

        # TODO
        self.target = XFormPrim(
            "/World/target",
            position=np.array([0.5, 0, 0.5]),
            orientation=np.array([0, 1, 0, 0]),
        )

        self.cmd_plan = None
        self.cmd_plans = []
        self.cmd_idx = 0
        self.num_targets = 0
        self.past_cmd = None
        self.pose_metic = None
        self.robot = robot
        self.robot._articulation_view.initialize()
        self.idx_list = [self.robot.get_dof_index(x) for x in j_names]
        self.robot.set_joint_positions(default_config, self.idx_list)
        self.robot._articulation_view.set_max_efforts(
            values=np.array([5000 for i in range(len(self.idx_list))]),
            joint_indices=self.idx_list,
        )
        self.reached = False
        self.success = False
        self.saved_poses = []

        # TODO
        self.my_world = World(stage_units_in_meters=1.0)
        stage = self.my_world.stage
        self.usd_help.load_stage(stage)
        self.time_index = 0
        self.spheres = None
        self.obstacle_spheres = None

        self.attached_objects = []
        self._extract_cached_obstacles(need_reset_cache=CuroboMotion.cached_obstacle_info == {})

        self.set_obstacles()
        self.link_names = self.motion_gen.kinematics.link_names
        self.ee_link_name = self.motion_gen.kinematics.ee_link
        kin_state = self.motion_gen.kinematics.get_state(self.motion_gen.get_retract_config().view(1, -1))
        link_retract_pose = kin_state.link_pose
        self.target_links = {}
        for i in self.link_names:
            k_pose = np.ravel(link_retract_pose[i].to_list())
            self.target_links[i] = XFormPrim(
                "/World/target_" + i,
                position=np.array(k_pose[:3]),
                orientation=np.array(k_pose[3:]),
            )
        self.lock_joint_states = None


    def set_obstacles(self):
        start_time = time.time()

        # Use cached geometry to quickly update poses
        updated_obstacles, has_update = self._update_obstacle_poses_fast()
        if has_update:
            # Create WorldConfig
            world_config = WorldConfig(**updated_obstacles)

            # Convert to collision detection world
            obstacle = world_config.get_collision_check_world()
            self.world_cfg = obstacle  # NOTE this world_config only affects visualization
            self.motion_gen.world_coll_checker.load_collision_model(
                obstacle, fix_cache_reference=self.motion_gen.use_cuda_graph
            )

        self.motion_gen.graph_planner.reset_buffer()

        elapsed_time = time.time() - start_time
        carb.log_warn(f"Fast obstacle update completed, time: {elapsed_time:.4f}s, needs update: {has_update}")


    def _update_obstacle_poses_fast(self):
        r_T_w = None
        if self.robot_reference_prim and self.robot_transform_cache:
            self.robot_transform_cache.Clear()
            self.robot_transform_cache.SetTime(0)
            r_T_w, _ = get_prim_world_pose(self.robot_transform_cache, self.robot_reference_prim, inverse=True)

        # Prepare updated obstacle list
        updated_obstacles = {
            "cuboid": [],
            "sphere": [],
            "mesh": [],
            "cylinder": [],
            "capsule": [],
        }

        has_update = False
        # Iterate through cached obstacle information, update each pose
        for prim_path, obstacle_info in CuroboMotion.cached_obstacle_info.items():
            try:
                is_attached = False
                for attached_path in self.attached_objects:
                    if prim_path.startswith(attached_path):
                        is_attached = True
                        break
                if is_attached:
                    continue
                prim = self.usd_help.stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                current_mat, _ = get_prim_world_pose(self.robot_transform_cache, prim)

                if r_T_w is not None:
                    current_mat = r_T_w @ current_mat

                tensor_mat = torch.as_tensor(current_mat, device=torch.device("cuda", 0))
                updated_pose = Pose.from_matrix(tensor_mat).tolist()

                object_position = np.array(updated_pose[:3])
                distance_to_robot = np.linalg.norm(object_position)

                if distance_to_robot > 3.0:
                    continue

                geometry = obstacle_info["geometry"]
                if not has_update and geometry.pose != updated_pose:
                    has_update = True
                geometry.pose = updated_pose

                obstacle_type = obstacle_info["type"]
                updated_obstacles[obstacle_type].append(geometry)

            except Exception as e:
                carb.log_warn(f"Error updating obstacle {prim_path} pose: {e}")
                continue

        return updated_obstacles, True
    

    def visualize_spheres(
        self,
        sph_list,
        spheres_buffer,
        prim_prefix="/curobo/robot_sphere_",
        color=np.array([0, 0.8, 0.2]),
    ):
        robot_prim_path = self.robot.prim_path
        if spheres_buffer is None:
            spheres_buffer = []
            for si, s in enumerate(sph_list[0]):
                sp = sphere.VisualSphere(
                    prim_path=robot_prim_path + prim_prefix + str(si),
                    radius=float(s.radius),
                    color=color,
                )
                sp.set_local_pose(translation=np.array([s.position[0], s.position[1], s.position[2]]))
                spheres_buffer.append(sp)
        else:
            if len(spheres_buffer) < len(sph_list[0]):
                for si in range(len(spheres_buffer), len(sph_list[0])):
                    sp = sphere.VisualSphere(
                        prim_path=robot_prim_path + prim_prefix + str(si),
                        radius=0.01,
                        color=color,
                    )
                    spheres_buffer.append(sp)
            for si, s in enumerate(sph_list[0]):
                if not np.isnan(s.position[0]):
                    spheres_buffer[si].set_local_pose(
                        translation=np.array([s.position[0], s.position[1], s.position[2]])
                    )
                    spheres_buffer[si].set_radius(float(s.radius))
    

    def visualize_obstacles(self):
        sph_list = []
        for obs in self.world_cfg.objects:
            sph = obs.get_bounding_spheres(
                300,
                surface_sphere_radius=0.01,
                pre_transform_pose=None,
                tensor_args=self.tensor_args,
            )
            sph_list += sph
        self.visualize_spheres(
            [sph_list],
            self.obstacle_spheres,
            prim_prefix="/curobo/obstacle_sphere_",
            color=np.array([1.0, 0.25, 0.25])
        )


    def visualize_robot_spheres(self):
        sim_js = self.robot.get_joints_state()
        sim_js_names = self.robot.dof_names
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities),
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js.acceleration *= 0.0
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        sph_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
        self.visualize_spheres(sph_list, self.spheres, prim_prefix="/curobo/robot_sphere_")


    def kinematic_forward(self, joint_states, output_link_names=None):
        t1 = time.time()
        result = self._get_curobo_kinematics().compute_kinematics(joint_states)
        links_position = result.links_position
        links_quaternion = result.links_quaternion
        link_names = result.link_names
        output = []
        for i in range(links_position.shape[0]):
            tmp_output = {}
            for j, link_name in enumerate(link_names):
                if output_link_names is None or not len(output_link_names) or link_name in output_link_names:
                    tmp_output[link_name] = [
                        links_position[i][j].cpu().numpy().tolist(),
                        links_quaternion[i][j].cpu().numpy().tolist(),
                    ]
            output.append(tmp_output)
        t2 = time.time()
        print(f"kinematic forward time{t2 - t1}")
        return output
    

    def solve_batch_ik(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        active_ee_name: str,
        output_link_pose=False,
    ):
        print("len positions is {}".format(len(positions)))
        t1 = time.time()
        results = []
        link_poses = None
        batch_size = CUROBO_BATCH_SIZE
        pos_num = positions.shape[0]
        num_splits = (pos_num + batch_size - 1) // batch_size
        remainder = pos_num % batch_size
        if remainder != 0:
            padding = batch_size - remainder
            positions = np.concatenate([positions, np.tile(positions[-1:], (padding, 1))], axis=0)
            rotations = np.concatenate([rotations, np.tile(rotations[-1:], (padding, 1))], axis=0)
        position_batched = np.array_split(positions, num_splits)
        rotation_batched = np.array_split(rotations, num_splits)
        if len(self.link_names) > 1:
            link_poses = {}
            # TODO  target links
            for i in self.target_links.keys():
                c_p, c_rot = self.target_links[i].get_world_pose()
                link_poses[i] = Pose(
                    position=self.tensor_args.to_device(np.tile(c_p, (batch_size, 1))),
                    quaternion=self.tensor_args.to_device(np.tile(c_rot, (batch_size, 1))),
                    batch=batch_size,
                )
        # TODO: 为什么这里是goal，上述是state的感觉
        ee_c_p, ee_c_rot = self.target_links[self.ee_link_name].get_world_pose()
        goal_pose = Pose(
            position=self.tensor_args.to_device(np.tile(ee_c_p, (batch_size, 1))),
            quaternion=self.tensor_args.to_device(np.tile(ee_c_rot, (batch_size, 1))),
            batch=batch_size,
        )
        if output_link_pose:
            self.update_curobo_kinematics_lock_joints(self.lock_joint_states)
        
        for i in range(num_splits):
            pos_batch = position_batched[i]
            rot_batch = rotation_batched[i]

            if link_poses and active_ee_name in link_poses:
                link_poses[active_ee_name] = Pose(
                    position=self.tensor_args.to_device(pos_batch),
                    quaternion=self.tensor_args.to_device(rot_batch),
                    batch=batch_size,
                )
            if active_ee_name == self.ee_link_name:
                goal_pose = Pose(
                    position=self.tensor_args.to_device(pos_batch),
                    quaternion=self.tensor_args.to_device(rot_batch),
                    batch=batch_size,
                )
            t00 = time.time()
            result = self.motion_gen.ik_solver.solve_batch(goal_pose, link_poses=link_poses)
            t11 = time.time()

            print("ik batch {} time is {}".format(i, t11 - t00))
            if output_link_pose:
                js = result.js_solution.get_ordered_joint_state(
                    self._get_curobo_kinematics().kinematics_config.joint_names
                )
                js = js.squeeze(1)
                ik_link_poses = self.kinematic_forward(js)
            for k in range(result.success.shape[0]):
                joint_positions = {}
                for j, name in enumerate(result.js_solution.joint_names):
                    joint_positions[name] = result.js_solution.position[k][0].cpu().tolist()[j]
                if output_link_pose:
                    results.append([result.success[k], joint_positions, ik_link_poses[k]])
                else:
                    results.append((result.success[k], joint_positions))
        t2 = time.time()
        print("ik time is {}".format(t2 - t1))
        return results[:pos_num]


    def update_lock_joints(self, locked_joints):
        if (
            self.lock_joint_states is None
            or np.abs(np.array(list(self.lock_joint_states.values())) - np.array(list(locked_joints.values()))).max() > 1e-3
        ):
            before = time.time()
            self.motion_gen.update_locked_joints(locked_joints, self.robot_cfg)
            self.lock_joint_states = locked_joints
            after = time.time()
            carb.log_warn("update lock joints time is {}".format(after - before))
        else:
            print("lock joints is the same, no need to update")
    

    def update_curobo_kinematics_lock_joints(self, locked_joints):
        before = time.time()
        kinematics = self._get_curobo_kinematics()
        if kinematics is not None and locked_joints is not None:
            if self.curobo_kinematics_robot_cfg["kinematics"]["lock_joints"] != locked_joints:
                print("update kinematics lock joints")
                self.curobo_kinematics_robot_cfg["kinematics"]["lock_joints"] = locked_joints
                robot_cfg = RobotConfig.from_dict(self.curobo_kinematics_robot_cfg, self.tensor_args)
                kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
        after = time.time()
        carb.log_warn("update curobo kinematics lock joints time is {}".format(after - before))


    def view_debug_world(self):
        if self.debug:
            self.visualize_robot_spheres()
            self.visualize_obstacles()

    
    def calculate_ik_goal(
        self,
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=None,
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links: list[str] = [],
        from_current_pose=False,
    ):
        # NOTE: Only load the collision model once — set_obstacles() calls
        # graph_planner.reset_buffer() and load_collision_model() which are
        # very expensive GPU operations. Calling them on every IK retry (when
        # success=False triggers re-entry every frame) stalls Isaac Sim.
        if not getattr(self, '_obstacles_initialized', False):
            t_so0 = time.time()
            self.set_obstacles()
            self._obstacles_initialized = True
            carb.log_warn(f"set obstacles cost time {time.time() - t_so0} s")

        t0 = time.time()
        self.reached = False
        if from_current_pose and not goal_offset:
            self.reached = True
            self.success = False
            print("from_current_pose is True, but goal_offset is None, return")
            return
        # TODO
        cube_position, cube_orientation = self.target.get_world_pose()
        if self.past_pose is None:
            self.past_pose = cube_position
        if self.target_pose is None:
            self.target_pose = cube_position
        if self.target_orientation is None:
            self.target_orientation = cube_orientation
        if self.past_orientation is None:
            self.past_orientation = cube_orientation
        
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        sim_js_names = []
        lock_idx = []
        sim_js_positions = []
        sim_js_velocities = []
        for idx, name in enumerate(js_names):
            if name not in self.lock_js_names:
                sim_js_names.append(name)
            else:
                lock_idx.append(idx)
        for idx, position in enumerate(sim_js.positions):
            if idx not in lock_idx:
                sim_js_positions.append(position)
        for idx, velocity in enumerate(sim_js.velocities):
            if idx not in lock_idx:
                sim_js_velocities.append(velocity)
        sim_js_positions = np.array(sim_js_positions)[np.newaxis, :]
        sim_js_velocities = np.array(sim_js_velocities)[np.newaxis, :]
        cu_js = JointState(
            position=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))),
            velocity=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            acceleration=self.tensor_args.to_device(np.tile(sim_js_velocities, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            jerk=self.tensor_args.to_device(np.tile(sim_js_velocities, (CUROBO_BATCH_SIZE, 1))) ** 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        start_time = time.time()
        if from_current_pose:
            start_pose = self.motion_gen.compute_kinematics(cu_js).ee_pose.clone()
            cube_position = start_pose.position.squeeze()[0].cpu().numpy()
            cube_orientation = start_pose.quaternion.squeeze()[0].cpu().numpy()
        
        ee_translation_goal = cube_position
        ee_orientation_teleop_goal = cube_orientation
        if goal_offset is not None:
            offset = Pose.from_list(goal_offset)
            goal_pose_list = np.concatenate([ee_translation_goal, ee_orientation_teleop_goal]).tolist()
            goal_pose = Pose.from_list(goal_pose_list)
            # TODO 这里的if else看起来是有点问题的
            if offset_and_constraint_in_goal_frame:
                offset_goal_pose = goal_pose.clone().multiply(offset)
            else:
                offset_goal_pose = offset.clone().multiply(goal_pose.clone())
            ee_translation_goal = offset_goal_pose.position.squeeze().cpu().numpy()
            ee_orientation_teleop_goal = offset_goal_pose.quaternion.squeeze().cpu().numpy()
        ik_goal = Pose(
            position=self.tensor_args.to_device(np.tile(ee_translation_goal, (CUROBO_BATCH_SIZE, 1))),
            quaternion=self.tensor_args.to_device(np.tile(ee_orientation_teleop_goal, (CUROBO_BATCH_SIZE, 1))),
            batch=CUROBO_BATCH_SIZE,
        )

        if path_constraint is not None and len(path_constraint) == 6:
            hold_pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args.to_device(path_constraint),
                project_to_goal_frame=offset_and_constraint_in_goal_frame,
            )
            self.plan_config.pose_cost_metric = hold_pose_cost_metric
        else:
            carb.log_warn("no valid path constraint provided")
            self.plan_config.pose_cost_metric = self.pose_metic
        link_poses = None
        if self.plan_config.pose_cost_metric:
            update_res = self.motion_gen.update_pose_cost_metric(self.plan_config.pose_cost_metric, cu_js, ik_goal)
            if not update_res:
                self.reached = True
                self.success = False
                carb.log_warn("update pose cost metric failed")
                return
        disable_collision_links = list(
            filter(
                lambda link: any(re.match(pattern, link) for pattern in disable_collision_links),
                self.collision_link_names,
            )
        )

        # TODO
        self.motion_gen.toggle_link_collision(disable_collision_links, False)
        try:
            result = self.motion_gen.plan_batch(
                cu_js,
                ik_goal,
                self.plan_config.clone(),
                link_poses=link_poses,
            )
            if result.success.any():
                self.reached = False
                self.success = True
                print("end_time is{}".format(time.time() - start_time))
                self.num_targets += 1
                paths = result.get_successful_paths()
                position_filter_res = filter_paths_by_position_error(paths, result.position_error[result.success])
                rotation_filter_res = filter_paths_by_rotation_error(paths, result.rotation_error[result.success])
                filtered_paths = []
                for i in range(len(paths)):
                    if position_filter_res[i] and rotation_filter_res[i]:
                        filtered_paths.append(paths[i])
                if len(filtered_paths) == 0:
                    filtered_paths = paths
                dof_weights = [
                    1.0,1.0,1.0,1.0,3.0,3.0,1.0,
                    # 1.0,1.0,1.0,1.0,3.0,3.0,1.0
                ]
                sorted_indices = sort_by_difference_js(
                    filtered_paths,
                    weights=self.tensor_args.to_device(dof_weights),
                )
                self.cmd_plan = paths[sorted_indices[0]]
                self.cmd_plan = self.motion_gen.get_full_js(self.cmd_plan)
                print(len(self.cmd_plan))
                self.idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in self.cmd_plan.joint_names:
                        self.idx_list.append(self.robot.get_dof_index(x))
                        common_js_names.append(x)
                self.cmd_plan = self.cmd_plan.get_ordered_joint_state(common_js_names)
                self.cmd_idx = 0
                carb.log_warn("plan success")
            else:
                self.reached = True
                self.success = False
                carb.log_warn("plan did not converge to a solution: {}".format(str(result.status)))
        except Exception as e:
            self.reached = True
            self.success = False
            carb.log_warn("plan got an exception: {}".format(str(e)))

        self.motion_gen.toggle_link_collision(disable_collision_links, True)
        self.target_pose = cube_position
        self.target_orientation = cube_orientation
        self.past_pose = cube_position
        self.past_orientation = cube_orientation
        t1 = time.time()
        print("total time is {}".format(t1 - t0))


    def exclude_js(self, joint_names):
        if self.cmd_plan:
            positions = []
            velocities = []
            self.idx_list = []
            for name in joint_names:
                self.idx_list.append(self.robot.get_dof_index(name))
            for index, pos in enumerate(self.cmd_plan.position):
                position = []
                velocity = []
                for name in joint_names:
                    idx = self.cmd_plan.joint_names.index(name)
                    position.append(pos.cpu().numpy()[idx])
                    velocity.append(self.cmd_plan.velocity.cpu().numpy()[index][idx])
                positions.append(position)
                velocities.append(velocity)
            self.cmd_plan = JointState(
                position=self.tensor_args.to_device(positions),
                velocity=self.tensor_args.to_device(velocities) * 0.0,
                acceleration=self.tensor_args.to_device(velocities) * 0.0,
                jerk=self.tensor_args.to_device(velocities) * 0.0,
                joint_names=joint_names,
            )

    def detach_obj(self):
        self.motion_gen.detach_object_from_robot("left_attached_object")
        self.motion_gen.detach_object_from_robot()
        self.attached_objects.clear()
        self.set_obstacles()

    def remove_objects_from_world(self, prim_paths):
        for x in prim_paths:
            obs = self.motion_gen.world_model.get_obstacle(x)
            if not obs:
                continue
            self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=x)
            self.motion_gen.world_model.remove_obstacle(x)

    def attach_obj(
        self,
        prim_paths,
        link_name = "attached_object",
        ee_position = [0, 0, 0],
        ee_rotation = [1, 0, 0, 0],
    ):
        self.motion_gen.detach_object_from_robot()
        attach_result = False
        self.set_obstacles()
        carb.log_warn(f"attach object_names={prim_paths}")
        ee_pose = Pose(
            position=self.tensor_args.to_device(ee_position),
            quaternion=self.tensor_args.to_device(ee_rotation),
        )
        attach_result = self.attach_objects_to_robot(
            object_names=prim_paths,
            link_name=link_name,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            surface_sphere_radius=0.005,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.005, 1, 0, 0, 0], self.tensor_args),
            remove_obstacles_from_world_config=True,
            ee_pose=ee_pose,
        )
        carb.log_warn(f"attach result = {attach_result}")

        return attach_result
    
    def attach_objects_to_robot(
        self,
        object_names,
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        world_objects_pose_offset=None,
        remove_obstacles_from_world_config: bool=False,
        ee_pose=None,
    ) -> bool:
        if world_objects_pose_offset is not None:
            ee_pose = world_objects_pose_offset.inverse().multiply(ee_pose)
        ee_pose = ee_pose.inverse()
        max_spheres = self.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(link_name)
        if len(object_names) == 0:
            return
        n_spheres = int(max_spheres / len(object_names))
        sphere_tensor = torch.zeros((max_spheres, 4))
        sphere_tensor[:, 3] = -10.0
        sph_list = []
        if n_spheres == 0:
            return False
        for i, object_name in enumerate(object_names):
            obs = self.motion_gen.world_model.get_obstacle(object_name)
            if not obs:
                continue
            sph = obs.get_bounding_spheres(
                n_spheres,
                surface_sphere_radius,
                pre_transform_pose=ee_pose,
                tensor_args=self.tensor_args,
                fit_type=sphere_fit_type,
                voxelize_method=voxelize_method,
            )
            sph_list += [s.position + [s.radius] for s in sph]
            self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=object_name)
            if remove_obstacles_from_world_config:
                self.motion_gen.world_model.remove_obstacle(object_name)
                if object_name not in self.attached_objects:
                    self.attached_objects.append(object_name)
        spheres = self.tensor_args.to_device(torch.as_tensor(sph_list))
        if not spheres.shape[0]:
            carb.log_warn("No spheres found for the given objects.")
            return False
        
        if spheres.shape[0] > max_spheres:
            spheres = spheres[: spheres.shape[0]]
        sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

        self.motion_gen.attach_spheres_to_robot(sphere_tensor=sphere_tensor, link_name=link_name)

        return True
    
    def get_articulation_action_without_lock_joints(self, cmd_state):
        tmp_idx_list = []
        for i in range(len(self.idx_list)):
            if cmd_state.joint_names[i] not in self.robot_cfg["kinematics"]["lock_joints"]:
                tmp_idx_list.append(i)
        
        art_action = ArticulationAction(
            cmd_state.position.cpu().numpy()[tmp_idx_list],
            cmd_state.velocity.cpu().numpy()[tmp_idx_list],
            joint_indices = np.array(self.idx_list)[tmp_idx_list],
        )
        return art_action
    
    def on_physics_step(self, run_ratio=1.0, additional_action: ArticulationAction = None):
        self.time_index += 1
        if run_ratio <= 0.0 or run_ratio > 1.0:
            carb.log_warn("run_ratio should be in the range (0, 1], setting to 1.0")
            run_ratio = 1.0
        
        if self.cmd_plan is not None:
            cmd_state = self.cmd_plan[self.cmd_idx]
            self.past_cmd = cmd_state.clone()
            art_action = self.get_articulation_action_without_lock_joints(cmd_state)
            if additional_action is not None and additional_action.joint_positions is not None:
                for idx in range(len(additional_action.joint_positions)):
                    if additional_action.joint_positions[idx] is None:
                        continue
                    if idx in art_action.joint_indices:
                        art_idx = art_action.joint_indices.index(idx)
                        art_action.joint_positions[art_idx] = additional_action.joint_positions[idx]
                        art_action.joint_velocities[art_idx] = additional_action.joint_velocities[idx]
                    else:
                        art_action.joint_indices = np.append(art_action.joint_indices, idx)
                        art_action.joint_positions = np.append(
                            art_action.joint_positions,
                            additional_action.joint_positions[idx],
                        )
                        art_action.joint_velocities = np.append(
                            art_action.joint_velocities,
                            additional_action.joint_velocities[idx],
                        )
            self.robot.apply_action(art_action)
            self.cmd_idx += 1
            if self.cmd_idx >= len(self.cmd_plan.position) * run_ratio:
                self.cmd_idx = 0
                self.cmd_plan = None
                self.past_cmd = None
                self.reached = True
                print(f"Reached {self.reached}")


class UIBuilder:
    def __init__(self, world: World, debug=False):
        self.debug = debug
        self._cs = _sensor.acquire_contact_sensor_interface()
        self._is = _sensor.acquire_imu_sensor_interface()
        self.frames = []
        self.wrapped_ui_elements = []
        self.collision_paths = []
        self._Joint_Info_Sliders = []
        self._Sensor_parent = None
        self._Camera_parent = None
        self.camera: Camera = None
        self.articulation = None
        self.articulation_rmpflow = None
        self.right_articulation_rmpflow = None
        self._taskspace_trajectory_generator = None
        self._target = None
        self._right_target = None
        self._currentCamera = ""
        self._followingPos = np.array([0, 0, 0])
        self._followingOrientation = np.array([1, 0, 0, 0])
        self.my_world: World = world
        self.currentImg = None
        self.currentCamInfo = None
        self.curoboMotion: Dict[str, CuroboMotion] = {}
        self.current_curobo_motion: Optional[CuroboMotion] = None
        self.camera_prim_list = []
        self.camera_list = []
        self.rmp_move = False
        self.cmd_list = None
        self.reached = False
        self.cameras = []
        self.art_controllers = []

    def initialize_articulation(self, batch_num=0):
        scene = self.my_world.scene
        if scene._scene_registry.name_exists(self.robot_name):
            self.articulation = scene.get_object(self.robot_name)
        else:
            self.articulation = Articulation(prim_path=self.robot_prim_path, name=self.robot_name)
            scene.add(self.articulation)
        self.articulation.initialize()
        self.ruckig_controller = RuckigController(self.dof_nums, self.joint_delta_time)
        robot_list = []
        for idx in range(batch_num):
            articulation = Articulation(
                prim_path=self.robot_prim_path + "_{}".format(idx),
                name=self.robot_name + "_{}".format(idx),
            )
            articulation.initialize()
            robot_list.append(articulation)
            self.art_controllers = [r.get_articulation_controller() for r in robot_list]
        if self.enable_curobo:
            for key, cfg in self.curobo_config_file.items():
                if not self.curoboMotion.get(key):
                    before = time.time()
                    curobo_motion = self._init_curobo(cfg, key)
                    after = time.time()
                    print(f"init curobo motion for {key} takes {after - before}")
                    self.curoboMotion[key] = curobo_motion
                else:
                    if self.arm_type == "dual":
                        for effector_key, value in self.end_effector_name.items():
                            if effector_key == "left":
                                current_position, rotation_matrix = self._get_ee_pose(False, is_local=True)
                            else:
                                current_position, rotation_matrix = self._get_ee_pose(True, is_local=True)
                            self.curoboMotion.get(key).init_ee_pose[value] = {
                                "position": current_position,
                                "orientation": transform_utils.mat2quat_wxyz(rotation_matrix),
                            }
                        self.curoboMotion.get(key).reset_link()
                    self.curoboMotion.get(key).reset()

    def _init_solver(self, robot, enable_curobo, batch_num):
        self.enable_curobo = enable_curobo
        self.robot_name = robot.robot_name
        self.robot_prim_path = robot.robot_prim_path
        self.dof_nums = robot.dof_nums
        self.lock_joints = robot.lock_joints
        self.joint_delta_time = robot.joint_delta_time
        self.curobo_config_file = robot.curobo_config_file
        self.cameras = robot.cameras
        self.init_joint_position = robot.init_joint_position
        self.end_effector_prim_path = robot.end_effector_prim_path
        self.initialize_articulation(batch_num)
        self.arm_type = robot.arm_type
        self.end_effector_name = robot.end_effector_name
        self.active_arm_joints = robot.active_arm_joints
    
    def _init_kinematic_solver(self, robot):
        if robot.arm_type == "dual":
            self.kinematics_solver = {
                "left": KinematicsSolver(
                    robot_description_path=robot.robot_description_path["left"],
                    urdf_path=robot.urdf_name,
                    end_effector_name=robot.end_effector_name["left"],
                    articulation=self.articulation,
                ),
                "right": KinematicsSolver(
                    robot_description_path=robot.robot_description_path["right"],
                    urdf_path=robot.urdf_name,
                    end_effector_name=robot.end_effector_name["right"],
                    articulation=self.articulation,
                ),
            }
        else:
            self.kinematics_solver = KinematicsSolver(
                robot_description_path=robot.robot_description_path,
                urdf_path=robot.urdf_name,
                end_effector_name=robot.end_effector_name,
                articulation=self.articulation,
            )

    def on_physics_step(self, step):
        for curoboMotion in self.curoboMotion.values():
            curoboMotion.on_physics_step()

    def _on_capture_cam(self, isRGB, isDepth, isSemantic):
        if self._currentCamera:
            resolution = [640, 480]
            if self._currentCamera in self.cameras:
                resolution = self.cameras[self._currentCamera]
            _Camera = Camera(prim_path=self._currentCamera, resolution=resolution)
            _Camera.initialize()
            self.camera_list.append(_Camera)
            self.camera_prim_list.append(self._currentCamera)
            focal_length = _Camera.get_focal_length()
            horizontal_aperture = _Camera.get_horizontal_aperture()
            vertical_aperture = _Camera.get_vertical_aperture()
            width, height = _Camera.get_resolution()
            fx = width * focal_length / horizontal_aperture
            fy = height * focal_length / vertical_aperture
            ppx = width * 0.5
            ppy = height * 0.5
            self.currentCamInfo = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "ppx": ppx,
                "ppy": ppy,
            }
            self.currentImg = {}
            self.currentImg["camera_info"] = self.currentCamInfo
            self.currentImg["rgb"] = []

    def _Generate_following_position(self, isRight=True):
        curoboMotion = self.get_curobo_motion(isRight)
        if curoboMotion:
            curoboMotion.target = XFormPrim(
                "/World/target",
                position=self._followingPos,
                orientation=self._followingOrientation,
            )
            target_world = XFormPrim(
                f"{self.robot_prim_path}/target",
            )
            target_world.set_local_pose(translation=self._followingPos, orientation=self._followingOrientation)
            if self.arm_type == "dual":
                key = self.end_effector_name["left"]
                if isRight:
                    key = self.end_effector_name["right"]
                    curoboMotion.target_links[key] = XFormPrim(
                        "/World/target_" + key,
                        position=np.array(self._followingPos),
                        orientation=np.array(self._followingOrientation),
                    )
            self._target = None

    def _follow_target(
        self,
        isRight=True,
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=None,
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links=[],
        from_current_pose=False,
    ):
        self._Generate_following_position(isRight)
        self.set_locked_joint_positions(isRight)
        curoboMotion = self.get_curobo_motion(isRight)
        curoboMotion.calculate_ik_goal(
            goal_offset=goal_offset,
            path_constraint=path_constraint,
            offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
            disable_collision_links=disable_collision_links,
            from_current_pose=from_current_pose,
        )
        if self.arm_type == "dual":
            js_names = self.active_arm_joints["left"]
            if isRight:
                js_names = self.active_arm_joints["right"]
            curoboMotion.exclude_js(js_names)

    def _get_ee_pose(self, is_right, is_local=False):
        robot_base_translation, robot_base_orientation = self.articulation.get_world_pose()
        if is_local:
            robot_base_translation = np.array([0.0, 0.0, 0.0])
            robot_base_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        if self.arm_type == "dual":
            key = "left"
            if is_right:
                key = "right"
            self.kinematics_solver[key]._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )
            return self.kinematics_solver[key]._articulation_kinematics_solver.compute_end_effector_pose()
        self.kinematics_solver._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        return self.kinematics_solver._articulation_kinematics_solver.compute_end_effector_pose()

    def _get_ik_status(self, target_position, target_orientation, isRight):
        robot_base_translation, robot_base_orientation = self.articulation.get_world_pose()
        if self.arm_type == "dual":
            key = "left"
            if isRight:
                key = "right"
            self.kinematics_solver[key]._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )
            actions, success = self.kinematics_solver[key]._articulation_kinematics_solver.compute_inverse_kinematics(
                target_position, target_orientation
            )
        else:
            self.kinematics_solver._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )
            actions, success = self.kinematics_solver._articulation_kinematics_solver.compute_inverse_kinematics(
                target_position, target_orientation
            )
        return success, actions

    def _limit_joint_positions(self, positions, joint_indices=None):
        if joint_indices is None:
            joint_indices = np.arange(len(positions))
        lowers = self.articulation.dof_properties["lower"][joint_indices]
        uppers = self.articulation.dof_properties["upper"][joint_indices]
        positions = np.clip(positions, lowers, uppers)
        return positions

    def _safe_set_joint_positions(self, positions, joint_indices=None):
        positions = self._limit_joint_positions(positions, joint_indices)
        self.articulation.set_joint_positions(positions, joint_indices=joint_indices)

    def _move_to(self, target_positions, joint_indices=None, is_trajectory=False, is_action=False):
        if not self.articulation:
            return
        self._currentLeftTask = 10
        self._currentRightTask = 10
        actions = ArticulationAction(joint_positions=target_positions)
        if not is_trajectory:
            if joint_indices is None:
                joint_indices = []
                positions = []
                for idx, joint_position in enumerate(target_positions):
                    if joint_position is not None:
                        positions.append(joint_position)
                        joint_indices.append(idx)
                self._safe_set_joint_positions(positions, joint_indices=joint_indices)
            else:
                self._safe_set_joint_positions(target_positions, joint_indices=joint_indices)
        else:
            self.reached = True
            self.articulation.apply_action(actions)

    def _trajectory_list_follow_target(
        self,
        target_position,
        target_orientation,
        is_right,
        ee_interpolation=False,
        distance_frame=0.01,
    ):
        current_position, rotation_matrix = self._get_ee_pose(is_right)
        XFormPrim("/ruckig", position=target_position, orientation=target_orientation)
        current_rotation = transform_utils.mat2quat_wxyz(rotation_matrix)
        if not self._get_ik_status(target_position, target_orientation, is_right)[0]:
            self.cmd_list = None
            self.reached = True
            print("IK not success")
            return
        target_arm_positions = self._get_ik_status(target_position, target_orientation, is_right)[1].joint_positions
        self.idx_list = self._get_ik_status(target_position, target_orientation, is_right)[1].joint_indices
        current_arm_positions = self.articulation.get_joint_positions(joint_indices=self.idx_list)

        def lerp(start, end, t):
            return start + t * (end - start)

        def slerp(q0, q1, t):
            dot = np.dot(q0, q1)
            if dot < 0.0:
                q1 = -q1
                dot = -dot

            dot = min(max(dot, -1.0), 1.0)
            theta_0 = np.arccos(dot)
            theta = theta_0 * t
            sin_theta = np.sin(theta)
            sin_theta_0 = np.sin(theta_0)
            if sin_theta == 0:
                return q0
            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0

            return s0 * q0 + s1 * q1

        self.cmd_list = []
        if ee_interpolation:
            distance = np.linalg.norm(target_position - current_position)
            joint_distance = 0
            step = (int)(distance / distance_frame)
            if step > 1:
                for i in range(step):
                    t = i / (step - 1)
                    position = lerp(current_position, target_position, t)
                    rotation = slerp(current_rotation, target_orientation, t)
                    issuccess, arm_position = self._get_ik_status(position, rotation, is_right)
                    joint_distance = np.linalg.norm(arm_position.joint_positions - current_arm_positions)
                    if joint_distance < 1:
                        self.cmd_list.append(arm_position.joint_positions)
                        current_arm_positions = arm_position.joint_positions
        else:
            cmd_list = self.ruckig_controller.caculate_trajectory(current_arm_positions, target_arm_positions)
            distance = np.linalg.norm(target_position - current_position)
            for position in cmd_list:
                joint_distance = np.linalg.norm(np.array(current_arm_positions) - np.array(position))
                current_arm_positions = position
                self.cmd_list.append(position)
        self.cmd_idx = 0
        self.reached = False
        self.time_index = 0
        if not self.cmd_list:
            self.reached = True

    def _on_every_frame_trajectory_list(self):
        if self.cmd_list and self.articulation:
            self.time_index += 1
            cmd_state = self.cmd_list[self.cmd_idx]
            art_action = ArticulationAction(joint_positions=cmd_state, joint_indices=self.idx_list)
            for robot in self.art_controllers:
                robot.apply_action(art_action)
            self.articulation.apply_action(art_action)
            self.cmd_idx += 1
            if self.cmd_idx >= len(self.cmd_list):
                self.cmd_idx = 0
                self.cmd_list = None
                self.reached = True

    def _on_init(self):
        self.articulation = None

    def _init_curobo(self, curobo_config, name):
        if self.articulation:
            curoboMotion = CuroboMotion(
                name,
                self.articulation,
                self.my_world,
                curobo_config,
                self.robot_prim_path,
                self.art_controllers,
                step=32,
                debug=self.debug,
            )
            curoboMotion.set_obstacles()
            return curoboMotion
        return None

    def get_curobo_motion(self, is_right=True):
        return self.curoboMotion.get("right" if is_right else "left")

    def attach_objs(self, prim_path_list, is_right=True):
        result = False
        curoboMotion = self.get_curobo_motion(is_right)
        if curoboMotion:
            print("Attach!!!!")

            link_name = "attached_object"
            position, rotation_matrix = self._get_ee_pose(is_right, is_local=True)
            rotation = transform_utils.mat2quat_wxyz(rotation_matrix)
            if self.arm_type == "dual" and not is_right:
                link_name = "left_attached_object"
            result = curoboMotion.attach_obj(prim_path_list, link_name, position, rotation)
            curoboMotion.view_debug_world()
        return result

    def detach_objs(self):
        if self.articulation:
            for curoboMotion in self.curoboMotion.values():
                curoboMotion.detach_obj()
                print("Detach!!!!!")
                curoboMotion.view_debug_world()
    
    def view_debug_world(self):
        if self.articulation:
            for cm in self.curoboMotion.values():
                cm.view_debug_world()


    def remove_objects_from_world(self, prim_paths):
        if self.articulation:
            for curoboMotion in self.curoboMotion.values():
                curoboMotion.remove_objects_from_world(prim_paths)

    def set_locked_joint_positions(self, is_right=True):
        if not self.lock_joints:
            return
        articulation = self.articulation
        joint_positions = articulation.get_joint_positions()
        articulation.dof_names
        curoboMotion = self.get_curobo_motion(is_right)
        ids = {}
        for idx in range(len(articulation.dof_names)):
            name = articulation.dof_names[idx]
            if name in curoboMotion.lock_js_names:
                ids[name] = float(joint_positions[idx])
        curoboMotion.update_lock_joints(ids)
        try:
            curoboMotion.update_curobo_kinematics_lock_joints(ids)
        except Exception:
            pass  # non-fatal; kinematics path may not be active

    def _on_reset(self):
        async def _on_rest_async():
            await omni.kit.app.get_app().next_update_async()
            self.initialize_articulation()
            self.rmp_move = False

        asyncio.ensure_future(_on_rest_async())
        return

    def _find_all_objects_of_type(self, obj_type):
        items = []
        stage = omni.usd.get_context().get_stage()
        if stage:
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/")):
                path = str(prim.GetPath())
                type = get_prim_object_type(path)
                if type == obj_type:
                    items.append(path)
        return items

    def _initialize_object_articulations(self):
        articulations = self._find_all_objects_of_type("articulation")
        for art in articulations:
            articulation = Articulation(art)
            if articulation:
                articulation.initialize()

    def set_articulation_state(self, state: bool):
        articulations = self._find_all_objects_of_type("articulation")
        for art in articulations:
            _prim = get_prim_at_path(art)
            if art not in self.robot_prim_path:
                _prim.GetAttribute("physxArticulation:articulationEnabled").Set(state)

    def remove_objects(self):
        parent_prim = get_prim_at_path("/World/Objects")
        if parent_prim.IsValid():
            prims = get_prim_children(parent_prim)
            for prim in prims:
                delete_prim(prim.GetPath())

    def remove_graph(self, prim_paths):
        for prim in prim_paths:
            replicator_prim = get_prim_at_path(prim)
            if replicator_prim.IsValid():
                delete_prim(prim)


import queue
import threading
import signal
import subprocess
import json

from omni.kit.viewport.utility import get_active_viewport_and_window
from omni.kit.viewport.utility.camera_state import ViewportCameraState
import omni.replicator.core as rep

# from source.data_collection.server.ros_publisher.base import USDBase
from source.data_collection.server.controllers.parallel_gripper import ParallelGripper

class CommandController:
    def __init__(
        self, 
        ui_builder: UIBuilder,
        enable_physics=False,
        enable_curobo=False,
        publish_ros=False,
        rendering_step=60,
        debug=False,
    ):

        self.sim_assets_root = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets"
        self.ui_builder = ui_builder
        self.debug = debug
        self.data = None
        self.Command = 0
        self.data_to_send = None
        self.gripper_L = None
        self.gripper_R = None
        self.gripper_state_L = ""
        self.gripper_state_R = ""
        self.gripper_state = ""
        self.condition = threading.Condition()
        self.result_queue = queue.Queue()
        self.target_position = np.array([0, 0, 0])
        self.target_rotation = np.array([0, 0, 0])
        self.target_joints_pose = None
        self.task_name = None
        self.cameras = {}
        self.step = 0
        self.path_to_save = None
        self.exit = False
        self.object_asset_dict = {}
        self.usd_objects = {}
        self.articulat_objects = {}
        self.rigid_bodies = {}
        self.enable_physics = enable_physics
        self.enable_curobo = enable_curobo
        self.trajectory_list = None
        self.trajectory_index = 0
        self.trajectory_reached = False
        self.target_joints_pose = []
        self.graph_path = []
        self.camera_graph_path = []
        self.loop_count = 0
        self.publish_ros = publish_ros
        self.rendering_step = rendering_step
        self.process = []
        self.extract_process = []
        self.target_point = None
        self.debug_view = {}
        self.timeline = omni.timeline.get_timeline_interface()
        self.light_config = []
        self.attached_joints = {}
        self.write_semantic = False
        self.motion_run_ratio = 1.0
        self.gripper_action_timing = None
        self.object_code_dict = {}
        self.task_description = {
            "task_name": "",
            "english_task_name": "",
            "init_scene_text": "",
        }
        self.task_metric = {}
        # self.sensor_base = USDBase()
        self.ros_node_initialized = False
        self.ros_step = 0  # control pub hz
        self.scene_usd = ""
        self.playback_timerange = []
        self.ros_publishers = []
        self.dof_names = []
        self.playback_frames = {}
        self.playback_waited_frame_num = 0
        self.attach_states = {}
        self.camera_info_list = {}
        self.fps = 60
        self.cur_runtime_checker = None
        # Timing statistics related
        self.timing_stats = {}  # Store total time for each function {function_name: total_time}
        self.timing_lock = threading.Lock()  # For thread-safe timing statistics

        if debug:
            self.cube_target = cuboid.VisualCuboid(
                "/World/cube_target",
                position=np.array([2.55, 1.1, 1.0]),
                orientation=np.array([1, 0, 0, 0]),
                color=np.array([1.0, 0, 0]),
                size=0.05,
            )

        # NOTE: Vertical adjustment state.
        # pending_leg_adjustment: leg joint targets to apply on next LINEAR_MOVE command.
        # default_leg_joints: captured after robot init; used as the reference/max-height state.
        # default_torso_pose/default_torso_z: stable torso pose at init — used as the
        # reference for the workspace check so restore decisions are based on the
        # default-height configuration, not the currently lowered one.
        self.pending_leg_adjustment = None
        self.default_leg_joints = {}
        self.default_torso_pose = None
        self.default_torso_z = None
        # Leg trajectory state for gradual movement (Fix 2: no sudden joint jumps).
        self.leg_trajectory = None      # np.ndarray of shape (n_steps, 3)
        self.leg_traj_idx = 0
        self.leg_joint_names = ["leg_joint1", "leg_joint2", "leg_joint3"]
        self.leg_joint_indices = None   # cached DOF indices
        # Flag: True on the step the trajectory finishes; cleared after curobo is force-refreshed.
        self.leg_just_finished = False
        self._diag_step = 0  # counter for right-arm diagnostic prints
        self._right_arm_diag_names = [
            "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
            "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
        ]
        self._right_arm_diag_indices = None  # cached DOF indices (computed once on first use)


    def _init_robot_cfg(
        self,
        robot_cfg,
        scene_usd,
        is_mocap=False,
        batch_num=0,
        init_position=[0, 0, 0],
        init_rotation=[1, 0, 0, 0],
        stand_type="cylinder",
        size_x=0.1,
        size_y=0.1,
        init_joint_position=[],
        init_joint_names=[],
    ):
        self.scene_usd = scene_usd
        current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        robot_config_dir = os.path.join(current_directory, "source/data_collection/config/robot_cfg")
        robot = RobotCfg(os.path.join(robot_config_dir, robot_cfg))

        self.robot_usd_path = os.path.join(self.sim_assets_root, robot.robot_usd)
        self.scene_usd_path = os.path.join(self.sim_assets_root, scene_usd)
        get_prim_at_path("/World")
        self.batch_num = batch_num
        if "World" not in robot.robot_prim_path:
            add_reference_to_stage(self.robot_usd_path, robot.robot_prim_path)
        else:
            add_reference_to_stage(self.robot_usd_path, "/World")
        add_reference_to_stage(self.scene_usd_path, "/World")
        robot_prim = robot.robot_prim_path
        self.usd_objects["robot"] = XFormPrim(
            prim_path=robot_prim,
            position=init_position,
            orientation=init_rotation,
        )
        # overwrite init joint position
        if len(init_joint_position) != len(init_joint_names):
            raise ValueError("robot init joint position and names length not match")
        if init_position[2] > 0:
            cube_position = [
                init_position[0],
                init_position[1],
                init_position[2] / 2,
            ]
            cube_scale = [size_x, size_y, init_position[2]]
            if stand_type == "cylinder":
                cylinder.VisualCylinder(
                    prim_path="/base_cube",
                    position=cube_position,
                    orientation=(1, 0, 0, 0),
                    scale=cube_scale,
                    color=np.array([1, 1, 1]),
                )
            else:
                cuboid.VisualCuboid(
                    prim_path="/base_cube",
                    position=cube_position,
                    orientation=(1, 0, 0, 0),
                    scale=cube_scale,
                    color=np.array([1, 1, 1]),
                )
        self.robot_init_position = init_position
        self.robot_init_rotation = init_rotation
        if "multispace" in scene_usd:
            self.scene_name = scene_usd.split("/")[-3] + "/" + scene_usd.split("/")[-2]
        else:
            self.scene_name = scene_usd.split("/")[-2]
        for idx in range(batch_num):
            if "World" not in robot.robot_prim_path:
                prim_path = robot.robot_prim_path + "_{}".format(idx)
                add_reference_to_stage(self.robot_usd_path, prim_path)
                XFormPrim(prim_path=prim_path, position=[0, 2 * idx + 1, 0])
            else:
                add_reference_to_stage(self.robot_usd_path, "/World_{}".format(idx))
            add_reference_to_stage(self.scene_usd_path, "/World_{}".format(idx))
            XFormPrim(prim_path="/World_{}".format(idx), position=[0, 2 * idx + 1, 0])

        camera_state = ViewportCameraState("/OmniverseKit_Persp")
        camera_state.set_position_world(
            Gf.Vec3d(2.65, 2.4, 1.74),
            True,
        )
        camera_state.set_target_world(Gf.Vec3d(init_position[0]+0.5, init_position[1], init_position[2]+0.8), True)

        stage = omni.usd.get_context().get_stage()
        self.scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        self.scene.CreateGravityMagnitudeAttr().Set(9.81)
        robot_rep = rep.get.prims(path_pattern=robot.robot_prim_path, prim_types=["Xform"])
        viewport, window = get_active_viewport_and_window()
        # if "galbot" in robot.robot_name.lower():
        #     viewport.set_active_camera("/galbot_one_golf/head_link2/head_front_left_color")
        # elif "G2" in robot.robot_name:
        #     viewport.set_active_camera("/G2/head_link3/head_front_Camera")  
        with robot_rep:
            rep.modify.semantics([("class", "robot")])
        self.robot_cfg = robot
        self._play()

        # init kinematic solver
        articulation = self._initialize_articulation()
        self.dof_names = articulation.dof_names
        # overwrite fixed joints in robot description
        joint_indices_mapping = {}
        joint_names = []
        if "galbot" in robot.robot_name:
            joint_names = [
                "leg_joint1",
                "leg_joint2",
                "leg_joint3",
                "leg_joint4",
                "leg_joint5",
                "head_joint1",
                "head_joint2"
            ]
        elif "G2" in robot.robot_name:
            joint_names = [
                "idx01_body_joint1",
                "idx02_body_joint2",
                "idx03_body_joint3",
                "idx04_body_joint4",
                "idx05_body_joint5",
                "idx11_head_joint1",
                "idx12_head_joint2",
                "idx13_head_joint3",
            ]
        joint_indices_mapping = {joint_name: articulation.get_dof_index(joint_name) for joint_name in joint_names}
        for idx, joint_name in enumerate(init_joint_names):
            joint_index = articulation.get_dof_index(joint_name)
            if joint_index < len(robot.init_joint_position):
                robot.init_joint_position[joint_index] = init_joint_position[idx]

        def create_temp_robot_description(robot_description_path):
            with open(robot_config_dir + robot_description_path, "r") as file:
                robot_description = yaml.safe_load(file)
                for cspace_rule in robot_description["cspace_to_urdf_rules"]:
                    if cspace_rule["name"] in joint_indices_mapping and cspace_rule["rule"] == "fixed":
                        joint_index = joint_indices_mapping[cspace_rule["name"]]
                        if joint_index < len(robot.init_joint_position):
                            cspace_rule["value"] = robot.init_joint_position[joint_index]
            temp_robot_description_path = robot_description_path.replace(".yaml", "_tmp.yaml")
            with open(robot_config_dir + temp_robot_description_path, "w") as file:
                yaml.dump(robot_description, file, default_flow_style=False)
            return temp_robot_description_path

        if robot.arm_type == "dual":
            left_description_path = robot.robot_description_path["left"]
            right_description_path = robot.robot_description_path["right"]
            robot.robot_description_path = {
                "left": create_temp_robot_description(left_description_path),
                "right": create_temp_robot_description(right_description_path),
            }
        else:
            robot.robot_description_path = create_temp_robot_description(robot.robot_description_path)
        self.robot_cfg = robot
        self.ui_builder._init_kinematic_solver(self.robot_cfg)
        self.init_joint_position = robot.init_joint_position
        self.ui_builder.init_joint_position = robot.init_joint_position
        self.goal_position, self.goal_rotation = self._get_ee_pose(True)

    def _play(self):
        self.ui_builder.my_world.play()
        self._init_robot(self.robot_cfg, False, self.enable_curobo)
        self.frame_status = []

    def _initialize_articulation(self):
        return self.ui_builder.articulation

    def _init_robot(self, robot: RobotCfg, is_mocap, enable_curobo):
        self.robot_name = robot.robot_name
        self.robot_prim_path = robot.robot_prim_path
        self.end_effector_prim_path = robot.end_effector_prim_path
        self.end_effector_center_prim_path = robot.end_effector_center_prim_path
        self.arm_base_prim_path = robot.arm_base_prim_path
        self.end_effector_name = robot.end_effector_name
        self.finger_names = robot.finger_names
        self.gripper_names = [robot.left_gripper_name, robot.right_gripper_name]
        self.gripper_controll_joint = robot.gripper_controll_joint
        self.opened_positions = robot.opened_positions
        self.closed_velocities = robot.closed_velocities
        self.closed_positions = robot.closed_positions
        self.cameras = robot.cameras
        self.is_single_gripper = robot.is_single
        self.gripper_type = robot.gripper_type
        self.gripper_max_force = robot.gripper_max_force
        self.init_joint_position = robot.init_joint_position
        self.ui_builder._init_solver(robot, enable_curobo, self.batch_num)
        # self._get_observation()

    def _init_grippers(self):
        robot = self._initialize_articulation()
        end_effector_prim_path = self.end_effector_prim_path["left"]
        right_end_effector_prim_path = self.end_effector_prim_path["right"]
        self.gripper_L = ParallelGripper(
            end_effector_prim_path=end_effector_prim_path,
            joint_prim_names=self.finger_names["left"],
            joint_closed_velocities=self.closed_velocities["left"],
            joint_closed_positions=self.closed_positions["left"],
            joint_opened_positions=self.opened_positions["left"],
            joint_controll_prim=self.gripper_controll_joint["left"],
            gripper_type=self.gripper_type,
            gripper_max_force=self.gripper_max_force,
            robot_name=self.robot_name,
        )
        self.gripper_L.initialize(
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )
        self.gripper_R = ParallelGripper(
            end_effector_prim_path=right_end_effector_prim_path,
            joint_prim_names=self.finger_names["right"],
            joint_closed_velocities=self.closed_velocities["right"],
            joint_closed_positions=self.closed_positions["right"],
            joint_opened_positions=self.opened_positions["right"],
            joint_controll_prim=self.gripper_controll_joint["right"],
            gripper_type=self.gripper_type,
            gripper_max_force=self.gripper_max_force,
            robot_name=self.robot_name,
        )
        self.gripper_R.initialize(
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )
        return robot
    
    def handle_init_robot(self):
        """Handle Command 21: InitRobot"""
        robot_cfg_file = self.data["robot_cfg_file"]
        scene_usd_path = self.data["scene_usd_path"]
        self._init_robot_cfg(
            robot_cfg=robot_cfg_file,
            scene_usd=scene_usd_path,
            init_position=self.data["robot_position"],
            init_rotation=self.data["robot_rotation"],
            stand_type=self.data["stand_type"],
            size_x=self.data["stand_size_x"],
            size_y=self.data["stand_size_y"],
            init_joint_position=self.data["init_joint_position"],
            init_joint_names=self.data["init_joint_names"],
        )
        self.data_to_send = "success"

    def _hand_moveto(
        self,
        position,
        rotation,
        isRight=True,
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=None,
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links=[],
        from_current_pose=False,
    ):
        self.ui_builder._followingPos = position
        self.ui_builder._followingOrientation = rotation
        self._initialize_articulation()
        self.ui_builder._follow_target(
            isRight=isRight,
            goal_offset=goal_offset,
            path_constraint=path_constraint,
            offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
            disable_collision_links=disable_collision_links,
            from_current_pose=from_current_pose,
        )
    
    def handle_linear_move(self):
        """Handle Command 2: LinearMove"""
        self.data_to_send = None

        # NOTE: Vertical adjustment - start leg trajectory on first call.
        if self.pending_leg_adjustment is not None:
            self._start_leg_trajectory(self.pending_leg_adjustment)
            self.pending_leg_adjustment = None
            # Reset target_position to force motion gen re-plan with new torso height
            self.target_position = np.array([0.0, 0.0, 0.0])

        # NOTE: Wait for leg trajectory to finish before starting arm motion gen.
        # Returning without setting data_to_send keeps the blocking thread waiting.
        if self._is_leg_moving():
            return

        # NOTE: After the trajectory ends, force curobo to refresh its locked joint state.
        # Problem: update_lock_joints() has a threshold comparison that may skip the update,
        # or motion_gen.update_locked_joints() may use a cached/compiled model that doesn't
        # pick up the change. Setting lock_joint_states=None forces an unconditional update
        # on the next set_locked_joint_positions() call inside _follow_target().
        if self.leg_just_finished:
            for key, curobo_motion in self.ui_builder.curoboMotion.items():
                curobo_motion.lock_joint_states = None
            self.leg_just_finished = False
            # Diagnostic: confirm what leg joint positions will be sent to curobo
            articulation = self._initialize_articulation()
            if articulation is not None:
                leg_vals = {j: float(articulation.get_joint_positions()[articulation.get_dof_index(j)])
                            for j in self.leg_joint_names}
                print(f"[Vertical] Force-refresh curobo locks. Actual leg joints now: {leg_vals}")

        target_position = self.data["target_position"]
        target_rotation = self.data["target_rotation"]
        is_backend = self.data["is_backend"]
        goal_offset = self.data.get("goal_offset", [0, 0, 0, 1, 0, 0, 0])
        path_constraint = self.data.get("path_constraint", None)
        offset_and_constraint_in_goal_frame = self.data.get("offset_and_constraint_in_goal_frame", True)
        disable_collision_links = self.data.get("disable_collision_links", [])
        from_current_pose = self.data.get("from_current_pose", False)
        is_Right = False

        # spawn_target_as_usd(target_position, target_rotation, axes_container_name="LinerMoveTarget")

        if self.data["isArmRight"]:
            is_Right = True
        if not is_backend:
            self.ui_builder.rmp_flow = False
            if (
                np.linalg.norm(self.target_position - target_position) != 0.0
                or np.linalg.norm(self.target_rotation - target_rotation) != 0.0
                or self.ui_builder.get_curobo_motion(is_Right).success is False
            ):
                self.motion_run_ratio = self.data.get("motion_run_ratio", 1.0)
                self.gripper_action_timing = self.data.get("gripper_action_timing", None)
                self.target_position = target_position
                self.target_rotation = target_rotation
                self._hand_moveto(
                    position=target_position,
                    rotation=target_rotation,
                    isRight=is_Right,
                    goal_offset=goal_offset,
                    path_constraint=path_constraint,
                    offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
                    disable_collision_links=disable_collision_links,
                    from_current_pose=from_current_pose,
                )
            if self.ui_builder.get_curobo_motion(is_Right).reached:
                self.data_to_send = self.ui_builder.get_curobo_motion(is_Right).success
                # self.ui_builder.view_debug_world()
                self.motion_run_ratio = 1.0
                self.gripper_action_timing = None
        else:
            if (
                np.linalg.norm(self.target_position - target_position) != 0.0
                or np.linalg.norm(self.target_rotation - target_rotation) != 0.0
            ):
                self.target_position = target_position
                self.target_rotation = target_rotation
                self.arm_move_rmp(
                    position=target_position,
                    rotation=target_rotation,
                    ee_interpolation=self.data["ee_interpolation"],
                    distance_frame=self.data["distance_frame"],
                    is_right=is_Right,
                )
            if self.ui_builder.reached:
                self.data_to_send = True

    def on_physics_step(self):
        # NOTE: Step leg trajectory first so joints are in place before curobo plans.
        if self._is_leg_moving():
            self._step_leg_trajectory()

        # Diagnostic: print right arm joint angles every 10 steps while active,
        # to verify whether the right arm is physically moving or it's a visual effect.
        if self._is_leg_moving() or self.Command == Command.LINEAR_MOVE:
            self._diag_step += 1
            if self._diag_step % 10 == 0:
                articulation = self._initialize_articulation()
                if articulation is not None:
                    try:
                        # Cache DOF indices on first use — get_dof_index() is a string lookup
                        # that's expensive to call 7 times per print interval.
                        if self._right_arm_diag_indices is None:
                            self._right_arm_diag_indices = [
                                articulation.get_dof_index(n) for n in self._right_arm_diag_names
                            ]
                        all_pos = articulation.get_joint_positions()
                        vals = {self._right_arm_diag_names[i]: round(float(all_pos[self._right_arm_diag_indices[i]]), 4)
                                for i in range(len(self._right_arm_diag_names))}
                        phase = "LEG_TRAJ" if self._is_leg_moving() else "ARM_IK"
                        print(f"[RightArm@{self._diag_step}|{phase}] {vals}")
                    except Exception as _e:
                        print(f"[RightArm diag error] {_e}")
        else:
            self._diag_step = 0

        self.ui_builder._on_every_frame_trajectory_list()
        # curobo step
        for key, curobo_motion in self.ui_builder.curoboMotion.items():
            additional_action = None
            if self.gripper_action_timing is not None:
                state = self.gripper_action_timing.get("state", None)
                timing = self.gripper_action_timing.get("timing", None)
                is_right = self.gripper_action_timing.get("is_right", True)
                if is_right and key == "right" or (not is_right and key == "left"):
                    if state is not None and timing is not None:
                        if curobo_motion.cmd_idx >= len(curobo_motion.cmd_plan.position) * timing:
                            additional_action = self._get_gripper_action(state, is_right)
            curobo_motion.on_physics_step(self.motion_run_ratio, additional_action)

        self.on_command_step()
        
    def command_thread(self):
        while True:
            self.on_command_step()

    def on_command_step(self):
        if not self.data or not self.Command:
            return
        else:
            if self.Command == Command.LINEAR_MOVE:
                self.handle_linear_move()
            elif self.Command == Command.INIT_ROBOT:
                self.handle_init_robot()
            elif self.Command == Command.SET_TASK_METRIC:
                self.handle_follow_cube()
        
        if self.Command:
            with self.condition:
                self.condition.notify_all()

    def _get_ee_pose(self, is_right: bool) -> Tuple[np.ndarray, np.ndarray]:
        position, rotation_matrix = self.ui_builder._get_ee_pose(is_right)
        rotation = transform_utils.mat2quat_wxyz(rotation_matrix)
        return position, rotation

    def _get_gripper_action(self, state: str, isRight: bool):
        if isRight:
            self.gripper_state_R = state
            action = self.gripper_R.forward(action=self.gripper_state_R)
            return action
        else:
            self.gripper_state_L = state
            action = self.gripper_L.forward(action=self.gripper_state_L)
            return action

    def _set_gripper_state(self, state: str, isRight: bool, width):
        self.robot = self._init_grippers()
        action = self._get_gripper_action(state, isRight)
        self.robot.apply_action(action)

    def _reset_stiffness(self):
        self._init_grippers()
        self.gripper_L.reset_stiffness()
        self.gripper_R.reset_stiffness()

    def _on_reset(self):
        self._reset_stiffness()
        self.ui_builder._on_reset()
        self.target_position = [0, 0, 0]
        # self._reset_scene_material()
        # self._get_observation()
        self.frame_status = []
        self.playback_frames = {}
        self.playback_timerange = []
        self.playback_waited_frame_num = 0

    # ============================================================
    # NOTE: Vertical adjustment methods (added for free_worker_vertical task).
    # These methods implement torso height control via leg_joint1/2/3.
    # The core constraint: joint1 + joint3 ≈ joint2 (keeps torso vertical).
    # ============================================================

    def _init_vertical_defaults(self):
        """Capture the current leg joint positions as the default/reference state.
        Called after robot init. The default state is the max height (no increase beyond this).
        NOTE: Must be called after the articulation is initialized and physics is playing.
        """
        articulation = self._initialize_articulation()
        if articulation is None:
            print("[Vertical] _init_vertical_defaults: articulation not ready, using fallback config values")
            self.default_leg_joints = dict(GALBOT_LEG_DEFAULT)
            return
        try:
            for name in ["leg_joint1", "leg_joint2", "leg_joint3"]:
                idx = articulation.get_dof_index(name)
                pos = articulation.get_joint_positions()[idx]
                self.default_leg_joints[name] = float(pos)
            print(f"[Vertical] Default leg joints captured: {self.default_leg_joints}")
        except Exception as e:
            print(f"[Vertical] Failed to read leg joints: {e}, using fallback config values")
            self.default_leg_joints = dict(GALBOT_LEG_DEFAULT)

        # Also capture the default torso pose as the stable workspace reference.
        # Fix: always use THIS pose (not the current torso pose) to avoid oscillation and
        # to allow restoring after the torso has shifted in x/y while squatting.
        torso_pose = self._get_torso_world_pose()
        if torso_pose is not None:
            self.default_torso_pose = np.asarray(torso_pose, dtype=np.float64)
            self.default_torso_z = float(self.default_torso_pose[2])
            print(f"[Vertical] Default torso_z captured: {self.default_torso_z:.4f} m")

    def _get_torso_world_pose(self):
        """Get the world-frame position of torso_base_link."""
        try:
            torso_prim = XFormPrim(GALBOT_TORSO_PRIM_PATH)
            pos, _ = torso_prim.get_world_pose()
            return np.asarray(pos, dtype=np.float64)
        except Exception as e:
            print(f"[Vertical] Failed to get torso_base_link pose: {e}")
            return None

    def _get_torso_world_z(self):
        """Get the world-frame Z position of torso_base_link."""
        torso_pos = self._get_torso_world_pose()
        if torso_pos is None:
            return None
        return float(torso_pos[2])

    def _compute_leg_targets_for_height(self, target_world_position):
        """Check if the target is vertically unreachable and compute leg joint targets.

        Uses a sphere-radius model: arm workspace is approximated as a sphere of
        radius GALBOT_ARM_REACH_RADIUS centered at the shoulder.
        Only lower the torso when the target is within the usable horizontal reach
        but below the minimum reachable Z for that XY distance.

        Fix (oscillation / restore): always uses self.default_torso_pose as the stable
        reference, NOT the current lowered torso pose. Using the current torso pose
        caused the robot to oscillate and also blocked restore decisions after the
        torso had shifted while squatting:
          cycle 1: target low → lower torso → torso_z drops
          cycle 2: current torso_z is lower → sphere check says target now reachable
                   → no adjustment → legs go back to default → repeat

        With the default torso pose as reference the computation gives identical
        results every cycle for the same target position, so the leg targets converge.

        If the target is reachable from the default height and legs have been lowered,
        returns the default joint values to restore the torso (no oscillation since
        restoration only triggers when the target is reachable from the default pose.

        Calibrated 2026-03-11 via calibrate_leg_height.py:
          NEGATIVE delta lowers torso: torso_z ≈ torso_z_default + 0.67 * delta
          → to lower by h, use delta = -h / 0.67  (delta is negative)

        Constraint enforced: joint1 + joint3 ≈ joint2
        Parameterization (delta < 0 to lower):
            joint1_new = joint1_default + delta
            joint2_new = joint2_default + 2 * delta
            joint3_new = joint3_default + delta

        Returns: dict {joint_name: value} or None if no adjustment needed.
        """
        target_world_position = np.asarray(target_world_position, dtype=np.float64)
        ref = self.default_leg_joints if self.default_leg_joints else dict(GALBOT_LEG_DEFAULT)

        # Use the stable default torso pose as reference (never the current lowered pose).
        torso_pose_ref = self.default_torso_pose
        if torso_pose_ref is None:
            # Fallback: read current on first call before init is done
            torso_pose_ref = self._get_torso_world_pose()
        if torso_pose_ref is None:
            print("[Vertical] Cannot compute torso pose reference, skipping adjustment")
            return None
        torso_pose_ref = np.asarray(torso_pose_ref, dtype=np.float64)
        torso_z_ref = float(torso_pose_ref[2])

        target_world_z = float(target_world_position[2])
        horizontal_distance = float(np.linalg.norm(target_world_position[:2] - torso_pose_ref[:2]))
        current_torso_z = self._get_torso_world_z()
        torso_is_lowered = current_torso_z is not None and (torso_z_ref - current_torso_z) > 0.01

        # Approximate shoulder height + minimum reachable Z for this XY distance.
        usable_radius = max(0.0, GALBOT_ARM_REACH_RADIUS - GALBOT_REACH_SAFETY_MARGIN)
        shoulder_z = torso_z_ref + GALBOT_SHOULDER_Z_ABOVE_TORSO
        if horizontal_distance >= usable_radius:
            if torso_is_lowered:
                print(f"[Vertical] target_xy_dist={horizontal_distance:.3f} exceeds usable reach radius "
                      f"{usable_radius:.3f}; restoring legs to default because lowering is not helping.")
                return dict(ref)
            print(f"[Vertical] target_xy_dist={horizontal_distance:.3f} exceeds usable reach radius "
                  f"{usable_radius:.3f}; skip torso lowering because it is not a height-only case.")
            return None
        vertical_reach = np.sqrt(max(0.0, usable_radius ** 2 - horizontal_distance ** 2))
        min_reach_z = shoulder_z - vertical_reach

        if target_world_z >= min_reach_z:
            # Target is reachable from the default height.
            # If legs are currently lowered, schedule a restore to default.
            if torso_is_lowered:
                print(f"[Vertical] target_z={target_world_z:.3f} >= min_reach_z={min_reach_z:.3f} "
                      f"at xy_dist={horizontal_distance:.3f}. "
                      f"Restoring legs to default (torso at {current_torso_z:.3f}, "
                      f"default {torso_z_ref:.3f}).")
                return dict(ref)
            print(f"[Vertical] target_z={target_world_z:.3f} >= min_reach_z={min_reach_z:.3f} "
                  f"at xy_dist={horizontal_distance:.3f} "
                  f"(default torso_z={torso_z_ref:.3f}), no adjustment needed.")
            return None

        # Height decrease required to bring min_reach_z down to target
        height_decrease = min_reach_z - target_world_z
        print(f"[Vertical] target_z={target_world_z:.3f} below min_reach_z={min_reach_z:.3f} "
              f"at xy_dist={horizontal_distance:.3f} "
              f"(default torso_z={torso_z_ref:.3f}), need to lower torso by {height_decrease:.3f}m")

        # delta is NEGATIVE to lower the torso; clamp magnitude to GALBOT_MAX_LEG_DELTA
        delta = -(height_decrease / GALBOT_HEIGHT_PER_LEG_DELTA)
        delta = max(delta, -GALBOT_MAX_LEG_DELTA)

        # Apply constraint: joint1+delta, joint2+2*delta, joint3+delta  (delta < 0)
        new_joints = {
            "leg_joint1": ref["leg_joint1"] + delta,
            "leg_joint2": ref["leg_joint2"] + 2.0 * delta,
            "leg_joint3": ref["leg_joint3"] + delta,
        }
        print(f"[Vertical] delta={delta:.3f} rad (negative=lower), new leg joints: {new_joints}")
        return new_joints

    def _start_leg_trajectory(self, targets, n_steps=GALBOT_LEG_TRAJ_STEPS):
        """Start a linear trajectory from current to target leg joint positions.
        Fix 2: replaces instant set_joint_positions with gradual movement.
        NOTE: Called from the physics thread (handle_linear_move) or on_physics_step.
        We do NOT call curoboMotion.update_lock_joints() here; the existing
        set_locked_joint_positions() inside _follow_target() handles that correctly.
        """
        articulation = self._initialize_articulation()
        if articulation is None:
            print("[Vertical] _start_leg_trajectory: articulation not ready")
            return

        # Cache DOF indices on first use
        if self.leg_joint_indices is None:
            self.leg_joint_indices = np.array(
                [articulation.get_dof_index(j) for j in self.leg_joint_names]
            )

        all_positions = articulation.get_joint_positions()
        start = np.array([float(all_positions[i]) for i in self.leg_joint_indices])
        end = np.array([targets[j] for j in self.leg_joint_names])
        if np.allclose(start, end, atol=1e-3):
            self.leg_trajectory = None
            self.leg_traj_idx = 0
            print("[Vertical] Leg targets already reached, skipping trajectory.")
            return

        # linspace from start to end, skip step-0 (already at start)
        self.leg_trajectory = np.linspace(start, end, n_steps + 1)[1:]
        self.leg_traj_idx = 0
        print(f"[Vertical] Leg trajectory: {start} → {end} over {n_steps} steps")

    def _step_leg_trajectory(self):
        """Apply one step of the running leg trajectory. Returns True when done."""
        if self.leg_trajectory is None or self.leg_traj_idx >= len(self.leg_trajectory):
            self.leg_trajectory = None
            return True

        articulation = self._initialize_articulation()
        if articulation is None:
            self.leg_trajectory = None
            return True

        positions = self.leg_trajectory[self.leg_traj_idx].astype(np.float32)
        lowers = articulation.dof_properties["lower"][self.leg_joint_indices]
        uppers = articulation.dof_properties["upper"][self.leg_joint_indices]
        positions = np.clip(positions, lowers, uppers)
        articulation.set_joint_positions(positions, joint_indices=self.leg_joint_indices)
        self.leg_traj_idx += 1

        if self.leg_traj_idx >= len(self.leg_trajectory):
            self.leg_trajectory = None
            self.leg_just_finished = True
            print("[Vertical] Leg trajectory complete.")
            return True
        return False

    def _is_leg_moving(self):
        """True while a leg trajectory is in progress."""
        return self.leg_trajectory is not None

    def _on_blocking_thread(self, data, Command):
        self.data = data
        self.Command = Command
        with self.condition:
            while self.data_to_send is None:
                self.condition.wait()
            result = self.data_to_send
            self.data_to_send = None
            self.Command = 0
            self.result_queue.put(result)

    def blocking_start_server(self, data, Command):
        self._on_blocking_thread(data, Command)
        if not self.result_queue.empty():
            result = self.result_queue.get()
            return result

    def manual_set_command(self, manual_command, data_dict):
        if manual_command == "init_robot":
            target_position = np.array(data_dict["target_position"])
            target_rotation = np.array(data_dict["target_rotation"])

            result = self.blocking_start_server(
                data={
                    "robot_cfg_file": data_dict["robot_cfg_file"],
                    "robot_usd_path": data_dict["robot_usd_path"],
                    "scene_usd_path": data_dict["scene_usd_path"],
                    "robot_position": target_position,
                    "robot_rotation": target_rotation,
                    "stand_type": data_dict["stand_type"],
                    "stand_size_x": data_dict["stand_size_x"],
                    "stand_size_y": data_dict["stand_size_y"],
                    "init_joint_position": data_dict["init_joint_position"],
                    "init_joint_names": data_dict["init_joint_names"],
                },
                Command=Command.INIT_ROBOT,
            )

            return result
        elif manual_command == "linear_move":
            isArmRight = False
            if data_dict["robot_name"] == "right":
                isArmRight = True

            isSuccess = self.blocking_start_server(
                data={
                    "isArmRight": isArmRight,
                },
                Command=Command.LINEAR_MOVE,
            )
            return isSuccess

        elif manual_command == "follow_cube":
            cube_position, cube_orientation = self.cube_target.get_world_pose()
            robot_position, robot_orientation = self.ui_builder.articulation.get_world_pose()

            # NOTE: Vertical adjustment check.
            # Check if the cube is vertically unreachable from the default torso height.
            # If not, compute new leg joint targets and store as pending_leg_adjustment.
            # The adjustment will be applied in handle_linear_move() (physics thread).
            leg_targets = self._compute_leg_targets_for_height(cube_position)
            if leg_targets is not None:
                self.pending_leg_adjustment = leg_targets
                # Reset target_position so handle_linear_move will trigger a fresh motion plan
                self.target_position = np.array([0.0, 0.0, 0.0])

            T_world_robot = np.eye(4)
            T_world_robot[:3, 3] = robot_position
            T_world_robot[:3, :3] = transform_utils.quat2mat_wxyz(robot_orientation)
            T_world_ee = np.eye(4)
            T_world_ee[:3, 3] = cube_position
            T_world_ee[:3, :3] = transform_utils.quat2mat_wxyz(cube_orientation)
            # T_robot_ee = T_world_robot.inv @ T_world_ee
            T_robot_ee = np.linalg.inv(T_world_robot) @ T_world_ee
            ee_translation_goal = T_robot_ee[:3, 3]
            ee_orientation_goal = transform_utils.mat2quat_wxyz(T_robot_ee[:3, :3])
            print(f"[Vertical] follow_cube: cube_world_z={float(cube_position[2]):.4f}, "
                  f"ee_goal_z={ee_translation_goal[2]:.4f} (in robot frame, uncompensated)")

            isSuccess = self.blocking_start_server(
                data={
                    "isArmRight": False,
                    "target_position": ee_translation_goal,
                    "target_rotation": ee_orientation_goal,
                    "is_backend": False,
                    "ee_interpolation": False,
                    "distance_frame": 0.0008,
                    "goal_offset": [0, 0, 0, 1, 0, 0, 0],
                    "path_constraint": None,
                    # "path_constraint": [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
                    "offset_and_constraint_in_goal_frame": True,
                    "disable_collision_links": [],
                    "motion_run_ratio": 1.0,
                    "gripper_action_timing": {},
                    "from_current_pose": False,
                },
                Command=Command.LINEAR_MOVE,
            )

            return isSuccess

class Configs:
    def __init__(self):

        # cube-guided  /  target-object
        self.mode = "cube_guided"

        # self.curobo_yaml = "/home/agxi/RealityLab/genie_sim/source/data_collection/config/curobo/configs/robot/galbot_fixed_left.yml"
        self.curobo_yaml = "/home/agxi/RealityLab/genie_sim/unit_lab/configs/basic_test.yaml"
        self.curobo_config = yaml.safe_load(open(self.curobo_yaml, "r"))
        self.robot_json = "galbot_fixed_dual.json"
        self.task_json = "/home/agxi/RealityLab/genie_sim/source/data_collection/tasks/geniesim_2025/place_object_into_box_of_specific_color/galbot/place_object_into_box_of_specific_color_blue_galbot.json"

        self.robot_usd_file = "/home/agxi/Documents/assets/robots/urdf_ws/src/galbot_one_golf_description/sim_ready/galbot_one_golf/usdv2/galbot_fixed.usda"
        self.scene_usd_file = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/background/home_b/home_b_00.usda"

        self.obstacle1_usd_file = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/beverage_bottle/benchmark_beverage_bottle_001/Aligned.usda"
        self.obstacle2_usd_file = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/storage_box/benchmark_storage_box_000/Aligned.usda"


class TaskManager:
    def __init__(
        self, 
        config: Configs,
        controller: CommandController,
    ):
        self.cfg = config
        self.task_info = json.load(open(self.cfg.task_json, 'r'))
        self.controller = controller
        self.ui_builder = controller.ui_builder

        init_settings = {}
        init_settings["robot_cfg_file"] = self.cfg.robot_json
        init_settings["robot_usd_path"] = "robot/galbot/galbot_fixed.usda"
        init_settings["scene_usd_path"] = "background/home_b/home_b_00.usda"
        init_settings["target_position"] = [1.9, 0.7757971635415469, 0.0]
        init_settings["target_rotation"] = [1, 0, 0, 0]
        init_settings["stand_type"] = "cylinder"
        init_settings["stand_size_x"] = 0.1
        init_settings["stand_size_y"] = 0.1
        init_settings["init_joint_position"] = []
        init_settings["init_joint_names"] = []
        
        init_arm_pose = self.task_info["robot"]["init_arm_pose"]
        for joint_name, pos in init_arm_pose.items():
            init_settings["init_joint_position"].append(pos)
            init_settings["init_joint_names"].append(joint_name)

        self.init_settings = init_settings

        add_reference_to_stage(self.cfg.obstacle1_usd_file, "/World/obstacle1")
        self.obs1 = XFormPrim(
            prim_path="/World/obstacle1",
            position=np.array([2.7, 0.85, 0.85]),
        )
        add_reference_to_stage(self.cfg.obstacle2_usd_file, "/World/obstacle2")
        self.obs2 = XFormPrim(
            prim_path="/World/obstacle2",
            position=np.array([2.9, 0.9, 0.85]),
        )

    def send_init_command(self):
        print("Sending init_robot command from background thread...")
        self.controller.manual_set_command("init_robot", self.init_settings)

        # NOTE: Capture default leg joint positions as the reference state.
        # Brief wait to let physics settle after init before reading joint positions.
        time.sleep(0.5)
        self.controller._init_vertical_defaults()

        print("Init command completed!")


    def run(self):
        pause_step = 0
        run_step = 0
        last_physics_time = 0
        last_render_time = 0
        self.task_finished = True

        init_thread = threading.Thread(target=self.send_init_command)
        init_thread.start()

        while simulation_app.is_running():
            self.ui_builder.my_world.step(render=False)
            current_time = self.ui_builder.my_world.current_time
            need_render = False
            if last_render_time == 0 or current_time - last_render_time >= rendering_dt:
                need_render = True
                last_render_time = current_time
            if need_render:
                self.ui_builder.my_world.render()
            # self.controller.manual_set_command("linear_move")
            run_step += 1

            if run_step % 500 == 0 and self.task_finished:
                self.task_finished = False
                # 在新线程里跑阻塞逻辑
                def task():
                    self.controller.manual_set_command("follow_cube", {})
                    self.task_finished = True # 执行完后解锁

                threading.Thread(target=task).start()

            self.controller.on_physics_step()
            if self.controller.exit:
                break
            if not self.ui_builder.my_world.is_playing():
                if pause_step % 100 == 0:
                    print("**** simulation paused ****")
                pause_step += 1
                continue

        simulation_app.close()


if __name__ == "__main__":
    cfg = Configs()
    debug = True

    physics_dt = (float)(1 / args.physics_step)
    rendering_dt = (float)(1 / 30)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=physics_dt,
        rendering_dt=rendering_dt,
        device="cpu",
    )

    ui_builder = UIBuilder(
        world,
        debug=debug,
    )

    sim_controller = CommandController(
        ui_builder=ui_builder,
        enable_physics=True,
        enable_curobo=True,
        publish_ros=False,
        rendering_step=int(1 / rendering_dt),
        debug=debug,
    )
    
    task_manager = TaskManager(cfg, sim_controller)
    task_manager.run()
    
