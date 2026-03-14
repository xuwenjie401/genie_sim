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
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim
from isaacsim.core.prims import SingleRigidPrim as RigidPrim
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from omni.physx.scripts import utils as physx_utils

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
    curobo_kinematics = None
    curobo_kinematics_robot_cfg = {}

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
        if CuroboMotion.curobo_kinematics is None and hasattr(self, "robot_cfg") and self.robot_cfg is not None:
            CuroboMotion.curobo_kinematics_robot_cfg = robot_cfg = copy.deepcopy(self.robot_cfg)
            if robot_cfg["kinematics"].get("link_names", None) is None:
                robot_cfg["kinematics"]["link_names"] = []
            for link_name in robot_cfg["kinematics"]["collision_link_names"]:
                # TODO
                if "arm" in link_name:
                    robot_cfg["kinematics"]["link_names"] = []
            cuda_robot_model_config = CudaRobotModelConfig.from_data_dict(
                data_dict=robot_cfg, tensor_args=self.tensor_args
            )
            CuroboMotion.curobo_kinematics = CudaRobotModel(cuda_robot_model_config)
        return CuroboMotion.curobo_kinematics

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
            new_obstacles_count = 0

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
        return spheres_buffer
    

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
        self.obstacle_spheres = self.visualize_spheres(
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
        self.spheres = self.visualize_spheres(sph_list, self.spheres, prim_prefix="/curobo/robot_sphere_")


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
        if CuroboMotion.curobo_kinematics is not None and locked_joints is not None:
            if CuroboMotion.curobo_kinematics_robot_cfg["kinematics"]["lock_joints"] != locked_joints:
                print("update kinematics lock joints")
                CuroboMotion.curobo_kinematics_robot_cfg["kinematics"]["lock_joints"] = locked_joints
                robot_cfg = RobotConfig.from_dict(CuroboMotion.curobo_kinematics_robot_cfg, self.tensor_args)
                CuroboMotion.curobo_kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
        after = time.time()
        carb.log_warn("update curobo kinematics lock joints time is {}".format(after - before))


    def view_debug_world(self, force=False, refresh_obstacles=True):
        if self.debug:
            if refresh_obstacles:
                self.set_obstacles()
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
        t_so0 = time.time()
        self.set_obstacles()
        carb.log_warn(f"set obstacles cost time {time.time() - t_so0} s")

        t0 = time.time()
        self.reached = False
        if from_current_pose and not goal_offset:
            self.reached = True
            self.success = False
            print("from_current_pose is True, but goal_offset is None, return")
            return
        cube_position, cube_orientation = self.target.get_world_pose()
        print(f"[IK DEBUG] target.get_world_pose(): pos={cube_position}, quat={cube_orientation}")
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
            CuroboMotion.cached_obstacle_info.pop(x, None)  # prevent re-add by set_obstacles()

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
            if curoboMotion.target is None:
                curoboMotion.target = XFormPrim(
                    "/World/target",
                    position=self._followingPos,
                    orientation=self._followingOrientation,
                )
            else:
                curoboMotion.target.set_local_pose(
                    translation=np.asarray(self._followingPos),
                    orientation=np.asarray(self._followingOrientation),
                )
            target_world = XFormPrim(
                f"{self.robot_prim_path}/target",
            )
            target_world.set_local_pose(translation=self._followingPos, orientation=self._followingOrientation)
            if self.arm_type == "dual":
                key = self.end_effector_name["left"]
                if isRight:
                    key = self.end_effector_name["right"]
                    if key not in curoboMotion.target_links or curoboMotion.target_links[key] is None:
                        curoboMotion.target_links[key] = XFormPrim(
                            "/World/target_" + key,
                            position=np.array(self._followingPos),
                            orientation=np.array(self._followingOrientation),
                        )
                    else:
                        curoboMotion.target_links[key].set_local_pose(
                            translation=np.array(self._followingPos),
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
    
    def view_debug_world(self, force=False, refresh_obstacles=True):
        if self.articulation:
            for cm in self.curoboMotion.values():
                cm.view_debug_world(force=force, refresh_obstacles=refresh_obstacles)


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

        self.show_debug_target = False
        self.cube_target_path = "/World/debug_no_collision_cube"
        self.cube_target_default_position = np.array([2.9, 0.95, 1.0], dtype=np.float64)
        self.cube_target = None

    def _cleanup_nonessential_prims(self):
        extra_prim_paths = [
            self.cube_target_path,
            "/base_cube",
            "/ruckig",
            "/curobo",
        ]
        for prim_path in extra_prim_paths:
            prim = get_prim_at_path(prim_path)
            if prim and prim.IsValid():
                delete_prim(prim_path)
        self.cube_target = None

    def _set_collision_enabled(self, prim_path: str, enabled: bool):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        for sub_prim in Usd.PrimRange(prim):
            if not sub_prim.IsA(UsdGeom.Gprim):
                continue
            collision_api = UsdPhysics.CollisionAPI.Get(stage, sub_prim.GetPath())
            if not collision_api:
                collision_api = UsdPhysics.CollisionAPI.Apply(sub_prim)
            if not collision_api:
                continue
            collision_attr = collision_api.GetCollisionEnabledAttr()
            if not collision_attr:
                collision_attr = collision_api.CreateCollisionEnabledAttr()
            collision_attr.Set(enabled)

    def _disable_hidden_collision_prims(self, prim_path: str):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        for sub_prim in Usd.PrimRange(prim):
            if not sub_prim.IsA(UsdGeom.Gprim):
                continue
            visibility_attr = sub_prim.GetAttribute("visibility")
            is_invisible = visibility_attr.IsValid() and visibility_attr.Get() == "invisible"
            is_bbox = "/lowpoly/bbox" in str(sub_prim.GetPath())
            if not is_invisible and not is_bbox:
                continue
            collision_api = UsdPhysics.CollisionAPI.Get(stage, sub_prim.GetPath())
            if not collision_api:
                collision_api = UsdPhysics.CollisionAPI.Apply(sub_prim)
            if not collision_api:
                continue
            collision_attr = collision_api.GetCollisionEnabledAttr()
            if not collision_attr:
                collision_attr = collision_api.CreateCollisionEnabledAttr()
            collision_attr.Set(False)
            print(f"[OBSTACLE DEBUG] disabled hidden collider: {sub_prim.GetPath()}")

    def _disable_child_rigid_bodies(self, prim_path: str):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        root = stage.GetPrimAtPath(prim_path)
        if not root or not root.IsValid():
            return
        for sub_prim in Usd.PrimRange(root):
            if str(sub_prim.GetPath()) == prim_path:
                continue
            rigid_body_api = UsdPhysics.RigidBodyAPI.Get(stage, sub_prim.GetPath())
            if not rigid_body_api:
                continue
            enabled_attr = rigid_body_api.GetRigidBodyEnabledAttr()
            if not enabled_attr:
                enabled_attr = rigid_body_api.CreateRigidBodyEnabledAttr()
            enabled_attr.Set(False)
            print(f"[OBSTACLE DEBUG] disabled child rigid body: {sub_prim.GetPath()}")

    def _ensure_debug_cube_target(self, position=None, orientation=None):
        if not self.show_debug_target:
            return None

        cube_prim = get_prim_at_path(self.cube_target_path)
        if cube_prim and cube_prim.IsValid():
            if self.cube_target is None:
                self.cube_target = XFormPrim(self.cube_target_path)
            if position is not None or orientation is not None:
                cube_position, current_orientation = self.cube_target.get_world_pose()
                if position is not None:
                    cube_position = np.array(position, dtype=np.float64)
                if orientation is not None:
                    current_orientation = np.array(orientation, dtype=np.float64)
                self.cube_target.set_world_pose(
                    position=np.array(cube_position, dtype=np.float64),
                    orientation=np.array(current_orientation, dtype=np.float64),
                )
        else:
            cube_position = np.array(
                self.cube_target_default_position if position is None else position,
                dtype=np.float64,
            )
            cube_orientation = np.array(
                [1, 0, 0, 0] if orientation is None else orientation,
                dtype=np.float64,
            )
            self.cube_target = cuboid.VisualCuboid(
                self.cube_target_path,
                position=cube_position,
                orientation=cube_orientation,
                color=np.array([0.1, 0.9, 0.1]),
                size=0.05,
            )

        # Keep this cube purely visual so it can be dragged in the UI
        # without entering the collision world used by planning/physics debug.
        self._set_collision_enabled(self.cube_target_path, False)
        return self.cube_target


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
        self._cleanup_nonessential_prims()

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
        if self.show_debug_target:
            self._ensure_debug_cube_target()
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
        target_position = self.data["target_position"]
        target_rotation = self.data["target_rotation"]
        is_backend = self.data["is_backend"]
        goal_offset = self.data.get("goal_offset", [0, 0, 0, 1, 0, 0, 0])
        path_constraint = self.data.get("path_constraint", None)
        offset_and_constraint_in_goal_frame = self.data.get("offset_and_constraint_in_goal_frame", True)
        disable_collision_links = self.data.get("disable_collision_links", [])
        from_current_pose = self.data.get("from_current_pose", False)
        is_Right = False

        # clear_grasp_axes()
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
                self.ui_builder.view_debug_world()
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

    def make_gripper_stop(self, isRight):
        if self.gripper_L is None or self.gripper_R is None:
            self.robot = self._init_grippers()
        else:
            self.robot = self._initialize_articulation()
        if isRight:
            action = self.gripper_R.instant_stop()
        else:
            action = self.gripper_L.instant_stop()
        self.robot.apply_action(action)

    def handle_set_gripper_state(self):
        """Handle Command SET_GRIPPER_STATE: blocking until gripper reaches target."""
        state = self.data["gripper_state"]
        isRight = self.data["is_gripper_right"]
        width = self.data.get("opened_width", 0.08)
        if self.gripper_state != state:
            self._set_gripper_state(state=state, isRight=isRight, width=width)
            self.gripper_state = state
        if isRight:
            is_reached = self.gripper_R.is_reached
        else:
            is_reached = self.gripper_L.is_reached
        if is_reached:
            if "galbot" in getattr(self, "robot_name", "galbot"):
                self.make_gripper_stop(isRight)
            self.gripper_state = ""
            self.data_to_send = "gripper_done"

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
            elif self.Command == Command.SET_GRIPPER_STATE:
                self.handle_set_gripper_state()
        
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
        if self.gripper_L is None or self.gripper_R is None:
            self.robot = self._init_grippers()
        else:
            self.robot = self._initialize_articulation()
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
            isArmRight = data_dict.get("robot_name", "left") == "right"
            move_data = {
                "isArmRight": isArmRight,
                "target_position": np.array(data_dict.get("target_position", [0, 0, 0])),
                "target_rotation": np.array(data_dict.get("target_rotation", [1, 0, 0, 0])),
                "is_backend": data_dict.get("is_backend", False),
                "ee_interpolation": data_dict.get("ee_interpolation", False),
                "distance_frame": data_dict.get("distance_frame", 0.0008),
                "goal_offset": data_dict.get("goal_offset", [0, 0, 0, 1, 0, 0, 0]),
                "path_constraint": data_dict.get("path_constraint"),
                "offset_and_constraint_in_goal_frame": data_dict.get("offset_and_constraint_in_goal_frame", True),
                "disable_collision_links": data_dict.get("disable_collision_links", []),
                "motion_run_ratio": data_dict.get("motion_run_ratio", 1.0),
                "gripper_action_timing": data_dict.get("gripper_action_timing", {}),
                "from_current_pose": data_dict.get("from_current_pose", False),
            }
            isSuccess = self.blocking_start_server(
                data=move_data,
                Command=Command.LINEAR_MOVE,
            )
            return isSuccess

        elif manual_command == "set_gripper_state":
            result = self.blocking_start_server(
                data={
                    "gripper_state": data_dict["state"],
                    "is_gripper_right": data_dict.get("is_right", False),
                    "opened_width": data_dict.get("opened_width", 0.08),
                },
                Command=Command.SET_GRIPPER_STATE,
            )
            return result

        elif manual_command == "follow_cube":
            self._ensure_debug_cube_target()
            if self.cube_target is None:
                return False
            cube_position, cube_orientation = self.cube_target.get_world_pose()
            robot_position, robot_orientation = self.ui_builder.articulation.get_world_pose()

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
        self.physics_grasp_mode = "gripper_only"  # or "pick_lift_place_hold"


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
        init_settings["target_position"] = [2.15, 0.7757971635415469, 0.0]
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

        obstacle1_init_pos = getattr(self, "bottle_init_pos", None)
        if obstacle1_init_pos is None:
            obstacle1_init_pos = np.array([2.8, 0.95, 0.85], dtype=np.float64)
        obstacle2_init_pos = getattr(self, "box_pos", None)
        if obstacle2_init_pos is None:
            obstacle2_init_pos = np.array([3.05, 0.85, 0.85], dtype=np.float64)

        add_reference_to_stage(self.cfg.obstacle1_usd_file, "/World/obstacle1")
        self.obs1 = XFormPrim(
            prim_path="/World/obstacle1",
            position=np.array(obstacle1_init_pos, dtype=np.float64),
        )
        add_reference_to_stage(self.cfg.obstacle2_usd_file, "/World/obstacle2")
        self.obs2 = XFormPrim(
            prim_path="/World/obstacle2",
            position=np.array(obstacle2_init_pos, dtype=np.float64),
        )

    def send_init_command(self):
        print("Sending init_robot command from background thread...")
        self.controller.manual_set_command("init_robot", self.init_settings)
        
        print("Init command completed!")


class PhysicsGraspTaskManager(TaskManager):
    def __init__(self, config, controller):
        self.bottle_init_pos = np.array([2.85, 1.05, 0.85], dtype=np.float64)
        self.box_pos = np.array([3.05, 0.85, 0.85], dtype=np.float64)
        super().__init__(config, controller)
        self.mode = getattr(config, "physics_grasp_mode", "gripper_only")

        self.bottle_pos = self.bottle_init_pos.copy()

        self.place_height_offset = 0.20
        self.lift_height = 0.20

        self.robot_xy_random_range = np.array([0.005, 0.005], dtype=np.float64)
        self.bottle_world_random_range = np.array([0.015, 0.015, 0.01], dtype=np.float64)
        self.pregrasp_height = 0.1
        self.grasp_height_offset = -0.03
        self.done_printed = False
        self.debug_world_needs_render = False
        self.left_motion_debug_pending_end = False
        self.left_gripper_open_requested = False
        self.left_gripper_open_wait_steps = 0
        self.left_gripper_close_wait_steps = 0
        self.left_gripper_close_timeout_steps = 90
        self.cube_replan_position_threshold = 0.01
        self.cube_replan_rotation_threshold = 0.05
        self.cube_replan_cooldown_steps = 10
        self.cube_replan_cooldown = 0
        self.last_cube_plan_pose = None
        self.place_release_debug_steps_remaining = 0

        self.paused = True
        self._keyboard_sub = None
        self._initial_joint_positions = None
        self._register_keyboard()

    def _register_keyboard(self):
        import omni.appwindow
        appwindow = omni.appwindow.get_default_app_window()
        keyboard = appwindow.get_keyboard()
        carb_input = carb.input.acquire_input_interface()
        self._keyboard_sub = carb_input.subscribe_to_keyboard_events(
            keyboard, self._on_keyboard_event
        )
        print("[INPUT] Keys: N = next phase, R = reset")

    def _on_keyboard_event(self, event, *args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.N:
                if self.paused:
                    print(f"[INPUT] Advancing → '{self.phase}'")
                    self.paused = False
            elif event.input == carb.input.KeyboardInput.R:
                print("[INPUT] Resetting scene …")
                self._do_reset()
        return True

    def build_init_settings_for_physics_test(self):
        init_settings = copy.deepcopy(self.init_settings)
        robot_pos = np.array(init_settings["target_position"], dtype=np.float64)
        # robot_pos[:2] += np.random.uniform(
        #     low=-self.robot_xy_random_range,
        #     high=self.robot_xy_random_range,
        # )
        init_settings["target_position"] = robot_pos.tolist()
        return init_settings

    def step_sim(self, steps=60):
        for _ in range(steps):
            self.ui_builder.my_world.step(render=False)
            self.controller.on_physics_step()
            if self.ui_builder.my_world.current_time > 0:
                self.ui_builder.my_world.render()

    def setup_test_objects(self):
        self.bottle_pos = self._sample_bottle_initial_pose()
        self.obs1.set_world_pose(position=self.bottle_pos)
        self.obs2.set_world_pose(position=self.box_pos)
        self._apply_physics_to_obstacle("/World/obstacle1", mass=0.05,
                                        static_friction=0.8, dynamic_friction=0.6)
        self._apply_physics_to_obstacle("/World/obstacle2", mass=1000.0,
                                        static_friction=0.5, dynamic_friction=0.4,
                                        model_type="convexDecomposition", 
                                        # model_type="meshSimplification",
                                        kinematic=False)
        self._sync_rigid_pose("/World/obstacle1", self.bottle_pos)
        self._sync_rigid_pose("/World/obstacle2", self.box_pos)

    def _sample_bottle_initial_pose(self):
        # world_noise = np.random.uniform(
        #     low=-self.bottle_world_random_range,
        #     high=self.bottle_world_random_range,
        # )
        # return np.asarray(self.bottle_init_pos, dtype=np.float64) + world_noise
        return np.asarray(self.bottle_init_pos, dtype=np.float64)

    def _apply_physics_to_obstacle(
        self,
        prim_path: str,
        mass: float = 0.05,
        static_friction: float = 0.8,
        dynamic_friction: float = 0.6,
        model_type: str = "convexDecomposition",
        kinematic: bool = False,
    ):
        """Apply rigid body, mass, and friction to obstacle prim for physics grasp."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        items = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
            path = str(prim.GetPath())
            p = get_prim_at_path(path)
            if p and p.IsA(UsdGeom.Mesh):
                items.append(path)
        if not items:
            return
        for _prim in items:
            geometry_prim = GeometryPrim(prim_path=_prim, reset_xform_properties=False)
            obj_physics_prim_path = f"{_prim}/object_physics"
            geometry_prim.apply_physics_material(
                PhysicsMaterial(
                    prim_path=obj_physics_prim_path,
                    static_friction=static_friction,
                    dynamic_friction=dynamic_friction,
                    restitution=None,
                )
            )
            obj_physics_prim = stage.GetPrimAtPath(obj_physics_prim_path)
            if obj_physics_prim:
                physx_material_api = PhysxSchema.PhysxMaterialAPI(obj_physics_prim)
                if physx_material_api is not None:
                    fric_combine_mode = physx_material_api.GetFrictionCombineModeAttr().Get()
                    if fric_combine_mode is None:
                        physx_material_api.CreateFrictionCombineModeAttr().Set("max")
                    elif fric_combine_mode != "max":
                        physx_material_api.GetFrictionCombineModeAttr().Set("max")
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            self.controller._disable_child_rigid_bodies(prim_path)
            physx_utils.setRigidBody(prim, model_type, kinematic)
            if not kinematic:
                physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                physx_rb_api.GetSleepThresholdAttr().Set(0.0)
            rigid_prim = RigidPrim(prim_path=prim_path, mass=mass)
            physics_api = UsdPhysics.MassAPI.Apply(rigid_prim.prim)
            physics_api.CreateMassAttr().Set(mass)
            rigid_prim.initialize()
            self.controller.rigid_bodies[prim_path] = rigid_prim
        if prim_path == "/World/obstacle2":
            self.controller._disable_hidden_collision_prims(prim_path)
        for curobo_motion in self.controller.ui_builder.curoboMotion.values():
            if curobo_motion:
                curobo_motion.add_obstacle_from_prim_path(prim_path, None)
                break
        if prim_path == "/World/obstacle2":
            self._debug_dump_obstacle_geom_tree(prim_path)

    def _sync_rigid_pose(self, prim_path: str, position, orientation=None):
        """Keep rigid-body pose aligned with the visible XForm pose."""
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        rigid_prim = self.controller.rigid_bodies.get(prim_path)
        if rigid_prim is None:
            rigid_prim = RigidPrim(prim_path=prim_path)
            rigid_prim.initialize()
            self.controller.rigid_bodies[prim_path] = rigid_prim
        rigid_prim.set_world_pose(
            position=np.asarray(position, dtype=np.float64),
            orientation=np.asarray(orientation, dtype=np.float64),
        )
        rigid_prim.set_linear_velocity(np.zeros(3, dtype=np.float64))
        rigid_prim.set_angular_velocity(np.zeros(3, dtype=np.float64))

    def _debug_dump_obstacle_geom_tree(self, prim_path: str):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        root = stage.GetPrimAtPath(prim_path)
        if not root or not root.IsValid():
            return
        print(f"[OBSTACLE DEBUG] geometry tree for {prim_path}")
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            path = str(prim.GetPath())
            collision_api = UsdPhysics.CollisionAPI.Get(stage, path)
            collision_enabled = None
            if collision_api:
                collision_attr = collision_api.GetCollisionEnabledAttr()
                if collision_attr.IsValid():
                    collision_enabled = collision_attr.Get()
            visibility_attr = prim.GetAttribute("visibility")
            visibility = visibility_attr.Get() if visibility_attr.IsValid() else None
            approximation_attr = prim.GetAttribute("physics:approximation")
            approximation = approximation_attr.Get() if approximation_attr.IsValid() else None
            print(
                "[OBSTACLE DEBUG] "
                f"path={path}, type={prim.GetTypeName()}, visibility={visibility}, "
                f"collision_enabled={collision_enabled}, approximation={approximation}"
            )

    def _init_robot_for_test(self):
        init_settings = self.build_init_settings_for_physics_test()
        self.controller._cleanup_nonessential_prims()
        self.controller._init_robot_cfg(
            robot_cfg=init_settings["robot_cfg_file"],
            scene_usd=init_settings["scene_usd_path"],
            init_position=np.array(init_settings["target_position"]),
            init_rotation=np.array(init_settings["target_rotation"]),
            stand_type=init_settings["stand_type"],
            size_x=init_settings["stand_size_x"],
            size_y=init_settings["stand_size_y"],
            init_joint_position=init_settings["init_joint_position"],
            init_joint_names=init_settings["init_joint_names"],
        )
        self.controller._init_grippers()
        self._force_left_gripper_open_pose()
        self.set_left_gripper("open")
        self.left_gripper_open_requested = True

    def _force_left_gripper_open_pose(self):
        robot = self.ui_builder.articulation
        gripper = self.controller.gripper_L
        if robot is None or gripper is None:
            return
        target_positions = np.asarray(gripper.joint_opened_positions, dtype=np.float64)
        joint_indices = np.asarray(gripper.joint_dof_indicies[: len(target_positions)], dtype=np.int64)
        robot.set_joint_positions(target_positions, joint_indices=joint_indices)
        robot.set_joint_velocities(np.zeros(len(target_positions), dtype=np.float64), joint_indices=joint_indices)

    def _left_gripper_is_fully_open(self, tol=0.02):
        robot = self.ui_builder.articulation
        gripper = self.controller.gripper_L
        if robot is None or gripper is None:
            return False
        if "galbot" in getattr(self.controller, "robot_name", "").lower():
            control_joint_idx = self.ui_builder.articulation.get_dof_index("left_gripper_r_knuckle_joint")
            current_position = robot.get_joint_positions(joint_indices=[control_joint_idx])
            if current_position is None:
                return False
            return abs(float(current_position[0])) <= tol
        target_positions = np.asarray(gripper.joint_opened_positions, dtype=np.float64)
        joint_indices = np.asarray(gripper.joint_dof_indicies[: len(target_positions)], dtype=np.int64)
        current_positions = robot.get_joint_positions(joint_indices=joint_indices)
        if current_positions is None:
            return False
        return np.all(np.abs(np.asarray(current_positions, dtype=np.float64) - target_positions) <= tol)

    def _world_pose_to_robot_frame(self, target_position, target_rotation):
        robot_position, robot_orientation = self.ui_builder.articulation.get_world_pose()
        print(f"[FRAME DEBUG] robot world pose: pos={robot_position}, quat={robot_orientation}")
        print(f"[FRAME DEBUG] target world pose: pos={target_position}, quat={target_rotation}")
        T_world_robot = np.eye(4)
        T_world_robot[:3, 3] = robot_position
        T_world_robot[:3, :3] = transform_utils.quat2mat_wxyz(robot_orientation)
        T_world_ee = np.eye(4)
        T_world_ee[:3, 3] = np.asarray(target_position, dtype=np.float64)
        T_world_ee[:3, :3] = transform_utils.quat2mat_wxyz(
            np.asarray(target_rotation, dtype=np.float64)
        )
        T_robot_ee = np.linalg.inv(T_world_robot) @ T_world_ee
        local_pos = T_robot_ee[:3, 3]
        local_quat = transform_utils.mat2quat_wxyz(T_robot_ee[:3, :3])
        print(f"[FRAME DEBUG] converted robot-frame pose: pos={local_pos}, quat={local_quat}")
        return local_pos, local_quat

    def move_left(self, target_position, target_rotation, from_current_pose=False,
                  disable_collision_links=None):
        """Use left-arm curobo motiongen to move from current pose to target world pose."""
        curobo_motion = self.ui_builder.get_curobo_motion(False)
        if curobo_motion is None:
            print("Left-arm curobo motion is not initialized.")
            return False

        local_pos, local_quat = self._world_pose_to_robot_frame(
            target_position, target_rotation
        )
        self.controller.motion_run_ratio = 1.0
        self.controller.gripper_action_timing = None
        self.controller.target_position = np.asarray(local_pos)
        self.controller.target_rotation = np.asarray(local_quat)
        self.controller._hand_moveto(
            position=local_pos,
            rotation=local_quat,
            isRight=False,
            goal_offset=[0, 0, 0, 1, 0, 0, 0],
            path_constraint=None,
            offset_and_constraint_in_goal_frame=True,
            disable_collision_links=disable_collision_links if disable_collision_links else [],
            from_current_pose=from_current_pose,
        )
        curobo_motion.reached = False
        self.left_motion_debug_pending_end = True
        self._request_debug_world_snapshot(refresh_obstacles=True)
        return True

    def set_left_gripper(self, state):
        self.controller._set_gripper_state(state=state, isRight=False, width=0.08)

    def _left_gripper_reached(self):
        if self.controller.gripper_L is None:
            return False
        return self.controller.gripper_L.is_reached

    def _finish_left_gripper_action(self):
        if self._left_gripper_reached():
            self.controller.make_gripper_stop(False)
            return True
        if self.left_gripper_close_wait_steps >= self.left_gripper_close_timeout_steps:
            print("Left gripper close timed out, continuing to lift.")
            self.controller.make_gripper_stop(False)
            return True
        return False

    def _get_left_motion_status(self):
        curobo_motion = self.ui_builder.get_curobo_motion(False)
        if curobo_motion is None:
            return False, False
        if curobo_motion.reached and self.left_motion_debug_pending_end:
            ee_pos, ee_quat = self.controller._get_ee_pose(is_right=False)
            bottle_pos, _ = self._get_current_bottle_pose()
            print(f"[REACH DEBUG] EE world pos: {ee_pos}")
            print(f"[REACH DEBUG] bottle world pos: {bottle_pos}")
            print(f"[REACH DEBUG] delta (EE - bottle): {ee_pos - bottle_pos}")
            self._request_debug_world_snapshot(refresh_obstacles=True)
            self.left_motion_debug_pending_end = False
        return curobo_motion.reached, curobo_motion.success

    def _get_current_bottle_pose(self):
        rigid_prim = self.controller.rigid_bodies.get("/World/obstacle1")
        if rigid_prim is not None:
            bottle_position, bottle_orientation = rigid_prim.get_world_pose()
        else:
            bottle_position, bottle_orientation = self.obs1.get_world_pose()
        self.bottle_pos = np.asarray(bottle_position, dtype=np.float64)
        return self.bottle_pos, np.asarray(bottle_orientation, dtype=np.float64)

    def _debug_log_place_drop_state(self, tag: str):
        bottle_pos, _ = self._get_current_bottle_pose()
        bottle_rigid = self.controller.rigid_bodies.get("/World/obstacle1")
        bottle_vel = None
        if bottle_rigid is not None and hasattr(bottle_rigid, "get_linear_velocity"):
            try:
                bottle_vel = np.asarray(bottle_rigid.get_linear_velocity(), dtype=np.float64)
            except Exception:
                bottle_vel = None
        box_xform_pos, _ = self.obs2.get_world_pose()
        box_rigid = self.controller.rigid_bodies.get("/World/obstacle2")
        if box_rigid is not None:
            box_rigid_pos, _ = box_rigid.get_world_pose()
            box_rigid_pos = np.asarray(box_rigid_pos, dtype=np.float64)
        else:
            box_rigid_pos = None
        print(
            "[PLACE DROP DEBUG] "
            f"{tag}: bottle_pos={bottle_pos}, bottle_vel={bottle_vel}, "
            f"box_xform_pos={np.asarray(box_xform_pos, dtype=np.float64)}, box_rigid_pos={box_rigid_pos}"
        )

    def _debug_track_place_release(self, tag: str):
        if self.place_release_debug_steps_remaining <= 0:
            return
        should_log = (
            self.place_release_debug_steps_remaining >= 86
            or self.place_release_debug_steps_remaining % 10 == 0
        )
        if should_log:
            self._debug_log_place_drop_state(tag)
        self.place_release_debug_steps_remaining -= 1

    def _get_debug_cube_pose(self):
        cube_target = self.controller._ensure_debug_cube_target()
        if cube_target is None:
            return None, None
        cube_position, cube_orientation = cube_target.get_world_pose()
        return np.asarray(cube_position, dtype=np.float64), np.asarray(cube_orientation, dtype=np.float64)

    def _cube_replan_needed(self, cube_position, cube_orientation):
        if self.last_cube_plan_pose is None:
            return True

        last_position, last_orientation = self.last_cube_plan_pose
        position_delta = np.linalg.norm(cube_position - last_position)
        orientation_dot = abs(float(np.dot(cube_orientation, last_orientation)))
        orientation_dot = min(max(orientation_dot, -1.0), 1.0)
        rotation_delta = 2.0 * np.arccos(orientation_dot)
        return (
            position_delta >= self.cube_replan_position_threshold
            or rotation_delta >= self.cube_replan_rotation_threshold
        )

    def _plan_to_debug_cube(self, force=False):
        cube_position, cube_orientation = self._get_debug_cube_pose()
        if cube_position is None:
            return False

        if not force and not self._cube_replan_needed(cube_position, cube_orientation):
            return False

        print(f"[CUBE TRACK] replan to cube pos={cube_position}, quat={cube_orientation}")
        if self.move_left(cube_position, cube_orientation, from_current_pose=False):
            self.last_cube_plan_pose = (cube_position.copy(), cube_orientation.copy())
            self.cube_replan_cooldown = self.cube_replan_cooldown_steps
            return True
        return False

    def _update_cube_tracking(self):
        curobo_motion = self.ui_builder.get_curobo_motion(False)
        if curobo_motion is None:
            self._begin_phase("done")
            return

        if self.cube_replan_cooldown > 0:
            self.cube_replan_cooldown -= 1

        reached, success = self._get_left_motion_status()
        if reached and not success:
            print("Left-arm motiongen failed while tracking cube.")

        cube_position, cube_orientation = self._get_debug_cube_pose()
        if cube_position is None:
            return

        if reached:
            self._plan_to_debug_cube(force=self._cube_replan_needed(cube_position, cube_orientation))
            return

        if self.cube_replan_cooldown == 0 and self._cube_replan_needed(cube_position, cube_orientation):
            self._plan_to_debug_cube(force=True)

    def _hold_left_gripper_at_grasp(self):
        """Switch gripper from velocity-close to position-hold at current joint position."""
        gripper = self.controller.gripper_L
        robot = self.ui_builder.articulation
        if gripper is None or robot is None:
            return
        # Switch USD drive to position control with non-zero stiffness
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(gripper._joint_control_prim)
        if prim:
            drive = UsdPhysics.DriveAPI.Get(prim, gripper.gripper_type)
            if drive:
                drive.GetStiffnessAttr().Set(5000)
                drive.GetMaxForceAttr().Set(10.0)
        # Read current position and set as hold target
        ctrl_dof = int(gripper._joint_dof_indicies[1])
        mirror_dof = int(gripper._joint_dof_indicies[0])
        current = robot.get_joint_positions(joint_indices=[ctrl_dof, mirror_dof])
        if current is None or len(current) < 2:
            return
        target_positions = [None] * gripper._articulation_num_dofs
        target_positions[ctrl_dof] = float(current[0])
        target_positions[mirror_dof] = float(current[1])
        robot.apply_action(ArticulationAction(joint_positions=target_positions))


    def _begin_phase(self, phase, wait_steps=0):
        self.phase = phase
        self.phase_wait_steps = wait_steps
        self.paused = True
        print(f"[PHASE] → '{phase}' | Press N to run, R to reset")

    def _request_debug_world_snapshot(self, refresh_obstacles=True):
        self.ui_builder.view_debug_world(force=True, refresh_obstacles=refresh_obstacles)
        self.debug_world_needs_render = True

    def _tick_phase_wait(self):
        if self.phase_wait_steps > 0:
            self.phase_wait_steps -= 1
            return False
        return True

    def _update_state_machine(self):
        # Re-assert position hold every step (including during pause) to outlast any drive override
        if self.phase in ("post_close_settle", "wait_lift", "wait_place"):
            self._hold_left_gripper_at_grasp()
        if self.paused:
            return
        if self.phase == "settle_after_init":
            if self._tick_phase_wait():
                self.setup_test_objects()
                self._begin_phase("settle_after_object_setup", 60)
        elif self.phase == "settle_after_object_setup":
            if self._tick_phase_wait():
                self._force_left_gripper_open_pose()
                self.set_left_gripper("open")
                self.left_gripper_open_requested = True
                self.left_gripper_open_wait_steps = 0
                self._begin_phase("wait_gripper_open")
        elif self.phase == "wait_gripper_open":
            self.left_gripper_open_wait_steps += 1
            if self._left_gripper_is_fully_open():
                self.controller.make_gripper_stop(False)
                self._begin_phase("post_open_settle", 30)
            else:
                if self.left_gripper_open_wait_steps % 20 == 0:
                    self._force_left_gripper_open_pose()
                    self.set_left_gripper("open")
        elif self.phase == "post_open_settle":
            if self._tick_phase_wait():
                ee_pos, ee_quat = self.controller._get_ee_pose(is_right=False)
                bottle_pos, _ = self._get_current_bottle_pose()
                pregrasp_pose = bottle_pos.copy()
                pregrasp_pose[2] += self.pregrasp_height
                if self.move_left(pregrasp_pose, ee_quat, from_current_pose=False):
                    self._begin_phase("wait_pregrasp")
                else:
                    self._begin_phase("done")
        elif self.phase == "wait_pregrasp":
            reached, success = self._get_left_motion_status()
            if reached:
                if success:
                    ee_pos, ee_quat = self.controller._get_ee_pose(is_right=False)
                    bottle_pos, _ = self._get_current_bottle_pose()
                    grasp_pose = bottle_pos.copy()
                    grasp_pose[2] += self.grasp_height_offset
                    if self.move_left(grasp_pose, ee_quat, from_current_pose=False):
                        self._begin_phase("wait_grasp_pose")
                    else:
                        self._begin_phase("done")
                else:
                    print("Left-arm motiongen failed at pregrasp.")
                    self._begin_phase("done")
        elif self.phase == "wait_grasp_pose":
            reached, success = self._get_left_motion_status()
            if reached:
                if success:
                    self.set_left_gripper("close")
                    self.left_gripper_close_wait_steps = 0
                    self._begin_phase("wait_gripper_close")
                else:
                    print("Left-arm motiongen failed at grasp pose.")
                    self._begin_phase("done")
        elif self.phase == "wait_gripper_close":
            self.left_gripper_close_wait_steps += 1
            if self._finish_left_gripper_action():
                self._hold_left_gripper_at_grasp()
                self._begin_phase("post_close_settle", 120)
        elif self.phase == "post_close_settle":
            if self._tick_phase_wait():
                self.ui_builder.remove_objects_from_world(["/World/obstacle1"])
                ee_pos, ee_quat = self.controller._get_ee_pose(is_right=False)
                lift_pose = ee_pos.copy()
                lift_pose[2] += self.lift_height
                if self.move_left(lift_pose, ee_quat, from_current_pose=False,
                                  disable_collision_links=["left_gripper.*"]):
                    self._begin_phase("wait_lift")
                else:
                    self._begin_phase("done")
        elif self.phase == "wait_lift":
            reached, success = self._get_left_motion_status()
            if reached:
                if not success:
                    print("Left-arm motiongen failed at lift.")
                    self._begin_phase("done")
                    return
                ee_pos, ee_quat = self.controller._get_ee_pose(is_right=False)
                place_pose = np.array(
                    [
                        self.box_pos[0],
                        self.box_pos[1],
                        self.box_pos[2] + self.place_height_offset,
                    ],
                    dtype=np.float64,
                )
                if self.move_left(place_pose, ee_quat,
                                  disable_collision_links=["left_gripper.*"]):
                    self._begin_phase("wait_place")
                else:
                    self._begin_phase("done")
        elif self.phase == "wait_place":
            reached, success = self._get_left_motion_status()
            if reached:
                if not success:
                    print("Left-arm motiongen failed at place.")
                    self._begin_phase("done")
                    return
                self.set_left_gripper("open")
                self.left_gripper_open_wait_steps = 0
                self.place_release_debug_steps_remaining = 90
                self._debug_log_place_drop_state("release_start")
                self._begin_phase("wait_place_gripper_open")
        elif self.phase == "wait_place_gripper_open":
            self.left_gripper_open_wait_steps += 1
            self._debug_track_place_release(f"wait_place_gripper_open step={self.left_gripper_open_wait_steps}")
            if self._left_gripper_is_fully_open():
                self.controller.make_gripper_stop(False)
                self._begin_phase("post_place_settle", 60)
            elif self.left_gripper_open_wait_steps >= 120:
                self.controller.make_gripper_stop(False)
                self._begin_phase("post_place_settle", 60)
            else:
                if self.left_gripper_open_wait_steps % 20 == 0:
                    self.set_left_gripper("open")
        elif self.phase == "post_place_settle":
            self._debug_track_place_release("post_place_settle")
            if self._tick_phase_wait():
                self._begin_phase("done")
        elif self.phase == "done":
            pass  # idle — press R to reset

    def _do_reset(self):
        self.controller._cleanup_nonessential_prims()
        # Stop curobo motion
        for cm in self.ui_builder.curoboMotion.values():
            if cm:
                cm.cmd_plan = None
                cm.cmd_idx = 0
                cm.reached = True
                cm.attached_objects = []
                cm.motion_gen.detach_object_from_robot()

        # Reset robot arm joints to snapshotted initial state
        articulation = self.ui_builder.articulation
        if articulation is not None and self._initial_joint_positions is not None:
            articulation.set_joint_positions(self._initial_joint_positions)
            articulation.set_joint_velocities(
                np.zeros(len(self._initial_joint_positions), dtype=np.float64))

        # Force gripper open
        self._force_left_gripper_open_pose()
        self.set_left_gripper("open")
        self.left_gripper_open_requested = True
        self.place_release_debug_steps_remaining = 0

        # Reset obstacle positions (both XForm and RigidPrim)
        self.bottle_pos = self._sample_bottle_initial_pose()
        self.obs1.set_world_pose(position=self.bottle_pos)
        self.obs2.set_world_pose(position=self.box_pos)
        self._sync_rigid_pose("/World/obstacle1", self.bottle_pos)
        self._sync_rigid_pose("/World/obstacle2", self.box_pos)

        # Rebuild curobo collision world
        CuroboMotion.cached_obstacle_info = {}
        for cm in self.ui_builder.curoboMotion.values():
            if cm:
                cm._extract_cached_obstacles()
                cm.set_obstacles()
                break  # cached_obstacle_info is class-level; one call suffices

        # Restart FSM (also re-sets paused + prints prompt)
        self._begin_phase("settle_after_init", 120)
        print("[RESET] Complete.")

    def run(self):
        self._init_robot_for_test()
        # Snapshot initial joint state before any motion
        articulation = self.ui_builder.articulation
        if articulation is not None:
            raw = articulation.get_joint_positions()
            if raw is not None:
                self._initial_joint_positions = np.asarray(raw, dtype=np.float64).copy()
        self.phase = "settle_after_init"
        self.phase_wait_steps = 120
        self.paused = True
        print("[PHASE] → 'settle_after_init' | Press N to run, R to reset")

        pause_step = 0
        last_render_time = 0
        while simulation_app.is_running():
            self.ui_builder.my_world.step(render=False)
            current_time = self.ui_builder.my_world.current_time
            need_render = False
            if last_render_time == 0 or current_time - last_render_time >= rendering_dt:
                need_render = True
                last_render_time = current_time

            self.controller.on_physics_step()

            if self.ui_builder.my_world.is_playing():
                self._update_state_machine()
            else:
                if pause_step % 100 == 0:
                    print("**** simulation paused ****")
                pause_step += 1

            if need_render or self.debug_world_needs_render:
                self.ui_builder.my_world.render()
                self.debug_world_needs_render = False

            if self.controller.exit:
                break

        if self._keyboard_sub is not None:
            import omni.appwindow
            carb.input.acquire_input_interface().unsubscribe_to_keyboard_events(
                omni.appwindow.get_default_app_window().get_keyboard(),
                self._keyboard_sub,
            )

        simulation_app.close()


if __name__ == "__main__":
    cfg = Configs()
    debug = False
    # debug = True

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
    
    physics_grasp_manager = PhysicsGraspTaskManager(cfg, sim_controller)
    physics_grasp_manager.run()

