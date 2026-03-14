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
from typing import Dict

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene

from isaacsim.core.api import World
from isaacsim.core.api.objects import cuboid, sphere
from isaacsim.core.utils import extensions
from isaacsim.core.utils.prims import get_prim_at_path, get_prim_object_type
from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim
from isaacsim.core.prims import SingleRigidPrim as RigidPrim
from isaacsim.core.prims import SingleXFormPrim as XFormPrim

try:
    from pxr import UsdGeom, UsdPhysics
except ImportError:
    pass

import omni
import asyncio
import yaml

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.sdf.world import CollisionCheckerType
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
from source.data_collection.common.base_utils import transform_utils

CUROBO_BATCH_SIZE = 10
MAX_MESH_FACES = 1000

class Configs:
    def __init__(self):

        # cube-guided  /  target-object
        self.mode = "cube_guided"

        # self.curobo_yaml = "/home/agxi/RealityLab/genie_sim/source/data_collection/config/curobo/configs/robot/galbot_fixed_left.yml"
        self.curobo_yaml = "/home/agxi/RealityLab/genie_sim/unit_lab/configs/basic_test.yaml"
        self.curobo_config = yaml.safe_load(open(self.curobo_yaml, "r"))

        self.robot_usd_file = "/home/agxi/Documents/assets/robots/urdf_ws/src/galbot_one_golf_description/sim_ready/galbot_one_golf/usdv2/galbot_fixed.usda"
        self.scene_usd_file = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/background/home_b/home_b_00.usda"

        self.obstacle1_usd_file = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/beverage_bottle/benchmark_beverage_bottle_001/Aligned.usda"
        self.obstacle2_usd_file = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/storage_box/benchmark_storage_box_000/Aligned.usda"


class UIBuilder:
    def _init_world(self):
        physics_dt = (float)(1 / args.physics_step)
        rendering_dt = (float)(1 / 30)

        # 创建仿真世界
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            device="cpu",
        )
        self.stage = self.world.stage

        robot_prim_path = "/galbot_one_golf"
        self.robot_init_position = [2.15, 0.7757971635415469, 0.0]
        self.robot_init_rotation = [1, 0, 0, 0]
        # self.robot_init_position = [1.2, 0.7757971635415469, 0.0]
        # self.robot_init_position = [-0.2, 0.0, 0.0]

        add_reference_to_stage(self.cfg.robot_usd_file, robot_prim_path)

        add_reference_to_stage(self.cfg.scene_usd_file, "/World")
        # 地面
        # self.world.scene.add_default_ground_plane()

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

        self.usd_robot = XFormPrim(
            prim_path=robot_prim_path,
            position=self.robot_init_position,
            orientation=self.robot_init_rotation
        )

        # Make a target to follow
        self.cube_target = cuboid.VisualCuboid(
            "/World/target",
            position=np.array([2.9, 0.8, 0.85]),
            # position=np.array([1.9, 0.8, 0.85]),
            # position=np.array([0.5, 0, 0.6]),
            orientation=np.array([1, 0, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )

        # 创建三色轴并分别设置局部缩放和位移
        # 轴的参数
        # axis_len = 0.1
        # thick = 0.005
        # # --- X 轴 (红) ---
        # self.ax_x = cuboid.VisualCuboid(
        #     prim_path="/World/target/ax_x", # 路径级联确保绑定
        #     name="local_ax_x",
        #     scale=np.array([axis_len, thick, thick]), # 初始化时定义局部缩放
        #     color=np.array([1, 0, 0])
        # )
        # # 修正局部位置（相对于父中心偏移）
        # self.ax_x.set_local_pose(translation=np.array([axis_len / 2, 0, 0]))

        # # --- Y 轴 (绿) ---
        # self.ax_y = cuboid.VisualCuboid(
        #     prim_path="/World/target/ax_y",
        #     name="local_ax_y",
        #     scale=np.array([thick, axis_len, thick]),
        #     color=np.array([0, 1, 0])
        # )
        # self.ax_y.set_local_pose(translation=np.array([0, axis_len / 2, 0]))

        # # --- Z 轴 (蓝) ---
        # self.ax_z = cuboid.VisualCuboid(
        #     prim_path="/World/target/ax_z",
        #     name="local_ax_z",
        #     scale=np.array([thick, thick, axis_len]),
        #     color=np.array([0, 0, 1])
        # )
        # self.ax_z.set_local_pose(translation=np.array([0, 0, axis_len / 2]))

        self.world.play()

    
    def initialize_articulation(self):
        self.robot_prim_path = self.cfg.curobo_config["robot"]["base_prim_path"]
        self.robot_name = self.cfg.curobo_config["robot"]["robot_name"]

        scene = self.world.scene
        if scene._scene_registry.name_exists(self.robot_name):
            self.articulation = scene.get_object(self.robot_name)
        else:
            self.articulation = Articulation(prim_path=self.robot_prim_path, name=self.robot_name)
            scene.add(self.articulation)
        self.articulation.initialize()

    
    def _init_robot_planner(self):

        # setup_curobo_logger("info")
        setup_curobo_logger("warn")
        n_obstacle_cuboids = 5
        n_obstacle_mesh = 5
        self.world_cfg = WorldConfig()
        self.tensor_args = TensorDeviceType()
        trajopt_dt = None
        optimize_dt = True
        trajopt_tsteps = 32
        trim_steps = None
        max_attempts = 4
        interpolation_dt = 0.05
        enable_finetune_trajopt = True

        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]
        self.lock_joints = self.robot_cfg["kinematics"]["lock_joints"]
        self.lock_js_names = []
        if self.lock_joints:
            for key in self.lock_joints:
                self.lock_js_names.append(key)
        self.lock_joint_states = None

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=4,
            num_graph_seeds=4,
            interpolation_dt=interpolation_dt,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            optimize_dt=optimize_dt,
            trajopt_dt=trajopt_dt,
            trajopt_tsteps=trajopt_tsteps,
            trim_steps=trim_steps,
            collision_activation_distance=0.01,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        print("Curobo is Ready")

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=max_attempts,
            enable_finetune_trajopt=enable_finetune_trajopt,
            time_dilation_factor=0.5,
        )

        self.usd_help = UsdHelper()
        self.usd_help.load_stage(self.world.stage)
        self.usd_help.add_world_to_stage(self.world_cfg, base_frame="/World")


    def __init__(self, configs: Configs):
        self.cfg = configs
        self.robot_cfg = self.cfg.curobo_config["robot_cfg"]
        self.debug = True

        self._init_world()
        self.initialize_articulation()
        self._init_robot_planner()

    
class TaskManager:
    def __init__(self, configs: Configs):
        self.cfg = configs

        self.sim = UIBuilder(self.cfg)

        self.articulation_controller = None
        self.tensor_args = TensorDeviceType()


    def run(self):
        if self.cfg.mode == "cube_guided":
            print("Running cube-guided mode")

            cmd_plan = None
            cmd_idx = 0
            target_pose = None
            past_pose = None
            i = 0
            spheres = None
            past_cmd = None
            num_targets = 0
            target_orientation = None
            past_orientation = None
            pose_metric = None

            while simulation_app.is_running():

                self.sim.world.step(render=False)
                self.sim.world.render()

                step_index = self.sim.world.current_time_step_index
                if self.articulation_controller is None:
                    self.articulation_controller = self.sim.articulation.get_articulation_controller()
                if step_index < 10:
                    self.sim.articulation._articulation_view.initialize()
                    idx_list = [self.sim.articulation.get_dof_index(x) for x in self.sim.j_names]
                    self.sim.articulation.set_joint_positions(self.sim.default_config, idx_list)

                    self.sim.articulation._articulation_view.set_max_efforts(
                        values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
                    )
                if step_index < 20:
                    continue

                if step_index == 50 or step_index % 1000 == 0.0:
                    print("Updating world, reading w.r.t.", self.sim.robot_prim_path)
                    obstacles = self.sim.usd_help.get_obstacles_from_stage(
                        only_paths=["/World"],
                        reference_prim_path=self.sim.robot_prim_path,
                        ignore_substring=[
                            self.sim.robot_prim_path,
                            "/World/target",
                            "/World/defaultGroundPlane",
                            # "/curobo",
                        ],
                    ).get_collision_check_world()
                    print(len(obstacles.objects))

                    self.sim.motion_gen.update_world(obstacles)
                    print("Updated World")
                
                # position and orientation of target virtual cube:
                cube_position, cube_orientation = self.sim.cube_target.get_world_pose()

                if past_pose is None:
                    past_pose = cube_position
                if target_pose is None:
                    target_pose = cube_position
                if target_orientation is None:
                    target_orientation = cube_orientation
                if past_orientation is None:
                    past_orientation = cube_orientation

                sim_js = self.sim.articulation.get_joints_state()
                if sim_js is None:
                    print("sim_js is None")
                    continue
                
                if np.any(np.isnan(sim_js.positions)):
                    log_error("isaac sim has returned NAN joint position values.")

                all_js_names = self.sim.articulation.dof_names
                sim_js_names = []
                lock_idx = []
                sim_js_positions = []
                sim_js_velocities = []
                for idx, name in enumerate(all_js_names):
                    if name not in self.sim.lock_js_names:
                        sim_js_names.append(name)
                    else:
                        lock_idx.append(idx)
                
                for idx, position in enumerate(sim_js.positions):
                    if idx not in lock_idx:
                        sim_js_positions.append(position)
                for idx, velocity in enumerate(sim_js.velocities):
                    if idx not in lock_idx:
                        sim_js_velocities.append(velocity)

                cu_js = JointState(
                    position=self.tensor_args.to_device(sim_js_positions),
                    velocity=self.tensor_args.to_device(sim_js_velocities),  # * 0.0,
                    acceleration=self.tensor_args.to_device(sim_js_velocities) * 0.0,
                    jerk=self.tensor_args.to_device(sim_js_velocities) * 0.0,
                    joint_names=sim_js_names,
                )

                if not args.reactive:
                    cu_js.velocity *= 0.0
                    cu_js.acceleration *= 0.0

                if args.reactive and past_cmd is not None:
                    cu_js.position[:] = past_cmd.position
                    cu_js.velocity[:] = past_cmd.velocity
                    cu_js.acceleration[:] = past_cmd.acceleration
                cu_js = cu_js.get_ordered_joint_state(self.sim.motion_gen.kinematics.joint_names)

                if args.visualize_spheres and step_index % 2 == 0:
                    sph_list = self.sim.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

                    if spheres is None:
                        spheres = []
                        # create spheres:

                        for si, s in enumerate(sph_list[0]):
                            sp = sphere.VisualSphere(
                                prim_path=self.sim.robot_prim_path + "/curobo/robot_sphere_" + str(si),
                                radius=float(s.radius),
                                color=np.array([0, 0.8, 0.2]),
                            )
                            sp.set_local_pose(translation=np.array([s.position[0], s.position[1], s.position[2]]))
                            spheres.append(sp)
                    else:
                        for si, s in enumerate(sph_list[0]):
                            if not np.isnan(s.position[0]):
                                spheres[si].set_local_pose(translation=np.array([s.position[0], s.position[1], s.position[2]]))
                                spheres[si].set_radius(float(s.radius))


                robot_static = False
                if (np.max(np.abs(sim_js.velocities)) < 0.5) or args.reactive:
                    robot_static = True
                if (
                    (
                        np.linalg.norm(cube_position - target_pose) > 1e-3
                        or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
                    )
                    and np.linalg.norm(past_pose - cube_position) == 0.0
                    and np.linalg.norm(past_orientation - cube_orientation) == 0.0
                    and robot_static
                ):
                    # Set EE teleop goals, use cube for simple non-vr init:
                    T_world_robot = np.eye(4)
                    T_world_robot[:3, 3] = self.sim.robot_init_position
                    T_world_robot[:3, :3] = transform_utils.quat2mat_wxyz(self.sim.robot_init_rotation)
                    T_world_ee = np.eye(4)
                    T_world_ee[:3, 3] = cube_position
                    T_world_ee[:3, :3] = transform_utils.quat2mat_wxyz(cube_orientation)
                    # T_robot_ee = T_world_robot.inv @ T_world_ee
                    T_robot_ee = np.linalg.inv(T_world_robot) @ T_world_ee
                    ee_translation_goal = T_robot_ee[:3, 3]
                    ee_orientation_teleop_goal = transform_utils.mat2quat_wxyz(T_robot_ee[:3, :3])

                    # compute curobo solution:
                    ik_goal = Pose(
                        position=self.tensor_args.to_device(ee_translation_goal),
                        quaternion=self.tensor_args.to_device(ee_orientation_teleop_goal),
                    )
                    self.sim.plan_config.pose_cost_metric = pose_metric
                    result = self.sim.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.sim.plan_config)
                    # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

                    succ = result.success.item()  # ik_result.success.item()
                    if num_targets == 1:
                        if args.constrain_grasp_approach:
                            pose_metric = PoseCostMetric.create_grasp_approach_metric()
                        if args.reach_partial_pose is not None:
                            reach_vec = self.sim.motion_gen.tensor_args.to_device(args.reach_partial_pose)
                            pose_metric = PoseCostMetric(
                                reach_partial_pose=True, reach_vec_weight=reach_vec
                            )
                        if args.hold_partial_pose is not None:
                            hold_vec = self.sim.motion_gen.tensor_args.to_device(args.hold_partial_pose)
                            pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
                    if succ:
                        num_targets += 1
                        cmd_plan = result.get_interpolated_plan()
                        cmd_plan = self.sim.motion_gen.get_full_js(cmd_plan)
                        # get only joint names that are in both:
                        idx_list = []
                        common_js_names = []
                        for x in sim_js_names:
                            if x in cmd_plan.joint_names:
                                idx_list.append(self.sim.articulation.get_dof_index(x))
                                common_js_names.append(x)
                        # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                        cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                        cmd_idx = 0

                    else:
                        carb.log_warn("Plan did not converge to a solution: " + str(result.status))
                    target_pose = cube_position
                    target_orientation = cube_orientation
                past_pose = cube_position
                past_orientation = cube_orientation
                if cmd_plan is not None:
                    cmd_state = cmd_plan[cmd_idx]
                    past_cmd = cmd_state.clone()
                    # get full dof state
                    art_action = ArticulationAction(
                        cmd_state.position.cpu().numpy(),
                        cmd_state.velocity.cpu().numpy(),
                        joint_indices=idx_list,
                    )
                    # set desired joint angles obtained from IK:
                    self.articulation_controller.apply_action(art_action)
                    cmd_idx += 1
                    for _ in range(2):
                        self.sim.world.step(render=False)
                    if cmd_idx >= len(cmd_plan.position):
                        cmd_idx = 0
                        cmd_plan = None
                        past_cmd = None

            simulation_app.close()


if __name__ == "__main__":
    cfg = Configs()
    task_manager = TaskManager(cfg)
    task_manager.run()

