try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch
import time

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

import omni
import asyncio
import yaml

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
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

CONFIG_FILE = "/home/agxi/RealityLab/genie_sim/unit_lab/configs/basic_test.yaml"


class Robot:
    def __init__(self, cfg):
        pass

    def reset(self, uibuilder):
        pass


class RobotCfg(Robot):
    def __init__(self, cfg_dict):
        robot_cfg = cfg_dict["robot"]

        self.robot_name = robot_cfg["robot_name"]
        self.robot_usd = robot_cfg["robot_usd"]
        self.robot_prim_path = robot_cfg["base_prim_path"]
        self.arm_type = robot_cfg["arm"]

        self.dof_nums = robot_cfg["dof_nums"]
        self.lock_joints = robot_cfg["lock_joints"]
        self.joint_delta_time = robot_cfg["joint_delta_time"]

        self.active_arm_joints = robot_cfg["active_arm_joints"]

        self.gripper_type = robot_cfg["gripper"]["gripper_type"]
        self.gripper_max_force = robot_cfg["gripper"]["max_force"]
        gripper_names = robot_cfg["gripper"].get("gripper_name", {"left": "omnipicker", "right": "omnipicker"})

        self.left_gripper_name = gripper_names["left"]
        self.right_gripper_name = gripper_names["right"]
        if robot_cfg["arm"] == "dual":
            self.end_effector_name = robot_cfg["gripper"]["end_effector_name"]
        elif robot_cfg["arm"] == "right":
            self.end_effector_name = robot_cfg["gripper"]["end_effector_name"]["right"]
        else:
            self.end_effector_name = robot_cfg["gripper"]["end_effector_name"]["left"]
        self.end_effector_prim_path = robot_cfg["gripper"]["end_effector_prim_path"]
        if "end_effector_center_prim_path" in robot_cfg["gripper"]:
            self.end_effector_center_prim_path = robot_cfg["gripper"]["end_effector_center_prim_path"]
        else:
            self.end_effector_center_prim_path = self.end_effector_prim_path
        if "arm_base_prim_path" in robot_cfg:
            self.arm_base_prim_path = robot_cfg["arm_base_prim_path"]
        else:
            self.arm_base_prim_path = self.robot_prim_path
        self.finger_names = robot_cfg["gripper"]["finger_names"]
        self.gripper_controll_joint = robot_cfg["gripper"]["gripper_controll_joint"]
        self.opened_positions = robot_cfg["gripper"]["opened_positions"]
        self.closed_velocities = robot_cfg["gripper"]["closed_velocities"]
        self.closed_positions = robot_cfg["gripper"]["closed_positions"]
        self.action_deltas = np.array([-0.1, -0.1])
        # init curobo
        self.curobo_config_file = robot_cfg["curobo"]["curobo_config_file"]
        self.curobo_urdf_path = robot_cfg["curobo"]["curobo_urdf_path"]
        self.curobo_urdf_name = robot_cfg["curobo"]["curobo_urdf_name"]

        self.init_position = robot_cfg["init_position"]
        self.init_rotation = robot_cfg["init_rotation"]
        self.init_joint_names = robot_cfg["init_joint_names"]
        self.init_joint_position = robot_cfg["init_joint_position"]


class BasicRunner:
    def __init__(self, cfg_file):

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

        # 地面
        self.world.scene.add_default_ground_plane()

        # 配置
        all_cfgs = yaml.safe_load(open(cfg_file, "r"))
        robot_cfg = all_cfgs["robot_cfg"]
        self.robot = RobotCfg(all_cfgs)

        self.usd_objects = {}

        self.init_robot()

        # Make a target to follow
        self.target = cuboid.VisualCuboid(
            "/World/target",
            position=np.array([0.5, 0, 0.6]),
            orientation=np.array([1, 0, 0, 0]),
            # position=np.array([0.4, -0.1, 0.2]),
            # orientation=np.array([0.56099, 0.56099, 0.43046, 0.43046]),
            # position=np.array([0.4, -0.1, 1.0]),
            # orientation=np.array([0.56099, 0.56099, 0.43046, 0.43046]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )

        # 2. 创建三色轴并分别设置局部缩放和位移
        # 轴的参数
        axis_len = 0.1
        thick = 0.005
        # --- X 轴 (红) ---
        self.ax_x = cuboid.VisualCuboid(
            prim_path="/World/target/ax_x", # 路径级联确保绑定
            name="local_ax_x",
            scale=np.array([axis_len, thick, thick]), # 初始化时定义局部缩放
            color=np.array([1, 0, 0])
        )
        # 修正局部位置（相对于父中心偏移）
        self.ax_x.set_local_pose(translation=np.array([axis_len / 2, 0, 0]))

        # --- Y 轴 (绿) ---
        self.ax_y = cuboid.VisualCuboid(
            prim_path="/World/target/ax_y",
            name="local_ax_y",
            scale=np.array([thick, axis_len, thick]),
            color=np.array([0, 1, 0])
        )
        self.ax_y.set_local_pose(translation=np.array([0, axis_len / 2, 0]))

        # --- Z 轴 (蓝) ---
        self.ax_z = cuboid.VisualCuboid(
            prim_path="/World/target/ax_z",
            name="local_ax_z",
            scale=np.array([thick, thick, axis_len]),
            color=np.array([0, 0, 1])
        )
        self.ax_z.set_local_pose(translation=np.array([0, 0, axis_len / 2]))

        # warmup curobo instance
        self.usd_help = UsdHelper()

        self.articulation_controller = None

        # setup_curobo_logger("info")
        setup_curobo_logger("warn")
        n_obstacle_cuboids = 30
        n_obstacle_mesh = 100
        self.world_cfg = WorldConfig()
        self.tensor_args = TensorDeviceType()
        trajopt_dt = None
        optimize_dt = True
        trajopt_tsteps = 32
        trim_steps = None
        max_attempts = 4
        interpolation_dt = 0.05
        enable_finetune_trajopt = True

        self.j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
        self.lock_joints = robot_cfg["kinematics"]["lock_joints"]
        self.lock_js_names = []
        if self.lock_joints:
            for key in self.lock_joints:
                self.lock_js_names.append(key)
        self.lock_joint_states = None

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            self.world_cfg,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=interpolation_dt,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            optimize_dt=optimize_dt,
            trajopt_dt=trajopt_dt,
            trajopt_tsteps=trajopt_tsteps,
            trim_steps=trim_steps,
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

        self.usd_help.load_stage(self.world.stage)
        self.usd_help.add_world_to_stage(self.world_cfg, base_frame="/World")


    def initialize_articulation(self):
        scene = self.world.scene
        if scene._scene_registry.name_exists(self.robot_name):
            self.articulation = scene.get_object(self.robot_name)
        else:
            self.articulation = Articulation(prim_path=self.robot_articulation_path, name=self.robot_name)
            scene.add(self.articulation)
        self.articulation.initialize()

        if self.enable_curobo:
            pass


    def _init_solver(self, robot: RobotCfg, enable_curobo):
        self.enable_curobo = enable_curobo
        self.robot_name = robot.robot_name
        self.robot_prim_path = robot.robot_prim_path
        self.robot_articulation_path = self.robot_prim_path
        # if "galbot" in self.robot_name:
        #     self.robot_articulation_path = "/galbot_one_golf/base_link"
        self.dof_nums = robot.dof_nums
        self.lock_joints = robot.lock_joints
        self.joint_delta_time = robot.joint_delta_time
        self.curobo_config_file = robot.curobo_config_file

        self.init_joint_position = robot.init_joint_position
        self.end_effector_prim_path = robot.end_effector_prim_path
        self.initialize_articulation()
        self.arm_type = robot.arm_type
        self.end_effector_name = robot.end_effector_name
        self.active_arm_joints = robot.active_arm_joints
    

    def _play(self):
        self.world.play()

        self.frame_status = []


    def _on_reset(self):
        async def _on_reset_async():
            await omni.kit.app.get_app().next_update_async()
            self.initialize_articulation()
            self.rmp_move = False

        asyncio.ensure_future(_on_reset_async())
        return
    

    def init_robot(self):
        get_prim_at_path("/World")
        if "World" not in self.robot.robot_prim_path:
            add_reference_to_stage(self.robot.robot_usd, self.robot.robot_prim_path)
        else:
            add_reference_to_stage(self.robot.robot_usd, "/World")

        self.usd_objects["robot"] = XFormPrim(
            prim_path=self.robot.robot_prim_path,
            position=self.robot.init_position,
            orientation=self.robot.init_rotation,
        )

        # overwrite init joint position
        if len(self.robot.init_joint_position) != len(self.robot.init_joint_names):
            raise ValueError("robot init joint position and names length not match")

        # 
        self._play()

        # robot 补充设定 部分

        # solver 初始化部分
        self._init_solver(self.robot, enable_curobo=True)


    def run(self):
        
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

            self.world.step(render=False)
            self.world.render()

            step_index = self.world.current_time_step_index
            if self.articulation_controller is None:
                self.articulation_controller = self.articulation.get_articulation_controller()
            if step_index < 10:
                self.articulation._articulation_view.initialize()
                idx_list = [self.articulation.get_dof_index(x) for x in self.j_names]
                self.articulation.set_joint_positions(self.default_config, idx_list)

                self.articulation._articulation_view.set_max_efforts(
                    values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
                )
            if step_index < 20:
                continue

            if step_index == 50 or step_index % 1000 == 0.0:
                print("Updating world, reading w.r.t.", self.robot_prim_path)
                obstacles = self.usd_help.get_obstacles_from_stage(
                    only_paths=["/World"],
                    reference_prim_path=self.robot_prim_path,
                    ignore_substring=[
                        self.robot_prim_path,
                        "/World/target",
                        "/World/defaultGroundPlane",
                        "/curobo",
                    ],
                ).get_collision_check_world()
                print(len(obstacles.objects))

                self.motion_gen.update_world(obstacles)
                print("Updated World")

            
            # position and orientation of target virtual cube:
            cube_position, cube_orientation = self.target.get_world_pose()

            if past_pose is None:
                past_pose = cube_position
            if target_pose is None:
                target_pose = cube_position
            if target_orientation is None:
                target_orientation = cube_orientation
            if past_orientation is None:
                past_orientation = cube_orientation

            sim_js = self.articulation.get_joints_state()
            if sim_js is None:
                print("sim_js is None")
                continue
            
            if np.any(np.isnan(sim_js.positions)):
                log_error("isaac sim has returned NAN joint position values.")

            all_js_names = self.articulation.dof_names
            sim_js_names = []
            lock_idx = []
            sim_js_positions = []
            sim_js_velocities = []
            for idx, name in enumerate(all_js_names):
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
            cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

            if args.visualize_spheres and step_index % 2 == 0:
                sph_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

                if spheres is None:
                    spheres = []
                    # create spheres:

                    for si, s in enumerate(sph_list[0]):
                        sp = sphere.VisualSphere(
                            prim_path="/curobo/robot_sphere_" + str(si),
                            position=np.ravel(s.position),
                            radius=float(s.radius),
                            color=np.array([0, 0.8, 0.2]),
                        )
                        spheres.append(sp)
                else:
                    for si, s in enumerate(sph_list[0]):
                        if not np.isnan(s.position[0]):
                            spheres[si].set_world_pose(position=np.ravel(s.position))
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
                ee_translation_goal = cube_position
                ee_orientation_teleop_goal = cube_orientation

                # compute curobo solution:
                ik_goal = Pose(
                    position=self.tensor_args.to_device(ee_translation_goal),
                    quaternion=self.tensor_args.to_device(ee_orientation_teleop_goal),
                )
                self.plan_config.pose_cost_metric = pose_metric
                result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config)
                # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

                succ = result.success.item()  # ik_result.success.item()
                if num_targets == 1:
                    if args.constrain_grasp_approach:
                        pose_metric = PoseCostMetric.create_grasp_approach_metric()
                    if args.reach_partial_pose is not None:
                        reach_vec = self.motion_gen.tensor_args.to_device(args.reach_partial_pose)
                        pose_metric = PoseCostMetric(
                            reach_partial_pose=True, reach_vec_weight=reach_vec
                        )
                    if args.hold_partial_pose is not None:
                        hold_vec = self.motion_gen.tensor_args.to_device(args.hold_partial_pose)
                        pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
                if succ:
                    num_targets += 1
                    cmd_plan = result.get_interpolated_plan()
                    cmd_plan = self.motion_gen.get_full_js(cmd_plan)
                    # get only joint names that are in both:
                    idx_list = []
                    common_js_names = []
                    for x in sim_js_names:
                        if x in cmd_plan.joint_names:
                            idx_list.append(self.articulation.get_dof_index(x))
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
                    self.world.step(render=False)
                if cmd_idx >= len(cmd_plan.position):
                    cmd_idx = 0
                    cmd_plan = None
                    past_cmd = None

        simulation_app.close()


if __name__ == "__main__":
    runner = BasicRunner(CONFIG_FILE)
    runner.run()

