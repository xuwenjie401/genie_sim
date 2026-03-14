try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import argparse
import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np
import yaml

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="Standalone pick-pose filter and motion-gen debugger")
parser.add_argument("--headless_mode", type=str, default=None)
parser.add_argument("--robot_yaml", type=str, default="/home/agxi/RealityLab/genie_sim/unit_lab/configs/basic_test.yaml")
parser.add_argument(
    "--robot_usd",
    type=str,
    default="/home/agxi/Documents/assets/robots/urdf_ws/src/galbot_one_golf_description/sim_ready/galbot_one_golf/usdv2/galbot_fixed.usda",
)
parser.add_argument(
    "--scene_usd",
    type=str,
    default="/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/background/home_b/home_b_00.usda",
)
parser.add_argument(
    "--object_usd",
    type=str,
    default="/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/beverage_bottle/benchmark_beverage_bottle_008/Aligned.usda",
)
parser.add_argument(
    "--grasp_pkl",
    type=str,
    default="/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/interaction/benchmark_beverage_bottle_008/grasp_pose/grasp_pose.pkl",
)
parser.add_argument(
    "--obstacle_usd",
    type=str,
    default="/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/storage_box/benchmark_storage_box_000/Aligned.usda",
)
parser.add_argument("--physics_step", type=int, default=60)
parser.add_argument("--arm", type=str, default="left", choices=["left", "right"])
parser.add_argument("--robot_cfg_name", type=str, default="galbot")
parser.add_argument("--robot_root", type=str, default="/galbot_one_golf")
parser.add_argument("--object_path", type=str, default="/World/obstacle1")
parser.add_argument("--obstacle_path", type=str, default="/World/obstacle2")
parser.add_argument("--robot_pos", nargs=3, type=float, default=[2.15, 0.7757971635415469, 0.0])
parser.add_argument("--robot_quat", nargs=4, type=float, default=[1.0, 0.0, 0.0, 0.0])
parser.add_argument("--object_pos", nargs=3, type=float, default=[2.85, 0.85, 0.85])
parser.add_argument("--object_quat", nargs=4, type=float, default=[1.0, 0.0, 0.0, 0.0])
parser.add_argument("--obstacle_pos", nargs=3, type=float, default=[3.00, 0.60, 0.85])
parser.add_argument("--obstacle_quat", nargs=4, type=float, default=[1.0, 0.0, 0.0, 0.0])
parser.add_argument("--grasp_offset", type=float, default=0.03)
parser.add_argument("--grasp_lower_percentile", type=float, default=0.0)
parser.add_argument("--grasp_upper_percentile", type=float, default=100.0)
parser.add_argument("--disable_upside_down", action="store_true", default=True)
parser.add_argument("--humanlike_filter", action="store_true", default=False)
parser.add_argument("--set_grasp_vertical", action="store_true", default=False)
parser.add_argument("--set_grasp_pose_xy", action="store_true", default=False)
parser.add_argument("--flip_grasp", action="store_true", default=False)
parser.add_argument("--filter_up_axis", type=str, default="")
parser.add_argument("--filter_target_axis", type=str, default="+z")
parser.add_argument("--filter_angle_deg", type=float, default=45.0)
parser.add_argument("--downsample", type=int, default=300)
parser.add_argument("--max_draw", type=int, default=80)
parser.add_argument("--max_replay", type=int, default=24)
parser.add_argument("--grasp_thickness", type=float, default=0.003)
parser.add_argument("--finger_len", type=float, default=0.04)
parser.add_argument("--handle_len", type=float, default=0.08)
parser.add_argument("--render_dt", type=float, default=1.0 / 30.0)
args = parser.parse_args()


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

import carb
import carb.input
import omni.appwindow
import omni.usd
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, UsdGeom

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
collection_dir = os.path.join(root_dir, "source/data_collection")
if collection_dir not in sys.path:
    sys.path.append(collection_dir)

from helper import add_robot_to_scene
from source.data_collection.client.planner.func.common import (
    filter_grasp_pose_by_gripper_up_direction,
    filter_grasp_poses_with_humanlike_posture,
    random_downsample,
)
from source.data_collection.common.base_utils.transform_utils import (
    calculate_rotation_matrix2,
    mat2quat_wxyz,
    pose_from_position_quaternion,
    quat2mat_wxyz,
    rotate_along_axis,
)
CUROBO_BATCH_SIZE = 32


def remove_prim_if_exists(stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def set_display_color(prim, rgb) -> None:
    rgb = np.asarray(rgb, dtype=np.float32).reshape(3)
    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))])


def np_to_gf_matrix4d(T: np.ndarray) -> Gf.Matrix4d:
    M = np.asarray(T, dtype=np.float64).T
    return Gf.Matrix4d(
        M[0, 0], M[0, 1], M[0, 2], M[0, 3],
        M[1, 0], M[1, 1], M[1, 2], M[1, 3],
        M[2, 0], M[2, 1], M[2, 2], M[2, 3],
        M[3, 0], M[3, 1], M[3, 2], M[3, 3],
    )


def set_local_matrix(prim, T_local_4x4: np.ndarray) -> None:
    xformable = UsdGeom.Xformable(prim)
    transform_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            transform_op = op
            break
    if transform_op is None:
        transform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    transform_op.Set(np_to_gf_matrix4d(T_local_4x4))


def set_local_translate_scale(prim, t_xyz, s_xyz) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    t_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "translateLocal")
    t_op.Set(Gf.Vec3d(float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2])))
    s_op = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "scaleLocal")
    s_op.Set(Gf.Vec3d(float(s_xyz[0]), float(s_xyz[1]), float(s_xyz[2])))


def build_grasp_marker(stage, root_path: str, width: float, color, with_axes: bool = False) -> None:
    root = create_prim(root_path, prim_type="Xform")
    t = float(args.grasp_thickness)
    width = max(float(width), 1e-4)
    cube_specs = [
        ("handle", (-args.handle_len / 2.0, 0.0, 0.0), (args.handle_len, t, t)),
        ("bar", (0.0, 0.0, 0.0), (t, width, t)),
        ("finger_L", (args.finger_len / 2.0, -width / 2.0, 0.0), (args.finger_len, t, t)),
        ("finger_R", (args.finger_len / 2.0, width / 2.0, 0.0), (args.finger_len, t, t)),
    ]
    for name, translation, scale in cube_specs:
        prim = create_prim(f"{root_path}/{name}", prim_type="Cube")
        UsdGeom.Cube(prim).CreateSizeAttr().Set(1.0)
        set_local_translate_scale(prim, translation, scale)
        set_display_color(prim, color)
    if with_axes:
        axis_specs = [
            ("x_axis", (0.03, 0.0, 0.0), (0.06, 0.0016, 0.0016), np.array([1.0, 0.0, 0.0])),
            ("y_axis", (0.0, 0.03, 0.0), (0.0016, 0.06, 0.0016), np.array([0.0, 1.0, 0.0])),
            ("z_axis", (0.0, 0.0, 0.03), (0.0016, 0.0016, 0.06), np.array([0.0, 0.0, 1.0])),
        ]
        for name, translation, scale, axis_color in axis_specs:
            prim = create_prim(f"{root_path}/{name}", prim_type="Cube")
            UsdGeom.Cube(prim).CreateSizeAttr().Set(1.0)
            set_local_translate_scale(prim, translation, scale)
            set_display_color(prim, axis_color)
    return root


@dataclass
class FilterSnapshot:
    name: str
    kept_poses: np.ndarray
    kept_widths: np.ndarray
    dropped_poses: np.ndarray
    dropped_widths: np.ndarray
    note: str


@dataclass
class ReplayPose:
    rank: int
    world_pose: np.ndarray
    width: float
    joint_distance: float


class PickPoseDebugWorker:
    def __init__(self) -> None:
        self._keyboard_sub = None
        self.running = True
        self.mode = "filters"
        self.pending_next = False
        self.pending_prev = False
        self.pending_reset = False
        self.snapshots: list[FilterSnapshot] = []
        self.snapshot_index = 0
        self.replay_poses: list[ReplayPose] = []
        self.replay_index = 0
        self.cmd_plan = None
        self.cmd_idx = 0
        self.cmd_joint_indices = []
        self.waiting_after_replay = False
        self.initial_joint_positions = None
        self.initial_joint_velocities = None

        self.robot_gripper_2_grasp_gripper = self._get_gripper_remap()
        self._init_world()
        self._init_planner()
        self._set_debug_camera()
        self._warm_up()
        self._snapshot_robot_state()
        self._compute_debug_data()
        self._register_keyboard()
        self._render_current_snapshot()
        self._print_help()

    def _get_gripper_remap(self) -> np.ndarray:
        cfg = args.robot_cfg_name.lower()
        if "omnipicker" in cfg or "agile" in cfg:
            return np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
        if "galbot" in cfg:
            return np.eye(3, dtype=np.float64)
        return np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)

    def _init_world(self) -> None:
        setup_curobo_logger("warn")
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(1.0 / args.physics_step),
            rendering_dt=float(args.render_dt),
            device="cpu",
        )
        self.stage = self.world.stage
        add_reference_to_stage(args.scene_usd, "/World")
        add_reference_to_stage(args.object_usd, args.object_path)
        self.object_prim = XFormPrim(args.object_path, position=np.array(args.object_pos), orientation=np.array(args.object_quat))
        if args.obstacle_usd:
            add_reference_to_stage(args.obstacle_usd, args.obstacle_path)
            self.extra_obstacle = XFormPrim(
                args.obstacle_path,
                position=np.array(args.obstacle_pos),
                orientation=np.array(args.obstacle_quat),
            )
        else:
            self.extra_obstacle = None
        self.robot_yaml = yaml.safe_load(open(args.robot_yaml, "r"))
        self.robot_cfg = self.robot_yaml["robot_cfg"]
        self.robot_cfg["kinematics"]["usd_path"] = args.robot_usd
        self.robot_cfg["kinematics"]["isaac_usd_path"] = args.robot_usd
        self.articulation, _ = add_robot_to_scene(
            self.robot_cfg,
            self.world,
            load_from_usd=True,
            robot_name=self.robot_yaml["robot"]["robot_name"],
            position=np.array(args.robot_pos),
            initialize_world=True,
        )
        self.robot_root = XFormPrim(args.robot_root)
        self.world.play()

    def _init_planner(self) -> None:
        self.tensor_args = TensorDeviceType()
        self.motion_gen = MotionGen(
            MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                WorldConfig(),
                self.tensor_args,
                collision_checker_type=CollisionCheckerType.MESH,
                use_cuda_graph=False,
                num_trajopt_seeds=4,
                num_graph_seeds=4,
                interpolation_dt=0.02,
                collision_cache={"obb": 16, "mesh": 32},
                optimize_dt=True,
                trajopt_tsteps=32,
                collision_activation_distance=0.01,
            )
        )
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=4,
            enable_finetune_trajopt=True,
            time_dilation_factor=0.5,
        )
        self.lock_joints = self.robot_cfg["kinematics"]["lock_joints"]
        self.lock_js_names = list(self.lock_joints.keys()) if self.lock_joints else []
        self.plan_joint_names = list(self.motion_gen.kinematics.joint_names)
        self.usd_help = UsdHelper()
        self.usd_help.load_stage(self.stage)
        self._update_world()

    def _set_debug_camera(self) -> None:
        if args.headless_mode is not None:
            return
        camera_state = ViewportCameraState("/OmniverseKit_Persp")
        camera_state.set_position_world(Gf.Vec3d(2.65, 2.4, 1.74), True)
        init_position = np.asarray(args.robot_pos, dtype=np.float64)
        camera_state.set_target_world(
            Gf.Vec3d(init_position[0] + 0.5, init_position[1], init_position[2] + 0.8),
            True,
        )

    def _warm_up(self) -> None:
        for _ in range(30):
            self.world.step(render=False)
            self.world.render()

    def _snapshot_robot_state(self) -> None:
        raw_positions = self.articulation.get_joint_positions()
        raw_velocities = self.articulation.get_joint_velocities()
        if raw_positions is not None:
            self.initial_joint_positions = np.asarray(raw_positions, dtype=np.float64).copy()
        if raw_velocities is not None:
            self.initial_joint_velocities = np.asarray(raw_velocities, dtype=np.float64).copy()

    def _get_world_to_robot(self) -> np.ndarray:
        pos, quat = self.robot_root.get_world_pose()
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.asarray(pos, dtype=np.float64)
        T[:3, :3] = quat2mat_wxyz(np.asarray(quat, dtype=np.float64))
        return np.linalg.inv(T)

    def _get_object_pose(self) -> np.ndarray:
        pos, quat = self.object_prim.get_world_pose()
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.asarray(pos, dtype=np.float64)
        T[:3, :3] = quat2mat_wxyz(np.asarray(quat, dtype=np.float64))
        return T

    def _get_current_joint_state(self) -> JointState:
        sim_js = self.articulation.get_joints_state()
        sim_js_names = []
        sim_js_positions = []
        sim_js_velocities = []
        for idx, name in enumerate(self.articulation.dof_names):
            if name in self.lock_js_names:
                continue
            sim_js_names.append(name)
            sim_js_positions.append(sim_js.positions[idx])
            sim_js_velocities.append(sim_js.velocities[idx])
        cu_js = JointState(
            position=self.tensor_args.to_device(np.asarray(sim_js_positions, dtype=np.float32)),
            velocity=self.tensor_args.to_device(np.asarray(sim_js_velocities, dtype=np.float32)) * 0.0,
            acceleration=self.tensor_args.to_device(np.asarray(sim_js_velocities, dtype=np.float32)) * 0.0,
            jerk=self.tensor_args.to_device(np.asarray(sim_js_velocities, dtype=np.float32)) * 0.0,
            joint_names=sim_js_names,
        )
        return cu_js.get_ordered_joint_state(self.plan_joint_names)

    def _split_world_poses_to_local_pose(self, world_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T_world_to_robot = self._get_world_to_robot()
        local_poses = T_world_to_robot[np.newaxis, ...] @ world_poses
        positions = local_poses[:, :3, 3]
        quats = np.asarray([mat2quat_wxyz(T[:3, :3]) for T in local_poses], dtype=np.float64)
        return positions, quats

    def _update_world(self) -> None:
        obstacles = self.usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path=args.robot_root,
            ignore_substring=[args.robot_root, "/World/PickPoseDebug", "/World/defaultGroundPlane"],
        )
        self.motion_gen.update_world(obstacles.get_collision_check_world())

    def _set_empty_world(self) -> None:
        self.motion_gen.clear_world_cache()
        self.motion_gen.update_world(WorldConfig().get_collision_check_world())

    def _load_grasp_pkl(self) -> tuple[np.ndarray, np.ndarray]:
        with open(args.grasp_pkl, "rb") as f:
            data = pickle.load(f)
        grasps = np.asarray(data["grasp_pose"], dtype=np.float64)
        widths = np.asarray(data["width"], dtype=np.float64)
        return grasps, widths

    def _record_snapshot(
        self,
        name: str,
        before_poses: np.ndarray,
        before_widths: np.ndarray,
        after_poses: np.ndarray,
        after_widths: np.ndarray,
        mask: np.ndarray | None,
        note: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if mask is None:
            mask = np.ones(len(before_poses), dtype=bool)
        dropped_poses = before_poses[~mask] if len(before_poses) else np.empty((0, 4, 4), dtype=np.float64)
        dropped_widths = before_widths[~mask] if len(before_widths) else np.empty((0,), dtype=np.float64)
        self.snapshots.append(
            FilterSnapshot(
                name=name,
                kept_poses=after_poses.copy(),
                kept_widths=after_widths.copy(),
                dropped_poses=dropped_poses.copy(),
                dropped_widths=dropped_widths.copy(),
                note=note,
            )
        )
        return after_poses, after_widths

    def _compute_debug_data(self) -> None:
        object_pose = self._get_object_pose()
        grasp_poses_canonical, grasp_widths = self._load_grasp_pkl()

        world_poses = object_pose[np.newaxis, ...] @ grasp_poses_canonical
        self.snapshots.append(
            FilterSnapshot(
                name="raw_world",
                kept_poses=world_poses.copy(),
                kept_widths=grasp_widths.copy(),
                dropped_poses=np.empty((0, 4, 4), dtype=np.float64),
                dropped_widths=np.empty((0,), dtype=np.float64),
                note=f"raw grasps from {os.path.basename(args.grasp_pkl)}",
            )
        )

        if args.set_grasp_pose_xy:
            grasp_poses_canonical[:, 0, 3] = 0.0
            grasp_poses_canonical[:, 2, 3] = 0.0

        before = grasp_poses_canonical.copy()
        before_w = grasp_widths.copy()
        z_values = grasp_poses_canonical[:, 1, 3]
        z_lower_threshold = np.percentile(z_values, args.grasp_lower_percentile)
        z_upper_threshold = np.percentile(z_values, args.grasp_upper_percentile)
        mask = (z_values <= z_upper_threshold) & (z_values >= z_lower_threshold)
        grasp_poses_canonical = grasp_poses_canonical[mask]
        grasp_widths = grasp_widths[mask]
        self._record_snapshot(
            "percentile_crop",
            object_pose[np.newaxis, ...] @ before,
            before_w,
            object_pose[np.newaxis, ...] @ grasp_poses_canonical,
            grasp_widths,
            mask,
            f"keep y-translation percentile [{args.grasp_lower_percentile}, {args.grasp_upper_percentile}]",
        )

        grasp_poses_canonical[:, :3, :3] = (
            grasp_poses_canonical[:, :3, :3] @ self.robot_gripper_2_grasp_gripper[np.newaxis, ...]
        )
        if args.set_grasp_vertical:
            target_y = np.array([0.0, 1.0, 0.0]) if args.arm == "right" else np.array([0.0, -1.0, 0.0])
            for i in range(grasp_poses_canonical.shape[0]):
                local_y = grasp_poses_canonical[i][:3, 1]
                rot = calculate_rotation_matrix2(local_y, target_y)
                grasp_poses_canonical[i][:3, :3] = rot @ grasp_poses_canonical[i][:3, :3]
        if args.flip_grasp and len(grasp_poses_canonical) > 0:
            flips = [rotate_along_axis(pose, 180, "z", use_local=True) for pose in grasp_poses_canonical]
            grasp_poses_canonical = np.concatenate([grasp_poses_canonical, np.stack(flips)], axis=0)
            grasp_widths = np.concatenate([grasp_widths, grasp_widths], axis=0)
        self.snapshots.append(
            FilterSnapshot(
                name="canonical_remap",
                kept_poses=object_pose[np.newaxis, ...] @ grasp_poses_canonical,
                kept_widths=grasp_widths.copy(),
                dropped_poses=np.empty((0, 4, 4), dtype=np.float64),
                dropped_widths=np.empty((0,), dtype=np.float64),
                note="applied robot grasp-frame remap and optional vertical/flip edits",
            )
        )

        grasp_poses = object_pose[np.newaxis, ...] @ grasp_poses_canonical
        if args.filter_up_axis:
            before = grasp_poses.copy()
            before_w = grasp_widths.copy()
            grasp_poses, grasp_widths, mask = filter_grasp_pose_by_gripper_up_direction(
                {
                    "gripper_up_axis": args.filter_up_axis,
                    "targrt_direction": args.filter_target_axis,
                    "threshold": np.deg2rad(args.filter_angle_deg),
                },
                grasp_poses,
                grasp_widths,
            )
            self._record_snapshot(
                "gripper_up_filter",
                before,
                before_w,
                grasp_poses,
                grasp_widths,
                mask,
                f"{args.filter_up_axis} aligned with {args.filter_target_axis} within {args.filter_angle_deg} deg",
            )

        if args.humanlike_filter and len(grasp_poses) > 0:
            before = grasp_poses.copy()
            before_w = grasp_widths.copy()
            grasp_poses, grasp_widths, mask = filter_grasp_poses_with_humanlike_posture(grasp_poses, grasp_widths)
            self._record_snapshot(
                "humanlike_filter",
                before,
                before_w,
                grasp_poses,
                grasp_widths,
                mask,
                "kept poses that match the humanlike posture heuristic",
            )

        if args.disable_upside_down and len(grasp_poses) > 0:
            before = grasp_poses.copy()
            before_w = grasp_widths.copy()
            if "omnipicker" in args.robot_cfg_name:
                mask = grasp_poses[:, 2, 1] < 0.0 if args.arm == "left" else grasp_poses[:, 2, 1] > 0.0
            elif "agile" in args.robot_cfg_name.lower():
                mask = grasp_poses[:, 2, 1] > 0.0
            elif "galbot" in args.robot_cfg_name.lower():
                mask = grasp_poses[:, 2, 2] > 0.0
            else:
                mask = grasp_poses[:, 2, 0] > 0.0
            grasp_poses = grasp_poses[mask]
            grasp_widths = grasp_widths[mask]
            self._record_snapshot(
                "upright_filter",
                before,
                before_w,
                grasp_poses,
                grasp_widths,
                mask,
                "removed upside-down grasp orientations",
            )

        if len(grasp_poses) == 0:
            return

        before = grasp_poses.copy()
        before_w = grasp_widths.copy()
        grasp_poses, random_indices = random_downsample(grasp_poses, args.downsample, False)
        if random_indices is None:
            mask = np.ones(len(before), dtype=bool)
        else:
            mask = np.zeros(len(before), dtype=bool)
            mask[random_indices] = True
            grasp_widths = grasp_widths[random_indices]
        self._record_snapshot(
            "downsample",
            before,
            before_w,
            grasp_poses,
            grasp_widths,
            mask,
            f"downsampled to at most {args.downsample} grasps",
        )

        before = grasp_poses.copy()
        before_w = grasp_widths.copy()
        grasp_rotate = grasp_poses.copy()
        grasp_rotate[:, :3, 3] = 0.0
        transport_vector = grasp_rotate @ np.array([0.0, 0.0, 1.0, 0.0])[:, np.newaxis]
        transport_vector = transport_vector[:, :3, 0]
        transport_vector = transport_vector / np.linalg.norm(transport_vector, axis=1, keepdims=True)
        grasp_poses[:, :3, 3] = grasp_poses[:, :3, 3] + transport_vector * args.grasp_offset
        self._record_snapshot(
            "grasp_offset",
            before,
            before_w,
            grasp_poses,
            grasp_widths,
            np.ones(len(before), dtype=bool),
            f"translated along local +z by {args.grasp_offset:.3f} m",
        )

        ik_success, ik_joint_positions, ik_joint_names = self._solve_ik_batch(grasp_poses)
        self._record_snapshot(
            "ik_check",
            grasp_poses,
            grasp_widths,
            grasp_poses[ik_success],
            grasp_widths[ik_success],
            ik_success,
            "simple IK batch pass/fail before motion planning",
        )
        if not np.any(ik_success):
            return

        ik_world_poses = grasp_poses[ik_success]
        ik_widths = grasp_widths[ik_success]
        ik_joint_positions = ik_joint_positions[ik_success]
        ik_joint_names = ik_joint_names[ik_success]
        current_js = self._get_current_joint_state()
        current_joint = current_js.position.detach().cpu().numpy()
        costs = []
        for pose_idx in range(len(ik_world_poses)):
            target = self._extract_joint_vector(ik_joint_positions[pose_idx], ik_joint_names[pose_idx], self.plan_joint_names)
            costs.append(float(np.linalg.norm(target - current_joint)))
        order = np.argsort(np.asarray(costs, dtype=np.float64))
        sorted_poses = ik_world_poses[order][: args.max_replay]
        sorted_widths = ik_widths[order][: args.max_replay]
        sorted_costs = np.asarray(costs, dtype=np.float64)[order][: args.max_replay]
        self.snapshots.append(
            FilterSnapshot(
                name="sorted_replay_set",
                kept_poses=sorted_poses.copy(),
                kept_widths=sorted_widths.copy(),
                dropped_poses=np.empty((0, 4, 4), dtype=np.float64),
                dropped_widths=np.empty((0,), dtype=np.float64),
                note=f"sorted IK-passed poses by current-joint distance, truncated to {len(sorted_poses)} replay poses",
            )
        )
        self.replay_poses = [
            ReplayPose(rank=i, world_pose=sorted_poses[i], width=float(sorted_widths[i]), joint_distance=float(sorted_costs[i]))
            for i in range(len(sorted_poses))
        ]

    def _extract_joint_vector(self, joint_position, joint_names, target_joint_names) -> np.ndarray:
        ordered = []
        joint_names = list(joint_names)
        for name in target_joint_names:
            ordered.append(float(joint_position[joint_names.index(name)]))
        return np.asarray(ordered, dtype=np.float64)

    def _solve_ik_batch(self, world_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(world_poses) == 0:
            return (
                np.zeros((0,), dtype=bool),
                np.empty((0, len(self.plan_joint_names)), dtype=np.float64),
                np.empty((0, len(self.plan_joint_names)), dtype=object),
            )
        # Match the main pipeline's loose "Simple" IK stage more closely:
        # do not use the scene collision world for this initial grasp filter.
        self._set_empty_world()
        positions, quats = self._split_world_poses_to_local_pose(world_poses)
        successes = []
        joint_positions = []
        joint_names = []
        for start in range(0, len(world_poses), CUROBO_BATCH_SIZE):
            end = min(start + CUROBO_BATCH_SIZE, len(world_poses))
            batch_pos = positions[start:end]
            batch_quat = quats[start:end]
            goal_pose = Pose(
                position=self.tensor_args.to_device(batch_pos),
                quaternion=self.tensor_args.to_device(batch_quat),
                batch=len(batch_pos),
            )
            result = self.motion_gen.ik_solver.solve_batch(goal_pose)
            batch_success = result.success.detach().cpu().numpy().astype(bool).reshape(-1)
            successes.append(batch_success)
            for idx in range(len(batch_pos)):
                if batch_success[idx]:
                    joint_positions.append(result.js_solution.position[idx][0].detach().cpu().numpy().copy())
                    joint_names.append(np.asarray(result.js_solution.joint_names, dtype=object))
                else:
                    joint_positions.append(np.zeros((len(result.js_solution.joint_names),), dtype=np.float64))
                    joint_names.append(np.asarray(result.js_solution.joint_names, dtype=object))
        self._update_world()
        return np.concatenate(successes, axis=0), np.asarray(joint_positions, dtype=np.float64), np.asarray(joint_names, dtype=object)

    def _render_current_snapshot(self) -> None:
        remove_prim_if_exists(self.stage, "/World/PickPoseDebug")
        root = create_prim("/World/PickPoseDebug", prim_type="Xform")
        if self.mode == "filters":
            if not self.snapshots:
                return
            snap = self.snapshots[self.snapshot_index]
            kept_limit = min(len(snap.kept_poses), args.max_draw)
            dropped_limit = min(len(snap.dropped_poses), args.max_draw)
            for i in range(kept_limit):
                path = f"{root.GetPath()}/kept_{i:04d}"
                prim = build_grasp_marker(self.stage, path, snap.kept_widths[i], np.array([0.1, 0.9, 0.2]), with_axes=False)
                set_local_matrix(prim, snap.kept_poses[i])
            for i in range(dropped_limit):
                path = f"{root.GetPath()}/dropped_{i:04d}"
                prim = build_grasp_marker(self.stage, path, snap.dropped_widths[i], np.array([0.9, 0.15, 0.15]), with_axes=False)
                set_local_matrix(prim, snap.dropped_poses[i])
            print(
                f"[FILTER {self.snapshot_index + 1}/{len(self.snapshots)}] {snap.name} | "
                f"kept={len(snap.kept_poses)} dropped={len(snap.dropped_poses)} | {snap.note}"
            )
        else:
            if not self.replay_poses:
                return
            for pose in self.replay_poses:
                color = np.array([0.15, 0.75, 0.95])
                if pose.rank == self.replay_index:
                    color = np.array([1.0, 0.75, 0.1])
                path = f"{root.GetPath()}/replay_{pose.rank:04d}"
                prim = build_grasp_marker(self.stage, path, pose.width, color, with_axes=pose.rank == self.replay_index)
                set_local_matrix(prim, pose.world_pose)
            pose = self.replay_poses[self.replay_index]
            print(
                f"[REPLAY {self.replay_index + 1}/{len(self.replay_poses)}] "
                f"joint_distance={pose.joint_distance:.4f} | "
                f"{'waiting for next' if self.waiting_after_replay else 'ready to run'}"
            )
        for _ in range(2):
            self.world.step(render=False)
            self.world.render()

    def _reset_robot(self) -> None:
        if self.initial_joint_positions is None:
            return
        self.cmd_plan = None
        self.cmd_idx = 0
        self.articulation.set_joint_positions(self.initial_joint_positions)
        if self.initial_joint_velocities is not None:
            self.articulation.set_joint_velocities(np.zeros_like(self.initial_joint_velocities))
        for _ in range(5):
            self.world.step(render=False)
            self.world.render()

    def _plan_replay_pose(self, replay_pose: ReplayPose) -> bool:
        self._reset_robot()
        self._update_world()
        local_pos, local_quat = self._split_world_poses_to_local_pose(replay_pose.world_pose[np.newaxis, ...])
        cu_js = self._get_current_joint_state()
        goal_pose = Pose(
            position=self.tensor_args.to_device(local_pos[0]),
            quaternion=self.tensor_args.to_device(local_quat[0]),
        )
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), goal_pose, self.plan_config)
        if not result.success.item():
            print(f"[PLAN] rank={replay_pose.rank + 1} failed: {result.status}")
            self.waiting_after_replay = True
            return False
        plan = result.get_interpolated_plan()
        plan = self.motion_gen.get_full_js(plan)
        common_js_names = []
        idx_list = []
        for name in self.articulation.dof_names:
            if name in plan.joint_names:
                idx_list.append(self.articulation.get_dof_index(name))
                common_js_names.append(name)
        self.cmd_plan = plan.get_ordered_joint_state(common_js_names)
        self.cmd_joint_indices = idx_list
        self.cmd_idx = 0
        self.waiting_after_replay = False
        print(f"[PLAN] rank={replay_pose.rank + 1} planned with {len(self.cmd_plan.position)} steps")
        return True

    def _tick_plan(self) -> None:
        if self.cmd_plan is None:
            return
        cmd_state = self.cmd_plan[self.cmd_idx]
        action = ArticulationAction(
            joint_positions=cmd_state.position.detach().cpu().numpy(),
            joint_velocities=cmd_state.velocity.detach().cpu().numpy(),
            joint_indices=self.cmd_joint_indices,
        )
        self.articulation.apply_action(action)
        self.cmd_idx += 1
        if self.cmd_idx >= len(self.cmd_plan.position):
            self.cmd_plan = None
            self.cmd_idx = 0
            self.waiting_after_replay = True
            print("[PLAN] replay finished. Press N for next pose or R to reset replay.")

    def _advance_filter(self, delta: int) -> None:
        self.snapshot_index = int(np.clip(self.snapshot_index + delta, 0, max(len(self.snapshots) - 1, 0)))
        self._render_current_snapshot()

    def _next_step(self) -> None:
        if self.mode == "filters":
            if self.snapshot_index < len(self.snapshots) - 1:
                self.snapshot_index += 1
                self._render_current_snapshot()
                return
            self.mode = "replay"
            self.replay_index = 0
            self.waiting_after_replay = False
            self._render_current_snapshot()
            print("[MODE] switched to replay mode. Press N to plan/run current ranked pose.")
            return

        if not self.replay_poses:
            print("[REPLAY] no IK-passed poses available")
            return
        if self.cmd_plan is not None:
            return
        if self.waiting_after_replay:
            self.replay_index += 1
            if self.replay_index >= len(self.replay_poses):
                self.replay_index = 0
            self.waiting_after_replay = False
            self._reset_robot()
            self._render_current_snapshot()
            return
        self._plan_replay_pose(self.replay_poses[self.replay_index])

    def _reset_view(self) -> None:
        self._reset_robot()
        self.mode = "filters"
        self.snapshot_index = 0
        self.replay_index = 0
        self.waiting_after_replay = False
        self._render_current_snapshot()
        print("[RESET] back to first filter snapshot")

    def _register_keyboard(self) -> None:
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        iface = carb.input.acquire_input_interface()
        key = carb.input.KeyboardInput

        def on_key(event, *args_unused):
            if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                return True
            if event.input == key.N:
                self.pending_next = True
            elif event.input == key.P:
                self.pending_prev = True
            elif event.input == key.R:
                self.pending_reset = True
            elif event.input == key.ESCAPE:
                self.running = False
            elif event.input == key.H:
                self._print_help()
            return True

        self._keyboard_sub = iface.subscribe_to_keyboard_events(keyboard, on_key)

    def _print_help(self) -> None:
        print("=" * 72)
        print("Pick pose filter / motion-gen debugger")
        print("N   : next filter snapshot, or run/advance replay pose")
        print("P   : previous filter snapshot")
        print("R   : reset to first filter snapshot and reset robot")
        print("H   : print help")
        print("Esc : quit")
        print("-" * 72)
        print("Flow:")
        print("  1. Inspect kept vs dropped grasps for each filter stage")
        print("  2. After final filter snapshot, N switches to replay mode")
        print("  3. In replay mode, N plans/runs current pose; after finish, N selects next pose")
        print("=" * 72)

    def run(self) -> None:
        while simulation_app.is_running() and self.running:
            self.world.step(render=False)
            self._tick_plan()
            self.world.render()
            if self.pending_next:
                self.pending_next = False
                self._next_step()
            if self.pending_prev:
                self.pending_prev = False
                if self.mode == "filters":
                    self._advance_filter(-1)
            if self.pending_reset:
                self.pending_reset = False
                self._reset_view()
        simulation_app.close()


if __name__ == "__main__":
    worker = PickPoseDebugWorker()
    worker.run()
