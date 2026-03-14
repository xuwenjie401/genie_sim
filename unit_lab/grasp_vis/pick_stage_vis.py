"""
Pick-stage visualization unit test.

Runs the galbot right-arm grasp filter pipeline step-by-step inside Isaac Sim,
drawing coloured "]–" grasp shapes at each filter stage.  After all filters
pass, plays CuRobo motion trajectories for each sorted pose interactively.

Controls (viewport must have focus):
    N   – advance to next filter step  /  execute next sorted pose
    R   – reset robot to home and re-plan current sorted pose
    ESC – quit

Grasp drawing convention (+Z = approach, fingers point +Z; this matches the
post-gripper-transform world-frame poses used by the pipeline):
    handle  → –Z side of the origin
    cross bar → along ±X
    fingers → +Z side at ±X/2
"""

# ──────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP  (must come before any isaacsim import)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import torch
import time

_cuda_warmup = torch.zeros(4, device="cuda:0")

import argparse

parser = argparse.ArgumentParser(description="Pick-stage grasp filter visualizer")
parser.add_argument("--headless_mode", type=str, default=None,
                    help="'native' or 'websocket' for headless")
parser.add_argument("--debug", action="store_true", default=False)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({
    "headless": args.headless_mode is not None,
    "width": "1920",
    "height": "1080",
})
simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
simulation_app._carb_settings.set("/omni/replicator/asyncRendering", False)
simulation_app._carb_settings.set("/app/asyncRendering", False)

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD IMPORTS  (after SimulationApp)
# ──────────────────────────────────────────────────────────────────────────────
import json
import os
import pickle
import queue
import sys
import threading
import yaml

import carb
import carb.input
import numpy as np
import omni
import omni.appwindow
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import cylinder
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.prims import get_prim_at_path  # noqa: F401
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction  # noqa: F401
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

# ──────────────────────────────────────────────────────────────────────────────
# PROJECT SYS-PATH  (unit_lab/grasp_vis/ → genie_sim root + data_collection)
# ──────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # genie_sim/
_COLLECTION = os.path.join(_ROOT, "source/data_collection")
for _d in (_ROOT, _COLLECTION):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from source.data_collection.common.base_utils import transform_utils        # noqa: E402
from source.data_collection.server.command_enum import Command              # noqa: E402
from source.data_collection.server.robot import RobotCfg                   # noqa: E402
from source.data_collection.server.ui_builder import UIBuilder              # noqa: E402
from source.data_collection.server.utils import (                           # noqa: E402
    batch_matrices_to_quaternions_scipy_w_first,
)
from source.data_collection.client.planner.func.common import random_downsample  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit here to change objects / arm / offsets)
# ──────────────────────────────────────────────────────────────────────────────
SIM_ASSETS_ROOT = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets"
ROBOT_JSON      = "galbot_fixed_dual.json"
SCENE_USD       = "background/home_b/home_b_00.usda"
ROBOT_POSITION  = [2.05, 0.7757971635415469, 0.0]
ROBOT_ROTATION  = [1.0, 0.0, 0.0, 0.0]

# Pick object
OBJECT_DATA_DIR = "objects/benchmark/beverage_bottle/benchmark_beverage_bottle_001"
OBJECT_USD_PATH = os.path.join(SIM_ASSETS_ROOT, OBJECT_DATA_DIR, "Aligned.usda")
GRASP_PKL_PATH  = os.path.join(SIM_ASSETS_ROOT,
                                "interaction/benchmark_beverage_bottle_001",
                                "grasp_pose/grasp_pose.pkl")
OBJECT_WORLD_POS = np.array([2.85, 0.75, 0.88])  # fixed pose for unit test

# Arm
ARM         = "right"
ARM_IS_RIGHT = True

# Filter params (from galbot task JSON)
GRASP_LOWER_PERCENTILE = 0
GRASP_UPPER_PERCENTILE = 100
DISABLE_UPSIDE_DOWN    = True
GRASP_OFFSET           = 0.0   # metres along approach (+Z of pose)
MAX_SIMPLE_IK          = 300
MAX_CUROBO_IK          = 100

# robot_gripper_2_grasp_gripper for galbot (omni_robot.py line 72)
ROBOT_GRIPPER_2_GRASP_GRIPPER = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)

TASK_JSON = os.path.join(
    _ROOT,
    "source/data_collection/tasks/geniesim_2025"
    "/place_object_into_box_of_specific_color/galbot"
    "/place_object_into_box_of_specific_color_blue_galbot.json",
)

# Colours (RGB float)
C_GREEN  = np.array([0.0, 1.0,  0.0])   # kept / passed
C_ORANGE = np.array([1.0, 0.3,  0.0])   # dropped
C_RED    = np.array([1.0, 0.0,  0.0])   # IK fail
C_BLUE   = np.array([0.0, 0.5,  1.0])   # CuRobo IK pass
C_YELLOW = np.array([1.0, 1.0,  0.0])   # current motion-gen target

# ──────────────────────────────────────────────────────────────────────────────
# USD PRIMITIVE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _remove_prim(stage, path: str) -> None:
    p = stage.GetPrimAtPath(path)
    if p and p.IsValid():
        stage.RemovePrim(path)


def _set_color(prim, rgb: np.ndarray) -> None:
    c = np.asarray(rgb, dtype=np.float32)
    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set(
        [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2]))]
    )


def _np_to_gf_matrix4d(T: np.ndarray) -> Gf.Matrix4d:
    M = np.asarray(T, dtype=np.float64).T.flatten().tolist()
    return Gf.Matrix4d(*M)


def _set_xform_matrix(prim, T: np.ndarray) -> None:
    xf = UsdGeom.Xformable(prim)
    op = next(
        (o for o in xf.GetOrderedXformOps()
         if o.GetOpType() == UsdGeom.XformOp.TypeTransform),
        None,
    )
    if op is None:
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(_np_to_gf_matrix4d(T))


def _set_translate_scale(prim, t, s) -> None:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    ot = xf.AddXformOp(UsdGeom.XformOp.TypeTranslate,
                       UsdGeom.XformOp.PrecisionDouble, opSuffix="t")
    ot.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
    os_ = xf.AddXformOp(UsdGeom.XformOp.TypeScale,
                        UsdGeom.XformOp.PrecisionDouble, opSuffix="s")
    os_.Set(Gf.Vec3d(float(s[0]), float(s[1]), float(s[2])))


# ──────────────────────────────────────────────────────────────────────────────
# GRASP DRAWING
# Two conventions because the pkl canonical frame and the robot gripper frame differ:
#   _GRASP_SPECS_X  : canonical pkl  — +X = approach, +Y = width, +Z = top
#   _GRASP_SPECS_Z  : robot gripper  — +Z = approach, +X = width, +Y = top
#                     (used after applying ROBOT_GRIPPER_2_GRASP_GRIPPER)
# ──────────────────────────────────────────────────────────────────────────────
_FINGER_LEN  = 0.05
_HANDLE_LEN  = 0.04
_GRIPPER_W   = 0.07
_THICKNESS   = 0.005

# canonical pkl: +X = approach  (matches grasp_viser_codex.py convention)
_GRASP_SPECS_X = [
    ("h",  (-_HANDLE_LEN / 2,  0.0,           0.0),  (_HANDLE_LEN, _THICKNESS,  _THICKNESS)),
    ("b",  (0.0,                0.0,           0.0),  (_THICKNESS,  _GRIPPER_W,  _THICKNESS)),
    ("fl", ( _FINGER_LEN / 2, -_GRIPPER_W / 2, 0.0), (_FINGER_LEN, _THICKNESS,  _THICKNESS)),
    ("fr", ( _FINGER_LEN / 2,  _GRIPPER_W / 2, 0.0), (_FINGER_LEN, _THICKNESS,  _THICKNESS)),
]

# robot gripper frame: +Z = approach  (after applying ROBOT_GRIPPER_2_GRASP_GRIPPER)
_GRASP_SPECS_Z = [
    ("h",  (0.0,           0.0,  -_HANDLE_LEN / 2),  (_THICKNESS,  _THICKNESS, _HANDLE_LEN)),
    ("b",  (0.0,           0.0,   0.0),               (_GRIPPER_W,  _THICKNESS, _THICKNESS)),
    ("fl", (-_GRIPPER_W/2, 0.0,   _FINGER_LEN / 2),  (_THICKNESS,  _THICKNESS, _FINGER_LEN)),
    ("fr", ( _GRIPPER_W/2, 0.0,   _FINGER_LEN / 2),  (_THICKNESS,  _THICKNESS, _FINGER_LEN)),
]


def _draw_single_grasp(stage, T: np.ndarray, color: np.ndarray, path: str,
                       specs=None) -> None:
    """Draw one grasp ]-shape at USD path."""
    if specs is None:
        specs = _GRASP_SPECS_Z
    _remove_prim(stage, path)
    root = stage.DefinePrim(path, "Xform")
    _set_xform_matrix(root, T)
    for nm, t, s in specs:
        p = stage.DefinePrim(f"{path}/{nm}", "Cube")
        UsdGeom.Cube(p).CreateSizeAttr().Set(1.0)
        _set_translate_scale(p, t, s)
        _set_color(p, color)


def draw_grasps(T_batch: np.ndarray, color: np.ndarray,
                container_path: str, specs=None) -> None:
    """Draw N grasps under a single USD Xform container."""
    if specs is None:
        specs = _GRASP_SPECS_Z
    stage = omni.usd.get_context().get_stage()
    _remove_prim(stage, container_path)
    stage.DefinePrim(container_path, "Xform")
    for i, T in enumerate(T_batch):
        _draw_single_grasp(stage, T, color, f"{container_path}/g{i:04d}", specs)


def _show_filter(stage, poses_all: np.ndarray, mask: np.ndarray,
                 container: str, step: int, label: str,
                 c_keep=None, c_drop=None, specs=None) -> None:
    """Draw kept (c_keep) and dropped (c_drop) grasps for a filter step."""
    if c_keep is None:
        c_keep = C_GREEN
    if c_drop is None:
        c_drop = C_ORANGE
    kept    = poses_all[mask]
    dropped = poses_all[~mask]
    _remove_prim(stage, container)
    stage.DefinePrim(container, "Xform")
    if dropped.shape[0]:
        draw_grasps(dropped, c_drop, f"{container}/dropped", specs)
    if kept.shape[0]:
        draw_grasps(kept,    c_keep, f"{container}/kept",    specs)
    print(f"  [Step {step}] {label}: {kept.shape[0]} kept, "
          f"{dropped.shape[0]} dropped")


# ──────────────────────────────────────────────────────────────────────────────
# SORTING HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _sort_by_joint_dist(joint_dicts: list, articulation,
                        arm_joint_names: list) -> np.ndarray:
    """Sort IK solutions by L2 distance to current robot joint positions."""
    if not joint_dicts or not arm_joint_names:
        return np.arange(len(joint_dicts))

    all_names = list(articulation.dof_names)
    cur_pos   = articulation.get_joint_positions()

    def _cur(name):
        return cur_pos[all_names.index(name)] if name in all_names else 0.0

    cur_arm = np.array([_cur(n) for n in arm_joint_names])
    tgt_arm = np.array([[jd.get(n, 0.0) for n in arm_joint_names]
                        for jd in joint_dicts])
    dists = np.linalg.norm(tgt_arm - cur_arm[np.newaxis, :], axis=1)
    return np.argsort(dists)


# ──────────────────────────────────────────────────────────────────────────────
# MINIMAL COMMAND CONTROLLER
# Supports INIT_ROBOT / GET_IK_STATUS / LINEAR_MOVE via blocking_start_server.
# ──────────────────────────────────────────────────────────────────────────────

class PickController:
    """Thin in-process command controller.  No ROS2 dependency."""

    def __init__(self, ui_builder: UIBuilder):
        self.ui_builder = ui_builder
        # Command-queue state
        self.data:          object   = None
        self.Command:       int      = 0
        self.data_to_send:  object   = None
        self.condition = threading.Condition()
        self.result_queue: queue.Queue = queue.Queue()
        # Motion state
        self.target_position = np.zeros(3)
        self.target_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.motion_run_ratio = 1.0
        # Robot state (filled after INIT_ROBOT)
        self.usd_objects:     dict = {}
        self.robot_cfg:       RobotCfg | None = None
        self.robot_init_position = np.zeros(3)
        self.robot_init_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.end_effector_name = None

    # ── physics loop ──────────────────────────────────────────────────────────

    def on_physics_step(self) -> None:
        self.ui_builder._on_every_frame_trajectory_list()
        for curobo_motion in self.ui_builder.curoboMotion.values():
            curobo_motion.on_physics_step(self.motion_run_ratio, None)
        self.on_command_step()

    def on_command_step(self) -> None:
        if not self.data or not self.Command:
            return
        if   self.Command == Command.INIT_ROBOT:
            self._handle_init_robot()
        elif self.Command == Command.GET_IK_STATUS:
            self._handle_get_ik_status()
        elif self.Command == Command.LINEAR_MOVE:
            self._handle_linear_move()
        if self.Command:
            with self.condition:
                self.condition.notify_all()

    # ── command handlers ──────────────────────────────────────────────────────

    def _handle_init_robot(self) -> None:
        d = self.data
        robot_config_dir = os.path.join(_ROOT, "source/data_collection/config/robot_cfg")
        robot = RobotCfg(os.path.join(robot_config_dir, d["robot_cfg_file"]))

        robot_usd_path = os.path.join(SIM_ASSETS_ROOT, robot.robot_usd)
        scene_usd_path = os.path.join(SIM_ASSETS_ROOT, d["scene_usd_path"])

        add_reference_to_stage(robot_usd_path, robot.robot_prim_path)
        add_reference_to_stage(scene_usd_path, "/World")

        self.usd_objects["robot"] = XFormPrim(
            prim_path=robot.robot_prim_path,
            position=d["robot_position"],
            orientation=d["robot_rotation"],
        )
        self.robot_init_position = np.array(d["robot_position"])
        self.robot_init_rotation = np.array(d["robot_rotation"])

        # Physics scene
        stage = omni.usd.get_context().get_stage()
        phys_scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
        phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        phys_scene.CreateGravityMagnitudeAttr().Set(9.81)

        self.robot_cfg = robot

        # Start simulation
        self.ui_builder.my_world.play()
        self.ui_builder._init_solver(robot, enable_curobo=True, batch_num=0)

        # Temp robot-description YAML with updated fixed-joint values
        init_joint_names    = d.get("init_joint_names", [])
        init_joint_position = d.get("init_joint_position", [])

        articulation = self.ui_builder.articulation
        fixed_joints = {}
        if "galbot" in robot.robot_name.lower():
            for jn in ("leg_joint1", "leg_joint2", "leg_joint3",
                       "leg_joint4", "leg_joint5",
                       "head_joint1", "head_joint2"):
                fixed_joints[jn] = articulation.get_dof_index(jn)

        for idx, jn in enumerate(init_joint_names):
            ji = articulation.get_dof_index(jn)
            if ji < len(robot.init_joint_position):
                robot.init_joint_position[ji] = init_joint_position[idx]

        def _make_tmp_yaml(rel_path: str) -> str:
            full = robot_config_dir + rel_path
            with open(full) as f:
                desc = yaml.safe_load(f)
            for rule in desc.get("cspace_to_urdf_rules", []):
                if rule.get("rule") == "fixed" and rule["name"] in fixed_joints:
                    ji = fixed_joints[rule["name"]]
                    if ji < len(robot.init_joint_position):
                        rule["value"] = float(robot.init_joint_position[ji])
            tmp = rel_path.replace(".yaml", "_tmp.yaml")
            with open(robot_config_dir + tmp, "w") as f:
                yaml.dump(desc, f, default_flow_style=False)
            return tmp

        if robot.arm_type == "dual":
            robot.robot_description_path = {
                k: _make_tmp_yaml(v)
                for k, v in robot.robot_description_path.items()
            }
        else:
            robot.robot_description_path = _make_tmp_yaml(
                robot.robot_description_path
            )

        self.ui_builder._init_kinematic_solver(robot)
        self.end_effector_name = robot.end_effector_name
        self.ui_builder.init_joint_position = robot.init_joint_position

        # Camera viewpoint – set after world.play() so it is not overridden
        init_pos = d["robot_position"]
        cam = ViewportCameraState("/OmniverseKit_Persp")
        cam.set_position_world(Gf.Vec3d(2.65, 2.4, 1.74), True)
        cam.set_target_world(
            Gf.Vec3d(init_pos[0] + 0.5, init_pos[1], init_pos[2] + 0.8), True
        )

        self.data_to_send = "success"

    def _handle_get_ik_status(self) -> None:
        d = self.data
        poses    = np.array(d["target_poses"])
        is_right = bool(d["isRight"])
        obs_avoid = bool(d["ObsAvoid"])
        out_link = bool(d["output_link_pose"])

        if not obs_avoid:
            # Simple Lula IK – one pose at a time
            results = []
            for i in range(poses.shape[0]):
                tp = poses[i, :3, 3]
                tr = transform_utils.mat2quat_wxyz(poses[i, :3, :3])
                success, _ = self.ui_builder._get_ik_status(tp, tr, is_right)
                results.append(bool(success))
        else:
            # CuRobo batch IK in robot-local frame
            R_init = transform_utils.quat2mat_wxyz(self.robot_init_rotation)
            T_wr = np.eye(4)
            T_wr[:3, :3] = R_init
            T_wr[:3, 3]  = self.robot_init_position
            poses_local = np.linalg.inv(T_wr)[np.newaxis, ...] @ poses
            pos_l = poses_local[:, :3, 3]
            rot_l = batch_matrices_to_quaternions_scipy_w_first(poses_local)

            curobo = self.ui_builder.get_curobo_motion(is_right)
            self.ui_builder.set_locked_joint_positions(is_right)
            if isinstance(self.end_effector_name, dict):
                ee = self.end_effector_name["right" if is_right else "left"]
            else:
                ee = self.end_effector_name
            results = curobo.solve_batch_ik(pos_l, rot_l, ee,
                                            output_link_pose=out_link)
        self.data_to_send = results

    def _handle_linear_move(self) -> None:
        d = self.data
        self.data_to_send = None
        tp       = np.array(d["target_position"])
        tr       = np.array(d["target_rotation"])
        is_right = bool(d["isArmRight"])

        curobo = self.ui_builder.get_curobo_motion(is_right)
        if (np.linalg.norm(self.target_position - tp) != 0.0
                or np.linalg.norm(self.target_rotation - tr) != 0.0
                or not curobo.success):
            self.target_position = tp
            self.target_rotation = tr
            self.ui_builder._followingPos        = tp
            self.ui_builder._followingOrientation = tr
            self.ui_builder._follow_target(isRight=is_right)
        if curobo.reached:
            self.data_to_send = curobo.success

    # ── blocking queue ────────────────────────────────────────────────────────

    def _on_blocking_thread(self, data, cmd: int) -> None:
        self.data    = data
        self.Command = cmd
        with self.condition:
            while self.data_to_send is None:
                self.condition.wait()
            result = self.data_to_send
            self.data_to_send = None
            self.Command      = 0
            self.result_queue.put(result)

    def blocking_start_server(self, data, cmd: int):
        self._on_blocking_thread(data, cmd)
        if not self.result_queue.empty():
            return self.result_queue.get()

    def manual_set_command(self, name: str, data_dict: dict):
        if name == "init_robot":
            return self.blocking_start_server(data_dict, Command.INIT_ROBOT)

        if name == "get_ik_status":
            is_right  = data_dict.get("arm", "right") == "right"
            obs_avoid = data_dict.get("type", "Simple") == "AvoidObs"
            return self.blocking_start_server(
                {
                    "target_poses":    data_dict["poses"],
                    "isRight":         is_right,
                    "ObsAvoid":        obs_avoid,
                    "output_link_pose": data_dict.get("output_link_pose", False),
                },
                Command.GET_IK_STATUS,
            )

        if name == "linear_move":
            return self.blocking_start_server(data_dict, Command.LINEAR_MOVE)

        raise ValueError(f"Unknown command: {name!r}")


# ──────────────────────────────────────────────────────────────────────────────
# PICK-STAGE VISUALIZER
# ──────────────────────────────────────────────────────────────────────────────

class PickStageVis:
    """Filter pipeline + motion-gen visualizer.

    Filter pipeline runs in a background thread; each step pauses on an Event
    until the user presses N.  IK checks go through blocking_start_server so
    the GPU work happens on the physics thread.

    Motion-gen is driven directly from on_physics_step() (called on the main
    thread) without a blocking queue.
    """

    _VIZ_ROOT = "/World/PickVis"

    def __init__(self, controller: PickController) -> None:
        self.ctrl         = controller
        self.ui_builder   = controller.ui_builder
        # keyboard
        self._kbd_sub     = None
        self._key_evt     = threading.Event()
        # state machine
        self._state       = "IDLE"
        # motion-gen data (set by filter pipeline when it finishes)
        self._sorted_poses: np.ndarray = np.empty((0, 4, 4))
        self._pose_idx: int = 0
        # arm joint names (set after init_robot)
        self._arm_joints: list[str] = []
        # USD draw queue: background thread submits callables; main thread executes them
        self._draw_queue: queue.Queue = queue.Queue()
        self._draw_done   = threading.Event()

    # ── keyboard ──────────────────────────────────────────────────────────────

    def setup_keyboard(self) -> None:
        kb    = omni.appwindow.get_default_app_window().get_keyboard()
        iface = carb.input.acquire_input_interface()
        ki    = carb.input.KeyboardInput

        def on_key(event, *_):
            if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                return True
            k = event.input
            if k == ki.N:
                self._on_next()
            elif k == ki.R:
                self._on_reset()
            elif k in (ki.ESCAPE, ki.Q):
                self._on_quit()
            return True

        self._kbd_sub = iface.subscribe_to_keyboard_events(kb, on_key)

    def _on_next(self) -> None:
        if self._state == "FILTER_WAIT":
            self._key_evt.set()
        elif self._state == "MOTION_WAIT":
            self._pose_idx += 1
            self._state = "MOTION_PLAN"

    def _on_reset(self) -> None:
        if self._state in ("MOTION_PLAN", "MOTION_EXECUTE", "MOTION_WAIT"):
            self._state = "MOTION_RESET"

    def _on_quit(self) -> None:
        self._state = "DONE"
        self._key_evt.set()

    # ── thread-safe USD drawing ───────────────────────────────────────────────

    def _submit_draw(self, fn) -> None:
        """Submit a USD draw callable to be executed on the main physics thread.

        Blocks the calling (background) thread until the main thread has run fn.
        USD stage modifications must only happen on the main thread in Isaac Sim.
        """
        self._draw_done.clear()
        self._draw_queue.put(fn)
        self._draw_done.wait()

    # ── filter-step N-gate (blocks background thread) ─────────────────────────

    def _wait_n(self, msg: str = "") -> bool:
        """Block background thread until user presses N.  Return False on quit."""
        self._state = "FILTER_WAIT"
        self._key_evt.clear()
        print(f"\n>>> {msg}  — click viewport for focus, then press N to continue (ESC to quit) <<<")
        self._key_evt.wait()
        return self._state != "DONE"

    # ── filter pipeline  (runs in background thread) ──────────────────────────

    def run_filter_pipeline(self, obj_T_world: np.ndarray,
                            grasp_poses_canonical: np.ndarray,
                            grasp_widths: np.ndarray) -> None:
        # Init VIZ_ROOT container on main thread
        root = self._VIZ_ROOT
        self._submit_draw(lambda: (
            _remove_prim(omni.usd.get_context().get_stage(), root),
            omni.usd.get_context().get_stage().DefinePrim(root, "Xform"),
        ))

        N = grasp_poses_canonical.shape[0]
        print(f"\n{'='*60}")
        print(f" Pick-stage filter pipeline  |  {N} raw grasp candidates")
        print(f"{'='*60}")

        # -- Step 0: raw candidates  (show in world frame) --------------------
        T_world_raw = obj_T_world[np.newaxis, ...] @ grasp_poses_canonical
        _T0, _p0 = T_world_raw, f"{root}/s0_raw"
        self._submit_draw(lambda: draw_grasps(_T0, C_GREEN, _p0, _GRASP_SPECS_X))
        print(f"\n[Step 0]  {N} raw candidates (green)")
        if not self._wait_n(f"{N} raw candidates"):
            return

        # -- Step 1: Y-percentile filter (col 1, row 3 of canonical T) --------
        y_vals = grasp_poses_canonical[:, 1, 3]
        y_lo   = np.percentile(y_vals, GRASP_LOWER_PERCENTILE)
        y_hi   = np.percentile(y_vals, GRASP_UPPER_PERCENTILE)
        mask   = (y_vals >= y_lo) & (y_vals <= y_hi)
        _T1, _m1, _p1 = T_world_raw, mask, f"{root}/s1_ypc"
        self._submit_draw(lambda: _show_filter(
            omni.usd.get_context().get_stage(), _T1, _m1, _p1,
            step=1, label="Y-percentile", specs=_GRASP_SPECS_X))
        grasp_poses_canonical = grasp_poses_canonical[mask]
        grasp_widths          = grasp_widths[mask]
        if not self._wait_n(f"{grasp_poses_canonical.shape[0]} after Y-percentile"):
            return

        # -- Step 2: apply robot_gripper_2_grasp_gripper + world transform ----
        canonical2 = grasp_poses_canonical.copy()
        canonical2[:, :3, :3] = (
            canonical2[:, :3, :3] @ ROBOT_GRIPPER_2_GRASP_GRIPPER[np.newaxis, ...]
        )
        poses_world = obj_T_world[np.newaxis, ...] @ canonical2
        N2 = poses_world.shape[0]
        _T2, _p2 = poses_world, f"{root}/s2_g2r"
        self._submit_draw(lambda: draw_grasps(_T2, C_GREEN, _p2))
        print(f"\n[Step 2]  {N2} after R_g2r + world-transform (green)")
        if not self._wait_n(f"{N2} poses in world frame"):
            return

        # -- Step 3: (no up-direction filter for this galbot task) ------------

        # -- Step 4: upside-down filter (galbot: col2,row2 > 0) ---------------
        if DISABLE_UPSIDE_DOWN:
            mask4 = poses_world[:, 2, 2] > 0.0
            _T4, _m4, _p4 = poses_world, mask4, f"{root}/s4_ud"
            self._submit_draw(lambda: _show_filter(
                omni.usd.get_context().get_stage(), _T4, _m4, _p4,
                step=4, label="upside-down (z-col>0)"))
            poses_world  = poses_world[mask4]
            canonical2   = canonical2[mask4]
            grasp_widths = grasp_widths[mask4]
            if not self._wait_n(f"{poses_world.shape[0]} not upside-down"):
                return

        # -- Step 5: downsample to MAX_SIMPLE_IK ------------------------------
        poses_world, idx5 = random_downsample(poses_world, MAX_SIMPLE_IK, False)
        if idx5 is not None:
            grasp_widths = grasp_widths[idx5]
        print(f"\n[Step 5]  downsampled → {poses_world.shape[0]}")

        # -- Step 6: apply grasp offset along approach axis (+Z col) ----------
        if GRASP_OFFSET > 0:
            approach = poses_world[:, :3, 2].copy()
            norms    = np.linalg.norm(approach, axis=1, keepdims=True) + 1e-8
            approach /= norms
            poses_world = poses_world.copy()
            poses_world[:, :3, 3] += approach * GRASP_OFFSET
            print(f"[Step 6]  applied {GRASP_OFFSET:.3f}m offset along approach (+Z)")

        # -- Step 7: Simple (Lula) IK filter ----------------------------------
        print(f"\n[Step 7]  Simple IK on {poses_world.shape[0]} poses …")
        ik_simple = self.ctrl.manual_set_command("get_ik_status", {
            "poses": poses_world.tolist(),
            "arm":   ARM,
            "type":  "Simple",
            "output_link_pose": False,
        })
        mask7 = np.array([bool(r) for r in ik_simple], dtype=bool)
        _T7, _m7, _p7 = poses_world, mask7, f"{root}/s7_sik"
        self._submit_draw(lambda: _show_filter(
            omni.usd.get_context().get_stage(), _T7, _m7, _p7,
            step=7, label="Simple IK", c_keep=C_GREEN, c_drop=C_RED))
        poses_world = poses_world[mask7]
        if not self._wait_n(f"{poses_world.shape[0]} passed Simple IK"):
            return

        if poses_world.shape[0] == 0:
            print("[PickStageVis] No poses passed Simple IK.  Done.")
            self._state = "DONE"
            return

        # -- Step 8: downsample to MAX_CUROBO_IK ------------------------------
        poses_world, _ = random_downsample(poses_world, MAX_CUROBO_IK, False)
        print(f"\n[Step 8]  downsampled → {poses_world.shape[0]}")

        # -- Step 9: CuRobo AvoidObs IK filter --------------------------------
        print(f"[Step 9]  CuRobo IK on {poses_world.shape[0]} poses …")
        ik_curobo = self.ctrl.manual_set_command("get_ik_status", {
            "poses": poses_world.tolist(),
            "arm":   ARM,
            "type":  "AvoidObs",
            "output_link_pose": False,
        })
        mask9       = np.array([bool(r[0]) for r in ik_curobo], dtype=bool)
        passed      = poses_world[mask9]
        joint_dicts = [ik_curobo[i][1] for i, ok in enumerate(mask9) if ok]
        _T9, _m9, _p9 = poses_world, mask9, f"{root}/s9_cik"
        self._submit_draw(lambda: _show_filter(
            omni.usd.get_context().get_stage(), _T9, _m9, _p9,
            step=9, label="CuRobo IK", c_keep=C_BLUE, c_drop=C_RED))
        if not self._wait_n(f"{passed.shape[0]} passed CuRobo IK"):
            return

        if passed.shape[0] == 0:
            print("[PickStageVis] No poses passed CuRobo IK.  Done.")
            self._state = "DONE"
            return

        # -- Step 10: sort by joint distance -----------------------------------
        sort_idx           = _sort_by_joint_dist(joint_dicts,
                                                  self.ui_builder.articulation,
                                                  self._arm_joints)
        self._sorted_poses = passed[sort_idx]
        sorted_joints      = [joint_dicts[i] for i in sort_idx]  # noqa: F841

        print(f"\n[Step 10] Sorted {len(self._sorted_poses)} poses by joint dist.")
        print("          Press N → start motion-gen,  R → reset robot,  ESC → quit")

        _T10, _p10 = self._sorted_poses, f"{root}/s10_sorted"
        self._submit_draw(lambda: draw_grasps(_T10, C_BLUE, _p10))

        # Transition to motion-gen (on_physics_step takes over)
        self._pose_idx = 0
        self._state    = "MOTION_PLAN"
        # background thread exits here

    # ── motion-gen state machine  (called every physics step) ─────────────────

    def on_physics_step(self) -> None:
        # Drain USD draw queue submitted by the background filter thread.
        # Must run in every state so _submit_draw() is never starved.
        while not self._draw_queue.empty():
            try:
                fn = self._draw_queue.get_nowait()
                fn()
                self._draw_done.set()
            except queue.Empty:
                break

        if self._state == "MOTION_PLAN":
            if self._pose_idx >= len(self._sorted_poses):
                print("\n[PickStageVis] All sorted poses played.  Visualization done.")
                self._state = "DONE"
                return
            self._execute_pose(self._pose_idx)
            self._state = "MOTION_EXECUTE"

        elif self._state == "MOTION_EXECUTE":
            curobo = self.ui_builder.get_curobo_motion(ARM_IS_RIGHT)
            if curobo and curobo.reached:
                n = len(self._sorted_poses)
                print(f"  Pose {self._pose_idx + 1}/{n} reached. "
                      "N=next  R=reset  ESC=quit")
                self._state = "MOTION_WAIT"

        elif self._state == "MOTION_RESET":
            # Cancel trajectory + reset joints to home
            curobo = self.ui_builder.get_curobo_motion(ARM_IS_RIGHT)
            if curobo:
                curobo.cmd_plan = None
                curobo.reached  = True
            init_pos = getattr(self.ui_builder, "init_joint_position", None)
            if init_pos is not None and self.ui_builder.articulation is not None:
                self.ui_builder.articulation.set_joint_positions(init_pos)
            self.ctrl.target_position = np.zeros(3)
            self.ctrl.target_rotation = np.array([1.0, 0.0, 0.0, 0.0])
            print(f"  Reset → replanning pose {self._pose_idx + 1}.")
            self._state = "MOTION_PLAN"

    def _execute_pose(self, idx: int) -> None:
        """Fire a motion-gen request for sorted_poses[idx] from the physics thread."""
        T_world = self._sorted_poses[idx]
        n       = len(self._sorted_poses)
        print(f"\n  Executing pose {idx + 1}/{n} …")

        # Highlight current pose yellow, rest blue
        stage = omni.usd.get_context().get_stage()
        _remove_prim(stage, f"{self._VIZ_ROOT}/s10_sorted")
        stage.DefinePrim(f"{self._VIZ_ROOT}/s10_sorted", "Xform")
        for i, T in enumerate(self._sorted_poses):
            c = C_YELLOW if i == idx else C_BLUE
            _draw_single_grasp(stage, T, c,
                               f"{self._VIZ_ROOT}/s10_sorted/g{i:04d}",
                               _GRASP_SPECS_Z)

        # Convert world pose → robot-local pose required by _follow_target
        R_init = transform_utils.quat2mat_wxyz(self.ctrl.robot_init_rotation)
        T_world_robot      = np.eye(4)
        T_world_robot[:3, :3] = R_init
        T_world_robot[:3, 3]  = self.ctrl.robot_init_position
        T_robot_ee = np.linalg.inv(T_world_robot) @ T_world

        pos_local = T_robot_ee[:3, 3]
        rot_local = transform_utils.mat2quat_wxyz(T_robot_ee[:3, :3])

        # Fire CuRobo motion plan directly (we are on the physics thread)
        self.ui_builder._followingPos        = pos_local
        self.ui_builder._followingOrientation = rot_local
        self.ctrl.target_position            = pos_local
        self.ctrl.target_rotation            = rot_local
        self.ui_builder._follow_target(isRight=ARM_IS_RIGHT)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    physics_dt   = 1.0 / 60.0
    rendering_dt = 1.0 / 30.0

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=physics_dt,
        rendering_dt=rendering_dt,
        device="cpu",
    )

    ui_builder = UIBuilder(world, debug=args.debug)
    controller = PickController(ui_builder)

    # Load task JSON for initial arm pose
    with open(TASK_JSON) as f:
        task_info = json.load(f)
    init_arm_pose = task_info["robot"]["init_arm_pose"]

    init_settings: dict = {
        "robot_cfg_file":    ROBOT_JSON,
        "scene_usd_path":    SCENE_USD,
        "robot_position":    ROBOT_POSITION,
        "robot_rotation":    ROBOT_ROTATION,
        "stand_type":        "cylinder",
        "stand_size_x":      0.1,
        "stand_size_y":      0.1,
        "init_joint_position": list(init_arm_pose.values()),
        "init_joint_names":    list(init_arm_pose.keys()),
    }

    # Spawn pick object in scene (before sim starts)
    add_reference_to_stage(OBJECT_USD_PATH, "/World/PickObject")
    XFormPrim(
        prim_path="/World/PickObject",
        position=OBJECT_WORLD_POS,
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    # Load grasp pkl
    with open(GRASP_PKL_PATH, "rb") as f:
        grasp_data = pickle.load(f)
    grasp_poses_canonical = np.asarray(grasp_data["grasp_pose"], dtype=np.float64)
    grasp_widths          = np.asarray(grasp_data["width"],      dtype=np.float64)
    print(f"[main] Loaded {grasp_poses_canonical.shape[0]} grasps from pkl.")

    # Create visualizer
    vis = PickStageVis(controller)
    vis.setup_keyboard()

    # Background thread: init robot → settle object → run filter pipeline
    def _bg_thread() -> None:
        print("[main] Initializing robot (this may take ~30 s for CuRobo) …")
        controller.manual_set_command("init_robot", init_settings)
        print("[main] Robot ready.  Waiting for object to settle on table …")

        # Physics is now running; wait ~1 s of sim time for the object to fall
        # and come to rest on the table surface.
        time.sleep(1.5)

        # Read the actual world pose of the object after settling
        obj_prim = XFormPrim("/World/PickObject")
        obj_pos, obj_rot = obj_prim.get_world_pose()
        obj_T_world = np.eye(4)
        obj_T_world[:3, :3] = transform_utils.quat2mat_wxyz(obj_rot)
        obj_T_world[:3, 3]  = obj_pos
        print(f"[main] Object settled at {obj_pos}. Starting filter pipeline …")

        # Retrieve arm joint names from robot config
        robot_cfg = controller.robot_cfg
        if robot_cfg and isinstance(getattr(robot_cfg, "active_arm_joints", None), dict):
            vis._arm_joints = robot_cfg.active_arm_joints.get(ARM, [])

        vis.run_filter_pipeline(obj_T_world, grasp_poses_canonical, grasp_widths)

    bg = threading.Thread(target=_bg_thread, daemon=True)
    bg.start()

    # ── main simulation loop ────────────────────────────────────────────────
    last_render = 0.0
    while simulation_app.is_running():
        world.step(render=False)
        t = world.current_time
        if t - last_render >= rendering_dt:
            world.render()
            last_render = t

        controller.on_physics_step()
        vis.on_physics_step()

        if vis._state == "DONE":
            print("[main] Visualization complete.")
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
