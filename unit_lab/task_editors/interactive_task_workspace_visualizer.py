#!/usr/bin/env python3
"""Interactive Isaac Sim editor for GenieSim task workspaces and sample poses.

This tool is aimed at task-template authoring. It loads a task JSON, the real
scene USD, and the configured robot USD, then exposes the task's placement
frames directly in Isaac Sim:

- `origin`
- `scene.function_space_objects` SPACE workspaces
- `scene.function_space_objects` SAMPLE poses such as `box_poses`

Edits can be made in two ways:

1. Change numeric fields in the Isaac Sim side-panel.
2. Select a handle prim from the panel, then move / rotate it with the normal
   viewport gizmo. The editor polls handle transforms and writes those changes
   back into the JSON-backed task model immediately.

The editor intentionally focuses on the fields that are difficult to tune by
trial-and-error in repeated simulator runs: workspace locations, sample pose
locations, pose random ranges, and the task origin. SAMPLE-workspace task
objects are previewed live with their workspace-relative offsets applied.

Usage:
    python unit_lab/task_editors/interactive_task_workspace_visualizer.py \
        --task-json source/data_collection/tasks/geniesim_2025/\
place_object_into_box_of_specific_color/galbot/\
place_object_into_box_of_specific_color_blue_galbot.json

Optional:
    --output <path>              Save to a different JSON path.
    --apply-robot-joints         Best-effort application of init_arm_pose.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass


DEFAULT_ASSET_ROOT = Path("/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets")
DEFAULT_TASK_JSON = Path(
    "/home/agxi/RealityLab/genie_sim/source/data_collection/tasks/geniesim_2025/"
    "place_object_into_box_of_specific_color/galbot/"
    "place_object_into_box_of_specific_color_blue_galbot_study.json"
)
DEFAULT_ROBOT_CFG_DIR = Path("/home/agxi/RealityLab/genie_sim/source/data_collection/config/robot_cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Isaac Sim task workspace visualizer/editor")
    parser.add_argument("--task-json", type=Path, default=DEFAULT_TASK_JSON, help="Task JSON to edit.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path. Defaults to --task-json.")
    parser.add_argument("--assets-root", type=Path, default=Path(os.environ.get("SIM_ASSETS", DEFAULT_ASSET_ROOT)))
    parser.add_argument("--robot-cfg-dir", type=Path, default=DEFAULT_ROBOT_CFG_DIR)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--apply-robot-joints", action="store_true", help="Best-effort application of init_arm_pose.")
    parser.add_argument("--hide-robot", action="store_true", help="Do not load the robot USD.")
    parser.add_argument("--hide-scene", action="store_true", help="Do not load the scene USD.")
    return parser.parse_known_args()[0]


ARGS = parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": ARGS.width, "height": ARGS.height})
simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
simulation_app._carb_settings.set("/omni/replicator/asyncRendering", False)
simulation_app._carb_settings.set("/app/asyncRendering", False)

import carb
import numpy as np
import omni.kit.app
import omni.ui as ui
import omni.usd
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Articulation
except Exception:  # pragma: no cover - Isaac Sim package availability is runtime-specific
    World = None
    Articulation = None

try:
    from isaacsim.gui.components.element_wrappers import ScrollingWindow
except Exception:  # pragma: no cover - Isaac Sim package availability is runtime-specific
    ScrollingWindow = None


ROOT_DIR = Path(__file__).resolve().parents[2]
COLLECTION_DIR = ROOT_DIR / "source" / "data_collection"
for candidate in (ROOT_DIR, COLLECTION_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from source.data_collection.common.base_utils.transform_utils import axis_to_quaternion, mat2quat_wxyz, quat2mat_wxyz
from source.data_collection.server.robot import RobotCfg


UI_TITLE = "GenieSim Task Workspace Visualizer"
EDITOR_ROOT = "/World/TaskEditor"
HANDLE_ROOT = f"{EDITOR_ROOT}/Handles"
OVERLAY_ROOT = f"{EDITOR_ROOT}/Overlays"
PREVIEW_ROOT = f"{EDITOR_ROOT}/PreviewObjects"
LIGHT_PATH = f"{EDITOR_ROOT}/EditorLight"

IDENTITY_POSITION = [0.0, 0.0, 0.0]
IDENTITY_QUATERNION = [1.0, 0.0, 0.0, 0.0]

COLOR_ORIGIN = np.array([1.0, 0.85, 0.2], dtype=np.float32)
COLOR_ROBOT = np.array([0.3, 1.0, 0.45], dtype=np.float32)
COLOR_WORKSPACE = np.array([0.15, 0.8, 0.95], dtype=np.float32)
COLOR_SAMPLE_POSE = np.array([1.0, 0.45, 0.15], dtype=np.float32)
COLOR_RANDOM_BOX = np.array([1.0, 0.7, 0.25], dtype=np.float32)
COLOR_PREVIEW_PLACEHOLDER = np.array([0.7, 0.7, 0.7], dtype=np.float32)
COLOR_BLOCKED = np.array([0.95, 0.25, 0.25], dtype=np.float32)
COMMAND_CONTROLLER_CAMERA_POSITION = [2.65, 2.4, 1.74]


@dataclass
class HandleRecord:
    key: str
    kind: str
    prim_path: str
    workspace_id: str | None = None
    pose_index: int | None = None
    last_world_matrix: np.ndarray | None = None


@dataclass
class PreviewSpec:
    object_id: str
    workspace_id: str
    pose_index: int
    prim_path: str
    data_info_dir: str | None
    relative_position: np.ndarray
    relative_quaternion: np.ndarray
    preview_label: str
    asset_path: Path | None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def sanitize_token(value: str) -> str:
    cleaned = "".join(ch if ch.isascii() and (ch.isalnum() or ch == "_") else "_" for ch in value)
    cleaned = cleaned.strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"item_{cleaned}"
    return cleaned or "item"


def normalize_relative_path(path_like: str) -> str:
    cleaned = str(path_like).strip().replace("\\", "/")
    return cleaned.lstrip("./").lstrip("/")


def ensure_vector(value: Any, length: int, default: float = 0.0) -> list[float]:
    if not isinstance(value, list):
        return [float(default)] * length
    result = [float(item) for item in value[:length]]
    if len(result) < length:
        result.extend([float(default)] * (length - len(result)))
    return result


def normalize_quaternion_wxyz(value: Any) -> list[float]:
    quat = np.asarray(ensure_vector(value, 4, 0.0), dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return IDENTITY_QUATERNION.copy()
    quat = quat / norm
    return quat.tolist()


def is_identity_origin(origin_entry: dict[str, Any]) -> bool:
    position = np.asarray(ensure_vector(origin_entry.get("position"), 3, 0.0), dtype=np.float64)
    quaternion = np.asarray(normalize_quaternion_wxyz(origin_entry.get("quaternion")), dtype=np.float64)
    return np.allclose(position, np.zeros(3), atol=1e-8) and np.allclose(
        quaternion, np.asarray(IDENTITY_QUATERNION), atol=1e-8
    )


def pose_matrix(position: list[float] | np.ndarray, quaternion_wxyz: list[float] | np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat2mat_wxyz(np.asarray(normalize_quaternion_wxyz(quaternion_wxyz), dtype=np.float64))
    matrix[:3, 3] = np.asarray(ensure_vector(list(position), 3, 0.0), dtype=np.float64)
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    position = matrix[:3, 3].astype(np.float64).tolist()
    quaternion = normalize_quaternion_wxyz(mat2quat_wxyz(matrix[:3, :3]).tolist())
    return position, quaternion


def np_to_gf_matrix4d(matrix: np.ndarray) -> Gf.Matrix4d:
    transposed = np.asarray(matrix, dtype=np.float64).T
    return Gf.Matrix4d(
        transposed[0, 0],
        transposed[0, 1],
        transposed[0, 2],
        transposed[0, 3],
        transposed[1, 0],
        transposed[1, 1],
        transposed[1, 2],
        transposed[1, 3],
        transposed[2, 0],
        transposed[2, 1],
        transposed[2, 2],
        transposed[2, 3],
        transposed[3, 0],
        transposed[3, 1],
        transposed[3, 2],
        transposed[3, 3],
    )


def gf_matrix_to_np(matrix: Gf.Matrix4d) -> np.ndarray:
    raw = np.array([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
    return raw.T


def set_local_matrix(prim, matrix: np.ndarray) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(np_to_gf_matrix4d(matrix))


def compute_world_matrix(prim) -> np.ndarray:
    xformable = UsdGeom.Xformable(prim)
    return gf_matrix_to_np(xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def remove_prim_if_exists(stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def set_display_color(prim, rgb: np.ndarray, opacity: float = 1.0) -> None:
    color = np.asarray(rgb, dtype=np.float32).reshape(3)
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    gprim.CreateDisplayOpacityAttr().Set([float(max(0.0, min(opacity, 1.0)))])


def create_colored_cube(path: str, color: np.ndarray, opacity: float = 1.0):
    prim = create_prim(path, prim_type="Cube")
    UsdGeom.Cube(prim).CreateSizeAttr().Set(1.0)
    set_display_color(prim, color, opacity=opacity)
    return prim


def create_bounding_box(root_path: str, color: np.ndarray, opacity: float = 1.0) -> None:
    create_prim(root_path, prim_type="Xform")
    for axis_name in ("x", "y", "z"):
        for edge_index in range(4):
            edge = create_colored_cube(f"{root_path}/edge_{axis_name}_{edge_index:02d}", color, opacity=opacity)
            set_local_matrix(edge, np.diag([1.0, 1.0, 1.0, 1.0]))


def scale_translate_matrix(scale: list[float] | np.ndarray, translation: list[float] | np.ndarray) -> np.ndarray:
    sx, sy, sz = (float(value) for value in scale)
    tx, ty, tz = (float(value) for value in translation)
    return np.array(
        [
            [sx, 0.0, 0.0, tx],
            [0.0, sy, 0.0, ty],
            [0.0, 0.0, sz, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def workspace_outline_thickness(size: list[float] | np.ndarray) -> float:
    size_values = np.asarray([float(value) for value in list(size)[:3]], dtype=np.float64)
    if size_values.size < 3:
        size_values = np.pad(size_values, (0, 3 - size_values.size), constant_values=0.1)
    min_dim = max(float(np.min(size_values)), 0.001)
    return min(max(min_dim * 0.08, 0.004), 0.02)


def set_bounding_box_geometry(stage, root_path: str, size: list[float] | np.ndarray, thickness: float) -> None:
    size_values = [float(value) for value in list(size)[:3]]
    if len(size_values) < 3:
        size_values.extend([0.001] * (3 - len(size_values)))
    x_size, y_size, z_size = (max(value, 0.001) for value in size_values)
    thickness = max(0.001, min(float(thickness), x_size, y_size, z_size))

    edge_specs: list[tuple[str, int, tuple[float, float, float], tuple[float, float, float]]] = []
    for edge_index, (y_sign, z_sign) in enumerate(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))):
        edge_specs.append(
            (
                "x",
                edge_index,
                (0.0, y_sign * y_size / 2.0, z_sign * z_size / 2.0),
                (x_size, thickness, thickness),
            )
        )
    for edge_index, (x_sign, z_sign) in enumerate(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))):
        edge_specs.append(
            (
                "y",
                edge_index,
                (x_sign * x_size / 2.0, 0.0, z_sign * z_size / 2.0),
                (thickness, y_size, thickness),
            )
        )
    for edge_index, (x_sign, y_sign) in enumerate(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))):
        edge_specs.append(
            (
                "z",
                edge_index,
                (x_sign * x_size / 2.0, y_sign * y_size / 2.0, 0.0),
                (thickness, thickness, z_size),
            )
        )

    for axis_name, edge_index, translation, scale in edge_specs:
        prim = stage.GetPrimAtPath(f"{root_path}/edge_{axis_name}_{edge_index:02d}")
        if prim and prim.IsValid():
            set_local_matrix(prim, scale_translate_matrix(scale, translation))


def normalize_blocked_zone(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None

    blocked_zone: list[list[float]] = []
    for axis_range in value:
        if not isinstance(axis_range, list):
            return None
        low, high = ensure_vector(axis_range, 2, 0.0)
        blocked_zone.append([min(float(low), float(high)), max(float(low), float(high))])
    return blocked_zone


def ensure_prim_path(stage, prim_path: str, leaf_type: str = "Xform"):
    prim_path = str(prim_path).strip()
    sdf_path = Sdf.Path(prim_path)
    if not prim_path or sdf_path.isEmpty or not sdf_path.IsAbsolutePath():
        raise ValueError(f"Invalid absolute prim path: {prim_path!r}")

    existing = stage.GetPrimAtPath(sdf_path)
    if existing and existing.IsValid():
        return existing

    parts = [part for part in prim_path.split("/") if part]
    if not parts:
        raise ValueError(f"Invalid prim path: {prim_path!r}")

    current_path = Sdf.Path.absoluteRootPath
    for index, part in enumerate(parts):
        current_path = current_path.AppendChild(part)
        prim = stage.GetPrimAtPath(current_path)
        if prim and prim.IsValid():
            continue
        prim_type = leaf_type if index == len(parts) - 1 else "Xform"
        stage.DefinePrim(current_path, prim_type)
    return stage.GetPrimAtPath(sdf_path)


def build_axes_marker(root_path: str, axis_len: float, axis_thickness: float) -> None:
    create_prim(root_path, prim_type="Xform")
    axis_specs = [
        ("x", (axis_len / 2.0, 0.0, 0.0), (axis_len, axis_thickness, axis_thickness), np.array([1.0, 0.0, 0.0])),
        ("y", (0.0, axis_len / 2.0, 0.0), (axis_thickness, axis_len, axis_thickness), np.array([0.0, 1.0, 0.0])),
        ("z", (0.0, 0.0, axis_len / 2.0), (axis_thickness, axis_thickness, axis_len), np.array([0.0, 0.0, 1.0])),
    ]
    for axis_name, translation, scale, color in axis_specs:
        cube = create_colored_cube(f"{root_path}/{axis_name}", color, opacity=1.0)
        set_local_matrix(
            cube,
            np.array(
                [
                    [scale[0], 0.0, 0.0, translation[0]],
                    [0.0, scale[1], 0.0, translation[1]],
                    [0.0, 0.0, scale[2], translation[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )


def resolve_path(root: Path, path_like: str | None) -> Path | None:
    if not path_like:
        return None
    candidate = Path(str(path_like))
    if candidate.is_absolute():
        return candidate
    return root / normalize_relative_path(str(path_like))


def choose_scene_usd(scene_usd: Any) -> str | None:
    if isinstance(scene_usd, list):
        if not scene_usd:
            return None
        first = scene_usd[0]
        if isinstance(first, dict) and first:
            return str(next(iter(first)))
        return str(first)
    if isinstance(scene_usd, dict) and scene_usd:
        return str(next(iter(scene_usd)))
    if scene_usd is None:
        return None
    return str(scene_usd)


class TaskWorkspaceEditor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.task_path = args.task_json.resolve()
        self.output_path = (args.output or args.task_json).resolve()
        self.assets_root = args.assets_root.resolve()
        self.robot_cfg_dir = args.robot_cfg_dir.resolve()

        self.window: ui.Window | None = None
        self._dock_task = None
        self.stage = None
        self.selection = omni.usd.get_context().get_selection()

        self.task_data: dict[str, Any] = {}
        self.workspace_entries: dict[str, dict[str, Any]] = {}
        self.workspace_order: list[str] = []
        self.handle_records: dict[str, HandleRecord] = {}
        self.preview_specs: list[PreviewSpec] = []
        self.preview_by_workspace: dict[str, list[PreviewSpec]] = {}
        self.object_param_cache: dict[str, dict[str, Any]] = {}

        self.robot_cfg: RobotCfg | None = None
        self.scene_usd_path: Path | None = None
        self.robot_usd_path: Path | None = None
        self.origin_explicit_in_file = False
        self.status_message = ""
        self.dirty = False

        self.world = None
        self.robot_articulation = None

        self.float_models: dict[str, ui.SimpleFloatModel] = {}
        self.float_getters: dict[str, Callable[[], float]] = {}
        self.float_setters: dict[str, Callable[[float], None]] = {}
        self.transient_float_keys: set[str] = set()
        self.suspend_model_callbacks = False
        self.suspend_handle_poll = False
        self.ui_dirty = True
        self.viewport_camera = {
            "position": COMMAND_CONTROLLER_CAMERA_POSITION.copy(),
            "target": [0.5, 0.0, 0.8],
        }

        self._warm_up()
        self._load_task_from_disk()
        self._create_float_models()
        self._rebuild_stage()
        self._sync_models_from_task()
        self._build_ui_window()
        self._set_status("Editor initialized. Move a selected handle with the viewport gizmo, then save.")

    def _warm_up(self) -> None:
        for _ in range(20):
            simulation_app.update()

    def _build_ui_window(self) -> None:
        window_kwargs = {
            "title": UI_TITLE,
            "width": 460,
            "height": 0,
            "visible": True,
            "dockPreference": ui.DockPreference.LEFT_BOTTOM,
        }
        if ScrollingWindow is not None:
            self.window = ScrollingWindow(**window_kwargs)
        else:
            self.window = ui.Window(**window_kwargs)
        self.window.visible = True
        frame = self.window.frame
        if hasattr(frame, "set_build_fn"):
            frame.set_build_fn(self._build_ui_contents)
        self._rebuild_ui()
        self._schedule_dock()

    def _schedule_dock(self) -> None:
        async def dock_window() -> None:
            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()

            window_handle = ui.Workspace.get_window(UI_TITLE)
            if window_handle is None:
                return

            targets = [
                ("Property", ui.DockPosition.SAME, 1.0),
                ("Stage", ui.DockPosition.SAME, 1.0),
                ("Viewport", ui.DockPosition.RIGHT, 0.28),
                ("DockSpace", ui.DockPosition.RIGHT, 0.28),
            ]
            for target_name, position, ratio in targets:
                target_handle = ui.Workspace.get_window(target_name)
                if target_handle is None:
                    continue
                try:
                    window_handle.dock_in(target_handle, position, ratio)
                    window_handle.focus()
                    return
                except Exception as exc:  # pragma: no cover - depends on Isaac Sim runtime
                    carb.log_warn(f"Failed to dock editor into {target_name}: {exc}")

        self._dock_task = asyncio.ensure_future(dock_window())

    def _set_status(self, message: str) -> None:
        self.status_message = message
        carb.log_info(message)
        self.ui_dirty = True

    def _mark_dirty(self, message: str) -> None:
        self.dirty = True
        self._set_status(message)

    def _load_task_from_disk(self) -> None:
        self.task_data = load_json(self.task_path)
        self.origin_explicit_in_file = "origin" in self.task_data
        self.task_data.setdefault("origin", {"position": IDENTITY_POSITION.copy(), "quaternion": IDENTITY_QUATERNION.copy()})
        self._normalize_task_defaults()
        self.workspace_entries = self._resolve_workspace_entries()
        self.workspace_order = list(self.workspace_entries.keys())

        robot_cfg_file = self.task_data.get("robot", {}).get("robot_cfg")
        self.robot_cfg = None
        self.robot_usd_path = None
        if robot_cfg_file and not self.args.hide_robot:
            robot_cfg_path = self.robot_cfg_dir / robot_cfg_file
            if robot_cfg_path.exists():
                self.robot_cfg = RobotCfg(str(robot_cfg_path))
                self.robot_usd_path = resolve_path(self.assets_root, self.robot_cfg.robot_usd)
            else:
                self._set_status(f"Robot config missing: {robot_cfg_path}")

        self.scene_usd_path = None
        scene_usd = choose_scene_usd(self.task_data.get("scene", {}).get("scene_usd"))
        if scene_usd and not self.args.hide_scene:
            self.scene_usd_path = resolve_path(self.assets_root, scene_usd)

        self.preview_specs = self._build_preview_specs()
        self.preview_by_workspace = {}
        for spec in self.preview_specs:
            self.preview_by_workspace.setdefault(spec.workspace_id, []).append(spec)
        self._reset_viewport_camera_state()

        self.dirty = False

    def _normalize_task_defaults(self) -> None:
        origin = self.task_data["origin"]
        origin["position"] = ensure_vector(origin.get("position"), 3, 0.0)
        origin["quaternion"] = normalize_quaternion_wxyz(origin.get("quaternion"))

        scene = self.task_data.setdefault("scene", {})
        function_spaces = scene.get("function_space_objects", {})
        if isinstance(function_spaces, dict):
            iterator = function_spaces.values()
        elif isinstance(function_spaces, list):
            iterator = function_spaces
        else:
            iterator = []

        for workspace in iterator:
            if not isinstance(workspace, dict):
                continue
            if "poses" in workspace:
                for pose in workspace.get("poses", []):
                    pose["position"] = ensure_vector(pose.get("position"), 3, 0.0)
                    pose["quaternion"] = normalize_quaternion_wxyz(pose.get("quaternion"))
                    if "random" in pose:
                        pose["random"]["delta_position"] = ensure_vector(
                            pose["random"].get("delta_position"), 3, 0.0
                        )
                        if "delta_angle" in pose["random"]:
                            pose["random"]["delta_angle"] = float(pose["random"].get("delta_angle", 0.0))
            else:
                workspace["position"] = ensure_vector(workspace.get("position"), 3, 0.0)
                workspace["quaternion"] = normalize_quaternion_wxyz(workspace.get("quaternion"))
                workspace["size"] = ensure_vector(workspace.get("size"), 3, 0.1)
                blocked_zone = normalize_blocked_zone(workspace.get("blocked_zone"))
                if blocked_zone is not None:
                    workspace["blocked_zone"] = blocked_zone

    def _resolve_workspace_entries(self) -> dict[str, dict[str, Any]]:
        workspaces = self.task_data.get("scene", {}).get("function_space_objects", {})
        if isinstance(workspaces, dict):
            if "position" in workspaces or "poses" in workspaces:
                return {"0": workspaces}
            return {str(key): value for key, value in workspaces.items()}
        if isinstance(workspaces, list):
            return {str(index): value for index, value in enumerate(workspaces)}
        return {}

    def _get_origin_entry(self) -> dict[str, Any]:
        return self.task_data["origin"]

    def _get_origin_matrix(self) -> np.ndarray:
        origin = self._get_origin_entry()
        return pose_matrix(origin["position"], origin["quaternion"])

    def _get_scene_workspace_key_for_robot(self) -> str:
        scene_id = str(self.task_data.get("scene", {}).get("scene_id", ""))
        return scene_id.split("/")[-1]

    def _get_robot_init_pose_entry(self) -> dict[str, Any] | None:
        robot_entry = self.task_data.get("robot")
        if not isinstance(robot_entry, dict):
            return None
        init_pose = robot_entry.get("robot_init_pose")
        if not isinstance(init_pose, dict):
            return None
        if "position" in init_pose:
            return init_pose
        key = self._get_scene_workspace_key_for_robot()
        if key in init_pose and isinstance(init_pose[key], dict):
            return init_pose[key]
        return init_pose if "position" in init_pose else None

    def _get_robot_world_matrix(self) -> np.ndarray | None:
        robot_pose = self._get_robot_init_pose_entry()
        if robot_pose is None:
            return None
        local_matrix = pose_matrix(
            ensure_vector(robot_pose.get("position"), 3, 0.0),
            normalize_quaternion_wxyz(robot_pose.get("quaternion")),
        )
        return self._get_origin_matrix() @ local_matrix

    def _default_viewport_camera_target(self) -> list[float]:
        robot_world = self._get_robot_world_matrix()
        if robot_world is not None:
            robot_position = robot_world[:3, 3]
            return [
                float(robot_position[0] + 0.5),
                float(robot_position[1]),
                float(robot_position[2] + 0.8),
            ]

        origin_position = np.asarray(self._get_origin_entry()["position"], dtype=np.float64)
        return [
            float(origin_position[0] + 0.5),
            float(origin_position[1]),
            float(origin_position[2] + 0.8),
        ]

    def _reset_viewport_camera_state(self) -> None:
        self.viewport_camera["position"] = COMMAND_CONTROLLER_CAMERA_POSITION.copy()
        self.viewport_camera["target"] = self._default_viewport_camera_target()

    def _viewport_camera_snippet(self) -> str:
        position = ensure_vector(self.viewport_camera.get("position"), 3, 0.0)
        target = ensure_vector(self.viewport_camera.get("target"), 3, 0.0)
        return (
            "camera_state.set_position_world("
            f"Gf.Vec3d({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}), True)\n"
            "camera_state.set_target_world("
            f"Gf.Vec3d({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}), True)"
        )

    def _workspace_world_matrix(self, workspace_id: str) -> np.ndarray:
        workspace = self.workspace_entries[workspace_id]
        return self._get_origin_matrix() @ pose_matrix(workspace["position"], workspace["quaternion"])

    def _sample_pose_world_matrix(self, workspace_id: str, pose_index: int) -> np.ndarray:
        pose_entry = self.workspace_entries[workspace_id]["poses"][pose_index]
        return self._get_origin_matrix() @ pose_matrix(pose_entry["position"], pose_entry["quaternion"])

    def _sample_random_box_world_matrix(self, workspace_id: str, pose_index: int) -> np.ndarray:
        pose_entry = self.workspace_entries[workspace_id]["poses"][pose_index]
        origin = self._get_origin_entry()
        box_quaternion = origin["quaternion"]
        box_position = (self._get_origin_matrix() @ np.array([*pose_entry["position"], 1.0], dtype=np.float64))[:3]
        return pose_matrix(box_position.tolist(), box_quaternion)

    def _resolve_object_parameters(self, data_info_dir: str | None) -> dict[str, Any]:
        if not data_info_dir:
            return {}
        normalized = normalize_relative_path(data_info_dir)
        if normalized in self.object_param_cache:
            return self.object_param_cache[normalized]

        object_dir = resolve_path(self.assets_root, normalized)
        object_parameters_path = object_dir / "object_parameters.json" if object_dir is not None else None
        if object_parameters_path is None or not object_parameters_path.exists():
            self.object_param_cache[normalized] = {}
            return {}

        try:
            params = load_json(object_parameters_path)
        except Exception as exc:  # pragma: no cover - file content is data-dependent
            carb.log_warn(f"Failed to load object parameters from {object_parameters_path}: {exc}")
            params = {}
        self.object_param_cache[normalized] = params
        return params

    def _resolve_preview_object_config(self, obj_entry: dict[str, Any]) -> dict[str, Any]:
        if "candidate_objects" not in obj_entry:
            return copy.deepcopy(obj_entry)
        candidates = obj_entry.get("candidate_objects") or []
        if not candidates:
            return copy.deepcopy(obj_entry)
        merged = copy.deepcopy(candidates[0])
        for key, value in obj_entry.items():
            if key != "candidate_objects":
                merged[key] = copy.deepcopy(value)
        return merged

    def _resolve_preview_relative_quaternion(self, obj_entry: dict[str, Any]) -> np.ndarray:
        rel_orientation = obj_entry.get("workspace_relative_orientation")
        if isinstance(rel_orientation, list) and len(rel_orientation) == 4:
            return np.asarray(normalize_quaternion_wxyz(rel_orientation), dtype=np.float64)

        params = self._resolve_object_parameters(obj_entry.get("data_info_dir"))
        up_axis = params.get("upAxis", ["y"])
        if isinstance(up_axis, list) and up_axis:
            up_axis_value = str(up_axis[0])
        else:
            up_axis_value = str(up_axis)
        upside_down = up_axis_value.startswith("-")
        axis_name = up_axis_value[1:] if upside_down else up_axis_value
        try:
            quat = axis_to_quaternion(axis_name or "y", "z", upside_down)
        except Exception:
            quat = np.asarray(IDENTITY_QUATERNION, dtype=np.float64)
        return np.asarray(normalize_quaternion_wxyz(quat.tolist()), dtype=np.float64)

    def _resolve_object_asset_path(self, obj_entry: dict[str, Any]) -> Path | None:
        data_info_dir = obj_entry.get("data_info_dir")
        object_dir = resolve_path(self.assets_root, data_info_dir)
        if object_dir is None:
            return None

        params = self._resolve_object_parameters(data_info_dir)

        # Match the main data_collection runtime first: it loads data_info_dir/Aligned.usd directly.
        runtime_candidates = [
            object_dir / "Aligned.usd",
            object_dir / "model.usd",
        ]
        for candidate in runtime_candidates:
            if candidate.exists():
                return candidate

        # Some asset metadata exposes an explicit model_path. Prefer resolving it from the global
        # asset root first because many entries are not relative to the object directory itself.
        model_path = params.get("model_path")
        if model_path:
            for root in (self.assets_root, object_dir):
                resolved = resolve_path(root, str(model_path))
                if resolved is not None and resolved.exists():
                    return resolved

        fallback_candidates = [
            object_dir / "Aligned.usda",
            object_dir / "model.usda",
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                return candidate
        return None

    def _build_preview_specs(self) -> list[PreviewSpec]:
        preview_specs: list[PreviewSpec] = []
        sample_pose_counts = {
            workspace_id: len(workspace.get("poses", []))
            for workspace_id, workspace in self.workspace_entries.items()
            if isinstance(workspace, dict) and "poses" in workspace
        }
        next_pose_index = {workspace_id: 0 for workspace_id in sample_pose_counts}

        task_objects = self.task_data.get("objects", {}).get("task_related_objects", [])
        for obj_entry in task_objects:
            workspace_id = str(obj_entry.get("workspace_id", ""))
            if workspace_id not in sample_pose_counts:
                continue

            pose_index = next_pose_index[workspace_id]
            if pose_index >= sample_pose_counts[workspace_id]:
                carb.log_warn(
                    f"Skipping preview for {obj_entry.get('object_id', 'unknown')} because workspace "
                    f"{workspace_id} has fewer poses than assigned task objects."
                )
                continue
            next_pose_index[workspace_id] += 1

            resolved = self._resolve_preview_object_config(obj_entry)
            rel_position = np.asarray(
                ensure_vector(resolved.get("workspace_relative_position"), 3, 0.0),
                dtype=np.float64,
            )
            rel_quaternion = self._resolve_preview_relative_quaternion(resolved)
            asset_path = self._resolve_object_asset_path(resolved)

            preview_specs.append(
                PreviewSpec(
                    object_id=str(resolved.get("object_id", f"preview_{workspace_id}_{pose_index}")),
                    workspace_id=workspace_id,
                    pose_index=pose_index,
                    prim_path=(
                        f"{PREVIEW_ROOT}/{sanitize_token(workspace_id)}/"
                        f"preview_{pose_index:02d}_{sanitize_token(str(resolved.get('object_id', 'preview')))}"
                    ),
                    data_info_dir=resolved.get("data_info_dir"),
                    relative_position=rel_position,
                    relative_quaternion=rel_quaternion,
                    preview_label=str(resolved.get("object_id", "preview")),
                    asset_path=asset_path,
                )
            )
        return preview_specs

    def _clear_stage_state(self) -> None:
        self.handle_records = {}
        self.stage = None
        self.world = None
        self.robot_articulation = None

    def _rebuild_stage(self) -> None:
        self._clear_stage_state()
        create_new_stage()
        for _ in range(5):
            simulation_app.update()

        self.stage = omni.usd.get_context().get_stage()

        if not self.args.hide_scene and self.scene_usd_path and self.scene_usd_path.exists():
            add_reference_to_stage(str(self.scene_usd_path), "/World")
        else:
            create_prim("/World", prim_type="Xform")
            if not self.args.hide_scene and self.scene_usd_path is not None:
                carb.log_warn(f"Scene USD missing: {self.scene_usd_path}")

        create_prim(EDITOR_ROOT, prim_type="Xform")
        create_prim(HANDLE_ROOT, prim_type="Xform")
        create_prim(OVERLAY_ROOT, prim_type="Xform")
        create_prim(PREVIEW_ROOT, prim_type="Xform")
        self._ensure_editor_light()
        self._load_robot_reference()
        if self.args.apply_robot_joints:
            self._try_apply_robot_joint_pose()
        self._build_handle_prims()
        self._build_preview_prims()
        self._apply_model_to_stage(sync_models=False)
        self._apply_viewport_camera(sync_models=False)
        self.ui_dirty = True

    def _ensure_editor_light(self) -> None:
        light = UsdLux.SphereLight.Define(self.stage, LIGHT_PATH)
        light.CreateIntensityAttr(70000.0)
        light.CreateRadiusAttr(0.25)
        light_prim = self.stage.GetPrimAtPath(LIGHT_PATH)
        set_local_matrix(
            light_prim,
            pose_matrix([1.3, 1.8, 2.2], IDENTITY_QUATERNION),
        )

    def _load_robot_reference(self) -> None:
        if self.args.hide_robot or self.robot_cfg is None or self.robot_usd_path is None:
            return
        if not self.robot_usd_path.exists():
            carb.log_warn(f"Robot USD missing: {self.robot_usd_path}")
            return
        add_reference_to_stage(str(self.robot_usd_path), self.robot_cfg.robot_prim_path)

    def _try_apply_robot_joint_pose(self) -> None:
        if self.robot_cfg is None or World is None or Articulation is None:
            return
        init_pose = self.task_data.get("robot", {}).get("init_arm_pose")
        if not isinstance(init_pose, dict) or not init_pose:
            return
        try:
            self.world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
            self.robot_articulation = Articulation(prim_path=self.robot_cfg.robot_prim_path, name="task_editor_robot")
            self.world.scene.add(self.robot_articulation)
            self.world.reset()
            self.robot_articulation.initialize()

            joint_indices: list[int] = []
            joint_values: list[float] = []
            for joint_name, joint_value in init_pose.items():
                dof_index = self.robot_articulation.get_dof_index(joint_name)
                if dof_index < 0:
                    continue
                joint_indices.append(int(dof_index))
                joint_values.append(float(joint_value))
            if joint_indices:
                self.robot_articulation.set_joint_positions(
                    np.asarray(joint_values, dtype=np.float64),
                    joint_indices=np.asarray(joint_indices, dtype=np.int64),
                )
                for _ in range(4):
                    self.world.step(render=True)
            if self.world is not None:
                self.world.pause()
        except Exception as exc:  # pragma: no cover - depends on Isaac Sim runtime
            carb.log_warn(f"Failed to apply robot joint pose: {exc}")
            self.world = None
            self.robot_articulation = None

    def _build_handle_prims(self) -> None:
        origin_handle_path = f"{HANDLE_ROOT}/origin"
        origin_prim = create_prim(origin_handle_path, prim_type="Xform")
        build_axes_marker(f"{origin_handle_path}/axes", axis_len=0.22, axis_thickness=0.01)
        origin_center = create_colored_cube(f"{origin_handle_path}/center", COLOR_ORIGIN, opacity=0.95)
        set_local_matrix(origin_center, np.diag([0.03, 0.03, 0.03, 1.0]))
        self.handle_records["origin"] = HandleRecord(key="origin", kind="origin", prim_path=origin_handle_path)

        robot_pose = self._get_robot_init_pose_entry()
        if robot_pose is not None:
            robot_handle_path = f"{HANDLE_ROOT}/robot_init"
            create_prim(robot_handle_path, prim_type="Xform")
            build_axes_marker(f"{robot_handle_path}/axes", axis_len=0.18, axis_thickness=0.009)
            robot_center = create_colored_cube(f"{robot_handle_path}/center", COLOR_ROBOT, opacity=0.95)
            set_local_matrix(robot_center, np.diag([0.035, 0.035, 0.035, 1.0]))
            self.handle_records[self._robot_handle_key()] = HandleRecord(
                key=self._robot_handle_key(),
                kind="robot",
                prim_path=robot_handle_path,
            )

        for workspace_id in self.workspace_order:
            workspace = self.workspace_entries[workspace_id]
            if "poses" in workspace:
                self._build_sample_workspace_handles(workspace_id, workspace)
            else:
                self._build_space_workspace_handle(workspace_id)

    def _build_space_workspace_handle(self, workspace_id: str) -> None:
        prim_path = f"{HANDLE_ROOT}/workspace_{sanitize_token(workspace_id)}"
        create_prim(prim_path, prim_type="Xform")
        build_axes_marker(f"{prim_path}/axes", axis_len=0.14, axis_thickness=0.008)
        center_cube = create_colored_cube(f"{prim_path}/center", COLOR_WORKSPACE, opacity=0.95)
        set_local_matrix(center_cube, np.diag([0.025, 0.025, 0.025, 1.0]))
        create_bounding_box(f"{prim_path}/volume_bbox", COLOR_WORKSPACE, opacity=0.95)
        blocked_zone_cube = create_colored_cube(f"{prim_path}/blocked_zone", COLOR_BLOCKED, opacity=0.0)
        set_local_matrix(blocked_zone_cube, np.diag([0.001, 0.001, 0.001, 1.0]))
        self.handle_records[self._workspace_handle_key(workspace_id)] = HandleRecord(
            key=self._workspace_handle_key(workspace_id),
            kind="workspace",
            prim_path=prim_path,
            workspace_id=workspace_id,
        )

    def _build_sample_workspace_handles(self, workspace_id: str, workspace: dict[str, Any]) -> None:
        group_path = f"{HANDLE_ROOT}/sample_{sanitize_token(workspace_id)}"
        create_prim(group_path, prim_type="Xform")
        for pose_index, _ in enumerate(workspace.get("poses", [])):
            pose_path = f"{group_path}/pose_{pose_index:02d}"
            create_prim(pose_path, prim_type="Xform")
            build_axes_marker(f"{pose_path}/axes", axis_len=0.12, axis_thickness=0.007)
            marker = create_colored_cube(f"{pose_path}/marker", COLOR_SAMPLE_POSE, opacity=0.95)
            set_local_matrix(marker, np.diag([0.03, 0.03, 0.03, 1.0]))

            random_box_path = f"{OVERLAY_ROOT}/sample_{sanitize_token(workspace_id)}_pose_{pose_index:02d}_random_box"
            random_box = create_colored_cube(random_box_path, COLOR_RANDOM_BOX, opacity=0.15)
            set_local_matrix(random_box, np.diag([1.0, 1.0, 1.0, 1.0]))

            self.handle_records[self._pose_handle_key(workspace_id, pose_index)] = HandleRecord(
                key=self._pose_handle_key(workspace_id, pose_index),
                kind="pose",
                prim_path=pose_path,
                workspace_id=workspace_id,
                pose_index=pose_index,
            )

    def _build_preview_prims(self) -> None:
        for spec in self.preview_specs:
            ensure_prim_path(self.stage, Sdf.Path(spec.prim_path).GetParentPath(), leaf_type="Xform")
            preview_root = ensure_prim_path(self.stage, spec.prim_path, leaf_type="Xform")
            if spec.asset_path is not None and spec.asset_path.exists():
                preview_root.GetReferences().AddReference(str(spec.asset_path))
            else:
                placeholder = create_colored_cube(f"{spec.prim_path}/placeholder", COLOR_PREVIEW_PLACEHOLDER, opacity=0.8)
                set_local_matrix(placeholder, np.diag([0.08, 0.08, 0.08, 1.0]))
            build_axes_marker(f"{spec.prim_path}/axes", axis_len=0.08, axis_thickness=0.004)

    def _apply_model_to_stage(self, sync_models: bool = True) -> None:
        if self.stage is None:
            return

        self.suspend_handle_poll = True
        try:
            origin_prim = self.stage.GetPrimAtPath(self.handle_records["origin"].prim_path)
            set_local_matrix(origin_prim, self._get_origin_matrix())

            self._apply_robot_handle_transform()
            for workspace_id in self.workspace_order:
                workspace = self.workspace_entries[workspace_id]
                if "poses" in workspace:
                    self._apply_sample_workspace_state(workspace_id, workspace)
                else:
                    self._apply_space_workspace_state(workspace_id, workspace)

            self._apply_robot_base_transform()
            self._apply_preview_transforms()
            self._snapshot_handle_matrices()
        finally:
            self.suspend_handle_poll = False

        if sync_models:
            self._sync_models_from_task()
        self.ui_dirty = True

    def _apply_space_workspace_state(self, workspace_id: str, workspace: dict[str, Any]) -> None:
        handle = self.handle_records[self._workspace_handle_key(workspace_id)]
        prim = self.stage.GetPrimAtPath(handle.prim_path)
        set_local_matrix(prim, self._workspace_world_matrix(workspace_id))

        size = ensure_vector(workspace.get("size"), 3, 0.1)
        set_bounding_box_geometry(
            self.stage,
            f"{handle.prim_path}/volume_bbox",
            size,
            thickness=workspace_outline_thickness(size),
        )

        blocked_zone_prim = self.stage.GetPrimAtPath(f"{handle.prim_path}/blocked_zone")
        blocked_zone = normalize_blocked_zone(workspace.get("blocked_zone"))
        if not blocked_zone_prim or not blocked_zone_prim.IsValid():
            return
        if blocked_zone is None:
            set_display_color(blocked_zone_prim, COLOR_BLOCKED, opacity=0.0)
            set_local_matrix(blocked_zone_prim, np.diag([0.001, 0.001, 0.001, 1.0]))
            return

        x_range, y_range = blocked_zone
        blocked_zone_size = [
            max(x_range[1] - x_range[0], 0.001),
            max(y_range[1] - y_range[0], 0.001),
            max(float(size[2]), 0.001),
        ]
        blocked_zone_center = [
            (x_range[0] + x_range[1]) / 2.0,
            (y_range[0] + y_range[1]) / 2.0,
            0.0,
        ]
        set_display_color(blocked_zone_prim, COLOR_BLOCKED, opacity=0.28)
        set_local_matrix(blocked_zone_prim, scale_translate_matrix(blocked_zone_size, blocked_zone_center))

    def _apply_sample_workspace_state(self, workspace_id: str, workspace: dict[str, Any]) -> None:
        for pose_index, pose_entry in enumerate(workspace.get("poses", [])):
            handle = self.handle_records[self._pose_handle_key(workspace_id, pose_index)]
            prim = self.stage.GetPrimAtPath(handle.prim_path)
            set_local_matrix(prim, self._sample_pose_world_matrix(workspace_id, pose_index))

            random_box_prim = self.stage.GetPrimAtPath(
                f"{OVERLAY_ROOT}/sample_{sanitize_token(workspace_id)}_pose_{pose_index:02d}_random_box"
            )
            random_delta = ensure_vector(pose_entry.get("random", {}).get("delta_position"), 3, 0.0)
            random_world = self._sample_random_box_world_matrix(workspace_id, pose_index)
            random_scale = [max(2.0 * delta, 0.004) for delta in random_delta]
            random_box_matrix = random_world.copy()
            random_box_matrix[:3, :3] = random_box_matrix[:3, :3] @ np.diag(random_scale)
            set_local_matrix(random_box_prim, random_box_matrix)

    def _apply_robot_base_transform(self) -> None:
        if self.robot_cfg is None:
            return
        robot_world = self._get_robot_world_matrix()
        if robot_world is None:
            return
        prim = self.stage.GetPrimAtPath(self.robot_cfg.robot_prim_path)
        if prim and prim.IsValid():
            set_local_matrix(prim, robot_world)

    def _apply_robot_handle_transform(self) -> None:
        handle = self.handle_records.get(self._robot_handle_key())
        if handle is None:
            return
        robot_world = self._get_robot_world_matrix()
        if robot_world is None:
            return
        prim = self.stage.GetPrimAtPath(handle.prim_path)
        if prim and prim.IsValid():
            set_local_matrix(prim, robot_world)

    def _apply_preview_transforms(self) -> None:
        for spec in self.preview_specs:
            if spec.workspace_id not in self.workspace_entries:
                continue
            pose_world = self._sample_pose_world_matrix(spec.workspace_id, spec.pose_index)
            relative = pose_matrix(spec.relative_position.tolist(), spec.relative_quaternion.tolist())
            final_world = pose_world @ relative
            prim = self.stage.GetPrimAtPath(spec.prim_path)
            if prim and prim.IsValid():
                set_local_matrix(prim, final_world)

    def _snapshot_handle_matrices(self) -> None:
        for record in self.handle_records.values():
            prim = self.stage.GetPrimAtPath(record.prim_path)
            if prim and prim.IsValid():
                record.last_world_matrix = compute_world_matrix(prim)

    def _workspace_handle_key(self, workspace_id: str) -> str:
        return f"workspace:{workspace_id}"

    def _pose_handle_key(self, workspace_id: str, pose_index: int) -> str:
        return f"pose:{workspace_id}:{pose_index}"

    def _robot_handle_key(self) -> str:
        return "robot_init"

    def _create_float_models(self) -> None:
        self.float_models = {}
        self.float_getters = {}
        self.float_setters = {}
        self.transient_float_keys = set()

        origin = self._get_origin_entry()
        for index, axis_name in enumerate(("x", "y", "z")):
            key = f"origin.position.{axis_name}"
            self._register_float_model(
                key=key,
                getter=lambda origin=origin, index=index: float(ensure_vector(origin.get("position"), 3, 0.0)[index]),
                setter=lambda value, origin=origin, index=index: self._set_vector_component(
                    origin, "position", index, value, length=3, default=0.0
                ),
            )
        for index, axis_name in enumerate(("w", "x", "y", "z")):
            key = f"origin.quaternion.{axis_name}"
            self._register_float_model(
                key=key,
                getter=lambda origin=origin, index=index: float(
                    normalize_quaternion_wxyz(origin.get("quaternion"))[index]
                ),
                setter=lambda value, origin=origin, index=index: self._set_quaternion_component(
                    origin, "quaternion", index, value
                ),
            )

        robot_pose = self._get_robot_init_pose_entry()
        if robot_pose is not None:
            for index, axis_name in enumerate(("x", "y", "z")):
                key = f"robot_init.position.{axis_name}"
                self._register_float_model(
                    key=key,
                    getter=lambda robot_pose=robot_pose, index=index: float(
                        ensure_vector(robot_pose.get("position"), 3, 0.0)[index]
                    ),
                    setter=lambda value, robot_pose=robot_pose, index=index: self._set_vector_component(
                        robot_pose, "position", index, value, length=3, default=0.0
                    ),
                )
            for index, axis_name in enumerate(("w", "x", "y", "z")):
                key = f"robot_init.quaternion.{axis_name}"
                self._register_float_model(
                    key=key,
                    getter=lambda robot_pose=robot_pose, index=index: float(
                        normalize_quaternion_wxyz(robot_pose.get("quaternion"))[index]
                    ),
                    setter=lambda value, robot_pose=robot_pose, index=index: self._set_quaternion_component(
                        robot_pose, "quaternion", index, value
                    ),
                )

        for index, axis_name in enumerate(("x", "y", "z")):
            key = f"viewport_camera.position.{axis_name}"
            self._register_float_model(
                key=key,
                getter=lambda index=index: float(
                    ensure_vector(self.viewport_camera.get("position"), 3, 0.0)[index]
                ),
                setter=lambda value, index=index: self._set_vector_component(
                    self.viewport_camera, "position", index, value, length=3, default=0.0
                ),
                transient=True,
            )
        for index, axis_name in enumerate(("x", "y", "z")):
            key = f"viewport_camera.target.{axis_name}"
            self._register_float_model(
                key=key,
                getter=lambda index=index: float(
                    ensure_vector(self.viewport_camera.get("target"), 3, 0.0)[index]
                ),
                setter=lambda value, index=index: self._set_vector_component(
                    self.viewport_camera, "target", index, value, length=3, default=0.0
                ),
                transient=True,
            )

        for workspace_id in self.workspace_order:
            workspace = self.workspace_entries[workspace_id]
            if "poses" in workspace:
                for pose_index, pose in enumerate(workspace.get("poses", [])):
                    for index, axis_name in enumerate(("x", "y", "z")):
                        key = f"pose.{workspace_id}.{pose_index}.position.{axis_name}"
                        self._register_float_model(
                            key=key,
                            getter=lambda pose=pose, index=index: float(
                                ensure_vector(pose.get("position"), 3, 0.0)[index]
                            ),
                            setter=lambda value, pose=pose, index=index: self._set_vector_component(
                                pose, "position", index, value, length=3, default=0.0
                            ),
                        )
                    for index, axis_name in enumerate(("w", "x", "y", "z")):
                        key = f"pose.{workspace_id}.{pose_index}.quaternion.{axis_name}"
                        self._register_float_model(
                            key=key,
                            getter=lambda pose=pose, index=index: float(
                                normalize_quaternion_wxyz(pose.get("quaternion"))[index]
                            ),
                            setter=lambda value, pose=pose, index=index: self._set_quaternion_component(
                                pose, "quaternion", index, value
                            ),
                        )
                    for index, axis_name in enumerate(("x", "y", "z")):
                        key = f"pose.{workspace_id}.{pose_index}.random_delta.{axis_name}"
                        self._register_float_model(
                            key=key,
                            getter=lambda pose=pose, index=index: float(
                                ensure_vector(pose.get("random", {}).get("delta_position"), 3, 0.0)[index]
                            ),
                            setter=lambda value, pose=pose, index=index: self._set_random_delta_component(
                                pose, index, value
                            ),
                        )
                    delta_angle_key = f"pose.{workspace_id}.{pose_index}.delta_angle"
                    self._register_float_model(
                        key=delta_angle_key,
                        getter=lambda pose=pose: float(pose.get("random", {}).get("delta_angle", 0.0)),
                        setter=lambda value, pose=pose: self._set_random_delta_angle(pose, value),
                    )
            else:
                for index, axis_name in enumerate(("x", "y", "z")):
                    key = f"workspace.{workspace_id}.position.{axis_name}"
                    self._register_float_model(
                        key=key,
                        getter=lambda workspace=workspace, index=index: float(
                            ensure_vector(workspace.get("position"), 3, 0.0)[index]
                        ),
                        setter=lambda value, workspace=workspace, index=index: self._set_vector_component(
                            workspace, "position", index, value, length=3, default=0.0
                        ),
                    )
                for index, axis_name in enumerate(("w", "x", "y", "z")):
                    key = f"workspace.{workspace_id}.quaternion.{axis_name}"
                    self._register_float_model(
                        key=key,
                        getter=lambda workspace=workspace, index=index: float(
                            normalize_quaternion_wxyz(workspace.get("quaternion"))[index]
                        ),
                        setter=lambda value, workspace=workspace, index=index: self._set_quaternion_component(
                            workspace, "quaternion", index, value
                        ),
                    )
                for index, axis_name in enumerate(("x", "y", "z")):
                    key = f"workspace.{workspace_id}.size.{axis_name}"
                    self._register_float_model(
                        key=key,
                        getter=lambda workspace=workspace, index=index: float(
                            ensure_vector(workspace.get("size"), 3, 0.1)[index]
                        ),
                        setter=lambda value, workspace=workspace, index=index: self._set_workspace_size_component(
                            workspace, index, value
                        ),
                    )

    def _register_float_model(
        self,
        key: str,
        getter: Callable[[], float],
        setter: Callable[[float], None],
        transient: bool = False,
    ) -> None:
        model = ui.SimpleFloatModel(float(getter()))
        model.add_end_edit_fn(lambda model, key=key: self._on_float_model_changed(key, model))
        self.float_models[key] = model
        self.float_getters[key] = getter
        self.float_setters[key] = setter
        if transient:
            self.transient_float_keys.add(key)

    def _on_float_model_changed(self, key: str, model: ui.SimpleFloatModel) -> None:
        if self.suspend_model_callbacks:
            return
        value = float(model.get_value_as_float())
        setter = self.float_setters[key]
        setter(value)
        if key in self.transient_float_keys:
            self._apply_viewport_camera(sync_models=False)
            self._set_status(f"Updated {key} (viewport only, not saved).")
            return
        self._apply_model_to_stage(sync_models=True)
        self._mark_dirty(f"Updated {key}")

    def _set_vector_component(
        self,
        container: dict[str, Any],
        field: str,
        index: int,
        value: float,
        length: int,
        default: float,
    ) -> None:
        container[field] = ensure_vector(container.get(field), length, default)
        container[field][index] = float(value)

    def _set_quaternion_component(self, container: dict[str, Any], field: str, index: int, value: float) -> None:
        current = ensure_vector(container.get(field), 4, 0.0)
        current[index] = float(value)
        container[field] = normalize_quaternion_wxyz(current)

    def _set_workspace_size_component(self, workspace: dict[str, Any], index: int, value: float) -> None:
        workspace["size"] = ensure_vector(workspace.get("size"), 3, 0.1)
        workspace["size"][index] = max(float(value), 0.001)

    def _set_random_delta_component(self, pose: dict[str, Any], index: int, value: float) -> None:
        pose.setdefault("random", {})
        pose["random"]["delta_position"] = ensure_vector(pose["random"].get("delta_position"), 3, 0.0)
        pose["random"]["delta_position"][index] = max(float(value), 0.0)

    def _set_random_delta_angle(self, pose: dict[str, Any], value: float) -> None:
        pose.setdefault("random", {})
        pose["random"]["delta_angle"] = float(value)

    def _sync_models_from_task(self) -> None:
        self.suspend_model_callbacks = True
        try:
            for key, model in self.float_models.items():
                model.set_value(float(self.float_getters[key]()))
        finally:
            self.suspend_model_callbacks = False
        self.ui_dirty = True

    def _poll_handle_updates(self) -> None:
        if self.stage is None or self.suspend_handle_poll:
            return

        changed_records: list[HandleRecord] = []
        for record in self.handle_records.values():
            prim = self.stage.GetPrimAtPath(record.prim_path)
            if not prim or not prim.IsValid():
                continue
            current_world = compute_world_matrix(prim)
            previous_world = record.last_world_matrix
            if previous_world is None:
                record.last_world_matrix = current_world
                continue
            if np.max(np.abs(current_world - previous_world)) <= 1e-5:
                continue
            changed_records.append(record)
            record.last_world_matrix = current_world

        if not changed_records:
            return

        origin_changed = any(record.kind == "origin" for record in changed_records)
        if origin_changed:
            for record in changed_records:
                if record.kind != "origin":
                    continue
                prim = self.stage.GetPrimAtPath(record.prim_path)
                if not prim or not prim.IsValid():
                    continue
                world_matrix = compute_world_matrix(prim)
                position, quaternion = matrix_to_pose(world_matrix)
                origin = self._get_origin_entry()
                origin["position"] = position
                origin["quaternion"] = quaternion
                break

        origin_inv = np.linalg.inv(self._get_origin_matrix())
        for record in changed_records:
            prim = self.stage.GetPrimAtPath(record.prim_path)
            if not prim or not prim.IsValid():
                continue
            world_matrix = compute_world_matrix(prim)
            if record.kind == "origin":
                continue
            elif record.kind == "robot":
                robot_pose = self._get_robot_init_pose_entry()
                if robot_pose is None:
                    continue
                local_matrix = origin_inv @ world_matrix
                position, quaternion = matrix_to_pose(local_matrix)
                robot_pose["position"] = position
                robot_pose["quaternion"] = quaternion
            elif record.kind == "workspace" and record.workspace_id is not None:
                local_matrix = origin_inv @ world_matrix
                position, quaternion = matrix_to_pose(local_matrix)
                workspace = self.workspace_entries[record.workspace_id]
                workspace["position"] = position
                workspace["quaternion"] = quaternion
            elif record.kind == "pose" and record.workspace_id is not None and record.pose_index is not None:
                local_matrix = origin_inv @ world_matrix
                position, quaternion = matrix_to_pose(local_matrix)
                pose = self.workspace_entries[record.workspace_id]["poses"][record.pose_index]
                pose["position"] = position
                pose["quaternion"] = quaternion

        self._apply_model_to_stage(sync_models=True)
        if origin_changed:
            self._mark_dirty("Origin moved from viewport handle.")
        elif any(record.kind == "robot" for record in changed_records):
            self._mark_dirty("Robot init pose moved from viewport handle.")
        else:
            self._mark_dirty("Workspace / pose handle moved from viewport.")

    def _select_handle(self, handle_key: str) -> None:
        record = self.handle_records.get(handle_key)
        if record is None:
            return
        try:
            self.selection.set_selected_prim_paths([record.prim_path], False)
        except TypeError:
            self.selection.set_selected_prim_paths([record.prim_path], False, "")
        self._set_status(f"Selected handle: {record.prim_path}")

    def _compute_frame_camera_pose(self) -> tuple[list[float], list[float]]:
        points: list[np.ndarray] = []
        points.append(np.asarray(self._get_origin_entry()["position"], dtype=np.float64))
        robot_world = self._get_robot_world_matrix()
        if robot_world is not None:
            points.append(robot_world[:3, 3])
        for workspace_id in self.workspace_order:
            workspace = self.workspace_entries[workspace_id]
            if "poses" in workspace:
                for pose_index in range(len(workspace.get("poses", []))):
                    points.append(self._sample_pose_world_matrix(workspace_id, pose_index)[:3, 3])
            else:
                points.append(self._workspace_world_matrix(workspace_id)[:3, 3])

        if not points:
            center = np.zeros(3, dtype=np.float64)
            span = 1.0
        else:
            stacked = np.stack(points)
            mins = np.min(stacked, axis=0)
            maxs = np.max(stacked, axis=0)
            center = (mins + maxs) / 2.0
            span = max(float(np.max(maxs - mins)), 0.8)

        eye = [center[0] + span * 1.8, center[1] + span * 1.35, center[2] + span * 1.25]
        target = [center[0], center[1], center[2] + span * 0.15]
        return eye, target

    def _apply_viewport_camera(self, sync_models: bool = True) -> None:
        position = ensure_vector(self.viewport_camera.get("position"), 3, 0.0)
        target = ensure_vector(self.viewport_camera.get("target"), 3, 0.0)
        set_camera_view(eye=position, target=target, camera_prim_path="/OmniverseKit_Persp")
        if sync_models:
            self._sync_models_from_task()

    def _frame_camera(self) -> None:
        eye, target = self._compute_frame_camera_pose()
        self.viewport_camera["position"] = [float(value) for value in eye]
        self.viewport_camera["target"] = [float(value) for value in target]
        self._apply_viewport_camera(sync_models=True)
        self._set_status("Viewport camera set from current scene bounds (not saved).")

    def _reset_command_camera(self) -> None:
        self._reset_viewport_camera_state()
        self._apply_viewport_camera(sync_models=True)
        self._set_status("Viewport camera reset to command_controller.py defaults (not saved).")

    def _apply_camera_from_fields(self) -> None:
        self._apply_viewport_camera(sync_models=True)
        self._set_status("Applied viewport camera fields (not saved).")

    def _log_camera_snippet(self) -> None:
        carb.log_info(self._viewport_camera_snippet())
        self._set_status("Logged current camera snippet to Isaac console (not saved).")

    def _save_task(self) -> None:
        save_data = copy.deepcopy(self.task_data)
        if not self.origin_explicit_in_file and is_identity_origin(save_data["origin"]):
            save_data.pop("origin", None)
        dump_json(self.output_path, save_data)
        self.dirty = False
        self._set_status(f"Saved task JSON to {self.output_path}")

    def _reload_from_disk(self) -> None:
        self._load_task_from_disk()
        self._create_float_models()
        self._rebuild_stage()
        self._sync_models_from_task()
        self._set_status(f"Reloaded task JSON from {self.task_path}")

    def _refresh_preview(self) -> None:
        self._apply_model_to_stage(sync_models=True)
        self._set_status("Preview refreshed from current task model.")

    def _redock_window(self) -> None:
        self._schedule_dock()
        self._set_status("Requested editor re-dock into Isaac UI side region.")

    def _rebuild_ui(self) -> None:
        if self.window is None:
            return
        frame = self.window.frame
        if hasattr(frame, "rebuild"):
            frame.rebuild()
        else:
            with frame:
                self._build_ui_contents()
        self.ui_dirty = False

    def _build_ui_contents(self) -> None:
        with ui.VStack(spacing=8, height=0):
            ui.Label(UI_TITLE, height=24)
            ui.Label(str(self.task_path), word_wrap=True, height=36)
            ui.Label(f"Save target: {self.output_path}", word_wrap=True, height=24)
            ui.Label(
                f"Dirty: {'yes' if self.dirty else 'no'} | Assets: {self.assets_root}",
                word_wrap=True,
                height=22,
            )
            ui.Label(self.status_message, word_wrap=True, height=44)

            with ui.HStack(height=30, spacing=8):
                ui.Button("Save", width=82, clicked_fn=self._save_task)
                ui.Button("Reload", width=82, clicked_fn=self._reload_from_disk)
                ui.Button("Refresh", width=82, clicked_fn=self._refresh_preview)
                ui.Button("Re-dock", width=82, clicked_fn=self._redock_window)
                ui.Button("Frame All", width=82, clicked_fn=self._frame_camera)

            ui.Separator(height=6)
            ui.Label(
                "Workflow: pick a handle with Select, use the viewport move/rotate gizmo, "
                "or edit the numeric fields below. Numeric edits apply on Enter or when the field loses focus. "
                "SAMPLE-workspace preview objects update live.",
                word_wrap=True,
                height=64,
            )

            self._build_viewport_camera_ui()
            ui.Separator(height=8)
            self._build_origin_ui()
            robot_pose = self._get_robot_init_pose_entry()
            if robot_pose is not None:
                ui.Separator(height=8)
                self._build_robot_ui()
            ui.Separator(height=8)

            for workspace_id in self.workspace_order:
                self._build_workspace_ui(workspace_id, self.workspace_entries[workspace_id])
                ui.Separator(height=8)

    def _build_origin_ui(self) -> None:
        ui.Label("Origin", height=22)
        with ui.HStack(height=28, spacing=8):
            ui.Button("Select Origin Handle", width=180, clicked_fn=lambda: self._select_handle("origin"))
        self._build_vector_row(
            "Position",
            [f"origin.position.{axis}" for axis in ("x", "y", "z")],
            axis_labels=("x", "y", "z"),
        )
        self._build_vector_row(
            "Quaternion",
            [f"origin.quaternion.{axis}" for axis in ("w", "x", "y", "z")],
            axis_labels=("w", "x", "y", "z"),
        )

    def _build_viewport_camera_ui(self) -> None:
        ui.Label("Viewport Camera (Not Saved)", height=22)
        ui.Label(
            "These fields are only for tuning the hardcoded command_controller.py camera. "
            "Save does not write them to JSON.",
            word_wrap=True,
            height=40,
        )
        with ui.HStack(height=28, spacing=8):
            ui.Button("Apply Camera", width=110, clicked_fn=self._apply_camera_from_fields)
            ui.Button("Cmd Default", width=110, clicked_fn=self._reset_command_camera)
            ui.Button("Log Snippet", width=110, clicked_fn=self._log_camera_snippet)
        self._build_vector_row(
            "Position",
            [f"viewport_camera.position.{axis}" for axis in ("x", "y", "z")],
            axis_labels=("x", "y", "z"),
        )
        self._build_vector_row(
            "Target",
            [f"viewport_camera.target.{axis}" for axis in ("x", "y", "z")],
            axis_labels=("x", "y", "z"),
        )
        ui.Label(self._viewport_camera_snippet(), word_wrap=True, height=50)

    def _build_robot_ui(self) -> None:
        ui.Label("Robot Init Pose", height=22)
        with ui.HStack(height=28, spacing=8):
            ui.Button(
                "Select Robot Handle",
                width=180,
                clicked_fn=lambda: self._select_handle(self._robot_handle_key()),
            )
        self._build_vector_row(
            "Position",
            [f"robot_init.position.{axis}" for axis in ("x", "y", "z")],
            axis_labels=("x", "y", "z"),
        )
        self._build_vector_row(
            "Quaternion",
            [f"robot_init.quaternion.{axis}" for axis in ("w", "x", "y", "z")],
            axis_labels=("w", "x", "y", "z"),
        )

    def _build_workspace_ui(self, workspace_id: str, workspace: dict[str, Any]) -> None:
        mode = "SAMPLE" if "poses" in workspace else "SPACE"
        ui.Label(f"Workspace: {workspace_id} ({mode})", height=22)
        if "poses" in workspace:
            previews = self.preview_by_workspace.get(workspace_id, [])
            if previews:
                for spec in previews:
                    ui.Label(
                        f"Preview: {spec.preview_label} -> pose[{spec.pose_index}]",
                        word_wrap=True,
                        height=22,
                    )
            for pose_index, _ in enumerate(workspace.get("poses", [])):
                pose_handle_key = self._pose_handle_key(workspace_id, pose_index)
                with ui.HStack(height=28, spacing=8):
                    ui.Label(f"Pose {pose_index}", width=80)
                    ui.Button(
                        "Select Handle",
                        width=120,
                        clicked_fn=lambda handle_key=pose_handle_key: self._select_handle(handle_key),
                    )
                self._build_vector_row(
                    "Position",
                    [f"pose.{workspace_id}.{pose_index}.position.{axis}" for axis in ("x", "y", "z")],
                    axis_labels=("x", "y", "z"),
                )
                self._build_vector_row(
                    "Quaternion",
                    [f"pose.{workspace_id}.{pose_index}.quaternion.{axis}" for axis in ("w", "x", "y", "z")],
                    axis_labels=("w", "x", "y", "z"),
                )
                self._build_vector_row(
                    "Random dPos",
                    [f"pose.{workspace_id}.{pose_index}.random_delta.{axis}" for axis in ("x", "y", "z")],
                    axis_labels=("x", "y", "z"),
                )
                self._build_vector_row(
                    "dAngle",
                    [f"pose.{workspace_id}.{pose_index}.delta_angle"],
                    axis_labels=("rad",),
                )
        else:
            workspace_handle_key = self._workspace_handle_key(workspace_id)
            with ui.HStack(height=28, spacing=8):
                ui.Button(
                    "Select Workspace Handle",
                    width=180,
                    clicked_fn=lambda handle_key=workspace_handle_key: self._select_handle(handle_key),
                )
            self._build_vector_row(
                "Position",
                [f"workspace.{workspace_id}.position.{axis}" for axis in ("x", "y", "z")],
                axis_labels=("x", "y", "z"),
            )
            self._build_vector_row(
                "Quaternion",
                [f"workspace.{workspace_id}.quaternion.{axis}" for axis in ("w", "x", "y", "z")],
                axis_labels=("w", "x", "y", "z"),
            )
            self._build_vector_row(
                "Size",
                [f"workspace.{workspace_id}.size.{axis}" for axis in ("x", "y", "z")],
                axis_labels=("x", "y", "z"),
            )

    def _build_vector_row(self, title: str, keys: list[str], axis_labels: tuple[str, ...]) -> None:
        with ui.HStack(height=26, spacing=4):
            ui.Label(title, width=100)
            for axis_label, key in zip(axis_labels, keys, strict=True):
                ui.Label(axis_label, width=16)
                ui.FloatField(model=self.float_models[key], width=88)

    def run(self) -> None:
        while simulation_app.is_running():
            self._poll_handle_updates()
            if self.ui_dirty:
                self._rebuild_ui()
            simulation_app.update()

    def close(self) -> None:
        simulation_app.close()


def main() -> None:
    editor = TaskWorkspaceEditor(ARGS)
    try:
        editor.run()
    except KeyboardInterrupt:
        pass
    finally:
        editor.close()


if __name__ == "__main__":
    main()
