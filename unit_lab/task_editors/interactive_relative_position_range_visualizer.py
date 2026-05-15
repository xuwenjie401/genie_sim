#!/usr/bin/env python3
"""Focused Isaac Sim editor for relative-position place success ranges.

This tool visualizes only task_metric rules named
`is_object_relative_position_in_target`.  The green range box is parented to the
target object's pose, matching the runtime checker:

    relative_pose = inv(target_pose) @ object_pose

Usage:
    python unit_lab/task_editors/interactive_relative_position_range_visualizer.py \
        --task-json source/data_collection/tasks/diy/single_task/\
left_place_cola_can_into_box_galbot_s3c2v1.json
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
    "/home/agxi/RealityLab/genie_sim/source/data_collection/tasks/diy/single_task/"
    "left_place_cola_can_into_box_galbot_s3c2v1.json"
)
DEFAULT_ROBOT_CFG_DIR = Path("/home/agxi/RealityLab/genie_sim/source/data_collection/config/robot_cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive visualizer for relative_position_range task metrics")
    parser.add_argument("--task-json", type=Path, default=DEFAULT_TASK_JSON, help="Task JSON to edit.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path. Defaults to --task-json.")
    parser.add_argument("--assets-root", type=Path, default=Path(os.environ.get("SIM_ASSETS", DEFAULT_ASSET_ROOT)))
    parser.add_argument("--robot-cfg-dir", type=Path, default=DEFAULT_ROBOT_CFG_DIR)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--hide-robot", action="store_true", help="Do not load the robot USD.")
    parser.add_argument("--hide-scene", action="store_true", help="Do not load the scene USD.")
    parser.add_argument(
        "--show-checked-objects",
        action="store_true",
        help="Also preview checked objects. SPACE-workspace objects are shown at the workspace center.",
    )
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
    from isaacsim.gui.components.element_wrappers import ScrollingWindow
except Exception:  # pragma: no cover - Isaac Sim package availability is runtime-specific
    ScrollingWindow = None


ROOT_DIR = Path(__file__).resolve().parents[2]
COLLECTION_DIR = ROOT_DIR / "source" / "data_collection"
for candidate in (ROOT_DIR, COLLECTION_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from source.data_collection.common.base_utils.transform_utils import axis_to_quaternion, quat2mat_wxyz
from source.data_collection.server.robot import RobotCfg


UI_TITLE = "Relative Position Range Visualizer"
VIS_ROOT = "/World/RelativePositionRangeVisualizer"
PREVIEW_ROOT = f"{VIS_ROOT}/TargetObjects"
RANGE_ROOT = f"{VIS_ROOT}/SuccessRanges"
CHECKED_ROOT = f"{VIS_ROOT}/CheckedObjects"
LIGHT_PATH = f"{VIS_ROOT}/VisualizerLight"

IDENTITY_POSITION = [0.0, 0.0, 0.0]
IDENTITY_QUATERNION = [1.0, 0.0, 0.0, 0.0]

COLOR_RANGE = np.array([0.05, 1.0, 0.5], dtype=np.float32)
COLOR_TARGET = np.array([1.0, 0.82, 0.18], dtype=np.float32)
COLOR_CHECKED = np.array([0.2, 0.7, 1.0], dtype=np.float32)
COLOR_PLACEHOLDER = np.array([0.72, 0.72, 0.72], dtype=np.float32)
COMMAND_CONTROLLER_CAMERA_POSITION = [2.65, 2.4, 1.74]


@dataclass
class ObjectPreviewSpec:
    object_id: str
    prim_path: str
    asset_path: Path | None
    world_matrix: np.ndarray
    label: str
    note: str
    is_target: bool


@dataclass
class RelativeRangeRule:
    rule_index: int
    params: dict[str, Any]
    target_id: str
    object_ids: list[str]


@dataclass
class RangeOverlaySpec:
    rule: RelativeRangeRule
    prim_path: str
    target_preview: ObjectPreviewSpec | None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def sanitize_token(value: str) -> str:
    cleaned = "".join(ch if ch.isascii() and (ch.isalnum() or ch == "_") else "_" for ch in str(value))
    cleaned = cleaned.strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"item_{cleaned}"
    return cleaned or "item"


def normalize_relative_path(path_like: str) -> str:
    cleaned = str(path_like).strip().replace("\\", "/")
    return cleaned.lstrip("./").lstrip("/")


def ensure_vector(value: Any, length: int, default: float = 0.0) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)
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
    return (quat / norm).tolist()


def normalize_range(value: Any) -> list[list[float]]:
    result: list[list[float]] = []
    for axis_index in range(3):
        axis_range = value[axis_index] if isinstance(value, list) and axis_index < len(value) else None
        low, high = ensure_vector(axis_range, 2, 0.0)
        result.append([min(float(low), float(high)), max(float(low), float(high))])
    return result


def normalize_object_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return []


def pose_matrix(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat2mat_wxyz(np.asarray(normalize_quaternion_wxyz(quaternion_wxyz), dtype=np.float64))
    matrix[:3, 3] = np.asarray(ensure_vector(position, 3, 0.0), dtype=np.float64)
    return matrix


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


def set_local_matrix(prim, matrix: np.ndarray) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(np_to_gf_matrix4d(matrix))


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


def scale_translate_matrix(scale: Any, translation: Any) -> np.ndarray:
    sx, sy, sz = (float(value) for value in ensure_vector(list(scale), 3, 0.001))
    tx, ty, tz = (float(value) for value in ensure_vector(list(translation), 3, 0.0))
    return np.array(
        [
            [sx, 0.0, 0.0, tx],
            [0.0, sy, 0.0, ty],
            [0.0, 0.0, sz, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def outline_thickness(size: Any) -> float:
    values = np.asarray(ensure_vector(list(size), 3, 0.1), dtype=np.float64)
    min_dim = max(float(np.min(values)), 0.001)
    return min(max(min_dim * 0.08, 0.004), 0.018)


def create_bounding_box(root_path: str, color: np.ndarray, opacity: float = 1.0) -> None:
    create_prim(root_path, prim_type="Xform")
    for axis_name in ("x", "y", "z"):
        for edge_index in range(4):
            edge = create_colored_cube(f"{root_path}/edge_{axis_name}_{edge_index:02d}", color, opacity=opacity)
            set_local_matrix(edge, np.diag([1.0, 1.0, 1.0, 1.0]))


def set_bounding_box_geometry(stage, root_path: str, size: Any, thickness: float) -> None:
    x_size, y_size, z_size = (max(float(value), 0.001) for value in ensure_vector(list(size), 3, 0.1))
    thickness = max(0.001, min(float(thickness), x_size, y_size, z_size))

    edge_specs: list[tuple[str, int, tuple[float, float, float], tuple[float, float, float]]] = []
    for edge_index, (y_sign, z_sign) in enumerate(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))):
        edge_specs.append(
            ("x", edge_index, (0.0, y_sign * y_size / 2.0, z_sign * z_size / 2.0), (x_size, thickness, thickness))
        )
    for edge_index, (x_sign, z_sign) in enumerate(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))):
        edge_specs.append(
            ("y", edge_index, (x_sign * x_size / 2.0, 0.0, z_sign * z_size / 2.0), (thickness, y_size, thickness))
        )
    for edge_index, (x_sign, y_sign) in enumerate(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))):
        edge_specs.append(
            ("z", edge_index, (x_sign * x_size / 2.0, y_sign * y_size / 2.0, 0.0), (thickness, thickness, z_size))
        )

    for axis_name, edge_index, translation, scale in edge_specs:
        prim = stage.GetPrimAtPath(f"{root_path}/edge_{axis_name}_{edge_index:02d}")
        if prim and prim.IsValid():
            set_local_matrix(prim, scale_translate_matrix(scale, translation))


def build_axes_marker(root_path: str, axis_len: float, axis_thickness: float) -> None:
    create_prim(root_path, prim_type="Xform")
    axis_specs = [
        ("x", (axis_len / 2.0, 0.0, 0.0), (axis_len, axis_thickness, axis_thickness), np.array([1.0, 0.0, 0.0])),
        ("y", (0.0, axis_len / 2.0, 0.0), (axis_thickness, axis_len, axis_thickness), np.array([0.0, 1.0, 0.0])),
        ("z", (0.0, 0.0, axis_len / 2.0), (axis_thickness, axis_thickness, axis_len), np.array([0.0, 0.0, 1.0])),
    ]
    for axis_name, translation, scale, color in axis_specs:
        cube = create_colored_cube(f"{root_path}/{axis_name}", color, opacity=1.0)
        set_local_matrix(cube, scale_translate_matrix(scale, translation))


def ensure_prim_path(stage, prim_path: str | Sdf.Path, leaf_type: str = "Xform"):
    prim_path = str(prim_path)
    sdf_path = Sdf.Path(prim_path)
    if not prim_path or sdf_path.isEmpty or not sdf_path.IsAbsolutePath():
        raise ValueError(f"Invalid absolute prim path: {prim_path!r}")
    existing = stage.GetPrimAtPath(sdf_path)
    if existing and existing.IsValid():
        return existing
    parts = [part for part in prim_path.split("/") if part]
    current_path = Sdf.Path.absoluteRootPath
    for index, part in enumerate(parts):
        current_path = current_path.AppendChild(part)
        prim = stage.GetPrimAtPath(current_path)
        if prim and prim.IsValid():
            continue
        stage.DefinePrim(current_path, leaf_type if index == len(parts) - 1 else "Xform")
    return stage.GetPrimAtPath(sdf_path)


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


class RelativePositionRangeVisualizer:
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
        self.origin_explicit_in_file = False
        self.workspace_entries: dict[str, dict[str, Any]] = {}
        self.metric_rules: list[RelativeRangeRule] = []
        self.target_previews_by_id: dict[str, list[ObjectPreviewSpec]] = {}
        self.checked_previews_by_id: dict[str, list[ObjectPreviewSpec]] = {}
        self.overlay_specs: list[RangeOverlaySpec] = []
        self.object_param_cache: dict[str, dict[str, Any]] = {}

        self.robot_cfg: RobotCfg | None = None
        self.robot_usd_path: Path | None = None
        self.scene_usd_path: Path | None = None

        self.float_models: dict[str, ui.SimpleFloatModel] = {}
        self.float_getters: dict[str, Callable[[], float]] = {}
        self.float_setters: dict[str, Callable[[float], None]] = {}
        self.transient_float_keys: set[str] = set()
        self.suspend_model_callbacks = False
        self.ui_dirty = True
        self.status_message = ""
        self.dirty = False
        self.viewport_camera = {
            "position": COMMAND_CONTROLLER_CAMERA_POSITION.copy(),
            "target": [0.4, 0.2, 0.8],
        }

        self._warm_up()
        self._load_task_from_disk()
        self._create_float_models()
        self._rebuild_stage()
        self._sync_models_from_task()
        self._build_ui_window()
        self._set_status("Visualizer initialized. Edit relative_position_range fields, then save.")

    def _warm_up(self) -> None:
        for _ in range(20):
            simulation_app.update()

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
        self.metric_rules = self._build_metric_rules()
        self._resolve_scene_and_robot_paths()
        self._build_preview_specs()
        self._build_overlay_specs()
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
            else:
                workspace["position"] = ensure_vector(workspace.get("position"), 3, 0.0)
                workspace["quaternion"] = normalize_quaternion_wxyz(workspace.get("quaternion"))
                workspace["size"] = ensure_vector(workspace.get("size"), 3, 0.1)

        for rule in self.task_data.get("task_metric", {}).get("filter_rules", []):
            if not isinstance(rule, dict) or rule.get("rule_name") != "is_object_relative_position_in_target":
                continue
            params = rule.setdefault("params", {})
            if isinstance(params, dict):
                params["relative_position_range"] = normalize_range(params.get("relative_position_range"))
                params["objects"] = normalize_object_ids(params.get("objects"))

    def _resolve_workspace_entries(self) -> dict[str, dict[str, Any]]:
        workspaces = self.task_data.get("scene", {}).get("function_space_objects", {})
        if isinstance(workspaces, dict):
            if "position" in workspaces or "poses" in workspaces:
                return {"0": workspaces}
            return {str(key): value for key, value in workspaces.items() if isinstance(value, dict)}
        if isinstance(workspaces, list):
            return {str(index): value for index, value in enumerate(workspaces) if isinstance(value, dict)}
        return {}

    def _build_metric_rules(self) -> list[RelativeRangeRule]:
        rules: list[RelativeRangeRule] = []
        filter_rules = self.task_data.get("task_metric", {}).get("filter_rules", [])
        if not isinstance(filter_rules, list):
            return rules
        for rule_index, rule in enumerate(filter_rules):
            if not isinstance(rule, dict) or rule.get("rule_name") != "is_object_relative_position_in_target":
                continue
            params = rule.get("params", {})
            if not isinstance(params, dict):
                continue
            params["relative_position_range"] = normalize_range(params.get("relative_position_range"))
            params["objects"] = normalize_object_ids(params.get("objects"))
            rules.append(
                RelativeRangeRule(
                    rule_index=rule_index,
                    params=params,
                    target_id=str(params.get("target", "")),
                    object_ids=params["objects"],
                )
            )
        return rules

    def _resolve_scene_and_robot_paths(self) -> None:
        self.scene_usd_path = None
        scene_usd = choose_scene_usd(self.task_data.get("scene", {}).get("scene_usd"))
        if scene_usd and not self.args.hide_scene:
            self.scene_usd_path = resolve_path(self.assets_root, scene_usd)

        self.robot_cfg = None
        self.robot_usd_path = None
        robot_cfg_file = self.task_data.get("robot", {}).get("robot_cfg")
        if robot_cfg_file and not self.args.hide_robot:
            robot_cfg_path = self.robot_cfg_dir / robot_cfg_file
            if robot_cfg_path.exists():
                self.robot_cfg = RobotCfg(str(robot_cfg_path))
                self.robot_usd_path = resolve_path(self.assets_root, self.robot_cfg.robot_usd)
            else:
                carb.log_warn(f"Robot config missing: {robot_cfg_path}")

    def _get_origin_matrix(self) -> np.ndarray:
        origin = self.task_data["origin"]
        return pose_matrix(origin["position"], origin["quaternion"])

    def _workspace_world_matrix(self, workspace_id: str) -> np.ndarray:
        workspace = self.workspace_entries[workspace_id]
        return self._get_origin_matrix() @ pose_matrix(workspace["position"], workspace["quaternion"])

    def _sample_pose_world_matrix(self, workspace_id: str, pose_index: int) -> np.ndarray:
        pose = self.workspace_entries[workspace_id]["poses"][pose_index]
        return self._get_origin_matrix() @ pose_matrix(pose["position"], pose["quaternion"])

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
        return None

    def _get_robot_world_matrix(self) -> np.ndarray | None:
        robot_pose = self._get_robot_init_pose_entry()
        if robot_pose is None:
            return None
        return self._get_origin_matrix() @ pose_matrix(
            ensure_vector(robot_pose.get("position"), 3, 0.0),
            normalize_quaternion_wxyz(robot_pose.get("quaternion")),
        )

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

    def _resolve_relative_quaternion(self, obj_entry: dict[str, Any]) -> np.ndarray:
        rel_orientation = obj_entry.get("workspace_relative_orientation")
        if isinstance(rel_orientation, list) and len(rel_orientation) == 4:
            return np.asarray(normalize_quaternion_wxyz(rel_orientation), dtype=np.float64)

        params = self._resolve_object_parameters(obj_entry.get("data_info_dir"))
        up_axis = params.get("upAxis", ["y"])
        up_axis_value = str(up_axis[0]) if isinstance(up_axis, list) and up_axis else str(up_axis)
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
        for candidate in (object_dir / "Aligned.usd", object_dir / "model.usd"):
            if candidate.exists():
                return candidate
        model_path = params.get("model_path")
        if model_path:
            for root in (self.assets_root, object_dir):
                resolved = resolve_path(root, str(model_path))
                if resolved is not None and resolved.exists():
                    return resolved
        for candidate in (object_dir / "Aligned.usda", object_dir / "model.usda"):
            if candidate.exists():
                return candidate
        return None

    def _iter_task_object_entries(self) -> list[dict[str, Any]]:
        entries = self.task_data.get("objects", {}).get("task_related_objects", [])
        return [entry for entry in entries if isinstance(entry, dict)]

    def _object_entry_world_matrices(self, obj_entry: dict[str, Any]) -> list[tuple[np.ndarray, str]]:
        if "position" in obj_entry:
            return [
                (
                    self._get_origin_matrix()
                    @ pose_matrix(
                        ensure_vector(obj_entry.get("position"), 3, 0.0),
                        normalize_quaternion_wxyz(obj_entry.get("quaternion")),
                    ),
                    "direct object pose",
                )
            ]

        workspace_id = str(obj_entry.get("workspace_id", ""))
        if workspace_id not in self.workspace_entries:
            return [(self._get_origin_matrix(), "fallback origin pose; workspace not found")]

        relative = pose_matrix(
            ensure_vector(obj_entry.get("workspace_relative_position"), 3, 0.0),
            self._resolve_relative_quaternion(obj_entry),
        )
        workspace = self.workspace_entries[workspace_id]
        if "poses" in workspace:
            matrices = []
            for pose_index, _ in enumerate(workspace.get("poses", [])):
                matrices.append(
                    (
                        self._sample_pose_world_matrix(workspace_id, pose_index) @ relative,
                        f"SAMPLE workspace {workspace_id} pose {pose_index}",
                    )
                )
            return matrices
        return [
            (
                self._workspace_world_matrix(workspace_id) @ relative,
                f"SPACE workspace {workspace_id} center fallback",
            )
        ]

    def _build_object_previews_for_ids(self, object_ids: set[str], is_target: bool) -> dict[str, list[ObjectPreviewSpec]]:
        previews_by_id: dict[str, list[ObjectPreviewSpec]] = {object_id: [] for object_id in object_ids}
        root = PREVIEW_ROOT if is_target else CHECKED_ROOT
        for obj_entry in self._iter_task_object_entries():
            object_id = str(obj_entry.get("object_id", ""))
            if object_id not in object_ids:
                continue
            resolved = self._resolve_preview_object_config(obj_entry)
            asset_path = self._resolve_object_asset_path(resolved)
            for pose_index, (world_matrix, note) in enumerate(self._object_entry_world_matrices(obj_entry)):
                previews_by_id[object_id].append(
                    ObjectPreviewSpec(
                        object_id=object_id,
                        prim_path=(
                            f"{root}/{sanitize_token(object_id)}/"
                            f"pose_{pose_index:02d}_{sanitize_token(str(resolved.get('object_id', object_id)))}"
                        ),
                        asset_path=asset_path,
                        world_matrix=world_matrix,
                        label=str(resolved.get("object_id", object_id)),
                        note=note,
                        is_target=is_target,
                    )
                )
        return previews_by_id

    def _build_preview_specs(self) -> None:
        target_ids = {rule.target_id for rule in self.metric_rules if rule.target_id}
        checked_ids = {object_id for rule in self.metric_rules for object_id in rule.object_ids}
        self.target_previews_by_id = self._build_object_previews_for_ids(target_ids, is_target=True)
        self.checked_previews_by_id = (
            self._build_object_previews_for_ids(checked_ids, is_target=False) if self.args.show_checked_objects else {}
        )

    def _build_overlay_specs(self) -> None:
        self.overlay_specs = []
        for rule in self.metric_rules:
            target_previews = self.target_previews_by_id.get(rule.target_id, [])
            if not target_previews:
                self.overlay_specs.append(
                    RangeOverlaySpec(
                        rule=rule,
                        prim_path=f"{RANGE_ROOT}/rule_{rule.rule_index:02d}_{sanitize_token(rule.target_id)}_unresolved",
                        target_preview=None,
                    )
                )
                continue
            for preview_index, target_preview in enumerate(target_previews):
                self.overlay_specs.append(
                    RangeOverlaySpec(
                        rule=rule,
                        prim_path=(
                            f"{RANGE_ROOT}/rule_{rule.rule_index:02d}_{sanitize_token(rule.target_id)}"
                            f"/target_pose_{preview_index:02d}"
                        ),
                        target_preview=target_preview,
                    )
                )

    def _reset_viewport_camera_state(self) -> None:
        points = [preview.world_matrix[:3, 3] for previews in self.target_previews_by_id.values() for preview in previews]
        if points:
            stacked = np.stack(points)
            center = np.mean(stacked, axis=0)
            self.viewport_camera["target"] = [float(center[0]), float(center[1]), float(center[2])]
            self.viewport_camera["position"] = [
                float(center[0] + 1.1),
                float(center[1] + 1.1),
                float(center[2] + 0.75),
            ]
        else:
            self.viewport_camera["position"] = COMMAND_CONTROLLER_CAMERA_POSITION.copy()
            self.viewport_camera["target"] = [0.4, 0.2, 0.8]

    def _rebuild_stage(self) -> None:
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

        create_prim(VIS_ROOT, prim_type="Xform")
        create_prim(PREVIEW_ROOT, prim_type="Xform")
        create_prim(RANGE_ROOT, prim_type="Xform")
        create_prim(CHECKED_ROOT, prim_type="Xform")
        self._ensure_light()
        self._load_robot_reference()
        self._build_object_preview_prims()
        self._build_range_prims()
        self._apply_model_to_stage(sync_models=False)
        self._apply_viewport_camera(sync_models=False)
        self.ui_dirty = True

    def _ensure_light(self) -> None:
        light = UsdLux.SphereLight.Define(self.stage, LIGHT_PATH)
        light.CreateIntensityAttr(60000.0)
        light.CreateRadiusAttr(0.25)
        set_local_matrix(self.stage.GetPrimAtPath(LIGHT_PATH), pose_matrix([1.3, 1.8, 2.2], IDENTITY_QUATERNION))

    def _load_robot_reference(self) -> None:
        if self.args.hide_robot or self.robot_cfg is None or self.robot_usd_path is None:
            return
        if not self.robot_usd_path.exists():
            carb.log_warn(f"Robot USD missing: {self.robot_usd_path}")
            return
        add_reference_to_stage(str(self.robot_usd_path), self.robot_cfg.robot_prim_path)

    def _build_object_preview_prims(self) -> None:
        all_previews = [
            preview
            for previews in list(self.target_previews_by_id.values()) + list(self.checked_previews_by_id.values())
            for preview in previews
        ]
        for preview in all_previews:
            ensure_prim_path(self.stage, Sdf.Path(preview.prim_path).GetParentPath(), leaf_type="Xform")
            preview_root = ensure_prim_path(self.stage, preview.prim_path, leaf_type="Xform")
            if preview.asset_path is not None and preview.asset_path.exists():
                preview_root.GetReferences().AddReference(str(preview.asset_path))
            else:
                placeholder = create_colored_cube(
                    f"{preview.prim_path}/placeholder",
                    COLOR_TARGET if preview.is_target else COLOR_CHECKED,
                    opacity=0.75,
                )
                set_local_matrix(placeholder, np.diag([0.08, 0.08, 0.08, 1.0]))
            build_axes_marker(f"{preview.prim_path}/axes", axis_len=0.11 if preview.is_target else 0.08, axis_thickness=0.005)

    def _build_range_prims(self) -> None:
        for overlay in self.overlay_specs:
            ensure_prim_path(self.stage, overlay.prim_path, leaf_type="Xform")
            volume = create_colored_cube(f"{overlay.prim_path}/volume", COLOR_RANGE, opacity=0.16)
            set_local_matrix(volume, np.diag([0.001, 0.001, 0.001, 1.0]))
            create_bounding_box(f"{overlay.prim_path}/bbox", COLOR_RANGE, opacity=0.95)

    def _apply_model_to_stage(self, sync_models: bool = True) -> None:
        if self.stage is None:
            return
        robot_world = self._get_robot_world_matrix()
        if self.robot_cfg is not None and robot_world is not None:
            robot_prim = self.stage.GetPrimAtPath(self.robot_cfg.robot_prim_path)
            if robot_prim and robot_prim.IsValid():
                set_local_matrix(robot_prim, robot_world)

        for previews in list(self.target_previews_by_id.values()) + list(self.checked_previews_by_id.values()):
            for preview in previews:
                prim = self.stage.GetPrimAtPath(preview.prim_path)
                if prim and prim.IsValid():
                    set_local_matrix(prim, preview.world_matrix)

        for overlay in self.overlay_specs:
            self._apply_overlay_state(overlay)

        if sync_models:
            self._sync_models_from_task()
        self.ui_dirty = True

    def _apply_overlay_state(self, overlay: RangeOverlaySpec) -> None:
        root_prim = self.stage.GetPrimAtPath(overlay.prim_path)
        if not root_prim or not root_prim.IsValid():
            return
        if overlay.target_preview is None:
            set_local_matrix(root_prim, np.diag([0.001, 0.001, 0.001, 1.0]))
            return

        range_values = normalize_range(overlay.rule.params.get("relative_position_range"))
        overlay.rule.params["relative_position_range"] = range_values
        lows = np.asarray([axis_range[0] for axis_range in range_values], dtype=np.float64)
        highs = np.asarray([axis_range[1] for axis_range in range_values], dtype=np.float64)
        center = ((lows + highs) * 0.5).tolist()
        size = np.maximum(highs - lows, 0.001).tolist()

        set_local_matrix(root_prim, overlay.target_preview.world_matrix)

        volume_prim = self.stage.GetPrimAtPath(f"{overlay.prim_path}/volume")
        if volume_prim and volume_prim.IsValid():
            set_display_color(volume_prim, COLOR_RANGE, opacity=0.16)
            set_local_matrix(volume_prim, scale_translate_matrix(size, center))

        bbox_path = f"{overlay.prim_path}/bbox"
        bbox_prim = self.stage.GetPrimAtPath(bbox_path)
        if bbox_prim and bbox_prim.IsValid():
            set_local_matrix(bbox_prim, pose_matrix(center, IDENTITY_QUATERNION))
            set_bounding_box_geometry(self.stage, bbox_path, size, thickness=outline_thickness(size))

    def _build_ui_window(self) -> None:
        kwargs = {
            "title": UI_TITLE,
            "width": 430,
            "height": 0,
            "visible": True,
            "dockPreference": ui.DockPreference.LEFT_BOTTOM,
        }
        self.window = ScrollingWindow(**kwargs) if ScrollingWindow is not None else ui.Window(**kwargs)
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
            for target_name, position, ratio in (
                ("Property", ui.DockPosition.SAME, 1.0),
                ("Stage", ui.DockPosition.SAME, 1.0),
                ("Viewport", ui.DockPosition.RIGHT, 0.28),
                ("DockSpace", ui.DockPosition.RIGHT, 0.28),
            ):
                target_handle = ui.Workspace.get_window(target_name)
                if target_handle is None:
                    continue
                try:
                    window_handle.dock_in(target_handle, position, ratio)
                    window_handle.focus()
                    return
                except Exception as exc:  # pragma: no cover - depends on Isaac Sim runtime
                    carb.log_warn(f"Failed to dock visualizer into {target_name}: {exc}")

        self._dock_task = asyncio.ensure_future(dock_window())

    def _create_float_models(self) -> None:
        self.float_models = {}
        self.float_getters = {}
        self.float_setters = {}
        self.transient_float_keys = set()
        for rule in self.metric_rules:
            for axis_index, axis_name in enumerate(("x", "y", "z")):
                for bound_index, bound_name in enumerate(("min", "max")):
                    key = self._range_key(rule.rule_index, axis_name, bound_name)
                    self._register_float_model(
                        key,
                        getter=lambda rule=rule, axis_index=axis_index, bound_index=bound_index: self._get_range_component(
                            rule, axis_index, bound_index
                        ),
                        setter=lambda value, rule=rule, axis_index=axis_index, bound_index=bound_index: self._set_range_component(
                            rule, axis_index, bound_index, value
                        ),
                    )

        for field_name in ("position", "target"):
            for index, axis_name in enumerate(("x", "y", "z")):
                key = f"camera.{field_name}.{axis_name}"
                self._register_float_model(
                    key,
                    getter=lambda field_name=field_name, index=index: float(
                        ensure_vector(self.viewport_camera.get(field_name), 3, 0.0)[index]
                    ),
                    setter=lambda value, field_name=field_name, index=index: self._set_vector_component(
                        self.viewport_camera, field_name, index, value
                    ),
                    transient=True,
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
        self.float_setters[key](float(model.get_value_as_float()))
        if key in self.transient_float_keys:
            self._apply_viewport_camera(sync_models=False)
            self._set_status(f"Updated {key} (viewport only, not saved).")
            return
        self._apply_model_to_stage(sync_models=True)
        self._mark_dirty(f"Updated {key}")

    def _range_key(self, rule_index: int, axis_name: str, bound_name: str) -> str:
        return f"range.{rule_index}.{axis_name}.{bound_name}"

    def _get_range_component(self, rule: RelativeRangeRule, axis_index: int, bound_index: int) -> float:
        range_values = normalize_range(rule.params.get("relative_position_range"))
        return float(range_values[axis_index][bound_index])

    def _set_range_component(self, rule: RelativeRangeRule, axis_index: int, bound_index: int, value: float) -> None:
        range_values = normalize_range(rule.params.get("relative_position_range"))
        range_values[axis_index][bound_index] = float(value)
        range_values[axis_index] = sorted(range_values[axis_index])
        rule.params["relative_position_range"] = range_values

    def _set_vector_component(self, container: dict[str, Any], field: str, index: int, value: float) -> None:
        container[field] = ensure_vector(container.get(field), 3, 0.0)
        container[field][index] = float(value)

    def _sync_models_from_task(self) -> None:
        self.suspend_model_callbacks = True
        try:
            for key, model in self.float_models.items():
                model.set_value(float(self.float_getters[key]()))
        finally:
            self.suspend_model_callbacks = False
        self.ui_dirty = True

    def _select_prim_path(self, prim_path: str) -> None:
        try:
            self.selection.set_selected_prim_paths([prim_path], False)
        except TypeError:
            self.selection.set_selected_prim_paths([prim_path], False, "")
        self._set_status(f"Selected prim: {prim_path}")

    def _apply_viewport_camera(self, sync_models: bool = True) -> None:
        position = ensure_vector(self.viewport_camera.get("position"), 3, 0.0)
        target = ensure_vector(self.viewport_camera.get("target"), 3, 0.0)
        set_camera_view(eye=position, target=target, camera_prim_path="/OmniverseKit_Persp")
        if sync_models:
            self._sync_models_from_task()

    def _frame_camera(self) -> None:
        points: list[np.ndarray] = []
        for previews in self.target_previews_by_id.values():
            for preview in previews:
                points.append(preview.world_matrix[:3, 3])
        for overlay in self.overlay_specs:
            if overlay.target_preview is None:
                continue
            range_values = normalize_range(overlay.rule.params.get("relative_position_range"))
            lows = np.asarray([axis_range[0] for axis_range in range_values], dtype=np.float64)
            highs = np.asarray([axis_range[1] for axis_range in range_values], dtype=np.float64)
            for x in (lows[0], highs[0]):
                for y in (lows[1], highs[1]):
                    for z in (lows[2], highs[2]):
                        point = overlay.target_preview.world_matrix @ np.array([x, y, z, 1.0], dtype=np.float64)
                        points.append(point[:3])
        if not points:
            return
        stacked = np.stack(points)
        mins = np.min(stacked, axis=0)
        maxs = np.max(stacked, axis=0)
        center = (mins + maxs) * 0.5
        span = max(float(np.max(maxs - mins)), 0.45)
        self.viewport_camera["target"] = [float(center[0]), float(center[1]), float(center[2])]
        self.viewport_camera["position"] = [
            float(center[0] + span * 2.2),
            float(center[1] + span * 1.7),
            float(center[2] + span * 1.4),
        ]
        self._apply_viewport_camera(sync_models=True)
        self._set_status("Viewport camera framed around target and success range.")

    def _save_task(self) -> None:
        save_data = copy.deepcopy(self.task_data)
        if not self.origin_explicit_in_file and self._is_identity_origin(save_data["origin"]):
            save_data.pop("origin", None)
        dump_json(self.output_path, save_data)
        self.dirty = False
        self._set_status(f"Saved task JSON to {self.output_path}")

    def _is_identity_origin(self, origin: dict[str, Any]) -> bool:
        position = np.asarray(ensure_vector(origin.get("position"), 3, 0.0), dtype=np.float64)
        quaternion = np.asarray(normalize_quaternion_wxyz(origin.get("quaternion")), dtype=np.float64)
        return np.allclose(position, np.zeros(3), atol=1e-8) and np.allclose(
            quaternion, np.asarray(IDENTITY_QUATERNION), atol=1e-8
        )

    def _reload_from_disk(self) -> None:
        self._load_task_from_disk()
        self._create_float_models()
        self._rebuild_stage()
        self._sync_models_from_task()
        self._set_status(f"Reloaded task JSON from {self.task_path}")

    def _refresh_preview(self) -> None:
        self._apply_model_to_stage(sync_models=True)
        self._set_status("Preview refreshed from current task model.")

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
            ui.Label(f"Dirty: {'yes' if self.dirty else 'no'} | Assets: {self.assets_root}", word_wrap=True, height=22)
            ui.Label(self.status_message, word_wrap=True, height=44)

            with ui.HStack(height=30, spacing=8):
                ui.Button("Save", width=82, clicked_fn=self._save_task)
                ui.Button("Reload", width=82, clicked_fn=self._reload_from_disk)
                ui.Button("Refresh", width=82, clicked_fn=self._refresh_preview)
                ui.Button("Frame All", width=92, clicked_fn=self._frame_camera)

            ui.Separator(height=6)
            self._build_camera_ui()
            ui.Separator(height=8)
            self._build_metric_ui()

    def _build_camera_ui(self) -> None:
        ui.Label("Viewport Camera (Not Saved)", height=22)
        with ui.HStack(height=28, spacing=8):
            ui.Button("Apply Camera", width=116, clicked_fn=lambda: self._apply_viewport_camera(sync_models=True))
            ui.Button("Frame All", width=92, clicked_fn=self._frame_camera)
        self._build_vector_row("Position", [f"camera.position.{axis}" for axis in ("x", "y", "z")], ("x", "y", "z"))
        self._build_vector_row("Target", [f"camera.target.{axis}" for axis in ("x", "y", "z")], ("x", "y", "z"))

    def _build_metric_ui(self) -> None:
        if not self.metric_rules:
            ui.Label("No is_object_relative_position_in_target rules found.", word_wrap=True, height=44)
            return
        ui.Label("Success Ranges", height=22)
        ui.Label(
            "Green boxes are target-local ranges. Only these boxes and the metric target object are shown by default.",
            word_wrap=True,
            height=44,
        )
        for rule in self.metric_rules:
            overlays = [overlay for overlay in self.overlay_specs if overlay.rule is rule]
            target_previews = self.target_previews_by_id.get(rule.target_id, [])
            with ui.HStack(height=28, spacing=8):
                ui.Label(f"Rule {rule.rule_index}", width=70)
                if overlays:
                    ui.Button(
                        "Select Range",
                        width=112,
                        clicked_fn=lambda prim_path=overlays[0].prim_path: self._select_prim_path(prim_path),
                    )
                if target_previews:
                    ui.Button(
                        "Select Target",
                        width=112,
                        clicked_fn=lambda prim_path=target_previews[0].prim_path: self._select_prim_path(prim_path),
                    )
            ui.Label(f"Target: {rule.target_id}", word_wrap=True, height=24)
            ui.Label(f"Objects: {', '.join(rule.object_ids) or 'none'}", word_wrap=True, height=24)
            for preview in target_previews:
                ui.Label(f"Target pose: {preview.note}", word_wrap=True, height=24)
            if not target_previews:
                ui.Label("Target preview not found in task_related_objects; range is hidden.", word_wrap=True, height=36)

            self._build_vector_row(
                "Range X",
                [self._range_key(rule.rule_index, "x", "min"), self._range_key(rule.rule_index, "x", "max")],
                ("min", "max"),
            )
            self._build_vector_row(
                "Range Y",
                [self._range_key(rule.rule_index, "y", "min"), self._range_key(rule.rule_index, "y", "max")],
                ("min", "max"),
            )
            self._build_vector_row(
                "Range Z",
                [self._range_key(rule.rule_index, "z", "min"), self._range_key(rule.rule_index, "z", "max")],
                ("min", "max"),
            )
            ui.Separator(height=6)

    def _build_vector_row(self, title: str, keys: list[str], axis_labels: tuple[str, ...]) -> None:
        with ui.HStack(height=26, spacing=4):
            ui.Label(title, width=92)
            for axis_label, key in zip(axis_labels, keys, strict=True):
                ui.Label(axis_label, width=22)
                ui.FloatField(model=self.float_models[key], width=88)

    def run(self) -> None:
        while simulation_app.is_running():
            if self.ui_dirty:
                self._rebuild_ui()
            simulation_app.update()

    def close(self) -> None:
        simulation_app.close()


def main() -> None:
    visualizer = RelativePositionRangeVisualizer(ARGS)
    try:
        visualizer.run()
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
