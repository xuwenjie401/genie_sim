"""
Interactive Isaac Sim browser for labeled interaction poses.

This viewer:
- loads benchmark objects from `objects/benchmark`
- loads interaction labels from `interaction/<object_id>/interaction.json`
- renders passive grasp PKLs as gripper geometry
- renders place / twist / push / cut / pour primitives as bracket-style pose markers
- exposes an Isaac Sim control window with prev/next selectors

Pose convention for grasp geometry is fixed:
    local +x = approach
    local +y = width
    local +z = top
"""

from __future__ import annotations

import argparse
import json
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import isaacsim  # noqa: F401

from omni.isaac.kit import SimulationApp


DEFAULT_ASSET_ROOT = Path("/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets")
DEFAULT_OBJECTS_ROOT = DEFAULT_ASSET_ROOT / "objects" / "benchmark"
DEFAULT_INTERACTION_ROOT = DEFAULT_ASSET_ROOT / "interaction"
ALL_FILTER = "__all__"
SCENE_ROOT = "/World/InteractionPoseBrowser"
OBJECT_PATH = f"{SCENE_ROOT}/Object"
DEBUG_ROOT = f"{OBJECT_PATH}/InteractionDebug"
UI_TITLE = "Interaction Pose Browser"
ROLE_ORDER = ("active", "passive")
POSE_TYPE_ORDER = ("grasp", "place", "twist", "push", "cut", "pour")


parser = argparse.ArgumentParser(description="Isaac Sim browser for benchmark interaction pose labels")
parser.add_argument("--objects_root", type=str, default=str(DEFAULT_OBJECTS_ROOT))
parser.add_argument("--interaction_root", type=str, default=str(DEFAULT_INTERACTION_ROOT))
parser.add_argument("--object_id", type=str, default="benchmark_storage_box_000")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--max_grasps", type=int, default=0, help="0 means show all grasps")
parser.add_argument("--finger_len", type=float, default=0.04)
parser.add_argument("--handle_len", type=float, default=0.08)
parser.add_argument("--grasp_thickness", type=float, default=0.003)
parser.add_argument("--axis_len", type=float, default=0.05)
parser.add_argument("--axis_thickness", type=float, default=0.0016)
parser.add_argument("--primitive_arrow_len", type=float, default=0.08)
parser.add_argument("--primitive_thickness", type=float, default=0.0035)
parser.add_argument("--primitive_marker_size", type=float, default=0.008)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"width": args.width, "height": args.height, "headless": False})

import omni.ui as ui
import omni.usd
import numpy as np
from isaacsim.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, Sdf, UsdGeom, UsdLux


def sorted_unique(values: list[str], preferred_order: tuple[str, ...] = ()) -> list[str]:
    seen = set(values)
    ordered: list[str] = []
    for item in preferred_order:
        if item in seen:
            ordered.append(item)
            seen.remove(item)
    ordered.extend(sorted(seen))
    return ordered


def filter_label(value: str) -> str:
    return "all" if value == ALL_FILTER else value


def sanitize_token(value: str) -> str:
    sanitized = "".join(ch if ch.isascii() and (ch.isalnum() or ch == "_") else "_" for ch in value)
    sanitized = sanitized.strip("_")
    return sanitized or "item"


def set_display_color(prim, rgb: np.ndarray) -> None:
    color = np.asarray(rgb, dtype=np.float32).reshape(3)
    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set(
        [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
    )


def remove_prim_if_exists(stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


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

    t_op = xformable.AddXformOp(
        UsdGeom.XformOp.TypeTranslate,
        UsdGeom.XformOp.PrecisionDouble,
        opSuffix="translateLocal",
    )
    t_op.Set(Gf.Vec3d(float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2])))

    s_op = xformable.AddXformOp(
        UsdGeom.XformOp.TypeScale,
        UsdGeom.XformOp.PrecisionDouble,
        opSuffix="scaleLocal",
    )
    s_op.Set(Gf.Vec3d(float(s_xyz[0]), float(s_xyz[1]), float(s_xyz[2])))


def build_rotation_from_x_axis(direction: np.ndarray) -> np.ndarray:
    x_axis = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(x_axis))
    if norm <= 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / norm

    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(x_axis, helper))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    y_axis = np.cross(helper, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-8:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_axis = np.cross(helper, x_axis)
        y_norm = float(np.linalg.norm(y_axis))
    y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / float(np.linalg.norm(z_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def group_base_color(role: str, pose_type: str) -> np.ndarray:
    palette = {
        ("passive", "grasp"): np.array([0.18, 0.74, 0.44], dtype=np.float32),
        ("active", "place"): np.array([0.93, 0.57, 0.19], dtype=np.float32),
        ("passive", "place"): np.array([0.24, 0.53, 0.93], dtype=np.float32),
        ("passive", "twist"): np.array([0.81, 0.38, 0.85], dtype=np.float32),
        ("active", "twist"): np.array([0.79, 0.49, 0.82], dtype=np.float32),
        ("passive", "push"): np.array([0.89, 0.27, 0.28], dtype=np.float32),
        ("active", "pour"): np.array([0.30, 0.71, 0.79], dtype=np.float32),
        ("passive", "pour"): np.array([0.20, 0.74, 0.80], dtype=np.float32),
        ("passive", "cut"): np.array([0.93, 0.79, 0.24], dtype=np.float32),
    }
    return palette.get((role, pose_type), np.array([0.62, 0.62, 0.62], dtype=np.float32))


def vary_color(base: np.ndarray, index: int, total: int) -> np.ndarray:
    if total <= 1:
        return np.asarray(base, dtype=np.float32)
    phase = index / float(max(total - 1, 1))
    jitter = np.array([0.08 * phase, 0.05 * (1.0 - phase), 0.07 * (0.5 - phase)], dtype=np.float32)
    return np.clip(np.asarray(base, dtype=np.float32) + jitter, 0.0, 1.0)


@dataclass
class BrowserState:
    object_id: str
    role_filter: str = ALL_FILTER
    pose_type_filter: str = ALL_FILTER
    primitive_filter: str = ALL_FILTER
    revision: int = 0


@dataclass
class CatalogEntry:
    object_id: str
    category: str
    usd_path: str
    object_dir: str
    interaction_dir: str
    interaction_json: str
    availability: dict[str, dict[str, list[str]]]
    size: list[float]


class InteractionCatalog:
    def __init__(self, objects_root: Path, interaction_root: Path) -> None:
        self.objects_root = objects_root
        self.interaction_root = interaction_root
        self.entries: dict[str, CatalogEntry] = {}
        self._interaction_cache: dict[str, dict[str, Any]] = {}
        self._grasp_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._build_index()

    def _build_index(self) -> None:
        object_dirs: dict[str, tuple[Path, Path]] = {}
        for category_dir in sorted(self.objects_root.iterdir()):
            if not category_dir.is_dir():
                continue
            for object_dir in sorted(category_dir.iterdir()):
                if not object_dir.is_dir():
                    continue
                usd_path = self._resolve_usd_path(object_dir)
                if usd_path is None:
                    continue
                object_dirs[object_dir.name] = (category_dir, object_dir)

        for interaction_dir in sorted(self.interaction_root.iterdir()):
            if not interaction_dir.is_dir():
                continue
            interaction_json_path = interaction_dir / "interaction.json"
            object_match = object_dirs.get(interaction_dir.name)
            if object_match is None or not interaction_json_path.exists():
                continue

            category_dir, object_dir = object_match
            with interaction_json_path.open("r", encoding="utf-8") as f:
                interaction_data = json.load(f).get("interaction", {})
            availability = self._extract_availability(interaction_data, interaction_dir)
            if not availability:
                continue

            size = self._load_object_size(object_dir)
            self.entries[interaction_dir.name] = CatalogEntry(
                object_id=interaction_dir.name,
                category=category_dir.name,
                usd_path=str(self._resolve_usd_path(object_dir)),
                object_dir=str(object_dir),
                interaction_dir=str(interaction_dir),
                interaction_json=str(interaction_json_path),
                availability=availability,
                size=size,
            )

        if not self.entries:
            raise RuntimeError("No benchmark objects with labeled interactions were found.")

    def _resolve_usd_path(self, object_dir: Path) -> Path | None:
        for name in ("Aligned.usda", "Aligned.usd"):
            candidate = object_dir / name
            if candidate.exists():
                return candidate
        return None

    def _load_object_size(self, object_dir: Path) -> list[float]:
        object_params_path = object_dir / "object_parameters.json"
        if not object_params_path.exists():
            return [0.3, 0.3, 0.3]
        with object_params_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        size = data.get("size")
        if isinstance(size, list) and len(size) == 3:
            return [float(size[0]), float(size[1]), float(size[2])]
        return [0.3, 0.3, 0.3]

    def _extract_availability(
        self,
        interaction_data: dict[str, Any],
        interaction_dir: Path,
    ) -> dict[str, dict[str, list[str]]]:
        availability: dict[str, dict[str, list[str]]] = {}
        for role in ROLE_ORDER:
            role_data = interaction_data.get(role, {})
            if not isinstance(role_data, dict):
                continue
            type_map: dict[str, list[str]] = {}
            for pose_type in sorted_unique(list(role_data.keys()), POSE_TYPE_ORDER):
                pose_data = role_data[pose_type]
                if not isinstance(pose_data, dict):
                    continue
                primitive_names: list[str] = []
                for primitive, payload in pose_data.items():
                    if self._payload_is_usable(role, pose_type, payload, interaction_dir):
                        primitive_names.append(primitive)
                if primitive_names:
                    type_map[pose_type] = sorted_unique(primitive_names)
            if type_map:
                availability[role] = type_map
        return availability

    def _payload_is_usable(self, role: str, pose_type: str, payload: Any, interaction_dir: Path) -> bool:
        if pose_type == "grasp" and role == "passive":
            paths = payload if isinstance(payload, list) else [payload]
            for rel_path in paths:
                if not isinstance(rel_path, str):
                    continue
                if (interaction_dir / rel_path).exists():
                    return True
            return False
        if not isinstance(payload, list):
            return False
        for item in payload:
            if not isinstance(item, dict):
                continue
            if "xyz" in item and "direction" in item:
                return True
        return False

    def object_ids(self) -> list[str]:
        return sorted(self.entries.keys())

    def default_object_id(self, preferred: str) -> str:
        if preferred in self.entries:
            return preferred
        return self.object_ids()[0]

    def get_entry(self, object_id: str) -> CatalogEntry:
        return self.entries[object_id]

    def load_interaction(self, object_id: str) -> dict[str, Any]:
        if object_id not in self._interaction_cache:
            entry = self.entries[object_id]
            with Path(entry.interaction_json).open("r", encoding="utf-8") as f:
                self._interaction_cache[object_id] = json.load(f).get("interaction", {})
        return self._interaction_cache[object_id]

    def load_grasp_pkl(self, pkl_path: Path) -> tuple[np.ndarray, np.ndarray]:
        key = str(pkl_path)
        if key not in self._grasp_cache:
            with pkl_path.open("rb") as f:
                data = pickle.load(f)
            T = np.asarray(data["grasp_pose"], dtype=np.float64)
            width = np.asarray(data["width"], dtype=np.float64)
            if T.ndim != 3 or T.shape[1:] != (4, 4):
                raise ValueError(f"grasp_pose should be (N,4,4), got {T.shape}")
            if width.ndim != 1 or width.shape[0] != T.shape[0]:
                raise ValueError(f"width should be (N,), got {width.shape}")
            self._grasp_cache[key] = (T, width)
        return self._grasp_cache[key]

    def available_roles(self, object_id: str) -> list[str]:
        entry = self.entries[object_id]
        return sorted_unique(list(entry.availability.keys()), ROLE_ORDER)

    def available_pose_types(self, object_id: str, role_filter: str) -> list[str]:
        entry = self.entries[object_id]
        values: list[str] = []
        if role_filter == ALL_FILTER:
            for role in entry.availability:
                values.extend(entry.availability[role].keys())
        else:
            values.extend(entry.availability.get(role_filter, {}).keys())
        return sorted_unique(values, POSE_TYPE_ORDER)

    def available_primitives(self, object_id: str, role_filter: str, pose_type_filter: str) -> list[str]:
        entry = self.entries[object_id]
        values: list[str] = []
        roles = entry.availability.keys() if role_filter == ALL_FILTER else [role_filter]
        for role in roles:
            by_type = entry.availability.get(role, {})
            types = by_type.keys() if pose_type_filter == ALL_FILTER else [pose_type_filter]
            for pose_type in types:
                values.extend(by_type.get(pose_type, []))
        return sorted_unique(values)

    def normalize_state(self, state: BrowserState) -> BrowserState:
        object_id = self.default_object_id(state.object_id)
        role_filter = state.role_filter
        pose_type_filter = state.pose_type_filter
        primitive_filter = state.primitive_filter

        valid_roles = [ALL_FILTER, *self.available_roles(object_id)]
        if role_filter not in valid_roles:
            role_filter = ALL_FILTER

        valid_types = [ALL_FILTER, *self.available_pose_types(object_id, role_filter)]
        if pose_type_filter not in valid_types:
            pose_type_filter = ALL_FILTER

        valid_primitives = [ALL_FILTER, *self.available_primitives(object_id, role_filter, pose_type_filter)]
        if primitive_filter not in valid_primitives:
            primitive_filter = ALL_FILTER

        return BrowserState(
            object_id=object_id,
            role_filter=role_filter,
            pose_type_filter=pose_type_filter,
            primitive_filter=primitive_filter,
            revision=state.revision,
        )

    def selected_groups(self, state: BrowserState) -> list[dict[str, Any]]:
        normalized = self.normalize_state(state)
        interaction = self.load_interaction(normalized.object_id)
        results: list[dict[str, Any]] = []
        roles = self.available_roles(normalized.object_id)
        for role in roles:
            if normalized.role_filter != ALL_FILTER and role != normalized.role_filter:
                continue
            role_data = interaction.get(role, {})
            for pose_type in sorted_unique(list(role_data.keys()), POSE_TYPE_ORDER):
                if normalized.pose_type_filter != ALL_FILTER and pose_type != normalized.pose_type_filter:
                    continue
                pose_data = role_data.get(pose_type, {})
                if not isinstance(pose_data, dict):
                    continue
                for primitive in sorted_unique(list(pose_data.keys())):
                    if normalized.primitive_filter != ALL_FILTER and primitive != normalized.primitive_filter:
                        continue
                    payload = pose_data[primitive]
                    if pose_type == "grasp" and role == "passive":
                        grasp_paths = payload if isinstance(payload, list) else [payload]
                        pkl_paths: list[Path] = []
                        pose_count = 0
                        for rel_path in grasp_paths:
                            abs_path = Path(self.entries[normalized.object_id].interaction_dir) / rel_path
                            if not abs_path.exists():
                                continue
                            T, _ = self.load_grasp_pkl(abs_path)
                            if T.shape[0] == 0:
                                continue
                            pose_count += int(T.shape[0])
                            pkl_paths.append(abs_path)
                        if pkl_paths:
                            results.append(
                                {
                                    "kind": "grasp",
                                    "role": role,
                                    "pose_type": pose_type,
                                    "primitive": primitive,
                                    "pkl_paths": pkl_paths,
                                    "count": pose_count,
                                }
                            )
                    else:
                        primitives = []
                        if isinstance(payload, list):
                            for item in payload:
                                if isinstance(item, dict) and "xyz" in item and "direction" in item:
                                    primitives.append(item)
                        if primitives:
                            results.append(
                                {
                                    "kind": "primitive",
                                    "role": role,
                                    "pose_type": pose_type,
                                    "primitive": primitive,
                                    "items": primitives,
                                    "count": len(primitives),
                                }
                            )
        return results

    def selection_summary(self, state: BrowserState) -> str:
        normalized = self.normalize_state(state)
        groups = self.selected_groups(normalized)
        group_count = len(groups)
        pose_count = sum(int(group["count"]) for group in groups)
        parts = [
            normalized.object_id,
            f"role={filter_label(normalized.role_filter)}",
            f"type={filter_label(normalized.pose_type_filter)}",
            f"label={filter_label(normalized.primitive_filter)}",
        ]
        return f"{', '.join(parts)} | {group_count} groups | {pose_count} poses"

class InteractionPoseBrowser:
    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()
        self.catalog = InteractionCatalog(Path(args.objects_root), Path(args.interaction_root))
        initial_state = BrowserState(object_id=self.catalog.default_object_id(args.object_id))
        self._state = self.catalog.normalize_state(initial_state)
        self._state_lock = threading.Lock()
        self._applied_revision = -1
        self._render_summary = ""
        self._ui_dirty = True
        self._window = None
        self._loaded_object_id: str | None = None
        self.running = True
        self._warm_up()
        self._ensure_scene_roots()
        self._ensure_light()
        self._build_ui_window()
        self._apply_state(force=True)

    def _warm_up(self) -> None:
        for _ in range(20):
            simulation_app.update()

    def _ensure_scene_roots(self) -> None:
        create_prim("/World", prim_type="Xform")
        create_prim(SCENE_ROOT, prim_type="Xform")

    def _ensure_light(self) -> None:
        light_path = Sdf.Path(f"{SCENE_ROOT}/KeyLight")
        if not self.stage.GetPrimAtPath(light_path):
            light = UsdLux.SphereLight.Define(self.stage, light_path)
            light.CreateIntensityAttr(70000.0)
            light.CreateRadiusAttr(0.3)
            xformable = UsdGeom.Xformable(self.stage.GetPrimAtPath(light_path))
            translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            translate_op.Set(Gf.Vec3d(1.0, 1.4, 1.8))

    def _build_ui_window(self) -> None:
        self._window = ui.Window(UI_TITLE, width=430, height=330)
        self._rebuild_ui()

    def _rebuild_ui(self) -> None:
        state = self.snapshot_state()
        entry = self.catalog.get_entry(state.object_id)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label(UI_TITLE, height=24)
                ui.Label(self._current_summary(state), word_wrap=True, height=36)
                ui.Separator(height=6)
                self._build_cycle_row("Object", state.object_id, self._cycle_object, -1, 1)
                self._build_cycle_row("Role", filter_label(state.role_filter), self._cycle_role, -1, 1)
                self._build_cycle_row("Type", filter_label(state.pose_type_filter), self._cycle_pose_type, -1, 1)
                self._build_cycle_row("Label", filter_label(state.primitive_filter), self._cycle_primitive, -1, 1)
                with ui.HStack(height=32, spacing=8):
                    ui.Button("Show All", clicked_fn=self._show_all_for_object)
                    ui.Button("Refresh", clicked_fn=self._force_refresh)
                ui.Separator(height=6)
                ui.Label(f"Object category: {entry.category}", height=22)
                ui.Label("Controls: Isaac panel only", word_wrap=True, height=22)
        self._ui_dirty = False

    def _build_cycle_row(self, title: str, value: str, callback, left_delta: int, right_delta: int) -> None:
        with ui.HStack(height=28, spacing=6):
            ui.Label(title, width=70)
            ui.Button("<", width=28, clicked_fn=lambda delta=left_delta: callback(delta))
            ui.Label(value, width=260)
            ui.Button(">", width=28, clicked_fn=lambda delta=right_delta: callback(delta))

    def snapshot_state(self) -> BrowserState:
        with self._state_lock:
            return BrowserState(
                object_id=self._state.object_id,
                role_filter=self._state.role_filter,
                pose_type_filter=self._state.pose_type_filter,
                primitive_filter=self._state.primitive_filter,
                revision=self._state.revision,
            )

    def _current_summary(self, state: BrowserState) -> str:
        if state.revision == self._applied_revision and self._render_summary:
            return self._render_summary
        return self.catalog.selection_summary(state)

    def update_state_from_payload(self, payload: dict[str, Any]) -> None:
        current = self.snapshot_state()
        candidate = BrowserState(
            object_id=str(payload.get("object_id", current.object_id)),
            role_filter=str(payload.get("role_filter", current.role_filter)),
            pose_type_filter=str(payload.get("pose_type_filter", current.pose_type_filter)),
            primitive_filter=str(payload.get("primitive_filter", current.primitive_filter)),
            revision=current.revision,
        )
        self._set_state(candidate)

    def _set_state(self, candidate: BrowserState) -> None:
        normalized = self.catalog.normalize_state(candidate)
        with self._state_lock:
            changed = (
                normalized.object_id != self._state.object_id
                or normalized.role_filter != self._state.role_filter
                or normalized.pose_type_filter != self._state.pose_type_filter
                or normalized.primitive_filter != self._state.primitive_filter
            )
            if not changed:
                return
            normalized.revision = self._state.revision + 1
            self._state = normalized
            self._ui_dirty = True

    def _cycle_value(self, current: str, options: list[str], delta: int) -> str:
        if not options:
            return current
        idx = options.index(current) if current in options else 0
        return options[(idx + delta) % len(options)]

    def _cycle_object(self, delta: int) -> None:
        state = self.snapshot_state()
        options = self.catalog.object_ids()
        object_id = self._cycle_value(state.object_id, options, delta)
        self._set_state(BrowserState(object_id=object_id))

    def _cycle_role(self, delta: int) -> None:
        state = self.snapshot_state()
        options = [ALL_FILTER, *self.catalog.available_roles(state.object_id)]
        role_filter = self._cycle_value(state.role_filter, options, delta)
        self._set_state(
            BrowserState(
                object_id=state.object_id,
                role_filter=role_filter,
                pose_type_filter=state.pose_type_filter,
                primitive_filter=state.primitive_filter,
            )
        )

    def _cycle_pose_type(self, delta: int) -> None:
        state = self.snapshot_state()
        options = [ALL_FILTER, *self.catalog.available_pose_types(state.object_id, state.role_filter)]
        pose_type_filter = self._cycle_value(state.pose_type_filter, options, delta)
        self._set_state(
            BrowserState(
                object_id=state.object_id,
                role_filter=state.role_filter,
                pose_type_filter=pose_type_filter,
                primitive_filter=state.primitive_filter,
            )
        )

    def _cycle_primitive(self, delta: int) -> None:
        state = self.snapshot_state()
        options = [ALL_FILTER, *self.catalog.available_primitives(state.object_id, state.role_filter, state.pose_type_filter)]
        primitive_filter = self._cycle_value(state.primitive_filter, options, delta)
        self._set_state(
            BrowserState(
                object_id=state.object_id,
                role_filter=state.role_filter,
                pose_type_filter=state.pose_type_filter,
                primitive_filter=primitive_filter,
            )
        )

    def _show_all_for_object(self) -> None:
        state = self.snapshot_state()
        self._set_state(BrowserState(object_id=state.object_id))

    def _force_refresh(self) -> None:
        self._ui_dirty = True
        self._applied_revision = -1

    def _apply_state(self, force: bool = False) -> None:
        state = self.snapshot_state()
        if not force and state.revision == self._applied_revision:
            return

        entry = self.catalog.get_entry(state.object_id)
        self._load_object(entry, force=force)
        self._build_debug_geometry(state)
        self._render_summary = self.catalog.selection_summary(state)
        self._applied_revision = state.revision
        self._ui_dirty = True
        print(f"[apply] {self._render_summary}")

    def _load_object(self, entry: CatalogEntry, force: bool = False) -> None:
        if force or self._loaded_object_id != entry.object_id or not self.stage.GetPrimAtPath(OBJECT_PATH):
            remove_prim_if_exists(self.stage, OBJECT_PATH)
            obj_prim = create_prim(OBJECT_PATH, prim_type="Xform")
            obj_prim.GetReferences().AddReference(entry.usd_path)
            self._loaded_object_id = entry.object_id
            for _ in range(10):
                simulation_app.update()

        size = np.asarray(entry.size, dtype=np.float64)
        max_dim = float(np.max(size)) if size.size == 3 else 0.3
        dist = max(0.35, max_dim * 3.2)
        set_camera_view(
            eye=[dist, dist * 0.7, dist * 0.95],
            target=[0.0, 0.0, 0.0],
            camera_prim_path="/OmniverseKit_Persp",
        )

    def _build_debug_geometry(self, state: BrowserState) -> None:
        remove_prim_if_exists(self.stage, DEBUG_ROOT)
        create_prim(DEBUG_ROOT, prim_type="Xform")

        groups = self.catalog.selected_groups(state)
        for group_index, group in enumerate(groups):
            group_path = (
                f"{DEBUG_ROOT}/"
                f"group_{group_index:02d}_{sanitize_token(group['role'])}_{sanitize_token(group['pose_type'])}"
            )
            root_prim = create_prim(group_path, prim_type="Xform")
            base_color = group_base_color(group["role"], group["pose_type"])
            if group["kind"] == "grasp":
                self._build_grasp_group(root_prim, group, base_color)
            else:
                self._build_primitive_group(root_prim, group, base_color)

        for _ in range(3):
            simulation_app.update()

    def _build_grasp_group(self, root_prim, group: dict[str, Any], base_color: np.ndarray) -> None:
        widths_all = []
        poses_all = []
        for pkl_path in group["pkl_paths"]:
            T, width = self.catalog.load_grasp_pkl(pkl_path)
            if T.shape[0] == 0:
                continue
            poses_all.append(T)
            widths_all.append(width)
        if not poses_all:
            return
        T_batch = np.concatenate(poses_all, axis=0)
        width_batch = np.concatenate(widths_all, axis=0)

        if args.max_grasps > 0:
            T_batch = T_batch[: args.max_grasps]
            width_batch = width_batch[: args.max_grasps]

        for i, (T, width) in enumerate(zip(T_batch, width_batch, strict=True)):
            grasp_path = f"{root_prim.GetPath()}/grasp_{i:04d}"
            grasp_prim = create_prim(grasp_path, prim_type="Xform")
            self._build_one_grasp(grasp_prim, float(width), vary_color(base_color, i, len(T_batch)))
            set_local_matrix(grasp_prim, T)

    def _build_one_grasp(self, root_prim, width: float, color: np.ndarray) -> None:
        self._build_bracket_marker(
            root_prim=root_prim,
            width=width,
            color=color,
            finger_len=float(args.finger_len),
            handle_len=float(args.handle_len),
            thickness=float(args.grasp_thickness),
            contact_scale=None,
        )

    def _build_bracket_marker(
        self,
        root_prim,
        width: float,
        color: np.ndarray,
        finger_len: float,
        handle_len: float,
        thickness: float,
        contact_scale: float | None,
    ) -> None:
        grasp_path = str(root_prim.GetPath())
        width = float(max(width, 1e-6))
        t = float(thickness)

        cube_specs = [
            ("handle", (-handle_len / 2.0, 0.0, 0.0), (handle_len, t, t)),
            ("bar", (0.0, 0.0, 0.0), (t, width, t)),
            ("finger_L", (finger_len / 2.0, -width / 2.0, 0.0), (finger_len, t, t)),
            ("finger_R", (finger_len / 2.0, width / 2.0, 0.0), (finger_len, t, t)),
        ]
        for name, translation, scale in cube_specs:
            prim = create_prim(f"{grasp_path}/{name}", prim_type="Cube")
            UsdGeom.Cube(prim).CreateSizeAttr().Set(1.0)
            set_local_translate_scale(prim, translation, scale)
            set_display_color(prim, color)

        if contact_scale is not None and contact_scale > 0.0:
            marker = create_prim(f"{grasp_path}/contact", prim_type="Cube")
            UsdGeom.Cube(marker).CreateSizeAttr().Set(1.0)
            set_local_translate_scale(marker, (0.0, 0.0, 0.0), (contact_scale, contact_scale, contact_scale))
            set_display_color(marker, np.clip(color + 0.18, 0.0, 1.0))

        axis_specs = [
            ("x_axis", (args.axis_len / 2.0, 0.0, 0.0), (args.axis_len, args.axis_thickness, args.axis_thickness), np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("y_axis", (0.0, args.axis_len / 2.0, 0.0), (args.axis_thickness, args.axis_len, args.axis_thickness), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ("z_axis", (0.0, 0.0, args.axis_len / 2.0), (args.axis_thickness, args.axis_thickness, args.axis_len), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ]
        axis_root = create_prim(f"{grasp_path}/axes", prim_type="Xform")
        for name, translation, scale, axis_color in axis_specs:
            prim = create_prim(f"{axis_root.GetPath()}/{name}", prim_type="Cube")
            UsdGeom.Cube(prim).CreateSizeAttr().Set(1.0)
            set_local_translate_scale(prim, translation, scale)
            set_display_color(prim, axis_color)

    def _build_primitive_group(self, root_prim, group: dict[str, Any], base_color: np.ndarray) -> None:
        items = group["items"]
        pose_finger_len = max(float(args.primitive_arrow_len) * 0.45, 0.02)
        pose_handle_len = max(float(args.primitive_arrow_len) * 0.55, 0.028)
        pose_width = max(float(args.primitive_arrow_len) * 0.42, float(args.primitive_marker_size) * 3.6)
        contact_scale = float(args.primitive_marker_size)

        for i, item in enumerate(items):
            xyz = np.asarray(item["xyz"], dtype=np.float64)
            direction = np.asarray(item["direction"], dtype=np.float64)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-8:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                direction = direction / direction_norm

            pose_path = f"{root_prim.GetPath()}/pose_{i:04d}"
            pose_prim = create_prim(pose_path, prim_type="Xform")
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = build_rotation_from_x_axis(direction)
            T[:3, 3] = xyz
            set_local_matrix(pose_prim, T)

            color = vary_color(base_color, i, len(items))
            self._build_bracket_marker(
                root_prim=pose_prim,
                width=pose_width,
                color=color,
                finger_len=pose_finger_len,
                handle_len=pose_handle_len,
                thickness=float(args.primitive_thickness),
                contact_scale=contact_scale,
            )

    def run(self) -> None:
        print("=" * 88)
        print("Interaction Pose Browser")
        print("=" * 88)
        print(f"Objects root     : {args.objects_root}")
        print(f"Interaction root : {args.interaction_root}")
        print("External UI      : disabled")
        print(f"Labeled objects  : {len(self.catalog.object_ids())}")
        print("Grasp convention : +x approach, +y width, +z top")
        print("=" * 88)

        while simulation_app.is_running() and self.running:
            if self.snapshot_state().revision != self._applied_revision:
                self._apply_state()
            if self._ui_dirty:
                self._rebuild_ui()
            simulation_app.update()

        self.close()

    def close(self) -> None:
        simulation_app.close()


def main() -> None:
    browser = InteractionPoseBrowser()
    browser.run()


if __name__ == "__main__":
    main()
