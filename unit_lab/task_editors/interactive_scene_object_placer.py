#!/usr/bin/env python3
"""Interactive Isaac Sim UI for adding small objects to a background scene.

The tool is meant for SLAM scene dressing: load a mostly-empty background USD,
pick benchmark objects from an Isaac side panel, drag a draft object with the
normal viewport gizmo, then confirm it. Confirmation enables rigid-body physics
and gravity, so the object either settles on valid collision geometry or falls
through missing/incorrect colliders.

Example:
    python unit_lab/task_editors/interactive_scene_object_placer.py \
        --task-json source/data_collection/tasks/diy/slam/galbot_slam_home_b.json

Placements are loaded from and saved to a lightweight USDA scene wrapper. There
is no sidecar layout JSON; the editor view and direct USD opening share the same
source of truth.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass


DEFAULT_ASSET_ROOT = Path("/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets")
DEFAULT_TASK_JSON = Path(
    "/home/agxi/RealityLab/genie_sim/source/data_collection/tasks/diy/slam/galbot_slam_home_b.json"
)
DEFAULT_ROBOT_CFG_DIR = Path("/home/agxi/RealityLab/genie_sim/source/data_collection/config/robot_cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Isaac Sim background object placer")
    parser.add_argument("--task-json", type=Path, default=DEFAULT_TASK_JSON, help="Optional task JSON to read scene/robot info from.")
    parser.add_argument("--scene-usd", type=str, default="", help="Scene USD relative to SIM_ASSETS/assets root. Overrides task scene.")
    parser.add_argument("--assets-root", type=Path, default=Path(os.environ.get("SIM_ASSETS", DEFAULT_ASSET_ROOT)))
    parser.add_argument("--objects-root", type=Path, default=None, help="Object catalog root. Defaults to assets-root/objects/benchmark.")
    parser.add_argument("--robot-cfg-dir", type=Path, default=DEFAULT_ROBOT_CFG_DIR)
    parser.add_argument("--stage-usd", type=Path, default=None, help="Optional scene USDA save path.")
    parser.add_argument("--object-id", type=str, default="", help="Initial object id or substring.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--physics-step", type=int, default=60)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--load-robot", action="store_true", help="Load robot from task JSON. Off by default for scene editing.")
    parser.add_argument("--hide-robot", action="store_true", help="Do not load robot from task JSON.")
    parser.add_argument("--apply-robot-joints", action="store_true", help="Best-effort application of task robot joint poses.")
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
import omni.kit.commands
import omni.ui as ui
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage
from isaacsim.core.utils.viewports import set_camera_view
from omni.physx.scripts import utils as physx_utils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

try:
    from isaacsim.gui.components.element_wrappers import ScrollingWindow
except Exception:  # pragma: no cover - Isaac Sim runtime dependent
    ScrollingWindow = None


ROOT_DIR = Path(__file__).resolve().parents[2]
COLLECTION_DIR = ROOT_DIR / "source" / "data_collection"
for candidate in (ROOT_DIR, COLLECTION_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from source.data_collection.common.base_utils.transform_utils import axis_to_quaternion, mat2quat_wxyz, quat2mat_wxyz
from source.data_collection.server.robot import RobotCfg


UI_TITLE = "GenieSim Scene Object Placer"
SCENE_ROOT = "/World/SceneObjectPlacer"
PLACED_ROOT = f"{SCENE_ROOT}/PlacedObjects"
LIGHT_PATH = f"{SCENE_ROOT}/EditorLight"
EDITOR_CAMERA_PRIM_PATH = "/OmniverseKit_Persp"

IDENTITY_QUATERNION = [1.0, 0.0, 0.0, 0.0]
DEFAULT_SPAWN_OFFSET = np.asarray([0.9, 0.0, 0.75], dtype=np.float64)
DEFAULT_CAMERA_EYE = [2.7, 2.4, 1.8]
DEFAULT_CAMERA_TARGET = [0.4, 0.0, 0.7]
MAX_PLACEMENT_ROWS = 14


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"item_{cleaned}"
    return cleaned or "item"


def normalize_relative_path(path_like: str) -> str:
    return str(path_like).strip().replace("\\", "/").lstrip("./").lstrip("/")


def normalize_relative_dir(path_like: str) -> str:
    cleaned = normalize_relative_path(path_like)
    if cleaned and not cleaned.endswith("/"):
        cleaned += "/"
    return cleaned


def ensure_vector(value: Any, length: int, default: float = 0.0) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result = [float(item) for item in value[:length]]
        if len(result) < length:
            result.extend([float(default)] * (length - len(result)))
        return result
    if value is None:
        return [float(default)] * length
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return [float(default)] * length
    return [scalar] * length


def normalize_quaternion_wxyz(value: Any) -> list[float]:
    quat = np.asarray(ensure_vector(value, 4, 0.0), dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return IDENTITY_QUATERNION.copy()
    return (quat / norm).tolist()


def normalize_scale(value: Any, default: float = 1.0) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        values = ensure_vector(value, 3, default)
        return [max(float(item), 1e-6) for item in values]
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        scalar = float(default)
    scalar = max(scalar, 1e-6)
    return [scalar, scalar, scalar]


def pose_matrix(position: list[float] | np.ndarray, quaternion_wxyz: list[float] | np.ndarray, scale=None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat2mat_wxyz(np.asarray(normalize_quaternion_wxyz(quaternion_wxyz), dtype=np.float64))
    if scale is not None:
        matrix[:3, :3] = matrix[:3, :3] @ np.diag(np.asarray(normalize_scale(scale), dtype=np.float64))
    matrix[:3, 3] = np.asarray(ensure_vector(position, 3, 0.0), dtype=np.float64)
    return matrix


def matrix_to_pose_scale(matrix: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    position = np.asarray(matrix[:3, 3], dtype=np.float64).tolist()
    linear = np.asarray(matrix[:3, :3], dtype=np.float64)
    scale = np.linalg.norm(linear, axis=0)
    scale[scale < 1e-8] = 1.0
    rotation = linear / scale.reshape(1, 3)
    if np.linalg.det(rotation) < 0.0:
        scale[0] *= -1.0
        rotation[:, 0] *= -1.0
    quaternion = normalize_quaternion_wxyz(mat2quat_wxyz(rotation).tolist())
    return position, quaternion, [float(item) for item in scale.tolist()]


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


def ensure_prim_path(stage, prim_path: str, leaf_type: str = "Xform"):
    path = Sdf.Path(prim_path)
    if path.isEmpty or not path.IsAbsolutePath():
        raise ValueError(f"Invalid absolute prim path: {prim_path!r}")
    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        return existing

    current = Sdf.Path.absoluteRootPath
    parts = [part for part in str(path).split("/") if part]
    for index, part in enumerate(parts):
        current = current.AppendChild(part)
        prim = stage.GetPrimAtPath(current)
        if prim and prim.IsValid():
            continue
        prim_type = leaf_type if index == len(parts) - 1 else "Xform"
        stage.DefinePrim(current, prim_type)
    return stage.GetPrimAtPath(path)


def remove_prim_if_exists(stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


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


def default_stage_path(scene_usd_path: Path | None, scene_usd: str | None) -> Path:
    if scene_usd_path is not None:
        if scene_usd_path.stem.endswith("_objects"):
            return scene_usd_path
        return scene_usd_path.with_name(scene_usd_path.stem + "_objects.usda")
    scene_stem = Path(scene_usd).stem if scene_usd else "scene"
    return ROOT_DIR / "source" / "data_collection" / "tasks" / "diy" / "slam" / f"{scene_stem}_objects.usda"


def base_scene_path_for_export(scene_usd_path: Path | None) -> Path | None:
    if scene_usd_path is None:
        return None
    if scene_usd_path.stem.endswith("_objects"):
        base_stem = scene_usd_path.stem[: -len("_objects")]
        base_path = scene_usd_path.with_name(base_stem + scene_usd_path.suffix)
        if base_path.exists():
            return base_path
    return scene_usd_path


def reference_path_for_usd(target_path: Path, layer_path: Path) -> str:
    relative = os.path.relpath(target_path, layer_path.parent).replace(os.sep, "/")
    if not relative.startswith("."):
        relative = f"./{relative}"
    return relative


def up_axis_to_z_quaternion(up_axis: Any) -> list[float]:
    if isinstance(up_axis, (list, tuple)) and up_axis:
        raw = str(up_axis[0]).strip().lower()
    else:
        raw = str(up_axis or "y").strip().lower()
    upside_down = raw.startswith("-")
    axis = raw[1:] if raw.startswith(("-", "+")) else raw
    if axis not in {"x", "y", "z"}:
        axis = "y"
    try:
        quat = axis_to_quaternion(axis, "z", upside_down)
    except Exception:
        quat = np.asarray(IDENTITY_QUATERNION, dtype=np.float64)
    return normalize_quaternion_wxyz(quat.tolist())


@dataclass(frozen=True)
class AssetRecord:
    object_id: str
    category: str
    data_info_dir: str
    object_dir: Path
    usd_path: Path
    size: list[float]
    scale: list[float]
    mass: float
    up_axis: Any
    semantic_name: str

    @property
    def label(self) -> str:
        semantic = self.semantic_name.strip()
        if semantic and semantic != self.object_id:
            return f"{self.object_id} ({semantic})"
        return self.object_id


class AssetCatalog:
    def __init__(self, asset_root: Path, objects_root: Path) -> None:
        self.asset_root = asset_root.resolve()
        self.objects_root = objects_root.resolve()
        self.entries: dict[str, AssetRecord] = {}
        self.by_data_info_dir: dict[str, AssetRecord] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.objects_root.exists():
            raise FileNotFoundError(f"Object catalog root does not exist: {self.objects_root}")

        for category_dir in sorted(self.objects_root.iterdir()):
            if not category_dir.is_dir():
                continue
            for object_dir in sorted(category_dir.iterdir()):
                if not object_dir.is_dir():
                    continue
                params = self._load_object_params(object_dir)
                usd_path = self._resolve_usd_path(object_dir, params)
                if usd_path is None:
                    continue
                object_id = object_dir.name
                try:
                    data_info_dir = normalize_relative_dir(str(object_dir.relative_to(self.asset_root)))
                except ValueError:
                    data_info_dir = normalize_relative_dir(str(object_dir))
                llm = params.get("llm_descriptions", {}) if isinstance(params.get("llm_descriptions"), dict) else {}
                semantic = params.get("semantic_name") or llm.get("semantic_name") or object_id
                if isinstance(semantic, list):
                    semantic = str(semantic[0]) if semantic else object_id
                record = AssetRecord(
                    object_id=object_id,
                    category=category_dir.name,
                    data_info_dir=data_info_dir,
                    object_dir=object_dir,
                    usd_path=usd_path,
                    size=ensure_vector(params.get("size"), 3, 0.2),
                    scale=normalize_scale(params.get("scale", 1.0)),
                    mass=float(params.get("mass") or 0.05),
                    up_axis=params.get("upAxis", ["y"]),
                    semantic_name=str(semantic),
                )
                self.entries[object_id] = record
                self.by_data_info_dir[data_info_dir] = record
        if not self.entries:
            raise RuntimeError(f"No object assets with Aligned.usd were found under {self.objects_root}")

    def _load_object_params(self, object_dir: Path) -> dict[str, Any]:
        params_path = object_dir / "object_parameters.json"
        if not params_path.exists():
            return {}
        try:
            return load_json(params_path)
        except Exception as exc:
            carb.log_warn(f"Failed to load object parameters from {params_path}: {exc}")
            return {}

    def _resolve_usd_path(self, object_dir: Path, params: dict[str, Any]) -> Path | None:
        for name in ("Aligned.usd", "model.usd", "Aligned.usda", "model.usda"):
            candidate = object_dir / name
            if candidate.exists():
                return candidate
        model_path = params.get("model_path")
        if model_path:
            for root in (self.asset_root, object_dir):
                resolved = resolve_path(root, str(model_path))
                if resolved is not None and resolved.exists():
                    return resolved
        return None

    def categories(self) -> list[str]:
        return sorted({entry.category for entry in self.entries.values()})

    def object_ids(self) -> list[str]:
        return sorted(self.entries.keys())

    def object_ids_in_category(self, category: str) -> list[str]:
        return sorted([object_id for object_id, entry in self.entries.items() if entry.category == category])

    def get(self, object_id: str) -> AssetRecord:
        return self.entries[object_id]

    def default_object_id(self, preferred: str = "") -> str:
        if preferred in self.entries:
            return preferred
        if preferred:
            lowered = preferred.lower()
            matches = [object_id for object_id in self.object_ids() if lowered in object_id.lower()]
            if matches:
                return matches[0]
        return self.object_ids()[0]

    def default_object_id_in_category(self, category: str, preferred: str | None = None) -> str:
        object_ids = self.object_ids_in_category(category)
        if preferred in object_ids:
            return str(preferred)
        if object_ids:
            return object_ids[0]
        return self.default_object_id(preferred or "")

    def cycle_category(self, current: str, delta: int) -> str:
        categories = self.categories()
        index = categories.index(current) if current in categories else 0
        return categories[(index + delta) % len(categories)]

    def cycle_object_in_category(self, category: str, current: str, delta: int) -> str:
        object_ids = self.object_ids_in_category(category)
        if not object_ids:
            return self.default_object_id(current)
        index = object_ids.index(current) if current in object_ids else 0
        return object_ids[(index + delta) % len(object_ids)]

    def record_from_data_info_dir(self, data_info_dir: str) -> AssetRecord | None:
        normalized = normalize_relative_dir(data_info_dir)
        if normalized in self.by_data_info_dir:
            return self.by_data_info_dir[normalized]
        object_id = Path(normalized.rstrip("/")).name
        return self.entries.get(object_id)


@dataclass
class PlacementRecord:
    placement_id: str
    source_object_id: str
    category: str
    data_info_dir: str
    prim_path: str
    position: list[float]
    quaternion: list[float]
    scale: list[float]
    mass: float
    model_type: str = "convexDecomposition"
    confirmed: bool = False
    kinematic: bool = True
    physics_body_path: str | None = None
    body_local_matrix: np.ndarray | None = None
    physics_configured: bool = False
    last_world_matrix: np.ndarray | None = None

    def pose_matrix(self) -> np.ndarray:
        return pose_matrix(self.position, self.quaternion, self.scale)


class SceneObjectPlacer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.asset_root = args.assets_root.resolve()
        objects_root = args.objects_root if args.objects_root is not None else self.asset_root / "objects" / "benchmark"
        if not objects_root.is_absolute():
            objects_root = self.asset_root / objects_root
        self.objects_root = objects_root.resolve()
        self.robot_cfg_dir = args.robot_cfg_dir.resolve()
        self.task_path = args.task_json.resolve() if args.task_json and args.task_json.exists() else None
        self.task_data: dict[str, Any] = load_json(self.task_path) if self.task_path else {}
        self.scene_usd_rel = self._resolve_scene_usd_rel()
        self.scene_usd_path = resolve_path(self.asset_root, self.scene_usd_rel)
        self.export_base_scene_path = base_scene_path_for_export(self.scene_usd_path)
        self.stage_path = (args.stage_usd or default_stage_path(self.scene_usd_path, self.scene_usd_rel)).resolve()

        self.window: ui.Window | None = None
        self._dock_task = None
        self.stage = None
        self.world: World | None = None
        self.selection = omni.usd.get_context().get_selection()

        self.catalog = AssetCatalog(self.asset_root, self.objects_root)
        self.selected_object_id = self.catalog.default_object_id(args.object_id)
        self.selected_category = self.catalog.get(self.selected_object_id).category
        self.category_search_model = ui.SimpleStringModel(self.selected_category)
        self.object_search_model = ui.SimpleStringModel(self.selected_object_id)

        self.placements: list[PlacementRecord] = []
        self.draft_id: str | None = None
        self.selected_placement_id: str | None = None
        self.status_message = "Ready."
        self.dirty = False
        self.ui_dirty = True
        self._placement_counter = 0
        self._pending_actions: list[tuple[str, Callable[[], None]]] = []

        self.robot_cfg: RobotCfg | None = None
        self.robot_usd_path: Path | None = None
        self.robot_init_pose = self.task_data.get("robot", {}).get("robot_init_pose", {}) if self.task_data else {}
        self._resolve_robot()

        self._warm_up()
        self._create_stage_and_world()
        self._build_scene_contents()
        self._build_ui_window()
        self._frame_camera()
        self._set_status(
            f"Loaded scene {self.scene_usd_rel}. Add a draft, move it with the viewport gizmo, then Confirm / Drop."
        )

    def _warm_up(self) -> None:
        for _ in range(20):
            simulation_app.update()

    def _resolve_scene_usd_rel(self) -> str | None:
        if self.args.scene_usd:
            return normalize_relative_path(self.args.scene_usd)
        scene_usd = choose_scene_usd(self.task_data.get("scene", {}).get("scene_usd")) if self.task_data else None
        return normalize_relative_path(scene_usd) if scene_usd else None

    def _resolve_robot(self) -> None:
        if self.args.hide_robot or not self.args.load_robot or not self.task_data:
            return
        robot_cfg_file = self.task_data.get("robot", {}).get("robot_cfg")
        if not robot_cfg_file:
            return
        robot_cfg_path = self.robot_cfg_dir / str(robot_cfg_file)
        if not robot_cfg_path.exists():
            self._set_status(f"Robot config missing: {robot_cfg_path}")
            return
        self.robot_cfg = RobotCfg(str(robot_cfg_path))
        self.robot_usd_path = resolve_path(self.asset_root, self.robot_cfg.robot_usd)

    def _create_stage_and_world(self) -> None:
        create_new_stage()
        for _ in range(5):
            simulation_app.update()
        self.stage = omni.usd.get_context().get_stage()
        physics_dt = 1.0 / max(float(self.args.physics_step), 1.0)
        rendering_dt = 1.0 / max(float(self.args.render_fps), 1.0)
        self.world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=rendering_dt, device="cpu")

    def _build_scene_contents(self) -> None:
        if self.scene_usd_path is not None and self.scene_usd_path.exists():
            add_reference_to_stage(str(self.scene_usd_path), "/World")
        else:
            create_prim("/World", prim_type="Xform")
            if self.scene_usd_path is not None:
                carb.log_warn(f"Scene USD missing: {self.scene_usd_path}")

        ensure_prim_path(self.stage, SCENE_ROOT)
        ensure_prim_path(self.stage, PLACED_ROOT)
        self._ensure_editor_light()
        self._ensure_physics_scene()
        self._discover_placements_from_stage()
        self._load_robot_reference()

        if self.world is not None:
            self.world.reset()
            self.world.play()

    def _ensure_editor_light(self) -> None:
        light = UsdLux.SphereLight.Define(self.stage, LIGHT_PATH)
        light.CreateIntensityAttr(65000.0)
        light.CreateRadiusAttr(0.3)
        light_prim = self.stage.GetPrimAtPath(LIGHT_PATH)
        set_local_matrix(light_prim, pose_matrix([1.2, 1.6, 2.4], IDENTITY_QUATERNION))

    def _ensure_physics_scene(self) -> None:
        physics_scene = UsdPhysics.Scene.Define(self.stage, "/physicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    def _load_robot_reference(self) -> None:
        if self.robot_cfg is None or self.robot_usd_path is None:
            return
        if not self.robot_usd_path.exists():
            carb.log_warn(f"Robot USD missing: {self.robot_usd_path}")
            return
        add_reference_to_stage(str(self.robot_usd_path), self.robot_cfg.robot_prim_path)
        robot_pose = self._robot_pose_entry()
        robot_prim = self.stage.GetPrimAtPath(self.robot_cfg.robot_prim_path)
        if robot_prim and robot_prim.IsValid() and robot_pose:
            set_local_matrix(
                robot_prim,
                pose_matrix(
                    ensure_vector(robot_pose.get("position"), 3, 0.0),
                    normalize_quaternion_wxyz(robot_pose.get("quaternion")),
                ),
            )
        if self.args.apply_robot_joints:
            self._try_apply_robot_joint_pose()

    def _robot_pose_entry(self) -> dict[str, Any] | None:
        if not isinstance(self.robot_init_pose, dict):
            return None
        if "position" in self.robot_init_pose:
            return self.robot_init_pose
        scene_key = str(self.task_data.get("scene", {}).get("scene_id", "")).rstrip("/").split("/")[-1]
        candidate = self.robot_init_pose.get(scene_key)
        return candidate if isinstance(candidate, dict) else None

    def _try_apply_robot_joint_pose(self) -> None:
        if self.robot_cfg is None or self.world is None:
            return
        joint_pose: dict[str, float] = {}
        robot_entry = self.task_data.get("robot", {})
        for key in ("fixed_joint_reset_pose", "init_joint_pose", "init_arm_pose"):
            value = robot_entry.get(key)
            if isinstance(value, dict):
                for joint_name, joint_value in value.items():
                    try:
                        joint_pose[str(joint_name)] = float(joint_value)
                    except (TypeError, ValueError):
                        pass
        if not joint_pose:
            return
        try:
            articulation = Articulation(prim_path=self.robot_cfg.robot_prim_path, name="scene_object_placer_robot")
            self.world.scene.add(articulation)
            self.world.reset()
            articulation.initialize()
            indices = []
            values = []
            for joint_name, joint_value in joint_pose.items():
                dof_index = articulation.get_dof_index(joint_name)
                if dof_index < 0:
                    continue
                indices.append(int(dof_index))
                values.append(float(joint_value))
            if indices:
                articulation.set_joint_positions(np.asarray(values, dtype=np.float64), joint_indices=np.asarray(indices, dtype=np.int64))
        except Exception as exc:  # pragma: no cover - Isaac runtime dependent
            carb.log_warn(f"Failed to apply robot joint pose: {exc}")

    def _build_ui_window(self) -> None:
        kwargs = {
            "title": UI_TITLE,
            "width": 470,
            "height": 0,
            "visible": True,
            "dockPreference": ui.DockPreference.LEFT_BOTTOM,
        }
        self.window = ScrollingWindow(**kwargs) if ScrollingWindow is not None else ui.Window(**kwargs)
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
                except Exception as exc:  # pragma: no cover - Isaac runtime dependent
                    carb.log_warn(f"Failed to dock placer into {target_name}: {exc}")

        self._dock_task = asyncio.ensure_future(dock_window())

    def _set_status(self, message: str) -> None:
        self.status_message = message
        self.ui_dirty = True
        carb.log_info(message)

    def _mark_dirty(self) -> None:
        self.dirty = True
        self.ui_dirty = True

    def _queue_action(self, label: str, callback: Callable[[], None]) -> None:
        self._pending_actions.append((label, callback))
        self._set_status(f"Queued: {label}")

    def _run_pending_actions(self) -> None:
        if not self._pending_actions:
            return
        pending = self._pending_actions
        self._pending_actions = []
        for label, callback in pending:
            try:
                callback()
            except Exception as exc:
                carb.log_error(f"Queued action failed ({label}): {exc}")
                self._set_status(f"Action failed ({label}): {exc}")

    def _string_model_value(self, model) -> str:
        try:
            return str(model.get_value_as_string())
        except Exception:
            return str(getattr(model, "as_string", ""))

    def _set_string_model_value(self, model, value: str) -> None:
        try:
            model.set_value(str(value))
        except Exception:
            try:
                model.as_string = str(value)
            except Exception:
                pass

    def _discover_placements_from_stage(self) -> None:
        self.placements = []
        placed_root = self.stage.GetPrimAtPath(PLACED_ROOT)
        if not placed_root or not placed_root.IsValid():
            return

        loaded = 0
        for child_prim in placed_root.GetChildren():
            asset = self._asset_record_from_placement_prim(child_prim)
            if asset is None:
                carb.log_warn(f"Skip existing placement with unknown asset reference: {child_prim.GetPath()}")
                continue

            placement_id = str(child_prim.GetName())
            position, quaternion, scale = matrix_to_pose_scale(compute_world_matrix(child_prim))
            record = PlacementRecord(
                placement_id=placement_id,
                source_object_id=asset.object_id,
                category=asset.category,
                data_info_dir=asset.data_info_dir,
                prim_path=str(child_prim.GetPath()),
                position=position,
                quaternion=quaternion,
                scale=scale,
                mass=asset.mass,
                confirmed=True,
                kinematic=True,
            )
            self.placements.append(record)
            self._enable_rigid_body(record, kinematic=True)
            loaded += 1
        self.dirty = False
        if loaded:
            self._set_status(f"Loaded {loaded} existing placement(s) from USDA.")

    def _asset_record_from_placement_prim(self, prim) -> AssetRecord | None:
        for resolved_path in self._placement_reference_paths(prim):
            asset = self._asset_record_from_usd_path(resolved_path)
            if asset is not None:
                return asset
        return self._asset_record_from_placement_id(str(prim.GetName()))

    def _placement_reference_paths(self, prim) -> list[Path]:
        paths: list[Path] = []
        for spec in prim.GetPrimStack():
            reference_list = getattr(spec, "referenceList", None)
            if reference_list is None:
                continue
            layer_path = self._layer_path(spec.layer)
            for list_name in ("prependedItems", "appendedItems", "addedItems", "explicitItems"):
                for reference in list(getattr(reference_list, list_name, []) or []):
                    asset_path = getattr(reference, "assetPath", "")
                    if not asset_path:
                        continue
                    candidate = Path(asset_path)
                    if not candidate.is_absolute() and layer_path is not None:
                        candidate = layer_path.parent / candidate
                    paths.append(candidate.resolve())
        return paths

    def _layer_path(self, layer) -> Path | None:
        raw_path = getattr(layer, "realPath", "") or getattr(layer, "identifier", "")
        if not raw_path or str(raw_path).startswith("anon:"):
            return None
        return Path(str(raw_path)).resolve()

    def _asset_record_from_usd_path(self, usd_path: Path) -> AssetRecord | None:
        try:
            resolved = usd_path.resolve()
        except OSError:
            resolved = usd_path
        for record in self.catalog.entries.values():
            try:
                if record.usd_path.resolve() == resolved or record.object_dir.resolve() == resolved.parent:
                    return record
            except OSError:
                continue
        try:
            data_info_dir = normalize_relative_dir(str(resolved.parent.relative_to(self.asset_root)))
        except ValueError:
            return None
        return self.catalog.record_from_data_info_dir(data_info_dir)

    def _asset_record_from_placement_id(self, placement_id: str) -> AssetRecord | None:
        matches = [
            record
            for object_id, record in self.catalog.entries.items()
            if placement_id == object_id or placement_id.startswith(object_id + "_")
        ]
        if not matches:
            return None
        return max(matches, key=lambda record: len(record.object_id))

    def _selected_asset(self) -> AssetRecord:
        return self.catalog.get(self.selected_object_id)

    def _set_selected_object(self, object_id: str) -> None:
        if object_id not in self.catalog.entries:
            return
        self.selected_object_id = object_id
        self.selected_category = self.catalog.get(object_id).category
        self._set_string_model_value(self.category_search_model, self.selected_category)
        self._set_string_model_value(self.object_search_model, self.selected_object_id)
        self.ui_dirty = True

    def _cycle_category(self, delta: int) -> None:
        self.selected_category = self.catalog.cycle_category(self.selected_category, delta)
        next_object = self.catalog.default_object_id_in_category(self.selected_category, self.selected_object_id)
        self._set_selected_object(next_object)
        self._set_status(f"Category set to {self.selected_category}.")

    def _cycle_object(self, delta: int) -> None:
        next_object = self.catalog.cycle_object_in_category(self.selected_category, self.selected_object_id, delta)
        self._set_selected_object(next_object)
        self._set_status(f"Object set to {self.selected_object_id}.")

    def _apply_category_query(self) -> None:
        query = self._string_model_value(self.category_search_model).strip()
        if not query:
            self._set_string_model_value(self.category_search_model, self.selected_category)
            self._set_status("Enter a category token before searching.")
            return
        categories = self.catalog.categories()
        category = query if query in categories else ""
        if not category:
            lowered = query.lower()
            matches = [name for name in categories if lowered in name.lower()]
            if not matches:
                self._set_status(f"No category matched {query!r}.")
                return
            category = matches[0]
        self.selected_category = category
        self._set_selected_object(self.catalog.default_object_id_in_category(category, self.selected_object_id))
        self._set_status(f"Category set to {self.selected_category}.")

    def _apply_object_query(self) -> None:
        query = self._string_model_value(self.object_search_model).strip()
        if not query:
            self._set_string_model_value(self.object_search_model, self.selected_object_id)
            self._set_status("Enter an object id or substring before searching.")
            return
        if query in self.catalog.entries:
            object_id = query
        else:
            lowered = query.lower()
            local_matches = [
                object_id for object_id in self.catalog.object_ids_in_category(self.selected_category) if lowered in object_id.lower()
            ]
            matches = local_matches or [object_id for object_id in self.catalog.object_ids() if lowered in object_id.lower()]
            if not matches:
                self._set_status(f"No object matched {query!r}.")
                return
            object_id = matches[0]
        self._set_selected_object(object_id)
        self._set_status(f"Object set to {self.selected_object_id}.")

    def _unique_placement_id(self, base: str) -> str:
        token = sanitize_token(base)
        existing = {record.placement_id for record in self.placements}
        if token not in existing:
            return token
        index = 1
        while f"{token}_{index:03d}" in existing:
            index += 1
        return f"{token}_{index:03d}"

    def _next_placement_id(self, asset: AssetRecord) -> str:
        self._placement_counter += 1
        base = f"{sanitize_token(asset.object_id)}_{self._placement_counter:03d}"
        return self._unique_placement_id(base)

    def _spawn_position(self) -> list[float]:
        robot_pose = self._robot_pose_entry()
        if robot_pose is not None:
            base = np.asarray(ensure_vector(robot_pose.get("position"), 3, 0.0), dtype=np.float64)
            return (base + DEFAULT_SPAWN_OFFSET).tolist()
        return DEFAULT_SPAWN_OFFSET.tolist()

    def _add_draft(self) -> None:
        if self.draft_id is not None:
            self._set_status("A draft already exists. Confirm / Drop or Delete it before adding another.")
            return
        asset = self._selected_asset()
        placement_id = self._next_placement_id(asset)
        record = PlacementRecord(
            placement_id=placement_id,
            source_object_id=asset.object_id,
            category=asset.category,
            data_info_dir=asset.data_info_dir,
            prim_path=f"{PLACED_ROOT}/{placement_id}",
            position=self._spawn_position(),
            quaternion=up_axis_to_z_quaternion(asset.up_axis),
            scale=asset.scale,
            mass=asset.mass,
            confirmed=False,
            kinematic=True,
        )
        self.placements.append(record)
        self.draft_id = record.placement_id
        self._create_or_update_placement_prim(record, enable_physics=True)
        self._select_record(record.placement_id)
        self._mark_dirty()
        self._set_status(f"Draft added: {record.placement_id}. Move it with the viewport gizmo, then Confirm / Drop.")

    def _create_or_update_placement_prim(self, record: PlacementRecord, enable_physics: bool) -> None:
        asset = self.catalog.record_from_data_info_dir(record.data_info_dir)
        if asset is None:
            carb.log_warn(f"Cannot create placement {record.placement_id}; missing asset {record.data_info_dir}")
            return
        parent_path = str(Sdf.Path(record.prim_path).GetParentPath())
        ensure_prim_path(self.stage, parent_path)
        remove_prim_if_exists(self.stage, record.prim_path)
        add_reference_to_stage(str(asset.usd_path), record.prim_path)
        prim = self.stage.GetPrimAtPath(record.prim_path)
        if prim and prim.IsValid():
            set_local_matrix(prim, record.pose_matrix())
            record.last_world_matrix = compute_world_matrix(prim)
            record.physics_configured = False
        if enable_physics:
            self._enable_rigid_body(record, kinematic=record.kinematic)
        else:
            self._freeze_existing_asset_bodies(record)

    def _enable_rigid_body(self, record: PlacementRecord, kinematic: bool) -> None:
        prim = self.stage.GetPrimAtPath(record.prim_path)
        if not prim or not prim.IsValid():
            self._set_status(f"Cannot enable physics; prim missing: {record.prim_path}")
            return

        target_bodies = self._physics_target_bodies(record, prim)
        if not record.physics_configured:
            self._configure_mesh_physics(record, prim)
            existing_bodies = self._topmost_rigid_body_prims(prim)
            descendant_bodies = [body for body in existing_bodies if str(body.GetPath()) != record.prim_path]
            if descendant_bodies:
                self._remove_root_rigid_body_if_nested(record.prim_path, prim)
                target_bodies = descendant_bodies
                self._remember_physics_body(record, target_bodies[0])
            else:
                try:
                    physx_utils.setRigidBody(prim, record.model_type, bool(kinematic))
                except Exception as exc:  # pragma: no cover - fallback for runtime variations
                    carb.log_warn(f"physx_utils.setRigidBody failed for {record.prim_path}: {exc}")
                    self._fallback_apply_collision_and_rigid_body(prim, kinematic)
                target_bodies = self._topmost_rigid_body_prims(prim) or [prim]
                self._remember_physics_body(record, target_bodies[0])
            mass_per_body = float(record.mass) / max(len(target_bodies), 1)
            for body in target_bodies:
                self._set_mass_api(body, mass_per_body)
            record.physics_configured = True

        for body in target_bodies:
            body_path = str(body.GetPath())
            self._set_kinematic(body_path, kinematic)
        record.kinematic = bool(kinematic)

    def _physics_target_bodies(self, record: PlacementRecord, root_prim) -> list:
        if record.physics_body_path:
            body_prim = self.stage.GetPrimAtPath(record.physics_body_path)
            if body_prim and body_prim.IsValid():
                return [body_prim]
        existing_bodies = self._topmost_rigid_body_prims(root_prim)
        descendant_bodies = [body for body in existing_bodies if str(body.GetPath()) != record.prim_path]
        return descendant_bodies or existing_bodies or [root_prim]

    def _configure_mesh_physics(
        self,
        record: PlacementRecord,
        root_prim,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
    ) -> None:
        for mesh_prim in Usd.PrimRange(root_prim):
            if not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            mesh_path = str(mesh_prim.GetPath())
            if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(mesh_prim)
            UsdPhysics.CollisionAPI(mesh_prim).CreateCollisionEnabledAttr().Set(True)
            try:
                geometry_prim = GeometryPrim(prim_path=mesh_path, reset_xform_properties=False)
                material_path = f"{mesh_path}/object_physics"
                geometry_prim.apply_physics_material(
                    PhysicsMaterial(
                        prim_path=material_path,
                        static_friction=static_friction,
                        dynamic_friction=dynamic_friction,
                        restitution=None,
                    )
                )
                material_prim = self.stage.GetPrimAtPath(material_path)
                physx_material_api = PhysxSchema.PhysxMaterialAPI(material_prim)
                if physx_material_api is not None:
                    mode_attr = physx_material_api.GetFrictionCombineModeAttr()
                    if mode_attr.Get() != "max":
                        physx_material_api.CreateFrictionCombineModeAttr().Set("max")
            except Exception as exc:
                carb.log_warn(f"Failed to apply physics material to {mesh_path}: {exc}")

    def _fallback_apply_collision_and_rigid_body(self, root_prim, kinematic: bool) -> None:
        if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(root_prim)
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                collision_api = UsdPhysics.CollisionAPI(prim)
                collision_api.CreateCollisionEnabledAttr().Set(True)
        self._set_kinematic(str(root_prim.GetPath()), kinematic)

    def _freeze_existing_asset_bodies(self, record: PlacementRecord) -> None:
        root_prim = self.stage.GetPrimAtPath(record.prim_path)
        if not root_prim or not root_prim.IsValid():
            return
        bodies = self._topmost_rigid_body_prims(root_prim)
        if not bodies:
            record.physics_body_path = None
            record.body_local_matrix = None
            return
        descendant_bodies = [body for body in bodies if str(body.GetPath()) != record.prim_path]
        if descendant_bodies:
            self._remove_root_rigid_body_if_nested(record.prim_path, root_prim)
            bodies = descendant_bodies
        self._remember_physics_body(record, bodies[0])
        for body in bodies:
            body_path = str(body.GetPath())
            self._set_kinematic(body_path, True)

    def _topmost_rigid_body_prims(self, root_prim) -> list:
        bodies = [prim for prim in Usd.PrimRange(root_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        body_paths = [body.GetPath() for body in bodies]
        root_path = root_prim.GetPath()
        topmost = []
        for body in bodies:
            path = body.GetPath()
            has_body_ancestor = any(
                path != other and other != root_path and path.HasPrefix(other)
                for other in body_paths
            )
            if not has_body_ancestor:
                topmost.append(body)
        return topmost

    def _remove_root_rigid_body_if_nested(self, root_path: str, root_prim) -> None:
        if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return
        try:
            omni.kit.commands.execute("RemovePhysicsAPI", prim_path=root_path, api=UsdPhysics.RigidBodyAPI)
        except Exception as exc:
            carb.log_warn(f"Failed to remove nested root RigidBodyAPI from {root_path}: {exc}")

    def _remember_physics_body(self, record: PlacementRecord, body_prim) -> None:
        root_prim = self.stage.GetPrimAtPath(record.prim_path)
        if not root_prim or not root_prim.IsValid() or not body_prim or not body_prim.IsValid():
            return
        record.physics_body_path = str(body_prim.GetPath())
        try:
            root_world = compute_world_matrix(root_prim)
            body_world = compute_world_matrix(body_prim)
            record.body_local_matrix = np.linalg.inv(root_world) @ body_world
        except Exception as exc:
            carb.log_warn(f"Failed to cache rigid body local matrix for {record.placement_id}: {exc}")
            record.body_local_matrix = None

    def _ensure_collision_enabled_under(self, root_prim) -> None:
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_api.CreateCollisionEnabledAttr().Set(True)

    def _set_mass_api(self, prim, mass: float) -> None:
        try:
            mass_api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr().Set(float(mass))
        except Exception as exc:
            carb.log_warn(f"Failed to set mass on {prim.GetPath()}: {exc}")

    def _set_kinematic(self, prim_path: str, kinematic: bool) -> bool:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return False
        try:
            rigid_api = UsdPhysics.RigidBodyAPI(prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI) else UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid_api.CreateKinematicEnabledAttr().Set(bool(kinematic))
            return True
        except Exception as exc:
            carb.log_warn(f"Failed to set kinematic={kinematic} on {prim_path}: {exc}")
            return False

    def _confirm_draft(self) -> None:
        record = self._draft_record()
        if record is None:
            selected = self._selected_record()
            if selected is not None:
                self._drop_record(selected)
            else:
                self._set_status("No draft selected. Add a draft first.")
            return
        self._sync_record_from_stage(record)
        record.confirmed = True
        record.kinematic = False
        self._enable_rigid_body(record, kinematic=False)
        self.draft_id = None
        if self.world is not None:
            self.world.play()
        self._mark_dirty()
        self._set_status(f"Confirmed {record.placement_id}: rigid body/colliders enabled, gravity active.")

    def _draft_record(self) -> PlacementRecord | None:
        if self.draft_id is None:
            return None
        return self._record_by_id(self.draft_id)

    def _selected_record(self) -> PlacementRecord | None:
        selection_record = self._record_from_viewport_selection()
        if selection_record is not None:
            self.selected_placement_id = selection_record.placement_id
            return selection_record
        if self.selected_placement_id is None:
            return None
        return self._record_by_id(self.selected_placement_id)

    def _record_by_id(self, placement_id: str) -> PlacementRecord | None:
        for record in self.placements:
            if record.placement_id == placement_id:
                return record
        return None

    def _record_from_viewport_selection(self) -> PlacementRecord | None:
        try:
            selected_paths = list(self.selection.get_selected_prim_paths())
        except Exception:
            return None
        if not selected_paths:
            return None
        for selected_path in selected_paths:
            selected = str(selected_path)
            matches = [
                record
                for record in self.placements
                if selected == record.prim_path or selected.startswith(record.prim_path + "/")
            ]
            if matches:
                return max(matches, key=lambda record: len(record.prim_path))
        return None

    def _select_record(self, placement_id: str) -> None:
        record = self._record_by_id(placement_id)
        if record is None:
            return
        self.selected_placement_id = placement_id
        try:
            self.selection.set_selected_prim_paths([record.prim_path], False)
        except TypeError:
            self.selection.set_selected_prim_paths([record.prim_path], False, "")
        self._set_status(f"Selected {record.placement_id}.")

    def _delete_selected(self) -> None:
        record = self._selected_record()
        if record is None:
            self._set_status("No placed object selected.")
            return
        remove_prim_if_exists(self.stage, record.prim_path)
        self.placements = [item for item in self.placements if item.placement_id != record.placement_id]
        if self.draft_id == record.placement_id:
            self.draft_id = None
        if self.selected_placement_id == record.placement_id:
            self.selected_placement_id = None
        self._mark_dirty()
        self._set_status(f"Deleted {record.placement_id}.")

    def _freeze_selected(self) -> None:
        record = self._selected_record()
        if record is None:
            self._set_status("No placed object selected.")
            return
        self._sync_record_from_stage(record)
        record.confirmed = True
        record.kinematic = True
        self._create_or_update_placement_prim(record, enable_physics=True)
        self._mark_dirty()
        self._set_status(f"Frozen {record.placement_id}. Move it, then Confirm / Drop to test gravity again.")

    def _drop_record(self, record: PlacementRecord) -> None:
        self._sync_record_from_stage(record)
        record.confirmed = True
        record.kinematic = False
        self._enable_rigid_body(record, kinematic=False)
        if self.world is not None:
            self.world.play()
        self._mark_dirty()
        self._set_status(f"Dropped {record.placement_id}: gravity active.")

    def _sync_record_from_stage(self, record: PlacementRecord) -> None:
        matrix = self._current_record_matrix(record)
        if matrix is None:
            return
        position, quaternion, scale = matrix_to_pose_scale(matrix)
        record.position = position
        record.quaternion = quaternion
        record.scale = scale
        record.last_world_matrix = matrix

    def _current_record_matrix(self, record: PlacementRecord) -> np.ndarray | None:
        root_prim = self.stage.GetPrimAtPath(record.prim_path)
        if not root_prim or not root_prim.IsValid():
            return None

        body_prim = None
        if record.physics_body_path and record.body_local_matrix is not None:
            candidate = self.stage.GetPrimAtPath(record.physics_body_path)
            if candidate and candidate.IsValid():
                body_prim = candidate

        if body_prim is not None and record.physics_body_path != record.prim_path:
            try:
                return compute_world_matrix(body_prim) @ np.linalg.inv(record.body_local_matrix)
            except Exception as exc:
                carb.log_warn(f"Failed to derive root pose from rigid body {record.physics_body_path}: {exc}")
        return compute_world_matrix(root_prim)

    def _poll_placement_transforms(self) -> None:
        if self.stage is None:
            return
        for record in self.placements:
            matrix = self._current_record_matrix(record)
            if matrix is None:
                continue
            previous = record.last_world_matrix
            if previous is not None and np.max(np.abs(matrix - previous)) <= 1e-5:
                continue
            position, quaternion, scale = matrix_to_pose_scale(matrix)
            record.position = position
            record.quaternion = quaternion
            record.scale = scale
            record.last_world_matrix = matrix
            self.dirty = True

    def _bake_record_pose_to_stage(self, record: PlacementRecord) -> None:
        root_prim = self.stage.GetPrimAtPath(record.prim_path)
        if not root_prim or not root_prim.IsValid():
            return
        set_local_matrix(root_prim, record.pose_matrix())
        if (
            record.physics_body_path
            and record.physics_body_path != record.prim_path
            and record.body_local_matrix is not None
        ):
            body_prim = self.stage.GetPrimAtPath(record.physics_body_path)
            if body_prim and body_prim.IsValid():
                set_local_matrix(body_prim, record.body_local_matrix)
                self._clear_physics_velocity_attrs(body_prim)

    def _clear_physics_velocity_attrs(self, prim) -> None:
        for attr_name in ("physics:velocity", "physics:angularVelocity"):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.HasAuthoredValueOpinion():
                attr.Clear()

    def _save_stage(self) -> None:
        self._poll_placement_transforms()
        confirmed = [record for record in self.placements if record.confirmed]
        for record in confirmed:
            self._bake_record_pose_to_stage(record)
        self.stage_path.parent.mkdir(parents=True, exist_ok=True)

        temp_stage_path = self.stage_path.with_name(
            f".{self.stage_path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}{self.stage_path.suffix}"
        )
        export_stage = Usd.Stage.CreateNew(str(temp_stage_path))
        export_stage.SetStartTimeCode(0)
        export_stage.SetEndTimeCode(1000000)
        export_stage.SetTimeCodesPerSecond(60)
        UsdGeom.SetStageUpAxis(export_stage, UsdGeom.Tokens.z)

        world_prim = UsdGeom.Xform.Define(export_stage, "/World").GetPrim()
        export_stage.SetDefaultPrim(world_prim)
        if self.export_base_scene_path is not None:
            world_prim.GetReferences().AddReference(reference_path_for_usd(self.export_base_scene_path, self.stage_path))

        ensure_prim_path(export_stage, SCENE_ROOT)
        ensure_prim_path(export_stage, PLACED_ROOT)
        for record in confirmed:
            asset = self.catalog.record_from_data_info_dir(record.data_info_dir)
            if asset is None:
                carb.log_warn(f"Skip export of {record.placement_id}; missing asset {record.data_info_dir}")
                continue
            placement_prim = UsdGeom.Xform.Define(export_stage, record.prim_path).GetPrim()
            placement_prim.GetReferences().AddReference(reference_path_for_usd(asset.usd_path, self.stage_path))
            set_local_matrix(placement_prim, record.pose_matrix())
            self._author_exported_rigid_body_overrides(export_stage, record)
            self._author_exported_collision_overrides(export_stage, record)

        root_layer = export_stage.GetRootLayer()
        try:
            root_layer.Save()
            os.replace(temp_stage_path, self.stage_path)
        except Exception:
            if temp_stage_path.exists():
                temp_stage_path.unlink()
            raise
        self.dirty = False
        self._set_status(f"Exported lightweight scene USDA with background reference to {self.stage_path}.")

    def _author_exported_rigid_body_overrides(self, export_stage, record: PlacementRecord) -> None:
        body_path = record.physics_body_path or record.prim_path
        body_prim = export_stage.OverridePrim(body_path)
        try:
            rigid_api = (
                UsdPhysics.RigidBodyAPI(body_prim)
                if body_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                else UsdPhysics.RigidBodyAPI.Apply(body_prim)
            )
            rigid_api.CreateRigidBodyEnabledAttr().Set(True)
            rigid_api.CreateKinematicEnabledAttr().Set(False)
        except Exception:
            body_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
            body_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        try:
            mass_api = (
                UsdPhysics.MassAPI(body_prim)
                if body_prim.HasAPI(UsdPhysics.MassAPI)
                else UsdPhysics.MassAPI.Apply(body_prim)
            )
            mass_api.CreateMassAttr().Set(float(record.mass))
        except Exception:
            body_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(float(record.mass))
        body_prim.CreateAttribute("physics:velocity", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        body_prim.CreateAttribute("physics:angularVelocity", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        self._prepend_api_schemas(body_prim, ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI"])

    def _author_exported_collision_overrides(self, export_stage, record: PlacementRecord) -> None:
        root_prim = self.stage.GetPrimAtPath(record.prim_path)
        if not root_prim or not root_prim.IsValid():
            return
        for mesh_prim in Usd.PrimRange(root_prim):
            if not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            export_mesh = export_stage.OverridePrim(str(mesh_prim.GetPath()))
            try:
                collision_api = (
                    UsdPhysics.CollisionAPI(export_mesh)
                    if export_mesh.HasAPI(UsdPhysics.CollisionAPI)
                    else UsdPhysics.CollisionAPI.Apply(export_mesh)
                )
                collision_api.CreateCollisionEnabledAttr().Set(True)
            except Exception:
                export_mesh.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
            try:
                mesh_collision_api = (
                    UsdPhysics.MeshCollisionAPI(export_mesh)
                    if export_mesh.HasAPI(UsdPhysics.MeshCollisionAPI)
                    else UsdPhysics.MeshCollisionAPI.Apply(export_mesh)
                )
                mesh_collision_api.CreateApproximationAttr().Set(record.model_type)
            except Exception:
                export_mesh.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(record.model_type)
            self._prepend_api_schemas(
                export_mesh,
                [
                    "PhysicsCollisionAPI",
                    "PhysicsMeshCollisionAPI",
                    "PhysxCollisionAPI",
                    "PhysxConvexDecompositionCollisionAPI",
                    "PhysxConvexHullCollisionAPI",
                ],
            )

    def _prepend_api_schemas(self, prim, schema_names: list[str]) -> None:
        list_op = Sdf.TokenListOp()
        current = []
        try:
            current = list(prim.GetAppliedSchemas())
        except Exception:
            pass
        list_op.prependedItems = list(dict.fromkeys([*schema_names, *current]))
        prim.SetMetadata("apiSchemas", list_op)

    def _redock_window(self) -> None:
        self._schedule_dock()
        self._set_status("Requested panel re-dock.")

    def _frame_camera(self) -> None:
        points: list[np.ndarray] = []
        robot_pose = self._robot_pose_entry()
        if robot_pose is not None:
            points.append(np.asarray(ensure_vector(robot_pose.get("position"), 3, 0.0), dtype=np.float64))
        for record in self.placements:
            points.append(np.asarray(record.position, dtype=np.float64))
        if points:
            stacked = np.stack(points)
            center = (np.min(stacked, axis=0) + np.max(stacked, axis=0)) * 0.5
            span = max(float(np.max(np.max(stacked, axis=0) - np.min(stacked, axis=0))), 1.0)
            eye = [center[0] + span * 1.8, center[1] + span * 1.4, center[2] + span * 1.2 + 0.4]
            target = [center[0], center[1], center[2] + 0.35]
        else:
            eye = DEFAULT_CAMERA_EYE
            target = DEFAULT_CAMERA_TARGET
        set_camera_view(eye=eye, target=target, camera_prim_path=EDITOR_CAMERA_PRIM_PATH)
        self._set_status("Viewport camera framed current placements.")

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
        selected_asset = self._selected_asset()
        selected_record = self._selected_record()
        confirmed_count = sum(1 for record in self.placements if record.confirmed)
        draft_text = self.draft_id or "none"
        with ui.VStack(spacing=8, height=0):
            ui.Label(UI_TITLE, height=24)
            ui.Label(f"Scene: {self.scene_usd_rel}", word_wrap=True, height=34)
            ui.Label(f"USDA: {self.stage_path}", word_wrap=True, height=34)
            ui.Label(f"Dirty: {'yes' if self.dirty else 'no'} | confirmed={confirmed_count} draft={draft_text}", height=22)
            ui.Label(self.status_message, word_wrap=True, height=46)

            with ui.HStack(height=30, spacing=7):
                ui.Button("Save Scene USDA", width=138, clicked_fn=self._save_stage)
                ui.Button("Frame All", width=84, clicked_fn=self._frame_camera)
                ui.Button("Re-dock", width=78, clicked_fn=self._redock_window)

            ui.Separator(height=8)
            ui.Label("Asset Picker", height=22)
            self._build_cycle_row("Category", self.selected_category, self._cycle_category, width=250)
            with ui.HStack(height=28, spacing=6):
                ui.Label("Find Cat", width=76)
                if hasattr(ui, "StringField"):
                    ui.StringField(model=self.category_search_model, width=238)
                else:
                    ui.Label(self.selected_category, width=238)
                ui.Button("Find", width=58, clicked_fn=self._apply_category_query)
            self._build_cycle_row("Object", self.selected_object_id, self._cycle_object, width=250)
            with ui.HStack(height=28, spacing=6):
                ui.Label("Find Obj", width=76)
                if hasattr(ui, "StringField"):
                    ui.StringField(model=self.object_search_model, width=238)
                else:
                    ui.Label(self.selected_object_id, width=238)
                ui.Button("Find", width=58, clicked_fn=self._apply_object_query)

            ui.Label(
                f"{selected_asset.label} | size={self._fmt_vec(selected_asset.size)} | mass={selected_asset.mass:.3f}",
                word_wrap=True,
                height=40,
            )
            ui.Label(selected_asset.data_info_dir, word_wrap=True, height=34)
            with ui.HStack(height=30, spacing=8):
                ui.Button("Add Draft", width=110, clicked_fn=lambda: self._queue_action("Add Draft", self._add_draft))
                ui.Button(
                    "Confirm / Drop",
                    width=125,
                    clicked_fn=lambda: self._queue_action("Confirm / Drop", self._confirm_draft),
                )
                ui.Button("Freeze", width=80, clicked_fn=lambda: self._queue_action("Freeze", self._freeze_selected))
                ui.Button("Delete", width=78, clicked_fn=lambda: self._queue_action("Delete", self._delete_selected))

            ui.Separator(height=8)
            ui.Label("Selected Placement", height=22)
            if selected_record is None:
                ui.Label("none", height=22)
            else:
                state = "draft" if selected_record.placement_id == self.draft_id else ("frozen" if selected_record.kinematic else "dynamic")
                ui.Label(
                    f"{selected_record.placement_id} | {state} | pos={self._fmt_vec(selected_record.position)}",
                    word_wrap=True,
                    height=42,
                )
                ui.Label(f"quat={self._fmt_vec(selected_record.quaternion)}", word_wrap=True, height=34)

            ui.Separator(height=8)
            ui.Label("Recent Placements", height=22)
            shown = list(reversed(self.placements[-MAX_PLACEMENT_ROWS:]))
            if not shown:
                ui.Label("No objects placed yet.", height=24)
            for record in shown:
                self._build_placement_row(record)

    def _build_cycle_row(self, title: str, value: str, callback, width: int = 250) -> None:
        with ui.HStack(height=28, spacing=6):
            ui.Label(title, width=76)
            ui.Button("<", width=28, clicked_fn=lambda: callback(-1))
            ui.Label(str(value), width=width, word_wrap=True)
            ui.Button(">", width=28, clicked_fn=lambda: callback(1))

    def _build_placement_row(self, record: PlacementRecord) -> None:
        state = "DRAFT" if record.placement_id == self.draft_id else ("FROZEN" if record.kinematic else "DROP")
        with ui.HStack(height=28, spacing=5):
            ui.Label(state, width=54)
            ui.Label(record.placement_id, width=205, word_wrap=True)
            ui.Button("Select", width=70, clicked_fn=lambda placement_id=record.placement_id: self._select_record(placement_id))
            ui.Button(
                "Drop",
                width=52,
                clicked_fn=lambda placement_id=record.placement_id: self._queue_action(
                    "Drop", lambda placement_id=placement_id: self._drop_by_id(placement_id)
                ),
            )

    def _drop_by_id(self, placement_id: str) -> None:
        record = self._record_by_id(placement_id)
        if record is not None:
            if self.draft_id == placement_id:
                self.draft_id = None
            self._drop_record(record)

    def _fmt_vec(self, values: Any, precision: int = 3) -> str:
        vector = ensure_vector(values, len(values) if isinstance(values, (list, tuple, np.ndarray)) else 3, 0.0)
        return "[" + ", ".join(f"{float(item):.{precision}f}" for item in vector) + "]"

    def run(self) -> None:
        render_interval = max(1, int(round(float(self.args.physics_step) / max(float(self.args.render_fps), 1.0))))
        frame = 0
        while simulation_app.is_running():
            self._run_pending_actions()
            self._poll_placement_transforms()
            if self.ui_dirty:
                self._rebuild_ui()
            if self.world is not None:
                self.world.step(render=(frame % render_interval == 0))
            else:
                simulation_app.update()
            frame += 1

    def close(self) -> None:
        simulation_app.close()


def main() -> None:
    editor = SceneObjectPlacer(ARGS)
    try:
        editor.run()
    except KeyboardInterrupt:
        pass
    finally:
        editor.close()


if __name__ == "__main__":
    main()
