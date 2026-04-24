#!/usr/bin/env python3
"""Interactive Isaac Sim editor for GenieSim interaction labels.

Features:
- browse all benchmark assets, including unlabeled ones
- load and save the existing `interaction.json` + `grasp_pose.pkl` schema
- add / delete / move primitive poses (`place`, `twist`, `push`, `cut`, `pour`)
- select a pose handle and edit it with the normal Isaac viewport gizmo
- launch a GraspGen client request from the UI and preview generated grasps
- calibrate generated grasps with axis remapping and local TCP offset
- accept generated grasps into `passive/grasp/<label>` and save to disk
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

from omni.isaac.kit import SimulationApp


DEFAULT_ASSET_ROOT = Path("/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets")
DEFAULT_OBJECTS_ROOT = DEFAULT_ASSET_ROOT / "objects" / "benchmark"
DEFAULT_INTERACTION_ROOT = DEFAULT_ASSET_ROOT / "interaction"
DEFAULT_GRASPGEN_ROOT = Path("/home/agxi/ManipLab/GraspGen")
DEFAULT_GRASPGEN_BRIDGE = Path(__file__).with_name("graspgen_client_bridge.py")
DEFAULT_GRIPPER_CONFIG = Path("/home/agxi/GraspGen/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml")
DEFAULT_TMP_DIR = Path("/tmp/genie_sim_graspgen_editor")
DEFAULT_OBJECT_ID = "benchmark_bottle_017"
DEFAULT_GRASPGEN_HOST = "127.0.0.1"
DEFAULT_GRASPGEN_PORT = 5556
DEFAULT_GRASPGEN_ENV = "GraspGen"
DEFAULT_GENERATED_GRASP_COUNT = 64
DEFAULT_APPROACH_AXIS = "+z"
DEFAULT_WIDTH_AXIS = "+x"
DEFAULT_TOP_AXIS = "+y"
DEFAULT_GRASP_LOCAL_OFFSET = (0.175, 0.0, 0.0)

ALL_FILTER = "__all__"
ROLE_ORDER = ("active", "passive")
POSE_TYPE_ORDER = ("grasp", "place", "twist", "push", "cut", "pour")
SCENE_ROOT = "/World/InteractiveInteractionPoseEditor"
OBJECT_PATH = f"{SCENE_ROOT}/Object"
OBJECT_ASSET_PREFIX = "Asset_"
OVERLAY_ROOT = f"{OBJECT_PATH}/InteractionOverlay"
HANDLE_ROOT = f"{OBJECT_PATH}/PoseHandles"
LIGHT_PATH = f"{SCENE_ROOT}/KeyLight"
UI_TITLE = "Interactive Interaction Pose Editor"
MAIN_WINDOW_HEIGHT = 940
CHOOSE_DIR_WINDOW_HEIGHT = 420
EDITOR_CAMERA_PRIM_PATH = "/OmniverseKit_Persp"
DEFAULT_EDITOR_CAMERA_TRANSLATE = (0.0, 0.28, -0.475)
DEFAULT_EDITOR_CAMERA_ROTATE_XYZ_DEG = (148.0, 0.0, 179.5)

AXES = ("+x", "-x", "+y", "-y", "+z", "-z")
AXIS_VECS = {
    "+x": None,
    "-x": None,
    "+y": None,
    "-y": None,
    "+z": None,
    "-z": None,
}


AXIS_VECS["+x"] = __import__("numpy").array([1.0, 0.0, 0.0], dtype=__import__("numpy").float64)
AXIS_VECS["-x"] = __import__("numpy").array([-1.0, 0.0, 0.0], dtype=__import__("numpy").float64)
AXIS_VECS["+y"] = __import__("numpy").array([0.0, 1.0, 0.0], dtype=__import__("numpy").float64)
AXIS_VECS["-y"] = __import__("numpy").array([0.0, -1.0, 0.0], dtype=__import__("numpy").float64)
AXIS_VECS["+z"] = __import__("numpy").array([0.0, 0.0, 1.0], dtype=__import__("numpy").float64)
AXIS_VECS["-z"] = __import__("numpy").array([0.0, 0.0, -1.0], dtype=__import__("numpy").float64)


ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR_STR = str(ROOT_DIR)
if ROOT_DIR_STR not in sys.path:
    sys.path.insert(0, ROOT_DIR_STR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Isaac Sim editor for GenieSim interaction labels")
    parser.add_argument("--objects_root", type=Path, default=DEFAULT_OBJECTS_ROOT)
    parser.add_argument("--interaction_root", type=Path, default=DEFAULT_INTERACTION_ROOT)
    parser.add_argument("--object_id", type=str, default=DEFAULT_OBJECT_ID)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--max_stored_grasps", type=int, default=96)
    parser.add_argument("--max_generated_grasps", type=int, default=DEFAULT_GENERATED_GRASP_COUNT)
    parser.add_argument("--finger_len", type=float, default=0.04)
    parser.add_argument("--handle_len", type=float, default=0.08)
    parser.add_argument("--grasp_thickness", type=float, default=0.003)
    parser.add_argument("--axis_len", type=float, default=0.05)
    parser.add_argument("--axis_thickness", type=float, default=0.0016)
    parser.add_argument("--primitive_arrow_len", type=float, default=0.08)
    parser.add_argument("--primitive_thickness", type=float, default=0.0035)
    parser.add_argument("--primitive_marker_size", type=float, default=0.008)
    parser.add_argument("--graspgen_root", type=Path, default=DEFAULT_GRASPGEN_ROOT)
    parser.add_argument("--graspgen_bridge", type=Path, default=DEFAULT_GRASPGEN_BRIDGE)
    parser.add_argument("--graspgen_gripper_config", type=Path, default=DEFAULT_GRIPPER_CONFIG)
    parser.add_argument("--graspgen_host", type=str, default=DEFAULT_GRASPGEN_HOST)
    parser.add_argument("--graspgen_port", type=int, default=DEFAULT_GRASPGEN_PORT)
    parser.add_argument("--graspgen_env", type=str, default=DEFAULT_GRASPGEN_ENV)
    parser.add_argument("--num_sample_points", type=int, default=2048)
    parser.add_argument("--num_grasps", type=int, default=120)
    parser.add_argument("--topk_num_grasps", type=int, default=DEFAULT_GENERATED_GRASP_COUNT)
    parser.add_argument("--mesh_scale", type=float, default=1.0)
    parser.add_argument("--tmp_dir", type=Path, default=DEFAULT_TMP_DIR)
    return parser.parse_known_args()[0]


ARGS = parse_args()
simulation_app = SimulationApp({"width": ARGS.width, "height": ARGS.height, "headless": False})

import numpy as np
import omni.ui as ui
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.prims import create_prim
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

try:
    from isaacsim.gui.components.element_wrappers import ScrollingWindow
except Exception:  # pragma: no cover - Isaac Sim package availability is runtime-specific
    ScrollingWindow = None

from source.data_collection.common.base_utils.transform_utils import euler2mat, mat2quat_wxyz, quat2mat_wxyz


@dataclass
class CatalogEntry:
    object_id: str
    category: str
    usd_path: Path
    object_dir: Path
    interaction_dir: Path
    interaction_json: Path
    size: list[float]
    scale: float
    has_interaction: bool


@dataclass
class AxisAssignment:
    approach: str = DEFAULT_APPROACH_AXIS
    width: str = DEFAULT_WIDTH_AXIS
    top: str = DEFAULT_TOP_AXIS

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.approach, self.width, self.top)


@dataclass
class GeneratedGraspResult:
    raw_poses: np.ndarray
    widths: np.ndarray
    confidences: np.ndarray
    source_path: Path


@dataclass
class HandleRecord:
    index: int
    prim_path: str
    last_world_matrix: np.ndarray | None = None


@dataclass
class GraspGenJob:
    object_id: str
    output_npz: Path
    requested_count: int = DEFAULT_GENERATED_GRASP_COUNT
    thread: threading.Thread | None = None
    done: bool = False
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


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
        return [1.0, 0.0, 0.0, 0.0]
    return (quat / norm).tolist()


def remove_prim_if_exists(stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def ensure_prim(stage, prim_path: str, prim_type: str = "Xform"):
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        return prim
    return create_prim(prim_path, prim_type=prim_type)


def set_display_color(prim, rgb: np.ndarray, opacity: float = 1.0) -> None:
    color = np.asarray(rgb, dtype=np.float32).reshape(3)
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    gprim.CreateDisplayOpacityAttr().Set([float(max(0.0, min(opacity, 1.0)))])


def np_to_gf_matrix4d(matrix: np.ndarray) -> Gf.Matrix4d:
    transposed = np.asarray(matrix, dtype=np.float64).T
    return Gf.Matrix4d(
        transposed[0, 0], transposed[0, 1], transposed[0, 2], transposed[0, 3],
        transposed[1, 0], transposed[1, 1], transposed[1, 2], transposed[1, 3],
        transposed[2, 0], transposed[2, 1], transposed[2, 2], transposed[2, 3],
        transposed[3, 0], transposed[3, 1], transposed[3, 2], transposed[3, 3],
    )


def gf_matrix_to_np(matrix: Gf.Matrix4d) -> np.ndarray:
    raw = np.array([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
    return raw.T


def set_local_matrix(prim, matrix: np.ndarray) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(np_to_gf_matrix4d(matrix))


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


def set_camera_transform(stage, camera_prim_path: str, translate_xyz: np.ndarray, rotate_xyz_deg: np.ndarray) -> bool:
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim or not camera_prim.IsValid():
        return False
    camera_matrix = np.eye(4, dtype=np.float64)
    camera_matrix[:3, :3] = euler2mat(np.radians(np.asarray(rotate_xyz_deg, dtype=np.float64)), order="xyz")
    camera_matrix[:3, 3] = np.asarray(translate_xyz, dtype=np.float64)
    set_local_matrix(camera_prim, camera_matrix)
    return True


def compute_world_matrix(prim) -> np.ndarray:
    xformable = UsdGeom.Xformable(prim)
    return gf_matrix_to_np(xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def pose_matrix(position: list[float] | np.ndarray, quaternion_wxyz: list[float] | np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat2mat_wxyz(np.asarray(normalize_quaternion_wxyz(quaternion_wxyz), dtype=np.float64))
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    position = matrix[:3, 3].astype(np.float64).tolist()
    quaternion = normalize_quaternion_wxyz(mat2quat_wxyz(matrix[:3, :3]).tolist())
    return position, quaternion


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


def rotation_z_matrix(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def is_passive_place_pose(role: str | None, pose_type: str | None) -> bool:
    return role == "passive" and pose_type == "place"


def default_direction_for_pose(role: str | None = None, pose_type: str | None = None) -> np.ndarray:
    if is_passive_place_pose(role, pose_type):
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def normalize_direction_vector(
    direction: list[float] | np.ndarray,
    role: str | None = None,
    pose_type: str | None = None,
) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return default_direction_for_pose(role, pose_type)
    return vec / norm


def build_rotation_from_negative_y_axis(direction: np.ndarray) -> np.ndarray:
    # `passive/place` uses local -Y as the fall/insert direction.
    return build_rotation_from_x_axis(direction) @ rotation_z_matrix(np.pi / 2.0)


def primitive_rotation_from_direction(
    direction: list[float] | np.ndarray,
    role: str | None = None,
    pose_type: str | None = None,
) -> np.ndarray:
    aligned_direction = normalize_direction_vector(direction, role, pose_type)
    if is_passive_place_pose(role, pose_type):
        return build_rotation_from_negative_y_axis(aligned_direction)
    return build_rotation_from_x_axis(aligned_direction)


def primitive_item_to_pose(
    item: dict[str, Any],
    role: str | None = None,
    pose_type: str | None = None,
) -> tuple[list[float], list[float]]:
    xyz = ensure_vector(item.get("xyz"), 3, 0.0)
    direction = ensure_vector(item.get("direction"), 3, 0.0)
    R = primitive_rotation_from_direction(direction, role=role, pose_type=pose_type)
    quat = normalize_quaternion_wxyz(mat2quat_wxyz(R).tolist())
    return xyz, quat


def pose_to_primitive_item(
    position: list[float] | np.ndarray,
    quaternion_wxyz: list[float] | np.ndarray,
    role: str | None = None,
    pose_type: str | None = None,
) -> dict[str, list[float]]:
    quat = np.asarray(normalize_quaternion_wxyz(quaternion_wxyz), dtype=np.float64)
    R = quat2mat_wxyz(quat)
    if is_passive_place_pose(role, pose_type):
        direction = -R[:, 1]
    else:
        direction = R[:, 0]
    direction = normalize_direction_vector(direction, role, pose_type)
    return {
        "xyz": [float(v) for v in np.asarray(position, dtype=np.float64)],
        "direction": [float(v) for v in direction],
    }


def direction_to_yaw_pitch_deg(direction: list[float] | np.ndarray) -> tuple[float, float]:
    vec = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        vec = vec / norm
    yaw_deg = float(np.degrees(np.arctan2(vec[1], vec[0])))
    pitch_deg = float(np.degrees(np.arctan2(vec[2], np.linalg.norm(vec[:2]))))
    return yaw_deg, pitch_deg


def yaw_pitch_deg_to_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.radians(float(yaw_deg))
    pitch = np.radians(float(pitch_deg))
    cos_pitch = float(np.cos(pitch))
    direction = np.array(
        [
            cos_pitch * float(np.cos(yaw)),
            cos_pitch * float(np.sin(yaw)),
            float(np.sin(pitch)),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return direction / norm


def axis_base(axis_name: str) -> str:
    return axis_name[-1]


def semantic_rotation(approach: str, width: str, top: str) -> np.ndarray:
    R = np.column_stack([AXIS_VECS[approach], AXIS_VECS[width], AXIS_VECS[top]])
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(f"invalid right-handed assignment: {(approach, width, top)}, det={det:.1f}")
    return R


def axis_name_from_vec(vec: np.ndarray) -> str:
    vec = np.asarray(vec, dtype=np.float64)
    idx = int(np.argmax(np.abs(vec)))
    sign = "+" if vec[idx] >= 0.0 else "-"
    base = ("x", "y", "z")[idx]
    unit = np.zeros(3, dtype=np.float64)
    unit[idx] = 1.0 if sign == "+" else -1.0
    if not np.allclose(vec, unit):
        raise ValueError(f"vector is not a signed basis axis: {vec}")
    return f"{sign}{base}"


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


class AssetCatalog:
    def __init__(self, objects_root: Path, interaction_root: Path) -> None:
        self.objects_root = objects_root
        self.interaction_root = interaction_root
        self.entries: dict[str, CatalogEntry] = {}
        self._build_index()

    def _build_index(self) -> None:
        for category_dir in sorted(self.objects_root.iterdir()):
            if not category_dir.is_dir():
                continue
            for object_dir in sorted(category_dir.iterdir()):
                if not object_dir.is_dir():
                    continue
                usd_path = self._resolve_usd_path(object_dir)
                if usd_path is None:
                    continue
                params = self._load_object_params(object_dir)
                object_id = object_dir.name
                interaction_dir = self.interaction_root / object_id
                interaction_json = interaction_dir / "interaction.json"
                self.entries[object_id] = CatalogEntry(
                    object_id=object_id,
                    category=category_dir.name,
                    usd_path=usd_path,
                    object_dir=object_dir,
                    interaction_dir=interaction_dir,
                    interaction_json=interaction_json,
                    size=params["size"],
                    scale=params["scale"],
                    has_interaction=interaction_json.exists(),
                )
        if not self.entries:
            raise RuntimeError("No benchmark objects were found under objects_root.")

    def _resolve_usd_path(self, object_dir: Path) -> Path | None:
        candidate = object_dir / "Aligned.usd"
        return candidate if candidate.exists() else None

    def _load_object_params(self, object_dir: Path) -> dict[str, Any]:
        params_path = object_dir / "object_parameters.json"
        if not params_path.exists():
            return {"size": [0.3, 0.3, 0.3], "scale": 1.0}
        try:
            with params_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {"size": [0.3, 0.3, 0.3], "scale": 1.0}
        size = data.get("size")
        if not isinstance(size, list) or len(size) != 3:
            size = [0.3, 0.3, 0.3]
        scale = data.get("scale", 1.0)
        try:
            scale_value = float(scale)
        except Exception:
            scale_value = 1.0
        return {"size": [float(size[0]), float(size[1]), float(size[2])], "scale": scale_value}

    def object_ids(self) -> list[str]:
        return sorted(self.entries.keys())

    def categories(self) -> list[str]:
        return sorted_unique([entry.category for entry in self.entries.values()])

    def object_ids_in_category(self, category: str) -> list[str]:
        return sorted([object_id for object_id, entry in self.entries.items() if entry.category == category])

    def default_object_id(self, preferred: str) -> str:
        if preferred in self.entries:
            return preferred
        return self.object_ids()[0]

    def default_object_id_in_category(self, category: str, preferred: str | None = None) -> str:
        object_ids = self.object_ids_in_category(category)
        if preferred is not None and preferred in object_ids:
            return preferred
        if object_ids:
            return object_ids[0]
        return self.default_object_id(preferred or "")

    def get_entry(self, object_id: str) -> CatalogEntry:
        return self.entries[object_id]

    def missing_object_ids(self) -> list[str]:
        return sorted([object_id for object_id, entry in self.entries.items() if not entry.has_interaction])

    @property
    def labeled_count(self) -> int:
        return sum(1 for entry in self.entries.values() if entry.has_interaction)

    @property
    def missing_count(self) -> int:
        return sum(1 for entry in self.entries.values() if not entry.has_interaction)

    def cycle_object(self, current: str, delta: int) -> str:
        object_ids = self.object_ids()
        idx = object_ids.index(current) if current in object_ids else 0
        return object_ids[(idx + delta) % len(object_ids)]

    def cycle_missing(self, current: str, delta: int) -> str:
        missing = self.missing_object_ids()
        if not missing:
            return current
        idx = missing.index(current) if current in missing else 0
        return missing[(idx + delta) % len(missing)]


class InteractionDocument:
    def __init__(self, entry: CatalogEntry) -> None:
        self.entry = entry
        self.root_data: dict[str, Any] = {}
        self.interaction: dict[str, Any] = {}
        self._dirty_grasps: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
        self._loaded_grasp_primitives: set[str] = set()
        self.reload()

    def reload(self) -> None:
        if self.entry.interaction_json.exists():
            try:
                with self.entry.interaction_json.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        else:
            data = {}
        if not isinstance(data, dict):
            data = {}
        interaction = data.get("interaction")
        if not isinstance(interaction, dict):
            interaction = {}
        data["interaction"] = interaction
        self.root_data = data
        self.interaction = interaction
        grasp_map = interaction.get("passive", {}).get("grasp", {}) if isinstance(interaction.get("passive", {}), dict) else {}
        self._loaded_grasp_primitives = {key for key in grasp_map.keys() if isinstance(key, str)} if isinstance(grasp_map, dict) else set()
        self._dirty_grasps = {}

    def available_roles(self) -> list[str]:
        values = [role for role, payload in self.interaction.items() if isinstance(payload, dict)]
        return sorted_unique(values, ROLE_ORDER)

    def available_pose_types(self, role_filter: str = ALL_FILTER) -> list[str]:
        values: list[str] = []
        roles = ROLE_ORDER if role_filter == ALL_FILTER else (role_filter,)
        for role in roles:
            role_data = self.interaction.get(role, {})
            if not isinstance(role_data, dict):
                continue
            values.extend([pose_type for pose_type, payload in role_data.items() if isinstance(payload, dict)])
        return sorted_unique(values, POSE_TYPE_ORDER)

    def available_primitives(self, role_filter: str = ALL_FILTER, pose_type_filter: str = ALL_FILTER) -> list[str]:
        values: list[str] = []
        roles = ROLE_ORDER if role_filter == ALL_FILTER else (role_filter,)
        for role in roles:
            role_data = self.interaction.get(role, {})
            if not isinstance(role_data, dict):
                continue
            pose_types = POSE_TYPE_ORDER if pose_type_filter == ALL_FILTER else (pose_type_filter,)
            for pose_type in pose_types:
                pose_data = role_data.get(pose_type, {})
                if isinstance(pose_data, dict):
                    values.extend(list(pose_data.keys()))
        return sorted_unique([value for value in values if isinstance(value, str)])

    def primitive_items(self, role: str, pose_type: str, primitive: str) -> list[dict[str, Any]]:
        if pose_type == "grasp" and role == "passive":
            return []
        role_data = self.interaction.get(role, {})
        pose_data = role_data.get(pose_type, {}) if isinstance(role_data, dict) else {}
        payload = pose_data.get(primitive, []) if isinstance(pose_data, dict) else []
        if not isinstance(payload, list):
            return []
        items: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if "xyz" not in item or "direction" not in item:
                continue
            cleaned = {key: copy.deepcopy(value) for key, value in item.items() if key not in {"xyz", "direction"}}
            cleaned["xyz"] = ensure_vector(item.get("xyz"), 3, 0.0)
            cleaned["direction"] = ensure_vector(item.get("direction"), 3, 0.0)
            items.append(cleaned)
        return items

    def set_primitive_items(self, role: str, pose_type: str, primitive: str, items: list[dict[str, Any]]) -> None:
        role_data = self.interaction.setdefault(role, {})
        pose_data = role_data.setdefault(pose_type, {})
        cleaned = []
        for item in items:
            normalized_item = {key: copy.deepcopy(value) for key, value in item.items() if key not in {"xyz", "direction"}}
            normalized_item["xyz"] = [float(v) for v in ensure_vector(item.get("xyz"), 3, 0.0)]
            normalized_item["direction"] = [float(v) for v in ensure_vector(item.get("direction"), 3, 0.0)]
            cleaned.append(normalized_item)
        if cleaned:
            pose_data[primitive] = cleaned
        else:
            pose_data.pop(primitive, None)
            if not pose_data:
                role_data.pop(pose_type, None)
            if not role_data:
                self.interaction.pop(role, None)

    def default_grasp_rel_path(self, primitive: str) -> str:
        if primitive == "default":
            return "grasp_pose/grasp_pose.pkl"
        return f"grasp_pose/{sanitize_token(primitive)}.pkl"

    def grasp_primitives(self) -> list[str]:
        grasp_map = self.interaction.get("passive", {}).get("grasp", {})
        if not isinstance(grasp_map, dict):
            return []
        return sorted_unique([key for key in grasp_map.keys() if isinstance(key, str)])

    def was_grasp_primitive_loaded(self, primitive: str) -> bool:
        return primitive in self._loaded_grasp_primitives

    def unique_grasp_primitive(self, preferred: str) -> str:
        candidate = sanitize_token(preferred)
        if not candidate:
            candidate = "graspgen"
        existing = set(self.grasp_primitives())
        if candidate not in existing:
            return candidate
        index = 1
        while True:
            resolved = f"{candidate}_{index:02d}"
            if resolved not in existing:
                return resolved
            index += 1

    def load_grasp_group(self, primitive: str = "default") -> tuple[np.ndarray, np.ndarray, str]:
        grasp_map = self.interaction.get("passive", {}).get("grasp", {})
        payload = grasp_map.get(primitive) if isinstance(grasp_map, dict) else None
        rel_paths: list[str] = []
        if isinstance(payload, str):
            rel_paths = [payload]
        elif isinstance(payload, list):
            rel_paths = [item for item in payload if isinstance(item, str)]
        rel_path = rel_paths[0] if rel_paths else self.default_grasp_rel_path(primitive)

        poses_list: list[np.ndarray] = []
        widths_list: list[np.ndarray] = []
        for path_str in rel_paths:
            abs_path = self.entry.interaction_dir / path_str
            if not abs_path.exists():
                continue
            with abs_path.open("rb") as f:
                data = pickle.load(f)
            poses = np.asarray(data.get("grasp_pose", []), dtype=np.float64)
            widths = np.asarray(data.get("width", []), dtype=np.float32)
            if poses.ndim != 3 or poses.shape[1:] != (4, 4):
                continue
            if widths.ndim != 1 or widths.shape[0] != poses.shape[0]:
                continue
            poses_list.append(poses)
            widths_list.append(widths)
        if not poses_list:
            return (
                np.zeros((0, 4, 4), dtype=np.float64),
                np.zeros((0,), dtype=np.float32),
                rel_path,
            )
        return np.concatenate(poses_list, axis=0), np.concatenate(widths_list, axis=0), rel_path

    def set_grasp_group(
        self,
        primitive: str,
        poses: np.ndarray,
        widths: np.ndarray,
        rel_path: str | None = None,
    ) -> None:
        poses_array = np.asarray(poses, dtype=np.float64)
        widths_array = np.asarray(widths, dtype=np.float32)
        if poses_array.ndim != 3 or poses_array.shape[1:] != (4, 4):
            raise ValueError(f"grasp poses should be (N,4,4), got {poses_array.shape}")
        if widths_array.ndim != 1 or widths_array.shape[0] != poses_array.shape[0]:
            raise ValueError(f"grasp widths should be (N,), got {widths_array.shape}")

        grasp_map = self.interaction.setdefault("passive", {}).setdefault("grasp", {})
        current_rel_path = rel_path
        if current_rel_path is None:
            current_payload = grasp_map.get(primitive)
            if isinstance(current_payload, str):
                current_rel_path = current_payload
            elif isinstance(current_payload, list) and current_payload and isinstance(current_payload[0], str):
                current_rel_path = current_payload[0]
            else:
                current_rel_path = self.default_grasp_rel_path(primitive)
        grasp_map[primitive] = [current_rel_path]
        self._dirty_grasps[primitive] = (poses_array, widths_array, current_rel_path)

    def save(self) -> None:
        self.entry.interaction_dir.mkdir(parents=True, exist_ok=True)
        for primitive, (poses, widths, rel_path) in self._dirty_grasps.items():
            abs_path = self.entry.interaction_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            with abs_path.open("wb") as f:
                pickle.dump(
                    {
                        "grasp_pose": np.asarray(poses, dtype=np.float64),
                        "width": np.asarray(widths, dtype=np.float32),
                    },
                    f,
                )
            grasp_map = self.interaction.setdefault("passive", {}).setdefault("grasp", {})
            grasp_map[primitive] = [rel_path]
        with self.entry.interaction_json.open("w", encoding="utf-8") as f:
            json.dump(self.root_data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        self._dirty_grasps = {}


class InteractiveInteractionPoseEditor:
    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()
        self.selection = omni.usd.get_context().get_selection()
        self.catalog = AssetCatalog(Path(ARGS.objects_root), Path(ARGS.interaction_root))

        self.object_id = self.catalog.default_object_id(ARGS.object_id)
        self.document = InteractionDocument(self.catalog.get_entry(self.object_id))
        self.quick_category = self.catalog.get_entry(self.object_id).category
        self.quick_object_id = self.object_id
        self.view_role_filter = ALL_FILTER
        self.view_pose_type_filter = ALL_FILTER
        self.view_primitive_filter = ALL_FILTER

        self.edit_role = "passive"
        self.edit_pose_type = "place"
        self.selected_pose_index = 0
        self.axis_assignment = AxisAssignment()
        self.grasp_local_offset = np.asarray(DEFAULT_GRASP_LOCAL_OFFSET, dtype=np.float64)
        self.generated_grasp_count = max(1, int(ARGS.topk_num_grasps))
        self.show_stored_overlays = True
        self.generated: GeneratedGraspResult | None = None
        self.status_message = "Ready."
        self.running = True

        self._window = None
        self._choose_dir_window = None
        self._loaded_object_id: str | None = None
        self._loaded_asset_path: str | None = None
        self._scene_dirty = True
        self._ui_dirty = True
        self._suspend_model_callbacks = False
        self._handle_records: dict[int, HandleRecord] = {}
        self._graspgen_job_lock = threading.Lock()
        self._graspgen_job: GraspGenJob | None = None

        self.primitive_name_model = ui.SimpleStringModel("default")
        self.object_search_model = ui.SimpleStringModel(self.object_id)
        self.generated_count_model = ui.SimpleStringModel(str(self.generated_grasp_count))
        self.quick_category_search_model = ui.SimpleStringModel(self.quick_category)
        self.quick_object_search_model = ui.SimpleStringModel(self.quick_object_id)
        self.float_models: dict[str, ui.SimpleFloatModel] = {}
        self._create_float_models()
        self._sync_models_from_state()

        self._warm_up()
        self._ensure_scene_roots()
        self._ensure_light()
        self._build_ui_window()
        self._apply_scene(force=True)

    def _warm_up(self) -> None:
        for _ in range(20):
            simulation_app.update()

    def _ensure_scene_roots(self) -> None:
        ensure_prim(self.stage, "/World", prim_type="Xform")
        ensure_prim(self.stage, SCENE_ROOT, prim_type="Xform")
        ensure_prim(self.stage, OBJECT_PATH, prim_type="Xform")

    def _ensure_light(self) -> None:
        light_path = Sdf.Path(LIGHT_PATH)
        if not self.stage.GetPrimAtPath(light_path):
            light = UsdLux.SphereLight.Define(self.stage, light_path)
            light.CreateIntensityAttr(70000.0)
            light.CreateRadiusAttr(0.3)
            xformable = UsdGeom.Xformable(self.stage.GetPrimAtPath(light_path))
            translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            translate_op.Set(Gf.Vec3d(1.0, 1.4, 1.8))

    def _create_float_models(self) -> None:
        keys = [
            "pose.x",
            "pose.y",
            "pose.z",
            "direction.x",
            "direction.y",
            "direction.z",
            "offset.x",
            "offset.y",
            "offset.z",
        ]
        for key in keys:
            model = ui.SimpleFloatModel(0.0)
            if hasattr(model, "add_value_changed_fn"):
                model.add_value_changed_fn(lambda model, key=key: self._on_float_model_changed(key, model, live=True))
            model.add_end_edit_fn(lambda model, key=key: self._on_float_model_changed(key, model, live=False))
            self.float_models[key] = model

    def _string_model_value(self, model) -> str:
        try:
            return str(model.get_value_as_string())
        except Exception:
            value = getattr(model, "as_string", "")
            return str(value)

    def _set_string_model_value(self, model, value: str) -> None:
        try:
            model.set_value(str(value))
        except Exception:
            try:
                model.as_string = str(value)
            except Exception:
                pass

    def _current_primitive_name(self) -> str:
        value = self._string_model_value(self.primitive_name_model).strip()
        return value or "default"

    def _current_object_query(self) -> str:
        return self._string_model_value(self.object_search_model).strip()

    def _current_generated_count_text(self) -> str:
        return self._string_model_value(self.generated_count_model).strip()

    def _current_quick_category_query(self) -> str:
        return self._string_model_value(self.quick_category_search_model).strip()

    def _current_quick_object_query(self) -> str:
        return self._string_model_value(self.quick_object_search_model).strip()

    def _set_status(self, message: str) -> None:
        self.status_message = message
        self._ui_dirty = True
        print(f"[status] {message}")

    def _mark_scene_dirty(self) -> None:
        self._scene_dirty = True
        self._ui_dirty = True

    def _selected_edit_items(self) -> list[dict[str, Any]]:
        if self.edit_pose_type == "grasp":
            return []
        return self.document.primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name())

    def _selected_edit_item(self) -> dict[str, Any] | None:
        items = self._selected_edit_items()
        if not items:
            return None
        self.selected_pose_index = max(0, min(self.selected_pose_index, len(items) - 1))
        return items[self.selected_pose_index]

    def _sync_models_from_state(self) -> None:
        self._suspend_model_callbacks = True
        try:
            item = self._selected_edit_item()
            if item is None:
                default_direction = default_direction_for_pose(self.edit_role, self.edit_pose_type)
                pose_values = {
                    "pose.x": 0.0,
                    "pose.y": 0.0,
                    "pose.z": 0.0,
                    "direction.x": float(default_direction[0]),
                    "direction.y": float(default_direction[1]),
                    "direction.z": float(default_direction[2]),
                }
            else:
                position = ensure_vector(item.get("xyz"), 3, 0.0)
                direction = ensure_vector(item.get("direction"), 3, 0.0)
                pose_values = {
                    "pose.x": float(position[0]),
                    "pose.y": float(position[1]),
                    "pose.z": float(position[2]),
                    "direction.x": float(direction[0]),
                    "direction.y": float(direction[1]),
                    "direction.z": float(direction[2]),
                }
            pose_values.update(
                {
                    "offset.x": float(self.grasp_local_offset[0]),
                    "offset.y": float(self.grasp_local_offset[1]),
                    "offset.z": float(self.grasp_local_offset[2]),
                }
            )
            for key, value in pose_values.items():
                self.float_models[key].set_value(float(value))
        finally:
            self._suspend_model_callbacks = False
        self._ui_dirty = True

    def _on_float_model_changed(self, key: str, model: ui.SimpleFloatModel, live: bool = False) -> None:
        if self._suspend_model_callbacks:
            return
        value = float(model.get_value_as_float())
        if key.startswith("offset."):
            axis = {"offset.x": 0, "offset.y": 1, "offset.z": 2}[key]
            self.grasp_local_offset[axis] = value
            self._mark_scene_dirty()
            if not live:
                self._set_status("Updated generated-grasp local offset.")
            return

        items = self.document.primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name())
        if not items:
            self._sync_models_from_state()
            return
        self.selected_pose_index = max(0, min(self.selected_pose_index, len(items) - 1))
        item = items[self.selected_pose_index]

        position = np.asarray(ensure_vector(item.get("xyz"), 3, 0.0), dtype=np.float64)
        direction = np.asarray(ensure_vector(item.get("direction"), 3, 0.0), dtype=np.float64)
        if key == "pose.x":
            position[0] = value
        elif key == "pose.y":
            position[1] = value
        elif key == "pose.z":
            position[2] = value
        elif key in ("direction.x", "direction.y", "direction.z"):
            raw_direction = np.array(
                [
                    float(self.float_models["direction.x"].get_value_as_float()),
                    float(self.float_models["direction.y"].get_value_as_float()),
                    float(self.float_models["direction.z"].get_value_as_float()),
                ],
                dtype=np.float64,
            )
            direction = normalize_direction_vector(
                raw_direction,
                role=self.edit_role,
                pose_type=self.edit_pose_type,
            )
        else:
            return

        item["xyz"] = [float(v) for v in position]
        item["direction"] = [float(v) for v in direction]
        self.document.set_primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name(), items)
        self._mark_scene_dirty()
        if not live:
            self._set_status("Updated primitive pose from numeric fields.")

    def _build_ui_window(self) -> None:
        window_kwargs = {
            "title": UI_TITLE,
            "width": 460,
            "height": MAIN_WINDOW_HEIGHT,
            "visible": True,
            "dockPreference": ui.DockPreference.LEFT_BOTTOM,
        }
        if ScrollingWindow is not None:
            self._window = ScrollingWindow(**window_kwargs)
        else:
            self._window = ui.Window(**window_kwargs)
        self._window.visible = True
        frame = self._window.frame
        if hasattr(frame, "set_build_fn"):
            frame.set_build_fn(self._build_ui_contents)
        self._rebuild_ui()

    def _rebuild_ui(self) -> None:
        if self._window is None:
            return
        frame = self._window.frame
        if hasattr(frame, "set_build_fn"):
            frame.set_build_fn(self._build_ui_contents)
        if hasattr(frame, "rebuild"):
            frame.rebuild()
        else:
            with frame:
                self._build_ui_contents()
        self._ui_dirty = False

    def _rebuild_choose_dir_window(self) -> None:
        if self._choose_dir_window is None:
            return
        frame = self._choose_dir_window.frame
        if hasattr(frame, "set_build_fn"):
            frame.set_build_fn(self._build_choose_dir_contents)
        if hasattr(frame, "rebuild"):
            frame.rebuild()
        else:
            with frame:
                self._build_choose_dir_contents()

    def _build_ui_contents(self) -> None:
        entry = self.catalog.get_entry(self.object_id)
        items = self._selected_edit_items()
        with ui.VStack(spacing=8, height=0):
            ui.Label(UI_TITLE, height=24)
            ui.Label(self._summary_line(), word_wrap=True, height=42)
            ui.Label(self.status_message, word_wrap=True, height=42)
            ui.Separator(height=6)

            self._build_cycle_row("Object", self.object_id, self._cycle_object, width=252)
            with ui.HStack(height=30, spacing=8):
                ui.Button("Choose Dir", width=120, clicked_fn=self._choose_object_dir)
                ui.Button("Next Dir", width=100, clicked_fn=lambda: self._cycle_object(1))
                ui.Button("Reload", width=90, clicked_fn=self._reload_current_object)
            with ui.HStack(height=30, spacing=8):
                ui.Button("Prev Missing", width=120, clicked_fn=lambda: self._jump_missing(-1))
                ui.Button("Next Missing", width=120, clicked_fn=lambda: self._jump_missing(1))
            ui.Label(
                f"Category: {entry.category} | Labeled: {'yes' if entry.has_interaction else 'no'}",
                word_wrap=True,
                height=22,
            )

            ui.Separator(height=6)
            ui.Label("View Filters", height=22)
            self._build_cycle_row("Role", filter_label(self.view_role_filter), self._cycle_view_role, width=252)
            self._build_cycle_row("Type", filter_label(self.view_pose_type_filter), self._cycle_view_pose_type, width=252)
            self._build_cycle_row("Label", filter_label(self.view_primitive_filter), self._cycle_view_primitive, width=252)
            with ui.HStack(height=30, spacing=8):
                ui.Button("Show All", width=100, clicked_fn=self._show_all)
                ui.Button("View Edit Target", width=140, clicked_fn=self._view_edit_target)
                ui.Button("Save Labels", width=110, clicked_fn=self._save_labels)
            with ui.HStack(height=28, spacing=8):
                ui.Button(
                    "Hide Labeled Poses" if self.show_stored_overlays else "Show Labeled Poses",
                    width=180,
                    clicked_fn=self._toggle_stored_overlays,
                )

            ui.Separator(height=6)
            ui.Label("Edit Target", height=22)
            self._build_cycle_row("Role", self.edit_role, self._cycle_edit_role, width=252)
            self._build_cycle_row("Type", self.edit_pose_type, self._cycle_edit_pose_type, width=252)
            with ui.HStack(height=28, spacing=6):
                ui.Label("Label", width=70)
                if hasattr(ui, "StringField"):
                    ui.StringField(model=self.primitive_name_model, width=250)
                else:
                    ui.Label(self._current_primitive_name(), width=250)
                ui.Button("Apply", width=70, clicked_fn=self._apply_label_name)

            if self.edit_pose_type != "grasp":
                ui.Separator(height=6)
                ui.Label(f"Primitive Poses ({len(items)})", height=22)
                self._build_cycle_row("Pose", self._pose_selection_label(), self._cycle_selected_pose, width=252)
                with ui.HStack(height=30, spacing=8):
                    ui.Button("Add Pose", width=90, clicked_fn=self._add_pose)
                    ui.Button("Delete Pose", width=100, clicked_fn=self._delete_selected_pose)
                    ui.Button("Duplicate", width=90, clicked_fn=self._duplicate_selected_pose)
                    ui.Button("Select Handle", width=120, clicked_fn=self._select_current_handle)
                self._build_vector_row("Position", ["pose.x", "pose.y", "pose.z"], ("x", "y", "z"))
                self._build_vector_row("Direction", ["direction.x", "direction.y", "direction.z"], ("x", "y", "z"))
                ui.Label(self._normalized_direction_preview_text(), word_wrap=True, height=22)
                ui.Label(self._primitive_quaternion_preview_text(), word_wrap=True, height=38)
            else:
                ui.Separator(height=6)
                ui.Label(
                    "Grasp labels are edited through the GraspGen section below. Primitive-pose handle editing is disabled for grasp groups.",
                    word_wrap=True,
                    height=42,
                )

            ui.Separator(height=6)
            ui.Label("GraspGen", height=22)
            ui.Label(
                f"Server: {ARGS.graspgen_host}:{ARGS.graspgen_port} | Env: {ARGS.graspgen_env}",
                word_wrap=True,
                height=22,
            )
            ui.Label(
                f"Direct-launch defaults: object_id={DEFAULT_OBJECT_ID} | server={DEFAULT_GRASPGEN_HOST}:{DEFAULT_GRASPGEN_PORT}",
                word_wrap=True,
                height=38,
            )
            ui.Label(
                f"Alignment default: approach={DEFAULT_APPROACH_AXIS} width={DEFAULT_WIDTH_AXIS} top={DEFAULT_TOP_AXIS} | local offset=({DEFAULT_GRASP_LOCAL_OFFSET[0]:.3f}, {DEFAULT_GRASP_LOCAL_OFFSET[1]:.3f}, {DEFAULT_GRASP_LOCAL_OFFSET[2]:.3f})",
                word_wrap=True,
                height=38,
            )
            ui.Label(
                f"Grasp label target: passive/grasp/{self._current_primitive_name()}",
                word_wrap=True,
                height=22,
            )
            with ui.HStack(height=28, spacing=6):
                ui.Label("Gen Count", width=70)
                if hasattr(ui, "StringField"):
                    ui.StringField(model=self.generated_count_model, width=140)
                else:
                    ui.Label(self._current_generated_count_text() or str(self.generated_grasp_count), width=140)
                ui.Button("Apply", width=70, clicked_fn=self._apply_generated_count)
            with ui.HStack(height=30, spacing=8):
                ui.Button("Run GraspGen", width=110, clicked_fn=self._start_graspgen_job)
                ui.Button("Clear Preview", width=110, clicked_fn=self._clear_generated_preview)
                ui.Button("Accept Preview", width=120, clicked_fn=self._accept_generated_grasps)
            ui.Label(self._graspgen_summary_line(), word_wrap=True, height=42)
            self._build_cycle_row("Approach", self.axis_assignment.approach, self._cycle_approach_axis, width=252)
            self._build_cycle_row("Width", self.axis_assignment.width, self._cycle_width_axis, width=252)
            self._build_cycle_row("Top", self.axis_assignment.top, self._cycle_top_axis, width=252)
            with ui.HStack(height=28, spacing=8):
                ui.Button("Reset Align", width=100, clicked_fn=self._reset_axis_assignment)
            self._build_vector_row("Local Offset", ["offset.x", "offset.y", "offset.z"], ("x", "y", "z"))

            ui.Separator(height=6)
            ui.Label(
                f"Objects: {len(self.catalog.object_ids())} total | {self.catalog.labeled_count} labeled | {self.catalog.missing_count} missing",
                word_wrap=True,
                height=22,
            )
            ui.Label(
                "Workflow: for primitive poses, add/select a handle, move it with the viewport gizmo, then Save Labels. For grasps, enter Gen Count, optionally hide labeled poses, click Run GraspGen to generate a preview, tune axes/offset, click Accept Preview to stage it, then Save Labels to write it to disk. Existing grasp labels are preserved by redirecting generated grasps into a new *_graspgen label when needed.",
                word_wrap=True,
                height=122,
            )

    def _build_choose_dir_contents(self) -> None:
        object_ids = self.catalog.object_ids_in_category(self.quick_category)
        object_count = len(object_ids)
        if self.quick_object_id not in object_ids and object_ids:
            self.quick_object_id = object_ids[0]
            self._set_string_model_value(self.quick_object_search_model, self.quick_object_id)
        selected_entry = self.catalog.entries.get(self.quick_object_id)
        selected_dir = str(selected_entry.object_dir) if selected_entry is not None else "No object directory available."

        with ui.VStack(spacing=8, height=0):
            ui.Label("Choose Benchmark Object", height=24)
            ui.Label(
                "Select a category, then pick the object directory to load into the editor.",
                word_wrap=True,
                height=42,
            )
            self._build_cycle_row("Category", self.quick_category, self._cycle_quick_category, width=252)
            with ui.HStack(height=28, spacing=6):
                ui.Label("Find Cat", width=70)
                if hasattr(ui, "StringField"):
                    ui.StringField(model=self.quick_category_search_model, width=250)
                else:
                    ui.Label(self._current_quick_category_query(), width=250)
                ui.Button("Find", width=70, clicked_fn=self._apply_quick_category_query)
            self._build_cycle_row("Object", self.quick_object_id, self._cycle_quick_object, width=252)
            with ui.HStack(height=28, spacing=6):
                ui.Label("Find Obj", width=70)
                if hasattr(ui, "StringField"):
                    ui.StringField(model=self.quick_object_search_model, width=250)
                else:
                    ui.Label(self._current_quick_object_query(), width=250)
                ui.Button("Find", width=70, clicked_fn=self._apply_quick_object_query)
            ui.Label(f"Objects in category: {object_count}", height=22)
            ui.Label(selected_dir, word_wrap=True, height=64)
            with ui.HStack(height=30, spacing=8):
                ui.Button("OK", width=90, clicked_fn=self._apply_quick_object_selection)
                ui.Button("Cancel", width=90, clicked_fn=self._close_choose_dir_window)

    def _build_cycle_row(self, title: str, value: str, callback, width: int = 260) -> None:
        with ui.HStack(height=28, spacing=6):
            ui.Label(title, width=70)
            ui.Button("<", width=28, clicked_fn=lambda delta=-1: callback(delta))
            ui.Label(value, width=width)
            ui.Button(">", width=28, clicked_fn=lambda delta=1: callback(delta))

    def _build_vector_row(self, title: str, keys: list[str], axis_labels: tuple[str, ...]) -> None:
        with ui.HStack(height=26, spacing=4):
            ui.Label(title, width=100)
            for axis_label, key in zip(axis_labels, keys, strict=True):
                ui.Label(axis_label, width=16)
                ui.FloatField(model=self.float_models[key], width=74)

    def _current_direction_input(self) -> np.ndarray:
        return np.array(
            [
                float(self.float_models["direction.x"].get_value_as_float()),
                float(self.float_models["direction.y"].get_value_as_float()),
                float(self.float_models["direction.z"].get_value_as_float()),
            ],
            dtype=np.float64,
        )

    def _normalized_direction_preview_text(self) -> str:
        direction = normalize_direction_vector(
            self._current_direction_input(),
            role=self.edit_role,
            pose_type=self.edit_pose_type,
        )
        return f"Normalized Direction: x={direction[0]:.4f} y={direction[1]:.4f} z={direction[2]:.4f}"

    def _primitive_quaternion_preview_text(self) -> str:
        direction = normalize_direction_vector(
            self._current_direction_input(),
            role=self.edit_role,
            pose_type=self.edit_pose_type,
        )
        quaternion = normalize_quaternion_wxyz(
            mat2quat_wxyz(
                primitive_rotation_from_direction(
                    direction,
                    role=self.edit_role,
                    pose_type=self.edit_pose_type,
                )
            ).tolist()
        )
        return (
            f"Derived Quaternion: w={quaternion[0]:.4f} x={quaternion[1]:.4f} "
            f"y={quaternion[2]:.4f} z={quaternion[3]:.4f}"
        )

    def _apply_generated_count(self) -> None:
        raw_value = self._current_generated_count_text()
        try:
            parsed = int(raw_value)
        except Exception:
            self._set_string_model_value(self.generated_count_model, str(self.generated_grasp_count))
            self._set_status(f"Generated grasp count must be an integer, got {raw_value!r}.")
            return
        self.generated_grasp_count = max(1, min(512, parsed))
        self._set_string_model_value(self.generated_count_model, str(self.generated_grasp_count))
        self._mark_scene_dirty()
        self._set_status(
            f"Generated grasp count set to {self.generated_grasp_count}. This affects preview display, Accept Preview, and the next GraspGen run."
        )

    def _apply_object_search(self) -> None:
        query = self._current_object_query()
        if not query:
            self._set_string_model_value(self.object_search_model, self.object_id)
            self._set_status("Enter an object id or a search token before loading.")
            return

        if query in self.catalog.entries:
            target = query
            matches = [query]
        else:
            lowered = query.lower()
            matches = [object_id for object_id in self.catalog.object_ids() if lowered in object_id.lower()]
            if not matches:
                self._set_status(f"No object id matched search query {query!r}.")
                return
            target = matches[0]

        self.object_id = target
        self._load_object_document(clear_generated=True)
        if len(matches) == 1:
            self._set_status(f"Loaded object {target}.")
        else:
            self._set_status(f"Loaded first search match {target} from {len(matches)} candidates for query {query!r}.")

    def _choose_object_dir(self) -> None:
        self.quick_category = self.catalog.get_entry(self.object_id).category
        self.quick_object_id = self.object_id
        self._set_string_model_value(self.quick_category_search_model, self.quick_category)
        self._set_string_model_value(self.quick_object_search_model, self.quick_object_id)
        if self._choose_dir_window is None:
            self._choose_dir_window = ui.Window(
                title="Choose Benchmark Object",
                width=430,
                height=CHOOSE_DIR_WINDOW_HEIGHT,
                visible=True,
                dockPreference=ui.DockPreference.LEFT_BOTTOM,
            )
        self._choose_dir_window.visible = True
        self._rebuild_choose_dir_window()
        self._set_status("Choose a benchmark category/object, then click OK to load it.")

    def _close_choose_dir_window(self) -> None:
        if self._choose_dir_window is None:
            return
        try:
            self._choose_dir_window.visible = False
        except Exception:
            pass

    def _apply_quick_category_query(self) -> None:
        query = self._current_quick_category_query()
        categories = self.catalog.categories()
        if not query:
            self._set_string_model_value(self.quick_category_search_model, self.quick_category)
            self._set_status("Enter a category token before searching.")
            return

        if query in categories:
            category = query
        else:
            lowered = query.lower()
            matches = [name for name in categories if lowered in name.lower()]
            if not matches:
                self._set_status(f"No category matched {query!r}.")
                return
            category = matches[0]

        self.quick_category = category
        self.quick_object_id = self.catalog.default_object_id_in_category(category, preferred=self.quick_object_id)
        self._set_string_model_value(self.quick_category_search_model, self.quick_category)
        self._set_string_model_value(self.quick_object_search_model, self.quick_object_id)
        self._rebuild_choose_dir_window()
        self._set_status(f"Chooser category set to {self.quick_category}.")

    def _apply_quick_object_query(self) -> None:
        query = self._current_quick_object_query()
        if not query:
            self._set_string_model_value(self.quick_object_search_model, self.quick_object_id)
            self._set_status("Enter an object id or substring before searching.")
            return

        if query in self.catalog.entries:
            object_id = query
        else:
            lowered = query.lower()
            category_matches = [obj_id for obj_id in self.catalog.object_ids_in_category(self.quick_category) if lowered in obj_id.lower()]
            matches = category_matches if category_matches else [obj_id for obj_id in self.catalog.object_ids() if lowered in obj_id.lower()]
            if not matches:
                self._set_status(f"No object id matched {query!r}.")
                return
            object_id = matches[0]

        entry = self.catalog.get_entry(object_id)
        self.quick_category = entry.category
        self.quick_object_id = object_id
        self._set_string_model_value(self.quick_category_search_model, self.quick_category)
        self._set_string_model_value(self.quick_object_search_model, self.quick_object_id)
        self._rebuild_choose_dir_window()
        self._set_status(f"Chooser object set to {self.quick_object_id}.")

    def _apply_quick_object_selection(self) -> None:
        if self.quick_object_id not in self.catalog.entries:
            self._set_status(f"Quick object selection is invalid: {self.quick_object_id!r}.")
            return
        selected_category = self.quick_category
        selected_object_id = self.quick_object_id
        self._close_choose_dir_window()
        self.object_id = selected_object_id
        self._load_object_document(clear_generated=True)
        self._set_status(f"Loaded object {selected_object_id} from category {selected_category}.")

    def _toggle_stored_overlays(self) -> None:
        self.show_stored_overlays = not self.show_stored_overlays
        self._mark_scene_dirty()
        state = "shown" if self.show_stored_overlays else "hidden"
        self._set_status(f"Stored labeled poses are now {state}.")

    def _summary_line(self) -> str:
        entry = self.catalog.get_entry(self.object_id)
        stored_grasps, _, _ = self.document.load_grasp_group(self._current_primitive_name())
        stored_state = "shown" if self.show_stored_overlays else "hidden"
        return (
            f"{self.object_id} | category={entry.category} | labeled={'yes' if entry.has_interaction else 'no'} | "
            f"view role={filter_label(self.view_role_filter)} type={filter_label(self.view_pose_type_filter)} label={filter_label(self.view_primitive_filter)} | "
            f"stored grasp[{self._current_primitive_name()}]={stored_grasps.shape[0]} | stored overlays={stored_state}"
        )

    def _graspgen_summary_line(self) -> str:
        with self._graspgen_job_lock:
            job = self._graspgen_job
        stored_state = "shown" if self.show_stored_overlays else "hidden"
        if job is not None and not job.done:
            return f"Running GraspGen for {job.object_id} ... requested={job.requested_count} | stored labels {stored_state}"
        if self.generated is None:
            return f"No generated grasp preview loaded. requested={self.generated_grasp_count} | stored labels {stored_state}"
        shown_count = min(int(self.generated_grasp_count), int(self.generated.raw_poses.shape[0]))
        return (
            f"Generated preview: loaded={self.generated.raw_poses.shape[0]} showing={shown_count} from {self.generated.source_path.name} | "
            f"confidence range {float(self.generated.confidences[:shown_count].min()):.3f}-{float(self.generated.confidences[:shown_count].max()):.3f}"
            if self.generated.confidences.size > 0 and shown_count > 0
            else f"Generated preview: loaded={self.generated.raw_poses.shape[0]} showing={shown_count}"
        )

    def _pose_selection_label(self) -> str:
        items = self._selected_edit_items()
        if not items:
            return "none"
        self.selected_pose_index = max(0, min(self.selected_pose_index, len(items) - 1))
        return f"{self.selected_pose_index + 1}/{len(items)}"

    def _available_view_role_options(self) -> list[str]:
        return [ALL_FILTER, *ROLE_ORDER]

    def _available_view_pose_type_options(self) -> list[str]:
        return [ALL_FILTER, *POSE_TYPE_ORDER]

    def _available_view_primitive_options(self) -> list[str]:
        values = self.document.available_primitives(self.view_role_filter, self.view_pose_type_filter)
        current_label = self._current_primitive_name()
        if current_label not in values:
            values.append(current_label)
        return [ALL_FILTER, *sorted_unique(values)]

    def _cycle_value(self, current: str, options: list[str], delta: int) -> str:
        if not options:
            return current
        idx = options.index(current) if current in options else 0
        return options[(idx + delta) % len(options)]

    def _cycle_object(self, delta: int) -> None:
        self.object_id = self.catalog.cycle_object(self.object_id, delta)
        self._load_object_document(clear_generated=True)

    def _cycle_quick_category(self, delta: int) -> None:
        categories = self.catalog.categories()
        if not categories:
            return
        self.quick_category = self._cycle_value(self.quick_category, categories, delta)
        self.quick_object_id = self.catalog.default_object_id_in_category(self.quick_category)
        self._set_string_model_value(self.quick_category_search_model, self.quick_category)
        self._set_string_model_value(self.quick_object_search_model, self.quick_object_id)
        self._rebuild_choose_dir_window()

    def _cycle_quick_object(self, delta: int) -> None:
        object_ids = self.catalog.object_ids_in_category(self.quick_category)
        if not object_ids:
            return
        self.quick_object_id = self._cycle_value(self.quick_object_id, object_ids, delta)
        self._set_string_model_value(self.quick_object_search_model, self.quick_object_id)
        self._rebuild_choose_dir_window()

    def _jump_missing(self, delta: int) -> None:
        target = self.catalog.cycle_missing(self.object_id, delta)
        if target == self.object_id and self.catalog.missing_count == 0:
            self._set_status("No missing interaction labels were found.")
            return
        self.object_id = target
        self._load_object_document(clear_generated=True)

    def _reload_current_object(self) -> None:
        self._load_object_document(clear_generated=False)
        self._set_status("Reloaded current object from disk.")

    def _show_all(self) -> None:
        self.view_role_filter = ALL_FILTER
        self.view_pose_type_filter = ALL_FILTER
        self.view_primitive_filter = ALL_FILTER
        self._mark_scene_dirty()

    def _view_edit_target(self) -> None:
        self.view_role_filter = self.edit_role
        self.view_pose_type_filter = self.edit_pose_type
        self.view_primitive_filter = self._current_primitive_name()
        self._mark_scene_dirty()

    def _cycle_view_role(self, delta: int) -> None:
        self.view_role_filter = self._cycle_value(self.view_role_filter, self._available_view_role_options(), delta)
        self.view_primitive_filter = ALL_FILTER
        self._mark_scene_dirty()

    def _cycle_view_pose_type(self, delta: int) -> None:
        self.view_pose_type_filter = self._cycle_value(
            self.view_pose_type_filter,
            self._available_view_pose_type_options(),
            delta,
        )
        self.view_primitive_filter = ALL_FILTER
        self._mark_scene_dirty()

    def _cycle_view_primitive(self, delta: int) -> None:
        self.view_primitive_filter = self._cycle_value(
            self.view_primitive_filter,
            self._available_view_primitive_options(),
            delta,
        )
        self._mark_scene_dirty()

    def _cycle_edit_role(self, delta: int) -> None:
        options = list(ROLE_ORDER)
        next_role = self._cycle_value(self.edit_role, options, delta)
        if self.edit_pose_type == "grasp":
            self.edit_role = "passive"
        else:
            self.edit_role = next_role
        self._sync_models_from_state()
        self._mark_scene_dirty()

    def _cycle_edit_pose_type(self, delta: int) -> None:
        self.edit_pose_type = self._cycle_value(self.edit_pose_type, list(POSE_TYPE_ORDER), delta)
        if self.edit_pose_type == "grasp":
            self.edit_role = "passive"
        self.selected_pose_index = 0
        if self.edit_pose_type == "grasp" and self._current_primitive_name() == "":
            self._set_string_model_value(self.primitive_name_model, "default")
        self._sync_models_from_state()
        self._mark_scene_dirty()

    def _apply_label_name(self) -> None:
        value = self._current_primitive_name()
        self._set_string_model_value(self.primitive_name_model, value)
        self.selected_pose_index = 0
        self._sync_models_from_state()
        self._mark_scene_dirty()
        self._set_status(f"Edit label set to {value!r}.")

    def _cycle_selected_pose(self, delta: int) -> None:
        items = self._selected_edit_items()
        if not items:
            self.selected_pose_index = 0
            self._sync_models_from_state()
            return
        self.selected_pose_index = (self.selected_pose_index + delta) % len(items)
        self._sync_models_from_state()
        self._ui_dirty = True

    def _add_pose(self) -> None:
        if self.edit_pose_type == "grasp":
            self._set_status("Use Run GraspGen / Accept Preview for grasp labels.")
            return
        items = self.document.primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name())
        default_direction = default_direction_for_pose(self.edit_role, self.edit_pose_type)
        items.append({"xyz": [0.0, 0.0, 0.0], "direction": [float(v) for v in default_direction]})
        self.document.set_primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name(), items)
        self.selected_pose_index = len(items) - 1
        self._sync_models_from_state()
        self._mark_scene_dirty()
        self._set_status("Added a new primitive pose at the object origin.")

    def _delete_selected_pose(self) -> None:
        if self.edit_pose_type == "grasp":
            self._set_status("Primitive-pose deletion is only available for non-grasp labels.")
            return
        items = self.document.primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name())
        if not items:
            self._set_status("No primitive poses to delete.")
            return
        self.selected_pose_index = max(0, min(self.selected_pose_index, len(items) - 1))
        items.pop(self.selected_pose_index)
        self.document.set_primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name(), items)
        if items:
            self.selected_pose_index = min(self.selected_pose_index, len(items) - 1)
        else:
            self.selected_pose_index = 0
        self._sync_models_from_state()
        self._mark_scene_dirty()
        self._set_status("Deleted selected primitive pose.")

    def _duplicate_selected_pose(self) -> None:
        if self.edit_pose_type == "grasp":
            self._set_status("Primitive-pose duplication is only available for non-grasp labels.")
            return
        items = self.document.primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name())
        item = self._selected_edit_item()
        if item is None:
            self._set_status("No primitive pose selected to duplicate.")
            return
        duplicated = {key: copy.deepcopy(value) for key, value in item.items() if key not in {"xyz", "direction"}}
        duplicated["xyz"] = list(item["xyz"])
        duplicated["direction"] = list(item["direction"])
        items.append(duplicated)
        self.document.set_primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name(), items)
        self.selected_pose_index = len(items) - 1
        self._sync_models_from_state()
        self._mark_scene_dirty()
        self._set_status("Duplicated selected primitive pose.")

    def _select_current_handle(self) -> None:
        if self.edit_pose_type == "grasp":
            self._set_status("Handle selection is only available for non-grasp labels.")
            return
        record = self._handle_records.get(self.selected_pose_index)
        if record is None:
            self._set_status("No pose handle is currently available for selection.")
            return
        try:
            self.selection.set_selected_prim_paths([record.prim_path], False)
        except TypeError:
            self.selection.set_selected_prim_paths([record.prim_path], False, "")
        self._set_status(f"Selected viewport handle: {record.prim_path}")

    def _is_valid_assignment(self, assignment: AxisAssignment) -> bool:
        bases = {axis_base(assignment.approach), axis_base(assignment.width), axis_base(assignment.top)}
        if len(bases) != 3:
            return False
        try:
            semantic_rotation(*assignment.as_tuple())
        except ValueError:
            return False
        return True

    def _completed_assignment(self, role: str, new_axis: str) -> AxisAssignment | None:
        candidate = AxisAssignment(
            approach=self.axis_assignment.approach,
            width=self.axis_assignment.width,
            top=self.axis_assignment.top,
        )
        setattr(candidate, role, new_axis)

        if role in ("approach", "width"):
            if axis_base(candidate.approach) == axis_base(candidate.width):
                return None
            top_vec = np.cross(AXIS_VECS[candidate.approach], AXIS_VECS[candidate.width])
            candidate.top = axis_name_from_vec(top_vec)
            return candidate if self._is_valid_assignment(candidate) else None

        if role == "top":
            if axis_base(candidate.approach) == axis_base(candidate.top):
                return None
            width_vec = np.cross(AXIS_VECS[candidate.top], AXIS_VECS[candidate.approach])
            candidate.width = axis_name_from_vec(width_vec)
            return candidate if self._is_valid_assignment(candidate) else None

        return None

    def _cycle_axis(self, role: str, delta: int) -> None:
        current = getattr(self.axis_assignment, role)
        start = AXES.index(current)
        for step in range(1, len(AXES) + 1):
            axis_name = AXES[(start + delta * step) % len(AXES)]
            candidate = self._completed_assignment(role, axis_name)
            if candidate is not None:
                self.axis_assignment = candidate
                self._mark_scene_dirty()
                self._set_status(
                    f"Updated generated-grasp axis assignment to approach={candidate.approach}, width={candidate.width}, top={candidate.top}."
                )
                return

    def _cycle_approach_axis(self, delta: int) -> None:
        self._cycle_axis("approach", delta)

    def _cycle_width_axis(self, delta: int) -> None:
        self._cycle_axis("width", delta)

    def _cycle_top_axis(self, delta: int) -> None:
        self._cycle_axis("top", delta)

    def _reset_axis_assignment(self) -> None:
        self.axis_assignment = AxisAssignment()
        self.grasp_local_offset = np.asarray(DEFAULT_GRASP_LOCAL_OFFSET, dtype=np.float64)
        self._sync_models_from_state()
        self._mark_scene_dirty()
        self._set_status(
            f"Reset generated-grasp alignment to approach={self.axis_assignment.approach}, width={self.axis_assignment.width}, top={self.axis_assignment.top}, offset={tuple(float(v) for v in self.grasp_local_offset)}."
        )

    def _load_object_document(self, clear_generated: bool) -> None:
        self.document = InteractionDocument(self.catalog.get_entry(self.object_id))
        self.quick_category = self.catalog.get_entry(self.object_id).category
        self.quick_object_id = self.object_id
        if clear_generated:
            self.generated = None
        self.selected_pose_index = 0
        self._set_string_model_value(self.object_search_model, self.object_id)
        self._set_string_model_value(self.generated_count_model, str(self.generated_grasp_count))
        self._sync_models_from_state()
        self._mark_scene_dirty()

    def _save_labels(self) -> None:
        try:
            self.document.save()
        except Exception as exc:
            self._set_status(f"Failed to save labels: {exc}")
            return
        self.catalog.entries[self.object_id].has_interaction = self.catalog.get_entry(self.object_id).interaction_json.exists()
        self._set_status(f"Saved labels to {self.document.entry.interaction_json}")
        self._ui_dirty = True

    def _applied_generated_grasps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.generated is None:
            return (
                np.zeros((0, 4, 4), dtype=np.float64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        count = max(1, min(int(self.generated_grasp_count), int(self.generated.raw_poses.shape[0])))
        semantic_T = np.eye(4, dtype=np.float64)
        semantic_T[:3, :3] = semantic_rotation(*self.axis_assignment.as_tuple())
        offset_T = np.eye(4, dtype=np.float64)
        offset_T[:3, 3] = np.asarray(self.grasp_local_offset, dtype=np.float64)
        poses = np.asarray(self.generated.raw_poses[:count], dtype=np.float64) @ semantic_T @ offset_T
        widths = np.asarray(self.generated.widths[:count], dtype=np.float32)
        confidences = np.asarray(self.generated.confidences[:count], dtype=np.float32)
        return poses, widths, confidences

    def _accept_generated_grasps(self) -> None:
        poses, widths, _ = self._applied_generated_grasps()
        if poses.shape[0] == 0:
            self._set_status("No generated grasp preview is available to accept.")
            return

        requested_primitive = self._current_primitive_name()
        primitive = requested_primitive
        redirected_from: str | None = None
        if self.document.was_grasp_primitive_loaded(requested_primitive):
            primitive = self.document.unique_grasp_primitive(f"{requested_primitive}_graspgen")
            redirected_from = requested_primitive
            self._set_string_model_value(self.primitive_name_model, primitive)

        self.document.set_grasp_group(primitive, poses, widths)
        self.edit_role = "passive"
        self.edit_pose_type = "grasp"
        self.view_role_filter = "passive"
        self.view_pose_type_filter = "grasp"
        self.view_primitive_filter = primitive
        self._sync_models_from_state()
        self._mark_scene_dirty()
        if redirected_from is not None:
            self._set_status(
                f"Accepted {poses.shape[0]} generated grasps into passive/grasp/{primitive}. "
                f"Kept the original passive/grasp/{redirected_from} label unchanged."
            )
        else:
            self._set_status(f"Accepted {poses.shape[0]} generated grasps into passive/grasp/{primitive}.")

    def _clear_generated_preview(self) -> None:
        self.generated = None
        self._mark_scene_dirty()
        self._set_status("Cleared generated grasp preview.")

    def _start_graspgen_job(self) -> None:
        with self._graspgen_job_lock:
            if self._graspgen_job is not None and self._graspgen_job.thread is not None and self._graspgen_job.thread.is_alive():
                self._set_status("A GraspGen request is already running.")
                return
        entry = self.catalog.get_entry(self.object_id)
        if not entry.usd_path.exists():
            self._set_status(f"Mesh file does not exist: {entry.usd_path}")
            return
        ARGS.tmp_dir.mkdir(parents=True, exist_ok=True)
        output_npz = ARGS.tmp_dir / f"{entry.object_id}_graspgen_preview.npz"
        job = GraspGenJob(
            object_id=entry.object_id,
            output_npz=output_npz,
            requested_count=max(1, int(self.generated_grasp_count)),
        )
        thread = threading.Thread(target=self._run_graspgen_worker, args=(job, entry), daemon=True)
        job.thread = thread
        with self._graspgen_job_lock:
            self._graspgen_job = job
        thread.start()
        self._ui_dirty = True
        self._set_status(f"Started GraspGen request for {entry.object_id} with {job.requested_count} grasps requested.")

    def _run_graspgen_worker(self, job: GraspGenJob, entry: CatalogEntry) -> None:
        conda_sh = Path("/home/agxi/miniconda3/etc/profile.d/conda.sh")
        mesh_scale = entry.scale if entry.scale > 0 else float(ARGS.mesh_scale)
        requested_count = max(1, int(job.requested_count))
        total_candidate_count = max(int(ARGS.num_grasps), requested_count)
        command = (
            f"source {shlex.quote(str(conda_sh))} && "
            f"conda activate {shlex.quote(ARGS.graspgen_env)} && "
            f"python {shlex.quote(str(ARGS.graspgen_bridge))} "
            f"--graspgen_root {shlex.quote(str(ARGS.graspgen_root))} "
            f"--mesh_file {shlex.quote(str(entry.usd_path))} "
            f"--mesh_scale {mesh_scale:.8f} "
            f"--gripper_config {shlex.quote(str(ARGS.graspgen_gripper_config))} "
            f"--host {shlex.quote(ARGS.graspgen_host)} "
            f"--port {int(ARGS.graspgen_port)} "
            f"--num_sample_points {int(ARGS.num_sample_points)} "
            f"--num_grasps {total_candidate_count} "
            f"--topk_num_grasps {requested_count} "
            f"--output_npz {shlex.quote(str(job.output_npz))}"
        )
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
        )
        job.stdout = result.stdout
        job.stderr = result.stderr
        if result.returncode != 0:
            job.error = (
                f"GraspGen bridge failed with code {result.returncode}. "
                f"stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}"
            )
        elif not job.output_npz.exists():
            job.error = "GraspGen bridge finished without producing the preview NPZ file."
        job.done = True

    def _poll_graspgen_job(self) -> None:
        with self._graspgen_job_lock:
            job = self._graspgen_job
        if job is None or not job.done:
            return
        with self._graspgen_job_lock:
            self._graspgen_job = None
        if job.object_id != self.object_id:
            self._set_status(f"Ignored GraspGen preview for {job.object_id} because the current object changed.")
            return
        if job.error is not None:
            self._set_status(job.error)
            return
        try:
            data = np.load(job.output_npz)
            self.generated = GeneratedGraspResult(
                raw_poses=np.asarray(data["grasp_pose_raw"], dtype=np.float64),
                widths=np.asarray(data["width"], dtype=np.float32),
                confidences=np.asarray(data["confidence"], dtype=np.float32),
                source_path=job.output_npz,
            )
        except Exception as exc:
            self._set_status(f"Failed to load generated preview: {exc}")
            return
        self.edit_role = "passive"
        self.edit_pose_type = "grasp"
        self.view_role_filter = "passive"
        self.view_pose_type_filter = "grasp"
        self.view_primitive_filter = self._current_primitive_name()
        self._sync_models_from_state()
        self._mark_scene_dirty()
        self._set_status(f"Loaded generated preview with {self.generated.raw_poses.shape[0]} grasps.")

    def _view_matches(self, role: str, pose_type: str, primitive: str) -> bool:
        if self.view_role_filter != ALL_FILTER and role != self.view_role_filter:
            return False
        if self.view_pose_type_filter != ALL_FILTER and pose_type != self.view_pose_type_filter:
            return False
        if self.view_primitive_filter != ALL_FILTER and primitive != self.view_primitive_filter:
            return False
        return True

    def _object_asset_path(self, object_id: str) -> str:
        return f"{OBJECT_PATH}/{OBJECT_ASSET_PREFIX}{sanitize_token(object_id)}"

    def _clear_loaded_object_assets(self) -> None:
        object_root = self.stage.GetPrimAtPath(OBJECT_PATH)
        if not object_root or not object_root.IsValid():
            self._loaded_asset_path = None
            return
        child_paths = []
        for child in object_root.GetChildren():
            if child.GetName().startswith(OBJECT_ASSET_PREFIX):
                child_paths.append(str(child.GetPath()))
        for child_path in child_paths:
            remove_prim_if_exists(self.stage, child_path)
        self._loaded_asset_path = None

    def _load_object(self, entry: CatalogEntry, force: bool = False) -> None:
        asset_path = self._object_asset_path(entry.object_id)
        asset_prim = self.stage.GetPrimAtPath(asset_path)
        if force or self._loaded_object_id != entry.object_id or not asset_prim or not asset_prim.IsValid():
            ensure_prim(self.stage, OBJECT_PATH, prim_type="Xform")
            self._clear_loaded_object_assets()
            for _ in range(2):
                simulation_app.update()
            asset_prim = create_prim(asset_path, prim_type="Xform")
            asset_prim.GetReferences().AddReference(str(entry.usd_path))
            self._loaded_object_id = entry.object_id
            self._loaded_asset_path = asset_path
            for _ in range(10):
                simulation_app.update()

        if not set_camera_transform(
            self.stage,
            EDITOR_CAMERA_PRIM_PATH,
            np.asarray(DEFAULT_EDITOR_CAMERA_TRANSLATE, dtype=np.float64),
            np.asarray(DEFAULT_EDITOR_CAMERA_ROTATE_XYZ_DEG, dtype=np.float64),
        ):
            set_camera_view(
                eye=[0.0, 0.28, -0.475],
                target=[0.0, 0.0, 0.0],
                camera_prim_path=EDITOR_CAMERA_PRIM_PATH,
            )

    def _apply_scene(self, force: bool = False) -> None:
        entry = self.catalog.get_entry(self.object_id)
        self._load_object(entry, force=force)
        self._rebuild_overlay()
        self._scene_dirty = False
        self._ui_dirty = True

    def _iter_document_groups(self) -> list[tuple[str, str, str, Any]]:
        groups: list[tuple[str, str, str, Any]] = []
        for role in ROLE_ORDER:
            role_data = self.document.interaction.get(role, {})
            if not isinstance(role_data, dict):
                continue
            for pose_type in POSE_TYPE_ORDER:
                pose_data = role_data.get(pose_type, {})
                if not isinstance(pose_data, dict):
                    continue
                for primitive in sorted_unique(list(pose_data.keys())):
                    groups.append((role, pose_type, primitive, pose_data[primitive]))
        return groups

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
            ("x_axis", (ARGS.axis_len / 2.0, 0.0, 0.0), (ARGS.axis_len, ARGS.axis_thickness, ARGS.axis_thickness), np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("y_axis", (0.0, ARGS.axis_len / 2.0, 0.0), (ARGS.axis_thickness, ARGS.axis_len, ARGS.axis_thickness), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ("z_axis", (0.0, 0.0, ARGS.axis_len / 2.0), (ARGS.axis_thickness, ARGS.axis_thickness, ARGS.axis_len), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ]
        axis_root = create_prim(f"{grasp_path}/axes", prim_type="Xform")
        for name, translation, scale, axis_color in axis_specs:
            prim = create_prim(f"{axis_root.GetPath()}/{name}", prim_type="Cube")
            UsdGeom.Cube(prim).CreateSizeAttr().Set(1.0)
            set_local_translate_scale(prim, translation, scale)
            set_display_color(prim, axis_color)

    def _build_passive_place_marker(
        self,
        root_prim,
        color: np.ndarray,
        arrow_len: float,
        thickness: float,
        dot_radius: float,
    ) -> None:
        marker_path = str(root_prim.GetPath())

        dot = create_prim(f"{marker_path}/dot", prim_type="Sphere")
        UsdGeom.Sphere(dot).CreateRadiusAttr().Set(float(dot_radius))
        set_display_color(dot, np.clip(color + 0.18, 0.0, 1.0))

        shaft_len = max(float(arrow_len) * 0.68, dot_radius * 3.0)
        head_len = max(float(arrow_len) - shaft_len, dot_radius * 2.8)
        shaft_radius = max(float(thickness) * 0.55, dot_radius * 0.35)
        head_radius = max(float(thickness) * 1.75, dot_radius * 0.75)

        shaft = create_prim(f"{marker_path}/shaft", prim_type="Cylinder")
        UsdGeom.Cylinder(shaft).CreateAxisAttr().Set(UsdGeom.Tokens.y)
        UsdGeom.Cylinder(shaft).CreateHeightAttr().Set(float(shaft_len))
        UsdGeom.Cylinder(shaft).CreateRadiusAttr().Set(float(shaft_radius))
        shaft_matrix = np.eye(4, dtype=np.float64)
        shaft_matrix[:3, 3] = np.array([0.0, -shaft_len / 2.0, 0.0], dtype=np.float64)
        set_local_matrix(shaft, shaft_matrix)
        set_display_color(shaft, color)

        head = create_prim(f"{marker_path}/head", prim_type="Cone")
        UsdGeom.Cone(head).CreateAxisAttr().Set(UsdGeom.Tokens.y)
        UsdGeom.Cone(head).CreateHeightAttr().Set(float(head_len))
        UsdGeom.Cone(head).CreateRadiusAttr().Set(float(head_radius))
        head_matrix = np.eye(4, dtype=np.float64)
        head_matrix[:3, :3] = rotation_z_matrix(np.pi)
        head_matrix[:3, 3] = np.array([0.0, -(shaft_len + head_len / 2.0), 0.0], dtype=np.float64)
        set_local_matrix(head, head_matrix)
        set_display_color(head, np.clip(color + 0.08, 0.0, 1.0))

    def _build_grasp_group_from_arrays(self, root_path: str, poses: np.ndarray, widths: np.ndarray, base_color: np.ndarray, max_count: int) -> None:
        create_prim(root_path, prim_type="Xform")
        if poses.shape[0] == 0:
            return
        count = poses.shape[0] if max_count <= 0 else min(poses.shape[0], max_count)
        poses = poses[:count]
        widths = widths[:count]
        for i, (pose, width) in enumerate(zip(poses, widths, strict=True)):
            grasp_prim = create_prim(f"{root_path}/grasp_{i:04d}", prim_type="Xform")
            self._build_bracket_marker(
                root_prim=grasp_prim,
                width=float(width),
                color=vary_color(base_color, i, count),
                finger_len=float(ARGS.finger_len),
                handle_len=float(ARGS.handle_len),
                thickness=float(ARGS.grasp_thickness),
                contact_scale=None,
            )
            set_local_matrix(grasp_prim, np.asarray(pose, dtype=np.float64))

    def _build_primitive_group(
        self,
        root_path: str,
        items: list[dict[str, Any]],
        base_color: np.ndarray,
        role: str,
        pose_type: str,
    ) -> None:
        create_prim(root_path, prim_type="Xform")
        pose_finger_len = max(float(ARGS.primitive_arrow_len) * 0.45, 0.02)
        pose_handle_len = max(float(ARGS.primitive_arrow_len) * 0.55, 0.028)
        pose_width = max(float(ARGS.primitive_arrow_len) * 0.42, float(ARGS.primitive_marker_size) * 3.6)
        contact_scale = float(ARGS.primitive_marker_size)
        arrow_len = max(float(ARGS.primitive_arrow_len), float(ARGS.primitive_marker_size) * 6.0)

        for i, item in enumerate(items):
            xyz = np.asarray(item["xyz"], dtype=np.float64)
            direction = np.asarray(item["direction"], dtype=np.float64)
            pose_prim = create_prim(f"{root_path}/pose_{i:04d}", prim_type="Xform")
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = primitive_rotation_from_direction(direction, role=role, pose_type=pose_type)
            T[:3, 3] = xyz
            set_local_matrix(pose_prim, T)
            color = vary_color(base_color, i, len(items))
            if is_passive_place_pose(role, pose_type):
                self._build_passive_place_marker(
                    root_prim=pose_prim,
                    color=color,
                    arrow_len=arrow_len,
                    thickness=float(ARGS.primitive_thickness),
                    dot_radius=contact_scale,
                )
            else:
                self._build_bracket_marker(
                    root_prim=pose_prim,
                    width=pose_width,
                    color=color,
                    finger_len=pose_finger_len,
                    handle_len=pose_handle_len,
                    thickness=float(ARGS.primitive_thickness),
                    contact_scale=contact_scale,
                )

    def _build_handle_group(self) -> None:
        self._handle_records = {}
        if self.edit_pose_type == "grasp":
            return
        items = self._selected_edit_items()
        if not items:
            return
        root_path = f"{HANDLE_ROOT}/{sanitize_token(self.edit_role)}_{sanitize_token(self.edit_pose_type)}_{sanitize_token(self._current_primitive_name())}"
        create_prim(root_path, prim_type="Xform")
        for i, item in enumerate(items):
            handle_prim = create_prim(f"{root_path}/pose_{i:04d}", prim_type="Xform")
            position, quaternion = primitive_item_to_pose(item, role=self.edit_role, pose_type=self.edit_pose_type)
            set_local_matrix(handle_prim, pose_matrix(position, quaternion))
            color = np.array([1.0, 0.88, 0.18], dtype=np.float32) if i == self.selected_pose_index else np.array([1.0, 0.52, 0.14], dtype=np.float32)
            if is_passive_place_pose(self.edit_role, self.edit_pose_type):
                self._build_passive_place_marker(
                    root_prim=handle_prim,
                    color=color,
                    arrow_len=max(float(ARGS.primitive_arrow_len), float(ARGS.primitive_marker_size) * 6.0),
                    thickness=float(ARGS.primitive_thickness) * 1.1,
                    dot_radius=float(ARGS.primitive_marker_size) * 1.15,
                )
            else:
                self._build_bracket_marker(
                    root_prim=handle_prim,
                    width=max(float(ARGS.primitive_arrow_len) * 0.42, float(ARGS.primitive_marker_size) * 3.6),
                    color=color,
                    finger_len=max(float(ARGS.primitive_arrow_len) * 0.45, 0.02),
                    handle_len=max(float(ARGS.primitive_arrow_len) * 0.55, 0.028),
                    thickness=float(ARGS.primitive_thickness) * 1.1,
                    contact_scale=float(ARGS.primitive_marker_size) * 1.15,
                )
            self._handle_records[i] = HandleRecord(index=i, prim_path=str(handle_prim.GetPath()), last_world_matrix=compute_world_matrix(handle_prim))

    def _rebuild_overlay(self) -> None:
        remove_prim_if_exists(self.stage, OVERLAY_ROOT)
        remove_prim_if_exists(self.stage, HANDLE_ROOT)
        create_prim(OVERLAY_ROOT, prim_type="Xform")
        create_prim(HANDLE_ROOT, prim_type="Xform")

        current_edit_label = self._current_primitive_name()
        if self.show_stored_overlays:
            for group_index, (role, pose_type, primitive, payload) in enumerate(self._iter_document_groups()):
                if not self._view_matches(role, pose_type, primitive):
                    continue
                if role == self.edit_role and pose_type == self.edit_pose_type and primitive == current_edit_label and pose_type != "grasp":
                    continue
                base_color = group_base_color(role, pose_type)
                group_path = f"{OVERLAY_ROOT}/group_{group_index:02d}_{sanitize_token(role)}_{sanitize_token(pose_type)}_{sanitize_token(primitive)}"
                if pose_type == "grasp" and role == "passive":
                    poses, widths, _ = self.document.load_grasp_group(primitive)
                    self._build_grasp_group_from_arrays(group_path, poses, widths, base_color, int(ARGS.max_stored_grasps))
                else:
                    items = self.document.primitive_items(role, pose_type, primitive)
                    self._build_primitive_group(group_path, items, base_color, role, pose_type)

        if self.generated is not None and self._view_matches("passive", "grasp", current_edit_label):
            poses, widths, _ = self._applied_generated_grasps()
            self._build_grasp_group_from_arrays(
                f"{OVERLAY_ROOT}/generated_preview",
                poses,
                widths,
                np.array([0.96, 0.25, 0.70], dtype=np.float32),
                int(self.generated_grasp_count),
            )

        self._build_handle_group()
        for _ in range(2):
            simulation_app.update()

    def _poll_handle_updates(self) -> None:
        if self.edit_pose_type == "grasp" or not self._handle_records:
            return
        object_prim = self.stage.GetPrimAtPath(OBJECT_PATH)
        if not object_prim or not object_prim.IsValid():
            return
        object_world = compute_world_matrix(object_prim)
        object_inv = np.linalg.inv(object_world)

        changed = False
        items = self.document.primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name())
        for index, record in self._handle_records.items():
            prim = self.stage.GetPrimAtPath(record.prim_path)
            if not prim or not prim.IsValid() or index >= len(items):
                continue
            current_world = compute_world_matrix(prim)
            if record.last_world_matrix is not None and np.max(np.abs(current_world - record.last_world_matrix)) <= 1e-5:
                continue
            local_matrix = object_inv @ current_world
            position, quaternion = matrix_to_pose(local_matrix)
            updated = pose_to_primitive_item(
                position,
                quaternion,
                role=self.edit_role,
                pose_type=self.edit_pose_type,
            )
            items[index]["xyz"] = updated["xyz"]
            items[index]["direction"] = updated["direction"]
            record.last_world_matrix = current_world
            changed = True

        if changed:
            self.document.set_primitive_items(self.edit_role, self.edit_pose_type, self._current_primitive_name(), items)
            self._sync_models_from_state()
            self._set_status("Updated primitive pose from viewport handle.")

    def run(self) -> None:
        print("=" * 88)
        print(UI_TITLE)
        print("=" * 88)
        print(f"Objects root      : {ARGS.objects_root}")
        print(f"Interaction root  : {ARGS.interaction_root}")
        print(f"GraspGen bridge   : {ARGS.graspgen_bridge}")
        print(f"GraspGen server   : {ARGS.graspgen_host}:{ARGS.graspgen_port}")
        print(f"Objects total     : {len(self.catalog.object_ids())}")
        print(f"Objects labeled   : {self.catalog.labeled_count}")
        print(f"Objects unlabeled : {self.catalog.missing_count}")
        print("Grasp convention  : local +x = approach, +y = width, +z = top")
        print("=" * 88)

        while simulation_app.is_running() and self.running:
            self._poll_graspgen_job()
            self._poll_handle_updates()
            if self._scene_dirty:
                self._apply_scene()
            if self._ui_dirty:
                self._rebuild_ui()
            simulation_app.update()

        self.close()

    def close(self) -> None:
        simulation_app.close()


def main() -> None:
    editor = InteractiveInteractionPoseEditor()
    editor.run()


if __name__ == "__main__":
    main()
