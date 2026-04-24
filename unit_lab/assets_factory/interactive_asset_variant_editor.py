#!/usr/bin/env python3
"""Simple interactive editor for benchmark asset scale/color variants.

This tool is intentionally much simpler than
`interactive_interaction_pose_editor.py`.

It focuses on one workflow only:
1. Browse and load a benchmark asset directory.
2. Preview xyz non-uniform scale and a simple color/material override.
3. Export a new asset under the same benchmark category with a new index.

Implementation notes:
- The authoritative asset file is `Aligned.usd`.
- Export bakes xyz scale into copied mesh geometry inside the new `Aligned.usd`.
- `Aligned.usda` is regenerated only as a z-up/bbox convenience wrapper.
- `object_parameters.json`, `item.py`, and `description.py` are synchronized.

This avoids relying on runtime non-uniform `object_parameters["scale"]`, because
the layout-side mesh utilities do not fully support that safely.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import copy
import json
import pprint
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ASSET_DIR = Path(
    "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/"
    "web_basket/benchmark_web_basket_0492"
)
DEFAULT_SCALE_XYZ = (1.0, 1.0, 1.0)
DEFAULT_COLOR_RGB = (1.0, 1.0, 1.0)
DEFAULT_COLOR_MODE = "flat"
DEFAULT_MASS = 0.01
VALID_COLOR_MODES = ("tint", "flat")

UI_TITLE = "Asset Variant Editor"
SCENE_ROOT = "/World/AssetVariantEditor"
OBJECT_PATH = f"{SCENE_ROOT}/Object"
OBJECT_REFERENCE_PATH = f"{OBJECT_PATH}/Asset"
LIGHT_PATH = f"{SCENE_ROOT}/KeyLight"
MAIN_WINDOW_WIDTH = 480
MAIN_WINDOW_HEIGHT = 660
CHOOSE_DIR_WINDOW_HEIGHT = 420
EDITOR_CAMERA_PRIM_PATH = "/OmniverseKit_Persp"
PREVIEW_CAMERA_DISTANCE_SCALE = 1.35
CAMERA_MIN_DISTANCE = 0.18
CAPTURE_WARMUP_FRAMES = 8
CAPTURE_RENDERED_FRAMES = 4
DEFAULT_CAMERA_EYE = np.asarray([0.0, 0.28, -0.475], dtype=np.float64)
DEFAULT_CAMERA_TARGET = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
CAMERA_VIEW_DIRECTION = DEFAULT_CAMERA_EYE - DEFAULT_CAMERA_TARGET
PREVIEW_RELOAD_OVERRIDE_FRAMES = 90
PREVIEW_EDIT_OVERRIDE_FRAMES = 12
SNAPSHOT_FILE_NAME = "Camera1.png"
ALIGNED_SIM_FILE_NAME = "Aligned_sim.png"

_PXR_IMPORTED = False


def import_pxr() -> None:
    global _PXR_IMPORTED, Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    if _PXR_IMPORTED:
        return
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("pxr is unavailable. Run this tool inside the `issac` conda environment.") from exc
    _PXR_IMPORTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview and export scaled/colorized benchmark asset variants.")
    parser.add_argument("--asset_dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--target_index", type=str, default="")
    parser.add_argument("--scale_x", type=float, default=DEFAULT_SCALE_XYZ[0])
    parser.add_argument("--scale_y", type=float, default=DEFAULT_SCALE_XYZ[1])
    parser.add_argument("--scale_z", type=float, default=DEFAULT_SCALE_XYZ[2])
    parser.add_argument("--color_r", type=float, default=DEFAULT_COLOR_RGB[0])
    parser.add_argument("--color_g", type=float, default=DEFAULT_COLOR_RGB[1])
    parser.add_argument("--color_b", type=float, default=DEFAULT_COLOR_RGB[2])
    parser.add_argument("--color_mode", choices=VALID_COLOR_MODES, default=DEFAULT_COLOR_MODE)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument(
        "--write_once",
        action="store_true",
        help="Export once with the supplied parameters and exit without starting Isaac Sim UI.",
    )
    return parser.parse_known_args()[0]


def normalize_name_text(value: str) -> str:
    return " ".join(value.strip().split())


def sanitize_token(value: str) -> str:
    cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"[^0-9a-z_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def humanize_token(value: str) -> str:
    token = sanitize_token(value)
    if not token:
        return "asset"
    return token.replace("_", " ")


def rounded(values: list[float] | tuple[float, ...] | np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


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


def clamp_color_rgb(color_rgb: list[float] | tuple[float, float, float] | np.ndarray) -> list[float]:
    color = np.asarray(color_rgb, dtype=np.float64).reshape(3)
    return [float(np.clip(channel, 0.0, 1.0)) for channel in color]


def validate_scale_xyz(scale_xyz: list[float] | tuple[float, float, float] | np.ndarray) -> list[float]:
    scale = np.asarray(scale_xyz, dtype=np.float64).reshape(3)
    if np.any(scale <= 0.0):
        raise ValueError(f"Scale must stay positive on every axis, got {scale.tolist()}")
    return [float(component) for component in scale]


def parse_mapping_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return {}
    try:
        data = json.loads(raw_text)
    except Exception:
        data = ast.literal_eval(raw_text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse into a dict")
    return data


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def serialize_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=4, ensure_ascii=False) + "\n"


def serialize_python_mapping(data: dict[str, Any]) -> str:
    return pprint.pformat(data, indent=4, width=100, sort_dicts=False) + "\n"


def preview_output_paths(asset_dir: Path) -> tuple[Path, Path]:
    return asset_dir / "snapshot" / SNAPSHOT_FILE_NAME, asset_dir / ALIGNED_SIM_FILE_NAME


def clear_preview_outputs(asset_dir: Path) -> tuple[Path, Path]:
    snapshot_path, aligned_sim_path = preview_output_paths(asset_dir)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (snapshot_path, aligned_sim_path):
        if path.exists():
            path.unlink()
    return snapshot_path, aligned_sim_path


def normalize_index_token(raw_index: str, width_hint: int) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "", (raw_index or "").strip())
    if not token:
        raise ValueError("Target index is empty.")
    if token.isdigit():
        return token.zfill(max(width_hint, len(token)))
    return token.lower()


def resolve_up_axis(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0]).strip().lower()
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return "y"


def object_space_to_item_space(
    center: tuple[float, float, float] | list[float],
    size: tuple[float, float, float] | list[float],
    up_axis: str,
) -> tuple[list[float], list[float]]:
    axis = up_axis.lstrip("+-").lower()
    if axis == "y":
        return (
            rounded([center[0], center[2], -center[1]]),
            rounded([size[0], size[2], size[1]]),
        )
    if axis == "x":
        return (
            rounded([center[2], center[1], -center[0]]),
            rounded([size[2], size[1], size[0]]),
        )
    return rounded(center), rounded(size)


def item_space_to_object_space(
    center: tuple[float, float, float] | list[float],
    size: tuple[float, float, float] | list[float],
    up_axis: str,
) -> tuple[list[float], list[float]]:
    axis = up_axis.lstrip("+-").lower()
    if axis == "y":
        return (
            rounded([center[0], -center[2], center[1]]),
            rounded([size[0], size[2], size[1]]),
        )
    if axis == "x":
        return (
            rounded([-center[2], center[1], center[0]]),
            rounded([size[2], size[1], size[0]]),
        )
    return rounded(center), rounded(size)


def z_up_quaternion_wxyz(up_axis: str) -> tuple[float, float, float, float]:
    axis = up_axis.strip().lower()
    if axis in {"z", "+z"}:
        return (1.0, 0.0, 0.0, 0.0)
    if axis in {"y", "+y"}:
        return (0.70710678, 0.70710678, 0.0, 0.0)
    if axis == "-y":
        return (0.70710678, -0.70710678, 0.0, 0.0)
    if axis in {"x", "+x"}:
        return (0.70710678, 0.0, -0.70710678, 0.0)
    if axis == "-x":
        return (0.70710678, 0.0, 0.70710678, 0.0)
    if axis == "-z":
        return (0.0, 1.0, 0.0, 0.0)
    return (0.70710678, 0.70710678, 0.0, 0.0)


def _format_scalar(value: float) -> str:
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    if text in {"-0", "-0.0", ""}:
        return "0"
    if "." not in text and "e" not in text and "E" not in text:
        return f"{text}.0"
    return text


def _format_vec3(value: list[float] | tuple[float, float, float]) -> str:
    return f"({_format_scalar(value[0])}, {_format_scalar(value[1])}, {_format_scalar(value[2])})"


def _format_quat_wxyz(value: tuple[float, float, float, float]) -> str:
    return (
        f"({_format_scalar(value[0])}, {_format_scalar(value[1])}, "
        f"{_format_scalar(value[2])}, {_format_scalar(value[3])})"
    )


def color_label(color_rgb: list[float] | tuple[float, float, float] | np.ndarray) -> str:
    color = clamp_color_rgb(color_rgb)
    return f"rgb({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})"


@dataclass(frozen=True)
class SourceAsset:
    asset_dir: Path
    benchmark_root: Path
    category_dir: str
    asset_id: str
    index_token: str
    aligned_usd: Path
    aligned_usda: Path | None
    object_parameters_path: Path
    description_path: Path
    item_path: Path
    object_parameters: dict[str, Any]
    description: dict[str, Any]
    item_mapping: dict[str, Any]
    up_axis: str
    base_size_m: tuple[float, float, float]
    bbox_center_obj_m: tuple[float, float, float]
    mass: float

    @property
    def width_hint(self) -> int:
        return len(self.index_token)

    def asset_id_for_index(self, index_token: str) -> str:
        return f"benchmark_{self.category_dir}_{index_token}"

    def target_dir_for_index(self, index_token: str) -> Path:
        return self.asset_dir.parent / self.asset_id_for_index(index_token)


@dataclass(frozen=True)
class AssetCatalogEntry:
    asset_id: str
    category: str
    asset_dir: Path


class AssetCatalog:
    def __init__(self, benchmark_root: Path) -> None:
        self.benchmark_root = benchmark_root.expanduser().resolve()
        self.entries: dict[str, AssetCatalogEntry] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.benchmark_root.exists() or not self.benchmark_root.is_dir():
            raise FileNotFoundError(f"Benchmark root not found: {self.benchmark_root}")

        for category_dir in sorted(self.benchmark_root.iterdir()):
            if not category_dir.is_dir():
                continue
            prefix = f"benchmark_{category_dir.name}_"
            for asset_dir in sorted(category_dir.iterdir()):
                if not asset_dir.is_dir():
                    continue
                if not asset_dir.name.startswith(prefix):
                    continue
                if not (asset_dir / "Aligned.usd").exists():
                    continue
                if not (asset_dir / "object_parameters.json").exists():
                    continue
                self.entries[asset_dir.name] = AssetCatalogEntry(
                    asset_id=asset_dir.name,
                    category=category_dir.name,
                    asset_dir=asset_dir,
                )
        if not self.entries:
            raise RuntimeError(f"No benchmark assets were found under {self.benchmark_root}.")

    def asset_ids(self) -> list[str]:
        return sorted(self.entries.keys())

    def categories(self) -> list[str]:
        return sorted({entry.category for entry in self.entries.values()})

    def asset_ids_in_category(self, category: str) -> list[str]:
        return sorted([asset_id for asset_id, entry in self.entries.items() if entry.category == category])

    def default_asset_id(self, preferred: str) -> str:
        if preferred in self.entries:
            return preferred
        asset_ids = self.asset_ids()
        if not asset_ids:
            raise RuntimeError("Asset catalog is empty.")
        return asset_ids[0]

    def default_asset_id_in_category(self, category: str, preferred: str | None = None) -> str:
        asset_ids = self.asset_ids_in_category(category)
        if preferred is not None and preferred in asset_ids:
            return preferred
        if asset_ids:
            return asset_ids[0]
        return self.default_asset_id(preferred or "")

    def get_entry(self, asset_id: str) -> AssetCatalogEntry:
        return self.entries[asset_id]

    def cycle_asset(self, current: str, delta: int) -> str:
        asset_ids = self.asset_ids()
        if not asset_ids:
            raise RuntimeError("Asset catalog is empty.")
        idx = asset_ids.index(current) if current in asset_ids else 0
        return asset_ids[(idx + delta) % len(asset_ids)]


def load_source_asset(asset_dir: Path) -> SourceAsset:
    asset_dir = asset_dir.expanduser().resolve()
    if not asset_dir.exists() or not asset_dir.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")

    category_dir = asset_dir.parent.name
    asset_id = asset_dir.name
    prefix = f"benchmark_{category_dir}_"
    if not asset_id.startswith(prefix):
        raise ValueError(f"Asset id `{asset_id}` does not match category directory `{category_dir}`.")
    index_token = asset_id[len(prefix) :]
    if not index_token:
        raise ValueError(f"Failed to parse asset index from `{asset_id}`.")

    aligned_usd = asset_dir / "Aligned.usd"
    if not aligned_usd.exists():
        raise FileNotFoundError(f"Missing authoritative core USD: {aligned_usd}")

    aligned_usda = asset_dir / "Aligned.usda"
    object_parameters_path = asset_dir / "object_parameters.json"
    if not object_parameters_path.exists():
        raise FileNotFoundError(f"Missing object_parameters.json: {object_parameters_path}")

    object_parameters = json.loads(object_parameters_path.read_text(encoding="utf-8"))
    description_path = asset_dir / "description.py"
    item_path = asset_dir / "item.py"
    description = parse_mapping_file(description_path)
    item_mapping = parse_mapping_file(item_path)

    up_axis = resolve_up_axis(object_parameters.get("upAxis", item_mapping.get("up_axis", "y")))
    base_size_m = tuple(ensure_vector(object_parameters.get("size", item_mapping.get("size")), 3, 0.1))
    mass = float(object_parameters.get("mass", DEFAULT_MASS))

    bbox_center_obj_m = (0.0, 0.0, 0.0)
    shapes = item_mapping.get("shapes")
    if isinstance(shapes, list):
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            if str(shape.get("name", "")).lower() != "bbox":
                continue
            item_center = ensure_vector(shape.get("position"), 3, 0.0)
            item_size = ensure_vector(shape.get("scale", shape.get("size", base_size_m)), 3, 0.0)
            bbox_center_obj_m, _ = item_space_to_object_space(item_center, item_size, up_axis)
            break

    return SourceAsset(
        asset_dir=asset_dir,
        benchmark_root=asset_dir.parent.parent,
        category_dir=category_dir,
        asset_id=asset_id,
        index_token=index_token,
        aligned_usd=aligned_usd,
        aligned_usda=aligned_usda if aligned_usda.exists() else None,
        object_parameters_path=object_parameters_path,
        description_path=description_path,
        item_path=item_path,
        object_parameters=object_parameters,
        description=description,
        item_mapping=item_mapping,
        up_axis=up_axis,
        base_size_m=base_size_m,
        bbox_center_obj_m=tuple(bbox_center_obj_m),
        mass=mass,
    )


def suggest_next_index(source: SourceAsset) -> str:
    if not source.index_token.isdigit():
        return f"{source.index_token}_copy"
    candidate = int(source.index_token) + 1
    while True:
        token = str(candidate).zfill(source.width_hint)
        if not source.target_dir_for_index(token).exists():
            return token
        candidate += 1


def resolve_target_index(source: SourceAsset, requested_index: str) -> str:
    if requested_index.strip():
        token = normalize_index_token(requested_index, source.width_hint)
        if token == source.index_token:
            raise ValueError("Target index must differ from the source asset index.")
        return token
    return suggest_next_index(source)


def build_description_mapping(
    source: SourceAsset,
    size_m: tuple[float, float, float],
    color_rgb: list[float],
) -> dict[str, Any]:
    description = copy.deepcopy(source.description) if source.description else {}
    semantic_name = source.object_parameters.get("semantic_name", humanize_token(source.category_dir))

    if "semantic_name" not in description:
        description["semantic_name"] = [semantic_name]
    if "object_category" not in description:
        description["object_category"] = [humanize_token(source.category_dir)]
    if "materials" not in description or not isinstance(description["materials"], list):
        description["materials"] = []
    if "shape" not in description:
        description["shape"] = ""
    if "descriptive_terms" not in description:
        description["descriptive_terms"] = []
    if "full_description" not in description:
        english_name = description.get("english_name") or humanize_token(source.category_dir)
        description["full_description"] = f"{normalize_name_text(str(english_name))}."

    description["dimensions"] = rounded(size_m)
    description["unit"] = "m"
    description["color"] = color_label(color_rgb)
    return description


def build_object_parameters_mapping(
    source: SourceAsset,
    target_asset_id: str,
    size_m: tuple[float, float, float],
) -> dict[str, Any]:
    object_parameters = copy.deepcopy(source.object_parameters)
    object_parameters["object_id"] = target_asset_id
    object_parameters["size"] = rounded(size_m)
    object_parameters["scale"] = 1
    object_parameters["unit"] = "m"
    object_parameters["mass"] = round(float(object_parameters.get("mass", source.mass)), 6)
    object_parameters["upAxis"] = [source.up_axis]
    object_parameters["model_path"] = f"objects/benchmark/{source.category_dir}/{target_asset_id}/Aligned.usd"
    object_parameters.setdefault("original_model_path", str(source.aligned_usd))

    llm_descriptions = object_parameters.get("llm_descriptions")
    if not isinstance(llm_descriptions, dict):
        llm_descriptions = {}
        object_parameters["llm_descriptions"] = llm_descriptions
    llm_descriptions["dimensions"] = rounded(size_m)
    llm_descriptions["unit"] = "m"
    semantic_name = object_parameters.get("semantic_name", humanize_token(source.category_dir))
    llm_descriptions.setdefault("semantic_name", [semantic_name])
    llm_descriptions.setdefault("object_category", [humanize_token(source.category_dir)])
    llm_descriptions.setdefault("full_description", [f"{semantic_name}."])
    return object_parameters


def build_item_mapping(
    source: SourceAsset,
    target_asset_id: str,
    bbox_center_obj_m: tuple[float, float, float],
    bbox_size_obj_m: tuple[float, float, float],
) -> dict[str, Any]:
    item_mapping = copy.deepcopy(source.item_mapping) if source.item_mapping else {}
    item_mapping["up_axis"] = source.up_axis
    item_mapping["has_joint"] = bool(item_mapping.get("has_joint", False))
    item_mapping["has_articulation"] = bool(item_mapping.get("has_articulation", False))
    item_mapping["id"] = target_asset_id
    item_mapping["size"] = rounded(bbox_size_obj_m)

    bbox_position, bbox_scale = object_space_to_item_space(bbox_center_obj_m, bbox_size_obj_m, source.up_axis)
    bbox_shape = {
        "name": "bbox",
        "type": "cube",
        "position": bbox_position,
        "quaternion": [0.0, 0.0, 0.0, 1.0],
        "size": bbox_scale,
        "scale": bbox_scale,
    }
    origin_shape = {
        "name": "origin",
        "type": "sphere",
        "position": [0.0, 0.0, 0.0],
        "quaternion": [0.0, 0.0, 0.0, 1.0],
        "size": [0.001, 0.001, 0.001],
        "scale": [0.001, 0.001, 0.001],
    }

    preserved_shapes: list[dict[str, Any]] = []
    shapes = item_mapping.get("shapes")
    if isinstance(shapes, list):
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            if str(shape.get("name", "")).lower() in {"bbox", "origin"}:
                continue
            preserved_shapes.append(shape)
    item_mapping["shapes"] = [bbox_shape, origin_shape, *preserved_shapes]
    return item_mapping


def build_z_up_usda_text(
    source: SourceAsset,
    bbox_center_obj_m: tuple[float, float, float],
    bbox_size_obj_m: tuple[float, float, float],
) -> str:
    bbox_position, bbox_size = object_space_to_item_space(bbox_center_obj_m, bbox_size_obj_m, source.up_axis)
    entity_orient = z_up_quaternion_wxyz(source.up_axis)
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    quatf xformOp:orient = (1.0, 0.0, 0.0, 0.0)
    uniform token[] xformOpOrder = ["xformOp:orient"]

    def Xform "entity" (
        delete apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        prepend payload = @./Aligned.usd@
    )
    {{
        float physics:mass = {_format_scalar(source.mass)}
        quatf xformOp:orient = {_format_quat_wxyz(entity_orient)}
        uniform token[] xformOpOrder = ["xformOp:orient"]

        over "body"
        {{
            over "visual" (
                prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "PhysxConvexDecompositionCollisionAPI", "PhysxConvexHullCollisionAPI"]
            )
            {{
                uniform token physics:approximation = "convexDecomposition"
                bool physics:collisionEnabled = 1
            }}
        }}
    }}

    def Xform "lowpoly"
    {{
        uniform token[] xformOpOrder = []

        def Cube "bbox"
        {{
            double size = 1
            token visibility = "invisible"
            quatf xformOp:orient = (1.0, 0.0, 0.0, 0.0)
            float3 xformOp:scale = {_format_vec3(bbox_size)}
            double3 xformOp:translate = {_format_vec3(bbox_position)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}
    }}
}}
"""


def compute_variant_bbox(
    source: SourceAsset,
    scale_xyz: list[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    scale = np.asarray(validate_scale_xyz(scale_xyz), dtype=np.float64)
    size = np.asarray(source.base_size_m, dtype=np.float64) * np.abs(scale)
    center = np.asarray(source.bbox_center_obj_m, dtype=np.float64) * scale
    return tuple(float(value) for value in center), tuple(float(value) for value in size)


def shader_input_if_valid(shader, name: str):
    input_api = shader.GetInput(name)
    if not input_api:
        return None
    attr = input_api.GetAttr()
    if attr is None or not attr.IsValid():
        return None
    return input_api


def apply_color_overrides(root_prim, color_rgb: list[float], color_mode: str) -> tuple[int, int]:
    import_pxr()
    if root_prim is None or not root_prim.IsValid():
        return (0, 0)

    color = clamp_color_rgb(color_rgb)
    mesh_count = 0
    shader_count = 0
    apply_display_color = color_mode == "flat"

    for prim in Usd.PrimRange(root_prim):
        if apply_display_color and prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
            mesh_count += 1

        if not prim.IsA(UsdShade.Shader):
            continue

        shader = UsdShade.Shader(prim)
        changed = False
        for input_name in (
            "diffuse_color_constant",
            "color_constant",
            "base_color_constant",
            "baseColorConstant",
            "diffuse_tint",
            "diffuseColor",
            "albedo",
            "base_color",
            "baseColor",
        ):
            shader_input = shader_input_if_valid(shader, input_name)
            if shader_input is None:
                continue
            shader_input.Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
            changed = True

        if color_mode == "flat":
            for texture_input_name in (
                "diffuse_texture",
                "diffuseTexture",
                "albedo_texture",
                "albedoTexture",
                "base_color_texture",
                "baseColorTexture",
            ):
                shader_input = shader_input_if_valid(shader, texture_input_name)
                if shader_input is None:
                    continue
                shader_input.Set(Sdf.AssetPath(""))
                changed = True

        if changed:
            shader_count += 1

    return mesh_count, shader_count


def transform_points_world_scaled(points: Any, world_transform, scale_xyz: list[float]) -> list[tuple[float, float, float]]:
    import_pxr()
    sx, sy, sz = validate_scale_xyz(scale_xyz)
    transformed_points: list[tuple[float, float, float]] = []
    for point in points:
        world_point = world_transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        transformed_points.append(
            (
                float(world_point[0]) * sx,
                float(world_point[1]) * sy,
                float(world_point[2]) * sz,
            )
        )
    return transformed_points


def transform_normals_world_scaled(normals: Any, world_transform, scale_xyz: list[float]) -> list[tuple[float, float, float]]:
    sx, sy, sz = validate_scale_xyz(scale_xyz)
    transformed_normals: list[tuple[float, float, float]] = []
    for normal in normals:
        world_dir = world_transform.TransformDir(Gf.Vec3d(float(normal[0]), float(normal[1]), float(normal[2])))
        adjusted = np.array(
            [
                float(world_dir[0]) / sx,
                float(world_dir[1]) / sy,
                float(world_dir[2]) / sz,
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(adjusted))
        if norm > 1e-8:
            adjusted = adjusted / norm
        transformed_normals.append((float(adjusted[0]), float(adjusted[1]), float(adjusted[2])))
    return transformed_normals


def compute_extent(points: list[tuple[float, float, float]]) -> list[Any]:
    if not points:
        return [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0)]
    points_array = np.asarray(points, dtype=np.float32)
    min_point = points_array.min(axis=0)
    max_point = points_array.max(axis=0)
    return [
        Gf.Vec3f(float(min_point[0]), float(min_point[1]), float(min_point[2])),
        Gf.Vec3f(float(max_point[0]), float(max_point[1]), float(max_point[2])),
    ]


def bake_scaled_usd(target_usd_path: Path, scale_xyz: list[float], color_rgb: list[float], color_mode: str) -> None:
    import_pxr()
    stage = Usd.Stage.Open(str(target_usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {target_usd_path}")

    time_code = Usd.TimeCode.Default()
    mesh_world_transforms: list[tuple[Any, Any]] = []
    xform_properties_to_remove: list[tuple[Any, list[str]]] = []

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(prim)
            op_names = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
            if op_names:
                xform_properties_to_remove.append((prim, op_names))
        if prim.IsA(UsdGeom.Mesh):
            world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
            mesh_world_transforms.append((UsdGeom.Mesh(prim), world_transform))

    for mesh, world_transform in mesh_world_transforms:
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get(time_code)
        if points:
            baked_points = transform_points_world_scaled(points, world_transform, scale_xyz)
            points_attr.Set(baked_points, time_code)
            mesh.CreateExtentAttr().Set(compute_extent(baked_points), time_code)

        normals_attr = mesh.GetNormalsAttr()
        normals = normals_attr.Get(time_code)
        if normals:
            baked_normals = transform_normals_world_scaled(normals, world_transform, scale_xyz)
            normals_attr.Set(baked_normals, time_code)

    for prim, op_names in xform_properties_to_remove:
        for op_name in op_names:
            prim.RemoveProperty(op_name)
        prim.RemoveProperty("xformOpOrder")

    root_prim = stage.GetDefaultPrim()
    if root_prim is None or not root_prim.IsValid():
        root_prim = stage.GetPseudoRoot()
    apply_color_overrides(root_prim, color_rgb, color_mode)
    stage.Save()


def export_variant(
    source: SourceAsset,
    target_index: str,
    scale_xyz: list[float],
    color_rgb: list[float],
    color_mode: str,
) -> Path:
    scale_xyz = validate_scale_xyz(scale_xyz)
    color_rgb = clamp_color_rgb(color_rgb)
    if color_mode not in VALID_COLOR_MODES:
        raise ValueError(f"Unsupported color_mode `{color_mode}`.")

    target_asset_id = source.asset_id_for_index(target_index)
    target_dir = source.target_dir_for_index(target_index)
    if target_dir.exists():
        raise FileExistsError(f"Target asset already exists: {target_dir}")

    shutil.copytree(source.asset_dir, target_dir)
    clear_preview_outputs(target_dir)
    target_usd_path = target_dir / "Aligned.usd"
    bake_scaled_usd(target_usd_path, scale_xyz, color_rgb, color_mode)

    bbox_center_obj_m, bbox_size_obj_m = compute_variant_bbox(source, scale_xyz)
    object_parameters = build_object_parameters_mapping(source, target_asset_id, bbox_size_obj_m)
    description = build_description_mapping(source, bbox_size_obj_m, color_rgb)
    item_mapping = build_item_mapping(source, target_asset_id, bbox_center_obj_m, bbox_size_obj_m)
    z_up_usda_text = build_z_up_usda_text(source, bbox_center_obj_m, bbox_size_obj_m)

    write_text_atomic(target_dir / "object_parameters.json", serialize_json(object_parameters))
    write_text_atomic(target_dir / "description.py", serialize_json(description))
    write_text_atomic(target_dir / "item.py", serialize_python_mapping(item_mapping))
    write_text_atomic(target_dir / "Aligned.usda", z_up_usda_text)
    return target_dir


def run_write_once(args: argparse.Namespace) -> int:
    source = load_source_asset(args.asset_dir)
    target_index = resolve_target_index(source, args.target_index)
    target_dir = export_variant(
        source=source,
        target_index=target_index,
        scale_xyz=[args.scale_x, args.scale_y, args.scale_z],
        color_rgb=[args.color_r, args.color_g, args.color_b],
        color_mode=args.color_mode,
    )
    print(f"Exported variant: {target_dir}")
    print("Preview images were cleared. Re-capture them in interactive mode if you need fresh snapshots.")
    return 0


ARGS = parse_args()

if not ARGS.write_once:
    try:
        import isaacsim  # noqa: F401
    except ImportError:
        pass

    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"width": ARGS.width, "height": ARGS.height, "headless": False})

    import omni.ui as ui
    import omni.usd
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    try:
        from isaacsim.gui.components.element_wrappers import ScrollingWindow
    except Exception:  # pragma: no cover - Isaac Sim package availability is runtime-specific
        ScrollingWindow = None

    import_pxr()


def run_interactive() -> int:
    source = load_source_asset(ARGS.asset_dir)

    def remove_prim_if_exists(stage, prim_path: str) -> None:
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            stage.RemovePrim(prim_path)

    def ensure_scene_roots(stage) -> None:
        create_prim("/World", prim_type="Xform")
        create_prim(SCENE_ROOT, prim_type="Xform")

    def ensure_light(stage) -> None:
        light_prim = stage.GetPrimAtPath(LIGHT_PATH)
        if light_prim and light_prim.IsValid():
            return
        light = UsdLux.SphereLight.Define(stage, LIGHT_PATH)
        light.CreateIntensityAttr(70000.0)
        light.CreateRadiusAttr(0.3)
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(LIGHT_PATH))
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(250.0, 350.0, 450.0))

    def set_preview_scale(prim, scale_xyz: list[float]) -> None:
        scale_xyz = validate_scale_xyz(scale_xyz)
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            prim.RemoveProperty(op.GetOpName())
        prim.RemoveProperty("xformOpOrder")
        scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, opSuffix="previewScale")
        scale_op.Set(Gf.Vec3d(float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])))

    def frame_camera(stage, prim_path: str, distance_scale: float = PREVIEW_CAMERA_DISTANCE_SCALE) -> None:
        prim = stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            return
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
        aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        min_point = aligned_box.GetMin()
        max_point = aligned_box.GetMax()
        center = np.array(
            [
                float((min_point[0] + max_point[0]) * 0.5),
                float((min_point[1] + max_point[1]) * 0.5),
                float((min_point[2] + max_point[2]) * 0.5),
            ],
            dtype=np.float64,
        )
        size = np.array(
            [
                float(max_point[0] - min_point[0]),
                float(max_point[1] - min_point[1]),
                float(max_point[2] - min_point[2]),
            ],
            dtype=np.float64,
        )
        max_dim = float(np.max(size)) if np.all(np.isfinite(size)) else CAMERA_MIN_DISTANCE
        default_distance = max(float(np.linalg.norm(CAMERA_VIEW_DIRECTION)), CAMERA_MIN_DISTANCE)
        bbox_distance = max(0.5 * max_dim * max(float(distance_scale), 0.1), CAMERA_MIN_DISTANCE)
        dist = max(default_distance, bbox_distance)
        direction = CAMERA_VIEW_DIRECTION / np.linalg.norm(CAMERA_VIEW_DIRECTION)
        eye = center + direction * dist
        set_camera_view(
            eye=[float(eye[0]), float(eye[1]), float(eye[2])],
            target=[float(center[0]), float(center[1]), float(center[2])],
            camera_prim_path=EDITOR_CAMERA_PRIM_PATH,
        )

    def snapshot_active_viewport_camera_state(stage) -> dict[str, Any] | None:
        viewport = get_active_viewport()
        if viewport is None:
            return None
        camera_path_value = getattr(viewport, "camera_path", None)
        camera_path = getattr(camera_path_value, "pathString", str(camera_path_value or "")).strip()
        if not camera_path:
            return None
        camera_prim = stage.GetPrimAtPath(camera_path)
        if camera_prim is None or not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            return {"camera_path": camera_path, "local_transform": None}

        local_transform = None
        try:
            local_transform = UsdGeom.Xformable(camera_prim).GetLocalTransformation(Usd.TimeCode.Default())
        except Exception:
            local_transform = None
        return {"camera_path": camera_path, "local_transform": local_transform}

    def restore_active_viewport_camera_state(stage, camera_state: dict[str, Any] | None) -> None:
        if not camera_state:
            return
        viewport = get_active_viewport()
        if viewport is None:
            return
        camera_path = str(camera_state.get("camera_path", "")).strip()
        if not camera_path:
            return
        try:
            viewport.camera_path = camera_path
        except Exception:
            pass

        local_transform = camera_state.get("local_transform")
        if local_transform is None:
            return

        camera_prim = stage.GetPrimAtPath(camera_path)
        if camera_prim is None or not camera_prim.IsValid():
            return
        xformable = UsdGeom.Xformable(camera_prim)
        for op in xformable.GetOrderedXformOps():
            camera_prim.RemoveProperty(op.GetOpName())
        camera_prim.RemoveProperty("xformOpOrder")
        xform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, opSuffix="capturedView")
        xform_op.Set(local_transform)

    class AssetVariantEditor:
        def __init__(self, source_asset: SourceAsset) -> None:
            self.source = source_asset
            self.catalog = AssetCatalog(self.source.benchmark_root)
            self.stage = omni.usd.get_context().get_stage()
            self.scale_xyz = validate_scale_xyz([ARGS.scale_x, ARGS.scale_y, ARGS.scale_z])
            self.color_rgb = clamp_color_rgb([ARGS.color_r, ARGS.color_g, ARGS.color_b])
            self.color_mode = ARGS.color_mode
            self.status_message = "Adjust scale/color, then export a new asset index."
            self.ui_dirty = True
            self.running = True
            self._window = None
            self._choose_dir_window = None
            self._preview_wrapper = None
            self._preview_root = None
            self._preview_loaded = False
            self._preview_update_requested = True
            self._preview_force_reload = True
            self._preview_warmup_frames = 0
            self._preview_override_frames_remaining = 0
            self._frame_camera_requested = True
            self._pending_snapshot_target_dir: Path | None = None
            self._pending_snapshot_camera_state: dict[str, Any] | None = None
            self._snapshot_capture_warmup_frames = 0
            self._snapshot_capture_task: asyncio.Task | None = None
            self._updating_float_models = False

            target_index = resolve_target_index(self.source, ARGS.target_index)
            self.target_index_model = ui.SimpleStringModel(target_index)
            if hasattr(self.target_index_model, "add_end_edit_fn"):
                self.target_index_model.add_end_edit_fn(lambda _model: self._on_target_index_changed())
            self.quick_category = self.source.category_dir
            self.quick_asset_id = self.source.asset_id
            self.quick_category_search_model = ui.SimpleStringModel(self.quick_category)
            self.quick_asset_search_model = ui.SimpleStringModel(self.quick_asset_id)

            self.float_models: dict[str, ui.SimpleFloatModel] = {}
            self._register_float_model("scale.x", self.scale_xyz[0])
            self._register_float_model("scale.y", self.scale_xyz[1])
            self._register_float_model("scale.z", self.scale_xyz[2])
            self._register_float_model("color.r", self.color_rgb[0])
            self._register_float_model("color.g", self.color_rgb[1])
            self._register_float_model("color.b", self.color_rgb[2])

            self._warm_up()
            ensure_scene_roots(self.stage)
            ensure_light(self.stage)
            self._build_ui_window()
            self._queue_preview_update(reload_preview=True, frame_camera_view=True)

        def _warm_up(self) -> None:
            for _ in range(20):
                simulation_app.update()

        def _register_float_model(self, key: str, value: float) -> None:
            model = ui.SimpleFloatModel(float(value))
            if hasattr(model, "add_value_changed_fn"):
                model.add_value_changed_fn(lambda model, key=key: self._on_float_changed(key, model))
            if hasattr(model, "add_end_edit_fn"):
                model.add_end_edit_fn(lambda model, key=key: self._on_float_changed(key, model))
            self.float_models[key] = model

        def _on_float_changed(self, key: str, model: ui.SimpleFloatModel) -> None:
            if self._updating_float_models:
                return
            value = float(model.get_value_as_float())
            old_scale = list(self.scale_xyz)
            old_color = list(self.color_rgb)
            self._updating_float_models = True
            try:
                if key.startswith("scale."):
                    value = max(value, 1e-4)
                    model.set_value(value)
                    axis_index = {"scale.x": 0, "scale.y": 1, "scale.z": 2}[key]
                    self.scale_xyz[axis_index] = value
                else:
                    value = float(np.clip(value, 0.0, 1.0))
                    model.set_value(value)
                    axis_index = {"color.r": 0, "color.g": 1, "color.b": 2}[key]
                    self.color_rgb[axis_index] = value
            finally:
                self._updating_float_models = False
            if old_scale != self.scale_xyz or old_color != self.color_rgb:
                self._queue_preview_update()

        def _sync_values_from_models(self) -> None:
            for key, model in self.float_models.items():
                self._on_float_changed(key, model)

        def _on_target_index_changed(self) -> None:
            self.ui_dirty = True

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

        def _current_quick_category_query(self) -> str:
            return self._string_model_value(self.quick_category_search_model).strip()

        def _current_quick_asset_query(self) -> str:
            return self._string_model_value(self.quick_asset_search_model).strip()

        def _cycle_value(self, current: str, options: list[str], delta: int) -> str:
            if not options:
                return current
            idx = options.index(current) if current in options else 0
            return options[(idx + delta) % len(options)]

        def _current_target_index(self) -> str:
            raw_value = self.target_index_model.get_value_as_string().strip()
            return resolve_target_index(self.source, raw_value)

        def _safe_target_index(self) -> tuple[str | None, str | None]:
            raw_value = self.target_index_model.get_value_as_string().strip()
            try:
                return resolve_target_index(self.source, raw_value), None
            except Exception as exc:
                return None, str(exc)

        def _current_target_asset_id(self) -> str:
            return self.source.asset_id_for_index(self._current_target_index())

        def _current_target_dir(self) -> Path:
            return self.source.target_dir_for_index(self._current_target_index())

        def _set_status(self, message: str) -> None:
            self.status_message = message
            print(message)
            self.ui_dirty = True

        def _sync_chooser_to_current_source(self) -> None:
            self.quick_category = self.source.category_dir
            self.quick_asset_id = self.source.asset_id
            self._set_string_model_value(self.quick_category_search_model, self.quick_category)
            self._set_string_model_value(self.quick_asset_search_model, self.quick_asset_id)

        def _can_switch_source(self) -> bool:
            if self._snapshot_capture_in_progress():
                self._set_status("Wait for the current snapshot capture to finish before switching source objects.")
                return False
            return True

        def _load_source_asset_by_id(self, asset_id: str, message: str | None = None) -> None:
            if asset_id not in self.catalog.entries:
                self._set_status(f"Unknown source asset: {asset_id}")
                return
            try:
                source = load_source_asset(self.catalog.get_entry(asset_id).asset_dir)
            except Exception as exc:
                self._set_status(f"Failed to load source asset {asset_id}: {exc}")
                return

            self.source = source
            self.target_index_model.set_value(suggest_next_index(self.source))
            self._preview_wrapper = None
            self._preview_root = None
            self._preview_loaded = False
            self._preview_force_reload = True
            self._preview_update_requested = False
            self._preview_warmup_frames = 0
            self._frame_camera_requested = True
            self._sync_chooser_to_current_source()
            self._rebuild_choose_dir_window()
            self._queue_preview_update(reload_preview=True, frame_camera_view=True)
            self._set_status(message or f"Loaded source asset {asset_id}.")

        def _queue_preview_update(self, reload_preview: bool = False, frame_camera_view: bool = False) -> None:
            if reload_preview:
                self._preview_force_reload = True
                self._preview_override_frames_remaining = max(
                    self._preview_override_frames_remaining,
                    PREVIEW_RELOAD_OVERRIDE_FRAMES,
                )
            else:
                self._preview_override_frames_remaining = max(
                    self._preview_override_frames_remaining,
                    PREVIEW_EDIT_OVERRIDE_FRAMES,
                )
            if frame_camera_view:
                self._frame_camera_requested = True
            self._preview_update_requested = True
            self.ui_dirty = True

        def _ensure_preview_reference(self) -> None:
            remove_prim_if_exists(self.stage, OBJECT_PATH)
            self._preview_wrapper = create_prim(OBJECT_PATH, prim_type="Xform")
            self._preview_root = create_prim(OBJECT_REFERENCE_PATH, prim_type="Xform")
            self._preview_root.GetReferences().AddReference(str(self.source.aligned_usd))
            self._preview_loaded = True
            self._preview_force_reload = False
            self._preview_warmup_frames = 6
            self._preview_override_frames_remaining = max(
                self._preview_override_frames_remaining,
                PREVIEW_RELOAD_OVERRIDE_FRAMES,
            )
            self._frame_camera_requested = True

        def _process_preview_update(self) -> None:
            if not self._preview_update_requested and self._preview_override_frames_remaining <= 0:
                return

            if (
                self._preview_force_reload
                or not self._preview_loaded
                or self._preview_wrapper is None
                or not self._preview_wrapper.IsValid()
                or self._preview_root is None
                or not self._preview_root.IsValid()
            ):
                self._ensure_preview_reference()
                return

            if self._preview_warmup_frames > 0:
                self._preview_warmup_frames -= 1
                return

            should_refresh_ui = self._preview_update_requested or self._frame_camera_requested
            set_preview_scale(self._preview_wrapper, self.scale_xyz)
            apply_color_overrides(self._preview_root, self.color_rgb, self.color_mode)
            if self._preview_override_frames_remaining > 0:
                self._preview_override_frames_remaining -= 1
            if self._frame_camera_requested:
                frame_camera(self.stage, OBJECT_PATH)
                self._frame_camera_requested = False
            self._preview_update_requested = False
            if should_refresh_ui:
                self.ui_dirty = True

        def _snapshot_capture_in_progress(self) -> bool:
            return self._pending_snapshot_target_dir is not None or (
                self._snapshot_capture_task is not None and not self._snapshot_capture_task.done()
            )

        async def _capture_snapshot_async(
            self,
            target_dir: Path,
            camera_state: dict[str, Any] | None,
        ) -> tuple[Path, Path]:
            import omni.kit.app

            snapshot_path, aligned_sim_path = preview_output_paths(target_dir)
            restore_active_viewport_camera_state(self.stage, camera_state)
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("Active viewport is unavailable, cannot capture snapshot.")

            if hasattr(viewport, "wait_for_rendered_frames"):
                await viewport.wait_for_rendered_frames(CAPTURE_RENDERED_FRAMES)
            else:
                for _ in range(CAPTURE_RENDERED_FRAMES):
                    await omni.kit.app.get_app().next_update_async()

            await capture_viewport_to_file(viewport, file_path=str(snapshot_path), is_hdr=False).wait_for_result()

            frames_left = 60
            while frames_left > 0 and not snapshot_path.is_file():
                await omni.kit.app.get_app().next_update_async()
                frames_left -= 1
            if not snapshot_path.is_file():
                raise RuntimeError(f"Viewport capture did not produce {snapshot_path.name}.")

            shutil.copy2(snapshot_path, aligned_sim_path)
            return snapshot_path, aligned_sim_path

        def _queue_snapshot_capture(self, target_dir: Path) -> None:
            clear_preview_outputs(target_dir)
            self._pending_snapshot_target_dir = target_dir
            self._pending_snapshot_camera_state = snapshot_active_viewport_camera_state(self.stage)
            self._snapshot_capture_warmup_frames = CAPTURE_WARMUP_FRAMES
            self._set_status(f"Exported variant to {target_dir}; capturing the current viewport...")

        def _process_snapshot_capture(self) -> None:
            if self._snapshot_capture_task is not None:
                if not self._snapshot_capture_task.done():
                    return

                try:
                    snapshot_path, aligned_sim_path = self._snapshot_capture_task.result()
                except Exception as exc:
                    self._set_status(f"Snapshot capture failed: {exc}")
                else:
                    self._set_status(
                        f"Export complete. Refreshed {snapshot_path.name} and {aligned_sim_path.name} for "
                        f"{snapshot_path.parent.parent.name}"
                    )
                self._snapshot_capture_task = None
                return

            if self._pending_snapshot_target_dir is None:
                return
            if self._preview_update_requested or self._preview_warmup_frames > 0 or not self._preview_loaded:
                return
            if self._snapshot_capture_warmup_frames > 0:
                self._snapshot_capture_warmup_frames -= 1
                return

            target_dir = self._pending_snapshot_target_dir
            camera_state = self._pending_snapshot_camera_state
            self._pending_snapshot_target_dir = None
            self._pending_snapshot_camera_state = None
            self._set_status(f"Capturing the active viewport for {target_dir.name}...")
            self._snapshot_capture_task = asyncio.ensure_future(self._capture_snapshot_async(target_dir, camera_state))

        def _cycle_color_mode(self, delta: int) -> None:
            options = list(VALID_COLOR_MODES)
            current_index = options.index(self.color_mode)
            self.color_mode = options[(current_index + delta) % len(options)]
            self._queue_preview_update()

        def _reset_values(self) -> None:
            self.scale_xyz = list(DEFAULT_SCALE_XYZ)
            self.color_rgb = list(DEFAULT_COLOR_RGB)
            self.color_mode = DEFAULT_COLOR_MODE
            self.float_models["scale.x"].set_value(self.scale_xyz[0])
            self.float_models["scale.y"].set_value(self.scale_xyz[1])
            self.float_models["scale.z"].set_value(self.scale_xyz[2])
            self.float_models["color.r"].set_value(self.color_rgb[0])
            self.float_models["color.g"].set_value(self.color_rgb[1])
            self.float_models["color.b"].set_value(self.color_rgb[2])
            self._set_status("Reset scale/color to defaults.")
            self._queue_preview_update()

        def _set_next_free_index(self) -> None:
            next_index = suggest_next_index(self.source)
            self.target_index_model.set_value(next_index)
            self._set_status(f"Suggested next free index: {next_index}")

        def _cycle_object(self, delta: int) -> None:
            if not self._can_switch_source():
                return
            target_asset_id = self.catalog.cycle_asset(self.source.asset_id, delta)
            self._load_source_asset_by_id(target_asset_id, message=f"Loaded source asset {target_asset_id}.")

        def _choose_object_dir(self) -> None:
            self._sync_chooser_to_current_source()
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
            self.quick_asset_id = self.catalog.default_asset_id_in_category(category, preferred=self.quick_asset_id)
            self._set_string_model_value(self.quick_category_search_model, self.quick_category)
            self._set_string_model_value(self.quick_asset_search_model, self.quick_asset_id)
            self._rebuild_choose_dir_window()
            self._set_status(f"Chooser category set to {self.quick_category}.")

        def _apply_quick_asset_query(self) -> None:
            query = self._current_quick_asset_query()
            if not query:
                self._set_string_model_value(self.quick_asset_search_model, self.quick_asset_id)
                self._set_status("Enter an object id or substring before searching.")
                return

            if query in self.catalog.entries:
                asset_id = query
            else:
                lowered = query.lower()
                category_matches = [
                    asset_id
                    for asset_id in self.catalog.asset_ids_in_category(self.quick_category)
                    if lowered in asset_id.lower()
                ]
                matches = category_matches if category_matches else [
                    asset_id for asset_id in self.catalog.asset_ids() if lowered in asset_id.lower()
                ]
                if not matches:
                    self._set_status(f"No object id matched {query!r}.")
                    return
                asset_id = matches[0]

            entry = self.catalog.get_entry(asset_id)
            self.quick_category = entry.category
            self.quick_asset_id = asset_id
            self._set_string_model_value(self.quick_category_search_model, self.quick_category)
            self._set_string_model_value(self.quick_asset_search_model, self.quick_asset_id)
            self._rebuild_choose_dir_window()
            self._set_status(f"Chooser object set to {self.quick_asset_id}.")

        def _apply_quick_object_selection(self) -> None:
            if self.quick_asset_id not in self.catalog.entries:
                self._set_status(f"Quick object selection is invalid: {self.quick_asset_id!r}.")
                return
            if not self._can_switch_source():
                return
            selected_category = self.quick_category
            selected_asset_id = self.quick_asset_id
            self._close_choose_dir_window()
            self._load_source_asset_by_id(
                selected_asset_id,
                message=f"Loaded source asset {selected_asset_id} from category {selected_category}.",
            )

        def _cycle_quick_category(self, delta: int) -> None:
            categories = self.catalog.categories()
            if not categories:
                return
            self.quick_category = self._cycle_value(self.quick_category, categories, delta)
            self.quick_asset_id = self.catalog.default_asset_id_in_category(self.quick_category)
            self._set_string_model_value(self.quick_category_search_model, self.quick_category)
            self._set_string_model_value(self.quick_asset_search_model, self.quick_asset_id)
            self._rebuild_choose_dir_window()

        def _cycle_quick_object(self, delta: int) -> None:
            asset_ids = self.catalog.asset_ids_in_category(self.quick_category)
            if not asset_ids:
                return
            self.quick_asset_id = self._cycle_value(self.quick_asset_id, asset_ids, delta)
            self._set_string_model_value(self.quick_asset_search_model, self.quick_asset_id)
            self._rebuild_choose_dir_window()

        def _export_variant(self) -> None:
            if self._snapshot_capture_in_progress():
                self._set_status("Wait for the current snapshot capture to finish before exporting another variant.")
                return
            self._sync_values_from_models()
            try:
                target_index = self._current_target_index()
                target_dir = export_variant(
                    source=self.source,
                    target_index=target_index,
                    scale_xyz=self.scale_xyz,
                    color_rgb=self.color_rgb,
                    color_mode=self.color_mode,
                )
            except Exception as exc:
                self._set_status(f"Export failed: {exc}")
                return

            self.target_index_model.set_value(suggest_next_index(self.source))
            self._queue_snapshot_capture(target_dir)

        def _build_ui_window(self) -> None:
            window_kwargs = {
                "title": UI_TITLE,
                "width": MAIN_WINDOW_WIDTH,
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
            self.ui_dirty = False

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

        def _build_vector_row(self, title: str, keys: list[str], axis_labels: tuple[str, ...]) -> None:
            with ui.HStack(height=28, spacing=4):
                ui.Label(title, width=70)
                for axis_label, key in zip(axis_labels, keys, strict=True):
                    ui.Label(axis_label, width=16)
                    ui.FloatField(model=self.float_models[key], width=92)

        def _build_cycle_row(self, title: str, value: str, callback, width: int = 200) -> None:
            with ui.HStack(height=28, spacing=6):
                ui.Label(title, width=70)
                ui.Button("<", width=28, clicked_fn=lambda delta=-1: callback(delta))
                ui.Label(value, width=width)
                ui.Button(">", width=28, clicked_fn=lambda delta=1: callback(delta))

        def _build_choose_dir_contents(self) -> None:
            asset_ids = self.catalog.asset_ids_in_category(self.quick_category)
            asset_count = len(asset_ids)
            if self.quick_asset_id not in asset_ids and asset_ids:
                self.quick_asset_id = asset_ids[0]
                self._set_string_model_value(self.quick_asset_search_model, self.quick_asset_id)
            selected_entry = self.catalog.entries.get(self.quick_asset_id)
            selected_dir = str(selected_entry.asset_dir) if selected_entry is not None else "No object directory available."

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
                self._build_cycle_row("Object", self.quick_asset_id, self._cycle_quick_object, width=252)
                with ui.HStack(height=28, spacing=6):
                    ui.Label("Find Obj", width=70)
                    if hasattr(ui, "StringField"):
                        ui.StringField(model=self.quick_asset_search_model, width=250)
                    else:
                        ui.Label(self._current_quick_asset_query(), width=250)
                    ui.Button("Find", width=70, clicked_fn=self._apply_quick_asset_query)
                ui.Label(f"Objects in category: {asset_count}", height=22)
                ui.Label(selected_dir, word_wrap=True, height=64)
                with ui.HStack(height=30, spacing=8):
                    ui.Button("OK", width=90, clicked_fn=self._apply_quick_object_selection)
                    ui.Button("Cancel", width=90, clicked_fn=self._close_choose_dir_window)

        def _build_ui_contents(self) -> None:
            safe_target_index, safe_target_error = self._safe_target_index()
            if safe_target_index is None:
                current_target_asset_id = "invalid target index"
                current_target_dir = "<invalid>"
            else:
                current_target_asset_id = self.source.asset_id_for_index(safe_target_index)
                current_target_dir = str(self.source.target_dir_for_index(safe_target_index))
            bbox_center_obj_m, bbox_size_obj_m = compute_variant_bbox(self.source, self.scale_xyz)

            with ui.VStack(spacing=8, height=0):
                ui.Label(UI_TITLE, height=24)
                ui.Label(self.status_message, word_wrap=True, height=42)
                ui.Separator(height=6)

                self._build_cycle_row("Object", self.source.asset_id, self._cycle_object, width=252)
                with ui.HStack(height=30, spacing=8):
                    ui.Button("Choose Dir", width=100, clicked_fn=self._choose_object_dir)
                    ui.Button("Prev Dir", width=90, clicked_fn=lambda: self._cycle_object(-1))
                    ui.Button("Next Dir", width=90, clicked_fn=lambda: self._cycle_object(1))
                    ui.Button(
                        "Reload Preview",
                        width=120,
                        clicked_fn=lambda: self._queue_preview_update(reload_preview=True, frame_camera_view=True),
                    )
                ui.Label(
                    f"Category: {self.source.category_dir} | Objects: {len(self.catalog.asset_ids())} total",
                    word_wrap=True,
                    height=22,
                )
                ui.Label(f"Source: {self.source.asset_id}", height=22)
                ui.Label(str(self.source.asset_dir), word_wrap=True, height=36)
                ui.Label(
                    f"Core USD: {self.source.aligned_usd.name} | up_axis={self.source.up_axis} | "
                    f"base size(m)={rounded(self.source.base_size_m)}",
                    word_wrap=True,
                    height=38,
                )

                ui.Separator(height=6)
                ui.Label("Scale", height=22)
                self._build_vector_row("xyz", ["scale.x", "scale.y", "scale.z"], ("x", "y", "z"))
                ui.Label(f"Variant size(m): {rounded(bbox_size_obj_m)}", height=22)
                ui.Label(f"Variant center(m): {rounded(bbox_center_obj_m)}", height=22)

                ui.Separator(height=6)
                ui.Label("Color", height=22)
                self._build_vector_row("rgb", ["color.r", "color.g", "color.b"], ("r", "g", "b"))
                self._build_cycle_row("Mode", self.color_mode, self._cycle_color_mode)
                ui.Label(
                    "`tint` keeps diffuse textures and changes the material tint. "
                    "`flat` clears the diffuse texture input and uses the chosen constant color. "
                    "Default is `flat` so exported variants visibly change color.",
                    word_wrap=True,
                    height=72,
                )

                ui.Separator(height=6)
                ui.Label("Target", height=22)
                with ui.HStack(height=28, spacing=6):
                    ui.Label("Index", width=70)
                    if hasattr(ui, "StringField"):
                        ui.StringField(model=self.target_index_model, width=110)
                    else:
                        ui.Label(safe_target_index or "<invalid>", width=110)
                    ui.Button("Next Free", width=90, clicked_fn=self._set_next_free_index)
                ui.Label(f"Asset id: {current_target_asset_id}", height=22)
                ui.Label(current_target_dir, word_wrap=True, height=36)
                if safe_target_error is not None:
                    ui.Label(f"Index error: {safe_target_error}", word_wrap=True, height=36)
                else:
                    ui.Spacer(height=36)

                with ui.HStack(height=30, spacing=8):
                    ui.Button("Reset", width=80, clicked_fn=self._reset_values)
                    ui.Button("Export Variant", width=130, clicked_fn=self._export_variant)

                ui.Separator(height=6)
                ui.Label(
                    "Export copies the whole source directory, bakes xyz scale into the new "
                    "`Aligned.usd`, reapplies the material/display color override, regenerates "
                    "`Aligned.usda`, updates `object_parameters.json`, `item.py`, and "
                    "`description.py`, then re-captures `snapshot/Camera1.png` and "
                    "`Aligned_sim.png` from the viewport view you had when you clicked export.",
                    word_wrap=True,
                    height=98,
                )

        def run(self) -> None:
            print("=" * 88)
            print(UI_TITLE)
            print("=" * 88)
            print(f"Source asset : {self.source.asset_dir}")
            print(f"Core USD     : {self.source.aligned_usd}")
            print(f"Default next : {self._current_target_index()}")
            print("Mode         : interactive preview + baked export")
            print("=" * 88)
            while simulation_app.is_running() and self.running:
                self._process_preview_update()
                self._process_snapshot_capture()
                if self.ui_dirty:
                    self._rebuild_ui()
                simulation_app.update()
            simulation_app.close()

    editor = AssetVariantEditor(source)
    editor.run()
    return 0


def main() -> int:
    if ARGS.write_once:
        return run_write_once(ARGS)
    return run_interactive()


if __name__ == "__main__":
    raise SystemExit(main())
