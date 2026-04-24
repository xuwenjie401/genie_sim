#!/usr/bin/env python3
"""Normalize loose asset folders into GenieSim benchmark assets.

Target layout:
    <benchmark_root>/<prefix>_<category>/<asset_id>/

The script is intended to run inside the `issac` conda environment so the
`pxr` USD bindings are available for bounding-box computation.
"""

from __future__ import annotations

import argparse
import json
import pprint
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageOps, ImageTk
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageOps = None
    ImageTk = None


DEFAULT_SOURCE_ROOT = Path(
    "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects"
)
DEFAULT_ASSET_ROOT = Path(
    "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark"
)
DEFAULT_PREFIX = "web"
DEFAULT_MASS = 0.01
DEFAULT_UP_AXIS = "y"
TARGET_USD_FILENAME = "Aligned.usd"
TARGET_USDA_FILENAME = "Aligned.usda"

SUPPORTED_MODEL_NAMES = (
    "Aligned_obj.usd",
    "Aligned_obj.usda",
    "Aligned.usd",
    "Aligned.usda",
)
PREFERRED_PREVIEW_NAMES = (
    "Aligned_sim.png",
    "Aligned_sim.jpg",
    "Aligned_sim.jpeg",
)
PREVIEW_SUFFIXES = {".png", ".jpg", ".jpeg"}
GENERIC_SOURCE_NAMES = {
    "",
    "assets",
    "asset",
    "inner",
    "outer",
    "objects",
    "object",
    "benchmark",
    "pre_train_place",
    "pre-train-place",
    "geniesimassets",
}
UI_FONT_CANDIDATES = [
    "Cabin",
    "Aptos",
    "Noto Sans",
    "Helvetica",
    "Segoe UI",
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
MONO_FONT_CANDIDATES = [
    "JetBrains Mono",
    "Menlo",
    "Consolas",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Courier New",
    "Courier",
]


@dataclass(frozen=True)
class SourceCandidate:
    source_dir: Path
    model_path: Path
    preview_path: Path | None
    category_hint: str
    index_hint: str
    english_name_hint: str


@dataclass(frozen=True)
class BoundingBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class NormalizationSpec:
    asset_root: Path
    prefix: str
    category: str
    index: str
    english_name: str
    chinese_name: str
    mass: float
    overwrite: bool
    up_axis: str = DEFAULT_UP_AXIS

    @property
    def prefix_token(self) -> str:
        return sanitize_token(self.prefix or DEFAULT_PREFIX)

    @property
    def category_token(self) -> str:
        return sanitize_token(self.category or "asset")

    @property
    def index_token(self) -> str:
        raw = re.sub(r"[^0-9A-Za-z]+", "", self.index.strip())
        return raw.lower() or "0000"

    @property
    def category_dir(self) -> str:
        if self.prefix_token:
            return f"{self.prefix_token}_{self.category_token}"
        return self.category_token

    @property
    def asset_id(self) -> str:
        return f"benchmark_{self.category_dir}_{self.index_token}"

    @property
    def object_dir(self) -> Path:
        return self.asset_root / self.category_dir / self.asset_id

    @property
    def semantic_name(self) -> str:
        english = normalize_name_text(self.english_name)
        if english:
            return english.lower()
        return humanize_token(self.category_token)

    @property
    def resolved_english_name(self) -> str:
        english = normalize_name_text(self.english_name)
        return english or humanize_token(self.category_token)

    @property
    def resolved_chinese_name(self) -> str:
        chinese = normalize_name_text(self.chinese_name)
        return chinese or self.resolved_english_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive UI for converting loose asset folders into "
            "GenieSim benchmark assets."
        )
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help="Benchmark root directory. New assets are written under <benchmark_root>/<prefix>_<category>/.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Optional source root or single asset directory to pre-load.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Default category prefix. Target category becomes <prefix>_<category>.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=DEFAULT_MASS,
        help="Default object mass written into object_parameters.json.",
    )
    parser.add_argument(
        "--write-once",
        action="store_true",
        help="Normalize a single source directory without opening the UI.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Override inferred category token for --write-once.",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Override inferred index token for --write-once.",
    )
    parser.add_argument(
        "--english-name",
        default="",
        help="Override english_name for --write-once.",
    )
    parser.add_argument(
        "--chinese-name",
        default="",
        help="Override chinese_name for --write-once.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing destination directory.",
    )
    return parser.parse_args()


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


def rounded(values: list[float] | tuple[float, ...], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def is_normalized_asset_dir(path: Path) -> bool:
    return (path / "object_parameters.json").exists() or (path / "description.py").exists()


def find_model_path(source_dir: Path) -> Path | None:
    for name in SUPPORTED_MODEL_NAMES:
        candidate = source_dir / name
        if candidate.exists():
            return candidate
    return None


def find_preview_path(source_dir: Path) -> Path | None:
    for name in PREFERRED_PREVIEW_NAMES:
        candidate = source_dir / name
        if candidate.exists():
            return candidate

    preferred_snapshot = source_dir / "snapshot" / "Camera1.png"
    if preferred_snapshot.exists():
        return preferred_snapshot

    snapshot_dir = source_dir / "snapshot"
    if snapshot_dir.exists():
        snapshot_candidates = sorted(
            path for path in snapshot_dir.iterdir() if path.suffix.lower() in PREVIEW_SUFFIXES
        )
        if snapshot_candidates:
            return snapshot_candidates[0]

    top_level_images = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in PREVIEW_SUFFIXES
    )
    if top_level_images:
        return top_level_images[0]
    return None


def infer_index_token(source_dir: Path) -> str:
    match = re.search(r"(\d+)$", source_dir.name)
    if match:
        return match.group(1)
    return "0000"


def infer_category_token(source_dir: Path) -> str:
    candidates: list[str] = []

    parent_token = sanitize_token(source_dir.parent.name)
    if parent_token not in GENERIC_SOURCE_NAMES:
        candidates.append(parent_token)

    base_name = re.sub(r"[_-]?\d+$", "", source_dir.name)
    base_token = sanitize_token(base_name)
    if base_token:
        candidates.append(base_token)

    if "-" in base_name:
        tail = sanitize_token(base_name.rsplit("-", 1)[-1])
        if tail:
            candidates.append(tail)

    for candidate in candidates:
        if candidate and candidate not in GENERIC_SOURCE_NAMES:
            return candidate
    return "asset"


def build_source_candidate(source_dir: Path) -> SourceCandidate | None:
    source_dir = source_dir.expanduser().resolve()
    model_path = find_model_path(source_dir)
    if model_path is None:
        return None
    category_hint = infer_category_token(source_dir)
    return SourceCandidate(
        source_dir=source_dir,
        model_path=model_path,
        preview_path=find_preview_path(source_dir),
        category_hint=category_hint,
        index_hint=infer_index_token(source_dir),
        english_name_hint=humanize_token(category_hint),
    )


def discover_source_candidates(source_path: Path) -> list[SourceCandidate]:
    source_path = source_path.expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if source_path.is_file():
        source_path = source_path.parent

    direct_candidate = build_source_candidate(source_path)
    if direct_candidate is not None:
        return [direct_candidate]

    candidate_dirs: set[Path] = set()
    for model_name in SUPPORTED_MODEL_NAMES:
        for model_path in source_path.rglob(model_name):
            parent = model_path.parent
            if is_normalized_asset_dir(parent):
                continue
            candidate_dirs.add(parent.resolve())

    candidates = []
    for candidate_dir in sorted(candidate_dirs):
        candidate = build_source_candidate(candidate_dir)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


@lru_cache(maxsize=256)
def compute_bbox_from_usd(usd_path_str: str) -> BoundingBox:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "pxr is unavailable. Run this tool inside the `issac` conda environment."
        ) from exc

    usd_path = Path(usd_path_str)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    prim = stage.GetDefaultPrim()
    if prim is None or not prim.IsValid():
        prim = stage.GetPseudoRoot()

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
    )
    aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    min_point = aligned_box.GetMin()
    max_point = aligned_box.GetMax()

    size = (
        float(max_point[0] - min_point[0]),
        float(max_point[1] - min_point[1]),
        float(max_point[2] - min_point[2]),
    )
    center = (
        float((max_point[0] + min_point[0]) / 2.0),
        float((max_point[1] + min_point[1]) / 2.0),
        float((max_point[2] + min_point[2]) / 2.0),
    )
    if any(value <= 0 for value in size):
        raise RuntimeError(f"Computed invalid bounding box for {usd_path}: size={size}")
    return BoundingBox(center=center, size=size)


def _xform_op_is_identity(op: Any, UsdGeom: Any) -> bool:
    value = op.Get()
    tolerance = 1e-6
    op_type = op.GetOpType()

    if op_type == UsdGeom.XformOp.TypeTranslate:
        return all(abs(float(value[i])) <= tolerance for i in range(3))
    if op_type == UsdGeom.XformOp.TypeScale:
        return all(abs(float(value[i]) - 1.0) <= tolerance for i in range(3))
    if op_type in {
        UsdGeom.XformOp.TypeRotateX,
        UsdGeom.XformOp.TypeRotateY,
        UsdGeom.XformOp.TypeRotateZ,
    }:
        return abs(float(value)) <= tolerance
    if op_type in {
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeRotateXZY,
        UsdGeom.XformOp.TypeRotateYXZ,
        UsdGeom.XformOp.TypeRotateYZX,
        UsdGeom.XformOp.TypeRotateZXY,
        UsdGeom.XformOp.TypeRotateZYX,
    }:
        return all(abs(float(value[i])) <= tolerance for i in range(3))
    if op_type == UsdGeom.XformOp.TypeOrient:
        imaginary = value.GetImaginary()
        return (
            abs(float(value.GetReal()) - 1.0) <= tolerance
            and all(abs(float(imaginary[i])) <= tolerance for i in range(3))
        )
    if op_type == UsdGeom.XformOp.TypeTransform:
        identity = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        return all(abs(float(value[row][col]) - identity[row][col]) <= tolerance for row in range(4) for col in range(4))
    return False


def _stage_requires_usd_normalization(stage: Any, UsdGeom: Any) -> bool:
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(meters_per_unit - 1.0) > 1e-6:
        return True
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        return True

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if not _xform_op_is_identity(op, UsdGeom):
                return True
    return False


def _find_promotable_child_rigidbody(stage: Any, Usd: Any, UsdGeom: Any, UsdPhysics: Any) -> Any | None:
    root_prim = stage.GetDefaultPrim()
    if root_prim is None or not root_prim.IsValid():
        return None
    if root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return None

    rigid_children = [child for child in root_prim.GetChildren() if child.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(rigid_children) != 1:
        return None

    rigid_child = rigid_children[0]
    if rigid_child.IsA(UsdGeom.Xformable):
        xformable = UsdGeom.Xformable(rigid_child)
        if xformable.GetOrderedXformOps():
            return None

    for prim in Usd.PrimRange(rigid_child):
        if prim != rigid_child and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return None

    return rigid_child


def _transform_points_with_matrix(points: Any, matrix: Any, Gf: Any) -> list[tuple[float, float, float]]:
    transformed_points: list[tuple[float, float, float]] = []
    for point in points:
        transformed_point = matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        transformed_points.append(
            (
                float(transformed_point[0]),
                float(transformed_point[1]),
                float(transformed_point[2]),
            )
        )
    return transformed_points


def _transform_normals_with_matrix(normals: Any, matrix: Any, Gf: Any) -> list[tuple[float, float, float]]:
    transformed_normals: list[tuple[float, float, float]] = []
    for normal in normals:
        transformed_normal = matrix.TransformDir(Gf.Vec3d(float(normal[0]), float(normal[1]), float(normal[2])))
        length = transformed_normal.GetLength()
        if length > 1e-8:
            transformed_normal /= length
        transformed_normals.append(
            (
                float(transformed_normal[0]),
                float(transformed_normal[1]),
                float(transformed_normal[2]),
            )
        )
    return transformed_normals


def _normalize_exported_primary_usd(target_path: Path) -> None:
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "pxr is unavailable. Run this tool inside the `issac` conda environment."
        ) from exc

    stage = Usd.Stage.Open(str(target_path))
    if stage is None:
        raise RuntimeError(f"Failed to open exported USD for normalization: {target_path}")

    mesh_world_transforms: list[tuple[Any, Any]] = []
    xform_properties_to_remove: list[tuple[Any, list[str]]] = []
    time_code = Usd.TimeCode.Default()

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(prim)
            op_names = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
            if op_names:
                xform_properties_to_remove.append((prim, op_names))
        if prim.IsA(UsdGeom.Mesh):
            mesh_world_transforms.append((UsdGeom.Mesh(prim), UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)))

    for mesh, world_transform in mesh_world_transforms:
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get(time_code)
        if points:
            points_attr.Set(_transform_points_with_matrix(points, world_transform, Gf), time_code)

        normals_attr = mesh.GetNormalsAttr()
        normals = normals_attr.Get(time_code)
        if normals:
            normals_attr.Set(_transform_normals_with_matrix(normals, world_transform, Gf), time_code)

    for prim, op_names in xform_properties_to_remove:
        for op_name in op_names:
            prim.RemoveProperty(op_name)
        prim.RemoveProperty("xformOpOrder")

    promotable_rigid_child = _find_promotable_child_rigidbody(stage, Usd, UsdGeom, UsdPhysics)
    if promotable_rigid_child is not None:
        root_prim = stage.GetDefaultPrim()
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        if promotable_rigid_child.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(root_prim)

        for attr in promotable_rigid_child.GetAttributes():
            attr_name = attr.GetName()
            if not attr_name.startswith("physics:"):
                continue
            if not attr.HasAuthoredValueOpinion():
                continue
            value = attr.Get(time_code)
            if value is None:
                continue
            root_attr = root_prim.GetAttribute(attr_name)
            if not root_attr or not root_attr.IsValid():
                root_attr = root_prim.CreateAttribute(attr_name, attr.GetTypeName(), attr.IsCustom())
            root_attr.Set(value, time_code)

        if promotable_rigid_child.HasAPI(UsdPhysics.RigidBodyAPI):
            promotable_rigid_child.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if promotable_rigid_child.HasAPI(UsdPhysics.MassAPI):
            promotable_rigid_child.RemoveAPI(UsdPhysics.MassAPI)

        for attr in list(promotable_rigid_child.GetAttributes()):
            if attr.GetName().startswith("physics:"):
                promotable_rigid_child.RemoveProperty(attr.GetName())

        print(
            f"Promoted rigid body from {promotable_rigid_child.GetPath()} "
            f"to {root_prim.GetPath()} in {target_path}"
        )

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.Save()


def target_model_filename() -> str:
    return TARGET_USD_FILENAME


def build_object_parameters(
    spec: NormalizationSpec,
    bbox: BoundingBox,
    source_model_path: Path,
) -> dict[str, Any]:
    size = rounded(bbox.size)
    category_label = humanize_token(spec.category_token)
    semantic_name = spec.semantic_name
    english_name = spec.resolved_english_name
    full_description = f"{english_name}."
    return {
        "object_id": spec.asset_id,
        "materialOptions": [],
        "size": size,
        "scale": 1,
        "unit": "m",
        "model_path": (
            f"objects/benchmark/{spec.category_dir}/{spec.asset_id}/"
            f"{target_model_filename()}"
        ),
        "upAxis": [spec.up_axis],
        "mass": round(float(spec.mass), 6),
        "original_model_path": str(source_model_path),
        "semantic_name": semantic_name,
        "llm_descriptions": {
            "semantic_name": [semantic_name],
            "object_category": [category_label],
            "dimensions": size,
            "unit": "m",
            "full_description": [full_description],
        },
    }


def build_description_mapping(spec: NormalizationSpec, bbox: BoundingBox) -> dict[str, Any]:
    size = rounded(bbox.size)
    category_label = humanize_token(spec.category_token)
    english_name = spec.resolved_english_name
    return {
        "semantic_name": [spec.semantic_name],
        "english_name": english_name,
        "chinese_name": spec.resolved_chinese_name,
        "object_category": [category_label],
        "color": "",
        "shape": "",
        "materials": [],
        "dimensions": size,
        "unit": "m",
        "descriptive_terms": [],
        "full_description": f"{english_name}.",
    }


def _item_space_geometry(center: tuple[float, float, float], size: tuple[float, float, float], up_axis: str) -> tuple[list[float], list[float]]:
    axis = up_axis.lstrip("+-").lower()
    if axis == "y":
        return (
            rounded([center[0], center[2], -center[1]], digits=6),
            rounded([size[0], size[2], size[1]], digits=6),
        )
    if axis == "x":
        return (
            rounded([center[2], center[1], -center[0]], digits=6),
            rounded([size[2], size[1], size[0]], digits=6),
        )
    return rounded(center, digits=6), rounded(size, digits=6)


def build_item_mapping(spec: NormalizationSpec, bbox: BoundingBox) -> dict[str, Any]:
    bbox_position, bbox_size = _item_space_geometry(bbox.center, bbox.size, spec.up_axis)
    return {
        "up_axis": spec.up_axis,
        "has_joint": False,
        "has_articulation": False,
        "id": spec.asset_id,
        "size": rounded(bbox.size),
        "shapes": [
            {
                "name": "bbox",
                "type": "cube",
                "position": bbox_position,
                "quaternion": [0.0, 0.0, 0.0, 1.0],
                "size": bbox_size,
                "scale": bbox_size,
            },
            {
                "name": "origin",
                "type": "sphere",
                "position": [0.0, 0.0, 0.0],
                "quaternion": [0.0, 0.0, 0.0, 1.0],
                "size": [0.001, 0.001, 0.001],
                "scale": [0.001, 0.001, 0.001],
            },
        ],
    }


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def serialize_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=4, ensure_ascii=False) + "\n"


def serialize_python_mapping(data: dict[str, Any]) -> str:
    return pprint.pformat(data, indent=4, width=100, sort_dicts=False) + "\n"


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


def build_z_up_usda_text(spec: NormalizationSpec, bbox: BoundingBox) -> str:
    bbox_position, bbox_size = _item_space_geometry(bbox.center, bbox.size, spec.up_axis)
    entity_orient = z_up_quaternion_wxyz(spec.up_axis)
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
        prepend payload = @./{TARGET_USD_FILENAME}@
    )
    {{
        float physics:mass = {_format_scalar(spec.mass)}
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


def write_primary_usd(source_model_path: Path, destination_dir: Path) -> Path:
    target_path = destination_dir / TARGET_USD_FILENAME

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "pxr is unavailable. Run this tool inside the `issac` conda environment."
        ) from exc

    stage = Usd.Stage.Open(str(source_model_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage for export: {source_model_path}")
    if source_model_path.suffix.lower() == ".usd":
        shutil.copy2(source_model_path, target_path)
    elif not stage.Export(str(target_path)):
        raise RuntimeError(f"Failed to export {source_model_path} to {target_path}")

    if _stage_requires_usd_normalization(stage, UsdGeom) or _find_promotable_child_rigidbody(
        stage, Usd, UsdGeom, UsdPhysics
    ):
        _normalize_exported_primary_usd(target_path)

    return target_path


def copy_source_tree(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        target = destination_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def ensure_snapshot_preview(destination_dir: Path, preview_path: Path | None) -> None:
    if preview_path is None or not preview_path.exists():
        return
    snapshot_dir = destination_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    target_path = snapshot_dir / "Camera1.png"
    if preview_path.resolve() == target_path.resolve():
        return
    shutil.copy2(preview_path, target_path)


def normalize_candidate(
    candidate: SourceCandidate,
    spec: NormalizationSpec,
    bbox: BoundingBox,
) -> Path:
    destination_dir = spec.object_dir
    if destination_dir.exists() and not spec.overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination_dir}\n"
            "Enable overwrite to write into the existing directory."
        )

    copy_source_tree(candidate.source_dir, destination_dir)

    write_primary_usd(candidate.model_path, destination_dir)
    write_text_atomic(
        destination_dir / TARGET_USDA_FILENAME,
        build_z_up_usda_text(spec, bbox),
    )
    ensure_snapshot_preview(destination_dir, candidate.preview_path)

    object_parameters = build_object_parameters(spec, bbox, candidate.model_path)
    description = build_description_mapping(spec, bbox)
    item_mapping = build_item_mapping(spec, bbox)

    write_text_atomic(
        destination_dir / "object_parameters.json",
        serialize_json(object_parameters),
    )
    write_text_atomic(
        destination_dir / "description.py",
        serialize_json(description),
    )
    write_text_atomic(
        destination_dir / "item.py",
        serialize_python_mapping(item_mapping),
    )
    return destination_dir


def choose_font_family(root: tk.Misc, candidates: list[str], fallback: str) -> str:
    available = {family.lower(): family for family in tkfont.families(root)}
    for candidate in candidates:
        family = available.get(candidate.lower())
        if family:
            return family
    return fallback


class AssetNormalizerApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        asset_root: Path,
        initial_source_dir: Path | None,
        default_prefix: str,
        default_mass: float,
    ) -> None:
        self.root = root
        self.root.title("Benchmark Asset Normalizer")
        self.root.geometry("1400x900")
        self.root.minsize(1180, 760)

        self.asset_root_var = tk.StringVar(value=str(asset_root))
        self.source_path_var = tk.StringVar(value=str(initial_source_dir or DEFAULT_SOURCE_ROOT))
        self.prefix_var = tk.StringVar(value=default_prefix)
        self.mass_var = tk.StringVar(value=f"{default_mass:.4f}")
        self.category_var = tk.StringVar(value="")
        self.index_var = tk.StringVar(value="")
        self.english_name_var = tk.StringVar(value="")
        self.chinese_name_var = tk.StringVar(value="")
        self.overwrite_var = tk.BooleanVar(value=False)

        self.scan_info_var = tk.StringVar(value="No source scanned")
        self.status_var = tk.StringVar(value="Select a source path to begin.")
        self.source_dir_var = tk.StringVar(value="-")
        self.source_model_var = tk.StringVar(value="-")
        self.preview_var = tk.StringVar(value="-")
        self.size_var = tk.StringVar(value="-")
        self.center_var = tk.StringVar(value="-")
        self.category_dir_var = tk.StringVar(value="-")
        self.asset_id_var = tk.StringVar(value="-")
        self.output_dir_var = tk.StringVar(value="-")
        self.output_model_var = tk.StringVar(value="-")

        self.candidates: list[SourceCandidate] = []
        self.current_index = -1
        self.current_photo: ImageTk.PhotoImage | tk.PhotoImage | None = None
        self.completed_sources: set[Path] = set()

        self._build_styles()
        self._build_ui()
        self._attach_traces()

        if initial_source_dir is not None:
            self.scan_source_path()

    def _build_styles(self) -> None:
        self.ui_font = choose_font_family(self.root, UI_FONT_CANDIDATES, "TkDefaultFont")
        self.mono_font = choose_font_family(self.root, MONO_FONT_CANDIDATES, "TkFixedFont")
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family=self.ui_font, size=11)

        title_font = tkfont.Font(family=self.ui_font, size=16, weight="bold")
        mono_font = tkfont.Font(family=self.mono_font, size=10)

        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=title_font)
        style.configure("Muted.TLabel", foreground="#5f6368")
        style.configure("Mono.TLabel", font=mono_font)
        style.configure("Path.TEntry", font=mono_font)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        header = ttk.Frame(self.root, padding=(18, 18, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Benchmark Asset Normalizer", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text=(
                "Select one loose asset directory or scan a directory tree. "
                "Then adjust fields and write a benchmark-style asset folder."
            ),
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        source_bar = ttk.LabelFrame(self.root, text="Source", padding=12)
        source_bar.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))
        source_bar.columnconfigure(1, weight=1)
        ttk.Label(source_bar, text="Path").grid(row=0, column=0, sticky="w")
        ttk.Entry(
            source_bar,
            textvariable=self.source_path_var,
            style="Path.TEntry",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(source_bar, text="Browse", command=self.choose_source_path).grid(
            row=0, column=2, padx=(0, 6)
        )
        ttk.Button(source_bar, text="Scan", command=self.scan_source_path).grid(row=0, column=3)
        ttk.Label(source_bar, textvariable=self.scan_info_var, style="Muted.TLabel").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )

        target_bar = ttk.LabelFrame(self.root, text="Target Defaults", padding=12)
        target_bar.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))
        for column in range(7):
            target_bar.columnconfigure(column, weight=1 if column == 1 else 0)

        ttk.Label(target_bar, text="Benchmark Root").grid(row=0, column=0, sticky="w")
        ttk.Entry(
            target_bar,
            textvariable=self.asset_root_var,
            style="Path.TEntry",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(target_bar, text="Browse", command=self.choose_asset_root).grid(
            row=0, column=2, padx=(0, 12)
        )
        ttk.Label(target_bar, text="Prefix").grid(row=0, column=3, sticky="e")
        ttk.Entry(target_bar, textvariable=self.prefix_var, width=12).grid(
            row=0, column=4, sticky="w", padx=(8, 12)
        )
        ttk.Label(target_bar, text="Mass").grid(row=0, column=5, sticky="e")
        ttk.Entry(target_bar, textvariable=self.mass_var, width=12).grid(
            row=0, column=6, sticky="w", padx=(8, 0)
        )
        ttk.Checkbutton(
            target_bar,
            text="Overwrite Existing",
            variable=self.overwrite_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.grid(row=3, column=0, sticky="nsew", padx=18, pady=(0, 18))

        left = ttk.Frame(content, padding=(0, 0, 12, 0))
        right = ttk.Frame(content)
        content.add(left, weight=1)
        content.add(right, weight=3)

        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Discovered Assets").grid(row=0, column=0, sticky="w")
        list_frame = ttk.Frame(left)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.candidate_listbox = tk.Listbox(
            list_frame,
            activestyle="none",
            exportselection=False,
            font=(self.mono_font, 10),
        )
        self.candidate_listbox.grid(row=0, column=0, sticky="nsew")
        self.candidate_listbox.bind("<<ListboxSelect>>", self.on_candidate_selected)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.candidate_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.candidate_listbox.configure(yscrollcommand=scrollbar.set)

        nav_buttons = ttk.Frame(left)
        nav_buttons.grid(row=2, column=0, sticky="ew")
        nav_buttons.columnconfigure(0, weight=1)
        nav_buttons.columnconfigure(1, weight=1)
        ttk.Button(nav_buttons, text="Prev", command=lambda: self.change_candidate(-1)).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(nav_buttons, text="Next", command=lambda: self.change_candidate(1)).grid(
            row=0, column=1, sticky="ew"
        )

        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.columnconfigure(0, weight=1)

        details = ttk.Frame(right)
        details.grid(row=0, column=0, sticky="nsew")
        details.columnconfigure(0, weight=3)
        details.columnconfigure(1, weight=2)
        details.rowconfigure(0, weight=1)

        meta = ttk.LabelFrame(details, text="Source Details", padding=12)
        meta.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        meta.columnconfigure(1, weight=1)
        for row in range(6):
            meta.rowconfigure(row, weight=0)

        meta_rows = [
            ("Source Dir", self.source_dir_var),
            ("Source Model", self.source_model_var),
            ("Preview", self.preview_var),
            ("BBox Size", self.size_var),
            ("BBox Center", self.center_var),
            ("Output Dir", self.output_dir_var),
        ]
        for row, (label, variable) in enumerate(meta_rows):
            ttk.Label(meta, text=label).grid(row=row, column=0, sticky="nw", pady=(0, 6))
            ttk.Label(
                meta,
                textvariable=variable,
                style="Mono.TLabel",
                wraplength=700,
                justify=tk.LEFT,
            ).grid(row=row, column=1, sticky="w", pady=(0, 6))

        preview_frame = ttk.LabelFrame(details, text="Preview", padding=12)
        preview_frame.grid(row=0, column=1, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(
            preview_frame,
            text="No preview",
            anchor=tk.CENTER,
            justify=tk.CENTER,
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        form = ttk.LabelFrame(right, text="Metadata", padding=12)
        form.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for column in range(4):
            form.columnconfigure(column, weight=1)

        ttk.Label(form, text="Category").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.category_var).grid(row=0, column=1, sticky="ew", padx=(8, 12))
        ttk.Label(form, text="Index").grid(row=0, column=2, sticky="w")
        ttk.Entry(form, textvariable=self.index_var).grid(row=0, column=3, sticky="ew", padx=(8, 0))

        ttk.Label(form, text="english_name").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(form, textvariable=self.english_name_var).grid(
            row=1, column=1, sticky="ew", padx=(8, 12), pady=(10, 0)
        )
        ttk.Label(form, text="chinese_name").grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Entry(form, textvariable=self.chinese_name_var).grid(
            row=1, column=3, sticky="ew", padx=(8, 0), pady=(10, 0)
        )

        ttk.Label(form, text="Category Dir").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Label(form, textvariable=self.category_dir_var, style="Mono.TLabel").grid(
            row=2, column=1, sticky="w", padx=(8, 12), pady=(10, 0)
        )
        ttk.Label(form, text="Asset ID").grid(row=2, column=2, sticky="w", pady=(10, 0))
        ttk.Label(form, textvariable=self.asset_id_var, style="Mono.TLabel").grid(
            row=2, column=3, sticky="w", padx=(8, 0), pady=(10, 0)
        )

        ttk.Label(form, text="Output Model").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Label(form, textvariable=self.output_model_var, style="Mono.TLabel").grid(
            row=3, column=1, columnspan=3, sticky="w", padx=(8, 0), pady=(10, 0)
        )

        actions = ttk.Frame(form)
        actions.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(14, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        ttk.Button(actions, text="Reset From Source", command=self.reset_form_from_current).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(actions, text="Create Asset", command=self.create_current_asset).grid(
            row=0, column=1, sticky="ew", padx=(0, 6)
        )
        ttk.Button(actions, text="Create And Next", command=lambda: self.create_current_asset(advance=True)).grid(
            row=0, column=2, sticky="ew"
        )

        footer = ttk.Frame(self.root, padding=(18, 0, 18, 18))
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")

    def _attach_traces(self) -> None:
        for variable in (
            self.asset_root_var,
            self.prefix_var,
            self.category_var,
            self.index_var,
            self.english_name_var,
            self.chinese_name_var,
        ):
            variable.trace_add("write", lambda *_args: self.refresh_computed_fields())

    def choose_source_path(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.root,
            title="Choose a loose asset directory or a directory tree to scan",
            initialdir=self.source_path_var.get() or str(Path.home()),
        )
        if selected:
            self.source_path_var.set(selected)

    def choose_asset_root(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.root,
            title="Choose benchmark root",
            initialdir=self.asset_root_var.get() or str(DEFAULT_ASSET_ROOT),
        )
        if selected:
            self.asset_root_var.set(selected)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def scan_source_path(self) -> None:
        source_text = self.source_path_var.get().strip()
        if not source_text:
            messagebox.showwarning("Scan Source", "Choose a source path first.")
            return

        source_path = Path(source_text).expanduser()
        try:
            self.candidates = discover_source_candidates(source_path)
        except Exception as exc:
            messagebox.showerror("Scan Source", str(exc))
            self.set_status(f"Scan failed: {exc}")
            return

        self.candidate_listbox.delete(0, tk.END)
        for candidate in self.candidates:
            label = self._candidate_label(candidate)
            self.candidate_listbox.insert(tk.END, label)

        if self.candidates:
            self.scan_info_var.set(f"Found {len(self.candidates)} candidate asset directories")
            self.select_candidate(0)
            self.set_status(f"Loaded {len(self.candidates)} candidate asset directories.")
        else:
            self.scan_info_var.set("No loose asset directories found")
            self.clear_current_candidate()
            self.set_status("No loose asset directories found under the selected path.")

    def _candidate_label(self, candidate: SourceCandidate) -> str:
        prefix = "OK " if candidate.source_dir in self.completed_sources else "   "
        return (
            f"{prefix}{candidate.source_dir.name}  |  "
            f"{candidate.category_hint}  |  {candidate.index_hint}"
        )

    def refresh_candidate_labels(self) -> None:
        current = self.current_index
        self.candidate_listbox.delete(0, tk.END)
        for candidate in self.candidates:
            self.candidate_listbox.insert(tk.END, self._candidate_label(candidate))
        if self.candidates and 0 <= current < len(self.candidates):
            self.candidate_listbox.selection_set(current)
            self.candidate_listbox.see(current)

    def on_candidate_selected(self, _event: tk.Event[Any] | None = None) -> None:
        selection = self.candidate_listbox.curselection()
        if not selection:
            return
        self.select_candidate(int(selection[0]))

    def change_candidate(self, delta: int) -> None:
        if not self.candidates:
            return
        new_index = min(max(self.current_index + delta, 0), len(self.candidates) - 1)
        self.select_candidate(new_index)

    def select_candidate(self, index: int) -> None:
        if not (0 <= index < len(self.candidates)):
            return
        self.current_index = index
        self.candidate_listbox.selection_clear(0, tk.END)
        self.candidate_listbox.selection_set(index)
        self.candidate_listbox.activate(index)
        self.candidate_listbox.see(index)
        self.reset_form_from_current()

    def clear_current_candidate(self) -> None:
        self.current_index = -1
        self.source_dir_var.set("-")
        self.source_model_var.set("-")
        self.preview_var.set("-")
        self.size_var.set("-")
        self.center_var.set("-")
        self.category_var.set("")
        self.index_var.set("")
        self.english_name_var.set("")
        self.chinese_name_var.set("")
        self.category_dir_var.set("-")
        self.asset_id_var.set("-")
        self.output_dir_var.set("-")
        self.output_model_var.set("-")
        self._set_preview_image(None)

    def current_candidate(self) -> SourceCandidate | None:
        if 0 <= self.current_index < len(self.candidates):
            return self.candidates[self.current_index]
        return None

    def reset_form_from_current(self) -> None:
        candidate = self.current_candidate()
        if candidate is None:
            return

        self.source_dir_var.set(str(candidate.source_dir))
        self.source_model_var.set(str(candidate.model_path))
        self.preview_var.set(str(candidate.preview_path) if candidate.preview_path else "-")

        try:
            bbox = compute_bbox_from_usd(str(candidate.model_path))
            self.size_var.set(str(rounded(bbox.size)))
            self.center_var.set(str(rounded(bbox.center)))
        except Exception as exc:
            self.size_var.set(f"Error: {exc}")
            self.center_var.set("Error")
            self.set_status(f"Bounding-box computation failed for {candidate.model_path}: {exc}")

        self.category_var.set(candidate.category_hint)
        self.index_var.set(candidate.index_hint)
        self.english_name_var.set(candidate.english_name_hint)
        self.chinese_name_var.set(candidate.english_name_hint)
        self._set_preview_image(candidate.preview_path)
        self.refresh_computed_fields()

    def refresh_computed_fields(self) -> None:
        candidate = self.current_candidate()
        if candidate is None:
            self.category_dir_var.set("-")
            self.asset_id_var.set("-")
            self.output_dir_var.set("-")
            self.output_model_var.set("-")
            return

        try:
            spec = self.build_current_spec(strict=False)
        except Exception:
            self.category_dir_var.set("-")
            self.asset_id_var.set("-")
            self.output_dir_var.set("-")
            self.output_model_var.set("-")
            return

        self.category_dir_var.set(spec.category_dir)
        self.asset_id_var.set(spec.asset_id)
        self.output_dir_var.set(str(spec.object_dir))
        self.output_model_var.set(f"{TARGET_USD_FILENAME} + {TARGET_USDA_FILENAME}")

    def build_current_spec(self, *, strict: bool) -> NormalizationSpec:
        candidate = self.current_candidate()
        if candidate is None:
            raise RuntimeError("No asset is selected.")

        asset_root_raw = self.asset_root_var.get().strip() or str(DEFAULT_ASSET_ROOT)
        asset_root = Path(asset_root_raw).expanduser()

        prefix = self.prefix_var.get().strip() or DEFAULT_PREFIX
        category = self.category_var.get().strip() or candidate.category_hint
        index = self.index_var.get().strip() or candidate.index_hint
        english_name = self.english_name_var.get().strip() or candidate.english_name_hint
        chinese_name = self.chinese_name_var.get().strip() or english_name

        mass_raw = self.mass_var.get().strip() or f"{DEFAULT_MASS}"
        try:
            mass = float(mass_raw)
        except ValueError as exc:
            if strict:
                raise ValueError(f"Invalid mass value: {mass_raw}") from exc
            mass = DEFAULT_MASS

        return NormalizationSpec(
            asset_root=asset_root,
            prefix=prefix,
            category=category,
            index=index,
            english_name=english_name,
            chinese_name=chinese_name,
            mass=mass,
            overwrite=self.overwrite_var.get(),
        )

    def _set_preview_image(self, preview_path: Path | None) -> None:
        self.current_photo = None
        if preview_path is None or not preview_path.exists():
            self.preview_label.configure(image="", text="No preview")
            return

        if Image is None or ImageTk is None:  # pragma: no cover - fallback path
            try:
                photo = tk.PhotoImage(file=str(preview_path))
            except Exception as exc:
                self.preview_label.configure(image="", text=f"Preview load failed\n{exc}")
                return
            self.current_photo = photo
            self.preview_label.configure(image=self.current_photo, text="")
            return

        try:
            image = Image.open(preview_path).convert("RGBA")
            image = ImageOps.contain(image, (420, 320))
            photo = ImageTk.PhotoImage(image)
        except Exception as exc:
            self.preview_label.configure(image="", text=f"Preview load failed\n{exc}")
            return

        self.current_photo = photo
        self.preview_label.configure(image=self.current_photo, text="")

    def create_current_asset(self, advance: bool = False) -> None:
        candidate = self.current_candidate()
        if candidate is None:
            messagebox.showwarning("Create Asset", "Select an asset first.")
            return

        try:
            spec = self.build_current_spec(strict=True)
            bbox = compute_bbox_from_usd(str(candidate.model_path))
            destination_dir = normalize_candidate(candidate, spec, bbox)
        except Exception as exc:
            messagebox.showerror("Create Asset", str(exc))
            self.set_status(f"Create failed: {exc}")
            return

        self.completed_sources.add(candidate.source_dir)
        self.refresh_candidate_labels()
        self.set_status(f"Wrote benchmark asset to {destination_dir}")
        if advance:
            self.change_candidate(1)


def run_write_once(args: argparse.Namespace) -> int:
    if args.source_dir is None:
        raise SystemExit("--write-once requires --source-dir")

    candidate = build_source_candidate(args.source_dir.expanduser())
    if candidate is None:
        raise SystemExit(f"No supported model file found under: {args.source_dir}")

    bbox = compute_bbox_from_usd(str(candidate.model_path))
    spec = NormalizationSpec(
        asset_root=args.asset_root.expanduser(),
        prefix=args.prefix,
        category=args.category or candidate.category_hint,
        index=args.index or candidate.index_hint,
        english_name=args.english_name or candidate.english_name_hint,
        chinese_name=args.chinese_name or args.english_name or candidate.english_name_hint,
        mass=args.mass,
        overwrite=args.overwrite,
    )
    destination_dir = normalize_candidate(candidate, spec, bbox)
    print(destination_dir)
    return 0


def main() -> int:
    args = parse_args()
    if args.write_once:
        return run_write_once(args)

    root = tk.Tk()
    app = AssetNormalizerApp(
        root,
        asset_root=args.asset_root.expanduser(),
        initial_source_dir=args.source_dir.expanduser() if args.source_dir else DEFAULT_SOURCE_ROOT,
        default_prefix=args.prefix,
        default_mass=args.mass,
    )
    app.set_status(
        "Ready. Run inside the `issac` conda environment so USD bounding-box computation is available."
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
