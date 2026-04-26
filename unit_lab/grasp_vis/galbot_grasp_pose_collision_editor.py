#!/usr/bin/env python3
"""Offline Galbot grasp pose collision calibration editor.

This tool validates and edits GenieSim `grasp_pose.pkl` labels against the
opened Galbot gripper geometry. It is intentionally independent from the data
collection pipeline: the scene contains only the object, labeled grasp poses,
and the opened gripper aligned by TCP.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from yourdfpy import URDF


Gf = None
Sdf = None
Usd = None
UsdGeom = None
UsdLux = None


def ensure_pxr_imported() -> None:
    global Gf, Sdf, Usd, UsdGeom, UsdLux
    if Usd is not None:
        return
    from pxr import Gf as _Gf
    from pxr import Sdf as _Sdf
    from pxr import Usd as _Usd
    from pxr import UsdGeom as _UsdGeom
    from pxr import UsdLux as _UsdLux

    Gf = _Gf
    Sdf = _Sdf
    Usd = _Usd
    UsdGeom = _UsdGeom
    UsdLux = _UsdLux


DEFAULT_ASSET_ROOT = Path("/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets")
DEFAULT_OBJECT_REL_DIR = Path("objects/benchmark/beverage_bottle/benchmark_beverage_bottle_001")
DEFAULT_ROBOT_URDF = DEFAULT_ASSET_ROOT / "robot" / "galbot" / "galbot_one_golf.urdf"
DEFAULT_APPROACH_AXIS = "+x"

UI_TITLE = "Galbot Grasp Pose Collision Editor"
SCENE_ROOT = "/World/GalbotGraspPoseCollisionEditor"
OBJECT_PATH = f"{SCENE_ROOT}/Object"
POSE_ROOT = f"{SCENE_ROOT}/GraspPoseGlyphs"
GRIPPER_ROOT = f"{SCENE_ROOT}/SelectedOpenedGripper"
LIGHT_PATH = f"{SCENE_ROOT}/KeyLight"
EDITOR_CAMERA_PRIM_PATH = "/OmniverseKit_Persp"

STATUS_SAFE = "safe"
STATUS_NEAR = "near"
STATUS_COLLISION = "collision"
STATUS_UNKNOWN = "unknown"

STATUS_COLORS = {
    STATUS_SAFE: np.array([0.12, 0.78, 0.32], dtype=np.float32),
    STATUS_NEAR: np.array([1.0, 0.72, 0.12], dtype=np.float32),
    STATUS_COLLISION: np.array([0.95, 0.15, 0.12], dtype=np.float32),
    STATUS_UNKNOWN: np.array([0.55, 0.55, 0.55], dtype=np.float32),
}

AXIS_VECS = {
    "+x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "+y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "+z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}
AXIS_ORDER = tuple(AXIS_VECS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=UI_TITLE)
    parser.add_argument("--asset_root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--object_dir", type=Path, default=DEFAULT_OBJECT_REL_DIR)
    parser.add_argument("--object_id", type=str, default="")
    parser.add_argument("--object_collision_mode", choices=("auto", "visual", "convex_hull"), default="auto")
    parser.add_argument("--grasp_pkl", type=Path, default=None)
    parser.add_argument("--robot_urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument(
        "--mesh_root",
        type=Path,
        action="append",
        default=[],
        help="Additional root used to resolve URDF mesh filenames. Can be passed multiple times.",
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--approach_axis", choices=AXIS_ORDER, default=DEFAULT_APPROACH_AXIS)
    parser.add_argument("--opened_knuckle", type=float, default=0.0)
    parser.add_argument("--sample_points_per_part", type=int, default=300)
    parser.add_argument("--near_threshold", type=float, default=0.002)
    parser.add_argument("--danger_distance", type=float, default=None)
    parser.add_argument("--penetration_epsilon", type=float, default=0.0003)
    parser.add_argument("--auto_offset_range", type=float, default=0.05)
    parser.add_argument("--auto_offset_step", type=float, default=0.002)
    parser.add_argument("--manual_offset_step", type=float, default=0.002)
    parser.add_argument("--glyph_size", type=float, default=0.006)
    parser.add_argument("--selected_glyph_size", type=float, default=0.011)
    parser.add_argument("--output_name", type=str, default="grasp_pose.pkl")
    parser.add_argument("--update_interaction_json", action="store_true")
    parser.add_argument("--no_backup", action="store_true")
    parser.add_argument("--headless_check", action="store_true")
    parser.add_argument("--max_poses", type=int, default=0, help="Headless-only limit for quick checks. 0 means all poses.")
    parser.add_argument("--progress_every", type=int, default=25, help="Headless progress interval. 0 disables progress logs.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_known_args()[0]


ARGS = parse_args()


def resolve_under_root(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def object_id_from_dir(object_dir: Path) -> str:
    return object_dir.name.rstrip("/")


def resolve_paths(args: argparse.Namespace) -> dict[str, Path | str]:
    asset_root = args.asset_root.expanduser().resolve()
    object_dir = resolve_under_root(asset_root, args.object_dir).resolve()
    object_id = args.object_id or object_id_from_dir(object_dir)
    interaction_dir = asset_root / "interaction" / object_id
    grasp_pkl = args.grasp_pkl
    if grasp_pkl is None:
        grasp_pkl = interaction_dir / "grasp_pose" / "grasp_pose.pkl"
    else:
        grasp_pkl = resolve_under_root(asset_root, grasp_pkl).resolve()
    return {
        "asset_root": asset_root,
        "object_dir": object_dir,
        "object_id": object_id,
        "object_usd": object_dir / "Aligned.usd",
        "object_params": object_dir / "object_parameters.json",
        "interaction_dir": interaction_dir,
        "interaction_json": interaction_dir / "interaction.json",
        "grasp_pkl": grasp_pkl,
        "robot_urdf": resolve_under_root(asset_root, args.robot_urdf).resolve(),
    }


PATHS = resolve_paths(ARGS)


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def resolve_mesh_roots(args: argparse.Namespace, paths: dict[str, Path | str]) -> list[Path]:
    asset_root = Path(paths["asset_root"])
    urdf_path = Path(paths["robot_urdf"])
    explicit = [resolve_under_root(asset_root, root) for root in args.mesh_root]
    defaults = [
        urdf_path.parent,
        urdf_path.parent / "urdf",
        asset_root / "robot" / "galbot",
        asset_root / "robot" / "curobo_robot" / "assets" / "robot" / "galbot",
        Path("/home/agxi/Documents/assets/robots/galbot_one_golf_description/sim_ready/galbot_one_golf/urdf"),
        Path("/home/agxi/Documents/assets/robots/galbot_one_golf_description/sim_ready/galbot_one_golf"),
        Path("/home/agxi/Documents/assets/robots/galbot_one_golf_description"),
        Path("/home/agxi/Documents/assets/robots/isaac_sim_galbot_one_golf"),
        Path("/home/agxi/Documents/assets/robots/urdf_ws/install/galbot_one_golf_description/share/galbot_one_golf_description"),
        Path("/home/agxi/Documents/assets/robots/urdf_ws/src/galbot_one_golf_description/sim_ready/galbot_one_golf/urdf"),
        Path("/home/agxi/Documents/assets/robots/urdf_ws/src/galbot_one_golf_description"),
    ]
    return unique_paths(explicit + defaults)


def gf_matrix_to_np(matrix: Gf.Matrix4d) -> np.ndarray:
    ensure_pxr_imported()
    raw = np.array([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
    return raw.T


def np_to_gf_matrix4d(matrix: np.ndarray) -> Gf.Matrix4d:
    ensure_pxr_imported()
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
    ensure_pxr_imported()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(np_to_gf_matrix4d(matrix))


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def axis_direction(pose: np.ndarray, axis_name: str) -> np.ndarray:
    return pose[:3, :3] @ AXIS_VECS[axis_name]


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-10 or abs(float(angle)) <= 1e-12:
        return np.eye(4, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(axis / norm * float(angle)).as_matrix()
    return matrix


def prismatic_along_axis(axis: np.ndarray, distance: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm > 1e-10:
        matrix[:3, 3] = axis / norm * float(distance)
    return matrix


def triangulate_faces(face_counts: list[int], face_indices: list[int]) -> np.ndarray:
    faces: list[list[int]] = []
    cursor = 0
    for count in face_counts:
        indices = list(face_indices[cursor : cursor + count])
        cursor += count
        if count < 3:
            continue
        for i in range(1, count - 1):
            faces.append([indices[0], indices[i], indices[i + 1]])
    return np.asarray(faces, dtype=np.int64)


def load_usd_mesh_as_trimesh(usd_path: Path) -> trimesh.Trimesh:
    ensure_pxr_imported()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    vertex_offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        local_vertices = np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float64)
        world_matrix = gf_matrix_to_np(cache.GetLocalToWorldTransform(prim))
        world_vertices = transform_points(world_matrix, local_vertices)
        faces = triangulate_faces(list(counts), list(indices))
        if len(faces) == 0:
            continue
        vertices_all.append(world_vertices)
        faces_all.append(faces + vertex_offset)
        vertex_offset += len(world_vertices)
    if not vertices_all:
        raise RuntimeError(f"No mesh prims found in USD stage: {usd_path}")
    merged = trimesh.Trimesh(
        vertices=np.vstack(vertices_all),
        faces=np.vstack(faces_all),
        process=True,
        validate=True,
    )
    merged.remove_unreferenced_vertices()
    return merged


def object_model_type(object_params: Path) -> str:
    if not object_params.exists():
        return ""
    try:
        with object_params.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    return str(data.get("model_type", ""))


def object_collision_geometry(
    visual_mesh: trimesh.Trimesh,
    paths: dict[str, Path | str],
    mode: str,
) -> tuple[trimesh.Trimesh, str, np.ndarray | None]:
    resolved_mode = mode
    if resolved_mode == "auto":
        model_type = object_model_type(Path(paths["object_params"]))
        resolved_mode = "convex_hull" if model_type.lower() == "convexhull" else "visual"
    if resolved_mode == "convex_hull":
        hull = ConvexHull(np.asarray(visual_mesh.vertices, dtype=np.float64))
        hull_mesh = trimesh.Trimesh(
            vertices=np.asarray(visual_mesh.vertices, dtype=np.float64),
            faces=np.asarray(hull.simplices, dtype=np.int64),
            process=True,
            validate=True,
        )
        hull_mesh.remove_unreferenced_vertices()
        return hull_mesh, resolved_mode, np.asarray(hull.equations, dtype=np.float64)
    return visual_mesh, resolved_mode, None


def load_pickle_grasps(grasp_pkl: Path) -> tuple[np.ndarray, np.ndarray]:
    with grasp_pkl.open("rb") as f:
        data = pickle.load(f)
    poses = np.asarray(data.get("grasp_pose"), dtype=np.float64)
    widths = np.asarray(data.get("width"), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Invalid grasp_pose shape in {grasp_pkl}: {poses.shape}")
    if widths.ndim == 0:
        widths = np.full((len(poses),), float(widths), dtype=np.float64)
    widths = widths.reshape(-1)
    if len(widths) != len(poses):
        raise ValueError(f"width count {len(widths)} does not match pose count {len(poses)}")
    return poses.copy(), widths.copy()


def save_pickle_grasps(
    output_path: Path,
    poses: np.ndarray,
    widths: np.ndarray,
    source_path: Path,
    backup: bool = True,
) -> Path | None:
    backup_path = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and backup:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_name(f"{output_path.name}.bak_{stamp}")
        shutil.copy2(output_path, backup_path)
    with output_path.open("wb") as f:
        pickle.dump({"grasp_pose": np.asarray(poses), "width": np.asarray(widths)}, f)
    if output_path != source_path and backup and source_path.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        source_backup = source_path.with_name(f"{source_path.name}.source_bak_{stamp}")
        shutil.copy2(source_path, source_backup)
    return backup_path


def update_interaction_json(interaction_json: Path, output_path: Path, interaction_dir: Path) -> None:
    if not interaction_json.exists():
        raise FileNotFoundError(interaction_json)
    with interaction_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rel_path = output_path.relative_to(interaction_dir).as_posix()
    interaction = data.setdefault("interaction", {})
    passive = interaction.setdefault("passive", {})
    grasp = passive.setdefault("grasp", {})
    grasp["default"] = [rel_path]
    with interaction_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@dataclass
class GripperPart:
    name: str
    link_name: str
    mesh: trimesh.Trimesh
    tcp_to_collision: np.ndarray
    sample_points: np.ndarray


@dataclass
class CollisionResult:
    status: str
    max_signed_distance: float
    clearance: float
    colliding_parts: tuple[str, ...]


def load_mesh_file(mesh_path: Path, scale: np.ndarray | None = None) -> trimesh.Trimesh:
    loaded = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        meshes = [m for m in loaded.dump() if isinstance(m, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"No mesh geometry loaded from {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    mesh = mesh.copy()
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float64)
        if scale.shape == (3,):
            mesh.apply_scale(scale)
        else:
            mesh.apply_scale(float(scale.reshape(-1)[0]))
    mesh.process(validate=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def sample_mesh_points(mesh: trimesh.Trimesh, max_points: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices.copy()
    vertex_budget = max(8, min(len(vertices), max_points // 3))
    surface_budget = max(0, max_points - vertex_budget)
    vertex_indices = np.linspace(0, len(vertices) - 1, vertex_budget, dtype=np.int64)
    chosen_vertices = vertices[vertex_indices]
    if surface_budget <= 0:
        return chosen_vertices
    sampled = deterministic_sample_surface(mesh, surface_budget)
    return np.vstack([chosen_vertices, np.asarray(sampled, dtype=np.float64)])


def deterministic_sample_surface(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 3), dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return vertices[np.linspace(0, len(vertices) - 1, count, dtype=np.int64)]
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if len(areas) != len(faces) or float(np.sum(areas)) <= 1e-12:
        return vertices[np.linspace(0, len(vertices) - 1, count, dtype=np.int64)]
    cumulative = np.cumsum(areas)
    targets = (np.arange(count, dtype=np.float64) + 0.5) / float(count) * cumulative[-1]
    face_indices = np.searchsorted(cumulative, targets, side="left")
    face_indices = np.clip(face_indices, 0, len(faces) - 1)
    triangles = vertices[faces[face_indices]]
    sequence = np.arange(count, dtype=np.float64) + 1.0
    u = np.mod(sequence * 0.7548776662466927, 1.0)
    v = np.mod(sequence * 0.5698402909980532, 1.0)
    sqrt_u = np.sqrt(u)
    return (
        (1.0 - sqrt_u)[:, None] * triangles[:, 0]
        + (sqrt_u * (1.0 - v))[:, None] * triangles[:, 1]
        + (sqrt_u * v)[:, None] * triangles[:, 2]
    )


def mesh_reference_variants(mesh_filename: str | Path) -> list[Path]:
    raw = str(mesh_filename)
    variants: list[str] = []
    if raw.startswith("file://"):
        variants.append(raw.removeprefix("file://"))
    elif raw.startswith("package://"):
        package_path = raw.removeprefix("package://")
        variants.append(package_path)
        if "/" in package_path:
            variants.append(package_path.split("/", 1)[1])
    else:
        variants.append(raw)
    return [Path(value).expanduser() for value in variants if value]


def resolve_mesh_file(mesh_filename: str | Path, mesh_roots: list[Path]) -> Path:
    checked: list[Path] = []
    for ref in mesh_reference_variants(mesh_filename):
        if ref.is_absolute():
            checked.append(ref.resolve())
        else:
            checked.extend((root / ref).resolve() for root in mesh_roots)
    for candidate in checked:
        if candidate.exists():
            return candidate
    checked_text = "\n  ".join(str(path) for path in checked[:30])
    more = "" if len(checked) <= 30 else f"\n  ... {len(checked) - 30} more"
    raise FileNotFoundError(f"Could not resolve URDF mesh '{mesh_filename}'. Checked:\n  {checked_text}{more}")


def joint_motion_matrix(joint: Any, q: float) -> np.ndarray:
    if joint.type in ("fixed", None):
        return np.eye(4, dtype=np.float64)
    if joint.type in ("revolute", "continuous"):
        return rotation_about_axis(np.asarray(joint.axis, dtype=np.float64), q)
    if joint.type == "prismatic":
        return prismatic_along_axis(np.asarray(joint.axis, dtype=np.float64), q)
    return np.eye(4, dtype=np.float64)


def joint_value(joint: Any, opened_joint_values: dict[str, float]) -> float:
    if joint.mimic is not None:
        parent_value = opened_joint_values.get(joint.mimic.joint, 0.0)
        return float(parent_value) * float(joint.mimic.multiplier or 1.0) + float(joint.mimic.offset or 0.0)
    return float(opened_joint_values.get(joint.name, 0.0))


def compute_link_transforms(
    robot: URDF,
    root_link: str,
    opened_joint_values: dict[str, float],
) -> dict[str, np.ndarray]:
    children_by_parent: dict[str, list[Any]] = {}
    for joint in robot.robot.joints:
        children_by_parent.setdefault(joint.parent, []).append(joint)
    transforms = {root_link: np.eye(4, dtype=np.float64)}
    stack = [root_link]
    while stack:
        parent = stack.pop()
        for joint in children_by_parent.get(parent, []):
            origin = np.asarray(joint.origin if joint.origin is not None else np.eye(4), dtype=np.float64)
            q = joint_value(joint, opened_joint_values)
            transforms[joint.child] = transforms[parent] @ origin @ joint_motion_matrix(joint, q)
            stack.append(joint.child)
    return transforms


def load_gripper_parts(
    urdf_path: Path,
    mesh_roots: list[Path],
    arm: str,
    opened_knuckle: float,
    sample_points_per_part: int,
) -> tuple[list[GripperPart], str]:
    urdf_path = urdf_path.resolve()
    robot = URDF.load(str(urdf_path), load_meshes=False, load_collision_meshes=False)
    root_link = f"{arm}_gripper_flange_link"
    tcp_link = f"{arm}_gripper_tcp_link"
    opened_joint_values = {f"{arm}_gripper_r_knuckle_joint": opened_knuckle}
    transforms = compute_link_transforms(robot, root_link, opened_joint_values)
    if tcp_link not in transforms:
        raise RuntimeError(f"Could not find transform for {tcp_link} from {root_link}")
    root_to_tcp = transforms[tcp_link]
    tcp_to_root = np.linalg.inv(root_to_tcp)
    parts: list[GripperPart] = []
    for link_name, root_to_link in transforms.items():
        if not link_name.startswith(f"{arm}_gripper_"):
            continue
        link = robot.link_map.get(link_name)
        if link is None:
            continue
        for collision_id, collision in enumerate(link.collisions):
            geometry = collision.geometry
            if geometry is None or geometry.mesh is None:
                continue
            mesh_path = resolve_mesh_file(geometry.mesh.filename, mesh_roots)
            scale = None if geometry.mesh.scale is None else np.asarray(geometry.mesh.scale, dtype=np.float64)
            mesh = load_mesh_file(mesh_path, scale=scale)
            origin = np.asarray(collision.origin if collision.origin is not None else np.eye(4), dtype=np.float64)
            tcp_to_collision = tcp_to_root @ root_to_link @ origin
            sample_points = sample_mesh_points(mesh, int(sample_points_per_part))
            parts.append(
                GripperPart(
                    name=f"{link_name}:{collision_id}",
                    link_name=link_name,
                    mesh=mesh,
                    tcp_to_collision=tcp_to_collision,
                    sample_points=sample_points,
                )
            )
    if not parts:
        raise RuntimeError(f"No collision meshes were loaded for {arm} gripper from {urdf_path}")
    return parts, tcp_link


class GraspCollisionEvaluator:
    def __init__(
        self,
        object_mesh: trimesh.Trimesh,
        gripper_parts: list[GripperPart],
        near_threshold: float,
        penetration_epsilon: float,
        collision_mode: str,
        convex_halfspaces: np.ndarray | None = None,
    ) -> None:
        self.object_mesh = object_mesh
        self.gripper_parts = gripper_parts
        self.near_threshold = float(near_threshold)
        self.penetration_epsilon = float(penetration_epsilon)
        self.collision_mode = collision_mode
        self.convex_halfspaces = convex_halfspaces
        self.object_watertight = bool(object_mesh.is_watertight)
        self.object_bounds = np.asarray(object_mesh.bounds, dtype=np.float64)
        self.proximity = None if convex_halfspaces is not None else trimesh.proximity.ProximityQuery(object_mesh)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        if self.convex_halfspaces is not None:
            normals = self.convex_halfspaces[:, :3]
            offsets = self.convex_halfspaces[:, 3]
            # scipy ConvexHull uses n.dot(x) + d <= 0 for inside. Match trimesh:
            # positive means inside/intersection, negative means outside/clear.
            return -np.max(points @ normals.T + offsets, axis=1)
        if self.proximity is None:
            return np.full((len(points),), -np.inf, dtype=np.float64)
        return self.proximity.signed_distance(points)

    def evaluate_pose(self, pose: np.ndarray) -> CollisionResult:
        points_by_part: list[np.ndarray] = []
        slices: list[tuple[GripperPart, slice]] = []
        cursor = 0
        for part in self.gripper_parts:
            transform = pose @ part.tcp_to_collision
            points_world = transform_points(transform, part.sample_points)
            points_by_part.append(points_world)
            next_cursor = cursor + len(points_world)
            slices.append((part, slice(cursor, next_cursor)))
            cursor = next_cursor
        if not points_by_part:
            return CollisionResult(STATUS_UNKNOWN, float("nan"), float("nan"), tuple())

        all_points = np.vstack(points_by_part)
        expanded_bounds = self.object_bounds.copy()
        expanded_bounds[0] -= self.near_threshold
        expanded_bounds[1] += self.near_threshold
        if np.any(all_points.max(axis=0) < expanded_bounds[0]) or np.any(all_points.min(axis=0) > expanded_bounds[1]):
            gap = np.maximum(expanded_bounds[0] - all_points.max(axis=0), all_points.min(axis=0) - expanded_bounds[1])
            clearance = float(max(0.0, np.max(gap)))
            return CollisionResult(STATUS_SAFE, -clearance, clearance, tuple())

        colliding_parts: list[str] = []
        try:
            signed = self.signed_distance(all_points)
        except Exception:
            signed = np.full((len(all_points),), -np.inf, dtype=np.float64)
        if signed.size == 0 or not np.any(np.isfinite(signed)):
            return CollisionResult(STATUS_UNKNOWN, float("nan"), float("nan"), tuple())
        max_signed_distance = float(np.max(signed))
        for part, part_slice in slices:
            part_signed = signed[part_slice]
            if part_signed.size == 0:
                continue
            part_max = float(np.max(part_signed))
            if part_max > self.penetration_epsilon:
                colliding_parts.append(part.name)
        clearance = max(0.0, -max_signed_distance)
        if max_signed_distance > self.penetration_epsilon:
            status = STATUS_COLLISION
        elif clearance <= self.near_threshold:
            status = STATUS_NEAR
        else:
            status = STATUS_SAFE
        return CollisionResult(status, max_signed_distance, clearance, tuple(colliding_parts))

    def evaluate_all(self, poses: np.ndarray, progress_every: int = 0) -> list[CollisionResult]:
        results: list[CollisionResult] = []
        total = len(poses)
        for index, pose in enumerate(poses):
            if progress_every > 0 and index % progress_every == 0:
                print(f"checked {index}/{total}", flush=True)
            results.append(self.evaluate_pose(pose))
        if progress_every > 0:
            print(f"checked {total}/{total}", flush=True)
        return results


def summarize_results(results: list[CollisionResult]) -> str:
    counts = {STATUS_SAFE: 0, STATUS_NEAR: 0, STATUS_COLLISION: 0, STATUS_UNKNOWN: 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return (
        f"total={len(results)} safe={counts[STATUS_SAFE]} near={counts[STATUS_NEAR]} "
        f"collision={counts[STATUS_COLLISION]} unknown={counts[STATUS_UNKNOWN]}"
    )


def build_data_and_evaluator(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, trimesh.Trimesh, list[GripperPart], GraspCollisionEvaluator, str]:
    paths = resolve_paths(args)
    visual_object_mesh = load_usd_mesh_as_trimesh(paths["object_usd"])  # type: ignore[arg-type]
    object_mesh, collision_mode, convex_halfspaces = object_collision_geometry(
        visual_object_mesh, paths, args.object_collision_mode
    )
    mesh_roots = resolve_mesh_roots(args, paths)
    gripper_parts, tcp_link = load_gripper_parts(
        paths["robot_urdf"],  # type: ignore[arg-type]
        mesh_roots,
        args.arm,
        args.opened_knuckle,
        args.sample_points_per_part,
    )
    poses, widths = load_pickle_grasps(paths["grasp_pkl"])  # type: ignore[arg-type]
    evaluator = GraspCollisionEvaluator(
        object_mesh,
        gripper_parts,
        args.near_threshold,
        args.penetration_epsilon,
        collision_mode,
        convex_halfspaces,
    )
    return poses, widths, object_mesh, gripper_parts, evaluator, tcp_link


def run_headless_check(args: argparse.Namespace) -> int:
    poses, widths, object_mesh, gripper_parts, evaluator, tcp_link = build_data_and_evaluator(args)
    check_count = len(poses) if args.max_poses <= 0 else min(len(poses), int(args.max_poses))
    check_poses = poses[:check_count]
    check_widths = widths[:check_count]
    results = evaluator.evaluate_all(check_poses, progress_every=max(0, int(args.progress_every)))
    print(UI_TITLE)
    print(f"object_usd      : {PATHS['object_usd']}")
    print(f"grasp_pkl       : {PATHS['grasp_pkl']}")
    print(f"robot_urdf      : {PATHS['robot_urdf']}")
    print(f"arm/tcp         : {args.arm}/{tcp_link}")
    print(f"object collision: {evaluator.collision_mode} model_type={object_model_type(Path(PATHS['object_params']))}")
    print(f"object watertight: {object_mesh.is_watertight}")
    print(f"object mesh     : vertices={len(object_mesh.vertices)} faces={len(object_mesh.faces)}")
    print(f"gripper parts   : {len(gripper_parts)}")
    print(f"checked poses   : {check_count}/{len(poses)}")
    print(summarize_results(results))
    worst = sorted(
        enumerate(results),
        key=lambda item: (-np.inf if not np.isfinite(item[1].max_signed_distance) else item[1].max_signed_distance),
        reverse=True,
    )[:20]
    print("worst poses:")
    for idx, result in worst:
        print(
            f"  {idx:04d} status={result.status:9s} "
            f"max_sd={result.max_signed_distance:+.6f} clearance={result.clearance:.6f} "
            f"width={check_widths[idx]:.5f} parts={','.join(result.colliding_parts[:3])}"
        )
    return 0


if not ARGS.headless_check:
    try:
        import isaacsim  # noqa: F401
    except ImportError:
        pass

    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"width": ARGS.width, "height": ARGS.height, "headless": False})
    ensure_pxr_imported()

    import omni.ui as ui
    import omni.usd
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.viewports import set_camera_view

    try:
        from isaacsim.gui.components.element_wrappers import ScrollingWindow
    except Exception:
        ScrollingWindow = None


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
    ensure_pxr_imported()
    color = np.asarray(rgb, dtype=np.float32).reshape(3)
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    gprim.CreateDisplayOpacityAttr().Set([float(max(0.0, min(opacity, 1.0)))])


def define_mesh_prim(stage, prim_path: str, mesh: trimesh.Trimesh, color: np.ndarray, opacity: float) -> None:
    ensure_pxr_imported()
    mesh_prim = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in np.asarray(mesh.vertices)]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    mesh_prim.CreatePointsAttr(points)
    mesh_prim.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh_prim.CreateFaceVertexIndicesAttr(faces.reshape(-1).astype(int).tolist())
    mesh_prim.CreateSubdivisionSchemeAttr("none")
    set_display_color(mesh_prim.GetPrim(), color, opacity)


class GalbotGraspPoseCollisionEditor:
    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()
        self.selection = omni.usd.get_context().get_selection()
        self.poses, self.widths, self.object_mesh, self.gripper_parts, self.evaluator, self.tcp_link = (
            build_data_and_evaluator(ARGS)
        )
        self.original_poses = self.poses.copy()
        self.keep_mask = np.ones((len(self.poses),), dtype=bool)
        self.offsets = np.zeros((len(self.poses),), dtype=np.float64)
        self.results = self.evaluator.evaluate_all(self.poses)
        self.selected_index = 0
        self.approach_axis = ARGS.approach_axis
        self.status_message = "Loaded."
        self.running = True
        self._window = None
        self._scene_dirty = True
        self._ui_dirty = True
        self._gripper_mesh_cache_created = False
        self.index_model = ui.SimpleStringModel(str(self.selected_index))
        danger_distance = ARGS.near_threshold if ARGS.danger_distance is None else ARGS.danger_distance
        self.danger_distance_model = ui.SimpleFloatModel(float(danger_distance))
        self.offset_step_model = ui.SimpleFloatModel(float(ARGS.manual_offset_step))
        self.auto_range_model = ui.SimpleFloatModel(float(ARGS.auto_offset_range))
        self.auto_step_model = ui.SimpleFloatModel(float(ARGS.auto_offset_step))
        self._warm_up()
        self._ensure_scene()
        self._build_ui_window()
        self._apply_scene(force=True)

    def _warm_up(self) -> None:
        for _ in range(20):
            simulation_app.update()

    def _ensure_scene(self) -> None:
        ensure_prim(self.stage, "/World", "Xform")
        ensure_prim(self.stage, SCENE_ROOT, "Xform")
        ensure_prim(self.stage, POSE_ROOT, "Xform")
        ensure_prim(self.stage, GRIPPER_ROOT, "Xform")
        if not self.stage.GetPrimAtPath(LIGHT_PATH):
            light = UsdLux.SphereLight.Define(self.stage, Sdf.Path(LIGHT_PATH))
            light.CreateIntensityAttr(65000.0)
            light.CreateRadiusAttr(0.35)
            set_local_matrix(
                light.GetPrim(),
                np.array(
                    [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 1.2],
                        [0.0, 0.0, 1.0, 1.8],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
            )
        remove_prim_if_exists(self.stage, OBJECT_PATH)
        create_prim(OBJECT_PATH, prim_type="Xform", usd_path=str(PATHS["object_usd"]))
        set_camera_view(
            eye=np.array([0.24, -0.42, 0.25], dtype=np.float64),
            target=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            camera_prim_path=EDITOR_CAMERA_PRIM_PATH,
        )

    def _build_ui_window(self) -> None:
        window_kwargs = {
            "title": UI_TITLE,
            "width": 470,
            "height": 780,
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

    def _build_cycle_axis_row(self) -> None:
        with ui.HStack(height=28, spacing=6):
            ui.Label("Approach", width=80)
            ui.Button("<", width=28, clicked_fn=lambda: self._cycle_axis(-1))
            ui.Label(self.approach_axis, width=80)
            ui.Button(">", width=28, clicked_fn=lambda: self._cycle_axis(1))

    def _build_float_row(self, label: str, model, width: int = 100) -> None:
        with ui.HStack(height=28, spacing=6):
            ui.Label(label, width=110)
            ui.FloatField(model=model, width=width)

    def _build_ui_contents(self) -> None:
        result = self.results[self.selected_index]
        with ui.VStack(spacing=8, height=0):
            ui.Label(UI_TITLE, height=24)
            ui.Label(f"Object: {PATHS['object_id']} | Arm: {ARGS.arm} | TCP: {self.tcp_link}", word_wrap=True, height=40)
            ui.Label(summarize_results(self.results), word_wrap=True, height=26)
            ui.Label(self.status_message, word_wrap=True, height=42)
            ui.Separator(height=6)

            with ui.HStack(height=30, spacing=8):
                ui.Button("Prev", width=70, clicked_fn=lambda: self._select_delta(-1))
                ui.Button("Next", width=70, clicked_fn=lambda: self._select_delta(1))
                ui.Button("Next Collision", width=130, clicked_fn=self._select_next_collision)
                ui.Button("Next Danger", width=120, clicked_fn=self._select_next_danger)
            with ui.HStack(height=28, spacing=6):
                ui.Button("Recheck All", width=110, clicked_fn=self._recheck_all)
                ui.Label("Danger Dist", width=90)
                ui.FloatField(model=self.danger_distance_model, width=100)
            ui.Label(
                "Next Danger jumps to non-collision poses whose clearance is <= Danger Dist.",
                word_wrap=True,
                height=34,
            )
            with ui.HStack(height=28, spacing=6):
                ui.Label("Index", width=80)
                ui.StringField(model=self.index_model, width=90)
                ui.Button("Go", width=50, clicked_fn=self._apply_index_model)
                ui.Label(f"/ {len(self.poses) - 1}", width=80)
            ui.Label(
                f"Selected: {self.selected_index} | status={result.status} | width={self.widths[self.selected_index]:.5f}",
                word_wrap=True,
                height=24,
            )
            ui.Label(
                f"max_signed={result.max_signed_distance:+.6f} m | clearance={result.clearance:.6f} m | offset={self.offsets[self.selected_index]:+.6f} m",
                word_wrap=True,
                height=36,
            )
            ui.Label(
                "parts: " + (", ".join(result.colliding_parts[:6]) if result.colliding_parts else "none"),
                word_wrap=True,
                height=48,
            )
            ui.Separator(height=6)

            self._build_cycle_axis_row()
            self._build_float_row("Manual Step", self.offset_step_model)
            ui.Label("Manual Step: one click translation distance along selected approach axis.", word_wrap=True, height=32)
            with ui.HStack(height=30, spacing=8):
                ui.Button("- Step", width=90, clicked_fn=lambda: self._apply_manual_offset(-self._manual_step()))
                ui.Button("+ Step", width=90, clicked_fn=lambda: self._apply_manual_offset(self._manual_step()))
                ui.Button("Reset Offset", width=105, clicked_fn=self._reset_selected_offset)
            with ui.HStack(height=30, spacing=8):
                ui.Button("- Step Colliding", width=130, clicked_fn=self._apply_negative_step_to_colliding)
                ui.Button("- Step All", width=110, clicked_fn=self._apply_negative_step_to_all)
            self._build_float_row("Auto Range", self.auto_range_model)
            ui.Label("Auto Range: max absolute offset tried by Auto Selected/Auto Colliding.", word_wrap=True, height=32)
            self._build_float_row("Auto Step", self.auto_step_model)
            ui.Label("Auto Step: positive search interval inside Auto Range; do not enter a negative value.", word_wrap=True, height=32)
            with ui.HStack(height=30, spacing=8):
                ui.Button("Auto Selected", width=130, clicked_fn=self._auto_offset_selected)
                ui.Button("Auto Colliding", width=130, clicked_fn=self._auto_offset_colliding)
            ui.Separator(height=6)

            with ui.HStack(height=30, spacing=8):
                ui.Button("Reject Selected", width=140, clicked_fn=self._reject_selected)
                ui.Button("Keep Selected", width=125, clicked_fn=self._keep_selected)
                ui.Button("Restore Selected", width=140, clicked_fn=self._restore_selected_pose)
            with ui.HStack(height=30, spacing=8):
                ui.Button("Save Selected Modified", width=200, clicked_fn=self._save_selected_modified)
                ui.Button("Save All Modified Labels", width=200, clicked_fn=self._save_all_modified_labels)
            kept = int(np.sum(self.keep_mask))
            selected_state = "kept" if self.keep_mask[self.selected_index] else "rejected"
            ui.Label(
                f"Selected is {selected_state}. Kept poses: {kept}/{len(self.keep_mask)} | Current output: {ARGS.output_name}",
                word_wrap=True,
                height=42,
            )
            ui.Label(f"Source: {PATHS['grasp_pkl']}", word_wrap=True, height=54)

    def _manual_step(self) -> float:
        return max(0.0, float(self.offset_step_model.get_value_as_float()))

    def _danger_distance(self) -> float:
        return max(0.0, float(self.danger_distance_model.get_value_as_float()))

    def _auto_range(self) -> float:
        return max(0.0, float(self.auto_range_model.get_value_as_float()))

    def _auto_step(self) -> float:
        return max(1e-6, float(self.auto_step_model.get_value_as_float()))

    def _set_status(self, message: str) -> None:
        self.status_message = message
        self._ui_dirty = True

    def _mark_scene_dirty(self) -> None:
        self._scene_dirty = True
        self._ui_dirty = True

    def _apply_index_model(self) -> None:
        try:
            index = int(self.index_model.get_value_as_string())
        except Exception:
            self._set_status("Invalid index.")
            return
        self._select_index(index)

    def _select_index(self, index: int) -> None:
        self.selected_index = int(np.clip(index, 0, len(self.poses) - 1))
        self.index_model.set_value(str(self.selected_index))
        self._mark_scene_dirty()

    def _select_delta(self, delta: int) -> None:
        self._select_index((self.selected_index + delta) % len(self.poses))

    def _select_next_collision(self) -> None:
        for step in range(1, len(self.results) + 1):
            index = (self.selected_index + step) % len(self.results)
            if self.results[index].status == STATUS_COLLISION:
                self._select_index(index)
                return
        self._set_status("No collision pose found.")

    def _select_next_danger(self) -> None:
        threshold = self._danger_distance()
        for step in range(1, len(self.results) + 1):
            index = (self.selected_index + step) % len(self.results)
            result = self.results[index]
            if result.status == STATUS_COLLISION:
                continue
            if np.isfinite(result.clearance) and result.clearance <= threshold:
                self._select_index(index)
                return
        self._set_status(f"No non-collision pose found within danger distance {threshold:.4f} m.")

    def _cycle_axis(self, delta: int) -> None:
        idx = AXIS_ORDER.index(self.approach_axis)
        self.approach_axis = AXIS_ORDER[(idx + delta) % len(AXIS_ORDER)]
        self._set_status(f"Approach axis set to {self.approach_axis}.")

    def _recheck_selected(self) -> None:
        self.results[self.selected_index] = self.evaluator.evaluate_pose(self.poses[self.selected_index])
        self._mark_scene_dirty()

    def _recheck_all(self) -> None:
        self.results = self.evaluator.evaluate_all(self.poses)
        self._set_status("Rechecked all poses.")
        self._mark_scene_dirty()

    def _apply_manual_offset(self, delta: float) -> None:
        pose = self.poses[self.selected_index]
        pose[:3, 3] += axis_direction(pose, self.approach_axis) * float(delta)
        self.offsets[self.selected_index] += float(delta)
        self.results[self.selected_index] = self.evaluator.evaluate_pose(pose)
        self._set_status(f"Applied {delta:+.4f} m along {self.approach_axis}.")
        self._mark_scene_dirty()

    def _apply_offset_to_indices(self, indices: Iterable[int], delta: float, target_name: str) -> None:
        count = 0
        for index in indices:
            pose = self.poses[index]
            pose[:3, 3] += axis_direction(pose, self.approach_axis) * float(delta)
            self.offsets[index] += float(delta)
            self.results[index] = self.evaluator.evaluate_pose(pose)
            count += 1
        self._set_status(f"Applied {delta:+.4f} m along {self.approach_axis} to {target_name} ({count} poses).")
        self._mark_scene_dirty()

    def _apply_negative_step_to_colliding(self) -> None:
        indices = [i for i, result in enumerate(self.results) if result.status == STATUS_COLLISION]
        delta = -self._manual_step()
        self._apply_offset_to_indices(indices, delta, "colliding poses")

    def _apply_negative_step_to_all(self) -> None:
        delta = -self._manual_step()
        self._apply_offset_to_indices(range(len(self.poses)), delta, "all poses")

    def _reset_selected_offset(self) -> None:
        index = self.selected_index
        self.poses[index] = self.original_poses[index].copy()
        self.offsets[index] = 0.0
        self.results[index] = self.evaluator.evaluate_pose(self.poses[index])
        self._set_status("Reset selected pose to original.")
        self._mark_scene_dirty()

    def _restore_selected_pose(self) -> None:
        self._reset_selected_offset()

    def _reject_selected(self) -> None:
        self.keep_mask[self.selected_index] = False
        self._set_status(f"Rejected selected pose {self.selected_index}. It will be excluded from saved labels.")
        self._mark_scene_dirty()

    def _keep_selected(self) -> None:
        self.keep_mask[self.selected_index] = True
        self._set_status(f"Kept selected pose {self.selected_index}. It will be included in saved labels.")
        self._mark_scene_dirty()

    def _candidate_offsets(self) -> list[float]:
        span = self._auto_range()
        step = self._auto_step()
        positives = list(np.arange(0.0, span + step * 0.5, step, dtype=np.float64))
        negatives = [-v for v in positives[1:]]
        return [float(v) for v in positives + negatives]

    def _auto_offset_index(self, index: int) -> bool:
        original_pose = self.poses[index].copy()
        original_offset = float(self.offsets[index])
        direction = axis_direction(original_pose, self.approach_axis)
        for candidate in self._candidate_offsets():
            trial_pose = original_pose.copy()
            trial_pose[:3, 3] = original_pose[:3, 3] + direction * candidate
            result = self.evaluator.evaluate_pose(trial_pose)
            if result.status in (STATUS_SAFE, STATUS_NEAR):
                self.poses[index] = trial_pose
                self.offsets[index] = original_offset + candidate
                self.results[index] = result
                return True
        self.results[index] = self.evaluator.evaluate_pose(self.poses[index])
        return False

    def _auto_offset_selected(self) -> None:
        ok = self._auto_offset_index(self.selected_index)
        self._set_status("Auto offset selected succeeded." if ok else "Auto offset selected failed.")
        self._mark_scene_dirty()

    def _auto_offset_colliding(self) -> None:
        indices = [i for i, result in enumerate(self.results) if result.status == STATUS_COLLISION]
        fixed = 0
        for index in indices:
            if self._auto_offset_index(index):
                fixed += 1
        self._set_status(f"Auto offset fixed {fixed}/{len(indices)} colliding poses.")
        self._mark_scene_dirty()

    def _save_labels_to_path(self, output_path: Path, *, backup: bool, update_interaction: bool) -> None:
        source_path = PATHS["grasp_pkl"]
        kept_poses = self.poses[self.keep_mask]
        kept_widths = self.widths[self.keep_mask]
        try:
            backup_path = save_pickle_grasps(
                output_path,  # type: ignore[arg-type]
                kept_poses,
                kept_widths,
                source_path,  # type: ignore[arg-type]
                backup=backup,
            )
            if update_interaction:
                update_interaction_json(
                    PATHS["interaction_json"],  # type: ignore[arg-type]
                    output_path,  # type: ignore[arg-type]
                    PATHS["interaction_dir"],  # type: ignore[arg-type]
                )
            backup_text = f" backup={backup_path}" if backup_path is not None else ""
            self._set_status(f"Saved {len(kept_poses)} poses to {output_path}.{backup_text}")
        except Exception as exc:
            self._set_status(f"Save failed: {exc}")

    def _save_selected_modified(self) -> None:
        source_path = PATHS["grasp_pkl"]
        index = self.selected_index
        try:
            poses, widths = load_pickle_grasps(source_path)  # type: ignore[arg-type]
            if len(poses) != len(self.poses):
                raise ValueError(f"Current pose count {len(self.poses)} does not match on-disk count {len(poses)}")
            poses[index] = self.poses[index]
            widths[index] = self.widths[index]
            backup_path = save_pickle_grasps(
                source_path,  # type: ignore[arg-type]
                poses,
                widths,
                source_path,  # type: ignore[arg-type]
                backup=not ARGS.no_backup,
            )
            self.original_poses[index] = self.poses[index].copy()
            self.offsets[index] = 0.0
            backup_text = f" backup={backup_path}" if backup_path is not None else ""
            self._set_status(f"Saved selected pose {index} to {source_path}.{backup_text}")
            self._mark_scene_dirty()
        except Exception as exc:
            self._set_status(f"Save selected failed: {exc}")

    def _save_all_modified_labels(self) -> None:
        source_path = PATHS["grasp_pkl"]
        output_path = source_path.with_name(ARGS.output_name)  # type: ignore[union-attr]
        self._save_labels_to_path(
            output_path,  # type: ignore[arg-type]
            backup=not ARGS.no_backup,
            update_interaction=ARGS.update_interaction_json,
        )

    def _apply_scene(self, force: bool = False) -> None:
        if not force and not self._scene_dirty:
            return
        self._draw_pose_glyphs()
        self._draw_selected_gripper()
        self._scene_dirty = False

    def _draw_pose_glyphs(self) -> None:
        remove_prim_if_exists(self.stage, POSE_ROOT)
        ensure_prim(self.stage, POSE_ROOT, "Xform")
        for index, pose in enumerate(self.poses):
            status = self.results[index].status if index < len(self.results) else STATUS_UNKNOWN
            color = STATUS_COLORS.get(status, STATUS_COLORS[STATUS_UNKNOWN]).copy()
            if not self.keep_mask[index]:
                color = np.array([0.22, 0.22, 0.22], dtype=np.float32)
            size = ARGS.selected_glyph_size if index == self.selected_index else ARGS.glyph_size
            prim_path = f"{POSE_ROOT}/pose_{index:04d}"
            prim = create_prim(prim_path, prim_type="Sphere")
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = np.eye(3) * float(size)
            matrix[:3, 3] = pose[:3, 3]
            set_local_matrix(prim, matrix)
            set_display_color(prim, color, 1.0)

    def _draw_selected_gripper(self) -> None:
        remove_prim_if_exists(self.stage, GRIPPER_ROOT)
        ensure_prim(self.stage, GRIPPER_ROOT, "Xform")
        pose = self.poses[self.selected_index]
        for part_id, part in enumerate(self.gripper_parts):
            part_root = f"{GRIPPER_ROOT}/part_{part_id:02d}"
            part_prim = ensure_prim(self.stage, part_root, "Xform")
            set_local_matrix(part_prim, pose @ part.tcp_to_collision)
            color = np.array([0.20, 0.48, 0.96], dtype=np.float32)
            if part.name in self.results[self.selected_index].colliding_parts:
                color = STATUS_COLORS[STATUS_COLLISION]
            define_mesh_prim(self.stage, f"{part_root}/mesh", part.mesh, color, 0.52)

    def run(self) -> None:
        print("=" * 88)
        print(UI_TITLE)
        print("=" * 88)
        print(f"Object USD      : {PATHS['object_usd']}")
        print(f"Grasp PKL       : {PATHS['grasp_pkl']}")
        print(f"Robot URDF      : {PATHS['robot_urdf']}")
        print(f"Arm/TCP         : {ARGS.arm}/{self.tcp_link}")
        print(f"Approach axis   : {self.approach_axis}")
        print(f"Object collision: {self.evaluator.collision_mode} model_type={object_model_type(Path(PATHS['object_params']))}")
        print(f"Object watertight: {self.object_mesh.is_watertight}")
        print(f"Object mesh     : vertices={len(self.object_mesh.vertices)} faces={len(self.object_mesh.faces)}")
        print(summarize_results(self.results))
        print("=" * 88)
        while simulation_app.is_running() and self.running:
            if self._scene_dirty:
                self._apply_scene()
            if self._ui_dirty:
                self._rebuild_ui()
            simulation_app.update()
        self.close()

    def close(self) -> None:
        simulation_app.close()


def main() -> int:
    if ARGS.headless_check:
        return run_headless_check(ARGS)
    editor = GalbotGraspPoseCollisionEditor()
    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
