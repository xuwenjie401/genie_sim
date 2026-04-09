# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os

import numpy as np
import trimesh
from pxr import Usd, UsdGeom
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R

from client.layout.object import OmniObject
from client.layout.utils.func import get_bott_up_point, random_point
from client.layout.utils.sdf import compute_sdf_from_obj_surface
from common.base_utils.logger import logger
from common.base_utils.transform_utils import farthest_point_sampling

DEFAULT_LAYOUT_UNIT_SCALE = 1000.0
LAYOUT_UNIT_SCALE_CANDIDATES = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])


def load_mesh_from_usd(usd_path):
    """
    Extract all meshes from USD file and merge into a single trimesh object.

    Args:
        usd_path: USD file path

    Returns:
        trimesh.Trimesh object, or None if no mesh is found
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return None

    all_vertices = []
    all_faces = []
    vertex_offset = 0
    adjusted_scale = 1.0

    def traverse_prims(prim):
        nonlocal vertex_offset, all_vertices, all_faces, adjusted_scale

        if prim.IsA(UsdGeom.Mesh):
            usd_mesh = UsdGeom.Mesh(prim)
            if prim.HasAttribute("xformOp:transform:transform1"):
                transform = prim.GetAttribute("xformOp:transform:transform1").Get()
                adjusted_scale = transform[0, 0]
            points = usd_mesh.GetPointsAttr().Get()
            indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
            face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()

            if points is None or indices is None or face_counts is None:
                return

            # Add vertices
            for point in points:
                all_vertices.append([point[0], point[1], point[2]])

            # Add faces (fan-style triangulation, consistent with generate_obj.py logic)
            idx = 0
            for face_count in face_counts:
                for i in range(1, face_count - 1):
                    all_faces.append(
                        [
                            indices[idx] + vertex_offset,
                            indices[idx + i] + vertex_offset,
                            indices[idx + i + 1] + vertex_offset,
                        ]
                    )
                idx += face_count

            vertex_offset += len(points)

        for child in prim.GetChildren():
            traverse_prims(child)

    root_prim = stage.GetPseudoRoot()
    for child in root_prim.GetChildren():
        traverse_prims(child)

    if not all_vertices or not all_faces:
        return None

    return (
        trimesh.Trimesh(vertices=np.array(all_vertices), faces=np.array(all_faces)),
        adjusted_scale,
    )


def _prepare_mesh(mesh, up_axis, scale=1.0):
    prepared_mesh = mesh.copy()
    if isinstance(up_axis, (list, tuple)):
        axis_key = up_axis[0] if up_axis else "y"
    else:
        axis_key = up_axis
    axis_key = str(axis_key).lower()

    if "z" in axis_key:
        align_rotation = R.from_euler("xyz", [0, 180, 0], degrees=True).as_matrix()
    elif "y" in axis_key:
        align_rotation = R.from_euler("xyz", [-90, 180, 0], degrees=True).as_matrix()
    elif "x" in axis_key:
        align_rotation = R.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
    else:
        align_rotation = R.from_euler("xyz", [-90, 180, 0], degrees=True).as_matrix()

    align_transform = np.eye(4)
    align_transform[:3, :3] = align_rotation
    prepared_mesh.apply_transform(align_transform)
    prepared_mesh.apply_scale(scale)
    return prepared_mesh


def load_and_prepare_mesh(usd_path, up_axis, scale=1.0):
    if not os.path.exists(usd_path):
        return None
    mesh_info = load_mesh_from_usd(usd_path)
    if mesh_info is None:
        return None
    mesh, adjusted_scale = mesh_info
    return _prepare_mesh(mesh, up_axis, scale * adjusted_scale)


def _normalize_object_scale(scale_value, object_id):
    if scale_value is None:
        return 1.0

    if isinstance(scale_value, (list, tuple, np.ndarray)):
        scale_array = np.asarray(scale_value, dtype=float).reshape(-1)
        if scale_array.size == 0:
            return 1.0
        if scale_array.size > 1 and not np.allclose(scale_array, scale_array[0]):
            logger.warning(
                f"Non-uniform layout scale for {object_id} is not fully supported, "
                f"using {float(scale_array[0])} for mesh preparation"
            )
        return float(scale_array[0])

    return float(scale_value)


def _estimate_layout_unit_scale(raw_extents, obj_info):
    try:
        expected_size = np.abs(np.asarray(obj_info["size"], dtype=float).reshape(-1)[:3]) * 1000.0
        raw_extents = np.abs(np.asarray(raw_extents, dtype=float).reshape(-1)[:3])
    except (KeyError, TypeError, ValueError):
        return DEFAULT_LAYOUT_UNIT_SCALE, None

    valid_axes = np.isfinite(expected_size) & np.isfinite(raw_extents) & (expected_size > 0) & (raw_extents > 0)
    if valid_axes.sum() < 2:
        return DEFAULT_LAYOUT_UNIT_SCALE, None

    best_scale = DEFAULT_LAYOUT_UNIT_SCALE
    best_error = float("inf")
    for candidate in LAYOUT_UNIT_SCALE_CANDIDATES:
        scaled_extents = raw_extents[valid_axes] * candidate
        relative_error = np.max(
            np.abs(scaled_extents - expected_size[valid_axes]) / np.maximum(expected_size[valid_axes], 1e-6)
        )
        if relative_error < best_error:
            best_error = float(relative_error)
            best_scale = float(candidate)

    return best_scale, best_error


def _resolve_layout_mesh_scale(obj_info, raw_extents):
    object_id = obj_info.get("object_id", "unknown_object")
    object_scale = _normalize_object_scale(obj_info.get("scale", 1.0), object_id)
    unit_scale, fit_error = _estimate_layout_unit_scale(raw_extents, obj_info)

    if fit_error is not None and (unit_scale != DEFAULT_LAYOUT_UNIT_SCALE or fit_error > 0.1):
        logger.info(
            f"Resolved layout unit scale for {object_id}: object_scale={object_scale}, "
            f"unit_scale={unit_scale}, fit_error={fit_error:.4f}"
        )

    return object_scale * unit_scale


def setup_sdf(mesh):
    _, sdf_voxels = compute_sdf_from_obj_surface(mesh)
    # create callable sdf function with interpolation

    min_corner = mesh.bounds[0]
    max_corner = mesh.bounds[1]
    x = np.linspace(min_corner[0], max_corner[0], sdf_voxels.shape[0])
    y = np.linspace(min_corner[1], max_corner[1], sdf_voxels.shape[1])
    z = np.linspace(min_corner[2], max_corner[2], sdf_voxels.shape[2])
    sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
    return sdf_func


class LayoutObject(OmniObject):
    def __init__(self, obj_info, use_sdf=False, N_collision_points=60, **kwargs):
        super().__init__(name=obj_info["object_id"], **kwargs)
        data_info_dir = obj_info["data_info_dir"]
        usd_path = os.path.join(os.environ.get("SIM_ASSETS"), data_info_dir, "Aligned.usd")
        up_aixs = obj_info["upAxis"]
        if len(up_aixs) == 0:
            up_aixs = ["y"]
        logger.info(f"usd_path: {usd_path}")
        self.up_side_down = obj_info.get("up_side_down", False)
        self.object_type = obj_info.get("type", "rigid_body")
        mesh_info = load_mesh_from_usd(usd_path) if os.path.exists(usd_path) else None
        if mesh_info is None:
            self.mesh = None
        else:
            raw_mesh, adjusted_scale = mesh_info
            raw_extents = raw_mesh.extents * adjusted_scale
            mesh_scale = _resolve_layout_mesh_scale(obj_info, raw_extents)
            self.mesh = _prepare_mesh(raw_mesh, up_aixs, mesh_scale * adjusted_scale)

        if use_sdf and self.mesh is not None:
            self.sdf = setup_sdf(self.mesh)

        if (self.mesh is not None) and (self.object_type == "rigid_body"):
            mesh_points, _ = trimesh.sample.sample_surface(self.mesh, 2000)  # Surface sampling
            if mesh_points.shape[0] > N_collision_points:
                self.collision_points = farthest_point_sampling(
                    mesh_points, N_collision_points
                )  # Collision detection points
            self.anchor_points = {}
            self.anchor_points["top"] = get_bott_up_point(mesh_points, 1.5, descending=False)
            self.anchor_points["buttom"] = get_bott_up_point(mesh_points, 1.5, descending=True)
            self.anchor_points["top"] = random_point(self.anchor_points["top"], 3)[np.newaxis, :]
            self.anchor_points["buttom"] = random_point(self.anchor_points["buttom"], 3)[np.newaxis, :]
            self.size = self.mesh.extents.copy()
        self.up_axis = up_aixs[0]
