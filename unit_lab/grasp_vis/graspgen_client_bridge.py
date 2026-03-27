#!/usr/bin/env python3
"""Bridge GraspGen ZMQ inference into GenieSim grasp-pose arrays.

Run this script inside the `GraspGen` conda environment. It queries a running
GraspGen ZMQ server, converts the predicted grasps back to the original object
frame, estimates Robotiq widths from the sampled object surface, and saves a
compact `.npz` payload that the Isaac-side editor can load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a GraspGen ZMQ server and save raw grasp results to NPZ.")
    parser.add_argument("--graspgen_root", type=Path, default=Path("/home/agxi/ManipLab/GraspGen"))
    parser.add_argument("--mesh_file", type=Path, required=True, help="Mesh or USD file to sample from.")
    parser.add_argument("--mesh_scale", type=float, default=1.0)
    parser.add_argument("--gripper_config", type=Path, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--num_sample_points", type=int, default=2048)
    parser.add_argument("--num_grasps", type=int, default=120)
    parser.add_argument("--topk_num_grasps", type=int, default=64)
    parser.add_argument("--grasp_threshold", type=float, default=-1.0)
    parser.add_argument("--output_npz", type=Path, required=True)
    return parser.parse_args()


def ensure_graspgen_imports(graspgen_root: Path) -> None:
    root = graspgen_root.resolve()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _merge_scene_mesh(mesh: Any) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if hasattr(mesh, "dump"):
        parts = [part for part in mesh.dump() if isinstance(part, trimesh.Trimesh)]
        if parts:
            return trimesh.util.concatenate(parts)
    raise TypeError(f"Unsupported mesh type: {type(mesh)}")


def load_mesh_data(mesh_file: Path, scale: float, num_sample_points: int) -> tuple[np.ndarray, np.ndarray]:
    suffix = mesh_file.suffix.lower()
    if suffix in {".usd", ".usda", ".usdc", ".usdz"}:
        import scene_synthesizer as synth

        asset = synth.Asset(str(mesh_file))
        mesh = asset.mesh()
    else:
        mesh = trimesh.load(str(mesh_file), force="mesh")

    mesh = _merge_scene_mesh(mesh).copy()
    mesh.apply_scale(float(scale))
    xyz, _ = trimesh.sample.sample_surface(mesh, int(num_sample_points))
    xyz = np.asarray(xyz, dtype=np.float32)

    center = xyz.mean(axis=0)
    center_shift = tra.translation_matrix(-center)
    xyz_centered = tra.transform_points(xyz, center_shift).astype(np.float32)
    return xyz_centered, center_shift


def load_gripper_config_data(gripper_config_path: Path) -> tuple[str, dict[str, Any]]:
    with gripper_config_path.open("r", encoding="utf-8") as f:
        grasp_cfg = yaml.safe_load(f)
    gripper_name = str(grasp_cfg["data"]["gripper_name"])

    from grasp_gen.robot import load_default_gripper_config

    return gripper_name, load_default_gripper_config(gripper_name)


def estimate_widths_from_points(
    point_cloud_world: np.ndarray,
    grasps_world: np.ndarray,
    gripper_data: dict[str, Any],
) -> np.ndarray:
    max_aperture = float(gripper_data.get("maximum_aperture", gripper_data.get("width", 0.08)))
    min_width = 0.005

    closing_regions = gripper_data.get("closing_regions", [])
    if closing_regions:
        region = closing_regions[0]
        center = np.asarray(region.get("translation", [0.0, 0.0, gripper_data.get("depth", 0.0)]), dtype=np.float64)
        extents = np.asarray(
            region.get(
                "extents",
                [
                    max_aperture,
                    0.02,
                    max(float(gripper_data.get("depth", 0.05)) * 0.35, 0.04),
                ],
            ),
            dtype=np.float64,
        )
    else:
        center = np.array([0.0, 0.0, float(gripper_data.get("depth", 0.0))], dtype=np.float64)
        extents = np.array([max_aperture, 0.02, max(float(gripper_data.get("depth", 0.05)) * 0.35, 0.04)], dtype=np.float64)

    if point_cloud_world.size == 0 or grasps_world.size == 0:
        return np.zeros((0,), dtype=np.float32)

    points_h = np.concatenate(
        [np.asarray(point_cloud_world, dtype=np.float64), np.ones((len(point_cloud_world), 1), dtype=np.float64)],
        axis=1,
    )
    widths: list[float] = []

    for grasp in np.asarray(grasps_world, dtype=np.float64):
        local = (np.linalg.inv(grasp) @ points_h.T).T[:, :3]
        centered = local - center[np.newaxis, :]

        mask = (
            (np.abs(centered[:, 1]) <= extents[1] * 0.75)
            & (np.abs(centered[:, 2]) <= extents[2] * 0.75)
        )
        selected = local[mask]

        if len(selected) < 8:
            mask = (
                (np.abs(centered[:, 1]) <= extents[1] * 1.5)
                & (np.abs(centered[:, 2]) <= extents[2] * 1.5)
            )
            selected = local[mask]

        if len(selected) < 4:
            mask = np.abs(centered[:, 2]) <= max(extents[2] * 2.0, 0.05)
            selected = local[mask]

        if len(selected) >= 2:
            width = float(np.max(selected[:, 0]) - np.min(selected[:, 0]))
            width = width * 1.05 + 0.002
        else:
            width = max_aperture * 0.35

        widths.append(float(np.clip(width, min_width, max_aperture)))

    return np.asarray(widths, dtype=np.float32)


def run_inference(
    point_cloud: np.ndarray,
    host: str,
    port: int,
    num_grasps: int,
    topk_num_grasps: int,
    grasp_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    from grasp_gen.serving.zmq_client import GraspGenClient

    with GraspGenClient(host=host, port=port) as client:
        grasps, confidences = client.infer(
            np.asarray(point_cloud, dtype=np.float32),
            num_grasps=int(num_grasps),
            topk_num_grasps=int(topk_num_grasps),
            grasp_threshold=float(grasp_threshold),
        )
    return np.asarray(grasps, dtype=np.float64), np.asarray(confidences, dtype=np.float32)


def main() -> int:
    args = parse_args()
    ensure_graspgen_imports(args.graspgen_root)

    point_cloud_centered, subtract_pc_mean = load_mesh_data(
        args.mesh_file,
        args.mesh_scale,
        args.num_sample_points,
    )
    grasps_centered, confidences = run_inference(
        point_cloud_centered,
        args.host,
        args.port,
        args.num_grasps,
        args.topk_num_grasps,
        args.grasp_threshold,
    )

    point_cloud_world = tra.transform_points(point_cloud_centered, tra.inverse_matrix(subtract_pc_mean))
    if len(grasps_centered) > 0:
        grasps_world = np.asarray(
            [tra.inverse_matrix(subtract_pc_mean) @ grasp for grasp in grasps_centered],
            dtype=np.float64,
        )
    else:
        grasps_world = np.zeros((0, 4, 4), dtype=np.float64)

    gripper_name, gripper_data = load_gripper_config_data(args.gripper_config)
    widths = estimate_widths_from_points(point_cloud_world, grasps_world, gripper_data)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        grasp_pose_raw=grasps_world.astype(np.float64),
        width=widths.astype(np.float32),
        confidence=confidences.astype(np.float32),
        gripper_name=np.asarray([gripper_name]),
    )
    print(f"saved {len(grasps_world)} grasps to {args.output_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
