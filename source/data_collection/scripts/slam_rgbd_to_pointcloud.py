#!/usr/bin/env python3
"""Reconstruct a colored point cloud from a SLAM RGB/depth ROS2 bag frame.

The tool intentionally keeps dependencies light: rosbags + numpy are required,
Open3D is not.  It writes a binary PLY that can be opened by CloudCompare,
MeshLab, Open3D, etc.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rosbags.typesys import Stores, get_typestore


DEFAULT_RGB_TOPIC = "/head_front_left_color_rgb"
DEFAULT_DEPTH_TOPIC = "/head_front_left_color_depth"
DEFAULT_CAMERA_INFO_TOPIC = "/head_front_left_color_camera_info"


@dataclass(frozen=True)
class FrameRef:
    topic: str
    index: int
    header_stamp: float
    bag_timestamp_ns: int


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    d: list[float]
    source: str


def main() -> int:
    args = parse_args()
    bag_path = Path(args.bag).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_topic = args.rgb_topic
    depth_topic = args.depth_topic
    camera_info_topic = args.camera_info_topic

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = {connection.topic: connection for connection in reader.connections}
        if args.list_topics:
            print_topic_table(reader)
            return 0

        require_topic(connections, rgb_topic)
        require_topic(connections, depth_topic)
        if camera_info_topic not in connections:
            print(
                f"[warn] camera info topic {camera_info_topic!r} not in bag; "
                "will try external intrinsics.",
                file=sys.stderr,
            )

        rgb_refs, depth_refs, bag_intrinsics = scan_bag_frames(
            reader=reader,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
        )
        if not rgb_refs:
            raise RuntimeError(f"No RGB frames found on {rgb_topic}")
        if not depth_refs:
            raise RuntimeError(f"No depth frames found on {depth_topic}")

        rgb_ref = select_rgb_frame(rgb_refs, args.frame_index, args.time)
        depth_ref = nearest_frame(depth_refs, rgb_ref.header_stamp)
        dt = abs(depth_ref.header_stamp - rgb_ref.header_stamp)
        if args.max_dt is not None and dt > args.max_dt:
            raise RuntimeError(
                f"Nearest depth frame is {dt:.6f}s from RGB frame, "
                f"larger than --max-dt={args.max_dt}"
            )

        intrinsics = bag_intrinsics
        if intrinsics is None:
            intrinsics = load_external_intrinsics(
                bag_path=bag_path,
                intrinsics_path=Path(args.intrinsics).expanduser().resolve()
                if args.intrinsics
                else None,
                rgb_topic=rgb_topic,
                depth_topic=depth_topic,
                camera_info_topic=camera_info_topic,
                camera=args.camera,
            )
        if intrinsics is None:
            raise RuntimeError(
                "No valid intrinsics found. CameraInfo K/P are probably zero; "
                "pass --intrinsics camera_intrinsics.json or place it in the bag directory."
            )
        intrinsics = apply_intrinsics_overrides(
            intrinsics,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
        )

        rgb_msg, depth_msg, mask_msg = load_selected_messages(
            reader=reader,
            rgb_ref=rgb_ref,
            depth_ref=depth_ref,
            mask_topic=args.mask_topic,
            mask_time=rgb_ref.header_stamp,
        )

    rgb = decode_rgb(rgb_msg)
    depth_m = decode_depth_meters(depth_msg, args.depth_scale)
    if args.depth_is_range:
        depth_m = range_to_image_plane_depth(depth_m, intrinsics)
    mask = decode_mask(mask_msg) if mask_msg is not None else None

    xyz, colors, stats = rgbd_to_pointcloud(
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=intrinsics,
        stride=max(1, int(args.stride)),
        min_depth=float(args.min_depth),
        max_depth=float(args.max_depth) if args.max_depth is not None else None,
        max_points=int(args.max_points) if args.max_points else None,
        mask=mask,
        exclude_mask=bool(args.exclude_mask),
        viewer_y_up=bool(args.viewer_y_up),
    )

    stem = build_output_stem(rgb_topic, rgb_ref.index, rgb_ref.header_stamp)
    ply_path = output_dir / f"{stem}.ply"
    summary_path = output_dir / f"{stem}_summary.json"
    write_binary_ply(ply_path, xyz, colors)
    maybe_write_debug_images(output_dir, stem, rgb, depth_m, mask)

    summary = {
        "bag": str(bag_path),
        "rgb_topic": rgb_topic,
        "depth_topic": depth_topic,
        "camera_info_topic": camera_info_topic,
        "rgb_frame": asdict(rgb_ref),
        "depth_frame": asdict(depth_ref),
        "rgb_depth_dt_sec": dt,
        "intrinsics": asdict(intrinsics),
        "depth_scale_to_meters": infer_depth_scale(depth_msg, args.depth_scale),
        "depth_is_range_converted_to_z": bool(args.depth_is_range),
        "coordinate_frame": "ROS camera optical: +X right, +Y down, +Z forward"
        if not args.viewer_y_up
        else "viewer preview: +X right, +Y up, +Z forward",
        "pointcloud": {
            "path": str(ply_path),
            "point_count": int(len(xyz)),
            **stats,
        },
        "mask": {
            "topic": args.mask_topic,
            "excluded": bool(args.exclude_mask),
            "present": mask is not None,
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)
        file.write("\n")

    print(f"Wrote {len(xyz)} colored points: {ply_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        "Intrinsics used: "
        f"fx={intrinsics.fx:.6g}, fy={intrinsics.fy:.6g}, "
        f"cx={intrinsics.cx:.6g}, cy={intrinsics.cy:.6g} "
        f"({intrinsics.source})"
    )
    print(f"RGB/depth dt: {dt:.6f}s")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a colored PLY point cloud from one RGB/depth frame in a SLAM ROS2 bag.",
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="ROS2 bag directory or db3 file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/slam_rgbd_pointcloud",
        help="Directory for PLY, debug PNGs, and summary JSON.",
    )
    parser.add_argument("--rgb-topic", default=DEFAULT_RGB_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--camera-info-topic", default=DEFAULT_CAMERA_INFO_TOPIC)
    parser.add_argument(
        "--camera",
        default="head_front_left_color",
        help="Camera name or prim path used when matching external intrinsics.",
    )
    parser.add_argument(
        "--intrinsics",
        default="",
        help="Optional camera_intrinsics.json or recording_info.json. Auto-detected in bag dir if omitted.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="RGB frame index to reconstruct. Negative values count from the end.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Optional sim/header time in seconds. Selects nearest RGB frame and overrides --frame-index.",
    )
    parser.add_argument(
        "--max-dt",
        type=float,
        default=0.02,
        help="Maximum accepted RGB/depth timestamp delta in seconds. Use -1 to disable.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Factor from raw depth values to meters. Auto: float depth=1.0, uint16=0.001.",
    )
    parser.add_argument(
        "--depth-is-range",
        action="store_true",
        help="Convert Euclidean camera range to image-plane z before unprojection. "
        "Do not use for Isaac DistanceToImagePlane depth.",
    )
    parser.add_argument("--fx", type=float, default=None, help="Override intrinsics fx.")
    parser.add_argument("--fy", type=float, default=None, help="Override intrinsics fy.")
    parser.add_argument("--cx", type=float, default=None, help="Override intrinsics cx.")
    parser.add_argument("--cy", type=float, default=None, help="Override intrinsics cy.")
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth pixel. Increase to make a lighter PLY.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=600000,
        help="Randomly downsample if valid points exceed this count. 0 disables.",
    )
    parser.add_argument(
        "--mask-topic",
        default="",
        help="Optional mono8 robot mask topic to load near the RGB timestamp.",
    )
    parser.add_argument(
        "--exclude-mask",
        action="store_true",
        help="Exclude nonzero pixels from --mask-topic.",
    )
    parser.add_argument(
        "--viewer-y-up",
        action="store_true",
        help="Flip Y for easier standalone viewing. This changes the coordinate convention.",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="List bag topics and exit.",
    )
    args = parser.parse_args()
    if args.max_dt is not None and args.max_dt < 0:
        args.max_dt = None
    if args.max_points is not None and args.max_points <= 0:
        args.max_points = None
    return args


def print_topic_table(reader: AnyReader) -> None:
    for connection in sorted(reader.connections, key=lambda item: item.topic):
        print(f"{connection.topic}\t{connection.msgtype}\t{connection.msgcount}")


def require_topic(connections: dict[str, Any], topic: str) -> None:
    if topic not in connections:
        available = "\n".join(sorted(connections))
        raise RuntimeError(f"Topic {topic!r} not found in bag. Available topics:\n{available}")


def scan_bag_frames(
    *,
    reader: AnyReader,
    rgb_topic: str,
    depth_topic: str,
    camera_info_topic: str,
) -> tuple[list[FrameRef], list[FrameRef], Intrinsics | None]:
    rgb_refs: list[FrameRef] = []
    depth_refs: list[FrameRef] = []
    camera_info: Intrinsics | None = None
    counts = {rgb_topic: 0, depth_topic: 0}

    for connection, bag_timestamp_ns, raw in reader.messages():
        if connection.topic == rgb_topic:
            msg = reader.deserialize(raw, connection.msgtype)
            rgb_refs.append(
                FrameRef(
                    topic=rgb_topic,
                    index=counts[rgb_topic],
                    header_stamp=header_stamp_sec(msg.header),
                    bag_timestamp_ns=int(bag_timestamp_ns),
                )
            )
            counts[rgb_topic] += 1
        elif connection.topic == depth_topic:
            msg = reader.deserialize(raw, connection.msgtype)
            depth_refs.append(
                FrameRef(
                    topic=depth_topic,
                    index=counts[depth_topic],
                    header_stamp=header_stamp_sec(msg.header),
                    bag_timestamp_ns=int(bag_timestamp_ns),
                )
            )
            counts[depth_topic] += 1
        elif connection.topic == camera_info_topic and camera_info is None:
            msg = reader.deserialize(raw, connection.msgtype)
            candidate = intrinsics_from_camera_info(msg, source=f"bag:{camera_info_topic}")
            if candidate is not None:
                camera_info = candidate

    return rgb_refs, depth_refs, camera_info


def select_rgb_frame(refs: list[FrameRef], frame_index: int, time_sec: float | None) -> FrameRef:
    if time_sec is not None:
        return nearest_frame(refs, time_sec)
    index = int(frame_index)
    if index < 0:
        index = len(refs) + index
    if index < 0 or index >= len(refs):
        raise IndexError(f"frame index {frame_index} out of range for {len(refs)} RGB frames")
    return refs[index]


def nearest_frame(refs: list[FrameRef], time_sec: float) -> FrameRef:
    times = np.array([ref.header_stamp for ref in refs], dtype=np.float64)
    index = int(np.argmin(np.abs(times - float(time_sec))))
    return refs[index]


def load_selected_messages(
    *,
    reader: AnyReader,
    rgb_ref: FrameRef,
    depth_ref: FrameRef,
    mask_topic: str,
    mask_time: float,
) -> tuple[Any, Any, Any | None]:
    rgb_msg = None
    depth_msg = None
    mask_msg = None
    counts = {rgb_ref.topic: 0, depth_ref.topic: 0}
    best_mask_dt = math.inf

    for connection, _, raw in reader.messages():
        if connection.topic == rgb_ref.topic:
            if counts[rgb_ref.topic] == rgb_ref.index:
                rgb_msg = reader.deserialize(raw, connection.msgtype)
            counts[rgb_ref.topic] += 1
        elif connection.topic == depth_ref.topic:
            if counts[depth_ref.topic] == depth_ref.index:
                depth_msg = reader.deserialize(raw, connection.msgtype)
            counts[depth_ref.topic] += 1
        elif mask_topic and connection.topic == mask_topic:
            candidate = reader.deserialize(raw, connection.msgtype)
            dt = abs(header_stamp_sec(candidate.header) - mask_time)
            if dt < best_mask_dt:
                mask_msg = candidate
                best_mask_dt = dt

        if rgb_msg is not None and depth_msg is not None and (not mask_topic or best_mask_dt == 0.0):
            if not mask_topic:
                break

    if rgb_msg is None:
        raise RuntimeError(f"Failed to reload selected RGB frame {rgb_ref}")
    if depth_msg is None:
        raise RuntimeError(f"Failed to reload selected depth frame {depth_ref}")
    if mask_topic and mask_msg is None:
        print(f"[warn] mask topic {mask_topic!r} had no messages", file=sys.stderr)
    return rgb_msg, depth_msg, mask_msg


def header_stamp_sec(header: Any) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def intrinsics_from_camera_info(msg: Any, source: str) -> Intrinsics | None:
    k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(k).all() or abs(k[0, 0]) < 1e-9 or abs(k[1, 1]) < 1e-9 or abs(k[2, 2]) < 1e-9:
        return None
    return Intrinsics(
        width=int(msg.width),
        height=int(msg.height),
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        distortion_model=str(getattr(msg, "distortion_model", "")),
        d=[float(value) for value in getattr(msg, "d", [])],
        source=source,
    )


def apply_intrinsics_overrides(
    intrinsics: Intrinsics,
    *,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
) -> Intrinsics:
    overrides = {
        "fx": intrinsics.fx if fx is None else float(fx),
        "fy": intrinsics.fy if fy is None else float(fy),
        "cx": intrinsics.cx if cx is None else float(cx),
        "cy": intrinsics.cy if cy is None else float(cy),
    }
    if (
        overrides["fx"] == intrinsics.fx
        and overrides["fy"] == intrinsics.fy
        and overrides["cx"] == intrinsics.cx
        and overrides["cy"] == intrinsics.cy
    ):
        return intrinsics
    return Intrinsics(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=overrides["fx"],
        fy=overrides["fy"],
        cx=overrides["cx"],
        cy=overrides["cy"],
        distortion_model=intrinsics.distortion_model,
        d=intrinsics.d,
        source=f"{intrinsics.source} + cli_override",
    )


def load_external_intrinsics(
    *,
    bag_path: Path,
    intrinsics_path: Path | None,
    rgb_topic: str,
    depth_topic: str,
    camera_info_topic: str,
    camera: str,
) -> Intrinsics | None:
    candidates: list[Path] = []
    if intrinsics_path is not None:
        candidates.append(intrinsics_path)
    bag_dir = bag_path if bag_path.is_dir() else bag_path.parent
    candidates.extend([bag_dir / "camera_intrinsics.json", bag_dir / "recording_info.json"])

    for path in candidates:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        intrinsics = intrinsics_from_external_payload(
            payload=payload,
            path=path,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
            camera=camera,
        )
        if intrinsics is not None:
            return intrinsics
    return None


def intrinsics_from_external_payload(
    *,
    payload: dict[str, Any],
    path: Path,
    rgb_topic: str,
    depth_topic: str,
    camera_info_topic: str,
    camera: str,
) -> Intrinsics | None:
    if "cameras" in payload:
        items = payload.get("cameras", {}).items()
    elif "camera_intrinsics" in payload:
        items = payload.get("camera_intrinsics", {}).items()
    else:
        items = [("", payload)]

    for key, value in items:
        if not isinstance(value, dict):
            continue
        if not external_camera_matches(
            key=key,
            value=value,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
            camera=camera,
        ):
            continue
        intrinsics = parse_external_intrinsics(value, source=str(path))
        if intrinsics is not None:
            return intrinsics
    return None


def external_camera_matches(
    *,
    key: str,
    value: dict[str, Any],
    rgb_topic: str,
    depth_topic: str,
    camera_info_topic: str,
    camera: str,
) -> bool:
    needles = {
        rgb_topic,
        depth_topic,
        camera_info_topic,
        camera,
        camera.strip("/").split("/")[-1],
        topic_to_camera_name(rgb_topic),
    }
    haystack = {
        key,
        str(value.get("camera_name", "")),
        str(value.get("camera_prim", "")),
        str(value.get("rgb_topic", "")),
        str(value.get("depth_topic", "")),
        str(value.get("camera_info_topic", "")),
    }
    haystack.update({item.strip("/").split("/")[-1] for item in list(haystack)})
    return any(needle and needle in haystack for needle in needles)


def parse_external_intrinsics(value: dict[str, Any], source: str) -> Intrinsics | None:
    k = value.get("K", value.get("k"))
    if k is None:
        return None
    k_matrix = parse_matrix(k, 3, 3)
    if k_matrix is None:
        return None
    if abs(k_matrix[0, 0]) < 1e-9 or abs(k_matrix[1, 1]) < 1e-9:
        return None
    return Intrinsics(
        width=int(value.get("width", 0)),
        height=int(value.get("height", 0)),
        fx=float(k_matrix[0, 0]),
        fy=float(k_matrix[1, 1]),
        cx=float(k_matrix[0, 2]),
        cy=float(k_matrix[1, 2]),
        distortion_model=str(value.get("distortion_model", "plumb_bob")),
        d=[float(item) for item in value.get("D", value.get("d", []))],
        source=source,
    )


def parse_matrix(raw: Any, rows: int, cols: int) -> np.ndarray | None:
    array = np.asarray(raw, dtype=np.float64)
    if array.shape == (rows, cols):
        return array
    if array.size == rows * cols:
        return array.reshape(rows, cols)
    return None


def topic_to_camera_name(topic: str) -> str:
    name = topic.strip("/").split("/")[-1]
    for suffix in ("_rgb", "_depth", "_camera_info", "_robot_mask"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def decode_rgb(msg: Any) -> np.ndarray:
    image = message_to_cvimage(msg, "rgb8")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Unexpected RGB image shape: {image.shape}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def decode_depth_meters(msg: Any, requested_scale: float | None) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    if encoding in {"32fc1", "passthrough"}:
        depth = message_to_cvimage(msg, "32FC1").astype(np.float32)
    elif encoding == "16uc1":
        depth = message_to_cvimage(msg, "16UC1").astype(np.float32)
    else:
        depth = message_to_cvimage(msg, msg.encoding).astype(np.float32)
    scale = infer_depth_scale(msg, requested_scale)
    return np.ascontiguousarray(depth * scale, dtype=np.float32)


def infer_depth_scale(msg: Any, requested_scale: float | None) -> float:
    if requested_scale is not None:
        return float(requested_scale)
    encoding = str(msg.encoding).lower()
    if encoding == "16uc1":
        return 0.001
    return 1.0


def decode_mask(msg: Any) -> np.ndarray:
    if str(msg.encoding).lower() == "mono8":
        mask = message_to_cvimage(msg, "mono8")
    else:
        mask = message_to_cvimage(msg, msg.encoding)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return np.ascontiguousarray(mask)


def range_to_image_plane_depth(depth_range: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    height, width = depth_range.shape
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x_norm = (uu - intrinsics.cx) / intrinsics.fx
    y_norm = (vv - intrinsics.cy) / intrinsics.fy
    ray_norm = np.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
    return depth_range / ray_norm


def rgbd_to_pointcloud(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    stride: int,
    min_depth: float,
    max_depth: float | None,
    max_points: int | None,
    mask: np.ndarray | None,
    exclude_mask: bool,
    viewer_y_up: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if rgb.shape[:2] != depth_m.shape[:2]:
        raise RuntimeError(f"RGB/depth size mismatch: rgb={rgb.shape}, depth={depth_m.shape}")
    height, width = depth_m.shape
    if intrinsics.width and intrinsics.width != width:
        print(
            f"[warn] intrinsics width={intrinsics.width}, image width={width}; using image width grid.",
            file=sys.stderr,
        )
    if intrinsics.height and intrinsics.height != height:
        print(
            f"[warn] intrinsics height={intrinsics.height}, image height={height}; using image height grid.",
            file=sys.stderr,
        )
    if mask is not None and mask.shape[:2] != depth_m.shape:
        raise RuntimeError(f"Mask/depth size mismatch: mask={mask.shape}, depth={depth_m.shape}")

    rows = np.arange(0, height, stride, dtype=np.float32)
    cols = np.arange(0, width, stride, dtype=np.float32)
    uu, vv = np.meshgrid(cols, rows)
    sampled_depth = depth_m[::stride, ::stride]
    sampled_rgb = rgb[::stride, ::stride, :]

    valid = np.isfinite(sampled_depth) & (sampled_depth >= min_depth)
    if max_depth is not None:
        valid &= sampled_depth <= max_depth
    if mask is not None and exclude_mask:
        valid &= mask[::stride, ::stride] == 0

    z = sampled_depth[valid].astype(np.float32)
    x = ((uu[valid] - intrinsics.cx) * z / intrinsics.fx).astype(np.float32)
    y = ((vv[valid] - intrinsics.cy) * z / intrinsics.fy).astype(np.float32)
    if viewer_y_up:
        y = -y
    xyz = np.column_stack((x, y, z)).astype(np.float32)
    colors = sampled_rgb[valid].reshape(-1, 3).astype(np.uint8)

    valid_before_downsample = int(len(xyz))
    if max_points is not None and len(xyz) > max_points:
        rng = np.random.default_rng(20260520)
        indices = rng.choice(len(xyz), size=max_points, replace=False)
        indices.sort()
        xyz = xyz[indices]
        colors = colors[indices]

    stats = {
        "image_width": int(width),
        "image_height": int(height),
        "stride": int(stride),
        "valid_points_before_downsample": valid_before_downsample,
        "min_depth_m": float(np.nanmin(z)) if len(z) else None,
        "max_depth_m": float(np.nanmax(z)) if len(z) else None,
        "mean_depth_m": float(np.nanmean(z)) if len(z) else None,
        "downsampled": bool(max_points is not None and valid_before_downsample > len(xyz)),
    }
    return xyz, colors, stats


def write_binary_ply(path: Path, xyz: np.ndarray, colors: np.ndarray) -> None:
    if xyz.shape[0] != colors.shape[0]:
        raise RuntimeError("xyz/color point count mismatch")
    points = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    points["x"] = xyz[:, 0]
    points["y"] = xyz[:, 1]
    points["z"] = xyz[:, 2]
    points["red"] = colors[:, 0]
    points["green"] = colors[:, 1]
    points["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        points.tofile(file)


def maybe_write_debug_images(
    output_dir: Path,
    stem: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray | None,
) -> None:
    try:
        import cv2
    except Exception:
        return

    cv2.imwrite(str(output_dir / f"{stem}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if np.any(valid):
        low, high = np.percentile(depth_m[valid], [1, 99])
        if high <= low:
            high = low + 1.0
        depth_norm = np.clip((depth_m - low) / (high - low), 0.0, 1.0)
        depth_u8 = (depth_norm * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(output_dir / f"{stem}_depth_preview.png"), depth_color)
    if mask is not None:
        cv2.imwrite(str(output_dir / f"{stem}_mask.png"), mask.astype(np.uint8))


def build_output_stem(rgb_topic: str, frame_index: int, stamp: float) -> str:
    camera = topic_to_camera_name(rgb_topic)
    stamp_token = f"{stamp:.6f}".replace(".", "_")
    return f"{camera}_frame_{frame_index:06d}_{stamp_token}"


if __name__ == "__main__":
    raise SystemExit(main())
