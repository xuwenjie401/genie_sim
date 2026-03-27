#!/usr/bin/env python3
"""Export one recording_data episode into trajectory JSON files.

Usage:
    python source/data_collection/scripts/extract_recording_trajectory_jsons.py \
        --dir /abs/path/to/recording_data/<episode_dir>

The script writes:
    - joint_states_trajectory.json
    - left_tcp_pose_trajectory.json
    - right_tcp_pose_trajectory.json
    - torso_pose_trajectory.json
    - base_link_pose_trajectory.json

`torso_pose_trajectory.json` is sourced from `robot.arm_base_pose`, which is
the recorded torso base pose in this dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


POSE_EXPORT_SPECS = (
    (
        "left_tcp_pose_trajectory.json",
        ("ee", "left", "pose"),
        "left_tcp",
        "left_tcp",
    ),
    (
        "right_tcp_pose_trajectory.json",
        ("ee", "right", "pose"),
        "right_tcp",
        "right_tcp",
    ),
    (
        "torso_pose_trajectory.json",
        ("robot", "arm_base_pose"),
        "torso",
        "torso_base_link",
    ),
    (
        "base_link_pose_trajectory.json",
        ("robot", "pose"),
        "base_link",
        "base_link",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract joint-state and pose trajectories from one recording_data directory."
    )
    parser.add_argument(
        "--dir",
        "--recording-dir",
        dest="recording_dir",
        required=True,
        help="Path to one recording_data episode directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for exported JSONs. Defaults to the recording directory itself.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level. Defaults to 2.",
    )
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Skip aligned_joints_all.h5 and read joints from state.json only.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any], indent: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
        f.write("\n")


def _state_chunk_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("state_"):
        suffix = stem.split("_", 1)[1]
        if suffix.isdigit():
            return (int(suffix), path.name)
    return (sys.maxsize, path.name)


def load_state_frames(recording_dir: Path) -> tuple[list[dict[str, Any]], str]:
    state_path = recording_dir / "state.json"
    if state_path.is_file():
        state = load_json(state_path)
        frames = state.get("frames", [])
        if not isinstance(frames, list):
            raise SystemExit(f"Invalid frames list in {state_path}")
        return frames, str(state_path)

    chunk_paths = sorted(recording_dir.glob("state_*.json"), key=_state_chunk_sort_key)
    if not chunk_paths:
        raise SystemExit(f"Missing required file: {state_path}")

    frames: list[dict[str, Any]] = []
    loaded_chunks = 0
    skipped_chunks: list[str] = []
    for chunk_path in chunk_paths:
        try:
            chunk_state = load_json(chunk_path)
        except json.JSONDecodeError as exc:
            skipped_chunks.append(f"{chunk_path.name} ({exc})")
            continue
        chunk_frames = chunk_state.get("frames", [])
        if not isinstance(chunk_frames, list):
            skipped_chunks.append(f"{chunk_path.name} (invalid frames)")
            continue
        frames.extend(chunk_frames)
        loaded_chunks += 1

    if not frames:
        skipped = ", ".join(skipped_chunks) if skipped_chunks else "none"
        raise SystemExit(
            f"Failed to load usable frames from chunked state files in {recording_dir}. "
            f"Skipped: {skipped}"
        )

    if skipped_chunks:
        print("warning: some chunked state files were skipped:", file=sys.stderr)
        for skipped in skipped_chunks:
            print(f"warning:   {skipped}", file=sys.stderr)

    return frames, f"{loaded_chunks} chunk file(s)"


def decode_names(values: Any) -> list[str]:
    names: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_recording_info(recording_dir: Path) -> dict[str, Any]:
    path = recording_dir / "recording_info.json"
    if not path.is_file():
        return {}
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def extract_state_timestamps(frames: list[dict[str, Any]]) -> list[float]:
    timestamps: list[float] = []
    for index, frame in enumerate(frames):
        if "time_stamp" not in frame:
            raise SystemExit(f"Missing time_stamp in state frame {index}")
        timestamps.append(float(frame["time_stamp"]))
    return timestamps


def timestamps_match(reference: list[float], candidate: list[float], tol: float = 1e-4) -> bool:
    if len(reference) != len(candidate):
        return False
    for left, right in zip(reference, candidate):
        if not math.isclose(left, right, abs_tol=tol):
            return False
    return True


def load_joint_trajectory_from_h5(
    recording_dir: Path, reference_timestamps: list[float]
) -> tuple[dict[str, Any], str] | None:
    h5_path = recording_dir / "aligned_joints_all.h5"
    if not h5_path.is_file():
        return None

    try:
        import h5py  # type: ignore
    except ImportError:
        print(
            "warning: h5py is not installed; falling back to state.json for joint states",
            file=sys.stderr,
        )
        return None

    try:
        with h5py.File(h5_path, "r") as f:
            if "timestamp" not in f or "state/joint/position" not in f:
                print(
                    f"warning: {h5_path} is missing required datasets; "
                    "falling back to state.json for joint states",
                    file=sys.stderr,
                )
                return None

            source_timestamps = [float(value) for value in f["timestamp"][:].tolist()]
            if len(reference_timestamps) != len(source_timestamps):
                print(
                    f"warning: {h5_path.name} frame count does not align with state frames; "
                    "falling back to state.json for joint states",
                    file=sys.stderr,
                )
                return None

            if not timestamps_match(reference_timestamps, source_timestamps):
                max_timestamp_diff = max(
                    abs(left - right) for left, right in zip(reference_timestamps, source_timestamps)
                )
                print(
                    f"warning: {h5_path.name} timestamps differ from state frames by up to "
                    f"{max_timestamp_diff:.6f}s; using state.json timestamps with HDF5 joint arrays",
                    file=sys.stderr,
                )

            names = decode_names(f["state/joint"].attrs.get("name", []))
            positions = [[maybe_float(value) for value in row] for row in f["state/joint/position"][:].tolist()]
            velocities = [[maybe_float(value) for value in row] for row in f["state/joint/velocity"][:].tolist()]
            efforts = [[maybe_float(value) for value in row] for row in f["state/joint/effort"][:].tolist()]
    except OSError as exc:
        print(
            f"warning: failed to read {h5_path}: {exc}; "
            "falling back to state.json for joint states",
            file=sys.stderr,
        )
        return None

    return (
        {
            "joint_names": names,
            "timestamps": reference_timestamps,
            "positions": positions,
            "velocities": velocities,
            "efforts": efforts,
            "time_stamp_source": "state.json",
        },
        str(h5_path),
    )


def load_joint_trajectory_from_state(
    frames: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    if not frames:
        raise SystemExit("No state frames available to extract joint states")

    first_joint_block = (((frames[0].get("robot") or {}).get("joints")) or {})
    names = [str(value) for value in first_joint_block.get("joint_name", [])]
    if not names:
        raise SystemExit("Missing robot.joints.joint_name in state.json")

    timestamps: list[float] = []
    positions: list[list[float | None]] = []
    velocities: list[list[float | None]] = []
    efforts: list[list[float | None]] = []

    for index, frame in enumerate(frames):
        robot = frame.get("robot")
        if not isinstance(robot, dict):
            raise SystemExit(f"Missing robot block in state frame {index}")
        joints = robot.get("joints")
        if not isinstance(joints, dict):
            raise SystemExit(f"Missing robot.joints in state frame {index}")

        joint_names = [str(value) for value in joints.get("joint_name", [])]
        if joint_names != names:
            raise SystemExit(f"Joint name mismatch in state frame {index}")

        timestamps.append(float(frame["time_stamp"]))
        positions.append([maybe_float(value) for value in joints.get("joint_position", [])])
        velocities.append([maybe_float(value) for value in joints.get("joint_velocity", [])])
        efforts.append([maybe_float(value) for value in joints.get("joint_effort", [])])

    return (
        {
            "joint_names": names,
            "timestamps": timestamps,
            "positions": positions,
            "velocities": velocities,
            "efforts": efforts,
            "time_stamp_source": "state.json",
        },
        "state.json",
    )


def validate_joint_series(data: dict[str, Any]) -> None:
    joint_names = data["joint_names"]
    expected_joint_count = len(joint_names)
    frame_count = len(data["timestamps"])

    for key in ("positions", "velocities", "efforts"):
        values = data[key]
        if len(values) != frame_count:
            raise SystemExit(f"Joint trajectory field '{key}' has inconsistent frame count")
        for index, row in enumerate(values):
            if len(row) != expected_joint_count:
                raise SystemExit(
                    f"Joint trajectory field '{key}' frame {index} has "
                    f"{len(row)} values, expected {expected_joint_count}"
                )


def build_joint_trajectory_payload(
    recording_dir: Path,
    recording_info: dict[str, Any],
    joint_data: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    validate_joint_series(joint_data)
    joint_names = joint_data["joint_names"]
    frames: list[dict[str, Any]] = []

    for index, timestamp in enumerate(joint_data["timestamps"]):
        joints: dict[str, dict[str, float | None]] = {}
        for joint_index, joint_name in enumerate(joint_names):
            joints[joint_name] = {
                "position": joint_data["positions"][index][joint_index],
                "velocity": joint_data["velocities"][index][joint_index],
                "effort": joint_data["efforts"][index][joint_index],
            }
        frames.append(
            {
                "frame_index": index,
                "time_stamp": float(timestamp),
                "joints": joints,
            }
        )

    return {
        "recording_name": recording_dir.name,
        "recording_dir": str(recording_dir),
        "robot_name": recording_info.get("robot_name"),
        "fps": recording_info.get("fps"),
        "joint_count": len(joint_names),
        "joint_names": joint_names,
        "frame_count": len(frames),
        "source": source_path,
        "time_stamp_source": joint_data.get("time_stamp_source", "state.json"),
        "frames": frames,
    }


def get_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


def normalize_quaternion_wxyz(quaternion: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 0.0:
        return [1.0, 0.0, 0.0, 0.0]
    return [value / norm for value in quaternion]


def rotation_matrix_to_quaternion_wxyz(rotation: list[list[float]]) -> list[float]:
    r11, r12, r13 = rotation[0]
    r21, r22, r23 = rotation[1]
    r31, r32, r33 = rotation[2]
    trace = r11 + r22 + r33

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r32 - r23) / s
        qy = (r13 - r31) / s
        qz = (r21 - r12) / s
    elif r11 > r22 and r11 > r33:
        s = math.sqrt(max(1.0 + r11 - r22 - r33, 0.0)) * 2.0
        qw = (r32 - r23) / s
        qx = 0.25 * s
        qy = (r12 + r21) / s
        qz = (r13 + r31) / s
    elif r22 > r33:
        s = math.sqrt(max(1.0 + r22 - r11 - r33, 0.0)) * 2.0
        qw = (r13 - r31) / s
        qx = (r12 + r21) / s
        qy = 0.25 * s
        qz = (r23 + r32) / s
    else:
        s = math.sqrt(max(1.0 + r33 - r11 - r22, 0.0)) * 2.0
        qw = (r21 - r12) / s
        qx = (r13 + r31) / s
        qy = (r23 + r32) / s
        qz = 0.25 * s

    return normalize_quaternion_wxyz([qw, qx, qy, qz])


def pose_matrix_to_payload(matrix: Any) -> dict[str, Any]:
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise ValueError("pose matrix must be a 4x4 list")

    matrix_4x4: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("pose matrix must be a 4x4 list")
        matrix_4x4.append([float(value) for value in row])

    xyz = [matrix_4x4[0][3], matrix_4x4[1][3], matrix_4x4[2][3]]
    rotation = [matrix_4x4[0][:3], matrix_4x4[1][:3], matrix_4x4[2][:3]]
    quaternion = rotation_matrix_to_quaternion_wxyz(rotation)

    return {
        "matrix": matrix_4x4,
        "translation_xyz": xyz,
        "quaternion_wxyz": quaternion,
    }


def build_pose_trajectory_payload(
    recording_dir: Path,
    recording_info: dict[str, Any],
    frames: list[dict[str, Any]],
    source_path: str,
    pose_path: tuple[str, ...],
    pose_name: str,
    link_name: str,
) -> dict[str, Any]:
    trajectory_frames: list[dict[str, Any]] = []

    for index, frame in enumerate(frames):
        try:
            matrix = get_nested(frame, pose_path)
        except KeyError as exc:
            raise SystemExit(f"Missing pose {exc} in state frame {index}") from exc

        try:
            pose_payload = pose_matrix_to_payload(matrix)
        except ValueError as exc:
            raise SystemExit(f"Invalid pose for {pose_name} in state frame {index}: {exc}") from exc

        trajectory_frames.append(
            {
                "frame_index": index,
                "time_stamp": float(frame["time_stamp"]),
                **pose_payload,
            }
        )

    return {
        "recording_name": recording_dir.name,
        "recording_dir": str(recording_dir),
        "robot_name": recording_info.get("robot_name"),
        "fps": recording_info.get("fps"),
        "pose_name": pose_name,
        "link_name": link_name,
        "reference_frame": "world",
        "frame_count": len(trajectory_frames),
        "source": source_path,
        "frames": trajectory_frames,
    }


def main() -> int:
    args = parse_args()
    recording_dir = Path(args.recording_dir).expanduser().resolve()
    if not recording_dir.is_dir():
        raise SystemExit(f"Recording directory does not exist: {recording_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else recording_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, state_source = load_state_frames(recording_dir)
    recording_info = load_recording_info(recording_dir)
    reference_timestamps = extract_state_timestamps(frames)

    joint_result: tuple[dict[str, Any], str] | None = None
    if not args.state_only:
        joint_result = load_joint_trajectory_from_h5(recording_dir, reference_timestamps)
    if joint_result is None:
        joint_result = load_joint_trajectory_from_state(frames)

    joint_data, joint_source = joint_result
    joint_payload = build_joint_trajectory_payload(
        recording_dir=recording_dir,
        recording_info=recording_info,
        joint_data=joint_data,
        source_path=joint_source,
    )

    outputs: list[tuple[str, dict[str, Any]]] = [
        ("joint_states_trajectory.json", joint_payload),
    ]

    for file_name, pose_path, pose_name, link_name in POSE_EXPORT_SPECS:
        payload = build_pose_trajectory_payload(
            recording_dir=recording_dir,
            recording_info=recording_info,
            frames=frames,
            source_path=state_source,
            pose_path=pose_path,
            pose_name=pose_name,
            link_name=link_name,
        )
        outputs.append((file_name, payload))

    for file_name, payload in outputs:
        output_path = output_dir / file_name
        write_json(output_path, payload, indent=args.indent)
        print(f"wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
