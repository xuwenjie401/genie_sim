#!/usr/bin/env python3
"""Recover binary gripper action labels for extracted recording_data episodes.

GA semantics:
    - 1: open or opening
    - 0: close or holding
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_COLLECTION_ROOT = SCRIPT_DIR.parent
if str(DATA_COLLECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_ROOT))

from common.base_utils.gripper_action_utils import (  # noqa: E402
    GA_CLOSED,
    GA_OPEN,
    build_gripper_action_frame_payload,
    build_gripper_action_matrix,
    build_gripper_action_summary,
    frames_to_joint_matrix,
    recover_binary_gripper_actions,
)


SUMMARY_FILENAME = "gripper_action_recovery.json"
PLOT_FILENAME = "gripper_action_recovery.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover binary gripper-action labels.")
    parser.add_argument(
        "--dir",
        dest="inputs",
        action="append",
        required=True,
        help="Episode directory or a parent directory that contains episode subdirectories.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob used when an input path is a parent directory.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write recovered GA into state.json, aligned_joints_all.h5, and a summary json.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write a diagnostic PNG for each processed episode.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def collect_episode_dirs(inputs: list[str], pattern: str) -> list[Path]:
    episode_dirs: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        if (path / "state.json").is_file():
            episode_dirs.append(path)
            continue
        if not path.is_dir():
            continue
        for candidate in sorted(path.glob(pattern)):
            if candidate.is_dir() and (candidate / "state.json").is_file():
                episode_dirs.append(candidate.resolve())

    seen: set[Path] = set()
    unique_dirs: list[Path] = []
    for episode_dir in episode_dirs:
        if episode_dir in seen:
            continue
        seen.add(episode_dir)
        unique_dirs.append(episode_dir)
    return unique_dirs


def apply_gripper_action_to_frames(
    state_payload: dict[str, Any],
    recovery: dict[str, Any],
) -> None:
    frames = state_payload.get("frames", [])
    for idx, frame in enumerate(frames):
        robot = frame.setdefault("robot", {})
        robot["gripper_action"] = build_gripper_action_frame_payload(recovery, idx)


def patch_gripper_action_h5(
    h5_path: Path,
    ga_matrix: np.ndarray,
) -> None:
    with h5py.File(h5_path, "r+") as hdf:
        for prefix in ("state", "action"):
            if prefix not in hdf:
                continue
            if "gripper_action" in hdf[prefix]:
                del hdf[f"{prefix}/gripper_action"]
            group = hdf[prefix].create_group("gripper_action")
            group.attrs["name"] = np.asarray(["left", "right"], dtype=object)
            group.attrs["category"] = np.asarray(["binary"], dtype=object)
            group.attrs["description"] = "1=open_or_opening, 0=close_or_holding"
            group.create_dataset("value", data=ga_matrix.astype(np.uint8))


def write_recovery_plot(
    plot_path: Path,
    recovery: dict[str, Any],
    ga_matrix: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    frame_axis = np.arange(ga_matrix.shape[0], dtype=np.int32)

    for idx, arm in enumerate(("left", "right")):
        ax = axes[idx]
        result = recovery[arm]
        ga = ga_matrix[:, idx]
        position = result.mean_position[: ga.shape[0]]
        smoothed = result.smoothed_position[: ga.shape[0]]

        ax.set_title(f"{arm} gripper")
        ax.fill_between(
            frame_axis,
            0.0,
            1.0,
            where=ga.astype(bool),
            color="#b7eb8f",
            alpha=0.18,
            transform=ax.get_xaxis_transform(),
            step="post",
            label="GA=1",
        )
        ax.fill_between(
            frame_axis,
            0.0,
            1.0,
            where=~ga.astype(bool),
            color="#ffccc7",
            alpha=0.18,
            transform=ax.get_xaxis_transform(),
            step="post",
            label="GA=0",
        )
        if position.size:
            ax.plot(frame_axis, position, color="#8c6d31", linewidth=1.0, label="mean joint")
        if smoothed.size:
            ax.plot(frame_axis, smoothed, color="#1677ff", linewidth=1.2, label="smoothed")
        for transition in result.transitions:
            color = "#1677ff" if transition["ga"] == GA_OPEN else "#d4380d"
            ax.axvline(int(transition["frame"]), color=color, linestyle="--", linewidth=1.0)
        ax.set_ylabel("joint value")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("frame")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def summarize_transitions(recovery: dict[str, Any]) -> str:
    parts: list[str] = []
    for arm in ("left", "right"):
        transitions = recovery[arm].transitions
        if not transitions:
            parts.append(f"{arm}: none")
            continue
        text = ", ".join(
            f"{item['event']}@{int(item['frame'])}" for item in transitions
        )
        parts.append(f"{arm}: {text}")
    return " | ".join(parts)


def process_episode(episode_dir: Path, write: bool, plot: bool) -> None:
    state_path = episode_dir / "state.json"
    state_payload = load_json(state_path)
    frames = state_payload.get("frames", [])
    joint_names, joint_positions = frames_to_joint_matrix(frames)
    if joint_positions.size == 0:
        print(f"{episode_dir.name}: skipped, no joint traces found")
        return

    recovery = recover_binary_gripper_actions(joint_names, joint_positions)
    ga_matrix = build_gripper_action_matrix(recovery)
    summary = build_gripper_action_summary(recovery)
    summary["recording_dir"] = str(episode_dir)
    summary["frame_count"] = int(ga_matrix.shape[0])

    left_counts = np.bincount(ga_matrix[:, 0], minlength=2).tolist() if ga_matrix.size else [0, 0]
    right_counts = np.bincount(ga_matrix[:, 1], minlength=2).tolist() if ga_matrix.size else [0, 0]
    print(
        f"{episode_dir.name}: "
        f"left(open={left_counts[GA_OPEN]}, close={left_counts[GA_CLOSED]}) "
        f"right(open={right_counts[GA_OPEN]}, close={right_counts[GA_CLOSED]}) "
        f"| {summarize_transitions(recovery)}"
    )

    if write:
        apply_gripper_action_to_frames(state_payload, recovery)
        dump_json(state_path, state_payload)
        dump_json(episode_dir / SUMMARY_FILENAME, summary)

        h5_path = episode_dir / "aligned_joints_all.h5"
        if h5_path.is_file():
            patch_gripper_action_h5(h5_path, ga_matrix)

    if plot:
        write_recovery_plot(episode_dir / PLOT_FILENAME, recovery, ga_matrix)


def main() -> int:
    args = parse_args()
    episode_dirs = collect_episode_dirs(args.inputs, args.pattern)
    if not episode_dirs:
        raise SystemExit("No episode directories found.")

    for episode_dir in episode_dirs:
        process_episode(episode_dir, write=args.write, plot=args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
