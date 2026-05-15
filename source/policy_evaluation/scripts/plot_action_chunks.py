#!/usr/bin/env python3
"""Plot and inspect pi0-style policy action chunks for evaluation episodes."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass
class ActionChunkRecord:
    record_index: int
    step: int
    time: float
    latency_sec: float | None
    prompt: str
    state: np.ndarray
    actions: np.ndarray
    image_shapes: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_len(self) -> int:
        return int(self.actions.shape[0]) if self.actions.ndim == 2 else 0

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1]) if self.actions.ndim == 2 else 0


@dataclass
class CheckConfig:
    arm_dim: int
    gripper_index: int
    open_threshold: float
    latency_warn_sec: float
    boundary_jump_warn: float
    state_action_warn: float
    gripper_soft_min: float
    gripper_soft_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize and check policy_io.jsonl action chunks saved by "
            "source/policy_evaluation episodes."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Path to a policy_io.jsonl file, one episode directory containing "
            "policy_io.jsonl, or a run directory containing episodes/*/policy_io.jsonl."
        ),
    )
    parser.add_argument("--output-dir", help="Output directory. Defaults next to the input episode/run.")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "overview", "samples", "chunks", "checks"],
        help="'chunks' is kept as an alias for 'samples'.",
    )
    parser.add_argument("--start-step", type=int, help="Only plot/check chunks starting at or after this step.")
    parser.add_argument("--end-step", type=int, help="Only plot/check chunks starting at or before this step.")
    parser.add_argument("--chunk-stride", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-limit", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--sample-count", type=int, default=40, help="Number of detailed sample frames to write.")
    parser.add_argument("--arm-dim", type=int, default=7, help="Number of arm action dimensions to plot/check.")
    parser.add_argument("--gripper-index", type=int, default=7, help="Action index used as gripper command.")
    parser.add_argument("--open-threshold", type=float, default=0.5, help="Gripper action >= threshold means open.")
    parser.add_argument("--latency-warn-sec", type=float, default=1.0, help="Warn when policy latency exceeds this.")
    parser.add_argument(
        "--boundary-jump-warn",
        type=float,
        default=0.35,
        help="Warn when max arm-action jump across chunk boundaries exceeds this.",
    )
    parser.add_argument(
        "--state-action-warn",
        type=float,
        default=0.35,
        help="Warn when max difference between state arm joints and first chunk action exceeds this.",
    )
    parser.add_argument(
        "--gripper-soft-min",
        type=float,
        default=-0.05,
        help="Warn when gripper action is below this soft range.",
    )
    parser.add_argument(
        "--gripper-soft-max",
        type=float,
        default=1.05,
        help="Warn when gripper action is above this soft range.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output figure DPI.")
    return parser.parse_args()


def resolve_policy_logs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    direct = input_path / "policy_io.jsonl"
    if direct.exists():
        return [direct]

    episode_root = input_path / "episodes"
    if episode_root.exists():
        logs = sorted(episode_root.glob("*/policy_io.jsonl"))
        if logs:
            return logs

    logs = sorted(input_path.glob("*/policy_io.jsonl"))
    if logs:
        return logs

    raise FileNotFoundError(f"No policy_io.jsonl found under: {input_path}")


def default_output_root(input_path: Path, policy_logs: list[Path]) -> Path:
    if len(policy_logs) == 1:
        return policy_logs[0].parent / "action_chunk_analysis"
    return input_path / "action_chunk_analysis"


def load_records(path: Path) -> list[ActionChunkRecord]:
    records: list[ActionChunkRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            actions = np.asarray(payload.get("actions", []), dtype=np.float32)
            if actions.ndim == 1 and actions.size > 0:
                actions = actions.reshape(1, -1)
            elif actions.ndim != 2:
                actions = np.zeros((0, 0), dtype=np.float32)
            records.append(
                ActionChunkRecord(
                    record_index=line_index,
                    step=int(payload.get("step", line_index)),
                    time=float(payload.get("time", np.nan)),
                    latency_sec=_optional_float(payload.get("latency_sec")),
                    prompt=str(payload.get("prompt", "")),
                    state=np.asarray(payload.get("state", []), dtype=np.float32),
                    actions=actions,
                    image_shapes=dict(payload.get("image_shapes", {})),
                )
            )
    if not records:
        raise ValueError(f"No policy records found in: {path}")
    return records


def filter_records(
    records: list[ActionChunkRecord],
    start_step: int | None,
    end_step: int | None,
) -> list[ActionChunkRecord]:
    filtered = []
    for record in records:
        if start_step is not None and record.step < start_step:
            continue
        if end_step is not None and record.step > end_step:
            continue
        filtered.append(record)
    if not filtered:
        raise ValueError("No chunks remain after step filtering.")
    return filtered


def build_checks(records: list[ActionChunkRecord], cfg: CheckConfig) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    chunk_lengths = [record.chunk_len for record in records]
    action_dims = [record.action_dim for record in records]
    latency_values = [record.latency_sec for record in records if record.latency_sec is not None]

    for idx, record in enumerate(records):
        if record.actions.size == 0:
            warnings.append(_warning(record, "empty_actions", "Action chunk is empty."))
            continue
        if record.action_dim <= cfg.gripper_index:
            warnings.append(
                _warning(
                    record,
                    "missing_gripper_dim",
                    f"Action dim {record.action_dim} does not include gripper index {cfg.gripper_index}.",
                )
            )
        if record.action_dim < cfg.arm_dim:
            warnings.append(
                _warning(record, "short_arm_dim", f"Action dim {record.action_dim} < arm_dim {cfg.arm_dim}.")
            )
        if not np.all(np.isfinite(record.actions)):
            warnings.append(_warning(record, "nonfinite_actions", "Action chunk contains NaN or inf."))
        if record.state.size and not np.all(np.isfinite(record.state)):
            warnings.append(_warning(record, "nonfinite_state", "State contains NaN or inf."))
        if record.latency_sec is not None and record.latency_sec > cfg.latency_warn_sec:
            warnings.append(
                _warning(
                    record,
                    "high_latency",
                    f"Policy latency {record.latency_sec:.3f}s > {cfg.latency_warn_sec:.3f}s.",
                    latency_sec=record.latency_sec,
                )
            )

        arm_dim = min(cfg.arm_dim, record.action_dim, record.state.size)
        if arm_dim > 0:
            delta = np.abs(record.actions[0, :arm_dim] - record.state[:arm_dim])
            max_delta = float(np.max(delta))
            if max_delta > cfg.state_action_warn:
                warnings.append(
                    _warning(
                        record,
                        "large_state_to_first_action_delta",
                        f"Max |state-action[0]| {max_delta:.4f} > {cfg.state_action_warn:.4f}.",
                        max_delta=max_delta,
                    )
                )

        gripper_values = gripper_action_values(record, cfg.gripper_index)
        if gripper_values.size:
            gmin = float(np.min(gripper_values))
            gmax = float(np.max(gripper_values))
            if gmin < cfg.gripper_soft_min or gmax > cfg.gripper_soft_max:
                warnings.append(
                    _warning(
                        record,
                        "gripper_value_outside_soft_range",
                        (
                            f"Gripper action range [{gmin:.4f}, {gmax:.4f}] outside "
                            f"[{cfg.gripper_soft_min:.4f}, {cfg.gripper_soft_max:.4f}]."
                        ),
                        gripper_min=gmin,
                        gripper_max=gmax,
                    )
                )
            transitions = count_binary_transitions(gripper_values >= cfg.open_threshold)
            if transitions > 1:
                warnings.append(
                    _warning(
                        record,
                        "multiple_gripper_transitions_in_chunk",
                        f"Gripper command crosses threshold {transitions} times inside one chunk.",
                        transitions=transitions,
                    )
                )

        if idx > 0:
            prev = records[idx - 1]
            expected_step = prev.step + prev.chunk_len
            if record.step != expected_step:
                warnings.append(
                    _warning(
                        record,
                        "chunk_step_gap",
                        f"Chunk step {record.step} != previous step {prev.step} + chunk_len {prev.chunk_len}.",
                        expected_step=expected_step,
                        previous_step=prev.step,
                    )
                )

            boundary_dim = min(cfg.arm_dim, prev.action_dim, record.action_dim)
            if boundary_dim > 0 and prev.chunk_len > 0 and record.chunk_len > 0:
                jump = np.abs(record.actions[0, :boundary_dim] - prev.actions[-1, :boundary_dim])
                max_jump = float(np.max(jump))
                if max_jump > cfg.boundary_jump_warn:
                    warnings.append(
                        _warning(
                            record,
                            "large_chunk_boundary_jump",
                            f"Max action jump across chunk boundary {max_jump:.4f} > {cfg.boundary_jump_warn:.4f}.",
                            max_jump=max_jump,
                            previous_step=prev.step,
                        )
                    )

    executed = reconstruct_executed_actions(records)
    gripper_all = executed["gripper"]
    return {
        "records": len(records),
        "step_start": records[0].step,
        "step_end": records[-1].step,
        "chunk_lengths": {
            "min": int(min(chunk_lengths)),
            "max": int(max(chunk_lengths)),
            "unique": sorted({int(value) for value in chunk_lengths}),
        },
        "action_dims": {
            "min": int(min(action_dims)),
            "max": int(max(action_dims)),
            "unique": sorted({int(value) for value in action_dims}),
        },
        "latency_sec": _stats(latency_values),
        "executed_actions": {
            "count": int(executed["steps"].size),
            "step_start": int(executed["steps"][0]) if executed["steps"].size else None,
            "step_end": int(executed["steps"][-1]) if executed["steps"].size else None,
            "gripper_open_count": int(np.sum(gripper_all >= cfg.open_threshold)) if gripper_all.size else 0,
            "gripper_close_count": int(np.sum(gripper_all < cfg.open_threshold)) if gripper_all.size else 0,
        },
        "warnings": warnings,
    }


def reconstruct_executed_actions(records: list[ActionChunkRecord]) -> dict[str, np.ndarray]:
    steps = []
    actions = []
    chunk_ids = []
    for chunk_idx, record in enumerate(records):
        if record.chunk_len <= 0:
            continue
        for offset, action in enumerate(record.actions):
            steps.append(record.step + offset)
            actions.append(action)
            chunk_ids.append(chunk_idx)
    if not actions:
        return {
            "steps": np.asarray([], dtype=np.int32),
            "actions": np.zeros((0, 0), dtype=np.float32),
            "chunk_ids": np.asarray([], dtype=np.int32),
            "gripper": np.asarray([], dtype=np.float32),
        }
    action_array = np.asarray(actions, dtype=np.float32)
    gripper = action_array[:, 7] if action_array.shape[1] > 7 else np.asarray([], dtype=np.float32)
    return {
        "steps": np.asarray(steps, dtype=np.int32),
        "actions": action_array,
        "chunk_ids": np.asarray(chunk_ids, dtype=np.int32),
        "gripper": gripper,
    }


def draw_overview(
    records: list[ActionChunkRecord],
    checks: dict[str, Any],
    output_path: Path,
    cfg: CheckConfig,
    dpi: int,
) -> None:
    executed = reconstruct_executed_actions(records)
    steps = executed["steps"]
    actions = executed["actions"]
    chunk_ids = executed["chunk_ids"]
    chunk_count = max(len(records), 1)

    figure = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = figure.add_gridspec(nrows=3, ncols=1, height_ratios=[2.0, 1.2, 1.1])
    arm_axis = figure.add_subplot(grid[0, 0])
    gripper_axis = figure.add_subplot(grid[1, 0])
    checks_axis = figure.add_subplot(grid[2, 0])

    arm_axis.grid(True, linestyle="--", alpha=0.25)
    if actions.size:
        arm_dim = min(cfg.arm_dim, actions.shape[1])
        markers = ["o", "s", "^", "v", "D", "P", "X"]
        scatter_handle = None
        for dim in range(arm_dim):
            scatter_handle = arm_axis.scatter(
                steps,
                actions[:, dim],
                c=chunk_ids,
                cmap="turbo",
                vmin=0,
                vmax=max(chunk_count - 1, 1),
                s=9,
                alpha=0.72,
                marker=markers[dim % len(markers)],
                linewidths=0.0,
                label=f"joint[{dim}]",
            )
        for record in records:
            arm_axis.axvline(record.step, color="#999999", alpha=0.12, linewidth=0.8)
        if scatter_handle is not None:
            colorbar = figure.colorbar(scatter_handle, ax=arm_axis, pad=0.01, aspect=30)
            colorbar.set_label("chunk index", fontsize=9)
        arm_axis.legend(ncol=4, fontsize=8, loc="upper right")
    else:
        arm_axis.text(0.5, 0.5, "no actions", transform=arm_axis.transAxes, ha="center", va="center")
    arm_axis.set_title("Executed Arm Actions (scatter, color = chunk)", fontsize=12)
    arm_axis.set_xlabel("policy step")
    arm_axis.set_ylabel("joint target")

    gripper_axis.grid(True, linestyle="--", alpha=0.25)
    gripper = executed["gripper"]
    if gripper.size:
        scatter_handle = gripper_axis.scatter(
            steps,
            gripper,
            c=chunk_ids,
            cmap="turbo",
            vmin=0,
            vmax=max(chunk_count - 1, 1),
            s=13,
            alpha=0.8,
            linewidths=0.0,
        )
        gripper_axis.axhline(cfg.open_threshold, color="#333333", linestyle="--", linewidth=1.0)
        for record in records:
            gripper_axis.axvline(record.step, color="#999999", alpha=0.12, linewidth=0.8)
        colorbar = figure.colorbar(scatter_handle, ax=gripper_axis, pad=0.01, aspect=30)
        colorbar.set_label("chunk index", fontsize=9)
        gripper_axis.set_ylim(min(-0.08, float(np.min(gripper)) - 0.05), max(1.08, float(np.max(gripper)) + 0.05))
    else:
        gripper_axis.text(0.5, 0.5, "no gripper dimension", transform=gripper_axis.transAxes, ha="center", va="center")
    gripper_axis.set_title("Executed Gripper Actions (scatter, color = chunk)", fontsize=12)
    gripper_axis.set_xlabel("policy step")
    gripper_axis.set_ylabel(f"action[{cfg.gripper_index}]")

    checks_axis.axis("off")
    prompt = records[0].prompt if records else ""
    summary_lines = [
        f"policy_io: {checks.get('policy_io', '(unknown)')}",
        f"prompt: {textwrap.shorten(prompt, width=130, placeholder='...') if prompt else '(empty)'}",
        f"chunks: {checks['records']}  action_dims: {checks['action_dims']['unique']}  chunk_lengths: {checks['chunk_lengths']['unique']}",
        (
            f"executed actions: {checks['executed_actions']['count']}  "
            f"open/close: {checks['executed_actions']['gripper_open_count']}/"
            f"{checks['executed_actions']['gripper_close_count']}"
        ),
        f"warnings: {len(checks['warnings'])}",
    ]
    for warning in checks["warnings"][:12]:
        summary_lines.append(
            f"- step {warning.get('step')}: {warning.get('type')}  {warning.get('message')}"
        )
    if len(checks["warnings"]) > 12:
        summary_lines.append(f"- ... {len(checks['warnings']) - 12} more warnings in checks.json")
    checks_axis.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        fontsize=10,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    figure.suptitle("Policy Evaluation Action Chunk Overview", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def draw_arm_chunk_axis(axis, record: ActionChunkRecord, cfg: CheckConfig) -> None:
    axis.grid(True, linestyle="--", alpha=0.25)
    axis.set_title("Arm Action Chunk" if record.record_index == 0 else "", fontsize=11)
    axis.set_xlabel("chunk frame")
    axis.set_ylabel("joint target")
    if record.actions.size == 0:
        axis.text(0.5, 0.5, "empty actions", transform=axis.transAxes, ha="center", va="center")
        return
    arm_dim = min(cfg.arm_dim, record.action_dim)
    chunk_x = np.arange(record.chunk_len)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(arm_dim, 1)))
    for dim in range(arm_dim):
        axis.scatter(chunk_x, record.actions[:, dim], s=18, color=colors[dim], label=f"j{dim}", alpha=0.85)
        if record.state.size > dim:
            axis.scatter([0], [record.state[dim]], color=colors[dim], s=20, edgecolors="black", linewidths=0.3)
    axis.legend(ncol=4, fontsize=7, loc="best")


def draw_gripper_chunk_axis(axis, record: ActionChunkRecord, cfg: CheckConfig) -> None:
    axis.grid(True, linestyle="--", alpha=0.25)
    axis.set_title("Gripper Action Chunk" if record.record_index == 0 else "", fontsize=11)
    axis.set_xlabel("chunk frame")
    axis.set_ylabel(f"action[{cfg.gripper_index}]")
    gripper = gripper_action_values(record, cfg.gripper_index)
    if gripper.size == 0:
        axis.text(0.5, 0.5, "missing gripper dim", transform=axis.transAxes, ha="center", va="center")
        return
    chunk_x = np.arange(gripper.size)
    axis.scatter(chunk_x, gripper, color="#d62728", s=20, alpha=0.9)
    axis.axhline(cfg.open_threshold, color="#333333", linestyle="--", linewidth=1.0)
    axis.fill_between(
        chunk_x,
        cfg.open_threshold,
        np.maximum(gripper, cfg.open_threshold),
        where=gripper >= cfg.open_threshold,
        color="#2ca02c",
        alpha=0.15,
        step=None,
    )
    axis.fill_between(
        chunk_x,
        np.minimum(gripper, cfg.open_threshold),
        cfg.open_threshold,
        where=gripper < cfg.open_threshold,
        color="#d62728",
        alpha=0.12,
        step=None,
    )
    axis.set_ylim(min(-0.08, float(np.min(gripper)) - 0.05), max(1.08, float(np.max(gripper)) + 0.05))


def write_sample_frames(
    records: list[ActionChunkRecord],
    policy_log: Path,
    output_dir: Path,
    cfg: CheckConfig,
    sample_count: int,
    dpi: int,
) -> list[str]:
    sample_dir = output_dir / "sample_frames"
    sample_dir.mkdir(parents=True, exist_ok=True)
    selected = select_sample_records(records, sample_count)
    max_policy_step = max((record.step + max(record.chunk_len - 1, 0) for record in records), default=0)
    video_paths = {
        "head": policy_log.parent / "video" / "head.mp4",
        "wrist": policy_log.parent / "video" / "wrist.mp4",
        "observer": policy_log.parent / "video" / "observer.mp4",
    }

    written: list[str] = []
    for sample_index, record in enumerate(selected, start=1):
        frame_path = sample_dir / f"sample_{sample_index:03d}_step_{record.step:06d}.png"
        figure = build_sample_frame(
            record=record,
            sample_index=sample_index,
            sample_total=len(selected),
            video_paths=video_paths,
            max_policy_step=max_policy_step,
            cfg=cfg,
        )
        figure.savefig(frame_path, dpi=dpi)
        plt.close(figure)
        written.append(str(frame_path))
    return written


def select_sample_records(records: list[ActionChunkRecord], sample_count: int) -> list[ActionChunkRecord]:
    if sample_count <= 0 or sample_count >= len(records):
        return list(records)
    steps = np.asarray([record.step for record in records], dtype=np.int32)
    target_steps = np.linspace(int(steps[0]), int(steps[-1]), sample_count)
    indices = []
    for target in target_steps:
        insert_at = int(np.searchsorted(steps, target))
        candidates = []
        if insert_at < len(steps):
            candidates.append(insert_at)
        if insert_at > 0:
            candidates.append(insert_at - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: abs(float(steps[idx]) - float(target)))
        if not indices or indices[-1] != best:
            indices.append(best)
    return [records[index] for index in indices]


def build_sample_frame(
    record: ActionChunkRecord,
    sample_index: int,
    sample_total: int,
    video_paths: dict[str, Path],
    max_policy_step: int,
    cfg: CheckConfig,
):
    figure = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = figure.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.1, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
    )

    head_axis = figure.add_subplot(grid[0, 0])
    wrist_axis = figure.add_subplot(grid[0, 1])
    observer_axis = figure.add_subplot(grid[0, 2])
    info_axis = figure.add_subplot(grid[1, 0])
    arm_axis = figure.add_subplot(grid[1, 1])
    gripper_axis = figure.add_subplot(grid[1, 2])

    draw_image_axis(
        head_axis,
        read_video_frame(video_paths["head"], record.step, max_policy_step),
        "cam_head input (video approx.)",
        f"missing {video_paths['head'].name}",
    )
    draw_image_axis(
        wrist_axis,
        read_video_frame(video_paths["wrist"], record.step, max_policy_step),
        "cam_left input (video approx.)",
        f"missing {video_paths['wrist'].name}",
    )
    draw_image_axis(
        observer_axis,
        read_video_frame(video_paths["observer"], record.step, max_policy_step),
        "observer (video approx.)",
        f"missing {video_paths['observer'].name}",
    )
    draw_sample_info_axis(info_axis, record, cfg)
    draw_arm_chunk_axis(arm_axis, record, cfg)
    draw_gripper_chunk_axis(gripper_axis, record, cfg)

    figure.suptitle(
        (
            f"Action Chunk Sample {sample_index}/{sample_total}  "
            f"policy_step={record.step}  chunk={record.chunk_len}x{record.action_dim}\n"
            f"{textwrap.shorten(record.prompt, width=150, placeholder='...') if record.prompt else '(empty prompt)'}"
        ),
        fontsize=14,
    )
    return figure


def draw_image_axis(axis, image: np.ndarray | None, title: str, fallback: str) -> None:
    axis.set_title(title, fontsize=11)
    axis.axis("off")
    if image is None:
        axis.text(0.5, 0.5, fallback, ha="center", va="center", fontsize=10, color="#666666")
        return
    axis.imshow(image)


def draw_sample_info_axis(axis, record: ActionChunkRecord, cfg: CheckConfig) -> None:
    axis.axis("off")
    gripper_state = float(record.state[cfg.gripper_index]) if record.state.size > cfg.gripper_index else float("nan")
    gripper_values = gripper_action_values(record, cfg.gripper_index)
    lines = [
        f"record_index: {record.record_index}",
        f"policy_step: {record.step}",
        f"chunk_shape: {record.chunk_len} x {record.action_dim}",
        f"gripper_state(state[{cfg.gripper_index}]): {gripper_state:.5f}",
        "image note: policy_io stores image_shapes only;",
        "            displayed frames are sampled from saved mp4.",
    ]
    if gripper_values.size:
        open_mask = gripper_values >= cfg.open_threshold
        lines.extend(
            [
                f"gripper_action_min: {float(np.min(gripper_values)):.5f}",
                f"gripper_action_max: {float(np.max(gripper_values)):.5f}",
                f"gripper_open/close: {int(np.sum(open_mask))}/{int(np.sum(~open_mask))}",
                f"gripper_transitions: {count_binary_transitions(open_mask)}",
            ]
        )
    else:
        lines.append("gripper_action: missing")
    if record.latency_sec is not None:
        lines.append(f"latency_sec: {record.latency_sec:.5f}")
    if record.image_shapes:
        lines.append("image_shapes:")
        for key, shape in sorted(record.image_shapes.items()):
            lines.append(f"  {key}: {shape}")
    axis.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10, family="monospace")


def read_video_frame(video_path: Path, policy_step: int, max_policy_step: int) -> np.ndarray | None:
    if not video_path.exists():
        return None
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None
        if max_policy_step > 0 and frame_count >= max_policy_step + 1:
            frame_index = min(max(int(policy_step), 0), frame_count - 1)
        elif max_policy_step > 0:
            ratio = float(np.clip(policy_step / max_policy_step, 0.0, 1.0))
            frame_index = int(round(ratio * (frame_count - 1)))
        else:
            frame_index = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def write_checks(path: Path, checks: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(checks, file, indent=2, ensure_ascii=False)


def process_policy_log(policy_log: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(policy_log)
    records = filter_records(records, args.start_step, args.end_step)
    cfg = CheckConfig(
        arm_dim=args.arm_dim,
        gripper_index=args.gripper_index,
        open_threshold=args.open_threshold,
        latency_warn_sec=args.latency_warn_sec,
        boundary_jump_warn=args.boundary_jump_warn,
        state_action_warn=args.state_action_warn,
        gripper_soft_min=args.gripper_soft_min,
        gripper_soft_max=args.gripper_soft_max,
    )
    checks = build_checks(records, cfg)
    checks["policy_io"] = str(policy_log)
    checks["output_dir"] = str(output_dir)

    if args.mode in ("all", "overview"):
        draw_overview(records, checks, output_dir / "overview.png", cfg, args.dpi)
    if args.mode in ("all", "samples", "chunks"):
        sample_frames = write_sample_frames(
            records,
            policy_log,
            output_dir,
            cfg,
            sample_count=args.sample_count,
            dpi=args.dpi,
        )
        checks["sample_frames"] = {
            "count": len(sample_frames),
            "directory": str(output_dir / "sample_frames"),
            "files": sample_frames,
        }
    if args.mode in ("all", "checks", "overview", "samples", "chunks"):
        write_checks(output_dir / "checks.json", checks)
    return checks


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    policy_logs = resolve_policy_logs(input_path)
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_root(input_path, policy_logs)

    index = []
    for policy_log in policy_logs:
        if len(policy_logs) == 1:
            output_dir = output_root
        else:
            output_dir = output_root / policy_log.parent.name
        checks = process_policy_log(policy_log, output_dir, args)
        index.append(
            {
                "policy_io": str(policy_log),
                "output_dir": str(output_dir),
                "records": checks["records"],
                "warnings": len(checks["warnings"]),
            }
        )
        print(
            f"[action_chunk_plot] {policy_log} -> {output_dir} "
            f"records={checks['records']} warnings={len(checks['warnings'])}"
        )

    if len(index) > 1:
        output_root.mkdir(parents=True, exist_ok=True)
        with (output_root / "index.json").open("w", encoding="utf-8") as file:
            json.dump(index, file, indent=2, ensure_ascii=False)
        print(f"[action_chunk_plot] wrote index: {output_root / 'index.json'}")


def gripper_action_values(record: ActionChunkRecord, gripper_index: int) -> np.ndarray:
    if record.actions.ndim != 2 or record.action_dim <= gripper_index:
        return np.asarray([], dtype=np.float32)
    return record.actions[:, gripper_index].astype(np.float32)


def count_binary_transitions(values: np.ndarray) -> int:
    if values.size <= 1:
        return 0
    return int(np.sum(values[1:] != values[:-1]))


def _warning(record: ActionChunkRecord, warning_type: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "type": warning_type,
        "message": message,
        "record_index": record.record_index,
        "step": record.step,
        "chunk_len": record.chunk_len,
        "action_dim": record.action_dim,
    }
    payload.update(extra)
    return payload


def _stats(values: list[float | None]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_optional(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
