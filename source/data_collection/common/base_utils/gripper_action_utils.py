from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


GA_OPEN = 1
GA_CLOSED = 0
LEFT_GRIPPER_ACTION_KEY = "left_gripper_action"
RIGHT_GRIPPER_ACTION_KEY = "right_gripper_action"
GA_DESCRIPTION = {
    GA_OPEN: "open_or_opening",
    GA_CLOSED: "close_or_holding",
}

_SMOOTH_WINDOW = 5
_MIN_ACTIVE_STROKE = 0.05
_STATIC_OPEN_THRESHOLD = 0.15
_STEP_THRESHOLD_RATIO = 0.007
_STEP_THRESHOLD_ABS = 0.002
_SWITCH_THRESHOLD_RATIO = 0.08
_SWITCH_THRESHOLD_ABS = 0.03
_LOOKAHEAD_FRAMES = 12
_REFRACTORY_FRAMES = 6


@dataclass(frozen=True)
class GripperArmRecovery:
    arm: str
    joint_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    mean_position: np.ndarray
    smoothed_position: np.ndarray
    closedness: np.ndarray
    binary_action: np.ndarray
    transitions: tuple[dict[str, Any], ...]
    active: bool
    initial_action: int
    open_anchor: float
    closed_anchor: float
    stroke: float
    step_threshold: float
    switch_threshold: float


def frames_to_joint_matrix(frames: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    joint_names: list[str] = []
    joint_rows: list[list[float]] = []

    for frame in frames:
        robot = frame.get("robot", {})
        joints = robot.get("joints", {})
        names = joints.get("joint_name", [])
        positions = joints.get("joint_position", [])
        if not names or not positions:
            continue
        if not joint_names:
            joint_names = [str(name) for name in names]
        if len(positions) != len(joint_names):
            continue
        joint_rows.append([float(value) for value in positions])

    if not joint_names or not joint_rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    return joint_names, np.asarray(joint_rows, dtype=np.float32)


def recover_binary_gripper_actions(
    joint_names: list[str],
    joint_positions: np.ndarray,
) -> dict[str, GripperArmRecovery]:
    if joint_positions.size == 0 or not joint_names:
        return {
            "left": _empty_recovery("left"),
            "right": _empty_recovery("right"),
        }

    return {
        arm: _recover_single_arm(arm, joint_names, joint_positions)
        for arm in ("left", "right")
    }


def build_gripper_action_matrix(recovery: dict[str, GripperArmRecovery]) -> np.ndarray:
    left = recovery["left"].binary_action
    right = recovery["right"].binary_action
    if left.size == 0 and right.size == 0:
        return np.zeros((0, 2), dtype=np.uint8)
    if left.size == 0:
        left = np.full_like(right, GA_OPEN, dtype=np.uint8)
    if right.size == 0:
        right = np.full_like(left, GA_OPEN, dtype=np.uint8)
    frame_count = min(left.shape[0], right.shape[0])
    return np.stack([left[:frame_count], right[:frame_count]], axis=1).astype(np.uint8)


def build_gripper_action_frame_payload(
    recovery: dict[str, GripperArmRecovery],
    frame_idx: int,
) -> dict[str, int]:
    payload: dict[str, int] = {}
    for arm, key in (
        ("left", LEFT_GRIPPER_ACTION_KEY),
        ("right", RIGHT_GRIPPER_ACTION_KEY),
    ):
        binary = recovery[arm].binary_action
        if binary.size == 0:
            payload[key] = GA_OPEN
            continue
        safe_idx = min(frame_idx, binary.shape[0] - 1)
        payload[key] = int(binary[safe_idx])
    return payload


def build_gripper_action_summary(
    recovery: dict[str, GripperArmRecovery],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm, result in recovery.items():
        arms[arm] = {
            "joint_names": list(result.joint_names),
            "joint_indices": list(result.joint_indices),
            "active": bool(result.active),
            "initial_action": int(result.initial_action),
            "initial_action_description": GA_DESCRIPTION[int(result.initial_action)],
            "open_anchor": float(result.open_anchor),
            "closed_anchor": float(result.closed_anchor),
            "stroke": float(result.stroke),
            "step_threshold": float(result.step_threshold),
            "switch_threshold": float(result.switch_threshold),
            "transitions": [
                {
                    "frame": int(item["frame"]),
                    "ga": int(item["ga"]),
                    "event": str(item["event"]),
                    "description": GA_DESCRIPTION[int(item["ga"])],
                }
                for item in result.transitions
            ],
        }
    return {
        "schema_version": 1,
        "source": "joint_state_trend_recovery",
        "ga_semantics": {
            "1": GA_DESCRIPTION[GA_OPEN],
            "0": GA_DESCRIPTION[GA_CLOSED],
        },
        "arms": arms,
    }


def load_gripper_action_matrix_from_frames(frames: list[dict[str, Any]]) -> np.ndarray | None:
    values: list[list[int]] = []
    for frame in frames:
        robot = frame.get("robot", {})
        if LEFT_GRIPPER_ACTION_KEY in robot and RIGHT_GRIPPER_ACTION_KEY in robot:
            values.append(
                [
                    int(robot[LEFT_GRIPPER_ACTION_KEY]),
                    int(robot[RIGHT_GRIPPER_ACTION_KEY]),
                ]
            )
            continue

        payload = robot.get("gripper_action")
        if isinstance(payload, dict) and "left" in payload and "right" in payload:
            values.append([int(payload["left"]), int(payload["right"])])
            continue
        return None
    if not values:
        return None
    return np.asarray(values, dtype=np.uint8)


def transitions_from_binary_action(binary_action: np.ndarray) -> tuple[dict[str, int | str], ...]:
    if binary_action.size == 0:
        return tuple()
    transitions: list[dict[str, int | str]] = []
    current = int(binary_action[0])
    for idx in range(1, binary_action.shape[0]):
        value = int(binary_action[idx])
        if value == current:
            continue
        current = value
        transitions.append(
            {
                "frame": idx,
                "ga": current,
                "event": "opening" if current == GA_OPEN else "closing",
            }
        )
    return tuple(transitions)


def _recover_single_arm(
    arm: str,
    joint_names: list[str],
    joint_positions: np.ndarray,
) -> GripperArmRecovery:
    indices = _find_gripper_joint_indices(arm, joint_names)
    if not indices:
        return _empty_recovery(arm, frame_count=joint_positions.shape[0])

    values = np.asarray(joint_positions[:, indices], dtype=np.float32)
    mean_position = values.mean(axis=1)
    smoothed = _smooth_trace(mean_position)
    stroke = float(np.max(mean_position) - np.min(mean_position))
    open_anchor, closed_anchor = _estimate_open_and_closed_anchor(mean_position)

    if stroke < _MIN_ACTIVE_STROKE:
        initial_action = GA_OPEN if float(mean_position[0]) <= _STATIC_OPEN_THRESHOLD else GA_CLOSED
        binary = np.full(smoothed.shape[0], initial_action, dtype=np.uint8)
        return GripperArmRecovery(
            arm=arm,
            joint_indices=tuple(indices),
            joint_names=tuple(joint_names[idx] for idx in indices),
            mean_position=mean_position,
            smoothed_position=smoothed,
            closedness=np.zeros_like(smoothed, dtype=np.float32),
            binary_action=binary,
            transitions=tuple(),
            active=False,
            initial_action=int(initial_action),
            open_anchor=float(open_anchor),
            closed_anchor=float(closed_anchor),
            stroke=stroke,
            step_threshold=0.0,
            switch_threshold=0.0,
        )

    closedness = _normalized_closedness(mean_position, open_anchor, closed_anchor)
    initial_action = (
        GA_OPEN
        if abs(float(mean_position[0]) - open_anchor) <= abs(float(mean_position[0]) - closed_anchor)
        else GA_CLOSED
    )
    step_threshold = max(_STEP_THRESHOLD_RATIO, _STEP_THRESHOLD_ABS / stroke)
    switch_threshold = max(_SWITCH_THRESHOLD_RATIO, _SWITCH_THRESHOLD_ABS / stroke)
    binary, transitions = _recover_binary_from_closedness(
        closedness=closedness,
        initial_action=initial_action,
        step_threshold=step_threshold,
        switch_threshold=switch_threshold,
    )
    return GripperArmRecovery(
        arm=arm,
        joint_indices=tuple(indices),
        joint_names=tuple(joint_names[idx] for idx in indices),
        mean_position=mean_position,
        smoothed_position=smoothed,
        closedness=closedness,
        binary_action=binary,
        transitions=transitions,
        active=True,
        initial_action=int(initial_action),
        open_anchor=float(open_anchor),
        closed_anchor=float(closed_anchor),
        stroke=stroke,
        step_threshold=float(step_threshold),
        switch_threshold=float(switch_threshold),
    )


def _recover_binary_from_closedness(
    closedness: np.ndarray,
    initial_action: int,
    step_threshold: float,
    switch_threshold: float,
) -> tuple[np.ndarray, tuple[dict[str, int | str], ...]]:
    binary = np.full(closedness.shape[0], int(initial_action), dtype=np.uint8)
    if closedness.shape[0] <= 1:
        return binary, tuple()

    delta = np.diff(closedness)
    event_candidates: list[tuple[int, str]] = []
    for direction, mask in (
        ("closing", delta >= step_threshold),
        ("opening", delta <= -step_threshold),
    ):
        for diff_idx in _run_starts(mask):
            start_frame = int(diff_idx + 1)
            future_frame = min(closedness.shape[0] - 1, start_frame + _LOOKAHEAD_FRAMES)
            cumulative = float(closedness[future_frame] - closedness[diff_idx])
            if direction == "closing" and cumulative >= switch_threshold:
                event_candidates.append((start_frame, direction))
            if direction == "opening" and cumulative <= -switch_threshold:
                event_candidates.append((start_frame, direction))

    event_candidates.sort(key=lambda item: item[0])

    current_action = int(initial_action)
    last_transition_frame = -_REFRACTORY_FRAMES
    accepted: list[dict[str, int | str]] = []
    for frame_idx, direction in event_candidates:
        if frame_idx - last_transition_frame < _REFRACTORY_FRAMES:
            continue
        next_action = GA_CLOSED if direction == "closing" else GA_OPEN
        if next_action == current_action:
            continue
        binary[frame_idx:] = next_action
        current_action = next_action
        last_transition_frame = frame_idx
        accepted.append(
            {
                "frame": int(frame_idx),
                "ga": int(next_action),
                "event": direction,
            }
        )
    return binary, tuple(accepted)


def _smooth_trace(values: np.ndarray, window: int = _SMOOTH_WINDOW) -> np.ndarray:
    if values.size <= 2 or window <= 1:
        return values.astype(np.float32, copy=True)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _normalized_closedness(
    values: np.ndarray,
    open_anchor: float,
    closed_anchor: float,
) -> np.ndarray:
    stroke = max(abs(closed_anchor - open_anchor), 1e-6)
    if closed_anchor >= open_anchor:
        normalized = (values - open_anchor) / stroke
    else:
        normalized = (open_anchor - values) / stroke
    return np.clip(normalized.astype(np.float32), 0.0, 1.0)


def _estimate_open_and_closed_anchor(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if abs(high - low) < 1e-6:
        return low, high
    low_score = abs(float(values[0]) - low) + abs(float(values[-1]) - low)
    high_score = abs(float(values[0]) - high) + abs(float(values[-1]) - high)
    if low_score <= high_score:
        return low, high
    return high, low


def _run_starts(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros((0,), dtype=np.int32)
    shifted = np.concatenate(([False], mask[:-1]))
    return np.flatnonzero(mask & ~shifted).astype(np.int32)


def _find_gripper_joint_indices(arm: str, joint_names: list[str]) -> list[int]:
    explicit_matches: list[int] = []
    fallback_matches: list[int] = []
    for idx, name in enumerate(joint_names):
        lower = name.lower()
        if "gripper" not in lower and "knuckle" not in lower:
            continue
        if arm == "right":
            if (
                lower.startswith("right_")
                or lower.startswith("fr_")
                or "right" in lower
                or "_right_" in lower
            ):
                explicit_matches.append(idx)
            elif lower.startswith("r_") or "_r_" in lower:
                fallback_matches.append(idx)
        else:
            if (
                lower.startswith("left_")
                or lower.startswith("fl_")
                or "left" in lower
                or "_left_" in lower
            ):
                explicit_matches.append(idx)
            elif lower.startswith("l_") or "_l_" in lower:
                fallback_matches.append(idx)
    if explicit_matches:
        return explicit_matches
    return fallback_matches


def _empty_recovery(arm: str, frame_count: int = 0) -> GripperArmRecovery:
    binary = np.full(frame_count, GA_OPEN, dtype=np.uint8)
    empty_float = np.zeros((frame_count,), dtype=np.float32)
    return GripperArmRecovery(
        arm=arm,
        joint_indices=tuple(),
        joint_names=tuple(),
        mean_position=empty_float,
        smoothed_position=empty_float,
        closedness=empty_float,
        binary_action=binary,
        transitions=tuple(),
        active=False,
        initial_action=GA_OPEN,
        open_anchor=0.0,
        closed_anchor=0.0,
        stroke=0.0,
        step_threshold=0.0,
        switch_threshold=0.0,
    )
