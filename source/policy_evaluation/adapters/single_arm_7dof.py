"""Generic single-arm 7DoF adapter for pi0-style policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from policy_evaluation.config import AdapterConfig


@dataclass
class DecodedAction:
    arm_positions: np.ndarray
    gripper_open: bool | None
    raw_action: np.ndarray


class SingleArm7DofAdapter:
    def __init__(self, config: AdapterConfig):
        self.config = config

    @property
    def joint_names(self) -> list[str]:
        return self.config.joint_names

    @property
    def gripper_joint_names(self) -> list[str]:
        return self.config.gripper_joint_names

    def build_state(self, joint_state: dict[str, float], gripper_state: dict[str, float] | None = None) -> np.ndarray:
        state = np.zeros((self.config.state_dim,), dtype=np.float32)
        for idx, joint_name in enumerate(self.config.joint_names[: self.config.action_arm_dim]):
            state[idx] = float(joint_state.get(joint_name, 0.0))
        if gripper_state and self.config.action_gripper_index < self.config.state_dim:
            state[self.config.action_gripper_index] = self._encode_gripper_state(gripper_state)
        return state

    def _encode_gripper_state(self, gripper_state: dict[str, float]) -> float:
        values = [float(gripper_state.get(name, 0.0)) for name in self.config.gripper_joint_names]
        if not values:
            return 0.0
        mean_value = float(np.mean(values))
        if self.config.gripper_state_mode == "raw_mean":
            return mean_value
        if self.config.gripper_state_mode == "binary_open":
            return 1.0 if self._normalized_open_value(mean_value) >= 0.5 else 0.0
        return self._normalized_open_value(mean_value)

    def _normalized_open_value(self, mean_value: float) -> float:
        open_positions = self.config.open_gripper_positions
        closed_positions = self.config.closed_gripper_positions
        if not open_positions or not closed_positions:
            return mean_value
        open_value = float(np.mean(open_positions))
        closed_value = float(np.mean(closed_positions))
        stroke = closed_value - open_value
        if abs(stroke) < 1e-6:
            return mean_value
        closed_ratio = (mean_value - open_value) / stroke
        return float(np.clip(1.0 - closed_ratio, 0.0, 1.0))

    def build_payload(self, observation: dict[str, Any], prompt: str) -> dict[str, Any]:
        head = _to_chw(observation["images"]["head"])
        left = _to_chw(observation["images"]["wrist"])
        right_image = observation["images"].get("right_wrist")
        if right_image is None:
            right = np.zeros_like(left)
        else:
            right = _to_chw(right_image)
        keys = self.config.image_keys
        return {
            "state": observation["state"],
            "images": {
                keys.get("head", "cam_head"): head,
                keys.get("wrist", "cam_left"): left,
                keys.get("right_wrist", "cam_right"): right,
            },
            "prompt": prompt,
        }

    def decode_action(self, action: np.ndarray) -> DecodedAction:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.config.action_arm_dim:
            raise ValueError(f"Action has {action.shape[0]} dims, expected at least {self.config.action_arm_dim}")
        arm_positions = action[: self.config.action_arm_dim].copy()
        gripper_open = None
        if action.shape[0] > self.config.action_gripper_index:
            gripper_open = bool(action[self.config.action_gripper_index] >= self.config.action_open_threshold)
        return DecodedAction(arm_positions=arm_positions, gripper_open=gripper_open, raw_action=action)


def _to_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.transpose(image, (2, 0, 1))
