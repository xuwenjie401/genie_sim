"""Galbot leg-joint vertical lift control for SLAM teleop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.base_utils.logger import logger
from slam_collection.base_motion import BaseCommand


@dataclass
class VerticalLiftConfig:
    enabled: bool = True
    joint_names: tuple[str, str, str] = ("leg_joint1", "leg_joint2", "leg_joint3")
    speed: float = 0.25
    max_accel: float = 0.8
    max_lower_delta: float = 0.4


class GalbotVerticalLiftController:
    """Move Galbot torso height by coordinating leg_joint1/2/3.

    The captured startup joint state is treated as the maximum-height state.
    Holding the up key moves delta back toward 0. Holding the down key moves
    delta negative, clipped by max_lower_delta.
    """

    def __init__(self, articulation, config: VerticalLiftConfig) -> None:
        self.articulation = articulation
        self.config = config
        self.joint_names = tuple(config.joint_names)
        self.joint_indices = self._resolve_joint_indices() if config.enabled else np.array([], dtype=np.int32)
        self.default_positions = self._capture_default_positions() if config.enabled else np.zeros(3, dtype=np.float64)
        self.delta = 0.0
        self.velocity = 0.0

    @property
    def enabled(self) -> bool:
        return self.config.enabled and len(self.joint_indices) == 3

    def step(self, command: BaseCommand, dt: float) -> tuple[list[str], np.ndarray] | None:
        if not self.enabled:
            return None

        dt = max(float(dt), 1e-6)
        if command.brake:
            target_velocity = 0.0
        else:
            target_velocity = float(np.clip(command.vertical, -1.0, 1.0)) * self.config.speed

        self.velocity = _move_toward_scalar(
            self.velocity,
            target_velocity,
            self.config.max_accel * dt,
        )
        self.delta = float(np.clip(self.delta + self.velocity * dt, -self.config.max_lower_delta, 0.0))
        if self.delta <= -self.config.max_lower_delta and self.velocity < 0.0:
            self.velocity = 0.0
        if self.delta >= 0.0 and self.velocity > 0.0:
            self.velocity = 0.0

        offsets = np.array([self.delta, 2.0 * self.delta, self.delta], dtype=np.float64)
        targets = self.default_positions + offsets
        targets = self._clip_to_joint_limits(targets)
        return list(self.joint_names), targets

    def _resolve_joint_indices(self) -> np.ndarray:
        indices = []
        for joint_name in self.joint_names:
            try:
                indices.append(int(self.articulation.get_dof_index(joint_name)))
            except Exception as exc:
                logger.warning(f"Vertical lift disabled; cannot resolve {joint_name}: {exc}")
                return np.array([], dtype=np.int32)
        return np.asarray(indices, dtype=np.int32)

    def _capture_default_positions(self) -> np.ndarray:
        if len(self.joint_indices) != 3:
            return np.zeros(3, dtype=np.float64)
        positions = self.articulation.get_joint_positions()
        defaults = np.asarray([float(positions[index]) for index in self.joint_indices], dtype=np.float64)
        logger.info(
            "Vertical lift max-height defaults: "
            + ", ".join(f"{name}={value:.4f}" for name, value in zip(self.joint_names, defaults))
        )
        return defaults

    def _clip_to_joint_limits(self, positions: np.ndarray) -> np.ndarray:
        try:
            lowers = self.articulation.dof_properties["lower"][self.joint_indices]
            uppers = self.articulation.dof_properties["upper"][self.joint_indices]
            return np.clip(positions, lowers, uppers)
        except Exception:
            return positions


def vertical_lift_config_from_task(task_info: dict) -> VerticalLiftConfig:
    vertical = task_info.get("slam_setting", {}).get("vertical", {})
    joint_names = vertical.get("joint_names", VerticalLiftConfig.joint_names)
    if len(joint_names) != 3:
        raise ValueError("slam_setting.vertical.joint_names must contain exactly 3 joint names")
    return VerticalLiftConfig(
        enabled=bool(vertical.get("enabled", True)),
        joint_names=tuple(str(name) for name in joint_names),
        speed=float(vertical.get("speed", VerticalLiftConfig.speed)),
        max_accel=float(vertical.get("max_accel", VerticalLiftConfig.max_accel)),
        max_lower_delta=abs(float(vertical.get("max_lower_delta", VerticalLiftConfig.max_lower_delta))),
    )


def _move_toward_scalar(current: float, target: float, max_delta: float) -> float:
    delta = float(target) - float(current)
    if abs(delta) <= max_delta:
        return float(target)
    return float(current) + float(np.sign(delta)) * max_delta
