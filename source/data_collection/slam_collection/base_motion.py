"""Smooth omni-base root-pose integration for Galbot SLAM collection."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class BaseCommand:
    forward: float = 0.0
    strafe: float = 0.0
    yaw: float = 0.0
    vertical: float = 0.0
    brake: bool = False
    quit: bool = False


@dataclass
class BaseMotionConfig:
    linear_speed: float = 0.4
    angular_speed: float = 0.7
    max_linear_accel: float = 0.8
    max_angular_accel: float = 1.2
    command_timeout_sec: float = 0.25


class OmniBaseMotion:
    """Integrate a planar omni-base command into a world root pose."""

    def __init__(
        self,
        initial_position,
        initial_quaternion,
        config: BaseMotionConfig,
    ) -> None:
        self.position = np.array(initial_position, dtype=np.float64)
        if self.position.shape != (3,):
            raise ValueError("initial_position must have shape (3,)")
        self.position[2] = float(initial_position[2])
        self.yaw = yaw_from_quat_wxyz(initial_quaternion)
        self.config = config
        self.local_velocity = np.zeros(2, dtype=np.float64)
        self.angular_velocity = 0.0
        self._last_command_time = time.monotonic()

    def mark_command_time(self) -> None:
        self._last_command_time = time.monotonic()

    def step(self, command: BaseCommand, dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(float(dt), 1e-6)
        active_command = command
        if time.monotonic() - self._last_command_time > self.config.command_timeout_sec:
            active_command = BaseCommand()

        if active_command.brake:
            target_local_velocity = np.zeros(2, dtype=np.float64)
            target_angular_velocity = 0.0
        else:
            target_local_velocity = np.array(
                [
                    _clamp(active_command.forward, -1.0, 1.0) * self.config.linear_speed,
                    _clamp(active_command.strafe, -1.0, 1.0) * self.config.linear_speed,
                ],
                dtype=np.float64,
            )
            target_angular_velocity = (
                _clamp(active_command.yaw, -1.0, 1.0) * self.config.angular_speed
            )

        self.local_velocity = _move_toward(
            self.local_velocity,
            target_local_velocity,
            self.config.max_linear_accel * dt,
        )
        self.angular_velocity = _move_toward_scalar(
            self.angular_velocity,
            target_angular_velocity,
            self.config.max_angular_accel * dt,
        )

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        world_velocity = np.array(
            [
                cos_yaw * self.local_velocity[0] - sin_yaw * self.local_velocity[1],
                sin_yaw * self.local_velocity[0] + cos_yaw * self.local_velocity[1],
            ],
            dtype=np.float64,
        )
        self.position[:2] += world_velocity * dt
        self.yaw = _wrap_pi(self.yaw + self.angular_velocity * dt)
        return self.position.copy(), quat_wxyz_from_yaw(self.yaw)


def yaw_from_quat_wxyz(quaternion) -> float:
    quat = np.array(quaternion, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError("quaternion must be [w, x, y, z]")
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return 0.0
    w, x, y, z = quat / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * float(yaw)
    return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float64)


def _move_toward(current: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    delta = target - current
    norm = float(np.linalg.norm(delta))
    if norm <= max_delta or norm < 1e-9:
        return target.copy()
    return current + delta / norm * max_delta


def _move_toward_scalar(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi
