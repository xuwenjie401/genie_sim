"""Left Galbot gripper state machine for single-process SLAM teleop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.base_utils.logger import logger


@dataclass
class GripperControlConfig:
    close_max_force: float = 0.2
    hold_stiffness: float = 10000.0
    hold_max_force: float = 10.0
    open_max_force: float = 100.0
    open_tolerance: float = 0.01
    settle_interval_sec: float = 0.1
    settle_position_delta: float = 0.01
    settle_max_samples: int = 50


class LeftGripperStateMachine:
    """Velocity close/open with latched hold for the fragile Galbot gripper."""

    def __init__(self, articulation, robot_cfg, physics_hz: float, config: GripperControlConfig) -> None:
        self.articulation = articulation
        self.robot_cfg = robot_cfg
        self.physics_hz = max(float(physics_hz), 1.0)
        self.config = config
        self.arm = "left"
        self.mode = "idle"
        self.command = ""
        self.last_sample_position = None
        self.sample_count = 0
        self.frame_count = 0
        self.hold_positions = None
        self._drive_cache = {}

    @property
    def controlled_joint_names(self) -> list[str]:
        return list(self.robot_cfg.finger_names.get(self.arm, []))

    def command_open(self) -> None:
        self._start_command("open")
        self._set_gripper_drive(stiffness=0.0, max_force=self.config.open_max_force)

    def command_close(self) -> None:
        self._start_command("close")
        self._set_gripper_drive(stiffness=0.0, max_force=self.config.close_max_force)

    def is_holding(self) -> bool:
        return self.command == "close" and self.mode == "holding"

    def step(self) -> None:
        if self.mode == "idle":
            return
        if self.mode == "opening":
            if self._is_open():
                self._apply_stop()
                self.mode = "idle"
                logger.info("Left gripper opened")
                return
            self._apply_open_velocity()
            return
        if self.mode == "closing":
            self._apply_close_velocity()
            if self._is_close_motion_settled():
                self._latch_hold_positions()
                self._apply_stop()
                self.mode = "holding"
                logger.info("Left gripper closed and holding")
            return
        if self.mode == "holding":
            self._hold_current_pose()

    def _start_command(self, command: str) -> None:
        if self.command == command and self.mode in {"opening", "closing", "holding"}:
            return
        self.command = command
        self.mode = "opening" if command == "open" else "closing"
        self.last_sample_position = None
        self.sample_count = 0
        self.frame_count = 0
        self.hold_positions = None
        logger.info(f"Left gripper command: {command}")

    def _joint_indices(self) -> list[int]:
        indices = []
        for joint_name in self.controlled_joint_names:
            try:
                indices.append(int(self.articulation.get_dof_index(joint_name)))
            except Exception as exc:
                logger.warning(f"Skip unavailable gripper joint {joint_name}: {exc}")
        return indices

    def _open_positions(self, count: int) -> np.ndarray:
        return _fit_vector(self.robot_cfg.opened_positions.get(self.arm, []), count, default=0.0)

    def _closed_velocities(self, count: int) -> np.ndarray:
        return _fit_vector(self.robot_cfg.closed_velocities.get(self.arm, []), count, default=80.0)

    def _apply_open_velocity(self) -> None:
        indices = self._joint_indices()
        if not indices:
            return
        current = self.articulation.get_joint_positions(joint_indices=indices)
        if current is None or len(current) != len(indices):
            return
        current = np.asarray(current, dtype=np.float64)
        opened = self._open_positions(len(indices))
        speeds = np.minimum(np.abs(self._closed_velocities(len(indices))), 40.0)
        speeds = np.where(speeds > 0.0, speeds, 40.0)
        velocities = np.sign(opened - current) * speeds
        velocities[np.abs(opened - current) <= 0.0025] = 0.0
        self._apply_velocity(indices, velocities)

    def _apply_close_velocity(self) -> None:
        indices = self._joint_indices()
        if indices:
            self._apply_velocity(indices, self._closed_velocities(len(indices)))

    def _apply_stop(self) -> None:
        indices = self._joint_indices()
        if indices:
            self._apply_velocity(indices, np.zeros(len(indices), dtype=np.float64))

    def _apply_velocity(self, indices: list[int], velocities: np.ndarray) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        target_velocities = [None] * len(self.articulation.dof_names)
        for joint_index, velocity in zip(indices, velocities):
            target_velocities[int(joint_index)] = float(velocity)
        self.articulation.apply_action(ArticulationAction(joint_velocities=target_velocities))

    def _hold_current_pose(self) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        indices = self._joint_indices()
        if not indices:
            return
        self._set_gripper_drive(
            stiffness=self.config.hold_stiffness,
            max_force=self.config.hold_max_force,
        )
        hold_positions = self.hold_positions
        if hold_positions is None:
            hold_positions = self._latch_hold_positions()
        if hold_positions is None or len(hold_positions) != len(indices):
            return
        target_positions = [None] * len(self.articulation.dof_names)
        for joint_index, position in zip(indices, hold_positions):
            target_positions[int(joint_index)] = float(position)
        self.articulation.apply_action(ArticulationAction(joint_positions=target_positions))

    def _latch_hold_positions(self) -> np.ndarray | None:
        indices = self._joint_indices()
        if not indices:
            return None
        current = self.articulation.get_joint_positions(joint_indices=indices)
        if current is None or len(current) != len(indices):
            return None
        self.hold_positions = np.asarray(current, dtype=np.float64).copy()
        return self.hold_positions

    def _is_open(self) -> bool:
        indices = self._joint_indices()
        if not indices:
            return True
        current = self.articulation.get_joint_positions(joint_indices=indices)
        if current is None or len(current) != len(indices):
            return False
        target = self._open_positions(len(indices))
        return bool(np.all(np.abs(np.asarray(current, dtype=np.float64) - target) <= self.config.open_tolerance))

    def _is_close_motion_settled(self) -> bool:
        indices = self._joint_indices()
        if not indices:
            return True
        sample_interval = max(1, int(round(self.config.settle_interval_sec * self.physics_hz)))
        self.frame_count += 1
        if self.frame_count % sample_interval != 0:
            return False
        current = self.articulation.get_joint_positions(joint_indices=indices)
        if current is None or len(current) == 0:
            return False
        drive_position = float(current[0])
        last_position = self.last_sample_position
        self.last_sample_position = drive_position
        self.sample_count += 1
        if last_position is None:
            return False
        return (
            abs(drive_position - float(last_position)) <= self.config.settle_position_delta
            or self.sample_count > self.config.settle_max_samples
        )

    def _set_gripper_drive(self, stiffness: float | None = None, max_force: float | None = None) -> None:
        from pxr import UsdPhysics
        import omni.usd

        control_prim_path = self.robot_cfg.gripper_controll_joint.get(self.arm, "")
        if not control_prim_path:
            return
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(control_prim_path)
        if not prim or not prim.IsValid():
            return
        drive = UsdPhysics.DriveAPI.Get(prim, self.robot_cfg.gripper_type)
        if not drive:
            return
        if stiffness is not None and self._drive_cache.get("stiffness") != float(stiffness):
            drive.GetStiffnessAttr().Set(float(stiffness))
            self._drive_cache["stiffness"] = float(stiffness)
        if max_force is not None and self._drive_cache.get("max_force") != float(max_force):
            drive.GetMaxForceAttr().Set(float(max_force))
            self._drive_cache["max_force"] = float(max_force)


def gripper_control_config_from_task(task_info: dict) -> GripperControlConfig:
    raw = task_info.get("slam_setting", {}).get("manipulation", {}).get("gripper", {})
    return GripperControlConfig(
        close_max_force=float(raw.get("close_max_force", GripperControlConfig.close_max_force)),
        hold_stiffness=float(raw.get("hold_stiffness", GripperControlConfig.hold_stiffness)),
        hold_max_force=float(raw.get("hold_max_force", GripperControlConfig.hold_max_force)),
        open_max_force=float(raw.get("open_max_force", GripperControlConfig.open_max_force)),
        open_tolerance=float(raw.get("open_tolerance", GripperControlConfig.open_tolerance)),
        settle_interval_sec=float(raw.get("settle_interval_sec", GripperControlConfig.settle_interval_sec)),
        settle_position_delta=float(raw.get("settle_position_delta", GripperControlConfig.settle_position_delta)),
        settle_max_samples=int(raw.get("settle_max_samples", GripperControlConfig.settle_max_samples)),
    )


def _fit_vector(values, count: int, default: float) -> np.ndarray:
    if not values:
        return np.full(count, float(default), dtype=np.float64)
    vector = np.asarray(values, dtype=np.float64)
    if len(vector) == count:
        return vector
    if len(vector) == 1:
        return np.full(count, float(vector[0]), dtype=np.float64)
    if len(vector) > count:
        return vector[:count]
    padded = np.full(count, float(default), dtype=np.float64)
    padded[: len(vector)] = vector
    return padded
