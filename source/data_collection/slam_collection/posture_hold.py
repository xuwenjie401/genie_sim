"""Articulation posture hold for interactive SLAM teleop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.base_utils.logger import logger


@dataclass
class PostureHoldConfig:
    enabled: bool = True
    max_effort: float = 5000.0
    stiffness: float = 50000.0
    damping: float = 5000.0


class ArticulationPostureHold:
    """Continuously command captured joint positions so Galbot recovers from pushes."""

    def __init__(self, articulation, config: PostureHoldConfig) -> None:
        self.articulation = articulation
        self.config = config
        self.dof_names = list(getattr(articulation, "dof_names", []) or [])
        self.name_to_index = {name: index for index, name in enumerate(self.dof_names)}
        self.targets: dict[str, float] = {}
        self.suspended_joint_names: set[str] = set()
        if config.enabled:
            self._strengthen_drives()
            self.capture_current_targets()

    def capture_current_targets(self) -> None:
        positions = self.articulation.get_joint_positions()
        if positions is None:
            return
        self.targets = {
            name: float(positions[index])
            for name, index in self.name_to_index.items()
            if index < len(positions)
        }
        logger.info(f"Articulation posture hold enabled for {len(self.targets)} joints")

    def update_targets(self, joint_names: list[str], positions) -> None:
        if not self.config.enabled:
            return
        for joint_name, position in zip(joint_names, positions):
            if joint_name in self.name_to_index:
                self.targets[joint_name] = float(position)

    def update_targets_from_current(self, joint_names: list[str]) -> None:
        if not self.config.enabled or not joint_names:
            return
        indices = []
        names = []
        for joint_name in joint_names:
            joint_index = self.name_to_index.get(joint_name)
            if joint_index is None:
                continue
            indices.append(int(joint_index))
            names.append(joint_name)
        if not indices:
            return
        positions = self.articulation.get_joint_positions(joint_indices=np.asarray(indices, dtype=np.int32))
        if positions is None:
            return
        self.update_targets(names, positions)

    def set_suspended_joint_names(self, joint_names) -> None:
        self.suspended_joint_names = {str(joint_name) for joint_name in joint_names}

    def apply(self) -> None:
        if not self.config.enabled or not self.targets:
            return
        try:
            from isaacsim.core.utils.types import ArticulationAction

            indices = []
            positions = []
            for joint_name, position in self.targets.items():
                if joint_name in self.suspended_joint_names:
                    continue
                joint_index = self.name_to_index.get(joint_name)
                if joint_index is None:
                    continue
                indices.append(int(joint_index))
                positions.append(float(position))
            if not indices:
                return

            joint_indices = np.asarray(indices, dtype=np.int32)
            target_positions = self._clip_positions(joint_indices, np.asarray(positions, dtype=np.float64))
            if hasattr(self.articulation, "set_joint_position_targets"):
                self.articulation.set_joint_position_targets(target_positions, joint_indices=joint_indices)
            if hasattr(self.articulation, "set_joint_velocity_targets"):
                self.articulation.set_joint_velocity_targets(
                    np.zeros(len(joint_indices), dtype=np.float64),
                    joint_indices=joint_indices,
                )
            self.articulation.apply_action(
                ArticulationAction(
                    joint_positions=target_positions,
                    joint_indices=joint_indices,
                )
            )
        except Exception as exc:
            logger.warning(f"Failed to apply articulation posture hold: {exc}")

    def _strengthen_drives(self) -> None:
        joint_indices = np.arange(len(self.dof_names), dtype=np.int32)
        if len(joint_indices) == 0:
            return
        try:
            articulation_view = self.articulation._articulation_view
            articulation_view.set_max_efforts(
                values=np.full(len(joint_indices), self.config.max_effort, dtype=np.float64),
                joint_indices=joint_indices,
            )
            if hasattr(articulation_view, "set_gains"):
                kps = np.full((1, len(joint_indices)), self.config.stiffness, dtype=np.float64)
                kds = np.full((1, len(joint_indices)), self.config.damping, dtype=np.float64)
                try:
                    articulation_view.set_gains(kps=kps, kds=kds, joint_indices=joint_indices)
                except Exception:
                    articulation_view.set_gains(
                        kps=kps.reshape(-1),
                        kds=kds.reshape(-1),
                        joint_indices=joint_indices,
                    )
            logger.info(f"Strengthened articulation drives for {len(joint_indices)} joints")
        except Exception as exc:
            logger.warning(f"Failed to strengthen articulation drives: {exc}")

    def _clip_positions(self, joint_indices: np.ndarray, positions: np.ndarray) -> np.ndarray:
        try:
            lowers = self.articulation.dof_properties["lower"][joint_indices]
            uppers = self.articulation.dof_properties["upper"][joint_indices]
            return np.clip(positions, lowers, uppers)
        except Exception:
            return positions


def posture_hold_config_from_task(task_info: dict) -> PostureHoldConfig:
    hold = task_info.get("slam_setting", {}).get("articulation_hold", {})
    return PostureHoldConfig(
        enabled=bool(hold.get("enabled", True)),
        max_effort=float(hold.get("max_effort", PostureHoldConfig.max_effort)),
        stiffness=float(hold.get("stiffness", PostureHoldConfig.stiffness)),
        damping=float(hold.get("damping", PostureHoldConfig.damping)),
    )
