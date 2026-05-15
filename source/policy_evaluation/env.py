"""Policy evaluation environment wrapper."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from policy_evaluation.adapters.single_arm_7dof import SingleArm7DofAdapter
from policy_evaluation.config import EvalConfig
from policy_evaluation.evaluation.final_checker import FinalTaskChecker, CheckResult
from policy_evaluation.world import EvalWorld


class DataCollectionTaskEnv:
    def __init__(
        self,
        world: EvalWorld,
        task_instance: dict[str, Any],
        adapter: SingleArm7DofAdapter,
        config: EvalConfig,
    ):
        self.world = world
        self.task_instance = task_instance
        self.adapter = adapter
        self.config = config
        self.checker = FinalTaskChecker(task_instance)
        self.current_step = 0
        self.arm_reset_reference: np.ndarray | None = None
        self.arm_reset_armed = False
        self.arm_reset_consecutive = 0
        self.arm_reset_max_error = 0.0

    def reset(self) -> dict[str, Any]:
        self.current_step = 0
        observation = self.observe()
        self.arm_reset_reference = self._arm_joint_vector(observation)
        self.arm_reset_armed = False
        self.arm_reset_consecutive = 0
        self.arm_reset_max_error = 0.0
        return observation

    def observe(self) -> dict[str, Any]:
        joint_state = self.world.get_joint_state(self.adapter.joint_names)
        gripper_state = self.world.get_joint_state(self.adapter.gripper_joint_names) if self.adapter.gripper_joint_names else {}
        images = {
            "head": self.world.capture(self.config.cameras.head),
            "wrist": self.world.capture(self.config.cameras.wrist),
        }
        if self.config.cameras.right_policy != "zeros_like_left":
            images["right_wrist"] = self.world.capture(self.config.cameras.right_policy)
        state = self.adapter.build_state(joint_state, gripper_state)
        return {
            "state": state,
            "joint_state": joint_state,
            "gripper_state": gripper_state,
            "images": images,
        }

    def capture_observer(self) -> np.ndarray | None:
        observer = self.config.cameras.observer
        if not observer:
            return None
        return self.world.capture(observer.get("prim_path", "/World/PolicyEval/ObserverCamera"))

    def capture_video_views(self) -> dict[str, np.ndarray | None]:
        return {
            "head": self.world.capture(self.config.cameras.head),
            "wrist": self.world.capture(self.config.cameras.wrist),
            "observer": self.capture_observer(),
        }

    def step(
        self,
        action: np.ndarray,
        video_frame_callback: Callable[[], None] | None = None,
        record_action_frame: bool = True,
    ) -> dict[str, Any]:
        decoded = self.adapter.decode_action(action)
        self.world.command_joint_positions(self.adapter.joint_names, decoded.arm_positions)
        wait_for_close_hold = decoded.gripper_open is False and self.config.wait_for_gripper_close_hold
        if decoded.gripper_open is not None:
            self.world.command_gripper(self.config.adapter.arm, decoded.gripper_open)
        frames_per_action = max(1, int(round(self.config.physics_step / self.config.control_hz)))
        self.world.step_frames(frames_per_action)
        if video_frame_callback is not None and record_action_frame:
            video_frame_callback()
        if wait_for_close_hold:
            self._wait_for_gripper_close_hold(video_frame_callback)
        self.current_step += 1
        return self.observe()

    def final_check(self) -> CheckResult:
        return self.checker.check(self.world)

    def check_grasp_lost_termination(self, step_count: int) -> dict[str, Any]:
        if not self.config.terminate_on_grasp_lost:
            return {"enabled": False, "terminated": False}
        if step_count <= self.config.grasp_steps_threshold:
            return {
                "enabled": True,
                "terminated": False,
                "step": step_count,
                "grasp_steps_threshold": self.config.grasp_steps_threshold,
            }

        object_id = self.world.target_object_id
        if not object_id:
            return {
                "enabled": True,
                "terminated": False,
                "step": step_count,
                "error": "target grasp object id is empty",
            }

        tcp_position, tcp_prim_path = self.world.get_tcp_position(self.config.adapter.arm)
        object_position = self.world.get_object_position(object_id)
        distance = float(np.linalg.norm(tcp_position - object_position))
        threshold = float(self.config.grasp_judge_distance_threshold)
        return {
            "enabled": True,
            "terminated": distance > threshold,
            "step": step_count,
            "arm": self.config.adapter.arm,
            "tcp_prim_path": tcp_prim_path,
            "object_id": object_id,
            "distance": distance,
            "threshold": threshold,
            "grasp_steps_threshold": self.config.grasp_steps_threshold,
            "tcp_position": tcp_position.tolist(),
            "object_position": object_position.tolist(),
        }

    def check_arm_reset_termination(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self.config.terminate_on_arm_reset:
            return {"enabled": False, "terminated": False, "ready": False}

        current = self._arm_joint_vector(observation)
        if self.arm_reset_reference is None:
            self.arm_reset_reference = current.copy()

        max_error = float(np.max(np.abs(current - self.arm_reset_reference))) if current.size else 0.0
        self.arm_reset_max_error = max(self.arm_reset_max_error, max_error)
        near_reset = max_error <= self.config.arm_reset_tolerance
        if max_error >= self.config.arm_reset_away_threshold:
            self.arm_reset_armed = True
            self.arm_reset_consecutive = 0
        elif self.arm_reset_armed and near_reset:
            self.arm_reset_consecutive += 1
        elif self.arm_reset_armed:
            self.arm_reset_consecutive = 0

        ready = (
            self.arm_reset_armed
            and self.arm_reset_consecutive >= self.config.arm_reset_consecutive_steps
        )
        terminated = self.config.terminate_on_arm_reset and ready
        return {
            "enabled": True,
            "terminated": terminated,
            "ready": ready,
            "armed": self.arm_reset_armed,
            "near_reset": near_reset,
            "max_abs_error": max_error,
            "max_seen_error": self.arm_reset_max_error,
            "consecutive": self.arm_reset_consecutive,
            "tolerance": self.config.arm_reset_tolerance,
            "away_threshold": self.config.arm_reset_away_threshold,
            "required_consecutive": self.config.arm_reset_consecutive_steps,
            "joint_names": list(self.adapter.joint_names),
            "reference": self.arm_reset_reference.tolist(),
            "current": current.tolist(),
        }

    def _wait_for_gripper_close_hold(self, video_frame_callback: Callable[[], None] | None = None) -> None:
        max_frames = max(1, int(round(self.config.gripper_close_hold_timeout_sec * self.config.physics_step)))
        for _ in range(max_frames):
            if self.world.is_gripper_holding(self.config.adapter.arm):
                self.world.step_frames(1)
                if video_frame_callback is not None:
                    video_frame_callback()
                return
            self.world.step_frames(1)
            if video_frame_callback is not None:
                video_frame_callback()
        print(
            f"[policy_eval] gripper close hold wait timed out after "
            f"{self.config.gripper_close_hold_timeout_sec:.2f}s",
            flush=True,
        )

    def _arm_joint_vector(self, observation: dict[str, Any]) -> np.ndarray:
        joint_state = observation.get("joint_state", {})
        return np.asarray(
            [float(joint_state.get(name, 0.0)) for name in self.adapter.joint_names],
            dtype=np.float64,
        )
