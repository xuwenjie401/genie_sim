"""Keyboard-triggered left-arm manipulation for Galbot SLAM teleop."""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from common.base_utils.logger import logger
from slam_collection.gripper_control import LeftGripperStateMachine


@dataclass
class LocalCollisionConfig:
    enabled: bool = True
    radius: float = 0.45
    max_obstacles: int = 12
    max_mesh_faces: int = 128
    include_prefixes: tuple[str, ...] = (
        "/World/SceneObjectPlacer/PlacedObjects",
    )


@dataclass
class ManipulationConfig:
    enabled: bool = True
    arm: str = "left"
    prewarm_on_start: bool = True
    prewarm_interaction_assets: bool = True
    empty_position_base: tuple[float, float, float] = (0.55, 0.16, 0.90)
    empty_quaternion_base: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    empty_use_current_orientation: bool = True
    grasp_offset_base: tuple[float, float, float] = (0.0, 0.0, 0.16)
    place_offset_base: tuple[float, float, float] = (0.0, 0.0, 0.18)
    lift_offset_base: tuple[float, float, float] = (0.0, 0.0, 0.18)
    grasp_quaternion_base: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    place_quaternion_base: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    grasp_target: Any = None
    place_target: Any = None
    trajopt_steps: int = 100
    use_interaction_poses: bool = True
    fallback_to_object_offset: bool = True
    max_grasp_candidates: int = 12
    max_place_candidates: int = 12
    place_align_samples: int = 12
    place_approach_up: float = 0.05
    grasp_upper_percentile: float = 75.0
    disable_upside_down_grasp: bool = True
    grasp_vertical_threshold_deg: float = 20.0
    grasp_reject_towards_robot: bool = True
    grasp_towards_robot_max_dot: float = 0.0
    grasp_preferred_height_percentile: float = 45.0
    grasp_distance_weight: float = 1.0
    grasp_approach_weight: float = 0.35
    grasp_height_weight: float = 0.25
    grasp_orientation_weight: float = 0.35
    grasp_upright_weight: float = 0.20
    grasp_support_collision: bool = True
    grasp_support_z_margin: float = 0.10
    grasp_support_xy_margin: float = 0.15
    grasp_max_support_obstacles: int = 4
    attach_on_close: bool = True
    attach_distance_threshold: float = 0.25
    attach_max_closed_fraction: float = 0.98
    detach_on_open_close_fraction: float = 0.35
    detach_on_open_fraction_drop: float = 0.25
    detach_on_open_timeout_sec: float = 1.2
    disable_attached_target_collisions: bool = True
    attached_support_clearance: float = 0.015
    joint_select_candidate_count: int = 4
    joint_select_reject_threshold: float = 2.8
    joint_select_path_weight: float = 0.05
    joint_delta_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 1.2)
    lift_force_support_collision: bool = False
    lift_retry_without_collision: bool = True
    local_collision: LocalCollisionConfig = field(default_factory=LocalCollisionConfig)


@dataclass
class TargetContext:
    prim_path: str
    asset_id: str
    data_info_dir: str
    pose_world: np.ndarray


@dataclass
class AttachedTarget:
    prim_path: str
    tcp_to_object: np.ndarray
    kinematic_attrs: list[tuple[Any, Any]] = field(default_factory=list)
    collision_attrs: list[tuple[Any, Any]] = field(default_factory=list)
    local_bbox_corners: np.ndarray | None = None
    support_top_z: float | None = None


class InteractionObject:
    def __init__(self, name: str, pose_world: np.ndarray, size, elements: dict) -> None:
        self.name = name
        self.obj_pose = pose_world
        self.obj_length = np.asarray(size if size is not None else [0.001, 0.001, 0.001], dtype=np.float64)
        self.elements = elements or {}
        self.xyz = np.zeros(3, dtype=np.float64)
        self.direction = np.array([0.0, 0.0, 0.05], dtype=np.float64)
        self.constraint_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.angle_sample_num = 72

    def update_aligned_info(self, element: dict) -> None:
        self.xyz = np.asarray(element.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.direction = np.asarray(element.get("direction", [0.0, 0.0, 0.05]), dtype=np.float64)
        self.constraint_axis = np.asarray(element.get("constraint_axis", [1.0, 0.0, 0.0]), dtype=np.float64)
        self.angle_sample_num = int(element.get("angle_sample_num", 72))


class LeftArmManipulationController:
    """Discrete manipulation actions for the left arm.

    Arm trajectories are planned with Curobo MotionGen. Target poses are in the
    robot-base frame because the local collision world is also expressed in that
    frame.
    """

    def __init__(
        self,
        world,
        articulation,
        robot_root,
        robot_cfg,
        task_info: dict,
        gripper: LeftGripperStateMachine,
    ) -> None:
        self.world = world
        self.articulation = articulation
        self.robot_root = robot_root
        self.robot_cfg = robot_cfg
        self.task_info = task_info
        self.gripper = gripper
        self.config = manipulation_config_from_task(task_info)
        self.enabled = self.config.enabled and self.config.arm == "left"
        self.motion = None
        self.active_label = ""
        self.left_arm_joint_names = list(robot_cfg.active_arm_joints.get("left", []))
        self.left_gripper_joint_names = list(robot_cfg.finger_names.get("left", []))
        self.reset_joint_targets = self._initial_reset_joint_targets()
        self._last_motion_active = False
        self._completed_arm_motion = False
        self._attached_target: AttachedTarget | None = None
        self._pending_detach_on_open = False
        self._pending_detach_start_time = 0.0
        self._pending_detach_start_close_fraction: float | None = None

        if self.enabled:
            if self.config.prewarm_on_start:
                logger.info("Left arm Curobo MotionGen will be prewarmed before keyboard teleop starts")
            else:
                logger.info("Left arm Curobo MotionGen will be initialized lazily on the first manipulation action")

    def handle_action(self, action: str) -> None:
        if action == "gripper_open":
            self._begin_gripper_release()
            self.gripper.command_open()
            return
        if action == "gripper_close":
            if self._pending_detach_on_open:
                self._complete_gripper_release(force_open=True, reason="new close command")
            self.gripper.command_close()
            return
        if not self.enabled:
            logger.warning(f"Manipulation action ignored because controller is disabled: {action}")
            return
        self._ensure_curobo()
        if self.motion is None:
            logger.warning(f"Manipulation action ignored because Curobo initialization failed: {action}")
            return
        if self.is_motion_active():
            logger.warning(f"Manipulation action ignored while {self.active_label} is still running: {action}")
            return

        if action == "empty_move":
            empty_quaternion = self.config.empty_quaternion_base
            if self.config.empty_use_current_orientation:
                current_quaternion = self._current_tcp_quaternion_base()
                if current_quaternion is not None:
                    empty_quaternion = current_quaternion
            self._plan_to_base_pose(
                self.config.empty_position_base,
                empty_quaternion,
                label="empty_move",
            )
        elif action == "move_grasp":
            self._plan_move_grasp()
        elif action == "move_place":
            self._plan_move_place()
        elif action == "lift":
            self._plan_lift()
        elif action == "reset_arm":
            self._detach_grasp_target()
            self.gripper.command_open()
            self._plan_reset()

    def step(self) -> None:
        self.gripper.step()
        self._maybe_detach_after_open()
        self._update_grasp_attachment()
        if self.motion is None:
            self._follow_attached_target()
            return

        was_active = self.is_motion_active()
        if was_active:
            self.motion.on_physics_step()
        is_active = self.is_motion_active()
        if was_active and not is_active:
            label = self.active_label
            success = bool(getattr(self.motion, "success", False))
            self.active_label = ""
            self._completed_arm_motion = True
            logger.info(f"Left arm motion finished: {label}, success={success}")
        self._last_motion_active = is_active
        self._follow_attached_target()

    def is_motion_active(self) -> bool:
        return self.motion is not None and getattr(self.motion, "cmd_plan", None) is not None

    def suspended_joint_names(self) -> set[str]:
        names = set(self.left_gripper_joint_names)
        if self.is_motion_active():
            names.update(self.left_arm_joint_names)
        return names

    def arm_joint_names(self) -> list[str]:
        return list(self.left_arm_joint_names)

    def consume_completed_arm_motion(self) -> bool:
        completed = self._completed_arm_motion
        self._completed_arm_motion = False
        return completed

    def prewarm(self) -> None:
        if not self.enabled or not self.config.prewarm_on_start:
            return
        start_time = time.monotonic()
        logger.info("Prewarming left arm manipulation controller")
        self._ensure_curobo()
        self._preload_collision_modules()
        if self.config.prewarm_interaction_assets:
            self._preload_interaction_assets()
        logger.info(f"Left arm manipulation prewarm finished in {time.monotonic() - start_time:.3f}s")

    def _preload_collision_modules(self) -> None:
        try:
            from curobo.geom.types import WorldConfig  # noqa: F401
            from server.motion_generator.mesh_utils import simplify_obstacles_from_stage  # noqa: F401
        except Exception as exc:
            logger.warning(f"Failed to preload Curobo collision modules: {exc}")

    def _preload_interaction_assets(self) -> None:
        warmed = []
        for label, target in (("grasp", self.config.grasp_target), ("place", self.config.place_target)):
            context = self._resolve_target_context(target, label)
            if context is None:
                continue
            try:
                _load_interaction(context.asset_id, context.data_info_dir)
                _load_object_parameters(context.asset_id, context.data_info_dir)
                if label == "grasp":
                    _load_grasp_poses(context.asset_id, context.data_info_dir)
                    _interaction_elements(context.asset_id, context.data_info_dir, "active", "place")
                else:
                    _interaction_elements(context.asset_id, context.data_info_dir, "passive", "place")
                warmed.append(f"{label}:{context.asset_id}")
            except Exception as exc:
                logger.warning(f"Failed to preload {label} interaction assets for {context.asset_id}: {exc}")
        if warmed:
            logger.info(f"Preloaded interaction assets: {', '.join(warmed)}")

    def _initialize_curobo(self) -> None:
        if self.motion is not None:
            return
        os.environ.setdefault("GENIESIM_CUROBO_INCLUDE_BACKGROUND_OBSTACLES", "false")
        os.environ.setdefault("GENIESIM_CUROBO_OBSTACLE_DIAGNOSTICS", "false")
        os.environ["GENIESIM_CUROBO_MESH_CACHE_SIZE"] = str(max(4, int(self.config.local_collision.max_obstacles)))

        from server.motion_generator.motion_gen_reacher import CuroboMotion

        self.motion = CuroboMotion(
            name="slam_left_arm",
            robot=self.articulation,
            world=self.world,
            robot_cfg=self.robot_cfg.curobo_config_file["left"],
            robot_prim_path=self.robot_cfg.robot_prim_path,
            robot_list=[],
            step=int(self.config.trajopt_steps),
            debug=False,
            skip_initial_obstacles=True,
            apply_retract_config=False,
        )
        self.motion.set_obstacles = lambda: None
        logger.info("Left arm Curobo MotionGen initialized for SLAM teleop")

    def _ensure_curobo(self) -> None:
        if self.motion is not None:
            return
        try:
            self._initialize_curobo()
        except Exception as exc:
            self.motion = None
            logger.warning(f"Failed to initialize left arm Curobo MotionGen: {exc}")

    def _plan_move_grasp(self) -> None:
        candidates = []
        ignore_paths = []
        force_paths = []
        if self.config.use_interaction_poses:
            context = self._resolve_target_context(self.config.grasp_target, "grasp")
            if context is not None:
                force_paths = [context.prim_path] + self._support_collision_paths(context)
                candidates = self._grasp_candidates_from_interaction(context)
        if not candidates and self.config.fallback_to_object_offset:
            target = self._target_pose_from_object(
                self.config.grasp_target,
                self.config.grasp_offset_base,
                self.config.grasp_quaternion_base,
                "grasp",
            )
            if target is not None:
                position, quaternion, _prim_path = target
                candidates = [(position, quaternion)]
        self._plan_candidate_poses(
            candidates,
            "move_grasp",
            ignore_paths,
            self.config.max_grasp_candidates,
            force_prim_paths=force_paths,
        )

    def _plan_move_place(self) -> None:
        candidates = []
        ignore_paths = []
        force_paths = []
        if self.config.use_interaction_poses:
            grasp_context = self._resolve_target_context(self.config.grasp_target, "grasp")
            place_context = self._resolve_target_context(self.config.place_target, "place")
            if grasp_context is not None and place_context is not None:
                ignore_paths = [grasp_context.prim_path]
                force_paths = [place_context.prim_path]
                candidates = self._place_candidates_from_interaction(grasp_context, place_context)
        if not candidates and self.config.fallback_to_object_offset:
            target = self._target_pose_from_object(
                self.config.place_target,
                self.config.place_offset_base,
                self.config.place_quaternion_base,
                "place",
            )
            if target is not None:
                position, quaternion, _prim_path = target
                candidates = [(position, quaternion)]
        self._plan_candidate_poses(
            candidates,
            "move_place",
            ignore_paths,
            self.config.max_place_candidates,
            force_prim_paths=force_paths,
        )

    def _plan_candidate_poses(
        self,
        candidates,
        label: str,
        ignore_prim_paths: list[str],
        max_candidates: int,
        force_prim_paths: list[str] | None = None,
    ) -> None:
        if not candidates:
            logger.warning(f"No candidate poses available for {label}")
            return
        candidate_limit = max(1, min(int(max_candidates), int(self.config.joint_select_candidate_count)))
        limited = list(candidates)[:candidate_limit]
        logger.info(f"Trying {len(limited)}/{len(candidates)} candidate poses for {label}")
        best_plan = None
        best_idx_list = None
        best_score = float("inf")
        best_max_delta = float("inf")
        best_index = -1
        for index, (position_base, quaternion_base) in enumerate(limited):
            if self._plan_to_base_pose(
                position_base,
                quaternion_base,
                label=f"{label}[{index}]",
                ignore_prim_paths=ignore_prim_paths,
                force_prim_paths=force_prim_paths or [],
            ):
                score, max_delta = self._current_plan_joint_score()
                logger.info(f"{label}[{index}] joint-change score={score:.3f}, max_delta={max_delta:.3f}")
                if score < best_score:
                    best_plan = self.motion.cmd_plan
                    best_idx_list = list(getattr(self.motion, "idx_list", []) or [])
                    best_score = score
                    best_max_delta = max_delta
                    best_index = index
        if best_plan is None:
            logger.warning(f"Curobo failed all candidate poses for {label}")
            return

        self.motion.cmd_plan = best_plan
        self.motion.idx_list = best_idx_list or []
        self.motion.cmd_idx = 0
        self.active_label = label
        threshold = float(self.config.joint_select_reject_threshold)
        if best_max_delta > threshold:
            logger.warning(
                f"Selected {label}[{best_index}] despite large joint delta: "
                f"max_delta={best_max_delta:.3f}, threshold={threshold:.3f}"
            )
        else:
            logger.info(f"Selected {label}[{best_index}] with lower joint change")

    def _plan_to_base_pose(
        self,
        position_base,
        quaternion_base,
        label: str,
        ignore_prim_paths=None,
        force_prim_paths=None,
    ) -> bool:
        if self.motion is None:
            return False
        self._sync_locked_joints()
        position_base = np.asarray(position_base, dtype=np.float64)
        quaternion_base = _normalize_quat(np.asarray(quaternion_base, dtype=np.float64))
        self.motion.cmd_plan = None
        self._load_local_collision_world(
            position_base,
            ignore_prim_paths=ignore_prim_paths or [],
            force_prim_paths=force_prim_paths or [],
        )
        self.motion.target = _make_target_xform(position_base, quaternion_base)
        self.motion.caculate_ik_goal()
        self.motion.exclude_js(self.left_arm_joint_names)
        if self.motion.cmd_plan is None:
            logger.warning(f"Curobo failed to plan left arm motion: {label}")
            self.active_label = ""
            return False
        self.active_label = label
        logger.info(f"Curobo planned left arm motion: {label}")
        return True

    def _current_plan_joint_score(self) -> tuple[float, float]:
        if self.motion is None or getattr(self.motion, "cmd_plan", None) is None:
            return float("inf"), float("inf")
        plan = self.motion.cmd_plan
        joint_names = list(plan.joint_names)
        if not joint_names:
            return float("inf"), float("inf")
        positions = plan.position.detach().cpu().numpy()
        if positions.ndim != 2 or len(positions) == 0:
            return float("inf"), float("inf")
        current_all = self.articulation.get_joint_positions()
        if current_all is None:
            return float("inf"), float("inf")
        current = []
        for joint_name in joint_names:
            try:
                current.append(float(current_all[self.articulation.get_dof_index(joint_name)]))
            except Exception:
                current.append(float(positions[0, len(current)]))
        current = np.asarray(current, dtype=np.float64)
        final_delta = np.abs(_wrap_joint_delta(positions[-1] - current))
        if len(positions) > 1:
            path_delta = np.sum(np.abs(_wrap_joint_delta(np.diff(positions, axis=0))), axis=0)
        else:
            path_delta = np.zeros_like(final_delta)
        weights = _fit_weights(self.config.joint_delta_weights, len(joint_names))
        score = float(np.sum(weights * final_delta) + float(self.config.joint_select_path_weight) * np.sum(weights * path_delta))
        return score, float(np.max(final_delta))

    def _plan_lift(self) -> None:
        if self.motion is None:
            return
        self._sync_locked_joints()
        current_position_base = self._current_tcp_position_base()
        current_quaternion_base = self._current_tcp_quaternion_base()
        if current_position_base is None:
            current_position_base = np.asarray(self.config.empty_position_base, dtype=np.float64)
        if current_quaternion_base is None:
            current_quaternion_base = np.asarray(self.config.empty_quaternion_base, dtype=np.float64)
        lift_offset = np.asarray(self.config.lift_offset_base, dtype=np.float64)
        ignore_paths = []
        force_paths = []
        grasp_context = self._resolve_target_context(self.config.grasp_target, "grasp")
        if grasp_context is not None:
            ignore_paths.append(grasp_context.prim_path)
            if self.config.lift_force_support_collision:
                force_paths.extend(self._support_collision_paths(grasp_context))
        lift_offset = self._adjust_lift_offset_for_attached_clearance(lift_offset)
        lift_goal_position_base = current_position_base + lift_offset
        if not self._try_plan_lift(
            current_position_base,
            current_quaternion_base,
            lift_offset,
            lift_goal_position_base,
            ignore_paths,
            force_paths,
            empty_collision_world=False,
        ):
            if not self.config.lift_retry_without_collision:
                logger.warning("Curobo failed to plan left arm lift")
                return
            logger.warning("Retrying left arm lift with empty collision world")
            if not self._try_plan_lift(
                current_position_base,
                current_quaternion_base,
                lift_offset,
                lift_goal_position_base,
                ignore_paths,
                force_paths=[],
                empty_collision_world=True,
            ):
                logger.warning("Curobo failed to plan left arm lift")
                return
        self.active_label = "lift"
        logger.info("Curobo planned left arm lift with partial path constraint")

    def _try_plan_lift(
        self,
        current_position_base,
        current_quaternion_base,
        lift_offset,
        lift_goal_position_base,
        ignore_paths,
        force_paths,
        empty_collision_world: bool,
    ) -> bool:
        if self.motion is None:
            return False
        self.motion.cmd_plan = None
        if empty_collision_world:
            self._load_empty_collision_world()
        else:
            self._load_local_collision_world(
                lift_goal_position_base,
                ignore_prim_paths=ignore_paths,
                force_prim_paths=force_paths,
            )
        self.motion.target = _make_target_xform(current_position_base, current_quaternion_base)
        lift = list(lift_offset) + [1.0, 0.0, 0.0, 0.0]
        self.motion.caculate_ik_goal(
            goal_offset=lift,
            path_constraint=_lift_path_constraint(lift_offset),
            offset_and_constraint_in_goal_frame=False,
            from_current_pose=True,
        )
        self.motion.exclude_js(self.left_arm_joint_names)
        return self.motion.cmd_plan is not None

    def _plan_reset(self) -> None:
        if self.motion is None:
            return
        self._sync_locked_joints()
        self._load_local_collision_world(self.config.empty_position_base, ignore_prim_paths=[])
        if not self.motion.plan_joint_goal(self.reset_joint_targets):
            logger.warning("Curobo failed to plan left arm reset")
            return
        self.motion.exclude_js(self.left_arm_joint_names)
        self.active_label = "reset_arm"
        logger.info("Curobo planned left arm reset")

    def _load_local_collision_world(
        self,
        goal_position_base,
        ignore_prim_paths: list[str],
        force_prim_paths: list[str] | None = None,
    ) -> None:
        if self.motion is None or not self.config.local_collision.enabled:
            return
        from curobo.geom.types import WorldConfig
        from server.motion_generator.mesh_utils import simplify_obstacles_from_stage

        local_paths = self._select_local_collision_paths(
            goal_position_base,
            ignore_prim_paths,
            force_prim_paths=force_prim_paths or [],
        )
        if not local_paths:
            world_config = WorldConfig()
        else:
            ignore_substring = [
                self.robot_cfg.robot_prim_path,
                "/World/target",
                "/World/SLAMThirdPersonCamera",
                "/curobo",
            ] + [str(path) for path in ignore_prim_paths if path]
            self.motion.usd_help.background_collision_only = True
            self.motion.usd_help.obstacle_diagnostic_enabled = False
            world_config = self.motion.usd_help.get_obstacles_from_stage(
                only_paths=local_paths,
                ignore_substring=ignore_substring,
                reference_prim_path=self.robot_cfg.robot_prim_path,
                timecode=0,
            )
            world_config = simplify_obstacles_from_stage(
                world_config,
                max_faces=int(self.config.local_collision.max_mesh_faces),
            )
        obstacle_world = world_config.get_collision_check_world()
        self.motion.world_cfg = obstacle_world
        self.motion.motion_gen.world_coll_checker.load_collision_model(
            obstacle_world,
            fix_cache_reference=self.motion.motion_gen.use_cuda_graph,
        )
        self.motion.motion_gen.graph_planner.reset_buffer()
        logger.info(f"Loaded local Curobo collision world with {len(getattr(obstacle_world, 'objects', []))} objects")

    def _load_empty_collision_world(self) -> None:
        if self.motion is None:
            return
        from curobo.geom.types import WorldConfig

        obstacle_world = WorldConfig().get_collision_check_world()
        self.motion.world_cfg = obstacle_world
        self.motion.motion_gen.world_coll_checker.load_collision_model(
            obstacle_world,
            fix_cache_reference=self.motion.motion_gen.use_cuda_graph,
        )
        self.motion.motion_gen.graph_planner.reset_buffer()
        logger.info("Loaded empty Curobo collision world")

    def _sync_locked_joints(self) -> None:
        if self.motion is None:
            return
        locked_names = set(getattr(self.motion, "lock_js_names", []) or [])
        if not locked_names:
            return
        joint_positions = self.articulation.get_joint_positions()
        if joint_positions is None:
            return
        locked_positions = {}
        for index, joint_name in enumerate(self.articulation.dof_names):
            if joint_name in locked_names and index < len(joint_positions):
                locked_positions[joint_name] = float(joint_positions[index])
        if not locked_positions:
            return
        self.motion.update_lock_joints(locked_positions)
        try:
            self.motion.update_curobo_kinematics_lock_joints(locked_positions)
        except Exception as exc:
            logger.warning(f"Failed to update Curobo kinematics locked joints: {exc}")

    def _select_local_collision_paths(
        self,
        goal_position_base,
        ignore_prim_paths: list[str],
        force_prim_paths: list[str] | None = None,
    ) -> list[str]:
        try:
            import omni.usd
            from pxr import Usd, UsdGeom
        except Exception as exc:
            logger.warning(f"Cannot select local collision paths: {exc}")
            return []

        stage = omni.usd.get_context().get_stage()
        start_world = self._current_tcp_position_world()
        goal_world = self._base_point_to_world(np.asarray(goal_position_base, dtype=np.float64))
        if start_world is None:
            start_world = goal_world
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
        include_prefixes = tuple(self.config.local_collision.include_prefixes)
        ignore_prim_paths = [str(path) for path in ignore_prim_paths if path]
        forced_mesh_paths = self._mesh_paths_for_prim_paths(stage, force_prim_paths or [], ignore_prim_paths)
        forced_set = set(forced_mesh_paths)
        candidates = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            prim_path = str(prim.GetPath())
            if prim_path.startswith(self.robot_cfg.robot_prim_path):
                continue
            if any(prim_path.startswith(path) for path in ignore_prim_paths):
                continue
            if prim_path in forced_set:
                continue
            if include_prefixes and not any(prim_path.startswith(prefix) for prefix in include_prefixes):
                continue
            try:
                box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                min_point = np.array(box.GetMin(), dtype=np.float64)
                max_point = np.array(box.GetMax(), dtype=np.float64)
                center = 0.5 * (min_point + max_point)
                half_diag = 0.5 * float(np.linalg.norm(max_point - min_point))
                distance = _distance_point_to_segment(center, start_world, goal_world)
                if distance <= self.config.local_collision.radius + half_diag:
                    candidates.append((distance, prim_path))
            except Exception:
                continue
        candidates.sort(key=lambda item: item[0])
        max_obstacles = int(self.config.local_collision.max_obstacles)
        remaining_count = max(0, max_obstacles - len(forced_mesh_paths))
        return forced_mesh_paths[:max_obstacles] + [path for _distance, path in candidates[:remaining_count]]

    def _mesh_paths_for_prim_paths(self, stage, prim_paths: list[str], ignore_prim_paths: list[str]) -> list[str]:
        try:
            from pxr import Usd, UsdGeom
        except Exception:
            return []

        mesh_paths = []
        seen = set()
        for prim_path in [str(path) for path in prim_paths if path]:
            if any(prim_path.startswith(ignore_path) for ignore_path in ignore_prim_paths):
                continue
            root = stage.GetPrimAtPath(prim_path)
            if not root or not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                path = str(prim.GetPath())
                if path in seen:
                    continue
                if path.startswith(self.robot_cfg.robot_prim_path):
                    continue
                if any(path.startswith(ignore_path) for ignore_path in ignore_prim_paths):
                    continue
                if prim.IsA(UsdGeom.Mesh):
                    seen.add(path)
                    mesh_paths.append(path)
        return mesh_paths

    def _support_collision_paths(self, context: TargetContext) -> list[str]:
        if not self.config.grasp_support_collision:
            return []
        try:
            import omni.usd
            from pxr import Usd, UsdGeom
        except Exception as exc:
            logger.warning(f"Cannot select grasp support collision paths: {exc}")
            return []

        stage = omni.usd.get_context().get_stage()
        target_prim = stage.GetPrimAtPath(context.prim_path)
        if not target_prim or not target_prim.IsValid():
            return []

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
        try:
            target_box = bbox_cache.ComputeWorldBound(target_prim).ComputeAlignedBox()
            target_min = np.array(target_box.GetMin(), dtype=np.float64)
            target_max = np.array(target_box.GetMax(), dtype=np.float64)
        except Exception:
            return []

        xy_margin = float(self.config.grasp_support_xy_margin)
        z_margin = float(self.config.grasp_support_z_margin)
        candidates = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            prim_path = str(prim.GetPath())
            if prim_path.startswith(self.robot_cfg.robot_prim_path):
                continue
            if prim_path.startswith(context.prim_path):
                continue
            try:
                box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                min_point = np.array(box.GetMin(), dtype=np.float64)
                max_point = np.array(box.GetMax(), dtype=np.float64)
            except Exception:
                continue
            if max_point[2] > target_min[2] + z_margin:
                continue
            if max_point[2] < target_min[2] - 0.5:
                continue
            overlaps_xy = (
                max_point[0] >= target_min[0] - xy_margin
                and min_point[0] <= target_max[0] + xy_margin
                and max_point[1] >= target_min[1] - xy_margin
                and min_point[1] <= target_max[1] + xy_margin
            )
            if not overlaps_xy:
                continue
            z_gap = abs(float(target_min[2] - max_point[2]))
            xy_center = 0.5 * (min_point[:2] + max_point[:2])
            target_center = 0.5 * (target_min[:2] + target_max[:2])
            xy_distance = float(np.linalg.norm(xy_center - target_center))
            candidates.append((z_gap + 0.1 * xy_distance, prim_path))

        candidates.sort(key=lambda item: item[0])
        paths = [path for _score, path in candidates[: int(self.config.grasp_max_support_obstacles)]]
        if paths:
            logger.info(f"Forced grasp support collision paths: {paths}")
        return paths

    def _support_top_z(self, prim_paths: list[str]) -> float | None:
        tops = []
        for prim_path in prim_paths:
            bounds = self._world_aabb(prim_path)
            if bounds is not None:
                _min_point, max_point = bounds
                tops.append(float(max_point[2]))
        if not tops:
            return None
        return float(max(tops))

    def _world_aabb(self, prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            import omni.usd
            from pxr import Usd, UsdGeom
        except Exception:
            return None

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        try:
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
            box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            return np.array(box.GetMin(), dtype=np.float64), np.array(box.GetMax(), dtype=np.float64)
        except Exception:
            return None

    def _target_local_bbox_corners(self, prim_path: str, pose_world: np.ndarray) -> np.ndarray | None:
        bounds = self._world_aabb(prim_path)
        if bounds is None:
            return None
        min_point, max_point = bounds
        corners_world = _aabb_corners(min_point, max_point)
        corners_world_h = np.concatenate([corners_world, np.ones((len(corners_world), 1), dtype=np.float64)], axis=1)
        corners_local_h = (np.linalg.inv(pose_world) @ corners_world_h.T).T
        return corners_local_h[:, :3]

    def _adjust_lift_offset_for_attached_clearance(self, lift_offset: np.ndarray) -> np.ndarray:
        if self._attached_target is None or self._attached_target.support_top_z is None:
            return lift_offset
        if self._attached_target.local_bbox_corners is None:
            return lift_offset
        if lift_offset.shape[0] != 3 or lift_offset[2] <= 0.0:
            return lift_offset
        tcp_pose = self._current_tcp_pose_world()
        if tcp_pose is None:
            return lift_offset
        object_pose = tcp_pose @ self._attached_target.tcp_to_object
        min_z = _transformed_points_min_z(object_pose, self._attached_target.local_bbox_corners)
        required = float(self._attached_target.support_top_z) + float(self.config.attached_support_clearance) - min_z
        if required <= lift_offset[2]:
            return lift_offset
        adjusted = lift_offset.copy()
        adjusted[2] = required
        logger.info(f"Adjusted lift z offset for attached-object clearance: {lift_offset[2]:.3f} -> {adjusted[2]:.3f}")
        return adjusted

    def _target_pose_from_object(self, target_spec, offset_base, quaternion_base, label: str):
        prim_path = self._resolve_target_prim_path(target_spec)
        if not prim_path:
            logger.warning(f"No {label} target configured or found")
            return None
        try:
            prim = _xform_prim(prim_path)
            position_world, _orientation_world = prim.get_world_pose()
            position_base = self._world_point_to_base(np.asarray(position_world, dtype=np.float64))
            position_base = position_base + np.asarray(offset_base, dtype=np.float64)
            return position_base, _normalize_quat(np.asarray(quaternion_base, dtype=np.float64)), prim_path
        except Exception as exc:
            logger.warning(f"Failed to read {label} target pose from {prim_path}: {exc}")
            return None

    def _grasp_candidates_from_interaction(self, context: TargetContext) -> list[tuple[np.ndarray, np.ndarray]]:
        grasp_poses = _load_grasp_poses(context.asset_id, context.data_info_dir)
        if grasp_poses is None or len(grasp_poses) == 0:
            logger.warning(f"No interaction grasp poses found for {context.asset_id}")
            return []
        grasp_poses = grasp_poses.copy()
        input_count = len(grasp_poses)
        if self.config.grasp_upper_percentile < 100.0 and len(grasp_poses) > 1:
            y_values = grasp_poses[:, 1, 3]
            upper = np.percentile(y_values, float(self.config.grasp_upper_percentile))
            grasp_poses = grasp_poses[y_values <= upper]
        world_poses = context.pose_world[np.newaxis, ...] @ grasp_poses
        if len(world_poses) == 0:
            return []
        world_poses = self._filter_and_sort_grasp_candidates(grasp_poses, world_poses)
        logger.info(f"Galbot grasp candidates kept/sorted: {len(world_poses)}/{input_count}")
        return [self._world_matrix_to_base_pose(pose) for pose in world_poses]

    def _filter_and_sort_grasp_candidates(self, grasp_poses: np.ndarray, world_poses: np.ndarray) -> np.ndarray:
        if len(world_poses) == 0:
            return world_poses

        mask = np.ones(len(world_poses), dtype=bool)
        if self.config.disable_upside_down_grasp:
            mask &= world_poses[:, 2, 2] > 0.0

        base_rotation = self._base_pose_matrix()[:3, :3]
        approach_world = world_poses[:, :3, 0]
        approach_base = (base_rotation.T @ approach_world.T).T
        if not self._allows_top_down_grasp_filter():
            vertical_cos = np.cos(np.deg2rad(float(self.config.grasp_vertical_threshold_deg)))
            mask &= approach_world[:, 2] > -vertical_cos
        if self.config.grasp_reject_towards_robot:
            mask &= approach_base[:, 0] >= float(self.config.grasp_towards_robot_max_dot)

        if not np.any(mask):
            logger.warning("Galbot grasp approach filter removed all candidates; falling back to upside-down-only filter")
            mask = np.ones(len(world_poses), dtype=bool)
            if self.config.disable_upside_down_grasp:
                mask &= world_poses[:, 2, 2] > 0.0
        if not np.any(mask):
            mask = np.ones(len(world_poses), dtype=bool)

        grasp_poses = grasp_poses[mask]
        world_poses = world_poses[mask]
        approach_base = approach_base[mask]

        current_tcp = self._current_tcp_position_world()
        if current_tcp is None:
            current_tcp = world_poses[0, :3, 3]
        distance_cost = _normalized_cost(np.linalg.norm(world_poses[:, :3, 3] - current_tcp[np.newaxis, :], axis=1))
        approach_cost = _normalized_cost(1.0 - approach_base[:, 0])
        upright_cost = _normalized_cost(1.0 - world_poses[:, 2, 2])

        canonical_height = grasp_poses[:, 1, 3]
        preferred_height = np.percentile(
            canonical_height,
            float(np.clip(self.config.grasp_preferred_height_percentile, 0.0, 100.0)),
        )
        height_cost = _normalized_cost(np.abs(canonical_height - preferred_height))
        orientation_cost = self._orientation_cost_to_current_tcp(world_poses)

        score = (
            float(self.config.grasp_distance_weight) * distance_cost
            + float(self.config.grasp_approach_weight) * approach_cost
            + float(self.config.grasp_height_weight) * height_cost
            + float(self.config.grasp_orientation_weight) * orientation_cost
            + float(self.config.grasp_upright_weight) * upright_cost
        )
        return world_poses[np.argsort(score)]

    def _allows_top_down_grasp_filter(self) -> bool:
        return float(self.config.grasp_vertical_threshold_deg) >= 90.0

    def _orientation_cost_to_current_tcp(self, poses_world: np.ndarray) -> np.ndarray:
        current_pose = self._current_tcp_pose_world()
        if current_pose is None or len(poses_world) == 0:
            return np.zeros(len(poses_world), dtype=np.float64)
        current_rotation = current_pose[:3, :3]
        costs = []
        for pose in poses_world:
            delta = current_rotation.T @ pose[:3, :3]
            cos_angle = np.clip((float(np.trace(delta)) - 1.0) * 0.5, -1.0, 1.0)
            costs.append(np.arccos(cos_angle) / np.pi)
        return np.asarray(costs, dtype=np.float64)

    def _place_candidates_from_interaction(
        self,
        grasp_context: TargetContext,
        place_context: TargetContext,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        active_elements = _interaction_elements(grasp_context.asset_id, grasp_context.data_info_dir, "active", "place")
        passive_elements = _interaction_elements(place_context.asset_id, place_context.data_info_dir, "passive", "place")
        if not active_elements:
            logger.warning(f"No active place interaction labels found for {grasp_context.asset_id}")
            return []
        if not passive_elements:
            logger.warning(f"No passive place interaction labels found for {place_context.asset_id}")
            return []

        current_tcp_pose = self._current_tcp_pose_world()
        if current_tcp_pose is None:
            logger.warning("Cannot compute place interaction pose without current TCP pose")
            return []
        gripper_to_object = np.linalg.inv(grasp_context.pose_world) @ current_tcp_pose

        from client.planner.common import get_aligned_pose

        grasp_params = _load_object_parameters(grasp_context.asset_id, grasp_context.data_info_dir)
        place_params = _load_object_parameters(place_context.asset_id, place_context.data_info_dir)
        active_obj = InteractionObject(
            grasp_context.asset_id,
            grasp_context.pose_world,
            grasp_params.get("size"),
            _load_interaction(grasp_context.asset_id, grasp_context.data_info_dir),
        )
        passive_obj = InteractionObject(
            place_context.asset_id,
            place_context.pose_world,
            place_params.get("size"),
            _load_interaction(place_context.asset_id, place_context.data_info_dir),
        )

        candidate_world_poses = []
        for active_element in active_elements:
            active_obj.update_aligned_info(active_element)
            for passive_element in passive_elements:
                passive_obj.update_aligned_info(passive_element)
                sample_count = int(np.lcm(active_obj.angle_sample_num, passive_obj.angle_sample_num))
                sample_count = max(1, min(sample_count, int(self.config.place_align_samples)))
                try:
                    target_object_poses = get_aligned_pose(active_obj, passive_obj, N=sample_count)
                except Exception as exc:
                    logger.warning(f"Failed to generate aligned place poses: {exc}")
                    continue
                target_gripper_poses = target_object_poses @ gripper_to_object[np.newaxis, ...]
                if self.config.place_approach_up:
                    target_gripper_poses[:, :3, 3] += np.array(
                        [0.0, 0.0, float(self.config.place_approach_up)],
                        dtype=np.float64,
                    )
                candidate_world_poses.extend(target_gripper_poses)

        if not candidate_world_poses:
            return []
        world_poses = self._sort_pose_candidates(np.asarray(candidate_world_poses, dtype=np.float64))
        return [self._world_matrix_to_base_pose(pose) for pose in world_poses]

    def _update_grasp_attachment(self) -> None:
        if (
            self.enabled
            and self.config.attach_on_close
            and not self._pending_detach_on_open
            and self.gripper.consume_holding_started()
        ):
            self._attach_grasp_target()

    def _begin_gripper_release(self) -> None:
        if self._attached_target is None:
            self._pending_detach_on_open = False
            self._pending_detach_start_time = 0.0
            self._pending_detach_start_close_fraction = None
            return
        self._pending_detach_on_open = True
        self._pending_detach_start_time = time.monotonic()
        self._pending_detach_start_close_fraction = self.gripper.close_fraction()
        logger.info("Delaying grasp target detach until the gripper opens")

    def _maybe_detach_after_open(self) -> None:
        if not self._pending_detach_on_open:
            return
        if self._attached_target is None:
            self._pending_detach_on_open = False
            self._pending_detach_start_time = 0.0
            self._pending_detach_start_close_fraction = None
            return
        close_fraction = self.gripper.close_fraction()
        detach_threshold = float(self.config.detach_on_open_close_fraction)
        if self._pending_detach_start_close_fraction is not None:
            relative_threshold = self._pending_detach_start_close_fraction - float(
                self.config.detach_on_open_fraction_drop
            )
            detach_threshold = min(detach_threshold, max(0.0, relative_threshold))
        opened_enough = (
            close_fraction is not None
            and close_fraction <= detach_threshold
        )
        timed_out = (
            float(self.config.detach_on_open_timeout_sec) > 0.0
            and time.monotonic() - self._pending_detach_start_time
            >= float(self.config.detach_on_open_timeout_sec)
        )
        if timed_out and not opened_enough:
            self._complete_gripper_release(force_open=True, reason="open timeout")
            return
        if opened_enough or self.gripper.mode == "idle":
            if close_fraction is None:
                logger.info("Releasing grasp target after gripper open")
            else:
                logger.info(
                    f"Releasing grasp target at gripper close fraction {close_fraction:.3f} "
                    f"(threshold={detach_threshold:.3f})"
                )
            self._complete_gripper_release(force_open=False, reason="gripper opened")

    def _complete_gripper_release(self, force_open: bool, reason: str) -> None:
        if force_open:
            logger.info(f"Force-completing gripper release: {reason}")
            self.gripper.force_open_pose()
        self._detach_grasp_target()

    def _attach_grasp_target(self) -> None:
        if self._attached_target is not None:
            return
        context = self._resolve_target_context(self.config.grasp_target, "grasp")
        tcp_pose = self._current_tcp_pose_world()
        if context is None or tcp_pose is None:
            return

        close_fraction = self.gripper.close_fraction()
        if (
            close_fraction is not None
            and float(self.config.attach_max_closed_fraction) > 0.0
            and close_fraction >= float(self.config.attach_max_closed_fraction)
        ):
            logger.warning(
                f"Skip grasp target attachment: gripper closed fraction is {close_fraction:.3f} "
                f"(threshold={self.config.attach_max_closed_fraction:.3f}); likely empty close"
            )
            return

        distance = float(np.linalg.norm(context.pose_world[:3, 3] - tcp_pose[:3, 3]))
        if distance > float(self.config.attach_distance_threshold):
            logger.warning(
                f"Skip grasp target attachment: target is {distance:.3f}m from TCP "
                f"(threshold={self.config.attach_distance_threshold:.3f}m)"
            )
            return

        support_paths = self._support_collision_paths(context)
        support_top_z = self._support_top_z(support_paths)
        kinematic_attrs = self._set_target_kinematic(context.prim_path, True)
        collision_attrs = []
        if self.config.disable_attached_target_collisions:
            collision_attrs = self._set_target_collision_enabled(context.prim_path, False)
        self._zero_target_velocities(context.prim_path)
        self._attached_target = AttachedTarget(
            prim_path=context.prim_path,
            tcp_to_object=np.linalg.inv(tcp_pose) @ context.pose_world,
            kinematic_attrs=kinematic_attrs,
            collision_attrs=collision_attrs,
            local_bbox_corners=self._target_local_bbox_corners(context.prim_path, context.pose_world),
            support_top_z=support_top_z,
        )
        logger.info(f"Attached grasp target to left TCP follow: {context.prim_path}")

    def _follow_attached_target(self) -> None:
        if self._attached_target is None:
            return
        tcp_pose = self._current_tcp_pose_world()
        if tcp_pose is None:
            return
        object_pose = tcp_pose @ self._attached_target.tcp_to_object
        try:
            _xform_prim(self._attached_target.prim_path).set_world_pose(
                position=object_pose[:3, 3],
                orientation=_matrix_to_quat_wxyz(object_pose[:3, :3]),
            )
            self._zero_target_velocities(self._attached_target.prim_path)
        except Exception as exc:
            logger.warning(f"Failed to follow attached grasp target; detaching: {exc}")
            self._detach_grasp_target()

    def _detach_grasp_target(self) -> None:
        self._pending_detach_on_open = False
        self._pending_detach_start_time = 0.0
        self._pending_detach_start_close_fraction = None
        if self._attached_target is None:
            return
        prim_path = self._attached_target.prim_path
        for attr, previous_value in self._attached_target.collision_attrs:
            try:
                if previous_value is None:
                    attr.Clear()
                else:
                    attr.Set(previous_value)
            except Exception:
                continue
        for attr, previous_value in self._attached_target.kinematic_attrs:
            try:
                if previous_value is None:
                    attr.Clear()
                else:
                    attr.Set(previous_value)
            except Exception:
                continue
        self._zero_target_velocities(prim_path)
        self._attached_target = None
        logger.info(f"Detached grasp target from left TCP follow: {prim_path}")

    def _set_target_kinematic(self, prim_path: str, enabled: bool) -> list[tuple[Any, Any]]:
        try:
            import omni.usd
            from pxr import UsdPhysics
        except Exception:
            return []

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(prim_path)
        if not root or not root.IsValid():
            return []

        saved_attrs = []
        for prim in _iter_prim_tree(root):
            attr = prim.GetAttribute("physics:kinematicEnabled")
            if not attr or not attr.IsValid():
                try:
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        attr = UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr()
                    else:
                        continue
                except Exception:
                    continue
            try:
                saved_attrs.append((attr, attr.Get()))
                attr.Set(bool(enabled))
            except Exception:
                continue
        return saved_attrs

    def _set_target_collision_enabled(self, prim_path: str, enabled: bool) -> list[tuple[Any, Any]]:
        try:
            import omni.usd
            from pxr import UsdPhysics
        except Exception:
            return []

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(prim_path)
        if not root or not root.IsValid():
            return []

        saved_attrs = []
        for prim in _iter_prim_tree(root):
            attr = prim.GetAttribute("physics:collisionEnabled")
            if not attr or not attr.IsValid():
                try:
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        attr = UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr()
                    else:
                        continue
                except Exception:
                    continue
            try:
                saved_attrs.append((attr, attr.Get()))
                attr.Set(bool(enabled))
            except Exception:
                continue
        return saved_attrs

    def _zero_target_velocities(self, prim_path: str) -> None:
        try:
            import omni.usd
            from pxr import Gf
        except Exception:
            return

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(prim_path)
        if not root or not root.IsValid():
            return
        zero = Gf.Vec3f(0.0, 0.0, 0.0)
        for prim in _iter_prim_tree(root):
            for attr_name in ("physics:velocity", "physics:angularVelocity"):
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid():
                    try:
                        attr.Set(zero)
                    except Exception:
                        continue

    def _resolve_target_prim_path(self, target_spec) -> str:
        if not target_spec:
            return ""
        if isinstance(target_spec, str):
            if target_spec.startswith("/"):
                return target_spec
            object_id = target_spec
        elif isinstance(target_spec, dict):
            prim_path = str(target_spec.get("prim_path", "")).strip()
            if prim_path and _prim_path_exists(prim_path):
                return prim_path
            object_id = str(target_spec.get("object_id", "")).strip()
        else:
            return ""
        if not object_id:
            return ""
        return self._find_prim_by_object_id(object_id)

    def _find_prim_by_object_id(self, object_id: str) -> str:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        fallback = ""
        normalized_object_id = _normalized_lookup_token(object_id)
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            name = prim.GetName()
            if name == object_id or path.endswith("/" + object_id):
                return path
            if not fallback and object_id in path:
                fallback = path
            if not fallback and normalized_object_id and normalized_object_id in _normalized_lookup_token(path):
                fallback = path
        return fallback

    def _resolve_target_context(self, target_spec, label: str) -> TargetContext | None:
        prim_path = self._resolve_target_prim_path(target_spec)
        if not prim_path:
            logger.warning(f"No {label} target prim configured or found")
            return None
        try:
            pose_world = self._prim_pose_matrix(prim_path)
        except Exception as exc:
            logger.warning(f"Failed to read {label} target pose from {prim_path}: {exc}")
            return None
        asset_id = _target_spec_value(target_spec, "asset_id") or _target_spec_value(target_spec, "interaction_asset_id")
        data_info_dir = _target_spec_value(target_spec, "data_info_dir")
        if not data_info_dir:
            data_info_dir = _data_info_dir_from_prim_references(prim_path)
        if not asset_id:
            asset_id = _asset_id_from_data_info_dir(data_info_dir) or _infer_asset_id_from_prim_path(prim_path)
        if not data_info_dir:
            data_info_dir = _infer_data_info_dir(asset_id)
        return TargetContext(
            prim_path=prim_path,
            asset_id=asset_id,
            data_info_dir=data_info_dir,
            pose_world=pose_world,
        )

    def _initial_reset_joint_targets(self) -> dict[str, float]:
        init_arm_pose = self.task_info.get("robot", {}).get("init_arm_pose", {})
        targets = {
            joint_name: float(init_arm_pose[joint_name])
            for joint_name in self.left_arm_joint_names
            if joint_name in init_arm_pose
        }
        if len(targets) == len(self.left_arm_joint_names):
            return targets
        current = self.articulation.get_joint_positions()
        if current is None:
            return targets
        for joint_name in self.left_arm_joint_names:
            if joint_name in targets:
                continue
            try:
                targets[joint_name] = float(current[self.articulation.get_dof_index(joint_name)])
            except Exception:
                pass
        return targets

    def _current_tcp_position_world(self) -> np.ndarray | None:
        try:
            prim = _xform_prim(self.robot_cfg.end_effector_center_prim_path["left"])
            position, _orientation = prim.get_world_pose()
            return np.asarray(position, dtype=np.float64)
        except Exception:
            return None

    def _current_tcp_pose_world(self) -> np.ndarray | None:
        try:
            return self._prim_pose_matrix(self.robot_cfg.end_effector_center_prim_path["left"])
        except Exception:
            return None

    def _current_tcp_position_base(self) -> np.ndarray | None:
        position_world = self._current_tcp_position_world()
        if position_world is None:
            return None
        return self._world_point_to_base(position_world)

    def _current_tcp_quaternion_base(self) -> np.ndarray | None:
        pose_world = self._current_tcp_pose_world()
        if pose_world is None:
            return None
        _position, quaternion = self._world_matrix_to_base_pose(pose_world)
        return quaternion

    def _world_point_to_base(self, point_world: np.ndarray) -> np.ndarray:
        base_position, base_quat = self.robot_root.get_world_pose()
        rotation = _quat_to_matrix(_normalize_quat(np.asarray(base_quat, dtype=np.float64)))
        return rotation.T @ (np.asarray(point_world, dtype=np.float64) - np.asarray(base_position, dtype=np.float64))

    def _base_point_to_world(self, point_base: np.ndarray) -> np.ndarray:
        base_position, base_quat = self.robot_root.get_world_pose()
        rotation = _quat_to_matrix(_normalize_quat(np.asarray(base_quat, dtype=np.float64)))
        return np.asarray(base_position, dtype=np.float64) + rotation @ np.asarray(point_base, dtype=np.float64)

    def _prim_pose_matrix(self, prim_path: str) -> np.ndarray:
        prim = _xform_prim(prim_path)
        position, orientation = prim.get_world_pose()
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _quat_to_matrix(_normalize_quat(np.asarray(orientation, dtype=np.float64)))
        matrix[:3, 3] = np.asarray(position, dtype=np.float64)
        return matrix

    def _base_pose_matrix(self) -> np.ndarray:
        base_position, base_quat = self.robot_root.get_world_pose()
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _quat_to_matrix(_normalize_quat(np.asarray(base_quat, dtype=np.float64)))
        matrix[:3, 3] = np.asarray(base_position, dtype=np.float64)
        return matrix

    def _world_matrix_to_base_pose(self, pose_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose_base = np.linalg.inv(self._base_pose_matrix()) @ np.asarray(pose_world, dtype=np.float64)
        return pose_base[:3, 3], _matrix_to_quat_wxyz(pose_base[:3, :3])

    def _sort_pose_candidates(self, poses_world: np.ndarray) -> np.ndarray:
        current_tcp = self._current_tcp_position_world()
        if current_tcp is None:
            current_tcp = poses_world[0, :3, 3]
        distances = np.linalg.norm(poses_world[:, :3, 3] - current_tcp[np.newaxis, :], axis=1)
        return poses_world[np.argsort(distances)]


def manipulation_config_from_task(task_info: dict) -> ManipulationConfig:
    raw = task_info.get("slam_setting", {}).get("manipulation", {})
    collision = raw.get("local_collision", {})
    return ManipulationConfig(
        enabled=bool(raw.get("enabled", True)),
        arm=str(raw.get("arm", "left")),
        prewarm_on_start=bool(raw.get("prewarm_on_start", ManipulationConfig.prewarm_on_start)),
        prewarm_interaction_assets=bool(
            raw.get("prewarm_interaction_assets", ManipulationConfig.prewarm_interaction_assets)
        ),
        empty_position_base=_tuple3(raw.get("empty_position_base"), ManipulationConfig.empty_position_base),
        empty_quaternion_base=_tuple4(raw.get("empty_quaternion_base"), ManipulationConfig.empty_quaternion_base),
        empty_use_current_orientation=bool(
            raw.get("empty_use_current_orientation", ManipulationConfig.empty_use_current_orientation)
        ),
        grasp_offset_base=_tuple3(raw.get("grasp_offset_base"), ManipulationConfig.grasp_offset_base),
        place_offset_base=_tuple3(raw.get("place_offset_base"), ManipulationConfig.place_offset_base),
        lift_offset_base=_tuple3(raw.get("lift_offset_base"), ManipulationConfig.lift_offset_base),
        grasp_quaternion_base=_tuple4(raw.get("grasp_quaternion_base"), ManipulationConfig.grasp_quaternion_base),
        place_quaternion_base=_tuple4(raw.get("place_quaternion_base"), ManipulationConfig.place_quaternion_base),
        grasp_target=raw.get("grasp_target"),
        place_target=raw.get("place_target"),
        trajopt_steps=int(raw.get("trajopt_steps", ManipulationConfig.trajopt_steps)),
        use_interaction_poses=bool(raw.get("use_interaction_poses", ManipulationConfig.use_interaction_poses)),
        fallback_to_object_offset=bool(
            raw.get("fallback_to_object_offset", ManipulationConfig.fallback_to_object_offset)
        ),
        max_grasp_candidates=int(raw.get("max_grasp_candidates", ManipulationConfig.max_grasp_candidates)),
        max_place_candidates=int(raw.get("max_place_candidates", ManipulationConfig.max_place_candidates)),
        place_align_samples=int(raw.get("place_align_samples", ManipulationConfig.place_align_samples)),
        place_approach_up=float(raw.get("place_approach_up", ManipulationConfig.place_approach_up)),
        grasp_upper_percentile=float(raw.get("grasp_upper_percentile", ManipulationConfig.grasp_upper_percentile)),
        disable_upside_down_grasp=bool(
            raw.get("disable_upside_down_grasp", ManipulationConfig.disable_upside_down_grasp)
        ),
        grasp_vertical_threshold_deg=float(
            raw.get("grasp_vertical_threshold_deg", ManipulationConfig.grasp_vertical_threshold_deg)
        ),
        grasp_reject_towards_robot=bool(
            raw.get("grasp_reject_towards_robot", ManipulationConfig.grasp_reject_towards_robot)
        ),
        grasp_towards_robot_max_dot=float(
            raw.get("grasp_towards_robot_max_dot", ManipulationConfig.grasp_towards_robot_max_dot)
        ),
        grasp_preferred_height_percentile=float(
            raw.get("grasp_preferred_height_percentile", ManipulationConfig.grasp_preferred_height_percentile)
        ),
        grasp_distance_weight=float(raw.get("grasp_distance_weight", ManipulationConfig.grasp_distance_weight)),
        grasp_approach_weight=float(raw.get("grasp_approach_weight", ManipulationConfig.grasp_approach_weight)),
        grasp_height_weight=float(raw.get("grasp_height_weight", ManipulationConfig.grasp_height_weight)),
        grasp_orientation_weight=float(
            raw.get("grasp_orientation_weight", ManipulationConfig.grasp_orientation_weight)
        ),
        grasp_upright_weight=float(raw.get("grasp_upright_weight", ManipulationConfig.grasp_upright_weight)),
        grasp_support_collision=bool(
            raw.get("grasp_support_collision", ManipulationConfig.grasp_support_collision)
        ),
        grasp_support_z_margin=float(raw.get("grasp_support_z_margin", ManipulationConfig.grasp_support_z_margin)),
        grasp_support_xy_margin=float(raw.get("grasp_support_xy_margin", ManipulationConfig.grasp_support_xy_margin)),
        grasp_max_support_obstacles=int(
            raw.get("grasp_max_support_obstacles", ManipulationConfig.grasp_max_support_obstacles)
        ),
        attach_on_close=bool(raw.get("attach_on_close", ManipulationConfig.attach_on_close)),
        attach_distance_threshold=float(
            raw.get("attach_distance_threshold", ManipulationConfig.attach_distance_threshold)
        ),
        attach_max_closed_fraction=float(
            raw.get("attach_max_closed_fraction", ManipulationConfig.attach_max_closed_fraction)
        ),
        detach_on_open_close_fraction=float(
            raw.get("detach_on_open_close_fraction", ManipulationConfig.detach_on_open_close_fraction)
        ),
        detach_on_open_fraction_drop=float(
            raw.get("detach_on_open_fraction_drop", ManipulationConfig.detach_on_open_fraction_drop)
        ),
        detach_on_open_timeout_sec=float(
            raw.get("detach_on_open_timeout_sec", ManipulationConfig.detach_on_open_timeout_sec)
        ),
        disable_attached_target_collisions=bool(
            raw.get("disable_attached_target_collisions", ManipulationConfig.disable_attached_target_collisions)
        ),
        attached_support_clearance=float(
            raw.get("attached_support_clearance", ManipulationConfig.attached_support_clearance)
        ),
        joint_select_candidate_count=int(
            raw.get("joint_select_candidate_count", ManipulationConfig.joint_select_candidate_count)
        ),
        joint_select_reject_threshold=float(
            raw.get("joint_select_reject_threshold", ManipulationConfig.joint_select_reject_threshold)
        ),
        joint_select_path_weight=float(
            raw.get("joint_select_path_weight", ManipulationConfig.joint_select_path_weight)
        ),
        joint_delta_weights=_tuple_float(
            raw.get("joint_delta_weights"),
            ManipulationConfig.joint_delta_weights,
        ),
        lift_force_support_collision=bool(
            raw.get("lift_force_support_collision", ManipulationConfig.lift_force_support_collision)
        ),
        lift_retry_without_collision=bool(
            raw.get("lift_retry_without_collision", ManipulationConfig.lift_retry_without_collision)
        ),
        local_collision=LocalCollisionConfig(
            enabled=bool(collision.get("enabled", True)),
            radius=float(collision.get("radius", LocalCollisionConfig.radius)),
            max_obstacles=int(collision.get("max_obstacles", LocalCollisionConfig.max_obstacles)),
            max_mesh_faces=int(collision.get("max_mesh_faces", LocalCollisionConfig.max_mesh_faces)),
            include_prefixes=tuple(collision.get("include_prefixes", LocalCollisionConfig.include_prefixes)),
        ),
    )


def _make_target_xform(position, quaternion):
    from isaacsim.core.prims import SingleXFormPrim as XFormPrim

    return XFormPrim(
        "/World/target",
        position=np.asarray(position, dtype=np.float64),
        orientation=np.asarray(quaternion, dtype=np.float64),
    )


def _xform_prim(prim_path: str):
    from isaacsim.core.prims import SingleXFormPrim as XFormPrim

    return XFormPrim(prim_path)


def _normalize_quat(quaternion) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_to_matrix(quaternion) -> np.ndarray:
    w, x, y, z = _normalize_quat(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diag_index = int(np.argmax(np.diag(matrix)))
        if diag_index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif diag_index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    return _normalize_quat(np.array([w, x, y, z], dtype=np.float64))


def _normalized_cost(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if value_max - value_min < 1e-8:
        return np.zeros_like(values)
    return (values - value_min) / (value_max - value_min)


def _aabb_corners(min_point: np.ndarray, max_point: np.ndarray) -> np.ndarray:
    min_point = np.asarray(min_point, dtype=np.float64)
    max_point = np.asarray(max_point, dtype=np.float64)
    return np.array(
        [
            [x, y, z]
            for x in (min_point[0], max_point[0])
            for y in (min_point[1], max_point[1])
            for z in (min_point[2], max_point[2])
        ],
        dtype=np.float64,
    )


def _transformed_points_min_z(transform: np.ndarray, points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    transformed = (np.asarray(transform, dtype=np.float64) @ points_h.T).T
    return float(np.min(transformed[:, 2]))


def _wrap_joint_delta(delta) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _lift_path_constraint(lift_offset) -> list[float]:
    """Hold translation axes orthogonal to the requested base-frame lift."""
    offset = np.abs(np.asarray(lift_offset, dtype=np.float64))
    if offset.shape[0] != 3 or float(np.max(offset)) < 1e-8:
        return [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    lift_axis = int(np.argmax(offset))
    position_weights = [1.0, 1.0, 1.0]
    position_weights[lift_axis] = 0.0
    return [0.0, 0.0, 0.0] + position_weights


def _target_spec_value(target_spec, key: str) -> str:
    if isinstance(target_spec, dict):
        return str(target_spec.get(key, "") or "").strip()
    return ""


def _prim_path_exists(prim_path: str) -> bool:
    try:
        import omni.usd

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        return bool(prim and prim.IsValid())
    except Exception:
        return False


def _iter_prim_tree(root_prim):
    yield root_prim
    for child in root_prim.GetChildren():
        yield from _iter_prim_tree(child)


def _normalized_lookup_token(value: str) -> str:
    parts = re.findall(r"[A-Za-z]+|\d+", str(value).lower())
    normalized = []
    for part in parts:
        if part.isdigit():
            normalized.append(str(int(part)))
        else:
            normalized.append(part)
    return "".join(normalized)


def _infer_asset_id_from_prim_path(prim_path: str) -> str:
    asset_id = _asset_id_from_data_info_dir(_data_info_dir_from_prim_references(prim_path))
    if asset_id:
        return asset_id
    name = str(prim_path).rstrip("/").rsplit("/", 1)[-1]
    match = re.match(r"(.+)_\d+$", name)
    if match:
        return match.group(1)
    return name


def _asset_id_from_data_info_dir(data_info_dir: str) -> str:
    if not data_info_dir:
        return ""
    return Path(data_info_dir).parts[-1]


def _data_info_dir_from_prim_references(prim_path: str) -> str:
    for asset_path in _prim_reference_asset_paths(prim_path):
        parts = Path(asset_path.replace("\\", "/")).parts
        if "objects" not in parts:
            continue
        start = parts.index("objects")
        if parts[-1] == "Aligned.usd" and len(parts) > start + 1:
            return str(Path(*parts[start:-1]))
    return ""


def _prim_reference_asset_paths(prim_path: str) -> list[str]:
    try:
        import omni.usd

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        references = prim.GetMetadata("references") if prim and prim.IsValid() else None
    except Exception:
        return []
    if references is None:
        return []
    items = []
    for attr_name in ("prependedItems", "addedItems", "explicitItems"):
        items.extend(list(getattr(references, attr_name, []) or []))
    return [str(getattr(item, "assetPath", "") or "") for item in items if getattr(item, "assetPath", None)]


def _sim_assets_root() -> Path:
    sim_assets = os.environ.get("SIM_ASSETS")
    if not sim_assets:
        raise RuntimeError("SIM_ASSETS is not set; cannot load interaction poses")
    return Path(sim_assets)


def _category_from_asset_id(asset_id: str) -> str:
    match = re.match(r"benchmark_(.+)_\d+$", asset_id)
    return match.group(1) if match else ""


@lru_cache(maxsize=256)
def _infer_data_info_dir(asset_id: str) -> str:
    if not asset_id:
        return ""
    category = _category_from_asset_id(asset_id)
    if category:
        candidate = Path("objects") / "benchmark" / category / asset_id
        if (_sim_assets_root() / candidate / "object_parameters.json").exists():
            return str(candidate)
    objects_root = _sim_assets_root() / "objects"
    for object_dir in objects_root.rglob(asset_id):
        if object_dir.is_dir() and (object_dir / "object_parameters.json").exists():
            return str(object_dir.relative_to(_sim_assets_root()))
    return ""


@lru_cache(maxsize=256)
def _load_object_parameters(asset_id: str, data_info_dir: str) -> dict:
    object_dir = _sim_assets_root() / data_info_dir if data_info_dir else _sim_assets_root() / _infer_data_info_dir(asset_id)
    params_path = object_dir / "object_parameters.json"
    if not params_path.exists():
        logger.warning(f"object_parameters.json not found for {asset_id}: {params_path}")
        return {}
    with open(params_path, "r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=256)
def _load_interaction(asset_id: str, data_info_dir: str) -> dict:
    if not asset_id:
        return {}
    interaction_path = _sim_assets_root() / "interaction" / asset_id / "interaction.json"
    if not interaction_path.exists():
        asset_id = _asset_id_from_data_info_dir(data_info_dir) or asset_id
        interaction_path = _sim_assets_root() / "interaction" / asset_id / "interaction.json"
    if not interaction_path.exists():
        logger.warning(f"interaction.json not found for {asset_id}: {interaction_path}")
        return {}
    with open(interaction_path, "r", encoding="utf-8") as file:
        return json.load(file).get("interaction", {})


@lru_cache(maxsize=256)
def _load_grasp_poses(asset_id: str, data_info_dir: str) -> np.ndarray | None:
    interaction = _load_interaction(asset_id, data_info_dir)
    grasp_spec = interaction.get("passive", {}).get("grasp", {}).get("default", [])
    if isinstance(grasp_spec, str):
        grasp_spec = [grasp_spec]
    poses = []
    for relative_path in grasp_spec:
        pkl_path = _sim_assets_root() / "interaction" / asset_id / str(relative_path)
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as file:
            data = pickle.load(file)
        grasp_pose = np.asarray(data.get("grasp_pose", []), dtype=np.float64)
        if grasp_pose.ndim == 3 and grasp_pose.shape[1:] == (4, 4) and len(grasp_pose) > 0:
            poses.append(grasp_pose)
    if not poses:
        return None
    return np.concatenate(poses, axis=0)


def _interaction_elements(asset_id: str, data_info_dir: str, role: str, action: str) -> list[dict]:
    interaction = _load_interaction(asset_id, data_info_dir)
    action_elements = interaction.get(role, {}).get(action, {})
    elements = []
    for value in action_elements.values():
        if isinstance(value, list):
            elements.extend(value)
        elif isinstance(value, dict):
            elements.append(value)
    return [element for element in elements if isinstance(element, dict)]


def _distance_point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom < 1e-9:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    closest = start + ratio * segment
    return float(np.linalg.norm(point - closest))


def _tuple3(value, default) -> tuple[float, float, float]:
    if value is None:
        return tuple(float(item) for item in default)
    if len(value) != 3:
        raise ValueError("Expected a 3-value vector")
    return tuple(float(item) for item in value)


def _tuple4(value, default) -> tuple[float, float, float, float]:
    if value is None:
        return tuple(float(item) for item in default)
    if len(value) != 4:
        raise ValueError("Expected a 4-value quaternion")
    return tuple(float(item) for item in value)


def _tuple_float(value, default) -> tuple[float, ...]:
    if value is None:
        return tuple(float(item) for item in default)
    return tuple(float(item) for item in value)


def _fit_weights(values, count: int) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if len(weights) == count:
        return weights
    if len(weights) == 0:
        return np.ones(count, dtype=np.float64)
    if len(weights) > count:
        return weights[:count]
    padded = np.ones(count, dtype=np.float64)
    padded[: len(weights)] = weights
    return padded
