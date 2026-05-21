#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Single-process Galbot SLAM keyboard teleop and ROS2 bag recording."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_directory)

from common.base_utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Galbot SLAM keyboard teleop")
    parser.add_argument(
        "--task_config",
        type=str,
        default="source/data_collection/tasks/diy/slam/galbot_slam_home_b.json",
        help="Path to the SLAM task JSON.",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--physics_step", type=int, default=60)
    parser.add_argument("--render_fps", type=int, default=30)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional max runtime in seconds. Defaults to slam_setting.duration_sec or unlimited.",
    )
    return parser.parse_args()


args = parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "disable_viewport_updates": args.headless,
        "renderer": "RayTracedLighting",
        "extra_args": [
            "--/persistent/rtx/modes/rt2/enabled=true",
        ],
    }
)
simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
simulation_app._carb_settings.set("/omni/replicator/asyncRendering", False)
simulation_app._carb_settings.set("/app/asyncRendering", False)

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils import extensions
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, UsdPhysics

extensions.enable_extension("isaacsim.ros2.bridge")

from server.robot import RobotCfg
from slam_collection.base_motion import BaseMotionConfig, OmniBaseMotion
from slam_collection.gripper_control import LeftGripperStateMachine, gripper_control_config_from_task
from slam_collection.keyboard_controller import KeyboardBaseController
from slam_collection.manipulation_control import LeftArmManipulationController
from slam_collection.posture_hold import ArticulationPostureHold, posture_hold_config_from_task
from slam_collection.rosbag_recorder import SlamRosbagRecorder
from slam_collection.vertical_lift import GalbotVerticalLiftController, vertical_lift_config_from_task
from slam_collection.viewport_display import (
    ViewportDisplay,
    clear_viewport_selection,
    head_view_from_config,
    remove_known_debug_prims,
    third_person_view_from_config,
)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(root_directory))
    candidate = os.path.abspath(os.path.join(repo_root, path))
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(os.path.join(root_directory, path))


def resolve_asset_path(relative_path: str) -> str:
    sim_assets_root = os.environ.get("SIM_ASSETS")
    if not sim_assets_root:
        raise RuntimeError("SIM_ASSETS is not set; cannot resolve scene or robot assets")
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(sim_assets_root, relative_path)


def normalize_scene_usd(scene_usd) -> str:
    if isinstance(scene_usd, list):
        candidates = [item for item in scene_usd if isinstance(item, str) and item.strip()]
        if not candidates:
            raise ValueError("scene.scene_usd list is empty")
        return candidates[0]
    if isinstance(scene_usd, str) and scene_usd.strip():
        return scene_usd
    raise ValueError("scene.scene_usd must be a non-empty string or list")


def validate_task_config(task_info: dict) -> None:
    robot_info = task_info.get("robot", {})
    robot_id = str(robot_info.get("robot_id", "")).lower()
    robot_cfg = str(robot_info.get("robot_cfg", "")).lower()
    if robot_id != "galbot" or "galbot" not in robot_cfg:
        raise ValueError("SLAM keyboard teleop currently supports only Galbot robot configs")


def build_motion_config(task_info: dict) -> BaseMotionConfig:
    slam_setting = task_info.get("slam_setting", {})
    return BaseMotionConfig(
        linear_speed=float(slam_setting.get("linear_speed", 0.4)),
        angular_speed=float(slam_setting.get("angular_speed", 0.7)),
        max_linear_accel=float(slam_setting.get("max_linear_accel", 0.8)),
        max_angular_accel=float(slam_setting.get("max_angular_accel", 1.2)),
        command_timeout_sec=float(slam_setting.get("command_timeout_sec", 0.25)),
    )


def collect_joint_pose(task_info: dict) -> dict[str, float]:
    robot_info = task_info.get("robot", {})
    joint_pose = {}
    for key in ("fixed_joint_reset_pose", "init_joint_pose", "init_arm_pose"):
        value = robot_info.get(key)
        if isinstance(value, dict):
            for joint_name, position in value.items():
                if position is not None and np.isfinite(float(position)):
                    joint_pose[joint_name] = float(position)
    return joint_pose


def set_joint_pose(articulation: Articulation, joint_pose: dict[str, float]) -> None:
    if not joint_pose:
        return
    joint_indices = []
    joint_positions = []
    for joint_name, position in joint_pose.items():
        try:
            joint_indices.append(articulation.get_dof_index(joint_name))
            joint_positions.append(position)
        except Exception as exc:
            logger.warning(f"Skip unknown or unavailable joint {joint_name}: {exc}")
    if not joint_indices:
        return
    articulation.set_joint_positions(np.array(joint_positions, dtype=np.float64), joint_indices=joint_indices)
    if hasattr(articulation, "set_joint_velocities"):
        articulation.set_joint_velocities(np.zeros(len(joint_positions)), joint_indices=joint_indices)
    if hasattr(articulation, "set_joint_position_targets"):
        articulation.set_joint_position_targets(
            np.array(joint_positions, dtype=np.float64),
            joint_indices=joint_indices,
        )
    if hasattr(articulation, "set_joint_velocity_targets"):
        articulation.set_joint_velocity_targets(np.zeros(len(joint_positions)), joint_indices=joint_indices)


def ensure_physics_scene() -> None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)


def main() -> int:
    if args.headless:
        raise RuntimeError("Keyboard SLAM teleop requires GUI mode; rerun without --headless")

    task_config_path = resolve_repo_path(args.task_config)
    task_info = load_json(task_config_path)
    validate_task_config(task_info)

    robot_info = task_info["robot"]
    scene_info = task_info["scene"]
    slam_setting = task_info.get("slam_setting", {})

    robot_cfg_path = os.path.join(root_directory, "config", "robot_cfg", robot_info["robot_cfg"])
    robot_cfg = RobotCfg(robot_cfg_path)
    scene_usd = normalize_scene_usd(scene_info["scene_usd"])
    robot_usd_path = resolve_asset_path(robot_cfg.robot_usd)
    scene_usd_path = resolve_asset_path(scene_usd)
    if not os.path.exists(robot_usd_path):
        raise FileNotFoundError(f"Robot USD not found: {robot_usd_path}")
    if not os.path.exists(scene_usd_path):
        raise FileNotFoundError(f"Scene USD not found: {scene_usd_path}")

    physics_dt = 1.0 / float(args.physics_step)
    rendering_dt = 1.0 / float(args.render_fps)
    world = World(
        stage_units_in_meters=1,
        physics_dt=physics_dt,
        rendering_dt=rendering_dt,
        device="cpu",
    )

    add_reference_to_stage(robot_usd_path, robot_cfg.robot_prim_path)
    add_reference_to_stage(scene_usd_path, "/World")
    ensure_physics_scene()
    remove_known_debug_prims()

    robot_init_pose = robot_info.get("robot_init_pose", {})
    initial_position = list(robot_init_pose.get("position", [0.0, 0.0, 0.0]))
    initial_quaternion = list(robot_init_pose.get("quaternion", [1.0, 0.0, 0.0, 0.0]))

    robot_root = XFormPrim(
        prim_path=robot_cfg.robot_prim_path,
        position=initial_position,
        orientation=initial_quaternion,
    )
    articulation = Articulation(prim_path=robot_cfg.robot_prim_path, name=robot_cfg.robot_name)
    world.scene.add(articulation)
    world.reset()
    articulation.initialize()
    robot_root.set_world_pose(position=initial_position, orientation=initial_quaternion)
    set_joint_pose(articulation, collect_joint_pose(task_info))
    left_gripper = LeftGripperStateMachine(
        articulation,
        robot_cfg,
        physics_hz=float(args.physics_step),
        config=gripper_control_config_from_task(task_info),
    )
    manipulation = LeftArmManipulationController(
        world,
        articulation,
        robot_root,
        robot_cfg,
        task_info,
        left_gripper,
    )
    posture_hold = ArticulationPostureHold(articulation, posture_hold_config_from_task(task_info))
    vertical_lift = GalbotVerticalLiftController(articulation, vertical_lift_config_from_task(task_info))
    clear_viewport_selection()
    manipulation.prewarm()

    keyboard = KeyboardBaseController()
    keyboard.start()
    motion = OmniBaseMotion(initial_position, initial_quaternion, build_motion_config(task_info))
    viewport_display = ViewportDisplay(
        third_person_view_from_config(task_info),
        head_view_from_config(task_info),
    )
    viewport_display.initialize(initial_position, initial_quaternion)
    repo_root = os.path.dirname(os.path.dirname(root_directory))
    recorder = SlamRosbagRecorder(
        task_info=task_info,
        task_config_path=task_config_path,
        repo_root=repo_root,
        robot_cfg=robot_cfg,
        scene_usd=scene_usd,
        scene_usd_path=scene_usd_path,
        robot_usd_path=robot_usd_path,
        rendering_dt=rendering_dt,
    )

    control_hz = float(slam_setting.get("control_hz", 60.0))
    control_dt = 1.0 / max(control_hz, 1.0)
    duration = args.duration
    if duration is None and slam_setting.get("duration_sec") is not None:
        duration = float(slam_setting["duration_sec"])

    logger.info("Galbot SLAM keyboard teleop started")
    logger.info("Controls: W/S forward/back, A/D strafe, Q/E yaw, R/F lift up/down, Space brake, X/Esc quit")
    logger.info("Recording: B start rosbag, N stop rosbag")
    logger.info("Manipulation: H empty move, J move grasp, K move place, U open, I close, L lift, O reset")
    logger.info(f"Task config: {task_config_path}")

    start_time = time.monotonic()
    last_step_time = start_time
    last_control_time = 0.0
    frame = 0

    try:
        if recorder.prewarm_publishers:
            logger.info("Prewarming SLAM ROS publishers before keyboard control loop")
            recorder.initialize()

        while simulation_app.is_running():
            now = time.monotonic()
            dt = now - last_step_time
            last_step_time = now
            command = keyboard.command()
            for action in keyboard.consume_actions():
                if action == "record_start":
                    recorder.start()
                elif action == "record_stop":
                    recorder.stop()
                else:
                    manipulation.handle_action(action)
            if command.quit:
                logger.info("Quit requested from keyboard")
                break
            if (
                abs(command.forward) > 0.0
                or abs(command.strafe) > 0.0
                or abs(command.yaw) > 0.0
                or abs(command.vertical) > 0.0
                or command.brake
            ):
                motion.mark_command_time()

            if now - last_control_time >= control_dt:
                control_step_dt = now - last_control_time if last_control_time else dt
                position, orientation = motion.step(command, control_step_dt)
                robot_root.set_world_pose(position=position, orientation=orientation)
                vertical_targets = vertical_lift.step(command, control_step_dt)
                if vertical_targets is not None:
                    vertical_joint_names, vertical_joint_positions = vertical_targets
                    posture_hold.update_targets(vertical_joint_names, vertical_joint_positions)
                viewport_display.update(position, orientation)
                clear_viewport_selection()
                last_control_time = now

            manipulation.step()
            if manipulation.consume_completed_arm_motion():
                posture_hold.update_targets_from_current(manipulation.arm_joint_names())
            posture_hold.set_suspended_joint_names(manipulation.suspended_joint_names())
            posture_hold.apply()
            world.step(render=False)
            if frame % max(1, int(args.physics_step / args.render_fps)) == 0:
                world.render()
                recorder.tick(world.current_time)

            if duration is not None and now - start_time >= duration:
                logger.info(f"Duration reached: {duration:.2f}s")
                break
            frame += 1
    finally:
        try:
            recorder.shutdown()
        finally:
            keyboard.stop()
            simulation_app.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logger.error(f"SLAM keyboard teleop failed: {exc}")
        simulation_app.close()
        raise
