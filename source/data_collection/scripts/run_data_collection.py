# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import glob
import json
import os
import sys

root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_directory)
from client.agent.omniagent import DataCollectionAgent
from client.layout.task_generate import TaskGenerator
from client.robot.omni_robot import IsaacSimRpcRobot
from common.base_utils.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimGraspingAgent Command Line Interface")
    parser.add_argument(
        "--client_host",
        type=str,
        default="localhost:50051",
        help="The client host for SimGraspingAgent (default: localhost:50051)",
    )
    parser.add_argument(
        "--use_recording",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--task_template",
        type=str,
        default="tasks/task.json",
        help="",
    )
    args = parser.parse_args()
    task_template_file = args.task_template
    with open(task_template_file, "r") as file:
        task_info = json.load(file)

    # generate task info
    task_generator = TaskGenerator(task_info)
    task_folder = "saved_task/%s" % (task_info["task"])
    task_generator.generate_tasks(
        save_path=task_folder,
        task_num=task_info["recording_setting"]["num_of_episode"],
        task_name=task_info["task"],
    )
    generated_tasks = sorted(glob.glob(os.path.join(task_folder, "*.json")))
    if not generated_tasks:
        raise RuntimeError(f"No generated tasks found in {task_folder}")
    with open(generated_tasks[0], "r") as file:
        startup_task_info = json.load(file)
    startup_robot_info = startup_task_info.get("robot", task_info.get("robot", {}))
    startup_robot_pose = startup_robot_info.get("robot_init_pose", {})
    robot_position = startup_robot_pose.get("position", [0, 0, 0])
    robot_rotation = startup_robot_pose.get("quaternion", [1, 0, 0, 0])
    stand = {"stand_type": "cylinder", "stand_size_x": 0.1, "stand_size_y": 0.1}
    robot_init_arm_pose = None
    robot_init_arm_pose_noise = None
    robot_cfg = startup_robot_info.get("robot_cfg", task_info["robot"]["robot_cfg"])
    if "stand" in startup_robot_info:
        stand = startup_robot_info["stand"]
    if "init_arm_pose" in startup_robot_info:
        robot_init_arm_pose = startup_robot_info["init_arm_pose"]
    if "init_arm_pose_noise" in startup_robot_info:
        robot_init_arm_pose_noise = startup_robot_info["init_arm_pose_noise"]

    robot = IsaacSimRpcRobot(
        robot_cfg=robot_cfg,
        scene_usd=startup_task_info.get("scene_usd", task_info["scene"]["scene_usd"]),
        client_host=args.client_host,
        position=robot_position,
        rotation=robot_rotation,
        stand_type=stand["stand_type"],
        stand_size_x=stand["stand_size_x"],
        stand_size_y=stand["stand_size_y"],
        robot_init_arm_pose=robot_init_arm_pose,
        robot_init_arm_pose_noise=robot_init_arm_pose_noise,
    )
    agent = DataCollectionAgent(robot)
    render_semantic = False
    if "render_semantic" in task_info["recording_setting"]:
        render_semantic = task_info["recording_setting"]["render_semantic"]
    agent.run(
        task_folder=task_folder,
        camera_list=task_info["recording_setting"]["camera_list"],
        use_recording=args.use_recording,
        workspaces=task_generator.workspaces_in_world_frame,
        fps=task_info["recording_setting"]["fps"],
        render_semantic=render_semantic,
        origin_task_info=task_info,
    )
    logger.info("job done")
    robot.client.exit()
