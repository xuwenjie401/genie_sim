# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import copy
import glob
import json
import os
import random
import sys
from datetime import datetime

root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(os.path.dirname(root_directory))
sys.path.append(root_directory)

from client.agent.omniagent import DataCollectionAgent
from client.layout.task_generate import TaskGenerator
from client.robot.omni_robot import IsaacSimRpcRobot
from common.base_utils.logger import logger

META_SKIPPED_TASK_STATUSES = {
    "skipped_invalid",
    "skipped_incompatible",
    "failed_to_generate",
}


def _load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def _now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _relativize_path(path):
    if not path:
        return ""
    abs_path = os.path.abspath(path)
    try:
        relative_path = os.path.relpath(abs_path, repo_root)
    except ValueError:
        return abs_path
    if relative_path.startswith(".."):
        return abs_path
    return relative_path


def _write_json_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)


def _task_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def _normalize_scene_usd_candidates(scene_usd):
    if isinstance(scene_usd, list):
        return [value for value in scene_usd if isinstance(value, str) and value.strip()]
    if isinstance(scene_usd, str) and scene_usd.strip():
        return [scene_usd]
    return []


def _resolve_task_template_path(task_template, meta_task_path):
    if not isinstance(task_template, str) or not task_template.strip():
        return None

    normalized_task_template = task_template.strip().replace("\\", "/")
    if os.path.isabs(normalized_task_template):
        return os.path.abspath(normalized_task_template) if os.path.exists(normalized_task_template) else None

    source_prefix = "source/data_collection/"
    candidate_paths = [
        os.path.abspath(os.path.join(os.path.dirname(meta_task_path), normalized_task_template)),
        os.path.abspath(os.path.join(root_directory, normalized_task_template)),
        os.path.abspath(os.path.join(repo_root, normalized_task_template)),
    ]
    if normalized_task_template.startswith(source_prefix):
        candidate_paths.append(
            os.path.abspath(os.path.join(root_directory, normalized_task_template[len(source_prefix) :]))
        )

    checked_paths = set()
    for candidate_path in candidate_paths:
        if candidate_path in checked_paths:
            continue
        checked_paths.add(candidate_path)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def _parse_meta_task_entries(meta_task_info):
    raw_tasks = meta_task_info.get("tasks", [])
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("meta task config requires a non-empty tasks list")

    parsed_entries = []
    for index, raw_task in enumerate(raw_tasks):
        if isinstance(raw_task, dict) and "task_template" in raw_task:
            parsed_entries.append(
                {
                    "index": index,
                    "task_template": raw_task.get("task_template"),
                    "episodes": raw_task.get("episodes"),
                }
            )
            continue

        if isinstance(raw_task, dict):
            logger.warning(
                f"Legacy meta-task mapping detected at tasks[{index}]. "
                "Use task_template/episodes objects for new configs."
            )
            for task_template, episodes in raw_task.items():
                parsed_entries.append(
                    {
                        "index": index,
                        "task_template": task_template,
                        "episodes": episodes,
                    }
                )
            continue

        raise ValueError(f"Unsupported meta task entry at tasks[{index}]: {raw_task}")
    return parsed_entries


def _extract_task_signature(task_info):
    scene_info = task_info.get("scene", {})
    robot_info = task_info.get("robot", {})
    scene_usd_candidates = _normalize_scene_usd_candidates(scene_info.get("scene_usd"))

    required_fields = {
        "scene.scene_id": scene_info.get("scene_id"),
        "scene.scene_info_dir": scene_info.get("scene_info_dir"),
        "robot.robot_id": robot_info.get("robot_id"),
        "robot.robot_cfg": robot_info.get("robot_cfg"),
    }
    missing_fields = [name for name, value in required_fields.items() if not value]
    if not scene_usd_candidates:
        missing_fields.append("scene.scene_usd")
    if missing_fields:
        raise ValueError(f"missing required fields: {', '.join(missing_fields)}")

    return {
        "scene_id": scene_info["scene_id"],
        "scene_info_dir": scene_info["scene_info_dir"],
        "scene_usd_candidates": scene_usd_candidates,
        "robot_id": robot_info["robot_id"],
        "robot_cfg": robot_info["robot_cfg"],
    }


def _normalize_episode_count(episodes):
    if isinstance(episodes, bool):
        raise ValueError("episodes must be a positive integer")
    episode_count = int(episodes)
    if episode_count <= 0:
        raise ValueError("episodes must be a positive integer")
    return episode_count


def _build_runtime_task_info(task_info, episodes, locked_scene_usd):
    runtime_task_info = copy.deepcopy(task_info)
    runtime_task_info.setdefault("recording_setting", {})
    runtime_task_info["recording_setting"]["num_of_episode"] = episodes
    runtime_task_info.setdefault("scene", {})
    runtime_task_info["scene"]["scene_usd"] = locked_scene_usd
    return runtime_task_info


def _build_task_state(task_template, runtime_task_id, episodes):
    return {
        "task_template": task_template,
        "resolved_task_template": "",
        "runtime_task_id": runtime_task_id,
        "requested_episodes": episodes,
        "generated_episodes": 0,
        "attempted_episodes": 0,
        "successful_episodes": 0,
        "failed_episodes": 0,
        "status": "pending",
        "skip_reason": "",
        "saved_task_dir": "",
    }


def _prepare_meta_task_entries(meta_task_path, meta_task_info):
    task_states = []
    prepared_entries = []
    locked_scene_usd = None
    baseline_signature = None

    for order, entry in enumerate(_parse_meta_task_entries(meta_task_info)):
        raw_task_template = entry["task_template"]
        runtime_task_id = _task_id_from_path(str(raw_task_template))
        task_state = _build_task_state(str(raw_task_template), runtime_task_id, entry["episodes"])
        task_states.append(task_state)

        try:
            episode_count = _normalize_episode_count(entry["episodes"])
            task_state["requested_episodes"] = episode_count
        except (TypeError, ValueError) as exc:
            task_state["status"] = "skipped_invalid"
            task_state["skip_reason"] = str(exc)
            logger.warning(f"Skip meta-task child {raw_task_template}: {exc}")
            continue

        resolved_task_template = _resolve_task_template_path(raw_task_template, meta_task_path)
        if not resolved_task_template:
            task_state["status"] = "skipped_invalid"
            task_state["skip_reason"] = "Task template file not found"
            logger.warning(f"Skip meta-task child {raw_task_template}: task template file not found")
            continue

        task_state["resolved_task_template"] = _relativize_path(resolved_task_template)
        try:
            task_info = _load_json(resolved_task_template)
            task_signature = _extract_task_signature(task_info)
        except Exception as exc:
            task_state["status"] = "skipped_invalid"
            task_state["skip_reason"] = str(exc)
            logger.warning(f"Skip meta-task child {raw_task_template}: {exc}")
            continue

        if baseline_signature is None:
            baseline_signature = {
                "scene_id": task_signature["scene_id"],
                "scene_info_dir": task_signature["scene_info_dir"],
                "robot_id": task_signature["robot_id"],
                "robot_cfg": task_signature["robot_cfg"],
            }
            locked_scene_usd = random.choice(task_signature["scene_usd_candidates"])
            logger.info(f"Locked meta-task scene_usd: {locked_scene_usd}")
        else:
            incompatible_reasons = []
            for key in ("scene_id", "scene_info_dir", "robot_id", "robot_cfg"):
                if task_signature[key] != baseline_signature[key]:
                    incompatible_reasons.append(
                        f"{key} mismatch ({task_signature[key]} != {baseline_signature[key]})"
                    )
            if locked_scene_usd not in task_signature["scene_usd_candidates"]:
                incompatible_reasons.append(f"scene_usd does not include locked scene {locked_scene_usd}")
            if incompatible_reasons:
                task_state["status"] = "skipped_incompatible"
                task_state["skip_reason"] = "; ".join(incompatible_reasons)
                logger.warning(
                    f"Skip incompatible meta-task child {raw_task_template}: {task_state['skip_reason']}"
                )
                continue

        prepared_entries.append(
            {
                "order": order,
                "runtime_task_id": runtime_task_id,
                "resolved_task_template": resolved_task_template,
                "task_info": _build_runtime_task_info(task_info, episode_count, locked_scene_usd),
                "state": task_state,
            }
        )

    return prepared_entries, task_states, locked_scene_usd


def _create_meta_progress(meta_task_path, locked_scene_usd, task_states):
    timestamp = _now_iso()
    return {
        "meta_task_file": _relativize_path(meta_task_path),
        "status": "pending",
        "started_at": timestamp,
        "updated_at": timestamp,
        "locked_scene_usd": locked_scene_usd,
        "tasks": task_states,
        "totals": {},
    }


def _build_meta_progress_path(meta_task_path, meta_task_name, timestamp):
    progress_dir = os.path.join(os.path.dirname(meta_task_path), "progress")
    return os.path.join(progress_dir, f"{meta_task_name}_progress_{timestamp}.json")


def _recalculate_progress_totals(progress):
    totals = {
        "requested_episodes": 0,
        "generated_episodes": 0,
        "attempted_episodes": 0,
        "successful_episodes": 0,
        "failed_episodes": 0,
        "skipped_tasks": 0,
    }
    for task_state in progress["tasks"]:
        totals["requested_episodes"] += int(task_state.get("requested_episodes", 0))
        totals["generated_episodes"] += int(task_state.get("generated_episodes", 0))
        totals["attempted_episodes"] += int(task_state.get("attempted_episodes", 0))
        totals["successful_episodes"] += int(task_state.get("successful_episodes", 0))
        totals["failed_episodes"] += int(task_state.get("failed_episodes", 0))
        if task_state.get("status") in META_SKIPPED_TASK_STATUSES:
            totals["skipped_tasks"] += 1
    progress["totals"] = totals


def _flush_meta_progress(progress_path, progress, status=None, finished=False):
    if status is not None:
        progress["status"] = status
    progress["updated_at"] = _now_iso()
    if finished:
        progress["finished_at"] = progress["updated_at"]
    _recalculate_progress_totals(progress)
    _write_json_atomic(progress_path, progress)


def _determine_meta_final_status(task_states):
    runnable_task_states = [state for state in task_states if state.get("status") not in META_SKIPPED_TASK_STATUSES]
    if not runnable_task_states:
        return "failed"
    if any(state.get("status") in META_SKIPPED_TASK_STATUSES for state in task_states):
        return "completed_with_skips"
    return "completed"


def _generate_task_bundle(task_info, task_folder, runtime_task_id):
    task_generator = TaskGenerator(task_info)
    task_generator.generate_tasks(
        save_path=task_folder,
        task_num=task_info["recording_setting"]["num_of_episode"],
        task_name=runtime_task_id,
    )
    generated_tasks = sorted(glob.glob(os.path.join(task_folder, "*.json")))
    return task_generator, generated_tasks


def _create_robot(task_info, startup_task_info, client_host):
    startup_robot_info = startup_task_info.get("robot", task_info.get("robot", {}))
    startup_robot_pose = startup_robot_info.get("robot_init_pose", {})
    robot_position = startup_robot_pose.get("position", [0, 0, 0])
    robot_rotation = startup_robot_pose.get("quaternion", [1, 0, 0, 0])
    stand = {"stand_type": "cylinder", "stand_size_x": 0.1, "stand_size_y": 0.1}
    robot_init_arm_pose = None
    fixed_joint_reset_pose = None
    robot_init_arm_pose_noise = None
    robot_cfg = startup_robot_info.get("robot_cfg", task_info["robot"]["robot_cfg"])
    if "stand" in startup_robot_info:
        stand = startup_robot_info["stand"]
    if "init_arm_pose" in startup_robot_info:
        robot_init_arm_pose = startup_robot_info["init_arm_pose"]
    if "fixed_joint_reset_pose" in startup_robot_info:
        fixed_joint_reset_pose = startup_robot_info["fixed_joint_reset_pose"]
    elif "init_joint_pose" in startup_robot_info:
        fixed_joint_reset_pose = startup_robot_info["init_joint_pose"]
    if "init_arm_pose_noise" in startup_robot_info:
        robot_init_arm_pose_noise = startup_robot_info["init_arm_pose_noise"]

    return IsaacSimRpcRobot(
        robot_cfg=robot_cfg,
        scene_usd=startup_task_info.get("scene_usd", task_info["scene"]["scene_usd"]),
        client_host=client_host,
        position=robot_position,
        rotation=robot_rotation,
        stand_type=stand["stand_type"],
        stand_size_x=stand["stand_size_x"],
        stand_size_y=stand["stand_size_y"],
        robot_init_arm_pose=robot_init_arm_pose,
        fixed_joint_reset_pose=fixed_joint_reset_pose,
        robot_init_arm_pose_noise=robot_init_arm_pose_noise,
    )


def _run_task_folder(agent, task_folder, task_generator, task_info, use_recording, episode_result_callback=None):
    recording_setting = task_info.get("recording_setting", {})
    return agent.run(
        task_folder=task_folder,
        camera_list=recording_setting["camera_list"],
        use_recording=use_recording,
        workspaces=task_generator.workspaces_in_world_frame,
        fps=recording_setting["fps"],
        render_semantic=recording_setting.get("render_semantic", False),
        origin_task_info=task_info,
        episode_result_callback=episode_result_callback,
    )


def _run_single_task(args, task_template_file):
    task_info = _load_json(task_template_file)
    task_folder = os.path.join(os.getcwd(), "saved_task", task_info["task"])
    task_generator, generated_tasks = _generate_task_bundle(task_info, task_folder, task_info["task"])
    if not generated_tasks:
        raise RuntimeError(f"No generated tasks found in {task_folder}")

    startup_task_info = _load_json(generated_tasks[0])
    robot = _create_robot(task_info, startup_task_info, args.client_host)
    try:
        agent = DataCollectionAgent(robot)
        _run_task_folder(
            agent=agent,
            task_folder=task_folder,
            task_generator=task_generator,
            task_info=task_info,
            use_recording=args.use_recording,
        )
        logger.info("job done")
    finally:
        robot.client.exit()


def _run_meta_task(args, meta_task_path, meta_task_info):
    prepared_entries, task_states, locked_scene_usd = _prepare_meta_task_entries(meta_task_path, meta_task_info)
    meta_task_name = _task_id_from_path(meta_task_path)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    progress_path = _build_meta_progress_path(meta_task_path, meta_task_name, timestamp)
    progress = _create_meta_progress(meta_task_path, locked_scene_usd, task_states)
    _flush_meta_progress(progress_path, progress, status="pending")

    if not prepared_entries:
        _flush_meta_progress(progress_path, progress, status="failed", finished=True)
        raise RuntimeError(f"No runnable child tasks found in meta task {meta_task_path}")

    robot = None
    agent = None
    try:
        for entry in prepared_entries:
            task_state = entry["state"]
            task_state["status"] = "running"
            task_folder = os.path.join(
                os.getcwd(),
                "saved_task",
                meta_task_name,
                f"{entry['order']:02d}_{entry['runtime_task_id']}",
            )
            task_state["saved_task_dir"] = _relativize_path(task_folder)
            _flush_meta_progress(progress_path, progress, status="running")

            task_generator, generated_tasks = _generate_task_bundle(
                entry["task_info"],
                task_folder,
                entry["runtime_task_id"],
            )
            task_state["generated_episodes"] = len(generated_tasks)
            _flush_meta_progress(progress_path, progress, status="running")

            if not generated_tasks:
                task_state["status"] = "failed_to_generate"
                task_state["skip_reason"] = "No generated task json files were created"
                logger.error(
                    f"Meta-task child {entry['resolved_task_template']} did not generate any runnable task json files"
                )
                _flush_meta_progress(progress_path, progress, status="running")
                continue

            if robot is None:
                startup_task_info = _load_json(generated_tasks[0])
                robot = _create_robot(entry["task_info"], startup_task_info, args.client_host)
                agent = DataCollectionAgent(robot)

            def _episode_result_callback(result, state=task_state):
                episode_status = result.get("status", "failed")
                if episode_status in {"success", "failed"}:
                    state["attempted_episodes"] += 1
                if episode_status == "success":
                    state["successful_episodes"] += 1
                else:
                    state["failed_episodes"] += 1
                _flush_meta_progress(progress_path, progress, status="running")

            _run_task_folder(
                agent=agent,
                task_folder=task_folder,
                task_generator=task_generator,
                task_info=entry["task_info"],
                use_recording=args.use_recording,
                episode_result_callback=_episode_result_callback,
            )
            task_state["status"] = "completed"
            _flush_meta_progress(progress_path, progress, status="running")

        final_status = _determine_meta_final_status(task_states)
        _flush_meta_progress(progress_path, progress, status=final_status, finished=True)
        logger.info(f"job done, progress saved to {progress_path}")
    except Exception:
        _flush_meta_progress(progress_path, progress, status="failed", finished=True)
        raise
    finally:
        if robot is not None:
            robot.client.exit()


def main():
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

    task_template_file = os.path.abspath(args.task_template)
    task_info = _load_json(task_template_file)
    if task_info.get("meta_task", False):
        _run_meta_task(args, task_template_file, task_info)
    else:
        _run_single_task(args, task_template_file)


if __name__ == "__main__":
    main()
