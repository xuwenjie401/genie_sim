"""Evaluation runner for pi0-style policies on data-collection tasks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import time
import traceback

from policy_evaluation.adapters.single_arm_7dof import SingleArm7DofAdapter
from policy_evaluation.config import EvalConfig, TaskSpec
from policy_evaluation.env import DataCollectionTaskEnv
from policy_evaluation.policy.pi0_client import Pi0PolicyClient
from policy_evaluation.task_instance import TaskInstanceGenerator
from policy_evaluation.task_loader import PROJECT_ROOT, load_json, resolve_path
from policy_evaluation.video import MultiViewVideoRecorder
from policy_evaluation.world import EvalWorld


class EvaluationRunner:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(config.output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        self.output_dir = output_root / config.name / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir = self.output_dir / "generated"
        self.results_path = self.output_dir / "results.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.instance_generator = TaskInstanceGenerator(self.generated_dir)

    def run(self) -> dict[str, Any]:
        all_results = []
        total_episodes = sum(task.episodes for task in self.config.tasks)
        print(
            f"[policy_eval] run_id={self.run_id} output_dir={self.output_dir} "
            f"policy={self.config.policy.host}:{self.config.policy.port} episodes={total_episodes}",
            flush=True,
        )
        for task in self.config.tasks:
            for episode_idx in range(task.episodes):
                print(
                    f"[policy_eval] start task={task.display_name} episode={episode_idx + 1}/{task.episodes}",
                    flush=True,
                )
                result = self._run_episode(task, episode_idx)
                all_results.append(result)
                self._append_result(result)
                self._write_summary(all_results)
                status = "SUCCESS" if result["success"] else "FAIL"
                print(
                    f"[policy_eval] end task={task.display_name} episode={episode_idx + 1}/{task.episodes} "
                    f"status={status} steps={result['steps']} "
                    f"termination={result['termination_reason']} error={result['error'] or '<none>'}",
                    flush=True,
                )
        summary = self._build_summary(all_results)
        self._write_summary(all_results)
        return summary

    def _run_episode(self, task: TaskSpec, episode_idx: int) -> dict[str, Any]:
        task_path = resolve_path(task.task_template)
        prompt = task.prompt or self.config.prompt
        episode_name = f"{task.display_name}_ep{episode_idx:06d}"
        episode_dir = self.output_dir / "episodes" / _safe_name(episode_name)
        episode_dir.mkdir(parents=True, exist_ok=True)

        instance_path: Path | None = None
        success = False
        check_details = {}
        started_at = time.time()
        steps = 0
        error = ""
        termination_reason = "max_steps"
        arm_reset_status: dict[str, Any] = {"enabled": False, "ready": False, "terminated": False}
        success_seen = False
        first_success_step: int | None = None
        last_success_step: int | None = None
        success_seen_details: dict[str, Any] = {}
        videos = {}
        policy = None
        video = None
        try:
            instance_path = self.instance_generator.generate(task_path, episode_idx, task.display_name)
            task_instance = load_json(instance_path)
            world = EvalWorld(
                physics_step=self.config.physics_step,
                rendering_step=self.config.rendering_step,
                render=self.config.render,
                config=self.config,
            )
            world.setup_episode(task_instance, self.config.adapter, self.config.cameras)
            adapter = SingleArm7DofAdapter(self.config.adapter)
            env = DataCollectionTaskEnv(world, task_instance, adapter, self.config)
            if self.config.initial_wait_sec > 0:
                print(
                    f"[policy_eval] wait {self.config.initial_wait_sec:.1f}s after scene setup before inference",
                    flush=True,
                )
                world.step_seconds(self.config.initial_wait_sec)
            policy = Pi0PolicyClient(
                host=self.config.policy.host,
                port=self.config.policy.port,
                timeout_sec=self.config.policy.timeout_sec,
                log_path=episode_dir / "policy_io.jsonl",
            )
            video = MultiViewVideoRecorder(
                episode_dir / "video",
                fps=self.config.video.fps,
                enabled=self.config.video.enabled,
            )
            observation = env.reset()

            def record_video_frame() -> None:
                views = env.capture_video_views()
                video.add("head", views.get("head"))
                video.add("wrist", views.get("wrist"))
                video.add("observer", views.get("observer"))

            for step in range(self.config.max_steps):
                payload = adapter.build_payload(observation, prompt)
                action = policy.act(payload, step=step)
                observation = env.step(
                    action,
                    video_frame_callback=record_video_frame if self.config.video.enabled else None,
                    record_action_frame=step % self.config.video.every_n_steps == 0,
                )
                steps = step + 1
                arm_reset_status = env.check_arm_reset_termination(observation)
                if arm_reset_status.get("terminated"):
                    termination_reason = "arm_reset"
                    check = env.final_check()
                    success = check.success
                    check_details = dict(check.details)
                    check_details["arm_reset_termination"] = arm_reset_status
                    print(
                        f"[policy_eval] arm reset termination at step={steps} "
                        f"max_abs_error={arm_reset_status['max_abs_error']:.4f} "
                        f"max_seen_error={arm_reset_status['max_seen_error']:.4f}",
                        flush=True,
                    )
                    break
                grasp_lost_status = env.check_grasp_lost_termination(steps)
                if grasp_lost_status.get("terminated"):
                    if success_seen:
                        continue
                    check = env.final_check()
                    check_details = dict(check.details)
                    if check.success:
                        success_seen = True
                        last_success_step = steps
                        success_seen_details = check_details
                        if first_success_step is None:
                            first_success_step = steps
                    else:
                        termination_reason = "grasp_lost"
                        success = False
                        check_details["grasp_lost_termination"] = grasp_lost_status
                        print(
                            f"[policy_eval] grasp lost termination at step={steps} "
                            f"distance={grasp_lost_status['distance']:.4f} "
                            f"threshold={grasp_lost_status['threshold']:.4f}",
                            flush=True,
                        )
                        break
                if self.config.success_check_interval > 0 and steps % self.config.success_check_interval == 0:
                    check = env.final_check()
                    check_details = dict(check.details)
                    if check.success:
                        success_seen = True
                        last_success_step = steps
                        success_seen_details = check_details
                        if first_success_step is None:
                            first_success_step = steps
                            print(
                                f"[policy_eval] task success condition first observed at step={steps}; "
                                "continuing until an episode termination condition",
                                flush=True,
                            )
            if termination_reason not in ("arm_reset", "grasp_lost"):
                check = env.final_check()
                success = check.success
                check_details = dict(check.details)
            if success_seen:
                check_details["success_seen"] = {
                    "first_step": first_success_step,
                    "last_step": last_success_step,
                    "details": success_seen_details,
                }
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            termination_reason = "error"
            check_details = {"traceback": traceback.format_exc()}
        finally:
            if video is not None:
                videos = video.close()
            if policy is not None:
                policy.close()

        return {
            "single_task": str(task_path),
            "single_task_name": task.display_name,
            "episode": episode_idx,
            "success": bool(success),
            "success_seen": bool(success_seen),
            "first_success_step": first_success_step,
            "last_success_step": last_success_step,
            "steps": steps,
            "termination_reason": termination_reason,
            "prompt": prompt,
            "duration": time.time() - started_at,
            "instance_json": str(instance_path) if instance_path is not None else "",
            "episode_dir": str(episode_dir),
            "policy_log": str(episode_dir / "policy_io.jsonl"),
            "videos": videos,
            "check": check_details,
            "error": error,
        }

    def _append_result(self, result: dict[str, Any]) -> None:
        with self.results_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _write_summary(self, results: list[dict[str, Any]]) -> None:
        summary = self._build_summary(results)
        with self.summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

    def _build_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        by_task: dict[str, dict[str, Any]] = {}
        for result in results:
            key = result["single_task_name"]
            stats = by_task.setdefault(key, {"episodes": 0, "success": 0, "success_rate": 0.0})
            stats["episodes"] += 1
            stats["success"] += int(bool(result["success"]))
        for stats in by_task.values():
            stats["success_rate"] = stats["success"] / stats["episodes"] if stats["episodes"] else 0.0
        total = len(results)
        successes = sum(int(bool(item["success"])) for item in results)
        return {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "overall": {
                "episodes": total,
                "success": successes,
                "success_rate": successes / total if total else 0.0,
            },
            "by_single_task": by_task,
        }


def _safe_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("_", "-", ".") else "_" for char in name)
    return safe.strip("_") or "episode"
