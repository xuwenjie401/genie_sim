"""Configuration helpers for policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PolicyConfig:
    host: str = "localhost"
    port: int = 8181
    timeout_sec: float = 30.0


@dataclass
class AdapterConfig:
    arm: str = "left"
    state_dim: int = 8
    action_arm_dim: int = 7
    action_gripper_index: int = 7
    action_open_threshold: float = 0.5
    gripper_state_mode: str = "raw_mean"
    image_keys: dict[str, str] = field(
        default_factory=lambda: {
            "head": "cam_head",
            "wrist": "cam_left",
            "right_wrist": "cam_right",
        }
    )
    joint_names: list[str] = field(default_factory=list)
    gripper_joint_names: list[str] = field(default_factory=list)
    open_gripper_positions: list[float] = field(default_factory=list)
    closed_gripper_positions: list[float] = field(default_factory=list)


@dataclass
class CameraConfig:
    head: str = ""
    wrist: str = ""
    right_policy: str = "zeros_like_left"
    observer: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoConfig:
    enabled: bool = True
    fps: int = 20
    every_n_steps: int = 1


@dataclass
class TaskSpec:
    task_template: str
    episodes: int
    prompt: str
    name: str = ""

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        return Path(self.task_template).name


@dataclass
class EvalConfig:
    name: str = "policy_eval"
    output_dir: str = "output/policy_evaluation"
    prompt: str = ""
    max_steps: int = 600
    success_check_interval: int = 10
    control_hz: int = 30
    physics_step: int = 60
    rendering_step: int = 30
    initial_wait_sec: float = 10.0
    enable_distractors: bool = False
    terminate_on_arm_reset: bool = False
    arm_reset_tolerance: float = 0.1
    arm_reset_away_threshold: float = 0.25
    arm_reset_consecutive_steps: int = 3
    terminate_on_grasp_lost: bool = False
    grasp_steps_threshold: int = 0
    grasp_judge_distance_threshold: float = 0.25
    wait_for_gripper_close_hold: bool = False
    gripper_close_hold_timeout_sec: float = 6.0
    gripper_close_max_force: float = 0.2
    gripper_hold_stiffness: float = 10000.0
    gripper_hold_max_force: float = 10.0
    gripper_static_friction: float | None = None
    gripper_dynamic_friction: float | None = None
    object_static_friction: float | None = None
    object_dynamic_friction: float | None = None
    target_object_mass_override: float | None = None
    object_mass_overrides: dict[str, float] = field(default_factory=dict)
    disable_object_sleep: bool = True
    headless: bool = False
    render: bool = True
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    cameras: CameraConfig = field(default_factory=CameraConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    tasks: list[TaskSpec] = field(default_factory=list)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_path(path: str | Path, config_dir: Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    roots = []
    if config_dir is not None:
        roots.append(config_dir)
    roots.extend([Path.cwd(), PROJECT_ROOT])

    seen = set()
    candidates = []
    for root_path in roots:
        candidate = (root_path / path).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_tasks(raw: dict[str, Any], base_prompt: str) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    if "tasks" in raw:
        for item in raw["tasks"]:
            prompt = item.get("prompt", base_prompt)
            tasks.append(
                TaskSpec(
                    task_template=item["task_template"],
                    episodes=int(item.get("episodes", raw.get("episodes", 1))),
                    prompt=prompt,
                    name=item.get("name", ""),
                )
            )
        return tasks

    if "task_template" in raw:
        tasks.append(
            TaskSpec(
                task_template=raw["task_template"],
                episodes=int(raw.get("episodes", 1)),
                prompt=raw.get("prompt", base_prompt),
                name=raw.get("name", ""),
            )
        )
        return tasks

    raise ValueError("Evaluation config must contain either 'tasks' or 'task_template'.")


def _merge_meta_task(raw: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """Allow configs that point at an existing data-collection meta_task file."""
    meta_task_path = raw.get("meta_task_file")
    if not meta_task_path:
        return raw

    meta_task_path = _resolve_path(meta_task_path, config_dir=config_dir)
    meta = _load_json(meta_task_path)
    if not meta.get("meta_task"):
        raise ValueError(f"meta_task_file is not a meta task: {meta_task_path}")

    merged = dict(raw)
    merged.setdefault("tasks", meta.get("tasks", []))
    return merged


def load_eval_config(path: str | Path) -> EvalConfig:
    config_path = _resolve_path(path)
    raw = _merge_meta_task(_load_json(config_path), config_path.parent)
    base_prompt = raw.get("prompt", "")
    policy = _as_dict(raw.get("policy"))
    adapter = _as_dict(raw.get("adapter"))
    cameras = _as_dict(raw.get("cameras"))
    video = _as_dict(raw.get("video"))

    return EvalConfig(
        name=raw.get("name", "policy_eval"),
        output_dir=raw.get("output_dir", "output/policy_evaluation"),
        prompt=base_prompt,
        max_steps=int(raw.get("max_steps", 600)),
        success_check_interval=int(raw.get("success_check_interval", 10)),
        control_hz=int(raw.get("control_hz", 30)),
        physics_step=int(raw.get("physics_step", 60)),
        rendering_step=int(raw.get("rendering_step", 30)),
        initial_wait_sec=float(raw.get("initial_wait_sec", 10.0)),
        enable_distractors=bool(raw.get("enable_distractors", False)),
        terminate_on_arm_reset=bool(raw.get("terminate_on_arm_reset", False)),
        arm_reset_tolerance=float(raw.get("arm_reset_tolerance", 0.1)),
        arm_reset_away_threshold=float(raw.get("arm_reset_away_threshold", 0.25)),
        arm_reset_consecutive_steps=max(1, int(raw.get("arm_reset_consecutive_steps", 3))),
        terminate_on_grasp_lost=bool(raw.get("terminate_on_grasp_lost", False)),
        grasp_steps_threshold=max(0, int(raw.get("grasp_steps_threshold", 0))),
        grasp_judge_distance_threshold=float(raw.get("grasp_judge_distance_threshold", 0.25)),
        wait_for_gripper_close_hold=bool(raw.get("wait_for_gripper_close_hold", False)),
        gripper_close_hold_timeout_sec=float(raw.get("gripper_close_hold_timeout_sec", 6.0)),
        gripper_close_max_force=float(raw.get("gripper_close_max_force", 0.2)),
        gripper_hold_stiffness=float(raw.get("gripper_hold_stiffness", 10000.0)),
        gripper_hold_max_force=float(raw.get("gripper_hold_max_force", 10.0)),
        gripper_static_friction=_optional_float(raw.get("gripper_static_friction")),
        gripper_dynamic_friction=_optional_float(raw.get("gripper_dynamic_friction")),
        object_static_friction=_optional_float(raw.get("object_static_friction")),
        object_dynamic_friction=_optional_float(raw.get("object_dynamic_friction")),
        target_object_mass_override=_optional_float(raw.get("target_object_mass_override")),
        object_mass_overrides={
            str(key): float(value)
            for key, value in _as_dict(raw.get("object_mass_overrides")).items()
        },
        disable_object_sleep=bool(raw.get("disable_object_sleep", True)),
        headless=bool(raw.get("headless", False)),
        render=bool(raw.get("render", True)),
        policy=PolicyConfig(
            host=policy.get("host", "localhost"),
            port=int(policy.get("port", 8181)),
            timeout_sec=float(policy.get("timeout_sec", 30.0)),
        ),
        adapter=AdapterConfig(
            arm=adapter.get("arm", "left"),
            state_dim=int(adapter.get("state_dim", 8)),
            action_arm_dim=int(adapter.get("action_arm_dim", 7)),
            action_gripper_index=int(adapter.get("action_gripper_index", 7)),
            action_open_threshold=float(adapter.get("action_open_threshold", 0.5)),
            gripper_state_mode=adapter.get("gripper_state_mode", "raw_mean"),
            image_keys=dict(
                adapter.get(
                    "image_keys",
                    {
                        "head": "cam_head",
                        "wrist": "cam_left",
                        "right_wrist": "cam_right",
                    },
                )
            ),
            joint_names=list(adapter.get("joint_names", [])),
            gripper_joint_names=list(adapter.get("gripper_joint_names", [])),
            open_gripper_positions=list(adapter.get("open_gripper_positions", [])),
            closed_gripper_positions=list(adapter.get("closed_gripper_positions", [])),
        ),
        cameras=CameraConfig(
            head=cameras.get("head", ""),
            wrist=cameras.get("wrist", ""),
            right_policy=cameras.get("right_policy", "zeros_like_left"),
            observer=dict(cameras.get("observer", {})),
        ),
        video=VideoConfig(
            enabled=bool(video.get("enabled", True)),
            fps=int(video.get("fps", 20)),
            every_n_steps=max(1, int(video.get("every_n_steps", 1))),
        ),
        tasks=_load_tasks(raw, base_prompt),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
