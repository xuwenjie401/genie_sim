"""Task JSON loading for policy evaluation."""

from __future__ import annotations

from pathlib import Path
import json

from policy_evaluation.config import TaskSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    roots = []
    if root is not None:
        roots.append(Path(root).expanduser())
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


def load_meta_like_tasks(path: str | Path, prompt: str = "") -> list[TaskSpec]:
    raw = load_json(path)
    if not raw.get("meta_task"):
        raise ValueError(f"Expected a meta_task JSON: {path}")
    tasks = []
    for item in raw.get("tasks", []):
        tasks.append(
            TaskSpec(
                task_template=item["task_template"],
                episodes=int(item.get("episodes", 1)),
                prompt=item.get("prompt", prompt),
                name=item.get("name", ""),
            )
        )
    return tasks
