"""Generate concrete episode instances from data-collection task templates."""

from __future__ import annotations

from pathlib import Path
import copy
import json
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_COLLECTION_ROOT = PROJECT_ROOT / "source" / "data_collection"
if str(DATA_COLLECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_ROOT))


class TaskInstanceGenerator:
    """Thin wrapper around data_collection.client.layout.task_generate.TaskGenerator."""

    def __init__(self, generated_root: str | Path):
        self.generated_root = Path(generated_root)
        self.generated_root.mkdir(parents=True, exist_ok=True)

    def generate(self, task_template_path: str | Path, episode_index: int, task_name: str) -> Path:
        if not os.environ.get("SIM_ASSETS"):
            raise RuntimeError("SIM_ASSETS must point to GenieSimAssets before generating task instances.")

        from client.layout.task_generate import TaskGenerator

        task_template_path = Path(task_template_path)
        with open(task_template_path, "r", encoding="utf-8") as file:
            task_template = json.load(file)

        task_dir = self.generated_root / _safe_name(task_name)
        task_dir.mkdir(parents=True, exist_ok=True)
        output_file = task_dir / f"episode_{episode_index:06d}.json"
        generator = TaskGenerator(copy.deepcopy(task_template))
        max_attempts = 5
        for attempt in range(max_attempts):
            ok = generator.generate(str(output_file))
            if ok:
                break
            print(
                f"[policy_eval] layout generation attempt {attempt + 1}/{max_attempts} failed "
                f"for {task_template_path}; retrying",
                flush=True,
            )
        else:
            raise RuntimeError(f"Failed to generate task instance from {task_template_path}")
        return output_file


def _safe_name(name: str) -> str:
    keep = []
    for char in name:
        if char.isalnum() or char in ("_", "-", "."):
            keep.append(char)
        else:
            keep.append("_")
    safe = "".join(keep).strip("_")
    return safe or "task"
