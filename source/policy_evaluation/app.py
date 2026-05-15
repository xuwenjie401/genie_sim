"""Isaac Sim entrypoint for data-collection task policy evaluation."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pi0-style policy on data-collection task JSONs.")
    parser.add_argument("--config", required=True, help="Policy evaluation JSON config.")
    parser.add_argument("--headless", action="store_true", help="Override config and run Isaac headless.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from policy_evaluation.config import load_eval_config

    config = load_eval_config(args.config)
    if args.headless:
        config.headless = True

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": config.headless,
            "disable_viewport_updates": config.headless,
            "renderer": "RayTracedLighting",
        }
    )

    try:
        from policy_evaluation.runner import EvaluationRunner

        summary = EvaluationRunner(config).run()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
