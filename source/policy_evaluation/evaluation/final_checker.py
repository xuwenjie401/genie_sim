"""Final, episode-level success checks based on data-collection task metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CheckResult:
    success: bool
    details: dict[str, Any] = field(default_factory=dict)


class FinalTaskChecker:
    def __init__(self, task_instance: dict[str, Any]):
        self.task_instance = task_instance
        self.rules = task_instance.get("task_metric", {}).get("filter_rules", [])
        self.fallback = _fallback_pick_place_rule(task_instance)

    def check(self, world) -> CheckResult:
        recognized_results = []
        details: dict[str, Any] = {"rules": []}
        for rule in self.rules:
            name = rule.get("rule_name")
            params = rule.get("params", {})
            if name == "is_object_relative_position_in_target":
                ok, rule_details = self._check_relative_position(world, params)
            elif name == "distance_to_target":
                ok, rule_details = self._check_distance(world, params)
            else:
                continue
            recognized_results.append(ok)
            details["rules"].append({"rule_name": name, "success": ok, **rule_details})

        if recognized_results:
            return CheckResult(success=all(recognized_results), details=details)

        if self.fallback:
            ok, rule_details = self._check_distance(world, self.fallback)
            details["rules"].append({"rule_name": "fallback_distance_to_place_target", "success": ok, **rule_details})
            return CheckResult(success=ok, details=details)

        return CheckResult(success=False, details={"error": "no supported final success rule found"})

    def _check_relative_position(self, world, params: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        target_id = params["target"]
        ranges = np.asarray(params["relative_position_range"], dtype=np.float64)
        target_pose = world.get_object_pose_matrix(target_id)
        target_inv = np.linalg.inv(target_pose)
        per_object = []
        ok_all = True
        for object_id in params.get("objects", []):
            obj_pose = world.get_object_pose_matrix(object_id)
            relative = target_inv @ obj_pose
            rel_pos = relative[:3, 3]
            ok = bool(np.all(rel_pos >= ranges[:, 0]) and np.all(rel_pos <= ranges[:, 1]))
            per_object.append({"object_id": object_id, "relative_position": rel_pos.tolist(), "success": ok})
            ok_all = ok_all and ok
        return ok_all, {"target": target_id, "objects": per_object}

    def _check_distance(self, world, params: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        object_id = params["object_id"]
        target_id = params["target_id"]
        threshold = float(params.get("value", 0.1))
        object_pos = world.get_object_position(object_id)
        target_pos = world.get_object_position(target_id)
        target_offset = params.get("target_offset", {}).get("position", [0.0, 0.0, 0.0])
        target_pos = target_pos + np.asarray(target_offset, dtype=np.float64)
        distance = float(np.linalg.norm(object_pos - target_pos))
        rule = params.get("rule", "lessThan")
        if rule == "greaterThan":
            ok = distance > threshold
        else:
            ok = distance < threshold
        return ok, {"object_id": object_id, "target_id": target_id, "distance": distance, "threshold": threshold}


def _fallback_pick_place_rule(task_instance: dict[str, Any]) -> dict[str, Any] | None:
    for stage in reversed(task_instance.get("stages", [])):
        if stage.get("action") != "place":
            continue
        active_id = stage.get("active", {}).get("object_id")
        passive_id = stage.get("passive", {}).get("object_id")
        if active_id and passive_id:
            return {
                "object_id": active_id,
                "target_id": passive_id,
                "value": 0.12,
                "rule": "lessThan",
            }
    return None

