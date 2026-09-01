from __future__ import annotations

from typing import Any, Dict, Mapping


def build_objective_history_by_level(solutions: Mapping[Any, Any]) -> Dict[str, list[float]]:
    objective_history_by_level: Dict[str, list[float]] = {}
    for level, payload in solutions.items():
        if not isinstance(payload, dict):
            continue
        history = payload.get("history")
        if not isinstance(history, list):
            continue
        objective_history_by_level[str(level)] = [float(x) for x in history]
    return objective_history_by_level
