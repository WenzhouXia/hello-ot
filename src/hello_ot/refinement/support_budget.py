from __future__ import annotations

import math
from typing import Any


POSITIVE_SUPPORT_TOLERANCE = 1e-15


def normalize_pricing_topk(value: Any) -> int:
    """
    CN: 将论文参数 gamma 规范化为严格的正整数。
    EN: Normalize the paper parameter gamma to a strict positive integer.
    """
    if isinstance(value, bool):
        raise ValueError("pricing_topk must be a positive integer")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("pricing_topk must be a positive integer") from error
    if not math.isfinite(parsed) or parsed < 1.0 or not parsed.is_integer():
        raise ValueError("pricing_topk must be a positive integer")
    return int(parsed)


def validate_support_budget_factor(beta: Any, gamma: int) -> float:
    """
    CN: 验证论文要求的 beta >= gamma + 2，并返回有限浮点预算系数。
    EN: Validate the paper requirement beta >= gamma + 2 and return a finite budget factor.
    """
    if isinstance(beta, bool):
        raise ValueError("support_budget_factor must be finite and positive")
    try:
        parsed = float(beta)
    except (TypeError, ValueError) as error:
        raise ValueError("support_budget_factor must be finite and positive") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("support_budget_factor must be finite and positive")
    minimum = int(gamma) + 2
    if parsed < minimum:
        raise ValueError(
            "HELLO requires support_budget_factor >= pricing_topk + 2 "
            f"(beta={parsed:g}, gamma={int(gamma)}, required beta>={minimum})"
        )
    return parsed


def active_support_budget(beta: float, n_source: int, n_target: int) -> int:
    """
    CN: 返回论文中的 active-support 预算 ceil(beta * (m+n))。
    EN: Return the paper active-support budget ceil(beta * (m+n)).
    """
    return int(math.ceil(float(beta) * (int(n_source) + int(n_target))))


def primal_support_cap(
    beta: float,
    gamma: int,
    n_source: int,
    n_target: int,
) -> int:
    """
    CN: 返回为 feasible basis 与 dual violations 预留空间后的 primal-support cap。
    EN: Return the primal-support cap after reserving room for the feasible basis and dual violations.
    """
    vertex_count = int(n_source) + int(n_target)
    return active_support_budget(beta, n_source, n_target) - (vertex_count - 1) - int(gamma) * vertex_count


__all__ = [
    "POSITIVE_SUPPORT_TOLERANCE",
    "active_support_budget",
    "normalize_pricing_topk",
    "primal_support_cap",
    "validate_support_budget_factor",
]
