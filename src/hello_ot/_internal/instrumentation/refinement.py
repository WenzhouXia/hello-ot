"""
CN: refinement 的稳定化与 budgeted pruning 诊断辅助函数。
EN: Stabilization and budgeted-pruning diagnostic helpers for refinement.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict
import numpy as np
import torch

from hello_ot.refinement.support_budget import (
    POSITIVE_SUPPORT_TOLERANCE,
    active_support_budget,
)

if TYPE_CHECKING:
    from hello_ot._internal.core.solver import HierarchicalOTSolver

def _attach_dual_stabilization_info(
    info: Dict[str, Any],
    stabilization_info: Dict[str, Any],
    *,
    dual_feasibility_present: bool = False,
) -> None:
    info["dual_stabilization_mode"] = str(stabilization_info.get("mode", "off"))
    info["dual_stabilization_alpha"] = float(stabilization_info.get("alpha", 0.0))
    info["dual_stabilization_enabled"] = bool(stabilization_info.get("enabled", False))
    info["dual_stabilization_applied"] = bool(stabilization_info.get("applied", False))
    info["dual_stabilization_reset"] = bool(stabilization_info.get("reset", False))
    info["dual_stabilization_delta_rel"] = float(
        stabilization_info.get("delta_rel", 0.0)
    )
    info["dual_stabilization_gauge_normalization"] = bool(
        stabilization_info.get("gauge_normalization", False)
    )
    info["dual_stabilization_gauge_shift"] = float(
        stabilization_info.get("gauge_shift", 0.0)
    )
    if dual_feasibility_present:
        info["dual_feasibility_dual_stabilized"] = bool(
            stabilization_info.get("applied", False)
        )


def _array_to_numpy_1d(value: Any, dtype: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
    return np.asarray(value, dtype=dtype).reshape(-1)


def _build_cleaning_profile_info(
    solver: HierarchicalOTSolver,
    *,
    n_s: int,
    n_t: int,
    before: int,
    after: int,
    current_inner_iter: int,
) -> Dict[str, Any]:
    threshold = float(getattr(solver.cleaning_strategy, "threshold", 0.0))
    budget = (
        active_support_budget(threshold, n_s, n_t)
        if threshold > 0.0
        else int(before)
    )
    active_support = getattr(solver, "active_support", None)
    old_pool_mask = None
    old_zero_mask = None
    protected_structural = 0
    protected_current_iteration = 0
    old_pool = 0
    old_zero = 0
    nnz = 0
    if (
        active_support is not None
        and bool(getattr(active_support, "track_creation", False))
        and getattr(active_support, "creation_iteration", None) is not None
        and int(before) > 0
    ):
        creation = _array_to_numpy_1d(active_support.creation_iteration, np.int64)
        x_prev = _array_to_numpy_1d(active_support.x_prev, np.float64)
        old_pool_mask = (creation != -1) & (creation != int(current_inner_iter))
        protected_structural = int(np.count_nonzero(creation == -1))
        protected_current_iteration = int(
            np.count_nonzero(creation == int(current_inner_iter))
        )
        old_zero_mask = old_pool_mask & (
            np.abs(x_prev) <= POSITIVE_SUPPORT_TOLERANCE
        )
        nnz = int(
            np.count_nonzero(np.abs(x_prev) > POSITIVE_SUPPORT_TOLERANCE)
        )
        old_pool = int(np.count_nonzero(old_pool_mask))
        old_zero = int(np.count_nonzero(old_zero_mask))
    return {
        "n_s": int(n_s),
        "n_t": int(n_t),
        "budget": budget,
        "before": int(before),
        "after": int(after),
        "need": max(0, int(before) - budget),
        "removed": max(0, int(before) - int(after)),
        "nnz": nnz,
        "old_zero": old_zero,
        "old_pool": old_pool,
        "removed_old_zero_flow": 0,
        "removed_old_nonzero_flow": 0,
        "protected_structural": protected_structural,
        "protected_current_iteration": protected_current_iteration,
        "budget_overflow": max(0, int(after) - budget),
        "budget_overflow_ratio": float(max(0, int(after) - budget))
        / float(max(1, int(n_s + n_t))),
        "_old_zero_mask": old_zero_mask,
        "_old_pool_mask": old_pool_mask,
    }
