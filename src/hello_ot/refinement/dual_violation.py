from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
from scipy.sparse import csc_matrix
import numpy as np
import torch
from hello_ot.utilities import trace_span
from hello_ot._internal.runtime_context import record_solve_event

logger = logging.getLogger(__name__)


def fmt_size_ratio(count: int, n_s: int, n_t: int) -> str:
    denom = n_s + n_t
    ratio = (count / denom) if denom > 0 else 0.0
    return f"{count:,} (x{ratio:.2f})"


def _as_numpy_1d(array: Any, dtype: Any) -> np.ndarray:
    if torch.is_tensor(array):
        return array.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
    return np.asarray(array, dtype=dtype).reshape(-1)


def _compute_added_relative_violation_stats(
    solver,
    *,
    added_rows: Any,
    added_cols: Any,
    dual_new: np.ndarray,
    n_source: int,
    threshold: float,
) -> Dict[str, Any]:
    """
    CN: 统计新增边中相对 dual violation 超过阈值的数量。
    EN: Count added edges whose relative dual violation exceeds the threshold.
    """
    added_count = int(solver.active_support._array_size(added_rows))
    threshold = float(threshold)
    stats: Dict[str, Any] = {
        "added_rel_violation_threshold": threshold,
        "added_rel_violation_count": int(added_count),
        "added_rel_violation_gt_threshold": 0,
        "added_rel_violation_max": None,
        "added_rel_violation_mean": None,
    }
    if added_count <= 0:
        return stats

    rows_np = _as_numpy_1d(added_rows, np.int64)
    cols_np = _as_numpy_1d(added_cols, np.int64)
    costs = solver._compute_pair_costs_arrays(added_rows, added_cols)
    costs_np = _as_numpy_1d(costs, np.float64)
    dual_np = np.asarray(dual_new, dtype=np.float64).reshape(-1)
    u = dual_np[: int(n_source)]
    v = dual_np[int(n_source) :]
    denom = np.maximum(np.abs(costs_np), 1e-12)
    rel_violation = (u[rows_np] + v[cols_np]) / denom - 1.0
    finite = np.isfinite(rel_violation)
    if not np.any(finite):
        return stats
    rel_finite = rel_violation[finite]
    stats["added_rel_violation_gt_threshold"] = int(np.count_nonzero(rel_finite > threshold))
    stats["added_rel_violation_max"] = float(np.max(rel_finite))
    stats["added_rel_violation_mean"] = float(np.mean(rel_finite))
    stats["_added_pair_costs"] = costs
    return stats



def append_candidate_pairs(
    solver,
    lvl_s: Any,
    lvl_t: Any,
    candidate_rows: Any,
    candidate_cols: Any,
    dual_new: np.ndarray,
    iter_idx: int,
    *,
    pricing_duration: float = 0.0,
    active_before: Optional[int] = None,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.solve.finalize_iteration.pricing",
    report_added_violation_stats: Optional[bool] = None,
    added_violation_rel_threshold: Optional[float] = None,
    include_strategy_dual_info: bool = False,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    CN: 对 pricing 产生的候选边执行去重、append 与诊断统计。
    EN: Deduplicate, append, and diagnose candidate edges produced by pricing.
    """
    n_s, n_t = len(lvl_s.points), len(lvl_t.points)
    active_before_value = int(solver.active_support.size) if active_before is None else int(active_before)
    c_rows, c_cols = candidate_rows, candidate_cols
    original_found_count = int(solver.active_support._array_size(c_rows))
    # CN: active support 负责去重，只保留当前 support 中尚不存在的候选边。
    # EN: Active support handles deduplication and keeps only candidate pairs not already present.
    with trace_span(trace_collector, trace_prefix, "filter_new_candidates"):
        added_rows, added_cols, found_count, added_count = solver.active_support.filter_new_candidate_pairs(
            c_rows,
            c_cols,
        )
    already_active_count = int(found_count - added_count)

    violation_stats: Dict[str, Any] = {}
    c_add = None
    report_stats = (
        bool(getattr(solver, "_report_added_violation_stats", False))
        if report_added_violation_stats is None
        else bool(report_added_violation_stats)
    )
    # CN: 诊断模式下统计新增边的相对 dual violation，并复用已算出的 pair costs 作为 append 附加数据。
    # EN: In diagnostic mode, measure relative dual violation for new pairs and reuse computed pair costs for append metadata.
    if report_stats:
        threshold = (
            float(getattr(solver, "_added_violation_rel_threshold", 1e-6))
            if added_violation_rel_threshold is None
            else float(added_violation_rel_threshold)
        )
        with trace_span(
            trace_collector,
            trace_prefix,
            "added_relative_violation_stats",
            args={"added_count": int(added_count), "threshold": float(threshold)},
        ):
            violation_stats = _compute_added_relative_violation_stats(
                solver,
                added_rows=added_rows,
                added_cols=added_cols,
                dual_new=dual_new,
                n_source=n_s,
                threshold=threshold,
            )
            c_add = violation_stats.pop("_added_pair_costs", None)

    # CN: 只有真正新增的候选边会写入 active support；已有边只计入诊断。
    # EN: Only genuinely new candidate pairs are appended to active support; existing pairs are diagnostics only.
    if added_count > 0:
        if logger.isEnabledFor(logging.INFO):
            logger.info("  [Pricing] add_new=%s", fmt_size_ratio(added_count, n_s, n_t))
        with trace_span(
            trace_collector,
            trace_prefix,
            "append_new_pairs",
            args={"added_count": int(added_count)},
        ):
            solver._append_new_pairs_arrays(added_rows, added_cols, iter_idx, c_add=c_add)
    record_solve_event(
        "insert_violations",
        iteration=int(iter_idx),
        added_rows=added_rows,
        added_cols=added_cols,
        rows=solver.active_support.rows,
        cols=solver.active_support.cols,
    )

    # CN: 返回值服务 runtime log 和 iteration_records；dump-only 大数组不混入该诊断字典。
    # EN: The returned info feeds runtime logs and iteration records; dump-only arrays do not enter this dict.
    info = {
        "time": float(pricing_duration),
        "active_before": active_before_value,
        "found": int(original_found_count),
        "found_already_active": already_active_count,
        "added": added_count,
        "added_rows": added_rows,
        "added_cols": added_cols,
        "active_after": int(solver.active_support.size),
    }
    if bool(include_strategy_dual_info):
        dual_feasibility_info = getattr(solver.strategy, "last_dual_feasibility_info", None)
        if isinstance(dual_feasibility_info, dict) and dual_feasibility_info:
            info.update(dual_feasibility_info)
    if isinstance(extra_info, dict) and extra_info:
        info.update(extra_info)
    info.update(violation_stats)
    return info


def detect_and_append_dual_violations(
    solver,
    lvl_s: Any,
    lvl_t: Any,
    primal_new: csc_matrix,
    dual_new: np.ndarray,
    iter_idx: int,
    *,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.solve.finalize_iteration.pricing",
    report_added_violation_stats: Optional[bool] = None,
    added_violation_rel_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    n_s, n_t = len(lvl_s.points), len(lvl_t.points)
    level_idx = int(getattr(lvl_s, "level_idx", -1))
    active_before = int(solver.active_support.size)
    t_price_start = time.perf_counter()
    pricing_args = (
        primal_new,
        (dual_new[:n_s], dual_new[n_s:]),
        solver.active_support.level_cache,
    )
    with trace_span(
        trace_collector,
        trace_prefix,
        "strategy_generate",
        args={"level_idx": int(getattr(lvl_s, "level_idx", -1)), "inner_iter": int(iter_idx)},
    ):
        new_cands = solver.strategy.generate(
            *pricing_args,
            level_idx=level_idx,
            inner_iter=int(iter_idx),
            trace_collector=trace_collector,
            trace_prefix=f"{trace_prefix}.strategy",
        )
    pricing_duration = time.perf_counter() - t_price_start

    c_rows, c_cols = new_cands
    info = append_candidate_pairs(
        solver,
        lvl_s,
        lvl_t,
        c_rows,
        c_cols,
        dual_new,
        iter_idx,
        pricing_duration=float(pricing_duration),
        active_before=int(active_before),
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        report_added_violation_stats=report_added_violation_stats,
        added_violation_rel_threshold=added_violation_rel_threshold,
        include_strategy_dual_info=True,
    )
    return info


__all__ = [
    "append_candidate_pairs",
    "detect_and_append_dual_violations",
]
