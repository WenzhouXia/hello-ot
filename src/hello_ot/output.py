from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from .types import IterationStats, Result, SolveStageResult


def iteration_stats_record(stats: IterationStats) -> Dict[str, Any]:
    """
    CN: 在兼容日志边界将强类型迭代统计序列化为旧字段布局。
    EN: Serialize typed iteration statistics to the legacy field layout at compatibility log boundaries.
    """
    pruning = stats.budgeted_pruning
    convergence = stats.convergence
    solve_lp = stats.solve_lp
    return {
        "iter": stats.index,
        "objective": stats.objective,
        "lp_time": solve_lp.wall_time,
        "full_scan_time": stats.full_scan_time,
        "lp_iters": solve_lp.backend_iterations,
        "lp_diag": {
            "solver_algorithm_rel_primal_res_l2": solve_lp.primal_feasibility,
            "solver_relative_primal_dual_gap": solve_lp.primal_dual_gap,
            "solver_termination_norm": solve_lp.termination_norm,
            "solver_termination_reason": solve_lp.termination_reason,
            "solver_termination_rel_primal_res": solve_lp.termination_primal_feasibility,
            "solver_termination_rel_dual_res": solve_lp.termination_dual_feasibility,
            "solver_algorithm_rel_dual_res_l2": solve_lp.algorithm_dual_feasibility,
        },
        "active_support_before_lp": stats.support.before_lp,
        "active_support_after_pricing": stats.support.after_dual_violation,
        "active_support_after_cleaning": stats.support.after_budgeted_pruning,
        "pricing_info": {
            "time": stats.dual_violation.wall_time,
            "found": stats.dual_violation.violations_found,
            "found_count": stats.dual_violation.violations_found,
            "added": stats.dual_violation.edges_added,
            "added_count": stats.dual_violation.edges_added,
        },
        "cleaning_info": {
            "before": pruning.support_before,
            "after": pruning.support_after,
            "removed": max(0, pruning.support_before - pruning.support_after),
            "budget": pruning.budget,
            "budget_overflow": max(0, pruning.support_after - pruning.budget),
        },
        "convergence_info": {
            "is_converged": convergence.passed,
            "dual_feasibility": convergence.relative_full_dual_feasibility,
            "relative_linf_dual_feasibility": convergence.relative_linf_dual_feasibility,
            "dual_feasibility_max_violation": convergence.max_dual_violation,
            "finest_dual_feasibility_norm": convergence.norm,
            "stopping_dual_feasibility": convergence.dual_feasibility,
            "stopping_dual_feasibility_tol": convergence.tolerance,
            "stopping_dual_feasibility_passed": convergence.passed,
        },
        "finalize_components": {
            "pricing": stats.dual_violation.wall_time,
            "cleaning": pruning.wall_time,
            "state_update": stats.bookkeeping_time,
            "convergence_check": 0.0,
        },
        "wall_time": stats.wall_time,
    }


def relative_kkt_fields(
    *,
    primal_feasibility: Any,
    primal_dual_gap: Any,
    full_dual_feasibility: Any,
) -> Dict[str, Any]:
    """
    CN: 汇总同一最终 primal--dual 解的三个相对残差及其 KKT 最大值。
    EN: Collect three relative residuals and their KKT maximum for one final primal--dual solution.
    """
    values = {}
    for key, value in (
        ("relative_primal_feasibility", primal_feasibility),
        ("relative_primal_dual_gap", primal_dual_gap),
        ("relative_full_dual_feasibility", full_dual_feasibility),
    ):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None and not math.isfinite(parsed):
            parsed = None
        values[key] = parsed
    finite = [value for value in values.values() if value is not None]
    return {**values, "relative_kkt_error": max(finite) if len(finite) == 3 else None}


def _stage_summary(stage: SolveStageResult) -> Dict[str, Any]:
    """
    CN: 将不含大数组的 stage 统计转换为 JSON-safe 字典。
    EN: Convert array-free stage statistics into a JSON-safe dictionary.
    """
    return asdict(stage)


def summary_dict(result: Result) -> Dict[str, Any]:
    """
    CN: 导出 HELLO 的轻量层级统计，不复制最终 primal/dual 数组。
    EN: Export compact hierarchical HELLO statistics without copying final primal/dual arrays.
    """
    return {
        "objective": float(result.objective),
        "shape": [int(result.solution.shape[0]), int(result.solution.shape[1])],
        "solution_nnz": int(result.solution.values.size),
        "total_wall_time": float(result.total_wall_time),
        "solve_stage": _stage_summary(result.solve_stage),
        "trace_recorded": result.trace is not None,
        "metadata": dict(result.metadata),
    }


def save_solution_npz(result: Result, path: Union[str, Path]) -> None:
    """
    CN: 单独保存最终稀疏 coupling 与对偶势，避免把数组塞入 JSON summary。
    EN: Save the final sparse coupling and dual potentials separately from the JSON summary.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        shape=np.asarray(result.solution.shape, dtype=np.int64),
        rows=result.solution.rows,
        cols=result.solution.cols,
        values=result.solution.values,
        source_dual=result.solution.source_dual,
        target_dual=result.solution.target_dual,
    )


__all__ = [
    "iteration_stats_record",
    "relative_kkt_fields",
    "save_solution_npz",
    "summary_dict",
]
