from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from .types import Result, SolveStageResult


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


__all__ = ["relative_kkt_fields", "save_solution_npz", "summary_dict"]
