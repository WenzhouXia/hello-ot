"""
CN: 按 cost family 调度完整最优性扫描，不组织迭代。
EN: Dispatch full optimality scans by cost family without organizing iterations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional
import numpy as np
from hello_ot._internal.trace import _ChromeTraceCollector
from hello_ot.config import SolverRuntimeConfig
from hello_ot.refinement.dual_feasibility import DualFeasibilityNorm, run_lowrank_dual_feasibility_scan

if TYPE_CHECKING:
    from hello_ot._internal.core.solver import HierarchicalOTSolver

def _run_dual_feasibility_scan_for_solver(
    *,
    solver: HierarchicalOTSolver,
    config: SolverRuntimeConfig,
    lvl_s: Any,
    lvl_t: Any,
    dual_uv: np.ndarray,
    inner_iter: int,
    trace_collector: Optional[_ChromeTraceCollector],
    trace_prefix: str,
    dual_feasibility_norm: DualFeasibilityNorm = "l2",
) -> Any:
    """
    CN: 根据 cost family 调用完整边集 dual-feasibility scan。
    EN: Dispatch the full-edge dual-feasibility scan by cost family.
    """
    cost_type = str(
        getattr(solver, "_cost_type", getattr(config, "cost_type", "lowrank"))
    ).lower()
    if cost_type in {"l1", "linf", "l2"}:
        from hello_ot.costs.metric import run_metric_dual_feasibility_scan

        return run_metric_dual_feasibility_scan(
            config=config,
            lvl_s=lvl_s,
            lvl_t=lvl_t,
            dual_uv=dual_uv,
            inner_iter=int(inner_iter),
            trace_collector=trace_collector,
            trace_prefix=str(trace_prefix),
        )
    if cost_type in {"l2^2", "lowrank"}:
        return run_lowrank_dual_feasibility_scan(
            config=config,
            lvl_s=lvl_s,
            lvl_t=lvl_t,
            dual_uv=dual_uv,
            inner_iter=int(inner_iter),
            trace_collector=trace_collector,
            trace_prefix=str(trace_prefix),
            dual_feasibility_norm=dual_feasibility_norm,
        )
    raise ValueError("Need cost_type in l1, l2, l2^2, lowrank, linf")
