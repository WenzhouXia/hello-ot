from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np

from hello_ot._internal.core.solver import HierarchicalOTSolver
from hello_ot._internal.instrumentation.costs import _preprocess_sqeuclidean_to_lowrank
from hello_ot._internal.instrumentation.reporting import _print_warm_start_init_profile
from hello_ot._internal.trace import _ChromeTraceCollector
from hello_ot.config import SolverRuntimeConfig
from hello_ot.output import iteration_stats_record
from hello_ot.refinement.dual_feasibility import (
    DualFeasibilityNorm,
    normalize_dual_feasibility_norm,
)
from hello_ot.restricted_ot.runtime import (
    _init_lowrank_warm_start_refinement,
    finalize_solve_output,
    state_from_active_support,
    sum_level_summary_metric,
)
from hello_ot.state import (
    GPUWarmStartState as OTWarmStartGPUState,
    WarmStartState as OTWarmStartState,
)


def _run_single_level_active_support_refinement_loop(
    solver: HierarchicalOTSolver,
    config: SolverRuntimeConfig,
    warm_start_dual: Optional[np.ndarray],
    trace_collector: Optional[_ChromeTraceCollector] = None,
    trace_prefix: str = "solve_ot",
    warm_start_profile_depth: Optional[int] = None,
    dual_feasibility_norm: DualFeasibilityNorm = "l2",
) -> Dict[str, Any]:
    """
    CN: 为非 HELLO hierarchy 调用执行相同的三个论文级算子。
    EN: Run the same three paper-level operators for non-HELLO-hierarchy callers.
    """
    from hello_ot.refinement.iterations import (
        begin_refinement,
        check_optimality,
        finalize_refinement,
        solve_lp,
        update_support,
    )

    runtime = begin_refinement(
        solver,
        config,
        warm_start_dual,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        warm_start_profile_depth=warm_start_profile_depth,
        dual_feasibility_norm=dual_feasibility_norm,
    )
    for iteration_index in range(int(config.max_inner_iter)):
        iteration = solve_lp(runtime, iteration_index)
        certificate = check_optimality(runtime, iteration)
        if certificate.converged:
            break
        update_support(runtime, iteration, certificate)
    return finalize_refinement(runtime)


def _refine_lowrank_from_warm_start(
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    source_mass: Optional[np.ndarray],
    target_mass: Optional[np.ndarray],
    log: bool,
    return_coupling: bool,
    return_state: bool,
    config: SolverRuntimeConfig,
    warm_start: OTWarmStartState | OTWarmStartGPUState,
    skip_initial_pricing: bool = False,
    use_lp_warm_start_dual: bool = True,
    trace_collector: Optional[_ChromeTraceCollector] = None,
    trace_prefix: str = "solve_ot",
    warm_start_profile_depth: Optional[int] = None,
    pricing_index_pool: Optional[Any] = None,
    warm_start_cost_vec_chunk_size: int = 65536,
    warm_start_cost_vec_feature_chunk_size: Optional[int] = None,
    _consume_warm_start_gpu_state: bool = False,
    dual_feasibility_norm: DualFeasibilityNorm = "l2",
    dot_scale: float = 1.0,
) -> Any:
    """
    CN: 从已有 warm start 运行独立的 lowrank 单层 refinement。
    EN: Run a standalone low-rank level refinement from an existing warm start.
    """
    n_source = int(source_F.shape[0])
    n_target = int(target_G.shape[0])
    if source_mass is None:
        source_mass = np.full(n_source, 1.0 / n_source, dtype=np.float64)
    if target_mass is None:
        target_mass = np.full(n_target, 1.0 / n_target, dtype=np.float64)
    initialized = _init_lowrank_warm_start_refinement(
        source_F=source_F,
        target_G=target_G,
        source_cost_vec=source_cost_vec,
        target_cost_vec=target_cost_vec,
        source_mass=np.asarray(source_mass, dtype=np.float64),
        target_mass=np.asarray(target_mass, dtype=np.float64),
        config=config,
        warm_start=warm_start,
        skip_initial_pricing=bool(skip_initial_pricing),
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        pricing_index_pool=pricing_index_pool,
        warm_start_cost_vec_chunk_size=int(warm_start_cost_vec_chunk_size),
        warm_start_cost_vec_feature_chunk_size=warm_start_cost_vec_feature_chunk_size,
        _consume_warm_start_gpu_state=bool(_consume_warm_start_gpu_state),
        dot_scale=float(dot_scale),
    )
    solver = initialized.solver
    initial_support_info = initialized.initialization.statistics
    _print_warm_start_init_profile(
        solver,
        components=dict(initial_support_info.get("components", {})),
        primal_nnz=int(initial_support_info.get("primal_nnz", 0)),
        active_size=int(
            initial_support_info.get(
                "post_pricing_support_size",
                solver.active_support.size,
            )
        ),
        bfs_added=int(initial_support_info.get("northwest_added", 0)),
        profile_depth=warm_start_profile_depth,
    )
    with (
        trace_collector.span(f"{trace_prefix}.total", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        warm_result = _run_single_level_active_support_refinement_loop(
            solver,
            initialized.cfg,
            (
                initialized.initialization.dual_warm_start
                if use_lp_warm_start_dual
                else None
            ),
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            warm_start_profile_depth=warm_start_profile_depth,
            dual_feasibility_norm=normalize_dual_feasibility_norm(
                dual_feasibility_norm
            ),
        )
    state = (
        state_from_active_support(
            solver,
            n_source=n_source,
            n_target=n_target,
            dual=warm_result["dual"],
        )
        if return_state or log
        else None
    )
    out = finalize_solve_output(
        distance=warm_result["distance"],
        coupling=warm_result["coupling"],
        state=state,
        dual_source=warm_result["dual"],
        level_summaries=warm_result["level_summaries"],
        lp_solve_time_total=sum_level_summary_metric(
            warm_result["level_summaries"],
            "lp_time",
        ),
        elapsed=solver.build_time + warm_result["solve_time"],
        log=log,
        return_coupling=return_coupling,
        return_state=return_state,
    )
    if log and isinstance(out, dict):
        for key in (
            "lp_backend_peak_mem_mib",
            "lp_backend_delta_peak_mem_mib",
            "lp_backend_abs_peak_mem_mib",
            "pricing_peak_mem_mib",
            "dual_feas_peak_mem_mib",
        ):
            out[key] = warm_result.get(key)
        out["warm_start_iteration_records"] = [
            iteration_stats_record(record)
            for record in warm_result.get("iteration_records") or []
        ]
        out["warm_start_init"] = dict(initial_support_info)
    return out


def _refine_sqeuclidean_from_warm_start(
    source_X: np.ndarray,
    target_X: np.ndarray,
    source_mass: Optional[np.ndarray],
    target_mass: Optional[np.ndarray],
    log: bool,
    return_coupling: bool,
    return_state: bool,
    config: SolverRuntimeConfig,
    warm_start: OTWarmStartState,
) -> Any:
    source_F, source_cost_vec = _preprocess_sqeuclidean_to_lowrank(source_X)
    target_G, target_cost_vec = _preprocess_sqeuclidean_to_lowrank(target_X)
    return _refine_lowrank_from_warm_start(
        source_F=source_F,
        target_G=target_G,
        source_cost_vec=source_cost_vec,
        target_cost_vec=target_cost_vec,
        source_mass=source_mass,
        target_mass=target_mass,
        log=log,
        return_coupling=return_coupling,
        return_state=return_state,
        config=config,
        warm_start=warm_start,
    )


__all__ = [
    "_refine_lowrank_from_warm_start",
    "_refine_sqeuclidean_from_warm_start",
    "_run_single_level_active_support_refinement_loop",
]
