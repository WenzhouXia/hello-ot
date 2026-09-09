from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np
import torch

from hello_ot._internal.core.solver import HierarchicalOTSolver
from hello_ot._internal.instrumentation.costs import _preprocess_sqeuclidean_to_lowrank
from hello_ot._internal.instrumentation.reporting import _print_warm_start_init_profile
from hello_ot._internal.trace import _ChromeTraceCollector
from hello_ot.config import SolverRuntimeConfig
from hello_ot.refinement.dual_feasibility import (
    DualFeasibilityNorm,
    normalize_dual_feasibility_norm,
    run_lowrank_dual_feasibility_scan,
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
        from hello_ot.kernels.norm_cost_scan import run_metric_dual_feasibility_scan

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


def _normalize_ot_dual_gauge(
    dual: np.ndarray,
    *,
    n_source: int,
) -> tuple[np.ndarray, float]:
    dual_out = np.asarray(dual, dtype=np.float64).reshape(-1).copy()
    source_count = int(n_source)
    if source_count <= 0 or source_count >= int(dual_out.size):
        raise ValueError("n_source must split a non-empty OT dual into source and target parts.")
    shift = float(np.mean(dual_out[:source_count].astype(np.float64, copy=False)))
    dual_out[:source_count] -= shift
    dual_out[source_count:] += shift
    return dual_out, shift


def _dual_stabilization_pricing_dual(
    dual_new: np.ndarray,
    previous_stable: Optional[np.ndarray],
    *,
    n_source: int,
    mode: str,
    alpha: float,
    gauge_normalization: bool = False,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    CN: 构造 pricing 使用的可选 EMA-stabilized dual，不修改 LP 返回的 raw dual。
    EN: Build the optional EMA-stabilized pricing dual without modifying the raw LP dual.
    """
    dual_raw = np.asarray(dual_new, dtype=np.float64).reshape(-1)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode != "ema":
        return dual_raw, dual_raw.copy(), {
            "mode": "off",
            "alpha": float(alpha),
            "enabled": False,
            "applied": False,
            "reset": False,
            "delta_rel": 0.0,
            "gauge_normalization": False,
            "gauge_shift": 0.0,
        }

    gauge_enabled = bool(gauge_normalization)
    gauge_shift = 0.0
    if gauge_enabled:
        dual_raw, gauge_shift = _normalize_ot_dual_gauge(
            dual_raw,
            n_source=int(n_source),
        )
    previous_raw = (
        None
        if previous_stable is None
        else np.asarray(previous_stable, dtype=np.float64).reshape(-1)
    )
    reset = bool(previous_raw is None or previous_raw.shape != dual_raw.shape)
    previous = (
        None
        if reset or previous_raw is None
        else (
            _normalize_ot_dual_gauge(previous_raw, n_source=int(n_source))[0]
            if gauge_enabled
            else previous_raw
        )
    )
    applied = bool(not reset and float(alpha) > 0.0)
    pricing_dual = (
        ((1.0 - float(alpha)) * dual_raw + float(alpha) * previous).astype(
            np.float64,
            copy=False,
        )
        if applied
        else dual_raw
    )
    delta = np.asarray(pricing_dual, dtype=np.float64).reshape(-1) - dual_raw
    denominator = max(
        float(np.linalg.norm(dual_raw.astype(np.float64, copy=False))),
        1e-12,
    )
    return (
        np.asarray(pricing_dual, dtype=np.float64).copy(),
        np.asarray(pricing_dual, dtype=np.float64).copy(),
        {
            "mode": "ema",
            "alpha": float(alpha),
            "enabled": True,
            "applied": applied,
            "reset": reset,
            "delta_rel": float(np.linalg.norm(delta) / denominator),
            "gauge_normalization": gauge_enabled,
            "gauge_shift": float(gauge_shift),
        },
    )


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


def _compact_pricing_info(info: Any) -> Dict[str, Any]:
    """
    CN: 从持久诊断中移除候选边数组，避免长期持有大数组。
    EN: Remove candidate-edge arrays from persistent diagnostics to avoid retaining large arrays.
    """
    if not isinstance(info, dict):
        return {}
    return {
        key: value
        for key, value in info.items()
        if key not in {"added_rows", "added_cols"}
    }


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
    budget = int(threshold * int(n_s + n_t)) if threshold > 0.0 else int(before)
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
        primal_tol = float(getattr(solver.cleaning_strategy, "primal_tol", 1e-10))
        old_zero_mask = old_pool_mask & (np.abs(x_prev) < primal_tol)
        nnz = int(np.count_nonzero(np.abs(x_prev) > primal_tol))
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
        out["warm_start_iteration_records"] = list(
            warm_result.get("iteration_records") or []
        )
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
