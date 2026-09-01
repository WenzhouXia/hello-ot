from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from hello_ot._internal.core.solver import HierarchicalOTSolver
from hello_ot._internal.instrumentation.costs import _preprocess_sqeuclidean_to_lowrank
from hello_ot._internal.instrumentation.reporting import (
    _config_runtime_log_enabled,
    _print_warm_start_init_profile,
    _print_warm_start_iter_profile,
    _runtime_log,
)
from hello_ot._internal.trace import _ChromeTraceCollector, _trace_device_synchronize
from hello_ot._internal.runtime_context import record_solve_event
from hello_ot.refinement.budgeted_pruning import apply_budgeted_pruning
from hello_ot.refinement.dual_feasibility import (
    DualFeasibilityNorm,
    normalize_dual_feasibility_norm,
    run_lowrank_dual_feasibility_scan,
    stopping_dual_feasibility,
)
from hello_ot.restricted_ot.solve_lp import _copy_primal_for_active_support, solve_lp
from hello_ot.restricted_ot.runtime import _init_lowrank_warm_start_refinement
from hello_ot.refinement.dual_violation import append_candidate_pairs, detect_and_append_dual_violations
from hello_ot.output import relative_kkt_fields
from hello_ot.config import SolverRuntimeConfig
from hello_ot.state import GPUWarmStartState as OTWarmStartGPUState, WarmStartState as OTWarmStartState
from hello_ot.restricted_ot.runtime import (
    finalize_solve_output,
    sparse_coupling_from_active_support,
    state_from_active_support,
    sum_level_summary_metric,
)
from hello_ot.initialization.initial_support import (
    _max_optional_float,
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
    cost_type = str(getattr(solver, "_cost_type", getattr(config, "cost_type", "lowrank"))).lower()
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
    elif cost_type in {"l2^2", "lowrank"}:
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
    else:
        raise ValueError("Need cost_type in l1, l2, l2^2, lowrank, linf")


def _normalize_ot_dual_gauge(dual: np.ndarray, *, n_source: int) -> tuple[np.ndarray, float]:
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
    dual_raw = np.asarray(dual_new, dtype=np.float64).reshape(-1)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode != "ema":
        info = {
            "mode": "off",
            "alpha": float(alpha),
            "enabled": False,
            "applied": False,
            "reset": False,
            "delta_rel": 0.0,
            "gauge_normalization": False,
            "gauge_shift": 0.0,
        }
        return dual_raw, dual_raw.copy(), info

    gauge_enabled = bool(gauge_normalization)
    gauge_shift = 0.0
    if gauge_enabled:
        dual_raw, gauge_shift = _normalize_ot_dual_gauge(dual_raw, n_source=int(n_source))
    stable_prev_raw = None if previous_stable is None else np.asarray(previous_stable, dtype=np.float64).reshape(-1)
    reset = bool(stable_prev_raw is None or stable_prev_raw.shape != dual_raw.shape)
    stable_prev = (
        None
        if reset or stable_prev_raw is None
        else (
            _normalize_ot_dual_gauge(stable_prev_raw, n_source=int(n_source))[0]
            if gauge_enabled
            else stable_prev_raw
        )
    )
    applied = bool(not reset and float(alpha) > 0.0)
    if applied:
        dual_pricing = ((1.0 - float(alpha)) * dual_raw + float(alpha) * stable_prev).astype(np.float64, copy=False)
    else:
        dual_pricing = dual_raw
    delta = np.asarray(dual_pricing, dtype=np.float64).reshape(-1) - dual_raw
    denom = max(float(np.linalg.norm(dual_raw.astype(np.float64, copy=False))), 1e-12)
    delta_rel = float(np.linalg.norm(delta.astype(np.float64, copy=False)) / denom)
    info = {
        "mode": "ema",
        "alpha": float(alpha),
        "enabled": True,
        "applied": bool(applied),
        "reset": bool(reset),
        "delta_rel": float(delta_rel),
        "gauge_normalization": bool(gauge_enabled),
        "gauge_shift": float(gauge_shift),
    }
    return np.asarray(dual_pricing, dtype=np.float64).copy(), np.asarray(dual_pricing, dtype=np.float64).copy(), info


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
    info["dual_stabilization_delta_rel"] = float(stabilization_info.get("delta_rel", 0.0))
    info["dual_stabilization_gauge_normalization"] = bool(
        stabilization_info.get("gauge_normalization", False)
    )
    info["dual_stabilization_gauge_shift"] = float(stabilization_info.get("gauge_shift", 0.0))
    if bool(dual_feasibility_present):
        info["dual_feasibility_dual_stabilized"] = bool(stabilization_info.get("applied", False))


def _array_to_numpy_1d(value: Any, dtype: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
    return np.asarray(value, dtype=dtype).reshape(-1)


def _compact_pricing_info(info: Any) -> Dict[str, Any]:
    """
    CN: 从每轮 pricing 诊断中移除候选边数组，避免结果长期持有 GPU/CPU 大数组。
    EN: Remove candidate-edge arrays from per-iteration pricing diagnostics to avoid retaining large GPU/CPU arrays.
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
    need = max(0, int(before) - int(budget))
    removed = max(0, int(before) - int(after))
    active_support = getattr(solver, "active_support", None)
    old_pool = 0
    old_zero = 0
    nnz = 0
    old_pool_mask = None
    old_zero_mask = None
    protected_structural = 0
    protected_current_iteration = 0
    if (
        active_support is not None
        and bool(getattr(active_support, "track_creation", False))
        and getattr(active_support, "creation_iteration", None) is not None
        and int(before) > 0
    ):
        creation = _array_to_numpy_1d(active_support.creation_iteration, np.int64)
        x_prev = _array_to_numpy_1d(active_support.x_prev, np.float64)
        old_mask = (creation != -1) & (creation != int(current_inner_iter))
        protected_structural = int(np.count_nonzero(creation == -1))
        protected_current_iteration = int(
            np.count_nonzero(creation == int(current_inner_iter))
        )
        primal_tol = float(getattr(solver.cleaning_strategy, "primal_tol", 1e-10))
        nnz = int(np.count_nonzero(np.abs(x_prev) > primal_tol))
        old_pool_mask = old_mask
        old_zero_mask = old_mask & (np.abs(x_prev) < primal_tol)
        old_pool = int(np.count_nonzero(old_mask))
        old_zero = int(np.count_nonzero(old_zero_mask))
    return {
        "n_s": int(n_s),
        "n_t": int(n_t),
        "budget": int(budget),
        "before": int(before),
        "after": int(after),
        "need": int(need),
        "removed": int(removed),
        "nnz": int(nnz),
        "old_zero": int(old_zero),
        "old_pool": int(old_pool),
        "removed_old_zero_flow": 0,
        "removed_old_nonzero_flow": 0,
        "protected_structural": int(protected_structural),
        "protected_current_iteration": int(protected_current_iteration),
        "budget_overflow": max(0, int(after) - int(budget)),
        "budget_overflow_ratio": (
            float(max(0, int(after) - int(budget))) / float(max(1, int(n_s + n_t)))
        ),
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
    CN: 在单个层级上执行 active-set refinement 主循环。
    EN: Run the single-level active-set refinement main loop.

    CN: 每轮先解当前 restricted LP，再用全局 dual feasibility 判敛；未收敛时用 pricing 扩充 active support，
        随后执行 cleaning、记录诊断。
    EN: Each iteration solves the current restricted LP, checks global dual feasibility, expands
        the active support via pricing when needed, runs cleaning, and records diagnostics.
    """
    lvl_s = solver.hierarchy_s.finest_level
    lvl_t = solver.hierarchy_t.finest_level
    lp_tolerance = config.normalized_tolerance()
    stopping_norm = normalize_dual_feasibility_norm(dual_feasibility_norm)
    # CN: 循环级状态：objective history 仅用于诊断，last_* 用于下一轮 LP warm start。
    # EN: Loop-level state: objective history is diagnostic only, and last_* warms the next LP.
    objective_history: List[float] = []
    lp_time = 0.0
    pricing_time = 0.0
    last_primal = None
    last_dual = np.asarray(warm_start_dual, dtype=np.float64).copy() if warm_start_dual is not None else None
    last_res = None
    iteration_records: List[Dict[str, Any]] = []
    last_dual_feasibility_value: Optional[float] = None
    last_dual_feasibility_num_sq: Optional[float] = None
    last_dual_feasibility_den_sq: Optional[float] = None
    last_dual_feasibility_max_violation: Optional[float] = None
    last_dual_feasibility_cost_linf: Optional[float] = None
    last_relative_linf_dual_feasibility: Optional[float] = None
    last_dual_feasibility_source: Optional[str] = None
    last_dual_feasibility_tol: float = float(getattr(config, "dual_feasibility_tol", 1e-6))
    last_dual_feasibility_passed: Optional[bool] = None
    last_stopping_dual_feasibility: Optional[float] = None
    last_stopping_dual_feasibility_passed: Optional[bool] = None
    consecutive_no_new_edge_iterations = 0
    last_stopping_stalled_no_new_edges = False
    lp_backend_peak_mem_mib: Optional[float] = None
    lp_backend_abs_peak_mem_mib: Optional[float] = None
    pricing_peak_mem_mib: Optional[float] = None
    dual_feas_peak_mem_mib: Optional[float] = None
    dual_stabilization_mode = str(getattr(config, "dual_stabilization_mode", "off")).strip().lower()
    dual_stabilization_alpha = float(getattr(config, "dual_stabilization_alpha", 0.5))
    dual_stabilization_gauge_normalization = bool(
        getattr(config, "dual_stabilization_gauge_normalization", False)
    )
    dual_stable_prev: Optional[np.ndarray] = None
    peak_active_support_size = int(getattr(solver.active_support, "size", 0))

    # CN: 开始 active-support refinement 主迭代。
    # EN: Start the active-support refinement main iteration.
    t_solve_start = time.perf_counter()
    for inner_iter in range(int(config.max_inner_iter)):
        t_iter_start = time.perf_counter()
        active_support_before_lp = int(getattr(solver.active_support, "size", 0))
        peak_active_support_size = max(
            int(peak_active_support_size), int(active_support_before_lp)
        )
        with (
            trace_collector.span(
                f"{trace_prefix}.iter_total",
                "solve_ot",
                args={"inner_iter": int(inner_iter + 1)},
            )
            if trace_collector is not None
            else nullcontext()
        ):
            with (
                trace_collector.span(
                    f"{trace_prefix}.lp",
                    "solve_ot",
                    args={"inner_iter": int(inner_iter + 1)},
                )
                if trace_collector is not None
                else nullcontext()
            ):
                _trace_device_synchronize(trace_collector)
                lp_trace_args = {
                    "level": 0,
                    "inner_iter": int(inner_iter + 1),
                    "phase": "warm_start_lp",
                }
                # CN: 基于当前 active_support 组装 LP，调用后端求解，并把 primal/dual、耗时和 backend 诊断打包返回。
                # EN: it assembles the LP from the current active_support, calls the backend, and returns primal/dual, timing, and diagnostics.
                lp_pack = solve_lp(
                    solver,
                    lvl_s,
                    lvl_t,
                    lp_tolerance,
                    warm_start_dual=last_dual,
                    trace_collector=trace_collector,
                    trace_prefix=f"{trace_prefix}.lp",
                    trace_args=lp_trace_args,
                )
                _trace_device_synchronize(trace_collector)
            if not lp_pack["success"]:
                raise RuntimeError(f"Warm-start single-level LP failed at inner_iter={inner_iter}.")
            # CN: LP backend 显存峰值跨迭代取最大值，供 level summary 和上层 hello 诊断使用。
            # EN: LP backend memory peaks are accumulated as maxima across iterations for level summaries and hello diagnostics.
            lp_backend_peak_mem_mib = _max_optional_float(
                lp_backend_peak_mem_mib,
                lp_pack.get("lp_backend_peak_mem_mib"),
            )
            lp_backend_abs_peak_mem_mib = _max_optional_float(
                lp_backend_abs_peak_mem_mib,
                lp_pack.get("lp_backend_abs_peak_mem_mib"),
            )
            primal_new = lp_pack["primal"]
            dual_new = lp_pack["dual"]
            res = lp_pack["res"]
            # CN: dual_pricing 可能是 raw dual，也可能是 EMA-stabilized dual；raw dual 仍用于 LP warm start 和 cleaning。
            # EN: dual_pricing may be raw or EMA-stabilized; the raw dual is still used for LP warm starts and cleaning.
            with (
                trace_collector.span(
                    f"{trace_prefix}.dual_stabilization",
                    "solve_ot",
                    args={"inner_iter": int(inner_iter + 1)},
                )
                if trace_collector is not None
                else nullcontext()
            ):
                dual_pricing, dual_stable_prev, dual_stabilization_info = _dual_stabilization_pricing_dual(
                    dual_new,
                    dual_stable_prev,
                    n_source=len(lvl_s.points),
                    mode=dual_stabilization_mode,
                    alpha=dual_stabilization_alpha,
                    gauge_normalization=dual_stabilization_gauge_normalization,
                )
            t_finalize_start = time.perf_counter()
            t_state = time.perf_counter()
            with (
                trace_collector.span(
                    f"{trace_prefix}.state_update",
                    "solve_ot",
                    args={"inner_iter": int(inner_iter + 1)},
                )
                if trace_collector is not None
                else nullcontext()
            ):
                # CN: 这里把 LP 结果提交为下一阶段的循环状态；之后的 pricing/cleaning 都基于这次 LP 解。
                # EN: Commit the LP result as loop state; subsequent pricing/cleaning use this iteration LP solution.
                objective_history.append(float(res.obj_val))
                lp_time += float(lp_pack["lp_time"])
                last_primal = primal_new
                last_dual = dual_new
                last_res = res
                # CN: cleaning 必须基于当前 LP 解 x_k 判断 inactive 列，而不是旧 warm-start x_prev。
                # EN: Cleaning must use the current LP solution x_k to classify inactive columns, not stale warm-start x_prev.
                solver.active_support.set_x_prev(_copy_primal_for_active_support(solver.active_support, res.x))
            state_update_dt = time.perf_counter() - t_state

            t_conv = time.perf_counter()
            with (
                trace_collector.span(
                    f"{trace_prefix}.convergence_check",
                    "solve_ot",
                    args={"inner_iter": int(inner_iter + 1)},
                )
                if trace_collector is not None
                else nullcontext()
            ):
                # CN: 判敛只使用全局 dual feasibility；objective history 仅保留为诊断。
                # EN: Convergence uses only global dual feasibility; objective history is diagnostic only.
                is_converged = False
                dual_feasibility_value = None
                dual_feasibility_passed = None
                dual_feasibility_source = None
                dual_feasibility_scan_time = 0.0
                stopping_dual_feasibility_value = None
                stopping_dual_feasibility_passed = None
                early_scan_result = None
                early_scan_dt = 0.0
                early_dual_feas_args: Dict[str, Any] = {}
            convergence_dt = time.perf_counter() - t_conv

            # CN: 每轮 LP 后立即生成全局 dual-feasibility 证书；若证书通过则直接停止。
            # CN: fused lowrank scan 会按需同时产出 pricing 候选，但这只适用于 nodewise + 双向 pricing 路径。
            # EN: Build the global dual-feasibility certificate immediately after each LP; stop as soon as it passes.
            # EN: The fused lowrank scan can also produce pricing candidates on demand, but only for the nodewise + bidirectional pricing path.
            cost_type = str(getattr(config, "cost_type", "lowrank")).lower()
            use_fused_feasibility_pricing = cost_type in {"l1", "linf", "l2"} or bool(
                getattr(config, "use_fused_lowrank_feasibility_pricing", True)
            )
            support_staging_profile: Dict[str, Any] = {}
            # CN: feasibility/pricing scan 只读取 feature、cost 与 dual；暂存 support 可避免其与 resident feature 重叠。
            # EN: The feasibility/pricing scan reads only features, costs, and duals; staging support avoids overlap with resident features.
            support_staging_context = (
                solver.active_support.stage_storage_on_cpu()
                if bool(getattr(solver.active_support, "is_cuda", False))
                else nullcontext({})
            )
            with support_staging_context as support_staging_profile:
                with (
                    trace_collector.span(
                        f"{trace_prefix}.dual_feasibility_scan_total",
                        "solve_ot",
                        args={
                            "inner_iter": int(inner_iter + 1),
                            "use_fused_lowrank_feasibility_pricing": bool(use_fused_feasibility_pricing),
                        },
                    )
                    if trace_collector is not None
                    else nullcontext()
                ):
                    early_scan_result = _run_dual_feasibility_scan_for_solver(
                        solver=solver,
                        config=config,
                        lvl_s=lvl_s,
                        lvl_t=lvl_t,
                        dual_uv=dual_pricing,
                        inner_iter=int(inner_iter),
                        trace_collector=trace_collector,
                        trace_prefix=trace_prefix,
                        dual_feasibility_norm=stopping_norm,
                    )
            early_scan_result.diagnostics.update(dict(support_staging_profile))
            dual_feasibility_value = float(early_scan_result.dual_feasibility)
            dual_feasibility_source = early_scan_result.dual_feasibility_source
            early_dual_feas_args = dict(early_scan_result.diagnostics)
            dual_feasibility_num_sq = early_dual_feas_args.get(
                "dual_feasibility_num_sq"
            )
            dual_feasibility_den_sq = early_dual_feas_args.get(
                "dual_feasibility_den_sq"
            )
            dual_feasibility_max_violation = early_dual_feas_args.get(
                "dual_feasibility_max_violation"
            )
            dual_feasibility_cost_linf = early_dual_feas_args.get(
                "dual_feasibility_cost_linf"
            )
            relative_linf_dual_feasibility = early_dual_feas_args.get(
                "relative_linf_dual_feasibility"
            )
            early_scan_dt = float(early_scan_result.scan_time)
            dual_feasibility_scan_time = float(early_scan_dt)
            dual_feas_peak_mem_mib = _max_optional_float(
                dual_feas_peak_mem_mib,
                early_scan_result.peak_mem_mib,
            )
            if not (bool(use_fused_feasibility_pricing) and early_scan_result.has_candidate_pairs):
                convergence_dt += float(early_scan_dt)

            dual_feasibility_passed = bool(
                float(dual_feasibility_value) <= float(getattr(config, "dual_feasibility_tol", 1e-6))
            )
            stopping_dual_feasibility_value = stopping_dual_feasibility(
                norm=stopping_norm,
                l2_value=float(dual_feasibility_value),
                diagnostics=early_dual_feas_args,
            )
            stopping_dual_feasibility_passed = bool(
                float(stopping_dual_feasibility_value)
                <= float(getattr(config, "dual_feasibility_tol", 1e-6))
            )
            is_converged = bool(stopping_dual_feasibility_passed)
            record_solve_event(
                "check_optimality",
                iteration=int(inner_iter + 1),
                dual_feasibility=float(stopping_dual_feasibility_value),
                converged=bool(is_converged),
            )

            if bool(is_converged):
                level_pricing_info = {
                    "time": 0.0,
                    "skipped": True,
                    "reason": "dual_feasible_before_pricing",
                    "dual_feasibility": float(dual_feasibility_value),
                    "dual_feasibility_source": dual_feasibility_source,
                    "dual_feasibility_scan_time": float(dual_feasibility_scan_time),
                }
                level_pricing_info.update(
                    {
                        k: v
                        for k, v in early_dual_feas_args.items()
                        if k.startswith("source_F_")
                        or k.startswith("target_G_")
                        or k.startswith("fused_scan_")
                        or k.startswith("active_support_")
                        or k
                        in {
                            "resident_side",
                            "chunk_rows",
                            "dual_feas_entry_used_mem_mib",
                            "dual_feas_after_resident_upload_used_mem_mib",
                            "dual_feas_exit_used_mem_mib",
                        }
                    }
                )
                if early_scan_result is not None and early_scan_result.peak_mem_mib is not None:
                    level_pricing_info["dual_feas_peak_mem_mib"] = float(early_scan_result.peak_mem_mib)
                    level_pricing_info["dual_feas_peak_mem_source"] = "driver_mem_get_info"
                level_pricing_time = 0.0
                cleaning_dt = 0.0
                active_size_for_cleaning = int(getattr(solver.active_support, "size", 0))
                cleaning_info = _build_cleaning_profile_info(
                    solver,
                    n_s=len(lvl_s.points),
                    n_t=len(lvl_t.points),
                    before=active_size_for_cleaning,
                    after=active_size_for_cleaning,
                    current_inner_iter=int(inner_iter),
                )
                active_support_after_pricing = int(active_size_for_cleaning)
                active_support_after_cleaning = int(active_size_for_cleaning)
            else:
                with (
                    trace_collector.span(f"{trace_prefix}.pricing", "solve_ot", args={"inner_iter": int(inner_iter + 1)})
                    if trace_collector is not None
                    else nullcontext()
                ):
                    _trace_device_synchronize(trace_collector)
                    # CN: fused feasibility scan 若已返回候选，直接 append；否则调用常规 pricing strategy。
                    # EN: If the fused feasibility scan returned candidates, append them directly; otherwise run regular pricing.
                    if early_scan_result is not None and early_scan_result.has_candidate_pairs:
                        fused_extra_info = early_scan_result.pricing_extra_info()
                        if early_scan_result.peak_mem_mib is not None:
                            fused_extra_info["dual_feas_peak_mem_source"] = "driver_mem_get_info"
                            fused_extra_info["pricing_peak_mem_mib"] = float(early_scan_result.peak_mem_mib)
                            fused_extra_info["pricing_peak_mem_source"] = "driver_mem_get_info"
                        # CN: fused scan 已经产出违反 dual feasibility 的候选边，这里只复用并入 active support。
                        # EN: The fused scan already produced dual-feasibility violators; reuse and append them to active support here.
                        level_pricing_info = append_candidate_pairs(
                            solver,
                            lvl_s,
                            lvl_t,
                            early_scan_result.rows,
                            early_scan_result.cols,
                            dual_pricing,
                            inner_iter,
                            pricing_duration=float(early_scan_dt),
                            active_before=int(getattr(solver.active_support, "size", 0)),
                            trace_collector=trace_collector,
                            trace_prefix=f"{trace_prefix}.pricing",
                            report_added_violation_stats=bool(getattr(config, "report_added_violation_stats", False)),
                            added_violation_rel_threshold=float(getattr(config, "added_violation_rel_threshold", 1e-6)),
                            extra_info=fused_extra_info,
                        )
                    else:
                        level_pricing_info = detect_and_append_dual_violations(
                            solver,
                            lvl_s,
                            lvl_t,
                            primal_new,
                            dual_pricing,
                            inner_iter,
                            trace_collector=trace_collector,
                            trace_prefix=f"{trace_prefix}.pricing",
                            report_added_violation_stats=bool(getattr(config, "report_added_violation_stats", False)),
                            added_violation_rel_threshold=float(getattr(config, "added_violation_rel_threshold", 1e-6)),
                        )
                    _trace_device_synchronize(trace_collector)
                level_pricing_time = float(level_pricing_info.get("time", 0.0))
                pricing_peak_mem_mib = _max_optional_float(
                    pricing_peak_mem_mib,
                    level_pricing_info.get("pricing_peak_mem_mib"),
                )
                level_pricing_info.update(
                    {
                        "dual_feasibility": float(dual_feasibility_value),
                        "dual_feasibility_source": dual_feasibility_source,
                        "dual_feasibility_scan_time": float(dual_feasibility_scan_time),
                    }
                )
                pricing_time += float(level_pricing_time)
                # CN: cleaning 基于 raw dual 和当前 LP primal，保留必要支撑并控制 active support 规模。
                # EN: Cleaning uses the raw dual and current LP primal to preserve required support while controlling active-set size.
                t_clean = time.perf_counter()
                active_before_clean_size = int(getattr(solver.active_support, "size", 0))
                active_support_after_pricing = int(active_before_clean_size)
                cleaning_info = _build_cleaning_profile_info(
                    solver,
                    n_s=len(lvl_s.points),
                    n_t=len(lvl_t.points),
                    before=active_before_clean_size,
                    after=active_before_clean_size,
                    current_inner_iter=int(inner_iter),
                )
                with (
                    trace_collector.span(f"{trace_prefix}.cleaning", "solve_ot", args={"inner_iter": int(inner_iter + 1)})
                        if trace_collector is not None
                        else nullcontext()
                ):
                    removed_indices = apply_budgeted_pruning(
                        solver,
                        dual_new,
                        len(lvl_s.points),
                        len(lvl_t.points),
                        trace_collector=trace_collector,
                        trace_prefix=f"{trace_prefix}.cleaning",
                        current_inner_iter=int(inner_iter),
                    )
                cleaning_dt = time.perf_counter() - t_clean
                active_after_clean_size = int(getattr(solver.active_support, "size", 0))
                active_support_after_cleaning = int(active_after_clean_size)
                record_solve_event(
                    "update_support",
                    iteration=int(inner_iter + 1),
                    support_after_dual_violation=int(active_support_after_pricing),
                    support_after_budgeted_pruning=int(active_support_after_cleaning),
                )
                cleaning_info["after"] = int(active_after_clean_size)
                cleaning_info["removed"] = max(0, int(active_before_clean_size) - int(active_after_clean_size))
                cleaning_info["budget_overflow"] = max(
                    0,
                    int(active_after_clean_size) - int(cleaning_info.get("budget", 0)),
                )
                cleaning_info["budget_overflow_ratio"] = float(
                    cleaning_info["budget_overflow"]
                ) / float(max(1, len(lvl_s.points) + len(lvl_t.points)))
                removed_indices_np = np.asarray(removed_indices, dtype=np.int64).reshape(-1)
                old_zero_mask = cleaning_info.get("_old_zero_mask")
                old_pool_mask = cleaning_info.get("_old_pool_mask")
                if (
                    removed_indices_np.size > 0
                    and isinstance(old_zero_mask, np.ndarray)
                    and isinstance(old_pool_mask, np.ndarray)
                ):
                    valid_removed = removed_indices_np[
                        (removed_indices_np >= 0) & (removed_indices_np < int(old_zero_mask.size))
                    ]
                    removed_zero_mask = old_zero_mask[valid_removed]
                    removed_old_pool_mask = old_pool_mask[valid_removed]
                    cleaning_info["removed_old_zero_flow"] = int(
                        np.count_nonzero(removed_zero_mask)
                    )
                    cleaning_info["removed_old_nonzero_flow"] = int(
                        np.count_nonzero(removed_old_pool_mask & (~removed_zero_mask))
                    )
                cleaning_info.pop("_old_zero_mask", None)
                cleaning_info.pop("_old_pool_mask", None)
                if isinstance(level_pricing_info, dict):
                    _attach_dual_stabilization_info(
                        level_pricing_info,
                        dual_stabilization_info,
                        dual_feasibility_present=True,
                    )
            peak_active_support_size = max(
                int(peak_active_support_size),
                int(active_support_after_pricing),
                int(active_support_after_cleaning),
            )
            if isinstance(level_pricing_info, dict):
                _attach_dual_stabilization_info(
                    level_pricing_info,
                    dual_stabilization_info,
                    dual_feasibility_present=dual_feasibility_value is not None
                    or level_pricing_info.get("dual_feasibility") is not None,
                )
            if (
                dual_feasibility_value is None
                and isinstance(level_pricing_info, dict)
                and level_pricing_info.get("dual_feasibility") is not None
            ):
                dual_feasibility_value = level_pricing_info.get("dual_feasibility")
                dual_feasibility_source = level_pricing_info.get("dual_feasibility_source")
                dual_feasibility_passed = bool(
                    float(dual_feasibility_value) <= float(getattr(config, "dual_feasibility_tol", 1e-6))
                )
            pricing_added = (
                int(level_pricing_info.get("added", 0))
                if isinstance(level_pricing_info, dict) and not bool(is_converged)
                else None
            )
            stopping_stalled_no_new_edges = bool(
                not bool(stopping_dual_feasibility_passed) and pricing_added == 0
            )
            if bool(stopping_stalled_no_new_edges):
                consecutive_no_new_edge_iterations += 1
            else:
                consecutive_no_new_edge_iterations = 0
            finalize_t = time.perf_counter() - t_finalize_start
            # CN: convergence_info 和 iteration_records 是外部诊断、绘图和复现实验的主要来源。
            # EN: convergence_info and iteration_records are the main sources for diagnostics, plotting, and reproduction.
            with (
                trace_collector.span(
                    f"{trace_prefix}.iteration_diagnostics",
                    "solve_ot",
                    args={"inner_iter": int(inner_iter + 1)},
                )
                if trace_collector is not None
                else nullcontext()
            ):
                convergence_info = {
                    "signed_rel_obj_change": (
                        (float(objective_history[-1]) - float(objective_history[-2]))
                        / (abs(float(objective_history[-2])) + 1e-9)
                    )
                    if len(objective_history) >= 2
                    else None,
                    "is_converged": bool(is_converged),
                    "objective_converged": False,
                    "criterion": "dual_feasibility",
                    "plateau_counter": 0,
                    "required_plateau": None,
                    "objective_tol": None,
                    "require_dual_feasibility_convergence": False,
                    "dual_feasibility": None if dual_feasibility_value is None else float(dual_feasibility_value),
                    "dual_feasibility_num_sq": (
                        None
                        if dual_feasibility_num_sq is None
                        else float(dual_feasibility_num_sq)
                    ),
                    "dual_feasibility_den_sq": (
                        None
                        if dual_feasibility_den_sq is None
                        else float(dual_feasibility_den_sq)
                    ),
                    "dual_feasibility_max_violation": (
                        None
                        if dual_feasibility_max_violation is None
                        else float(dual_feasibility_max_violation)
                    ),
                    "dual_feasibility_cost_linf": (
                        None
                        if dual_feasibility_cost_linf is None
                        else float(dual_feasibility_cost_linf)
                    ),
                    "relative_linf_dual_feasibility": (
                        None
                        if relative_linf_dual_feasibility is None
                        else float(relative_linf_dual_feasibility)
                    ),
                    "dual_feasibility_source": dual_feasibility_source,
                    "dual_feasibility_tol": float(getattr(config, "dual_feasibility_tol", 1e-6)),
                    "dual_feasibility_passed": dual_feasibility_passed,
                    "finest_dual_feasibility_norm": str(stopping_norm),
                    "stopping_dual_feasibility": (
                        None
                        if stopping_dual_feasibility_value is None
                        else float(stopping_dual_feasibility_value)
                    ),
                    "stopping_dual_feasibility_tol": float(
                        getattr(config, "dual_feasibility_tol", 1e-6)
                    ),
                    "stopping_dual_feasibility_passed": stopping_dual_feasibility_passed,
                    "stopping_stalled_no_new_edges": bool(stopping_stalled_no_new_edges),
                    "consecutive_no_new_edge_iterations": int(consecutive_no_new_edge_iterations),
                    "require_added_convergence": False,
                    "previous_pricing_added": None,
                    "added_convergence_threshold": None,
                    "added_convergence_ratio": None,
                    "added_convergence_passed": None,
                    "dual_stabilization_mode": str(dual_stabilization_info.get("mode", "off")),
                    "dual_stabilization_alpha": float(dual_stabilization_info.get("alpha", 0.0)),
                    "dual_stabilization_enabled": bool(dual_stabilization_info.get("enabled", False)),
                    "dual_stabilization_applied": bool(dual_stabilization_info.get("applied", False)),
                    "dual_stabilization_reset": bool(dual_stabilization_info.get("reset", False)),
                    "dual_stabilization_delta_rel": float(dual_stabilization_info.get("delta_rel", 0.0)),
                    "dual_stabilization_gauge_normalization": bool(
                        dual_stabilization_info.get("gauge_normalization", False)
                    ),
                    "dual_stabilization_gauge_shift": float(
                        dual_stabilization_info.get("gauge_shift", 0.0)
                    ),
                    "dual_feasibility_dual_stabilized": (
                        bool(dual_stabilization_info.get("applied", False))
                        if dual_feasibility_value is not None
                        else None
                    ),
                }
            if convergence_info.get("dual_feasibility") is not None:
                last_dual_feasibility_value = float(convergence_info["dual_feasibility"])
                if convergence_info.get("dual_feasibility_num_sq") is not None:
                    last_dual_feasibility_num_sq = float(
                        convergence_info["dual_feasibility_num_sq"]
                    )
                if convergence_info.get("dual_feasibility_den_sq") is not None:
                    last_dual_feasibility_den_sq = float(
                        convergence_info["dual_feasibility_den_sq"]
                    )
                if convergence_info.get("dual_feasibility_max_violation") is not None:
                    last_dual_feasibility_max_violation = float(
                        convergence_info["dual_feasibility_max_violation"]
                    )
                if convergence_info.get("dual_feasibility_cost_linf") is not None:
                    last_dual_feasibility_cost_linf = float(
                        convergence_info["dual_feasibility_cost_linf"]
                    )
                if convergence_info.get("relative_linf_dual_feasibility") is not None:
                    last_relative_linf_dual_feasibility = float(
                        convergence_info["relative_linf_dual_feasibility"]
                    )
                last_dual_feasibility_source = convergence_info.get("dual_feasibility_source")
                last_dual_feasibility_tol = float(convergence_info["dual_feasibility_tol"])
                last_dual_feasibility_passed = convergence_info.get("dual_feasibility_passed")
                if convergence_info.get("stopping_dual_feasibility") is not None:
                    last_stopping_dual_feasibility = float(
                        convergence_info["stopping_dual_feasibility"]
                    )
                last_stopping_dual_feasibility_passed = convergence_info.get(
                    "stopping_dual_feasibility_passed"
                )
                last_stopping_stalled_no_new_edges = bool(
                    convergence_info.get("stopping_stalled_no_new_edges", False)
                )
            finalize_components = {
                "pricing": level_pricing_time,
                "cleaning": cleaning_dt,
                "state_update": state_update_dt,
                "convergence_check": convergence_dt,
            }
            with (
                trace_collector.span(
                    f"{trace_prefix}.print_iter_profile",
                    "solve_ot",
                    args={"inner_iter": int(inner_iter + 1)},
                )
                if trace_collector is not None
                else nullcontext()
            ):
                _print_warm_start_iter_profile(
                    solver,
                    inner_iter=inner_iter,
                    lp_pack=lp_pack,
                    pricing_info=level_pricing_info,
                    convergence_info=convergence_info,
                    cleaning_info=cleaning_info,
                    finalize_components=finalize_components,
                    iter_wall=time.perf_counter() - t_iter_start,
                    solve_t=float(lp_pack["lp_time"]),
                    finalize_t=finalize_t,
                    profile_depth=warm_start_profile_depth,
                    dual_stabilization_info=dual_stabilization_info,
                )
            # CN: 正式 iteration records 只保留标量统计；完整 dual 与候选边属于历史 debug artifacts。
            # EN: Formal iteration records retain compact statistics only; full duals and candidate edges are legacy debug artifacts.
            with (
                trace_collector.span(
                    f"{trace_prefix}.iteration_record_pack",
                    "solve_ot",
                    args={"inner_iter": int(inner_iter + 1)},
                )
                if trace_collector is not None
                else nullcontext()
            ):
                iteration_records.append(
                    {
                        "iter": int(inner_iter + 1),
                        "objective": float(res.obj_val),
                        "lp_time": float(lp_pack["lp_time"]),
                        "lp_iters": (
                            int(lp_pack["res"].iterations)
                            if getattr(lp_pack.get("res"), "iterations", None) is not None
                            else None
                        ),
                        "lp_diag": dict(lp_pack.get("diag", {}) or {}),
                        "active_support_before_lp": int(active_support_before_lp),
                        "active_support_after_pricing": int(
                            active_support_after_pricing
                        ),
                        "active_support_after_cleaning": int(
                            active_support_after_cleaning
                        ),
                        "pricing_info": _compact_pricing_info(level_pricing_info),
                        "cleaning_info": dict(cleaning_info),
                        "convergence_info": dict(convergence_info),
                        "finalize_components": dict(finalize_components),
                    }
                )
            if is_converged:
                _runtime_log(
                    solver,
                    "warm_start",
                    f"[WarmStartSolve] stop at iter={inner_iter + 1}/{int(config.max_inner_iter)}",
                )
                break

    if last_primal is None or last_dual is None or last_res is None:
        raise RuntimeError("Warm-start single-level loop did not produce a solution.")

    with (
        trace_collector.span(f"{trace_prefix}.export_coupling", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        # CN: 最终 coupling 只从 active_support 的 x_prev 导出；因此前面每次 LP 后必须及时刷新 x_prev。
        # EN: The final coupling is exported from active_support.x_prev, which is why x_prev is refreshed after each LP solve.
        coupling = sparse_coupling_from_active_support(
            solver,
            (len(lvl_s.points), len(lvl_t.points)),
        )
    with (
        trace_collector.span(f"{trace_prefix}.build_level_summaries", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        # CN: level_summaries 是 warm-start/refinement 对外的压缩统计视图，避免上层解析完整 iteration_records。
        # EN: level_summaries provide the compact external statistics view so callers do not need to parse full iteration_records.
        final_is_converged = bool(
            iteration_records
            and isinstance(iteration_records[-1].get("convergence_info"), dict)
            and iteration_records[-1]["convergence_info"].get("is_converged", False)
        )
        hit_itermax = (not final_is_converged) and int(len(objective_history)) >= int(config.max_inner_iter)
        level_summaries = [
            {
                "level": 0,
                "n_source": len(lvl_s.points),
                "n_target": len(lvl_t.points),
                "iters": int(len(objective_history)),
                "time": float(time.perf_counter() - t_solve_start),
                "objective": float(last_res.obj_val),
                "lp_time": float(lp_time),
                "pricing_time": float(pricing_time),
                "support_final": int(coupling.nnz),
                "peak_active_support_size": int(peak_active_support_size),
                "lp_backend_peak_mem_mib": lp_backend_peak_mem_mib,
                "lp_backend_delta_peak_mem_mib": lp_backend_peak_mem_mib,
                "lp_backend_abs_peak_mem_mib": lp_backend_abs_peak_mem_mib,
                "pricing_peak_mem_mib": pricing_peak_mem_mib,
                "dual_feas_peak_mem_mib": dual_feas_peak_mem_mib,
                "dual_feas_peak_mem_source": (
                    "driver_mem_get_info" if dual_feas_peak_mem_mib is not None else None
                ),
                "dual_feasibility": last_dual_feasibility_value,
                "dual_feasibility_num_sq": last_dual_feasibility_num_sq,
                "dual_feasibility_den_sq": last_dual_feasibility_den_sq,
                "dual_feasibility_max_violation": last_dual_feasibility_max_violation,
                "dual_feasibility_cost_linf": last_dual_feasibility_cost_linf,
                "relative_linf_dual_feasibility": last_relative_linf_dual_feasibility,
                "dual_feasibility_source": last_dual_feasibility_source,
                "dual_feasibility_tol": float(last_dual_feasibility_tol),
                "dual_feasibility_passed": last_dual_feasibility_passed,
                "finest_dual_feasibility_norm": str(stopping_norm),
                "stopping_dual_feasibility": last_stopping_dual_feasibility,
                "stopping_dual_feasibility_tol": float(last_dual_feasibility_tol),
                "stopping_dual_feasibility_passed": last_stopping_dual_feasibility_passed,
                "stopping_stalled_no_new_edges": bool(last_stopping_stalled_no_new_edges),
                "consecutive_no_new_edge_iterations": int(consecutive_no_new_edge_iterations),
                "converged": bool(final_is_converged),
                "hit_itermax": bool(hit_itermax),
                "converged_before_itermax": bool(final_is_converged) and not bool(hit_itermax),
                **relative_kkt_fields(
                    primal_feasibility=getattr(last_res, "primal_feas", None),
                    primal_dual_gap=getattr(last_res, "gap", None),
                    full_dual_feasibility=last_dual_feasibility_value,
                ),
            }
        ]
    with (
        trace_collector.span(f"{trace_prefix}.copy_dual", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        dual_out = np.asarray(last_dual, dtype=np.float64).copy()
    with (
        trace_collector.span(f"{trace_prefix}.build_result", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        solve_time = float(time.perf_counter() - t_solve_start)
    return {
        "distance": float(last_res.obj_val),
        "dual": dual_out,
        "coupling": coupling,
        "level_summaries": level_summaries,
        "solve_time": solve_time,
        "lp_backend_peak_mem_mib": lp_backend_peak_mem_mib,
        "lp_backend_delta_peak_mem_mib": lp_backend_peak_mem_mib,
        "lp_backend_abs_peak_mem_mib": lp_backend_abs_peak_mem_mib,
        "pricing_peak_mem_mib": pricing_peak_mem_mib,
        "dual_feas_peak_mem_mib": dual_feas_peak_mem_mib,
        "peak_active_support_size": int(peak_active_support_size),
        "iteration_records": iteration_records,
    }


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
    CN: 从已有 warm-start 支撑继续求解 lowrank OT refinement。
    CN: 入口负责补齐缺省质量、初始化 restricted-OT solver 和 active support，然后运行 refinement 主循环；判敛只使用全局 dual feasibility。
    EN: Continue lowrank OT refinement from an existing warm-start support.
    EN: This entry fills missing masses, initializes the restricted-OT solver and active support, then runs refinement; convergence uses only global dual feasibility.
    """
    n_s = source_F.shape[0]
    n_t = target_G.shape[0]
    if source_mass is None:
        source_mass = np.full(n_s, 1.0 / n_s, dtype=np.float64)
    if target_mass is None:
        target_mass = np.full(n_t, 1.0 / n_t, dtype=np.float64)

    init_result = _init_lowrank_warm_start_refinement(
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
    # CN: 初始化已经把 warm start 写入 solver.active_support，normal path 不再需要原始入参。
    # EN: Initialization has constructed solver.active_support, so the normal path no longer needs the original input.
    solver = init_result.solver
    cfg = init_result.cfg
    dual_uv = init_result.initialization.dual_warm_start
    initial_support_info = init_result.initialization.statistics

    # CN: 初始化 profile 在主循环前打印，便于区分 initial-support/pricing 成本和迭代 LP 成本。
    # EN: Print init profiling before the main loop to separate initial-support/pricing cost from iterative LP cost.
    _print_warm_start_init_profile(
        solver,
        components=dict(initial_support_info.get("components", {})),
        primal_nnz=int(initial_support_info.get("primal_nnz", 0)),
        active_size=int(initial_support_info.get("post_pricing_support_size", solver.active_support.size)),
        bfs_added=int(initial_support_info.get("northwest_added", 0)),
        profile_depth=warm_start_profile_depth,
    )
    # CN: 主循环反复执行 restricted LP、pricing、cleaning 和判敛；dual_feas/objective 语义主要在这里生效。
    # EN: The main loop repeatedly runs restricted LP, pricing, cleaning, and convergence checks; dual_feas/objective semantics live here.
    with (
        trace_collector.span(f"{trace_prefix}.total", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        warm_result = _run_single_level_active_support_refinement_loop(
            solver,
            cfg,
            dual_uv if bool(use_lp_warm_start_dual) else None,
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            warm_start_profile_depth=warm_start_profile_depth,
            dual_feasibility_norm=normalize_dual_feasibility_norm(dual_feasibility_norm),
        )

    # CN: log/return_state 需要把 solver.active_support 转回可复用的 warm_start_state，并携带最终 dual。
    # EN: log/return_state converts solver.active_support back to reusable warm_start_state with the final dual.
    state = None
    if return_state or log:
        with (
            trace_collector.span(f"{trace_prefix}.build_state_output", "solve_ot")
            if trace_collector is not None
            else nullcontext()
        ):
            state = state_from_active_support(
                solver,
                n_source=n_s,
                n_target=n_t,
                dual=warm_result["dual"],
            )

    # CN: 统一打包 refinement 输出，并补充 initialization profile 与 iteration records。
    # EN: Package the refinement output and attach the initialization profile and iteration records.
    with (
        trace_collector.span(f"{trace_prefix}.finalize_output", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        out = finalize_solve_output(
            distance=warm_result["distance"],
            coupling=warm_result["coupling"],
            state=state,
            dual_source=warm_result["dual"],
            level_summaries=warm_result["level_summaries"],
            lp_solve_time_total=sum_level_summary_metric(warm_result["level_summaries"], "lp_time"),
            elapsed=solver.build_time + warm_result["solve_time"],
            log=log,
            return_coupling=return_coupling,
            return_state=return_state,
        )
        if log and isinstance(out, dict):
            out["lp_backend_peak_mem_mib"] = warm_result.get("lp_backend_peak_mem_mib")
            out["lp_backend_delta_peak_mem_mib"] = warm_result.get("lp_backend_delta_peak_mem_mib")
            out["lp_backend_abs_peak_mem_mib"] = warm_result.get("lp_backend_abs_peak_mem_mib")
            out["pricing_peak_mem_mib"] = warm_result.get("pricing_peak_mem_mib")
            out["dual_feas_peak_mem_mib"] = warm_result.get("dual_feas_peak_mem_mib")
            out["warm_start_iteration_records"] = list(warm_result.get("iteration_records") or [])
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
    "_run_single_level_active_support_refinement_loop",
    "_refine_lowrank_from_warm_start",
    "_refine_sqeuclidean_from_warm_start",
]
