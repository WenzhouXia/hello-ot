from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from hello_ot._internal.core.solver import HierarchicalOTSolver
from hello_ot._internal.instrumentation.reporting import (
    _print_warm_start_iter_profile,
    _runtime_log,
)
from hello_ot._internal.runtime_context import record_solve_event
from hello_ot._internal.trace import _ChromeTraceCollector, _trace_device_synchronize
from hello_ot.config import SolverRuntimeConfig
from hello_ot.initialization.initial_support import _max_optional_float
from hello_ot.output import relative_kkt_fields
from hello_ot.refinement.budgeted_pruning import apply_budgeted_pruning
from hello_ot.refinement.dual_feasibility import (
    DualFeasibilityNorm,
    normalize_dual_feasibility_norm,
    stopping_dual_feasibility,
)
from hello_ot.refinement.dual_violation import (
    append_candidate_pairs,
    detect_and_append_dual_violations,
)
from hello_ot.restricted_ot.solve_lp import (
    _copy_primal_for_active_support,
    solve_lp as _solve_restricted_lp,
)
from hello_ot.restricted_ot.runtime import sparse_coupling_from_active_support


@dataclass
class RefinementRuntime:
    """
    CN: 单层 active-support refinement 的可变运行状态；数组均沿用 solver 已有引用。
    EN: Mutable runtime for one active-support refinement level; arrays reuse the solver's existing references.
    """

    solver: HierarchicalOTSolver
    config: SolverRuntimeConfig
    trace_collector: Optional[_ChromeTraceCollector]
    trace_prefix: str
    warm_start_profile_depth: Optional[int]
    stopping_norm: DualFeasibilityNorm
    level_source: Any
    level_target: Any
    lp_tolerance: Dict[str, float]
    solve_started_at: float = field(default_factory=time.perf_counter)
    objective_history: List[float] = field(default_factory=list)
    lp_time: float = 0.0
    pricing_time: float = 0.0
    full_scan_time: float = 0.0
    last_primal: Any = None
    last_dual: Any = None
    last_result: Any = None
    iteration_records: List[Dict[str, Any]] = field(default_factory=list)
    last_dual_feasibility_value: Optional[float] = None
    last_dual_feasibility_num_sq: Optional[float] = None
    last_dual_feasibility_den_sq: Optional[float] = None
    last_dual_feasibility_max_violation: Optional[float] = None
    last_dual_feasibility_cost_linf: Optional[float] = None
    last_relative_linf_dual_feasibility: Optional[float] = None
    last_dual_feasibility_source: Optional[str] = None
    last_dual_feasibility_tol: float = 1e-6
    last_dual_feasibility_passed: Optional[bool] = None
    last_stopping_dual_feasibility: Optional[float] = None
    last_stopping_dual_feasibility_passed: Optional[bool] = None
    consecutive_no_new_edge_iterations: int = 0
    last_stopping_stalled_no_new_edges: bool = False
    lp_backend_peak_mem_mib: Optional[float] = None
    lp_backend_abs_peak_mem_mib: Optional[float] = None
    pricing_peak_mem_mib: Optional[float] = None
    dual_feas_peak_mem_mib: Optional[float] = None
    dual_stabilization_mode: str = "off"
    dual_stabilization_alpha: float = 0.5
    dual_stabilization_gauge_normalization: bool = False
    dual_stable_previous: Optional[np.ndarray] = None
    peak_active_support_size: int = 0


@dataclass
class LPIteration:
    """
    CN: 一次 restricted LP 后供 certificate 与 support update 直接复用的结果。
    EN: Result of one restricted LP, reused directly by the certificate and support update.
    """

    index: int
    started_at: float
    finalize_started_at: float
    active_support_before_lp: int
    pack: Dict[str, Any]
    primal: Any
    dual: Any
    result: Any
    pricing_dual: np.ndarray
    stabilization: Dict[str, Any]
    state_update_time: float
    trace_context: Any


@dataclass
class OptimalityCertificate:
    """
    CN: 全边集 optimality certificate，并保留 fused scan 已产生的候选边。
    EN: Full-edge optimality certificate retaining candidate edges produced by the fused scan.
    """

    converged: bool
    dual_feasibility: float
    dual_feasibility_source: Optional[str]
    scan_time: float
    stopping_dual_feasibility: float
    stopping_passed: bool
    feasibility_passed: bool
    diagnostics: Dict[str, Any]
    scan_result: Any
    convergence_time: float


@dataclass
class SupportUpdate:
    """
    CN: dual-violation insertion 与 budgeted pruning 的单轮结果。
    EN: Per-iteration result of dual-violation insertion and budgeted pruning.
    """

    pricing_info: Dict[str, Any]
    pricing_time: float
    pruning_time: float
    pruning_info: Dict[str, Any]
    support_after_dual_violation: int
    support_after_budgeted_pruning: int
    added_keys: Any = None
    pruned_keys: Any = None


def _iteration_full_scan_time(
    certificate: OptimalityCertificate, update: SupportUpdate
) -> float:
    """
    CN: 汇总一轮中所有不重叠的完整边集扫描；fused 候选复用时只计一次。
    EN: Sum non-overlapping full-edge scans in an iteration, counting a reused fused scan once.
    """
    scan_candidates_reused = bool(
        certificate.scan_result is not None
        and certificate.scan_result.has_candidate_pairs
    )
    separate_violation_scan = (
        0.0
        if certificate.converged or scan_candidates_reused
        else float(update.pricing_time)
    )
    return float(certificate.scan_time) + separate_violation_scan


def begin_refinement(
    solver: HierarchicalOTSolver,
    config: SolverRuntimeConfig,
    warm_start_dual: Optional[np.ndarray],
    *,
    trace_collector: Optional[_ChromeTraceCollector] = None,
    trace_prefix: str = "solve_ot",
    warm_start_profile_depth: Optional[int] = None,
    dual_feasibility_norm: DualFeasibilityNorm = "l2",
) -> RefinementRuntime:
    """
    CN: 建立一个单层 refinement runtime，不复制 active support。
    EN: Begin one level-refinement runtime without copying the active support.
    """
    runtime = RefinementRuntime(
        solver=solver,
        config=config,
        trace_collector=trace_collector,
        trace_prefix=str(trace_prefix),
        warm_start_profile_depth=warm_start_profile_depth,
        stopping_norm=normalize_dual_feasibility_norm(dual_feasibility_norm),
        level_source=solver.hierarchy_s.finest_level,
        level_target=solver.hierarchy_t.finest_level,
        lp_tolerance=config.normalized_tolerance(),
        last_dual=(
            np.asarray(warm_start_dual, dtype=np.float64).copy()
            if warm_start_dual is not None
            else None
        ),
        last_dual_feasibility_tol=float(getattr(config, "dual_feasibility_tol", 1e-6)),
        dual_stabilization_mode=str(
            getattr(config, "dual_stabilization_mode", "off")
        ).strip().lower(),
        dual_stabilization_alpha=float(
            getattr(config, "dual_stabilization_alpha", 0.5)
        ),
        dual_stabilization_gauge_normalization=bool(
            getattr(config, "dual_stabilization_gauge_normalization", False)
        ),
        peak_active_support_size=int(getattr(solver.active_support, "size", 0)),
    )
    return runtime


def solve_lp(runtime: RefinementRuntime, iteration: int) -> LPIteration:
    """
    CN: 求解当前 active support 上的 restricted LP，并原地提交 primal/dual 状态。
    EN: Solve the restricted LP on the current active support and commit primal/dual state in place.
    """
    solver = runtime.solver
    trace_collector = runtime.trace_collector
    trace_prefix = runtime.trace_prefix
    inner_iter = int(iteration)
    started_at = time.perf_counter()
    support_before = int(getattr(solver.active_support, "size", 0))
    runtime.peak_active_support_size = max(runtime.peak_active_support_size, support_before)
    iteration_trace = (
        trace_collector.span(
            f"{trace_prefix}.iter_total",
            "solve_ot",
            args={"inner_iter": int(inner_iter + 1)},
        )
        if trace_collector is not None
        else nullcontext()
    )
    iteration_trace.__enter__()

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
        lp_pack = _solve_restricted_lp(
            solver,
            runtime.level_source,
            runtime.level_target,
            runtime.lp_tolerance,
            warm_start_dual=runtime.last_dual,
            trace_collector=trace_collector,
            trace_prefix=f"{trace_prefix}.lp",
            trace_args={
                "level": 0,
                "inner_iter": int(inner_iter + 1),
                "phase": "warm_start_lp",
            },
        )
        _trace_device_synchronize(trace_collector)
    if not lp_pack["success"]:
        iteration_trace.__exit__(None, None, None)
        raise RuntimeError(
            f"Warm-start single-level LP failed at inner_iter={inner_iter}."
        )

    runtime.lp_backend_peak_mem_mib = _max_optional_float(
        runtime.lp_backend_peak_mem_mib,
        lp_pack.get("lp_backend_peak_mem_mib"),
    )
    runtime.lp_backend_abs_peak_mem_mib = _max_optional_float(
        runtime.lp_backend_abs_peak_mem_mib,
        lp_pack.get("lp_backend_abs_peak_mem_mib"),
    )
    primal = lp_pack["primal"]
    dual = lp_pack["dual"]
    result = lp_pack["res"]
    for values in (result.x, dual):
        finite = bool(torch.isfinite(values).all().item()) if torch.is_tensor(values) else bool(np.isfinite(values).all())
        if not finite:
            iteration_trace.__exit__(None, None, None)
            raise RuntimeError("restricted LP returned non-finite primal or dual values")
    if not np.isfinite(result.obj_val):
        iteration_trace.__exit__(None, None, None)
        raise RuntimeError("restricted LP returned a non-finite objective")

    from hello_ot.refinement.loop import _dual_stabilization_pricing_dual

    with (
        trace_collector.span(
            f"{trace_prefix}.dual_stabilization",
            "solve_ot",
            args={"inner_iter": int(inner_iter + 1)},
        )
        if trace_collector is not None
        else nullcontext()
    ):
        pricing_dual, runtime.dual_stable_previous, stabilization = (
            _dual_stabilization_pricing_dual(
                dual,
                runtime.dual_stable_previous,
                n_source=len(runtime.level_source.points),
                mode=runtime.dual_stabilization_mode,
                alpha=runtime.dual_stabilization_alpha,
                gauge_normalization=runtime.dual_stabilization_gauge_normalization,
            )
        )

    finalize_started_at = time.perf_counter()
    state_started_at = time.perf_counter()
    with (
        trace_collector.span(
            f"{trace_prefix}.state_update",
            "solve_ot",
            args={"inner_iter": int(inner_iter + 1)},
        )
        if trace_collector is not None
        else nullcontext()
    ):
        runtime.objective_history.append(float(result.obj_val))
        runtime.lp_time += float(lp_pack["lp_time"])
        runtime.last_primal = primal
        runtime.last_dual = dual
        runtime.last_result = result
        solver.active_support.set_x_prev(
            _copy_primal_for_active_support(solver.active_support, result.x)
        )

    return LPIteration(
        index=inner_iter,
        started_at=started_at,
        finalize_started_at=finalize_started_at,
        active_support_before_lp=support_before,
        pack=lp_pack,
        primal=primal,
        dual=dual,
        result=result,
        pricing_dual=pricing_dual,
        stabilization=stabilization,
        state_update_time=float(time.perf_counter() - state_started_at),
        trace_context=iteration_trace,
    )


def check_optimality(
    runtime: RefinementRuntime,
    iteration: LPIteration,
) -> OptimalityCertificate:
    """
    CN: 扫描完整边集形成 optimality certificate；fused 候选保存在返回值中。
    EN: Scan the full edge set for an optimality certificate; fused candidates remain in the result.
    """
    from hello_ot.refinement.loop import _run_dual_feasibility_scan_for_solver

    config = runtime.config
    solver = runtime.solver
    trace_collector = runtime.trace_collector
    trace_prefix = runtime.trace_prefix
    inner_iter = int(iteration.index)
    started_at = time.perf_counter()
    cost_type = str(getattr(config, "cost_type", "lowrank")).lower()
    use_fused = cost_type in {"l1", "linf", "l2"} or bool(
        getattr(config, "use_fused_lowrank_feasibility_pricing", True)
    )
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
                    "use_fused_lowrank_feasibility_pricing": bool(use_fused),
                },
            )
            if trace_collector is not None
            else nullcontext()
        ):
            scan_result = _run_dual_feasibility_scan_for_solver(
                solver=solver,
                config=config,
                lvl_s=runtime.level_source,
                lvl_t=runtime.level_target,
                dual_uv=iteration.pricing_dual,
                inner_iter=inner_iter,
                trace_collector=trace_collector,
                trace_prefix=trace_prefix,
                dual_feasibility_norm=runtime.stopping_norm,
            )
    scan_result.diagnostics.update(dict(support_staging_profile))
    diagnostics = dict(scan_result.diagnostics)
    value = float(scan_result.dual_feasibility)
    scan_time = float(scan_result.scan_time)
    runtime.dual_feas_peak_mem_mib = _max_optional_float(
        runtime.dual_feas_peak_mem_mib,
        scan_result.peak_mem_mib,
    )
    feasibility_passed = bool(
        value <= float(getattr(config, "dual_feasibility_tol", 1e-6))
    )
    stopping_value = stopping_dual_feasibility(
        norm=runtime.stopping_norm,
        l2_value=value,
        diagnostics=diagnostics,
    )
    stopping_passed = bool(
        float(stopping_value)
        <= float(getattr(config, "dual_feasibility_tol", 1e-6))
    )
    convergence_time = float(time.perf_counter() - started_at)
    if bool(use_fused) and scan_result.has_candidate_pairs:
        convergence_time = max(0.0, convergence_time - scan_time)
    certificate = OptimalityCertificate(
        converged=stopping_passed,
        dual_feasibility=value,
        dual_feasibility_source=scan_result.dual_feasibility_source,
        scan_time=scan_time,
        stopping_dual_feasibility=float(stopping_value),
        stopping_passed=stopping_passed,
        feasibility_passed=feasibility_passed,
        diagnostics=diagnostics,
        scan_result=scan_result,
        convergence_time=convergence_time,
    )
    record_solve_event(
        "check_optimality",
        iteration=int(inner_iter + 1),
        dual_feasibility=float(stopping_value),
        converged=bool(stopping_passed),
    )
    if certificate.converged:
        _complete_iteration(
            runtime,
            iteration,
            certificate,
            _converged_support_update(runtime, certificate, inner_iter),
        )
    return certificate


def update_support(
    runtime: RefinementRuntime,
    iteration: LPIteration,
    certificate: OptimalityCertificate,
    *,
    track_reentry: bool = False,
) -> SupportUpdate:
    """
    CN: 复用 certificate 候选插入 dual violations，再执行 budgeted pruning。
    EN: Reuse certificate candidates to insert dual violations, then run budgeted pruning.
    """
    if certificate.converged:
        raise ValueError("update_support must not run after a converged certificate")
    solver = runtime.solver
    config = runtime.config
    trace_collector = runtime.trace_collector
    trace_prefix = runtime.trace_prefix
    inner_iter = int(iteration.index)
    scan_result = certificate.scan_result

    with (
        trace_collector.span(
            f"{trace_prefix}.pricing",
            "solve_ot",
            args={"inner_iter": int(inner_iter + 1)},
        )
        if trace_collector is not None
        else nullcontext()
    ):
        _trace_device_synchronize(trace_collector)
        if scan_result is not None and scan_result.has_candidate_pairs:
            fused_extra_info = scan_result.pricing_extra_info()
            if scan_result.peak_mem_mib is not None:
                fused_extra_info["dual_feas_peak_mem_source"] = "driver_mem_get_info"
                fused_extra_info["pricing_peak_mem_mib"] = float(scan_result.peak_mem_mib)
                fused_extra_info["pricing_peak_mem_source"] = "driver_mem_get_info"
            pricing_info = append_candidate_pairs(
                solver,
                runtime.level_source,
                runtime.level_target,
                scan_result.rows,
                scan_result.cols,
                iteration.pricing_dual,
                inner_iter,
                pricing_duration=float(certificate.scan_time),
                active_before=int(getattr(solver.active_support, "size", 0)),
                trace_collector=trace_collector,
                trace_prefix=f"{trace_prefix}.pricing",
                report_added_violation_stats=bool(
                    getattr(config, "report_added_violation_stats", False)
                ),
                added_violation_rel_threshold=float(
                    getattr(config, "added_violation_rel_threshold", 1e-6)
                ),
                extra_info=fused_extra_info,
            )
        else:
            pricing_info = detect_and_append_dual_violations(
                solver,
                runtime.level_source,
                runtime.level_target,
                iteration.primal,
                iteration.pricing_dual,
                inner_iter,
                trace_collector=trace_collector,
                trace_prefix=f"{trace_prefix}.pricing",
                report_added_violation_stats=bool(
                    getattr(config, "report_added_violation_stats", False)
                ),
                added_violation_rel_threshold=float(
                    getattr(config, "added_violation_rel_threshold", 1e-6)
                ),
            )
        _trace_device_synchronize(trace_collector)
    pricing_time = float(pricing_info.get("time", 0.0))
    runtime.pricing_peak_mem_mib = _max_optional_float(
        runtime.pricing_peak_mem_mib,
        pricing_info.get("pricing_peak_mem_mib"),
    )
    pricing_info.update(
        {
            "dual_feasibility": float(certificate.dual_feasibility),
            "dual_feasibility_source": certificate.dual_feasibility_source,
            "dual_feasibility_scan_time": float(certificate.scan_time),
        }
    )
    runtime.pricing_time += pricing_time

    from hello_ot.refinement.loop import (
        _attach_dual_stabilization_info,
        _build_cleaning_profile_info,
    )

    pruning_started_at = time.perf_counter()
    removed_keys = [] if track_reentry else None
    support_after_violation = int(getattr(solver.active_support, "size", 0))
    pruning_info = _build_cleaning_profile_info(
        solver,
        n_s=len(runtime.level_source.points),
        n_t=len(runtime.level_target.points),
        before=support_after_violation,
        after=support_after_violation,
        current_inner_iter=inner_iter,
    )
    with (
        trace_collector.span(
            f"{trace_prefix}.cleaning",
            "solve_ot",
            args={"inner_iter": int(inner_iter + 1)},
        )
        if trace_collector is not None
        else nullcontext()
    ):
        removed_indices = apply_budgeted_pruning(
            solver,
            iteration.dual,
            len(runtime.level_source.points),
            len(runtime.level_target.points),
            trace_collector=trace_collector,
            trace_prefix=f"{trace_prefix}.cleaning",
            current_inner_iter=inner_iter,
            removed_keys_out=removed_keys,
        )
    pruning_time = float(time.perf_counter() - pruning_started_at)
    support_after_pruning = int(getattr(solver.active_support, "size", 0))
    record_solve_event(
        "update_support",
        iteration=int(inner_iter + 1),
        support_after_dual_violation=support_after_violation,
        support_after_budgeted_pruning=support_after_pruning,
    )
    pruning_info["after"] = support_after_pruning
    pruning_info["removed"] = max(0, support_after_violation - support_after_pruning)
    pruning_info["budget_overflow"] = max(
        0, support_after_pruning - int(pruning_info.get("budget", 0))
    )
    pruning_info["budget_overflow_ratio"] = float(
        pruning_info["budget_overflow"]
    ) / float(max(1, len(runtime.level_source.points) + len(runtime.level_target.points)))
    removed = np.asarray(removed_indices, dtype=np.int64).reshape(-1)
    old_zero_mask = pruning_info.get("_old_zero_mask")
    old_pool_mask = pruning_info.get("_old_pool_mask")
    if (
        removed.size > 0
        and isinstance(old_zero_mask, np.ndarray)
        and isinstance(old_pool_mask, np.ndarray)
    ):
        valid = removed[(removed >= 0) & (removed < int(old_zero_mask.size))]
        removed_zero_mask = old_zero_mask[valid]
        removed_old_pool_mask = old_pool_mask[valid]
        pruning_info["removed_old_zero_flow"] = int(np.count_nonzero(removed_zero_mask))
        pruning_info["removed_old_nonzero_flow"] = int(
            np.count_nonzero(removed_old_pool_mask & (~removed_zero_mask))
        )
    pruning_info.pop("_old_zero_mask", None)
    pruning_info.pop("_old_pool_mask", None)
    _attach_dual_stabilization_info(
        pricing_info,
        iteration.stabilization,
        dual_feasibility_present=True,
    )
    update = SupportUpdate(
        pricing_info=dict(pricing_info),
        pricing_time=pricing_time,
        pruning_time=pruning_time,
        pruning_info=pruning_info,
        support_after_dual_violation=support_after_violation,
        support_after_budgeted_pruning=support_after_pruning,
    )
    if track_reentry:
        from .reentry import edge_keys

        update.added_keys = edge_keys(
            pricing_info["added_rows"], pricing_info["added_cols"], len(runtime.level_target.points)
        )
        update.pruned_keys = removed_keys[0] if removed_keys else None
    _complete_iteration(runtime, iteration, certificate, update)
    return update


def finish_interrupted_iteration(
    runtime: RefinementRuntime, iteration: LPIteration, certificate: OptimalityCertificate
) -> None:
    """
    CN: 成本切换前完成当前 LP 和证书记录，跳过插入与剪枝。
    EN: Record the current LP and certificate before switching costs, without insertion or pruning.
    """
    update = _converged_support_update(runtime, certificate, iteration.index)
    update.pricing_info["reason"] = "cost_perturbation_requested"
    _complete_iteration(runtime, iteration, certificate, update)


def _converged_support_update(
    runtime: RefinementRuntime,
    certificate: OptimalityCertificate,
    inner_iter: int,
) -> SupportUpdate:
    from hello_ot.refinement.loop import _build_cleaning_profile_info

    solver = runtime.solver
    scan_result = certificate.scan_result
    pricing_info: Dict[str, Any] = {
        "time": 0.0,
        "skipped": True,
        "reason": "dual_feasible_before_pricing",
        "dual_feasibility": float(certificate.dual_feasibility),
        "dual_feasibility_source": certificate.dual_feasibility_source,
        "dual_feasibility_scan_time": float(certificate.scan_time),
    }
    pricing_info.update(
        {
            key: value
            for key, value in certificate.diagnostics.items()
            if key.startswith("source_F_")
            or key.startswith("target_G_")
            or key.startswith("fused_scan_")
            or key.startswith("active_support_")
            or key
            in {
                "resident_side",
                "chunk_rows",
                "dual_feas_entry_used_mem_mib",
                "dual_feas_after_resident_upload_used_mem_mib",
                "dual_feas_exit_used_mem_mib",
            }
        }
    )
    if scan_result is not None and scan_result.peak_mem_mib is not None:
        pricing_info["dual_feas_peak_mem_mib"] = float(scan_result.peak_mem_mib)
        pricing_info["dual_feas_peak_mem_source"] = "driver_mem_get_info"
    support_size = int(getattr(solver.active_support, "size", 0))
    pruning_info = _build_cleaning_profile_info(
        solver,
        n_s=len(runtime.level_source.points),
        n_t=len(runtime.level_target.points),
        before=support_size,
        after=support_size,
        current_inner_iter=inner_iter,
    )
    return SupportUpdate(
        pricing_info=pricing_info,
        pricing_time=0.0,
        pruning_time=0.0,
        pruning_info=pruning_info,
        support_after_dual_violation=support_size,
        support_after_budgeted_pruning=support_size,
    )


def _complete_iteration(
    runtime: RefinementRuntime,
    iteration: LPIteration,
    certificate: OptimalityCertificate,
    update: SupportUpdate,
) -> None:
    from hello_ot.refinement.loop import (
        _attach_dual_stabilization_info,
        _compact_pricing_info,
    )

    runtime.peak_active_support_size = max(
        runtime.peak_active_support_size,
        update.support_after_dual_violation,
        update.support_after_budgeted_pruning,
    )
    _attach_dual_stabilization_info(
        update.pricing_info,
        iteration.stabilization,
        dual_feasibility_present=True,
    )
    pricing_added = (
        None
        if certificate.converged
        else int(update.pricing_info.get("added", 0))
    )
    stalled = bool(not certificate.stopping_passed and pricing_added == 0)
    runtime.consecutive_no_new_edge_iterations = (
        runtime.consecutive_no_new_edge_iterations + 1 if stalled else 0
    )
    diagnostics = certificate.diagnostics
    convergence_info = {
        "signed_rel_obj_change": (
            (float(runtime.objective_history[-1]) - float(runtime.objective_history[-2]))
            / (abs(float(runtime.objective_history[-2])) + 1e-9)
            if len(runtime.objective_history) >= 2
            else None
        ),
        "is_converged": bool(certificate.converged),
        "objective_converged": False,
        "criterion": "dual_feasibility",
        "plateau_counter": 0,
        "required_plateau": None,
        "objective_tol": None,
        "require_dual_feasibility_convergence": False,
        "dual_feasibility": float(certificate.dual_feasibility),
        "dual_feasibility_num_sq": diagnostics.get("dual_feasibility_num_sq"),
        "dual_feasibility_den_sq": diagnostics.get("dual_feasibility_den_sq"),
        "dual_feasibility_max_violation": diagnostics.get("dual_feasibility_max_violation"),
        "dual_feasibility_cost_linf": diagnostics.get("dual_feasibility_cost_linf"),
        "relative_linf_dual_feasibility": diagnostics.get("relative_linf_dual_feasibility"),
        "dual_feasibility_source": certificate.dual_feasibility_source,
        "dual_feasibility_tol": float(getattr(runtime.config, "dual_feasibility_tol", 1e-6)),
        "dual_feasibility_passed": bool(certificate.feasibility_passed),
        "finest_dual_feasibility_norm": str(runtime.stopping_norm),
        "stopping_dual_feasibility": float(certificate.stopping_dual_feasibility),
        "stopping_dual_feasibility_tol": float(getattr(runtime.config, "dual_feasibility_tol", 1e-6)),
        "stopping_dual_feasibility_passed": bool(certificate.stopping_passed),
        "stopping_stalled_no_new_edges": stalled,
        "consecutive_no_new_edge_iterations": int(runtime.consecutive_no_new_edge_iterations),
        "require_added_convergence": False,
        "previous_pricing_added": None,
        "added_convergence_threshold": None,
        "added_convergence_ratio": None,
        "added_convergence_passed": None,
        "dual_stabilization_mode": str(iteration.stabilization.get("mode", "off")),
        "dual_stabilization_alpha": float(iteration.stabilization.get("alpha", 0.0)),
        "dual_stabilization_enabled": bool(iteration.stabilization.get("enabled", False)),
        "dual_stabilization_applied": bool(iteration.stabilization.get("applied", False)),
        "dual_stabilization_reset": bool(iteration.stabilization.get("reset", False)),
        "dual_stabilization_delta_rel": float(iteration.stabilization.get("delta_rel", 0.0)),
        "dual_stabilization_gauge_normalization": bool(
            iteration.stabilization.get("gauge_normalization", False)
        ),
        "dual_stabilization_gauge_shift": float(iteration.stabilization.get("gauge_shift", 0.0)),
        "dual_feasibility_dual_stabilized": bool(
            iteration.stabilization.get("applied", False)
        ),
    }
    runtime.last_dual_feasibility_value = float(certificate.dual_feasibility)
    runtime.last_dual_feasibility_num_sq = diagnostics.get("dual_feasibility_num_sq")
    runtime.last_dual_feasibility_den_sq = diagnostics.get("dual_feasibility_den_sq")
    runtime.last_dual_feasibility_max_violation = diagnostics.get(
        "dual_feasibility_max_violation"
    )
    runtime.last_dual_feasibility_cost_linf = diagnostics.get(
        "dual_feasibility_cost_linf"
    )
    runtime.last_relative_linf_dual_feasibility = diagnostics.get(
        "relative_linf_dual_feasibility"
    )
    runtime.last_dual_feasibility_source = certificate.dual_feasibility_source
    runtime.last_dual_feasibility_tol = float(
        getattr(runtime.config, "dual_feasibility_tol", 1e-6)
    )
    runtime.last_dual_feasibility_passed = bool(certificate.feasibility_passed)
    runtime.last_stopping_dual_feasibility = float(
        certificate.stopping_dual_feasibility
    )
    runtime.last_stopping_dual_feasibility_passed = bool(certificate.stopping_passed)
    runtime.last_stopping_stalled_no_new_edges = stalled
    full_scan_time = _iteration_full_scan_time(certificate, update)
    runtime.full_scan_time += full_scan_time
    finalize_components = {
        "pricing": float(update.pricing_time),
        "cleaning": float(update.pruning_time),
        "state_update": float(iteration.state_update_time),
        "convergence_check": float(certificate.convergence_time),
    }
    finalize_time = float(time.perf_counter() - iteration.finalize_started_at)
    with (
        runtime.trace_collector.span(
            f"{runtime.trace_prefix}.print_iter_profile",
            "solve_ot",
            args={"inner_iter": int(iteration.index + 1)},
        )
        if runtime.trace_collector is not None
        else nullcontext()
    ):
        _print_warm_start_iter_profile(
            runtime.solver,
            inner_iter=iteration.index,
            lp_pack=iteration.pack,
            pricing_info=update.pricing_info,
            convergence_info=convergence_info,
            cleaning_info=update.pruning_info,
            finalize_components=finalize_components,
            iter_wall=float(time.perf_counter() - iteration.started_at),
            solve_t=float(iteration.pack["lp_time"]),
            finalize_t=finalize_time,
            profile_depth=runtime.warm_start_profile_depth,
            dual_stabilization_info=iteration.stabilization,
        )
    runtime.iteration_records.append(
        {
            "iter": int(iteration.index + 1),
            "objective": float(iteration.result.obj_val),
            "lp_time": float(iteration.pack["lp_time"]),
            "full_scan_time": full_scan_time,
            "lp_iters": (
                int(iteration.pack["res"].iterations)
                if getattr(iteration.pack.get("res"), "iterations", None) is not None
                else None
            ),
            "lp_diag": dict(iteration.pack.get("diag", {}) or {}),
            "active_support_before_lp": int(iteration.active_support_before_lp),
            "active_support_after_pricing": int(update.support_after_dual_violation),
            "active_support_after_cleaning": int(update.support_after_budgeted_pruning),
            "pricing_info": _compact_pricing_info(update.pricing_info),
            "cleaning_info": dict(update.pruning_info),
            "convergence_info": dict(convergence_info),
            "finalize_components": dict(finalize_components),
            "wall_time": float(time.perf_counter() - iteration.started_at),
        }
    )
    if certificate.converged:
        _runtime_log(
            runtime.solver,
            "warm_start",
            f"[WarmStartSolve] stop at iter={iteration.index + 1}/{int(runtime.config.max_inner_iter)}",
        )
    iteration.trace_context.__exit__(None, None, None)


def finalize_refinement(runtime: RefinementRuntime) -> Dict[str, Any]:
    """
    CN: 从原地更新后的 runtime 构造该层 refinement 结果。
    EN: Build the level-refinement result from the runtime updated in place.
    """
    if runtime.last_primal is None or runtime.last_dual is None or runtime.last_result is None:
        raise RuntimeError("Warm-start single-level loop did not produce a solution.")
    with (
        runtime.trace_collector.span(
            f"{runtime.trace_prefix}.export_coupling", "solve_ot"
        )
        if runtime.trace_collector is not None
        else nullcontext()
    ):
        coupling = sparse_coupling_from_active_support(
            runtime.solver,
            (len(runtime.level_source.points), len(runtime.level_target.points)),
        )
    final_is_converged = bool(
        runtime.iteration_records
        and runtime.iteration_records[-1]["convergence_info"].get("is_converged", False)
    )
    hit_itermax = bool(
        not final_is_converged
        and len(runtime.objective_history) >= int(runtime.config.max_inner_iter)
    )
    level_summaries = [
        {
            "level": 0,
            "n_source": len(runtime.level_source.points),
            "n_target": len(runtime.level_target.points),
            "iters": len(runtime.objective_history),
            "time": float(time.perf_counter() - runtime.solve_started_at),
            "objective": float(runtime.last_result.obj_val),
            "lp_time": float(runtime.lp_time),
            "pricing_time": float(runtime.pricing_time),
            "full_scan_time": float(runtime.full_scan_time),
            "support_final": int(coupling.nnz),
            "peak_active_support_size": int(runtime.peak_active_support_size),
            "lp_backend_peak_mem_mib": runtime.lp_backend_peak_mem_mib,
            "lp_backend_delta_peak_mem_mib": runtime.lp_backend_peak_mem_mib,
            "lp_backend_abs_peak_mem_mib": runtime.lp_backend_abs_peak_mem_mib,
            "pricing_peak_mem_mib": runtime.pricing_peak_mem_mib,
            "dual_feas_peak_mem_mib": runtime.dual_feas_peak_mem_mib,
            "dual_feas_peak_mem_source": (
                "driver_mem_get_info"
                if runtime.dual_feas_peak_mem_mib is not None
                else None
            ),
            "dual_feasibility": runtime.last_dual_feasibility_value,
            "dual_feasibility_num_sq": runtime.last_dual_feasibility_num_sq,
            "dual_feasibility_den_sq": runtime.last_dual_feasibility_den_sq,
            "dual_feasibility_max_violation": runtime.last_dual_feasibility_max_violation,
            "dual_feasibility_cost_linf": runtime.last_dual_feasibility_cost_linf,
            "relative_linf_dual_feasibility": runtime.last_relative_linf_dual_feasibility,
            "dual_feasibility_source": runtime.last_dual_feasibility_source,
            "dual_feasibility_tol": runtime.last_dual_feasibility_tol,
            "dual_feasibility_passed": runtime.last_dual_feasibility_passed,
            "finest_dual_feasibility_norm": str(runtime.stopping_norm),
            "stopping_dual_feasibility": runtime.last_stopping_dual_feasibility,
            "stopping_dual_feasibility_tol": runtime.last_dual_feasibility_tol,
            "stopping_dual_feasibility_passed": runtime.last_stopping_dual_feasibility_passed,
            "stopping_stalled_no_new_edges": runtime.last_stopping_stalled_no_new_edges,
            "consecutive_no_new_edge_iterations": runtime.consecutive_no_new_edge_iterations,
            "converged": final_is_converged,
            "hit_itermax": hit_itermax,
            "converged_before_itermax": final_is_converged and not hit_itermax,
            **relative_kkt_fields(
                primal_feasibility=getattr(runtime.last_result, "primal_feas", None),
                primal_dual_gap=getattr(runtime.last_result, "gap", None),
                full_dual_feasibility=runtime.last_dual_feasibility_value,
            ),
        }
    ]
    dual_out = np.asarray(runtime.last_dual, dtype=np.float64).copy()
    return {
        "distance": float(runtime.last_result.obj_val),
        "dual": dual_out,
        "coupling": coupling,
        "level_summaries": level_summaries,
        "solve_time": float(time.perf_counter() - runtime.solve_started_at),
        "lp_backend_peak_mem_mib": runtime.lp_backend_peak_mem_mib,
        "lp_backend_delta_peak_mem_mib": runtime.lp_backend_peak_mem_mib,
        "lp_backend_abs_peak_mem_mib": runtime.lp_backend_abs_peak_mem_mib,
        "pricing_peak_mem_mib": runtime.pricing_peak_mem_mib,
        "dual_feas_peak_mem_mib": runtime.dual_feas_peak_mem_mib,
        "peak_active_support_size": int(runtime.peak_active_support_size),
        "iteration_records": runtime.iteration_records,
    }


__all__ = [
    "LPIteration",
    "OptimalityCertificate",
    "RefinementRuntime",
    "SupportUpdate",
    "begin_refinement",
    "check_optimality",
    "finalize_refinement",
    "solve_lp",
    "update_support",
]
