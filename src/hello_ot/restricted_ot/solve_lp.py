from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from hello_ot._internal.types.runtime import StepResult
from hello_ot._internal.instrumentation.phase_names import (
    COMPONENT_CLEANING,
    COMPONENT_STATE_UPDATE,
)
from hello_ot._internal.runtime_context import record_solve_event
from hello_ot.refinement import budgeted_pruning as cleaning
from hello_ot.refinement import dual_violation as pricing
from hello_ot.refinement.dual_feasibility import run_lowrank_dual_feasibility_scan
from hello_ot.restricted_ot.backend import solve_lp as _solve_lp_backend


def _copy_primal_for_active_support(active_support: Any, value: Any) -> Any:
    """
    CN: 按 active-support backend 复制 primal；GPU 走 D2D，CPU 仅在该边界走 D2H。
    EN: Copy the primal for the active-support backend; use D2D on GPU and D2H only at the CPU boundary.
    """
    if hasattr(value, "detach"):
        tensor = value.detach()
        if bool(getattr(active_support, "is_torch_backend", False)):
            return tensor.clone()
        return tensor.cpu().numpy().copy()
    return value.copy()


@dataclass(frozen=True)
class SolveLPResult:
    """
    CN: 一次 restricted OT 求解的无副作用结果；由 refinement 决定何时提交 primal。
    EN: Side-effect-free result of one restricted OT solve; refinement decides when to commit the primal.
    """

    success: bool
    primal: Any = None
    dual: Any = None
    backend_result: Any = None
    wall_time: float = 0.0
    components: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    lp_backend_abs_peak_mem_mib: Optional[float] = None
    lp_backend_delta_peak_mem_mib: Optional[float] = None

    def as_legacy_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "primal": self.primal,
            "dual": self.dual,
            "res": self.backend_result,
            "lp_time": float(self.wall_time),
            "components": dict(self.components),
            "diag": dict(self.diagnostics),
            "lp_backend_abs_peak_mem_mib": self.lp_backend_abs_peak_mem_mib,
            "lp_backend_delta_peak_mem_mib": self.lp_backend_delta_peak_mem_mib,
            "lp_backend_peak_mem_mib": self.lp_backend_delta_peak_mem_mib,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_legacy_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_legacy_dict().get(key, default)


def solve_lp(
    solver,
    lvl_s: Any,
    lvl_t: Any,
    tolerance: Dict[str, float],
    warm_start_dual: Optional[np.ndarray] = None,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.lp",
    trace_args: Optional[Dict[str, Any]] = None,
    _require_gpu: bool = True,
) -> SolveLPResult:
    """
    CN: 在当前 active support 上执行一次 restricted LP；由 refinement 决定何时刷新 primal warm start。
    EN: Solve one restricted LP on the current active support; refinement decides when to refresh the primal warm start.
    """
    if (
        bool(_require_gpu)
        and str(getattr(solver, "_backend", "native")) == "native"
        and not bool(getattr(solver.active_support, "is_cuda", False))
    ):
        raise RuntimeError("formal HELLO SolveLP requires a CUDA active support")
    n_s, n_t = len(lvl_s.points), len(lvl_t.points)
    t_lp_start = time.perf_counter()
    primal_new, dual_new, res, solve_meta = _solve_lp_backend(
        solver,
        lvl_s,
        lvl_t,
        tolerance,
        warm_start_dual=warm_start_dual,
        verbose=solver.lp_solver_verbose,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        trace_args=trace_args,
    )
    lp_duration = time.perf_counter() - t_lp_start

    if not res.success:
        return SolveLPResult(
            success=False,
            wall_time=float(lp_duration),
            components=dict((solve_meta or {}).get("components", {})),
            diagnostics=dict((solve_meta or {}).get("diag", {})),
        )

    if solver._logger_is_enabled_info():
        primal_nnz = primal_new.nnz
        primal_nnz_thr = int(np.count_nonzero(primal_new.data > 1e-10)) if primal_new.data.size > 0 else 0
        solver._logger_info(
            "  [Primal] nnz=%s; nnz@1e-10=%s",
            pricing.fmt_size_ratio(primal_nnz, n_s, n_t),
            pricing.fmt_size_ratio(primal_nnz_thr, n_s, n_t),
        )

    diag = dict((solve_meta or {}).get("diag", {}) or {})
    solver_diag = getattr(res, "solver_diag", None)
    if not isinstance(solver_diag, dict):
        solver_diag = {}
    lp_backend_abs_peak = diag.get("lp_backend_abs_peak_mem_mib", solver_diag.get("lp_backend_abs_peak_mem_mib", getattr(res, "peak_mem", None)))
    lp_backend_peak = diag.get("lp_backend_delta_peak_mem_mib", solver_diag.get("lp_backend_delta_peak_mem_mib"))
    if lp_backend_peak is None and lp_backend_abs_peak is not None and diag.get("lp_backend_entry_mem_mib") is not None:
        lp_backend_peak = max(0.0, float(lp_backend_abs_peak) - float(diag["lp_backend_entry_mem_mib"]))
    if lp_backend_abs_peak is not None:
        diag["lp_backend_abs_peak_mem_mib"] = float(lp_backend_abs_peak)
    if lp_backend_peak is not None:
        diag["lp_backend_delta_peak_mem_mib"] = float(lp_backend_peak)
        diag["lp_backend_peak_mem_mib"] = float(lp_backend_peak)
    record_solve_event(
        "solve_lp",
        rows=solver.active_support.rows,
        cols=solver.active_support.cols,
        primal=primal_new,
        dual=dual_new,
    )
    return SolveLPResult(
        success=True,
        primal=primal_new,
        dual=dual_new,
        backend_result=res,
        wall_time=float(lp_duration),
        components=dict((solve_meta or {}).get("components", {})),
        diagnostics=diag,
        lp_backend_abs_peak_mem_mib=(None if lp_backend_abs_peak is None else float(lp_backend_abs_peak)),
        lp_backend_delta_peak_mem_mib=(None if lp_backend_peak is None else float(lp_backend_peak)),
    )


def solve_active_support_lp_and_commit(
    solver,
    lvl_s: Any,
    lvl_t: Any,
    tolerance: Dict[str, float],
    warm_start_dual: Optional[np.ndarray] = None,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.lp",
    trace_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    CN: restricted LP 成功后立即提交 primal 的层级调度入口。
    EN: Hierarchical-dispatch entry point that commits the primal immediately after a successful restricted LP.
    """
    result = solve_lp(
        solver,
        lvl_s,
        lvl_t,
        tolerance,
        warm_start_dual=warm_start_dual,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        trace_args=trace_args,
        _require_gpu=False,
    )
    if result.success:
        solver.active_support.set_x_prev(
            _copy_primal_for_active_support(solver.active_support, result.backend_result.x)
        )
    return result.as_legacy_dict()


def extract_step_objective(step_pack: Dict[str, Any]) -> Optional[float]:
    """
    CN: 从单步结果中提取 LP 目标值；若不可用则返回 None。
    EN: Extract the LP objective from a step pack, or return None when it is unavailable.
    """
    res = step_pack.get("res")
    objective = getattr(res, "obj_val", None)
    if objective is None:
        return None
    return float(objective)


def finalize_iteration(
    solver,
    level_state: Dict[str, Any],
    run_state: Dict[str, Any],
    step_pack: Dict[str, Any],
    convergence_criterion: str,
    tolerance: Dict[str, float],
    *,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.solve.finalize_iteration",
    config: Optional[Any] = None,
) -> None:
    """
    CN: 持久化本轮迭代结果，并执行 dual-violation detection 与 budgeted pruning。
    EN: Persist the current iteration result and run dual-violation detection plus budgeted pruning.
    """
    del run_state
    t_state = time.perf_counter()
    if level_state.get("is_coarsest"):
        level_state["primal_curr"] = step_pack["primal"]
        level_state["dual_curr"] = step_pack["dual"]
        level_state["level_obj_hist"] = [step_pack["res"].obj_val]
        level_state["level_lp_time"] = step_pack["lp_time"]
        level_state["level_pricing_time"] = 0.0
        step_pack["pricing_info"] = None
        step_pack["convergence_info"] = None
        return

    primal_curr = step_pack["primal"]
    dual_curr = step_pack["dual"]
    level_state["primal_curr"] = primal_curr
    level_state["dual_curr"] = dual_curr
    level_state["level_obj_hist"].append(step_pack["res"].obj_val)
    level_state["level_lp_time"] += step_pack["lp_time"]
    state_update_dt = time.perf_counter() - t_state
    dual_scan_result = None
    dual_feasibility_only = str(convergence_criterion) == "dual_feasibility"
    config_cost_type = str(getattr(config, "cost_type", "")).strip().lower()
    if (
        bool(dual_feasibility_only)
        and config is not None
        and config_cost_type == "lowrank"
    ):
        dual_scan_result = run_lowrank_dual_feasibility_scan(
            config=config,
            lvl_s=level_state["level_s"],
            lvl_t=level_state["level_t"],
            dual_uv=dual_curr,
            inner_iter=int(level_state["current_iter"]),
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
        )
        dual_feasibility_passed = bool(
            float(dual_scan_result.dual_feasibility)
            <= float(getattr(config, "dual_feasibility_tol", tolerance.get("dual_feasibility", 1e-6)))
        )
        if bool(dual_feasibility_passed):
            pricing_info = {
                "time": 0.0,
                "skipped": True,
                "reason": "dual_feasible_before_pricing",
            }
            pricing_info.update(dual_scan_result.pricing_extra_info())
            step_pack["pricing_info"] = pricing_info
            return

    if dual_scan_result is not None and dual_scan_result.has_candidate_pairs:
        fused_extra_info = dual_scan_result.pricing_extra_info()
        if dual_scan_result.peak_mem_mib is not None:
            fused_extra_info["pricing_peak_mem_mib"] = float(dual_scan_result.peak_mem_mib)
            fused_extra_info["pricing_peak_mem_source"] = "driver_mem_get_info"
        pricing_info = pricing.append_candidate_pairs(
            solver,
            level_state["level_s"],
            level_state["level_t"],
            dual_scan_result.rows,
            dual_scan_result.cols,
            dual_curr,
            level_state["current_iter"],
            pricing_duration=float(dual_scan_result.scan_time),
            active_before=int(getattr(solver.active_support, "size", 0)),
            trace_collector=trace_collector,
            trace_prefix=f"{trace_prefix}.pricing",
            report_added_violation_stats=bool(getattr(config, "report_added_violation_stats", False)),
            added_violation_rel_threshold=float(getattr(config, "added_violation_rel_threshold", 1e-6)),
            extra_info=fused_extra_info,
        )
    else:
        pricing_info = pricing.detect_and_append_dual_violations(
            solver,
            level_state["level_s"],
            level_state["level_t"],
            primal_curr,
            dual_curr,
            level_state["current_iter"],
            trace_collector=trace_collector,
            trace_prefix=f"{trace_prefix}.pricing",
        )
        if dual_scan_result is not None:
            pricing_info.update(dual_scan_result.pricing_extra_info())
    pricing_time = float(pricing_info.get("time", 0.0))
    level_state["level_pricing_time"] += pricing_time
    t_clean = time.perf_counter()
    cleaning.apply_budgeted_pruning(
        solver,
        dual_curr,
        len(level_state["level_s"].points),
        len(level_state["level_t"].points),
        trace_collector=trace_collector,
        trace_prefix=f"{trace_prefix}.cleaning",
        current_inner_iter=int(level_state["current_iter"]),
    )
    cleaning_dt = time.perf_counter() - t_clean
    step_pack["pricing_info"] = pricing_info


def prepare_inner(problem_def, algorithm_state, level_state) -> None:
    """
    CN: 为单轮 hierarchy 迭代预留预处理钩子；当前实现为 no-op。
    EN: Reserve a preprocessing hook for one hierarchy iteration; currently a no-op.
    """
    del problem_def, algorithm_state, level_state


def solve_hierarchy_iteration_lp(problem_def, algorithm_state, level_state) -> StepResult:
    """
    CN: 执行一次 active-support hierarchy 迭代并打包为 StepResult。
    EN: Run one active-support hierarchy iteration and package it as a StepResult.
    """
    solver = problem_def.solver
    level_data = level_state.data
    run_state = algorithm_state.run_state
    lp_trace_args = {
        "level": int(level_data["level_idx"]),
        "iter": int(level_data.get("current_iter", 0)),
        "inner_iter": int(level_data.get("current_iter", 0) + 1),
        "is_coarsest": bool(level_data.get("is_coarsest", False)),
        "phase": "solve_iteration_lp",
    }

    if level_data.get("is_coarsest"):
        lvl_s = level_data["level_s"]
        lvl_t = level_data["level_t"]
        t_start = time.perf_counter()
        primal_sol, dual_sol, result, solve_meta = solve_lp(
            solver,
            lvl_s,
            lvl_t,
            run_state["tolerance"],
            verbose=solver.lp_solver_verbose,
            trace_collector=getattr(problem_def, "trace_collector", None),
            trace_prefix=f"{getattr(problem_def, 'trace_prefix', 'solve_ot')}.solve.solve_iteration_lp.hello",
            trace_args=lp_trace_args,
        )
        if not result.success:
            step_data = {"success": False}
        else:
            total_time = time.perf_counter() - t_start
            step_data = {
                "success": True,
                "primal": primal_sol,
                "dual": dual_sol,
                "res": result,
                "lp_time": total_time,
                "pricing_time": 0.0,
                "components": {},
            }
    else:
        lp_pack = solve_active_support_lp_and_commit(
            solver,
            level_data["level_s"],
            level_data["level_t"],
            run_state["tolerance"],
            warm_start_dual=level_data["dual_curr"],
            trace_collector=getattr(problem_def, "trace_collector", None),
            trace_prefix=f"{getattr(problem_def, 'trace_prefix', 'solve_ot')}.solve.solve_iteration_lp.hello",
            trace_args=lp_trace_args,
        )
        if not lp_pack["success"]:
            step_data = {"success": False}
        else:
            step_data = {
                "success": True,
                "primal": lp_pack["primal"],
                "dual": lp_pack["dual"],
                "res": lp_pack["res"],
                "lp_time": lp_pack["lp_time"],
                "pricing_time": 0.0,
                "components": {},
            }

    return StepResult(
        success=bool(step_data.get("success", False)),
        data=step_data,
        objective=extract_step_objective(step_data),
    )
