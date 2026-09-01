from __future__ import annotations

import time
import sys
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from hello_ot._internal.trace import _ChromeTraceCollector
from hello_ot._internal.instrumentation.memory_accounting import use_solve_memory_tracker
from hello_ot.config import SolverRuntimeConfig

from .algorithm import _solve_hello_stage
from .cost import (
    _BilinearCostRuntime,
    _NormCostRuntime,
    _PreparedOTProblem,
    prepare_cost,
)
from .hierarchy.construction import resolve_hierarchy_depth
from .types import (
    ActiveSupportStats,
    BudgetedPruningStats,
    CoarsestSolveStats,
    ConvergenceStats,
    DualViolationStats,
    Result,
    InitializationStats,
    IterationStats,
    LevelResult,
    Problem,
    RefinementResult,
    RefinementSummary,
    SolveLPStats,
    SolveStageResult,
    SparseOTSolution,
    TraceResult,
)
from .config import _AlgorithmConfig
from .kernels.torch_scan import resolve_torch_device


def _runtime_config(
    config: _AlgorithmConfig,
    *,
    cost_type: str,
    dot_scale: float = 1.0,
) -> SolverRuntimeConfig:
    """
    CN: 把正式 HELLO 配置投影到内部 restricted-OT runtime 视图。
    EN: Project formal HELLO configuration onto the internal restricted-OT runtime view.
    """
    runtime = SolverRuntimeConfig.from_options(config, cost_type=cost_type)
    runtime.dot_scale = float(dot_scale)
    runtime.validate()
    return runtime


def _cost_runtime(problem: Problem | _PreparedOTProblem) -> _BilinearCostRuntime | _NormCostRuntime:
    """
    CN: 将正式或内部 problem 降低为唯一的 cost runtime。
    EN: Lower a formal or internal problem to the single cost runtime.
    """
    if isinstance(problem, Problem):
        return prepare_cost(problem)
    if isinstance(problem, _PreparedOTProblem):
        return problem.runtime
    raise TypeError("unsupported HELLO problem type")


def _reported_cost_type(problem: Problem | _PreparedOTProblem) -> str:
    if isinstance(problem, Problem):
        return str(problem.cost_type)
    if isinstance(problem, _PreparedOTProblem):
        return str(problem.reported_cost_type)
    raise TypeError("unsupported HELLO problem type")


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _optional_norm(value: Any) -> Optional[str]:
    parsed = str(value).strip().lower() if value is not None else ""
    return parsed if parsed in {"l2", "linf"} else None


def _synchronize_cuda_for_timing() -> None:
    """
    CN: 在正式 HELLO wall-time 边界同步当前 CUDA device，避免异步 kernel 落到计时区间之外。
    EN: Synchronize the current CUDA device at formal HELLO wall-time boundaries
        so asynchronous kernels stay inside the measured interval.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _require_cuda_runtime() -> None:
    """
    CN: 正式 HELLO 实现只支持 CUDA。
    EN: The formal HELLO implementation is CUDA-only.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("HELLO requires a CUDA-capable PyTorch runtime.")


def _iteration_stats(records: Iterable[Mapping[str, Any]], *, config: _AlgorithmConfig) -> tuple[IterationStats, ...]:
    output = []
    for record in records:
        pricing = dict(record.get("pricing_info") or {})
        cleaning = dict(record.get("cleaning_info") or {})
        convergence = dict(record.get("convergence_info") or {})
        lp_diag = dict(record.get("lp_diag") or {})
        components = dict(record.get("finalize_components") or {})
        before = int(record.get("active_support_before_lp", 0) or 0)
        after_pricing = int(record.get("active_support_after_pricing", before) or before)
        after_cleaning = int(record.get("active_support_after_cleaning", after_pricing) or after_pricing)
        budget = int(cleaning.get("budget", int(config.support_budget_factor * max(1, before))))
        output.append(
            IterationStats(
                index=int(record.get("iter", len(output) + 1)),
                objective=float(record.get("objective", float("nan"))),
                wall_time=float(record.get("lp_time", 0.0)) + sum(float(v or 0.0) for v in components.values()),
                bookkeeping_time=float(components.get("state_update", 0.0)) + float(components.get("convergence_check", 0.0)),
                solve_lp=SolveLPStats(
                    wall_time=float(record.get("lp_time", 0.0)),
                    backend_iterations=(None if record.get("lp_iters") is None else int(record["lp_iters"])),
                    primal_feasibility=_optional_float(lp_diag.get("solver_algorithm_rel_primal_res_l2")),
                    primal_dual_gap=_optional_float(lp_diag.get("solver_relative_primal_dual_gap")),
                    termination_norm=_optional_norm(lp_diag.get("solver_termination_norm")),
                    termination_reason=(
                        None
                        if lp_diag.get("solver_termination_reason") is None
                        else str(lp_diag.get("solver_termination_reason"))
                    ),
                    termination_primal_feasibility=_optional_float(
                        lp_diag.get("solver_termination_rel_primal_res")
                    ),
                    termination_dual_feasibility=_optional_float(
                        lp_diag.get("solver_termination_rel_dual_res")
                    ),
                    algorithm_dual_feasibility=_optional_float(
                        lp_diag.get("solver_algorithm_rel_dual_res_l2")
                    ),
                ),
                dual_violation=DualViolationStats(
                    wall_time=float(pricing.get("time", 0.0) or 0.0),
                    violations_found=int(pricing.get("found_count", pricing.get("found", 0)) or 0),
                    edges_added=int(pricing.get("added_count", pricing.get("added", 0)) or 0),
                ),
                budgeted_pruning=BudgetedPruningStats(
                    wall_time=float(components.get("cleaning", 0.0) or 0.0),
                    support_before=int(cleaning.get("before", after_pricing) or after_pricing),
                    support_after=int(cleaning.get("after", after_cleaning) or after_cleaning),
                    budget=budget,
                    budget_exceeded=bool(int(cleaning.get("budget_overflow", 0) or 0) > 0),
                ),
                support=ActiveSupportStats(
                    before_lp=before,
                    after_dual_violation=after_pricing,
                    after_budgeted_pruning=after_cleaning,
                    peak=max(before, after_pricing, after_cleaning),
                ),
                convergence=ConvergenceStats(
                    dual_feasibility=_optional_float(
                        convergence.get("stopping_dual_feasibility", convergence.get("dual_feasibility"))
                    ),
                    tolerance=float(config.dual_feasibility_tolerance),
                    passed=bool(convergence.get("is_converged", False)),
                    relative_full_dual_feasibility=_optional_float(
                        convergence.get("dual_feasibility")
                    ),
                    relative_linf_dual_feasibility=_optional_float(
                        convergence.get("relative_linf_dual_feasibility")
                    ),
                    max_dual_violation=_optional_float(
                        convergence.get("dual_feasibility_max_violation")
                    ),
                    norm=_optional_norm(convergence.get("finest_dual_feasibility_norm")),
                ),
            )
        )
    return tuple(output)


def _stage_result(
    raw: Mapping[str, Any],
    *,
    config: _AlgorithmConfig,
    wall_time: Optional[float] = None,
) -> SolveStageResult:
    diagnostics = dict(raw.get("hello_diagnostics") or {})
    raw_levels = list(diagnostics.get("levels") or [])
    levels = []
    for traversal_index, raw_level in enumerate(raw_levels):
        # CN: tuple 保持执行顺序 coarsest->finest；论文层号固定 finest=0。
        # EN: Keep tuple traversal coarsest->finest while paper level indices use finest=0.
        level_index = int(len(raw_levels) - 1 - traversal_index)
        kind = str(raw_level.get("kind", "node"))
        n_source = int(raw_level.get("n_source", 0))
        n_target = int(raw_level.get("n_target", 0))
        level_wall_time = float(raw_level.get("time", 0.0) or 0.0)
        if kind == "leaf":
            levels.append(
                LevelResult(
                    level_index=level_index,
                    kind="coarsest",
                    n_source=n_source,
                    n_target=n_target,
                    initialization=None,
                    solve=CoarsestSolveStats(
                        wall_time=level_wall_time,
                        objective=float((raw_level.get("solve_summary") or {}).get("distance", float("nan"))),
                        support_size=int(raw_level.get("support_size", 0)),
                    ),
                    wall_time=level_wall_time,
                )
            )
            continue
        solve_summary = dict(raw_level.get("solve_summary") or {})
        initialization = dict(solve_summary.get("initialization") or {})
        components = dict(initialization.get("components") or {})
        iterations = _iteration_stats(solve_summary.get("objective_trace") or [], config=config)
        converged = bool(iterations and iterations[-1].convergence.passed)
        support_before = int(raw_level.get("support_before_solve", 0) or 0)
        support_after_forward = int(raw_level.get("topk_forward_support_size", support_before) or support_before)
        reverse_value = raw_level.get("topk_reverse_support_size")
        levels.append(
            LevelResult(
                level_index=level_index,
                kind="refined",
                n_source=n_source,
                n_target=n_target,
                initialization=InitializationStats(
                    wall_time=sum(float(value or 0.0) for value in components.values()),
                    dual_propagation_time=float((raw_level.get("dual_completion_profile") or {}).get("total_time", 0.0) or 0.0),
                    dual_assignment_time=float((raw_level.get("augment_profile") or {}).get("total_time", 0.0) or 0.0),
                    northwest_augmentation_time=float(components.get("bfs_skeleton", 0.0) or 0.0),
                    inherited_dual_side=("target" if raw_level.get("split_axis") == "source" else "source"),
                    assignment_topk=int(config.assignment_topk),
                    support_after_forward=support_after_forward,
                    support_after_reverse=(None if reverse_value is None else int(reverse_value)),
                    northwest_edges_added=int(initialization.get("northwest_added", 0) or 0),
                    initial_active_support_size=int(initialization.get("post_pricing_support_size", support_before) or support_before),
                ),
                solve=RefinementResult(
                    iterations=iterations,
                    summary=RefinementSummary(
                        wall_time=float(solve_summary.get("time", 0.0) or 0.0),
                        iterations=len(iterations),
                        final_objective=float(solve_summary.get("distance", float("nan"))),
                        final_active_support_size=int(raw_level.get("support_after_solve", 0) or 0),
                        peak_active_support_size=max((item.support.peak for item in iterations), default=support_before),
                        stop_reason="converged" if converged else "max_refinement_iterations",
                        converged=converged,
                    ),
                ),
                wall_time=level_wall_time,
            )
        )
    measured = float(raw.get("time", 0.0) or 0.0) if wall_time is None else float(wall_time)
    return SolveStageResult(levels=tuple(levels), wall_time=measured)


def _solution(raw: Mapping[str, Any], *, shape: tuple[int, int]) -> SparseOTSolution:
    coupling = raw.get("sparse_coupling")
    state = raw.get("warm_start_state")
    if coupling is None or state is None or state.dual_uv is None:
        raise RuntimeError("HELLO runtime did not return the final coupling and dual potentials.")
    coo = coupling.tocoo(copy=False)
    dual_value = state.dual_uv
    if hasattr(dual_value, "detach"):
        dual_value = dual_value.detach().cpu().numpy()
    dual = np.asarray(dual_value, dtype=np.float64).reshape(-1)
    return SparseOTSolution(
        shape=shape,
        rows=np.asarray(coo.row, dtype=np.int64),
        cols=np.asarray(coo.col, dtype=np.int64),
        values=np.asarray(coo.data, dtype=np.float64),
        source_dual=dual[: shape[0]],
        target_dual=dual[shape[0] :],
    )


def _solve_raw(
    problem: Problem | _PreparedOTProblem,
    config: _AlgorithmConfig,
    *,
    tracer: Optional[_ChromeTraceCollector] = None,
) -> tuple[Dict[str, Any], Optional[TraceResult]]:
    runtime = _cost_runtime(problem)
    is_bilinear = isinstance(runtime, _BilinearCostRuntime)
    cost_type = "lowrank" if is_bilinear else str(runtime.cost_type)
    runtime_config = _runtime_config(
        config,
        cost_type=cost_type,
        dot_scale=float(runtime.dot_scale) if is_bilinear else 1.0,
    )
    depth = resolve_hierarchy_depth(
        "auto",
        n_source=problem.shape[0],
        n_target=problem.shape[1],
        split_count=int(config.split_count),
        coarsest_size_threshold=int(config.coarsest_size_threshold),
    )
    if tracer is None and config.record_trace:
        tracer = _ChromeTraceCollector(enabled=True)
    if is_bilinear:
        source_representation = runtime.source_points
        target_representation = runtime.target_points
        source_cost = runtime.source_offset
        target_cost = runtime.target_offset
    else:
        source_representation = runtime.source_points
        target_representation = runtime.target_points
        source_cost = np.zeros(problem.shape[0], dtype=np.float64)
        target_cost = np.zeros(problem.shape[1], dtype=np.float64)
    raw = _solve_hello_stage(
            source_F_full=source_representation,
            target_G_full=target_representation,
            source_cost_vec_full=source_cost,
            target_cost_vec_full=target_cost,
            source_mass_raw=problem.source_mass,
            target_mass_raw=problem.target_mass,
            config=runtime_config,
            depth_remaining=int(depth),
            assignment_topk=int(config.assignment_topk),
            split_count=int(config.split_count),
            dual_feasibility_tol=float(config.dual_feasibility_tolerance),
            finest_dual_feasibility_norm=config.dual_feasibility_norm,
            finest_lp_stopping_norm=config.lp_stopping_norm,
            node_seed=int(config.random_seed),
            tracer=tracer,
            pricing_index_pool=None,
            log=True,
            return_coupling=True,
            return_state=True,
            progress=bool(config.verbose),
            consume_input_features=bool(config.consume_input_features),
    )
    if not isinstance(raw, dict):
        raise RuntimeError("HELLO runtime did not return its structured result.")
    return raw, (None if tracer is None else TraceResult(tracer.export()))


def _solve(problem: Problem | _PreparedOTProblem, config: _AlgorithmConfig) -> Result:
    if config.backend == "native":
        from hello_ot._native_compat import require_native_runtime_compatibility

        require_native_runtime_compatibility()
        _require_cuda_runtime()
        solve_device = torch.device("cuda", torch.cuda.current_device())
    else:
        solve_device = resolve_torch_device(config.torch_device)
        print(
            f"[hello_ot] backend=torch device={solve_device}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[hello_ot] Portable PyTorch backend selected; timings are not representative "
            "of the native paper backend.",
            file=sys.stderr,
            flush=True,
        )
    if solve_device.type == "cuda":
        _synchronize_cuda_for_timing()
    memory_enabled = bool(config.profile_memory)
    memory_context = (
        use_solve_memory_tracker(solve_device)
        if memory_enabled and solve_device.type == "cuda"
        else nullcontext(None)
    )
    memory_summary: Optional[Dict[str, Any]] = None
    with memory_context as memory_tracker:
        if memory_tracker is not None:
            memory_tracker.start()
        started = time.perf_counter()
        stage_started = time.perf_counter()
        try:
            raw, trace = _solve_raw(problem, config)
        except Exception:
            if config.backend == "torch":
                print("[hello_ot] backend=torch failed", file=sys.stderr, flush=True)
            raise
        if solve_device.type == "cuda":
            _synchronize_cuda_for_timing()
        stage_wall_time = float(time.perf_counter() - stage_started)
        stage = _stage_result(
            raw,
            config=config,
            wall_time=stage_wall_time,
        )
        solution = _solution(raw, shape=problem.shape)
        if solve_device.type == "cuda":
            _synchronize_cuda_for_timing()
        total_wall_time = float(time.perf_counter() - started)
        if memory_tracker is not None:
            memory_summary = memory_tracker.finish()
    peak_gpu_memory_mib = (
        None
        if memory_summary is None
        else _optional_float(memory_summary.get("peak_gpu_memory_mib"))
    )
    metadata: Dict[str, Any] = {
        "cost_type": _reported_cost_type(problem),
        "assignment_mode": "nodewise_bidirectional",
        "assignment_topk": int(config.assignment_topk),
        "feature_precision": "fp64" if config.backend == "torch" else "fp32",
        "score_precision": "fp64" if config.backend == "torch" else "fp32_dimension_fp64_scalar",
        "backend": str(config.backend),
        "device": str(solve_device),
        "coarsest_solver": "pot",
        "consume_input_features": bool(config.consume_input_features),
    }
    if memory_summary is not None and bool(config.verbose):
        metadata["memory_debug"] = memory_summary
    result = Result(
        objective=float(raw.get("distance", float("nan"))),
        solution=solution,
        solve_stage=stage,
        total_wall_time=total_wall_time,
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        trace=trace,
        metadata=metadata,
    )
    if config.backend == "torch":
        print(
            f"[hello_ot] backend=torch finished in {total_wall_time:.3f}s",
            file=sys.stderr,
            flush=True,
        )
    return result


def solve_problem(problem: Problem, config: _AlgorithmConfig) -> Result:
    """
    CN: 使用统一点云接口求解 l2^2、l1、l2 或 linf cost 的平衡 OT。
    EN: Solve balanced OT with l2^2, l1, l2, or linf cost through the unified point-cloud interface.
    """
    if not isinstance(problem, Problem):
        raise TypeError("solve_problem requires Problem")
    return _solve(problem, config)


__all__ = ["solve_problem"]
