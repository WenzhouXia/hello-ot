from __future__ import annotations

import sys
import time
from contextlib import nullcontext, redirect_stdout
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from hello_ot._internal.trace import _ChromeTraceCollector
from hello_ot._internal.instrumentation.memory_accounting import use_solve_memory_tracker
from hello_ot.config import SolverRuntimeConfig, finest_stopping_norm, level_cost_precision

from .algorithm import _solve_hello_stage
from .cost import (
    _BilinearCostRuntime,
    _NormCostRuntime,
    _PreparedOTProblem,
    prepare_cost,
)
from .hierarchy.construction import resolve_hierarchy_depth
from .hierarchy.records import HierarchyLevelRecord
from .types import (
    Result,
    IterationStats,
    Problem,
    RefinementResult,
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


def _iteration_stats(records: Iterable[IterationStats]) -> tuple[IterationStats, ...]:
    """
    CN: 验证 refinement 层传入的强类型迭代记录，不接受可静默缺字段的字典。
    EN: Validate typed refinement iteration records, rejecting mappings that can silently omit fields.
    """
    output: list[IterationStats] = []
    for record in records:
        if not isinstance(record, IterationStats):
            raise TypeError(
                "HELLO refinement objective_trace must contain IterationStats values"
            )
        output.append(record)
    return tuple(output)


def _stage_result(
    raw: Mapping[str, Any],
    *,
    config: _AlgorithmConfig,
    wall_time: Optional[float] = None,
) -> SolveStageResult:
    """
    CN: 直接投影完成点的强类型记录；独立单层适配器没有 hierarchy records。
    EN: Project typed completion records directly; standalone adapters have no hierarchy records.
    """
    if "hello_diagnostics" in raw and "hierarchy_levels" not in raw:
        raise TypeError("HELLO hierarchy output requires typed hierarchy_levels")
    records = tuple(raw.get("hierarchy_levels", ()))
    if not all(isinstance(level, HierarchyLevelRecord) for level in records):
        raise TypeError("hierarchy_levels must contain HierarchyLevelRecord values")
    measured = float(raw.get("time", 0.0) or 0.0) if wall_time is None else float(wall_time)
    return SolveStageResult(levels=tuple(level.result for level in records), wall_time=measured)


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
            stopping_norm=config.stopping_norm,
            node_seed=int(config.random_seed),
            tracer=tracer,
            pricing_index_pool=None,
            log=True,
            return_coupling=True,
            return_state=True,
            progress=config.verbose == "detailed",
            consume_input_features=bool(config.consume_input_features),
            cost_perturbation=config.cost_perturbation,
            cost_perturbation_relative_scale=config.cost_perturbation_relative_scale,
    )
    if not isinstance(raw, dict):
        raise RuntimeError("HELLO runtime did not return its structured result.")
    return raw, (None if tracer is None else TraceResult(tracer.export()))


def _solve_impl(problem: Problem | _PreparedOTProblem, config: _AlgorithmConfig) -> Result:
    if config.backend == "native":
        from hello_ot._native_compat import require_native_runtime_compatibility

        require_native_runtime_compatibility()
        _require_cuda_runtime()
        solve_device = torch.device("cuda", torch.cuda.current_device())
    else:
        solve_device = resolve_torch_device(config.torch_device)
    from .progress import report_final, report_header

    report_header(
        config.verbose,
        backend=str(config.backend),
        device=str(solve_device),
        cost=_reported_cost_type(problem),
        n_source=int(problem.shape[0]),
        n_target=int(problem.shape[1]),
        dimension=int(_cost_runtime(problem).source_points.shape[1]),
        perturbation=config.cost_perturbation,
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
        output_context = redirect_stdout(sys.stderr) if config.verbose == "detailed" else nullcontext()
        with output_context:
            raw, trace = _solve_raw(problem, config)
        if solve_device.type == "cuda":
            _synchronize_cuda_for_timing()
        stage_wall_time = float(time.perf_counter() - stage_started)
        cost_stages = tuple(
            item.to_result() for item in raw.get("cost_stages", [])
        )
        stage = _stage_result(
            raw,
            config=config,
            wall_time=cost_stages[-1].solve.wall_time if cost_stages else stage_wall_time,
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
        "cost_precision": str(config.cost_precision),
        "input_feature_precision": "fp32",
        "intermediate_cost_precision": level_cost_precision(
            config.cost_precision, level_index=1
        ),
        "finest_cost_precision": level_cost_precision(
            config.cost_precision, level_index=0
        ),
        "backend": str(config.backend),
        "device": str(solve_device),
        "coarsest_solver": "pot",
        "consume_input_features": bool(config.consume_input_features),
        "stopping_norm": str(config.stopping_norm),
        "intermediate_stopping_norm": "l2",
        "finest_stopping_norm": finest_stopping_norm(config.stopping_norm),
        "cost_perturbation": raw.get("cost_perturbation", {"policy": "off", "activated": False}),
        "total_lp_solve_time": float(raw.get("lp_solve_time_total", 0.0)),
        "total_refinement_iterations": sum(
            len(level.solve.iterations) for cost_stage in cost_stages
            for level in cost_stage.solve.levels if isinstance(level.solve, RefinementResult)
        ),
    }
    if memory_summary is not None and config.verbose == "detailed":
        metadata["memory_debug"] = memory_summary
    result = Result(
        objective=float(raw.get("distance", float("nan"))),
        solution=solution,
        solve_stage=stage,
        total_wall_time=total_wall_time,
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        trace=trace,
        metadata=metadata,
        cost_stages=cost_stages,
    )
    report_final(config.verbose, result)
    return result


def _solve(problem: Problem | _PreparedOTProblem, config: _AlgorithmConfig) -> Result:
    """
    CN: 在公开输出边界报告失败，同时保留原始异常与 traceback。
    EN: Report failures at the public output boundary while preserving the original exception and traceback.
    """
    started = time.perf_counter()
    try:
        return _solve_impl(problem, config)
    except Exception as error:
        from .progress import report_failure

        report_failure(config.verbose, error, time.perf_counter() - started)
        raise


def solve_problem(problem: Problem, config: _AlgorithmConfig) -> Result:
    """
    CN: 使用统一点云接口求解 l2^2、l1、l2 或 linf cost 的平衡 OT。
    EN: Solve balanced OT with l2^2, l1, l2, or linf cost through the unified point-cloud interface.
    """
    if not isinstance(problem, Problem):
        raise TypeError("solve_problem requires Problem")
    if config.gpu_resident:
        from .gpu_resident.solver import solve as solve_gpu

        return solve_gpu(problem, config)
    if any(torch.is_tensor(value) for value in (problem.source_points, problem.target_points, problem.source_mass, problem.target_mass)):
        def host(value):
            if torch.is_tensor(value):
                value = value.detach().cpu()
                return (value.float() if value.dtype == torch.bfloat16 else value).numpy()
            return value
        problem = Problem(host(problem.source_points), host(problem.target_points), problem.cost_type,
                          host(problem.source_mass), host(problem.target_mass))
    return _solve(problem, config)


__all__ = ["solve_problem"]
