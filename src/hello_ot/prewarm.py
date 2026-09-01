from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import torch

from hello_ot._internal.prewarm import (
    prewarm_hierarchy_cold_paths,
    prewarm_pot_backend,
)
from hello_ot.hierarchy.plan import _prewarm_inplace_feature_reorder

from .config import SolverOptions, SolverRuntimeConfig, _AlgorithmConfig
from .kernels.norm_cost_scan import _fused_metric_kmin, run_metric_dual_feasibility_scan
from .types import Problem, PrewarmStats


def _prewarm_torch_backend(problem: Problem, config: _AlgorithmConfig) -> tuple[str, ...]:
    """
    CN: 在所选 Torch 设备上触发共享 blockwise scan、restricted LP 与 POT 导入路径。
    EN: Exercise the shared blockwise scan, restricted LP, and POT import paths on the selected Torch device.
    """
    from hello_ot._internal.lp.torch_restricted_ot import TorchRestrictedOTPDLP
    from hello_ot.kernels.torch_scan import bidirectional_violation_scan, resolve_torch_device

    device = resolve_torch_device(config.torch_device)
    source = np.ascontiguousarray(problem.source_points[:8], dtype=np.float64)
    target = np.ascontiguousarray(problem.target_points[:8], dtype=np.float64)
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("prewarm requires a non-empty problem")
    if problem.cost_type == "l2^2":
        source_offset = np.sum(source * source, axis=1)
        target_offset = np.sum(target * target, axis=1)
        score_family, scan_cost, dot_scale = "inner_product", "lowrank", 2.0
    else:
        source_offset = target_offset = None
        score_family, scan_cost, dot_scale = "norm_cost", str(problem.cost_type), 1.0
    bidirectional_violation_scan(
        source_points=source,
        target_points=target,
        source_offset=source_offset,
        target_offset=target_offset,
        source_dual=np.zeros(source.shape[0], dtype=np.float64),
        target_dual=np.zeros(target.shape[0], dtype=np.float64),
        score_family=score_family,
        cost_type=scan_cost,
        dot_scale=dot_scale,
        topk=1,
        theta=0.0,
        device=device,
        max_tile_bytes=4096,
    )
    lp_solver = TorchRestrictedOTPDLP(
        device=device,
        evaluation_frequency=20,
        iteration_limit=2000,
        time_limit_seconds=10.0,
    )
    lp_result = lp_solver.solve_restricted(
        rows=np.asarray([0, 0, 1, 1], dtype=np.int64),
        cols=np.asarray([0, 1, 0, 1], dtype=np.int64),
        costs=np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        source_mass=np.asarray([0.5, 0.5], dtype=np.float64),
        target_mass=np.asarray([0.5, 0.5], dtype=np.float64),
        tolerance=1e-4,
    )
    if not lp_result.success:
        raise RuntimeError(f"Torch restricted-OT prewarm failed: {lp_result.termination_reason}")
    pot = prewarm_pot_backend()
    if not bool(pot.get("success", False)):
        raise RuntimeError(f"POT prewarm failed: {pot.get('error')}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (f"torch_scan:{device}", f"torch_restricted_lp:{device}", "pot_coarsest")


def _prewarm_norm_cost(problem: Problem, config: _AlgorithmConfig) -> tuple[str, ...]:
    """
    CN: 触发范数 cost 的 assignment 与 dual-certificate fused kernels。
    EN: Exercise the assignment and dual-certificate fused kernels for a norm cost.
    """
    size = max(16, int(config.assignment_topk), int(round(config.pricing_topk)))
    source = np.ascontiguousarray(problem.source_points[:size], dtype=np.float32)
    target = np.ascontiguousarray(problem.target_points[:size], dtype=np.float32)
    if source.shape[0] < size:
        source = np.repeat(problem.source_points[:1], size, axis=0).astype(np.float32, copy=False)
    if target.shape[0] < size:
        target = np.repeat(problem.target_points[:1], size, axis=0).astype(np.float32, copy=False)
    zero = np.zeros(size, dtype=np.float64)
    for query, database, query_is_source in (
        (source, target, True),
        (target, source, False),
    ):
        _fused_metric_kmin(
            query_points=query,
            database_points=database,
            known_dual=zero,
            cost_type=problem.cost_type,  # type: ignore[arg-type]
            k=int(config.assignment_topk),
            query_is_source=query_is_source,
        )
    runtime = SolverRuntimeConfig.from_options(config, cost_type=str(problem.cost_type))
    run_metric_dual_feasibility_scan(
        config=runtime,
        lvl_s=SimpleNamespace(points=source),
        lvl_t=SimpleNamespace(points=target),
        dual_uv=np.zeros(2 * size, dtype=np.float64),
        inner_iter=-1,
        trace_collector=None,
        trace_prefix="hello.prewarm.metric",
    )
    torch.cuda.synchronize()
    return ("norm_assignment", "norm_dual_certificate")


def prewarm(problem: Problem, options: SolverOptions | None = None) -> PrewarmStats:
    """
    CN: 在正式计时前预热当前 point-cloud cost/backend 的 kernel 路径，不运行完整 OT。
    EN: Prewarm kernel paths for the selected point-cloud cost/backend before timed solving without running full OT.
    """
    if not isinstance(problem, Problem):
        raise TypeError("prewarm requires Problem")
    resolved_options = SolverOptions() if options is None else options
    if not isinstance(resolved_options, SolverOptions):
        raise TypeError("options must be SolverOptions or None")
    config = resolved_options._to_internal_config(max_iterations=100, random_seed=42)
    started = time.perf_counter()
    if bool(config.consume_input_features):
        _prewarm_inplace_feature_reorder()
    if config.backend == "torch":
        operations = _prewarm_torch_backend(problem, config)
        if bool(config.consume_input_features):
            operations = (*operations, "inplace_feature_reorder")
        return PrewarmStats(
            wall_time=float(time.perf_counter() - started),
            operations=operations,
        )
    if not torch.cuda.is_available():
        raise RuntimeError("HELLO prewarm requires a CUDA-capable PyTorch runtime")
    cold = prewarm_hierarchy_cold_paths(
        source_f=problem.source_points,
        solver_engine="cupdlpx",
        gpu_id=int(torch.cuda.current_device()),
        variable_bound_mode=str(config.variable_bound_mode),
        matrix_value_mode=str(config.matrix_value_mode),
        vector_sum_mode=str(config.vector_sum_mode),
    )
    if not bool(cold.get("success", False)):
        failures = {
            name: stage.get("error")
            for name, stage in dict(cold.get("stages") or {}).items()
            if not bool(stage.get("success", False))
        }
        raise RuntimeError(f"HELLO prewarm failed: {failures}")
    operations = [str(name) for name in dict(cold.get("stages") or {})]
    if bool(config.consume_input_features):
        operations.append("inplace_feature_reorder")
    if problem.cost_type in {"l1", "l2", "linf"}:
        operations.extend(_prewarm_norm_cost(problem, config))
    return PrewarmStats(
        wall_time=float(time.perf_counter() - started),
        operations=tuple(operations),
    )
