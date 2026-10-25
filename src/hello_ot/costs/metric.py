from __future__ import annotations

# CN: 组织范数 cost 的初始化、dual assignment 与完整最优性检查。
# EN: Coordinate initialization, dual assignment, and full optimality checks for norm costs.

import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch

from hello_ot._internal.core.solver import HierarchicalOTSolver
from hello_ot.costs.pairs import (
    _cost_pairs_euclidean_batched,
    _cost_pairs_l1_batched,
    _cost_pairs_linf_batched,
    _prepare_level_cost_cache_euclidean,
    _prepare_level_cost_cache_l1,
    _prepare_level_cost_cache_linf,
)
from hello_ot.hierarchy.coarsest import _solve_lowrank_leaf_with_pot_from_precomputed_cost
from hello_ot.hierarchy.utilities import _normalize_subproblem_masses
from hello_ot.initialization.state import (
    DualAssignmentState,
    DualCompletionCandidate,
    _normalize_dual_assignment_state,
    _state_from_gpu,
    _state_to_cpu_public,
    _unique_merge_rows_cols_vals,
)
from hello_ot._internal.trace import _ChromeTraceCollector, _trace_device_synchronize
from hello_ot.kernels.scan_contract import (
    _TOPK_SLOT_BYTES,
    DualFeasibilityCertificate,
    DualViolationScanResult,
    custom_topk_bucket,
    plan_resident_side,
    plan_stream_chunk_rows,
    reusable_cuda_memory_bytes,
)
from hello_ot.initialization.initial_support import (
    _WARM_START_INITIAL_CREATION_ITER,
    _augment_active_support_with_northwest_corner,
    _mark_existing_northwest_positions,
)
from hello_ot.restricted_ot.runtime import (
    build_metric_restricted_solver,
)
from hello_ot.restricted_ot.active_support import ActiveSupport
from hello_ot.config import SolverRuntimeConfig
from hello_ot.state import (
    GPUWarmStartState as OTWarmStartGPUState,
    WarmStartState as OTWarmStartState,
)


MetricCostType = Literal["l1", "linf", "l2"]


from hello_ot.kernels.norm_cost_scan import (_fused_metric_kmin, detect_metric_violations_and_check_feasibility, check_metric_dual_feasibility)
from hello_ot.costs.metric_perturbation import _apply_metric_pair_perturbation

def _metric_cost_matrix(
    source: np.ndarray,
    target: np.ndarray,
    cost_type: MetricCostType,
    *,
    compute_precision: str = "fp32",
) -> np.ndarray:
    dtype = np.float64 if compute_precision == "fp64" else np.float32
    src = np.asarray(source, dtype=dtype, order="C")
    tgt = np.asarray(target, dtype=dtype, order="C")
    diff = np.abs(src[:, None, :] - tgt[None, :, :])
    if str(cost_type) == "l1":
        return np.sum(diff, axis=2, dtype=dtype).astype(np.float64)
    if str(cost_type) == "linf":
        return np.max(diff, axis=2).astype(np.float64)
    if str(cost_type) == "l2":
        return np.sqrt(np.sum(diff * diff, axis=2, dtype=dtype)).astype(np.float64)
    raise ValueError("metric hello supports cost_type in {'l1', 'linf', 'l2'}.")


def pair_costs_metric(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    rows: Any,
    cols: Any,
    cost_type: MetricCostType,
    batch_size: int = 65536,
    perturbation: Optional[Dict[str, Any]] = None,
    compute_precision: str = "fp32",
) -> Any:
    if compute_precision == "fp64":
        rows_is_torch = torch.is_tensor(rows)
        if rows_is_torch:
            device = rows.device
            rr = rows.to(device=device, dtype=torch.long)
            cc = cols.to(device=device, dtype=torch.long) if torch.is_tensor(cols) else torch.as_tensor(cols, device=device, dtype=torch.long)
            source = torch.as_tensor(source_points, device=device, dtype=torch.float64)
            target = torch.as_tensor(target_points, device=device, dtype=torch.float64)
            difference = torch.abs(source.index_select(0, rr) - target.index_select(0, cc))
            if cost_type == "l1":
                costs = torch.sum(difference, dim=1)
            elif cost_type == "linf":
                costs = torch.amax(difference, dim=1)
            else:
                costs = torch.linalg.vector_norm(difference, dim=1)
        else:
            rr = np.asarray(rows, dtype=np.int64)
            cc = np.asarray(cols, dtype=np.int64)
            difference = np.abs(
                np.asarray(source_points, dtype=np.float64)[rr]
                - np.asarray(target_points, dtype=np.float64)[cc]
            )
            if cost_type == "l1":
                costs = np.sum(difference, axis=1, dtype=np.float64)
            elif cost_type == "linf":
                costs = np.max(difference, axis=1)
            else:
                costs = np.linalg.norm(difference, axis=1)
        return _apply_metric_pair_perturbation(costs, rows, cols, perturbation)
    cost_name = str(cost_type)
    if cost_name == "l1":
        level_cache = _prepare_level_cost_cache_l1(source_points, target_points)
    elif cost_name == "linf":
        level_cache = _prepare_level_cost_cache_linf(source_points, target_points)
    elif cost_name == "l2":
        level_cache = _prepare_level_cost_cache_euclidean(source_points, target_points)
    else:
        raise ValueError("metric hello supports cost_type in {'l1', 'linf', 'l2'}.")
    if str(cost_type) == "l1":
        costs = _cost_pairs_l1_batched(level_cache, rows, cols, batch_size=int(batch_size))
    elif str(cost_type) == "linf":
        costs = _cost_pairs_linf_batched(level_cache, rows, cols, batch_size=int(batch_size))
    else:
        costs = _cost_pairs_euclidean_batched(level_cache, rows, cols, batch_size=int(batch_size))
    costs = costs.to(dtype=torch.float64) if torch.is_tensor(costs) else np.asarray(costs, dtype=np.float64)
    return _apply_metric_pair_perturbation(costs, rows, cols, perturbation)


def solve_leaf_metric(
    *,
    source_points_full: np.ndarray,
    target_points_full: np.ndarray,
    source_mass_raw: np.ndarray,
    target_mass_raw: np.ndarray,
    source_start: int,
    source_stop: int,
    target_start: int,
    target_stop: int,
    cost_type: MetricCostType,
    node_trace_args: Dict[str, Any],
    tracer: Optional[_ChromeTraceCollector],
    perturbation: Optional[Dict[str, Any]] = None,
    compute_precision: str = "fp32",
) -> Tuple[DualAssignmentState, Dict[str, Any]]:
    source = np.asarray(source_points_full[int(source_start) : int(source_stop)], dtype=np.float32, order="C")
    target = np.asarray(target_points_full[int(target_start) : int(target_stop)], dtype=np.float32, order="C")
    t_cost = time.perf_counter()
    cost = _metric_cost_matrix(
        source, target, cost_type, compute_precision=compute_precision
    )
    if perturbation is not None:
        rows, cols = np.indices(cost.shape)
        cost = _apply_metric_pair_perturbation(cost.reshape(-1), rows.reshape(-1), cols.reshape(-1), perturbation).reshape(cost.shape)
    cost_build_time = float(time.perf_counter() - t_cost)
    source_mass, target_mass, _ = _normalize_subproblem_masses(
        np.asarray(source_mass_raw[int(source_start) : int(source_stop)], dtype=np.float64, order="C"),
        np.asarray(target_mass_raw[int(target_start) : int(target_stop)], dtype=np.float64, order="C"),
    )
    result = _solve_lowrank_leaf_with_pot_from_precomputed_cost(
        cost=cost,
        source_f=source,
        target_g=target,
        source_mass=source_mass,
        target_mass=target_mass,
        tracer=tracer,
        trace_args=node_trace_args,
        precomputed_cost_build_time=cost_build_time,
        cost_build_backend=f"metric_{cost_type}_dense",
        cost_build_batch_size=1,
        cost_build_device="cpu",
    )
    return result["warm_start_state"], {
        "node_build_time_total": float(time.perf_counter() - t_cost),
        "leaf_stage_profile": result["leaf_stage_profile"],
        "solve_summary": {
            "distance": float(result["distance"]),
            "time": float(result["time"]),
            "lp_solve_time_total": float(result["lp_solve_time_total"]),
        },
    }


def _build_augmented_state(
    state: DualAssignmentState,
    *,
    rows_add_t: torch.Tensor,
    cols_add_t: torch.Tensor,
    n_source: int,
    n_target: int,
    cost_type: MetricCostType,
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
) -> DualAssignmentState:
    normalized = _normalize_dual_assignment_state(state, pipeline="gpu")
    state_cpu = _state_from_gpu(normalized) if isinstance(normalized, OTWarmStartGPUState) else _state_to_cpu_public(normalized)
    rows_old = np.asarray(state_cpu.rows, dtype=np.int32)
    cols_old = np.asarray(state_cpu.cols, dtype=np.int32)
    vals_old = np.asarray(state_cpu.x_prev, dtype=np.float64)
    rows_new = rows_add_t.detach().cpu().numpy().astype(np.int32, copy=False)
    cols_new = cols_add_t.detach().cpu().numpy().astype(np.int32, copy=False)
    vals_new = np.zeros(int(rows_new.size), dtype=np.float64)
    rows, cols, vals, _backend = _unique_merge_rows_cols_vals(
        np.concatenate([rows_old, rows_new]),
        np.concatenate([cols_old, cols_new]),
        np.concatenate([vals_old, vals_new]),
        n_target=int(n_target),
        tracer=tracer,
        trace_args=trace_args,
        trace_prefix="metric_augment.unique_merge",
    )
    out = OTWarmStartState(
        rows=np.asarray(rows, dtype=np.int32),
        cols=np.asarray(cols, dtype=np.int32),
        x_prev=np.asarray(vals, dtype=np.float64),
        dual_uv=np.asarray(state_cpu.dual_uv, dtype=np.float64).copy(),
        n_source=int(n_source),
        n_target=int(n_target),
    )
    return _normalize_dual_assignment_state(out, pipeline="gpu")


def augment_topk_metric(
    *,
    state: DualAssignmentState,
    source_points: np.ndarray,
    target_points: np.ndarray,
    known_side: Literal["source", "target"],
    cost_type: MetricCostType,
    assignment_topk: int,
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
    perturbation: Optional[Dict[str, Any]] = None,
    compute_precision: str = "fp32",
) -> Tuple[DualAssignmentState, Dict[str, Any], Dict[str, Any], DualCompletionCandidate]:
    if str(known_side) not in {"source", "target"}:
        raise ValueError("known_side must be either 'source' or 'target'.")
    t_total = time.perf_counter()
    query_is_source = str(known_side) == "target"
    query_points = np.asarray(source_points if query_is_source else target_points, dtype=np.float32, order="C")
    database_points = np.asarray(target_points if query_is_source else source_points, dtype=np.float32, order="C")
    n_source = int(source_points.shape[0])
    n_target = int(target_points.shape[0])

    state_cpu = _state_from_gpu(state) if isinstance(state, OTWarmStartGPUState) else _state_to_cpu_public(state)
    dual = np.asarray(state_cpu.dual_uv, dtype=np.float64)
    block_parts = [np.arange(int(database_points.shape[0]), dtype=np.int64)]
    block_duals = [dual[n_source:] if query_is_source else dual[:n_source]]
    k = int(assignment_topk)

    rows_parts: list[torch.Tensor] = []
    cols_parts: list[torch.Tensor] = []
    best_score_t: Optional[torch.Tensor] = None
    best_index_t: Optional[torch.Tensor] = None
    topk_time = 0.0
    for part_idx, known_dual in zip(block_parts, block_duals):
        t_topk = time.perf_counter()
        values_t, local_idx_t = _fused_metric_kmin(
            query_points=query_points,
            database_points=database_points[part_idx],
            known_dual=np.asarray(known_dual, dtype=np.float64),
            cost_type=cost_type,
            k=int(k),
            query_is_source=query_is_source,
            query_global_index=None if perturbation is None else perturbation["source_global_index" if query_is_source else "target_global_index"],
            database_global_index=None if perturbation is None else perturbation["target_global_index" if query_is_source else "source_global_index"][part_idx],
            perturbation=perturbation,
            compute_precision=compute_precision,
        )
        topk_time += float(time.perf_counter() - t_topk)
        part_t = torch.as_tensor(part_idx, dtype=torch.int64, device=local_idx_t.device)
        global_idx_t = part_t.index_select(0, local_idx_t.reshape(-1)).reshape(local_idx_t.shape)
        query_idx_t = torch.arange(int(query_points.shape[0]), dtype=torch.int64, device=local_idx_t.device)[:, None].expand_as(global_idx_t)
        if query_is_source:
            rows_parts.append(query_idx_t.reshape(-1).to(dtype=torch.int64))
            cols_parts.append(global_idx_t.reshape(-1).to(dtype=torch.int64))
        else:
            rows_parts.append(global_idx_t.reshape(-1).to(dtype=torch.int64))
            cols_parts.append(query_idx_t.reshape(-1).to(dtype=torch.int64))
        score_t = -values_t
        if best_score_t is None:
            best_score_t = score_t[:, 0].contiguous()
            best_index_t = global_idx_t[:, 0].contiguous()
        else:
            better_t = score_t[:, 0] > best_score_t
            best_score_t = torch.where(better_t, score_t[:, 0], best_score_t).contiguous()
            best_index_t = torch.where(better_t, global_idx_t[:, 0], best_index_t).contiguous()

    rows_add_t = torch.cat(rows_parts, dim=0) if rows_parts else torch.empty(0, dtype=torch.int64, device="cuda")
    cols_add_t = torch.cat(cols_parts, dim=0) if cols_parts else torch.empty(0, dtype=torch.int64, device=rows_add_t.device)
    parent_state = _build_augmented_state(
        state,
        rows_add_t=rows_add_t,
        cols_add_t=cols_add_t,
        n_source=n_source,
        n_target=n_target,
        cost_type=cost_type,
        tracer=tracer,
        trace_args=trace_args,
    )
    if best_score_t is None or best_index_t is None:
        raise RuntimeError("metric augment did not produce completion seed.")
    completion_candidate = DualCompletionCandidate(
        known_side=known_side,
        best_index=best_index_t.to(dtype=torch.int64).contiguous(),
        best_score=best_score_t.to(dtype=torch.float64).contiguous(),
        n_source=n_source,
        n_target=n_target,
        backend=f"norm_cost_scan_{cost_type}",
        score_recompute_time=0.0,
    )
    raw_added = int(rows_add_t.numel())
    support_after = int(rows_add_t.numel())
    try:
        support_after = int(parent_state.rows.numel()) if isinstance(parent_state, OTWarmStartGPUState) else int(np.asarray(parent_state.rows).size)
    except Exception:
        pass
    topk_stats = {
        "cross_transfer_added_raw": int(raw_added),
        "cross_transfer_added_unique": int(support_after),
    }
    profile = {
        "backend": f"fused_{cost_type}",
        "topk_backend": "fused",
        "topk_total_time": float(topk_time),
        "merge_unique_time": float(max(0.0, time.perf_counter() - t_total - topk_time)),
        "total_time": float(time.perf_counter() - t_total),
    }
    return parent_state, topk_stats, profile, completion_candidate


def complete_dual_metric(
    *,
    state: DualAssignmentState,
    source_points: np.ndarray,
    target_points: np.ndarray,
    known_side: Literal["source", "target"],
    cost_type: MetricCostType,
    completion_candidate: Any,
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
    trace_prefix: str,
    perturbation: Optional[Dict[str, Any]] = None,
    compute_precision: str = "fp32",
) -> Tuple[DualAssignmentState, Dict[str, Any]]:
    del tracer, trace_args, trace_prefix
    t_total = time.perf_counter()
    normalized = _normalize_dual_assignment_state(state, pipeline="gpu")
    if not isinstance(normalized, OTWarmStartGPUState):
        raise RuntimeError("metric hello dual completion requires a GPU warm-start state.")
    if normalized.dual_uv is None:
        raise ValueError("metric hello dual completion requires state.dual_uv.")
    dual = normalized.dual_uv.to(dtype=torch.float64).clone()
    n_source = int(normalized.n_source)
    if completion_candidate is not None and isinstance(completion_candidate, DualCompletionCandidate):
        best_score = completion_candidate.best_score
        best_score_t = torch.as_tensor(best_score, dtype=torch.float64, device=dual.device)
        if str(known_side) == "target":
            dual[:n_source] = -best_score_t
        else:
            dual[n_source:] = -best_score_t
    else:
        if str(known_side) == "target":
            known = dual[n_source:]
            values_t, _idx_t = _fused_metric_kmin(
                query_points=source_points,
                query_is_source=True,
                perturbation=perturbation,
                query_global_index=None if perturbation is None else perturbation["source_global_index"],
                database_global_index=None if perturbation is None else perturbation["target_global_index"],
                database_points=target_points,
                known_dual=known,
                cost_type=cost_type,
                k=1,
                compute_precision=compute_precision,
            )
            dual[:n_source] = values_t[:, 0]
        else:
            known = dual[:n_source]
            values_t, _idx_t = _fused_metric_kmin(
                query_points=target_points,
                query_is_source=False,
                perturbation=perturbation,
                query_global_index=None if perturbation is None else perturbation["target_global_index"],
                database_global_index=None if perturbation is None else perturbation["source_global_index"],
                database_points=source_points,
                known_dual=known,
                cost_type=cost_type,
                k=1,
                compute_precision=compute_precision,
            )
            dual[n_source:] = values_t[:, 0]
    out = OTWarmStartGPUState(
        rows=normalized.rows,
        cols=normalized.cols,
        x_prev=normalized.x_prev,
        dual_uv=dual.contiguous(),
        n_source=int(normalized.n_source),
        n_target=int(normalized.n_target),
        device=str(normalized.rows.device),
        keys=normalized.keys,
        northwest_positions=normalized.northwest_positions,
    )
    return out, {
        "backend": f"fused_{cost_type}",
        "total_time": float(time.perf_counter() - t_total),
    }


def _validate_metric_warm_start(
    warm_start: OTWarmStartState | OTWarmStartGPUState,
    *,
    n_s: int,
    n_t: int,
    cost_type: MetricCostType,
) -> Tuple[Any, Any, Any, Optional[np.ndarray]]:
    if not isinstance(warm_start, (OTWarmStartState, OTWarmStartGPUState)):
        raise TypeError("warm_start must be an OTWarmStartState or OTWarmStartGPUState instance.")
    if int(warm_start.n_source) != int(n_s) or int(warm_start.n_target) != int(n_t):
        raise ValueError(
            "warm_start dimensions do not match the current problem: "
            f"got ({warm_start.n_source}, {warm_start.n_target}), expected ({n_s}, {n_t})."
        )
    if isinstance(warm_start, OTWarmStartGPUState):
        rows = warm_start.rows.detach()
        cols = warm_start.cols.detach()
        x_prev = warm_start.x_prev.detach()
        dual_uv = None
        if warm_start.dual_uv is not None:
            dual_uv = warm_start.dual_uv.detach().cpu().numpy().astype(np.float64, copy=False)
        return rows, cols, x_prev, dual_uv
    rows = np.asarray(warm_start.rows, dtype=np.int32)
    cols = np.asarray(warm_start.cols, dtype=np.int32)
    x_prev = np.asarray(warm_start.x_prev, dtype=np.float64)
    dual_uv = None if warm_start.dual_uv is None else np.asarray(warm_start.dual_uv, dtype=np.float64)
    if rows.ndim != 1 or cols.ndim != 1 or x_prev.ndim != 1 or not (rows.size == cols.size == x_prev.size):
        raise ValueError("warm_start rows, cols, and x_prev must be 1D arrays with the same length.")
    if np.any(rows < 0) or np.any(rows >= int(n_s)) or np.any(cols < 0) or np.any(cols >= int(n_t)):
        raise ValueError("warm_start contains out-of-range indices.")
    if dual_uv is not None and (dual_uv.ndim != 1 or dual_uv.shape[0] != int(n_s + n_t)):
        raise ValueError("warm_start.dual_uv must have length n_source + n_target.")
    return rows, cols, x_prev, dual_uv


def _seed_metric_active_support(
    solver: HierarchicalOTSolver,
    warm_start: OTWarmStartState | OTWarmStartGPUState,
    *,
    cost_type: MetricCostType,
    trace_collector: Optional[_ChromeTraceCollector],
    trace_prefix: str,
) -> Dict[str, Any]:
    lvl_s = solver.hierarchy_s.finest_level
    lvl_t = solver.hierarchy_t.finest_level
    components: Dict[str, float] = {}
    use_gpu_state = isinstance(warm_start, OTWarmStartGPUState)
    if str(cost_type) == "l1":
        level_cache = _prepare_level_cost_cache_l1(lvl_s.points, lvl_t.points)
    elif str(cost_type) == "linf":
        level_cache = _prepare_level_cost_cache_linf(lvl_s.points, lvl_t.points)
    elif str(cost_type) == "l2":
        level_cache = _prepare_level_cost_cache_euclidean(lvl_s.points, lvl_t.points)
    else:
        raise ValueError("metric hello supports cost_type in {'l1', 'linf', 'l2'}.")
    t0 = time.perf_counter()
    active_support = ActiveSupport(
        n_source=len(lvl_s.points),
        n_target=len(lvl_t.points),
        level_cache=level_cache,
        track_creation=bool(solver.cleaning_strategy.needs_creation_iteration()),
        device=str(warm_start.device) if use_gpu_state else "cuda",
    )
    components["init_active_support"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows, cols, x_prev, _dual_uv = _validate_metric_warm_start(
        warm_start,
        n_s=len(lvl_s.points),
        n_t=len(lvl_t.points),
        cost_type=cost_type,
    )
    components["validate_warm_start"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    active_support.add_pairs_placeholder(rows, cols, iter_idx=_WARM_START_INITIAL_CREATION_ITER)
    active_support.set_x_prev(x_prev)
    components["add_pairs"] = time.perf_counter() - t0
    solver.active_support = active_support
    initial_support_size = int(active_support.size)
    t0 = time.perf_counter()
    if _mark_existing_northwest_positions(solver, getattr(warm_start, "northwest_positions", None)):
        northwest_added = 0
    else:
        northwest_added = _augment_active_support_with_northwest_corner(
            solver,
            lvl_s.masses,
            lvl_t.masses,
            placeholder_costs=True,
        )
    components["bfs_skeleton"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _trace_device_synchronize(trace_collector)
    c_vec = solver._compute_pair_costs_arrays(active_support.rows, active_support.cols)
    active_support.replace_costs(c_vec)
    _trace_device_synchronize(trace_collector)
    components["compute_c_vec"] = time.perf_counter() - t0
    return {
        "components": components,
        "primal_nnz": int(rows.numel()) if torch.is_tensor(rows) else int(np.asarray(rows).shape[0]),
        "initial_support_size": int(initial_support_size),
        "northwest_added": int(northwest_added),
        "warm_dual_pricing_added": 0,
        "post_pricing_support_size": int(solver.active_support.size),
    }


def _init_metric_warm_start_refinement(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    cost_type: MetricCostType,
    warm_start: OTWarmStartState | OTWarmStartGPUState,
    config: SolverRuntimeConfig,
    dual_feasibility_tol: float,
    tracer: Optional[_ChromeTraceCollector],
    trace_prefix: str,
    perturbation: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    CN: 构建范数 cost 的单层 GPU runtime，并在首个 restricted LP 前完成 support 初始化。
    EN: Build a one-level GPU runtime for a norm cost and initialize its support before the first restricted LP.
    """
    from hello_ot.restricted_ot.runtime import InitializedLevel
    from hello_ot.types import InitializationResult

    sub_source_mass, sub_target_mass, _ = _normalize_subproblem_masses(source_mass, target_mass)
    solve_cfg = replace(config, metric_cost_perturbation=perturbation)
    solve_cfg.cost_type = str(cost_type)
    solve_cfg.pricing_strategy = "nodewise_full"
    solve_cfg.convergence_criterion = "dual_feasibility"
    solve_cfg.require_dual_feasibility_convergence = False
    solve_cfg.dual_feasibility_tol = float(dual_feasibility_tol)
    solve_cfg.require_added_convergence = False
    solve_cfg.validate()
    solver, cfg = build_metric_restricted_solver(
        np.asarray(source_points, dtype=np.float32, order="C"),
        np.asarray(target_points, dtype=np.float32, order="C"),
        np.asarray(sub_source_mass, dtype=np.float64, order="C"),
        np.asarray(sub_target_mass, dtype=np.float64, order="C"),
        solve_cfg,
        cost_type=cost_type,
    )
    initial_support_info = _seed_metric_active_support(
        solver,
        warm_start,
        cost_type=cost_type,
        trace_collector=tracer,
        trace_prefix=trace_prefix,
    )
    dual_uv = _validate_metric_warm_start(
        warm_start,
        n_s=int(source_points.shape[0]),
        n_t=int(target_points.shape[0]),
        cost_type=cost_type,
    )[3]
    return InitializedLevel(
        solver=solver,
        cfg=cfg,
        initialization=InitializationResult(
            active_support=solver.active_support,
            dual_warm_start=dual_uv,
            statistics=initial_support_info,
        ),
    )


def run_metric_dual_feasibility_scan(
    *,
    config: SolverRuntimeConfig,
    lvl_s: Any,
    lvl_t: Any,
    dual_uv: np.ndarray,
    inner_iter: int,
    trace_collector: Optional[Any],
    trace_prefix: str,
) -> Any:
    from hello_ot.refinement.dual_feasibility import LowrankDualFeasibilityScan

    cost_type = str(getattr(config, "cost_type", "l1")).lower()
    if cost_type not in {"l1", "linf", "l2"}:
        raise ValueError("metric dual feasibility scan supports cost_type in {'l1', 'linf', 'l2'}.")
    perturbation = config.metric_cost_perturbation
    perturb_args = {} if perturbation is None else {"perturbation": perturbation, "source_global_index": perturbation["source_global_index"], "target_global_index": perturbation["target_global_index"]}
    topk = max(1, int(round(float(getattr(config, "pricing_topk", 1.0)))))
    scan_start = time.perf_counter()
    diagnostics: Dict[str, Any] = {
        "inner_iter": int(inner_iter + 1),
        "n_source": int(len(lvl_s.points)),
        "n_target": int(len(lvl_t.points)),
        "use_fused_metric_feasibility_pricing": True,
        "metric_cost_type": cost_type,
    }
    if str(getattr(config, "backend", "native")) in {"torch", "cpp"}:
        if config.backend == "cpp":
            from hello_ot.kernels.cpp_scan import cpp_violation_scan as bidirectional_violation_scan
        else:
            from hello_ot.kernels.torch_scan import bidirectional_violation_scan

        source_np = np.asarray(lvl_s.points, dtype=np.float32, order="C")
        target_np = np.asarray(lvl_t.points, dtype=np.float32, order="C")
        scan = bidirectional_violation_scan(
            source_points=source_np,
            target_points=target_np,
            source_offset=None,
            target_offset=None,
            source_dual=dual_uv[: int(source_np.shape[0])],
            target_dual=dual_uv[int(source_np.shape[0]) :],
            score_family="norm_cost",
            cost_type=cost_type,
            dot_scale=1.0,
            topk=int(topk),
            theta=0.0,
            device=str(getattr(config, "torch_device", "auto")),
            compute_precision=str(getattr(config, "cost_precision", "fp32")),
            perturbation=perturbation,
        )
        diagnostics.update(scan.diagnostics)
        diagnostics.setdefault("edge_selection_mode", "nodewise")
        return LowrankDualFeasibilityScan(
            dual_feasibility=float(scan.dual_feasibility),
            dual_feasibility_source="early_torch_blockwise_metric_scan",
            scan_time=float(time.perf_counter() - scan_start),
            diagnostics=diagnostics,
            rows=scan.rows,
            cols=scan.cols,
            peak_mem_mib=None,
        )
    with (
        trace_collector.span(f"{trace_prefix}.early_fused_metric_scan", "solve_ot", args=diagnostics)
        if trace_collector is not None
        else nullcontext()
    ):
        source_np = np.asarray(lvl_s.points, dtype=np.float32, order="C")
        target_np = np.asarray(lvl_t.points, dtype=np.float32, order="C")
        device = torch.device("cuda")
        dual_t = torch.as_tensor(np.asarray(dual_uv, dtype=np.float64), dtype=torch.float64, device=device).view(-1)
        u_t = dual_t[: int(source_np.shape[0])]
        v_t = dual_t[int(source_np.shape[0]) :]
        topk_int = int(topk)
        custom_topk_bucket(topk_int, score_family="norm_cost")
        scan_result = detect_metric_violations_and_check_feasibility(
            **perturb_args,
            source_points=source_np,
            target_points=target_np,
            source_dual=u_t,
            target_dual=v_t,
            cost_type=cost_type,  # type: ignore[arg-type]
            topk=topk_int,
            compute_precision=str(getattr(config, "cost_precision", "fp32")),
        )
        certificate = scan_result.certificate
        num_t = certificate.l2_numerator_sq
        den_t = certificate.l2_denominator_sq
        diagnostics.update(scan_result.diagnostics)
        diagnostics.update(
            {
                "dual_feasibility_positive_count": int(certificate.positive_count.detach().cpu().item()),
                "dual_feasibility_max_violation": float(certificate.max_positive_violation.detach().cpu().item()),
                "dual_feasibility_cost_linf": float(certificate.cost_linf.detach().cpu().item()),
            }
        )
        dual_feasibility = float(
            (
                torch.sqrt(torch.clamp(num_t, min=0.0))
                / (1.0 + torch.sqrt(torch.clamp(den_t, min=0.0)))
            )
            .detach()
            .cpu()
            .item()
        )
        relative_linf_dual_feasibility = float(
            certificate.max_positive_violation.detach().cpu().item()
        ) / (1.0 + float(certificate.cost_linf.detach().cpu().item()))
        diagnostics["relative_linf_dual_feasibility"] = float(relative_linf_dual_feasibility)
        row_values_t = scan_result.source_values
        row_idx_t = scan_result.source_indices
        row_keep = row_values_t > 0
        row_rows = torch.arange(int(source_np.shape[0]), dtype=torch.int64, device=device)[:, None].expand_as(row_idx_t)[row_keep]
        row_cols = row_idx_t[row_keep].to(dtype=torch.int64)
        col_values_t = scan_result.target_values
        col_idx_t = scan_result.target_indices
        col_keep = col_values_t > 0
        col_cols = torch.arange(int(target_np.shape[0]), dtype=torch.int64, device=device)[:, None].expand_as(col_idx_t)[col_keep]
        col_rows = col_idx_t[col_keep].to(dtype=torch.int64)
        rows_t = torch.cat([row_rows, col_rows], dim=0)
        cols_t = torch.cat([row_cols, col_cols], dim=0)
        if int(rows_t.numel()) > 0:
            keys_t = rows_t * int(target_np.shape[0]) + cols_t
            keys_t = torch.unique(keys_t, sorted=True)
            rows_np = torch.div(keys_t, int(target_np.shape[0]), rounding_mode="floor").detach().cpu().numpy().astype(np.int32, copy=False)
            cols_np = torch.remainder(keys_t, int(target_np.shape[0])).detach().cpu().numpy().astype(np.int32, copy=False)
        else:
            rows_np = np.empty(0, dtype=np.int32)
            cols_np = np.empty(0, dtype=np.int32)
    _trace_device_synchronize(trace_collector)
    diagnostics["fused_scan_topk"] = int(topk)
    diagnostics.setdefault("edge_selection_mode", "nodewise")
    return LowrankDualFeasibilityScan(
        dual_feasibility=float(dual_feasibility),
        dual_feasibility_source="early_fused_metric_scan",
        scan_time=float(time.perf_counter() - scan_start),
        diagnostics=diagnostics,
        rows=rows_np,
        cols=cols_np,
        peak_mem_mib=None,
    )
