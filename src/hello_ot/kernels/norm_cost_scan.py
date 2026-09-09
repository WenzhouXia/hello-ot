from __future__ import annotations

# CN: 本模块包含 L1、L2 与 L-infinity cost 的 fused-scan 与调度原语。
# EN: This module contains fused-scan and dispatch primitives for L1, L2, and L-infinity costs.

import importlib
import time
from functools import lru_cache
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from hello_ot._internal.core.solver import (
    HierarchicalOTSolver,
    _cost_pairs_euclidean_batched,
    _cost_pairs_l1_batched,
    _cost_pairs_linf_batched,
    _northwest_corner_numba,
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


def _metric_cost_matrix(source: np.ndarray, target: np.ndarray, cost_type: MetricCostType) -> np.ndarray:
    src = np.asarray(source, dtype=np.float32, order="C")
    tgt = np.asarray(target, dtype=np.float32, order="C")
    diff = np.abs(src[:, None, :] - tgt[None, :, :])
    if str(cost_type) == "l1":
        return np.sum(diff, axis=2, dtype=np.float32).astype(np.float64)
    if str(cost_type) == "linf":
        return np.max(diff, axis=2).astype(np.float64)
    if str(cost_type) == "l2":
        return np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32)).astype(np.float64)
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
) -> Any:
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
) -> Tuple[DualAssignmentState, Dict[str, Any]]:
    source = np.asarray(source_points_full[int(source_start) : int(source_stop)], dtype=np.float32, order="C")
    target = np.asarray(target_points_full[int(target_start) : int(target_stop)], dtype=np.float32, order="C")
    t_cost = time.perf_counter()
    cost = _metric_cost_matrix(source, target, cost_type)
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
    return result["warm_start_state"], result


_NORM_COST_SCAN_EXT_MODULE = "hello_ot._native.norm_cost_scan.hierot_norm_cost_scan_ext"
_FUSED_TOP_K_VALUES = (1, 2, 4, 8, 16, 32)
_COST_TYPE_IDS = {"l1": 0, "linf": 1, "l2": 2}


def _require_norm_cost_scan_ext() -> Any:
    """
    CN: 惰性加载安装时编译的 norm-cost scan CUDA 扩展；失败直接报错，
        不提供 KeOps 回退。
    EN: Lazily load the install-time compiled norm-cost scan CUDA extension;
        fail hard without a KeOps fallback.
    """
    try:
        module = importlib.import_module(_NORM_COST_SCAN_EXT_MODULE)
        if not getattr(module, "supports_cost_perturbation", False):
            raise RuntimeError("installed norm-cost extension predates cost perturbation; rebuild the native extensions")
        if getattr(module, "index_hash_arithmetic", None) != "fp32_remainder_v1":
            raise RuntimeError("installed norm-cost extension uses a different index hash; rebuild the native extensions")
        return module
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "norm-cost hello requires the custom scan CUDA extension "
            f"({_NORM_COST_SCAN_EXT_MODULE}); rebuild with `HELLO_OT_BUILD_NATIVE=1 pip install -e .` on a "
            "CUDA-capable machine. There is no KeOps fallback in the metric chain."
        ) from exc


def _fused_cost_type_id(cost_type: MetricCostType) -> int:
    if str(cost_type) not in _COST_TYPE_IDS:
        raise ValueError("metric hello supports cost_type in {'l1', 'linf', 'l2'}.")
    return int(_COST_TYPE_IDS[str(cost_type)])


def _fused_metric_kmin(
    *,
    query_points: np.ndarray,
    database_points: np.ndarray,
    known_dual: np.ndarray | torch.Tensor,
    cost_type: MetricCostType,
    k: int,
    query_is_source: bool = True,
    query_global_index: Optional[np.ndarray] = None,
    database_global_index: Optional[np.ndarray] = None,
    perturbation: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CN: 用 fused kernel 计算每行 (cost+perturb-dual) 的 K 个最小值。
        1..32 的任意 K 向上映射到 CUDA bucket，然后裁剪为请求的 K；不做静默回退。
    EN: Use the fused kernel for the K smallest (cost+perturb-dual) values per
        row. Any K in 1..32 is rounded up to a CUDA bucket and sliced back to the
        requested K; no silent fallback is allowed.
    """
    k_int = min(int(k), int(database_points.shape[0]))
    if k_int < 1:
        raise ValueError("metric dual assignment requires a non-empty database")
    kernel_k = custom_topk_bucket(k_int, score_family="norm_cost")
    ext = _require_norm_cost_scan_ext()
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("custom norm-cost scan requires CUDA")
    query_np = np.asarray(query_points, dtype=np.float32, order="C")
    database_np = np.asarray(database_points, dtype=np.float32, order="C")
    known_dual_np = (
        known_dual.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(known_dual)
        else np.asarray(known_dual, dtype=np.float64)
    )
    device = torch.device("cuda")
    available_bytes, driver_free_bytes, reclaimable_cache_bytes = reusable_cuda_memory_bytes(device)
    memory_plan = plan_resident_side(
        source_count=int(query_np.shape[0]),
        target_count=int(database_np.shape[0]),
        feature_dim=int(query_np.shape[1]),
        topk_bucket=int(kernel_k),
        available_bytes=int(available_bytes),
        driver_free_bytes=int(driver_free_bytes),
        reclaimable_cache_bytes=int(reclaimable_cache_bytes),
    )
    streamed_count = (
        int(query_np.shape[0])
        if memory_plan.streamed_side == "source"
        else int(database_np.shape[0])
    )
    chunk_rows = plan_stream_chunk_rows(
        memory_plan,
        streamed_count=streamed_count,
        feature_dim=int(query_np.shape[1]),
        max_rows=8192,
    )

    def run_chunk(
        query_t: torch.Tensor,
        database_t: torch.Tensor,
        dual_t: torch.Tensor,
        query_index: Optional[np.ndarray],
        database_index: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma, seed, mode, q_idx_t, d_idx_t, coeffs = _fused_perturb_arguments(
            perturbation=perturbation,
            query_global_index=query_index,
            database_global_index=database_index,
            device=device,
        )
        return ext.fused_gcost_topk(
            query_t,
            database_t,
            dual_t,
            int(kernel_k),
            int(_fused_cost_type_id(cost_type)),
            q_idx_t,
            d_idx_t,
            float(sigma),
            int(seed),
            mode,
            bool(query_is_source),
            coeffs,
        )

    if memory_plan.resident_side == "target":
        database_t = torch.as_tensor(database_np, dtype=torch.float32, device=device).contiguous()
        dual_t = torch.as_tensor(known_dual_np, dtype=torch.float64, device=device).contiguous().view(-1)
        values_parts = []
        indices_parts = []
        for q_start in range(0, int(query_np.shape[0]), chunk_rows):
            q_stop = min(int(query_np.shape[0]), q_start + chunk_rows)
            query_t = torch.as_tensor(query_np[q_start:q_stop], dtype=torch.float32, device=device).contiguous()
            q_index = None if query_global_index is None else np.asarray(query_global_index)[q_start:q_stop]
            values_t, indices_t = run_chunk(
                query_t,
                database_t,
                dual_t,
                q_index,
                database_global_index,
            )
            values_parts.append(values_t[:, :k_int].contiguous())
            indices_parts.append(indices_t[:, :k_int].to(torch.int64).contiguous())
        return torch.cat(values_parts, dim=0), torch.cat(indices_parts, dim=0)

    query_t = torch.as_tensor(query_np, dtype=torch.float32, device=device).contiguous()
    best_values = torch.full((int(query_np.shape[0]), k_int), torch.inf, dtype=torch.float64, device=device)
    best_indices = torch.full((int(query_np.shape[0]), k_int), -1, dtype=torch.int64, device=device)
    for d_start in range(0, int(database_np.shape[0]), chunk_rows):
        d_stop = min(int(database_np.shape[0]), d_start + chunk_rows)
        database_t = torch.as_tensor(database_np[d_start:d_stop], dtype=torch.float32, device=device).contiguous()
        dual_t = torch.as_tensor(known_dual_np[d_start:d_stop], dtype=torch.float64, device=device).contiguous()
        d_index = None if database_global_index is None else np.asarray(database_global_index)[d_start:d_stop]
        values_t, indices_t = run_chunk(query_t, database_t, dual_t, query_global_index, d_index)
        candidate_values = torch.cat([best_values, values_t[:, :k_int]], dim=1)
        candidate_indices = torch.cat(
            [best_indices, indices_t[:, :k_int].to(torch.int64) + int(d_start)], dim=1
        )
        best_values, order = torch.topk(candidate_values, k=k_int, dim=1, largest=False, sorted=True)
        best_indices = torch.gather(candidate_indices, 1, order).contiguous()
    return best_values.contiguous(), best_indices.contiguous()


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


def detect_metric_violations_and_check_feasibility(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_dual: np.ndarray | torch.Tensor,
    target_dual: np.ndarray | torch.Tensor,
    cost_type: MetricCostType,
    topk: int,
    perturbation: Optional[Dict[str, Any]] = None,
    source_global_index: Optional[np.ndarray] = None,
    target_global_index: Optional[np.ndarray] = None,
) -> DualViolationScanResult:
    """
    CN: 以 resident-streamed custom traversal 完成双向违例检测与完整 certificate。
    EN: Complete bidirectional violation detection and the full certificate with resident-streamed custom traversals.
    """
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("custom norm-cost fused refinement requires CUDA")
    requested_k = int(topk)
    kernel_k = custom_topk_bucket(requested_k, score_family="norm_cost")
    source_np = np.asarray(source_points, dtype=np.float32, order="C")
    target_np = np.asarray(target_points, dtype=np.float32, order="C")
    source_dual_np = (
        source_dual.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(source_dual)
        else np.asarray(source_dual, dtype=np.float64)
    )
    target_dual_np = (
        target_dual.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(target_dual)
        else np.asarray(target_dual, dtype=np.float64)
    )
    device = torch.device("cuda")
    available_bytes, driver_free_bytes, reclaimable_cache_bytes = reusable_cuda_memory_bytes(device)
    memory_plan = plan_resident_side(
        source_count=int(source_np.shape[0]),
        target_count=int(target_np.shape[0]),
        feature_dim=int(source_np.shape[1]),
        topk_bucket=int(kernel_k),
        available_bytes=int(available_bytes),
        driver_free_bytes=int(driver_free_bytes),
        reclaimable_cache_bytes=int(reclaimable_cache_bytes),
    )
    ext = _require_norm_cost_scan_ext()
    if not hasattr(ext, "fused_gcost_bidir_certificate"):
        raise RuntimeError(
            "Installed norm-cost scan extension is outdated and lacks fused_gcost_bidir_certificate; "
            "reinstall hierarchical_ot."
        )

    if memory_plan.resident_side == "target":
        resident_points_np, resident_dual_np = target_np, target_dual_np
        stream_points_np, stream_dual_np = source_np, source_dual_np
        resident_index, stream_index = target_global_index, source_global_index
        row_is_source = True
    else:
        resident_points_np, resident_dual_np = source_np, source_dual_np
        stream_points_np, stream_dual_np = target_np, target_dual_np
        resident_index, stream_index = source_global_index, target_global_index
        row_is_source = False
    resident_points_t = torch.as_tensor(resident_points_np, dtype=torch.float32, device=device).contiguous()
    resident_dual_t = torch.as_tensor(resident_dual_np, dtype=torch.float64, device=device).contiguous()
    n_resident = int(resident_points_np.shape[0])
    n_stream = int(stream_points_np.shape[0])
    partial_bytes_per_block = max(1, n_resident * kernel_k * 12)
    scratch_budget = max(0, int(memory_plan.usable_bytes - memory_plan.resident_bytes))
    minimum_stream_bytes = 8 * (int(stream_points_np.shape[1]) + 1) * 4
    if partial_bytes_per_block + minimum_stream_bytes > scratch_budget:
        raise RuntimeError(
            "norm-cost fused refinement exceeds single-GPU scratch capacity after placing the resident side: "
            f"requires at least {partial_bytes_per_block + minimum_stream_bytes} scratch bytes, "
            f"but {scratch_budget} bytes remain."
        )
    # CN: 最多将剩余 scratch 的一半分给 column partials，保留另一半给 stream feature 与临时输出。
    # EN: Give column partials at most half the remaining scratch, reserving the other half for streamed features and outputs.
    max_blocks = max(1, (scratch_budget // 2) // max(partial_bytes_per_block, 1))
    stream_row_bytes = max(1, (int(stream_points_np.shape[1]) + 4) * 4)
    max_feature_rows = (scratch_budget // 2) // stream_row_bytes
    stream_chunk = min(n_stream, 8192, int(max_blocks) * 8, int(max_feature_rows))
    if stream_chunk < 1:
        raise RuntimeError(
            "norm-cost fused refinement cannot fit one streamed feature row after reserving column partials"
        )

    resident_values = torch.full((n_resident, requested_k), -torch.inf, dtype=torch.float64, device=device)
    resident_indices = torch.full((n_resident, requested_k), -1, dtype=torch.int64, device=device)
    stream_values_parts = []
    stream_indices_parts = []
    total_num = torch.zeros((), dtype=torch.float64, device=device)
    total_den = torch.zeros((), dtype=torch.float64, device=device)
    max_violation = torch.zeros((), dtype=torch.float64, device=device)
    cost_linf = torch.zeros((), dtype=torch.float64, device=device)
    positive_count = torch.zeros((), dtype=torch.int64, device=device)
    # CN: K=32 的通用 fused column reduction 在中等维度慢于三次 custom scan；专用归约完成前使用分解路径。
    # EN: The generic K=32 fused column reduction is slower than three custom scans at medium d; use the decomposed path until a specialized reduction exists.
    use_decomposed_large_k = int(kernel_k) >= 32

    for start in range(0, n_stream, stream_chunk):
        stop = min(n_stream, start + stream_chunk)
        query_t = torch.as_tensor(stream_points_np[start:stop], dtype=torch.float32, device=device).contiguous()
        query_dual_t = torch.as_tensor(stream_dual_np[start:stop], dtype=torch.float64, device=device).contiguous()
        query_index = None if stream_index is None else np.asarray(stream_index)[start:stop]
        sigma, seed, mode, q_idx_t, d_idx_t, coeffs = _fused_perturb_arguments(
            perturbation=perturbation,
            query_global_index=query_index,
            database_global_index=resident_index,
            device=device,
        )
        if use_decomposed_large_k:
            q_cost_values, q_indices = ext.fused_gcost_topk(
                query_t,
                resident_points_t,
                resident_dual_t,
                int(kernel_k),
                int(_fused_cost_type_id(cost_type)),
                q_idx_t,
                d_idx_t,
                float(sigma),
                int(seed),
                mode,
                bool(row_is_source),
                coeffs,
            )
            r_cost_values, r_indices = ext.fused_gcost_topk(
                resident_points_t,
                query_t,
                query_dual_t,
                int(kernel_k),
                int(_fused_cost_type_id(cost_type)),
                d_idx_t,
                q_idx_t,
                float(sigma),
                int(seed),
                mode,
                not bool(row_is_source),
                coeffs,
            )
            num_t, den_t, max_t, cost_linf_t, count_t = ext.fused_gcost_certificate(
                query_t,
                resident_points_t,
                query_dual_t,
                resident_dual_t,
                int(_fused_cost_type_id(cost_type)),
                q_idx_t,
                d_idx_t,
                float(sigma),
                int(seed),
                mode,
                bool(row_is_source),
                coeffs,
                1,
            )
            q_values = query_dual_t[:, None] - q_cost_values
            r_values = resident_dual_t[:, None] - r_cost_values
        else:
            q_values, q_indices, r_values, r_indices, num_t, den_t, max_t, cost_linf_t, count_t = (
                ext.fused_gcost_bidir_certificate(
                    query_t,
                    resident_points_t,
                    query_dual_t,
                    resident_dual_t,
                    int(kernel_k),
                    int(_fused_cost_type_id(cost_type)),
                    q_idx_t,
                    d_idx_t,
                    float(sigma),
                    int(seed),
                    mode,
                    bool(row_is_source),
                    coeffs,
                )
            )
        stream_values_parts.append(q_values[:, :requested_k].contiguous())
        stream_indices_parts.append(q_indices[:, :requested_k].to(torch.int64).contiguous())
        candidate_values = torch.cat([resident_values, r_values[:, :requested_k]], dim=1)
        candidate_indices = torch.cat(
            [resident_indices, r_indices[:, :requested_k].to(torch.int64) + int(start)], dim=1
        )
        resident_values, order = torch.topk(
            candidate_values, k=requested_k, dim=1, largest=True, sorted=True
        )
        resident_indices = torch.gather(candidate_indices, 1, order).contiguous()
        total_num.add_(num_t)
        total_den.add_(den_t)
        max_violation = torch.maximum(max_violation, max_t)
        cost_linf = torch.maximum(cost_linf, cost_linf_t)
        positive_count.add_(count_t)

    stream_values = torch.cat(stream_values_parts, dim=0)
    stream_indices = torch.cat(stream_indices_parts, dim=0)
    if memory_plan.resident_side == "target":
        source_values, source_indices = stream_values, stream_indices
        target_values, target_indices = resident_values, resident_indices
    else:
        source_values, source_indices = resident_values, resident_indices
        target_values, target_indices = stream_values, stream_indices
    return DualViolationScanResult(
        source_values=source_values,
        source_indices=source_indices,
        target_values=target_values,
        target_indices=target_indices,
        certificate=DualFeasibilityCertificate(
            l2_numerator_sq=total_num,
            l2_denominator_sq=total_den,
            max_positive_violation=max_violation,
            cost_linf=cost_linf,
            positive_count=positive_count,
        ),
        diagnostics={
            "scan_backend": "custom",
            "score_family": "norm_cost",
            "resident_side": str(memory_plan.resident_side),
            "resident_bytes": int(memory_plan.resident_bytes),
            "stream_chunk_rows": int(stream_chunk),
            "topk": int(requested_k),
            "topk_bucket": int(kernel_k),
            "certificate_mode": "l2_and_linf",
            "pairwise_passes": 3 if use_decomposed_large_k else 1,
            "scan_traversal": "decomposed_large_k_custom" if use_decomposed_large_k else "fused_one_pass_custom",
        },
    )


def check_metric_dual_feasibility(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_dual: np.ndarray | torch.Tensor,
    target_dual: np.ndarray | torch.Tensor,
    cost_type: MetricCostType,
    perturbation: Optional[Dict[str, Any]] = None,
    source_global_index: Optional[np.ndarray] = None,
    target_global_index: Optional[np.ndarray] = None,
    compute_denominator: bool = True,
) -> Tuple[DualFeasibilityCertificate, Dict[str, Any]]:
    """
    CN: 单侧常驻、另一侧流式地计算完整 L2/Linf dual-feasibility certificate。
    EN: Compute the complete L2/Linf dual-feasibility certificate with one resident side and one streamed side.
    """
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("custom norm-cost certificate requires CUDA")
    source_np = np.asarray(source_points, dtype=np.float32, order="C")
    target_np = np.asarray(target_points, dtype=np.float32, order="C")
    source_dual_np = (
        source_dual.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(source_dual)
        else np.asarray(source_dual, dtype=np.float64)
    )
    target_dual_np = (
        target_dual.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(target_dual)
        else np.asarray(target_dual, dtype=np.float64)
    )
    device = torch.device("cuda")
    available_bytes, driver_free_bytes, reclaimable_cache_bytes = reusable_cuda_memory_bytes(device)
    memory_plan = plan_resident_side(
        source_count=int(source_np.shape[0]),
        target_count=int(target_np.shape[0]),
        feature_dim=int(source_np.shape[1]),
        topk_bucket=1,
        available_bytes=int(available_bytes),
        driver_free_bytes=int(driver_free_bytes),
        reclaimable_cache_bytes=int(reclaimable_cache_bytes),
    )
    ext = _require_norm_cost_scan_ext()
    cid = int(_fused_cost_type_id(cost_type))
    total_num = torch.zeros((), dtype=torch.float64, device=device)
    total_den = torch.zeros((), dtype=torch.float64, device=device)
    max_violation = torch.zeros((), dtype=torch.float64, device=device)
    cost_linf = torch.zeros((), dtype=torch.float64, device=device)
    positive_count = torch.zeros((), dtype=torch.int64, device=device)

    if memory_plan.resident_side == "target":
        resident_points = torch.as_tensor(target_np, dtype=torch.float32, device=device).contiguous()
        resident_dual = torch.as_tensor(target_dual_np, dtype=torch.float64, device=device).contiguous()
        stream_points = source_np
        stream_dual = source_dual_np
        stream_index = source_global_index
        resident_index = target_global_index
        row_is_source = True
    else:
        resident_points = torch.as_tensor(source_np, dtype=torch.float32, device=device).contiguous()
        resident_dual = torch.as_tensor(source_dual_np, dtype=torch.float64, device=device).contiguous()
        stream_points = target_np
        stream_dual = target_dual_np
        stream_index = target_global_index
        resident_index = source_global_index
        row_is_source = False

    chunk_rows = plan_stream_chunk_rows(
        memory_plan,
        streamed_count=int(stream_points.shape[0]),
        feature_dim=int(stream_points.shape[1]),
        max_rows=8192,
    )

    for start in range(0, int(stream_points.shape[0]), chunk_rows):
        stop = min(int(stream_points.shape[0]), start + chunk_rows)
        query_t = torch.as_tensor(stream_points[start:stop], dtype=torch.float32, device=device).contiguous()
        query_dual_t = torch.as_tensor(stream_dual[start:stop], dtype=torch.float64, device=device).contiguous()
        query_index = None if stream_index is None else np.asarray(stream_index)[start:stop]
        sigma, seed, mode, q_idx_t, d_idx_t, coeffs = _fused_perturb_arguments(
            perturbation=perturbation,
            query_global_index=query_index,
            database_global_index=resident_index,
            device=device,
        )
        num_t, den_t, max_t, cost_linf_t, count_t = ext.fused_gcost_certificate(
            query_t,
            resident_points,
            query_dual_t,
            resident_dual,
            cid,
            q_idx_t,
            d_idx_t,
            float(sigma),
            int(seed),
            mode,
            bool(row_is_source),
            coeffs,
            int(bool(compute_denominator)),
        )
        total_num.add_(num_t)
        if bool(compute_denominator):
            total_den.add_(den_t)
        max_violation = torch.maximum(max_violation, max_t)
        cost_linf = torch.maximum(cost_linf, cost_linf_t)
        positive_count.add_(count_t)

    certificate = DualFeasibilityCertificate(
        l2_numerator_sq=total_num,
        l2_denominator_sq=total_den,
        max_positive_violation=max_violation,
        cost_linf=cost_linf,
        positive_count=positive_count,
    )
    return certificate, {
        "scan_backend": "custom",
        "score_family": "norm_cost",
        "resident_side": str(memory_plan.resident_side),
        "resident_bytes": int(memory_plan.resident_bytes),
        "stream_chunk_rows": int(chunk_rows),
        "certificate_mode": "l2_and_linf",
        "pairwise_passes": 1,
    }


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
    if str(getattr(config, "backend", "native")) == "torch":
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


_INDEX_HASH_PRIMES = (2003, 2011)
_PERTURB_MODE_IDS = {"rank2": 0, "index_hash": 1}
MetricPerturbMode = Literal["rank2", "index_hash"]

def _full_metric_perturbation_enabled(perturbation: Optional[Dict[str, Any]]) -> bool:
    return (
        isinstance(perturbation, dict)
        and str(perturbation.get("mode", "")).strip().lower() == "full_cost_perturb"
        and float(perturbation.get("sigma", 0.0)) != 0.0
    )

@lru_cache(maxsize=32)
def _index_hash_parameters(seed: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    CN: 从种子生成两个模素数哈希的系数；取值范围保证 float32 的 KeOps 中间整数可精确表示。
    EN: Generate coefficients for two prime-modulus hashes while keeping KeOps float32 integer intermediates exact.
    """
    rng = np.random.default_rng(int(seed) & ((1 << 63) - 1))
    return tuple(
        (
            int(rng.integers(1, prime)),
            int(rng.integers(1, prime)),
            int(rng.integers(0, prime)),
        )
        for prime in _INDEX_HASH_PRIMES
    )

def _metric_index_hash_noise_numpy(source_index: Any, target_index: Any, *, seed: int) -> np.ndarray:
    """
    CN: 使用与 Torch/CUDA 相同的 FP32 近似取模与运算顺序；不截断边界值。
    EN: Use the same FP32 approximate remainder and operation order as Torch/CUDA without clipping.
    """
    src = np.asarray(source_index, dtype=np.int64)
    tgt = np.asarray(target_index, dtype=np.int64)
    hashes = []
    for prime, (src_coeff, tgt_coeff, offset) in zip(_INDEX_HASH_PRIMES, _index_hash_parameters(seed)):
        src_residue = np.remainder(src, prime).astype(np.float32)
        tgt_residue = np.remainder(tgt, prime).astype(np.float32)
        value = ((np.float32(src_coeff) * src_residue + np.float32(tgt_coeff) * tgt_residue)
                 + src_residue * tgt_residue) + np.float32(offset)
        hashes.append(value - np.float32(prime) * np.floor(value * np.float32(1.0 / prime)))
    combined = hashes[0] + np.float32(_INDEX_HASH_PRIMES[0]) * hashes[1]
    modulus = _INDEX_HASH_PRIMES[0] * _INDEX_HASH_PRIMES[1]
    return (combined + np.float32(0.5)) * np.float32(1.0 / modulus)

def _metric_noise_numpy(source_index: Any, target_index: Any, *, seed: int) -> np.ndarray:
    src = np.asarray(source_index, dtype=np.float32)
    tgt = np.asarray(target_index, dtype=np.float32)
    z = src * np.float32(12.9898) + tgt * np.float32(78.233) + np.float32(int(seed) * 0.001)
    return (np.sin(z).astype(np.float32, copy=False) * np.float32(0.5) + np.float32(0.5)).astype(np.float32, copy=False)

def _metric_perturb_mode(perturbation: Dict[str, Any]) -> MetricPerturbMode:
    mode = str(perturbation.get("noise", "rank2")).strip().lower()
    if mode not in {"rank2", "index_hash"}:
        raise ValueError("metric perturbation noise must be one of: rank2, index_hash")
    return mode

def _apply_metric_pair_perturbation(
    costs: Any,
    rows: Any,
    cols: Any,
    perturbation: Optional[Dict[str, Any]],
) -> Any:
    if not _full_metric_perturbation_enabled(perturbation):
        return costs
    sigma = float(perturbation.get("sigma", 0.0))
    seed = int(perturbation.get("seed", 0))
    perturb_mode = _metric_perturb_mode(perturbation)
    source_index = perturbation.get("source_global_index")
    target_index = perturbation.get("target_global_index")
    if source_index is None or target_index is None:
        raise ValueError("full metric perturbation requires source_global_index and target_global_index.")
    if torch.is_tensor(costs):
        device = costs.device
        rows_t = rows.to(device=device, dtype=torch.long) if torch.is_tensor(rows) else torch.as_tensor(rows, dtype=torch.long, device=device)
        cols_t = cols.to(device=device, dtype=torch.long) if torch.is_tensor(cols) else torch.as_tensor(cols, dtype=torch.long, device=device)
        src_idx = torch.as_tensor(np.asarray(source_index, dtype=np.int64), dtype=torch.long, device=device).index_select(0, rows_t)
        tgt_idx = torch.as_tensor(np.asarray(target_index, dtype=np.int64), dtype=torch.long, device=device).index_select(0, cols_t)
        if perturb_mode == "rank2":
            noise = torch.sin(src_idx.float() * 12.9898 + tgt_idx.float() * 78.233 + float(seed) * 0.001) * 0.5 + 0.5
        else:
            hashes = []
            for prime, (src_coeff, tgt_coeff, offset) in zip(_INDEX_HASH_PRIMES, _index_hash_parameters(seed)):
                src_residue = torch.remainder(src_idx, prime).float()
                tgt_residue = torch.remainder(tgt_idx, prime).float()
                value = ((float(src_coeff) * src_residue + float(tgt_coeff) * tgt_residue)
                         + src_residue * tgt_residue) + float(offset)
                hashes.append(value - float(prime) * torch.floor(value * float(np.float32(1.0 / prime))))
            modulus = _INDEX_HASH_PRIMES[0] * _INDEX_HASH_PRIMES[1]
            noise = (hashes[0] + float(_INDEX_HASH_PRIMES[0]) * hashes[1]).add(0.5).mul(float(np.float32(1.0 / modulus)))
        return costs + noise.to(dtype=costs.dtype) * float(sigma)
    rows_np = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols_np = np.asarray(cols, dtype=np.int64).reshape(-1)
    src_idx = np.asarray(source_index, dtype=np.int64)[rows_np]
    tgt_idx = np.asarray(target_index, dtype=np.int64)[cols_np]
    noise = (
        _metric_noise_numpy(src_idx, tgt_idx, seed=int(seed))
        if perturb_mode == "rank2"
        else _metric_index_hash_noise_numpy(src_idx, tgt_idx, seed=int(seed))
    )
    return np.asarray(costs, dtype=np.float64) + noise.astype(np.float64, copy=False) * float(sigma)

def _fused_perturb_arguments(
    *,
    perturbation: Optional[Dict[str, Any]],
    query_global_index: Optional[np.ndarray],
    database_global_index: Optional[np.ndarray],
    device: torch.device,
) -> Tuple[float, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CN: 把 metric 链的 perturbation dict 转成 fused kernel 的入参
        (sigma, seed, mode_id, query_global_index, database_global_index, coeffs)；
        无扰动时传 sigma=0 与空索引。
    EN: Convert a metric-chain perturbation dict into fused kernel arguments
        (sigma, seed, mode_id, query_global_index, database_global_index, coeffs);
        pass sigma=0 and empty indices when perturbation is disabled.
    """
    empty_idx = torch.empty(0, dtype=torch.int64, device=device)
    if not _full_metric_perturbation_enabled(perturbation):
        return 0.0, 0, 0, empty_idx, empty_idx, torch.zeros(6, dtype=torch.int64, device=device)
    sigma = float(perturbation.get("sigma", 0.0))
    seed = int(perturbation.get("seed", 0))
    mode = int(_PERTURB_MODE_IDS[_metric_perturb_mode(perturbation)])
    q_idx = None if query_global_index is None else np.asarray(query_global_index, dtype=np.int64)
    d_idx = None if database_global_index is None else np.asarray(database_global_index, dtype=np.int64)
    if q_idx is None or d_idx is None:
        raise ValueError("full metric perturbation requires row and column global indices for the fused kernel.")
    q_t = torch.as_tensor(q_idx, dtype=torch.int64, device=device).contiguous()
    d_t = torch.as_tensor(d_idx, dtype=torch.int64, device=device).contiguous()
    if mode == 1:
        coeffs = torch.as_tensor(
            np.asarray(sum(_index_hash_parameters(seed), ()), dtype=np.int64),
            dtype=torch.int64,
            device=device,
        ).contiguous()
    else:
        coeffs = torch.zeros(6, dtype=torch.int64, device=device)
    return sigma, seed, mode, q_t, d_t, coeffs
