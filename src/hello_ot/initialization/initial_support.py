from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from hello_ot._internal.core.solver import (
    HierarchicalOTSolver,
    _northwest_corner_numba,
    _prepare_level_cost_cache_lowrank,
)
from hello_ot._internal.instrumentation.costs import _compute_lowrank_cost_vec_from_pairs_chunked
from hello_ot._internal.trace import _ChromeTraceCollector, _trace_device_synchronize
from hello_ot.restricted_ot.active_support import ActiveSupport
from hello_ot.config import SolverRuntimeConfig
from hello_ot.state import (
    GPUWarmStartState as OTWarmStartGPUState,
    WarmStartState as OTWarmStartState,
    consume_gpu_warm_start,
    warm_start_from_gpu as _warm_start_from_gpu,
)


_WARM_START_INITIAL_CREATION_ITER = -2


def _max_optional_float(lhs: Optional[float], rhs: Any) -> Optional[float]:
    """
    CN: 汇总可选 float peak；rhs 不可转为 float 时忽略。
    EN: Aggregate optional float peaks and ignore rhs when it cannot be converted.
    """
    try:
        value = float(rhs)
    except (TypeError, ValueError):
        return lhs
    if not math.isfinite(value):
        return lhs
    return value if lhs is None else max(float(lhs), float(value))


def _get_or_prepare_shared_lowrank_level_cache(
    solver: HierarchicalOTSolver,
    level_cache_holder: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    CN: 在单次 warm-start solve 内复用 finest-level lowrank cache，避免重复 pinned copy。
    EN: Reuse the finest-level lowrank cache within one warm-start solve to avoid repeated pinned copies.
    """
    lvl_s = solver.hierarchy_s.finest_level
    lvl_t = solver.hierarchy_t.finest_level
    expected_shape = (
        tuple(np.asarray(lvl_s.points).shape),
        tuple(np.asarray(lvl_t.points).shape),
        tuple(np.asarray(lvl_s.cost_vec).shape),
        tuple(np.asarray(lvl_t.cost_vec).shape),
        float(getattr(solver, "_cost_dot_scale", 1.0)),
    )
    if level_cache_holder is not None:
        cached = level_cache_holder.get("level_cache")
        cached_shape = level_cache_holder.get("shape")
        if isinstance(cached, dict) and cached_shape == expected_shape:
            return cached

    level_cache = _prepare_level_cost_cache_lowrank(
        lvl_s.points,
        lvl_t.points,
        lvl_s.cost_vec,
        lvl_t.cost_vec,
        dot_scale=float(getattr(solver, "_cost_dot_scale", 1.0)),
    )
    if level_cache_holder is not None:
        level_cache_holder["level_cache"] = level_cache
        level_cache_holder["shape"] = expected_shape
    return level_cache



def _validate_lowrank_warm_start(
    warm_start: OTWarmStartState | OTWarmStartGPUState,
    *,
    n_s: int,
    n_t: int,
    include_dual: bool = True,
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
        if rows.ndim != 1 or cols.ndim != 1 or x_prev.ndim != 1:
            raise ValueError("warm_start rows, cols, and x_prev must be 1D arrays.")
        if not (int(rows.numel()) == int(cols.numel()) == int(x_prev.numel())):
            raise ValueError("warm_start rows, cols, and x_prev must have the same length.")
        if bool(torch.any((rows < 0) | (rows >= int(n_s))).item()):
            raise ValueError("warm_start.rows contains out-of-range indices.")
        if bool(torch.any((cols < 0) | (cols >= int(n_t))).item()):
            raise ValueError("warm_start.cols contains out-of-range indices.")
        dual_uv = None
        if bool(include_dual) and warm_start.dual_uv is not None:
            dual_t = warm_start.dual_uv.detach()
            if dual_t.ndim != 1 or int(dual_t.numel()) != int(n_s + n_t):
                raise ValueError(
                    "warm_start.dual_uv must be a 1D array with length n_source + n_target."
                )
            dual_uv = dual_t.cpu().numpy().astype(np.float64, copy=False)
        return rows, cols, x_prev, dual_uv

    rows = np.asarray(warm_start.rows, dtype=np.int32)
    cols = np.asarray(warm_start.cols, dtype=np.int32)
    x_prev = np.asarray(warm_start.x_prev, dtype=np.float64)
    if rows.ndim != 1 or cols.ndim != 1 or x_prev.ndim != 1:
        raise ValueError("warm_start rows, cols, and x_prev must be 1D arrays.")
    if not (rows.size == cols.size == x_prev.size):
        raise ValueError("warm_start rows, cols, and x_prev must have the same length.")
    if np.any(rows < 0) or np.any(rows >= int(n_s)):
        raise ValueError("warm_start.rows contains out-of-range indices.")
    if np.any(cols < 0) or np.any(cols >= int(n_t)):
        raise ValueError("warm_start.cols contains out-of-range indices.")

    dual_uv = None
    if bool(include_dual) and warm_start.dual_uv is not None:
        dual_uv = np.asarray(warm_start.dual_uv, dtype=np.float64)
        if dual_uv.ndim != 1 or dual_uv.shape[0] != int(n_s + n_t):
            raise ValueError(
                "warm_start.dual_uv must be a 1D array with length n_source + n_target."
            )
    return rows, cols, x_prev, dual_uv


def _augment_active_support_with_northwest_corner(
    solver: HierarchicalOTSolver,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    northwest_holder: Optional[Dict[str, Any]] = None,
    *,
    placeholder_costs: bool = False,
) -> int:
    n_target = solver.active_support.n_target
    expected_shape = (tuple(np.asarray(source_mass).shape), tuple(np.asarray(target_mass).shape), int(n_target))
    nw_rows = nw_cols = nw_keys = None
    if northwest_holder is not None and northwest_holder.get("shape") == expected_shape:
        nw_rows = northwest_holder.get("rows")
        nw_cols = northwest_holder.get("cols")
        nw_keys = northwest_holder.get("keys")
    if nw_rows is None or nw_cols is None or nw_keys is None:
        nw_rows, nw_cols = _northwest_corner_numba(
            np.asarray(source_mass, dtype=np.float64),
            np.asarray(target_mass, dtype=np.float64),
        )
        nw_keys = nw_rows.astype(np.int64) * int(n_target) + nw_cols.astype(np.int64)
        if northwest_holder is not None:
            northwest_holder["shape"] = expected_shape
            northwest_holder["rows"] = nw_rows
            northwest_holder["cols"] = nw_cols
            northwest_holder["keys"] = nw_keys
    if nw_rows.size == 0:
        return 0
    if getattr(solver.active_support, "is_torch_backend", False):
        device = solver.active_support.device
        nw_keys_t = torch.as_tensor(nw_keys, dtype=torch.int64, device=device)
        active_keys_t = solver.active_support.keys.to(dtype=torch.int64)
        if (
            bool(getattr(solver.active_support, "track_creation", False))
            and solver.active_support.creation_iteration is not None
            and int(active_keys_t.numel()) > 0
        ):
            existing_nw_mask_t = torch.isin(active_keys_t, nw_keys_t)
            if bool(torch.any(existing_nw_mask_t).item()):
                solver.active_support.creation_iteration[existing_nw_mask_t] = -1
        if int(solver.active_support.keys.numel()) == 0:
            missing_keys_t = nw_keys_t
        else:
            missing_mask_t = ~torch.isin(nw_keys_t, active_keys_t)
            missing_keys_t = nw_keys_t[missing_mask_t]
        if int(missing_keys_t.numel()) == 0:
            return 0
        rows_add_t = torch.div(missing_keys_t, int(n_target), rounding_mode="floor").to(dtype=torch.int32)
        cols_add_t = torch.remainder(missing_keys_t, int(n_target)).to(dtype=torch.int32)
        if bool(placeholder_costs):
            solver.active_support.add_pairs_placeholder(rows_add_t, cols_add_t, iter_idx=-1)
        else:
            solver._append_new_pairs_arrays(rows_add_t, cols_add_t, -1)
        return int(rows_add_t.numel())
    if (
        bool(getattr(solver.active_support, "track_creation", False))
        and solver.active_support.creation_iteration is not None
        and np.asarray(solver.active_support.keys).size > 0
    ):
        existing_nw_mask = np.isin(np.asarray(solver.active_support.keys, dtype=np.int64), nw_keys, assume_unique=False)
        if np.any(existing_nw_mask):
            solver.active_support.creation_iteration[existing_nw_mask] = -1
    missing_keys = np.setdiff1d(nw_keys, solver.active_support.keys, assume_unique=False)
    if missing_keys.size == 0:
        return 0
    rows_add = (missing_keys // n_target).astype(np.int32)
    cols_add = (missing_keys % n_target).astype(np.int32)
    if bool(placeholder_costs):
        solver.active_support.add_pairs_placeholder(rows_add, cols_add, iter_idx=-1)
    else:
        solver._append_new_pairs_arrays(rows_add, cols_add, -1)
    return int(rows_add.size)


def _mark_existing_northwest_positions(
    solver: HierarchicalOTSolver,
    northwest_positions: Any,
) -> bool:
    """
    CN: 用已知位置把 active support 中已有的 NW skeleton 边标为 protected。
    EN: Mark existing NW skeleton edges in active support as protected using known positions.
    """
    if northwest_positions is None:
        return False
    active_support = solver.active_support
    if not (
        bool(getattr(active_support, "track_creation", False))
        and active_support.creation_iteration is not None
    ):
        return True
    if getattr(active_support, "is_torch_backend", False):
        if torch is None:
            return False
        if torch.is_tensor(northwest_positions):
            pos_t = northwest_positions.to(device=active_support.device, dtype=torch.long)
        else:
            pos_t = torch.as_tensor(northwest_positions, dtype=torch.long, device=active_support.device)
        pos_t = pos_t.reshape(-1)
        if int(pos_t.numel()) == 0:
            return True
        valid_t = (pos_t >= 0) & (pos_t < int(active_support.creation_iteration.numel()))
        if not bool(torch.all(valid_t).item()):
            return False
        active_support.creation_iteration[pos_t] = -1
        return True
    pos_np = np.asarray(northwest_positions, dtype=np.int64).reshape(-1)
    if int(pos_np.size) == 0:
        return True
    if np.any(pos_np < 0) or np.any(pos_np >= int(np.asarray(active_support.creation_iteration).size)):
        return False
    active_support.creation_iteration[pos_np] = -1
    return True


def _gpu_warm_start_support_bytes(warm_start: OTWarmStartGPUState) -> int:
    """
    CN: 估算 initial-support 构造期间可通过 CPU staging 移除的 GPU support bytes。
    EN: Estimate GPU support bytes removable from the initial-support construction peak through CPU staging.
    """
    tensors = (warm_start.rows, warm_start.cols, warm_start.x_prev, warm_start.keys)
    seen: set[Tuple[int, int]] = set()
    total = 0
    for tensor in tensors:
        if not torch.is_tensor(tensor) or not tensor.is_cuda:
            continue
        storage = tensor.untyped_storage()
        key = (int(tensor.device.index or 0), int(storage.data_ptr()))
        if key in seen:
            continue
        seen.add(key)
        total += int(storage.nbytes())
    return int(total)


def _prepare_cpu_initial_support(
    warm_start: OTWarmStartGPUState,
    *,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    northwest_holder: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    CN: 将已完成 dual assignment 的临时 support 导出到 CPU，并补齐/标记 NW skeleton。
    EN: Export the temporary dual-assigned support to CPU and complete/mark the NW skeleton.
    """
    n_source = int(warm_start.n_source)
    n_target = int(warm_start.n_target)
    rows = warm_start.rows.detach().cpu().numpy().astype(np.int32, copy=False).copy()
    cols = warm_start.cols.detach().cpu().numpy().astype(np.int32, copy=False).copy()
    x_prev = warm_start.x_prev.detach().cpu().numpy().astype(np.float64, copy=False).copy()
    primal_nnz = int(rows.size)
    keys = rows.astype(np.int64) * np.int64(n_target) + cols.astype(np.int64)
    creation = np.full(int(rows.size), _WARM_START_INITIAL_CREATION_ITER, dtype=np.int32)
    northwest_positions = None
    if warm_start.northwest_positions is not None:
        northwest_positions = (
            warm_start.northwest_positions.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1).copy()
        )

    northwest_added = 0
    if northwest_positions is not None and (
        northwest_positions.size == 0
        or (np.all(northwest_positions >= 0) and np.all(northwest_positions < int(rows.size)))
    ):
        creation[northwest_positions] = -1
    else:
        expected_shape = (tuple(np.asarray(source_mass).shape), tuple(np.asarray(target_mass).shape), n_target)
        nw_rows = nw_cols = nw_keys = None
        if northwest_holder is not None and northwest_holder.get("shape") == expected_shape:
            nw_rows = northwest_holder.get("rows")
            nw_cols = northwest_holder.get("cols")
            nw_keys = northwest_holder.get("keys")
        if nw_rows is None or nw_cols is None or nw_keys is None:
            nw_rows, nw_cols = _northwest_corner_numba(
                np.asarray(source_mass, dtype=np.float64),
                np.asarray(target_mass, dtype=np.float64),
            )
            nw_keys = nw_rows.astype(np.int64) * np.int64(n_target) + nw_cols.astype(np.int64)
            if northwest_holder is not None:
                northwest_holder.update(
                    {"shape": expected_shape, "rows": nw_rows, "cols": nw_cols, "keys": nw_keys}
                )
        if int(nw_keys.size) > 0 and int(keys.size) > 0:
            order = np.argsort(keys, kind="stable")
            sorted_keys = keys[order]
            positions = np.searchsorted(sorted_keys, nw_keys)
            found = positions < int(sorted_keys.size)
            found[found] &= sorted_keys[positions[found]] == nw_keys[found]
            if np.any(found):
                creation[order[positions[found]]] = -1
        else:
            found = np.zeros(int(nw_keys.size), dtype=bool)
        missing = ~found
        if np.any(missing):
            rows = np.concatenate([rows, np.asarray(nw_rows[missing], dtype=np.int32)])
            cols = np.concatenate([cols, np.asarray(nw_cols[missing], dtype=np.int32)])
            x_prev = np.concatenate([x_prev, np.zeros(int(np.count_nonzero(missing)), dtype=np.float64)])
            keys = np.concatenate([keys, np.asarray(nw_keys[missing], dtype=np.int64)])
            creation = np.concatenate(
                [creation, np.full(int(np.count_nonzero(missing)), -1, dtype=np.int32)]
            )
            northwest_added = int(np.count_nonzero(missing))
    return {
        "rows": rows,
        "cols": cols,
        "x_prev": x_prev,
        "keys": keys,
        "creation_iteration": creation,
        "primal_nnz": primal_nnz,
        "northwest_added": int(northwest_added),
    }


def _construct_from_consumed_gpu_warm_start(
    solver: HierarchicalOTSolver,
    warm_start: OTWarmStartGPUState,
    *,
    trace_collector: Optional[_ChromeTraceCollector],
    trace_prefix: str,
    cost_vec_chunk_size: int,
    cost_vec_feature_chunk_size: Optional[int],
    level_cache_holder: Optional[Dict[str, Any]],
    northwest_holder: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    CN: 消费 GPU warm-start，统一通过 CPU staging 延迟构造正式 ActiveSupport。
    EN: Consume a GPU warm start and uniformly defer formal ActiveSupport construction through CPU staging.
    """
    lvl_s = solver.hierarchy_s.finest_level
    lvl_t = solver.hierarchy_t.finest_level
    components: Dict[str, float] = {}
    t0 = time.perf_counter()
    level_cache = _get_or_prepare_shared_lowrank_level_cache(solver, level_cache_holder)
    components["build_level_cache"] = time.perf_counter() - t0
    staged_bytes = _gpu_warm_start_support_bytes(warm_start)
    warm_start_device = str(warm_start.device)
    t0 = time.perf_counter()
    pending = _prepare_cpu_initial_support(
        warm_start,
        source_mass=lvl_s.masses,
        target_mass=lvl_t.masses,
        northwest_holder=northwest_holder,
    )
    consume_gpu_warm_start(warm_start, keep_dual=False)
    components["stage_support_cpu"] = time.perf_counter() - t0
    components["init_active_support"] = 0.0
    components["validate_warm_start"] = 0.0
    components["add_pairs"] = 0.0
    components["warm_start_fill"] = 0.0
    components["bfs_skeleton"] = 0.0
    components["order_cache_build"] = 0.0

    t0 = time.perf_counter()
    c_vec = _compute_lowrank_cost_vec_from_pairs_chunked(
        level_cache,
        pending["rows"],
        pending["cols"],
        chunk_size=int(cost_vec_chunk_size),
        feature_chunk_size=cost_vec_feature_chunk_size,
        progress_tag="InitialSupport",
        verbose=False,
        tracer=trace_collector,
        trace_args=None,
        support_order_cache=None,
    )
    components["compute_c_vec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    active_support = ActiveSupport(
        n_source=len(lvl_s.points),
        n_target=len(lvl_t.points),
        level_cache=level_cache,
        track_creation=bool(solver.cleaning_strategy.needs_creation_iteration()),
        device=warm_start_device,
    )
    active_support.initialize_from_arrays(
        rows=pending["rows"],
        cols=pending["cols"],
        costs=c_vec,
        x_prev=pending["x_prev"],
        keys=pending["keys"],
        creation_iteration=pending["creation_iteration"],
    )
    solver.active_support = active_support
    components["materialize_active_support"] = time.perf_counter() - t0
    primal_nnz = int(pending["primal_nnz"])
    initial_support_size = int(primal_nnz)
    northwest_added = int(pending["northwest_added"])
    del pending, c_vec

    t0 = time.perf_counter()
    active_support.get_order_cache(
        n_source=len(lvl_s.points),
        n_target=len(lvl_t.points),
        device=active_support.device,
    )
    components["order_cache_build"] = time.perf_counter() - t0
    return {
        "components": components,
        "primal_nnz": int(primal_nnz),
        "initial_support_size": int(initial_support_size),
        "northwest_added": int(northwest_added),
        "initial_support_staging": "cpu",
        "initial_support_staged_bytes": int(staged_bytes),
        "consumed_warm_start_gpu_state": True,
    }


def _construct_initial_active_support_basic(
    solver: HierarchicalOTSolver,
    warm_start: OTWarmStartState | OTWarmStartGPUState,
    *,
    trace_collector: Optional[_ChromeTraceCollector] = None,
    trace_prefix: str = "solve_ot",
    cost_vec_chunk_size: int = 65536,
    cost_vec_feature_chunk_size: Optional[int] = None,
    level_cache_holder: Optional[Dict[str, Any]] = None,
    northwest_holder: Optional[Dict[str, Any]] = None,
    consume_warm_start_gpu_state: bool = False,
) -> Dict[str, Any]:
    if bool(consume_warm_start_gpu_state) and isinstance(warm_start, OTWarmStartGPUState):
        return _construct_from_consumed_gpu_warm_start(
            solver,
            warm_start,
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            cost_vec_chunk_size=int(cost_vec_chunk_size),
            cost_vec_feature_chunk_size=cost_vec_feature_chunk_size,
            level_cache_holder=level_cache_holder,
            northwest_holder=northwest_holder,
        )
    lvl_s = solver.hierarchy_s.finest_level
    lvl_t = solver.hierarchy_t.finest_level
    components: Dict[str, float] = {}
    use_gpu_state = isinstance(warm_start, OTWarmStartGPUState)

    t0 = time.perf_counter()
    with (
        trace_collector.span(f"{trace_prefix}.init.build_level_cache", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        level_cache = _get_or_prepare_shared_lowrank_level_cache(solver, level_cache_holder)
    components["build_level_cache"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with (
        trace_collector.span(f"{trace_prefix}.init.init_active_support", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        active_support = ActiveSupport(
            n_source=len(lvl_s.points),
            n_target=len(lvl_t.points),
            level_cache=level_cache,
            track_creation=bool(solver.cleaning_strategy.needs_creation_iteration()),
            device=str(warm_start.device) if use_gpu_state else "cuda",
        )
    components["init_active_support"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with (
        trace_collector.span(f"{trace_prefix}.init.validate_warm_start", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        rows, cols, x_prev, _ = _validate_lowrank_warm_start(
            warm_start,
            n_s=len(lvl_s.points),
            n_t=len(lvl_t.points),
            include_dual=False,
        )
    components["validate_warm_start"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with (
        trace_collector.span(f"{trace_prefix}.init.add_pairs", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        active_support.add_pairs_placeholder(rows, cols, iter_idx=_WARM_START_INITIAL_CREATION_ITER)
        active_support.set_x_prev(x_prev)
    components["add_pairs"] = time.perf_counter() - t0
    components["warm_start_fill"] = 0.0 if use_gpu_state else components["add_pairs"]
    solver.active_support = active_support
    initial_support_size = int(active_support.size)

    t0 = time.perf_counter()
    with (
        trace_collector.span(f"{trace_prefix}.init.bfs_skeleton", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        if _mark_existing_northwest_positions(solver, getattr(warm_start, "northwest_positions", None)):
            northwest_added = 0
        else:
            northwest_added = _augment_active_support_with_northwest_corner(
                solver,
                lvl_s.masses,
                lvl_t.masses,
                northwest_holder=northwest_holder,
                placeholder_costs=True,
            )
    components["bfs_skeleton"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with (
        trace_collector.span(f"{trace_prefix}.init.order_cache_build", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        order_cache = active_support.get_order_cache(
            n_source=len(lvl_s.points),
            n_target=len(lvl_t.points),
            device=str(warm_start.device) if use_gpu_state else "cuda",
        )
    components["order_cache_build"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with (
        trace_collector.span(f"{trace_prefix}.init.compute_c_vec", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        _trace_device_synchronize(trace_collector)
        c_vec = _compute_lowrank_cost_vec_from_pairs_chunked(
            level_cache,
            active_support.rows,
            active_support.cols,
            chunk_size=int(cost_vec_chunk_size),
            feature_chunk_size=cost_vec_feature_chunk_size,
            progress_tag="InitialSupport",
            verbose=False,
            tracer=trace_collector,
            trace_args=None,
            support_order_cache=order_cache,
        )
        active_support.replace_costs(c_vec)
        _trace_device_synchronize(trace_collector)
    components["compute_c_vec"] = time.perf_counter() - t0

    return {
        "components": components,
        "primal_nnz": int(rows.numel()) if torch.is_tensor(rows) else int(np.asarray(rows).shape[0]),
        "initial_support_size": initial_support_size,
        "northwest_added": int(northwest_added),
    }


def _construct_initial_active_support(
    solver: HierarchicalOTSolver,
    warm_start: OTWarmStartState | OTWarmStartGPUState,
    *,
    skip_initial_pricing: bool = False,
    trace_collector: Optional[_ChromeTraceCollector] = None,
    trace_prefix: str = "solve_ot",
    cost_vec_chunk_size: int = 65536,
    cost_vec_feature_chunk_size: Optional[int] = None,
    level_cache_holder: Optional[Dict[str, Any]] = None,
    northwest_holder: Optional[Dict[str, Any]] = None,
    consume_warm_start_gpu_state: bool = False,
) -> Dict[str, Any]:
    lvl_s = solver.hierarchy_s.finest_level
    lvl_t = solver.hierarchy_t.finest_level
    if bool(skip_initial_pricing):
        initial_support_info = _construct_initial_active_support_basic(
            solver,
            warm_start,
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            cost_vec_chunk_size=int(cost_vec_chunk_size),
            cost_vec_feature_chunk_size=cost_vec_feature_chunk_size,
            level_cache_holder=level_cache_holder,
            northwest_holder=northwest_holder,
            consume_warm_start_gpu_state=bool(consume_warm_start_gpu_state),
        )
        components = dict(initial_support_info.get("components", {}))
        components["initial_pricing"] = 0.0
        components["append_initial_candidates"] = 0.0
        return {
            **initial_support_info,
            "components": components,
            "warm_dual_pricing_added": 0,
            "post_pricing_support_size": int(solver.active_support.size),
        }

    rows, cols, x_prev, dual_uv = _validate_lowrank_warm_start(
        warm_start,
        n_s=len(lvl_s.points),
        n_t=len(lvl_t.points),
    )
    if dual_uv is None:
        return _construct_initial_active_support_basic(
            solver,
            warm_start,
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            cost_vec_chunk_size=int(cost_vec_chunk_size),
            cost_vec_feature_chunk_size=cost_vec_feature_chunk_size,
            level_cache_holder=level_cache_holder,
            northwest_holder=northwest_holder,
            consume_warm_start_gpu_state=False,
        )

    initial_support_info = _construct_initial_active_support_basic(
        solver,
        warm_start,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        cost_vec_chunk_size=int(cost_vec_chunk_size),
        cost_vec_feature_chunk_size=cost_vec_feature_chunk_size,
        level_cache_holder=level_cache_holder,
        northwest_holder=northwest_holder,
        consume_warm_start_gpu_state=False,
    )

    n_s = len(lvl_s.points)
    components = dict(initial_support_info.get("components", {}))
    t0 = time.perf_counter()
    with (
        trace_collector.span(f"{trace_prefix}.init.initial_pricing", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        with (
            trace_collector.span(f"{trace_prefix}.init.initial_pricing.build_primal_curr", "solve_ot")
            if trace_collector is not None
            else nullcontext()
        ):
            if isinstance(warm_start, OTWarmStartGPUState):
                rows_primal = rows.detach().cpu().numpy().astype(np.int64, copy=False)
                cols_primal = cols.detach().cpu().numpy().astype(np.int64, copy=False)
                x_prev_primal = x_prev.detach().cpu().numpy().astype(np.float64, copy=False)
            else:
                rows_primal = rows.astype(np.int64, copy=False)
                cols_primal = cols.astype(np.int64, copy=False)
                x_prev_primal = x_prev
            primal_curr = sp.csc_matrix(
                (x_prev_primal, (rows_primal, cols_primal)),
                shape=(n_s, len(lvl_t.points)),
                dtype=np.float64,
            )
        with (
            trace_collector.span(f"{trace_prefix}.init.initial_pricing.build_dual_pair", "solve_ot")
            if trace_collector is not None
            else nullcontext()
        ):
            dual_pair = (
                np.asarray(dual_uv[:n_s], dtype=np.float64),
                np.asarray(dual_uv[n_s:], dtype=np.float64),
            )
        with (
            trace_collector.span(f"{trace_prefix}.init.initial_pricing.strategy_generate", "solve_ot")
            if trace_collector is not None
            else nullcontext()
        ):
            _trace_device_synchronize(trace_collector)
            init_cands = solver.strategy.generate(
                primal_curr,
                dual_pair,
                solver.active_support.level_cache,
                level_idx=0,
                inner_iter=-1,
                trace_collector=trace_collector,
                trace_prefix=f"{trace_prefix}.init.initial_pricing",
            )
            _trace_device_synchronize(trace_collector)
    components["initial_pricing"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with (
        trace_collector.span(f"{trace_prefix}.init.append_initial_candidates", "solve_ot")
        if trace_collector is not None
        else nullcontext()
    ):
        init_rows, init_cols = init_cands
        init_found_count = int(solver.active_support._array_size(init_rows))
        added_rows, added_cols, found_count, added_count = solver.active_support.filter_new_candidate_pairs(
            init_rows,
            init_cols,
        )
        solver._append_new_pairs_arrays(added_rows, added_cols, _WARM_START_INITIAL_CREATION_ITER)
    components["append_initial_candidates"] = time.perf_counter() - t0

    return {
        **initial_support_info,
        "components": components,
        "warm_dual_pricing_found": int(init_found_count),
        "warm_dual_pricing_added": int(added_count),
        "warm_dual_pricing_already_active": int(found_count - added_count),
        "post_pricing_support_size": int(solver.active_support.size),
    }



__all__ = [
    "_max_optional_float",
    "_get_or_prepare_shared_lowrank_level_cache",
    "_validate_lowrank_warm_start",
    "_augment_active_support_with_northwest_corner",
    "_mark_existing_northwest_positions",
    "_construct_initial_active_support_basic",
    "_construct_initial_active_support",
    "_WARM_START_INITIAL_CREATION_ITER",
]
