from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from hello_ot._internal.core.solver import (
    HierarchicalOTSolver,
    _northwest_corner_numba,
    _prepare_level_cost_cache_lowrank,
)
from hello_ot._internal.instrumentation.costs import (
    _compute_lowrank_cost_vec_from_pairs_chunked,
    _extract_level_shape_from_cache,
)
from hello_ot._internal.instrumentation.reporting import (
    _config_runtime_log_enabled,
    _fmt_lp_dimensions,
    _fmt_optional_fixed,
    _fmt_optional_sci,
    _format_profile_components,
)
from hello_ot._internal.trace import _ChromeTraceCollector
from hello_ot._internal.lp.wrapper import SolverResult
from hello_ot.refinement.budgeted_pruning import DualGapCleaning, NoCleaning
from hello_ot.restricted_ot.runtime import extract_level_zero_summary
from hello_ot.config import SolverRuntimeConfig
from hello_ot.state import GPUWarmStartState as OTWarmStartGPUState, WarmStartState as OTWarmStartState

from hello_ot.initialization.state import _normalize_dual_assignment_state, _state_support_size
from hello_ot.hierarchy.utilities import _normalize_subproblem_masses

_POT_MODULE: Optional[Any] = None


def _get_pot_module() -> Any:
    global _POT_MODULE
    if _POT_MODULE is None:
        try:
            import ot  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("POT leaf solver requires package 'ot' to be installed.") from exc
        _POT_MODULE = ot
    return _POT_MODULE



def _solve_lowrank_leaf_with_pot_from_precomputed_cost(
    *,
    cost: np.ndarray,
    source_f: np.ndarray,
    target_g: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_args: Optional[Dict[str, Any]] = None,
    precomputed_cost_build_time: float = 0.0,
    cost_build_backend: str = "precomputed",
    cost_build_batch_size: int = 1,
    cost_build_device: str = "cpu",
) -> Dict[str, Any]:
    t_import = time.perf_counter()
    with (tracer.span("leaf.pot_import", "leaf", args=trace_args) if tracer is not None else nullcontext()):
        ot = _get_pot_module()
    import_time = float(time.perf_counter() - t_import)

    t_mass = time.perf_counter()
    with (tracer.span("leaf.mass_cast", "leaf", args=trace_args) if tracer is not None else nullcontext()):
        source_mass64 = np.asarray(source_mass, dtype=np.float64)
        target_mass64 = np.asarray(target_mass, dtype=np.float64)
        cost64 = np.asarray(cost, dtype=np.float64)
    mass_cast_time = float(time.perf_counter() - t_mass)

    t0 = time.perf_counter()
    with (tracer.span("leaf.ot_solve", "leaf", args=trace_args) if tracer is not None else nullcontext()):
        plan, log = ot.emd(
            source_mass64,
            target_mass64,
            cost64,
            numItermax=10_000_000,
            log=True,
        )
    pot_solve_time = float(time.perf_counter() - t0)
    t_sparse = time.perf_counter()
    with (tracer.span("leaf.post_sparse_extract", "leaf", args=trace_args) if tracer is not None else nullcontext()):
        rows, cols = np.nonzero(plan > 1e-12)
        vals = np.asarray(plan[rows, cols], dtype=np.float64)
    post_sparse_extract_time = float(time.perf_counter() - t_sparse)
    t_dual = time.perf_counter()
    with (tracer.span("leaf.post_dual_pack", "leaf", args=trace_args) if tracer is not None else nullcontext()):
        dual_u = np.asarray(log.get("u"), dtype=np.float64)
        dual_v = np.asarray(log.get("v"), dtype=np.float64)
        dual_uv = np.concatenate([dual_u, dual_v]).astype(np.float64, copy=False)
    post_dual_pack_time = float(time.perf_counter() - t_dual)
    t_state = time.perf_counter()
    with (tracer.span("leaf.post_state_build", "leaf", args=trace_args) if tracer is not None else nullcontext()):
        warm_state = OTWarmStartState(
            rows=np.asarray(rows, dtype=np.int32),
            cols=np.asarray(cols, dtype=np.int32),
            x_prev=vals,
            dual_uv=dual_uv,
            n_source=int(source_f.shape[0]),
            n_target=int(target_g.shape[0]),
        )
    post_state_build_time = float(time.perf_counter() - t_state)
    t_distance = time.perf_counter()
    with (tracer.span("leaf.post_distance_eval", "leaf", args=trace_args) if tracer is not None else nullcontext()):
        if "cost" in log:
            distance = float(log["cost"])
        else:
            distance = float(np.asarray(plan, dtype=np.float64)[rows, cols] @ cost64[rows, cols])
    post_distance_eval_time = float(time.perf_counter() - t_distance)
    post_total_time = (
        post_sparse_extract_time
        + post_dual_pack_time
        + post_state_build_time
        + post_distance_eval_time
    )
    leaf_stage_profile = {
        "pot_import_time": import_time,
        "pot_mass_cast_time": mass_cast_time,
        "pot_cost_build_time": float(precomputed_cost_build_time),
        "pot_solve_time": pot_solve_time,
        "post_sparse_extract_time": post_sparse_extract_time,
        "post_dual_pack_time": post_dual_pack_time,
        "post_state_build_time": post_state_build_time,
        "post_distance_eval_time": post_distance_eval_time,
        "post_total_time": post_total_time,
        "cost_build_backend": str(cost_build_backend),
        "cost_build_batch_size": int(cost_build_batch_size),
        "cost_build_device": str(cost_build_device),
    }
    return {
        "distance": distance,
        "time": pot_solve_time,
        "lp_solve_time_total": pot_solve_time,
        "warm_start_state": warm_state,
        "leaf_stage_profile": leaf_stage_profile,
        "level_summaries": [
            {
                "level": 0,
                "n_source": int(source_f.shape[0]),
                "n_target": int(target_g.shape[0]),
                "iters": 1,
                "time": pot_solve_time,
                "objective": distance,
                "lp_time": pot_solve_time,
                "pricing_time": 0.0,
                "support_final": int(rows.size),
            }
        ],
    }


def _leaf_cost_build_device() -> torch.device:
    """
    CN: 选择 leaf Torch cost-build 使用的设备，优先 CUDA，缺失时回落 CPU。
    EN: Choose the device for leaf Torch cost building, preferring CUDA and falling back to CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _build_leaf_costs_pad_batch_gemm(
    *,
    leaf_inputs: List[Dict[str, Any]],
    tracer: Optional[_ChromeTraceCollector] = None,
    depth_level: Optional[int] = None,
    dot_scale: float = 1.0,
) -> Dict[int, Dict[str, Any]]:
    """
    CN: 将不同尺寸 leaf padding 到同一矩形 batch，用一次 Torch bmm 构造 cost 后再裁回原尺寸。
    EN: Pad variable-size leaves into one rectangular batch, build costs with one Torch bmm, then crop back to original shapes.
    """
    outputs: Dict[int, Dict[str, Any]] = {}
    if not leaf_inputs:
        return outputs
    device = _leaf_cost_build_device()
    max_n_source = max(int(item["n_source"]) for item in leaf_inputs)
    max_n_target = max(int(item["n_target"]) for item in leaf_inputs)
    feat_dim = int(leaf_inputs[0]["source_f"].shape[1])
    batch_size = int(len(leaf_inputs))
    trace_args = {
        "depth": int(depth_level) if depth_level is not None else -1,
        "mode": "pad_batch_gemm",
        "batch_size": batch_size,
        "max_n_source": int(max_n_source),
        "max_n_target": int(max_n_target),
        "feat_dim": int(feat_dim),
        "device": str(device),
    }
    with (tracer.span("leaf.batch_cost_build", "leaf", args=trace_args) if tracer is not None else nullcontext()):
        t0 = time.perf_counter()
        with (tracer.span("leaf.batch_cost_build.prepare_tensors", "leaf", args=trace_args) if tracer is not None else nullcontext()):
            # CN: padding 区域的 cost 后续会被裁掉，因此这里仅需要保证真实区域写入正确。
            # EN: Padded cost entries are cropped later, so only real regions need to be filled correctly.
            source_f_batch = torch.zeros((batch_size, max_n_source, feat_dim), dtype=torch.float32, device=device)
            target_g_batch = torch.zeros((batch_size, max_n_target, feat_dim), dtype=torch.float32, device=device)
            source_cost_batch = torch.zeros((batch_size, max_n_source), dtype=torch.float64, device=device)
            target_cost_batch = torch.zeros((batch_size, max_n_target), dtype=torch.float64, device=device)
            for idx, item in enumerate(leaf_inputs):
                ns = int(item["n_source"])
                nt = int(item["n_target"])
                source_f_batch[idx, :ns, :] = torch.as_tensor(item["source_f"], dtype=torch.float32, device=device)
                target_g_batch[idx, :nt, :] = torch.as_tensor(item["target_g"], dtype=torch.float32, device=device)
                source_cost_batch[idx, :ns] = torch.as_tensor(item["source_cost"], dtype=torch.float64, device=device)
                target_cost_batch[idx, :nt] = torch.as_tensor(item["target_cost"], dtype=torch.float64, device=device)
        with (tracer.span("leaf.batch_cost_build.gemm", "leaf", args=trace_args) if tracer is not None else nullcontext()):
            score_batch = torch.bmm(source_f_batch, target_g_batch.transpose(1, 2))
        with (tracer.span("leaf.batch_cost_build.bias_add", "leaf", args=trace_args) if tracer is not None else nullcontext()):
            cost_batch = (
                source_cost_batch[:, :, None]
                + target_cost_batch[:, None, :]
                - float(dot_scale) * score_batch
            )
        with (tracer.span("leaf.batch_cost_build.copy_to_cpu", "leaf", args=trace_args) if tracer is not None else nullcontext()):
            cost_batch_np = cost_batch.detach().cpu().numpy()
        batch_dt = float(time.perf_counter() - t0)
    total_work = max(sum(int(item["n_source"]) * int(item["n_target"]) for item in leaf_inputs), 1)
    for idx, item in enumerate(leaf_inputs):
        ns = int(item["n_source"])
        nt = int(item["n_target"])
        share = float(ns * nt) / float(total_work)
        outputs[int(item["leaf_id"])] = {
            "cost": np.asarray(cost_batch_np[idx, :ns, :nt], dtype=np.float64),
            "cost_build_time": float(batch_dt * share),
            "batch_size": batch_size,
            "device": str(device),
            "backend": "pad_batch_gemm",
        }
    return outputs



def _solve_bilinear_coarsest_problem(
    *,
    source_F_full: np.ndarray,
    target_G_full: np.ndarray,
    source_cost_vec_full: np.ndarray,
    target_cost_vec_full: np.ndarray,
    source_mass_raw: np.ndarray,
    target_mass_raw: np.ndarray,
    source_indices_global: np.ndarray,
    target_indices_global: np.ndarray,
    source_start: Optional[int] = None,
    source_stop: Optional[int] = None,
    target_start: Optional[int] = None,
    target_stop: Optional[int] = None,
    config: SolverRuntimeConfig,
    dual_assignment_pipeline: Literal["auto", "gpu", "cpu"] = "auto",
    node_trace_args: Dict[str, Any],
    node_index: Optional[int],
    node_path: str,
    depth_level: int,
    tracer: Optional[_ChromeTraceCollector] = None,
    dot_scale: float = 1.0,
) -> Tuple[OTWarmStartState | OTWarmStartGPUState, Dict[str, Any]]:
    """
    CN: 求解单个双线性 coarsest 子问题，返回 warm-start state 和 leaf report。
    EN: Solve one bilinear coarsest subproblem and return its warm-start state plus leaf report.
    """
    t_node_start = time.perf_counter()
    src_start = int(source_start) if source_start is not None and int(source_start) >= 0 else None
    src_stop = int(source_stop) if source_stop is not None and int(source_stop) >= 0 else None
    tgt_start = int(target_start) if target_start is not None and int(target_start) >= 0 else None
    tgt_stop = int(target_stop) if target_stop is not None and int(target_stop) >= 0 else None
    if src_start is not None and src_stop is not None:
        # CN: 连续计划节点直接使用 slice；非连续 DFS/asymmetric 节点使用显式全局索引。
        # EN: Contiguous plan nodes use slices directly; non-contiguous DFS/asymmetric nodes use explicit global indices.
        n_source = int(src_stop - src_start)
        source_selector = slice(src_start, src_stop)
    else:
        n_source = int(np.asarray(source_indices_global).size)
        source_selector = np.asarray(source_indices_global, dtype=np.int64)
    if tgt_start is not None and tgt_stop is not None:
        n_target = int(tgt_stop - tgt_start)
        target_selector = slice(tgt_start, tgt_stop)
    else:
        n_target = int(np.asarray(target_indices_global).size)
        target_selector = np.asarray(target_indices_global, dtype=np.int64)
    t_prepare_slice = time.perf_counter()
    with (tracer.span("leaf.prepare_slice", "leaf", args=node_trace_args) if tracer is not None else nullcontext()):
        sub_source_F = np.asarray(source_F_full[source_selector], dtype=np.float32, order="C")
        sub_target_G = np.asarray(target_G_full[target_selector], dtype=np.float32, order="C")
        sub_source_cost = np.asarray(source_cost_vec_full[source_selector], dtype=np.float64, order="C")
        sub_target_cost = np.asarray(target_cost_vec_full[target_selector], dtype=np.float64, order="C")
    prepare_slice_time = float(time.perf_counter() - t_prepare_slice)
    t_prepare_mass = time.perf_counter()
    with (tracer.span("leaf.prepare_mass", "leaf", args=node_trace_args) if tracer is not None else nullcontext()):
        sub_u, sub_v, _ = _normalize_subproblem_masses(source_mass_raw, target_mass_raw)
    prepare_mass_time = float(time.perf_counter() - t_prepare_mass)
    prepare_total_time = prepare_slice_time + prepare_mass_time

    if _config_runtime_log_enabled(config, "warm_start"):
        print(
            f"[Profile][HELLO][D{int(depth_level)}][N{int(node_index) if node_index is not None else '?'}] "
            f"start full_ot path={node_path}, shape=({int(n_source):,}, {int(n_target):,})"
        )
    with (tracer.span("leaf.solve_total", "leaf", args=node_trace_args) if tracer is not None else nullcontext()):
        leaf_inputs = [
            {
                "leaf_id": 0,
                "source_f": sub_source_F,
                "target_g": sub_target_G,
                "source_cost": sub_source_cost,
                "target_cost": sub_target_cost,
                "n_source": int(sub_source_F.shape[0]),
                "n_target": int(sub_target_G.shape[0]),
            }
        ]
        prebuilt = _build_leaf_costs_pad_batch_gemm(
            leaf_inputs=leaf_inputs,
            tracer=tracer,
            depth_level=int(depth_level),
            dot_scale=float(dot_scale),
        )[0]
        sub_result = _solve_lowrank_leaf_with_pot_from_precomputed_cost(
            cost=prebuilt["cost"],
            source_f=sub_source_F,
            target_g=sub_target_G,
            source_mass=sub_u,
            target_mass=sub_v,
            tracer=tracer,
            trace_args=node_trace_args,
            precomputed_cost_build_time=float(prebuilt["cost_build_time"]),
            cost_build_backend=str(prebuilt["backend"]),
            cost_build_batch_size=int(prebuilt["batch_size"]),
            cost_build_device=str(prebuilt["device"]),
        )
    state = sub_result.get("warm_start_state")
    if not isinstance(state, OTWarmStartState):
        raise RuntimeError("coarsest hierarchy solve did not return warm_start_state")
    state = _normalize_dual_assignment_state(
        state,
        pipeline=dual_assignment_pipeline,
    )
    leaf_stage_profile = dict(sub_result.get("leaf_stage_profile") or {})
    # CN: 统一整理 leaf 阶段耗时，便于 BFS/DFS report 使用同一套字段。
    # EN: Normalize leaf-stage timings so BFS/DFS reports use the same fields.
    leaf_stage_profile["prepare_slice_time"] = prepare_slice_time
    leaf_stage_profile["prepare_mass_time"] = prepare_mass_time
    leaf_stage_profile["prepare_total_time"] = prepare_total_time
    total_non_ot_time = (
        prepare_total_time
        + float(leaf_stage_profile.get("pot_import_time", 0.0))
        + float(leaf_stage_profile.get("pot_mass_cast_time", 0.0))
        + float(leaf_stage_profile.get("pot_cost_build_time", 0.0))
        + float(leaf_stage_profile.get("post_total_time", 0.0))
    )
    leaf_total_time = float(time.perf_counter() - t_node_start)
    tracked_total_time = total_non_ot_time + float(sub_result.get("time", 0.0))
    leaf_stage_profile["total_non_ot_time"] = total_non_ot_time
    leaf_stage_profile["total_time"] = leaf_total_time
    leaf_stage_profile["untracked_time"] = max(leaf_total_time - tracked_total_time, 0.0)
    node_report = {
        "path": str(node_path),
        "depth": int(depth_level),
        "kind": "leaf",
        "n_source": int(n_source),
        "n_target": int(n_target),
        "source_mass_sum_raw": float(np.asarray(source_mass_raw, dtype=np.float64).sum()),
        "target_mass_sum_raw": float(np.asarray(target_mass_raw, dtype=np.float64).sum()),
        "support_size": int(_state_support_size(state)),
        "node_build_time_total": leaf_total_time,
        "leaf_stage_profile": leaf_stage_profile,
        "leaf_total_non_ot_time": float(leaf_stage_profile.get("total_non_ot_time", 0.0)),
        "leaf_untracked_time": float(leaf_stage_profile.get("untracked_time", 0.0)),
        "solve_summary": {
            "distance": float(sub_result.get("distance", float("nan"))),
            "time": float(sub_result.get("time", float("nan"))),
            "lp_solve_time_total": float(sub_result.get("lp_solve_time_total", float("nan"))),
            **extract_level_zero_summary(sub_result),
        },
        "children": [],
    }
    return state, node_report
