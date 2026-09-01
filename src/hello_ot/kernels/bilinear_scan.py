from __future__ import annotations

# CN: 本模块包含 HELLO 私有双线性 cost runtime 的 fused-scan 与调度原语。
# EN: This module contains fused-scan and dispatch primitives for HELLO's private bilinear cost runtime.

from dataclasses import replace
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from hello_ot.hierarchy.coarsest import _solve_bilinear_coarsest_problem
from hello_ot.initialization.state import (
    DeferredDualAssignmentCandidates,
    DualAssignmentState,
    _assign_state_from_known_dual_directional_scan,
    _complete_dual_from_candidate,
    _complete_dual_by_ctransform_top1,
    _normalize_dual_assignment_state,
    _materialize_deferred_candidate_union,
    _state_from_gpu,
    _state_to_cpu_public,
    _unique_merge_rows_cols_vals,
)
from hello_ot.hierarchy.utilities import _normalize_subproblem_masses
from hello_ot._internal.trace import _ChromeTraceCollector
from hello_ot.refinement.loop import _refine_lowrank_from_warm_start
from hello_ot.config import SolverRuntimeConfig
from hello_ot.state import GPUWarmStartState as OTWarmStartGPUState, WarmStartState as OTWarmStartState


def solve_leaf_lowrank(
    *,
    source_F_full: np.ndarray,
    target_G_full: np.ndarray,
    source_cost_vec_full: np.ndarray,
    target_cost_vec_full: np.ndarray,
    source_mass_raw: np.ndarray,
    target_mass_raw: np.ndarray,
    source_start: int,
    source_stop: int,
    target_start: int,
    target_stop: int,
    config: SolverRuntimeConfig,
    node_trace_args: Dict[str, Any],
    node_path: str,
    depth_level: int,
    tracer: Optional[_ChromeTraceCollector],
    dot_scale: float = 1.0,
) -> Tuple[DualAssignmentState, Dict[str, Any]]:
    return _solve_bilinear_coarsest_problem(
        source_F_full=source_F_full,
        target_G_full=target_G_full,
        source_cost_vec_full=source_cost_vec_full,
        target_cost_vec_full=target_cost_vec_full,
        source_mass_raw=np.asarray(source_mass_raw[int(source_start) : int(source_stop)], dtype=np.float64, order="C"),
        target_mass_raw=np.asarray(target_mass_raw[int(target_start) : int(target_stop)], dtype=np.float64, order="C"),
        source_indices_global=np.arange(int(source_start), int(source_stop), dtype=np.int64),
        target_indices_global=np.arange(int(target_start), int(target_stop), dtype=np.int64),
        config=config,
        node_trace_args=node_trace_args,
        node_index=None,
        node_path=str(node_path),
        depth_level=int(depth_level),
        tracer=tracer,
        dot_scale=float(dot_scale),
    )


def augment_topk_lowrank(
    *,
    state: DualAssignmentState,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    known_side: Literal["source", "target"],
    assignment_topk: int,
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
    dot_scale: float = 1.0,
) -> Tuple[DualAssignmentState, Dict[str, Any], Dict[str, Any], Any]:
    known_cost_vec = target_cost_vec if str(known_side) == "target" else source_cost_vec
    return _assign_state_from_known_dual_directional_scan(
        state,
        source_F=source_F,
        target_G=target_G,
        known_cost_vec=known_cost_vec,
        assignment_topk=int(assignment_topk),
        known_side=str(known_side),
        include_scaffold=True,
        dual_assignment_pipeline="gpu",
        return_completion_candidate=True,
        tracer=tracer,
        trace_args=trace_args,
        dot_scale=float(dot_scale),
    )


def scan_topk_lowrank_candidates(
    *,
    state: DualAssignmentState,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    known_side: Literal["source", "target"],
    assignment_topk: int,
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
    dot_scale: float = 1.0,
) -> Tuple[
    DualAssignmentState,
    Dict[str, Any],
    Dict[str, Any],
    Any,
    DeferredDualAssignmentCandidates,
]:
    """
    CN: 执行一次 initialization directional scan，并将候选暂存 CPU 而不构造 GPU support。
    EN: Run one initialization directional scan and stage candidates on CPU without building GPU support.
    """
    known_cost_vec = target_cost_vec if str(known_side) == "target" else source_cost_vec
    result = _assign_state_from_known_dual_directional_scan(
        state,
        source_F=source_F,
        target_G=target_G,
        known_cost_vec=known_cost_vec,
        assignment_topk=int(assignment_topk),
        known_side=str(known_side),
        include_scaffold=True,
        dual_assignment_pipeline="gpu",
        return_completion_candidate=True,
        defer_state_build=True,
        tracer=tracer,
        trace_args=trace_args,
        dot_scale=float(dot_scale),
    )
    return result


def materialize_lowrank_candidate_union(
    *,
    state: DualAssignmentState,
    candidates: Tuple[DeferredDualAssignmentCandidates, DeferredDualAssignmentCandidates],
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
) -> Tuple[OTWarmStartGPUState, Dict[str, Any]]:
    """
    CN: 在两遍 scan 后用一次 GPU unique 构造最终 warm-start support。
    EN: Build final warm-start support with one GPU unique after both scans.
    """
    return _materialize_deferred_candidate_union(
        state,
        candidates,
        tracer=tracer,
        trace_args=trace_args,
    )


def complete_dual_lowrank(
    *,
    state: DualAssignmentState,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    known_side: Literal["source", "target"],
    completion_candidate: Any,
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
    trace_prefix: str,
    dot_scale: float = 1.0,
) -> Tuple[DualAssignmentState, Dict[str, Any]]:
    if completion_candidate is not None:
        return _complete_dual_from_candidate(
            state,
            source_cost_vec=source_cost_vec,
            target_cost_vec=target_cost_vec,
            known_side=str(known_side),
            seed=completion_candidate,
            dual_assignment_pipeline="gpu",
            tracer=tracer,
            trace_args=trace_args,
            trace_prefix=str(trace_prefix),
        )
    return _complete_dual_by_ctransform_top1(
        state,
        source_F=source_F,
        target_G=target_G,
        source_cost_vec=source_cost_vec,
        target_cost_vec=target_cost_vec,
        known_side=str(known_side),
        dual_assignment_pipeline="gpu",
        tracer=tracer,
        trace_args=trace_args,
        trace_prefix=str(trace_prefix),
        dot_scale=float(dot_scale),
    )


def refine_node_lowrank(
    *,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    scope: Literal["internal", "root"],
    warm_start: OTWarmStartState | OTWarmStartGPUState,
    config: SolverRuntimeConfig,
    skip_initial_pricing: bool,
    dual_feasibility_tol: float,
    dual_feasibility_norm: Literal["l2", "linf"] = "l2",
    lp_termination_norm: Literal["l2", "linf"] = "l2",
    tracer: Optional[_ChromeTraceCollector],
    trace_prefix: str,
    pricing_index_pool: Optional[Any],
    warm_start_cost_vec_chunk_size: int = 65536,
    warm_start_cost_vec_feature_chunk_size: Optional[int] = None,
    warm_start_profile_depth: int = 0,
    dot_scale: float = 1.0,
) -> Tuple[DualAssignmentState, Dict[str, Any]]:
    sub_source_mass, sub_target_mass, _ = _normalize_subproblem_masses(source_mass, target_mass)
    solve_cfg = replace(config)

    solve_cfg.pricing_strategy = "nodewise_full"
    solve_cfg.convergence_criterion = "dual_feasibility"
    solve_cfg.require_dual_feasibility_convergence = False
    solve_cfg.dual_feasibility_tol = float(dual_feasibility_tol)
    solve_cfg.lp_termination_norm = str(lp_termination_norm)
    solve_cfg.require_added_convergence = False
    solve_cfg.validate()

    solve_cfg.cost_type = "lowrank"
    solve_cfg.dot_scale = float(dot_scale)
    solve_cfg.validate()

    result = _refine_lowrank_from_warm_start(
        source_F=source_F,
        target_G=target_G,
        source_cost_vec=source_cost_vec,
        target_cost_vec=target_cost_vec,
        source_mass=sub_source_mass,
        target_mass=sub_target_mass,
        log=True,
        return_coupling=True,
        return_state=True,
        config=solve_cfg,
        warm_start=warm_start,
        skip_initial_pricing=bool(skip_initial_pricing),
        use_lp_warm_start_dual=True,
        trace_collector=tracer,
        trace_prefix=str(trace_prefix),
        warm_start_profile_depth=int(warm_start_profile_depth),
        pricing_index_pool=pricing_index_pool,
        warm_start_cost_vec_chunk_size=int(warm_start_cost_vec_chunk_size),
        warm_start_cost_vec_feature_chunk_size=warm_start_cost_vec_feature_chunk_size,
        _consume_warm_start_gpu_state=True,
        dual_feasibility_norm=dual_feasibility_norm,
        dot_scale=float(dot_scale),
    )
    if not isinstance(result, dict):
        raise RuntimeError("HELLO refinement expected a structured result dictionary.")
    state = result.get("warm_start_state")
    if not isinstance(state, OTWarmStartState):
        raise RuntimeError("hello refine did not return warm_start_state.")
    normalized_state = (
        state
        if str(getattr(config, "backend", "native")) == "torch"
        else _normalize_dual_assignment_state(state, pipeline="gpu")
    )
    result["warm_start_state"] = normalized_state
    return normalized_state, result
