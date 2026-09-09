from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from hello_ot.cost import (
    HelloCostContext,
    assign_from_complete_dual,
    complete_dual,
    materialize_deferred_assignment_union,
    propagate_and_assign_dual,
    begin_refinement_by_cost,
    finalize_refinement_by_cost,
    scan_deferred_assignment_candidates,
    solve_leaf_by_cost,
)
from hello_ot.refinement.iterations import check_optimality, solve_lp, update_support
from hello_ot.perturbation import switch_level_to_perturbed_cost as _switch_level_to_perturbed_cost
from hello_ot.hierarchy.construction import (
    HierarchyNodeRange as _HierarchyNodeRange,
    first_child_range as _first_child_range,
    slice_node_arrays as _slice_node_arrays,
)
from hello_ot.hierarchy.plan import (
    _hierarchy_global_reorder,
    _make_shuffle_once_permutation,
    _map_warm_start_state_to_original_order,
)
from hello_ot.initialization.state import (
    _ensure_lowrank_final_dual_feasible,
    _normalize_dual_assignment_state,
    _state_support_size,
    _stitch_hierarchy_states,
)
from hello_ot._internal.trace import _ChromeTraceCollector
from hello_ot._internal.runtime_context import record_solve_event
from hello_ot.restricted_ot.runtime import extract_level_zero_summary
from hello_ot.config import (
    SolverRuntimeConfig,
    StoppingNormPolicy,
    finest_stopping_norm,
    level_stopping_norm,
)
from hello_ot.state import (
    GPUWarmStartState as OTWarmStartGPUState,
    WarmStartState as OTWarmStartState,
)

def _dual_only_stitched_state(
    state: OTWarmStartState | OTWarmStartGPUState,
) -> OTWarmStartState | OTWarmStartGPUState:
    """
    CN: 保留 stitched dual 与问题元数据，同时移除从 coarse child 继承的 support scaffold。
    EN: Preserve stitched duals and problem metadata while dropping the support scaffold inherited from the coarse child.
    """
    if isinstance(state, OTWarmStartGPUState):
        device = state.rows.device
        return OTWarmStartGPUState(
            rows=torch.empty(0, dtype=torch.int32, device=device),
            cols=torch.empty(0, dtype=torch.int32, device=device),
            x_prev=torch.empty(0, dtype=torch.float64, device=device),
            dual_uv=state.dual_uv,
            n_source=int(state.n_source),
            n_target=int(state.n_target),
            device=str(device),
            keys=torch.empty(0, dtype=torch.int64, device=device),
            northwest_positions=None,
        )
    return OTWarmStartState(
        rows=np.empty(0, dtype=np.int32),
        cols=np.empty(0, dtype=np.int32),
        x_prev=np.empty(0, dtype=np.float64),
        dual_uv=state.dual_uv,
        n_source=int(state.n_source),
        n_target=int(state.n_target),
        northwest_positions=None,
    )


def _map_sparse_coupling_to_original_order(
    coupling: Optional[sp.spmatrix],
    *,
    source_perm: np.ndarray,
    target_perm: np.ndarray,
    shape: Tuple[int, int],
) -> Optional[sp.coo_matrix]:
    if coupling is None:
        return None
    coo = coupling.tocoo(copy=False)
    rows = np.asarray(np.asarray(source_perm, dtype=np.int64)[np.asarray(coo.row, dtype=np.int64)], dtype=np.int32)
    cols = np.asarray(np.asarray(target_perm, dtype=np.int64)[np.asarray(coo.col, dtype=np.int64)], dtype=np.int32)
    order = np.argsort(rows.astype(np.int64) * np.int64(int(shape[1])) + cols.astype(np.int64), kind="stable")
    return sp.coo_matrix(
        (
            np.asarray(coo.data, dtype=np.float64)[order],
            (np.asarray(rows[order], dtype=np.int32), np.asarray(cols[order], dtype=np.int32)),
        ),
        shape=(int(shape[0]), int(shape[1])),
        dtype=np.float64,
    )


def _map_dual_to_original_order(
    dual_uv: Any,
    *,
    source_perm: np.ndarray,
    target_perm: np.ndarray,
) -> Any:
    if dual_uv is None:
        return None
    dual = np.asarray(dual_uv, dtype=np.float64)
    n_source = int(np.asarray(source_perm).size)
    n_target = int(np.asarray(target_perm).size)
    if int(dual.size) != n_source + n_target:
        return dual_uv
    out = np.empty_like(dual)
    out[np.asarray(source_perm, dtype=np.int64)] = dual[:n_source]
    out[n_source + np.asarray(target_perm, dtype=np.int64)] = dual[n_source:]
    return out


def _standard_output_from_log_result(
    result: Dict[str, Any],
    *,
    log: bool,
    return_coupling: bool,
    return_state: bool,
) -> Any:
    if bool(log):
        if not bool(return_coupling):
            result["sparse_coupling"] = None
        return result
    distance = float(result.get("distance", float("nan")))
    coupling = result.get("sparse_coupling")
    state = result.get("warm_start_state")
    if bool(return_coupling) and bool(return_state):
        return distance, coupling, state
    if bool(return_coupling):
        return distance, coupling
    if bool(return_state):
        return distance, state
    return distance


def _format_profile_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        if not np.isfinite(parsed):
            return "-"
        return f"{parsed:.6g}"
    return str(value)


def _print_augment_profile_before_refine(
    *,
    depth: int,
    assignment_topk: int,
    support_after_forward: int,
    support_after_reverse: Optional[int],
) -> None:
    print(
        f"[Profile][Augment&Completion][D{int(depth)}] "
        "assignment=nodewise_bidirectional "
        f"assignment_topk={_format_profile_value(assignment_topk)} "
        f"support_after_forward={_format_profile_value(support_after_forward)} "
        f"support_after_reverse={_format_profile_value(support_after_reverse)}"
    )


@dataclass
class _DualAssignmentResult:
    state: OTWarmStartState | OTWarmStartGPUState
    topk_stats: Dict[str, Any]
    augment_profile: Dict[str, Any]
    dual_completion_profile: Dict[str, Any]
    forward_topk_stats: Dict[str, Any]
    support_after_forward: int
    reverse_topk_stats: Dict[str, Any]
    support_after_reverse: Optional[int]


@dataclass
class _HierarchyExecution:
    """
    CN: hierarchy 算法之外准备好的数据与运行设施，避免其污染论文级控制流。
    EN: Data and runtime facilities prepared outside the hierarchy algorithm so they do not pollute its paper-level control flow.
    """

    cost_context: HelloCostContext
    source_F: np.ndarray
    target_G: np.ndarray
    source_cost: np.ndarray
    target_cost: np.ndarray
    source_mass: np.ndarray
    target_mass: np.ndarray
    config: SolverRuntimeConfig
    assignment_topk: int
    split_count: int
    dual_feasibility_tol: float
    stopping_norm: StoppingNormPolicy
    tracer: Optional[_ChromeTraceCollector]
    pricing_index_pool: Optional[Any]
    memory_recorder: Optional[Any]
    log: bool
    perturbation: Any = None

    def context_for(self, node: _HierarchyNodeRange) -> HelloCostContext:
        """
        CN: 获取当前层成本上下文，保留扰动的全局边标识。
        EN: Get this level's cost context while preserving global perturbation identities.
        """
        return self.cost_context.sliced(
            slice(node.source_start, node.source_stop),
            slice(node.target_start, node.target_stop),
        )

    def new_reentry_detector(self) -> Any:
        """
        CN: 为当前原成本层建立独立检测器。
        EN: Create an independent detector for the current original-cost level.
        """
        return None if self.perturbation is None else self.perturbation.new_detector()

    def stopping_norm_for(self, node: _HierarchyNodeRange) -> Literal["l2", "linf"]:
        """
        CN: 粗层固定使用 L2；finest_linf 只将最精细层切换到 L-infinity。
        EN: Keep coarse levels on L2; finest_linf switches only the finest level to L-infinity.
        """
        return level_stopping_norm(self.stopping_norm, level_index=int(node.depth))


@dataclass
class _InitializedHierarchyLevel:
    node: _HierarchyNodeRange
    arrays: Dict[str, np.ndarray]
    state: OTWarmStartState | OTWarmStartGPUState
    split_axis: str
    trace_args: Dict[str, Any]
    started_at: float
    initialization_elapsed: float
    support_before_solve: int
    stitch_profile: Dict[str, Any]
    assignment: _DualAssignmentResult
    refinement: Any
    refine_span: Any


def _dual_assign_stitched_node(
    *,
    node: _HierarchyNodeRange,
    child_state: OTWarmStartState | OTWarmStartGPUState,
    stitched_state: OTWarmStartState | OTWarmStartGPUState,
    arrays: Dict[str, np.ndarray],
    cost_context: HelloCostContext,
    known_side: Literal["source", "target"],
    assignment_topk: int,
    split_count: int,
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
    memory_recorder: Optional[Any],
    backend: str,
    torch_device: str,
) -> _DualAssignmentResult:
    """
    CN: 对 stitched scaffold 执行 forward augment、dual completion，并按需执行 reverse augment。
    EN: Run forward augmentation, dual completion, and optional reverse augmentation on a stitched scaffold.
    """
    completion_candidate = None
    forward_candidates = None
    # CN: forward pass 在完整对侧上执行 nodewise top-k dual assignment。
    # EN: The forward pass runs nodewise top-k dual assignment over the complete opposite side.
    augment_span = memory_recorder.begin_span(phase="augment", node=node, state=stitched_state) if memory_recorder is not None else None
    with (tracer.span("hello.initialization.dual_assignment", "hierarchy", args={**trace_args, "augment_pass": "forward"}) if tracer is not None else nullcontext()):
        if cost_context.score_family == "inner_product" and str(backend) == "native":
            parent_state, topk_stats, augment_profile, completion_candidate, forward_candidates = (
                scan_deferred_assignment_candidates(
                    cost_context=cost_context,
                    state=stitched_state,
                    source_F=arrays["source_F"],
                    target_G=arrays["target_G"],
                    source_cost_vec=arrays["source_cost_vec"],
                    target_cost_vec=arrays["target_cost_vec"],
                    known_side=str(known_side),
                    assignment_topk=int(assignment_topk),
                    tracer=tracer,
                    trace_args=trace_args,
                )
            )
        else:
            parent_state, topk_stats, augment_profile, completion_candidate = propagate_and_assign_dual(
                cost_context=cost_context,
                state=stitched_state,
                source_F=arrays["source_F"],
                target_G=arrays["target_G"],
                source_cost_vec=arrays["source_cost_vec"],
                target_cost_vec=arrays["target_cost_vec"],
                known_side=str(known_side),
                assignment_topk=int(assignment_topk),
                tracer=tracer,
                trace_args=trace_args,
                backend=str(backend),
                torch_device=str(torch_device),
            )
        if memory_recorder is not None:
            memory_recorder.end_span(augment_span, state=parent_state)
            memory_recorder.record(phase="after_augment", node=node, state=parent_state)
    forward_topk_stats = dict(topk_stats)
    support_after_forward = int(_state_support_size(parent_state))
    # CN: reverse 相关变量先按 single-direction 的空结果初始化，便于后面的日志和 levels 统一读取。
    # EN: Initialize reverse-related values as empty single-direction results so logging and levels can read them uniformly.
    reverse_topk_stats = {"cross_transfer_added_raw": 0, "cross_transfer_added_unique": 0}
    support_after_reverse: Optional[int] = None
    reverse_augment_profile: Optional[Dict[str, Any]] = None
    reverse_candidates = None

    # CN: augment 后只有一侧 dual 是已知的， completion 用 c-transform 补齐另一侧 dual。
    # EN: After augmentation only one side of the dual is known; dual completion fills the other side via c-transform.
    dual_completion_span = memory_recorder.begin_span(phase="dual_completion", node=node, state=parent_state) if memory_recorder is not None else None
    parent_state, dual_completion_profile = complete_dual(
        cost_context=cost_context,
        state=parent_state,
        source_F=arrays["source_F"],
        target_G=arrays["target_G"],
        source_cost_vec=arrays["source_cost_vec"],
        target_cost_vec=arrays["target_cost_vec"],
        known_side=str(known_side),
        completion_candidate=completion_candidate,
        tracer=tracer,
        trace_args=trace_args,
        trace_prefix="hello.initialization.dual_completion",
        backend=str(backend),
        torch_device=str(torch_device),
    )
    if memory_recorder is not None:
        memory_recorder.end_span(dual_completion_span, state=parent_state)
        memory_recorder.record(phase="after_dual_completion", node=node, state=parent_state)

    # CN: dual 补齐后，从相反已知侧再做一次 directional assignment，取两次 top-k support 的并集。
    # EN: After dual completion, run one directional assignment from the opposite known side and union the two top-k supports.
    reverse_known_side = "source" if str(known_side) == "target" else "target"
    reverse_augment_span = memory_recorder.begin_span(phase="augment_reverse", node=node, state=parent_state) if memory_recorder is not None else None
    with (tracer.span("hello.initialization.dual_assignment", "hierarchy", args={**trace_args, "augment_pass": "reverse"}) if tracer is not None else nullcontext()):
        # CN: reverse pass 只用于补充候选边；dual 已在 forward pass 后补齐，所以返回的 completion seed 不再使用。
        # EN: The reverse pass only adds candidate edges; duals were completed after the forward pass, so its completion seed is intentionally unused.
        if forward_candidates is not None:
            parent_state, reverse_topk_stats, reverse_augment_profile, _reverse_completion_candidate, reverse_candidates = (
                scan_deferred_assignment_candidates(
                    cost_context=cost_context,
                    state=parent_state,
                    source_F=arrays["source_F"],
                    target_G=arrays["target_G"],
                    source_cost_vec=arrays["source_cost_vec"],
                    target_cost_vec=arrays["target_cost_vec"],
                    known_side=reverse_known_side,
                    assignment_topk=int(assignment_topk),
                    tracer=tracer,
                    trace_args=trace_args,
                )
            )
            parent_state, deferred_union_profile = materialize_deferred_assignment_union(
                cost_context=cost_context,
                state=parent_state,
                candidates=(forward_candidates, reverse_candidates),
                tracer=tracer,
                trace_args=trace_args,
            )
            support_after_forward = int(deferred_union_profile["support_after_forward"])
            support_after_reverse = int(deferred_union_profile["support_after_reverse"])
            forward_topk_stats["cross_transfer_added_unique"] = int(
                max(0, support_after_forward - int(_state_support_size(stitched_state)))
            )
            reverse_topk_stats["cross_transfer_added_unique"] = int(
                max(0, support_after_reverse - support_after_forward)
            )
            reverse_augment_profile = {
                **dict(reverse_augment_profile),
                "deferred_union": dict(deferred_union_profile),
                "merge_unique_time": float(deferred_union_profile.get("merge_unique_time", 0.0)),
                "state_build_time": float(deferred_union_profile.get("state_build_time", 0.0)),
                "total_time": float(reverse_augment_profile.get("total_time", 0.0))
                + float(deferred_union_profile.get("merge_unique_time", 0.0))
                + float(deferred_union_profile.get("state_build_time", 0.0)),
            }
        else:
            parent_state, reverse_topk_stats, reverse_augment_profile, _reverse_completion_candidate = assign_from_complete_dual(
                cost_context=cost_context,
                state=parent_state,
                source_F=arrays["source_F"],
                target_G=arrays["target_G"],
                source_cost_vec=arrays["source_cost_vec"],
                target_cost_vec=arrays["target_cost_vec"],
                known_side=reverse_known_side,
                assignment_topk=int(assignment_topk),
                tracer=tracer,
                trace_args=trace_args,
                backend=str(backend),
                torch_device=str(torch_device),
            )
            support_after_reverse = int(_state_support_size(parent_state))
    if memory_recorder is not None:
        memory_recorder.end_span(reverse_augment_span, state=parent_state)
        memory_recorder.record(phase="after_reverse_augment", node=node, state=parent_state)
    # CN: 对外诊断统计合并 forward/reverse；support size 使用 reverse 后的状态大小，即并集大小。
    # EN: Merge forward/reverse diagnostics; the support size after reverse is the union size.
    topk_stats = {
        "cross_transfer_added_raw": int(forward_topk_stats.get("cross_transfer_added_raw", 0))
        + int(reverse_topk_stats.get("cross_transfer_added_raw", 0)),
        "cross_transfer_added_unique": int(support_after_reverse),
    }
    augment_profile = {
        **dict(augment_profile),
        "total_time": float(augment_profile.get("total_time", 0.0))
        + float(reverse_augment_profile.get("total_time", 0.0)),
        "forward": dict(augment_profile),
        "reverse": dict(reverse_augment_profile),
    }

    return _DualAssignmentResult(
        state=parent_state,
        topk_stats=dict(topk_stats),
        augment_profile=dict(augment_profile),
        dual_completion_profile=dict(dual_completion_profile),
        forward_topk_stats=forward_topk_stats,
        support_after_forward=int(support_after_forward),
        reverse_topk_stats=dict(reverse_topk_stats),
        support_after_reverse=support_after_reverse,
    )


def _initialize_hierarchy_level(
    *,
    node: _HierarchyNodeRange,
    child_state: OTWarmStartState | OTWarmStartGPUState,
    execution: _HierarchyExecution,
) -> _InitializedHierarchyLevel:
    """
    CN: 执行论文的层间 initialization：继承 dual、c-transform completion、双向 dual assignment 与可行支撑构造。
    EN: Run paper-level initialization: dual inheritance, c-transform completion, bidirectional dual assignment, and feasible-support construction.
    """
    started_at = time.perf_counter()
    cost_context = execution.context_for(node)
    config = execution.config
    tracer = execution.tracer
    memory_recorder = execution.memory_recorder
    trace_args = {
        "path": str(node.path),
        "depth": int(node.depth),
        "depth_remaining": int(node.depth_remaining),
        "n_source": int(node.n_source),
        "n_target": int(node.n_target),
        "algorithm": "hello",
        "kind": "node",
    }
    split_axis = "source" if node.n_source >= node.n_target else "target"
    _, chosen_partition = _first_child_range(
        node,
        split_axis=split_axis,
        split_count=int(execution.split_count),
    )
    if split_axis == "source":
        source_partitions = [np.asarray(chosen_partition, dtype=np.int64)]
        target_partitions = [np.arange(int(node.n_target), dtype=np.int64)]
        known_side = "target"
    else:
        source_partitions = [np.arange(int(node.n_source), dtype=np.int64)]
        target_partitions = [np.asarray(chosen_partition, dtype=np.int64)]
        known_side = "source"

    stitch_span = (
        memory_recorder.begin_span(phase="stitch", node=node, state=child_state)
        if memory_recorder is not None
        else None
    )
    stitched_state, stitch_profile = _stitch_hierarchy_states(
        [child_state],
        source_partitions,
        target_partitions,
        [1.0],
        n_source=int(node.n_source),
        n_target=int(node.n_target),
        use_primal_values=False,
        tracer=tracer,
        trace_args=trace_args,
    )
    stitched_state = _dual_only_stitched_state(stitched_state)
    if memory_recorder is not None:
        memory_recorder.end_span(stitch_span, state=stitched_state)

    arrays = _slice_node_arrays(
        node=node,
        source_F_full=execution.source_F,
        target_G_full=execution.target_G,
        source_cost_vec_full=execution.source_cost,
        target_cost_vec_full=execution.target_cost,
        source_mass_raw=execution.source_mass,
        target_mass_raw=execution.target_mass,
    )
    assignment = _dual_assign_stitched_node(
        node=node,
        child_state=child_state,
        stitched_state=stitched_state,
        arrays=arrays,
        cost_context=cost_context,
        known_side=known_side,
        assignment_topk=int(execution.assignment_topk),
        split_count=int(execution.split_count),
        tracer=tracer,
        trace_args=trace_args,
        memory_recorder=memory_recorder,
        backend=str(config.backend),
        torch_device=str(config.torch_device),
    )
    del stitched_state
    state = assignment.state
    support_before_solve = int(_state_support_size(state))
    record_solve_event(
        "initialize_level",
        path=str(node.path),
        rows=state.rows,
        cols=state.cols,
        primal=state.x_prev,
        dual=state.dual_uv,
    )
    if execution.log:
        _print_augment_profile_before_refine(
            depth=int(node.depth),
            assignment_topk=int(execution.assignment_topk),
            support_after_forward=int(assignment.support_after_forward),
            support_after_reverse=assignment.support_after_reverse,
        )
    refine_span = (
        memory_recorder.begin_span(phase="refine", node=node, state=state)
        if memory_recorder is not None
        else None
    )
    level_stopping_norm = execution.stopping_norm_for(node)
    refinement = begin_refinement_by_cost(
        cost_context=cost_context,
        source_F=arrays["source_F"],
        target_G=arrays["target_G"],
        source_cost_vec=arrays["source_cost_vec"],
        target_cost_vec=arrays["target_cost_vec"],
        source_mass=arrays["source_mass"],
        target_mass=arrays["target_mass"],
        warm_start=state,
        config=config,
        skip_initial_pricing=(cost_context.score_family == "inner_product"),
        dual_feasibility_tol=float(execution.dual_feasibility_tol),
        dual_feasibility_norm=level_stopping_norm,
        lp_termination_norm=level_stopping_norm,
        tracer=tracer,
        trace_prefix="hello.refinement",
        pricing_index_pool=execution.pricing_index_pool,
        warm_start_profile_depth=int(node.depth),
    )
    initialization_elapsed = float(time.perf_counter() - started_at)
    return _InitializedHierarchyLevel(
        node=node,
        arrays=arrays,
        state=state,
        split_axis=split_axis,
        trace_args=trace_args,
        started_at=started_at,
        initialization_elapsed=initialization_elapsed,
        support_before_solve=support_before_solve,
        stitch_profile=dict(stitch_profile),
        assignment=assignment,
        refinement=refinement,
        refine_span=refine_span,
    )


def _finalize_hierarchy_level(
    initialized: _InitializedHierarchyLevel,
    *,
    levels: List[Dict[str, Any]],
    execution: _HierarchyExecution,
) -> Tuple[OTWarmStartState | OTWarmStartGPUState, Dict[str, Any]]:
    """
    CN: 完成当前层并封装诊断；算法迭代不隐藏在此函数中。
    EN: Finalize the current level and package diagnostics; no algorithm iteration is hidden here.
    """
    node = initialized.node
    memory_recorder = execution.memory_recorder
    final_state, node_result = finalize_refinement_by_cost(initialized.refinement)
    record_solve_event(
        "refine_level",
        path=str(node.path),
        rows=final_state.rows,
        cols=final_state.cols,
        primal=final_state.x_prev,
        dual=final_state.dual_uv,
    )
    if memory_recorder is not None:
        memory_recorder.end_span(initialized.refine_span, state=final_state)

    level_summary = extract_level_zero_summary(node_result)
    solve_summary = {
        "distance": float(node_result.get("distance", float("nan"))),
        "time": float(node_result.get("time", 0.0)),
        "lp_solve_time_total": float(node_result.get("lp_solve_time_total", 0.0)),
        "full_scan_time": float(
            sum(
                float(record.get("full_scan_time", 0.0) or 0.0)
                for record in node_result.get("warm_start_iteration_records", ())
            )
        ),
        "lp_backend_peak_mem_mib": node_result.get("lp_backend_peak_mem_mib"),
        "pricing_peak_mem_mib": node_result.get("pricing_peak_mem_mib"),
        "dual_feas_peak_mem_mib": node_result.get("dual_feas_peak_mem_mib"),
        "objective_trace": list(node_result.get("warm_start_iteration_records") or []),
        "initialization": dict(node_result.get("warm_start_init") or {}),
        **level_summary,
    }
    assignment = initialized.assignment
    support_after_solve = int(_state_support_size(final_state))
    levels.append(
        {
            "path": str(node.path),
            "depth": int(node.depth),
            "kind": "node",
            "split_axis": initialized.split_axis,
            "selected_child_index": 0,
            "split_count": int(execution.split_count),
            "augment_direction": "bidirectional",
            "n_source": int(node.n_source),
            "n_target": int(node.n_target),
            "source_range": [int(node.source_start), int(node.source_stop)],
            "target_range": [int(node.target_start), int(node.target_stop)],
            "support_before_solve": int(initialized.support_before_solve),
            "support_after_solve": support_after_solve,
            "support_size": support_after_solve,
            "initial_active_support_size": int(initialized.support_before_solve),
            "topk_added_raw": int(assignment.topk_stats.get("cross_transfer_added_raw", 0)),
            "topk_added_unique": int(assignment.topk_stats.get("cross_transfer_added_unique", 0)),
            "topk_forward_added_raw": int(assignment.forward_topk_stats.get("cross_transfer_added_raw", 0)),
            "topk_forward_support_size": int(assignment.support_after_forward),
            "topk_reverse_added_raw": int(assignment.reverse_topk_stats.get("cross_transfer_added_raw", 0)),
            "topk_reverse_added_unique": (
                0
                if assignment.support_after_reverse is None
                else max(0, int(assignment.support_after_reverse) - int(assignment.support_after_forward))
            ),
            "topk_reverse_support_size": assignment.support_after_reverse,
            "topk_bidirectional_union_support_size": assignment.support_after_reverse,
            "local_build_time": float(
                initialized.stitch_profile.get("total_time", 0.0)
                + assignment.augment_profile.get("total_time", 0.0)
                + assignment.dual_completion_profile.get("total_time", 0.0)
            ),
            "initialization_time": float(initialized.initialization_elapsed),
            "time": float(time.perf_counter() - initialized.started_at),
            "stitch_profile": initialized.stitch_profile,
            "augment_profile": assignment.augment_profile,
            "dual_completion_profile": assignment.dual_completion_profile,
            "solve_summary": solve_summary,
        }
    )
    from .progress import report_level_done

    report_level_done(
        execution,
        node,
        node_result,
        active=support_after_solve,
        initialization_elapsed=initialized.initialization_elapsed,
        elapsed=time.perf_counter() - initialized.started_at,
    )
    return final_state, node_result


def _solve_coarsest_level(
    *,
    node: _HierarchyNodeRange,
    execution: _HierarchyExecution,
) -> Tuple[OTWarmStartState | OTWarmStartGPUState, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    CN: 直接求解 hierarchy 的 coarsest 层。
    EN: Solve the coarsest hierarchy level directly.
    """
    from .progress import report_coarsest_done, report_level_start

    report_level_start(execution, node, coarsest=True)
    t_node = time.perf_counter()
    cost_context = execution.context_for(node)
    config = execution.config
    tracer = execution.tracer
    memory_recorder = execution.memory_recorder
    trace_args = {
        "path": str(node.path),
        "depth": int(node.depth),
        "depth_remaining": int(node.depth_remaining),
        "n_source": int(node.n_source),
        "n_target": int(node.n_target),
        "algorithm": "hello",
        "kind": "leaf" if int(node.depth_remaining) <= 0 else "node",
    }
    if memory_recorder is not None:
        memory_recorder.record(phase="node_enter", node=node)
    with (tracer.span("hello.level", "hierarchy", args=trace_args) if tracer is not None else nullcontext()):
        if int(node.depth_remaining) <= 0:
            leaf_span = memory_recorder.begin_span(phase="leaf_solve", node=node) if memory_recorder is not None else None
            leaf_state, leaf_report = solve_leaf_by_cost(
                cost_context=cost_context,
                source_F_full=execution.source_F,
                target_G_full=execution.target_G,
                source_cost_vec_full=execution.source_cost,
                target_cost_vec_full=execution.target_cost,
                source_mass_raw=execution.source_mass,
                target_mass_raw=execution.target_mass,
                source_start=int(node.source_start),
                source_stop=int(node.source_stop),
                target_start=int(node.target_start),
                target_stop=int(node.target_stop),
                config=config,
                node_trace_args=trace_args,
                node_path=str(node.path),
                depth_level=int(node.depth),
                tracer=tracer,
            )
            normalized_leaf_state = _normalize_dual_assignment_state(
                leaf_state,
                pipeline="gpu" if str(config.backend) == "native" else "cpu",
            )
            record_solve_event(
                "initialize_coarsest",
                path=str(node.path),
                rows=normalized_leaf_state.rows,
                cols=normalized_leaf_state.cols,
                primal=normalized_leaf_state.x_prev,
                dual=normalized_leaf_state.dual_uv,
            )
            if memory_recorder is not None:
                memory_recorder.end_span(leaf_span, state=normalized_leaf_state)
            level = {
                "path": str(node.path),
                "depth": int(node.depth),
                "kind": "leaf",
                "n_source": int(node.n_source),
                "n_target": int(node.n_target),
                "support_size": int(_state_support_size(normalized_leaf_state)),
                "time": float(leaf_report.get("node_build_time_total", 0.0)) if isinstance(leaf_report, dict) else float(time.perf_counter() - t_node),
                "lp_solve_time_total": _leaf_report_lp_time(leaf_report),
                "solve_summary": dict(leaf_report.get("solve_summary") or {}),
            }
            if memory_recorder is not None:
                memory_recorder.record(phase="node_exit", node=node, state=normalized_leaf_state)
            root_result = None
            if int(node.depth) == 0:
                # CN: 当整个问题已位于 coarsest threshold 内时，leaf exact solve 就是正式最终结果。
                # EN: When the full problem is within the coarsest threshold, the exact leaf solve is the formal final result.
                state_cpu = _normalize_dual_assignment_state(normalized_leaf_state, pipeline="cpu")
                leaf_solve = dict(leaf_report.get("solve_summary") or {})
                root_result = {
                    "distance": float(leaf_solve.get("distance", float("nan"))),
                    "time": float(leaf_solve.get("time", 0.0) or 0.0),
                    "lp_solve_time_total": float(
                        leaf_solve.get("lp_solve_time_total", 0.0) or 0.0
                    ),
                    "level_summaries": [
                        {
                            "level": 0,
                            "n_source": int(node.n_source),
                            "n_target": int(node.n_target),
                            "iters": int(leaf_solve.get("level0_inner_iterations", 1) or 1),
                            "time": float(leaf_solve.get("time", 0.0) or 0.0),
                            "objective": float(leaf_solve.get("distance", float("nan"))),
                            "lp_time": float(
                                leaf_solve.get("lp_solve_time_total", 0.0) or 0.0
                            ),
                            "pricing_time": 0.0,
                            "support_final": int(_state_support_size(state_cpu)),
                            "converged": True,
                        }
                    ],
                }
                root_result["sparse_coupling"] = sp.coo_matrix(
                    (
                        np.asarray(state_cpu.x_prev, dtype=np.float64),
                        (
                            np.asarray(state_cpu.rows, dtype=np.int32),
                            np.asarray(state_cpu.cols, dtype=np.int32),
                        ),
                    ),
                    shape=(int(node.n_source), int(node.n_target)),
                    dtype=np.float64,
                )
                root_result["dual_source"] = np.asarray(state_cpu.dual_uv, dtype=np.float64)
                root_result["warm_start_state"] = state_cpu
            report_coarsest_done(execution, node, level)
            return normalized_leaf_state, [level], root_result
        raise ValueError("_solve_coarsest_level requires a coarsest hierarchy node")

def _solve_hierarchy(
    *,
    root: _HierarchyNodeRange,
    execution: _HierarchyExecution,
) -> Tuple[OTWarmStartState | OTWarmStartGPUState, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    CN: 按论文顺序执行 HELLO：构造 hierarchy，在 coarsest 层求解，再逐层 initialization 与 refinement。
    EN: Execute HELLO in paper order: build the hierarchy, solve the coarsest level, then initialize and refine each finer level.
    """
    hierarchy = [root]
    while hierarchy[-1].depth_remaining > 0:
        parent = hierarchy[-1]
        split_axis = "source" if parent.n_source >= parent.n_target else "target"
        child, _ = _first_child_range(
            parent,
            split_axis=split_axis,
            split_count=int(execution.split_count),
        )
        hierarchy.append(child)

    state, levels, result = _solve_coarsest_level(
        node=hierarchy[-1],
        execution=execution,
    )
    from .progress import report_iteration, report_level_initialized, report_level_start

    for level in reversed(hierarchy[:-1]):
        report_level_start(execution, level)
        initialized = _initialize_hierarchy_level(
            node=level,
            child_state=state,
            execution=execution,
        )
        report_level_initialized(execution, initialized)
        refinement = initialized.refinement
        detector = execution.new_reentry_detector()
        iteration_index = 0
        certificate = None
        while iteration_index < refinement.max_iterations:
            iteration = solve_lp(refinement.runtime, iteration_index)
            certificate = check_optimality(refinement.runtime, iteration)
            if certificate.converged:
                break
            if detector is not None and detector.should_trigger():
                initialized = _switch_level_to_perturbed_cost(
                    initialized, iteration, certificate, execution, levels, detector
                )
                refinement = initialized.refinement
                detector = None
                iteration_index = 0
                continue
            update = update_support(
                refinement.runtime, iteration, certificate, track_reentry=detector is not None
            )
            report_iteration(execution, refinement.runtime)
            if detector is not None:
                detector.observe(update)
            iteration_index += 1

        if certificate is not None and certificate.converged:
            report_iteration(execution, refinement.runtime)

        state, result = _finalize_hierarchy_level(
            initialized,
            levels=levels,
            execution=execution,
        )
    return state, levels, result


def _leaf_report_lp_time(leaf_report: Any) -> float:
    """
    CN: 从 HELLO leaf report 的 solve_summary 中读取 LP 时间。
    EN: Read LP time from the solve_summary nested in an HELLO leaf report.
    """
    if not isinstance(leaf_report, dict):
        return 0.0
    solve_summary = leaf_report.get("solve_summary")
    if not isinstance(solve_summary, dict):
        return float(leaf_report.get("lp_solve_time_total", 0.0) or 0.0)
    return float(solve_summary.get("lp_solve_time_total", 0.0) or 0.0)


def _root_refinement_lp_time(root_level: Dict[str, Any]) -> float:
    """
    CN: 读取 root refinement 自身的 LP 时间，不包含其余 hierarchy 层级。
    EN: Read the root refinement's own LP time, excluding other hierarchy levels.
    """
    solve_summary = root_level.get("solve_summary")
    if not isinstance(solve_summary, dict):
        return 0.0
    return float(solve_summary.get("lp_solve_time_total", 0.0) or 0.0)


def _aggregate_hierarchy_lp_times(levels: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    CN: 汇总 hierarchy 中 coarsest solve 与 refinement 的 LP 时间且每层只计一次。
    EN: Sum coarsest-solve and refinement LP time across the hierarchy exactly once per level.
    """
    leaf_total = 0.0
    refinement_total = 0.0
    for level in levels:
        if str(level.get("kind", "")) == "leaf":
            leaf_total += float(level.get("lp_solve_time_total", 0.0) or 0.0)
            continue
        solve_summary = level.get("solve_summary")
        if isinstance(solve_summary, dict):
            refinement_total += float(solve_summary.get("lp_solve_time_total", 0.0) or 0.0)
    return {
        "leaf_lp_time_total": float(leaf_total),
        "refinement_lp_time_total": float(refinement_total),
        "lp_solve_time_total": float(leaf_total + refinement_total),
    }


def _hierarchy_timing_summary(levels: List[Dict[str, Any]], *, hierarchy_time: float) -> Dict[str, float]:
    """
    CN: 构造 HELLO 整段 hierarchy traversal 的耗时汇总。
    EN: Build the timing summary for the complete HELLO hierarchy traversal.
    """
    return {
        "hierarchy_time": float(hierarchy_time),
        **_aggregate_hierarchy_lp_times(levels),
    }



def _solve_hello_stage(
    *,
    source_F_full: np.ndarray,
    target_G_full: np.ndarray,
    source_cost_vec_full: np.ndarray,
    target_cost_vec_full: np.ndarray,
    source_mass_raw: np.ndarray,
    target_mass_raw: np.ndarray,
    config: SolverRuntimeConfig,
    depth_remaining: int,
    assignment_topk: int,
    split_count: int,
    dual_feasibility_tol: float,
    stopping_norm: StoppingNormPolicy = "l2",
    node_seed: int,
    tracer: Optional[_ChromeTraceCollector] = None,
    pricing_index_pool: Optional[Any],
    log: bool,
    return_coupling: bool,
    return_state: bool,
    progress: Optional[bool] = None,
    consume_input_features: bool = False,
    cost_perturbation: str = "off",
    cost_perturbation_relative_scale: float = 0.01,
) -> Any:
    """
    CN: 执行固定语义的 HELLO，并返回内部层级求解结果。
    EN: Run the fixed-semantics HELLO and return its internal hierarchical result.
    """
    t_total = time.perf_counter()
    progress_enabled = bool(log) if progress is None else bool(progress)
    # CN: 整次 solve 的显存由 API 边界上的 SolveMemoryTracker 统一统计；hierarchy 内部不再重置 allocator peak。
    # EN: The API-boundary SolveMemoryTracker owns whole-solve accounting; hierarchy traversal does not reset allocator peaks.
    memory_recorder = None
    n_source = int(source_F_full.shape[0])
    n_target = int(target_G_full.shape[0])
    # CN: 先把外部配置规范化为 hierarchy 内部只接受的少数模式，避免层级遍历中出现隐式分支。
    # EN: Normalize external options to the small mode set accepted by the hierarchy traversal.
    cost_name = str(getattr(config, "cost_type", "lowrank")).strip().lower()
    if stopping_norm not in {"l2", "finest_linf"}:
        raise ValueError("stopping_norm must be one of: l2, finest_linf")
    resolved_stopping_norm: StoppingNormPolicy = stopping_norm
    resolved_finest_norm = finest_stopping_norm(resolved_stopping_norm)
    if cost_name == "lowrank":
        cost_context = HelloCostContext(
            score_family="inner_product",
            cost_type="lowrank",
            dot_scale=float(getattr(config, "dot_scale", 1.0)),
        )
    elif cost_name in {"l1", "linf", "l2"}:
        cost_context = HelloCostContext(score_family="norm_cost", cost_type=cost_name)
    else:
        raise ValueError("hello supports cost_type in {'lowrank', 'l1', 'linf', 'l2'}.")
    if resolved_finest_norm == "linf" and cost_name != "lowrank":
        raise ValueError("stopping_norm='finest_linf' is currently supported only for native l2^2 costs.")
    if resolved_finest_norm == "linf" and str(getattr(config, "backend", "native")) != "native":
        raise ValueError("stopping_norm='finest_linf' is currently supported only by the native backend.")
    if resolved_finest_norm == "linf" and not bool(
        getattr(config, "use_fused_lowrank_feasibility_pricing", True)
    ):
        raise ValueError("stopping_norm='finest_linf' requires the fused lowrank feasibility/pricing scan.")
    if bool(progress_enabled):
        print(
            "[Stopping] "
            f"policy={resolved_stopping_norm} "
            f"intermediate_norm=l2 finest_norm={resolved_finest_norm}"
        )
    # CN: shuffle-once 在 hierarchy 构造前固定全局顺序；之后所有层都用连续区间表示，最后再映射回原始坐标。
    # EN: Shuffle-once fixes global order before hierarchy construction; levels then use contiguous ranges and are mapped back at output time.
    source_perm, target_perm = _make_shuffle_once_permutation(
        n_source=n_source,
        n_target=n_target,
        node_seed=int(node_seed),
        split_count=int(split_count),
    )
    record_solve_event(
        "hierarchy_permutation",
        source_permutation=source_perm,
        target_permutation=target_perm,
    )
    global_reorder_span = (
        memory_recorder.begin_span(
            phase="global_reorder",
            path="root",
            depth=0,
            n_source=n_source,
            n_target=n_target,
        )
        if memory_recorder is not None
        else None
    )
    with _hierarchy_global_reorder(
        source_F_full=source_F_full,
        target_G_full=target_G_full,
        source_cost_vec_full=np.asarray(source_cost_vec_full, dtype=np.float64, order="C"),
        target_cost_vec_full=np.asarray(target_cost_vec_full, dtype=np.float64, order="C"),
        source_mass_raw=np.asarray(source_mass_raw, dtype=np.float64, order="C"),
        target_mass_raw=np.asarray(target_mass_raw, dtype=np.float64, order="C"),
        source_perm=source_perm,
        target_perm=target_perm,
        consume_input_features=bool(consume_input_features),
    ) as (
        source_F_reordered,
        target_G_reordered,
        source_cost_reordered,
        target_cost_reordered,
        source_mass_reordered,
        target_mass_reordered,
    ):
        if memory_recorder is not None:
            memory_recorder.end_span(global_reorder_span)
            memory_recorder.record(
                phase="after_global_reorder",
                path="root",
                depth=0,
                n_source=n_source,
                n_target=n_target,
            )
        root = _HierarchyNodeRange(
            path="root",
            depth=0,
            depth_remaining=int(depth_remaining),
            source_start=0,
            source_stop=n_source,
            target_start=0,
            target_stop=n_target,
        )
        # CN: 主算法显式构造 hierarchy，再从 coarsest 到 finest 逐层初始化和 refinement。
        # EN: The main algorithm explicitly builds the hierarchy, then initializes and refines from coarsest to finest.
        hierarchy_execution = _HierarchyExecution(
            cost_context=cost_context,
            source_F=source_F_reordered,
            target_G=target_G_reordered,
            source_cost=source_cost_reordered,
            target_cost=target_cost_reordered,
            source_mass=source_mass_reordered,
            target_mass=target_mass_reordered,
            config=config,
            assignment_topk=int(assignment_topk),
            split_count=int(split_count),
            dual_feasibility_tol=float(dual_feasibility_tol),
            stopping_norm=resolved_stopping_norm,
            tracer=tracer,
            pricing_index_pool=pricing_index_pool,
            memory_recorder=memory_recorder,
            log=bool(progress_enabled),
        )
        t_hierarchy = time.perf_counter()
        from .perturbation import CostPerturbationRun, finish_original_problem

        perturbation = CostPerturbationRun(
            policy=cost_perturbation,
            relative_scale=cost_perturbation_relative_scale,
            random_seed=node_seed,
            source_index=source_perm,
            target_index=target_perm,
        )
        hierarchy_execution.perturbation = perturbation
        if cost_perturbation == "on":
            perturbation.activate(hierarchy_execution)
        root_state, levels, root_result = _solve_hierarchy(
            root=root,
            execution=hierarchy_execution,
        )
        if perturbation.activated:
            root_state, root_result = finish_original_problem(root, root_state, hierarchy_execution, levels)
        if root_result is None:
            raise RuntimeError("hello root node did not produce a solve result")
        perturbation.record_stage(list(levels))
        root_result["cost_stages"] = perturbation.stages
        root_result["cost_perturbation"] = {
            "policy": cost_perturbation, "activated": perturbation.activated, **perturbation.metadata,
        }
        hierarchy_time = float(time.perf_counter() - t_hierarchy)
    if memory_recorder is not None:
        memory_recorder.record(phase="after_hierarchy_solve", node=root, state=root_state)
    if root_result is None:
        raise RuntimeError("hello root node did not produce a solve result.")
    all_stage_levels = [level for stage in perturbation.stages for level in stage["levels"]]
    hierarchy_timing = _hierarchy_timing_summary(all_stage_levels, hierarchy_time=hierarchy_time)
    root_result.update(
        {
            key: float(value)
            for key, value in hierarchy_timing.items()
            if key != "hierarchy_time"
        }
    )
    root_level = next(
        (item for item in levels if int(item.get("depth", -1)) == 0 and str(item.get("path", "")) == "root"),
        {},
    )
    root_refinement_lp_time = _root_refinement_lp_time(root_level)
    support_before_solve = int(root_level.get("support_before_solve") or _state_support_size(root_state))
    # CN: 释放gpu显存
    # EN: Release GPU memory
    del root_state
    if memory_recorder is not None:
        memory_recorder.record(
            phase="after_del_root_state",
            path="root",
            depth=0,
            n_source=n_source,
            n_target=n_target,
        )

    output_mapping_span = (
        memory_recorder.begin_span(
            phase="output_mapping",
            path="root",
            depth=0,
            n_source=n_source,
            n_target=n_target,
        )
        if memory_recorder is not None
        else None
    )
    # CN: root_result 仍在 shuffle-once 坐标系中；对外返回前恢复 coupling、dual 和 warm-start state 的原始顺序。
    # EN: root_result is still in shuffle-once coordinates; restore coupling, duals, and warm-start state before returning.
    root_result["sparse_coupling"] = _map_sparse_coupling_to_original_order(
        root_result.get("sparse_coupling"),
        source_perm=source_perm,
        target_perm=target_perm,
        shape=(n_source, n_target),
    )
    root_result["dual_source"] = _map_dual_to_original_order(
        root_result.get("dual_source"),
        source_perm=source_perm,
        target_perm=target_perm,
    )
    state_for_return = root_result.get("warm_start_state")
    if state_for_return is not None:
        root_result["warm_start_state"] = _map_warm_start_state_to_original_order(
            _normalize_dual_assignment_state(state_for_return, pipeline="cpu"),
            source_perm=source_perm,
            target_perm=target_perm,
        )
    if bool(config.ensure_final_dual_feasible):
        if cost_context.score_family != "inner_product":
            raise ValueError(
                "ensure_final_dual_feasible currently supports only lowrank hello."
            )
        mapped_state = root_result.get("warm_start_state")
        if mapped_state is None:
            raise RuntimeError(
                "final dual feasibility correction requires hello warm_start_state."
            )
        corrected_state, correction_diagnostics = (
            _ensure_lowrank_final_dual_feasible(
                mapped_state,
                source_F=source_F_full,
                target_G=target_G_full,
                source_cost_vec=source_cost_vec_full,
                target_cost_vec=target_cost_vec_full,
                source_mass=source_mass_raw,
                target_mass=target_mass_raw,
                tracer=tracer,
                trace_prefix="hello.final_dual_feasibility_correction",
                dot_scale=float(cost_context.dot_scale),
            )
        )
        root_result["warm_start_state"] = corrected_state
        root_result["dual_source"] = np.asarray(
            corrected_state.dual_uv, dtype=np.float64
        )
        root_result.update(correction_diagnostics)
    level0_summary = extract_level_zero_summary(root_result)
    if memory_recorder is not None:
        memory_recorder.end_span(output_mapping_span)
        memory_recorder.record(
            phase="before_return",
            path="root",
            depth=0,
            n_source=n_source,
            n_target=n_target,
        )
    
    total_time = float(time.perf_counter() - t_total)
    # CN: diagnostics 汇总算法设置、层级记录和最终 root refinement 指标；实际数值结果仍由 root_result 承载。
    # EN: Diagnostics summarize settings, hierarchy records, and final root-refinement metrics; numeric outputs remain in root_result.
    diagnostics = {
        "enabled": True,
        "algorithm": "hello",
        "split_count": int(split_count),
        "assignment_mode": "nodewise_bidirectional",
        "edge_selection_mode": "nodewise",
        "augment_direction": "bidirectional",
        "cost_type": str(cost_context.cost_type),
        "score_family": str(cost_context.score_family),
        "stopping_norm": str(resolved_stopping_norm),
        "intermediate_stopping_norm": "l2",
        "finest_dual_feasibility_norm": str(resolved_finest_norm),
        "finest_lp_stopping_norm": str(resolved_finest_norm),
        "finest_stopping_norm_mismatch": False,
        "gpu_pipeline_enabled": True,
        "assignment_topk": int(assignment_topk),
        "hierarchy_depth": int(depth_remaining),
        "external_dual_root_initialization": False,
        "external_known_side": None,
        **hierarchy_timing,
        "warm_start_support_size": support_before_solve,
        "initial_active_support_size": support_before_solve,
        "final_warm_start_solve_time": float(root_result.get("time", 0.0)),
        "final_warm_start_lp_time": root_refinement_lp_time,
        "final_warm_start_iters": int(level0_summary.get("level0_inner_iterations", 0)),
        "total_hierarchy_solve_time": float(total_time),
        "num_leaf_subproblems": 1,
        "num_internal_nodes": int(
            sum(1 for item in levels if str(item.get("kind")) == "node" and int(item.get("depth", 0)) > 0)
        ),
        "levels": levels,
        "shuffle_once": {
            "source_perm_size": int(source_perm.size),
            "target_perm_size": int(target_perm.size),
        },
    }
    root_result["hello"] = {
        **diagnostics,
        "elapsed": float(total_time),
        "requested_hierarchy_depth": str(depth_remaining),
    }
    root_result["hello_diagnostics"] = dict(diagnostics)
    return _standard_output_from_log_result(
        root_result,
        log=bool(log),
        return_coupling=bool(return_coupling),
        return_state=bool(return_state),
    )


__all__ = ["build_parent_warm_start", "_solve_hello_stage"]
