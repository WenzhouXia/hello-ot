from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from .types import Problem

from hello_ot.initialization.state import (
    DeferredDualAssignmentCandidates,
    DualAssignmentState,
)
from hello_ot._internal.trace import _ChromeTraceCollector
from hello_ot.config import SolverRuntimeConfig
from hello_ot.state import GPUWarmStartState as OTWarmStartGPUState, WarmStartState as OTWarmStartState

from .kernels.norm_cost_scan import (
    _init_metric_warm_start_refinement,
    augment_topk_metric,
    check_metric_dual_feasibility,
    complete_dual_metric,
    pair_costs_metric,
    solve_leaf_metric,
)
from .kernels.scan_contract import DualFeasibilityCertificate
from .kernels.bilinear_scan import (
    augment_topk_lowrank,
    complete_dual_lowrank,
    materialize_lowrank_candidate_union,
    scan_topk_lowrank_candidates,
    solve_leaf_lowrank,
)
from hello_ot._internal.instrumentation.reporting import _print_warm_start_init_profile
from hello_ot.hierarchy.utilities import _normalize_subproblem_masses
from hello_ot.initialization.state import _normalize_dual_assignment_state
from hello_ot.refinement.iterations import (
    RefinementRuntime,
    begin_refinement,
    finalize_refinement,
)
from hello_ot.refinement.dual_feasibility import normalize_dual_feasibility_norm
from hello_ot.restricted_ot.runtime import (
    InitializedLevel,
    _init_lowrank_warm_start_refinement,
    finalize_solve_output,
    state_from_active_support,
    sum_level_summary_metric,
)


@dataclass(frozen=True)
class _BilinearCostRuntime:
    """
    CN: 私有双线性代价表示 c_ij=p_i+q_j-s<S_i,T_j>。
    EN: Private bilinear cost representation c_ij=p_i+q_j-s<S_i,T_j>.
    """

    source_points: np.ndarray
    target_points: np.ndarray
    source_offset: np.ndarray
    target_offset: np.ndarray
    dot_scale: float
    cost_type: Literal["l2^2", "factorized"]

    def __post_init__(self) -> None:
        scale = float(self.dot_scale)
        if scale not in {1.0, 2.0}:
            raise ValueError("dot_scale must be exactly 1 or 2")
        object.__setattr__(self, "dot_scale", scale)


@dataclass(frozen=True)
class _NormCostRuntime:
    """
    CN: 私有范数代价表示；点云保持原始坐标。
    EN: Private norm-cost representation retaining the original point coordinates.
    """

    source_points: np.ndarray
    target_points: np.ndarray
    cost_type: Literal["l1", "l2", "linf"]


CostRuntime = _BilinearCostRuntime | _NormCostRuntime


@dataclass(frozen=True)
class _PreparedOTProblem:
    """
    CN: HELLO 变体内部使用的已降低问题；正式公开入口仍只接收 Problem。
    EN: Lowered problem used internally by HELLO variants; the formal public entry still accepts only Problem.
    """

    runtime: CostRuntime
    source_mass: np.ndarray
    target_mass: np.ndarray
    reported_cost_type: str

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.runtime.source_points.shape[0]), int(self.runtime.target_points.shape[0])


def _prepare_bilinear_problem(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_offset: np.ndarray,
    target_offset: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    dot_scale: float = 1.0,
    reported_cost_type: str = "factorized",
) -> _PreparedOTProblem:
    """
    CN: 为 GW/UOT/SDOT 的内部 OT 子问题构造私有双线性 runtime。
    EN: Build a private bilinear runtime for internal GW/UOT/SDOT OT subproblems.
    """
    source = np.ascontiguousarray(source_points, dtype=np.float32)
    target = np.ascontiguousarray(target_points, dtype=np.float32)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("bilinear source and target points must be 2D with a shared feature dimension")
    source_bias = np.ascontiguousarray(source_offset, dtype=np.float64).reshape(-1)
    target_bias = np.ascontiguousarray(target_offset, dtype=np.float64).reshape(-1)
    source_weights = np.ascontiguousarray(source_mass, dtype=np.float64).reshape(-1)
    target_weights = np.ascontiguousarray(target_mass, dtype=np.float64).reshape(-1)
    if source_bias.size != source.shape[0] or source_weights.size != source.shape[0]:
        raise ValueError("source offsets and masses must match source points")
    if target_bias.size != target.shape[0] or target_weights.size != target.shape[0]:
        raise ValueError("target offsets and masses must match target points")
    return _PreparedOTProblem(
        runtime=_BilinearCostRuntime(
            source_points=source,
            target_points=target,
            source_offset=source_bias,
            target_offset=target_bias,
            dot_scale=float(dot_scale),
            cost_type="factorized",
        ),
        source_mass=source_weights,
        target_mass=target_weights,
        reported_cost_type=str(reported_cost_type),
    )


def prepare_cost(problem: Problem) -> CostRuntime:
    """
    CN: 将统一点云问题降低为 HELLO 内部 cost runtime；不物化 2X。
    EN: Lower a unified point-cloud problem to HELLO's internal cost runtime without materializing 2X.
    """
    if not isinstance(problem, Problem):
        raise TypeError("prepare_cost requires Problem")
    if problem.cost_type == "l2^2":
        source_offset = np.einsum("ij,ij->i", problem.source_points, problem.source_points, dtype=np.float64)
        target_offset = np.einsum("ij,ij->i", problem.target_points, problem.target_points, dtype=np.float64)
        return _BilinearCostRuntime(
            source_points=problem.source_points,
            target_points=problem.target_points,
            source_offset=np.ascontiguousarray(source_offset, dtype=np.float64),
            target_offset=np.ascontiguousarray(target_offset, dtype=np.float64),
            dot_scale=2.0,
            cost_type="l2^2",
        )
    return _NormCostRuntime(
        source_points=problem.source_points,
        target_points=problem.target_points,
        cost_type=problem.cost_type,
    )


def evaluate_pair_costs(runtime: CostRuntime, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """
    CN: 以统一语义计算稀疏边代价。
    EN: Evaluate sparse edge costs with unified semantics.
    """
    row_index = np.asarray(rows, dtype=np.int64)
    col_index = np.asarray(cols, dtype=np.int64)
    source = runtime.source_points[row_index]
    target = runtime.target_points[col_index]
    if isinstance(runtime, _BilinearCostRuntime):
        dots = np.einsum("ij,ij->i", source, target, dtype=np.float32)
        return np.asarray(
            runtime.source_offset[row_index]
            + runtime.target_offset[col_index]
            - float(runtime.dot_scale) * dots.astype(np.float64),
            dtype=np.float64,
        )
    difference = np.abs(source - target)
    if runtime.cost_type == "l1":
        return np.sum(difference, axis=1, dtype=np.float32).astype(np.float64)
    if runtime.cost_type == "linf":
        return np.max(difference, axis=1).astype(np.float64)
    return np.sqrt(np.sum(difference * difference, axis=1, dtype=np.float32)).astype(np.float64)


@dataclass(frozen=True)
class HelloCostContext:
    """
    CN: 将公开 cost 映射到共享算子 API 与各自优化 kernel 的内部 dispatch key。
    EN: Internal dispatch key mapping public costs to shared operator APIs and specialized kernels.
    """

    score_family: Literal["inner_product", "norm_cost"]
    cost_type: Literal["lowrank", "l1", "linf", "l2"]
    dot_scale: float = 1.0
    perturbation: Optional[Dict[str, Any]] = None

    def sliced(self, source_slice: slice, target_slice: slice) -> "HelloCostContext":
        """
        CN: 保留全局边标识，将同一扰动限制到当前层。
        EN: Restrict the same perturbation to this level while preserving global edge identities.
        """
        if self.perturbation is None:
            return self
        return replace(self, perturbation={
            **self.perturbation,
            "source_global_index": self.perturbation["source_global_index"][source_slice],
            "target_global_index": self.perturbation["target_global_index"][target_slice],
        })

    def __post_init__(self) -> None:
        if self.score_family == "inner_product" and self.cost_type != "lowrank":
            raise ValueError("inner_product score family requires cost_type='lowrank'")
        if self.score_family == "norm_cost" and self.cost_type not in {"l1", "l2", "linf"}:
            raise ValueError("norm_cost score family requires cost_type in {'l1', 'l2', 'linf'}")
        if self.score_family == "inner_product" and float(self.dot_scale) not in {1.0, 2.0}:
            raise ValueError("bilinear dot_scale must be exactly 1 or 2")
        if self.score_family == "norm_cost" and float(self.dot_scale) != 1.0:
            raise ValueError("norm costs do not accept dot_scale")


def solve_leaf_by_cost(
    *,
    cost_context: HelloCostContext,
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
) -> Tuple[DualAssignmentState, Dict[str, Any]]:
    if cost_context.score_family == "inner_product":
        return solve_leaf_lowrank(
            source_F_full=source_F_full,
            target_G_full=target_G_full,
            source_cost_vec_full=source_cost_vec_full,
            target_cost_vec_full=target_cost_vec_full,
            source_mass_raw=source_mass_raw,
            target_mass_raw=target_mass_raw,
            source_start=int(source_start),
            source_stop=int(source_stop),
            target_start=int(target_start),
            target_stop=int(target_stop),
            config=config,
            node_trace_args=node_trace_args,
            node_path=str(node_path),
            depth_level=int(depth_level),
            tracer=tracer,
            dot_scale=float(cost_context.dot_scale),
        )
    if cost_context.score_family == "norm_cost":
        return solve_leaf_metric(
            source_points_full=source_F_full,
            target_points_full=target_G_full,
            source_mass_raw=source_mass_raw,
            target_mass_raw=target_mass_raw,
            source_start=int(source_start),
            source_stop=int(source_stop),
            target_start=int(target_start),
            target_stop=int(target_stop),
            cost_type=cost_context.cost_type,  # type: ignore[arg-type]
            node_trace_args=node_trace_args,
            tracer=tracer,
            perturbation=cost_context.perturbation,
        )
    raise ValueError(f"unsupported score family={cost_context.score_family!r}")


def _directional_assignment_by_cost(
    *,
    cost_context: HelloCostContext,
    state: DualAssignmentState,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    known_side: Literal["source", "target"],
    assignment_topk: int,
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
    backend: str = "native",
    torch_device: str = "auto",
) -> Tuple[DualAssignmentState, Dict[str, Any], Dict[str, Any], Any]:
    if str(backend) == "torch":
        from .kernels.torch_scan import directional_assignment

        return directional_assignment(
            state=state,
            source_points=source_F,
            target_points=target_G,
            source_offset=source_cost_vec if cost_context.score_family == "inner_product" else None,
            target_offset=target_cost_vec if cost_context.score_family == "inner_product" else None,
            score_family=str(cost_context.score_family),
            cost_type=str(cost_context.cost_type),
            dot_scale=float(cost_context.dot_scale),
            known_side=known_side,
            topk=int(assignment_topk),
            device=str(torch_device),
            perturbation=cost_context.perturbation,
        )
    if cost_context.score_family == "inner_product":
        return augment_topk_lowrank(
            state=state,
            source_F=source_F,
            target_G=target_G,
            source_cost_vec=source_cost_vec,
            target_cost_vec=target_cost_vec,
            known_side=known_side,
            assignment_topk=int(assignment_topk),
            tracer=tracer,
            trace_args=trace_args,
            dot_scale=float(cost_context.dot_scale),
        )
    if cost_context.score_family == "norm_cost":
        return augment_topk_metric(
            state=state,
            source_points=source_F,
            target_points=target_G,
            known_side=known_side,
            cost_type=cost_context.cost_type,  # type: ignore[arg-type]
            assignment_topk=int(assignment_topk),
            tracer=tracer,
            trace_args=trace_args,
            perturbation=cost_context.perturbation,
        )
    raise ValueError(f"unsupported score family={cost_context.score_family!r}")


def propagate_and_assign_dual(**kwargs: Any) -> Tuple[DualAssignmentState, Dict[str, Any], Dict[str, Any], Any]:
    """
    CN: initialization 第一阶段：从继承的单侧 dual 做 directional assignment，并返回 completion seed。
    EN: Initialization stage one: run directional assignment from one inherited dual side and return a completion seed.
    """
    return _directional_assignment_by_cost(**kwargs)


def assign_from_complete_dual(**kwargs: Any) -> Tuple[DualAssignmentState, Dict[str, Any], Dict[str, Any], Any]:
    """
    CN: initialization 第二阶段：在完整 dual 上执行另一方向的 assignment。
    EN: Initialization stage two: run the opposite directional assignment from a complete dual pair.
    """
    state = kwargs.get("state")
    if state is None or getattr(state, "dual_uv", None) is None:
        raise ValueError("assign_from_complete_dual requires a complete dual pair")
    return _directional_assignment_by_cost(**kwargs)


def scan_deferred_assignment_candidates(
    *,
    cost_context: HelloCostContext,
    state: DualAssignmentState,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    known_side: Literal["source", "target"],
    assignment_topk: int,
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
) -> Tuple[
    DualAssignmentState,
    Dict[str, Any],
    Dict[str, Any],
    Any,
    DeferredDualAssignmentCandidates,
]:
    """
    CN: 为 inner-product initialization 执行不立即构造 support 的 directional scan。
    EN: Run an inner-product initialization directional scan without immediate support construction.
    """
    if cost_context.score_family != "inner_product":
        raise ValueError("deferred assignment candidates currently require the inner_product score family")
    return scan_topk_lowrank_candidates(
        state=state,
        source_F=source_F,
        target_G=target_G,
        source_cost_vec=source_cost_vec,
        target_cost_vec=target_cost_vec,
        known_side=known_side,
        assignment_topk=int(assignment_topk),
        tracer=tracer,
        trace_args=trace_args,
        dot_scale=float(cost_context.dot_scale),
    )


def materialize_deferred_assignment_union(
    *,
    cost_context: HelloCostContext,
    state: DualAssignmentState,
    candidates: Tuple[DeferredDualAssignmentCandidates, DeferredDualAssignmentCandidates],
    tracer: Optional[_ChromeTraceCollector],
    trace_args: Dict[str, Any],
) -> Tuple[OTWarmStartGPUState, Dict[str, Any]]:
    """
    CN: 两遍 inner-product scan 后只做一次 GPU candidate unique。
    EN: Perform one GPU candidate unique after both inner-product scans.
    """
    if cost_context.score_family != "inner_product":
        raise ValueError("deferred assignment materialization currently requires the inner_product score family")
    return materialize_lowrank_candidate_union(
        state=state,
        candidates=candidates,
        tracer=tracer,
        trace_args=trace_args,
    )


def complete_dual(
    *,
    cost_context: HelloCostContext,
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
    backend: str = "native",
    torch_device: str = "auto",
) -> Tuple[DualAssignmentState, Dict[str, Any]]:
    if str(backend) == "torch":
        from .kernels.torch_scan import TorchCompletionCandidate, complete_from_candidate

        if not isinstance(completion_candidate, TorchCompletionCandidate):
            raise TypeError("Torch dual completion requires TorchCompletionCandidate")
        return complete_from_candidate(
            state=state,
            candidate=completion_candidate,
            device=str(torch_device),
        )
    if cost_context.score_family == "inner_product":
        return complete_dual_lowrank(
            state=state,
            source_F=source_F,
            target_G=target_G,
            source_cost_vec=source_cost_vec,
            target_cost_vec=target_cost_vec,
            known_side=known_side,
            completion_candidate=completion_candidate,
            tracer=tracer,
            trace_args=trace_args,
            trace_prefix=str(trace_prefix),
            dot_scale=float(cost_context.dot_scale),
        )
    if cost_context.score_family == "norm_cost":
        return complete_dual_metric(
            state=state,
            source_points=source_F,
            target_points=target_G,
            known_side=known_side,
            cost_type=cost_context.cost_type,  # type: ignore[arg-type]
            completion_candidate=completion_candidate,
            tracer=tracer,
            trace_args=trace_args,
            trace_prefix=str(trace_prefix),
            perturbation=cost_context.perturbation,
        )
    raise ValueError(f"unsupported score family={cost_context.score_family!r}")


def check_dual_feasibility(
    *,
    cost_context: HelloCostContext,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    dual_uv: Any,
    gpu_id: Optional[int] = None,
) -> Tuple[DualFeasibilityCertificate, Dict[str, Any]]:
    """
    CN: 以统一 API 在全边集上返回 L2/L∞ dual-feasibility certificate。
    EN: Return an L2/Linf dual-feasibility certificate over the full edge set through one API.
    """
    if cost_context.score_family == "inner_product":
        # CN: 惰性 import 避免 cost dispatch 与 refinement loop 之间的模块循环。
        # EN: Import lazily to avoid a module cycle between cost dispatch and the refinement loop.
        from hello_ot.refinement.stopping import lowrank_dual_feasibility_infeasibility

        diagnostics: Dict[str, Any] = {}
        lowrank_dual_feasibility_infeasibility(
            source_F,
            target_G,
            source_cost_vec,
            target_cost_vec,
            dual_uv,
            gpu_id=gpu_id,
            diagnostics=diagnostics,
            dot_scale=float(cost_context.dot_scale),
        )
        return (
            DualFeasibilityCertificate(
                l2_numerator_sq=float(diagnostics["dual_feasibility_num_sq"]),
                l2_denominator_sq=float(diagnostics["dual_feasibility_den_sq"]),
                max_positive_violation=float(diagnostics["dual_feasibility_max_violation"]),
                cost_linf=float(diagnostics["dual_feasibility_cost_linf"]),
                positive_count=int(diagnostics["dual_feasibility_positive_count"]),
            ),
            diagnostics,
        )
    if cost_context.score_family == "norm_cost":
        dual_data = dual_uv.detach().cpu().numpy() if hasattr(dual_uv, "detach") else dual_uv
        dual = np.asarray(dual_data, dtype=np.float64).reshape(-1)
        n_source = int(np.asarray(source_F).shape[0])
        n_target = int(np.asarray(target_G).shape[0])
        if int(dual.size) != n_source + n_target:
            raise ValueError("dual_uv length must equal n_source + n_target")
        return check_metric_dual_feasibility(
            source_points=source_F,
            target_points=target_G,
            source_dual=dual[:n_source],
            target_dual=dual[n_source:],
            cost_type=cost_context.cost_type,  # type: ignore[arg-type]
            perturbation=cost_context.perturbation,
            source_global_index=None if cost_context.perturbation is None else cost_context.perturbation["source_global_index"],
            target_global_index=None if cost_context.perturbation is None else cost_context.perturbation["target_global_index"],
        )
    raise ValueError(f"unsupported score family={cost_context.score_family!r}")


@dataclass(frozen=True)
class PreparedCostRefinement:
    """
    CN: cost dispatch 完成后的单层 refinement；主算法只读取 runtime 与迭代上限。
    EN: A cost-dispatched level refinement; the main algorithm only reads its runtime and iteration limit.
    """

    runtime: RefinementRuntime
    initialized: InitializedLevel
    n_source: int
    n_target: int
    backend: str
    trace_context: Any

    @property
    def max_iterations(self) -> int:
        return int(self.runtime.config.max_inner_iter)


def begin_refinement_by_cost(
    *,
    cost_context: HelloCostContext,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    warm_start: OTWarmStartState | OTWarmStartGPUState,
    config: SolverRuntimeConfig,
    skip_initial_pricing: bool,
    dual_feasibility_tol: float,
    dual_feasibility_norm: Literal["l2", "linf"] = "l2",
    lp_termination_norm: Literal["l2", "linf"] = "l2",
    tracer: Optional[_ChromeTraceCollector],
    trace_prefix: str,
    pricing_index_pool: Optional[Any],
    warm_start_profile_depth: int,
) -> PreparedCostRefinement:
    """
    CN: 按 cost family 构造 restricted-OT runtime，但不执行 refinement 迭代。
    EN: Build the restricted-OT runtime for a cost family without running refinement iterations.
    """
    n_source = int(source_F.shape[0])
    n_target = int(target_G.shape[0])
    trace_context = (
        tracer.span(f"{trace_prefix}.total", "solve_ot")
        if tracer is not None
        else nullcontext()
    )
    if cost_context.score_family == "inner_product":
        normalized_source_mass, normalized_target_mass, _ = _normalize_subproblem_masses(
            source_mass,
            target_mass,
        )
        solve_config = replace(config)
        solve_config.pricing_strategy = "nodewise_full"
        solve_config.convergence_criterion = "dual_feasibility"
        solve_config.require_dual_feasibility_convergence = False
        solve_config.dual_feasibility_tol = float(dual_feasibility_tol)
        solve_config.lp_termination_norm = str(lp_termination_norm)
        solve_config.require_added_convergence = False
        solve_config.cost_type = "lowrank"
        solve_config.dot_scale = float(cost_context.dot_scale)
        solve_config.validate()
        initialized = _init_lowrank_warm_start_refinement(
            source_F=source_F,
            target_G=target_G,
            source_cost_vec=source_cost_vec,
            target_cost_vec=target_cost_vec,
            source_mass=np.asarray(normalized_source_mass, dtype=np.float64),
            target_mass=np.asarray(normalized_target_mass, dtype=np.float64),
            config=solve_config,
            warm_start=warm_start,
            skip_initial_pricing=bool(skip_initial_pricing),
            trace_collector=tracer,
            trace_prefix=trace_prefix,
            pricing_index_pool=pricing_index_pool,
            warm_start_cost_vec_chunk_size=65536,
            warm_start_cost_vec_feature_chunk_size=None,
            _consume_warm_start_gpu_state=True,
            dot_scale=float(cost_context.dot_scale),
        )
        initial_support_info = dict(initialized.initialization.statistics)
        _print_warm_start_init_profile(
            initialized.solver,
            components=dict(initial_support_info.get("components", {})),
            primal_nnz=int(initial_support_info.get("primal_nnz", 0)),
            active_size=int(
                initial_support_info.get(
                    "post_pricing_support_size",
                    initialized.solver.active_support.size,
                )
            ),
            bfs_added=int(initial_support_info.get("northwest_added", 0)),
            profile_depth=int(warm_start_profile_depth),
        )
        trace_context.__enter__()
        runtime = begin_refinement(
            initialized.solver,
            initialized.cfg,
            initialized.initialization.dual_warm_start,
            trace_collector=tracer,
            trace_prefix=trace_prefix,
            warm_start_profile_depth=warm_start_profile_depth,
            dual_feasibility_norm=normalize_dual_feasibility_norm(
                dual_feasibility_norm
            ),
        )
        return PreparedCostRefinement(
            runtime=runtime,
            initialized=initialized,
            n_source=n_source,
            n_target=n_target,
            backend=str(getattr(config, "backend", "native")),
            trace_context=trace_context,
        )
    if cost_context.score_family == "norm_cost":
        if str(lp_termination_norm) != "l2":
            raise ValueError("L-infinity restricted-LP termination is currently supported only for lowrank/l2^2 costs.")
        if str(dual_feasibility_norm) != "l2":
            raise ValueError("L-infinity finest-level stopping is currently supported only for lowrank/l2^2 costs.")
        if bool(skip_initial_pricing):
            raise ValueError("skip_initial_pricing is only supported by lowrank asymmetric-chain refinement.")
        del pricing_index_pool
        initialized = _init_metric_warm_start_refinement(
            source_points=source_F,
            target_points=target_G,
            source_mass=source_mass,
            target_mass=target_mass,
            cost_type=cost_context.cost_type,  # type: ignore[arg-type]
            warm_start=warm_start,
            config=config,
            dual_feasibility_tol=float(dual_feasibility_tol),
            tracer=tracer,
            trace_prefix=str(trace_prefix),
            perturbation=cost_context.perturbation,
        )
        trace_context.__enter__()
        runtime = begin_refinement(
            initialized.solver,
            initialized.cfg,
            initialized.initialization.dual_warm_start,
            trace_collector=tracer,
            trace_prefix=trace_prefix,
            warm_start_profile_depth=warm_start_profile_depth,
        )
        return PreparedCostRefinement(
            runtime=runtime,
            initialized=initialized,
            n_source=n_source,
            n_target=n_target,
            backend=str(getattr(config, "backend", "native")),
            trace_context=trace_context,
        )
    raise ValueError(f"unsupported score family={cost_context.score_family!r}")


def finalize_refinement_by_cost(
    prepared: PreparedCostRefinement,
) -> Tuple[DualAssignmentState, Dict[str, Any]]:
    """
    CN: 将已完成迭代的 runtime 整理为 hierarchy 可继续传播的 state 与层级结果。
    EN: Finalize an iterated runtime into the state and level result consumed by the hierarchy.
    """
    warm_result = finalize_refinement(prepared.runtime)
    prepared.trace_context.__exit__(None, None, None)
    solver = prepared.initialized.solver
    state = state_from_active_support(
        solver,
        n_source=prepared.n_source,
        n_target=prepared.n_target,
        dual=warm_result["dual"],
    )
    result = finalize_solve_output(
        distance=warm_result["distance"],
        coupling=warm_result["coupling"],
        state=state,
        dual_source=warm_result["dual"],
        level_summaries=warm_result["level_summaries"],
        lp_solve_time_total=sum_level_summary_metric(
            warm_result["level_summaries"], "lp_time"
        ),
        elapsed=solver.build_time + warm_result["solve_time"],
        log=True,
        return_coupling=True,
        return_state=True,
    )
    if not isinstance(result, dict):
        raise RuntimeError("HELLO refinement expected a structured result dictionary.")
    result["lp_backend_peak_mem_mib"] = warm_result.get("lp_backend_peak_mem_mib")
    result["lp_backend_delta_peak_mem_mib"] = warm_result.get(
        "lp_backend_delta_peak_mem_mib"
    )
    result["lp_backend_abs_peak_mem_mib"] = warm_result.get(
        "lp_backend_abs_peak_mem_mib"
    )
    result["pricing_peak_mem_mib"] = warm_result.get("pricing_peak_mem_mib")
    result["dual_feas_peak_mem_mib"] = warm_result.get("dual_feas_peak_mem_mib")
    result["warm_start_iteration_records"] = list(
        warm_result.get("iteration_records") or []
    )
    result["warm_start_init"] = dict(prepared.initialized.initialization.statistics)
    normalized_state = (
        state
        if prepared.backend == "torch"
        else _normalize_dual_assignment_state(state, pipeline="gpu")
    )
    result["warm_start_state"] = normalized_state
    return normalized_state, result


def pair_costs_by_cost(
    *,
    cost_context: HelloCostContext,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    rows: Any,
    cols: Any,
) -> Any:
    if cost_context.score_family == "norm_cost":
        return pair_costs_metric(
            source_points=source_F,
            target_points=target_G,
            rows=rows,
            cols=cols,
            cost_type=cost_context.cost_type,  # type: ignore[arg-type]
            perturbation=cost_context.perturbation,
        )
    if cost_context.score_family == "inner_product":
        source = np.asarray(source_F, dtype=np.float32, order="C")
        target = np.asarray(target_G, dtype=np.float32, order="C")
        rr = np.asarray(rows, dtype=np.int64)
        cc = np.asarray(cols, dtype=np.int64)
        dot = np.einsum("ij,ij->i", source[rr], target[cc], optimize=True)
        return np.asarray(
            np.asarray(source_cost_vec, dtype=np.float64)[rr]
            + np.asarray(target_cost_vec, dtype=np.float64)[cc]
            - float(cost_context.dot_scale) * dot.astype(np.float64),
            dtype=np.float64,
        )
    raise ValueError(f"unsupported score family={cost_context.score_family!r}")


__all__ = [
    "HelloCostContext",
    "assign_from_complete_dual",
    "check_dual_feasibility",
    "complete_dual",
    "evaluate_pair_costs",
    "pair_costs_by_cost",
    "prepare_cost",
    "propagate_and_assign_dual",
    "begin_refinement_by_cost",
    "finalize_refinement_by_cost",
    "solve_leaf_by_cost",
]
