from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp

from hello_ot.api import _require_cuda_runtime, _runtime_config, _solution, _solve, _stage_result
from hello_ot.config import SolverOptions, _AlgorithmConfig, finest_stopping_norm
from hello_ot.cost import HelloCostContext, _PreparedOTProblem, _prepare_bilinear_problem
from hello_ot.initialization.dual_assignment import (
    DualAssignmentProblem,
    build_warm_start_from_dual,
)
from hello_ot.refinement.loop import _refine_lowrank_from_warm_start
from hello_ot.state import WarmStartState
from hello_ot.types import DualPotentials, DualPreparation, PreparedDual, Result


@dataclass(frozen=True)
class BalancedOTResult:
    """
    CN: variants 内部消费的平衡 HELLO 子问题结果。
    EN: Balanced HELLO subproblem result consumed internally by variants.
    """

    objective: float
    coupling: sp.coo_matrix
    state: WarmStartState
    hello_result: Result
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class BilinearCostFactors:
    """
    CN: 可复用的双线性代价因子，表示 c_ij=p_i+q_j-s<S_i,T_j>。
    EN: Reusable bilinear factors representing c_ij=p_i+q_j-s<S_i,T_j>.
    """

    source_points: np.ndarray
    target_points: np.ndarray
    source_offset: np.ndarray
    target_offset: np.ndarray
    dot_scale: float = 1.0


def algorithm_config(
    options: SolverOptions | None,
    *,
    max_iterations: int = 100,
    random_seed: int = 42,
    assignment_topk: int | None = None,
) -> _AlgorithmConfig:
    """
    CN: 将 variants 的内部平衡 OT 设置降低为正式 HELLO 配置。
    EN: Lower variant balanced-OT settings to the formal HELLO configuration.
    """
    resolved = SolverOptions(verbose="off") if options is None else options
    if not isinstance(resolved, SolverOptions):
        raise TypeError("inner_options must be SolverOptions or None")
    config = resolved._to_internal_config(
        max_iterations=int(max_iterations),
        random_seed=int(random_seed),
    )
    if assignment_topk is not None:
        config = replace(config, assignment_topk=int(assignment_topk))
    return replace(config, cost_perturbation="off", verbose="off")


def _diagnostics(result: Result) -> Dict[str, Any]:
    levels = result.solve_stage.levels
    total_iterations = 0
    peak_support = 0
    for level in levels:
        summary = getattr(level.solve, "summary", None)
        if summary is not None:
            total_iterations += int(summary.iterations)
            peak_support = max(peak_support, int(summary.peak_active_support_size))
        else:
            peak_support = max(peak_support, int(getattr(level.solve, "support_size", 0)))
    return {
        "wall_time": float(result.solve_stage.wall_time),
        "total_inner_iterations": int(total_iterations),
        "peak_active_support_size": int(peak_support),
        "levels": levels,
    }


def result_record(result: BalancedOTResult) -> Dict[str, Any]:
    """
    CN: 为 GW/UOT 外层算法生成标量和稀疏结果记录。
    EN: Build a scalar-and-sparse record for GW/UOT outer algorithms.
    """
    level_summaries = []
    for level in result.hello_result.solve_stage.levels:
        summary = getattr(level.solve, "summary", None)
        level_summaries.append(
            {
                "level": int(level.level_index),
                "n_source": int(level.n_source),
                "n_target": int(level.n_target),
                "iters": 0 if summary is None else int(summary.iterations),
                "time": float(level.wall_time),
                "lp_time": float(sum(item.solve_lp.wall_time for item in getattr(level.solve, "iterations", ()))),
                "support_final": int(getattr(summary, "final_active_support_size", getattr(level.solve, "support_size", 0))),
                "peak_active_support_size": int(getattr(summary, "peak_active_support_size", getattr(level.solve, "support_size", 0))),
            }
        )
    return {
        "distance": float(result.objective),
        "sparse_coupling": result.coupling,
        "warm_start_state": result.state,
        "level_summaries": level_summaries,
        "elapsed": float(result.hello_result.total_wall_time),
        "lp_solve_time_total": float(sum(item["lp_time"] for item in level_summaries)),
        "peak_active_support_size": int(result.diagnostics["peak_active_support_size"]),
    }


def _dual_assignment_problem(problem: _PreparedOTProblem) -> DualAssignmentProblem:
    runtime = problem.runtime
    return DualAssignmentProblem(
        cost_context=HelloCostContext(
            score_family="inner_product",
            cost_type="lowrank",
            dot_scale=float(runtime.dot_scale),
        ),
        source_representation=runtime.source_points,
        target_representation=runtime.target_points,
        source_cost_vector=runtime.source_offset,
        target_cost_vector=runtime.target_offset,
    )


def _solve_from_dual(
    problem: _PreparedOTProblem,
    dual: DualPotentials | PreparedDual,
    config: _AlgorithmConfig,
    *,
    preparation: DualPreparation,
    support_proposal_problem: _PreparedOTProblem | None,
) -> Result:
    """
    CN: 从相关问题的 dual potentials 构造 active support 并执行单层 refinement。
    EN: Build active support from related-problem duals and run single-level refinement.
    """
    if config.backend != "native":
        # CN: portable backend 尚无 CPU dual-assignment kernel；冷启动仍求解同一平衡 OT 子问题。
        # EN: The portable backend has no CPU dual-assignment kernel yet; a cold start still solves the same balanced OT subproblem.
        return _solve(problem, config)
    from hello_ot._native_compat import require_native_runtime_compatibility

    require_native_runtime_compatibility()
    _require_cuda_runtime()
    proposal = problem if support_proposal_problem is None else support_proposal_problem
    if proposal.shape != problem.shape:
        raise ValueError("support-proposal and target problems must have the same shape")
    warm_start, initialization_profile = build_warm_start_from_dual(
        _dual_assignment_problem(proposal),
        dual,
        preparation=preparation,
        assignment="nodewise_bidirectional",
        assignment_topk=int(config.assignment_topk),
    )
    runtime = problem.runtime
    runtime_config = _runtime_config(config, cost_type="lowrank", dot_scale=float(runtime.dot_scale))
    started = time.perf_counter()
    raw = _refine_lowrank_from_warm_start(
        source_F=runtime.source_points,
        target_G=runtime.target_points,
        source_cost_vec=runtime.source_offset,
        target_cost_vec=runtime.target_offset,
        source_mass=problem.source_mass,
        target_mass=problem.target_mass,
        log=True,
        return_coupling=True,
        return_state=True,
        config=runtime_config,
        warm_start=warm_start,
        skip_initial_pricing=True,
        use_lp_warm_start_dual=True,
        dual_feasibility_norm=finest_stopping_norm(config.stopping_norm),
        dot_scale=float(runtime.dot_scale),
    )
    wall_time = float(time.perf_counter() - started)
    if not isinstance(raw, dict):
        raise RuntimeError("HELLO dual-initialized refinement did not return a structured result")
    return Result(
        objective=float(raw.get("distance", float("nan"))),
        solution=_solution(raw, shape=problem.shape),
        solve_stage=_stage_result(raw, config=config, wall_time=wall_time),
        total_wall_time=wall_time,
        metadata={
            "cost_type": str(problem.reported_cost_type),
            "backend": str(config.backend),
            "dual_preparation": str(preparation),
            "support_proposal_cost": "target" if support_proposal_problem is None else "related",
            "initialization_profile": initialization_profile,
        },
    )


def solve_bilinear_subproblem(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_offset: np.ndarray,
    target_offset: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    config: _AlgorithmConfig,
    dot_scale: float = 1.0,
    inherited_dual: Optional[DualPotentials | PreparedDual] = None,
    assignment_topk: Optional[int] = None,
    preparation: DualPreparation = "preserve",
    support_proposal_cost: Optional[BilinearCostFactors] = None,
) -> BalancedOTResult:
    """
    CN: 从私有双线性 runtime 求解 variants 的平衡 OT 子问题。
    EN: Solve a variant balanced-OT subproblem from private bilinear factors.
    """
    problem = _prepare_bilinear_problem(
        source_points=source_points,
        target_points=target_points,
        source_offset=source_offset,
        target_offset=target_offset,
        source_mass=source_mass,
        target_mass=target_mass,
        dot_scale=float(dot_scale),
    )
    local_config = config if assignment_topk is None else replace(config, assignment_topk=int(assignment_topk))
    proposal_problem = None
    if support_proposal_cost is not None:
        proposal_problem = _prepare_bilinear_problem(
            source_points=support_proposal_cost.source_points,
            target_points=support_proposal_cost.target_points,
            source_offset=support_proposal_cost.source_offset,
            target_offset=support_proposal_cost.target_offset,
            source_mass=source_mass,
            target_mass=target_mass,
            dot_scale=float(support_proposal_cost.dot_scale),
        )
    result = (
        _solve(problem, local_config)
        if inherited_dual is None
        else _solve_from_dual(
            problem,
            inherited_dual,
            local_config,
            preparation=preparation,
            support_proposal_problem=proposal_problem,
        )
    )
    coupling = result.solution.to_sparse_matrix().tocoo(copy=False)
    return BalancedOTResult(
        objective=float(result.objective),
        coupling=coupling,
        state=result.solution.to_warm_start_state(),
        hello_result=result,
        diagnostics=_diagnostics(result),
    )


def inherited_dual_from_state(state: WarmStartState) -> PreparedDual:
    """
    CN: 从相关 OT 状态中提取双侧 dual potentials，不继承 primal support。
    EN: Extract two-sided dual potentials from a related OT state without inheriting primal support.
    """
    if state.dual_uv is None:
        raise ValueError("related OT state does not contain dual potentials")
    values = np.asarray(state.dual_uv, dtype=np.float64).reshape(-1)
    expected = int(state.n_source) + int(state.n_target)
    if values.size != expected:
        raise ValueError("related OT state has incompatible dual length")
    return PreparedDual(
        source=np.ascontiguousarray(values[: state.n_source]),
        target=np.ascontiguousarray(values[state.n_source :]),
    )


__all__ = [
    "BalancedOTResult",
    "BilinearCostFactors",
    "algorithm_config",
    "inherited_dual_from_state",
    "result_record",
    "solve_bilinear_subproblem",
]
