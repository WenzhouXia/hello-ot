from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .problem import LinearOTProblem
from .result import LinearOTResult


@dataclass(frozen=True)
class HelloSolveArtifacts:
    """
    CN: 保存统一 HELLO 结果及可供精确 solver warm start 的 dual。
    EN: Store the unified HELLO result and duals usable to warm-start an exact solver.
    """

    result: LinearOTResult
    source_dual: np.ndarray
    target_dual: np.ndarray


def solve_hello(
    problem: LinearOTProblem,
    *,
    run_dir: Path | None = None,
    seed: int = 42,
    coarsest_size_threshold: int = 1024,
    split_count: int = 4,
    assignment_topk: int = 16,
    pricing_topk: float = 2.0,
    cleaning_threshold_factor: float = 10.0,
    max_inner_iter: int = 100,
    stopping_norm: str = "l2",
    cost_perturbation: str = "off",
    cost_perturbation_relative_scale: float = 0.01,
    hello_verbose: str = "off",
) -> LinearOTResult:
    """
    CN: 从 squared-L2 点云通过正式点云 API 运行 HELLO。
    EN: Run HELLO from squared-L2 point clouds through the formal point-cloud API.
    """
    artifacts = solve_hello_with_duals(
        problem,
        run_dir=run_dir,
        seed=seed,
        coarsest_size_threshold=coarsest_size_threshold,
        split_count=split_count,
        assignment_topk=assignment_topk,
        pricing_topk=pricing_topk,
        cleaning_threshold_factor=cleaning_threshold_factor,
        max_inner_iter=max_inner_iter,
        stopping_norm=stopping_norm,
        cost_perturbation=cost_perturbation,
        cost_perturbation_relative_scale=cost_perturbation_relative_scale,
        hello_verbose=hello_verbose,
    )
    return artifacts.result


def solve_hello_with_duals(
    problem: LinearOTProblem,
    *,
    run_dir: Path | None = None,
    seed: int = 42,
    coarsest_size_threshold: int = 1024,
    split_count: int = 4,
    assignment_topk: int = 16,
    pricing_topk: float = 2.0,
    cleaning_threshold_factor: float = 10.0,
    max_inner_iter: int = 100,
    stopping_norm: str = "l2",
    cost_perturbation: str = "off",
    cost_perturbation_relative_scale: float = 0.01,
    hello_verbose: str = "off",
) -> HelloSolveArtifacts:
    """
    CN: 运行 HELLO，同时导出其最终 dual 供后续精确求解器 warm start。
    EN: Run HELLO and export its final duals for a subsequent exact solver warm start.
    """
    del run_dir
    if problem.cost_type != "l2^2":
        raise ValueError("HELLO Pareto adapter currently supports only cost_type='l2^2'.")
    import torch

    import hello_ot

    if int(pricing_topk) != pricing_topk:
        raise ValueError("pricing_topk must be an integer")
    options = hello_ot.SolverOptions(
        split_count=int(split_count), coarsest_size_threshold=int(coarsest_size_threshold),
        assignment_topk=int(assignment_topk), pricing_topk=int(pricing_topk),
        support_budget_factor=float(cleaning_threshold_factor),
        stopping_norm=str(stopping_norm), backend="native",
        cost_perturbation=str(cost_perturbation),
        cost_perturbation_relative_scale=float(cost_perturbation_relative_scale),
        profile_memory=False, verbose=str(hello_verbose), record_trace=False,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("HELLO linear OT adapter requires a CUDA GPU.")
    torch.cuda.synchronize()
    solve_t0 = time.perf_counter()
    result = hello_ot.solve(
        hello_ot.Problem(problem.source_points, problem.target_points,
                         source_mass=problem.source_mass, target_mass=problem.target_mass,
                         cost_type="l2^2"),
        random_seed=int(seed), max_iterations=int(max_inner_iter), options=options,
    )
    coupling = result.solution.to_sparse_matrix()
    source_dual = np.ascontiguousarray(result.solution.source_dual, dtype=np.float64)
    target_dual = np.ascontiguousarray(result.solution.target_dual, dtype=np.float64)
    converged, diag_metrics = _result_metrics(result, problem)
    torch.cuda.synchronize()
    runtime_sec = float(time.perf_counter() - solve_t0)
    linear_result = LinearOTResult(
        method="hello",
        solver_objective=float(result.objective),
        runtime_sec=runtime_sec,
        transport_kind="sparse",
        transport=coupling,
        converged=converged,
        status="converged" if converged else "not_converged",
        diagnostics={
            **diag_metrics,
            "coarsest_size_threshold": int(coarsest_size_threshold),
            "split_count": int(split_count),
            "assignment_topk": int(assignment_topk),
            "pricing_topk": float(pricing_topk),
            "dual_feasibility_tol": 1.0e-6,
            "cleaning_threshold_factor": float(cleaning_threshold_factor),
            "tolerance": 1.0e-6,
            "max_inner_iter": int(max_inner_iter),
            "use_faiss_backend": False,
            "variable_bound_mode": "constant",
            "matrix_value_mode": "implicit_aty",
            "vector_sum_mode": "direct_reduce",
            "stopping_norm": str(stopping_norm),
            "intermediate_stopping_norm": "l2",
            "finest_stopping_norm": "linf" if stopping_norm == "finest_linf" else "l2",
            "cost_perturbation": str(cost_perturbation),
        },
    )
    return HelloSolveArtifacts(
        result=linear_result,
        source_dual=source_dual,
        target_dual=target_dual,
    )



def _result_metrics(result, problem=None):
    """
    CN: 缺失最终诊断时报错，不填造成功状态。
    EN: Reject missing final diagnostics instead of fabricating success.
    """
    levels = [level for level in result.solve_stage.levels if level.level_index == 0]
    if len(levels) == 1 and levels[0].kind == "coarsest" and problem is not None:
        return _coarsest_metrics(result, problem)
    if len(levels) != 1 or levels[0].kind != "refined" or not levels[0].solve.iterations:
        raise RuntimeError("Missing final HELLO refinement diagnostics")
    level = levels[0]
    final = level.solve.iterations[-1]
    return bool(level.solve.summary.converged), {
        "relative_primal_feasibility": final.solve_lp.primal_feasibility,
        "relative_primal_dual_gap": final.solve_lp.primal_dual_gap,
        "relative_full_dual_feasibility": final.convergence.relative_full_dual_feasibility,
        "relative_linf_dual_feasibility": final.convergence.relative_linf_dual_feasibility,
        "dual_feasibility_max_violation": final.convergence.max_dual_violation,
        "level0_inner_iterations": len(level.solve.iterations),
        "peak_active_support_size": int(result.solution.values.size),
    }


def _coarsest_metrics(result, problem):
    """
    CN: 单层精确解没有 refinement 记录，直接重算原问题 KKT。
    EN: A single-level exact solve has no refinement record; recompute its original KKT.
    """
    from scipy.spatial.distance import cdist
    plan = result.solution.to_sparse_matrix().tocoo()
    a, b = problem.source_mass, problem.target_mass
    u, v = result.solution.source_dual, result.solution.target_dual
    primal = float(np.linalg.norm(np.concatenate((
        np.asarray(plan.sum(axis=1)).ravel() - a, np.asarray(plan.sum(axis=0)).ravel() - b
    ))) / (1 + np.linalg.norm(np.concatenate((a, b)))))
    x, y = problem.source_points.astype(np.float64), problem.target_points.astype(np.float64)
    objective = float(np.sum(plan.data * np.sum((x[plan.row] - y[plan.col])**2, axis=1)))
    dual = float(a @ u + b @ v)
    gap = abs(objective-dual) / (1 + abs(objective) + abs(dual))
    num, den, maximum = 0.0, 0.0, 0.0
    for i in range(0, len(x), 256):
        for j in range(0, len(y), 256):
            cost = cdist(x[i:i+256], y[j:j+256], metric="sqeuclidean")
            violation = np.maximum(u[i:i+256, None] + v[None, j:j+256] - cost, 0)
            num += float(np.sum(violation**2))
            den += float(np.sum(cost**2))
            maximum = max(maximum, float(violation.max()))
    feasibility = num**0.5 / (1 + den**0.5)
    values = (primal, gap, feasibility, objective, dual, maximum, result.objective)
    converged = bool(all(np.isfinite(t) for t in values) and max(primal, gap, feasibility) < 1e-6
                     and np.isfinite(plan.data).all() and np.all(plan.data >= -1e-10)
                     and abs(objective-result.objective) / max(1, abs(objective)) < 1e-6)
    return converged, {
        "relative_primal_feasibility": primal, "relative_primal_dual_gap": gap,
        "relative_full_dual_feasibility": feasibility, "dual_feasibility_max_violation": maximum,
        "level0_inner_iterations": 0, "peak_active_support_size": int(plan.nnz),
    }
