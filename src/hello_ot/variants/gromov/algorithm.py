from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

from hello_ot.config import SolverOptions
from hello_ot.types import SparseOTSolution
from hello_ot.variants._balanced import (
    algorithm_config,
    inherited_dual_from_state,
    solve_bilinear_subproblem,
)
from hello_ot.variants._validation import matrix, probability_mass

from .initialization import build_lower_bound_sparse_init_coupling
from .moments import (
    compute_sqeuclidean_gw_objective,
    compute_sqeuclidean_gw_static_terms,
    linearized_sqeuclidean_gw_factors,
)
from .types import GromovIterationRecord, GromovResult


ProgressCallback = Callable[[GromovIterationRecord], None]


def solve_gromov(
    source_points: Any,
    target_points: Any,
    source_mass: Optional[Any] = None,
    target_mass: Optional[Any] = None,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    options: Optional[SolverOptions] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> GromovResult:
    """
    CN: 求解基于低秩张量分解与 full-step 外层更新的平方欧氏 Gromov-Wasserstein (GW)。
    EN: Solve squared-Euclidean Gromov-Wasserstein (GW) via low-rank factorization and full-step updates.
    """
    started_total = time.perf_counter()

    # CN: 1. 严格输入校验
    # EN: 1. Strict input validation
    src_pts = matrix(source_points, name="source_points")
    tgt_pts = matrix(target_points, name="target_points")
    n_source = int(src_pts.shape[0])
    n_target = int(tgt_pts.shape[0])

    src_mass = probability_mass(source_mass, size=n_source, name="source_mass")
    tgt_mass = probability_mass(target_mass, size=n_target, name="target_mass")

    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least 1")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be strictly positive")

    cfg = algorithm_config(options)

    # CN: 2. 预计算静态项并使用 lower_bound_sparse 初始化
    # EN: 2. Precompute static terms and initialize via lower_bound_sparse
    static_terms = compute_sqeuclidean_gw_static_terms(src_pts, tgt_pts, src_mass, tgt_mass)
    current_coupling = build_lower_bound_sparse_init_coupling(src_pts, tgt_pts, src_mass, tgt_mass)
    current_obj = compute_sqeuclidean_gw_objective(static_terms, src_mass, tgt_mass, current_coupling)

    backend_warm_start = None
    records: list[GromovIterationRecord] = []
    converged = False
    iterations_run = 0
    last_subproblem = None

    # CN: 3. GW full-step 外层迭代主循环
    # EN: 3. GW full-step outer iteration loop
    for outer_iter in range(int(max_iterations)):
        iterations_run += 1
        t_iter = time.perf_counter()

        # CN: 3a. 在当前 coupling 处线性化二次 GW 代价
        # EN: 3a. Linearize quadratic GW cost around current coupling
        f, g, a_vec, b_vec = linearized_sqeuclidean_gw_factors(
            src_pts,
            tgt_pts,
            current_coupling,
            src_mass,
            tgt_mass,
        )

        # CN: 3b. 求解双线性平衡 OT 子问题
        # EN: 3b. Solve bilinear balanced OT subproblem
        inherited = (
            inherited_dual_from_state(backend_warm_start)
            if backend_warm_start is not None
            else None
        )
        subproblem = solve_bilinear_subproblem(
            source_points=f,
            target_points=g,
            source_offset=a_vec,
            target_offset=b_vec,
            source_mass=src_mass,
            target_mass=tgt_mass,
            config=cfg,
            inherited_dual=inherited,
            preparation="preserve",
        )
        last_subproblem = subproblem
        backend_warm_start = subproblem.state

        # CN: 3c. Full-step 更新当前 coupling 并评估真实 GW 目标值
        # EN: 3c. Full-step update current coupling and evaluate true GW objective
        next_coupling = subproblem.coupling
        next_obj = compute_sqeuclidean_gw_objective(static_terms, src_mass, tgt_mass, next_coupling)
        rel_change = float(abs(next_obj - current_obj) / max(abs(current_obj), 1.0))

        current_coupling = next_coupling
        current_obj = next_obj

        rec = GromovIterationRecord(
            iteration=int(outer_iter + 1),
            wall_time=float(time.perf_counter() - t_iter),
            objective=float(current_obj),
            relative_objective_change=rel_change,
            sparse_nnz=int(current_coupling.nnz),
        )
        records.append(rec)
        if progress_callback is not None:
            progress_callback(rec)

        if rel_change <= float(tolerance):
            converged = True
            break

    total_wall_time = float(time.perf_counter() - started_total)

    # CN: 4. 封装 GromovResult
    # EN: 4. Wrap GromovResult
    src_dual = (
        last_subproblem.hello_result.solution.source_dual
        if last_subproblem is not None
        else np.zeros(n_source, dtype=np.float64)
    )
    tgt_dual = (
        last_subproblem.hello_result.solution.target_dual
        if last_subproblem is not None
        else np.zeros(n_target, dtype=np.float64)
    )
    solution = SparseOTSolution(
        shape=(n_source, n_target),
        rows=current_coupling.row,
        cols=current_coupling.col,
        values=current_coupling.data,
        source_dual=src_dual,
        target_dual=tgt_dual,
    )

    return GromovResult(
        objective=float(current_obj),
        solution=solution,
        total_wall_time=total_wall_time,
        iterations=iterations_run,
        converged=converged,
        records=tuple(records),
        metadata={
            "initialization": "lower_bound_sparse",
            "outer_update": "full_step",
            "backend": str(cfg.backend),
        },
    )


__all__ = ["solve_gromov"]
