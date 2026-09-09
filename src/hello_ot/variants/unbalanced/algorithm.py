from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np

from hello_ot.config import SolverOptions
from hello_ot.types import SparseOTSolution
from hello_ot.variants._balanced import (
    algorithm_config,
    inherited_dual_from_state,
    solve_bilinear_subproblem,
)

from .correction import fully_correct_atom_weights
from .dual_atoms import DualAtomMixture, canonicalize_atom
from .initialization import initialize_unbalanced_fcfw
from .objective import (
    compute_translated_dual_state,
    evaluate_primal_certificate,
)
from .types import UnbalancedResult


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _positive_mass(value: Any | None, *, size: int, name: str) -> np.ndarray:
    """
    CN: 校验非平衡边际质量；严禁自动归一化，保留用户传入的真实绝对质量。
    EN: Validate unbalanced mass vector; strictly forbid normalization, preserving absolute mass.
    """
    if value is None:
        return np.full(int(size), 1.0 / float(size), dtype=np.float64)
    mass = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    if mass.size != int(size):
        raise ValueError(f"{name} must have length {size}")
    if np.any(mass <= 0.0) or not np.all(np.isfinite(mass)):
        raise ValueError(f"{name} must contain only positive and finite values")
    return mass


def _sqeuclidean_factors(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    factors = points * np.float32(math.sqrt(2.0))
    offsets = np.einsum("ij,ij->i", points, points, dtype=np.float32, optimize=False)
    return np.ascontiguousarray(factors, dtype=np.float32), np.ascontiguousarray(offsets, dtype=np.float32)


def solve_unbalanced(
    source_points: Any,
    target_points: Any,
    source_mass: Optional[Any] = None,
    target_mass: Optional[Any] = None,
    *,
    rho_source: float = 1.0,
    rho_target: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    initialization: Optional[str] = "ott_sinkhorn",
    options: Optional[SolverOptions] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> UnbalancedResult:
    """
    CN: 求解基于 Fully-Corrective Frank-Wolfe (FCFW) 的 KL 非平衡最优传输。
    EN: Solve KL-unbalanced optimal transport via Fully-Corrective Frank-Wolfe (FCFW).
    """
    started_total = time.perf_counter()

    # CN: 1. 严格输入校验与特征转换
    # EN: 1. Strict input validation and feature factorization
    src_pts = _matrix(source_points, name="source_points")
    tgt_pts = _matrix(target_points, name="target_points")
    if src_pts.shape[1] != tgt_pts.shape[1]:
        raise ValueError(
            f"Point cloud feature dimensions must match; got {src_pts.shape[1]} != {tgt_pts.shape[1]}"
        )
    n_source = int(src_pts.shape[0])
    n_target = int(tgt_pts.shape[0])

    src_mass = _positive_mass(source_mass, size=n_source, name="source_mass")
    tgt_mass = _positive_mass(target_mass, size=n_target, name="target_mass")

    if float(rho_source) <= 0.0 or float(rho_target) <= 0.0:
        raise ValueError("rho_source and rho_target must be strictly positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least 1")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be strictly positive")

    source_factors, source_offset = _sqeuclidean_factors(src_pts)
    target_factors, target_offset = _sqeuclidean_factors(tgt_pts)

    cfg = algorithm_config(options)
    source_total = float(np.sum(src_mass))
    target_total = float(np.sum(tgt_mass))
    log_src_mass = np.log(src_mass)
    log_tgt_mass = np.log(tgt_mass)

    # CN: 2. 初始化 DualAtomMixture 与首个 primal certificate
    # EN: 2. Initialize DualAtomMixture and first primal certificate
    mixture, best_cert, subproblem = initialize_unbalanced_fcfw(
        source_factors=source_factors,
        target_factors=target_factors,
        source_offset=source_offset,
        target_offset=target_offset,
        source_points=src_pts,
        target_points=tgt_pts,
        source_mass=src_mass,
        target_mass=tgt_mass,
        rho_source=float(rho_source),
        rho_target=float(rho_target),
        config=cfg,
        initialization=initialization,
    )
    best_primal = float("inf") if best_cert is None else float(best_cert["primal_objective"])
    best_dual = -float("inf") if best_cert is None else float(best_cert["dual_objective"])
    backend_warm_start: Any = None if subproblem is None else subproblem.state

    records = []
    converged = False
    iterations_run = 0

    # CN: 3. FCFW 外部主循环
    # EN: 3. FCFW outer loop
    for outer_iter in range(int(max_iterations)):
        iterations_run += 1
        t_iter = time.perf_counter()

        # CN: 3a. 计算当前 dual potentials 与归一化边际
        # EN: 3a. Compute current combined potentials and normalized marginals
        f_curr, g_curr = mixture.current()
        state = compute_translated_dual_state(
            f_curr,
            g_curr,
            log_src_mass,
            log_tgt_mass,
            float(rho_source),
            float(rho_target),
            source_total,
            target_total,
        )
        current_dual = float(state["dual_objective"])
        if current_dual > best_dual:
            best_dual = current_dual

        # CN: 3b. 求解平衡 OT 子问题以生成新 Frank-Wolfe atom
        # EN: 3b. Solve balanced OT subproblem to find next Frank-Wolfe atom
        inherited = (
            inherited_dual_from_state(backend_warm_start)
            if backend_warm_start is not None
            else None
        )
        subproblem = solve_bilinear_subproblem(
            source_points=source_factors,
            target_points=target_factors,
            source_offset=source_offset,
            target_offset=target_offset,
            source_mass=state["source_normalized"],
            target_mass=state["target_normalized"],
            config=cfg,
            inherited_dual=inherited,
            preparation="preserve",
        )
        backend_warm_start = subproblem.state

        atom_src, atom_tgt = canonicalize_atom(
            subproblem.hello_result.solution.source_dual,
            subproblem.hello_result.solution.target_dual,
        )

        # CN: 3c. 方向导数与对偶 gap 检查
        # EN: 3c. Directional derivative and duality gap check
        d_src = atom_src - state["source_potential"]
        d_tgt = atom_tgt - state["target_potential"]
        fw_gap = float(
            np.dot(state["source_gradient"], d_src)
            + np.dot(state["target_gradient"], d_tgt)
        )

        # CN: 评估当前子问题的 primal certificate
        # EN: Evaluate primal certificate of current subproblem
        cert = evaluate_primal_certificate(
            subproblem.coupling,
            state["transported_mass"],
            src_mass,
            tgt_mass,
            source_factors,
            target_factors,
            source_offset,
            target_offset,
            float(rho_source),
            float(rho_target),
        )
        if cert["primal_objective"] < best_primal:
            best_primal = float(cert["primal_objective"])
            best_cert = cert

        gap = max(0.0, best_primal - best_dual)
        rel_gap = float(gap / max(abs(best_primal), abs(best_dual), 1e-15))

        rec = {
            "iteration": int(outer_iter + 1),
            "wall_time": float(time.perf_counter() - t_iter),
            "primal_objective": best_primal,
            "dual_objective": best_dual,
            "relative_primal_dual_gap": rel_gap,
            "fw_directional_gap": fw_gap,
            "atom_count": mixture.atom_count,
            "nonzero_atom_count": mixture.nonzero_atom_count,
        }
        records.append(rec)
        if progress_callback is not None:
            progress_callback(rec)

        # CN: 收敛判定
        # EN: Convergence check
        if rel_gap <= float(tolerance) or fw_gap <= float(tolerance):
            converged = True
            break

        # CN: 3d. 将新 atom 加入原子池并执行单纯形完全矫正
        # EN: 3d. Append new atom and execute fully corrective simplex optimization
        mixture.append(atom_src, atom_tgt)
        fully_correct_atom_weights(
            mixture,
            log_src_mass,
            log_tgt_mass,
            float(rho_source),
            float(rho_target),
            source_total,
            target_total,
        )

    total_wall_time = float(time.perf_counter() - started_total)

    # CN: 4. 封装结果对象
    # EN: 4. Wrap final result
    f_final, g_final = mixture.current()
    best_coo = best_cert["coupling"].tocoo(copy=False)
    solution = SparseOTSolution(
        shape=(n_source, n_target),
        rows=best_coo.row,
        cols=best_coo.col,
        values=best_coo.data,
        source_dual=f_final,
        target_dual=g_final,
    )

    final_gap = max(0.0, best_primal - best_dual)
    final_rel_gap = float(final_gap / max(abs(best_primal), abs(best_dual), 1e-15))

    return UnbalancedResult(
        objective=best_primal,
        dual_objective=best_dual,
        relative_primal_dual_gap=final_rel_gap,
        solution=solution,
        source_marginal=best_cert["source_marginal"],
        target_marginal=best_cert["target_marginal"],
        transported_mass=float(best_cert["transported_mass"]),
        total_wall_time=total_wall_time,
        iterations=iterations_run,
        converged=converged,
        records=tuple(records),
        metadata={
            "rho_source": float(rho_source),
            "rho_target": float(rho_target),
            "initialization": str(initialization),
            "backend": str(cfg.backend),
        },
    )


__all__ = ["solve_unbalanced"]
