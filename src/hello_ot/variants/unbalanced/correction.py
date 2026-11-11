from __future__ import annotations

import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from .dual_atoms import DualAtomMixture
from .objective import compute_translated_dual_state


def _normalized_weight_objective(
    variables: np.ndarray,
    objective: Callable[[np.ndarray], Tuple[float, np.ndarray]],
) -> Tuple[float, np.ndarray]:
    """
    CN: 将非负变量归一化为 simplex 权重，并计算链式法则梯度。
    EN: Normalize nonnegative variables to simplex weights and compute the chain-rule gradient.
    """
    total = float(np.sum(variables))
    if not np.isfinite(total) or total <= 0.0:
        return 1e100, np.zeros_like(variables)
    weights = variables / total
    value, gradient = objective(weights)
    return value, (gradient - np.dot(gradient, weights)) / total


def _minimize_atom_weights_slsqp(objective, initial_weights, *, max_iter, ftol):
    """
    CN: 保留旧 SLSQP 实现供内部数值回归，不增加公开求解器选项。
    EN: Retain the legacy SLSQP implementation for internal regression without a public solver option.
    """
    return minimize(
        objective, initial_weights, method="SLSQP", jac=True,
        bounds=[(0.0, 1.0)] * len(initial_weights),
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                     "jac": lambda w: np.ones_like(w)},
        options={"maxiter": int(max_iter), "ftol": float(ftol), "disp": False},
    )


def _minimize_atom_weights(objective, initial_weights, *, max_iter, ftol):
    """
    CN: L-BFGS-B 优化同一 simplex 问题，允许精确的零权重。
    EN: Optimize the same simplex problem with L-BFGS-B, allowing exact zero weights.
    """
    result = minimize(
        lambda z: _normalized_weight_objective(z, objective), initial_weights,
        method="L-BFGS-B", jac=True, bounds=[(0.0, 1.0)] * len(initial_weights),
        options={"maxiter": int(max_iter), "ftol": float(ftol), "gtol": float(ftol)},
    )
    total = float(np.sum(result.x))
    if np.isfinite(result.x).all() and total > 0.0:
        result.x = result.x / total
    return result


def _directional_derivative(
    alpha: float,
    source_potential: np.ndarray,
    target_potential: np.ndarray,
    source_direction: np.ndarray,
    target_direction: np.ndarray,
    log_source_mass: np.ndarray,
    log_target_mass: np.ndarray,
    rho_source: float,
    rho_target: float,
) -> float:
    state = compute_translated_dual_state(
        source_potential + float(alpha) * source_direction,
        target_potential + float(alpha) * target_direction,
        log_source_mass,
        log_target_mass,
        rho_source,
        rho_target,
        source_total=1.0,
        target_total=1.0,
    )
    return float(
        state["transported_mass"]
        * (
            np.dot(state["source_normalized"], source_direction)
            + np.dot(state["target_normalized"], target_direction)
        )
    )


def _exact_segment_search(
    source_potential: np.ndarray,
    target_potential: np.ndarray,
    source_direction: np.ndarray,
    target_direction: np.ndarray,
    alpha_max: float,
    log_source_mass: np.ndarray,
    log_target_mass: np.ndarray,
    rho_source: float,
    rho_target: float,
    *,
    max_iter: int = 80,
    tolerance: float = 1e-12,
) -> float:
    """
    CN: 在保持原子权重非负的可行线段 [0, alpha_max] 上精确最大化对偶目标。
    EN: Maximize the concave dual along the feasible line segment [0, alpha_max].
    """
    alpha_upper = float(alpha_max)
    if not np.isfinite(alpha_upper) or alpha_upper <= 0.0:
        return 0.0

    d0 = _directional_derivative(
        0.0,
        source_potential,
        target_potential,
        source_direction,
        target_direction,
        log_source_mass,
        log_target_mass,
        rho_source,
        rho_target,
    )
    if d0 <= 0.0:
        return 0.0

    d_upper = _directional_derivative(
        alpha_upper,
        source_potential,
        target_potential,
        source_direction,
        target_direction,
        log_source_mass,
        log_target_mass,
        rho_source,
        rho_target,
    )
    if d_upper >= 0.0:
        return alpha_upper

    lower = 0.0
    upper = alpha_upper
    for _ in range(int(max_iter)):
        middle = 0.5 * (lower + upper)
        dm = _directional_derivative(
            middle,
            source_potential,
            target_potential,
            source_direction,
            target_direction,
            log_source_mass,
            log_target_mass,
            rho_source,
            rho_target,
        )
        if dm >= 0.0:
            lower = middle
        else:
            upper = middle
        if upper - lower <= float(tolerance):
            break
    return float(0.5 * (lower + upper))


def fully_correct_atom_weights(
    mixture: DualAtomMixture,
    log_source_mass: np.ndarray,
    log_target_mass: np.ndarray,
    rho_source: float,
    rho_target: float,
    source_total: float,
    target_total: float,
    *,
    max_iter: int = 200,
    ftol: float = 1e-12,
) -> Dict[str, Any]:
    """
    CN: 用 L-BFGS-B 对所有 atoms 的 simplex 权重完成完全矫正。
    EN: Fully correct all atoms' simplex weights using L-BFGS-B.
    """
    k = mixture.atom_count
    if k <= 1:
        mixture.weights = np.ones(1, dtype=np.float64)
        return {"correction_iterations": 0, "correction_wall_time": 0.0}

    # CN: 1. 线段搜索初始化权重
    # EN: 1. Initialize weights using exact line segment search
    old_weights = mixture.weights[:-1]
    old_sum = float(np.sum(old_weights))
    if old_sum > 0:
        old_normalized = old_weights / old_sum
    else:
        old_normalized = np.full(k - 1, 1.0 / float(k - 1), dtype=np.float64)

    src_prev = mixture.source_atoms[:, :-1] @ old_normalized
    tgt_prev = mixture.target_atoms[:, :-1] @ old_normalized
    src_new = mixture.source_atoms[:, -1]
    tgt_new = mixture.target_atoms[:, -1]

    d_src = src_new - src_prev
    d_tgt = tgt_new - tgt_prev

    alpha_opt = _exact_segment_search(
        src_prev,
        tgt_prev,
        d_src,
        d_tgt,
        alpha_max=1.0,
        log_source_mass=log_source_mass,
        log_target_mass=log_target_mass,
        rho_source=rho_source,
        rho_target=rho_target,
    )

    initial_weights = np.empty(k, dtype=np.float64)
    initial_weights[:-1] = (1.0 - alpha_opt) * old_normalized
    initial_weights[-1] = alpha_opt

    # CN: 2. L-BFGS-B 单纯形完全矫正
    # EN: 2. L-BFGS-B fully corrective optimization on probability simplex
    source_atoms = mixture.source_atoms
    target_atoms = mixture.target_atoms

    def objective_and_gradient(weights: np.ndarray) -> Tuple[float, np.ndarray]:
        src_pot = source_atoms @ weights
        tgt_pot = target_atoms @ weights
        state = compute_translated_dual_state(
            src_pot,
            tgt_pot,
            log_source_mass,
            log_target_mass,
            rho_source,
            rho_target,
            source_total,
            target_total,
        )
        grad = (
            source_atoms.T @ state["source_gradient"]
            + target_atoms.T @ state["target_gradient"]
        )
        return -float(state["dual_objective"]), -grad

    t0 = time.perf_counter()
    opt_result = _minimize_atom_weights(
        objective_and_gradient,
        initial_weights,
        max_iter=max_iter,
        ftol=ftol,
    )
    wall_time = float(time.perf_counter() - t0)

    # CN: 仅接受有限、可行且不恶化的权重；异常退出时保留线段搜索初值。
    # EN: Accept only finite, feasible, nonworsening weights; otherwise retain the segment-search initialization.
    candidate = np.asarray(opt_result.x, dtype=np.float64)
    valid = (np.isfinite(candidate).all() and np.all(candidate >= 0.0)
             and np.isclose(candidate.sum(), 1.0, rtol=0.0, atol=1e-10))
    f_init = objective_and_gradient(initial_weights)[0]
    f_curr = objective_and_gradient(candidate)[0] if valid else float("inf")
    mixture.weights = candidate if np.isfinite(f_curr) and f_curr <= f_init else initial_weights

    mixture.normalize_weights()
    return {
        "correction_wall_time": wall_time,
        "correction_iterations": int(opt_result.nit),
        "correction_function_evals": int(opt_result.nfev),
    }


__all__ = ["fully_correct_atom_weights"]
