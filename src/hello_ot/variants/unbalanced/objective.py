from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp, xlogy


def kl_divergence(transported: np.ndarray, reference: np.ndarray) -> float:
    """
    CN: 计算广义 Kullback-Leibler 散度 KL(p || q) = sum (p log(p/q) - p + q)。
    EN: Compute generalized Kullback-Leibler divergence KL(p || q) = sum (p log(p/q) - p + q).
    """
    transported_arr = np.asarray(transported, dtype=np.float64)
    reference_arr = np.asarray(reference, dtype=np.float64)
    return float(
        np.sum(
            xlogy(transported_arr, transported_arr / reference_arr)
            - transported_arr
            + reference_arr
        )
    )


def compute_translated_dual_state(
    bar_f: np.ndarray,
    bar_g: np.ndarray,
    log_source_mass: np.ndarray,
    log_target_mass: np.ndarray,
    rho_source: float,
    rho_target: float,
    source_total: float,
    target_total: float,
) -> Dict[str, Any]:
    """
    CN: 沿等价规范平移 dual potentials，计算 translation-invariant 对偶目标、边际分布与梯度。
    EN: Shift potentials along invariant gauge, computing invariant dual objective, marginals, and gradients.
    """
    rho_a = float(rho_source)
    rho_b = float(rho_target)
    log_r_before = log_source_mass - np.asarray(bar_f, dtype=np.float64) / rho_a
    log_s_before = log_target_mass - np.asarray(bar_g, dtype=np.float64) / rho_b
    log_r_sum = float(logsumexp(log_r_before))
    log_s_sum = float(logsumexp(log_s_before))
    rho_sum = float(rho_a + rho_b)

    translation = float(rho_a * rho_b / rho_sum) * (log_r_sum - log_s_sum)
    f = np.asarray(bar_f, dtype=np.float64) + translation
    g = np.asarray(bar_g, dtype=np.float64) - translation

    source_normalized = np.exp(log_r_before - log_r_sum)
    target_normalized = np.exp(log_s_before - log_s_sum)

    log_common_mass = float(rho_a / rho_sum) * log_r_sum + float(rho_b / rho_sum) * log_s_sum
    transported_mass = float(np.exp(log_common_mass))
    if not np.isfinite(transported_mass) or transported_mass <= 0.0:
        raise FloatingPointError("KL-UOT transported mass is non-positive or non-finite.")

    dual_objective = float(
        rho_a * source_total
        + rho_b * target_total
        - rho_sum * transported_mass
    )
    return {
        "source_potential": f,
        "target_potential": g,
        "source_normalized": source_normalized,
        "target_normalized": target_normalized,
        "transported_mass": transported_mass,
        "translation": translation,
        "dual_objective": dual_objective,
        "source_gradient": transported_mass * source_normalized,
        "target_gradient": transported_mass * target_normalized,
    }


def sparse_linear_cost(
    coupling: sp.spmatrix,
    source_factors: np.ndarray,
    target_factors: np.ndarray,
    source_offset: np.ndarray,
    target_offset: np.ndarray,
    dot_scale: float = 1.0,
) -> float:
    """
    CN: 在 O(nnz * d) 复杂度下精确评估稀疏 coupling 相对双线性代价的线性项 <C, P>。
    EN: Evaluate linear cost <C, P> on sparse coupling in O(nnz * d) time using bilinear factors.
    """
    coo = coupling.tocoo(copy=False)
    rows = np.asarray(coo.row, dtype=np.int64)
    cols = np.asarray(coo.col, dtype=np.int64)
    values = np.asarray(coo.data, dtype=np.float64)

    # CN: 分别累加 source / target 偏移项
    # EN: Accumulate source and target offset terms
    source_marginal = np.bincount(rows, weights=values, minlength=int(source_offset.size))
    target_marginal = np.bincount(cols, weights=values, minlength=int(target_offset.size))
    offset_cost = float(np.dot(source_marginal, source_offset)) + float(
        np.dot(target_marginal, target_offset)
    )

    # CN: 分块累加双线性内积项
    # EN: Accumulate bilinear dot product in chunks
    interaction = 0.0
    chunk_size = 65_536
    for start in range(0, int(values.size), chunk_size):
        end = min(int(values.size), start + chunk_size)
        pair_inner = np.einsum(
            "ij,ij->i",
            source_factors[rows[start:end]],
            target_factors[cols[start:end]],
            optimize=True,
        )
        interaction += float(np.dot(values[start:end], pair_inner))

    return float(offset_cost - float(dot_scale) * interaction)


def evaluate_primal_certificate(
    plan_normalized: sp.spmatrix,
    transported_mass: float,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    source_factors: np.ndarray,
    target_factors: np.ndarray,
    source_offset: np.ndarray,
    target_offset: np.ndarray,
    rho_source: float,
    rho_target: float,
    dot_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    CN: 缩放归一化 transport plan 并评估完整的原问题目标值与边际分布。
    EN: Scale normalized transport plan and compute full primal objective and marginals.
    """
    coo = plan_normalized.tocoo(copy=True)
    coo.data = coo.data * float(transported_mass)
    source_marginal = np.bincount(
        np.asarray(coo.row, dtype=np.int64),
        weights=coo.data,
        minlength=int(source_mass.size),
    )
    target_marginal = np.bincount(
        np.asarray(coo.col, dtype=np.int64),
        weights=coo.data,
        minlength=int(target_mass.size),
    )
    linear = sparse_linear_cost(
        coo,
        source_factors,
        target_factors,
        source_offset,
        target_offset,
        dot_scale=dot_scale,
    )
    source_kl = kl_divergence(source_marginal, source_mass)
    target_kl = kl_divergence(target_marginal, target_mass)
    primal_obj = float(linear + rho_source * source_kl + rho_target * target_kl)
    return {
        "coupling": coo,
        "source_marginal": source_marginal,
        "target_marginal": target_marginal,
        "transported_mass": float(transported_mass),
        "linear_cost": linear,
        "source_kl": source_kl,
        "target_kl": target_kl,
        "primal_objective": primal_obj,
    }


__all__ = [
    "compute_translated_dual_state",
    "evaluate_primal_certificate",
    "kl_divergence",
    "sparse_linear_cost",
]
