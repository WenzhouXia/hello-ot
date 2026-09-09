from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist


ROUNDING_REQUIRED_METHODS = frozenset(
    {
        "asymmetric_chain",
        "hello",
        "mdot_tnt_dense",
        "mdot_tnt_keops",
        "mdot_tnt_pypi",
        "neufeld_cutplane",
        "ott_jax_sinkhorn_l1_negdot_std",
        "pot_proximal_point",
        "zanetti_ipm",
    }
)

ROUNDED_GAP_RELATIVE_TOL = 1.0e-12
ROUNDING_VALIDATION_ATOL = 1.0e-10
ROUNDING_METHOD = "row_col_scaling_rank_one_correction"


def requires_objective_rounding(method: str) -> bool:
    """
    CN: 按方法能力判断 objective 评测是否需要 primal feasible rounding。
    EN: Decide from method capability whether objective evaluation requires primal feasible rounding.
    """
    return str(method).strip().lower() in ROUNDING_REQUIRED_METHODS


def relative_error_from_rounded_objective(
    rounded_objective: float,
    reference_objective: float,
    *,
    relative_tolerance: float = ROUNDED_GAP_RELATIVE_TOL,
) -> tuple[float | None, bool]:
    """
    CN: 计算可行 primal objective 的相对误差，并拒绝明显低于 reference 的结果。
    EN: Compute relative error for a feasible primal objective and reject values materially below the reference.
    """
    rounded = float(rounded_objective)
    reference = float(reference_objective)
    if not np.isfinite(rounded) or not np.isfinite(reference):
        return None, False
    gap = (rounded - reference) / max(abs(reference), 1.0e-12)
    if gap < -float(relative_tolerance):
        return None, False
    return float(max(gap, 0.0)), True


def rounded_transport_objective(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    transport: Any,
    cost_type: str,
    block_size: int = 256,
) -> float:
    """
    CN: 隐式执行边缘 rounding 并计算 objective，不物化 dense rounded coupling。
    EN: Round marginals implicitly and evaluate the objective without materializing a dense rounded coupling.
    """
    implicit_value = getattr(transport, "rounded_objective", None)
    if implicit_value is not None:
        return float(implicit_value)
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    source_marginal = np.asarray(source_mass, dtype=np.float64).reshape(-1)
    target_marginal = np.asarray(target_mass, dtype=np.float64).reshape(-1)
    _validate_inputs(source, target, source_marginal, target_marginal, transport)

    if sparse.issparse(transport):
        base_objective, rounded_row, rounded_col = _scaled_sparse_objective_and_marginals(
            source,
            target,
            source_marginal,
            target_marginal,
            transport,
            cost_type=str(cost_type),
        )
    else:
        base_objective, rounded_row, rounded_col = _scaled_dense_objective_and_marginals(
            source,
            target,
            source_marginal,
            target_marginal,
            np.asarray(transport),
            cost_type=str(cost_type),
            block_size=max(int(block_size), 1),
        )

    source_residual = _nonnegative_residual(source_marginal - rounded_row)
    target_residual = _nonnegative_residual(target_marginal - rounded_col)
    source_residual_mass = float(source_residual.sum())
    target_residual_mass = float(target_residual.sum())
    if abs(source_residual_mass - target_residual_mass) > ROUNDING_VALIDATION_ATOL:
        raise RuntimeError(
            "Rounding residual masses disagree in float64: "
            f"source={source_residual_mass:.16e}, target={target_residual_mass:.16e}."
        )
    if source_residual_mass <= np.finfo(np.float64).eps:
        _validate_rounded_marginals(
            rounded_row,
            rounded_col,
            source_marginal,
            target_marginal,
        )
        return float(base_objective)
    correction = _rank_one_cost(
        source,
        target,
        source_residual,
        target_residual,
        residual_mass=source_residual_mass,
        cost_type=str(cost_type),
        block_size=max(int(block_size), 1),
    )
    correction_row = source_residual * (target_residual_mass / source_residual_mass)
    correction_col = target_residual
    _validate_rounded_marginals(
        rounded_row + correction_row,
        rounded_col + correction_col,
        source_marginal,
        target_marginal,
    )
    return float(base_objective + correction)


def _validate_inputs(
    source: np.ndarray,
    target: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    transport: Any,
) -> None:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source_points and target_points must be compatible two-dimensional arrays.")
    if source_mass.shape != (source.shape[0],) or target_mass.shape != (target.shape[0],):
        raise ValueError("Marginal shapes must match point-cloud sizes.")
    if not np.isclose(source_mass.sum(), target_mass.sum(), rtol=0.0, atol=1.0e-12):
        raise ValueError("Feasible rounding requires balanced source and target masses.")
    if np.any(source_mass < 0.0) or np.any(target_mass < 0.0):
        raise ValueError("Marginals must be nonnegative.")
    if tuple(transport.shape) != (source.shape[0], target.shape[0]):
        raise ValueError("Transport shape does not match the point clouds.")


def _scaling(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = np.ones_like(numerator, dtype=np.float64)
    positive = denominator > 0.0
    ratio[positive] = numerator[positive] / denominator[positive]
    return np.minimum(ratio, 1.0)


def _scaled_sparse_objective_and_marginals(
    source: np.ndarray,
    target: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    transport: Any,
    *,
    cost_type: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    coo = transport.tocoo(copy=False)
    values = np.maximum(np.asarray(coo.data, dtype=np.float64), 0.0)
    rows = np.asarray(coo.row, dtype=np.int64)
    cols = np.asarray(coo.col, dtype=np.int64)
    row = np.bincount(rows, weights=values, minlength=source.shape[0]).astype(np.float64)
    row_scale = _scaling(source_mass, row)
    row_scaled_values = values * row_scale[rows]
    col_after_row_scale = np.bincount(
        cols,
        weights=row_scaled_values,
        minlength=target.shape[0],
    ).astype(np.float64)
    col_scale = _scaling(target_mass, col_after_row_scale)
    scaled_values = row_scaled_values * col_scale[cols]
    rounded_row = np.bincount(rows, weights=scaled_values, minlength=source.shape[0]).astype(np.float64)
    rounded_col = np.bincount(cols, weights=scaled_values, minlength=target.shape[0]).astype(np.float64)
    costs = _paired_cost(source, target, rows, cols, cost_type=cost_type)
    return float(np.dot(scaled_values, costs)), rounded_row, rounded_col


def _scaled_dense_objective_and_marginals(
    source: np.ndarray,
    target: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    transport: np.ndarray,
    *,
    cost_type: str,
    block_size: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(transport)) or np.any(transport < -1.0e-12):
        raise ValueError("Transport must be finite and nonnegative up to numerical tolerance.")
    n_source, n_target = transport.shape
    row = np.empty(n_source, dtype=np.float64)
    for start in range(0, n_source, block_size):
        stop = min(start + block_size, n_source)
        row[start:stop] = np.sum(np.maximum(transport[start:stop], 0.0), axis=1, dtype=np.float64)
    row_scale = _scaling(source_mass, row)
    col_after_row_scale = np.zeros(n_target, dtype=np.float64)
    for start in range(0, n_source, block_size):
        stop = min(start + block_size, n_source)
        block = np.maximum(np.asarray(transport[start:stop]), 0.0)
        col_after_row_scale += np.sum(block * row_scale[start:stop, None], axis=0, dtype=np.float64)
    col_scale = _scaling(target_mass, col_after_row_scale)

    rounded_row = np.empty(n_source, dtype=np.float64)
    rounded_col = np.zeros(n_target, dtype=np.float64)
    objective = 0.0
    for start in range(0, n_source, block_size):
        stop = min(start + block_size, n_source)
        block = np.maximum(np.asarray(transport[start:stop]), 0.0)
        scaled = block * row_scale[start:stop, None] * col_scale[None, :]
        rounded_row[start:stop] = np.sum(scaled, axis=1, dtype=np.float64)
        rounded_col += np.sum(scaled, axis=0, dtype=np.float64)
        objective += float(np.sum(scaled * _cost_block(source[start:stop], target, cost_type=cost_type)))
    return objective, rounded_row, rounded_col


def _nonnegative_residual(residual: np.ndarray) -> np.ndarray:
    if float(np.min(residual, initial=0.0)) < -1.0e-10:
        raise RuntimeError("Marginal scaling produced a materially negative residual.")
    return np.maximum(np.asarray(residual, dtype=np.float64), 0.0)


def _validate_rounded_marginals(
    row: np.ndarray,
    col: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
) -> None:
    """
    CN: 验证标准 rank-one correction 后的行列边缘可行性。
    EN: Validate row and column feasibility after the standard rank-one correction.
    """
    row_l1 = float(np.sum(np.abs(np.asarray(row, dtype=np.float64) - source_mass)))
    col_l1 = float(np.sum(np.abs(np.asarray(col, dtype=np.float64) - target_mass)))
    if not np.isfinite(row_l1) or not np.isfinite(col_l1):
        raise RuntimeError("Rounding produced non-finite marginal diagnostics.")
    if max(row_l1, col_l1) > ROUNDING_VALIDATION_ATOL:
        raise RuntimeError(
            "Rounded transport failed marginal validation in float64: "
            f"row_l1={row_l1:.16e}, col_l1={col_l1:.16e}."
        )


def _rank_one_cost(
    source: np.ndarray,
    target: np.ndarray,
    source_residual: np.ndarray,
    target_residual: np.ndarray,
    *,
    residual_mass: float,
    cost_type: str,
    block_size: int,
) -> float:
    if cost_type == "l2^2":
        source_norm = np.einsum("ij,ij->i", source, source)
        target_norm = np.einsum("ij,ij->i", target, target)
        numerator = (
            float(np.dot(source_residual, source_norm)) * float(target_residual.sum())
            + float(source_residual.sum()) * float(np.dot(target_residual, target_norm))
            - 2.0
            * float(np.dot(source_residual @ source, target_residual @ target))
        )
        return float(numerator / residual_mass)

    numerator = 0.0
    for start in range(0, source.shape[0], block_size):
        stop = min(start + block_size, source.shape[0])
        costs = _cost_block(source[start:stop], target, cost_type=cost_type)
        numerator += float(np.sum(source_residual[start:stop, None] * costs * target_residual[None, :]))
    return float(numerator / residual_mass)


def _cost_block(source: np.ndarray, target: np.ndarray, *, cost_type: str) -> np.ndarray:
    if cost_type == "l2^2":
        source_norm = np.einsum("ij,ij->i", source, source)
        target_norm = np.einsum("ij,ij->i", target, target)
        return np.maximum(source_norm[:, None] + target_norm[None, :] - 2.0 * (source @ target.T), 0.0)
    if cost_type == "l2":
        return np.asarray(cdist(source, target, metric="euclidean"), dtype=np.float64)
    if cost_type == "l1":
        return np.asarray(cdist(source, target, metric="cityblock"), dtype=np.float64)
    if cost_type == "linf":
        return np.asarray(cdist(source, target, metric="chebyshev"), dtype=np.float64)
    raise ValueError(f"Unsupported cost_type: {cost_type}")


def _paired_cost(
    source: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    cost_type: str,
) -> np.ndarray:
    diff = source[rows] - target[cols]
    if cost_type == "l2^2":
        return np.einsum("ij,ij->i", diff, diff)
    if cost_type == "l2":
        return np.linalg.norm(diff, ord=2, axis=1)
    if cost_type == "l1":
        return np.linalg.norm(diff, ord=1, axis=1)
    if cost_type == "linf":
        return np.linalg.norm(diff, ord=np.inf, axis=1)
    raise ValueError(f"Unsupported cost_type: {cost_type}")
