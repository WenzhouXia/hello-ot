from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import sparse

from .problem import LinearOTProblem
from .result import TransportEvaluation


def evaluate_transport(
    problem: LinearOTProblem,
    *,
    transport_kind: str,
    transport: Any,
    dense_block_size: int = 256,
) -> TransportEvaluation:
    """
    CN: 不物化完整 cost matrix，统一重算 transport objective 和 L2 primal feasibility。
    EN: Recompute transport objective and L2 primal feasibility without materializing the full cost matrix.
    """
    kind = str(transport_kind).strip().lower()
    if kind == "implicit":
        evaluation = getattr(transport, "evaluation", None)
        if not isinstance(evaluation, TransportEvaluation):
            raise TypeError("implicit transport must carry a TransportEvaluation.")
        if evaluation.transport_shape != problem.shape:
            raise ValueError(
                f"implicit transport has shape {evaluation.transport_shape}, expected {problem.shape}."
            )
        return evaluation
    if kind == "dense":
        objective, row_marginal, col_marginal, nnz = _evaluate_dense(
            problem,
            np.asarray(transport),
            block_size=int(dense_block_size),
        )
    elif kind == "sparse":
        objective, row_marginal, col_marginal, nnz = _evaluate_sparse(problem, transport)
    elif kind == "permutation":
        objective, row_marginal, col_marginal, nnz = _evaluate_permutation(problem, transport)
    else:
        raise ValueError(f"Unsupported transport_kind: {transport_kind}")

    source_residual = row_marginal - problem.source_mass
    target_residual = col_marginal - problem.target_mass
    row_l2 = float(np.linalg.norm(source_residual))
    col_l2 = float(np.linalg.norm(target_residual))
    abs_error = float(math.sqrt(row_l2 * row_l2 + col_l2 * col_l2))
    bound_norm = float(np.linalg.norm(np.concatenate([problem.source_mass, problem.target_mass])))
    return TransportEvaluation(
        objective=float(objective),
        primal_feasibility=float(abs_error / (1.0 + bound_norm)),
        primal_l2_abs_error=abs_error,
        row_marginal_l2_error=row_l2,
        col_marginal_l2_error=col_l2,
        transport_mass_error=float(abs(float(col_marginal.sum()) - float(problem.source_mass.sum()))),
        transport_kind=kind,
        transport_shape=problem.shape,
        transport_nnz=int(nnz),
    )


def _evaluate_dense(
    problem: LinearOTProblem,
    plan: np.ndarray,
    *,
    block_size: int,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    matrix = np.asarray(plan, dtype=np.float64, order="C")
    if matrix.shape != problem.shape:
        raise ValueError(f"dense transport has shape {matrix.shape}, expected {problem.shape}.")
    if np.any(matrix < -1.0e-12) or not np.all(np.isfinite(matrix)):
        raise ValueError("dense transport must be finite and nonnegative up to numerical tolerance.")
    matrix = np.maximum(matrix, 0.0)
    objective = 0.0
    for start in range(0, matrix.shape[0], max(int(block_size), 1)):
        stop = min(start + max(int(block_size), 1), matrix.shape[0])
        objective += float(np.sum(matrix[start:stop] * _cost_block(problem, start, stop)))
    return objective, matrix.sum(axis=1), matrix.sum(axis=0), int(np.count_nonzero(matrix))


def _evaluate_sparse(
    problem: LinearOTProblem,
    plan: Any,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    matrix = plan.tocoo(copy=False) if sparse.issparse(plan) else sparse.coo_matrix(plan)
    if matrix.shape != problem.shape:
        raise ValueError(f"sparse transport has shape {matrix.shape}, expected {problem.shape}.")
    values = np.asarray(matrix.data, dtype=np.float64)
    if np.any(values < -1.0e-12) or not np.all(np.isfinite(values)):
        raise ValueError("sparse transport must be finite and nonnegative up to numerical tolerance.")
    rows = np.asarray(matrix.row, dtype=np.int64)
    cols = np.asarray(matrix.col, dtype=np.int64)
    costs = _paired_cost(problem, rows, cols)
    objective = float(np.dot(np.maximum(values, 0.0), costs))
    csr = matrix.tocsr()
    return (
        objective,
        np.asarray(csr.sum(axis=1), dtype=np.float64).reshape(-1),
        np.asarray(csr.sum(axis=0), dtype=np.float64).reshape(-1),
        int(csr.nnz),
    )


def _evaluate_permutation(
    problem: LinearOTProblem,
    mapping: Any,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    array = np.asarray(mapping, dtype=np.int64)
    if array.ndim == 1:
        rows = np.arange(array.size, dtype=np.int64)
        cols = array
    elif array.ndim == 2 and array.shape[1] == 2:
        rows = np.asarray(array[:, 0], dtype=np.int64)
        cols = np.asarray(array[:, 1], dtype=np.int64)
    else:
        raise ValueError("permutation transport must be a column vector or a two-column mapping.")
    n_source, n_target = problem.shape
    if np.any(rows < 0) or np.any(rows >= n_source) or np.any(cols < 0) or np.any(cols >= n_target):
        raise ValueError("permutation transport contains an out-of-range index.")
    weights = problem.source_mass[rows]
    objective = float(np.dot(weights, _paired_cost(problem, rows, cols)))
    row_marginal = np.bincount(rows, weights=weights, minlength=n_source).astype(np.float64)
    col_marginal = np.bincount(cols, weights=weights, minlength=n_target).astype(np.float64)
    return objective, row_marginal, col_marginal, int(rows.size)


def _cost_block(problem: LinearOTProblem, start: int, stop: int) -> np.ndarray:
    source = np.asarray(problem.source_points[start:stop], dtype=np.float64)
    target = np.asarray(problem.target_points, dtype=np.float64)
    if problem.cost_type == "l2^2":
        source_norm = np.sum(source * source, axis=1)
        target_norm = np.sum(target * target, axis=1)
        cost = source_norm[:, None] + target_norm[None, :] - 2.0 * (source @ target.T)
        return np.maximum(cost, 0.0)
    diff = source[:, None, :] - target[None, :, :]
    if problem.cost_type == "l2":
        return np.linalg.norm(diff, ord=2, axis=2)
    return np.linalg.norm(diff, ord=1, axis=2)


def _paired_cost(problem: LinearOTProblem, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    source = np.asarray(problem.source_points[rows], dtype=np.float64)
    target = np.asarray(problem.target_points[cols], dtype=np.float64)
    diff = source - target
    if problem.cost_type == "l2^2":
        return np.einsum("ij,ij->i", diff, diff)
    if problem.cost_type == "l2":
        return np.linalg.norm(diff, ord=2, axis=1)
    return np.linalg.norm(diff, ord=1, axis=1)
