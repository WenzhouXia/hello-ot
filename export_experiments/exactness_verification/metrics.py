"""CN: Exactness 实验的稀疏 coupling 指标。EN: Sparse-coupling metrics for exactness export_experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import sparse


def evaluate_sparse_coupling(
    coupling: sparse.spmatrix,
    *,
    ground_truth_cols: np.ndarray,
    algorithm_objective: float,
    ground_truth_objective: float,
    support_threshold: float,
    algorithm_mapping: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    CN: 相对 permutation ground truth 评估稀疏 coupling。
    EN: Evaluate a sparse coupling against permutation ground truth.
    """
    ground_truth_cols = np.asarray(ground_truth_cols, dtype=np.int64)
    n_value = int(ground_truth_cols.size)
    coo = coupling.tocoo(copy=False)
    keep = np.asarray(coo.data, dtype=np.float64) > float(support_threshold)
    rows = np.asarray(coo.row, dtype=np.int64)[keep]
    cols = np.asarray(coo.col, dtype=np.int64)[keep]
    values = np.asarray(coo.data, dtype=np.float64)[keep]
    keys = np.unique(rows * n_value + cols)
    ground_truth_keys = np.arange(n_value, dtype=np.int64) * n_value + ground_truth_cols
    hits = int(np.intersect1d(keys, ground_truth_keys, assume_unique=True).size)

    row_argmax = _row_argmax(rows, cols, values, n_value)
    row_hits = int(np.count_nonzero(row_argmax == ground_truth_cols))
    strict = _strict_mapping_metrics(algorithm_mapping, coupling, ground_truth_cols, support_threshold)

    ground_truth_mass = 1.0 / float(n_value)
    coupling64 = coupling.astype(np.float64)
    matrix_sq_sum = float(coupling64.multiply(coupling64).sum())
    csr = coupling64.tocsr()
    overlap = float(csr[np.arange(n_value), ground_truth_cols].sum())
    difference_sq = max(matrix_sq_sum + 1.0 / float(n_value) - 2.0 * ground_truth_mass * overlap, 0.0)
    difference_norm = math.sqrt(difference_sq)
    ground_truth_norm = 1.0 / math.sqrt(float(n_value))

    row_mass = np.asarray(coupling64.sum(axis=1)).reshape(-1)
    col_mass = np.asarray(coupling64.sum(axis=0)).reshape(-1)
    expected = np.full(n_value, ground_truth_mass, dtype=np.float64)
    return {
        "relative_objective_error": abs(float(algorithm_objective) - float(ground_truth_objective))
        / max(abs(float(ground_truth_objective)), 1.0e-12),
        "support_recall": float(hits / n_value),
        "support_precision": float(hits / keys.size) if keys.size else float("nan"),
        "support_hits": hits,
        "algorithm_support_size": int(keys.size),
        "row_argmax_recall": float(row_hits / n_value),
        "row_argmax_hits": row_hits,
        "row_argmax_size": int(np.count_nonzero(row_argmax >= 0)),
        **strict,
        "primal_frobenius_relative_error": difference_norm / ground_truth_norm,
        "primal_frobenius_error": difference_norm,
        "ground_truth_frobenius_norm": ground_truth_norm,
        "row_marginal_l2_error": float(np.linalg.norm(row_mass - expected)),
        "col_marginal_l2_error": float(np.linalg.norm(col_mass - expected)),
    }


def _row_argmax(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, n_value: int) -> np.ndarray:
    output = np.full(n_value, -1, dtype=np.int64)
    best = np.full(n_value, -np.inf, dtype=np.float64)
    for row, col, value in zip(rows, cols, values):
        if value > best[row]:
            best[row] = value
            output[row] = col
    return output


def _strict_mapping_metrics(
    mapping: np.ndarray | None,
    coupling: sparse.spmatrix,
    ground_truth_cols: np.ndarray,
    support_threshold: float,
) -> dict[str, Any]:
    n_value = int(ground_truth_cols.size)
    if mapping is None:
        coo = coupling.tocoo(copy=False)
        keep = np.asarray(coo.data, dtype=np.float64) > float(support_threshold)
        costs = 1.0 + float(np.max(coo.data[keep], initial=0.0)) - np.asarray(coo.data)[keep]
        graph = sparse.csr_matrix((costs, (np.asarray(coo.row)[keep], np.asarray(coo.col)[keep])), shape=coupling.shape)
        try:
            from scipy.sparse.csgraph import min_weight_full_bipartite_matching

            matched_rows, matched_cols = min_weight_full_bipartite_matching(graph)
            mapping = np.column_stack([matched_rows, matched_cols])
        except (ValueError, TypeError):
            return _blank_strict("no_perfect_matching")
    mapping = np.asarray(mapping, dtype=np.int64)
    if mapping.ndim != 2 or mapping.shape != (n_value, 2):
        return _blank_strict("invalid_mapping")
    order = np.argsort(mapping[:, 0])
    rows = mapping[order, 0]
    cols = mapping[order, 1]
    if not np.array_equal(rows, np.arange(n_value)) or np.unique(cols).size != n_value:
        return _blank_strict("invalid_mapping")
    hits = int(np.count_nonzero(cols == ground_truth_cols))
    return {
        "matching_1to1_recall": float(hits / n_value),
        "matching_1to1_hits": hits,
        "matching_1to1_size": n_value,
        "matching_1to1_status": "ok",
    }


def _blank_strict(status: str) -> dict[str, Any]:
    return {
        "matching_1to1_recall": None,
        "matching_1to1_hits": None,
        "matching_1to1_size": None,
        "matching_1to1_status": status,
    }
