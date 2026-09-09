from __future__ import annotations

from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import torch


def _lower_bound_signature_one_side(
    points: np.ndarray,
    mass: np.ndarray,
    *,
    max_chunk_entries: int = 16_000_000,
) -> np.ndarray:
    """
    CN: 分块计算下界一维特征 sqrt(sum_j ||x_i-x_j||_2^4 m_j)。
    EN: Compute 1D lower-bound signature sqrt(sum_j ||x_i-x_j||_2^4 m_j) in row chunks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points_gpu = torch.as_tensor(points, dtype=torch.float32, device=device)
    mass_gpu = torch.as_tensor(mass, dtype=torch.float32, device=device).reshape(-1)

    n_points = int(points_gpu.shape[0])
    chunk_rows = max(1, min(n_points, int(max_chunk_entries) // max(n_points, 1)))

    norm_gpu = torch.sum(points_gpu * points_gpu, dim=1)
    accum_gpu = torch.empty((n_points,), device=device, dtype=torch.float32)
    target_t = points_gpu.T.contiguous()

    for start in range(0, n_points, chunk_rows):
        end = min(start + chunk_rows, n_points)
        chunk = points_gpu[start:end]
        sqdist = norm_gpu[start:end, None] + norm_gpu[None, :] - 2.0 * (chunk @ target_t)
        sqdist.clamp_(min=0.0)
        accum_gpu[start:end] = (sqdist * sqdist) @ mass_gpu

    signature = torch.sqrt(torch.clamp(accum_gpu, min=0.0)).detach().cpu().numpy()
    return np.ascontiguousarray(signature, dtype=np.float64)


def _solve_1d_ot_sorted_sparse(
    source_values: np.ndarray,
    target_values: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    *,
    atol: float = 1e-12,
) -> sp.coo_matrix:
    """
    CN: 通过排序双指针精确求解一维 OT，并返回稀疏 coupling。
    EN: Solve one-dimensional OT exactly with sorted two-pointer matching and return a sparse coupling.
    """
    s_val = np.asarray(source_values, dtype=np.float64).reshape(-1)
    t_val = np.asarray(target_values, dtype=np.float64).reshape(-1)
    a = np.asarray(source_mass, dtype=np.float64).reshape(-1)
    b = np.asarray(target_mass, dtype=np.float64).reshape(-1)

    total_a = float(a.sum())
    total_b = float(b.sum())
    mass_atol = max(float(atol), 1e-10 * max(total_a, total_b))

    source_order = np.argsort(s_val, kind="mergesort")
    target_order = np.argsort(t_val, kind="mergesort")
    source_remaining = a[source_order].copy()
    target_remaining = b[target_order].copy()

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    i = 0
    j = 0
    n_s = int(s_val.shape[0])
    n_t = int(t_val.shape[0])
    while i < n_s and j < n_t:
        while i < n_s and source_remaining[i] <= mass_atol:
            i += 1
        while j < n_t and target_remaining[j] <= mass_atol:
            j += 1
        if i >= n_s or j >= n_t:
            break
        amount = min(float(source_remaining[i]), float(target_remaining[j]))
        if amount > mass_atol:
            rows.append(int(source_order[i]))
            cols.append(int(target_order[j]))
            data.append(amount)
        source_remaining[i] -= amount
        target_remaining[j] -= amount

    return sp.coo_matrix(
        (
            np.asarray(data, dtype=np.float64),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(n_s, n_t),
    )


def build_lower_bound_sparse_init_coupling(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
) -> sp.coo_matrix:
    """
    CN: 基于 lower-bound 1D signature 精确单调匹配生成稀疏初始 coupling。
    EN: Construct sparse initial coupling from exact monotonic 1D matching on lower-bound signatures.
    """
    source_lb = _lower_bound_signature_one_side(source_points, source_mass)
    target_lb = _lower_bound_signature_one_side(target_points, target_mass)
    return _solve_1d_ot_sorted_sparse(
        source_lb,
        target_lb,
        source_mass,
        target_mass,
    )


__all__ = ["build_lower_bound_sparse_init_coupling"]
