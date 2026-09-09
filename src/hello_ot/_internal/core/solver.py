from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple
from scipy.sparse import csc_matrix, coo_matrix
import numba
from numba import types
import torch
import time
import numpy as np
import logging
from ..types.base import BaseHierarchy, BaseStrategy
from ..lp.wrapper import LPSolver
from ..trace import _ChromeTraceCollector
from hello_ot.restricted_ot.active_support import ActiveSupport


logger = logging.getLogger(__name__)


@numba.jit(types.Tuple((types.int32[:], types.int32[:]))(types.float64[:], types.float64[:]), nopython=True, cache=True)
def _northwest_corner_numba(p_in, q_in):
    """
    使用 Numba 加速的西北角法。
    返回 rows, cols 两个数组。
    """
    # 拷贝一份避免修改原数据（Numba中 float64 数组拷贝很快）
    p = p_in.copy()
    q = q_in.copy()

    m = p.shape[0]
    n = q.shape[0]

    # 预分配最大可能的空间 (m + n)
    # 实际通常是 m + n - 1
    max_edges = m + n + 1
    rows = np.empty(max_edges, dtype=np.int32)
    cols = np.empty(max_edges, dtype=np.int32)

    i = 0
    j = 0
    count = 0

    # 简单的容差
    tol = 1e-20

    while i < m and j < n:
        # 记录当前边
        rows[count] = i
        cols[count] = j
        count += 1

        val = min(p[i], q[j])
        p[i] -= val
        q[j] -= val

        # 判断移动方向
        p_empty = p[i] < tol
        q_empty = q[j] < tol

        if p_empty and q_empty:
            # 同时耗尽：为了防止跳过节点导致断链，
            # 我们优先移动一个方向，这里选择优先向下 (i+1)
            # 下一次循环 q[j] 依然是 0 (或 <tol)，会触发 j+1
            i += 1
            # 注意：这里只移动 i，不要同时移动 j，
            # 否则可能会在矩阵中形成“断点”
        elif p_empty:
            i += 1
        else:
            j += 1

    # 截取有效部分返回
    return rows[:count], cols[:count]


fill_x_prev_sig = types.float32[:](
    types.int32[:],  # primal_rows
    types.int32[:],  # primal_cols
    types.float32[:],  # primal_data
    types.int64[:],  # sorted_inc_keys (必须是64位，防止线性化坐标溢出)
    types.int64[:],  # sorted_inc_pos
    types.float32[:],  # x_prev_filled
    types.int64     # num_cols
)


@numba.jit(fill_x_prev_sig, nopython=True, fastmath=True, parallel=True, cache=True)
def _fill_x_prev_numba(primal_rows, primal_cols, primal_data,
                       sorted_inc_keys, sorted_inc_pos, x_prev_filled, num_cols):
    """
    Fills the warm-start vector `x_prev_filled` using a JIT-compiled parallel loop.
    It performs a fast binary search for each element of the refined solution
    within the active support set's lookup table.
    """
    num_primal = len(primal_rows)
    # numba.prange enables parallel execution across your CPU cores
    for i in numba.prange(num_primal):
        # 1. Linearize the (row, col) coordinate to a single key
        key_to_find = np.int64(primal_rows[i]) * num_cols + primal_cols[i]

        # 2. Perform a highly efficient binary search on the sorted lookup table
        index = np.searchsorted(sorted_inc_keys, key_to_find)

        # 3. Verify that the key was actually found
        if index < len(sorted_inc_keys) and sorted_inc_keys[index] == key_to_find:
            # 4. If found, get the original position and assign the value
            pos_to_update = sorted_inc_pos[index]
            value_to_assign = primal_data[i]
            x_prev_filled[pos_to_update] = value_to_assign

    return x_prev_filled


_expand_kernel_sig = types.void(
    types.int32[:],   # coarse_rows
    types.int32[:],   # coarse_cols
    types.float64[:],  # coarse_data
    types.int32[:],   # s_ptr (由 bincount 或大矩阵推导，通常为 int64)
    types.int32[:],   # s_indices
    types.float64[:],  # s_weights
    types.int32[:],   # t_ptr
    types.int32[:],   # t_indices
    types.float64[:],  # t_weights
    types.int64[:],   # edge_offsets (关键：必须 64 位防止溢出)
    types.int32[:],   # out_rows
    types.int32[:],   # out_cols
    types.float32[:]  # out_data
)


@numba.njit(_expand_kernel_sig, parallel=True, cache=True)
def _numba_expand_kernel(
    coarse_rows, coarse_cols, coarse_data,
    s_ptr, s_indices, s_weights,
    t_ptr, t_indices, t_weights,
    edge_offsets,
    out_rows, out_cols, out_data
):
    n_edges = len(coarse_data)
    for k in numba.prange(n_edges):
        c_r = coarse_rows[k]
        c_c = coarse_cols[k]
        val = coarse_data[k]

        s_start, s_end = s_ptr[c_r], s_ptr[c_r+1]
        t_start, t_end = t_ptr[c_c], t_ptr[c_c+1]

        write_idx = edge_offsets[k]

        for i in range(s_start, s_end):
            f_r = s_indices[i]
            w_r = s_weights[i]
            for j in range(t_start, t_end):
                f_c = t_indices[j]
                w_c = t_weights[j]

                out_rows[write_idx] = f_r
                out_cols[write_idx] = f_c
                out_data[write_idx] = val * w_r * w_c
                write_idx += 1


@numba.njit(types.int64[:](types.int32[:], types.int32[:], types.int64[:], types.int64[:]), cache=True)
def _precalc_offsets(coarse_rows, coarse_cols, s_counts, t_counts):
    n = len(coarse_rows)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    total = 0
    for i in range(n):
        # 直接使用传入的 counts
        size = s_counts[coarse_rows[i]] * t_counts[coarse_cols[i]]
        offsets[i] = total
        total += size
    offsets[n] = total
    return offsets


def _to_pinned_numpy(array: np.ndarray) -> np.ndarray:
    """
    CN: CUDA 可用时分配锁页内存；CPU-only PyTorch 使用普通连续数组。
    EN: Allocate pinned memory when CUDA is available; use an ordinary contiguous array on CPU-only PyTorch.
    """
    if not torch.cuda.is_available():
        return np.ascontiguousarray(array, dtype=np.float32)
    # 1. 创建一个 pinned 的 CPU Tensor
    tensor = torch.empty(array.shape, dtype=torch.float32, pin_memory=True)
    # 2. 将数据拷贝进去
    tensor.copy_(torch.from_numpy(array))
    # 3. 返回共享内存的 numpy 视图
    return tensor.numpy()


def _normalize_cost_type_name(cost_type: str) -> str:
    """
    统一内部 cost_type 命名（兼容旧别名）：
      lowrank / l1 / linf / l2 / l2^2
    """
    c = str(cost_type).strip().lower()
    if c in ("lowrank",):
        return "lowrank"
    if c in ("l1",):
        return "l1"
    if c in ("linf", "l_inf"):
        return "linf"
    if c in ("l2", "euclidean"):
        return "l2"
    if c in ("l2^2", "l2sq", "sqeuclidean", "sq_euclidean"):
        return "l2^2"
    raise ValueError(f"Unknown cost_type: {cost_type}")


def _prepare_level_cost_cache_sqeuclidean(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    S = np.asarray(points_s, dtype=np.float32, order='C')
    T = np.asarray(points_t, dtype=np.float32, order='C')
    s2 = np.einsum('ij,ij->i', S, S)
    t2 = np.einsum('ij,ij->i', T, T)
    return {'S': S, 'T': T, 's2': s2, 't2': t2, 'cost_type': 'l2^2'}


def _prepare_level_cost_cache_euclidean(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    """Euclidean(L2) cost: 只需要原始坐标，不需要预计算 s2/t2。"""
    S = np.asarray(points_s, dtype=np.float32, order='C')
    T = np.asarray(points_t, dtype=np.float32, order='C')
    return {'S': S, 'T': T, 'cost_type': 'l2'}


def _prepare_level_cost_cache_lowrank(
    F: np.ndarray,
    G: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    dot_scale: float = 1.0,
) -> Dict[str, Any]:
    F_mat = _to_pinned_numpy(F)
    G_mat = _to_pinned_numpy(G)
    a_vec = _to_pinned_numpy(a)
    b_vec = _to_pinned_numpy(b)

    # 返回字典，结构不变，下游代码（包括Scipy）完全无感，但在传输给GPU时会触发非阻塞加速
    scale = float(dot_scale)
    if scale not in {1.0, 2.0}:
        raise ValueError("dot_scale must be exactly 1 or 2")
    return {'F': F_mat, 'G': G_mat, 'a': a_vec, 'b': b_vec, 'dot_scale': scale, 'cost_type': 'lowrank'}


def _prepare_level_cost_cache_l1(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    """L1 cost: 只需要原始坐标，不需要预计算"""
    S = np.asarray(points_s, dtype=np.float32, order='C')
    T = np.asarray(points_t, dtype=np.float32, order='C')
    return {'S': S, 'T': T, 'cost_type': 'l1'}


def _prepare_level_cost_cache_linf(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    """Linf cost: 只需要原始坐标，不需要预计算"""
    S = np.asarray(points_s, dtype=np.float32, order='C')
    T = np.asarray(points_t, dtype=np.float32, order='C')
    return {'S': S, 'T': T, 'cost_type': 'linf'}


def _cost_pairs_l1_batched(
    level_cache,
    rows: np.ndarray,
    cols: np.ndarray,
    batch_size: int = 65536,
    free_after: bool = True
) -> np.ndarray:
    """
    L1 cost: c_ij = ||S[i] - T[j]||_1 = sum_k |S[i,k] - T[j,k]|
    """
    S, T = level_cache['S'], level_cache['T']
    rows_is_torch = torch.is_tensor(rows)
    target_device = rows.device if rows_is_torch else None
    rows_np = rows.detach().cpu().numpy() if rows_is_torch else np.asarray(rows, dtype=np.int64)
    cols_np = cols.detach().cpu().numpy() if torch.is_tensor(cols) else np.asarray(cols, dtype=np.int64)
    n = int(rows_np.shape[0])
    if n == 0:
        if rows_is_torch:
            return torch.empty(0, dtype=torch.float32, device=target_device)
        return np.array([], dtype=S.dtype)

    device = target_device if target_device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        out_cpu = np.sum(np.abs(S[rows_np] - T[cols_np]), axis=1).astype(S.dtype, copy=False)
        if rows_is_torch:
            return torch.as_tensor(out_cpu, dtype=torch.float32, device=target_device)
        return out_cpu

    # GPU 计算
    S_gpu = torch.from_numpy(np.ascontiguousarray(S)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    T_gpu = torch.from_numpy(np.ascontiguousarray(T)).to(
        device=device, dtype=torch.float32, non_blocking=False)

    out = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        rows_b = rows_np[i:j]
        cols_b = cols_np[i:j]

        rows_t = torch.from_numpy(rows_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)
        cols_t = torch.from_numpy(cols_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)

        Si = S_gpu.index_select(0, rows_t)  # (B, d)
        Tj = T_gpu.index_select(0, cols_t)  # (B, d)
        l1_dist = torch.sum(torch.abs(Si - Tj), dim=1)  # (B,)

        out[i:j] = l1_dist.detach().cpu().numpy()
        i = j
        del rows_t, cols_t, Si, Tj, l1_dist

    if free_after and device.type == 'cuda':
        del S_gpu, T_gpu
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    out_cpu = out.astype(S.dtype, copy=False)
    if rows_is_torch:
        return torch.as_tensor(out_cpu, dtype=torch.float32, device=target_device)
    return out_cpu


def _cost_pairs_linf_batched(
    level_cache,
    rows: np.ndarray,
    cols: np.ndarray,
    batch_size: int = 65536,
    free_after: bool = True
) -> np.ndarray:
    """
    Linf cost: c_ij = ||S[i] - T[j]||_inf = max_k |S[i,k] - T[j,k]|
    """
    S, T = level_cache['S'], level_cache['T']
    rows_is_torch = torch.is_tensor(rows)
    target_device = rows.device if rows_is_torch else None
    rows_np = rows.detach().cpu().numpy() if rows_is_torch else np.asarray(rows, dtype=np.int64)
    cols_np = cols.detach().cpu().numpy() if torch.is_tensor(cols) else np.asarray(cols, dtype=np.int64)
    n = int(rows_np.shape[0])
    if n == 0:
        if rows_is_torch:
            return torch.empty(0, dtype=torch.float32, device=target_device)
        return np.array([], dtype=S.dtype)

    device = target_device if target_device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        out_cpu = np.max(np.abs(S[rows_np] - T[cols_np]), axis=1).astype(S.dtype, copy=False)
        if rows_is_torch:
            return torch.as_tensor(out_cpu, dtype=torch.float32, device=target_device)
        return out_cpu

    # GPU 计算
    S_gpu = torch.from_numpy(np.ascontiguousarray(S)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    T_gpu = torch.from_numpy(np.ascontiguousarray(T)).to(
        device=device, dtype=torch.float32, non_blocking=False)

    out = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        rows_b = rows_np[i:j]
        cols_b = cols_np[i:j]

        rows_t = torch.from_numpy(rows_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)
        cols_t = torch.from_numpy(cols_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)

        Si = S_gpu.index_select(0, rows_t)  # (B, d)
        Tj = T_gpu.index_select(0, cols_t)  # (B, d)
        linf_dist = torch.max(torch.abs(Si - Tj), dim=1)[0]  # (B,)

        out[i:j] = linf_dist.detach().cpu().numpy()
        i = j
        del rows_t, cols_t, Si, Tj, linf_dist

    if free_after and device.type == 'cuda':
        del S_gpu, T_gpu
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    out_cpu = out.astype(S.dtype, copy=False)
    if rows_is_torch:
        return torch.as_tensor(out_cpu, dtype=torch.float32, device=target_device)
    return out_cpu


def _cost_pairs_lowrank_batched(
    level_cache,
    rows: np.ndarray,
    cols: np.ndarray,
    batch_size: int = 65536,
    d_chunk: int = None,
    free_after: bool = True,
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_args: Optional[Dict[str, Any]] = None,
    empty_cache_after: bool = False,
    copy_back_once: bool = True,
    support_order_cache: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    低秩成本:
      c_ij = a[i] + b[j] - <F[i], G[j]>

    参数
    ----
    level_cache: dict，包含 numpy 数组：
        F: (Ns, d) float32/float64
        G: (Nt, d)
        a: (Ns,)
        b: (Nt,)
    rows, cols: (n,) int64/int32 索引
    batch_size: SDDMM 每个 source-row block 的最大 active edge 数
    d_chunk:    兼容旧接口；SDDMM GPU 路径不再按特征维分块
    free_after: 计算完是否释放 GPU 缓存（进入 LP 前建议 True）

    返回
    ----
    c_add_all: (n,) 与 level_cache['F'] 同 dtype 的 numpy 向量
    """
    F, G, a, b = level_cache['F'], level_cache['G'], level_cache['a'], level_cache['b']
    dot_scale = float(level_cache.get('dot_scale', 1.0))
    rows_is_torch = torch.is_tensor(rows)
    cols_is_torch = torch.is_tensor(cols)
    target_device = rows.device if rows_is_torch else None
    if rows_is_torch and target_device is not None:
        device = target_device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rows_np = None
    cols_np = None
    rows_t_all = None
    cols_t_all = None
    with (tracer.span("hierarchy.internal_batch_compute_cost_vec.prepare_indices", "hierarchy", args=trace_args) if tracer is not None else nullcontext()):
        if rows_is_torch and device.type == 'cuda':
            rows_t_all = rows.detach()
            if rows_t_all.device != device:
                rows_t_all = rows_t_all.to(device=device)
            rows_t_all = rows_t_all.to(dtype=torch.long).contiguous().view(-1)
        else:
            rows_np = rows.detach().cpu().numpy() if rows_is_torch else np.asarray(rows, dtype=np.int64)
            rows_np = np.asarray(rows_np, dtype=np.int64).reshape(-1)

        if cols_is_torch and device.type == 'cuda':
            cols_t_all = cols.detach()
            if cols_t_all.device != device:
                cols_t_all = cols_t_all.to(device=device)
            cols_t_all = cols_t_all.to(dtype=torch.long).contiguous().view(-1)
        else:
            cols_np = cols.detach().cpu().numpy() if cols_is_torch else np.asarray(cols, dtype=np.int64)
            cols_np = np.asarray(cols_np, dtype=np.int64).reshape(-1)

        if rows_t_all is not None:
            n = int(rows_t_all.numel())
        else:
            n = int(rows_np.shape[0])
        if cols_t_all is not None:
            cols_n = int(cols_t_all.numel())
        else:
            cols_n = int(cols_np.shape[0])
        if int(n) != int(cols_n):
            raise ValueError("rows and cols must have the same length.")
    if n == 0:
        if rows_is_torch:
            return torch.empty(0, dtype=torch.float32, device=target_device)
        return np.array([], dtype=F.dtype)

    # 小 d 且无 GPU：直接 CPU
    d = int(F.shape[1])
    if device.type == 'cpu':
        if rows_np is None:
            rows_np = rows.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        if cols_np is None:
            cols_np = cols.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        c_add = a[rows_np] + b[cols_np]
        dots = np.einsum('nd,nd->n', F[rows_np], G[cols_np])
        out_cpu = (c_add - dot_scale * dots).astype(F.dtype, copy=False)
        if rows_is_torch:
            return torch.as_tensor(out_cpu, dtype=torch.float32, device=target_device)
        return out_cpu

    # ---- 1) Active-set SDDMM：full target resident，source row blocks streaming ----
    if rows_np is None:
        rows_np = rows_t_all.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    if cols_np is None:
        cols_np = cols_t_all.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)

    prepare_args = trace_args
    if trace_args is not None:
        prepare_args = {**trace_args, "order_cache_used": bool(support_order_cache is not None)}
    with (tracer.span("hierarchy.internal_batch_compute_cost_vec.sddmm_prepare_csr", "hierarchy", args=prepare_args) if tracer is not None else nullcontext()):
        cache_usable = (
            support_order_cache is not None
            and int(support_order_cache.get("n_vars", -1)) == int(n)
            and "source_order_cpu" in support_order_cache
            and "source_rowptr_cpu" in support_order_cache
            and "source_nonempty_cpu" in support_order_cache
        )
        if cache_usable:
            perm = np.asarray(support_order_cache["source_order_cpu"], dtype=np.int64)
            full_rowptr = np.asarray(support_order_cache["source_rowptr_cpu"], dtype=np.int64)
            unique_rows = np.asarray(support_order_cache["source_nonempty_cpu"], dtype=np.int64)
            cols_sorted = cols_np[perm]
            counts = (full_rowptr[unique_rows + 1] - full_rowptr[unique_rows]).astype(np.int64, copy=False)
            rowptr = np.empty(int(unique_rows.shape[0]) + 1, dtype=np.int64)
            rowptr[0] = 0
            np.cumsum(counts, out=rowptr[1:])
        else:
            perm = np.argsort(rows_np, kind="stable")
            rows_sorted = rows_np[perm]
            cols_sorted = cols_np[perm]
            unique_rows, counts = np.unique(rows_sorted, return_counts=True)
            rowptr = np.empty(int(unique_rows.shape[0]) + 1, dtype=np.int64)
            rowptr[0] = 0
            np.cumsum(counts.astype(np.int64, copy=False), out=rowptr[1:])
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(n, dtype=np.int64)

    with (tracer.span("hierarchy.internal_batch_compute_cost_vec.sddmm_upload_target", "hierarchy", args=trace_args) if tracer is not None else nullcontext()):
        G_gpu = torch.from_numpy(np.ascontiguousarray(G, dtype=np.float32)).to(
            device=device, dtype=torch.float32, non_blocking=True)
        b_gpu = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).to(
            device=device, dtype=torch.float32, non_blocking=True)

    out_sorted = torch.empty(n, dtype=torch.float32, device=device)
    max_nnz_per_block = max(1, int(batch_size))
    source_block_rows = max(1, min(8192, max(1024, max_nnz_per_block // 32)))
    U = int(unique_rows.shape[0])
    p = 0
    while p < U:
        q = p
        while (
            q < U
            and (q - p) < source_block_rows
            and int(rowptr[q + 1] - rowptr[p]) <= max_nnz_per_block
        ):
            q += 1
        if q == p:
            q = p + 1

        base = int(rowptr[p])
        end = int(rowptr[q])
        block_args = None
        if trace_args is not None:
            block_args = {
                **trace_args,
                "source_block_rows": int(q - p),
                "block_nnz": int(end - base),
                "sddmm_full_target_resident": True,
            }
        with (tracer.span("hierarchy.internal_batch_compute_cost_vec.sddmm_block_upload_source", "hierarchy", args=block_args) if tracer is not None else nullcontext()):
            rows_block_np = np.asarray(unique_rows[p:q], dtype=np.int64)
            cols_block_np = np.asarray(cols_sorted[base:end], dtype=np.int64)
            counts_block = np.diff(rowptr[p:q + 1]).astype(np.int64, copy=False)
            crow_np = np.empty(int(q - p) + 1, dtype=np.int64)
            crow_np[0] = 0
            np.cumsum(counts_block, out=crow_np[1:])
            F_block = torch.from_numpy(np.ascontiguousarray(F[rows_block_np], dtype=np.float32)).to(
                device=device, dtype=torch.float32, non_blocking=True)
            a_block = torch.from_numpy(np.ascontiguousarray(a[rows_block_np], dtype=np.float32)).to(
                device=device, dtype=torch.float32, non_blocking=True)
            crow_t = torch.from_numpy(crow_np).to(device=device, dtype=torch.int64)
            col_t = torch.from_numpy(np.ascontiguousarray(cols_block_np, dtype=np.int64)).to(
                device=device, dtype=torch.int64, non_blocking=True)

        # CN: CSR mask 只在 active set 上采样 F_block @ G_gpu.T，避免 per-edge feature materialization。
        # EN: The CSR mask samples F_block @ G_gpu.T only on active edges, avoiding per-edge feature materialization.
        with (tracer.span("hierarchy.internal_batch_compute_cost_vec.sddmm_sampled_addmm", "hierarchy", args=block_args) if tracer is not None else nullcontext()):
            local_rows_t = torch.repeat_interleave(
                torch.arange(int(q - p), dtype=torch.int64, device=device),
                crow_t[1:] - crow_t[:-1],
            )
            vals0 = a_block.index_select(0, local_rows_t) + b_gpu.index_select(0, col_t)
            with torch.no_grad():
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")
                    mask = torch.sparse_csr_tensor(crow_t, col_t, vals0, size=(int(q - p), int(G.shape[0])), device=device)
                out_sp = torch.sparse.sampled_addmm(mask, F_block, G_gpu.t(), beta=1.0, alpha=-dot_scale)
            out_sorted[base:end] = out_sp.values()
        del rows_block_np, cols_block_np, counts_block, crow_np, F_block, a_block, crow_t, col_t, local_rows_t, vals0, mask, out_sp
        p = q

    with (tracer.span("hierarchy.internal_batch_compute_cost_vec.sddmm_restore_order", "hierarchy", args=trace_args) if tracer is not None else nullcontext()):
        inv_perm_t = torch.from_numpy(inv_perm).to(device=device, dtype=torch.int64, non_blocking=True)
        out = out_sorted.index_select(0, inv_perm_t)

    # ---- 2) 可选释放 ----
    if free_after and device.type == 'cuda':
        with (tracer.span("hierarchy.internal_batch_compute_cost_vec.gpu_cleanup", "hierarchy", args=trace_args) if tracer is not None else nullcontext()):
            del G_gpu, b_gpu, out_sorted, inv_perm_t
            if rows_t_all is not None:
                del rows_t_all
            if cols_t_all is not None:
                del cols_t_all
            if bool(empty_cache_after):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    with (tracer.span("hierarchy.internal_batch_compute_cost_vec.finalize_output", "hierarchy", args=trace_args) if tracer is not None else nullcontext()):
        if rows_is_torch:
            if torch.is_tensor(out):
                return out.to(dtype=torch.float32, device=target_device)
            return torch.as_tensor(out, dtype=torch.float32, device=target_device)
        if torch.is_tensor(out):
            out_cpu = out.detach().cpu().numpy().astype(F.dtype, copy=False)
        else:
            out_cpu = out.astype(F.dtype, copy=False)
        return out_cpu


def _cost_pairs_sqeuclidean_batched(
    level_cache,
    rows: np.ndarray,
    cols: np.ndarray,
    batch_size: int = 65536,
    d_chunk: int = None,
    free_after: bool = True
) -> np.ndarray:
    """
    平方欧氏成本:
      ||S[i]-T[j]||^2 = s2[i] + t2[j] - 2 <S[i], T[j]>
    """
    S, T, s2, t2 = level_cache['S'], level_cache['T'], level_cache['s2'], level_cache['t2']
    rows_is_torch = torch.is_tensor(rows)
    target_device = rows.device if rows_is_torch else None
    rows_np = rows.detach().cpu().numpy() if rows_is_torch else np.asarray(rows, dtype=np.int64)
    cols_np = cols.detach().cpu().numpy() if torch.is_tensor(cols) else np.asarray(cols, dtype=np.int64)
    n = int(rows_np.shape[0])
    if n == 0:
        if rows_is_torch:
            return torch.empty(0, dtype=torch.float32, device=target_device)
        return np.array([], dtype=S.dtype)

    device = target_device if target_device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = int(S.shape[1])
    if device.type == 'cpu':
        c_add = s2[rows_np] + t2[cols_np]
        dots = np.einsum('nd,nd->n', S[rows_np], T[cols_np])
        out_cpu = (c_add - 2.0 * dots).astype(S.dtype, copy=False)
        if rows_is_torch:
            return torch.as_tensor(out_cpu, dtype=torch.float32, device=target_device)
        return out_cpu

    # ---- 1) 整块搬到 GPU（FP32） ----
    S_gpu = torch.from_numpy(np.ascontiguousarray(S)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    T_gpu = torch.from_numpy(np.ascontiguousarray(T)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    s2_gpu = torch.from_numpy(np.ascontiguousarray(s2)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    t2_gpu = torch.from_numpy(np.ascontiguousarray(t2)).to(
        device=device, dtype=torch.float32, non_blocking=False)

    # ---- 2) 分批 (rows, cols) ----
    out = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        rows_b = rows_np[i:j]
        cols_b = cols_np[i:j]

        rows_t = torch.from_numpy(rows_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)
        cols_t = torch.from_numpy(cols_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)

        if d_chunk is None or d_chunk >= d:
            Si = S_gpu.index_select(0, rows_t)        # (B, d)
            Tj = T_gpu.index_select(0, cols_t)        # (B, d)
            dots = (Si * Tj).sum(dim=1)               # (B,)
        else:
            B = rows_t.shape[0]
            acc = torch.zeros(B, dtype=torch.float32, device=device)
            for k in range(0, d, d_chunk):
                kk = min(k + d_chunk, d)
                Si = S_gpu.index_select(0, rows_t)[:, k:kk]
                Tj = T_gpu.index_select(0, cols_t)[:, k:kk]
                acc += (Si * Tj).sum(dim=1)
            dots = acc

        c_add = s2_gpu.index_select(0, rows_t) + \
            t2_gpu.index_select(0, cols_t) - 2.0 * dots
        out[i:j] = c_add.detach().cpu().numpy()
        i = j

    if free_after and device.type == 'cuda':
        del S_gpu, T_gpu, s2_gpu, t2_gpu, rows_t, cols_t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    out_cpu = out.astype(S.dtype, copy=False)
    if rows_is_torch:
        return torch.as_tensor(out_cpu, dtype=torch.float32, device=target_device)
    return out_cpu


def _cost_pairs_euclidean_batched(
    level_cache,
    rows: np.ndarray,
    cols: np.ndarray,
    batch_size: int = 65536,
    free_after: bool = True
) -> np.ndarray:
    """
    欧氏成本:
      ||S[i]-T[j]||_2
    """
    S, T = level_cache['S'], level_cache['T']
    rows_is_torch = torch.is_tensor(rows)
    target_device = rows.device if rows_is_torch else None
    rows_np = rows.detach().cpu().numpy() if rows_is_torch else np.asarray(rows, dtype=np.int64)
    cols_np = cols.detach().cpu().numpy() if torch.is_tensor(cols) else np.asarray(cols, dtype=np.int64)
    n = int(rows_np.shape[0])
    if n == 0:
        if rows_is_torch:
            return torch.empty(0, dtype=torch.float32, device=target_device)
        return np.array([], dtype=S.dtype)

    device = target_device if target_device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        diff = S[rows_np] - T[cols_np]
        out_cpu = np.sqrt(np.sum(diff * diff, axis=1)).astype(S.dtype, copy=False)
        if rows_is_torch:
            return torch.as_tensor(out_cpu, dtype=torch.float32, device=target_device)
        return out_cpu

    S_gpu = torch.from_numpy(np.ascontiguousarray(S)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    T_gpu = torch.from_numpy(np.ascontiguousarray(T)).to(
        device=device, dtype=torch.float32, non_blocking=False)

    out = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        rows_b = rows_np[i:j]
        cols_b = cols_np[i:j]

        rows_t = torch.from_numpy(rows_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)
        cols_t = torch.from_numpy(cols_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)

        Si = S_gpu.index_select(0, rows_t)
        Tj = T_gpu.index_select(0, cols_t)
        sq = torch.sum((Si - Tj) * (Si - Tj), dim=1)
        c_add = torch.sqrt(torch.clamp(sq, min=0.0) + 1e-12)
        out[i:j] = c_add.detach().cpu().numpy()
        i = j
        del rows_t, cols_t, Si, Tj, sq, c_add

    if free_after and device.type == 'cuda':
        del S_gpu, T_gpu
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    out_cpu = out.astype(S.dtype, copy=False)
    if rows_is_torch:
        return torch.as_tensor(out_cpu, dtype=torch.float32, device=target_device)
    return out_cpu


class HierarchicalOTSolver:
    def __init__(
        self,
        hierarchy_s: BaseHierarchy,
        hierarchy_t: BaseHierarchy,
        strategy: Optional[BaseStrategy],
        solver: LPSolver,
        cleaning_strategy: Optional[CleaningStrategy] = None,
        lp_solver_verbose: int = 0,
    ):
        self.hierarchy_s = hierarchy_s
        self.hierarchy_t = hierarchy_t
        self.lp_solver = solver
        self.strategy = strategy
        self.cleaning_strategy = cleaning_strategy
        self.lp_solver_verbose = lp_solver_verbose

        self.solutions: Dict[int, Dict[str, Any]] = {}
        self.active_support: Optional[ActiveSupport] = None

        self.build_time = hierarchy_s.build_time + hierarchy_t.build_time
        self._profiling_enabled = False
        self._lp_solver_kwargs: Dict[str, Any] = {}
        self._use_gpu_active_support = False
        self._active_support_device = "cpu"
        self._report_added_violation_stats = False
        self._added_violation_rel_threshold = 1e-6
        self._runtime_logging: Dict[str, Any] = {
            "enabled": True,
            "progress": True,
            "profile_iter": True,
            "profile_level": True,
            "profile_run": True,
            "warm_start": True,
            "iter_interval": 10,
        }

    def _runtime_log_enabled(self, category: str) -> bool:
        cfg = getattr(self, "_runtime_logging", None)
        if not isinstance(cfg, dict):
            return True
        if not bool(cfg.get("enabled", True)):
            return False
        return bool(cfg.get(category, True))

    def _runtime_log(self, category: str, message: str, *, flush: bool = False) -> None:
        if self._runtime_log_enabled(category):
            print(message, flush=flush)

    @staticmethod
    def _format_profile_components(
        components: Dict[str, float],
        max_items: int = 6,
        base_total: Optional[float] = None,
    ) -> str:
        if not components:
            return "-"
        items = [(k, float(v)) for k, v in components.items() if float(v) > 0.0]
        if not items:
            return "-"
        items.sort(key=lambda kv: kv[1], reverse=True)
        if base_total is None:
            base_total = sum(v for _, v in items)
        base = max(float(base_total), 1e-12)
        return ", ".join(f"{k}={v:.2f}s({(v/base)*100.0:.1f}%)" for k, v in items[:max_items])

    @staticmethod
    def _fmt_optional_sci(value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.{digits}e}"

    @staticmethod
    def _fmt_optional_fixed(value: Any, digits: int = 6) -> str:
        if value is None:
            return "n/a"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.{digits}f}"

    def _extract_lp_profile_summary(self, step_pack: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(step_pack, dict):
            return None
        if not bool(step_pack.get("success", False)):
            return None

        if step_pack.get("res") is not None:
            res = step_pack["res"]
            diag = step_pack.get("diag", {}) if isinstance(step_pack.get("diag"), dict) else {}
            return {
                "time": getattr(res, "duration", None),
                "iters": getattr(res, "iterations", None),
                "primal_obj": getattr(res, "obj_val", None),
                "dual_obj": getattr(res, "dual_obj_val", None),
                "primal_feas": getattr(res, "primal_feas", None),
                "dual_feas": getattr(res, "dual_feas", None),
                "gap": getattr(res, "gap", None),
                "diag": diag,
            }

        lp_result = step_pack.get("lp_result")
        if isinstance(lp_result, dict):
            diag = lp_result.get("diag", {}) if isinstance(lp_result.get("diag"), dict) else {}
            return {
                "time": diag.get("backend_solve_wall_time"),
                "iters": lp_result.get("iters"),
                "primal_obj": lp_result.get("obj"),
                "dual_obj": lp_result.get("dual_obj"),
                "primal_feas": lp_result.get("primal_feas"),
                "dual_feas": lp_result.get("dual_feas"),
                "gap": lp_result.get("gap"),
                "diag": diag,
            }
        return None

    def _print_profile_lp(
        self,
        level_idx: int,
        inner_iter: int,
        step_pack: Optional[Dict[str, Any]],
    ) -> None:
        if not self._profiling_enabled or not self._runtime_log_enabled("profile_iter"):
            return
        summary = self._extract_lp_profile_summary(step_pack)
        if not summary:
            return
        self._runtime_log(
            "profile_iter",
            f"[Profile][L{level_idx}][I{inner_iter + 1}][lp] "
            f"time={self._fmt_optional_fixed(summary.get('time'), digits=2)}s, "
            f"iter={summary.get('iters', 'n/a')}, "
            f"obj={self._fmt_optional_fixed(summary.get('primal_obj'))} / {self._fmt_optional_fixed(summary.get('dual_obj'))}, "
            f"RelRes={self._fmt_optional_sci(summary.get('primal_feas'))} / {self._fmt_optional_sci(summary.get('dual_feas'))}, "
            f"Gap={self._fmt_optional_sci(summary.get('gap'), digits=4)}"
        )
        diag = summary.get("diag") if isinstance(summary.get("diag"), dict) else {}
        if diag.get("matrix_value_mode") is not None:
            self._runtime_log(
                "profile_iter",
                f"[Profile][L{level_idx}][I{inner_iter + 1}][lp_implicit] "
                f"mode={diag.get('matrix_value_mode')} "
                f"implicit_ax={1 if bool(diag.get('implicit_ax_enabled', False)) else 0} "
                f"implicit_aty={1 if bool(diag.get('implicit_aty_enabled', False)) else 0} "
                f"ax_agg={diag.get('implicit_ax_agg', 'none')} "
                f"row0_p50={diag.get('implicit_ax_row0_unique_p50', 'n/a')} "
                f"row1_p50={diag.get('implicit_ax_row1_unique_p50', 'n/a')}",
            )

    def _print_profile_pricing(
        self,
        level_idx: int,
        inner_iter: int,
        step_pack: Optional[Dict[str, Any]],
    ) -> None:
        if not self._profiling_enabled or not self._runtime_log_enabled("profile_iter"):
            return
        if not isinstance(step_pack, dict):
            return
        info = step_pack.get("pricing_info")
        if not isinstance(info, dict):
            return
        self._runtime_log(
            "profile_iter",
            f"[Profile][L{level_idx}][I{inner_iter + 1}][pricing] "
            f"time={self._fmt_optional_fixed(info.get('time'), digits=2)}s, "
            f"active={int(info.get('active_before', 0)):,}, "
            f"found={int(info.get('found', 0)):,}, "
            f"added={int(info.get('added', 0)):,}"
        )

    def _print_profile_convergence_check(
        self,
        level_idx: int,
        inner_iter: int,
        step_pack: Optional[Dict[str, Any]],
    ) -> None:
        if not self._profiling_enabled or not self._runtime_log_enabled("profile_iter"):
            return
        if not isinstance(step_pack, dict):
            return
        info = step_pack.get("convergence_info")
        if not isinstance(info, dict):
            return
        criterion = info.get("criterion", "n/a")
        plateau_counter = info.get("plateau_counter", "n/a")
        required_plateau = info.get("required_plateau", "n/a")
        objective_tol = self._fmt_optional_sci(info.get("objective_tol"))
        self._runtime_log(
            "profile_iter",
            f"[Profile][L{level_idx}][I{inner_iter + 1}][convergence_check] "
            f"signed_rel_obj_change={self._fmt_optional_sci(info.get('signed_rel_obj_change'))}, "
            f"converged={bool(info.get('is_converged', False))}, "
            f"criterion={criterion}, "
            f"plateau={plateau_counter}/{required_plateau}, "
            f"tol={objective_tol}, "
            f"dual_feas={self._fmt_optional_sci(info.get('dual_feasibility'))}, "
            f"dual_feas_tol={self._fmt_optional_sci(info.get('dual_feasibility_tol'))}, "
            f"dual_feas_passed={info.get('dual_feasibility_passed', 'n/a')}"
        )

    def _print_profile_inner(
        self,
        level_idx: int,
        inner_iter: int,
        iter_profile: Optional[Dict[str, Any]],
        step_pack: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._profiling_enabled or not self._runtime_log_enabled("profile_iter"):
            return
        if not iter_profile:
            return
        iter_wall = float(iter_profile.get("time", 0.0))
        phase_times = iter_profile.get("phase_times", {})
        if not isinstance(phase_times, dict):
            phase_times = {}
        phases = iter_profile.get("phases", {})
        if not isinstance(phases, dict):
            phases = {}
        solve_t = float(phase_times.get(PHASE_SOLVE_ITERATION_LP, 0.0))
        finalize_t = float(phase_times.get(PHASE_FINALIZE_ITERATION, 0.0))
        finalize_components = phases.get(PHASE_FINALIZE_ITERATION, {})
        if isinstance(finalize_components, dict):
            finalize_components = finalize_components.get("components", {})
        else:
            finalize_components = {}
        base = max(iter_wall, 1e-12)
        else_t = max(iter_wall - solve_t - finalize_t, 0.0)
        finalize_component_total = sum(float(v) for v in finalize_components.values() if float(v) > 0.0)
        finalize_component_base = max(finalize_component_total, 1e-12)
        self._print_profile_lp(level_idx, inner_iter, step_pack)
        self._print_profile_pricing(level_idx, inner_iter, step_pack)
        self._print_profile_convergence_check(level_idx, inner_iter, step_pack)
        self._runtime_log(
            "profile_iter",
            f"[Profile][L{level_idx}][I{inner_iter + 1}][finalize] "
            f"{self._format_profile_components(finalize_components, max_items=4, base_total=finalize_component_base)}"
        )
        self._runtime_log(
            "profile_iter",
            f"[Profile][L{level_idx}][I{inner_iter + 1}] "
            f"total={iter_wall:.2f}s "
            f"solve={solve_t:.2f}s({(solve_t/base)*100.0:.1f}%) "
            f"finalize={finalize_t:.2f}s({(finalize_t/base)*100.0:.1f}%) "
            f"else={else_t:.2f}s({(else_t/base)*100.0:.1f}%)"
        )
        self._runtime_log("profile_iter", "---")

    def _print_profile_level(
        self,
        level_idx: int,
        level_state: Dict[str, Any],
        level_profile: Optional[Dict[str, Any]],
    ) -> None:
        if not self._profiling_enabled or not self._runtime_log_enabled("profile_level"):
            return
        if not level_profile:
            return
        level_wall = float(level_profile.get("time", 0.0))
        phase_times = level_profile.get("phase_times", {})
        if not isinstance(phase_times, dict):
            phase_times = {}
        iters = int(level_profile.get("num_iters", 0))
        if iters <= 0:
            iters = int(level_state.get("completed_iters", level_state.get("current_iter", 0)))
        active = level_state.get("curr_active_size")
        active_text = f" active={int(active)}" if active is not None else ""
        lp_time = float(level_state.get("level_lp_time", 0.0))
        pricing_time = float(level_state.get("level_pricing_time", 0.0))
        base = max(level_wall, 1e-12)
        else_t = max(level_wall - lp_time - pricing_time, 0.0)
        self._runtime_log("profile_level", f"=== Summary level {level_idx} ===")
        self._runtime_log(
            "profile_level",
            f"[Profile][L{level_idx}] total={level_wall:.2f}s iters={iters} "
            f"lp={lp_time:.2f}s({(lp_time/base)*100.0:.1f}%) "
            f"pricing={pricing_time:.2f}s({(pricing_time/base)*100.0:.1f}%) "
            f"else={else_t:.2f}s({(else_t/base)*100.0:.1f}%){active_text}"
        )

    def _print_profile_init_level(
        self,
        level_idx: int,
        components: Dict[str, float],
        *,
        primal_nnz: int,
        active_size: int,
        bfs_added: int = 0,
    ) -> None:
        if not self._runtime_log_enabled("profile_level"):
            return
        total = sum(float(v) for v in components.values() if float(v) > 0.0)
        base = max(total, 1e-12)
        self._runtime_log(
            "profile_level",
            f"[Profile][L{level_idx}][Init] total={total:.2f}s "
            f"primal_nnz={int(primal_nnz):,} active={int(active_size):,} bfs_added={int(bfs_added):,}"
        )
        self._runtime_log(
            "profile_level",
            f"[Profile][L{level_idx}][Init][components] "
            f"{self._format_profile_components(components, max_items=6, base_total=base)}"
        )
        self._runtime_log("profile_level", "---")

    def _print_profile_run(
        self,
        profiling: Optional[Dict[str, Any]],
        *,
        include_level_breakdown: bool = True,
    ) -> None:
        if not self._profiling_enabled or not profiling or not self._runtime_log_enabled("profile_run"):
            return
        solve_time = float(profiling.get("solve_time", 0.0))
        build_time = float(profiling.get("build_time", 0.0))
        total_time = float(profiling.get("total_time", 0.0))
        run_components = profiling.get("components", {})
        if not isinstance(run_components, dict):
            run_components = {}
        lp_time = float(run_components.get("lp", 0.0))
        pricing_time = float(run_components.get("pricing", 0.0))
        else_t = max(solve_time - lp_time - pricing_time, 0.0)
        solve_base = max(float(solve_time), 1e-12)
        self._runtime_log(
            "profile_run",
            f"[Profile][Run] total={total_time:.2f}s solve={solve_time:.2f}s build={build_time:.2f}s | "
            f"lp={lp_time:.2f}s({(lp_time/solve_base)*100.0:.1f}%), "
            f"pricing={pricing_time:.2f}s({(pricing_time/solve_base)*100.0:.1f}%), "
            f"else={else_t:.2f}s({(else_t/solve_base)*100.0:.1f}%)"
        )
        if not include_level_breakdown:
            return
        self._print_profile_run_level_breakdown(profiling)

    def _print_profile_run_level_breakdown(self, profiling: Optional[Dict[str, Any]]) -> None:
        if not self._profiling_enabled or not profiling or not self._runtime_log_enabled("profile_run"):
            return
        run = profiling.get("run", {})
        if not isinstance(run, dict):
            run = {}
        levels = run.get("levels", profiling.get("levels", []))
        if not isinstance(levels, list) or not levels:
            return
        self._runtime_log("profile_run", "=== Overall Breakdown ===")
        for row in levels:
            if not isinstance(row, dict):
                continue
            level = int(row.get("level_idx", row.get("level", -1)))
            iters = int(row.get("num_iters", row.get("iters", 0)))
            t_level = float(row.get("time", 0.0))
            components = row.get("components", {})
            if not isinstance(components, dict):
                components = {}
            level_base = max(t_level, 1e-12)
            level_lp = float(components.get("lp", 0.0))
            level_pricing = float(components.get("pricing", 0.0))
            level_else = max(t_level - level_lp - level_pricing, 0.0)
            self._runtime_log(
                "profile_run",
                f"[Profile][Run][L{level}] total={t_level:.2f}s iters={iters} "
                f"(lp={(level_lp/level_base)*100.0:.1f}%, "
                f"pricing={(level_pricing/level_base)*100.0:.1f}%, "
                f"else={(level_else/level_base)*100.0:.1f}%)"
            )

    @staticmethod
    def _normalize_cost_type_name(cost_type: str) -> str:
        return _normalize_cost_type_name(cost_type)

    @staticmethod
    def _prepare_level_cost_cache_sqeuclidean(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
        return _prepare_level_cost_cache_sqeuclidean(points_s, points_t)

    @staticmethod
    def _prepare_level_cost_cache_euclidean(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
        return _prepare_level_cost_cache_euclidean(points_s, points_t)

    @staticmethod
    def _prepare_level_cost_cache_lowrank(
        F: np.ndarray,
        G: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        *,
        dot_scale: float = 1.0,
    ) -> Dict[str, Any]:
        return _prepare_level_cost_cache_lowrank(F, G, a, b, dot_scale=dot_scale)

    @staticmethod
    def _prepare_level_cost_cache_l1(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
        return _prepare_level_cost_cache_l1(points_s, points_t)

    @staticmethod
    def _prepare_level_cost_cache_linf(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
        return _prepare_level_cost_cache_linf(points_s, points_t)

    @staticmethod
    def _northwest_corner(p_in: np.ndarray, q_in: np.ndarray):
        return _northwest_corner_numba(p_in, q_in)

    @staticmethod
    def _fill_x_prev(
        primal_rows,
        primal_cols,
        primal_data,
        sorted_inc_keys,
        sorted_inc_pos,
        x_prev_filled,
        num_cols,
    ):
        return _fill_x_prev_numba(
            primal_rows,
            primal_cols,
            primal_data,
            sorted_inc_keys,
            sorted_inc_pos,
            x_prev_filled,
            num_cols,
        )

    def _create_active_support(self, level_cache: Dict[str, Any], *, track_creation: bool):
        return ActiveSupport(
            level_cache,
            track_creation=track_creation,
            device=getattr(self, "_active_support_device", "cpu"),
        )

    @staticmethod
    def _decode_keep_to_struct(keep: np.ndarray, n_t: int):
        return decode_keep_1d_to_struct(keep, n_t)

    @staticmethod
    def _remap_duals_for_warm_start(y_solution_last, keep):
        return remap_duals_for_warm_start(y_solution_last, keep)

    @staticmethod
    def _logger_is_enabled_info() -> bool:
        return logger.isEnabledFor(logging.INFO)

    @staticmethod
    def _logger_info(message: str, *args: Any) -> None:
        logger.info(message, *args)

    @staticmethod
    def _time_now() -> float:
        return time.perf_counter()

    def _initialize_level(
        self,
        level_idx: int,
        cost_type: str,
        use_bfs_skeleton: bool
    ) -> Tuple[csc_matrix, np.ndarray]:
        return cluster_initial.initialize_level(self, level_idx, cost_type, use_bfs_skeleton)

    def _solve_exact_flat(
        self,
        tolerance: Dict[str, float],
        *,
        trace_collector: Optional[_ChromeTraceCollector] = None,
        trace_prefix: str = "solve_ot",
    ) -> Dict[str, Any]:
        return cluster_initial.solve_exact_flat(
            self,
            tolerance,
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
        )

    @staticmethod
    def _sum_level_summary_metric(
        level_summaries: List[Dict[str, Any]],
        key: str,
    ) -> float:
        return float(
            sum(float(item.get(key, 0.0)) for item in level_summaries if isinstance(item, dict))
        )

    def _compute_pair_costs_arrays(
        self, rows_add, cols_add, **kwargs
    ):
        """
        CN: 为所有新增支撑边应用与扫描一致的成本扰动。
        EN: Apply the same cost perturbation as the scans to every inserted support edge.
        """
        costs = self._compute_base_pair_costs_arrays(rows_add, cols_add, **kwargs)
        perturbation = getattr(self, "_metric_cost_perturbation", None)
        if perturbation is not None:
            from hello_ot.kernels.norm_cost_scan import _apply_metric_pair_perturbation

            costs = _apply_metric_pair_perturbation(costs, rows_add, cols_add, perturbation)
        return costs

    def _compute_base_pair_costs_arrays(
        self,
        rows_add: np.ndarray,
        cols_add: np.ndarray,
        support_order_cache: Optional[Dict[str, Any]] = None,
    ):
        """
        CN: 按当前 active-support 的 level_cache 计算一组边的成本。
        EN: Compute costs for edge pairs using the current active-support level_cache.
        """
        level_cache = self.active_support.level_cache
        cost_type = _normalize_cost_type_name(level_cache.get('cost_type', 'l2^2'))
        if cost_type == 'lowrank':
            return _cost_pairs_lowrank_batched(
                level_cache,
                rows_add,
                cols_add,
                support_order_cache=support_order_cache,
            )
        elif cost_type == 'l1':
            return _cost_pairs_l1_batched(level_cache, rows_add, cols_add)
        elif cost_type == 'linf':
            return _cost_pairs_linf_batched(level_cache, rows_add, cols_add)
        elif cost_type == 'l2':
            return _cost_pairs_euclidean_batched(level_cache, rows_add, cols_add)
        return _cost_pairs_sqeuclidean_batched(level_cache, rows_add, cols_add)

    def _append_new_pairs_arrays(
        self,
        rows_add: np.ndarray,
        cols_add: np.ndarray,
        current_inner_iter: int,
        c_add=None,
    ):
        if self.active_support._array_size(rows_add) == 0:
            return

        if c_add is None:
            c_add = self._compute_pair_costs_arrays(rows_add, cols_add)

        if getattr(self.active_support, "is_torch_backend", False) and not torch.is_tensor(c_add):
            c_add = torch.as_tensor(c_add, dtype=torch.float32, device=self.active_support.device)

        self.active_support.add_pairs(
            rows_add, cols_add, c_add, current_inner_iter)

        logger.debug(f"  [_append] Added {len(rows_add)} pairs.")

    def _refine_solution(self, coarse_sol, coarse_lvl_s, coarse_lvl_t, fine_lvl_s, fine_lvl_t):
        primal_coarse, duals_coarse = coarse_sol['primal'], coarse_sol['dual']

        # Duals Broadcast
        f_coarse, g_coarse = duals_coarse[:len(
            coarse_lvl_s.points)], duals_coarse[len(coarse_lvl_s.points):]
        duals_fine = np.concatenate(
            (f_coarse[fine_lvl_s.child_labels], g_coarse[fine_lvl_t.child_labels]))

        # Primal Interpolation (Numba Optimized)
        p_s, p_t = fine_lvl_s.child_labels, fine_lvl_t.child_labels
        n_fine_s, n_fine_t = len(p_s), len(p_t)
        n_coarse_s, n_coarse_t = primal_coarse.shape

        s_counts = np.bincount(p_s, minlength=n_coarse_s)
        t_counts = np.bincount(p_t, minlength=n_coarse_t)

        primal_coarse_coo = primal_coarse.tocoo()
        coarse_data = np.asarray(primal_coarse_coo.data, dtype=np.float64)

        offsets = _precalc_offsets(
            primal_coarse_coo.row, primal_coarse_coo.col, s_counts, t_counts)
        total_nnz = offsets[-1]
        out_rows = np.empty(total_nnz, dtype=np.int32)
        out_cols = np.empty(total_nnz, dtype=np.int32)
        out_data = np.empty(total_nnz, dtype=np.float32)

        # Build Maps
        w_s = 1.0 / np.maximum(s_counts[p_s], 1.0)
        w_t = 1.0 / np.maximum(t_counts[p_t], 1.0)

        map_s = coo_matrix((w_s, (p_s, np.arange(n_fine_s))),
                           shape=(n_coarse_s, n_fine_s)).tocsr()
        map_t = coo_matrix((w_t, (p_t, np.arange(n_fine_t))),
                           shape=(n_coarse_t, n_fine_t)).tocsr()

        _numba_expand_kernel(
            primal_coarse_coo.row, primal_coarse_coo.col, coarse_data,
            map_s.indptr, map_s.indices, map_s.data,
            map_t.indptr, map_t.indices, map_t.data,
            offsets, out_rows, out_cols, out_data
        )

        refined_primal = coo_matrix(
            (out_data, (out_rows, out_cols)), shape=(n_fine_s, n_fine_t))
        return refined_primal, duals_fine
