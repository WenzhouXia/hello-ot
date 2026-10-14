from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Dict, Optional
import numpy as np
import torch
from hello_ot._internal.trace import _ChromeTraceCollector

def _to_pinned_numpy(array: np.ndarray, *, dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    """
    CN: CUDA 可用时分配锁页内存；CPU-only PyTorch 使用普通连续数组。
    EN: Allocate pinned memory when CUDA is available; use an ordinary contiguous array on CPU-only PyTorch.
    """
    if not torch.cuda.is_available():
        return np.ascontiguousarray(array, dtype=dtype)
    # CN: 1. 创建锁页 CPU tensor。
    # EN: 1. Create a pinned CPU tensor.
    torch_dtype = torch.float64 if dtype == np.dtype(np.float64) else torch.float32
    tensor = torch.empty(array.shape, dtype=torch_dtype, pin_memory=True)
    # CN: 2. 复制数据。
    # EN: 2. Copy the data.
    tensor.copy_(torch.from_numpy(array))
    # CN: 3. 返回共享内存的 NumPy 视图。
    # EN: 3. Return a NumPy view over the shared memory.
    return tensor.numpy()


def _prepare_level_cost_cache_sqeuclidean(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    S = np.asarray(points_s, dtype=np.float32, order='C')
    T = np.asarray(points_t, dtype=np.float32, order='C')
    s2 = np.einsum('ij,ij->i', S, S)
    t2 = np.einsum('ij,ij->i', T, T)
    return {'S': S, 'T': T, 's2': s2, 't2': t2, 'cost_type': 'l2^2'}


def _prepare_level_cost_cache_euclidean(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    """
    CN: 构建 l2 代价缓存，只保留原始坐标。
    EN: Build the l2 cost cache using only the original coordinates.
    """
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
    compute_precision: str = "fp32",
) -> Dict[str, Any]:
    F_mat = _to_pinned_numpy(F)
    G_mat = _to_pinned_numpy(G)
    scalar_dtype = np.dtype(np.float64 if compute_precision == "fp64" else np.float32)
    a_vec = _to_pinned_numpy(a, dtype=scalar_dtype)
    b_vec = _to_pinned_numpy(b, dtype=scalar_dtype)

    # CN: 保留原有字典结构，锁页数组可加速后续的非阻塞 GPU 传输。
    # EN: Preserve the dictionary layout while pinned arrays accelerate later nonblocking GPU transfers.
    scale = float(dot_scale)
    if scale not in {1.0, 2.0}:
        raise ValueError("dot_scale must be exactly 1 or 2")
    return {'F': F_mat, 'G': G_mat, 'a': a_vec, 'b': b_vec, 'dot_scale': scale, 'cost_type': 'lowrank', 'compute_precision': compute_precision}


def _prepare_level_cost_cache_l1(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    """
    CN: 构建 l1 代价缓存，只保留原始坐标。
    EN: Build the l1 cost cache using only the original coordinates.
    """
    S = np.asarray(points_s, dtype=np.float32, order='C')
    T = np.asarray(points_t, dtype=np.float32, order='C')
    return {'S': S, 'T': T, 'cost_type': 'l1'}


def _prepare_level_cost_cache_linf(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    """
    CN: 构建 linf 代价缓存，只保留原始坐标。
    EN: Build the linf cost cache using only the original coordinates.
    """
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
    CN: 分批计算 l1 代价 c_ij = ||S[i] - T[j]||_1。
    EN: Compute l1 costs c_ij = ||S[i] - T[j]||_1 in batches.
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

    # CN: 在 GPU 上分批计算。
    # EN: Compute batches on the GPU.
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
        l1_dist = torch.sum(torch.abs(Si - Tj), dim=1)

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
    CN: 分批计算 linf 代价 c_ij = ||S[i] - T[j]||_inf。
    EN: Compute linf costs c_ij = ||S[i] - T[j]||_inf in batches.
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

    # CN: 在 GPU 上分批计算。
    # EN: Compute batches on the GPU.
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
        linf_dist = torch.max(torch.abs(Si - Tj), dim=1)[0]

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
    CN: 用 SDDMM 分批计算 lowrank 代价 c_ij = a[i] + b[j] - <F[i], G[j]>。
    EN: Compute lowrank costs c_ij = a[i] + b[j] - <F[i], G[j]> in SDDMM batches.

    CN: batch_size 限制每个 source-row block 的 active edge 数；d_chunk 仅保留用于兼容旧接口。
    EN: batch_size limits active edges per source-row block; d_chunk remains only for legacy API compatibility.
    """
    F, G, a, b = level_cache['F'], level_cache['G'], level_cache['a'], level_cache['b']
    dot_scale = float(level_cache.get('dot_scale', 1.0))
    compute_precision = str(level_cache.get('compute_precision', 'fp32'))
    np_dtype = np.float64 if compute_precision == 'fp64' else np.float32
    torch_dtype = torch.float64 if compute_precision == 'fp64' else torch.float32
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
            return torch.empty(0, dtype=torch_dtype, device=target_device)
        return np.array([], dtype=np_dtype)

    # CN: CPU 路径直接计算所有指定边的代价。
    # EN: The CPU path directly computes costs for all requested edges.
    d = int(F.shape[1])
    if device.type == 'cpu':
        if rows_np is None:
            rows_np = rows.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        if cols_np is None:
            cols_np = cols.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        c_add = a[rows_np].astype(np_dtype, copy=False) + b[cols_np].astype(np_dtype, copy=False)
        dots = np.einsum('nd,nd->n', F[rows_np].astype(np_dtype), G[cols_np].astype(np_dtype), dtype=np_dtype)
        out_cpu = (c_add - dot_scale * dots).astype(np_dtype, copy=False)
        if rows_is_torch:
            return torch.as_tensor(out_cpu, dtype=torch_dtype, device=target_device)
        return out_cpu

    # CN: 1. active-support SDDMM 使 target 常驻，并流式处理 source row blocks。
    # EN: 1. Active-support SDDMM keeps the target resident and streams source-row blocks.
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
        G_gpu = torch.from_numpy(np.ascontiguousarray(G, dtype=np_dtype)).to(
            device=device, dtype=torch_dtype, non_blocking=True)
        b_gpu = torch.from_numpy(np.ascontiguousarray(b, dtype=np_dtype)).to(
            device=device, dtype=torch_dtype, non_blocking=True)

    out_sorted = torch.empty(n, dtype=torch_dtype, device=device)
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
            F_block = torch.from_numpy(np.ascontiguousarray(F[rows_block_np], dtype=np_dtype)).to(
                device=device, dtype=torch_dtype, non_blocking=True)
            a_block = torch.from_numpy(np.ascontiguousarray(a[rows_block_np], dtype=np_dtype)).to(
                device=device, dtype=torch_dtype, non_blocking=True)
            crow_t = torch.from_numpy(crow_np).to(device=device, dtype=torch.int64)
            col_t = torch.from_numpy(np.ascontiguousarray(cols_block_np, dtype=np.int64)).to(
                device=device, dtype=torch.int64, non_blocking=True)

        # CN: CSR mask 只在 active support 上采样 F_block @ G_gpu.T，避免 per-edge feature materialization。
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

    # CN: 2. 按配置释放 GPU 临时缓存。
    # EN: 2. Release temporary GPU cache when configured.
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
                return out.to(dtype=torch_dtype, device=target_device)
            return torch.as_tensor(out, dtype=torch_dtype, device=target_device)
        if torch.is_tensor(out):
            out_cpu = out.detach().cpu().numpy().astype(np_dtype, copy=False)
        else:
            out_cpu = out.astype(np_dtype, copy=False)
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
    CN: 分批计算 l2^2 代价 ||S[i]-T[j]||^2。
    EN: Compute l2^2 costs ||S[i]-T[j]||^2 in batches.
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

    # CN: 1. 将特征和平方范数整体上传到 GPU，并转为 FP32。
    # EN: 1. Upload features and squared norms to the GPU as FP32 arrays.
    S_gpu = torch.from_numpy(np.ascontiguousarray(S)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    T_gpu = torch.from_numpy(np.ascontiguousarray(T)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    s2_gpu = torch.from_numpy(np.ascontiguousarray(s2)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    t2_gpu = torch.from_numpy(np.ascontiguousarray(t2)).to(
        device=device, dtype=torch.float32, non_blocking=False)

    # CN: 2. 分批处理 (rows, cols) 边。
    # EN: 2. Process (rows, cols) edges in batches.
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
            Si = S_gpu.index_select(0, rows_t)
            Tj = T_gpu.index_select(0, cols_t)
            dots = (Si * Tj).sum(dim=1)
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
    CN: 分批计算 l2 代价 ||S[i]-T[j]||_2。
    EN: Compute l2 costs ||S[i]-T[j]||_2 in batches.
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
