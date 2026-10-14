from __future__ import annotations

import importlib
from typing import Any, Dict, Literal, Optional, Tuple
import numpy as np
import torch
from hello_ot.kernels.scan_contract import (_TOPK_SLOT_BYTES, DualFeasibilityCertificate, DualViolationScanResult, custom_topk_bucket, plan_resident_side, plan_stream_chunk_rows, reusable_cuda_memory_bytes)
from hello_ot.costs.metric_perturbation import _fused_perturb_arguments

MetricCostType = Literal["l1", "linf", "l2"]
_NORM_COST_SCAN_EXT_MODULE = "hello_ot._native.norm_cost_scan.hierot_norm_cost_scan_ext"
_FUSED_TOP_K_VALUES = (1, 2, 4, 8, 16, 32)
_COST_TYPE_IDS = {"l1": 0, "linf": 1, "l2": 2}
_COMPUTE_PRECISION_IDS = {"fp32": 0, "fp64": 1}

def _require_norm_cost_scan_ext() -> Any:
    """
    CN: 惰性加载安装时编译的 norm-cost scan CUDA 扩展；失败直接报错，
        不提供 KeOps 回退。
    EN: Lazily load the install-time compiled norm-cost scan CUDA extension;
        fail hard without a KeOps fallback.
    """
    try:
        module = importlib.import_module(_NORM_COST_SCAN_EXT_MODULE)
        if not getattr(module, "supports_cost_perturbation", False):
            raise RuntimeError("installed norm-cost extension predates cost perturbation; rebuild the native extensions")
        if not getattr(module, "supports_fp64_accumulation", False):
            raise RuntimeError("installed norm-cost extension predates FP64 cost accumulation; rebuild the native extensions")
        if getattr(module, "index_hash_arithmetic", None) != "fp32_remainder_v1":
            raise RuntimeError("installed norm-cost extension uses a different index hash; rebuild the native extensions")
        return module
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "norm-cost hello requires the custom scan CUDA extension "
            f"({_NORM_COST_SCAN_EXT_MODULE}); rebuild with `HELLO_OT_BUILD_NATIVE=1 pip install -e .` on a "
            "CUDA-capable machine. There is no KeOps fallback in the metric chain."
        ) from exc


def _fused_cost_type_id(cost_type: MetricCostType) -> int:
    if str(cost_type) not in _COST_TYPE_IDS:
        raise ValueError("metric hello supports cost_type in {'l1', 'linf', 'l2'}.")
    return int(_COST_TYPE_IDS[str(cost_type)])


def _compute_precision_id(compute_precision: str) -> int:
    normalized = str(compute_precision).strip().lower()
    if normalized not in _COMPUTE_PRECISION_IDS:
        raise ValueError("compute_precision must be one of: fp32, fp64")
    return int(_COMPUTE_PRECISION_IDS[normalized])


def _require_compute_precision_capability(ext: Any, compute_precision_id: int) -> None:
    """
    CN: FP64 请求必须由明确声明支持该 ABI 的 native 扩展处理。
    EN: FP64 requests require a native extension that explicitly advertises the matching ABI.
    """
    if int(compute_precision_id) == 1 and not bool(
        getattr(ext, "supports_fp64_accumulation", False)
    ):
        raise RuntimeError(
            "installed norm-cost extension predates FP64 accumulation; rebuild the native extensions"
        )


def _fused_metric_kmin(
    *,
    query_points: np.ndarray,
    database_points: np.ndarray,
    known_dual: np.ndarray | torch.Tensor,
    cost_type: MetricCostType,
    k: int,
    query_is_source: bool = True,
    query_global_index: Optional[np.ndarray] = None,
    database_global_index: Optional[np.ndarray] = None,
    perturbation: Optional[Dict[str, Any]] = None,
    compute_precision: str = "fp32",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CN: 用 fused kernel 计算每行 (cost+perturb-dual) 的 K 个最小值。
        1..32 的任意 K 向上映射到 CUDA bucket，然后裁剪为请求的 K；不做静默回退。
    EN: Use the fused kernel for the K smallest (cost+perturb-dual) values per
        row. Any K in 1..32 is rounded up to a CUDA bucket and sliced back to the
        requested K; no silent fallback is allowed.
    """
    compute_precision_id = _compute_precision_id(compute_precision)
    k_int = min(int(k), int(database_points.shape[0]))
    if k_int < 1:
        raise ValueError("metric dual assignment requires a non-empty database")
    kernel_k = custom_topk_bucket(k_int, score_family="norm_cost")
    ext = _require_norm_cost_scan_ext()
    _require_compute_precision_capability(ext, compute_precision_id)
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("custom norm-cost scan requires CUDA")
    query_np = np.asarray(query_points, dtype=np.float32, order="C")
    database_np = np.asarray(database_points, dtype=np.float32, order="C")
    known_dual_np = (
        known_dual.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(known_dual)
        else np.asarray(known_dual, dtype=np.float64)
    )
    device = torch.device("cuda")
    available_bytes, driver_free_bytes, reclaimable_cache_bytes = reusable_cuda_memory_bytes(device)
    memory_plan = plan_resident_side(
        source_count=int(query_np.shape[0]),
        target_count=int(database_np.shape[0]),
        feature_dim=int(query_np.shape[1]),
        topk_bucket=int(kernel_k),
        available_bytes=int(available_bytes),
        driver_free_bytes=int(driver_free_bytes),
        reclaimable_cache_bytes=int(reclaimable_cache_bytes),
    )
    streamed_count = (
        int(query_np.shape[0])
        if memory_plan.streamed_side == "source"
        else int(database_np.shape[0])
    )
    chunk_rows = plan_stream_chunk_rows(
        memory_plan,
        streamed_count=streamed_count,
        feature_dim=int(query_np.shape[1]),
        max_rows=8192,
    )

    def run_chunk(
        query_t: torch.Tensor,
        database_t: torch.Tensor,
        dual_t: torch.Tensor,
        query_index: Optional[np.ndarray],
        database_index: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma, seed, mode, q_idx_t, d_idx_t, coeffs = _fused_perturb_arguments(
            perturbation=perturbation,
            query_global_index=query_index,
            database_global_index=database_index,
            device=device,
        )
        return ext.fused_gcost_topk(
            query_t,
            database_t,
            dual_t,
            int(kernel_k),
            int(_fused_cost_type_id(cost_type)),
            q_idx_t,
            d_idx_t,
            float(sigma),
            int(seed),
            mode,
            bool(query_is_source),
            coeffs,
            compute_precision_id,
        )

    if memory_plan.resident_side == "target":
        database_t = torch.as_tensor(database_np, dtype=torch.float32, device=device).contiguous()
        dual_t = torch.as_tensor(known_dual_np, dtype=torch.float64, device=device).contiguous().view(-1)
        values_parts = []
        indices_parts = []
        for q_start in range(0, int(query_np.shape[0]), chunk_rows):
            q_stop = min(int(query_np.shape[0]), q_start + chunk_rows)
            query_t = torch.as_tensor(query_np[q_start:q_stop], dtype=torch.float32, device=device).contiguous()
            q_index = None if query_global_index is None else np.asarray(query_global_index)[q_start:q_stop]
            values_t, indices_t = run_chunk(
                query_t,
                database_t,
                dual_t,
                q_index,
                database_global_index,
            )
            values_parts.append(values_t[:, :k_int].contiguous())
            indices_parts.append(indices_t[:, :k_int].to(torch.int64).contiguous())
        return torch.cat(values_parts, dim=0), torch.cat(indices_parts, dim=0)

    query_t = torch.as_tensor(query_np, dtype=torch.float32, device=device).contiguous()
    best_values = torch.full((int(query_np.shape[0]), k_int), torch.inf, dtype=torch.float64, device=device)
    best_indices = torch.full((int(query_np.shape[0]), k_int), -1, dtype=torch.int64, device=device)
    for d_start in range(0, int(database_np.shape[0]), chunk_rows):
        d_stop = min(int(database_np.shape[0]), d_start + chunk_rows)
        database_t = torch.as_tensor(database_np[d_start:d_stop], dtype=torch.float32, device=device).contiguous()
        dual_t = torch.as_tensor(known_dual_np[d_start:d_stop], dtype=torch.float64, device=device).contiguous()
        d_index = None if database_global_index is None else np.asarray(database_global_index)[d_start:d_stop]
        values_t, indices_t = run_chunk(query_t, database_t, dual_t, query_global_index, d_index)
        candidate_values = torch.cat([best_values, values_t[:, :k_int]], dim=1)
        candidate_indices = torch.cat(
            [best_indices, indices_t[:, :k_int].to(torch.int64) + int(d_start)], dim=1
        )
        best_values, order = torch.topk(candidate_values, k=k_int, dim=1, largest=False, sorted=True)
        best_indices = torch.gather(candidate_indices, 1, order).contiguous()
    return best_values.contiguous(), best_indices.contiguous()


def detect_metric_violations_and_check_feasibility(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_dual: np.ndarray | torch.Tensor,
    target_dual: np.ndarray | torch.Tensor,
    cost_type: MetricCostType,
    topk: int,
    perturbation: Optional[Dict[str, Any]] = None,
    source_global_index: Optional[np.ndarray] = None,
    target_global_index: Optional[np.ndarray] = None,
    compute_precision: str = "fp32",
) -> DualViolationScanResult:
    """
    CN: 以 resident-streamed custom traversal 完成双向违例检测与完整 certificate。
    EN: Complete bidirectional violation detection and the full certificate with resident-streamed custom traversals.
    """
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("custom norm-cost fused refinement requires CUDA")
    requested_k = int(topk)
    compute_precision_id = _compute_precision_id(compute_precision)
    kernel_k = custom_topk_bucket(requested_k, score_family="norm_cost")
    source_np = np.asarray(source_points, dtype=np.float32, order="C")
    target_np = np.asarray(target_points, dtype=np.float32, order="C")
    source_dual_np = (
        source_dual.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(source_dual)
        else np.asarray(source_dual, dtype=np.float64)
    )
    target_dual_np = (
        target_dual.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(target_dual)
        else np.asarray(target_dual, dtype=np.float64)
    )
    device = torch.device("cuda")
    available_bytes, driver_free_bytes, reclaimable_cache_bytes = reusable_cuda_memory_bytes(device)
    memory_plan = plan_resident_side(
        source_count=int(source_np.shape[0]),
        target_count=int(target_np.shape[0]),
        feature_dim=int(source_np.shape[1]),
        topk_bucket=int(kernel_k),
        available_bytes=int(available_bytes),
        driver_free_bytes=int(driver_free_bytes),
        reclaimable_cache_bytes=int(reclaimable_cache_bytes),
    )
    ext = _require_norm_cost_scan_ext()
    _require_compute_precision_capability(ext, compute_precision_id)
    if not hasattr(ext, "fused_gcost_bidir_certificate"):
        raise RuntimeError(
            "Installed norm-cost scan extension is outdated and lacks fused_gcost_bidir_certificate; "
            "reinstall hierarchical_ot."
        )

    if memory_plan.resident_side == "target":
        resident_points_np, resident_dual_np = target_np, target_dual_np
        stream_points_np, stream_dual_np = source_np, source_dual_np
        resident_index, stream_index = target_global_index, source_global_index
        row_is_source = True
    else:
        resident_points_np, resident_dual_np = source_np, source_dual_np
        stream_points_np, stream_dual_np = target_np, target_dual_np
        resident_index, stream_index = source_global_index, target_global_index
        row_is_source = False
    resident_points_t = torch.as_tensor(resident_points_np, dtype=torch.float32, device=device).contiguous()
    resident_dual_t = torch.as_tensor(resident_dual_np, dtype=torch.float64, device=device).contiguous()
    n_resident = int(resident_points_np.shape[0])
    n_stream = int(stream_points_np.shape[0])
    # CN: norm-cost CUDA partials 与最终状态都使用 FP64 value 和 int64 index。
    # EN: Norm-cost CUDA partials and final state both use FP64 values and int64 indices.
    partial_bytes_per_block = max(1, n_resident * kernel_k * _TOPK_SLOT_BYTES)
    scratch_budget = max(0, int(memory_plan.usable_bytes - memory_plan.resident_bytes))
    minimum_stream_bytes = 8 * (int(stream_points_np.shape[1]) + 1) * 4
    if partial_bytes_per_block + minimum_stream_bytes > scratch_budget:
        raise RuntimeError(
            "norm-cost fused refinement exceeds single-GPU scratch capacity after placing the resident side: "
            f"requires at least {partial_bytes_per_block + minimum_stream_bytes} scratch bytes, "
            f"but {scratch_budget} bytes remain."
        )
    # CN: 最多将剩余 scratch 的一半分给 column partials，保留另一半给 stream feature 与临时输出。
    # EN: Give column partials at most half the remaining scratch, reserving the other half for streamed features and outputs.
    max_blocks = max(1, (scratch_budget // 2) // max(partial_bytes_per_block, 1))
    stream_row_bytes = max(1, (int(stream_points_np.shape[1]) + 4) * 4)
    max_feature_rows = (scratch_budget // 2) // stream_row_bytes
    stream_chunk = min(n_stream, 8192, int(max_blocks) * 8, int(max_feature_rows))
    if stream_chunk < 1:
        raise RuntimeError(
            "norm-cost fused refinement cannot fit one streamed feature row after reserving column partials"
        )

    resident_values = torch.full((n_resident, requested_k), -torch.inf, dtype=torch.float64, device=device)
    resident_indices = torch.full((n_resident, requested_k), -1, dtype=torch.int64, device=device)
    stream_values_parts = []
    stream_indices_parts = []
    total_num = torch.zeros((), dtype=torch.float64, device=device)
    total_den = torch.zeros((), dtype=torch.float64, device=device)
    max_violation = torch.zeros((), dtype=torch.float64, device=device)
    cost_linf = torch.zeros((), dtype=torch.float64, device=device)
    positive_count = torch.zeros((), dtype=torch.int64, device=device)
    # CN: K=32 的通用 fused column reduction 在中等维度慢于三次 custom scan；专用归约完成前使用分解路径。
    # EN: The generic K=32 fused column reduction is slower than three custom scans at medium d; use the decomposed path until a specialized reduction exists.
    use_decomposed_large_k = int(kernel_k) >= 32

    for start in range(0, n_stream, stream_chunk):
        stop = min(n_stream, start + stream_chunk)
        query_t = torch.as_tensor(stream_points_np[start:stop], dtype=torch.float32, device=device).contiguous()
        query_dual_t = torch.as_tensor(stream_dual_np[start:stop], dtype=torch.float64, device=device).contiguous()
        query_index = None if stream_index is None else np.asarray(stream_index)[start:stop]
        sigma, seed, mode, q_idx_t, d_idx_t, coeffs = _fused_perturb_arguments(
            perturbation=perturbation,
            query_global_index=query_index,
            database_global_index=resident_index,
            device=device,
        )
        if use_decomposed_large_k:
            q_cost_values, q_indices = ext.fused_gcost_topk(
                query_t,
                resident_points_t,
                resident_dual_t,
                int(kernel_k),
                int(_fused_cost_type_id(cost_type)),
                q_idx_t,
                d_idx_t,
                float(sigma),
                int(seed),
                mode,
                bool(row_is_source),
                coeffs,
                compute_precision_id,
            )
            r_cost_values, r_indices = ext.fused_gcost_topk(
                resident_points_t,
                query_t,
                query_dual_t,
                int(kernel_k),
                int(_fused_cost_type_id(cost_type)),
                d_idx_t,
                q_idx_t,
                float(sigma),
                int(seed),
                mode,
                not bool(row_is_source),
                coeffs,
                compute_precision_id,
            )
            num_t, den_t, max_t, cost_linf_t, count_t = ext.fused_gcost_certificate(
                query_t,
                resident_points_t,
                query_dual_t,
                resident_dual_t,
                int(_fused_cost_type_id(cost_type)),
                q_idx_t,
                d_idx_t,
                float(sigma),
                int(seed),
                mode,
                bool(row_is_source),
                coeffs,
                1,
                compute_precision_id,
            )
            q_values = query_dual_t[:, None] - q_cost_values
            r_values = resident_dual_t[:, None] - r_cost_values
        else:
            q_values, q_indices, r_values, r_indices, num_t, den_t, max_t, cost_linf_t, count_t = (
                ext.fused_gcost_bidir_certificate(
                    query_t,
                    resident_points_t,
                    query_dual_t,
                    resident_dual_t,
                    int(kernel_k),
                    int(_fused_cost_type_id(cost_type)),
                    q_idx_t,
                    d_idx_t,
                    float(sigma),
                    int(seed),
                    mode,
                    bool(row_is_source),
                    coeffs,
                    compute_precision_id,
                )
            )
        stream_values_parts.append(q_values[:, :requested_k].contiguous())
        stream_indices_parts.append(q_indices[:, :requested_k].to(torch.int64).contiguous())
        candidate_values = torch.cat([resident_values, r_values[:, :requested_k]], dim=1)
        candidate_indices = torch.cat(
            [resident_indices, r_indices[:, :requested_k].to(torch.int64) + int(start)], dim=1
        )
        resident_values, order = torch.topk(
            candidate_values, k=requested_k, dim=1, largest=True, sorted=True
        )
        resident_indices = torch.gather(candidate_indices, 1, order).contiguous()
        total_num.add_(num_t)
        total_den.add_(den_t)
        max_violation = torch.maximum(max_violation, max_t)
        cost_linf = torch.maximum(cost_linf, cost_linf_t)
        positive_count.add_(count_t)

    stream_values = torch.cat(stream_values_parts, dim=0)
    stream_indices = torch.cat(stream_indices_parts, dim=0)
    if memory_plan.resident_side == "target":
        source_values, source_indices = stream_values, stream_indices
        target_values, target_indices = resident_values, resident_indices
    else:
        source_values, source_indices = resident_values, resident_indices
        target_values, target_indices = stream_values, stream_indices
    return DualViolationScanResult(
        source_values=source_values,
        source_indices=source_indices,
        target_values=target_values,
        target_indices=target_indices,
        certificate=DualFeasibilityCertificate(
            l2_numerator_sq=total_num,
            l2_denominator_sq=total_den,
            max_positive_violation=max_violation,
            cost_linf=cost_linf,
            positive_count=positive_count,
        ),
        diagnostics={
            "scan_backend": "custom",
            "score_family": "norm_cost",
            "cost_compute_precision": str(compute_precision),
            "resident_side": str(memory_plan.resident_side),
            "resident_bytes": int(memory_plan.resident_bytes),
            "stream_chunk_rows": int(stream_chunk),
            "topk": int(requested_k),
            "topk_bucket": int(kernel_k),
            "certificate_mode": "l2_and_linf",
            "pairwise_passes": 3 if use_decomposed_large_k else 1,
            "scan_traversal": "decomposed_large_k_custom" if use_decomposed_large_k else "fused_one_pass_custom",
        },
    )


def check_metric_dual_feasibility(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_dual: np.ndarray | torch.Tensor,
    target_dual: np.ndarray | torch.Tensor,
    cost_type: MetricCostType,
    perturbation: Optional[Dict[str, Any]] = None,
    source_global_index: Optional[np.ndarray] = None,
    target_global_index: Optional[np.ndarray] = None,
    compute_denominator: bool = True,
    compute_precision: str = "fp32",
) -> Tuple[DualFeasibilityCertificate, Dict[str, Any]]:
    """
    CN: 单侧常驻、另一侧流式地计算完整 L2/Linf dual-feasibility certificate。
    EN: Compute the complete L2/Linf dual-feasibility certificate with one resident side and one streamed side.
    """
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("custom norm-cost certificate requires CUDA")
    compute_precision_id = _compute_precision_id(compute_precision)
    source_np = np.asarray(source_points, dtype=np.float32, order="C")
    target_np = np.asarray(target_points, dtype=np.float32, order="C")
    source_dual_np = (
        source_dual.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(source_dual)
        else np.asarray(source_dual, dtype=np.float64)
    )
    target_dual_np = (
        target_dual.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(target_dual)
        else np.asarray(target_dual, dtype=np.float64)
    )
    device = torch.device("cuda")
    available_bytes, driver_free_bytes, reclaimable_cache_bytes = reusable_cuda_memory_bytes(device)
    memory_plan = plan_resident_side(
        source_count=int(source_np.shape[0]),
        target_count=int(target_np.shape[0]),
        feature_dim=int(source_np.shape[1]),
        topk_bucket=1,
        available_bytes=int(available_bytes),
        driver_free_bytes=int(driver_free_bytes),
        reclaimable_cache_bytes=int(reclaimable_cache_bytes),
    )
    ext = _require_norm_cost_scan_ext()
    _require_compute_precision_capability(ext, compute_precision_id)
    cid = int(_fused_cost_type_id(cost_type))
    total_num = torch.zeros((), dtype=torch.float64, device=device)
    total_den = torch.zeros((), dtype=torch.float64, device=device)
    max_violation = torch.zeros((), dtype=torch.float64, device=device)
    cost_linf = torch.zeros((), dtype=torch.float64, device=device)
    positive_count = torch.zeros((), dtype=torch.int64, device=device)

    if memory_plan.resident_side == "target":
        resident_points = torch.as_tensor(target_np, dtype=torch.float32, device=device).contiguous()
        resident_dual = torch.as_tensor(target_dual_np, dtype=torch.float64, device=device).contiguous()
        stream_points = source_np
        stream_dual = source_dual_np
        stream_index = source_global_index
        resident_index = target_global_index
        row_is_source = True
    else:
        resident_points = torch.as_tensor(source_np, dtype=torch.float32, device=device).contiguous()
        resident_dual = torch.as_tensor(source_dual_np, dtype=torch.float64, device=device).contiguous()
        stream_points = target_np
        stream_dual = target_dual_np
        stream_index = target_global_index
        resident_index = source_global_index
        row_is_source = False

    chunk_rows = plan_stream_chunk_rows(
        memory_plan,
        streamed_count=int(stream_points.shape[0]),
        feature_dim=int(stream_points.shape[1]),
        max_rows=8192,
    )

    for start in range(0, int(stream_points.shape[0]), chunk_rows):
        stop = min(int(stream_points.shape[0]), start + chunk_rows)
        query_t = torch.as_tensor(stream_points[start:stop], dtype=torch.float32, device=device).contiguous()
        query_dual_t = torch.as_tensor(stream_dual[start:stop], dtype=torch.float64, device=device).contiguous()
        query_index = None if stream_index is None else np.asarray(stream_index)[start:stop]
        sigma, seed, mode, q_idx_t, d_idx_t, coeffs = _fused_perturb_arguments(
            perturbation=perturbation,
            query_global_index=query_index,
            database_global_index=resident_index,
            device=device,
        )
        num_t, den_t, max_t, cost_linf_t, count_t = ext.fused_gcost_certificate(
            query_t,
            resident_points,
            query_dual_t,
            resident_dual,
            cid,
            q_idx_t,
            d_idx_t,
            float(sigma),
            int(seed),
            mode,
            bool(row_is_source),
            coeffs,
            int(bool(compute_denominator)),
            compute_precision_id,
        )
        total_num.add_(num_t)
        if bool(compute_denominator):
            total_den.add_(den_t)
        max_violation = torch.maximum(max_violation, max_t)
        cost_linf = torch.maximum(cost_linf, cost_linf_t)
        positive_count.add_(count_t)

    certificate = DualFeasibilityCertificate(
        l2_numerator_sq=total_num,
        l2_denominator_sq=total_den,
        max_positive_violation=max_violation,
        cost_linf=cost_linf,
        positive_count=positive_count,
    )
    return certificate, {
        "scan_backend": "custom",
        "score_family": "norm_cost",
        "cost_compute_precision": str(compute_precision),
        "resident_side": str(memory_plan.resident_side),
        "resident_bytes": int(memory_plan.resident_bytes),
        "stream_chunk_rows": int(chunk_rows),
        "certificate_mode": "l2_and_linf",
        "pairwise_passes": 1,
    }
