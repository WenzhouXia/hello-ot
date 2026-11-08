from __future__ import annotations

from hello_ot.kernels.scan_planner import (
    streamed_memory_estimate_mib as _raw_memory_estimate_mib,
    select_streamed_tiles,
)
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from hello_ot.kernels.scan_contract import (
    custom_topk_bucket,
    plan_resident_side,
    reusable_cuda_memory_bytes,
)

_EXT_MODULE: Any = None
_EXT_LOAD_ERROR: str | None = None


@dataclass
class LowrankBidirScanResult:
    """
    CN: raw bidirectional lowrank scan 的结果。
    EN: Result of the raw bidirectional lowrank scan.
    """

    dual_feasibility: float
    rows: np.ndarray
    cols: np.ndarray
    diagnostics: Dict[str, Any]


def _load_extension() -> Any:
    """
    CN: 加载统一扫描入口，并验证发行包中的成本扩展。
    EN: Load unified scan entry points and verify the packaged cost extension.
    """
    global _EXT_MODULE, _EXT_LOAD_ERROR
    if _EXT_MODULE is None:
        from hello_ot.kernels.generic_scan import ScanExtension, _module
        try:
            _module("cost")
        except Exception as exc:
            _EXT_LOAD_ERROR = f"Unified scan CUDA extension unavailable; rebuild hello-ot-native: {exc!r}"
            raise RuntimeError(_EXT_LOAD_ERROR) from exc
        _EXT_MODULE = ScanExtension()
    return _EXT_MODULE


def extension_available() -> bool:
    try:
        _load_extension()
        return True
    except Exception:  # noqa: BLE001
        return False


def last_extension_error() -> str | None:
    return _EXT_LOAD_ERROR


def _parse_supported_topk(value: float) -> int:
    k_round = int(round(float(value)))
    if abs(float(value) - float(k_round)) > 1e-9:
        raise ValueError("fused inner-product scan requires an integer pricing_topk")
    custom_topk_bucket(k_round, score_family="inner_product")
    return int(k_round)


def validate_fused_lowrank_bidir_scan_config(config: Any, *, cost_type: str) -> int:
    """
    CN: 校验 warm_start fused scan 支持范围，并返回整数 top-k。
    EN: Validate the supported warm-start fused scan scope and return integer top-k.
    """
    if str(cost_type).lower() != "lowrank":
        raise ValueError("fused lowrank feasibility pricing only supports cost_type='lowrank'")
    if str(getattr(config, "pricing_strategy", "")).lower() != "nodewise_full":
        raise ValueError("fused lowrank feasibility pricing only supports pricing_strategy='nodewise_full'")
    if str(getattr(config, "pricing_direction", "")).lower() != "both":
        raise ValueError("fused lowrank feasibility pricing only supports pricing_direction='both'")
    if not torch.cuda.is_available():
        raise RuntimeError("fused lowrank feasibility pricing requires CUDA")
    _load_extension()
    return _parse_supported_topk(float(getattr(config, "pricing_topk", 1.0)))


def _empty_cpu_staging(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    CN: 分配有界 pinned staging；不支持 pinned allocation 时保留同样有界的普通 CPU buffer。
    EN: Allocate bounded pinned staging, retaining the same bounded ordinary CPU buffer when pinning is unavailable.
    """
    try:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    except Exception:  # noqa: BLE001
        return torch.empty(shape, dtype=dtype, device="cpu")


def _copy_cpu_slice_to_staging(
    array: Any,
    *,
    start: int,
    end: int,
    staging: torch.Tensor,
) -> torch.Tensor:
    """
    CN: 把一个 CPU slice 写入复用 staging，不物化完整 pinned feature。
    EN: Copy one CPU slice into reusable staging without materializing a full pinned feature.
    """
    rows = int(end) - int(start)
    output = staging[:rows]
    if torch.is_tensor(array):
        source = array.detach()[int(start):int(end)].to(
            device="cpu", dtype=staging.dtype
        ).contiguous()
        output.copy_(source)
        return output
    np_dtype = np.float64 if staging.dtype == torch.float64 else np.float32
    source_np = np.asarray(array[int(start):int(end)], dtype=np_dtype, order="C")
    np.copyto(output.numpy(), source_np)
    return output


def _to_device_1d(array: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(array):
        return array.detach().to(device=device, dtype=torch.float64).contiguous().view(-1)
    return torch.as_tensor(np.ascontiguousarray(array, dtype=np.float64), dtype=torch.float64, device=device).contiguous().view(-1)


def _to_device_2d(array: Any, device: torch.device, *, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(array):
        return array.detach().to(device=device, dtype=dtype).contiguous()
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    return torch.as_tensor(np.ascontiguousarray(array, dtype=np_dtype), dtype=dtype, device=device).contiguous()


def _merge_topk(old_vals: torch.Tensor, old_idx: torch.Tensor, new_vals: torch.Tensor, new_idx: torch.Tensor, *, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
    vals = torch.cat([old_vals, new_vals], dim=1)
    idx = torch.cat([old_idx, new_idx], dim=1)
    merged_vals, order = torch.topk(vals, int(topk), dim=1, largest=True, sorted=True)
    merged_idx = torch.gather(idx, 1, order)
    return merged_vals.contiguous(), merged_idx.contiguous()


def _select_tiles(**kwargs):
    """
    CN: 扫描后端在同一规划入口申报工作区，执行器直接使用规划结果。
    EN: Scan backends declare scratch through one planning entry point, and executors consume the resulting plan directly.
    """
    return select_streamed_tiles(**kwargs)




def _flatten_positive_query_candidates(
    vals: torch.Tensor,
    idx: torch.Tensor,
    *,
    query_offset: int,
    query_is_source: bool,
    theta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (vals > float(theta)) & (idx >= 0)
    if not bool(mask.any().item()):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    query_local = torch.nonzero(mask, as_tuple=False)[:, 0].to(torch.int64) + int(query_offset)
    match = idx[mask].to(torch.int64)
    if bool(query_is_source):
        rows_t = query_local
        cols_t = match
    else:
        rows_t = match
        cols_t = query_local
    return rows_t.detach().cpu().numpy().astype(np.int64, copy=False), cols_t.detach().cpu().numpy().astype(np.int64, copy=False)


def _flatten_positive_resident_candidates(
    vals: torch.Tensor,
    idx: torch.Tensor,
    *,
    resident_is_source: bool,
    theta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (vals > float(theta)) & (idx >= 0)
    if not bool(mask.any().item()):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    resident = torch.nonzero(mask, as_tuple=False)[:, 0].to(torch.int64)
    match = idx[mask].to(torch.int64)
    if bool(resident_is_source):
        rows_t = resident
        cols_t = match
    else:
        rows_t = match
        cols_t = resident
    return rows_t.detach().cpu().numpy().astype(np.int64, copy=False), cols_t.detach().cpu().numpy().astype(np.int64, copy=False)


def raw_lowrank_bidir_feasibility_pricing_scan(
    source_feat: Any,
    target_feat: Any,
    source_cost: Any,
    target_cost: Any,
    source_dual: Any,
    target_dual: Any,
    *,
    topk: int,
    theta: float = 0.0,
    gpu_id: int = 0,
    memory_floor_mib: float = 768.0,
    resident_multiplier: float = 1.10,
    col_reduce_tile: int = 256,
    col_kernel: str = "auto",
    collect_timing: bool = False,
    collect_linf_diagnostics: bool = False,
    dot_scale: float = 1.0,
    compute_precision: str = "fp32",
) -> LowrankBidirScanResult:
    """
    CN: 执行 raw lowrank 双向 top-k scan，同时计算 dual feasibility。
    EN: Run a raw lowrank bidirectional top-k scan while computing dual feasibility.
    """
    # CN: 校验 top-k/CUDA/extension，并确定本次 scan 使用的 GPU。
    # EN: Validate top-k/CUDA/extension availability and choose the GPU for this scan.
    k = _parse_supported_topk(int(topk))
    kernel_k = custom_topk_bucket(k, score_family="inner_product")
    if not torch.cuda.is_available():
        raise RuntimeError("fused lowrank feasibility pricing requires CUDA")
    ext = _load_extension()
    device = torch.device(f"cuda:{int(gpu_id)}")
    if str(compute_precision) not in {"fp32", "fp64"}:
        raise ValueError("compute_precision must be one of: fp32, fp64")
    feature_dtype = torch.float64 if str(compute_precision) == "fp64" else torch.float32
    feature_itemsize = 8 if feature_dtype == torch.float64 else 4

    # CN: 读取问题规模并检查 source/target 的 lowrank feature 维度一致。
    # EN: Read problem sizes and ensure source/target lowrank feature dimensions match.
    n_source = int(source_feat.shape[0])
    n_target = int(target_feat.shape[0])
    if n_source <= 0 or n_target <= 0:
        raise ValueError("source and target must both be non-empty")
    d = int(source_feat.shape[1])
    if int(target_feat.shape[1]) != d:
        raise ValueError("source and target feature dimensions must match")

    with nullcontext() as memory_span:
        available_bytes, driver_free_bytes, reclaimable_cache_bytes = reusable_cuda_memory_bytes(device)
        memory_plan = plan_resident_side(
            source_count=int(n_source),
            target_count=int(n_target),
            feature_dim=int(d * feature_itemsize // 4),
            topk_bucket=int(kernel_k),
            available_bytes=int(available_bytes),
            driver_free_bytes=int(driver_free_bytes),
            reclaimable_cache_bytes=int(reclaimable_cache_bytes),
        )
        # CN: planner 选择能完整常驻的一侧，另一侧流式上传。
        # EN: The planner selects one side that fully fits and streams the other side.
        resident_is_source = memory_plan.resident_side == "source"
        if resident_is_source:
            resident_feat = _to_device_2d(source_feat, device, dtype=feature_dtype)
            resident_cost = _to_device_1d(source_cost, device)
            resident_dual = _to_device_1d(source_dual, device)
            stream_feat_source = target_feat
            stream_cost_source = target_cost
            stream_dual_source = target_dual
        else:
            resident_feat = _to_device_2d(target_feat, device, dtype=feature_dtype)
            resident_cost = _to_device_1d(target_cost, device)
            resident_dual = _to_device_1d(target_dual, device)
            stream_feat_source = source_feat
            stream_cost_source = source_cost
            stream_dual_source = source_dual

        # CN: 根据 resident/stream 规模和显存预算选择 stream chunk、query tile、database tile。
        # EN: Select stream chunk, query tile, and database tile from resident/stream sizes and the memory budget.
        n_resident = int(resident_feat.shape[0])
        n_stream = int(stream_feat_source.shape[0])
        tile = _select_tiles(
            n_stream=int(n_stream),
            n_resident=int(n_resident),
            d=int(d),
            topk=int(kernel_k),
            memory_floor_mib=float(memory_floor_mib),
            resident_multiplier=float(resident_multiplier),
            max_budget_mib=float(memory_plan.usable_bytes) / float(1024 * 1024),
            feature_itemsize=int(feature_itemsize),
        )
        source_chunk = int(tile["source_chunk"])
        query_tile = int(tile["query_tile"])
        db_tile = int(tile["db_tile"])
        staging_rows = min(int(source_chunk), int(n_stream))
        # CN: 双缓冲允许 CPU 准备下一块时上一块仍在 GPU 计算；总 pinned 内存固定为 O(chunk*d)。
        # EN: Double buffering lets the CPU prepare the next chunk while the previous chunk runs on GPU; pinned memory stays O(chunk*d).
        feature_staging = [
            _empty_cpu_staging((staging_rows, d), dtype=feature_dtype)
            for _ in range(2)
        ]
        cost_staging = [
            _empty_cpu_staging((staging_rows,), dtype=torch.float64)
            for _ in range(2)
        ]
        dual_staging = [
            _empty_cpu_staging((staging_rows,), dtype=torch.float64)
            for _ in range(2)
        ]
        staging_events: list[torch.cuda.Event | None] = [None, None]
        # CN: 选择 CUDA extension 内部的列归约 kernel 变体；auto 会按 top-k 选择。
        # EN: Select the column-reduction kernel variant used by the CUDA extension; auto depends on top-k.
        if str(col_kernel) == "auto":
            col_kernel_id = 2 if int(kernel_k) == 2 else (1 if int(kernel_k) == 4 else 0)
        else:
            col_kernel_id = {"current": 0, "warp4": 1, "warp2": 2}.get(str(col_kernel))
        if col_kernel_id is None:
            raise ValueError("col_kernel must be one of {auto, current, warp2, warp4}")

        # CN: 初始化 resident 侧常驻 buffer 和累计统计量；这些 tensor 贯穿整个 scan 生命周期。
        # EN: Initialize resident-side persistent buffers and accumulated statistics; these tensors live for the full scan.
        resident_bias = (resident_dual - resident_cost).contiguous()
        resident_vals = torch.full((n_resident, int(k)), -torch.inf, dtype=torch.float64, device=device)
        resident_idx = torch.full((n_resident, int(k)), -1, dtype=torch.int64, device=device)
        total_stats = torch.zeros((4,), dtype=torch.float64, device=device)
        rows_parts: List[np.ndarray] = []
        cols_parts: List[np.ndarray] = []
        max_violation = 0.0
        start_time = time.perf_counter()
        # CN: 所有支持的k均在一次完整遍历中计算双向候选和证书。
        # EN: Compute both candidate directions and the certificate in one traversal for every supported k.
        for chunk_index, start in enumerate(range(0, int(n_stream), int(source_chunk))):
            end = min(int(n_stream), int(start) + int(source_chunk))
            staging_slot = int(chunk_index % 2)
            prior_copy = staging_events[staging_slot]
            if prior_copy is not None:
                prior_copy.synchronize()
            query_feat_cpu = _copy_cpu_slice_to_staging(
                stream_feat_source,
                start=int(start),
                end=int(end),
                staging=feature_staging[staging_slot],
            )
            query_cost_cpu = _copy_cpu_slice_to_staging(
                stream_cost_source,
                start=int(start),
                end=int(end),
                staging=cost_staging[staging_slot],
            )
            query_dual_cpu = _copy_cpu_slice_to_staging(
                stream_dual_source,
                start=int(start),
                end=int(end),
                staging=dual_staging[staging_slot],
            )
            # CN: 当前 chunk 的 feature/cost/dual 会临时驻留 GPU，是该循环内主要的 PyTorch 分配之一。
            # EN: The current chunk's feature/cost/dual temporarily live on GPU and are one main PyTorch allocation in this loop.
            query_feat = query_feat_cpu.to(device=device, dtype=feature_dtype, non_blocking=True).contiguous()
            query_cost = query_cost_cpu.to(device=device, dtype=torch.float64, non_blocking=True).contiguous()
            query_dual = query_dual_cpu.to(device=device, dtype=torch.float64, non_blocking=True).contiguous()
            copy_complete = torch.cuda.Event()
            copy_complete.record(torch.cuda.current_stream(device))
            staging_events[staging_slot] = copy_complete
            query_bias = (query_dual - query_cost).contiguous()
            q_vals, q_idx, r_vals, r_idx, stats, _phase_ms = ext.fused_bidir_topk_stats_raw(
                query_feat,
                resident_feat,
                query_cost,
                resident_cost,
                query_bias,
                resident_bias,
                int(query_tile),
                int(db_tile),
                int(col_reduce_tile),
                int(kernel_k),
                True,
                bool(collect_linf_diagnostics),
                False,
                int(col_kernel_id),
                float(dot_scale),
            )
            # CN: query 侧 top-k 立即筛出正 dual-score 候选边并转回 CPU 保存。
            # EN: Query-side top-k candidates are filtered for positive dual score and copied back to CPU immediately.
            q_vals = q_vals[:, : int(k)].contiguous()
            q_idx = q_idx[:, : int(k)].contiguous()
            r_vals = r_vals[:, : int(k)].contiguous()
            r_idx = r_idx[:, : int(k)].contiguous()
            q_rows, q_cols = _flatten_positive_query_candidates(
                q_vals,
                q_idx,
                query_offset=int(start),
                query_is_source=not bool(resident_is_source),
                theta=float(theta),
            )
            max_violation = max(
                max_violation,
                float(torch.max(q_vals).item()),
            )
            if q_rows.size > 0:
                rows_parts.append(q_rows)
                cols_parts.append(q_cols)
            # CN: resident 侧需要跨所有 stream chunk 合并 top-k，因此 resident_vals/resident_idx 持续更新到循环结束。
            # EN: The resident side must merge top-k across all stream chunks, so resident_vals/resident_idx persist until the loop ends.
            resident_vals, resident_idx = _merge_topk(
                resident_vals,
                resident_idx,
                r_vals,
                r_idx.to(torch.int64) + int(start),
                topk=int(k),
            )
            stats64 = stats.to(torch.float64)
            total_stats[:3].add_(stats64[:3])
            if bool(collect_linf_diagnostics):
                total_stats[3] = torch.maximum(total_stats[3], stats64[3])

        # CN: 所有 stream chunk 扫描完成后，再从 resident 侧累计 top-k 中抽取正候选边。
        # EN: After all stream chunks are scanned, extract positive candidates from the accumulated resident-side top-k.
        r_rows, r_cols = _flatten_positive_resident_candidates(
            resident_vals,
            resident_idx,
            resident_is_source=bool(resident_is_source),
            theta=float(theta),
        )
        if r_rows.size > 0:
            rows_parts.append(r_rows)
            cols_parts.append(r_cols)

        # CN: 将 kernel 累计统计量转回 CPU，计算双可行性指标。
        # EN: Move accumulated kernel statistics back to CPU and compute the dual-feasibility metric.
        stats_cpu = total_stats.detach().cpu().to(torch.float64)
        num_sq = float(stats_cpu[0].item())
        den_sq = float(stats_cpu[1].item())
        positive_count = int(round(float(stats_cpu[2].item())))
        cost_linf = float(stats_cpu[3].item()) if bool(collect_linf_diagnostics) else None
        dual_feas = math.sqrt(max(num_sq, 0.0)) / (1.0 + math.sqrt(max(den_sq, 0.0)))
        # CN: 合并 query/resident 两侧候选边；没有候选时返回空 int64 数组以保持下游类型稳定。
        # EN: Merge query/resident candidate edges; return empty int64 arrays when no candidates exist to keep downstream types stable.
        if rows_parts:
            rows = np.concatenate(rows_parts).astype(np.int64, copy=False)
            cols = np.concatenate(cols_parts).astype(np.int64, copy=False)
        else:
            rows = np.empty(0, dtype=np.int64)
            cols = np.empty(0, dtype=np.int64)
        elapsed = time.perf_counter() - start_time
        # CN: diagnostics 记录实际 scan 结果和 tile 选择；其中 tile 的 est_peak 是预算估算，不是实测 driver peak。
        # EN: Diagnostics record scan results and tile selection; tile est_peak is a budget estimate, not a measured driver peak.
        diagnostics: Dict[str, Any] = {
            "dual_feasibility": float(dual_feas),
            "kernel_backend": ext.last_backend,
            "memory_estimate_scope": "active_geometry",
            "dual_feasibility_num_sq": float(num_sq),
            "dual_feasibility_den_sq": float(den_sq),
            "dual_feasibility_positive_count": int(positive_count),
            "dual_feasibility_max_violation": float(max(max_violation, 0.0)),
            "dual_feasibility_source": "early_fused_lowrank_bidir_scan",
            "fused_scan_time": float(elapsed),
            "fused_scan_candidate_count": int(rows.size),
            "fused_scan_resident_side": "source" if resident_is_source else "target",
            "fused_scan_resident_bytes": int(memory_plan.resident_bytes),
            "fused_scan_n_source": int(n_source),
            "fused_scan_n_target": int(n_target),
            "fused_scan_topk": int(k),
            "fused_scan_topk_bucket": int(kernel_k),
            "fused_scan_col_kernel_id": int(col_kernel_id),
            "pairwise_passes": 1,
            "scan_traversal": "fused_one_pass_custom",
            "host_staging_mode": "bounded_double_buffer",
            "host_staging_rows": int(staging_rows),
            "host_staging_buffer_count": 2,
            "host_staging_pinned": bool(all(buffer.is_pinned() for buffer in feature_staging)),
            "host_staging_bytes": int(
                2 * int(staging_rows) * (int(d) * int(feature_itemsize) + 2 * 8)
            ),
            "cost_compute_precision": str(compute_precision),
        }
        if cost_linf is not None:
            diagnostics.update(
                {
                    "dual_feasibility_cost_linf": float(cost_linf),
                    "relative_linf_dual_feasibility": float(max(max_violation, 0.0))
                    / (1.0 + float(cost_linf)),
                }
            )
        diagnostics.update({f"fused_scan_{key}": value for key, value in tile.items()})
        if memory_span is not None:
            memory_span.update_metadata(dict(diagnostics))
        return LowrankBidirScanResult(
            dual_feasibility=float(dual_feas),
            rows=rows,
            cols=cols,
            diagnostics=diagnostics,
        )
