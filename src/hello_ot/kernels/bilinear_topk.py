from __future__ import annotations

import importlib
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
_EXT_MODULE_NAME = "hello_ot._native.inner_product_scan.hierot_inner_product_scan_ext"


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
    global _EXT_MODULE, _EXT_LOAD_ERROR
    if _EXT_MODULE is not None:
        return _EXT_MODULE
    if _EXT_LOAD_ERROR is not None:
        raise RuntimeError(_EXT_LOAD_ERROR)
    try:
        module = importlib.import_module(_EXT_MODULE_NAME)
    except Exception as exc:  # noqa: BLE001
        _EXT_LOAD_ERROR = (
            f"Failed to import installed lowrank scan CUDA extension {_EXT_MODULE_NAME!r}. "
            f"Reinstall hello-ot so the extension is built. Original error: {exc!r}"
        )
        raise RuntimeError(_EXT_LOAD_ERROR) from exc
    if not hasattr(module, "fused_bidir_topk_stats_raw"):
        _EXT_LOAD_ERROR = (
            f"Installed CUDA extension {_EXT_MODULE_NAME!r} does not expose "
            "fused_bidir_topk_stats_raw. Reinstall hello-ot."
        )
        raise RuntimeError(_EXT_LOAD_ERROR)
    _EXT_MODULE = module
    return module


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


def _to_device_2d(array: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(array):
        return array.detach().to(device=device, dtype=torch.float32).contiguous()
    return torch.as_tensor(np.ascontiguousarray(array, dtype=np.float32), dtype=torch.float32, device=device).contiguous()


def _merge_topk(old_vals: torch.Tensor, old_idx: torch.Tensor, new_vals: torch.Tensor, new_idx: torch.Tensor, *, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
    vals = torch.cat([old_vals, new_vals], dim=1)
    idx = torch.cat([old_idx, new_idx], dim=1)
    merged_vals, order = torch.topk(vals, int(topk), dim=1, largest=True, sorted=True)
    merged_idx = torch.gather(idx, 1, order)
    return merged_vals.contiguous(), merged_idx.contiguous()


def _raw_memory_estimate_mib(
    *,
    n_stream: int,
    n_resident: int,
    d: int,
    topk: int,
    stream_chunk: int,
    query_tile: int,
    db_tile: int,
) -> Dict[str, float]:
    stream_chunk_eff = max(1, min(int(stream_chunk), int(n_stream)))
    query_tile_eff = max(1, min(int(query_tile), stream_chunk_eff))
    db_tile_eff = max(1, min(int(db_tile), int(n_resident)))
    mib = float(1024 * 1024)
    resident_mib = float(n_resident) * float(d + 2) * 4.0 / mib
    stream_chunk_mib = float(stream_chunk_eff) * float(d + 2) * 4.0 / mib
    score_tile_mib = float(query_tile_eff) * float(db_tile_eff) * 4.0 / mib
    partial_resident_mib = float(math.ceil(float(stream_chunk_eff) / float(query_tile_eff))) * float(n_resident) * float(topk) * 8.0 / mib
    final_outputs_mib = float(n_resident + stream_chunk_eff) * float(topk) * 8.0 / mib
    est_peak_mib = resident_mib + stream_chunk_mib + score_tile_mib + partial_resident_mib + final_outputs_mib
    return {
        "resident_feature_mib": float(resident_mib),
        "stream_chunk_mib": float(stream_chunk_mib),
        "score_tile_mib": float(score_tile_mib),
        "partial_resident_mib": float(partial_resident_mib),
        "final_outputs_mib": float(final_outputs_mib),
        "est_peak_mib": float(est_peak_mib),
    }


def _select_tiles(
    *,
    n_stream: int,
    n_resident: int,
    d: int,
    topk: int,
    memory_floor_mib: float,
    resident_multiplier: float,
    max_budget_mib: float,
) -> Dict[str, Any]:
    resident_mib = _raw_memory_estimate_mib(
        n_stream=int(n_stream),
        n_resident=int(n_resident),
        d=int(d),
        topk=int(topk),
        stream_chunk=1,
        query_tile=1,
        db_tile=1,
    )["resident_feature_mib"]
    target_budget_mib = min(
        float(max_budget_mib),
        max(float(memory_floor_mib), float(resident_multiplier) * float(resident_mib)),
    )
    candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int]] = set()
    for query_candidate in (1024, 2048, 4096, 8192):
        for stream_mult in (1, 2):
            stream_chunk = min(int(n_stream), int(query_candidate) * int(stream_mult))
            query_tile = min(int(query_candidate), int(stream_chunk))
            if stream_chunk <= 0 or query_tile <= 0:
                continue
            for db_candidate in (8192, 16384, 32768, 65536):
                db_tile = min(int(db_candidate), int(n_resident))
                key = (int(stream_chunk), int(query_tile), int(db_tile))
                if key in seen:
                    continue
                seen.add(key)
                mem = _raw_memory_estimate_mib(
                    n_stream=int(n_stream),
                    n_resident=int(n_resident),
                    d=int(d),
                    topk=int(topk),
                    stream_chunk=int(stream_chunk),
                    query_tile=int(query_tile),
                    db_tile=int(db_tile),
                )
                row: Dict[str, Any] = {
                    "source_chunk": int(stream_chunk),
                    "query_tile": int(query_tile),
                    "db_tile": int(db_tile),
                    "target_budget_mib": float(target_budget_mib),
                    "fits_soft_budget": bool(mem["est_peak_mib"] <= target_budget_mib),
                }
                row.update(mem)
                candidates.append(row)
    preferred = [row for row in candidates if bool(row["fits_soft_budget"])]
    if not preferred:
        minimum_peak = min(float(row["est_peak_mib"]) for row in candidates)
        raise RuntimeError(
            "fused inner-product refinement exceeds single-GPU scratch capacity: "
            f"minimum estimated peak is {minimum_peak:.1f} MiB, budget is {target_budget_mib:.1f} MiB."
        )
    pool = preferred

    def _rank(row: Dict[str, Any]) -> Tuple[float, int, int, int]:
        source_bonus = 1 if int(row["source_chunk"]) == int(row["query_tile"]) else 0
        return (float(source_bonus), int(row["db_tile"]), int(row["query_tile"]), -int(row["source_chunk"]))

    if not pool:
        raise RuntimeError("no fused lowrank scan tile candidates were generated")
    selected = dict(max(pool, key=_rank))
    selected["candidate_count"] = int(len(candidates))
    selected["preferred_candidate_count"] = int(len(preferred))
    return selected


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
            feature_dim=int(d),
            topk_bucket=int(kernel_k),
            available_bytes=int(available_bytes),
            driver_free_bytes=int(driver_free_bytes),
            reclaimable_cache_bytes=int(reclaimable_cache_bytes),
        )
        # CN: planner 选择能完整常驻的一侧，另一侧流式上传。
        # EN: The planner selects one side that fully fits and streams the other side.
        resident_is_source = memory_plan.resident_side == "source"
        if resident_is_source:
            resident_feat = _to_device_2d(source_feat, device)
            resident_cost = _to_device_1d(source_cost, device)
            resident_dual = _to_device_1d(source_dual, device)
            stream_feat_source = target_feat
            stream_cost_source = target_cost
            stream_dual_source = target_dual
        else:
            resident_feat = _to_device_2d(target_feat, device)
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
        )
        source_chunk = int(tile["source_chunk"])
        query_tile = int(tile["query_tile"])
        db_tile = int(tile["db_tile"])
        staging_rows = min(int(source_chunk), int(n_stream))
        # CN: 双缓冲允许 CPU 准备下一块时上一块仍在 GPU 计算；总 pinned 内存固定为 O(chunk*d)。
        # EN: Double buffering lets the CPU prepare the next chunk while the previous chunk runs on GPU; pinned memory stays O(chunk*d).
        feature_staging = [
            _empty_cpu_staging((staging_rows, d), dtype=torch.float32)
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
        # CN: K=16/32 的通用 column reduction 目前慢于三次 custom directional/certificate scan。
        #     在专用大 K reduction 完成前，使用仍满足单侧常驻契约的三次 custom traversal。
        # EN: The generic K=16/32 column reduction is currently slower than three custom
        #     directional/certificate scans. Until a specialized large-K reduction exists,
        #     use three custom traversals while preserving the one-resident-side contract.
        use_decomposed_large_k = int(kernel_k) >= 16
        certificate_indices = (
            torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device)
            if use_decomposed_large_k
            else None
        )

        # CN: 分块上传 stream 侧数据；小 K 使用一次 fused traversal，大 K 使用三次 custom traversal。
        # EN: Upload the streamed side chunk by chunk; use one fused traversal for small K and three custom traversals for large K.
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
            query_feat = query_feat_cpu.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            query_cost = query_cost_cpu.to(device=device, dtype=torch.float64, non_blocking=True).contiguous()
            query_dual = query_dual_cpu.to(device=device, dtype=torch.float64, non_blocking=True).contiguous()
            copy_complete = torch.cuda.Event()
            copy_complete.record(torch.cuda.current_stream(device))
            staging_events[staging_slot] = copy_complete
            query_bias = (query_dual - query_cost).contiguous()
            if use_decomposed_large_k:
                q_vals, q_idx = ext.fused_directional_topk_raw(
                    query_feat,
                    resident_feat,
                    query_bias,
                    resident_bias,
                    int(query_tile),
                    int(db_tile),
                    int(kernel_k),
                    float(dot_scale),
                )
                r_vals, r_idx = ext.fused_directional_topk_raw(
                    resident_feat,
                    query_feat,
                    resident_bias,
                    query_bias,
                    int(query_tile),
                    int(db_tile),
                    int(kernel_k),
                    float(dot_scale),
                )
                standalone_stats = ext.raw_lowrank_stats_only(
                    query_feat,
                    resident_feat,
                    query_cost,
                    resident_cost,
                    query_dual,
                    resident_dual,
                    int(query_tile),
                    int(db_tile),
                    float(dot_scale),
                )
                stats = standalone_stats[certificate_indices]
            else:
                # CN: K<=8 由一次 pairwise traversal 同时返回双向 top-k 和 certificate。
                # EN: For K<=8, one pairwise traversal returns bidirectional top-k and the certificate.
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
                    bool(collect_timing),
                    int(col_kernel_id),
                    float(dot_scale),
                )
            # CN: query 侧 top-k 立即筛出正 reduced-cost 候选边并转回 CPU 保存。
            # EN: Query-side top-k candidates are filtered for positive reduced cost and copied back to CPU immediately.
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
            "pairwise_passes": 3 if use_decomposed_large_k else 1,
            "scan_traversal": "decomposed_large_k_custom" if use_decomposed_large_k else "fused_one_pass_custom",
            "host_staging_mode": "bounded_double_buffer",
            "host_staging_rows": int(staging_rows),
            "host_staging_buffer_count": 2,
            "host_staging_pinned": bool(all(buffer.is_pinned() for buffer in feature_staging)),
            "host_staging_bytes": int(
                2 * int(staging_rows) * (int(d) * 4 + 2 * 8)
            ),
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
