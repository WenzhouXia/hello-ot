from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch

from hello_ot.state import GPUWarmStartState, WarmStartState


TorchState = WarmStartState | GPUWarmStartState
KnownSide = Literal["source", "target"]


@dataclass(frozen=True)
class TorchCompletionCandidate:
    known_side: KnownSide
    values: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class TorchViolationScan:
    rows: torch.Tensor
    cols: torch.Tensor
    dual_feasibility: float
    diagnostics: dict[str, Any]


def resolve_torch_device(value: str | torch.device) -> torch.device:
    """
    CN: 解析 portable backend 的 auto/cpu/cuda 设备选择，并对显式 CUDA 请求 fail-fast。
    EN: Resolve auto/cpu/cuda selection for the portable backend and fail fast on unavailable explicit CUDA.
    """
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch_device='cuda' requires a CUDA-capable PyTorch runtime")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("torch_device must resolve to CPU or CUDA")
    return device


def _tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=dtype).contiguous()
    return torch.as_tensor(value, dtype=dtype, device=device).contiguous()


def _state_tensors(state: TorchState, *, device: torch.device) -> tuple[torch.Tensor, ...]:
    rows = _tensor(state.rows, dtype=torch.int32, device=device).view(-1)
    cols = _tensor(state.cols, dtype=torch.int32, device=device).view(-1)
    values = _tensor(state.x_prev, dtype=torch.float64, device=device).view(-1)
    if state.dual_uv is None:
        raise ValueError("dual assignment requires dual potentials")
    dual = _tensor(state.dual_uv, dtype=torch.float64, device=device).view(-1)
    return rows, cols, values, dual


def _cost_tile(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    score_family: str,
    cost_type: str,
    source_offset: Optional[torch.Tensor],
    target_offset: Optional[torch.Tensor],
    dot_scale: float,
) -> torch.Tensor:
    """
    CN: 在一个受控 tile 内计算 point-cloud cost；调用方负责 blockwise 遍历。
    EN: Compute point-cloud costs inside one bounded tile; callers own the blockwise traversal.
    """
    if str(score_family) == "inner_product":
        if source_offset is None or target_offset is None:
            raise ValueError("inner-product score family requires source/target offsets")
        return source_offset[:, None] + target_offset[None, :] - float(dot_scale) * (source @ target.T)
    difference = source[:, None, :] - target[None, :, :]
    if str(cost_type) == "l1":
        return torch.sum(torch.abs(difference), dim=2)
    if str(cost_type) == "linf":
        return torch.amax(torch.abs(difference), dim=2)
    if str(cost_type) == "l2":
        return torch.linalg.vector_norm(difference, dim=2)
    raise ValueError(f"unsupported Torch scan cost_type={cost_type!r}")


def _tile_shape(n_source: int, n_target: int, *, max_tile_bytes: int) -> tuple[int, int]:
    # CN: score tile 使用 FP64；norm cost 还会短暂持有 feature difference，保守限制为预算的四分之一。
    # EN: Score tiles use FP64; norm costs also hold a feature difference temporarily, so use a conservative quarter budget.
    tile_elements = max(1, int(max_tile_bytes) // 32)
    source_block = min(int(n_source), 256)
    target_block = min(int(n_target), max(1, tile_elements // max(1, source_block)))
    return max(1, source_block), max(1, target_block)


def _merge_topk(
    current_values: torch.Tensor,
    current_indices: torch.Tensor,
    new_values: torch.Tensor,
    new_indices: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat((current_values, new_values), dim=1)
    indices = torch.cat((current_indices, new_indices), dim=1)
    kept_values, order = torch.topk(values, k=int(k), dim=1, largest=True, sorted=True)
    return kept_values, torch.gather(indices, 1, order)


def _problem_tensors(
    *,
    source_points: Any,
    target_points: Any,
    source_offset: Any,
    target_offset: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    source = _tensor(source_points, dtype=torch.float64, device=device)
    target = _tensor(target_points, dtype=torch.float64, device=device)
    source_bias = None if source_offset is None else _tensor(source_offset, dtype=torch.float64, device=device).view(-1)
    target_bias = None if target_offset is None else _tensor(target_offset, dtype=torch.float64, device=device).view(-1)
    return source, target, source_bias, target_bias


def _merge_state_candidates(
    state: TorchState,
    *,
    candidate_rows: torch.Tensor,
    candidate_cols: torch.Tensor,
    device: torch.device,
) -> GPUWarmStartState:
    rows, cols, values, dual = _state_tensors(state, device=device)
    rows_all = torch.cat((rows.to(torch.int64), candidate_rows.to(torch.int64)))
    cols_all = torch.cat((cols.to(torch.int64), candidate_cols.to(torch.int64)))
    values_all = torch.cat(
        (values, torch.zeros(candidate_rows.numel(), dtype=torch.float64, device=device))
    )
    keys_all = rows_all * int(state.n_target) + cols_all
    keys, inverse = torch.unique(keys_all, sorted=True, return_inverse=True)
    merged_values = torch.zeros(keys.numel(), dtype=torch.float64, device=device)
    merged_values.index_add_(0, inverse, values_all)
    merged_rows = torch.div(keys, int(state.n_target), rounding_mode="floor").to(torch.int32)
    merged_cols = torch.remainder(keys, int(state.n_target)).to(torch.int32)
    return GPUWarmStartState(
        rows=merged_rows,
        cols=merged_cols,
        x_prev=merged_values,
        dual_uv=dual,
        n_source=int(state.n_source),
        n_target=int(state.n_target),
        device=str(device),
        keys=keys,
        northwest_positions=None,
    )


def directional_assignment(
    *,
    state: TorchState,
    source_points: Any,
    target_points: Any,
    source_offset: Any,
    target_offset: Any,
    score_family: str,
    cost_type: str,
    dot_scale: float,
    known_side: KnownSide,
    topk: int,
    device: str | torch.device,
    max_tile_bytes: int = 64 * 1024 * 1024,
) -> tuple[GPUWarmStartState, dict[str, Any], dict[str, Any], TorchCompletionCandidate]:
    """
    CN: 用共享 blockwise dual-score primitive 执行一个方向的 dual assignment。
    EN: Run one direction of dual assignment with the shared blockwise dual-score primitive.
    """
    started = time.perf_counter()
    dev = resolve_torch_device(device)
    source, target, source_bias, target_bias = _problem_tensors(
        source_points=source_points,
        target_points=target_points,
        source_offset=source_offset,
        target_offset=target_offset,
        device=dev,
    )
    _rows, _cols, _values, dual = _state_tensors(state, device=dev)
    n_source, n_target = int(source.shape[0]), int(target.shape[0])
    if dual.numel() != n_source + n_target:
        raise ValueError("dual length does not match the directional scan problem")
    source_block, target_block = _tile_shape(n_source, n_target, max_tile_bytes=int(max_tile_bytes))
    if str(known_side) == "target":
        query_count, database_count = n_source, n_target
        requested_k = min(int(topk), n_target)
        best_values = torch.full((n_source, requested_k), -torch.inf, dtype=torch.float64, device=dev)
        best_indices = torch.full((n_source, requested_k), -1, dtype=torch.int64, device=dev)
        for source_start in range(0, n_source, source_block):
            source_stop = min(n_source, source_start + source_block)
            local_values = best_values[source_start:source_stop]
            local_indices = best_indices[source_start:source_stop]
            for target_start in range(0, n_target, target_block):
                target_stop = min(n_target, target_start + target_block)
                cost = _cost_tile(
                    source[source_start:source_stop],
                    target[target_start:target_stop],
                    score_family=score_family,
                    cost_type=cost_type,
                    source_offset=None if source_bias is None else source_bias[source_start:source_stop],
                    target_offset=None if target_bias is None else target_bias[target_start:target_stop],
                    dot_scale=dot_scale,
                )
                score = dual[n_source + target_start : n_source + target_stop][None, :] - cost
                local_values, local_indices = _merge_topk(
                    local_values,
                    local_indices,
                    score,
                    torch.arange(target_start, target_stop, dtype=torch.int64, device=dev)[None, :].expand(source_stop - source_start, -1),
                    k=requested_k,
                )
            best_values[source_start:source_stop] = local_values
            best_indices[source_start:source_stop] = local_indices
        candidate_rows = torch.arange(n_source, device=dev, dtype=torch.int64)[:, None].expand(-1, requested_k).reshape(-1)
        candidate_cols = best_indices.reshape(-1)
    else:
        query_count, database_count = n_target, n_source
        requested_k = min(int(topk), n_source)
        best_values = torch.full((n_target, requested_k), -torch.inf, dtype=torch.float64, device=dev)
        best_indices = torch.full((n_target, requested_k), -1, dtype=torch.int64, device=dev)
        for target_start in range(0, n_target, target_block):
            target_stop = min(n_target, target_start + target_block)
            local_values = best_values[target_start:target_stop]
            local_indices = best_indices[target_start:target_stop]
            for source_start in range(0, n_source, source_block):
                source_stop = min(n_source, source_start + source_block)
                cost = _cost_tile(
                    source[source_start:source_stop],
                    target[target_start:target_stop],
                    score_family=score_family,
                    cost_type=cost_type,
                    source_offset=None if source_bias is None else source_bias[source_start:source_stop],
                    target_offset=None if target_bias is None else target_bias[target_start:target_stop],
                    dot_scale=dot_scale,
                )
                score = dual[source_start:source_stop][:, None] - cost
                local_values, local_indices = _merge_topk(
                    local_values,
                    local_indices,
                    score.T,
                    torch.arange(source_start, source_stop, dtype=torch.int64, device=dev)[None, :].expand(target_stop - target_start, -1),
                    k=requested_k,
                )
            best_values[target_start:target_stop] = local_values
            best_indices[target_start:target_stop] = local_indices
        candidate_rows = best_indices.reshape(-1)
        candidate_cols = torch.arange(n_target, device=dev, dtype=torch.int64)[:, None].expand(-1, requested_k).reshape(-1)
    augmented = _merge_state_candidates(
        state,
        candidate_rows=candidate_rows,
        candidate_cols=candidate_cols,
        device=dev,
    )
    candidate = TorchCompletionCandidate(
        known_side=known_side,
        values=best_values[:, 0].contiguous(),
        indices=best_indices[:, 0].contiguous(),
    )
    elapsed = float(time.perf_counter() - started)
    stats = {
        "cross_transfer_added_raw": int(candidate_rows.numel()),
        "cross_transfer_added_unique": int(augmented.rows.numel()),
    }
    profile = {
        "backend": "torch_blockwise",
        "device": str(dev),
        "known_side": str(known_side),
        "query_count": int(query_count),
        "database_count": int(database_count),
        "topk": int(requested_k),
        "source_block": int(source_block),
        "target_block": int(target_block),
        "max_tile_elements": int(source_block * target_block),
        "total_time": elapsed,
    }
    return augmented, stats, profile, candidate


def complete_from_candidate(
    *,
    state: TorchState,
    candidate: TorchCompletionCandidate,
    device: str | torch.device,
) -> tuple[GPUWarmStartState, dict[str, Any]]:
    """
    CN: 根据 directional scan 的 top-1 seed 完成缺失侧 c-transform 对偶势。
    EN: Complete the missing c-transform dual potential from a directional scan's top-1 seed.
    """
    started = time.perf_counter()
    dev = resolve_torch_device(device)
    rows, cols, values, dual = _state_tensors(state, device=dev)
    dual = dual.clone()
    if candidate.known_side == "target":
        dual[: int(state.n_source)] = -candidate.values
    else:
        dual[int(state.n_source) :] = -candidate.values
    completed = GPUWarmStartState(
        rows=rows,
        cols=cols,
        x_prev=values,
        dual_uv=dual,
        n_source=int(state.n_source),
        n_target=int(state.n_target),
        device=str(dev),
        keys=rows.to(torch.int64) * int(state.n_target) + cols.to(torch.int64),
        northwest_positions=getattr(state, "northwest_positions", None),
    )
    return completed, {
        "backend": "torch_blockwise",
        "device": str(dev),
        "known_side": str(candidate.known_side),
        "total_time": float(time.perf_counter() - started),
    }


def bidirectional_violation_scan(
    *,
    source_points: Any,
    target_points: Any,
    source_offset: Any,
    target_offset: Any,
    source_dual: Any,
    target_dual: Any,
    score_family: str,
    cost_type: str,
    dot_scale: float,
    topk: int,
    theta: float,
    device: str | torch.device,
    collect_linf_diagnostics: bool = True,
    max_tile_bytes: int = 64 * 1024 * 1024,
) -> TorchViolationScan:
    """
    CN: 一次 blockwise traversal 同时计算双向违例候选和 full dual-feasibility certificate。
    EN: Compute bidirectional violation candidates and a full dual-feasibility certificate in one blockwise traversal.
    """
    started = time.perf_counter()
    dev = resolve_torch_device(device)
    source, target, source_bias, target_bias = _problem_tensors(
        source_points=source_points,
        target_points=target_points,
        source_offset=source_offset,
        target_offset=target_offset,
        device=dev,
    )
    source_dual_t = _tensor(source_dual, dtype=torch.float64, device=dev).view(-1)
    target_dual_t = _tensor(target_dual, dtype=torch.float64, device=dev).view(-1)
    n_source, n_target = int(source.shape[0]), int(target.shape[0])
    row_k = min(int(topk), n_target)
    col_k = min(int(topk), n_source)
    row_values = torch.full((n_source, row_k), -torch.inf, dtype=torch.float64, device=dev)
    row_indices = torch.full((n_source, row_k), -1, dtype=torch.int64, device=dev)
    col_values = torch.full((n_target, col_k), -torch.inf, dtype=torch.float64, device=dev)
    col_indices = torch.full((n_target, col_k), -1, dtype=torch.int64, device=dev)
    numerator_sq = torch.zeros((), dtype=torch.float64, device=dev)
    denominator_sq = torch.zeros((), dtype=torch.float64, device=dev)
    positive_count = torch.zeros((), dtype=torch.int64, device=dev)
    max_violation = torch.zeros((), dtype=torch.float64, device=dev)
    cost_linf = torch.zeros((), dtype=torch.float64, device=dev)
    source_block, target_block = _tile_shape(n_source, n_target, max_tile_bytes=int(max_tile_bytes))
    for source_start in range(0, n_source, source_block):
        source_stop = min(n_source, source_start + source_block)
        local_row_values = row_values[source_start:source_stop]
        local_row_indices = row_indices[source_start:source_stop]
        for target_start in range(0, n_target, target_block):
            target_stop = min(n_target, target_start + target_block)
            cost = _cost_tile(
                source[source_start:source_stop],
                target[target_start:target_stop],
                score_family=score_family,
                cost_type=cost_type,
                source_offset=None if source_bias is None else source_bias[source_start:source_stop],
                target_offset=None if target_bias is None else target_bias[target_start:target_stop],
                dot_scale=dot_scale,
            )
            score = (
                source_dual_t[source_start:source_stop, None]
                + target_dual_t[None, target_start:target_stop]
                - cost
            )
            positive = torch.clamp_min(score, 0.0)
            numerator_sq.add_(torch.sum(positive * positive))
            denominator_sq.add_(torch.sum(cost * cost))
            positive_count.add_(torch.count_nonzero(score > 0.0))
            max_violation = torch.maximum(max_violation, torch.amax(positive))
            cost_linf = torch.maximum(cost_linf, torch.amax(torch.abs(cost)))
            local_row_values, local_row_indices = _merge_topk(
                local_row_values,
                local_row_indices,
                score,
                torch.arange(target_start, target_stop, dtype=torch.int64, device=dev)[None, :].expand(source_stop - source_start, -1),
                k=row_k,
            )
            updated_col_values, updated_col_indices = _merge_topk(
                col_values[target_start:target_stop],
                col_indices[target_start:target_stop],
                score.T,
                torch.arange(source_start, source_stop, dtype=torch.int64, device=dev)[None, :].expand(target_stop - target_start, -1),
                k=col_k,
            )
            col_values[target_start:target_stop] = updated_col_values
            col_indices[target_start:target_stop] = updated_col_indices
        row_values[source_start:source_stop] = local_row_values
        row_indices[source_start:source_stop] = local_row_indices

    row_mask = row_values > float(theta)
    col_mask = col_values > float(theta)
    candidate_rows = torch.cat(
        (
            torch.arange(n_source, dtype=torch.int64, device=dev)[:, None].expand(-1, row_k)[row_mask],
            col_indices[col_mask],
        )
    )
    candidate_cols = torch.cat(
        (
            row_indices[row_mask],
            torch.arange(n_target, dtype=torch.int64, device=dev)[:, None].expand(-1, col_k)[col_mask],
        )
    )
    numerator = math.sqrt(max(0.0, float(numerator_sq.item())))
    denominator = math.sqrt(max(0.0, float(denominator_sq.item())))
    relative = numerator / (1.0 + denominator)
    diagnostics = {
        "scan_backend": "torch_blockwise",
        "device": str(dev),
        "dual_feasibility": float(relative),
        "dual_feasibility_num_sq": float(numerator_sq.item()),
        "dual_feasibility_den_sq": float(denominator_sq.item()),
        "dual_feasibility_positive_count": int(positive_count.item()),
        "dual_feasibility_max_violation": float(max_violation.item()),
        "dual_feasibility_cost_linf": float(cost_linf.item()),
        "relative_linf_dual_feasibility": (
            float(max_violation.item()) / (1.0 + float(cost_linf.item()))
            if collect_linf_diagnostics
            else None
        ),
        "fused_scan_candidate_count": int(candidate_rows.numel()),
        "fused_scan_topk": int(topk),
        "source_block": int(source_block),
        "target_block": int(target_block),
        "max_tile_elements": int(source_block * target_block),
        "pairwise_passes": 1,
        "fused_scan_time": float(time.perf_counter() - started),
    }
    return TorchViolationScan(
        rows=candidate_rows.to(torch.int64),
        cols=candidate_cols.to(torch.int64),
        dual_feasibility=float(relative),
        diagnostics=diagnostics,
    )


__all__ = [
    "TorchCompletionCandidate",
    "TorchViolationScan",
    "bidirectional_violation_scan",
    "complete_from_candidate",
    "directional_assignment",
    "resolve_torch_device",
]
