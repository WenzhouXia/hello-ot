from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional


ScoreFamily = Literal["inner_product", "norm_cost"]
ResidentSide = Literal["source", "target"]

_CUSTOM_TOPK_BUCKETS = (1, 2, 4, 8, 16, 32)
_MIN_MEMORY_RESERVE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ScanMemoryPlan:
    """
    CN: 单卡 resident-streamed scan 的显存计划。
    EN: Memory plan for a single-GPU resident-streamed scan.
    """

    resident_side: ResidentSide
    resident_bytes: int
    streamed_side: ResidentSide
    streamed_bytes: int
    available_bytes: int
    usable_bytes: int
    reserve_bytes: int
    driver_free_bytes: Optional[int] = None
    reclaimable_cache_bytes: Optional[int] = None


@dataclass(frozen=True)
class DualFeasibilityCertificate:
    """
    CN: 全边集 dual-feasibility check 返回的设备侧统计量。
    EN: Device-side statistics returned by a full-edge dual-feasibility check.
    """

    l2_numerator_sq: Any
    l2_denominator_sq: Any
    max_positive_violation: Any
    cost_linf: Any
    positive_count: Any


@dataclass(frozen=True)
class DualAssignmentScanResult:
    """
    CN: dual assignment scan 的设备侧结果；completion seed 可选。
    EN: Device-side result of a dual-assignment scan with an optional completion seed.
    """

    source_values: Any
    source_indices: Any
    target_values: Optional[Any]
    target_indices: Optional[Any]
    completion_values: Optional[Any]
    completion_indices: Optional[Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DualViolationScanResult:
    """
    CN: 双向 dual-violation detection 与 feasibility check 的复合结果。
    EN: Composite result of bidirectional dual-violation detection and feasibility checking.
    """

    source_values: Any
    source_indices: Any
    target_values: Any
    target_indices: Any
    certificate: DualFeasibilityCertificate
    diagnostics: dict[str, Any]


def custom_topk_bucket(topk: int, *, score_family: ScoreFamily) -> int:
    """
    CN: 将任意 1..32 的请求映射到已实例化的 CUDA top-k bucket。
    EN: Map any request in 1..32 to an instantiated CUDA top-k bucket.
    """
    requested = int(topk)
    if requested < 1:
        raise ValueError("topk must be >= 1")
    for bucket in _CUSTOM_TOPK_BUCKETS:
        if requested <= bucket:
            return int(bucket)
    raise ValueError(f"custom {score_family} scan supports topk <= 32; received topk={requested}.")


def estimate_resident_bytes(*, point_count: int, feature_dim: int, topk_bucket: int) -> int:
    """
    CN: 估算单侧常驻 feature、cost/dual/bias 和双向 top-k 状态的字节数。
    EN: Estimate bytes for one resident feature side, cost/dual/bias, and bidirectional top-k state.
    """
    count = int(point_count)
    dim = int(feature_dim)
    bucket = int(topk_bucket)
    if count < 1 or dim < 1:
        raise ValueError("point_count and feature_dim must be >= 1")
    if bucket not in _CUSTOM_TOPK_BUCKETS:
        raise ValueError(f"topk_bucket must be one of {_CUSTOM_TOPK_BUCKETS}")
    feature_and_vectors = count * (dim * 4 + 3 * 8)
    topk_state = count * bucket * (8 + 4)
    return int(feature_and_vectors + topk_state)


def plan_resident_side(
    *,
    source_count: int,
    target_count: int,
    feature_dim: int,
    topk_bucket: int,
    available_bytes: int,
    reserve_fraction: float = 0.05,
    minimum_reserve_bytes: int = _MIN_MEMORY_RESERVE_BYTES,
    driver_free_bytes: Optional[int] = None,
    reclaimable_cache_bytes: Optional[int] = None,
    preferred_side_on_tie: Optional[ResidentSide] = None,
    preferred_resident_side: Optional[ResidentSide] = None,
) -> ScanMemoryPlan:
    """
    CN: 选择能完整驻留的一侧；两侧都放不下时明确报单卡不支持。
    EN: Select a side that fully fits; fail explicitly when neither side fits on one GPU.
    """
    if preferred_side_on_tie not in {None, "source", "target"}:
        raise ValueError("preferred_side_on_tie must be one of: None, source, target")
    if preferred_resident_side not in {None, "source", "target"}:
        raise ValueError("preferred_resident_side must be one of: None, source, target")
    if not 0.0 <= float(reserve_fraction) < 1.0:
        raise ValueError("reserve_fraction must be in [0, 1)")
    available = int(available_bytes)
    if available < 1:
        raise ValueError("available_bytes must be >= 1")
    reserve = (
        0
        if float(reserve_fraction) == 0.0
        else max(int(minimum_reserve_bytes), int(available * float(reserve_fraction)))
    )
    usable = max(0, int(available - reserve))
    source_bytes = estimate_resident_bytes(
        point_count=int(source_count), feature_dim=int(feature_dim), topk_bucket=int(topk_bucket)
    )
    target_bytes = estimate_resident_bytes(
        point_count=int(target_count), feature_dim=int(feature_dim), topk_bucket=int(topk_bucket)
    )
    candidates = []
    if source_bytes <= usable:
        candidates.append(("source", source_bytes, int(source_count)))
    if target_bytes <= usable:
        candidates.append(("target", target_bytes, int(target_count)))
    if not candidates:
        required = min(source_bytes, target_bytes)
        raise RuntimeError(
            "scan problem exceeds single-GPU resident-side capacity: "
            f"source requires {source_bytes} bytes, target requires {target_bytes} bytes, "
            f"but only {usable} of {available} bytes are usable; at least {required} bytes are required."
        )

    # CN: 调用方可优先指定任一能放下的侧；其次处理同规模偏好，最后默认常驻较大侧。
    # EN: Callers may prefer any fitting side, then apply an equal-size tie preference; otherwise keep the larger side resident.
    available_preference = (
        preferred_resident_side
        if any(item[0] == preferred_resident_side for item in candidates)
        else None
    )
    tied_preference = None
    if available_preference is None and int(source_count) == int(target_count):
        tied_preference = (
            preferred_side_on_tie
            if any(item[0] == preferred_side_on_tie for item in candidates)
            else None
        )
    selected_preference = available_preference if available_preference is not None else tied_preference
    if selected_preference is None:
        resident_name, resident_bytes, _resident_count = max(
            candidates, key=lambda item: (item[2], -item[1])
        )
    else:
        resident_name, resident_bytes, _resident_count = next(
            item for item in candidates if item[0] == selected_preference
        )
    streamed_name: ResidentSide = "target" if resident_name == "source" else "source"
    streamed_bytes = target_bytes if streamed_name == "target" else source_bytes
    plan = ScanMemoryPlan(
        resident_side=resident_name,  # type: ignore[arg-type]
        resident_bytes=int(resident_bytes),
        streamed_side=streamed_name,
        streamed_bytes=int(streamed_bytes),
        available_bytes=int(available),
        usable_bytes=int(usable),
        reserve_bytes=int(reserve),
        driver_free_bytes=None if driver_free_bytes is None else int(driver_free_bytes),
        reclaimable_cache_bytes=(
            None if reclaimable_cache_bytes is None else int(reclaimable_cache_bytes)
        ),
    )
    # CN: 仅在整次 solve 显存统计开启时保留 planner 决策；正常路径不创建诊断记录。
    # EN: Retain planner decisions only during whole-solve memory profiling; the normal path creates no records.
    from hello_ot._internal.instrumentation.memory_accounting import current_solve_memory_tracker

    tracker = current_solve_memory_tracker()
    if tracker is not None:
        tracker.record_plan(asdict(plan))
    return plan


def reusable_cuda_memory_bytes(device: Any) -> tuple[int, int, int]:
    """
    CN: 返回 PyTorch 新 allocation 可复用的容量、driver free 与 allocator cache。
    EN: Return memory reusable by new PyTorch allocations, driver-free bytes, and reclaimable allocator cache.
    """
    import torch

    driver_free, _total = torch.cuda.mem_get_info(device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    reclaimable = max(0, reserved - allocated)
    return int(driver_free) + int(reclaimable), int(driver_free), int(reclaimable)


def plan_stream_chunk_rows(
    plan: ScanMemoryPlan,
    *,
    streamed_count: int,
    feature_dim: int,
    max_rows: int,
    vector_count: int = 4,
) -> int:
    """
    CN: 用 resident 之外的可用预算限制流式 feature chunk；至少一行放不下时明确报错。
    EN: Bound the streamed feature chunk by the budget left after residency and fail if even one row cannot fit.
    """
    count = int(streamed_count)
    dim = int(feature_dim)
    cap = int(max_rows)
    if count < 1 or dim < 1 or cap < 1:
        raise ValueError("streamed_count, feature_dim, and max_rows must be >= 1")
    per_row_bytes = int(dim * 4 + max(0, int(vector_count)) * 8)
    scratch_bytes = max(0, int(plan.usable_bytes) - int(plan.resident_bytes))
    rows = min(count, cap, int(scratch_bytes // max(1, per_row_bytes)))
    if rows < 1:
        raise RuntimeError(
            "scan problem exceeds single-GPU streamed-chunk capacity after placing the resident side: "
            f"resident={plan.resident_bytes} bytes, usable={plan.usable_bytes} bytes, "
            f"one streamed row requires {per_row_bytes} bytes."
        )
    return int(rows)


__all__ = [
    "DualAssignmentScanResult",
    "DualFeasibilityCertificate",
    "DualViolationScanResult",
    "ResidentSide",
    "ScanMemoryPlan",
    "ScoreFamily",
    "custom_topk_bucket",
    "estimate_resident_bytes",
    "plan_resident_side",
    "plan_stream_chunk_rows",
    "reusable_cuda_memory_bytes",
    "validate_backend",
]
