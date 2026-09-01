from __future__ import annotations

import importlib
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore

from ..state import (
    GPUWarmStartState as OTWarmStartGPUState,
    WarmStartState as OTWarmStartState,
    normalize_cpu_warm_start as _warm_start_to_cpu_public,
    warm_start_from_gpu as _warm_start_from_gpu,
    warm_start_support_size as _warm_start_support_size,
    warm_start_to_gpu as _warm_start_to_gpu,
)
from hello_ot._internal.trace import _ChromeTraceCollector, _trace_device_synchronize


from ..kernels.scan_contract import (
    custom_topk_bucket,
    plan_resident_side,
    plan_stream_chunk_rows,
    reusable_cuda_memory_bytes,
)

logger = logging.getLogger(__name__)
_TOPK_NUMPY_EXACT_MAX_SCORE_PAIRS = 0
_DUAL_ASSIGNMENT_GPU_UNIQUE_MIN_KEYS = 1 << 18
_DUAL_ASSIGNMENT_GROUPED_TOPK_MAX_TASKS = 64
_DUAL_ASSIGNMENT_GROUPED_TOPK_TORCH_FALLBACK_MAX_SCORE_BYTES = 512 << 20
_DUAL_ASSIGNMENT_GROUPED_TOPK_BLOCK_M = 32
_DUAL_ASSIGNMENT_GROUPED_TOPK_BLOCK_N = 64
_DUAL_ASSIGNMENT_GROUPED_TOPK_BLOCK_D = 32
_DUAL_ASSIGNMENT_ENABLE_GROUPED_TRITON_TOPK = False
_DUAL_ASSIGNMENT_STREAM_TOPK_QUERY_CHUNK_ROWS = 8192
_FUSED_MIPS_TOP_K_VALUES = (1, 2, 4, 8, 16, 32)
_FUSED_MIPS_EXT_MODULE = "hello_ot._native.inner_product_scan.hierot_inner_product_scan_ext"

_fused_mips_ext: Any = None
_fused_mips_ext_error: Optional[str] = None


def _fused_mips_topk_ext() -> Any:
    """
    CN: 惰性加载 inner-product scan CUDA 扩展；失败时明确报错，不静默回退。
    EN: Lazily load the inner-product scan CUDA extension; fail explicitly without a silent fallback.
    """
    global _fused_mips_ext, _fused_mips_ext_error
    if _fused_mips_ext is not None:
        return _fused_mips_ext
    if _fused_mips_ext_error is not None:
        raise RuntimeError(_fused_mips_ext_error)
    try:
        module = importlib.import_module(_FUSED_MIPS_EXT_MODULE)
    except Exception as exc:  # pragma: no cover
        _fused_mips_ext_error = (
            f"Custom inner-product scan extension {_FUSED_MIPS_EXT_MODULE!r} is unavailable; "
            "reinstall hello-ot to build it. "
            f"Original error: {exc!r}"
        )
        raise RuntimeError(_fused_mips_ext_error) from exc
    if not hasattr(module, "initialization_directional_topk_raw"):  # pragma: no cover
        _fused_mips_ext_error = (
            "Installed custom inner-product scan extension is outdated and does not expose "
            "initialization_directional_topk_raw; reinstall hello-ot."
        )
        raise RuntimeError(_fused_mips_ext_error)
    _fused_mips_ext = module
    return module


def _fused_mips_topk_supported(k: int, dim: Optional[int] = None) -> bool:
    """
    CN: 判断给定 k 是否可由 cuBLAS tiled inner-product scan 执行。
    EN: Decide whether the requested k can use the tiled-cuBLAS inner-product scan.
    """
    del dim
    try:
        custom_topk_bucket(int(k), score_family="inner_product")
    except ValueError:
        return False
    return True

DualAssignmentState = Union[OTWarmStartState, OTWarmStartGPUState]


@dataclass
class DualCompletionCandidate:
    """
    CN: 由 blockwise augment 顺手产出的 c-transform top1 复用信息。
    EN: Reusable c-transform top1 information produced as a side output of blockwise augment.
    """

    known_side: Literal["source", "target"]
    best_index: Any
    best_score: Any
    n_source: int
    n_target: int
    backend: str
    score_recompute_time: float = 0.0


@dataclass
class DeferredDualAssignmentCandidates:
    """
    CN: directional scan 后暂存在 CPU、尚未并入 GPU support 的候选边。
    EN: Candidate edges staged on CPU after a directional scan and not yet merged into GPU support.
    """

    rows: np.ndarray
    cols: np.ndarray
    known_side: Literal["source", "target"]

    def __post_init__(self) -> None:
        self.rows = np.asarray(self.rows, dtype=np.int32).reshape(-1)
        self.cols = np.asarray(self.cols, dtype=np.int32).reshape(-1)
        if int(self.rows.size) != int(self.cols.size):
            raise ValueError("deferred candidate rows and cols must have equal lengths")


def _torch_cuda_available() -> bool:
    """
    CN: 判断当前环境是否可用 torch CUDA backend，供 dual-assignment 的 GPU fast path 决策使用。
    EN: Check whether a torch CUDA backend is available for dual-assignment GPU fast paths.
    """
    return bool(torch is not None and torch.cuda.is_available())


def _should_use_gpu_unique_merge(num_keys: int) -> bool:
    """
    CN: 仅在键数量足够大时启用 GPU unique merge，避免小问题上的额外拷贝开销。
    EN: Enable GPU unique merge only for sufficiently large key counts to avoid transfer overhead on small problems.
    """
    return _torch_cuda_available() and int(num_keys) >= int(_DUAL_ASSIGNMENT_GPU_UNIQUE_MIN_KEYS)


def _dual_assignment_uses_gpu_pipeline(
    pipeline: Literal["auto", "gpu", "cpu"] = "auto",
) -> bool:
    """
    CN: 判断当前 dual-assignment 运行模式是否应当走 GPU 内部状态与 GPU dataflow。
    EN: Decide whether dual-assignment should use GPU internal state and GPU dataflow under the requested mode.
    """
    mode = str(pipeline).strip().lower()
    if mode not in {"auto", "gpu", "cpu"}:
        raise ValueError("dual_assignment_pipeline must be one of: auto, gpu, cpu.")
    cuda_available = _torch_cuda_available()
    if mode == "gpu" and not cuda_available:
        raise RuntimeError("dual_assignment_pipeline='gpu' requires torch CUDA to be available.")
    return mode == "gpu" or (mode == "auto" and cuda_available)


def _as_cuda_1d_tensor(
    value: Any,
    *,
    dtype: Any,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """
    CN: 将输入规范化为指定 Torch device 上的 1D tensor，供 stitch/augment 复用。
    EN: Normalize an input into a 1D tensor on the requested Torch device for stitch/augment reuse.
    """
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type == "cuda" and not _torch_cuda_available():
        raise RuntimeError("CUDA tensor conversion requires torch CUDA to be available.")
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.device != dev:
            tensor = tensor.to(device=dev)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor.contiguous().view(-1)
    return torch.as_tensor(value, dtype=dtype, device=dev).contiguous().view(-1)


def _as_cuda_2d_tensor(
    value: Any,
    *,
    dtype: Any = None,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """
    CN: 将输入规范化为指定 Torch device 上的 2D tensor，避免重复的 host-device 桥接。
    EN: Normalize an input into a 2D tensor on the requested Torch device to avoid repeated transfers.
    """
    if dtype is None:
        dtype = torch.float32
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type == "cuda" and not _torch_cuda_available():
        raise RuntimeError("CUDA tensor conversion requires torch CUDA to be available.")
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.device != dev:
            tensor = tensor.to(device=dev)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor.contiguous()
    return torch.as_tensor(value, dtype=dtype, device=dev).contiguous()


def _make_query_feature_chunk_cuda(
    query_np: np.ndarray,
    q_start: int,
    q_stop: int,
    *,
    device: "torch.device",
) -> "torch.Tensor":
    """
    CN: 仅上传原始 feature chunk；bias 由 scan ABI 单独传递，禁止物化增广 feature。
    EN: Upload only the raw feature chunk; the scan ABI receives bias separately and never materializes augmented features.
    """
    return _as_cuda_2d_tensor(
        query_np[int(q_start) : int(q_stop)],
        dtype=torch.float32,
        device=device,
    )



def _state_to_gpu(
    state: OTWarmStartState,
    *,
    device: str = "cuda",
) -> OTWarmStartGPUState:
    return _warm_start_to_gpu(state, device=device)


def _state_from_gpu(
    state: OTWarmStartGPUState,
) -> OTWarmStartState:
    return _warm_start_from_gpu(state)


def _state_to_cpu_public(
    state: OTWarmStartState,
) -> OTWarmStartState:
    return _warm_start_to_cpu_public(state)


def _normalize_dual_assignment_state(
    state: DualAssignmentState,
    *,
    pipeline: Literal["auto", "gpu", "cpu"] = "auto",
) -> DualAssignmentState:
    """
    CN: 按 pipeline 模式把状态规范化到 CPU public state 或 GPU internal state。
    EN: Normalize the state into either the CPU public form or the GPU internal form according to the pipeline mode.
    """
    use_gpu = _dual_assignment_uses_gpu_pipeline(pipeline)
    if isinstance(state, OTWarmStartGPUState):
        return state if use_gpu else _state_from_gpu(state)
    if use_gpu:
        return _state_to_gpu(state)
    return _state_to_cpu_public(state)


def _state_support_size(state: DualAssignmentState) -> int:
    return _warm_start_support_size(state)


def _unique_merge_rows_cols_vals(
    rows_full: np.ndarray,
    cols_full: np.ndarray,
    vals_full: np.ndarray,
    *,
    n_target: int,
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_args: Optional[Dict[str, Any]] = None,
    trace_prefix: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    CN: 对 `(row, col, value)` 三元组做按 key 的 unique merge；大问题优先走 torch/CUDA fast path。
    EN: Perform key-based unique merge on `(row, col, value)` triples; large problems prefer the torch/CUDA fast path.
    """
    force_torch_backend = bool(torch is not None and torch.is_tensor(rows_full))
    if force_torch_backend:
        num_keys = int(rows_full.numel())
    else:
        num_keys = int(np.asarray(rows_full).size)
    if force_torch_backend or _should_use_gpu_unique_merge(int(num_keys)):
        with (tracer.span(f"{trace_prefix}.torch_unique_merge", "augment", args=trace_args) if tracer is not None else nullcontext()):
            _trace_device_synchronize(tracer)
            device = torch.device("cuda")
            if torch.is_tensor(rows_full):
                rows_t = rows_full.detach().to(device=device, dtype=torch.int64).contiguous().view(-1)
                cols_t = cols_full.detach().to(device=device, dtype=torch.int64).contiguous().view(-1)
                vals_t = vals_full.detach().to(device=device, dtype=torch.float64).contiguous().view(-1)
            else:
                rows_t = torch.as_tensor(np.asarray(rows_full, dtype=np.int64), dtype=torch.int64, device=device)
                cols_t = torch.as_tensor(np.asarray(cols_full, dtype=np.int64), dtype=torch.int64, device=device)
                vals_t = torch.as_tensor(np.asarray(vals_full, dtype=np.float64), dtype=torch.float64, device=device)
            keys_t = rows_t * int(n_target) + cols_t
            uniq_keys_t, inverse_t = torch.unique(keys_t, sorted=True, return_inverse=True)
            merged_vals_t = torch.zeros(int(uniq_keys_t.numel()), dtype=torch.float64, device=device)
            merged_vals_t.scatter_add_(0, inverse_t, vals_t)
            merged_rows = torch.div(uniq_keys_t, int(n_target), rounding_mode="floor").to(dtype=torch.int32)
            merged_cols = torch.remainder(uniq_keys_t, int(n_target)).to(dtype=torch.int32)
            _trace_device_synchronize(tracer)
            return (
                merged_rows.detach().cpu().numpy().astype(np.int32, copy=False),
                merged_cols.detach().cpu().numpy().astype(np.int32, copy=False),
                merged_vals_t.detach().cpu().numpy().astype(np.float64, copy=False),
                "torch_cuda",
            )
    else:
        with (tracer.span(f"{trace_prefix}.numpy_unique_merge", "augment", args=trace_args) if tracer is not None else nullcontext()):
            keys = np.asarray(rows_full, dtype=np.int64) * np.int64(n_target) + np.asarray(cols_full, dtype=np.int64)
            uniq_keys, inverse = np.unique(keys, return_inverse=True)
            merged_vals = np.zeros(uniq_keys.shape[0], dtype=np.float64)
            np.add.at(merged_vals, inverse, np.asarray(vals_full, dtype=np.float64))
            merged_rows = (uniq_keys // np.int64(n_target)).astype(np.int32, copy=False)
            merged_cols = (uniq_keys % np.int64(n_target)).astype(np.int32, copy=False)
            return merged_rows, merged_cols, merged_vals, "numpy"


def _stitch_hierarchy_states(
    states: Sequence[DualAssignmentState],
    src_partitions: Sequence[np.ndarray],
    tgt_partitions: Sequence[np.ndarray],
    block_weights: Sequence[float],
    n_source: int,
    n_target: int,
    use_primal_values: bool = True,
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_args: Optional[Dict[str, Any]] = None,
) -> DualAssignmentState:
    if not (len(states) == len(src_partitions) == len(tgt_partitions) == len(block_weights)):
        raise ValueError("states, partitions, and block_weights must have the same length")
    return_gpu_state = bool(states) and isinstance(states[0], OTWarmStartGPUState)
    if return_gpu_state:
        if any(not isinstance(state, OTWarmStartGPUState) for state in states):
            raise RuntimeError("GPU stitch requires every child state to be a OTWarmStartGPUState.")
        t_total = time.perf_counter()
        device = torch.device(str(states[0].rows.device))
        stitched_dual_t = torch.zeros(int(n_source + n_target), dtype=torch.float64, device=device)
        dual_counts_t = torch.zeros(int(n_source + n_target), dtype=torch.float64, device=device)
        row_parts_t: List[torch.Tensor] = []
        col_parts_t: List[torch.Tensor] = []
        val_parts_t: List[torch.Tensor] = []
        map_rows_cols_time = 0.0
        dual_accumulate_time = 0.0
        for sub_id, state in enumerate(states):
            block_trace_args = {**(trace_args or {}), "block_id": int(sub_id)}
            with (tracer.span("stitch.loop_block", "stitch", args=block_trace_args) if tracer is not None else nullcontext()):
                src_idx_t = _as_cuda_1d_tensor(src_partitions[sub_id], dtype=torch.int64, device=device)
                tgt_idx_t = _as_cuda_1d_tensor(tgt_partitions[sub_id], dtype=torch.int64, device=device)
                rows_local_t = state.rows.to(dtype=torch.int64)
                cols_local_t = state.cols.to(dtype=torch.int64)
                t_map = time.perf_counter()
                with (tracer.span("stitch.map_rows_cols", "stitch", args=block_trace_args) if tracer is not None else nullcontext()):
                    if bool(use_primal_values):
                        vals_local_t = state.x_prev.to(dtype=torch.float64) * float(block_weights[sub_id])
                    else:
                        vals_local_t = torch.zeros_like(state.x_prev, dtype=torch.float64)
                    row_parts_t.append(src_idx_t.index_select(0, rows_local_t))
                    col_parts_t.append(tgt_idx_t.index_select(0, cols_local_t))
                    val_parts_t.append(vals_local_t)
                map_rows_cols_time += float(time.perf_counter() - t_map)
                t_dual = time.perf_counter()
                with (tracer.span("stitch.dual_accumulate", "stitch", args=block_trace_args) if tracer is not None else nullcontext()):
                    if state.dual_uv is None:
                        raise ValueError("GPU stitch requires child_state.dual_uv to be present.")
                    dual_t = state.dual_uv.to(dtype=torch.float64)
                    stitched_dual_t.index_add_(0, src_idx_t, dual_t[: src_idx_t.numel()])
                    stitched_dual_t.index_add_(0, int(n_source) + tgt_idx_t, dual_t[src_idx_t.numel() :])
                    dual_counts_t.index_add_(0, src_idx_t, torch.ones_like(src_idx_t, dtype=torch.float64))
                    dual_counts_t.index_add_(0, int(n_source) + tgt_idx_t, torch.ones_like(tgt_idx_t, dtype=torch.float64))
                dual_accumulate_time += float(time.perf_counter() - t_dual)
        with (tracer.span("stitch.concat_parts", "stitch", args=trace_args) if tracer is not None else nullcontext()):
            rows_full_t = torch.cat(row_parts_t, dim=0) if row_parts_t else torch.empty(0, dtype=torch.int64, device=device)
            cols_full_t = torch.cat(col_parts_t, dim=0) if col_parts_t else torch.empty(0, dtype=torch.int64, device=device)
            vals_full_t = torch.cat(val_parts_t, dim=0) if val_parts_t else torch.empty(0, dtype=torch.float64, device=device)
        if int(rows_full_t.numel()) == 0:
            raise ValueError("stitched hierarchy warm start is empty")
        with (tracer.span("stitch.normalize_dual", "stitch", args=trace_args) if tracer is not None else nullcontext()):
            valid_dual_t = dual_counts_t > 0
            stitched_dual_t[valid_dual_t] = stitched_dual_t[valid_dual_t] / dual_counts_t[valid_dual_t]
        t_merge = time.perf_counter()
        with (tracer.span("stitch.unique_merge", "stitch", args=trace_args) if tracer is not None else nullcontext()):
            merged_rows_t, merged_cols_t, merged_vals_t, merge_backend = _unique_merge_rows_cols_vals(
                rows_full_t,
                cols_full_t,
                vals_full_t,
                n_target=int(n_target),
                tracer=tracer,
                trace_args=trace_args,
                trace_prefix="stitch.unique_merge",
            )
        unique_merge_time = float(time.perf_counter() - t_merge)
        t_state = time.perf_counter()
        with (tracer.span("stitch.state_build", "stitch", args=trace_args) if tracer is not None else nullcontext()):
            stitched_state = OTWarmStartGPUState(
                rows=torch.as_tensor(merged_rows_t, dtype=torch.int32, device=device).contiguous(),
                cols=torch.as_tensor(merged_cols_t, dtype=torch.int32, device=device).contiguous(),
                x_prev=torch.as_tensor(merged_vals_t, dtype=torch.float64, device=device).contiguous(),
                dual_uv=stitched_dual_t.contiguous(),
                n_source=int(n_source),
                n_target=int(n_target),
                device=str(device),
                keys=None,
            )
            stitched_state.keys = stitched_state.rows.to(dtype=torch.int64) * int(n_target) + stitched_state.cols.to(dtype=torch.int64)
        state_build_time = float(time.perf_counter() - t_state)
        stitch_profile = {
            "backend": "torch_cuda",
            "host_export_time": 0.0,
            "map_rows_cols_time": float(map_rows_cols_time),
            "dual_accumulate_time": float(dual_accumulate_time),
            "unique_merge_time": float(unique_merge_time),
            "unique_merge_backend": str(merge_backend),
            "state_build_time": float(state_build_time),
            "total_time": float(time.perf_counter() - t_total),
        }
        return stitched_state, stitch_profile
    t_total = time.perf_counter()
    stitched_dual = np.zeros(int(n_source + n_target), dtype=np.float64)
    dual_counts = np.zeros(int(n_source + n_target), dtype=np.int32)
    row_parts: List[np.ndarray] = []
    col_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    map_rows_cols_time = 0.0
    dual_accumulate_time = 0.0
    for sub_id, state in enumerate(states):
        block_trace_args = {**(trace_args or {}), "block_id": int(sub_id)}
        with (tracer.span("stitch.loop_block", "stitch", args=block_trace_args) if tracer is not None else nullcontext()):
            src_idx = np.asarray(src_partitions[sub_id], dtype=np.int64)
            tgt_idx = np.asarray(tgt_partitions[sub_id], dtype=np.int64)
            state_cpu = _state_from_gpu(state) if isinstance(state, OTWarmStartGPUState) else state
            rows_local = np.asarray(state_cpu.rows, dtype=np.int64)
            cols_local = np.asarray(state_cpu.cols, dtype=np.int64)
            t_map = time.perf_counter()
            with (tracer.span("stitch.map_rows_cols", "stitch", args=block_trace_args) if tracer is not None else nullcontext()):
                if bool(use_primal_values):
                    vals_local = np.asarray(state_cpu.x_prev, dtype=np.float64) * float(block_weights[sub_id])
                else:
                    vals_local = np.zeros_like(np.asarray(state_cpu.x_prev, dtype=np.float64))
                row_parts.append(src_idx[rows_local])
                col_parts.append(tgt_idx[cols_local])
                val_parts.append(vals_local)
            map_rows_cols_time += float(time.perf_counter() - t_map)
            t_dual = time.perf_counter()
            with (tracer.span("stitch.dual_accumulate", "stitch", args=block_trace_args) if tracer is not None else nullcontext()):
                dual = np.asarray(state_cpu.dual_uv, dtype=np.float64)
                stitched_dual[src_idx] += dual[: src_idx.size]
                stitched_dual[n_source + tgt_idx] += dual[src_idx.size :]
                dual_counts[src_idx] += 1
                dual_counts[n_source + tgt_idx] += 1
            dual_accumulate_time += float(time.perf_counter() - t_dual)

    with (tracer.span("stitch.concat_parts", "stitch", args=trace_args) if tracer is not None else nullcontext()):
        rows_full = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
        cols_full = np.concatenate(col_parts) if col_parts else np.empty(0, dtype=np.int64)
        vals_full = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.float64)
    if rows_full.size == 0:
        raise ValueError("stitched hierarchy warm start is empty")
    with (tracer.span("stitch.normalize_dual", "stitch", args=trace_args) if tracer is not None else nullcontext()):
        valid_dual = dual_counts > 0
        stitched_dual[valid_dual] /= dual_counts[valid_dual].astype(np.float64)

    t_merge = time.perf_counter()
    with (tracer.span("stitch.unique_merge", "stitch", args=trace_args) if tracer is not None else nullcontext()):
        merged_rows, merged_cols, merged_vals, merge_backend = _unique_merge_rows_cols_vals(
            rows_full,
            cols_full,
            vals_full,
            n_target=int(n_target),
            tracer=tracer,
            trace_args=trace_args,
            trace_prefix="stitch.unique_merge",
        )
    unique_merge_time = float(time.perf_counter() - t_merge)
    t_state = time.perf_counter()
    with (tracer.span("stitch.state_build", "stitch", args=trace_args) if tracer is not None else nullcontext()):
        stitched_state = OTWarmStartState(
            rows=np.asarray(merged_rows, dtype=np.int32),
            cols=np.asarray(merged_cols, dtype=np.int32),
            x_prev=np.asarray(merged_vals, dtype=np.float64),
            dual_uv=np.asarray(stitched_dual, dtype=np.float64),
            n_source=int(n_source),
            n_target=int(n_target),
        )
        if return_gpu_state:
            stitched_state = _state_to_gpu(stitched_state)
    state_build_time = float(time.perf_counter() - t_state)
    stitch_profile = {
        "backend": "numpy",
        "host_export_time": 0.0,
        "map_rows_cols_time": float(map_rows_cols_time),
        "dual_accumulate_time": float(dual_accumulate_time),
        "unique_merge_time": float(unique_merge_time),
        "unique_merge_backend": str(merge_backend),
        "state_build_time": float(state_build_time),
        "total_time": float(time.perf_counter() - t_total),
    }
    return stitched_state, stitch_profile



def _topk_ip_indices_streamed_cuda(
    query_rows: int,
    make_query_feature_chunk,
    database_feature: Any,
    database_bias: Any,
    top_k: int,
    *,
    dot_scale: float = 1.0,
    device: Optional[Any] = None,
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_args: Optional[Dict[str, Any]] = None,
    trace_prefix: str = "augment.topk.streamed",
) -> Tuple["torch.Tensor", "torch.Tensor", Dict[str, Any]]:
    """
    CN: 对 feature 与独立 bias 做 directional inner-product scan；custom 路径从不物化增广 feature。
    EN: Run a directional inner-product scan over features and separate bias; the custom path never materializes augmented features.
    """
    if not _torch_cuda_available():
        raise RuntimeError("streamed CUDA top-k requires torch CUDA to be available.")
    db_t = None
    db_np = None
    # CN: database 只常驻一种表示；优先复用 CUDA tensor，否则保留 CPU numpy 供流式上传。
    # EN: Keep one resident database representation; reuse CUDA tensors or retain CPU numpy for streamed upload.
    if torch.is_tensor(database_feature):
        tensor = database_feature.detach()
        if tensor.device.type == "cuda":
            db_t = tensor.contiguous()
            db_rows = int(db_t.shape[0])
            db_dim = int(db_t.shape[1]) if db_t.ndim == 2 else 0
            device_obj = db_t.device
            database_add_mode = "cuda_tensor"
        else:
            db_np = np.asarray(tensor.cpu().numpy(), dtype=np.float32, order="C")
            db_rows = int(db_np.shape[0])
            db_dim = int(db_np.shape[1]) if db_np.ndim == 2 else 0
            device_obj = torch.device("cuda" if device is None else device)
            database_add_mode = "cpu_tensor_numpy"
    else:
        db_np = np.asarray(database_feature, dtype=np.float32, order="C")
        db_rows = int(db_np.shape[0])
        db_dim = int(db_np.shape[1]) if db_np.ndim == 2 else 0
        device_obj = torch.device("cuda" if device is None else device)
        database_add_mode = "cpu_numpy"
    if db_dim <= 0 or db_rows <= 0:
        raise ValueError("database_feature must be a non-empty 2D array/tensor.")
    if db_t is None and (db_np is None or db_np.ndim != 2):
        raise ValueError("database_feature must be a non-empty 2D array/tensor.")
    if db_t is not None and db_t.ndim != 2:
        raise ValueError("database_feature must be a non-empty 2D array/tensor.")
    if torch.is_tensor(database_bias):
        db_bias_cpu = database_bias.detach().to(device="cpu", dtype=torch.float64).contiguous().view(-1)
        db_bias_np = db_bias_cpu.numpy()
    else:
        db_bias_np = np.asarray(database_bias, dtype=np.float64).reshape(-1)
    if int(db_bias_np.size) != int(db_rows):
        raise ValueError("database_bias length must match database_feature rows.")
    q_rows = int(query_rows)
    k = int(min(max(1, top_k), db_rows))
    kernel_k = custom_topk_bucket(k, score_family="inner_product")
    custom_ext = _fused_mips_topk_ext()
    fused_available = custom_ext is not None and _fused_mips_topk_supported(int(k), int(db_dim))
    if float(dot_scale) not in {1.0, 2.0}:
        raise ValueError(f"inner-product scan supports dot_scale in {{1, 2}}; received {dot_scale}.")
    if not fused_available:
        raise ValueError(f"inner-product scan supports topk <= 32; received topk={k}.")
    backend = "inner_product_scan_cuda_streamed"
    memory_plan = None
    custom_chunk_rows = int(_DUAL_ASSIGNMENT_STREAM_TOPK_QUERY_CHUNK_ROWS)
    if q_rows > 0:
        available_bytes, driver_free_bytes, reclaimable_cache_bytes = reusable_cuda_memory_bytes(device_obj)
        memory_plan = plan_resident_side(
            source_count=int(q_rows),
            target_count=int(db_rows),
            feature_dim=int(db_dim),
            topk_bucket=int(kernel_k),
            available_bytes=int(available_bytes),
            driver_free_bytes=int(driver_free_bytes),
            reclaimable_cache_bytes=int(reclaimable_cache_bytes),
            preferred_resident_side="target",
        )
        streamed_count = q_rows if memory_plan.streamed_side == "source" else db_rows
        custom_chunk_rows = plan_stream_chunk_rows(
            memory_plan,
            streamed_count=int(streamed_count),
            feature_dim=int(db_dim),
            max_rows=int(_DUAL_ASSIGNMENT_STREAM_TOPK_QUERY_CHUNK_ROWS),
        )
    profile = {
        "query_augment_time": 0.0,
        "torch_score_time": 0.0,
        "torch_topk_time": 0.0,
        "numpy_score_time": 0.0,
        "numpy_topk_time": 0.0,
        "fused_mips_time": 0.0,
        "backend": backend,
        "database_add_mode": str(database_add_mode),
        "database_rows": int(db_rows),
        "database_dim": int(db_dim),
        "topk": int(k),
        "topk_bucket": None if kernel_k is None else int(kernel_k),
        "resident_side": None if memory_plan is None else str(memory_plan.resident_side),
        "resident_bytes": None if memory_plan is None else int(memory_plan.resident_bytes),
        "total_time": 0.0,
        "query_chunk_rows": int(_DUAL_ASSIGNMENT_STREAM_TOPK_QUERY_CHUNK_ROWS),
    }
    t_total = time.perf_counter()
    if q_rows <= 0:
        empty_vals = torch.empty((0, k), dtype=torch.float64, device=device_obj)
        empty_idx = torch.empty((0, k), dtype=torch.int64, device=device_obj)
        profile["total_time"] = float(time.perf_counter() - t_total)
        return empty_vals, empty_idx, profile

    if profile["backend"] == "inner_product_scan_cuda_streamed":
        # CN: custom 路径只常驻一侧 feature，另一侧按 chunk 上传；扩展内部用 cuBLAS tile 扫描。
        # EN: The custom path keeps one feature side resident, streams the other, and scans with tiled cuBLAS.
        t_fused = time.perf_counter()
        with (tracer.span(f"{trace_prefix}.fused_mips", "augment", args=trace_args) if tracer is not None else nullcontext()):
            _trace_device_synchronize(tracer)
            chunk_rows = int(custom_chunk_rows)
            profile["query_chunk_rows"] = int(chunk_rows)
            if memory_plan.resident_side == "target":
                if db_t is None:
                    db_t = _as_cuda_2d_tensor(db_np, dtype=torch.float32, device=device_obj)
                db_bias = _as_cuda_1d_tensor(db_bias_np, dtype=torch.float64, device=device_obj)
                # CN: 直接写入最终输出，避免 chunk list 与 torch.cat 同时持有两份完整 top-k。
                # EN: Write directly into final outputs, avoiding full top-k duplication by chunk lists plus torch.cat.
                vals_all = torch.empty((q_rows, k), dtype=torch.float64, device=device_obj)
                idx_all = torch.empty((q_rows, k), dtype=torch.int32, device=device_obj)
                for q_start in range(0, q_rows, chunk_rows):
                    q_stop = min(q_rows, q_start + chunk_rows)
                    t_query_prepare = time.perf_counter()
                    q_feat_t = make_query_feature_chunk(q_start, q_stop).to(device=device_obj, dtype=torch.float32).contiguous()
                    profile["query_augment_time"] += float(time.perf_counter() - t_query_prepare)
                    query_bias = torch.zeros(int(q_stop - q_start), dtype=torch.float64, device=device_obj)
                    vals_t, idx_t = custom_ext.initialization_directional_topk_raw(
                        q_feat_t,
                        db_t,
                        query_bias,
                        db_bias,
                        min(2048, int(q_stop - q_start)),
                        min(8192, int(db_rows)),
                        int(kernel_k),
                        float(dot_scale),
                    )
                    vals_all[q_start:q_stop].copy_(vals_t[:, :k])
                    idx_all[q_start:q_stop].copy_(idx_t[:, :k])
                    del q_feat_t, query_bias, vals_t, idx_t
            else:
                # CN: query 侧常驻时，每个 database chunk 产生一组候选，再在设备上精确合并。
                # EN: With resident queries, each database chunk produces candidates that are merged exactly on device.
                if db_np is not None:
                    database_cpu = torch.as_tensor(db_np, dtype=torch.float32)
                else:
                    database_cpu = db_t.detach().to(device="cpu", dtype=torch.float32)
                    del db_t
                resident_query = make_query_feature_chunk(0, q_rows).to(device=device_obj, dtype=torch.float32).contiguous()
                resident_bias = torch.zeros(q_rows, dtype=torch.float64, device=device_obj)
                best_vals = torch.full((q_rows, k), -torch.inf, dtype=torch.float64, device=device_obj)
                best_idx = torch.full((q_rows, k), -1, dtype=torch.int64, device=device_obj)
                for d_start in range(0, db_rows, chunk_rows):
                    d_stop = min(db_rows, d_start + chunk_rows)
                    database_chunk = database_cpu[d_start:d_stop].to(device=device_obj, dtype=torch.float32).contiguous()
                    database_bias = _as_cuda_1d_tensor(
                        db_bias_np[d_start:d_stop],
                        dtype=torch.float64,
                        device=device_obj,
                    )
                    vals_t, idx_t = custom_ext.initialization_directional_topk_raw(
                        resident_query,
                        database_chunk,
                        resident_bias,
                        database_bias,
                        min(2048, int(q_rows)),
                        min(8192, int(d_stop - d_start)),
                        int(kernel_k),
                        float(dot_scale),
                    )
                    candidate_vals = torch.cat([best_vals, vals_t[:, :k]], dim=1)
                    candidate_idx = torch.cat([best_idx, idx_t[:, :k].to(torch.int64) + int(d_start)], dim=1)
                    best_vals, order = torch.topk(candidate_vals, k=k, dim=1, largest=True, sorted=True)
                    best_idx = torch.gather(candidate_idx, 1, order).contiguous()
                    del database_chunk, database_bias, vals_t, idx_t, candidate_vals, candidate_idx, order
                vals_all, idx_all = best_vals.contiguous(), best_idx.contiguous()
            _trace_device_synchronize(tracer)
        profile["fused_mips_time"] = float(time.perf_counter() - t_fused)
    else:
        raise AssertionError(f"unreachable scan backend: {profile['backend']}")
    profile["total_time"] = float(time.perf_counter() - t_total)
    return vals_all, idx_all, profile


def _complete_dual_by_ctransform_top1(
    state: DualAssignmentState,
    *,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    known_side: Literal["source", "target"],
    dual_assignment_pipeline: Literal["gpu"] = "gpu",
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_args: Optional[Dict[str, Any]] = None,
    trace_prefix: str = "dual_completion",
    dot_scale: float = 1.0,
) -> Tuple[OTWarmStartGPUState, Dict[str, Any]]:
    if str(dual_assignment_pipeline) != "gpu":
        raise ValueError("dual_assignment_pipeline must be 'gpu'.")
    t_total = time.perf_counter()
    with (tracer.span(str(trace_prefix), "ctransform", args=trace_args) if tracer is not None else nullcontext()):
        t_normalize = time.perf_counter()
        with (tracer.span(f"{trace_prefix}.normalize_state", "ctransform", args=trace_args) if tracer is not None else nullcontext()):
            normalized_state = _normalize_dual_assignment_state(state, pipeline=dual_assignment_pipeline)
            if not isinstance(normalized_state, OTWarmStartGPUState):
                raise RuntimeError("GPU-native c-transform completion requires a OTWarmStartGPUState.")
            side = str(known_side)
            if side not in {"source", "target"}:
                raise ValueError("known_side must be 'source' or 'target'")
            n_source = int(normalized_state.n_source)
            n_target = int(normalized_state.n_target)
            if normalized_state.dual_uv is None:
                raise ValueError("state.dual_uv must not be None")
            device = normalized_state.rows.device
            dual_t = normalized_state.dual_uv.detach()
            if dual_t.device != device:
                dual_t = dual_t.to(device=device)
            dual_t = dual_t.to(dtype=torch.float64).contiguous().view(-1)
            if int(dual_t.numel()) != n_source + n_target:
                raise ValueError("state.dual_uv must have length n_source + n_target")
        normalize_state_time = float(time.perf_counter() - t_normalize)

        t_materialize = time.perf_counter()
        with (tracer.span(f"{trace_prefix}.materialize_inputs", "ctransform", args=trace_args) if tracer is not None else nullcontext()):
            src_f = np.asarray(source_F, dtype=np.float32, order="C")
            tgt_g = np.asarray(target_G, dtype=np.float32, order="C")
            src_cost = np.asarray(source_cost_vec, dtype=np.float64)
            tgt_cost = np.asarray(target_cost_vec, dtype=np.float64)
            if src_f.shape[0] != n_source or src_cost.shape[0] != n_source:
                raise ValueError("source arrays do not match state.n_source")
            if tgt_g.shape[0] != n_target or tgt_cost.shape[0] != n_target:
                raise ValueError("target arrays do not match state.n_target")
        materialize_inputs_time = float(time.perf_counter() - t_materialize)

        profile: Dict[str, Any] = {
            "known_side": side,
            "backend": "unknown",
            "torch_score_time": 0.0,
            "torch_topk_time": 0.0,
            "ctransform_recompute_backend": "torch_cuda_streamed",
            "ctransform_torch_recompute_time": 0.0,
            "export_backend": "gpu_internal",
            "export_boundary_time": 0.0,
            "numpy_score_time": 0.0,
            "numpy_topk_time": 0.0,
            "normalize_state_time": normalize_state_time,
            "materialize_inputs_time": materialize_inputs_time,
            "selected_score_recompute_time": 0.0,
            "total_time": 0.0,
        }
        if side == "target":
            beta_t = dual_t[n_source:]
            beta_np = beta_t.detach().cpu().numpy().astype(np.float64, copy=False)
            db_bias_np = np.asarray(beta_np - tgt_cost, dtype=np.float64)
            vals_t, topk_idx, topk_profile = _topk_ip_indices_streamed_cuda(
                n_source,
                lambda q_start, q_stop: _make_query_feature_chunk_cuda(src_f, q_start, q_stop, device=device),
                tgt_g,
                db_bias_np,
                1,
                dot_scale=float(dot_scale),
                device=device,
                tracer=tracer,
                trace_args=trace_args,
                trace_prefix=f"{trace_prefix}.top1_streamed",
            )
            t_recompute = time.perf_counter()
            with (tracer.span(f"{trace_prefix}.build_dual_from_streamed_scores", "ctransform", args=trace_args) if tracer is not None else nullcontext()):
                src_cost_t = _as_cuda_1d_tensor(src_cost, dtype=torch.float64, device=device)
                alpha_t = src_cost_t - vals_t[:, 0].to(dtype=torch.float64)
                full_dual_t = torch.cat([alpha_t, beta_t], dim=0).contiguous()
            del db_bias_np, vals_t, topk_idx
        else:
            alpha_t = dual_t[:n_source]
            alpha_np = alpha_t.detach().cpu().numpy().astype(np.float64, copy=False)
            db_bias_np = np.asarray(alpha_np - src_cost, dtype=np.float64)
            vals_t, topk_idx, topk_profile = _topk_ip_indices_streamed_cuda(
                n_target,
                lambda q_start, q_stop: _make_query_feature_chunk_cuda(tgt_g, q_start, q_stop, device=device),
                src_f,
                db_bias_np,
                1,
                dot_scale=float(dot_scale),
                device=device,
                tracer=tracer,
                trace_args=trace_args,
                trace_prefix=f"{trace_prefix}.top1_streamed",
            )
            t_recompute = time.perf_counter()
            with (tracer.span(f"{trace_prefix}.build_dual_from_streamed_scores", "ctransform", args=trace_args) if tracer is not None else nullcontext()):
                tgt_cost_t = _as_cuda_1d_tensor(tgt_cost, dtype=torch.float64, device=device)
                beta_t = tgt_cost_t - vals_t[:, 0].to(dtype=torch.float64)
                full_dual_t = torch.cat([alpha_t, beta_t], dim=0).contiguous()
            del db_bias_np, vals_t, topk_idx

        profile["selected_score_recompute_time"] = float(time.perf_counter() - t_recompute)
        profile["ctransform_torch_recompute_time"] = float(profile["selected_score_recompute_time"])
        profile["backend"] = str(topk_profile.get("backend", "unknown"))
        profile["torch_score_time"] = float(topk_profile.get("torch_score_time", 0.0))
        profile["torch_topk_time"] = float(topk_profile.get("torch_topk_time", 0.0))
        profile["numpy_score_time"] = float(topk_profile.get("numpy_score_time", 0.0))
        profile["numpy_topk_time"] = float(topk_profile.get("numpy_topk_time", 0.0))
        profile["total_time"] = float(time.perf_counter() - t_total)
        t_export = time.perf_counter()
        with (tracer.span(f"{trace_prefix}.build_output_state", "ctransform", args=trace_args) if tracer is not None else nullcontext()):
            out_state = OTWarmStartGPUState(
                rows=normalized_state.rows,
                cols=normalized_state.cols,
                x_prev=normalized_state.x_prev,
                dual_uv=full_dual_t,
                n_source=n_source,
                n_target=n_target,
                device=str(device),
                keys=normalized_state.keys,
                northwest_positions=normalized_state.northwest_positions,
            )
        profile["export_boundary_time"] = float(time.perf_counter() - t_export)
        return out_state, profile


def _ensure_lowrank_final_dual_feasible(
    state: DualAssignmentState,
    *,
    source_F: np.ndarray,
    target_G: np.ndarray,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_prefix: str = "final_dual_feasibility_correction",
    dot_scale: float = 1.0,
) -> Tuple[OTWarmStartState, Dict[str, Any]]:
    """
    CN: 用一次逐点 c-transform 修正最终 lowrank dual，并同步返回 CPU warm-start state。
    EN: Correct the final low-rank dual with one pointwise c-transform and return a synchronized CPU warm-start state.
    """
    before = _normalize_dual_assignment_state(state, pipeline="cpu")
    if not isinstance(before, OTWarmStartState) or before.dual_uv is None:
        raise RuntimeError("final dual feasibility correction requires a CPU state with dual_uv.")
    n_source = int(before.n_source)
    n_target = int(before.n_target)
    known_side: Literal["source", "target"] = (
        "target" if n_source >= n_target else "source"
    )
    corrected_gpu, profile = _complete_dual_by_ctransform_top1(
        before,
        source_F=source_F,
        target_G=target_G,
        source_cost_vec=source_cost_vec,
        target_cost_vec=target_cost_vec,
        known_side=known_side,
        dual_assignment_pipeline="gpu",
        tracer=tracer,
        trace_prefix=str(trace_prefix),
        dot_scale=float(dot_scale),
    )
    corrected = _normalize_dual_assignment_state(corrected_gpu, pipeline="cpu")
    if not isinstance(corrected, OTWarmStartState) or corrected.dual_uv is None:
        raise RuntimeError("final c-transform correction did not return a CPU dual state.")

    source = np.asarray(source_mass, dtype=np.float64).reshape(-1)
    target = np.asarray(target_mass, dtype=np.float64).reshape(-1)
    source /= float(np.sum(source))
    target /= float(np.sum(target))
    dual_before = np.asarray(before.dual_uv, dtype=np.float64)
    dual_after = np.asarray(corrected.dual_uv, dtype=np.float64)
    objective_change = float(
        np.dot(source, dual_after[:n_source] - dual_before[:n_source])
        + np.dot(target, dual_after[n_source:] - dual_before[n_source:])
    )
    return corrected, {
        "final_dual_feasibility_corrected": True,
        "final_dual_correction_dual_objective_change": objective_change,
        "final_dual_correction_time": float(profile.get("total_time", 0.0)),
    }


def _complete_dual_from_candidate(
    state: DualAssignmentState,
    *,
    source_cost_vec: np.ndarray,
    target_cost_vec: np.ndarray,
    known_side: Literal["source", "target"],
    seed: DualCompletionCandidate,
    dual_assignment_pipeline: Literal["gpu"] = "gpu",
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_args: Optional[Dict[str, Any]] = None,
    trace_prefix: str = "dual_completion_candidate",
) -> Tuple[OTWarmStartGPUState, Dict[str, Any]]:
    """
    CN: 使用 augment 产出的 top1 score 完成 c-transform，避免重复 top1 检索。
    EN: Complete c-transform from augment-produced top1 scores and avoid a repeated top1 search.
    """
    if str(dual_assignment_pipeline) != "gpu":
        raise ValueError("dual_assignment_pipeline must be 'gpu'.")
    t_total = time.perf_counter()
    with (tracer.span(str(trace_prefix), "ctransform", args=trace_args) if tracer is not None else nullcontext()):
        normalized_state = _normalize_dual_assignment_state(state, pipeline=dual_assignment_pipeline)
        if not isinstance(normalized_state, OTWarmStartGPUState):
            raise RuntimeError("GPU-native seeded c-transform completion requires a OTWarmStartGPUState.")
        side = str(known_side)
        if side not in {"source", "target"}:
            raise ValueError("known_side must be 'source' or 'target'")
        if str(seed.known_side) != side:
            raise ValueError("dual completion seed known_side does not match the requested known_side.")
        n_source = int(normalized_state.n_source)
        n_target = int(normalized_state.n_target)
        if int(seed.n_source) != n_source or int(seed.n_target) != n_target:
            raise ValueError("dual completion seed dimensions do not match the state dimensions.")
        if normalized_state.dual_uv is None:
            raise ValueError("state.dual_uv must not be None")

        profile: Dict[str, Any] = {
            "known_side": side,
            "backend": str(seed.backend),
            "source": "assignment_candidate",
            "top1_reused": True,
            "candidate_score_recompute_time": float(seed.score_recompute_time),
            "materialize_inputs_time": 0.0,
            "dual_build_time": 0.0,
            "export_backend": "gpu_internal",
            "export_boundary_time": 0.0,
            "total_time": 0.0,
        }
        t_build = time.perf_counter()
        device = normalized_state.rows.device
        dual_t = normalized_state.dual_uv.detach()
        if dual_t.device != device:
            dual_t = dual_t.to(device=device)
        dual_t = dual_t.to(dtype=torch.float64).contiguous().view(-1)
        if int(dual_t.numel()) != n_source + n_target:
            raise ValueError("state.dual_uv must have length n_source + n_target")
        score_t = _as_cuda_1d_tensor(seed.best_score, dtype=torch.float64, device=device)
        if side == "target":
            if int(score_t.numel()) != n_source:
                raise ValueError("target-known completion seed must have one score per source row.")
            src_cost_t = _as_cuda_1d_tensor(source_cost_vec, dtype=torch.float64, device=device)
            alpha_t = src_cost_t - score_t
            beta_t = dual_t[n_source:]
            full_dual_t = torch.cat([alpha_t, beta_t], dim=0).contiguous()
        else:
            if int(score_t.numel()) != n_target:
                raise ValueError("source-known completion seed must have one score per target row.")
            tgt_cost_t = _as_cuda_1d_tensor(target_cost_vec, dtype=torch.float64, device=device)
            alpha_t = dual_t[:n_source]
            beta_t = tgt_cost_t - score_t
            full_dual_t = torch.cat([alpha_t, beta_t], dim=0).contiguous()
        profile["dual_build_time"] = float(time.perf_counter() - t_build)
        t_export = time.perf_counter()
        out_state = OTWarmStartGPUState(
            rows=normalized_state.rows,
            cols=normalized_state.cols,
            x_prev=normalized_state.x_prev,
            dual_uv=full_dual_t,
            n_source=n_source,
            n_target=n_target,
            device=str(device),
            keys=normalized_state.keys,
            northwest_positions=normalized_state.northwest_positions,
        )
        profile["export_boundary_time"] = float(time.perf_counter() - t_export)
        profile["total_time"] = float(time.perf_counter() - t_total)
        return out_state, profile


def _build_state_from_candidates(
    base_state: DualAssignmentState,
    *,
    add_rows: np.ndarray,
    add_cols: np.ndarray,
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_args: Optional[Dict[str, Any]] = None,
) -> Tuple[DualAssignmentState, Dict[str, Any]]:
    return_gpu_state = isinstance(base_state, OTWarmStartGPUState)
    if return_gpu_state:
        device = torch.device(str(base_state.rows.device))
        rows_t = base_state.rows.to(dtype=torch.int64)
        cols_t = base_state.cols.to(dtype=torch.int64)
        x_prev_t = base_state.x_prev.to(dtype=torch.float64)
        n_target = int(base_state.n_target)
        add_rows_t = _as_cuda_1d_tensor(add_rows, dtype=torch.int64, device=device)
        add_cols_t = _as_cuda_1d_tensor(add_cols, dtype=torch.int64, device=device)
        add_vals_t = torch.zeros(int(add_rows_t.numel()), dtype=torch.float64, device=device)
        all_rows_t = torch.cat([rows_t, add_rows_t], dim=0)
        all_cols_t = torch.cat([cols_t, add_cols_t], dim=0)
        all_vals_t = torch.cat([x_prev_t, add_vals_t], dim=0)
        t_merge = time.perf_counter()
        with (tracer.span("augment.merge_unique", "augment", args=trace_args) if tracer is not None else nullcontext()):
            merged_rows_t, merged_cols_t, merged_vals_t, merge_backend = _unique_merge_rows_cols_vals(
                all_rows_t,
                all_cols_t,
                all_vals_t,
                n_target=int(n_target),
                tracer=tracer,
                trace_args=trace_args,
                trace_prefix="augment.merge_unique",
            )
        merge_unique_time = float(time.perf_counter() - t_merge)
        t_state = time.perf_counter()
        with (tracer.span("augment.state_build", "augment", args=trace_args) if tracer is not None else nullcontext()):
            augmented = OTWarmStartGPUState(
                rows=torch.as_tensor(merged_rows_t, dtype=torch.int32, device=device).contiguous(),
                cols=torch.as_tensor(merged_cols_t, dtype=torch.int32, device=device).contiguous(),
                x_prev=torch.as_tensor(merged_vals_t, dtype=torch.float64, device=device).contiguous(),
                dual_uv=None if base_state.dual_uv is None else base_state.dual_uv.to(dtype=torch.float64).contiguous(),
                n_source=int(base_state.n_source),
                n_target=int(base_state.n_target),
                device=str(device),
                keys=None,
            )
            augmented.keys = augmented.rows.to(dtype=torch.int64) * int(base_state.n_target) + augmented.cols.to(dtype=torch.int64)
        state_build_time = float(time.perf_counter() - t_state)
        return augmented, {
            "merge_unique_time": float(merge_unique_time),
            "merge_unique_backend": str(merge_backend),
            "state_build_time": float(state_build_time),
            "added_raw": int(add_rows_t.numel()),
            "added_unique": int(max(0, _state_support_size(augmented) - int(rows_t.numel()))),
        }
    base_state_cpu = _state_from_gpu(base_state) if return_gpu_state else base_state
    rows = np.asarray(base_state_cpu.rows, dtype=np.int64)
    cols = np.asarray(base_state_cpu.cols, dtype=np.int64)
    x_prev = np.asarray(base_state_cpu.x_prev, dtype=np.float64)
    n_source = int(base_state_cpu.n_source)
    n_target = int(base_state_cpu.n_target)
    add_rows64 = np.asarray(add_rows, dtype=np.int64)
    add_cols64 = np.asarray(add_cols, dtype=np.int64)
    add_vals = np.zeros(add_rows64.shape[0], dtype=np.float64)
    all_rows = np.concatenate([rows, add_rows64])
    all_cols = np.concatenate([cols, add_cols64])
    all_vals = np.concatenate([x_prev, add_vals])
    t_merge = time.perf_counter()
    with (tracer.span("augment.merge_unique", "augment", args=trace_args) if tracer is not None else nullcontext()):
        merged_rows, merged_cols, merged_vals, merge_backend = _unique_merge_rows_cols_vals(
            all_rows,
            all_cols,
            all_vals,
            n_target=int(n_target),
            tracer=tracer,
            trace_args=trace_args,
            trace_prefix="augment.merge_unique",
        )
    merge_unique_time = float(time.perf_counter() - t_merge)
    t_state = time.perf_counter()
    with (tracer.span("augment.state_build", "augment", args=trace_args) if tracer is not None else nullcontext()):
        augmented: DualAssignmentState = OTWarmStartState(
            rows=np.asarray(merged_rows, dtype=np.int32),
            cols=np.asarray(merged_cols, dtype=np.int32),
            x_prev=np.asarray(merged_vals, dtype=np.float64),
            dual_uv=np.asarray(base_state_cpu.dual_uv, dtype=np.float64).copy() if base_state_cpu.dual_uv is not None else None,
            n_source=n_source,
            n_target=n_target,
        )
        if return_gpu_state:
            augmented = _state_to_gpu(augmented)
    state_build_time = float(time.perf_counter() - t_state)
    return augmented, {
        "merge_unique_time": float(merge_unique_time),
        "merge_unique_backend": str(merge_backend),
        "state_build_time": float(state_build_time),
        "added_raw": int(add_rows64.size),
        "added_unique": int(max(0, _state_support_size(augmented) - int(rows.size))),
    }


def _materialize_deferred_candidate_union(
    base_state: DualAssignmentState,
    candidates: Sequence[DeferredDualAssignmentCandidates],
    *,
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_args: Optional[Dict[str, Any]] = None,
) -> Tuple[OTWarmStartGPUState, Dict[str, Any]]:
    """
    CN: 两遍 directional scan 结束后，将 CPU 候选上传 GPU 并只做一次 unique/support 构造。
    EN: Upload CPU candidates after both directional scans and perform one GPU unique/support build.
    """
    normalized = _normalize_dual_assignment_state(base_state, pipeline="gpu")
    if not isinstance(normalized, OTWarmStartGPUState):
        raise RuntimeError("deferred candidate materialization requires a GPU warm-start state")
    if normalized.dual_uv is None:
        raise ValueError("deferred candidate materialization requires completed dual potentials")
    if not candidates:
        raise ValueError("at least one deferred candidate batch is required")
    device = normalized.rows.device
    n_target = int(normalized.n_target)
    candidate_rows = np.concatenate([batch.rows for batch in candidates]).astype(np.int32, copy=False)
    candidate_cols = np.concatenate([batch.cols for batch in candidates]).astype(np.int32, copy=False)
    forward_raw = int(candidates[0].rows.size)
    base_count = int(normalized.rows.numel())
    t_merge = time.perf_counter()
    with (
        tracer.span("augment.deferred_union_gpu_unique", "augment", args=trace_args)
        if tracer is not None
        else nullcontext()
    ):
        rows_add_t = torch.as_tensor(candidate_rows, dtype=torch.int32, device=device)
        cols_add_t = torch.as_tensor(candidate_cols, dtype=torch.int32, device=device)
        base_keys_t = normalized.keys
        if base_keys_t is None or int(base_keys_t.numel()) != base_count:
            base_keys_t = normalized.rows.to(torch.int64) * n_target + normalized.cols.to(torch.int64)
        else:
            base_keys_t = base_keys_t.to(device=device, dtype=torch.int64).contiguous()
        candidate_keys_t = rows_add_t.to(torch.int64) * n_target + cols_add_t.to(torch.int64)
        all_keys_t = torch.cat([base_keys_t, candidate_keys_t], dim=0)
        unique_keys_t, inverse_t = torch.unique(all_keys_t, sorted=True, return_inverse=True)
        merged_x_t = torch.zeros(int(unique_keys_t.numel()), dtype=torch.float64, device=device)
        if base_count > 0:
            merged_x_t.scatter_add_(
                0,
                inverse_t[:base_count],
                normalized.x_prev.to(dtype=torch.float64),
            )
        forward_seen_t = torch.zeros(int(unique_keys_t.numel()), dtype=torch.bool, device=device)
        forward_seen_t[inverse_t[: base_count + forward_raw]] = True
        support_after_forward = int(torch.count_nonzero(forward_seen_t).item())
        rows_t = torch.div(unique_keys_t, n_target, rounding_mode="floor").to(torch.int32)
        cols_t = torch.remainder(unique_keys_t, n_target).to(torch.int32)
    merge_time = float(time.perf_counter() - t_merge)
    state = OTWarmStartGPUState(
        rows=rows_t.contiguous(),
        cols=cols_t.contiguous(),
        x_prev=merged_x_t.contiguous(),
        dual_uv=normalized.dual_uv.to(dtype=torch.float64).contiguous(),
        n_source=int(normalized.n_source),
        n_target=n_target,
        device=str(device),
        keys=unique_keys_t.contiguous(),
        northwest_positions=None,
    )
    return state, {
        "merge_unique_time": float(merge_time),
        "merge_unique_backend": "torch_cuda_deferred_union",
        "state_build_time": 0.0,
        "added_raw": int(candidate_rows.size),
        "added_unique": int(max(0, state.rows.numel() - base_count)),
        "support_after_forward": int(support_after_forward),
        "support_after_reverse": int(state.rows.numel()),
        "candidate_staging": "cpu_then_single_gpu_unique",
    }



def _assign_state_from_known_dual_directional_scan(
    state: DualAssignmentState,
    *,
    source_F: np.ndarray,
    target_G: np.ndarray,
    known_cost_vec: np.ndarray,
    assignment_topk: int,
    known_side: Literal["source", "target"],
    include_scaffold: bool = False,
    dual_assignment_pipeline: Literal["gpu"] = "gpu",
    return_completion_candidate: bool = False,
    defer_state_build: bool = False,
    tracer: Optional[_ChromeTraceCollector] = None,
    trace_args: Optional[Dict[str, Any]] = None,
    dot_scale: float = 1.0,
) -> Any:
    """
    CN: 使用某一侧已知 dual 做全局 top-k augment；known_side 指出已知 dual 所在侧。
    EN: Run global top-k augmentation from the known dual side; known_side identifies the anchored side.
    """
    if known_side not in {"source", "target"}:
        raise ValueError("known_side must be either 'source' or 'target'.")
    query_is_source = str(known_side) == "target"
    query_points = source_F if query_is_source else target_G
    database_points = target_G if query_is_source else source_F
    t_total = time.perf_counter()
    normalized_state = _normalize_dual_assignment_state(state, pipeline=dual_assignment_pipeline)
    if not isinstance(normalized_state, OTWarmStartGPUState):
        raise RuntimeError(f"GPU-native global {known_side}-dual top-k augmentation requires a OTWarmStartGPUState.")
    device = torch.device(str(normalized_state.rows.device))
    if normalized_state.dual_uv is None:
        raise ValueError(f"GPU-native global {known_side}-dual top-k augmentation requires state.dual_uv.")
    t_prepare = time.perf_counter()
    with (tracer.span("augment.global_prepare_streamed_inputs", "augment", args=trace_args) if tracer is not None else nullcontext()):
        query_np = np.asarray(query_points, dtype=np.float32, order="C")
        database_np = np.asarray(database_points, dtype=np.float32, order="C")
        known_cost_np = np.asarray(known_cost_vec, dtype=np.float64)
        n_source = int(normalized_state.n_source)
        n_target = int(normalized_state.n_target)
        if query_is_source:
            known_dual_np = normalized_state.dual_uv[n_source:].detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            known_dual_np = normalized_state.dual_uv[:n_source].detach().cpu().numpy().astype(np.float64, copy=False)
        known_bias_np = np.asarray(known_dual_np - known_cost_np, dtype=np.float64)
    prepare_inputs_time = float(time.perf_counter() - t_prepare)

    vals_t, topk_idx_t, topk_profile = _topk_ip_indices_streamed_cuda(
        int(query_np.shape[0]),
        lambda q_start, q_stop: _make_query_feature_chunk_cuda(query_np, q_start, q_stop, device=device),
        database_np,
        known_bias_np,
        int(assignment_topk),
        dot_scale=float(dot_scale),
        device=device,
        tracer=tracer,
        trace_args={**(trace_args or {}), "query_side_streamed": True, "database_side_resident": True},
        trace_prefix="augment.directional_topk_streamed" if query_is_source else "augment.reverse_directional_topk_streamed",
    )
    if bool(defer_state_build) and not bool(include_scaffold):
        raise ValueError("defer_state_build requires include_scaffold=True")
    completion_candidate = None
    completion_score_t = vals_t[:, 0].to(dtype=torch.float64).contiguous() if bool(return_completion_candidate) else None
    del vals_t
    t_select = time.perf_counter()
    with (tracer.span("augment.select_targets", "augment", args=trace_args) if tracer is not None else nullcontext()):
        topk_width = int(topk_idx_t.shape[1])
        deferred_candidates = None
        if bool(defer_state_build):
            # CN: 只把紧凑 int32 indices 暂存到 CPU；不在 CPU 做排序或 unique。
            # EN: Stage only compact int32 indices on CPU; do not sort or unique on CPU.
            topk_idx_np = topk_idx_t.detach().cpu().numpy().astype(np.int32, copy=True)
            if query_is_source:
                add_rows = np.repeat(np.arange(n_source, dtype=np.int32), topk_width)
                add_cols = topk_idx_np.reshape(-1)
            else:
                add_rows = topk_idx_np.reshape(-1)
                add_cols = np.repeat(np.arange(n_target, dtype=np.int32), topk_width)
            deferred_candidates = DeferredDualAssignmentCandidates(
                rows=add_rows,
                cols=add_cols,
                known_side=known_side,
            )
            raw_added = int(deferred_candidates.rows.size)
            del topk_idx_np
        else:
            if query_is_source:
                add_rows_t = torch.repeat_interleave(
                    torch.arange(n_source, dtype=torch.int64, device=device),
                    topk_width,
                )
                add_cols_t = topk_idx_t.reshape(-1).to(dtype=torch.int64)
            else:
                add_rows_t = topk_idx_t.reshape(-1).to(dtype=torch.int64)
                add_cols_t = torch.repeat_interleave(
                    torch.arange(n_target, dtype=torch.int64, device=device),
                    topk_width,
                )
            raw_added = int(add_rows_t.numel())
    if bool(return_completion_candidate):
        completion_candidate = DualCompletionCandidate(
            known_side=known_side,
            best_index=topk_idx_t[:, 0].detach().cpu().numpy().astype(np.int32, copy=True)
            if bool(defer_state_build)
            else topk_idx_t[:, 0].to(dtype=torch.int64).contiguous(),
            best_score=completion_score_t,
            n_source=n_source,
            n_target=n_target,
            backend="torch_cuda_streamed",
            score_recompute_time=0.0,
        )
    del topk_idx_t
    select_targets_time = float(time.perf_counter() - t_select)
    base_state: DualAssignmentState = normalized_state
    if not bool(include_scaffold):
        base_state = OTWarmStartGPUState(
            rows=torch.empty(0, dtype=torch.int32, device=device),
            cols=torch.empty(0, dtype=torch.int32, device=device),
            x_prev=torch.empty(0, dtype=torch.float64, device=device),
            dual_uv=normalized_state.dual_uv.to(dtype=torch.float64).contiguous(),
            n_source=n_source,
            n_target=n_target,
            device=str(device),
            keys=torch.empty(0, dtype=torch.int64, device=device),
        )
    if bool(defer_state_build):
        augmented = base_state
        state_build_stats = {
            "merge_unique_time": 0.0,
            "merge_unique_backend": "deferred",
            "state_build_time": 0.0,
            "added_raw": int(raw_added),
            "added_unique": 0,
        }
    else:
        augmented, state_build_stats = _build_state_from_candidates(
            base_state,
            add_rows=add_rows_t,
            add_cols=add_cols_t,
            tracer=tracer,
            trace_args=trace_args,
        )
    total_time = float(time.perf_counter() - t_total)
    tracked_time = float(prepare_inputs_time) + float(topk_profile.get("total_time", 0.0)) + float(select_targets_time) + float(state_build_stats.get("merge_unique_time", 0.0)) + float(state_build_stats.get("state_build_time", 0.0))
    resident_scan_side = topk_profile.get("resident_side")
    query_side_streamed = resident_scan_side != "source"
    augment_profile = {
        "backend": "torch_cuda_streamed",
        "host_export_time": 0.0,
        "prepare_inputs_time": float(prepare_inputs_time),
        "topk_total_time": float(topk_profile.get("total_time", 0.0)),
        "topk_query_augment_time": float(topk_profile.get("query_augment_time", 0.0)),
        "topk_torch_score_time": float(topk_profile.get("torch_score_time", 0.0)),
        "topk_torch_topk_time": float(topk_profile.get("torch_topk_time", 0.0)),
        "topk_numpy_score_time": float(topk_profile.get("numpy_score_time", 0.0)),
        "topk_numpy_topk_time": float(topk_profile.get("numpy_topk_time", 0.0)),
        "select_targets_time": float(select_targets_time),
        "merge_unique_time": float(state_build_stats.get("merge_unique_time", 0.0)),
        "state_build_time": float(state_build_stats.get("state_build_time", 0.0)),
        "topk_backend": str(topk_profile.get("backend", "unknown")),
        "total_time": float(total_time),
        "untracked_time": max(float(total_time) - float(tracked_time), 0.0),
        "query_side_streamed": bool(query_side_streamed),
        "database_side_resident": bool(query_side_streamed),
        "completion_candidate_score_time": 0.0,
        "candidate_staging": "cpu_deferred" if bool(defer_state_build) else "gpu_immediate",
    }
    stats = {
        "cross_transfer_added_raw": int(raw_added),
        "cross_transfer_added_unique": int(state_build_stats.get("added_unique", 0)),
    }
    if bool(defer_state_build):
        return augmented, stats, augment_profile, completion_candidate, deferred_candidates
    return (augmented, stats, augment_profile, completion_candidate) if bool(return_completion_candidate) else (augmented, stats, augment_profile)
