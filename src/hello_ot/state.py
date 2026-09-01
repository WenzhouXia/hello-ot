from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch


@dataclass
class WarmStartState:
    """
    CN: CPU 侧稀疏 OT 状态；主要用于公开结果与设备边界。
    EN: CPU sparse-OT state used primarily at result and device boundaries.
    """

    rows: np.ndarray
    cols: np.ndarray
    x_prev: np.ndarray
    dual_uv: Optional[np.ndarray]
    n_source: int
    n_target: int
    northwest_positions: Optional[np.ndarray] = None


@dataclass
class GPUWarmStartState:
    """
    CN: initialization 与 refinement 之间传递的 Torch-device 稀疏 OT 状态。
    EN: Torch-device sparse-OT state passed from initialization to refinement.
    """

    rows: torch.Tensor
    cols: torch.Tensor
    x_prev: torch.Tensor
    dual_uv: Optional[torch.Tensor]
    n_source: int
    n_target: int
    device: str = "cuda"
    keys: Optional[torch.Tensor] = None
    northwest_positions: Optional[torch.Tensor] = None


WarmStartLike = Union[WarmStartState, GPUWarmStartState]


def warm_start_to_gpu(state: WarmStartState, *, device: str = "cuda") -> GPUWarmStartState:
    """
    CN: 把 CPU warm-start state 显式转移到 CUDA。
    EN: Explicitly transfer a CPU warm-start state to CUDA.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("warm_start_to_gpu requires torch CUDA")
    dev = torch.device(str(device))
    rows = torch.as_tensor(np.asarray(state.rows, dtype=np.int32), dtype=torch.int32, device=dev).contiguous()
    cols = torch.as_tensor(np.asarray(state.cols, dtype=np.int32), dtype=torch.int32, device=dev).contiguous()
    x_prev = torch.as_tensor(np.asarray(state.x_prev, dtype=np.float64), dtype=torch.float64, device=dev).contiguous()
    dual = None if state.dual_uv is None else torch.as_tensor(
        np.asarray(state.dual_uv, dtype=np.float64), dtype=torch.float64, device=dev
    ).contiguous()
    northwest = None if state.northwest_positions is None else torch.as_tensor(
        np.asarray(state.northwest_positions, dtype=np.int64), dtype=torch.int64, device=dev
    ).contiguous()
    return GPUWarmStartState(
        rows=rows,
        cols=cols,
        x_prev=x_prev,
        dual_uv=dual,
        n_source=int(state.n_source),
        n_target=int(state.n_target),
        device=str(dev),
        keys=rows.to(torch.int64) * int(state.n_target) + cols.to(torch.int64),
        northwest_positions=northwest,
    )


def warm_start_from_gpu(state: GPUWarmStartState) -> WarmStartState:
    """
    CN: 把 GPU warm-start state 显式导出为 CPU 数组。
    EN: Explicitly export a GPU warm-start state to CPU arrays.
    """
    def cpu(value: torch.Tensor, dtype: Any) -> np.ndarray:
        return value.detach().cpu().numpy().astype(dtype, copy=False).copy()

    return WarmStartState(
        rows=cpu(state.rows, np.int32),
        cols=cpu(state.cols, np.int32),
        x_prev=cpu(state.x_prev, np.float64),
        dual_uv=None if state.dual_uv is None else cpu(state.dual_uv, np.float64),
        n_source=int(state.n_source),
        n_target=int(state.n_target),
        northwest_positions=None if state.northwest_positions is None else cpu(state.northwest_positions, np.int64),
    )


def normalize_cpu_warm_start(state: WarmStartState) -> WarmStartState:
    """
    CN: 将可能含 CUDA tensor 字段的输入规范化为 CPU arrays。
    EN: Normalize an input that may contain CUDA tensors to CPU arrays.
    """
    def array(value: Any, dtype: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().astype(dtype, copy=False).copy()
        return np.asarray(value, dtype=dtype).copy()

    return WarmStartState(
        rows=array(state.rows, np.int32),
        cols=array(state.cols, np.int32),
        x_prev=array(state.x_prev, np.float64),
        dual_uv=None if state.dual_uv is None else array(state.dual_uv, np.float64),
        n_source=int(state.n_source),
        n_target=int(state.n_target),
        northwest_positions=None if state.northwest_positions is None else array(state.northwest_positions, np.int64),
    )


def consume_gpu_warm_start(state: WarmStartLike, *, keep_dual: bool = False) -> bool:
    """
    CN: ownership transfer 后释放已消费 Torch-device state 的大张量。
    EN: Release large tensors from a consumed Torch-device state after ownership transfer.
    """
    if not isinstance(state, GPUWarmStartState):
        return False
    device = torch.device(str(state.device))
    state.rows = torch.empty(0, dtype=torch.int32, device=device)
    state.cols = torch.empty(0, dtype=torch.int32, device=device)
    state.x_prev = torch.empty(0, dtype=torch.float64, device=device)
    if not keep_dual:
        state.dual_uv = None
    state.keys = None
    state.northwest_positions = None
    return True


def warm_start_support_size(state: WarmStartLike) -> int:
    """
    CN: 在不做隐式 device transfer 的情况下读取 active-support 大小。
    EN: Read active-support size without an implicit device transfer.
    """
    if isinstance(state, GPUWarmStartState):
        return int(state.rows.numel())
    return int(np.asarray(state.rows).size)


__all__ = [
    "GPUWarmStartState",
    "WarmStartLike",
    "WarmStartState",
    "consume_gpu_warm_start",
    "normalize_cpu_warm_start",
    "warm_start_from_gpu",
    "warm_start_support_size",
    "warm_start_to_gpu",
]
