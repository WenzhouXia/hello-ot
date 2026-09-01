from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SupportSparsificationResult:
    """
    CN: restricted-OT support 稀疏化的无副作用结果。
    EN: Side-effect-free result of restricted-OT support sparsification.
    """

    values: np.ndarray
    input_nnz: int
    output_nnz: int
    target_nnz: int
    cycle_updates: int
    tree_operations: int
    wall_time: float

    @property
    def positive_mask(self) -> np.ndarray:
        """
        CN: 返回稀疏化后正 support 的布尔掩码。
        EN: Return the Boolean mask of the positive support after sparsification.
        """
        return self.values > 0.0


def _load_extension() -> Any:
    """
    CN: 延迟加载正式 C++ 扩展，使未使用稀疏化的求解路径不受影响。
    EN: Lazily load the formal C++ extension so unrelated solver paths remain unaffected.
    """
    try:
        from hello_ot._native.support_sparsifier import _support_sparsifier_ext
    except ImportError as error:  # pragma: no cover - depends on the local extension build
        raise RuntimeError(
            "The restricted-OT support sparsifier extension is not built. "
            "Build the hello_ot extensions before using this routine."
        ) from error
    return _support_sparsifier_ext


def sparsify_transport_support(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    costs: np.ndarray,
    *,
    n_source: int,
    n_target: int,
    target_nnz: int,
    zero_tolerance: float = 1e-15,
) -> SupportSparsificationResult:
    """
    CN: 沿非增代价方向消去 support cycle，直至正 support 不超过目标大小。
    EN: Cancel support cycles in non-increasing-cost directions until the positive support reaches the target size.
    """
    n_source = int(n_source)
    n_target = int(n_target)
    target_nnz = int(target_nnz)
    zero_tolerance = float(zero_tolerance)
    if n_source <= 0 or n_target <= 0:
        raise ValueError("n_source and n_target must be positive")
    if target_nnz < n_source + n_target - 1:
        raise ValueError("target_nnz must be at least n_source + n_target - 1")
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative")

    rows_array = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1)
    cols_array = np.ascontiguousarray(cols, dtype=np.int64).reshape(-1)
    values_array = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    costs_array = np.ascontiguousarray(costs, dtype=np.float64).reshape(-1)
    edge_count = int(rows_array.size)
    if cols_array.size != edge_count or values_array.size != edge_count or costs_array.size != edge_count:
        raise ValueError("rows, cols, values, and costs must have equal lengths")
    if np.any(rows_array < 0) or np.any(rows_array >= n_source):
        raise ValueError("rows contain an out-of-range source index")
    if np.any(cols_array < 0) or np.any(cols_array >= n_target):
        raise ValueError("cols contain an out-of-range target index")
    if not np.all(np.isfinite(values_array)) or np.any(values_array < -zero_tolerance):
        raise ValueError("values must be finite and nonnegative up to zero_tolerance")
    if not np.all(np.isfinite(costs_array)):
        raise ValueError("costs must be finite")

    input_nnz = int(np.count_nonzero(values_array > zero_tolerance))
    if input_nnz <= target_nnz:
        output = values_array.copy()
        output[output <= zero_tolerance] = 0.0
        return SupportSparsificationResult(
            values=output,
            input_nnz=input_nnz,
            output_nnz=input_nnz,
            target_nnz=target_nnz,
            cycle_updates=0,
            tree_operations=0,
            wall_time=0.0,
        )

    extension = _load_extension()
    started = time.perf_counter()
    raw = extension.sparsify_transport_support(
        n_source,
        n_target,
        rows_array,
        cols_array,
        values_array,
        costs_array,
        target_nnz,
        zero_tolerance,
    )
    wall_time = float(time.perf_counter() - started)
    output = np.asarray(raw["values"], dtype=np.float64)
    return SupportSparsificationResult(
        values=output,
        input_nnz=input_nnz,
        output_nnz=int(raw["output_nnz"]),
        target_nnz=target_nnz,
        cycle_updates=int(raw["cycle_updates"]),
        tree_operations=int(raw["tree_operations"]),
        wall_time=wall_time,
    )
