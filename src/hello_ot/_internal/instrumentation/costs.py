from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from ..trace import _ChromeTraceCollector
from ..core.solver import _cost_pairs_lowrank_batched


def _extract_level_shape_from_cache(level_cache: Dict[str, Any]) -> Tuple[int, int, int]:
    if "F" in level_cache and "G" in level_cache:
        src = np.asarray(level_cache["F"])
        tgt = np.asarray(level_cache["G"])
        return int(src.shape[0]), int(tgt.shape[0]), int(src.shape[1])
    if "S" in level_cache and "T" in level_cache:
        src = np.asarray(level_cache["S"])
        tgt = np.asarray(level_cache["T"])
        return int(src.shape[0]), int(tgt.shape[0]), int(src.shape[1])
    raise ValueError("Unsupported level_cache layout for nodewise_auto pricing")


def sqeuclidean_to_lowrank(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 将点云转换为平方欧氏距离的低秩表示项。
    EN: Convert a point cloud to squared-Euclidean lowrank cost terms.
    """
    scaled = np.array(points, dtype=np.float32, order="C", copy=True)
    cost_vec = np.einsum("ij,ij->i", scaled, scaled).astype(np.float32, copy=False)
    scaled *= np.float32(math.sqrt(2.0))
    return scaled, cost_vec


def sqeuclidean_pair_to_lowrank(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CN: 将 source/target 点云转换为 C_ij=||x_i-y_j||^2 的低秩输入。
    EN: Convert source/target point clouds to lowrank inputs for C_ij=||x_i-y_j||^2.
    """
    source_f, source_cost_vec = sqeuclidean_to_lowrank(source_points)
    target_g, target_cost_vec = sqeuclidean_to_lowrank(target_points)
    return source_f, target_g, source_cost_vec, target_cost_vec


_preprocess_sqeuclidean_to_lowrank = sqeuclidean_to_lowrank


def _compute_lowrank_cost_vec_from_pairs_chunked(
    level_cache: Dict[str, Any],
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    chunk_size: int = 65536,
    feature_chunk_size: int | None = None,
    progress_tag: str = "InitialSupport",
    verbose: bool = True,
    tracer: _ChromeTraceCollector | None = None,
    trace_args: Dict[str, Any] | None = None,
    support_order_cache: Dict[str, Any] | None = None,
) -> np.ndarray:
    total = int(rows.shape[0])
    if total == 0:
        return np.empty(0, dtype=np.float32)

    num_chunks = (total + chunk_size - 1) // chunk_size
    if verbose:
        print(
            f"[{progress_tag}] start standard lowrank batched c_vec build: "
            f"edges={total:,}, batch_size={chunk_size}, chunks={num_chunks}",
            flush=True,
        )
    out = _cost_pairs_lowrank_batched(
        level_cache,
        rows,
        cols,
        batch_size=chunk_size,
        d_chunk=feature_chunk_size,
        free_after=True,
        tracer=tracer,
        trace_args=trace_args,
        support_order_cache=support_order_cache,
    )
    if torch is not None and torch.is_tensor(out):
        out = out.to(dtype=torch.float32)
    else:
        out = np.asarray(out).astype(np.float32, copy=False)
    if verbose:
        print(f"[{progress_tag}] finish chunked c_vec build", flush=True)
    return out
