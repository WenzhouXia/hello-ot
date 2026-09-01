from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Tuple

import numba
import numpy as np

from hello_ot.hierarchy.utilities import _derive_hierarchy_seeds
from hello_ot.state import WarmStartState as OTWarmStartState


@numba.njit(cache=True)
def _permute_feature_rows_inplace(array: np.ndarray, permutation: np.ndarray) -> None:
    """
    CN: 用 cycle decomposition 原地实现 new[i] = old[permutation[i]]。
    EN: Apply new[i] = old[permutation[i]] in place through cycle decomposition.
    """
    row_count, feature_dim = array.shape
    visited = np.zeros(row_count, dtype=np.uint8)
    row_buffer = np.empty(feature_dim, dtype=array.dtype)
    for start in range(row_count):
        if visited[start] != 0:
            continue
        current = start
        for feature_index in range(feature_dim):
            row_buffer[feature_index] = array[current, feature_index]
        while True:
            visited[current] = 1
            next_row = permutation[current]
            if visited[next_row] != 0:
                for feature_index in range(feature_dim):
                    array[current, feature_index] = row_buffer[feature_index]
                break
            for feature_index in range(feature_dim):
                array[current, feature_index] = array[next_row, feature_index]
            current = next_row


def _require_inplace_feature_array(array: np.ndarray, *, name: str) -> np.ndarray:
    """
    CN: 校验 consume-input 模式不会通过隐式复制掩盖输入布局问题。
    EN: Validate that consume-input mode cannot hide input-layout problems through an implicit copy.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array when consume_input_features=True")
    if array.ndim != 2 or array.dtype != np.float32 or not array.flags.c_contiguous:
        raise ValueError(
            f"{name} must be a C-contiguous FP32 matrix when consume_input_features=True"
        )
    if not array.flags.writeable:
        raise ValueError(f"{name} must be writable when consume_input_features=True")
    return array


def _prewarm_inplace_feature_reorder() -> None:
    """
    CN: 在正式 solve 计时外编译并校验原地重排 kernel。
    EN: Compile and validate the in-place reorder kernel outside formal solve timing.
    """
    values = np.arange(35, dtype=np.float32).reshape(7, 5)
    original = values.copy()
    permutation = np.asarray([3, 0, 6, 2, 5, 1, 4], dtype=np.int64)
    inverse = np.argsort(permutation).astype(np.int64, copy=False)
    _permute_feature_rows_inplace(values, permutation)
    if not np.array_equal(values, original[permutation]):
        raise RuntimeError("in-place feature reorder failed its forward validation")
    _permute_feature_rows_inplace(values, inverse)
    if not np.array_equal(values, original):
        raise RuntimeError("in-place feature reorder failed to restore its input")


def _make_shuffle_once_permutation(
    *,
    n_source: int,
    n_target: int,
    node_seed: int,
    split_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 为 hello 生成全局 source/target 一次性随机重排。
    EN: Generate one global source/target random permutation for hello.
    """
    src_seed, tgt_seed, _ = _derive_hierarchy_seeds(int(node_seed), num_children=int(split_count))
    source_perm = np.asarray(np.random.default_rng(src_seed).permutation(int(n_source)), dtype=np.int64)
    target_perm = np.asarray(np.random.default_rng(tgt_seed).permutation(int(n_target)), dtype=np.int64)
    return source_perm, target_perm


def _apply_hierarchy_global_reorder(
    *,
    source_F_full: np.ndarray,
    target_G_full: np.ndarray,
    source_cost_vec_full: np.ndarray,
    target_cost_vec_full: np.ndarray,
    source_mass_raw: np.ndarray,
    target_mass_raw: np.ndarray,
    source_perm: np.ndarray,
    target_perm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CN: 按全局重排同时重排低秩特征、cost 向量和质量。
    EN: Apply global reordering to low-rank features, cost vectors, and masses.
    """
    source_perm = np.asarray(source_perm, dtype=np.int64)
    target_perm = np.asarray(target_perm, dtype=np.int64)
    source_F_reordered = np.asarray(source_F_full[source_perm], dtype=np.float32, order="C")
    target_G_reordered = np.asarray(target_G_full[target_perm], dtype=np.float32, order="C")
    source_cost_reordered = np.asarray(source_cost_vec_full[source_perm], dtype=np.float64, order="C")
    target_cost_reordered = np.asarray(target_cost_vec_full[target_perm], dtype=np.float64, order="C")
    source_mass_reordered = np.asarray(source_mass_raw[source_perm], dtype=np.float64, order="C")
    target_mass_reordered = np.asarray(target_mass_raw[target_perm], dtype=np.float64, order="C")
    return (
        source_F_reordered,
        target_G_reordered,
        source_cost_reordered,
        target_cost_reordered,
        source_mass_reordered,
        target_mass_reordered,
    )


@contextmanager
def _hierarchy_global_reorder(
    *,
    source_F_full: np.ndarray,
    target_G_full: np.ndarray,
    source_cost_vec_full: np.ndarray,
    target_cost_vec_full: np.ndarray,
    source_mass_raw: np.ndarray,
    target_mass_raw: np.ndarray,
    source_perm: np.ndarray,
    target_perm: np.ndarray,
    consume_input_features: bool,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    CN: 默认物化重排 feature；consume-input 模式原地重排并在退出时尽力恢复。
    EN: Materialize reordered features by default, or reorder and restore them in place in consume-input mode.
    """
    if not bool(consume_input_features):
        yield _apply_hierarchy_global_reorder(
            source_F_full=source_F_full,
            target_G_full=target_G_full,
            source_cost_vec_full=source_cost_vec_full,
            target_cost_vec_full=target_cost_vec_full,
            source_mass_raw=source_mass_raw,
            target_mass_raw=target_mass_raw,
            source_perm=source_perm,
            target_perm=target_perm,
        )
        return

    source = _require_inplace_feature_array(source_F_full, name="source_F_full")
    target = _require_inplace_feature_array(target_G_full, name="target_G_full")
    if np.may_share_memory(source, target):
        raise ValueError(
            "source_F_full and target_G_full must not share memory when "
            "consume_input_features=True"
        )
    source_perm = np.ascontiguousarray(source_perm, dtype=np.int64)
    target_perm = np.ascontiguousarray(target_perm, dtype=np.int64)
    source_inverse = np.argsort(source_perm).astype(np.int64, copy=False)
    target_inverse = np.argsort(target_perm).astype(np.int64, copy=False)
    source_mutated = False
    target_mutated = False
    try:
        _permute_feature_rows_inplace(source, source_perm)
        source_mutated = True
        _permute_feature_rows_inplace(target, target_perm)
        target_mutated = True
        yield (
            source,
            target,
            np.asarray(source_cost_vec_full[source_perm], dtype=np.float64, order="C"),
            np.asarray(target_cost_vec_full[target_perm], dtype=np.float64, order="C"),
            np.asarray(source_mass_raw[source_perm], dtype=np.float64, order="C"),
            np.asarray(target_mass_raw[target_perm], dtype=np.float64, order="C"),
        )
    finally:
        try:
            if target_mutated:
                _permute_feature_rows_inplace(target, target_inverse)
        finally:
            if source_mutated:
                _permute_feature_rows_inplace(source, source_inverse)


def _map_warm_start_state_to_original_order(
    state: OTWarmStartState,
    *,
    source_perm: np.ndarray,
    target_perm: np.ndarray,
) -> OTWarmStartState:
    """
    CN: 将重排坐标系中的 root warm-start state 映射回原始 source/target 顺序。
    EN: Map the root warm-start state from reordered coordinates back to the original source/target order.
    """
    rows_reordered = np.asarray(state.rows, dtype=np.int64)
    cols_reordered = np.asarray(state.cols, dtype=np.int64)
    rows_original = np.asarray(np.asarray(source_perm, dtype=np.int64)[rows_reordered], dtype=np.int32)
    cols_original = np.asarray(np.asarray(target_perm, dtype=np.int64)[cols_reordered], dtype=np.int32)
    order = np.argsort(
        rows_original.astype(np.int64) * np.int64(int(state.n_target)) + cols_original.astype(np.int64),
        kind="stable",
    )
    rows_sorted = np.asarray(rows_original[order], dtype=np.int32, order="C")
    cols_sorted = np.asarray(cols_original[order], dtype=np.int32, order="C")
    x_sorted = np.asarray(np.asarray(state.x_prev, dtype=np.float64)[order], dtype=np.float64, order="C")
    dual_copy = None
    if state.dual_uv is not None:
        dual_reordered = np.asarray(state.dual_uv, dtype=np.float64, order="C")
        n_source = int(state.n_source)
        n_target = int(state.n_target)
        if dual_reordered.shape[0] != n_source + n_target:
            raise ValueError("warm_start dual_uv length does not match state dimensions")
        dual_copy = np.empty_like(dual_reordered)
        dual_copy[np.asarray(source_perm, dtype=np.int64)] = dual_reordered[:n_source]
        dual_copy[n_source + np.asarray(target_perm, dtype=np.int64)] = dual_reordered[n_source:]
    return OTWarmStartState(
        rows=rows_sorted,
        cols=cols_sorted,
        x_prev=x_sorted,
        dual_uv=dual_copy,
        n_source=int(state.n_source),
        n_target=int(state.n_target),
    )
