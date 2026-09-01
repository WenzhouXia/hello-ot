from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class HierarchyNodeRange:
    """
    CN: HELLO chain hierarchy 中一个连续区间节点的内部坐标描述。
    EN: Internal coordinate description of one contiguous-range node in the HELLO chain hierarchy.
    """

    path: str
    depth: int
    depth_remaining: int
    source_start: int
    source_stop: int
    target_start: int
    target_stop: int

    @property
    def n_source(self) -> int:
        return int(self.source_stop - self.source_start)

    @property
    def n_target(self) -> int:
        return int(self.target_stop - self.target_start)


def contiguous_partitions(n_items: int, split_count: int) -> List[np.ndarray]:
    """
    CN: 把当前连续区间的局部坐标均分成 split_count 个连续块。
    EN: Split local coordinates of the current contiguous range into split_count contiguous blocks.
    """
    n = int(n_items)
    if n < int(split_count):
        raise ValueError("HELLO chain split requires at least split_count points on the selected axis.")
    return [
        np.asarray(chunk, dtype=np.int64)
        for chunk in np.array_split(np.arange(n, dtype=np.int64), int(split_count))
    ]


def first_child_range(
    node: HierarchyNodeRange,
    *,
    split_axis: str,
    split_count: int,
) -> Tuple[HierarchyNodeRange, np.ndarray]:
    """
    CN: 按 selected axis 切分并固定选择第 0 块作为 chain child。
    EN: Split the selected axis and choose block 0 as the chain child.
    """
    if str(split_axis) == "source":
        partitions = contiguous_partitions(node.n_source, int(split_count))
        chosen = np.asarray(partitions[0], dtype=np.int64)
        child = HierarchyNodeRange(
            path=f"{node.path}/0",
            depth=int(node.depth) + 1,
            depth_remaining=int(node.depth_remaining) - 1,
            source_start=int(node.source_start + int(chosen[0])),
            source_stop=int(node.source_start + int(chosen[-1]) + 1),
            target_start=int(node.target_start),
            target_stop=int(node.target_stop),
        )
        return child, chosen
    if str(split_axis) == "target":
        partitions = contiguous_partitions(node.n_target, int(split_count))
        chosen = np.asarray(partitions[0], dtype=np.int64)
        child = HierarchyNodeRange(
            path=f"{node.path}/0",
            depth=int(node.depth) + 1,
            depth_remaining=int(node.depth_remaining) - 1,
            source_start=int(node.source_start),
            source_stop=int(node.source_stop),
            target_start=int(node.target_start + int(chosen[0])),
            target_stop=int(node.target_start + int(chosen[-1]) + 1),
        )
        return child, chosen
    raise ValueError(f"unsupported HELLO chain split_axis={split_axis!r}")


def slice_node_arrays(
    *,
    node: HierarchyNodeRange,
    source_F_full: np.ndarray,
    target_G_full: np.ndarray,
    source_cost_vec_full: np.ndarray,
    target_cost_vec_full: np.ndarray,
    source_mass_raw: np.ndarray,
    target_mass_raw: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    CN: 取出当前连续节点的特征、cost 向量与质量。
    EN: Slice features, cost vectors, and masses for one contiguous node.
    """
    out = {
        "source_F": np.asarray(source_F_full[node.source_start : node.source_stop], dtype=np.float32, order="C"),
        "target_G": np.asarray(target_G_full[node.target_start : node.target_stop], dtype=np.float32, order="C"),
        "source_cost_vec": np.asarray(source_cost_vec_full[node.source_start : node.source_stop], dtype=np.float64, order="C"),
        "target_cost_vec": np.asarray(target_cost_vec_full[node.target_start : node.target_stop], dtype=np.float64, order="C"),
        "source_mass": np.asarray(source_mass_raw[node.source_start : node.source_stop], dtype=np.float64, order="C"),
        "target_mass": np.asarray(target_mass_raw[node.target_start : node.target_stop], dtype=np.float64, order="C"),
    }
    return out


def resolve_hierarchy_depth(
    hierarchy_depth: Union[int, str],
    *,
    n_source: int,
    n_target: int,
    split_count: int,
    coarsest_size_threshold: int,
) -> int:
    """
    CN: 解析 chain 深度；auto 逐层切分较大侧，直到进入 coarsest threshold。
    EN: Resolve chain depth; auto repeatedly splits the larger side until reaching the coarsest threshold.
    """
    token = str(hierarchy_depth).strip().lower()
    if token != "auto":
        depth = int(hierarchy_depth)
        if depth < 0:
            raise ValueError("hierarchy_depth must be >= 0")
        return depth
    source_size = int(n_source)
    target_size = int(n_target)
    depth = 0
    while max(source_size, target_size) > int(coarsest_size_threshold):
        if source_size >= target_size:
            source_size = (source_size + int(split_count) - 1) // int(split_count)
        else:
            target_size = (target_size + int(split_count) - 1) // int(split_count)
        depth += 1
    return int(depth)


__all__ = [
    "HierarchyNodeRange",
    "contiguous_partitions",
    "first_child_range",
    "resolve_hierarchy_depth",
    "slice_node_arrays",
]
