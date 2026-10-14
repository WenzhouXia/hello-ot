from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time

import numba
import numpy as np
import torch


@dataclass
class Transfers:
    """
    CN: 记录初始化和两个显式 CPU 例外的数据搬运。
    EN: Account for initialization transfers and the two explicit CPU exceptions.
    """

    device: torch.device
    records: dict = field(default_factory=dict)

    def move(self, value, device, reason):
        tensor = value.detach() if torch.is_tensor(value) else torch.from_numpy(np.asarray(value))
        device = torch.device(device)
        if tensor.device == device:
            return tensor
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = tensor.to(device=device)
        torch.cuda.synchronize(self.device)
        record = self.records.setdefault(reason, {"calls": 0, "bytes": 0, "seconds": 0.0})
        record["calls"] += 1
        record["bytes"] += tensor.numel() * tensor.element_size()
        record["seconds"] += time.perf_counter() - started
        return result

    def host(self, value, reason):
        if reason not in {"leaf_pot", "support_sparsification"}:
            raise RuntimeError(f"GPU pipeline forbids CPU array export: {reason}")
        return self.move(value, "cpu", reason).numpy()


@numba.njit(cache=True)
def _swap_pairs(permutation):
    """
    CN: 将置换分解成两轮互不重叠的行交换，不复制 feature。
    EN: Decompose a permutation into two rounds of disjoint row swaps without copying features.
    """
    size = len(permutation)
    visited = np.zeros(size, dtype=np.uint8)
    cycle = np.empty(size, dtype=np.int64)
    pairs = np.empty((2, size, 2), dtype=np.int64)
    counts = np.zeros(2, dtype=np.int64)
    for start in range(size):
        if visited[start]:
            continue
        length, current = 0, start
        while not visited[current]:
            visited[current] = 1
            cycle[length] = current
            length += 1
            current = permutation[current]
        for phase in range(2):
            for i in range(length):
                j = (1 - phase - i) % length
                if i < j:
                    pairs[phase, counts[phase], 0] = cycle[i]
                    pairs[phase, counts[phase], 1] = cycle[j]
                    counts[phase] += 1
    return pairs[0, :counts[0]].copy(), pairs[1, :counts[1]].copy()


@contextmanager
def reordered(source, target, source_perm, target_perm, transfers, consume):
    """
    CN: 用有界 GPU 临时块原地重排，退出时逆序恢复已完成的交换。
    EN: Reorder in place with bounded GPU scratch and undo completed swaps in reverse on exit.
    """
    if not consume:
        source_index = transfers.move(source_perm, source.device, "initialization")
        target_index = transfers.move(target_perm, target.device, "initialization")
        yield source.index_select(0, source_index), target.index_select(0, target_index)
        return
    if source.untyped_storage().data_ptr() == target.untyped_storage().data_ptr():
        raise ValueError("consume_input_features requires source and target with disjoint storage")
    swaps = []
    block = max(1, (4 * 1024**2) // (source.shape[1] * source.element_size()))
    for value, permutation in ((source, source_perm), (target, target_perm)):
        for phase in _swap_pairs(permutation):
            pairs = transfers.move(np.ascontiguousarray(phase.T), value.device, "initialization")
            for start in range(0, pairs.shape[1], block):
                swaps.append((value, pairs[:, start:start + block]))
    completed = []
    # CN: 恢复时复用预留缓冲，不在异常路径上申请新的 feature 工作区。
    # EN: Reserve scratch for restoration instead of allocating feature workspace during exception handling.
    max_rows = max((pairs.shape[1] for _, pairs in swaps), default=0)
    scratch_left = torch.empty((max_rows, source.shape[1]), dtype=source.dtype, device=source.device)
    scratch_right = torch.empty_like(scratch_left)

    def exchange(value, pairs):
        left, right = pairs[0], pairs[1]
        a, b = scratch_left[:len(left)], scratch_right[:len(right)]
        torch.index_select(value, 0, left, out=a)
        torch.index_select(value, 0, right, out=b)
        value.index_copy_(0, left, b)
        value.index_copy_(0, right, a)

    with torch.no_grad():
        try:
            for value, pairs in swaps:
                exchange(value, pairs)
                completed.append((value, pairs))
            yield source, target
        finally:
            for value, pairs in reversed(completed):
                exchange(value, pairs)
            torch.cuda.synchronize(source.device)
