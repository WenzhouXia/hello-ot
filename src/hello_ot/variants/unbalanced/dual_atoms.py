from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DualAtomMixture:
    """
    CN: 保存用于 FCFW 的 balanced-OT dual atoms 凸组合。
    EN: Represents a convex combination of balanced-OT dual atoms for FCFW.
    """

    source_atoms: np.ndarray
    target_atoms: np.ndarray
    weights: np.ndarray

    @classmethod
    def zeros(cls, n_source: int, n_target: int) -> DualAtomMixture:
        """
        CN: 构造全零初始原子混合（冷启动）。
        EN: Construct a zero initial dual-atom mixture (for cold start).
        """
        src = np.zeros((int(n_source), 1), dtype=np.float64)
        tgt = np.zeros((int(n_target), 1), dtype=np.float64)
        weights = np.ones(1, dtype=np.float64)
        return cls(source_atoms=src, target_atoms=tgt, weights=weights)

    @classmethod
    def from_atom(
        cls,
        source_atom: np.ndarray,
        target_atom: np.ndarray,
    ) -> DualAtomMixture:
        """
        CN: 从单个初始 dual atom 构造原子混合。
        EN: Construct a dual-atom mixture from a single initial atom.
        """
        src = np.asarray(source_atom, dtype=np.float64).reshape(-1, 1)
        tgt = np.asarray(target_atom, dtype=np.float64).reshape(-1, 1)
        weights = np.ones(1, dtype=np.float64)
        return cls(source_atoms=src, target_atoms=tgt, weights=weights)

    def current(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        CN: 计算当前凸组合对应的 source/target dual potentials。
        EN: Compute current combined source and target dual potentials.
        """
        return self.source_atoms @ self.weights, self.target_atoms @ self.weights

    def append(self, source_atom: np.ndarray, target_atom: np.ndarray) -> int:
        """
        CN: 向原子池添加新的 dual atom，初始权重设为 0。
        EN: Append a new dual atom to the mixture pool with initial weight 0.
        """
        self.source_atoms = np.column_stack(
            (self.source_atoms, np.asarray(source_atom, dtype=np.float64).reshape(-1))
        )
        self.target_atoms = np.column_stack(
            (self.target_atoms, np.asarray(target_atom, dtype=np.float64).reshape(-1))
        )
        self.weights = np.append(self.weights, 0.0)
        return int(self.weights.size - 1)

    def normalize_weights(self) -> None:
        """
        CN: 投影与归一化原子权重，确保落在概率单纯形上。
        EN: Project and normalize atom weights onto the probability simplex.
        """
        if np.min(self.weights) < -1e-10:
            raise RuntimeError(
                f"Dual-atom mixture update produced an invalid negative weight: {np.min(self.weights):.3e}"
            )
        self.weights[np.abs(self.weights) < 1e-14] = 0.0
        np.maximum(self.weights, 0.0, out=self.weights)
        total = float(np.sum(self.weights))
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("Dual-atom mixture weights have non-positive total mass.")
        self.weights /= total

    @property
    def atom_count(self) -> int:
        """
        CN: 原子总数量。
        EN: Total number of atoms.
        """
        return int(self.weights.size)

    @property
    def nonzero_atom_count(self) -> int:
        """
        CN: 权重非零的有效原子数量。
        EN: Number of atoms with strictly positive weights.
        """
        return int(np.count_nonzero(self.weights > 1e-12))


def canonicalize_atom(
    source_potential: np.ndarray,
    target_potential: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 固定 mean(f)=0 的 gauge；该常数平移不改变 dual feasibility。
    EN: Fix the mean(f)=0 gauge; this translation leaves dual feasibility invariant.
    """
    source = np.asarray(source_potential, dtype=np.float64).reshape(-1)
    target = np.asarray(target_potential, dtype=np.float64).reshape(-1)
    translation = -float(np.mean(source))
    return np.ascontiguousarray(source + translation), np.ascontiguousarray(target - translation)


__all__ = ["DualAtomMixture", "canonicalize_atom"]
