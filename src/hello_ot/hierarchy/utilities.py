from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _normalize_subproblem_masses(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    CN: 将子问题 source/target 质量归一化到各自总和为 1，并返回原始质量统计。
    EN: Normalize subproblem source/target masses to unit sums and return raw mass statistics.
    """
    src = np.asarray(source_mass, dtype=np.float64).copy()
    tgt = np.asarray(target_mass, dtype=np.float64).copy()
    src_sum_raw = float(src.sum())
    tgt_sum_raw = float(tgt.sum())
    if src_sum_raw <= 0.0 or tgt_sum_raw <= 0.0:
        raise ValueError("subproblem mass sum must be positive")
    src /= max(src_sum_raw, 1e-12)
    tgt /= max(tgt_sum_raw, 1e-12)
    src_sum_normalized = float(np.asarray(src, dtype=np.float64).sum())
    tgt_sum_normalized = float(np.asarray(tgt, dtype=np.float64).sum())
    mass_sum_gap_normalized = float(src_sum_normalized - tgt_sum_normalized)
    if mass_sum_gap_normalized < 0.0:
        idx = int(np.argmax(src))
        src[idx] = np.float64(src[idx]) - mass_sum_gap_normalized
    elif mass_sum_gap_normalized > 0.0:
        idx = int(np.argmax(tgt))
        tgt[idx] = np.float64(tgt[idx]) + mass_sum_gap_normalized
    mass_sum_gap_final = float(
        np.asarray(src, dtype=np.float64).sum() - np.asarray(tgt, dtype=np.float64).sum()
    )
    if mass_sum_gap_final < 0.0:
        idx = int(np.argmax(src))
        src[idx] = np.float64(src[idx]) - mass_sum_gap_final
    elif mass_sum_gap_final > 0.0:
        idx = int(np.argmax(tgt))
        tgt[idx] = np.float64(tgt[idx]) + mass_sum_gap_final
    if np.any(src < 0.0) or np.any(tgt < 0.0):
        raise ValueError("subproblem mass balancing produced a negative entry")
    mass_sum_gap_final = float(
        np.asarray(src, dtype=np.float64).sum() - np.asarray(tgt, dtype=np.float64).sum()
    )
    return src, tgt, {
        "source_mass_sum_raw": float(src_sum_raw),
        "target_mass_sum_raw": float(tgt_sum_raw),
        "mass_sum_gap_raw": float(abs(src_sum_raw - tgt_sum_raw)),
        "mass_sum_gap_normalized": float(abs(mass_sum_gap_normalized)),
        "mass_sum_gap_final": float(abs(mass_sum_gap_final)),
    }


def _derive_hierarchy_seeds(seed: int, *, num_children: int) -> Tuple[int, int, List[int]]:
    """
    CN: 从层级节点的随机种子派生 source split、target split 与 child seed。
    EN: Derive deterministic source-split, target-split, and child seeds from a hierarchy-node seed.
    """
    rng = np.random.default_rng(int(seed))
    src_seed = int(rng.integers(0, 2**31 - 1))
    tgt_seed = int(rng.integers(0, 2**31 - 1))
    child_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(int(num_children))]
    return src_seed, tgt_seed, child_seeds
