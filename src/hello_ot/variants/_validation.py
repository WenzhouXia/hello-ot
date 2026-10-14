from __future__ import annotations

from typing import Any

import numpy as np


def matrix(value: Any, *, name: str) -> np.ndarray:
    """
    CN: 校验并返回非空、有限、C 连续的 FP32 二维数组。
    EN: Validate and return a non-empty, finite, C-contiguous FP32 matrix.
    """
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def probability_mass(value: Any | None, *, size: int, name: str) -> np.ndarray:
    """
    CN: 校验概率质量；显式输入必须非负且和为 1，不静默归一化。
    EN: Validate probability mass; explicit values must be nonnegative and sum to one without normalization.
    """
    if value is None:
        return np.full(int(size), 1.0 / float(size), dtype=np.float64)
    mass = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    if mass.size != int(size):
        raise ValueError(f"{name} must have length {size}")
    if np.any(mass < 0.0) or not np.all(np.isfinite(mass)):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isclose(float(mass.sum()), 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{name} must sum to 1; HELLO does not normalize explicit masses")
    return mass


def positive_mass(value: Any | None, *, size: int, name: str) -> np.ndarray:
    """
    CN: 校验非平衡 OT 的绝对质量；所有值必须为正且有限，不执行归一化。
    EN: Validate absolute mass for unbalanced OT; values must be positive and finite without normalization.
    """
    if value is None:
        return np.full(int(size), 1.0 / float(size), dtype=np.float64)
    mass = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
    if mass.size != int(size):
        raise ValueError(f"{name} must have length {size}")
    if np.any(mass <= 0.0) or not np.all(np.isfinite(mass)):
        raise ValueError(f"{name} must contain only positive and finite values")
    return mass


__all__ = ["matrix", "positive_mass", "probability_mass"]
