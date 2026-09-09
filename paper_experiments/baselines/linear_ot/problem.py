from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearOTProblem:
    """
    CN: 以两侧点云和可选质量描述一个平衡线性 OT 问题。
    EN: Describe a balanced linear-OT problem by two point clouds and optional masses.
    """

    source_points: np.ndarray
    target_points: np.ndarray
    source_mass: np.ndarray | None = None
    target_mass: np.ndarray | None = None
    cost_type: str = "l2^2"

    def __post_init__(self) -> None:
        source = _point_matrix(self.source_points, name="source_points")
        target = _point_matrix(self.target_points, name="target_points")
        if source.shape[1] != target.shape[1]:
            raise ValueError("source_points and target_points must have the same feature dimension.")
        cost_type = str(self.cost_type).strip().lower()
        if cost_type not in {"l2^2", "l2", "l1"}:
            raise ValueError("cost_type must be one of: l2^2, l2, l1")
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)
        object.__setattr__(self, "source_mass", _mass(self.source_mass, source.shape[0], name="source_mass"))
        object.__setattr__(self, "target_mass", _mass(self.target_mass, target.shape[0], name="target_mass"))
        object.__setattr__(self, "cost_type", cost_type)

    @property
    def shape(self) -> tuple[int, int]:
        """CN: 返回 transport matrix shape。 EN: Return the transport-matrix shape."""
        return int(self.source_points.shape[0]), int(self.target_points.shape[0])

    @property
    def is_square_uniform(self) -> bool:
        """CN: 判断问题是否为方形均匀质量。 EN: Report whether the problem is square with uniform masses."""
        n_source, n_target = self.shape
        if n_source != n_target:
            return False
        return bool(
            np.allclose(self.source_mass, 1.0 / float(n_source))
            and np.allclose(self.target_mass, 1.0 / float(n_target))
        )


def _point_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32, order="C")
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _mass(value: np.ndarray | None, size: int, *, name: str) -> np.ndarray:
    if value is None:
        return np.full(int(size), 1.0 / float(size), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (int(size),):
        raise ValueError(f"{name} must have shape ({size},).")
    if np.any(array < 0.0) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and nonnegative.")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass.")
    return np.asarray(array / total, dtype=np.float64)
