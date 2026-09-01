"""CN: 公开实验使用的确定性 synthetic Brenier 数据。EN: Deterministic synthetic Brenier data for public experiments."""

from __future__ import annotations

import numpy as np


def make_brenier_problem(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 构造由凸势梯度映射连接的等质量点云。
    EN: Construct equal-mass point clouds linked by the gradient of a convex potential.
    """
    rng = np.random.default_rng(int(seed))
    source = rng.normal(size=(int(n), int(d))).astype(np.float32)
    target = source + np.float32(0.1) * np.tanh(source)
    target = target[rng.permutation(int(n))]
    return source, np.ascontiguousarray(target, dtype=np.float32)
