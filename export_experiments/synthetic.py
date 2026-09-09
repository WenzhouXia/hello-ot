"""CN: 公开实验使用的确定性 synthetic Brenier 数据。EN: Deterministic synthetic Brenier data for public export_experiments."""

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


def make_correlated_brenier_problem(
    n: int,
    d: int,
    seed: int,
    *,
    rho: float = 0.999,
    latent_dim: int = 4,
    linear_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    CN: 构造带已知最优 permutation 的强相关 Brenier benchmark。
    EN: Build a strongly correlated Brenier benchmark with a known optimal permutation.
    """
    n = int(n)
    d = int(d)
    latent_dim = min(int(latent_dim), d)
    if n < 1 or d < 1 or latent_dim < 1:
        raise ValueError("n, d, and latent_dim must be positive")
    if not 0.0 <= float(rho) < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if float(linear_scale) <= 0.0:
        raise ValueError("linear_scale must be positive")

    random_state = np.random.RandomState(int(seed))
    latent = random_state.standard_normal((n, latent_dim))
    group_indices = np.arange(d, dtype=np.int64) % latent_dim
    source = np.empty((n, d), dtype=np.float32)
    block_rows = max(1, min(n, 8_388_608 // d))
    latent_weight = np.sqrt(float(rho))
    noise_weight = np.sqrt(1.0 - float(rho))
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        noise = random_state.standard_normal((stop - start, d))
        block = latent_weight * latent[start:stop, group_indices] + noise_weight * noise
        source[start:stop] = block.astype(np.float32, copy=False)

    permutation = random_state.permutation(n)
    target = np.ascontiguousarray(
        np.float32(linear_scale) * source[permutation],
        dtype=np.float32,
    )
    ground_truth_cols = np.argsort(permutation).astype(np.int64, copy=False)
    difference = np.asarray(source, dtype=np.float64) - np.asarray(
        target[ground_truth_cols], dtype=np.float64
    )
    objective = float(np.mean(np.einsum("ij,ij->i", difference, difference), dtype=np.float64))
    return source, target, ground_truth_cols, objective
