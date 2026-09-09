from __future__ import annotations

import math
from typing import Callable, Protocol, Tuple, runtime_checkable

import numpy as np


SourceSampler = Callable[[np.random.Generator, int, int], np.ndarray]
_CUDA_SOURCE_CHUNK_ROWS = 8192


@runtime_checkable
class SourceDistribution(Protocol):
    """
    CN: 可扩展 semi-discrete source distribution 的最小采样协议。
    EN: Minimal sampling protocol for extensible semi-discrete source distributions.
    """

    def sample(self, rng: np.random.Generator, sample_count: int, dimension: int) -> np.ndarray:
        ...


def standard_normal(
    rng: np.random.Generator,
    sample_count: int,
    dimension: int,
) -> np.ndarray:
    """
    CN: 从 d 维单位高斯分布采样。
    EN: Sample from the d-dimensional standard Gaussian distribution.
    """
    return rng.standard_normal((int(sample_count), int(dimension)), dtype=np.float32)


def fill_standard_normal_lowrank(
    rng: np.random.Generator,
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 就地生成单位高斯样本及平方欧氏低秩表示。
    EN: Generate standard-Gaussian samples and their squared-Euclidean low-rank representation in place.
    """
    rng.standard_normal(out=features, dtype=np.float32)
    np.multiply(features, np.float32(math.sqrt(2.0)), out=features)
    cost = np.einsum("ij,ij->i", features, features, dtype=np.float32, optimize=False)
    cost *= np.float32(0.5)
    return features, np.ascontiguousarray(cost, dtype=np.float32)


class CudaStandardNormalWorkspace:
    """
    CN: 在 CUDA 上分块生成单位高斯低秩输入并复用 pinned host buffers。
    EN: Generate standard-Gaussian low-rank inputs on CUDA in chunks while reusing pinned host buffers.
    """

    def __init__(self, sample_count: int, dimension: int, *, device: str = "cuda") -> None:
        import torch

        self._torch = torch
        self.device = torch.device(str(device))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA source sampling requires an available CUDA device")
        chunk_rows = min(int(sample_count), _CUDA_SOURCE_CHUNK_ROWS)
        self._features_host = torch.empty(
            (int(sample_count), int(dimension)), dtype=torch.float32, pin_memory=True
        )
        self._cost_host = torch.empty(int(sample_count), dtype=torch.float32, pin_memory=True)
        self._features_device = torch.empty(
            (chunk_rows, int(dimension)), device=self.device, dtype=torch.float32
        )

    def fill(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        sample_count = int(self._features_host.shape[0])
        chunk_rows = int(self._features_device.shape[0])
        for start in range(0, sample_count, chunk_rows):
            stop = min(start + chunk_rows, sample_count)
            block = self._features_device[: stop - start]
            block.normal_(generator=generator)
            block.mul_(math.sqrt(2.0))
            cost = torch.sum(block * block, dim=1).mul_(0.5)
            self._features_host[start:stop].copy_(block, non_blocking=True)
            self._cost_host[start:stop].copy_(cost, non_blocking=True)
        torch.cuda.synchronize(self.device)
        return self._features_host.numpy(), self._cost_host.numpy()


def resolve_sampler(distribution: SourceDistribution | SourceSampler | None) -> SourceSampler:
    if distribution is None:
        return standard_normal
    if isinstance(distribution, SourceDistribution):
        return distribution.sample
    if callable(distribution):
        return distribution
    raise TypeError("source_distribution must provide sample(...) or be callable")


__all__ = [
    "CudaStandardNormalWorkspace",
    "SourceDistribution",
    "SourceSampler",
    "fill_standard_normal_lowrank",
    "resolve_sampler",
    "standard_normal",
]
