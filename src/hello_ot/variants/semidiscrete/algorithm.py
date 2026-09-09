from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from hello_ot.config import SolverOptions
from hello_ot.types import DualPotentials
from hello_ot.variants._balanced import algorithm_config, solve_bilinear_subproblem

from .distributions import (
    CudaStandardNormalWorkspace,
    SourceDistribution,
    SourceSampler,
    fill_standard_normal_lowrank,
    resolve_sampler,
    standard_normal,
)
from .types import SemiDiscreteResult


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _probability_mass(value: Any | None, *, size: int, name: str) -> np.ndarray:
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


def _center_dual(value: Any) -> np.ndarray:
    dual = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    dual -= np.mean(dual, dtype=np.float32)
    return np.ascontiguousarray(dual, dtype=np.float32)


def _sqeuclidean_factors(points: Any) -> tuple[np.ndarray, np.ndarray]:
    array = _matrix(points, name="source sample")
    factors = np.empty_like(array)
    np.multiply(array, np.float32(math.sqrt(2.0)), out=factors)
    offsets = np.einsum("ij,ij->i", array, array, dtype=np.float32, optimize=False)
    return factors, np.ascontiguousarray(offsets, dtype=np.float32)


def solve_semidiscrete(
    target_points: Any,
    *,
    target_mass: Any | None = None,
    num_repeats: int = 32,
    source_sample_count: int | None = None,
    source_distribution: SourceDistribution | SourceSampler | None = None,
    source_mass: Any | None = None,
    source_sampling_backend: str = "cpu",
    random_seed: int = 42,
    source_seeds: Sequence[int] | None = None,
    max_iterations: int = 100,
    options: SolverOptions | None = None,
    inner_options: SolverOptions | None = None,
    store_sample_duals: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> SemiDiscreteResult:
    """
    CN: 重复求解经验平衡 OT 并平均 target dual，以近似 semi-discrete OT 对偶势。
    EN: Approximate the semi-discrete OT target dual by averaging repeated empirical balanced-OT solves.

    CN: source_distribution 默认为维数匹配 target 的单位高斯分布。
    EN: source_distribution defaults to a standard Gaussian matching the target dimension.
    """
    target = _matrix(target_points, name="target_points")
    n_target, dimension = map(int, target.shape)
    repeats = int(num_repeats)
    if repeats < 1:
        raise ValueError("num_repeats must be >= 1")
    sample_count = n_target if source_sample_count is None else int(source_sample_count)
    if sample_count < 1:
        raise ValueError("source_sample_count must be >= 1")
    if not isinstance(random_seed, int):
        raise TypeError("random_seed must be an int")
    seeds = (
        tuple(int(random_seed) + index for index in range(repeats))
        if source_seeds is None
        else tuple(int(seed) for seed in source_seeds)
    )
    if len(seeds) != repeats:
        raise ValueError("source_seeds length must equal num_repeats")
    sampling_backend = str(source_sampling_backend).strip().lower()
    if sampling_backend not in {"cpu", "cuda"}:
        raise ValueError("source_sampling_backend must be 'cpu' or 'cuda'")
    if source_distribution is not None and sampling_backend == "cuda":
        raise ValueError("CUDA sampling currently supports only the default standard Gaussian")

    target_weights = _probability_mass(target_mass, size=n_target, name="target_mass")
    source_weights = _probability_mass(source_mass, size=sample_count, name="source_mass")
    target_factors, target_offsets = _sqeuclidean_factors(target)
    sampler = resolve_sampler(source_distribution)
    default_gaussian = source_distribution is None
    host_buffer = (
        np.empty((sample_count, dimension), dtype=np.float32)
        if default_gaussian and sampling_backend == "cpu"
        else None
    )
    cuda_workspace = (
        CudaStandardNormalWorkspace(sample_count, dimension)
        if default_gaussian and sampling_backend == "cuda"
        else None
    )
    resolved_options = options if options is not None else inner_options
    config = algorithm_config(
        resolved_options,
        max_iterations=int(max_iterations),
        random_seed=int(random_seed),
    )

    average: np.ndarray | None = None
    stored_duals: list[np.ndarray] | None = [] if store_sample_duals else None
    records: list[Mapping[str, Any]] = []

    for repeat_index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        if cuda_workspace is not None:
            source_factors, source_offsets = cuda_workspace.fill(int(seed))
        elif host_buffer is not None:
            source_factors, source_offsets = fill_standard_normal_lowrank(rng, host_buffer)
        else:
            sample = _matrix(sampler(rng, sample_count, dimension), name="source_distribution sample")
            if sample.shape != (sample_count, dimension):
                raise ValueError(
                    "source_distribution must return shape "
                    f"({sample_count}, {dimension}); got {sample.shape}"
                )
            source_factors, source_offsets = _sqeuclidean_factors(sample)

        started = time.perf_counter()
        subproblem = solve_bilinear_subproblem(
            source_points=source_factors,
            target_points=target_factors,
            source_offset=source_offsets,
            target_offset=target_offsets,
            source_mass=source_weights,
            target_mass=target_weights,
            config=config,
            inherited_dual=(
                None
                if average is None
                else DualPotentials(target=_center_dual(average))
            ),
            preparation="source_from_target" if average is not None else "preserve",
        )
        current = _center_dual(subproblem.hello_result.solution.target_dual)
        average = (
            current.copy()
            if average is None
            else _center_dual(average + (current - average) / float(repeat_index + 1))
        )
        if stored_duals is not None:
            stored_duals.append(current.copy())
        record = {
            "repeat_index": int(repeat_index),
            "source_seed": int(seed),
            "mode": "cold" if repeat_index == 0 else "target_dual_warm_start",
            "wall_time": float(time.perf_counter() - started),
            "objective": float(subproblem.objective),
            "peak_active_support_size": int(subproblem.diagnostics["peak_active_support_size"]),
        }
        records.append(record)
        if progress_callback is not None:
            progress_callback(
                {
                    "repeat_index": int(repeat_index),
                    "num_repeats": repeats,
                    "record": dict(record),
                    "target_dual": _center_dual(average),
                    "sample_dual": current.copy(),
                }
            )

    assert average is not None
    sample_duals = (
        None
        if stored_duals is None
        else np.ascontiguousarray(np.stack(stored_duals), dtype=np.float32)
    )
    return SemiDiscreteResult(
        target_dual=_center_dual(average),
        sample_duals=sample_duals,
        source_seeds=seeds,
        num_repeats=repeats,
        source_sample_count=sample_count,
        target_shape=(n_target, dimension),
        source_sampling_backend=sampling_backend,
        records=tuple(records),
        metadata={
            "source_distribution": "standard_normal" if default_gaussian else "custom",
            "cost": "l2^2",
            "backend": str(config.backend),
        },
    )


__all__ = ["solve_semidiscrete"]
