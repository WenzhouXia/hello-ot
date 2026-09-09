"""
CN: HELLO 的公开 Python API。
EN: Public Python API for HELLO.
"""

from __future__ import annotations

from typing import Any

from .config import SolverOptions
from .types import PointCloudCostType, PrewarmStats, Problem, Result
from .variants import (
    GromovResult,
    SemiDiscreteResult,
    UnbalancedResult,
    solve_gromov,
    solve_semidiscrete,
    solve_unbalanced,
)


def prewarm(*args: Any, **kwargs: Any) -> PrewarmStats:
    """
    CN: 预热 HELLO 使用的 CUDA kernels。
    EN: Prewarm the CUDA kernels used by HELLO.
    """
    from .prewarm import prewarm as implementation

    return implementation(*args, **kwargs)


def solve(
    source_points: Problem | Any,
    target_points: Any | None = None,
    *,
    source_mass: Any | None = None,
    target_mass: Any | None = None,
    cost: PointCloudCostType | None = None,
    max_iterations: int = 100,
    random_seed: int = 42,
    options: SolverOptions | None = None,
) -> Result:
    """
    CN: 求解平衡点云最优传输；数组调用默认使用平方欧氏代价和均匀质量。
    EN: Solve balanced point-cloud OT; array calls default to squared Euclidean cost and uniform masses.

    CN: LP 与对偶可行性容差固定为论文设置 1e-6，不作为公开调参项。
    EN: LP and dual-feasibility tolerances are fixed to the paper setting 1e-6 and are not public knobs.
    """
    if isinstance(source_points, Problem):
        if target_points is not None or source_mass is not None or target_mass is not None or cost is not None:
            raise TypeError(
                "When the first argument is Problem, target_points, masses, and cost must not be repeated."
            )
        problem = source_points
    else:
        if target_points is None:
            raise TypeError("target_points is required when solving from arrays")
        problem = Problem(
            source_points=source_points,
            target_points=target_points,
            source_mass=source_mass,
            target_mass=target_mass,
            cost_type="l2^2" if cost is None else cost,
        )
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be >= 1")
    if not isinstance(random_seed, int):
        raise TypeError("random_seed must be an int")
    resolved_options = SolverOptions() if options is None else options
    if not isinstance(resolved_options, SolverOptions):
        raise TypeError("options must be SolverOptions or None")

    from .api import solve_problem

    return solve_problem(
        problem,
        resolved_options._to_internal_config(
            max_iterations=int(max_iterations),
            random_seed=int(random_seed),
        ),
    )


__all__ = [
    "GromovResult",
    "PrewarmStats",
    "Problem",
    "Result",
    "SemiDiscreteResult",
    "SolverOptions",
    "UnbalancedResult",
    "prewarm",
    "solve",
    "solve_gromov",
    "solve_semidiscrete",
    "solve_unbalanced",
]
