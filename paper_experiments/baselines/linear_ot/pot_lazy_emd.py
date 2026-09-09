from __future__ import annotations

import time

import numpy as np

from .problem import LinearOTProblem
from .result import LinearOTResult


def solve_pot_lazy_emd(
    problem: LinearOTProblem,
    *,
    max_iterations: int = 100_000_000,
    potentials_init: tuple[np.ndarray, np.ndarray] | None = None,
    method_name: str = "pot_lazy_emd",
) -> LinearOTResult:
    """
    CN: 从点云直接运行 POT lazy exact EMD，并返回稀疏 transport。
    EN: Run POT lazy exact EMD directly from point clouds and return a sparse transport.
    """
    try:
        import ot
    except Exception as exc:
        raise ImportError("POT >= 0.9.7 is required for pot_lazy_emd.") from exc

    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive.")
    initial_potentials = _prepare_potentials_init(problem, potentials_init)
    solve_t0 = time.perf_counter()
    result = ot.solve_sample(
        problem.source_points,
        problem.target_points,
        a=problem.source_mass,
        b=problem.target_mass,
        metric=_pot_metric(problem.cost_type),
        lazy=True,
        max_iter=int(max_iterations),
        potentials_init=initial_potentials,
    )
    plan = result.sparse_plan
    if plan is None:
        raise RuntimeError("POT lazy EMD did not return sparse_plan.")
    objective = float(result.value_linear if result.value_linear is not None else result.value)
    runtime_sec = float(time.perf_counter() - solve_t0)
    status = str(result.status or "")
    converged = status.strip().lower() in {"", "converged", "success", "optimal"}
    return LinearOTResult(
        method=str(method_name),
        solver_objective=objective,
        runtime_sec=runtime_sec,
        transport_kind="sparse",
        transport=plan,
        converged=bool(converged),
        status=status or ("converged" if converged else "unknown"),
        diagnostics={
            "pot_version": str(getattr(ot, "__version__", "")),
            "max_iterations": int(max_iterations),
            "solver_backend": "ot.solve_sample(lazy=True)",
            "potentials_init": initial_potentials is not None,
        },
    )


def _prepare_potentials_init(
    problem: LinearOTProblem,
    potentials_init: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    CN: 校验 warm-start dual，并按 POT lazy network simplex 的要求转为连续 float64。
    EN: Validate warm-start duals and convert them to contiguous float64 for POT's lazy network simplex.
    """
    if potentials_init is None:
        return None
    source_dual, target_dual = potentials_init
    source = np.ascontiguousarray(np.asarray(source_dual, dtype=np.float64).reshape(-1))
    target = np.ascontiguousarray(np.asarray(target_dual, dtype=np.float64).reshape(-1))
    if source.shape != (problem.shape[0],) or target.shape != (problem.shape[1],):
        raise ValueError(
            "potentials_init lengths must match the source and target point counts."
        )
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("potentials_init must contain only finite values.")
    return source, target


def _pot_metric(cost_type: str) -> str:
    mapping: dict[str, str] = {
        "l2^2": "sqeuclidean",
        "l2": "euclidean",
        "l1": "cityblock",
    }
    try:
        return mapping[str(cost_type)]
    except KeyError as exc:
        raise ValueError(f"POT lazy EMD does not support cost_type={cost_type!r}.") from exc
