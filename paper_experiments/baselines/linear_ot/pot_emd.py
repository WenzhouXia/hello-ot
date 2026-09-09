from __future__ import annotations

import time

import numpy as np

from .pot_lazy_emd import _pot_metric, _prepare_potentials_init
from .problem import LinearOTProblem
from .result import LinearOTResult


def solve_pot_emd(
    problem: LinearOTProblem,
    *,
    max_iterations: int = 100_000_000,
    potentials_init: tuple[np.ndarray, np.ndarray] | None = None,
    method_name: str = "pot_emd",
) -> LinearOTResult:
    """
    CN: 从点云直接运行 POT dense exact EMD，并返回稀疏 transport。
    EN: Run POT dense exact EMD directly from point clouds and return a sparse transport.
    """
    try:
        import ot
    except Exception as exc:
        raise ImportError("POT >= 0.9.7 is required for pot_emd.") from exc

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
        lazy=False,
        max_iter=int(max_iterations),
        potentials_init=initial_potentials,
    )
    plan = result.sparse_plan
    if plan is None:
        raise RuntimeError("POT dense EMD did not return sparse_plan.")
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
            "solver_backend": "ot.solve_sample(lazy=False)",
            "potentials_init": initial_potentials is not None,
        },
    )
