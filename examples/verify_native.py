from __future__ import annotations

import time

import numpy as np

import hello_ot
from hello_ot.diagnose import collect_diagnostics


def main() -> None:
    """
    CN: 显式运行 native backend，并验证边际残差与四个扩展的加载状态。
    EN: Explicitly run the native backend and validate marginal residuals and all four extensions.
    """
    report = collect_diagnostics()
    missing = [name for name, item in report["native_extensions"].items() if not item["available"]]
    if missing:
        raise RuntimeError(f"native extensions are unavailable: {missing}")
    rng = np.random.default_rng(11)
    size, dimension = 128, 8
    source = rng.normal(size=(size, dimension))
    target = rng.normal(size=(size, dimension))
    options = hello_ot.SolverOptions(
        backend="native",
        coarsest_size_threshold=32,
        assignment_topk=8,
    )
    started = time.perf_counter()
    result = hello_ot.solve(source, target, options=options)
    elapsed = time.perf_counter() - started
    source_marginal = np.bincount(result.solution.rows, weights=result.solution.values, minlength=size)
    target_marginal = np.bincount(result.solution.cols, weights=result.solution.values, minlength=size)
    residual = max(
        float(np.linalg.norm(source_marginal - np.full(size, 1.0 / size))),
        float(np.linalg.norm(target_marginal - np.full(size, 1.0 / size))),
    )
    if not np.isfinite(result.objective) or residual > 1e-5:
        raise RuntimeError(f"native verification failed: objective={result.objective} residual={residual}")
    print(f"backend=native objective={result.objective:.10f} edges={result.solution.values.size}")
    print(f"marginal_l2={residual:.3e} elapsed={elapsed:.3f}s")


if __name__ == "__main__":
    main()
