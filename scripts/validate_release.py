from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import hello_ot
from hello_ot.diagnose import collect_diagnostics


def _run_backend(backend: str) -> dict[str, Any]:
    """
    CN: 在确定性小问题上运行指定 backend，并返回数值与耗时摘要。
    EN: Run one backend on a deterministic small problem and return numerical and timing summaries.
    """
    rng = np.random.default_rng(19)
    size, dimension = 256, 16
    source = rng.normal(size=(size, dimension))
    target = rng.normal(size=(size, dimension))
    options = hello_ot.SolverOptions(
        backend=backend,  # type: ignore[arg-type]
        torch_device="cuda" if backend == "torch" else "auto",
        coarsest_size_threshold=64,
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
    passed = bool(np.isfinite(result.objective) and residual <= 1e-5)
    return {
        "backend": backend,
        "passed": passed,
        "objective": float(result.objective),
        "edges": int(result.solution.values.size),
        "marginal_l2": residual,
        "elapsed_seconds": float(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an installed HELLO-OT release.")
    parser.add_argument("--backend", choices=("native", "torch", "both"), default="both")
    parser.add_argument("--output", type=Path, default=Path("hello_ot_validation.json"))
    args = parser.parse_args()
    backends = ("torch", "native") if args.backend == "both" else (args.backend,)
    report = collect_diagnostics()
    report["validation"] = [_run_backend(backend) for backend in backends]
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"validation_report={args.output.resolve()}")
    if not all(item["passed"] for item in report["validation"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
