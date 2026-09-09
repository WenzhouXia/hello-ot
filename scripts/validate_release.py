from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import hello_ot
from hello_ot.diagnose import collect_diagnostics


def _run_backend(backend: str, cost: str = "l2^2", policy: str = "off", torch_device: str = "cuda") -> dict[str, Any]:
    """
    CN: 在确定性小问题上运行指定 backend，并返回数值与耗时摘要。
    EN: Run one backend on a deterministic small problem and return numerical and timing summaries.
    """
    rng = np.random.default_rng(42)
    size, dimension = 256, 16
    source = rng.normal(size=(size, dimension))
    target = rng.normal(size=(size, dimension))
    options = hello_ot.SolverOptions(
        backend=backend,  # type: ignore[arg-type]
        torch_device=torch_device if backend == "torch" else "auto",
        cost_perturbation=policy,
        coarsest_size_threshold=64,
        assignment_topk=8,
    )
    started = time.perf_counter()
    result = hello_ot.solve(source, target, options=options, cost=cost, random_seed=42)
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
        "cost": cost,
        "cost_perturbation": policy,
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
    parser.add_argument("--torch-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    backends = ("torch", "native") if args.backend == "both" else (args.backend,)
    report = collect_diagnostics()
    report["validation"] = []
    for backend in backends:
        for cost in ("l2^2", "l2", "l1", "linf"):
            for policy in ("off", "on", "auto"):
                try:
                    item = _run_backend(backend, cost, policy, args.torch_device)
                except Exception as exc:
                    item = dict(backend=backend, cost=cost, cost_perturbation=policy,
                                passed=False, error=str(exc))
                report["validation"].append(item)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"validation_report={args.output.resolve()}")
    if not all(item["passed"] for item in report["validation"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
