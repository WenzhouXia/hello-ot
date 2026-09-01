"""CN: 运行公开 accuracy-runtime Pareto case。EN: Run one public accuracy-runtime Pareto case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hello_ot

from experiments.synthetic import make_brenier_problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assignment-topk", type=int, default=16)
    parser.add_argument("--pricing-topk", type=int, default=2)
    parser.add_argument("--support-budget-factor", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source, target = make_brenier_problem(args.n, args.d, args.seed)
    options = hello_ot.SolverOptions(
        assignment_topk=args.assignment_topk,
        pricing_topk=args.pricing_topk,
        support_budget_factor=args.support_budget_factor,
    )
    result = hello_ot.solve(source, target, random_seed=args.seed, options=options)
    final_level = result.solve_stage.levels[-1]
    iterations = final_level.solve.iterations if hasattr(final_level.solve, "iterations") else ()
    residual = None if not iterations else iterations[-1].convergence.dual_feasibility
    record = {
        "suite": "accuracy_runtime_pareto",
        "n": int(args.n),
        "d": int(args.d),
        "seed": int(args.seed),
        "assignment_topk": int(args.assignment_topk),
        "pricing_topk": int(args.pricing_topk),
        "support_budget_factor": float(args.support_budget_factor),
        "objective": float(result.objective),
        "dual_feasibility": residual,
        "wall_time": float(result.total_wall_time),
        "peak_gpu_memory_mib": result.peak_gpu_memory_mib,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
