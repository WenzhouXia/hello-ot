"""CN: 运行公开 main-scaling benchmark。EN: Run the public main-scaling benchmark."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source, target = make_brenier_problem(args.n, args.d, args.seed)
    hello_ot.prewarm(hello_ot.Problem(source, target, cost_type="l2^2"))
    result = hello_ot.solve(source, target, random_seed=args.seed)
    record = {
        "suite": "main_scaling",
        "n": int(args.n),
        "d": int(args.d),
        "seed": int(args.seed),
        "objective": float(result.objective),
        "wall_time": float(result.total_wall_time),
        "peak_gpu_memory_mib": result.peak_gpu_memory_mib,
        "support_size": int(result.solution.values.size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
