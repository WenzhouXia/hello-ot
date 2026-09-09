"""CN: 运行仅使用公开 SolverOptions 的参数敏感性实验。EN: Run parameter sensitivity through public SolverOptions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import hello_ot
from export_experiments.main_scaling.data import load_representative_features
from export_experiments.main_scaling.run import make_gaussian_source


SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_PATH = SCRIPT_DIR / "paper_matrix.json"


def _load_matrix() -> dict[str, Any]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def build_cases(
    *,
    n_values: list[int] | None = None,
    d_values: list[int] | None = None,
    seeds: list[int] | None = None,
    configurations: list[str] | None = None,
) -> list[tuple[int, int, int, str, dict[str, Any]]]:
    """
    CN: 展开 one-factor-at-a-time sensitivity matrix。
    EN: Expand the one-factor-at-a-time sensitivity matrix.
    """
    matrix = _load_matrix()
    if seeds is not None and seeds != [42]:
        raise ValueError("Public paper experiments require seed=42.")
    all_configurations = dict(matrix["configurations"])
    selected = list(all_configurations) if configurations is None else configurations
    unknown = sorted(set(selected) - set(all_configurations))
    if unknown:
        raise ValueError(f"Unknown configurations: {', '.join(unknown)}")
    return [
        (int(n), int(d), int(seed), name, dict(all_configurations[name]))
        for n in (matrix["n"] if n_values is None else n_values)
        for d in (matrix["d"] if d_values is None else d_values)
        for seed in (matrix["seeds"] if seeds is None else seeds)
        for name in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public SolverOptions sensitivity.")
    parser.add_argument("--n", type=int, nargs="+")
    parser.add_argument("--d", type=int, nargs="+")
    parser.add_argument("--seed", type=int, choices=[42], nargs="+")
    parser.add_argument("--configuration", nargs="+")
    parser.add_argument("--feature-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/parameter_sensitivity"))
    args = parser.parse_args()

    cases = build_cases(
        n_values=args.n,
        d_values=args.d,
        seeds=args.seed,
        configurations=args.configuration,
    )
    print(f"[parameter_sensitivity] running {len(cases)} cases", flush=True)
    dimensions = sorted({case[1] for case in cases})
    if args.feature_file is not None and len(dimensions) != 1:
        raise ValueError("--feature-file requires exactly one --d.")
    features = {
        dimension: load_representative_features(
            feature_file=args.feature_file,
            min_rows=max(case[0] for case in cases),
            min_dimension=dimension,
        )
        for dimension in dimensions
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prewarmed = False
    problem_key: tuple[int, int, int] | None = None
    source: np.ndarray | None = None
    target: np.ndarray | None = None
    for index, (n_value, d_value, seed, name, values) in enumerate(cases, start=1):
        key = (n_value, d_value, seed)
        if key != problem_key:
            source = make_gaussian_source(n_value, d_value, seed)
            target = np.ascontiguousarray(features[d_value][:n_value], dtype=np.float32)
            problem_key = key
        if source is None or target is None:
            raise RuntimeError("parameter-sensitivity problem was not initialized")
        problem = hello_ot.Problem(source, target, cost_type="l2^2")
        if not prewarmed:
            hello_ot.prewarm(problem)
            prewarmed = True
        print(
            f"[parameter_sensitivity] [{index}/{len(cases)}] config={name} n={n_value} d={d_value} seed={seed}",
            flush=True,
        )
        options = hello_ot.SolverOptions(
            cost_perturbation="off",
            split_count=int(values["split_count"]),
            assignment_topk=int(values["assignment_topk"]),
            pricing_topk=int(values["pricing_topk"]),
            support_budget_factor=float(values["support_budget_factor"]),
            backend="native",
            profile_memory=True,
            verbose="off",
        )
        result = hello_ot.solve(problem, random_seed=seed, options=options)
        record = {
            "suite": "parameter_sensitivity",
            "dataset": "imagenet_vavae_pca_gaussian_to_data",
            "configuration": name,
            "n": n_value,
            "d": d_value,
            "seed": seed,
            **values,
            "objective": float(result.objective),
            "wall_time": float(result.total_wall_time),
            "peak_gpu_memory_mib": result.peak_gpu_memory_mib,
            "support_size": int(result.solution.values.size),
        }
        output = args.output_dir / f"{name}_n{n_value}_d{d_value}_seed{seed}.json"
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"[parameter_sensitivity] finished in {result.total_wall_time:.3f}s; wrote {output}", flush=True)


if __name__ == "__main__":
    main()
