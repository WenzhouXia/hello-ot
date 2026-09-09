"""CN: 运行 ImageNet feature representative main scaling。EN: Run representative ImageNet-feature scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import hello_ot
from export_experiments.main_scaling.data import load_representative_features


SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_PATH = SCRIPT_DIR / "paper_matrix.json"


def _load_matrix() -> dict[str, Any]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def build_cases(
    *,
    n_values: list[int] | None = None,
    d_values: list[int] | None = None,
    seeds: list[int] | None = None,
) -> list[tuple[int, int, int]]:
    """
    CN: 展开 representative main-scaling matrix。
    EN: Expand the representative main-scaling matrix.
    """
    matrix = _load_matrix()
    if seeds is not None and seeds != [42]:
        raise ValueError("Public paper experiments require seed=42.")
    return [
        (int(n), int(d), int(seed))
        for n in (matrix["n"] if n_values is None else n_values)
        for d in (matrix["d"] if d_values is None else d_values)
        for seed in (matrix["seeds"] if seeds is None else seeds)
    ]


def make_gaussian_source(n: int, d: int, seed: int) -> np.ndarray:
    """
    CN: 用固定 seed 直接生成 FP32 Gaussian source。
    EN: Generate the FP32 Gaussian source directly from a fixed seed.
    """
    return np.random.default_rng(int(seed)).standard_normal((int(n), int(d)), dtype=np.float32)


def run_cases(
    cases: list[tuple[int, int, int]],
    *,
    features: dict[int, np.ndarray],
    output_dir: Path,
) -> None:
    """
    CN: 顺序执行 main-scaling cases，并持续打印进度。
    EN: Run main-scaling cases sequentially while continuously reporting progress.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prewarmed = False
    total = len(cases)
    for index, (n_value, d_value, seed) in enumerate(cases, start=1):
        source = make_gaussian_source(n_value, d_value, seed)
        target = np.ascontiguousarray(features[d_value][:n_value], dtype=np.float32)
        problem = hello_ot.Problem(source, target, cost_type="l2^2")
        if not prewarmed:
            hello_ot.prewarm(problem)
            prewarmed = True
        print(
            f"[main_scaling] [{index}/{total}] n={n_value} d={d_value} seed={seed}",
            flush=True,
        )
        result = hello_ot.solve(
            problem,
            random_seed=int(seed),
            options=hello_ot.SolverOptions(
                cost_perturbation="off", backend="native", profile_memory=True, verbose="off"
            ),
        )
        record = {
            "suite": "main_scaling",
            "dataset": "imagenet_vavae_pca_gaussian_to_data",
            "backend": "native",
            "n": int(n_value),
            "d": int(d_value),
            "seed": int(seed),
            "objective": float(result.objective),
            "wall_time": float(result.total_wall_time),
            "peak_gpu_memory_mib": result.peak_gpu_memory_mib,
            "support_size": int(result.solution.values.size),
        }
        output = output_dir / f"n{n_value}_d{d_value}_seed{seed}.json"
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"[main_scaling] finished in {result.total_wall_time:.3f}s; wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run representative ImageNet-feature scaling.")
    parser.add_argument("--n", type=int, nargs="+")
    parser.add_argument("--d", type=int, nargs="+")
    parser.add_argument("--seed", type=int, choices=[42], nargs="+")
    parser.add_argument("--feature-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/main_scaling"))
    args = parser.parse_args()

    cases = build_cases(n_values=args.n, d_values=args.d, seeds=args.seed)
    print(f"[main_scaling] running {len(cases)} cases", flush=True)
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
    run_cases(cases, features=features, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
