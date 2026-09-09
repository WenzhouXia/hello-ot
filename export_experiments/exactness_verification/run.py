"""CN: 运行公开 Exactness experiment。EN: Run the public exactness experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

import hello_ot
from export_experiments.exactness_verification.metrics import evaluate_sparse_coupling
from export_experiments.synthetic import make_correlated_brenier_problem


SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_PATH = SCRIPT_DIR / "paper_matrix.json"


def _load_matrix() -> dict[str, Any]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def build_cases(
    *,
    n_values: list[int] | None = None,
    d_values: list[int] | None = None,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
) -> list[tuple[int, int, int, str]]:
    """
    CN: 展开 Exactness representative matrix。
    EN: Expand the representative exactness matrix.
    """
    matrix = _load_matrix()
    if seeds is not None and seeds != [42]:
        raise ValueError("Public paper experiments require seed=42.")
    return [
        (int(n), int(d), int(seed), str(method))
        for n in (matrix["n"] if n_values is None else n_values)
        for d in (matrix["d"] if d_values is None else d_values)
        for seed in (matrix["seeds"] if seeds is None else seeds)
        for method in (matrix["methods"] if methods is None else methods)
    ]


def _method_available(method: str) -> tuple[bool, str | None]:
    if method == "hello":
        return True, None
    if method == "hiref":
        upstream = Path(__file__).resolve().parents[2] / "third_party" / "HiRef" / "src"
        legacy = Path(__file__).resolve().parents[2] / "paper_experiments" / "baselines" / "linear_ot" / "hiref" / "src"
        available = upstream.is_dir() or legacy.is_dir()
        return available, None if available else "clone the official HiRef repository into third_party/HiRef"
    if method.startswith("ott_jax_sinkhorn"):
        if importlib.util.find_spec("jax") is None or importlib.util.find_spec("ott") is None:
            return False, "install the optional JAX and OTT-JAX dependencies"
        return True, None
    return False, f"unknown method: {method}"


def _run_hello(source: np.ndarray, target: np.ndarray, seed: int) -> tuple[sparse.spmatrix, float, float, dict[str, Any]]:
    options = hello_ot.SolverOptions(
        cost_perturbation="off", backend="native", profile_memory=True, verbose="off"
    )
    result = hello_ot.solve(source, target, random_seed=int(seed), options=options)
    return (
        result.solution.to_sparse_matrix(),
        float(result.objective),
        float(result.total_wall_time),
        {"peak_gpu_memory_mib": result.peak_gpu_memory_mib},
    )


def _run_hiref(source: np.ndarray, target: np.ndarray, seed: int) -> tuple[sparse.spmatrix, float, float, dict[str, Any]]:
    from export_experiments.exactness_verification.hiref_adapter import run_hiref

    payload = run_hiref(
        source,
        target,
        hierarchy_depth=3,
        max_q=2048,
        max_rank=256,
        base_rank=1,
        sq_euclidean=True,
        seed=int(seed),
        return_mapping=True,
    )
    mapping = np.asarray(payload.pop("mapping"), dtype=np.int64)
    coupling = sparse.coo_matrix(
        (np.full(mapping.shape[0], 1.0 / float(source.shape[0])), (mapping[:, 0], mapping[:, 1])),
        shape=(source.shape[0], target.shape[0]),
    )
    payload["algorithm_mapping"] = mapping
    return coupling, float(payload["distance"]), float(payload["runtime_sec"]), payload


def _run_ott(
    source: np.ndarray,
    target: np.ndarray,
    *,
    epsilon: float,
    ground_truth_cols: np.ndarray,
    ground_truth_objective: float,
    support_threshold: float,
    batch_size: int,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    from paper_experiments.baselines.linear_ot.ott_sinkhorn import (
        evaluate_ott_sinkhorn_duals,
        solve_ott_jax_sinkhorn_l1_negdot_std_dual,
        transport_values_at_indices,
    )
    from paper_experiments.baselines.linear_ot.problem import LinearOTProblem

    problem = LinearOTProblem(source_points=source, target_points=target)
    solution = solve_ott_jax_sinkhorn_l1_negdot_std_dual(
        problem,
        epsilon=float(epsilon),
        max_iterations=50_000,
        tolerance=1.0e-3,
        dtype_name="float64",
        batch_size=int(batch_size),
    )
    evaluation, rounded_objective, details, evaluation_time, rounding_time = evaluate_ott_sinkhorn_duals(
        solution,
        block_size=int(batch_size),
        support_threshold=float(support_threshold),
    )
    rows = np.arange(source.shape[0], dtype=np.int64)
    ground_truth_values = transport_values_at_indices(solution, rows, ground_truth_cols)
    hits = int(np.count_nonzero(ground_truth_values > float(support_threshold)))
    row_argmax = np.asarray(details["row_argmax_cols"], dtype=np.int64)
    row_hits = int(np.count_nonzero(row_argmax == ground_truth_cols))
    matrix_sq_sum = float(details["matrix_sq_sum"])
    overlap = float(np.sum(ground_truth_values, dtype=np.float64))
    n_value = int(source.shape[0])
    difference_sq = max(matrix_sq_sum + 1.0 / n_value - 2.0 * overlap / n_value, 0.0)
    metrics = {
        "relative_objective_error": abs(float(rounded_objective) - float(ground_truth_objective))
        / max(abs(float(ground_truth_objective)), 1.0e-12),
        "support_recall": float(hits / n_value),
        "support_precision": float(hits / int(details["algorithm_support_size"]))
        if int(details["algorithm_support_size"]) else float("nan"),
        "support_hits": hits,
        "algorithm_support_size": int(details["algorithm_support_size"]),
        "row_argmax_recall": float(row_hits / n_value),
        "row_argmax_hits": row_hits,
        "row_argmax_size": n_value,
        "matching_1to1_recall": None,
        "matching_1to1_hits": None,
        "matching_1to1_size": None,
        "matching_1to1_status": "not_materialized",
        "primal_frobenius_relative_error": math.sqrt(difference_sq) * math.sqrt(n_value),
        "primal_frobenius_error": math.sqrt(difference_sq),
        "ground_truth_frobenius_norm": 1.0 / math.sqrt(n_value),
        "row_marginal_l2_error": float(evaluation.row_marginal_l2_error),
        "col_marginal_l2_error": float(evaluation.col_marginal_l2_error),
    }
    diagnostics = {
        "raw_objective": float(evaluation.objective),
        "rounded_objective": float(rounded_objective),
        "evaluation_time_sec": float(evaluation_time),
        "rounding_time_sec": float(rounding_time),
        "converged": bool(solution.converged),
    }
    return metrics, float(solution.runtime_sec), diagnostics


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run representative exactness verification.")
    parser.add_argument("--n", type=int, nargs="+")
    parser.add_argument("--d", type=int, nargs="+")
    parser.add_argument("--seed", type=int, choices=[42], nargs="+")
    parser.add_argument("--method", nargs="+", choices=["hello", "ott_jax_sinkhorn_eps_1e-2", "ott_jax_sinkhorn_eps_1e-3", "hiref"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/exactness_verification"))
    parser.add_argument("--support-threshold", type=float, default=1.0e-8)
    parser.add_argument("--sinkhorn-batch-size", type=int, default=2048)
    args = parser.parse_args()

    matrix = _load_matrix()
    dataset = matrix["dataset"]
    cases = build_cases(n_values=args.n, d_values=args.d, seeds=args.seed, methods=args.method)
    runnable: list[tuple[int, int, int, str]] = []
    for case in cases:
        available, reason = _method_available(case[3])
        if available:
            runnable.append(case)
        else:
            print(f"[exactness] skip method={case[3]}: {reason}", flush=True)
    print(f"[exactness] running {len(runnable)} cases", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cached_problem: tuple[int, int, int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] | None = None
    for index, (n_value, d_value, seed, method) in enumerate(runnable, start=1):
        key = (n_value, d_value, seed)
        if cached_problem is None or cached_problem[:3] != key:
            problem = make_correlated_brenier_problem(
                n_value,
                d_value,
                seed,
                rho=float(dataset["rho"]),
                latent_dim=int(dataset["latent_dim"]),
                linear_scale=float(dataset["linear_scale"]),
            )
            cached_problem = (*key, problem)
        source, target, ground_truth_cols, ground_truth_objective = cached_problem[3]
        print(
            f"[exactness] [{index}/{len(runnable)}] method={method} n={n_value} d={d_value} seed={seed}",
            flush=True,
        )
        if method == "hello":
            coupling, objective, runtime, diagnostics = _run_hello(source, target, seed)
            metrics = evaluate_sparse_coupling(
                coupling,
                ground_truth_cols=ground_truth_cols,
                algorithm_objective=objective,
                ground_truth_objective=ground_truth_objective,
                support_threshold=args.support_threshold,
            )
        elif method == "hiref":
            coupling, objective, runtime, diagnostics = _run_hiref(source, target, seed)
            metrics = evaluate_sparse_coupling(
                coupling,
                ground_truth_cols=ground_truth_cols,
                algorithm_objective=objective,
                ground_truth_objective=ground_truth_objective,
                support_threshold=args.support_threshold,
                algorithm_mapping=np.asarray(diagnostics.pop("algorithm_mapping")),
            )
        else:
            epsilon = 1.0e-2 if method.endswith("1e-2") else 1.0e-3
            metrics, runtime, diagnostics = _run_ott(
                source,
                target,
                epsilon=epsilon,
                ground_truth_cols=ground_truth_cols,
                ground_truth_objective=ground_truth_objective,
                support_threshold=args.support_threshold,
                batch_size=args.sinkhorn_batch_size,
            )
            objective = float(diagnostics["rounded_objective"])
        record = {
            "suite": "exactness_verification",
            "dataset": str(dataset["name"]),
            "method": method,
            "n": n_value,
            "d": d_value,
            "seed": seed,
            "objective": float(objective),
            "ground_truth_objective": float(ground_truth_objective),
            "wall_time": float(runtime),
            **metrics,
            "diagnostics": diagnostics,
        }
        output = args.output_dir / f"{method}_n{n_value}_d{d_value}_seed{seed}.json"
        output.write_text(json.dumps(_jsonable(record), indent=2) + "\n", encoding="utf-8")
        print(f"[exactness] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
