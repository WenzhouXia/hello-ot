"""CN: 独立运行一个公开 Pareto 配置。EN: Run one public Pareto configuration in isolation."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

import numpy as np

from .protocol import fingerprint, load_problem, solve_method, write_json


def reference_objective(problem, backend):
    """CN: 使用 HELLO dual warm start 求 EMD 参考值并检查结果。EN: Solve and validate an EMD reference initialized with HELLO duals."""
    from paper_experiments.baselines.linear_ot.hello_solver import solve_hello_with_duals
    from paper_experiments.baselines.linear_ot.pot_emd import solve_pot_emd
    from paper_experiments.baselines.linear_ot.pot_lazy_emd import solve_pot_lazy_emd
    from paper_experiments.baselines.linear_ot.evaluation import evaluate_transport

    hello = solve_hello_with_duals(problem)
    duals = (hello.source_dual, hello.target_dual)
    solver = solve_pot_emd if backend == "dense" else solve_pot_lazy_emd
    try:
        result = solver(problem, potentials_init=duals, max_iterations=10000000)
    except MemoryError as exc:
        raise RuntimeError("Dense EMD ran out of memory; rerun with --reference lazy.") from exc
    if not result.converged:
        raise RuntimeError(f"EMD reference did not converge: {result.status}")
    evaluation = evaluate_transport(problem, transport=result.transport, transport_kind=result.transport_kind)
    if not np.isfinite(evaluation.objective) or evaluation.primal_feasibility > 1e-8:
        raise RuntimeError(f"Invalid EMD reference: {evaluation}")
    return float(evaluation.objective)


def main():
    """CN: 隔离预热、求解和误差评价，并将失败诊断写入日志。EN: Isolate warmup, solve and error evaluation, logging failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    output = Path(request["output"])
    method = request["method"]
    record = dict(
        method=method, parameter=request.get("parameter"), n=request["n"], d=request["d"],
        seed=42, runtime_sec=None, relative_error=None, status="failed", error="",
    )
    try:
        problem = load_problem(Path(request["feature_file"]), request["n"], request["d"])
        if fingerprint(problem) != request["fingerprint"]:
            raise ValueError("Data fingerprint changed after scheduling.")
        if method == "reference":
            value = reference_objective(problem, request["reference"])
            write_json(output, dict(
                fingerprint=request["fingerprint"], backend=request["reference"],
                objective=value, status="success",
            ))
            return
        # CN: 同 shape、同参数预热以覆盖 JAX 静态编译；预热时间不进入结果。
        # EN: Warm up the same shape and parameters to cover static JAX compilation; exclude warmup time.
        warmup = solve_method(problem, method, request["parameter"])
        del warmup
        gc.collect()
        result = solve_method(problem, method, request["parameter"])
        record["runtime_sec"] = float(result.runtime_sec)
        if not result.converged:
            raise RuntimeError(f"Solver did not converge: {result.status}")
        from paper_experiments.common.feasible_rounding import (
            relative_error_from_rounded_objective, rounded_transport_objective,
        )
        objective = rounded_transport_objective(
            source_points=problem.source_points, target_points=problem.target_points,
            source_mass=problem.source_mass, target_mass=problem.target_mass,
            transport=result.transport, cost_type="l2^2",
        )
        reference = request.get("reference_objective")
        if reference is None:
            raise RuntimeError("Reference unavailable; generate it with --reference dense or --reference lazy.")
        relative, valid = relative_error_from_rounded_objective(
            objective, reference, relative_tolerance=1e-6,
        )
        if not valid:
            raise RuntimeError(f"Invalid rounded objective {objective}; reference={reference}")
        if not np.isfinite(record["runtime_sec"]) or record["runtime_sec"] <= 0:
            raise RuntimeError("Invalid solver runtime")
        record.update(relative_error=relative, status="success")
    except Exception as exc:
        traceback.print_exc()
        record["error"] = str(exc)
        if "result" in locals():
            print("Failure diagnostics:", result.diagnostics, flush=True)
    write_json(output, record)


if __name__ == "__main__":
    main()
