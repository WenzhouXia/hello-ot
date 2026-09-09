"""CN: 运行公开四方法 Pareto 扫描。EN: Run the public four-method Pareto sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .protocol import METHODS, fingerprint, load_config, load_problem, write_json


def run_worker(request, root, timeout):
    """CN: 独立进程运行；超时或进程被杀也保留失败记录。EN: Run in isolation and retain failures on timeout or process termination."""
    output = Path(request["output"])
    request_path = output.with_suffix(".request.json")
    log_path = output.with_suffix(".log")
    write_json(request_path, request)
    output.unlink(missing_ok=True)
    environment = dict(os.environ)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["JAX_ENABLE_X64"] = "true"
    environment["JAX_PLATFORMS"] = "cuda"
    error = ""
    with log_path.open("w") as log:
        try:
            process = subprocess.run(
                [sys.executable, "-m", "export_experiments.accuracy_runtime_pareto.worker",
                 "--request", str(request_path)],
                cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
            )
            if process.returncode:
                error = f"Worker exited with code {process.returncode}; see {log_path.name}"
        except subprocess.TimeoutExpired:
            error = f"Worker exceeded {timeout}s; see {log_path.name}"
    if not output.exists():
        if request["method"] == "reference" and request["reference"] == "dense":
            error += "; if memory is insufficient, rerun with --reference lazy"
        write_json(output, dict(
            method=request["method"], parameter=request.get("parameter"),
            n=request["n"], d=request["d"], seed=42,
            runtime_sec=None, relative_error=None, status="failed", error=error or "No worker result",
        ))
    return json.loads(output.read_text())


def main():
    """CN: 准备数据和参考值，再顺序运行各方法。EN: Prepare data and references, then run methods sequentially."""
    config = load_config()
    parser = argparse.ArgumentParser(description="Public Gaussian-to-ImageNet Pareto experiment (seed=42, squared-L2).")
    parser.add_argument("--n", type=int, default=config["n"])
    parser.add_argument("--d", type=int, nargs="+", choices=config["d"], default=config["d"])
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--parameter", type=float, help="Run one parameter instead of the default sweep (one method only).")
    parser.add_argument("--reference", choices=["dense", "lazy"], default=config["reference"])
    parser.add_argument("--feature-file", type=Path, help="Local dimension-specific file; requires one --d.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/accuracy_runtime_pareto"))
    parser.add_argument("--timeout", type=float, default=3600, help="Seconds per worker, including warmup and evaluation.")
    args = parser.parse_args()
    if not 1 <= args.n <= 262144 or args.timeout <= 0:
        parser.error("n must be between 1 and 262144; timeout must be positive")
    if args.feature_file and len(args.d) != 1:
        parser.error("--feature-file requires one --d")
    if args.parameter is not None and (len(args.method) != 1 or args.method == ["hello"] or args.parameter <= 0):
        parser.error("--parameter requires one non-HELLO method and a positive value")
    root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    from export_experiments.main_scaling.data import ensure_representative_features
    for dimension in args.d:
        path = ensure_representative_features(args.feature_file, dimension=dimension)
        problem = load_problem(path, args.n, dimension)
        data_hash = fingerprint(problem)
        del problem
        case_dir = output_dir / f"n{args.n}_d{dimension}"
        case_dir.mkdir(exist_ok=True)
        identity_path = case_dir / "dataset.json"
        if identity_path.exists() and json.loads(identity_path.read_text())["fingerprint"] != data_hash:
            raise ValueError("Output directory contains a different dataset; choose a new --output-dir.")
        write_json(identity_path, dict(fingerprint=data_hash, dataset="imagenet_latent_gaussian_to_data",
                                      cost="l2^2", seed=42))
        common = dict(n=args.n, d=dimension, fingerprint=data_hash, feature_file=str(path),
                      reference=args.reference)
        reference_path = case_dir / f"reference_{args.reference}_{data_hash[:16]}.json"
        # CN: 每次调用重新生成参考值，避免版本变更后复用旧 solver 的结果。
        # EN: Recompute references on each invocation to avoid reusing results across solver revisions.
        reference = run_worker(
            dict(common, method="reference", output=str(reference_path)), root, args.timeout,
        )
        reference_value = reference.get("objective") if reference.get("status") == "success" else None
        if reference_value is None:
            print(f"Reference failed: {reference.get('error')}. Try --reference lazy if dense EMD ran out of memory.", flush=True)
        for method in args.method:
            parameters = [args.parameter] if args.parameter is not None else config["methods"][method]
            for parameter in parameters:
                name = "default" if parameter is None else format(parameter, ".8g")
                result = run_worker(dict(
                    common, method=method, parameter=parameter, reference_objective=reference_value,
                    output=str(case_dir / f"{method}_{name}.json"),
                ), root, args.timeout)
                print(f"N={args.n} D={dimension} {method} {name}: {result['status']} {result.get('error', '')}", flush=True)
    from .collect import collect
    collect(output_dir)


if __name__ == "__main__":
    main()
