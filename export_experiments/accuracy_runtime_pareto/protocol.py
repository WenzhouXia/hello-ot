"""
CN: 公开 Pareto 的固定参数、数据指纹和简洁输出。
EN: Fixed public Pareto settings, data fingerprints and compact output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from paper_experiments.baselines.linear_ot.problem import LinearOTProblem

CONFIG_PATH = Path(__file__).with_name("config.json")
METHODS = ("hello", "sinkhorn", "ipot", "mdot")
FIELDS = ("method", "parameter", "n", "d", "seed", "runtime_sec", "relative_error", "status", "error")


def load_config():
    """CN: 读取公开默认配置。EN: Read the public defaults."""
    return json.loads(CONFIG_PATH.read_text())


def load_problem(feature_file: Path, n: int, d: int):
    """CN: 按维度读取 target 并生成 seed=42 source。EN: Load a dimension-specific target and generate seed=42 source."""
    from export_experiments.main_scaling.data import load_representative_features
    target = load_representative_features(feature_file=feature_file, min_rows=n, min_dimension=d)
    source = np.random.default_rng(42).standard_normal((n, d), dtype=np.float32)
    return LinearOTProblem(source, np.array(target[:n], dtype=np.float32, order="C", copy=True))


def fingerprint(problem):
    """CN: 将数据和质量绑定到参考值。EN: Bind reference values to data and masses."""
    digest = hashlib.sha256(b"hello-public-pareto-v1:l2^2:seed42")
    for array in (problem.source_points, problem.target_points, problem.source_mass, problem.target_mass):
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def solve_method(problem, method, parameter):
    """CN: 调用共享 adapter，保持评价在 solver 计时外。EN: Call shared adapters with evaluation outside solver timing."""
    if method == "hello":
        from paper_experiments.baselines.linear_ot.hello_solver import solve_hello
        return solve_hello(problem)
    if method == "sinkhorn":
        from paper_experiments.baselines.linear_ot.ott_sinkhorn import solve_ott_jax_sinkhorn_l1_negdot_std
        return solve_ott_jax_sinkhorn_l1_negdot_std(
            problem, epsilon=parameter, max_iterations=50000, tolerance=1e-3,
            dtype_name="float64", batch_size=2048,
        )
    if method == "ipot":
        from paper_experiments.baselines.linear_ot.pot_proximal import solve_pot_proximal_point
        return solve_pot_proximal_point(
            problem, backend="keops", max_iterations=10000, tolerance=1e-5,
            inner_iterations=1, inner_regularization=parameter, dtype_name="float32",
        )
    if method == "mdot":
        from paper_experiments.baselines.linear_ot.mdot_tnt_adapter import solve_mdot_tnt
        return solve_mdot_tnt(problem, gamma_f=parameter, backend="keops")
    raise ValueError(f"Unknown method: {method}")


def write_json(path, value):
    """CN: 原子写入结果。EN: Write results atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
