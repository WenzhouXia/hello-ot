"""CN: 在只安装 PyTorch、POT 与基础科学计算包的设备上运行 HELLO。
EN: Run HELLO on a device with only PyTorch, POT, and basic scientific packages.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

import hello_ot


def _positive_int(value: str) -> int:
    """
    CN: 解析严格为正的命令行整数。
    EN: Parse a strictly positive command-line integer.
    """
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _hierarchical_size(value: str) -> int:
    """
    CN: 要求 quickstart 规模足以触发 hierarchy。
    EN: Require a quickstart size large enough to trigger the hierarchy.
    """
    parsed = int(value)
    if parsed <= 1024:
        raise argparse.ArgumentTypeError(
            "must be greater than 1024 to trigger the hierarchy"
        )
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    CN: 解析 quickstart 的问题规模、代价与 backend。
    EN: Parse the quickstart problem size, cost, and backend.
    """
    parser = argparse.ArgumentParser(
        description="Run HELLO on a synthetic Brenier problem."
    )
    parser.add_argument(
        "--n",
        type=_hierarchical_size,
        default=2048,
        help="points per marginal (default: 2048)",
    )
    parser.add_argument(
        "--dim",
        type=_positive_int,
        default=32,
        help="point dimension (default: 32)",
    )
    parser.add_argument(
        "--cost-type",
        choices=("l2^2", "l2", "l1", "linf"),
        default="l2^2",
        help="ground cost (default: l2^2)",
    )
    parser.add_argument(
        "--backend",
        choices=("native", "torch"),
        default="torch",
        help="solver backend (default: torch)",
    )
    return parser.parse_args(argv)


def _brenier_problem(*, n: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    CN: 生成严格凸 Brenier 映射 T(x)=x+2 tanh(x) 对应的离散 OT 问题。
    EN: Generate a discrete OT problem from the strictly convex Brenier map T(x)=x+2 tanh(x).
    """
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n, dim)).astype(np.float32)
    mapped = source + np.float32(2.0) * np.tanh(source)
    target = mapped[rng.permutation(n)]
    displacement = source.astype(np.float64) - mapped.astype(np.float64)
    ground_truth_objective = float(np.mean(np.sum(displacement * displacement, axis=1)))
    return np.ascontiguousarray(source), np.ascontiguousarray(target), ground_truth_objective


def _final_diagnostics(result: hello_ot.Result) -> tuple[float, float, float, float]:
    """
    CN: 读取最终原成本、finest-level 的最后一轮 KKT 诊断，并拒绝不完整结果。
    EN: Read the last finest-level KKT diagnostics for the final original cost and reject incomplete results.
    """
    from hello_ot.types import RefinementResult

    refined = [level for level in result.solve_stage.levels if isinstance(level.solve, RefinementResult)]
    if not refined or not refined[-1].solve.iterations:
        raise RuntimeError("quickstart requires final finest-level refinement diagnostics")
    final_level = refined[-1].solve
    if not final_level.summary.converged:
        raise RuntimeError(f"HELLO did not converge: {final_level.summary.stop_reason}")
    final = final_level.iterations[-1]
    values = (
        final.solve_lp.primal_feasibility,
        final.convergence.dual_feasibility,
        final.solve_lp.primal_dual_gap,
    )
    if any(value is None or not np.isfinite(float(value)) for value in values):
        raise RuntimeError("quickstart final KKT diagnostics are incomplete")
    pfeas, dfeas, gap = (float(value) for value in values)
    return pfeas, dfeas, gap, max(abs(pfeas), abs(dfeas), abs(gap))


def _elapsed(value: float) -> str:
    return f"{value:.2f}s" if value < 10.0 else f"{value:.1f}s"


def main(argv: Sequence[str] | None = None) -> None:
    """
    CN: 在可选 backend 上求解合成问题，并打印可复核的求解摘要。
    EN: Solve the synthetic problem on the selected backend and print a checkable summary.
    """
    args = _parse_args(argv)
    n = args.n
    dim = args.dim
    cost_type = args.cost_type
    backend = args.backend
    seed = 42
    source, target, ground_truth_objective = _brenier_problem(
        n=n, dim=dim, seed=seed
    )
    options = hello_ot.SolverOptions(
        backend=backend,
        torch_device="auto",
        coarsest_size_threshold=1024,
        cost_perturbation="auto",
        verbose="compact",
    )
    result = hello_ot.solve(
        source,
        target,
        cost=cost_type,
        random_seed=seed,
        options=options,
    )
    pfeas, dfeas, gap, kkt = _final_diagnostics(result)
    perturbation = dict(result.metadata["cost_perturbation"])

    print("Quickstart result")
    print(f"problem: n={n} dim={dim} cost={cost_type}")
    print(f"backend: {result.metadata['backend']}")
    print(f"device: {result.metadata['device']}")
    print(
        "cost perturbation: "
        f"policy={perturbation['policy']} activated={perturbation['activated']}"
    )
    print(f"objective: {result.objective:.8f}")
    if cost_type == "l2^2":
        relative_objective_error = (
            abs(result.objective - ground_truth_objective)
            / abs(ground_truth_objective)
        )
        print(f"ground-truth objective: {ground_truth_objective:.8f}")
        print(f"relative objective error: {relative_objective_error:.2e}")
    else:
        print("ground-truth objective: unavailable (Brenier certificate requires l2^2)")
    print(f"primal feasibility: {pfeas:.2e}")
    print(f"dual feasibility: {dfeas:.2e}")
    print(f"primal-dual gap: {gap:.2e}")
    print(f"KKT error: {kkt:.2e}")
    print(f"elapsed: {_elapsed(result.total_wall_time)}")


if __name__ == "__main__":
    main()
