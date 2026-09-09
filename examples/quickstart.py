"""CN: 在只安装 PyTorch、POT 与基础科学计算包的设备上运行 HELLO。
EN: Run HELLO on a device with only PyTorch, POT, and basic scientific packages.
"""

from __future__ import annotations

import numpy as np

import hello_ot


def _brenier_problem(*, size: int, dimension: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    CN: 生成严格凸 Brenier 映射 T(x)=x+2 tanh(x) 对应的离散 OT 问题。
    EN: Generate a discrete OT problem from the strictly convex Brenier map T(x)=x+2 tanh(x).
    """
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(size, dimension)).astype(np.float32)
    mapped = source + np.float32(2.0) * np.tanh(source)
    target = mapped[rng.permutation(size)]
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


def main() -> None:
    """
    CN: 显式选择 portable Torch backend，并打印可复核的求解摘要。
    EN: Explicitly select the portable Torch backend and print a checkable solve summary.
    """
    size = 2048
    dimension = 32
    seed = 42
    source, target, ground_truth_objective = _brenier_problem(
        size=size, dimension=dimension, seed=seed
    )
    options = hello_ot.SolverOptions(
        backend="torch",
        torch_device="auto",
        coarsest_size_threshold=1024,
        cost_perturbation="auto",
        verbose="compact",
    )
    result = hello_ot.solve(source, target, cost="l2^2", random_seed=seed, options=options)
    pfeas, dfeas, gap, kkt = _final_diagnostics(result)
    perturbation = dict(result.metadata["cost_perturbation"])
    relative_objective_error = abs(result.objective - ground_truth_objective) / abs(ground_truth_objective)

    print("Quickstart result")
    print(f"backend: {result.metadata['backend']}")
    print(f"device: {result.metadata['device']}")
    print(f"cost perturbation: policy={perturbation['policy']} activated={perturbation['activated']}")
    print(f"objective: {result.objective:.8f}")
    print(f"ground-truth objective: {ground_truth_objective:.8f}")
    print(f"relative objective error: {relative_objective_error:.2e}")
    print(f"primal feasibility: {pfeas:.2e}")
    print(f"dual feasibility: {dfeas:.2e}")
    print(f"primal-dual gap: {gap:.2e}")
    print(f"KKT error: {kkt:.2e}")
    print(f"elapsed: {_elapsed(result.total_wall_time)}")


if __name__ == "__main__":
    main()
