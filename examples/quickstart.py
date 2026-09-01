"""CN: 在只安装 PyTorch、POT 与基础科学计算包的设备上运行 HELLO。
EN: Run HELLO on a device with only PyTorch, POT, and basic scientific packages.
"""

from __future__ import annotations

import time

import numpy as np

import hello_ot


def main() -> None:
    """
    CN: 显式选择 portable Torch backend，并打印可复核的求解摘要。
    EN: Explicitly select the portable Torch backend and print a checkable solve summary.
    """
    size = 2048
    dimension = 32
    rng = np.random.default_rng(7)
    source = rng.normal(size=(size, dimension))
    target = rng.normal(size=(size, dimension))
    options = hello_ot.SolverOptions(
        backend="torch",
        torch_device="auto",
        coarsest_size_threshold=1024,
    )
    started = time.perf_counter()
    result = hello_ot.solve(source, target, options=options)
    elapsed = time.perf_counter() - started
    source_marginal = np.bincount(result.solution.rows, weights=result.solution.values, minlength=size)
    target_marginal = np.bincount(result.solution.cols, weights=result.solution.values, minlength=size)
    source_residual = np.linalg.norm(source_marginal - np.full(size, 1.0 / size))
    target_residual = np.linalg.norm(target_marginal - np.full(size, 1.0 / size))
    print(f"backend={result.metadata['backend']} device={result.metadata['device']}")
    print(f"objective={result.objective:.10f} edges={result.solution.values.size}")
    print(f"marginal_l2=max({source_residual:.3e}, {target_residual:.3e}) elapsed={elapsed:.3f}s")
    print("Paper experiments should explicitly use SolverOptions(backend='native').")


if __name__ == "__main__":
    main()
