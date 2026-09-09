from __future__ import annotations

import numpy as np
import ot
import pytest
import torch

import hello_ot


def test_default_native_failure_never_falls_back_to_torch(monkeypatch, capsys) -> None:
    # CN: native 环境缺失必须显式失败，绝不进入 portable backend。
    # EN: A missing native environment must fail explicitly and never enter the portable backend.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        hello_ot.solve(np.zeros((2, 1)), np.ones((2, 1)))
    error_output = capsys.readouterr().err
    assert "backend=torch" not in error_output
    assert "HELLO failed | error=" in error_output


def test_torch_off_is_silent(monkeypatch, capsys) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source = np.arange(8, dtype=np.float32).reshape(4, 2)
    hello_ot.solve(
        source,
        source.copy(),
        options=hello_ot.SolverOptions(
            backend="torch", torch_device="cpu", cost_perturbation="off", verbose="off"
        ),
    )
    assert capsys.readouterr().err == ""


def test_torch_cpu_runs_non_leaf_hello_without_native(monkeypatch, capsys) -> None:
    # CN: native import 一旦发生就失败，确保端到端 portable 路径没有静默混用扩展。
    # EN: Fail on any native import so the end-to-end portable path cannot silently mix extensions.
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if str(name).startswith("hello_ot._native"):
            raise AssertionError(f"Torch backend imported native module: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(7)
    source = rng.normal(size=(32, 2))
    target = rng.normal(size=(32, 2))
    result = hello_ot.solve(
        source,
        target,
        max_iterations=10,
        options=hello_ot.SolverOptions(
            backend="torch",
            torch_device="cpu",
            coarsest_size_threshold=8,
            assignment_topk=2,
            pricing_topk=1,
            support_budget_factor=6.0,
        ),
    )
    captured = capsys.readouterr()
    assert "HELLO | backend=torch device=cpu" in captured.err
    assert "[original] level=" in captured.err
    assert "init done |" in captured.err
    assert "iter=1 obj=" in captured.err
    assert " lp=" in captured.err
    assert " full_scan=" in captured.err
    assert "HELLO converged | obj=" in captured.err
    assert result.metadata["backend"] == "torch"
    assert result.metadata["device"] == "cpu"
    assert len(result.solve_stage.levels) >= 2
    assert any(level.kind == "refined" for level in result.solve_stage.levels)
    assert all(
        level.solve.summary.converged
        for level in result.solve_stage.levels
        if level.kind == "refined"
    )
    assert all(
        iteration.full_scan_time > 0.0
        for level in result.solve_stage.levels
        if level.kind == "refined"
        for iteration in level.solve.iterations
    )

    solution = result.solution
    source_marginal = np.bincount(solution.rows, weights=solution.values, minlength=32)
    target_marginal = np.bincount(solution.cols, weights=solution.values, minlength=32)
    np.testing.assert_allclose(source_marginal, np.full(32, 1.0 / 32.0), atol=2e-7)
    np.testing.assert_allclose(target_marginal, np.full(32, 1.0 / 32.0), atol=2e-7)
    cost = ot.dist(source, target, metric="sqeuclidean")
    reference = ot.emd2(np.full(32, 1.0 / 32.0), np.full(32, 1.0 / 32.0), cost)
    assert abs(result.objective - float(reference)) < 2e-6


@pytest.mark.parametrize(
    ("cost_type", "pot_metric"),
    [("l1", "cityblock"), ("l2", "euclidean"), ("linf", "chebyshev")],
)
def test_torch_cpu_supports_all_norm_costs(cost_type: str, pot_metric: str) -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(16, 3))
    target = rng.normal(size=(16, 3))
    result = hello_ot.solve(
        source,
        target,
        cost=cost_type,
        max_iterations=12,
        options=hello_ot.SolverOptions(
            backend="torch",
            torch_device="cpu",
            coarsest_size_threshold=4,
            assignment_topk=2,
            pricing_topk=1,
            support_budget_factor=6.0,
        ),
    )
    cost = ot.dist(source, target, metric=pot_metric)
    reference = ot.emd2(np.full(16, 1.0 / 16.0), np.full(16, 1.0 / 16.0), cost)
    assert abs(result.objective - float(reference)) < 3e-6
    assert all(
        level.solve.summary.converged
        for level in result.solve_stage.levels
        if level.kind == "refined"
    )
