from __future__ import annotations

import inspect

import numpy as np
import pytest

import hello_ot


def test_public_surface_is_small_and_flat() -> None:
    assert sorted(hello_ot.__all__) == sorted([
        "GromovResult",
        "PrewarmStats",
        "Problem",
        "Result",
        "SemiDiscreteResult",
        "SolverOptions",
        "UnbalancedResult",
        "prewarm",
        "solve",
        "solve_gromov",
        "solve_semidiscrete",
        "solve_unbalanced",
    ])
    parameters = inspect.signature(hello_ot.solve).parameters
    assert "tolerance" not in parameters
    assert "config" not in parameters
    assert "solve_stage" in inspect.signature(hello_ot.Result).parameters
    assert "original_cost_stage" not in inspect.signature(hello_ot.Result).parameters
    assert "perturbed_cost_stage" not in inspect.signature(hello_ot.Result).parameters


def test_array_input_uses_public_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_solve_problem(problem, config):
        captured["problem"] = problem
        captured["config"] = config
        return object()

    import hello_ot.api

    monkeypatch.setattr(hello_ot.api, "solve_problem", fake_solve_problem)
    result = hello_ot.solve(np.zeros((2, 3)), np.ones((4, 3)))

    assert result is not None
    problem = captured["problem"]
    config = captured["config"]
    assert problem.cost_type == "l2^2"
    np.testing.assert_array_equal(problem.source_mass, np.full(2, 0.5))
    np.testing.assert_array_equal(problem.target_mass, np.full(4, 0.25))
    assert config.lp_tolerance == 1e-6
    assert config.dual_feasibility_tolerance == 1e-6
    assert config.variable_bound_mode == "constant"
    assert config.matrix_value_mode == "implicit_aty"
    assert config.vector_sum_mode == "direct_reduce"
    assert config.backend == "native"
    assert config.torch_device == "auto"
    assert config.cost_perturbation == "auto"
    assert config.cost_perturbation_relative_scale == 0.01


def test_problem_input_rejects_repeated_problem_data(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = hello_ot.Problem(np.zeros((2, 1)), np.ones((2, 1)), "l1")
    with pytest.raises(TypeError, match="must not be repeated"):
        hello_ot.solve(problem, cost="l2^2")


def test_advanced_options_are_flat() -> None:
    options = hello_ot.SolverOptions(assignment_topk=8, support_budget_factor=6.0)
    config = options._to_internal_config(max_iterations=12, random_seed=7)
    assert config.assignment_topk == 8
    assert config.support_budget_factor == 6.0
    assert config.max_iterations == 12
    assert config.random_seed == 7
    assert "primal_tolerance" not in inspect.signature(hello_ot.SolverOptions).parameters


def test_verbose_modes_are_strict_and_compact_by_default() -> None:
    assert hello_ot.SolverOptions().verbose == "compact"
    assert hello_ot.SolverOptions(verbose="off").verbose == "off"
    assert hello_ot.SolverOptions(verbose="detailed").verbose == "detailed"
    for value in (False, True, "yes", None):
        with pytest.raises(ValueError, match="off, compact, detailed"):
            hello_ot.SolverOptions(verbose=value)  # type: ignore[arg-type]


def test_backend_selection_is_explicit_and_validated() -> None:
    assert hello_ot.SolverOptions().backend == "native"
    assert hello_ot.SolverOptions(backend="torch", torch_device="cpu").backend == "torch"
    with pytest.raises(ValueError, match="backend"):
        hello_ot.SolverOptions(backend="auto")  # type: ignore[arg-type]


def test_stopping_norm_public_choices() -> None:
    assert hello_ot.SolverOptions().stopping_norm == "l2"
    assert hello_ot.SolverOptions(stopping_norm="finest_linf").stopping_norm == "finest_linf"
    with pytest.raises(ValueError, match="stopping_norm"):
        hello_ot.SolverOptions(stopping_norm="linf")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stopping_norm"):
        hello_ot.SolverOptions(stopping_norm="other")  # type: ignore[arg-type]


def test_finest_linf_stopping_schedule() -> None:
    from hello_ot.config import level_stopping_norm

    assert level_stopping_norm("l2", level_index=0) == "l2"
    assert level_stopping_norm("l2", level_index=3) == "l2"
    assert level_stopping_norm("finest_linf", level_index=0) == "linf"
    assert level_stopping_norm("finest_linf", level_index=1) == "l2"
