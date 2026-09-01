from __future__ import annotations

import numpy as np
import ot
import pytest
import torch

from hello_ot._internal.lp.torch_restricted_ot import (
    TorchRestrictedOTPDLP,
    _gather_transpose,
    _residuals,
    _scatter_marginals,
)


def _small_problem() -> dict[str, np.ndarray]:
    return {
        "rows": np.array([0, 0, 1, 1], dtype=np.int64),
        "cols": np.array([0, 1, 0, 1], dtype=np.int64),
        "costs": np.array([0.0, 2.0, 1.0, 0.0], dtype=np.float64),
        "source_mass": np.array([0.4, 0.6], dtype=np.float64),
        "target_mass": np.array([0.5, 0.5], dtype=np.float64),
    }


def test_endpoint_operators_match_incidence_matrix() -> None:
    rows = torch.tensor([0, 0, 1, 1])
    cols = torch.tensor([0, 1, 0, 1])
    x = torch.tensor([0.1, 0.3, 0.4, 0.2], dtype=torch.float64)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(
        _scatter_marginals(x, rows, cols, 2, 2),
        torch.tensor([0.4, 0.6, 0.5, 0.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        _gather_transpose(y, rows, cols, 2),
        torch.tensor([4.0, 5.0, 5.0, 6.0], dtype=torch.float64),
    )


def test_residual_contract_on_known_optimum() -> None:
    problem = _small_problem()
    residual = _residuals(
        primal=torch.tensor([0.4, 0.0, 0.1, 0.5], dtype=torch.float64),
        dual=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        rows=torch.as_tensor(problem["rows"]),
        cols=torch.as_tensor(problem["cols"]),
        costs=torch.as_tensor(problem["costs"]),
        masses=torch.as_tensor(np.concatenate((problem["source_mass"], problem["target_mass"]))),
        n_source=2,
        n_target=2,
    )
    assert residual.absolute_primal == pytest.approx(0.0, abs=1e-15)
    assert residual.absolute_dual == pytest.approx(0.0, abs=1e-15)
    assert residual.primal_objective == pytest.approx(0.1)


@pytest.mark.parametrize("warm_start", [False, True])
def test_torch_restricted_ot_solves_small_problem_on_cpu(warm_start: bool) -> None:
    problem = _small_problem()
    kwargs = {}
    if warm_start:
        kwargs = {
            "warm_start_primal": np.array([0.35, 0.05, 0.15, 0.45], dtype=np.float64),
            "warm_start_dual": np.zeros(4, dtype=np.float64),
        }
    result = TorchRestrictedOTPDLP(
        device="cpu",
        evaluation_frequency=20,
        iteration_limit=200_000,
        record_state_trace=True,
    ).solve_restricted(**problem, tolerance=1e-7, **kwargs)
    assert result.success, result.solver_diag
    assert result.x.dtype == torch.float64
    assert result.y.dtype == torch.float64
    assert result.obj_val == pytest.approx(0.1, abs=2e-6)
    assert result.primal_feas < 1e-7
    assert result.dual_feas < 1e-7
    assert result.gap < 1e-7
    assert result.solver_diag["state_trace"]


def test_torch_restricted_ot_rejects_unbalanced_mass() -> None:
    problem = _small_problem()
    problem["target_mass"] = np.array([0.4, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="equal source and target total mass"):
        TorchRestrictedOTPDLP(device="cpu").solve_restricted(**problem)


def test_torch_restricted_ot_matches_pot_on_random_dense_support() -> None:
    rng = np.random.default_rng(3)
    n_source, n_target = 4, 5
    source_mass = np.full(n_source, 1.0 / n_source)
    target_mass = np.full(n_target, 1.0 / n_target)
    costs = rng.random((n_source, n_target))
    rows, cols = np.indices(costs.shape)
    result = TorchRestrictedOTPDLP(device="cpu").solve_restricted(
        rows=rows.ravel(),
        cols=cols.ravel(),
        costs=costs.ravel(),
        source_mass=source_mass,
        target_mass=target_mass,
        tolerance=1e-6,
    )
    reference = ot.emd(source_mass, target_mass, costs)
    assert result.success, result.solver_diag
    assert result.obj_val == pytest.approx(float(np.sum(reference * costs)), abs=2e-6)
    assert result.primal_feas < 1e-6
    assert result.dual_feas < 1e-6
    assert result.gap < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Torch CUDA is unavailable")
def test_torch_restricted_ot_cuda_matches_cpu() -> None:
    problem = _small_problem()
    cpu = TorchRestrictedOTPDLP(device="cpu", evaluation_frequency=20).solve_restricted(
        **problem, tolerance=1e-7
    )
    cuda = TorchRestrictedOTPDLP(device="cuda", evaluation_frequency=20).solve_restricted(
        **problem, tolerance=1e-7
    )
    assert cuda.success
    torch.testing.assert_close(cuda.x.cpu(), cpu.x.cpu(), rtol=1e-7, atol=1e-8)
    assert cuda.obj_val == pytest.approx(cpu.obj_val, rel=1e-7, abs=1e-8)
