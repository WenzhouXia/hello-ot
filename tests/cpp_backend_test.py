from __future__ import annotations

import importlib
import sys

import numpy as np
import ot
import pytest
import torch

import hello_ot


@pytest.fixture
def cpp_available():
    pytest.importorskip("hello_ot_cpp._lp")
    pytest.importorskip("hello_ot_cpp._scan")


def test_cpp_missing_extension_has_install_instructions(monkeypatch):
    from hello_ot import _cpp_backend

    original = importlib.import_module

    def blocked(name, *args, **kwargs):
        if name.startswith("hello_ot_cpp"):
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    _cpp_backend.load_cpp_extension.cache_clear()
    monkeypatch.setattr(importlib, "import_module", blocked)
    with pytest.raises(RuntimeError, match="install_cpp.sh"):
        hello_ot.solve(np.zeros((4, 2)), np.ones((4, 2)),
                       options=hello_ot.SolverOptions(backend="cpp", verbose="off"))
    _cpp_backend.load_cpp_extension.cache_clear()


@pytest.mark.parametrize("cost,metric", [("l2^2", "sqeuclidean"), ("l1", "cityblock"),
                                         ("l2", "euclidean"), ("linf", "chebyshev")])
@pytest.mark.parametrize("precision", ["fp32", "fp64", "finest_fp64"])
@pytest.mark.parametrize("perturbation", ["off", "on", "auto"])
def test_cpp_hierarchical_costs_match_pot(cpp_available, monkeypatch, cost, metric, precision, perturbation):
    import hello_ot.kernels.torch_scan as scans

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(47)
    x = rng.normal(size=(24, 3)).astype(np.float32)
    y = rng.normal(size=(19, 3)).astype(np.float32)
    a, b = rng.random(24), rng.random(19)
    a, b = a/a.sum(), b/b.sum()
    original_functions = (scans._merge_topk, scans.bidirectional_violation_scan, scans.directional_assignment)
    previous = torch.get_num_threads()
    native_modules = {name for name in sys.modules if name.startswith("hello_ot_native")}
    result = hello_ot.solve(x, y, source_mass=a, target_mass=b, cost=cost, random_seed=42,
                            options=hello_ot.SolverOptions(backend="cpp", num_threads=2, verbose="off",
                                coarsest_size_threshold=4, assignment_topk=2, pricing_topk=1,
                                support_budget_factor=6, cost_perturbation=perturbation, cost_precision=precision))
    assert torch.get_num_threads() == previous
    assert original_functions == (scans._merge_topk, scans.bidirectional_violation_scan, scans.directional_assignment)
    assert result.metadata["backend"] == "cpp" and result.metadata["device"] == "cpu"
    assert result.metadata["num_threads"] == 2
    assert any(level.kind == "refined" for level in result.solve_stage.levels)
    coupling = result.solution.to_sparse_matrix()
    np.testing.assert_allclose(np.asarray(coupling.sum(axis=1)).ravel(), a, atol=3e-6)
    np.testing.assert_allclose(np.asarray(coupling.sum(axis=0)).ravel(), b, atol=3e-6)
    reference = ot.emd2(a, b, ot.dist(x.astype(float), y.astype(float), metric=metric))
    assert result.objective == pytest.approx(reference, rel=5e-5, abs=1e-5)
    assert all(level.solve.summary.converged for level in result.solve_stage.levels if level.kind == "refined")
    assert {name for name in sys.modules if name.startswith("hello_ot_native")} == native_modules


def test_cpp_threads_restore_on_failure(cpp_available, monkeypatch):
    import hello_ot.api as api

    previous = torch.get_num_threads()

    def fail(*args):
        assert torch.get_num_threads() == 2
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(api, "_solve_impl_unscoped", fail)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        hello_ot.solve(np.zeros((2, 1)), np.ones((2, 1)),
                       options=hello_ot.SolverOptions(backend="cpp", num_threads=2, verbose="off"))
    assert torch.get_num_threads() == previous


@pytest.mark.parametrize("threads", [None, 1, 4])
def test_cpp_lp_matches_torch_checkpoints(cpp_available, threads):
    from hello_ot._internal.lp.cpp_restricted_ot import CppRestrictedOTPDLP
    from hello_ot._internal.lp.torch_restricted_ot import TorchRestrictedOTPDLP

    rng = np.random.default_rng(12)
    rows, cols = np.indices((7, 5))
    args = dict(rows=rows.ravel(), cols=cols.ravel(), costs=rng.random(35),
                source_mass=np.full(7, 1/7), target_mass=np.full(5, 1/5), tolerance=0.0)
    settings = dict(evaluation_frequency=5, iteration_limit=40, record_state_trace=True)
    expected = TorchRestrictedOTPDLP(**settings).solve_restricted(**args)
    actual = CppRestrictedOTPDLP(num_threads=threads, **settings).solve_restricted(**args)
    np.testing.assert_allclose(actual.x, expected.x, atol=2e-11, rtol=2e-10)
    np.testing.assert_allclose(actual.y, expected.y, atol=2e-11, rtol=2e-10)
    assert actual.iterations == expected.iterations == 40
    assert actual.solver_diag["restart_count"] == expected.solver_diag["restart_count"]


def test_cpp_prewarm(cpp_available, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    x = np.arange(32, dtype=np.float32).reshape(16, 2)
    stats = hello_ot.prewarm(hello_ot.Problem(x, x+1, "l2^2"),
                            hello_ot.SolverOptions(backend="cpp", num_threads=2, verbose="off"))
    assert "cpp_restricted_lp:cpu" in stats.operations


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_cpp_thread_validation(threads):
    with pytest.raises(ValueError, match="num_threads"):
        hello_ot.SolverOptions(backend="cpp", num_threads=threads)


def test_cpp_rejects_cuda_and_variants():
    with pytest.raises(ValueError, match="CPU only"):
        hello_ot.SolverOptions(backend="cpp", torch_device="cuda")
    from hello_ot.variants._balanced import algorithm_config

    with pytest.raises(ValueError, match="balanced point-cloud OT only"):
        algorithm_config(hello_ot.SolverOptions(backend="cpp"), max_iterations=10, random_seed=42)
