from __future__ import annotations

import gc
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import hello_ot


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from hello_ot._native_compat import require_native_runtime_compatibility
    require_native_runtime_compatibility()
    from hello_ot._native import pycupdlpx
    assert pycupdlpx.GPU_ARRAY_IO_VERSION >= 1, "rebuild native extensions"
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize("cost", ["l2^2", "l1", "l2", "linf"])
@pytest.mark.parametrize("perturbation", ["off", "on"])
def test_gpu_solution_against_dense_pot(cuda, cost, perturbation):
    import ot
    rng = np.random.default_rng(910)
    x = torch.tensor(rng.normal(size=(37, 7)), device=cuda, dtype=torch.float32)
    y = torch.tensor(rng.normal(size=(43, 7)), device=cuda, dtype=torch.float32)
    before_x, before_y = x.clone(), y.clone()
    a, b = rng.uniform(0.1, 2, 37), rng.uniform(0.1, 2, 43)
    a, b = a / a.sum(), b / b.sum()
    result = hello_ot.solve(x, y, source_mass=torch.tensor(a, device=cuda), target_mass=b, cost=cost,
        options=hello_ot.SolverOptions(gpu_resident=True, consume_input_features=True,
            coarsest_size_threshold=8, assignment_topk=2, pricing_topk=2,
            support_budget_factor=4, cost_perturbation=perturbation, verbose="off"))
    assert torch.equal(x, before_x) and torch.equal(y, before_y)
    solution = result.solution
    assert all(getattr(solution, k).is_cuda for k in ("rows", "cols", "values", "source_dual", "target_dual"))
    assert not solution.values.requires_grad
    dense = solution.to_sparse_matrix().to_dense()
    torch.testing.assert_close(dense.sum(1), torch.tensor(a, device=cuda), atol=4e-6, rtol=0)
    torch.testing.assert_close(dense.sum(0), torch.tensor(b, device=cuda), atol=4e-6, rtol=0)
    delta = x.double()[:, None] - y.double()[None]
    costs = {"l2^2": lambda: delta.square().sum(-1), "l1": lambda: delta.abs().sum(-1),
             "l2": lambda: delta.square().sum(-1).sqrt(), "linf": lambda: delta.abs().amax(-1)}[cost]()
    expected = ot.emd2(a, b, costs.cpu().numpy())
    assert result.objective == pytest.approx(expected, abs=4e-5, rel=1e-5)
    expected_stages = ["original"] if perturbation == "off" else ["perturbed", "original_final"]
    assert [stage.name for stage in result.cost_stages] == expected_stages
    violation = (solution.source_dual[:, None] + solution.target_dual[None] - costs).clamp_min(0)
    assert float(violation.norm() / (1 + costs.norm())) <= 1.1e-6
    assert set(result.metadata["transfers"]) <= {"initialization", "leaf_pot", "support_sparsification"}
    assert isinstance(solution.to_cpu().values, np.ndarray)
    assert solution.to_warm_start_state().x_prev.is_cuda


def test_gpu_auxiliary_iteration_limit_reaches_original_final(cuda):
    """
    CN: GPU-resident 的层级额度耗尽不抛错，并继续进入原成本最终阶段。
    EN: A GPU-resident level budget exhaustion does not raise and still reaches the final original-cost stage.
    """
    torch.manual_seed(47)
    x, y = torch.randn(32, 3, device=cuda), torch.randn(32, 3, device=cuda)
    result = hello_ot.solve(
        x,
        y,
        options=hello_ot.SolverOptions(
            gpu_resident=True,
            cost_perturbation="on",
            coarsest_size_threshold=8,
            assignment_topk=1,
            pricing_topk=1,
            verbose="off",
        ),
        max_iterations=1,
    )
    assert [stage.name for stage in result.cost_stages] == ["perturbed", "original_final"]
    auxiliary = [
        level.solve.summary
        for level in result.cost_stages[0].solve.levels
        if hasattr(level.solve, "summary")
    ]
    assert auxiliary
    assert all(summary.iterations == 1 for summary in auxiliary)
    assert len(result.cost_stages[-1].solve.levels[-1].solve.iterations) == 1


@pytest.mark.parametrize("force_restart", [False, True])
def test_gpu_auto_jump_probe_controls_full_restart(cuda, monkeypatch, force_restart):
    """
    CN: GPU-resident 的 auto 探测在低跳变时原地继续，在高跳变时记录探测轮并完整重启。
    EN: The GPU-resident auto probe continues in place for a small jump and records the probe before a full restart for a large jump.
    """
    from hello_ot.gpu_resident import solver

    original_refine = solver._refine
    original_perturb = solver._perturb
    perturbations = []

    def force_auto_trigger(*args, **kwargs):
        support, dual, level, trigger, jump, restart = original_refine(*args, **kwargs)
        detect = bool(args[8])
        if not detect:
            return support, dual, level, trigger, jump, restart
        summary = replace(
            level.solve.summary,
            stop_reason="cost_perturbation_activated",
            converged=False,
        )
        return support, dual, replace(level, solve=replace(level.solve, summary=summary)), True, None, False

    def force_jump_decision(before, after):
        return {
            "dual_feasibility_before_switch": float(before),
            "dual_feasibility_after_switch": float(after),
            "dual_feasibility_jump_ratio": 10.0 if force_restart else 1.0,
            "dual_feasibility_jump_threshold": 10.0,
            "restarted_from_perturbed_coarsest": force_restart,
        }

    def count_perturbation(*args, **kwargs):
        cost, metadata = original_perturb(*args, **kwargs)
        perturbations.append((metadata["sigma"], metadata["seed"]))
        return cost, metadata

    monkeypatch.setattr(solver, "_refine", force_auto_trigger)
    monkeypatch.setattr(solver, "dual_feasibility_jump_diagnostics", force_jump_decision)
    monkeypatch.setattr(solver, "_perturb", count_perturbation)
    torch.manual_seed(53)
    x, y = torch.randn(32, 3, device=cuda), torch.randn(32, 3, device=cuda)
    result = hello_ot.solve(
        x,
        y,
        options=hello_ot.SolverOptions(
            gpu_resident=True,
            cost_perturbation="auto",
            coarsest_size_threshold=8,
            assignment_topk=1,
            pricing_topk=1,
            verbose="off",
        ),
        max_iterations=4,
    )
    assert [stage.name for stage in result.cost_stages] == [
        "original",
        "perturbed",
        "original_final",
    ]
    perturbed = result.cost_stages[1]
    if force_restart:
        assert perturbed.solve.levels[0].solve.summary.stop_reason == "dual_feasibility_jump_restart"
        assert perturbed.solve.levels[0].solve.summary.iterations == 1
        assert perturbed.solve.levels[1].kind == "coarsest"
    else:
        assert all(level.kind == "refined" for level in perturbed.solve.levels)
        indices = [item.index for item in perturbed.solve.levels[0].solve.iterations]
        assert indices == list(range(len(indices)))
    assert perturbed.solve.levels[-1].level_index == 0
    metadata = result.metadata["cost_perturbation"]
    assert perturbations == [(metadata["sigma"], metadata["seed"])]
    assert metadata["dual_feasibility_jump_ratio"] == (10.0 if force_restart else 1.0)
    assert metadata["restarted_from_perturbed_coarsest"] is force_restart
    assert metadata["restart_trigger_level"] == (
        perturbed.solve.levels[0].level_index if force_restart else None
    )
    recorded_iterations = sum(
        len(level.solve.iterations)
        for stage in result.cost_stages
        for level in stage.solve.levels
        if hasattr(level.solve, "iterations")
    )
    assert result.metadata["total_refinement_iterations"] == recorded_iterations


@pytest.mark.parametrize("precision", ["fp32", "fp64", "finest_fp64"])
def test_precision_and_linf(cuda, precision):
    torch.manual_seed(24)
    x, y = torch.randn(19, 5, device=cuda), torch.randn(23, 5, device=cuda)
    result = hello_ot.solve(x, y, options=hello_ot.SolverOptions(gpu_resident=True,
        cost_precision=precision, stopping_norm="finest_linf", cost_perturbation="on",
        coarsest_size_threshold=8, verbose="off"))
    assert result.solve_stage.levels[-1].solve.iterations[-1].convergence.passed


def test_inplace_views_restored_on_exception(cuda, monkeypatch):
    from hello_ot.gpu_resident import solver
    torch.manual_seed(3)
    x, y = torch.randn(37, 5, device=cuda), torch.randn(43, 5, device=cuda)
    xx, yy = x.clone(), y.clone()

    def fail(cost, *args):
        assert cost.source.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        assert cost.target.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
        raise RuntimeError("injected leaf failure")

    monkeypatch.setattr(solver, "_leaf", fail)
    with pytest.raises(RuntimeError, match="injected leaf failure"):
        hello_ot.solve(x, y, options=hello_ot.SolverOptions(gpu_resident=True,
            consume_input_features=True, coarsest_size_threshold=8, verbose="off"))
    assert torch.equal(x, xx) and torch.equal(y, yy)


def test_default_cuda_input_uses_cpu_path(cuda, monkeypatch):
    import hello_ot.api
    captured = {}

    def solve(problem, config):
        captured["problem"] = problem
        assert not config.gpu_resident
        return "cpu path"

    monkeypatch.setattr(hello_ot.api, "_solve", solve)
    x = torch.randn(5, 3, device=cuda, dtype=torch.bfloat16, requires_grad=True)
    assert hello_ot.solve(x, x) == "cpu path"
    np.testing.assert_array_equal(captured["problem"].source_points, x.detach().float().cpu().numpy())


def test_gpu_input_validation(cuda):
    x = torch.randn(5, 3, device=cuda)
    options = hello_ot.SolverOptions(gpu_resident=True, consume_input_features=True, verbose="off")
    with pytest.raises(ValueError, match="disjoint storage"):
        hello_ot.solve(x, x, options=options)
    with pytest.raises(ValueError, match="contiguous FP32"):
        hello_ot.solve(x.double(), x.clone(), options=options)
    with pytest.raises(ValueError, match="same device"):
        hello_ot.solve(x, x.cpu(), options=options)


def test_gpu_pipeline_host_arrays_only_at_explicit_exceptions(cuda, monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode
    from hello_ot.gpu_resident.storage import Transfers

    allowed = False
    move = Transfers.move

    def tracked_move(self, value, device, reason):
        nonlocal allowed
        old, allowed = allowed, reason in {"leaf_pot", "support_sparsification"}
        try:
            return move(self, value, device, reason)
        finally:
            allowed = old

    class Guard(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func == torch.ops.aten._to_copy.default and args[0].is_cuda:
                device = kwargs.get("device", args[0].device)
                if torch.device(device).type == "cpu" and args[0].numel() > 1:
                    assert allowed, "unexpected GPU array export"
            return func(*args, **kwargs)

    monkeypatch.setattr(Transfers, "move", tracked_move)
    torch.manual_seed(23)
    x, y = torch.randn(35, 5, device=cuda), torch.randn(43, 5, device=cuda)
    with Guard():
        result = hello_ot.solve(x, y, options=hello_ot.SolverOptions(gpu_resident=True,
            consume_input_features=True, coarsest_size_threshold=8, profile_memory=True,
            record_trace=True, cost_perturbation="on", verbose="off"))
    assert result.metadata["memory_profile"]["lp_candidates"]
    assert result.peak_gpu_memory_mib > 0


@pytest.mark.parametrize("custom_stream", [False, True])
def test_cupdlpx_gpu_output_warm_start_and_continuation(cuda, custom_stream):
    from hello_ot._internal.lp.cupdlpx import CuPDLPxSolver
    c = torch.tensor([1., 2.], device=cuda, dtype=torch.float64)
    rhs = torch.tensor([1.], device=cuda, dtype=torch.float64)
    payload = dict(row_ptr=torch.tensor([0, 2], device=cuda, dtype=torch.int32),
        col_ind=torch.tensor([0, 1], device=cuda, dtype=torch.int32),
        values=torch.ones(2, device=cuda, dtype=torch.float64), c=c, rhs=rhs,
        lb=0., ub=float("inf"), m=1, n=2, n_eqs=1, device=cuda,
        variable_bound_mode="constant", matrix_value_mode="implicit_aty")

    def solve(**kw):
        params = kw.pop("params", {})
        stream = torch.cuda.Stream(device=cuda) if custom_stream else torch.cuda.current_stream(cuda)
        with torch.cuda.stream(stream):
            out = CuPDLPxSolver().solve(c, None, rhs, 0., float("inf"), 1,
                device_problem_data=payload, solver_params={"return_device": "cuda", **params}, **kw)
        torch.cuda.current_stream(cuda).wait_stream(stream)
        return out

    result = solve()
    assert result.success and result.x.is_cuda and result.y.is_cuda
    continuation = result.solver_diag["continuation_state"]
    for key in ("anchor_primal_unscaled", "anchor_dual_unscaled", "current_primal_unscaled", "current_dual_unscaled"):
        assert continuation[key].is_cuda
    saved = result.x.clone()
    gc.collect()
    torch.cuda.empty_cache()
    torch.testing.assert_close(result.x, saved, atol=0, rtol=0)
    resumed = solve(warm_start_primal=result.x, warm_start_dual=result.y,
                    params={"continuation_state": continuation})
    assert resumed.success and resumed.obj_val == pytest.approx(1., abs=1e-5)
    torch.testing.assert_close(result.x, saved, atol=0, rtol=0)


def test_actual_support_sparsification_cpu_exception(cuda, monkeypatch):
    from hello_ot.gpu_resident.cost import Cost, offsets
    from hello_ot.gpu_resident.storage import Transfers
    from hello_ot.gpu_resident.support import Support
    from hello_ot._internal.lp.cupdlpx import CuPDLPxSolver

    n = 6
    x, y = torch.randn(n, 2, device=cuda), torch.randn(n, 2, device=cuda)
    idx = torch.arange(n, device=cuda)
    cost = Cost(x, y, "l2^2", offsets(x), offsets(y), idx, idx)
    keys = torch.arange(n*n, device=cuda)
    values = torch.full((n*n,), 1/(n*n), device=cuda, dtype=torch.float64)
    support = Support(keys, values, torch.zeros(n*n, device=cuda, dtype=torch.bool), n, n)
    dual = torch.zeros(2*n, device=cuda, dtype=torch.float64)
    monkeypatch.setattr(CuPDLPxSolver, "solve", lambda *a, **kw: SimpleNamespace(success=True, x=values, y=dual))
    transfers = Transfers(cuda)
    mass = torch.full((n,), 1/n, device=cuda, dtype=torch.float64)
    config = hello_ot.SolverOptions(support_budget_factor=3, assignment_topk=1, pricing_topk=1)._to_internal_config(max_iterations=10, random_seed=1)
    support.solve_lp(cost, mass, mass, dual, config, "l2", transfers)
    assert transfers.records["support_sparsification"]["calls"] == 5
    assert torch.count_nonzero(support.values) <= 2*n
    dense = support.values.reshape(n, n)
    torch.testing.assert_close(dense.sum(0), mass, atol=1e-12, rtol=0)
    torch.testing.assert_close(dense.sum(1), mass, atol=1e-12, rtol=0)
