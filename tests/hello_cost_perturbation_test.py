from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import ot
import pytest
import torch

import hello_ot
from hello_ot.refinement.reentry import ReentryDetector


def test_reentry_uses_only_previous_pruning_and_one_lp_probation():
    """
    CN: 只认上一轮删除的边；本轮回流的怀疑留给下一次 LP 检查。
    EN: Only previous-round prunes count; re-entry leaves probation for the next LP check.
    """
    detector = ReentryDetector()
    detector.observe(SimpleNamespace(added_keys=np.array([1]), pruned_keys=np.array([9])))
    assert not detector.should_trigger()
    detector.observe(SimpleNamespace(added_keys=np.array([2]), pruned_keys=np.array([8])))
    detector.observe(SimpleNamespace(added_keys=np.array([9]), pruned_keys=np.array([7])))
    assert not detector.should_trigger()
    detector.observe(SimpleNamespace(added_keys=torch.tensor([7, 12]), pruned_keys=None))
    assert detector.should_trigger()
    assert detector.reentry_count == 1
    detector.observe(SimpleNamespace(added_keys=np.array([7]), pruned_keys=None))
    assert not detector.should_trigger()


@pytest.mark.parametrize("cost,metric", [("l2^2", "sqeuclidean"), ("l1", "cityblock"),
                                         ("l2", "euclidean"), ("linf", "chebyshev")])
@pytest.mark.parametrize("policy", ["off", "on", "auto"])
@pytest.mark.parametrize("backend", ["torch", pytest.param("native", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires native CUDA runtime"))])
def test_modes_return_original_problem_solution(cost, metric, policy, backend):
    """
    CN: 三种模式最终满足原成本目标、边际约束，并完整记录阶段。
    EN: All modes return the original-cost objective and marginals with complete stage records.
    """
    rng = np.random.default_rng(11)
    source, target = rng.normal(size=(16, 3)), rng.normal(size=(16, 3))
    options = hello_ot.SolverOptions(backend=backend, torch_device="cpu", cost_perturbation=policy,
                                    coarsest_size_threshold=4, assignment_topk=2, pricing_topk=1)
    result = hello_ot.solve(source, target, cost=cost, options=options, max_iterations=12)
    costs = ot.dist(source, target, metric=metric)
    masses = np.full(16, 1 / 16)
    assert abs(result.objective - ot.emd2(masses, masses, costs)) < 4e-6
    solution = result.solution
    np.testing.assert_allclose(np.bincount(solution.rows, weights=solution.values, minlength=16), masses, atol=2e-6)
    np.testing.assert_allclose(np.bincount(solution.cols, weights=solution.values, minlength=16), masses, atol=2e-6)
    assert result.solve_stage.levels[-1].solve.summary.converged
    assert sum(stage.solve.wall_time for stage in result.cost_stages) <= result.total_wall_time
    if policy == "on":
        assert [stage.name for stage in result.cost_stages] == ["perturbed", "original_final"]
        assert len(result.solve_stage.levels) == 1
    if policy == "off":
        assert not result.metadata["cost_perturbation"]["activated"]


@pytest.mark.parametrize("cost", ["l2^2", "l1", "l2", "linf"])
def test_auto_switches_in_flight_without_restarting_hierarchy(monkeypatch, cost):
    """
    CN: 强制检测事件以独立验证切换控制；真正的回流规则由检测器测试验证。
    EN: Force a detector event to independently test switching; detector tests cover actual re-entry rules.
    """
    monkeypatch.setattr(ReentryDetector, "should_trigger", lambda self: True)
    rng = np.random.default_rng(7)
    source = rng.normal(size=(32, 2)).astype(np.float32)
    target = rng.normal(size=(32, 2)).astype(np.float32)
    original_source, original_target = source.copy(), target.copy()
    options = hello_ot.SolverOptions(backend="torch", torch_device="cpu", coarsest_size_threshold=8,
                                    assignment_topk=1, pricing_topk=1, consume_input_features=True)
    result = hello_ot.solve(source, target, cost=cost, options=options, max_iterations=12)
    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(target, original_target)
    assert [stage.name for stage in result.cost_stages] == ["original", "perturbed", "original_final"]
    initial, auxiliary, final = result.cost_stages
    assert initial.solve.levels[0].kind == "coarsest"
    assert all(level.kind == "refined" for level in auxiliary.solve.levels)
    trigger = initial.solve.levels[-1]
    assert trigger.solve.summary.stop_reason == "cost_perturbation_requested"
    assert trigger.solve.iterations[-1].index == result.metadata["cost_perturbation"]["trigger_iteration"]
    assert auxiliary.solve.levels[0].solve.iterations[0].index == 1
    assert auxiliary.solve.levels[0].level_index == trigger.level_index
    assert len(final.solve.levels) == 1
    baseline = hello_ot.solve(source, target, cost=cost, options=replace(options, cost_perturbation="off"), max_iterations=12)
    assert abs(result.objective - baseline.objective) < 4e-6


@pytest.mark.parametrize("cost", ["l2^2", "l1", "l2", "linf"])
def test_zero_sample_mean_errors_and_restores_consumed_inputs(cost):
    """
    CN: 零采样均值显式失败；异常退出也恢复调用方输入。
    EN: A zero sampled mean fails explicitly and restores caller-owned inputs on exception.
    """
    source = np.zeros((8, 2), dtype=np.float32)
    target = source.copy()
    options = hello_ot.SolverOptions(backend="torch", torch_device="cpu", cost_perturbation="on",
                                    consume_input_features=True)
    with pytest.raises(ValueError, match="positive sampled cost mean"):
        hello_ot.solve(source, target, cost=cost, options=options)
    np.testing.assert_array_equal(source, 0)
    np.testing.assert_array_equal(target, 0)
    result = hello_ot.solve(source, target, cost=cost, options=replace(options, cost_perturbation="auto"))
    assert not result.metadata["cost_perturbation"]["activated"]


def test_numpy_and_torch_noise_match_for_global_indices_and_slices():
    """
    CN: 哈希不受局部层编号和 FP32 索引精度限制影响。
    EN: Hashes are independent of local level numbering and FP32 index precision limits.
    """
    from hello_ot.kernels.norm_cost_scan import _apply_metric_pair_perturbation
    from hello_ot.cost import HelloCostContext

    perturbation = {"mode": "full_cost_perturb", "noise": "index_hash", "sigma": 0.017,
                    "seed": 123, "source_global_index": np.array([7, 2**25 + 1, 41]),
                    "target_global_index": np.array([99, 2**26 + 3, 3])}
    rows, cols = np.array([1, 2]), np.array([2, 1])
    expected = _apply_metric_pair_perturbation(np.zeros(2), rows, cols, perturbation)
    actual = _apply_metric_pair_perturbation(torch.zeros(2, dtype=torch.float64), rows, cols, perturbation)
    np.testing.assert_array_equal(actual.numpy(), expected)
    context = HelloCostContext("norm_cost", "l1", perturbation=perturbation)
    sliced = context.sliced(slice(1, 3), slice(1, 3))
    np.testing.assert_array_equal(
        _apply_metric_pair_perturbation(np.zeros(2), rows - 1, cols - 1, sliced.perturbation), expected)
    assert np.all((expected >= 0) & (expected <= perturbation["sigma"]))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))])
def test_fp32_hash_remainder_boundaries_match_numpy(device):
    """
    CN: 定向覆盖整除与整数边界，验证正式 NumPy/Torch 成本逐位一致。
    EN: Target divisibility boundaries and verify bitwise equality of production NumPy/Torch costs.
    """
    from hello_ot.kernels.norm_cost_scan import _apply_metric_pair_perturbation, _index_hash_parameters
    pairs = []
    seed = 558333475
    for p, (a, b, c) in zip((2003, 2011), _index_hash_parameters(seed)):
        for s in range(p):
            if (b+s) % p:
                for remainder in (0, p-1):
                    pairs.append((s, ((remainder-a*s-c)*pow((b+s) % p, -1, p)) % p))
    indices = np.asarray(pairs, dtype=np.int64)
    perturbation = {"mode": "full_cost_perturb", "noise": "index_hash", "sigma": .5318021687458532,
                    "seed": seed, "source_global_index": indices[:, 0], "target_global_index": indices[:, 1]}
    positions = np.arange(len(indices))
    base = np.full(len(indices), 3.0)
    expected = _apply_metric_pair_perturbation(base, positions, positions, perturbation)
    actual = _apply_metric_pair_perturbation(torch.as_tensor(base, device=device), positions, positions, perturbation)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native CUDA scans")
def test_fp32_native_hash_boundary_costs_match_exactly():
    """
    CN: 完整扫描在 FP32 边界样本上与 LP 成本逐位一致。
    EN: Full scans match LP costs bitwise on FP32 boundary samples.
    """
    from hello_ot.kernels.norm_cost_scan import (
        _require_norm_cost_scan_ext, _apply_metric_pair_perturbation, _index_hash_parameters,
    )
    seed = 558333475
    p = 2011
    a, b, c = _index_hash_parameters(seed)[1]
    src = np.arange(32, dtype=np.int64)
    tgt = np.asarray([((-a*int(s)-c)*pow((b+int(s)) % p, -1, p)) % p for s in src])
    params = {"mode": "full_cost_perturb", "noise": "index_hash", "sigma": .5318021687458532,
              "seed": seed, "source_global_index": src, "target_global_index": tgt}
    rows, cols = np.indices((32,32))
    expected = _apply_metric_pair_perturbation(np.zeros(1024), rows.ravel(), cols.ravel(), params).reshape(32,32)
    ext = _require_norm_cost_scan_ext()
    points = torch.zeros((32,4), device="cuda", dtype=torch.float32)
    dual = torch.zeros(32, device="cuda", dtype=torch.float64)
    coeff = torch.tensor(_index_hash_parameters(seed), dtype=torch.int64).reshape(-1)
    for reverse in (False, True):
        ix, iy = (tgt, src) if reverse else (src, tgt)
        out = ext.fused_gcost_bidir_certificate(points, points, dual, dual, 2, 0,
            torch.as_tensor(ix, device="cuda"), torch.as_tensor(iy, device="cuda"),
            params["sigma"], seed, 1, not reverse, coeff)
        scores = -expected.T if reverse else -expected
        np.testing.assert_array_equal(out[0].cpu().numpy(), np.sort(scores, axis=1)[:, -2:][:, ::-1])
        assert out[7].item() == float(expected.max())
        np.testing.assert_allclose(out[5].item(), np.square(expected).sum(), rtol=1e-14, atol=0)


def test_auxiliary_iteration_limit_still_enters_original_stage():
    """
    CN: 辅助阶段未收敛仍能转入原问题，且各阶段独立遵守额度。
    EN: An unconverged auxiliary stage still transfers to the original problem, with independent budgets.
    """
    rng = np.random.default_rng(7)
    result = hello_ot.solve(rng.normal(size=(32, 2)), rng.normal(size=(32, 2)), max_iterations=1,
        options=hello_ot.SolverOptions(backend="torch", torch_device="cpu", cost_perturbation="on",
                                      coarsest_size_threshold=8, assignment_topk=1, pricing_topk=1))
    assert [stage.name for stage in result.cost_stages] == ["perturbed", "original_final"]
    summaries = [level.solve.summary for level in result.cost_stages[0].solve.levels if level.kind == "refined"]
    assert any(not summary.converged for summary in summaries)
    assert all(summary.iterations == 1 for summary in summaries)
    assert len(result.solve_stage.levels[-1].solve.iterations) == 1


def test_variants_explicitly_disable_perturbation():
    """
    CN: variants 的内部求解保持 off，不继承标准 API 的 auto 默认值。
    EN: Variant inner solves remain off instead of inheriting the standard API's auto default.
    """
    from hello_ot.variants._balanced import algorithm_config

    assert hello_ot.SolverOptions().cost_perturbation == "auto"
    assert algorithm_config(None).cost_perturbation == "off"
    assert algorithm_config(None).verbose == "off"
    assert algorithm_config(hello_ot.SolverOptions(verbose="detailed")).verbose == "off"


@pytest.mark.parametrize("cost", ["l2^2", "l1", "l2", "linf"])
def test_actual_reentry_triggers_at_round_three_and_respects_limit(cost, capsys):
    """
    CN: 紧支撑预算产生真实边回流；两轮额度不触发，第三轮触发并只切换一次。
    EN: A tight support budget creates real re-entry; two rounds cannot trigger, and round three switches once.
    """
    rng = np.random.default_rng(7)
    source, target = rng.normal(size=(48, 2)), rng.normal(size=(48, 2))
    options = hello_ot.SolverOptions(backend="torch", torch_device="cpu", coarsest_size_threshold=8,
                                    assignment_topk=1, pricing_topk=1, support_budget_factor=0.75)
    limited = hello_ot.solve(source, target, cost=cost, options=options, max_iterations=2)
    assert not limited.metadata["cost_perturbation"]["activated"]
    result = hello_ot.solve(source, target, cost=cost, options=options, max_iterations=8)
    metadata = result.metadata["cost_perturbation"]
    assert metadata["activated"]
    assert metadata["trigger_iteration"] == 3
    assert metadata["reentry_count"] > 0
    assert [stage.name for stage in result.cost_stages] == ["original", "perturbed", "original_final"]
    assert result.metadata["total_refinement_iterations"] == sum(
        len(level.solve.iterations) for stage in result.cost_stages for level in stage.solve.levels
        if level.kind == "refined"
    )
    progress = capsys.readouterr().err
    assert "cost perturbation activated |" in progress
    assert "perturbed stage completed; switching back to original cost" in progress
    assert "[original] level=0 kind=final-refinement" in progress


def test_converged_certificate_precedes_trigger(monkeypatch):
    """
    CN: 即使处于观察期，下一轮已经收敛也不得触发扰动。
    EN: A converged next LP must win over a pending probation trigger.
    """
    def unexpected_check(self):
        raise AssertionError("detector consulted after convergence")

    monkeypatch.setattr(ReentryDetector, "should_trigger", unexpected_check)
    source = np.arange(8, dtype=np.float32).reshape(-1, 1)
    result = hello_ot.solve(source, source.copy(), options=hello_ot.SolverOptions(
        backend="torch", torch_device="cpu", coarsest_size_threshold=2, assignment_topk=8))
    assert not result.metadata["cost_perturbation"]["activated"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native CUDA scans")
@pytest.mark.parametrize("cost", ["l1", "l2", "linf"])
def test_native_perturbed_assignment_and_certificate_share_sparse_costs(cost):
    """
    CN: CUDA 两个扫描方向与证书必须使用和稀疏边相同的扰动成本。
    EN: Both CUDA scan directions and the certificate must use the same perturbed costs as sparse edges.
    """
    from hello_ot.kernels.norm_cost_scan import (
        _apply_metric_pair_perturbation, _fused_metric_kmin, _metric_cost_matrix,
        check_metric_dual_feasibility,
    )

    rng = np.random.default_rng(8)
    source, target = rng.normal(size=(13, 3)), rng.normal(size=(11, 3))
    source_ids, target_ids = rng.permutation(13) + 101, rng.permutation(11) + 203
    perturbation = {"mode": "full_cost_perturb", "noise": "index_hash", "sigma": 0.173,
                    "seed": 51, "source_global_index": source_ids, "target_global_index": target_ids}
    rows, cols = np.indices((13, 11))
    costs = _apply_metric_pair_perturbation(
        _metric_cost_matrix(source, target, cost).reshape(-1), rows.reshape(-1), cols.reshape(-1),
        perturbation).reshape(13, 11)
    u, v = rng.normal(size=13) + 2, rng.normal(size=11) + 2
    for forward in (True, False):
        values, _ = _fused_metric_kmin(
            query_points=source if forward else target, database_points=target if forward else source,
            known_dual=v if forward else u, cost_type=cost, k=2, query_is_source=forward,
            query_global_index=source_ids if forward else target_ids,
            database_global_index=target_ids if forward else source_ids, perturbation=perturbation)
        expected = np.sort(costs - v if forward else costs.T - u, axis=1)[:, :2]
        np.testing.assert_allclose(values.cpu().numpy(), expected, atol=2e-6, rtol=1e-6)
    certificate, _ = check_metric_dual_feasibility(
        source_points=source, target_points=target, source_dual=u, target_dual=v, cost_type=cost,
        perturbation=perturbation, source_global_index=source_ids, target_global_index=target_ids)
    expected = np.square(np.maximum(u[:, None] + v[None, :] - costs, 0)).sum()
    np.testing.assert_allclose(certificate.l2_numerator_sq.cpu().item(), expected, rtol=2e-6)
