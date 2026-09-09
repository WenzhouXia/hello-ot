from __future__ import annotations

import inspect
import numpy as np
import pytest
import scipy.sparse as sp
import torch

import hello_ot


def test_variants_public_surface() -> None:
    """
    CN: 检查变体公开入口与返回类型是否在 hello_ot 根包暴露。
    EN: Check that variant entry points and result types are exposed at hello_ot root.
    """
    expected = [
        "solve_semidiscrete",
        "solve_unbalanced",
        "solve_gromov",
        "SemiDiscreteResult",
        "UnbalancedResult",
        "GromovResult",
    ]
    for name in expected:
        assert hasattr(hello_ot, name), f"hello_ot should export {name}"
        assert name in hello_ot.__all__, f"{name} should be in hello_ot.__all__"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HELLO is CUDA-only")
def test_solve_semidiscrete_basic() -> None:
    """
    CN: 检查 semi-discrete OT 基础求解流程与质量严格校验。
    EN: Check semi-discrete OT basic solve flow and strict mass validation.
    """
    rng = np.random.default_rng(42)
    target_points = rng.normal(size=(20, 3)).astype(np.float32)

    # CN: 校验未归一化质量直接报错
    # EN: Unnormalized mass must raise ValueError
    with pytest.raises(ValueError, match="sum to 1"):
        hello_ot.solve_semidiscrete(target_points, target_mass=np.ones(20) * 2.0)

    res = hello_ot.solve_semidiscrete(
        target_points,
        num_repeats=2,
        source_sample_count=40,
        options=hello_ot.SolverOptions(coarsest_size_threshold=16),
    )
    assert isinstance(res, hello_ot.SemiDiscreteResult)
    assert res.target_dual.shape == (20,)
    assert len(res.records) == 2
    assert np.isclose(float(np.mean(res.target_dual)), 0.0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HELLO is CUDA-only")
def test_solve_unbalanced_cold() -> None:
    """
    CN: 检查 KL-UOT FCFW 冷启动求解流程及严格边际质量语义。
    EN: Check KL-UOT FCFW cold-start solving and strict marginal mass semantics.
    """
    rng = np.random.default_rng(42)
    src_pts = rng.normal(size=(16, 2)).astype(np.float32)
    tgt_pts = rng.normal(size=(20, 2)).astype(np.float32)

    src_mass = rng.uniform(0.5, 1.5, size=16)
    tgt_mass = rng.uniform(0.5, 1.5, size=20)

    # CN: 校验负边际质量报错
    # EN: Non-positive mass must raise ValueError
    invalid_mass = src_mass.copy()
    invalid_mass[0] = -1.0
    with pytest.raises(ValueError, match="positive and finite"):
        hello_ot.solve_unbalanced(src_pts, tgt_pts, source_mass=invalid_mass, target_mass=tgt_mass)

    # CN: 1. 冷启动求解
    # EN: 1. Cold-start solve
    res_cold = hello_ot.solve_unbalanced(
        src_pts,
        tgt_pts,
        source_mass=src_mass,
        target_mass=tgt_mass,
        rho_source=1.0,
        rho_target=1.0,
        max_iterations=4,
        initialization=None,
        options=hello_ot.SolverOptions(coarsest_size_threshold=8),
    )
    assert isinstance(res_cold, hello_ot.UnbalancedResult)
    assert res_cold.objective > 0.0
    assert res_cold.dual_objective <= res_cold.objective + 1e-4
    assert res_cold.transported_mass > 0.0
    assert sp.issparse(res_cold.solution.to_sparse_matrix())
    assert res_cold.solution.shape == (16, 20)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="HELLO is CUDA-only")
def test_solve_gromov_asymmetric_dimensions() -> None:
    """
    CN: 检查低秩平方欧氏 Gromov-Wasserstein 支持跨模态不同维度及质量严格校验。
    EN: Check low-rank squared-Euclidean GW supporting cross-modal dimensions and strict mass validation.
    """
    rng = np.random.default_rng(42)
    src_pts = rng.normal(size=(20, 3)).astype(np.float32)
    tgt_pts = rng.normal(size=(24, 2)).astype(np.float32)

    # CN: 校验未归一化概率分布报错
    # EN: Unnormalized probability masses must raise ValueError
    with pytest.raises(ValueError, match="sum to 1"):
        hello_ot.solve_gromov(src_pts, tgt_pts, source_mass=np.ones(20))

    res = hello_ot.solve_gromov(
        src_pts,
        tgt_pts,
        max_iterations=3,
        options=hello_ot.SolverOptions(coarsest_size_threshold=8),
    )
    assert isinstance(res, hello_ot.GromovResult)
    assert res.objective > 0.0
    assert res.solution.shape == (20, 24)
    sparse_p = res.solution.to_sparse_matrix()
    assert sp.issparse(sparse_p)
    assert np.isclose(float(sparse_p.sum()), 1.0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_solve_unbalanced_ott():
    """CN: 验证可选 OTT 初始化。EN: Validate optional OTT initialization."""
    pytest.importorskip("ott")
    rng = np.random.default_rng(42)
    src_pts = rng.normal(size=(16, 2)).astype(np.float32)
    tgt_pts = rng.normal(size=(20, 2)).astype(np.float32)
    src_mass = rng.uniform(0.5, 1.5, size=16)
    tgt_mass = rng.uniform(0.5, 1.5, size=20)
    # CN: 2. OTT-JAX 预热求解
    # EN: 2. OTT-JAX warm-start solve
    res_ott = hello_ot.solve_unbalanced(
        src_pts,
        tgt_pts,
        source_mass=src_mass,
        target_mass=tgt_mass,
        rho_source=1.0,
        rho_target=1.0,
        max_iterations=4,
        initialization="ott_sinkhorn",
        options=hello_ot.SolverOptions(coarsest_size_threshold=8),
    )
    assert isinstance(res_ott, hello_ot.UnbalancedResult)
    assert res_ott.objective > 0.0
    assert res_ott.dual_objective <= res_ott.objective + 1e-4
    assert res_ott.solution.shape == (16, 20)
