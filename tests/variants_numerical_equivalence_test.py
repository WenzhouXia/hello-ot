from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp
import torch

import hello_ot

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "variants_baseline_32k.npz"


@pytest.fixture(scope="module")
def baseline_data() -> dict[str, np.ndarray]:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Baseline fixture not found at {FIXTURE_PATH}")
    return dict(np.load(FIXTURE_PATH))


@pytest.fixture(scope="module")
def dataset_32k() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, d = 32768, 32
    rng = np.random.default_rng(20260902)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=(n, d)).astype(np.float32)
    a = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    b = rng.uniform(0.5, 1.5, size=n).astype(np.float64)
    na = a / a.sum()
    nb = b / b.sum()
    return x, y, a, b, na, nb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HELLO is CUDA-only")
def test_sdot_32k_numerical_equivalence(baseline_data, dataset_32k) -> None:
    """
    CN: 比较 32k SDOT 历史对偶势，rtol=0.05、atol=0.2。
    EN: Compare historical 32k SDOT duals with rtol=0.05 and atol=0.2.
    """
    x, y, a, b, na, nb = dataset_32k
    n = int(y.shape[0])
    res = hello_ot.solve_semidiscrete(
        y,
        num_repeats=2,
        source_sample_count=n,
        target_mass=nb,
        random_seed=42,
    )
    np.testing.assert_allclose(
        res.target_dual,
        baseline_data["sdot_target_dual"],
        rtol=5e-2,
        atol=0.2,
        err_msg="SDOT 32k target dual mismatch",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HELLO is CUDA-only")
def test_gw_32k_numerical_equivalence(baseline_data, dataset_32k) -> None:
    """
    CN: 检查两次 GW 外层迭代后的目标、质量与支撑规模回归，不要求支撑逐项相等。
    EN: Check objective, mass and support-size regression after two GW outer iterations, not identical supports.
    """
    x, y, a, b, na, nb = dataset_32k
    res = hello_ot.solve_gromov(
        x,
        y,
        source_mass=na,
        target_mass=nb,
        max_iterations=2,
        tolerance=1e-12,
    )
    np.testing.assert_allclose(
        res.objective,
        float(baseline_data["gw_objective"][0]),
        rtol=5e-5,
        err_msg="GW 32k objective mismatch",
    )
    coo = res.solution.to_sparse_matrix().tocoo()
    # CN: 校验总质量与边际质量守恒
    # EN: Verify total mass and marginal mass conservation
    assert np.isclose(float(coo.data.sum()), 1.0, atol=1e-5), "GW 32k total mass must equal 1"
    row_sums = np.bincount(coo.row, weights=coo.data, minlength=int(x.shape[0]))
    col_sums = np.bincount(coo.col, weights=coo.data, minlength=int(y.shape[0]))
    np.testing.assert_allclose(row_sums, na, rtol=1e-4, atol=1e-6, err_msg="GW 32k source marginal mismatch")
    np.testing.assert_allclose(col_sums, nb, rtol=1e-4, atol=1e-6, err_msg="GW 32k target marginal mismatch")
    # CN: 校验非零元规模处于同一稀疏度量级 (< 1% 偏差)
    # EN: Verify non-zero support size is in the same sparsity scale (< 1% diff)
    assert abs(coo.nnz - int(baseline_data["gw_rows"].size)) / float(baseline_data["gw_rows"].size) < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HELLO is CUDA-only")
def test_uot_32k_numerical_equivalence(baseline_data, dataset_32k) -> None:
    """
    CN: 检查两次 UOT 外层迭代后的历史数值回归，不作为最终收敛证明。
    EN: Check historical numerics after two UOT outer iterations, not final convergence.
    """
    x, y, a, b, na, nb = dataset_32k
    res = hello_ot.solve_unbalanced(
        x,
        y,
        source_mass=a,
        target_mass=b,
        rho_source=1.0,
        rho_target=1.0,
        max_iterations=2,
        initialization=None,
    )
    np.testing.assert_allclose(
        res.objective,
        float(baseline_data["uot_objective"][0]),
        rtol=1e-5,
        err_msg="UOT 32k primal objective mismatch",
    )
    np.testing.assert_allclose(
        res.dual_objective,
        float(baseline_data["uot_dual_objective"][0]),
        rtol=1e-5,
        err_msg="UOT 32k dual objective mismatch",
    )
    np.testing.assert_allclose(
        res.relative_primal_dual_gap,
        float(baseline_data["uot_relative_gap"][0]),
        rtol=1e-4,
        err_msg="UOT 32k relative duality gap mismatch",
    )
    if "uot_rows" in baseline_data:
        coo = res.solution.to_sparse_matrix().tocoo()
        np.testing.assert_array_equal(coo.row, baseline_data["uot_rows"], err_msg="UOT 32k row indices mismatch")
        np.testing.assert_array_equal(coo.col, baseline_data["uot_cols"], err_msg="UOT 32k col indices mismatch")
        np.testing.assert_allclose(coo.data, baseline_data["uot_data"], rtol=1e-5, atol=1e-7, err_msg="UOT 32k coupling data mismatch")
