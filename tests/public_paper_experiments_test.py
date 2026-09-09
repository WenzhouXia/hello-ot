from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from export_experiments.exactness_verification.metrics import evaluate_sparse_coupling
from export_experiments.exactness_verification.run import build_cases as build_exactness_cases
from export_experiments.main_scaling.data import load_representative_features
from export_experiments.main_scaling.run import build_cases as build_scaling_cases
from export_experiments.parameter_sensitivity.run import build_cases as build_sensitivity_cases
from export_experiments.synthetic import make_correlated_brenier_problem


def test_representative_matrices_match_the_public_protocol() -> None:
    scaling = build_scaling_cases()
    exactness = build_exactness_cases()
    sensitivity = build_sensitivity_cases()

    assert len(scaling) == 5 * 4
    assert {case[0] for case in scaling} == {2**14, 2**15, 2**16, 2**17, 2**18}
    assert {case[1] for case in scaling} == {4, 32, 256, 2048}
    assert {case[2] for case in scaling} == {42}
    assert len(exactness) == 1 * 3 * 4
    assert len(sensitivity) == 1 * 4 * 9


def test_local_feature_artifact_is_memory_mapped_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.zeros((8, 4), dtype=np.float32), allow_pickle=False)

    array = load_representative_features(feature_file=path, min_rows=8, min_dimension=4)

    assert array.shape == (8, 4)
    assert array.dtype == np.float32


def test_correlated_brenier_ground_truth_has_perfect_metrics() -> None:
    source, target, ground_truth_cols, objective = make_correlated_brenier_problem(32, 8, 42)
    assert source.shape == target.shape == (32, 8)
    rows = np.arange(32, dtype=np.int64)
    coupling = sparse.coo_matrix(
        (np.full(32, 1.0 / 32.0), (rows, ground_truth_cols)),
        shape=(32, 32),
    )

    metrics = evaluate_sparse_coupling(
        coupling,
        ground_truth_cols=ground_truth_cols,
        algorithm_objective=objective,
        ground_truth_objective=objective,
        support_threshold=1.0e-8,
    )

    assert metrics["relative_objective_error"] == 0.0
    assert metrics["support_recall"] == 1.0
    assert metrics["support_precision"] == 1.0
    assert metrics["row_argmax_recall"] == 1.0
    assert metrics["matching_1to1_recall"] == 1.0
    assert metrics["primal_frobenius_relative_error"] == 0.0


def test_public_cases_reject_other_seeds() -> None:
    for build in (build_scaling_cases, build_sensitivity_cases, build_exactness_cases):
        with pytest.raises(ValueError, match="seed=42"):
            build(seeds=[43])


def test_features_cannot_be_sliced_to_a_lower_dimension(tmp_path: Path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.zeros((8, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        load_representative_features(feature_file=path, min_rows=8, min_dimension=4)
