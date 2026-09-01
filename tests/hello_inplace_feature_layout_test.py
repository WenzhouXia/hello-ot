from __future__ import annotations

import numpy as np
import pytest
import torch

from hello_ot import Problem, SolverOptions, prewarm, solve
from hello_ot.hierarchy.plan import _hierarchy_global_reorder


def _layout_inputs() -> tuple[np.ndarray, ...]:
    source = np.arange(35, dtype=np.float32).reshape(7, 5)
    target = (100.0 + np.arange(30, dtype=np.float32)).reshape(6, 5)
    source_cost = np.arange(7, dtype=np.float64)
    target_cost = np.arange(6, dtype=np.float64)
    source_mass = np.full(7, 1.0 / 7.0, dtype=np.float64)
    target_mass = np.full(6, 1.0 / 6.0, dtype=np.float64)
    source_perm = np.asarray([3, 0, 6, 2, 5, 1, 4], dtype=np.int64)
    target_perm = np.asarray([4, 1, 5, 0, 3, 2], dtype=np.int64)
    return (
        source,
        target,
        source_cost,
        target_cost,
        source_mass,
        target_mass,
        source_perm,
        target_perm,
    )


def test_inplace_hierarchy_layout_restores_features_after_success() -> None:
    inputs = _layout_inputs()
    source, target, *_values, source_perm, target_perm = inputs
    source_original = source.copy()
    target_original = target.copy()
    with _hierarchy_global_reorder(
        source_F_full=source,
        target_G_full=target,
        source_cost_vec_full=inputs[2],
        target_cost_vec_full=inputs[3],
        source_mass_raw=inputs[4],
        target_mass_raw=inputs[5],
        source_perm=source_perm,
        target_perm=target_perm,
        consume_input_features=True,
    ) as reordered:
        assert reordered[0] is source
        assert reordered[1] is target
        np.testing.assert_array_equal(source, source_original[source_perm])
        np.testing.assert_array_equal(target, target_original[target_perm])
    np.testing.assert_array_equal(source, source_original)
    np.testing.assert_array_equal(target, target_original)


def test_inplace_hierarchy_layout_restores_features_after_exception() -> None:
    inputs = _layout_inputs()
    source, target, *_values, source_perm, target_perm = inputs
    source_original = source.copy()
    target_original = target.copy()
    with pytest.raises(RuntimeError, match="intentional"):
        with _hierarchy_global_reorder(
            source_F_full=source,
            target_G_full=target,
            source_cost_vec_full=inputs[2],
            target_cost_vec_full=inputs[3],
            source_mass_raw=inputs[4],
            target_mass_raw=inputs[5],
            source_perm=source_perm,
            target_perm=target_perm,
            consume_input_features=True,
        ):
            raise RuntimeError("intentional")
    np.testing.assert_array_equal(source, source_original)
    np.testing.assert_array_equal(target, target_original)


def test_inplace_hierarchy_layout_rejects_readonly_features() -> None:
    inputs = _layout_inputs()
    source, target, *_values, source_perm, target_perm = inputs
    source.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        with _hierarchy_global_reorder(
            source_F_full=source,
            target_G_full=target,
            source_cost_vec_full=inputs[2],
            target_cost_vec_full=inputs[3],
            source_mass_raw=inputs[4],
            target_mass_raw=inputs[5],
            source_perm=source_perm,
            target_perm=target_perm,
            consume_input_features=True,
        ):
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_inplace_hello_matches_materialized_and_restores_inputs() -> None:
    rng = np.random.default_rng(23)
    source = rng.standard_normal((2048, 16), dtype=np.float32)
    target = rng.standard_normal((2048, 16), dtype=np.float32)
    source_original = source.copy()
    target_original = target.copy()
    materialized_options = SolverOptions(
        split_count=4,
        coarsest_size_threshold=1024,
        assignment_topk=16,
        pricing_topk=2,
    )
    inplace_options = SolverOptions(
        split_count=4,
        coarsest_size_threshold=1024,
        assignment_topk=16,
        pricing_topk=2,
        consume_input_features=True,
    )
    materialized_problem = Problem(source.copy(), target.copy(), cost_type="l2^2")
    inplace_problem = Problem(source, target, cost_type="l2^2")
    prewarm(inplace_problem, inplace_options)
    materialized = solve(
        materialized_problem,
        max_iterations=20,
        random_seed=23,
        options=materialized_options,
    )
    inplace = solve(
        inplace_problem,
        max_iterations=20,
        random_seed=23,
        options=inplace_options,
    )
    np.testing.assert_array_equal(source, source_original)
    np.testing.assert_array_equal(target, target_original)
    assert inplace.metadata["consume_input_features"] is True
    np.testing.assert_allclose(inplace.objective, materialized.objective, rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(inplace.solution.rows, materialized.solution.rows)
    np.testing.assert_array_equal(inplace.solution.cols, materialized.solution.cols)
    np.testing.assert_allclose(inplace.solution.values, materialized.solution.values, rtol=1e-7, atol=1e-10)
