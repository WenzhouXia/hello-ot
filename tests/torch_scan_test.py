from __future__ import annotations

import numpy as np
import pytest
import torch

from hello_ot.kernels.torch_scan import (
    bidirectional_violation_scan,
    complete_from_candidate,
    directional_assignment,
)
from hello_ot.state import WarmStartState


def _empty_state(source_dual: np.ndarray, target_dual: np.ndarray) -> WarmStartState:
    return WarmStartState(
        rows=np.empty(0, dtype=np.int32),
        cols=np.empty(0, dtype=np.int32),
        x_prev=np.empty(0, dtype=np.float64),
        dual_uv=np.concatenate((source_dual, target_dual)),
        n_source=int(source_dual.size),
        n_target=int(target_dual.size),
    )


def test_l2_squared_directional_assignment_and_completion_match_dense() -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, -1.0]], dtype=np.float32)
    target = np.array([[0.25, 0.0], [1.5, 0.25]], dtype=np.float32)
    source_offset = np.sum(source.astype(np.float64) ** 2, axis=1)
    target_offset = np.sum(target.astype(np.float64) ** 2, axis=1)
    target_dual = np.array([0.2, -0.1], dtype=np.float64)
    state = _empty_state(np.zeros(source.shape[0]), target_dual)
    assigned, _stats, profile, candidate = directional_assignment(
        state=state,
        source_points=source,
        target_points=target,
        source_offset=source_offset,
        target_offset=target_offset,
        score_family="inner_product",
        cost_type="lowrank",
        dot_scale=2.0,
        known_side="target",
        topk=1,
        device="cpu",
        max_tile_bytes=64,
    )
    dense_cost = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    expected_cols = np.argmax(target_dual[None, :] - dense_cost, axis=1)
    np.testing.assert_array_equal(assigned.cols.numpy(), expected_cols)
    completed, completion_profile = complete_from_candidate(
        state=assigned,
        candidate=candidate,
        device="cpu",
    )
    expected_source_dual = np.min(dense_cost - target_dual[None, :], axis=1)
    np.testing.assert_allclose(completed.dual_uv[: source.shape[0]].numpy(), expected_source_dual, atol=1e-12)
    assert profile["max_tile_elements"] < source.shape[0] * target.shape[0]
    assert completion_profile["backend"] == "torch_blockwise"


@pytest.mark.parametrize("cost_type", ["l1", "l2", "linf"])
def test_norm_violation_scan_matches_dense_certificate(cost_type: str) -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, -1.0]], dtype=np.float64)
    target = np.array([[0.25, 0.0], [1.5, 0.25]], dtype=np.float64)
    source_dual = np.array([0.4, 0.1, -0.2], dtype=np.float64)
    target_dual = np.array([0.2, -0.1], dtype=np.float64)
    difference = source[:, None, :] - target[None, :, :]
    if cost_type == "l1":
        dense_cost = np.sum(np.abs(difference), axis=2)
    elif cost_type == "linf":
        dense_cost = np.max(np.abs(difference), axis=2)
    else:
        dense_cost = np.linalg.norm(difference, axis=2)
    score = source_dual[:, None] + target_dual[None, :] - dense_cost
    scan = bidirectional_violation_scan(
        source_points=source,
        target_points=target,
        source_offset=None,
        target_offset=None,
        source_dual=source_dual,
        target_dual=target_dual,
        score_family="norm_cost",
        cost_type=cost_type,
        dot_scale=1.0,
        topk=1,
        theta=0.0,
        device="cpu",
        max_tile_bytes=64,
    )
    numerator = np.linalg.norm(np.maximum(score, 0.0))
    denominator = np.linalg.norm(dense_cost)
    assert scan.dual_feasibility == pytest.approx(numerator / (1.0 + denominator), abs=1e-12)
    assert scan.diagnostics["dual_feasibility_max_violation"] == pytest.approx(
        float(np.max(np.maximum(score, 0.0))), abs=1e-12
    )
    assert scan.diagnostics["max_tile_elements"] < source.shape[0] * target.shape[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Torch CUDA is unavailable")
def test_torch_scan_cuda_matches_cpu() -> None:
    source = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    target = np.array([[0.25], [1.5]], dtype=np.float32)
    source_offset = np.sum(source.astype(np.float64) ** 2, axis=1)
    target_offset = np.sum(target.astype(np.float64) ** 2, axis=1)
    kwargs = dict(
        source_points=source,
        target_points=target,
        source_offset=source_offset,
        target_offset=target_offset,
        source_dual=np.array([0.1, 0.2, -0.2]),
        target_dual=np.array([0.3, -0.1]),
        score_family="inner_product",
        cost_type="lowrank",
        dot_scale=2.0,
        topk=1,
        theta=0.0,
    )
    cpu = bidirectional_violation_scan(**kwargs, device="cpu")
    cuda = bidirectional_violation_scan(**kwargs, device="cuda")
    assert cuda.dual_feasibility == pytest.approx(cpu.dual_feasibility, rel=1e-12, abs=1e-12)
    torch.testing.assert_close(cuda.rows.cpu(), cpu.rows.cpu())
    torch.testing.assert_close(cuda.cols.cpu(), cpu.cols.cpu())
