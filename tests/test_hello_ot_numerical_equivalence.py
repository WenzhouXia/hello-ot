from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

import hello_ot


BASELINE_SHA256 = {
    "l2^2": {
        "objective": "c908c4278d916efa0ee879ba9aabf04255c38e732868dbc7ddafb0bbfd66c9f9",
        "keys": "5eb94b773beec757806ba755f45f1492e75029917bd40a9fa8b712964eb48314",
        "values": "6dbc905846d47029e29ad5d67d9eedc51d44757483800e468eebd923757ed5f3",
        "source_dual": "ff16806fbaac9c0b40e1a0fc8a45ec5a09b9a45540720b839861f31cbc722ca2",
        "target_dual": "a1bcdf50e6f82068250ac43a1007d1928fee155721bb2b7e741b59fe069664e2",
    },
    "l1": {
        "objective": "24ac2960d4aaa38ee72e66b56e26d9ef3044889e9efd62e0705672957e431638",
        "keys": "91c48461ac7c658b889cfcc91157fe3c10672facd416acf6f94733e904dee34a",
        "values": "513d22d767a01fd1486cddc725993c2f780ecc0f80ece589c9eb89cc193a0ff8",
        "source_dual": "9e114d70aea7f0e5232aa124799d00e327f76264e029809f25fc06982c81e9be",
        "target_dual": "b833ac42a7e85e01c982f5517c34397e95cc3dc2f653a67c74df61aa918277e4",
    },
    "l2": {
        "objective": "defdb027a5310ffef99c49075671f9e6f87e3d9abee643310d7802763364eda7",
        "keys": "d10e2a486b10b0c1bf50f96557d353765102715e62aca9fee88aec62f404d442",
        "values": "d21c35a70af61ad97b0a2897950833a6ff41779dbe7f68c9c116200edd730b5a",
        "source_dual": "9e2a2391914c8e8346fe7e7521eea651000eee896f6d0530205240701066ef1f",
        "target_dual": "7b83a4424a6ba5274918d6a2edc4da7db7dafcb1d88989eaf740aba292a89b62",
    },
    "linf": {
        "objective": "e0c258f12e9be636b82b750fa7caa1839b52c070b7ac46f88933aac96878a73f",
        "keys": "2baf2ca623c5e86535f368945b9baf5c48961ad36a3daeee2819435d0b99e7de",
        "values": "94bf0a2b285a154ab4403a98ffe4ebf804920c1f103898bed84893c5311e74f6",
        "source_dual": "b9cb0570b7001021fb3d749b97bb7a39d68bc45b1de38995f619564a0a49c60f",
        "target_dual": "8dfa5aa0b44573112206a98d0d21dbfa9c90b15968056f4bf3ab41eaefffc069",
    },
}


def _digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HELLO is CUDA-only")
@pytest.mark.parametrize("cost", ("l2^2", "l1", "l2", "linf"))
def test_paper_defaults_are_bitwise_equivalent_to_pre_migration_baseline(cost: str) -> None:
    """
    CN: 固化提交 f9b6e17 在论文参数下的 objective、support、primal 与 dual 字节结果。
    EN: Freeze objective, support, primal, and dual bytes from commit f9b6e17 under paper parameters.
    """
    rng = np.random.default_rng(20260901)
    source = rng.normal(size=(24, 4)).astype(np.float32)
    target = rng.normal(size=(28, 4)).astype(np.float32)
    result = hello_ot.solve(
        source,
        target,
        cost=cost,
        max_iterations=4,
        random_seed=42,
        options=hello_ot.SolverOptions(coarsest_size_threshold=8),
    )
    solution = result.solution
    keys = np.asarray(solution.rows, dtype=np.int64) * int(solution.shape[1]) + np.asarray(
        solution.cols, dtype=np.int64
    )
    order = np.argsort(keys, kind="stable")
    actual = {
        "objective": np.asarray([result.objective], dtype=np.float64),
        "keys": keys[order],
        "values": np.asarray(solution.values, dtype=np.float64)[order],
        "source_dual": np.asarray(solution.source_dual, dtype=np.float64),
        "target_dual": np.asarray(solution.target_dual, dtype=np.float64),
    }
    assert {name: _digest(value) for name, value in actual.items()} == BASELINE_SHA256[cost]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HELLO is CUDA-only")
def test_refinement_executes_paper_operator_order() -> None:
    """
    CN: 由测试 observer 验证真实热路径按 SolveLP、CheckOptimality、UpdateSupport 执行。
    EN: Use the test observer to verify the real hot path executes SolveLP, CheckOptimality, UpdateSupport.
    """
    from hello_ot._internal.runtime_context import use_solve_runtime

    rng = np.random.default_rng(7)
    source = rng.normal(size=(32, 4)).astype(np.float32)
    target = rng.normal(size=(35, 4)).astype(np.float32)
    events = []
    with use_solve_runtime(observer=lambda name, payload: events.append((name, payload))):
        hello_ot.solve(
            source,
            target,
            max_iterations=3,
            options=hello_ot.SolverOptions(
                coarsest_size_threshold=4,
                assignment_topk=1,
                pricing_topk=1,
                support_budget_factor=2.0,
            ),
        )

    operators = [(name, payload) for name, payload in events if name in {"solve_lp", "check_optimality", "update_support"}]
    assert any(name == "update_support" for name, _ in operators)
    for index, (name, payload) in enumerate(operators):
        if name == "solve_lp":
            assert operators[index + 1][0] == "check_optimality"
        if name == "check_optimality" and not bool(payload["converged"]):
            assert operators[index + 1][0] == "update_support"
