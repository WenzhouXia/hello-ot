from __future__ import annotations

import builtins

import numpy as np
import torch

import hello_ot


def test_torch_cpu_prewarm_does_not_import_native(monkeypatch) -> None:
    # CN: portable prewarm 不得探测或加载 native 扩展。
    # EN: Portable prewarm must neither probe nor import native extensions.
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if str(name).startswith("hello_ot._native"):
            raise AssertionError(f"Torch prewarm imported native module: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    problem = hello_ot.Problem(
        np.asarray([[0.0], [1.0]], dtype=np.float64),
        np.asarray([[0.25], [0.75]], dtype=np.float64),
        "l2^2",
    )
    stats = hello_ot.prewarm(
        problem,
        hello_ot.SolverOptions(backend="torch", torch_device="cpu"),
    )
    assert any(item == "torch_scan:cpu" for item in stats.operations)
    assert any(item == "torch_restricted_lp:cpu" for item in stats.operations)
    assert "pot_coarsest" in stats.operations
