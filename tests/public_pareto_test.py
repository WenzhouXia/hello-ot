from pathlib import Path
import json

import numpy as np
import pytest

from export_experiments.accuracy_runtime_pareto.protocol import fingerprint, load_config, load_problem, write_json
from export_experiments.accuracy_runtime_pareto.collect import collect
from paper_experiments.baselines.linear_ot.problem import LinearOTProblem


def test_default_sweep_is_the_agreed_64_cases():
    config = load_config()
    assert config["n"] == 65536 and config["seed"] == 42
    assert config["d"] == [4, 32, 256, 2048]
    assert sum(map(len, config["methods"].values())) * len(config["d"]) == 64
    assert config["methods"]["ipot"] == [1, .3, .1, .03, .01]
    assert config["methods"]["mdot"] == [64, 512, 4096, 32768, 262144]


def test_fingerprint_changes_with_actual_target(tmp_path):
    path = tmp_path / "d4.npy"
    target = np.arange(32, dtype=np.float32).reshape(8, 4)
    np.save(path, target)
    first = load_problem(path, 8, 4)
    target[0, 0] += 1
    np.save(path, target)
    assert fingerprint(first) != fingerprint(load_problem(path, 8, 4))


def test_collect_keeps_failure_but_excludes_reference_and_request(tmp_path):
    folder = tmp_path / "n8_d4"
    record = dict(method="ipot", parameter=1, n=8, d=4, seed=42,
                  runtime_sec=None, relative_error=None, status="failed", error="OOM")
    write_json(folder / "ipot_1.json", record)
    write_json(folder / "ipot_1.request.json", record)
    write_json(folder / "reference_dense_hash.json", {"objective": 1, "status": "success"})
    assert collect(tmp_path) == [record]
    assert "OOM" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("backend", ["dense", "lazy"])
def test_reference_uses_hello_duals_and_validates_exact_solution(monkeypatch, backend):
    from paper_experiments.baselines.linear_ot import hello_solver
    from export_experiments.accuracy_runtime_pareto.worker import reference_objective
    source = np.array([[0, 0, 0, 0], [3, 0, 0, 0]], dtype=np.float32)
    problem = LinearOTProblem(source, source + 1)
    fake = type("Artifacts", (), {"source_dual": np.zeros(2), "target_dual": np.zeros(2)})()
    monkeypatch.setattr(hello_solver, "solve_hello_with_duals", lambda p: fake)
    assert reference_objective(problem, backend) == pytest.approx(4)


def test_missing_worker_result_retains_failure(tmp_path):
    from export_experiments.accuracy_runtime_pareto.run import run_worker
    request = dict(method="reference", n=8, d=4, output=str(tmp_path / "result.json"),
                   feature_file="/missing/file.npy", fingerprint="bad", reference="dense")
    result = run_worker(request, Path(__file__).resolve().parents[1], 30)
    assert result["status"] == "failed"
    assert result["error"]


def test_exported_scope_excludes_budgeted_pypi():
    from scripts.export_public import EXTRA_FILES, DIRECTORIES
    assert "paper_experiments/baselines/linear_ot/mdot_tnt_adapter.py" in EXTRA_FILES
    assert not any("budgeted" in p for p in (*EXTRA_FILES, *DIRECTORIES))
    text = Path("paper_experiments/baselines/linear_ot/mdot_tnt_adapter.py").read_text()
    assert "from mdot_tnt.lowmem" not in text
    assert 'backend: str = "keops"' in text
