from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from hello_ot.diagnose import collect_diagnostics
from hello_ot._native_compat import native_runtime_compatibility


ROOT = Path(__file__).resolve().parents[1]


def test_diagnostics_reports_all_native_extensions() -> None:
    # CN: 诊断必须逐一报告四个扩展，缺失时也不能在收集阶段崩溃。
    # EN: Diagnostics must report all four extensions without crashing when any are absent.
    report = collect_diagnostics()
    assert set(report["native_extensions"]) == {
        "inner_product_scan",
        "norm_cost_scan",
        "support_sparsifier",
        "cupdlpx",
    }
    assert isinstance(report["wsl2"], bool)
    assert report["native_runtime_compatible"] == native_runtime_compatibility()[0]


def test_public_export_contains_release_entrypoints_and_valid_manifest(tmp_path: Path) -> None:
    # CN: allowlist 导出必须自带 quickstart、native 验证、全部 release tests 与可校验清单。
    # EN: The allowlist export must include quickstart, native validation, all release tests, and a valid manifest.
    destination = tmp_path / "public"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_public.py"), str(destination)],
        cwd=ROOT,
        check=True,
    )
    required = {
        "examples/quickstart.py",
        "examples/verify_native.py",
        "scripts/build_native_wheel.sh",
        "scripts/validate_release.py",
        "tests/torch_scan_test.py",
        "tests/hello_inplace_feature_layout_test.py",
        ".github/workflows/ci.yml",
        "export_experiments/exactness_verification/run.py",
        "export_experiments/parameter_sensitivity/run.py",
        "export_experiments/main_scaling/data_manifest.json",
        "third_party/README.md",
    }
    assert all((destination / relative).is_file() for relative in required)
    assert not (destination / "experiments").exists()
    # CN: 只使用导出目录的源码，验证公开入口不依赖内部工作区。
    # EN: Use only exported sources to verify public entry points are independent of the internal workspace.
    environment = dict(os.environ, PYTHONPATH=str(destination / "src"))
    subprocess.run(
        [sys.executable, "-c",
         "import paper_experiments.baselines.linear_ot as b; "
         "assert 'solve_public_hello' not in b.__all__; "
         "assert 'solve_hello' in b.__all__; "
         "assert 'solve_neufeld_cutplane' not in b.__all__; "
         "assert 'solve_zanetti_ipm' not in b.__all__; "
         "assert hasattr(b, 'solve_hello')"],
        cwd=destination, env=environment, check=True,
    )
    for name in ("hello_cost_perturbation_test.py", "hello_algorithm_structure_test.py",
                 "hello_ot_variants_test.py", "variants_numerical_equivalence_test.py"):
        assert (destination / "tests" / name).is_file()
    for family in ("main_scaling", "exactness_verification", "parameter_sensitivity", "accuracy_runtime_pareto"):
        subprocess.run(
            [sys.executable, "-m", f"export_experiments.{family}.run", "--help"],
            cwd=destination,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    manifest = destination / "PUBLIC_MANIFEST.sha256"
    entries = manifest.read_text(encoding="utf-8").splitlines()
    assert entries
    for entry in entries:
        digest, relative = entry.split("  ", maxsplit=1)
        assert hashlib.sha256((destination / relative).read_bytes()).hexdigest() == digest


def test_native_build_script_has_explicit_release_architectures() -> None:
    # CN: 构建脚本不能根据构建机可见 GPU 隐式缩窄发行架构。
    # EN: The build script must not implicitly narrow release architectures to GPUs visible on the builder.
    text = (ROOT / "scripts/build_native_wheel.sh").read_text(encoding="utf-8")
    assert "8.0 8.6 8.9 9.0+PTX" in text
    assert "torch==2.5.1+cu118" in text
    assert "release 11.8" in text
