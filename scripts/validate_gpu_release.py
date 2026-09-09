"""
CN: 在公开仓库中检查已安装 native 包，不构建或安装依赖。
EN: Validate an installed native package from the public repository without building or installing dependencies.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """CN: 先确认 CUDA/native 可用，再执行发布测试。EN: Require CUDA/native availability before running release tests."""
    parser = argparse.ArgumentParser(description="GPU release checks for an already installed HELLO build.")
    parser.add_argument("--extended", action="store_true", help="Include 32k variant equivalence fixtures.")
    parser.add_argument("--with-ott", action="store_true", help="Also check optional OTT initialization.")
    parser.add_argument("--output", type=Path, default=Path("gpu_release_tests.xml"))
    args = parser.parse_args()
    from hello_ot.diagnose import collect_diagnostics
    report = collect_diagnostics()
    if not report["cuda_available"] or not report["native_runtime_compatible"]:
        raise SystemExit("A compatible CUDA runtime is required; no CPU fallback.")
    missing = [name for name, item in report["native_extensions"].items() if not item["available"]]
    if missing:
        raise SystemExit(f"Missing native extensions: {missing}")
    if args.with_ott:
        import importlib.util
        if importlib.util.find_spec("ott") is None:
            raise SystemExit("--with-ott requires the optional OTT-JAX dependency.")
        import jax
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise SystemExit("--with-ott requires working GPU JAX; check CUDA library paths.")
    root = Path(__file__).resolve().parents[1]
    tests = ["tests/hello_cost_perturbation_test.py", "tests/hello_ot_variants_test.py"]
    if args.extended:
        fixture = root / "tests/fixtures/variants_baseline_32k.npz"
        if not fixture.is_file():
            raise SystemExit(f"Missing required fixture: {fixture}")
        tests.append("tests/variants_numerical_equivalence_test.py")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", *tests, f"--junitxml={output}"]
    if not args.with_ott:
        command += ["-k", "not test_solve_unbalanced_ott"]
    raise SystemExit(subprocess.run(command, cwd=root).returncode)


if __name__ == "__main__":
    main()
