"""
CN: 用已安装 wheel 在三张 GPU 上独立运行 32k variants 回归测试。
EN: Run independent 32k variant regressions on three GPUs using an installed wheel.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    """CN: 每个算法一个进程；跳过和超时均失败。EN: Use one process per algorithm; skips and timeouts fail."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="Three distinct physical GPUs, e.g. 4,5,6")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 3 or len(set(gpus)) != 3 or not all(g.isdigit() for g in gpus):
        parser.error("--gpus requires three distinct physical GPU indices")
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/fixtures/variants_baseline_32k.npz"
    if not fixture.is_file():
        raise SystemExit(f"Missing fixture: {fixture}")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="8", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    env.pop("PYTHONPATH", None)
    # CN: 在独立进程中确认包位于当前安装环境，不从源码目录加载。
    # EN: Confirm in an isolated process that the package belongs to the installation environment.
    probe = """
import hashlib,json,pathlib,sysconfig
import hello_ot
from hello_ot.diagnose import collect_diagnostics
p=pathlib.Path(hello_ot.__file__).resolve()
assert p.is_relative_to(pathlib.Path(sysconfig.get_path('purelib')).resolve()), str(p)
d=collect_diagnostics()
assert d['cuda_available'] and d['native_runtime_compatible'], d
assert all(x['available'] for x in d['native_extensions'].values()), d
print(json.dumps(dict(package=str(p),diagnostics=d),default=str))
"""
    def run(item):
        name, gpu = item
        local_env = dict(env, CUDA_VISIBLE_DEVICES=gpu)
        result = dict(variant=name, gpu=gpu, passed=False)
        print(f"GPU {gpu}: {name} started", flush=True)
        try:
            check = subprocess.run([sys.executable, "-I", "-c", probe], env=local_env,
                                   capture_output=True, text=True, check=True, timeout=120)
            (args.output / f"{name}_environment.json").write_text(check.stdout)
            xml = args.output / f"{name}.xml"
            with (args.output / f"{name}.log").open("w") as stream:
                completed = subprocess.run(
                    [sys.executable, "-I", "-m", "pytest", "-q",
                     str(root / "tests/variants_numerical_equivalence_test.py") +
                     f"::test_{name}_32k_numerical_equivalence", f"--junitxml={xml}"],
                    cwd=root, env=local_env, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=args.timeout)
            result["exit_code"] = completed.returncode
            tree = ET.parse(xml)
            cases = tree.findall(".//testcase")
            result["passed"] = completed.returncode == 0 and len(cases) == 1 and not any(
                c.find(tag) is not None for c in cases for tag in ("skipped", "failure", "error"))
        except Exception as exc:
            result["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                result["stderr"] = exc.stderr
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2))
        print(f"{name}: passed={result['passed']}", flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run, zip(("sdot", "gw", "uot"), gpus)))
    report = dict(python=sys.executable, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  results=rows, passed=all(r["passed"] for r in rows))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
