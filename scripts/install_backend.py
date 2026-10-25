"""CN: 在已有环境中独立安装可选后端。EN: Install an optional backend independently in an existing environment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


VERSION = "0.2.0"
ROOT = Path(__file__).resolve().parents[1]


def environment_command(manager, environment, *command):
    """
    CN: 保持 Conda/Micromamba 的已有环境选择方式，不自动创建环境。
    EN: Select an existing Conda/Micromamba environment without creating one.
    """
    return [manager, "run", *(["--no-capture-output"] if manager == "conda" else []), "-n", environment, *command]


def main():
    """
    CN: 先报告兼容性，再分别安装核心和选定扩展，最后运行相应 quickstart。
    EN: Report compatibility before installing the core and selected extension, then run its quickstart.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["native", "cpp"])
    parser.add_argument("environment", nargs="?", default="hello_ot")
    parser.add_argument("--wheel", help="local wheel path or release URL")
    args = parser.parse_args()
    if not args.environment or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in args.environment):
        parser.error("environment name may contain only letters, digits, '.', '_' and '-'")
    probe = '''import json, platform, sys
from importlib import metadata
import torch
try:
    distribution = metadata.distribution("hello-ot")
    legacy = any(str(p).startswith("hello_ot/") and str(p).endswith(".so") for p in distribution.files or [])
except metadata.PackageNotFoundError:
    legacy = False
print(json.dumps(dict(python=list(sys.version_info[:2]), system=platform.system(), machine=platform.machine(),
                      torch=torch.__version__, cuda=torch.version.cuda, abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), legacy=legacy)))'''
    candidates = []
    errors = []
    for manager in ["conda", "micromamba"]:
        try:
            result = subprocess.run(environment_command(manager, args.environment, "python3", "-c", probe),
                                    capture_output=True, text=True)
            if result.returncode == 0:
                info = json.loads(result.stdout.strip().splitlines()[-1])
                candidates.append((manager, info))
            else:
                errors.append(f"{manager}: {result.stderr.strip() or result.stdout.strip()}")
        except FileNotFoundError:
            continue
    if len(candidates) != 1:
        raise SystemExit("Expected exactly one existing Conda/Micromamba environment with PyTorch named "
                         f"{args.environment!r}; found {len(candidates)}. Prepare the documented environment first.\n" + "\n".join(errors))
    manager, info = candidates[0]
    print("Detected environment: " + json.dumps(info), flush=True)
    compatible = info["python"] == [3, 12] and info["system"] == "Linux" and info["machine"] == "x86_64"
    if args.backend == "native":
        compatible = compatible and info["torch"].split("+")[0] == "2.7.1" and info["cuda"] == "11.8" and info["abi"]
    if not compatible:
        expected = "Linux x86-64, Python 3.12"
        if args.backend == "native":
            expected += ", torch 2.7.1+cu118, C++11 ABI=True"
        raise SystemExit(f"Incompatible environment. Expected: {expected}. Actual: {info}.\n"
                         "Use the documented supported environment. Source-build instructions are in README.md: "
                         f"bash scripts/build_{'native' if args.backend == 'native' else 'cpp'}_wheel.sh dist. "
                         "Source builds retain the documented environment requirements.")
    if args.backend == "cpp" and info["legacy"]:
        raise SystemExit("This environment contains the old combined native/core distribution. "
                         "Migrate once with the new scripts/install_native.sh before adding C++; "
                         "installing the new core here would otherwise remove the legacy CUDA files.")

    def run(*command):
        """
        CN: 使用精确选择的环境执行安装及验证。
        EN: Install and validate in the precisely selected environment.
        """
        subprocess.run(environment_command(manager, args.environment, *command), check=True, cwd=ROOT)

    wheel = args.wheel or (f"https://github.com/WenzhouXia/hello-ot/releases/download/v{VERSION}/"
                           f"hello_ot_{args.backend}-{VERSION}-cp312-cp312-linux_x86_64.whl")
    run("python3", "-m", "pip", "install", str(ROOT))
    if args.backend == "cpp":
        run("python3", "-m", "pip", "install", "threadpoolctl>=3,<4")
    run("python3", "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel)
    run("python3", "-m", "hello_ot.diagnose", f"--require-{args.backend}")
    run("python3", "examples/quickstart.py", "--backend", args.backend)


if __name__ == "__main__":
    main()
