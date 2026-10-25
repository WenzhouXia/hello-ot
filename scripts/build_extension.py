"""CN: 在隔离暂存目录构建独立扩展发行包。EN: Build independent extension distributions in isolated staging directories."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib


def main():
    """
    CN: 仅复制公开扩展源码和发行元数据，避免纳入开发实验。
    EN: Copy only public extension sources and distribution metadata, excluding development experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["cpp", "native"])
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"hello-{args.kind}-build-") as temporary:
        stage = Path(temporary) / "package"
        shutil.copytree(root / "packaging" / args.kind, stage,
                        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", "dist"))
        shutil.copy2(root / "LICENSE", stage / "LICENSE")
        source = root / "src/hello_ot/_native"
        destination = stage / "native_src"
        destination.mkdir()
        for name in (["solver.cpp", "scan.cpp"] if args.kind == "cpp" else
                     ["inner_product_scan", "norm_cost_scan", "support_sparsifier", "cupdlpx"]):
            path = source / "cpp" / name if args.kind == "cpp" else source / name
            if path.is_file():
                shutil.copy2(path, destination / name)
            else:
                shutil.copytree(path, destination / name,
                                ignore=shutil.ignore_patterns("*.so", "*.pyc", "__pycache__", "build", "*.egg-info", "*.o"))
        subprocess.run([sys.executable, "-c", "import setuptools.build_meta as b; b.build_sdist(" + repr(str(output)) + ")"],
                       cwd=stage, check=True)
        version = tomllib.loads((stage / "pyproject.toml").read_text())["project"]["version"]
        archive = output / f"hello_ot_{args.kind}-{version}.tar.gz"
        subprocess.run([sys.executable, "-m", "pip", "wheel", str(archive), "--no-deps", "--no-build-isolation",
                        "--wheel-dir", str(output)], cwd=stage, check=True)


if __name__ == "__main__":
    main()
