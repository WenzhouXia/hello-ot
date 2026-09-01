"""CN: 按 allowlist 导出无需源码重写的公开仓库。EN: Export the public repository from an allowlist without rewriting sources."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


FILES = (
    "LICENSE",
    ".gitignore",
    "MANIFEST.in",
    "README.md",
    "README.zh-CN.md",
    "THIRD_PARTY_NOTICES.md",
    "TERMINOLOGY.md",
    "pyproject.toml",
    "setup.py",
)
DIRECTORIES = (
    "src/hello_ot",
    "experiments/main_scaling",
    "experiments/accuracy_runtime_pareto",
)
IGNORE = shutil.ignore_patterns("*.so", "*.pyc", "__pycache__", "*.egg-info")
EXTRA_FILES = (
    ".github/workflows/ci.yml",
    "docs/algorithm.md",
    "examples/quickstart.py",
    "examples/verify_native.py",
    "experiments/README.md",
    "experiments/__init__.py",
    "experiments/synthetic.py",
    "scripts/build_native_wheel.sh",
    "scripts/export_public.py",
    "scripts/validate_release.py",
    "tests/hello_inplace_feature_layout_test.py",
    "tests/test_hello_ot_public_api.py",
    "tests/test_hello_ot_dependency_boundary.py",
    "tests/test_hello_ot_numerical_equivalence.py",
    "tests/release_tooling_test.py",
    "tests/torch_backend_end_to_end_test.py",
    "tests/torch_packaging_test.py",
    "tests/torch_prewarm_test.py",
    "tests/torch_restricted_ot_pdlp_test.py",
    "tests/torch_scan_test.py",
)


def _write_manifest(destination: Path) -> None:
    """
    CN: 为公开快照中的全部文件生成稳定的 SHA-256 清单。
    EN: Generate a stable SHA-256 manifest for every file in the public snapshot.
    """
    entries = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        relative = path.relative_to(destination).as_posix()
        if relative == "PUBLIC_MANIFEST.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {relative}")
    (destination / "PUBLIC_MANIFEST.sha256").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"destination must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for relative in FILES + EXTRA_FILES:
        source = root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative in DIRECTORIES:
        shutil.copytree(root / relative, destination / relative, ignore=IGNORE)
    _write_manifest(destination)
    print(destination)


if __name__ == "__main__":
    main()
