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
    "export_experiments/main_scaling",
    "export_experiments/accuracy_runtime_pareto",
    "export_experiments/exactness_verification",
    "export_experiments/parameter_sensitivity",
    "paper_experiments/baselines/linear_ot/mdot_tnt",
)
IGNORE = shutil.ignore_patterns("*.so", "*.pyc", "__pycache__", "*.egg-info")
EXTRA_FILES = (
    ".github/workflows/ci.yml",
    "docs/algorithm.md",
    "docs/release_checklist.md",
    "scripts/validate_gpu_release.py",
    "scripts/validate_variants_extended.py",
    "scripts/check_sdist.py",
    "tests/hello_cost_perturbation_test.py",
    "tests/hello_algorithm_structure_test.py",
    "tests/hello_ot_variants_test.py",
    "tests/variants_numerical_equivalence_test.py",
    "tests/fixtures/variants_baseline_32k.npz",
    "examples/quickstart.py",
    "examples/verify_native.py",
    "export_experiments/README.md",
    "export_experiments/__init__.py",
    "export_experiments/synthetic.py",
    "third_party/README.md",
    "paper_experiments/__init__.py",
    "paper_experiments/baselines/__init__.py",
    "paper_experiments/baselines/linear_ot/__init__.py",
    "paper_experiments/baselines/linear_ot/problem.py",
    "paper_experiments/baselines/linear_ot/result.py",
    "paper_experiments/baselines/linear_ot/ott_sinkhorn.py",
    "paper_experiments/baselines/linear_ot/mdot_tnt_adapter.py",
    "paper_experiments/baselines/linear_ot/pot_proximal.py",
    "paper_experiments/baselines/linear_ot/POT_LICENSE",
    "paper_experiments/baselines/linear_ot/pot_emd.py",
    "paper_experiments/baselines/linear_ot/pot_lazy_emd.py",
    "paper_experiments/baselines/linear_ot/hello_solver.py",
    "paper_experiments/baselines/linear_ot/evaluation.py",
    "paper_experiments/common/__init__.py",
    "paper_experiments/common/feasible_rounding.py",
    "tests/public_pareto_test.py",
    "scripts/build_native_wheel.sh",
    "scripts/export_public.py",
    "scripts/validate_release.py",
    "tests/hello_inplace_feature_layout_test.py",
    "tests/test_hello_ot_public_api.py",
    "tests/test_hello_ot_dependency_boundary.py",
    "tests/test_hello_ot_numerical_equivalence.py",
    "tests/release_tooling_test.py",
    "tests/public_paper_experiments_test.py",
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
