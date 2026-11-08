from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hello_ot.diagnose import collect_diagnostics
from hello_ot._native_compat import native_runtime_compatibility


ROOT = Path(__file__).resolve().parents[1]


def test_diagnostics_reports_all_native_extensions() -> None:
    # CN: 诊断必须逐一报告全部扩展，缺失时也不能在收集阶段崩溃。
    # EN: Diagnostics must report all required extensions without crashing when any are absent.
    report = collect_diagnostics()
    assert set(report["native_extensions"]) == {
        "sparse_pair_cost",
        *{f"scan_{part}" for part in ("initial", "initial_small", "pricing", "pricing_small", "cost")},
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
        "docs/generic_scan.md",
        "scripts/create_hello_ot_env.sh",
        "scripts/install_native.sh",
        "scripts/install_jax_gpu.sh",
        "scripts/build_native_wheel.sh",
        "scripts/build_cpp_wheel.sh",
        "scripts/install_cpp.sh",
        "scripts/install_backend.py",
        "packaging/cpp/setup.py",
        "packaging/native/setup.py",
        "scripts/validate_release.py",
        "tests/hello_gpu_resident_test.py",
        "tests/torch_scan_test.py",
        "tests/hello_inplace_feature_layout_test.py",
        ".github/workflows/ci.yml",
        "export_experiments/exactness_verification/run.py",
        "export_experiments/parameter_sensitivity/run.py",
        "export_experiments/main_scaling/data_manifest.json",
        "third_party/README.md",
    }
    assert all((destination / relative).is_file() for relative in required)
    terminology = (destination / "TERMINOLOGY.md").read_text(encoding="utf-8")
    assert "## GPU-resident execution" in terminology
    assert "## Cost perturbation" in terminology
    assert "Hierarchical KL-UOT" not in terminology
    assert "RDM" not in terminology
    assert "Online Laguerre-weighted coreset" not in terminology
    assert "hier_wb" not in terminology
    assert "restricted-WB" not in terminology
    assert "barycenter" not in (destination / ".gitignore").read_text()
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
    assert "torch==2.7.1+cu118" in text
    assert "release 11.8" in text
    assert "cuda_runtime.h" in text
    assert "_GLIBCXX_USE_CXX11_ABI" in text


def test_release_uploads_validated_persistent_artifacts() -> None:
    # CN: push 阶段必须上传 build 已校验的持久化产物，不能引用未填充的临时目录。
    # EN: Push must upload build-validated persistent artifacts, not an unpopulated temporary directory.
    release_script = ROOT / "export_hello_ot.sh"
    if not release_script.is_file():
        pytest.skip("publication orchestration is internal and excluded from public snapshots")
    text = release_script.read_text(encoding="utf-8")
    assert 'local artifacts="$ARTIFACT_DIR"' in text
    assert 'local artifacts="$STAGING_ROOT/artifacts"' not in text


def test_layered_environment_installers_pin_compatible_stacks() -> None:
    # CN: 三层安装必须共享 Python/PyTorch 数值栈，并隔离 JAX 的 cuDNN 8。
    # EN: The three installers must share one Python/PyTorch stack and isolate JAX cuDNN 8.
    create = (ROOT / "scripts/create_hello_ot_env.sh").read_text(encoding="utf-8")
    native = (ROOT / "scripts/install_native.sh").read_text(encoding="utf-8")
    jax = (ROOT / "scripts/install_jax_gpu.sh").read_text(encoding="utf-8")
    assert "numpy==1.26.4 scipy==1.15.3" in create
    installer = (ROOT / "scripts/install_backend.py").read_text(encoding="utf-8")
    assert 'install_backend.py" native' in native
    assert '"--no-deps", "--force-reinstall"' in installer
    assert "hello_ot_{args.backend}" in installer
    assert "jaxlib==0.4.25+cuda11.cudnn86" in jax
    assert "nvidia-cudnn-cu11==8.9.6.50" in jax
    assert 'ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"' in jax
    assert 'LD_LIBRARY_PATH="$JAX_CUDNN_LIB' in jax
    assert "PYTORCH_CUDA_LIBS" in jax


def test_readme_push_updates_only_docs_and_preserves_release_tag(tmp_path: Path) -> None:
    """
    CN: 用临时本地 Git 仓库验证文档快捷发布，检查范围、清单和已有 release tag。
    EN: Verify fast README publishing using temporary local Git repositories, including scope, manifest and release tag.
    """
    script = ROOT / "export_hello_ot.sh"
    if not script.is_file():
        pytest.skip("publication orchestration is internal and excluded from public snapshots")

    def git(directory, *args):
        """
        CN: 仅操作测试专用的本地仓库。
        EN: Operate only on the local test repositories.
        """
        return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    public = tmp_path / "public"
    subprocess.run(["git", "clone", str(remote), str(public)], check=True)
    git(public, "symbolic-ref", "HEAD", "refs/heads/main")
    git(public, "config", "user.name", "Release test")
    git(public, "config", "user.email", "release-test@example.invalid")
    git(public, "config", "commit.gpgsign", "false")
    git(public, "config", "tag.gpgsign", "false")
    originals = {"README.md": "Old English\n", "README.zh-CN.md": "旧中文\n", "solver.py": "original runtime\n"}
    for name, value in originals.items():
        (public / name).write_text(value, encoding="utf-8")
    (public / "PUBLIC_MANIFEST.sha256").write_text("".join(
        f"{hashlib.sha256(value.encode()).hexdigest()}  {name}\n"
        for name, value in sorted(originals.items())
    ))
    git(public, "add", "--", *originals, "PUBLIC_MANIFEST.sha256")
    git(public, "commit", "-m", "test initial snapshot")
    git(public, "tag", "v0.2.0")
    git(public, "push", "origin", "main", "v0.2.0")
    release_revision = git(remote, "rev-parse", "v0.2.0")

    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(script, source / script.name)
    (source / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/install_backend.py", source / "scripts/install_backend.py")
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    (source / "README.md").write_text("New English\n")
    (source / "README.zh-CN.md").write_text("新中文\n", encoding="utf-8")
    (source / "solver.py").write_text("unrelated change\n")
    environment = dict(os.environ, HELLO_OT_PUBLIC_REPO_DIR=str(public),
                       HELLO_OT_REMOTE_URL=str(remote), HELLO_OT_REMOTE_NAME="origin",
                       HELLO_OT_REMOTE_BRANCH="main", HELLO_OT_EXPORT_DIR=str(tmp_path / "unused-export"),
                       HELLO_OT_ARTIFACT_DIR=str(tmp_path / "unused-artifacts"))
    command = ["bash", str(source / script.name), "push-readme", "docs: test README update"]
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    assert set(git(public, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()) == {
        "README.md", "README.zh-CN.md", "PUBLIC_MANIFEST.sha256"}
    assert (public / "solver.py").read_text() == originals["solver.py"]
    for name in ("README.md", "README.zh-CN.md"):
        assert (public / name).read_bytes() == (source / name).read_bytes()
    for line in (public / "PUBLIC_MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((public / name).read_bytes()).hexdigest() == digest
    assert git(remote, "rev-parse", "v0.2.0") == release_revision
    assert git(remote, "rev-parse", "main") == git(public, "rev-parse", "HEAD")
    assert not (tmp_path / "unused-export").exists()
    assert not (tmp_path / "unused-artifacts").exists()

    (public / "solver.py").write_text("user work\n")
    before = git(remote, "rev-parse", "main")
    rejected = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "uncommitted changes" in rejected.stderr
    assert (public / "solver.py").read_text() == "user work\n"
    assert git(remote, "rev-parse", "main") == before


@pytest.mark.parametrize("mode,version", [("new", "0.3.0"), ("new", "0.3.0rc1"),
                                         ("release", "0.3.0"), ("tag", "0.3.0"),
                                         ("api_error", "0.3.0"), ("stale", "0.3.0")])
def test_versioned_publication_preserves_existing_releases(tmp_path, mode, version):
    """
    CN: 只用本地bare仓库和模拟gh验证新版本发布，已有版本、网络错误和旧快照在推送前停止。
    EN: Use only a local bare repository and mocked gh to verify new releases and reject existing versions, network failures and stale snapshots before pushing.
    """
    import json

    script = ROOT / "export_hello_ot.sh"
    if not script.is_file():
        pytest.skip("publication orchestration is internal and excluded from public snapshots")

    def git(directory, *args):
        """
        CN: 仅操作测试专用的本地仓库。
        EN: Operate only on test-owned local repositories.
        """
        return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()

    remote, public = tmp_path / "remote.git", tmp_path / "public"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    subprocess.run(["git", "clone", str(remote), str(public)], check=True)
    git(public, "symbolic-ref", "HEAD", "refs/heads/main")
    for key, value in (("user.name", "Release test"), ("user.email", "test@example.invalid"),
                       ("commit.gpgsign", "false"), ("tag.gpgsign", "false")):
        git(public, "config", key, value)
    (public / "README.md").write_text("Previous release\n")
    git(public, "add", "README.md")
    git(public, "commit", "-m", "Previous release")
    git(public, "tag", "v0.2.0")
    git(public, "push", "origin", "main", "v0.2.0")
    before = git(remote, "rev-parse", "main")
    if mode == "tag":
        git(public, "tag", f"v{version}")
        git(public, "push", "origin", f"v{version}")

    source, snapshot, artifacts = (tmp_path / name for name in ("source", "snapshot", "artifacts"))
    for directory in (source, snapshot):
        (directory / "scripts").mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/install_backend.py", directory / "scripts/install_backend.py")
        selected = "0.2.0" if mode == "stale" and directory == snapshot else version
        (directory / "pyproject.toml").write_text(f'[project]\nname = "hello-ot"\nversion = "{selected}"\n')
    shutil.copy2(script, source / script.name)
    (snapshot / "README.md").write_text("New release\n")
    (snapshot / "PUBLIC_MANIFEST.sha256").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(snapshot).as_posix()}\n"
        for path in sorted(snapshot.rglob("*")) if path.is_file()))
    artifacts.mkdir()
    names = [f"hello_ot-{version}-py3-none-any.whl", f"hello_ot-{version}.tar.gz"]
    for kind in ("native", "cpp"):
        names += [f"hello_ot_{kind}-{version}-cp312-cp312-linux_x86_64.whl", f"hello_ot_{kind}-{version}.tar.gz"]
    for name in names:
        (artifacts / name).write_bytes(b"mock artifact\n")
    (artifacts / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((artifacts / name).read_bytes()).hexdigest()}  {name}\n" for name in names))
    (artifacts / "PUBLIC_SNAPSHOT.sha256").write_text(hashlib.sha256((snapshot / "PUBLIC_MANIFEST.sha256").read_bytes()).hexdigest()+"\n")
    log, fake = tmp_path / "gh.jsonl", tmp_path / "gh"
    fake.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
a=sys.argv[1:]
with Path(os.environ["MOCK_GH_LOG"]).open("a") as f:f.write(json.dumps(a)+"\\n")
if a[:2]==["auth","status"]:sys.exit(0)
if a[0]=="api":
    if os.environ["MOCK_GH_MODE"]=="api_error":sys.exit(1)
    print("v0.2.0")
    if os.environ["MOCK_GH_MODE"]=="release":print("v"+os.environ["MOCK_VERSION"])
    sys.exit(0)
if a[:2]==["release","create"]:
    for p in a[3:a.index("--repo")]:assert Path(p).is_file(),p
    sys.exit(0)
raise SystemExit("Unexpected gh mutation: "+str(a))
''')
    fake.chmod(0o755)
    environment = dict(os.environ, HELLO_OT_EXPORT_DIR=str(snapshot), HELLO_OT_ARTIFACT_DIR=str(artifacts),
                       HELLO_OT_PUBLIC_REPO_DIR=str(public), HELLO_OT_REMOTE_URL=str(remote),
                       HELLO_OT_REMOTE_NAME="origin", HELLO_OT_REMOTE_BRANCH="main",
                       HELLO_OT_GH_COMMAND=str(fake), HELLO_OT_GITHUB_REPOSITORY="test/local",
                       MOCK_GH_LOG=str(log), MOCK_GH_MODE=mode, MOCK_VERSION=version)
    result = subprocess.run(["bash", str(source / script.name), "push", "Release test"],
                            env=environment, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    assert git(remote, "rev-parse", "v0.2.0") == before
    assert all(call[:2] != ["release", "delete"] for call in calls)
    if mode == "new":
        assert result.returncode == 0, result.stdout+result.stderr
        create = next(call for call in calls if call[:2] == ["release", "create"])
        assert create[2] == f"v{version}"
        assert ("--prerelease" in create) == ("rc" in version)
        assert create[create.index("--target")+1] == git(remote, "rev-parse", "main")
        assert git(remote, "rev-parse", "main") != before
    else:
        assert result.returncode != 0
        assert git(remote, "rev-parse", "main") == before
        assert all(call[:2] != ["release", "create"] for call in calls)
