"""CN: 下载并读取 representative ImageNet PCA feature。EN: Download and load representative ImageNet PCA features."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR / "data_manifest.json"


def load_manifest() -> dict[str, Any]:
    """CN: 读取固定的数据 artifact manifest。EN: Read the pinned data-artifact manifest."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def default_cache_path(dimension: int = 2048) -> Path:
    """CN: 返回用户级 HELLO 数据缓存路径。EN: Return the user-level HELLO data cache path."""
    manifest = load_manifest()
    cache_root = Path(os.environ.get("HELLO_OT_DATA_DIR", Path.home() / ".cache" / "hello_ot"))
    return cache_root / str(manifest["artifacts"][str(dimension)]["filename"])


def artifact_url(manifest: dict[str, Any] | None = None, *, dimension: int = 2048) -> str:
    """CN: 构造 Hugging Face resolve URL。EN: Build the Hugging Face resolve URL."""
    value = load_manifest() if manifest is None else manifest
    if not value.get("revision"):
        raise ValueError("Dataset release is not pinned yet; use --feature-file with local generated data.")
    return (
        f"https://huggingface.co/datasets/{value['repository']}/resolve/"
        f"{value['revision']}/{value['artifacts'][str(dimension)]['filename']}?download=true"
    )


def ensure_representative_features(feature_file: Path | None = None, *, dimension: int = 2048) -> Path:
    """
    CN: 确保 feature artifact 存在；缺失时从 Hugging Face 下载。
    EN: Ensure the feature artifact exists, downloading it from Hugging Face when absent.
    """
    path = default_cache_path(dimension) if feature_file is None else Path(feature_file).expanduser().resolve()
    manifest = load_manifest()
    entry = manifest["artifacts"][str(dimension)]
    if feature_file is None and not entry.get("sha256"):
        raise ValueError("Dataset checksums are not generated yet; use --feature-file for local data.")
    if path.is_file():
        # CN: 只对 canonical 下载缓存强制 manifest checksum；显式本地文件可用于开发子集。
        # EN: Enforce the manifest checksum only for the canonical cache; explicit local files may be development subsets.
        if feature_file is None:
            _verify_sha256(path, entry.get("sha256"))
        return path
    if feature_file is not None:
        raise FileNotFoundError(f"ImageNet PCA feature file not found: {path}")
    url = artifact_url(manifest, dimension=dimension)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    print(f"[main_scaling] downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, temporary, _download_progress)
        _verify_sha256(temporary, entry.get("sha256"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_representative_features(
    *,
    feature_file: Path | None,
    min_rows: int,
    min_dimension: int,
) -> np.ndarray:
    """
    CN: 以内存映射读取 artifact 并验证 dtype/shape。
    EN: Memory-map the artifact and validate its dtype and shape.
    """
    if min_dimension not in (4, 32, 256, 2048):
        raise ValueError("Published dimensions are 4, 32, 256, and 2048.")
    path = ensure_representative_features(feature_file, dimension=min_dimension)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.dtype != np.float32:
        raise ValueError(f"Expected a two-dimensional float32 feature matrix; got {array.shape}, {array.dtype}.")
    if array.shape[0] < int(min_rows) or array.shape[1] != int(min_dimension):
        raise ValueError(
            f"Feature artifact shape {array.shape} does not match required rows/dimension {(int(min_rows), int(min_dimension))}."
        )
    return array


def _verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != str(expected):
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}.")


def _download_progress(block_count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(int(block_count) * int(block_size), int(total_size))
    percent = 100.0 * downloaded / float(total_size)
    print(f"\r[main_scaling] download {percent:5.1f}%", end="", flush=True)
    if downloaded >= total_size:
        print(flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and verify the seed=42 PCA artifacts.")
    parser.add_argument("--d", type=int, nargs="+", choices=[4, 32, 256, 2048], default=[4, 32, 256, 2048])
    args = parser.parse_args()
    for dimension in args.d:
        print(ensure_representative_features(dimension=dimension))
