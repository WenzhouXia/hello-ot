"""CN: 从本地 raw ImageNet latent 生成公开 PCA feature artifact。EN: Build the public PCA artifact from raw latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_PATH = Path(__file__).resolve().parent / "data_manifest.json"
PCA_SAMPLE_COUNT = 50_000
OUTPUT_ROWS = 262_144
OUTPUT_DIMENSIONS = (4, 32, 256, 2048)


def _load_stats(path: Path, device: Any) -> tuple[Any, Any]:
    """CN: 加载 latent channel mean/std。EN: Load latent channel mean/std statistics."""
    import torch

    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    return payload["mean"], payload["std"]


def _fit_pca(files: list[Path], mean: Any, std: Any, *, device: Any, dimension: int) -> tuple[Any, Any]:
    """
    CN: 从固定的 50,000 个标准化 latent 拟合 PCA。
    EN: Fit PCA from the fixed sample of 50,000 standardized latents.
    """
    import torch
    from safetensors import safe_open
    from tqdm import tqdm

    shuffled = list(files)
    random.shuffle(shuffled)
    blocks = []
    count = 0
    for path in tqdm(shuffled, desc="PCA fitting load"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            latent = handle.get_slice("latents")[:].to(device)
        latent = (latent - mean) / std
        latent = latent.flatten(start_dim=1)
        blocks.append(latent)
        count += int(latent.shape[0])
        if count >= PCA_SAMPLE_COUNT:
            break
    data = torch.cat(blocks, dim=0)[:PCA_SAMPLE_COUNT]
    pca_mean = torch.mean(data, dim=0)
    centered = data - pca_mean
    covariance = centered.T @ centered / float(data.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:dimension]
    return pca_mean, eigenvectors[:, order].T


def build_artifact(latent_dir: Path, output: Path, dimension: int) -> None:
    """
    CN: 依次投影前 262,144 个正向 latent，并直接写入 NPY memmap。
    EN: Project the first 262,144 positive latents directly into an NPY memmap.
    """
    import torch
    from safetensors import safe_open
    from tqdm import tqdm

    files = sorted(latent_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {latent_dir}")
    stats_path = latent_dir / "latents_stats.pt"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Latent statistics not found: {stats_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    torch.manual_seed(42)
    mean, std = _load_stats(stats_path, device)
    pca_mean, components = _fit_pca(files, mean, std, device=device, dimension=dimension)

    output.parent.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=(OUTPUT_ROWS, dimension),
    )
    current = 0
    for path in tqdm(files, desc="Projecting data"):
        if current >= OUTPUT_ROWS:
            break
        with safe_open(path, framework="pt", device="cpu") as handle:
            latent = handle.get_slice("latents")[:].to(device)
        latent = ((latent - mean) / std).flatten(start_dim=1)
        projected = (latent - pca_mean) @ components.T
        rows = min(int(projected.shape[0]), OUTPUT_ROWS - current)
        target[current : current + rows] = projected[:rows].to(
            device="cpu", dtype=torch.float32
        ).numpy()
        current += rows
    if current != OUTPUT_ROWS:
        raise RuntimeError(f"Raw archive provided {current} rows; expected {OUTPUT_ROWS}.")
    target.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the representative ImageNet PCA artifact.")
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["revision"] = None
    for dimension in OUTPUT_DIMENSIONS:
        entry = manifest["artifacts"][str(dimension)]
        output = output_dir / entry["filename"]
        temporary = output.with_suffix(".npy.part")
        try:
            build_artifact(args.latent_dir.expanduser().resolve(), temporary, dimension)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        entry["sha256"] = _sha256(output)
        entry["size_bytes"] = output.stat().st_size
        print(f"artifact={output} sha256={entry['sha256']}", flush=True)
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (output_dir / "data_manifest.json").write_text(manifest_text, encoding="utf-8")
    MANIFEST_PATH.write_text(manifest_text, encoding="utf-8")
    (output_dir / "README.md").write_text(
        (MANIFEST_PATH.parent / "HUGGING_FACE_DATASET_CARD.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
