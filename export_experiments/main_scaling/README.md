# Main scaling

This representative experiment solves Gaussian-to-ImageNet-feature OT problems
for `N=2^14,...,2^18`, dimensions `4, 32, 256, 2048`, and seed=42 only.
The native backend is always used.

Run the complete matrix:

```bash
python3 -m export_experiments.main_scaling.run
```

Run the smallest case (the same experiment, not a separate smoke preset):

```bash
python3 -m export_experiments.main_scaling.run --n 16384 --d 4 --seed 42
```

Each dimension has an independently fitted/projected file; no column slicing
from the 2048-dimensional artifact is used. The four files total about 2.29 GiB.
Files are cached under `~/.cache/hello_ot`. Pass `--feature-file` with exactly
one `--d` to use a local file of that dimension.

## Prepare the release

Run from the repository root in an environment with Torch, NumPy, safetensors
and tqdm. A working CUDA device is recommended: PCA uses 50,000 flattened
8192-dimensional latents. The script independently fits each dimension with
seed=42, then projects the first 262144 positive training latents.

```bash
python3 -m export_experiments.main_scaling.prepare_artifact \
  --latent-dir data/imagenet1k/latents_imagenet_1k_vl_enriched/imagenet_1k_vl_enriched/train_256 \
  --output-dir data/hello_ot_public_seed42
```

This writes four NPY files, a generated checksum manifest and README into the
output directory, and updates the repository's data manifest. Do not upload an
incomplete output directory. Existing release hashes are not reused.

## Upload and pin

Install `huggingface_hub`, then log in and upload the completed directory:

```bash
hf auth login
hf upload WenzhouXia/hello-ot-imagenet-pca data/hello_ot_public_seed42 . --repo-type dataset
```

Ensure the dataset is public and ungated. Copy the upload commit SHA into
`export_experiments/main_scaling/data_manifest.json` as `revision` (replacing null).
Keep the generated file checksums. Commit this updated manifest with the code.

Verify anonymous downloading, preferably with an empty cache:

```bash
HELLO_OT_DATA_DIR=/tmp/hello_ot_download_check python3 -m export_experiments.main_scaling.data
```

The downloader uses unauthenticated requests and verifies every file's SHA-256.
Before generation/pinning it fails explicitly instead of using the old artifact.
No public artifact is required for N greater than 262144.
