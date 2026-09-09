---
license: other
pretty_name: HELLO representative ImageNet VA-VAE PCA features
---

# HELLO representative ImageNet VA-VAE PCA features

This dataset contains four independently generated `float32` NumPy matrices,
with shapes `(262144, d)` for `d = 4, 32, 256, 2048`, totaling about 2.29 GiB.
All use seed=42 and serve the public main-scaling and parameter-sensitivity
export_experiments. Larger sample counts are outside the published data scope.

The artifact contains numeric PCA-projected features only. It contains no
images, labels, captions, filenames, or ImageNet identifiers. It is intended
for non-commercial research and educational reproducibility.

## Provenance

- Source dataset: `visual-layer/imagenet-1k-vl-enriched`
- Source revision: `ac6afcdeb3be31c5ff6a7ff579874b3d372b7074`
- Encoder family: LightningDiT VA-VAE f16d32
- Raw latent shape: `32 x 16 x 16`
- Standardization: channel statistics stored with the original latent archive
- PCA fit: independently for each requested dimension, 50,000 standardized
  positive training latents; sorted shard paths shuffled with Python random seed=42
- Published rows: the first 262,144 projected training latents
- Published dimensions: separate files for 4, 32, 256, and 2048 components
- Post-PCA normalization: disabled

The original extraction workspace did not preserve the exact VA-VAE checkpoint
hash or LightningDiT commit. The encoder family is identified from the retained
latent archive and its `32 x 16 x 16` tensor shape. This limitation is recorded
explicitly rather than assigning an unsupported checkpoint identifier.

## Integrity

Files follow `imagenet_vavae_pca_n262144_d{d}_seed42.npy`.
The generated `data_manifest.json` records each file's shape, byte size and
SHA-256. The HELLO repository verifies each downloaded file against that manifest.
The release commit is pinned in the consuming HELLO repository after upload.
Gaussian source points are generated locally using NumPy FP32 standard_normal
with seed=42 for each requested `(N, d)`; they are not stored in this dataset.

## Upstream terms

The artifact is derived from ImageNet data. Users should use it only for
non-commercial research or educational purposes and follow the ImageNet access
terms. The LightningDiT code repository is distributed under the MIT License.
