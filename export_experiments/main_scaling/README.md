# Main Scaling

## Overview

This experiment reproduces the HELLO large-scale scaling results in **Table 2** of the paper using a single seed (`seed=42`).

It evaluates native HELLO on Gaussian-to-ImageNet feature transport across sample sizes $n \in \{2^{14}, \dots, 2^{18}\}$ and dimensions $d \in \{4, 32, 256, 2048\}$.

## Running the Experiment

### Quick Test

Run the smallest configuration ($n=16384, d=4$):

```bash
python3 -m export_experiments.main_scaling.run --n 16384 --d 4 --seed 42
```

### Full Benchmark

Run the complete matrix ($n=2^{14} \sim 2^{18}$, four dimensions):

```bash
python3 -m export_experiments.main_scaling.run
```

Results are saved to `results/main_scaling/`.

## Dataset & Caching

The benchmark features are hosted on Hugging Face (`WenzhouXia/hello-ot-imagenet-pca`). On first run, the requested dimension files are automatically downloaded, verified against SHA-256 checksums, and cached under `~/.cache/hello_ot/`.

To use a pre-downloaded or local feature file, pass `--feature-file <path>` with a single `--d`.
