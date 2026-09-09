# Exactness verification

This experiment uses a strongly correlated synthetic Brenier problem with a
known optimal permutation. The representative matrix is defined in
`paper_matrix.json`.

Run the complete matrix:

```bash
python3 -m export_experiments.exactness_verification.run
```

Run one case:

```bash
python3 -m export_experiments.exactness_verification.run --n 16384 --d 4 --seed 42 --method hello
```

HELLO always uses the native paper backend. HiRef is an optional third-party
checkout; clone its official repository into `third_party/HiRef`. JAX and
OTT-JAX are optional. Unavailable optional methods are skipped explicitly.
