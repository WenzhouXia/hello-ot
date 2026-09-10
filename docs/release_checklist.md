# Release checklist

## Repository and distribution boundaries

- `experiments/`: exploratory work.
- `paper_experiments/`: internal paper experiments and shared baseline modules.
- `export_experiments/`: public reproduction entry points, retaining their paths after export.
- GitHub contains public experiments and their selected shared dependencies.
- Wheel and sdist serve package installation/building; full experiment reproduction
  requires the GitHub repository. The sdist excludes experiments and repository tests.

Public squared-L2 experiments explicitly use cost perturbation off and seed=42.
The library API retains its default auto policy.

## CPU checks

GitHub Actions runs portable API, perturbation, explicit algorithm structure,
variant interface, public experiment and export tests on Python 3.12 with PyTorch 2.7.1.
GPU-only tests are skipped there. Build a portable wheel from the sdist to check
that the source archive has all required package/build files.

## Release tasks

- [x] Review AGY's migration benchmark and resolve correctness/performance findings.
- [x] Migrate internal `hello_solver.py` and the EMD warm-start call chain to the
  public HELLO API after that validation is stable.
- [ ] Run small GPU release checks using the current source/binary combination.
- [ ] Validate all four public experiment entry points with actual small GPU runs.
- [ ] Rebuild final native wheels from the final source revision.
- [ ] Install the final wheels in an isolated environment and repeat GPU checks.
- [ ] Complete extended GW/UOT/SDOT numerical-equivalence checks as appropriate.
- [ ] Record wheel checksums, versions, GPU/driver and validation outputs before publishing.

No GPU validation or native rebuild is performed as part of the CPU release cleanup.
The internal MDOT budgeted/PyPI experiment, Zanetti–Gondzio IPM and
Neufeld–Xiang cutting-plane baseline are outside the public export scope.

## GPU commands to run later

From a public repository checkout with the intended native wheel already installed:

```bash
bash scripts/create_hello_ot_env.sh hello_ot
bash scripts/install_native.sh hello_ot
bash scripts/install_jax_gpu.sh hello_ot
```

The three cumulative installers provide the portable PyTorch package, the native
wheel, and GPU JAX/reproduction dependencies in one environment. Then run:

```bash
python3 scripts/validate_gpu_release.py
python3 scripts/validate_gpu_release.py --extended
```

Use `CUDA_VISIBLE_DEVICES` to select an appropriate GPU. Add `--with-ott` only
when testing the optional OTT initialization with its GPU dependencies installed.
The script checks runtime availability, performs no install/build, and writes JUnit
results. A missing CUDA runtime or fixture is an error, not a passing skipped run.
The default covers four costs, on/off/auto perturbation and small GW/UOT/SDOT cases.
`--extended` adds the 32k fixtures. These are correctness checks, not timing claims.

The standalone `scripts/validate_release.py` remains an installed-package smoke
check; it is not a replacement for the full GPU suite.
