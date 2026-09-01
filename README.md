# HELLO

HELLO is a solver for large-scale balanced optimal transport on point clouds. It localizes a sparse active support through a dual-guided hierarchy, solves restricted transport problems, and certifies convergence on the full edge set. The default `native` backend is the CUDA implementation used for paper experiments; an explicitly selected portable `torch` backend runs from ordinary PyTorch operators on CPU or CUDA.

The implementation supports squared Euclidean, L1, L2, and L-infinity costs. Its Python entry point is deliberately small:

```python
import numpy as np
import hello_ot

rng = np.random.default_rng(0)
source = rng.normal(size=(2048, 32))
target = rng.normal(size=(2048, 32))

result = hello_ot.solve(
    source,
    target,
    options=hello_ot.SolverOptions(backend="torch", torch_device="auto"),
)
print(result.objective)
print(result.solution.shape, result.solution.values.size)
```

Array inputs use uniform masses and squared Euclidean cost by default. To specify masses or another supported cost:

```python
result = hello_ot.solve(
    source,
    target,
    source_mass=source_mass,
    target_mass=target_mass,
    cost="l1",
)
```

Most users do not need a configuration object. Advanced experiments can change the few supported algorithmic options through `hello_ot.SolverOptions`; the LP tolerance and full dual-feasibility tolerance are fixed to the paper setting.

## Installation

The default installation is pure Python and requires no compiler. It installs the portable PyTorch backend; selecting that backend is always explicit, so a broken native installation can never silently change performance.

```bash
python3 -m pip install .
python3 examples/quickstart.py
```

The quickstart uses `backend="torch"` on a non-toy `2048 x 2048`, 32-dimensional problem. It used about 700 MiB host RSS and 7.5 seconds end-to-end on a 64-core server CPU; ordinary laptops may take longer. Its timings are not representative of the native paper backend.

`SolverOptions()` deliberately keeps `backend="native"`. On a machine prepared for paper experiments, build the bundled C++/CUDA extensions explicitly:

```bash
python3 -m pip install '.[native-build]'
scripts/build_native_wheel.sh dist
```

The `v0.1.0-rc2` native wheels target Linux x86-64 with glibc 2.29 or newer, Python 3.10/3.11, PyTorch 2.5.1+cu118, and CUDA compute capabilities 8.0, 8.6, 8.9, and 9.0, with PTX at 9.0 for forward compatibility. This covers A100, RTX 3080 Ti/3090, RTX 4060 Ti/4090, and H100. These release-candidate wheels use the platform-specific `linux_x86_64` tag rather than a manylinux tag. Building from source needs CUDA 11.8, a compatible C++ compiler, Ninja, and pybind11; installing a prebuilt wheel does not.

Install PyTorch first, then the wheel matching the Python version from the GitHub Release:

```bash
python3 -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install --no-deps --only-binary=:all: ./hello_ot-0.1.0rc2-*.whl
python3 -m hello_ot.diagnose --require-native
python3 examples/verify_native.py
```

WSL2 uses the Linux wheel and the NVIDIA driver supplied by the Windows host. Do not install a Linux NVIDIA display driver inside WSL2. Native Windows Python is not supported by this release.

The portable `torch` backend may run with newer PyTorch versions allowed by the package metadata. The prebuilt native wheel is ABI-bound to the release matrix above and fails before loading its extensions when the PyTorch/CUDA runtime does not match.

For a shareable release-validation report:

```bash
python3 scripts/validate_release.py --backend both --output hello_ot_validation.json
```

Run the release test suite with:

```bash
python3 -m pytest -q tests/test_hello_ot_public_api.py \
  tests/test_hello_ot_dependency_boundary.py \
  tests/test_hello_ot_numerical_equivalence.py \
  tests/release_tooling_test.py \
  tests/torch_backend_end_to_end_test.py \
  tests/torch_packaging_test.py \
  tests/torch_prewarm_test.py \
  tests/torch_restricted_ot_pdlp_test.py \
  tests/torch_scan_test.py \
  tests/hello_inplace_feature_layout_test.py
```

The source layout is:

```text
src/hello_ot/
  algorithm.py       paper-level hierarchy and refinement orchestration
  initialization/    dual propagation, dual assignment, feasible support
  refinement/        optimality checks and active-support updates
  restricted_ot/     restricted LP construction and solve
  kernels/           Python interfaces to resident-streamed CUDA scans
  _native/           CuPDLPx and custom C++/CUDA extension sources
```

See `docs/algorithm.md` for the paper-to-code map and `experiments/README.md` for the reproducibility entry points.

## License

HELLO is released under the Apache License 2.0. Bundled third-party components and their licenses are listed in `THIRD_PARTY_NOTICES.md`.
