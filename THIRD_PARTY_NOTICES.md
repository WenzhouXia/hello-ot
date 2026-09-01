# Third-party notices

HELLO includes modified source code from **CuPDLPx**, distributed under the
Apache License 2.0. Its license is preserved at
`src/hello_ot/_native/cupdlpx/LICENSE`.

The bundled CuPDLPx binding and solver sources were adapted to build as the
private module `hello_ot._native.pycupdlpx`. The public package does not bundle
the pybind11 source tree; pybind11 2.12 is used as a build dependency under its
BSD-3-Clause license.

Python dependencies retain their respective upstream licenses and are not
vendored in this repository.
