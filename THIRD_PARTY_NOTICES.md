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

HiRef is an optional paper baseline obtained directly from its official
repository at <https://github.com/raphael-group/HiRef>. HELLO does not
redistribute the HiRef source because that upstream repository does not
currently expose a license file. HiRef is not imported by the `hello_ot`
package and is not included in the wheel.

The experiment-only MDOT-TNT source at
`paper_experiments/baselines/linear_ot/mdot_tnt/` is from
`metekemertas/mdot_tnt`, commit
`6f5aea26cd1cfe30012b9fd31e6c4eae49c25f69`.
It retains the upstream PolyForm Noncommercial License 1.0.0 and Required
Notice in that directory's LICENSE. The shared adapter implements the KeOps
path. These experiment files are not covered by HELLO's Apache license and
are not included in the hello_ot wheel.

The IPOT experiment adapter follows POT 0.9.7.post1's log-domain recurrence,
with its sign correction documented in the source, and adds KeOps execution.
POT is distributed under the MIT License; the experiment adapter retains its
source attribution. IPOT is not part of the hello_ot package.
The POT license and copyright notice are preserved in
`paper_experiments/baselines/linear_ot/POT_LICENSE`.
