# Algorithm map

The production path in `src/hello_ot/algorithm.py` follows the paper operators directly:

1. **BuildHierarchy** creates the dual-guided hierarchy with one global shuffle and repeated splitting of the larger side.
2. The **coarsest solve** obtains the first transport plan and dual potentials.
3. **Initialization** propagates one inherited dual side, completes the other by a c-transform, performs bidirectional dual assignment, and augments the resulting candidates with a feasible northwest-corner basis.
4. **Refinement** repeatedly solves the restricted OT problem, checks full-edge dual feasibility, and updates the active support through dual-violation insertion and budgeted pruning.

The top-level orchestration intentionally stays in one file. CUDA scan details, sparse support storage, CuPDLPx bindings, memory planning, and diagnostics live in deeper modules so that they do not obscure the paper-level control flow.

The implementation uses paper terminology in names and documentation. `TERMINOLOGY.md` is the authoritative paper-to-code glossary.
