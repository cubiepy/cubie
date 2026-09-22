# Completeness and freshness ledger

## Repository coverage

| Receipt | Result |
|---|---|
| Tracked entries | 585 |
| Regular tracked files | 566 |
| Tracked symbolic links | 19 |
| Exact preassigned lane coverage | 524 entries |
| Exact repository-support complement | 61 entries |
| reStructuredText pages | 149 |
| Missing or unreferenced toctree pages | 0 |
| Binary assets | 1 NPZ file, inventoried without text claims |
| Nested systems-lane instruction files | 10 files and 1,068 lines |

Every tracked text file was read by at least one lane. Human-authored prose,
code, tests, and configuration were read line by line. Generated inventories,
schemas, and fixtures were traversed in full and received format-specific
validation. Symbolic-link mirrors were reconciled with their targets.

## Documentation page coverage

| Page group | Pages | Result |
|---|---:|---|
| User-facing, tutorial, theory, and guide pages | 29 | Read and source-checked |
| Developer, batch, and integrator API pages | 69 | Read and source-checked |
| Memory, ODE-system, output, and GUI API pages | 51 | Read and source-checked |
| Total | 149 | Complete |

All 149 pages are reachable from the Sphinx page tree. Fifty-eight autodoc
directives in the batch and integrator lane were checked. Fifty-four resolve
and four name removed objects. Forty-nine directives in the systems and output
lane were checked. One uses the wrong directive type. Seven optional GUI
targets exist statically but could not be imported without `qtpy`.

Sphinx is absent from the system Python and project virtual environment. No
documentation build was run. The exact failed import is recorded in
[systems_output_inventory.md](systems_output_inventory.md).

## Freshness priorities

### Build and navigation

- `docs/source/conf.py` reports release `0.0.4`. `pyproject.toml` reports
  `0.4.0`. See UFR-004.
- The landing page uses a former repository URL and the wrong Reference Manual
  target. See UFR-005.
- Four integrator API directives name removed objects. See INT-02, INT-03, and
  INT-04.
- `current_cupy_stream` is a class documented as a function. See SOD-005.

### Public behavior

- Cache layers, cache routing, invalidation, and compilation claims are stale.
  See UFR-006, UFR-007, UFR-012, and UFR-C01.
- CellML version support and DAE limitations are incorrect. See UFR-008,
  UFR-C03, SOD-013, SOD-014, and SOD-015.
- Algorithm inventories, aliases, controller defaults, and method properties
  are incomplete or incorrect. See UFR-009, UFR-010, INT-05 through INT-11,
  D2-03, and D2-04.
- Memory ownership, stream concurrency, spill behavior, and cleanup contracts
  are incomplete or incorrect. See UFR-014, UFR-C02, API-02, SOD-001 through
  SOD-008, and RB-02.
- Output defaults, result status flags, device-view lifetime, and timing
  aliases are incomplete or incorrect. See UFR-015, UFR-016, UFR-C05, API-03,
  SOD-018 through SOD-032, and RB-03.
- Two checked-in documentation examples call stale or malformed interfaces.
  See UFR-026 and UFR-027.

### Missing coverage

- Public exports lack API pages in the batch, memory, ODE-system, output, and
  integrator packages. See API-01, API-03, INT-15, SOD-034 through SOD-037,
  RB-06, and RB-07.
- Developer guidance omits current factory, buffer, component-registration,
  result-code, and testing contracts. See DG-01 through DG-19 and D2-01 through
  D2-11.
- Seven nested instruction files contain stale Jacobian, system-size, CellML,
  summary-metric, test-map, or import guidance. See SOD-039 through SOD-045.
- `tests/_inventory.json` contains removed symbols and incorrect descriptions.
  Its query tool names a missing regeneration script. See D3-F04 and D3-F05.
- Two profiling scripts contain a nonexistent personal worktree path. See
  D3-F03.

## Language review

Each documentation inventory records em dashes, semicolons, framing colons,
false-drama phrasing, and rendered source docstrings. Syntax required by code,
URLs, RST, math, tables, and protocol strings is marked for retention. Prose
uses have replacements in the paired proposal artifact.

The review also covers prospective autodoc text. New API pages would otherwise
render prohibited punctuation or framing from `BatchInputHandler`,
`DeviceSolveResult`, BiCGSTAB classes, `TimeLogger`, `ArrayInterpolator`,
`load_cellml_model`, `MeanStd`, and `StdRms`. See STYLE-PROSPECTIVE-01, RB-06,
RB-07, SOD-036, and SOD-037.

Two integrator test docstrings contain explicit AI-authorship narratives.
D2-12 provides factual replacements.

## Review order

1. Correct factual contradictions and broken targets.
2. Add missing public API and lifecycle coverage.
3. Apply source-docstring rewrites required by existing and proposed autodoc.
4. Apply page-level language rewrites.
5. Regenerate the stale test inventory and repair its query workflow.
6. Build the complete Sphinx tree in the documented development environment.
