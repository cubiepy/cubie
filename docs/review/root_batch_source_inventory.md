# Root and batch source coverage inventory

## Scope and receipt

This inventory covers the exact union of these tracked pathspecs.

- Root-level `src/cubie/*.py`
- Every tracked file under `src/cubie/batchsolving/**`, excluding
  `CLAUDE.md` symlink duplicates
- Root-level `tests/*.py`
- Every tracked file under `tests/batchsolving/**`

The manifest was built with `git ls-files`, sorted, and deduplicated. Every
byte was decoded and every physical line was visited. Every non-empty Python
file was also parsed into an AST and checked against the current Sphinx pages,
the three existing review inventories, and the three existing rewrite-proposal
artifacts.

Receipt: **70 tracked text files, 45,405 physical lines, 0 binary assets**.
The per-file counts below sum to 45,405 and the rows sum to 70.

Instruction receipt: `AGENTS.md` (126 lines), `src/cubie/AGENTS.md`
(210 lines), `src/cubie/batchsolving/AGENTS.md` (160 lines), and
`src/cubie/batchsolving/arrays/AGENTS.md` (141 lines) were read in full before
classification. The two batch instruction files are also rows in the scoped
manifest.

`Existing` means the documentation impact is already recorded in one of the
six existing review artifacts. `Supplemental` points to a new proposal in
`root_batch_supplemental_proposals.md`. `None` means the line review found no
additional documentation impact.

## Manifest

| Tracked file | Physical lines | Coverage | Documentation impact |
|---|---:|---|---|
| `src/cubie/__init__.py` | 56 | READ-COMPLETE | Supplemental RB-06; confirms the undocumented top-level `TimeLogger` and `default_timelogger` exports. |
| `src/cubie/_env.py` | 180 | READ-COMPLETE | Existing UFR-C01. |
| `src/cubie/_mlir_compat.py` | 2,896 | READ-COMPLETE | None; internal backend shims agree with the package-root architecture mirror. |
| `src/cubie/_numba_cuda_compat.py` | 1,057 | READ-COMPLETE | None; internal backend shims agree with the package-root architecture mirror. |
| `src/cubie/_serialize.py` | 220 | READ-COMPLETE | Existing DG-09 and the package-root architecture mirror cover canonical config hashing. |
| `src/cubie/_utils.py` | 793 | READ-COMPLETE | None beyond existing developer-guide configuration findings. |
| `src/cubie/array_interpolator.py` | 1,151 | READ-COMPLETE | Supplemental RB-07; the documented direct-use class has no API-reference page, the prospective member list omitted `evaluation_function` and `coefficients`, and `update_from_dict` contains Markdown that is invalid in rendered RST. Existing UFR-013 and UFR-026 cover the stale timing key and example. |
| `src/cubie/batchsolving/__init__.py` | 74 | READ-COMPLETE | Existing API-03; public batch exports are missing from the page tree. |
| `src/cubie/batchsolving/_utils.py` | 47 | READ-COMPLETE | None; docstring-only internal module. |
| `src/cubie/batchsolving/AGENTS.md` | 160 | READ-COMPLETE | Supplemental RB-01 and RB-02. |
| `src/cubie/batchsolving/arrays/__init__.py` | 0 | READ-COMPLETE | None; tracked empty file. |
| `src/cubie/batchsolving/arrays/AGENTS.md` | 141 | READ-COMPLETE | None; source and tests confirm the documented ownership, chunking, staging, and loan lifecycle. |
| `src/cubie/batchsolving/arrays/BaseArrayManager.py` | 1,185 | READ-COMPLETE | Existing API-02, STYLE-DOC-02, and STYLE-DOC-03. |
| `src/cubie/batchsolving/arrays/BatchInputArrays.py` | 585 | READ-COMPLETE | Existing API-02, STYLE-DOC-02, and STYLE-DOC-03. |
| `src/cubie/batchsolving/arrays/BatchOutputArrays.py` | 581 | READ-COMPLETE | Existing API-02, STYLE-DOC-02, and STYLE-DOC-03. |
| `src/cubie/batchsolving/BatchInputHandler.py` | 1,423 | READ-COMPLETE | Existing API-03; the public class lacks a page. STYLE-PROSPECTIVE-01 records its prospective autodoc punctuation and framing rewrites. |
| `src/cubie/batchsolving/BatchSolverConfig.py` | 199 | READ-COMPLETE | Supplemental RB-01; existing UFR-012 and STYLE-DOC-02/03 cover user-facing kernel keys and punctuation. |
| `src/cubie/batchsolving/BatchSolverKernel.py` | 1,663 | READ-COMPLETE | Existing API-03 and STYLE-DOC-01/02/03. |
| `src/cubie/batchsolving/solver.py` | 1,324 | READ-COMPLETE | Supplemental RB-03 and RB-04; existing API-01/03 and UFR findings cover the other public behavior. |
| `src/cubie/batchsolving/solveresult.py` | 1,080 | READ-COMPLETE | Supplemental RB-02. Existing result findings cover `SolveResult`; STYLE-PROSPECTIVE-01 records the prospective `DeviceSolveResult` autodoc findings. |
| `src/cubie/batchsolving/SystemInterface.py` | 513 | READ-COMPLETE | Supplemental RB-05; existing STYLE-DOC-03 covers the rendered `update` wording. |
| `src/cubie/batchsolving/writeback_watcher.py` | 407 | READ-COMPLETE | None; internal asynchronous transfer service is covered by the batch architecture mirror. |
| `src/cubie/buffer_registry.py` | 1,254 | READ-COMPLETE | Existing DG-11 through DG-13. |
| `src/cubie/cache_root.py` | 78 | READ-COMPLETE | Existing UFR-007 and UFR-C01. |
| `src/cubie/cubie_cache.py` | 889 | READ-COMPLETE | Existing UFR-006, UFR-007, and UFR-012. |
| `src/cubie/cuda_backend.py` | 123 | READ-COMPLETE | Existing UFR-001, UFR-002, and UFR-C01. |
| `src/cubie/cuda_simsafe.py` | 734 | READ-COMPLETE | Existing UFR-C01 and UFR-029. |
| `src/cubie/CUDAFactory.py` | 878 | READ-COMPLETE | Existing DG-07 through DG-10 and UFR-029. |
| `src/cubie/result_codes.py` | 91 | READ-COMPLETE | Existing UFR-016, UFR-C05, INT-02, and INT-15. |
| `src/cubie/time_logger.py` | 803 | READ-COMPLETE | Supplemental RB-03 and RB-06. RB-06 includes the semicolon and framing rewrites that its proposed `:members:` page would expose. Existing UFR-021 and UFR-022 cover capability claims. |
| `tests/__init__.py` | 0 | READ-COMPLETE | None; tracked empty file. |
| `tests/_precompile_hashing.py` | 146 | READ-COMPLETE | None; test helper only. |
| `tests/_utils.py` | 2,405 | READ-COMPLETE | None; shared fixtures and test helpers only. |
| `tests/banked_plugin.py` | 67 | READ-COMPLETE | None; pytest plugin helper only. |
| `tests/batchsolving/arrays/test_basearraymanager.py` | 2,533 | READ-COMPLETE | Confirms existing API-02 and array architecture findings; no additional documentation impact. |
| `tests/batchsolving/arrays/test_batchinputarrays.py` | 305 | READ-COMPLETE | Confirms existing API-02 and device-input lifecycle findings; no additional documentation impact. |
| `tests/batchsolving/arrays/test_batchoutputarrays.py` | 565 | READ-COMPLETE | Confirms existing API-02 and output-allocation findings; no additional documentation impact. |
| `tests/batchsolving/arrays/test_chunking.py` | 755 | READ-COMPLETE | Confirms existing UFR-014, UFR-C02, and API-02; no additional documentation impact. |
| `tests/batchsolving/arrays/test_managed_array.py` | 52 | READ-COMPLETE | Confirms the arrays architecture mirror; no additional documentation impact. |
| `tests/batchsolving/test_batch_input_handler.py` | 1,076 | READ-COMPLETE | Confirms existing API-03 and batching behavior; no additional documentation impact. |
| `tests/batchsolving/test_BatchSolverConfig.py` | 211 | READ-COMPLETE | Confirms Supplemental RB-01 and existing ActiveOutputs findings. |
| `tests/batchsolving/test_config_plumbing.py` | 795 | READ-COMPLETE | Confirms Supplemental RB-04 and existing UFR-012/UFR-C04 configuration findings. |
| `tests/batchsolving/test_memory_pressure.py` | 391 | READ-COMPLETE | Confirms Supplemental RB-04 and existing UFR-014/UFR-C02 memory findings. |
| `tests/batchsolving/test_runparams.py` | 318 | READ-COMPLETE | Confirms the batch architecture mirror; no additional documentation impact. |
| `tests/batchsolving/test_solver_teardown.py` | 341 | READ-COMPLETE | Confirms Supplemental RB-02 and existing UFR-C02 cleanup findings. |
| `tests/batchsolving/test_solver.py` | 2,042 | READ-COMPLETE | Confirms Supplemental RB-03/RB-04 and the existing API/UFR solver findings. |
| `tests/batchsolving/test_solveresult.py` | 798 | READ-COMPLETE | Confirms Supplemental RB-02 and existing result findings. |
| `tests/batchsolving/test_SolverKernel.py` | 559 | READ-COMPLETE | Confirms existing kernel API and timing-validation findings; no additional documentation impact. |
| `tests/batchsolving/test_system_interface.py` | 382 | READ-COMPLETE | Confirms Supplemental RB-05. |
| `tests/batchsolving/test_utils.py` | 27 | READ-COMPLETE | None; internal helper test only. |
| `tests/batchsolving/test_writeback_watcher.py` | 564 | READ-COMPLETE | Confirms the batch architecture mirror; no additional documentation impact. |
| `tests/conftest.py` | 1,644 | READ-COMPLETE | Existing DG-17 through DG-19 and UFR-028 cover test setup and fixture policy. |
| `tests/numba_cache_locator.py` | 30 | READ-COMPLETE | None; subprocess cache helper only. |
| `tests/precompile_plugin.py` | 633 | READ-COMPLETE | None; pytest precompile plugin only. |
| `tests/query_inventory.py` | 130 | READ-COMPLETE | None; test-selection inventory helper only. |
| `tests/system_fixtures.py` | 697 | READ-COMPLETE | Existing DG-18, DG-19, and UFR-028 cover fixture use. |
| `tests/test_array_interpolator.py` | 1,406 | READ-COMPLETE | Confirms Supplemental RB-07 and existing UFR-013/UFR-026. |
| `tests/test_buffer_registry.py` | 1,200 | READ-COMPLETE | Confirms existing DG-11 through DG-13. |
| `tests/test_cache_config.py` | 692 | READ-COMPLETE | Confirms existing UFR-007 and UFR-012. |
| `tests/test_cache_root.py` | 101 | READ-COMPLETE | Confirms existing UFR-007 and UFR-C01. |
| `tests/test_cubie_cache.py` | 791 | READ-COMPLETE | Confirms existing UFR-006, UFR-007, and UFR-012. |
| `tests/test_cuda_simsafe.py` | 60 | READ-COMPLETE | Confirms existing UFR-C01 and UFR-029. |
| `tests/test_CUDAFactory.py` | 896 | READ-COMPLETE | Confirms existing DG-07 through DG-10. |
| `tests/test_env.py` | 101 | READ-COMPLETE | Confirms existing UFR-C01 and Supplemental RB-03's disabled timing default. |
| `tests/test_package_source_hash.py` | 69 | READ-COMPLETE | None beyond existing cache identity coverage. |
| `tests/test_precompile_hashing.py` | 42 | READ-COMPLETE | None beyond existing cache identity coverage. |
| `tests/test_result_codes.py` | 59 | READ-COMPLETE | Confirms existing UFR-016, UFR-C05, and INT-02. |
| `tests/test_serialize.py` | 212 | READ-COMPLETE | Confirms existing DG-09 and the package-root architecture mirror. |
| `tests/test_time_logger.py` | 851 | READ-COMPLETE | Confirms Supplemental RB-03/RB-06 and existing UFR-021/UFR-022. |
| `tests/test_utils.py` | 874 | READ-COMPLETE | None beyond existing developer-guide configuration findings. |

## Binary and non-text assets

NONE. All 70 scoped tracked files are text.

## Reconciliation

- Manifest rows: 70
- Text files: 70
- Binary or non-text assets: 0
- Physical lines: 45,405
- READ-COMPLETE rows: 70
- Unreconciled paths or lines: 0
