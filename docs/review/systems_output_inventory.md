# Systems and output documentation inventory

## Audit totals

| Corpus | Files | Lines | Status |
| --- | ---: | ---: | --- |
| Scoped reStructuredText pages | 51 | 695 | Every line read |
| Rendered autodoc source docstrings | 261 class or member objects plus 5 functions | 2,959 | Every line read |
| Prospective autodoc docstrings for missing pages | 3 objects | 105 | Every line read; rewrites recorded in SOD-036 and SOD-037 |
| Memory source | 6 | 3,129 | Line scan and public behavior audit complete |
| ODE-system source | 53 | 24,782 | Line scan and public behavior audit complete |
| Output-handling source | 27 | 6,092 | Line scan and public behavior audit complete |
| GUI source | 3 | 873 | Line scan and static target audit complete |
| Vendored source | 11 | 4,271 | Line scan complete, no CuBIE public API |
| Relevant tests | 50 | 21,456 | Line scan and cited behavior audit complete |
| Engine `CLAUDE.md` mirror | 1 | 1 | Read and compared with its 80-line `AGENTS.md` target |
| Regular nested instruction files | 10 | 1,068 | Every line read and checked against source, tests, and documentation requirements |

The source total is 39,147 lines. The audit used the current working tree,
the code knowledge graph, package export lists, source definitions, and tests.

Assigned-entry reconciliation is 51 RST pages, ten regular `AGENTS.md` files,
and one regular `CLAUDE.md` mirror. These 62 lane entries are included in the
parent review's 524 assigned entries.

## Autodoc resolution

There are 49 autodoc directives in scope.

* 41 targets resolved at runtime with
  `NUMBA_ENABLE_CUDASIM=1` under `.venv`.
* `current_cupy_stream` resolved as a class but is declared with
  `autofunction`. This is SOD-005.
* Seven GUI targets were verified statically in source. Runtime import was
  blocked by the absent optional `qtpy` package.
* No other target was missing or had a directive-type mismatch.

## Page inventory

`Read` is `YES` for every row. `Runtime OK` means the module and target
resolved with the directive's expected object type. `Static OK` means the
definition exists but the optional GUI dependency prevented import.

| Page | Lines | Read | Autodoc status | Freshness and completeness |
| --- | ---: | --- | --- | --- |
| `docs/source/API_reference/gui/constants_editor.rst` | 13 | YES | Static OK, `qtpy` absent | CLEAN |
| `docs/source/API_reference/gui/index.rst` | 14 | YES | N/A | CLEAN |
| `docs/source/API_reference/gui/pre_parse_editor.rst` | 10 | YES | Static OK, `qtpy` absent | CLEAN |
| `docs/source/API_reference/gui/states_editor.rst` | 10 | YES | Static OK, `qtpy` absent | CLEAN |
| `docs/source/API_reference/memory.rst` | 69 | YES | N/A | SOD-001, SOD-002, SOD-003, SOD-004, SOD-034, SOD-035 |
| `docs/source/API_reference/memory/array_request.rst` | 9 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/memory/array_response.rst` | 9 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/memory/current_cupy_stream.rst` | 6 | YES | Type mismatch | SOD-005 |
| `docs/source/API_reference/memory/default_memmgr.rst` | 6 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/memory/memory_manager.rst` | 8 | YES | Runtime OK | SOD-006, SOD-007, SOD-033 |
| `docs/source/API_reference/memory/stream_groups.rst` | 9 | YES | Runtime OK | SOD-008, SOD-033 |
| `docs/source/API_reference/odesystems/base_ode.rst` | 8 | YES | Runtime OK | SOD-017, SOD-033 |
| `docs/source/API_reference/odesystems/create_ode_system.rst` | 6 | YES | Runtime OK | SOD-013, SOD-014, SOD-015, SOD-016 |
| `docs/source/API_reference/odesystems/index.rst` | 62 | YES | N/A | SOD-009, SOD-010, SOD-011, SOD-036 |
| `docs/source/API_reference/odesystems/ode_cache.rst` | 8 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/odesystems/ode_data.rst` | 8 | YES | Runtime OK | SOD-033 |
| `docs/source/API_reference/odesystems/symbolic.rst` | 30 | YES | N/A | SOD-012 |
| `docs/source/API_reference/odesystems/symbolic_ode.rst` | 8 | YES | Runtime OK | SOD-016, SOD-033 |
| `docs/source/API_reference/odesystems/system_sizes.rst` | 8 | YES | Runtime OK | SOD-010 |
| `docs/source/API_reference/odesystems/system_values.rst` | 8 | YES | Runtime OK | SOD-033 |
| `docs/source/API_reference/outputhandling/batch_input_sizes.rst` | 9 | YES | Runtime OK | SOD-029 |
| `docs/source/API_reference/outputhandling/batch_output_sizes.rst` | 9 | YES | Runtime OK | SOD-029 |
| `docs/source/API_reference/outputhandling/index.rst` | 75 | YES | N/A | SOD-018, SOD-019, SOD-020, SOD-021 |
| `docs/source/API_reference/outputhandling/output_array_heights.rst` | 9 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/output_compile_flags.rst` | 9 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/output_config.rst` | 8 | YES | Runtime OK | SOD-026, SOD-027, SOD-028 |
| `docs/source/API_reference/outputhandling/output_function_cache.rst` | 9 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/output_functions.rst` | 8 | YES | Runtime OK | SOD-030 |
| `docs/source/API_reference/outputhandling/register_metric.rst` | 6 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/single_run_output_sizes.rst` | 9 | YES | Runtime OK | SOD-029 |
| `docs/source/API_reference/outputhandling/summary_metrics.rst` | 6 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/summarymetrics.rst` | 75 | YES | N/A | SOD-022, SOD-023, SOD-024, SOD-025, SOD-037 |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/d2xdt2_extrema.rst` | 8 | YES | Runtime OK | SOD-032 |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/d2xdt2_max.rst` | 8 | YES | Runtime OK | SOD-032 |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/d2xdt2_min.rst` | 8 | YES | Runtime OK | SOD-032 |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/dxdt_extrema.rst` | 8 | YES | Runtime OK | SOD-032 |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/dxdt_max.rst` | 8 | YES | Runtime OK | SOD-032 |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/dxdt_min.rst` | 8 | YES | Runtime OK | SOD-032 |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/extrema.rst` | 8 | YES | Runtime OK | SOD-032 |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/max.rst` | 8 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/max_magnitude.rst` | 8 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/mean.rst` | 8 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/mean_std_rms.rst` | 8 | YES | Runtime OK | SOD-032 |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/metric_func_cache.rst` | 9 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/min.rst` | 8 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/negative_peaks.rst` | 8 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/peaks.rst` | 8 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/rms.rst` | 8 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/std.rst` | 8 | YES | Runtime OK | SOD-032 |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/summary_metric.rst` | 9 | YES | Runtime OK | CLEAN |
| `docs/source/API_reference/outputhandling/summarymetrics/metrics/summary_metrics.rst` | 8 | YES | Runtime OK | SOD-031 |
| `src/cubie/odesystems/symbolic/engine/CLAUDE.md` | 1 | YES | N/A | SOD-038 |

## Regular instruction-file manifest

All line ranges are inclusive. `CURRENT` means the factual instructions match
the cited source and tests. `STALE` identifies the exact proposal that restores
freshness. Every row also carries SOD-046 for the corpus-wide punctuation and
emphasis rewrite.

| Path and exact range | Lines | Freshness status | Language status | Evidence and proposal |
| --- | ---: | --- | --- | --- |
| `src/cubie/gui/AGENTS.md:1-31` | 31 | CURRENT | STYLE | Definitions at `src/cubie/gui/constants_editor.py:37` and `src/cubie/gui/states_editor.py:29`; no `tests/gui` module in the code graph. SOD-046 |
| `src/cubie/memory/AGENTS.md:1-165` | 165 | STALE at `:149-154` | STYLE | Memory implementation and four test modules checked. Duplicate test entry corrected by SOD-039. SOD-046 |
| `src/cubie/odesystems/AGENTS.md:1-124` | 124 | STALE at `:6-13`, `:22`, `:124` | STYLE | Jacobian IR at `src/cubie/odesystems/symbolic/codegen/jacobian.py:233`; count consumers at `src/cubie/integrators/SingleIntegratorRunCore.py:161` and `src/cubie/outputhandling/output_sizes.py:274`. SOD-040, SOD-046 |
| `src/cubie/odesystems/symbolic/AGENTS.md:1-117` | 117 | STALE at `:117` | STYLE | Generated header at `src/cubie/odesystems/symbolic/odefile.py:20-23`. SOD-041, SOD-046 |
| `src/cubie/odesystems/symbolic/codegen/AGENTS.md:1-147` | 147 | STALE at `:6-21`, `:145-147` | STYLE | Jacobian returns at `src/cubie/odesystems/symbolic/codegen/jacobian.py:233` and `:297`; generated header at `src/cubie/odesystems/symbolic/odefile.py:20-23`. SOD-042, SOD-046 |
| `src/cubie/odesystems/symbolic/parsing/AGENTS.md:1-127` | 127 | STALE at `:80-88`, `:111-115`, `:124-127` | STYLE | Vendored import at `src/cubie/odesystems/symbolic/parsing/cellml.py:24-36`; core dependencies at `pyproject.toml:29-37`. SOD-043, SOD-046 |
| `src/cubie/odesystems/symbolic/structural/AGENTS.md:1-86` | 86 | CURRENT | STYLE | Structural implementation and five matching test modules checked. SOD-046 |
| `src/cubie/outputhandling/AGENTS.md:1-104` | 104 | STALE at `:102-104` | STYLE | Backend-neutral imports at `src/cubie/outputhandling/save_state.py:19`, `update_summaries.py:29`, and `save_summaries.py:29`. SOD-044, SOD-046 |
| `src/cubie/outputhandling/summarymetrics/AGENTS.md:1-114` | 114 | STALE at `:93-94`, `:112-114` | STYLE | Shift lifecycle at `src/cubie/outputhandling/summarymetrics/std.py:103` and `:152`; backend-neutral import at `mean.py:16`. SOD-045, SOD-046 |
| `src/cubie/vendored/AGENTS.md:1-53` | 53 | CURRENT | STYLE | Snapshot headers, package data, local CellML changes, and dependency declarations checked against `src/cubie/vendored/numba_cuda_cache.py:1`, `src/cubie/vendored/cellmlmanip/`, and `pyproject.toml:9-37`. SOD-046 |
| **Subtotal** | **1,068** | **3 CURRENT, 7 STALE** | **10 STYLE** | **Ten of ten files and every line accounted for** |

### Instruction punctuation and language inventory

| File | Em dash | Semicolon | Colon | Bold | Italic | Affected-line manifest |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gui/AGENTS.md` | 1 | 6 | 2 | 0 | 1 | Em `24`; semicolon `7,20,22,25,27,30`; colon `1,20`; italic `8` |
| `memory/AGENTS.md` | 12 | 31 | 11 | 7 | 0 | Em `10,16,24,27,28,34,107,117,143,151,162`; semicolon `23,24,25,33,35,38,41,61,76,78,82,84,91,95,108,110,114,115,116,118,145,153,158,162`; colon `1,16,35,63,70,71,73,80,85,103,147`; bold `10,12,38,72,107,124,131` |
| `odesystems/AGENTS.md` | 16 | 14 | 15 | 7 | 3 | Em `7,21-24,44,49,78,80,83,85,91,98,99,105`; semicolon `13,23,62,84,89,96,111,115,120,121,124`; colon `1,8,21,23,36,37,60,70,72,80,81,87,94,110`; bold `7,47,70,78,81,84,87`; italic `46,82,93` |
| `symbolic/AGENTS.md` | 10 | 15 | 9 | 2 | 1 | Em `19,30,45,64,70,76,92,94,98`; semicolon `14,23,28,40,41,46,58,63,106,111-114,117`; colon `1,7,15,30,33,38,49,72,89`; bold `75,81`; italic `101` |
| `codegen/AGENTS.md` | 13 | 21 | 18 | 9 | 2 | Em `15-17,27,38,49,55,66,75,84,91,134`; semicolon `27,30,32,34,42,44,46,59,73,74,78,80,100,102,105,122,125,127,141,142`; colon `1,10,13,19,29-31,33-35,39,42,44,46,57,68,79,110`; bold `6,9,15-17,41,46,57,121`; italic `20,51` |
| `parsing/AGENTS.md` | 18 | 41 | 8 | 2 | 2 | Em `6,7,13,25-30,34,35,51,54,64,69,71,98,114`; semicolon `10,14,21-24,27,29,30,39,41,49,50,65,69,72,77,81,83,86,91,95,113,114,119-122,125,126`; colon `1,10,30,40,47,56,99,108`; bold `95,99`; italic `51,57` |
| `structural/AGENTS.md` | 3 | 11 | 12 | 0 | 0 | Em `12,61,83`; semicolon `14,27,32,44,48,63,66-68,70,82`; colon `1,6,14,22,25,27,33,41,53,60,73,80` |
| `outputhandling/AGENTS.md` | 12 | 13 | 7 | 6 | 0 | Em `8,9,20,21,23-25,39,43,45,59,86`; semicolon `20,21,55,69,75,81,99,100,103,104`; colon `1,22,37,62,74,89,92`; bold `38,42,45,57,62,85` |
| `summarymetrics/AGENTS.md` | 5 | 21 | 10 | 2 | 0 | Em `20,45,46,65,103`; semicolon `13,20,26,29-31,35,37,46,49,61,75,80,83,87,90,109,111,113`; colon `1,19,44,55,70,82,83,89,93`; bold `61,91` |
| `vendored/AGENTS.md` | 4 | 11 | 7 | 1 | 0 | Em `15,19,33,41`; semicolon `14-16,21,27,29,31,33,39,44,52`; colon `1,15,38,46,49,50,52`; bold `24` |
| **Total** | **94** | **184** | **99** | **36** | **9** | **SOD-046 removes all dashes, semicolons, and emphasis plus 83 prose colons. Sixteen syntax colons remain** |

The ten `Parent:` metadata colons remain. The other retained colons occur in
`memory/AGENTS.md:103`, `odesystems/AGENTS.md:21`, `:36`, and `:80`, and
`summarymetrics/AGENTS.md:82` and `:83`. They encode a dictionary entry, type
annotations, or lambda syntax. No en dash, listed empty-framing phrase, or
false-drama staccato example occurs in these ten files.

## Punctuation and language inventory

### reStructuredText source

| Character or tell | Count | Disposition |
| --- | ---: | --- |
| Em dash | 1 | Removed by SOD-004 |
| En dash | 49 | All removed by SOD-003, SOD-010, SOD-012, SOD-020, SOD-023, and SOD-024 |
| Semicolon | 0 | CLEAN |
| Colon | 591 | One prose colon removed by SOD-001. The remaining 590 are required RST syntax |
| Listed empty-framing phrases | 0 | CLEAN |
| False-drama staccato examples | 0 | CLEAN |

### rendered source docstrings

| Character or tell | Count | Disposition |
| --- | ---: | --- |
| Em dash | 8 | SOD-014, SOD-015, and SOD-033 remove all occurrences |
| En dash | 0 | CLEAN |
| Semicolon | 12 | SOD-015 and SOD-033 remove all occurrences |
| Colon | 137 | SOD-007, SOD-032, and SOD-033 remove avoidable prose colons. RST roles, numpydoc type declarations, and literal mappings retain the rest |
| Readiness, convenience, automaticity, efficiency, class, or correctness framing | Detected | SOD-010, SOD-016, SOD-028, SOD-029, SOD-030, and SOD-031 |
| Listed empty-framing phrases | 0 | CLEAN |
| False-drama staccato examples | 0 | CLEAN |

### prospective generated docstrings

These counts cover `load_cellml_model`, `MeanStd`, and `StdRms`, which would
become rendered by SOD-036 and SOD-037.

| Character or tell | Current count | Count after proposed rewrite | Disposition |
| --- | ---: | ---: | --- |
| Em dash | 0 | 0 | CLEAN |
| En dash | 0 | 0 | CLEAN |
| Semicolon | 1 | 0 | Removed by SOD-036 |
| Colon | 12 | 8 | SOD-036 and SOD-037 remove four prose colons. Eight numpydoc type declarations remain |
| Readiness or function framing | 2 blocks | 0 | Removed by SOD-036 |
| Listed empty-framing phrases | 0 | 0 | CLEAN |
| False-drama staccato examples | 0 | 0 | CLEAN |

## Completeness and freshness inventory

### Package exports

| Surface | Coverage |
| --- | --- |
| `cubie.memory.MemoryManager` | API page present. Rendered docstring has SOD-006, SOD-007, and SOD-033 |
| `cubie.memory.default_memmgr` | API page present and current |
| `cubie.memory.current_cupy_stream` | API page present with wrong directive type, SOD-005 |
| `cubie.memory.NoCudaDeviceError` | Exported, no API or user/developer coverage, SOD-034 |
| `cubie.memory.CuPyAsyncNumbaManager` | Exported on real GPU builds and assigned `None` under CUDA simulation. No API coverage. SOD-035 proposes a manual Python-domain page because `autoclass` cannot resolve the simulated value |
| `cubie.odesystems.BaseODE` | API page present. Lazy-helper cache wording is stale, SOD-017 |
| `cubie.odesystems.ODECache` | API page present and current |
| `cubie.odesystems.ODEData` | API page present. Overview description corrected by SOD-010 |
| `cubie.odesystems.SystemSizes` | API page present. Overview and rendered class framing corrected by SOD-010 |
| `cubie.odesystems.SystemValues` | API page present. Overview description corrected by SOD-010 |
| `cubie.odesystems.SymbolicODE` | API page present. Overview, readiness, and member punctuation issues are SOD-012, SOD-016, and SOD-033 |
| `cubie.odesystems.create_ODE_system` | API page present. Input types and rendered prose are stale, SOD-013 through SOD-016 |
| `cubie.odesystems.load_cellml_model` | Exported, no API page. Its prospective rendered docstring has stale dependency information and language findings covered by SOD-036 |
| All ten `cubie.outputhandling` exports | API pages present |
| Both `cubie.outputhandling.summarymetrics` exports | API pages present |
| Both `cubie.gui` exports | API pages present. Runtime build requires optional Qt dependencies |

### Public submodule surfaces without generated API coverage

These symbols are public in their submodule or are named implementation
capabilities. They are not exported from `cubie` unless stated above.

| Domain | Missing or stale coverage |
| --- | --- |
| Memory allocation internals | `PinnedBuffer`, `ChunkBufferPool`, `InstanceMemorySettings`, pinned-budget accounting, pageable and memory-mapped host backing, idle-owner eviction, and active/passive limit modes have no developer page. The generic `MemoryManager` autodoc exposes members but does not explain the lifecycle as a coherent workflow. |
| Memory module helpers | `total_system_ram`, `available_system_ram`, `host_headroom_bytes`, `defer_instance_teardown`, `run_instance_teardown`, `get_portioned_request_size`, `is_request_chunkable`, `replace_with_chunked_size`, `install_async_emm`, `placeholder_invalidate`, and `placeholder_dataready` are module-public names with no API page. `install_async_emm` has real-device and simulator definitions at `src/cubie/memory/cupy_emm.py:86` and `:97`. The placeholder hooks are at `src/cubie/memory/mem_manager.py:203` and `:211`. These names should either be documented for developers or made private. |
| Solver-helper model | `SolverHelperKind`, `HelperKindTraits`, `resolve_preconditioner_kind`, `resolve_chained_kind`, `SolverHelperRequest`, `HelperResult`, and `SolverHelperCache` have no API page. The developer guide still describes string-based helper dispatch. |
| Symbolic codegen exports | `generate_operator_apply_code`, `generate_cached_operator_apply_code`, `generate_prepare_jac_code`, `generate_cached_jvp_code`, `generate_n_stage_linear_operator_code`, `build_stage_jvp_assignments`, `generate_residual_code`, `generate_stage_residual_code`, `generate_n_stage_residual_code`, `build_stage_substitutions`, all seven exported preconditioner generators, `CUDA_FUNCTIONS`, `print_cuda`, and `print_cuda_multiple` have no generated API reference. |
| Expression-engine exports | The 52 names in `cubie.odesystems.symbolic.engine.__all__` have no generated API reference. They include all IR node types, constructors, transforms, SymPy conversion functions, and printers. The developer codegen page gives only an overview. |
| Structural exports | `structural_simplify`, `SimplifiedSystem`, `StructuralState`, `ExtraEquationsSystemError`, `ExtraVariablesSystemError`, and `InvalidSystemError` have no API reference. User documentation mentions structural simplification without documenting result or error types. |
| Other symbolic capabilities | `IndexedBaseMap`, `IndexedBases`, `ODEFile`, CellML caching, callable inspection, auxiliary caching, and helper source/member identity have no generated API pages. |
| Output sizing base | `ArraySizingClass` is module-public and undocumented as its own API. Its `nonzero` property is inherited into the four sizing pages. |
| Metric infrastructure | `MetricConfig` is module-public but lacks an API page. `SummaryMetric`, `SummaryMetrics`, and `MetricFuncCache` have pages. |
| Built-in metrics | Sixteen of 18 classes have pages. `MeanStd` and `StdRms` are missing, SOD-037. |
| GUI helpers | `FloatLineEdit`, `PreParseEditor`, `edit_pre_parse_dicts`, `show_constants_editor`, and `show_states_editor` have pages. Package exports remain limited to `ConstantsEditor` and `StatesEditor`. |
| Vendored code | No CuBIE public API is intended. `cellmlmanip` and the Numba CUDA cache snapshot are correctly absent from API navigation. Their consuming public behavior should be documented through CellML and cache pages. |

### Cross-lane freshness findings handed to the owning reviewers

* `docs/source/user_guide/memory.rst:14` through `:16` describes every solve as
  allocating and freeing all device arrays. The registry retains allocations
  until release or eviction.
* `docs/source/user_guide/memory.rst:51` through `:54` describes chunks from
  one solve as concurrent across stream groups. Stream groups coordinate
  registered instances. Chunk staging is pipelined within the run stream.
* `docs/source/user_guide/cellml.rst:55` through `:58` says only ODE CellML
  models are supported and hedges about CellML 2.0. The loader accepts CellML
  1.0 and 1.1 and the parser structurally simplifies DAE-shaped input.
* `docs/source/developer_guide/adding_metrics.rst:23` through `:115` contains a
  wrong registry import, obsolete callback signatures, a direct
  `numba.cuda` import, and stale cadence configuration.
* `docs/source/developer_guide/codegen.rst:71` through `:74` describes helper
  dispatch by strings rather than `SolverHelperRequest`.

## Sphinx build

Sphinx is not installed in either available Python environment. No build was
run because the requested dependency gate failed.

System interpreter command:

```text
> python -m sphinx --version
C:\Program Files\Python314\python.exe: No module named sphinx
```

Workspace virtual-environment command:

```text
> .\.venv\Scripts\python.exe -m sphinx --version
C:\local_working_projects\cubie\.venv\Scripts\python.exe: No module named sphinx
```

The runtime autodoc probe also recorded the exact GUI import blocker:

```text
ModuleNotFoundError: No module named 'qtpy'
```

Static inspection resolved all seven GUI definitions in
`src/cubie/gui/constants_editor.py` and `src/cubie/gui/states_editor.py`.

## Instruction mirror inventory

`src/cubie/odesystems/symbolic/engine/CLAUDE.md` contains one line,
`AGENTS.md`, but is stored as Git mode `100644`. Every other nested
`CLAUDE.md` mirror is mode `120000`. It therefore fails to expose the engine
instructions on this checkout. SOD-038 records the exact repair. The target
`src/cubie/odesystems/symbolic/engine/AGENTS.md` was read in full and agrees
with the current IR, assignment, conversion, and printer implementation.
