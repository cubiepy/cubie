# User-facing documentation review inventory

## Coverage result

The lane read 42 files and 6,973 lines. Every file below has status
`read-complete`. Source comparisons used the current checkout. `CLEAN` means no
specific factual contradiction or prohibited prose construction was found in
this lane. It does not mean the file was executed.

The lane then reread all 6,973 lines for meaning-level framing,
qualification, hedging, emphasis, intensification, false drama, reader-role
assignment, and high-bar colon or semicolon use. This second pass did not rely
on the lexical-tell count below.

| File | Lines | Freshness and completeness result | Proposal IDs |
|---|---:|---|---|
| `readme.md` | 125 | Backend and install description stale. Benchmark lacks a reproducibility record. Capability intensifiers and acknowledgement punctuation need revision. | UFR-001, UFR-002, UFR-003, UFR-S01, UFR-S02, UFR-C01 |
| `infra/fleet/README.md` | 263 | Claims match current Fleet, workflow, dashboard, and IAM code. Dense punctuation, justification framing, and measured anecdotes remain. | UFR-031, UFR-S01, UFR-S02 |
| `docs/Makefile` | 20 | Build syntax only. | CLEAN |
| `docs/make.bat` | 35 | Build syntax only. | CLEAN |
| `docs/source/conf.py` | 58 | Sphinx release is 0.0.4 while the package is 0.4.0. | UFR-004 |
| `docs/source/_static/custom.css` | 5 | CSS declarations only. | CLEAN |
| `docs/source/index.rst` | 26 | Former repository URL and wrong Reference Manual target. | UFR-005 |
| `docs/source/getting_started.rst` | 139 | Opening contains framing. One-call compilation claim is false. | UFR-006, UFR-S01, UFR-S02, UFR-C01 |
| `docs/source/examples/array_interpolation_example.py` | 147 | Stale class path and nonexistent compiled-function property. The module description contains an intensifier. | UFR-026, UFR-S01 |
| `docs/source/examples/controller_step_analysis.py` | 888 | `OutputFunctions` is constructed with shifted, wrong positional arguments. The module and plotting docstrings contain intent and change-history framing. | UFR-027, UFR-S01 |
| `docs/source/examples/cpu_gpu_driver_evaluator_comparison.py` | 212 | Current evaluator usage matches the checkout. The module docstring uses example framing. | UFR-S01 |
| `docs/source/theory/cuda.rst` | 87 | Thread mapping, scaling, memory placement, on-chip storage, and TimeLogger claims are stale or unsupported. | UFR-022, UFR-S01, UFR-S02 |
| `docs/source/theory/index.rst` | 16 | Entire opening is optionality framing. | UFR-S01 |
| `docs/source/theory/jacobians.rst` | 79 | Describes the former SymPy compute pipeline instead of the current expression IR. Its opening describes the page instead of the mechanism. The finite-difference comparison uses unsupported qualitative judgments. | UFR-023, UFR-S01 |
| `docs/source/theory/numerical_integration.rst` | 135 | Confuses Euler local and global error and gives a wrong Dormand-Prince stage count. Family descriptions contain subjective recommendations. | UFR-024, UFR-S01, UFR-S02 |
| `docs/source/theory/solvers.rst` | 97 | Newton convergence, available preconditioners, and return-code packing are stale. Its opening describes the page instead of the mechanism. | UFR-025, UFR-S01, UFR-S02 |
| `docs/source/tutorials/extracting_summaries.rst` | 138 | Metric inventory is current. Framing and punctuation need revision. | UFR-S01, UFR-S02 |
| `docs/source/tutorials/first_sweep.rst` | 177 | Workflow is current. Framing and a false-drama conclusion need revision. | UFR-S01, UFR-S02 |
| `docs/source/tutorials/index.rst` | 18 | Toctree matches the three tutorials. The introduction uses metaphorical and scale framing. | UFR-S01 |
| `docs/source/tutorials/stiff_systems.rst` | 196 | Methods are current. Absolute and promotional solver descriptions need revision. | UFR-S01, UFR-S02 |
| `docs/source/user_guide/batching.rst` | 97 | Grid-type and direct-array descriptions match current interfaces. The opening uses strength and page-description framing. | UFR-S01 |
| `docs/source/user_guide/caching.rst` | 79 | Cache layer count, `cache` behavior, policy surface, and package invalidation are stale. | UFR-007, UFR-S02, UFR-C01 |
| `docs/source/user_guide/cellml.rst` | 61 | CellML version and DAE caveats are wrong. | UFR-008, UFR-S02, UFR-C03 |
| `docs/source/user_guide/choosing_algorithms.rst` | 244 | Omits Kvaerno3, Kvaerno5, and Gauss-Legendre order 8. Controller defaults are wrong. Subjective recommendations replace properties. | UFR-009, UFR-010, UFR-S01, UFR-S02, UFR-C04 |
| `docs/source/user_guide/coming_from_matlab.rst` | 150 | `ode23s` mapping is wrong. Page-description and note framing need revision. Code semicolons are valid MATLAB syntax. | UFR-011, UFR-S01, UFR-S02 |
| `docs/source/user_guide/coming_from_scipy.rst` | 156 | Direct Radau alias is omitted and one-call compile wording is false. The opening, batch heading, and note lead-in use framing. | UFR-006, UFR-011, UFR-S01, UFR-S02 |
| `docs/source/user_guide/configuration.rst` | 213 | Group count, cache routing, `auto_memory`, kernel settings, and implicit-family controller statement are incomplete or wrong. The opening is document framing. | UFR-010, UFR-012, UFR-S01, UFR-S02, UFR-C01, UFR-C02 |
| `docs/source/user_guide/drivers.rst` | 164 | Uses nonexistent `dt` driver key and omits uniform-spacing requirement. Opening is overframed. | UFR-013, UFR-S01, UFR-S02 |
| `docs/source/user_guide/index.rst` | 28 | Toctree covers all current user-guide pages. The repository URL uses the former owner. | UFR-005 |
| `docs/source/user_guide/memory.rst` | 54 | Allocation lifetime and stream-group concurrency claims are wrong. Cleanup and spill behavior are missing. The opening adds reader framing. | UFR-014, UFR-S01, UFR-S02, UFR-C02 |
| `docs/source/user_guide/optional_arguments.rst` | 254 | Output selection default needs clarification. Definition prose is punctuation-heavy. Cache, memory, and DAE settings are incomplete. Reader-role, expertise, and default-quality framing needs revision. | UFR-015, UFR-S01, UFR-S02, UFR-C01, UFR-C02, UFR-C03 |
| `docs/source/user_guide/results.rst` | 252 | Advertises rejected timing aliases and says observables save by default. Status vocabulary and device lifecycle are incomplete. Attribute, note, preference, and usefulness framing needs revision. | UFR-015, UFR-016, UFR-S01, UFR-S02, UFR-C05 |
| `docs/source/user_guide/solving.rst` | 127 | Interface examples are current. Batch-performance framing and ASCII em dashes need revision. | UFR-S01, UFR-S02 |
| `docs/source/user_guide/speed.rst` | 115 | Universal scaling, bottleneck, compile-time, buffer-placement, and precision claims are unsupported or stale. The logger-access lead-in assigns an action to the reader. | UFR-021, UFR-S01 |
| `docs/source/user_guide/systems.rst` | 215 | Names nonexistent `GenericODE`, denies callable `user_functions`, and treats algebraic unknowns as invalid states. | UFR-017, UFR-S01, UFR-S02, UFR-C03 |
| `docs/source/user_guide/timing.rst` | 344 | Timing model is otherwise detailed and current. `duration` is incorrectly marked required. Usefulness, reasonableness, and example framing need revision. | UFR-019, UFR-S01, UFR-S02 |
| `docs/source/user_guide/troubleshooting.rst` | 88 | Newton instruction says tighter tolerances give more room. Status-specific diagnosis is missing. Advice and usefulness framing need revision. | UFR-020, UFR-S01, UFR-S02, UFR-C05 |
| `docs/source/user_guide/userfunctions.rst` | 166 | Exposes internal parsing/codegen APIs, unpacks the wrong arity, and imports a backend directly. Case and tips framing need revision. | UFR-018, UFR-S01, UFR-S02 |
| `src/cubie/writing_cuda_functions.md` | 44 | Stub and open questions do not provide binding factual instructions. | UFR-029 |
| `tests/README.md` | 609 | Canonical claims conflict with the current suite. CUDA-check helper is absent. Contains AI-agent narrative. | UFR-028 |
| `tests/integrated_numerical_tests/julia_reference/data/README.md` | 29 | Data inventory matches current named artifacts. Punctuation can be simplified. | UFR-030 |
| `CHANGELOG.md` | 623 | Current through 0.4.0. No contradiction found within historical release context. Style review intentionally excluded by project policy. | CLEAN, historical exemption |

## Punctuation receipt

The scan counted punctuation characters, not only matching lines. The `---`
column counts ASCII em-dash substitutes used inside prose words or clauses. It
does not count RST heading underlines, Markdown rules, table rules, or source
comments used as separators. The lexical-tell column is a narrow automated
receipt. Manual findings are recorded under UFR-S01 and in the file inventory.

Disposition codes are:

- `K` keeps punctuation required by code, math, URLs, RST, CSS, Make, or a
  literal mapping.
- `S` rewrites prose under UFR-S01 or UFR-S02.
- `R` replaces the containing section or file under a factual proposal.
- `H` records historical changelog punctuation without proposing an edit.

| File | `—` | prose `---` | `;` | `:` | lexical tells | Disposition |
|---|---:|---:|---:|---:|---:|---|
| `readme.md` | 3 | 0 | 1 | 27 | 0 | S, R, K for URLs |
| `infra/fleet/README.md` | 7 | 0 | 18 | 35 | 1 | S, K for keys and URLs |
| `docs/Makefile` | 0 | 0 | 0 | 4 | 0 | K |
| `docs/make.bat` | 0 | 0 | 0 | 3 | 0 | K |
| `docs/source/conf.py` | 0 | 0 | 0 | 16 | 0 | K |
| `docs/source/_static/custom.css` | 0 | 0 | 3 | 5 | 0 | K |
| `docs/source/index.rst` | 0 | 0 | 0 | 17 | 1 | R, K for RST and URL |
| `docs/source/getting_started.rst` | 0 | 0 | 1 | 55 | 0 | S, R, K |
| `docs/source/examples/array_interpolation_example.py` | 0 | 0 | 0 | 39 | 0 | K, R for stale lines, S |
| `docs/source/examples/controller_step_analysis.py` | 0 | 0 | 0 | 224 | 0 | K, R for stale call, S |
| `docs/source/examples/cpu_gpu_driver_evaluator_comparison.py` | 0 | 0 | 0 | 48 | 0 | K, S |
| `docs/source/theory/cuda.rst` | 0 | 0 | 1 | 11 | 1 | R, S, K |
| `docs/source/theory/index.rst` | 0 | 0 | 0 | 6 | 0 | S, K |
| `docs/source/theory/jacobians.rst` | 0 | 0 | 0 | 48 | 0 | R, S, K |
| `docs/source/theory/numerical_integration.rst` | 0 | 0 | 2 | 78 | 0 | R, S, K for math |
| `docs/source/theory/solvers.rst` | 0 | 2 | 1 | 69 | 1 | R, S, K for math |
| `docs/source/tutorials/extracting_summaries.rst` | 1 | 0 | 3 | 47 | 1 | S, K |
| `docs/source/tutorials/first_sweep.rst` | 0 | 0 | 3 | 66 | 1 | S, K |
| `docs/source/tutorials/index.rst` | 0 | 0 | 0 | 11 | 0 | S, K |
| `docs/source/tutorials/stiff_systems.rst` | 0 | 0 | 4 | 53 | 2 | S, K |
| `docs/source/user_guide/batching.rst` | 0 | 0 | 0 | 29 | 0 | S, K |
| `docs/source/user_guide/caching.rst` | 1 | 1 | 1 | 13 | 0 | R, K |
| `docs/source/user_guide/cellml.rst` | 2 | 0 | 1 | 7 | 0 | R, S, K |
| `docs/source/user_guide/choosing_algorithms.rst` | 0 | 0 | 19 | 49 | 0 | R, S, K |
| `docs/source/user_guide/coming_from_matlab.rst` | 2 | 0 | 17 | 52 | 0 | S, K for MATLAB |
| `docs/source/user_guide/coming_from_scipy.rst` | 7 | 0 | 6 | 45 | 0 | R, S, K |
| `docs/source/user_guide/configuration.rst` | 2 | 0 | 8 | 84 | 0 | R, S, K |
| `docs/source/user_guide/drivers.rst` | 1 | 0 | 1 | 43 | 3 | R, S, K |
| `docs/source/user_guide/index.rst` | 0 | 0 | 0 | 7 | 0 | R, K |
| `docs/source/user_guide/memory.rst` | 0 | 1 | 0 | 9 | 0 | R, S, K |
| `docs/source/user_guide/optional_arguments.rst` | 26 | 0 | 11 | 64 | 0 | S, R, K |
| `docs/source/user_guide/results.rst` | 8 | 0 | 4 | 51 | 0 | S, R, K |
| `docs/source/user_guide/solving.rst` | 0 | 1 | 2 | 66 | 0 | S, K |
| `docs/source/user_guide/speed.rst` | 0 | 0 | 0 | 25 | 1 | R, K |
| `docs/source/user_guide/systems.rst` | 1 | 0 | 4 | 61 | 0 | R, S, K |
| `docs/source/user_guide/timing.rst` | 0 | 0 | 1 | 88 | 0 | R, S, K |
| `docs/source/user_guide/troubleshooting.rst` | 0 | 0 | 3 | 19 | 0 | R, S, K |
| `docs/source/user_guide/userfunctions.rst` | 2 | 0 | 2 | 43 | 0 | R, S, K |
| `src/cubie/writing_cuda_functions.md` | 5 | 0 | 1 | 5 | 0 | R |
| `tests/README.md` | 26 | 0 | 4 | 66 | 2 | R |
| `tests/integrated_numerical_tests/julia_reference/data/README.md` | 3 | 0 | 1 | 7 | 0 | R, K |
| `CHANGELOG.md` | 0 | 0 | 19 | 591 | 0 | H |

### Prose dash and semicolon locations

Every prose occurrence below is assigned to `S` or a containing `R` proposal.
Occurrences omitted from this list are code, math, markup, URL, CSS, Make, or
historical syntax assigned `K` or `H` in the receipt.

- `readme.md`: em dash 101, 106, 112; semicolon 90.
- `infra/fleet/README.md`: em dash 11 (twice), 34, 82, 99, 101, 149;
  semicolon 21, 33, 70, 115, 116, 117, 128, 132, 142, 170, 173, 203, 221,
  226, 231, 243, 261.
- `docs/source/getting_started.rst`: semicolon 26.
- `docs/source/theory/cuda.rst`: semicolon 23.
- `docs/source/theory/numerical_integration.rst`: semicolon 33, 64.
- `docs/source/theory/solvers.rst`: ASCII em dash 19, 53; semicolon 15.
- `docs/source/tutorials/extracting_summaries.rst`: em dash 91; semicolon 6,
  71, 134.
- `docs/source/tutorials/first_sweep.rst`: semicolon 66, 69, 103.
- `docs/source/tutorials/stiff_systems.rst`: semicolon 117, 119, 188, 195.
- `docs/source/user_guide/caching.rst`: em dash 78; ASCII em dash 39;
  semicolon 59.
- `docs/source/user_guide/cellml.rst`: em dash 24, 50; semicolon 60.
- `docs/source/user_guide/choosing_algorithms.rst`: semicolon 6, 21, 24, 87,
  91, 100, 123, 127, 131, 146, 150, 165, 188, 192, 200, 208, 212, 222, 227.
- `docs/source/user_guide/coming_from_matlab.rst`: prose em dash 58, 137;
  prose semicolon 46, 50, 56, 69, 107. Semicolons 15-18 and 83-87 are MATLAB
  code and remain `K`.
- `docs/source/user_guide/coming_from_scipy.rst`: em dash 36, 69, 73, 75,
  123, 124, 155; semicolon 6, 14, 61, 66, 89, 116.
- `docs/source/user_guide/configuration.rst`: em dash 122, 197; semicolon 41,
  45, 50, 82, 86, 94, 132, 205.
- `docs/source/user_guide/drivers.rst`: em dash 141; semicolon 142.
- `docs/source/user_guide/memory.rst`: ASCII em dash 42.
- `docs/source/user_guide/optional_arguments.rst`: em dash 8, 24, 32, 43,
  52, 77, 84, 91, 100, 106, 111, 130, 141, 150, 156, 160, 170, 177, 185,
  191, 197, 206, 222, 226, 230, 250; semicolon 26, 34, 49, 54, 113, 167,
  179, 180, 187, 192, 238.
- `docs/source/user_guide/results.rst`: em dash 18, 42, 58, 59, 71, 84, 183,
  209; ASCII em dash 49-51; semicolon 31, 70, 75, 86.
- `docs/source/user_guide/solving.rst`: ASCII em dash 47, 125-127;
  semicolon 98, 111.
- `docs/source/user_guide/systems.rst`: em dash 177; semicolon 51, 122, 171,
  181.
- `docs/source/user_guide/timing.rst`: semicolon 237.
- `docs/source/user_guide/troubleshooting.rst`: semicolon 32, 68, 85.
- `docs/source/user_guide/userfunctions.rst`: em dash 58, 104; semicolon 149,
  165.
- `src/cubie/writing_cuda_functions.md`: em dash 6, 12, 24, 43, 44;
  semicolon 27.
- `tests/README.md`: all 26 em dashes and prose semicolons 101, 114, 487,
  496 are covered by the whole-file replacement UFR-028.
- `tests/integrated_numerical_tests/julia_reference/data/README.md`: em dash 7,
  19, 22; semicolon 24.

### Semantic prose locations

The second pass read every line without relying on lexical matches. `S` below
marks framing, qualifying, hedging, emphasis, intensification, false drama,
reader-role assignment, subjective recommendation, anecdote, or change-history
language. Exact replacements are under UFR-S01. A containing factual
replacement is marked `R` when it removes the same construction.

- `readme.md`: 11-22 R, 34-40 S, 49-63 R, 89-91 R, 125 R.
- `infra/fleet/README.md`: 8-28 S, 93-102 S, 113-117 S, 145-176 S,
  202-245 S.
- `docs/source/index.rst`: 4-8 R.
- `docs/source/getting_started.rst`: 4-6 S, 18-28 S, 33-36 S, 80-84 R,
  62-64 S, 102-105 S, 128-133 S.
- `docs/source/examples/array_interpolation_example.py`: 1-8 S and R,
  110-112 R.
- `docs/source/examples/controller_step_analysis.py`: 1-12 S, 298-324 R,
  679-683 S.
- `docs/source/examples/cpu_gpu_driver_evaluator_comparison.py`: 1-13 S.
- `docs/source/theory/cuda.rst`: 4-6 S, 11-87 R.
- `docs/source/theory/index.rst`: 4-7 S.
- `docs/source/theory/jacobians.rst`: 4-6 S, 22-65 R.
- `docs/source/theory/numerical_integration.rst`: 4-6 S, 11-22 S and R,
  60-65 R, 77-78 S, 92-113 S, 131-135 S.
- `docs/source/theory/solvers.rst`: 4-6 S, 17-19 S, 28-97 R.
- `docs/source/tutorials/extracting_summaries.rst`: 4-9 S, 14-15 S,
  63-68 S, 77-80 S, 89-98 S.
- `docs/source/tutorials/first_sweep.rst`: 4-10 S, 15-21 S, 41-48 S,
  68-70 S, 84-87 S, 103-111 S, 128 S, 166-167 S.
- `docs/source/tutorials/index.rst`: 4-11 S.
- `docs/source/tutorials/stiff_systems.rst`: 4-12 S, 39-42 S, 61-65 S,
  97-106 S, 152-167 S, 185-196 S.
- `docs/source/user_guide/batching.rst`: 4-5 S.
- `docs/source/user_guide/caching.rst`: 4-79 R.
- `docs/source/user_guide/cellml.rst`: 22-60 R.
- `docs/source/user_guide/choosing_algorithms.rst`: 4-7 S, 17-35 R and S,
  84-100 R and S, 143-173 R and S, 203-233 R and S.
- `docs/source/user_guide/coming_from_matlab.rst`: 4-7 S, 40-58 S,
  63-74 R, 136-149 S.
- `docs/source/user_guide/coming_from_scipy.rst`: 4-8 S, 82-95 R,
  97-102 S, 122-127 R, 145-146 S.
- `docs/source/user_guide/configuration.rst`: 4-9 S, 20-213 R and S.
- `docs/source/user_guide/drivers.rst`: 4-12 S, 74-87 S, 120-127 R and S,
  135-152 S.
- `docs/source/user_guide/index.rst`: 4-6 R.
- `docs/source/user_guide/memory.rst`: 4-6 S, 11-16 R, 40-54 R and S.
- `docs/source/user_guide/optional_arguments.rst`: 4-20 S, 69-73 S,
  91-94 S, 177-201 S, 216-217 S, 219-251 R and S.
- `docs/source/user_guide/results.rst`: 11 S, 47 S, 68-78 S, 121-131 R
  and S, 179-184 S, 186-190 R, 208-210 S.
- `docs/source/user_guide/solving.rst`: 4-13 S, 47-49 S, 74-78 S,
  100-106 S, 117-120 S, 125-127 S.
- `docs/source/user_guide/speed.rst`: 1-57 R, 78 S, 85-115 R.
- `docs/source/user_guide/systems.rst`: 4-16 R, 100-151 S,
  155-169 R, 176-182 S, 187-207 R.
- `docs/source/user_guide/timing.rst`: 111-119 S, 245-258 R,
  278-280 S, 311-315 S.
- `docs/source/user_guide/troubleshooting.rst`: 21-35 R, 44-48 S,
  64-69 S.
- `docs/source/user_guide/userfunctions.rst`: 4-11 S, 13-150 R,
  159-166 S.
- `src/cubie/writing_cuda_functions.md`: 1-44 R.
- `tests/README.md`: 1-609 R.
- `tests/integrated_numerical_tests/julia_reference/data/README.md`: 1-29 R.
- `CHANGELOG.md`: historical exemption. No style rewrite is proposed.

Files absent from this list contain only build syntax, configuration syntax,
stylesheet declarations, navigation structure, or code with no prose finding.

### Prose colon locations requiring rewrite

The following colons support framing or fragments rather than a required
mapping, signature, directive, formula, table field, or complete list
introduction. Their containing UFR-S01, UFR-S02, or factual replacement removes
them.

- `readme.md`: 125.
- `docs/source/getting_started.rst`: 5, 18.
- `docs/source/theory/numerical_integration.rst`: 20.
- `docs/source/tutorials/extracting_summaries.rst`: 15, 80, 97.
- `docs/source/tutorials/first_sweep.rst`: 18, 21, 48, 87, 166.
- `docs/source/tutorials/stiff_systems.rst`: 154.
- `docs/source/user_guide/caching.rst`: 4.
- `docs/source/user_guide/coming_from_matlab.rst`: 4, 137.
- `docs/source/user_guide/coming_from_scipy.rst`: 8 and the heading at 97.
- `docs/source/user_guide/configuration.rst`: 5.
- `docs/source/user_guide/drivers.rst`: 120.
- `docs/source/user_guide/optional_arguments.rst`: 4, 69, 75, 98, 128,
  148, 183, 195.
- `docs/source/user_guide/results.rst`: 11, 47, 72.
- `docs/source/user_guide/timing.rst`: 56, 79, 85, 314.
- `docs/source/user_guide/userfunctions.rst`: 5, 20, 61, 108.
- `src/cubie/writing_cuda_functions.md`: 3, 18, 26, 29, 34.
- `tests/README.md`: all prose colons are covered by UFR-028.

All other colon characters in the punctuation table are retained for code,
math, URLs, RST structure, literal mappings, or complete introductions to
lists, tables, examples, and formulas.

## Completeness and freshness inventory

| Missing or fragile topic | Current coverage | Required action |
|---|---|---|
| CUDA backend selection and environment variables | Split across README, getting started, troubleshooting, and source docstrings. | UFR-C01 |
| Three cache layers and cache policy | Current cache page describes two layers and one overloaded argument. | UFR-007, UFR-C01 |
| Managed allocation lifetime, spill, close, and context managers | Current memory page describes per-call allocation and release. | UFR-014, UFR-C02 |
| DAE structural workflow | Capability bullet exists. No end-to-end guide covers simplification controls or mass matrices. | UFR-C03 |
| Algorithm registry freshness | Narrative page already omits three registered methods. | UFR-009, UFR-C04 |
| Controller defaults | Narrative statements generalize by family and are already false for DIRK. | UFR-010, UFR-C04 |
| Driver sample timing contract | Current guide uses `dt` and omits uniform spacing. | UFR-013 |
| Result bit flags and status diagnosis | The user-facing corpus documents two of eight nonzero failure flags. Six are missing (`src/cubie/result_codes.py:51-58`; `docs/source/user_guide/results.rst:220-228`). | UFR-016, UFR-C05 |
| Device-result ownership and stream ordering | Mentioned in results but no complete lifecycle workflow. | UFR-C05 |
| Reproducible performance evidence | README contains machine numbers without a repository measurement record. | UFR-003 |
| Test policy source of truth | `tests/README.md` conflicts with the suite and root instructions. | UFR-028 |

## CLAUDE.md duplicate inventory

These paths were inventoried as duplicate instruction documents and were not
counted again in the 6,973-line total.

The following 19 files are symbolic links to the adjacent or root `AGENTS.md`:

- `CLAUDE.md`
- `src/cubie/CLAUDE.md`
- `src/cubie/batchsolving/CLAUDE.md`
- `src/cubie/batchsolving/arrays/CLAUDE.md`
- `src/cubie/gui/CLAUDE.md`
- `src/cubie/integrators/CLAUDE.md`
- `src/cubie/integrators/algorithms/CLAUDE.md`
- `src/cubie/integrators/loops/CLAUDE.md`
- `src/cubie/integrators/matrix_free_solvers/CLAUDE.md`
- `src/cubie/integrators/step_control/CLAUDE.md`
- `src/cubie/memory/CLAUDE.md`
- `src/cubie/odesystems/CLAUDE.md`
- `src/cubie/odesystems/symbolic/CLAUDE.md`
- `src/cubie/odesystems/symbolic/codegen/CLAUDE.md`
- `src/cubie/odesystems/symbolic/parsing/CLAUDE.md`
- `src/cubie/odesystems/symbolic/structural/CLAUDE.md`
- `src/cubie/outputhandling/CLAUDE.md`
- `src/cubie/outputhandling/summarymetrics/CLAUDE.md`
- `src/cubie/vendored/CLAUDE.md`

`src/cubie/odesystems/symbolic/engine/CLAUDE.md` is a regular file, not a
symbolic link, and its SHA-256 differs from the adjacent `AGENTS.md`. It cannot
be treated as a duplicate without a separate comparison. This is a coverage
gap outside the 42-file user-facing lane and was reported to the root reviewer.
