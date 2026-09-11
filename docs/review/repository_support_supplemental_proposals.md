# Repository support supplemental proposals

The file records review proposals. No existing source, test,
configuration, or documentation file was changed.

## Freshness corrections

### D3-F01 Lorenz command help

Evidence is at `benchmarks/lorenz_mean_runtime.py:520-527` and
`benchmarks/lorenz_mean_runtime.py:651-654`.

Replace the `ArgumentParser` description.

```python
description=(
    "Kernel-runtime benchmark for the Lorenz ensemble. "
    "Prints compile metrics and the mean of the lowest kernel and "
    "wall times for each configuration. Use ab_gate.py for A/B "
    "comparisons."
)
```

The calculations at lines 520 through 523 and the result description at
lines 525 through 527 use the mean of the lowest values.

### D3-F02 Memory heuristic destination

Evidence is at `benchmarks/memory_location_sweep.py:1088`. The target exists
at `src/cubie/integrators/memory_heuristics.py`.

Replace the printed destination.

```python
print("\npaste into src/cubie/integrators/memory_heuristics.py")
```

### D3-F03 Profiling script import roots

Evidence is at `scratch_profile/bench_ncu_bicgstab.py:37-40` and
`scratch_profile/bench_ncu_bigsys.py:26-29`. The referenced
`C:\local_working_projects\cubie-bicgstab-review\src` directory does not
exist.

Use the checked-out repository containing each script.

```python
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
```

Apply the same replacement to both scripts.

### D3-F04 and D3-F05 Test inventory regeneration

The generated artifact at `tests/_inventory.json` contains distributed
stale content.

Confirmed drift locations include

- `tests/_inventory.json:8483-8727` describes the removed
  `numba_cuda_printer.py` and `CUDAPrinter`.
- `tests/_inventory.json:21716` says `batchsolving/_utils.py` has no public
  exports. The module exports `name_and_compile_kernel` at
  `src/cubie/batchsolving/_utils.py:13-16`.
- `tests/_inventory.json:3701-3715` describes the removed `CacheConfig`.
- `tests/_inventory.json:17585-17592` describes the removed
  `_check_saved_indices` method.
- `tests/_inventory.json:22319` describes the removed `_align_run_counts`
  method.
- A source-symbol comparison found candidate drift in 16 of the 100 file
  buckets.
- `tests/query_inventory.py:29` names the missing
  `tests/_convert_inventory.py` as its regeneration command.

Regenerate all 100 buckets from current source and tests. The generated
result must satisfy six checks.

1. Every bucket resolves to a tracked source file.
2. Every section symbol resolves to current code or is identified as an
   inherited symbol.
3. Every description states current behavior.
4. Every `file` field equals its bucket key.
5. The query command lists and retrieves every bucket.
6. The regeneration command named by the query tool exists and reproduces
   the checked-in JSON.

If the inventory remains a maintained file without a generator, replace the
missing-file branch in `tests/query_inventory.py:25-33` with truthful output.

```python
if not INVENTORY.exists():
    print(f"ERROR: {INVENTORY} not found.", file=sys.stderr)
    sys.exit(1)
```

The generator should remove the eight prose semicolons currently found in
`text` values. Apply the table replacements.

| Current location | Proposed text |
|---|---|
| `tests/_inventory.json:43` | Precision is removed from `output_settings`. The system precision is used. |
| `tests/_inventory.json:1565` | Dictionary values are unpacked into the result. The original key is tracked. |
| `tests/_inventory.json:2120` | Returns `(True, True)` when values change and invalidates the layouts. |
| `tests/_inventory.json:4467` | Construction with `dxdt` succeeds. Optional fields default to `-1`. |
| `tests/_inventory.json:4921` | Recomputes the hash and updates `gen_file` when the hash changes. |
| `tests/_inventory.json:5138` | Checks the `gen_file` cache and marks the operation skipped on a cache hit. |
| `tests/_inventory.json:5973` | Construction stores all fields and applies tuple converters to sequence fields. |
| `tests/_inventory.json:16408` | Adaptive mode calls the controller. Acceptance requires `accept_step[0]` and no step failure. |

The regenerated values contain many compact condition labels written as
`condition: result`. Render them as factual sentences. For example, replace
`In real CUDA: delegates to _is_cuda_array` with `Delegates to
_is_cuda_array under real CUDA`.

## Em dash rewrites

The missing-value glyph at `benchmarks/ncu_algorithm_comparison.py:427`
remains. It is table data, not sentence punctuation.

### `benchmarks/ab_gate.py`

Replace the affected prose at lines 4 through 18.

```text
For each installed CUDA backend the gate starts two persistent
``lorenz_mean_runtime.py --worker`` processes. Side A imports ``cubie``
from an ephemeral ``git worktree`` at ``--main``. The default is
``origin/main`` and the worktree is removed afterward. Side B imports
``cubie`` from the current repository. The gate ping-pongs short solve blocks
between them in drift-balancing ABBA order. Each block reports the mean
of its lowest ``k`` per-solve kernel times and the mean of its lowest
``k`` per-solve wall times.

Host scatter adds time. The wall floor excludes that scatter and detects
changes that lengthen the end-to-end critical path or remove chunk-transfer
overlap. Each verdict is the median paired percent
delta. Kernel deltas gate at ``--threshold``. Wall deltas gate at
``--wall-threshold``.
```

Replace the affected prose at lines 37 through 45.

```text
``host_overhead`` uses eight trajectories. Its sub-millisecond wall
statistic measures the per-call host cost of ``Solver.solve``. The wall
statistic gates in absolute milliseconds with
``--host-overhead-threshold`` because a fixed per-call cost is a rounding
error against the percentage thresholds used by the other configurations.
It has no kernel row because the eight-trajectory kernel is launch
dominated. Both workers run the current repository's benchmark script. Each side
uses its assigned ``cubie`` import. A configuration announced by one worker
is skipped with a notice.
```

Replace the affected prose at lines 47 through 60.

```text
Block design

The floor of the kernel-time distribution tracks the compiled kernel's
intrinsic cost and moves with GPU clock state.
Running paired blocks seconds apart minimizes clock-state drift within the
pair. The pairing cancels clock-state drift. Each side pays its process startup,
JIT compilation, and grid construction costs once. A row is marked
DISTRUST when at least two pairs lie on each side of the regression
threshold. Rerun after increasing ``--pairs`` or reducing background GPU
load before using a DISTRUST row. Constant background load increases
absolute times but cancels from the deltas. Both workers hold their device
pools concurrently. The default allocation is 0.7 GB per worker. Chunking
depends on ``--n-runs``.
```

Replace the comment at lines 341 through 343.

```python
# Size the wave configuration to two full waves at side A's
# occupancy and impose that count on both sides. An occupancy drop
# on side B must then add a third wave.
```

### `benchmarks/lorenz_mean_runtime.py`

Replace lines 4 through 10.

```text
The benchmark solves the Lorenz ensemble used by the GPUODEBenchmarks
CuBIE runner with the same solver settings. One warm-up solve absorbs JIT
compilation. ``timeit.repeat`` then runs one solve per repeat with garbage
collection enabled. The first 20 post-warm-up solves are discarded because
the GPU has not reached steady state. The statistic uses the next
``repeats`` solves.
```

Replace the affected Radau description at lines 19 through 24.

```text
The implicit solves cost tens of milliseconds at the configured batch size. Generated
symbol aliasing prevents Newton from reaching its iteration cap. Launch
effects do not blur the measured runtime. Adaptive blocks stay within the gate's runtime
budget.
```

Replace the affected descriptions at lines 43 through 64.

```text
``host_overhead`` runs the fixed configuration with eight trajectories.
The positional ``n_runs`` value does not scale it. Its kernel runs in tens
of microseconds. Its wall statistic measures the per-call host cost of
``Solver.solve`` for every batch size.

The kernel statistic is the mean of the lowest ``min_count`` per-solve
kernel times recorded by the per-chunk CUDA events. The selected solves ran
at full boost clock without on-GPU contention. They track the compiled
kernel's intrinsic cost. The wall statistic is the mean of the lowest
``min_count`` host times measured across ``Solver.solve``. Host delays add time. The
wall floor excludes those delays and detects longer critical paths and lost
chunk overlap. Wall thresholds account for host scatter.
```

Replace the affected compile-metric text at lines 66 through 76.

```text
Compile metrics include registers per thread, ptxas spill-store and
spill-load byte counts, memory sizes, launch geometry, occupancy, SM count,
run count, and chunk count. Spill counts come from the verbose link log for
each cubin digest. They report code-generation counts. They do not report
allocated bytes per thread or runtime traffic. The script clears caches at
startup so the kernels link in the benchmark process unless
``--no-clear-cache`` is given.
``ab_gate.py`` supplies that option with a fresh cache directory. All
metrics require a real GPU. The script exits under the CUDA simulator.
```

Replace lines 85 through 91.

```text
``--worker`` starts a persistent solve server for ``ab_gate.py``. It builds
the solvers, loads the grids, runs one warm-up solve for each configuration,
prints one ``@META`` line per configuration, and prints
``@READY <config>...``. The wave configuration is omitted from ``@READY``
because the gate supplies its size. The worker then accepts commands on
standard input until `quit` or end of file.
```

### Other em dash locations

Apply the direct replacements.

| Location | Proposed text |
|---|---|
| `benchmarks/ncu_algorithm_comparison.py:8-11` | Every launch runs `2**20` trajectories, providing ten-plus occupancy waves at 24 resident blocks per SM on a 56-SM GPU. |
| `benchmarks/ncu_algorithm_comparison.py:16-22` | MLIR strips line tables during LTO linking. Each algorithm is profiled with an `-lto` production arm followed by a `-nolto` source-attribution arm. The report places them next to each other. Use the non-LTO arm for source structure and the LTO arm for absolute measurements. |
| `benchmarks/ncu_algorithm_comparison.py:49-53` | Disable "Profile from start". The worker brackets the launches with `cuda.profile_start()` and `cuda.profile_stop()`. The brackets exclude the preparation solve. |
| `blackbox/__init__.py:8-11` | Every state, observable, and constant label is an opaque token such as `cN_vM`. The systems carry no domain semantics. They arrive unbuilt. CuBIE compiles their CUDA kernels on the first solve. |
| `noxfile.py:40-46` | The cache key does not include the CUDA toolkit version. Wipe the shared cache before each cell to prevent a later cell from loading a kernel compiled by an earlier cell. |
| `noxfile.py:53-55` | Nox runs the cells serially. Each cell keeps the project's `-n8` xdist setting used by local and CI test runs. |
| `pyproject.toml:56-61` | The `mlir*` extras install CuBIE's `cubie-numba-cuda-mlir` build with pending native fixes. It uses the `numba_cuda_mlir` import package and must not be installed with the stock wheel. Python compatibility fixes are applied by `cubie._mlir_compat`. |
| `scratch_profile/bench_ncu_bicgstab.py:24-29` | `nruns` defaults to 65,536, which supplies 256 blocks at block size 256 and saturates the 56 SMs on an RTX 4070 SUPER. `blocksize` defaults to 256. `limit_blocksize` first halves the block size when both `dynamic_sharedmem >= 32 KiB` and `blocksize > 32`. The performance stage stops at `blocksize == 32` regardless of footprint. A hardware stage permits `blocksize < 32` when the device's per-block limit requires it. The script prints the effective launch shape. |

## Semicolon rewrites

For every listed prose location, replace the semicolon with a period and
capitalize the next clause. In compact function docstrings, use two
sentences. In definition lists, move each item to its own sentence or list
entry.

- `benchmarks/ab_gate.py:7`, `benchmarks/ab_gate.py:25`,
  `benchmarks/ab_gate.py:31`, `benchmarks/ab_gate.py:33`,
  `benchmarks/ab_gate.py:36`, `benchmarks/ab_gate.py:44`,
  `benchmarks/ab_gate.py:59`, `benchmarks/ab_gate.py:70`,
  `benchmarks/ab_gate.py:130`, `benchmarks/ab_gate.py:175`,
  `benchmarks/ab_gate.py:187`, `benchmarks/ab_gate.py:240`,
  `benchmarks/ab_gate.py:252`, `benchmarks/ab_gate.py:264`,
  `benchmarks/ab_gate.py:368`, `benchmarks/ab_gate.py:414`,
  `benchmarks/ab_gate.py:444`, `benchmarks/ab_gate.py:450`,
  `benchmarks/ab_gate.py:461`, `benchmarks/ab_gate.py:476`,
  `benchmarks/ab_gate.py:485`, and `benchmarks/ab_gate.py:566`.
- `benchmarks/dense_prediction_ratio_sweep.py:9`,
  `benchmarks/dense_prediction_ratio_sweep.py:17`,
  `benchmarks/dense_prediction_ratio_sweep.py:20`,
  `benchmarks/dense_prediction_ratio_sweep.py:23`,
  `benchmarks/dense_prediction_ratio_sweep.py:27`,
  `benchmarks/dense_prediction_ratio_sweep.py:35`,
  `benchmarks/dense_prediction_ratio_sweep.py:53`, and
  `benchmarks/dense_prediction_ratio_sweep.py:502`.
- `benchmarks/lorenz_mean_runtime.py:66-69`,
  `benchmarks/lorenz_mean_runtime.py:75`,
  `benchmarks/lorenz_mean_runtime.py:367`,
  `benchmarks/lorenz_mean_runtime.py:375`,
  `benchmarks/lorenz_mean_runtime.py:384`,
  `benchmarks/lorenz_mean_runtime.py:406`,
  `benchmarks/lorenz_mean_runtime.py:419`,
  `benchmarks/lorenz_mean_runtime.py:471`,
  `benchmarks/lorenz_mean_runtime.py:493`,
  `benchmarks/lorenz_mean_runtime.py:593`,
  `benchmarks/lorenz_mean_runtime.py:603`,
  `benchmarks/lorenz_mean_runtime.py:608`, and
  `benchmarks/lorenz_mean_runtime.py:686`.
- `benchmarks/memory_location_sweep.py:12`,
  `benchmarks/memory_location_sweep.py:67`,
  `benchmarks/memory_location_sweep.py:320`, and
  `benchmarks/memory_location_sweep.py:712`.
- `benchmarks/ncu_algorithm_comparison.py:8`,
  `benchmarks/ncu_algorithm_comparison.py:50`,
  `benchmarks/ncu_algorithm_comparison.py:275`, and
  `benchmarks/ncu_algorithm_comparison.py:808`.
- `benchmarks/ncu_algorithm_worker.py:158` and
  `benchmarks/ncu_algorithm_worker.py:243`.
- `benchmarks/vendor_julia_reference.py:34` and
  `benchmarks/vendor_julia_reference.py:48`.
- `blackbox/__init__.py:9`, `blackbox_solve_2.py:51`, and
  `ci/tools/merge_junit.py:1`.
- `infra/fleet/cost_dashboard.py:15`,
  `infra/fleet/cost_dashboard.py:20`,
  `infra/fleet/cost_dashboard.py:30`,
  `infra/fleet/cost_dashboard.py:64`,
  `infra/fleet/cost_dashboard.py:95`,
  `infra/fleet/cost_dashboard.py:126`,
  `infra/fleet/cost_dashboard.py:1099`,
  `infra/fleet/cost_dashboard.py:1123`,
  `infra/fleet/cost_dashboard.py:1178`, and
  `infra/fleet/cost_dashboard.py:1226`.
- `scratch_profile/bench_ncu_bicgstab.py:21`,
  `scratch_profile/bench_ncu_bicgstab.py:26`, and
  `scratch_profile/bench_ncu_bigsys.py:20`.

Keep semicolons that are part of Python command strings, generated report
delimiters, CSS content-security policies, and MIME types. They are syntax,
not prose structure.

## Framing colon rewrites

Apply replacements where the colon acts as sentence framing.

| Location | Proposed text |
|---|---|
| `benchmarks/ab_gate.py:2` | `Block-interleaved A/B kernel-runtime gate comparing main with the worktree.` |
| `benchmarks/ab_gate.py:47` | Use the heading `Block design`. |
| `benchmarks/dense_prediction_ratio_sweep.py:43-45` | `float16 is not swept. No implicit-solver path runs at float16, and the seed comparison cannot discriminate at its tolerances. Its ceilings remain 0.0.` |
| `benchmarks/lorenz_mean_runtime.py:6-10` | `The benchmark runs one warm-up solve, then runs one solve per timeit.repeat sample with garbage collection enabled. It discards 20 post-warm-up solves before calculating the statistic.` |
| `benchmarks/lorenz_mean_runtime.py:12` | `Five configurations report compile metrics and two runtime statistics.` |
| `benchmarks/lorenz_mean_runtime.py:33-34` | `Its wall time detects critical-path transfer, staging, writeback, and lost overlap costs that kernel events omit.` |
| `benchmarks/lorenz_mean_runtime.py:37-42` | `The fixed configuration is sized to two full waves using 2 * SMs * blocks_per_SM * runs_per_block trajectories.` |
| `benchmarks/lorenz_mean_runtime.py:72-76` | `The script clears caches at startup so kernels link in the benchmark process unless --no-clear-cache is given. ab_gate.py supplies that option with a fresh cache directory. The metrics require a real GPU.` |
| `benchmarks/memory_location_sweep.py:23-24` | `The driver uses single-configuration mode internally and invokes one subprocess per configuration.` |
| `benchmarks/memory_location_sweep.py:29-31` | `To calibrate a card, run the sweep, run --fit, and commit the emitted entry under the card's architecture code.` |
| `benchmarks/memory_location_sweep.py:35-37` | `Timing uses the method in benchmarks/lorenz_mean_runtime.py. It records per-chunk CUDA events after one warm-up solve.` |
| `benchmarks/memory_location_sweep.py:179-183` | `Every state in the nonlinear nearest-neighbour chain couples to its ring neighbours through one nonlinear term.` |
| `benchmarks/memory_location_sweep.py:239-242` | `Both sides disable auto_memory. The sweep measures explicit placements and excludes the heuristics under calibration.` |
| `benchmarks/ncu_algorithm_worker.py:148-151` | `Each arm builds its own system instance. Two solvers sharing one SymbolicODE mutate shared factory state, making kernel output depend on build order.` |
| `blackbox_solve_2.py:61-62` | `Choose from state, observables, time, iteration_counters, and summaries.` |
| `scratch_profile/bench_ncu_bicgstab.py:15` | Use the heading `Variants`. |
| `scratch_profile/bench_ncu_bicgstab.py:31-35` | Use the heading `Metrics` followed by the metric list. |
| `scratch_profile/bench_ncu_bigsys.py:13` | Use the heading `Variants`. |
| `scratch_profile/bench_ncu_bigsys.py:22-24` | `Resident thread count sets the L1 and L2 capacity available to each 4 to 16 KB per-run working set under local placement.` |
| `tests/fixtures/cellml/_make_blackbox.py:19-21` | `The transform preserves element order. Cellmlmanip uses document order, so output column i matches source column i.` |

Retain colons required by reStructuredText `Usage::` blocks, numpydoc field
syntax, URLs, endpoint names, format strings, type annotations, dictionary
literals, and literal protocol labels. Each retained colon carries syntax or
introduces a structured list.

## Framing and emphasis cleanup

Replace the simulator marker description at `pyproject.toml:105`.

```toml
"sim_only marks simulator-only debug tests"
```

Replace the trusted-publishing permission comments at
`.github/workflows/ci_semver_manage.yml:75` and
`.github/workflows/test_pypi.yml:55`.

```yaml
# Trusted publishing requires the OIDC `id-token: write` permission.
```

Replace the protected-environment framing at
`.github/workflows/test_pypi.yml:58-59`.

```yaml
# The publishing job uses the protected testpypi environment.
```

Replace the one-line merge helper description at
`ci/tools/merge_junit.py:1`.

```python
"""Merge JUnit reports. Retain node IDs supplied in NODES.json."""
```

Replace the output-category help at `blackbox_solve_2.py:59-62` with
`Use 'state' for one state-output launch`.

Replace the spot-price comment at `packer/windows-gpu.pkr.hcl:31` with
`must clear`.
