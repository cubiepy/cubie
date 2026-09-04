# Policy-specific iteration labels

`counter_probe.py` labels an existing, completed targeted unroll cohort.
It does not extract structured timing samples and writes outside the bank.
These execution counters validate the physical workload description;
they are not inputs to the shipped pre-compile heuristic.

## Bank identity and execution

The probe reads the original `cohort_manifest`, `cohort_protocol`,
`wavedone`, and `configdone` records and checks their manifest digests.
It uses the protocol's actual duration and run count, including any
duration increase selected by the original harness. A policy must have
an eligible completed observation under `bank_analysis.analyse_config`.
Identical source/compiler/cubin observations can supply timing receipts
for an aliased policy; that policy still gets its own instrumented
solver and raw counter arrays. The duplicate all-full timing reference
does not constitute another unroll policy.

The imported unroll harness reconstructs the complete original manifest.
Source hash, both harness hashes, compiler/environment identity, package
versions, algorithm settings, precision, grid size, candidates, and timing
protocol must match exactly. Use the original frozen tree's `src` and
`benchmarks` through external `PYTHONPATH`; the script never changes
compiler environment variables. Preparation also generates each solver's
helpers through cached properties and verifies zero native overloads.
The source file, `fn_hash`, kernel config hashes, and exact input-array
hashes are retained.

For each policy and block size with eligible bank timings, execution
constructs a state-only reference and a separate solver using
`pl.make_solver(..., extra={"unroll": ul.unroll_flags(policy),
"output_types": ["state", "iteration_counters"]})`. The state-only
reference must reproduce the bank cubin hash. Execution installs
`pl.spill_helpers().install_spill_capture()` once, matching
`unroll_landscape.worker_main`: verbose linking and linker-cache bypass
affect cubin metadata as well as diagnostic availability. CPU preparation
does not install this hook. Both kernels retain their
own cubin, config hash, compiler identity and resource/occupancy figures.
The instrumented kernel may require a different dynamic shared-memory
size or limited block size. Its actual geometry is recorded independently.
Both compiled kernels must admit at least two full occupancy waves at the
unchanged cohort run count. Multi-chunk samples are retained but excluded
from validated labels because total-grid occupancy is insufficient to
validate each chunk's launch.

For each requested sample, the probe saves exact input arrays and a
compressed NPZ containing every state output, raw status word, all raw
counter rows, state-only reference arrays, and int64 per-run counter
totals. JSON records shape/dtype/value hashes, integer histograms, totals,
means, and exact state/status comparisons. Raw arrays are saved before
counter validation, so failed diagnostics remain inspectable. A label is
eligible only when states are finite and exactly equal, status words
match, both solves succeed, and both execute a single chunk. There is no
fitted or relaxed numerical tolerance. A mismatch remains recorded and
the command exits unsuccessfully; it does not discard or silently accept
the discrepancy. Instrumented timing is never used for the timing bank or
physical model. Retained raw Solver diagnostics can include its default
time-logger output; the probe does not extract those printed timings.

## Counter semantics and limits

The current output layout is `(save_row, counter, run)`, with int32
counters. The four columns are:

1. Newton iterations, summed over every Newton solve in the interval.
2. Linear-solver iterations, summed over every linear solve, including
   smoothed-error solves. The source calls this the Krylov counter; a
   direct LU solve reports one iteration.
3. Attempted time steps, including rejected attempts.
4. Rejected time steps.

The DAE initializer contributes to the first save row. Every save resets
the interval accumulator. Summing save rows therefore gives the recorded
per-run total, without multiplying by stage count again. A failed run may
not save its final interval. Int64 host summation prevents host overflow,
but cannot repair device int32 overflow. Negative values and rejected
counts exceeding attempts are rejected; positive modular wrap cannot be
ruled out by these tests alone.

The arrays do not identify individual stage solves, Newton iterations,
Krylov calls, or convergence-check timestamps. They cannot reconstruct
warp-voted loop maxima: maxima of accumulated per-run totals are not
the sum of maxima at individual warp-coherent loop exits. Per-run
distributions are labels, not instruction counts or register estimates.

Source receipts (relative to the frozen measurement tree):

- `benchmarks/unroll_landscape.py:108`: complete target manifest;
  `:1064`: final cohort duration/run count.
- `benchmarks/placement_landscape.py:434`: original algorithm settings;
  `:449`: solver construction; `:595`: compiled launch geometry.
- `src/cubie/outputhandling/output_sizes.py:378`: four columns and
  `(n_saves, 4, num_runs)` shape.
- `src/cubie/outputhandling/save_state.py:148`: raw counter stores.
- `src/cubie/integrators/loops/ode_loop.py:657`: initializer counters;
  `:676`: first-save accumulation; `:870`: reset before each attempted
  step; `:947`: accumulation, attempted/rejected columns;
  `:1048`: interval reset after saving.
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py:530`:
  additive Newton and total linear iterations.
- `src/cubie/integrators/matrix_free_solvers/lu_solver.py:164`:
  one iteration per direct solve.
- `tests/integrators/algorithms/test_iteration_counters.py:53`:
  family-specific solve multiplicities; `:79`: raw column selection;
  `:89`: exact multi-stage counter sums.

## Commands

Run from the research tree, selecting the frozen source externally:

```powershell
$env:PYTHONPATH='C:\local_working_projects\cubie-worktrees\hardware-epoch-ff3a567f\src;C:\local_working_projects\cubie-worktrees\hardware-epoch-ff3a567f\benchmarks;C:\local_working_projects\cubie-worktrees\hardware-unroll-placement\benchmarks'
$env:CUBIE_CUDA_BACKEND='mlir'
$env:NUMBA_ENABLE_CUDASIM='0'
& C:\local_working_projects\cubie\.venv\Scripts\python.exe benchmarks/hardware_model/counter_probe.py --bank C:\local_working_projects\cubie-notes\hardware_unroll_placement\lorenz_split_bridge_e1 --system lorenz --algo kvaerno3_bicgstab --cohort lorenz-split-bridge-e1 --policy u11111100 --out C:\local_working_projects\cubie-notes\hardware_unroll_placement\counter_labels_dirk
```

The default writes `prepared.json` and performs no kernel compilation or
GPU execution. Add `--execute` when the GPU measurement lane is available.
Use `--samples N` to retain N separate label solves per policy/geometry;
the default is one. Repeat `--policy` to select policies, or omit it to
probe every unique policy in the cohort. Execution creates a new directory
per policy/block size and refuses to overwrite an existing label run.
The existing timing bank is always read-only.
