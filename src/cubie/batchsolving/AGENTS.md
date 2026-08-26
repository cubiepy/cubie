<!-- Parent: ../AGENTS.md -->

# batchsolving

## Purpose
The high-level batch-integration layer, split in two roles. **`Solver`** (and the `solve_ivp`
wrapper) is the API / instantiation / user-facing-data layer: it takes a `BaseODE`/`SymbolicODE`
system plus user inputs (dicts or arrays of initial values, parameters, drivers), routes loose
kwargs into typed settings groups, builds the input grid, launches, and returns a `SolveResult`.
**`BatchSolverKernel`** is the CUDA layer: the **only** kernel with a batch-wide view — one launch
maps each run to a `SingleIntegratorRun` device function that integrates one system in isolation —
and it drives all the lower-level machinery (the integrator, the array managers, memory). Host/
device buffer coordination lives in `arrays/`.

See `CUDAFactory` (root) for build/cache/`update`, config, and attrs conventions.

## Key Files
| File | Description |
|------|-------------|
| `solver.py` | `Solver` + `solve_ivp()` — the public API. `solve_ivp` also accepts raw equations (callable / string / iterable of strings), building the system via `_system_from_equations` (state names from a `y0` dict, parameter defaults from a `parameters` dict; array `parameters` rejected). `Solver` owns `system_interface`, `input_handler` (`BatchInputHandler`), and `kernel` (`BatchSolverKernel`); `driver_interpolator` is a passthrough to the kernel-owned interpolator, and most getters are thin pass-throughs to `kernel`. `Solver.compile()` runs the same input processing as `solve` and compiles the kernel without launching. |
| `BatchSolverKernel.py` | `BatchSolverKernel(CUDAFactory)` — the batch `@cuda.jit` kernel; maps each run to the `SingleIntegratorRun` device loop. Owns the `ArrayInterpolator` as a direct child factory (`driver_interpolator`; `configure_drivers()` updates it and the dependent compile settings as one unit) and the `CubieCacheHandler` with its `CachePolicy`. Defines `RunParams` (frozen: duration/warmup/t0/runs + chunk metadata) and `BatchSolverCache`; owns the `InputArrays`/`OutputArrays` managers and memory-manager registration. `compile()` allocates the batch and compiles the specialised kernel through the disk cache without launching, via the backend-selected `compile_kernel_specialization` in `cuda_simsafe`. |
| `BatchSolverConfig.py` | `BatchSolverConfig(CUDAFactoryConfig)` — holds `precision`, `loop_fn`, `compile_flags`, `driver_coefficients_shape`. Cache policy is **not** a compile setting — it lives with the kernel's `CubieCacheHandler`. `ActiveOutputs(_CubieConfigBase)` — booleans for which output arrays are produced, built via `ActiveOutputs.from_compile_flags(...)`. |
| `BatchInputHandler.py` | `BatchInputHandler` (plain class) + module-level grid builders (`unique_cartesian_product`, `combinatorial_grid`, `verbatim_grid`, `generate_grid`, `combine_grids`, `extend_grid_to_array`). Converts user dicts/arrays into `(variable, run)` 2D arrays; assembled grids are planned compactly, then written straight into a buffer chosen by the kernel's registered host backing policy (pinned within the cumulative budget, memmap past the spill threshold), so no full-size intermediate coexists with the result. A right-sized correct-precision user array passes through untouched. |
| `SystemInterface.py` | `SystemInterface` — a live view onto the bound system's `SystemValues`; resolves labels↔indices, and `merge_variable_labels_and_idxs` merges `save_variables`/`summarise_variables` labels + index kwargs into final index arrays. |
| `calibration.py` | `Solver.calibrate` backend: `run_calibration` races candidate configurations (`CandidateSpec`) on one representative batch and returns a `CalibrationResult` (winner, ranking, per-candidate `CandidateResult` measurements). The race covers a few adaptive orders per family and, for implicit families, the preconditioner, linear-solver, Newton-variant, smoothed-error, and dense-predictor settings. |
| `solveresult.py` | `SolveSpec` (attrs config snapshot); `SolveResult` — owns the solve's host buffers via `OutputArrays.loan_host_arrays` (zero copy), applies NaN-on-error masking in place, carries the solve's `stream`, and derives `time`/`time_domain_array`/`summaries_array` plus `as_numpy`/`as_numpy_per_summary`/`as_pandas` lazily; `DeviceSolveResult` — device-array handles to the solve's output buffers plus the kernel's stream, returned by `Solver.solve(on_device=True)` with no D2H copy. Both are pure data containers: no stream or memory operations happen in this module. |
| `location_tuning.py` | `Solver.tune_locations` / `auto_memory="tune"` backend. `tune_locations` compiles the all-local kernel, compiles one variant per relocatable buffer (`candidate_buffers`: nonempty, local, with a `{name}_location` setting) in worker processes that share the kernel cache, times the variants whose compiled local frame shrank against all-local with interleaved solves on a multi-wave batch slice, then pairs of the winning singles, applies the best placement and persists it under `<cache root>/location_tuning/` keyed by the all-local kernel identity. Worker entry: `python -m cubie.batchsolving.location_tuning` (pickled job on stdin; systems pickle via `SymbolicODE.__reduce__`). |
| `writeback_watcher.py` | `WritebackWatcher` (daemon thread) + `WritebackTask` — polls CUDA events via `event.query()`, copies completed pinned-buffer data into host arrays (D2H writeback) or just releases H2D staging buffers. |
| `_utils.py` | Docstring only — no exports (dead validators removed). |
| `__init__.py` | Defines the `ArrayTypes` alias (`Optional[Union[NDArray, DeviceNDArrayBase, MappedNDArray]]`) and re-exports the public surface. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `arrays/` | Host/device array managers (`InputArrays`/`OutputArrays`/`BaseArrayManager`/`ManagedArray`) — allocation, chunked transfers, writeback. See `arrays/AGENTS.md`. |

## For AI Agents

### Data flow
`Solver.solve()` → `input_handler(...)` builds `(n_vars, n_runs)` `inits`/`params` →
`kernel.run()` sets `RunParams`, refreshes compile settings (`loop_fn` from
`SingleIntegratorRun.device_function`), queues allocations via `InputArrays.update`/
`OutputArrays.update`, calls `memory_manager.allocate_queue(self)` (which may split into
chunks), then loops chunks launching the compiled kernel. Results flow back through
`OutputArrays` → `SolveResult.from_solver`.

### Solver: settings routing
`Solver.__init__` uses `merge_kwargs_into_settings` to split loose kwargs against the
subcomponents' `ALL_*_PARAMETERS` sets (`ALL_OUTPUT_FUNCTION_PARAMETERS`,
`ALL_MEMORY_MANAGER_PARAMETERS`, `ALL_STEP_CONTROLLER_PARAMETERS`,
`ALL_ALGORITHM_STEP_PARAMETERS`, `ALL_LOOP_SETTINGS`, `ALL_CACHE_PARAMETERS`); a kwarg no
set consumes raises `KeyError`, and legacy timing spellings (`RENAMED_TIMING_KWARGS`, e.g.
`dt_save`) raise with a rename hint. Internal child `update` calls stay `silent=True` —
siblings must ignore each other's keys; only the top-level entry points enforce. Add a new
result accessor on `kernel` and expose it as a `Solver` property.

### Live system updates
System changes (constant values included) reach a live solver through
`Solver.update`/solve kwargs; after the kernel update, `Solver.update`
re-resolves the recorded output-variable selection against the system's
current layout. The kernel records the system's `config_hash` after every
update; a stale hash at `Solver.solve` triggers `kernel.resync_system()`,
which replays the system's current values through the same update chain.

### Solver teardown
`Solver.close()` waits only for its last run stream, drains staging work, and
deregisters the kernel and array managers. Explicit failures are reported and
the close can be retried. `solve_ivp` closes its temporary solver before it
returns. Finalizers provide best-effort cleanup for abandoned solvers.

Physical VRAM pressure evicts a completed solver's buffers (completion
checked with a CUDA event); the evicted solver reallocates on its next
run. Host arrays above `host_spill_threshold` use `numpy.memmap` and
pooled pinned staging. Results keep disk backing and support close or
context cleanup; `as_numpy`/`as_pandas` materialise in RAM on demand.
Spill settings live on the kernel and are passed to the memory manager
explicitly.

### Grids
`BatchInputHandler` converts user dicts/arrays into `(variable, run)` arrays via the
module-level grid builders. Two grid types: `combinatorial` (cartesian product across inputs)
and `verbatim` (inputs zipped run-for-run). **The default differs by entry point:** `solve_ivp`
defaults `grid_type="combinatorial"`; `Solver.solve`/`Solver.build_grid` default `"verbatim"`.

### Variable selection (states/observables)
`None` = use all, `[]` = explicitly none, labels + index kwargs = union.
`SystemInterface.merge_variable_labels_and_idxs` pops `save_variables`/`summarise_variables`
and writes `saved_*_indices`/`summarised_*_indices` into the settings dict in place;
summarised defaults to saved when all summarise inputs are `None`.

### The batch kernel
- One launch integrates many runs; each thread runs a `SingleIntegratorRun` device function
  over a single system in isolation. `BatchSolverKernel` is the only place with a batch-wide view.
- **`RunParams` is a frozen value object** (duration/warmup/t0/runs + chunk metadata). Frozen so
  per-chunk views and allocation updates produce independent copies via `attrs.evolve` instead of
  mutating shared state: `run_params[i]` returns a copy with that chunk's run count (last chunk
  gets the dangling remainder); `update_from_allocation` returns a copy carrying
  `num_chunks`/`chunk_length`.
- **Time is float64:** `duration`/`warmup`/`t0` are coerced to `float64` in `run()` for
  accumulation accuracy, then cast to `precision` per chunk at launch.
- **Chunking is driven by memory availability:** when the batch's arrays exceed available GPU
  memory less the manager's headroom, the memory manager splits it along the run axis into
  even chunks; `num_chunks`/`chunk_length` come back on the allocation response. The run
  loop iterates chunks, calling `input_arrays.initialise(i)` (H2D) and
  `output_arrays.finalise(i)` (D2H/writeback).
- **Shared-memory sizing:** `limit_blocksize` halves the block size until dynamic shared memory
  fits under a 32 KiB ceiling; `shared_memory_needs_padding` adds a 4-byte skew only for single
  precision with an even element count (float64 never pads — it would misalign).

### Results
Every solve returns one `SolveResult` that **owns the solve's host buffers** — nothing is
copied. `OutputArrays.loan_host_arrays` empties the slots into the result; if the result has
been garbage collected by the next solve the buffers return to their slots for reuse
(`reclaim_or_release_loan`), otherwise the next solve allocates fresh backing. Keep a result
alive while its data is needed. `state`/`observables`/summary buffers/`status_codes`/
`iteration_counters` are the kernel's arrays; `time` and `time_domain_array` are views when a
single time-domain source is active (two active sources concatenate lazily into RAM on first
access). `as_numpy`, `as_numpy_per_summary`, and `as_pandas` (lazy `pandas` import) build RAM
representations on demand. Trajectories for runs that errored (nonzero `status_codes`) are
NaN-masked **in place** on the owned buffers. Disk-backed results release their spill files on
`close()`, context exit, or collection. `SolveResult.status_messages` decodes the per-run
status word into named `CUBIE_RESULT_CODES` flags via `cubie.result_codes.decode_status_codes`.
`SolveSpec` is an attrs snapshot of the solve configuration.

### One stream per kernel
Every launch and transfer for a kernel runs on `kernel.stream` — the stream its memory
manager issued for its stream group. There is **no caller-supplied stream**: neither
`Solver.solve` nor `BatchSolverKernel.run` accepts one, and no code outside the memory
manager may synchronize anything wider than that stream (no `cuda.synchronize()`; #644
removed device-wide syncs for concurrent/multiprocess operation). `SolveResult.stream` and
`DeviceSolveResult.stream` carry this stream as data so callers can order follow-up work.

### Device-resident results and inputs
`Solver.solve(on_device=True)` skips the per-chunk `output_arrays.finalise` D2H
(`kernel.run(transfer_outputs=False)`), skips the end-of-solve sync/writeback wait, and
returns a `DeviceSolveResult`: the kernel's device output buffers plus `kernel.stream`.
Contents are valid once that stream is synchronized; work queued on it executes in order
after the solve. The handles are views the next `solve()` overwrites (and a reallocation
or memory-pressure eviction detaches) — no loan is taken. Single-chunk only; a chunked run
raises `ValueError`. `solve_ivp` has no `on_device` option (it closes its temporary solver
before returning).

`initial_values`/`parameters` supplied as device arrays (validated by
`BatchInputHandler._process_device_inputs`: 2D, exact variable count, exact dtype — raise,
never pad/cast) are wired directly into the kernel via `InputArrays._attach_device_inputs`
with no host staging, H2D copy, or managed-buffer allocation; the host-side counterpart of
a lone device input is paired verbatim (defaults or a single column broadcast to the device
run count). Device inputs are likewise single-chunk only. `Solver.device_initial_values` /
`Solver.device_parameters` return the last run's device input buffers, usable as device
inputs to the next `solve` with no upload; they raise `ValueError` after a chunked run. Every dim in `BatchInputSizes` is
concrete: `driver_coefficients_shape` is a `BatchSolverConfig` compile setting (the
`(num_segments, num_drivers, order + 1)` layout baked into the compiled driver evaluators as
closure constants), seeded at kernel construction from the kernel-owned interpolator and
refreshed through `kernel.configure_drivers`/`kernel.update` wherever the interpolator's
evaluators change, so shape checks compare against the layout the kernel was compiled for.
Driver dicts name their sample spacing `driver_sample_period` — `dt` is the integrator
timestep and never reaches the interpolator.

### Calibration (`Solver.calibrate`)
- One sibling `Solver` per candidate on the parent's system, memory manager, and stream group; trial solves gate candidates before full-length timing; full-length measurements are recorded per configuration and reused.
- No pre-filtering: candidates the package rejects drop individually with the package's error message.

### Testing
`tests/batchsolving/` (`test_solver.py`, `test_BatchSolverKernel.py`, input-handler/result tests,
`test_calibration.py`).
Prefer real system fixtures (`tests/system_fixtures.py`) over mocks.

## Dependencies
### Internal
- `cubie.CUDAFactory`; `cubie.integrators` (`SingleIntegratorRun`);
  `cubie.array_interpolator` (`ArrayInterpolator`);
  `cubie.memory` (`default_memmgr`, `MemoryManager`, `ArrayRequest`/`ArrayResponse`,
  `chunk_buffer_pool`) + `cubie.buffer_registry`; `cubie.outputhandling` (`OutputCompileFlags`,
  `output_sizes`, `summary_metrics`); `cubie.odesystems` (`BaseODE`, `SymbolicODE`,
  `SystemValues`); `cubie.cubie_cache` (`CachePolicy`, `CubieCacheHandler`,
  `ALL_CACHE_PARAMETERS`); `cubie.cuda_simsafe`; `cubie._utils`.
### External
- `numba`/`numba.cuda`; `numpy`; `attrs`; optional `pandas` (lazy in `as_pandas`).
