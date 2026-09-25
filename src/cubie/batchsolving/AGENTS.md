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
| `solver.py` | `Solver` + `solve_ivp()` — the public API. `solve_ivp` also accepts raw equations (callable / string / iterable of strings), building the system via `_system_from_equations` (state names from a `y0` dict, parameter defaults from a `parameters` dict; array `parameters` rejected). `Solver` owns `system_interface`, `input_handler` (`BatchInputHandler`), and `kernel` (`BatchSolverKernel`); `driver_interpolator` is a passthrough to the kernel-owned interpolator, and most getters are thin pass-throughs to `kernel`. `Solver.compile()` applies settings and compiles the kernel. `Solver.given` (a `SolverSettings`) and `Solver.effective` (an `EffectiveSettings`) are the settings the user set and the settings in use; `Solver.is_given(name)`; `Solver.settings_dict()` is the given record with the logger's level (multiprocessing safe with `for_new_process=True`); `Solver.copy()` is `Solver(system.copy(), **settings_dict())` at the current logging level; `optimisation_candidates(force)` keeps the given performance keys fixed. |
| `solver_settings.py` | `SolverSettings`, one field per setting a Solver accepts (`None` = not given) on `_CubieConfigBase`, and `EffectiveSettings` (adds the names only resolution sets; its `as_kwargs` keeps a `None` interval or memory proportion, the `passes_none` fields). Given arrays are stored as read-only copies and lists as tuples. `clashing_names` lets `BaseODE` reject a constant named like a setting. |
| `resolve_defaults.py` | `resolve(given, system, interface)` returns the `EffectiveSettings` (algorithm, controller and gains, step bounds, family/tableau/DAE step defaults, inner tolerances, output indices via `VariableSelection`, loop intervals and flags). |
| `BatchSolverKernel.py` | `BatchSolverKernel(CUDAFactory)` — the batch `@cuda.jit` kernel; maps each run to the `SingleIntegratorRun` device loop. Owns the `ArrayInterpolator` as a direct child factory (`driver_interpolator`, filled by the `drivers` setting; `update` refreshes the evaluator settings only when the interpolator's config hash changes; `run()` raises `ValueError` when the system declares drivers and none are given). `build_kernel()` attaches a `CUBIECache` built from the config's `cache` settings and `config_hash` to the dispatcher and keeps it for the flush-on-change path in `_invalidate_cache`. Defines `RunParams` (frozen: duration/warmup/t0/runs + chunk metadata) and `BatchSolverCache`; owns the `InputArrays`/`OutputArrays` managers and memory-manager registration. `compile()` records time parameters and compiles the kernel; `kernel_is_cached()` asks the disk cache instead. The constructor takes one flat dict (`BatchSolverKernel(system, **settings)`): the memory keys, its own config fields and everything the integrator's children take; the driver interpolator takes its keys from the same dict. `update` runs `memory_manager.update`, the interpolator, the integrator, then its own config with the integrator's `loop_fn` and compile flags. |
| `BatchSolverConfig.py` | `BatchSolverConfig(CUDAFactoryConfig)` — holds `precision`, `loop_fn`, `compile_flags`, `coefficients_shape`, `max_registers`, `kernel_name`, `blocksize` and `auto_performance` (both `eq=False`), and the hash-excluded (`eq=False`) nested `cache: CacheSettings` (`cache_enabled`/`cache_mode`/`max_cache_entries`/`cache_dir`, loose keys in `ALL_CACHE_PARAMETERS`, all part of `ALL_KERNEL_PARAMETERS`); the field's converter accepts the `cache=` shorthand (bool, `"flush_on_change"`, or a directory) and loose keys evolve the nested object like `UnrollFlags`. `ActiveOutputs(_CubieConfigBase)` — booleans for which output arrays are produced, built via `ActiveOutputs.from_compile_flags(...)`. |
| `BatchInputHandler.py` | `BatchInputHandler` (plain class) + module-level grid builders (`unique_cartesian_product`, `combinatorial_grid`, `verbatim_grid`, `generate_grid`, `combine_grids`, `extend_grid_to_array`). Converts user dicts/arrays into `(variable, run)` 2D arrays; assembled grids are planned compactly, then written straight into a buffer chosen by the kernel's registered host backing policy (pinned within the cumulative budget, memmap past the spill threshold), so no full-size intermediate coexists with the result. A right-sized correct-precision user array passes through untouched. |
| `SystemInterface.py` | `SystemInterface` — a live view onto the bound system's `SystemValues`; resolves labels↔indices, and `merge_variable_labels_and_idxs` merges `save_variables`/`summarise_variables` labels + index kwargs into final index arrays. |
| `comparison.py` | `ComparisonRunner(solver, inits, params, duration, settling_time, t0)`: the candidate-timing runner `calibrate` and `optimize` share. Stages the batch on the device once, switches the solver itself between `Candidate`s through `Solver.update` (plus block size and residency) and times each with device-only solves. `compile(candidates)` returns the accepted candidates (a rejected one keeps its error, no time); `time(candidates)` runs the fixed protocol and returns `CandidateTiming`s; `warm()`, `size_batch(candidates, waves)` and `fit_batch(measured, target_ms, grow, shrink)` are the shared warm-up and batch sizing; `rank_timings` ranks by success tier then time; `close` restores the configuration at `open`. |
| `calibration.py` | `Solver.calibrate` backend: `run_calibration` races `CandidateSpec`s (`algorithm` plus settings) in stages through a `ComparisonRunner` and returns a `CalibrationResult` (winner, ranking, per-candidate `CandidateResult`). Each stage's winner is the top of `rank_timings`; a configuration timed once is recalled by later stages. Sizes the batch at the given duration. |
| `optimize.py` | `Solver.optimize` backend: `run_optimization` times the solver's `optimisation_candidates(force)` at `launch_candidates(kernel, runs=)` through a `ComparisonRunner` and applies the best `LaunchResult` through `apply_launch`. Ramps the duration toward `target_ms` before sizing the batch. |
| `solveresult.py` | `SolveSpec` (attrs config snapshot); `SolveResult` — owns the solve's host buffers via `OutputArrays.loan_host_arrays` (zero copy), applies NaN-on-error masking in place, carries the solve's `stream`, and derives `time`/`time_domain_array`/`summaries_array` plus `as_numpy`/`as_numpy_per_summary`/`as_pandas` lazily; `DeviceSolveResult` — device-array handles to the solve's output buffers plus the kernel's stream, returned by `Solver.solve(on_device=True)` with no D2H copy. Both are pure data containers: no stream or memory operations happen in this module. |
| `writeback_watcher.py` | `WritebackWatcher` (daemon thread) + `WritebackTask` — polls CUDA events via `event.query()`, copies completed pinned-buffer data into host arrays (D2H writeback) or just releases H2D staging buffers. |
| `_utils.py` | Docstring only; no exports. |
| `__init__.py` | Defines the `ArrayTypes` alias (`Optional[Union[NDArray, DeviceNDArrayBase, MappedNDArray]]`) and re-exports the public surface. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `arrays/` | Host/device array managers (`InputArrays`/`OutputArrays`/`BaseArrayManager`/`ManagedArray`) — allocation, chunked transfers, writeback. See `arrays/AGENTS.md`. |

## Data flow
`Solver.solve()` → `update(**kwargs)` for solve-time settings → `check_duration` on the
effective timing → `input_handler(...)` builds `(n_vars, n_runs)` `inits`/`params` →
`kernel.run()` sets `RunParams`, queues allocations via
`InputArrays.update`/`OutputArrays.update`, calls `memory_manager.allocate_queue(self)`
(which may split the batch into chunks) and launches per chunk. Results return through
`OutputArrays` → `SolveResult.from_solver`.

## Solver settings
`__init__` and `update` flatten the settings groups, record `given`, update the system
(settings and constants by name), resolve, and pass `effective.as_kwargs()` (`None` for
every name not in effect) to `kernel.update`. The kernel fills a placement or unroll key
given `None` from `kernel.performance_defaults()` under `auto_performance`; otherwise it
falls to its declared default. `duration`, `settling_time` and `t0` are per-solve
arguments, never settings: `solve` checks them with `check_duration` and `compile` takes
none of them. `update` returns early when nothing changed and the system is not stale.
`None` returns a setting to its declared default; `time_logging_level` sets the global
logger. `settings_dict()` is the given record plus the logger's level; `copy()` rebuilds
from it on a copied system. Child `update` calls are `silent=True`. New result
accessors go on `kernel` with a `Solver` property.

A system changed outside the Solver is `kernel.system_config_stale`; the next `update`
pushes the whole effective record so every child re-reads its products.

## Teardown and memory pressure
`Solver.close()` waits for its last run stream, drains staging work and deregisters the
kernel and array managers; a failed close can be retried. `solve_ivp` closes its
temporary solver before returning. Finalizers clean up abandoned solvers.

VRAM pressure evicts a completed solver's buffers; it reallocates on its next run. Host
arrays above `HOST_SPILL_FRACTION` of RAM are `numpy.memmap` files in the cache root,
staged through the pinned pool; results keep the disk backing until close, and
`as_numpy`/`as_pandas` load them into RAM.

## Grids and variable selection
`BatchInputHandler` builds `(variable, run)` arrays with the module-level grid builders:
`combinatorial` (cartesian product) or `verbatim` (zipped run-for-run). `solve_ivp`
defaults to `"combinatorial"`, `Solver.solve`/`Solver.build_grid` to `"verbatim"`.

For states and observables, `None` = all, `[]` = none, labels plus index kwargs = union.
`SystemInterface.merge_variable_labels_and_idxs` pops `save_variables`/
`summarise_variables` and writes `saved_*_indices`/`summarised_*_indices` into the
settings dict; summarised defaults to saved when every summarise input is `None`.

## The batch kernel
- Each thread runs a `SingleIntegratorRun` device function over one system;
  `BatchSolverKernel` is the only batch-wide view.
- `RunParams` is frozen (duration/warmup/t0/runs + chunk metadata): `run_params[i]`
  returns a copy with chunk `i`'s run count (the last chunk takes the remainder);
  `update_from_allocation` returns a copy carrying `num_chunks`/`chunk_length`.
- `duration`/`warmup`/`t0` are `float64` in `run()` and cast to `precision` per chunk at
  launch.
- When the batch's arrays exceed available memory less the manager's headroom, the
  memory manager splits along the run axis into even chunks. The run loop calls
  `input_arrays.initialise(i)` (H2D) and `output_arrays.finalise(i)` (D2H) per chunk.
- `launch_geometry(blocksize=None, runs=None)` returns block size and dynamic shared
  bytes; `launchable_shapes(blocksizes, runs=None)` lists valid shapes and occupancy.
  `runs=None` sizes a full block. Launch choices and geometry are cached per signature;
  launch policies live in `optimize.py`.
- Kept across solves: the system snapshot identity, the chunk partition until an
  allocation replaces it, and the timing `CUDAEvent`s while timing is on.

## Results
A `SolveResult` owns the solve's host buffers with no copy: `OutputArrays.loan_host_arrays`
empties the slots into it. If the result has been garbage collected by the next solve,
`reclaim_or_release_loan` returns the buffers to their slots; otherwise the next solve
allocates fresh ones. `time`/`time_domain_array` are views when one time-domain source is
active (two concatenate into RAM on first access). `as_numpy`, `as_numpy_per_summary`
and `as_pandas` build RAM copies on demand. Runs with nonzero `status_codes` are
NaN-masked in place. `status_messages` decodes the status word via
`cubie.result_codes.decode_status_codes`. `SolveSpec` snapshots the solve configuration.

## One stream per kernel
Every launch and transfer runs on `kernel.stream`, the stream its memory manager issued
for its stream group. No caller-supplied stream; nothing outside the memory manager
synchronizes more than that stream (no `cuda.synchronize()`). `SolveResult.stream` and
`DeviceSolveResult.stream` expose it for ordering follow-up work.

## Device-resident results and inputs
`Solver.solve(on_device=True)` skips the D2H transfers, host output buffers and result
loan, and returns a `DeviceSolveResult`: the kernel's device output buffers plus
`kernel.stream`. The handles are views the next `solve()` overwrites. Single-chunk
only; a chunked run raises `ValueError`. `solve_ivp` has no `on_device`.

Device-array `initial_values`/`parameters` must be 2D with the exact variable count and
dtype (`BatchInputHandler._process_device_inputs` raises otherwise) and attach directly
through `InputArrays._attach_device_inputs`, with no host staging. A lone device input's
host counterpart is paired verbatim. Device inputs are single-chunk only.
`Solver.device_initial_values`/`device_parameters` return the last run's device inputs
for reuse; they raise `ValueError` after a chunked run.

`coefficients_shape` is a `BatchSolverConfig` compile setting (the
`(num_segments, num_drivers, order + 1)` layout compiled into the driver evaluators),
seeded from the kernel-owned interpolator and refreshed through `kernel.update`; shape
checks compare against it. A driverless kernel's layout has a zero first dimension. The
`drivers` setting is a `DriverSamples`; `driver_sample_period` is its sample spacing and
`dt` the integrator timestep.

## Candidate comparisons (`Solver.calibrate`, `Solver.optimize`)
- Both run on the solver through `comparison.ComparisonRunner`: one device input pair,
  one device output set, candidates switched by `Solver.update`.
- The timing protocol, success-tier ranking, warm-up and batch sizing are fixed in
  `comparison.py`; both entry points take `auto_size`, `waves` and `target_ms` only.
- A candidate the package rejects drops with its error message and no time.
- `run_optimization(compile_only=True)` compiles the candidate kernels and runs nothing.

## Dependencies
### Internal
- `cubie.CUDAFactory`; `cubie.integrators` (`SingleIntegratorRun`);
  `cubie.array_interpolator` (`ArrayInterpolator`);
  `cubie.memory` (`default_memmgr`, `MemoryManager`, `ArrayRequest`/`ArrayResponse`,
  `chunk_buffer_pool`) + `cubie.buffer_registry`; `cubie.outputhandling` (`OutputCompileFlags`,
  `output_sizes`, `summary_metrics`); `cubie.odesystems` (`BaseODE`, `SymbolicODE`,
  `SystemValues`); `cubie.cubie_cache` (`CUBIECache`); `cubie.cuda_simsafe`;
  `cubie._utils`.
### External
- `numba`/`numba.cuda`; `numpy`; `attrs`; optional `pandas` (lazy in `as_pandas`).
