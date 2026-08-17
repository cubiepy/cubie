<!-- Parent: ../AGENTS.md -->

# outputhandling

## Purpose
Compiles, caches, and configures the CUDA device functions that write solver state and
summary metrics during integration. A validated `OutputConfig` is turned by
`OutputFunctions(CUDAFactory)` into three compiled device functions — save-state,
update-summaries, save-summaries — which the integration loop calls at its output
intervals. Sizing helpers translate the same configuration into host-side array shapes for
buffer allocation.

See `CUDAFactory` (repo root) for build/cache/`update`, attrs-config, the config-mutation
rule, and the `ArraySizingClass`/`.nonzero` pattern. The metric registry the summary
callbacks compose lives in `summarymetrics/`.

## Key Files
| File | Description |
|------|-------------|
| `output_functions.py` | `OutputFunctions(CUDAFactory)` — `build()` compiles the three device functions into an `OutputFunctionCache`; exposes them plus sizing/flag properties; `ALL_OUTPUT_FUNCTION_PARAMETERS` is the accepted-kwarg set. |
| `output_config.py` | `OutputConfig(CUDAFactoryConfig)` — validated compile settings (index arrays, output-type flags, `sample_summaries_every`); `OutputCompileFlags(_CubieConfigBase)`; `_indices_validator` (bounds + uniqueness). |
| `output_sizes.py` | The directory's `ArraySizingClass` subclasses: `OutputArrayHeights`, `SingleRunOutputSizes`, `BatchInputSizes`, `BatchOutputSizes`. |
| `save_state.py` | `save_state_factory()` — device function copying selected states/observables/counters (and optionally time) into output windows. |
| `update_summaries.py` | `update_summary_factory()` + a recursive `chain_metrics` — accumulate each metric into working buffers every step. |
| `save_summaries.py` | `save_summary_factory()` + its own `chain_metrics` — flush accumulated metrics to output arrays each window. |
| `__init__.py` | Re-exports `OutputConfig`, `OutputCompileFlags`, `OutputFunctionCache`, `OutputFunctions`, the sizing classes, and `summary_metrics`/`register_metric`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `summarymetrics/` | The metric registry (`summary_metrics` singleton, `register_metric`) and the 18 built-in metric device-function pairs the summary callbacks compose. See `summarymetrics/AGENTS.md`. |

## For AI Agents

### The three compiled device functions
`OutputFunctions` compiles and caches three device functions the loop calls (accessed via
`save_state_func`, `update_summaries_func`, `save_summary_metrics_func`):
- **save-state** `(current_state, current_observables, current_counters, current_step,
  output_states_slice, output_observables_slice, output_counters_slice)` — writes the
  selected states, then the time value at slot `nstates` (if `save_time`), the selected
  observables, and the iteration counters (if `save_counters`), each via `stwt`.
- **update-summaries** `(current_state, current_observables, state_summary_buffer,
  observable_summary_buffer, current_step)` — runs the metric-update chain once per
  summarised variable, accumulating into that variable's summary-buffer segment.
- **save-summaries** — runs the metric-save chain to flush the accumulated buffers to the
  summary output arrays, once per window.

All three live in `OutputFunctionCache`, each validated `instance_of(Callable)`, so a
disabled path still compiles a real (possibly no-op) device function, never `None`.
Building the summary functions is also where `summary_metrics` receives the config's real
`precision` and `sample_summaries_every`.

### OutputConfig & validation
- `OutputConfig.from_loop_settings(...)` is the constructor used from solver/loop settings
  (`OutputFunctions.__init__` calls it); it normalises `None` indices to empty arrays.
- The output flags (`save_state`, `save_observables`, `save_time`, `save_counters`) and the
  `summary_types` tuple are **derived** from `output_types`, so any settings change must
  re-run `validation_passes()` (five checks, incl. `_check_for_no_outputs`, which raises if
  nothing is enabled) to refresh them. `update_compile_settings` does not —
  `OutputFunctions.update()` calls `validation_passes()` immediately after it.
- `ALL_OUTPUT_FUNCTION_PARAMETERS` is the accepted-kwarg filter for `update()`.
  `max_states`/`max_observables` are **not** in it: they are the total system
  dimensions (`system_sizes.states`/`.observables`), set at construction and
  refreshed only by `SingleIntegratorRunCore.update()` when a system
  re-specialisation changes the layout.

### output_types parsing
`update_from_outputs_list` sets the four flags from the literals `"state"`, `"observables"`,
`"time"`, `"iteration_counters"`, and treats any entry that prefix-matches a registered
metric name as a summary type; anything else warns and is dropped. `save_state`/
`save_observables` are additionally gated on a non-empty index array (requesting `"state"`
with no saved indices yields `save_state == False`).

### Time & counters layout
Time is not a separate column: save-state writes the current step into the state window at
slot `nstates`, just past the saved states; `OutputArrayHeights.from_output_fns` reflects
this as `n_saved_states + 1*save_time`. All writes use `stwt` (a store with a write-through
cache hint).

### Sizing classes
`OutputArrayHeights`, `SingleRunOutputSizes`, `BatchInputSizes`, `BatchOutputSizes` are this
directory's `ArraySizingClass` subclasses (host-side array shapes); call `.nonzero` before
allocation. See `CUDAFactory` (root) for the pattern.

### Metric chains
`update_summaries.py` and `save_summaries.py` each build a **recursive closure chain**
(`chain_metrics`) over the requested metric device functions rather than looping a list —
Numba can't JIT an iterable of device functions. The two are independent copies with
different signatures (save additionally threads output offsets/sizes). The chain runs once
per summarised variable. Metric device-function contract: see `summarymetrics/AGENTS.md`.

### Testing
`tests/outputhandling/` mirrors the files: `test_output_config.py`,
`test_output_functions.py`, `test_output_sizes.py`, `test_save_state.py`,
`test_save_summaries.py`, `test_update_summaries.py`.

## Dependencies
### Internal
- `cubie.CUDAFactory` (`CUDAFactory`, `CUDADispatcherCache`, `CUDAFactoryConfig`,
  `_CubieConfigBase`); `cubie._utils` (`PrecisionDType`, `opt_gttype_validator`);
  `cubie.cuda_simsafe` (`compile_kwargs`, `stwt`); `cubie.outputhandling.summarymetrics`
  (`summary_metrics`).
### External
- `numba` (`cuda.jit`, `int32`); `attrs` (`define`, `field`, `Factory`, `cmp_using`,
  `evolve`); `numpy` (`int_`, `asarray`, `array_equal`, `arange`, `unique`).
