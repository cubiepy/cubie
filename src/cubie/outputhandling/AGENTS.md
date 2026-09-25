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
| `summarymetrics/` | The metric registry (`summary_metrics` singleton, `register_metric`) and the built-in metric device-function pairs the summary callbacks compose. See `summarymetrics/AGENTS.md`. |

## The three device functions
`OutputFunctions` compiles three device functions the loop calls (`save_state_fn`,
`update_summaries_fn`, `save_summaries_fn`):
- save-state `(current_state, current_observables, current_counters, current_step,
  output_states_slice, output_observables_slice, output_counters_slice)` writes the
  selected states, the time at slot `nstates` (if `save_time`), the selected
  observables and the iteration counters (if `save_counters`), each with `stwt`.
- update-summaries `(current_state, current_observables, state_summary_buffer,
  observable_summary_buffer, current_step)` runs the metric update chain once per
  summarised variable.
- save-summaries runs the metric save chain, flushing the buffers to the summary
  outputs once per window.

`OutputFunctionCache` validates all three as callables, so a disabled path compiles a
no-op, never `None`. Building the summary functions passes the config's `precision` and
`sample_summaries_every` to `summary_metrics`.

## OutputConfig
- `OutputConfig.from_loop_settings(...)` builds the config (`OutputFunctions.__init__`
  calls it) and turns `None` indices into empty arrays.
- The output flags (`save_state`, `save_observables`, `save_time`, `save_counters`) and
  `summary_types` derive from `output_types`; `OutputFunctions.update()` re-runs
  `validation_passes()` after `update_compile_settings`, which raises when nothing is
  enabled.
- `ALL_OUTPUT_FUNCTION_PARAMETERS` filters `update()` keys. `n_states`/`n_observables`
  are the total system dimensions, set at construction and refreshed only by
  `SingleIntegratorRunCore.update()` when the system layout changes.
- `update_from_outputs_list` sets the flags from `"state"`, `"observables"`, `"time"`,
  `"iteration_counters"` and treats entries that prefix-match a registered metric as
  summary types; anything else warns and is dropped. `save_state`/`save_observables`
  also need a non-empty index array.

## Layout and sizing
Time is written into the state window at slot `nstates`;
`OutputArrayHeights.from_output_fns` gives `n_saved_states + 1*save_time`.
`OutputArrayHeights`, `SingleRunOutputSizes`, `BatchInputSizes` and `BatchOutputSizes`
are this directory's `ArraySizingClass` subclasses.

## Metric chains
`update_summaries.py` and `save_summaries.py` each build a recursive closure chain
(`chain_metrics`) over the requested metric device functions, since Numba cannot JIT an
iterable of device functions. The save chain also threads output offsets and sizes. The
metric contract is in `summarymetrics/AGENTS.md`.

## Dependencies
### Internal
- `cubie.CUDAFactory` (`CUDAFactory`, `CUDADispatcherCache`, `CUDAFactoryConfig`,
  `_CubieConfigBase`); `cubie._utils` (`PrecisionDType`, `opt_gttype_validator`);
  `cubie.cuda_simsafe` (`compile_kwargs`, `stwt`); `cubie.outputhandling.summarymetrics`
  (`summary_metrics`).
### External
- `numba` (`cuda.jit`, `int32`); `attrs` (`define`, `field`, `Factory`, `cmp_using`,
  `evolve`); `numpy` (`int_`, `asarray`, `array_equal`, `arange`, `unique`).
