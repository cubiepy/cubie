<!-- Parent: ../AGENTS.md -->

# summarymetrics

## Purpose
The extensible registry of CUDA summary metrics accumulated per thread over each summary
window. Each metric is a `SummaryMetric(CUDAFactory)` subclass compiling two device
functions (`update` per step, `save` per window). A module-level `summary_metrics`
singleton auto-registers 18 built-ins at import time via the `@register_metric` decorator,
and dispatches buffer sizing, offsets, and device functions for a requested metric set.

## Key Files
| File | Description |
|------|-------------|
| `metrics.py` | Infrastructure: `MetricFuncCache(update_fn, save_fn)`, `MetricConfig` (adds `sample_summaries_every`, default 0.01), `SummaryMetric` (abstract `CUDAFactory` base), `SummaryMetrics` (registry/dispatcher), `register_metric` decorator. |
| `__init__.py` | Creates the `summary_metrics` singleton (`precision=float32` as a bootstrap placeholder — the singleton is built at import time before precision is known; `OutputFunctions.build()` overwrites it via `summary_metrics.update(precision=…)`) then imports every metric module to register it. Exports `summary_metrics`, `register_metric`. |

### Built-in metrics (buffer_size / output_size)
| Metric | Sizes | Notes |
|--------|-------|-------|
| `mean` | 1 / 1 | running sum → `sum/summarise_every`. |
| `max` / `min` | 1 / 1 | running extremum; sentinel `-1e30` / `+1e30`. |
| `rms` | 1 / 1 | sum of squares → `sqrt(sum/summarise_every)`. |
| `std` | 3 / 1 | shifted-data algorithm (`[shift, Σ, Σ²]`). |
| `max_magnitude` | 1 / 1 | max `abs(value)`; sentinel `0.0`. |
| `peaks` / `negative_peaks` | `3+n` / `n` | local maxima/minima as summary-sample numbers (`current_index - 1`), counted from the run's first sample; unused slots hold 0; request `"peaks[5]"`; unit `s`. |
| `extrema` | 2 / 2 | max+min; substitutes `{max,min}`. |
| `mean_std` | 3 / 2 | substitutes `{mean,std}`. |
| `mean_std_rms` | 3 / 3 | substitutes `{mean,std,rms}`. |
| `std_rms` | 3 / 2 | substitutes `{std,rms}`. |
| `dxdt_max` / `dxdt_min` | 2 / 1 | first derivative (forward diff), scaled by `sample_summaries_every`; unit `[unit]*s^-1`. |
| `dxdt_extrema` | 3 / 2 | substitutes `{dxdt_max,dxdt_min}`. |
| `d2xdt2_max` / `d2xdt2_min` | 3 / 1 | second derivative (central diff), scaled by `sample_summaries_every²`; unit `[unit]*s^-2`. |
| `d2xdt2_extrema` | 4 / 2 | substitutes `{d2xdt2_max,d2xdt2_min}`. |

## Metric device contract
A metric's `build()` returns `MetricFuncCache(update_fn=update, save_fn=save)`, both
`@cuda.jit(device=True, inline=True)`:
- `update(value, buffer, current_index, customisable_variable)`: one sample.
- `save(buffer, output_array, summarise_every, customisable_variable)`: one window;
  writes the result(s) to `output_array` and resets `buffer` to its sentinel.
  `summarise_every` is the number of samples in the window: the loop passes
  `samples_per_summary`, or the run's sample count for a single end-of-run summary.

Both receive per-variable slices starting at offset 0; the registry owns the offsets.
`customisable_variable` is the parsed request parameter (`n` for `peaks[n]`), `0` for
fixed metrics. `current_index` is a global summary-sample counter that does not reset
per window.

## Registry
- `summary_metrics` (the singleton) answers everything for a requested metric-name
  list: `buffer_sizes`/`buffer_offsets`/`summaries_buffer_height`, the output
  equivalents, `legend`, `unit_modifications`, `params`, and
  `update_functions`/`save_functions` (pulled lazily from each metric's cache).
- `@register_metric(summary_metrics)` instantiates the class
  (`cls(registry.precision)`) and registers it at import; `__init__.py` creates the
  singleton before importing the metrics.
- `summary_metrics.update(precision=…, sample_summaries_every=…)` updates every
  registered metric with `silent=True`.

## Combined metrics
`_combined_metrics` (`metrics.py`) maps a `frozenset` of names to one combined metric:
`{mean,std,rms}`→`mean_std_rms`, `{mean,std}`→`mean_std`, `{std,rms}`→`std_rms`,
`{max,min}`→`extrema`, `{dxdt_max,dxdt_min}`→`dxdt_extrema`,
`{d2xdt2_max,d2xdt2_min}`→`d2xdt2_extrema`. `preprocess_request` parses parameters,
substitutes larger combinations first when the whole set is present, then drops
unregistered names with a `UserWarning`. The `legend` method takes a metric's `output_names`
when the metric declares them (`max` and `min` for `extrema`) and otherwise numbers
the columns of a multi-output metric (`peaks_1`, `peaks_2`).

## Buffers and sizing
- `buffer_size`/`output_size` are an `int` or a `callable(n)` resolved from the parsed
  parameter; a callable size with parameter `0` warns and is evaluated at 0 (a bare
  `"peaks"` gets a 3-slot buffer and no outputs).
- `peaks`/`negative_peaks`: `buffer_size = 3 + n`, `output_size = n`.
- `dxdt_*`/`d2xdt2_*` capture `sample_summaries_every` (or its square) at build and scale
  the finite difference in `save`, committing the extremum with `selp`.
- `save` restores each slot's sentinel (`-1e30` max/dxdt-max, `+1e30` min, `0.0`
  magnitude/sums), which is also the required initial state. The loop calls the save
  chain once at `t0` to set that state, only when `settling_time == 0`. `save` leaves the
  peak and derivative history slots alone, so the first samples of a window are compared
  with the last samples of the window before.
- `std`/`mean_std`/`mean_std_rms`/`std_rms` shift the data by `buffer[0]`: the run's
  first sample, stored when `current_index == 0`, then the previous window's mean, which
  `save` stores. The `rms` update function starts its sum from zero at
  `current_index == 0`.

## Adding a metric
Subclass `SummaryMetric` (set `name`, `buffer_size`, `output_size`,
`unit_modification`, and `output_names` for a combined metric), implement `build()` returning `MetricFuncCache(update_fn=update, save_fn=save)`,
decorate with `@register_metric(summary_metrics)` and import the module in
`__init__.py`. To combine with others, add a `frozenset` entry to `_combined_metrics`.

## Dependencies
### Internal
- `cubie.CUDAFactory` (`CUDAFactory`, `CUDAFactoryConfig`, `CUDADispatcherCache`);
  `cubie._utils` (`gttype_validator`, `PrecisionDType`, `precision_converter`,
  `precision_validator`); `cubie._cudasim_extensions` (`cuda`). Metrics compile with
  `self.jit_kwargs` from `CUDAFactory`.
### External
- `numba_cuda_mlir` (`types.int32`); `numpy` (`float32`, `dtype`, `int32`); `attrs`;
  `math` (`sqrt`, `fabs`).
