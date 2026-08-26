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
| `metrics.py` | Infrastructure: `MetricFuncCache(update, save)`, `MetricConfig` (adds `sample_summaries_every`, default 0.01), `SummaryMetric` (abstract `CUDAFactory` base), `SummaryMetrics` (registry/dispatcher), `register_metric` decorator. |
| `__init__.py` | Creates the `summary_metrics` singleton (`precision=float32` as a bootstrap placeholder — the singleton is built at import time before precision is known; `OutputFunctions.build()` overwrites it via `summary_metrics.update(precision=…)`) then imports every metric module to register it. Exports `summary_metrics`, `register_metric`. |

### Built-in metrics (buffer_size / output_size)
| Metric | Sizes | Notes |
|--------|-------|-------|
| `mean` | 1 / 1 | running sum → `sum/summarise_every`. |
| `max` / `min` | 1 / 1 | running extremum; sentinel `-1e30` / `+1e30`. |
| `rms` | 1 / 1 | sum of squares → `sqrt(sum/summarise_every)`. |
| `std` | 3 / 1 | shifted-data algorithm (`[shift, Σ, Σ²]`). |
| `max_magnitude` | 1 / 1 | max `abs(value)`; sentinel `0.0`. |
| `peaks` / `negative_peaks` | `3+n` / `n` | local maxima/minima step indices; request `"peaks[5]"`. |
| `extrema` | 2 / 2 | max+min; substitutes `{max,min}`. |
| `mean_std` | 3 / 2 | substitutes `{mean,std}`. |
| `mean_std_rms` | 3 / 3 | substitutes `{mean,std,rms}`. |
| `std_rms` | 3 / 2 | substitutes `{std,rms}`. |
| `dxdt_max` / `dxdt_min` | 2 / 1 | first derivative (forward diff), scaled by `sample_summaries_every`; unit `[unit]*s^-1`. |
| `dxdt_extrema` | 3 / 2 | substitutes `{dxdt_max,dxdt_min}`. |
| `d2xdt2_max` / `d2xdt2_min` | 3 / 1 | second derivative (central diff), scaled by `sample_summaries_every²`; unit `[unit]*s^-2`. |
| `d2xdt2_extrema` | 4 / 2 | substitutes `{d2xdt2_max,d2xdt2_min}`. |

## For AI Agents

### Metric device contract
A metric's `build()` returns `MetricFuncCache(update, save)`, both
`@cuda.jit(device=True, inline=True)`:
- `update(value, buffer, current_index, customisable_variable)` — one integration step.
- `save(buffer, output_array, summarise_every, customisable_variable)` — one window; writes
  the result(s) to `output_array` and resets `buffer` to its sentinel.

Both receive per-variable slices starting at offset 0; the registry owns the offset
arithmetic. `customisable_variable` is the parsed request parameter (e.g. `n` for
`peaks[n]`), `0` for fixed metrics. `current_index` is a global, monotonic summary-sample
counter (it does not reset per window).

### Registry & registration
- `summary_metrics` (the singleton) dispatches everything for a requested metric-name list:
  `buffer_sizes`/`buffer_offsets`/`summaries_buffer_height`, the output equivalents,
  `legend`, `unit_modifications`, `params`, and `update_functions`/`save_functions`
  (device functions pulled lazily from each metric's cache). Every dispatch method takes
  the request list as an argument.
- `@register_metric(summary_metrics)` instantiates the class (`cls(registry.precision)`)
  and registers it **at import time**; `__init__.py` must create the singleton before the
  metric imports.
- `summary_metrics.update(precision=…, sample_summaries_every=…)` propagates to every
  registered metric. `SummaryMetric.update` uses `update_compile_settings(..., silent=True)`,
  so kwargs a given metric doesn't recognise are dropped rather than raising — that is what
  lets the registry broadcast one update to all metrics.

### Combined-metric substitution
`_combined_metrics` (metrics.py) maps a `frozenset` of individual names to one combined
metric: `{mean,std,rms}`→`mean_std_rms`, `{mean,std}`→`mean_std`, `{std,rms}`→`std_rms`,
`{max,min}`→`extrema`, `{dxdt_max,dxdt_min}`→`dxdt_extrema`,
`{d2xdt2_max,d2xdt2_min}`→`d2xdt2_extrema`. `preprocess_request` (called by every dispatch
method) parses params, substitutes larger combinations first when the whole set is present
and the combined metric is registered, then drops unregistered names with a `UserWarning`.
Combined metrics compute shared statistics once and emit multiple outputs; `legend`
numbers multi-output columns (`extrema_1`, `extrema_2`).

### Buffers & sizing
- `buffer_size`/`output_size` are an `int` or a `callable(n)`. `_get_size` resolves
  callables from the parsed parameter; a callable size with parameter `0` warns and yields
  size 0.
- Parameterized metrics (`peaks`/`negative_peaks`): `buffer_size=lambda n: 3+n`,
  `output_size=lambda n: n`; the parsed `n` flows through to `customisable_variable`.

### Derivative & peak specifics
- `dxdt_*`/`d2xdt2_*` capture `sample_summaries_every` (or its square) into the closure at
  build and scale the finite difference in `save`; they track the unscaled-derivative
  extremum with a `selp` predicated commit.
- Sentinels: `save` restores each slot to its reset value (`-1e30` max/dxdt-max, `+1e30`
  min, `0.0` magnitude/sums); these are also the required initial buffer state. The
  peak/derivative history slots (previous value(s)) are **not** reset by `save`, so history
  is continuous across window boundaries.
- `std`/`mean_std`/`mean_std_rms`/`std_rms` use a shifted-data algorithm: `buffer[0]` holds
  the window's first sample as the shift, set when `current_index == 0`.

### Adding a metric
Subclass `SummaryMetric` (set `name`, `buffer_size`, `output_size`, `unit_modification`),
implement `build()` returning `MetricFuncCache(update, save)`, decorate with
`@register_metric(summary_metrics)`, and import the module in `__init__.py`. To auto-combine
with others, add a `frozenset` entry to `_combined_metrics`.

### Testing
`tests/outputhandling/summarymetrics/test_summary_metrics.py` — registry logic,
`register_metric`, combined substitution, `parse_string_for_params`, buffer/output sizing,
`legend`/`unit_modifications`, and a check that all 18 built-ins register.

## Dependencies
### Internal
- `cubie.CUDAFactory` (`CUDAFactory`, `CUDAFactoryConfig`, `CUDADispatcherCache`);
  `cubie._utils` (`gttype_validator`, `PrecisionDType`, `precision_converter`,
  `precision_validator`); `cubie.cuda_simsafe` (`selp`, `compile_kwargs`).
### External
- `numba` (`cuda.jit`, `int32`); `numpy` (`float32`, `floating`); `attrs`; `math`
  (`sqrt`, `fabs`).
