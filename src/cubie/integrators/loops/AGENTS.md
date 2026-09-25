<!-- Parent: ../AGENTS.md -->

# loops

## Purpose
The CUDA integration-loop factory. `IVPLoop(CUDAFactory)` compiles a single
`@cuda.jit(device=True, inline=True)` device function — the `loop_fn` closure — that
owns the entire per-thread integration lifecycle: buffer allocation, initial-value
seeding, the main `while True` timestep loop, predicated step commit, and all
output/summary scheduling, composing the compiled step, controller, output/summary,
and driver/observable device functions into one closure. `ODELoopConfig` holds every
compile-critical parameter shaping that closure (system sizes, buffer locations, timing
intervals, boolean control-flow constants, and device-function references).

## Key Files
| File | Description |
|------|-------------|
| `ode_loop.py` | `IVPLoop(CUDAFactory)` — registers the loop's buffers and compiles the integration-loop closure in `build()`; `IVPLoopCache(CUDADispatcherCache)` holds `loop_fn`; exports `ALL_LOOP_SETTINGS`. |
| `ode_loop_config.py` | `ODELoopConfig(CUDAFactoryConfig)` — system sizes, a location field per registered buffer (default `'local'`), `OutputCompileFlags`, timing fields, mode flags (`save_last`, `save_regularly`, `summarise_last`, `summarise_regularly`, `is_adaptive`), and device-function references; `samples_per_summary` property with integer-multiple validation. |
| `__init__.py` | Re-exports `IVPLoop`. |

## Registered buffers
Run precision: `state`, `proposed_state`, `parameters`, `drivers`, `proposed_drivers`,
`observables`, `proposed_observables`, `error`, `state_summary`, `observable_summary`,
`dt` (size 1). `np_int32`: `counters` (per-save iteration counts), `accept_step`
(size 1) and `proposed_counters` (size 2, Newton/Krylov). The controller returns the next
step in `dt[0]` and its accept flag in `accept_step[0]`.

`SingleIntegratorRunCore` registers the step, controller and `DAEInitialiser` as children
`'algorithm'`, `'controller'` and `'initialiser'` (the last aliasing `algorithm_shared`)
before `build()`, which fetches their `*_shared`/`*_persistent` allocators; `IVPLoop`
never calls `get_child_allocators()` itself.

## Consistent initialisation
The loop calls `initialise_state_fn` once after seeding state and parameters and
evaluating drivers at t0, before the t0 observables and save. Non-DAE configurations get
a no-op returning 0. The initialiser's iteration counts land in the t0 counter row. A
nonzero return (solver bits plus `DAE_INITIALISATION_FAILED`) is OR'd into `status` and
sets `irrecoverable`: the run ends at the t0 save with the uncorrected values.

## Output scheduling
Each timing parameter has a `next_*` event time the step is clamped not to overshoot:
- `save_every`: when a step reaches `next_save`, `save_state_fn` writes state,
  observables and per-save counters to the next row, then `next_save += save_every`.
- `sample_summaries_every`: when a step reaches `next_update_summary`,
  `update_summaries_fn` accumulates one sample, then
  `next_update_summary += sample_summaries_every`.
- `summarise_every`: sets `samples_per_summary = summarise_every /
  sample_summaries_every`; every `samples_per_summary` updates, `save_summaries_fn`
  flushes the window to the next summary row.
- `summarise_last`: samples on the `sample_summaries_every` grid and one
  `save_summaries_fn` at the `at_end` step, with `update_idx` as the divisor.

Outputs require acceptance (`do_save &= accept`, likewise for summaries).
`save_regularly`/`summarise` switch the grids on; otherwise only `save_last` applies.
`at_end` is the step landing on `t_end`.

## Loop behaviour
- The `while True` loop returns when `all_sync(mask, finished)`, so a warp exits
  together. `finished` is set once `save_count`/`summary_count` events have fired or on
  `irrecoverable`: a fixed-mode step failure, the controller's `STEP_TOO_SMALL`, or two
  consecutive steps that don't advance `t` (`STAGNATION`, which a single stalled step
  does not trigger).
- `t` is `float64`; `t_prec = precision(t)` goes to device functions.
  `t_next64 = t + float64(dt_raw)` and `t_next = narrow(t_next64)` are computed before
  the step in fixed mode and after the commit in adaptive mode; state, drivers,
  observables, `t` and `t_prec` commit with `selp(accept, new, old)`.
- When an output is due, `dt_eff = fmin(next_event - t_prec, dt_raw)` with
  `next_event = min(next_save, next_update_summary, t_end)`; `dt_raw` resumes after the
  boundary. The loop passes `truncated = (dt_eff != dt_raw)` to the controller. A clamped
  step commits the float64 event time in fixed mode and `t + float64(dt_eff)` in
  adaptive mode. `next_save` and `next_update_summary` clamp to `t_end` when they
  advance.

## Config
- `ODELoopConfig.samples_per_summary` raises `ValueError` unless `summarise_every` is
  within 1% of an integer multiple of `sample_summaries_every`; within 1% it warns and
  adjusts `summarise_every`. Evaluated on every `build()`.
- `ALL_LOOP_SETTINGS` lists the names parents filter updates against; add a new
  externally configurable `ODELoopConfig` field there.
- `tests/integrators/cpu_reference.py` (`run_reference_loop()`) is the CPU reference for
  loop behaviour.

## Dependencies
### Internal
- `cubie.CUDAFactory`; `cubie.buffer_registry`; `cubie._utils` (`PrecisionDType`,
  `unpack_dict_values`, `build_config`, validators); `cubie.cuda_simsafe`
  (`activemask`, `all_sync`, `selp`, `compile_kwargs`);
  `cubie.outputhandling.output_config` (`OutputCompileFlags`).
### External
- `numba` (`cuda.jit`, `int32`, `float64`, `bool_`); `numpy` (`int32 as np_int32`, for
  the integer-typed buffers); `attrs`.
