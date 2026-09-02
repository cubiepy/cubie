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
| `ode_loop.py` | `IVPLoop(CUDAFactory)` — registers the loop's buffers and compiles the integration-loop closure in `build()`; `IVPLoopCache(CUDADispatcherCache)` holds `loop_function`; exports `ALL_LOOP_SETTINGS`. |
| `ode_loop_config.py` | `ODELoopConfig(CUDAFactoryConfig)` — system sizes, 14 buffer-location fields (default `'local'`), `OutputCompileFlags`, timing fields, mode flags (`save_last`, `save_regularly`, `summarise_regularly`, `is_adaptive`), and device-function references; `samples_per_summary` property with integer-multiple validation. |
| `__init__.py` | Re-exports `IVPLoop`. |

## For AI Agents

### Registered buffers
`register_buffers()` registers 14 buffers: 11 at run precision — `state`,
`proposed_state`, `parameters`, `drivers`, `proposed_drivers`, `observables`,
`proposed_observables`, `error`, `state_summary`, `observable_summary`,
`dt` (size 1) — plus three **`np_int32`** buffers: `counters` (the per-save
iteration-counter accumulator), `accept_step` (size 1, the controller's accept
flag), and `proposed_counters` (size 2, holding Newton/Krylov iteration
counts). `dt[0]` and `accept_step[0]` are how the controller returns the next
step and its accept flag.

**Child allocators:** `IVPLoop` does not call `get_child_allocators()` itself — its
parent `SingleIntegratorRunCore` registers `_algo_step` and `_step_controller` as
children under names `'algorithm'` and `'controller'`. `build()` then fetches the
`"algorithm_shared"`, `"algorithm_persistent"`, `"controller_shared"`,
`"controller_persistent"` allocators. The parent registers its `DAEInitialiser`
as child `'initialiser'` (aliasing `algorithm_shared`); `build()` fetches the
`"initialiser_shared"` / `"initialiser_persistent"` allocators unconditionally.
If the parent hasn't registered the children before `build()`, the child
allocators are absent and `build()` fails.

### Consistent initialisation at loop entry
The loop calls `initialise_state` once at entry — after the state/parameter seed
and the t0 driver evaluation, before the t0 observables evaluation and save. The
`DAEInitialiser` supplies the function, a compiled no-op returning 0 for non-DAE
configurations. Its Newton/linear iteration counts land in the t0 save's counter row. A nonzero return carries the solver bits plus
`DAE_INITIALISATION_FAILED`; the loop ORs it into `status` and seeds
`irrecoverable`: the run ends at the t0 save with the uncorrected values.

### Timing & output scheduling
Three independent timing parameters drive what the loop emits and when; each has a
`next_*` event time the step is clamped not to overshoot (see boundary clamping):
- **`save_every`** — interval between full-state saves. When a step ends at/after
  `next_save`, the loop calls `save_state_fn` (writing the current `state`/`observables`
  and per-save iteration counters to the next output row), advances `save_idx`, resets
  the counters, and does `next_save += save_every`.
- **`sample_summaries_every`** — interval between summary *samples*. When a step ends
  at/after `next_update_summary`, the loop calls `update_summaries_fn` to accumulate one
  sample into the running `state_summary`/`observable_summary` buffers, advances
  `update_idx`, and does `next_update_summary += sample_summaries_every`.
- **`summarise_every`** — interval between summary *outputs*. Not used directly in the
  loop; it sets `samples_per_summary = summarise_every / sample_summaries_every`
  (validated to be an integer multiple). After every `samples_per_summary` updates
  (`update_idx % samples_per_summary == 0`), the loop calls `save_summaries_fn` to flush
  the accumulated window to the next summary row and reset, advancing `summary_idx`.

In short: `save_state_fn` fires on the `save_every` grid, `update_summaries_fn` on the
`sample_summaries_every` grid, and `save_summaries_fn` once per `summarise_every` window.
Output calls are predicated on step acceptance (`do_save &= accept`,
`do_update_summary &= accept`); `save_regularly` / `summarise_regularly` gate whether the
regular grids are active at all (vs. `save_last`-only).

### Loop behaviour
- **Termination:** the `while True` loop exits via `return status` gated by
  `all_sync(mask, finished)`, so a whole warp exits together even if some lanes finished
  earlier. `finished` is true once all scheduled outputs pass `t_end`, or when
  `irrecoverable` is set. `irrecoverable` is set by: a fixed-mode step failure
  (`step_status != 0`); the controller signalling step-too-small (status bit `0x8`); or
  stagnation (`stagnant_counts >= 2`, which also ORs `0x40` into `status`).
- **Time is `float64`:** `t = float64(t0)` regardless of system precision; `t_prec =
  precision(t)` is the low-precision copy passed to device functions. This avoids
  accumulation drift over long integrations. Time conversions happen before the
  controller call to hide f64-pipe latency; `t` and `t_prec` commit via
  `selp(accept, ...)`.
- **Predicated commit:** state/driver/observable buffers are updated via
  `selp(accept, new, old)`, and `do_save`/`do_update_summary` are AND-masked with
  `accept` before the output calls.
- **Output-boundary clamping:** when an output is due, `dt_eff` is clamped to
  `next_event - t_prec` (`next_event = min(next_save, next_update_summary, t_end)`); the
  controller-proposed `dt_raw` is preserved and resumes after the boundary. The loop
  passes `truncated = (dt_eff != dt_raw)` to the controller (see
  `../step_control/AGENTS.md` for the freeze semantics).
  Events are judged against `min(next_event, t_end)` on `end_of_step = t_prec + dt_raw`; the narrowed landing is capped at `t_prec + dt_eff` whenever that sum exceeds `t_prec`.
- **Stagnation** counts consecutive no-progress steps (not wall-clock): one step that
  doesn't advance `t` (e.g. `dt_eff` rounding to zero at a save boundary) is tolerated;
  two in a row trips `irrecoverable` and sets `status |= 0x40`.
- The `& 0x8` mask is `CUBIE_RESULT_CODES.STEP_TOO_SMALL` (the controller's reject-at-min);
  the `0x40` OR'd on stagnation is `CUBIE_RESULT_CODES.STAGNATION`. Both are captured as
  device closure constants from `cubie/result_codes.py`.

### Config & settings
- `samples_per_summary` (on `ODELoopConfig`) raises `ValueError` if `summarise_every` is
  not within 1% of an integer multiple of `sample_summaries_every`; a ≤1% deviation is
  accepted with a `UserWarning` and `summarise_every` adjusted. Evaluated on every
  `build()`.
- `ALL_LOOP_SETTINGS` is the exported set of recognised parameter names that parents
  (`SingleIntegratorRunCore`) filter updates against; add a name here when a new
  `ODELoopConfig` field should be externally configurable.

### Testing
Tests in `tests/integrators/loops/`. Loop correctness is also exercised end-to-end in
`tests/batchsolving/` and against `tests/integrators/cpu_reference.py`
(`run_reference_loop()`).

## Dependencies
### Internal
- `cubie.CUDAFactory`; `cubie.buffer_registry`; `cubie._utils` (`PrecisionDType`,
  `unpack_dict_values`, `build_config`, validators); `cubie.cuda_simsafe`
  (`activemask`, `all_sync`, `selp`, `compile_kwargs`);
  `cubie.outputhandling.output_config` (`OutputCompileFlags`).
### External
- `numba` (`cuda.jit`, `int32`, `float64`, `bool_`); `numpy` (`int32 as np_int32`, for
  the integer-typed buffers); `attrs`.
