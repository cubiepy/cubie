<!-- Parent: ../AGENTS.md -->

# step_control

## Purpose
Step-size controllers for the integrators. Each controller is a `CUDAFactory`
subclass compiling a device function that, given the latest error estimate, decides
whether to accept the step, proposes the next `dt`, and returns a status code.
`get_controller(precision, settings)` resolves a controller by the
`settings["step_controller"]` key against `_CONTROLLER_REGISTRY` (`"fixed"`, `"i"`,
`"pi"`, `"pid"`, `"gustafsson"`).

See `CUDAFactory` (repo root) for the build/cache/`update`, buffer-registry, and
attrs-config mechanics common to all factories; this file documents only the
controllers.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Exports the controller classes, `get_controller`, `_CONTROLLER_REGISTRY`. |
| `base_step_controller.py` | `BaseStepController` / `BaseStepControllerConfig` / `ControllerCache`; `ALL_STEP_CONTROLLER_PARAMETERS` (union of every controller's kwargs); `CONTROLLER_GAIN_PARAMETERS` (`kp`/`ki`/`kd`, excluded from swap carryover); `mass_flags` (one per state, default all differential, carried through `settings_dict` on swaps). |
| `adaptive_step_controller.py` | `BaseAdaptiveStepController` + `AdaptiveStepControlConfig` (shared adaptive config: `dt_min/max`, `atol/rtol`, `algorithm_order`, gain limits, deadband, safety); `build_error_norm` (the one error-norm device function every adaptive controller calls); `_ensure_sane_bounds`. |
| `fixed_step_controller.py` | `FixedStepController` — unconditional accept, returns `0`; no history. |
| `adaptive_I_controller.py` | `AdaptiveIController` (`IStepControlConfig`, `kp=1.0`) — integral-only; gain `safety·norm^(-kp/(2(1+order)))`; no history. |
| `adaptive_PI_controller.py` | `AdaptivePIController` (`PIStepControlConfig` extends `IStepControlConfig`, `kp=0.7`, `ki=-0.4`) — uses previous + current norm; gains take a float or callable of order. |
| `adaptive_PID_controller.py` | `AdaptivePIDController` (`PIDStepControlConfig` extends PI with `kd=0.0`) — uses two previous norms; `kd` likewise. |
| `gustafsson_controller.py` | `GustafssonController` (`safety=0.9`, `newton_target_iters=5`) — min of a basic gain and a Newton-iteration-aware predictive gain; stores previous `dt` + norm. |

## For AI Agents

### Device-function contract (the caller — `IVPLoop` — must match)
- Signature, identical across all controllers:
  `(dt, state, state_prev, error, niters, truncated, accept_out, shared_scratch, persistent_local)`.
- Writes `accept_out[0] = int32(1)` to accept the step, `int32(0)` to reject (a plain
  accept/reject flag, not a result code).
- `truncated`: set by the loop when it forced the step length onto an output boundary.
  On an **accepted** truncated step the adaptive controllers freeze `dt` and their
  history and return `SUCCESS`; a **rejected** one shrinks `dt` normally.
- Returns `CUBIE_RESULT_CODES.SUCCESS` normally, or `CUBIE_RESULT_CODES.STEP_TOO_SMALL`
  when the proposed step would fall at/below `dt_min` (reject-at-minimum-step — the loop
  uses this to stop adaptive retries). Both are captured as device closure constants from
  `cubie/result_codes.py`.

### Error norm
- `nrm2 = mean((|error_i| / (atol_i + rtol_i * max(|state_i|, |state_prev_i|)))**2)` over
  the rows whose `mass_flags` entry is set; non-finite results become `1e16`. Compiled once
  by `BaseAdaptiveStepController.build_error_norm`: a plain `range(n)` loop when every row
  is differential, otherwise a loop over a compile-time index array of the differential
  rows (the `saved_state_indices` pattern), so algebraic rows cost nothing at run time.

### History buffers
- Controllers that keep per-trajectory history register a single `timestep_buffer`:
  PI stores the previous error norm, PID the previous two norms, and Gustafsson the
  previous `dt` and norm (I and fixed keep no history). The slot count is the
  `_timestep_buffer_elements` class attribute (PI 1, PID/Gustafsson 2, fixed/I 0), which
  the base `register_buffers()` uses to register the buffer — controllers with 0 register
  nothing. There is **no** `persistent_local_elements` property; query the size the same way
  as any other buffer-registered factory, via the registry-derived
  `persistent_local_buffer_size`.

### Controller specifics
- **`_resolve_step_params`** differs by family: `FixedStepController` collapses
  `dt_min`/`dt_max`/`dt` to a single fixed `dt` (first non-None wins; the others are
  discarded). `BaseAdaptiveStepController` derives `dt_min = dt/100`, `dt_max = dt*100`
  from a lone `dt`, or `dt = sqrt(dt_min·dt_max)` when only bounds are given.
- **`_ensure_sane_bounds`** auto-corrects *derived* parameters silently but raises
  `ValueError` on user-supplied incompatible bounds (e.g. `dt_max < dt_min`).
- **Deadband**: `deadband_min == deadband_max == 1.0` elides the branch at compile time; accepted gains inside the band snap to 1.0; rejected steps skip the band and retry on the proportional gain (Gustafsson: basic gain).
- **`update` addition**: parameters present in `ALL_STEP_CONTROLLER_PARAMETERS` but not
  applicable to the current controller emit a `UserWarning` and are dropped (so
  cross-controller kwarg forwarding is safe); genuinely unknown keys still raise
  `KeyError` per the base contract.
- **Gain carryover**: `settings_dict` excludes `CONTROLLER_GAIN_PARAMETERS`
  (`kp`/`ki`/`kd`); a swapped-in controller uses its own gain defaults unless the
  ordering update supplies gains explicitly.
- **Adding a controller**: subclass the config + controller bases, set `_config_class`,
  implement `build_controller(...) → ControllerCache`, register the controller's history
  buffer (if any) in `register_buffers()`, register in `_CONTROLLER_REGISTRY`, and add any
  new fields to `ALL_STEP_CONTROLLER_PARAMETERS`.

### Gotchas
- **Uninitialised history**: first-call guards fall back to the current norm (PI/PID)
  or `max(..., 1e-16)` (Gustafsson) — there is no explicit buffer pre-fill.

### Testing
Tests under `tests/integrators/step_control/` (`test_controllers.py`,
`test_adaptive_step_controller.py`, `test_fixed_step_controller.py`); CPU reference in
`tests/integrators/cpu_reference/step_controllers.py`.

## Dependencies
Internal: `CUDAFactory`; `_utils` (`build_config`, `clamp_factory`, validators,
`tol_converter`, `PrecisionDType`); `buffer_registry` (`timestep_buffer`);
`cuda_simsafe` (`selp`, `compile_kwargs`). Consumed by `integrators.loops` /
`SingleIntegratorRun`.
External: `numba.cuda`, `attrs`, `numpy`, `math`.
