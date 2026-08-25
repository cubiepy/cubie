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
| `base_step_controller.py` | `BaseStepController` / `BaseStepControllerConfig` / `ControllerCache`; `ALL_STEP_CONTROLLER_PARAMETERS` (union of every controller's kwargs); `CONTROLLER_GAIN_NAMES` (PID gain keys) and `CONTROLLER_GAIN_PARAMETERS` (gain keys plus `filter_coefficients`, excluded from swap carryover); `BaseStepController.gain_names` (gain keys the config class carries); `GAIN_CONTROLLER_CHAIN` + `minimal_gain_controller` + `promoted_gain_controller` (smallest `i`/`pi`/`pid` for a set of nonzero gains; promotion only from a chain member); `FILTER_COEFFICIENT_PRESETS` + `filter_coefficients_to_gains` (`(beta1,beta2,beta3)` → gains). |
| `adaptive_step_controller.py` | `BaseAdaptiveStepController` + `AdaptiveStepControlConfig` (shared adaptive config: `dt_min/max`, `atol/rtol`, `algorithm_order`, `min/max_step_growth`, deadband, safety); `_ensure_sane_bounds`. |
| `fixed_step_controller.py` | `FixedStepController` — unconditional accept, returns `0`; no history. |
| `adaptive_I_controller.py` | `AdaptiveIController` (`IStepControlConfig`, `integral_gain=1.0`) — integral-only; gain `safety·norm^(-integral_gain/(2(1+order)))`; no history. |
| `adaptive_PI_controller.py` | `AdaptivePIController` (`PIStepControlConfig` extends `IStepControlConfig`, `integral_gain=0.3`, `proportional_gain=0.4`) — uses previous + current norm; gains take a float or callable of order. |
| `adaptive_PID_controller.py` | `AdaptivePIDController` (`PIDStepControlConfig` extends PI with `derivative_gain=0.0`) — uses two previous norms; `derivative_gain` likewise. |
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
- **Deadband**: `deadband_min == deadband_max == 1.0` elides the branch at compile time; accepted gains inside the band snap to 1.0; rejected steps skip the band and retry on the current-error term alone (Gustafsson: basic gain).
- **`update` addition**: parameters present in `ALL_STEP_CONTROLLER_PARAMETERS` but not
  applicable to the current controller emit a `UserWarning` and are dropped (so
  cross-controller kwarg forwarding is safe); genuinely unknown keys still raise
  `KeyError` per the base contract.
- **Gain carryover**: `settings_dict` excludes `CONTROLLER_GAIN_PARAMETERS`; a swapped-in controller uses its own gain defaults unless the ordering update supplies gains explicitly.
- **Gain semantics**: `beta1 = kI+kP+kD`, `beta2 = -(kP+2kD)`, `beta3 = kD`, each divided by `order+1` at build; `filter_coefficients` (beta triple or preset name) maps to gains on `i`/`pi`/`pid` and raises when mixed with explicit gains.
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
