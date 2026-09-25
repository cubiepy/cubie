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
| `base_step_controller.py` | `BaseStepController` / `BaseStepControllerConfig` / `ControllerCache`; `ALL_STEP_CONTROLLER_PARAMETERS` (union of every controller's kwargs); `CONTROLLER_GAIN_NAMES` (PID gain keys) and `CONTROLLER_GAIN_PARAMETERS` (gain keys plus `filter_coefficients`, excluded from swap carryover); `BaseStepController.gain_names` (gain keys the config class carries); `GAIN_CONTROLLER_CHAIN` + `minimal_gain_controller` + `promoted_gain_controller` (smallest `i`/`pi`/`pid` for a set of nonzero gains); `FILTER_COEFFICIENT_PRESETS` + `filter_coefficients_to_gains` (`(beta1,beta2,beta3)` → gains); `mass_flags` (one per state, default all differential, carried through `settings_dict` on swaps); `BaseStepController.build()` calls `compile_controller()` and fills the `ControllerCache`'s remaining fields from the same-named properties. |
| `adaptive_step_controller.py` | `BaseAdaptiveStepController` + `AdaptiveStepControlConfig` (shared adaptive config: `dt_min/max`, `atol/rtol`, `algorithm_order`, `min/max_step_growth`, deadband, safety); owns the `TwoRefMaskedScaledNorm` child `norm` (`update` forwards to it; its device function is the `eq=False` config field `norm_fn`). |
| `fixed_step_controller.py` | `FixedStepController` — unconditional accept, returns `0`; no history. |
| `adaptive_I_controller.py` | `AdaptiveIController` (`IStepControlConfig`, `integral_gain=1.0`) — integral-only; gain `safety·norm^(-integral_gain/(2(1+order)))`; no history. |
| `adaptive_PI_controller.py` | `AdaptivePIController` (`PIStepControlConfig` extends `IStepControlConfig`, `integral_gain=0.3`, `proportional_gain=0.4`) — uses previous + current norm; gains take a float or callable of order. |
| `adaptive_PID_controller.py` | `AdaptivePIDController` (`PIDStepControlConfig` extends PI with `derivative_gain=0.0`) — uses two previous norms; `derivative_gain` likewise. |
| `gustafsson_controller.py` | `GustafssonController` (`safety=0.9`, `newton_target_iters=5`) — min of a basic gain and a Newton-iteration-aware predictive gain; stores previous `dt` + norm. |

## Device-function contract (`IVPLoop` must match)
- Signature, identical for all controllers: `(dt, state, state_prev, error, niters,
  truncated, accept_out, shared_scratch, persistent_local)`.
- Writes `accept_out[0] = int32(1)` to accept, `int32(0)` to reject.
- `truncated` is set by the loop when it clamped the step onto an output boundary. An
  accepted truncated step leaves `dt` and the history unchanged and returns `SUCCESS`; a
  rejected one shrinks `dt` normally.
- Returns `SUCCESS`, or `STEP_TOO_SMALL` when the proposed step falls at or below
  `dt_min`, which ends the run's adaptive retries.

## Error norm
`nrm2 = mean((|error_i| / (atol_i + rtol_i * max(|state_i|, |state_prev_i|)))**2)` over
the rows whose `mass_flags` entry is set (`TwoRefMaskedScaledNorm`, `../norms.py`, called
as `error_norm(error, state, state_prev)`). A zero norm gives an `inf`/`nan` gain that
`clamp` (`fmax`/`fmin`, NaN dropped) resolves to `max_gain`; Gustafsson caps a non-finite
norm at `1e16` because its reject path runs through `clamp`.

## History buffers
Controllers with history register one `timestep_buffer` of
`_timestep_buffer_elements` slots: PI 1 (previous norm), PID 2 (two previous norms),
Gustafsson 2 (previous `dt` and norm); fixed and I register nothing. Query its size with
`persistent_local_buffer_size`. On the first call, PI/PID fall back to the current norm
and Gustafsson to `max(..., 1e-16)`; the buffer is not pre-filled.

## Controller specifics
- Step bounds are plain fields with `DEFAULT_*` defaults; an unset adaptive `dt` is the
  bounds' geometric mean; contradictory bounds raise `ValueError`.
- Deadband: `deadband_min == deadband_max == 1.0` compiles the branch out; accepted gains
  inside the band snap to 1.0; rejected steps skip the band and retry on the
  current-error term alone (Gustafsson: basic gain).
- `update` warns about and drops keys in `ALL_STEP_CONTROLLER_PARAMETERS` that the
  current controller does not use; unknown keys raise `KeyError`.
- Gains come from the Solver: the family defaults for the family's controller, the given
  gains otherwise; a swapped-in controller is built from the update's keys.
- `beta1 = kI+kP+kD`, `beta2 = -(kP+2kD)`, `beta3 = kD`, each divided by `order+1` at
  build. `filter_coefficients` (a beta triple or preset name) maps to gains on
  `i`/`pi`/`pid` and raises when mixed with explicit gains.
- Adding a controller: subclass the config and controller bases, set `_config_class`,
  implement `build_controller(...) -> ControllerCache` (or `compile_controller()` for a
  non-adaptive one), register any history buffer in `register_buffers()`, add it to
  `_CONTROLLER_REGISTRY` and its fields to `ALL_STEP_CONTROLLER_PARAMETERS`.
- CPU reference: `tests/integrators/cpu_reference/step_controllers.py`.

## Dependencies
Internal: `CUDAFactory`; `_utils` (`build_config`, `clamp_factory`, validators,
`tol_converter`, `PrecisionDType`); `buffer_registry` (`timestep_buffer`);
`cuda_simsafe` (`selp`, `compile_kwargs`); `integrators.norms`
(`TwoRefMaskedScaledNorm`). Consumed by `integrators.loops` /
`SingleIntegratorRun`.
External: `numba.cuda`, `attrs`, `numpy`, `math`.
