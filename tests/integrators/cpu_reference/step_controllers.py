"""CPU reference implementations of step controllers."""

import numpy as np

from cubie._utils import PrecisionDType

from .cpu_utils import Array


class CPUAdaptiveController:
    """Simple adaptive step controller mirroring GPU heuristics."""

    def __init__(
        self,
        *,
        kind: str,
        dt: float,
        dt_min: float,
        dt_max: float,
        atol: float,
        rtol: float,
        order: int,
        precision: PrecisionDType,
        integral_gain: float = 0.3,
        proportional_gain: float = 0.4,
        derivative_gain: float = 0.0,
        safety: float = 0.9,
        min_step_shrink: float = 0.5,
        max_step_growth: float = 2.0,
        newton_target_iters: int = 5,
        deadband_min: float = 1.0,
        deadband_max: float = 1.0,
        mass_flags=None,
    ) -> None:
        self.kind = kind.lower()
        # Rows with a zero mass diagonal stay out of the error norm.
        self.mass_flags = (
            None if mass_flags is None else np.asarray(mass_flags, dtype=bool)
        )
        self.dt_min = precision(dt_min)
        self.dt_max = precision(dt_max)
        # Without a user dt, run_reference_loop sets an adaptive run's
        # first step from starting_step.
        self.estimates_dt = dt is None and self.kind != "fixed"
        if dt is not None:
            self.dt0 = precision(dt)
        else:
            self.dt0 = precision(np.sqrt(dt_min * dt_max))
        self.dt = self.dt0
        self.atol = precision(atol)
        self.rtol = precision(rtol)
        self.order = order
        self.precision = precision
        self.safety = precision(safety)
        self.min_step_shrink = precision(min_step_shrink)
        self.max_step_growth = precision(max_step_growth)
        self.integral_gain = precision(integral_gain)
        self.proportional_gain = precision(proportional_gain)
        self.derivative_gain = precision(derivative_gain)
        self.newton_target_iters = int(newton_target_iters)
        self.deadband_min = precision(deadband_min)
        self.deadband_max = precision(deadband_max)
        self.unity_gain = precision(1.0)
        self._deadband_disabled = (self.deadband_min == self.unity_gain) and (
            self.deadband_max == self.unity_gain
        )
        zero = precision(0.0)
        self._history = [zero, zero]
        self._step_count = 0
        self._convergence_failed = False
        self._rejections_at_dt_min = 0
        self._prev_nrm2 = zero
        self._prev_prev_nrm2 = zero
        self._prev_dt = zero

    @property
    def is_adaptive(self) -> bool:
        return self.kind != "fixed"

    @property
    def prev_dt(self) -> np.floating:
        """Return the previous step size used by the controller."""

        return self._prev_dt

    def error_norm(
        self, state_prev: Array, state_new: Array, error: Array
    ) -> float:
        precision = self.precision
        if self.mass_flags is not None:
            error = error[self.mass_flags]
            state_prev = state_prev[self.mass_flags]
            state_new = state_new[self.mass_flags]
        # atol floors at 1e-16 on the host, as the device norm's does.
        atol = max(self.atol, precision(1e-16))
        scale = atol + self.rtol * np.maximum(
            np.abs(state_prev), np.abs(state_new)
        )
        ratio = np.abs(error) / scale
        # Reciprocal multiply, not division: rounding-sensitive.
        inv_n = precision(1.0 / len(error))
        return precision(np.sum(ratio * ratio) * inv_n)

    def propose_dt(
        self,
        error_vector: Array,
        prev_state: Array,
        new_state: Array,
        niters: int = 0,
        truncated: bool = False,
    ) -> bool:
        self._step_count += 1
        if not self.is_adaptive:
            return True
        errornorm = self.error_norm(
            state_prev=prev_state,
            state_new=new_state,
            error=error_vector,
        )

        accept = errornorm <= self.precision(1.0)

        # An accepted truncated step leaves dt and history unchanged.
        if truncated and accept:
            return accept

        current_dt = self.dt
        gain = self._gain(
            errornorm=errornorm,
            accept=accept,
            niters=niters,
            current_dt=current_dt,
        )

        unclamped_dt = self.precision(current_dt * gain)
        new_dt = self._clamp(unclamped_dt, self.dt_min, self.dt_max)
        self.dt = new_dt
        if accept:
            self._prev_dt = current_dt
            self._prev_prev_nrm2 = self._prev_nrm2
            self._prev_nrm2 = errornorm

        return accept

    def starting_step(
        self,
        evaluator,
        driver_evaluator,
        state: Array,
        params: Array,
        drivers: Array,
        observables: Array,
        time: float,
    ) -> np.floating:
        """Return the starting-step estimate at ``state``."""
        precision = self.precision
        derivative, _ = evaluator.rhs(
            state, params, drivers, observables, time
        )
        derivative = np.asarray(derivative, dtype=precision)
        state_norm2 = self.error_norm(state, state, state)
        derivative_norm2 = self.error_norm(state, state, derivative)

        # A system at rest divides by zero without error on the device.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            zero_norm2 = precision(1e-5**2)
            if state_norm2 < zero_norm2 or derivative_norm2 < zero_norm2:
                first_guess = precision(1e-6)
            else:
                first_guess = precision(
                    precision(0.01) * np.sqrt(state_norm2 / derivative_norm2)
                )
            first_guess = self._clamp(first_guess, self.dt_min, self.dt_max)

            # Hold the algebraic rows over the Euler step.
            euler_state = (state + first_guess * derivative).astype(precision)
            if self.mass_flags is not None:
                euler_state = np.where(self.mass_flags, euler_state, state)
                euler_state = euler_state.astype(precision)
            end_time = precision(time + first_guess)
            end_drivers = driver_evaluator(end_time)
            end_observables = evaluator.observables(
                euler_state, params, end_drivers, end_time
            )
            end_derivative, _ = evaluator.rhs(
                euler_state, params, end_drivers, end_observables, end_time
            )
            change = (
                np.asarray(end_derivative, dtype=precision) - derivative
            ).astype(precision)
            # Two divisions, not one by the square: rounding-sensitive.
            second_derivative_norm2 = precision(
                self.error_norm(state, state, change)
                / first_guess
                / first_guess
            )

            largest_norm2 = np.fmax(derivative_norm2, second_derivative_norm2)
            exponent = precision(0.5 / (self.order + 1))
            second_guess = precision(
                (precision(0.01**2) / largest_norm2) ** exponent
            )
            estimate = np.fmin(precision(100.0) * first_guess, second_guess)
            return self._clamp(estimate, self.dt_min, self.dt_max)

    def _clamp(self, value, minimum, maximum):
        """Clamp with the device's fmax/fmin NaN handling and order."""
        return self.precision(np.fmax(minimum, np.fmin(value, maximum)))

    def _gain(
        self,
        *,
        errornorm: float,
        accept: bool,
        niters: int,
        current_dt: float,
    ) -> float:
        # A zero norm raises inf/nan gains on the device without error.
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._gain_impl(
                errornorm=errornorm,
                accept=accept,
                niters=niters,
                current_dt=current_dt,
            )

    def _gain_impl(
        self,
        *,
        errornorm: float,
        accept: bool,
        niters: int,
        current_dt: float,
    ) -> float:
        precision = self.precision
        # Integer denominator: rounding-sensitive.
        order_denominator = 2 * (self.order + 1)
        expo_fraction = precision(1.0 / order_denominator)

        if self.kind == "i":
            expo1 = precision(self.integral_gain / order_denominator)
            exponent = -expo1
            gain = self.safety * precision(errornorm**exponent)
            gain_reject = gain

        elif self.kind == "pi":
            beta1 = self.integral_gain + self.proportional_gain
            beta2 = -self.proportional_gain
            expo1 = precision(beta1 / order_denominator)
            expo2 = precision(beta2 / order_denominator)
            prev = self._prev_nrm2 if self._prev_nrm2 > 0.0 else errornorm
            gain = (
                self.safety
                * precision(errornorm**-expo1)
                * precision(prev**-expo2)
            )
            gain_reject = self.safety * precision(errornorm**-expo1)

        elif self.kind == "pid":
            beta1 = (
                self.integral_gain
                + self.proportional_gain
                + self.derivative_gain
            )
            beta2 = -(self.proportional_gain + 2 * self.derivative_gain)
            beta3 = self.derivative_gain
            expo1 = precision(beta1 / order_denominator)
            expo2 = precision(beta2 / order_denominator)
            expo3 = precision(beta3 / order_denominator)
            prev_nrm2 = self._prev_nrm2 if self._prev_nrm2 > 0.0 else errornorm
            prev_prev = (
                self._prev_prev_nrm2
                if self._prev_prev_nrm2 > 0.0
                else prev_nrm2
            )
            gain = (
                self.safety
                * precision(errornorm**-expo1)
                * precision(prev_nrm2**-expo2)
                * precision(prev_prev**-expo3)
            )
            gain_reject = self.safety * precision(errornorm**-expo1)

        elif self.kind == "gustafsson":
            large = precision(1e16)
            errornorm = errornorm if errornorm <= large else large
            one = precision(1.0)
            two = precision(2.0)
            # The device uses the step's Newton count as reported.
            niters_eff = precision(niters)
            target_iters = self.newton_target_iters
            dt_prev = max(precision(1e-16), self._prev_dt)
            nrm2_prev = max(precision(1e-16), self._prev_nrm2)
            fac = min(
                self.safety,
                ((one + two * target_iters) * self.safety)
                / (niters_eff + two * target_iters),
            )
            gain_basic = precision(fac * (errornorm**-expo_fraction))

            # Always compute gain_gus, then fallback to gain_basic if needed
            ratio = (errornorm * errornorm) / nrm2_prev
            gain_gus = (
                self.safety
                * (current_dt / dt_prev)
                * precision(ratio**-expo_fraction)
            )
            gain = gain_gus if gain_gus < gain_basic else gain_basic
            # Fallback to gain_basic if step not accepted or no previous dt
            use_gus = accept and (self._prev_dt > precision(1e-16))
            gain = gain if use_gus else gain_basic
            gain_reject = gain_basic
        else:
            gain = precision(1.0)
            gain_reject = precision(1.0)

        gain = self._clamp(gain, self.min_step_shrink, self.max_step_growth)
        if not self._deadband_disabled:
            if self.deadband_min <= gain <= self.deadband_max:
                gain = self.unity_gain
        if not accept:
            # Rejected steps retry on the current error alone.
            if self.kind == "gustafsson":
                gain = self._clamp(
                    gain_reject, self.min_step_shrink, self.max_step_growth
                )
            else:
                gain = np.fmax(self.min_step_shrink, gain_reject)
        return precision(gain)


__all__ = ["CPUAdaptiveController"]
