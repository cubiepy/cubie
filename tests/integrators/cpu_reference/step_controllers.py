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
        kp: float = 0.7,
        ki: float = -0.4,
        kd: float = 0.0,
        safety: float = 0.9,
        min_gain: float = 0.5,
        max_gain: float = 2.0,
        newton_target_iters: int = 20,
        deadband_min: float = 1.0,
        deadband_max: float = 1.2,
        beta1: float = None,
        beta2: float = None,
        qoldinit: float = 1.0e-4,
    ) -> None:
        self.kind = kind.lower()
        self.dt_min = precision(dt_min)
        self.dt_max = precision(dt_max)
        # A user dt seeds the first step; the geometric mean of the
        # bounds is only a fallback.
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
        self.min_gain = precision(min_gain)
        self.max_gain = precision(max_gain)
        self.kp = precision(kp)
        self.ki = precision(ki)
        self.kd = precision(kd)
        self.newton_target_iters = int(newton_target_iters)
        self.deadband_min = precision(deadband_min)
        self.deadband_max = precision(deadband_max)
        self.unity_gain = precision(1.0)
        self._deadband_disabled = (self.deadband_min == self.unity_gain) and (
            self.deadband_max == self.unity_gain
        )
        if beta1 is not None:
            self.beta1 = precision(beta1)
        else:
            self.beta1 = precision(7.0 / (10.0 * order))
        if beta2 is not None:
            self.beta2 = precision(beta2)
        else:
            self.beta2 = precision(2.0 / (5.0 * order))
        self.qoldinit = precision(qoldinit)
        zero = precision(0.0)
        self._history = [zero, zero]
        self._step_count = 0
        self._convergence_failed = False
        self._rejections_at_dt_min = 0
        self._prev_nrm2 = zero
        self._prev_prev_nrm2 = zero
        self._prev_dt = zero
        self._qold = zero

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
        error = np.maximum(np.abs(error), precision(1e-16))
        scale = self.atol + self.rtol * np.maximum(
            np.abs(state_prev), np.abs(state_new)
        )
        ratio = error / scale
        # Reciprocal multiply, not division: rounding-sensitive.
        inv_n = precision(1.0 / len(error))
        nrm2 = precision(np.sum(ratio * ratio) * inv_n)
        if np.isnan(nrm2) or np.isinf(nrm2):
            nrm2 = precision(1e16)
        return nrm2

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
        if self.kind == "sciml_pi":
            return self._propose_dt_sciml_pi(
                error_vector=error_vector,
                prev_state=prev_state,
                new_state=new_state,
                truncated=truncated,
            )
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
        new_dt = min(self.dt_max, max(self.dt_min, unclamped_dt))
        self.dt = new_dt
        if accept:
            self._prev_dt = current_dt
            self._prev_prev_nrm2 = self._prev_nrm2
            self._prev_nrm2 = errornorm

        return accept

    def _propose_dt_sciml_pi(
        self,
        *,
        error_vector: Array,
        prev_state: Array,
        new_state: Array,
        truncated: bool,
    ) -> bool:
        """Mirror the device SciML PI controller exactly.

        The error has no magnitude floor and the limiter acts in
        ``q``-space, matching OrdinaryDiffEq.jl.
        """
        precision = self.precision
        scale = self.atol + self.rtol * np.maximum(
            np.abs(prev_state), np.abs(new_state)
        )
        ratio = error_vector / scale
        inv_n = precision(1.0 / len(error_vector))
        nrm2 = precision(np.sum(ratio * ratio) * inv_n)
        if np.isnan(nrm2) or np.isinf(nrm2):
            nrm2 = precision(1e16)
        eest = precision(np.sqrt(nrm2))

        qold = self._qold if self._qold > 0.0 else self.qoldinit
        accept = eest <= precision(1.0)

        inv_qmin = precision(1.0 / self.min_gain)
        inv_qmax = precision(1.0 / self.max_gain)
        if eest == precision(0.0):
            q11 = precision(1.0)
            q_raw = inv_qmax
        else:
            q11 = precision(eest**self.beta1)
            q_raw = q11 / precision(qold**self.beta2)

        q_accept = max(inv_qmax, min(inv_qmin, q_raw / self.safety))
        q_reject = min(inv_qmin, q11 / self.safety)
        q = q_accept if accept else q_reject
        current_dt = self.dt
        dt_new_raw = precision(current_dt / q)

        if truncated and accept:
            return accept

        self.dt = min(self.dt_max, max(self.dt_min, dt_new_raw))
        if accept:
            self._prev_dt = current_dt
            self._qold = max(eest, self.qoldinit)
        return accept

    def _gain(
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
        kp_exp = precision(self.kp / order_denominator)
        ki_exp = precision(self.ki / order_denominator)
        kd_exp = precision(self.kd / order_denominator)

        if self.kind == "i":
            exponent = -expo_fraction
            gain = self.safety * precision(errornorm**exponent)

        elif self.kind == "pi":
            prev = self._prev_nrm2 if self._prev_nrm2 > 0.0 else errornorm
            gain = (
                self.safety
                * precision(errornorm**-kp_exp)
                * precision(prev**-ki_exp)
            )

        elif self.kind == "pid":
            prev_nrm2 = self._prev_nrm2 if self._prev_nrm2 > 0.0 else errornorm
            prev_prev = (
                self._prev_prev_nrm2
                if self._prev_prev_nrm2 > 0.0
                else prev_nrm2
            )
            gain = (
                self.safety
                * precision(errornorm**-kp_exp)
                * precision(prev_nrm2**-ki_exp)
                * precision(prev_prev**-kd_exp)
            )

        elif self.kind == "gustafsson":
            if niters == 0:
                raise ValueError("Gustafsson gain requires niters > 0")
            one = precision(1.0)
            two = precision(2.0)
            niters_eff = precision(max(niters, 1))
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
        else:
            gain = precision(1.0)

        gain = min(self.max_gain, max(self.min_gain, gain))
        if not self._deadband_disabled:
            if self.deadband_min <= gain <= self.deadband_max:
                gain = self.unity_gain
        if not accept:
            # A rejected step must shrink dt (mirrors device controllers).
            gain = min(gain, self.safety)
        return precision(gain)


__all__ = ["CPUAdaptiveController"]
