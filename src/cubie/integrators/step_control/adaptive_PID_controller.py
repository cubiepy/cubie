"""Adaptive proportional--integral--derivative step-size controller.

Published Classes
-----------------
:class:`PIDStepControlConfig`
    Configuration for PID controllers, extending
    :class:`~cubie.integrators.step_control.adaptive_PI_controller.PIStepControlConfig`
    with a derivative gain.

    >>> from numpy import float64
    >>> config = PIDStepControlConfig(precision=float64, derivative_gain=0.05)
    >>> config.derivative_gain
    0.05

:class:`AdaptivePIDController`
    Proportional--integral--derivative step-size controller.

    >>> from numpy import float64
    >>> ctrl = AdaptivePIDController(
    ...     precision=float64, n=4, derivative_gain=0.05
    ... )
    >>> ctrl.is_adaptive
    True

See Also
--------
:class:`~cubie.integrators.step_control.adaptive_PI_controller.AdaptivePIController`
    PI controller without derivative term.
:class:`~cubie.integrators.step_control.adaptive_PI_controller.PIStepControlConfig`
    Parent configuration class.
"""

from typing import Any, Callable

from numpy import ndarray
from cubie.cuda_simsafe import cuda, int32
from attrs import field, frozen
from math import isnan, isinf
from cubie._utils import PrecisionDType
from cubie.buffer_registry import buffer_registry
from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController,
    gain_converter,
)
from cubie.integrators.step_control.adaptive_PI_controller import (
    PIStepControlConfig,
)
from cubie.cuda_simsafe import selp
from cubie.result_codes import CUBIE_RESULT_CODES
from cubie.integrators.step_control.base_step_controller import ControllerCache


@frozen
class PIDStepControlConfig(PIStepControlConfig):
    """Configuration for a proportional–integral–derivative controller."""

    _derivative_gain: Any = field(default=0.0, converter=gain_converter)

    @property
    def derivative_gain(self) -> float:
        """Return the derivative gain; divided by order+1 at build."""
        return self._resolve_gain(self._derivative_gain)


class AdaptivePIDController(BaseAdaptiveStepController):
    """Adaptive PID step size controller."""

    _config_class = PIDStepControlConfig
    _gain_parameters = (
        "integral_gain",
        "proportional_gain",
        "derivative_gain",
    )

    @property
    def integral_gain(self) -> float:
        """Return the integral gain."""
        return self.compile_settings.integral_gain

    @property
    def proportional_gain(self) -> float:
        """Return the proportional gain."""
        return self.compile_settings.proportional_gain

    @property
    def derivative_gain(self) -> float:
        """Return the derivative gain."""
        return self.compile_settings.derivative_gain

    _timestep_buffer_elements = 2  # previous two error norms

    def build_controller(
        self,
        precision: PrecisionDType,
        clamp: Callable,
        min_step_shrink: float,
        max_step_growth: float,
        dt_min: float,
        dt_max: float,
        n: int,
        atol: ndarray,
        rtol: ndarray,
        algorithm_order: int,
        safety: float,
    ) -> ControllerCache:
        """Create the device function for the PID controller.

        Parameters
        ----------
        precision
            Precision callable used to coerce scalars on device.
        clamp
            Callable that clamps proposed step sizes.
        min_step_shrink
            Most the step may shrink per adjustment.
        max_step_growth
            Most the step may grow per adjustment.
        dt_min
            Minimum permissible step size.
        dt_max
            Maximum permissible step size.
        n
            Number of state variables controlled per step.
        atol
            Absolute tolerance vector.
        rtol
            Relative tolerance vector.
        algorithm_order
            Order of the integration algorithm.
        safety
            Safety factor used when scaling the step size.

        Returns
        -------
        Callable
            CUDA device function implementing the PID controller.
        """
        alloc_timestep_buffer = buffer_registry.get_allocator(
            "timestep_buffer", self
        )

        integral_gain = self.integral_gain
        proportional_gain = self.proportional_gain
        derivative_gain = self.derivative_gain
        # PETSc gain map: beta1=kI+kP+kD, beta2=-(kP+2kD), beta3=kD.
        beta1 = integral_gain + proportional_gain + derivative_gain
        beta2 = -(proportional_gain + 2 * derivative_gain)
        beta3 = derivative_gain
        expo1 = precision(beta1 / (2 * (algorithm_order + 1)))
        expo2 = precision(beta2 / (2 * (algorithm_order + 1)))
        expo3 = precision(beta3 / (2 * (algorithm_order + 1)))
        safety = precision(safety)
        typed_one = precision(1.0)
        typed_zero = precision(0.0)
        min_step_shrink = precision(min_step_shrink)
        max_step_growth = precision(max_step_growth)
        dt_min = precision(dt_min)
        dt_max = precision(dt_max)
        deadband_min = precision(self.deadband_min)
        deadband_max = precision(self.deadband_max)
        deadband_disabled = (deadband_min == typed_one) and (
            deadband_max == typed_one
        )
        precision = self.compile_settings.numba_precision
        n = int32(n)
        inv_n = precision(1.0 / n)
        typed_large = precision(1e16)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        step_too_small = int32(CUBIE_RESULT_CODES.STEP_TOO_SMALL)
        # step sizes and norms can be approximate - fastmath is fine
        # no cover: start

        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def controller_PID(
            dt,
            state,
            state_prev,
            error,
            niters,
            truncated,
            accept_out,
            shared_scratch,
            persistent_local,
        ):  # pragma: no cover - CUDA
            """Proportional–integral–derivative accept/step controller.

            Parameters
            ----------
            dt : device array
                Current integration step size.
            state : device array
                Current state vector.
            state_prev : device array
                Previous state vector.
            error : device array
                Estimated local error vector.
            niters : device array
                Iteration counters from the integrator loop.
            truncated : bool
                True when the loop forced the step onto an output
                boundary.
            accept_out : device array
                Output flag indicating acceptance of the step.
            shared_scratch : device array
                Shared memory scratch space.
            persistent_local : device array
                Persistent local memory for controller state.

            Returns
            -------
            int32
                Non-zero when the step is rejected at the minimum size.
            """
            timestep_buffer = alloc_timestep_buffer(
                shared_scratch, persistent_local
            )

            err_prev = timestep_buffer[0]
            err_prev_prev = timestep_buffer[1]
            nrm2 = typed_zero

            for i in range(n):
                error_i = max(abs(error[i]), precision(1e-16))
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                ratio = error_i / tol
                nrm2 += ratio * ratio

            nrm2 = nrm2 * inv_n
            nrm2 = typed_large if (isnan(nrm2) or isinf(nrm2)) else nrm2

            accept = nrm2 <= typed_one
            accept_out[0] = int32(1) if accept else int32(0)
            err_prev_safe = err_prev if err_prev > typed_zero else nrm2
            err_prev_prev_safe = (
                err_prev_prev if err_prev_prev > typed_zero else err_prev_safe
            )

            gain_current = precision(nrm2 ** (-expo1))
            gain_new = precision(
                safety
                * gain_current
                * (err_prev_safe ** (-expo2))
                * (err_prev_prev_safe ** (-expo3))
            )
            gain = clamp(gain_new, min_step_shrink, max_step_growth)
            if not deadband_disabled:
                within_deadband = (gain >= deadband_min) and (
                    gain <= deadband_max
                )
                gain = selp(within_deadband, typed_one, gain)

            # Rejected steps retry on the current error alone.
            gain_reject = max(min_step_shrink, safety * gain_current)
            gain = selp(accept, gain, gain_reject)

            # A truncated step's error norm carries no step-size
            # info: on accept, freeze dt and report success. History
            # commits only after ordinary accepted steps, so rejected
            # or truncated attempts never overwrite it.
            freeze = accept and truncated
            commit_history = accept and not truncated
            dt_new_raw = dt[0] * gain
            dt[0] = selp(freeze, dt[0], clamp(dt_new_raw, dt_min, dt_max))
            timestep_buffer[1] = selp(commit_history, err_prev, err_prev_prev)
            timestep_buffer[0] = selp(commit_history, nrm2, err_prev)

            ret = (
                success
                if (freeze or dt_new_raw > dt_min)
                else step_too_small
            )
            return ret

        # no cover: end
        return ControllerCache(device_function=controller_PID)
