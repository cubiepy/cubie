"""Adaptive proportional--integral step-size controller.

Published Classes
-----------------
:class:`PIStepControlConfig`
    Configuration for proportional--integral controllers.

    >>> from numpy import float64
    >>> config = PIStepControlConfig(precision=float64)
    >>> float(config.proportional_gain)
    0.4

:class:`AdaptivePIController`
    Proportional--integral step-size controller.

    >>> from numpy import float64
    >>> ctrl = AdaptivePIController(precision=float64, n=4)
    >>> ctrl.is_adaptive
    True

See Also
--------
:class:`~cubie.integrators.step_control.adaptive_step_controller.BaseAdaptiveStepController`
    Abstract base class for adaptive controllers.
:class:`~cubie.integrators.step_control.adaptive_I_controller.IStepControlConfig`
    Parent configuration class supplying ``integral_gain``.
"""

from typing import Any, Callable

from cubie.cuda_simsafe import cuda, int32
from attrs import field, frozen

from cubie._utils import PrecisionDType
from cubie.buffer_registry import buffer_registry
from cubie.integrators.step_control.adaptive_I_controller import (
    IStepControlConfig,
)
from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController,
    gain_converter,
)
from cubie.cuda_simsafe import selp
from cubie.result_codes import CUBIE_RESULT_CODES
from cubie.integrators.step_control.base_step_controller import ControllerCache


@frozen
class PIStepControlConfig(IStepControlConfig):
    """Configuration for proportional–integral adaptive controllers."""

    _integral_gain: Any = field(default=0.3, converter=gain_converter)
    _proportional_gain: Any = field(default=0.4, converter=gain_converter)

    @property
    def proportional_gain(self) -> float:
        """Return the proportional gain; divided by order+1 at build."""
        return self._resolve_gain(self._proportional_gain)


class AdaptivePIController(BaseAdaptiveStepController):
    """Proportional–integral step-size controller."""

    _config_class = PIStepControlConfig

    @property
    def integral_gain(self) -> float:
        """Return the integral gain."""
        return self.compile_settings.integral_gain

    @property
    def proportional_gain(self) -> float:
        """Return the proportional gain."""
        return self.compile_settings.proportional_gain

    _timestep_buffer_elements = 1  # previous error norm

    def build_controller(
        self,
        precision: PrecisionDType,
        clamp: Callable,
        min_step_shrink: float,
        max_step_growth: float,
        dt_min: float,
        dt_max: float,
        algorithm_order: int,
        safety: float,
        error_norm: Callable,
    ) -> ControllerCache:
        """Create the device function for the PI controller.

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
        algorithm_order
            Order of the integration algorithm.
        safety
            Safety factor used when scaling the step size.
        error_norm
            Device function returning the mean squared scaled error.

        Returns
        -------
        Callable
            CUDA device function implementing the PI controller.
        """
        alloc_timestep_buffer = buffer_registry.get_allocator(
            "timestep_buffer", self
        )

        beta1 = self.integral_gain + self.proportional_gain
        beta2 = -self.proportional_gain
        expo1 = precision(beta1 / ((algorithm_order + 1) * 2))
        expo2 = precision(beta2 / ((algorithm_order + 1) * 2))
        typed_one = precision(1.0)
        typed_zero = precision(0.0)
        safety = precision(safety)
        min_step_shrink = precision(min_step_shrink)
        max_step_growth = precision(max_step_growth)
        deadband_min = precision(self.deadband_min)
        deadband_max = precision(self.deadband_max)
        deadband_disabled = (deadband_min == typed_one) and (
            deadband_max == typed_one
        )
        precision = self.compile_settings.numba_precision
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        step_too_small = int32(CUBIE_RESULT_CODES.STEP_TOO_SMALL)

        # step sizes and norms can be approximate - fastmath is fine
        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def controller_PI(
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
            """Proportional–integral accept/step-size controller.

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
            nrm2 = error_norm(error, state, state_prev)
            accept = nrm2 <= typed_one
            accept_out[0] = int32(1) if accept else int32(0)

            gain_current = precision(nrm2 ** (-expo1))
            # Handle uninitialized err_prev by using current error as fallback
            err_source = err_prev if err_prev > typed_zero else nrm2
            gain_history = precision(err_source ** (-expo2))
            gain_new = safety * gain_current * gain_history
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
            timestep_buffer[0] = selp(commit_history, nrm2, err_prev)

            ret = (
                success
                if (freeze or dt_new_raw > dt_min)
                else step_too_small
            )
            return ret

        # no cover: end
        return ControllerCache(device_function=controller_PI)
