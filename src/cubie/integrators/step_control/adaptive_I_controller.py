"""Adaptive integral step-size controller.

Published Classes
-----------------
:class:`IStepControlConfig`
    Configuration for integral controllers.

    >>> from numpy import float64
    >>> config = IStepControlConfig(precision=float64)
    >>> float(config.kp)
    1.0

:class:`AdaptiveIController`
    Integral-only adaptive step-size controller.

    >>> from numpy import float64
    >>> ctrl = AdaptiveIController(precision=float64, n=4)
    >>> ctrl.is_adaptive
    True

See Also
--------
:class:`~cubie.integrators.step_control.adaptive_step_controller.BaseAdaptiveStepController`
    Abstract base class for adaptive controllers.
:class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
    Parent configuration class.
"""
from typing import Any, Callable

from cubie.cuda_simsafe import cuda, int32
from numpy import ndarray
from attrs import field, frozen
from math import isnan, isinf

from cubie._utils import PrecisionDType
from cubie.integrators.step_control.adaptive_step_controller import (
    AdaptiveStepControlConfig,
    BaseAdaptiveStepController,
    gain_converter,
)
from cubie.cuda_simsafe import selp
from cubie.result_codes import CUBIE_RESULT_CODES

from cubie.integrators.step_control.base_step_controller import ControllerCache


@frozen
class IStepControlConfig(AdaptiveStepControlConfig):
    """Configuration for integral adaptive controllers."""

    _kp: Any = field(default=1.0, converter=gain_converter)

    @property
    def kp(self) -> float:
        """Return the gain on the current error, resolved at the order."""
        return self._resolve_gain(self._kp)

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update({"kp": self._kp})
        return settings_dict


class AdaptiveIController(BaseAdaptiveStepController):
    """Integral step-size controller using only previous error."""

    _config_class = IStepControlConfig

    @property
    def kp(self) -> float:
        """Return the gain on the current error."""
        return self.compile_settings.kp

    def build_controller(
        self,
        precision: PrecisionDType,
        clamp: Callable,
        min_gain: float,
        max_gain: float,
        dt_min: float,
        dt_max: float,
        n: int,
        atol: ndarray,
        rtol: ndarray,
        algorithm_order: int,
        safety: float,
    ) -> ControllerCache:
        """Create the device function for the integral controller.

        Parameters
        ----------
        precision
            Precision callable used to coerce scalars on device.
        clamp
            Callable that clamps proposed step sizes.
        min_gain
            Minimum allowed gain when adapting the step size.
        max_gain
            Maximum allowed gain when adapting the step size.
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
            CUDA device function implementing the integral controller.
        """
        order_exponent = precision(self.kp / (2 * (1 + algorithm_order)))
        typed_one = precision(1.0)
        typed_zero = precision(0.0)
        deadband_min = precision(self.deadband_min)
        deadband_max = precision(self.deadband_max)
        safety = precision(safety)
        min_gain = precision(min_gain)
        max_gain = precision(max_gain)
        deadband_disabled = (
            (deadband_min == typed_one)
            and (deadband_max == typed_one)
        )
        n = int32(n)
        inv_n = precision(1.0 / n)
        typed_large = precision(1e16)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        step_too_small = int32(CUBIE_RESULT_CODES.STEP_TOO_SMALL)

        precision = self.compile_settings.numba_precision
        # step sizes and norms can be approximate - fastmath is fine
        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def controller_I(
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
            """Integral accept/step-size controller.

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
            nrm2 = typed_zero
            for i in range(n):
                error_i = max(abs(error[i]), precision(1e-16))
                tol = (
                    atol[i] + rtol[i] * max(abs(state[i]), abs(state_prev[i]))
                )
                ratio = error_i / tol
                nrm2 += ratio * ratio

            nrm2 = nrm2 * inv_n
            nrm2 = typed_large if (isnan(nrm2) or isinf(nrm2)) else nrm2

            accept = nrm2 <= typed_one
            accept_out[0] = int32(1) if accept else int32(0)

            gaintmp = precision(safety * nrm2 ** (-order_exponent))
            gain = clamp(gaintmp, min_gain, max_gain)
            if not deadband_disabled:
                within_deadband = (
                    (gain >= deadband_min)
                    and (gain <= deadband_max)
                )
                gain = selp(within_deadband, typed_one, gain)

            # Rejected steps retry with the undeadbanded gain.
            gain_reject = max(min_gain, gaintmp)
            gain = selp(accept, gain, gain_reject)

            # A truncated step's error norm carries no step-size
            # info: on accept, freeze dt and report success.
            freeze = accept and truncated
            dt_new_raw = dt[0] * gain
            dt[0] = selp(freeze, dt[0], clamp(dt_new_raw, dt_min, dt_max))

            ret = (
                success
                if (freeze or dt_new_raw > dt_min)
                else step_too_small
            )
            return ret

        # no cover: end
        return ControllerCache(device_function=controller_I)
