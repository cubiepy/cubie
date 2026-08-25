"""Gustafsson predictive step-size controller.

Published Classes
-----------------
:class:`GustafssonStepControlConfig`
    Configuration for the Gustafsson controller, extending
    :class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
    with damping and Newton iteration parameters.

    >>> from numpy import float64
    >>> config = GustafssonStepControlConfig(precision=float64)
    >>> config.newton_target_iters
    5

:class:`GustafssonController`
    Adaptive controller using Gustafsson acceleration for implicit
    integrators.

    >>> from numpy import float64
    >>> ctrl = GustafssonController(precision=float64, n=4)
    >>> ctrl.is_adaptive
    True

See Also
--------
:class:`~cubie.integrators.step_control.adaptive_step_controller.BaseAdaptiveStepController`
    Abstract base class for adaptive controllers.
:class:`~cubie.integrators.step_control.adaptive_step_controller.AdaptiveStepControlConfig`
    Parent configuration class.
"""

from typing import Callable

from cubie.cuda_simsafe import cuda, int32
from attrs import field, frozen

from cubie.buffer_registry import buffer_registry
from cubie.integrators.step_control.adaptive_step_controller import (
    BaseAdaptiveStepController,
    AdaptiveStepControlConfig,
)
from cubie._utils import (
    PrecisionDType,
    getype_validator,
)
from cubie.cuda_simsafe import selp
from cubie.result_codes import CUBIE_RESULT_CODES
from cubie.integrators.step_control.base_step_controller import ControllerCache


@frozen
class GustafssonStepControlConfig(AdaptiveStepControlConfig):
    """Configuration for Gustafsson-like predictive controller.

    Notes
    -----
    Includes the reference Newton iteration count used by Gustafsson's
    work-sensitive damping for implicit integrators.
    """

    _newton_target_iters: int = field(
        default=5,
        validator=getype_validator(int, 0),
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def newton_target_iters(self) -> int:
        """Return the Newton-work reference used for gain damping."""
        return int(self._newton_target_iters)

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update({"newton_target_iters": self.newton_target_iters})
        return settings_dict


class GustafssonController(BaseAdaptiveStepController):
    """Adaptive controller using Gustafsson acceleration."""

    _config_class = GustafssonStepControlConfig

    @property
    def newton_target_iters(self) -> int:
        """Return the Newton-work reference used for gain damping."""

        return self.compile_settings.newton_target_iters

    _timestep_buffer_elements = 2  # previous dt and error norm

    def build_controller(
        self,
        precision: PrecisionDType,
        clamp: Callable,
        min_gain: float,
        max_gain: float,
        dt_min: float,
        dt_max: float,
        algorithm_order: int,
        safety: float,
        error_norm: Callable,
    ) -> ControllerCache:
        """Create the device function for the Gustafsson controller.

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
        algorithm_order
            Order of the integration algorithm.
        safety
            Safety factor used when scaling the step size.
        error_norm
            Device function returning the mean squared scaled error
            norm over the differential states.

        Returns
        -------
        Callable
            CUDA device function implementing the Gustafsson controller.
        """
        alloc_timestep_buffer = buffer_registry.get_allocator(
            "timestep_buffer", self
        )

        expo = precision(1.0 / (2 * (algorithm_order + 1)))
        safety = precision(safety)
        newton_target_iters = int(self.newton_target_iters)
        gain_numerator = precision((1 + 2 * newton_target_iters)) * safety
        typed_one = precision(1.0)
        deadband_min = precision(self.deadband_min)
        deadband_max = precision(self.deadband_max)
        min_gain = precision(min_gain)
        max_gain = precision(max_gain)
        deadband_disabled = (deadband_min == typed_one) and (
            deadband_max == typed_one
        )
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        step_too_small = int32(CUBIE_RESULT_CODES.STEP_TOO_SMALL)

        # step sizes and norms can be approximate - fastmath is fine
        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def controller_gustafsson(
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
            """Gustafsson accept/step controller.

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
            niters : int32
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

            current_dt = dt[0]
            dt_prev = max(timestep_buffer[0], precision(1e-16))
            err_prev = max(timestep_buffer[1], precision(1e-16))

            nrm2 = error_norm(state, state_prev, error)

            accept = nrm2 <= typed_one
            accept_out[0] = int32(1) if accept else int32(0)

            denom = precision(niters + 2 * newton_target_iters)
            tmp = gain_numerator / denom
            fac = safety if safety < tmp else tmp
            gain_basic = precision(fac * (nrm2 ** (-expo)))

            ratio = nrm2 * nrm2 / err_prev
            gain_gus = precision(
                safety * (dt[0] / dt_prev) * (ratio**-expo)
            )
            gain = gain_gus if gain_gus < gain_basic else gain_basic
            gain = (
                gain
                if (accept and dt_prev > precision(1e-16))
                else (gain_basic)
            )

            gain = clamp(gain, min_gain, max_gain)
            if not deadband_disabled:
                within_deadband = (gain >= deadband_min) and (
                    gain <= deadband_max
                )
                gain = selp(within_deadband, typed_one, gain)

            # Rejected steps retry with the basic gain alone.
            gain_reject = clamp(gain_basic, min_gain, max_gain)
            gain = selp(accept, gain, gain_reject)

            # A truncated step's error norm carries no step-size
            # info: on accept, freeze dt and report success. History
            # commits only after ordinary accepted steps, so rejected
            # or truncated attempts never overwrite it.
            freeze = accept and truncated
            commit_history = accept and not truncated
            dt_new_raw = current_dt * gain
            dt[0] = selp(freeze, current_dt, clamp(dt_new_raw, dt_min, dt_max))

            timestep_buffer[0] = selp(
                commit_history, current_dt, timestep_buffer[0]
            )
            timestep_buffer[1] = selp(
                commit_history, nrm2, timestep_buffer[1]
            )

            ret = (
                success
                if (freeze or dt_new_raw > dt_min)
                else step_too_small
            )
            return ret

        # no cover: end
        return ControllerCache(device_function=controller_gustafsson)
