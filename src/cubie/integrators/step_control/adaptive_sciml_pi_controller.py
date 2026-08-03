"""OrdinaryDiffEq.jl's proportional--integral step controller.

Published Classes
-----------------
:class:`SciMLPIStepControlConfig`
    Configuration for the SciML PI controller.

:class:`SciMLPIController`
    SciML-matching proportional--integral step-size controller.

See Also
--------
:class:`~cubie.integrators.step_control.adaptive_PI_controller.AdaptivePIController`
    Cubie's native PI controller.
"""

from typing import Callable, Optional

from cubie.cuda_simsafe import cuda, int32
from numpy import ndarray
from attrs import field, frozen
from math import isnan, isinf, sqrt

from cubie._utils import (
    PrecisionDType,
    opt_getype_validator,
)
from cubie.buffer_registry import buffer_registry
from cubie.integrators.step_control.adaptive_step_controller import (
    AdaptiveStepControlConfig,
    BaseAdaptiveStepController,
)
from cubie.cuda_simsafe import selp
from cubie.result_codes import CUBIE_RESULT_CODES
from cubie.integrators.step_control.base_step_controller import ControllerCache


@frozen
class SciMLPIStepControlConfig(AdaptiveStepControlConfig):
    """Configuration for the SciML proportional--integral controller.

    ``beta1``/``beta2`` default to ``7/(10*order)`` and
    ``2/(5*order)``; ``min_gain``/``max_gain``/``safety`` are
    OrdinaryDiffEq's ``qmin``/``qmax``/``gamma``.
    """

    _beta1: Optional[float] = field(
        default=None, validator=opt_getype_validator(float, 0)
    )
    _beta2: Optional[float] = field(
        default=None, validator=opt_getype_validator(float, 0)
    )
    _qoldinit: float = field(default=1.0e-4)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def beta1(self) -> float:
        """Return the proportional exponent."""
        if self._beta1 is not None:
            return self.precision(self._beta1)
        return self.precision(7.0 / (10.0 * self.algorithm_order))

    @property
    def beta2(self) -> float:
        """Return the integral exponent."""
        if self._beta2 is not None:
            return self.precision(self._beta2)
        return self.precision(2.0 / (5.0 * self.algorithm_order))

    @property
    def qoldinit(self) -> float:
        """Return the initial (and floor) value of the qold memory."""
        return self.precision(self._qoldinit)

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "qoldinit": self.qoldinit,
            }
        )
        return settings_dict


class SciMLPIController(BaseAdaptiveStepController):
    """SciML-matching proportional--integral step-size controller."""

    _config_class = SciMLPIStepControlConfig

    _timestep_buffer_elements = 1  # qold: previous accepted error norm

    def __init__(
        self,
        precision: PrecisionDType,
        dt: float = None,
        n: int = 1,
        **kwargs,
    ) -> None:
        """Initialise with OrdinaryDiffEq.jl's limiter defaults."""
        kwargs.setdefault("min_gain", 0.2)
        kwargs.setdefault("max_gain", 10.0)
        kwargs.setdefault("safety", 0.9)
        kwargs.setdefault("deadband_min", 1.0)
        kwargs.setdefault("deadband_max", 1.0)
        super().__init__(precision=precision, dt=dt, n=n, **kwargs)

    @property
    def beta1(self) -> float:
        """Return the proportional exponent."""
        return self.compile_settings.beta1

    @property
    def beta2(self) -> float:
        """Return the integral exponent."""
        return self.compile_settings.beta2

    @property
    def qoldinit(self) -> float:
        """Return the initial (and floor) value of the qold memory."""
        return self.compile_settings.qoldinit

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        return super().settings_dict

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
        """Create the device function for the SciML PI controller.

        Parameters
        ----------
        precision
            Precision callable used to coerce scalars on device.
        clamp
            Callable that clamps proposed step sizes.
        min_gain
            Minimum allowed gain (OrdinaryDiffEq's ``qmin``).
        max_gain
            Maximum allowed gain (OrdinaryDiffEq's ``qmax``).
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
            Safety factor (OrdinaryDiffEq's ``gamma``).

        Returns
        -------
        ControllerCache
            Cache holding the compiled controller device function.
        """
        alloc_timestep_buffer = buffer_registry.get_allocator(
            "timestep_buffer", self
        )

        beta1 = precision(self.beta1)
        beta2 = precision(self.beta2)
        qoldinit = precision(self.qoldinit)
        inv_qmin = precision(1.0 / min_gain)
        inv_qmax = precision(1.0 / max_gain)
        typed_one = precision(1.0)
        typed_zero = precision(0.0)
        safety = precision(safety)
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
        def controller_sciml_pi(
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
            """SciML PI accept/step-size controller.

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

            # Zero marks a fresh buffer; start at qoldinit.
            qold_stored = timestep_buffer[0]
            qold = selp(qold_stored > typed_zero, qold_stored, qoldinit)

            nrm2 = typed_zero
            for i in range(n):
                tol = atol[i] + rtol[i] * max(
                    abs(state[i]), abs(state_prev[i])
                )
                ratio = error[i] / tol
                nrm2 += ratio * ratio

            nrm2 = nrm2 * inv_n
            nrm2 = typed_large if (isnan(nrm2) or isinf(nrm2)) else nrm2
            eest = precision(sqrt(nrm2))

            accept = eest <= typed_one
            accept_out[0] = int32(1) if accept else int32(0)

            # q = EEst^beta1 / qold^beta2; zero error grows at qmax.
            zero_error = eest == typed_zero
            eest_safe = selp(zero_error, typed_one, eest)
            q11 = precision(eest_safe ** beta1)
            q_raw = q11 / precision(qold ** beta2)
            q_raw = selp(zero_error, inv_qmax, q_raw)

            # Accepted: q = max(1/qmax, min(1/qmin, q/gamma)).
            q_accept = max(inv_qmax, min(inv_qmin, q_raw / safety))
            # Rejected: dt /= min(1/qmin, q11/gamma).
            q_reject = min(inv_qmin, q11 / safety)

            q = selp(accept, q_accept, q_reject)
            dt_new_raw = dt[0] / q

            # Accepted truncated steps freeze dt and commit nothing.
            freeze = accept and truncated
            commit_history = accept and not truncated
            dt[0] = selp(freeze, dt[0], clamp(dt_new_raw, dt_min, dt_max))
            timestep_buffer[0] = selp(
                commit_history, max(eest, qoldinit), qold_stored
            )

            ret = (
                success
                if (freeze or dt_new_raw > dt_min)
                else step_too_small
            )
            return ret

        # no cover: end
        return ControllerCache(device_function=controller_sciml_pi)
