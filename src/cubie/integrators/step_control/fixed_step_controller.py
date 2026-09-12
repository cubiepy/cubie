"""Fixed step-size controller.

Published Classes
-----------------
:class:`FixedStepControlConfig`
    Configuration container for fixed-step controllers.

    >>> from numpy import float32
    >>> config = FixedStepControlConfig(precision=float32, dt=1e-3)
    >>> config.dt
    0.001

:class:`FixedStepController`
    Controller that enforces a constant time step.

    >>> from numpy import float64
    >>> ctrl = FixedStepController(precision=float64, dt=0.01)
    >>> ctrl.is_adaptive
    False

See Also
--------
:class:`~cubie.integrators.step_control.base_step_controller.BaseStepController`
    Abstract base class for all controllers.
:class:`~cubie.integrators.step_control.base_step_controller.BaseStepControllerConfig`
    Base configuration class.
"""

from typing import Optional

from attrs import field, frozen
from numpy import sqrt
from cubie.cuda_simsafe import cuda, int32
from cubie.result_codes import CUBIE_RESULT_CODES

from cubie._utils import opt_getype_validator
from cubie.integrators.step_control.base_step_controller import (
    BaseStepControllerConfig,
    BaseStepController,
    ControllerCache,
)

DEFAULT_FIXED_DT = 1e-3
"""Fixed step when no ``dt`` or bound is given."""


@frozen
class FixedStepControlConfig(BaseStepControllerConfig):
    """Configuration for fixed-step integrator loops.

    Attributes
    ----------
    precision
        Precision used for numerical operations.
    n_states
        Number of state variables controlled per step.
    atol
        Absolute tolerance vector inherited from the base config. The
        fixed controller never rejects steps, but implicit algorithms
        derive their inner-solver tolerances from it.
    rtol
        Relative tolerance vector, on the same terms as ``atol``.
    """

    _dt_min: Optional[float] = field(
        default=None, validator=opt_getype_validator(float, 0)
    )
    _dt_max: Optional[float] = field(
        default=None, validator=opt_getype_validator(float, 0)
    )

    @property
    def dt(self) -> float:
        """Given, else the bounds' geometric mean, else a bound, else 1e-3."""
        if self._dt is not None:
            return self.precision(self._dt)
        if self._dt_min is not None and self._dt_max is not None:
            return self.precision(sqrt(self._dt_min * self._dt_max))
        for value in (self._dt_min, self._dt_max):
            if value is not None:
                return self.precision(value)
        return self.precision(DEFAULT_FIXED_DT)

    @property
    def dt_min(self) -> float:
        """Return the minimum time step size."""
        return self.dt

    @property
    def dt_max(self) -> float:
        """Return the maximum step size."""
        return self.dt

    @property
    def is_adaptive(self) -> bool:
        """Return ``False`` because the controller is not adaptive."""
        return False



class FixedStepController(BaseStepController):
    """Controller that enforces a constant time step."""

    _config_class = FixedStepControlConfig

    def build(self) -> ControllerCache:
        """Return a device function that always accepts with fixed step.

        Returns
        -------
        ControllerCache
            Cache containing the compiled fixed-step device function.
        """
        success = int32(CUBIE_RESULT_CODES.SUCCESS)

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def controller_fixed_step(
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
            """Fixed-step controller device function.

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
                boundary. Unused.
            accept_out : device array
                Output flag indicating acceptance of the step.
            shared_scratch : device array
                Shared memory scratch space.
            persistent_local : device array
                Persistent local memory for controller state.

            Returns
            -------
            int32
                Zero, indicating that the current step size should be kept.
            """
            accept_out[0] = int32(1)
            return success

        # no cover: end
        return ControllerCache(step_controller_fn=controller_fixed_step)
