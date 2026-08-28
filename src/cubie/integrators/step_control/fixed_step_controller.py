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

from attrs import field, frozen
from cubie.cuda_simsafe import cuda, int32
from cubie.result_codes import CUBIE_RESULT_CODES

from cubie._utils import getype_validator
from cubie.integrators.step_control.base_step_controller import (
    BaseStepControllerConfig,
    BaseStepController,
    ControllerCache,
)


@frozen
class FixedStepControlConfig(BaseStepControllerConfig):
    """Configuration for fixed-step integrator loops.

    Attributes
    ----------
    precision
        Precision used for numerical operations.
    n
        Number of state variables controlled per step.
    atol
        Absolute tolerance vector inherited from the base config. The
        fixed controller never rejects steps, but implicit algorithms
        derive their inner-solver tolerances from it.
    rtol
        Relative tolerance vector, on the same terms as ``atol``.
    """

    _dt: float = field(default=1e-3, validator=getype_validator(float, 0))

    def __attrs_post_init__(self) -> None:
        """Validate configuration after initialisation."""
        super().__attrs_post_init__()
        self._validate_config()

    def _validate_config(self) -> None:
        """Confirm that the configuration is internally consistent."""

    @property
    def dt(self) -> float:
        """Return the fixed step size."""
        return self.precision(self._dt)

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

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update({"dt": self.dt})
        return settings_dict


class FixedStepController(BaseStepController):
    """Controller that enforces a constant time step."""

    _config_class = FixedStepControlConfig

    def _resolve_step_params(self, dt: float, kwargs: dict) -> None:
        """Collapse dt_min/dt_max to dt for fixed-step control.

        Parameters
        ----------
        dt
            Fixed step size, or None if not provided.
        kwargs
            Mutable dict of keyword arguments. Modified in place.
        """
        dt_min = kwargs.pop("dt_min", None)
        dt_max = kwargs.pop("dt_max", None)

        resolved = dt or dt_min or dt_max
        if resolved is not None:
            self._user_step_params["dt"] = resolved
            kwargs["dt"] = resolved

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
            inline=False,
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
        return ControllerCache(device_function=controller_fixed_step)
