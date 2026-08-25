"""Shared infrastructure for adaptive step-size controllers.

Published Classes
-----------------
:class:`AdaptiveStepControlConfig`
    Attrs configuration shared by all adaptive controllers.

    >>> from numpy import float64
    >>> config = AdaptiveStepControlConfig(precision=float64)
    >>> config.is_adaptive
    True

:class:`BaseAdaptiveStepController`
    Abstract factory base for adaptive controllers.

See Also
--------
:class:`~cubie.integrators.step_control.base_step_controller.BaseStepController`
    Abstract base class for all controllers.
:class:`~cubie.integrators.step_control.base_step_controller.BaseStepControllerConfig`
    Base configuration class.
"""

from abc import abstractmethod
from inspect import getsource
from typing import Callable, Optional

from numpy import sqrt
from attrs import field, frozen

from cubie._utils import (
    PrecisionDType,
    clamp_factory,
    getype_validator,
    inrangetype_validator,
)
from cubie.cuda_simsafe import cuda, selp
from cubie.integrators.norms import TwoRefMaskedScaledNorm
from cubie.integrators.step_control.base_step_controller import (
    BaseStepController,
    BaseStepControllerConfig,
    ControllerCache,
)


class OrderDependentGain:
    """A controller gain as a pure callable of the algorithm order.

    Equality, hashing, and canonical identity key on the callable's
    source text.
    """

    __slots__ = ("fn", "source")

    def __init__(self, fn: Callable[[int], float]) -> None:
        self.fn = fn
        self.source = getsource(fn)

    def __call__(self, order: int) -> float:
        return float(self.fn(order))

    def __eq__(self, other) -> bool:
        if isinstance(other, OrderDependentGain):
            return self.source == other.source
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.source)

    def __repr__(self) -> str:
        return f"OrderDependentGain({self.fn!r})"

    def _cubie_canonical_(self) -> str:
        return self.source


def gain_converter(value):
    """Coerce a gain spec to a float or an order-dependent gain."""
    if isinstance(value, OrderDependentGain):
        return value
    if callable(value):
        return OrderDependentGain(value)
    return float(value)


@frozen
class AdaptiveStepControlConfig(BaseStepControllerConfig):
    """Configuration for adaptive step controllers.

    Notes
    -----
    Parameters influencing compilation should live here so that device
    functions are rebuilt when they change.
    """

    _dt_min: float = field(default=1e-6, validator=getype_validator(float, 0))
    _dt_max: Optional[float] = field(
        default=1.0, validator=getype_validator(float, 0)
    )
    algorithm_order: int = field(default=1, validator=getype_validator(int, 1))
    _min_gain: float = field(
        default=0.3,
        validator=inrangetype_validator(float, 0, 0.95),
    )
    _max_gain: float = field(
        default=2.0,
        validator=getype_validator(float, 1),
    )
    _safety: float = field(
        default=0.9,
        validator=inrangetype_validator(float, 0, 1),
    )
    _deadband_min: float = field(
        default=1.0,
        validator=inrangetype_validator(float, 0, 1.0),
    )
    _deadband_max: float = field(
        default=1.0,
        validator=getype_validator(float, 1.0),
    )

    def _resolve_gain(self, gain) -> float:
        """Return a gain as a precision float at the algorithm order."""
        if isinstance(gain, OrderDependentGain):
            gain = gain(self.algorithm_order)
        return self.precision(gain)

    @property
    def dt_min(self) -> float:
        """Return the minimum permissible step size."""
        return self.precision(self._dt_min)

    @property
    def dt_max(self) -> float:
        """Return the maximum permissible step size."""
        return self.precision(self._dt_max)

    @property
    def dt(self) -> float:
        """Return the initial step size.

        When the user has not provided an explicit dt, returns the
        geometric mean of dt_min and dt_max.
        """
        if self._dt is not None:
            return self.precision(self._dt)
        return self.precision(sqrt(self._dt_min * self._dt_max))

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` because the controller adapts step size."""
        return True

    @property
    def min_gain(self) -> float:
        """Return the minimum gain factor."""
        return self.precision(self._min_gain)

    @property
    def max_gain(self) -> float:
        """Return the maximum gain factor."""
        return self.precision(self._max_gain)

    @property
    def safety(self) -> float:
        """Return the safety scaling factor."""
        return self.precision(self._safety)

    @property
    def deadband_min(self) -> float:
        """Return the lower gain threshold for the unity deadband."""

        return self.precision(self._deadband_min)

    @property
    def deadband_max(self) -> float:
        """Return the upper gain threshold for the unity deadband."""

        return self.precision(self._deadband_max)

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the configuration as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update(
            {
                "dt_min": self.dt_min,
                "dt_max": self.dt_max,
                "algorithm_order": self.algorithm_order,
                "min_gain": self.min_gain,
                "max_gain": self.max_gain,
                "safety": self.safety,
                "deadband_min": self.deadband_min,
                "deadband_max": self.deadband_max,
                "dt": self.dt,
            }
        )
        return settings_dict


class BaseAdaptiveStepController(BaseStepController):
    """Base class for adaptive step-size controllers; owns ``norm``."""

    _config_class = AdaptiveStepControlConfig

    def __init__(
        self,
        precision: PrecisionDType,
        dt: float = None,
        n: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, dt=dt, n=n, **kwargs)
        config = self.compile_settings
        self.norm = TwoRefMaskedScaledNorm(
            precision=config.precision,
            solver_width=config.n,
            n=config.n,
            atol=config.atol,
            rtol=config.rtol,
            mass_flags=config.mass_flags,
            jit_flags=config.jit_flags,
        )

    def update(
        self,
        updates_dict: Optional[dict[str, object]] = None,
        silent: bool = False,
        **kwargs: object,
    ) -> set[str]:
        """Propagate updates to the owned norm and then the controller."""
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        updates_dict.update(kwargs)
        norm_updates = dict(updates_dict)
        if "n" in norm_updates:
            norm_updates["solver_width"] = norm_updates["n"]
        self.norm.update(norm_updates, silent=True)
        return super().update(updates_dict, silent=silent)

    def _resolve_step_params(self, dt: float, kwargs: dict) -> None:
        """Derive bounds from dt and track user-provided values.

        Parameters
        ----------
        dt
            Initial step size, or None if not provided.
        kwargs
            Mutable dict of keyword arguments. Modified in place.
        """
        # Track user-provided values BEFORE derivation
        if dt is not None:
            self._user_step_params["dt"] = dt
        if "dt_min" in kwargs:
            self._user_step_params["dt_min"] = kwargs["dt_min"]
        if "dt_max" in kwargs:
            self._user_step_params["dt_max"] = kwargs["dt_max"]

        # Derive missing values
        if dt is not None:
            kwargs.setdefault("dt_min", dt / 100)
            kwargs.setdefault("dt_max", dt * 100)
            kwargs["dt"] = dt
        else:
            # dt not provided; derive from bounds if both present
            dt_min = kwargs.get("dt_min")
            dt_max = kwargs.get("dt_max")
            if dt_min is not None and dt_max is not None:
                kwargs["dt"] = sqrt(dt_min * dt_max)

    def _ensure_sane_bounds(self) -> None:
        """Validate step bounds; fix only non-user-provided parameters.

        Raises
        ------
        ValueError
            If user-provided bounds are inverted (dt_max < dt_min) or if
            dt falls outside a user-provided bound.
        """
        dt = self.dt
        dt_min = self.dt_min
        dt_max = self.dt_max

        dt_min_user = self._user_step_params.get("dt_min") is not None
        dt_max_user = self._user_step_params.get("dt_max") is not None

        # Inverted bounds: error only if both user-provided
        if dt_max < dt_min and dt_min_user and dt_max_user:
            raise ValueError(
                f"dt_max ({dt_max}) < dt_min ({dt_min}). "
                f"Provide compatible bounds."
            )

        # dt outside user-provided bounds is an error
        if dt < dt_min and dt_min_user:
            raise ValueError(
                f"dt ({dt}) < dt_min ({dt_min}). "
                f"Provide a compatible dt or adjust dt_min."
            )
        if dt > dt_max and dt_max_user:
            raise ValueError(
                f"dt ({dt}) > dt_max ({dt_max}). "
                f"Provide a compatible dt or adjust dt_max."
            )

        # Auto-fix non-user-provided parameters
        fixes = {}
        if dt_max < dt_min and not dt_max_user:
            # Inverted bounds with auto-derived dt_max: fix dt_max
            fixes["dt_max"] = dt_min * 100
        if dt_max < dt_min and not dt_min_user:
            # Inverted bounds with auto-derived dt_min: fix dt_min
            fixes["dt_min"] = dt_max / 100
        if dt < dt_min and not dt_min_user:
            fixes["dt_min"] = dt / 100
        if dt > dt_max and not dt_max_user:
            fixes["dt_max"] = dt * 100

        if fixes:
            self.update_compile_settings(fixes, silent=True)

    def build(self) -> ControllerCache:
        """Construct the device function implementing the controller.

        Returns
        -------
        ControllerCache
            Cache containing the compiled adaptive controller device
            function.
        """
        return self.build_controller(
            precision=self.precision,
            clamp=clamp_factory(self.precision),
            min_gain=self.min_gain,
            max_gain=self.max_gain,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            algorithm_order=self.compile_settings.algorithm_order,
            safety=self.compile_settings.safety,
            error_norm=self.build_error_norm(),
        )

    def build_error_norm(self) -> Callable:
        """Wrap ``norm`` as ``error_norm``, clamped to [1e-16, 1e16]."""
        scaled_norm = self.norm.device_function
        numba_precision = self.compile_settings.numba_precision
        typed_large = numba_precision(1e16)
        typed_floor = numba_precision(1e-16)

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def error_norm(state, state_prev, error):
            """Return the mean squared scaled error norm."""
            nrm2 = scaled_norm(error, state, state_prev)
            nrm2 = selp(nrm2 <= typed_large, nrm2, typed_large)
            return selp(nrm2 >= typed_floor, nrm2, typed_floor)

        # no cover: end
        return error_norm

    @abstractmethod
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
        """Create the device function for the specific controller.

        Parameters
        ----------
        precision
            Precision callable used to coerce values.
        clamp
            Callable that limits step updates.
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
            Device function returning the mean squared scaled error.

        Returns
        -------
        ControllerCache
            Cache containing the compiled controller device function.
        """
        raise NotImplementedError

    @property
    def min_gain(self) -> float:
        """Return the minimum gain factor."""

        return self.compile_settings.min_gain

    @property
    def max_gain(self) -> float:
        """Return the maximum gain factor."""

        return self.compile_settings.max_gain

    @property
    def safety(self) -> float:
        """Return the safety scaling factor."""

        return self.compile_settings.safety

    @property
    def deadband_min(self) -> float:
        """Return the lower gain threshold for unity selection."""

        return self.compile_settings.deadband_min

    @property
    def deadband_max(self) -> float:
        """Return the upper gain threshold for unity selection."""

        return self.compile_settings.deadband_max

    @property
    def algorithm_order(self) -> int:
        """Return the integration algorithm order assumed by the controller."""

        return int(self.compile_settings.algorithm_order)
