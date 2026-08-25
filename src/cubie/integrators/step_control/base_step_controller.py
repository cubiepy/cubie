"""Abstract interfaces for step-size controller configuration and
factories.

Published Classes
-----------------
:class:`ControllerCache`
    Cache container for compiled controller device functions.

:class:`BaseStepControllerConfig`
    Abstract attrs configuration shared by all controllers.

:class:`BaseStepController`
    Abstract factory base compiling CUDA step-size controllers.

Constants
---------
:data:`ALL_STEP_CONTROLLER_PARAMETERS`
    Union of all keyword arguments accepted across controller types.

Notes
-----
Concrete controllers extend these classes to compile CUDA device
functions that implement specific control strategies. Fixed and
adaptive controllers share the configuration and buffer registration
interfaces defined here.

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactory`
    Parent factory providing compilation and cache management.
:mod:`cubie.integrators.step_control`
    Package-level entry point and controller registry.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
import warnings

from attrs import (
    Converter,
    cmp_using,
    define,
    field,
    fields_dict,
    validators,
    frozen,
)
from numpy import array_equal, asarray, ndarray

from cubie.CUDAFactory import (
    CUDAFactory,
    CUDAFactoryConfig,
    CUDADispatcherCache,
)
from cubie._utils import (
    getype_validator,
    nonnegative_float_array_validator,
    opt_getype_validator,
    build_config,
    PrecisionDType,
    tol_converter,
)
from cubie.buffer_registry import buffer_registry

ALL_STEP_CONTROLLER_PARAMETERS = {
    "precision",
    "n",
    "step_controller",
    "dt",
    "dt_min",
    "dt_max",
    "atol",
    "rtol",
    "algorithm_order",
    "min_step_shrink",
    "max_step_growth",
    "safety",
    "integral_gain",
    "proportional_gain",
    "derivative_gain",
    "filter_coefficients",
    "deadband_min",
    "deadband_max",
    "newton_target_iters",
    "timestep_memory_location",
}
"""All keyword arguments accepted by step controllers.

These parameters can be passed as keyword arguments to any step
controller constructor or to :func:`get_controller`. The set is used
by parent components to filter kwargs before forwarding them.

.. list-table:: Parameter Summary
   :header-rows: 1

   * - Parameter
     - Accepted By
     - Description
   * - ``precision``
     - :class:`BaseStepControllerConfig`
     - Floating-point dtype for controller computations.
   * - ``n``
     - :class:`BaseStepControllerConfig`
     - Number of state variables controlled per step.
   * - ``step_controller``
     - :func:`~cubie.integrators.step_control.get_controller`
     - Controller type string (``'fixed'``, ``'i'``, ``'pi'``,
       ``'pid'``, ``'gustafsson'``).
   * - ``dt``
     - :class:`~.fixed_step_controller.FixedStepControlConfig`
     - Fixed step size.
   * - ``dt_min``
     - :class:`~.adaptive_step_controller.AdaptiveStepControlConfig`
     - Minimum permissible step size.
   * - ``dt_max``
     - :class:`~.adaptive_step_controller.AdaptiveStepControlConfig`
     - Maximum permissible step size.
   * - ``atol``
     - :class:`BaseStepControllerConfig`
     - Absolute tolerance vector.
   * - ``rtol``
     - :class:`BaseStepControllerConfig`
     - Relative tolerance vector.
   * - ``algorithm_order``
     - :class:`~.adaptive_step_controller.AdaptiveStepControlConfig`
     - Order of the integration algorithm.
   * - ``min_step_shrink``
     - :class:`~.adaptive_step_controller.AdaptiveStepControlConfig`
     - Most the step may shrink per adjustment.
   * - ``max_step_growth``
     - :class:`~.adaptive_step_controller.AdaptiveStepControlConfig`
     - Most the step may grow per adjustment.
   * - ``safety``
     - :class:`~.adaptive_step_controller.AdaptiveStepControlConfig`
     - Safety scaling factor for step-size proposals.
   * - ``integral_gain``
     - :class:`~.adaptive_I_controller.IStepControlConfig`
     - Integral gain on the log-error control signal.
   * - ``proportional_gain``
     - :class:`~.adaptive_PI_controller.PIStepControlConfig`
     - Proportional gain on the log-error control signal.
   * - ``derivative_gain``
     - :class:`~.adaptive_PID_controller.PIDStepControlConfig`
     - Derivative gain on the log-error control signal.
   * - ``filter_coefficients``
     - :class:`BaseStepController`
     - Filter exponents ``(beta1, beta2, beta3)`` or a preset name,
       translated to gains via
       :func:`filter_coefficients_to_gains`.
   * - ``deadband_min``
     - :class:`~.adaptive_step_controller.AdaptiveStepControlConfig`
     - Lower gain threshold for the unity deadband.
   * - ``deadband_max``
     - :class:`~.adaptive_step_controller.AdaptiveStepControlConfig`
     - Upper gain threshold for the unity deadband.
   * - ``newton_target_iters``
     - :class:`~.gustafsson_controller.GustafssonStepControlConfig`
     - Reference Newton-work count used to damp step-size proposals.
   * - ``timestep_memory_location``
     - :class:`BaseStepControllerConfig`
     - Memory location for the timestep buffer (``'local'`` or
       ``'shared'``).
"""

CONTROLLER_GAIN_NAMES = (
    "integral_gain",
    "proportional_gain",
    "derivative_gain",
)
"""PID gain keys, in filter order."""

CONTROLLER_GAIN_PARAMETERS = frozenset(
    {*CONTROLLER_GAIN_NAMES, "filter_coefficients"}
)
"""Gain keys excluded from ``settings_dict`` swap carryover."""

GAIN_CONTROLLER_CHAIN = ("i", "pi", "pid")
"""Gain-carrying controllers, each a superset of the one before."""

FILTER_COEFFICIENT_PRESETS = {
    "basic": (1.0, 0.0, 0.0),
    "pi42": (0.6, -0.2, 0.0),
    "pi33": (2.0 / 3.0, -1.0 / 3.0, 0.0),
    "pi34": (0.7, -0.4, 0.0),
    "h211pi": (1.0 / 6.0, 1.0 / 6.0, 0.0),
    "h312pid": (1.0 / 18.0, 1.0 / 9.0, 1.0 / 18.0),
}
"""Named ``(beta1, beta2, beta3)`` triples; matched case-insensitively."""


def filter_coefficients_to_gains(value) -> dict[str, float]:
    """Translate filter exponents into PID gain settings.

    Parameters
    ----------
    value
        A preset name or ``(beta1, beta2, beta3)`` sequence.

    Returns
    -------
    dict[str, float]
        Gain settings reproducing the requested filter.

    Raises
    ------
    ValueError
        If the preset name or sequence shape is invalid.
    """
    if isinstance(value, str):
        try:
            value = FILTER_COEFFICIENT_PRESETS[value.lower()]
        except KeyError:
            known = ", ".join(sorted(FILTER_COEFFICIENT_PRESETS))
            raise ValueError(
                f"Unknown filter_coefficients preset {value!r}; "
                f"known presets: {known}."
            ) from None
    try:
        beta1, beta2, beta3 = (float(entry) for entry in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "filter_coefficients must be a preset name or a sequence "
            f"of three numbers (beta1, beta2, beta3); got {value!r}."
        ) from exc
    derivative_gain = beta3
    proportional_gain = -beta2 - 2.0 * beta3
    integral_gain = beta1 + beta2 + beta3
    return {
        "integral_gain": integral_gain,
        "proportional_gain": proportional_gain,
        "derivative_gain": derivative_gain,
    }


def minimal_gain_controller(settings) -> Optional[str]:
    """Return the smallest ``i``/``pi``/``pid`` carrying the given gains.

    Parameters
    ----------
    settings
        Mapping of gain keys or ``filter_coefficients``.

    Returns
    -------
    str or None
        ``None`` when no gain is given.
    """
    gains = {}
    filter_value = settings.get("filter_coefficients")
    if filter_value is not None:
        gains.update(filter_coefficients_to_gains(filter_value))
    for name in CONTROLLER_GAIN_NAMES:
        value = settings.get(name)
        if value is not None:
            gains[name] = value
    if not gains:
        return None

    def nonzero(value) -> bool:
        return callable(value) or float(value) != 0.0

    if nonzero(gains.get("derivative_gain", 0.0)):
        return "pid"
    if nonzero(gains.get("proportional_gain", 0.0)):
        return "pi"
    return "i"


def gain_controller_carries(name: str, needed: str) -> bool:
    """Return whether controller ``name`` carries ``needed``'s gains."""
    if name not in GAIN_CONTROLLER_CHAIN:
        return False
    return GAIN_CONTROLLER_CHAIN.index(name) >= GAIN_CONTROLLER_CHAIN.index(
        needed
    )


@define
class ControllerCache(CUDADispatcherCache):
    """Cache container for compiled step-controller device functions.

    Attributes
    ----------
    device_function
        Compiled CUDA device function, or ``-1`` before compilation.
    """

    device_function: Union[Callable, int] = field(default=-1)


@frozen
class BaseStepControllerConfig(CUDAFactoryConfig, ABC):
    """Configuration interface for step-size controllers.

    Attributes
    ----------
    precision
        Precision used for controller calculations.
    n
        Number of state variables controlled per step.
    atol
        Absolute tolerance vector. Adaptive controllers scale their
        error norms with it; every controller carries it so implicit
        algorithms can derive inner-solver tolerances from it.
    rtol
        Relative tolerance vector, carried on the same terms as
        ``atol``.
    """

    n: int = field(default=1, validator=getype_validator(int, 0))
    _dt: Optional[float] = field(
        default=None, validator=opt_getype_validator(float, 0)
    )
    timestep_memory_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    atol: ndarray = field(
        default=asarray([1e-6]),
        validator=nonnegative_float_array_validator,
        converter=Converter(tol_converter, takes_self=True),
        eq=cmp_using(eq=array_equal),
    )
    rtol: ndarray = field(
        default=asarray([1e-6]),
        validator=nonnegative_float_array_validator,
        converter=Converter(tol_converter, takes_self=True),
        eq=cmp_using(eq=array_equal),
    )

    @property
    def tol_length(self) -> int:
        """Return the tolerance-array length for tol_converter."""
        return self.n

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    @abstractmethod
    def dt_min(self) -> float:
        """Return the minimum supported step size."""

    @property
    @abstractmethod
    def dt_max(self) -> float:
        """Return the maximum supported step size."""

    @property
    @abstractmethod
    def dt(self) -> float:
        """Return the initial step size used when integration starts."""

    @property
    @abstractmethod
    def is_adaptive(self) -> bool:
        """Return ``True`` when the controller adapts its step size."""

    @property
    @abstractmethod
    def settings_dict(self) -> dict[str, object]:
        """Return a dictionary of configuration settings."""

        return {
            "n": self.n,
            "atol": self.atol,
            "rtol": self.rtol,
        }


class BaseStepController(CUDAFactory):
    """Factory interface for compiling CUDA step-size controllers."""

    _config_class = None  # Subclasses must override
    _timestep_buffer_elements = 0  # History slots; overridden per controller

    def __init__(
        self,
        precision: PrecisionDType,
        dt: float = None,
        n: int = 1,
        **kwargs,
    ) -> None:
        """Initialise the step controller.

        Parameters
        ----------
        precision
            Precision used for controller calculations.
        dt
            Step size or initial step size.
        n
            Number of state variables.
        **kwargs
            Additional parameters passed to the config class.
        """
        super().__init__()
        self._user_step_params = {}
        self._resolve_step_params(dt, kwargs)
        self._apply_filter_coefficients(kwargs)
        config = build_config(
            self._config_class,
            required={"precision": precision, "n": n},
            **kwargs,
        )
        self.setup_compile_settings(config)
        self._ensure_sane_bounds()
        self.register_buffers()

    def _resolve_step_params(self, dt: float, kwargs: dict) -> None:
        """Resolve step parameters and track user-provided values.

        Subclasses override to implement controller-specific translation
        and set entries in ``self._user_step_params`` for user-provided
        values.

        Parameters
        ----------
        dt
            Step size, or None if not provided.
        kwargs
            Mutable dict of keyword arguments. Modified in place.
        """
        pass

    def _ensure_sane_bounds(self) -> None:
        """Ensure step bounds satisfy constraints.

        Called during __init__ and after update(). Subclasses override
        to validate bounds and fix constraint violations on
        non-user-provided parameters.
        """
        pass

    @property
    def gain_names(self) -> tuple[str, ...]:
        """Return the PID gain keys this controller's config carries."""
        config_fields = fields_dict(self._config_class)
        return tuple(
            name for name in CONTROLLER_GAIN_NAMES
            if f"_{name}" in config_fields
        )

    def _apply_filter_coefficients(self, params: dict) -> set[str]:
        """Translate ``filter_coefficients`` in ``params`` into gains.

        Parameters
        ----------
        params
            Mutable settings mapping; modified in place.

        Returns
        -------
        set[str]
            The consumed keys.

        Raises
        ------
        ValueError
            On mixed gain and filter input, or an unsupported gain.
        """
        gain_names = self.gain_names
        if "filter_coefficients" not in params or not gain_names:
            return set()
        value = params.pop("filter_coefficients")
        if value is None:
            return {"filter_coefficients"}
        gains = filter_coefficients_to_gains(value)
        conflicts = set(gains) & set(params)
        if conflicts:
            conflict_str = ", ".join(sorted(conflicts))
            raise ValueError(
                "filter_coefficients cannot be combined with explicit "
                f"gain settings ({conflict_str}); provide one or the "
                "other."
            )
        for key, gain in gains.items():
            if key in gain_names:
                params[key] = gain
            elif gain != 0.0:
                raise ValueError(
                    f"filter_coefficients {value!r} requires a nonzero "
                    f"{key}, which {type(self).__name__} does not "
                    "support."
                )
        return {"filter_coefficients"}

    def register_buffers(self) -> None:
        """Register controller buffers with the central buffer registry.

        Registers the ``timestep_buffer`` at ``_timestep_buffer_elements``
        history slots in the location given by
        ``compile_settings.timestep_memory_location``. Controllers that keep
        no history (``_timestep_buffer_elements == 0``) register nothing, so
        the registry owns the size like every other buffer-registered class.
        """
        size = self._timestep_buffer_elements
        if size == 0:
            return

        config = self.compile_settings
        buffer_registry.register(
            "timestep_buffer",
            self,
            size,
            config.timestep_memory_location,
            persistent=True,
        )

    @abstractmethod
    def build(self) -> ControllerCache:
        """Compile and return the CUDA device controller.

        Returns
        -------
        ControllerCache
            Cache containing the compiled controller device function.
        """

    @property
    def n(self) -> int:
        """Return the number of controlled state variables."""

        return self.compile_settings.n

    @property
    def dt_min(self) -> float:
        """Return the minimum supported step size."""

        return self.compile_settings.dt_min

    @property
    def dt_max(self) -> float:
        """Return the maximum supported step size."""

        return self.compile_settings.dt_max

    @property
    def dt(self) -> float:
        """Return the initial step size."""

        return self.compile_settings.dt

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` if the controller is adaptive."""

        return self.compile_settings.is_adaptive

    @property
    def atol(self) -> ndarray:
        """Return absolute tolerance."""

        return self.compile_settings.atol

    @property
    def rtol(self) -> ndarray:
        """Return relative tolerance."""

        return self.compile_settings.rtol

    @property
    def settings_dict(self) -> dict[str, object]:
        """Return the compile-time settings as a dictionary."""
        return self.compile_settings.settings_dict

    def update(
        self,
        updates_dict: Optional[dict[str, object]] = None,
        silent: bool = False,
        **kwargs: object,
    ) -> set[str]:
        """Propagate configuration updates to the compiled controller.

        Parameters
        ----------
        updates_dict
            Dictionary of configuration values to update.
        silent
            When ``True`` suppress warnings for recognised but unused
            controller parameters.
        **kwargs
            Additional configuration key-value pairs to update.

        Returns
        -------
        set[str]
            Names of parameters that were applied successfully.

        Raises
        ------
        KeyError
            Raised when an update references parameters that are not defined
            for any controller.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        # Track newly user-set step params
        for key in ("dt", "dt_min", "dt_max"):
            if key in updates_dict:
                self._user_step_params[key] = updates_dict[key]

        recognised = self._apply_filter_coefficients(updates_dict)
        recognised |= self.update_compile_settings(updates_dict, silent=True)
        unrecognised = set(updates_dict.keys()) - recognised

        # Check if unrecognized parameters are valid step controller parameters
        # but not applicable to this specific controller
        valid_but_inapplicable = unrecognised & ALL_STEP_CONTROLLER_PARAMETERS
        truly_invalid = unrecognised - ALL_STEP_CONTROLLER_PARAMETERS

        # Mark valid controller parameters as recognized to prevent error
        # propagation
        recognised |= valid_but_inapplicable

        if valid_but_inapplicable and not silent:
            controller_type = self.__class__.__name__
            params_str = ", ".join(sorted(valid_but_inapplicable))
            warnings.warn(
                (
                    f"Parameters {{{params_str}}} are not recognized by "
                    f"{controller_type}; updates have been ignored."
                ),
                UserWarning,
                stacklevel=2,
            )

        if not silent and truly_invalid:
            raise KeyError(
                f"Unrecognized parameters in update: {truly_invalid}. "
                "These parameters were not updated.",
            )

        self._ensure_sane_bounds()
        self.register_buffers()
        return recognised
