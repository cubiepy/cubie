"""Step-size controller package.

Published Classes
-----------------
:class:`FixedStepController`
    Controller that enforces a constant time step.
:class:`AdaptiveIController`
    Integral-only adaptive step-size controller.
:class:`AdaptivePIController`
    Proportional--integral step-size controller.
:class:`AdaptivePIDController`
    Proportional--integral--derivative step-size controller.
:class:`GustafssonController`
    Gustafsson predictive controller for implicit integrators.

Module-Level Functions
----------------------
:func:`get_controller`
    Resolve a controller class from a settings mapping.

    >>> from numpy import float64
    >>> ctrl = get_controller(
    ...     precision=float64,
    ...     settings={"step_controller": "fixed", "dt": 0.01},
    ... )
    >>> ctrl.is_adaptive
    False

See Also
--------
:class:`~cubie.integrators.step_control.base_step_controller.BaseStepController`
    Abstract base class for all controllers.
:data:`~cubie.integrators.step_control.base_step_controller.ALL_STEP_CONTROLLER_PARAMETERS`
    Union of all accepted keyword arguments.
"""

from typing import Any, Dict, Mapping, Optional, Type

from cubie._utils import PrecisionDType

from .adaptive_I_controller import AdaptiveIController
from .adaptive_PI_controller import AdaptivePIController
from .adaptive_PID_controller import AdaptivePIDController
from .fixed_step_controller import FixedStepController
from .gustafsson_controller import GustafssonController
from .base_step_controller import (
    BaseStepController,
    CONTROLLER_GAIN_NAMES,
    CONTROLLER_GAIN_PARAMETERS,
    FILTER_COEFFICIENT_PRESETS,
    GAIN_CONTROLLER_CHAIN,
    filter_coefficients_to_gains,
    minimal_gain_controller,
    promoted_gain_controller,
)

__all__ = [
    "AdaptiveIController",
    "AdaptivePIController",
    "AdaptivePIDController",
    "GustafssonController",
    "FixedStepController",
    "get_controller",
    "CONTROLLER_GAIN_NAMES",
    "CONTROLLER_GAIN_PARAMETERS",
    "FILTER_COEFFICIENT_PRESETS",
    "GAIN_CONTROLLER_CHAIN",
    "filter_coefficients_to_gains",
    "minimal_gain_controller",
    "promoted_gain_controller",
    "_CONTROLLER_REGISTRY",
]

_CONTROLLER_REGISTRY: Dict[str, Type[BaseStepController]] = {
    "fixed": FixedStepController,
    "i": AdaptiveIController,
    "pi": AdaptivePIController,
    "pid": AdaptivePIDController,
    "gustafsson": GustafssonController,
}


def get_controller(
    precision: PrecisionDType,
    settings: Optional[Mapping[str, Any]] = None,
    warn_on_unused: bool = False,
    **kwargs: Any,
) -> BaseStepController:
    """Return a controller instance from a settings mapping.

    Parameters
    ----------
    precision
        Floating-point dtype used when compiling the controller.
    settings
        Mapping of step control keyword arguments supplied by the caller.
        Values in ``settings`` are merged with ``kwargs``. The mapping must
        include ``"step_controller"`` to identify the controller factory.
    warn_on_unused
        Emit a warning when unused settings remain after filtering.
    **kwargs
        Additional keyword arguments that override entries in
        ``settings``.

    Returns
    -------
    BaseStepController
        Instance of the requested controller.

    Raises
    ------
    ValueError
        Raised when ``step_controller`` does not match a known controller type
        or when required configuration keys are missing.
    """

    controller_settings = {}
    if settings is not None:
        controller_settings.update(settings)
    controller_settings.update(kwargs)

    step_controller_value = controller_settings.pop("step_controller", None)
    if step_controller_value is None:
        raise ValueError("No step controller specified")
    controller_key = step_controller_value.lower()

    try:
        controller_type = _CONTROLLER_REGISTRY[controller_key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Unknown controller type: {step_controller_value}"
        ) from exc

    controller_settings["precision"] = precision

    # Pass all settings to controller __init__ which uses build_config
    # internally build_config filters to valid config fields and handles
    # defaults
    return controller_type(**controller_settings)
