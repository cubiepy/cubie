"""Compile settings of a single integrator run.

Published Classes
-----------------
:class:`IntegratorRunSettings`
    Attrs container holding the algorithm and step controller names,
    the driver device functions and the captured loop device function
    alongside the inherited precision and compile flags.

    >>> from numpy import float32
    >>> settings = IntegratorRunSettings(
    ...     precision=float32, algorithm="erk", step_controller="pid"
    ... )
    >>> settings.algorithm
    'erk'

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactoryConfig`
    Parent class providing precision, compile flags and numba type
    conversion.
:class:`~cubie.integrators.SingleIntegratorRunCore.SingleIntegratorRunCore`
    Consumer that uses these settings to select algorithm and
    controller factories.
"""

from typing import Callable, Optional

import attrs
from attrs import field, validators

from cubie._utils import device_function_field
from cubie.CUDAFactory import CUDAFactoryConfig


@attrs.frozen
class IntegratorRunSettings(CUDAFactoryConfig):
    """Container for the run and controller settings used by IVP loops.

    Attributes
    ----------
    precision
        Numerical precision used for timing comparisons.
    algorithm
        Name of the integration step algorithm.
    step_controller
        Name of the step-size controller.
    drivers_fn
        Device function evaluating the drivers at a time.
    driver_derivative_fn
        Device function evaluating the drivers' time derivative.
    loop_fn
        The loop device function captured after every update.
    """

    algorithm: str = field(
        default="euler",
        converter=str.lower,
        validator=validators.instance_of(str),
    )
    step_controller: str = field(
        default="fixed",
        converter=str.lower,
        validator=validators.instance_of(str),
    )
    drivers_fn: Optional[Callable] = device_function_field()
    driver_derivative_fn: Optional[Callable] = device_function_field()
    loop_fn: Optional[Callable] = device_function_field()
