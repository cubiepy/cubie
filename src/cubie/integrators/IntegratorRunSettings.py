"""Runtime configuration settings for numerical integration algorithms.

Published Classes
-----------------
:class:`IntegratorRunSettings`
    Attrs container holding algorithm name and step controller name
    alongside the inherited precision field.

    >>> from numpy import float32
    >>> settings = IntegratorRunSettings(
    ...     precision=float32, algorithm="erk", step_controller="pid"
    ... )
    >>> settings.algorithm
    'erk'

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactoryConfig`
    Parent class providing precision and numba type conversion.
:class:`~cubie.integrators.SingleIntegratorRunCore.SingleIntegratorRunCore`
    Consumer that uses these settings to select algorithm and
    controller factories.
"""

from typing import Callable, Optional

import attrs

from cubie._utils import device_function_field
from cubie.CUDAFactory import CUDAFactoryConfig


@attrs.frozen
class IntegratorRunSettings(CUDAFactoryConfig):
    """Container for runtime and controller settings used by IVP loops.

    Attributes
    ----------
    precision
        Numerical precision used for timing comparisons.
    algorithm
        Name of the integration step algorithm.
    step_controller
        Name of the step-size controller.
    auto_performance
        Fill unset unroll and placement settings at build; hash-excluded.
    loop_fn
        The loop's compiled device function, captured by ``update``.
    """

    algorithm: str = attrs.field(
        default="euler",
        validator=attrs.validators.instance_of(str),
    )
    step_controller: str = attrs.field(
        default="fixed",
        validator=attrs.validators.instance_of(str),
    )
    auto_performance: bool = attrs.field(
        default=True,
        validator=attrs.validators.instance_of(bool),
        eq=False,
    )
    loop_fn: Optional[Callable] = device_function_field()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
