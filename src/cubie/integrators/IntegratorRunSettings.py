"""Settings of a single integrator run.

Published Classes
-----------------
:class:`IntegratorRunSettings`
    Algorithm and controller names, the driver functions and the loop.
"""

from typing import Callable, Optional

import attrs
from attrs import field, validators

from cubie._utils import device_function_field
from cubie.CUDAFactory import CUDAFactoryConfig


@attrs.frozen
class IntegratorRunSettings(CUDAFactoryConfig):
    """Algorithm and controller names, the driver functions and the loop.

    Attributes
    ----------
    algorithm
        Step algorithm name.
    step_controller
        Step controller name.
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
