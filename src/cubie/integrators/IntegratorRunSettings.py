"""Settings of a single integrator run.

Published Classes
-----------------
:class:`IntegratorRunSettings`
    Algorithm and controller names and the compiled loop.
"""

from typing import Callable, Optional

import attrs
from attrs import field, validators

from cubie._utils import device_function_field
from cubie.CUDAFactory import CUDAFactoryConfig


@attrs.frozen
class IntegratorRunSettings(CUDAFactoryConfig):
    """Algorithm and controller names and the compiled loop.

    Attributes
    ----------
    algorithm
        Step algorithm name.
    step_controller
        Step controller name.
    loop_fn
        Compiled loop device function.
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
    loop_fn: Optional[Callable] = device_function_field()
