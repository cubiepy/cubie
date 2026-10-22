"""Containers describing ODE system metadata used by factories.

Published Classes
-----------------
:class:`SystemSizes`
    Frozen counts for each component category in an ODE system.

    >>> sizes = SystemSizes(states=4, observables=2, swept_parameters=3,
    ...                     drivers=1, driver_derivatives=0)
    >>> sizes.states
    4

:class:`ODEData`
    Bundle of :class:`SystemValues` instances and derived sizes for CUDA
    compilation.

    >>> from numpy import float32
    >>> data = ODEData.from_BaseODE_initargs(
    ...     precision=float32,
    ...     default_initial_values={"x": 0.0, "y": 1.0},
    ...     default_parameters={"a": 0.5, "g": 9.81},
    ...     default_observable_names={"v": 0.0},
    ... )
    >>> data.num_states
    2

See Also
--------
:class:`~cubie.odesystems.SystemValues.SystemValues`
    Keyed parameter container stored inside ``ODEData``.
:class:`~cubie.CUDAFactory.CUDAFactoryConfig`
    Parent class providing precision and hashing.
:class:`~cubie.odesystems.baseODE.BaseODE`
    Abstract ODE factory that owns an ``ODEData`` as compile settings.
"""

from attrs import Factory, field, frozen
from attrs.validators import (
    in_ as attrsval_in,
    instance_of as attrsval_instance_of,
)

from cubie._env import operation_ordering_default
from cubie.CUDAFactory import CUDAFactoryConfig
from cubie.odesystems.layout import SystemLayout


ALL_ODE_PARAMETERS = frozenset({"operation_ordering"})
OPERATION_ORDERINGS = ("kahn", "greedy", "dfs", "liveness_auto")


@frozen
class SystemSizes:
    """Store counts for each component category in an ODE system.

    Parameters
    ----------
    states
        Number of state variables in the system.
    observables
        Number of observable variables in the system.
    swept_parameters
        Number of swept parameters.
    drivers
        Drivers-buffer length: drivers plus driver derivatives read.
    driver_derivatives
        Number of driver derivatives the equations read; they sit in
        the drivers buffer after the drivers.

    Notes
    -----
    This data class is passed to CUDA kernels so they can size device buffers
    and shared-memory structures correctly.
    """

    states: int = field(validator=attrsval_instance_of(int))
    observables: int = field(validator=attrsval_instance_of(int))
    swept_parameters: int = field(validator=attrsval_instance_of(int))
    drivers: int = field(validator=attrsval_instance_of(int))
    driver_derivatives: int = field(validator=attrsval_instance_of(int))

    @classmethod
    def of_layout(cls, layout: SystemLayout) -> "SystemSizes":
        return cls(
            states=layout.states.n,
            observables=layout.observables.n,
            swept_parameters=len(layout.swept_names),
            drivers=layout.num_drivers,
            driver_derivatives=len(layout.driver_derivatives),
        )


@frozen
class ODEData(CUDAFactoryConfig):
    """Bundle numerical values and metadata for an ODE system.

    Parameters
    ----------
    parameters
        Every parameter of the system with its default value.
    initial_states
        Initial state values for the ODE system.
    observables
        Observable variables to track during simulation.
    precision
        Precision factory used for numerical calculations. Defaults to
        :class:`numpy.float32`.
    num_drivers
        Drivers-buffer length: drivers plus driver derivatives read.
        Defaults to ``1``.
    driver_derivatives
        The driver index and order of each driver derivative the
        equations read, in the order the derivatives follow the
        drivers in the drivers buffer.
    swept_parameters
        Names of the parameters read from the parameters array.
    fixed_parameters
        Values compiled into the code for every other parameter.

    Notes
    -----
    This container holds only ODE-system state. Solver-helper request
    parameters (operator_beta, operator_gamma, preconditioner order,
    stage tableaus)
    belong to the requesting algorithm's compile settings and reach
    the system as immutable
    :class:`~cubie.odesystems.solver_helpers.SolverHelperRequest`
    values, so the system's identity never depends on helper request
    order.
    """

    operation_ordering: str = field(
        default=Factory(operation_ordering_default),
        validator=attrsval_in(OPERATION_ORDERINGS),
    )
