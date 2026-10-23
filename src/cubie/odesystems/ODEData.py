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
    Compile settings of a :class:`~cubie.odesystems.baseODE.BaseODE`:
    the precision, the operation ordering and the hash of the generated
    source.

    >>> from numpy import float32
    >>> data = ODEData(precision=float32, operation_ordering="kahn")
    >>> data.operation_ordering
    'kahn'

See Also
--------
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
        """Return the sizes of every buffer in ``layout``."""
        return cls(
            states=layout.states.n,
            observables=layout.observables.n,
            swept_parameters=len(layout.swept_names),
            drivers=layout.num_drivers,
            driver_derivatives=len(layout.driver_derivatives),
        )


@frozen
class ODEData(CUDAFactoryConfig):
    """Compile settings of an ODE system.

    Attributes
    ----------
    precision
        NumPy floating-point data type of the generated device functions.
    operation_ordering
        Ordering policy for the generated operations: ``"kahn"``,
        ``"greedy"``, ``"dfs"`` or ``"liveness_auto"``.
    fn_hash
        Hash of the values written into the generated source.
        ``BaseODE`` reads it from the specialiser on each update, so a
        new specialisation changes these settings.
    """

    operation_ordering: str = field(
        default=Factory(operation_ordering_default),
        validator=attrsval_in(OPERATION_ORDERINGS),
    )
    fn_hash: str = field(default="", validator=attrsval_instance_of(str))
