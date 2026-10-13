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
    A system's states, parameters and observables, with derived sizes,
    as the compile settings of a :class:`~cubie.odesystems.baseODE.BaseODE`.

    >>> from numpy import float32
    >>> from cubie.odesystems.variables import (
    ...     Parameter, ParameterSet, Variable, VariableSet,
    ... )
    >>> declared = VariableSet.of(Parameter, {"a": 0.5, "g": 9.81})
    >>> data = ODEData(
    ...     precision=float32,
    ...     states=VariableSet.of(Variable, ["x", "y"]),
    ...     parameters=ParameterSet.build(declared, ["a"]),
    ...     observables=VariableSet.of(Variable, ["v"]),
    ... )
    >>> data.num_states
    2

See Also
--------
:mod:`cubie.odesystems.variables`
    The variable sets ``ODEData`` holds.
:class:`~cubie.CUDAFactory.CUDAFactoryConfig`
    Parent class providing precision and hashing.
:class:`~cubie.odesystems.baseODE.BaseODE`
    Abstract ODE factory that owns an ``ODEData`` as compile settings.
"""

from typing import Any, Dict, Tuple

from attrs import (
    Factory,
    cmp_using as attrs_cmp_using,
    define,
    field,
    frozen,
)
from attrs.validators import (
    deep_iterable as attrsval_deep_iterable,
    in_ as attrsval_in,
    instance_of as attrsval_instance_of,
)
from numpy import array as np_array, float64 as np_float64

from cubie._env import operation_ordering_default
from cubie.CUDAFactory import CUDAFactoryConfig
from cubie._utils import mass_equal
from cubie.odesystems.variables import ParameterSet, VariableSet


ALL_ODE_PARAMETERS = frozenset({"operation_ordering"})
OPERATION_ORDERINGS = ("kahn", "greedy", "dfs", "liveness_auto")


def _mass_matrix_converter(value: Any) -> Any:
    """Normalise a mass matrix to ``None`` or a sealed float64 array.

    SymPy matrices, NumPy arrays, and nested sequences normalise to
    one canonical numeric array; symbolic entries raise. The stored
    array is an owned read-only copy.
    """
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    array = np_array(value, dtype=np_float64)
    array.setflags(write=False)
    return array


@define
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


@frozen
class ODEData(CUDAFactoryConfig):
    """Bundle the variables and metadata an ODE system compiles against.

    We keep only fields that change the generated code here, so a
    change to any of them rebuilds the system. We keep defaults on the
    system instead: a state's default initial value never reaches the
    build, and a parameter's default reaches it only as a fixed value in
    ``parameters``.

    Parameters
    ----------
    parameters
        Each parameter as the current batch compiles it: swept, or
        fixed at a value.
    states
        The names and units of the states, in array order.
    observables
        The names and units of the observables, in array order.
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

    Notes
    -----
    This container holds only ODE-system state. Solver-helper request
    parameters (operator_beta, operator_gamma, preconditioner order,
    stage tableaus) belong to the requesting algorithm's compile
    settings and reach the system as immutable
    :class:`~cubie.odesystems.solver_helpers.SolverHelperRequest`
    values, so the system's identity never depends on helper request
    order.
    """

    parameters: ParameterSet = field(
        validator=attrsval_instance_of(ParameterSet)
    )
    states: VariableSet = field(
        validator=attrsval_instance_of(VariableSet)
    )
    observables: VariableSet = field(
        validator=attrsval_instance_of(VariableSet)
    )
    num_drivers: int = field(validator=attrsval_instance_of(int), default=1)
    driver_derivatives: Tuple[Tuple[int, int], ...] = field(
        default=(),
        validator=attrsval_deep_iterable(
            attrsval_deep_iterable(
                attrsval_instance_of(int), attrsval_instance_of(tuple)
            ),
            attrsval_instance_of(tuple),
        ),
    )
    operation_ordering: str = field(
        default=Factory(operation_ordering_default),
        validator=attrsval_in(OPERATION_ORDERINGS),
    )
    _mass: Any = field(
        default=None,
        converter=_mass_matrix_converter,
        eq=attrs_cmp_using(eq=mass_equal),
    )

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        return self.states.n

    @property
    def num_observables(self) -> int:
        """Number of observable variables."""
        return self.observables.n

    @property
    def num_parameters(self) -> int:
        """Number of parameters."""
        return self.parameters.n

    @property
    def swept_parameters(self) -> Tuple[str, ...]:
        """Names of the parameters array's rows, in order."""
        return self.parameters.swept

    @property
    def num_swept_parameters(self) -> int:
        """Number of swept parameters."""
        return len(self.swept_parameters)

    @property
    def sizes(self) -> SystemSizes:
        """System component sizes grouped for CUDA kernels."""
        return SystemSizes(
            states=self.num_states,
            observables=self.num_observables,
            swept_parameters=self.num_swept_parameters,
            drivers=self.num_drivers,
            driver_derivatives=len(self.driver_derivatives),
        )

    @property
    def fixed_parameter_values(self) -> Dict[str, float]:
        """Values compiled into the code, keyed by parameter name."""
        return self.parameters.fixed

    @property
    def mass(self) -> Any:
        """Return the cached solver mass matrix."""
        return self._mass
