"""Containers describing ODE system metadata used by factories.

Published Classes
-----------------
:class:`SystemSizes`
    Frozen counts for each component category in an ODE system.

    >>> sizes = SystemSizes(states=4, observables=2, swept_parameters=3,
    ...                     drivers=1)
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

from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from attrs import (
    Factory,
    cmp_using as attrs_cmp_using,
    define,
    evolve,
    field,
    frozen,
)
from attrs.validators import (
    in_ as attrsval_in,
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
)
from numpy import array as np_array, float64 as np_float64


from cubie._env import operation_ordering_default
from cubie.CUDAFactory import CUDAFactoryConfig
from cubie._utils import (
    PrecisionDType,
    mass_equal,
)
from cubie.odesystems.SystemValues import SystemValues


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


def _runtime_values_converter(value: Any) -> Any:
    """Seal the structure of a runtime-valued container.

    Values stay updatable in place: they are runtime data outside
    configuration identity.
    """
    if value is None:
        return None
    return value.freeze(values_writable=True)


def _parameters_converter(value: Any) -> Any:
    """Seal the parameter container."""
    if value is None:
        return None
    return value.freeze(values_writable=False)


def _ordered_names(names: Iterable[str]) -> Tuple[str, ...]:
    """Return ``names`` as a tuple of strings, rejecting repeats."""
    ordered = tuple(str(name) for name in names)
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"Swept parameters {list(ordered)} repeat a name.")
    return ordered


def _sorted_values(
    values: Union[Mapping[str, float], Iterable[Tuple[str, float]]],
) -> Tuple[Tuple[str, float], ...]:
    """Return ``values`` as name-sorted ``(name, value)`` pairs."""
    if isinstance(values, Mapping):
        values = values.items()
    return tuple(
        sorted((str(name), float(value)) for name, value in values)
    )


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
        Number of driver variables in the system.

    Notes
    -----
    This data class is passed to CUDA kernels so they can size device buffers
    and shared-memory structures correctly.
    """

    states: int = field(validator=attrsval_instance_of(int))
    observables: int = field(validator=attrsval_instance_of(int))
    swept_parameters: int = field(validator=attrsval_instance_of(int))
    drivers: int = field(validator=attrsval_instance_of(int))


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
        Number of driver or forcing functions. Defaults to ``1``.
    swept_parameters
        Names of the parameters read from the parameters array, in
        row order.
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

    parameters: Optional[SystemValues] = field(
        converter=_parameters_converter,
        validator=attrsval_optional(
            attrsval_instance_of(
                SystemValues,
            ),
        ),
    )
    initial_states: SystemValues = field(
        converter=_runtime_values_converter,
        validator=attrsval_optional(
            attrsval_instance_of(
                SystemValues,
            ),
        ),
    )
    observables: SystemValues = field(
        converter=_runtime_values_converter,
        validator=attrsval_optional(
            attrsval_instance_of(
                SystemValues,
            ),
        ),
    )
    num_drivers: int = field(validator=attrsval_instance_of(int), default=1)
    swept_parameters: Tuple[str, ...] = field(
        default=(), converter=_ordered_names
    )
    fixed_parameters: Tuple[Tuple[str, float], ...] = field(
        default=(), converter=_sorted_values
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
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.parameters is None:
            return
        names = set(self.parameters.names)
        swept = set(self.swept_parameters)
        fixed = {name for name, _ in self.fixed_parameters}
        if not swept <= names or fixed != names - swept:
            raise ValueError(
                f"Swept parameters {sorted(swept)} and fixed parameters "
                f"{sorted(fixed)} must together name each parameter of "
                f"this system once."
            )

    def update(
        self, updates_dict: dict = None, **kwargs
    ) -> Tuple["ODEData", Set[str], Set[str]]:
        """Derive a replacement snapshot, propagating precision changes.

        A changed ``precision`` re-materialises every embedded
        :class:`SystemValues` container at the new precision on the
        replacement snapshot, so packed value arrays always match the
        configured precision.
        """
        replacement, recognized, changed = super().update(
            updates_dict, **kwargs
        )
        if "precision" in changed:
            precision = replacement.precision
            reprecisioned = {}
            for name in (
                "parameters",
                "initial_states",
                "observables",
            ):
                container = getattr(replacement, name)
                if container is not None:
                    reprecisioned[name] = container.with_precision(precision)
            replacement = evolve(replacement, **reprecisioned)
        return replacement, recognized, changed

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        return self.initial_states.n

    @property
    def num_observables(self) -> int:
        """Number of observable variables."""
        return self.observables.n

    @property
    def num_parameters(self) -> int:
        """Number of parameters."""
        return self.parameters.n

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
        )

    @property
    def fixed_parameter_values(self) -> Dict[str, float]:
        """Values compiled into the code, keyed by parameter name."""
        return dict(self.fixed_parameters)

    @property
    def mass(self) -> Any:
        """Return the cached solver mass matrix."""
        return self._mass

    @property
    def parameter_values(self) -> Dict[str, float]:
        """Parameter values as plain floats keyed by name."""
        return self.parameters.as_float_dict

    @property
    def initial_state_values(self) -> Dict[str, float]:
        """Initial state values as plain floats keyed by name."""
        return self.initial_states.as_float_dict

    @classmethod
    def from_BaseODE_initargs(
        cls,
        precision: PrecisionDType,
        initial_values: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, float]] = None,
        observables: Optional[Dict[str, float]] = None,
        default_initial_values: Optional[Dict[str, float]] = None,
        default_parameters: Optional[Dict[str, float]] = None,
        default_observable_names: Optional[Dict[str, float]] = None,
        num_drivers: int = 1,
        operation_ordering: str = operation_ordering_default(),
        swept_parameters: Iterable[str] = (),
    ) -> "ODEData":
        """Create :class:`ODEData` from ``BaseODE`` initialization arguments.

        Parameters
        ----------
        initial_values
            Initial values for state variables.
        parameters
            Parameter values for the system.
        observables
            Auxiliary variables to track during simulation.
        default_initial_values
            Default initial values if ``initial_values`` omits entries.
        default_parameters
            Default parameter values if ``parameters`` omits entries.
        default_observable_names
            Default observable names if ``observables`` omits entries.
        precision
            Precision factory used for calculations.
        num_drivers
            Number of driver or forcing functions. Defaults to ``1``.
        operation_ordering
            Generated-operation ordering policy: stable ``"kahn"``,
            fixed ``"greedy"`` or ``"dfs"``, or thresholded
            ``"liveness_auto"`` selection.
        swept_parameters
            Names of the parameters read from the parameters array.
            Every other parameter compiles in at its default.

        Returns
        -------
        ODEData
            Initialised data container for CUDA compilation.
        """
        init_values = SystemValues(
            initial_values, precision, default_initial_values, name="States"
        )
        parameters = SystemValues(
            parameters,
            precision,
            default_parameters,
            name="Parameters",
        )
        observables = SystemValues(
            observables,
            precision,
            default_observable_names,
            name="Observables",
        )

        swept_parameters = tuple(swept_parameters)
        fixed_parameters = {
            name: value
            for name, value in parameters.as_float_dict.items()
            if name not in swept_parameters
        }
        return cls(
            parameters=parameters,
            initial_states=init_values,
            observables=observables,
            precision=precision,
            num_drivers=num_drivers,
            operation_ordering=operation_ordering,
            swept_parameters=swept_parameters,
            fixed_parameters=fixed_parameters,
        )
