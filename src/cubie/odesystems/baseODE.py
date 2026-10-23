"""Base classes for defining and compiling CUDA-backed ODE systems.

Published Classes
-----------------
:class:`ODECache`
    Attrs container caching compiled CUDA device functions for an ODE
    system (``dxdt``, linear operator, preconditioner, etc.).

:class:`BaseODE`
    Abstract factory base for CUDA-backed ODE systems. Manages value
    containers, precision selection, and cache invalidation.

    >>> from numpy import float32
    >>> # Subclass and override build() to use:
    >>> class MyODE(BaseODE):
    ...     def build(self):
    ...         pass  # compile dxdt here
    >>> ode = MyODE(
    ...     precision=float32,
    ...     default_initial_values={"x": 0.0},
    ...     default_parameters={"k": 1.0},
    ... )
    >>> ode.num_states
    1

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactory`
    Parent factory class providing compilation and caching.
:class:`~cubie.odesystems.ODEData.ODEData`
    Compile settings container owned by ``BaseODE``.
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`
    Concrete subclass that generates device functions from SymPy
    expressions.
"""

from abc import abstractmethod
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
)

from attrs import define, field
from numpy import float32

from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie._utils import PrecisionDType
from cubie._env import operation_ordering_default
from cubie.odesystems.ODEData import ODEData
from cubie.odesystems._mass_utils import mass_diagonal_flags
from cubie.odesystems.solver_helpers import (
    HelperResult,
    OperationCounts,
    SolverHelperCache,
)
from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.variables import (
    ParameterSet,
    State,
    ValuedSet,
    VariableSet,
)


@define
class ODECache(CUDADispatcherCache):
    """Cache the compiled base outputs and helper products of an ODE build.

    Attributes
    ----------
    dxdt
        Compiled right-hand-side device function.
    observables
        Compiled observables device function.
    helpers
        Memoized solver-helper factories and bound members for this
        build. A true compile-setting change produces a fresh
        ``ODECache`` and therefore a fresh member map.
    operation_counts
        Binary-operator counts of the ``dxdt`` and observables sources.
    """

    dxdt_fn: Callable = field()
    observables_fn: Optional[Callable] = field(default=None)
    helpers: SolverHelperCache = field(factory=SolverHelperCache)
    operation_counts: OperationCounts = field(factory=OperationCounts)


class BaseODE(CUDAFactory):
    """Abstract base for CUDA-backed ordinary differential equation systems.

    Subclasses override :meth:`build` to compile a CUDA device function that
    advances the system state and, optionally, provide analytic helpers via
    :meth:`get_solver_helper`. The base class handles value management,
    precision selection, and caching through :class:`CUDAFactory`.

    Notes
    -----
    Only functions cached during :meth:`build` (typically ``dxdt``) are
    available on this base class. Solver helper functions such as the linear
    operator or preconditioner are generated only by subclasses like
    :class:`SymbolicODE`.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        parameters: ValuedSet,
        specialiser: CUDAFactory,
        initial_values: Optional[Mapping[str, float]] = None,
        operation_ordering: str = operation_ordering_default(),
        name: Optional[str] = None,
    ) -> None:
        """Initialize the ODE system.

        Parameters
        ----------
        initial_values
            Initial values for state variables.
        parameters
            Parameter values for the system.
        observables
            Observable values to track.
        default_initial_values
            Default initial values if ``initial_values`` omits entries.
        default_parameters
            Default parameter values if ``parameters`` omits entries.
        default_observable_names
            Default observable names if ``observables`` omits entries.
        precision
            Precision factory used for calculations. Defaults to
            :class:`numpy.float32`.
        num_drivers
            Drivers-buffer length: drivers plus driver derivatives read.
            Defaults to ``1``.
        operation_ordering
            Generated-operation ordering policy:
            ``"liveness_auto"``, ``"kahn"``, ``"greedy"``, or
            ``"dfs"``. Defaults to ``CUBIE_OPERATION_ORDERING``
            (``liveness_auto`` when unset).
        name
            Printable identifier for the system. Defaults to ``None``.
        swept_parameters
            Names of the parameters read from the parameters array.
            Every other parameter compiles in at its default.
        driver_derivatives
            The driver index and order of each driver derivative the
            equations read, in the order the derivatives follow the
            drivers in the drivers buffer.
        """
        super().__init__()
        self._parameters = parameters
        self._initial_values = VariableSet.of(State, initial_values or {})
        self._specialiser = specialiser
        self.setup_compile_settings(
            ODEData(
                precision=precision,
                operation_ordering=operation_ordering,
                fn_hash=specialiser.fn_hash,
            )
        )
        self.name = name

    @property
    def specialiser(self) -> CUDAFactory:
        return self._specialiser

    @property
    def mass(self) -> Any:
        """Return the system's mass matrix.

        ``None`` implies identity, otherwise a diagonal 0/1 matrix
        produced in structural simplification.
        """

        return self._specialiser.mass

    @property
    def mass_diagonal_flags(self) -> tuple:
        """Return per-state mass flags, ``True`` for a differential row."""

        return mass_diagonal_flags(self.mass, self.num_states)

    @property
    def operation_ordering(self) -> str:
        """Return the generated-operation ordering policy."""

        return self.compile_settings.operation_ordering

    def __repr__(self) -> str:
        if self.name is None:
            name = "ODE System"
        else:
            name = self.name
        return (
            f"{name}"
            "--"
            f"\n{self.states},"
            f"\n{self.parameters},"
            f"\n{self.observables},"
            f"\n{self.num_drivers})"
        )

    @abstractmethod
    def build(self) -> ODECache:
        """Compile the ``dxdt`` system as a CUDA device function.

        Returns
        -------
        ODECache
            Cache containing the built ``dxdt`` function. Subclasses may add
            further solver helpers to this cache as needed.
        """

    def _update(self, updates: Dict[str, Any], silent: bool) -> Set[str]:
        recognised = self._specialiser.update(updates, silent=True)
        recognised |= self.update_compile_settings(updates, silent=True)
        self.update_compile_settings(
            {"fn_hash": self._specialiser.fn_hash}, silent=True
        )
        return recognised

    def set_swept_parameters(
        self,
        names: Iterable[str],
        fixed: Optional[Mapping[str, float]] = None,
    ) -> None:
        names = tuple(names)
        fixed = dict(fixed or {})
        parameters = self._parameters
        compiled = self._derived(
            "compiled_parameters",
            (parameters, names, fixed),
            lambda: ParameterSet.build(parameters, names, fixed),
        )
        self.update(compiled_parameters=compiled)

    def set_default_parameters(self, values: Mapping[str, float]) -> None:
        """Set parameter defaults.

        Parameters that are not swept compile in at their new default.

        Parameters
        ----------
        values
            Parameter names mapped to new default values.

        Raises
        ------
        KeyError
            If a name is not a parameter of the system.
        """
        self._parameters = self._parameters.with_defaults(values)
        # We copy each fixed value into the compiled set when we build
        # it, so build it again from the new defaults.
        self.set_swept_parameters(self.swept_parameters)

    def set_initial_values(self, values: Mapping[str, float]) -> None:
        unknown = sorted(set(values) - set(self.states.names))
        if unknown:
            raise KeyError(f"{unknown} are not states of this system.")
        self._initial_values = VariableSet.of(
            State, {**self._initial_values.defaults, **values}
        )

    @property
    def swept_parameters(self) -> Tuple[str, ...]:
        """Names of the parameters array's rows, in order."""
        return self._specialiser.swept_parameters

    @property
    def fixed_parameter_values(self) -> Dict[str, float]:
        """Values compiled into the code, keyed by parameter name."""
        return self._specialiser.fixed_parameter_values

    @property
    def parameters(self) -> "SystemValues":
        """Every parameter with its default value."""
        return self._parameters

    @property
    def compiled_parameters(self) -> ParameterSet:
        return self._specialiser.compiled_parameters

    @property
    def swept_set(self) -> ValuedSet:
        return self._parameters.subset(self.compiled_parameters.swept)

    @property
    def states(self) -> ValuedSet:
        defaults = self._initial_values.defaults
        return ValuedSet(
            [
                State(state.name, state.unit, defaults.get(state.name, 0.0))
                for state in self.layout.states
            ]
        )

    @property
    def initial_values(self) -> ValuedSet:
        """Alias for :attr:`states`."""
        return self.states

    @property
    def layout(self) -> SystemLayout:
        return self._specialiser.layout

    @property
    def observables(self) -> VariableSet:
        return self.layout.observables

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        return self.layout.states.n

    @property
    def num_observables(self) -> int:
        """Number of observable variables."""
        return self.layout.observables.n

    @property
    def num_parameters(self) -> int:
        """Number of parameters."""
        return self._parameters.n

    @property
    def num_drivers(self) -> int:
        """Length of the drivers buffer the device functions read."""
        return self.layout.num_drivers

    @property
    def driver_derivatives(self) -> Tuple[Tuple[int, int], ...]:
        """The driver index and order of each driver derivative read.

        The derivatives follow the drivers in the drivers buffer, in
        this order.
        """
        return self.layout.driver_derivatives

    @property
    def sizes(self):
        """System component sizes cached for solvers."""
        return self._specialiser.sizes

    def __getstate__(self) -> dict:
        """Return the pickled state without the build cache."""
        state = dict(self.__dict__)
        state["_cache"] = None
        state["_cache_valid"] = False
        return state

    def copy(self) -> "BaseODE":
        """Return an independent system with these values and no build."""
        return deepcopy(self)

    @property
    def dxdt_fn(self):
        """Compiled ``dxdt(state, parameters, drivers, observables, out, t)``
        device function.
        """
        return self.get_cached_output("dxdt_fn")

    @property
    def observables_fn(self) -> Callable:
        """Compiled ``get_observables(state, parameters, drivers, observables,
        t)`` device function.
        """
        return self.get_cached_output("observables_fn")

    @property
    def operation_count(self) -> int:
        """Binary-operator count of the ``dxdt`` and observables sources."""
        return self.get_cached_output("operation_counts").total(
            ("dxdt", "observables")
        )

    def get_solver_helper(
        self,
        role: str,
        **request_kwargs: Any,
    ) -> HelperResult:
        """Return the bound helper member for one role and variant.

        Mass-consuming helpers read the system's own :attr:`mass`.

        Parameters
        ----------
        role
            Registered role name or preconditioner type name.
        **request_kwargs
            Remaining :class:`SolverHelperRequest` fields.

        Returns
        -------
        HelperResult
            The bound device callable and its typed metadata.

        Raises
        ------
        NotImplementedError
            Always; only :class:`SymbolicODE` generates helpers.
        """
        raise NotImplementedError(
            "Solver helpers are generated from symbolic systems; "
            f"{type(self).__name__} does not provide "
            f"'{role}'. Define the system through "
            "create_ODE_system or SymbolicODE to use implicit "
            "algorithms."
        )
