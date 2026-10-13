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
    >>> from cubie.odesystems.layout import SystemLayout
    >>> from cubie.odesystems.variables import (
    ...     Parameter, ParameterSet, VariableSet,
    ... )
    >>> # Subclass and override build() to use:
    >>> class MyODE(BaseODE):
    ...     def build(self):
    ...         pass  # compile dxdt here
    >>> parameters = VariableSet.of(Parameter, {"k": 1.0})
    >>> ode = MyODE(
    ...     float32,
    ...     parameters,
    ...     SystemLayout(
    ...         states=["x"], parameters=ParameterSet.build(parameters)
    ...     ),
    ...     initial_values={"x": 0.0},
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
from collections import OrderedDict
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


# The number of batches, layouts and sweeps whose compiled parameters
# and variable sets a system keeps, so repeated solves rebuild nothing.
VIEW_CACHE_SIZE = 8


def _remember(cache: OrderedDict, key: Any, build: Callable) -> Any:
    """Return ``cache[key]``, building and storing it when missing."""
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    value = build()
    cache[key] = value
    if len(cache) > VIEW_CACHE_SIZE:
        cache.popitem(last=False)
    return value


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

    We keep the parameters' defaults and the states' initial values
    outside the compile settings, so changing one rebuilds nothing
    unless it changes a value we compile into the code.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        parameters: ValuedSet,
        layout: SystemLayout,
        initial_values: Optional[Mapping[str, float]] = None,
        operation_ordering: str = operation_ordering_default(),
        name: Optional[str] = None,
    ) -> None:
        """Initialize the ODE system.

        Parameters
        ----------
        precision
            Precision factory used for calculations.
        parameters
            The parameters with their defaults.
        layout
            The variables in each device buffer and the parameters as
            the first batch compiles them.
        initial_values
            Default initial values by state name. States left out start
            from ``0.0``.
        operation_ordering
            Generated-operation ordering policy:
            ``"liveness_auto"``, ``"kahn"``, ``"greedy"``, or
            ``"dfs"``. Defaults to ``CUBIE_OPERATION_ORDERING``
            (``liveness_auto`` when unset).
        name
            Printable identifier for the system. Defaults to ``None``.

        Raises
        ------
        ValueError
            If the layout compiles different parameters from
            ``parameters``.
        """
        super().__init__()
        self._parameters = parameters
        self._check_parameters(layout.parameters)
        self._initial_values = dict(initial_values or {})
        self._batches = OrderedDict()
        self._swept_sets = OrderedDict()
        self._state_sets = OrderedDict()
        self.setup_compile_settings(
            ODEData(
                precision=precision,
                layout=layout,
                operation_ordering=operation_ordering,
            )
        )
        self.name = name

    @property
    def mass(self) -> Any:
        """Return the system's mass matrix.

        ``None`` implies identity, otherwise a diagonal 0/1 matrix
        produced in structural simplification.
        """

        return self.compile_settings.mass

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
        # return ODECache(dxdt=dxdt)

    def set_default_parameters(self, values: Mapping[str, float]) -> None:
        """Set parameter defaults.

        Afterwards we compile every parameter that is not swept at its
        default, dropping any values a solve gave for its own batch. We
        rebuild only when a compiled-in value changes, so a new default
        for a swept parameter rebuilds nothing.

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
        self._batches.clear()
        self._swept_sets.clear()
        self.set_swept_parameters(self.swept_parameters)

    def set_initial_values(self, values: Mapping[str, float]) -> None:
        """Set the default initial values of states.

        We read initial values only when a run starts, so setting them
        never rebuilds the system.

        Parameters
        ----------
        values
            State names mapped to new default initial values.

        Raises
        ------
        KeyError
            If a name is not a state of the system.
        """
        unknown = sorted(set(values) - set(self.states.names))
        if unknown:
            raise KeyError(f"{unknown} are not states of this system.")
        self._initial_values.update(
            {name: float(value) for name, value in values.items()}
        )
        self._state_sets.clear()

    def set_swept_parameters(self, names: Iterable[str]) -> None:
        """Sweep ``names`` and compile every other parameter at its default.

        Parameters
        ----------
        names
            Names of the parameters to sweep, in any order. The rows of
            the parameters array follow the system's parameter order.

        Raises
        ------
        KeyError
            If a name is not a parameter of the system.
        ValueError
            If a name repeats.
        """
        self.apply_batch(self.batch_for(names))

    def apply_batch(self, parameters: ParameterSet) -> None:
        """Compile the system for a batch's parameters.

        Parameters
        ----------
        parameters
            The parameters as the batch compiles them, built from this
            system's parameters.

        Raises
        ------
        ValueError
            If ``parameters`` names different parameters from the
            system.
        """
        self._check_parameters(parameters)
        compiled = self.compile_settings.parameters
        if parameters is not compiled and parameters != compiled:
            self.update(parameters=parameters)

    def _check_parameters(self, parameters: ParameterSet) -> None:
        """Raise ``ValueError`` if ``parameters`` are not this system's."""
        if parameters.names != self._parameters.names:
            raise ValueError(
                f"The batch compiles parameters {list(parameters.names)}, "
                f"but the system has {list(self._parameters.names)}."
            )

    def batch_for(
        self,
        swept: Iterable[str] = (),
        values: Optional[Mapping[str, float]] = None,
    ) -> ParameterSet:
        """Return this system's parameters as a batch compiles them.

        We build the same batch once, so a repeated solve compares it
        with the compiled parameters by identity.

        Parameters
        ----------
        swept
            Names of the parameters to sweep, in any order.
        values
            Values to compile in for some of the other parameters.

        Returns
        -------
        ParameterSet
            The compiled parameters, as :meth:`ParameterSet.build`
            makes them from this system's parameters.
        """
        swept = tuple(swept)
        values = values or {}
        key = (swept, tuple(sorted(values.items())))
        return _remember(
            self._batches,
            key,
            lambda: ParameterSet.build(self._parameters, swept, values),
        )

    def swept_set_for(self, parameters: ParameterSet) -> ValuedSet:
        """Return a batch's swept parameters with their defaults.

        Parameters
        ----------
        parameters
            The parameters as the batch compiles them.
        """
        swept = parameters.swept
        return _remember(
            self._swept_sets,
            swept,
            lambda: self._parameters.subset(swept),
        )

    def _states_of(self, layout: SystemLayout) -> ValuedSet:
        """Return the states of ``layout`` with our initial values."""
        return _remember(
            self._state_sets,
            layout,
            lambda: ValuedSet(
                [
                    State(
                        state.name,
                        state.unit,
                        self._initial_values.get(state.name, 0.0),
                    )
                    for state in layout.states
                ]
            ),
        )

    @property
    def swept_parameters(self) -> Tuple[str, ...]:
        """Names of the parameters array's rows, in order."""
        return self.compile_settings.swept_parameters

    @property
    def fixed_parameter_values(self) -> Dict[str, float]:
        """Values compiled into the code, keyed by parameter name."""
        return self.compile_settings.fixed_parameter_values

    @property
    def parameters(self) -> ValuedSet:
        """Every parameter with its default."""
        return self._parameters

    @property
    def compiled_parameters(self) -> ParameterSet:
        """Every parameter as the current batch compiles it."""
        return self.compile_settings.parameters

    @property
    def states(self) -> ValuedSet:
        """Every state with its default initial value."""
        return self._states_of(self.compile_settings.layout)

    @property
    def initial_values(self) -> ValuedSet:
        """Alias for :attr:`states`."""
        return self.states

    @property
    def layout(self) -> SystemLayout:
        """The variables in each device buffer, in buffer order."""
        return self.compile_settings.layout

    @property
    def observables(self) -> VariableSet:
        """The observables the system records."""
        return self.compile_settings.observables

    @property
    def num_states(self) -> int:
        """Number of state variables."""
        return self.compile_settings.num_states

    @property
    def num_observables(self) -> int:
        """Number of observable variables."""
        return self.compile_settings.num_observables

    @property
    def num_parameters(self) -> int:
        """Number of parameters."""
        return self.compile_settings.num_parameters

    @property
    def num_drivers(self) -> int:
        """Length of the drivers buffer the device functions read."""
        return self.compile_settings.num_drivers

    @property
    def driver_derivatives(self) -> Tuple[Tuple[int, int], ...]:
        """The driver index and order of each driver derivative read.

        The derivatives follow the drivers in the drivers buffer, in
        this order.
        """
        return self.compile_settings.driver_derivatives

    @property
    def sizes(self):
        """System component sizes cached for solvers."""
        return self.compile_settings.sizes

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
