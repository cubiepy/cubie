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
    Set,
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


# Keep this many specialisations, and this many of each kind of
# variable set built from them, on each system.
SPECIALISATION_CACHE_SIZE = 8


def _remember(cache: OrderedDict, key: Any, build: Callable) -> Any:
    """Return ``cache[key]``, building and storing it when missing."""
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    value = build()
    cache[key] = value
    if len(cache) > SPECIALISATION_CACHE_SIZE:
        cache.popitem(last=False)
    return value


def _request_key(
    swept: Tuple[str, ...], fixed: Optional[Mapping[str, float]]
) -> tuple:
    """Return the memo key for a request to :meth:`BaseODE.specialise`."""
    if not fixed:
        return (swept, ())
    values = sorted(
        (str(name), float(value)) for name, value in fixed.items()
    )
    return (swept, tuple(values))


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
        swept_parameters: Iterable[str] = (),
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
        swept_parameters
            Names of the parameters to sweep in the specialisation the
            system compiles first. The system compiles every other
            parameter in at its default.
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
        """
        super().__init__()
        self._parameters = parameters
        self._initial_values = dict(initial_values or {})
        self._requests = OrderedDict()
        self._specialisations = OrderedDict()
        self._swept_sets = OrderedDict()
        self._state_sets = OrderedDict()
        specialisation = self.specialise(swept_parameters)
        self._specialisation = specialisation
        self.setup_compile_settings(
            ODEData(
                precision=precision,
                layout=specialisation.layout,
                operation_ordering=operation_ordering,
                mass=specialisation.mass_matrix,
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

    @abstractmethod
    def _build_specialisation(self, parameters: ParameterSet) -> Any:
        """Return the system specialised for ``parameters``.

        Parameters
        ----------
        parameters
            Every parameter, swept or fixed at the value to compile in.

        Returns
        -------
        object
            A specialisation with a ``layout`` (:class:`SystemLayout`)
            and a ``mass_matrix`` (``None`` for an ODE).
        """

    def specialise(
        self,
        swept: Iterable[str] = (),
        fixed: Optional[Mapping[str, float]] = None,
    ) -> Any:
        """Return the system specialised for a batch.

        Parameters
        ----------
        swept
            Names of the parameters to read from the parameters array,
            in any order.
        fixed
            Values to compile in for some of the other parameters.
            The system compiles every parameter that is neither swept
            nor named here at its default.

        Returns
        -------
        object
            The specialisation. Return the same object for a repeated
            request, and for a request that differs only by fixing a
            parameter at its default.

        Raises
        ------
        KeyError
            If a name is not a parameter of the system.
        ValueError
            If a swept name repeats or also has a fixed value.
        """
        swept = tuple(swept)
        request = _request_key(swept, fixed)
        found = self._requests.get(request)
        if found is not None:
            self._requests.move_to_end(request)
            return found
        parameters = ParameterSet.build(self._parameters, swept, fixed)
        # We key built specialisations on their compiled values, so
        # two requests for the same values share one.
        found = _remember(
            self._specialisations,
            parameters._cubie_canonical_(),
            lambda: self._build_specialisation(parameters),
        )
        return _remember(self._requests, request, lambda: found)

    @property
    def specialisation(self) -> Any:
        """The specialisation the system compiles."""
        return self._specialisation

    def _update(self, updates: Dict[str, Any], silent: bool) -> Set[str]:
        """Apply settings, compiling a new ``specialisation`` if given.

        Parameters
        ----------
        updates
            Setting names to new values.
        silent
            Whether :meth:`update` ignores unrecognised names.

        Returns
        -------
        set[str]
            The recognised names.
        """
        if "specialisation" not in updates:
            return self.update_compile_settings(updates, silent=True)
        specialisation = updates.pop("specialisation")
        if specialisation is self._specialisation:
            recognised = self.update_compile_settings(updates, silent=True)
            return recognised | {"specialisation"}
        derived = {
            "layout": specialisation.layout,
            "mass": specialisation.mass_matrix,
        }
        recognised = self.update_compile_settings(
            {**updates, **derived}, silent=True
        )
        self._specialisation = specialisation
        return (recognised - (set(derived) - set(updates))) | {
            "specialisation"
        }

    def set_swept_parameters(self, names: Iterable[str]) -> None:
        """Compile the system with ``names`` swept and the rest at defaults.

        Parameters
        ----------
        names
            Names of the parameters to sweep, in any order.

        Raises
        ------
        KeyError
            If a name is not a parameter of the system.
        ValueError
            If a name repeats.
        """
        self.update(specialisation=self.specialise(names))

    def set_default_parameters(self, values: Mapping[str, float]) -> None:
        """Set parameter defaults.

        Afterwards the system compiles its swept parameters as before
        and every other parameter at its default.

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
        # We keep the specialisations, which we key on their compiled
        # values, and drop the requests, because a request that leaves
        # a parameter at its default now means a different value.
        self._requests.clear()
        self._swept_sets.clear()
        self.set_swept_parameters(self.swept_parameters)

    def set_initial_values(self, values: Mapping[str, float]) -> None:
        """Set the default initial values of states.

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

    def swept_set_for(self, layout: SystemLayout) -> ValuedSet:
        """Return a layout's swept parameters with their defaults.

        Parameters
        ----------
        layout
            The layout of a specialisation of this system.
        """
        swept = layout.swept_names
        return _remember(
            self._swept_sets,
            swept,
            lambda: self._parameters.subset(swept),
        )

    def states_for(self, layout: SystemLayout) -> ValuedSet:
        """Return a layout's states with their default initial values.

        Parameters
        ----------
        layout
            The layout of a specialisation of this system.
        """
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
        """Every parameter as the system compiles it."""
        return self.compile_settings.parameters

    @property
    def states(self) -> ValuedSet:
        """Every state with its default initial value."""
        return self.states_for(self.compile_settings.layout)

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
