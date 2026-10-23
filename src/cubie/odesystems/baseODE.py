"""Base classes for defining and compiling CUDA-backed ODE systems.

Published Classes
-----------------
:class:`ODECache`
    Attrs container caching compiled CUDA device functions for an ODE
    system (``dxdt``, linear operator, preconditioner, etc.).

:class:`BaseODE`
    Abstract factory base for CUDA-backed ODE systems. Owns the
    declared defaults, the child factory that assembles the system for
    the parameters a batch compiles, and the codegen settings.

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
    Set,
    Tuple,
)

from attrs import define, field

from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie._utils import PrecisionDType
from cubie._env import operation_ordering_default
from cubie.odesystems.ODEData import ODEData, SystemSizes
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

    ``BaseODE`` keeps parameter defaults and initial values outside its
    compile settings, so it rebuilds only when a new default changes a
    value compiled into the code.
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
        precision
            Precision factory used for calculations.
        parameters
            The parameters with their defaults.
        specialiser
            Child factory holding the ``compiled_parameters`` setting,
            every parameter swept or fixed at a value, and building the
            ``layout``, ``mass`` and equations of the system assembled
            for them.
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
        """The child factory that assembles the compiled parameters."""
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
        """Apply settings to the specialiser, then to this system.

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
        """Set the swept parameters to ``names``.

        Parameters
        ----------
        names
            Names of the parameters to sweep, in any order.
        fixed
            Fixed values for a subset of the other parameters. This
            method fixes every parameter neither swept nor named here
            at its default value.

        Raises
        ------
        KeyError
            If a name is not a parameter of the system.
        ValueError
            If a swept name repeats or also has a fixed value.
        """
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

        This method fixes every parameter that is not swept at its new
        default value.

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
    def parameters(self) -> ValuedSet:
        """Every parameter with its default value."""
        return self._parameters

    @property
    def compiled_parameters(self) -> ParameterSet:
        """The parameter set of the currently compiled system."""
        return self._specialiser.compiled_parameters

    @property
    def swept_set(self) -> ValuedSet:
        """The swept parameters with their default values, in row order."""
        return self._parameters.subset(self.compiled_parameters.swept)

    @property
    def states(self) -> ValuedSet:
        """Every state with its default initial value, or ``0.0`` for a
        state given none."""
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
        """The variables in each device buffer, in buffer order."""
        return self._specialiser.layout

    @property
    def observables(self) -> VariableSet:
        """The system's observables."""
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
    def sizes(self) -> SystemSizes:
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
