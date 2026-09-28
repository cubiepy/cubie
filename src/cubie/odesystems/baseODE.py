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
    Set,
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
from cubie.odesystems.SystemValues import SystemValues
from cubie.batchsolving.solver_settings import clashing_names


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
        precision: PrecisionDType = float32,
        initial_values: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, float]] = None,
        observables: Optional[Dict[str, float]] = None,
        default_initial_values: Optional[Dict[str, float]] = None,
        default_parameters: Optional[Dict[str, float]] = None,
        default_observable_names: Optional[Dict[str, float]] = None,
        num_drivers: int = 1,
        operation_ordering: str = operation_ordering_default(),
        name: Optional[str] = None,
        swept_parameters: Iterable[str] = (),
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
            Number of driver or forcing functions. Defaults to ``1``.
        operation_ordering
            Generated-operation ordering policy:
            ``"liveness_auto"``, ``"kahn"``, ``"greedy"``, or
            ``"dfs"``. Defaults to ``CUBIE_OPERATION_ORDERING``
            (``liveness_auto`` when unset).
        name
            Printable identifier for the system. Defaults to ``None``.
        swept_parameters
            Parameters read per run, in row order; the rest compile in.
        """
        super().__init__()
        clashes = clashing_names(
            {**(default_parameters or {}), **(parameters or {})}
        )
        if clashes:
            raise ValueError(
                f"Parameters {sorted(clashes)} share names with Solver "
                "settings and could not be given by name; rename them."
            )
        system_data = ODEData.from_BaseODE_initargs(
            initial_values=initial_values,
            parameters=parameters,
            observables=observables,
            default_initial_values=default_initial_values,
            default_parameters=default_parameters,
            default_observable_names=default_observable_names,
            precision=precision,
            num_drivers=num_drivers,
            operation_ordering=operation_ordering,
            swept_parameters=swept_parameters,
        )
        self.setup_compile_settings(system_data)
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

        Notes
        -----
        Bring compile settings into local (outer) scope before defining
        ``dxdt`` because CUDA device functions cannot reference ``self``.
        """
        # return ODECache(dxdt=dxdt)

    def _update(self, updates: Dict[str, Any], silent: bool) -> Set[str]:
        """Apply compile settings, then parameter values.

        Parameters
        ----------
        updates
            Setting names or parameter names to new values.
        silent
            Whether :meth:`update` ignores unrecognised names.

        Returns
        -------
        set[str]
            Names the settings and parameters recognised.
        """
        names = set(self.parameters.names)
        values = {
            key: value for key, value in updates.items() if key in names
        }
        settings = {
            key: value
            for key, value in updates.items()
            if key not in names and key != "swept_parameters"
        }
        recognised = self.update_compile_settings(settings, silent=True)
        if values:
            recognised |= self.set_parameter_values(values)
        return recognised

    def set_parameter_values(
        self, values: Mapping[str, float]
    ) -> Set[str]:
        """Set parameter values; each named parameter is compiled in.

        Parameters
        ----------
        values
            Parameter names to new values.

        Returns
        -------
        set of str
            Names that were recognised.

        Raises
        ------
        KeyError
            If a name is not a parameter of the system.
        """
        self.bind(values=values)
        return set(values)

    def set_swept_parameters(self, names: Iterable[str]) -> bool:
        """Sweep ``names`` in the given row order; fix the rest.

        Parameters
        ----------
        names
            Parameter names in parameter-table row order.

        Returns
        -------
        bool
            Whether the system changed.

        Raises
        ------
        KeyError
            If a name is not a parameter of the system.
        """
        return self.bind(swept=names)

    def set_initial_values(self, values: Mapping[str, float]) -> None:
        """Set the stored initial values of the named states."""
        self.initial_values.update_from_dict(values)

    def bind(
        self,
        swept: Optional[Iterable[str]] = None,
        values: Optional[Mapping[str, float]] = None,
    ) -> bool:
        """Set parameter values and swept names, re-specialising once.

        Parameters
        ----------
        swept
            Parameters read per run, in row order. ``None`` keeps the
            current names, less any given in ``values``.
        values
            Parameter names to new values. A named parameter is
            compiled in unless ``swept`` lists it.

        Returns
        -------
        bool
            Whether the system changed.

        Raises
        ------
        KeyError
            If a name is not a parameter of the system.
        """
        values = {
            str(name): float(value)
            for name, value in (values or {}).items()
        }
        stored = self.compile_settings.parameter_values
        if swept is None:
            swept = tuple(
                name for name in self.swept_parameters if name not in values
            )
        else:
            swept = tuple(str(name) for name in swept)
        unknown = (set(values) | set(swept)) - set(stored)
        if unknown:
            raise KeyError(
                f"{sorted(unknown)} are not parameters of this system."
            )
        new_values = {
            name: value
            for name, value in values.items()
            if stored[name] != value
        }
        if swept == self.swept_parameters and not new_values:
            return False
        parameters = self.parameters
        if new_values:
            parameters = parameters.copy()
            parameters.update_from_dict(new_values)
        self._respecialise(swept, parameters)
        return True

    def _respecialise(
        self, swept: Tuple[str, ...], parameters: "SystemValues"
    ) -> None:
        """Store the swept names and parameter values."""
        self.update_compile_settings(
            swept_parameters=swept, parameters=parameters, silent=True
        )

    @property
    def swept_parameters(self) -> Tuple[str, ...]:
        """Parameter names read per run, in parameter-table row order."""
        return self.compile_settings.swept_parameters

    @property
    def fixed_parameter_values(self) -> Dict[str, float]:
        """Values of the parameters compiled into generated code."""
        return self.compile_settings.fixed_parameter_values

    @property
    def parameters(self) -> "SystemValues":
        """Parameter values configured for the system."""
        return self.compile_settings.parameters

    @property
    def states(self) -> "SystemValues":
        """Initial state values configured for the system."""
        return self.compile_settings.initial_states

    @property
    def initial_values(self) -> "SystemValues":
        """Alias for :attr:`states`."""
        return self.compile_settings.initial_states

    @property
    def observables(self) -> "SystemValues":
        """Observable definitions configured for the system."""
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
        """Number of driver variables."""
        return self.compile_settings.num_drivers

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
