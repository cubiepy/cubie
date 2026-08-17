"""Symbolic ODE system built from :mod:`sympy` expressions.

Published Classes
-----------------
:class:`SymbolicODE`
    Concrete :class:`~cubie.odesystems.baseODE.BaseODE` subclass that
    generates CUDA device functions from SymPy equations. Handles
    codegen caching, solver helper generation, and constant/parameter
    conversion.

    >>> from cubie.odesystems.symbolic.symbolicODE import (
    ...     create_ODE_system,
    ... )
    >>> ode = create_ODE_system(
    ...     dxdt="dx = -k * x",
    ...     states={"x": 1.0},
    ...     parameters={"k": 0.5},
    ... )
    >>> ode.num_states
    1

Published Functions
-------------------
:func:`create_ODE_system`
    Convenience wrapper around :meth:`SymbolicODE.create`.

    >>> ode = create_ODE_system("dx = -x", states={"x": 1.0})
    >>> ode.num_states
    1

See Also
--------
:class:`~cubie.odesystems.baseODE.BaseODE`
    Abstract parent providing cache management and value containers.
:class:`~cubie.odesystems.symbolic.odefile.ODEFile`
    Disk-backed cache for generated factory functions.
:mod:`cubie.odesystems.symbolic.parsing.parser`
    Parses string or SymPy equations into structured components.
:mod:`cubie.odesystems.symbolic.codegen`
    Code generation modules invoked by :meth:`SymbolicODE.get_solver_helper`.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Set,
    Union,
)

from numpy import asarray, dtype as np_dtype, float32
import sympy as sp
from cubie.array_interpolator import ArrayInterpolator
from cubie.odesystems.symbolic.codegen.dxdt import (
    generate_dxdt_fac_code,
    generate_observables_fac_code,
)
from cubie.odesystems.symbolic.codegen.jacobian import generate_analytical_jvp
from cubie.odesystems.symbolic.codegen.neumann_convergence import (
    NeumannRHSEvaluator,
)
from cubie.odesystems.symbolic.helper_registry import (
    SOLVER_HELPER_REGISTRY,
    helper_member_hash,
    helper_source_hash,
)
from cubie.odesystems.symbolic.odefile import ODEFile
from cubie.odesystems.symbolic.parsing import (
    IndexedBases,
    JVPEquations,
    ParsedEquations,
    parse_input,
)
from cubie.odesystems.symbolic.parsing.parsed_system import ParsedSystem
from cubie.odesystems.symbolic.sym_utils import hash_system_definition
from cubie.odesystems.baseODE import BaseODE, ODECache
from cubie.odesystems.SystemValues import SystemValues
from cubie.odesystems.solver_helpers import (
    HelperResult,
    SolverHelperRequest,
)
from cubie._serialize import canonical_digest
from cubie._env import operation_ordering_default
from cubie._utils import PrecisionDType, is_devfunc
from cubie.cubie_cache import CachePolicy
from cubie.time_logger import default_timelogger

def _system_source_hash(equations, index_map) -> str:
    """Return the source hash for equations and their array layout."""

    return hash_system_definition(
        equations,
        index_map.constants.default_values,
        state_labels=index_map.state_names,
        dxdt_labels=index_map.dxdt_names,
        parameter_labels=index_map.parameter_names,
        driver_labels=index_map.driver_names,
        observable_labels=index_map.observable_names,
        derivative_names=equations.derivative_names,
        function_aliases=equations.function_aliases,
    )


def _operation_source_hash(fn_hash: str, operation_ordering: str) -> str:
    """Return generated-source identity for one ordering policy."""

    return canonical_digest(
        ("cubie-ode-source", fn_hash, operation_ordering)
    )


def create_ODE_system(
    dxdt: Union[str, Iterable[str], Callable],
    precision: PrecisionDType = float32,
    states: Optional[Union[dict[str, float], Iterable[str]]] = None,
    observables: Optional[Iterable[str]] = None,
    parameters: Optional[Union[dict[str, float], Iterable[str]]] = None,
    constants: Optional[Union[dict[str, float], Iterable[str]]] = None,
    drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
    user_functions: Optional[dict[str, Callable]] = None,
    user_function_derivatives: Optional[dict[str, Callable]] = None,
    name: Optional[str] = None,
    strict: bool = False,
    state_priority: Optional[dict[str, float]] = None,
    irreducible: Optional[Iterable[str]] = None,
    simplify_options: Optional[dict[str, Any]] = None,
    operation_ordering: str = operation_ordering_default(),
) -> "SymbolicODE":
    """Create a :class:`SymbolicODE` from SymPy definitions.

    Parameters
    ----------
    dxdt
        System equations defined as a single string, an iterable of equation
        strings in ``lhs = rhs`` form, or a Python callable. When a callable
        is provided its signature must be ``f(t, y, ...)`` where ``t`` is
        time, ``y`` is the state vector, and additional arguments map to
        parameters or constants. State access patterns supported:
        ``y[0]`` (positional), ``y["name"]`` (string), ``y.name``
        (attribute). The return value must be a list, tuple, or dict of
        derivative expressions.
    states
        State labels either as an iterable or as a mapping to default initial
        values.
    observables
        Observable variable labels to expose from the generated system.
    parameters
        Parameter labels either as an iterable or as a mapping to default
        values.
    constants
        Constant labels either as an iterable or as a mapping to default
        values.
    drivers
        External driver variable labels required at runtime. Accepts either
        an iterable of driver symbol names or a dictionary mapping driver
        names to default values or driver-array samples and configuration
        entries.
    user_functions
        Custom callables referenced within ``dxdt`` expressions.
    user_function_derivatives
        Mapping of user-function names to callables evaluating their
        analytic derivatives, used when generating Jacobian-based
        solver helpers.
    name
        Identifier used for generated files. Defaults to the hash of the system
        definition.
    precision
        Target floating-point precision used when compiling the system.
    strict
        When ``True`` require every symbol to be explicitly categorised.
    state_priority
        Per-unknown state-selection priorities (higher values are
        preferred as solver states).
    irreducible
        Unknowns that must not be eliminated.
    simplify_options
        Extra keyword arguments forwarded to
        :func:`~cubie.odesystems.symbolic.structural.simplify.structural_simplify`.
    operation_ordering
        Generated-operation ordering policy. ``"liveness_auto"``
        applies thresholded liveness-based selection; ``"kahn"``
        preserves stable breadth-first ordering, and ``"greedy"``
        and ``"dfs"`` select fixed alternatives. Defaults to
        ``CUBIE_OPERATION_ORDERING`` (``liveness_auto`` when unset).

    Returns
    -------
    SymbolicODE
        Fully constructed symbolic system ready for compilation.
    """
    symbolic_ode = SymbolicODE.create(
        dxdt=dxdt,
        states=states,
        observables=observables,
        parameters=parameters,
        constants=constants,
        drivers=drivers,
        user_functions=user_functions,
        user_function_derivatives=user_function_derivatives,
        name=name,
        precision=precision,
        strict=strict,
        state_priority=state_priority,
        irreducible=irreducible,
        simplify_options=simplify_options,
        operation_ordering=operation_ordering,
    )
    return symbolic_ode


class SymbolicODE(BaseODE):
    """Symbolic representation of an ODE system.

    Parameters are provided as SymPy symbols and the differential equations are
    supplied as ``(lhs, rhs)`` tuples where the left-hand side is a derivative
    or observable symbol. Right-hand sides combine states, parameters,
    constants, and intermediate observables.

    Parameters
    ----------
    equations
        Parsed equations describing the system dynamics.
    all_indexed_bases
        Indexed base collections providing access to state, parameter,
        constant, and observable metadata.
    all_symbols
        Mapping from symbol names to their :class:`sympy.Symbol` instances.
    precision
        Target floating-point precision used for generated kernels.
    fn_hash
        Precomputed system hash. When omitted it is derived from the equations
        and constants.
    user_functions
        Runtime callables referenced within the symbolic expressions.
    name
        Identifier used for generated modules.
    """

    # Diagnostic evaluators live in a dict so child-factory
    # discovery skips them and config_hash is unaffected.

    def __init__(
        self,
        equations: ParsedEquations,
        precision: PrecisionDType,
        all_indexed_bases: IndexedBases,
        all_symbols: Optional[dict[str, sp.Symbol]] = None,
        fn_hash: Optional[str] = None,
        user_functions: Optional[dict[str, Callable]] = None,
        name: Optional[str] = None,
        operation_ordering: str = operation_ordering_default(),
        parsed_system: Optional[ParsedSystem] = None,
    ):
        """Initialise the symbolic system instance.

        Parameters
        ----------
        equations
            Parsed equations describing the system dynamics; the
            solver mass matrix rides on ``equations.mass_matrix``.
        all_indexed_bases
            Indexed base collections providing access to state, parameter,
            constant, and observable metadata.
        all_symbols
            Mapping from symbol names to their :class:`sympy.Symbol` instances.
        precision
            Target floating-point precision used for generated kernels.
        fn_hash
            Precomputed system hash. When omitted it is derived from the
            equations and constants.
        user_functions
            Runtime callables referenced within the symbolic expressions.
        name
            Identifier used for generated modules.
        operation_ordering
            Generated-operation ordering policy.
        parsed_system
            Constants-symbolic checkpoint from the parser; rebuilt
            from ``equations`` and re-specialised when omitted.

        Notes
        -----
        The mass matrix is never an input: structural simplification
        derives it (``None`` for solved systems, a 0/1 diagonal for
        torn ones), it rides on ``equations.mass_matrix``, and it
        reaches compile settings through :meth:`_seed_derived_mass`.
        """
        if all_symbols is None:
            all_symbols = all_indexed_bases.all_symbols
        self.all_symbols = all_symbols

        if parsed_system is None:
            parsed_system = ParsedSystem.from_parsed_equations(
                equations,
                all_indexed_bases,
                user_functions=user_functions,
            )
            (
                all_indexed_bases,
                self.all_symbols,
                user_functions,
                equations,
                fn_hash,
            ) = parsed_system.specialise()
        self._parsed_system = parsed_system

        derived_mass_matrix = equations.mass_matrix

        if fn_hash is None:
            fn_hash = _system_source_hash(equations, all_indexed_bases)
        if name is None:
            name = fn_hash

        self.name = name

        ndriv = all_indexed_bases.drivers.length
        self.equations = equations
        self.indices = all_indexed_bases
        self.fn_hash = fn_hash
        self.user_functions = user_functions
        self.driver_defaults = all_indexed_bases.drivers.default_values
        self.registered_helper_events = set()

        super().__init__(
            initial_values=all_indexed_bases.state_values,
            parameters=all_indexed_bases.parameter_values,
            constants=all_indexed_bases.constant_values,
            observables=all_indexed_bases.observable_names,
            precision=precision,
            num_drivers=ndriv,
            name=name,
            operation_ordering=operation_ordering,
        )
        self._seed_derived_mass(derived_mass_matrix)
        self.gen_file = ODEFile(
            name,
            _operation_source_hash(
                fn_hash,
                self.compile_settings.operation_ordering,
            ),
        )
        self._jvp_exprs: Optional[JVPEquations] = None
        self._jvp_exprs_key = None

        system_name = name
        if system_name == fn_hash:
            system_name = f"unnamed_{fn_hash[:8]}"
        self._diagnostic_system_name = system_name
        self._neumann_diagnostics = {}

    def _seed_derived_mass(self, mass_matrix) -> None:
        """Seed compile settings with the simplification-derived mass.

        Parameters
        ----------
        mass_matrix
            ``None`` for solved systems, or the 0/1 diagonal from
            structural simplification (nested lists or an array).
        """
        if mass_matrix is None:
            return
        self.update_compile_settings(
            {"mass": asarray(mass_matrix, dtype=self.precision)},
            silent=True,
        )

    @classmethod
    def create(
        cls,
        dxdt: Union[str, Iterable[str], Callable],
        precision: PrecisionDType,
        states: Optional[Union[dict[str, float], Iterable[str]]] = None,
        observables: Optional[Iterable[str]] = None,
        parameters: Optional[Union[dict[str, float], Iterable[str]]] = None,
        constants: Optional[Union[dict[str, float], Iterable[str]]] = None,
        drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
        user_functions: Optional[dict[str, Callable]] = None,
        user_function_derivatives: Optional[dict[str, Callable]] = None,
        name: Optional[str] = None,
        strict: bool = False,
        state_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
        parameter_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
        constant_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
        observable_units: Optional[
            Union[dict[str, str], Iterable[str]]
        ] = None,
        driver_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
        state_priority: Optional[dict[str, float]] = None,
        irreducible: Optional[Iterable[str]] = None,
        simplify_options: Optional[dict[str, Any]] = None,
        operation_ordering: str = operation_ordering_default(),
    ) -> "SymbolicODE":
        """Parse user inputs and instantiate a :class:`SymbolicODE`.

        Parameters
        ----------
        dxdt
            System equations defined as a single string, an iterable of
            equation strings in ``lhs = rhs`` form, or a Python callable
            with a ``(t, y, ...)`` signature.
        states
            State labels either as an iterable or as a mapping to default
            initial values.
        observables
            Observable variable labels to expose from the generated system.
        parameters
            Parameter labels either as an iterable or as a mapping to default
            values.
        constants
            Constant labels either as an iterable or as a mapping to default
            values.
        drivers
            External driver variable labels required at runtime. May be an
            iterable of driver labels or a dictionary describing driver
            defaults or driver-array samples alongside configuration entries.
        user_functions
            Custom callables referenced within ``dxdt`` expressions.
        user_function_derivatives
            Mapping of user-function names to callables evaluating
            their analytic derivatives, used when generating
            Jacobian-based solver helpers.
        name
            Identifier used for generated files. Defaults to the hash of the
            system definition.
        precision
            Target floating-point precision used when compiling the system.
        strict
            When ``True`` require every symbol to be explicitly categorised.
        state_units
            Optional units for states. Defaults to "dimensionless".
        parameter_units
            Optional units for parameters. Defaults to "dimensionless".
        constant_units
            Optional units for constants. Defaults to "dimensionless".
        observable_units
            Optional units for observables. Defaults to "dimensionless".
        driver_units
            Optional units for drivers. Defaults to "dimensionless".
        state_priority
            Per-unknown state-selection priorities (higher values are
            preferred as solver states).
        irreducible
            Unknowns that must not be eliminated.
        simplify_options
            Extra keyword arguments forwarded to
            :func:`~cubie.odesystems.symbolic.structural.simplify.structural_simplify`.
        operation_ordering
            Generated-operation ordering policy:
            ``"liveness_auto"``, ``"kahn"``, ``"greedy"``, or
            ``"dfs"``. Defaults to ``CUBIE_OPERATION_ORDERING``.

        Returns
        -------
        SymbolicODE
            Fully constructed symbolic system ready for compilation.
        """

        if isinstance(drivers, dict) and (
            "time" in drivers or "driver_sample_period" in drivers
        ):
            ArrayInterpolator(precision=precision, input_dict=drivers)

        # Register timing event for parsing (one-time registration)
        default_timelogger.register_event(
            "symbolic_ode_parsing",
            "codegen",
            "Codegen time for symbolic ODE parsing",
        )

        # Start timing for parsing operation
        default_timelogger.start_event("symbolic_ode_parsing")
        (
            index_map,
            all_symbols,
            functions,
            equations,
            fn_hash,
            parsed_system,
        ) = parse_input(
            dxdt=dxdt,
            states=states,
            observables=observables,
            parameters=parameters,
            constants=constants,
            drivers=drivers,
            user_functions=user_functions,
            user_function_derivatives=user_function_derivatives,
            strict=strict,
            state_units=state_units,
            parameter_units=parameter_units,
            constant_units=constant_units,
            observable_units=observable_units,
            driver_units=driver_units,
            state_priority=state_priority,
            irreducible=irreducible,
            simplify_options=simplify_options,
        )
        symbolic_ode = cls(
            equations=equations,
            all_indexed_bases=index_map,
            all_symbols=all_symbols,
            name=name,
            fn_hash=fn_hash,
            user_functions=functions,
            precision=precision,
            operation_ordering=operation_ordering,
            parsed_system=parsed_system,
        )
        default_timelogger.stop_event("symbolic_ode_parsing")
        return symbolic_ode

    @property
    def state_units(self) -> dict[str, str]:
        """Return units for state variables."""
        return self.indices.states.units

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return units for parameters."""
        return self.indices.parameters.units

    @property
    def constant_units(self) -> dict[str, str]:
        """Return units for constants."""
        return self.indices.constants.units

    @property
    def observable_units(self) -> dict[str, str]:
        """Return units for observables."""
        return self.indices.observables.units

    @property
    def driver_units(self) -> dict[str, str]:
        """Return units for drivers."""
        return self.indices.drivers.units

    def _get_jvp_exprs(self) -> JVPEquations:
        """Return Jacobian-vector assignments for the current system.

        The cache keys on the system hash and ordering policy, so a
        re-specialisation or ordering change recomputes on next use.
        """

        key = (self.fn_hash, self.compile_settings.operation_ordering)
        if self._jvp_exprs is None or self._jvp_exprs_key != key:
            self._jvp_exprs = generate_analytical_jvp(
                self.equations,
                input_order=self.indices.states.index_map,
                output_order=self.indices.dxdt.index_map,
                observables=self.indices.observable_symbols,
                cse=True,
                operation_ordering=(
                    self.compile_settings.operation_ordering
                ),
            )
            self._jvp_exprs_key = key
        return self._jvp_exprs

    def update(
        self,
        updates_dict: Optional[Dict[str, float]] = None,
        silent: bool = False,
        **kwargs: float,
    ) -> Set[str]:
        """Update system settings, forwarding to diagnostic factories.

        Updates are offered to the system's own settings and to the
        compile settings of every existing convergence evaluator.

        Parameters
        ----------
        updates_dict
            Dictionary of updates to apply.
        silent
            Set to ``True`` to suppress errors about unrecognized keys.
        **kwargs
            Additional updates specified as keyword arguments.

        Returns
        -------
        set of str
            Labels that were recognized and updated.
        """
        if updates_dict is None:
            updates_dict = {}
        updates = updates_dict.copy()
        if kwargs:
            updates.update(kwargs)
        if updates == {}:
            return set()

        recognised = super().update(updates, silent=True)
        for evaluator in self._neumann_diagnostics.values():
            recognised |= evaluator.update_compile_settings(
                updates, silent=True
            )

        if not silent:
            unrecognised = set(updates.keys()) - recognised
            if unrecognised:
                raise KeyError(
                    "Unrecognized parameters in update: "
                    f"{unrecognised}. These parameters were not updated.",
                )
        return recognised

    def _get_neumann_evaluator(
        self, cache_policy: CachePolicy
    ) -> NeumannRHSEvaluator:
        """Return the convergence evaluator for a consumer's policy.

        Each policy gets its own evaluator, created on its first
        request, kept in a plain dict so the evaluators stay out of
        child-factory discovery and ``config_hash``.
        ``settings_and_constants_hash`` stands in for ``config_hash``,
        which would self-reference if the evaluator's configuration
        fed back into it.

        Parameters
        ----------
        cache_policy
            Cache policy of the requesting consumer.
        """
        if cache_policy not in self._neumann_diagnostics:
            self._neumann_diagnostics[cache_policy] = NeumannRHSEvaluator(
                precision=self.precision,
                cache_policy=cache_policy,
                system_name=self._diagnostic_system_name,
            )
        evaluator = self._neumann_diagnostics[cache_policy]
        evaluator.update_compile_settings(
            {
                "dxdt_function": self.evaluate_f,
                "dxdt_settings_hash": self.settings_and_constants_hash,
                "precision": self.precision,
                "jit_flags": self.compile_settings.jit_flags,
                "system_hash": self.fn_hash,
            },
            silent=True,
        )
        return evaluator

    def _device_function_injections(self) -> dict[str, Callable]:
        """Collect device callables the generated module must resolve.

        Generated factories call user device functions (and their
        derivative helpers) by name, but the generated module is
        imported standalone, so those callables are injected as module
        attributes before the factory is compiled.

        Returns
        -------
        dict[str, Callable]
            Mapping from printed function name to device callable.
        """
        injections = {}
        for name, func in (self.user_functions or {}).items():
            if is_devfunc(func):
                injections[name] = func
        all_symbols = self.all_symbols or {}
        for name, obj in all_symbols.items():
            if name == "__function_aliases__":
                continue
            if is_devfunc(obj):
                injections[name] = obj
        # The string parser renames user functions (trailing
        # underscore), and generated source prints the renamed symbol,
        # so each device callable is injected under its alias too.
        aliases = all_symbols.get("__function_aliases__", {}) or {}
        for sym_name, orig_name in aliases.items():
            func = (self.user_functions or {}).get(orig_name)
            if func is not None and is_devfunc(func):
                injections[sym_name] = func
        return injections

    def build(self) -> ODECache:
        """Compile the ``dxdt`` factory and refresh the cache.

        Returns
        -------
        ODECache
            Cache populated with the compiled ``dxdt`` callable.
        """
        numba_precision = self.numba_precision
        constants = self.constants.values_dict
        lineinfo = self.compile_settings.lineinfo
        new_hash = _system_source_hash(self.equations, self.indices)
        source_hash = _operation_source_hash(
            new_hash,
            self.compile_settings.operation_ordering,
        )
        if new_hash != self.fn_hash or self.gen_file.fn_hash != source_hash:
            self.gen_file = ODEFile(self.name, source_hash)
            self.fn_hash = new_hash

        dxdt_code = None
        if not self.gen_file.function_is_cached("dxdt_factory"):
            dxdt_code = generate_dxdt_fac_code(
                self.equations,
                self.indices,
                "dxdt_factory",
                operation_ordering=(
                    self.compile_settings.operation_ordering
                ),
            )
        dxdt_factory, _ = self.gen_file.import_function(
            "dxdt_factory",
            dxdt_code,
            injections=self._device_function_injections(),
        )
        dxdt_func = dxdt_factory(
            constants,
            numba_precision,
            lineinfo=lineinfo,
        )

        obs_code = None
        if not self.gen_file.function_is_cached("observables_factory"):
            obs_code = generate_observables_fac_code(
                self.equations, self.indices,
                func_name="observables_factory",
                operation_ordering=(
                    self.compile_settings.operation_ordering
                ),
            )
        observables_factory, _ = self.gen_file.import_function(
            "observables_factory",
            obs_code,
            injections=self._device_function_injections(),
        )
        evaluate_observables = observables_factory(
            constants,
            numba_precision,
            lineinfo=lineinfo,
        )

        return ODECache(
            dxdt=dxdt_func,
            observables=evaluate_observables,
        )

    def _specialise(
        self,
        constant_values: dict[str, float],
        parsed_system: ParsedSystem,
    ) -> None:
        """Derive the system's products for one set of constant values.

        Specialises the checkpoint, swaps in the derived equations,
        layouts, and hash, and pushes the changed compile settings
        in one call. ``parsed_system`` replaces the stored
        checkpoint on success; nothing mutates on a raise.

        Parameters
        ----------
        constant_values
            Complete mapping of constant names to their new values.
        parsed_system
            Checkpoint to specialise from.
        """

        precision = self.precision
        settings = self.compile_settings
        current_params = settings.parameter_values
        (
            index_map,
            all_symbols,
            funcs,
            parsed,
            fn_hash,
        ) = parsed_system.specialise(
            constant_values,
            state_values=settings.initial_state_values,
        )
        # Runtime parameter values survive re-specialisation.
        index_map.parameters.update_values(current_params)

        self._parsed_system = parsed_system
        self.equations = parsed
        self.indices = index_map
        self.all_symbols = all_symbols
        self.user_functions = funcs
        self.fn_hash = fn_hash
        self.driver_defaults = index_map.drivers.default_values

        updates: dict[str, Any] = {
            "constants": SystemValues(
                index_map.constant_values,
                precision,
                name="Constants",
            )
        }
        if index_map.state_names != settings.initial_states.names:
            updates["initial_states"] = SystemValues(
                index_map.state_values, precision, name="States"
            )
        if index_map.parameter_names != settings.parameters.names:
            updates["parameters"] = SystemValues(
                index_map.parameter_values,
                precision,
                name="Parameters",
            )
        if index_map.observable_names != settings.observables.names:
            updates["observables"] = SystemValues(
                index_map.observable_names,
                precision,
                name="Observables",
            )
        mass = parsed.mass_matrix
        if mass is not None:
            mass = asarray(mass, dtype=precision)
        updates["mass"] = mass
        self.update_compile_settings(updates, silent=True)

    def set_constants(
        self,
        updates_dict: Optional[dict[str, float]] = None,
        silent: bool = False,
        **kwargs: float,
    ) -> Set[str]:
        """Update constant values, re-specialising the system.

        Parameters
        ----------
        updates_dict
            Mapping from constant names to replacement values.
        silent
            When ``True`` suppress warnings for unknown labels.
        **kwargs
            Additional constant overrides supplied as keyword arguments.

        Returns
        -------
        set[str]
            Labels that were recognised and updated.

        Notes
        -----
        A value change re-runs constant specialisation, so system
        structure follows the new values.
        """
        updates = dict(updates_dict or {})
        updates.update(kwargs)
        if not updates:
            return set()

        # An update that rounds to the stored value is not a change.
        precision = self.precision
        current = self.compile_settings.constant_values
        recognised = set(updates) & set(current)
        unrecognised = set(updates) - recognised
        changed = {
            label
            for label in recognised
            if float(precision(updates[label])) != current[label]
        }
        if changed:
            new_values = dict(current)
            new_values.update(
                {label: float(updates[label]) for label in recognised}
            )
            self._specialise(new_values, self._parsed_system)

        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return recognised

    def make_parameter(self, name: str) -> None:
        """Convert a constant to a swept parameter.

        The symbol returns to the equations in place of the folded
        literal; the current value becomes the parameter's default.

        Parameters
        ----------
        name
            Name of the constant to convert.

        Raises
        ------
        KeyError
            If the name is not found in constants.
        """
        current = dict(self.compile_settings.constant_values)
        if name not in current:
            raise KeyError(
                f"{name} is not a constant of this system."
            )
        value = current.pop(name)
        self._specialise(
            current,
            self._parsed_system.constant_to_parameter(name, value),
        )

    def make_constant(self, name: str) -> None:
        """Convert a parameter to a compile-time constant.

        The parameter's value folds into the source as a literal.

        Parameters
        ----------
        name
            Name of the parameter to convert.

        Raises
        ------
        KeyError
            If the name is not found in parameters.
        """
        parameter_values = self.compile_settings.parameter_values
        if name not in parameter_values:
            raise KeyError(
                f"{name} is not a parameter of this system."
            )
        value = parameter_values[name]
        current = dict(self.compile_settings.constant_values)
        current[name] = value
        self._specialise(
            current,
            self._parsed_system.parameter_to_constant(name, value),
        )

    def set_constant_value(self, name: str, value: float) -> None:
        """Set the value of a constant.

        Parameters
        ----------
        name
            Name of the constant.
        value
            New value for the constant.

        Raises
        ------
        KeyError
            If the name is not found in constants.
        """
        self.set_constants({name: value})

    def set_parameter_value(self, name: str, value: float) -> None:
        """Set the default value of a parameter.

        Parameters
        ----------
        name
            Name of the parameter.
        value
            New default value for the parameter.

        Raises
        ------
        KeyError
            If the name is not found in parameters.
        """
        self.parameters[name] = value
        self.indices.parameters.update_values({name: value})

    def set_initial_value(self, name: str, value: float) -> None:
        """Set the initial value of a state variable.

        Parameters
        ----------
        name
            Name of the state variable.
        value
            New initial value.

        Raises
        ------
        KeyError
            If the name is not found in states.
        """
        self.initial_values[name] = value
        self.indices.states.update_values({name: value})

    def get_constants_info(self) -> list[dict]:
        """Return information about all constants.

        Returns
        -------
        list of dict
            Each dict contains 'name', 'value', and 'unit' keys.
        """
        result = []
        for name in self.indices.constant_names:
            result.append({
                'name': name,
                'value': self.constants.values_dict.get(name, 0.0),
                'unit': self.constant_units.get(name, 'dimensionless'),
            })
        return result

    def get_parameters_info(self) -> list[dict]:
        """Return information about all parameters.

        Returns
        -------
        list of dict
            Each dict contains 'name', 'value', and 'unit' keys.
        """
        result = []
        for name in self.indices.parameter_names:
            result.append({
                'name': name,
                'value': self.parameters.values_dict.get(name, 0.0),
                'unit': self.parameter_units.get(name, 'dimensionless'),
            })
        return result

    def get_states_info(self) -> list[dict]:
        """Return information about all state variables.

        Returns
        -------
        list of dict
            Each dict contains 'name', 'value', and 'unit' keys.
        """
        result = []
        for name in self.indices.state_names:
            result.append({
                'name': name,
                'value': self.initial_values.values_dict.get(name, 0.0),
                'unit': self.state_units.get(name, 'dimensionless'),
            })
        return result

    def constants_gui(self, blocking: bool = True) -> None:
        # no cover: start
        """Launch a Qt GUI for editing constants and parameters.

        The GUI displays all constants and parameters with their values and
        units. Users can convert between constants and parameters using a
        checkbox, and edit values directly.

        Parameters
        ----------
        blocking
            If True (default), block until the dialog is closed.
            If False, return immediately with the dialog still open.

        Notes
        -----
        Requires a Qt binding (PyQt6, PyQt5, PySide6, or PySide2).

        Example
        -------
        >>> ode = load_cellml_model("model.cellml")
        >>> ode.constants_gui()  # Opens editor dialog
        """
        from cubie.gui.constants_editor import show_constants_editor
        show_constants_editor(self, blocking=blocking)
        # no cover: end

    def states_gui(self, blocking: bool = True) -> None:
        # no cover: start
        """Launch a Qt GUI for editing initial state values.

        The GUI displays all state variables with their initial values and
        units. Users can edit the initial values directly.

        Parameters
        ----------
        blocking
            If True (default), block until the dialog is closed.
            If False, return immediately with the dialog still open.

        Notes
        -----
        Requires a Qt binding (PyQt6, PyQt5, PySide6, or PySide2).

        Example
        -------
        >>> ode = load_cellml_model("model.cellml")
        >>> ode.states_gui()  # Opens editor dialog
        """
        from cubie.gui.states_editor import show_states_editor
        show_states_editor(self, blocking=blocking)
        # no cover: end

    def get_solver_helper(
        self,
        request: SolverHelperRequest,
        cache_policy: Optional[CachePolicy] = None,
    ) -> HelperResult:
        """Return the bound helper member for ``request``.

        Parameters
        ----------
        request
            Immutable description of the requested helper.
        cache_policy
            The requesting consumer's cache policy, passed through
            to diagnostic services run on its behalf. ``None``
            selects the default policy.

        Returns
        -------
        HelperResult
            The bound device callable and its typed metadata. For
            ``prepare_jac`` the result carries
            ``cached_auxiliary_count``.

        Notes
        -----
        Helpers that consume a mass matrix read the system's own
        ``compile_settings.mass``. An exact repeated request returns
        the same member object; different bindings that share emitted
        source reuse one generated factory.
        """
        if cache_policy is None:
            cache_policy = CachePolicy()
        entry = SOLVER_HELPER_REGISTRY[request.kind]
        kind_name = request.kind.value

        event_name = f"solver_helper_{kind_name}"
        if event_name not in self.registered_helper_events:
            default_timelogger.register_event(
                event_name,
                "codegen",
                f"Codegen time for solver helper {kind_name}",
            )
            self.registered_helper_events.add(event_name)

        # Validation hooks (the Neumann convergence diagnostic) run on
        # every request so warnings surface for reused code too.
        if entry.validation_hook is not None:
            entry.validation_hook(self, request, cache_policy)

        helpers = self.get_cached_output("helpers")

        # The generated function's name contains the full source hash.
        source_hash = helper_source_hash(self, request)
        factory_name = f"{kind_name}_s{source_hash}"

        if source_hash not in helpers.factories:
            is_cached = self.gen_file.function_is_cached(factory_name)
            default_timelogger.start_event(event_name, skipped=is_cached)
            code = None
            if not is_cached:
                generated = entry.generate(self, request, factory_name)
                if entry.returns_aux_count:
                    code, _ = generated
                else:
                    code = generated
            factory, _ = self.gen_file.import_function(
                factory_name,
                code,
                injections=self._device_function_injections(),
            )
            default_timelogger.stop_event(event_name)
            helpers.factories[source_hash] = factory
        factory = helpers.factories[source_hash]

        config = self.compile_settings
        precision = config.precision
        constants = self.constants.values_dict
        available_args = {
            "constants": constants,
            "precision": self.numba_precision,
            "beta": precision(request.beta),
            "gamma": precision(request.gamma),
            "order": request.preconditioner_order,
            "lineinfo": config.lineinfo,
        }
        canonical_by_name = {
            "constants": tuple(
                (label, float(value))
                for label, value in sorted(constants.items())
            ),
            "precision": np_dtype(precision).name,
            "beta": float(request.beta),
            "gamma": float(request.gamma),
            "order": int(request.preconditioner_order),
            "lineinfo": bool(config.lineinfo),
        }
        canonical_args = tuple(
            (name, canonical_by_name[name])
            for name in entry.factory_args
        )
        member_hash = helper_member_hash(source_hash, canonical_args)

        member = helpers.members.get(member_hash)
        if member is not None:
            return member

        bound_kwargs = {
            name: available_args[name] for name in entry.factory_args
        }
        device_function = factory(**bound_kwargs)
        # Generated prepare_jac source stamps aux_count on the factory.
        member = HelperResult(
            device_function=device_function,
            cached_auxiliary_count=(
                factory.aux_count if entry.returns_aux_count else None
            ),
        )
        helpers.members[member_hash] = member
        return member
