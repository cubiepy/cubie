"""Symbolic ODE system built from :mod:`sympy` expressions.

Published Classes
-----------------
:class:`SymbolicODE`
    Concrete :class:`~cubie.odesystems.baseODE.BaseODE` subclass that
    generates CUDA device functions from SymPy equations. Handles
    codegen caching, solver helper generation, and swept parameters.

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
    Iterable,
    Optional,
    Union,
)

from numpy import dtype as np_dtype, float32
from cubie.odesystems.symbolic.codegen.dxdt import (
    generate_dxdt_fac_code,
    generate_observables_fac_code,
)
from cubie.odesystems.symbolic.codegen.jacobian import generate_analytical_jvp
from cubie.odesystems.symbolic.helper_registry import (
    helper_member_hash,
    helper_source_hash,
)
from cubie.odesystems.symbolic.odefile import ODEFile
from cubie.odesystems.symbolic.parsing import (
    JVPEquations,
    parse_system,
)
from cubie.odesystems.symbolic.parsing.parsed_system import ParsedSystem
from cubie.odesystems.symbolic.engine.adapter import buffer_maps
from cubie.odesystems.symbolic.specialiser import SystemSpecialiser
from cubie.odesystems.baseODE import BaseODE, ODECache
from cubie.odesystems.variables import ParameterSet
from cubie.odesystems.solver_helpers import (
    HelperResult,
    OperationCounts,
    SolverHelperRequest,
    device_function_operation_count,
)
from cubie._serialize import canonical_digest
from cubie._env import operation_ordering_default
from cubie._utils import PrecisionDType, is_devfunc
from cubie.time_logger import default_timelogger


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
    drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
    user_functions: Optional[dict[str, Callable]] = None,
    user_function_derivatives: Optional[dict[str, Callable]] = None,
    name: Optional[str] = None,
    strict: bool = False,
    state_priority: Optional[dict[str, float]] = None,
    irreducible: Optional[Iterable[str]] = None,
    simplify_options: Optional[dict[str, Any]] = None,
    operation_ordering: Optional[str] = None,
    swept_parameters: Optional[Iterable[str]] = None,
) -> "SymbolicODE":
    """Create a :class:`SymbolicODE` from SymPy definitions.

    Parameters
    ----------
    dxdt
        System equations defined as a single string, an iterable of equation
        strings in ``lhs = rhs`` form, or a Python callable. When a callable
        is provided its signature must be ``f(t, y, ...)`` where ``t`` is
        time, ``y`` is the state vector, and additional arguments map to
        parameters or drivers. State access patterns supported:
        ``y[0]`` (positional), ``y["name"]`` (string), ``y.name``
        (attribute). ``y[i]`` reads the i-th state in the order you
        declare them. Read a parameter or driver from any later
        argument by name (``p.k`` or ``p["k"]``), or by position in
        the order you declare them: with two or more such arguments,
        ``d[i]`` on the last reads the i-th driver and ``p[i]`` on any
        other the i-th parameter; with one, ``p[i]`` counts through the
        parameters and then the drivers. The return value must be a
        list, tuple, or dict of derivative expressions.
    states
        State labels either as an iterable or as a mapping to default initial
        values.
    observables
        Observable variable labels to expose from the generated system.
    parameters
        Parameter labels either as an iterable or as a mapping to default
        values. A parameter that holds one value across a batch is
        compiled into the generated code as a number.
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
        solver helpers. A value may also be a list of callables
        evaluating the first, second, ... derivatives; the order-``n``
        callable takes the function's arguments followed by ``n``
        argument indices.
    name
        Identifier used for generated files. Defaults to ``unnamed_``
        followed by the start of a hash of the equations.
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
    swept_parameters
        Names of the parameters that vary between the runs of a batch.
        Every other parameter is compiled in at its default value, and
        when you leave this out every parameter is compiled in. A solve
        sets the swept parameters to match its inputs, so giving them
        here saves specialising the system again at the first solve,
        which can take 10-100 ms.

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
        swept_parameters=swept_parameters,
    )
    return symbolic_ode


class SymbolicODE(BaseODE):
    """Symbolic representation of an ODE system.

    The system holds a :class:`ParsedSystem`, with every parameter
    left as a symbol, and a :class:`SystemSpecialiser` child that
    assembles it for the parameters the current batch compiles.

    Parameters
    ----------
    parsed_system
        Parsed system with every parameter left as a symbol.
    precision
        Target floating-point precision used for generated kernels.
    name
        Identifier used for generated modules.
    operation_ordering
        Generated-operation ordering policy.
    swept_parameters
        Names of the parameters that vary between the runs of a batch.
        Every other parameter is compiled in at its default value, and
        when you leave this out every parameter is compiled in. A solve
        sets the swept parameters to match its inputs, so giving them
        here saves specialising the system again at the first solve,
        which can take 10-100 ms.
    """

    # Diagnostic evaluators live in a dict so child-factory
    # discovery skips them and config_hash is unaffected.

    def __init__(
        self,
        parsed_system: ParsedSystem,
        precision: PrecisionDType,
        name: Optional[str] = None,
        operation_ordering: Optional[str] = None,
        swept_parameters: Optional[Iterable[str]] = None,
    ):
        """Specialise a parsed system for a set of swept parameters.

        Parameters
        ----------
        parsed_system
            Parsed system with every parameter left as a symbol.
        precision
            Target floating-point precision used for generated kernels.
        name
            Identifier used for generated modules. Defaults to
            ``unnamed_`` followed by the start of a hash of the
            equations, which is the same whichever parameters are
            swept.
        operation_ordering
            Policy for ordering the generated operations. Defaults to
            ``CUBIE_OPERATION_ORDERING``, read when the system is
            built, or ``liveness_auto`` if that is unset.
        swept_parameters
            Names of the parameters that vary between the runs of a
            batch. Every other parameter is compiled in at its default
            value, and when you leave this out every parameter is
            compiled in. A solve sets the swept parameters to match its
            inputs, so giving them here saves specialising the system
            again at the first solve, which can take 10-100 ms.
        """
        if operation_ordering is None:
            operation_ordering = operation_ordering_default()
        self._parsed_system = parsed_system
        if name is None:
            name = f"unnamed_{parsed_system.definition_hash[:8]}"
        declared = parsed_system.declared_parameters
        specialiser = SystemSpecialiser(
            parsed_system, ParameterSet.build(declared, swept_parameters or ())
        )
        # We keep every initial value we are given, including values
        # for states that we remove in structural simplification.
        # When we compile a specialisation that keeps such a state, we
        # start it from the value the user set.
        super().__init__(
            precision,
            declared,
            specialiser,
            initial_values=parsed_system.states,
            operation_ordering=operation_ordering,
            name=name,
        )
        self.gen_file = ODEFile(
            name,
            _operation_source_hash(
                self.fn_hash,
                self.compile_settings.operation_ordering,
            ),
        )

    @property
    def equations(self):
        """The equations with every fixed value folded in."""
        return self._specialiser.equations

    @property
    def user_functions(self) -> dict[str, Callable]:
        """Every callable the generated code may call, by name."""
        return self._specialiser.functions

    @property
    def fn_hash(self) -> str:
        """Hash of every value the generated source embeds."""
        return self._specialiser.fn_hash

    @classmethod
    def create(
        cls,
        dxdt: Union[str, Iterable[str], Callable],
        precision: PrecisionDType,
        states: Optional[Union[dict[str, float], Iterable[str]]] = None,
        observables: Optional[Iterable[str]] = None,
        parameters: Optional[Union[dict[str, float], Iterable[str]]] = None,
        drivers: Optional[Union[Iterable[str], dict[str, Any]]] = None,
        user_functions: Optional[dict[str, Callable]] = None,
        user_function_derivatives: Optional[dict[str, Callable]] = None,
        name: Optional[str] = None,
        strict: bool = False,
        state_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
        parameter_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
        observable_units: Optional[
            Union[dict[str, str], Iterable[str]]
        ] = None,
        driver_units: Optional[Union[dict[str, str], Iterable[str]]] = None,
        state_priority: Optional[dict[str, float]] = None,
        irreducible: Optional[Iterable[str]] = None,
        simplify_options: Optional[dict[str, Any]] = None,
        operation_ordering: Optional[str] = None,
        swept_parameters: Optional[Iterable[str]] = None,
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
        drivers
            External driver variable labels required at runtime. May be an
            iterable of driver labels or a dictionary describing driver
            defaults or driver-array samples alongside configuration entries.
        user_functions
            Custom callables referenced within ``dxdt`` expressions.
        user_function_derivatives
            Mapping of user-function names to callables evaluating
            their analytic derivatives, used when generating
            Jacobian-based solver helpers. A value may also be a list
            of callables evaluating the first, second, ...
            derivatives; the order-``n`` callable takes the function's
            arguments followed by ``n`` argument indices.
        name
            Identifier used for generated files. Defaults to
            ``unnamed_`` followed by the start of a hash of the
            equations.
        precision
            Target floating-point precision used when compiling the system.
        strict
            When ``True`` require every symbol to be explicitly categorised.
        state_units
            Optional units for states. Defaults to "dimensionless".
        parameter_units
            Optional units for parameters. Defaults to "dimensionless".
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
        swept_parameters
            Names of the parameters that vary between the runs of a
            batch. Every other parameter is compiled in at its default
            value, and when you leave this out every parameter is
            compiled in. A solve sets the swept parameters to match its
            inputs, so giving them here saves specialising the system
            again at the first solve, which can take 10-100 ms.

        Returns
        -------
        SymbolicODE
            Fully constructed symbolic system ready for compilation.
        """

        # Register timing event for parsing (one-time registration)
        default_timelogger.register_event(
            "symbolic_ode_parsing",
            "codegen",
            "Codegen time for symbolic ODE parsing",
        )

        # Start timing for parsing operation
        default_timelogger.start_event("symbolic_ode_parsing")
        parsed_system = parse_system(
            dxdt=dxdt,
            states=states,
            observables=observables,
            parameters=parameters,
            drivers=drivers,
            user_functions=user_functions,
            user_function_derivatives=user_function_derivatives,
            strict=strict,
            state_units=state_units,
            parameter_units=parameter_units,
            observable_units=observable_units,
            driver_units=driver_units,
            state_priority=state_priority,
            irreducible=irreducible,
            simplify_options=simplify_options,
        )
        symbolic_ode = cls(
            parsed_system,
            precision,
            name=name,
            operation_ordering=operation_ordering,
            swept_parameters=swept_parameters,
        )
        default_timelogger.stop_event("symbolic_ode_parsing")
        return symbolic_ode

    @property
    def state_units(self) -> dict[str, str]:
        """Return units for state variables."""
        return self.states.units

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return units for parameters."""
        return self.parameters.units

    @property
    def observable_units(self) -> dict[str, str]:
        """Return units for observables."""
        return self.observables.units

    @property
    def driver_units(self) -> dict[str, str]:
        """Return units for drivers."""
        return self.layout.drivers.units

    def _get_jvp_exprs(self) -> JVPEquations:
        """Return Jacobian-vector assignments for the current system.

        Return the same assignments until the system is specialised
        again or its ordering policy changes.
        """
        equations = self.equations
        layout = self.layout
        ordering = self.compile_settings.operation_ordering

        def generate():
            maps = buffer_maps(layout)
            return generate_analytical_jvp(
                equations,
                input_order=maps.state_order,
                output_order=maps.dxdt_order,
                observables=maps.observable_symbols,
                cse=True,
                operation_ordering=ordering,
            )

        return self._derived(
            "jvp_exprs", (equations, layout, ordering), generate
        )

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
        functions = self.user_functions or {}
        injections = {
            name: func for name, func in functions.items() if is_devfunc(func)
        }
        # The string parser renames user functions with a trailing
        # underscore and the generated source calls the renamed symbol,
        # so inject each device callable under its alias as well.
        for alias, original in self.equations.function_aliases.items():
            func = functions.get(original)
            if func is not None and is_devfunc(func):
                injections[alias] = func
        return injections

    def build(self) -> ODECache:
        """Compile the ``dxdt`` factory and refresh the cache.

        Returns
        -------
        ODECache
            Cache populated with the compiled ``dxdt`` callable.
        """
        numba_precision = self.numba_precision
        lineinfo = self.compile_settings.lineinfo
        source_hash = _operation_source_hash(
            self.fn_hash,
            self.compile_settings.operation_ordering,
        )
        if self.gen_file.fn_hash != source_hash:
            self.gen_file = ODEFile(self.name, source_hash)

        dxdt_code = None
        if not self.gen_file.function_is_cached("dxdt_factory"):
            dxdt_code = generate_dxdt_fac_code(
                self.equations,
                self.layout,
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
            numba_precision,
            lineinfo=lineinfo,
        )

        obs_code = None
        if not self.gen_file.function_is_cached("observables_factory"):
            obs_code = generate_observables_fac_code(
                self.equations, self.layout,
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
        observables_fn = observables_factory(
            numba_precision,
            lineinfo=lineinfo,
        )

        return ODECache(
            dxdt_fn=dxdt_func,
            observables_fn=observables_fn,
            operation_counts=OperationCounts(
                dxdt=device_function_operation_count(dxdt_func),
                observables=device_function_operation_count(
                    observables_fn
                ),
            ),
        )

    def get_parameters_info(self) -> list[dict]:
        """Return information about all parameters.

        Returns
        -------
        list of dict
            Each dict contains 'name', 'value', and 'unit' keys.
        """
        return [
            {
                'name': parameter.name,
                'value': parameter.default,
                'unit': parameter.unit,
            }
            for parameter in self.parameters
        ]

    def get_states_info(self) -> list[dict]:
        """Return information about all state variables.

        Returns
        -------
        list of dict
            Each dict contains 'name', 'value', and 'unit' keys.
        """
        return [
            {
                'name': state.name,
                'value': state.default,
                'unit': state.unit,
            }
            for state in self.states
        ]

    def parameters_gui(self, blocking: bool = True) -> None:
        # no cover: start
        """Launch a Qt GUI for editing parameter values.

        The GUI displays every parameter with its value and unit.

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
        >>> ode.parameters_gui()  # Opens editor dialog
        """
        from cubie.gui.parameters_editor import show_parameters_editor
        show_parameters_editor(self, blocking=blocking)
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
        role: str,
        **request_kwargs: Any,
    ) -> HelperResult:
        """Return the bound helper member for one role and variant.

        Parameters
        ----------
        role
            Registered role name (``"linear_operator"``,
            ``"residual"``, ...) or preconditioner type name
            (``"neumann"``, ``"jacobi"``).
        **request_kwargs
            Remaining :class:`SolverHelperRequest` fields:
            ``jacobian_at``, ``prefactored``, ``stacked``,
            ``operator_beta``, ``operator_gamma``,
            ``preconditioner_order``, and stage data.

        Returns
        -------
        HelperResult
            The bound device callable and its typed metadata. Cached
            Jacobian-carrying members carry ``prepare_jac`` and
            ``cached_auxiliary_count``.

        Notes
        -----
        Mass-consuming helpers read :attr:`mass`. A
        repeated request returns the same member; bindings sharing
        source reuse one generated factory.
        """
        request = SolverHelperRequest(role=role, **request_kwargs)
        role = request.role

        event_name = (
            f"solver_helper_{role.name}_{request.variant.value}"
        )
        default_timelogger.register_event(
            event_name,
            "codegen",
            f"Codegen time for solver helper {role.name} "
            f"({request.variant.value})",
        )

        # Validation hooks run on every request, cache hits included.
        role.validate(self, request)

        helpers = self.get_cached_output("helpers")

        # The generated function's name contains the full source hash.
        source_hash = helper_source_hash(self, request)
        factory_name = (
            f"{role.name}_{request.variant.value}_s{source_hash}"
        )

        if source_hash not in helpers.factories:
            is_cached = self.gen_file.function_is_cached(factory_name)
            default_timelogger.start_event(event_name, skipped=is_cached)
            code = None
            if not is_cached:
                code = role.generate(self, request, factory_name)
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
        available_args = {
            "precision": self.numba_precision,
            "order": request.preconditioner_order,
            "lineinfo": config.lineinfo,
            "unroll_solver_element": request.unroll_solver_element,
            "unroll_other_small": request.unroll_other_small,
        }
        canonical_by_name = {
            "precision": np_dtype(precision).name,
            "order": int(request.preconditioner_order),
            "lineinfo": bool(config.lineinfo),
            "unroll_solver_element": request.unroll_solver_element,
            "unroll_other_small": request.unroll_other_small,
        }
        canonical_args = tuple(
            (name, canonical_by_name[name])
            for name in role.factory_args
        )
        member_hash = helper_member_hash(source_hash, canonical_args)

        member = helpers.members.get(member_hash)
        if member is not None:
            return member

        bound_kwargs = {
            name: available_args[name] for name in role.factory_args
        }
        device_function = factory(**bound_kwargs)
        # Get the prepare-cache helper
        prepare_member = None
        if (
            request.variant.uses_cached_aux
            and role.jacobian_carrying
            and not role.is_prepare_helper
        ):
            prepare_kwargs = role.prepare_request_kwargs(request)
            prepare_member = self.get_solver_helper(
                prepare_kwargs.pop("role"),
                **prepare_kwargs,
            )
        # get extra buffer sizes if they exist
        aux_count = factory.aux_count
        if aux_count is None and prepare_member is not None:
            aux_count = prepare_member.cached_auxiliary_count
        member = HelperResult(
            device_function=device_function,
            cached_auxiliary_count=aux_count,
            prepare_jac=(
                prepare_member.device_function
                if prepare_member is not None
                else None
            ),
            lu_nnz=factory.lu_nnz,
            operation_count=device_function_operation_count(
                device_function
            ),
            prepare_operation_count=(
                prepare_member.operation_count
                if prepare_member is not None
                else 0
            ),
        )
        helpers.members[member_hash] = member
        return member
