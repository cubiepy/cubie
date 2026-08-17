"""Parse ODE and DAE definitions into engine IR equations.

String, SymPy, and callable inputs produce :class:`ParsedEquations`.
DAEs pass through structural simplification before assembly.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Union,
)

import sympy as sp

from ..engine import expr as ir_expr
from ..engine.from_sympy import (
    convert_assignments,
    derivative_name_map,
)
from ..indexedbasemaps import IndexedBases
from .function_parser import (
    infer_function_states,
    parse_function_input,
)
from .normalise import normalise_input
from .parse_primitives import TIME_SYMBOL
from .parsed_system import ParsedSystem

DRIVER_SETTING_KEYS = {"time", "driver_sample_period", "wrap", "order"}


def _detect_input_type(dxdt: Union[str, Iterable, Callable]) -> str:
    """Detect whether dxdt contains strings, SymPy expressions, or a callable.

    Determines input format by inspecting the type of dxdt itself and,
    for iterables, examining the first element to categorize as either
    string-based or SymPy-based input.

    Parameters
    ----------
    dxdt
        System equations as string, iterable, or callable.

    Returns
    -------
    str
        Either 'string', 'sympy', or 'function' indicating input format.

    Raises
    ------
    TypeError
        If input type cannot be determined or is invalid.
    ValueError
        If empty iterable is provided.
    """
    if dxdt is None:
        raise TypeError("dxdt cannot be None")

    if callable(dxdt) and not isinstance(dxdt, (str, list, tuple)):
        return "function"

    if isinstance(dxdt, str):
        return "string"

    if isinstance(dxdt, (sp.Equality, sp.Expr)):
        return "sympy"

    try:
        items = list(dxdt)
    except TypeError:
        raise TypeError(
            f"dxdt must be string or iterable, got {type(dxdt).__name__}"
        )

    if len(items) == 0:
        raise ValueError("dxdt iterable cannot be empty")

    first_elem = items[0]

    if isinstance(first_elem, str):
        return "string"
    elif isinstance(first_elem, (sp.Expr, sp.Equality)):
        return "sympy"
    elif isinstance(first_elem, tuple) and len(first_elem) == 2:
        # A (lhs, rhs) pair; both members must be engine IR nodes or
        # convertible to SymPy expressions, or the pair is rejected
        # here rather than failing deep inside the normaliser.
        for side, member in zip(("lhs", "rhs"), first_elem):
            if not isinstance(
                member,
                (ir_expr.Expr, sp.Basic, str, int, float, complex),
            ):
                raise TypeError(
                    f"dxdt element 0 is a (lhs, rhs) tuple whose "
                    f"{side} is {member!r} "
                    f"({type(member).__name__}); each member must "
                    f"be an IR or SymPy expression, string, or "
                    f"number."
                )
        return "sympy"

    raise TypeError(
        f"dxdt elements must be strings or symbolic expressions, "
        f"got {type(first_elem).__name__}. "
        f"Valid symbolic formats: sp.Equality or a (lhs, rhs) tuple."
    )


def _process_parameters(
    states: Union[Dict[str, float], Iterable[str]],
    parameters: Union[Dict[str, float], Iterable[str]],
    constants: Union[Dict[str, float], Iterable[str]],
    observables: Iterable[str],
    drivers: Iterable[str],
    state_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    parameter_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    constant_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    observable_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    driver_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
) -> IndexedBases:
    """Convert user-specified symbols into ``IndexedBases`` structures.

    Parameters
    ----------
    states
        State symbols or mapping to initial values.
    parameters
        Parameter symbols or mapping to default values.
    constants
        Constant symbols or mapping to default values.
    observables
        Observable symbol names supplied by the user.
    drivers
        External driver symbol names.
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

    Returns
    -------
    IndexedBases
        Structured representation of all indexed symbol collections.
    """
    indexed_bases = IndexedBases.from_user_inputs(
        states,
        parameters,
        constants,
        observables,
        drivers,
        state_units=state_units,
        parameter_units=parameter_units,
        constant_units=constant_units,
        observable_units=observable_units,
        driver_units=driver_units,
    )
    return indexed_bases


def parse_input(
    dxdt: Union[str, Iterable, Callable],
    states: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    observables: Optional[Iterable[str]] = None,
    parameters: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    constants: Optional[Union[Dict[str, float], Iterable[str]]] = None,
    drivers: Optional[Union[Iterable[str], Dict[str, Any]]] = None,
    user_functions: Optional[Dict[str, Callable]] = None,
    user_function_derivatives: Optional[Dict[str, Callable]] = None,
    strict: bool = False,
    state_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    parameter_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    constant_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    observable_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    driver_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    state_priority: Optional[Dict[str, float]] = None,
    irreducible: Optional[Iterable[str]] = None,
    simplify_options: Optional[Dict[str, Any]] = None,
):
    """Process user equations and symbol metadata into structured components.

    Parameters
    ----------
    dxdt
        System equations as a newline-delimited string, an iterable
        of strings, SymPy equations, or a callable. In addition to
        explicit forms, implicit equations (``0 = g(...)``),
        higher-order/nested derivatives, derivative terms inside
        expressions, and algebraic unknowns are accepted (symbolic
        input only); such systems route through structural
        simplification automatically.
    states
        All unknowns of the system (differential or algebraic) as
        names or a mapping to initial values.
    observables
        Observable variable names whose trajectories should be saved.
    parameters
        Parameter names or mapping to default values.
    constants
        Constant names or mapping to default values that remain fixed across
        runs.
    drivers
        Driver variable names supplied at runtime. Accepts either an iterable
        of driver labels or a dictionary mapping driver labels to default
        values and, when using driver arrays, configuration entries such as
        ``time``, ``dt``, ``wrap``, and ``order``.
    user_functions
        Mapping of callable names used in equations to their implementations.
    user_function_derivatives
        Mapping of callable names to derivative helper functions.
    strict
        When ``False``, infer missing symbol declarations from equation usage.
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

    Returns
    -------
    tuple
        ``(index_map, all_symbols, funcs, parsed_equations, fn_hash,
        parsed_system)``. The derived mass matrix rides on
        ``parsed_equations.mass_matrix``; ``parsed_system`` is the
        constants-symbolic checkpoint used to re-specialise the
        system when constant values change.

    Notes
    -----
    With ``strict=False``, undeclared variables inferred from usage
    are added automatically. Constant values fold into the equations
    as literals, so ``parsed_equations`` and ``fn_hash`` are
    value-specific.
    """
    input_type = _detect_input_type(dxdt)

    if input_type == "function":
        return _parse_function_path(
            dxdt,
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
        )

    if states is None and strict:
        raise ValueError(
            "No state symbols were provided - if you want to build a model "
            "from a set of equations alone, set strict=False"
        )

    states_dict = dict(states) if isinstance(states, dict) else {
        str(name): 0.0 for name in (states or [])
    }
    observables = list(observables or [])
    parameters = parameters if parameters is not None else {}
    constants = constants if constants is not None else {}

    driver_dict = None
    if drivers is None:
        driver_names = []
    elif isinstance(drivers, dict):
        driver_dict = drivers
        driver_names = [
            key for key in drivers.keys() if key not in DRIVER_SETTING_KEYS
        ]
        if not driver_names:
            raise ValueError(
                "Driver dictionary must include at least one driver symbol."
            )
    else:
        driver_names = list(drivers)

    known_symbol_map = {}
    for name in list(parameters) + list(constants) + driver_names:
        known_symbol_map[str(name)] = sp.Symbol(str(name), real=True)

    unknown_names = {str(name) for name in states_dict}
    unknown_names |= {str(name) for name in observables}

    normalised = normalise_input(
        dxdt,
        unknown_names,
        known_symbol_map,
        user_functions,
        user_function_derivatives,
        strict,
        set(states_dict),
    )
    for name in normalised.inferred_states:
        states_dict[name] = 0.0

    if isinstance(parameters, dict):
        parameters_dict = {
            str(name): value for name, value in parameters.items()
        }
    else:
        parameters_dict = {str(name): 0.0 for name in parameters}
    if isinstance(constants, dict):
        constants_dict = {
            str(name): float(value)
            for name, value in constants.items()
        }
    else:
        constants_dict = {str(name): 0.0 for name in constants}

    parsed_system = ParsedSystem(
        normalised=normalised,
        states=states_dict,
        observables=observables,
        parameters=parameters_dict,
        constants=constants_dict,
        driver_names=driver_names,
        driver_dict=driver_dict,
        known_symbol_map=known_symbol_map,
        user_functions=user_functions,
        user_function_derivatives=user_function_derivatives,
        state_priority=state_priority,
        irreducible=irreducible,
        simplify_options=simplify_options,
        state_units=state_units,
        parameter_units=parameter_units,
        constant_units=constant_units,
        observable_units=observable_units,
        driver_units=driver_units,
    )
    products = parsed_system.specialise()
    return (*products, parsed_system)


def _parse_function_path(
    dxdt: Callable,
    states,
    observables,
    parameters,
    constants,
    drivers,
    user_functions,
    user_function_derivatives,
    strict,
    state_units,
    parameter_units,
    constant_units,
    observable_units,
    driver_units,
):
    """Parse callable ``dxdt`` input."""

    if states is None:
        if strict:
            raise ValueError(
                "No state symbols were provided - if you want to build a "
                "model from a set of equations alone, set strict=False"
            )
        states = infer_function_states(dxdt)
    if observables is None:
        observables = []
    if parameters is None:
        parameters = {}
    if constants is None:
        constants = {}
    driver_dict = None
    if drivers is None:
        drivers = []
    elif isinstance(drivers, dict):
        driver_dict = drivers
        drivers = [
            key for key in drivers.keys() if key not in DRIVER_SETTING_KEYS
        ]
        if len(drivers) == 0:
            raise ValueError(
                "Driver dictionary must include at least one driver symbol."
            )

    index_map = _process_parameters(
        states=states,
        parameters=parameters,
        constants=constants,
        observables=observables,
        drivers=drivers,
        state_units=state_units,
        parameter_units=parameter_units,
        constant_units=constant_units,
        observable_units=observable_units,
        driver_units=driver_units,
    )

    equation_map, funcs, new_params = parse_function_input(
        func=dxdt,
        index_map=index_map,
        observables=list(observables),
        user_functions=user_functions,
        user_function_derivatives=user_function_derivatives,
        strict=strict,
        declared_states=list(states),
    )
    # Derivative placeholder names must be recovered from the SymPy
    # function objects before the equations convert to IR.
    function_derivative_names = derivative_name_map(equation_map)
    equation_map = convert_assignments(
        equation_map,
        allowed_functions=user_functions,
    )

    states_dict = {
        str(name): float(value)
        for name, value in index_map.state_values.items()
    }
    observables = list(observables)
    parameters_dict = {
        str(name): float(value)
        for name, value in index_map.parameter_values.items()
    }
    for param in new_params:
        parameters_dict.setdefault(str(param), 0.0)
    constants_dict = {
        str(name): float(value)
        for name, value in index_map.constant_values.items()
    }

    known_symbol_map = {}
    for name in (
        list(parameters_dict) + list(constants_dict) + list(drivers)
    ):
        known_symbol_map[str(name)] = sp.Symbol(str(name), real=True)

    unknown_names = set(states_dict) | set(observables)
    normalised = normalise_input(
        list(equation_map),
        unknown_names,
        known_symbol_map,
        funcs,
        user_function_derivatives,
        strict,
        set(states_dict),
    )
    normalised.derivative_names.update(function_derivative_names)

    parsed_system = ParsedSystem(
        normalised=normalised,
        states=states_dict,
        observables=observables,
        parameters=parameters_dict,
        constants=constants_dict,
        driver_names=list(drivers),
        driver_dict=driver_dict,
        known_symbol_map=known_symbol_map,
        user_functions=funcs,
        user_function_derivatives=user_function_derivatives,
        state_units=state_units,
        parameter_units=parameter_units,
        constant_units=constant_units,
        observable_units=observable_units,
        driver_units=driver_units,
    )
    products = parsed_system.specialise()
    (
        index_map,
        all_symbols,
        assembled_funcs,
        parsed_equations,
        fn_hash,
    ) = products
    # Inlined non-device callables keep their entries.
    funcs = {**funcs, **(assembled_funcs or {})}
    all_symbols.setdefault("t", TIME_SYMBOL)
    if funcs:
        all_symbols.update({name: fn for name, fn in funcs.items()})
        if user_function_derivatives:
            all_symbols.update(
                {
                    fn.__name__: fn
                    for fn in user_function_derivatives.values()
                    if callable(fn)
                }
            )

    return (
        index_map,
        all_symbols,
        funcs,
        parsed_equations,
        fn_hash,
        parsed_system,
    )
