"""Bridge function inspection output to the parser's equation format.

Converts the AST metadata from :func:`inspect_ode_function` into
``(equation_map, funcs, new_params)`` — the same triple that the string
and SymPy branches of :func:`parse_input` produce.

Published Functions
-------------------
:func:`parse_function_input`
    Convert a callable ODE definition into structured equation data.
:func:`infer_function_states`
    Derive state names from a callable when ``states`` is omitted.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Set, Tuple

import ast
import warnings

import sympy as sp

from ..indexedbasemaps import IndexedBases
from .function_inspector import (
    AstToSympyConverter,
    FunctionInspection,
    _resolve_func_name,
    inspect_ode_function,
)
from .parser import (
    EquationWarning,
    KNOWN_FUNCTIONS,
    TIME_SYMBOL,
    _build_sympy_user_functions,
)


def parse_function_input(
    func: Callable,
    index_map: IndexedBases,
    observables: Optional[List[str]] = None,
    user_functions: Optional[Dict[str, Callable]] = None,
    user_function_derivatives: Optional[Dict[str, Callable]] = None,
    strict: bool = False,
    declared_states: Optional[List[str]] = None,
) -> Tuple[
    List[Tuple[sp.Symbol, sp.Expr]],
    Dict[str, Callable],
    List[sp.Symbol],
]:
    """Convert a callable ODE definition into structured equations.

    Parameters
    ----------
    func
        Python function defining the ODE right-hand side.
    index_map
        Pre-built indexed bases from user-supplied state/param metadata.
    observables
        Observable variable names to extract from local assignments.
    user_functions
        Mapping of callable names used in the function body to their
        implementations. Non-device callables are inlined
        symbolically; device functions and callables with derivative
        helpers stay as symbolic calls.
    user_function_derivatives
        Mapping of callable names to derivative helper functions.
    strict
        When ``True`` container accesses on undeclared names raise
        instead of inferring new parameters.
    declared_states
        State names in written order, which positional access and
        list returns bind to.

    Returns
    -------
    tuple
        ``(equation_map, funcs, new_params)`` matching the output contract
        of the string and SymPy branches in :func:`parse_input`.
    """
    if observables is None:
        observables = []

    inspection = inspect_ode_function(func)
    state_order = (
        [str(name) for name in declared_states]
        if declared_states is not None
        else index_map.state_names
    )
    symbol_map, new_params = _build_symbol_map(
        inspection, index_map, strict=strict, state_order=state_order
    )

    parse_locals, _, dev_map = _build_sympy_user_functions(
        user_functions or {}, {}, user_function_derivatives
    )
    symbolic_user_names = {
        name for name, is_dev in dev_map.items() if is_dev
    }
    symbolic_user_names |= set((user_function_derivatives or {}).keys())

    dxdt_alias_set = set(index_map.dxdt_names)
    inline_assignments = {
        name: node
        for name, node in inspection.assignments.items()
        if name in dxdt_alias_set
    }
    name_hints = _build_name_hints(inspection, index_map)

    converter = AstToSympyConverter(
        symbol_map,
        user_callables=user_functions or {},
        user_function_classes=parse_locals,
        symbolic_user_names=symbolic_user_names,
        inline_assignments=inline_assignments,
        strict_names=True,
        name_hints=name_hints,
    )

    state_names = index_map.state_names
    dxdt_names = list(index_map.dxdt_names)

    # Auxiliary assignments (non-state, non-observable locals). Their
    # symbols are seeded into the map before any conversion so that
    # return expressions and other auxiliaries can reference them.
    observable_set = set(observables)
    dxdt_set = set(dxdt_names)
    state_set = set(state_names)
    aux_names = []
    for name, expr_node in inspection.assignments.items():
        if name in observable_set or name in dxdt_set or name in state_set:
            continue
        if name in inspection.constant_params:
            continue
        if name == inspection.state_param:
            continue
        # Skip aliases that are direct state/constant accesses
        if _is_access_alias(expr_node, inspection):
            continue
        aux_names.append(name)
        if name not in symbol_map:
            symbol_map[name] = sp.Symbol(name, real=True)

    # Convert return expressions to derivative equations.
    # Local assignments whose names collide with dxdt symbols (e.g.
    # ``dx = expr``) are inlined so the return resolves to the RHS
    # expression rather than the dxdt output symbol.
    ret_value = inspection.return_node.value
    ret_exprs = _unpack_return(
        ret_value, converter, inspection.assignments
    )

    if len(ret_exprs) != len(state_order):
        raise ValueError(
            f"Return has {len(ret_exprs)} elements but system has "
            f"{len(state_order)} states: {state_order}"
        )

    # Build equation map: auxiliaries first, then observables, then dxdt
    equation_map: List[Tuple[sp.Symbol, sp.Expr]] = []

    for name in aux_names:
        sym = symbol_map[name]
        rhs = converter.convert(inspection.assignments[name])
        equation_map.append((sym, rhs))

    # Observable assignments
    for obs_name in observables:
        if obs_name in inspection.assignments:
            obs_sym = index_map.observables.symbol_map.get(obs_name)
            if obs_sym is None:
                obs_sym = sp.Symbol(obs_name, real=True)
            rhs = converter.convert(inspection.assignments[obs_name])
            equation_map.append((obs_sym, rhs))

    # State derivative equations
    if isinstance(ret_value, ast.Dict):
        # Dict return: keys map to state names
        for key_node, expr in zip(ret_value.keys, ret_exprs):
            if isinstance(key_node, ast.Constant):
                sname = key_node.value
            else:
                raise ValueError(
                    "Dict return keys must be string literals"
                )
            if sname not in state_set:
                raise ValueError(
                    f"Dict return key '{sname}' is not a declared state"
                )
            dx_name = f"d{sname}"
            dx_sym = index_map.dxdt.symbol_map.get(dx_name)
            if dx_sym is None:
                dx_sym = sp.Symbol(dx_name, real=True)
            equation_map.append((dx_sym, expr))
    else:
        # List/tuple return: positional against the written state order
        for i, expr in enumerate(ret_exprs):
            dx_name = f"d{state_order[i]}"
            dx_sym = index_map.dxdt.symbol_map[dx_name]
            equation_map.append((dx_sym, expr))

    funcs = _resolve_called_functions(
        inspection.function_calls, user_functions
    )

    return equation_map, funcs, new_params


def _resolve_called_functions(
    function_calls: Set[str],
    user_functions: Optional[Dict[str, Callable]],
) -> Dict[str, Callable]:
    """Map called names to callables, mirroring ``_process_calls``.

    Parameters
    ----------
    function_calls
        Raw call names collected during AST inspection.
    user_functions
        User-provided callable mapping.

    Returns
    -------
    dict
        Resolved callables keyed by the names used in the function
        body. Unknown names are omitted; reachable unknown calls have
        already raised during expression conversion.
    """
    resolved: Dict[str, Callable] = {}
    ufuncs = user_functions or {}
    for called in function_calls:
        bare = _resolve_func_name(called)
        if called in ufuncs:
            resolved[called] = ufuncs[called]
        elif bare in ufuncs:
            resolved[bare] = ufuncs[bare]
        elif bare in KNOWN_FUNCTIONS:
            resolved[bare] = KNOWN_FUNCTIONS[bare]
    return resolved


def _build_name_hints(
    inspection: FunctionInspection,
    index_map: IndexedBases,
) -> Dict[str, str]:
    """Suggest container spellings for declared-but-bare names.

    Parameters
    ----------
    inspection
        Result of AST analysis.
    index_map
        Pre-built indexed bases.

    Returns
    -------
    dict
        Mapping from declared symbol names to the container access the
        unknown-name error message should recommend.
    """
    hints: Dict[str, str] = {}
    scalar_set = set(inspection.scalar_params)
    containers = [
        cp for cp in inspection.constant_params if cp not in scalar_set
    ]
    container = containers[0] if containers else "p"
    # A trailing container argument (e.g. ``def f(t, y, p, d)``) is the
    # natural spelling for drivers; with a single container both hints
    # coincide.
    driver_container = containers[-1] if containers else "p"
    for ibm in (index_map.parameters, index_map.constants):
        for name in ibm.symbol_map:
            hints[name] = f"{container}.{name}"
    for name in index_map.drivers.symbol_map:
        hints[name] = f"{driver_container}.{name}"
    for name in index_map.states.symbol_map:
        hints.setdefault(name, f"{inspection.state_param}.{name}")
    return hints


def infer_function_states(func: Callable) -> Dict[str, float]:
    """Derive state names from a callable when ``states`` is omitted.

    Parameters
    ----------
    func
        Python function defining the ODE right-hand side.

    Returns
    -------
    dict
        Mapping from inferred state names to a default initial value
        of ``0.0``, in state order.

    Raises
    ------
    ValueError
        When the return form carries no binding between return
        positions and state names (a list/tuple return combined with
        named state access).

    Notes
    -----
    A dict return names and orders the states through its keys. A
    list/tuple (or scalar) return with purely positional state access
    (``y[0]``, ``y[1]``) binds return position ``i`` to state ``i``,
    so names are synthesised from the state argument name (``y0``,
    ``y1``, ...). A list/tuple return combined with attribute or
    string state access is genuinely ambiguous and raises.
    """
    inspection = inspect_ode_function(func)
    ret_value = inspection.return_node.value

    if isinstance(ret_value, ast.Dict):
        names = []
        for key_node in ret_value.keys:
            if not (
                isinstance(key_node, ast.Constant)
                and isinstance(key_node.value, str)
            ):
                raise ValueError(
                    "Dict return keys must be string literals"
                )
            names.append(key_node.value)
        return {name: 0.0 for name in names}

    patterns = {a["pattern_type"] for a in inspection.state_accesses}
    patterns.discard("expr")
    patterns.discard("name")
    if patterns <= {"int"}:
        if isinstance(ret_value, (ast.List, ast.Tuple)):
            count = len(ret_value.elts)
        else:
            count = 1
        int_keys = [
            a["key"]
            for a in inspection.state_accesses
            if a["pattern_type"] == "int"
        ]
        if int_keys:
            count = max(count, max(int_keys) + 1)
        prefix = inspection.state_param
        return {f"{prefix}{i}": 0.0 for i in range(count)}

    raise ValueError(
        "Cannot infer state names: the function returns a list/tuple "
        "but accesses states by name, so return positions carry no "
        "state binding. Pass states=..., or return a dict keyed by "
        "state name."
    )


def _build_symbol_map(
    inspection: FunctionInspection,
    index_map: IndexedBases,
    strict: bool = False,
    state_order: Optional[List[str]] = None,
) -> Tuple[Dict[str, sp.Basic], List[sp.Symbol]]:
    """Build a mapping from function-local names to SymPy symbols.

    Parameters
    ----------
    inspection
        Result of AST analysis.
    index_map
        Pre-built indexed bases.
    strict
        When ``True`` undeclared container accesses raise instead of
        inferring new parameters.
    state_order
        State names in written order, used for positional access.

    Returns
    -------
    tuple
        ``(symbol_map, inferred_params)`` — the variable name to SymPy
        symbol/expression mapping, and parameter symbols inferred from
        undeclared container accesses in non-strict mode.
    """
    smap: Dict[str, sp.Basic] = {}

    # Time parameter
    smap[inspection.param_names[0]] = TIME_SYMBOL

    if state_order is None:
        state_order = index_map.state_names
    state_names = list(state_order)
    state_symbols = [
        index_map.states.symbol_map[name] for name in state_names
    ]

    # State accesses: build lookup keys for subscript/attribute patterns
    for acc in inspection.state_accesses:
        key = acc["key"]
        ptype = acc["pattern_type"]
        if ptype == "int":
            if 0 <= key < len(state_names):
                sym = state_symbols[key]
                lookup = f"{acc['base']}[{key}]"
                smap[lookup] = sym
            else:
                raise ValueError(
                    f"State access '{acc['base']}[{key}]' is out of "
                    f"range: the system has {len(state_names)} "
                    f"states: {state_names}"
                )
        elif ptype == "string":
            if key in index_map.states.symbol_map:
                sym = index_map.states.symbol_map[key]
                lookup = f"{acc['base']}[{key!r}]"
                smap[lookup] = sym
            else:
                raise ValueError(
                    f"Unknown state '{key}' accessed as "
                    f"'{acc['base']}[{key!r}]'. Declared states: "
                    f"{state_names}"
                )
        elif ptype == "attribute":
            if key in index_map.states.symbol_map:
                sym = index_map.states.symbol_map[key]
                lookup = f"{acc['base']}.{key}"
                smap[lookup] = sym
            else:
                raise ValueError(
                    f"Unknown state '{key}' accessed as "
                    f"'{acc['base']}.{key}'. Declared states: "
                    f"{state_names}"
                )

    # Constant/parameter/driver accesses on generic container args
    inferred_params: "OrderedDict[str, sp.Symbol]" = OrderedDict()
    searched_maps = (
        index_map.parameters,
        index_map.constants,
        index_map.drivers,
    )
    for acc in inspection.constant_accesses:
        key = acc["key"]
        ptype = acc["pattern_type"]
        base = acc["base"]
        if ptype not in ("int", "string", "attribute"):
            continue
        # Search parameters, then constants, then drivers
        target_sym = None
        for ibm in searched_maps:
            if ptype == "int":
                names = list(ibm.symbol_map.keys())
                if 0 <= key < len(names):
                    target_sym = ibm.symbol_map[names[key]]
                    break
            elif key in ibm.symbol_map:
                target_sym = ibm.symbol_map[key]
                break

        if ptype == "int":
            lookup = f"{base}[{key}]"
        elif ptype == "string":
            lookup = f"{base}[{key!r}]"
        else:
            lookup = f"{base}.{key}"

        if target_sym is None:
            if ptype == "int":
                raise ValueError(
                    f"Container access '{lookup}' is out of range for "
                    f"the declared parameters "
                    f"({list(index_map.parameters.symbol_map)}), "
                    f"constants "
                    f"({list(index_map.constants.symbol_map)}), and "
                    f"drivers ({list(index_map.drivers.symbol_map)})."
                )
            if key in index_map.states.symbol_map:
                raise ValueError(
                    f"'{key}' is a state but was accessed as "
                    f"'{lookup}'. Access states through the state "
                    f"argument: '{inspection.state_param}.{key}'."
                )
            if key in index_map.observables.symbol_map:
                raise ValueError(
                    f"'{key}' is an observable but was accessed as "
                    f"'{lookup}'. Observables are assigned locally in "
                    f"the function body and referenced by bare name."
                )
            if strict:
                raise ValueError(
                    f"Container access '{lookup}' does not match any "
                    f"declared parameter, constant, or driver, and "
                    f"strict=True forbids inference. Declare '{key}' "
                    f"or set strict=False."
                )
            if key in inferred_params:
                target_sym = inferred_params[key]
            else:
                target_sym = sp.Symbol(key, real=True)
                inferred_params[key] = target_sym
                warnings.warn(
                    f"Container access '{lookup}' does not match any "
                    f"declared parameter, constant, or driver; "
                    f"'{key}' was added as a parameter with a "
                    f"default value of 0.0.",
                    EquationWarning,
                    stacklevel=3,
                )
        smap[lookup] = target_sym

    # Scalar extra arguments bind to the like-named declared symbol
    # (SciPy's args= convention: def f(t, y, mu, k) with mu, k
    # declared as parameters, constants, or drivers).
    for arg_name in inspection.scalar_params:
        target_sym = None
        for ibm in searched_maps:
            if arg_name in ibm.symbol_map:
                target_sym = ibm.symbol_map[arg_name]
                break
        if target_sym is None:
            if arg_name in index_map.states.symbol_map:
                raise ValueError(
                    f"Scalar argument '{arg_name}' matches a declared "
                    f"state. States are passed together in the second "
                    f"argument; access '{arg_name}' as "
                    f"'{inspection.state_param}.{arg_name}'."
                )
            if strict:
                raise ValueError(
                    f"Scalar argument '{arg_name}' does not match any "
                    f"declared parameter, constant, or driver, and "
                    f"strict=True forbids inference. Declare "
                    f"'{arg_name}' or set strict=False."
                )
            if arg_name in inferred_params:
                target_sym = inferred_params[arg_name]
            else:
                target_sym = sp.Symbol(arg_name, real=True)
                inferred_params[arg_name] = target_sym
                warnings.warn(
                    f"Scalar argument '{arg_name}' does not match any "
                    f"declared parameter, constant, or driver; it was "
                    f"added as a parameter with a default value of "
                    f"0.0.",
                    EquationWarning,
                    stacklevel=3,
                )
        smap[arg_name] = target_sym

    # Assignments that alias state accesses: map local name to state symbol
    for name, expr_node in inspection.assignments.items():
        sym = _resolve_alias(expr_node, inspection, smap)
        if sym is not None:
            smap[name] = sym

    # NOTE: dxdt symbols (dx, dv, ...) are intentionally NOT added to
    # the symbol map.  Users commonly write ``dx = expr; return [dx]``
    # and those Names must resolve to their assigned expression, not to
    # the dxdt output symbol (which would create a circular reference).

    # Observable symbols
    for obs_name, obs_sym in index_map.observables.symbol_map.items():
        smap[obs_name] = obs_sym

    return smap, list(inferred_params.values())


def _resolve_alias(
    expr_node: ast.expr,
    inspection: FunctionInspection,
    smap: Dict[str, sp.Basic],
) -> Optional[sp.Basic]:
    """If expr_node is a direct access on state/constant, return symbol."""
    if isinstance(expr_node, ast.Subscript):
        if isinstance(expr_node.value, ast.Name):
            base = expr_node.value.id
            slc = expr_node.slice
            if isinstance(slc, ast.Constant):
                key = slc.value
                if isinstance(key, str):
                    lookup = f"{base}[{key!r}]"
                else:
                    lookup = f"{base}[{key}]"
                return smap.get(lookup)
    elif isinstance(expr_node, ast.Attribute):
        if isinstance(expr_node.value, ast.Name):
            lookup = f"{expr_node.value.id}.{expr_node.attr}"
            return smap.get(lookup)
    return None


def _is_access_alias(
    expr_node: ast.expr, inspection: FunctionInspection
) -> bool:
    """Check if an assignment is a direct state/constant access."""
    if isinstance(expr_node, ast.Subscript):
        if isinstance(expr_node.value, ast.Name):
            base = expr_node.value.id
            return (
                base == inspection.state_param
                or base in inspection.constant_params
            )
    elif isinstance(expr_node, ast.Attribute):
        if isinstance(expr_node.value, ast.Name):
            base = expr_node.value.id
            return (
                base == inspection.state_param
                or base in inspection.constant_params
            )
    return False


def _unpack_return(
    node: ast.expr,
    converter: AstToSympyConverter,
    assignments: Optional[Dict[str, ast.expr]] = None,
) -> List[sp.Expr]:
    """Unpack a return value into a list of SymPy expressions.

    Parameters
    ----------
    node
        The return value AST node.
    converter
        AST-to-SymPy converter with populated symbol map.
    assignments
        Local assignments from the function body.  When a return
        element is a bare ``ast.Name`` that matches an assignment,
        the assignment expression is converted instead (inlining).

    Returns
    -------
    list
        One SymPy expression per state derivative.
    """
    if assignments is None:
        assignments = {}

    def _convert_element(elt: ast.expr) -> sp.Expr:
        # Inline local assignments so that ``dx = expr; return [dx]``
        # resolves to *expr*, not to the dxdt output symbol.
        if isinstance(elt, ast.Name) and elt.id in assignments:
            return converter.convert(assignments[elt.id])
        return converter.convert(elt)

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_convert_element(elt) for elt in node.elts]
    elif isinstance(node, ast.Dict):
        return [_convert_element(v) for v in node.values]
    else:
        # Single expression — single-state system
        return [_convert_element(node)]
