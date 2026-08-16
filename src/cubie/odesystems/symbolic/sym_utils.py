"""Utility helpers for symbolic ODE construction.

Published Functions
-------------------
:func:`topological_sort`
    Return assignments sorted by dependency order using Kahn's algorithm.

    >>> import sympy as sp
    >>> x, y = sp.symbols("x y")
    >>> topological_sort([(y, x + 1), (x, sp.Integer(2))])
    [(x, 2), (y, x + 1)]

:func:`cse_and_stack`
    Apply common subexpression elimination and topologically sort the
    combined result.

    >>> import sympy as sp
    >>> a, b = sp.symbols("a b")
    >>> result = cse_and_stack([(a, sp.sin(b)), (b, sp.Integer(1))])

:func:`hash_system_definition`
    Deterministic SHA-256 hash for a set of symbolic ODE equations.

    >>> import sympy as sp
    >>> x = sp.Symbol("x")
    >>> h = hash_system_definition([(x, -x)])
    >>> len(h)
    64

:func:`prune_unused_assignments`
    Remove assignments that do not contribute to output symbols.

    >>> import sympy as sp
    >>> a, b, c = sp.symbols("a b c")
    >>> out = sp.Symbol("out[0]")
    >>> pruned = prune_unused_assignments(
    ...     [(a, sp.Integer(1)), (b, a), (out, b), (c, sp.Integer(9))],
    ...     output_symbols=[out],
    ... )
    >>> [str(lhs) for lhs, _ in pruned]
    ['a', 'b', 'out[0]']

See Also
--------
:mod:`cubie.odesystems.symbolic.codegen`
    Code generation modules that consume these helpers.
:class:`~cubie.odesystems.symbolic.indexedbasemaps.IndexedBases`
    Symbol-to-array mapping used alongside these utilities.
"""

from hashlib import sha256

from typing import TYPE_CHECKING
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Tuple, Union

import sympy as sp

if TYPE_CHECKING:
    from cubie.odesystems.symbolic.parsing import ParsedEquations

# Namespace reserved for generated bindings;
# IndexedBases.from_user_inputs rejects user names carrying it.
RESERVED_CODEGEN_PREFIX = "_cubie_codegen_"


def topological_sort(
    assignments: Union[
        List[Tuple[sp.Symbol, sp.Expr]],
        Dict[sp.Symbol, sp.Expr],
    ],
) -> List[Tuple[sp.Symbol, sp.Expr]]:
    """Return assignments sorted by their dependency order.

    Parameters
    ----------
    assignments
        Either an iterable of ``(symbol, expression)`` pairs or a mapping from
        each symbol to its defining expression.

    Returns
    -------
    list[tuple[sympy.Symbol, sympy.Expr]]
        Assignments ordered such that dependencies are defined before use.

    Raises
    ------
    ValueError
        Raised when a circular dependency prevents topological sorting.

    Notes
    -----
    Uses Kahn's algorithm for topological sorting. Refer to the Wikipedia
    article on topological sorting for additional background.
    """
    # Build symbol to expression mapping
    if isinstance(assignments, list):
        sym_map = {sym: expr for sym, expr in assignments}
    else:
        sym_map = assignments.copy()

    deps = {}
    all_assignees = set(sym_map.keys())
    for sym, expr in sym_map.items():
        expr_deps = expr.free_symbols & all_assignees
        deps[sym] = expr_deps

    # Kahn's algorithm
    incoming_edges = {sym: len(dep_syms) for sym, dep_syms in deps.items()}

    graph = defaultdict(set)
    for sym, dep_syms in deps.items():
        for dep_sym in dep_syms:
            graph[dep_sym].add(sym)

    # Start with all symbols without dependencies
    queue = deque(
        [sym for sym, degree in incoming_edges.items() if degree == 0]
    )
    result = []

    # Remove incoming edges for fully defined dependencies until none remain
    while queue:
        defined_symbol = queue.popleft()
        # Find the assignment tuple for this symbol
        assignment = sym_map[defined_symbol]
        result.append((defined_symbol, assignment))

        # Sorted iteration keeps the output order independent of
        # Symbol hash values (set order follows PYTHONHASHSEED), so
        # generated code is identical across processes.
        for dependent in sorted(graph[defined_symbol], key=str):
            incoming_edges[dependent] -= 1
            if incoming_edges[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(sym_map):
        remaining = all_assignees - {sym for sym, _ in result}
        raise ValueError(
            f"Circular dependency detected. Remaining symbols: {remaining}"
        )

    return result


def cse_and_stack(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    symbol: Optional[str] = None,
) -> List[Tuple[sp.Symbol, sp.Expr]]:
    """Perform common subexpression elimination and stack the results.

    Parameters
    ----------
    equations
        ``(symbol, expression)`` pairs that define the system.
    symbol
        Prefix to use for the generated common-subexpression symbols. Defaults
        to ``"_cse"`` when not provided.

    Returns
    -------
    list[tuple[sympy.Symbol, sympy.Expr]]
        Combined list of original expressions rewritten in terms of CSE
        symbols followed by the generated common subexpressions.
    """
    if symbol is None:
        symbol = "_cse"
    expr_labels = [lhs for lhs, _ in equations]
    all_rhs = (rhs for _, rhs in equations)

    # Find the highest existing numbered symbol with the same prefix and
    # continue numbering from there. If the prefix exists but no numeric
    # suffixes are found, start numbering at 1.
    start_index = 0
    max_index = -1
    prefix_found = False
    for label in expr_labels:
        label_str = str(label)
        if label_str.startswith(symbol):
            prefix_found = True
            suffix = label_str[len(symbol) :]
            if suffix.isdigit():
                idx = int(suffix)
                if idx > max_index:
                    max_index = idx

    if prefix_found:
        start_index = max_index + 1

    cse_exprs, reduced_exprs = sp.cse(
        all_rhs,
        symbols=sp.numbered_symbols(symbol, start=start_index),
        order="none",
    )
    expressions = list(zip(expr_labels, reduced_exprs)) + list(cse_exprs)
    sorted_expressions = topological_sort(expressions)
    return sorted_expressions


def hash_system_definition(
    equations: Union[
        "ParsedEquations",
        Iterable[Tuple[sp.Symbol, sp.Expr]],
    ],
    constants: Optional[Dict[str, float]] = None,
    observable_labels: Optional[Iterable[str]] = None,
    state_labels: Optional[Iterable[str]] = None,
    dxdt_labels: Optional[Iterable[str]] = None,
    parameter_labels: Optional[Iterable[str]] = None,
    driver_labels: Optional[Iterable[str]] = None,
    derivative_names: Optional[Dict[str, str]] = None,
    function_aliases: Optional[Dict[str, str]] = None,
) -> str:
    """Return the generated-source hash for a symbolic system.

    Parameters
    ----------
    equations
        Parsed equations object or iterable of (symbol, expression)
        tuples representing the system.
    constants
        Optional mapping of constant names to values.
    observable_labels
        Observable names in output-array order.
    state_labels
        State names in input-array order.
    dxdt_labels
        Derivative names in output-array order.
    parameter_labels
        Parameter names in input-array order.
    driver_labels
        Driver names in input-array order.
    derivative_names
        Generated derivative helper names keyed by function name.
    function_aliases
        IR call names and their generated-source names.

    Returns
    -------
    str
        Hash of every value embedded in generated source.
    """
    # Extract equations from ParsedEquations or convert from provided tuple
    if hasattr(equations, "ordered"):
        eq_list = list(equations.ordered)
    else:
        eq_list = list(equations)

    # Sort equations alphabetically by LHS symbol name
    sorted_eqs = sorted(eq_list, key=lambda eq: str(eq[0]))

    # Join all equations into a single string and remove whitespace
    eq_strings = [f"{str(lhs)}={str(rhs)}" for lhs, rhs in sorted_eqs]
    dxdt_str = "|".join(eq_strings)
    normalized_dxdt = "".join(dxdt_str.split())

    # Append sorted constants labels. When constants vs parameters change,
    # we need to re-codegen. When values change, we just need to rebuild,
    # so this is handled in the config hash for caching.
    constants_str = ""
    if constants is not None:
        # Keys in `constants` may be SymPy Symbols (for example from
        # an index_map) as well as plain strings; SymPy Symbol keys are
        # not directly orderable, so str() is used to obtain a stable
        # string-based sort order for all key types.
        label_strings = [str(k) for k in constants.keys()]
        sorted_constants = sorted(label_strings)
        constants_str = "|".join(f"{label}" for label in sorted_constants)

    def ordered_labels(labels):
        if labels is None:
            return ""
        return "|".join(str(label) for label in labels)

    if derivative_names is None:
        derivative_names = getattr(equations, "derivative_names", None)
    derivatives_str = ""
    if derivative_names:
        derivatives_str = "|".join(
            f"{name}={derivative_names[name]}"
            for name in sorted(derivative_names)
        )
    if function_aliases is None:
        function_aliases = getattr(equations, "function_aliases", None)
    aliases_str = ""
    if function_aliases:
        aliases_str = "|".join(
            f"{name}={function_aliases[name]}"
            for name in sorted(function_aliases)
        )

    # Combine and hash
    combined = (
        f"dxdt:{normalized_dxdt}|constants:{constants_str}"
        f"|states:{ordered_labels(state_labels)}"
        f"|dxdt_layout:{ordered_labels(dxdt_labels)}"
        f"|parameters:{ordered_labels(parameter_labels)}"
        f"|drivers:{ordered_labels(driver_labels)}"
        f"|observables:{ordered_labels(observable_labels)}"
        f"|derivatives:{derivatives_str}"
        f"|function_aliases:{aliases_str}"
    )
    return sha256(combined.encode("utf-8")).hexdigest()


def prune_unused_assignments(
    expressions: Iterable[Tuple[sp.Symbol, sp.Expr]],
    outputsym_str: str = "jvp",
    output_symbols: Optional[Iterable[sp.Symbol]] = None,
) -> List[Tuple[sp.Symbol, sp.Expr]]:
    """Remove assignments that do not contribute to output symbols.

    Parameters
    ----------
    expressions
        Topologically sorted assignments ``(lhs, rhs)``.
    outputsym_str
        Prefix identifying output symbols by name convention
        (matched as ``f"{outputsym_str}["``). Ignored when
        ``output_symbols`` is provided. Defaults to ``"jvp"``.
    output_symbols
        Explicit collection of output symbols to retain. When
        supplied, ``outputsym_str`` is ignored.

    Returns
    -------
    list[tuple[sympy.Symbol, sympy.Expr]]
        Pruned assignments required to compute the output symbols.

    Notes
    -----
    Assumes topologically sorted input. Output symbols are detected by
    name convention (``"prefix["`` pattern) or via the explicit
    ``output_symbols`` set. Relative order of kept assignments is
    preserved.
    """
    exprs = list(expressions)
    if not exprs:
        return exprs

    lhs_symbols = [lhs for lhs, _ in exprs]
    all_lhs = set(lhs_symbols)

    # Detect outputs by name convention
    if output_symbols is not None:
        output_syms = set(output_symbols) & all_lhs
    else:
        output_syms = {
            lhs
            for lhs in lhs_symbols
            if str(lhs).startswith(f"{outputsym_str}[")
        }

    # If we can't detect outputs, do nothing
    if not output_syms:
        return exprs

    used: set[sp.Symbol] = set(output_syms)
    kept: list[Tuple[sp.Symbol, sp.Expr]] = []

    for lhs, rhs in reversed(exprs):
        if lhs in used:
            kept.append((lhs, rhs))
            # Only follow dependencies that are assigned to
            deps = rhs.free_symbols & all_lhs
            deps_syms = {s for s in deps if isinstance(s, sp.Symbol)}
            used.update(deps_syms)
    kept.reverse()
    return kept
