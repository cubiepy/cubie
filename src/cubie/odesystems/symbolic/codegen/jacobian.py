"""Analytic Jacobian and JVP expression generation on the IR engine.

Derives Jacobian and Jacobian-vector-product expressions from parsed
ODE equations by chain-rule differentiation over the auxiliary
assignment graph. All symbolic compute runs on the engine IR;
SymPy inputs are converted once at entry.

Published Functions
-------------------
:func:`generate_analytical_jvp`
    Differentiate the parsed equations to produce a
    :class:`~cubie.odesystems.symbolic.parsing.jvp_equations.JVPEquations`
    instance containing ordered JVP assignments.

:func:`generate_jacobian`
    Compute the full Jacobian as a row-major list of IR expression
    rows.

:func:`get_cache_key`
    Build a hashable key for the module-level Jacobian cache.

See Also
--------
:class:`~cubie.odesystems.symbolic.parsing.jvp_equations.JVPEquations`
    Container returned by :func:`generate_analytical_jvp`.
:mod:`cubie.odesystems.symbolic.codegen.linear_operators`
    Consumes JVP expressions to generate linear operator code.
"""

from typing import Dict, Iterable, List, Optional, Tuple


from cubie.odesystems.symbolic.engine import expr as ir
from cubie._env import operation_ordering_default
from cubie.odesystems.symbolic.engine.assignments import (
    cse_and_stack,
    prune_unused,
    topological_sort,
)
from cubie.odesystems.symbolic.engine.from_sympy import (
    convert_assignments,
    derivative_name_map,
)
from cubie.odesystems.symbolic.parsing import JVPEquations

CacheKey = Tuple[
    Tuple[Tuple[ir.Expr, ir.Expr], ...],
    Tuple[Tuple[ir.Sym, int], ...],
    Tuple[Tuple[ir.Sym, int], ...],
    bool,
    str,
    Tuple[Tuple[str, str], ...],
]

_cache: Dict[CacheKey, Dict[str, object]] = {}


def _ir_order(order: Dict) -> Dict[ir.Sym, int]:
    """Convert a SymPy ``symbol -> position`` map to IR symbols."""
    converted = {}
    for symbol, position in order.items():
        if isinstance(symbol, ir.Sym):
            converted[symbol] = position
        else:
            converted[ir.sym(str(symbol))] = position
    return converted


def _ir_equations(equations) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Convert equations (ParsedEquations or pair iterable) to IR."""
    if hasattr(equations, "to_equation_list"):
        eq_list = equations.to_equation_list()
    else:
        eq_list = list(equations)
    if eq_list and isinstance(eq_list[0][0], ir.Expr):
        return eq_list
    return convert_assignments(eq_list)


def _sympy_equation_list(equations) -> list:
    """Return the SymPy pair list backing ``equations`` when present."""
    if hasattr(equations, "to_equation_list"):
        return equations.to_equation_list()
    eq_list = list(equations)
    if eq_list and isinstance(eq_list[0][0], ir.Expr):
        return []
    return eq_list


def _derivative_names(equations) -> Dict[str, str]:
    """Return user derivative placeholder names for ``equations``.

    IR-native parser output carries the map on the container; SymPy
    pairs recover it from the applied functions' ``fdiff`` classes.
    """
    names = getattr(equations, "derivative_names", None)
    if names:
        return dict(names)
    return derivative_name_map(_sympy_equation_list(equations))


def get_cache_key(
    equations: Iterable[Tuple[ir.Expr, ir.Expr]],
    input_order: Dict[ir.Sym, int],
    output_order: Dict[ir.Sym, int],
    cse: bool,
    operation_ordering: str = operation_ordering_default(),
    derivative_names: Optional[Dict[str, str]] = None,
) -> CacheKey:
    """Generate the cache key from IR equations, orders, and CSE flag.

    Parameters
    ----------
    equations
        IR assignment pairs (interned nodes, so tuples are hashable).
    input_order
        Mapping from each input symbol to its position.
    output_order
        Mapping from each output symbol to its position.
    cse
        Whether common-subexpression elimination is enabled.
    derivative_names
        User-function derivative placeholder names. Included so two
        systems whose equations intern identically but bind
        different derivative helpers to a same-named function do
        not share a cache entry.

    Returns
    -------
    CacheKey
        Hashable representation of the computation inputs.
    """
    if isinstance(equations, dict):
        eq_tuple = tuple(equations.items())
    else:
        eq_tuple = tuple(tuple(pair) for pair in equations)
    input_tuple = tuple(input_order.items())
    output_tuple = tuple(output_order.items())
    names_tuple = tuple(sorted((derivative_names or {}).items()))
    return (
        eq_tuple,
        input_tuple,
        output_tuple,
        bool(cse),
        operation_ordering,
        names_tuple,
    )


def _chain_rule_jacobian(
    eq_list: List[Tuple[ir.Expr, ir.Expr]],
    input_order: Dict[ir.Sym, int],
    output_order: Dict[ir.Sym, int],
    derivative_names: Dict[str, str],
    operation_ordering: str = operation_ordering_default(),
) -> List[List[ir.Expr]]:
    """Build the full Jacobian via chain rule over auxiliaries.

    Returns
    -------
    list of list
        Row-major Jacobian: ``jac[output_index][input_index]``.
    """
    sorted_inputs = sorted(
        input_order.keys(), key=lambda symbol: input_order[symbol]
    )
    output_symbols = set(output_order.keys())
    num_in = len(sorted_inputs)

    ordered = topological_sort(eq_list, operation_ordering)
    auxiliary_equations = [
        (lhs, rhs) for lhs, rhs in ordered if lhs not in output_symbols
    ]
    aux_symbols = {lhs for lhs, _ in auxiliary_equations}
    output_equations = [
        (lhs, rhs) for lhs, rhs in ordered if lhs in output_symbols
    ]

    # One diff memo per input symbol, shared across every expression,
    # so repeated subtrees differentiate once.
    diff_memos: List[Dict] = [dict() for _ in range(num_in)]

    def gradient(expression: ir.Expr) -> List[ir.Expr]:
        return [
            ir.diff(
                expression,
                in_sym,
                memo=diff_memos[j],
                derivative_names=derivative_names,
            )
            for j, in_sym in enumerate(sorted_inputs)
        ]

    auxiliary_gradients: Dict[ir.Expr, List[ir.Expr]] = {}
    for aux_sym, expression in auxiliary_equations:
        direct = gradient(expression)
        chain = [ir.ZERO] * num_in
        for other_sym in sorted(
            ir.free_atoms(expression) & aux_symbols,
            key=lambda node: node.sort_key,
        ):
            partial = ir.diff(
                expression,
                other_sym,
                derivative_names=derivative_names,
            )
            other_grad = auxiliary_gradients[other_sym]
            chain = [
                ir.add(chain[j], ir.mul(partial, other_grad[j]))
                for j in range(num_in)
            ]
        auxiliary_gradients[aux_sym] = [
            ir.add(direct[j], chain[j]) for j in range(num_in)
        ]

    num_out = len(output_symbols)
    jac: List[List[ir.Expr]] = [
        [ir.ZERO] * num_in for _ in range(num_out)
    ]
    for out_sym, out_expr in output_equations:
        direct_row = gradient(out_expr)
        chain_row = [ir.ZERO] * num_in
        for aux_sym in sorted(
            ir.free_atoms(out_expr) & aux_symbols,
            key=lambda node: node.sort_key,
        ):
            partial = ir.diff(
                out_expr,
                aux_sym,
                derivative_names=derivative_names,
            )
            aux_grad = auxiliary_gradients[aux_sym]
            chain_row = [
                ir.add(chain_row[j], ir.mul(partial, aux_grad[j]))
                for j in range(num_in)
            ]
        jac[output_order[out_sym]] = [
            ir.add(direct_row[j], chain_row[j])
            for j in range(num_in)
        ]
    return jac


def generate_jacobian(
    equations,
    input_order: Dict,
    output_order: Dict,
    use_cache: bool = True,
    cache_cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> List[List[ir.Expr]]:
    """Return the Jacobian for the given equations as IR rows.

    Parameters
    ----------
    equations
        Parsed equations (SymPy ``ParsedEquations`` or IR pairs).
    input_order
        Mapping from each input symbol to its position in the input
        vector (SymPy or IR symbols).
    output_order
        Mapping from each output symbol to its position in the output
        vector (SymPy or IR symbols).
    use_cache
        Whether to reuse cached Jacobian computations when available.
    cache_cse
        CSE flag folded into the cache key (the Jacobian itself is
        CSE-independent).

    Returns
    -------
    list of list
        Row-major Jacobian of IR expressions,
        ``jac[output_index][input_index]``.
    """
    eq_list = _ir_equations(equations)
    ir_inputs = _ir_order(input_order)
    ir_outputs = _ir_order(output_order)
    derivative_names = _derivative_names(equations)

    cache_key = None
    if use_cache:
        cache_key = get_cache_key(
            eq_list,
            ir_inputs,
            ir_outputs,
            cse=cache_cse,
            operation_ordering=operation_ordering,
            derivative_names=derivative_names,
        )
        cached_entry = _cache.get(cache_key)
        if isinstance(cached_entry, dict) and "jac" in cached_entry:
            return cached_entry["jac"]

    jac = _chain_rule_jacobian(
        eq_list,
        ir_inputs,
        ir_outputs,
        derivative_names,
        operation_ordering,
    )

    if use_cache and cache_key is not None:
        entry = _cache.setdefault(cache_key, {})
        entry["jac"] = jac
    return jac


def generate_analytical_jvp(
    equations,
    input_order: Dict,
    output_order: Dict,
    observables: Optional[Iterable] = None,
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> JVPEquations:
    """Return structured assignments for the Jacobian-vector product.

    Parameters
    ----------
    equations
        Parsed equations including intermediates and outputs.
    input_order
        Mapping from each input symbol to its position in the input
        vector.
    output_order
        Mapping from each output symbol to its position in the output
        vector.
    observables
        Symbols renamed to ``_cubie_codegen_aux_<n>`` auxiliaries
        before differentiation.
    cse
        Apply common-subexpression elimination before producing
        assignments.

    Returns
    -------
    JVPEquations
        Structured assignments and dependency metadata for the JVP.
    """
    eq_list = _ir_equations(equations)
    ir_inputs = _ir_order(input_order)
    ir_outputs = _ir_order(output_order)
    derivative_names = _derivative_names(equations)

    obs_subs: Dict[ir.Expr, ir.Expr] = {}
    if observables is not None:
        for position, obs in enumerate(observables):
            obs_sym = (
                obs
                if isinstance(obs, ir.Sym)
                else ir.sym(str(obs))
            )
            obs_subs[obs_sym] = ir.sym(
                f"_cubie_codegen_aux_{position + 1}"
            )

    if obs_subs:
        memo: Dict = {}
        substituted = [
            (
                ir.xreplace(lhs, obs_subs, memo),
                ir.xreplace(rhs, obs_subs, memo),
            )
            for lhs, rhs in eq_list
        ]
    else:
        substituted = list(eq_list)

    cache_key = get_cache_key(
        substituted,
        ir_inputs,
        ir_outputs,
        cse=cse,
        operation_ordering=operation_ordering,
        derivative_names=derivative_names,
    )
    cached_entry = _cache.get(cache_key)
    if isinstance(cached_entry, dict) and "jvp" in cached_entry:
        return cached_entry["jvp"]

    n_inputs = len(ir_inputs)
    jac = _cached_jacobian_for(
        substituted,
        ir_inputs,
        ir_outputs,
        cse,
        derivative_names,
        operation_ordering,
    )

    prod_exprs: List[Tuple[ir.Expr, ir.Expr]] = []
    j_symbols: Dict[Tuple[int, int], ir.Sym] = {}
    n_outputs = len(ir_outputs)
    for i in range(n_outputs):
        for j in range(n_inputs):
            entry = jac[i][j]
            if entry is ir.ZERO:
                continue
            # Separators keep multi-digit indices unambiguous.
            j_sym = ir.sym(f"_cubie_codegen_j_{i}_{j}")
            prod_exprs.append((j_sym, entry))
            j_symbols[(i, j)] = j_sym

    sorted_outputs = sorted(
        ir_outputs.keys(), key=lambda symbol: ir_outputs[symbol]
    )
    for out_sym in sorted_outputs:
        i = ir_outputs[out_sym]
        terms = []
        for j in range(n_inputs):
            j_sym = j_symbols.get((i, j))
            if j_sym is not None:
                terms.append(ir.mul(j_sym, ir.arr("v", j)))
        prod_exprs.append((ir.arr("jvp", i), ir.add(*terms)))

    exprs = [
        pair for pair in substituted if pair[0] not in ir_outputs
    ]
    all_exprs = exprs + prod_exprs

    if cse:
        all_exprs = cse_and_stack(
            all_exprs,
            operation_ordering=operation_ordering,
        )
    else:
        all_exprs = topological_sort(
            all_exprs,
            operation_ordering=operation_ordering,
        )

    all_exprs = prune_unused(all_exprs, output_name="jvp")

    equations_obj = JVPEquations(all_exprs)
    entry = _cache.setdefault(cache_key, {})
    entry["jvp"] = equations_obj
    return equations_obj


def _cached_jacobian_for(
    substituted: List[Tuple[ir.Expr, ir.Expr]],
    ir_inputs: Dict[ir.Sym, int],
    ir_outputs: Dict[ir.Sym, int],
    cse: bool,
    derivative_names: Dict[str, str],
    operation_ordering: str = operation_ordering_default(),
) -> List[List[ir.Expr]]:
    """Return the Jacobian for pre-substituted IR equations, cached."""
    cache_key = get_cache_key(
        substituted,
        ir_inputs,
        ir_outputs,
        cse=cse,
        operation_ordering=operation_ordering,
        derivative_names=derivative_names,
    )
    cached_entry = _cache.get(cache_key)
    if isinstance(cached_entry, dict) and "jac" in cached_entry:
        return cached_entry["jac"]
    jac = _chain_rule_jacobian(
        substituted,
        ir_inputs,
        ir_outputs,
        derivative_names,
        operation_ordering,
    )
    entry = _cache.setdefault(cache_key, {})
    entry["jac"] = jac
    return jac
