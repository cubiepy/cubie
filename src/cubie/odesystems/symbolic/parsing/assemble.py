"""Assemble normalised systems into parser products.

:func:`assemble_simplified` structurally simplifies a normalised
system and maps the result into ``ParsedEquations``; the derived
mass matrix for torn systems rides on
``ParsedEquations.mass_matrix``. Equations are engine IR pairs
throughout; SymPy appears only in the ``all_symbols`` table consumed
by GUIs and device-function injection.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional
from warnings import warn

import sympy as sp

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie.odesystems.symbolic.parsing.normalise import (
    NormalisedSystem,
)
from cubie.odesystems.symbolic.parsing.parse_primitives import (
    EquationWarning,
    ParsedEquations,
    TIME_SYMBOL,
)
from cubie.odesystems.symbolic.structural.simplify import (
    structural_simplify,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)
from cubie.odesystems.symbolic.sym_utils import hash_system_definition
from cubie.cuda_simsafe import devfunc_returns_nonfloat


def _observable_substitutions(
    definitions: List,
) -> Dict[ir.Expr, ir.Expr]:
    """Fully expand observable definitions against each other.

    ``definitions`` is a list of ``(symbol, expression)`` IR pairs.
    Raises when the definitions contain a cycle.
    """

    subs = {sym: expr for sym, expr in definitions}
    for _ in range(len(subs) + 1):
        expanded = {
            sym: ir.xreplace(expr, subs)
            for sym, expr in subs.items()
        }
        if expanded == subs:
            return subs
        subs = expanded
    raise AssertionError(
        "observable inlining did not converge; the observable "
        "definitions contain a cycle"
    )


def _finalise_symbols_and_products(
    equation_map,
    index_map,
    user_functions,
    user_function_derivatives,
    rename,
    derivative_names=None,
    extra_symbol_names=(),
    mass_matrix=None,
):
    """Build ``all_symbols``, ``ParsedEquations``, and the hash."""

    all_symbols = index_map.all_symbols.copy()
    all_symbols.setdefault("t", TIME_SYMBOL)
    for name in extra_symbol_names:
        all_symbols.setdefault(name, sp.Symbol(name, real=True))

    if user_functions:
        all_symbols.update(
            {name: fn for name, fn in user_functions.items()}
        )
        if user_function_derivatives:
            all_symbols.update(
                {
                    fn.__name__: fn
                    for fn in user_function_derivatives.values()
                    if callable(fn)
                }
            )
        if rename:
            all_symbols["__function_aliases__"] = {
                v: k for k, v in rename.items()
            }

    function_aliases = {
        name: name for name in (user_functions or {})
    }
    function_aliases.update(
        {renamed: original for original, renamed in (rename or {}).items()}
    )
    nonfloat_functions = set()
    for orig_name, func in (user_functions or {}).items():
        if devfunc_returns_nonfloat(func):
            nonfloat_functions.add(orig_name)
            renamed = (rename or {}).get(orig_name)
            if renamed:
                nonfloat_functions.add(renamed)
    for func in (user_function_derivatives or {}).values():
        if callable(func) and devfunc_returns_nonfloat(func):
            nonfloat_functions.add(func.__name__)
    parsed_equations = ParsedEquations.from_equations(
        equation_map,
        index_map,
        derivative_names=derivative_names,
        function_aliases=function_aliases,
        nonfloat_functions=nonfloat_functions,
        mass_matrix=mass_matrix,
    )
    fn_hash = hash_system_definition(
        parsed_equations,
        index_map.constants.default_values,
        state_labels=index_map.state_names,
        dxdt_labels=index_map.dxdt_names,
        parameter_labels=index_map.parameter_names,
        driver_labels=index_map.driver_names,
        observable_labels=index_map.observables.ref_map.keys(),
        derivative_names=parsed_equations.derivative_names,
        function_aliases=parsed_equations.function_aliases,
        nonfloat_functions=parsed_equations.nonfloat_functions,
    )
    return all_symbols, parsed_equations, fn_hash


def assemble_simplified(
    normalised: NormalisedSystem,
    states: Dict[str, float],
    observables: List[str],
    parameters: Dict[str, float],
    constants: Dict[str, float],
    driver_names: List[str],
    driver_dict: Optional[Dict[str, Any]],
    known_symbol_map: Dict[str, sp.Symbol],
    user_functions: Optional[Dict[str, Callable]],
    user_function_derivatives: Optional[Dict[str, Callable]],
    state_priority: Optional[Dict[str, float]] = None,
    irreducible: Optional[Iterable[str]] = None,
    state_units=None,
    parameter_units=None,
    constant_units=None,
    observable_units=None,
    driver_units=None,
    simplify_options: Optional[Dict[str, Any]] = None,
):
    """Structurally simplify a normalised system and package it.

    Returns ``(index_map, all_symbols, funcs, parsed_equations,
    fn_hash)``; the derived mass matrix rides on
    ``parsed_equations.mass_matrix``. Solver states are held in sorted
    name order.
    """

    parameters = dict(parameters)
    for name in normalised.new_params:
        parameters.setdefault(name, 0.0)
        known_symbol_map.setdefault(
            name, sp.Symbol(name, real=True)
        )

    unknown_syms = [
        ir.sym(name) for name in sorted(normalised.unknown_names)
    ]
    priorities = {}
    for name, value in (state_priority or {}).items():
        priorities[ir.sym(str(name))] = value
    irreducible_syms = [
        ir.sym(str(name)) for name in (irreducible or [])
    ]

    structural_state = StructuralState(
        normalised.equations,
        unknown_syms,
        normalised.registry,
        {ir.sym(name) for name in known_symbol_map},
        ir.sym("t"),
        state_priorities=priorities,
        irreducibles=irreducible_syms,
    )
    simplified = structural_simplify(
        structural_state, **(simplify_options or {})
    )

    # Sorted, matching the layout IndexedBaseMap builds.
    surviving = {sym.name: sym for sym in simplified.states}
    final_states = [surviving[name] for name in sorted(surviving)]

    final_state_values = {}
    default_warned = []
    for sym in final_states:
        if sym.name in states:
            final_state_values[sym.name] = states[sym.name]
        else:
            final_state_values[sym.name] = 0.0
            default_warned.append(sym.name)
    if default_warned:
        warn(
            f"States {default_warned} were introduced by structural "
            "simplification and default to initial value 0.0. "
            "Consistent values are solved at the start of each run "
            "(see dae_initialisation); values set through the solver "
            "serve as the starting guess for that solve.",
            EquationWarning,
        )

    observed_lhs_names = {sym.name for sym, _ in simplified.observed}
    eliminated = [
        name
        for name in states
        if name not in surviving and name not in observables
    ]
    if eliminated:
        reduced = [
            name for name in eliminated if name in observed_lhs_names
        ]
        removed = [
            name
            for name in eliminated
            if name not in observed_lhs_names
        ]
        if reduced:
            warn(
                f"States {reduced} were eliminated by structural "
                "simplification (they reduce to functions of the "
                "remaining states); their initial values are "
                "ignored. Declare them as observables to record "
                "their trajectories.",
                EquationWarning,
            )
        if removed:
            warn(
                f"States {removed} were removed entirely by "
                "structural simplification; their initial values "
                "are ignored.",
                EquationWarning,
            )

    final_state_names = set(final_state_values)
    final_observables = []
    for name in observables:
        if name in final_state_names:
            warn(
                f"Declared observable {name} was selected as a "
                "solver state; its trajectory is available as a "
                "state output.",
                EquationWarning,
            )
            continue
        if name not in observed_lhs_names:
            raise ValueError(
                f"Observable {name} was declared but has no "
                "defining equation after simplification."
            )
        final_observables.append(name)

    if state_units is not None:
        if not isinstance(state_units, dict):
            state_units = dict(zip(states, state_units))
        state_units = {
            k: v
            for k, v in state_units.items()
            if k in final_state_names
        }

    index_map = IndexedBases.from_user_inputs(
        final_state_values,
        parameters,
        constants,
        final_observables,
        driver_names,
        state_units=state_units,
        parameter_units=parameter_units,
        constant_units=constant_units,
        observable_units=observable_units,
        driver_units=driver_units,
    )
    if driver_dict is not None:
        index_map.drivers.set_passthrough_defaults(driver_dict)

    # Declared observables are pure outputs in cubie: the dxdt device
    # function excludes observable-LHS assignments and reads
    # observable symbols from a buffer that is stale during stage
    # evaluation. Inline their defining expressions into every
    # consuming equation so the dynamics never read that buffer.
    final_observable_set = set(final_observables)
    observable_sub = _observable_substitutions(
        [
            (sym, expr)
            for sym, expr in simplified.observed
            if sym.name in final_observable_set
        ]
    )
    inline_memo = {}

    def _inline_observables(expr: ir.Expr) -> ir.Expr:
        return ir.xreplace(expr, observable_sub, inline_memo)

    diff_set = set(simplified.differential_states)
    residual_for = dict(
        zip(simplified.algebraic_states, simplified.residuals)
    )
    equation_map = []
    for sym in final_states:
        lhs = ir.sym(f"d{sym.name}")
        if sym in diff_set:
            equation_map.append(
                (lhs, _inline_observables(simplified.dxdt[sym]))
            )
        elif sym in residual_for:
            equation_map.append(
                (lhs, _inline_observables(residual_for[sym]))
            )
    for sym, expr in simplified.observed:
        if sym in observable_sub:
            # Declared observables keep their chained form; the
            # observables pass evaluates them in topological order.
            equation_map.append((sym, expr))
        else:
            equation_map.append((sym, _inline_observables(expr)))

    # Mass over the final state order: 1 diagonal for differential
    # states, 0 for torn algebraic rows.
    mass = None
    if simplified.mass_matrix is not None:
        n = len(final_states)
        mass = tuple(
            tuple(
                1.0 if (i == j and sym in diff_set) else 0.0
                for j in range(n)
            )
            for i, sym in enumerate(final_states)
        )

    observed_names = [sym.name for sym, _ in simplified.observed]
    all_symbols, parsed_equations, fn_hash = (
        _finalise_symbols_and_products(
            equation_map,
            index_map,
            user_functions,
            user_function_derivatives,
            normalised.rename,
            derivative_names=normalised.derivative_names,
            extra_symbol_names=observed_names,
            mass_matrix=mass,
        )
    )
    return (
        index_map,
        all_symbols,
        normalised.funcs,
        parsed_equations,
        fn_hash,
    )
