"""Assemble normalised systems into parser products.

:func:`assemble_simplified` structurally simplifies a normalised
system and maps the result into ``ParsedEquations``, storing the
derived mass matrix of a torn system in
``ParsedEquations.mass_matrix``. Equations are engine IR pairs
throughout; SymPy appears only in the ``all_symbols`` table consumed
by GUIs and device-function injection.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
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
    check_derivative_orders,
    derivative_helpers,
)
from cubie.odesystems.symbolic.structural.reassemble import (
    SimplifiedSystem,
)
from cubie.odesystems.symbolic.structural.simplify import (
    structural_simplify,
)
from cubie.odesystems.symbolic.structural.symbolics import (
    DerivativeRegistry,
    fixpoint_sub,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)
from cubie.odesystems.symbolic.sym_utils import hash_system_definition
from cubie._utils import devfunc_returns_nonfloat


def _mass_matrix_converter(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    array = np_array(value, dtype=np_float64)
    array.setflags(write=False)
    return array


@frozen
class SpecialisedSystem(CUDADispatcherCache):

    layout: SystemLayout = field()
    equations: ParsedEquations = field()
    functions: Dict[str, Callable] = field(factory=dict)
    fn_hash: str = field(default="")
    mass: Any = field(default=None, converter=_mass_matrix_converter)


def _driver_derivatives(
    simplified: SimplifiedSystem,
    registry: DerivativeRegistry,
    driver_names: Iterable[str],
) -> Dict[ir.Sym, Tuple[ir.Sym, int]]:
    """Map each driver derivative the reduced system reads.

    Parameters
    ----------
    simplified
        The structurally simplified system.
    registry
        Derivative registry recording each symbol's base and order.
    driver_names
        Names of the system's drivers.

    Returns
    -------
    dict
        Each driver derivative symbol mapped to ``(driver, order)``,
        sorted by driver name and then order.
    """

    expressions = [
        *simplified.dxdt.values(),
        *simplified.residuals,
        *(expr for _, expr in simplified.observed),
    ]
    atoms = set().union(*(ir.free_atoms(expr) for expr in expressions))
    drivers = {ir.sym(name) for name in driver_names}
    derivatives = {}
    for atom in atoms:
        driver, order = registry.base_and_order(atom)
        if order > 0 and driver in drivers:
            derivatives[atom] = (driver, order)
    return dict(
        sorted(
            derivatives.items(),
            key=lambda item: (item[1][0].name, item[1][1]),
        )
    )


def _finalise_symbols_and_products(
    equation_map,
    index_map,
    user_functions,
    user_function_derivatives,
    rename,
    derivative_names=None,
    mass_matrix=None,
    driver_derivatives=None,
):
    """Build ``all_symbols``, ``ParsedEquations``, and the hash."""

    functions = dict(user_functions or {})
    functions.update(
            {
                fn.__name__: fn
                for fn in derivative_helpers(user_function_derivatives)
            }
        )

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
    for func in derivative_helpers(user_function_derivatives):
        if devfunc_returns_nonfloat(func):
            nonfloat_functions.add(func.__name__)
    parsed_equations = ParsedEquations.from_equations(
        equation_map,
        index_map,
        derivative_names=derivative_names,
        function_aliases=function_aliases,
        nonfloat_functions=nonfloat_functions,
        mass_matrix=mass_matrix,
        driver_derivatives=driver_derivatives,
    )
    check_derivative_orders(
        (rhs for _, rhs in parsed_equations.ordered),
        function_aliases,
        parsed_equations.derivative_names,
        "reduced DAE",
    )
    fn_hash = hash_system_definition(
        parsed_equations,
        state_labels=index_map.state_names,
        dxdt_labels=index_map.dxdt_names,
        parameter_labels=index_map.parameter_names,
        driver_labels=index_map.driver_names,
        observable_labels=index_map.observables.ref_map.keys(),
        derivative_names=parsed_equations.derivative_names,
        function_aliases=parsed_equations.function_aliases,
        nonfloat_functions=parsed_equations.nonfloat_functions,
    )
    return functions, parsed_equations, fn_hash


def assemble_simplified(
    normalised: NormalisedSystem,
    states: Dict[str, float],
    observables: List[str],
    parameters: ParameterSet,
    driver_names: List[str],
    known_symbol_map: Dict[str, sp.Symbol],
    user_functions: Optional[Dict[str, Callable]],
    user_function_derivatives: Optional[Dict[str, Callable]],
    state_priority: Optional[Dict[str, float]] = None,
    irreducible: Optional[Iterable[str]] = None,
    state_units: Optional[Dict[str, str]] = None,
    observable_units: Optional[Dict[str, str]] = None,
    driver_units: Optional[Dict[str, str]] = None,
    simplify_options: Optional[Dict[str, Any]] = None,
) -> SpecialisedSystem:
    """Structurally simplify a normalised system and package it.

    ``parameters`` holds the swept parameters, in order. The others are
    already numbers in ``normalised``. Returns
    ``(index_map, all_symbols, funcs, parsed_equations, fn_hash)``,
    with the derived mass matrix in ``parsed_equations.mass_matrix``.
    Solver states are held in sorted name order.
    """

    priorities = {}
    for name, value in (state_priority or {}).items():
        priorities[ir.sym(str(name))] = value
    irreducible_syms = [
        ir.sym(str(name)) for name in (irreducible or [])
    ]

    structural_state = StructuralState(
        normalised.equations,
        normalised.registry,
        {ir.sym(name) for name in known_symbol_map},
        ir.sym("t"),
        derivative_names=normalised.derivative_names,
        drivers=[ir.sym(name) for name in driver_names],
        state_priorities=priorities,
        irreducibles=irreducible_syms,
    )
    simplified = structural_simplify(
        structural_state, **(simplify_options or {})
    )
    driver_derivatives = _driver_derivatives(
        simplified, normalised.registry, driver_names
    )

    # Sorted, matching the layout IndexedBaseMap builds.
    surviving = {
        sym.name: sym
        for sym in simplified.differential_states
        + simplified.algebraic_states
    }
    final_states = [surviving[name] for name in sorted(surviving)]

    final_state_names = [sym.name for sym in final_states]
    default_warned = [
        name for name in final_state_names if name not in states
    ]
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

    final_observables = sorted(final_observables)
    final_drivers = sorted(driver_names)
    driver_index = {name: index for index, name in enumerate(final_drivers)}
    layout = SystemLayout(
        states=VariableSet.of(Variable, final_state_names, state_units),
        parameters=parameters,
        observables=VariableSet.of(
            Variable, final_observables, observable_units
        ),
        drivers=VariableSet.of(Variable, final_drivers, driver_units),
        driver_derivatives=tuple(
            (driver_index[driver.name], int(order))
            for driver, order in driver_derivatives.values()
        ),
    )

    # Declared observables are pure outputs in cubie: the dxdt device
    # function excludes observable-LHS assignments and reads
    # observable symbols from a buffer that is stale during stage
    # evaluation. Inline their defining expressions into every
    # consuming equation so the dynamics never read that buffer.
    final_observable_set = set(final_observables)
    observable_sub = {
        sym: expr
        for sym, expr in simplified.observed
        if sym.name in final_observable_set
    }
    inline_memo = {}

    def _inline_observables(expr: ir.Expr) -> ir.Expr:
        return fixpoint_sub(expr, observable_sub, inline_memo)

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
    if simplified.residuals:
        n = len(final_states)
        mass = tuple(
            tuple(
                1.0 if (i == j and sym in diff_set) else 0.0
                for j in range(n)
            )
            for i, sym in enumerate(final_states)
        )

    functions, parsed_equations, fn_hash = _finalise_symbols_and_products(
            equation_map,
            index_map,
            user_functions,
            user_function_derivatives,
            normalised.rename,
            derivative_names=normalised.derivative_names,
            mass_matrix=mass,
            driver_derivatives=driver_derivatives,
    )
    # Inlined non-device callables keep their entries.
    functions = {**functions, **(normalised.funcs or {})}
    return SpecialisedSystem(
        layout=layout,
        equations=parsed_equations,
        functions=functions,
        fn_hash=fn_hash,
        mass=mass,
    )
