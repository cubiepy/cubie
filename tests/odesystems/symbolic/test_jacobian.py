import sympy as sp

from cubie.odesystems.symbolic.codegen.jacobian import (
    _cache,
    generate_analytical_jvp,
    generate_jacobian,
)
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine import to_sympy
from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.symbolic.parsing import ParsedEquations
from cubie.odesystems.symbolic.engine.adapter import buffer_maps


def _jac_matrix(jac):
    """Convert IR Jacobian rows to a SymPy matrix for comparison."""
    return sp.Matrix(
        [[to_sympy(entry) for entry in row] for row in jac]
    )


def _clear_cache():
    """Clear the Jacobian/JVP module cache between tests."""
    _cache.clear()


def _get_cache_counts():
    """Count cached Jacobian and JVP artifacts."""
    counts = {"jac": 0, "jvp": 0}
    for value in _cache.values():
        if isinstance(value, dict):
            if "jac" in value:
                counts["jac"] += 1
            if "jvp" in value:
                counts["jvp"] += 1
    return counts


def test_generate_jacobian_with_auxiliary():
    """Jacobian matches expected expressions with auxiliaries."""

    layout = SystemLayout(states=("x", "y"), parameters=("a", "b"))
    x, y = [sp.Symbol(name, real=True) for name in layout.state_names]
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    dx, dy = [sp.Symbol(name, real=True) for name in layout.dxdt_names]
    aux = sp.Symbol("aux", real=True)
    equations = [
        (aux, a * x + b * y),
        (dx, aux - x),
        (dy, -aux + y),
    ]
    parsed = ParsedEquations.from_equations(equations, layout)
    jac = generate_jacobian(
        parsed,
        buffer_maps(layout).state_order,
        buffer_maps(layout).dxdt_order,
    )
    expected = sp.Matrix([[a - 1, b], [-a, -b + 1]])
    assert _jac_matrix(jac).equals(expected)


def test_generate_jacobian_coupled_nonlinear():
    """Jacobian of a coupled nonlinear system matches full expression."""

    layout = SystemLayout(states=("x0", "x1"))
    x0, x1 = [sp.Symbol(name, real=True) for name in layout.state_names]
    dx0, dx1 = [sp.Symbol(name, real=True) for name in layout.dxdt_names]
    equations = [
        (dx0, sp.sin(x0) + x0 * x1 ** 2),
        (dx1, x0 ** 2 + sp.exp(x1)),
    ]
    parsed = ParsedEquations.from_equations(equations, layout)
    jac = generate_jacobian(
        parsed,
        buffer_maps(layout).state_order,
        buffer_maps(layout).dxdt_order,
    )
    expected = sp.Matrix(
        [
            [sp.cos(x0) + x1 ** 2, 2 * x0 * x1],
            [2 * x0, sp.exp(x1)],
        ]
    )
    assert _jac_matrix(jac).equals(expected)


def test_jacobian_caching():
    """Repeated calls reuse cached Jacobian and JVP."""

    layout = SystemLayout(states=("x", "y"))
    x, y = [sp.Symbol(name, real=True) for name in layout.state_names]
    dx = sp.Symbol(layout.dxdt_names[0], real=True)
    equations = [(dx, x + y)]
    parsed = ParsedEquations.from_equations(equations, layout)
    _clear_cache()
    generate_jacobian(
        parsed, buffer_maps(layout).state_order, buffer_maps(layout).dxdt_order
    )
    generate_analytical_jvp(
        parsed, buffer_maps(layout).state_order, buffer_maps(layout).dxdt_order
    )
    counts = _get_cache_counts()
    assert counts == {"jac": 1, "jvp": 1}
    generate_jacobian(
        parsed, buffer_maps(layout).state_order, buffer_maps(layout).dxdt_order
    )
    generate_analytical_jvp(
        parsed, buffer_maps(layout).state_order, buffer_maps(layout).dxdt_order
    )
    counts2 = _get_cache_counts()
    assert counts2 == counts


def test_jvp_graph_defines_every_jacobian_entry():
    """Every nonzero entry symbol has a defining graph assignment."""

    layout = SystemLayout(
        states=("x", "y"), parameters=("a",), observables=("obs1",)
    )
    x, y = [sp.Symbol(name, real=True) for name in layout.state_names]
    a = sp.Symbol("a", real=True)
    obs = sp.Symbol("obs1", real=True)
    dx, dy = [sp.Symbol(name, real=True) for name in layout.dxdt_names]
    aux = sp.Symbol("aux", real=True)
    equations = [
        (aux, x * y),
        (obs, sp.sin(aux) + a * x),
        (dx, obs + x ** 2),
        (dy, obs * y),
    ]
    parsed = ParsedEquations.from_equations(equations, layout)
    jvp = generate_analytical_jvp(
        parsed,
        buffer_maps(layout).state_order,
        buffer_maps(layout).dxdt_order,
        observables=buffer_maps(layout).observable_symbols,
    )
    entries = jvp.jacobian_entry_symbols
    assert len(entries) == 4
    for (row, col), symbol in entries.items():
        assert symbol.name == f"_cubie_codegen_j_{row}_{col}"
        assert symbol in jvp.non_jvp_exprs
        assert jvp.jacobian_entry(row, col) is symbol


def test_structurally_zero_jacobian_entry_is_zero():
    """A structurally zero entry reads back as the zero literal."""

    layout = SystemLayout(states=("x", "y"), parameters=("a",))
    x, y = [sp.Symbol(name, real=True) for name in layout.state_names]
    a = sp.Symbol("a", real=True)
    dx, dy = [sp.Symbol(name, real=True) for name in layout.dxdt_names]
    equations = [
        (dx, a * y),
        (dy, a * x),
    ]
    parsed = ParsedEquations.from_equations(equations, layout)
    jvp = generate_analytical_jvp(
        parsed,
        buffer_maps(layout).state_order,
        buffer_maps(layout).dxdt_order,
    )
    assert jvp.jacobian_entry(0, 0) is ir.ZERO
    assert jvp.jacobian_entry(1, 1) is ir.ZERO
    assert jvp.jacobian_entry(0, 1) is jvp.jacobian_entry_symbols[(0, 1)]
    assert jvp.jacobian_entry(1, 0) is jvp.jacobian_entry_symbols[(1, 0)]
