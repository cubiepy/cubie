import numpy as np
import pytest
import sympy as sp

from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.symbolic.parsing import ParsedEquations
from cubie.odesystems.symbolic.symbolicODE import (
    SymbolicODE,
    create_ODE_system,
)
from tests._utils import sweep


@pytest.fixture(scope="session")
def torn_dae_system():
    """Torn index-1 DAE (dx = -z under z**5 + z = x), float64."""
    return create_ODE_system(
        dxdt="""
        dx = -z
        0 = z**5 + z - x
        """,
        states={"x": 2.0, "z": 1.0},
        precision=np.float64,
        name="torn_dae",
    )


@pytest.fixture(scope="session")
def simple_system_defaults():
    states = {"one": 0.9, "foo": 0.5}
    parameters = {"zebra": 0.2, "fox": 0.4, "apple": 0.43, "linen": 0.32}
    drivers = ["driver1"]
    observables = ["safari", "zoo"]

    # Out of order, aux-refs-aux, dx-refs-dx
    dxdt_str = """safari = linen * fox + apple * zebra + zoo
    zoo = linen ** 2 * fox + apple * one**2 - zebra * foo + driver1
    uninited = zoo**2 + safari**2
    done = uninited*2 + dfoo
    dfoo = zoo + safari"""
    dxdt_list = [
        "safari = linen * fox + apple * zebra + zoo",
        "zoo = linen ** 2 * fox + apple * one**2 - zebra * foo + driver1",
        "uninited = zoo**2 + safari**2",
        "done = uninited*2 + dfoo",
        "dfoo = zoo + safari",
    ]

    return (
        states,
        parameters,
        drivers,
        observables,
        dxdt_str,
        dxdt_list,
    )


@pytest.fixture
def simple_symbols():
    """Basic SymPy symbols for testing."""
    x, y, z = sp.symbols("x y z", real=True)
    a, b, c = sp.symbols("a b c", real=True)
    return {"states": [x, y], "params": [a, b, c], "aux": [z]}


@pytest.fixture
def simple_equations(layout):
    """Simple symbolic equations for testing."""
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)

    equations = [
        (dx, -a * x + b * y),
        (dy, a * x - b * y),
    ]
    return ParsedEquations.from_equations(equations, layout)


@pytest.fixture
def complex_equations(layout):
    """More complex equations with auxiliary variables."""
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    c = sp.Symbol("c", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)
    obs = sp.Symbol("obs1", real=True)

    aux = sp.Symbol("aux", real=True)
    equations = [
        (aux, a * x + b * y),
        (dx, -aux + c),
        (dy, aux - c * y),
        (obs, x + y),
    ]
    return ParsedEquations.from_equations(equations, layout)


@pytest.fixture(scope="session")
def observables_kernel_system(precision):
    """Return a ``SymbolicODE`` used for observables parity tests."""

    dxdt_lines = [
        "obs_rate = alpha * x + c0",
        "obs_total = obs_rate + beta * y + drive",
        "dx = obs_total - y + alpha * drive",
        "dy = obs_rate * x + c0",
    ]

    system = SymbolicODE.create(
        dxdt=dxdt_lines,
        states={"x": precision(0.0), "y": precision(0.0)},
        parameters={
            "alpha": precision(0.0),
            "beta": precision(0.0),
            "c0": precision(1.1),
        },
        drivers={"drive": precision(0.0)},
        observables=["obs_rate", "obs_total"],
        precision=precision,
        strict=True,
        name="observables_kernel_system",
    )
    sweep(system, ["alpha", "beta"])

    return system


@pytest.fixture
def layout():
    """Sample layout for testing."""
    return SystemLayout(
        states=("x", "y"),
        parameters=("a", "b", "c", "d"),
        observables=("obs1",),
        drivers=("driver1",),
    )


@pytest.fixture
def sample_hash():
    """Sample hash string for testing."""
    return "# hash: test_hash_123456"


@pytest.fixture
def linear_system_equations(layout):
    """Linear system equations for Jacobian testing."""
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    c = sp.Symbol("c", real=True)
    d = sp.Symbol("d", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)

    equations = [
        (dx, a * x + b * y),
        (dy, c * x + d * y),
    ]
    return ParsedEquations.from_equations(equations, layout)


@pytest.fixture
def nonlinear_equations(layout):
    """Nonlinear equations for comprehensive testing."""
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)

    equations = [
        (dx, a * x - b * x * y),
        (dy, b * x * y - a * y),
    ]
    return ParsedEquations.from_equations(equations, layout)


@pytest.fixture
def bare_layout():
    """A layout with two states and two parameters only.

    No observables or drivers: codegen tests use this to
    reach the no-observable and no-driver generator branches.
    """
    return SystemLayout(states=("x", "y"), parameters=("a", "b"))


@pytest.fixture(scope="session")
def solver_scaling_collision_layout():
    """A layout with parameters named like solver scalings."""
    return SystemLayout(states=("x",), parameters=("beta", "gamma"))


@pytest.fixture(scope="session")
def solver_scaling_collision_equations(
    solver_scaling_collision_layout,
):
    """One-state equation using colliding beta/gamma parameters."""
    ib = solver_scaling_collision_layout
    x = sp.Symbol("x", real=True)
    beta = sp.Symbol("beta", real=True)
    gamma = sp.Symbol("gamma", real=True)
    dx = sp.Symbol("dx", real=True)
    return ParsedEquations.from_equations(
        [(dx, beta * x + gamma)],
        ib,
    )


@pytest.fixture
def bare_nonlinear_equations(bare_layout):
    """Two-state nonlinear equations with no cacheable auxiliaries."""
    ib = bare_layout
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)

    equations = [
        (dx, a * x - b * x * y),
        (dy, b * x * y - a * y),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def zero_diagonal_equations(bare_layout):
    """Two cross-coupled states with vanishing Jacobian diagonals."""
    ib = bare_layout
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)

    equations = [
        (dx, a * y),
        (dy, b * x),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def identical_diagonal_equations(bare_layout):
    """Two cross-coupled states sharing one Jacobian diagonal."""
    ib = bare_layout
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)

    equations = [
        (dx, a * y - x),
        (dy, b * x - y),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def cacheable_equations(bare_layout):
    """Equations whose shared transcendental aux triggers caching.

    The Jacobian entries reuse ``sin(x)*cos(y) + exp(x*y)`` across
    both outputs, so the default cache planner selects auxiliaries to
    cache; this drives the ``cached_aux`` branches of the cached
    generators.
    """
    ib = bare_layout
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)

    shared = sp.sin(x) * sp.cos(y) + sp.exp(x * y)
    equations = [
        (dx, a * shared + x**3),
        (dy, b * shared + y**3),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def chained_aux_equations(bare_layout):
    """Two chained auxiliaries on a system without drivers/observables."""
    ib = bare_layout
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)

    first = sp.Symbol("first")
    second = sp.Symbol("second")
    equations = [
        (first, x * y),
        (second, first + a * x),
        (dx, second * a),
        (dy, first * b),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def observable_driver_layout():
    """A layout with an observable and a driver declared."""
    return SystemLayout(
        states=("x", "y"),
        parameters=("a", "b"),
        observables=("obs1",),
        drivers=("driver1",),
    )


@pytest.fixture
def observable_driver_equations(observable_driver_layout):
    """Two-state equations using an observable and a driver."""
    ib = observable_driver_layout
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    drive = sp.Symbol("driver1", real=True)
    obs = sp.Symbol("obs1", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)

    equations = [
        (obs, a * x + drive),
        (dx, obs * y - b * x * y),
        (dy, b * x * y - a * y),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def cacheable_observable_equations(observable_driver_layout):
    """Cache-triggering equations with an observable and a chained aux."""
    ib = observable_driver_layout
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    b = sp.Symbol("b", real=True)
    obs = sp.Symbol("obs1", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)

    first = sp.Symbol("first")
    shared = sp.sin(x) * sp.cos(y) + sp.exp(x * y)
    equations = [
        (first, x * y + sp.tan(y)),
        (obs, shared + first),
        (dx, a * obs + x**3),
        (dy, b * obs + y**3 + first),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def single_observable_layout():
    """A layout with a single state and one observable."""
    return SystemLayout(
        states=("x",), parameters=("a",), observables=("obs1",)
    )


@pytest.fixture
def single_observable_equations(single_observable_layout):
    """Single-state equations carrying one observable."""
    ib = single_observable_layout
    x = sp.Symbol("x", real=True)
    a = sp.Symbol("a", real=True)
    obs = sp.Symbol("obs1", real=True)
    dx = sp.Symbol("dx", real=True)

    equations = [
        (obs, a * x * x),
        (dx, obs + a * x),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def lower_triangular_stage_coefficients():
    """Butcher tableau slice with a structural zero coupling.

    The lower-triangular ``a`` matrix exercises the
    ``coeff_value == 0`` skip in stage-coupling loops; the nodes are
    the matching Radau-style abscissae.
    """
    a = [
        [sp.Rational(1, 4), 0],
        [sp.Rational(1, 2), sp.Rational(1, 4)],
    ]
    nodes = [sp.Rational(1, 4), sp.Rational(3, 4)]
    return a, nodes
