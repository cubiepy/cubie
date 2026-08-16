import numpy as np
import pytest
import sympy as sp

from cubie.odesystems.symbolic.indexedbasemaps import (
    IndexedBaseMap,
    IndexedBases,
)
from cubie.odesystems.symbolic.parsing import ParsedEquations
from cubie.odesystems.symbolic.symbolicODE import (
    SymbolicODE,
    create_ODE_system,
)


@pytest.fixture(scope="session")
def torn_dae_system():
    """Structurally torn index-1 DAE: dx = -z under z**5 + z = x.

    Precision is pinned to float64 and the fixture is independent of
    solver_settings_override, so it builds once per worker however the
    chain is parametrised. Shared by the DAE parser and solve tests.
    """
    return create_ODE_system(
        dxdt="""
        dx = -z
        0 = z**5 + z - x
        """,
        states={"x": 2.0, "z": 1.0},
        precision=np.float64,
        simplify=True,
        name="torn_dae",
    )


RING_MODULATOR_CONSTANTS = {
    "C": 1.6e-8,
    "Cp": 1.0e-8,
    "Lh": 4.45,
    "Ls1": 0.002,
    "Ls2": 5.0e-4,
    "Ls3": 5.0e-4,
    "gamma": 40.67286402e-9,
    "R": 25000.0,
    "Rp": 50.0,
    "Rg1": 36.3,
    "Rg2": 17.3,
    "Rg3": 17.3,
    "Ri": 50.0,
    "Rc": 600.0,
    "delta": 17.7493332,
    "w1": 6283.185307179586,
    "w2": 62831.85307179586,
}

RING_MODULATOR_STATES = {
    name: 0.0
    for name in (
        "U1", "U2", "U3", "U4", "U5", "U6", "U7",
        "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8",
    )
}

RING_MODULATOR_AUXILIARIES = """
    Uin1 = Uin1_amplitude * sin(w1 * t)
    Uin2 = 2.0 * sin(w2 * t)
    UD1 = U3 - U5 - U7 - Uin2
    UD2 = -U4 + U6 - U7 - Uin2
    UD3 = U4 + U5 + U7 + Uin2
    UD4 = -U3 - U6 + U7 + Uin2
    qD1 = gamma * (exp(delta * UD1) - 1.0)
    qD2 = gamma * (exp(delta * UD2) - 1.0)
    qD3 = gamma * (exp(delta * UD3) - 1.0)
    qD4 = gamma * (exp(delta * UD4) - 1.0)
"""

RING_MODULATOR_COMMON = """
    dU1 = (I1 - 0.5 * I3 + 0.5 * I4 + I7 - U1 / R) / C
    dU2 = (I2 - 0.5 * I5 + 0.5 * I6 + I8 - U2 / R) / C
    dU7 = (-U7 / Rp + qD1 + qD2 - qD3 - qD4) / Cp
    dI1 = -U1 / Lh
    dI2 = -U2 / Lh
    dI3 = (0.5 * U1 - U3 - Rg2 * I3) / Ls2
    dI4 = (-0.5 * U1 + U4 - Rg3 * I4) / Ls3
    dI5 = (0.5 * U2 - U5 - Rg2 * I5) / Ls2
    dI6 = (-0.5 * U2 + U6 - Rg3 * I6) / Ls3
    dI7 = (-U1 + Uin1 - (Ri + Rg1) * I7) / Ls1
    dI8 = (-U2 - (Rc + Rg1) * I8) / Ls1
"""


def _ring_modulator_index2(equations, constants, name):
    return create_ODE_system(
        equations,
        states=dict(RING_MODULATOR_STATES),
        parameters={"Uin1_amplitude": 0.5},
        constants=constants,
        observables=["U3", "U4", "U6", "I3"],
        precision=np.float64,
        simplify=True,
        name=name,
    )


@pytest.fixture(scope="session")
def ring_modulator_index2_system():
    """Index-2 ring modulator (Test Set II-3, Cs = 0); float64 only."""
    equations = RING_MODULATOR_AUXILIARIES + """
    0 = I3 - qD1 + qD4
    0 = -I4 + qD2 - qD3
    0 = I5 + qD1 - qD3
    0 = -I6 - qD2 + qD4
""" + RING_MODULATOR_COMMON
    return _ring_modulator_index2(
        equations,
        dict(RING_MODULATOR_CONSTANTS),
        "ring_modulator_index2",
    )


@pytest.fixture(scope="session")
def ring_modulator_index2_scaled_system():
    """The same index-2 system written as ``Cs*dX = ...`` with Cs = 0."""
    equations = RING_MODULATOR_AUXILIARIES + """
    Cs * dU3 = I3 - qD1 + qD4
    Cs * dU4 = -I4 + qD2 - qD3
    Cs * dU5 = I5 + qD1 - qD3
    Cs * dU6 = -I6 - qD2 + qD4
""" + RING_MODULATOR_COMMON
    return _ring_modulator_index2(
        equations,
        dict(RING_MODULATOR_CONSTANTS, Cs=0.0),
        "ring_modulator_index2_scaled",
    )


@pytest.fixture(scope="session")
def simple_system_defaults():
    states = {"one": 0.9, "foo": 0.5}
    parameters = {"zebra": 0.2, "fox": 0.4}
    constants = {"apple": 0.43, "linen": 0.32}
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
        constants,
        drivers,
        observables,
        dxdt_str,
        dxdt_list,
    )


@pytest.fixture(scope="session")
def simple_symbols_dict(simple_system_defaults):
    (
        states,
        parameters,
        constants,
        drivers,
        observables,
        dxdt_str,
        dxdt_list,
    ) = simple_system_defaults
    ib = IndexedBases.from_user_inputs(
        states, parameters, constants, observables, drivers
    )
    symbols = ib.all_symbols
    return symbols


@pytest.fixture
def simple_symbols():
    """Basic SymPy symbols for testing."""
    x, y, z = sp.symbols("x y z", real=True)
    a, b, c = sp.symbols("a b c", real=True)
    return {"states": [x, y], "params": [a, b], "constants": [c], "aux": [z]}


@pytest.fixture
def simple_equations(indexed_bases):
    """Simple symbolic equations for testing."""
    x = indexed_bases.states.symbol_map["x"]
    y = indexed_bases.states.symbol_map["y"]
    a = indexed_bases.parameters.symbol_map["a"]
    b = indexed_bases.parameters.symbol_map["b"]
    dx = indexed_bases.dxdt.symbol_map["dx"]
    dy = indexed_bases.dxdt.symbol_map["dy"]

    equations = [
        (dx, -a * x + b * y),
        (dy, a * x - b * y),
    ]
    return ParsedEquations.from_equations(equations, indexed_bases)


@pytest.fixture
def complex_equations(indexed_bases):
    """More complex equations with auxiliary variables."""
    x = indexed_bases.states.symbol_map["x"]
    y = indexed_bases.states.symbol_map["y"]
    a = indexed_bases.parameters.symbol_map["a"]
    b = indexed_bases.parameters.symbol_map["b"]
    c = indexed_bases.constants.symbol_map["c"]
    dx = indexed_bases.dxdt.symbol_map["dx"]
    dy = indexed_bases.dxdt.symbol_map["dy"]
    obs = indexed_bases.observables.symbol_map["obs1"]

    aux = sp.Symbol("aux", real=True)
    equations = [
        (aux, a * x + b * y),
        (dx, -aux + c),
        (dy, aux - c * y),
        (obs, x + y),
    ]
    return ParsedEquations.from_equations(equations, indexed_bases)


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
        parameters={"alpha": precision(0.0), "beta": precision(0.0)},
        constants={"c0": precision(1.1)},
        drivers={"drive": precision(0.0)},
        observables=["obs_rate", "obs_total"],
        precision=precision,
        strict=True,
        name="observables_kernel_system",
    )

    return system


@pytest.fixture
def indexed_base_map():
    """Sample IndexedBaseMap for testing."""
    symbols = ["x", "y", "z"]
    defaults = [1.0, 2.0, 3.0]
    return IndexedBaseMap("test_base", symbols, defaults)


@pytest.fixture
def indexed_bases():
    """Sample IndexedBases for testing."""
    states = ["x", "y"]
    parameters = ["a", "b"]
    constants = ["c", "d"]
    observables = ["obs1"]
    drivers = ["driver1"]

    return IndexedBases.from_user_inputs(
        states=states,
        parameters=parameters,
        constants=constants,
        observables=observables,
        drivers=drivers,
    )


@pytest.fixture
def sample_hash():
    """Sample hash string for testing."""
    return "# hash: test_hash_123456"


@pytest.fixture
def linear_system_equations(indexed_bases):
    """Linear system equations for Jacobian testing."""
    x = indexed_bases.states.symbol_map["x"]
    y = indexed_bases.states.symbol_map["y"]
    a = indexed_bases.parameters.symbol_map["a"]
    b = indexed_bases.parameters.symbol_map["b"]
    c = indexed_bases.constants.symbol_map["c"]
    d = indexed_bases.constants.symbol_map["d"]
    dx = indexed_bases.dxdt.symbol_map["dx"]
    dy = indexed_bases.dxdt.symbol_map["dy"]

    equations = [
        (dx, a * x + b * y),
        (dy, c * x + d * y),
    ]
    return ParsedEquations.from_equations(equations, indexed_bases)


@pytest.fixture
def nonlinear_equations(indexed_bases):
    """Nonlinear equations for comprehensive testing."""
    x = indexed_bases.states.symbol_map["x"]
    y = indexed_bases.states.symbol_map["y"]
    a = indexed_bases.parameters.symbol_map["a"]
    b = indexed_bases.parameters.symbol_map["b"]
    dx = indexed_bases.dxdt.symbol_map["dx"]
    dy = indexed_bases.dxdt.symbol_map["dy"]

    equations = [
        (dx, a * x - b * x * y),
        (dy, b * x * y - a * y),
    ]
    return ParsedEquations.from_equations(equations, indexed_bases)


@pytest.fixture
def bare_indexed_bases():
    """IndexedBases with two states and two parameters only.

    No constants, observables, or drivers: codegen tests use this to
    reach the no-observable and no-driver generator branches.
    """
    return IndexedBases.from_user_inputs(
        states=["x", "y"],
        parameters=["a", "b"],
        constants=[],
        observables=[],
        drivers=[],
    )


@pytest.fixture(scope="session")
def solver_scaling_collision_indexed_bases():
    """IndexedBases with constants named like solver scalings."""
    return IndexedBases.from_user_inputs(
        states={"x": 1.0},
        parameters={},
        constants={"beta": 2.0, "gamma": 3.0},
        observables=[],
        drivers=[],
    )


@pytest.fixture(scope="session")
def solver_scaling_collision_equations(
    solver_scaling_collision_indexed_bases,
):
    """One-state equation using colliding beta/gamma constants."""
    ib = solver_scaling_collision_indexed_bases
    x = ib.states.symbol_map["x"]
    beta = ib.constants.symbol_map["beta"]
    gamma = ib.constants.symbol_map["gamma"]
    dx = ib.dxdt.symbol_map["dx"]
    return ParsedEquations.from_equations(
        [(dx, beta * x + gamma)],
        ib,
    )


@pytest.fixture
def bare_nonlinear_equations(bare_indexed_bases):
    """Two-state nonlinear equations with no cacheable auxiliaries."""
    ib = bare_indexed_bases
    x = ib.states.symbol_map["x"]
    y = ib.states.symbol_map["y"]
    a = ib.parameters.symbol_map["a"]
    b = ib.parameters.symbol_map["b"]
    dx = ib.dxdt.symbol_map["dx"]
    dy = ib.dxdt.symbol_map["dy"]

    equations = [
        (dx, a * x - b * x * y),
        (dy, b * x * y - a * y),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def cacheable_equations(bare_indexed_bases):
    """Equations whose shared transcendental aux triggers caching.

    The Jacobian entries reuse ``sin(x)*cos(y) + exp(x*y)`` across
    both outputs, so the default cache planner selects auxiliaries to
    cache; this drives the ``cached_aux`` branches of the cached
    generators.
    """
    ib = bare_indexed_bases
    x = ib.states.symbol_map["x"]
    y = ib.states.symbol_map["y"]
    a = ib.parameters.symbol_map["a"]
    b = ib.parameters.symbol_map["b"]
    dx = ib.dxdt.symbol_map["dx"]
    dy = ib.dxdt.symbol_map["dy"]

    shared = sp.sin(x) * sp.cos(y) + sp.exp(x * y)
    equations = [
        (dx, a * shared + x**3),
        (dy, b * shared + y**3),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def chained_aux_equations(bare_indexed_bases):
    """Two chained auxiliaries on a system without drivers/observables."""
    ib = bare_indexed_bases
    x = ib.states.symbol_map["x"]
    y = ib.states.symbol_map["y"]
    a = ib.parameters.symbol_map["a"]
    b = ib.parameters.symbol_map["b"]
    dx = ib.dxdt.symbol_map["dx"]
    dy = ib.dxdt.symbol_map["dy"]

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
def observable_driver_indexed_bases():
    """IndexedBases with an observable and a driver declared."""
    return IndexedBases.from_user_inputs(
        states=["x", "y"],
        parameters=["a", "b"],
        constants=[],
        observables=["obs1"],
        drivers=["driver1"],
    )


@pytest.fixture
def observable_driver_equations(observable_driver_indexed_bases):
    """Two-state equations using an observable and a driver."""
    ib = observable_driver_indexed_bases
    x = ib.states.symbol_map["x"]
    y = ib.states.symbol_map["y"]
    a = ib.parameters.symbol_map["a"]
    b = ib.parameters.symbol_map["b"]
    drive = ib.drivers.symbol_map["driver1"]
    obs = ib.observables.symbol_map["obs1"]
    dx = ib.dxdt.symbol_map["dx"]
    dy = ib.dxdt.symbol_map["dy"]

    equations = [
        (obs, a * x + drive),
        (dx, obs * y - b * x * y),
        (dy, b * x * y - a * y),
    ]
    return ParsedEquations.from_equations(equations, ib)


@pytest.fixture
def single_observable_indexed_bases():
    """IndexedBases with a single state and one observable."""
    return IndexedBases.from_user_inputs(
        states=["x"],
        parameters=["a"],
        constants=[],
        observables=["obs1"],
        drivers=[],
    )


@pytest.fixture
def single_observable_equations(single_observable_indexed_bases):
    """Single-state equations carrying one observable."""
    ib = single_observable_indexed_bases
    x = ib.states.symbol_map["x"]
    a = ib.parameters.symbol_map["a"]
    obs = ib.observables.symbol_map["obs1"]
    dx = ib.dxdt.symbol_map["dx"]

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
