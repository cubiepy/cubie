"""DAE parser front-end and SymbolicODE integration tests."""

import numpy as np
import pytest
import sympy as sp

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import to_sympy
from cubie.odesystems.symbolic.parsing import (
    EquationWarning,
    parse_input,
)
from cubie.odesystems.symbolic.structural.errors import (
    ExtraEquationsSystemError,
    InvalidSystemError,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from tests._utils import run_device_dxdt, run_device_observables


def parse_dae_input(**kwargs):
    """Parse and drop the checkpoint from the products."""

    return parse_input(**kwargs)[:5]


class TestParseDaeInput:
    def test_string_implicit_and_alias(self):
        index_map, _syms, _funcs, parsed, _h = (
            parse_dae_input(
                dxdt=["dx = -k*x + y", "y = 2*x"],
                states={"x": 1.0},
                observables=["y"],
                parameters={"k": 0.5},
            )
        )
        assert list(index_map.state_names) == ["x"]
        assert list(index_map.observable_names) == ["y"]
        assert parsed.mass_matrix is None
        eqs = {lhs.name: rhs for lhs, rhs in parsed.ordered}
        x, k = ir.sym("x"), ir.sym("k")
        # Observable y inlined into the dynamics.
        assert sp.simplify(
            to_sympy(eqs["dx"] - (-k * x + 2 * x))
        ) == 0
        assert sp.simplify(to_sympy(eqs["y"] - 2 * x)) == 0

    def test_nested_derivative_string(self):
        index_map, _syms, _funcs, parsed, _h = (
            parse_dae_input(
                dxdt=["d(d(x, t), t) = -x"],
                states={"x": 1.0},
            )
        )
        # Two differential states, no torn residuals.
        assert len(index_map.state_names) == 2
        assert parsed.mass_matrix is None

    def test_sympy_higher_order_derivative(self):
        x = sp.Symbol("x", real=True)
        t = sp.Symbol("t", real=True)
        index_map, _syms, _funcs, parsed, _h = (
            parse_dae_input(
                dxdt=[(sp.Derivative(x, t, 2), -x)],
                states={"x": 1.0},
            )
        )
        # The second-order derivative introduces a second
        # differential state and no torn residuals.
        assert len(index_map.state_names) == 2
        assert parsed.mass_matrix is None

    def test_torn_system_mass_and_defaults_warning(self):
        with pytest.warns(EquationWarning):
            index_map, _s, _f, parsed, _h = (
                parse_dae_input(
                    dxdt="""
                    dx = vx
                    dy = vy
                    dvx = T*x
                    dvy = T*y - g
                    0 = x**2 + y**2 - L**2
                    """,
                    states={
                        "x": 1.0,
                        "y": 0.0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "T": 0.0,
                    },
                    constants={"g": 9.81, "L": 1.0},
                    state_priority={"y": 10, "vy": 10},
                )
            )
        assert len(index_map.state_names) == 5
        mass = np.asarray(parsed.mass_matrix)
        # Three torn residual rows leave three zero diagonals.
        assert sum(mass[i, i] == 0.0 for i in range(5)) == 3

    @pytest.mark.parametrize(
        "dxdt, observables",
        [
            (
                """
                dx = -z1
                dy = -z2
                0 = z1**5 + z1 - x
                0 = z2**3 + z2 - y
                """,
                [],
            ),
            # z1's loop reads z2, so its residual comes second in
            # solve order while z1 comes first by name.
            (
                """
                dx = -z1
                dy = -z2
                0 = w2 - z2**3 - x
                0 = z2**5 + z2 + w2
                0 = w1 - z1**3 - z2
                0 = z1**5 + z1 + w1
                """,
                ["w1", "w2"],
            ),
        ],
        ids=["independent", "chained_loops"],
    )
    def test_torn_residuals_pair_rows(self, dxdt, observables):
        # Each algebraic row's residual depends on that row's state,
        # directly or through the assignments it reads.
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=dxdt,
            states={"x": 1.0, "y": 1.0, "z1": 0.5, "z2": 0.5},
            observables=observables,
        )
        names = list(index_map.state_names)
        mass = np.asarray(parsed.mass_matrix)
        algebraic = [n for i, n in enumerate(names) if mass[i, i] == 0]
        assert sorted(algebraic) == ["z1", "z2"]
        assignments = {lhs: rhs for lhs, rhs in parsed.ordered}

        def dependencies(expr):
            found = set()
            pending = list(ir.free_atoms(expr))
            while pending:
                atom = pending.pop()
                if atom in found:
                    continue
                found.add(atom)
                if atom in assignments:
                    pending.extend(ir.free_atoms(assignments[atom]))
            return found

        for name in algebraic:
            residual = assignments[ir.sym("d" + name)]
            assert ir.sym(name) in dependencies(residual)

    def test_numeric_literal_implicit_lhs(self):
        # Implicit equations accept any numeric-literal LHS, not
        # just the exact token "0".
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=["dx = -z", "0.0 = z + 2*x"],
            states={"x": 1.0, "z": 0.0},
        )
        # z solves through: one state, no mass, z an auxiliary.
        assert list(index_map.state_names) == ["x"]
        assert parsed.mass_matrix is None
        x = ir.sym("x")
        eqs = {lhs.name: rhs for lhs, rhs in parsed.ordered}
        assert sp.simplify(to_sympy(eqs["z"] + 2 * x)) == 0

    def test_undeclared_symbol_inferred_parameter(self):
        index_map, _s, _f, _p, _h = parse_dae_input(
            dxdt=["dx = -mu * x"],
            states={"x": 1.0},
        )
        assert "mu" in index_map.parameter_names

    def test_strict_rejects_undeclared(self):
        with pytest.raises(ValueError, match="(?i)undefined symbol"):
            parse_dae_input(
                dxdt=["dx = -mu * x"],
                states={"x": 1.0},
                strict=True,
            )

    def test_callable_input_routes_structurally(self):
        def rhs(t, y):
            return [-y[0]]

        _, _, _, parsed, _ = parse_dae_input(
            dxdt=rhs,
            states={"x": 1.0},
        )
        assert parsed.mass_matrix is None
        assert len(parsed.ordered) == 1

    def test_assigning_parameter_rejected(self):
        with pytest.raises(ValueError, match="immutable"):
            parse_dae_input(
                dxdt=["k = 2*x", "dx = -k*x"],
                states={"x": 1.0},
                parameters={"k": 0.5},
            )

    def test_unassigned_observable_rejected(self):
        with pytest.raises(ValueError, match="no.*defining"):
            parse_dae_input(
                dxdt=["dx = -x"],
                states={"x": 1.0},
                observables=["y"],
            )


class TestSymbolicODEIntegration:
    def test_simplified_system_compiles_and_evaluates(self):
        ode = create_ODE_system(
            dxdt=["dx = -k*x + y", "y = 2*x"],
            states={"x": 1.0},
            observables=["y"],
            parameters={"k": 0.5},
            precision=np.float64,
            name="dae_test_alias",
        )
        assert ode.compile_settings.mass is None
        state = np.array([2.0], dtype=np.float64)
        params = np.array([0.5], dtype=np.float64)
        drivers = np.zeros(1, dtype=np.float64)
        obs = np.zeros(1, dtype=np.float64)
        out = np.zeros(1, dtype=np.float64)
        run_device_dxdt(
            ode.dxdt_fn, state, params, drivers, obs, out, 0.0
        )
        assert out[0] == pytest.approx(3.0, abs=1e-13)
        run_device_observables(
            ode.observables_fn, state, params, drivers, obs, 0.0
        )
        assert obs[0] == pytest.approx(4.0, abs=1e-13)

    def test_torn_dae_mass_matrix_reaches_system(self, torn_dae_system):
        mass = torn_dae_system.compile_settings.mass
        assert mass is not None
        assert mass.shape == (2, 2)
        assert mass[0, 0] == 1.0 and mass[1, 1] == 0.0
        names = list(torn_dae_system.indices.state_names)
        assert names == ["x", "z"]
        state = np.array([2.0, 1.0], dtype=np.float64)
        params = np.zeros(1, dtype=np.float64)
        drivers = np.zeros(1, dtype=np.float64)
        obs = np.zeros(1, dtype=np.float64)
        out = np.zeros(2, dtype=np.float64)
        run_device_dxdt(
            torn_dae_system.dxdt_fn,
            state,
            params,
            drivers,
            obs,
            out,
            0.0,
        )
        # dx = -z = -1; residual = z^5 + z - x = 0 at (2, 1).
        assert out[0] == pytest.approx(-1.0, abs=1e-13)
        assert out[1] == pytest.approx(0.0, abs=1e-13)


class TestScaledDerivativeLhs:
    def test_string_coefficient_solves_through(self):
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt="""
            M * dv = -k * x - c * v
            dx = v
            """,
            states={"x": 1.0, "v": 0.0},
            constants={"M": 2.0, "k": 4.0, "c": 0.5},
        )
        assert set(index_map.state_names) == {"x", "v"}
        assert parsed.mass_matrix is None
        eqs = {lhs.name: rhs for lhs, rhs in parsed.ordered}
        x, v = sp.symbols("x v", real=True)
        # Constant values fold as literals before simplification.
        assert sp.simplify(
            to_sympy(eqs["dv"]) - (-4.0 * x - 0.5 * v) / 2.0
        ) == 0
        assert sp.simplify(to_sympy(eqs["dx"]) - v) == 0

    def test_sympy_tuple_coefficient_solves_through(self):
        x, v = sp.symbols("x v", real=True)
        M, k = sp.symbols("M k", real=True)
        dv = sp.Symbol("dv", real=True)
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=[(M * dv, -k * x), ("dx", "v")],
            states={"x": 1.0, "v": 0.0},
            constants={"M": 2.0, "k": 4.0},
        )
        assert set(index_map.state_names) == {"x", "v"}
        assert parsed.mass_matrix is None
        eqs = {lhs.name: rhs for lhs, rhs in parsed.ordered}
        assert sp.simplify(
            to_sympy(eqs["dv"]) - (-4.0 * x) / 2.0
        ) == 0

    @pytest.mark.parametrize(
        "solver_settings_override",
        [
            {
                "system_type": "ring_modulator_index2",
                "precision": np.float64,
            }
        ],
        indirect=True,
    )
    def test_zero_coefficient_matches_algebraic_form(
        self,
        system,
        ring_modulator_scaled_system,
    ):
        # Cs = 0 reduces identically to the explicit 0 = form.
        zero = system
        scaled = ring_modulator_scaled_system
        assert list(scaled.indices.state_names) == list(
            zero.indices.state_names
        )
        assert list(scaled.indices.observable_names) == list(
            zero.indices.observable_names
        )
        assert np.array_equal(scaled.mass, zero.mass)
        assert (
            scaled.equations.to_equation_list()
            == zero.equations.to_equation_list()
        )

    def test_assigned_dx_auxiliary_keeps_reference_semantics(self):
        # An assigned dfoo stays a reference, not d(foo)/dt.
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt="""
            dfoo = 2*y
            foo = y + 1
            q + dfoo = 3
            dy = -y
            """,
            states={"y": 1.0},
            observables=["q"],
        )
        assert list(index_map.state_names) == ["y"]
        eqs = {lhs.name: rhs for lhs, rhs in parsed.ordered}
        y = sp.Symbol("y", real=True)
        dfoo = sp.Symbol("dfoo", real=True)
        assert sp.simplify(to_sympy(eqs["q"]) - (3 - dfoo)) == 0
        assert sp.simplify(to_sympy(eqs["dfoo"]) - 2 * y) == 0


class TestDerivativeBlockPolicy:
    def test_zero_parameter_pivot_never_divides(self):
        system = create_ODE_system(
            "p*dx = -x\ndx = -y",
            states={"x": 1.0, "y": 1.0},
            parameters={"p": 0.0},
            precision=np.float32,
            simplify_options={"allow_parameter": False},
            name="policy_pivot",
        )
        assert list(system.indices.states.symbol_map) == ["x", "y"]
        assert system.mass.tolist() == [[1.0, 0.0], [0.0, 0.0]]
        p = sp.Symbol("p", real=True)
        for _, rhs in system.equations.ordered:
            assert not sp.denom(to_sympy(rhs)).has(p)

    def test_sum_of_parameters_coefficient_reduces_like_a_number(self):
        from tests.system_fixtures import (
            TRANSAMP_CONSTANTS,
            TRANSAMP_DC_STATES,
            TRANSAMP_EQUATIONS,
        )

        numeric = create_ODE_system(
            TRANSAMP_EQUATIONS,
            states=dict(TRANSAMP_DC_STATES),
            observables=["y1", "y4", "y7"],
            constants=dict(TRANSAMP_CONSTANTS),
            precision=np.float32,
            name="transamp_numeric_c1",
        )
        constants = dict(TRANSAMP_CONSTANTS)
        del constants["c1"]
        split = create_ODE_system(
            TRANSAMP_EQUATIONS.replace("c1", "(c1a + c1b)"),
            states=dict(TRANSAMP_DC_STATES),
            observables=["y1", "y4", "y7"],
            parameters={"c1a": 0.5e-6, "c1b": 0.5e-6},
            constants=constants,
            precision=np.float32,
            name="transamp_split_c1",
        )
        assert list(split.indices.states.symbol_map) == list(
            numeric.indices.states.symbol_map
        )
        assert len(split.indices.states.symbol_map) == 8


def solved(parsed):
    return {lhs.name: to_sympy(rhs) for lhs, rhs in parsed.ordered}


def equivalent(left, right):
    return sp.simplify(left - right) == 0


def real_symbols(names):
    return sp.symbols(names, real=True)


class TestStructuralInputPaths:
    def test_extra_equation_left_as_residual(self):
        # Two equations fix y; with z free, the spare one becomes the
        # residual row of z.
        x, y = real_symbols("x y")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=[
                "dx = -x + y + z",
                "0 = y - sin(x)",
                "0 = y**2 - sin(x)**2",
                "dw = -w",
            ],
            states={"x": 1.0, "y": 0.0, "z": 0.0, "w": 1.0},
            simplify_options={"fully_determined": False},
        )
        assert list(index_map.state_names) == ["w", "x", "z"]
        assert parsed.mass_matrix == (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        eqs = solved(parsed)
        assert equivalent(eqs["y"], sp.sin(x))
        assert equivalent(eqs["dz"], y**2 - sp.sin(x) ** 2)

    def test_extra_differentiated_equations_warn_unpaired(self):
        with pytest.warns(
            UserWarning,
            match="2 residual equations for 0 algebraic states",
        ):
            index_map, _s, _f, _p, _h = parse_dae_input(
                dxdt=[
                    "d(d(x,t),t) = -x + y",
                    "0 = y - sin(x)",
                    "0 = y**2 - sin(x)**2",
                    "0 = d(x,t) - cos(y)",
                ],
                states={"x": 1.0, "y": 0.0},
                simplify_options={"fully_determined": False},
            )
        assert list(index_map.state_names) == ["x", "x_t"]

    def test_allow_symbolic_divides_by_state_coefficient(self):
        x = real_symbols("x")
        dxdt = ["dx = -x + y", "0 = x*y - 1"]
        states = {"x": 1.0, "y": 1.0}
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=dxdt,
            states=states,
            simplify_options={"allow_symbolic": True},
        )
        assert list(index_map.state_names) == ["x"]
        assert equivalent(solved(parsed)["y"], 1 / x)
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=dxdt, states=states
        )
        assert list(index_map.state_names) == ["x", "y"]
        assert parsed.mass_matrix == ((1.0, 0.0), (0.0, 0.0))

    def test_integer_constraint_reduced_twice(self):
        # The constraint is differentiated twice; its integer
        # Jacobian selects the dummy derivatives at both levels.
        t, x2, x2_t = real_symbols("t x2 x2_t")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=[
                "d(d(x1,t),t) = -x1 + lam",
                "d(d(x2,t),t) = -x2 + 2*lam",
                "0 = x1 + 2*x2 - sin(t)",
            ],
            states={"x1": 0.0, "x2": 0.0, "lam": 0.0},
        )
        assert list(index_map.state_names) == ["x1_tt", "x2", "x2_t"]
        eqs = solved(parsed)
        assert equivalent(eqs["x1"], sp.sin(t) - 2 * x2)
        assert equivalent(eqs["x1_t"], sp.cos(t) - 2 * x2_t)

    def test_alias_exposed_by_integer_elimination(self):
        a, x = real_symbols("a x")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=[
                "dx = -x + a",
                "0 = a + b + c",
                "0 = a + b + 2*c",
                "0 = b - sin(x)",
            ],
            states={"x": 1.0, "a": 0.0, "b": 0.0, "c": 0.0},
        )
        assert list(index_map.state_names) == ["x"]
        eqs = solved(parsed)
        assert eqs["c"] == 0
        assert equivalent(eqs["b"], -a)
        assert equivalent(eqs["a"], -sp.sin(x))

    def test_alias_exposed_by_integer_elimination_moves_derivative(self):
        x = real_symbols("x")
        with pytest.warns(
            UserWarning,
            match="1 residual equations for 0 algebraic states",
        ):
            index_map, _s, _f, parsed, _h = parse_dae_input(
                dxdt=[
                    "dx = -x + y",
                    "dy = -y",
                    "0 = x + y + c",
                    "0 = x + y + 2*c",
                ],
                states={"x": 1.0, "y": 0.0, "c": 0.0},
                simplify_options={"fully_determined": False},
            )
        assert list(index_map.state_names) == ["x"]
        eqs = solved(parsed)
        assert equivalent(eqs["y"], -x)
        assert equivalent(eqs["dx"], -2 * x)

    def test_conflicting_aliases_keep_irreducible_member(self):
        a = real_symbols("a")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=["dx = -x + a", "0 = a - b", "0 = a + b"],
            states={"x": 1.0, "a": 0.0, "b": 0.0},
            irreducible=["a"],
        )
        assert list(index_map.state_names) == ["a", "x"]
        assert parsed.mass_matrix == ((0.0, 0.0), (0.0, 1.0))
        eqs = solved(parsed)
        assert eqs["b"] == 0
        assert equivalent(eqs["da"], -a)

    def test_conflicting_aliases_zero_derivative_chain(self):
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=["dx = -x + y", "0 = x - y", "0 = x + y"],
            states={"x": 1.0, "y": 0.0},
            simplify_options={"fully_determined": False},
        )
        assert list(index_map.state_names) == []
        eqs = solved(parsed)
        assert eqs["x"] == 0
        assert eqs["y"] == 0

    def test_conflict_group_absorbs_later_alias(self):
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=["dx = -x + a + c", "0 = a - b", "0 = a + b", "0 = c - a"],
            states={"x": 1.0, "a": 0.0, "b": 0.0, "c": 0.0},
            simplify_options={"fully_determined": False},
        )
        assert list(index_map.state_names) == ["x"]
        eqs = solved(parsed)
        assert [eqs[name] for name in ("a", "b", "c")] == [0, 0, 0]

    def test_smaller_alias_group_joins_larger(self):
        b = real_symbols("b")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=[
                "dx = -x + a**3 + c**3",
                "0 = cos(b) - 0.5",
                "0 = 2*b - 2*c",
                "0 = a - b",
            ],
            states={"x": 0.1, "a": 0.1, "b": 0.1, "c": 0.1},
        )
        assert list(index_map.state_names) == ["b", "x"]
        eqs = solved(parsed)
        assert eqs["a"] == b
        assert eqs["c"] == b

    def test_product_of_two_unknowns_is_not_an_alias(self):
        a, b, x = real_symbols("a b x")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=["dx = -x + a + b", "0 = a*b", "0 = a - sin(x)"],
            states={"x": 1.0, "a": 0.0, "b": 0.0},
        )
        assert list(index_map.state_names) == ["b", "x"]
        eqs = solved(parsed)
        assert equivalent(eqs["a"], sp.sin(x))
        assert equivalent(eqs["db"], a * b)

    def test_assignment_to_differential_state_stays_a_constraint(self):
        x, z = real_symbols("x z")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=["dx = -x + z", "x = sin(z)"],
            states={"x": 0.5, "z": 0.5},
        )
        assert list(index_map.state_names) == ["x", "z"]
        assert parsed.mass_matrix == ((1.0, 0.0), (0.0, 0.0))
        assert equivalent(solved(parsed)["dz"], sp.sin(z) - x)

    def test_identity_assignment_is_not_torn(self):
        x = real_symbols("x")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=["dx = -x + y", "y = y", "0 = y - sin(x)"],
            states={"x": 1.0, "y": 0.0},
            simplify_options={"fully_determined": False},
        )
        assert list(index_map.state_names) == ["x"]
        assert equivalent(solved(parsed)["y"], sp.sin(x))

    def test_repeated_alias_between_irreducibles_merges(self):
        a, b = real_symbols("a b")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=[
                "dx = -x + a",
                "0 = a - b",
                "0 = 2*a - 2*b",
                "0 = b - sin(x)",
            ],
            states={"x": 1.0, "a": 0.0, "b": 0.0},
            irreducible=["a", "b"],
            simplify_options={"fully_determined": False},
        )
        assert list(index_map.state_names) == ["a", "b", "x"]
        assert equivalent(solved(parsed)["da"], 2 * a - 2 * b)

    def test_user_symbol_named_like_internal_derivative(self):
        x, d1 = real_symbols("x _cubie_D1_x")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=["dx = -x + _cubie_D1_x", "0 = _cubie_D1_x - sin(x)"],
            states={"x": 1.0, "_cubie_D1_x": 0.0},
        )
        assert list(index_map.state_names) == ["x"]
        eqs = solved(parsed)
        assert equivalent(eqs["dx"], d1 - x)
        assert equivalent(eqs["_cubie_D1_x"], sp.sin(x))

    def test_index_reduction_rejects_excess_equations(self):
        with pytest.raises(InvalidSystemError, match="structurally"):
            parse_dae_input(
                dxdt=["dx = -x", "0 = y - sin(x)", "0 = y**2 - x"],
                states={"x": 1.0, "y": 0.0},
                simplify_options={"consistency_check": False},
            )

    def test_index_reduction_skips_equation_without_unknowns(self):
        x = real_symbols("x")
        index_map, _s, _f, parsed, _h = parse_dae_input(
            dxdt=["dx = -x + y", "0 = y - sin(x)", "0 = k - 1"],
            states={"x": 1.0, "y": 0.0},
            parameters={"k": 1.0},
            simplify_options={"consistency_check": False},
        )
        assert list(index_map.state_names) == ["x"]
        assert equivalent(solved(parsed)["y"], sp.sin(x))

    def test_cancelling_alias_substitution_warns_singular_solve(self):
        with pytest.warns(UserWarning, match=r"for Sym\(a\) is singular"):
            index_map, _s, _f, _p, _h = parse_dae_input(
                dxdt=[
                    "dx = -x + a",
                    "0 = a + b + c",
                    "0 = a + b + 2*c",
                    "0 = a + exp(b) - sin(x)",
                ],
                states={"x": 1.0, "a": 0.0, "b": 0.0, "c": 0.0},
            )
        assert list(index_map.state_names) == ["a", "x"]

    def test_inconsistent_constraints_warn_singular(self):
        with pytest.warns(UserWarning) as record:
            index_map, _s, _f, _p, _h = parse_dae_input(
                dxdt=[
                    "dx = -x + z",
                    "0 = x + y - sin(t)",
                    "0 = x + y - cos(t)",
                ],
                states={"x": 0.1, "y": 0.1, "z": 0.1},
            )
        messages = [str(w.message) for w in record]
        assert "The DAE system is singular!" in messages
        assert (
            "The number of dummy derivatives (1) does not match the "
            "number of differentiated equations (2)." in messages
        )
        assert list(index_map.state_names) == ["x_t", "y"]

    def test_variable_cancelled_from_all_equations_not_counted(self):
        # y = -x cancels z from the last equation, leaving it unused.
        with pytest.raises(
            ExtraEquationsSystemError,
            match="1 highest order derivative variables and 2 equations",
        ):
            parse_dae_input(
                dxdt=[
                    "dx = -x",
                    "0 = x + y",
                    "0 = cos(y) + y*z + z*x - 0.5",
                ],
                states={"x": 0.1, "y": 0.1, "z": 0.1},
            )

    def test_unmatched_unknown_stays_a_state(self):
        with pytest.warns(
            UserWarning,
            match="0 residual equations for 1 algebraic states",
        ):
            index_map, _s, _f, _p, _h = parse_dae_input(
                dxdt=[
                    "d(x0,t) = -x0 + sin(x0)",
                    "0 = 2*z1 - z0 + 2*x0",
                    "0 = -2*x0 - 2*z2 - z1",
                ],
                states={"x0": 0.1, "z0": 0.1, "z1": 0.1, "z2": 0.1},
                simplify_options={"fully_determined": False},
            )
        assert list(index_map.state_names) == ["x0", "z2"]

    def test_coupled_linear_loop_torn_acyclic(self):
        x0, z2, z3 = real_symbols("x0 z2 z3")
        with pytest.warns(
            UserWarning,
            match="1 residual equations for 2 algebraic states",
        ):
            index_map, _s, _f, parsed, _h = parse_dae_input(
                dxdt=[
                    "d(x0,t) = -x0 + z3*z2",
                    "0 = -2*z1 + z0 - 2*x0",
                    "0 = z2 + z3 + z0",
                    "0 = 2*z0 + 3*z1 + z2 - sin(t)",
                ],
                states={
                    "x0": 0.1,
                    "z0": 0.1,
                    "z1": 0.1,
                    "z2": 0.1,
                    "z3": 0.1,
                },
                simplify_options={"fully_determined": False},
            )
        assert list(index_map.state_names) == ["x0", "z2", "z3"]
        assert equivalent(solved(parsed)["z1"], -(z2 + z3 + 2 * x0) / 2)
