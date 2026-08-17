"""DAE parser front-end and SymbolicODE integration tests."""

import numpy as np
import pytest
import sympy as sp

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import to_sympy
from cubie.odesystems.symbolic.parsing.parser import (
    EquationWarning,
    parse_input,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from tests._utils import run_device_dxdt, run_device_observables


def parse_dae_input(**kwargs):
    """Parse and drop the checkpoint from the products."""

    return parse_input(**kwargs)[:6]


class TestParseDaeInput:
    def test_string_implicit_and_alias(self):
        index_map, _syms, _funcs, parsed, _h, simplified = (
            parse_dae_input(
                dxdt=["dx = -k*x + y", "y = 2*x"],
                states={"x": 1.0},
                observables=["y"],
                parameters={"k": 0.5},
            )
        )
        assert list(index_map.state_names) == ["x"]
        assert list(index_map.observable_names) == ["y"]
        assert simplified.mass_matrix is None
        eqs = {lhs.name: rhs for lhs, rhs in parsed.ordered}
        x, k = ir.sym("x"), ir.sym("k")
        # Observable y inlined into the dynamics.
        assert sp.simplify(
            to_sympy(eqs["dx"] - (-k * x + 2 * x))
        ) == 0
        assert sp.simplify(to_sympy(eqs["y"] - 2 * x)) == 0

    def test_nested_derivative_string(self):
        index_map, _syms, _funcs, parsed, _h, simplified = (
            parse_dae_input(
                dxdt=["d(d(x, t), t) = -x"],
                states={"x": 1.0},
            )
        )
        assert len(index_map.state_names) == 2
        assert len(simplified.differential_states) == 2
        assert not simplified.residuals

    def test_sympy_higher_order_derivative(self):
        x = sp.Symbol("x", real=True)
        t = sp.Symbol("t", real=True)
        index_map, _syms, _funcs, _parsed, _h, simplified = (
            parse_dae_input(
                dxdt=[(sp.Derivative(x, t, 2), -x)],
                states={"x": 1.0},
            )
        )
        assert len(simplified.differential_states) == 2

    def test_torn_system_mass_and_defaults_warning(self):
        with pytest.warns(EquationWarning):
            index_map, _s, _f, parsed, _h, simplified = (
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
        assert simplified.mass_matrix is not None
        assert len(simplified.residuals) == 3

    def test_two_torn_residuals_pair_rows(self):
        # Residual i constrains algebraic state i and the mass
        # matrix carries identity for differential states, zeros
        # for the residual rows, in state order.
        _im, _s, _f, _p, _h, simplified = parse_dae_input(
            dxdt="""
            dx = -z1
            dy = -z2
            0 = z1**5 + z1 - x
            0 = z2**3 + z2 - y
            """,
            states={"x": 1.0, "y": 1.0, "z1": 0.5, "z2": 0.5},
        )
        z1, z2 = ir.sym("z1"), ir.sym("z2")
        assert len(simplified.residuals) == 2
        assert set(simplified.algebraic_states) == {z1, z2}
        mass = np.asarray(simplified.mass_matrix)
        assert mass.shape == (4, 4)
        assert [mass[i, i] for i in range(4)] == [1, 1, 0, 0]
        for state_sym, residual in zip(
            simplified.algebraic_states, simplified.residuals
        ):
            other = z2 if state_sym == z1 else z1
            assert state_sym in ir.free_atoms(residual)
            assert other not in ir.free_atoms(residual)

    def test_numeric_literal_implicit_lhs(self):
        # Implicit equations accept any numeric-literal LHS, not
        # just the exact token "0".
        _im, _s, _f, _p, _h, simplified = parse_dae_input(
            dxdt=["dx = -z", "0.0 = z + 2*x"],
            states={"x": 1.0, "z": 0.0},
        )
        assert not simplified.residuals
        x = ir.sym("x")
        assert simplified.states == [x]
        obs = dict(simplified.observed)
        z = ir.sym("z")
        assert sp.simplify(to_sympy(obs[z] + 2 * x)) == 0

    def test_undeclared_symbol_inferred_parameter(self):
        index_map, _s, _f, _p, _h, _simplified = parse_dae_input(
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

        _, _, _, parsed, _, simplified = parse_dae_input(
            dxdt=rhs,
            states={"x": 1.0},
        )
        assert simplified.mass_matrix is None
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
            ode.evaluate_f, state, params, drivers, obs, out, 0.0
        )
        assert out[0] == pytest.approx(3.0, abs=1e-13)
        run_device_observables(
            ode.evaluate_observables, state, params, drivers, obs, 0.0
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
            torn_dae_system.evaluate_f,
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
        index_map, _s, _f, parsed, _h, simplified = parse_dae_input(
            dxdt="""
            M * dv = -k * x - c * v
            dx = v
            """,
            states={"x": 1.0, "v": 0.0},
            constants={"M": 2.0, "k": 4.0, "c": 0.5},
        )
        assert set(index_map.state_names) == {"x", "v"}
        assert simplified.mass_matrix is None
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
        index_map, _s, _f, parsed, _h, simplified = parse_dae_input(
            dxdt=[(M * dv, -k * x), ("dx", "v")],
            states={"x": 1.0, "v": 0.0},
            constants={"M": 2.0, "k": 4.0},
        )
        assert set(index_map.state_names) == {"x", "v"}
        assert simplified.mass_matrix is None
        eqs = {lhs.name: rhs for lhs, rhs in parsed.ordered}
        assert sp.simplify(
            to_sympy(eqs["dv"]) - (-4.0 * x) / 2.0
        ) == 0

    def test_zero_coefficient_matches_algebraic_form(
        self,
        ring_modulator_index2_system,
        ring_modulator_index2_scaled_system,
    ):
        # Cs = 0 reduces identically to the explicit 0 = form.
        zero = ring_modulator_index2_system
        scaled = ring_modulator_index2_scaled_system
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
        index_map, _s, _f, parsed, _h, _simplified = parse_dae_input(
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
