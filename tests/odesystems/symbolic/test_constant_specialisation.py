"""Constant-value specialisation of generated source."""

import warnings

import numpy as np
import pytest

from cubie import Solver, create_ODE_system
from cubie.odesystems.symbolic.engine import expr as ir


def _dxdt_source(system):
    """Compile ``dxdt`` and return the generated module source."""
    _ = system.evaluate_f
    return system.gen_file.file_path.read_text(encoding="utf-8")


class TestLiteralFolding:
    """Constants appear in source as literals only."""

    def test_values_folded_and_never_named(self, amp_constant_system):
        # The constant folds into the source as a literal; the whole
        # derivative row is this one folded expression.
        source = _dxdt_source(amp_constant_system)
        assert (
            "out[0] = -(precision(3.0)*parameters[0]*state[0])"
            in source
        )

    def test_zero_constant_prunes_term(self, precision):
        system = create_ODE_system(
            ["dx = -x + c0 * y", "dy = x - y"],
            states={"x": 1.0, "y": 0.0},
            constants={"c0": 0.0},
            precision=precision,
            name="fold_zero_prunes",
        )
        source = _dxdt_source(system)
        # The coupling term is gone entirely.
        assert "out[0] = -state[0]" in source

    def test_callable_input_folds_constants(self, precision):
        def rhs(t, y, c):
            return [-c.rate * y[0]]

        system = create_ODE_system(
            rhs,
            states={"x": 1.0},
            constants={"rate": 0.25},
            precision=precision,
            name="fold_callable",
        )
        source = _dxdt_source(system)
        assert "out[0] = -(precision(0.25)*state[0])" in source

    def test_constant_hash_varies_with_value(self, amp_constant_system):
        system = amp_constant_system
        default_hash = system.fn_hash
        system.set_constants({"amp": 4.0})
        assert system.fn_hash != default_hash
        system.set_constants({"amp": 2.0})
        assert system.fn_hash == default_hash


class TestBranchPruning:
    """Constant-condition Piecewise branches disappear from source."""

    def test_toggle_selects_single_branch(self, toggle_system):
        # The whole row is the surviving branch; the dead branch and
        # its selp are pruned before codegen.
        on = _dxdt_source(toggle_system)
        assert "out[0] = -(parameters[0]*state[0])" in on
        toggle_system.set_constants({"tog": 0.0})
        off = _dxdt_source(toggle_system)
        assert (
            "out[0] = -(precision(2)*parameters[0]*state[0])" in off
        )


class TestRespecialisation:
    """set_constants re-runs specialisation from the checkpoint."""

    def test_value_change_regenerates_source(self, amp_constant_system):
        system = amp_constant_system
        first_hash = system.fn_hash
        first = _dxdt_source(system)
        assert "precision(3.0)" in first
        system.set_constants({"amp": 4.0})
        assert system.fn_hash != first_hash
        second = _dxdt_source(system)
        assert "precision(5.0)" in second
        assert float(system.constants["amp"]) == 4.0

    def test_unchanged_value_keeps_hash(self, amp_constant_system):
        system = amp_constant_system
        first_hash = system.fn_hash
        recognised = system.set_constants({"amp": 2.0})
        assert recognised == {"amp"}
        assert system.fn_hash == first_hash

    def test_unknown_constant_raises(self, amp_constant_system):
        with pytest.raises(KeyError, match="Unrecognized"):
            amp_constant_system.set_constants({"not_a_constant": 1.0})

    def test_make_parameter_restores_symbol(self, amp_constant_system):
        system = amp_constant_system
        system.make_parameter("amp")
        assert "amp" in system.parameters.values_dict
        source = _dxdt_source(system)
        # The freed symbol reads from the parameters array again.
        assert (
            "out[0] = -(parameters[0]*state[0]"
            "*(parameters[1] + precision(1.0)))"
        ) in source

    def test_make_constant_folds_value(self, amp_constant_system):
        system = amp_constant_system
        system.make_parameter("amp")
        system.make_constant("amp")
        assert "amp" in system.constants.values_dict
        source = _dxdt_source(system)
        assert (
            "out[0] = -(precision(3.0)*parameters[0]*state[0])"
            in source
        )


class TestStructuralRespecialisation:
    """Constant changes re-run structural simplification."""

    def test_zero_coefficient_yields_singular_mass(
        self, scaled_cs_system
    ):
        assert scaled_cs_system.mass is not None
        diag = np.diag(np.asarray(scaled_cs_system.mass))
        assert 0.0 in diag

    def test_constant_change_restructures_system(self, scaled_cs_system):
        system = scaled_cs_system
        zero_states = list(system.initial_values.values_dict)
        assert system.mass is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.set_constants({"Cs": 2e-2})
        assert system.mass is None
        new_states = list(system.initial_values.values_dict)
        assert set(new_states) == {"U3", "I1", "I3"}
        assert new_states != zero_states
        source = _dxdt_source(system)
        # 1/Cs = 50 folds into the derivative row.
        assert "precision(50.0)" in source
        # And back: the row turns algebraic again.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.set_constants({"Cs": 0.0})
        assert system.mass is not None

    def test_state_values_survive_respecialisation(
        self, scaled_cs_system
    ):
        system = scaled_cs_system
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.set_constants({"Cs": 2e-2})
            system.set_initial_value("U3", 0.75)
            system.set_constants({"Cs": 4e-2})
        assert float(
            system.initial_values.values_dict["U3"]
        ) == pytest.approx(0.75)


class TestStructuralSortKey:
    """Folded literals keep the structural sort key finite."""

    def test_zero_base_negative_exponent_parses(self):
        # Hill-type term K/ACh**n with ACh = 0 folds to 0.0**-n,
        # which the sort key evaluates with IEEE inf semantics.
        system = create_ODE_system(
            "dx = -x + 3.57/(1.0 + 18003.4*ACh**(-1.6951))",
            states={"x": 1.0},
            constants={"ACh": 0.0},
            precision=np.float64,
            name="sort_key_zero_pow",
        )
        # The folded power survives to source and evaluates to its
        # IEEE limit (inf denominator, so the term contributes 0).
        assert (
            "precision(0.0)**precision(-1.6951)"
            in _dxdt_source(system)
        )


class TestEngineConditionFolding:
    """Relational and boolean constructors fold numeric operands."""

    def test_rel_folds_numeric_operands(self):
        assert ir.rel(">", ir.num(1.0), ir.num(0.0)) is ir.TRUE
        assert ir.rel("<", ir.num(1.0), ir.num(0.0)) is ir.FALSE
        assert ir.rel("==", ir.num(2), ir.num(2.0)) is ir.TRUE
        assert ir.rel("!=", ir.num(2), ir.num(2.0)) is ir.FALSE
        assert ir.rel(">=", ir.num(0.5), ir.num(0.5)) is ir.TRUE
        assert ir.rel("<=", ir.num(0.6), ir.num(0.5)) is ir.FALSE

    def test_rel_keeps_symbolic_operands(self):
        node = ir.rel(">", ir.sym("x"), ir.num(0.0))
        assert isinstance(node, ir.Rel)

    def test_bool_op_folds_literals(self):
        x_pos = ir.rel(">", ir.sym("x"), ir.num(0.0))
        assert ir.bool_op("not", ir.TRUE) is ir.FALSE
        assert ir.bool_op("not", ir.FALSE) is ir.TRUE
        assert ir.bool_op("and", ir.TRUE, x_pos) is x_pos
        assert ir.bool_op("and", ir.FALSE, x_pos) is ir.FALSE
        assert ir.bool_op("or", ir.FALSE, x_pos) is x_pos
        assert ir.bool_op("or", ir.TRUE, x_pos) is ir.TRUE
        assert ir.bool_op("and", ir.TRUE, ir.TRUE) is ir.TRUE
        assert ir.bool_op("or", ir.FALSE, ir.FALSE) is ir.FALSE

    def test_zero_base_negative_power_stays_symbolic(self):
        # A zero base with a negative exponent must not fold.
        node = ir.pow_(ir.num(0.0), ir.num(-1))
        assert isinstance(node, ir.Pow)
        toggle = ir.sym("toggle")
        expression = ir.div(ir.sym("x"), toggle)
        folded = ir.xreplace(expression, {toggle: ir.num(0.0)})
        assert any(
            isinstance(atom, ir.Pow)
            for atom in [folded, *getattr(folded, "args", ())]
        )

    def test_piecewise_prunes_on_substitution(self):
        x = ir.sym("x")
        toggle = ir.sym("toggle")
        expression = ir.piecewise(
            (ir.mul(2, x), ir.rel(">", toggle, ir.num(0.5))),
            (ir.mul(3, x), ir.TRUE),
        )
        on = ir.xreplace(expression, {toggle: ir.num(1.0)})
        off = ir.xreplace(expression, {toggle: ir.num(0.0)})
        assert on is ir.mul(2, x)
        assert off is ir.mul(3, x)


class TestLiveSolverRespecialisation:
    """A live Solver follows constant updates through Solver.update."""

    def test_value_change_rebuilds_live_solver(self, amp_constant_system):
        system = amp_constant_system
        solver = Solver(
            system, algorithm="euler", dt=0.01, save_every=0.1
        )
        first = solver.solve(
            {"x": [1.0]}, {"k": [1.0]}, duration=1.0
        ).time_domain_array.copy()

        solver.update({"amp": 4.0})
        live = solver.solve(
            {"x": [1.0]}, {"k": [1.0]}, duration=1.0
        ).time_domain_array.copy()

        fresh_solver = Solver(
            system, algorithm="euler", dt=0.01, save_every=0.1
        )
        fresh = fresh_solver.solve(
            {"x": [1.0]}, {"k": [1.0]}, duration=1.0
        ).time_domain_array

        assert not np.allclose(live, first)
        assert np.array_equal(live, fresh)

    def test_direct_system_mutation_raises_at_solve(
        self, amp_constant_system
    ):
        system = amp_constant_system
        solver = Solver(
            system, algorithm="euler", dt=0.01, save_every=0.1
        )
        solver.solve({"x": [1.0]}, {"k": [1.0]}, duration=1.0)
        system.set_constants({"amp": 3.0})
        with pytest.raises(RuntimeError, match="Solver.update"):
            solver.solve({"x": [1.0]}, {"k": [1.0]}, duration=1.0)
        # The sanctioned path recovers the solver.
        solver.update({"amp": 3.0})
        result = solver.solve(
            {"x": [1.0]}, {"k": [1.0]}, duration=1.0
        )
        assert np.isfinite(result.time_domain_array).all()

    def test_structural_flip_rebuilds_live_solver(self, scaled_cs_system):
        system = scaled_cs_system
        assert system.mass is not None
        solver_settings = {
            "algorithm": "backwards_euler",
            "dt": 1e-3,
            "save_every": 0.005,
            "preconditioner_type": "jacobi",
            "linear_correction_type": "bicgstab",
        }
        solver = Solver(system, **solver_settings)
        y0 = {
            name: np.array([float(value)])
            for name, value in (
                system.initial_values.values_dict.items()
            )
        }
        algebraic = solver.solve(y0, {}, duration=0.01)
        assert algebraic.time_domain_array.shape[1] == 2
        assert np.isfinite(algebraic.time_domain_array).all()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver.update({"Cs": 2e-2})
        assert system.mass is None
        y0 = {
            name: np.array([float(value)])
            for name, value in (
                system.initial_values.values_dict.items()
            )
        }
        live = solver.solve(y0, {}, duration=0.01)
        assert live.time_domain_array.shape[1] == 3
        assert np.isfinite(live.time_domain_array).all()

        fresh_solver = Solver(system, **solver_settings)
        fresh = fresh_solver.solve(y0, {}, duration=0.01)
        assert (
            live.time_domain_array.shape
            == fresh.time_domain_array.shape
        )
        assert np.array_equal(
            live.time_domain_array, fresh.time_domain_array
        )
