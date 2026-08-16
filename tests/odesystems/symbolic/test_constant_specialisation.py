"""Constant-value specialisation of generated source.

Constant values substitute into the equations as IR literals at the
head of the codegen pipeline: generated source carries the values as
literals (never named bindings or closure captures), dead algebra and
constant-condition branches fold away, and a constant change
re-specialises the system from its saved constants-symbolic
definition — re-running classification, structural simplification,
and tearing where the system was parser-normalised.
"""

import warnings

import numpy as np
import pytest
import sympy as sp

from cubie import create_ODE_system
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.parsing.definition import (
    AssembledSystemDefinition,
    NormalisedSystemDefinition,
)


def _dxdt_source(system):
    """Compile ``dxdt`` and return the generated module source."""
    _ = system.evaluate_f
    return system.gen_file.file_path.read_text(encoding="utf-8")


class TestLiteralFolding:
    """Constants appear in source as literals only."""

    def test_values_folded_and_never_named(self, precision):
        system = create_ODE_system(
            "dx = -k * x * (1.0 + amp)",
            states={"x": 1.0},
            parameters={"k": 0.5},
            constants={"amp": 2.0},
            precision=precision,
            name="fold_literal_basic",
        )
        source = _dxdt_source(system)
        assert "precision(3.0)" in source
        assert "_cubie_codegen_const_" not in source
        assert "constants['amp']" not in source
        assert "amp" not in source.split("def dxdt(", 1)[1]

    def test_zero_constant_prunes_term(self, precision):
        system = create_ODE_system(
            ["dx = -x + c0 * y", "dy = x - y"],
            states={"x": 1.0, "y": 0.0},
            constants={"c0": 0.0},
            precision=precision,
            name="fold_zero_prunes",
        )
        source = _dxdt_source(system)
        body = source.split("def dxdt(", 1)[1]
        assert "c0" not in body
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
        assert isinstance(
            system._definition, AssembledSystemDefinition
        )
        source = _dxdt_source(system)
        assert "precision(0.25)" in source
        assert "constants['rate']" not in source

    def test_constant_hash_varies_with_value(self, precision):
        def build(value, name):
            return create_ODE_system(
                "dx = -k * x",
                states={"x": 1.0},
                constants={"k": value},
                precision=precision,
                name=name,
            )

        low = build(0.5, "fold_hash_low")
        high = build(2.0, "fold_hash_high")
        same = build(0.5, "fold_hash_same")
        assert low.fn_hash != high.fn_hash
        assert low.fn_hash == same.fn_hash


class TestBranchPruning:
    """Constant-condition Piecewise branches disappear from source."""

    @pytest.mark.parametrize("toggle", [0.0, 1.0])
    def test_toggle_selects_single_branch(self, precision, toggle):
        x = sp.Symbol("x", real=True)
        k = sp.Symbol("k", real=True)
        tog = sp.Symbol("tog", real=True)
        dx = sp.Symbol("dx", real=True)
        equations = [
            (dx, sp.Piecewise((-k * x, tog > 0.5), (-2 * k * x, True)))
        ]
        system = create_ODE_system(
            equations,
            states={"x": 1.0},
            parameters={"k": 0.3},
            constants={"tog": toggle},
            precision=precision,
            name=f"fold_toggle_{int(toggle)}",
        )
        source = _dxdt_source(system)
        assert "selp" not in source
        if toggle > 0.5:
            assert "out[0] = -(parameters[0]*state[0])" in source
        else:
            assert (
                "out[0] = -(precision(2)*parameters[0]*state[0])"
                in source
            )

    def test_toggle_flip_switches_branch(self, precision):
        x = sp.Symbol("x", real=True)
        k = sp.Symbol("k", real=True)
        tog = sp.Symbol("tog", real=True)
        dx = sp.Symbol("dx", real=True)
        system = create_ODE_system(
            [
                (
                    dx,
                    sp.Piecewise(
                        (-k * x, tog > 0.5),
                        (-2 * k * x, True),
                    ),
                )
            ],
            states={"x": 1.0},
            parameters={"k": 0.3},
            constants={"tog": 1.0},
            precision=precision,
            name="fold_toggle_flip",
        )
        first = _dxdt_source(system)
        assert "out[0] = -(parameters[0]*state[0])" in first
        system.set_constants({"tog": 0.0})
        second = _dxdt_source(system)
        assert (
            "out[0] = -(precision(2)*parameters[0]*state[0])"
            in second
        )
        assert "selp" not in second


class TestRespecialisation:
    """set_constants re-runs specialisation from the definition."""

    def test_value_change_regenerates_source(self, precision):
        system = create_ODE_system(
            "dx = -k * x * (1.0 + amp)",
            states={"x": 1.0},
            parameters={"k": 0.5},
            constants={"amp": 2.0},
            precision=precision,
            name="respec_value_change",
        )
        first_hash = system.fn_hash
        first = _dxdt_source(system)
        assert "precision(3.0)" in first
        system.set_constants({"amp": 4.0})
        assert system.fn_hash != first_hash
        second = _dxdt_source(system)
        assert "precision(5.0)" in second
        assert float(system.constants["amp"]) == 4.0

    def test_unchanged_value_keeps_hash(self, precision):
        system = create_ODE_system(
            "dx = -k * x",
            states={"x": 1.0},
            constants={"k": 0.5},
            precision=precision,
            name="respec_no_change",
        )
        first_hash = system.fn_hash
        recognised = system.set_constants({"k": 0.5})
        assert recognised == {"k"}
        assert system.fn_hash == first_hash

    def test_unknown_constant_raises(self, precision):
        system = create_ODE_system(
            "dx = -k * x",
            states={"x": 1.0},
            constants={"k": 0.5},
            precision=precision,
            name="respec_unknown",
        )
        with pytest.raises(KeyError, match="Unrecognized"):
            system.set_constants({"not_a_constant": 1.0})

    def test_make_parameter_restores_symbol(self, precision):
        system = create_ODE_system(
            "dx = -k * x * (1.0 + amp)",
            states={"x": 1.0},
            parameters={"k": 0.5},
            constants={"amp": 2.0},
            precision=precision,
            name="respec_make_parameter",
        )
        _ = _dxdt_source(system)
        system.make_parameter("amp")
        assert "amp" in system.parameters.values_dict
        assert "amp" not in system.constants.values_dict
        source = _dxdt_source(system)
        # The freed symbol reads from the parameters array again.
        assert "parameters[" in source
        assert "precision(3.0)" not in source

    def test_make_constant_folds_value(self, precision):
        system = create_ODE_system(
            "dx = -k * x * (1.0 + amp)",
            states={"x": 1.0},
            parameters={"k": 0.5, "amp": 2.0},
            precision=precision,
            name="respec_make_constant",
        )
        _ = _dxdt_source(system)
        system.make_constant("amp")
        assert "amp" in system.constants.values_dict
        source = _dxdt_source(system)
        assert "precision(3.0)" in source


class TestStructuralRespecialisation:
    """Constant changes re-run structural simplification."""

    SCALED = """
    Cs*dU3 = I3 - 0.5*I1
    dI1 = -U3 - 0.2*I1
    dI3 = U3 - 0.1*I3
    """
    STATES = {"U3": 0.0, "I1": 0.0, "I3": 0.0}

    def _build(self, cs, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return create_ODE_system(
                self.SCALED,
                states=dict(self.STATES),
                constants={"Cs": cs},
                precision=np.float64,
                simplify=True,
                name=name,
            )

    def test_zero_coefficient_yields_singular_mass(self):
        system = self._build(0.0, "structural_cs_zero")
        assert isinstance(
            system._definition, NormalisedSystemDefinition
        )
        assert system.mass is not None
        diag = np.diag(np.asarray(system.mass))
        assert 0.0 in diag

    def test_nonzero_coefficient_yields_explicit_system(self):
        system = self._build(2e-2, "structural_cs_nonzero")
        assert system.mass is None
        assert set(system.initial_values.values_dict) == set(
            self.STATES
        )
        source = _dxdt_source(system)
        # 1/Cs = 50 folds into the derivative row.
        assert "precision(50.0)" in source

    def test_constant_change_restructures_system(self):
        system = self._build(0.0, "structural_cs_flip")
        zero_states = list(system.initial_values.values_dict)
        assert system.mass is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.set_constants({"Cs": 2e-2})
        assert system.mass is None
        new_states = list(system.initial_values.values_dict)
        assert set(new_states) == set(self.STATES)
        assert new_states != zero_states
        source = _dxdt_source(system)
        assert "precision(50.0)" in source
        # And back: the row turns algebraic again.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.set_constants({"Cs": 0.0})
        assert system.mass is not None

    def test_state_values_survive_respecialisation(self):
        system = self._build(2e-2, "structural_cs_state_values")
        system.set_initial_value("U3", 0.75)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system.set_constants({"Cs": 4e-2})
        assert float(
            system.initial_values.values_dict["U3"]
        ) == pytest.approx(0.75)


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
        # A zero constant in a denominator must not fold eagerly:
        # the runtime value (IEEE inf inside a dead selp branch)
        # only exists at evaluation time.
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
