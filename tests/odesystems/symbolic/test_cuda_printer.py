"""Tests for the IR CUDA printer.

Covers the printer emission rules:
precision wrapping, power rewrites (squares/cubes to multiplication
chains, halves to sqrt, negatives to guarded reciprocals), piecewise
ternaries, function mapping, scalar-to-array symbol remapping, and
the constant integer-exponent alias.
"""

import numpy as np
import pytest
import sympy as sp

from cubie import create_ODE_system, solve_ivp
from cubie.odesystems.symbolic.engine import (
    TRUE,
    add,
    arr,
    call,
    div,
    from_sympy,
    mul,
    num,
    piecewise,
    pow_,
    print_cuda,
    print_cuda_multiple,
    rel,
    sym,
)
from cubie.odesystems.symbolic.sym_utils import (
    CONSTANT_ALIAS_PREFIX,
    EXPONENT_ALIAS_PREFIX,
)


class TestPrecisionWrapping:
    def test_integer_literals_wrapped(self):
        assert print_cuda(add(sym("x"), num(5))) == "x + precision(5)"
        assert print_cuda(mul(num(2), sym("x"))) == "precision(2)*x"

    def test_subtraction_prints_minus(self):
        result = print_cuda(add(sym("x"), num(-3)))
        assert result == "x - precision(3)"

    def test_float_literals_wrapped(self):
        assert print_cuda(num(0.5)) == "precision(0.5)"
        assert print_cuda(num(0.0)) == "precision(0.0)"

    def test_scientific_notation_wrapped(self):
        result = print_cuda(num(1.5e-10))
        assert result.startswith("precision(")
        assert "e-" in result.lower()

    def test_rational_wrapped_as_ratio(self):
        assert print_cuda(from_sympy(sp.Rational(1, 2))) == (
            "precision(1/2)"
        )
        assert print_cuda(from_sympy(sp.Rational(-1, 3))) == (
            "precision(-1/3)"
        )

    def test_array_indices_not_wrapped(self):
        result = print_cuda(arr("state", 0))
        assert result == "state[0]"
        assert "precision" not in result

    def test_indexed_with_literal_expression(self):
        result = print_cuda(mul(arr("state", 0), num(2.5)))
        assert "state[0]" in result
        assert result.count("precision(") == 1


class TestPowerRewrites:
    def test_square_becomes_multiplication(self):
        assert print_cuda(pow_(sym("x"), num(2))) == "(x*x)"

    def test_cube_becomes_multiplication(self):
        assert print_cuda(pow_(sym("x"), num(3))) == "(x*x*x)"

    def test_float_exponents_optimised(self):
        assert print_cuda(pow_(sym("x"), num(2.0))) == "(x*x)"
        assert print_cuda(pow_(sym("x"), num(3.0))) == "(x*x*x)"

    def test_parenthesised_base_square(self):
        base = add(sym("a"), sym("b"))
        assert print_cuda(pow_(base, num(2))) == "((a + b)*(a + b))"

    def test_indexed_base_square(self):
        result = print_cuda(pow_(arr("state", 0), num(2)))
        assert result == "(state[0]*state[0])"

    def test_higher_integer_power_wraps_exponent(self):
        assert print_cuda(pow_(sym("x"), num(5))) == "x**precision(5)"
        assert print_cuda(from_sympy(sp.Symbol("x") ** 4)) == (
            "x**precision(4)"
        )

    def test_half_power_is_sqrt(self):
        assert print_cuda(pow_(sym("x"), num(0.5))) == "math.sqrt(x)"
        assert print_cuda(from_sympy(sp.sqrt(sp.Symbol("x")))) == (
            "math.sqrt(x)"
        )

    def test_sqrt_of_sum(self):
        result = print_cuda(
            from_sympy(sp.sqrt(sp.Symbol("a") + sp.Symbol("b")))
        )
        assert result == "math.sqrt(a + b)"

    def test_negative_half_power_is_reciprocal_sqrt(self):
        result = print_cuda(pow_(sym("x"), num(-0.5)))
        assert result == "(precision(1)/math.sqrt(x))"

    def test_reciprocal(self):
        assert print_cuda(pow_(sym("c"), num(-1))) == (
            "(precision(1)/c)"
        )

    def test_reciprocal_square(self):
        assert print_cuda(pow_(sym("c"), num(-2))) == (
            "(precision(1)/(c*c))"
        )

    def test_division_by_square_does_not_cancel(self):
        # Regression guard for the regex-era denominator
        # cancellation: x / c**2 must keep its denominator.
        result = print_cuda(div(sym("x"), pow_(sym("c"), num(2))))
        assert result == "x/(c*c)"

    def test_rational_power_prints_wrapped(self):
        result = print_cuda(
            from_sympy(sp.Symbol("x") ** sp.Rational(3, 2))
        )
        assert result == "x**precision(3/2)"


class TestConstantExponentAlias:
    """Constant exponents print as their integer-exponent alias."""

    def test_constant_exponent_prints_alias(self):
        result = print_cuda(
            pow_(sym("x"), sym("n")), constant_names={"n"}
        )
        assert result == f"x**{EXPONENT_ALIAS_PREFIX}n"

    def test_non_constant_symbol_exponent_unchanged(self):
        assert print_cuda(pow_(sym("x"), sym("n"))) == "x**n"

    def test_mapped_symbol_exponent_not_aliased(self):
        result = print_cuda(
            pow_(sym("x"), sym("n")),
            symbol_map={"n": arr("parameters", 0)},
            constant_names={"n"},
        )
        assert result == "x**parameters[0]"

    def test_constant_base_prints_prefixed_local(self):
        result = print_cuda(
            pow_(sym("n"), sym("x")), constant_names={"n"}
        )
        assert result == f"{CONSTANT_ALIAS_PREFIX}n**x"

    def test_constant_read_prints_prefixed_local(self):
        result = print_cuda(sym("n"), constant_names={"n"})
        assert result == f"{CONSTANT_ALIAS_PREFIX}n"

    def test_mapped_constant_prints_array_reference(self):
        result = print_cuda(
            sym("n"),
            symbol_map={"n": arr("parameters", 0)},
            constant_names={"n"},
        )
        assert result == "parameters[0]"

    def test_alias_used_via_print_cuda_multiple(self):
        lines = print_cuda_multiple(
            [(sym("out"), pow_(sym("x"), sym("n")))],
            constant_names={"n"},
        )
        assert lines == [f"out = x**{EXPONENT_ALIAS_PREFIX}n"]


class TestFunctionsAndPiecewise:
    def test_known_functions_map_to_math(self):
        assert print_cuda(call("exp", sym("x"))) == "math.exp(x)"
        assert print_cuda(call("Abs", sym("x"))) == "math.fabs(x)"
        assert print_cuda(call("ceiling", sym("x"))) == "math.ceil(x)"
        assert print_cuda(call("Min", sym("x"), sym("y"))) == (
            "min(x, y)"
        )

    def test_unknown_function_raises(self):
        with pytest.raises(ValueError, match="unsupported function"):
            print_cuda(call("myfunc_", sym("x")))

    def test_sign_emits_copysign_selection(self):
        assert print_cuda(call("sign", sym("x"))) == (
            "(precision(0) if x == precision(0) else "
            "math.copysign(precision(1), x))"
        )

    def test_mod_emits_modulo_operator(self):
        result = print_cuda(
            call("Mod", add(sym("x"), sym("y")), num(3))
        )
        assert result == "(x + y) % precision(3)"

    def test_mod_factor_keeps_grouping(self):
        product = mul(sym("z"), call("Mod", sym("x"), sym("y")))
        assert print_cuda(product) == "z*(x % y)"

    def test_mod_in_denominator_keeps_grouping(self):
        quotient = div(sym("z"), call("Mod", sym("x"), sym("y")))
        assert print_cuda(quotient) == "z/(x % y)"

    def test_heaviside_converts_to_piecewise(self):
        result = print_cuda(from_sympy(sp.Heaviside(sp.Symbol("x"))))
        assert " if " in result
        assert "Heaviside" not in result

    def test_derivative_placeholder_prints_plainly(self):
        result = print_cuda(
            call("d_myfunc", sym("x"), num(0)),
            function_aliases={"d_myfunc": "d_myfunc"},
        )
        assert result == "d_myfunc(x, precision(0))"

    def test_function_alias_resolution(self):
        result = print_cuda(
            call("myfunc_", sym("x")),
            function_aliases={"myfunc_": "myfunc"},
        )
        assert result == "myfunc(x)"

    def test_piecewise_emits_nested_ternaries(self):
        expr = piecewise(
            (sym("a"), rel("<", sym("x"), num(0))),
            (sym("b"), TRUE),
        )
        assert print_cuda(expr) == (
            "(a if x < precision(0) else (b))"
        )

    def test_piecewise_assignment_is_wrapped_outside(self):
        expr = from_sympy(
            sp.Piecewise(
                (
                    sp.Symbol("_cse1")
                    * (sp.Symbol("_cse2") + sp.Symbol("aux_2")),
                    sp.Symbol("_cse3") > 0,
                ),
                (0.0, True),
            )
        )
        line = print_cuda_multiple([(sym("aux_4"), expr)])[0]
        assert line.startswith("aux_4 = (")
        assert " if " in line
        assert line.rstrip().endswith("(precision(0.0)))")

    def test_piecewise_inside_expression(self):
        inner = piecewise(
            (sym("_cse1"), rel(">", sym("_cse3"), num(0))),
            (num(0.0), TRUE),
        )
        result = print_cuda(mul(sym("E_v"), inner))
        assert result == (
            "E_v*(_cse1 if _cse3 > precision(0) else "
            "(precision(0.0)))"
        )

    def test_piecewise_literals_wrapped(self):
        expr = piecewise(
            (num(0.5), rel(">", sym("x"), num(0.0))),
            (num(0.0), TRUE),
        )
        result = print_cuda(expr)
        assert result.count("precision(") >= 2
        assert " if " in result


class TestSymbolMapping:
    def test_scalar_symbols_remap_to_arrays(self):
        lines = print_cuda_multiple(
            [(arr("out", 0), add(sym("p"), sym("x")))],
            symbol_map={
                "x": arr("state", 0),
                "p": arr("parameters", 1),
            },
        )
        assert lines == ["out[0] = parameters[1] + state[0]"]

    def test_unmapped_symbols_print_by_name(self):
        lines = print_cuda_multiple(
            [(sym("local"), mul(sym("h"), sym("x")))],
        )
        assert lines == ["local = h*x"]

    def test_assignment_target_remaps(self):
        lines = print_cuda_multiple(
            [(sym("dx"), sym("x"))],
            symbol_map={"dx": arr("out", 2)},
        )
        assert lines == ["out[2] = x"]


class TestExpressionShapes:
    def test_negative_coefficient_prints_unary_minus(self):
        assert print_cuda(mul(num(-1), sym("x"))) == "-x"

    def test_division_groups_denominator(self):
        result = print_cuda(div(sym("a"), mul(sym("x"), sym("y"))))
        assert result == "a/(x*y)"

    def test_subtracted_sum_keeps_parentheses(self):
        from cubie.odesystems.symbolic.engine import sub

        result = print_cuda(
            sub(sym("x"), add(sym("y"), sym("z")))
        )
        assert result == "x - (y + z)"

    def test_leading_negated_sum_keeps_parentheses(self):
        from cubie.odesystems.symbolic.engine import neg

        result = print_cuda(neg(add(sym("y"), sym("z"))))
        assert result == "-(y + z)"

    def test_generated_source_compiles(self):
        expr = from_sympy(
            sp.sympify(
                "x**2*sin(y) + Piecewise((x, x > 0), (-x, True))"
            )
        )
        line = print_cuda_multiple([(arr("out", 0), expr)])[0]
        compile(line, "<generated>", "exec")

    def test_empty_assignment_list(self):
        assert print_cuda_multiple([]) == []


def _check_sqrt_rhs_analytic(precision):
    """Integrate ``dx = -sqrt(x)`` and compare against the exact form."""
    system = create_ODE_system(
        "dx = -sqrt(x)",
        states={"x": 4.0},
        precision=precision,
        name=f"sqrt_lowering_analytic_{np.dtype(precision).name}",
    )
    result = solve_ivp(
        system,
        y0={"x": 4.0},
        method="tsit5",
        duration=1.0,
        dt=0.01,
        save_every=0.1,
        output_types=["state", "time"],
    )
    assert not np.any(result.status_codes)
    time = np.asarray(result.time).reshape(-1)
    state = np.squeeze(result.time_domain_array)
    expected = (2.0 - time / 2.0) ** 2
    np.testing.assert_allclose(state, expected, rtol=1e-4, atol=1e-5)


def test_sqrt_rhs_matches_analytic_solution(precision):
    """A sqrt right-hand side integrates to its analytic solution.

    ``dx = -sqrt(x)`` with ``x(0) = 4`` has the exact solution
    ``x(t) = (2 - t/2)**2``, so the generated ``math.sqrt`` lowering is
    checked end to end against a closed-form reference.
    """
    _check_sqrt_rhs_analytic(precision)


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"precision": np.float64}],
    indirect=True,
)
def test_sqrt_rhs_matches_analytic_solution_float64(precision):
    """The sqrt lowering holds at double precision as well."""
    _check_sqrt_rhs_analytic(precision)
