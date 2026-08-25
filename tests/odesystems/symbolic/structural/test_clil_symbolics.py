"""Exact linear algebra and engine-IR primitive tests."""

from fractions import Fraction

import sympy as sp

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import to_sympy
from cubie.odesystems.symbolic.structural.clil import (
    SparseMatrixCLIL,
    bareiss_update_virtual_colswap_clil,
    exactdiv,
    nullspace_rank,
)
from cubie.odesystems.symbolic.structural.symbolics import (
    DerivativeRegistry,
    as_small_int,
    fixpoint_sub,
    linear_dependencies,
    linear_expansion,
    lower_varname,
    solve_linear,
    total_derivative,
)


def dense_from_clil(mm):
    rows, cols = mm.size()
    out = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for c, v in zip(mm.row_cols[i], mm.row_vals[i]):
            out[i][c] = v
    return out


class TestClil:
    def test_exactdiv_raises_on_remainder(self):
        assert exactdiv(6, 3) == 2
        try:
            exactdiv(7, 3)
        except AssertionError:
            pass
        else:
            raise AssertionError("inexact division did not raise")

    def test_elimination_step_matches_dense_bareiss(self):
        # M = [[2, 1, 0], [4, 3, 1], [6, 1, 2]]; eliminate col 0
        # with pivot M[0][0]=2, last_pivot=1: row_i <- (2*row_i -
        # coeff*row_0) / 1.
        mm = SparseMatrixCLIL(
            3,
            3,
            [0, 1, 2],
            [[0, 1], [0, 1, 2], [0, 1, 2]],
            [[2, 1], [4, 3, 1], [6, 1, 2]],
        )
        bareiss_update_virtual_colswap_clil(mm, 0, 0, 2, 1)
        dense = dense_from_clil(mm)
        assert dense[1] == [0, 2, 2]
        assert dense[2] == [0, -4, 4]

    def test_pivot_equal_optimization_skips_disjoint_rows(self):
        mm = SparseMatrixCLIL(
            2, 3, [0, 1], [[0], [1, 2]], [[1], [5, 7]]
        )
        bareiss_update_virtual_colswap_clil(mm, 0, 0, 1, 1)
        assert mm.row_vals[1] == [5, 7]

    def test_nullspace_rank(self):
        col_order = []
        rank = nullspace_rank(
            [[1, 2, 3], [2, 4, 6], [0, 1, 1]], col_order
        )
        assert rank == 2
        assert len(col_order) == 3

    def test_nullspace_rank_full(self):
        assert nullspace_rank([[1, 0], [0, 1]]) == 2
        assert nullspace_rank([[0, 0], [0, 0]]) == 0

    def test_nullspace_rank_column_swap_order(self):
        # A zero leading column forces a column swap; col_order's
        # first `rank` entries are the pivot columns in elimination
        # order, the remainder the free columns.
        col_order = []
        rank = nullspace_rank([[0, 2, 1], [0, 4, 3]], col_order)
        assert rank == 2
        assert col_order == [1, 2, 0]

    def test_nullspace_rank_no_swap_identity_order(self):
        col_order = []
        rank = nullspace_rank(
            [[1, 2, 3], [2, 4, 6], [0, 1, 1]], col_order
        )
        assert rank == 2
        assert col_order[:2] == [0, 1]
        assert sorted(col_order) == [0, 1, 2]


class TestLinearExpansion:
    x, y, k = ir.sym("x"), ir.sym("y"), ir.sym("k")

    def test_simple_linear(self):
        a, b, lin = linear_expansion(2 * self.x + self.y, self.x)
        assert lin and a is ir.num(2) and b is self.y

    def test_symbolic_coefficient(self):
        a, b, lin = linear_expansion(
            self.k * self.x + 1, self.x
        )
        assert lin and a is self.k and b is ir.ONE

    def test_nonlinear_power(self):
        _, _, lin = linear_expansion(self.x**2, self.x)
        assert not lin

    def test_nonlinear_function(self):
        _, _, lin = linear_expansion(
            ir.call("sin", self.x) + self.y, self.x
        )
        assert not lin

    def test_product_of_var_terms_nonlinear(self):
        _, _, lin = linear_expansion(
            self.x * (self.x + 1), self.x
        )
        assert not lin

    def test_absent_variable(self):
        a, b, lin = linear_expansion(self.y + 1, self.x)
        assert lin and a is ir.ZERO and b is self.y + 1

    def test_solve_linear(self):
        sol = solve_linear(
            ir.ZERO, 2 * self.x - self.y, self.x
        )
        assert sol is self.y / 2

    def test_solve_linear_singular(self):
        assert solve_linear(self.y, self.y, self.x) is None


class TestLinearDependencies:
    def test_scaled_row_is_dependent_with_exact_multiplier(self):
        c = ir.sym("c")
        rows = [
            {0: -c, 1: c},
            {0: ir.mul(3, c), 1: ir.mul(-3, c)},
        ]
        assert linear_dependencies(rows) == [
            (1, {1: ir.ONE, 0: ir.num(3)})
        ]

    def test_rational_literals_cancel_exactly(self):
        # 0.7 * (1 / 0.7) is not 1.0 in floats; as rationals it is.
        c = ir.num(Fraction(0.7))
        d = ir.num(Fraction(2.1))
        rows = [{0: c, 1: d}, {0: ir.mul(-1, c), 1: ir.mul(-1, d)}]
        assert linear_dependencies(rows) == [
            (1, {1: ir.ONE, 0: ir.ONE})
        ]

    def test_three_row_combination(self):
        a, b = ir.sym("a"), ir.sym("b")
        rows = [
            {0: a, 1: b},
            {1: b, 2: a},
            {0: a, 1: ir.mul(2, b), 2: a},
        ]
        assert linear_dependencies(rows) == [
            (2, {2: ir.ONE, 0: ir.NEG_ONE, 1: ir.NEG_ONE})
        ]

    def test_independent_rows_report_nothing(self):
        rows = [{0: ir.num(2), 1: ir.num(3)}, {0: ir.num(-2), 1: ir.num(3)}]
        assert linear_dependencies(rows) == []

    def test_sum_entries_reduce_through_expansion(self):
        a, b = ir.sym("a"), ir.sym("b")
        rows = [
            {0: a, 1: b},
            {0: ir.add(a, b), 1: ir.add(b, ir.mul(b, b, ir.pow_(a, -1)))},
        ]
        dependent = linear_dependencies(rows)
        assert [index for index, _ in dependent] == [1]
        multipliers = dependent[0][1]
        assert multipliers[1] is ir.ONE
        assert multipliers[0] is ir.expand(
            ir.neg(ir.div(ir.add(a, b), a))
        )


class TestSymbolics:
    t = ir.sym("t")

    def test_fixpoint_sub_chains(self):
        a, b, c = ir.sym("a"), ir.sym("b"), ir.sym("c")
        result = fixpoint_sub(a, {a: b + 1, b: c})
        assert result is c + ir.ONE

    def test_as_small_int(self):
        assert as_small_int(ir.num(-5)) == -5
        assert as_small_int(ir.num(3.0)) == 3
        assert as_small_int(ir.num(1000)) is None
        assert as_small_int(ir.num(Fraction(1, 2))) is None
        assert as_small_int(ir.sym("q")) is None

    def test_total_derivative(self):
        x, dx, w = ir.sym("x"), ir.sym("dx_sym"), ir.sym("w")
        expr = x**2 + self.t * w
        result = total_derivative(expr, {x: dx}, self.t)
        assert sp.simplify(
            to_sympy(result - (2 * x * dx + w))
        ) == 0

    def test_total_derivative_known_map(self):
        x, dx, drv = ir.sym("x"), ir.sym("dx_sym"), ir.sym("drv")
        result = total_derivative(
            x * drv, {x: dx}, self.t, {drv: ir.ONE}
        )
        assert sp.simplify(
            to_sympy(result - (dx * drv + x))
        ) == 0

    def test_registry_chain_and_rename(self):
        x = ir.sym("x")
        reg = DerivativeRegistry({"x", "t"})
        d1 = reg.derivative(x)
        d2 = reg.derivative(d1)
        assert reg.base_and_order(d2) == (x, 2)
        assert reg.lower_order(d2) is d1
        x_t = ir.sym("x_t")
        reg.rename(d1, x_t)
        assert reg.lower_order(d2) is x_t
        # x_t becomes an ordinary chain root (diff2term semantics).
        assert reg.base_and_order(d2) == (x_t, 1)
        assert reg.base_and_order(x_t) == (x_t, 0)

    def test_lower_varname_collision(self):
        reserved = {"x_t"}
        assert lower_varname("x", 1, reserved) == "x_t_"
        assert lower_varname("x", 2, reserved) == "x_tt"
