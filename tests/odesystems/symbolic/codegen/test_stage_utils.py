"""Tests for cubie.odesystems.symbolic.codegen._stage_utils."""

from fractions import Fraction

import sympy as sp

from cubie.odesystems.symbolic.codegen._stage_utils import (
    build_stage_metadata,
    prepare_stage_data,
)
from cubie.odesystems.symbolic.engine import expr as ir


# ── prepare_stage_data ──────────────────────────────── #

def test_prepare_stage_data_converts_coefficient_matrix():
    """Coefficient matrix entries convert to exact IR values."""
    coeffs = [[sp.Rational(1, 4), 0], [sp.Rational(1, 2), sp.Rational(1, 4)]]
    nodes = [sp.Rational(1, 4), sp.Rational(3, 4)]
    matrix, _, _ = prepare_stage_data(coeffs, nodes)
    assert matrix[0][0] is ir.num(Fraction(1, 4))
    assert matrix[1][0] is ir.num(Fraction(1, 2))
    assert matrix[1][1] is ir.num(Fraction(1, 4))
    assert ir.is_zero(matrix[0][1])


def test_prepare_stage_data_converts_nodes():
    """Node expressions convert to exact IR values."""
    coeffs = [[sp.Rational(1, 2)]]
    nodes = [sp.Rational(1, 2)]
    _, node_exprs, _ = prepare_stage_data(coeffs, nodes)
    assert node_exprs == (ir.num(Fraction(1, 2)),)


def test_prepare_stage_data_returns_stage_count():
    """Stage count equals the number of rows in the coefficient matrix."""
    coeffs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    nodes = [0.0, 0.5, 1.0]
    _, _, n_stages = prepare_stage_data(coeffs, nodes)
    assert n_stages == 3


# ── build_stage_metadata ────────────────────────────── #

def test_build_stage_metadata_node_symbols():
    """Node symbols are named _cubie_codegen_c_<i> with correct values."""
    coeffs, nodes, _ = prepare_stage_data(
        [[sp.Rational(1, 4), 0], [0, sp.Rational(3, 4)]],
        [sp.Rational(1, 4), sp.Rational(3, 4)],
    )
    metadata, _, node_syms = build_stage_metadata(coeffs, nodes)
    assert len(node_syms) == 2
    assert node_syms[0] is ir.sym("_cubie_codegen_c_0")
    assert node_syms[1] is ir.sym("_cubie_codegen_c_1")
    node_assignments = {
        lhs: value
        for lhs, value in metadata
        if lhs.name.startswith("_cubie_codegen_c_")
    }
    assert node_assignments[ir.sym("_cubie_codegen_c_0")] is ir.num(
        Fraction(1, 4)
    )
    assert node_assignments[ir.sym("_cubie_codegen_c_1")] is ir.num(
        Fraction(3, 4)
    )


def test_build_stage_metadata_coeff_symbols():
    """Coefficient symbols are named _cubie_codegen_a_i_j with correct values.
    """
    coeffs, nodes, _ = prepare_stage_data(
        [
            [sp.Rational(1, 3), sp.Rational(2, 3)],
            [sp.Rational(1, 2), sp.Rational(1, 2)],
        ],
        [0, 1],
    )
    metadata, coeff_syms, _ = build_stage_metadata(coeffs, nodes)
    assert len(coeff_syms) == 2
    assert len(coeff_syms[0]) == 2
    assert coeff_syms[0][0] is ir.sym("_cubie_codegen_a_0_0")
    assert coeff_syms[1][1] is ir.sym("_cubie_codegen_a_1_1")
    coeff_assignments = {
        lhs: value
        for lhs, value in metadata
        if lhs.name.startswith("_cubie_codegen_a_")
    }
    assert coeff_assignments[ir.sym("_cubie_codegen_a_0_0")] is ir.num(
        Fraction(1, 3)
    )
    assert coeff_assignments[ir.sym("_cubie_codegen_a_0_1")] is ir.num(
        Fraction(2, 3)
    )
    assert coeff_assignments[ir.sym("_cubie_codegen_a_1_0")] is ir.num(
        Fraction(1, 2)
    )
    assert coeff_assignments[ir.sym("_cubie_codegen_a_1_1")] is ir.num(
        Fraction(1, 2)
    )


def test_build_stage_metadata_returns_triple():
    """Return value is (metadata_exprs, coeff_symbols, node_symbols)."""
    coeffs, nodes, _ = prepare_stage_data([[1]], [0])
    result = build_stage_metadata(coeffs, nodes)
    metadata, coeff_syms, node_syms = result
    assert len(metadata) == 2  # 1 node + 1 coeff for 1x1
    assert len(coeff_syms) == 1
    assert len(coeff_syms[0]) == 1
    assert len(node_syms) == 1
