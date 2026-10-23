"""Source-level tests for the direct LU solve generators."""

import ast

import pytest
import sympy as sp

from cubie.odesystems.solver_helpers import HelperVariant
from cubie.odesystems.symbolic.codegen.jacobian import generate_jacobian
from cubie.odesystems.symbolic.codegen.lu_solver import (
    _markowitz_symbolic_lu,
    generate_lu_solve_code,
)
from cubie.odesystems.symbolic.engine.adapter import buffer_maps
from cubie.odesystems.symbolic.parsing import ParsedEquations
from tests.odesystems.symbolic.codegen._source_checks import (
    factory_name_bindings,
    loaded_name_count,
)


def test_lu_solve_hoists_the_jacobian_scale(
    bare_nonlinear_equations, bare_layout
):
    """One named scale product feeds every W entry."""
    code, _ = generate_lu_solve_code(
        bare_nonlinear_equations,
        bare_layout,
        a_ij=0.435866,
    )
    ast.parse(code)
    assert (
        "_cubie_codegen_lu_scale = "
        "precision(0.435866)*_cubie_codegen_h" in code
    )
    assert loaded_name_count(code, "_cubie_codegen_lu_scale") >= 2



def test_lu_solve_uses_a_given_jacobian(
    bare_nonlinear_equations, bare_layout
):
    """A Jacobian passed in emits the source the generator builds itself."""
    maps = buffer_maps(bare_layout)
    jacobian = generate_jacobian(
        bare_nonlinear_equations,
        input_order=maps.state_order,
        output_order=maps.dxdt_order,
    )
    built = generate_lu_solve_code(bare_nonlinear_equations, bare_layout)
    given = generate_lu_solve_code(
        bare_nonlinear_equations, bare_layout, jacobian=jacobian
    )
    assert given == built

def test_prefactored_preamble_binds_only_compared_diagonals(
    bare_nonlinear_equations, bare_layout
):
    """Every bound diagonal literal is read by a branch test."""
    code, _ = generate_lu_solve_code(
        bare_nonlinear_equations,
        bare_layout,
        variant=HelperVariant.PREFACTORED,
        stage_coefficients=[[0.25, 0.0], [0.4, 0.5]],
        stage_nodes=[0.25, 0.9],
    )
    ast.parse(code)
    referenced, defined = factory_name_bindings(code)
    prefix = "_cubie_codegen_lu_diag_"
    bound = {name for name in defined if name.startswith(prefix)}
    read = {name for name in referenced if name.startswith(prefix)}
    assert bound == read == {"_cubie_codegen_lu_diag_0"}


def test_torn_offslot_constraint_pivots_off_diagonal(
    bare_layout,
):
    """A constraint without its slot variable factorises via an
    off-diagonal pivot."""
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    a = sp.Symbol("a", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)
    equations = ParsedEquations.from_equations(
        [(dx, a * y), (dy, x)], bare_layout
    )
    code, _ = generate_lu_solve_code(
        equations,
        bare_layout,
        M=[[1, 0], [0, 0]],
    )
    ast.parse(code)
    assert "x[0]" in code and "x[1]" in code


def test_block_size_keeps_pivots_inside_diagonal_blocks():
    """Same-block pivots win over a lower-fill cross-block entry."""
    pattern = {
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1),
        (2, 0), (2, 3),
        (3, 3),
    }
    row_perm, col_perm, _, _ = _markowitz_symbolic_lu(
        set(pattern), 4
    )
    assert (row_perm[0], col_perm[0]) == (0, 2)
    row_perm, col_perm, _, _ = _markowitz_symbolic_lu(
        set(pattern), 4, block_size=2
    )
    assert all(
        i // 2 == j // 2 for i, j in zip(row_perm, col_perm)
    )


def test_structurally_singular_pattern_refused(bare_layout):
    """A structurally singular shifted matrix fails at generation."""
    x = sp.Symbol("x", real=True)
    a = sp.Symbol("a", real=True)
    dx = sp.Symbol("dx", real=True)
    dy = sp.Symbol("dy", real=True)
    equations = ParsedEquations.from_equations(
        [(dx, a * x), (dy, a)], bare_layout
    )
    with pytest.raises(ValueError, match="structurally singular"):
        generate_lu_solve_code(
            equations,
            bare_layout,
            M=[[1, 0], [0, 0]],
        )
