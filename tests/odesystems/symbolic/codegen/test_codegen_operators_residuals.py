"""Test generated linear-operator and residual source."""

import ast

import pytest

from cubie.odesystems.solver_helpers import HelperVariant
from cubie.odesystems.symbolic.codegen.linear_operators import (
    generate_linear_operator_code,
    generate_prepare_jac_code,
)
from cubie.odesystems.symbolic.codegen.nonlinear_residuals import (
    generate_residual_code,
)
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie.odesystems.symbolic.parsing import ParsedEquations


# ── cached linear operator / JVP / prepare ──────────────────────── #

def test_cached_operator_reads_cache_buffer(
    cacheable_equations, bare_indexed_bases
):
    """Cached operator body indexes ``cached_aux`` and defaults the mass."""
    code = generate_linear_operator_code(
        cacheable_equations,
        bare_indexed_bases,
        variant=HelperVariant.CACHED,
    )
    ast.parse(code)
    assert "cached_aux[" in code
    assert "def operator_apply(" in code


def test_prepare_jac_populates_cache_slots(
    cacheable_equations, bare_indexed_bases
):
    """prepare_jac writes selected auxiliaries into ``cached_aux``."""
    code, aux_count = generate_prepare_jac_code(
        cacheable_equations, bare_indexed_bases
    )
    ast.parse(code)
    assert aux_count > 0
    assert "cached_aux[" in code
    assert f".aux_count = {aux_count}" in code


def test_prepare_jac_without_cache_emits_pass(
    bare_nonlinear_equations, bare_indexed_bases
):
    """With nothing cached, prepare_jac body is a bare ``pass``."""
    code, aux_count = generate_prepare_jac_code(
        bare_nonlinear_equations, bare_indexed_bases
    )
    ast.parse(code)
    assert aux_count == 0
    assert "\n        pass\n" in code


# ── n-stage linear operator ─────────────────────────────────────── #

def test_n_stage_operator_isolates_user_constants_from_scalings(
    solver_scaling_collision_equations,
    solver_scaling_collision_indexed_bases,
):
    """User beta/gamma constants cannot replace solver scalings."""
    code = generate_linear_operator_code(
        solver_scaling_collision_equations,
        solver_scaling_collision_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=[[1.0]],
        stage_nodes=[1.0],
    )

    assert "_cubie_codegen_beta = precision(beta)" in code
    assert "_cubie_codegen_gamma = precision(gamma)" in code
    # Constants fold to literals; no load or bare binding exists.
    assert "_cubie_codegen_const_" not in code
    assert "constants['beta']" not in code
    assert "constants['gamma']" not in code
    assert "\n    beta = " not in code
    assert "\n    gamma = " not in code


def test_n_stage_operator_skips_zero_stage_coupling(
    bare_nonlinear_equations,
    bare_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """Lower-triangular tableau parses with default mass and JVP."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_linear_operator_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
    )
    ast.parse(code)
    assert "def operator_apply(" in code


def test_n_stage_operator_without_cse(
    bare_nonlinear_equations,
    bare_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """Topological-sort emission (``cse=False``) still parses."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_linear_operator_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        cse=False,
    )
    ast.parse(code)


def test_operator_zero_mass_row_emits_residual_form():
    """A zero mass row emits only the Jacobian term for its output."""
    index_map = IndexedBases.from_user_inputs(
        states=["x0", "x1"],
        parameters=[],
        constants=[],
        observables=[],
        drivers=[],
    )
    equations = ParsedEquations.from_equations(
        [
            (ir.sym("dx0"), ir.sym("x1")),
            (ir.sym("dx1"), ir.sym("x0")),
        ],
        index_map,
    )
    mass = [[1.0, 0.0], [0.0, 0.0]]

    code = generate_linear_operator_code(
        equations,
        index_map,
        M=mass,
    )

    assert "_cubie_codegen_m_" not in code
    # out[0] keeps beta*v; out[1] carries only the Jacobian term.
    lines = {
        line.strip().split(" = ")[0]: line
        for line in code.splitlines()
        if line.strip().startswith("out[")
    }
    assert "_cubie_codegen_beta" in lines["out[0]"]
    assert "_cubie_codegen_beta" not in lines["out[1]"]


def test_operator_rejects_general_mass_matrix():
    """Anything but a 0/1 diagonal is rejected at generation."""
    index_map = IndexedBases.from_user_inputs(
        states=["x0", "x1"],
        parameters=[],
        constants=[],
        observables=[],
        drivers=[],
    )
    equations = ParsedEquations.from_equations(
        [
            (ir.sym("dx0"), ir.sym("x1")),
            (ir.sym("dx1"), ir.sym("x0")),
        ],
        index_map,
    )
    with pytest.raises(ValueError, match="0/1 diagonal"):
        generate_linear_operator_code(
            equations,
            index_map,
            M=[[2.0, 0.0], [0.0, 1.0]],
        )
    with pytest.raises(ValueError, match="0/1 diagonal"):
        generate_residual_code(
            equations,
            index_map,
            M=[[1.0, 0.5], [0.0, 1.0]],
        )


# ── nonlinear residuals ─────────────────────────────────────────── #

def test_residual_without_cse(bare_nonlinear_equations, bare_indexed_bases):
    """Single-stage residual parses under topological sort."""
    code = generate_residual_code(
        bare_nonlinear_equations, bare_indexed_bases, cse=False
    )
    ast.parse(code)
    assert "def residual(" in code


def test_n_stage_residual_skips_zero_stage_coupling(
    bare_nonlinear_equations,
    bare_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """Lower-triangular FIRK residual parses with default identity mass."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_residual_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
    )
    ast.parse(code)
    assert "def residual(" in code


def test_n_stage_residual_without_cse(
    bare_nonlinear_equations,
    bare_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """FIRK residual parses under topological sort."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_residual_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        cse=False,
    )
    ast.parse(code)
