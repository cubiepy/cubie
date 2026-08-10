"""Test generated linear-operator and residual source."""

import ast

from cubie.odesystems.symbolic.codegen.linear_operators import (
    generate_cached_operator_apply_code,
    generate_cached_jvp_code,
    generate_prepare_jac_code,
    generate_n_stage_linear_operator_code,
    generate_operator_apply_code,
)
from cubie.odesystems.symbolic.codegen.nonlinear_residuals import (
    generate_residual_code,
    generate_n_stage_residual_code,
)
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie.odesystems.symbolic.parsing import ParsedEquations


# ── cached linear operator / JVP / prepare ──────────────────────── #

def test_cached_operator_reads_cache_buffer(
    cacheable_equations, bare_indexed_bases
):
    """Cached operator body indexes ``cached_aux`` and defaults the mass."""
    code = generate_cached_operator_apply_code(
        cacheable_equations, bare_indexed_bases
    )
    ast.parse(code)
    assert "cached_aux[" in code
    assert "def operator_apply(" in code


def test_cached_jvp_reads_cache_buffer(
    cacheable_equations, bare_indexed_bases
):
    """Cached JVP body indexes the ``cached_aux`` buffer."""
    code = generate_cached_jvp_code(cacheable_equations, bare_indexed_bases)
    ast.parse(code)
    assert "cached_aux[" in code


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
    code = generate_n_stage_linear_operator_code(
        solver_scaling_collision_equations,
        solver_scaling_collision_indexed_bases,
        stage_coefficients=[[1.0]],
        stage_nodes=[1.0],
    )

    assert "_cubie_codegen_beta = precision(beta)" in code
    assert "_cubie_codegen_gamma = precision(gamma)" in code
    assert (
        "_cubie_codegen_const_beta = precision(constants['beta'])"
        in code
    )
    assert (
        "_cubie_codegen_const_gamma = precision(constants['gamma'])"
        in code
    )
    # The user constants must never bind the bare solver names.
    assert "\n    beta = " not in code
    assert "\n    gamma = " not in code


def test_n_stage_operator_skips_zero_stage_coupling(
    bare_nonlinear_equations,
    bare_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """Lower-triangular tableau parses with default mass and JVP."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_n_stage_linear_operator_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
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
    code = generate_n_stage_linear_operator_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        cse=False,
    )
    ast.parse(code)


def test_wide_mass_matrix_names_are_unambiguous():
    """Mass entries with multi-digit indices use distinct locals."""
    state_count = 12
    index_map = IndexedBases.from_user_inputs(
        states=[f"x{i}" for i in range(state_count)],
        parameters=[],
        constants=[],
        observables=[],
        drivers=[],
    )
    equations = ParsedEquations.from_equations(
        [
            (ir.sym(f"dx{i}"), ir.sym(f"x{i}"))
            for i in range(state_count)
        ],
        index_map,
    )
    mass = [[0] * state_count for _ in range(state_count)]
    mass[1][10] = 2
    mass[11][0] = 3

    code = generate_operator_apply_code(
        equations,
        index_map,
        M=mass,
    )

    assert "m_1_10 = precision(2.0)" in code
    assert "m_11_0 = precision(3.0)" in code
    assert "m_110" not in code


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
    code = generate_n_stage_residual_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
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
    code = generate_n_stage_residual_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        cse=False,
    )
    ast.parse(code)
