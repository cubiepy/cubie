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
from tests.odesystems.symbolic.codegen._source_checks import (
    factory_name_bindings,
    loaded_name_count,
)


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


@pytest.mark.parametrize(
    "equations_fixture,bases_fixture",
    [
        ("cacheable_equations", "bare_indexed_bases"),
        (
            "cacheable_observable_equations",
            "observable_driver_indexed_bases",
        ),
    ],
    ids=["plain-aux", "observable-and-aux"],
)
def test_cached_operator_defines_every_referenced_auxiliary(
    request, equations_fixture, bases_fixture
):
    """Every auxiliary a cached operator body references is defined."""
    code = generate_linear_operator_code(
        request.getfixturevalue(equations_fixture),
        request.getfixturevalue(bases_fixture),
        variant=HelperVariant.CACHED,
    )
    referenced, defined = factory_name_bindings(code)
    assert referenced <= defined


@pytest.mark.parametrize(
    "equations_fixture,bases_fixture",
    [
        ("cacheable_equations", "bare_indexed_bases"),
        (
            "cacheable_observable_equations",
            "observable_driver_indexed_bases",
        ),
    ],
    ids=["plain-aux", "observable-and-aux"],
)
def test_prepare_jac_stores_every_slot_in_order(
    request, equations_fixture, bases_fixture
):
    """prepare_jac stores each cached slot and defines every name."""
    code, aux_count = generate_prepare_jac_code(
        request.getfixturevalue(equations_fixture),
        request.getfixturevalue(bases_fixture),
    )
    assert aux_count > 0
    for slot in range(aux_count):
        assert f"cached_aux[{slot}] = " in code
    referenced, defined = factory_name_bindings(code)
    assert referenced <= defined


def test_operator_hoists_the_jacobian_scale(
    bare_nonlinear_equations, bare_indexed_bases
):
    """One named scale product feeds every operator row."""
    code = generate_linear_operator_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        a_ij=0.435866,
    )
    ast.parse(code)
    assert (
        "_cubie_codegen_jac_scale = "
        "precision(0.435866)*_cubie_codegen_h" in code
    )
    assert loaded_name_count(code, "_cubie_codegen_jac_scale") == 2


@pytest.mark.parametrize(
    "variant", [HelperVariant.PLAIN, HelperVariant.CACHED],
    ids=["plain", "cached"],
)
def test_operator_bakes_stage_diagonal(
    cacheable_equations, bare_indexed_bases, variant
):
    """A baked ``a_ij`` folds into the operator as a literal."""
    runtime = generate_linear_operator_code(
        cacheable_equations, bare_indexed_bases, variant=variant
    )
    baked = generate_linear_operator_code(
        cacheable_equations,
        bare_indexed_bases,
        variant=variant,
        a_ij=0.435866,
    )
    ast.parse(baked)
    assert "0.435866" in baked
    # The signature keeps the argument; the body drops every read.
    assert loaded_name_count(baked, "_cubie_codegen_a_ij") == 0
    assert loaded_name_count(runtime, "_cubie_codegen_a_ij") > 0


@pytest.mark.parametrize(
    "variant", [HelperVariant.PLAIN, HelperVariant.CACHED],
    ids=["plain", "cached"],
)
def test_residual_bakes_stage_diagonal(
    cacheable_equations, bare_indexed_bases, variant
):
    """A baked ``a_ij`` folds into the residual as a literal."""
    runtime = generate_residual_code(
        cacheable_equations, bare_indexed_bases, variant=variant
    )
    baked = generate_residual_code(
        cacheable_equations,
        bare_indexed_bases,
        variant=variant,
        a_ij=0.435866,
    )
    ast.parse(baked)
    assert "0.435866" in baked
    assert loaded_name_count(baked, "_cubie_codegen_a_ij") == 0
    assert loaded_name_count(runtime, "_cubie_codegen_a_ij") > 0


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
        beta=2.0,
        gamma=3.0,
    )

    # Solver scalings fold in as literals; the user symbols keep
    # their own values in the equation body.
    assert "precision(2.0)" in code
    assert "precision(3.0)" in code
    # Constants fold to literals; no load or bare binding exists.
    assert "_cubie_codegen_const_" not in code
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
        beta=2.0,
    )

    assert "_cubie_codegen_m_" not in code
    # out[0] keeps beta*v; out[1] carries only the Jacobian term.
    lines = {
        line.strip().split(" = ")[0]: line
        for line in code.splitlines()
        if line.strip().startswith("out[")
    }
    assert "precision(2.0)*v[0]" in lines["out[0]"]
    assert "precision(2.0)" not in lines["out[1]"]


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
