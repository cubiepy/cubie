"""Source-level tests for preconditioner code generators.

These exercise the generator *branches* that the default real-GPU test
configuration never reaches: the diagonal Jacobi FIRK preconditioner,
the cached (Rosenbrock) Neumann/Jacobi paths, lower-triangular stage
coupling, non-CSE emission, and 0/1 mass-diagonal handling. Every
generator
returns a Python source string, so the assertions check the emitted
source (structure, parseability, cache references) rather than
compiling a device kernel. Equation-set fixtures live in the local
conftest.
"""

import ast
import re

import pytest

from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations
from cubie.odesystems.solver_helpers import HelperVariant
from cubie.odesystems.symbolic.codegen.preconditioners import (
    generate_jacobi_preconditioner_code,
    generate_neumann_preconditioner_code,
)
from tests.odesystems.symbolic.codegen._source_checks import (
    factory_name_bindings,
    loaded_name_count,
)


# ── cached Neumann preconditioner ───────────────────────────────── #

def test_neumann_cached_reads_cache_buffer(
    cacheable_equations, bare_indexed_bases
):
    """Cached Neumann body indexes the ``cached_aux`` buffer."""
    code = generate_neumann_preconditioner_code(
        cacheable_equations,
        bare_indexed_bases,
        variant=HelperVariant.CACHED,
    )
    ast.parse(code)
    assert "cached_aux[" in code
    assert "def preconditioner(" in code


def test_neumann_empty_jvp_emits_pass_body(
    bare_nonlinear_equations, bare_indexed_bases
):
    """An empty JVP assignment set yields a ``pass`` Horner body."""
    code = generate_neumann_preconditioner_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        jvp_equations=JVPEquations([]),
    )
    ast.parse(code)
    # The order loop body collapses to a bare ``pass`` statement.
    assert "\n            pass\n" in code


@pytest.mark.parametrize(
    "generate",
    [
        generate_neumann_preconditioner_code,
        generate_jacobi_preconditioner_code,
    ],
    ids=["neumann", "jacobi"],
)
@pytest.mark.parametrize(
    "variant", [HelperVariant.PLAIN, HelperVariant.CACHED],
    ids=["plain", "cached"],
)
def test_preconditioner_bakes_stage_diagonal(
    cacheable_equations, bare_indexed_bases, generate, variant
):
    """A baked ``a_ij`` folds into the preconditioner as a literal."""
    runtime = generate(
        cacheable_equations, bare_indexed_bases, variant=variant
    )
    baked = generate(
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


# ── n-stage Neumann preconditioner ──────────────────────────────── #

def test_n_stage_preconditioners_isolate_user_constants(
    solver_scaling_collision_equations,
    solver_scaling_collision_indexed_bases,
):
    """FIRK preconditioners preserve solver beta/gamma values."""
    # Neumann folds 1/beta and gamma/beta; Jacobi folds beta, gamma.
    cases = (
        (
            generate_neumann_preconditioner_code,
            ("precision(0.5)", "1.5"),
        ),
        (
            generate_jacobi_preconditioner_code,
            ("precision(2.0)", "precision(3.0)"),
        ),
    )
    for generate, expected_literals in cases:
        code = generate(
            solver_scaling_collision_equations,
            solver_scaling_collision_indexed_bases,
            variant=HelperVariant.STACKED_STAGES,
            stage_coefficients=[[1.0]],
            stage_nodes=[1.0],
            beta=2.0,
            gamma=3.0,
        )
        for literal in expected_literals:
            assert literal in code
        # Constants fold to literals; no load or bare binding.
        assert "_cubie_codegen_const_" not in code
        assert "\n    beta = " not in code
        assert "\n    gamma = " not in code


def test_n_stage_neumann_skips_zero_stage_coupling(
    bare_nonlinear_equations,
    bare_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """Lower-triangular tableau parses with skipped zero coefficients."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_neumann_preconditioner_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
    )
    ast.parse(code)
    assert "_cubie_codegen_n = int32(4)" in code
    assert "def preconditioner(" in code


def test_n_stage_neumann_without_cse(
    bare_nonlinear_equations,
    bare_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """Topological-sort emission (``cse=False``) still parses."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_neumann_preconditioner_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        cse=False,
    )
    ast.parse(code)


# ── n-stage diagonal Jacobi preconditioner ──────────────────────── #

def test_n_stage_jacobi_source_structure(
    observable_driver_equations,
    observable_driver_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """FIRK Jacobi preconditioner emits guarded per-stage diagonals."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_jacobi_preconditioner_code(
        observable_driver_equations,
        observable_driver_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
    )
    ast.parse(code)
    assert "safe_diag_" in code
    # Per-stage guard locals show each stage carries its own diagonal.
    assert "_cubie_codegen_safe_diag_1_" in code


def test_n_stage_jacobi_without_drivers_or_observables(
    chained_aux_equations,
    bare_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """FIRK Jacobi handles a plain multi-auxiliary system.

    Two chained auxiliaries and no drivers/observables exercise the
    no-driver and no-observable stage-substitution branches.
    """
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_jacobi_preconditioner_code(
        chained_aux_equations,
        bare_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
    )
    ast.parse(code)
    assert "safe_diag_" in code


def test_n_stage_jacobi_without_cse(
    observable_driver_equations,
    observable_driver_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """FIRK Jacobi preconditioner parses under topological sort."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_jacobi_preconditioner_code(
        observable_driver_equations,
        observable_driver_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        cse=False,
    )
    ast.parse(code)
    assert "safe_diag_" in code


def test_n_stage_jacobi_zero_mass_row_drops_beta(
    observable_driver_equations,
    observable_driver_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """A zero mass row leaves the pure Jacobian-diagonal term."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    identity = generate_jacobi_preconditioner_code(
        observable_driver_equations,
        observable_driver_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        beta=2.0,
    )
    torn = generate_jacobi_preconditioner_code(
        observable_driver_equations,
        observable_driver_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        M=[[1, 0], [0, 0]],
        beta=2.0,
    )
    ast.parse(torn)
    # The zero row drops the folded beta literal from its diagonal.
    assert identity != torn
    assert identity.count("precision(2.0)") > torn.count(
        "precision(2.0)"
    )


# ── single-system and cached Jacobi preconditioners ─────────────── #

def test_jacobi_single_without_cse(
    bare_nonlinear_equations, bare_indexed_bases
):
    """Single-system Jacobi body parses under topological sort."""
    code = generate_jacobi_preconditioner_code(
        bare_nonlinear_equations, bare_indexed_bases, cse=False
    )
    ast.parse(code)
    assert "safe_diag_" in code


def test_jacobi_single_zero_mass_row_drops_beta(
    bare_nonlinear_equations, bare_indexed_bases
):
    """A zero mass row leaves the pure Jacobian-diagonal term."""
    identity = generate_jacobi_preconditioner_code(
        bare_nonlinear_equations, bare_indexed_bases, beta=2.0
    )
    torn = generate_jacobi_preconditioner_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        M=[[1, 0], [0, 0]],
        beta=2.0,
    )
    ast.parse(torn)
    assert identity != torn
    assert identity.count("precision(2.0)") > torn.count(
        "precision(2.0)"
    )


def test_jacobi_rejects_general_mass_matrix(
    bare_nonlinear_equations, bare_indexed_bases
):
    """Anything but a 0/1 diagonal is rejected at generation."""
    with pytest.raises(ValueError, match="0/1 diagonal"):
        generate_jacobi_preconditioner_code(
            bare_nonlinear_equations,
            bare_indexed_bases,
            M=[[2, 0], [0, 1]],
        )
    with pytest.raises(ValueError, match="0/1 diagonal"):
        generate_jacobi_preconditioner_code(
            bare_nonlinear_equations,
            bare_indexed_bases,
            M=[[1.0, 0.5], [0.0, 1.0]],
        )


def test_jacobi_cached_partitions_auxiliaries(
    cacheable_equations, bare_indexed_bases
):
    """Cached Jacobi generator runs the cached-partition branch."""
    code = generate_jacobi_preconditioner_code(
        cacheable_equations,
        bare_indexed_bases,
        variant=HelperVariant.CACHED,
    )
    ast.parse(code)
    assert "def preconditioner(" in code


def test_jacobi_cached_without_cse(
    cacheable_equations, bare_indexed_bases
):
    """Cached Jacobi generator parses under topological sort."""
    code = generate_jacobi_preconditioner_code(
        cacheable_equations,
        bare_indexed_bases,
        variant=HelperVariant.CACHED,
        cse=False,
    )
    ast.parse(code)
    assert "safe_diag_" in code


_CACHED_SOURCE_SYSTEMS = [
    ("cacheable_equations", "bare_indexed_bases"),
    (
        "cacheable_observable_equations",
        "observable_driver_indexed_bases",
    ),
]


@pytest.mark.parametrize(
    "equations_fixture,bases_fixture",
    _CACHED_SOURCE_SYSTEMS,
    ids=["plain-aux", "observable-and-aux"],
)
def test_cached_jacobi_defines_every_referenced_auxiliary(
    request, equations_fixture, bases_fixture
):
    """Every auxiliary a cached Jacobi body references is defined."""
    code = generate_jacobi_preconditioner_code(
        request.getfixturevalue(equations_fixture),
        request.getfixturevalue(bases_fixture),
        variant=HelperVariant.CACHED,
    )
    referenced, defined = factory_name_bindings(code)
    assert referenced <= defined
    assert re.search(r"= cached_aux\[\d+\]", code)


@pytest.mark.parametrize(
    "equations_fixture,bases_fixture",
    _CACHED_SOURCE_SYSTEMS,
    ids=["plain-aux", "observable-and-aux"],
)
def test_cached_neumann_defines_every_referenced_auxiliary(
    request, equations_fixture, bases_fixture
):
    """Every auxiliary a cached Neumann body references is defined."""
    code = generate_neumann_preconditioner_code(
        request.getfixturevalue(equations_fixture),
        request.getfixturevalue(bases_fixture),
        variant=HelperVariant.CACHED,
    )
    referenced, defined = factory_name_bindings(code)
    assert referenced <= defined
    assert re.search(r"= cached_aux\[\d+\]", code)


# ── Jacobi series (order > 0) ───────────────────────────────────── #

def test_jacobi_emits_both_order_branches(
    bare_nonlinear_equations, bare_indexed_bases
):
    """One source carries both bodies; the bound order picks one."""
    code = generate_jacobi_preconditioner_code(
        bare_nonlinear_equations, bare_indexed_bases
    )
    ast.parse(code)
    assert code.count("def preconditioner(") == 2
    assert "    if order > 0:" in code
    assert "for _ in range(_cubie_codegen_order):" in code


def test_jacobi_series_divides_by_the_guarded_diagonal(
    bare_nonlinear_equations, bare_indexed_bases
):
    """Series terms divide by the guarded diagonal, not a fresh one."""
    code = generate_jacobi_preconditioner_code(
        bare_nonlinear_equations, bare_indexed_bases, gamma=3.0
    )
    update = (
        "out[0] = out[0] + (v[0] + _cubie_codegen_h_eff * jvp[0]"
        " - out[0])"
        " / _cubie_codegen_safe_diag_0"
    )
    assert update in code
    assert (
        "_cubie_codegen_h_eff = precision(3.0) "
        "* _cubie_codegen_h * _cubie_codegen_a_ij" in code
    )


def test_jacobi_series_drops_beta_on_algebraic_rows(
    bare_nonlinear_equations, bare_indexed_bases
):
    """A zero mass row removes nothing but ``beta`` from the update."""
    code = generate_jacobi_preconditioner_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        M=[[1, 0], [0, 0]],
    )
    ast.parse(code)
    assert (
        "out[1] = out[1] + (v[1] + _cubie_codegen_h_eff * jvp[1])"
        " / _cubie_codegen_safe_diag_1" in code
    )


def test_n_stage_jacobi_series_scales_without_a_ij(
    observable_driver_equations,
    observable_driver_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """FIRK series terms carry ``a_ij`` inside the stage JVP."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_jacobi_preconditioner_code(
        observable_driver_equations,
        observable_driver_indexed_bases,
        variant=HelperVariant.STACKED_STAGES,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        gamma=3.0,
    )
    ast.parse(code)
    assert code.count("def preconditioner(") == 2
    assert (
        "_cubie_codegen_h_eff = precision(3.0) "
        "* _cubie_codegen_h\n" in code
    )
    assert (
        "out[3] = out[3] + (v[3] + _cubie_codegen_h_eff * jvp[3]"
        " - out[3])"
        " / _cubie_codegen_safe_diag_1_1" in code
    )


def _assigned_and_used_diagonals(code):
    """Return the guarded diagonals assigned and referenced in ``code``."""
    name = r"_cubie_codegen_safe_diag_\d+(?:_\d+)?"
    assigned = set(re.findall(rf"^\s*({name}) = ", code, re.MULTILINE))
    used = set(re.findall(name, code))
    return assigned, used


@pytest.mark.parametrize(
    "mass",
    [None, [[1, 0], [0, 0]]],
    ids=["identity-mass", "torn-mass"],
)
@pytest.mark.parametrize(
    "variant,stage_kwargs",
    [
        (HelperVariant.PLAIN, {}),
        (HelperVariant.CACHED, {}),
        (
            HelperVariant.STACKED_STAGES,
            {
                "stage_coefficients": [[0.25, 0.0], [0.5, 0.25]],
                "stage_nodes": [0.25, 0.75],
            },
        ),
    ],
    ids=["single", "cached", "stacked"],
)
def test_jacobi_series_defines_every_diagonal_it_divides_by(
    variant, stage_kwargs, mass, zero_diagonal_equations, bare_indexed_bases
):
    """Every diagonal the series loop divides by is assigned."""
    code = generate_jacobi_preconditioner_code(
        zero_diagonal_equations,
        bare_indexed_bases,
        variant=variant,
        M=mass,
        **stage_kwargs,
    )
    ast.parse(code)
    assigned, used = _assigned_and_used_diagonals(code)
    assert used
    assert used == assigned
