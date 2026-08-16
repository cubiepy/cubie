"""Source-level tests for preconditioner code generators.

These exercise the generator *branches* that the default real-GPU test
configuration never reaches: the diagonal Jacobi FIRK preconditioner,
the cached (Rosenbrock) Neumann/Jacobi paths, lower-triangular stage
coupling, non-CSE emission, and mass-matrix handling. Every generator
returns a Python source string, so the assertions check the emitted
source (structure, parseability, cache references) rather than
compiling a device kernel. Equation-set fixtures live in the local
conftest.
"""

import ast

from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations
from cubie.odesystems.symbolic.codegen.preconditioners import (
    generate_neumann_preconditioner_code,
    generate_neumann_preconditioner_cached_code,
    generate_n_stage_neumann_preconditioner_code,
    generate_jacobi_preconditioner_code,
    generate_jacobi_preconditioner_cached_code,
    generate_n_stage_jacobi_preconditioner_code,
)


# ── cached Neumann preconditioner ───────────────────────────────── #

def test_neumann_cached_reads_cache_buffer(
    cacheable_equations, bare_indexed_bases
):
    """Cached Neumann body indexes the ``cached_aux`` buffer."""
    code = generate_neumann_preconditioner_cached_code(
        cacheable_equations, bare_indexed_bases
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


# ── n-stage Neumann preconditioner ──────────────────────────────── #

def test_n_stage_preconditioners_isolate_user_constants(
    solver_scaling_collision_equations,
    solver_scaling_collision_indexed_bases,
):
    """FIRK preconditioners preserve solver beta/gamma values."""
    generators = (
        generate_n_stage_neumann_preconditioner_code,
        generate_n_stage_jacobi_preconditioner_code,
    )
    for generate in generators:
        code = generate(
            solver_scaling_collision_equations,
            solver_scaling_collision_indexed_bases,
            stage_coefficients=[[1.0]],
            stage_nodes=[1.0],
        )
        assert "_cubie_codegen_beta = precision(beta)" in code
        assert "_cubie_codegen_gamma = precision(gamma)" in code
        # Constants fold to literals; no load or bare binding.
        assert "_cubie_codegen_const_" not in code
        assert "constants['beta']" not in code
        assert "constants['gamma']" not in code
        assert "\n    beta = " not in code
        assert "\n    gamma = " not in code


def test_n_stage_neumann_skips_zero_stage_coupling(
    bare_nonlinear_equations,
    bare_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """Lower-triangular tableau parses with skipped zero coefficients."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_n_stage_neumann_preconditioner_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
    )
    ast.parse(code)
    assert "total_n = int32(4)" in code
    assert "def preconditioner(" in code


def test_n_stage_neumann_without_cse(
    bare_nonlinear_equations,
    bare_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """Topological-sort emission (``cse=False``) still parses."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_n_stage_neumann_preconditioner_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
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
    code = generate_n_stage_jacobi_preconditioner_code(
        observable_driver_equations,
        observable_driver_indexed_bases,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
    )
    ast.parse(code)
    assert "safe_diag_" in code
    assert "total_n = int32(4)" in code
    assert "stage_width = int32(2)" in code


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
    code = generate_n_stage_jacobi_preconditioner_code(
        chained_aux_equations,
        bare_indexed_bases,
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
    code = generate_n_stage_jacobi_preconditioner_code(
        observable_driver_equations,
        observable_driver_indexed_bases,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        cse=False,
    )
    ast.parse(code)
    assert "safe_diag_" in code


def test_n_stage_jacobi_integer_mass_matrix(
    observable_driver_equations,
    observable_driver_indexed_bases,
    lower_triangular_stage_coefficients,
):
    """Integer mass entries are cast to float in the diagonal term."""
    stage_coefficients, stage_nodes = lower_triangular_stage_coefficients
    code = generate_n_stage_jacobi_preconditioner_code(
        observable_driver_equations,
        observable_driver_indexed_bases,
        stage_coefficients=stage_coefficients,
        stage_nodes=stage_nodes,
        M=[[2, 0], [0, 1]],
    )
    ast.parse(code)
    # beta multiplies the 2.0 mass diagonal in the emitted source.
    assert "2.0" in code


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


def test_jacobi_single_integer_mass_matrix(
    bare_nonlinear_equations, bare_indexed_bases
):
    """Single-system Jacobi casts integer mass diagonal to float."""
    code = generate_jacobi_preconditioner_code(
        bare_nonlinear_equations, bare_indexed_bases, M=[[2, 0], [0, 1]]
    )
    ast.parse(code)
    assert "2.0" in code


def test_jacobi_cached_partitions_auxiliaries(
    cacheable_equations, bare_indexed_bases
):
    """Cached Jacobi generator runs the cached-partition branch."""
    code = generate_jacobi_preconditioner_cached_code(
        cacheable_equations, bare_indexed_bases
    )
    ast.parse(code)
    assert "def preconditioner(" in code


def test_jacobi_cached_without_cse(
    cacheable_equations, bare_indexed_bases
):
    """Cached Jacobi generator parses under topological sort."""
    code = generate_jacobi_preconditioner_cached_code(
        cacheable_equations, bare_indexed_bases, cse=False
    )
    ast.parse(code)
    assert "safe_diag_" in code
