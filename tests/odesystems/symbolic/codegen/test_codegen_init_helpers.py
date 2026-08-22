"""Test generated consistent-initialisation helper source."""

import ast

import pytest

from cubie.odesystems.symbolic.codegen.lu_solver import (
    generate_init_lu_solve_code,
    generate_lu_solve_code,
)
from cubie.odesystems.symbolic.codegen.nonlinear_residuals import (
    generate_init_residual_code,
)
from tests._utils import TORN_INIT_COMMON
from tests.odesystems.symbolic.codegen._source_checks import (
    loaded_name_count,
)

# torn_time: x0 differential, x1 torn algebraic.
pytestmark = pytest.mark.parametrize(
    "solver_settings_override", [TORN_INIT_COMMON], indirect=True
)


def _codegen_inputs(system):
    """Return the generator inputs the helper registry passes."""
    return (
        system.equations,
        system.indices,
        system.compile_settings.mass,
    )


def _output_lines(code):
    """Map each ``out[i]`` assignment to its full source line."""
    return {
        line.strip().split(" = ")[0]: line.strip()
        for line in code.splitlines()
        if line.strip().startswith("out[")
    }


def test_init_residual_pins_differential_rows(system):
    """Rows are ``u[i]`` (differential) and ``-f_i`` (algebraic)."""
    equations, index_map, mass = _codegen_inputs(system)
    code = generate_init_residual_code(equations, index_map, M=mass)
    ast.parse(code)
    lines = _output_lines(code)
    assert lines["out[0]"] == "out[0] = u[0]"
    assert lines["out[1]"].startswith("out[1] = -")


def test_init_residual_evaluates_at_base_plus_increment(system):
    """States evaluate at ``base_state + u`` with h and a_ij unused."""
    equations, index_map, mass = _codegen_inputs(system)
    code = generate_init_residual_code(equations, index_map, M=mass)
    assert "base_state[" in code
    assert loaded_name_count(code, "_cubie_codegen_h") == 0
    assert loaded_name_count(code, "_cubie_codegen_a_ij") == 0


def test_init_lu_solve_parses_and_reports_factor_length(system):
    """The LU factory parses and stamps a positive factor length."""
    equations, index_map, mass = _codegen_inputs(system)
    code, lu_nnz = generate_init_lu_solve_code(
        equations, index_map, M=mass
    )
    ast.parse(code)
    assert lu_nnz > 0
    assert f".lu_nnz = {lu_nnz}" in code
    assert "def lu_solve(" in code


def test_init_lu_solve_drops_differential_jacobian_entries(system):
    """The differential row factorises as a bare identity row."""
    equations, index_map, mass = _codegen_inputs(system)
    _, init_nnz = generate_init_lu_solve_code(
        equations, index_map, M=mass
    )
    _, full_nnz = generate_lu_solve_code(
        equations, index_map, M=mass
    )
    assert init_nnz == 3
    assert full_nnz == 4
