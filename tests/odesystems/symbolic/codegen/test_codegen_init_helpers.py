"""Test generated consistent-initialisation helper source."""

import ast

import pytest

from cubie.odesystems.symbolic.codegen.linear_operators import (
    generate_init_operator_code,
)
from cubie.odesystems.symbolic.codegen.lu_solver import (
    generate_init_lu_solve_code,
    generate_lu_solve_code,
)
from cubie.odesystems.symbolic.codegen.nonlinear_residuals import (
    generate_init_residual_code,
)
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie.odesystems.symbolic.parsing import ParsedEquations
from tests.odesystems.symbolic.codegen._source_checks import (
    loaded_name_count,
)


@pytest.fixture(scope="module")
def torn_equation_setup():
    """Two-state torn system: x differential, z algebraic."""
    index_map = IndexedBases.from_user_inputs(
        states=["x", "z"],
        parameters=["a"],
        constants=[],
        observables=[],
        drivers=[],
    )
    equations = ParsedEquations.from_equations(
        [
            (ir.sym("dx"), ir.mul(ir.num(-1.0), ir.sym("z"))),
            (
                ir.sym("dz"),
                ir.add(
                    ir.mul(ir.sym("a"), ir.sym("z"), ir.sym("z")),
                    ir.sym("z"),
                    ir.mul(ir.num(-1.0), ir.sym("x")),
                ),
            ),
        ],
        index_map,
    )
    mass = [[1.0, 0.0], [0.0, 0.0]]
    return equations, index_map, mass


def _output_lines(code):
    """Map each ``out[i]`` assignment to its full source line."""
    return {
        line.strip().split(" = ")[0]: line.strip()
        for line in code.splitlines()
        if line.strip().startswith("out[")
    }


def test_init_residual_pins_differential_rows(torn_equation_setup):
    """Differential rows return the raw increment; algebraic rows
    carry the negated constraint."""
    equations, index_map, mass = torn_equation_setup
    code = generate_init_residual_code(equations, index_map, M=mass)
    ast.parse(code)
    lines = _output_lines(code)
    assert lines["out[0]"] == "out[0] = u[0]"
    assert lines["out[1]"].startswith("out[1] = -")


def test_init_residual_evaluates_at_base_plus_increment(
    torn_equation_setup,
):
    """States evaluate at ``base_state + u`` with h and a_ij unused."""
    equations, index_map, mass = torn_equation_setup
    code = generate_init_residual_code(equations, index_map, M=mass)
    assert "base_state[" in code
    assert loaded_name_count(code, "_cubie_codegen_h") == 0
    assert loaded_name_count(code, "_cubie_codegen_a_ij") == 0


def test_init_operator_identity_and_negated_jacobian(
    torn_equation_setup,
):
    """Differential rows pass v through; algebraic rows negate J@v."""
    equations, index_map, mass = torn_equation_setup
    code = generate_init_operator_code(equations, index_map, M=mass)
    ast.parse(code)
    lines = _output_lines(code)
    assert lines["out[0]"] == "out[0] = v[0]"
    assert lines["out[1]"].startswith("out[1] = -")
    assert loaded_name_count(code, "_cubie_codegen_h") == 0
    assert loaded_name_count(code, "_cubie_codegen_a_ij") == 0


def test_init_lu_solve_parses_and_reports_factor_length(
    torn_equation_setup,
):
    """The LU factory parses and stamps a positive factor length."""
    equations, index_map, mass = torn_equation_setup
    code, lu_nnz = generate_init_lu_solve_code(
        equations, index_map, M=mass
    )
    ast.parse(code)
    assert lu_nnz > 0
    assert f".lu_nnz = {lu_nnz}" in code
    assert "def lu_solve(" in code


def test_init_lu_solve_drops_differential_jacobian_entries(
    torn_equation_setup,
):
    """The differential row factorises as a bare identity row.

    ``dx = -z`` contributes ``J[0][1] = -1``; the initialisation
    matrix replaces that row with the identity, so its pattern is
    strictly sparser than the standard stage matrix, which keeps
    every Jacobian entry.
    """
    equations, index_map, mass = torn_equation_setup
    _, init_nnz = generate_init_lu_solve_code(
        equations, index_map, M=mass
    )
    _, full_nnz = generate_lu_solve_code(
        equations, index_map, M=mass
    )
    assert init_nnz == 3
    assert full_nnz == 4
