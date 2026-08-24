"""Source-level tests for the prefactored LU slot plan."""

import ast
import re

import pytest

from cubie.odesystems.solver_helpers import HelperVariant
from cubie.odesystems.symbolic.codegen.lu_solver import (
    generate_lu_prepare_blocks_code,
    generate_lu_smoothing_solve_code,
    generate_lu_solve_code,
)

RADAU_A = [
    [0.1968154772236604, -0.06553542585019839, 0.02377097434822015],
    [0.3944243147390873, 0.2920734116652285, -0.04154875212599793],
    [0.3764030627004673, 0.5124858261884216, 0.1111111111111111],
]
RADAU_C = [0.1550510257216822, 0.6449489742783178, 1.0]

DIRK_A = [[0.25, 0.0], [0.4, 0.5]]
DIRK_C = [0.25, 0.9]


def _written_slots(code):
    return {
        int(index)
        for index in re.findall(r"cached_aux\[(\d+)\] = ", code)
    }


def _read_slots(code):
    return {
        int(index)
        for index in re.findall(r"cached_aux\[(\d+)\]", code)
    }


@pytest.mark.parametrize(
    "mass,beta",
    [(None, 1.0), ([[1, 0], [0, 0]], 2.0)],
    ids=["identity-mass", "torn-mass"],
)
@pytest.mark.parametrize(
    "variant,tableau",
    [
        (HelperVariant.PREFACTORED, (DIRK_A, DIRK_C)),
        (HelperVariant.PREFACTORED_STACKED, (RADAU_A, RADAU_C)),
    ],
    ids=["per-diagonal", "block-transform"],
)
def test_prepare_fills_every_compacted_slot(
    bare_nonlinear_equations,
    bare_indexed_bases,
    variant,
    tableau,
    mass,
    beta,
):
    """Prepare writes each compacted slot; solves read within it."""
    coefficients, nodes = tableau
    prepare_code, aux_count = generate_lu_prepare_blocks_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=variant,
        M=mass,
        stage_coefficients=coefficients,
        stage_nodes=nodes,
        beta=beta,
    )
    solve_code, _ = generate_lu_solve_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=variant,
        M=mass,
        stage_coefficients=coefficients,
        stage_nodes=nodes,
        beta=beta,
    )
    ast.parse(prepare_code)
    ast.parse(solve_code)
    assert _written_slots(prepare_code) == set(range(aux_count))
    assert _read_slots(solve_code) <= set(range(aux_count))


def test_smoothing_reads_within_the_compacted_real_block(
    bare_nonlinear_equations, bare_indexed_bases
):
    """The smoothing solve reads only slots prepare fills."""
    prepare_code, aux_count = generate_lu_prepare_blocks_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=HelperVariant.PREFACTORED_STACKED,
        stage_coefficients=RADAU_A,
        stage_nodes=RADAU_C,
        beta=2.0,
    )
    smoothing_code, _ = generate_lu_smoothing_solve_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        stage_coefficients=RADAU_A,
        stage_nodes=RADAU_C,
        beta=2.0,
    )
    ast.parse(smoothing_code)
    assert _read_slots(smoothing_code) <= _written_slots(
        prepare_code
    )


def test_stacked_plan_prunes_zero_and_duplicate_slots(
    bare_nonlinear_equations, bare_indexed_bases
):
    """The block-transform aux shrinks below the dense layout."""
    _, aux_count = generate_lu_prepare_blocks_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=HelperVariant.PREFACTORED_STACKED,
        stage_coefficients=RADAU_A,
        stage_nodes=RADAU_C,
    )
    # The dense layout is 12 slots; pruning shrinks it to 10.
    assert aux_count == 10


def test_prepare_stores_reciprocal_diagonals(
    bare_nonlinear_equations, bare_indexed_bases
):
    """Diagonal factor slots hold reciprocals of the raw pivots."""
    prepare_code, _ = generate_lu_prepare_blocks_code(
        bare_nonlinear_equations,
        bare_indexed_bases,
        variant=HelperVariant.PREFACTORED,
        stage_coefficients=DIRK_A,
        stage_nodes=DIRK_C,
    )
    ast.parse(prepare_code)
    assert re.search(
        r"cached_aux\[\d+\] = \(precision\(1\)"
        r"/_cubie_codegen_lu_b\d+_",
        prepare_code,
    )
