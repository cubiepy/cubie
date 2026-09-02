"""Tests covering DIRK tableau registration and selection."""

import pytest
from numpy.testing import assert_array_equal

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
    DEFAULT_DIRK_TABLEAU_NAME,
    DIRK_TABLEAU_REGISTRY,
    DIRKTableau,
    KVAERNO3_TABLEAU,
    KVAERNO5_TABLEAU,
    L_STABLE_DIRK3_TABLEAU,
    L_STABLE_SDIRK4_TABLEAU,
)


def _consistent_dummy(nodes):
    """Build a DIRK tableau whose only meaningful content is ``c``.

    ``prediction_source_stages`` reads the node vector alone, but the
    constructor now enforces ``c[i] == sum(a[i])``. Placing each node
    on its own diagonal entry satisfies that relation while leaving the
    stage-node structure under test untouched; the weights merely sum
    to one so the tableau validates.
    """

    stage_count = len(nodes)
    a = tuple(
        tuple(
            nodes[row] if column == row else 0.0
            for column in range(stage_count)
        )
        for row in range(stage_count)
    )
    return DIRKTableau(
        a=a,
        b=(1.0 / stage_count,) * stage_count,
        c=nodes,
        order=1,
    )


@pytest.mark.parametrize(
    "expected_key",
    [
        "implicit_midpoint",
        "trapezoidal_dirk",
        "sdirk_2_2",
        "kvaerno3",
        "kvaerno5",
        "l_stable_dirk_3",
        "l_stable_sdirk_4",
    ],
)
def test_dirk_tableau_registry_contains_expected_entries(expected_key):
    """Registry must expose the documented DIRK tableaus."""

    assert expected_key in DIRK_TABLEAU_REGISTRY
    assert isinstance(DIRK_TABLEAU_REGISTRY[expected_key], DIRKTableau)


def test_dirk_tableau_default_matches_registry():
    """Default DIRK tableau should coincide with the registry entry."""

    assert (
        DIRK_TABLEAU_REGISTRY[DEFAULT_DIRK_TABLEAU_NAME]
        is DEFAULT_DIRK_TABLEAU
    )


def test_l_stable_sdirk4_fourth_stage_is_consistent():
    """Hairer4's fourth-stage row sum must equal its abscissa."""

    assert sum(L_STABLE_SDIRK4_TABLEAU.a[3]) == pytest.approx(
        L_STABLE_SDIRK4_TABLEAU.c[3]
    )


def test_explicit_first_stage_classifies_by_diagonal():
    """A zero first diagonal marks the first stage explicit."""

    assert KVAERNO3_TABLEAU.explicit_first_stage
    assert KVAERNO5_TABLEAU.explicit_first_stage
    assert not DIRK_TABLEAU_REGISTRY[
        "sdirk_2_2"
    ].explicit_first_stage
    assert not DIRK_TABLEAU_REGISTRY[
        "implicit_midpoint"
    ].explicit_first_stage


@pytest.mark.parametrize(
    "tableau,expected",
    [
        (L_STABLE_DIRK3_TABLEAU, (0, 1, 2)),
        (KVAERNO3_TABLEAU, (0, 1, 2, 2)),
        (_consistent_dummy((0.0, 0.5, 1.0, 0.5)), (0, 1, 2, 1)),
        (_consistent_dummy((0.0, 0.5, 0.0)), (0, 1, 0)),
        (_consistent_dummy((0.5, 0.5, 0.5)), (0, 0, 1)),
    ],
    ids=[
        "distinct-nodes",
        "adjacent-repeat",
        "non-adjacent-repeat",
        "explicit-first-source",
        "triple-repeat",
    ],
)
def test_prediction_source_stages_mappings(tableau, expected):
    """Each stage's starting guess reads the latest earlier stage at
    its time, or its own row when its time is new.

    Real tableaus supply the cases they exhibit: ``l_stable_dirk_3``
    has three distinct nodes, and ``kvaerno3`` repeats its final node
    for stiff accuracy. No registered DIRK tableau revisits an earlier
    interior node, so a consistent dummy carries the non-adjacent,
    explicit-first-source, and triple-repeat cases.
    """

    assert_array_equal(tableau.prediction_source_stages, expected)


def test_dirk_step_accepts_tableau_instance(precision):
    """DIRKStep should consume explicit tableau instances."""

    custom_name = "sdirk_2_2"
    custom_tableau = DIRK_TABLEAU_REGISTRY[custom_name]
    step = DIRKStep(precision=precision, n=2, tableau=custom_tableau)
    assert step.compile_settings.tableau is custom_tableau


def test_dirk_tableau_rejects_inconsistent_stage_nodes():
    """A ``c`` entry that disagrees with its ``A`` row sum must raise.

    The coefficients are the former ``lobatto_iiic_3`` tableau, whose
    first two stage nodes disagreed with their ``A`` row sums.
    """

    with pytest.raises(ValueError, match="row sum"):
        DIRKTableau(
            a=(
                (1.0 / 6.0, 0.0, 0.0),
                (2.0 / 3.0, 1.0 / 6.0, 0.0),
                (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            ),
            b=(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            c=(0.0, 0.5, 1.0),
            order=4,
        )


@pytest.mark.parametrize("name", sorted(DIRK_TABLEAU_REGISTRY))
def test_registry_stage_kind_flags(name):
    """Registry tableaus are explicit only in their first stage."""

    tableau = DIRK_TABLEAU_REGISTRY[name]
    assert tableau.explicit_last_stage is False
    assert tableau.last_implicit_stage == tableau.stage_count - 1


def test_explicit_last_stage_tableau_flags():
    """A zero last diagonal peels the last stage off the Newton loop."""

    tableau = DIRKTableau(
        a=((1.0, 0.0), (1.0, 0.0)),
        b=(1.0, 0.0),
        c=(1.0, 1.0),
        order=1,
        b_hat=(0.5, 0.5),
        embedded_order=2,
    )
    assert tableau.explicit_first_stage is False
    assert tableau.explicit_last_stage is True
    assert tableau.last_implicit_stage == 0
    assert tableau.last_stage_coupling == 1.0
    assert tableau.supports_smoothed_error is False


def test_interior_explicit_stage_rejected():
    """A zero diagonal on an interior stage must raise."""

    with pytest.raises(ValueError, match="first and last stages"):
        DIRKTableau(
            a=((0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.25, 0.25, 0.5)),
            b=(0.25, 0.25, 0.5),
            c=(0.5, 0.5, 1.0),
            order=1,
        )


def test_fsal_requires_explicit_first_stage():
    """An implicit first stage disqualifies stage-0 RHS reuse even
    when ``c[0] == 0``, ``c[-1] == 1``, and the last row equals ``b``."""

    implicit_first = ButcherTableau(
        a=(
            (1.0 / 6.0, 0.0, 0.0),
            (2.0 / 3.0, 1.0 / 6.0, 0.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        ),
        b=(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        c=(0.0, 0.5, 1.0),
        order=4,
    )
    assert not implicit_first.first_same_as_last
    assert implicit_first.can_reuse_accepted_start


def test_fsal_true_for_stiffly_accurate_esdirk():
    """ESDIRK tableaus with an explicit first stage keep FSAL reuse."""

    assert KVAERNO3_TABLEAU.first_same_as_last
    assert DIRK_TABLEAU_REGISTRY["trapezoidal_dirk"].first_same_as_last
    assert not DEFAULT_DIRK_TABLEAU.first_same_as_last


@pytest.mark.parametrize(
    "tableau,stage_count,b_row,b_hat_row,expected_d",
    [
        (
            KVAERNO3_TABLEAU,
            4,
            3,
            2,
            (
                -0.18175341844607201,
                1.4169932981732141,
                -1.671106401227145,
                0.4358665215,
            ),
        ),
        (
            KVAERNO5_TABLEAU,
            7,
            6,
            5,
            (
                -0.0019588905362793174,
                0.0,
                -0.012515715947863326,
                -0.06565284626324187,
                0.010502658265357234,
                -0.1903752055179727,
                0.26,
            ),
        ),
    ],
)
def test_kvaerno_tableau_invariants(
    tableau,
    stage_count,
    b_row,
    b_hat_row,
    expected_d,
):
    """Kvaerno pairs expose stiff-accuracy and embedded-row invariants."""

    assert tableau.stage_count == stage_count
    assert tableau.b_matches_a_row == b_row
    assert tableau.b_hat_matches_a_row == b_hat_row
    assert tableau.first_same_as_last
    assert tableau.can_reuse_accepted_start
    assert tableau.has_error_estimate
    assert tableau.d == expected_d


@pytest.mark.parametrize(
    "tableau, expected",
    [
        (KVAERNO3_TABLEAU, 0.4358665215),
        (KVAERNO5_TABLEAU, 0.26),
        (L_STABLE_SDIRK4_TABLEAU, 0.25),
        (L_STABLE_DIRK3_TABLEAU, 0.43586652150845895),
    ],
)
def test_equal_diagonals_returns_common_value(tableau, expected):
    """SDIRK/ESDIRK tableaus expose their shared implicit diagonal."""

    assert tableau.equal_diagonals == expected
