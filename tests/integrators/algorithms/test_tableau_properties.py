"""Unit tests for ButcherTableau row-matching properties."""

from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    ROS3P_TABLEAU,
)
from cubie.integrators.algorithms.generic_firk_tableaus import (
    RADAU_IIA_5_TABLEAU,
)


def test_b_matches_a_row_radauiia5():
    """Test b_matches_a_row returns correct index for RadauIIA5."""
    tableau = RADAU_IIA_5_TABLEAU
    result = tableau.b_matches_a_row
    assert result == 2, (
        f"Expected b_matches_a_row=2 for RadauIIA5, got {result}"
    )


def test_b_matches_a_row_ros3p_none():
    """Test b_matches_a_row returns None for tableaus without match."""
    tableau = ROS3P_TABLEAU
    result = tableau.b_matches_a_row
    assert result is None, (
        f"Expected b_matches_a_row=None for ROS3P, got {result}"
    )


def test_b_hat_matches_a_row_none_when_no_b_hat():
    """Test b_hat_matches_a_row returns None when b_hat is None."""
    from cubie.integrators.algorithms.base_algorithm_step import (
        ButcherTableau,
    )

    test_tableau = ButcherTableau(
        a=((0.0, 0.0), (0.5, 0.5)),
        b=(0.0, 1.0),
        c=(0.0, 1.0),
        order=1,
        b_hat=None,
    )
    result = test_tableau.b_hat_matches_a_row
    assert result is None, (
        f"Expected b_hat_matches_a_row=None when b_hat is None, "
        f"got {result}"
    )


def test_floating_point_tolerance():
    """Test that row matching uses proper floating-point tolerance."""
    from cubie.integrators.algorithms.base_algorithm_step import (
        ButcherTableau,
    )

    # Create tableau where b nearly matches a row
    a_row_value = 0.333333333333333
    b_value = 1.0 / 3.0  # Should match within 1e-15

    test_tableau = ButcherTableau(
        a=((0.0, 0.0), (a_row_value, 1.0 - a_row_value)),
        b=(b_value, 1.0 - b_value),
        c=(0.0, 1.0),
        order=1,
    )
    result = test_tableau.b_matches_a_row
    # Should match due to tolerance
    assert result is not None, (
        "Expected match within tolerance for floating-point values"
    )
