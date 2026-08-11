"""Unit tests for ButcherTableau row-matching properties."""

import numpy as np
import pytest

from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    ROS3P_TABLEAU,
)
from cubie.integrators.algorithms.generic_firk_tableaus import (
    RADAU_IIA_5_TABLEAU,
    compute_embedded_weights,
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


def test_compute_embedded_weights_radauiia_defaults_order_to_stage_count():
    """order=None defaults to the exact (square) collocation system."""
    c = np.asarray(RADAU_IIA_5_TABLEAU.c)
    weights = compute_embedded_weights(c, order=None)
    assert weights.shape == (len(c),)


def test_compute_embedded_weights_radauiia_rejects_order_above_stages():
    """order exceeding the number of stages raises ValueError."""
    c = np.asarray(RADAU_IIA_5_TABLEAU.c)
    with pytest.raises(ValueError, match="Cannot achieve order"):
        compute_embedded_weights(c, order=len(c) + 1)


def test_compute_embedded_weights_radauiia_bitwise_reproducible():
    """Embedded weights carry the exact same bits on every host.

    The weights are part of the tableau, which is hashed into
    kernel-cache config keys. A solver that dispatches SIMD kernels
    by CPU microarchitecture (LAPACK lstsq/solve) drifts in the last
    ulp between machines, keying the same kernel differently on the
    CI precompile runner and the GPU runner. Scalar arithmetic pins
    the exact values, asserted here bit-for-bit.
    """
    c = np.asarray(RADAU_IIA_5_TABLEAU.c)
    weights = compute_embedded_weights(c, order=2)
    expected = (
        float.fromhex("0x1.d3e58763aeaeep-2"),
        float.fromhex("0x1.488c3fb8c3184p-2"),
        float.fromhex("0x1.c71c71c71c71cp-3"),
    )
    assert tuple(weights.tolist()) == expected
    assert tuple(RADAU_IIA_5_TABLEAU.b_hat) == expected
