"""Tests covering FIRK tableau construction and embedded weights."""

import numpy as np
import pytest

from cubie.integrators.algorithms.generic_firk_tableaus import (
    FIRKTableau,
    GAUSS_LEGENDRE_4_TABLEAU,
    RADAU_IIA_5_TABLEAU,
    RADAU_IIA_9_TABLEAU,
    compute_embedded_weights,
)
from cubie.odesystems.symbolic.codegen._matrix_utils import (
    block_eigenstructure,
)


def _collocation_coefficients(nodes):
    """Return the collocation ``(a, b)`` at ``nodes``."""
    poly = np.polynomial.polynomial
    stage_count = len(nodes)
    basis_integrals = []
    for basis_index in range(stage_count):
        other_nodes = [
            nodes[node_index]
            for node_index in range(stage_count)
            if node_index != basis_index
        ]
        coefficients = poly.polyfromroots(other_nodes)
        coefficients = coefficients / poly.polyval(
            nodes[basis_index], coefficients
        )
        basis_integrals.append(poly.polyint(coefficients))
    a_matrix = tuple(
        tuple(
            float(poly.polyval(node, integral))
            for integral in basis_integrals
        )
        for node in nodes
    )
    b_weights = tuple(
        float(poly.polyval(1.0, integral))
        for integral in basis_integrals
    )
    return a_matrix, b_weights


def _gauss_legendre_collocation_tableau(stage_count):
    """Build the Gauss--Legendre collocation tableau of order ``2s``."""
    legendre_roots, _ = np.polynomial.legendre.leggauss(stage_count)
    nodes = 0.5 * (legendre_roots + 1.0)
    a_matrix, b_weights = _collocation_coefficients(nodes)
    return FIRKTableau(
        a=a_matrix,
        b=b_weights,
        c=tuple(float(node) for node in nodes),
        order=2 * stage_count,
    )


def _radau_iia_collocation_tableau(stage_count):
    """Build the Radau IIA collocation tableau of order ``2s - 1``.

    The right-Radau nodes are the roots of the ``(s-1)``-th derivative
    of ``x**(s-1) * (x-1)**s``."""
    poly = np.polynomial.polynomial
    left = poly.polypow((0.0, 1.0), stage_count - 1)
    right = poly.polypow((-1.0, 1.0), stage_count)
    derivative = poly.polyder(
        poly.polymul(left, right), stage_count - 1
    )
    nodes = np.sort(poly.polyroots(derivative).real)
    a_matrix, b_weights = _collocation_coefficients(nodes)
    return FIRKTableau(
        a=a_matrix,
        b=b_weights,
        c=tuple(float(node) for node in nodes),
        order=2 * stage_count - 1,
    )


def test_radau_iia_9_registry_literals_match_construction():
    """The registry's literals reproduce the collocation build.

    The 1e-12 tolerance covers the construction's float64 roots of a
    degree-nine polynomial. ``b[-1]`` and ``c[-1]`` are exact."""
    constructed = _radau_iia_collocation_tableau(5)
    np.testing.assert_allclose(
        np.asarray(RADAU_IIA_9_TABLEAU.a),
        np.asarray(constructed.a),
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(RADAU_IIA_9_TABLEAU.b),
        np.asarray(constructed.b),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(RADAU_IIA_9_TABLEAU.c),
        np.asarray(constructed.c),
        rtol=1e-12,
    )
    # Radau IIA has b[s-1] = 1/s**2 exactly, and c[s-1] = 1.
    assert RADAU_IIA_9_TABLEAU.b[-1] == 1.0 / 25.0
    assert RADAU_IIA_9_TABLEAU.c[-1] == 1.0


def test_gauss_legendre_4_registry_literals_match_construction():
    """The registry's literals reproduce the collocation build."""
    constructed = _gauss_legendre_collocation_tableau(4)
    np.testing.assert_allclose(
        np.asarray(GAUSS_LEGENDRE_4_TABLEAU.a),
        np.asarray(constructed.a),
        rtol=1e-15,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        np.asarray(GAUSS_LEGENDRE_4_TABLEAU.b),
        np.asarray(constructed.b),
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(GAUSS_LEGENDRE_4_TABLEAU.c),
        np.asarray(constructed.c),
        rtol=1e-15,
    )


def test_compute_embedded_weights_defaults_order_to_stage_count():
    """order=None defaults to the exact (square) collocation system."""
    c = np.asarray(RADAU_IIA_5_TABLEAU.c)
    weights = compute_embedded_weights(c, order=None)
    assert weights.shape == (len(c),)


def test_compute_embedded_weights_rejects_order_above_stages():
    """order exceeding the number of stages raises ValueError."""
    c = np.asarray(RADAU_IIA_5_TABLEAU.c)
    with pytest.raises(ValueError, match="Cannot achieve order"):
        compute_embedded_weights(c, order=len(c) + 1)


def test_compute_embedded_weights_bitwise_reproducible():
    """Embedded weights carry the exact same bits on every host.

    The weights are hashed into kernel-cache config keys, so any
    last-ulp drift between machines keys the same kernel differently.
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


def test_block_transform_reassembles_inverse_a():
    """The block transform reassembles inv(a) for every radau tableau."""
    for tableau in (RADAU_IIA_5_TABLEAU, RADAU_IIA_9_TABLEAU):
        reals, pairs, transform, inverse_transform = (
            block_eigenstructure(tableau.stage_coefficients)
        )
        stage_count = tableau.stage_count
        assert len(reals) + 2 * len(pairs) == stage_count
        lam = np.zeros((stage_count, stage_count))
        for slot, value in enumerate(reals):
            lam[slot, slot] = value
        offset = len(reals)
        for slot, (alpha, beta) in enumerate(pairs):
            assert beta > 0.0
            row = offset + 2 * slot
            lam[row, row] = alpha
            lam[row, row + 1] = beta
            lam[row + 1, row] = -beta
            lam[row + 1, row + 1] = alpha
        inverse_a = np.linalg.inv(np.asarray(tableau.a))
        reassembled = (
            np.asarray(transform)
            @ lam
            @ np.asarray(inverse_transform)
        )
        assert np.allclose(reassembled, inverse_a, rtol=1e-9, atol=1e-9)
        # The sole real eigenvalue matches the smoothing derivation.
        assert len(reals) == 1
        assert np.isclose(1.0 / reals[0], tableau.smoothing_gamma)
