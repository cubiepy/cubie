"""Fully implicit Runge–Kutta tableau definitions.

Published Classes
-----------------
:class:`FIRKTableau`
    Extends :class:`~base_algorithm_step.ButcherTableau` with FIRK
    weight validation.

:class:`RadauIIATableau`
    Radau IIA collocation tableau carrying the smoothed error
    estimate.

Module-Level Functions
----------------------
:func:`compute_embedded_weights_radauIIA`
    Compute embedded weights for Radau IIA collocation nodes via
    moment conditions.

Constants
---------
:data:`GAUSS_LEGENDRE_2_TABLEAU`
    Two-stage, fourth-order Gauss–Legendre tableau.

:data:`RADAU_IIA_5_TABLEAU`
    Three-stage, fifth-order Radau IIA tableau with second-order
    embedded error estimate.

:data:`FIRK_TABLEAU_REGISTRY`
    Name → tableau mapping for alias-based lookup.

:data:`DEFAULT_FIRK_TABLEAU`
    Default tableau (Gauss–Legendre 2-stage).

See Also
--------
:class:`~cubie.integrators.algorithms.generic_firk.FIRKStep`
    Step factory consuming these tableaus.
:class:`~cubie.integrators.algorithms.base_algorithm_step.ButcherTableau`
    Parent tableau class.
"""

from fractions import Fraction
from functools import lru_cache
from typing import Dict, Optional, Tuple

from attrs import frozen
from numpy import (
    abs as np_abs,
    array as np_array,
    asarray as np_asarray,
    sqrt as np_sqrt,
)
from numpy.linalg import eigvals as np_eigvals, inv as np_inv
from sympy import Matrix, Rational

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau


@frozen
class FIRKTableau(ButcherTableau):
    """Coefficient tableau describing a fully implicit RK scheme."""

    def __attrs_post_init__(self) -> None:
        """Validate structure and Runge--Kutta weight sums."""
        super().__attrs_post_init__()
        self._validate_weight_sums()

    def smoothed_error_weights(
        self,
        numba_precision: type,
    ) -> Optional[Tuple[float, ...]]:
        """Return smoothed error weights; ``None`` without a derivation."""
        return None


@lru_cache(maxsize=None)
def _reciprocal_real_eigenvalue(
    a: Tuple[Tuple[float, ...], ...],
) -> Optional[float]:
    """Return 1/lambda for inv(a)'s sole real eigenvalue, else None."""
    eigenvalues = np_eigvals(np_inv(np_asarray(a, dtype=float)))
    real = eigenvalues[np_abs(eigenvalues.imag) < 1e-12].real
    if real.size != 1:
        return None
    return float(1.0 / real[0])


@frozen
class RadauIIATableau(FIRKTableau):
    """Radau IIA collocation tableau with a smoothed error estimate."""

    @property
    def supports_smoothed_error(self) -> bool:
        """Return whether ``inv(a)`` has exactly one real eigenvalue."""
        return _reciprocal_real_eigenvalue(self.a) is not None

    @property
    def smoothing_gamma(self) -> float:
        """Return the reciprocal real eigenvalue of ``inv(a)``."""
        return _reciprocal_real_eigenvalue(self.a)

    def smoothed_error_weights(
        self,
        numba_precision: type,
    ) -> Tuple[float, ...]:
        """Return the ``d_i`` of ``sum d_i*k_i - gamma*h*f(y_n)``."""
        embedded = compute_embedded_weights_radauIIA(
            self.c, f0_weight=self.smoothing_gamma
        )
        return self.typed_vector(
            tuple(
                weight - embedded_weight
                for weight, embedded_weight in zip(self.b, embedded)
            ),
            numba_precision,
        )


SQRT3 = np_sqrt(3)

GAUSS_LEGENDRE_2_TABLEAU = FIRKTableau(
    a=(
        (0.25, 0.25 - SQRT3 / 6.0),
        (0.25 + SQRT3 / 6.0, 0.25),
    ),
    b=(0.5, 0.5),
    c=(0.5 - SQRT3 / 6.0, 0.5 + SQRT3 / 6.0),
    order=4,
    dense_prediction_ratio_float32=8.0,
    dense_prediction_ratio_float64=8.0,
)


def compute_embedded_weights_radauIIA(c, order=None, f0_weight=0.0):
    """Compute embedded weights for Radau IIA collocation nodes.

    Solves :math:`\\sum_i b^*_i \\, c_i^{k-1} = 1/k` for
    :math:`k = 1, \\ldots, \\text{order}`, so the weights integrate
    polynomials up to degree ``order - 1`` exactly. When ``order < s`` the
    system is underdetermined and the minimum-norm solution is
    returned. The solve runs in exact rational arithmetic (each
    float node is a dyadic rational) with one correctly-rounded
    conversion back to float, so the weights are identical on every
    platform.

    Parameters
    ----------
    c : array_like, shape (s,)
        Collocation nodes.
    order : int, optional
        Order of the embedded method (must be ``<= s``). When
        ``None``, defaults to ``s``.
    f0_weight : float, optional
        Weight given to an extra ``f(y_n)`` sample at node zero. The
        stage weights must then sum to ``1 - f0_weight``, so only the
        ``k = 1`` condition changes.

    Returns
    -------
    ndarray, shape (s,)
        Embedded weights satisfying the moment conditions.

    Raises
    ------
    ValueError
        If ``order`` exceeds the number of stages.
    """
    nodes = [Rational(Fraction(float(value))) for value in np_asarray(c)]
    s = len(nodes)

    if order is None:
        order = s
    if order > s:
        raise ValueError(f"Cannot achieve order {order} with {s} stages")

    f0_term = Rational(Fraction(float(f0_weight)))

    # Quadrature condition k: sum_i b_i * c_i**(k-1) = 1/k.
    moments = Matrix(order, s, lambda k, i: nodes[i] ** k)
    rhs = Matrix(
        [
            Rational(1, k) - (f0_term if k == 1 else 0)
            for k in range(1, order + 1)
        ]
    )

    if order == s:
        b_star = moments.solve(rhs)
    else:
        # Minimum-norm solution of the underdetermined system,
        # b* = M^T (M M^T)^{-1} r, matching ``lstsq``.
        b_star = moments.T * (moments * moments.T).inv() * rhs

    return np_array([float(value) for value in b_star])


SQRT6 = np_sqrt(6)
_RADAU_IIA_5_c = ((4 - SQRT6) / 10.0, (4 + SQRT6) / 10.0, 1.0)
_RADAU_IIA_5_b_hat = tuple(
    compute_embedded_weights_radauIIA(_RADAU_IIA_5_c, order=2).tolist()
)

RADAU_IIA_5_TABLEAU = RadauIIATableau(
    a=(
        (
            (88 - 7 * SQRT6) / 360.0,
            (296 - 169 * SQRT6) / 1800.0,
            (-2 + 3 * SQRT6) / 225.0,
        ),
        (
            (296 + 169 * SQRT6) / 1800.0,
            (88 + 7 * SQRT6) / 360.0,
            (-2 - 3 * SQRT6) / 225.0,
        ),
        ((16 - SQRT6) / 36.0, (16 + SQRT6) / 36.0, 1.0 / 9.0),
    ),
    b=((16 - SQRT6) / 36.0, (16 + SQRT6) / 36.0, 1.0 / 9.0),
    b_hat=_RADAU_IIA_5_b_hat,
    c=_RADAU_IIA_5_c,
    order=5,
    dense_prediction_ratio_float32=8.0,
    dense_prediction_ratio_float64=8.0,
)

GAUSS_LEGENDRE_4_TABLEAU = FIRKTableau(
    a=(
        (
            0.08696371128436345,
            -0.026604180084998798,
            0.012627462689404742,
            -0.003555149685795684,
        ),
        (
            0.188118117499868,
            0.16303628871563644,
            -0.027880428602470995,
            0.006735500594538164,
        ),
        (
            0.1671919219741886,
            0.3539530060337439,
            0.16303628871563638,
            -0.014190694931141147,
        ),
        (
            0.17748257225452202,
            0.3134451147418685,
            0.3526767575162713,
            0.08696371128436325,
        ),
    ),
    b=(
        0.17392742256872662,
        0.32607257743127305,
        0.3260725774312722,
        0.1739274225687269,
    ),
    c=(
        0.06943184420297371,
        0.33000947820757187,
        0.6699905217924281,
        0.9305681557970262,
    ),
    order=8,
    dense_prediction_ratio_float32=8.0,
    dense_prediction_ratio_float64=8.0,
)
"""Four-stage Gauss--Legendre collocation tableau of order eight.

Collocation at the four Gauss--Legendre quadrature nodes on the unit
interval: each ``a[i][j]`` integrates the j-th Lagrange basis
polynomial from zero to ``c[i]`` and each ``b[j]`` integrates it over
the whole step. The coefficients are float64 literals of that
construction. No embedded error estimate exists, so the method
requires a fixed step controller.

References
----------
Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential
Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.).
Springer. Theorem IV.5.2.
"""

DEFAULT_FIRK_TABLEAU = GAUSS_LEGENDRE_2_TABLEAU


FIRK_TABLEAU_REGISTRY: Dict[str, FIRKTableau] = {
    "firk_gauss_legendre_2": GAUSS_LEGENDRE_2_TABLEAU,
    "firk_gauss_legendre_4": GAUSS_LEGENDRE_4_TABLEAU,
    "radau_iia_5": RADAU_IIA_5_TABLEAU,
    "radau": RADAU_IIA_5_TABLEAU,
}
