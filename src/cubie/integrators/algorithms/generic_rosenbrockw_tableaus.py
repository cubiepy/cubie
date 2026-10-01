"""Rosenbrock-W method tableaus and registry utilities.

Published Classes
-----------------
:class:`RosenbrockTableau`
    Extends :class:`~base_algorithm_step.ButcherTableau` with ``C``,
    ``gamma``, and ``gamma_stages`` fields and typed accessors.

Constants
---------
:data:`ROS3P_TABLEAU`
    Three-stage, third-order ROS3P tableau (Rang & Angermann 2005).

:data:`RODAS3P_TABLEAU`
    Five-stage, third-order RODAS3P tableau (Kaps–Rentrop).

:data:`ROSENBROCK_23_SCIML_TABLEAU`
    Three-stage, third-order SciML Rosenbrock-23 variant.

:data:`ROSENBROCK_32_SCIML_TABLEAU`
    Three-stage SciML Rosenbrock-32: third order, Rosenbrock23 embedded.

``ROS*``, ``RODAS*``, ``GRK4*``, ``VELD*4``, ``ROK4A``, ...
    OrdinaryDiffEq.jl's Rosenbrock and Rodas tableaus, one constant per
    registry entry.

:data:`ROSENBROCK_TABLEAUS`
    Name → tableau mapping for alias-based lookup.

:data:`DEFAULT_ROSENBROCK_TABLEAU`
    Default tableau (ROS3P).

See Also
--------
:class:`~cubie.integrators.algorithms.generic_rosenbrock_w.GenericRosenbrockWStep`
    Step factory consuming these tableaus.
:class:`~cubie.integrators.algorithms.base_algorithm_step.ButcherTableau`
    Parent tableau class.
"""

from math import sqrt
from typing import Dict, Optional, Tuple

import attrs
from numpy import ndarray as np_ndarray

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau


@attrs.frozen
class RosenbrockTableau(ButcherTableau):
    """Coefficient tableau describing a Rosenbrock-W integration scheme.

    Parameters
    ----------
    a
        Lower-triangular matrix of stage coupling coefficients.
    b
        Weights applied to the stage increments when forming the solution.
    c
        Stage abscissae expressed as fractions of the step size.
    order
        Classical order of the Rosenbrock-W method.
    b_hat
        Optional embedded weights that deliver an error estimate.
    C
        Lower-triangular matrix containing Jacobian update coefficients.
    gamma
        Diagonal shift applied to the stage Jacobian solves.
    gamma_stages
        Optional per-stage diagonal shifts applied to the Jacobian solves.

    """

    C: Tuple[Tuple[float, ...], ...] = attrs.field(factory=tuple)
    gamma: float = attrs.field(default=0.25)
    gamma_stages: Tuple[float, ...] = attrs.field(factory=tuple)

    def typed_gamma_stages(
        self,
        precision: PrecisionDType,
    ) -> np_ndarray:
        """Return the per-stage gamma shifts as an array in the precision."""

        return self.typed_vector(self.gamma_stages, precision)

    @property
    def supports_smoothed_error(self) -> bool:
        """Return whether the stage operator can smooth the error."""
        return True


def _julia_tableau(
    a: Tuple[Tuple[float, ...], ...],
    C: Tuple[Tuple[float, ...], ...],
    c: Tuple[float, ...],
    gamma: float,
    gamma_stages: Tuple[float, ...],
    b: Tuple[float, ...],
    error_weights: Optional[Tuple[float, ...]],
    order: int,
    embedded_order: Optional[int],
) -> RosenbrockTableau:
    """Return a tableau from OrdinaryDiffEq.jl ``RodasTableau`` fields.

    Julia's ``A``, ``C`` and ``d`` are this step's ``a``, ``C`` and
    ``gamma_stages``; ``b_hat`` is ``b - btilde``.
    """
    b_hat = None
    if error_weights is not None:
        b_hat = tuple(
            weight - error for weight, error in zip(b, error_weights)
        )
    return RosenbrockTableau(
        a=a,
        C=C,
        b=b,
        b_hat=b_hat,
        c=c,
        order=order,
        embedded_order=embedded_order,
        gamma=gamma,
        gamma_stages=gamma_stages,
    )


def _stiffly_accurate_tableau(
    a: Tuple[Tuple[float, ...], ...],
    C: Tuple[Tuple[float, ...], ...],
    c: Tuple[float, ...],
    gamma: float,
    gamma_stages: Tuple[float, ...],
    order: int,
) -> RosenbrockTableau:
    """Return a tableau solving ``Y_s + K_s`` with embedded ``Y_s``.

    The last ``a`` row builds the final stage state ``Y_s``; the
    error estimate is the last increment ``K_s``.
    """
    last_row = tuple(a[-1])
    return RosenbrockTableau(
        a=a,
        C=C,
        b=last_row[:-1] + (1.0,),
        b_hat=last_row,
        c=c,
        order=order,
        embedded_order=order - 1,
        gamma=gamma,
        gamma_stages=gamma_stages,
    )


# --------------------------------------------------------------------------
# ROS3P (Rang & Angermann 2005), constants and structure cross-checked with:
# - SciML/OrdinaryDiffEq.jl (commit c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813)
#   lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl (ROS3PTableau)
#   https://github.com/SciML/OrdinaryDiffEq.jl/blob/c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl
# --------------------------------------------------------------------------
def _ros3p_tableau() -> RosenbrockTableau:
    """Return the three-stage third-order ROS3P tableau.

    References
    ----------
    - Rang, J., & Angermann, L. (2005). New Rosenbrock–W methods of order 3.
    - SciML/OrdinaryDiffEq.jl ROS3PTableau (see link above).
    """

    gamma = 0.5 + sqrt(3.0) / 6.0
    igamma = 1.0 / gamma
    c_matrix = (
        (0.0, 0.0, 0.0),
        (-(igamma**2), 0.0, 0.0),
        (
            -igamma * (1.0 + igamma * (2.0 - 0.5 * igamma)),
            -igamma * (2.0 - 0.5 * igamma),
            0.0,
        ),
    )
    b_aux = igamma * (2.0 / 3.0 - (1.0 / 6.0) * igamma)
    tableau = RosenbrockTableau(
        a=(
            (0.0, 0.0, 0.0),
            (igamma, 0.0, 0.0),
            (igamma, 0.0, 0.0),
        ),
        C=c_matrix,
        b=(
            igamma * (1.0 + b_aux),
            b_aux,
            igamma / 3.0,
        ),
        b_hat=(
            2.113248654051871,
            1.0,
            0.4226497308103742,
        ),
        c=(0.0, 1.0, 1.0),
        order=3,
        embedded_order=2,
        gamma=gamma,
        gamma_stages=(gamma, -0.2113248654051871, 0.5 - 2.0 * gamma),
    )
    return tableau


ROS3P_TABLEAU = _ros3p_tableau()


# --------------------------------------------------------------------------
# RODAS3P (p=3) — Kaps-Rentrop type
# Source of constants:
# - SciML/OrdinaryDiffEq.jl (commit c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813)
#   lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl (Rodas3PTableau)
#   https://github.com/SciML/OrdinaryDiffEq.jl/blob/c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl
# --------------------------------------------------------------------------
def _rodas3p_tableau() -> RosenbrockTableau:
    """Return the five-stage third-order RODAS3P tableau (Kaps–Rentrop p=3)."""

    gamma = 1.0 / 3.0

    a = (
        (0.0, 0.0, 0.0, 0.0, 0.0),  # 1
        (4.0 / 3.0, 0.0, 0.0, 0.0, 0.0),  # 2
        (0.0, 0.0, 0.0, 0.0, 0.0),  # 3
        (2.90625, 3.375, 0.40625, 0.0, 0.0),  # 4
        (2.90625, 3.375, 0.40625, 0.0, 0.0),  # 5
    )

    # Full lower-triangular (padded to 5x5).
    C = (
        (0.0, 0.0, 0.0, 0.0, 0.0),  # 1
        (-4.0, 0.0, 0.0, 0.0, 0.0),  # 2
        (8.25, 6.75, 0.0, 0.0, 0.0),  # 3
        (1.21875, -5.0625, -1.96875, 0.0, 0.0),  # 4
        (4.03125, -15.1875, -4.03125, 6.0, 0.0),  # 5
    )

    # Final (p=3): u_{n+1} = (u_n + a41*k1 + a42*k2 + a43*k3) + k5
    b = (2.90625, 3.375, 0.40625, 0.0, 1.0)

    # Embedded (p=2): û = (u_n + a41*k1 + a42*k2 + a43*k3) + k4
    b_hat = (2.90625, 3.375, 0.40625, 1.0, 0.0)

    c = (0.0, 4.0 / 9.0, 0.0, 1.0, 1.0)
    gamma_stages = (1.0 / 3.0, -1.0 / 9.0, 1.0, 0.0, 0.0)

    return RosenbrockTableau(
        a=a,
        C=C,
        b=b,
        b_hat=b_hat,
        c=c,
        order=3,
        embedded_order=2,
        gamma=gamma,
        gamma_stages=gamma_stages,
    )


RODAS3P_TABLEAU = _rodas3p_tableau()

# --------------------------------------------------------------------------
# Rosenbrock23 (3-stage, order 2 with order-3 error estimate) — SciML
# variant. Untransformed coefficients from:
# - SciML/OrdinaryDiffEq.jl (commit c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813)
#   lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl
#   (Rosenbrock23Tableau: c32=6+sqrt(2), d=1/(2+sqrt(2)))
#   https://github.com/SciML/OrdinaryDiffEq.jl/blob/c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl
# - Algorithm form and residuals:
#   lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl (perform_step!
#   for Rosenbrock23)
#   https://github.com/SciML/OrdinaryDiffEq.jl/blob/c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl
#
# SciML expresses the method in gradient form (stage vectors k_i,
# W k = f(...) with explicit -J k couplings). This module's step uses
# the transformed increment form of Hairer & Wanner (Solving ODEs II,
# section IV.7, eq. 7.17): K = Gamma @ k with
# Gamma = d*[[1,0,0],[-1,1,0],[c32-2,-c32,1]], giving
# a = alpha @ inv(Gamma), C = strict lower part of -inv(Gamma),
# b = b_k @ inv(Gamma), b - b_hat = e_k @ inv(Gamma) for the SciML
# alpha = [[0],[1/2],[0,1]], b_k = (0,1,0), e_k = (1/6,-1/3,1/6),
# and per-stage time-derivative weights (d, 0, -d) — the row sums of
# Gamma.
# --------------------------------------------------------------------------
def _rosenbrock_23_sciml_tableau() -> RosenbrockTableau:
    """Return the transformed 3-stage Rosenbrock 23 tableau (order 2)."""

    sqrt2 = sqrt(2.0)
    d = 1.0 / (2.0 + sqrt2)  # shift used in W
    inv_d = 2.0 + sqrt2

    # Stage coupling in increment form: Y_1 = u + K_0/(2d),
    # Y_2 = u + (K_0 + K_1)/d
    a = (
        (0.0, 0.0, 0.0),
        (0.5 * inv_d, 0.0, 0.0),
        (inv_d, inv_d, 0.0),
    )

    C = (
        (0.0, 0.0, 0.0),
        (-inv_d, 0.0, 0.0),
        (-2.0 * inv_d, -(14.0 + 8.0 * sqrt2), 0.0),
    )

    # y_new = y + (K_0 + K_1)/d equals the stage-2 base state, so the
    # b-matches-a-row fast path applies.
    b = (inv_d, inv_d, 0.0)

    # b - b_hat = e_k @ inv(Gamma) reproduces SciML's
    # utilde = (dt/6)(k1 - 2k2 + k3) exactly through O(dt^3).
    b_hat = (
        5.0 * sqrt2 / 6.0 + 5.0 / 3.0,
        1.0 / 3.0,
        -1.0 / 3.0 - sqrt2 / 6.0,
    )

    c = (0.0, 0.5, 1.0)

    # Row sums of Gamma: stage time-derivative weights transform too.
    gamma_stages = (d, 0.0, -d)

    return RosenbrockTableau(
        a=a,
        C=C,
        b=b,
        b_hat=b_hat,
        c=c,
        order=2,
        embedded_order=3,
        gamma=d,
        gamma_stages=gamma_stages,
    )


ROSENBROCK_23_SCIML_TABLEAU = _rosenbrock_23_sciml_tableau()


def _rosenbrock_32_tableau() -> RosenbrockTableau:
    """Return the transformed 3-stage Rosenbrock 32 tableau (order 3).

    Rosenbrock23's stages with SciML's third-order solution
    ``u = u_n + (dt/6)(k1 + 4 k2 + k3)``; the embedded solution is
    Rosenbrock23's.
    """

    sqrt2 = sqrt(2.0)
    d = 1.0 / (2.0 + sqrt2)
    inv_d = 2.0 + sqrt2

    a = (
        (0.0, 0.0, 0.0),
        (0.5 * inv_d, 0.0, 0.0),
        (inv_d, inv_d, 0.0),
    )

    C = (
        (0.0, 0.0, 0.0),
        (-inv_d, 0.0, 0.0),
        (-2.0 * inv_d, -(14.0 + 8.0 * sqrt2), 0.0),
    )

    # (1/6, 4/6, 1/6) @ inv(Gamma), inv(Gamma) = [[1,0,0],[1,1,0],[2,c32,1]]/d
    b = (
        7.0 * inv_d / 6.0,
        (10.0 + sqrt2) * inv_d / 6.0,
        inv_d / 6.0,
    )

    b_hat = (inv_d, inv_d, 0.0)

    return RosenbrockTableau(
        a=a,
        C=C,
        b=b,
        b_hat=b_hat,
        c=(0.0, 0.5, 1.0),
        order=3,
        embedded_order=2,
        gamma=d,
        gamma_stages=(d, 0.0, -d),
    )


ROSENBROCK_32_SCIML_TABLEAU = _rosenbrock_32_tableau()


# --------------------------------------------------------------------------
# Tableaus below map OrdinaryDiffEq.jl's RodasTableau fields directly:
# - SciML/OrdinaryDiffEq.jl (commit 739379950dc33ccd6bb427931225013aca5486b7)
#   lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl
#   lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl
# --------------------------------------------------------------------------
# ROS2: Verwer et al. (1999), SIAM J. Sci. Comput. 20(4).
# Julia ROS2RodasTableau (RosenbrockTableaus package).
ROS2_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0),
        (0.585786437626905, 0.0),
    ),
    C=(
        (0.0, 0.0),
        (-1.17157287525381, 0.0),
    ),
    c=(0.0, 1.0),
    gamma=1.7071067811865475,
    gamma_stages=(1.7071067811865475, -1.7071067811865475),
    b=(0.8786796564403574, 0.2928932188134525),
    error_weights=(0.2928932188134525, 0.2928932188134525),
    order=2,
    embedded_order=1,
)


# ROS2PR: Rang (2014), doi:10.24355/dbbs.084-201408121139-0.
# Julia ROS2PRRodasTableau (RosenbrockTableaus package).
ROS2PR_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0),
        (4.382975767906234, 0.0, 0.0),
        (4.382975767906234, 4.382975767906234, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0),
        (-4.382975767906234, 0.0, 0.0),
        (-4.382975767906234, -16.827500814147, 0.0),
    ),
    c=(0.0, 1.0, 1.0),
    gamma=0.228155493653962,
    gamma_stages=(0.228155493653962, 0.0, -2.7755575615628914e-17),
    b=(4.382975767906234, 4.382975767906234, 1.0),
    error_weights=(-9.968705307220848e-18, 3.3829757679062333, 1.0),
    order=2,
    embedded_order=1,
)


# ROS2S: Rang (2014), doi:10.24355/dbbs.084-201408121139-0.
# Julia ROS2SRodasTableau (RosenbrockTableaus package).
ROS2S_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0),
        (2.0000000000000036, 0.0, 0.0),
        (6.828427124746214, 3.4142135623731007, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0),
        (-6.828427124746214, 0.0, 0.0),
        (-10.949747468305889, -7.535533905932761, 0.0),
    ),
    c=(0.0, 0.585786437626905, 1.0),
    gamma=0.292893218813452,
    gamma_stages=(
        0.292893218813452,
        -0.292893218813453,
        -5.551115123125783e-17,
    ),
    b=(6.828427124746214, 3.414213562373101, 1.0),
    error_weights=(
        -0.23570226039551292,
        -0.23570226039551567,
        -0.13807118745769906,
    ),
    order=2,
    embedded_order=1,
)


# ROS3: Hairer & Wanner, Solving ODEs II (1996).
# Julia ROS3RodasTableau (RosenbrockTableaus package).
ROS3_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0),
        (-1.0156171083877703, 0.0, 0.0),
        (4.07599564525377, 9.20767942983308, 0.0),
    ),
    c=(0.0, 0.435866521508459, 0.435866521508459),
    gamma=0.435866521508459,
    gamma_stages=(0.435866521508459, 0.24291996454816805, 2.185138002766406),
    b=(1.0000000000000002, 6.1697947043828245, -0.42772256543218573),
    error_weights=(
        0.49999999999999983,
        -2.907955871680547,
        0.22354069897811568,
    ),
    order=3,
    embedded_order=2,
)


# ROS3PR: Rang (2014), doi:10.24355/dbbs.084-201408121139-0.
# Julia ROS3PRRodasTableau (RosenbrockTableaus package).
ROS3PR_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0),
        (3.0000000000000018, 0.0, 0.0),
        (3.80384757729337, 1.2679491924311226, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0),
        (-3.80384757729337, 0.0, 0.0),
        (-5.673079295488928, -1.7384634363911504, 0.0),
    ),
    c=(0.0, 2.36602540378444, 1.0),
    gamma=0.788675134594813,
    gamma_stages=(0.788675134594813, -1.577350269189627, -0.577350269189621),
    b=(4.5358983848622305, 1.2679491924311173, 1.0),
    error_weights=(-0.4598572229918937, -0.22992861149594657, 0.0),
    order=3,
    embedded_order=2,
)


# Scholz4_7: Rang (2014), doi:10.24355/dbbs.084-201408121139-0.
# Julia Scholz4_7RodasTableau (RosenbrockTableaus package).
SCHOLZ4_7_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0),
        (3.0000000000000018, 0.0, 0.0),
        (4.120834875401151, 1.2679491924311226, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0),
        (-3.80384757729337, 0.0, 0.0),
        (-6.310062633644914, -1.7746264440079669, 0.0),
    ),
    c=(0.0, 2.36602540378444, 1.25),
    gamma=0.788675134594813,
    gamma_stages=(0.788675134594813, -1.577350269189627, -0.928571905524962),
    b=(4.096775673951863, 0.953252779460065, 0.7833659121534696),
    error_weights=(
        0.3028225394953986,
        -0.06093909935296366,
        0.36071618134309585,
    ),
    order=3,
    embedded_order=2,
)


# ROS34PW1a: Rang & Angermann (2005), BIT 45, 761-787.
# Julia ROS34PW1aRodasTableau (RosenbrockTableaus package).
ROS34PW1A_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (5.0905205106702045, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (4.005173696367865, 0.19316470237944158, 1.147140180139521, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-11.679081231228288, 0.0, 0.0, 0.0),
        (-0.7100952636543062, -0.04165460771675499, 0.0, 0.0),
        (-11.979557762226603, -0.48054400523894975, 1.4350493021655284, 0.0),
    ),
    c=(0.0, 2.218787467653286, 0.0, 1.7837037931914073),
    gamma=0.435866521508459,
    gamma_stages=(
        0.435866521508459,
        -1.7829209461448272,
        0.33333333333333337,
        -1.258070496147625,
    ),
    b=(
        6.1538321465310215,
        -0.8364233759732359,
        -0.8614792120957679,
        2.294280360279042,
    ),
    error_weights=(-5.429679341539398, -1.3273810331413745, 0.0, 0.0),
    order=3,
    embedded_order=2,
)


# ROS34PW1b: Rang & Angermann (2005), BIT 45, 761-787.
# Julia ROS34PW1bRodasTableau (RosenbrockTableaus package).
ROS34PW1B_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (5.0905205106702045, 0.0, 0.0, 0.0),
        (5.0905205106702045, 0.0, 0.0, 0.0),
        (4.976281110107875, 0.027726816471584953, 0.22942803602790418, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-11.679081231228288, 0.0, 0.0, 0.0),
        (-16.40573264673668, -0.27726816471584953, 0.0, 0.0),
        (-8.38103960500476, -0.8483284091993433, 0.28700986043310556, 0.0),
    ),
    c=(0.0, 2.218787467653286, 2.218787467653286, 1.553923375357884),
    gamma=0.435866521508459,
    gamma_stages=(
        0.435866521508459,
        -1.7829209461448272,
        -2.4654190049693425,
        -0.8055299979063697,
    ),
    b=(
        5.2258276123309395,
        -0.5569711481541647,
        0.35797946935364533,
        1.7233739852106407,
    ),
    error_weights=(-5.168452127840395, -1.2635194260384186, 0.0, 0.0),
    order=3,
    embedded_order=2,
)


# ROS34PW2: Rang & Angermann (2005), BIT 45, 761-787.
# Julia ROS34PW2RodasTableau (RosenbrockTableaus package).
ROS34PW2_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (1.4192173174557647, -0.2592322116729697, 0.0, 0.0),
        (4.18476048231916, -0.28519201735549593, 2.294280360279042, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-4.588560720558084, 0.0, 0.0, 0.0),
        (-4.18476048231916, 0.28519201735549593, 0.0, 0.0),
        (-6.368179200128359, -6.795620944466837, 2.8700986043310563, 0.0),
    ),
    c=(0.0, 0.871733043016918, 0.7315799577888524, 1.0),
    gamma=0.435866521508459,
    gamma_stages=(
        0.435866521508459,
        -0.435866521508459,
        -0.4133333762338865,
        -5.551115123125783e-17,
    ),
    b=(4.1847604823191595, -0.28519201735549565, 2.2942803602790414, 1.0),
    error_weights=(
        0.2777499476479681,
        -1.4032398951759992,
        1.7726301276675507,
        0.5,
    ),
    order=3,
    embedded_order=2,
)


# ROS34PW3: Rang & Angermann (2005), BIT 45, 761-787.
# Julia ROS34PW3RodasTableau (Rosenbrock package).
ROS34PW3_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (2.3541034887609085, 0.0, 0.0, 0.0),
        (2.1274518517432335, 0.7018666706430658, 0.0, 0.0),
        (1.6573541366907125, 0.37998365119129385, 0.7677537933767512, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-2.20302237067446, 0.0, 0.0, 0.0),
        (-2.750060114993467, -0.8408569631081119, 0.0, 0.0),
        (-3.0077871973155896, -0.7070774625618249, -1.1874362749274354, 0.0),
    ),
    c=(0.0, 2.5155456020628817, 1.2577728010314408, 0.6288864005157204),
    gamma=1.0685790213016289,
    gamma_stages=(
        1.0685790213016289,
        -1.4469665807612528,
        -0.7714762485313431,
        -0.29371172261080924,
    ),
    b=(
        2.5468758076906703,
        0.5527809704437551,
        0.920521963049926,
        0.7201674983256354,
    ),
    error_weights=(
        0.20632710213677674,
        0.0026042321416049896,
        0.006723394920071124,
        0.7201674983256354,
    ),
    order=4,
    embedded_order=3,
)


# ROS34PRw: Rang (2015), doi:10.1016/j.cam.2015.03.010.
# Julia ROS34PRwRodasTableau (RosenbrockTableaus package).
ROS34PRW_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (1.9166355646921893, -0.7305046154473316, 0.0, 0.0),
        (3.7075384385487764, 1.984721005641544, -0.7228174329072325, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-4.588560720558084, 0.0, 0.0, 0.0),
        (-1.4496008611374558, 2.6585485498967283, 0.0, 0.0),
        (-0.8142320398640468, 2.1949369533270104, -0.9042300763629808, 0.0),
    ),
    c=(0.0, 0.871733043016918, 1.1537997822626886, 0.9999999999999999),
    gamma=0.435866521508459,
    gamma_stages=(
        0.435866521508459,
        -0.435866521508459,
        -0.34459816128502135,
        5.551115123125783e-17,
    ),
    b=(3.7075384385487764, 1.9847210056415439, -0.7228174329072324, 1.0),
    error_weights=(
        -0.08016142700721947,
        0.15059517863671545,
        -0.29187352202361583,
        0.26131506383377556,
    ),
    order=3,
    embedded_order=2,
)


# ROS3PRL: Rang (2014), doi:10.24355/dbbs.084-201408121139-0.
# Julia ROS3PRLRodasTableau (RosenbrockTableaus package).
ROS3PRL_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (1.147140180139521, 0.0, 0.0, 0.0),
        (2.4630707730300534, 1.147140180139521, 0.0, 0.0),
        (2.4630707730300534, 1.147140180139521, 0.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-2.631861185781065, 0.0, 0.0, 0.0),
        (-2.038451402734394, 1.8551577240019121, 0.0, 0.0),
        (-1.8050630466729911, 3.411439279441918, -1.7057196397209593, 0.0),
    ),
    c=(0.0, 0.5, 1.0, 1.0),
    gamma=0.435866521508459,
    gamma_stages=(
        0.435866521508459,
        -0.064133478491541,
        -0.0032561147686690495,
        0.0,
    ),
    b=(2.4630707730300534, 1.1471401801395211, 0.0, 1.0),
    error_weights=(
        0.14188781262114447,
        0.9677841576948438,
        -0.06855321332716582,
        0.26131506383377634,
    ),
    order=3,
    embedded_order=2,
)


# ROS3PRL2: Rang (2014), doi:10.24355/dbbs.084-201408121139-0.
# Julia ROS3PRL2RodasTableau (RosenbrockTableaus package).
ROS3PRL2_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (3.000000000000007, 0.0, 0.0, 0.0),
        (4.588560720558092, 1.147140180139521, 0.0, 0.0),
        (4.588560720558092, 1.147140180139521, 0.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-6.882841080837141, 0.0, 0.0, 0.0),
        (-12.579179703104524, -2.9475127180857186, 0.0, 0.0),
        (3.556592450580385, -0.4663124315128337, 3.545260255335102, 0.0),
    ),
    c=(0.0, 1.30759956452538, 1.0, 1.0),
    gamma=0.435866521508459,
    gamma_stages=(
        0.435866521508459,
        -0.871733043016921,
        -0.8339865967040412,
        -5.551115123125783e-17,
    ),
    b=(4.5885607205580925, 1.1471401801395211, 0.0, 1.0),
    error_weights=(
        0.8808636262903982,
        0.3041167493114213,
        0.14248471842891264,
        0.26131506383377634,
    ),
    order=3,
    embedded_order=2,
)


# ROK4a: Tranquilli & Sandu (2014), doi:10.1137/130923336.
# Julia ROK4aRodasTableau (Rosenbrock package).
ROK4A_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (1.745761101158346, 0.0, 0.0, 0.0),
        (2.4703844781940627, 0.6835475189193349, 0.0, 0.0),
        (1.6819503991264864, 0.25286213499736193, -0.13856798941027462, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-5.825741115110917, 0.0, 0.0, 0.0),
        (1.0021333747582308, 0.0, 0.0, 0.0),
        (-2.0798465277651803, -0.7428770881288675, -0.5200138498012211, 0.0),
    ),
    c=(0.0, 1.0, 0.5, 0.5),
    gamma=0.572816062482135,
    gamma_stages=(
        0.572816062482135,
        -1.338715867278416,
        0.9016343030936702,
        0.19147495119907043,
    ),
    b=(
        2.6484813878883307,
        0.7862115756123027,
        0.3466758998674807,
        1.1638407341055639,
    ),
    error_weights=(
        0.3665053527208292,
        0.2997106934923117,
        -0.03500203986368344,
        1.1638407341055639,
    ),
    order=4,
    embedded_order=3,
)


# RosShamp4: Shampine (1982), ACM TOMS 8(2), 93-113.
# Julia RosShamp4RodasTableau (RosenbrockTableaus package).
ROSSHAMP4_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (1.92, 0.24, 0.0, 0.0),
        (1.92, 0.24, 0.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-8.0, 0.0, 0.0, 0.0),
        (14.88, 2.4, 0.0, 0.0),
        (-0.896, -0.432, -0.4, 0.0),
    ),
    c=(0.0, 1.0, 0.6, 0.6),
    gamma=0.5,
    gamma_stages=(0.5, -1.5, 2.42, 0.116),
    b=(2.111111111111111, 0.5, 0.23148148148148148, 1.1574074074074074),
    error_weights=(
        0.3148148148148148,
        0.19444444444444445,
        0.0,
        1.1574074074074074,
    ),
    order=4,
    embedded_order=3,
)


# Veldd4: van Veldhuizen (1984), Computing 32, 229.
# Julia Veldd4RodasTableau (RosenbrockTableaus package).
VELDD4_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (4.812234362695436, 4.578146956747842, 0.0, 0.0),
        (4.812234362695436, 4.578146956747842, 0.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-5.333333333333331, 0.0, 0.0, 0.0),
        (6.100529678848254, 1.804736797378427, 0.0, 0.0),
        (-2.540515456634749, -9.443746328915205, -1.988471753215993, 0.0),
    ),
    c=(0.0, 0.4514162296451364, 0.8755928946018455, 0.8755928946018455),
    gamma=0.2257081148225682,
    gamma_stages=(
        0.2257081148225682,
        -0.04599403502680582,
        0.5177590504944076,
        -0.03805623938054428,
    ),
    b=(
        4.289339254654537,
        5.036098482851414,
        0.6085736420673917,
        1.355958941201148,
    ),
    error_weights=(
        2.175672787531755,
        2.950911222575741,
        -0.785974454488743,
        -1.355958941201148,
    ),
    order=4,
    embedded_order=3,
)


# Velds4: van Veldhuizen (1984), Computing 32, 229.
# Julia Velds4RodasTableau (RosenbrockTableaus package).
VELDS4_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (1.75, 0.25, 0.0, 0.0),
        (1.75, 0.25, 0.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-8.0, 0.0, 0.0, 0.0),
        (-8.0, -1.0, 0.0, 0.0),
        (0.5, -0.5, 2.0, 0.0),
    ),
    c=(0.0, 1.0, 0.5, 0.5),
    gamma=0.5,
    gamma_stages=(0.5, -1.5, -0.75, 0.25),
    b=(
        1.3333333333333333,
        0.6666666666666666,
        -1.3333333333333333,
        1.3333333333333333,
    ),
    error_weights=(
        -0.3333333333333333,
        -0.3333333333333333,
        0.0,
        -1.3333333333333333,
    ),
    order=4,
    embedded_order=3,
)


# GRK4T: Kaps & Rentrop (1979), Numer. Math. 33, 55.
# Julia GRK4TRodasTableau (RosenbrockTableaus package).
GRK4T_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (4.524708207373116, 4.163528788597648, 0.0, 0.0),
        (4.524708207373116, 4.163528788597648, 0.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-5.071675338776316, 0.0, 0.0, 0.0),
        (6.020152728650786, 0.1597506846727117, 0.0, 0.0),
        (-1.856343618686113, -8.505380858179826, -2.084075136023187, 0.0),
    ),
    c=(0.0, 0.462, 0.8802083333333334, 0.8802083333333334),
    gamma=0.231,
    gamma_stages=(
        0.231,
        -0.03962966775244303,
        0.5507789395789127,
        -0.05535098457052764,
    ),
    b=(
        3.957503746640777,
        4.624892388363313,
        0.6174772638750108,
        1.282612945269037,
    ),
    error_weights=(
        2.302155402932996,
        3.073634485392623,
        -0.8732808018045032,
        -1.282612945269037,
    ),
    order=4,
    embedded_order=3,
)


# GRK4A: Kaps & Rentrop (1979), Numer. Math. 33, 55.
# Julia GRK4ARodasTableau (RosenbrockTableaus package).
GRK4A_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (1.108860759493671, 0.0, 0.0, 0.0),
        (2.37708526198336, 0.1850114988899692, 0.0, 0.0),
        (2.37708526198336, 0.1850114988899692, 0.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-4.920188402397641, 0.0, 0.0, 0.0),
        (1.055588686048583, 3.351817267668938, 0.0, 0.0),
        (3.846869007049313, 3.42710924126818, -2.162408848753263, 0.0),
    ),
    c=(0.0, 0.438, 0.87, 0.87),
    gamma=0.395,
    gamma_stages=(
        0.395,
        -0.372672395484092,
        0.06629196544571492,
        0.4340946962568634,
    ),
    b=(
        1.84568324040584,
        0.1369796894360503,
        0.7129097783291559,
        0.6329113924050632,
    ),
    error_weights=(
        0.04831870177201765,
        -0.6471108651049505,
        0.218687666050024,
        -0.6329113924050632,
    ),
    order=4,
    embedded_order=3,
)


# Ros4LStab: Hairer & Wanner, Solving ODEs II (1996).
# Julia Ros4LStabRodasTableau (RosenbrockTableaus package).
ROS4LSTAB_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (1.867943637803922, 0.2344449711399156, 0.0, 0.0),
        (1.867943637803922, 0.2344449711399156, 0.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-7.13761503641231, 0.0, 0.0, 0.0),
        (2.580708087951457, 0.6515950076447975, 0.0, 0.0),
        (-2.137148994382534, -0.3214669691237626, -0.6949742501781779, 0.0),
    ),
    c=(0.0, 1.14564, 0.65521686381559, 0.65521686381559),
    gamma=0.57282,
    gamma_stages=(
        0.57282,
        -1.769193891319233,
        0.7592633437920482,
        -0.104902108710045,
    ),
    b=(
        2.255570073418735,
        0.2870493262186792,
        0.435317943184018,
        1.093502252409163,
    ),
    error_weights=(
        -0.2815431932141155,
        -0.0727619912493892,
        -0.1082196201495311,
        -1.093502252409163,
    ),
    order=4,
    embedded_order=3,
)


# RosenbrockW6S4OS: doi:10.1016/j.cam.2009.09.017.
# Julia RosenbrockW6S4OSRodasTableau (RosenbrockTableaus package).
ROSENBROCKW6S4OS_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.5812383407115008, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.903962441371467, 1.861519155534501, 0.0, 0.0, 0.0, 0.0),
        (2.076579719675, 0.1884255381414796, 1.870158967491032, 0.0, 0.0, 0.0),
        (
            4.435550638484312,
            5.457181798610189,
            4.61635078806893,
            3.118111952402361,
            0.0,
            0.0,
        ),
        (
            10.79170169848326,
            -10.05691522584131,
            14.99564485428419,
            5.274339954390943,
            1.42973087126119,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-2.661294105131369, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-3.128450202373838, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            -6.920335474535658,
            -1.202675288266817,
            -9.73356181141362,
            0.0,
            0.0,
            0.0,
        ),
        (
            -28.09530629102695,
            20.37126295479377,
            -41.04375275302869,
            -19.66373175620895,
            0.0,
            0.0,
        ),
        (
            9.7998186780974,
            11.93579288660318,
            3.673874929013201,
            14.8078285410955,
            0.831858399869068,
            0.0,
        ),
    ),
    c=(
        0.0,
        0.1453095851778752,
        0.3817422770256738,
        0.6367813704374599,
        0.7560744496323561,
        0.927104723987567,
    ),
    gamma=0.25,
    gamma_stages=(
        0.25,
        0.0836691184292894,
        0.0544718623516351,
        -0.3402289722355864,
        0.0337651588339529,
        -0.090307426761854,
    ),
    b=(
        6.456217074653235,
        -4.853141317768053,
        9.76531833406926,
        2.081084177278723,
        0.6603936866352417,
        0.6,
    ),
    error_weights=None,
    order=4,
    embedded_order=None,
)


# Rodas3: Sandu et al. (1997), Atmos. Environ. 31(19), 3151-3166.
# Julia Rodas3RodasTableau (RosenbrockTableaus package).
RODAS3_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 1.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0, 0.0),
        (1.0, -1.0, 0.0, 0.0),
        (1.0, -1.0, -2.6666666666666665, 0.0),
    ),
    c=(0.0, 0.0, 1.0, 1.0),
    gamma=0.5,
    gamma_stages=(0.5, 1.5, 0.0, 0.0),
    b=(2.0, 0.0, 1.0, 1.0),
    error_weights=(0.0, 0.0, 0.0, 1.0),
    order=3,
    embedded_order=2,
)


# Rodas3d: Yu, Gu, Xu & Lu (2024), arXiv:2312.02809.
# Julia Rodas3dRodasTableau (RosenbrockTableaus package).
RODAS3D_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (2.173656234277416, 0.0, 0.0, 0.0),
        (1.745761108723104, 0.0, 0.0, 0.0),
        (1.745761108723104, 0.0, 1.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0),
        (-13.387001858207178, 0.0, 0.0, 0.0),
        (0.3044231400659693, 0.307452788261533, 0.0, 0.0),
        (0.5728764641408153, 0.3477109860569936, -2.74253406964739, 0.0),
    ),
    c=(0.0, 1.2451051999132263, 1.0, 1.0),
    gamma=0.57281606,
    gamma_stages=(0.57281606, -3.819703409768521, 0.0, 0.0),
    b=(1.745761108723104, 0.0, 1.0, 1.0),
    error_weights=(0.0, 0.0, 0.0, 1.0),
    order=3,
    embedded_order=2,
)


# Rodas23W: Steinebach (2024), Proceedings of the JuliaCon Conferences.
# Julia Rodas23WRodasTableau (Rosenbrock package).
RODAS23W_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1.3333333333333333, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (2.90625, 3.375, 0.40625, 0.0, 0.0),
        (2.90625, 3.375, 0.40625, 0.0, 0.0),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (-4.0, 0.0, 0.0, 0.0, 0.0),
        (8.25, 6.75, 0.0, 0.0, 0.0),
        (1.21875, -5.0625, -1.96875, 0.0, 0.0),
        (4.03125, -15.1875, -4.03125, 6.0, 0.0),
    ),
    c=(0.0, 0.4444444444444444, 0.0, 1.0, 1.0),
    gamma=0.3333333333333333,
    gamma_stages=(0.3333333333333333, -0.1111111111111111, 1.0, 0.0, 0.0),
    b=(2.90625, 3.375, 0.40625, 1.0, 0.0),
    error_weights=(0.0, 0.0, 0.0, 1.0, -1.0),
    order=2,
    embedded_order=3,
)


# Rodas4: Hairer & Wanner, Solving ODEs II (1996).
# Julia Rodas4Tableau (RosenbrockTableaus package).
RODAS4_TABLEAU = _stiffly_accurate_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.544, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.9466785280815826, 0.2557011698983284, 0.0, 0.0, 0.0, 0.0),
        (
            3.314825187068521,
            2.896124015972201,
            0.9986419139977817,
            0.0,
            0.0,
            0.0,
        ),
        (
            1.221224509226641,
            6.019134481288629,
            12.53708332932087,
            -0.687886036105895,
            0.0,
            0.0,
        ),
        (
            1.221224509226641,
            6.019134481288629,
            12.53708332932087,
            -0.687886036105895,
            1.0,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-5.6688, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-2.430093356833875, -0.2063599157091915, 0.0, 0.0, 0.0, 0.0),
        (
            -0.1073529058151375,
            -9.594562251023355,
            -20.47028614809616,
            0.0,
            0.0,
            0.0,
        ),
        (
            7.496443313967647,
            -10.24680431464352,
            -33.99990352819905,
            11.7089089320616,
            0.0,
            0.0,
        ),
        (
            8.083246795921522,
            -7.981132988064893,
            -31.52159432874371,
            16.31930543123136,
            -6.058818238834054,
            0.0,
        ),
    ),
    c=(0.0, 0.386, 0.21, 0.63, 1.0, 1.0),
    gamma=0.25,
    gamma_stages=(0.25, -0.1043, 0.1035, -0.0362, 0.0, 0.0),
    order=4,
)


# Rodas42: Hairer & Wanner, Solving ODEs II (1996).
# Julia Rodas42Tableau (RosenbrockTableaus package).
RODAS42_TABLEAU = _stiffly_accurate_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.4028884, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.6581212688557198, -1.320936088384301, 0.0, 0.0, 0.0, 0.0),
        (
            7.131197445744498,
            16.02964143958207,
            -5.561572550509766,
            0.0,
            0.0,
            0.0,
        ),
        (
            22.73885722420363,
            67.38147284535289,
            -31.2187749303856,
            0.7285641833203814,
            0.0,
            0.0,
        ),
        (
            22.73885722420363,
            67.38147284535289,
            -31.2187749303856,
            0.7285641833203814,
            1.0,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-5.1043536, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-2.899967805418783, 4.040399359702244, 0.0, 0.0, 0.0, 0.0),
        (
            -32.64449927841361,
            -99.35311008728094,
            49.99119122405989,
            0.0,
            0.0,
            0.0,
        ),
        (
            -76.46023087151691,
            -278.5942120829058,
            153.9294840910643,
            10.97101866258358,
            0.0,
            0.0,
        ),
        (
            -76.29701586804983,
            -294.2795630511232,
            162.0029695867566,
            23.6516690309527,
            -7.652977706771382,
            0.0,
        ),
    ),
    c=(0.0, 0.3507221, 0.2557041, 0.681779, 1.0, 1.0),
    gamma=0.25,
    gamma_stages=(0.25, -0.0690221, -0.0009672, -0.087979, 0.0, 0.0),
    order=4,
)


# Rodas4P: Steinebach (1995), Preprint 1741, TH Darmstadt.
# Julia Rodas4PTableau (RosenbrockTableaus package).
RODAS4P_TABLEAU = _stiffly_accurate_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.831036793486759, 0.4955183967433795, 0.0, 0.0, 0.0, 0.0),
        (
            2.304376582692669,
            -0.05249275245743001,
            -1.176798761832782,
            0.0,
            0.0,
            0.0,
        ),
        (
            -7.170454962423024,
            -4.741636671481785,
            -16.31002631330971,
            -1.062004044111401,
            0.0,
            0.0,
        ),
        (
            -7.170454962423024,
            -4.741636671481785,
            -16.31002631330971,
            -1.062004044111401,
            1.0,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-12.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-8.791795173947035, -2.207865586973518, 0.0, 0.0, 0.0, 0.0),
        (
            10.81793056857153,
            6.780270611428266,
            19.5348594464241,
            0.0,
            0.0,
            0.0,
        ),
        (
            34.19095006749676,
            15.49671153725963,
            54.7476087596413,
            14.16005392148534,
            0.0,
            0.0,
        ),
        (
            34.62605830930532,
            15.30084976114473,
            56.99955578662667,
            18.40807009793095,
            -5.714285714285717,
            0.0,
        ),
    ),
    c=(0.0, 0.75, 0.21, 0.63, 1.0, 1.0),
    gamma=0.25,
    gamma_stages=(0.25, -0.5, -0.023504, -0.0362, 0.0, 0.0),
    order=4,
)


# Rodas4P2: Steinebach (2020), Progress in DAEs II, 165-184.
# Julia Rodas4P2Tableau (RosenbrockTableaus package).
RODAS4P2_TABLEAU = _stiffly_accurate_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.906377755268814, -0.189707390391685, 0.0, 0.0, 0.0, 0.0),
        (
            3.758617027739064,
            1.161741776019525,
            -0.849258085312803,
            0.0,
            0.0,
            0.0,
        ),
        (
            7.089566927282776,
            4.573591406461604,
            -8.423496976860259,
            -0.959280113459775,
            0.0,
            0.0,
        ),
        (
            7.089566927282776,
            4.573591406461604,
            -8.423496976860259,
            -0.959280113459775,
            1.0,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-12.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-6.354581592719008, 0.338972550544623, 0.0, 0.0, 0.0, 0.0),
        (
            -8.575016317114033,
            -7.606483992117508,
            12.22499765012482,
            0.0,
            0.0,
            0.0,
        ),
        (
            -5.888975457523102,
            -8.15739661784182,
            24.805546872612922,
            12.79040151279698,
            0.0,
            0.0,
        ),
        (
            -4.408651676063871,
            -6.692003137674639,
            24.625568527593117,
            16.627521966636085,
            -5.714285714285718,
            0.0,
        ),
    ),
    c=(0.0, 0.75, 0.321448134013046, 0.519745732277726, 1.0, 1.0),
    gamma=0.25,
    gamma_stages=(0.25, -0.5, -0.189532918363016, 0.085612108792769, 0.0, 0.0),
    order=4,
)


# Rodas4PW: Steinebach (2026), in preparation.
# Julia Rodas4PWTableau (RosenbrockTableaus package).
RODAS4PW_TABLEAU = _stiffly_accurate_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            2.9351406859394085,
            -0.2900839547917462,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.5023808526102953,
            0.16405393216724992,
            0.5533771741907447,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            3.750686547696225,
            -3.4281864369523043,
            3.040310330969146,
            3.9186650053435343,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            6.144641638053066,
            -3.5875118429024027,
            4.896418320248426,
            3.22902579478038,
            1.8769572180937995,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            7.024950636618132,
            -0.16004404386219917,
            3.011738084182477,
            2.6255276779916845,
            1.064210639984151,
            1.458733196183423,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.435229134779613,
            -3.645692955953369,
            1.8428193142644484,
            -1.3604738908042335,
            1.9007297861385437,
            1.2049388559324536,
            0.5873423923113256,
            0.0,
            0.0,
        ),
        (
            2.435229134779612,
            -3.6456929559533693,
            1.8428193142644478,
            -1.3604738908042329,
            1.9007297861385437,
            1.2049388559324536,
            0.5873423923113257,
            1.0,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            0.11431136138808462,
            5.111960520884256,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -8.68739530736641,
            -0.21553233254388848,
            -2.2134894813624935,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -10.812900826181544,
            5.419387230163944,
            -6.845923178683723,
            -5.6022546699245925,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -13.17876026787501,
            8.4031432262977,
            -14.402912022742473,
            -1.8949265796948946,
            -7.507992684163551,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.4245226070350387,
            3.916211538629165,
            6.334875580590554,
            -18.204882737666242,
            11.2386753540205,
            4.394215434356857,
            0.0,
            0.0,
            0.0,
        ),
        (
            12.966547752453305,
            1.6658702157873542,
            -1.952036642801561,
            44.8607385053866,
            -16.33815656155744,
            -7.339351477557218,
            6.415927010710538,
            0.0,
            0.0,
        ),
        (
            -3.4230613926569706,
            10.549037870633642,
            -22.013331787653808,
            59.20345912605831,
            -38.48170615459573,
            -17.83635248514032,
            5.0140912391709005,
            -5.714232814492047,
            0.0,
        ),
    ),
    c=(
        0.0,
        0.5,
        0.806306160182747,
        0.5500769630661511,
        0.645123695869199,
        0.7460176197877518,
        0.38581216095480647,
        1.0,
        1.0,
    ),
    gamma=0.25,
    gamma_stages=(
        0.25,
        -0.25,
        -0.062353072468327386,
        -0.24498696841659628,
        -0.3146820705324951,
        -0.16763676372651826,
        0.10469903956242915,
        0.0,
        0.0,
    ),
    order=4,
)


# Rodas5: Di Marzo (1993), MSc thesis, University of Geneva.
# Julia Rodas5Tableau (RosenbrockTableaus package).
RODAS5_TABLEAU = _stiffly_accurate_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3.040894194418781, 1.041747909077569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            2.576417536461461,
            1.62208306077664,
            -0.9089668560264532,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.760842080225597,
            1.446624659844071,
            -0.3036980084553738,
            0.2877498600325443,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -14.09640773051259,
            6.925207756232704,
            -41.47510893210728,
            2.343771018586405,
            24.13215229196062,
            0.0,
            0.0,
            0.0,
        ),
        (
            -14.09640773051259,
            6.925207756232704,
            -41.47510893210728,
            2.343771018586405,
            24.13215229196062,
            1.0,
            0.0,
            0.0,
        ),
        (
            -14.09640773051259,
            6.925207756232704,
            -41.47510893210728,
            2.343771018586405,
            24.13215229196062,
            1.0,
            1.0,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-10.31323885133993, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-21.04823117650003, -7.234992135176716, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            32.22751541853323,
            -4.943732386540191,
            19.44922031041879,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -20.69865579590063,
            -8.816374604402768,
            1.260436877740897,
            -0.7495647613787146,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -46.22004352711257,
            -17.49534862857472,
            -289.6389582892057,
            93.60855400400906,
            318.3822534212147,
            0.0,
            0.0,
            0.0,
        ),
        (
            34.20013733472935,
            -14.1553540271769,
            57.823356409884,
            25.83362985412365,
            1.408950972071624,
            -6.551835421242162,
            0.0,
            0.0,
        ),
        (
            42.57076742291101,
            -13.80770672017997,
            93.98938432427124,
            18.77919633714503,
            -31.5835918722337,
            -6.685968952921985,
            -5.810979938412932,
            0.0,
        ),
    ),
    c=(
        0.0,
        0.38,
        0.3878509998321533,
        0.483971893787384,
        0.457047700881958,
        1.0,
        1.0,
        1.0,
    ),
    gamma=0.19,
    gamma_stages=(
        0.19,
        -0.18230792253337147,
        -0.3192318321868749,
        0.3449828624725343,
        -0.37741756439208984,
        0.0,
        0.0,
        0.0,
    ),
    order=5,
)


# Rodas5P: Steinebach (2023), BIT 63, 27.
# Julia Rodas5PTableau (Rosenbrock package).
RODAS5P_TABLEAU = _stiffly_accurate_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2.849394379747939, 0.45842242204463923, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            -6.954028509809101,
            2.489845061869568,
            -10.358996098473584,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.8029986275628964,
            0.5072464736228206,
            -0.3988312541770524,
            -0.04721187230404641,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -7.502846399306121,
            2.561846144803919,
            -11.627539656261098,
            -0.18268767659942256,
            0.030198172008377946,
            0.0,
            0.0,
            0.0,
        ),
        (
            -7.502846399306121,
            2.561846144803919,
            -11.627539656261098,
            -0.18268767659942256,
            0.030198172008377946,
            1.0,
            0.0,
            0.0,
        ),
        (
            -7.502846399306121,
            2.561846144803919,
            -11.627539656261098,
            -0.18268767659942256,
            0.030198172008377946,
            1.0,
            1.0,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-14.155112264123755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-17.97296035885952, -2.859693295451294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            147.12150275711716,
            -1.41221402718213,
            71.68940251302358,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            165.43517024871676,
            -0.4592823456491126,
            42.90938336958603,
            -5.961986721573306,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            24.854864614690072,
            -3.0009227002832186,
            47.4931110020768,
            5.5814197821558125,
            -0.6610691825249471,
            0.0,
            0.0,
            0.0,
        ),
        (
            30.91273214028599,
            -3.1208243349937974,
            77.79954646070892,
            34.28646028294783,
            -19.097331116725623,
            -28.087943162872662,
            0.0,
            0.0,
        ),
        (
            37.80277123390563,
            -3.2571969029072276,
            112.26918849496327,
            66.9347231244047,
            -40.06618937091002,
            -54.66780262877968,
            -9.48861652309627,
            0.0,
        ),
    ),
    c=(
        0.0,
        0.6358126895828704,
        0.4095798393397535,
        0.9769306725060716,
        0.4288403609558664,
        1.0,
        1.0,
        1.0,
    ),
    gamma=0.21193756319429014,
    gamma_stages=(
        0.21193756319429014,
        -0.42387512638858027,
        -0.3384627126235924,
        1.8046452872882734,
        2.325825639765069,
        0.0,
        0.0,
        0.0,
    ),
    order=5,
)


# Rodas5Pe: Steinebach (2024), Proceedings of the JuliaCon Conferences.
# Julia Rodas5PeTableau (Rosenbrock package).
RODAS5PE_TABLEAU = _julia_tableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2.849394379747939, 0.45842242204463923, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            -6.954028509809101,
            2.489845061869568,
            -10.358996098473584,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.8029986275628964,
            0.5072464736228206,
            -0.3988312541770524,
            -0.04721187230404641,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -7.502846399306121,
            2.561846144803919,
            -11.627539656261098,
            -0.18268767659942256,
            0.030198172008377946,
            0.0,
            0.0,
            0.0,
        ),
        (
            -7.502846399306121,
            2.561846144803919,
            -11.627539656261098,
            -0.18268767659942256,
            0.030198172008377946,
            1.0,
            0.0,
            0.0,
        ),
        (
            -7.502846399306121,
            2.561846144803919,
            -11.627539656261098,
            -0.18268767659942256,
            0.030198172008377946,
            1.0,
            1.0,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-14.155112264123755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-17.97296035885952, -2.859693295451294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            147.12150275711716,
            -1.41221402718213,
            71.68940251302358,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            165.43517024871676,
            -0.4592823456491126,
            42.90938336958603,
            -5.961986721573306,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            24.854864614690072,
            -3.0009227002832186,
            47.4931110020768,
            5.5814197821558125,
            -0.6610691825249471,
            0.0,
            0.0,
            0.0,
        ),
        (
            30.91273214028599,
            -3.1208243349937974,
            77.79954646070892,
            34.28646028294783,
            -19.097331116725623,
            -28.087943162872662,
            0.0,
            0.0,
        ),
        (
            37.80277123390563,
            -3.2571969029072276,
            112.26918849496327,
            66.9347231244047,
            -40.06618937091002,
            -54.66780262877968,
            -9.48861652309627,
            0.0,
        ),
    ),
    c=(
        0.0,
        0.6358126895828704,
        0.4095798393397535,
        0.9769306725060716,
        0.4288403609558664,
        1.0,
        1.0,
        1.0,
    ),
    gamma=0.21193756319429014,
    gamma_stages=(
        0.21193756319429014,
        -0.42387512638858027,
        -0.3384627126235924,
        1.8046452872882734,
        2.325825639765069,
        0.0,
        0.0,
        0.0,
    ),
    b=(
        -7.502846399306121,
        2.561846144803919,
        -11.627539656261098,
        -0.18268767659942256,
        0.030198172008377946,
        1.0,
        1.0,
        1.0,
    ),
    error_weights=(
        0.2606326497975715,
        -0.005158627295444251,
        1.3038988631109731,
        1.235000722062074,
        -0.7931985603795049,
        -1.005448461135913,
        -0.18044626132120234,
        0.17051519239113755,
    ),
    order=5,
    embedded_order=4,
)


# Rodas6P: Steinebach (2025), arXiv:2511.21252.
# Julia Rodas6PTableau (Rosenbrock package).
# Stages 17-19 feed only Julia's dense output and are dropped.
RODAS6P_TABLEAU = _stiffly_accurate_tableau(
    a=(
        (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            1.7111784962693573,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            3.338661438538325,
            1.7785154948506772,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.936071270275081,
            0.9182685464146361,
            0.3700626437020361,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            4.659498341685848,
            1.750740798902701,
            0.5870646872926452,
            0.8880273208834594,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            4.0197306615530755,
            2.839611966871549,
            -0.5985886977898102,
            0.08804800108767567,
            1.5622259206803966,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            1.988416726724047,
            -0.379547946940864,
            0.9004347186464728,
            1.4277449221484224,
            -0.7433508015345144,
            -0.042432590368607255,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            1.8376133238441654,
            1.9114959548124457,
            -0.6715227349230231,
            0.2358079620635186,
            3.6095202089874117,
            0.8151701113738031,
            0.9206065341545108,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -0.766306772356088,
            3.209956697664864,
            -3.3123779344961592,
            -3.0203200762095332,
            4.800864725315542,
            1.1604579105760842,
            0.4424812765132964,
            0.3706918590956091,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            6.232416226700401,
            2.6089061288608786,
            -0.6004565639275875,
            -3.3845987889094653,
            0.42397260663019737,
            0.35421155529651493,
            0.30716464971632756,
            1.5008969261275715,
            0.5102657561692372,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.023392109748070492,
            1.4081998520657641,
            -0.7199787823918794,
            0.7361286083371824,
            2.4632772861278043,
            0.46923886035475726,
            0.1205787235019629,
            -0.8578747086506138,
            -0.2588726092696778,
            -0.4397748045492015,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.953951852943472,
            -0.5094757863221286,
            0.3109577019600045,
            -3.5298051247141733,
            -3.545755924579993,
            -0.33681829638738314,
            -0.5663219967973026,
            1.1332773651373889,
            0.15030559921640937,
            0.25755454716019555,
            0.29836356640198125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            3.613614004197333,
            0.6635854700997046,
            0.021719370087612728,
            -1.4950066071478674,
            0.7257768429136315,
            -0.05542296424699332,
            0.6617050893162496,
            1.5916006835996634,
            0.004468857383033254,
            0.3492741589610665,
            -0.20270398239783438,
            0.6407744206145284,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.6650675322630164,
            3.8649437891996143,
            -3.5568168140908862,
            0.30445082364848014,
            6.687033712252074,
            1.7577448564663951,
            0.7252352806302017,
            0.8340620415656512,
            0.288756122559755,
            -0.014344518613253377,
            -0.9202387269679146,
            0.1235675186947092,
            0.5210532009614854,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.6650675322630163,
            3.864943789199614,
            -3.5568168140908876,
            0.30445082364847914,
            6.687033712252074,
            1.7577448564663951,
            0.7252352806302018,
            0.834062041565651,
            0.2887561225597551,
            -0.014344518613253487,
            -0.9202387269679145,
            0.12356751869470915,
            0.5210532009614851,
            1.0,
            0.0,
            0.0,
        ),
        (
            0.6650675322630177,
            3.864943789199614,
            -3.5568168140908876,
            0.30445082364847964,
            6.687033712252074,
            1.7577448564663947,
            0.7252352806302018,
            0.8340620415656512,
            0.2887561225597553,
            -0.014344518613253388,
            -0.9202387269679146,
            0.12356751869470915,
            0.5210532009614847,
            0.9999999999999998,
            1.0,
            0.0,
        ),
    ),
    C=(
        (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -6.581455754882143,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -17.99898897860265,
            -8.573983492685619,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -9.381383431453385,
            -3.147640353879416,
            -1.3459246069197102,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -2.6265331637613007,
            -4.114341661049238,
            2.3552716210903446,
            0.7916860595752533,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            13.234071865054425,
            -6.5531726714288245,
            10.73126008968739,
            7.881893740344428,
            -12.771533510641573,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.830906994202388,
            0.2604988641272497,
            1.2537810312593667,
            -3.3671244579321455,
            -10.786563365589606,
            -1.9308385166591397,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -11.060311196714387,
            -1.3456656966931244,
            -0.7657970115506183,
            6.107723730659436,
            2.2037867523584938,
            -0.07238767937020778,
            -0.8050462039096485,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -18.12844382677043,
            -8.753725825758918,
            2.21059342699439,
            11.608007179779365,
            -0.05812583279366939,
            -0.5568300956262869,
            0.22469855334210373,
            -3.2370311176417705,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -27.976343920035056,
            -8.591555353546772,
            0.7281452536154736,
            17.638493457476986,
            -5.306189757628467,
            -3.0476569146401444,
            -5.904770682441327,
            -11.929084037829442,
            -5.050568446376497,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            16.354749252471326,
            1.4669223994142209,
            5.928484441955681,
            10.74513480723443,
            10.673355125609953,
            3.688805562318594,
            9.180717730517506,
            10.247712646451996,
            1.465303310058304,
            2.6508985881732774,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            6.914935441832399,
            5.38984958631352,
            5.862037438875566,
            5.348681436005972,
            -7.013382408252529,
            -1.0246660674824237,
            -2.9837100715597376,
            -5.836566084094612,
            -1.6109549842142277,
            -1.1760399923017764,
            1.9280128334739643,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            4.198711896534994,
            0.6181084056782703,
            0.8167077246607498,
            -11.236495255410707,
            -4.824409089261172,
            0.7728367826492113,
            1.3033341851336357,
            3.1220057171705977,
            1.917167519177393,
            0.740911936448596,
            2.3884839541537675,
            -1.0646261824518854,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            8.439457273665866,
            -12.395217711948733,
            8.717809686910918,
            -22.620864451522902,
            -20.113605406532766,
            -4.689776805197894,
            1.6982447341017708,
            4.543406791431926,
            1.613366829028557,
            1.9707273458768597,
            5.732578765805261,
            1.7624664316708778,
            -5.245856774591262,
            0.0,
            0.0,
            0.0,
        ),
        (
            1.71450490512272,
            -13.840042084553074,
            6.437347401872298,
            -39.22048912909508,
            -25.603335547270504,
            -8.064795628637777,
            -0.0416146802576695,
            1.2346482358729156,
            2.7009872115209252,
            0.896525973043981,
            11.303609670096813,
            1.7547024586563815,
            -10.02276713006834,
            -6.93945857648056,
            0.0,
            0.0,
        ),
        (
            19.61213176916848,
            -14.311603723508286,
            12.96694128305561,
            -28.107332490598257,
            -27.192311861144052,
            -4.26577843330566,
            4.2210733410202135,
            10.487402162301366,
            0.8300940888935481,
            2.8025411207314455,
            8.878715452726594,
            1.4612348690788786,
            -9.853595041510669,
            -7.6808377648250294,
            -6.870056791600298,
            0.0,
        ),
    ),
    c=(
        0.0,
        0.4449064090300329,
        0.5391930604628539,
        0.3920739557917205,
        0.5393851240464334,
        0.7496615946466092,
        0.09171052879621677,
        0.716762001806476,
        0.9201684737037024,
        0.7017495611178288,
        0.5587152179138446,
        0.10896187906446,
        0.5073827520419607,
        0.9999999999999999,
        0.9999999999999999,
        1.0000000000000002,
    ),
    gamma=0.26,
    gamma_stages=(
        0.26,
        -0.18490640903003291,
        -0.5445316852875675,
        -0.03230297796648507,
        -0.05985832397786847,
        0.08292573124960323,
        0.4158601113780379,
        -0.4887636036121086,
        -0.5305551731438798,
        0.12166683722729399,
        -0.14899579330238244,
        0.20995126195089908,
        -0.06287825975966793,
        -1.1102230246251565e-16,
        1.1102230246251565e-16,
        2.220446049250313e-16,
    ),
    order=6,
)


ROSENBROCK_TABLEAUS: Dict[str, RosenbrockTableau] = {
    "ros3p": ROS3P_TABLEAU,
    "rodas3p": RODAS3P_TABLEAU,
    "rosenbrock23": ROSENBROCK_23_SCIML_TABLEAU,  # MATLAB ode23s 2(3)
    "ode23s": ROSENBROCK_23_SCIML_TABLEAU,
    "rosenbrock23_sciml": ROSENBROCK_23_SCIML_TABLEAU,  # 3-stage SciML variant
    "rosenbrock32": ROSENBROCK_32_SCIML_TABLEAU,
    "ros2": ROS2_TABLEAU,
    "ros2pr": ROS2PR_TABLEAU,
    "ros2s": ROS2S_TABLEAU,
    "ros3": ROS3_TABLEAU,
    "ros3pr": ROS3PR_TABLEAU,
    "scholz4_7": SCHOLZ4_7_TABLEAU,
    "ros34pw1a": ROS34PW1A_TABLEAU,
    "ros34pw1b": ROS34PW1B_TABLEAU,
    "ros34pw2": ROS34PW2_TABLEAU,
    "ros34pw3": ROS34PW3_TABLEAU,
    "ros34prw": ROS34PRW_TABLEAU,
    "ros3prl": ROS3PRL_TABLEAU,
    "ros3prl2": ROS3PRL2_TABLEAU,
    "rok4a": ROK4A_TABLEAU,
    "rosshamp4": ROSSHAMP4_TABLEAU,
    "veldd4": VELDD4_TABLEAU,
    "velds4": VELDS4_TABLEAU,
    "grk4t": GRK4T_TABLEAU,
    "grk4a": GRK4A_TABLEAU,
    "ros4lstab": ROS4LSTAB_TABLEAU,
    "rosenbrockw6s4os": ROSENBROCKW6S4OS_TABLEAU,
    "rodas3": RODAS3_TABLEAU,
    "rodas3d": RODAS3D_TABLEAU,
    "rodas23w": RODAS23W_TABLEAU,
    "rodas4": RODAS4_TABLEAU,
    "rodas42": RODAS42_TABLEAU,
    "rodas4p": RODAS4P_TABLEAU,
    "rodas4p2": RODAS4P2_TABLEAU,
    "rodas4pw": RODAS4PW_TABLEAU,
    "rodas5": RODAS5_TABLEAU,
    "rodas5p": RODAS5P_TABLEAU,
    "rodas5pe": RODAS5PE_TABLEAU,
    "rodas6p": RODAS6P_TABLEAU,
}

DEFAULT_ROSENBROCK_TABLEAU_NAME = "ros3p"
DEFAULT_ROSENBROCK_TABLEAU = ROSENBROCK_TABLEAUS[
    DEFAULT_ROSENBROCK_TABLEAU_NAME
]


__all__ = [
    "RosenbrockTableau",
    "ROS3P_TABLEAU",
    "RODAS3P_TABLEAU",
    "ROSENBROCK_23_SCIML_TABLEAU",
    "ROSENBROCK_32_SCIML_TABLEAU",
    "ROS2_TABLEAU",
    "ROS2PR_TABLEAU",
    "ROS2S_TABLEAU",
    "ROS3_TABLEAU",
    "ROS3PR_TABLEAU",
    "SCHOLZ4_7_TABLEAU",
    "ROS34PW1A_TABLEAU",
    "ROS34PW1B_TABLEAU",
    "ROS34PW2_TABLEAU",
    "ROS34PW3_TABLEAU",
    "ROS34PRW_TABLEAU",
    "ROS3PRL_TABLEAU",
    "ROS3PRL2_TABLEAU",
    "ROK4A_TABLEAU",
    "ROSSHAMP4_TABLEAU",
    "VELDD4_TABLEAU",
    "VELDS4_TABLEAU",
    "GRK4T_TABLEAU",
    "GRK4A_TABLEAU",
    "ROS4LSTAB_TABLEAU",
    "ROSENBROCKW6S4OS_TABLEAU",
    "RODAS3_TABLEAU",
    "RODAS3D_TABLEAU",
    "RODAS23W_TABLEAU",
    "RODAS4_TABLEAU",
    "RODAS42_TABLEAU",
    "RODAS4P_TABLEAU",
    "RODAS4P2_TABLEAU",
    "RODAS4PW_TABLEAU",
    "RODAS5_TABLEAU",
    "RODAS5P_TABLEAU",
    "RODAS5PE_TABLEAU",
    "RODAS6P_TABLEAU",
    "ROSENBROCK_TABLEAUS",
    "DEFAULT_ROSENBROCK_TABLEAU",
    "DEFAULT_ROSENBROCK_TABLEAU_NAME",
]
