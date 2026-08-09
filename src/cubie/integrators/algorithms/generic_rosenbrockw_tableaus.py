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
from typing import Dict, Tuple

import attrs

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
        numba_precision: type,
    ) -> Tuple[float, ...]:
        """Return stage-specific gamma shifts typed to ``numba_precision``."""

        return self.typed_vector(self.gamma_stages, numba_precision)

    @property
    def supports_smoothed_error(self) -> bool:
        """Return whether the stage operator can smooth the error."""
        return True


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

# RODAS4P and RODAS5P don't have step-end embedded weights - architectural
# rework would be required to implement these.
# --------------------------------------------------------------------------
# RODAS4P (p=4)
# Source of constants:
# - SciML/OrdinaryDiffEq.jl (commit c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813)
#   lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl (RODAS4PA, RODAS4PC, RODAS4Pc, RODAS4Pd)
#   https://github.com/SciML/OrdinaryDiffEq.jl/blob/c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl
# --------------------------------------------------------------------------
# RODAS4P_TABLEAU = RosenbrockTableau(
#     a=(
#         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (1.831036793486759, 0.4955183967433795, 0.0, 0.0, 0.0, 0.0),
#         (2.304376582692669, -0.05249275245743001, -1.176798761832782, 0.0, 0.0, 0.0),
#         (-7.170454962423024, -4.741636671481785, -16.31002631330971, -1.062004044111401, 0.0, 0.0),
#         (-7.170454962423024, -4.741636671481785, -16.31002631330971, -1.062004044111401, 1.0, 0.0),
#     ),
#     # Full 6x6 lower-triangular (padded last column with zeros)
#     C=(
#         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (-12.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (-8.791795173947035, -2.207865586973518, 0.0, 0.0, 0.0, 0.0),
#         (10.81793056857153, 6.780270611428266, 19.53485944642410, 0.0, 0.0, 0.0),
#         (34.19095006749676, 15.49671153725963, 54.74760875964130, 14.16005392148534, 0.0, 0.0),
#         (34.62605830930532, 15.30084976114473, 56.99955578662667, 18.40807009793095, -5.714285714285717, 0.0),
#     ),
#     b=(
#         -7.170454962423024,
#         -4.741636671481785,
#         -16.31002631330971,
#         -1.062004044111401,
#         1.0,
#         0.0,
#     ),
#     b_hat=(
#         -7.170454962423024,
#         -4.741636671481785,
#         -16.31002631330971,
#         -1.062004044111401,
#         0.0,
#         0.0,
#     ),
#     c=(0.0, 0.75, 0.21, 0.63, 1.0, 1.0),
#     order=4,
#     gamma=0.25,
#     gamma_stages=(0.25, -0.5, -0.023504, -0.0362, 0.0, 0.0),
# )
#
#
# # --------------------------------------------------------------------------
# # RODAS5P (p=5)
# # Source of constants:
# # - SciML/OrdinaryDiffEq.jl (commit c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813)
# #   lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl (RODAS5PA, RODAS5PC, RODAS5Pc, RODAS5Pd)
# #   https://github.com/SciML/OrdinaryDiffEq.jl/blob/c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl
# # --------------------------------------------------------------------------
# RODAS5P_TABLEAU = RosenbrockTableau(
#     a=(
#         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (2.849394379747939, 0.45842242204463923, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (-6.954028509809101, 2.489845061869568, -10.358996098473584, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (2.8029986275628964, 0.5072464736228206, -0.3988312541770524, -0.04721187230404641, 0.0, 0.0, 0.0, 0.0),
#         (-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 0.0, 0.0, 0.0),
#         (-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1.0, 0.0, 0.0),
#         (-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1.0, 1.0, 0.0),
#     ),
#     # Full 8x8 lower-triangular (padded last column with zeros)
#     C=(
#         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (-14.155112264123755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (-17.97296035885952, -2.859693295451294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (147.12150275711716, -1.41221402718213, 71.68940251302358, 0.0, 0.0, 0.0, 0.0, 0.0),
#         (165.43517024871676, -0.4592823456491126, 42.90938336958603, -5.961986721573306, 0.0, 0.0, 0.0, 0.0),
#         (24.854864614690072, -3.0009227002832186, 47.4931110020768, 5.5814197821558125, -0.6610691825249471, 0.0, 0.0, 0.0),
#         (30.91273214028599, -3.1208243349937974, 77.79954646070892, 34.28646028294783, -19.097331116725623, -28.087943162872662, 0.0, 0.0),
#         (37.80277123390563, -3.2571969029072276, 112.26918849496327, 66.9347231244047, -40.06618937091002, -54.66780262877968, -9.48861652309627, 0.0),
#     ),
#     b=(
#         -7.502846399306121,
#         2.561846144803919,
#         -11.627539656261098,
#         -0.18268767659942256,
#         0.030198172008377946,
#         1.0,
#         1.0,
#         0.0,
#     ),
#     b_hat=(
#         -7.502846399306121,
#         2.561846144803919,
#         -11.627539656261098,
#         -0.18268767659942256,
#         0.030198172008377946,
#         1.0,
#         0.0,
#         0.0,
#     ),
#     c=(
#         0.0,
#         0.6358126895828704,
#         0.4095798393397535,
#         0.9769306725060716,
#         0.4288403609558664,
#         1.0,
#         1.0,
#         1.0,
#     ),
#     order=5,
#     gamma=0.21193756319429014,
#     gamma_stages=(
#         0.21193756319429014,
#         -0.42387512638858027,
#         -0.3384627126235924,
#         1.8046452872882734,
#         2.325825639765069,
#         0.0,
#         0.0,
#         0.0,
#     ),
# )

# NOT WORKING
# --------------------------------------------------------------------------
# Rosenbrock 2(3) method used by MATLAB ode23s (Shampine & Reichelt, 1997)
# (kept as-is; independent of SciML’s 3-stage Rosenbrock23 below)
# --------------------------------------------------------------------------
# r23_gamma = 1.0 / (2.0 + 2.0**0.5)
# r23_C10 = 2.0 * (1.0 - r23_gamma) / (r23_gamma * r23_gamma)
#
#
# ROSENBROCK_23_TABLEAU = RosenbrockTableau(
#     a=(
#         (0.0, 0.0),
#         (1.0, 0.0),
#     ),
#     C=(
#         (0.0, 0.0),
#         (r23_C10, 0.0),
#     ),
#     b=(0.5, 0.5),
#     b_hat=(1.0, 0.0),
#     c=(0.0, 1.0),
#     order=2,
#     gamma=1.0 / (2.0 + 2.0**0.5),
#     gamma_stages=(r23_gamma, r23_gamma),
# )


# --------------------------------------------------------------------------
# Rosenbrock23 (3-stage, order 2 with order-3 error estimate) — SciML
# variant. Untransformed coefficients from:
# - SciML/OrdinaryDiffEq.jl (commit c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813)
#   lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl (Rosenbrock23Tableau: c32=6+sqrt(2), d=1/(2+sqrt(2)))
#   https://github.com/SciML/OrdinaryDiffEq.jl/blob/c174fbc1b07c252fe8ec8ad5b6e4d5fb9979c813/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl
# - Algorithm form and residuals:
#   lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl (perform_step! for Rosenbrock23)
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


# ROSENBROCK_W6S4OS_TABLEAU = RosenbrockTableau(
#     ...existing code...
# )


ROSENBROCK_TABLEAUS: Dict[str, RosenbrockTableau] = {
    "ros3p": ROS3P_TABLEAU,
    "rodas3p": RODAS3P_TABLEAU,
    # "rodas4p": RODAS4P_TABLEAU,
    # "rodas5p": RODAS5P_TABLEAU,
    "rosenbrock23": ROSENBROCK_23_SCIML_TABLEAU,  # MATLAB ode23s 2(3)
    "ode23s": ROSENBROCK_23_SCIML_TABLEAU,
    "rosenbrock23_sciml": ROSENBROCK_23_SCIML_TABLEAU,  # 3-stage SciML variant
}

DEFAULT_ROSENBROCK_TABLEAU_NAME = "ros3p"
DEFAULT_ROSENBROCK_TABLEAU = ROSENBROCK_TABLEAUS[
    DEFAULT_ROSENBROCK_TABLEAU_NAME
]


__all__ = [
    "RosenbrockTableau",
    "ROS3P_TABLEAU",
    "RODAS3P_TABLEAU",
    # "RODAS4P_TABLEAU",
    # "RODAS5P_TABLEAU",
    "ROSENBROCK_23_SCIML_TABLEAU",
    "ROSENBROCK_TABLEAUS",
    "DEFAULT_ROSENBROCK_TABLEAU",
    "DEFAULT_ROSENBROCK_TABLEAU_NAME",
]
