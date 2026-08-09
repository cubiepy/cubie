"""Tableaus for diagonally implicit Runge–Kutta (DIRK) methods.

Published Classes
-----------------
:class:`DIRKTableau`
    Extends :class:`~base_algorithm_step.ButcherTableau` with a
    ``diagonal()`` accessor for the :math:`A` matrix diagonal.

Constants
---------
:data:`IMPLICIT_MIDPOINT_TABLEAU`
    Single-stage, second-order symplectic method.

:data:`TRAPEZOIDAL_DIRK_TABLEAU`
    Two-stage ESDIRK Crank–Nicolson (trapezoidal) rule.

:data:`KVAERNO3_TABLEAU`
    Four-stage, third-order A-L stable stiffly accurate ESDIRK method.

:data:`KVAERNO5_TABLEAU`
    Seven-stage, fifth-order A-L stable stiffly accurate ESDIRK method.

:data:`SDIRK_2_2_TABLEAU`
    Two-stage, second-order L-stable SDIRK by Alexander.

:data:`L_STABLE_DIRK3_TABLEAU`
    Three-stage, third-order L-stable stiffly-accurate DIRK.

:data:`L_STABLE_SDIRK4_TABLEAU`
    Five-stage, fourth-order Hairer–Wanner L-stable SDIRK.

:data:`DIRK_TABLEAU_REGISTRY`
    Name → tableau mapping for alias-based lookup.

:data:`DEFAULT_DIRK_TABLEAU`
    Default tableau (three-stage L-stable DIRK).

See Also
--------
:class:`~cubie.integrators.algorithms.generic_dirk.DIRKStep`
    Step factory consuming these tableaus.
:class:`~cubie.integrators.algorithms.base_algorithm_step.ButcherTableau`
    Parent tableau class.
"""

from typing import Dict, Tuple

import attrs
import math

from cubie.cuda_simsafe import int32
from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau


@attrs.frozen
class DIRKTableau(ButcherTableau):
    """Coefficient tableau describing a diagonally implicit RK scheme.

    The tableau stores the Runge--Kutta coefficients required by
    diagonally implicit methods, including singly diagonally implicit
    (SDIRK) and explicit-first-stage diagonally implicit (ESDIRK) variants.

    Methods
    -------
    diagonal(precision)
        Return the diagonal elements of the :math:`A` matrix as a
        precision-typed tuple.
    prediction_source_stages
        Return the history row each stage's starting guess reads.

    References
    ----------
    Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential
    Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.).
    Springer.
    """

    def __attrs_post_init__(self) -> None:
        """Validate structure, weight sums, and stage-node consistency."""
        super().__attrs_post_init__()
        self._validate_weight_sums()
        self._validate_stage_node_consistency()

    @property
    def supports_smoothed_error(self) -> bool:
        """Return whether the last diagonal supplies a smoothing operator."""
        return self.a[-1][-1] != 0.0

    def diagonal(self, precision: type) -> Tuple[float, ...]:
        """Return the diagonal entries of the tableau."""

        diagonal_entries = tuple(
            self.a[idx][idx] for idx in range(self.stage_count)
        )
        return self.typed_vector(diagonal_entries, precision)

    @property
    def prediction_source_stages(self) -> Tuple[int, ...]:
        """Return the history row each stage's starting guess reads.

        A stage that repeats an earlier stage's time starts its
        solve from that stage's converged increment; every other
        stage starts from its own predicted increment. Members are
        ``int32`` for direct use in device code.
        """

        latest_stage_at_node = {}
        sources = []
        for stage, node in enumerate(self.c):
            sources.append(latest_stage_at_node.get(node, stage))
            latest_stage_at_node[node] = stage
        return tuple(int32(source) for source in sources)


IMPLICIT_MIDPOINT_TABLEAU = DIRKTableau(
    a=((0.5,),),
    b=(1.0,),
    c=(0.5,),
    order=2,
    dense_prediction_ratio_float32=1.0,
    dense_prediction_ratio_float64=1.0,
)
"""DIRK tableau for the implicit midpoint rule (second order).

The method is singly diagonally implicit with a single stage whose
coefficient equals :math:`1/2`. It is symplectic and A-stable, making it
useful for Hamiltonian systems.

References
----------
Sanz-Serna, J. M. (1988). Runge--Kutta schemes for Hamiltonian systems.
*BIT Numerical Mathematics*, 28(4), 877-883.
"""

TRAPEZOIDAL_DIRK_TABLEAU = DIRKTableau(
    a=(
        (0.0, 0.0),
        (0.5, 0.5),
    ),
    b=(0.5, 0.5),
    c=(0.0, 1.0),
    order=2,
    dense_prediction_ratio_float32=0.39,
    dense_prediction_ratio_float64=1.21,
)
"""DIRK tableau for the Crank--Nicolson (trapezoidal) rule.

The first stage is explicit while the second stage is implicit, placing
this scheme in the ESDIRK family. It is A-stable and time-reversible,
which makes it a popular choice for moderately stiff problems.

References
----------
Crank, J., & Nicolson, P. (1947). A practical method for numerical
solution of partial differential equations of the heat-conduction type.
*Mathematical Proceedings of the Cambridge Philosophical Society*,
43(1), 50-67.
"""

KVAERNO3_GAMMA = 0.4358665215
KVAERNO3_TABLEAU = DIRKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (KVAERNO3_GAMMA, KVAERNO3_GAMMA, 0.0, 0.0),
        (
            0.490563388419108,
            0.073570090080892,
            KVAERNO3_GAMMA,
            0.0,
        ),
        (
            0.308809969973036,
            1.490563388254106,
            -1.235239879727145,
            KVAERNO3_GAMMA,
        ),
    ),
    b=(
        0.308809969973036,
        1.490563388254106,
        -1.235239879727145,
        KVAERNO3_GAMMA,
    ),
    b_hat=(
        0.490563388419108,
        0.073570090080892,
        KVAERNO3_GAMMA,
        0.0,
    ),
    c=(0.0, 2.0 * KVAERNO3_GAMMA, 1.0, 1.0),
    order=3,
    embedded_order=2,
    dense_prediction_ratio_float32=0.85,
    dense_prediction_ratio_float64=1.28,
)
"""Four-stage, third-order Kvaerno ESDIRK tableau.

The method is A-L stable, stiffly accurate, and has an explicit first
stage. The embedded estimate is the third stage.

References
----------
Kvaerno, A. (2004). Singly diagonally implicit Runge--Kutta methods
with an explicit first stage. *BIT Numerical Mathematics*, 44,
489--502.
"""

KVAERNO5_GAMMA = 0.26
KVAERNO5_TABLEAU = DIRKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (KVAERNO5_GAMMA, KVAERNO5_GAMMA, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            0.13,
            0.84033320996790809,
            KVAERNO5_GAMMA,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.22371961478320505,
            0.47675532319799699,
            -0.06470895363112615,
            KVAERNO5_GAMMA,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.16648564323248321,
            0.1045001884159172,
            0.03631482272098715,
            -0.13090704451073998,
            KVAERNO5_GAMMA,
            0.0,
            0.0,
        ),
        (
            0.13855640231268224,
            0.0,
            -0.04245337201752043,
            0.02446657898003141,
            0.61943039072480676,
            KVAERNO5_GAMMA,
            0.0,
        ),
        (
            0.13659751177640291,
            0.0,
            -0.05496908796538376,
            -0.04118626728321046,
            0.62993304899016403,
            0.06962479448202728,
            KVAERNO5_GAMMA,
        ),
    ),
    b=(
        0.13659751177640291,
        0.0,
        -0.05496908796538376,
        -0.04118626728321046,
        0.62993304899016403,
        0.06962479448202728,
        KVAERNO5_GAMMA,
    ),
    b_hat=(
        0.13855640231268224,
        0.0,
        -0.04245337201752043,
        0.02446657898003141,
        0.61943039072480676,
        KVAERNO5_GAMMA,
        0.0,
    ),
    c=(
        0.0,
        2.0 * KVAERNO5_GAMMA,
        1.230333209967908,
        0.895765984350076,
        0.436393609858648,
        1.0,
        1.0,
    ),
    order=5,
    embedded_order=4,
)
"""Seven-stage, fifth-order Kvaerno ESDIRK tableau.

The method is A-L stable, stiffly accurate, and has an explicit first
stage. The embedded estimate is the sixth stage.

References
----------
Kvaerno, A. (2004). Singly diagonally implicit Runge--Kutta methods
with an explicit first stage. *BIT Numerical Mathematics*, 44,
489--502.
"""

SQRT2 = 2**0.5
SDIRK2_GAMMA = (2 - SQRT2) / 2.0
SDIRK_2_2_TABLEAU = DIRKTableau(
    a=(
        (SDIRK2_GAMMA, 0.0),
        (1.0 - SDIRK2_GAMMA, SDIRK2_GAMMA),
    ),
    b=(1 - SDIRK2_GAMMA, SDIRK2_GAMMA),
    c=(SDIRK2_GAMMA, 1.0),
    order=2,
    dense_prediction_ratio_float32=1.07,
    dense_prediction_ratio_float64=1.21,
)
"""Two-stage, second-order SDIRK tableau by Alexander.

The tableau is L-stable and singly diagonally implicit with diagonal
coefficient :math:`1 - \\tfrac{1}{\\sqrt{2}}`. No natural embedded pair
exists, so the method carries no error estimate and requires a fixed
step controller; other implementations derive an estimate via divided
differences.

References
----------
Alexander, R. (1977). Diagonally implicit Runge--Kutta methods for
stiff ODEs. *SIAM Journal on Numerical Analysis*, 14(6), 1006-1021.
Further cited with embedded weights in NASA's review: 
https://ntrs.nasa.gov/api/citations/20160005923/downloads/20160005923.pdf
"""
SQRT6 = 6**0.5
ARCTAN_TERM = math.atan(SQRT2 / 4.0) / 3.0
L_STABLE_DIRK3_GAMMA = (
    -SQRT2 * math.cos(ARCTAN_TERM) / 2.0
    + SQRT6 * math.sin(ARCTAN_TERM) / 2.0
    + 1.0
)
L_STABLE_DIRK3_TABLEAU = DIRKTableau(
    a=(
        (L_STABLE_DIRK3_GAMMA, 0.0, 0.0),
        ((1.0 - L_STABLE_DIRK3_GAMMA) / 2.0, L_STABLE_DIRK3_GAMMA, 0.0),
        (
            (
                -6.0 * L_STABLE_DIRK3_GAMMA**2
                + 16.0 * L_STABLE_DIRK3_GAMMA
                - 1.0
            )
            / 4.0,
            (6.0 * L_STABLE_DIRK3_GAMMA**2 - 20.0 * L_STABLE_DIRK3_GAMMA + 5.0)
            / 4.0,
            L_STABLE_DIRK3_GAMMA,
        ),
    ),
    b=(
        (-6.0 * L_STABLE_DIRK3_GAMMA**2 + 16.0 * L_STABLE_DIRK3_GAMMA - 1.0)
        / 4.0,
        (6.0 * L_STABLE_DIRK3_GAMMA**2 - 20.0 * L_STABLE_DIRK3_GAMMA + 5.0)
        / 4.0,
        L_STABLE_DIRK3_GAMMA,
    ),
    c=(
        L_STABLE_DIRK3_GAMMA,
        (1.0 + L_STABLE_DIRK3_GAMMA) / 2.0,
        1.0,
    ),
    order=3,
    dense_prediction_ratio_float32=0.85,
    dense_prediction_ratio_float64=1.07,
)
"""Three-stage, third-order L-stable DIRK method with stiff accuracy.

The tableau follows the coefficients published in MOOSE's
``LStableDirk3`` time integrator, derived from Alexander's family of
L-stable singly diagonally implicit schemes. All stages share the
diagonal value :math:`\\gamma`, and the last row equals the weight
vector, so the method is stiffly accurate.

References
----------
MOOSE Framework documentation. "LStableDirk3" time integrator.
https://mooseframework.inl.gov/source/timeintegrators/LStableDirk3.html
"""

QUARTER = 0.25
L_STABLE_SDIRK4_TABLEAU = DIRKTableau(
    a=(
        (QUARTER, 0.0, 0.0, 0.0, 0.0),
        (0.5, QUARTER, 0.0, 0.0, 0.0),
        (17.0 / 50.0, -1.0 / 25.0, QUARTER, 0.0, 0.0),
        (
            371.0 / 1360.0,
            -137.0 / 2720.0,
            15.0 / 544.0,
            QUARTER,
            0.0,
        ),
        (
            25.0 / 24.0,
            -49.0 / 48.0,
            125.0 / 16.0,
            -85.0 / 12.0,
            QUARTER,
        ),
    ),
    b=(
        25.0 / 24.0,
        -49.0 / 48.0,
        125.0 / 16.0,
        -85.0 / 12.0,
        QUARTER,
    ),
    b_hat=(59.0 / 48.0, -17.0 / 96.0, 225.0 / 32.0, -85.0 / 12.0, 0.0),
    c=(
        QUARTER,
        3.0 / 4.0,
        11.0 / 20.0,
        0.5,
        1.0,
    ),
    order=4,
    embedded_order=3,
    dense_prediction_ratio_float32=0.79,
    dense_prediction_ratio_float64=0.79,
)
"""Hairer--Wanner L-stable SDIRK tableau of order four.

The five-stage scheme delivers fourth-order accuracy with stiff
accuracy, reusing :math:`\\gamma = 1/4` on the diagonal. The tableau
matches the coefficients tabulated in Hairer and Wanner's *Solving
Ordinary Differential Equations II* (Table 6.5).

References
----------
Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential
Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.).
Springer.
"""

DIRK_TABLEAU_REGISTRY: Dict[str, DIRKTableau] = {
    "implicit_midpoint": IMPLICIT_MIDPOINT_TABLEAU,
    "trapezoidal_dirk": TRAPEZOIDAL_DIRK_TABLEAU,
    "ode23t": TRAPEZOIDAL_DIRK_TABLEAU,
    "kvaerno3": KVAERNO3_TABLEAU,
    "kvaerno5": KVAERNO5_TABLEAU,
    "sdirk_2_2": SDIRK_2_2_TABLEAU,
    "l_stable_dirk_3": L_STABLE_DIRK3_TABLEAU,
    "l_stable_sdirk_4": L_STABLE_SDIRK4_TABLEAU,
}
"""Registry of named DIRK tableaus available to the integrator."""

DEFAULT_DIRK_TABLEAU_NAME = "l_stable_dirk_3"
DEFAULT_DIRK_TABLEAU = DIRK_TABLEAU_REGISTRY[DEFAULT_DIRK_TABLEAU_NAME]
