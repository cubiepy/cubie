"""Tableaus for backward differentiation formula (BDF) methods.

Published Classes
-----------------
:class:`BDFTableau`
    Maximum order and step-ratio limits of a BDF step.

Functions
---------
:func:`zero_stable_ratio_limit`
    Largest constant step ratio keeping a BDF order zero-stable.

Constants
---------
:data:`BDF_TABLEAU_REGISTRY`
    Name → tableau mapping for alias-based lookup.

:data:`DEFAULT_BDF_TABLEAU`
    Default tableau (BDF2).

See Also
--------
:class:`~cubie.integrators.algorithms.generic_bdf.BDFStep`
    Step factory consuming these tableaus.
"""

from functools import lru_cache
from math import inf
from typing import Dict, Optional, Tuple

import attrs
from numpy import (
    abs as np_abs,
    argsort as np_argsort,
    array as np_array,
    concatenate as np_concatenate,
    ndarray,
    roots as np_roots,
)

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau

MAX_BDF_ORDER = 5
"""Highest zero-stable order the step supports."""

_BISECTION_STEPS = 200


def _constant_ratio_weights(order: int, ratio: float) -> ndarray:
    """Return BDF weights on past states for a constant step ratio."""
    # Node m sits s[m] current steps before the new point.
    offsets = [0.0]
    total = 0.0
    step = 1.0
    for _ in range(order):
        total += step
        offsets.append(total)
        step /= ratio
    leading = sum(1.0 / offsets[m] for m in range(1, order + 1))
    weights = []
    for j in range(1, order + 1):
        numerator = 1.0
        denominator = -offsets[j]
        for m in range(1, order + 1):
            if m != j:
                numerator *= offsets[m]
                denominator *= offsets[m] - offsets[j]
        weights.append(-(numerator / denominator) / leading)
    return np_array(weights)


def _parasitic_radius(order: int, ratio: float) -> float:
    """Return the largest root modulus other than the root at one."""
    weights = _constant_ratio_weights(order, ratio)
    roots = np_roots(np_concatenate(([1.0], -weights)))
    parasitic = roots[np_argsort(np_abs(roots - 1.0))][1:]
    return float(max(np_abs(parasitic)))


@lru_cache(maxsize=None)
def zero_stable_ratio_limit(order: int) -> float:
    """Return the largest constant step ratio keeping ``order`` zero-stable.

    Bisects on the parasitic roots; ``inf`` for order one.
    """
    if order < 2:
        return inf
    stable, unstable = 1.0, 10.0
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (stable + unstable)
        if _parasitic_radius(order, middle) < 1.0:
            stable = middle
        else:
            unstable = middle
    return stable


@attrs.frozen
class BDFTableau(ButcherTableau):
    """Backward differentiation formula up to ``order``.

    The arrays are backward Euler's one step-end stage; ``order`` is
    the highest order the step runs.

    References
    ----------
    Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential
    Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.).
    Springer. Section III.5.
    """

    a: Tuple[Tuple[float, ...], ...] = attrs.field(default=((1.0,),))
    b: Tuple[float, ...] = attrs.field(default=(1.0,))
    c: Tuple[float, ...] = attrs.field(default=(1.0,))
    order: int = attrs.field(default=2)

    def __attrs_post_init__(self) -> None:
        """Validate the order range."""
        super().__attrs_post_init__()
        if not 1 <= self.order <= MAX_BDF_ORDER:
            raise ValueError(
                f"BDF order must be between 1 and {MAX_BDF_ORDER}; got "
                f"{self.order}."
            )

    @property
    def has_error_estimate(self) -> bool:
        """Return ``True``: the predictor supplies the estimate."""
        return True

    def error_weights(self, precision) -> Optional[ndarray]:
        """Return ``None``; the estimate has no stage weights."""
        return None

    @property
    def history_length(self) -> int:
        """Return the stored states: the predictor's plus one spare."""
        return self.order + 2

    @property
    def ratio_limits(self) -> Tuple[float, ...]:
        """Return the step-ratio limit of orders 1 to ``order``.

        Order one takes the BDF2 limit.
        """
        return tuple(
            zero_stable_ratio_limit(max(order, 2))
            for order in range(1, self.order + 1)
        )


def _bdf_tableau(order: int) -> BDFTableau:
    """Return the order-``order`` tableau with its growth default."""
    defaults = {}
    if order >= 2:
        defaults["max_step_growth"] = zero_stable_ratio_limit(order)
    return BDFTableau(order=order, defaults=defaults)


BDF_TABLEAU_REGISTRY: Dict[str, BDFTableau] = {
    f"bdf{order}": _bdf_tableau(order)
    for order in range(1, MAX_BDF_ORDER + 1)
}
"""Registry of named BDF tableaus available to the integrator."""

DEFAULT_BDF_TABLEAU_NAME = "bdf2"
DEFAULT_BDF_TABLEAU = BDF_TABLEAU_REGISTRY[DEFAULT_BDF_TABLEAU_NAME]
