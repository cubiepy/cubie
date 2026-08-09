"""Tests for the embedded-pair order used in step-size control."""

import numpy as np
import pytest

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    KVAERNO3_TABLEAU,
)
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DORMAND_PRINCE_54_TABLEAU,
)
from cubie.integrators.algorithms.generic_firk import FIRKStep
from cubie.integrators.algorithms.generic_firk_tableaus import (
    RADAU_IIA_5_TABLEAU,
)


def test_embedded_order_declared_with_b_hat():
    """b_hat and embedded_order are declared together or not at all."""

    with pytest.raises(ValueError, match="declared together"):
        ButcherTableau(
            a=((0.0, 0.0), (1.0, 0.0)),
            b=(0.5, 0.5),
            c=(0.0, 1.0),
            order=2,
            b_hat=(1.0, 0.0),
        )
    with pytest.raises(ValueError, match="declared together"):
        ButcherTableau(
            a=((0.0, 0.0), (1.0, 0.0)),
            b=(0.5, 0.5),
            c=(0.0, 1.0),
            order=2,
            embedded_order=1,
        )


def test_controller_order_is_the_embedded_pair_order():
    """The controller sees the embedded order, not the main order."""

    assert DORMAND_PRINCE_54_TABLEAU.embedded_order == 4
    step = DIRKStep(precision=np.float64, n=2, tableau=KVAERNO3_TABLEAU)
    assert step.order == 3
    assert step.controller_order == 2


def test_smoothed_radau_controller_order():
    """Smoothing swaps radau's controller order to the smoothed pair."""

    step = FIRKStep(
        precision=np.float64, n=2, tableau=RADAU_IIA_5_TABLEAU
    )
    assert step.order == 5
    assert step.controller_order == 2
    step.update(use_smoothed_error=True)
    assert step.controller_order == 3
