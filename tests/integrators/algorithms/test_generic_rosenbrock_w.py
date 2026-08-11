"""Tests for cubie.integrators.algorithms.generic_rosenbrock_w."""

import attrs
import numpy as np

from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.generic_rosenbrock_w import (
    GenericRosenbrockWStep,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    ROS3P_TABLEAU,
)
from cubie.odesystems.solver_helpers import (
    SolverHelperKind,
    SolverHelperRequest,
)


def test_errorless_tableau_selects_fixed_controller_defaults():
    """A Rosenbrock tableau without an error estimate selects the fixed

    step-controller defaults instead of the adaptive PI defaults.
    """
    errorless_tableau = attrs.evolve(
        ROS3P_TABLEAU, b_hat=None, embedded_order=None
    )
    assert errorless_tableau.has_error_estimate is False

    step = GenericRosenbrockWStep(
        precision=np.float32, n=3, tableau=errorless_tableau,
    )
    defaults = step.controller_defaults.step_controller
    assert defaults["step_controller"] == "fixed"


def test_cached_auxiliaries_sized_after_helper_refresh(precision, system):
    """The auxiliary cache is registered at zero size and takes its

    real size from prepare_jac's HelperResult during the helper
    refresh; the step keeps no ambient auxiliary-count state.
    """
    step = GenericRosenbrockWStep(
        precision=precision,
        n=system.sizes.states,
        evaluate_f=system.evaluate_f,
        evaluate_observables=system.evaluate_observables,
        get_solver_helper_fn=system.get_solver_helper,
        tableau=DEFAULT_ROSENBROCK_TABLEAU,
    )
    entry = buffer_registry._groups[step].entries["cached_auxiliaries"]
    assert entry.size == 0
    assert not hasattr(step, "_cached_auxiliary_count")

    step.build_implicit_helpers()

    expected = system.get_solver_helper(
        SolverHelperRequest(kind=SolverHelperKind.PREPARE_JAC)
    ).cached_auxiliary_count
    entry = buffer_registry._groups[step].entries["cached_auxiliaries"]
    assert entry.size == expected
