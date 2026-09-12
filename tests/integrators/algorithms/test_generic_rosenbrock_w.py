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


def test_errorless_tableau_selects_fixed_controller_defaults(system):
    """A Rosenbrock tableau without an error estimate selects the fixed

    step-controller defaults instead of the adaptive PI defaults.
    """
    errorless_tableau = attrs.evolve(
        ROS3P_TABLEAU, b_hat=None, embedded_order=None
    )
    assert errorless_tableau.has_error_estimate is False

    step = GenericRosenbrockWStep(
        get_solver_helper_fn=system.get_solver_helper,
        precision=np.float32, n_states=3, tableau=errorless_tableau,
    )
    defaults = step.controller_default_settings
    assert defaults["step_controller"] == "fixed"


def test_shared_stage_increment_gets_its_own_window(system):
    """Shared stage_increment gets a window disjoint from stage_store."""
    step = GenericRosenbrockWStep(
        get_solver_helper_fn=system.get_solver_helper,
        precision=np.float32,
        n_states=3,
        tableau=ROS3P_TABLEAU,
        stage_rhs_location="shared",
        stage_store_location="shared",
    )
    group = buffer_registry._groups[step]
    store = group.shared_layout["stage_store"]
    increment = group.shared_layout["stage_increment"]
    assert increment.stop - increment.start == 3
    assert (
        increment.start >= store.stop or increment.stop <= store.start
    )


def test_cached_auxiliaries_sized_by_the_helper(precision, system):
    """The auxiliary cache takes its size from prepare_jac's
    HelperResult when the helpers are wired at construction; the step
    keeps no ambient auxiliary-count state.
    """
    step = GenericRosenbrockWStep(
        get_solver_helper_fn=system.get_solver_helper,
        precision=precision,
        n_states=system.sizes.states,
        dxdt_fn=system.dxdt_fn,
        observables_fn=system.observables_fn,
        tableau=DEFAULT_ROSENBROCK_TABLEAU,
    )
    assert not hasattr(step, "_cached_auxiliary_count")

    expected = system.get_solver_helper(
        role="prepare_jac", jacobian_at="step"
    ).cached_auxiliary_count
    entry = buffer_registry._groups[step].entries["cached_auxiliaries"]
    assert entry.size == expected
