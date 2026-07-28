"""Tests for Rosenbrock-W tableau registry integration."""

import pytest

from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    DEFAULT_ROSENBROCK_TABLEAU_NAME,
    ROS3P_TABLEAU,
    ROSENBROCK_TABLEAUS,
)
from tests.integrators.cpu_reference import CPUODESystem, get_ref_stepper


def test_rosenbrock_registry_contains_expected_tableaus():
    """Registry should expose documented Rosenbrock-W tableaus."""

    registered = set(ROSENBROCK_TABLEAUS)
    assert {"ros3p"}.issubset(registered)
    assert DEFAULT_ROSENBROCK_TABLEAU_NAME == "ros3p"
    assert DEFAULT_ROSENBROCK_TABLEAU is ROS3P_TABLEAU


def test_rosenbrock_step_function_accepts_registry_key(
    cpu_system: CPUODESystem,
    cpu_driver_evaluator,
):
    """CPU reference stepper should resolve registry keys."""

    stepper = get_ref_stepper(
        cpu_system,
        cpu_driver_evaluator,
        "rosenbrock",
        newton_tol=1e-10,
        newton_max_iters=25,
        linear_tol=1e-10,
        linear_max_iters=cpu_system.system.sizes.states,
        linear_correction_type="minimal_residual",
        preconditioner_order=2,
        tableau="ros3p",
    )
    assert hasattr(stepper, "tableau")
    assert stepper.tableau is ROSENBROCK_TABLEAUS["ros3p"]

@pytest.mark.parametrize(
    "solver_settings_override",
    # Unique set: the assertion is that this exact registry key
    # resolves to its tableau through the chain.
    [{"algorithm": "ros3p", "step_controller": "pid"}],
    indirect=True,
)
def test_rosenbrock_step_accepts_registry_tableau(step_object):
    """GPU Rosenbrock step should load tableaus from the registry."""

    assert step_object.tableau is ROSENBROCK_TABLEAUS["ros3p"]
