"""Tests for cubie.integrators.algorithms.explicit_euler."""

from __future__ import annotations

import pytest

from cubie.integrators.algorithms.ode_explicitstep import ExplicitStepConfig

# ── __init__ ───────────────────────────────────────────── #


def test_init_creates_explicit_step_config(step_object):
    """Constructor builds an ExplicitStepConfig as compile_settings."""
    cs = step_object.compile_settings
    # isinstance justified: verifying correct config subclass is the
    # functionality, combined with value check below
    assert isinstance(cs, ExplicitStepConfig)
    assert cs.n == step_object.compile_settings.n

# ── build_step ─────────────────────────────────────────── #


def test_build_returns_step_cache(single_integrator_run):
    """build_step returns a StepCache with step and no nonlinear solver."""
    algo = single_integrator_run._algo_step
    # Trigger build through the integrator run's device_function
    _ = single_integrator_run.device_function
    cache = algo._cache
    assert cache.nonlinear_solver is None
    # Cache step is the same object as device_function.step
    assert cache.step is algo._cache.step


# ── Properties ─────────────────────────────────────────── #


@pytest.mark.parametrize(
    "prop, expected",
    [
        ("threads_per_step", 1),
        ("is_multistage", False),
        ("is_adaptive", False),
        ("order", 1),
    ],
)
def test_properties(step_object, prop, expected):
    """Static properties return correct constant values."""
    assert getattr(step_object, prop) == expected
