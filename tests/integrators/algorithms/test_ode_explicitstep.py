"""Tests for cubie.integrators.algorithms.ode_explicitstep."""

from __future__ import annotations

import pytest
from cubie.cuda_simsafe import numba_from_dtype as from_dtype
from numpy import dtype as np_dtype

from cubie.integrators.algorithms.ode_explicitstep import (
    ExplicitStepConfig,
    ODEExplicitStep,
)
from cubie.integrators.algorithms.base_algorithm_step import BaseStepConfig


# ── ExplicitStepConfig ──────────────────────────────── #

def test_explicit_step_config_is_subclass_of_base_step_config():
    """ExplicitStepConfig inherits from BaseStepConfig."""
    assert issubclass(ExplicitStepConfig, BaseStepConfig)


# ── build / build_step ──────────────────────────────── #

def test_build_delegates_to_build_step(step_object):
    """build() unpacks config and delegates to build_step."""
    # Access step_fn triggers build; the result is cached.
    sf = step_object.step_fn
    cache = step_object._cache
    assert cache.step_fn is sf


def test_build_unpacks_config_fields(step_object, system):
    """build() extracts dxdt_fn, n, etc. from compile_settings."""
    cs = step_object.compile_settings
    assert cs.n_states == system.sizes.states
    assert cs.n_drivers == system.num_drivers
    assert cs.numba_precision == from_dtype(np_dtype(cs.precision))


# ── build_step (abstract) ──────────────────────────── #

def test_build_step_is_abstract():
    """build_step raises NotImplementedError on the base class."""
    with pytest.raises(TypeError):
        # Cannot instantiate abstract class
        ODEExplicitStep(owner=None)


# ── is_implicit ─────────────────────────────────────── #

def test_is_implicit_returns_false(step_object):
    """Explicit algorithms report is_implicit == False."""
    assert step_object.is_implicit is False
