"""Structural tests for :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from typing import Callable

import pytest

from tests._utils import STATE_OBS_NO_TIMING


# Build, update, getter tests combined
def test_getters(
    loop_mutable,
    precision,
    solver_settings,
):
    loop = loop_mutable
    assert isinstance(loop.device_function, Callable), "Loop builds"

    # Test getters get
    assert loop.precision == precision, "precision getter"
    assert loop.save_every == precision(solver_settings['save_every']), \
        "save_every getter"
    assert loop.summarise_every == precision(
        solver_settings['summarise_every']), \
        "summarise_every getter"
    # test update
    loop.update({"save_every": 2 * solver_settings["save_every"]})
    assert loop.save_every == pytest.approx(
        2 * solver_settings["save_every"], rel=1e-6, abs=1e-6
    )


def test_device_function_forwarding_getters(loop):
    """compile_settings-forwarding properties return the cached values."""
    cs = loop.compile_settings
    assert loop.step_controller_fn is cs.step_controller_fn
    assert loop.step_function is cs.step_function
    assert loop.evaluate_driver_at_t is cs.evaluate_driver_at_t
    assert loop.evaluate_observables is cs.evaluate_observables
    assert loop.dt == cs.dt
    assert loop.is_adaptive == cs.is_adaptive


def test_update_with_no_changes_returns_empty_set(loop_mutable):
    """update() with no arguments returns an empty set without error."""
    assert loop_mutable.update() == set()
    assert loop_mutable.update(updates_dict={}) == set()


def test_update_raises_on_unrecognised_parameter(loop_mutable):
    """An unrecognised update key raises KeyError."""
    with pytest.raises(KeyError, match="Unrecognized parameters"):
        loop_mutable.update(not_a_real_parameter=1)


@pytest.mark.parametrize(
    "solver_settings_override",
    [STATE_OBS_NO_TIMING],
    indirect=True,
)
def test_save_last_flag_from_config(loop_mutable):
    """Verify IVPLoop reads save_last flag from ODELoopConfig.

    When all timing parameters are None, ODELoopConfig sets save_last=True.
    IVPLoop.build() should read this from config.save_last.
    """
    config = loop_mutable.compile_settings
    assert config.save_last is True
