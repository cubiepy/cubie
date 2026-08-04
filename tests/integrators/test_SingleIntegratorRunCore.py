"""Tests for cubie.integrators.SingleIntegratorRunCore."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cubie.integrators.algorithms.generic_erk_tableaus import (
    CLASSICAL_RK4_TABLEAU,
    DORMAND_PRINCE_54_TABLEAU,
)
from cubie.integrators.SingleIntegratorRunCore import SingleIntegratorRunCore
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from tests._utils import (
    ALGORITHM_CHAIN_SETS,
    DEVICE_SOLVE_SETTINGS,
    STATE_OBS_NO_TIMING,
    SUMMARY_ONLY_NO_TIMING,
    SUMMARY_ONLY_TIMED,
    _get_evaluate_driver_at_t,
)
from tests._utils import (
    CN_ADAPTIVE_KRYLOV_GIVEN,
    RODAS3P_ADAPTIVE_KRYLOV_DEFAULT,
    RODAS3P_ADAPTIVE_KRYLOV_GIVEN,
)


# ── Construction (__init__) ─────────────────────────────────────────────── #

def test_construction_minimal(system):
    """Construction succeeds with minimal required args."""
    core = SingleIntegratorRunCore(
        system=system,
        algorithm_settings={"algorithm": "euler"},
    )
    assert core._system is system
    assert core._algo_step is not None
    assert core._step_controller is not None
    assert core._loop is not None
    assert core._output_functions is not None


def test_construction_omits_algorithm_settings(system):
    """algorithm_settings=None defaults to {} before the required

    'algorithm' key is validated, so construction still raises
    ValueError once get_algorithm_step finds no algorithm selected.
    """
    with pytest.raises(ValueError, match="must include 'algorithm'"):
        SingleIntegratorRunCore(system=system)


def test_algorithm_step_receives_driver_count(system):
    """The algorithm step's config carries the system's driver count."""
    core = SingleIntegratorRunCore(
        system=system,
        algorithm_settings={"algorithm": "firk"},
    )
    assert system.sizes.drivers > 0
    assert core._algo_step.n_drivers == system.sizes.drivers


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["erk"]],
    indirect=True,
)
def test_construction_explicit_settings(
    single_integrator_run,
    solver_settings,
    tolerance,
):
    """Construction with explicit values produces matching configuration."""
    run = single_integrator_run
    assert run.algorithm == "erk"
    assert run.step_controller == "pid"
    assert run.is_adaptive is True
    assert run.dt_min == pytest.approx(
        solver_settings["dt_min"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )


def test_default_controller_settings_from_algorithm(
    system,
    driver_array,
    algorithm_settings,
    output_settings,
    loop_settings,
):
    """When no step_control_settings, algorithm defaults are applied.

    ``step_control_settings=None`` is a constructor input the chain
    fixtures cannot express (they always pass a full settings dict),
    so this constructor-shape test builds directly.
    """
    run = SingleIntegratorRun(
        system=system,
        loop_settings=dict(loop_settings),
        evaluate_driver_at_t=_get_evaluate_driver_at_t(driver_array),
        step_control_settings=None,
        algorithm_settings=dict(algorithm_settings),
        output_settings=dict(output_settings),
    )

    defaults = run._algo_step.controller_defaults.copy()
    assert (
        run.step_controller
        == defaults.step_controller["step_controller"]
    )
    controller_settings = run._step_controller.settings_dict
    defaults.step_controller.pop("step_controller")
    for key, expected in defaults.step_controller.items():
        assert key in controller_settings
        actual = controller_settings[key]
        if isinstance(expected, (float, np.floating)):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected
    assert run._step_controller.n == system.sizes.states
    if hasattr(run._step_controller, "algorithm_order"):
        assert run._step_controller.algorithm_order == run._algo_step.order


def test_precision_popped_from_output_settings(
    system,
    driver_array,
    algorithm_settings,
    output_settings,
    loop_settings,
):
    """Precision in output_settings is ignored; system precision is used.

    A wrong ``precision`` key inside ``output_settings`` is a
    constructor input the chain fixtures cannot express, so this
    constructor-shape test builds directly.
    """
    wrong_precision = (
        np.float64 if system.precision == np.float32 else np.float32
    )
    settings = dict(output_settings)
    settings["precision"] = wrong_precision
    run = SingleIntegratorRun(
        system=system,
        loop_settings=dict(loop_settings),
        evaluate_driver_at_t=_get_evaluate_driver_at_t(driver_array),
        algorithm_settings=dict(algorithm_settings),
        output_settings=settings,
    )
    assert run._output_functions.compile_settings.precision == system.precision


def test_dt_from_step_control_reaches_controller(
    system,
    driver_array,
    algorithm_settings,
    output_settings,
    loop_settings,
):
    """dt from step_control_settings flows through to the controller.

    A bare ``{"dt": ...}`` step-control dict is a constructor input
    the chain fixtures cannot express, so this constructor-shape test
    builds directly.
    """
    run = SingleIntegratorRun(
        system=system,
        loop_settings=dict(loop_settings),
        evaluate_driver_at_t=_get_evaluate_driver_at_t(driver_array),
        step_control_settings={"dt": 0.005},
        algorithm_settings=dict(algorithm_settings),
        output_settings=dict(output_settings),
    )
    assert run.dt == pytest.approx(0.005, rel=1e-3)


def test_user_step_control_overrides_algorithm_defaults(
    system,
    driver_array,
    algorithm_settings,
    output_settings,
    loop_settings,
):
    """User-supplied step_control_settings override algorithm defaults.

    A partial step-control dict with no ``step_controller`` key is a
    constructor input the chain fixtures cannot express, so this
    constructor-shape test builds directly.
    """
    precision = system.precision
    overrides = {"dt_min": 5e-5, "dt_max": 5e-2, "min_gain": 0.3}
    override_settings = {
        key: precision(value) if isinstance(value, float) else value
        for key, value in overrides.items()
    }
    settings = dict(algorithm_settings)
    settings["algorithm"] = "crank_nicolson"
    run = SingleIntegratorRun(
        system=system,
        loop_settings=dict(loop_settings),
        evaluate_driver_at_t=_get_evaluate_driver_at_t(driver_array),
        step_control_settings=dict(override_settings),
        algorithm_settings=settings,
        output_settings=dict(output_settings),
    )

    assert run.step_controller == "gustafsson"
    assert run.dt_min == pytest.approx(override_settings["dt_min"])
    assert run.dt_max == pytest.approx(override_settings["dt_max"])
    controller_settings = run._step_controller.settings_dict
    assert controller_settings["min_gain"] == pytest.approx(
        override_settings["min_gain"]
    )
    assert controller_settings["algorithm_order"] == run._algo_step.order


# ── _process_loop_timing ────────────────────────────────────────────────── #

@pytest.mark.parametrize(
    "solver_settings_override",
    [STATE_OBS_NO_TIMING],
    indirect=True,
)
def test_save_last_when_no_save_every(single_integrator_run):
    """save_last=True when time-domain outputs requested without save_every."""
    assert single_integrator_run.save_last is True


@pytest.mark.parametrize(
    "solver_settings_override",
    [SUMMARY_ONLY_NO_TIMING],
    indirect=True,
)
def test_is_duration_dependent_no_timing(single_integrator_run):
    """is_duration_dependent True when summaries requested with no timing."""
    assert single_integrator_run.is_duration_dependent is True


@pytest.mark.parametrize(
    "solver_settings_override",
    # Unique set: a sample cadence with summarise_every still unset
    # is exactly the condition that must stay duration-dependent.
    [{**SUMMARY_ONLY_NO_TIMING, "sample_summaries_every": 0.01}],
    indirect=True,
)
def test_is_duration_dependent_with_sample_timing(single_integrator_run):
    """is_duration_dependent True when summarise_every unset."""
    assert single_integrator_run.is_duration_dependent is True


@pytest.mark.parametrize(
    "solver_settings_override",
    # Unique set: summarise_every given with the sample cadence unset
    # is exactly the condition that triggers the /10 derivation.
    [{**SUMMARY_ONLY_TIMED, "sample_summaries_every": None}],
    indirect=True,
)
def test_sample_summaries_auto_derived(single_integrator_run):
    """sample_summaries_every = summarise_every / 10 when not provided."""
    run = single_integrator_run
    expected = float(run.summarise_every) / 10.0
    assert run.sample_summaries_every == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "solver_settings_override",
    [STATE_OBS_NO_TIMING],
    indirect=True,
)
def test_save_regularly_and_summarise_regularly(single_integrator_run):
    """save_regularly and summarise_regularly booleans on loop compile_settings."""
    run = single_integrator_run
    loop_cfg = run._loop.compile_settings
    has_save = run._loop.save_every is not None
    has_summ = run._loop.summarise_every is not None
    assert loop_cfg.save_regularly == (
        has_save and run.time_domain_outputs_requested
    )
    assert loop_cfg.summarise_regularly == (
        has_summ and run.summary_outputs_requested
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [STATE_OBS_NO_TIMING],
    indirect=True,
)
def test_no_summary_timing_when_no_summary_outputs(single_integrator_run):
    """summarise_every and sample_summaries_every forced None when no
    summary outputs requested."""
    loop_cfg = single_integrator_run._loop.compile_settings
    assert loop_cfg._summarise_every is None
    assert loop_cfg._sample_summaries_every is None


# ── set_summary_timing_from_duration ────────────────────────────────────── #

@pytest.mark.parametrize(
    "solver_settings_override",
    [SUMMARY_ONLY_TIMED],
    indirect=True,
)
def test_set_summary_timing_noop_when_not_dependent(
    single_integrator_run_mutable,
):
    """Explicit timing means set_summary_timing_from_duration is a no-op."""
    run = single_integrator_run_mutable
    initial = run.sample_summaries_every
    assert initial == pytest.approx(0.05)
    run.set_summary_timing_from_duration(duration=1.0)
    assert run.sample_summaries_every == pytest.approx(0.05)


@pytest.mark.parametrize(
    "solver_settings_override",
    [SUMMARY_ONLY_NO_TIMING],
    indirect=True,
)
def test_set_summary_timing_from_duration_dependent(
    single_integrator_run_mutable,
):
    """Duration-dependent path sets summarise_every = duration."""
    run = single_integrator_run_mutable
    assert run.is_duration_dependent is True
    run.set_summary_timing_from_duration(duration=1.0)
    assert run.summarise_every == pytest.approx(1.0, rel=1e-5)
    assert run.sample_summaries_every == pytest.approx(0.01, rel=1e-5)


# ── n_error property ───────────────────────────────────────────────────── #

@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["erk"]],
    indirect=True,
)
def test_n_error_adaptive(single_integrator_run, system):
    """n_error equals system states when algorithm is adaptive."""
    assert single_integrator_run.n_error == system.sizes.states


def test_n_error_fixed(single_integrator_run):
    """n_error is 0 for non-adaptive (euler) algorithm."""
    assert single_integrator_run.n_error == 0


# ── check_compatibility ─────────────────────────────────────────────────── #

def test_errorless_euler_with_adaptive_warns_and_replaces(system):
    """Errorless Euler + adaptive PID warns and replaces with fixed."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        core = SingleIntegratorRunCore(
            system=system,
            algorithm_settings={"algorithm": "euler"},
            step_control_settings={
                "step_controller": "pid",
                "dt_min": 1e-6,
                "dt_max": 1e-1,
            },
        )
        compat = [x for x in w if "cannot be used with" in str(x.message)]
        assert len(compat) >= 1
        assert issubclass(compat[0].category, UserWarning)
        msg = str(compat[0].message).lower()
        assert "euler" in msg
        assert "pid" in msg
        assert "fixed" in msg
        assert "error estimate" in msg
        assert not core._step_controller.is_adaptive


def test_replacement_controller_uses_original_dt(system):
    """Replacement fixed controller uses dt from original adaptive."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        core = SingleIntegratorRunCore(
            system=system,
            algorithm_settings={"algorithm": "euler"},
            step_control_settings={
                "step_controller": "pid",
                "dt_min": 1e-6,
                "dt_max": 1e-1,
            },
        )
        # The fixed replacement should use dt computed from user's bounds
        # dt = sqrt(dt_min * dt_max) = sqrt(1e-6 * 1e-1) = sqrt(1e-7)
        expected_dt = pytest.approx(
            (1e-6 * 1e-1) ** 0.5, rel=1e-3
        )
        assert core._step_controller.dt == expected_dt


def test_errorless_rk4_with_adaptive_warns(system):
    """Errorless RK4 tableau + adaptive PID warns and replaces."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        core = SingleIntegratorRunCore(
            system=system,
            algorithm_settings={
                "algorithm": "erk",
                "tableau": CLASSICAL_RK4_TABLEAU,
            },
            step_control_settings={
                "step_controller": "pid",
                "dt_min": 1e-6,
                "dt_max": 1e-1,
            },
        )
        compat = [x for x in w if "cannot be used with" in str(x.message)]
        assert len(compat) >= 1
        assert not core._step_controller.is_adaptive


def test_adaptive_algo_with_adaptive_controller_no_warning(system):
    """Adaptive Dormand-Prince + PID succeeds without warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        core = SingleIntegratorRunCore(
            system=system,
            algorithm_settings={
                "algorithm": "erk",
                "tableau": DORMAND_PRINCE_54_TABLEAU,
            },
            step_control_settings={
                "step_controller": "pid",
                "dt_min": 1e-6,
                "dt_max": 1e-1,
            },
        )
        compat = [x for x in w if "cannot be used with" in str(x.message)]
        assert len(compat) == 0
        assert core._algo_step.is_adaptive
        assert core._step_controller.is_adaptive


def test_errorless_euler_with_fixed_no_warning(system):
    """Errorless Euler + fixed controller succeeds without warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        core = SingleIntegratorRunCore(
            system=system,
            algorithm_settings={"algorithm": "euler"},
            step_control_settings={
                "step_controller": "fixed",
                "dt": 1e-3,
            },
        )
        compat = [x for x in w if "cannot be used with" in str(x.message)]
        assert len(compat) == 0
        assert not core._algo_step.is_adaptive
        assert not core._step_controller.is_adaptive


# ── update ──────────────────────────────────────────────────────────────── #

def test_update_routes_to_children(
    single_integrator_run_mutable,
    solver_settings,
    system,
    tolerance,
    precision,
):
    """All components receive updates and report new configuration."""
    run = single_integrator_run_mutable
    new_dt = solver_settings["dt_min"] * 0.5
    new_saved_states = [0]
    new_saved_observables = [0]
    new_constant = system.constants.values_array[0] * 1.2

    updates = {
        "dt": new_dt,
        "output_types": ["state", "observables", "mean"],
        "saved_state_indices": new_saved_states,
        "saved_observable_indices": new_saved_observables,
        "summarised_state_indices": new_saved_states,
        "summarised_observable_indices": new_saved_observables,
        "c0": new_constant,
    }

    recognized = run.update(updates)
    expected_keys = {
        "dt",
        "saved_state_indices",
        "saved_observable_indices",
        "summarised_state_indices",
        "summarised_observable_indices",
        "c0",
    }
    assert expected_keys.issubset(recognized)
    assert run.cache_valid is False

    # Controller received dt update
    assert run.dt == pytest.approx(
        new_dt, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert run.dt_min == pytest.approx(
        new_dt, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert run.dt_max == pytest.approx(
        new_dt, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )

    # Output functions received index updates
    flags = run.output_compile_flags
    expected_saved_states = (
        np.asarray(new_saved_states)
        if flags.save_state
        else np.empty(0, dtype=np.int64)
    )
    expected_saved_obs = (
        np.asarray(new_saved_observables)
        if flags.save_observables
        else np.empty(0, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        run.saved_state_indices, expected_saved_states
    )
    np.testing.assert_array_equal(
        run.saved_observable_indices, expected_saved_obs
    )

    # System received constant update
    assert float(system.constants.values_array[0]) == pytest.approx(
        new_constant,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )


def test_update_empty_dict_noop(single_integrator_run_mutable):
    """Empty updates dict returns empty set immediately."""
    result = single_integrator_run_mutable.update({})
    assert result == set()


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["erk"]],
    indirect=True,
)
def test_algorithm_hot_swap_preserves_controller_buffers(
    single_integrator_run_mutable,
):
    """An algorithm swap keeps the controller's buffer registrations."""
    run = single_integrator_run_mutable
    run.update({"algorithm": "bogacki-shampine-32"})
    assert run.device_function is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["erk"]],
    indirect=True,
)
def test_controller_hot_swap_preserves_algorithm_buffers(
    single_integrator_run_mutable,
):
    """A controller swap keeps the algorithm's buffer registrations."""
    run = single_integrator_run_mutable
    run.update({"step_controller": "gustafsson"})
    assert run.device_function is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["backwards_euler"]],
    indirect=True,
)
def test_implicit_algorithm_hot_swap_clears_solver_chain(
    single_integrator_run_mutable,
):
    """Swapping away from an implicit step drops its solver chain groups."""
    from cubie.buffer_registry import buffer_registry

    run = single_integrator_run_mutable
    old_step = run._algo_step
    old_solver = old_step.solver
    old_linear_solver = old_solver.linear_solver

    run.update({"algorithm": "crank_nicolson"})
    assert run.device_function is not None
    assert old_step not in buffer_registry._groups
    assert old_solver not in buffer_registry._groups
    assert old_linear_solver not in buffer_registry._groups


def test_update_unrecognised_raises(single_integrator_run_mutable):
    """Unrecognised keys raise KeyError when silent=False."""
    with pytest.raises(KeyError, match="Unrecognized"):
        single_integrator_run_mutable.update(
            {"nonexistent_param_xyz": 42}, silent=False
        )


def test_update_unrecognised_silent(single_integrator_run_mutable):
    """Unrecognised keys do not raise when silent=True."""
    result = single_integrator_run_mutable.update(
        {"nonexistent_param_xyz": 42}, silent=True
    )
    assert "nonexistent_param_xyz" not in result


def test_update_kwargs_merged(single_integrator_run_mutable, tolerance):
    """Keyword arguments are merged into updates_dict."""
    run = single_integrator_run_mutable
    new_dt = 0.002
    recognized = run.update(dt=new_dt)
    assert "dt" in recognized
    assert run.dt == pytest.approx(new_dt, rel=tolerance.rel_tight)


def test_update_nested_dict_flattened(single_integrator_run_mutable):
    """Nested dicts are flattened and their wrapper keys returned."""
    run = single_integrator_run_mutable
    recognized = run.update({
        "step_controller_settings": {"dt": 0.003},
    })
    assert "step_controller_settings" in recognized
    assert run.dt == pytest.approx(0.003, rel=1e-3)


def test_update_switch_algorithm(single_integrator_run_mutable):
    """Updating algorithm swaps the algo step and updates compile_settings."""
    run = single_integrator_run_mutable
    original_algo = run.algorithm
    new_algo = "rk4" if "euler" in original_algo else "euler"
    recognized = run.update({"algorithm": new_algo})
    assert "algorithm" in recognized
    assert new_algo in run.algorithm
    assert run.compile_settings.algorithm == run.algorithm
    # Algorithm defaults should have been applied to controller
    assert run._step_controller is not None
    assert run.cache_valid is False


def test_update_switch_controller(single_integrator_run_mutable):
    """Updating step_controller swaps the controller."""
    run = single_integrator_run_mutable
    # Switch to a known adaptive algorithm first so PID is valid
    run.update({"algorithm": "bogacki-shampine-32"})
    recognized = run.update({"step_controller": "pid"})
    assert "step_controller" in recognized
    assert run.step_controller == "pid"
    assert run.compile_settings.step_controller == "pid"
    assert run._step_controller.is_adaptive is True
    assert run.cache_valid is False


def test_update_switch_algorithm_carries_old_settings(
    single_integrator_run_mutable,
):
    """Switching algorithm preserves settings from the old algo step."""
    run = single_integrator_run_mutable
    # Get original n from algo_step
    original_n = run._algo_step.n
    run.update({"algorithm": "rk4"})
    assert run._algo_step.n == original_n


def test_update_switch_controller_carries_old_settings(
    single_integrator_run_mutable,
    tolerance,
):
    """Switching controller preserves settings from the old controller."""
    run = single_integrator_run_mutable
    # Switch to adaptive algo so PID is valid
    run.update({"algorithm": "bogacki-shampine-32"})
    original_n = run._step_controller.n
    run.update({"step_controller": "pid"})
    assert run._step_controller.n == original_n


def test_update_check_compatibility_after_switch(
    single_integrator_run_mutable,
):
    """Switching to incompatible combo auto-corrects via check_compatibility."""
    run = single_integrator_run_mutable
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run.update({
            "algorithm": "euler",
            "step_controller": "pid",
        })
        compat = [x for x in w if "cannot be used with" in str(x.message)]
        assert len(compat) >= 1
        assert not run._step_controller.is_adaptive


def test_update_process_loop_timing_called(
    single_integrator_run_mutable,
):
    """Update with timing params routes through _process_loop_timing."""
    run = single_integrator_run_mutable
    run.update({
        "output_types": ["state"],
        "save_every": 0.05,
        "summarise_every": None,
        "sample_summaries_every": None,
    })
    assert run.save_every == pytest.approx(0.05, rel=1e-3)
    loop_cfg = run._loop.compile_settings
    assert loop_cfg._summarise_every is None


# ── Computed properties ─────────────────────────────────────────────────── #

def test_time_domain_outputs_requested(single_integrator_run):
    """time_domain_outputs_requested reflects output_functions."""
    run = single_integrator_run
    assert (
        run.time_domain_outputs_requested
        == run._output_functions.has_time_domain_outputs
    )


def test_summary_outputs_requested(single_integrator_run):
    """summary_outputs_requested reflects output_functions."""
    run = single_integrator_run
    assert (
        run.summary_outputs_requested
        == run._output_functions.has_summary_outputs
    )


def test_has_time_domain_outputs_with_save_every(single_integrator_run):
    """has_time_domain_outputs True with default settings (state + save_every)."""
    assert single_integrator_run.has_time_domain_outputs is True


@pytest.mark.parametrize(
    "solver_settings_override",
    [STATE_OBS_NO_TIMING],
    indirect=True,
)
def test_has_time_domain_outputs_save_last(single_integrator_run):
    """has_time_domain_outputs True when save_last set (no save_every)."""
    assert single_integrator_run.save_last is True
    assert single_integrator_run.has_time_domain_outputs is True


@pytest.mark.parametrize(
    "solver_settings_override",
    [SUMMARY_ONLY_TIMED],
    indirect=True,
)
def test_has_time_domain_outputs_false_no_types(single_integrator_run):
    """has_time_domain_outputs False when no time-domain output types."""
    assert single_integrator_run.has_time_domain_outputs is False


def test_has_summary_outputs_with_timing(single_integrator_run):
    """has_summary_outputs True with default settings (mean + summarise_every)."""
    assert single_integrator_run.has_summary_outputs is True


def test_has_time_domain_outputs_false_no_types_with_timing(
    single_integrator_run_mutable,
):
    """has_time_domain_outputs False when timing set but no types."""
    run = single_integrator_run_mutable
    run.update({
        "output_types": ["mean"],
        "save_every": 0.05,
        "summarise_every": 0.1,
        "sample_summaries_every": 0.01,
    })
    assert run.has_time_domain_outputs is False


@pytest.mark.parametrize(
    "solver_settings_override",
    [SUMMARY_ONLY_NO_TIMING],
    indirect=True,
)
def test_has_summary_outputs_false_no_timing(single_integrator_run):
    """has_summary_outputs False when summary types but no summarise_every."""
    assert single_integrator_run.has_summary_outputs is False


@pytest.mark.parametrize(
    "solver_settings_override",
    [STATE_OBS_NO_TIMING],
    indirect=True,
)
def test_has_summary_outputs_false_no_types(single_integrator_run):
    """has_summary_outputs False when no summary types requested."""
    assert single_integrator_run.has_summary_outputs is False


# ── instantiate_loop ───────────────────────────────────────────────────── #

def test_loop_n_states_matches_system(single_integrator_run, system):
    """Loop receives n_states from system via instantiate_loop."""
    loop_cfg = single_integrator_run._loop.compile_settings
    assert loop_cfg.n_states == system.sizes.states


def test_loop_n_observables_matches_system(single_integrator_run, system):
    """Loop receives n_observables from system via instantiate_loop."""
    loop_cfg = single_integrator_run._loop.compile_settings
    assert loop_cfg.n_observables == system.sizes.observables


def test_loop_n_parameters_matches_system(single_integrator_run, system):
    """Loop receives n_parameters from system via instantiate_loop."""
    loop_cfg = single_integrator_run._loop.compile_settings
    assert loop_cfg.n_parameters == system.sizes.parameters


def test_loop_n_error_matches_core(single_integrator_run):
    """Loop receives n_error from core.n_error via instantiate_loop."""
    run = single_integrator_run
    assert run._loop.compile_settings.n_error == run.n_error


def test_loop_n_counters_zero_without_counters(single_integrator_run):
    """n_counters = 0 when iteration_counters not in output_types."""
    run = single_integrator_run
    assert "iteration_counters" not in run._output_functions.output_types
    assert run._loop.compile_settings.n_counters == 0


@pytest.mark.parametrize(
    # Any chain that requests iteration_counters serves this test.
    "solver_settings_override",
    [DEVICE_SOLVE_SETTINGS],
    indirect=True,
)
def test_loop_n_counters_four_with_counters(single_integrator_run):
    """n_counters = 4 when iteration_counters in output_types."""
    assert single_integrator_run._loop.compile_settings.n_counters == 4


def test_loop_compile_flags_from_output_functions(single_integrator_run):
    """Loop compile_flags come from output_functions."""
    run = single_integrator_run
    assert (run._loop.compile_settings.compile_flags
            == run._output_functions.compile_flags)


# ── build ──────────────────────────────────────────────────────────────── #

def test_device_function_callable(single_integrator_run):
    """device_function returns a callable (triggers build)."""
    assert callable(single_integrator_run.device_function)


def test_build_returns_cache_with_loop_function(single_integrator_run):
    """The built cache wraps the loop's device_function."""
    run = single_integrator_run
    _ = run.device_function  # trigger build
    cache = run._cache
    assert hasattr(cache, "single_integrator_function")
    assert callable(cache.single_integrator_function)


def test_build_compiled_functions_reach_loop(single_integrator_run):
    """After build, loop has output/controller/algo step functions."""
    run = single_integrator_run
    _ = run.device_function  # trigger build
    loop = run._loop
    assert loop.save_state_fn is run._output_functions.save_state_func
    assert loop.update_summaries_fn is run._output_functions.update_summaries_func
    assert loop.save_summaries_fn is run._output_functions.save_summary_metrics_func


# ── duration_dependent warning (Solver level) ─────────────────────────── #

@pytest.mark.parametrize(
    "solver_settings_override",
    [SUMMARY_ONLY_NO_TIMING],
    indirect=True,
)
def test_duration_dependent_warning_on_solve(
    solver, solver_settings, batch_input_arrays, driver_settings,
):
    """Solver emits warning when is_duration_dependent is True."""
    duration = float(solver_settings["duration"])
    initial_values, parameters = batch_input_arrays
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver.solve(
            initial_values=initial_values,
            parameters=parameters,
            drivers=driver_settings,
            duration=duration,
        )
        timing_warns = [
            x for x in w
            if "sample_summaries_every" in str(x.message).lower()
            or "duration" in str(x.message).lower()
        ]
        assert len(timing_warns) >= 1


# ── no-op selector updates keep buffer registration ───────────────────── #

def test_update_same_selectors_still_builds(single_integrator_run_mutable):
    """Re-supplying the current controller and algorithm names builds."""
    run = single_integrator_run_mutable
    run.update({
        "step_controller": run.compile_settings.step_controller,
        "algorithm": run.compile_settings.algorithm,
    })
    assert run.device_function is not None


def test_update_controller_swap_builds(single_integrator_run_mutable):
    """A genuine controller swap reconstructs and builds."""
    run = single_integrator_run_mutable
    target = "i" if run.compile_settings.step_controller != "i" else "pi"
    run.update({"step_controller": target})
    assert run.compile_settings.step_controller == target
    assert run.device_function is not None


# ── Inner-solver tolerance defaults ─────────────────────────────────── #





@pytest.mark.parametrize(
    "solver_settings_override", [CN_ADAPTIVE_KRYLOV_GIVEN], indirect=True
)
def test_explicit_inner_tolerance_survives_derivation(
    single_integrator_run,
):
    """An explicit inner tolerance survives; unset ones are derived."""
    run = single_integrator_run
    algo = run._algo_step
    controller = run._step_controller
    assert controller.is_adaptive
    assert algo.is_implicit
    assert not algo.is_linear

    assert np.allclose(algo.krylov_atol, 3e-5)
    # The unset linear weight derives the controller's tolerance
    # directly, placing the weighted floor at the step tolerance
    # envelope.
    assert np.allclose(
        np.asarray(algo.krylov_rtol), np.asarray(controller.rtol)
    )
    # Derived Newton rtol caps at max(controller rtol, 4-ULP floor).
    assert not np.allclose(algo.newton_atol, 1e-6)
    assert np.all(
        np.asarray(algo.newton_atol) <= np.asarray(controller.atol)
    )
    newton_rtol_floor = 4.0 * np.finfo(run.precision).eps
    assert np.all(
        np.asarray(algo.newton_rtol)
        <= np.maximum(np.asarray(controller.rtol), newton_rtol_floor)
    )
    # Newton-owned linear solves retain the controller's rtol directly.
    expected_reduction = run.precision(
        float(np.min(np.asarray(controller.rtol)))
    )
    assert algo.krylov_residual_reduction == expected_reduction
    assert np.isclose(
        float(algo.krylov_residual_floor),
        float(np.finfo(run.precision).eps) ** 0.5,
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [RODAS3P_ADAPTIVE_KRYLOV_DEFAULT],
    indirect=True,
)
def test_linear_step_reduction_defaults_to_rtol_over_100(
    single_integrator_run,
):
    """A linearly-implicit step defaults to one percent of rtol."""
    run = single_integrator_run
    algo = run._algo_step
    controller = run._step_controller
    assert controller.is_adaptive
    assert algo.is_implicit
    assert algo.is_linear

    controller_rtol_floor = float(
        np.min(np.asarray(controller.rtol))
    )
    expected_reduction = run.precision(
        0.01 * controller_rtol_floor
    )
    assert algo.krylov_residual_reduction == expected_reduction


@pytest.mark.parametrize(
    "solver_settings_override",
    [RODAS3P_ADAPTIVE_KRYLOV_GIVEN],
    indirect=True,
)
def test_linear_step_reduction_override_is_preserved(
    single_integrator_run,
):
    """An explicit reduction on a linearly-implicit step is kept."""
    run = single_integrator_run
    algo = run._algo_step
    assert algo.is_linear
    assert algo.krylov_residual_reduction == run.precision(0.03125)
