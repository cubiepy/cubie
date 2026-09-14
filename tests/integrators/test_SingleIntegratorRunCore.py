"""Tests for cubie.integrators.SingleIntegratorRunCore."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from attrs import evolve, fields_dict

from cubie.integrators.algorithms import DIRK_TABLEAU_REGISTRY
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DORMAND_PRINCE_54_TABLEAU,
)
from cubie.integrators.SingleIntegratorRunCore import SingleIntegratorRunCore
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie.integrators.step_control.adaptive_I_controller import (
    IStepControlConfig,
)
from cubie.integrators.step_control.adaptive_PI_controller import (
    PIStepControlConfig,
)
from cubie.integrators.step_control.adaptive_PID_controller import (
    PIDStepControlConfig,
)
from tests._utils import (
    ALGORITHM_CHAIN_SETS,
    DEVICE_SOLVE_SETTINGS,
    SPECIFIC_ALGORITHM_COMBOS,
    STATE_OBS_NO_TIMING,
    SUMMARY_ONLY_TIMED,
    TORN_NO_OBSERVABLES,
    UNSET_LINEAR_SOLVE,
    _get_evaluate_driver_at_t,
)
from tests._utils import (
    CN_ADAPTIVE_KRYLOV_GIVEN,
    FIRK_PER_STATE_TOLERANCES,
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


def _declared_gain(config_class, gain):
    """Return a controller config's own declared default for a gain."""
    return fields_dict(config_class)[f"_{gain}"].default


def _variant_probe_tableau():
    """Return a DIRK tableau declaring test-owned solver defaults."""
    return evolve(
        DIRK_TABLEAU_REGISTRY["kvaerno3"],
        defaults={
            "linear_correction_type": "minimal_residual",
            "inexact_newton": True,
        },
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["backwards_euler"]],
    indirect=True,
)
def test_constant_change_replaces_the_step_device_functions(
    solver_mutable,
):
    """A constant pushed through the Solver replaces every device function."""
    run = solver_mutable.kernel.single_integrator
    step = run._algo_step
    loop_fn = run.device_function
    dxdt_fn = step.compile_settings.dxdt_fn
    residual_fn = step.solver.compile_settings.residual_fn
    step_fn = step.step_fn
    solver_mutable.update(system_constants={"c0": 0.75})
    assert step.compile_settings.dxdt_fn is not dxdt_fn
    assert step.solver.compile_settings.residual_fn is not residual_fn
    assert step.step_fn is not step_fn
    assert run.device_function is not loop_fn


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


def _build_run(system, driver_array, effective_settings, **overrides):
    """Return a run built from the effective settings plus ``overrides``."""
    return SingleIntegratorRun(
        system,
        drivers_fn=_get_evaluate_driver_at_t(driver_array),
        **{**effective_settings, **overrides},
    )


def test_controller_takes_its_own_gains(
    system, driver_array, effective_settings
):
    """A named controller builds with its own gains; a given gain wins."""
    run = _build_run(
        system,
        driver_array,
        effective_settings,
        algorithm="bogacki-shampine-32",
        step_controller="pi",
    )
    pi_integral = _declared_gain(PIStepControlConfig, "integral_gain")
    pi_proportional = _declared_gain(
        PIStepControlConfig, "proportional_gain"
    )
    assert run.step_controller == "pi"
    assert run._step_controller.integral_gain == pytest.approx(pi_integral)
    assert run._step_controller.proportional_gain == pytest.approx(
        pi_proportional
    )

    explicit = _build_run(
        system,
        driver_array,
        effective_settings,
        algorithm="bogacki-shampine-32",
        step_controller="pi",
        integral_gain=0.9,
    )
    assert explicit._step_controller.integral_gain == pytest.approx(0.9)
    assert explicit._step_controller.proportional_gain == pytest.approx(
        pi_proportional
    )


def test_update_named_controller_is_not_promoted(
    single_integrator_run_mutable,
):
    """A named controller keeps its class despite foreign gains."""
    run = single_integrator_run_mutable
    run.update({"algorithm": "bogacki-shampine-32"})
    run.update({"step_controller": "i", "proportional_gain": 0.4})
    assert run.step_controller == "i"


def test_precision_follows_the_system(
    system, driver_array, effective_settings
):
    """A precision in the settings is ignored; the system's is used."""
    wrong_precision = (
        np.float64 if system.precision == np.float32 else np.float32
    )
    run = _build_run(
        system, driver_array, effective_settings, precision=wrong_precision
    )
    assert run.precision == system.precision
    assert run._output_functions.compile_settings.precision == system.precision
    assert run._algo_step.precision == system.precision


def test_dt_reaches_the_controller(system, driver_array, effective_settings):
    """A given dt flows through to the controller."""
    run = _build_run(system, driver_array, effective_settings, dt=0.005)
    assert run.dt == pytest.approx(0.005, rel=1e-3)


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
    """save_regularly and summarise_regularly booleans on loop
    compile_settings.
    """
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


# ── summary_window ──────────────────────────────────────────────────────── #

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


@pytest.mark.parametrize(
    "solver_settings_override",
    [SPECIFIC_ALGORITHM_COMBOS["dirk-kvaerno5-fixed"]],
    indirect=True,
)
def test_n_error_fixed_controller_on_embedded_tableau(
    single_integrator_run,
):
    """A fixed controller drops the error estimate of an embedded tableau."""
    run = single_integrator_run
    step = run._algo_step
    assert step.has_error_estimate
    assert not run._step_controller.is_adaptive
    assert step.is_adaptive is False
    assert step.compile_settings.is_adaptive is False
    assert step.uses_error is False
    assert all(weight == 0.0 for weight in step.error_weights)
    assert len(step.error_weights) == step.tableau.stage_count
    assert run.n_error == 0
    assert run._loop.compile_settings.n_error == 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [SPECIFIC_ALGORITHM_COMBOS["dirk-kvaerno5-fixed"]],
    indirect=True,
)
def test_uses_error_follows_controller_swap(
    single_integrator_run_mutable, system
):
    """Swapping to an adaptive controller restores the error buffer."""
    run = single_integrator_run_mutable
    step = run._algo_step
    run.update({"step_controller": "pid"})
    assert run._step_controller.is_adaptive
    assert step.is_adaptive is True
    assert step.uses_error is True
    tableau_weights = step.tableau.error_weights(
        step.compile_settings.precision
    )
    assert tuple(step.error_weights) == tuple(tableau_weights)
    assert run.n_error == system.sizes.states
    assert run._loop.compile_settings.n_error == system.sizes.states
    run.update({"step_controller": "fixed"})
    assert step.is_adaptive is False
    assert step.uses_error is False
    assert all(weight == 0.0 for weight in step.error_weights)
    assert run.n_error == 0
    assert run._loop.compile_settings.n_error == 0


# ── check_compatibility ─────────────────────────────────────────────────── #

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

    updates = {
        "dt": new_dt,
        "output_types": ["state", "observables", "mean"],
        "saved_state_indices": new_saved_states,
        "saved_observable_indices": new_saved_observables,
        "summarised_state_indices": new_saved_states,
        "summarised_observable_indices": new_saved_observables,
    }

    recognized = run.update(updates)
    expected_keys = {
        "dt",
        "saved_state_indices",
        "saved_observable_indices",
        "summarised_state_indices",
        "summarised_observable_indices",
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
    original_n = run._algo_step.n_states
    run.update({"algorithm": "rk4"})
    assert run._algo_step.n_states == original_n


def test_update_switch_controller_carries_old_settings(
    single_integrator_run_mutable,
    tolerance,
):
    """Switching controller preserves settings from the old controller."""
    run = single_integrator_run_mutable
    # Switch to adaptive algo so PID is valid
    run.update({"algorithm": "bogacki-shampine-32"})
    original_n = run._step_controller.n_states
    run.update({"step_controller": "pid"})
    assert run._step_controller.n_states == original_n


def test_update_switch_controller_reverts_gains(
    single_integrator_run_mutable,
):
    """A controller swap reverts gains to the new controller's defaults.

    Explicit gains in the update that orders the swap still apply.
    """
    run = single_integrator_run_mutable
    run.update({"algorithm": "bogacki-shampine-32"})

    run.update({"step_controller": "pi"})
    assert run._step_controller.integral_gain == pytest.approx(
        _declared_gain(PIStepControlConfig, "integral_gain")
    )
    assert run._step_controller.proportional_gain == pytest.approx(
        _declared_gain(PIStepControlConfig, "proportional_gain")
    )

    run.update({"step_controller": "pid", "integral_gain": 0.9})
    assert run._step_controller.integral_gain == pytest.approx(0.9)
    assert run._step_controller.proportional_gain == pytest.approx(
        _declared_gain(PIDStepControlConfig, "proportional_gain")
    )
    assert run._step_controller.derivative_gain == pytest.approx(
        _declared_gain(PIDStepControlConfig, "derivative_gain")
    )


def test_update_algo_swap_with_controller_override_skips_family_gains(
    single_integrator_run_mutable,
):
    """An explicit controller in an algorithm swap keeps its gains."""
    run = single_integrator_run_mutable
    tableau = evolve(
        DIRK_TABLEAU_REGISTRY["kvaerno3"],
        defaults={"step_controller": "pi", "integral_gain": 3.0},
    )
    run.update(
        {
            "algorithm": "dirk",
            "tableau": tableau,
            "step_controller": "i",
        }
    )
    assert run.step_controller == "i"
    assert run._step_controller.integral_gain == pytest.approx(
        _declared_gain(IStepControlConfig, "integral_gain")
    )


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
    """has_time_domain_outputs True with default settings (state + save_every).
    """
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
    """has_summary_outputs True with default settings (mean + summarise_every).
    """
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


def test_update_output_types_adds_counters_to_loop(
    single_integrator_run_mutable,
):
    """Requesting iteration_counters via update sizes the loop row."""
    run = single_integrator_run_mutable
    assert run._loop.compile_settings.n_counters == 0
    run.update({"output_types": ["state", "iteration_counters"]})
    assert run._output_functions.save_counters is True
    assert run._loop.compile_settings.compile_flags.save_counters is True
    assert run._loop.compile_settings.n_counters == 4


@pytest.mark.parametrize(
    # Any chain that requests iteration_counters serves this test.
    "solver_settings_override",
    [DEVICE_SOLVE_SETTINGS],
    indirect=True,
)
def test_update_output_types_drops_counters_from_loop(
    single_integrator_run_mutable,
):
    """Dropping iteration_counters via update collapses the loop row."""
    run = single_integrator_run_mutable
    assert run._loop.compile_settings.n_counters == 4
    run.update({"output_types": ["state", "time"]})
    assert run._output_functions.save_counters is False
    assert run._loop.compile_settings.compile_flags.save_counters is False
    assert run._loop.compile_settings.n_counters == 0


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
    assert hasattr(cache, "loop_fn")
    assert callable(cache.loop_fn)


def test_build_compiled_functions_reach_loop(single_integrator_run):
    """After build, loop has output/controller/algo step functions."""
    run = single_integrator_run
    _ = run.device_function  # trigger build
    loop = run._loop
    assert loop.save_state_fn is run._output_functions.save_state_fn
    output_functions = run._output_functions
    assert loop.update_summaries_fn is (
        output_functions.update_summaries_fn
    )
    assert loop.save_summaries_fn is (
        output_functions.save_summaries_fn
    )


# ── duration_dependent warning (Solver level) ─────────────────────────── #

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
    run.update({"algorithm": "bogacki-shampine-32", "step_controller": target})
    assert run.compile_settings.step_controller == target
    assert run.device_function is not None


# ── Inner-solver tolerance defaults ─────────────────────────────────── #


@pytest.mark.parametrize(
    "solver_settings_override", [FIRK_PER_STATE_TOLERANCES], indirect=True
)
def test_per_state_tolerances_reach_coupled_firk_norms(
    single_integrator_run, system
):
    """A per-state tolerance vector reaches the coupled FIRK norms."""
    run = single_integrator_run
    algo = run._algo_step
    controller = run._step_controller
    n = system.sizes.states

    assert algo.is_implicit
    assert controller.atol.shape == (n,)
    assert np.asarray(algo.krylov_atol).shape == (n,)
    assert np.asarray(algo.newton_atol).shape == (n,)
    assert np.allclose(algo.krylov_atol, controller.atol)
    assert np.allclose(algo.newton_atol, controller.atol / 10.0)

    # The coupled solve is wider than the physical state, and the
    # norms keep their tolerances at the physical length.
    newton_norm = algo.solver.norm
    krylov_norm = algo.solver.linear_solver.norm
    assert krylov_norm.solver_width > n
    for norm in (newton_norm, krylov_norm):
        assert norm.compile_settings.n_states == n
        assert norm.compile_settings.tol_length == n
        assert norm.atol.shape == (n,)
        assert norm.rtol.shape == (n,)

    assert run.device_function is not None


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


@pytest.mark.parametrize(
    "solver_settings_override",
    [{
        **ALGORITHM_CHAIN_SETS["dirk"],
        **UNSET_LINEAR_SOLVE,
        "lu_factor_location": "shared",
    }],
    indirect=True,
)
def test_constructor_linear_kwargs_survive_defaults_swap(
    single_integrator_run,
):
    """Constructor kwargs land on the defaults-selected solver class."""
    step = single_integrator_run._algo_step
    assert step.linear_correction_type == "lu"
    linear = step.solver.linear_solver
    assert linear.compile_settings.lu_factor_location == "shared"


# ── Mass flags reach the step controller ───────────────────────────────── #


TORN_DIRK_ADAPTIVE = {
    "system_type": "torn_driver",
    "algorithm": "l_stable_dirk_3",
    "step_controller": "pi",
    **TORN_NO_OBSERVABLES,
    **UNSET_LINEAR_SOLVE,
}


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_DIRK_ADAPTIVE], indirect=True
)
def test_controller_mass_flags_follow_system(
    single_integrator_run_mutable, system
):
    """Controller mass flags match the system's across a swap."""
    run = single_integrator_run_mutable
    assert system.mass_diagonal_flags == (True, False)
    assert run._step_controller.mass_flags == (True, False)
    run.update({"step_controller": "pid"})
    assert run._step_controller.mass_flags == (True, False)
    run.update({"step_controller": "fixed"})
    assert run._step_controller.mass_flags == (True, False)
