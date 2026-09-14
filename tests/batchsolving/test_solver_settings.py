"""Tests for the solver-level settings record and its resolution."""

import warnings

import numpy as np
import pytest
from attrs import evolve

from cubie.batchsolving.solver_settings import (
    SolverSettings,
    resolve,
    resolve_inner_tolerances,
    resolve_loop_timing,
    resolve_step_bounds,
)
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.integrators.algorithms import DIRK_TABLEAU_REGISTRY
from tests._utils import (
    LARGE_DIRK,
    SUMMARY_ONLY_NO_TIMING,
    TORN_NO_OBSERVABLES,
    _build_solver_instance,
)


def _effective(system, **given):
    """Resolve ``given`` for ``system`` and return the effective record."""
    settings = SolverSettings.from_kwargs(**given)
    return resolve(settings, system, SystemInterface(system)).effective


def _notices(system, **given):
    settings = SolverSettings.from_kwargs(**given)
    return resolve(settings, system, SystemInterface(system)).notices


# ── The record ──────────────────────────────────────────────────────── #


def test_grouped_dicts_flatten_and_unknown_names_raise():
    """Grouped dicts flatten into the record; unknown names raise."""
    given = SolverSettings.from_kwargs(
        step_control_settings={"dt": 0.01}, algorithm="euler"
    )
    assert given.dt == 0.01
    assert given.algorithm == "euler"
    with pytest.raises(KeyError, match="Unrecognized"):
        SolverSettings.from_kwargs(not_a_setting=1)
    with pytest.raises(KeyError, match="dt_save"):
        SolverSettings.from_kwargs(dt_save=0.1)


def test_derived_names_are_not_settings():
    """A name only resolution sets is not a provided setting."""
    with pytest.raises(KeyError, match="is_adaptive"):
        SolverSettings.from_kwargs(is_adaptive=True)


def test_update_records_and_none_unsets():
    """``updated`` records new values and ``None`` unsets a setting."""
    given = SolverSettings.from_kwargs(dt=0.01, atol=1e-6)
    later, recognised = given.updated({"dt": None, "rtol": 1e-4})
    assert recognised == {"dt", "rtol"}
    assert later.dt is None
    assert later.rtol == 1e-4
    assert later.atol == 1e-6
    assert given.dt == 0.01


def test_gains_with_a_filter_raise():
    """Gains and a filter cannot both be provided."""
    with pytest.raises(ValueError, match="filter_coefficients"):
        SolverSettings.from_kwargs(
            filter_coefficients="pi42", integral_gain=0.5
        )


# ── Step keys ───────────────────────────────────────────────────────── #


def test_explicit_step_setting_overrides_step_default(system):
    """A provided step key survives the family defaults."""
    effective = _effective(
        system, algorithm="kvaerno3", preconditioner_type="none"
    )
    assert effective.preconditioner_type == "none"


def _variant_probe_tableau():
    """Return a DIRK tableau declaring test-owned solver defaults."""
    return evolve(
        DIRK_TABLEAU_REGISTRY["kvaerno3"],
        defaults={
            "linear_correction_type": "minimal_residual",
            "inexact_newton": True,
        },
    )


def test_tableau_defaults_override_family_defaults(system):
    """A tableau's defaults dict overrides the family default keys."""
    effective = _effective(
        system, algorithm="dirk", tableau=_variant_probe_tableau()
    )
    assert effective.linear_correction_type == "minimal_residual"


def test_matching_solver_choice_keeps_variant_defaults(system):
    """A choice matching the declared linear solver keeps its variants."""
    effective = _effective(
        system,
        algorithm="dirk",
        tableau=_variant_probe_tableau(),
        linear_correction_type="minimal_residual",
    )
    assert effective.inexact_newton is True


def test_different_solver_choice_drops_variant_defaults(system):
    """A choice differing from the declared linear solver drops them."""
    effective = _effective(
        system,
        algorithm="dirk",
        tableau=_variant_probe_tableau(),
        linear_correction_type="bicgstab",
    )
    assert effective.linear_correction_type == "bicgstab"
    assert effective.inexact_newton is None


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "torn_time", **TORN_NO_OBSERVABLES}],
    indirect=True,
)
def test_neumann_on_a_mass_matrix_system_raises(system):
    """A Neumann preconditioner cannot serve a mass-matrix system."""
    with pytest.raises(ValueError, match="Neumann"):
        _effective(
            system, algorithm="backwards_euler", preconditioner_type="neumann"
        )


# ── Controller ──────────────────────────────────────────────────────── #


def test_unnamed_controller_promotes_to_carry_given_gains(system):
    """A provided gain promotes the family controller; family gains drop."""
    effective = _effective(system, algorithm="kvaerno3", derivative_gain=0.05)
    assert effective.step_controller == "pid"
    assert effective.derivative_gain == 0.05
    assert effective.integral_gain is None


def test_named_controller_is_not_promoted(system):
    """A named controller drops the gains it lacks."""
    effective = _effective(
        system, algorithm="kvaerno3", step_controller="i", derivative_gain=0.05
    )
    assert effective.step_controller == "i"
    assert effective.derivative_gain is None


def test_filter_coefficients_replace_the_family_gains(system):
    """A filter preset passes through and the family gains drop."""
    effective = _effective(
        system, algorithm="kvaerno3", filter_coefficients="pi42"
    )
    assert effective.filter_coefficients == "pi42"
    assert effective.integral_gain is None


def test_errorless_algorithm_replaces_an_adaptive_request(system):
    """An adaptive request on an errorless step resolves to fixed."""
    effective = _effective(system, algorithm="euler", step_controller="pid")
    assert effective.step_controller == "fixed"
    assert effective.is_adaptive is False
    notices = _notices(system, algorithm="euler", step_controller="pid")
    assert any("cannot be used with" in notice for notice in notices)


# ── Step bounds and tolerances ──────────────────────────────────────── #


def test_adaptive_bounds_follow_a_lone_dt():
    """A lone dt gives bounds a hundredth and a hundred times it."""
    bounds = resolve_step_bounds(0.01, None, None, True)
    assert bounds == {"dt": 0.01, "dt_min": 1e-4, "dt_max": 1.0}


def test_adaptive_dt_is_the_geometric_mean_of_given_bounds():
    """Bounds alone give dt as their geometric mean."""
    bounds = resolve_step_bounds(None, 1e-4, 1e-2, True)
    assert bounds["dt"] == pytest.approx(1e-3)


def test_fixed_step_from_bounds():
    """A fixed step is the bounds' geometric mean or a lone bound."""
    assert resolve_step_bounds(None, 1e-4, 1e-2, False) == {
        "dt": pytest.approx(1e-3)
    }
    assert resolve_step_bounds(None, 1e-4, None, False) == {"dt": 1e-4}
    assert resolve_step_bounds(None, None, 0.5, False) == {"dt": 0.5}


def test_inner_tolerances_derive_from_the_controller():
    """Newton takes a tenth, Krylov the controller's, reduction min rtol."""
    given = SolverSettings()
    tolerances = resolve_inner_tolerances(
        given, 1e-6, 1e-4, True, False, np.float32
    )
    assert tolerances["newton_atol"] == pytest.approx(1e-7)
    assert tolerances["newton_rtol"] == pytest.approx(1e-5)
    assert tolerances["krylov_atol"] == pytest.approx(1e-6)
    assert tolerances["krylov_residual_reduction"] == pytest.approx(1e-4)


def test_linear_step_reduction_is_a_hundredth_of_rtol():
    """A linearly-implicit step reduces to a hundredth of rtol."""
    tolerances = resolve_inner_tolerances(
        SolverSettings(), 1e-6, 1e-4, True, True, np.float32
    )
    assert tolerances["krylov_residual_reduction"] == pytest.approx(1e-6)


def test_fixed_step_reduction_is_machine_epsilon():
    """Without adaptivity the reduction is the working epsilon."""
    tolerances = resolve_inner_tolerances(
        SolverSettings(), 1e-6, 1e-4, False, False, np.float32
    )
    assert tolerances["krylov_residual_reduction"] == pytest.approx(
        float(np.finfo(np.float32).eps)
    )


def test_given_inner_tolerance_survives_derivation():
    """A provided inner tolerance is kept."""
    tolerances = resolve_inner_tolerances(
        SolverSettings(newton_atol=3e-9), 1e-6, 1e-4, True, False, np.float32
    )
    assert tolerances["newton_atol"] == 3e-9


# ── Timing ──────────────────────────────────────────────────────────── #


def test_save_last_without_save_every():
    """Time-domain outputs without an interval save the last state."""
    timing = resolve_loop_timing(None, 0.1, None, True, False)
    assert timing["save_last"] is True
    assert timing["save_regularly"] is False
    assert timing["summarise_every"] is None
    assert timing["summarise_regularly"] is False


def test_given_window_samples_a_tenth():
    """A provided window without a sample interval samples a tenth of it."""
    timing = resolve_loop_timing(0.02, 0.1, None, True, True)
    assert timing["sample_summaries_every"] == pytest.approx(0.01)
    assert timing["save_regularly"] is True
    assert timing["summarise_regularly"] is True


def test_derived_window_is_the_duration():
    """An unset window takes the duration and a tenth as the sample."""
    timing = resolve_loop_timing(None, None, None, False, True, 2.0)
    assert timing["summarise_every"] == 2.0
    assert timing["sample_summaries_every"] == pytest.approx(0.2)


def test_given_sample_interval_survives_a_derived_window():
    """A provided sample interval is kept under a duration-derived window."""
    timing = resolve_loop_timing(None, None, 0.05, False, True, 2.0)
    assert timing["summarise_every"] == 2.0
    assert timing["sample_summaries_every"] == 0.05


# ── Outputs ─────────────────────────────────────────────────────────── #


def test_output_labels_resolve_to_indices(system):
    """Labels resolve against the system; summaries follow the saved set."""
    name = system.initial_values.names[0]
    effective = _effective(system, save_variables=[name])
    np.testing.assert_array_equal(effective.saved_state_indices, [0])
    np.testing.assert_array_equal(effective.summarised_state_indices, [0])


def test_out_of_range_output_index_raises(system):
    """An index the system does not have is an error."""
    with pytest.raises(ValueError, match="range"):
        _effective(
            system, saved_state_indices=[system.sizes.states + 5]
        )


# ── Solver ──────────────────────────────────────────────────────────── #


def test_settings_dict_is_the_provided_settings(solver, solver_settings):
    """The solver reports what it was provided, not what it derived."""
    settings = solver.settings_dict
    assert settings["algorithm"] == solver_settings["algorithm"]
    assert settings["dt"] == solver_settings["dt"]
    assert solver.effective_settings["save_regularly"] is True


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "kvaerno3", "step_controller": "pid"}],
    indirect=True,
)
def test_given_controller_survives_an_algorithm_change(solver_mutable):
    """A provided controller stays through an algorithm change."""
    solver_mutable.update(algorithm="crank_nicolson")
    run = solver_mutable.kernel.single_integrator
    assert run.step_controller == "pid"
    assert run.algorithm == "crank_nicolson"


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "kvaerno3", "step_controller": "pid"}],
    indirect=True,
)
def test_errorless_algorithm_warns_and_replaces_on_update(solver_mutable):
    """An adaptive controller on an errorless algorithm is replaced."""
    with pytest.warns(UserWarning, match="cannot be used with"):
        solver_mutable.update(algorithm="euler")
    run = solver_mutable.kernel.single_integrator
    assert run.step_controller == "fixed"
    assert run._algo_step.is_adaptive is False
    assert run.n_error == 0
    assert run._loop.compile_settings.n_error == 0
    assert solver_mutable.settings_dict["step_controller"] == "pid"


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "kvaerno3", "step_controller": "pid"}],
    indirect=True,
)
def test_derived_tolerances_follow_a_tolerance_update(solver_mutable):
    """Unset inner tolerances track the controller's on update."""
    solver_mutable.update(atol=1e-5, rtol=1e-3)
    step = solver_mutable.kernel.single_integrator._algo_step
    assert np.allclose(step.newton_atol, 1e-6)
    assert np.allclose(step.newton_rtol, 1e-4)
    assert np.allclose(
        step.krylov_atol, solver_mutable.settings_dict["krylov_atol"]
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "kvaerno3", "step_controller": "pid"}],
    indirect=True,
)
def test_filter_after_gains_raises(solver_mutable):
    """A filter cannot join provided gains without unsetting them."""
    solver_mutable.update(integral_gain=0.5)
    with pytest.raises(ValueError, match="filter_coefficients"):
        solver_mutable.update(filter_coefficients="pi42")
    solver_mutable.update(integral_gain=None, filter_coefficients="pi42")
    assert solver_mutable.settings_dict["filter_coefficients"] == "pi42"


@pytest.mark.parametrize(
    "solver_settings_override", [SUMMARY_ONLY_NO_TIMING], indirect=True
)
def test_summary_window_follows_the_duration(
    solver_mutable, batch_input_arrays, driver_settings
):
    """An unset window takes each solve's duration."""
    initial_values, parameters = batch_input_arrays
    solver_mutable.solve(
        initial_values=initial_values,
        parameters=parameters,
        drivers=driver_settings,
        duration=0.5,
    )
    assert solver_mutable.summarise_every == pytest.approx(0.5)
    assert solver_mutable.sample_summaries_every == pytest.approx(0.05)


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**SUMMARY_ONLY_NO_TIMING, "sample_summaries_every": 0.01}],
    indirect=True,
)
def test_given_sample_interval_is_kept_under_a_derived_window(
    solver_mutable, batch_input_arrays, driver_settings
):
    """A provided sample interval is honoured with a duration window."""
    initial_values, parameters = batch_input_arrays
    solver_mutable.solve(
        initial_values=initial_values,
        parameters=parameters,
        drivers=driver_settings,
        duration=0.5,
    )
    assert solver_mutable.summarise_every == pytest.approx(0.5)
    assert solver_mutable.sample_summaries_every == pytest.approx(0.01)


@pytest.mark.parametrize(
    "solver_settings_override", [SUMMARY_ONLY_NO_TIMING], indirect=True
)
def test_derived_window_warns_at_construction(
    system, solver_settings, driver_settings
):
    """Summaries without a window warn when the solver is built."""
    with pytest.warns(UserWarning, match="summarise_every"):
        built = _build_solver_instance(
            system, solver_settings, driver_settings
        )
    built.close()


def test_duration_with_explicit_timing_keeps_the_build(solver_mutable):
    """A new duration under explicit timing changes nothing below."""
    solver_mutable.kernel.kernel
    assert solver_mutable.kernel._cache_valid
    solver_mutable.update(duration=0.9)
    assert solver_mutable.kernel._cache_valid


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "kvaerno3", "step_controller": "pi"}],
    indirect=True,
)
def test_newton_rtol_inversion_warns(system, solver_settings, driver_settings):
    """A sub-floor controller rtol warns of the Newton inversion."""
    with pytest.warns(UserWarning, match="newton_rtol"):
        built = _build_solver_instance(
            system, {**solver_settings, "rtol": 1e-10}, driver_settings
        )
    built.close()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        built = _build_solver_instance(
            system, {**solver_settings, "rtol": 1e-4}, driver_settings
        )
    built.close()
    assert not [w for w in caught if "newton_rtol" in str(w.message)]


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_DIRK], indirect=True
)
def test_auto_performance_off_keeps_the_derived_values(solver_mutable):
    """Turning auto_performance off leaves the last derived flags."""
    step = solver_mutable.kernel.single_integrator._algo_step
    rolled = step.compile_settings.unroll.unroll_newton_exits
    assert rolled == (True, 1)
    solver_mutable.update(auto_performance=False)
    step = solver_mutable.kernel.single_integrator._algo_step
    assert step.compile_settings.unroll.unroll_newton_exits == rolled


def test_precision_reaches_every_factory(
    system, solver_settings, driver_settings
):
    """A provided precision updates the system and every child."""
    other = np.float64 if system.precision == np.float32 else np.float32
    built = _build_solver_instance(
        system.copy(), solver_settings, driver_settings
    )
    built.update(precision=other)
    run = built.kernel.single_integrator
    assert built.system.precision == other
    assert built.kernel.precision == other
    assert run._algo_step.precision == other
    assert run._step_controller.precision == other
    assert run._loop.precision == other
    assert run._output_functions.precision == other
    built.close()


def test_system_constants_reach_the_system(
    system, solver_settings, driver_settings
):
    """Provided constants are written into the system."""
    name = system.constants.names[0]
    value = float(system.constants.values_dict[name]) * 1.5
    built = _build_solver_instance(
        system.copy(), solver_settings, driver_settings
    )
    built.update(system_constants={name: value})
    assert float(built.system.constants.values_dict[name]) == (
        pytest.approx(value)
    )
    assert built.settings_dict["system_constants"] == {name: value}
    built.close()
