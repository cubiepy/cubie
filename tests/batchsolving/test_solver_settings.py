"""Tests for the solver-level settings resolution."""

import warnings

import numpy as np
import pytest
from attrs import evolve

from cubie.batchsolving.solver_settings import (
    resolve_inner_tolerances,
    resolve_loop_timing,
    resolve_settings,
    resolve_step_bounds,
)
from cubie.integrators.algorithms import DIRK_TABLEAU_REGISTRY
from cubie.integrators.algorithms.generic_dirk import (
    DIRK_ADAPTIVE_DEFAULTS,
    DIRK_SOLVER_DEFAULTS,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    DAE_SOLVER_DEFAULTS,
)
from cubie.integrators.step_control.adaptive_step_controller import (
    DEFAULT_DT_MAX,
    DEFAULT_DT_MIN,
)
from cubie.integrators.step_control.fixed_step_controller import (
    DEFAULT_FIXED_DT,
)
from tests._utils import (
    LARGE_DIRK,
    SUMMARY_ONLY_NO_TIMING,
    TORN_NO_OBSERVABLES,
    _build_solver_instance,
)


# ── Step keys ───────────────────────────────────────────────────────── #


def test_step_defaults_apply_to_unset_step_keys(system):
    """Unset step keys take the family's declared defaults."""
    effective = resolve_settings({"algorithm": "kvaerno3"}, system).effective
    for key, value in DIRK_SOLVER_DEFAULTS.items():
        assert effective[key] == value


def test_explicit_step_setting_overrides_step_default(system):
    """A given step key survives the family defaults."""
    effective = resolve_settings(
        {"algorithm": "kvaerno3", "preconditioner_type": "none"}, system
    ).effective
    assert effective["preconditioner_type"] == "none"


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
    effective = resolve_settings(
        {"algorithm": "dirk", "tableau": _variant_probe_tableau()}, system
    ).effective
    assert effective["linear_correction_type"] == "minimal_residual"


def test_matching_solver_choice_keeps_variant_defaults(system):
    """A choice matching the declared linear solver keeps its variants."""
    effective = resolve_settings(
        {
            "algorithm": "dirk",
            "tableau": _variant_probe_tableau(),
            "linear_correction_type": "minimal_residual",
        },
        system,
    ).effective
    assert effective["inexact_newton"] is True


def test_different_solver_choice_drops_variant_defaults(system):
    """A choice differing from the declared linear solver drops them."""
    effective = resolve_settings(
        {
            "algorithm": "dirk",
            "tableau": _variant_probe_tableau(),
            "linear_correction_type": "bicgstab",
        },
        system,
    ).effective
    assert effective["linear_correction_type"] == "bicgstab"
    assert "inexact_newton" not in effective


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "torn_time", **TORN_NO_OBSERVABLES}],
    indirect=True,
)
def test_mass_matrix_systems_take_the_dae_solver_defaults(system):
    """A mass-matrix system takes the DAE linear solve over the family's."""
    effective = resolve_settings(
        {"algorithm": "backwards_euler"}, system
    ).effective
    for key, value in DAE_SOLVER_DEFAULTS.items():
        assert effective[key] == value
    with pytest.raises(ValueError, match="Neumann"):
        resolve_settings(
            {"algorithm": "backwards_euler", "preconditioner_type": "neumann"},
            system,
        )


# ── Controller ──────────────────────────────────────────────────────── #


def test_default_controller_and_gains_come_from_the_family(system):
    """An unnamed controller is the family's with the family's gains."""
    effective = resolve_settings({"algorithm": "kvaerno3"}, system).effective
    defaults = DIRK_ADAPTIVE_DEFAULTS.settings
    assert effective["step_controller"] == defaults["step_controller"]
    assert effective["integral_gain"] == defaults["integral_gain"]
    assert effective["proportional_gain"] == defaults["proportional_gain"]
    assert effective["is_adaptive"] is True


def test_unnamed_controller_promotes_to_carry_given_gains(system):
    """A given gain promotes the family controller; family gains drop."""
    effective = resolve_settings(
        {"algorithm": "kvaerno3", "derivative_gain": 0.05}, system
    ).effective
    assert effective["step_controller"] == "pid"
    assert effective["derivative_gain"] == 0.05
    assert "integral_gain" not in effective


def test_named_controller_is_not_promoted(system):
    """A named controller drops the gains it lacks."""
    effective = resolve_settings(
        {
            "algorithm": "kvaerno3",
            "step_controller": "i",
            "derivative_gain": 0.05,
        },
        system,
    ).effective
    assert effective["step_controller"] == "i"
    assert "derivative_gain" not in effective


def test_filter_coefficients_replace_the_family_gains(system):
    """A filter preset passes through and the family gains drop."""
    effective = resolve_settings(
        {"algorithm": "kvaerno3", "filter_coefficients": "pi42"}, system
    ).effective
    assert effective["filter_coefficients"] == "pi42"
    assert "integral_gain" not in effective


def test_errorless_algorithm_replaces_an_adaptive_request(system):
    """An adaptive request on an errorless step resolves to fixed."""
    resolved = resolve_settings(
        {"algorithm": "euler", "step_controller": "pid"}, system
    )
    assert resolved.replaced_controller == "pid"
    assert resolved.effective["step_controller"] == "fixed"
    assert resolved.effective["is_adaptive"] is False


# ── Step bounds and tolerances ──────────────────────────────────────── #


def test_adaptive_bounds_follow_a_lone_dt():
    """A lone dt gives bounds a hundredth and a hundred times it."""
    bounds = resolve_step_bounds(0.01, None, None, True)
    assert bounds == {"dt": 0.01, "dt_min": 1e-4, "dt_max": 1.0}


def test_adaptive_dt_is_the_geometric_mean_of_given_bounds():
    """Bounds alone give dt as their geometric mean."""
    bounds = resolve_step_bounds(None, 1e-4, 1e-2, True)
    assert bounds["dt"] == pytest.approx(1e-3)


def test_adaptive_defaults_apply_without_dt_or_bounds():
    """Nothing given takes the adaptive defaults."""
    bounds = resolve_step_bounds(None, None, None, True)
    assert bounds["dt_min"] == DEFAULT_DT_MIN
    assert bounds["dt_max"] == DEFAULT_DT_MAX
    assert bounds["dt"] == pytest.approx(
        np.sqrt(DEFAULT_DT_MIN * DEFAULT_DT_MAX)
    )


def test_fixed_step_from_bounds():
    """A fixed step is the bounds' geometric mean, a lone bound, or 1e-3."""
    assert resolve_step_bounds(None, 1e-4, 1e-2, False) == {
        "dt": pytest.approx(1e-3)
    }
    assert resolve_step_bounds(None, 1e-4, None, False) == {"dt": 1e-4}
    assert resolve_step_bounds(None, None, 0.5, False) == {"dt": 0.5}
    assert resolve_step_bounds(None, None, None, False) == {
        "dt": DEFAULT_FIXED_DT
    }


def test_inner_tolerances_derive_from_the_controller():
    """Newton takes a tenth, Krylov the controller's, reduction min rtol."""
    tolerances = resolve_inner_tolerances(
        {}, 1e-6, 1e-4, True, False, np.float32
    )
    assert tolerances["newton_atol"] == pytest.approx(1e-7)
    assert tolerances["newton_rtol"] == pytest.approx(1e-5)
    assert tolerances["krylov_atol"] == pytest.approx(1e-6)
    assert tolerances["krylov_residual_reduction"] == pytest.approx(1e-4)


def test_linear_step_reduction_is_a_hundredth_of_rtol():
    """A linearly-implicit step reduces to a hundredth of rtol."""
    tolerances = resolve_inner_tolerances(
        {}, 1e-6, 1e-4, True, True, np.float32
    )
    assert tolerances["krylov_residual_reduction"] == pytest.approx(1e-6)


def test_fixed_step_reduction_is_machine_epsilon():
    """Without adaptivity the reduction is the working epsilon."""
    tolerances = resolve_inner_tolerances(
        {}, 1e-6, 1e-4, False, False, np.float32
    )
    assert tolerances["krylov_residual_reduction"] == pytest.approx(
        float(np.finfo(np.float32).eps)
    )


def test_given_inner_tolerance_survives_derivation():
    """A given inner tolerance is kept."""
    tolerances = resolve_inner_tolerances(
        {"newton_atol": 3e-9}, 1e-6, 1e-4, True, False, np.float32
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
    """A given window without a sample interval samples a tenth of it."""
    timing = resolve_loop_timing(0.02, 0.1, None, True, True)
    assert timing["sample_summaries_every"] == pytest.approx(0.01)
    assert timing["save_regularly"] is True
    assert timing["summarise_regularly"] is True


def test_derived_window_is_the_duration():
    """An unset window takes the duration and a hundredth as sample."""
    timing = resolve_loop_timing(None, None, None, False, True, 2.0)
    assert timing["summarise_every"] == 2.0
    assert timing["sample_summaries_every"] == pytest.approx(0.02)


# ── Solver ──────────────────────────────────────────────────────────── #


def test_settings_dict_is_the_user_given(solver, solver_settings):
    """The solver reports what it was given, not what it derived."""
    settings = solver.settings_dict
    assert settings["algorithm"] == solver_settings["algorithm"]
    assert settings["dt"] == solver_settings["dt"]
    assert "save_last" not in settings
    assert "is_adaptive" not in settings
    assert solver.effective_settings["save_regularly"] is True


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "kvaerno3", "step_controller": "pid"}],
    indirect=True,
)
def test_given_controller_survives_an_algorithm_change(solver_mutable):
    """A given controller stays through an algorithm change."""
    solver_mutable.update(algorithm="crank_nicolson")
    run = solver_mutable.kernel.single_integrator
    assert run.step_controller == "pid"
    assert run._algo_step.algorithm_family == "crank_nicolson"


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
def test_last_of_filter_or_gains_wins(solver_mutable):
    """A later gain unsets an earlier filter and the reverse."""
    solver_mutable.update(filter_coefficients="pi42")
    assert "filter_coefficients" in solver_mutable.settings_dict
    solver_mutable.update(integral_gain=0.5)
    assert "filter_coefficients" not in solver_mutable.settings_dict
    controller = solver_mutable.kernel.single_integrator._step_controller
    assert controller.integral_gain == pytest.approx(0.5)
    solver_mutable.update(filter_coefficients="pi42")
    assert "integral_gain" not in solver_mutable.settings_dict


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
    assert solver_mutable.sample_summaries_every == pytest.approx(0.005)
    assert "summarise_every" not in solver_mutable.settings_dict


@pytest.mark.parametrize(
    "solver_settings_override", [SUMMARY_ONLY_NO_TIMING], indirect=True
)
def test_derived_window_warns_at_construction(
    system, solver_settings, driver_settings
):
    """Summaries without a window warn when the solver is built."""
    with pytest.warns(UserWarning, match="sample_summaries_every"):
        built = _build_solver_instance(
            system, solver_settings, driver_settings
        )
    built.close()


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
    assert "unroll_newton_exits" not in solver_mutable.settings_dict
