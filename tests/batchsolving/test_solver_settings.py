"""Tests for the solver-level settings record and its resolution."""

import warnings

import numpy as np
import pytest
from attrs import evolve, fields, fields_dict

from cubie.array_interpolator import ALL_INTERPOLATOR_PARAMETERS
from cubie.batchsolving.BatchSolverConfig import ALL_KERNEL_PARAMETERS
from cubie.batchsolving.resolve_defaults import (
    resolve,
    resolve_inner_tolerances,
    resolve_loop_timing,
    resolve_step_bounds,
    unset_updates,
)
from cubie.batchsolving.solver import Solver
from cubie.batchsolving.solver_settings import (
    EffectiveSettings,
    SolverSettings,
)
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.CUDAFactory import ALL_JIT_PARAMETERS, ALL_UNROLL_PARAMETERS
from cubie.integrators.algorithms import DIRK_TABLEAU_REGISTRY
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
    BaseAlgorithmStep,
)
from cubie.integrators.algorithms.ode_implicitstep import ImplicitStepConfig
from cubie.integrators.dae_initialiser import DAEInitialiser
from cubie.integrators.loops.ode_loop import ALL_LOOP_SETTINGS
from cubie.integrators.step_control import _CONTROLLER_REGISTRY
from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
    BaseStepController,
)
from cubie.memory.mem_manager import ALL_MEMORY_MANAGER_PARAMETERS
from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.symbolic import create_ODE_system
from cubie.outputhandling.output_functions import (
    ALL_OUTPUT_FUNCTION_PARAMETERS,
    OutputFunctions,
)
from cubie.time_logger import default_timelogger
from tests._utils import (
    LARGE_DIRK,
    SUMMARY_ONLY_LAST,
    TORN_NO_OBSERVABLES,
    _build_solver_instance,
)


def _given(**settings):
    """Return a record with ``settings`` given."""
    record, _, _ = SolverSettings().update(settings)
    return record


def _effective(system, **given):
    """Resolve ``given`` for ``system`` and return the effective record."""
    return resolve(_given(**given), system, SystemInterface(system))


def _controller_default(name, gain):
    """Return the gain default of the named controller's config."""
    config_class = _CONTROLLER_REGISTRY[name]._config_class
    return getattr(fields(config_class), f"_{gain}").default


def _names(cls):
    """Return the init field names of a record class."""
    return {fld.name for fld in fields(cls) if fld.init}


# ── The record ──────────────────────────────────────────────────────── #


def test_record_matches_the_children_settings(system):
    """Record = children's settings minus system inputs plus the Solver's."""
    children = (
        set(ALL_ALGORITHM_STEP_PARAMETERS)
        | set(ALL_STEP_CONTROLLER_PARAMETERS)
        | set(ALL_LOOP_SETTINGS)
        | set(ALL_OUTPUT_FUNCTION_PARAMETERS)
        | set(ALL_KERNEL_PARAMETERS)
        | set(ALL_MEMORY_MANAGER_PARAMETERS)
        | set(ALL_INTERPOLATOR_PARAMETERS)
        | set(ALL_UNROLL_PARAMETERS)
        | set(ALL_JIT_PARAMETERS)
    )
    system_inputs = (
        set(
            BaseAlgorithmStep.system_inputs(
                system,
                drivers_fn=None,
                driver_derivative_fn=None,
                is_adaptive=False,
            )
        )
        | set(BaseStepController.system_inputs(system, algorithm_order=1))
        | set(DAEInitialiser.system_inputs(system))
        | set(OutputFunctions.system_inputs(system))
    )
    solver_own = {
        "duration",
        "tableau",
        "save_variables",
        "summarise_variables",
        "time_logging_level",
        "operation_ordering",
    }
    step_own = {"operator_beta", "operator_gamma"}
    resolved_only = _names(EffectiveSettings) - _names(SolverSettings)
    expected = (
        (children - system_inputs - step_own - resolved_only)
        | solver_own
        | {"precision"}
    )
    assert _names(SolverSettings) == expected


def test_grouped_dicts_flatten_and_unknown_names_raise(system):
    """Settings groups flatten at the Solver; unknown names raise."""
    built = Solver(system, step_control_settings={"dt": 0.01})
    assert built.given.dt == 0.01
    built.close()
    with pytest.raises(KeyError, match="Unrecognized"):
        Solver(system, not_a_setting=1)


def test_signature_defaults_are_not_given(system):
    """A Solver given nothing records nothing and resolves euler."""
    built = Solver(system)
    assert built.settings_dict == {}
    assert built.effective.algorithm == "euler"
    assert built.effective.step_controller == "fixed"
    assert built.kernel.compile_settings.auto_performance is True
    assert built.kernel.compile_settings.cache.cache_enabled is True
    built.close()


def test_resolved_names_are_not_settings():
    """A name only resolution sets is not a given setting."""
    _, recognised, _ = SolverSettings().update({"is_adaptive": True})
    assert recognised == set()
    assert "is_adaptive" in _names(EffectiveSettings)
    assert _names(SolverSettings) < _names(EffectiveSettings)


def test_update_records_and_none_makes_not_given():
    """``update`` records, ``None`` makes not given, changes are reported."""
    given = _given(dt=0.01, atol=1e-6)
    later, recognised, changed = given.update({"dt": None, "rtol": 1e-4})
    assert recognised == {"dt", "rtol"}
    assert changed == {"dt", "rtol"}
    assert later.is_given("dt") is False
    assert later.rtol == 1e-4
    assert later.atol == 1e-6
    assert given.dt == 0.01
    same, _, changed = later.update({"rtol": 1e-4})
    assert same is later
    assert changed == set()


def test_update_compares_index_arrays_by_value():
    """Index arrays of different lengths compare as a change."""
    given = _given(saved_state_indices=[0, 1])
    _, _, changed = given.update({"saved_state_indices": [0, 1, 2]})
    assert changed == {"saved_state_indices"}
    _, _, changed = given.update({"saved_state_indices": [0, 1]})
    assert changed == set()


def test_as_kwargs_is_the_given_settings():
    """``as_kwargs`` carries the given fields only."""
    given = _given(dt=0.01, algorithm="euler")
    assert given.as_kwargs() == {"dt": 0.01, "algorithm": "euler"}


def test_effective_as_kwargs_keeps_none_intervals(system):
    """The effective record keeps a ``None`` interval and memory value."""
    effective = _effective(system, output_types=["state"])
    kwargs = effective.as_kwargs()
    assert kwargs["summarise_every"] is None
    assert kwargs["mem_proportion"] is None


# ── Step keys ───────────────────────────────────────────────────────── #


def test_gains_with_a_filter_raise(system):
    """Gains and a filter cannot both be given."""
    with pytest.raises(ValueError, match="filter_coefficients"):
        _effective(
            system,
            algorithm="kvaerno3",
            filter_coefficients="pi42",
            integral_gain=0.5,
        )


def test_explicit_step_setting_overrides_step_default(system):
    """A given step key survives the family defaults."""
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


def test_different_solver_choice_takes_the_step_variant_defaults(system):
    """A non-family linear solver takes the step's own variant defaults."""
    effective = _effective(
        system,
        algorithm="dirk",
        tableau=_variant_probe_tableau(),
        linear_correction_type="bicgstab",
    )
    assert effective.linear_correction_type == "bicgstab"
    step_fields = fields_dict(ImplicitStepConfig)
    assert effective.inexact_newton is step_fields["inexact_newton"].default
    assert effective.prefactored is step_fields["prefactored"].default


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


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "torn_time", **TORN_NO_OBSERVABLES}],
    indirect=True,
)
def test_mass_matrix_system_takes_the_dae_linear_solve(system):
    """A mass-matrix system resolves to the DAE linear solver."""
    effective = _effective(system, algorithm="backwards_euler")
    assert effective.linear_correction_type == "lu"
    assert effective.preconditioner_type == "jacobi"


# ── Controller ──────────────────────────────────────────────────────── #


def test_unnamed_controller_promotes_to_carry_given_gains(system):
    """A given gain promotes the controller; other gains are its defaults."""
    effective = _effective(system, algorithm="kvaerno3", derivative_gain=0.05)
    assert effective.step_controller == "pid"
    assert effective.derivative_gain == 0.05
    assert effective.integral_gain == _controller_default(
        "pid", "integral_gain"
    )


def test_named_controller_is_not_promoted(system):
    """A named controller drops the gains it lacks."""
    effective = _effective(
        system, algorithm="kvaerno3", step_controller="i", derivative_gain=0.05
    )
    assert effective.step_controller == "i"
    assert effective.derivative_gain is None
    assert effective.integral_gain == _controller_default(
        "i", "integral_gain"
    )


def test_fixed_controller_resolves_no_gains(system):
    """A fixed controller resolves neither gains nor adaptive limits."""
    effective = _effective(system, algorithm="euler")
    kwargs = effective.as_kwargs()
    assert effective.step_controller == "fixed"
    assert set(kwargs) & {
        "integral_gain",
        "proportional_gain",
        "derivative_gain",
        "dt_min",
        "dt_max",
    } == set()


def test_filter_coefficients_replace_the_family_gains(system):
    """A filter passes through and no gain is derived beside it."""
    effective = _effective(
        system, algorithm="kvaerno3", filter_coefficients="pi42"
    )
    assert effective.filter_coefficients == "pi42"
    assert effective.integral_gain is None
    assert effective.proportional_gain is None


def test_errorless_algorithm_replaces_an_adaptive_request(system):
    """An adaptive request on an errorless step is fixed, with a warning."""
    with pytest.warns(UserWarning, match="cannot be used with"):
        effective = _effective(
            system, algorithm="euler", step_controller="pid"
        )
    assert effective.step_controller == "fixed"
    assert effective.is_adaptive is False


# ── Step bounds and tolerances ──────────────────────────────────────── #


def test_adaptive_bounds_follow_a_lone_dt():
    """A lone dt gives bounds three decades either side."""
    bounds = resolve_step_bounds(0.01, None, None, True)
    assert bounds == {"dt": 0.01, "dt_min": 1e-5, "dt_max": 10.0}


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
    tolerances = resolve_inner_tolerances(
        SolverSettings(), 1e-6, 1e-4, True, False, np.float32
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
    """A given inner tolerance is kept."""
    tolerances = resolve_inner_tolerances(
        _given(newton_atol=3e-9), 1e-6, 1e-4, True, False, np.float32
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


def test_summaries_need_a_sample_interval():
    """Summary outputs without a sample interval raise."""
    with pytest.raises(ValueError, match="sample_summaries_every"):
        resolve_loop_timing(0.02, 0.1, None, True, True)
    with pytest.raises(ValueError, match="sample_summaries_every"):
        resolve_loop_timing(None, None, None, False, True)


def test_given_window_summarises_regularly():
    """A provided window with its sample interval summarises regularly."""
    timing = resolve_loop_timing(0.02, 0.1, 0.01, True, True)
    assert timing["summarise_every"] == 0.1
    assert timing["sample_summaries_every"] == 0.01
    assert timing["save_regularly"] is True
    assert timing["summarise_regularly"] is True
    assert timing["summarise_last"] is False


def test_unset_window_summarises_last():
    """An unset window is one summary over the run at its end."""
    timing = resolve_loop_timing(None, None, 0.05, False, True)
    assert timing["summarise_every"] is None
    assert timing["sample_summaries_every"] == 0.05
    assert timing["summarise_last"] is True
    assert timing["summarise_regularly"] is False


def test_unset_updates_carry_unset_intervals(system):
    """An interval unset to ``None`` is pushed; other unsets are not."""
    before = _effective(
        system,
        output_types=["state", "mean"],
        save_every=0.02,
        summarise_every=0.04,
        sample_summaries_every=0.02,
        blocksize=64,
    )
    after = _effective(
        system,
        output_types=["state", "mean"],
        sample_summaries_every=0.02,
    )
    assert unset_updates(before, after) == {
        "save_every": None,
        "summarise_every": None,
    }


def test_no_summaries_clears_the_summary_timing():
    """Without summary outputs neither summary flag is set."""
    timing = resolve_loop_timing(None, 0.1, 0.05, True, False)
    assert timing["summarise_every"] is None
    assert timing["sample_summaries_every"] is None
    assert timing["summarise_last"] is False
    assert timing["summarise_regularly"] is False


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


def test_empty_output_types_raise(system):
    """An empty output selection is an error, not the default."""
    with pytest.raises(ValueError, match="At least one output type"):
        _effective(system, output_types=[])


# ── System definition ───────────────────────────────────────────────── #


def test_constant_named_as_a_setting_is_rejected_at_definition():
    """A constant sharing a setting's name cannot be defined."""
    with pytest.raises(ValueError, match="order"):
        create_ODE_system(
            dxdt=["dx0 = -order * x0"],
            states={"x0": 1.0},
            constants={"order": 2.0},
        )


def test_default_constant_named_as_a_setting_is_rejected():
    """A default constant sharing a setting's name cannot be defined."""

    class ConstantsOnly(BaseODE):
        def build(self):
            raise NotImplementedError

    with pytest.raises(ValueError, match="order"):
        ConstantsOnly(
            initial_values={"x0": 1.0},
            constants={"c1": 2.0},
            default_constants={"order": 2.0},
        )


# ── Solver ──────────────────────────────────────────────────────────── #


def test_settings_dict_is_the_given_settings(solver, solver_settings):
    """The solver reports what it was given, not what it derived."""
    settings = solver.settings_dict
    assert settings["algorithm"] == solver_settings["algorithm"]
    assert settings["dt"] == solver_settings["dt"]
    assert set(settings) <= _names(SolverSettings)
    assert solver.effective.save_regularly is True


def test_is_given_reads_the_record(solver, solver_settings):
    """``Solver.is_given`` reports the record's givenness."""
    assert solver.is_given("dt") is True
    assert solver.is_given("dt") == solver.given.is_given("dt")
    assert solver.is_given("kernel_name") is False


def test_time_logging_level_applies_on_update(solver_mutable):
    """An updated logging level reaches the global logger."""
    previous = default_timelogger.verbosity
    try:
        solver_mutable.update(time_logging_level="silent")
        assert default_timelogger.verbosity == "silent"
        assert solver_mutable.settings_dict["time_logging_level"] == "silent"
    finally:
        default_timelogger.set_verbosity(previous)


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
def test_variant_default_follows_a_solver_change(solver_mutable):
    """A non-family solver resets a variant the family solver set."""
    family_solver = solver_mutable.effective.linear_correction_type
    other = "bicgstab" if family_solver == "lu" else "lu"
    solver_mutable.update(inexact_newton=True)
    step = solver_mutable.kernel.single_integrator._algo_step
    assert step.compile_settings.inexact_newton is True
    solver_mutable.update(inexact_newton=None, linear_correction_type=other)
    step = solver_mutable.kernel.single_integrator._algo_step
    assert solver_mutable.effective.linear_correction_type == other
    assert step.compile_settings.inexact_newton is (
        fields_dict(ImplicitStepConfig)["inexact_newton"].default
    )
    solver_mutable.update(linear_correction_type=family_solver)


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "kvaerno3", "step_controller": "pid"}],
    indirect=True,
)
def test_filter_after_gains_raises(solver_mutable):
    """A filter cannot join given gains without unsetting them."""
    solver_mutable.update(integral_gain=0.5)
    with pytest.raises(ValueError, match="filter_coefficients"):
        solver_mutable.update(filter_coefficients="pi42")
    solver_mutable.update(integral_gain=None, filter_coefficients="pi42")
    assert solver_mutable.settings_dict["filter_coefficients"] == "pi42"


def test_none_on_a_plain_setting_keeps_the_factory_value(solver_mutable):
    """A plain setting made not given leaves the factory as it was."""
    solver_mutable.update(max_registers=96)
    assert solver_mutable.kernel.compile_settings.max_registers == 96
    solver_mutable.update(max_registers=None)
    assert solver_mutable.is_given("max_registers") is False
    assert solver_mutable.kernel.compile_settings.max_registers == 96


def test_none_on_an_interval_reaches_the_loop(solver_mutable):
    """An unset loop interval resolves and reaches the loop as ``None``."""
    solver_mutable.update(save_every=0.05)
    loop = solver_mutable.kernel.single_integrator._loop
    assert loop.compile_settings.save_every == pytest.approx(0.05)
    solver_mutable.update(save_every=None)
    loop = solver_mutable.kernel.single_integrator._loop
    assert loop.compile_settings.save_every is None
    assert loop.compile_settings.save_last is True
    assert solver_mutable.effective.save_regularly is False


def test_output_selection_grows_on_update(solver_mutable):
    """A longer index selection replaces a shorter one."""
    solver_mutable.update(saved_state_indices=[0])
    np.testing.assert_array_equal(solver_mutable.saved_state_indices, [0])
    solver_mutable.update(saved_state_indices=[0, 1])
    np.testing.assert_array_equal(
        solver_mutable.saved_state_indices, [0, 1]
    )


def test_rejected_update_changes_nothing(solver_mutable):
    """An update with an unknown name leaves the solver as it was."""
    name = solver_mutable.system.constants.names[0]
    before = float(solver_mutable.system.constants.values_dict[name])
    given = solver_mutable.given
    verbosity = default_timelogger.verbosity
    with pytest.raises(KeyError, match="typo"):
        solver_mutable.update(
            {name: before * 2.0, "dt": 0.123, "typo": 1,
             "time_logging_level": "silent"}
        )
    assert float(solver_mutable.system.constants.values_dict[name]) == (
        pytest.approx(before)
    )
    assert solver_mutable.given is given
    assert default_timelogger.verbosity == verbosity


def test_memory_manager_cannot_change_on_a_live_solver(solver_mutable):
    """A different memory manager is refused."""
    with pytest.raises(ValueError, match="memory manager"):
        solver_mutable.update(memory_manager=object())


@pytest.mark.parametrize(
    "solver_settings_override", [SUMMARY_ONLY_LAST], indirect=True
)
def test_unset_window_keeps_the_build_across_durations(
    solver_mutable, batch_input_arrays, driver_settings
):
    """One whole-run summary per solve, whatever the duration."""
    initial_values, parameters = batch_input_arrays
    first = solver_mutable.solve(
        initial_values=initial_values,
        parameters=parameters,
        drivers=driver_settings,
        duration=0.5,
    )
    assert solver_mutable.summarise_every is None
    assert solver_mutable.sample_summaries_every == pytest.approx(0.02)
    assert solver_mutable.kernel.single_integrator.summarise_last is True
    assert first.state_summaries.shape[0] == 1
    assert solver_mutable.kernel._cache_valid
    second = solver_mutable.solve(
        initial_values=initial_values,
        parameters=parameters,
        drivers=driver_settings,
        duration=0.9,
    )
    assert solver_mutable.kernel._cache_valid
    assert second.state_summaries.shape[0] == 1


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**SUMMARY_ONLY_LAST, "sample_summaries_every": None}],
    indirect=True,
)
def test_summaries_without_a_sample_interval_raise_at_construction(
    system, solver_settings, driver_settings
):
    """Summaries without a sample interval raise when the solver is built."""
    with pytest.raises(ValueError, match="sample_summaries_every"):
        _build_solver_instance(system, solver_settings, driver_settings)


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
    """A given precision updates the system and every child."""
    other = np.float64 if system.precision == np.float32 else np.float32
    built = _build_solver_instance(
        system.copy(), solver_settings, driver_settings
    )
    built.update(precision=other)
    run = built.kernel.single_integrator
    assert built.system.precision == other
    assert built.effective.precision == other
    assert built.kernel.precision == other
    assert run._algo_step.precision == other
    assert run._step_controller.precision == other
    assert run._loop.precision == other
    assert run._output_functions.precision == other
    built.close()


def test_compile_flags_reach_every_factory(solver_mutable):
    """A loose unroll key and a jit flag reach the kernel and children."""
    solver_mutable.update(unroll_stage=(True, 2), lineinfo=True)
    kernel = solver_mutable.kernel
    run = kernel.single_integrator
    for factory in (
        kernel,
        kernel.driver_interpolator,
        run,
        run._loop,
        run._output_functions,
        run._algo_step,
        run._step_controller,
        run._dae_initialiser,
    ):
        assert factory.compile_settings.unroll.unroll_stage == (True, 2)
        assert factory.compile_settings.jit_flags.lineinfo is True


def test_constants_reach_the_system_by_name(
    system, solver_settings, driver_settings
):
    """A constant given by name reaches the system; it is not a setting."""
    name = system.constants.names[0]
    value = float(system.constants.values_dict[name]) * 1.5
    built = _build_solver_instance(
        system.copy(), {**solver_settings, name: value}, driver_settings
    )
    assert float(built.system.constants.values_dict[name]) == (
        pytest.approx(value)
    )
    assert set(built.settings_dict) <= _names(SolverSettings)
    recognised = built.update(**{name: value * 2.0})
    assert recognised == {name}
    assert float(built.system.constants.values_dict[name]) == (
        pytest.approx(value * 2.0)
    )
    built.close()
