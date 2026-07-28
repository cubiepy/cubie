"""Numerical correctness tests for the ODE integration loop.

Compares device loop outputs against CPU reference for various
algorithm, precision, and timing configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

from cubie.integrators.algorithms.generic_firk_tableaus import (
    FIRKTableau,
    GAUSS_LEGENDRE_4_TABLEAU,
)

from tests._utils import (
    ALGORITHM_CHAIN_CASES,
    ALGORITHM_CHAIN_SETS,
    DURATION_ONLY_MIXED_OUTPUTS,
    LARGE_T0_SMALL_STEPS_F32,
    LARGE_T0_SMALL_STEPS_F64,
    SAVE_LAST_EXPLICIT_FLAG,
    TIMED_MIXED_OUTPUTS,
    TINY_DT_ADAPTIVE_CN,
    WARMUP_SAVE_BOUNDARY,
    assert_integration_outputs,
    MID_RUN_PARAMS,
    merge_dicts,
    ALGORITHM_PARAM_SETS,
)


def _gauss_legendre_collocation_tableau(stage_count):
    """Build the Gauss--Legendre collocation tableau of order ``2s``.

    Collocation at the Gauss--Legendre nodes defines the classical
    fully implicit method of order ``2 * stage_count`` (Hairer &
    Wanner, Solving ODEs II, Theorem 7.7): each ``a[i][j]`` is the
    integral of the j-th Lagrange basis polynomial from 0 to ``c[i]``
    and each ``b[j]`` its integral over the whole step.
    """
    poly = np.polynomial.polynomial
    legendre_roots, _ = np.polynomial.legendre.leggauss(stage_count)
    nodes = 0.5 * (legendre_roots + 1.0)
    basis_integrals = []
    for basis_index in range(stage_count):
        other_nodes = [
            nodes[node_index]
            for node_index in range(stage_count)
            if node_index != basis_index
        ]
        coefficients = poly.polyfromroots(other_nodes)
        coefficients = coefficients / poly.polyval(
            nodes[basis_index], coefficients
        )
        basis_integrals.append(poly.polyint(coefficients))
    a_matrix = tuple(
        tuple(
            float(poly.polyval(node, integral))
            for integral in basis_integrals
        )
        for node in nodes
    )
    b_weights = tuple(
        float(poly.polyval(1.0, integral))
        for integral in basis_integrals
    )
    return FIRKTableau(
        a=a_matrix,
        b=b_weights,
        c=tuple(float(node) for node in nodes),
        order=2 * stage_count,
    )


def test_gauss_legendre_4_registry_literals_match_construction():
    """The registry's literals reproduce the collocation build."""
    constructed = _gauss_legendre_collocation_tableau(4)
    np.testing.assert_allclose(
        np.asarray(GAUSS_LEGENDRE_4_TABLEAU.a),
        np.asarray(constructed.a),
        rtol=1e-15,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        np.asarray(GAUSS_LEGENDRE_4_TABLEAU.b),
        np.asarray(constructed.b),
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(GAUSS_LEGENDRE_4_TABLEAU.c),
        np.asarray(constructed.c),
        rtol=1e-15,
    )


def test_initial_observable_seed_matches_reference(
    device_loop_outputs,
    cpu_loop_outputs,
    tolerance,
):

    np.testing.assert_allclose(
        device_loop_outputs.observables[0],
        cpu_loop_outputs["observables"][0],
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )

@pytest.mark.parametrize(
    "solver_settings_override",
    ALGORITHM_PARAM_SETS,
    indirect=True,
)
def test_loop(
    device_loop_outputs,
    step_object,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    atol = tolerance.abs_loose
    rtol = tolerance.rel_loose

    # High stage-count explicit methods compound roundoff between
    # GPU and CPU; relax tolerance for these.
    if step_object.stage_count > 10:
        rtol = tolerance.rel_loose * 5
        atol = tolerance.abs_loose * 5
    assert_integration_outputs(
        reference=cpu_loop_outputs,
        device=device_loop_outputs,
        output_functions=output_functions,
        rtol=rtol,
        atol=atol,
    )
    assert device_loop_outputs.status == 0


# Add combos
metric_test_output_cases = (
        {"output_types": [  # combined metrics
            "state",
            "mean",
            "std",
            "rms",
            "max",
            "min",
            "time",
            "max_magnitude",
            "peaks[3]",
            "negative_peaks[3]",
            "dxdt_max",
            "dxdt_min",
            "d2xdt2_max",
            "d2xdt2_min",
            ],
        },
        {  # no combos
            "output_types": [
                "state",
                "mean",
                "rms",
                "max",
                "min",
                "time",
                "max_magnitude",
                "negative_peaks[3]",
                "dxdt_max",
                "d2xdt2_max",
            ],
        },
)

metric_test_ids = (
        "combined metrics",
        "no combos",
)

METRIC_TEST_CASES_MERGED = [merge_dicts(MID_RUN_PARAMS, case)
                            for case in metric_test_output_cases]


@pytest.mark.parametrize(
    "solver_settings_override",
    METRIC_TEST_CASES_MERGED,
    ids=metric_test_ids,
    indirect=True,
)
def test_all_summary_metrics_numerical_check(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """Verify all summary metrics produce numerically correct
    results in loop context.

    Uses loose tolerance (1e-5) because of roundoff in the
    second-derivative methods - the cpu reference functions have
    no precision enforcement.
    """
    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_loose * 5,
        atol=tolerance.abs_loose * 5,
    )

    assert device_loop_outputs.status == 0, (
        "Integration should complete successfully"
    )


zero_state_metric_cases = (
    {"output_types": ["state", "time", "dxdt_max", "d2xdt2_max"]},
    {
        "output_types": [
            "state",
            "time",
            "dxdt_max",
            "dxdt_min",
            "d2xdt2_max",
            "d2xdt2_min",
        ],
    },
)


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        merge_dicts(
            MID_RUN_PARAMS,
            {
                "system_type": "linear",
                "summarised_state_indices": [0],
                "summarised_observable_indices": [],
            },
            case,
        )
        for case in zero_state_metric_cases
    ],
    ids=["individual metrics", "combined extrema"],
    indirect=True,
)
@pytest.mark.parametrize(
    "initial_state", [[0.0, 1.0, 1.0]], indirect=True
)
def test_derivative_metrics_zero_valued_variable(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """Derivative metrics of an identically zero variable are zero.

    The linear system's ``dx0 = -x0`` keeps ``x0`` at exactly 0.0 from
    a zero initial condition. The derivative metrics' history guard
    must gate on the sample counter, not on a previous value being
    nonzero: a value-based guard never primes on an all-zero
    trajectory and the save step then emits the scaled tracking
    sentinel (-1e30 / sample_summaries_every**k) instead of zero.
    """
    state_summaries = np.asarray(device_loop_outputs.state_summaries)
    assert np.all(state_summaries == 0.0), (
        "derivative metrics of an all-zero variable should be exactly "
        f"zero, got:\n{state_summaries}"
    )
    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )
    assert device_loop_outputs.status == 0


@pytest.mark.parametrize("solver_settings_override",
                         [
                             LARGE_T0_SMALL_STEPS_F32,
                             LARGE_T0_SMALL_STEPS_F64,
                         ],
                         indirect=True,
                         ids=["float32", "float64"])
def test_large_t0_with_small_steps(device_loop_outputs, precision):
    """Verify long integrations with small steps complete correctly."""
    assert np.isclose(device_loop_outputs.state[-2, -1],
                      precision(100.0008),
                      atol=2e-7)


@pytest.mark.parametrize("solver_settings_override",
                         [TINY_DT_ADAPTIVE_CN],
                         indirect=True,
                         ids=[""])
def test_adaptive_controller_with_float32(
    device_loop_outputs, precision
):
    """Verify adaptive controllers work with float32 and small
    dt_min."""
    assert device_loop_outputs.state[-2, -1] == pytest.approx(
        precision(1.00008)
    )


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override",
    [
        WARMUP_SAVE_BOUNDARY
    ],
    indirect=True,
)
def test_save_at_settling_time_boundary(
    device_loop_outputs, precision, solver_settings
):
    """A save lands exactly on the warmup boundary and the schedule
    continues through the saved window.

    The run spans t0 + warmup + duration = 1.3, with saves at 1.1
    (the warmup boundary), 1.2, and the run end. Save stamps are
    ``precision(t)`` where ``t`` accumulates in float64, so the
    final stamp is the float32 view of the committed end-of-run
    time: on hardware it equals ``precision(t0 + warmup +
    duration)``, one float32 ulp below the accumulated float32
    schedule value (1.3000001) that the third ``next_save``
    overshoots to. Marked nocudasim: the last-ulp outcome of the
    boundary-straddling truncation decisions differs between the
    simulator's Python arithmetic and compiled device arithmetic
    (the simulator stamps 1.3000001 here); the hardware value is
    the specification.
    """
    step = precision(0.1)
    boundary = precision(precision(1.0) + step)
    second_save = precision(boundary + step)
    end_time = precision(
        float(solver_settings["t0"])
        + float(solver_settings["warmup"])
        + float(solver_settings["duration"])
    )
    assert device_loop_outputs.state[-1, -1] == end_time
    assert device_loop_outputs.state[-2, -1] == second_save


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        DURATION_ONLY_MIXED_OUTPUTS
    ],
    indirect=True,
)
def test_final_summary(
    device_loop_outputs,
    precision,
):
    """Verify summaries collected at end of run with summaries unset.

    When all timing parameters are None, the loop should collect a
    summary at the end of the integration run.
    """
    state_summaries = device_loop_outputs.state_summaries

    assert state_summaries is not None, (
        "State summaries should be collected"
    )
    assert state_summaries.shape[0] >= 1, (
        "At least one summary should exist"
    )

    final_summary = state_summaries[0]
    assert not np.isnan(final_summary).any(), (
        "Summary should not contain NaN"
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        TIMED_MIXED_OUTPUTS
    ],
    indirect=True,
)
def test_summarise_every(
    device_loop_outputs,
    precision,
):
    """Verify summarise_every works without double-write.

    When both periodic summaries and summarise_last are enabled,
    the loop should collect summaries at regular intervals and also
    at the end.
    """
    state_summaries = device_loop_outputs.state_summaries

    assert state_summaries is not None, (
        "State summaries should be collected"
    )
    assert state_summaries.shape[0] >= 3, (
        "Multiple summaries expected"
    )

    for i in range(min(4, state_summaries.shape[0])):
        assert not np.isnan(state_summaries[i]).any(), \
            f"Summary {i} should not contain NaN"


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        SAVE_LAST_EXPLICIT_FLAG
    ],
    indirect=True,
)
def test_save_last_with_save_every(
    device_loop_outputs,
    precision,
):
    """Verify save_last and save_every can be used together.

    When both periodic saves and save_last are enabled, the final
    state should be saved even if it doesn't align with a periodic
    save point.
    """
    states = device_loop_outputs.state

    assert states is not None, "State outputs should be collected"
    assert states.shape[0] >= 4, "At least 4 saves expected"

    final_time = states[-1, -1]
    assert final_time == pytest.approx(
        precision(0.15), rel=1e-5
    ), "Final save should be at t_end"


def test_finish_check_no_float32_stagnation():
    """Regression: finish check must not stagnate in float32.

    When t0 is large relative to dt, float32 addition stagnates:
    float32(1000) + float32(1e-12) == float32(1000).  The finish
    check at ode_loop.py line 703 uses float64 to avoid this.

    Builds a SingleIntegratorRun with no regular outputs so that
    the no-regular-outputs branch is exercised.  Session fixtures
    cannot test this because their loop has summaries baked in.

    Under cudasim, the f32 overshoot (~6e7 extra steps) still
    completes in seconds, so this test validates the code path
    exists and produces correct status rather than detecting a
    hang.  On real CUDA hardware the overshoot would be
    proportionally more costly.
    """
    from cubie.buffer_registry import buffer_registry
    from cubie.integrators.SingleIntegratorRun import (
        SingleIntegratorRun,
    )
    from tests.system_fixtures import (
        build_three_state_constant_deriv_system,
    )
    from tests._utils import run_device_loop

    precision = np.float32
    system = build_three_state_constant_deriv_system(precision)
    buffer_registry.reset()

    sir = SingleIntegratorRun(
        system=system,
        step_control_settings={"step_controller": "fixed", "dt": 1e-8},
        algorithm_settings={"algorithm": "euler"},
        output_settings={
            "output_types": ["state", "time"],
            "saved_state_indices": [0],
            "saved_observable_indices": [],
            "summarised_state_indices": [],
            "summarised_observable_indices": [],
        },
        loop_settings={},
    )

    # Verify the no-regular-outputs branch is exercised
    loop_cfg = sir._loop.compile_settings
    assert not loop_cfg.save_regularly
    assert not loop_cfg.summarise_regularly

    solver_config = {
        "warmup": np.float64(0.0),
        "duration": np.float64(1e-6),
        "t0": np.float64(1000.0),
        "driverspline_order": 3,
    }
    result = run_device_loop(
        singleintegratorrun=sir,
        system=system,
        initial_state=system.initial_values.values_array.astype(
            precision
        ),
        solver_config=solver_config,
    )
    assert result.status == 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            **ALGORITHM_CHAIN_SETS["firk"],
            "preconditioned_vec_location": "shared",
        }
    ],
    ids=["firk-shared-solver-buffer"],
    indirect=True,
)
def test_firk_with_shared_solver_buffer_matches_reference(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """FIRK with a shared-located linear-solver buffer runs correctly.

    Verifies that placing the linear-solver buffer in shared memory
    produces a correctly sized pool and that the integration result
    matches the CPU reference within loose tolerances (issue #520).
    """
    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )
    assert device_loop_outputs.status == 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_CASES["firk-gauss-legendre-4"]],
    ids=["firk-four-stage-dense-predictor"],
    indirect=True,
)
def test_four_stage_firk_dense_predictor_matches_reference(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """The four-stage predictor path updates every state component."""

    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )
    assert device_loop_outputs.status == 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_CASES["dirk"]],
    ids=["dirk-dense-predictor-fixed"],
    indirect=True,
)
def test_dirk_dense_predictor_matches_reference(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """DIRK stage-history prediction matches the CPU reference."""

    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )
    assert device_loop_outputs.status == 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_CASES["dirk-trapezoidal"]],
    ids=["dirk-dense-predictor-explicit-first-stage"],
    indirect=True,
)
def test_explicit_stage_dirk_dense_predictor_matches_reference(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """History slots no solver writes carry the transform's values.

    An explicit first stage never solves, so its history slot is
    only ever refreshed by the read-ahead itself; the device and CPU
    reference must roll it forward identically.
    """

    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )
    assert device_loop_outputs.status == 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_CASES["firk-radau"]],
    ids=["firk-dense-predictor-adaptive"],
    indirect=True,
)
def test_adaptive_firk_dense_predictor_matches_reference(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """Ratio-based prediction under an adaptive controller matches."""

    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )
    assert device_loop_outputs.status == 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_CASES["dirk-l-stable-4"]],
    ids=["dirk-dense-predictor-adaptive"],
    indirect=True,
)
def test_adaptive_dirk_dense_predictor_matches_reference(
    device_loop_outputs,
    cpu_loop_outputs,
    output_functions,
    tolerance,
):
    """DIRK stage-history prediction under an adaptive controller."""

    assert_integration_outputs(
        cpu_loop_outputs,
        device_loop_outputs,
        output_functions,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )
    assert device_loop_outputs.status == 0
