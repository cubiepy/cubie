"""Unit tests for the dense stage predictor factory."""

import attrs
import numpy as np
import pytest

from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.base_algorithm_step import (
    ButcherTableau,
)
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DIRK_TABLEAU_REGISTRY,
    DIRKTableau,
)
from cubie.integrators.algorithms.generic_firk_tableaus import (
    DEFAULT_FIRK_TABLEAU,
    FIRK_TABLEAU_REGISTRY,
    RADAU_IIA_5_TABLEAU,
)
from cubie.integrators.stage_predictors import (
    DenseStagePredictor,
    dense_predictor_matrix,
    dense_predictor_ratio_coefficients,
    tableau_supports_dense_prediction,
)
from tests._utils import run_dense_predictor_step


REPEATED_NODE_TABLEAU = ButcherTableau(
    a=((0.5, 0.0), (0.0, 0.5)),
    b=(0.5, 0.5),
    c=(0.5, 0.5),
    order=1,
)

ZERO_NODE_SINGULAR_TABLEAU = ButcherTableau(
    a=((0.0, 0.0), (0.5, 0.5)),
    b=(0.5, 0.5),
    c=(0.0, 1.0),
    order=2,
)


def test_repeated_node_tableau_reads_through_last_sample():
    """A single distinct node yields a ratio-scaled carry of the
    last stage's sample, shared by both stages."""
    assert tableau_supports_dense_prediction(REPEATED_NODE_TABLEAU)
    predictor = dense_predictor_matrix(REPEATED_NODE_TABLEAU, 0.5)
    np.testing.assert_allclose(
        predictor, [[0.0, 0.5], [0.0, 0.5]]
    )


def test_zero_node_singular_tableau_is_supported():
    """Derivative samples need neither nonzero nodes nor invertible A."""
    assert tableau_supports_dense_prediction(
        ZERO_NODE_SINGULAR_TABLEAU
    )


NEAR_DUPLICATE_NODE_TABLEAU = ButcherTableau(
    a=((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)),
    b=(0.4, 0.3, 0.3),
    c=(0.5, 0.5 + 1e-9, 1.0),
    order=1,
)


def test_near_duplicate_node_tableau_is_rejected():
    """Nearly coincident nodes amplify sample error into garbage."""
    assert not tableau_supports_dense_prediction(
        NEAR_DUPLICATE_NODE_TABLEAU
    )
    with pytest.raises(ValueError, match="amplifies"):
        dense_predictor_matrix(NEAR_DUPLICATE_NODE_TABLEAU)


FIRK_SUPPORT_EXPECTATIONS = {
    "firk_gauss_legendre_2": True,
    "firk_gauss_legendre_4": True,
    "radau_iia_5": True,
}

DIRK_SUPPORT_EXPECTATIONS = {
    "implicit_midpoint": True,
    "trapezoidal_dirk": True,
    "ode23t": True,
    "kvaerno3": True,
    "kvaerno5": True,
    "sdirk_2_2": True,
    "l_stable_dirk_3": True,
    "l_stable_sdirk_4": True,
}


def test_registry_tableaus_support_expectations():
    """Known registry tableaus land on the expected side of the gate."""
    for name, expected in FIRK_SUPPORT_EXPECTATIONS.items():
        tableau = FIRK_TABLEAU_REGISTRY[name]
        assert (
            tableau_supports_dense_prediction(tableau) is expected
        ), name
    for name, expected in DIRK_SUPPORT_EXPECTATIONS.items():
        tableau = DIRK_TABLEAU_REGISTRY[name]
        assert (
            tableau_supports_dense_prediction(tableau) is expected
        ), name


@pytest.mark.parametrize(
    "tableau",
    [
        DIRK_TABLEAU_REGISTRY["l_stable_dirk_3"],
        DIRK_TABLEAU_REGISTRY["sdirk_2_2"],
        RADAU_IIA_5_TABLEAU,
    ],
)
@pytest.mark.parametrize("step_ratio", [0.5, 1.0, 1.7])
def test_dense_predictor_reads_derivative_samples_ahead(
    tableau, step_ratio
):
    """The matrix equals an independent derivative-sample fit."""
    stage_nodes = np.asarray(tableau.c)
    increments = np.arange(1, tableau.stage_count + 1, dtype=float)
    polynomial = np.polynomial.polynomial.polyfit(
        stage_nodes, increments, tableau.stage_count - 1
    )
    expected = step_ratio * np.polynomial.polynomial.polyval(
        1.0 + step_ratio * stage_nodes, polynomial
    )
    predictor = dense_predictor_matrix(tableau, step_ratio)
    np.testing.assert_allclose(predictor @ increments, expected)


@pytest.mark.parametrize(
    "tableau",
    [
        DIRK_TABLEAU_REGISTRY["kvaerno3"],
        DIRK_TABLEAU_REGISTRY["kvaerno5"],
    ],
)
@pytest.mark.parametrize("step_ratio", [0.5, 1.0, 1.7])
def test_repeated_nodes_fit_through_distinct_subset(
    tableau, step_ratio
):
    """Repeated nodes drop to the last stage's sample; every stage
    reads the fit through the distinct subset."""
    stage_nodes = np.asarray(tableau.c)
    increments = np.arange(1, tableau.stage_count + 1, dtype=float)
    kept_samples = {
        node: index for index, node in enumerate(stage_nodes.tolist())
    }
    distinct_nodes = np.asarray(list(kept_samples.keys()))
    polynomial = np.polynomial.polynomial.polyfit(
        distinct_nodes,
        increments[list(kept_samples.values())],
        len(distinct_nodes) - 1,
    )
    expected = step_ratio * np.polynomial.polynomial.polyval(
        1.0 + step_ratio * stage_nodes, polynomial
    )
    predictor = dense_predictor_matrix(tableau, step_ratio)
    np.testing.assert_allclose(
        predictor @ increments, expected, rtol=1e-9
    )
    dropped = [
        index
        for index in range(tableau.stage_count)
        if index not in kept_samples.values()
    ]
    assert np.all(predictor[:, dropped] == 0.0)


@pytest.mark.parametrize(
    "tableau",
    [DEFAULT_FIRK_TABLEAU, RADAU_IIA_5_TABLEAU],
)
@pytest.mark.parametrize("step_ratio", [0.5, 1.0, 1.7])
def test_dense_predictor_extrapolates_stage_curve(tableau, step_ratio):
    """For collocation tableaus the read-ahead extrapolates the
    state collocation polynomial, checked by an independent fit."""
    stage_matrix = np.asarray(tableau.a)
    stage_nodes = np.asarray(tableau.c)
    increments = np.arange(1, tableau.stage_count + 1, dtype=float)
    old_stage_values = stage_matrix @ increments

    polynomial = np.polynomial.polynomial.polyfit(
        np.concatenate(([0.0], stage_nodes)),
        np.concatenate(([0.0], old_stage_values)),
        tableau.stage_count,
    )
    shifted_stage_values = np.polynomial.polynomial.polyval(
        1.0 + step_ratio * stage_nodes, polynomial
    ) - np.polynomial.polynomial.polyval(1.0, polynomial)
    expected = np.linalg.solve(stage_matrix, shifted_stage_values)

    predictor = dense_predictor_matrix(tableau, step_ratio)
    np.testing.assert_allclose(predictor @ increments, expected)


@pytest.mark.parametrize(
    "tableau",
    [
        DEFAULT_FIRK_TABLEAU,
        RADAU_IIA_5_TABLEAU,
        DIRK_TABLEAU_REGISTRY["l_stable_dirk_3"],
        DIRK_TABLEAU_REGISTRY["sdirk_2_2"],
        DIRK_TABLEAU_REGISTRY["kvaerno3"],
        DIRK_TABLEAU_REGISTRY["kvaerno5"],
    ],
)
@pytest.mark.parametrize("step_ratio", [0.65, 1.3])
def test_ratio_coefficients_reconstruct_matrix(tableau, step_ratio):
    """Summing the power stack reproduces the matrix at any ratio."""
    coefficients = dense_predictor_ratio_coefficients(tableau)
    reconstructed = np.zeros_like(coefficients[0])
    for power_index in range(coefficients.shape[0]):
        reconstructed += (
            coefficients[power_index] * step_ratio ** (power_index + 1)
        )
    direct = dense_predictor_matrix(tableau, step_ratio)
    scale = np.max(np.abs(direct))
    assert np.max(np.abs(reconstructed - direct)) < 1e-12 * scale


_LORENZ_ITERATION_BASE = {
    "system_type": "lorenz_julia",
    "output_types": ["state", "iteration_counters"],
    "saved_state_indices": [0, 1, 2],
    "saved_observable_indices": [],
    "summarised_state_indices": [],
    "summarised_observable_indices": [],
    "summarise_every": None,
    "sample_summaries_every": None,
}

_RADAU_ADAPTIVE_CASE = {
    **_LORENZ_ITERATION_BASE,
    "algorithm": "radau",
    "step_controller": "gustafsson",
    "dt_min": 1e-6,
    "dt_max": 0.02,
    "atol": 1e-6,
    "rtol": 1e-6,
}


@pytest.mark.parametrize(
    "solver_settings_override",
    [_RADAU_ADAPTIVE_CASE],
    indirect=True,
)
def test_predictor_update_flows_through_solver(solver_mutable):
    """Prediction settings ride the standard solver update path."""
    algo_step = solver_mutable.kernel.single_integrator._algo_step
    predictor = algo_step.dense_predictor
    assert (
        algo_step.compile_settings.previous_step_size_location
        == "local"
    )
    solver_mutable.update(previous_step_size_location="shared")
    assert (
        algo_step.compile_settings.previous_step_size_location
        == "shared"
    )
    # The previous-step-size scalar belongs to the algorithm, not
    # the predictor.
    assert not hasattr(
        predictor.compile_settings, "previous_step_size_location"
    )
    assert algo_step.dense_prediction
    solver_mutable.update(attempt_dense_prediction=False)
    assert not algo_step.dense_prediction


CALIBRATED_CEILINGS = {
    "implicit_midpoint": (1.0, 1.0),
    "trapezoidal_dirk": (0.39, 1.21),
    "kvaerno3": (0.85, 1.28),
    "kvaerno5": (0.0, 0.0),
    "sdirk_2_2": (1.07, 1.21),
    "l_stable_dirk_3": (0.85, 1.07),
    "l_stable_sdirk_4": (0.79, 0.79),
    "firk_gauss_legendre_2": (8.0, 8.0),
    "firk_gauss_legendre_4": (8.0, 8.0),
    "radau_iia_5": (8.0, 8.0),
}
"""Reviewed (float32, float64) ceilings from the device sweep
(``benchmarks/dense_prediction_ratio_sweep.py``)."""


def test_registry_ceilings_match_calibration():
    """Every registered implicit-RK tableau stores the reviewed
    per-precision ceilings: float16 disabled, float32/float64 the
    sweep's values, typed to the requested precision."""
    registries = {**DIRK_TABLEAU_REGISTRY, **FIRK_TABLEAU_REGISTRY}
    for name, expected in CALIBRATED_CEILINGS.items():
        tableau = registries[name]
        limit_f16 = tableau.dense_prediction_ratio_limit(np.float16)
        limit_f32 = tableau.dense_prediction_ratio_limit(np.float32)
        limit_f64 = tableau.dense_prediction_ratio_limit(np.float64)
        assert float(limit_f16) == 0.0, name
        assert isinstance(limit_f32, np.float32), name
        assert isinstance(limit_f64, np.float64), name
        assert limit_f32 == np.float32(expected[0]), name
        assert limit_f64 == np.float64(expected[1]), name


def test_uncalibrated_tableau_disables_prediction():
    """A custom tableau without calibrated ceilings compiles without
    dense prediction even when prediction is requested."""
    custom = DIRKTableau(
        a=((0.5, 0.0), (0.25, 0.5)),
        b=(0.5, 0.5),
        c=(0.5, 0.75),
        order=2,
    )
    step = DIRKStep(
        precision=np.float64,
        n=2,
        tableau=custom,
        attempt_dense_prediction=True,
    )
    assert not step.dense_prediction
    opened = attrs.evolve(custom, dense_prediction_ratio_float64=4.0)
    step.update(tableau=opened)
    assert step.dense_prediction


@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("apply_flag", [True, False])
def test_device_predictor_commit_flag(precision, apply_flag):
    """The compiled transform writes only when the flag allows.

    A false flag must leave the vector bit-for-bit unchanged; a true
    flag must write the host transform's values. The non-constant
    history catches row-order and precision-capture mistakes.
    """
    tableau = attrs.evolve(
        DIRK_TABLEAU_REGISTRY["l_stable_dirk_3"],
        dense_prediction_ratio_float32=8.0,
        dense_prediction_ratio_float64=8.0,
    )
    n = 2
    stage_count = tableau.stage_count
    predictor = DenseStagePredictor(
        precision=precision,
        n=n,
        tableau=tableau,
    )
    predict = predictor.device_function
    persistent_len = buffer_registry.persistent_local_buffer_size(
        predictor
    )

    history = np.linspace(
        0.3, 1.7, stage_count * n
    ).astype(precision)
    ratio = precision(1.25)
    result = run_dense_predictor_step(
        predict,
        history,
        ratio,
        apply_flag,
        precision,
        persistent_len,
    )

    if not apply_flag:
        assert np.array_equal(result, history)
        return
    matrix = dense_predictor_matrix(tableau, float(ratio))
    stage_major = history.astype(np.float64).reshape(stage_count, n)
    expected = matrix @ stage_major
    rtol = 1e-5 if precision is np.float32 else 1e-12
    np.testing.assert_allclose(
        result.reshape(stage_count, n).astype(np.float64),
        expected,
        rtol=rtol,
        atol=rtol,
    )


DENSE_PREDICTION_ITERATION_CASES = [
    pytest.param(
        {
            **_LORENZ_ITERATION_BASE,
            "algorithm": "firk",
            "step_controller": "fixed",
            "dt": 0.005,
        },
        id="firk-fixed",
    ),
    # The only DIRK tableau whose float32 ceiling (1.07) sits above
    # the fixed controller's ratio of 1, so prediction applies on
    # every step at the fixture's float32 default.
    pytest.param(
        {
            **_LORENZ_ITERATION_BASE,
            "algorithm": "sdirk_2_2",
            "step_controller": "fixed",
            "dt": 0.005,
        },
        id="dirk-fixed",
    ),
    # These tableaus' float32 ceilings sit below the fixed
    # controller's nominal ratio of 1; prediction applies on the
    # tiny clamped steps float32 save-boundary rounding inserts,
    # which is enough for the strict iteration guard.
    pytest.param(
        {
            **_LORENZ_ITERATION_BASE,
            "algorithm": "trapezoidal_dirk",
            "step_controller": "fixed",
            "dt": 0.005,
        },
        id="dirk-explicit-first-stage",
    ),
    pytest.param(
        {
            **_LORENZ_ITERATION_BASE,
            "algorithm": "kvaerno3",
            "step_controller": "fixed",
            "dt": 0.005,
        },
        id="dirk-repeated-nodes",
    ),
    pytest.param(_RADAU_ADAPTIVE_CASE, id="firk-adaptive"),
]


@pytest.mark.parametrize(
    "solver_settings_override",
    DENSE_PREDICTION_ITERATION_CASES,
    indirect=True,
)
def test_dense_prediction_reduces_newton_iterations(solver_mutable):
    """A live predictor strictly beats carried increments.

    Guards against the predictor silently compiling to a no-op
    (e.g. its persistent step-size scalar missing from the loop's
    layout): converged results cannot distinguish a dead predictor,
    but the Newton iteration count can. The chaotic ``rho = 28``
    regime is critical: there the stage curve moves enough between
    steps for the read-ahead to strictly beat carried increments on
    every method; the fixture default ``rho = 21`` settles onto a
    fixed point where both seeds are equally converged.
    """
    solve_kwargs = dict(
        initial_values={
            "x": np.array([1.05]),
            "y": np.array([0.97]),
            "z": np.array([1.02]),
        },
        parameters={"rho": np.array([28.0])},
        grid_type="verbatim",
        duration=0.1,
        save_every=0.025,
    )

    def newton_total(result):
        assert not np.any(result.status_codes)
        counters = np.asarray(result.iteration_counters)
        return int(counters.sum(axis=0)[0].sum())

    predicted = newton_total(solver_mutable.solve(**solve_kwargs))
    solver_mutable.update(attempt_dense_prediction=False)
    carried = newton_total(solver_mutable.solve(**solve_kwargs))
    assert predicted < carried
