"""Structural tests for single-step integration algorithms."""

from typing import Any

import numpy as np
import pytest

from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.algorithms.backwards_euler import BackwardsEulerStep
from cubie.integrators.algorithms.backwards_euler_predict_correct import (
    BackwardsEulerPCStep,
)
from cubie.integrators.algorithms.crank_nicolson import CrankNicolsonStep
from cubie.integrators.algorithms.explicit_euler import ExplicitEulerStep
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
    DIRK_TABLEAU_REGISTRY,
)
from cubie.integrators.algorithms.generic_erk import ERKStep
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DEFAULT_ERK_TABLEAU,
    ERK_TABLEAU_REGISTRY,
)
from cubie.integrators.algorithms.generic_firk import FIRKStep
from cubie.integrators.algorithms.generic_firk_tableaus import (
    DEFAULT_FIRK_TABLEAU,
    FIRK_TABLEAU_REGISTRY,
)
from cubie.integrators.norms import DIRKCorrectionNorm, FIRKCorrectionNorm
from cubie.integrators.algorithms.generic_rosenbrock_w import (
    GenericRosenbrockWStep,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    ROSENBROCK_TABLEAUS,
)
from tests.integrators.cpu_reference import (
    CPUODESystem,
    get_ref_step_factory,
)
from tests.integrators.cpu_reference.algorithms import (
    CPUDIRKStep,
    CPUERKStep,
    CPUFIRKStep,
    CPURosenbrockWStep,
)
from tests._utils import ALGORITHM_PARAM_SETS


def _expected_order(step_object: Any, tableau: Any) -> int:
    """Return the theoretical order of accuracy for ``step_object``."""

    if tableau is not None:
        return tableau.order
    if isinstance(
        step_object,
        (
            ExplicitEulerStep,
            BackwardsEulerStep,
            BackwardsEulerPCStep,
        ),
    ):
        return 1
    if isinstance(step_object, CrankNicolsonStep):
        return 2
    raise NotImplementedError(
        f"Order expectation missing for {type(step_object).__name__}."
    )


def _expected_memory_requirements(
    step_object: Any,
    tableau: Any,
    n_states: int,
    extra_shared: int,
    n_drivers: int = 1
) -> tuple[int, int]:
    """Return the expected shared and local scratch requirements."""

    if isinstance(step_object, ERKStep):
        stage_count = tableau.stage_count
        accumulator_span = max(stage_count - 1, 0) * n_states
        return accumulator_span, n_states
    if isinstance(step_object, DIRKStep):
        stage_count = tableau.stage_count
        accumulator_span = max(stage_count - 1, 0) * n_states
        shared = accumulator_span + 2 * n_states + extra_shared
        local = 2 * n_states
        return shared, local
    if isinstance(step_object, FIRKStep):
        stage_count = tableau.stage_count
        driver_stack = stage_count * n_drivers
        accumulator_span = stage_count * n_states
        solver_elements = 2 * accumulator_span
        shared = solver_elements + driver_stack + accumulator_span
        local = n_states
        return shared, local
    if isinstance(step_object, GenericRosenbrockWStep):
        stage_count = tableau.stage_count
        accumulator_span = stage_count * n_states
        shared = accumulator_span + n_states + extra_shared
        local = 0
        return shared, local
    if isinstance(
        step_object,
        (BackwardsEulerStep, BackwardsEulerPCStep, CrankNicolsonStep),
    ):
        shared = 2 * n_states + extra_shared
        return shared, 0
    if isinstance(step_object, ExplicitEulerStep):
        return 0, 0
    raise NotImplementedError(
        "Memory expectation missing for "
        f"{type(step_object).__name__}."
    )


ALIAS_CASES = [
    pytest.param(
        "erk",
        ERKStep,
        DEFAULT_ERK_TABLEAU,
        CPUERKStep,
        id="erk",
    ),
    pytest.param(
        "dirk",
        DIRKStep,
        DEFAULT_DIRK_TABLEAU,
        CPUDIRKStep,
        id="dirk",
    ),
    pytest.param(
        "firk",
        FIRKStep,
        DEFAULT_FIRK_TABLEAU,
        CPUFIRKStep,
        id="firk",
    ),
    pytest.param(
        "rosenbrock",
        GenericRosenbrockWStep,
        DEFAULT_ROSENBROCK_TABLEAU,
        CPURosenbrockWStep,
        id="rosenbrock",
    ),
    pytest.param(
        "dormand-prince-54",
        ERKStep,
        ERK_TABLEAU_REGISTRY["dormand-prince-54"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-dormand-prince-54",
    ),
    pytest.param(
        "dopri54",
        ERKStep,
        DEFAULT_ERK_TABLEAU,
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-dopri54",
    ),
    pytest.param(
        "cash-karp-54",
        ERKStep,
        ERK_TABLEAU_REGISTRY["cash-karp-54"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-cash-karp-54",
    ),
    pytest.param(
        "fehlberg-45",
        ERKStep,
        ERK_TABLEAU_REGISTRY["fehlberg-45"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-fehlberg-45",
    ),
    pytest.param(
        "bogacki-shampine-32",
        ERKStep,
        ERK_TABLEAU_REGISTRY["bogacki-shampine-32"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-bogacki-shampine-32",
    ),
    pytest.param(
        "heun-21",
        ERKStep,
        ERK_TABLEAU_REGISTRY["heun-21"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-heun-21",
    ),
    pytest.param(
        "ralston-33",
        ERKStep,
        ERK_TABLEAU_REGISTRY["ralston-33"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-ralston-33",
    ),
    pytest.param(
        "classical-rk4",
        ERKStep,
        ERK_TABLEAU_REGISTRY["classical-rk4"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-classical-rk4",
    ),
    pytest.param(
        "implicit_midpoint",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["implicit_midpoint"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-implicit-midpoint",
    ),
    pytest.param(
        "trapezoidal_dirk",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["trapezoidal_dirk"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-trapezoidal",
    ),
    pytest.param(
        "sdirk_2_2",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["sdirk_2_2"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-sdirk-2-2",
    ),
    pytest.param(
        "l_stable_dirk_3",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["l_stable_dirk_3"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-l-stable-3",
    ),
    pytest.param(
        "l_stable_sdirk_4",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["l_stable_sdirk_4"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-l-stable-4",
    ),
    pytest.param(
        "kvaerno3",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["kvaerno3"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-kvaerno3",
    ),
    pytest.param(
        "kvaerno5",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["kvaerno5"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-kvaerno5",
    ),
    pytest.param(
        "ros3p",
        GenericRosenbrockWStep,
        ROSENBROCK_TABLEAUS["ros3p"],
        CPURosenbrockWStep,
        marks=pytest.mark.specific_algos,
        id="rosenbrock-ros3p",
    ),
    pytest.param(
        "dop853",
        ERKStep,
        ERK_TABLEAU_REGISTRY["dop853"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-dop853",
    ),
    pytest.param(
        "ode23t",
        DIRKStep,
        DIRK_TABLEAU_REGISTRY["trapezoidal_dirk"],
        CPUDIRKStep,
        marks=pytest.mark.specific_algos,
        id="dirk-ode23t",
    ),
    pytest.param(
        "radau",
        FIRKStep,
        FIRK_TABLEAU_REGISTRY["radau"],
        CPUFIRKStep,
        marks=pytest.mark.specific_algos,
        id="firk-radau",
    ),
    pytest.param(
        "ode23s",
        GenericRosenbrockWStep,
        ROSENBROCK_TABLEAUS["ode23s"],
        CPURosenbrockWStep,
        marks=pytest.mark.specific_algos,
        id="rosenbrock-ode23s",
    ),
    pytest.param(
        "rk23",
        ERKStep,
        ERK_TABLEAU_REGISTRY["bogacki-shampine-32"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-rk23",
    ),
    pytest.param(
        "ode23",
        ERKStep,
        ERK_TABLEAU_REGISTRY["bogacki-shampine-32"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-ode23",
    ),
    pytest.param(
        "rk45",
        ERKStep,
        ERK_TABLEAU_REGISTRY["dormand-prince-54"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-rk45",
    ),
    pytest.param(
        "ode45",
        ERKStep,
        ERK_TABLEAU_REGISTRY["dormand-prince-54"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-ode45",
    ),
    pytest.param(
        "tsit5",
        ERKStep,
        ERK_TABLEAU_REGISTRY["tsit5"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-tsit5",
    ),
    pytest.param(
        "vern7",
        ERKStep,
        ERK_TABLEAU_REGISTRY["vern7"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-vern7",
    ),
    pytest.param(
        "dormand-prince-853",
        ERKStep,
        ERK_TABLEAU_REGISTRY["dormand-prince-853"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-dormand-prince-853",
    ),
    pytest.param(
        "Tsit5",
        ERKStep,
        ERK_TABLEAU_REGISTRY["Tsit5"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-Tsit5",
    ),
    pytest.param(
        "Vern7",
        ERKStep,
        ERK_TABLEAU_REGISTRY["Vern7"],
        CPUERKStep,
        marks=pytest.mark.specific_algos,
        id="erk-Vern7",
    ),
    pytest.param(
        "firk_gauss_legendre_2",
        FIRKStep,
        FIRK_TABLEAU_REGISTRY["firk_gauss_legendre_2"],
        CPUFIRKStep,
        marks=pytest.mark.specific_algos,
        id="firk-gauss-legendre-2",
    ),
    pytest.param(
        "radau_iia_3",
        FIRKStep,
        FIRK_TABLEAU_REGISTRY["radau_iia_3"],
        CPUFIRKStep,
        marks=pytest.mark.specific_algos,
        id="firk-radau-iia-3",
    ),
    pytest.param(
        "radau_iia_5",
        FIRKStep,
        FIRK_TABLEAU_REGISTRY["radau_iia_5"],
        CPUFIRKStep,
        marks=pytest.mark.specific_algos,
        id="firk-radau-iia-5",
    ),
    pytest.param(
        "radau_iia_9",
        FIRKStep,
        FIRK_TABLEAU_REGISTRY["radau_iia_9"],
        CPUFIRKStep,
        marks=pytest.mark.specific_algos,
        id="firk-radau-iia-9",
    ),
    pytest.param(
        "rodas3p",
        GenericRosenbrockWStep,
        ROSENBROCK_TABLEAUS["rodas3p"],
        CPURosenbrockWStep,
        marks=pytest.mark.specific_algos,
        id="rosenbrock-rodas3p",
    ),
    pytest.param(
        "rosenbrock23",
        GenericRosenbrockWStep,
        ROSENBROCK_TABLEAUS["rosenbrock23"],
        CPURosenbrockWStep,
        marks=pytest.mark.specific_algos,
        id="rosenbrock-23",
    ),
    pytest.param(
        "rosenbrock23_sciml",
        GenericRosenbrockWStep,
        ROSENBROCK_TABLEAUS["rosenbrock23_sciml"],
        CPURosenbrockWStep,
        marks=pytest.mark.specific_algos,
        id="rosenbrock-23-sciml",
    ),
]


@pytest.mark.parametrize(
    "alias_key, expected_step_type, expected_tableau, expected_cpu_step",
    ALIAS_CASES,
)
def test_algorithm_factory_resolves_tableau_alias(
    alias_key,
    expected_step_type,
    expected_tableau,
    expected_cpu_step,
):
    """Algorithm factory should inject the tableau advertised for aliases."""

    step = get_algorithm_step(
        np.float64,
        settings={"algorithm": alias_key, "n": 2, "dt": 1e-3},
        warn_on_unused=False,
    )
    assert isinstance(step, expected_step_type)
    tableau_value = getattr(step, "tableau", None)
    if tableau_value is None:
        tableau_value = step.compile_settings.tableau
    assert tableau_value is expected_tableau


@pytest.mark.parametrize(
    "alias_key, expected_step_type, expected_tableau, expected_cpu_step",
    ALIAS_CASES,
)
def test_cpu_reference_resolves_tableau_alias(
    alias_key,
    expected_step_type,
    expected_tableau,
    expected_cpu_step,
    cpu_system: CPUODESystem,
    cpu_driver_evaluator,
):
    """CPU reference helpers should resolve alias tableaus consistently."""
    factory = get_ref_step_factory(alias_key)
    bound_step = factory(
        cpu_system,
        cpu_driver_evaluator,
        newton_tol=1e-10,
        newton_max_iters=25,
        linear_tol=1e-10,
        linear_max_iters=cpu_system.system.sizes.states,
        linear_correction_type="minimal_residual",
        preconditioner_order=2,
    )

    assert isinstance(bound_step, expected_cpu_step)
    assert bound_step.tableau is expected_tableau

def generate_step_props(n_states: int) -> dict[str, dict[str, Any]]:
    """Generate expected properties for each algorithm given n_states."""
    return {
        "euler": {
            "threads_per_step": 1,
            "persistent_local_buffer_size": 0,
            "is_multistage": False,
            "is_implicit": False,
            "is_adaptive": False,
        },
        "backwards_euler": {
            "threads_per_step": 1,
            "persistent_local_buffer_size": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": False,
        },
        "backwards_euler_pc": {
            "threads_per_step": 1,
            "persistent_local_buffer_size": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": False,
        },
        "crank_nicolson": {
            "threads_per_step": 1,
            "persistent_local_buffer_size": 0,
            "is_multistage": False,
            "is_implicit": True,
            "is_adaptive": True,
        },
        "rosenbrock": {
            "threads_per_step": 1,
            "persistent_local_buffer_size": 0,
            "is_multistage": True,
            "is_implicit": True,
            "is_adaptive": True,
        },
        "erk": {
            "threads_per_step": 1,
            "persistent_local_buffer_size": 0,
            "is_multistage": DEFAULT_ERK_TABLEAU.stage_count > 1,
            "is_implicit": False,
            "is_adaptive": DEFAULT_ERK_TABLEAU.has_error_estimate,
        },
        "dirk": {
            "threads_per_step": 1,
            "persistent_local_buffer_size": 0,
            "is_multistage": DEFAULT_DIRK_TABLEAU.stage_count > 1,
            "is_implicit": True,
            "is_adaptive": DEFAULT_DIRK_TABLEAU.has_error_estimate,
        },
    }

@pytest.fixture(scope="session")
def expected_step_properties(system) -> dict[str, Any]:
    """Generate expected properties for each algorithm given n_states."""
    return generate_step_props(n_states=system.sizes.states)


@pytest.mark.parametrize(
    "solver_settings_override",
    ALGORITHM_PARAM_SETS,
    indirect=True,
)
def test_algorithm(
       step_object_mutable,
       solver_settings,
       system,
       precision,
       expected_step_properties,
       tolerance,
       ) -> None:
    """Ensure the step function is compiled and callable."""
    step_object = step_object_mutable
    # Test that it builds
    assert callable(step_object.step_function), "step_function_builds"

    # test getters
    algorithm = solver_settings[("algorithm")]
    properties = expected_step_properties.get(algorithm)
    if properties is not None:
        assert step_object.is_implicit is properties["is_implicit"], \
            "is_implicit getter"
        assert step_object.is_adaptive is properties["is_adaptive"], \
            "is_adaptive getter"
        assert step_object.is_multistage is properties["is_multistage"],\
            "is_multistage getter"
        assert (
            step_object.threads_per_step == properties["threads_per_step"]
        ), "threads_per_step getter"
    config = step_object.compile_settings
    assert config.n == system.sizes.states, "compile_settings.n getter"
    assert config.precision == precision, "compile_settings.precision getter"

    tableau = getattr(step_object, "tableau", None)
    if tableau is not None and tableau.b_hat is not None:
        expected_error = tuple(
            b_value - b_hat_value
            for b_value, b_hat_value in zip(tableau.b, tableau.b_hat)
        )
        assert tableau.d == pytest.approx(expected_error), "embedded weights"

    expected_order = _expected_order(step_object, tableau)
    assert step_object.order == expected_order, "order getter"


    if properties is not None and properties["is_implicit"]:
        if isinstance(step_object, GenericRosenbrockWStep):
            assert step_object.krylov_max_iters == solver_settings[
                "krylov_max_iters"
            ], "krylov_max_iters set"
            assert step_object.linear_correction_type == solver_settings[
                "linear_correction_type"
            ], "linear_correction_type set"
            assert step_object.krylov_atol == pytest.approx(
                solver_settings["krylov_atol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_atol set"
            assert step_object.krylov_rtol == pytest.approx(
                solver_settings["krylov_rtol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_rtol set"
        else:
            assert step_object.preconditioner_order == solver_settings[
                "preconditioner_order"
            ], "preconditioner order set"
            assert step_object.krylov_max_iters == solver_settings[
                "krylov_max_iters"
            ], "krylov_max_iters set"
            assert step_object.linear_correction_type == solver_settings[
                "linear_correction_type"
            ], "linear_correction_type set"
            assert step_object.newton_max_iters == solver_settings[
                "newton_max_iters"
            ], "newton_max_iters set"
            assert step_object.krylov_atol == pytest.approx(
                solver_settings["krylov_atol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_atol set"
            assert step_object.krylov_rtol == pytest.approx(
                solver_settings["krylov_rtol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_rtol set"
            assert step_object.newton_atol == pytest.approx(
                solver_settings["newton_atol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "newton_atol set"
            # Nonzero newton_rtol reads back at the 4-ULP floor.
            requested_newton_rtol = solver_settings["newton_rtol"]
            newton_rtol_floor = 4.0 * np.finfo(precision).eps
            if requested_newton_rtol > 0.0:
                expected_newton_rtol = max(
                    requested_newton_rtol, newton_rtol_floor
                )
            else:
                expected_newton_rtol = requested_newton_rtol
            assert step_object.newton_rtol == pytest.approx(
                expected_newton_rtol,
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "newton_rtol set"
        assert callable(system.get_solver_helper)

    if step_object.is_implicit:
        if isinstance(step_object, GenericRosenbrockWStep):
            updates = {
                "krylov_max_iters": max(
                    1, solver_settings["krylov_max_iters"] // 2
                ),
                "krylov_atol": solver_settings["krylov_atol"] * 0.5,
                "krylov_rtol": solver_settings["krylov_rtol"] * 0.5,
                "linear_correction_type": "steepest_descent",
            }
            recognised = step_object.update(updates)
            assert set(updates).issubset(recognised), "updates recognised"
            config = step_object.compile_settings
            assert step_object.krylov_max_iters == updates["krylov_max_iters"], \
                "krylov_max_iters update"
            assert step_object.linear_correction_type == updates[
                "linear_correction_type"], "linear_correction_type update"
            assert step_object.krylov_atol == pytest.approx(
                updates["krylov_atol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_atol update"
            assert step_object.krylov_rtol == pytest.approx(
                updates["krylov_rtol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_rtol update"
        else:
            updates = {
                "newton_max_iters": int(
                    max(1, solver_settings["newton_max_iters"] // 2)
                ),
                "krylov_atol":
                solver_settings["krylov_atol"] * 0.5,
                "krylov_rtol":
                solver_settings["krylov_rtol"] * 0.5,
                "newton_atol":
                solver_settings["newton_atol"] * 0.5,
                "newton_rtol":
                solver_settings["newton_rtol"] * 0.5,
                "preconditioner_order":
                solver_settings["preconditioner_order"] + 1,
            }
            recognised = step_object.update(updates)
            assert set(updates).issubset(recognised), "updates recognised"
            config = step_object.compile_settings
            assert step_object.newton_max_iters == updates["newton_max_iters"], \
                "newton_max_iters update"
            assert step_object.preconditioner_order == updates[
                "preconditioner_order"
            ], "preconditioner_order update"
            assert step_object.krylov_atol == pytest.approx(
                updates["krylov_atol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_atol update"
            assert step_object.krylov_rtol == pytest.approx(
                updates["krylov_rtol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "krylov_rtol update"
            assert step_object.newton_atol == pytest.approx(
                updates["newton_atol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "newton_atol update"
            # Nonzero newton_rtol reads back at the 4-ULP floor.
            updated_newton_rtol = updates["newton_rtol"]
            newton_rtol_floor = 4.0 * np.finfo(precision).eps
            if updated_newton_rtol > 0.0:
                expected_newton_rtol = max(
                    updated_newton_rtol, newton_rtol_floor
                )
            else:
                expected_newton_rtol = updated_newton_rtol
            assert step_object.newton_rtol == pytest.approx(
                expected_newton_rtol,
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), "newton_rtol update"


def test_firk_step_is_multistage_matches_tableau():
    """FIRKStep.is_multistage forwards stage_count > 1 from the tableau."""
    step = FIRKStep(
        precision=np.float32, n=3, tableau=DEFAULT_FIRK_TABLEAU,
    )
    assert step.is_multistage == (DEFAULT_FIRK_TABLEAU.stage_count > 1)


@pytest.mark.parametrize(
    "step_class,tableau,norm_type",
    [
        (BackwardsEulerStep, None, DIRKCorrectionNorm),
        (CrankNicolsonStep, None, DIRKCorrectionNorm),
        (DIRKStep, DEFAULT_DIRK_TABLEAU, DIRKCorrectionNorm),
        (FIRKStep, DEFAULT_FIRK_TABLEAU, FIRKCorrectionNorm),
    ],
)
def test_implicit_algorithm_selects_correction_norm(
    step_class, tableau, norm_type
):
    """Each implicit family selects its correction norm."""
    kwargs = {"precision": np.float32, "n": 3}
    if tableau is not None:
        kwargs["tableau"] = tableau
    step = step_class(**kwargs)

    assert isinstance(step.solver.norm, norm_type)


# Test controller defaults selection based on tableau error estimation
@pytest.mark.parametrize(
    "step_class,tableau,expected_dict",
    [
        # ERK errorless tableaus default to fixed
        (ERKStep, ERK_TABLEAU_REGISTRY["classical-rk4"], {"step_controller": "fixed"}),
        (ERKStep, ERK_TABLEAU_REGISTRY["heun-21"], {"step_controller": "fixed"}),
        # ERK adaptive tableaus default to PI
        (ERKStep, ERK_TABLEAU_REGISTRY["dormand-prince-54"],
         {"step_controller": "pi"}),
        (ERKStep, DEFAULT_ERK_TABLEAU, {"step_controller": "pi"}),
        # DIRK without an error estimate defaults to fixed
        (DIRKStep, DIRK_TABLEAU_REGISTRY["sdirk_2_2"],
         {"step_controller": "fixed"}),
        # DIRK with an embedded error estimate defaults to PI
        (DIRKStep, DIRK_TABLEAU_REGISTRY["l_stable_sdirk_4"],
         {"step_controller": "pi"}),
        # FIRK with error estimate defaults to Gustafsson
        (FIRKStep, FIRK_TABLEAU_REGISTRY["radau"],
         {"step_controller": "gustafsson"}),
        # Rosenbrock with error estimate defaults to Gustafsson
        (GenericRosenbrockWStep, DEFAULT_ROSENBROCK_TABLEAU,
         {"step_controller": "gustafsson"}),
    ],
)
def test_tableau_controller_defaults(step_class, tableau, expected_dict):
    """Test that tableaus select appropriate controller defaults."""
    step = step_class(
        precision=np.float32,
        n=3,
        tableau=tableau,
    )

    defaults = step.controller_defaults.step_controller
    for key, expected_value in expected_dict.items():
        assert defaults[key] == expected_value, \
            f"{step_class.__name__} with {tableau} should have {key}={expected_value}"
