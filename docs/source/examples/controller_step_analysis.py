"""Visualise CPU reference controller step-size behaviour across systems.

This module recreates the CPU loop used in the integration tests so we can
inspect how the adaptive step-size controllers behave for a selection of ODE
systems.  The script mirrors the fixtures from :mod:`tests.conftest` closely,
particularly the solver configuration dictionaries and the driver generation
helpers.  Running the module will produce one figure per controller type where
accepted steps are plotted in blue and rejected steps in red for each system.

The implementation intentionally favours clarity over performance so that the
behaviour matches the reference implementation used in the tests.
"""

import attrs
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from cubie.array_interpolator import ArrayInterpolator
from cubie.memory import default_memmgr
from cubie.outputhandling.output_functions import OutputFunctions
from tests._utils import _driver_sequence
from tests.integrators.cpu_reference import (
    Array,
    CPUAdaptiveController,
    CPUODESystem,
    DriverEvaluator,
    STATUS_MASK,
    _collect_saved_outputs,
    get_ref_stepper,
)
from tests.system_fixtures import (
    build_three_state_nonlinear_system,
    build_three_state_very_stiff_system,
)

ArrayFloat = NDArray[np.floating[Any]]


@attrs.define
class StepRecord:
    """Time-stamped record of a proposed step.

    Parameters
    ----------
    time : float
        Simulation time when the step was proposed.
    step_size : float
        Proposed step size in seconds.
    accepted : bool
        Whether the controller accepted the step.
    run_label : str
        Identifier describing which Monte-Carlo run produced the record.
    attempt_index : int
        Monotonic counter of how many proposals have been made.
    driver_values : tuple(float, ...)
        Driver sample(s) at proposal time. Empty when system has no drivers.
    """

    time: float
    step_size: float
    accepted: bool
    run_label: str
    attempt_index: int
    driver_values: Tuple[float, ...] = ()


SYSTEM_BUILDERS: Dict[str, Any] = {
    # "large": build_large_nonlinear_system,
    # "threecm": build_three_chamber_system,
    "nonlinear": build_three_state_nonlinear_system,
    "stiff": build_three_state_very_stiff_system,
}
SYSTEM_LABELS: Dict[str, str] = {
    # "large": "Large",
    # "threecm": "Three-chamber",
    "nonlinear": "Nonlinear",
    "stiff": "Stiff",
}
SYSTEM_ORDER: Tuple[str, ...] = tuple(SYSTEM_BUILDERS.keys())


def build_solver_settings(precision: type[np.floating[Any]]) -> Dict[str, Any]:
    """Return solver configuration aligned with `tests.conftest` and
    `tests.integrators.loops.test_ode_loop` defaults.

    """

    defaults: Dict[str, Any] = {
        "algorithm": "sdirk_2_2",
        "duration": precision(1.0),
        "warmup": precision(0.0),
        "dt": precision(0.001953125),
        "dt_min": precision(1e-9),
        "dt_max": precision(0.5),
        "save_every": precision(0.01),
        "summarise_every": precision(0.2),
        "atol": precision(1e-5),
        "rtol": precision(1e-6),
        "saved_state_indices": [0, 1, 2],
        "saved_observable_indices": [0, 1],
        "summarised_state_indices": [0, 1],
        "summarised_observable_indices": [0, 1],
        "output_types": ["state", "time"],
        "blocksize": 32,
        "stream": 0,
        "lineinfo": False,
        "memory_manager": default_memmgr,
        "stream_group": "test_group",
        "mem_proportion": None,
        "step_controller": "fixed",
        "precision": precision,
        "driverspline_order": 3,
        "driverspline_wrap": False,
        "driverspline_boundary_condition": "clamped",
        "krylov_atol": precision(1e-7),
        "krylov_rtol": precision(1e-7),
        "linear_correction_type": "minimal_residual",
        "newton_atol": precision(1e-7),
        "newton_rtol": precision(1e-7),
        "preconditioner_order": 2,
        "krylov_max_iters": 500,
        "newton_max_iters": 500,
        "newton_target_iters": 20,
        "min_gain": precision(0.2),
        "max_gain": precision(2.0),
        "safety": precision(0.9),
        "kp": precision(0.7),
        "ki": precision(-0.4),
        "kd": precision(0.0),
        "deadband_min": precision(1.0),
        "deadband_max": precision(1.0),
    }
    return defaults


def build_implicit_step_settings(
    solver_settings: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the implicit solver defaults from the fixtures.

    Parameters
    ----------
    solver_settings
        Mapping describing the solver configuration.

    Returns
    -------
    dict
        Dictionary of implicit solver helper settings.
    """

    return {
        "atol": solver_settings["atol"],
        "rtol": solver_settings["rtol"],
        "krylov_atol": solver_settings["krylov_atol"],
        "krylov_rtol": solver_settings["krylov_rtol"],
        "newton_atol": solver_settings["newton_atol"],
        "newton_rtol": solver_settings["newton_rtol"],
        "linear_correction_type": solver_settings["linear_correction_type"],
        "preconditioner_order": solver_settings["preconditioner_order"],
        "krylov_max_iters": solver_settings["krylov_max_iters"],
        "newton_max_iters": solver_settings["newton_max_iters"],
        "tableau": solver_settings.get("tableau", "sdirk_2_2"),
    }


def build_step_controller_settings(
    solver_settings: Mapping[str, Any],
    system: Any,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the adaptive controller settings, applying overrides.

    Parameters
    ----------
    solver_settings
        Base solver configuration.
    system
        Symbolic system instance providing shape information.
    overrides
        Optional mapping of override values.

    Returns
    -------
    dict
        Controller settings compatible with ``CPUAdaptiveController``.
    """

    precision = solver_settings["precision"]
    base_settings: Dict[str, Any] = {
        "kind": solver_settings["step_controller"].lower(),
        "dt": precision(solver_settings["dt"]),
        "dt_min": precision(solver_settings["dt_min"]),
        "dt_max": precision(solver_settings["dt_max"]),
        "atol": precision(solver_settings["atol"]),
        "rtol": precision(solver_settings["rtol"]),
        "order": 5,
        "min_gain": precision(solver_settings["min_gain"]),
        "max_gain": precision(solver_settings["max_gain"]),
        "kp": precision(solver_settings["kp"]),
        "ki": precision(solver_settings["ki"]),
        "kd": precision(solver_settings["kd"]),
        "deadband_min": precision(solver_settings["deadband_min"]),
        "deadband_max": precision(solver_settings["deadband_max"]),
        "newton_target_iters": int(
            solver_settings["newton_target_iters"]
        ),
        "safety": precision(solver_settings["safety"]),
    }

    if overrides:
        float_keys = {
            "dt",
            "dt_min",
            "dt_max",
            "atol",
            "rtol",
            "min_gain",
            "max_gain",
            "kp",
            "ki",
            "kd",
            "deadband_min",
            "deadband_max",
        }
        for key, value in overrides.items():
            if key in float_keys:
                base_settings[key] = precision(value)
            else:
                base_settings[key] = value

    return base_settings


def build_driver_settings(
    solver_settings: Mapping[str, Any],
    system: Any,
    precision: type[np.floating[Any]],
) -> Optional[Dict[str, Any]]:
    """Create the driver settings dictionary used by the fixtures.

    Parameters
    ----------
    solver_settings
        Mapping describing solver behaviour.
    system
        Symbolic system providing driver metadata.
    precision
        Floating-point dtype for generated arrays.

    Returns
    -------
    dict or None
        Dictionary consumed by :class:`ArrayInterpolator`, or ``None`` when the
        system has no drivers.
    """

    if system.num_drivers == 0:
        return None

    dt_sample = precision(solver_settings["save_every"]) / 2.0
    total_span = precision(solver_settings["duration"])  # match test fixtures
    t0 = precision(solver_settings["warmup"])  # align with conftest driver t0
    order = int(solver_settings["driverspline_order"])

    samples = int(np.ceil(total_span / dt_sample)) + 1
    samples = max(samples, order + 1)
    # use precision arithmetic like the tests
    total_time = precision(dt_sample) * max(samples - 1, 1)

    driver_matrix = _driver_sequence(
        samples=samples,
        total_time=total_time,
        n_drivers=system.num_drivers,
        precision=precision,
    )

    driver_names = list(system.indices.driver_names)
    drivers_dict: Dict[str, Any] = {
        name: np.array(driver_matrix[:, idx], dtype=precision, copy=True)
        for idx, name in enumerate(driver_names)
    }
    drivers_dict["driver_sample_period"] = precision(dt_sample)
    drivers_dict["wrap"] = bool(solver_settings["driverspline_wrap"])
    drivers_dict["boundary_condition"] = solver_settings.get(
        "driverspline_boundary_condition",
        solver_settings.get("driverspline_end_condition", "clamped"),
    )
    drivers_dict["order"] = order
    drivers_dict["t0"] = t0

    return drivers_dict


def create_output_functions(
    system: Any, solver_settings: Mapping[str, Any]
) -> OutputFunctions:
    """Build :class:`OutputFunctions` mirroring the fixture helper.

    Parameters
    ----------
    system
        Symbolic system supplying size information.
    solver_settings
        Solver configuration mapping.

    Returns
    -------
    OutputFunctions
        Configured output handler instance.
    """

    return OutputFunctions(
        system.sizes.states,
        system.sizes.parameters,
        solver_settings["output_types"],
        solver_settings["saved_state_indices"],
        solver_settings["saved_observable_indices"],
        solver_settings["summarised_state_indices"],
        solver_settings["summarised_observable_indices"],
    )


def create_controller(
    settings: Mapping[str, Any], precision: type[np.floating[Any]]
) -> CPUAdaptiveController:
    """Instantiate a CPU controller matching the GPU configuration.

    Parameters
    ----------
    settings
        Controller configuration mapping.
    precision
        Floating-point dtype for the controller internals.

    Returns
    -------
    CPUAdaptiveController
        Configured controller instance.
    """

    controller = CPUAdaptiveController(
        kind=settings["kind"],
        dt=settings["dt"],
        dt_min=settings["dt_min"],
        dt_max=settings["dt_max"],
        atol=settings["atol"],
        rtol=settings["rtol"],
        order=settings["order"],
        precision=precision,
        min_gain=settings["min_gain"],
        max_gain=settings["max_gain"],
        deadband_min=settings["deadband_min"],
        deadband_max=settings["deadband_max"],
    )
    kind = settings["kind"].lower()
    if kind == "pi":
        controller.kp = settings["kp"]
        controller.ki = settings["ki"]
    elif kind == "pid":
        controller.kp = settings["kp"]
        controller.ki = settings["ki"]
        controller.kd = settings["kd"]
    return controller


def create_driver_array(
    driver_settings: Optional[Mapping[str, Any]],
    precision: type[np.floating[Any]],
) -> Optional[ArrayInterpolator]:
    """Instantiate :class:`ArrayInterpolator` when drivers are present.

    Parameters
    ----------
    driver_settings
        Mapping produced by :func:`build_driver_settings`.
    precision
        Floating-point dtype for interpolation coefficients.

    Returns
    -------
    ArrayInterpolator or None
        Interpolator matching the fixture behaviour.
    """

    if driver_settings is None:
        return None
    return ArrayInterpolator(precision=precision, input_dict=driver_settings)


def create_driver_evaluator(
    solver_settings: Mapping[str, Any],
    system: Any,
    precision: type[np.floating[Any]],
    driver_array: Optional[ArrayInterpolator],
) -> DriverEvaluator:
    """Create a driver evaluator based on the configured splines.

    Parameters
    ----------
    solver_settings
        Mapping describing solver configuration.
    system
        Symbolic system providing driver metadata.
    precision
        Floating-point dtype for generated arrays.
    driver_array
        Optional interpolator used to compute spline coefficients.

    Returns
    -------
    DriverEvaluator
        CPU-side evaluator matching the fixtures.
    """

    width = system.num_drivers
    order = int(solver_settings["driverspline_order"])
    if driver_array is None or width == 0:
        coeffs = np.zeros((1, width, order + 1), dtype=precision)
        dt_value = precision(solver_settings["save_every"]) / 2.0
        t0_value = precision(0.0)
        wrap_value = bool(solver_settings["driverspline_wrap"])
        boundary = None
    else:
        coeffs = np.array(
            driver_array.coefficients, dtype=precision, copy=True
        )
        dt_value = precision(driver_array.driver_sample_period)
        t0_value = precision(driver_array.t0)
        wrap_value = bool(driver_array.wrap)
        boundary = driver_array.boundary_condition

    return DriverEvaluator(
        coefficients=coeffs,
        dt=dt_value,
        t0=t0_value,
        wrap=wrap_value,
        precision=precision,
        boundary_condition=boundary,
    )


def generate_inputs(
    solver_settings: Mapping[str, Any],
    system: Any,
    precision: type[np.floating[Any]],
    initial_state: ArrayFloat,
    parameters: ArrayFloat,
    driver_array: Optional[ArrayInterpolator],
) -> Dict[str, ArrayFloat]:
    """Create the inputs dictionary consumed by the reference loop.

    Parameters
    ----------
    solver_settings
        Mapping describing solver behaviour.
    system
        Symbolic system providing driver metadata.
    precision
        Floating-point dtype for generated arrays.
    initial_state
        Initial condition for the run.
    parameters
        Parameter vector for the run.
    driver_array
        Optional driver interpolator providing spline coefficients.

    Returns
    -------
    dict
        Inputs compatible with :func:`run_reference_loop`.
    """

    inputs: Dict[str, ArrayFloat] = {
        "initial_values": np.asarray(initial_state, dtype=precision).copy(),
        "parameters": np.asarray(parameters, dtype=precision).copy(),
    }
    if driver_array is not None:
        inputs["driver_coefficients"] = np.asarray(
            driver_array.coefficients, dtype=precision
        ).copy()
    else:
        width = system.num_drivers
        order = int(solver_settings["driverspline_order"])
        inputs["driver_coefficients"] = np.zeros(
            (1, width, order + 1), dtype=precision
        )
    return inputs


def run_reference_loop_with_history(
    evaluator: CPUODESystem,
    inputs: Mapping[str, Array],
    solver_settings: Mapping[str, Any],
    implicit_step_settings: Mapping[str, Any],
    controller: CPUAdaptiveController,
    driver_evaluator: DriverEvaluator,
    output_functions: OutputFunctions,
    *,
    run_label: str,
) -> Tuple[Dict[str, Array], List[StepRecord]]:
    """Execute a CPU loop while recording proposed step sizes.

    Parameters
    ----------
    evaluator
        CPU representation of the symbolic system.
    inputs
        Dictionary of initial values, parameters, and driver coefficients.
    solver_settings
        Mapping describing solver behaviour.
    implicit_step_settings
        Helper settings used by implicit algorithms.
    controller
        Adaptive or fixed step controller instance.
    driver_evaluator
        Callable returning driver samples for requested times.
    output_functions
        Output configuration mirroring the GPU loop.
    run_label
        Identifier describing the Monte-Carlo variant.

    Returns
    -------
    tuple
        Tuple containing loop outputs and the recorded step history.
    """

    precision = evaluator.precision
    initial_state = inputs["initial_values"].astype(precision, copy=True)
    params = inputs["parameters"].astype(precision, copy=True)
    driver_coefficients = inputs.get("driver_coefficients")
    if driver_coefficients is not None:
        evaluate_driver_at_t = driver_evaluator.with_coefficients(
            np.asarray(driver_coefficients, dtype=precision)
        )
    else:
        evaluate_driver_at_t = driver_evaluator

    duration = precision(solver_settings["duration"])
    warmup = precision(solver_settings["warmup"])
    save_every = precision(solver_settings["save_every"])
    status_flags = 0
    zero = precision(0.0)
    step_records: List[StepRecord] = []

    tableau = implicit_step_settings.get("tableau")
    stepper = get_ref_stepper(
        evaluator,
        evaluate_driver_at_t,
        solver_settings["algorithm"],
        newton_tol=implicit_step_settings["newton_atol"],
        newton_max_iters=implicit_step_settings["newton_max_iters"],
        linear_tol=implicit_step_settings["krylov_atol"],
        linear_max_iters=implicit_step_settings["krylov_max_iters"],
        linear_correction_type=implicit_step_settings[
            "linear_correction_type"
        ],
        preconditioner_order=implicit_step_settings["preconditioner_order"],
        tableau=tableau,
    )

    saved_state_indices = np.asarray(
        solver_settings["saved_state_indices"], dtype=np.int32
    )

    save_time = output_functions.save_time
    max_save_samples = int(np.floor(duration / save_every)) + 1

    state = initial_state.copy()
    state_history = [state.copy()]
    observable_history: List[Array] = []
    time_history: List[float] = []
    t = precision(0.0)
    controller.dt = controller.dt0
    drivers_initial = evaluate_driver_at_t(precision(t))
    observables = evaluator.observables(
        state,
        params,
        drivers_initial,
        t,
    )

    if warmup > zero:
        next_save_time = warmup
    else:
        next_save_time = save_every
        observable_history.append(observables.copy())
        time_history.append(float(t))

    end_time = precision(warmup + duration)
    equality_breaker = (
        precision(1e-7) if precision is np.float32 else precision(1e-14)
    )
    attempt_index = 0

    while t < end_time - equality_breaker:
        dt = precision(min(controller.dt, end_time - t))
        do_save = False
        if t + dt >= next_save_time:
            dt = precision(next_save_time - t)
            do_save = True

        try:
            sampled = evaluate_driver_at_t(precision(t))
            sampled_tuple: Tuple[float, ...] = tuple(
                np.asarray(sampled, dtype=float).tolist()
            )
        except Exception:
            sampled_tuple = ()

        result = stepper.step(
            state=state,
            params=params,
            dt=dt,
            time=precision(t),
        )

        step_status = int(result.status)
        status_flags |= step_status & STATUS_MASK
        accept = controller.propose_dt(
            error_vector=result.error,
            prev_state=state,
            new_state=result.state,
            niters=result.niters,
        )
        step_records.append(
            StepRecord(
                time=float(t),
                step_size=float(dt),
                accepted=accept,
                run_label=run_label,
                attempt_index=attempt_index,
                driver_values=sampled_tuple,
            )
        )
        attempt_index += 1
        if not accept:
            continue

        state = result.state.copy()
        observables = result.observables.copy()
        t += precision(dt)

        if do_save:
            if len(state_history) < max_save_samples:
                state_history.append(result.state.copy())
                observable_history.append(result.observables.copy())
                time_history.append(float(next_save_time - warmup))
            next_save_time += save_every

    state_output = _collect_saved_outputs(
        state_history, max_save_samples, saved_state_indices, precision
    )
    if save_time and time_history:
        state_output = np.column_stack(
            (state_output, np.asarray(time_history, dtype=precision))
        )
    final_status = status_flags & STATUS_MASK

    outputs = {
        "state": state_output,
        "status": final_status,
        "state_history": np.asarray(state_history, dtype=precision),
        "time_history": np.asarray(time_history, dtype=precision),
    }
    return outputs, step_records


def plot_step_histories(
    controller: str,
    driver_evaluator,
    histories: Mapping[str, Sequence[StepRecord]],
    state_histories: Optional[Mapping[str, Sequence[tuple]]] = None,
) -> None:
    """Render the controller-specific figure showing accepted and rejected steps.

    Modified to produce two rows per system (step-size history above, states
    + drivers below) and to add a grid to the state plot.
    """

    n_systems = len(SYSTEM_ORDER)
    fig, axes = plt.subplots(2 * n_systems, 1, figsize=(10, 3 * 2 * n_systems))
    axes = np.atleast_1d(axes)
    controller_label = controller.upper()

    for idx, system_name in enumerate(SYSTEM_ORDER):
        top_ax = axes[2 * idx]
        bottom_ax = axes[2 * idx + 1]
        records = histories.get(system_name, [])
        accepted = [record for record in records if record.accepted]
        rejected = [record for record in records if not record.accepted]

        # Step-size history (top row)
        if rejected:
            top_ax.scatter(
                [record.time for record in rejected],
                [record.step_size for record in rejected],
                color="tab:red",
                alpha=0.7,
                marker="x",
                s=18,
                label="rejected",
            )
        if accepted:
            top_ax.scatter(
                [record.time for record in accepted],
                [record.step_size for record in accepted],
                color="tab:blue",
                alpha=0.7,
                marker="o",
                s=18,
                label="accepted",
            )

        top_ax.set_title(SYSTEM_LABELS.get(system_name, system_name))
        top_ax.set_xlabel("Time (s)")
        top_ax.set_ylabel("Step size (s)")
        top_ax.set_yscale("log")
        top_ax.grid(True, linestyle=":", linewidth=0.6)
        if accepted and rejected:
            top_ax.legend(loc="best")

        # State + driver plot (bottom row)
        entry = (
            state_histories.get(system_name, None) if state_histories else None
        )
        if entry is not None:
            variants = entry.get("variants", [])
            if variants:
                times, states = variants[0]
                times_arr = np.asarray(times, dtype=float)
                states_arr = np.asarray(states, dtype=float)

                # plot states if shape matches (saved per save_every)
                if (
                    states_arr.ndim == 2
                    and states_arr.shape[0] == times_arr.size
                ):
                    n_states = states_arr.shape[1]
                    cmap = plt.get_cmap("tab10")
                    for si in range(n_states):
                        bottom_ax.plot(
                            times_arr,
                            states_arr[:, si],
                            label=f"{SYSTEM_LABELS.get(system_name, system_name)} state {si}",
                            color=cmap(si % 10),
                            linewidth=1.2,
                        )

                # evaluate drivers across the saved times and overlay
                n_drivers = driver_evaluator._width
                drivers_arr = np.zeros((n_drivers, times_arr.shape[0]))
                for i in range(times_arr.shape[0]):
                    drivers_arr[:, i] = driver_evaluator(times_arr[i])

                for di in range(drivers_arr.shape[0]):
                    ydrv = drivers_arr[di, :]
                    bottom_ax.plot(
                        times_arr,
                        ydrv,
                        linestyle=":",
                        marker=None,
                        alpha=0.8,
                        label=f"{SYSTEM_LABELS.get(system_name, system_name)} driver {di}",
                    )

        bottom_ax.set_xlabel("Time (s)")
        bottom_ax.set_ylabel("Value")
        bottom_ax.grid(
            True, linestyle=":", linewidth=0.6
        )  # add grid to state plot
        bottom_ax.legend(loc="best")

    fig.suptitle(
        f"Step-size history for {controller_label} controller", fontsize=14
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    plt.show()


def run_analysis() -> Dict[str, Dict[str, List[StepRecord]]]:
    """Execute integrations for every controller and system combination."""

    all_histories: Dict[str, Dict[str, List[StepRecord]]] = {}
    for controller in ("i", "pi", "pid", "gustafsson"):
        controller_histories: Dict[str, List[StepRecord]] = {}
        # per-system state + driver info used by the plotting helper
        controller_state_histories: Dict[str, dict] = {}

        driver_evaluator = None
        for system_name, builder in SYSTEM_BUILDERS.items():
            print(
                "Running analysis for "
                f"{controller} controller on {system_name}"
            )
            precision = np.float64 if system_name == "stiff" else np.float32
            system = builder(precision)
            solver_settings = build_solver_settings(precision)
            solver_settings = {
                **solver_settings,
                "step_controller": controller,
            }
            step_settings = build_step_controller_settings(
                solver_settings, system
            )
            implicit_settings = build_implicit_step_settings(solver_settings)
            output_functions = create_output_functions(system, solver_settings)
            cpu_system = CPUODESystem(system)

            driver_settings = build_driver_settings(
                solver_settings, system, precision
            )
            driver_array = create_driver_array(driver_settings, precision)
            driver_evaluator = create_driver_evaluator(
                solver_settings,
                system,
                precision,
                driver_array,
            )

            histories: List[StepRecord] = []
            state_variants: List[tuple] = []
            # Use a single deterministic variant matching the test fixtures:
            # initial values and parameters come from the symbolic system
            print("  Running single deterministic variant (matches test_loop)")
            initial_state = system.initial_values.values_array.astype(
                precision, copy=True
            )
            parameters = system.parameters.values_array.astype(
                precision, copy=True
            )
            label = "run1"
            print("    initial_state:", initial_state)

            inputs = generate_inputs(
                solver_settings=solver_settings,
                system=system,
                precision=precision,
                initial_state=initial_state,
                parameters=parameters,
                driver_array=driver_array,
            )
            controller_instance = create_controller(step_settings, precision)
            outputs, step_records = run_reference_loop_with_history(
                evaluator=cpu_system,
                inputs=inputs,
                solver_settings=solver_settings,
                implicit_step_settings=implicit_settings,
                controller=controller_instance,
                driver_evaluator=driver_evaluator,
                output_functions=output_functions,
                run_label=label,
            )
            histories.extend(step_records)
            # collect raw saved state/time for plotting
            state_variants.append(
                (
                    outputs.get("time_history", np.array([])),
                    outputs.get("state_history", np.array([])),
                )
            )

            controller_histories[system_name] = histories
            controller_state_histories[system_name] = {
                "variants": state_variants,
            }

        all_histories[controller] = controller_histories
        plot_step_histories(
            controller,
            driver_evaluator,
            controller_histories,
            controller_state_histories,
        )


def main() -> Dict[str, Dict[str, List[StepRecord]]]:
    """Run the analysis pipeline and return the recorded histories."""
    return run_analysis()


if __name__ == "__main__":
    main()
