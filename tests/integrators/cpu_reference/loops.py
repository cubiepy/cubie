"""CPU reference loop implementations for integrator tests."""

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau

from .algorithms import get_ref_stepper
from .cpu_ode_system import CPUODESystem
from .cpu_utils import Array, DriverEvaluator, STATUS_MASK, _ensure_array
from .step_controllers import CPUAdaptiveController


def _collect_saved_outputs(
    save_history: list[Array],
    output_length: int,
    indices: Sequence[int],
    dtype: np.dtype,
) -> Array:
    """Return the saved samples at ``indices`` as a dense array.

    Parameters
    ----------
    save_history
        Sequence of saved state or observable samples.
    indices
        Column indices to extract from each saved sample.
    dtype
        Target dtype for the dense output array.

    Returns
    -------
    Array
        Dense array containing the selected samples.
    """

    width = len(indices)
    if width == 0:
        return np.zeros((output_length, 0), dtype=dtype)
    data = np.zeros((output_length, width), dtype=dtype)
    for row, sample in enumerate(save_history):
        data[row, :width] = sample[indices]
    return data


def run_reference_loop(
    evaluator: CPUODESystem,
    inputs: Mapping[str, Array],
    driver_evaluator: DriverEvaluator,
    solver_settings,
    output_functions,
    controller: CPUAdaptiveController,
    *,
    tableau: Optional[Union[str, ButcherTableau]] = None,
    step_object: Any = None,
) -> Mapping[str, Array]:
    """Execute a CPU loop mirroring :class:`IVPLoop` behaviour.

    Parameters
    ----------
    evaluator
        CPU-side ODE evaluator used for state updates.
    inputs
        Mapping providing initial conditions and parameters.
    driver_evaluator
        Callable returning driver samples for a requested time.
    solver_settings
        Settings dictionary describing the integration problem.
    output_functions
        Output configuration mirroring the device loop behaviour.
    controller
        Adaptive or fixed step controller used during the run.
    tableau
        Optional tableau override supplied as a name or instance.
    step_object
        Built device step; unset linear-solve keys are read from it.

    Returns
    -------
    Mapping[str, Array]
        Dictionary containing state, observable, summary, and status arrays.
    """

    precision = evaluator.precision

    initial_state = inputs["initial_values"].astype(precision, copy=True)
    params = inputs["parameters"].astype(precision, copy=True)
    duration = np.float64(solver_settings["duration"])
    warmup = np.float64(solver_settings["warmup"])
    t0 = np.float64(solver_settings["t0"])
    save_every = precision(solver_settings["save_every"])
    summarise_every = precision(solver_settings["summarise_every"])
    sample_summaries_every = precision(
        solver_settings.get(
            "sample_summaries_every", solver_settings["save_every"]
        )
    )

    # Mirrors SingleIntegratorRunCore's derivation: an unset
    # reduction follows the adaptive controller's rtol; a
    # pure-absolute controller leaves the floor governing.
    residual_reduction = solver_settings.get("krylov_residual_reduction")
    if (
        residual_reduction is None
        and solver_settings["step_controller"] != "fixed"
    ):
        controller_rtol_floor = float(np.min(solver_settings["rtol"]))
        if controller_rtol_floor > 0.0:
            residual_reduction = controller_rtol_floor

    # Unset inner tolerances follow the controller's (Newton at 1/10).
    controller_atol = float(np.min(solver_settings["atol"]))
    controller_rtol = float(np.min(solver_settings["rtol"]))
    inner_defaults = {
        "newton_atol": controller_atol / 10.0,
        "newton_rtol": controller_rtol / 10.0,
        "krylov_atol": controller_atol,
        "krylov_rtol": controller_rtol,
    }
    inner_tols = {
        key: default if solver_settings.get(key) is None
        else solver_settings[key]
        for key, default in inner_defaults.items()
    }

    stepper = get_ref_stepper(
        evaluator,
        driver_evaluator,
        solver_settings["algorithm"],
        newton_tol=inner_tols["newton_atol"],
        newton_rtol=inner_tols["newton_rtol"],
        newton_max_iters=solver_settings["newton_max_iters"],
        linear_tol=inner_tols["krylov_atol"],
        linear_rtol=inner_tols["krylov_rtol"],
        linear_max_iters=solver_settings["krylov_max_iters"],
        linear_correction_type=solver_settings["linear_correction_type"],
        preconditioner_order=solver_settings["preconditioner_order"],
        residual_reduction=residual_reduction,
        residual_floor=solver_settings.get("krylov_residual_floor"),
        tableau=tableau,
        attempt_dense_prediction=solver_settings[
            "attempt_dense_prediction"
        ],
        use_smoothed_error=solver_settings["use_smoothed_error"],
        step_object=step_object,
    )

    saved_state_indices = _ensure_array(
        solver_settings["saved_state_indices"], np.int32
    )
    saved_observable_indices = _ensure_array(
        solver_settings["saved_observable_indices"], np.int32
    )
    summarised_state_indices = _ensure_array(
        solver_settings["summarised_state_indices"], np.int32
    )
    summarised_observable_indices = _ensure_array(
        solver_settings["summarised_observable_indices"], np.int32
    )

    save_time = output_functions.save_time
    max_save_samples = (
        int(np.floor(precision(duration) / precision(save_every))) + 1
    )

    # Calculate summary sample counts
    max_summary_samples = (
        int(np.floor(precision(duration) / precision(sample_summaries_every)))
        + 1
    )
    samples_per_summary = int(summarise_every / sample_summaries_every)

    state = initial_state.copy()
    state_history = []
    observable_history = []
    time_history = []
    # Separate history for summary calculations when cadences differ
    summary_state_history = []
    summary_observable_history = []

    t = t0
    t32 = precision(t)
    drivers_initial = driver_evaluator(t32)
    observables = evaluator.observables(
        state,
        params,
        drivers_initial,
        t32,
    )

    if warmup > np.float64(0.0):
        next_save_time = precision(warmup + t0)
        next_summary_sample_time = precision(warmup + t0)
        save_idx = 0
    else:
        next_save_time = precision(warmup + t0 + save_every)
        next_summary_sample_time = precision(
            warmup + t0 + sample_summaries_every
        )
        state_history = [state.copy()]
        observable_history.append(observables.copy())
        time_history = [precision(t)]
        save_idx = 1

    end_time = precision(warmup + t0 + duration)

    status_flags = 0
    # Mirrors the device loop's previous-proposal-accepted flag.
    prev_accepted = True

    # Track when we need to sample for summaries vs save for output
    while next_save_time <= end_time or next_summary_sample_time <= end_time:
        dt = controller.dt
        do_save = False
        do_summary_sample = False
        truncated = False

        # Determine next event time
        next_event_time = min(
            next_save_time if next_save_time <= end_time else end_time + 1,
            next_summary_sample_time
            if next_summary_sample_time <= end_time
            else end_time + 1,
        )
        if next_event_time > end_time:
            break

        # An unclamped step would reach the next event.
        if precision(t + dt) >= next_event_time:
            forced_dt = min(precision(next_event_time - t32), dt)
            truncated = bool(forced_dt != dt)
            dt = forced_dt
            if next_event_time == next_save_time:
                do_save = True
            if next_event_time == next_summary_sample_time:
                do_summary_sample = True

        result = stepper.step(
            state=state,
            params=params,
            dt=dt,
            time=t32,
            prev_accepted=prev_accepted,
        )

        step_status = int(result.status)
        status_flags |= step_status & STATUS_MASK
        accept = controller.propose_dt(
            error_vector=result.error,
            prev_state=state,
            new_state=result.state,
            niters=result.niters,
            truncated=truncated,
        )
        prev_accepted = bool(accept)
        if not accept:
            continue

        state = result.state.copy()
        observables = result.observables.copy()
        # A clamped step ends on the event time.
        t = np.float64(next_event_time) if truncated else t + dt
        t32 = precision(t)

        if do_save:
            if len(state_history) < max_save_samples:
                state_history.append(result.state.copy())
                observable_history.append(result.observables.copy())
                time_history.append(precision(t32 - warmup))
            next_save_time = next_save_time + save_every
            save_idx += 1

        if do_summary_sample:
            if len(summary_state_history) < max_summary_samples:
                summary_state_history.append(result.state.copy())
                summary_observable_history.append(result.observables.copy())
            next_summary_sample_time = (
                next_summary_sample_time + sample_summaries_every
            )

    state_output = _collect_saved_outputs(
        state_history,
        max_save_samples,
        saved_state_indices,
        precision,
    )

    observables_output = _collect_saved_outputs(
        observable_history,
        max_save_samples,
        saved_observable_indices,
        precision,
    )
    if save_time:
        if len(time_history) < state_output.shape[0]:
            time_history.append(precision(0))
        state_output = np.column_stack(
            (state_output, np.asarray(time_history))
        )

    # Use summary-cadence data for summary calculations
    summary_state_output = _collect_saved_outputs(
        summary_state_history,
        max_summary_samples,
        summarised_state_indices,
        precision,
    )
    summary_observable_output = _collect_saved_outputs(
        summary_observable_history,
        max_summary_samples,
        summarised_observable_indices,
        precision,
    )

    # Import has been dumped here to avoid a circular dependency that it's
    # easier not to fix.
    from tests._utils import calculate_expected_summaries

    state_summary, observable_summary = calculate_expected_summaries(
        summary_state_output,
        summary_observable_output,
        np.arange(len(summarised_state_indices), dtype=np.int32),
        np.arange(len(summarised_observable_indices), dtype=np.int32),
        samples_per_summary,
        output_functions.compile_settings.output_types,
        output_functions.summaries_output_height_per_var,
        precision,
        sample_summaries_every=sample_summaries_every,
    )
    final_status = status_flags & STATUS_MASK

    return {
        "state": state_output,
        "observables": observables_output,
        "state_summaries": state_summary,
        "observable_summaries": observable_summary,
        "status": final_status,
    }


__all__ = ["run_reference_loop", "_collect_saved_outputs"]
