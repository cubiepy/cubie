"""Test comprehensive configuration plumbing through solver update.

This module tests that every compile-sensitive setting is correctly propagated
through the solver hierarchy when update() is called.
"""

import pytest

from cubie.batchsolving.BatchSolverConfig import ActiveOutputs


def extend_expected_settings(settings, precision):
    """Extend settings dict with derived/computed values.

    Parameters
    ----------
    settings
        Base settings dictionary.
    precision
        Numpy precision type.

    Returns
    -------
    dict
        Extended settings with all derived values.
    """
    extended = settings.copy()

    # Compute compile_flags based on output_types
    output_types = settings.get("output_types", [])
    summary_types = {"mean", "max", "min", "rms", "std"}
    has_summaries = any(t in output_types for t in summary_types)
    extended["save_state"] = "state" in output_types
    extended["save_observables"] = "observables" in output_types
    extended["save_time"] = "time" in output_types
    extended["save_counters"] = "iteration_counters" in output_types
    extended["summarise"] = has_summaries
    extended["summarise_state"] = has_summaries
    has_obs_indices = bool(settings.get("summarised_observable_indices", []))
    extended["summarise_observables"] = has_summaries and has_obs_indices
    extended["ActiveOutputs"] = ActiveOutputs(
        state="state" in output_types,
        observables="observables" in output_types,
        state_summaries=has_summaries,
        observable_summaries=(has_summaries and has_obs_indices),
        status_codes=True,
        iteration_counters="iteration_counters" in output_types,
    )
    # is_adaptive depends on step_controller type
    extended["is_adaptive"] = (
        settings.get("step_controller", "fixed") != "fixed"
    )

    # Always include duration, warmup, t0 in expected settings
    extended.setdefault("duration", 0.0)
    extended.setdefault("warmup", 0.0)
    extended.setdefault("t0", 0.0)

    # Compute expected indices based on output_types and summaries
    # If 'state' in output_types, saved_state_indices = settings value, else []
    if "state" in output_types:
        extended["saved_state_indices"] = settings.get(
            "saved_state_indices", []
        )
    else:
        extended["saved_state_indices"] = []

    # If 'observables' in output_types, saved_observable_indices = settings
    # value, else []
    if "observables" in output_types:
        extended["saved_observable_indices"] = settings.get(
            "saved_observable_indices", []
        )
    else:
        extended["saved_observable_indices"] = []

    # If has_summaries, summarised_state_indices = settings value, else []
    if has_summaries:
        extended["summarised_state_indices"] = settings.get(
            "summarised_state_indices", []
        )
    else:
        extended["summarised_state_indices"] = []

    # If has_summaries, summarised_observable_indices = settings value, else []
    if has_summaries:
        extended["summarised_observable_indices"] = settings.get(
            "summarised_observable_indices", []
        )
    else:
        extended["summarised_observable_indices"] = []

    return extended


def assert_solver_config(solver, settings, tolerance):
    """Assert that solver attributes match expected settings.

    ALL solver properties are checked. Properties not derived from settings
    are documented with comments explaining why they're not tested.
    """
    # Check if adaptive or fixed step
    is_adaptive = solver.kernel.single_integrator.is_adaptive

    # Direct solver properties from settings For adaptive controllers, dt is
    # recomputed from atol/rtol, so we don't check it
    if not is_adaptive:
        assert solver.dt == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )

    expected_dt_min = settings["dt_min"] if is_adaptive else settings["dt"]
    assert solver.dt_min == pytest.approx(
        expected_dt_min, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )

    expected_dt_max = settings["dt_max"] if is_adaptive else settings["dt"]
    assert solver.dt_max == pytest.approx(
        expected_dt_max, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )

    assert solver.save_every == pytest.approx(
        settings["save_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )

    assert solver.summarise_every == pytest.approx(
        settings["summarise_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )

    # Always check duration, warmup, t0
    assert solver.duration == pytest.approx(
        settings["duration"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert solver.warmup == pytest.approx(
        settings["warmup"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert solver.t0 == pytest.approx(
        settings["t0"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )

    assert solver.algorithm == settings["algorithm"]
    assert solver.precision == settings["precision"]
    assert solver.output_types == tuple(settings["output_types"])

    # mem_proportion can be recomputed internally, so check it's in valid range
    if settings.get("mem_proportion") is not None:
        assert solver.mem_proportion > 0.0
        assert solver.mem_proportion <= 1.0

    # memory_manager - checked but may be None stream_group - cannot be updated
    # after construction, so we don't assert changes

    # Saved and summarised indices - compare directly to expected indices from
    # extended settings
    assert list(solver.saved_state_indices) == settings["saved_state_indices"]
    assert (
        list(solver.saved_observable_indices)
        == settings["saved_observable_indices"]
    )
    assert (
        list(solver.summarised_state_indices)
        == settings["summarised_state_indices"]
    )
    assert (
        list(solver.summarised_observable_indices)
        == settings["summarised_observable_indices"]
    )


def assert_solverkernel_config(kernel, settings, tolerance):
    """Assert that BatchSolverKernel attributes match expected settings.

    ALL kernel properties and compile_settings attributes are checked.
    """
    # Direct kernel properties Note: kernel.dt might not exist for all
    # algorithms - check if it's adaptive/fixed
    is_adaptive = settings["is_adaptive"]

    # dt is not set for adaptive controllers (it's None)
    if not is_adaptive:
        assert kernel.dt == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )

    assert kernel.duration == pytest.approx(
        settings["duration"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert kernel.warmup == pytest.approx(
        settings["warmup"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert kernel.t0 == pytest.approx(
        settings["t0"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )

    assert kernel.algorithm == settings["algorithm"]
    assert kernel.save_every == pytest.approx(
        settings["save_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert kernel.summarise_every == pytest.approx(
        settings["summarise_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert kernel.output_types == tuple(settings["output_types"])

    # Saved and summarised indices - compare directly to expected indices from
    # extended settings
    assert list(kernel.saved_state_indices) == settings["saved_state_indices"]
    assert (
        list(kernel.saved_observable_indices)
        == settings["saved_observable_indices"]
    )
    assert (
        list(kernel.summarised_state_indices)
        == settings["summarised_state_indices"]
    )
    assert (
        list(kernel.summarised_observable_indices)
        == settings["summarised_observable_indices"]
    )

    # Check compile_settings.ActiveOutputs
    kernel_active = kernel.active_outputs
    expected_active = settings["ActiveOutputs"]

    assert kernel_active == expected_active

    # Check ALL compile_settings attributes
    assert kernel.compile_settings is not None
    # loop_fn is the compiled function - not tested as it's not a setting

    # Not tested from kernel properties (computed/runtime values):
    # - atol, rtol: None for fixed-step
    # - cache_valid: runtime state
    # - chunks: batching configuration, not settings
    # - device_* arrays: runtime GPU memory, not settings
    # - driver_coefficients: runtime data
    # - input_arrays, output_arrays: runtime data structures
    # - iteration_counters, status_codes: runtime outputs
    # - num_runs: computed from input
    # - observables, state, state_summaries, observable_summaries: runtime
    # outputs
    # - output_*, summaries_*: computed from settings
    # - parameters, initial_values: runtime input data
    # - lineinfo: compile setting on every CUDAFactoryConfig
    # - save_time: derived from output configuration
    # - shared_memory_bytes, shared_memory_needs_padding: computed
    # - single_integrator: tested separately
    # - stream: CUDA stream, runtime
    # - system, system_sizes: tested separately
    # - threads_per_loop: computed
    # - warmup_length, output_length, summaries_length: computed


def assert_singleintegratorrun_config(single_integrator, settings, tolerance):
    """Assert that SingleIntegratorRun attributes match expected settings.

    ALL properties and compile_settings attributes are checked.
    """
    is_adaptive = settings["is_adaptive"]

    # For adaptive, dt is computed from atol/rtol, not from settings["dt"]
    if not is_adaptive:
        assert single_integrator.dt == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )

    expected_dt_min = settings["dt_min"] if is_adaptive else settings["dt"]
    assert single_integrator.dt_min == pytest.approx(
        expected_dt_min, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )

    expected_dt_max = settings["dt_max"] if is_adaptive else settings["dt"]
    assert single_integrator.dt_max == pytest.approx(
        expected_dt_max, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )

    # Check adaptive-specific tolerances
    if is_adaptive:
        assert single_integrator.atol == pytest.approx(
            settings["atol"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
        assert single_integrator.rtol == pytest.approx(
            settings["rtol"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )

    # Check compile_settings
    cs = single_integrator.compile_settings
    assert cs.algorithm == settings["algorithm"]
    assert cs.step_controller == settings["step_controller"]

    # Check precision
    assert single_integrator.precision == settings["precision"]

    # Not tested (runtime/computed values):
    # - numba_precision, algorithm_key: aliases/derived
    # - shared_memory_elements, persistent_local_elements: computed
    # - compiled_loop_function, threads_per_loop: compiled artifacts
    # - _algo_step, _step_controller, _loop, _output_functions: tested
    # separately


def assert_ivploop_config(loop, settings, tolerance):
    """Assert that IVPLoop attributes match expected settings.

    ALL properties and compile_settings attributes are checked.
    """
    assert loop.save_every == pytest.approx(
        settings["save_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert loop.summarise_every == pytest.approx(
        settings["summarise_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert loop.sample_summaries_every == pytest.approx(
        settings["sample_summaries_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )

    # Check precision
    assert loop.precision == settings["precision"]

    # Check ALL compile_settings attributes
    cs = loop.compile_settings

    is_adaptive = settings["is_adaptive"]

    # For adaptive, dt is computed from atol/rtol
    if not is_adaptive:
        assert cs.dt == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    assert cs.save_every == pytest.approx(
        settings["save_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert cs.summarise_every == pytest.approx(
        settings["summarise_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert cs.sample_summaries_every == pytest.approx(
        settings["sample_summaries_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )

    assert cs.is_adaptive == is_adaptive

    # Check compile_flags
    cf = cs.compile_flags
    assert cf.save_state == settings["save_state"]
    assert cf.save_observables == settings["save_observables"]
    assert cf.summarise == settings["summarise"]
    assert cf.summarise_state == settings["summarise_state"]
    assert cf.summarise_observables == settings["summarise_observables"]
    assert cf.save_counters == settings["save_counters"]

    # Not tested (computed/runtime):
    # - shared_buffer_indices, local_indices: computed from sizes
    # - loop_shared_elements, loop_local_elements: computed
    # - saves_per_summary: computed
    # - evaluate_driver_at_t, evaluate_observables: function references
    # - precision, numba_precision, simsafe_precision: derived


def assert_output_functions_config(output_functions, settings, tolerance):
    """Assert that OutputFunctions attributes match expected settings.

    ALL properties and compile_flags are checked.
    """
    assert output_functions.output_types == tuple(
        settings["output_types"]
    )

    # Saved and summarised indices - compare directly to expected indices from
    # extended settings
    assert (
        list(output_functions.saved_state_indices)
        == settings["saved_state_indices"]
    )
    assert (
        list(output_functions.saved_observable_indices)
        == settings["saved_observable_indices"]
    )
    assert (
        list(output_functions.summarised_state_indices)
        == settings["summarised_state_indices"]
    )
    assert (
        list(output_functions.summarised_observable_indices)
        == settings["summarised_observable_indices"]
    )

    # Check compile_flags
    cf = output_functions.compile_flags
    assert cf.save_state == settings["save_state"]
    assert cf.save_observables == settings["save_observables"]
    assert cf.summarise == settings["summarise"]
    assert cf.summarise_state == settings["summarise_state"]
    assert cf.summarise_observables == settings["summarise_observables"]
    assert cf.save_counters == settings["save_counters"]

    # Check save_time
    assert output_functions.save_time == settings["save_time"]

    # Not tested (computed/function references):
    # - save_state_func, update_summaries_func, save_summary_metrics_func:
    # compiled functions
    # - n_saved_states, n_saved_observables: computed from indices
    # - summaries_buffer_sizes, output_array_heights: computed


def assert_symbolic_ode_config(system, settings, tolerance):
    """Assert that SymbolicODE attributes match expected settings."""
    assert system.precision == settings["precision"]

    # Not tested (system structure, not settings):
    # - equations, observables, states, parameters, etc: system definition
    # - num_*, sizes: computed from system
    # - evaluate_f, evaluate_observables: compiled functions


def assert_step_algorithm_config(step_algorithm, settings, tolerance):
    """Assert that step algorithm attributes match expected settings.

    ALL properties and compile_settings attributes are checked.
    """
    # Check algorithm-specific settings
    # For implicit algorithms, check krylov and newton tolerances
    algorithm = settings["algorithm"]

    # Check compile_settings
    cs = step_algorithm.compile_settings

    # dt might be algorithm-specific - check if it exists
    if hasattr(cs, "dt"):
        assert cs.dt == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )

    # Check algorithm-specific tolerances for implicit algorithms
    if algorithm in [
        "backwards_euler",
        "backwards_euler_pc",
        "crank_nicolson",
    ]:
        if hasattr(step_algorithm, "krylov_atol"):
            assert step_algorithm.krylov_atol == pytest.approx(
                settings["krylov_atol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
        if hasattr(step_algorithm, "krylov_rtol"):
            assert step_algorithm.krylov_rtol == pytest.approx(
                settings["krylov_rtol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
        if hasattr(step_algorithm, "newton_atol"):
            assert step_algorithm.newton_atol == pytest.approx(
                settings["newton_atol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
        if hasattr(step_algorithm, "newton_rtol"):
            assert step_algorithm.newton_rtol == pytest.approx(
                settings["newton_rtol"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )

    # Not tested (algorithm-specific/computed):
    # - n: system parameter, not a setting
    # - stage_count, can_reuse_accepted_start, first_same_as_last: algorithm
    # structure
    # - evaluate_driver_at_t: function reference
    # - n_drivers: system property
    # - simsafe_precision: derived
    # - settings_dict: internal storage


def assert_step_controller_config(
    step_controller, settings, tolerance, is_adaptive
):
    """Assert that step controller attributes match expected settings.

    ALL properties and compile_settings attributes are checked.
    """
    # Adaptive controllers don't have dt property, fixed controllers do
    if not is_adaptive:
        assert step_controller.dt == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )

    # Check compile_settings
    cs = step_controller.compile_settings

    # Adaptive controllers don't have dt in compile_settings
    if not is_adaptive:
        assert cs.dt == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )

    # For fixed-step, dt_min and dt_max equal dt
    # For adaptive, they match settings
    if is_adaptive:
        assert cs.dt_min == pytest.approx(
            settings["dt_min"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        assert cs.dt_max == pytest.approx(
            settings["dt_max"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
    else:
        assert cs.dt_min == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
        assert cs.dt_max == pytest.approx(
            settings["dt"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )

    assert cs.is_adaptive == is_adaptive

    # Check adaptive controller-specific properties
    if is_adaptive:
        controller_type = settings["step_controller"]

        # All adaptive controllers have atol, rtol
        assert step_controller.atol == pytest.approx(
            settings["atol"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
        assert step_controller.rtol == pytest.approx(
            settings["rtol"], rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )

        # PID, PI, Gustafsson controllers have step-factor bounds
        if controller_type in ["PID", "PI", "Gustafsson"]:
            assert step_controller.min_step_shrink == pytest.approx(
                settings["min_step_shrink"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
            assert step_controller.max_step_growth == pytest.approx(
                settings["max_step_growth"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )

        # PID controller has all three gains
        if controller_type == "PID":
            assert step_controller.integral_gain == pytest.approx(
                settings["integral_gain"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
            assert step_controller.proportional_gain == pytest.approx(
                settings["proportional_gain"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
            assert step_controller.derivative_gain == pytest.approx(
                settings["derivative_gain"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )

        # PI controller has integral and proportional gains
        if controller_type == "PI":
            assert step_controller.integral_gain == pytest.approx(
                settings["integral_gain"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )
            assert step_controller.proportional_gain == pytest.approx(
                settings["proportional_gain"],
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            )

        # Check deadband for controllers that have it
        if controller_type in ["PID", "PI"]:
            if hasattr(step_controller, "deadband_min"):
                assert step_controller.deadband_min == pytest.approx(
                    settings["deadband_min"],
                    rel=tolerance.rel_tight,
                    abs=tolerance.abs_tight,
                )
            if hasattr(step_controller, "deadband_max"):
                assert step_controller.deadband_max == pytest.approx(
                    settings["deadband_max"],
                    rel=tolerance.rel_tight,
                    abs=tolerance.abs_tight,
                )

    # Not tested:
    # - n: system parameter, not a setting
    # - settings_dict: internal storage


def assert_summary_metrics_config(output_functions, settings, tolerance):
    """Assert that summary metrics configuration matches expected settings.

    Tests each individual metric object for precision.
    """
    # Import the summary_metrics singleton
    from cubie.outputhandling import summary_metrics

    # Get the summary metrics objects and check precision for each
    for metric_name, metric_object in summary_metrics._metric_objects.items():
        assert metric_object.precision == settings["precision"], (
            f"Metric {metric_name} precision mismatch"
        )


@pytest.mark.parametrize(
    "algorithm,controller",
    [
        ("backwards_euler", "fixed"),
        ("crank_nicolson", "i"),
        ("rk23", "gustafsson"),  # Embedded RK with Gustafsson
        ("rk45", "pid"),  # Embedded RK with PID
        ("dopri54", "pi"),  # ERK with PI
        ("tsit5", "pi"),  # Rosenbrock-type with PI
    ],
)
def test_comprehensive_config_plumbing(
    solver_mutable,
    solver_settings,
    system,
    precision,
    tolerance,
    algorithm,
    controller,
):
    """Test that all config updates propagate through the solver hierarchy.

    This test validates that EVERY property and compile_settings attribute
    is correctly updated when solver.update() is called.

    Parameterized to test switching between different algorithm/controller
    combinations.
    """
    solver = solver_mutable

    # Build kernel to ensure all objects are initialized

    _ = solver.kernel.kernel

    # Get references to all objects in the hierarchy
    kernel = solver.kernel
    single_integrator = kernel.single_integrator
    step_algorithm = single_integrator._algo_step
    step_controller = single_integrator._step_controller
    loop = single_integrator._loop
    output_functions = single_integrator._output_functions

    # === PHASE 1: Assert default settings ===
    print("\n=== Testing default settings ===")

    # Create extended settings with all derived values
    initial_settings = {
        k: v
        for k, v in solver_settings.items()
        if k not in ("duration", "warmup", "t0")
    }
    initial_settings = extend_expected_settings(initial_settings, precision)

    is_adaptive = initial_settings["is_adaptive"]

    assert_solver_config(solver, initial_settings, tolerance)
    assert_solverkernel_config(kernel, initial_settings, tolerance)
    assert_singleintegratorrun_config(
        single_integrator, initial_settings, tolerance
    )
    assert_ivploop_config(loop, initial_settings, tolerance)
    assert_output_functions_config(
        output_functions, initial_settings, tolerance
    )
    assert_symbolic_ode_config(system, initial_settings, tolerance)
    assert_step_algorithm_config(step_algorithm, initial_settings, tolerance)
    assert_step_controller_config(
        step_controller, initial_settings, tolerance, is_adaptive
    )
    assert_summary_metrics_config(
        output_functions, initial_settings, tolerance
    )

    # === PHASE 2: Create updates dict with EVERY value different ===
    print("\n=== Creating updates with all different values ===")

    # Note: obsolete, durtion, warmup, t0 are only set from .run()
    # Set duration, warmup, t0 to different values from defaults
    solver.kernel.duration = precision(3.0)  # Different from initial 0.0
    solver.kernel.warmup = precision(0.2)  # Different from initial 0.0
    solver.kernel.t0 = precision(1.0)  # Different from initial 0.0

    # Create updates where EVERY value is different from defaults
    # Algorithm and controller from parameterized test
    updates = {
        # Algorithm and controller (parameterized)
        "algorithm": algorithm,
        "step_controller": controller,
        # Time stepping
        "dt": precision(0.005),
        "dt_min": precision(5e-8),
        "dt_max": precision(0.5),
        "save_every": precision(0.05),
        "summarise_every": precision(0.15),
        "sample_summaries_every": precision(0.05),
        # Tolerances
        "atol": precision(5e-7),
        "rtol": precision(5e-7),
        # Output configuration
        "output_types": ["state", "observables", "mean", "max"],
        "saved_state_indices": [0, 2],
        "saved_observable_indices": [0],
        "summarised_state_indices": [1, 2],
        "summarised_observable_indices": [0, 2],
        # Memory settings
        "mem_proportion": 0.15,
        # Step controller settings (only for adaptive, but included for
        # completeness)
        "min_step_shrink": precision(0.15),
        "max_step_growth": precision(3.0),
        "integral_gain": precision(1 / 15),
        "proportional_gain": precision(1 / 7),
        "derivative_gain": precision(1 / 15),
        "deadband_min": precision(0.9),
        "deadband_max": precision(1.3),
        # Algorithm settings (only for implicit, but included)
        "krylov_atol": precision(5e-7),
        "krylov_rtol": precision(5e-7),
        "newton_atol": precision(5e-7),
        "newton_rtol": precision(5e-7),
        "krylov_max_iters": 300,
        "newton_max_iters": 300,
        "preconditioner_order": 1,
    }

    # === PHASE 3: Update solver and rebuild ===
    print("\n=== Updating solver ===")
    updated_keys = solver.update(updates)
    print(f"Updated keys: {sorted(updated_keys)}")

    # Rebuild kernel to apply changes
    _ = solver.kernel.kernel

    # Get fresh references to all objects after rebuild (algorithm/controller
    # may have changed)
    kernel = solver.kernel
    single_integrator = kernel.single_integrator
    step_algorithm = single_integrator._algo_step
    step_controller = single_integrator._step_controller
    loop = single_integrator._loop
    output_functions = single_integrator._output_functions

    # === PHASE 4: Assert updated settings ===
    print("\n=== Testing updated settings ===")

    # Merge updates into expected settings
    expected_settings = solver_settings.copy()
    expected_settings.update(updates)
    # Add manually set properties (different from initial phase)
    expected_settings["duration"] = precision(3.0)
    expected_settings["warmup"] = precision(0.2)
    expected_settings["t0"] = precision(1.0)

    # Extend with all derived values
    expected_settings = extend_expected_settings(expected_settings, precision)

    is_adaptive = expected_settings["is_adaptive"]

    assert_solver_config(solver, expected_settings, tolerance)
    assert_solverkernel_config(kernel, expected_settings, tolerance)
    assert_singleintegratorrun_config(
        single_integrator, expected_settings, tolerance
    )
    assert_ivploop_config(loop, expected_settings, tolerance)
    assert_output_functions_config(
        output_functions, expected_settings, tolerance
    )
    assert_symbolic_ode_config(system, expected_settings, tolerance)
    assert_step_algorithm_config(step_algorithm, expected_settings, tolerance)
    assert_step_controller_config(
        step_controller, expected_settings, tolerance, is_adaptive
    )
    assert_summary_metrics_config(
        output_functions, expected_settings, tolerance
    )

    print("\n=== All assertions passed! ===")
