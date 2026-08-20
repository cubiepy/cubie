import numpy as np
import pytest

from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.buffer_registry import buffer_registry
from cubie.outputhandling.output_sizes import BatchOutputSizes
from cubie.outputhandling.output_config import OutputCompileFlags
from cubie.batchsolving.BatchSolverConfig import ActiveOutputs


def test_kernel_builds(solverkernel):
    """Test that the solver builds without errors."""
    assert solverkernel.kernel is not None


def test_algorithm_change(solverkernel_mutable):
    solverkernel = solverkernel_mutable
    solverkernel.update(
        {"algorithm": "crank_nicolson", "step_controller": "pid"}
    )
    assert solverkernel.single_integrator._step_controller.atol is not None


def test_getters_get(solverkernel):
    """Check for dead getters"""
    assert solverkernel.shared_memory_bytes is not None, (
        "BatchSolverKernel.shared_memory_bytes returning None"
    )
    assert solverkernel.shared_memory_elements is not None, (
        "BatchSolverKernel.shared_memory_elements_per_run returning None"
    )
    assert solverkernel.precision is not None, (
        "BatchSolverKernel.precision returning None"
    )
    assert solverkernel.threads_per_loop is not None, (
        "BatchSolverKernel.threads_per_loop returning None"
    )
    assert solverkernel.output_heights is not None, (
        "BatchSolverKernel.output_heights returning None"
    )
    assert solverkernel.output_length is not None, (
        "BatchSolverKernel.output_length returning None"
    )
    assert solverkernel.summaries_length is not None, (
        "BatchSolverKernel.summaries_length returning None"
    )
    assert solverkernel.num_runs is not None, (
        "BatchSolverKernel.num_runs returning None"
    )
    assert solverkernel.system is not None, (
        "BatchSolverKernel.system returning None"
    )
    assert solverkernel.duration is not None, (
        "BatchSolverKernel.duration returning None"
    )
    assert solverkernel.warmup is not None, (
        "BatchSolverKernel.warmup returning None"
    )
    assert solverkernel.save_every is not None, (
        "BatchSolverKernel.save_every returning None"
    )
    assert solverkernel.summarise_every is not None, (
        "BatchSolverKernel.summarise_every returning None"
    )
    assert solverkernel.system_sizes is not None, (
        "BatchSolverKernel.system_sizes returning None"
    )
    assert solverkernel.summary_legend_per_variable is not None, (
        "BatchSolverKernel.summary_legend_per_variable returning None"
    )
    assert solverkernel.saved_state_indices is not None, (
        "BatchSolverKernel.saved_state_indices returning None"
    )
    assert solverkernel.saved_observable_indices is not None, (
        "BatchSolverKernel.saved_observable_indices returning None"
    )
    assert solverkernel.summarised_state_indices is not None, (
        "BatchSolverKernel.summarised_state_indices returning None"
    )
    assert solverkernel.summarised_observable_indices is not None, (
        "BatchSolverKernel.summarised_observable_indices returning None"
    )
    assert solverkernel.active_outputs is not None, (
        "BatchSolverKernel.active_outputs returning None"
    )
    # device arrays SHOULD be None.


def test_all_lower_plumbing(
    system,
    solverkernel_mutable,
    step_controller_settings,
    algorithm_settings,
    precision,
    driver_array,
):
    """Big plumbing integration check - check that config classes match exactly
    between an updated solver and one instantiated with the update settings."""
    solverkernel = solverkernel_mutable

    # Limit indices to actual system sizes to prevent IndexError
    n_states = system.sizes.states
    n_obs = system.sizes.observables

    saved_state_idx = list(range(min(3, n_states)))
    saved_obs_idx = list(range(min(3, n_obs)))
    summarised_state_idx = [0] if n_states > 0 else []
    summarised_obs_idx = [0] if n_obs > 0 else []

    new_settings = {
        # "duration": 1.0,
        "dt_min": 0.0001,
        "dt_max": 0.01,
        "save_every": 0.01,
        "summarise_every": 0.1,
        "sample_summaries_every": 0.05,
        "atol": 1e-2,
        "rtol": 1e-1,
        "saved_state_indices": saved_state_idx,
        "saved_observable_indices": saved_obs_idx,
        "summarised_state_indices": summarised_state_idx,
        "summarised_observable_indices": summarised_obs_idx,
        "output_types": [
            "state",
            "observables",
            "mean",
            "max",
            "rms",
            "peaks[3]",
        ],
    }
    solverkernel.update(new_settings)
    updated_controller_settings = step_controller_settings.copy()
    updated_controller_settings.update(
        {
            "dt_min": 0.0001,
            "dt_max": 0.01,
            "atol": 1e-2,
            "rtol": 1e-1,
        }
    )
    output_settings = {
        "saved_state_indices": np.asarray(saved_state_idx),
        "saved_observable_indices": np.asarray(saved_obs_idx),
        "summarised_state_indices": np.asarray(summarised_state_idx),
        "summarised_observable_indices": np.asarray(summarised_obs_idx),
        "output_types": [
            "state",
            "observables",
            "mean",
            "max",
            "rms",
            "peaks[3]",
        ],
    }
    freshsolver = BatchSolverKernel(
        system,
        step_control_settings=updated_controller_settings,
        algorithm_settings=algorithm_settings,
        output_settings=output_settings,
        loop_settings={
            "save_every": 0.01,
            "summarise_every": 0.1,
            "sample_summaries_every": 0.05,
        },
    )
    inits = np.ones((n_states, 1), dtype=precision)
    params = np.ones((system.sizes.parameters, 1), dtype=precision)
    driver_coefficients = driver_array.coefficients
    freshsolver.run(
        inits=inits,
        params=params,
        driver_coefficients=driver_coefficients,
        duration=0.1,
    )
    solverkernel.run(
        inits=inits,
        params=params,
        driver_coefficients=driver_coefficients,
        duration=0.1,
    )
    assert (
        freshsolver.single_integrator.compile_settings
        == solverkernel.single_integrator.compile_settings
    ), "IntegratorRunSettings mismatch"
    assert (
        freshsolver.single_integrator._step_controller.compile_settings
        == solverkernel.single_integrator._step_controller.compile_settings
    )
    assert (
        freshsolver.single_integrator._algo_step.compile_settings
        == solverkernel.single_integrator._algo_step.compile_settings
    )
    assert (
        freshsolver.single_integrator._loop.compile_settings
        == solverkernel.single_integrator._loop.compile_settings
    )
    assert (
        freshsolver.single_integrator._output_functions.compile_settings
        == solverkernel.single_integrator._output_functions.compile_settings
    ), "OutputFunctions mismatch"
    assert (
        freshsolver.single_integrator._system.compile_settings
        == solverkernel.single_integrator._system.compile_settings
    ), "SystemCompileSettings mismatch"
    assert BatchOutputSizes.from_solver(
        freshsolver
    ) == BatchOutputSizes.from_solver(solverkernel), (
        "BatchOutputSizes mismatch"
    )


# ============================================================================
# Additional coverage: no-op update(), simple properties, timing
# validation, and shared-memory padding
# ============================================================================


def test_kernel_update_no_args_returns_empty_set(solverkernel_mutable):
    """update() with neither updates_dict nor kwargs is a no-op."""
    assert solverkernel_mutable.update() == set()
    assert solverkernel_mutable.update(None) == set()


def test_kernel_compile_flags_property(solverkernel):
    """compile_flags reads through to compile_settings.compile_flags."""
    assert (
        solverkernel.compile_flags
        is solverkernel.compile_settings.compile_flags
    )


def test_kernel_initial_values_parameters_driver_coefficients_properties(
    solverkernel,
):
    """initial_values, parameters, driver_coefficients, and
    device_driver_coefficients pass through to input_arrays."""
    assert (
        solverkernel.initial_values
        is solverkernel.input_arrays.initial_values
    )
    assert solverkernel.parameters is solverkernel.input_arrays.parameters
    assert (
        solverkernel.driver_coefficients
        is solverkernel.input_arrays.driver_coefficients
    )
    assert (
        solverkernel.device_driver_coefficients
        is solverkernel.input_arrays.device_driver_coefficients
    )


def test_kernel_update_lineinfo(solverkernel_mutable):
    """update routes lineinfo into the kernel's compile settings."""
    kernel = solverkernel_mutable
    updated = kernel.update({"lineinfo": True})
    assert "lineinfo" in updated
    assert kernel.compile_settings.lineinfo is True
    updated = kernel.update({"lineinfo": False})
    assert "lineinfo" in updated
    assert kernel.compile_settings.lineinfo is False


def test_shared_memory_needs_padding_matches_precision_and_parity(
    solverkernel,
):
    """shared_memory_needs_padding follows precision/parity rules: never
    pads float64, and only pads an even, nonzero element count."""
    result = solverkernel.shared_memory_needs_padding
    elements = solverkernel.shared_memory_elements
    if solverkernel.precision == np.float64:
        expected = False
    elif elements == 0:
        expected = False
    elif elements % 2 == 0:
        expected = True
    else:
        expected = False
    assert result == expected


# NOTE: BatchSolverKernel._validate_timing_parameters lines 450-457
# (the sample_summaries_every-is-None and summarise_every-is-None
# ValueError branches) appear unreachable through the public update()
# API. SingleIntegratorRunCore._process_loop_timing re-derives
# sample_summaries_every from summarise_every whenever
# has_summary_outputs is True and summarise_every is not None, and
# flips has_summary_outputs to False (deferring to "duration
# dependent" resolution) the moment summarise_every is cleared while
# summary metrics are requested. In manual testing, clearing either
# or both of these settings via kernel.update() on a summary-active
# kernel always leaves has_summary_outputs False by the time
# _validate_timing_parameters runs, so the guarded branch is never
# entered from any code path reachable via update()/run(). See the
# coverage report for details; not exercised here to avoid
# constructing a stand-in object for the method's ``self``.


def test_bogus_update_fails(solverkernel_mutable):
    solverkernel = solverkernel_mutable
    solverkernel.update(dt_min=0.0001)
    with pytest.raises(KeyError):
        solverkernel.update(obviously_bogus_key="this should not work")


class TestTimingParameterValidation:
    """Tests for timing parameter validation in BatchSolverKernel.run().

    Solve-time timing kwargs permanently reconfigure the solver (the
    validation error is raised after the settings update is applied),
    so every test here uses the function-scoped ``solver_mutable``
    rather than the shared session ``solver`` fixture.
    """

    def test_save_every_greater_than_duration_no_save_last_raises(
        self, system, precision, driver_array, solver_mutable,
        driver_settings
    ):
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        with pytest.raises(
            ValueError, match=r"save_every.*>.*duration.*no outputs"
        ):
            solver_mutable.solve(
                inits,
                params,
                driver_settings,
                save_every=1.0,
                duration=0.5,
            )

    def test_save_every_greater_than_duration_with_save_last_succeeds(
        self, system, precision, driver_array, solver_mutable,
        driver_settings
    ):
        """Test that save_every >= duration with save_last=True is valid."""
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        # Should not raise when save_last is True (default when
        # save_every=None)
        solver_mutable.solve(
            inits,
            params,
            drivers=driver_settings,
            save_every=None,
            duration=0.05,
            dt=0.02,
        )

    def test_summarise_every_greater_than_duration_raises(
        self, system, precision, driver_array, solver_mutable,
        driver_settings
    ):
        """Test that summarise_every > duration raises."""
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        with pytest.raises(
            ValueError,
            match=r"summarise_every.*>.*duration.*no summary outputs",
        ):
            solver_mutable.solve(
                inits,
                params,
                driver_settings,
                summarise_every=0.6,
                duration=0.5,
            )

    def test_sample_summaries_every_gte_summarise_every_raises(
        self, system, precision, driver_array, solver_mutable,
        driver_settings
    ):
        """Test that sample_summaries_every >= summarise_every raises."""
        inits = np.ones((3, 1), dtype=precision)
        params = np.ones((3, 1), dtype=precision)

        with pytest.raises(
            ValueError, match=r"sample_summaries_every.*>=.*summarise_every"
        ):
            solver_mutable.solve(
                inits,
                params,
                drivers=driver_settings,
                summarise_every=0.01,
                sample_summaries_every=0.01,
                duration=1.0,
            )


class TestActiveOutputsFromCompileFlags:
    """Tests for ActiveOutputs.from_compile_flags factory method."""

    def test_all_flags_true(self, precision):
        """Test mapping when all compile flags are enabled."""
        # Use specific flags (summarise_state, summarise_observables) which are
        # what ActiveOutputs.from_compile_flags() reads; the general
        # 'summarise' flag is redundant here but included for completeness
        flags = OutputCompileFlags(
            save_state=True,
            save_observables=True,
            summarise_observables=True,
            summarise_state=True,
            save_counters=True,
        )
        active = ActiveOutputs.from_compile_flags(flags)

        assert active.state is True
        assert active.observables is True
        assert active.state_summaries is True
        assert active.observable_summaries is True
        assert active.iteration_counters is True
        assert active.status_codes is True

    def test_all_flags_false(self, precision):
        """Test mapping when all compile flags are disabled."""
        flags = OutputCompileFlags(
            save_state=False,
            save_observables=False,
            summarise_observables=False,
            summarise_state=False,
            save_counters=False,
        )
        active = ActiveOutputs.from_compile_flags(flags)

        assert active.state is False
        assert active.observables is False
        assert active.state_summaries is False
        assert active.observable_summaries is False
        assert active.iteration_counters is False
        # status_codes is ALWAYS True
        assert active.status_codes is True

    def test_status_codes_always_true(self, precision):
        """Verify status_codes is always True regardless of flags."""
        flags = OutputCompileFlags()  # All defaults (False)
        active = ActiveOutputs.from_compile_flags(flags)
        assert active.status_codes is True

    def test_partial_flags(self, precision):
        """Test with only some flags enabled."""
        flags = OutputCompileFlags(
            save_state=True,
            save_observables=False,
            summarise=True,
            summarise_observables=False,
            summarise_state=True,
            save_counters=False,
        )
        active = ActiveOutputs.from_compile_flags(flags)

        assert active.state is True
        assert active.observables is False
        assert active.state_summaries is True
        assert active.observable_summaries is False
        assert active.iteration_counters is False
        assert active.status_codes is True


class TestRunParamsIntegration:
    """Tests for RunParams integration into BatchSolverKernel."""

    def test_runparams_initialized_on_construction(self, solverkernel_mutable):
        """Verify BatchSolverKernel initializes run_params with defaults."""
        solverkernel = solverkernel_mutable  # A used solverkernel might be
        # updated

        assert hasattr(solverkernel, "run_params")
        assert solverkernel.run_params.duration == 0.0
        assert solverkernel.run_params.warmup == 0.0
        assert solverkernel.run_params.t0 == 0.0
        assert solverkernel.run_params.runs == 1
        assert solverkernel.run_params.num_chunks == 1
        assert solverkernel.run_params.chunk_length == 0


def test_limit_blocksize_floors_at_one_warp(solverkernel):
    """Performance-stage reduction stops at 32 threads.

    Per-run shared demand over the 32 kiB target but within the
    device per-block limit at one warp keeps blocksize 32 and warns
    instead of halving into sub-warp blocks.
    """
    bytes_per_run = 1200
    blocksize = 256
    smem = bytes_per_run * blocksize
    with pytest.warns(UserWarning, match="performance target"):
        new_blocksize, new_smem = solverkernel.limit_blocksize(
            blocksize, smem, bytes_per_run, 65536
        )
    assert new_blocksize == 32
    assert new_smem == bytes_per_run * 32


def test_limit_blocksize_subwarp_only_when_hardware_requires(
    solverkernel,
):
    """Sub-warp blocks appear only past the device per-block limit.

    4096 B/run needs 128 kiB at one warp — over the 48 kiB device
    limit — so the hardware stage halves to the largest launchable
    block size (8 runs, 32 kiB).
    """
    bytes_per_run = 4096
    blocksize = 256
    smem = bytes_per_run * blocksize
    with pytest.warns(UserWarning, match="below warp width"):
        new_blocksize, new_smem = solverkernel.limit_blocksize(
            blocksize, smem, bytes_per_run, 65536
        )
    assert new_blocksize == 8
    assert new_smem == bytes_per_run * 8
    assert new_smem <= 49152


def test_limit_blocksize_raises_when_one_run_cannot_fit(solverkernel):
    """A single run over the device limit is unlaunchable: raise."""
    bytes_per_run = 50000
    with pytest.raises(ValueError, match="single run"):
        solverkernel.limit_blocksize(
            256, bytes_per_run * 256, bytes_per_run, 65536
        )


def test_limit_blocksize_halves_to_fit(solverkernel):
    """Reduction still finds the largest fitting block size."""
    bytes_per_run = 320
    blocksize = 256
    smem = bytes_per_run * blocksize
    new_blocksize, new_smem = solverkernel.limit_blocksize(
        blocksize, smem, bytes_per_run, 65536
    )
    assert new_blocksize == 64
    assert new_smem == bytes_per_run * 64
    assert new_smem < 32768


def test_limit_blocksize_leaves_fitting_requests_alone(solverkernel):
    """Requests already under the ceiling pass through unchanged."""
    new_blocksize, new_smem = solverkernel.limit_blocksize(
        256, 16384, 64, 65536
    )
    assert new_blocksize == 256
    assert new_smem == 16384


def test_persistent_array_sized_from_persistent_layout(solverkernel):
    """The persistent scratch array is sized by the persistent layout.

    The loop's persistent slices index into the array that
    ``buffer_registry.get_toplevel_allocators`` sizes from
    ``persistent_local_elements``; any other sizing source lets the
    slices run past the array end and corrupt per-thread storage.
    """
    loop = solverkernel.single_integrator._loop
    assert solverkernel.persistent_local_elements == (
        buffer_registry.persistent_local_buffer_size(loop)
    )
