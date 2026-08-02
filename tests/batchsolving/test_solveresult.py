from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._utils import STATE_ONLY_NO_SUMMARIES

from cubie.batchsolving.solver import Solver
from cubie.batchsolving.BatchSolverConfig import ActiveOutputs
from cubie.batchsolving.solveresult import DeviceSolveResult, SolveResult
from cubie.memory import MemoryManager

Array = np.ndarray


def test_results_take_ownership_of_host_buffers(
    solver_mutable, batch_input_arrays, driver_settings
):
    """Results own the solve's buffers without copying."""
    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)

    result = solver_mutable.solve(y0, params, **solve_kwargs)
    # The result holds the buffers; the solver's slots are empty.
    assert result.state is not None
    assert solver_mutable.kernel.output_arrays.state is None

    # A solve while the result lives allocates fresh backing; both
    # results stay valid and independent.
    second = solver_mutable.solve(y0, params, **solve_kwargs)
    assert second.state is not result.state
    assert np.isfinite(result.time_domain_array).all()
    assert np.isfinite(second.time_domain_array).all()
    assert np.array_equal(result.state, second.state)
    result.close()
    second.close()


def test_dropped_result_buffers_are_reused(
    solver_mutable, batch_input_arrays, driver_settings
):
    """A collected result's buffers return to the solver for reuse."""
    import weakref

    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)

    result = solver_mutable.solve(y0, params, **solve_kwargs)
    buffer_ref = weakref.ref(result.state)
    del result
    second = solver_mutable.solve(y0, params, **solve_kwargs)
    assert second.state is buffer_ref()
    second.close()


def test_spill_result_context_releases_shared_mapping(tmp_path):
    """Result context cleanup releases each spill mapping once."""

    class SpillClient:
        def notice_invalidate(self):
            pass

    manager = MemoryManager()
    client = SpillClient()
    manager.register(client, spill_directory=tmp_path)
    shared = manager.create_host_array(
        (2, 2, 2), np.float64, "memmap", instance=client
    )
    path = Path(shared._cubie_spill_path)
    with SolveResult(
        state=shared,
        observables=shared,
        active_outputs=ActiveOutputs(state=True, observables=True),
        memory_manager=manager,
    ) as result:
        assert path.exists()
        assert result.state is result.observables
    assert not path.exists()
    result.close()


@pytest.fixture(scope="session")
def solver_with_arrays(
    solver,
    batch_input_arrays,
    solver_settings,
    precision,
    driver_array,
):
    """Solver with actual arrays computed - ready for SolveResult instantiation"""
    inits, params = batch_input_arrays
    solver.kernel.run(
        duration=solver_settings["duration"],
        params=params,
        inits=inits,
        driver_coefficients=driver_array.coefficients,
        blocksize=solver_settings["blocksize"],
        warmup=solver_settings["warmup"],
    )
    # kernel.run launches and copies back asynchronously; wait for the
    # host arrays like Solver.solve does before results are read.
    solver.memory_manager.sync_stream(solver.kernel)
    solver.kernel.wait_for_writeback()

    return solver


class TestSolveResultStaticMethods:
    """Test static methods with minimal unit tests using known arrays."""

    @pytest.mark.parametrize(
        "state_flag, obs_flag",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_combine_active(self, state_flag, obs_flag):
        """Combined arrays concatenate; a single source is a view."""
        # Arrays with shape (time, variable, run) = (2, 1, 2)
        s = np.array([[[1, 2]], [[3, 4]]])
        o = np.array([[[5, 6]], [[7, 8]]])
        result = SolveResult._combine_active(
            s if state_flag else None, o if obs_flag else None
        )
        if state_flag and obs_flag:
            assert np.array_equal(result, np.concatenate((s, o), axis=1))
        elif state_flag:
            assert result is s
        elif obs_flag:
            assert result is o
        else:
            assert result.size == 0

    def test_cleave_time_with_time_saved(self):
        """Test time cleaving when time is saved."""
        # Create test array with shape (time, variable, run)
        # time has 2 samples, 3 variables (including time at last position), 1 run
        state = np.array([[[1], [2], [0.1]], [[3], [4], [0.2]]])  # (2, 3, 1)
        time_result, state_result = SolveResult.cleave_time(
            state, time_saved=True
        )

        expected_time = np.array([[0.1], [0.2]])
        expected_state = np.array([[[1], [2]], [[3], [4]]])  # (2, 2, 1)

        assert np.array_equal(time_result, expected_time)
        assert np.array_equal(state_result, expected_state)

    def test_cleave_time_without_time_saved(self):
        """Test time cleaving when time is not saved."""
        state = np.array([[[1], [2]], [[3], [4]]])  # (2, 2, 1) shape
        time_result, state_result = SolveResult.cleave_time(
            state, time_saved=False
        )

        assert time_result is None
        assert np.array_equal(state_result, state)


class TestSolveResultRepresentations:
    """The RAM representations derive from one owned result."""

    def test_representations_are_self_consistent(self, solver_with_arrays):
        """Repeated accessor calls return equal representations."""
        full_result = SolveResult.from_solver(solver_with_arrays)

        numpy_result = full_result.as_numpy
        for key, value in full_result.as_numpy.items():
            if isinstance(value, np.ndarray):
                assert np.allclose(
                    numpy_result[key], value, equal_nan=True
                )
            else:
                assert numpy_result[key] == value

        per_summary = full_result.as_numpy_per_summary
        assert set(per_summary.keys()) == set(
            full_result.as_numpy_per_summary.keys()
        )
        pandas_result = full_result.as_pandas
        assert pandas_result["time_domain"].equals(
            full_result.as_pandas["time_domain"]
        )
        assert pandas_result["summaries"].equals(
            full_result.as_pandas["summaries"]
        )

    def test_from_solver_full_instantiation(self, solver_with_arrays):
        """Test full SolveResult instantiation from solver."""
        result = SolveResult.from_solver(solver_with_arrays)

        assert isinstance(result, SolveResult)
        assert result.time_domain_array.size > 0
        assert result.summaries_array.size > 0
        assert len(result.time_domain_legend) > 0
        assert len(result.summaries_legend) > 0
        assert (
            result._stride_order
            == solver_with_arrays.kernel.output_arrays.host.state.stride_order
        )

    def test_as_numpy_contents(self, solver_with_arrays):
        """as_numpy returns RAM copies keyed like the legends."""
        result = SolveResult.from_solver(solver_with_arrays).as_numpy

        assert isinstance(result, dict)
        assert "time_domain_array" in result
        assert "summaries_array" in result
        assert "time_domain_legend" in result
        assert "summaries_legend" in result
        assert "iteration_counters" in result
        assert isinstance(result["time_domain_array"], np.ndarray)

    def test_as_numpy_per_summary_contents(self, solver_with_arrays):
        """as_numpy_per_summary splits summaries by metric."""
        result = SolveResult.from_solver(
            solver_with_arrays
        ).as_numpy_per_summary

        assert isinstance(result, dict)
        assert "mean" in result
        assert "time_domain_array" in result
        assert "iteration_counters" in result
        # Check that summary types are separated
        for (
            summary_type
        ) in solver_with_arrays.summary_legend_per_variable.values():
            assert summary_type in result

    def test_as_pandas_contents(self, solver_with_arrays):
        """as_pandas returns time-domain and summaries DataFrames."""
        result = SolveResult.from_solver(solver_with_arrays).as_pandas

        assert isinstance(result, dict)
        assert "time_domain" in result
        assert "summaries" in result
        assert isinstance(result["time_domain"], pd.DataFrame)
        assert isinstance(result["summaries"], pd.DataFrame)


class TestSolveResultFromSolver:
    """Test SolveResult creation and methods using real solver instances."""

    def test_time_domain_legend_from_solver(self, solver_with_arrays):
        """Test time domain legend creation from real solver."""
        legend = SolveResult.time_domain_legend_from_solver(solver_with_arrays)

        assert isinstance(legend, dict)
        assert len(legend) > 0
        # Check that state variables are included
        for i, state_name in enumerate(solver_with_arrays.saved_states):
            assert legend[i] == state_name
        # Check that observables are included when configured
        if solver_with_arrays.saved_observables:
            # Observables are configured - verify they're in the legend
            num_states = len(solver_with_arrays.saved_states)
            for i, obs_name in enumerate(solver_with_arrays.saved_observables):
                assert legend[num_states + i] == obs_name

    def test_summary_legend_from_solver(self, solver_with_arrays):
        """Test summary legend creation from real solver."""
        legend = SolveResult.summary_legend_from_solver(solver_with_arrays)

        assert isinstance(legend, dict)
        # Check that summary metrics are in the legend values
        legend_values = list(legend.values())
        summary_metrics = [
            s
            for s in solver_with_arrays.output_types
            if any(s.startswith(metric) for metric in ["mean", "rms"])
        ]

        for metric in summary_metrics:
            assert any(metric in val for val in legend_values)

    def test_stride_order_from_solver(self, solver_with_arrays):
        """Test that stride order is correctly captured from solver."""
        result = SolveResult.from_solver(solver_with_arrays)

        assert (
            result._stride_order
            == solver_with_arrays.kernel.output_arrays.host.state.stride_order
        )

        # Test that run dimension is found correctly
        run_dim = result._stride_order.index("run")
        assert isinstance(run_dim, int)
        assert 0 <= run_dim < len(result._stride_order)


class TestSolveResultProperties:
    """Test SolveResult property methods using real solver data."""

    def test_as_numpy_property(self, solver_with_arrays):
        """Test as_numpy property returns correct structure."""
        result = SolveResult.from_solver(solver_with_arrays)
        numpy_dict = result.as_numpy

        assert isinstance(numpy_dict, dict)
        assert "time_domain_array" in numpy_dict
        assert "summaries_array" in numpy_dict
        assert "time_domain_legend" in numpy_dict
        assert "summaries_legend" in numpy_dict
        assert "iteration_counters" in numpy_dict
        # Verify arrays are copies
        assert np.array_equal(
            numpy_dict["time_domain_array"],
            result.time_domain_array,
            equal_nan=True,
        )
        assert numpy_dict["time_domain_array"] is not result.time_domain_array
        if result.iteration_counters is not None:
            assert np.array_equal(
                numpy_dict["iteration_counters"],
                result.iteration_counters,
                equal_nan=True,
            )
            assert (
                numpy_dict["iteration_counters"]
                is not result.iteration_counters
            )

    def test_per_summary_arrays_property(self, solver_with_arrays):
        """Test per_summary_arrays property splits summaries correctly."""
        result = SolveResult.from_solver(solver_with_arrays)
        per_summary = result.per_summary_arrays

        assert isinstance(per_summary, dict)
        singlevar_legend = result._singlevar_summary_legend

        # Check that each summary type has its own array
        for summary_name in singlevar_legend.values():
            assert summary_name in per_summary
            assert isinstance(per_summary[summary_name], np.ndarray)

    def test_as_pandas_property(self, solver_with_arrays):
        """Test as_pandas property creates proper DataFrames."""
        result = SolveResult.from_solver(solver_with_arrays)
        pandas_dict = result.as_pandas

        assert isinstance(pandas_dict, dict)
        assert "time_domain" in pandas_dict
        assert "summaries" in pandas_dict

        time_df = pandas_dict["time_domain"]
        summaries_df = pandas_dict["summaries"]

        assert isinstance(time_df, pd.DataFrame)
        assert isinstance(summaries_df, pd.DataFrame)

        # Check MultiIndex columns for multiple runs
        if result.time_domain_array.ndim == 3:
            assert isinstance(time_df.columns, pd.MultiIndex)
            # Check run naming
            run_levels = time_df.columns.get_level_values(0).unique()
            expected_runs = result.time_domain_array.shape[
                result._stride_order.index("run")
            ]
            assert len(run_levels) == expected_runs
            for i, run_name in enumerate(run_levels):
                assert run_name == f"run_{i}"

    def test_active_outputs_property(self, solver_with_arrays):
        """Test active_outputs property returns correct ActiveOutputs."""
        result = SolveResult.from_solver(solver_with_arrays)
        active = result.active_outputs

        assert isinstance(active, ActiveOutputs)
        assert active == solver_with_arrays.active_outputs


class TestSolveResultDefaultBehavior:
    """Test SolveResult default instantiation and edge cases."""

    def test_default_instantiation(self):
        """Test SolveResult instantiation with defaults."""
        result = SolveResult()

        assert result.time_domain_array.size == 0
        assert result.summaries_array.size == 0
        assert result.time is None
        assert len(result.time_domain_legend) == 0
        assert len(result.summaries_legend) == 0
        assert result._stride_order == ("time", "variable", "run")


class TestSolveResultPandasIntegration:
    """Test pandas-specific functionality."""

    def test_pandas_shape_consistency(self, solver_with_arrays):
        """Test pandas DataFrame shapes are consistent with array dimensions."""
        result = SolveResult.from_solver(solver_with_arrays)
        td_df, sum_df = (
            result.as_pandas["time_domain"],
            result.as_pandas["summaries"],
        )

        array_shape = result.time_domain_array.shape
        if result.time_domain_array.ndim == 3:
            # For 3D arrays: (time, variable, run) -> DataFrame (time, variable*run)
            expected_cols = array_shape[1] * array_shape[2]  # variable * run
            assert td_df.shape == (array_shape[0], expected_cols)

        # Check summaries DataFrame columns match legend
        # With layout (time, variable, run): shape[2] is n_runs
        if result.summaries_array.size > 0:
            expected_sum_cols = (
                len(result.summaries_legend) * array_shape[2]
                if result.summaries_array.ndim == 3
                else len(result.summaries_legend)
            )
            assert sum_df.shape[1] == expected_sum_cols

    def test_pandas_time_indexing(self, solver_with_arrays):
        """Test that time array is used as DataFrame index when available."""
        result = SolveResult.from_solver(solver_with_arrays)
        td_df = result.as_pandas["time_domain"]

        if result.time is not None and result.time.size > 0:
            # Check that time values are used as index
            if result.time.ndim == 1:
                assert np.array_equal(td_df.index.values, result.time)
            else:
                # For multi-run, should use time from first run or appropriate run
                expected_time = (
                    result.time[:, 0]
                    if result.time.shape[1] > 0
                    else result.time.flatten()
                )
                assert len(td_df.index) == len(expected_time)


class TestSolveResultErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_summaries_per_summary_arrays(self, solver_settings):
        """Test per_summary_arrays with no summaries."""
        result = SolveResult()
        result._active_outputs = ActiveOutputs(
            state_summaries=False, observable_summaries=False
        )

        per_summary = result.per_summary_arrays
        assert per_summary == {}


def test_status_codes_attribute(solver_with_arrays):
    """Verify status_codes attribute is present in SolveResult."""
    result = SolveResult.from_solver(solver_with_arrays)

    assert hasattr(result, "status_codes")
    assert result.status_codes is not None
    assert isinstance(result.status_codes, np.ndarray)
    assert result.status_codes.dtype == np.int32

    # Verify shape matches number of runs
    n_runs = solver_with_arrays.num_runs
    assert result.status_codes.shape == (n_runs,)
    assert np.all(result.status_codes == 0)


@pytest.fixture(scope="session")
def error_injection_solver(system):
    """One solver dedicated to the destructive NaN-masking tests.

    ``SolveResult.from_solver(..., nan_error_trajectories=True)``
    NaN-masks the owned buffers in place, so these tests never touch
    the shared ``solver`` fixture.
    """
    solver = Solver(system, save_every=0.01, stream_group="nan_masking")
    yield solver
    solver.close()


@pytest.fixture()
def solved_batch_solver_errorcode(error_injection_solver):
    """Provide a freshly solved 3-run solver with run 1 marked failed.

    The solve is repeated per test because NaN masking mutates the
    owned buffers in place: each test needs an unmasked solve. The
    solve's loan is reclaimed so the error code can be injected into
    the recovered status-codes slot before the test builds its
    result.
    """
    solver = error_injection_solver

    solver.solve(
        initial_values={
            "x0": [1.0, 2.0, 3.0],
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.0],
        },
        parameters={
            "p0": [0.1, 0.2, 0.3],
            "p1": [0.1, 0.2, 0.3],
            "p2": [0.1, 0.2, 0.3],
        },
        duration=0.1,
    )
    outputs = solver.kernel.output_arrays
    outputs.reclaim_or_release_loan()
    outputs.host.status_codes.array[1] = 1
    return solver


class TestNaNProcessing:
    """Test NaN processing by manually modifying status codes.

    These tests manually set status codes to test error handling
    without running expensive failing solves.
    """

    def test_nan_processing_with_simulated_errors(
        self, solved_batch_solver_errorcode
    ):
        """Verify NaN processing works by manually setting error codes."""
        result = SolveResult.from_solver(
            solved_batch_solver_errorcode, nan_error_trajectories=True
        )

        # Verify run 1 is all NaN
        run_slice = (slice(None), slice(None), 1)
        assert np.all(np.isnan(result.time_domain_array[run_slice]))
        if result.summaries_array.size > 0:
            assert np.all(np.isnan(result.summaries_array[run_slice]))

        # Verify runs 0 and 2 are NOT NaN
        assert not np.all(np.isnan(result.time_domain_array[..., 0]))
        assert not np.all(np.isnan(result.time_domain_array[..., 2]))

    def test_nan_disabled_preserves_error_data(
        self, solved_batch_solver_errorcode
    ):
        """Verify nan_error_trajectories=False preserves data even with errors."""
        result = SolveResult.from_solver(
            solved_batch_solver_errorcode, nan_error_trajectories=False
        )
        assert not np.all(np.isnan(result.time_domain_array[..., 1]))

    def test_successful_runs_unchanged_with_nan_enabled(
        self, solved_batch_solver_errorcode
    ):
        """Verify successful runs are not modified when NaN processing enabled."""
        outputs = solved_batch_solver_errorcode.kernel.output_arrays
        outputs.host.status_codes.array[1] = 0
        result = SolveResult.from_solver(
            solved_batch_solver_errorcode, nan_error_trajectories=True
        )

        # All runs should have status code 0 (success)
        assert np.all(result.status_codes == 0)

        # No data should be NaN
        assert not np.any(np.isnan(result.time_domain_array))
        if result.summaries_array.size > 0:
            assert not np.any(np.isnan(result.summaries_array))

    def test_multiple_errors_all_set_to_nan(
        self, solved_batch_solver_errorcode
    ):
        """Verify multiple failed runs all get NaN'd."""
        outputs = solved_batch_solver_errorcode.kernel.output_arrays
        outputs.host.status_codes.array[0] = 2
        outputs.host.status_codes.array[2] = 3

        result = SolveResult.from_solver(
            solved_batch_solver_errorcode, nan_error_trajectories=True
        )

        assert np.all(np.isnan(result.time_domain_array[..., 0]))
        assert np.all(np.isnan(result.time_domain_array[..., 2]))
        assert np.all(np.isnan(result.time_domain_array[..., 1]))


@pytest.fixture(scope="session")
def solved_summary_only_solver(system):
    """Solver run with a summary-only, fusing output configuration.

    No state or observable output is requested, and the requested
    summary metrics (``mean``, ``max``, ``min``) trigger the
    ``extrema`` combined-metric substitution, so the result
    exercises both the summary-only legend path and requested-name
    reporting for fused metrics. The output set is a construction
    setting, so the solver holds one configuration for its whole
    life.
    """
    solver = Solver(
        system,
        output_types=["mean", "max", "min"],
        stream_group="summary_only",
    )

    solver.solve(
        initial_values={
            "x0": [1.0, 2.0, 3.0],
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.0],
        },
        parameters={
            "p0": [0.1, 0.2, 0.3],
            "p1": [0.1, 0.2, 0.3],
            "p2": [0.1, 0.2, 0.3],
        },
        duration=0.1,
    )
    return solver


class TestSummaryOnlyRegression:
    """Summary-only legends and fused-metric output naming."""

    def test_summary_only_solve_populates_legend(
        self, solved_summary_only_solver
    ):
        """A summary-only solve produces a non-empty
        summaries_legend whose row count matches the variable dimension
        of summaries_array."""
        result = SolveResult.from_solver(solved_summary_only_solver)

        assert result.summaries_legend != {}
        variable_index = result._stride_order.index("variable")
        assert (
            len(result.summaries_legend)
            == result.summaries_array.shape[variable_index]
        )

    def test_fused_extrema_reports_requested_names(
        self, solved_summary_only_solver
    ):
        """The fused max/min metric reports its outputs under the
        requested names, not extrema_1/extrema_2, with max >= min
        elementwise."""
        per_summary = SolveResult.from_solver(
            solved_summary_only_solver
        ).as_numpy_per_summary

        assert "max" in per_summary
        assert "min" in per_summary
        assert "extrema_1" not in per_summary
        assert "extrema_2" not in per_summary
        assert np.all(per_summary["max"] >= per_summary["min"])


class TestSolveSpecFields:
    """Test SolveSpec field attributes."""

    def test_solvespec_has_all_expected_attributes(self):
        """Verify SolveSpec has all required attributes."""
        from cubie.batchsolving.solveresult import SolveSpec

        expected_attrs = [
            "dt",
            "dt_min",
            "dt_max",
            "save_every",
            "summarise_every",
            "sample_summaries_every",
            "atol",
            "rtol",
            "duration",
            "warmup",
            "t0",
            "algorithm",
            "saved_states",
            "saved_observables",
            "summarised_states",
            "summarised_observables",
            "output_types",
            "precision",
        ]

        spec = SolveSpec(
            dt=0.001,
            dt_min=0.0001,
            dt_max=0.01,
            save_every=0.1,
            summarise_every=1.0,
            sample_summaries_every=0.1,
            atol=1e-6,
            rtol=1e-3,
            duration=10.0,
            warmup=0.0,
            t0=0.0,
            algorithm="euler",
            saved_states=["x0"],
            saved_observables=None,
            summarised_states=["x0"],
            summarised_observables=None,
            output_types=["state", "mean"],
            precision="float32",
        )

        for attr in expected_attrs:
            assert hasattr(spec, attr), f"SolveSpec missing attribute: {attr}"


def test_format_time_domain_label_dimensionless_omits_unit():
    """_format_time_domain_label returns the bare label for a
    dimensionless unit instead of appending '[unit]'."""
    from cubie.batchsolving.solveresult import _format_time_domain_label

    assert _format_time_domain_label("x0", "dimensionless") == "x0"


def test_format_time_domain_label_appends_unit():
    """_format_time_domain_label appends the unit when not
    dimensionless."""
    from cubie.batchsolving.solveresult import _format_time_domain_label

    assert _format_time_domain_label("v", "mV") == "v [mV]"


def test_status_messages_property(solver_with_arrays):
    """SolveResult.status_messages decodes the stored status codes."""
    result = SolveResult.from_solver(solver_with_arrays)
    messages = result.status_messages
    assert isinstance(messages, dict)
    assert messages == {}


def test_from_solver_hands_over_buffers(solver_with_arrays):
    """The result owns the solver's host buffers without copying."""
    outputs = solver_with_arrays.kernel.output_arrays
    # Earlier results may have died without a reclaim; recover the
    # buffers so the slot contents can be captured for comparison.
    outputs.reclaim_or_release_loan()
    state_buffer = solver_with_arrays.state
    status_buffer = solver_with_arrays.status_codes
    result = SolveResult.from_solver(solver_with_arrays)

    assert result.state is state_buffer
    assert result.status_codes is status_buffer
    assert solver_with_arrays.kernel.output_arrays.state is None
    result.close()


def test_from_solver_result_carries_stream(solver_with_arrays):
    """The result records the kernel's memory-manager stream."""
    result = SolveResult.from_solver(solver_with_arrays)
    assert result.stream is solver_with_arrays.kernel.stream
    assert result.stream is not None
    result.close()


def test_device_result_from_solver(solver_with_arrays):
    """DeviceSolveResult holds the solver's device buffers and stream."""
    result = DeviceSolveResult.from_solver(solver_with_arrays)

    assert result.state is solver_with_arrays.device_state
    assert (
        result.status_codes
        is solver_with_arrays.device_status_codes
    )
    assert result.stream is solver_with_arrays.kernel.stream
    assert result.active_outputs == solver_with_arrays.active_outputs
    assert (
        result.stride_order
        == solver_with_arrays.kernel.output_arrays.host.state.stride_order
    )


def test_device_result_handles_match_host_buffers(solver_with_arrays):
    """Synchronized handle copies match the host arrays of the solve."""
    outputs = solver_with_arrays.kernel.output_arrays
    outputs.reclaim_or_release_loan()
    host_state = np.array(solver_with_arrays.state, copy=True)
    host_status = np.array(
        solver_with_arrays.status_codes, copy=True
    )
    result = DeviceSolveResult.from_solver(solver_with_arrays)
    result.stream.synchronize()
    np.testing.assert_array_equal(
        result.state.copy_to_host(), host_state
    )
    np.testing.assert_array_equal(
        result.status_codes.copy_to_host(), host_status
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [STATE_ONLY_NO_SUMMARIES],
    indirect=True,
)
def test_as_pandas_without_summaries_returns_a_dataframe(
    solver_with_arrays,
):
    """as_pandas returns a usable summaries DataFrame with no metrics.

    When no summary metrics are active, each run contributes an empty
    per-run DataFrame so the concatenated ``summaries`` entry is a
    real (empty) DataFrame.
    """
    result = SolveResult.from_solver(solver_with_arrays)
    assert result.active_outputs.state_summaries is False
    assert result.active_outputs.observable_summaries is False

    pandas_dict = result.as_pandas
    assert isinstance(pandas_dict["summaries"], pd.DataFrame)
    assert pandas_dict["summaries"].empty
