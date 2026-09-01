from typing import Iterable

import pytest
import numpy as np

from tests._utils import (
    ALGORITHM_CHAIN_SETS,
    FLOAT64_PRECISION,
    _build_solver_instance,
)
from tests.system_fixtures import (
    build_diagonally_dominant_system,
    build_three_state_nonlinear_system,
)

from cubie import create_ODE_system
from cubie.batchsolving.solver import Solver, solve_ivp
from cubie.batchsolving.solveresult import (
    DeviceSolveResult,
    SolveResult,
    SolveSpec,
)
from cubie.batchsolving.BatchInputHandler import BatchInputHandler
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.cuda_simsafe import (
    ALL_UNROLL_PARAMETERS,
    UnrollFlags,
    cuda,
    is_device_array,
)
from cubie.integrators.matrix_free_solvers.bicgstab_solver import (
    BiCGSTABSolver,
)
from tests._utils import (
    DEVICE_SOLVE_SETTINGS,
    FIXED_EULER_TIMED_STATE,
    MOVABLE_LOCATION_KEYS,
    UNROLL_SETTINGS,
)


@pytest.fixture(scope="session")
def simple_initial_values(system):
    """Create simple initial values for testing."""
    return {
        list(system.initial_values.names)[0]: [0.1, 0.5],
        list(system.initial_values.names)[1]: [0.2, 0.6],
    }


@pytest.fixture(scope="session")
def simple_parameters(system):
    """Create simple parameters for testing."""
    return {
        list(system.parameters.names)[0]: [1.0, 2.0],
        list(system.parameters.names)[1]: [0.5, 1.5],
    }


@pytest.fixture(scope="session")
def solved_solver_simple(
    solver,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test basic solve functionality."""
    result = solver.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.1,
        settling_time=0.0,
        blocksize=32,
        grid_type="combinatorial",
    )

    return solver, result


def test_solver_initialization(solver, system):
    """Test that the solver initializes correctly."""
    assert solver is not None
    assert solver.system_interface is not None
    assert solver.input_handler is not None
    assert solver.kernel is not None
    assert isinstance(solver.system_interface, SystemInterface)
    assert isinstance(solver.input_handler, BatchInputHandler)


def test_solver_properties(solver, solver_settings):
    """Test that solver properties return expected values."""
    assert solver.precision == solver_settings["precision"]
    assert solver.system_sizes is not None
    assert solver.output_array_heights is not None
    assert solver.stream_group == solver_settings["stream_group"]


@pytest.mark.parametrize(
    "solver_settings_override", [DEVICE_SOLVE_SETTINGS], indirect=True
)
def test_manual_proportion(solver):
    """A manual memory proportion reaches the built solver."""
    assert solver.mem_proportion == 0.1


def test_variable_indices_methods(solver, system):
    """Test methods for getting variable indices."""
    state_names = list(system.initial_values.names)
    observable_names = (
        list(system.observables.names)
        if hasattr(system.observables, "names")
        else []
    )

    # Test with specific labels
    if len(state_names) > 0:
        indices = solver.get_state_indices([state_names[0]])
        assert isinstance(indices, np.ndarray)
        assert len(indices) >= 1

    # Test with None (all states)
    all_state_indices = solver.get_state_indices(None)
    assert isinstance(all_state_indices, np.ndarray)

    if len(observable_names) > 0:
        obs_indices = solver.get_observable_indices([observable_names[0]])
        assert isinstance(obs_indices, np.ndarray)


def test_saved_variables_properties(solver):
    """Test properties related to saved variables."""
    assert solver.saved_state_indices is not None
    assert solver.saved_observable_indices is not None
    assert solver.saved_states is not None
    assert solver.saved_observables is not None

    # Test that saved states/observables return lists of strings
    assert isinstance(solver.saved_states, list)
    assert isinstance(solver.saved_observables, list)


def test_summarised_variables_properties(solver):
    """Test properties related to summarised variables."""
    assert solver.summarised_state_indices is not None
    assert solver.summarised_observable_indices is not None
    assert solver.summarised_states is not None
    assert solver.summarised_observables is not None

    # Test that summarised states/observables return lists of strings
    assert isinstance(solver.summarised_states, list)
    assert isinstance(solver.summarised_observables, list)


def test_output_properties(solver):
    """Test output-related properties."""
    assert solver.active_outputs is not None
    assert solver.output_types is not None
    assert isinstance(solver.output_types, Iterable)


def test_solve_info_property(
    precision,
    solver,
    solver_settings,
    tolerance,
):
    """Test that solve_info returns a valid SolveSpec."""
    previous_duration = solver.kernel.duration
    solver.kernel.duration = 1.0
    solve_info = solver.solve_info
    assert isinstance(solve_info, SolveSpec)

    if solver.kernel.single_integrator.is_adaptive:
        assert solve_info.dt_min == pytest.approx(
            solver_settings["dt_min"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        assert solve_info.dt_max == pytest.approx(
            solver_settings["dt_max"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        assert solve_info.atol == pytest.approx(
            solver_settings["atol"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        assert solve_info.rtol == pytest.approx(
            solver_settings["rtol"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
    else:
        assert solve_info.dt == pytest.approx(
            solver_settings["dt"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        assert solve_info.dt_min == pytest.approx(
            solver_settings["dt"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        assert solve_info.dt_max == pytest.approx(
            solver_settings["dt"],
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
    assert solve_info.save_every == pytest.approx(
        solver_settings["save_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert solve_info.summarise_every == pytest.approx(
        solver_settings["summarise_every"],
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )

    assert solve_info.algorithm == solver_settings["algorithm"]
    assert solve_info.output_types == tuple(
        solver_settings["output_types"]
    )
    assert solve_info.precision == solver_settings["precision"]

    # Test that solver kernel properties are correctly exposed
    assert solve_info.duration == solver.duration
    assert solve_info.warmup == solver.warmup
    assert solve_info.t0 == solver.t0

    # Test that variable lists are correctly exposed
    assert solve_info.saved_states == solver.saved_states
    assert solve_info.saved_observables == solver.saved_observables
    assert solve_info.summarised_states == solver.summarised_states
    assert solve_info.summarised_observables == solver.summarised_observables

    assert hasattr(solve_info, "summarised_observables")

    # Hand the session-scoped solver back at its previous duration.
    solver.kernel.duration = previous_duration


def test_solve_info_cached(solver_mutable):
    """solve_info reuses its SolveSpec until settings or times change."""
    solver = solver_mutable
    solver.kernel.duration = 1.0
    first = solver.solve_info
    assert solver.solve_info is first

    solver.kernel.duration = 2.0
    changed_duration = solver.solve_info
    assert changed_duration is not first
    assert changed_duration.duration == solver.duration

    solver.update({"save_every": solver.save_every})
    assert solver.solve_info is not changed_duration


def test_solve_basic(
    solver_mutable,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test basic solve functionality."""
    result = solver_mutable.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.05,
        settling_time=0.0,
        blocksize=32,
        grid_type="combinatorial",
    )

    assert isinstance(result, SolveResult)
    assert hasattr(result, "time_domain_array")
    assert hasattr(result, "summaries_array")


def test_compile_then_solve(
    solver_mutable,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Compile prepares the batch; a following solve is valid."""
    expected_inits, expected_params = solver_mutable.build_grid(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        grid_type="combinatorial",
    )
    solver_mutable.compile(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.05,
        grid_type="combinatorial",
    )
    kernel = solver_mutable.kernel
    assert kernel.run_params.duration == 0.05
    assert kernel.run_params.runs == expected_inits.shape[1]
    assert (
        kernel.input_arrays.device_initial_values.shape
        == expected_inits.shape
    )
    assert (
        kernel.input_arrays.device_parameters.shape
        == expected_params.shape
    )
    result = solver_mutable.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.05,
        settling_time=0.0,
        blocksize=32,
        grid_type="combinatorial",
    )
    assert isinstance(result, SolveResult)
    assert np.all(np.isfinite(result.state))


@pytest.mark.nocudasim
def test_compile_publishes_solve_specialization(
    solver_mutable,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Compile publishes the exact specialization a solve reuses."""
    solver_mutable.compile(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.05,
        blocksize=32,
        grid_type="combinatorial",
    )
    dispatcher = solver_mutable.kernel.kernel
    keys_after_compile = set(dispatcher.overloads)
    assert len(keys_after_compile) == 1
    result = solver_mutable.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.05,
        settling_time=0.0,
        blocksize=32,
        grid_type="combinatorial",
    )
    assert solver_mutable.kernel.kernel is dispatcher
    assert set(dispatcher.overloads) == keys_after_compile
    assert np.all(np.isfinite(result.state))


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["firk"]],
    indirect=True,
)
def test_solve_firk_with_driver_arrays(
    solver_mutable,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """A driven FIRK solve exercises the stage driver stack."""
    result = solver_mutable.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.1,
        settling_time=0.0,
        blocksize=32,
        grid_type="verbatim",
    )
    assert not np.any(result.status_codes)
    assert np.all(np.isfinite(result.time_domain_array))


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["erk"]],
    indirect=True,
)
def test_algorithm_hot_swap_after_solve(
    solver_mutable,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Swapping the algorithm after a solve leaves the solver usable."""
    solve_kwargs = dict(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.2,
        save_every=0.02,
        settling_time=0.0,
        blocksize=32,
        grid_type="combinatorial",
    )
    first = solver_mutable.solve(**solve_kwargs)
    assert not np.any(first.status_codes)

    solver_mutable.update(algorithm="bogacki-shampine-32")
    second = solver_mutable.solve(**solve_kwargs)
    assert not np.any(second.status_codes)
    assert np.all(np.isfinite(second.time_domain_array))


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["firk"]],
    indirect=True,
)
def test_linear_solver_hot_swap_after_solve(
    solver_mutable,
    solver_settings,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Swapping the linear-solver class preserves solve results."""
    solve_kwargs = dict(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.1,
        save_every=0.02,
        settling_time=0.0,
        blocksize=32,
        grid_type="verbatim",
    )
    first = solver_mutable.solve(**solve_kwargs)
    assert not np.any(first.status_codes)

    solver_mutable.update(linear_correction_type="bicgstab")
    algo_step = solver_mutable.kernel.single_integrator._algo_step
    assert isinstance(algo_step.linear_solver, BiCGSTABSolver)
    assert algo_step.linear_correction_type == "bicgstab"
    second = solver_mutable.solve(**solve_kwargs)
    assert not np.any(second.status_codes)
    # The fixed controller walks both solves through identical steps,
    # so the trajectories differ only by inner-solver convergence.
    newton_atol = solver_settings["newton_atol"]
    if newton_atol is None:
        newton_atol = solver_settings["atol"] / 10.0
    tolerance = 100.0 * float(newton_atol)
    assert np.allclose(
        second.time_domain_array,
        first.time_domain_array,
        rtol=tolerance,
        atol=tolerance,
    )


def test_solve_with_different_grid_types(
    solver_mutable,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Test solve with different grid types."""
    # Test combinatorial grid
    result_comb = solver_mutable.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.05,
        grid_type="combinatorial",
    )

    assert isinstance(result_comb, SolveResult)

    # Test verbatim grid with properly shaped arrays
    state_names = list(simple_initial_values.keys())
    param_names = list(simple_parameters.keys())

    # Create verbatim arrays (same length for all variables)
    verbatim_initial_values = {
        state_names[0]: [0.1, 0.5],
        state_names[1]: [0.2, 0.6],
    }
    verbatim_parameters = {
        param_names[0]: [1.0, 2.0],
        param_names[1]: [0.5, 1.5],
    }
    result_verb = solver_mutable.solve(
        initial_values=verbatim_initial_values,
        parameters=verbatim_parameters,
        drivers=driver_settings,
        duration=0.05,
        grid_type="verbatim",
    )

    assert isinstance(result_verb, SolveResult)


def test_solve_result_representations(
    solver_mutable,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """The result derives its RAM representations on demand."""
    result = solver_mutable.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.05,
    )

    assert isinstance(result, SolveResult)
    as_numpy = result.as_numpy
    assert isinstance(as_numpy["time_domain_array"], np.ndarray)
    assert np.array_equal(
        as_numpy["time_domain_array"], result.time_domain_array
    )


def test_full_results_carry_stream(unchunked_solved_solver):
    """Host results expose the stream the solve ran on."""
    solver, result = unchunked_solved_solver
    assert result.stream is solver.kernel.stream
    assert result.stream is not None


@pytest.mark.parametrize(
    "solver_settings_override", [DEVICE_SOLVE_SETTINGS], indirect=True
)
def test_device_results_match_host(
    solver_mutable,
    solver_settings,
    system,
    precision,
    driver_settings,
):
    """Device results hold the same values as a host solve."""
    solver = solver_mutable
    n_runs = 5
    inits = np.ones((system.sizes.states, n_runs), dtype=precision)
    params = np.ones(
        (system.sizes.parameters, n_runs), dtype=precision
    )

    host = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=solver_settings["duration"],
        nan_error_trajectories=False,
    )
    host_state = np.array(host.state, copy=True)
    host_status = np.array(host.status_codes, copy=True)

    device = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=solver_settings["duration"],
        on_device=True,
    )
    assert isinstance(device, DeviceSolveResult)
    assert is_device_array(device.state)
    assert is_device_array(device.status_codes)
    assert device.stream is solver.kernel.stream

    # The documented host-read pattern: synchronize the solve's
    # stream, then copy the handles.
    device.stream.synchronize()
    np.testing.assert_array_equal(
        device.state.copy_to_host(), host_state
    )
    np.testing.assert_array_equal(
        device.status_codes.copy_to_host(), host_status
    )


@pytest.mark.parametrize(
    "solver_settings_override", [DEVICE_SOLVE_SETTINGS], indirect=True
)
@pytest.mark.parametrize("forced_free_mem", [600], indirect=True)
def test_device_results_chunked_raises(
    low_mem_solver,
    solver_settings,
    system,
    precision,
    driver_settings,
):
    """Device results refuse a run that chunks along the run axis."""
    n_runs = 5
    inits = np.ones((system.sizes.states, n_runs), dtype=precision)
    params = np.ones(
        (system.sizes.parameters, n_runs), dtype=precision
    )
    with pytest.raises(ValueError, match="single chunk"):
        low_mem_solver.solve(
            inits,
            params,
            drivers=driver_settings,
            duration=solver_settings["duration"],
            on_device=True,
        )


@pytest.mark.parametrize(
    "solver_settings_override", [DEVICE_SOLVE_SETTINGS], indirect=True
)
def test_device_inputs_match_host_inputs(
    solver_mutable,
    solver_settings,
    system,
    precision,
    driver_settings,
):
    """Device-array inputs reproduce a host-input solve exactly."""
    solver = solver_mutable
    n_runs = 4
    rng_vals = np.linspace(0.5, 1.5, system.sizes.states * n_runs)
    inits = rng_vals.reshape(
        (system.sizes.states, n_runs)
    ).astype(precision)
    params = np.ones(
        (system.sizes.parameters, n_runs), dtype=precision
    )

    host_result = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=solver_settings["duration"],
    )
    host_state = host_result.time_domain_array.copy()

    d_inits = cuda.to_device(inits)
    d_params = cuda.to_device(params)
    device_result = solver.solve(
        d_inits,
        d_params,
        drivers=driver_settings,
        duration=solver_settings["duration"],
    )
    # The caller's arrays are wired straight into the kernel.
    input_arrays = solver.kernel.input_arrays
    assert input_arrays.has_device_inputs
    assert input_arrays.device_initial_values is d_inits
    assert input_arrays.device_parameters is d_params
    assert solver.initial_values is d_inits
    np.testing.assert_array_equal(
        device_result.time_domain_array, host_state
    )

    # A later host-input solve reallocates and still matches.
    host_again = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=solver_settings["duration"],
    )
    assert not input_arrays.has_device_inputs
    np.testing.assert_array_equal(
        host_again.time_domain_array, host_state
    )


@pytest.mark.parametrize(
    "solver_settings_override", [DEVICE_SOLVE_SETTINGS], indirect=True
)
def test_device_inputs_device_results_roundtrip(
    solver_mutable,
    solver_settings,
    system,
    precision,
    driver_settings,
):
    """A fully device-resident solve returns valid device handles."""
    solver = solver_mutable
    n_runs = 3
    inits = np.ones((system.sizes.states, n_runs), dtype=precision)
    params = np.ones(
        (system.sizes.parameters, n_runs), dtype=precision
    )

    reference = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=solver_settings["duration"],
        nan_error_trajectories=False,
    )
    ref_state = np.array(reference.state, copy=True)

    device = solver.solve(
        cuda.to_device(inits),
        cuda.to_device(params),
        drivers=driver_settings,
        duration=solver_settings["duration"],
        on_device=True,
    )
    device.stream.synchronize()
    np.testing.assert_array_equal(
        device.state.copy_to_host(), ref_state
    )


@pytest.mark.parametrize(
    "solver_settings_override", [DEVICE_SOLVE_SETTINGS], indirect=True
)
@pytest.mark.parametrize("forced_free_mem", [400], indirect=True)
def test_device_inputs_chunked_raises(
    low_mem_solver,
    solver_settings,
    system,
    precision,
    driver_settings,
):
    """Device-array inputs refuse a run that chunks.

    Free memory is pinned low enough that the output arrays alone
    force chunking: attached device inputs queue no input buffers,
    so they cannot contribute to the footprint.
    """
    n_runs = 5
    inits = np.ones((system.sizes.states, n_runs), dtype=precision)
    params = np.ones(
        (system.sizes.parameters, n_runs), dtype=precision
    )
    with pytest.raises(ValueError, match="single chunk"):
        low_mem_solver.solve(
            cuda.to_device(inits),
            cuda.to_device(params),
            drivers=driver_settings,
            duration=solver_settings["duration"],
        )


@pytest.mark.parametrize(
    "solver_settings_override", [DEVICE_SOLVE_SETTINGS], indirect=True
)
def test_resident_device_inputs_run_without_an_upload(
    solver_mutable,
    solver_settings,
    system,
    precision,
    driver_settings,
):
    """Resident device inputs run in place with no upload."""
    solver = solver_mutable
    n_runs = 4
    inits = np.linspace(
        0.5, 1.5, system.sizes.states * n_runs
    ).reshape((system.sizes.states, n_runs)).astype(precision)
    params = np.ones(
        (system.sizes.parameters, n_runs), dtype=precision
    )

    host = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=solver_settings["duration"],
        nan_error_trajectories=False,
    )
    host_state = np.array(host.state, copy=True)
    d_inits = solver.device_initial_values
    d_params = solver.device_parameters
    input_arrays = solver.kernel.input_arrays
    assert is_device_array(d_inits)
    assert is_device_array(d_params)
    assert d_inits is input_arrays.device.initial_values.array
    assert d_params is input_arrays.device.parameters.array

    # The host arrays change in place after their upload.
    inits[:] = 2.0 * inits
    device = solver.solve(
        d_inits,
        d_params,
        drivers=driver_settings,
        duration=solver_settings["duration"],
        on_device=True,
    )
    device.stream.synchronize()
    assert not input_arrays.has_device_inputs
    assert solver.device_initial_values is d_inits
    assert solver.device_parameters is d_params
    np.testing.assert_array_equal(
        device.state.copy_to_host(), host_state
    )

    # The attached host arrays upload into the same buffers.
    changed = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=solver_settings["duration"],
        nan_error_trajectories=False,
    )
    changed_state = np.array(changed.state, copy=True)
    assert solver.device_initial_values is d_inits
    assert solver.device_parameters is d_params
    reference = solver.solve(
        inits.copy(),
        params.copy(),
        drivers=driver_settings,
        duration=solver_settings["duration"],
        nan_error_trajectories=False,
    )
    np.testing.assert_array_equal(changed_state, reference.state)


@pytest.mark.parametrize(
    "solver_settings_override", [DEVICE_SOLVE_SETTINGS], indirect=True
)
@pytest.mark.parametrize("forced_free_mem", [600], indirect=True)
def test_device_inputs_after_a_chunked_run_raise(
    low_mem_solver,
    solver_settings,
    system,
    precision,
    driver_settings,
):
    """Resident-input accessors raise after a chunked run."""
    n_runs = 5
    inits = np.ones((system.sizes.states, n_runs), dtype=precision)
    params = np.ones(
        (system.sizes.parameters, n_runs), dtype=precision
    )
    low_mem_solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=solver_settings["duration"],
    )
    assert low_mem_solver.chunks > 1
    with pytest.raises(ValueError, match="one chunk"):
        low_mem_solver.device_initial_values
    with pytest.raises(ValueError, match="one chunk"):
        low_mem_solver.device_parameters


def test_update_basic(solver_mutable, tolerance, precision):
    """Test basic update functionality updating the fixed step size."""
    solver = solver_mutable
    original_dt = solver.dt
    # Choose a new dt distinct from the original
    new_dt = precision(
        original_dt * 0.5 if original_dt not in (0, None) else 1e-7
    )
    updated_keys = solver.update({"dt": new_dt})
    assert "dt" in updated_keys
    # For fixed-step integrators dt should now reflect the new value
    if not getattr(
        solver.kernel.single_integrator, "is_adaptive", False
    ):
        assert solver.dt == pytest.approx(
            new_dt, rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )


def test_update_with_kwargs(solver_mutable, tolerance):
    """Test update with keyword arguments."""
    solver = solver_mutable
    original_dt = solver.kernel.single_integrator.dt

    updated_keys = solver.update({}, dt=1e-8)

    assert "dt" in updated_keys
    assert solver.kernel.single_integrator.dt == pytest.approx(
        1e-8,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert solver.kernel.single_integrator.dt != pytest.approx(
        original_dt,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )


def test_update_unrecognized_keys(solver_mutable):
    """Test that update raises KeyError for unrecognized keys."""
    solver = solver_mutable
    with pytest.raises(KeyError):
        solver.update({"nonexistent_parameter": 42})


def test_update_silent_mode(precision, solver_mutable):
    """Test update in silent mode ignores unrecognized keys."""
    solver = solver_mutable
    original_dt = solver.dt
    # Choose a new dt distinct from the original
    new_dt = precision(
        original_dt * 0.5 if original_dt not in (0, None) else 1e-7
    )
    updated_keys = solver.update(
        {"dt": new_dt, "nonexistent_parameter": 42}, silent=True
    )

    assert "dt" in updated_keys

    assert "nonexistent_parameter" not in updated_keys
    assert solver.kernel.dt == new_dt


def test_update_lineinfo(solver_mutable):
    """Test that update routes lineinfo through compile settings."""
    solver = solver_mutable

    updated_keys = solver.update({"lineinfo": True})
    assert "lineinfo" in updated_keys
    assert solver.kernel.compile_settings.lineinfo is True
    assert not solver.kernel.cache_valid

    updated_keys = solver.update({"lineinfo": False})
    assert "lineinfo" in updated_keys
    assert solver.kernel.compile_settings.lineinfo is False


def test_lineinfo_constructor_propagates_to_children(precision):
    """Explicit lineinfo reaches every child factory's compile settings.

    Uses a private system: lineinfo propagates into the system's own
    compile settings, and flipping the shared session system would
    leak lineinfo-flavoured device functions into every kernel later
    tests build on the same worker.
    """
    system = build_three_state_nonlinear_system(precision)
    solver = Solver(system, algorithm="euler", lineinfo=True)

    kernel = solver.kernel
    assert kernel.compile_settings.lineinfo is True

    integrator = kernel.single_integrator
    assert integrator._loop.compile_settings.lineinfo is True
    assert integrator._algo_step.compile_settings.lineinfo is True
    assert integrator._step_controller.compile_settings.lineinfo is True
    assert integrator._output_functions.compile_settings.lineinfo is True
    assert integrator._system.compile_settings.lineinfo is True


def test_driver_evaluator_wired_at_construction(precision):
    """A fresh solver wires the interpolator's evaluator in."""
    system = build_three_state_nonlinear_system(precision)
    solver = Solver(system, algorithm="radau")

    integrator = solver.kernel.single_integrator
    evaluator = solver.kernel.driver_interpolator.evaluation_function
    assert (
        integrator._loop.compile_settings.evaluate_driver_at_t
        is evaluator
    )
    assert (
        integrator._algo_step.compile_settings.evaluate_driver_at_t
        is evaluator
    )


def test_driverless_system_has_no_driver_evaluator(precision):
    """A driverless system leaves ``evaluate_driver_at_t`` unset."""
    system = build_diagonally_dominant_system(precision)
    solver = Solver(system, algorithm="radau")

    integrator = solver.kernel.single_integrator
    assert (
        integrator._loop.compile_settings.evaluate_driver_at_t is None
    )
    assert (
        integrator._algo_step.compile_settings.evaluate_driver_at_t
        is None
    )


def test_update_saved_variables(solver_mutable, system):
    """Test updating saved variables with labels."""
    solver = solver_mutable
    state_names = list(system.initial_values.names)
    observable_names = (
        list(system.observables.names)
        if hasattr(system.observables, "names")
        else []
    )

    if len(state_names) > 0 and len(observable_names) > 0:
        all_vars = [state_names[0]]
        if observable_names:
            all_vars.append(observable_names[0])

        updates = {
            "save_variables": all_vars,
        }

        updated_keys = solver.update(updates)

        assert len(updated_keys) > 0


def test_memory_settings_update(solver_mutable):
    """Test updating memory-related settings."""
    solver = solver_mutable
    # Test memory proportion update
    updated_keys = solver.update_memory_settings({"mem_proportion": 0.1})
    assert "mem_proportion" in updated_keys


def test_data_properties_after_solve(solved_solver_simple):
    """Test that data properties are accessible after solving."""
    # These should be accessible after solving
    solver, result = solved_solver_simple

    assert isinstance(result.time_domain_array, np.ndarray)
    assert isinstance(result.summaries_array, np.ndarray)

    assert not np.all(result.time_domain_array == 0)


def test_output_length_and_summaries_length(solver):
    """Test output length and summaries length properties."""
    assert solver.output_length is not None
    assert solver.summaries_length is not None
    assert isinstance(solver.output_length, int)
    assert isinstance(solver.summaries_length, int)


def test_chunk_related_properties(solved_solver_simple):
    """Test chunk-related properties."""
    solver, result = solved_solver_simple
    assert solver.chunks == 1


def test_variable_labels_properties(solver):
    """Test properties that return variable labels."""
    assert solver.input_variables is not None
    assert solver.output_variables is not None
    assert isinstance(solver.input_variables, list)
    assert isinstance(solver.output_variables, list)


# Test the solve_ivp convenience function
def test_solve_ivp_function(
    system, simple_initial_values, simple_parameters, driver_settings
):
    """Test the solve_ivp convenience function."""
    result = solve_ivp(
        system=system,
        y0=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        dt=1e-2,
        save_every=0.02,
        duration=0.05,
        summarise_every=0.04,
        output_types=["state", "time", "observables", "mean"],
        method="euler",
        settling_time=0.0,
    )

    assert isinstance(result, SolveResult)


def test_solve_ivp_adaptive_controller(
    system, simple_initial_values, simple_parameters, driver_settings
):
    """solve_ivp with an adaptive controller solves and keeps settings."""
    result = solve_ivp(
        system=system,
        y0=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        method="tsit5",
        step_controller="pi",
        atol=1e-6,
        rtol=1e-6,
        dt_min=1e-7,
        dt_max=0.1,
        save_every=0.02,
        duration=0.05,
        output_types=["state", "time"],
    )

    assert isinstance(result, SolveResult)
    spec = result.solve_settings
    assert spec.atol == pytest.approx(1e-6)
    assert spec.rtol == pytest.approx(1e-6)
    assert spec.save_every == pytest.approx(0.02)


def test_solve_ivp_forwards_save_every_and_settling_time(
    system, simple_initial_values, simple_parameters, driver_settings
):
    """solve_ivp applies save_every and settling_time to the solve."""
    result = solve_ivp(
        system=system,
        y0=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        method="euler",
        dt=1e-2,
        save_every=0.04,
        duration=0.08,
        settling_time=0.04,
        output_types=["state", "time"],
    )

    spec = result.solve_settings
    assert spec.save_every == pytest.approx(0.04)
    assert spec.warmup == pytest.approx(0.04)
    times = np.asarray(result.time)
    times = times.reshape(times.shape[0], -1)[:, 0]
    assert np.diff(times) == pytest.approx(0.04)


def test_solve_ivp_accepts_callable():
    """solve_ivp builds the system from a SciPy-style callable."""
    def vdp(t, y, mu):
        return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

    solve_kwargs = dict(
        y0={"x": [1.0], "v": [0.0]},
        parameters={"mu": [1.5]},
        dt=1e-2,
        duration=0.05,
        save_every=0.01,
        output_types=["state"],
        method="euler",
    )
    result = solve_ivp(vdp, **solve_kwargs)
    assert isinstance(result, SolveResult)
    direct = np.asarray(result.as_numpy["time_domain_array"])
    assert np.all(np.isfinite(direct))

    from cubie import create_ODE_system

    prebuilt = create_ODE_system(
        vdp, states={"x": 1.0, "v": 0.0}, parameters={"mu": 1.5}
    )
    result_two_step = solve_ivp(prebuilt, **solve_kwargs)
    two_step = np.asarray(
        result_two_step.as_numpy["time_domain_array"]
    )
    assert np.array_equal(direct, two_step)


def test_solve_ivp_accepts_equation_strings():
    """solve_ivp builds the system from equation strings."""
    result = solve_ivp(
        ["dx = v", "dv = mu * (1 - x*x) * v - x"],
        y0={"x": [1.0], "v": [0.0]},
        parameters={"mu": [1.5]},
        dt=1e-2,
        duration=0.05,
        save_every=0.01,
        output_types=["state"],
        method="euler",
    )
    assert isinstance(result, SolveResult)
    values = np.asarray(result.as_numpy["time_domain_array"])
    assert np.all(np.isfinite(values))


def test_solver_routes_system_ordering_settings_before_kernel(precision):
    """Explicit and loose system settings reach the ODE before build."""
    explicit_system = create_ODE_system(
        "dx = -x",
        states={"x": 1.0},
        precision=precision,
        name="solver_explicit_ordering",
    )
    explicit_solver = Solver(
        explicit_system,
        system_settings={"operation_ordering": "liveness_auto"},
        algorithm="euler",
        auto_memory=False,
    )
    try:
        assert (
            explicit_solver.system.operation_ordering
            == "liveness_auto"
        )
    finally:
        explicit_solver.close()

    loose_system = create_ODE_system(
        "dx = -x",
        states={"x": 1.0},
        precision=precision,
        name="solver_loose_ordering",
    )
    loose_solver = Solver(
        loose_system,
        operation_ordering="greedy",
        algorithm="euler",
        auto_memory=False,
    )
    try:
        assert loose_solver.system.operation_ordering == "greedy"
        recognised = loose_solver.update(
            system_settings={"operation_ordering": "kahn"}
        )
        assert "system_settings" in recognised
        assert loose_solver.system.operation_ordering == "kahn"
        recognised = loose_solver.update(
            operation_ordering="dfs"
        )
        assert "operation_ordering" in recognised
        assert loose_solver.system.operation_ordering == "dfs"
    finally:
        loose_solver.close()


def test_solve_ivp_routes_loose_operation_ordering():
    """The convenience API accepts the ODE compile setting loosely."""
    result = solve_ivp(
        "dx = -x",
        y0={"x": [1.0]},
        method="euler",
        dt=1e-2,
        duration=0.02,
        save_every=0.01,
        output_types=["state"],
        operation_ordering="liveness_auto",
    )
    values = np.asarray(result.as_numpy["time_domain_array"])
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize(
    "bad_parameters",
    [np.array([[0.5]]), [0.5], (0.5,)],
    ids=["ndarray", "list", "tuple"],
)
def test_solve_ivp_raw_equations_reject_array_parameters(bad_parameters):
    """Raw-equation solve_ivp needs named parameters, not sequences."""
    def decay(t, y, k):
        return [-k * y[0]]

    with pytest.raises(TypeError, match="dict"):
        solve_ivp(
            decay,
            y0={"x": [1.0]},
            parameters=bad_parameters,
            duration=0.05,
            method="euler",
        )


def test_solver_with_different_algorithms(solver, system, solver_settings):
    """Test solver with different algorithms.

    The shared solver already carries the chain's algorithm, so only
    the second algorithm needs a construction of its own.
    """
    assert solver.kernel.algorithm == solver_settings["algorithm"]

    other_solver = Solver(
        system,
        algorithm="backwards_euler_pc",
        dt_min=solver_settings["dt_min"],
        dt=solver_settings["dt"],
        dt_max=solver_settings["dt_max"],
        precision=solver_settings["precision"],
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
        loop_settings={"save_every": solver_settings["save_every"]},
    )

    assert other_solver is not None
    assert other_solver.kernel.algorithm == "backwards_euler_pc"


def test_solver_output_types(system, solver_settings):
    """Test solver with different output types."""
    output_types_list = [
        ["state"],
        ["state", "observables"],
        ["state", "observables", "mean"],
        ["state", "observables", "mean", "max", "rms"],
    ]

    for output_types in output_types_list:
        solver = Solver(
            system,
            output_types=output_types,
            dt_min=solver_settings["dt_min"],
            dt_max=solver_settings["dt_max"],
            precision=solver_settings["precision"],
            memory_manager=solver_settings["memory_manager"],
            stream_group=solver_settings["stream_group"],
            loop_settings={"save_every": solver_settings["save_every"]},
        )

        assert solver.output_types == tuple(output_types)


def test_solver_summary_legend(solver):
    """Test that summary legend property works."""
    legend = solver.summary_legend_per_variable
    assert isinstance(legend, dict)


def test_solver_save_time_property(solver):
    """Test save_time property."""
    save_time = solver.save_time
    assert isinstance(save_time, bool)


def test_solver_state_and_observable_summaries(solver):
    """Summary buffer properties are readable at any lifecycle stage.

    Before the first solve the slots are unallocated; after a solve
    the buffers belong to the returned result, so the slots may again
    read as None.
    """
    for value in (solver.state_summaries, solver.observable_summaries):
        assert value is None or isinstance(value, np.ndarray)


def test_solver_num_runs_property(solver):
    """Test num_runs property."""
    num_runs = solver.num_runs
    # num_runs might be None before solving, so just check it's accessible
    assert num_runs is not None or num_runs == 1


# ============================================================================
# Time Precision Tests (float64 time accumulation)
# ============================================================================


def _assert_time_parameters_are_float64(solver):
    """Assert a solver's time parameters round-trip as float64."""
    # Set time parameters as float32
    solver.kernel.duration = np.float32(10.0)
    solver.kernel.warmup = np.float32(1.0)
    solver.kernel.t0 = np.float32(5.0)

    # Verify retrieved as float64
    assert isinstance(solver.kernel.duration, (float, np.floating))
    assert isinstance(solver.kernel.warmup, (float, np.floating))
    assert isinstance(solver.kernel.t0, (float, np.floating))

    # Verify values preserved
    assert np.isclose(solver.kernel.duration, 10.0)
    assert np.isclose(solver.kernel.warmup, 1.0)
    assert np.isclose(solver.kernel.t0, 5.0)

    # Time should be float64 even when state precision is float32
    assert solver.kernel.duration == np.float64(10.0)
    assert solver.kernel.t0 == np.float64(5.0)


def test_time_precision_independent_of_state_precision(solver_mutable):
    """Verify time precision is float64 at the default state precision.

    The default chain is float32, so this covers the case where the
    time parameters are wider than the state precision.
    """
    _assert_time_parameters_are_float64(solver_mutable)


@pytest.mark.parametrize(
    "solver_settings_override",
    [FLOAT64_PRECISION],
    indirect=True,
)
def test_time_precision_with_float64_states(solver_mutable):
    """Verify time precision is float64 with float64 states."""
    _assert_time_parameters_are_float64(solver_mutable)


# ============================================================================
# build_grid() Tests
# ============================================================================


def test_build_grid_returns_correct_shape(
    solver, simple_initial_values, simple_parameters
):
    """Test that build_grid returns arrays with correct shapes."""
    inits, params = solver.build_grid(
        simple_initial_values, simple_parameters, grid_type="verbatim"
    )

    assert isinstance(inits, np.ndarray)
    assert isinstance(params, np.ndarray)
    assert inits.ndim == 2
    assert params.ndim == 2
    assert inits.shape[0] == solver.system_sizes.states
    assert params.shape[0] == solver.system_sizes.parameters
    # Verbatim: run count matches input length
    assert inits.shape[1] == params.shape[1]


def test_build_grid_combinatorial(
    solver, simple_initial_values, simple_parameters
):
    """Test that build_grid with combinatorial creates product grid."""
    inits, params = solver.build_grid(
        simple_initial_values, simple_parameters, grid_type="combinatorial"
    )

    # Combinatorial produces more runs than verbatim
    n_init_values = len(list(simple_initial_values.values())[0])
    n_param_values = len(list(simple_parameters.values())[0])
    # Number of runs is product of all value counts
    assert inits.shape[1] >= n_init_values
    assert params.shape[1] >= n_param_values


def test_build_grid_precision(
    solver, simple_initial_values, simple_parameters
):
    """Test that build_grid returns arrays with correct precision."""
    inits, params = solver.build_grid(simple_initial_values, simple_parameters)

    assert inits.dtype == solver.precision
    assert params.dtype == solver.precision


# ============================================================================
# solve() Fast Path Tests
# ============================================================================


def test_solve_array_path_matches_dict_path(
    solver_mutable, simple_initial_values, simple_parameters,
    driver_settings
):
    """Test that array fast path produces same results as dict path."""
    # Solve with dict inputs
    result_dict = solver_mutable.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.05,
        grid_type="verbatim",
    )

    # Build grid and solve with arrays
    inits, params = solver_mutable.build_grid(
        simple_initial_values, simple_parameters, grid_type="verbatim"
    )
    result_array = solver_mutable.solve(
        initial_values=inits,
        parameters=params,
        drivers=driver_settings,
        duration=0.05,
    )

    # Results should match
    assert (
        result_dict.time_domain_array.shape
        == result_array.time_domain_array.shape
    )
    np.testing.assert_allclose(
        result_dict.time_domain_array,
        result_array.time_domain_array,
        rtol=1e-5,
        atol=1e-7,
    )


def test_solve_ivp_positional_argument_order(system, solver_settings):
    """Verify positional args to solve_ivp route correctly.

    Regression test: y0 (states) must go to states bucket,
    parameters must go to params bucket, even without keywords.
    The underlying routing is verified in test_batch_input_handler.py.
    """
    n_states = system.sizes.states
    n_params = system.sizes.parameters

    # Use distinctive values to verify routing
    states = np.full((n_states, 2), 1.5, dtype=system.precision)
    params = np.full((n_params, 2), 99.0, dtype=system.precision)

    # Solve should complete without error using positional args
    result = solve_ivp(
        system,
        states,  # positional: y0
        params,  # positional: parameters
        duration=0.02,
        dt=0.01,
        save_every=0.01,
    )

    # Verify result structure is valid
    assert hasattr(result, "time_domain_array")
    assert hasattr(result, "summaries_array")
    # Verify correct number of runs were executed
    assert result.time_domain_array.shape[2] == 2, (
        "Should have 2 runs from 2-column input arrays"
    )


# ============================================================================
# save_variables and summarise_variables Tests
# ============================================================================


@pytest.mark.parametrize(
    "setting_key,prefix",
    [
        ("save_variables", "saved"),
        ("summarise_variables", "summarised"),
    ],
)
def test_output_variable_labels_separated_for_states_only(
    solver, system, setting_key, prefix
):
    """Resolve state labels into the correct index array.

    This only validates label resolution and separation into state vs
    observable index arrays.
    """
    state_names = list(system.initial_values.names)[:2]

    output_settings = {setting_key: state_names}
    solver.convert_output_labels(output_settings)

    assert setting_key not in output_settings
    assert f"{prefix}_state_indices" in output_settings
    assert len(output_settings[f"{prefix}_state_indices"]) == 2

    obs_key = f"{prefix}_observable_indices"
    if obs_key in output_settings:
        assert len(output_settings[obs_key]) == 0


@pytest.mark.parametrize(
    "setting_key,prefix",
    [
        ("save_variables", "saved"),
        ("summarise_variables", "summarised"),
    ],
)
def test_output_variable_labels_separated_for_observables_only(
    solver, system, setting_key, prefix
):
    """Resolve observable labels into the correct index array."""
    observable_names = list(system.observables.names)
    if len(observable_names) < 1:
        pytest.skip("System fixture has no observables")

    output_settings = {setting_key: observable_names[:1]}
    solver.convert_output_labels(output_settings)

    assert setting_key not in output_settings
    assert f"{prefix}_observable_indices" in output_settings
    assert len(output_settings[f"{prefix}_observable_indices"]) == 1

    state_key = f"{prefix}_state_indices"
    if state_key in output_settings:
        assert len(output_settings[state_key]) == 0


@pytest.mark.parametrize(
    "setting_key,prefix",
    [
        ("save_variables", "saved"),
        ("summarise_variables", "summarised"),
    ],
)
def test_output_variable_labels_separated_for_mixed(
    solver, system, setting_key, prefix
):
    """Resolve a mix of state and observable labels."""
    observable_names = list(system.observables.names)
    if len(observable_names) < 1:
        pytest.skip("System fixture has no observables")

    state_names = list(system.initial_values.names)

    output_settings = {
        setting_key: [state_names[0], observable_names[0]],
    }
    solver.convert_output_labels(output_settings)

    assert setting_key not in output_settings
    assert len(output_settings[f"{prefix}_state_indices"]) == 1
    assert len(output_settings[f"{prefix}_observable_indices"]) == 1


@pytest.mark.parametrize(
    "setting_key,prefix,state_index_key",
    [
        ("save_variables", "saved", "saved_state_indices"),
        ("summarise_variables", "summarised", "summarised_state_indices"),
    ],
)
def test_output_variable_labels_union_with_existing_indices(
    solver, system, setting_key, prefix, state_index_key
):
    """Labels combine with explicit indices via union semantics."""
    state_names = list(system.initial_values.names)

    output_settings = {
        state_index_key: np.array([0], dtype=np.int32),
        setting_key: [state_names[1]],
    }
    solver.convert_output_labels(output_settings)

    result = output_settings[state_index_key]
    assert len(result) == 2
    assert 0 in result
    assert 1 in result


@pytest.mark.parametrize(
    "setting_key,prefix",
    [
        ("save_variables", "saved"),
        ("summarise_variables", "summarised"),
    ],
)
def test_output_variable_labels_empty_list_explicit_none(
    solver, setting_key, prefix
):
    """Empty list explicitly selects no variables for that operation."""
    output_settings = {setting_key: []}
    solver.convert_output_labels(output_settings)

    assert setting_key not in output_settings
    assert f"{prefix}_state_indices" in output_settings
    assert f"{prefix}_observable_indices" in output_settings
    assert len(output_settings[f"{prefix}_state_indices"]) == 0
    assert len(output_settings[f"{prefix}_observable_indices"]) == 0


def test_output_variable_labels_none_means_use_all_defaults(solver):
    """None selects all variables for saving (default configuration)."""
    output_settings = {"save_variables": None}
    solver.convert_output_labels(output_settings)

    n_states = solver.system_sizes.states
    n_observables = solver.system_sizes.observables

    assert "save_variables" not in output_settings
    assert len(output_settings["saved_state_indices"]) == n_states
    assert len(output_settings["saved_observable_indices"]) == n_observables


@pytest.mark.parametrize(
    "setting_key",
    [
        "save_variables",
        "summarise_variables",
    ],
)
def test_output_variable_labels_invalid_name_raises(solver, setting_key):
    """Unknown labels raise a clear error."""
    output_settings = {setting_key: ["nonexistent_variable"]}

    with pytest.raises(ValueError, match="Variables not found"):
        solver.convert_output_labels(output_settings)


def test_save_variables_error_includes_available_names(solver):
    """Test error message includes available variable names."""
    output_settings = {"save_variables": ["nonexistent_variable"]}

    try:
        solver.convert_output_labels(output_settings)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Available states:" in str(e)
        assert "Available observables:" in str(e)


def test_array_only_fast_path(solver):
    """Test array-only parameters don't trigger name resolution."""
    import time

    output_settings = {"saved_state_indices": np.array([0, 1], dtype=np.int32)}

    # Time the fast path (should be very quick)
    start = time.perf_counter()
    for _ in range(1000):
        settings_copy = output_settings.copy()
        solver.convert_output_labels(settings_copy)
    fast_time = time.perf_counter() - start

    # Should be well under 1 second for 1000 iterations
    assert fast_time < 1.0


def test_solve_ivp_with_save_variables(system):
    """Test solve_ivp accepts save_variables and produces correct output."""
    state_names = list(system.initial_values.names)[:2]

    result = solve_ivp(
        system,
        y0={state_names[0]: [1.0, 2.0]},
        parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
        save_variables=state_names,
        save_every=0.01,
        duration=0.02,
        dt=0.01,
        method="euler",
    )

    # Verify result contains saved states
    assert result is not None
    assert hasattr(result, "time_domain_array")
    assert result.time_domain_array is not None
    # Verify shape matches number of save_variables
    # time_domain_array shape is (time, variable, run) by default
    assert result.time_domain_array.shape[1] == len(state_names)


def test_solver_solve_with_save_variables(
    solver_mutable, system, driver_settings
):
    """Test Solver.solve accepts save_variables parameter."""
    state_names = list(system.initial_values.names)[:1]

    result = solver_mutable.solve(
        initial_values={state_names[0]: [1.0, 2.0]},
        parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
        drivers=driver_settings,
        save_variables=state_names,
        duration=0.1,
    )

    assert result is not None
    # Verify saved output contains requested states
    assert result.time_domain_array.shape[1] >= 1


def test_system_no_observables_default(precision, solver_settings):
    """Test default behavior with system having no observables.

    When a system has no observables, observable_indices should be
    empty arrays, not errors. The system declares state derivatives
    only, so ``observables`` is genuinely empty rather than merely
    suppressed by the requested output types.
    """
    NO_OBSERVABLE_EQUATIONS = [
        "dx0 = -x0 * p0",
        "dx1 = -x1 * p1",
        "dx2 = -x2 * p2",
    ]

    NO_OBSERVABLE_STATES = {"x0": 1.0, "x1": 1.0, "x2": 1.0}
    NO_OBSERVABLE_PARAMETERS = {"p0": 1.0, "p1": 2.0, "p2": 3.0}
    system = create_ODE_system(
        dxdt=NO_OBSERVABLE_EQUATIONS,
        states=NO_OBSERVABLE_STATES,
        parameters=NO_OBSERVABLE_PARAMETERS,
        precision=precision,
        name="three_state_no_observables",
        strict=False,
    )
    assert system.sizes.observables == 0
    solver = Solver(
        system,
        output_types=["state"],
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
    )

    # Observable indices should be empty when no observables exist
    assert len(solver.saved_observable_indices) == 0
    assert len(solver.summarised_observable_indices) == 0


# ============================================================================
# Cache Keyword Argument Tests
# ============================================================================


def test_solver_accepts_cache_mode_kwarg(system, solver_settings):
    """Verify Solver(system, cache_mode='flush_on_change') is recognized."""
    solver = Solver(
        system,
        algorithm=solver_settings["algorithm"],
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
        cache_mode="flush_on_change",
    )

    assert solver.kernel.cache_handler.policy.cache_mode == "flush_on_change"


def test_solver_accepts_max_cache_entries_kwarg(system, solver_settings):
    """Verify Solver(system, max_cache_entries=5) is recognized."""
    solver = Solver(
        system,
        algorithm=solver_settings["algorithm"],
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
        max_cache_entries=5,
    )

    assert solver.kernel.cache_handler.policy.max_cache_entries == 5


def test_solver_accepts_max_registers_kwarg(system, solver_settings):
    """Verify Solver(system, max_registers=128) is recognized."""
    solver = Solver(
        system,
        algorithm=solver_settings["algorithm"],
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
        max_registers=128,
    )

    assert solver.kernel.compile_settings.max_registers == 128


def test_solver_rejects_negative_spill_threshold(system, solver_settings):
    """Solver(host_spill_threshold=-1) raises at construction."""
    with pytest.raises(ValueError, match="host_spill_threshold"):
        Solver(
            system,
            algorithm=solver_settings["algorithm"],
            memory_manager=solver_settings["memory_manager"],
            stream_group=solver_settings["stream_group"],
            host_spill_threshold=-1,
        )


def test_solver_rejects_missing_spill_directory(
    system, solver_settings, tmp_path
):
    """Solver(spill_directory=<nonexistent>) raises at construction."""
    with pytest.raises(ValueError, match="existing directory"):
        Solver(
            system,
            algorithm=solver_settings["algorithm"],
            memory_manager=solver_settings["memory_manager"],
            stream_group=solver_settings["stream_group"],
            spill_directory=tmp_path / "missing",
        )


def test_solve_ivp_passes_cache_kwargs(
    system, simple_initial_values, simple_parameters, driver_settings
):
    """Verify solve_ivp(system, y0, params, cache_mode='hash') works.

    The entry limit sits far above any realistic entry count: a small
    limit against a shared kernel-cache directory (the CI artifact)
    would evict the shared system's precompiled kernels, and an
    isolated directory would guarantee a consumer-leg compile.
    """
    result = solve_ivp(
        system=system,
        y0=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        dt=1e-2,
        duration=0.05,
        method="euler",
        cache_mode="hash",
        max_cache_entries=100000,
    )

    assert isinstance(result, SolveResult)


# ============================================================================
# Unrecognized / renamed keyword argument tests
# ============================================================================


def test_solver_unknown_kwarg_raises(system, solver_settings):
    """An unconsumed keyword argument raises at construction."""
    with pytest.raises(KeyError, match="Unrecognized"):
        Solver(
            system,
            algorithm=solver_settings["algorithm"],
            memory_manager=solver_settings["memory_manager"],
            stream_group=solver_settings["stream_group"],
            not_a_real_setting=1.0,
        )


def test_solver_dt_save_raises_with_rename_hint(system, solver_settings):
    """The legacy dt_save spelling raises and names save_every."""
    with pytest.raises(KeyError, match="save_every"):
        Solver(
            system,
            algorithm=solver_settings["algorithm"],
            memory_manager=solver_settings["memory_manager"],
            stream_group=solver_settings["stream_group"],
            dt_save=1.0,
        )


def test_solve_dt_save_raises_with_rename_hint(
    solver_mutable, simple_initial_values, simple_parameters, driver_settings
):
    """The legacy dt_save spelling raises from solve-time kwargs."""
    with pytest.raises(KeyError, match="save_every"):
        solver_mutable.solve(
            initial_values=simple_initial_values,
            parameters=simple_parameters,
            drivers=driver_settings,
            duration=0.1,
            dt_save=0.05,
        )


def test_solve_unknown_kwarg_raises(
    solver_mutable, simple_initial_values, simple_parameters, driver_settings
):
    """An unconsumed solve-time keyword argument raises."""
    with pytest.raises(KeyError, match="Unrecognized"):
        solver_mutable.solve(
            initial_values=simple_initial_values,
            parameters=simple_parameters,
            drivers=driver_settings,
            duration=0.1,
            not_a_real_setting=1.0,
        )


# ============================================================================
# Final-save scheduling tests
# ============================================================================


@pytest.mark.parametrize(
    "solver_settings_override",
    # dt 0.0625 divides the 0.25 duration exactly in float32, which
    # is the boundary the save-last write must survive.
    [{**FIXED_EULER_TIMED_STATE, "save_every": None, "dt": 0.0625}],
    indirect=True,
)
def test_save_last_written_when_duration_is_step_multiple(
    solver, simple_initial_values, simple_parameters, driver_settings
):
    """save_last fires when the fixed step lands exactly on t_end."""
    duration = 0.25
    result = solver.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=duration,
        grid_type="verbatim",
        nan_error_trajectories=False,
    )
    times = np.asarray(result.time)
    if times.ndim > 1:
        times = times[:, 0]
    states = np.asarray(result.time_domain_array)
    assert times.shape[0] == 2
    assert times[0] == pytest.approx(0.0)
    assert times[1] == pytest.approx(duration, rel=1e-6)
    final_states = states[-1, : len(simple_initial_values), :]
    assert np.all(np.isfinite(final_states))
    assert np.any(final_states != 0.0)


@pytest.mark.parametrize(
    "solver_settings_override",
    # The three cadences divide the 0.2 duration exactly, inexactly
    # below, and inexactly above a row boundary in float32.
    [
        {**FIXED_EULER_TIMED_STATE, "save_every": 0.1, "dt": 0.01},
        {**FIXED_EULER_TIMED_STATE, "save_every": 0.04, "dt": 0.01},
        {**FIXED_EULER_TIMED_STATE, "save_every": 0.07, "dt": 0.01},
    ],
    indirect=True,
)
def test_regular_saves_fill_allocation_at_fp_endpoints(
    solver,
    solver_settings,
    simple_initial_values,
    simple_parameters,
    driver_settings,
):
    """Every allocated output row is written when save_every does not
    divide the duration exactly in the working precision."""
    duration = 0.2
    save_every = float(solver_settings["save_every"])
    result = solver.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=duration,
        grid_type="verbatim",
        nan_error_trajectories=False,
    )
    times = np.asarray(result.time)
    if times.ndim > 1:
        times = times[:, 0]
    precision = solver.precision
    expected_saves = np.floor(
        precision(duration) / precision(save_every)
    )
    assert np.all(np.diff(times) > 0.0)
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(
        expected_saves * save_every, rel=1e-5
    )
    states = np.asarray(result.time_domain_array)
    final_states = states[-1, : len(simple_initial_values), :]
    assert np.all(np.isfinite(final_states))
    assert np.any(final_states != 0.0)


@pytest.mark.nocudasim
def test_save_boundary_zero_gap_run_completes():
    """A driven stiff Rosenbrock run survives t rounding onto a save
    boundary.

    Float32 time accumulation can land the committed time exactly on
    ``next_save`` without the save firing, because the pre-step save
    prediction and the post-step time commit use different arithmetic.
    A step clamped to that boundary would have length zero, which the
    step function cannot integrate; the positive-gap-only clamp keeps
    dt_raw instead, so the run completes.

    This test keeps its own system because landing the committed
    time exactly on a save boundary needs this particular driven
    stiff system with these solver settings; the shared fixture
    systems do not reproduce the coincidence.
    """
    forced = create_ODE_system(
        "dx = v\ndv = mu * (1 - x*x) * v - x + forcing",
        parameters={"mu": 50.0},
        states={"x": 2.0, "v": 0.0},
        drivers=["forcing"],
        precision=np.float32,
        name="ForcedVanDerPol548",
    )
    time = np.linspace(0.0, 20.0, 400)
    signal = 5.0 * np.sin(2.0 * np.pi * 0.25 * time)

    result = solve_ivp(
        forced,
        y0={"x": np.array([2.0]), "v": np.array([0.0])},
        parameters={"mu": np.array([64.0])},
        drivers={"forcing": signal, "time": time},
        method="rosenbrock",
        duration=20.0,
        save_every=0.05,
    )

    assert result.status_messages == {}
    states = np.asarray(result.time_domain_array)
    assert np.all(np.isfinite(states))


# ============================================================================
# Additional coverage: precision passthrough, update() no-ops, memory
# settings, profiling toggles, and pass-through properties
# ============================================================================


def test_solve_ivp_raw_equations_precision_override():
    """solve_ivp forwards a precision override to the built system
    (_system_from_equations create_kwargs branch)."""
    def decay(t, y, k):
        return [-k * y[0]]

    result = solve_ivp(
        decay,
        y0={"x": [1.0]},
        parameters={"k": [0.5]},
        duration=0.02,
        dt=0.01,
        save_every=0.01,
        method="euler",
        precision=np.float64,
    )
    assert isinstance(result, SolveResult)
    assert result.solve_settings.precision == np.float64


def test_solve_ivp_forwards_summarise_variables(system):
    """solve_ivp threads summarise_variables through to Solver kwargs."""
    state_names = list(system.initial_values.names)[:1]
    result = solve_ivp(
        system,
        y0={state_names[0]: [1.0, 2.0]},
        parameters={list(system.parameters.names)[0]: [0.1, 0.2]},
        summarise_variables=state_names,
        save_every=0.01,
        summarise_every=0.02,
        duration=0.02,
        dt=0.01,
        method="euler",
        output_types=["state", "mean"],
    )
    assert isinstance(result, SolveResult)
    assert result.solve_settings.summarised_states == state_names


def test_solver_update_with_no_args_returns_empty_set(solver_mutable):
    """update() with no updates_dict and no kwargs is a no-op."""
    solver = solver_mutable
    assert solver.update() == set()
    assert solver.update(None) == set()


def test_solver_update_memory_settings_no_args_returns_empty_set(
    solver_mutable,
):
    """update_memory_settings() with no updates_dict and no kwargs is a
    no-op."""
    solver = solver_mutable
    assert solver.update_memory_settings() == set()
    assert solver.update_memory_settings(None) == set()


def test_solver_update_memory_settings_accepts_kwargs_form(solver_mutable):
    """update_memory_settings merges **kwargs into updates_dict."""
    solver = solver_mutable
    updated = solver.update_memory_settings(mem_proportion=0.15)
    assert "mem_proportion" in updated


def test_solver_update_memory_settings_none_proportion_sets_auto(
    solver_mutable,
):
    """mem_proportion=None switches the memory manager to auto-limit
    mode instead of raising or being ignored."""
    solver = solver_mutable
    updated = solver.update_memory_settings({"mem_proportion": None})
    assert "mem_proportion" in updated
    # Auto-limit mode still reports a usable (non-None) proportion.
    assert solver.mem_proportion is not None


def test_solver_update_memory_settings_unrecognized_raises(solver_mutable):
    """An unrecognized memory setting raises KeyError unless silent."""
    solver = solver_mutable
    with pytest.raises(KeyError, match="Unrecognized"):
        solver.update_memory_settings({"not_a_real_memory_setting": 1})


def test_solver_compile_flags_property(solver):
    """compile_flags passes through to the kernel's compile flags."""
    assert solver.compile_flags is solver.kernel.compile_flags


def test_solver_status_messages_property_before_solve(solver):
    """status_messages decodes the kernel's current status codes."""
    messages = solver.status_messages
    assert isinstance(messages, dict)


def test_solver_parameters_initial_values_driver_coefficients_properties(
    solved_solver_simple,
):
    """parameters, initial_values, and driver_coefficients pass through
    to the kernel after a solve."""
    solver, _ = solved_solver_simple
    assert solver.parameters is solver.kernel.parameters
    assert solver.initial_values is solver.kernel.initial_values
    assert solver.driver_coefficients is solver.kernel.driver_coefficients


def test_solver_stream_property(solver):
    """stream passes through to the kernel's stream."""
    assert solver.stream == solver.kernel.stream


def test_solver_set_verbosity(solver_mutable):
    """set_verbosity updates the global time logger verbosity."""
    from cubie.time_logger import default_timelogger

    solver = solver_mutable
    solver.set_verbosity("verbose")
    assert default_timelogger.verbosity == "verbose"
    solver.set_verbosity(None)
    assert default_timelogger.verbosity is None


"""Every movable loop and DIRK work-buffer location setting.

Pinning all of them makes both solvers fully explicit: every auto
placement group contains a user-set key, so the heuristics are
blocked on both sides and each solver's layout is exactly what the
test states.
"""


@pytest.mark.parametrize(
    "solver_settings_override",
    [{
        "system_type": "medium",
        "algorithm": "dirk",
        "duration": 0.02,
        "output_types": ["state"],
        "saved_observable_indices": [],
        "summarised_observable_indices": [],
        **{key: "local" for key in MOVABLE_LOCATION_KEYS},
    }],
    indirect=True,
)
def test_shared_loop_buffers_leave_results_unchanged(
    solver,
    solver_settings,
    system,
    driver_settings,
    thread_mem_manager,
    simple_initial_values,
    simple_parameters,
):
    """Buffer placement is storage-only: moving every movable
    buffer to shared memory reproduces the all-local trajectories.

    Both solvers pin every movable location explicitly - all local
    on the reference, all shared on the comparison - so the auto
    placement heuristics are blocked on both sides and the pair
    differs only in buffer placement. On the 20-state system the
    shared placements shrink the loop's plain-local pool well below
    the DIRK step's persistent requirement, so this fails loudly if
    the persistent scratch array is ever again sized from the
    plain-local total instead of the persistent layout. The
    all-local reference is the shared ``solver`` fixture; only the
    relocated comparison solver is built fresh, because buffer
    placement is a construction setting.
    """
    shared_locations = {
        key: "shared" for key in MOVABLE_LOCATION_KEYS
    }

    def run_solve(active_solver):
        result = active_solver.solve(
            initial_values=simple_initial_values,
            parameters=simple_parameters,
            drivers=driver_settings,
            duration=solver_settings["duration"],
        )
        return np.asarray(result.time_domain_array)

    local_output = run_solve(solver)

    shared_settings = dict(solver_settings)
    shared_settings.update(shared_locations)
    shared_settings["stream_group"] = "shared_loop_buffers"
    shared_solver = _build_solver_instance(
        system=system,
        solver_settings=shared_settings,
        driver_settings=driver_settings,
        memory_manager=thread_mem_manager,
    )
    try:
        shared_output = run_solve(shared_solver)
    finally:
        shared_solver.close()

    assert np.all(np.isfinite(local_output))
    np.testing.assert_array_equal(shared_output, local_output)


def test_driver_setting_update_syncs_evaluator_and_coefficients(
    solver_mutable,
    system,
    solver_settings,
    driver_settings,
    thread_mem_manager,
    simple_initial_values,
    simple_parameters,
):
    """Settings-only driver updates flow through ``Solver.update``.

    Switching ``boundary_condition`` from "clamped" to "natural"
    changes the coefficient tensor's segment count, so the updated
    solver only reproduces a natural-from-scratch solver when the
    evaluator and coefficients are refreshed together.
    """

    def run_solve(active_solver, drivers):
        result = active_solver.solve(
            initial_values=simple_initial_values,
            parameters=simple_parameters,
            drivers=drivers,
            duration=solver_settings["duration"],
        )
        return np.asarray(result.time_domain_array)

    run_solve(solver_mutable, driver_settings)
    solver_mutable.update({"boundary_condition": "natural"})
    updated_output = run_solve(solver_mutable, None)

    natural_settings = dict(driver_settings)
    natural_settings["boundary_condition"] = "natural"
    reference_solver = _build_solver_instance(
        system=system,
        solver_settings=solver_settings,
        driver_settings=natural_settings,
        memory_manager=thread_mem_manager,
    )
    reference_output = run_solve(reference_solver, None)

    assert np.all(np.isfinite(updated_output))
    np.testing.assert_array_equal(updated_output, reference_output)


SIBLING_STREAM_GROUP = "sibling_allocation"


def _build_sibling_solver(system, solver_settings, driver_settings, manager):
    return _build_solver_instance(
        system=system,
        solver_settings={**solver_settings, "stream_group": SIBLING_STREAM_GROUP},
        driver_settings=driver_settings,
        memory_manager=manager,
    )


def test_repeat_solve_with_live_result_after_sibling_construction(
    system,
    solver_settings,
    driver_settings,
    batch_input_arrays,
    thread_mem_manager,
):
    """A held result plus newly built siblings leaves the batch intact."""
    y0, params = batch_input_arrays
    n_runs = y0.shape[1]
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)
    base = _build_sibling_solver(
        system, solver_settings, driver_settings, thread_mem_manager
    )
    siblings = []
    try:
        base.solve(y0, params, **solve_kwargs)
        siblings = [
            _build_sibling_solver(
                system, solver_settings, driver_settings, thread_mem_manager
            )
            for _ in range(2)
        ]
        held = base.solve(y0, params, **solve_kwargs)
        for sibling in siblings:
            sibling.solve(y0, params, **solve_kwargs)

        repeat = base.solve(y0, params, **solve_kwargs)

        outputs = base.kernel.output_arrays
        assert outputs.device_state.shape[2] == n_runs
        assert outputs.device_status_codes.shape[0] == n_runs
        assert base.kernel.chunks == 1
        np.testing.assert_array_equal(repeat.state, held.state)
        np.testing.assert_array_equal(
            repeat.status_codes, held.status_codes
        )
    finally:
        base.close()
        for sibling in siblings:
            sibling.close()


def _unroll_factories(solver):
    """Return every factory of an implicit-step ``solver``."""
    kernel = solver.kernel
    integrator = kernel.single_integrator
    algo_step = integrator._algo_step
    return (
        kernel,
        kernel.driver_interpolator,
        integrator._loop,
        integrator._output_functions,
        integrator._dae_initialiser,
        algo_step,
        algo_step.solver,
        algo_step.solver.norm,
        algo_step.linear_solver,
        algo_step.linear_solver.norm,
    )


@pytest.mark.parametrize(
    "solver_settings_override", [UNROLL_SETTINGS], indirect=True
)
def test_unroll_settings_reach_every_factory(solver, solver_settings):
    """Loose ``unroll_*`` settings land on every child factory."""
    expected = UnrollFlags(
        **{key: solver_settings[key] for key in ALL_UNROLL_PARAMETERS}
    )
    for factory in _unroll_factories(solver):
        assert factory.compile_settings.unroll == expected
    algo_step = solver.kernel.single_integrator._algo_step
    request_kwargs = algo_step._helper_request_kwargs()
    assert request_kwargs["unroll_solver_element"] == (True, 2)
    assert request_kwargs["unroll_other_small"] == (True, None)


@pytest.mark.parametrize(
    "solver_settings_override", [UNROLL_SETTINGS], indirect=True
)
def test_unroll_settings_solve(
    solver, simple_initial_values, simple_parameters, driver_settings
):
    """A solve with every loop group hinted completes."""
    result = solver.solve(
        initial_values=simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.1,
        settling_time=0.0,
        blocksize=32,
        grid_type="combinatorial",
    )
    assert np.all(np.isfinite(result.state))


@pytest.mark.parametrize(
    "solver_settings_override", [UNROLL_SETTINGS], indirect=True
)
def test_unroll_object_with_loose_keys(solver_mutable):
    """Loose keys override the fields of a supplied ``UnrollFlags``."""
    solver = solver_mutable
    updated_keys = solver.update(
        {
            "unroll": UnrollFlags(unroll_accumulator=True),
            "unroll_stage": (True, 2),
        }
    )
    assert {"unroll", "unroll_stage"} <= updated_keys
    expected = UnrollFlags(unroll_accumulator=True, unroll_stage=(True, 2))
    for factory in _unroll_factories(solver):
        assert factory.compile_settings.unroll == expected


def test_update_unroll_loose_key(solver_mutable):
    """A loose ``unroll_*`` update reaches the kernel and its children."""
    solver = solver_mutable
    algo_step = solver.kernel.single_integrator._algo_step

    updated_keys = solver.update({"unroll_norms": True})
    assert "unroll_norms" in updated_keys
    assert solver.kernel.compile_settings.unroll.unroll_norms == (True, None)
    assert algo_step.compile_settings.unroll.unroll_norms == (True, None)
    assert not solver.kernel.cache_valid

    updated_keys = solver.update({"unroll_norms": (True, 3)})
    assert "unroll_norms" in updated_keys
    assert solver.kernel.compile_settings.unroll.unroll_norms == (True, 3)
    assert algo_step.compile_settings.unroll.unroll_norms == (True, 3)

    updated_keys = solver.update({"unroll_norms": False})
    assert "unroll_norms" in updated_keys
    assert solver.kernel.compile_settings.unroll.unroll_norms == (
        False,
        None,
    )
