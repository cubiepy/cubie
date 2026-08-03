"""CUDA kernel numerical tests for output functions.

Tests save_state, update_summaries, and save_summary_metrics device
functions by running a test kernel that processes deterministic input
arrays and comparing results against CPU-computed expected values.

Recovered from git history (pre-Phase 1 refactor).
"""

import numpy as np
import pytest

from tests._utils import FLOAT64_PRECISION
from numpy.testing import assert_allclose
from cubie.cuda_simsafe import cuda, numba_from_dtype as from_dtype
from cubie.memory import default_memmgr

from cubie.outputhandling import OutputFunctions
from tests._utils import deterministic_array, calculate_expected_summaries


@pytest.fixture(scope="session")
def output_test_settings(output_test_settings_overrides, precision):
    """Parameters for instantiating and testing OutputFunctions."""
    output_test_settings_dict = {
        "num_samples": 10,
        "num_summaries": 1,
        "num_states": 10,
        "num_observables": 10,
        "saved_state_indices": [0, 1],
        "saved_observable_indices": [0, 1],
        "output_types": ["state"],
        "precision": precision,
        "test_shared_mem": True,
    }
    output_test_settings_dict.update(**output_test_settings_overrides)
    return output_test_settings_dict


@pytest.fixture(scope="session")
def output_test_settings_overrides(request):
    """Parametrize this fixture indirectly to change test settings."""
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="session")
def output_functions(output_test_settings, precision):
    """OutputFunctions object under test."""
    return OutputFunctions(
        output_test_settings["num_states"],
        output_test_settings["num_observables"],
        precision,
        output_test_settings["output_types"],
        output_test_settings["saved_state_indices"],
        output_test_settings["saved_observable_indices"],
    )


@pytest.fixture(scope="function")
def random_scale(request):
    """Magnitude guidance for the deterministic input arrays.

    Parametrise indirectly to sweep input value ranges; the scale
    reaches only the host input arrays, so a sweep costs no compile.
    """
    return request.param if hasattr(request, "param") else 1.0


@pytest.fixture(scope="function")
def input_arrays(output_test_settings, random_scale):
    """Deterministic input state and observable arrays for tests."""
    num_states = output_test_settings["num_states"]
    num_observables = output_test_settings["num_observables"]
    num_samples = output_test_settings["num_samples"]
    precision = output_test_settings["precision"]

    states = deterministic_array(
        precision, (num_samples, num_states), random_scale
    )
    observables = deterministic_array(
        precision, (num_samples, num_observables), random_scale
    )

    return states, observables


@pytest.fixture(scope="session")
def empty_output_arrays(output_test_settings, output_functions):
    """Empty output arrays for testing."""

    n_saved_states = output_functions.n_saved_states
    n_saved_observables = output_functions.n_saved_observables
    n_summarised_states = len(
        output_functions.summarised_state_indices
    )
    n_summarised_observables = len(
        output_functions.summarised_observable_indices
    )
    num_samples = output_test_settings["num_samples"]
    num_summaries = output_test_settings["num_summaries"]
    summary_height_per_variable = (
        output_functions.summaries_output_height_per_var
    )
    state_summary_height = (
        summary_height_per_variable * n_summarised_states
    )
    observable_summary_height = (
        summary_height_per_variable * n_summarised_observables
    )

    precision = output_test_settings["precision"]
    save_time = "time" in output_test_settings["output_types"]

    if save_time:
        n_saved_states += 1

    state_out = np.zeros(
        (num_samples, n_saved_states), dtype=precision
    )
    observable_out = np.zeros(
        (num_samples, n_saved_observables), dtype=precision
    )
    state_summary = np.zeros(
        (num_summaries, state_summary_height), dtype=precision
    )
    observable_summary = np.zeros(
        (num_summaries, observable_summary_height), dtype=precision
    )
    counters_out = np.zeros((num_samples, 4), dtype=np.int32)

    return (
        state_out,
        observable_out,
        state_summary,
        observable_summary,
        counters_out,
    )


@pytest.fixture(scope="function")
def expected_outputs(output_test_settings, input_arrays, precision):
    """Selected portions of input arrays matching test kernel."""
    num_samples = output_test_settings["num_samples"]
    state_in, observables_in = input_arrays
    state_out = state_in[
        :, output_test_settings["saved_state_indices"]
    ]
    observable_out = observables_in[
        :, output_test_settings["saved_observable_indices"]
    ]
    save_time = "time" in output_test_settings["output_types"]

    if save_time:
        time_output = np.arange(num_samples, dtype=precision)
        time_output = time_output.reshape((num_samples, 1))
        state_out = np.concatenate(
            (state_out, time_output), axis=1
        )

    return state_out, observable_out


@pytest.fixture(scope="function")
def expected_summaries(
    output_test_settings,
    empty_output_arrays,
    expected_outputs,
    output_functions,
    precision,
):
    """Default expected summaries for the output functions."""
    state_output, observables_output = expected_outputs
    if output_functions.save_time:
        state_output = state_output[:, :-1]
    summarise_every = (
        output_test_settings["num_samples"]
        // output_test_settings["num_summaries"]
    )
    output_types = output_test_settings["output_types"]
    summary_height_per_variable = (
        output_functions.summaries_output_height_per_var
    )

    state_summaries, observable_summaries = (
        calculate_expected_summaries(
            state_output,
            observables_output,
            output_test_settings['saved_state_indices'],
            output_test_settings['saved_observable_indices'],
            summarise_every,
            output_types,
            summary_height_per_variable,
            precision,
        )
    )

    return state_summaries, observable_summaries


@pytest.fixture(scope="session")
def output_functions_test_kernel(
    precision, output_test_settings, output_functions
):
    """Kernel that writes input to local state, then uses output
    functions to save every sample and summarise periodically."""
    summarise_every = (
        output_test_settings["num_samples"]
        // output_test_settings["num_summaries"]
    )
    test_shared_mem = output_test_settings["test_shared_mem"]

    save_state_func = output_functions.save_state_func
    update_summary_metrics_func = (
        output_functions.update_summaries_func
    )
    save_summary_metrics_func = (
        output_functions.save_summary_metrics_func
    )

    num_states = output_test_settings["num_states"]
    num_observables = output_test_settings["num_observables"]

    n_summarised_states = len(
        output_functions.summarised_state_indices
    )
    n_summarised_observables = len(
        output_functions.summarised_observable_indices
    )

    shared_memory_requirements = (
        output_functions.summaries_buffer_height_per_var
    )
    state_summary_buffer_length = (
        shared_memory_requirements * n_summarised_states
    )
    obs_summary_buffer_length = (
        shared_memory_requirements * n_summarised_observables
    )

    state_summary_output_height = (
        max(output_functions.state_summaries_output_height, 1)
    )
    obs_summary_output_height = (
        max(output_functions.observable_summaries_output_height, 1)
    )

    if test_shared_mem is False:
        num_states = 1 if num_states == 0 else num_states
        num_observables = (
            1 if num_observables == 0 else num_observables
        )
        state_summary_buffer_length = (
            1
            if state_summary_buffer_length == 0
            else state_summary_buffer_length
        )
        obs_summary_buffer_length = (
            1
            if obs_summary_buffer_length == 0
            else obs_summary_buffer_length
        )

    numba_precision = from_dtype(precision)

    @cuda.jit()
    def _output_functions_test_kernel(
        _state_input,
        _observable_input,
        _state_output,
        _observable_output,
        _state_summaries_output,
        _observable_summaries_output,
        _counters_output,
    ):
        """Test kernel for output functions."""

        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x

        if tx != 0 or bx != 0:
            return

        if test_shared_mem:
            shared = cuda.shared.array(0, dtype=numba_precision)

            observables_start_idx = num_states
            state_summaries_start_idx = (
                observables_start_idx + num_observables
            )
            obs_summaries_start_idx = (
                state_summaries_start_idx
                + state_summary_buffer_length
            )
            obs_summaries_end_idx = (
                obs_summaries_start_idx
                + obs_summary_buffer_length
            )

            current_state = shared[:observables_start_idx]
            current_observable = shared[
                observables_start_idx:state_summaries_start_idx
            ]
            state_summaries = shared[
                state_summaries_start_idx:obs_summaries_start_idx
            ]
            observable_summaries = shared[
                obs_summaries_start_idx:obs_summaries_end_idx
            ]

        else:
            current_state = cuda.local.array(
                num_states, dtype=numba_precision
            )
            current_observable = cuda.local.array(
                num_observables, dtype=numba_precision
            )
            state_summaries = cuda.local.array(
                state_summary_buffer_length, dtype=numba_precision
            )
            observable_summaries = cuda.local.array(
                obs_summary_buffer_length, dtype=numba_precision
            )

        current_state[:] = 0.0
        current_observable[:] = 0.0
        state_summaries[:] = 0.0
        observable_summaries[:] = 0.0

        counters = cuda.local.array(4, dtype=np.int32)
        counters[:] = 0

        # Initialize summary buffers with sentinel values
        dummy_state_summary_out = cuda.local.array(
            state_summary_output_height, dtype=numba_precision
        )
        dummy_obs_summary_out = cuda.local.array(
            obs_summary_output_height, dtype=numba_precision
        )
        save_summary_metrics_func(
            state_summaries,
            observable_summaries,
            dummy_state_summary_out,
            dummy_obs_summary_out,
            summarise_every,
        )

        for i in range(_state_input.shape[0]):
            for j in range(num_states):
                current_state[j] = _state_input[i, j]
            for j in range(num_observables):
                current_observable[j] = _observable_input[i, j]

            save_state_func(
                current_state,
                current_observable,
                counters,
                i,
                _state_output[i, :],
                _observable_output[i, :],
                _counters_output[i, :],
            )

            update_summary_metrics_func(
                current_state,
                current_observable,
                state_summaries,
                observable_summaries,
                i,
            )

            if (i + 1) % summarise_every == 0:
                sample_index = int(i / summarise_every)
                save_summary_metrics_func(
                    state_summaries,
                    observable_summaries,
                    _state_summaries_output[sample_index, :],
                    _observable_summaries_output[sample_index, :],
                    summarise_every,
                )

    return _output_functions_test_kernel


@pytest.fixture(scope="function")
def compare_input_output(
    output_functions_test_kernel,
    output_functions,
    output_test_settings,
    input_arrays,
    empty_output_arrays,
    expected_outputs,
    expected_summaries,
    precision,
    tolerance,
):
    """Run kernel and compare against expected values."""

    state_input, observable_input = input_arrays
    (
        state_output,
        observable_output,
        state_summaries_output,
        observable_summaries_output,
        counters_output,
    ) = empty_output_arrays

    n_summarised_states = len(
        output_functions.summarised_state_indices
    )
    n_summarised_observables = len(
        output_functions.summarised_observable_indices
    )

    n_states = output_test_settings["num_states"]
    n_observables = output_test_settings["num_observables"]

    d_state_input = cuda.to_device(state_input)
    d_observable_input = cuda.to_device(observable_input)
    d_state_output = cuda.to_device(state_output)
    d_observable_output = cuda.to_device(observable_output)
    d_state_summaries_output = cuda.to_device(
        state_summaries_output
    )
    d_observable_summaries_output = cuda.to_device(
        observable_summaries_output
    )
    d_counters_output = cuda.to_device(counters_output)

    kernel_shared_memory = n_states + n_observables
    summary_buffer_size = (
        output_functions.summaries_buffer_height_per_var
    )
    summary_shared_memory = (
        n_summarised_states + n_summarised_observables
    ) * summary_buffer_size
    dynamic_shared_memory = (
        kernel_shared_memory + summary_shared_memory
    ) * precision().itemsize

    stream = default_memmgr.get_group_stream()
    output_functions_test_kernel[1, 1, stream, dynamic_shared_memory](
        d_state_input,
        d_observable_input,
        d_state_output,
        d_observable_output,
        d_state_summaries_output,
        d_observable_summaries_output,
        d_counters_output,
    )

    stream.synchronize()

    state_output = d_state_output.copy_to_host()
    observable_output = d_observable_output.copy_to_host()
    state_summaries_output = d_state_summaries_output.copy_to_host()
    observable_summaries_output = (
        d_observable_summaries_output.copy_to_host()
    )

    expected_state_output, expected_observable_output = (
        expected_outputs
    )
    expected_state_summaries, expected_observable_summaries = (
        expected_summaries
    )

    if output_functions.compile_settings.save_state:
        assert_allclose(
            state_output,
            expected_state_output,
            atol=tolerance.abs_tight,
            rtol=tolerance.rel_tight,
            err_msg="State &| time values were not saved correctly",
        )

    if output_functions.compile_settings.save_observables:
        assert_allclose(
            observable_output,
            expected_observable_output,
            atol=tolerance.abs_tight,
            rtol=tolerance.rel_tight,
            err_msg="Observable values were not saved correctly",
        )

    if output_functions.compile_settings.summarise_state:
        assert_allclose(
            expected_state_summaries,
            state_summaries_output,
            atol=1e-4,
            rtol=1e-4,
            err_msg=(
                "State summaries didn't match expected values. "
                f"Shapes: expected[{expected_state_summaries.shape}"
                f", actual[{state_summaries_output.shape}]"
            ),
            verbose=True,
        )
    if output_functions.compile_settings.summarise_observables:
        assert_allclose(
            expected_observable_summaries,
            observable_summaries_output,
            atol=1e-4,
            rtol=1e-4,
            err_msg=(
                "Observable summaries didn't match expected values."
                f" Shapes: expected"
                f"[{expected_observable_summaries.shape}"
                f", actual[{observable_summaries_output.shape}]"
            ),
            verbose=True,
        )


ALL_SUMMARY_TYPES = [
    "state",
    "observables",
    "mean",
    "std",
    "rms",
    "max",
    "min",
    "max_magnitude",
    "negative_peaks[3]",
    "peaks[3]",
]

LONG_RUN_OUTPUT_SETTINGS = {
    "output_types": ALL_SUMMARY_TYPES,
    "num_samples": 1000,
    "num_summaries": 100,
}

LONG_WINDOW_OUTPUT_SETTINGS = {
    "output_types": ALL_SUMMARY_TYPES,
    "num_samples": 500,
    "num_summaries": 1,
}


@pytest.mark.parametrize("random_scale", [1e1], indirect=True)
@pytest.mark.parametrize(
    "output_test_settings_overrides",
    [LONG_RUN_OUTPUT_SETTINGS],
    ids=["large_dataset"],
    indirect=True,
)
def test_all_summaries_long_run(compare_input_output):
    """Test a long run with frequent summaries."""


@pytest.mark.parametrize(
    "solver_settings_override",
    [FLOAT64_PRECISION],
    ids=["float64"],
    indirect=True,
)
@pytest.mark.parametrize("random_scale", [1e1], indirect=True)
@pytest.mark.parametrize(
    "output_test_settings_overrides",
    [LONG_RUN_OUTPUT_SETTINGS],
    ids=["large_dataset"],
    indirect=True,
)
def test_all_summaries_long_run_float64(compare_input_output):
    """Test a long run with frequent summaries in float64."""


@pytest.mark.parametrize("random_scale", [1e1], indirect=True)
@pytest.mark.parametrize(
    "output_test_settings_overrides",
    [LONG_WINDOW_OUTPUT_SETTINGS],
    ids=["large_dataset"],
    indirect=True,
)
def test_all_summaries_long_window(compare_input_output):
    """Test a long summary window (500 samples)."""


@pytest.mark.parametrize(
    "solver_settings_override",
    [FLOAT64_PRECISION],
    ids=["float64"],
    indirect=True,
)
@pytest.mark.parametrize("random_scale", [1e1], indirect=True)
@pytest.mark.parametrize(
    "output_test_settings_overrides",
    [LONG_WINDOW_OUTPUT_SETTINGS],
    ids=["large_dataset"],
    indirect=True,
)
def test_all_summaries_long_window_float64(compare_input_output):
    """Test a long summary window (500 samples) in float64."""


@pytest.mark.parametrize(
    "output_test_settings_overrides",
    [
        {
            "output_types": [
                "state",
                "observables",
                "mean",
                "max",
                "peaks[3]",
            ],
            "test_shared_mem": False,
        },
        {
            "output_types": [
                "state",
                "observables",
                "mean",
                "max",
                "peaks[3]",
            ],
            "test_shared_mem": True,
        },
    ],
    ids=["local_mem", "shared_mem"],
    indirect=True,
)
def test_memory_types(compare_input_output):
    """Test shared vs local memory with complex output configs."""
    pass


@pytest.mark.parametrize(
    "random_scale",
    [1e-6, 1e6, [-12, 12]],
    ids=["tiny_values", "large_values", "wide_range"],
    indirect=True,
)
def test_input_value_ranges(compare_input_output):
    """Test different input scales and simulation lengths."""


@pytest.mark.parametrize(
    "output_test_settings_overrides",
    # state-with-observables saving is covered by the all-on case, so
    # the remaining cases each toggle one save flag off.
    [
        {"output_types": ["observables"]},
        {"output_types": ["observables", "time"]},
        {"output_types": ["time"]},
        {"output_types": ["state", "observables", "time"]},
    ],
    ids=[
        "obs_only",
        "obs_time",
        "time_only",
        "state_obs_time",
    ],
    indirect=True,
)
def test_no_summarys(compare_input_output):
    """Test basic state and observable output configurations."""
    pass


@pytest.mark.parametrize(
    "output_test_settings_overrides",
    [
        {"output_types": ["mean", "max", "rms"]},
        {"output_types": ["peaks[10]"]},
        {
            "output_types": [
                "state",
                "observables",
                "time",
                "mean",
                "std",
                "rms",
                "max",
                "min",
                "max_magnitude",
                "negative_peaks[1]",
                "peaks[1]",
            ]
        },
        # A single metric beside a single save type; the observable
        # twin of this case is covered by the all-on case above.
        {"output_types": ["state", "mean"]},
    ],
    ids=[
        "basic_summaries",
        "peaks_only",
        "all",
        "state_and_mean",
    ],
    indirect=True,
)
def test_various_summaries(compare_input_output):
    """Test various summary metrics configurations."""
    pass


@pytest.mark.parametrize(
    "output_test_settings_overrides",
    [
        {
            "saved_state_indices": [0],
            "saved_observable_indices": [0],
        },
        {
            "num_states": 11,
            "num_observables": 6,
            "saved_state_indices": list(range(10)),
            "saved_observable_indices": list(range(5)),
        },
        {
            "num_states": 101,
            "num_observables": 100,
            "saved_state_indices": list(range(100)),
            "saved_observable_indices": list(range(100)),
        },
    ],
    ids=["1_1", "10_5", "100_100"],
    indirect=True,
)
def test_big_and_small_systems(compare_input_output):
    """Test various small to medium system sizes."""
    pass


@pytest.mark.parametrize(
    "output_test_settings_overrides",
    [
        {
            "num_summaries": 5,
            "output_types": [
                "state",
                "observables",
                "mean",
                "max",
                "rms",
            ],
        }
    ],
    indirect=True,
)
def test_frequent_summaries(compare_input_output):
    """Summary metrics stay correct at a shorter summary cadence."""
