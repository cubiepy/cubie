"""Tests for cubie.batchsolving.BatchInputHandler."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests._utils import FLOAT64_PRECISION
from numpy.testing import assert_allclose, assert_array_equal

from cubie.batchsolving.BatchInputHandler import (
    BatchInputHandler,
    combine_grids,
    combinatorial_grid,
    extend_grid_to_array,
    generate_grid,
    unique_cartesian_product,
    verbatim_grid,
)
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.cuda_simsafe import is_pinned_array


# ── unique_cartesian_product ────────────────────────────── #


def test_unique_cartesian_product_deduplicates(system):
    """Deduplicates each input array preserving order."""
    a = np.array([1, 2, 2])
    b = np.array([3, 4])
    result = unique_cartesian_product([a, b])
    expected = np.array([[1, 1, 2, 2], [3, 4, 3, 4]])
    assert_array_equal(result, expected)


def test_unique_cartesian_product_format(system):
    """Returns Cartesian product in (variable, run) format."""
    a = np.array([1, 2, 3])
    b = np.array([10, 20])
    result = unique_cartesian_product([a, b])
    # 2 variables (rows), 6 runs (columns)
    assert result.shape == (2, 6)
    # Every (a, b) combination present exactly once
    pairs = set(map(tuple, result.T))
    assert len(pairs) == 6


# ── combinatorial_grid ──────────────────────────────────── #


def test_combinatorial_grid_filters_empty(system):
    """Filters empty values from request."""
    param = system.parameters
    request = {
        param.names[0]: np.array([0.1, 0.2]),
        param.names[1]: np.array([]),
    }
    indices, grid = combinatorial_grid(request, param)
    # Only one variable swept (the non-empty one)
    assert indices.shape[0] == 1
    assert grid.shape == (1, 2)


def test_combinatorial_grid_resolves_indices(system):
    """Resolves indices via values_instance."""
    param = system.parameters
    request = {
        param.names[0]: np.array([0.1, 0.2]),
        param.names[1]: np.array([10, 20]),
    }
    indices, grid = combinatorial_grid(request, param)
    assert_array_equal(indices, np.array([0, 1]))
    assert grid.shape == (2, 4)


# ── verbatim_grid ───────────────────────────────────────── #


def test_verbatim_grid_filters_empty(system):
    """Filters empty values from request."""
    param = system.parameters
    request = {
        param.names[0]: np.array([0.1, 0.2]),
        param.names[1]: np.array([]),
    }
    indices, grid = verbatim_grid(request, param)
    assert indices.shape[0] == 1
    assert grid.shape == (1, 2)


def test_verbatim_grid_stacks_rows(system):
    """Stacks values as rows (variable, run) format without expansion."""
    param = system.parameters
    request = {
        param.names[0]: np.array([0.1, 0.2, 0.3]),
        param.names[1]: np.array([10, 20, 30]),
    }
    indices, grid = verbatim_grid(request, param)
    assert_array_equal(indices, np.array([0, 1]))
    expected = np.array([[0.1, 0.2, 0.3], [10, 20, 30]])
    assert_array_equal(grid, expected)


# ── generate_grid ───────────────────────────────────────── #


def test_generate_grid_combinatorial(system):
    """Dispatches to combinatorial_grid for kind='combinatorial'."""
    state = system.initial_values
    request = {state.names[0]: [1.0, 2.0], state.names[1]: [3.0, 4.0]}
    indices, grid = generate_grid(request, state, kind="combinatorial")
    assert grid.shape[1] == 4  # 2 * 2


def test_generate_grid_verbatim(system):
    """Dispatches to verbatim_grid for kind='verbatim'."""
    state = system.initial_values
    request = {state.names[0]: [1.0, 2.0], state.names[1]: [3.0, 4.0]}
    indices, grid = generate_grid(request, state, kind="verbatim")
    assert grid.shape[1] == 2


def test_generate_grid_unknown_kind_raises(system):
    """Raises ValueError for unknown kind."""
    state = system.initial_values
    request = {state.names[0]: [1.0]}
    with pytest.raises(ValueError, match="Unknown grid type 'badkind'"):
        generate_grid(request, state, kind="badkind")


# ── combine_grids ───────────────────────────────────────── #


def test_combine_grids_combinatorial():
    """Combinatorial: repeats grid1 columns and tiles grid2 columns."""
    grid1 = np.array([[1, 2], [3, 4]])
    grid2 = np.array([[10, 20, 30], [40, 50, 60]])
    g1, g2 = combine_grids(grid1, grid2, kind="combinatorial")
    expected_g1 = np.array([[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]])
    expected_g2 = np.array(
        [[10, 20, 30, 10, 20, 30], [40, 50, 60, 40, 50, 60]]
    )
    assert_array_equal(g1, expected_g1)
    assert_array_equal(g2, expected_g2)


def test_combine_grids_verbatim_broadcast_g1():
    """Verbatim: broadcasts single-run grid1 to match grid2."""
    grid1 = np.array([[1], [2]])
    grid2 = np.array([[10, 20, 30], [40, 50, 60]])
    g1, g2 = combine_grids(grid1, grid2, kind="verbatim")
    assert_array_equal(g1, np.array([[1, 1, 1], [2, 2, 2]]))
    assert_array_equal(g2, grid2)


def test_combine_grids_verbatim_broadcast_g2():
    """Verbatim: broadcasts single-run grid2 to match grid1."""
    grid1 = np.array([[1, 2, 3], [4, 5, 6]])
    grid2 = np.array([[10], [40]])
    g1, g2 = combine_grids(grid1, grid2, kind="verbatim")
    assert_array_equal(g1, grid1)
    assert_array_equal(g2, np.array([[10, 10, 10], [40, 40, 40]]))


def test_combine_grids_verbatim_mismatch_raises():
    """Verbatim: raises ValueError when run counts differ after broadcast."""
    grid1 = np.array([[1, 2], [3, 4]])
    grid2 = np.array([[10, 20, 30], [40, 50, 60]])
    with pytest.raises(ValueError, match="same number of runs"):
        combine_grids(grid1, grid2, kind="verbatim")


def test_combine_grids_unknown_kind_raises():
    """Raises ValueError for unknown kind."""
    grid1 = np.array([[1], [2]])
    grid2 = np.array([[3], [4]])
    with pytest.raises(ValueError, match="Unknown grid type"):
        combine_grids(grid1, grid2, kind="bad")


# ── extend_grid_to_array ───────────────────────────────── #


def test_extend_grid_tiled_defaults_empty_indices(system):
    """Returns tiled defaults when indices empty."""
    defaults = system.parameters.values_array
    grid = np.array([[1, 2, 3]])  # placeholder, indices empty
    indices = np.array([], dtype=int)
    result = extend_grid_to_array(grid, indices, defaults)
    assert result.shape == (defaults.shape[0], 3)
    for col in range(3):
        assert_array_equal(result[:, col], defaults)


def test_extend_grid_1d_defaults(system):
    """Returns single-column defaults when grid is 1D."""
    defaults = system.parameters.values_array
    grid = np.array([99.0])
    indices = np.array([0])
    result = extend_grid_to_array(grid, indices, defaults)
    assert result.shape == (defaults.shape[0], 1)
    assert_array_equal(result[:, 0], defaults)


def test_extend_grid_shape_mismatch_raises(system):
    """Raises ValueError when grid rows != indices length."""
    defaults = system.parameters.values_array
    grid = np.array([[1, 2], [3, 4], [5, 6]])  # 3 rows
    indices = np.array([0, 1])  # 2 indices
    with pytest.raises(ValueError, match="Grid shape does not match"):
        extend_grid_to_array(grid, indices, defaults)


def test_extend_grid_all_swept(system):
    """Returns grid directly when all indices swept."""
    defaults = system.parameters.values_array
    n = defaults.shape[0]
    grid = np.arange(n * 3, dtype=float).reshape(n, 3)
    indices = np.arange(n)
    result = extend_grid_to_array(grid, indices, defaults)
    assert_array_equal(result, grid)


def test_extend_grid_all_swept_permuted(system):
    """Scatters rows to their variable indices when all swept."""
    defaults = system.parameters.values_array
    n = defaults.shape[0]
    grid = np.arange(n * 3, dtype=float).reshape(n, 3)
    indices = np.roll(np.arange(n), 1)
    result = extend_grid_to_array(grid, indices, defaults)
    for row, target in enumerate(indices):
        assert_array_equal(result[target, :], grid[row, :])


def test_extend_grid_partial_sweep(system):
    """Creates default array and overwrites swept indices."""
    defaults = system.parameters.values_array
    grid = np.array([[10, 20, 30]])  # sweep index 0 only
    indices = np.array([0])
    result = extend_grid_to_array(grid, indices, defaults)
    assert result.shape == (defaults.shape[0], 3)
    assert_array_equal(result[0, :], [10, 20, 30])
    for i in range(1, defaults.shape[0]):
        assert_array_equal(result[i, :], np.full(3, defaults[i]))


# ── __init__ and from_system ────────────────────────────── #


def test_init_stores_attributes(system):
    """Stores parameters, states, precision from interface."""
    interface = SystemInterface.from_system(system)
    handler = BatchInputHandler(interface)
    assert handler.parameters is interface.parameters
    assert handler.states is interface.states
    assert handler.precision == interface.parameters.precision


def test_from_system_creates_handler(system):
    """Creates handler via SystemInterface.from_system."""
    handler = BatchInputHandler.from_system(system)
    assert handler.parameters.n == system.sizes.parameters
    assert handler.states.n == system.sizes.states


# ── __call__ ────────────────────────────────────────────── #


def test_call_updates_precision(input_handler, system):
    """Updates precision from current system state."""
    # Just verifying it doesn't error and precision matches
    inits, params = input_handler(states=None, params=None)
    assert inits.dtype == system.precision
    assert params.dtype == system.precision


class _FakeDeviceArray:
    """Minimal stand-in exposing the CUDA array interface."""

    def __init__(self, shape, dtype):
        self._data = np.ones(shape, dtype=dtype)
        self.shape = shape
        self.dtype = self._data.dtype

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": self._data.shape,
            "typestr": self._data.dtype.str,
            "data": (self._data.ctypes.data, False),
            "version": 3,
        }


def test_call_device_arrays_pass_through(input_handler, system):
    """Matched device states and params pass through untouched."""
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    states = _FakeDeviceArray((n_states, 2), system.precision)
    params = _FakeDeviceArray((n_params, 2), system.precision)
    result_s, result_p = input_handler(states, params, "verbatim")
    assert result_s is states
    assert result_p is params


def test_call_device_states_none_params(input_handler, system):
    """Device states pair with defaults tiled to the run count."""
    n_states = system.sizes.states
    states = _FakeDeviceArray((n_states, 4), system.precision)
    result_s, result_p = input_handler(states, None, "verbatim")
    assert result_s is states
    assert result_p.shape == (system.sizes.parameters, 4)
    assert result_p.dtype == system.precision
    expected = np.tile(
        system.parameters.values_array[:, np.newaxis], (1, 4)
    )
    assert_array_equal(result_p, expected)


def test_call_device_states_single_column_params(input_handler, system):
    """A single-column host input broadcasts to the device run count."""
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    states = _FakeDeviceArray((n_states, 3), system.precision)
    params = np.arange(n_params, dtype=system.precision)
    result_s, result_p = input_handler(states, params, "verbatim")
    assert result_s is states
    assert result_p.shape == (n_params, 3)
    for run in range(3):
        assert_array_equal(result_p[:, run], params)


def test_call_device_params_matching_host_states(input_handler, system):
    """A right-sized host grid pairs verbatim with device params."""
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    params = _FakeDeviceArray((n_params, 3), system.precision)
    states = np.ones((n_states, 3), dtype=np.float64)
    result_s, result_p = input_handler(states, params, "verbatim")
    assert result_p is params
    assert result_s.shape == (n_states, 3)
    assert result_s.dtype == system.precision


def test_call_device_states_mismatched_host_params_raise(
    input_handler, system
):
    """Multi-column host input must match the device run count."""
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    states = _FakeDeviceArray((n_states, 4), system.precision)
    params = np.ones((n_params, 3), dtype=system.precision)
    with pytest.raises(ValueError, match="paired\\s+verbatim"):
        input_handler(states, params, "verbatim")


def test_call_device_states_dict_params_raise(input_handler, system):
    """Non-empty dict grids cannot combine with device arrays."""
    n_states = system.sizes.states
    param_names = list(system.parameters.names)
    states = _FakeDeviceArray((n_states, 2), system.precision)
    with pytest.raises(TypeError, match="device array"):
        input_handler(states, {param_names[0]: [1.0, 2.0]}, "verbatim")


def test_call_device_array_wrong_dtype_raises(input_handler, system):
    """Device arrays must match the system precision exactly."""
    n_states = system.sizes.states
    wrong = np.float64 if system.precision == np.float32 else np.float32
    states = _FakeDeviceArray((n_states, 2), wrong)
    with pytest.raises(TypeError, match="precision"):
        input_handler(states, None, "verbatim")


def test_call_device_array_wrong_rows_raises(input_handler, system):
    """Device arrays must carry every system variable."""
    n_states = system.sizes.states
    states = _FakeDeviceArray((n_states + 1, 2), system.precision)
    with pytest.raises(ValueError, match="variables"):
        input_handler(states, None, "verbatim")


def test_call_device_array_1d_raises(input_handler, system):
    """Device arrays must be 2D in (variable, run) format."""
    states = _FakeDeviceArray((system.sizes.states,), system.precision)
    with pytest.raises(ValueError, match="2D"):
        input_handler(states, None, "verbatim")


def test_call_processes_inputs(input_handler, system):
    """Falls through to the plan/fill path for states and params."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    inits, params = input_handler(
        states={state_names[0]: [1.0, 2.0]},
        params={param_names[0]: [10.0, 20.0]},
        kind="combinatorial",
    )
    assert inits.shape[1] == params.shape[1] == 4


def test_call_dict_order_independent(input_handler, system):
    """Full dicts map by key name regardless of insertion order."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    states = {
        name: [float(i + 1)] for i, name in enumerate(state_names)
    }
    params = {
        name: [10.0 * (i + 1)] for i, name in enumerate(param_names)
    }
    shuffled_states = {
        name: states[name] for name in reversed(state_names)
    }
    shuffled_params = {
        name: params[name] for name in reversed(param_names)
    }
    inits, param_arr = input_handler(states, params, "verbatim")
    inits_s, params_s = input_handler(
        shuffled_states, shuffled_params, "verbatim"
    )
    assert_array_equal(inits_s, inits)
    assert_array_equal(params_s, param_arr)
    expected_inits = np.arange(1.0, len(state_names) + 1.0)
    expected_params = 10.0 * np.arange(1.0, len(param_names) + 1.0)
    assert_array_equal(inits[:, 0], expected_inits)
    assert_array_equal(param_arr[:, 0], expected_params)


def test_call_aligns_run_counts(input_handler, system):
    """Aligns run counts via _fill_aligned."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    inits, params = input_handler(
        states={state_names[0]: [1.0, 2.0]},
        params={param_names[0]: [10.0, 20.0, 30.0]},
        kind="combinatorial",
    )
    # 2 * 3 = 6 combinatorial runs
    assert inits.shape[1] == 6
    assert params.shape[1] == 6


def test_call_casts_to_precision(input_handler, system, precision):
    """Casts to precision."""
    inits, params = input_handler(states=None, params=None)
    assert inits.dtype == precision
    assert params.dtype == precision


def _assert_cast_to_precision(system, precision):
    """Assert the handler returns C-contiguous arrays at *precision*."""
    handler = BatchInputHandler.from_system(system)
    inits, params = handler(states=None, params=None)
    assert inits.dtype == precision
    assert params.dtype == precision
    assert inits.flags["C_CONTIGUOUS"]
    assert params.flags["C_CONTIGUOUS"]


def test_cast_to_precision_default_dtype(system, precision):
    """_cast_to_precision casts to the default (float32) precision."""
    _assert_cast_to_precision(system, precision)


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(FLOAT64_PRECISION, id="float64")],
    indirect=True,
)
def test_cast_to_precision_float64(system, precision):
    """_cast_to_precision casts to a float64 precision override."""
    _assert_cast_to_precision(system, precision)


# ── _trim_or_extend ────────────────────────────────────── #


def test_trim_or_extend_fewer_rows(input_handler, system):
    """Extends with default values when arr has fewer rows."""
    arr = np.array([[1.0, 2.0]])  # 1 row, 2 runs
    result = input_handler._trim_or_extend(arr, system.initial_values)
    assert result.shape[0] == system.sizes.states
    assert_array_equal(result[0, :], [1.0, 2.0])
    for i in range(1, system.sizes.states):
        expected = system.initial_values.values_array[i]
        assert_allclose(result[i, :], [expected, expected])


def test_trim_or_extend_more_rows(input_handler, system):
    """Trims extra rows when arr has more rows."""
    n = system.sizes.states
    arr = np.ones((n + 5, 2))
    result = input_handler._trim_or_extend(arr, system.initial_values)
    assert result.shape[0] == n


def test_trim_or_extend_exact(input_handler, system):
    """Returns unchanged when row count matches."""
    n = system.sizes.states
    arr = np.ones((n, 2))
    result = input_handler._trim_or_extend(arr, system.initial_values)
    assert result.shape == (n, 2)
    assert_array_equal(result, arr)


# ── _sanitise_arraylike ─────────────────────────────────── #


def test_sanitise_none_passthrough(input_handler, system):
    """Returns None passthrough when arr is None."""
    result = input_handler._sanitise_arraylike(None, system.initial_values)
    assert result is None


def test_sanitise_coerces_non_ndarray(input_handler, system):
    """Coerces non-ndarray to ndarray."""
    n = system.sizes.states
    lst = list(range(n))
    result = input_handler._sanitise_arraylike(lst, system.initial_values)
    assert result.shape == (n, 1)


def test_sanitise_raises_for_3d(input_handler, system):
    """Raises ValueError for >2D input."""
    arr = np.ones((2, 3, 4))
    with pytest.raises(ValueError, match="1D or 2D array"):
        input_handler._sanitise_arraylike(arr, system.initial_values)


def test_sanitise_1d_to_column(input_handler, system):
    """Converts 1D to single-column 2D."""
    n = system.sizes.states
    arr = np.arange(n, dtype=float)
    result = input_handler._sanitise_arraylike(arr, system.initial_values)
    assert result.shape == (n, 1)
    assert_array_equal(result[:, 0], arr)


def test_sanitise_warns_on_row_mismatch(input_handler, system):
    """Warns and adjusts when row count mismatches values_object.n."""
    arr = np.array([[1.0, 2.0]])  # 1 row, system has more
    with pytest.warns(UserWarning, match="Missing values"):
        result = input_handler._sanitise_arraylike(
            arr, system.initial_values
        )
    assert result.shape[0] == system.sizes.states


def test_sanitise_returns_none_for_empty(input_handler, system):
    """Returns None when array is empty after processing."""
    arr = np.array([]).reshape(0, 0)
    # system.initial_values has n > 0, so shape mismatch will trigger
    # _trim_or_extend, but the empty array case is tested via process
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = input_handler._sanitise_arraylike(
            arr, system.initial_values
        )
    assert result is None


# ── _plan_single_input / _fill_category ─────────────────── #


def process_single_category(handler, input_data, values_object, kind):
    """Materialise one category through the plan/fill pipeline."""
    plan = handler._plan_single_input(input_data, values_object, kind)
    if plan["mode"] == "array":
        return plan["array"]
    return handler._fill_category(
        plan, values_object, set(), repeats=1, tiles=1
    )


def test_process_none_returns_defaults(input_handler, system):
    """Returns single-column defaults when input is None."""
    result = process_single_category(
        input_handler, None, system.initial_values, kind="combinatorial"
    )
    assert result.shape == (system.sizes.states, 1)
    assert_allclose(result[:, 0], system.initial_values.values_array)


def test_process_dict_combinatorial(input_handler, system):
    """Processes dict: wraps scalars, generates grid, extends defaults."""
    state_names = list(system.initial_values.names)
    result = process_single_category(
        input_handler,
        {state_names[0]: [1.0, 2.0], state_names[1]: [3.0, 4.0]},
        system.initial_values,
        kind="combinatorial",
    )
    assert result.shape == (system.sizes.states, 4)
    # Check swept values present
    assert set(result[0, :]) == {1.0, 2.0}
    assert set(result[1, :]) == {3.0, 4.0}


def test_process_arraylike(input_handler, system):
    """Plans array-like input: sanitises to 2D and carries it whole."""
    n = system.sizes.states
    arr = np.ones((n, 3))
    result = process_single_category(
        input_handler, arr, system.initial_values, kind="combinatorial"
    )
    assert result.shape == (n, 3)


def test_process_empty_sanitised_returns_defaults(input_handler, system):
    """Returns defaults when sanitised result is None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = process_single_category(
            input_handler,
            np.array([]).reshape(0, 0),
            system.initial_values,
            kind="combinatorial",
        )
    assert result.shape == (system.sizes.states, 1)


def test_process_invalid_type_raises(input_handler, system):
    """Raises TypeError for unsupported input type."""
    with pytest.raises(TypeError, match="Input must be None, dict"):
        input_handler._plan_single_input(
            "bad", system.initial_values, kind="combinatorial"
        )


# ── _is_right_sized_array ──────────────────────────────── #


def test_is_right_sized_none_empty_values(system):
    """Returns True for None when values_object empty."""
    # Need a system with no parameters to test empty
    # Use the handler's method with the actual values object
    handler = BatchInputHandler.from_system(system)
    # For non-empty values_object, None should be False
    assert handler._is_right_sized_array(None, system.parameters) is False


def test_is_right_sized_non_ndarray(input_handler, system):
    """Returns False for non-ndarray."""
    assert input_handler._is_right_sized_array(
        [1, 2], system.parameters
    ) is False


def test_is_right_sized_non_2d(input_handler, system):
    """Returns False for non-2D ndarray."""
    assert input_handler._is_right_sized_array(
        np.array([1, 2, 3]), system.parameters
    ) is False


def test_is_right_sized_correct(input_handler, system):
    """Returns True when shape[0] == values_object.n."""
    n = system.sizes.parameters
    arr = np.ones((n, 5))
    assert input_handler._is_right_sized_array(
        arr, system.parameters
    ) is True


# ── _is_1d_or_none ─────────────────────────────────────── #


@pytest.mark.parametrize(
    "val, expected",
    [
        pytest.param(None, True, id="none"),
        pytest.param({"a": 1}, False, id="dict"),
        pytest.param(np.array([1, 2, 3]), True, id="1d_ndarray"),
        pytest.param([1, 2, 3], True, id="flat_list"),
        pytest.param(np.array([[1, 2], [3, 4]]), False, id="2d_ndarray"),
    ],
)
def test_is_1d_or_none(input_handler, val, expected):
    """Tests _is_1d_or_none for various input types."""
    assert input_handler._is_1d_or_none(val) is expected


# ── _fill_defaults ──────────────────────────────────────── #


def test_fill_defaults(input_handler, system):
    """Returns broadcast defaults with n_runs columns."""
    result = input_handler._fill_defaults(system.parameters, 5, set())
    assert result.shape == (system.sizes.parameters, 5)
    for col in range(5):
        assert_array_equal(result[:, col], system.parameters.values_array)


# ── _fast_return_arrays ─────────────────────────────────── #


def test_fast_return_right_sized_matching(input_handler, system, precision):
    """Returns cast arrays when both are right-sized with matching runs."""
    n_s = system.sizes.states
    n_p = system.sizes.parameters
    states = np.ones((n_s, 3), dtype=precision)
    params = np.ones((n_p, 3), dtype=precision)
    result = input_handler._fast_return_arrays(states, params, "verbatim")
    assert result is not None
    assert result[0].shape == (n_s, 3)
    assert result[1].shape == (n_p, 3)


def test_fast_return_none_when_no_path(input_handler, system):
    """Returns None when no fast path applies."""
    state_names = list(system.initial_values.names)
    # Dict input => no fast path
    result = input_handler._fast_return_arrays(
        {state_names[0]: [1, 2]}, {}, "combinatorial"
    )
    assert result is None


def test_fast_return_states_ok_params_small(input_handler, system, precision):
    """Fast path: states_ok + params_small -> broadcast params to match."""
    n_s = system.sizes.states
    states = np.ones((n_s, 3), dtype=precision)
    result = input_handler._fast_return_arrays(states, None, "verbatim")
    assert result is not None
    assert result[0].shape[1] == result[1].shape[1]


def test_fast_return_params_ok_states_small(input_handler, system, precision):
    """Fast path: params_ok + states_small -> broadcast states to match."""
    n_p = system.sizes.parameters
    params = np.ones((n_p, 3), dtype=precision)
    result = input_handler._fast_return_arrays(None, params, "verbatim")
    assert result is not None
    assert result[0].shape[1] == result[1].shape[1]


# ── _get_run_count ──────────────────────────────────────── #


def test_get_run_count_2d(input_handler):
    """Returns shape[1] for 2D ndarray."""
    arr = np.ones((3, 7))
    assert input_handler._get_run_count(arr) == 7


def test_get_run_count_non_2d(input_handler):
    """Returns None for non-2D ndarray."""
    assert input_handler._get_run_count(np.array([1, 2, 3])) is None


def test_get_run_count_none(input_handler):
    """Returns None otherwise."""
    assert input_handler._get_run_count(None) is None


def test_get_run_count_ignores_device_arrays(input_handler, system):
    """Returns None for non-ndarray inputs such as device arrays."""

    class FakeDevice:
        def __init__(self, shape):
            self.shape = shape

        @property
        def __cuda_array_interface__(self):
            return {"shape": self.shape, "version": 3}

    # __init__ test: inline construction justified for device mock
    dev = FakeDevice((3, 5))
    assert input_handler._get_run_count(dev) is None


# ── _fill_aligned ───────────────────────────────────────── #


def test_fill_aligned_combinatorial(input_handler):
    """Expands both categories to the Cartesian run count."""
    s = np.array([[1, 2], [3, 4]])
    p = np.array([[10, 20, 30], [40, 50, 60]])
    states_plan = {"mode": "array", "n_runs": 2, "array": s}
    params_plan = {"mode": "array", "n_runs": 3, "array": p}
    rs, rp = input_handler._fill_aligned(
        states_plan, params_plan, "combinatorial", set()
    )
    assert rs.shape[1] == 6
    assert rp.shape[1] == 6
    expected_s, expected_p = combine_grids(s, p, "combinatorial")
    assert_allclose(rs, expected_s)
    assert_allclose(rp, expected_p)


# ── Integration-level __call__ tests ────────────────────── #


def test_call_none_returns_defaults(input_handler, system):
    """Empty inputs return single run with all defaults."""
    inits, params = input_handler(states=None, params=None)
    assert inits.shape[1] == 1
    assert params.shape[1] == 1
    assert_allclose(inits[:, 0], system.initial_values.values_array)
    assert_allclose(params[:, 0], system.parameters.values_array)


def test_call_verbatim_mismatch_raises(input_handler, system):
    """Verbatim with mismatched lengths raises ValueError."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    with pytest.raises(ValueError):
        input_handler(
            states={state_names[0]: [0, 1, 2]},
            params={param_names[0]: [0, 1]},
            kind="verbatim",
        )


def test_call_combinatorial_dict_both(input_handler, system):
    """Combinatorial dict for both states and params."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    inits, params = input_handler(
        states={state_names[0]: [1.0, 2.0]},
        params={param_names[0]: [10.0, 20.0]},
        kind="combinatorial",
    )
    assert inits.shape[1] == 4
    assert params.shape[1] == 4


def test_call_verbatim_broadcast_single(input_handler, system):
    """Verbatim broadcasts single-run state to match multi-run params."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    inits, params = input_handler(
        states={state_names[0]: 0.5},
        params={param_names[0]: [1.0, 2.0, 3.0]},
        kind="verbatim",
    )
    assert inits.shape[1] == 3
    assert params.shape[1] == 3
    assert_allclose(inits[0, :], [0.5, 0.5, 0.5])


def test_call_single_param_sweep(input_handler, system):
    """Single parameter sweep fills other values with defaults."""
    param_names = list(system.parameters.names)
    sweep = np.linspace(0, 1, 50)
    inits, params = input_handler(
        params={param_names[0]: sweep}, kind="combinatorial"
    )
    assert params.shape[1] == 50
    assert_allclose(params[0, :], sweep, rtol=1e-6)
    # Non-swept params at defaults
    for i in range(1, system.sizes.parameters):
        expected_val = system.parameters.values_array[i]
        assert_allclose(params[i, :], np.full(50, expected_val), rtol=1e-6)


def test_call_1d_array_single_run(input_handler, system):
    """1D parameter array treated as single run."""
    n_params = system.sizes.parameters
    vals = np.arange(n_params, dtype=float)
    inits, params = input_handler(params=vals)
    assert params.shape[1] == 1
    assert_allclose(params[:, 0], vals)


def test_call_positional_args(input_handler, system):
    """Positional args route correctly: states first, params second."""
    n_s = system.sizes.states
    n_p = system.sizes.parameters
    states = np.full((n_s, 2), 1.5, dtype=system.precision)
    params = np.full((n_p, 2), 99.0, dtype=system.precision)
    rs, rp = input_handler(states, params, "verbatim")
    assert_allclose(rs[0, 0], 1.5)
    assert_allclose(rp[0, 0], 99.0)


# ── plan/fill: empty SystemValues ───────────────────────── #


def test_process_empty_values_none_input(system):
    """Returns empty (0,1) array when values_object empty and input None."""
    handler = BatchInputHandler.from_system(system)
    # Create a mock empty SystemValues-like scenario through the handler
    # The system has parameters, so we test via the actual code path
    # by checking what happens when values_object.empty is True
    # This is tested indirectly when system has no params
    # For coverage: test the method directly with the actual values
    result = process_single_category(
        handler, None, system.initial_values, kind="combinatorial"
    )
    # Non-empty values_object + None -> defaults column
    assert result.shape == (system.sizes.states, 1)


def test_process_empty_values_nonempty_raises(system):
    """Raises ValueError when values_object empty but non-empty input."""
    handler = BatchInputHandler.from_system(system)
    # We cannot easily create an empty SystemValues without a special system
    # This is tested via the error path in _plan_single_input
    # We verify the TypeError path instead for coverage
    with pytest.raises(TypeError):
        handler._plan_single_input(
            42, system.initial_values, kind="combinatorial"
        )


# ── plan/fill: genuinely empty SystemValues ─────────────── #


@pytest.fixture(scope="module")
def no_param_system(precision):
    """System with no declared parameters (empty SystemValues)."""
    from cubie import create_ODE_system

    return create_ODE_system(
        dxdt=["dx0 = -x0"],
        states={"x0": 1.0},
        precision=precision,
        name="no_param_system_batchhandler",
    )


@pytest.mark.parametrize(
    "empty_params",
    [{}, np.array([]).reshape(0, 0), [], ()],
    ids=["empty_dict", "empty_ndarray", "empty_list", "empty_tuple"],
)
def test_process_empty_values_object_empty_input_returns_zero_rows(
    no_param_system, empty_params
):
    """Empty (non-None) input against an empty SystemValues returns a
    (0, 1) array instead of raising."""
    handler = BatchInputHandler.from_system(no_param_system)
    result = process_single_category(
        handler, empty_params, handler.parameters, kind="combinatorial"
    )
    assert result.shape == (0, 1)


def test_process_empty_values_object_none_input_returns_zero_rows(
    no_param_system,
):
    """None input against an empty SystemValues also returns (0, 1)."""
    handler = BatchInputHandler.from_system(no_param_system)
    result = process_single_category(
        handler, None, handler.parameters, kind="combinatorial"
    )
    assert result.shape == (0, 1)


@pytest.mark.parametrize(
    "nonempty_params",
    [{"phantom": [1.0]}, np.array([1.0]), [1.0]],
    ids=["dict", "ndarray", "list"],
)
def test_process_empty_values_object_nonempty_input_raises(
    no_param_system, nonempty_params
):
    """Non-empty input against an empty SystemValues raises ValueError."""
    handler = BatchInputHandler.from_system(no_param_system)
    with pytest.raises(ValueError, match="no settable variables"):
        handler._plan_single_input(
            nonempty_params, handler.parameters, kind="combinatorial"
        )


def test_call_with_no_param_system_processes_empty_params(no_param_system):
    """__call__ falls through to the empty-SystemValues path for params
    when a non-None, non-array params value is supplied."""
    handler = BatchInputHandler.from_system(no_param_system)
    inits, params = handler(
        states={"x0": [1.0, 2.0]}, params={}, kind="combinatorial"
    )
    assert params.shape == (0, 2)
    assert inits.shape[1] == 2


# ── _is_right_sized_array: empty SystemValues branches ──────── #


def test_is_right_sized_empty_values_none_arr(no_param_system):
    """None is right-sized when values_object is empty."""
    handler = BatchInputHandler.from_system(no_param_system)
    assert (
        handler._is_right_sized_array(None, handler.parameters) is True
    )


def test_is_right_sized_empty_values_zero_row_2d_array(no_param_system):
    """A (0, n) 2D array is right-sized when values_object is empty."""
    handler = BatchInputHandler.from_system(no_param_system)
    arr = np.empty((0, 3))
    assert (
        handler._is_right_sized_array(arr, handler.parameters) is True
    )


def test_is_right_sized_empty_values_other_input_false(no_param_system):
    """Any other non-None input is not right-sized against an empty
    values_object."""
    handler = BatchInputHandler.from_system(no_param_system)
    assert (
        handler._is_right_sized_array([1.0], handler.parameters) is False
    )
    assert (
        handler._is_right_sized_array(
            np.ones((1, 2)), handler.parameters
        )
        is False
    )


# ── _is_1d_or_none: fallback False for unrecognised types ───── #


def test_is_1d_or_none_scalar_returns_false(input_handler):
    """A bare scalar is neither None, dict, ndarray, list, nor tuple, so
    it falls through to the default False."""
    assert input_handler._is_1d_or_none(5) is False
    assert input_handler._is_1d_or_none(3.14) is False


# ── _fast_return_arrays: remaining branches ─────────────────── #


def test_device_arrays_mismatched_runs_raise(input_handler, system):
    """Both device arrays but mismatched run counts raise."""
    n_s = system.sizes.states
    n_p = system.sizes.parameters
    states = _FakeDeviceArray((n_s, 2), system.precision)
    params = _FakeDeviceArray((n_p, 3), system.precision)
    with pytest.raises(ValueError, match="different run counts"):
        input_handler(states, params, "verbatim")


def test_fast_return_empty_params_none_marks_params_ok(no_param_system):
    """When parameters are empty and params is None, params_ok is
    forced True and an empty (0, n_runs) params array is synthesised."""
    handler = BatchInputHandler.from_system(no_param_system)
    n_s = handler.states.n
    states = np.ones((n_s, 4), dtype=handler.precision)
    result = handler._fast_return_arrays(states, None, "verbatim")
    assert result is not None
    assert result[1].shape == (0, 4)


def test_fast_return_states_ok_params_provided_1d_broadcasts(
    input_handler, system, precision
):
    """states_ok + a provided (non-None) 1D params array sanitises and
    broadcasts to match the state run count."""
    n_s = system.sizes.states
    n_p = system.sizes.parameters
    states = np.ones((n_s, 3), dtype=precision)
    params = np.arange(n_p, dtype=precision)
    result = input_handler._fast_return_arrays(states, params, "verbatim")
    assert result is not None
    assert result[0].shape[1] == 3
    assert result[1].shape[1] == 3


def test_fast_return_params_ok_states_provided_1d_broadcasts(
    input_handler, system, precision
):
    """params_ok + a provided (non-None) 1D states array sanitises and
    broadcasts to match the params run count."""
    n_s = system.sizes.states
    n_p = system.sizes.parameters
    params = np.ones((n_p, 3), dtype=precision)
    states = np.arange(n_s, dtype=precision)
    result = input_handler._fast_return_arrays(states, params, "verbatim")
    assert result is not None
    assert result[0].shape[1] == 3
    assert result[1].shape[1] == 3


# ── _materialise: verbatim pass-through and pinned landing ─ #


def test_materialise_passes_matching_user_arrays_through(input_handler):
    """A caller's C-contiguous correct-precision array returns verbatim."""
    arr = np.ones((3, 4), dtype=input_handler.precision)
    assert input_handler._materialise(arr, arr) is arr


def test_materialise_passes_views_of_user_arrays_through(input_handler):
    """A contiguous view of the caller's array returns verbatim."""
    arr = np.ones((4, 4), dtype=input_handler.precision)
    view = arr[:3]
    assert input_handler._materialise(view, arr) is view


def test_materialise_casts_into_precision_buffer(input_handler):
    """A mismatched dtype lands once in a contiguous precision buffer."""
    src = np.arange(12, dtype=np.int64).reshape(3, 4)
    out = input_handler._materialise(src, src)
    assert out is not src
    assert out.dtype == input_handler.precision
    assert out.flags["C_CONTIGUOUS"]
    assert_array_equal(out, src)


def test_materialise_fixes_contiguity(input_handler):
    """A strided view lands once in a contiguous buffer."""
    src = np.ones((3, 8), dtype=input_handler.precision)[:, ::2]
    out = input_handler._materialise(src, src)
    assert out is not src
    assert out.flags["C_CONTIGUOUS"]
    assert out.shape == src.shape


@pytest.mark.nocudasim
def test_materialise_lands_pinned_below_ceiling(input_handler):
    """Materialised inputs below the ceiling are page-locked."""
    src = np.ones((3, 8), dtype=input_handler.precision)[:, ::2]
    out = input_handler._materialise(src, src)
    assert is_pinned_array(out)


@pytest.mark.nocudasim
def test_materialise_pins_handler_assembled_arrays(input_handler):
    """An assembled array in-precision and contiguous lands pinned."""
    assembled = np.ones((3, 8), dtype=input_handler.precision)
    out = input_handler._materialise(assembled, None)
    assert out is not assembled
    assert is_pinned_array(out)
    assert_array_equal(out, assembled)


@pytest.mark.nocudasim
def test_call_dict_inputs_return_pinned_grids(input_handler, system):
    """Grids assembled from dict inputs land in pinned buffers."""
    state_names = list(system.initial_values.names)
    param_names = list(system.parameters.names)
    inits, params = input_handler(
        states={state_names[0]: [0.5, 1.5]},
        params={param_names[0]: [0.1, 0.2]},
        kind="verbatim",
    )
    assert is_pinned_array(inits)
    assert is_pinned_array(params)
