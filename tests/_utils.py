from __future__ import annotations

import attrs
import math
from functools import lru_cache
from typing import Mapping, Optional, Union, Dict, Any, Callable

import numpy as np
import pytest
from cubie.cuda_simsafe import cuda, int32, numba_from_dtype as from_dtype
from cubie.memory import default_memmgr
from cubie.memory.mem_manager import MemoryManager
from numpy.testing import assert_allclose

from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie.integrators.step_control import (
    CONTROLLER_GAIN_NAMES,
    _CONTROLLER_REGISTRY,
    filter_coefficients_to_gains,
)
from cubie.odesystems.symbolic import SymbolicODE
from cubie.batchsolving.solver import Solver
from cubie.outputhandling import OutputFunctions
from cubie.array_interpolator import ArrayInterpolator
from cubie.odesystems.baseODE import BaseODE
from numpy.typing import NDArray
from tests.integrators.cpu_reference import CPUAdaptiveController

Array = NDArray[np.floating]


class MockMemoryManager(MemoryManager):
    """Memory manager whose reported free memory is settable.

    Chunking is a solve-time decision keyed off the manager's
    reported free memory, so tests set ``_custom_limit`` (or pass
    ``forced_free_mem``) to force a chunk count without touching
    real device state.
    """

    def __init__(self, **kwargs):
        # Set the limit first: attrs __init__ probes get_memory_info.
        self._custom_limit = kwargs.get("forced_free_mem", 950)
        super().__init__()

    def get_memory_info(self):
        return int(self._custom_limit), int(8192)


# --------------------------------------------------------------------------- #
#                      Standard Parameter Sets                                #
# --------------------------------------------------------------------------- #

MID_RUN_PARAMS = {
    "dt": 0.001,
    "save_every": 0.02,
    "summarise_every": 0.1,
    "sample_summaries_every": 0.02,
    "dt_max": 0.5,
    "output_types": ["state", "time", "observables", "mean"],
}

# The shared pool of session override sets. Draw from these before
# adding a new set: every distinct dict keys its own session fixture
# chain, so a near-duplicate costs a chain rebuild for nothing. A
# genuinely unique set stays at its test with a comment naming the
# test condition that requires it.

# One representative algorithm/controller combo per algorithm family.
# STEP_CASES wraps these for the per-algorithm numerical tests, and
# ALGORITHM_CHAIN_SETS below merges them over MID_RUN_PARAMS — tests
# that just need "an adaptive chain" or "an implicit chain" use those
# merged sets so they ride the numerical tests' session chains
# instead of keying new ones.
ALGORITHM_CONTROLLER_COMBOS = {
    "euler": {"algorithm": "euler", "step_controller": "fixed"},
    "backwards_euler": {
        "algorithm": "backwards_euler", "step_controller": "fixed",
    },
    "backwards_euler_pc": {
        "algorithm": "backwards_euler_pc", "step_controller": "fixed",
    },
    "crank_nicolson": {
        "algorithm": "crank_nicolson", "step_controller": "pid",
    },
    "rosenbrock": {"algorithm": "rosenbrock", "step_controller": "i"},
    "erk": {"algorithm": "erk", "step_controller": "pid"},
    "dirk": {"algorithm": "dirk", "step_controller": "fixed"},
    "firk": {"algorithm": "firk", "step_controller": "fixed"},
}

# Precision flip; the float32 case is the unparametrised default.
FLOAT64_PRECISION = {"precision": np.float64}

# Timing keys all cleared: save_last path and no-summaries path.
STATE_OBS_NO_TIMING = {
    "output_types": ["state", "observables"],
    "save_every": None,
    "summarise_every": None,
    "sample_summaries_every": None,
}

# Summary-only outputs with no timing (duration-dependent path).
SUMMARY_ONLY_NO_TIMING = {
    "output_types": ["mean"],
    "save_every": None,
    "summarise_every": None,
    "sample_summaries_every": None,
}

# Summary-only outputs with explicit timing (no derivation needed).
SUMMARY_ONLY_TIMED = {
    "output_types": ["mean"],
    "save_every": None,
    "summarise_every": 0.1,
    "sample_summaries_every": 0.05,
}

# None overrides unset the spine's explicit linear solve values.
UNSET_LINEAR_SOLVE = {
    "linear_correction_type": None,
    "krylov_max_iters": None,
    "preconditioner_type": None,
}

# The torn systems have no observables to save.
TORN_NO_OBSERVABLES = {
    "output_types": ["state", "time"],
    "saved_observable_indices": [],
    "summarised_observable_indices": [],
}

# torn_time chain for the DAE-init solve and codegen source tests.
TORN_INIT_COMMON = {
    "system_type": "torn_time",
    "precision": np.float64,
    "algorithm": "backwards_euler",
    "step_controller": "fixed",
    "dt": 1e-3,
    "save_every": 0.025,
    "newton_atol": 1e-10,
    "newton_rtol": 1e-10,
    # Stage-solver budget; the initialiser keeps its own fixed cap.
    "newton_max_iters": 12,
    **TORN_NO_OBSERVABLES,
    **UNSET_LINEAR_SOLVE,
}

# One set per adaptive controller kind. rtol is pinned to zero so
# the scaled norm's denominator is exactly atol, independent of the
# state values — the injected error vectors in the controller tests
# then map to known norm ratios.
CONTROLLER_TOLERANCE_SETS = {
    "i": {"step_controller": "i", "atol": 1e-3, "rtol": 0.0},
    "pi": {"step_controller": "pi", "atol": 1e-3, "rtol": 0.0},
    "pid": {"step_controller": "pid", "atol": 1e-3, "rtol": 0.0},
    "gustafsson": {
        "step_controller": "gustafsson", "atol": 1e-3, "rtol": 0.0,
    },
}


# Specific-tableau combos, marked specific_algos: both CI legs
# deselect them, so per-tableau coverage runs only on demand. The
# per-tableau loop tests parametrize with the same merged cases via
# ALGORITHM_CHAIN_CASES, so the two suites share one chain per name.
# Default tableaus appear only under their family alias ("erk" is
# dormand-prince-54, "dirk" is l_stable_dirk_3, "firk" is
# gauss_legendre_2) — an explicit-alias twin would key a duplicate
# chain for the same configuration.
SPECIFIC_ALGORITHM_COMBOS = {
    # Specific ERK tableaus
    "erk-cash-karp-54": {
        "algorithm": "cash-karp-54", "step_controller": "pid",
    },
    "erk-fehlberg-45": {
        "algorithm": "fehlberg-45", "step_controller": "i",
    },
    "erk-bogacki-shampine-32": {
        "algorithm": "bogacki-shampine-32", "step_controller": "pid",
    },
    "erk-heun-21": {"algorithm": "heun-21", "step_controller": "fixed"},
    "erk-ralston-33": {
        "algorithm": "ralston-33", "step_controller": "fixed",
    },
    "erk-classical-rk4": {
        "algorithm": "classical-rk4", "step_controller": "fixed",
    },
    "erk-dop853": {"algorithm": "dop853", "step_controller": "pid"},
    "erk-tsit5": {"algorithm": "tsit5", "step_controller": "pid"},
    "erk-vern7": {"algorithm": "vern7", "step_controller": "pid"},
    # Specific DIRK tableaus
    "dirk-implicit-midpoint": {
        "algorithm": "implicit_midpoint", "step_controller": "fixed",
    },
    "dirk-trapezoidal": {
        "algorithm": "trapezoidal_dirk", "step_controller": "fixed",
    },
    "dirk-sdirk-2-2": {
        "algorithm": "sdirk_2_2", "step_controller": "fixed",
    },
    "dirk-l-stable-4": {
        "algorithm": "l_stable_sdirk_4",
        "step_controller": "pi",
        "dt_min": 1e-6,
        "dt": 1e-3,
        "dt_max": 0.5,
    },
    "dirk-kvaerno3-fixed": {
        "algorithm": "kvaerno3", "step_controller": "fixed",
    },
    "dirk-kvaerno3-adaptive": {
        "algorithm": "kvaerno3", "step_controller": "pid",
    },
    "dirk-kvaerno5-fixed": {
        "algorithm": "kvaerno5", "step_controller": "fixed",
    },
    "dirk-kvaerno5-adaptive": {
        "algorithm": "kvaerno5", "step_controller": "pid",
    },
    # Specific FIRK tableaus
    "firk-radau": {
        "algorithm": "radau",
        "step_controller": "pi",
        "dt_min": 1e-6,
        "dt": 1e-3,
        "dt_max": 0.5,
    },
    "firk-radau-iia-3": {
        "algorithm": "radau_iia_3",
        "step_controller": "gustafsson",
        "dt_min": 1e-6,
        "dt": 1e-3,
        "dt_max": 0.5,
    },
    "firk-radau-iia-9": {
        "algorithm": "radau_iia_9",
        "step_controller": "gustafsson",
        "dt_min": 1e-6,
        "dt": 1e-3,
        "dt_max": 0.5,
    },
    "firk-gauss-legendre-4": {
        "algorithm": "firk_gauss_legendre_4", "step_controller": "fixed",
    },
    "firk-gauss-legendre-4-adaptive": {
        "algorithm": "firk_gauss_legendre_4",
        "step_controller": "gustafsson",
        "dt_min": 1e-6,
        "dt": 1e-3,
        "dt_max": 0.5,
    },
    # Specific Rosenbrock-W tableaus
    "rosenbrock-ros3p": {"algorithm": "ros3p", "step_controller": "pid"},
    "rosenbrock-ode23s": {
        "algorithm": "ode23s", "step_controller": "pid",
    },
    "rosenbrock-rodas3p": {
        "algorithm": "rodas3p", "step_controller": "pid",
    },
}

STEP_CASES = [
    pytest.param(combo, id=name)
    for name, combo in ALGORITHM_CONTROLLER_COMBOS.items()
] + [
    pytest.param(combo, id=name, marks=pytest.mark.specific_algos)
    for name, combo in SPECIFIC_ALGORITHM_COMBOS.items()
]


def merge_dicts(*dicts):
    """Merge multiple dictionaries, later dicts override earlier ones.

    Used to combine base settings (e.g., MID_RUN_PARAMS) with
    test-specific overrides into a single solver_settings_override.

    Parameters
    ----------
    *dicts : dict
        Dictionaries to merge. Later dicts override earlier ones.

    Returns
    -------
    dict
        Merged dictionary.
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def merge_param(base_settings, param):
    """Merge base settings into a pytest.param case.

    Combines base settings (e.g., MID_RUN_PARAMS) with test case settings,
    preserving pytest.param id and marks.

    Parameters
    ----------
    base_settings : dict
        Base settings to merge (applied first).
    param : pytest.param or dict
        Test case param. Can be pytest.param with id/marks or plain dict.

    Returns
    -------
    pytest.param
        Merged param with combined settings, original id and marks.
    """
    import pytest

    if hasattr(param, "values"):
        # It's a pytest.param
        case_settings = param.values[0] if param.values else {}
        merged = merge_dicts(base_settings, case_settings)
        return pytest.param(
            merged,
            id=param.id,
            marks=param.marks if param.marks else (),
        )
    else:
        # It's a plain dict
        return pytest.param(merge_dicts(base_settings, param))


# Merged cases with STEP_OVERRIDES baked in
ALGORITHM_PARAM_SETS = [
    merge_param(MID_RUN_PARAMS, case) for case in STEP_CASES
]

# The same merged cases keyed by name, marks preserved: the
# per-tableau loop tests and any test needing "the adaptive chain"
# or "the implicit chain" parametrize with these, so they share the
# per-algorithm numerical tests' session chains (identical dict
# objects) and the specific_algos marks stay consistent.
ALGORITHM_CHAIN_CASES = dict(
    zip(
        list(ALGORITHM_CONTROLLER_COMBOS) + list(SPECIFIC_ALGORITHM_COMBOS),
        ALGORITHM_PARAM_SETS,
    )
)
ALGORITHM_CHAIN_SETS = {
    name: ALGORITHM_CHAIN_CASES[name].values[0]
    for name in ALGORITHM_CONTROLLER_COMBOS
}


def calculate_expected_summaries(
    state,
    observables,
    summarised_state_indices,
    summarised_observable_indices,
    samples_per_summary,
    output_types,
    summary_height_per_variable,
    precision,
    sample_summaries_every=1.0,
    exclude_first=False,
):
    """Helper function to calculate expected summary values from a given
    pair of state and observable arrays. Summarises the whole output state
    and observable array, select from within this if testing for selective
    summarisation.

    Arguments:
    - state: 2D array of shape (summary_samples, n_saved_states)
        output generated by system.
    - observables: 2D array of shape (summary_samples,
        n_saved_observables) output generated by system.
    - samples_per_summary: Number of samples to summarise over (batch size)
    - output_types: List of output function names to apply
        (e.g. ["mean", "peaks[3]", "max", "rms"])
    - precision: Numpy dtype to use for the output arrays
        (e.g. np.float32 or np.float64)
    - sample_summaries_every: Time between summary samples
        (for derivative calculations). Default: 1.0
    - exclude_first: If True, exclude the first sample (t=0) from summary
        calculations. Used when mimicking IVP loop behavior. Default: False.

    Returns:
    - expected_state_summaries: 2D array of shape (summary_samples,
        n_saved_states * summary_size_per_state)
    - expected_obs_summaries: 2D array of shape (summary_samples,
        n_saved_observables * summary_size_per_state)
    """
    # Optionally exclude t=0 row (first sample) from summary calculations
    # to match IVP loop behavior where first update_summaries is skipped
    if exclude_first:
        state = state[1:, summarised_state_indices]
        observables = observables[1:, summarised_observable_indices]
    else:
        state = state[:, summarised_state_indices]
        observables = observables[:, summarised_observable_indices]
    n_saved_states = state.shape[1]
    n_saved_observables = observables.shape[1]
    saved_samples = state.shape[0]
    summary_samples = int(saved_samples / samples_per_summary)

    state_summaries_height = summary_height_per_variable * n_saved_states
    obs_summaries_height = summary_height_per_variable * n_saved_observables

    expected_state_summaries = np.zeros(
        (summary_samples, state_summaries_height), dtype=precision
    )
    expected_obs_summaries = np.zeros(
        (summary_samples, obs_summaries_height), dtype=precision
    )

    for _input_array, _output_array in (
        (state, expected_state_summaries),
        (observables, expected_obs_summaries),
    ):
        # When exclude_first=True, peak indices need +1 offset to convert
        # from sliced array indices to original save_idx values
        peak_index_offset = 1 if exclude_first else 0
        calculate_single_summary_array(
            _input_array,
            samples_per_summary,
            summary_height_per_variable,
            output_types,
            output_array=_output_array,
            sample_summaries_every=sample_summaries_every,
            peak_index_offset=peak_index_offset,
        )

    return expected_state_summaries, expected_obs_summaries


def calculate_single_summary_array(
    input_array,
    samples_per_summary,
    summary_size_per_state,
    output_functions_list,
    output_array,
    sample_summaries_every=1.0,
    peak_index_offset=0,
):
    """Summarise states in input array in the same way that the device
    functions do.

    Arguments:
    - input_array: 2D array of shape (n_items, n_samples) with the input
        data to summarise
    - samples_per_summary: Number of samples to summarise over
    - summary_size_per_state: Number of summary values per state
        (e.g. 1 for mean, 1 + n_peaks for mean and peaks[n])
    - output_functions_list: List of output function names to apply
        (e.g. ["mean", "peaks[3]", "max", "rms"])
    - n_peaks: Number of peaks to find in the "peaks[n]" output function
    - output_array: 2D array to store the summarised output,
        shape (n_items * summary_size_per_state, n_samples)
    - sample_summaries_every: Time between summary samples
        (for derivative calculations). Default: 1.0
    - peak_index_offset: Offset added to peak indices. When exclude_first
        is True, this is 1 to convert sliced array indices to save_idx
        values. Default: 0.

    Returns:
    - None, but output_array is filled with the summarised values.

    """
    summary_samples = int(input_array.shape[0] / samples_per_summary)
    try:
        n_items = output_array.shape[1] // summary_size_per_state
    except ZeroDivisionError:
        n_items = 0

    # Manual cycling through possible summaries_array to match the approach
    # used when building the device functions
    for j in range(n_items):
        for i in range(summary_samples):
            summary_index = 0
            for output_type in output_functions_list:
                start_index = i * samples_per_summary
                end_index = (i + 1) * samples_per_summary
                if output_type == "mean":
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = np.mean(
                        input_array[start_index:end_index, j],
                        axis=0,
                    )
                    summary_index += 1

                if output_type.startswith("peaks"):
                    n_peaks = output_type.split("[", 1)[1].split("]", 1)[0]
                    n_peaks = int(n_peaks) if n_peaks else 0
                    # Use the last two samples, like the live version does
                    start_index = i * samples_per_summary - 2 if i > 0 else 0
                    maxima = (
                        local_maxima(
                            input_array[start_index:end_index, j],
                        )[:n_peaks]
                        + start_index
                        + peak_index_offset  # Offset for sliced array indexing
                    )
                    output_start_index = (
                        j * summary_size_per_state + summary_index
                    )
                    output_array[
                        i,
                        output_start_index : output_start_index + maxima.size,
                    ] = maxima
                    summary_index += n_peaks

                if output_type == "max":
                    _max = np.max(
                        input_array[start_index:end_index, j], axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _max
                    summary_index += 1

                if output_type == "rms":
                    rms = np.sqrt(
                        np.mean(
                            input_array[start_index:end_index, j] ** 2, axis=0
                        )
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = rms
                    summary_index += 1

                if output_type == "std":
                    std = np.std(input_array[start_index:end_index, j], axis=0)
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = std
                    summary_index += 1

                if output_type == "min":
                    _min = np.min(
                        input_array[start_index:end_index, j], axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _min
                    summary_index += 1

                if output_type == "max_magnitude":
                    max_mag = np.max(
                        np.abs(input_array[start_index:end_index, j]), axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = max_mag
                    summary_index += 1

                if output_type == "extrema":
                    _max = np.max(
                        input_array[start_index:end_index, j], axis=0
                    )
                    _min = np.min(
                        input_array[start_index:end_index, j], axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _max
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _min
                    summary_index += 2

                if output_type.startswith("negative_peaks"):
                    # Use the last two samples, like the live version does
                    start_index = i * samples_per_summary - 2 if i > 0 else 0
                    n_peaks = output_type.split("[", 1)[1].split("]", 1)[0]
                    n_peaks = int(n_peaks) if n_peaks else 0
                    minima = (
                        local_minima(
                            input_array[start_index:end_index, j],
                        )[:n_peaks]
                        + start_index
                        + peak_index_offset  # Offset for sliced array indexing
                    )
                    output_start_index = (
                        j * summary_size_per_state + summary_index
                    )
                    output_array[
                        i,
                        output_start_index : output_start_index + minima.size,
                    ] = minima
                    summary_index += n_peaks

                if output_type == "mean_std_rms":
                    _mean = np.mean(
                        input_array[start_index:end_index, j], axis=0
                    )
                    _std = np.std(
                        input_array[start_index:end_index, j], axis=0
                    )
                    _rms = np.sqrt(
                        np.mean(
                            input_array[start_index:end_index, j] ** 2, axis=0
                        )
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _mean
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _std
                    output_array[
                        i, j * summary_size_per_state + summary_index + 2
                    ] = _rms
                    summary_index += 3

                if output_type == "mean_std":
                    _mean = np.mean(
                        input_array[start_index:end_index, j], axis=0
                    )
                    _std = np.std(
                        input_array[start_index:end_index, j], axis=0
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _mean
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _std
                    summary_index += 2

                if output_type == "std_rms":
                    _std = np.std(
                        input_array[start_index:end_index, j], axis=0
                    )
                    _rms = np.sqrt(
                        np.mean(
                            input_array[start_index:end_index, j] ** 2, axis=0
                        )
                    )
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _std
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _rms
                    summary_index += 2

                if output_type == "dxdt_max":
                    # Get sample before to simulate continuity
                    start_index = i * samples_per_summary - 1 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 1:
                        derivatives = np.diff(values) / sample_summaries_every
                        _dxdt_max = np.max(derivatives)
                    else:
                        _dxdt_max = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _dxdt_max
                    summary_index += 1

                if output_type == "dxdt_min":
                    # Get sample before to simulate continuity
                    start_index = i * samples_per_summary - 1 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 1:
                        derivatives = np.diff(values) / sample_summaries_every
                        _dxdt_min = np.min(derivatives)
                    else:
                        _dxdt_min = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _dxdt_min
                    summary_index += 1

                if output_type == "dxdt_extrema":
                    # Get sample before to simulate continuity
                    start_index = i * samples_per_summary - 1 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 1:
                        derivatives = np.diff(values) / sample_summaries_every
                        _dxdt_max = np.max(derivatives)
                        _dxdt_min = np.min(derivatives)
                    else:
                        _dxdt_max = 0.0
                        _dxdt_min = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _dxdt_max
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _dxdt_min
                    summary_index += 2

                if output_type == "d2xdt2_max":
                    # Get two samples before to simulate buffer continuity
                    start_index = i * samples_per_summary - 2 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 2:
                        dt_sq = sample_summaries_every * sample_summaries_every
                        # Vectorized calculation matching np.diff
                        v2 = values[2:]
                        v1 = values[1:-1]
                        v0 = values[:-2]
                        second_derivatives = (v2 - 2.0 * v1 + v0) / dt_sq
                        _d2xdt2_max = np.max(second_derivatives)
                    else:
                        _d2xdt2_max = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _d2xdt2_max
                    summary_index += 1

                if output_type == "d2xdt2_min":
                    # Get two samples before to simulate buffer continuity
                    start_index = i * samples_per_summary - 2 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 2:
                        dt_sq = sample_summaries_every * sample_summaries_every
                        # Vectorized calculation matching np.diff
                        v2 = values[2:]
                        v1 = values[1:-1]
                        v0 = values[:-2]
                        second_derivatives = (v2 - 2.0 * v1 + v0) / dt_sq
                        _d2xdt2_min = np.min(second_derivatives)
                    else:
                        _d2xdt2_min = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _d2xdt2_min
                    summary_index += 1

                if output_type == "d2xdt2_extrema":
                    # Get two samples before to simulate buffer continuity
                    start_index = i * samples_per_summary - 2 if i > 0 else 0

                    values = input_array[start_index:end_index, j]
                    if len(values) > 2:
                        dt_sq = sample_summaries_every * sample_summaries_every
                        # Vectorized calculation matching np.diff
                        v2 = values[2:]
                        v1 = values[1:-1]
                        v0 = values[:-2]
                        second_derivatives = (v2 - 2.0 * v1 + v0) / dt_sq
                        _d2xdt2_max = np.max(second_derivatives)
                        _d2xdt2_min = np.min(second_derivatives)
                    else:
                        _d2xdt2_max = 0.0
                        _d2xdt2_min = 0.0
                    output_array[
                        i, j * summary_size_per_state + summary_index
                    ] = _d2xdt2_max
                    output_array[
                        i, j * summary_size_per_state + summary_index + 1
                    ] = _d2xdt2_min
                    summary_index += 2


def local_maxima(signal: np.ndarray) -> np.ndarray:
    """Find local maxima in a signal.

    Returns indices of local maxima. The +1 offset corrects for the
    signal[1:-1] slicing used in the comparison (flatnonzero returns
    indices into the sliced array, not the original signal).
    """
    return (
        np.flatnonzero(
            (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
        )
        + 1  # Correct for signal[1:-1] indexing offset
    )


def local_minima(signal: np.ndarray) -> np.ndarray:
    """Find local minima in a signal.

    Returns indices of local minima. The +1 offset corrects for the
    signal[1:-1] slicing used in the comparison (flatnonzero returns
    indices into the sliced array, not the original signal).
    """
    return (
        np.flatnonzero(
            (signal[1:-1] < signal[:-2]) & (signal[1:-1] < signal[2:])
        )
        + 1  # Correct for signal[1:-1] indexing offset
    )


def deterministic_array(precision, size: Union[int, tuple[int]], scale=1.0):
    """Generate a deterministic array of numerically challenging values.

    Creates reproducible test arrays with values spanning multiple orders
    of magnitude, including edge cases like near-zero values, large values,
    and mathematically interesting constants (π, e).

    Parameters
    ----------
    precision : numpy.dtype
        The desired data type of the array (np.float32 or np.float64).
    size : int or tuple of int
        The shape of the array to generate.
    scale : float, int, list, or tuple, optional
        Guidance for value magnitudes. Default is 1.0.
        - Single number: Values centered around that magnitude
        - Tuple/list of two numbers: Interpreted as (min_exp, max_exp)
          for values spanning 10^min_exp to 10^max_exp

    Returns
    -------
    numpy.ndarray
        A deterministic array of the specified shape and dtype filled
        with numerically challenging values.

    Notes
    -----
    The generated values include:
    - Very small positive values (1e-12, 1e-9, 1e-6, 1e-3)
    - Values near unity (0.1, 0.5, 1.0, 2.0)
    - Mathematical constants (π, e)
    - Large values (1e3, 1e6, 1e9, 1e12)
    - Alternating signs for additional coverage

    Values are tiled/broadcast to fill the requested shape and filtered
    based on the scale parameter to stay within appropriate ranges.
    """
    # Handle empty arrays
    if isinstance(size, int):
        shape = (size,)
    else:
        shape = tuple(size)
    total_elements = int(np.prod(shape))

    if total_elements == 0:
        return np.empty(shape, dtype=precision)

    # Interpret scale parameter
    if isinstance(scale, (list, tuple)) and len(scale) == 2:
        min_exp, max_exp = scale
    else:
        # Single scale value: create range centered around it
        if isinstance(scale, (list, tuple)):
            scale = scale[0]
        scale_exp = math.log10(abs(scale)) if scale != 0 else 0
        min_exp = scale_exp - 6
        max_exp = scale_exp + 6

    # Base set of challenging values (positive)
    base_values = [
        1e-12,
        1e-9,
        1e-6,
        1e-3,
        0.1,
        0.5,
        1.0,
        2.0,
        math.pi,
        math.e,
        1e3,
        1e6,
        1e9,
        1e12,
    ]

    # Filter values to be within the scale range
    filtered_values = []
    for v in base_values:
        v_exp = math.log10(v)
        if min_exp <= v_exp <= max_exp:
            filtered_values.append(v)

    # Ensure we have at least some values
    if not filtered_values:
        # Use scale-appropriate values if filter removed everything
        mid_exp = (min_exp + max_exp) / 2
        filtered_values = [
            10**min_exp,
            10 ** ((min_exp + mid_exp) / 2),
            10**mid_exp,
            10 ** ((mid_exp + max_exp) / 2),
            10**max_exp,
        ]

    # Create array with alternating signs
    values_with_signs = []
    for i, v in enumerate(filtered_values):
        sign = 1 if i % 2 == 0 else -1
        values_with_signs.append(sign * v)

    # Tile values to fill the requested size
    num_base = len(values_with_signs)
    result = np.empty(total_elements, dtype=precision)
    for i in range(total_elements):
        result[i] = values_with_signs[i % num_base]

    return result.reshape(shape)


# ******************** Device Test Kernels *********************************  #
@attrs.define
class LoopRunResult:
    """Container holding the outputs produced by a single loop execution."""

    state: Array
    observables: Array
    state_summaries: Array
    observable_summaries: Array
    status: int
    counters: Array = None


def run_device_loop(
    singleintegratorrun: SingleIntegratorRun,
    system: BaseODE,
    initial_state: Array,
    solver_config: Mapping[str, float],
    driver_array: Optional[ArrayInterpolator] = None,
) -> LoopRunResult:
    """Execute ``loop`` on the CUDA simulator and return host-side outputs."""

    precision = system.precision
    warmup = solver_config["warmup"]
    duration = solver_config["duration"]
    t0 = solver_config["t0"]
    save_samples = max(singleintegratorrun.output_length(duration), 1)
    summary_samples = max(singleintegratorrun.summaries_length(duration), 1)
    singleintegratorrun.set_summary_timing_from_duration(duration)
    heights = singleintegratorrun.output_array_heights

    state_width = max(heights.state, 1)
    observable_width = max(heights.observables, 1)
    state_summary_width = max(heights.state_summaries, 1)
    observable_summary_width = max(heights.observable_summaries, 1)

    state_output = np.zeros((save_samples, state_width), dtype=precision)
    observables_output = np.zeros(
        (save_samples, observable_width), dtype=precision
    )

    state_summary_output = np.zeros(
        (summary_samples, state_summary_width), dtype=precision
    )
    observable_summary_output = np.zeros(
        (summary_samples, observable_summary_width), dtype=precision
    )

    # Iteration counters output (4 counters per save)
    counters_output = np.zeros((save_samples, 4), dtype=np.int32)

    params = np.array(
        system.parameters.values_array,
        dtype=precision,
        copy=True,
    )
    init_state = np.array(initial_state, dtype=precision, copy=True)
    status = np.zeros(1, dtype=np.int32)

    d_init = cuda.to_device(init_state)
    d_params = cuda.to_device(params)
    if driver_array is None:
        order = int(solver_config["driverspline_order"])
        width = min(system.num_drivers, 1)
        coeff_shape = (1, width, order + 1)
        driver_coefficients = np.zeros(coeff_shape, dtype=precision)
    else:
        driver_coefficients = np.array(
            driver_array.coefficients, dtype=precision, copy=True
        )
    d_driver_coeffs = cuda.to_device(driver_coefficients)
    d_state_out = cuda.to_device(state_output)
    d_obs_out = cuda.to_device(observables_output)
    d_state_sum = cuda.to_device(state_summary_output)
    d_obs_sum = cuda.to_device(observable_summary_output)
    d_counters_out = cuda.to_device(counters_output)
    d_status = cuda.to_device(status)

    # Build before sizing: the build refreshes nested child buffer
    # sizes in the loop's group, mirroring BatchSolverKernel.run().
    loop_fn = singleintegratorrun.device_function

    shared_bytes = max(4, singleintegratorrun.shared_memory_bytes)
    shared_elements = max(1, singleintegratorrun.shared_memory_elements)
    persistent_required = max(1, singleintegratorrun.persistent_local_elements)

    numba_precision = from_dtype(precision)
    save_stop = precision(
        singleintegratorrun.save_stop_time(duration, warmup, t0)
    )
    summary_stop = precision(
        singleintegratorrun.summary_stop_time(duration, warmup, t0)
    )

    @cuda.jit(
        # (
        #     numba_precision[::1],
        #     numba_precision[::1],
        #     numba_precision[:,:,::1],
        #     numba_precision[:,::1],
        #     numba_precision[:,::1],
        #     numba_precision[:,::1],
        #     numba_precision[:,::1],
        #     numba_precision[:,::1],
        #     numba_precision[::1]
        # )
    )
    def kernel(
        init_vec,
        params_vec,
        driver_coeffs_vec,
        state_out_arr,
        obs_out_arr,
        state_sum_arr,
        obs_sum_arr,
        counters_out_arr,
        status_arr,
    ):
        idx = cuda.grid(1)
        if idx > 0:
            return

        shared = cuda.shared.array(shared_elements, dtype=numba_precision)
        shared[:] = numba_precision(0.0)
        local = cuda.local.array(persistent_required, dtype=numba_precision)
        local[:] = numba_precision(0.0)
        status_arr[0] = loop_fn(
            init_vec,
            params_vec,
            driver_coeffs_vec,
            shared,
            local,
            state_out_arr,
            obs_out_arr,
            state_sum_arr,
            obs_sum_arr,
            counters_out_arr,
            duration,
            warmup,
            t0,
            save_stop,
            summary_stop,
        )

    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream, shared_bytes](
        d_init,
        d_params,
        d_driver_coeffs,
        d_state_out,
        d_obs_out,
        d_state_sum,
        d_obs_sum,
        d_counters_out,
        d_status,
    )
    stream.synchronize()

    state_host = d_state_out.copy_to_host()
    observables_host = d_obs_out.copy_to_host()
    state_summary_host = d_state_sum.copy_to_host()
    observable_summary_host = d_obs_sum.copy_to_host()
    counters_host = d_counters_out.copy_to_host()
    status_value = int(d_status.copy_to_host()[0])

    return LoopRunResult(
        state=state_host,
        observables=observables_host,
        state_summaries=state_summary_host,
        observable_summaries=observable_summary_host,
        counters=counters_host,
        status=status_value,
    )


def assert_integration_outputs(
    reference,
    device,
    output_functions,
    rtol: float,
    atol: float,
) -> None:
    """Compare state, summary, and time outputs between CPU and device."""
    if isinstance(reference, dict):
        reference = LoopRunResult(**reference)
    flags = output_functions.compile_flags
    if device.counters is None:
        print("\nNo counters provided")
    else:
        print(device.counters)
    state_ref, time_ref = extract_state_and_time(
        reference.state, output_functions
    )
    state_dev, time_dev = extract_state_and_time(
        device.state,
        output_functions,
    )
    observables_ref = reference.observables
    observables_dev = device.observables

    if output_functions.save_time:
        assert_allclose(
            time_dev,
            time_ref,
            rtol=rtol,
            atol=atol,
            err_msg="time mismatch.\n"
            f"device: {time_dev}\nreference: {time_ref}",
        )

    if flags.save_state:
        assert_allclose(
            state_dev,
            state_ref,
            rtol=rtol,
            atol=atol,
            verbose=True,
            err_msg="state mismatch.\n"
            f"device: {state_dev}\nreference: {state_ref}\ndelta (ref - "
            f"dev): {state_ref - state_dev}\n",
        )

    if flags.save_observables:
        assert_allclose(
            observables_dev,
            observables_ref,
            rtol=rtol,
            atol=atol,
            err_msg="observables mismatch.\n"
            f"device: {observables_dev}\n"
            f"reference: {observables_ref}",
        )

    if flags.summarise_state:
        assert_summaries_close(
            device.state_summaries,
            reference.state_summaries,
            samples=state_ref,
            output_functions=output_functions,
            rtol=rtol,
            atol=atol,
            label="state summaries",
        )

    if flags.summarise_observables:
        assert_summaries_close(
            device.observable_summaries,
            reference.observable_summaries,
            samples=observables_ref,
            output_functions=output_functions,
            rtol=rtol,
            atol=atol,
            label="observable summaries",
        )


def summary_atol_per_column(
    output_functions: OutputFunctions,
    amplitude: float,
    base_atol: float,
    eps: float,
) -> Array:
    """Per-column absolute tolerances for a summaries array.

    Derivative metrics amplify sample-level rounding noise: a
    perturbation of one ulp in one sample shifts a first difference by
    up to 2*eps*amplitude and a second difference (stencil coefficients
    1, -2, 1) by up to 4*eps*amplitude, and the metrics scale by
    1/sample_summaries_every and 1/sample_summaries_every**2
    respectively. Columns belonging to dxdt/d2xdt2 metrics therefore
    get ``base_atol`` plus that amplification floor; all other columns
    keep ``base_atol``.

    Parameters
    ----------
    output_functions
        Configured output functions; supplies the per-variable summary
        legend and ``sample_summaries_every``.
    amplitude
        Maximum absolute sample value feeding the summaries.
    base_atol
        Absolute tolerance for non-derivative columns.
    eps
        Machine epsilon of the comparison precision.

    Returns
    -------
    Array
        Absolute tolerance for each per-variable summary column.
    """
    legend = output_functions.summary_legend_per_variable
    sample_every = float(
        output_functions.compile_settings.sample_summaries_every
    )
    tolerances = []
    for index in range(len(legend)):
        name = legend[index]
        if name.startswith("d2xdt2"):
            tolerance = base_atol + (
                4.0 * eps * amplitude / (sample_every * sample_every)
            )
        elif name.startswith("dxdt"):
            tolerance = base_atol + 2.0 * eps * amplitude / sample_every
        else:
            tolerance = base_atol
        tolerances.append(tolerance)
    return np.asarray(tolerances)


def assert_summaries_close(
    device_summaries,
    reference_summaries,
    samples,
    output_functions: OutputFunctions,
    rtol: float,
    atol: float,
    label: str,
) -> None:
    """Compare summary arrays with metric-type-scaled tolerances.

    The last axis of the summary arrays is laid out as
    ``variable_index * per_variable_height + metric_column``; the
    per-variable tolerance vector from :func:`summary_atol_per_column`
    is tiled across variables and applied elementwise as
    ``|device - reference| <= atol_column + rtol * |reference|``.

    Parameters
    ----------
    device_summaries
        Summary array produced on the device.
    reference_summaries
        Summary array produced by the CPU reference.
    samples
        Sampled values the summaries were computed from; supplies the
        amplitude and the machine epsilon for the derivative-metric
        tolerance floor. ``None`` or empty falls back to an amplitude
        of 1.0 and the summary dtype's epsilon.
    output_functions
        Configured output functions for legend and timing metadata.
    rtol
        Relative tolerance applied against the reference magnitude.
    atol
        Base absolute tolerance for non-derivative columns.
    label
        Array description used in the failure message.
    """
    device_summaries = np.asarray(device_summaries)
    reference_summaries = np.asarray(reference_summaries)
    samples = np.asarray(samples) if samples is not None else None
    if samples is not None and samples.size > 0:
        # The ulp floor absorbs sample-level rounding differences, so
        # eps belongs to the sample precision, not the summary dtype.
        amplitude = float(np.max(np.abs(samples)))
        eps = float(np.finfo(samples.dtype).eps)
    else:
        amplitude = 1.0
        eps = float(np.finfo(reference_summaries.dtype).eps)
    per_variable = summary_atol_per_column(
        output_functions, amplitude, atol, eps
    )
    n_columns = device_summaries.shape[-1]
    if n_columns % len(per_variable) != 0:
        raise AssertionError(
            f"{label}: last axis ({n_columns}) is not a multiple of the "
            f"per-variable summary height ({len(per_variable)})"
        )
    atol_columns = np.tile(per_variable, n_columns // len(per_variable))
    difference = np.abs(
        device_summaries.astype(np.float64)
        - reference_summaries.astype(np.float64)
    )
    allowed = atol_columns + rtol * np.abs(
        reference_summaries.astype(np.float64)
    )
    device_nan = np.isnan(device_summaries)
    reference_nan = np.isnan(reference_summaries)
    both_nan = device_nan & reference_nan
    # One-sided NaN makes difference NaN, and NaN > allowed is False,
    # so it must be flagged explicitly rather than fall through.
    nan_mismatch = device_nan ^ reference_nan
    violations = ((difference > allowed) & ~both_nan) | nan_mismatch
    assert not np.any(violations), (
        f"{label} mismatch at {np.argwhere(violations).tolist()}.\n"
        f"device: {device_summaries}\n"
        f"reference: {reference_summaries}\n"
        f"difference: {difference}\n"
        f"allowed: {np.broadcast_to(allowed, difference.shape)}"
    )


def extract_state_and_time(
    state_output: Array, output_functions: OutputFunctions
) -> tuple[Array, Optional[Array]]:
    """Split state output into state variables and optional time column."""
    n_state_columns = output_functions.n_saved_states
    if not output_functions.save_time:
        return state_output, None
    if state_output.ndim == 2:
        state_values = state_output[:, :n_state_columns]
        time_values = state_output[:, n_state_columns : n_state_columns + 1]
    else:
        state_values = state_output[:, :, :n_state_columns]
        time_values = state_output[:, :, n_state_columns:]

    return state_values, time_values


def _driver_sequence(
    *,
    samples: int,
    total_time: float,
    n_drivers: int,
    precision,
) -> Array:
    """Drive system with a sine wave."""

    width = max(n_drivers, 1)
    drivers = np.zeros((samples, width), dtype=precision)
    if n_drivers > 0 and total_time > 0.0:
        times = np.linspace(0.0, total_time, samples, dtype=precision)
        for idx in range(n_drivers):
            drivers[:, idx] = precision(
                1.0 + np.sin(2 * np.pi * (idx + 1) * times / total_time)
            )
    return drivers


def _build_enhanced_algorithm_settings(
    algorithm_settings, system, driver_array
):
    """Add system and driver functions to algorithm settings.

    Functions are passed directly to get_algorithm_step, not stored
    in algorithm_settings dict.
    """
    enhanced = algorithm_settings.copy()
    enhanced["evaluate_f"] = system.evaluate_f
    enhanced["evaluate_observables"] = system.evaluate_observables
    enhanced["get_solver_helper_fn"] = system.get_solver_helper
    enhanced["n_drivers"] = system.num_drivers

    if driver_array is not None:
        enhanced["evaluate_driver_at_t"] = driver_array.evaluation_function
        enhanced["driver_del_t"] = driver_array.driver_del_t
    else:
        enhanced["evaluate_driver_at_t"] = None
        enhanced["driver_del_t"] = None

    return enhanced


# Keys in the shared solver_settings dict that are not Solver
# constructor settings: solve-time arguments, system-construction
# options (fix_singularities/voltage_variable feed
# load_cellml_model), driver-interpolation settings (driverspline_*
# are read by the conftest driver fixtures to configure the
# ArrayInterpolator), and test-harness metadata. Solver rejects
# unconsumed kwargs, so these are stripped before construction.
NON_SOLVER_SETTINGS = {
    "duration",
    "warmup",
    "t0",
    "blocksize",
    "system_type",
    "n_states",
    "n_parameters",
    "n_observables",
    "fix_singularities",
    "voltage_variable",
    "driverspline_order",
    "driverspline_wrap",
    "driverspline_boundary_condition",
}


def _build_solver_instance(
    system: SymbolicODE,
    solver_settings: Dict[str, Any],
    driver_settings: Optional[Dict[str, Any]],
    memory_manager: Optional[Any] = None,
) -> Solver:
    """Instantiate :class:`Solver` configured with ``solver_settings``."""
    settings = {
        key: value
        for key, value in solver_settings.items()
        if key not in NON_SOLVER_SETTINGS
    }
    if memory_manager:
        settings.update(memory_manager=memory_manager)
    solver = Solver(system, **settings)
    if driver_settings is not None:
        solver._configure_drivers(driver_settings)
    return solver


def _resolved_controller_gains(controller) -> Dict[str, float]:
    """Return a built controller's gains by name."""
    return {
        name: float(getattr(controller, name))
        for name in controller.gain_names
    }


def _build_cpu_step_controller(
    precision: np.dtype,
    step_controller_settings: Dict[str, Any],
    gains: Optional[Dict[str, float]] = None,
) -> CPUAdaptiveController:
    """Return a CPU controller from settings, ``gains`` overriding."""

    step_controller_settings = dict(step_controller_settings)
    if "filter_coefficients" in step_controller_settings:
        step_controller_settings.update(
            filter_coefficients_to_gains(
                step_controller_settings.pop("filter_coefficients")
            )
        )
    if gains is not None:
        step_controller_settings.update(gains)
    kind = step_controller_settings["step_controller"].lower()
    controller = CPUAdaptiveController(
        kind=kind,
        dt=step_controller_settings["dt"],
        dt_min=step_controller_settings["dt_min"],
        dt_max=step_controller_settings["dt_max"],
        atol=step_controller_settings["atol"],
        rtol=step_controller_settings["rtol"],
        order=step_controller_settings["algorithm_order"],
        min_step_shrink=step_controller_settings["min_step_shrink"],
        max_step_growth=step_controller_settings["max_step_growth"],
        precision=precision,
        deadband_min=step_controller_settings["deadband_min"],
        deadband_max=step_controller_settings["deadband_max"],
        safety=step_controller_settings["safety"],
        newton_target_iters=step_controller_settings[
            "newton_target_iters"
        ],
    )
    order = step_controller_settings["algorithm_order"]

    def resolve_gain(spec):
        # Gains arrive as floats or callables of the algorithm order.
        if callable(spec):
            return float(spec(order))
        return float(spec)

    config_class = _CONTROLLER_REGISTRY[kind]._config_class
    declared = attrs.fields_dict(config_class)
    for name in CONTROLLER_GAIN_NAMES:
        if f"_{name}" not in declared:
            continue
        spec = step_controller_settings.get(name)
        if spec is None:
            spec = declared[f"_{name}"].default
        setattr(controller, name, resolve_gain(spec))
    return controller


def _get_algorithm_order(
    algorithm_name_or_tableau, use_smoothed_error=False
):
    """Return the step-control order, mirroring ``controller_order``."""
    from cubie.integrators.algorithms import (
        resolve_alias,
        resolve_supplied_tableau,
    )
    from cubie.integrators.algorithms.generic_rosenbrock_w import (
        GenericRosenbrockWStep,
        DEFAULT_ROSENBROCK_TABLEAU,
    )

    if isinstance(algorithm_name_or_tableau, str):
        algorithm_type, tableau = resolve_alias(algorithm_name_or_tableau)
    else:
        algorithm_type, tableau = resolve_supplied_tableau(
            algorithm_name_or_tableau
        )

    # For rosenbrock without explicit tableau, use default
    if algorithm_type is GenericRosenbrockWStep and tableau is None:
        tableau = DEFAULT_ROSENBROCK_TABLEAU

    # Extract order from tableau if available
    if tableau is not None and hasattr(tableau, "order"):
        if use_smoothed_error and getattr(
            tableau, "supports_smoothed_error", False
        ):
            smoothed = getattr(tableau, "smoothed_embedded_order", None)
            if smoothed is not None:
                return min(tableau.order, smoothed)
        if tableau.embedded_order is not None:
            return min(tableau.order, tableau.embedded_order)
        return tableau.order

    # Default orders for algorithms without tableaus
    defaults = {
        "euler": 1,
        "backwards_euler": 1,
        "backwards_euler_pc": 1,
        "crank_nicolson": 2,
    }

    if isinstance(algorithm_name_or_tableau, str):
        algorithm_name = algorithm_name_or_tableau.lower()
        return defaults.get(algorithm_name, 1)

    return 1


def _get_algorithm_tableau(algorithm_name_or_tableau):
    """Get tableau for an algorithm without building step object.

    Parameters
    ----------
    algorithm_name_or_tableau : str or ButcherTableau
        Algorithm identifier or tableau instance.

    Returns
    -------
    tableau or None
        The tableau if available, None otherwise.
    """
    from cubie.integrators.algorithms import (
        resolve_alias,
        resolve_supplied_tableau,
    )
    from cubie.integrators.algorithms.generic_rosenbrock_w import (
        GenericRosenbrockWStep,
        DEFAULT_ROSENBROCK_TABLEAU,
    )

    if isinstance(algorithm_name_or_tableau, str):
        algorithm_type, tableau = resolve_alias(algorithm_name_or_tableau)
    else:
        algorithm_type, tableau = resolve_supplied_tableau(
            algorithm_name_or_tableau
        )

    # For rosenbrock without explicit tableau, use default
    if algorithm_type is GenericRosenbrockWStep and tableau is None:
        tableau = DEFAULT_ROSENBROCK_TABLEAU

    return tableau


def _get_evaluate_driver_at_t(
    driver_array: Optional[ArrayInterpolator],
) -> Optional[Callable[..., Any]]:
    """Return the evaluation callable for ``driver_array`` if it exists."""
    if driver_array is None:
        return None
    return driver_array.evaluation_function


def _get_driver_del_t(
    driver_array: Optional[ArrayInterpolator],
) -> Optional[Callable[..., Any]]:
    """Return the time-derivative evaluation callable for ``driver_array``."""

    if driver_array is None:
        return None
    return driver_array.driver_del_t


def make_slice_fn(run_axis_idx, chunk_size, ndim):
    """Create a slice function for chunked array access.

    Returns a callable that generates index tuples to extract a chunk from
    an array, slicing the run axis while preserving other dimensions.

    Parameters
    ----------
    run_axis_idx : int
        Index of the run axis in the array's shape.
    chunk_size : int
        Number of runs per chunk.
    ndim : int
        Number of dimensions in the array.

    Returns
    -------
    callable
        A function that takes a chunk index and returns a tuple of slices.
    """

    def slice_fn(chunk_idx):
        slices = [slice(None)] * ndim
        start = chunk_idx * chunk_size
        end = start + chunk_size
        slices[run_axis_idx] = slice(start, end)
        return tuple(slices)

    return slice_fn


def setup_chunked_arrays(manager, num_runs, num_chunks):
    """Configure chunked_shape and chunked_slice_fn on array manager slots.

    Sets up both host and device slots in the manager for chunked transfers.
    Arrays with 'run' in their stride_order get chunked shapes; others are
    left unchanged.

    Parameters
    ----------
    manager : InputArrays or OutputArrays
        The array manager with host and device containers.
    num_runs : int
        Total number of runs across all chunks.
    num_chunks : int
        Number of chunks to split runs into.
    """
    chunk_size = max(1, num_runs // num_chunks)

    for name, device_slot in manager.device.iter_managed_arrays():
        if "run" in device_slot.stride_order:
            run_idx = device_slot.stride_order.index("run")
            chunked = list(device_slot.shape)
            chunked[run_idx] = chunk_size
            chunked_shape = tuple(chunked)
            ndim = len(device_slot.shape)
            slice_fn = make_slice_fn(run_idx, chunk_size, ndim)
            device_slot.chunked_shape = chunked_shape
            device_slot.chunked_slice_fn = slice_fn
            # Also configure corresponding host array
            host_slot = manager.host.get_managed_array(name)
            host_slot.chunked_shape = chunked_shape
            host_slot.chunked_slice_fn = slice_fn


# --------------------------------------------------------------------------- #
#                  Memoised single-purpose device harnesses                   #
#                                                                             #
# Each factory below is keyed on the device function it wraps, so a wrapper   #
# kernel is compiled once per device function rather than once per call site  #
# or per parametrised case.                                                   #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _dxdt_kernel(device_fn):
    @cuda.jit()
    def kernel(state, params, drivers, obs, out, t):
        device_fn(state, params, drivers, obs, out, t)

    return kernel


def run_device_dxdt(device_fn, state, params, drivers, obs, out, t):
    """Launch a dxdt device function through a single-thread kernel."""
    stream = default_memmgr.get_group_stream()
    _dxdt_kernel(device_fn)[1, 1, stream](
        state, params, drivers, obs, out, t
    )
    stream.synchronize()


@lru_cache(maxsize=None)
def _observables_kernel(device_fn):
    @cuda.jit()
    def kernel(state, params, drivers, obs, t):
        device_fn(state, params, drivers, obs, t)

    return kernel


def run_device_observables(device_fn, state, params, drivers, obs, t):
    """Launch an observables device function via a kernel."""
    stream = default_memmgr.get_group_stream()
    _observables_kernel(device_fn)[1, 1, stream](
        state, params, drivers, obs, t
    )
    stream.synchronize()


@lru_cache(maxsize=None)
def _driver_eval_kernel(device_fn):
    @cuda.jit()
    def kernel(times, coeffs, out):
        idx = cuda.grid(1)
        if idx < times.size:
            device_fn(times[idx], coeffs, out[idx])

    return kernel


def run_driver_device_eval(device_fn, coefficients, query_times):
    """Evaluate a driver interpolation device function on the GPU.

    Parameters
    ----------
    device_fn : callable
        Driver evaluation or derivative device function taking
        ``(time, coefficients, out_row)``.
    coefficients : numpy.ndarray
        Segment-major polynomial coefficients.
    query_times : numpy.ndarray
        Time samples to evaluate on the device.

    Returns
    -------
    numpy.ndarray
        Evaluated input values, one row per query time.
    """
    n_times = query_times.size
    n_inputs = coefficients.shape[1]
    # Zero-filled: headless population runs never launch the kernel.
    out_host = np.zeros((n_times, n_inputs), dtype=coefficients.dtype)

    d_times = cuda.to_device(query_times)
    d_coeffs = cuda.to_device(coefficients)
    d_out = cuda.to_device(out_host)
    threads_per_block = 64
    blocks = (n_times + threads_per_block - 1) // threads_per_block
    stream = default_memmgr.get_group_stream()
    _driver_eval_kernel(device_fn)[blocks, threads_per_block, stream](
        d_times, d_coeffs, d_out
    )
    stream.synchronize()
    d_out.copy_to_host(out_host)
    return out_host


class StepResult:
    """Lightweight return container mirroring GPU kernel outputs."""

    def __init__(self, dt, accepted, local_mem, status=None):
        self.dt = dt
        self.accepted = accepted
        self.local_mem = local_mem
        self.status = status


@lru_cache(maxsize=None)
def _controller_step_kernel(device_func):
    @cuda.jit
    def kernel(
        dt_val,
        state_val,
        state_prev_val,
        err_val,
        niters_val,
        truncated_flag,
        accept_val,
        shared_val,
        persistent_val,
        status_val,
    ):
        status_val[0] = device_func(
            dt_val,
            state_val,
            state_prev_val,
            err_val,
            niters_val,
            truncated_flag,
            accept_val,
            shared_val,
            persistent_val,
        )

    return kernel


def run_controller_device_step(
    device_func,
    precision,
    dt0,
    error,
    *,
    local_mem=None,
    state=None,
    state_prev=None,
    niters=1,
    truncated=False,
):
    """Execute a step-controller device function once on the GPU."""

    err = np.asarray(error, dtype=precision)
    state_arr = (
        np.asarray(state, dtype=precision)
        if state is not None
        else np.zeros_like(err)
    )
    state_prev_arr = (
        np.asarray(state_prev, dtype=precision)
        if state_prev is not None
        else np.zeros_like(err)
    )

    dt = np.asarray([dt0], dtype=precision)
    accept = np.zeros(1, dtype=np.int32)
    status = np.zeros(1, dtype=np.int32)
    niters_val = np.int32(niters)
    truncated_val = bool(truncated)
    shared_scratch = np.zeros(1, dtype=precision)
    if local_mem is not None:
        persistent_local = np.asarray(local_mem, dtype=precision)
    else:
        persistent_local = np.zeros(2, dtype=precision)

    kernel = _controller_step_kernel(device_func)
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](
        dt, state_arr, state_prev_arr, err, niters_val, truncated_val,
        accept, shared_scratch, persistent_local, status,
    )
    stream.synchronize()
    return StepResult(
        precision(dt[0]), int(accept[0]), persistent_local.copy(),
        int(status[0]),
    )


@lru_cache(maxsize=None)
def _step_schedule_kernel(
    step_fn, n, n_obs, n_drv, persistent_len, numba_precision
):
    @cuda.jit
    def kernel(state_io, params_vec, driver_coeffs, dt_schedule,
               out_iters, status_vec):
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(
            persistent_len, dtype=numba_precision
        )
        state_vec = cuda.local.array(n, dtype=numba_precision)
        proposed = cuda.local.array(n, dtype=numba_precision)
        error = cuda.local.array(n, dtype=numba_precision)
        drivers = cuda.local.array(n_drv, dtype=numba_precision)
        proposed_drivers = cuda.local.array(
            n_drv, dtype=numba_precision
        )
        observables = cuda.local.array(n_obs, dtype=numba_precision)
        proposed_observables = cuda.local.array(
            n_obs, dtype=numba_precision
        )
        counters = cuda.local.array(2, dtype=int32)
        for i in range(persistent_len):
            persistent[i] = numba_precision(0.0)
        for i in range(n):
            state_vec[i] = state_io[i]
            proposed[i] = numba_precision(0.0)
            error[i] = numba_precision(0.0)
        for i in range(n_drv):
            drivers[i] = numba_precision(0.0)
            proposed_drivers[i] = numba_precision(0.0)
        for i in range(n_obs):
            observables[i] = numba_precision(0.0)
            proposed_observables[i] = numba_precision(0.0)
        counters[0] = 0
        counters[1] = 0
        time_value = numba_precision(0.0)
        status = int32(0)
        n_steps = dt_schedule.shape[0]
        for step_index in range(n_steps):
            if step_index == n_steps - 1:
                counters[0] = 0
                counters[1] = 0
            first_flag = int32(1) if step_index == 0 else int32(0)
            result = step_fn(
                state_vec,
                proposed,
                params_vec,
                driver_coeffs,
                drivers,
                proposed_drivers,
                observables,
                proposed_observables,
                error,
                dt_schedule[step_index],
                time_value,
                first_flag,
                int32(1),
                shared,
                persistent,
                counters,
            )
            status = int32(status | result)
            time_value += dt_schedule[step_index]
            for i in range(n):
                state_vec[i] = proposed[i]
        for i in range(n):
            state_io[i] = state_vec[i]
        out_iters[0] = counters[0]
        status_vec[0] = status

    return kernel


def run_device_step_schedule(
    step_object,
    system,
    precision,
    state,
    params,
    schedule,
    *,
    driver_coefficients=None,
):
    """Run consecutive accepted steps through an algorithm's step.

    Every step is accepted, so the proposed state becomes the next
    step's state. The Newton counters are reset before the last step,
    so the returned iteration count belongs to that step alone.

    Parameters
    ----------
    step_object
        Algorithm step whose ``step_function`` drives the schedule.
    system
        System supplying the state, observable and driver widths.
    precision
        Floating-point type of the device arrays.
    state
        Initial state vector.
    params
        Parameter vector.
    schedule
        Step sizes to take, in order.
    driver_coefficients
        Driver spline coefficients; a zero array when omitted.

    Returns
    -------
    tuple
        Final state and the last step's Newton iteration count.
    """
    numba_precision = from_dtype(precision)
    shared_elems = int(step_object.shared_buffer_size)
    shared_bytes = precision(0).itemsize * max(shared_elems, 1)
    persistent_len = max(
        1, int(step_object.persistent_local_buffer_size)
    )
    kernel = _step_schedule_kernel(
        step_object.step_function,
        int(system.sizes.states),
        max(1, int(system.sizes.observables)),
        max(1, int(system.sizes.drivers)),
        persistent_len,
        numba_precision,
    )

    if driver_coefficients is None:
        coefficients = np.zeros((1, 1, 1), dtype=precision)
    else:
        coefficients = np.asarray(driver_coefficients, dtype=precision)

    d_state = cuda.to_device(np.asarray(state, dtype=precision))
    d_params = cuda.to_device(np.asarray(params, dtype=precision))
    d_coeffs = cuda.to_device(coefficients)
    d_schedule = cuda.to_device(np.asarray(schedule, dtype=precision))
    d_iters = cuda.to_device(np.zeros(1, dtype=np.int32))
    d_status = cuda.to_device(np.zeros(1, dtype=np.int32))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream, shared_bytes](
        d_state, d_params, d_coeffs, d_schedule, d_iters, d_status
    )
    stream.synchronize()
    assert int(d_status.copy_to_host()[0]) == 0
    return d_state.copy_to_host(), int(d_iters.copy_to_host()[0])


@lru_cache(maxsize=None)
def _dense_predictor_kernel(device_fn, persistent_len, numba_precision):
    @cuda.jit
    def kernel(vector, step_ratio, flag):
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(
            persistent_len, dtype=numba_precision
        )
        for i in range(persistent_len):
            persistent[i] = numba_precision(0.0)
        device_fn(vector, step_ratio, flag, shared, persistent)

    return kernel


def run_dense_predictor_step(
    device_fn, vector, step_ratio, flag, precision, persistent_len
):
    """Apply a dense stage predictor once on the GPU.

    Parameters
    ----------
    device_fn : callable
        Compiled predictor device function.
    vector : numpy.ndarray
        Stage-major history vector, transformed in place on device.
    step_ratio : float
        Ratio of the proposed step to the previous one.
    flag : bool
        Commit flag: the transform writes only when it is true.
    precision : type
        Floating-point type of the device arrays.
    persistent_len : int
        Length of the predictor's persistent local buffer.

    Returns
    -------
    numpy.ndarray
        The vector as the predictor left it.
    """
    kernel = _dense_predictor_kernel(
        device_fn, max(1, int(persistent_len)), from_dtype(precision)
    )
    device_vector = cuda.to_device(
        np.array(vector, dtype=precision, copy=True)
    )
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](device_vector, precision(step_ratio), flag)
    stream.synchronize()
    return device_vector.copy_to_host()


# ---- pool sets migrated from per-file definitions ---- #


LARGE_STATE_ONLY = {
    "system_type": "large",
    "output_types": ["state"],
    "saved_observable_indices": [],
    "summarised_observable_indices": [],
}

LARGE_TSIT5 = {**LARGE_STATE_ONLY, "algorithm": "tsit5"}

LARGE_DIRK = {**LARGE_STATE_ONLY, "algorithm": "dirk"}

LARGE_FIRK = {**LARGE_STATE_ONLY, "algorithm": "firk"}

LARGE_BACKWARDS_EULER = {
    **LARGE_STATE_ONLY,
    "algorithm": "backwards_euler",
}

# Unique sets: the final-save schedule is a function of exact
# dt/save_every/duration ratios, so each case pins its own timing.
# The base pins a fixed euler step with time-domain output only.
FIXED_EULER_TIMED_STATE = {
    "summarise_every": None,
    "sample_summaries_every": None,
    "output_types": ["state", "time"],
    "algorithm": "euler",
    "step_controller": "fixed",
}

# One chain for the device-path, spill, proportion and counter tests.
DEVICE_SOLVE_SETTINGS = {
    "duration": 0.05,
    "dt": 0.01,
    "save_every": 0.01,
    "summarise_every": None,
    "output_types": ["state", "time", "iteration_counters"],
    "mem_proportion": 0.1,
    "host_spill_threshold": 512,
}

MOVABLE_LOCATION_KEYS = (
    "state_location",
    "proposed_state_location",
    "parameters_location",
    "drivers_location",
    "proposed_drivers_location",
    "observables_location",
    "proposed_observables_location",
    "error_location",
    "stage_increment_location",
    "stage_base_location",
    "accumulator_location",
    "stage_rhs_location",
)

# Driver-count and ordering checks need a system declaring two
# named drivers; the default chain systems declare one.
# Also carries the disabled singularity fix for the cellml test.
TWO_DRIVER_SYSTEM = {
    "system_type": "two_driver",
    "fix_singularities": False,
}

# The three-state linear system has the constant Jacobian the
# residual and helper-identity checks assume.
LINEAR_SYSTEM = {"system_type": "linear"}

# Transcendental-heavy testbed for the auxiliary-cache planner.
HODGKIN_HUXLEY_SYSTEM = {"system_type": "hodgkin_huxley"}

# The colliding-constants system shadows generated-code symbol
# names; the collision handling must hold at both precisions.
COLLIDING_CONSTANTS_F32 = {
    "system_type": "colliding_constants", "precision": np.float32,
}

COLLIDING_CONSTANTS_F64 = {
    "system_type": "colliding_constants", "precision": np.float64,
}


DIAGONALLY_DOMINANT = {
    "system_type": "diagonally_dominant",
    "precision": np.float64,
}

OFF_DIAGONAL_HEAVY = {
    "system_type": "off_diagonal_heavy",
    "precision": np.float64,
}

GATING_SINGULARITY = {
    "system_type": "gating_singularity",
    "precision": np.float64,
}

SINGULAR_INITIAL_STATE = {
    "system_type": "singular_initial_state",
    "precision": np.float64,
}

LORENZ_ITERATION_BASE = {
    "system_type": "lorenz_julia",
    "output_types": ["state", "iteration_counters"],
    "saved_state_indices": [0, 1, 2],
    "saved_observable_indices": [],
    "summarised_state_indices": [],
    "summarised_observable_indices": [],
    "summarise_every": None,
    "sample_summaries_every": None,
}

RADAU_ADAPTIVE_CASE = {
    **LORENZ_ITERATION_BASE,
    "algorithm": "radau",
    "step_controller": "gustafsson",
    "dt_min": 1e-6,
    "dt_max": 0.02,
    "atol": 1e-6,
    "rtol": 1e-6,
}

# Three non-uniform entries, one per state of the default nonlinear
# system. Unset inner tolerances derive from the controller's.
FIRK_PER_STATE_TOLERANCES = {
    "algorithm": "radau",
    "step_controller": "gustafsson",
    "atol": [1e-7, 1e-6, 1e-5],
    "rtol": [1e-5, 1e-4, 1e-3],
    "krylov_atol": None,
    "krylov_rtol": None,
    "newton_atol": None,
    "newton_rtol": None,
}

DENSE_PREDICTION_ITERATION_CASES = [
    pytest.param(
        {
            **LORENZ_ITERATION_BASE,
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
            **LORENZ_ITERATION_BASE,
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
            **LORENZ_ITERATION_BASE,
            "algorithm": "trapezoidal_dirk",
            "step_controller": "fixed",
            "dt": 0.005,
        },
        id="dirk-explicit-first-stage",
    ),
    pytest.param(RADAU_ADAPTIVE_CASE, id="firk-adaptive"),
]

LORENZ_DIRK = {
    "system_type": "lorenz_julia",
    "output_types": ["state"],
    "saved_state_indices": [0, 1, 2],
    "saved_observable_indices": [],
    "summarised_state_indices": [],
    "summarised_observable_indices": [],
    "summarise_every": None,
    "sample_summaries_every": None,
    "precision": np.float64,
    "algorithm": "l_stable_dirk_3",
    "step_controller": "fixed",
    "dt": 0.005,
    "newton_atol": 1e-10,
    "newton_rtol": 1e-10,
    "krylov_atol": 1e-10,
    "krylov_rtol": 1e-10,
    "newton_max_iters": 50,
    "krylov_max_iters": 100,
}

LOOSE_LORENZ_DIRK = {
    **LORENZ_DIRK,
    "newton_atol": 1e-3,
    "newton_rtol": 1e-3,
    "krylov_atol": 1e-4,
    "krylov_rtol": 1e-4,
}

SAVE_DRIFT = {
    "system_type": "coupled_oscillator",
    "algorithm": "radau",
    "step_controller": "gustafsson",
    "duration": 10.0,
    "dt_min": 1e-6,
    "dt_max": 1.0,
    "save_every": 0.1,
    "output_types": ["state", "time"],
    # The oscillator declares no observables; the shared defaults
    # index two of them.
    "saved_observable_indices": [],
    "summarised_observable_indices": [],
}

DRIFTED_GRID = {
    "algorithm": "euler",
    "step_controller": "fixed",
    "dt": 0.01,
    "duration": 1.0,
    "save_every": 0.1,
    "output_types": ["state", "time"],
}

ROUNDED_DOWN_COUNT = {
    "algorithm": "euler",
    "step_controller": "fixed",
    "dt": 0.0005,
    "duration": 0.01,
    "save_every": 0.001,
    "output_types": ["state", "time"],
}

RECOVERED_TRANSIENT = {
    "system_type": "staining_stiff",
    "precision": np.float64,
    "algorithm": "rodas3p",
    "step_controller": "pid",
    "duration": 1.0,
    "dt": 1.0,
    "dt_min": 1e-9,
    "dt_max": 1.0,
    "atol": 1e-6,
    "rtol": 1e-3,
    "save_every": 0.1,
    "krylov_max_iters": 2,
    "krylov_residual_reduction": 1e-12,
    "integral_gain": 0.2,
    "proportional_gain": 0.4,
    "deadband_min": 1.0,
    "deadband_max": 1.1,
    "min_step_shrink": 0.5,
    "max_step_growth": 2.0,
    "output_types": ["state", "time"],
    # The stiff two-state system declares no observables; the shared
    # defaults index two of them.
    "saved_observable_indices": [],
    "summarised_observable_indices": [],
}

IRRECOVERABLE = {
    "system_type": "stiff",
    "precision": np.float64,
    "algorithm": "rodas3p",
    "preconditioner_type": "neumann",
    "step_controller": "gustafsson",
    "deadband_min": 1.0,
    "deadband_max": 1.2,
    "min_step_shrink": 0.2,
    "max_step_growth": 8.0,
    "duration": 1.0,
    "dt": 0.5,
    "dt_min": 0.4,
    "dt_max": 0.5,
    "atol": 1e-13,
    "rtol": 1e-13,
    "save_every": 0.1,
    "output_types": ["state", "time"],
}

# One explicit inner tolerance; the rest are left unset (``None``
# marks not-given) and must derive from the controller.
CN_ADAPTIVE_KRYLOV_GIVEN = {
    "algorithm": "crank_nicolson",
    "step_controller": "pid",
    "atol": 1e-8,
    "rtol": 1e-8,
    "dt_min": 1e-10,
    "dt_max": 0.1,
    "krylov_atol": 3e-5,
    "krylov_rtol": None,
    "newton_atol": None,
    "newton_rtol": None,
}

RODAS3P_ADAPTIVE_KRYLOV_DEFAULT = {
    "algorithm": "rodas3p",
    "step_controller": "pid",
    "atol": 3e-7,
    "rtol": 2e-4,
    "dt_min": 1e-10,
    "dt_max": 0.1,
    "krylov_residual_reduction": None,
}

RODAS3P_ADAPTIVE_KRYLOV_GIVEN = {
    **RODAS3P_ADAPTIVE_KRYLOV_DEFAULT,
    "krylov_residual_reduction": 0.03125,
}

IMPOSSIBLE_TOLERANCE = {
    "algorithm": "crank_nicolson",
    "step_controller": "pid",
    "atol": 1e-13,
    "rtol": 1e-13,
    "dt": 0.01,
    "dt_min": 1e-6,
    "dt_max": 0.1,
    "duration": 0.2,
    "output_types": ["state", "time"],
}

# dt0 sits outside the chain-default [dt_min, dt_max] band.
DT_CLAMP_CASES = {
    "max_limit": {"dt0": 2.0, "error": np.asarray([1e-12, 1e-12, 1e-12])},
    "min_limit": {"dt0": 5e-8, "error": np.asarray([1e12, 1e12, 1e12])},
}

RESIDUAL_SETTINGS = {
    "krylov_residual_reduction": 0.2,
    "krylov_residual_floor": 0.03,
}

RESIDUAL_ARRANGEMENTS = [
    {**RESIDUAL_SETTINGS, "algorithm": "backwards_euler"},
    {
        **RESIDUAL_SETTINGS,
        "algorithm": "backwards_euler",
        "linear_correction_type": "bicgstab",
    },
    {**RESIDUAL_SETTINGS, "algorithm": "ros3p"},
]


_DRIVER_DURATION = 2.0 * np.pi
_DRIVER_SAMPLES = 127
# One full period on a uniform grid whose last knot lands exactly on
# the duration, so the periodic wrap closes on itself.
_DRIVER_SAMPLE_PERIOD = _DRIVER_DURATION / (_DRIVER_SAMPLES - 1)
_DRIVER_TIMES = np.arange(_DRIVER_SAMPLES) * _DRIVER_SAMPLE_PERIOD
_DRIVER_VALUES = np.sin(_DRIVER_TIMES)
_DRIVER_VALUES[-1] = _DRIVER_VALUES[0]

TIME_DRIVER_SETTINGS = {
    "system_type": "time_array_driver",
    "duration": _DRIVER_DURATION,
    "dt_min": 0.05,
    "dt_max": 0.05,
    "save_every": 0.05,
    "summarise_every": 0.1,
    "sample_summaries_every": 0.05,
    "saved_state_indices": [0],
    "saved_observable_indices": [0],
    "summarised_state_indices": [0],
    "summarised_observable_indices": [0],
    "output_types": ["state", "observables", "time"],
    "driverspline_wrap": True,
    "driverspline_boundary_condition": "periodic",
}

SINUSOID_DRIVER_SAMPLES = {
    "drive": _DRIVER_VALUES,
    "driver_sample_period": _DRIVER_SAMPLE_PERIOD,
}


# Bicgstab on both implicit step families, one preconditioner each.
BICGSTAB_STEP_CASES = [
    merge_param(MID_RUN_PARAMS, case)
    for case in [
        pytest.param(
            {
                "algorithm": "radau",
                "step_controller": "fixed",
                "linear_correction_type": "bicgstab",
                "preconditioner_type": "jacobi",
            },
            id="firk-bicgstab-jacobi",
        ),
        pytest.param(
            {
                "algorithm": "rosenbrock",
                "step_controller": "i",
                "linear_correction_type": "bicgstab",
                "preconditioner_type": "neumann",
            },
            id="rosenbrock-bicgstab-neumann",
        ),
    ]
]


# Direct LU solves: DIRK with smoothing (at-state solve),
# backwards Euler (base-class path), Rosenbrock-W (cached path),
# radau exact (coupled stacked factorisation) and the inexact-Newton
# pairings (prefactored DIRK, block-transform radau, frozen-J
# iterative).
LU_STEP_CASES = [
    merge_param(MID_RUN_PARAMS, case)
    for case in [
        pytest.param(
            {
                "algorithm": "kvaerno3",
                "step_controller": "pid",
                "use_smoothed_error": True,
                "linear_correction_type": "lu",
            },
            id="dirk-kvaerno3-lu-smoothed",
        ),
        pytest.param(
            {
                "algorithm": "backwards_euler",
                "step_controller": "fixed",
                "linear_correction_type": "lu",
            },
            id="backwards-euler-lu",
        ),
        pytest.param(
            {
                "algorithm": "rosenbrock",
                "step_controller": "i",
                "linear_correction_type": "lu",
            },
            id="rosenbrock-lu",
        ),
        pytest.param(
            {
                "algorithm": "radau",
                "step_controller": "fixed",
                "linear_correction_type": "lu",
            },
            id="firk-radau-lu-exact",
        ),
        pytest.param(
            {
                "algorithm": "radau",
                "step_controller": "fixed",
                "linear_correction_type": "lu",
                "inexact_newton": True,
            },
            id="firk-radau-lu-inexact",
        ),
        pytest.param(
            {
                "algorithm": "radau",
                "step_controller": "pi",
                "use_smoothed_error": True,
                "linear_correction_type": "lu",
                "inexact_newton": True,
            },
            id="firk-radau-lu-inexact-smoothed",
        ),
        pytest.param(
            {
                "algorithm": "kvaerno3",
                "step_controller": "pid",
                "linear_correction_type": "lu",
                "inexact_newton": True,
            },
            id="dirk-kvaerno3-lu-inexact",
        ),
        pytest.param(
            {
                "algorithm": "kvaerno3",
                "step_controller": "pid",
                "linear_correction_type": "minimal_residual",
                "inexact_newton": True,
            },
            id="dirk-kvaerno3-mr-inexact",
        ),
        pytest.param(
            {
                "algorithm": "radau",
                "step_controller": "fixed",
                "linear_correction_type": "bicgstab",
                "preconditioner_type": "jacobi",
                "inexact_newton": True,
            },
            id="firk-radau-bicgstab-inexact",
        ),
    ]
]


# The filtered embedded estimate, one case per implicit family.
SMOOTHED_ERROR_STEP_CASES = [
    merge_param(MID_RUN_PARAMS, case)
    for case in [
        pytest.param(
            {
                "algorithm": "radau",
                "step_controller": "pi",
                "use_smoothed_error": True,
            },
            id="firk-radau-smoothed",
        ),
        pytest.param(
            {
                "algorithm": "kvaerno3",
                "step_controller": "pid",
                "use_smoothed_error": True,
            },
            id="dirk-kvaerno3-smoothed",
        ),
        pytest.param(
            {
                "algorithm": "ros3p",
                "step_controller": "i",
                "use_smoothed_error": True,
            },
            id="rosenbrock-ros3p-smoothed",
        ),
    ]
]


# The multi-step history sequences only apply to controllers
# that carry state between steps.
HISTORY_CONTROLLER_TOLERANCE_SETS = {
    controller: CONTROLLER_TOLERANCE_SETS[controller]
    for controller in ("pi", "pid", "gustafsson")
}


# Precision/timing boundary scenarios (test_ode_loop) and the
# all-local large-DIRK placement base (test_solver).
LARGE_T0_SMALL_STEPS_F32 = {
    'precision': np.float32,
    'output_types': ['state', 'time'],
    'duration': 1e-3,
    'save_every': 2e-4,
    't0': 1e2,
    'algorithm': 'euler',
    'dt': 1e-6,
}

LARGE_T0_SMALL_STEPS_F64 = {
    'precision': np.float64,
    'output_types': ['state', 'time'],
    'duration': 1e-3,
    'save_every': 2e-4,
    't0': 1e2,
    'algorithm': 'euler',
    'dt': 1e-6,
}

TINY_DT_ADAPTIVE_CN = {
    'precision': np.float32,
    'duration': 1e-4,
    'save_every': 2e-5,
    't0': 1.0,
    'algorithm': 'crank_nicolson',
    'step_controller': 'PI',
    'output_types': ['state', 'time'],
    'dt_min': 1e-9,
    'dt': 5e-7,
    'dt_max': 1e-6,
}

WARMUP_SAVE_BOUNDARY = {
    "precision": np.float32,
    "duration": 0.2000,
    "warmup": 0.1,
    "t0": 1.0,
    "output_types": ["state", "time"],
    "algorithm": "euler",
    "dt": 1e-2,
    "save_every": 0.1,
}

DURATION_ONLY_MIXED_OUTPUTS = {
            "precision": np.float32,
            "duration": 0.1,
            "output_types": ["state", "time", "mean"],
            "algorithm": "euler",
            "dt": 0.01,
            "save_every": None,
            "summarise_every": None,
            "sample_summaries_every": None,
        }
