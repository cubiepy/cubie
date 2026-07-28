"""Tests for cubie.outputhandling.save_state."""

from __future__ import annotations

import numpy as np
import pytest
from cubie.cuda_simsafe import cuda
from cubie.memory import default_memmgr

from cubie.outputhandling.save_state import save_state_factory


# ── Structural kernel tests ─────────────────────────────── #


def _run_save_state_kernel(
    saved_state_indices,
    saved_observable_indices,
    save_state,
    save_observables,
    save_time,
    save_counters,
    state_values,
    observable_values,
    counter_values,
    current_step,
    precision,
):
    """Invoke save_state_factory output in a minimal CUDA kernel.

    Returns (state_output, obs_output, counters_output) as host arrays.
    """
    fn = save_state_factory(
        saved_state_indices=saved_state_indices,
        saved_observable_indices=saved_observable_indices,
        save_state=save_state,
        save_observables=save_observables,
        save_time=save_time,
        save_counters=save_counters,
    )

    n_state_cols = len(saved_state_indices) + (1 if save_time else 0)
    n_obs_cols = len(saved_observable_indices)
    n_counter_cols = 4

    # Ensure at least 1 element to avoid zero-size arrays
    state_out = np.full(max(n_state_cols, 1), -999.0, dtype=precision)
    obs_out = np.full(max(n_obs_cols, 1), -999.0, dtype=precision)
    counters_out = np.full(n_counter_cols, -999, dtype=precision)

    d_state = cuda.to_device(
        np.array(state_values, dtype=precision)
    )
    d_obs = cuda.to_device(
        np.array(observable_values, dtype=precision)
    )
    d_counters = cuda.to_device(
        np.array(counter_values, dtype=precision)
    )
    d_state_out = cuda.to_device(state_out)
    d_obs_out = cuda.to_device(obs_out)
    d_counters_out = cuda.to_device(counters_out)

    step_val = precision(current_step)

    @cuda.jit
    def kernel(
        st, obs, ctrs, step, st_out, obs_out, ctrs_out
    ):
        idx = cuda.grid(1)
        if idx > 0:
            return
        fn(st, obs, ctrs, step, st_out, obs_out, ctrs_out)

    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](
        d_state, d_obs, d_counters, step_val,
        d_state_out, d_obs_out, d_counters_out,
    )
    stream.synchronize()

    return (
        d_state_out.copy_to_host(),
        d_obs_out.copy_to_host(),
        d_counters_out.copy_to_host(),
    )


@pytest.mark.parametrize(
    "save_state_flag, save_time_flag, save_obs_flag, save_ctr_flag",
    [
        pytest.param(True, True, True, True, id="all-on"),
        pytest.param(True, False, False, False, id="state-only"),
        pytest.param(False, True, False, False, id="time-only"),
        pytest.param(False, False, True, False, id="obs-only"),
        pytest.param(False, False, False, True, id="counters-only"),
        pytest.param(False, False, False, False, id="all-off"),
    ],
)
def test_save_state_branches(
    save_state_flag, save_time_flag, save_obs_flag, save_ctr_flag,
):
    """Each save flag independently controls whether data is written."""
    precision = np.float32
    state_indices = (2, 0)
    obs_indices = (1, 0)
    state_vals = [10.0, 20.0, 30.0]
    obs_vals = [100.0, 200.0, 300.0]
    ctr_vals = [1.0, 2.0, 3.0, 4.0]
    step = 0.5

    st_out, obs_out, ctr_out = _run_save_state_kernel(
        saved_state_indices=state_indices,
        saved_observable_indices=obs_indices,
        save_state=save_state_flag,
        save_observables=save_obs_flag,
        save_time=save_time_flag,
        save_counters=save_ctr_flag,
        state_values=state_vals,
        observable_values=obs_vals,
        counter_values=ctr_vals,
        current_step=step,
        precision=precision,
    )

    sentinel = np.float32(-999.0)
    n_states = len(state_indices)

    # Items 172/173: state copy on/off
    if save_state_flag:
        # state_indices = [2, 0] -> values [30.0, 10.0]
        assert st_out[0] == pytest.approx(30.0)
        assert st_out[1] == pytest.approx(10.0)
    else:
        # First two slots untouched
        assert st_out[0] == pytest.approx(sentinel)
        assert st_out[1] == pytest.approx(sentinel)

    # Items 174/175: time append on/off
    if save_time_flag:
        # Time is placed at position nstates in the state output
        assert st_out[n_states] == pytest.approx(step)
    elif save_state_flag:
        # With state on but time off, no write beyond state slots
        pass  # no extra slot to check
    else:
        # Time off, no state: nothing written at nstates position
        assert st_out[0] == pytest.approx(sentinel)

    # Items 176/177: observables copy on/off
    if save_obs_flag:
        # obs_indices = [1, 0] -> values [200.0, 100.0]
        assert obs_out[0] == pytest.approx(200.0)
        assert obs_out[1] == pytest.approx(100.0)
    else:
        assert obs_out[0] == pytest.approx(sentinel)

    # Items 178/179: counters copy on/off
    if save_ctr_flag:
        assert ctr_out[0] == pytest.approx(1.0)
        assert ctr_out[1] == pytest.approx(2.0)
        assert ctr_out[2] == pytest.approx(3.0)
        assert ctr_out[3] == pytest.approx(4.0)
    else:
        assert ctr_out[0] == pytest.approx(sentinel)


def test_state_index_mapping():
    """State indices map source positions to output positions (item 180)."""
    precision = np.float32
    # Use non-trivial index mapping: output[0]=state[3], output[1]=state[1]
    state_indices = (3, 1)
    state_vals = [10.0, 20.0, 30.0, 40.0, 50.0]

    st_out, _, _ = _run_save_state_kernel(
        saved_state_indices=state_indices,
        saved_observable_indices=(0,),
        save_state=True,
        save_observables=False,
        save_time=False,
        save_counters=False,
        state_values=state_vals,
        observable_values=[0.0],
        counter_values=[0.0, 0.0, 0.0, 0.0],
        current_step=0.0,
        precision=precision,
    )

    # state_indices[0]=3 -> state_vals[3]=40.0
    # state_indices[1]=1 -> state_vals[1]=20.0
    assert st_out[0] == pytest.approx(40.0)
    assert st_out[1] == pytest.approx(20.0)


def test_observable_index_mapping():
    """Observable indices map source positions to output (item 181)."""
    precision = np.float32
    obs_indices = (2, 0, 3)
    obs_vals = [10.0, 20.0, 30.0, 40.0, 50.0]

    _, obs_out, _ = _run_save_state_kernel(
        saved_state_indices=(0,),
        saved_observable_indices=obs_indices,
        save_state=False,
        save_observables=True,
        save_time=False,
        save_counters=False,
        state_values=[0.0],
        observable_values=obs_vals,
        counter_values=[0.0, 0.0, 0.0, 0.0],
        current_step=0.0,
        precision=precision,
    )

    # obs_indices=[2,0,3] -> [30.0, 10.0, 40.0]
    assert obs_out[0] == pytest.approx(30.0)
    assert obs_out[1] == pytest.approx(10.0)
    assert obs_out[2] == pytest.approx(40.0)
