"""Tests for cubie.batchsolving.arrays.BatchInputArrays."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubie.batchsolving.arrays.BatchInputArrays import (
    InputArrayContainer,
    InputArrays,
)
from cubie.batchsolving.arrays.BaseArrayManager import ManagedArray
from cubie.outputhandling.output_sizes import BatchInputSizes


# ── InputArrayContainer fields (items 1-2) ──────────────── #


def test_container_three_managed_array_fields():
    """Container has initial_values, parameters, driver_coefficients."""
    # __init__ test: inline construction permitted
    container = InputArrayContainer()
    names = container.array_names()
    assert set(names) == {
        "initial_values", "parameters", "driver_coefficients",
    }
    for name in names:
        ma = container.get_managed_array(name)
        assert type(ma) is ManagedArray


def test_container_field_stride_orders_and_defaults():
    """initial_values/parameters have (variable, run); driver has 3D."""
    # __init__ test: inline construction permitted
    container = InputArrayContainer()
    iv = container.initial_values
    assert iv.stride_order == ("variable", "run")
    assert iv.default_shape == (1, 1)
    assert iv.dtype == np.float32

    p = container.parameters
    assert p.stride_order == ("variable", "run")
    assert p.default_shape == (1, 1)

    dc = container.driver_coefficients
    assert dc.is_chunked is False
    assert dc.default_shape == (1, 1, 1)


# ── host_factory / device_factory (items 3-4) ───────────── #


def test_host_factory_sets_pinned_memory():
    """host_factory creates container with pinned memory for all arrays."""
    # __init__ test: classmethod construction
    container = InputArrayContainer.host_factory()
    for _, ma in container.iter_managed_arrays():
        assert ma.memory_type == "pinned"


def test_host_factory_custom_memory_type():
    """host_factory accepts a custom memory_type argument."""
    # __init__ test: classmethod construction
    container = InputArrayContainer.host_factory(memory_type="host")
    for _, ma in container.iter_managed_arrays():
        assert ma.memory_type == "host"


def test_device_factory_sets_device_memory():
    """device_factory creates container with device memory for all arrays."""
    # __init__ test: classmethod construction
    container = InputArrayContainer.device_factory()
    for _, ma in container.iter_managed_arrays():
        assert ma.memory_type == "device"


# ── __attrs_post_init__ (items 5-6) ─────────────────────── #


def test_post_init_host_pinned_device_device(solverkernel):
    """After construction, host is pinned and device is device."""
    ia = solverkernel.input_arrays
    for _, ma in ia.host.iter_managed_arrays():
        assert ma.memory_type == "pinned"
    for _, ma in ia.device.iter_managed_arrays():
        assert ma.memory_type == "device"


# ── update (items 7-8) ──────────────────────────────────── #


def test_update_sets_host_arrays(
    solverkernel_mutable, system, precision
):
    """update stores initial_values and parameters on host container."""
    sk = solverkernel_mutable
    ia = sk.input_arrays
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    inits = np.ones((n_states, 1), dtype=precision)
    params = np.full((n_params, 1), 2.0, dtype=precision)
    ia.update(sk, inits, params, None)

    assert_array_equal(ia.initial_values, inits)
    assert_array_equal(ia.parameters, params)


def test_update_includes_driver_coefficients(
    solverkernel_mutable, system, precision
):
    """update includes driver_coefficients when provided."""
    sk = solverkernel_mutable
    ia = sk.input_arrays
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    n_drivers = system.sizes.drivers
    inits = np.ones((n_states, 1), dtype=precision)
    params = np.ones((n_params, 1), dtype=precision)
    drivers = np.ones((4, n_drivers, 1), dtype=precision) * 3.0
    ia.update(sk, inits, params, drivers)

    assert_array_equal(ia.driver_coefficients, drivers)


def test_update_fast_path_requeues_attached_inputs(
    solverkernel_mutable, system, precision
):
    """Re-supplying the attached arrays queues overwrites only."""
    sk = solverkernel_mutable
    ia = sk.input_arrays
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    n_drivers = system.sizes.drivers
    inits = np.ones((n_states, 1), dtype=precision)
    params = np.full((n_params, 1), 2.0, dtype=precision)
    drivers = np.ones((4, n_drivers, 1), dtype=precision) * 3.0
    ia.update(sk, inits, params, drivers)
    attached_inits = ia.host.initial_values.array
    attached_params = ia.host.parameters.array
    attached_drivers = ia.host.driver_coefficients.array
    ia._needs_overwrite = []

    inits[:] = 7.0
    drivers[:] = 9.0
    ia.update(sk, attached_inits, attached_params, drivers)

    assert ia.host.initial_values.array is attached_inits
    assert ia.host.parameters.array is attached_params
    assert ia.host.driver_coefficients.array is attached_drivers
    assert set(ia._needs_overwrite) == {
        "initial_values", "parameters", "driver_coefficients"
    }
    assert_array_equal(ia.initial_values, inits)
    assert_array_equal(ia.driver_coefficients, drivers)


# ── Forwarding properties (items 9-14) ──────────────────── #


@pytest.mark.parametrize(
    "prop, container_attr, field_name",
    [
        ("initial_values", "host", "initial_values"),
        ("parameters", "host", "parameters"),
        ("driver_coefficients", "host", "driver_coefficients"),
        ("device_initial_values", "device", "initial_values"),
        ("device_parameters", "device", "parameters"),
        ("device_driver_coefficients", "device", "driver_coefficients"),
    ],
)
def test_forwarding_properties(
    solverkernel, prop, container_attr, field_name
):
    """Forwarding properties return the same object as the container."""
    ia = solverkernel.input_arrays
    container = getattr(ia, container_attr)
    expected = container.get_managed_array(field_name).array
    actual = getattr(ia, prop)
    assert actual is expected


# ── from_solver (item 15) ───────────────────────────────── #


def test_from_solver_sizes_precision_manager(solverkernel):
    """from_solver sets sizes, precision, memory_manager, stream_group."""
    ia = InputArrays.from_solver(solverkernel)
    assert type(ia) is InputArrays
    assert ia._precision == solverkernel.precision
    assert type(ia._sizes) is BatchInputSizes
    assert ia._memory_manager is solverkernel.memory_manager
    assert ia._stream_group == solverkernel.stream_group


# ── update_from_solver (items 16-19) ────────────────────── #


def test_update_from_solver_sizes_precision_runs(solverkernel_mutable):
    """update_from_solver updates sizes, precision, runs, and dtypes."""
    sk = solverkernel_mutable
    ia = sk.input_arrays
    ia.update_from_solver(sk)

    # Item 16: _sizes updated
    assert type(ia._sizes) is BatchInputSizes

    # Item 17: _precision updated
    assert ia._precision == sk.precision

    # Item 18: num_runs set
    assert ia.num_runs == sk.num_runs

    # Item 19: floating-point array dtypes match precision
    for _, arr_obj in ia._iter_managed_arrays:
        if np.issubdtype(np.dtype(arr_obj.dtype), np.floating):
            assert arr_obj.dtype == sk.precision


# ── initialise non-chunked (items 21, 23, 25) ───────────── #


def test_initialise_non_chunked_clears_overwrite_list(
    solverkernel_mutable, system, precision
):
    """Non-chunked initialise copies _needs_overwrite arrays then clears."""
    sk = solverkernel_mutable
    ia = sk.input_arrays
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    inits = np.ones((n_states, 1), dtype=precision)
    params = np.ones((n_params, 1), dtype=precision)
    ia.update(sk, inits, params, None)
    ia._memory_manager.allocate_queue(ia)
    # Force non-chunked mode
    ia._chunks = 1
    ia.initialise(0)
    assert ia._needs_overwrite == []


# ── initialise chunked (items 22, 24) ───────────────────── #
# Chunked transfers require real memory manager allocation with
# multiple runs. Tested via the chunked_solved_solver fixture
# in test_chunking.py; items 22 and 24 are covered by those
# integration tests through the conftest fixtures.


# ── reset (items 27-28) ─────────────────────────────────── #


def test_reset_clears_pool_and_buffers(solverkernel_mutable):
    """reset calls super().reset() and clears the staging pool."""
    ia = solverkernel_mutable.input_arrays
    ia.reset()
    assert ia._buffer_pool._buffers == {}
    # super().reset() clears host/device and tracking lists
    assert ia._needs_reallocation == []
    assert ia._needs_overwrite == []
