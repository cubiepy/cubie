import ctypes
import gc
from pathlib import Path
import weakref

import pytest

from cubie.cuda_simsafe import cuda

from cubie.cuda_simsafe import (
    CudaSupportError,
    DeviceNDArray,
    Stream,
    empty_pinned,
    is_pinned_array,
)

from cubie.memory.cupy_emm import CuPyAsyncNumbaManager
from cubie.memory.mem_manager import (
    HOST_SPILL_FRACTION,
    MemoryManager,
    NoCudaDeviceError,
    ArrayRequest,
    ArrayResponse,
    InstanceMemorySettings,
    available_system_ram,
    total_system_ram,
    _numba_stream_ptr,
    current_cupy_stream,
    get_portioned_request_size,
    is_request_chunkable,
    placeholder_dataready,
    placeholder_invalidate,
    replace_with_chunked_size,
)
from tests.memory.conftest import (
    DummyStream,
    FakeAllocation,
    MemoryClient,
    registered_mgr_context_safe,
)

import numpy as np


class TestInstanceMemorySettings:
    def test_instantiation(self, instance_settings_obj, instance_settings):
        # Test that the settings in the object match the settings fixture
        assert (
            instance_settings_obj.proportion == instance_settings["proportion"]
        )
        assert callable(instance_settings_obj.invalidate_hook)
        assert isinstance(instance_settings_obj.allocations, dict)

    def test_add_allocation(self, instance_settings_obj):
        # test that add_allocation adds the reference to the allocations dict
        arr = np.ndarray((10,), dtype=np.float64)
        instance_settings_obj.add_allocation("foo", arr)
        assert "foo" in instance_settings_obj.allocations
        assert instance_settings_obj.allocations["foo"] is arr

    def test_free(self, instance_settings_obj):
        # test that free removes the reference to the allocations dict
        arr = np.ndarray((10,), dtype=np.float64)
        instance_settings_obj.add_allocation("foo", arr)
        instance_settings_obj.free("foo")
        assert "foo" not in instance_settings_obj.allocations

    def test_free_all(self, instance_settings_obj):
        # test that free_all removes all references to the allocations dict
        arr1 = np.ndarray((10,), dtype=np.float64)
        arr2 = np.ndarray((20,), dtype=np.float64)
        instance_settings_obj.add_allocation("foo", arr1)
        instance_settings_obj.add_allocation("bar", arr2)
        instance_settings_obj.free_all()
        assert instance_settings_obj.allocations == {}

    @pytest.mark.nocudasim
    def test_allocated_bytes(self, instance_settings_obj):
        # test that the allocated_bytes property returns the correct value
        arr1 = np.ndarray((100,), dtype=np.float64)
        arr2 = np.ndarray((25,), dtype=np.float64)
        instance_settings_obj.add_allocation("foo", arr1)
        instance_settings_obj.add_allocation("bar", arr2)
        expected_bytes = arr1.nbytes + arr2.nbytes
        assert instance_settings_obj.allocated_bytes == expected_bytes
        instance_settings_obj.free("foo")
        assert instance_settings_obj.allocated_bytes == arr2.nbytes


class TestMemoryManager:
    @pytest.mark.nocudasim
    def test_instantiation(self, mgr):
        """Test that the settings in the object match the settings fixture"""
        assert mgr.totalmem == 8 * 1024**3
        assert mgr.registry == {}
        assert mgr._mode in ["passive", "active"]
        assert isinstance(mgr.stream_groups, type(mgr.stream_groups))

    def test_register(self, registered_mgr, registered_instance):
        """Test that the register method adds the instance to the
        appropriate pool, the proportion and cap are set correctly,
        an entry is added to registry, that the instance gets a stream"""
        mgr = registered_mgr
        instance = registered_instance
        assert id(instance) in mgr.registry
        settings = mgr.registry[id(instance)]
        assert settings.proportion == instance.proportion
        assert settings.cap == int(mgr.totalmem * instance.proportion)
        group = mgr.stream_groups.get_group(instance)
        assert group == "default"
        assert id(instance) in mgr.stream_groups.groups[group]
        assert mgr.stream_groups.get_stream(instance) is not None

    def test_set_limit_mode(self, mgr):
        """Test that set_limit_mode assigns the mode correctly,
        and that it raises ValueError if an invalid mode is passed"""
        mgr.set_limit_mode("active")
        assert mgr._mode == "active"
        mgr.set_limit_mode("passive")
        assert mgr._mode == "passive"
        with pytest.raises(ValueError):
            mgr.set_limit_mode("invalid")

    @pytest.mark.nocudasim
    def test_get_stream(
        self,
        mem_manager_settings,
        registered_instance_settings,
        memory_client,
    ):
        """test that get_stream successfully passes a different stream for
        instances registered in different stream groups"""

        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        stream = regmgr.get_stream(instance)
        assert isinstance(stream, Stream)
        # Register another instance in a new group
        inst2 = memory_client
        regmgr.register(inst2, stream_group="other")
        stream2 = regmgr.get_stream(inst2)
        assert stream2 is not None
        assert int(stream.handle) != int(stream2.handle)

    def test_change_stream_group(
        self,
        mem_manager_settings,
        registered_instance_settings,
        memory_client,
    ):
        """test that change_stream_group changes the stream group of an
        instance, and that it raises ValueError if the instance wasn't
        already in a group"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        regmgr.change_stream_group(instance, "other")
        assert id(instance) in regmgr.stream_groups.groups["other"]
        dummy = memory_client
        with pytest.raises(ValueError):
            regmgr.change_stream_group(dummy, "newgroup")

    @pytest.mark.nocudasim
    def test_reinit_streams(
        self, mem_manager_settings, registered_instance_settings
    ):
        """test that reinit_streams causess a different stream to be
        returned from get_stream"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        stream1 = regmgr.get_stream(instance)
        regmgr.reinit_streams()
        stream2 = regmgr.get_stream(instance)
        assert int(stream1.handle) != int(stream2.handle)

    def test_invalidate_all(
        self,
        mem_manager_settings,
        registered_instance_settings,
        memory_client,
    ):
        """Add a new instance with a measurable invalidate hook, and check
        that it is called when invalidate all is called"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        called = {"flag": False}

        def hook():
            called["flag"] = True

        inst = memory_client
        regmgr.register(inst, invalidate_cache_hook=hook)
        regmgr.invalidate_all()
        assert called["flag"] is True

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": None}], indirect=True
    )
    def test_set_manual_limit_mode(
        self, mem_manager_settings, registered_instance_settings
    ):
        """Test that set_manual_limit_mode sets the instance to manual mode"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        instance_id = id(instance)
        regmgr.set_manual_limit_mode(instance, 0.3)
        assert instance_id in regmgr._manual_pool

    def test_set_auto_limit_mode(
        self, mem_manager_settings, registered_instance_settings
    ):
        """Test that set_auto_limit_mode sets the instance to auto mode"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        regmgr.set_auto_limit_mode(instance)
        instance_id = id(instance)
        assert instance_id not in regmgr._manual_pool

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": 0.5}], indirect=True
    )
    def test_proportion(
        self,
        mem_manager_settings,
        registered_instance_settings,
        memory_client,
    ):
        """Test that proportion returns the requested proportion if set,
        and 1.0 if not set (auto)"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        assert regmgr.proportion(instance) == instance.proportion
        # Register auto instance
        inst2 = memory_client
        regmgr.register(inst2, proportion=None)
        assert regmgr.proportion(inst2) == 0.5

    def test_cap(self, mem_manager_settings, registered_instance_settings):
        """Test that cap returns the correct value"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        proportion = instance.proportion
        testcap = regmgr.cap(instance)
        free, total = regmgr.get_memory_info()
        solvercap = total * proportion
        assert testcap == solvercap

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": None}], indirect=True
    )
    @pytest.mark.parametrize("memory_clients", [4], indirect=True)
    def test_pool_proportions(
        self,
        mem_manager_settings,
        registered_instance_settings,
        memory_clients,
    ):
        """Add a few auto and manual instances, check that the total manual
        and auto pool proportions match expected. Test add and rebalance
        methods along the way"""
        regmgr, instance = registered_mgr_context_safe(
            mem_manager_settings, registered_instance_settings
        )
        instance1 = instance
        instance2, instance3, instance4, instance5 = memory_clients

        # already registered
        with pytest.raises(ValueError):
            regmgr.register(instance1)
        regmgr.set_limit_mode("active")

        assert regmgr.proportion(instance1) == 1.0
        assert regmgr.manual_pool_proportion == 0.0
        assert regmgr.auto_pool_proportion == 1.0

        regmgr.register(instance2)
        assert abs(regmgr.proportion(instance2) - 0.5) < 1e-6
        assert abs(regmgr.manual_pool_proportion - 0.0) < 1e-6
        assert abs(regmgr.auto_pool_proportion - 1.0) < 1e-6

        regmgr.register(instance3, proportion=0.5)
        assert abs(regmgr.proportion(instance3) - 0.5) < 1e-6
        assert abs(regmgr.manual_pool_proportion - 0.5) < 1e-6
        assert abs(regmgr.auto_pool_proportion - 0.5) < 1e-6

        regmgr.register(instance4, proportion=0.4)
        assert abs(regmgr.proportion(instance4) - 0.4) < 1e-6
        assert abs(regmgr.manual_pool_proportion - 0.9) < 1e-6
        assert abs(regmgr.auto_pool_proportion - 0.1) < 1e-6

        regmgr.set_auto_limit_mode(instance3)
        assert abs(regmgr.proportion(instance3) - 0.2) < 1e-6
        assert abs(regmgr.manual_pool_proportion - 0.4) < 1e-6
        assert abs(regmgr.auto_pool_proportion - 0.6) < 1e-6

        regmgr.set_manual_limit_mode(instance3, 0.3)
        assert abs(regmgr.proportion(instance3) - 0.3) < 1e-6
        assert abs(regmgr.manual_pool_proportion - 0.7) < 1e-6
        assert abs(regmgr.auto_pool_proportion - 0.3) < 1e-6

        with pytest.raises(ValueError):
            regmgr.register(instance5, proportion=0.3)

    def test_process_request(self, mgr, memory_client):
        """Test single_request calls allocation hook with correct ArrayResponse."""
        # Create instance with callback to capture response
        instance = memory_client
        callback_called = {"flag": False, "response": None}

        def allocation_hook(response):
            callback_called["flag"] = True
            callback_called["response"] = response

        mgr.register(
            instance,
            allocation_ready_hook=allocation_hook,
            invalidate_cache_hook=instance.notice_invalidate,
        )

        requests = {
            "arr1": ArrayRequest(
                shape=(8, 8, 8),
                dtype=np.float32,
                memory="device",
                total_runs=8,
            ),
            "arr2": ArrayRequest(
                shape=(4, 4, 8),
                dtype=np.float32,
                memory="device",
                total_runs=8,
            ),
        }
        mgr.queue_request(instance, requests)
        mgr.allocate_queue(instance)
        # Check that callback was called with correct response
        assert callback_called["flag"] is True
        response = callback_called["response"]
        assert isinstance(response, ArrayResponse)
        assert set(response.arr.keys()) == set(requests.keys())
        for key, arr in response.arr.items():
            assert arr.shape == requests[key].shape
            assert arr.dtype == requests[key].dtype
        assert isinstance(response.chunks, int)

    def test_allocate_all(self, registered_mgr, registered_instance):
        """Test allocate_all allocates arrays for all requests and stores them in allocations."""
        mgr = registered_mgr
        instance = registered_instance
        requests = {
            "arr1": ArrayRequest(
                shape=(2, 2, 2),
                dtype=np.float32,
                memory="device",
                total_runs=2,
            ),
            "arr2": ArrayRequest(
                shape=(2, 2, 2),
                dtype=np.float32,
                memory="device",
                total_runs=2,
            ),
        }
        stream = mgr.get_stream(instance)
        arrays = mgr.allocate_all(requests, id(instance), stream)
        assert set(arrays.keys()) == set(requests.keys())
        for key, arr in arrays.items():
            assert arr.shape == requests[key].shape
            assert arr.dtype == requests[key].dtype
            assert key in mgr.registry[id(instance)].allocations

    @pytest.mark.nocudasim
    def test_allocate(self, mgr):
        """Test allocate returns correct array type and shape for each memory type."""
        for mem_type in ["device", "pinned"]:
            arr = mgr.allocate(
                shape=(2, 2), dtype=np.float32, memory_type=mem_type
            )
            if mem_type == "device":
                assert hasattr(arr, "__cuda_array_interface__")
            else:
                assert isinstance(arr, np.ndarray)
            assert arr.shape == (2, 2)
            assert arr.dtype == np.float32
        for unsupported in ["mapped", "managed", "invalid"]:
            with pytest.raises(ValueError):
                mgr.allocate(
                    shape=(1, 1), dtype=np.float32, memory_type=unsupported
                )

    @pytest.mark.nocudasim
    @pytest.mark.cupy
    def test_allocate_device_returns_native_array(self, mgr):
        """A "device" allocation is a native Numba device array.

        Device memory is drawn from CuPy's async pool through the EMM
        plugin, so the returned object is a Numba DeviceNDArray (fast
        kernel-launch path), not a raw CuPy ndarray.
        """
        import cupy as cp

        arr = mgr.allocate(
            shape=(4, 4), dtype=np.float32, memory_type="device"
        )
        assert isinstance(arr, DeviceNDArray)
        assert not isinstance(arr, cp.ndarray)
        context_manager = cuda.current_context().memory_manager
        assert isinstance(context_manager, CuPyAsyncNumbaManager)

    def test_free(self, registered_mgr, registered_instance):
        """Test free removes allocation by key from all instances."""
        mgr = registered_mgr
        instance = registered_instance
        arr = np.zeros((2, 2), dtype=np.float32)
        mgr.registry[id(instance)].add_allocation("foo", arr)
        mgr.free("foo")
        assert "foo" not in mgr.registry[id(instance)].allocations

    def test_free_all(self, registered_mgr, registered_instance):
        """Test free_all removes all allocations from all instances."""
        mgr = registered_mgr
        instance = registered_instance
        arr = np.zeros((2, 2), dtype=np.float32)
        mgr.registry[id(instance)].add_allocation("foo", arr)
        mgr.registry[id(instance)].add_allocation("bar", arr)
        mgr.free_all()
        assert mgr.registry[id(instance)].allocations == {}

    def test_get_memory_info(self, mgr):
        """Test get_memory_info returns tuple of free and total memory."""
        free, total = mgr.get_memory_info()
        assert free == 1 * 1024**3
        assert total == 8 * 1024**3

    def test_create_host_array_1d(self, mgr):
        """Test create_host_array returns correct 1D array."""
        arr = mgr.create_host_array(shape=(10,), dtype=np.float32)
        assert arr.shape == (10,)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, np.zeros(10, dtype=np.float32))

    def test_create_host_array_2d(self, mgr):
        """Test create_host_array returns correct 2D array."""
        arr = mgr.create_host_array(
            shape=(5, 3),
            dtype=np.float64,
        )
        assert arr.shape == (5, 3)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, np.zeros((5, 3), dtype=np.float64))

    def test_create_host_array_3d_default_stride(self, mgr):
        """Test create_host_array returns correct 3D array with default stride."""
        arr = mgr.create_host_array(
            shape=(2, 3, 4),
            dtype=np.float32,
        )
        assert arr.shape == (2, 3, 4)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(
            arr, np.zeros((2, 3, 4), dtype=np.float32)
        )

    def test_create_host_array_3d_custom_stride(self, mgr):
        """Test create_host_array returns C-contiguous 3D array."""
        arr = mgr.create_host_array(
            shape=(2, 3, 4),
            dtype=np.float32,
        )
        assert arr.shape == (2, 3, 4)
        assert arr.dtype == np.float32
        assert arr.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(
            arr, np.zeros((2, 3, 4), dtype=np.float32)
        )

    def test_create_host_array_pageable_memory(self, mgr):
        """Test create_host_array with host (pageable) memory type."""
        arr = mgr.create_host_array(
            shape=(2, 3, 4),
            dtype=np.float32,
            memory_type="host",
        )
        assert arr.shape == (2, 3, 4)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(
            arr, np.zeros((2, 3, 4), dtype=np.float32)
        )

    def test_create_host_array_invalid_memory_type(self, mgr):
        """Test create_host_array raises ValueError for invalid memory type."""
        with pytest.raises(ValueError, match="memory_type must be"):
            mgr.create_host_array(
                shape=(2, 3, 4),
                dtype=np.float32,
                memory_type="invalid",
            )

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": None}], indirect=True
    )
    def test_manual_pool_proportion(self, registered_mgr, registered_instance):
        """Test manual_pool_proportion property returns correct sum."""
        mgr = registered_mgr
        instance = registered_instance
        mgr.set_manual_limit_mode(instance, 0.3)
        assert abs(mgr.manual_pool_proportion - 0.3) < 1e-6

    def test_auto_pool_proportion(
        self, registered_mgr, registered_instance, memory_client
    ):
        """Test auto_pool_proportion property returns correct sum."""
        mgr = registered_mgr
        instance = registered_instance
        # Add an auto instance
        inst2 = memory_client
        mgr.register(inst2)
        expected = 1.0 - mgr.registry[id(instance)].proportion
        assert abs(mgr.auto_pool_proportion - expected) < 1e-6

    def test_set_manual_proportion(self, registered_mgr, registered_instance):
        """Test set_manual_proportion sets manual proportion and updates registry."""
        mgr = registered_mgr
        instance = registered_instance
        instance_id = id(instance)
        mgr.set_manual_proportion(instance, 0.4)
        assert mgr.registry[id(instance)].proportion == 0.4
        assert instance_id in mgr._manual_pool

    @pytest.mark.parametrize(
        "registered_instance_override", [{"proportion": None}], indirect=True
    )
    def test_set_manual_proportion_from_auto_pool(
        self, registered_mgr, registered_instance, memory_client
    ):
        """Test set_manual_proportion moves an auto instance to manual."""
        mgr = registered_mgr
        instance = registered_instance
        instance_id = id(instance)
        inst2 = memory_client
        mgr.register(inst2)
        assert instance_id in mgr._auto_pool
        mgr.set_manual_proportion(instance, 0.01)
        assert instance_id not in mgr._auto_pool
        assert instance_id in mgr._manual_pool
        assert mgr.registry[instance_id].proportion == 0.01
        assert mgr.registry[instance_id].cap == int(0.01 * mgr.totalmem)
        assert abs(mgr.registry[id(inst2)].proportion - 0.99) < 1e-6

    @pytest.mark.parametrize("memory_clients", [2], indirect=True)
    def test_rebalance_auto_pool(self, mgr, memory_clients):
        """Test _rebalance_auto_pool splits available proportion among auto pool."""
        inst1, inst2 = memory_clients
        mgr.register(inst1)
        mgr.register(inst2)
        assert mgr.registry[id(inst1)].proportion == 0.5
        assert mgr.auto_pool_proportion == 1.0
        mgr.set_manual_limit_mode(inst2, 0.75)
        mgr._rebalance_auto_pool()
        assert abs(mgr.registry[id(inst1)].proportion - 0.25) < 1e-6
        assert abs(mgr.auto_pool_proportion - 0.25) < 1e-6

    def test_get_stream_group(
        self, registered_mgr, registered_instance, memory_client
    ):
        """Test get_stream_group returns the correct group name."""
        mgr = registered_mgr
        instance = registered_instance
        group = mgr.get_stream_group(instance)
        assert group == "default"

        # Test with custom group
        inst2 = memory_client
        mgr.register(inst2, stream_group="custom")
        assert mgr.get_stream_group(inst2) == "custom"

    def test_check_requests(self, mgr):
        """Test _check_requests validates request format correctly."""
        # Valid requests should pass
        valid_requests = {
            "arr1": ArrayRequest(
                shape=(2, 2), dtype=np.float32, memory="device", total_runs=2
            ),
            "arr2": ArrayRequest(
                shape=(3, 3), dtype=np.float64, memory="pinned", total_runs=3
            ),
        }
        mgr._check_requests(valid_requests)  # Should not raise

        # Unsupported placements are rejected at request construction
        with pytest.raises(ValueError):
            ArrayRequest(
                shape=(3, 3), dtype=np.float64, memory="mapped", total_runs=3
            )
        with pytest.raises(ValueError):
            ArrayRequest(
                shape=(3, 3), dtype=np.float64, memory="managed", total_runs=3
            )

        # Invalid dict type should raise TypeError
        with pytest.raises(TypeError):
            mgr._check_requests("not a dict")

        # Invalid request values should raise TypeError
        invalid_requests = {
            "arr1": ArrayRequest(
                shape=(2, 2), dtype=np.float32, memory="device", total_runs=2
            ),
            "arr2": "not an ArrayRequest",
        }
        with pytest.raises(TypeError):
            mgr._check_requests(invalid_requests)

    def test_queue_request(self, registered_mgr, registered_instance):
        """Test queue_request adds requests to the queue correctly."""
        mgr = registered_mgr
        instance = registered_instance

        requests = {
            "arr1": ArrayRequest(
                shape=(2, 2), dtype=np.float32, memory="device", total_runs=2
            ),
        }

        mgr.queue_request(instance, requests)
        stream_group = mgr.get_stream_group(instance)
        assert stream_group in mgr._queued_allocations
        assert id(instance) in mgr._queued_allocations[stream_group]
        assert mgr._queued_allocations[stream_group][id(instance)] == requests

        # Test that subsequent requests from same instance overwrite (not append)
        requests2 = {
            "arr2": ArrayRequest(
                shape=(3, 3), dtype=np.float32, memory="device"
            ),
        }
        mgr.queue_request(instance, requests2)
        # Should only have the latest request
        assert mgr._queued_allocations[stream_group][id(instance)] == requests2

    def test_allocate_queue_single_instance(self, mgr, memory_client):
        """Test allocate_queue with single instance in queue."""
        instance = memory_client
        callback_called = {"flag": False, "response": ArrayResponse()}

        def allocation_hook(response):
            callback_called["flag"] = True
            callback_called["response"] = response

        mgr.register(
            instance,
            allocation_ready_hook=allocation_hook,
            invalidate_cache_hook=instance.notice_invalidate,
        )

        requests = {
            "arr1": ArrayRequest(
                shape=(2, 2, 2),
                dtype=np.float32,
                memory="device",
                total_runs=2,
            ),
        }

        mgr.queue_request(instance, requests)
        mgr.allocate_queue(instance)

        # Check that callback was called
        assert callback_called["flag"] is True
        assert isinstance(callback_called["response"], ArrayResponse)
        response = callback_called["response"]

        assert response.arr["arr1"].shape == (2, 2, 2)
        assert response.arr["arr1"].dtype == np.float32
        assert response.chunks == 1

    @pytest.mark.parametrize("memory_clients", [2], indirect=True)
    def test_allocate_queue_multiple_instances_group_limit(
        self, mgr, memory_clients
    ):
        """Test allocate_queue with multiple instances using group limit."""
        inst1, inst2 = memory_clients
        callbacks_called = {
            "inst1": {"flag": False, "response": ArrayResponse()},
            "inst2": {"flag": False, "response": ArrayResponse()},
        }

        def hook1(response):
            callbacks_called["inst1"]["flag"] = True
            callbacks_called["inst1"]["response"] = response

        def hook2(response):
            callbacks_called["inst2"]["flag"] = True
            callbacks_called["inst2"]["response"] = response

        mgr.register(
            inst1,
            allocation_ready_hook=hook1,
            invalidate_cache_hook=inst1.notice_invalidate,
            stream_group="test",
        )
        mgr.register(
            inst2,
            allocation_ready_hook=hook2,
            invalidate_cache_hook=inst2.notice_invalidate,
            stream_group="test",
        )

        requests1 = {
            "arr1": ArrayRequest(
                shape=(2, 2, 2),
                dtype=np.float32,
                memory="device",
                total_runs=2,
            )
        }
        requests2 = {
            "arr2": ArrayRequest(
                shape=(3, 3, 3),
                dtype=np.float32,
                memory="device",
                total_runs=3,
            )
        }

        mgr.queue_request(inst1, requests1)
        mgr.queue_request(inst2, requests2)
        mgr.allocate_queue(inst1)

        # Both callbacks should be called
        assert callbacks_called["inst1"]["flag"] is True
        assert callbacks_called["inst2"]["flag"] is True

    def test_allocate_queue_empty_queue(
        self, registered_mgr, registered_instance
    ):
        """Test allocate_queue with empty queue returns None."""
        mgr = registered_mgr
        instance = registered_instance

        mgr.allocate_queue(instance)

    @pytest.mark.parametrize(
        "fixed_mem_override", [{"free": 1024}], indirect=True
    )
    def test_allocate_queue_empty_rebroadcasts_chunk_parameters(
        self, mgr, memory_client
    ):
        """Test allocate_queue with nothing queued resends chunk params."""
        instance = memory_client
        responses = []
        mgr.register(
            instance,
            allocation_ready_hook=lambda r: responses.append(r),
            invalidate_cache_hook=instance.notice_invalidate,
        )

        requests = {
            "arr1": ArrayRequest(
                shape=(4, 100),
                dtype=np.float32,
                memory="device",
                chunk_axis_index=1,
                total_runs=100,
            ),
        }
        mgr.queue_request(instance, requests)
        mgr.allocate_queue(instance)

        assert len(responses) == 1
        first = responses[0]
        assert first.chunks > 1

        # Repeat call with nothing queued: chunk parameters rebroadcast
        mgr.allocate_queue(instance)
        assert len(responses) == 2
        second = responses[1]
        assert second.arr == {}
        assert second.chunks == first.chunks
        assert second.chunk_length == first.chunk_length

    @pytest.mark.parametrize("memory_clients", [5], indirect=True)
    def test_is_grouped(self, mgr, memory_clients):
        """Test is_grouped returns correct grouping status for instances."""
        (
            inst_default,
            inst_single,
            inst_group1,
            inst_group2,
            inst_group3,
        ) = memory_clients

        # Test instance in default group (should return False)
        mgr.register(inst_default, stream_group="default")
        assert mgr.is_grouped(inst_default) is False

        # Test single instance in named group (should return False)
        mgr.register(inst_single, stream_group="single_group")
        assert mgr.is_grouped(inst_single) is False

        # Test multiple instances in named group (should return True)
        mgr.register(inst_group1, stream_group="multi_group")
        mgr.register(inst_group2, stream_group="multi_group")
        assert mgr.is_grouped(inst_group1) is True
        assert mgr.is_grouped(inst_group2) is True

        # Test three instances in named group (should return True)
        mgr.register(inst_group3, stream_group="multi_group")
        assert mgr.is_grouped(inst_group1) is True
        assert mgr.is_grouped(inst_group2) is True
        assert mgr.is_grouped(inst_group3) is True

        # Test after moving instance to different group
        mgr.change_stream_group(inst_group3, "new_group")
        # inst_group3 should now be alone in "new_group"
        assert mgr.is_grouped(inst_group3) is False
        # inst_group1 and inst_group2 should still be grouped in "multi_group"
        assert mgr.is_grouped(inst_group1) is True
        assert mgr.is_grouped(inst_group2) is True

    @pytest.mark.nocudasim
    def test_to_device(self, registered_mgr, registered_instance):
        """Test to_device copies values to allocated device arrays correctly."""

        mgr = registered_mgr
        instance = registered_instance

        # Allocate device arrays through the memory manager
        requests = {
            "arr1": ArrayRequest(
                shape=(3, 4), dtype=np.float32, memory="device", total_runs=4
            ),
            "arr2": ArrayRequest(
                shape=(2, 3), dtype=np.float64, memory="device", total_runs=3
            ),
        }

        stream = mgr.get_stream(instance)
        device_arrays = mgr.allocate_all(requests, id(instance), stream)

        # Create host arrays with test data
        host_arr1 = np.arange(12, dtype=np.float32).reshape(3, 4)
        host_arr2 = np.arange(6, dtype=np.float64).reshape(2, 3) * 2.5

        # Copy to device using to_device method
        from_arrays = [host_arr1, host_arr2]
        to_arrays = [device_arrays["arr1"], device_arrays["arr2"]]

        mgr.to_device(instance, from_arrays, to_arrays)

        # Synchronize stream to ensure copy is complete
        stream.synchronize()

        # Copy back to host and verify values
        result_arr1 = device_arrays["arr1"].copy_to_host()
        result_arr2 = device_arrays["arr2"].copy_to_host()

        np.testing.assert_array_equal(result_arr1, host_arr1)
        np.testing.assert_array_equal(result_arr2, host_arr2)

    @pytest.mark.nocudasim
    def test_from_device(self, registered_mgr, registered_instance):
        """Test from_device copies values from allocated device arrays correctly."""

        mgr = registered_mgr
        instance = registered_instance

        # Allocate device arrays through the memory manager
        requests = {
            "arr1": ArrayRequest(
                shape=(2, 5), dtype=np.float32, memory="device", total_runs=5
            ),
            "arr2": ArrayRequest(
                shape=(3, 2), dtype=np.float64, memory="device", total_runs=2
            ),
        }

        stream = mgr.get_stream(instance)
        device_arrays = mgr.allocate_all(requests, id(instance), stream)

        # Create host arrays with test data and copy to device first
        host_source1 = np.arange(10, dtype=np.float32).reshape(2, 5) * 3.0
        host_source2 = np.arange(6, dtype=np.float64).reshape(3, 2) + 10.0

        # Copy test data to device arrays using the manager's own
        # to_device method (the sole allocation/transfer provider).
        mgr.to_device(
            instance,
            [host_source1, host_source2],
            [device_arrays["arr1"], device_arrays["arr2"]],
        )
        stream.synchronize()

        # Create empty host arrays to receive the data
        host_dest1 = np.zeros_like(host_source1)
        host_dest2 = np.zeros_like(host_source2)

        # Copy from device using from_device method
        from_arrays = [device_arrays["arr1"], device_arrays["arr2"]]
        to_arrays = [host_dest1, host_dest2]

        mgr.from_device(instance, from_arrays, to_arrays)

        # Synchronize stream to ensure copy is complete
        stream.synchronize()

        # Verify that the data was correctly copied from device to host
        np.testing.assert_array_equal(host_dest1, host_source1)
        np.testing.assert_array_equal(host_dest2, host_source2)


@pytest.mark.nocudasim
def test_ensure_cuda_context():
    """Test _ensure_cuda_context validates CUDA is available."""
    from cubie.memory.mem_manager import _ensure_cuda_context

    # This test should not crash when CUDA is available
    # It will raise RuntimeError if CUDA context cannot be initialized
    try:
        _ensure_cuda_context()
    except RuntimeError as e:
        pytest.fail(f"CUDA context validation failed: {e}")


def test_ensure_cuda_context_simulation():
    """Test _ensure_cuda_context is no-op in simulation mode."""
    from cubie.memory.mem_manager import _ensure_cuda_context
    from cubie.cuda_simsafe import CUDA_SIMULATION

    # In simulation mode, the function should do nothing and not raise
    if CUDA_SIMULATION:
        _ensure_cuda_context()  # Should not raise


class TestGetChunkParameters:
    """Tests for get_chunk_parameters method."""

    def test_get_chunk_parameters_unchunkable_exceeds_memory(
        self, mgr, memory_client
    ):
        """Verify error when unchunkable arrays exceed available memory."""
        # Register an instance
        inst = memory_client
        mgr.register(
            inst,
            stream_group="test",
            invalidate_cache_hook=inst.notice_invalidate,
        )

        # Create a request where unchunkable size exceeds available memory
        # Memory manager has 1GB free (1 * 1024**3 bytes)
        # Create a large unchunkable array (2GB total)
        huge_shape = (512, 512, 512)  # 512^3 * 4 bytes = 512MB per array
        requests = {
            id(inst): {
                "huge_unchunkable": ArrayRequest(
                    shape=huge_shape,
                    dtype=np.float32,
                    memory="device",
                    unchunkable=True,
                    total_runs=1,
                ),
                "huge_unchunkable2": ArrayRequest(
                    shape=huge_shape,
                    dtype=np.float32,
                    memory="device",
                    unchunkable=True,
                    total_runs=1,
                ),
                # Need at least one chunkable array to hit the unchunkable
                # exceeds memory path (otherwise hits all-unchunkable path)
                "small_chunkable": ArrayRequest(
                    shape=(1, 1),
                    dtype=np.float32,
                    memory="device",
                    unchunkable=False,
                    total_runs=1,
                ),
            }
        }

        with pytest.raises(ValueError, match="unchunkable"):
            mgr.get_chunk_parameters(
                requests=requests,
                axis_length=512,
                stream_group="test",
            )


class TestAllocateQueueExtractsNumRuns:
    """Tests for allocate_queue extraction of num_runs from triggering
    instance."""

    def test_allocate_queue_extracts_num_runs(self, mgr):
        """Verify allocate_queue correctly extracts num_runs from
        triggering instance.

        The allocate_queue method should extract num_runs from
        triggering_instance.run_params.runs instead of computing it
        from array request shapes. This test verifies that behavior.
        """

        # Create a mock instance with run_params
        class MockRunParams:
            def __init__(self, runs):
                self.runs = runs

        class MockInstance:
            def __init__(self, runs):
                self.run_params = MockRunParams(runs)

            def notice_invalidate(self):
                pass

        instance = MockInstance(runs=100)

        # Track whether the allocation callback was called with correct
        # response
        callback_called = {"flag": False, "response": None}

        def allocation_hook(response):
            callback_called["flag"] = True
            callback_called["response"] = response

        mgr.register(
            instance,
            allocation_ready_hook=allocation_hook,
            invalidate_cache_hook=instance.notice_invalidate,
        )

        # Create requests with arrays that have run axis
        # The shape indicates 50 runs, but run_params.runs is 100
        # allocate_queue should use 100 from run_params, not 50 from
        # shape
        requests = {
            "arr1": ArrayRequest(
                shape=(10, 50),  # 50 in run axis (second dimension)
                dtype=np.float32,
                memory="device",
                unchunkable=False,
                total_runs=100,
            ),
        }

        mgr.queue_request(instance, requests)
        mgr.allocate_queue(instance)

        # Verify callback was called
        assert callback_called["flag"] is True
        response = callback_called["response"]
        assert isinstance(response, ArrayResponse)

        # The key validation: chunk_length should be computed based on
        # num_runs=100 from run_params, not from the array shape
        # Since we have 1GB free and small arrays, no chunking should
        # occur, giving us chunk_length matching the full num_runs
        assert response.chunks == 1
        # chunk_length should equal num_runs when no chunking occurs
        assert response.chunk_length == 100

    def test_allocate_queue_chunks_correctly(self, mgr):
        """Verify chunking calculations work with num_runs from instance.

        When memory constraints force chunking, the chunk calculations
        should be based on num_runs extracted from
        triggering_instance.run_params.runs.
        """

        # Create instance with large num_runs to force chunking
        class MockRunParams:
            def __init__(self, runs):
                self.runs = runs

        class MockInstance:
            def __init__(self, runs):
                self.run_params = MockRunParams(runs)

            def notice_invalidate(self):
                pass

        # Use large num_runs to increase memory requirements
        instance = MockInstance(runs=10000)

        callback_called = {"flag": False, "response": None}

        def allocation_hook(response):
            callback_called["flag"] = True
            callback_called["response"] = response

        mgr.register(
            instance,
            allocation_ready_hook=allocation_hook,
            invalidate_cache_hook=instance.notice_invalidate,
        )

        # Create large requests to force chunking
        # Each element is 4 bytes, so 10000 runs * 100 vars * 4 bytes =
        # 4MB per array. With multiple arrays, this should exceed the
        # 1GB available and trigger chunking.
        requests = {
            "arr1": ArrayRequest(
                shape=(100, 10000),
                dtype=np.float32,
                memory="device",
                unchunkable=False,
                total_runs=10000,
            ),
            "arr2": ArrayRequest(
                shape=(100, 10000),
                dtype=np.float32,
                memory="device",
                unchunkable=False,
                total_runs=10000,
            ),
        }

        mgr.queue_request(instance, requests)
        mgr.allocate_queue(instance)

        # Verify callback was called
        assert callback_called["flag"] is True
        response = callback_called["response"]
        assert isinstance(response, ArrayResponse)

        # With 10000 runs and 1GB available memory, chunking should
        # occur. The exact chunk_length depends on available memory,
        # but it should be less than 10000.
        # Verify that chunking occurred
        if response.chunks > 1:
            # chunk_length should be calculated to fit in available
            # memory
            assert response.chunk_length < 10000
            assert response.chunk_length > 0
            # Verify chunks * chunk_length covers all runs (with
            # possible dangling chunk)
            total_covered = (
                response.chunks - 1
            ) * response.chunk_length + response.chunk_length
            assert total_covered >= 10000
        else:
            # If no chunking occurred (e.g., in CUDA sim with more
            # memory), chunk_length should equal num_runs
            assert response.chunk_length == 10000


def test_allocate_queue_no_chunked_slices_in_response(mgr):
    """Verify ArrayResponse from allocate_queue does not have chunked_slices.

    After refactoring to use on-demand chunk slice computation, the
    ArrayResponse should no longer contain a chunked_slices field. This test
    verifies that the response contains the necessary chunk parameters
    (chunks, , chunk_length, )
    """

    # Create instance with run_params
    class MockRunParams:
        def __init__(self, runs):
            self.runs = runs

    class MockInstance:
        def __init__(self, runs):
            self.run_params = MockRunParams(runs)

        def notice_invalidate(self):
            pass

    instance = MockInstance(runs=100)

    # Track the allocation callback response
    callback_called = {"flag": False, "response": None}

    def allocation_hook(response):
        callback_called["flag"] = True
        callback_called["response"] = response

    mgr.register(
            instance,
            allocation_ready_hook=allocation_hook,
            invalidate_cache_hook=instance.notice_invalidate,
        )

    # Create requests with chunkable arrays
    requests = {
        "arr1": ArrayRequest(
            shape=(10, 100),
            dtype=np.float32,
            memory="device",
            unchunkable=False,
            total_runs=100,
        ),
        "arr2": ArrayRequest(
            shape=(5, 100),
            dtype=np.float32,
            memory="device",
            unchunkable=False,
            total_runs=100,
        ),
    }

    mgr.queue_request(instance, requests)
    mgr.allocate_queue(instance)

    # Verify callback was called
    assert callback_called["flag"] is True
    response = callback_called["response"]
    assert isinstance(response, ArrayResponse)

    # Verify chunked_slices is NOT in the response
    assert not hasattr(response, "chunked_slices")

    # Verify that chunk parameters ARE in the response
    assert hasattr(response, "chunks")
    assert hasattr(response, "chunk_length")
    assert hasattr(response, "chunked_shapes")

    # Verify chunk parameters have expected values
    assert isinstance(response.chunks, int)
    assert isinstance(response.chunk_length, int)
    assert isinstance(response.chunked_shapes, dict)


def test_allocate_queue_uses_first_request_total_runs(mgr):
    """Verify allocate_queue extracts num_runs from first request in queue.

    After refactoring, allocate_queue should get num_runs directly from
    the first request's total_runs field instead of using the complex
    _extract_num_runs() method. This test verifies that behavior.
    """

    # Create instance without run_params (not needed anymore)
    class MockInstance:
        def notice_invalidate(self):
            pass

    instance = MockInstance()

    # Track the allocation callback response
    callback_called = {"flag": False, "response": None}

    def allocation_hook(response):
        callback_called["flag"] = True
        callback_called["response"] = response

    mgr.register(
            instance,
            allocation_ready_hook=allocation_hook,
            invalidate_cache_hook=instance.notice_invalidate,
        )

    # Create requests where first request has total_runs=150
    requests = {
        "arr1": ArrayRequest(
            shape=(10, 150),
            dtype=np.float32,
            memory="device",
            unchunkable=False,
            total_runs=150,
        ),
        "arr2": ArrayRequest(
            shape=(5, 150),
            dtype=np.float32,
            memory="device",
            unchunkable=False,
            total_runs=150,
        ),
    }

    mgr.queue_request(instance, requests)
    mgr.allocate_queue(instance)

    # Verify callback was called
    assert callback_called["flag"] is True
    response = callback_called["response"]
    assert isinstance(response, ArrayResponse)

    # Verify chunk_length matches num_runs=150 (no chunking with 1GB free)
    assert response.chunks == 1
    assert response.chunk_length == 150


def test_allocate_queue_handles_all_requests_same_total_runs(mgr):
    """Verify allocate_queue works when all requests have same total_runs.

    With the refactored implementation, all requests in a queue must have
    the same total_runs value (guaranteed by array managers). This test
    verifies that allocate_queue correctly handles this case.
    """

    # Create instance without run_params (not needed anymore)
    class MockInstance:
        def notice_invalidate(self):
            pass

    instance = MockInstance()

    # Track the allocation callback response
    callback_called = {"flag": False, "response": None}

    def allocation_hook(response):
        callback_called["flag"] = True
        callback_called["response"] = response

    mgr.register(
            instance,
            allocation_ready_hook=allocation_hook,
            invalidate_cache_hook=instance.notice_invalidate,
        )

    # Create multiple requests all with same total_runs=200
    requests = {
        "arr1": ArrayRequest(
            shape=(10, 200),
            dtype=np.float32,
            memory="device",
            unchunkable=False,
            total_runs=200,
        ),
        "arr2": ArrayRequest(
            shape=(5, 200),
            dtype=np.float32,
            memory="device",
            unchunkable=False,
            total_runs=200,
        ),
        "arr3": ArrayRequest(
            shape=(8, 200),
            dtype=np.float32,
            memory="device",
            unchunkable=False,
            total_runs=200,
        ),
    }

    mgr.queue_request(instance, requests)
    mgr.allocate_queue(instance)

    # Verify callback was called
    assert callback_called["flag"] is True
    response = callback_called["response"]
    assert isinstance(response, ArrayResponse)

    # Verify chunk_length matches num_runs=200 (no chunking with 1GB free)
    assert response.chunks == 1
    assert response.chunk_length == 200

    # Verify all arrays were allocated
    assert len(response.arr) == 3
    assert "arr1" in response.arr
    assert "arr2" in response.arr
    assert "arr3" in response.arr


@pytest.fixture(scope="session")
def stream1():
    return cuda.stream()


@pytest.fixture(scope="session")
def stream2():
    return cuda.stream()


@pytest.mark.nocudasim
def test_numba_stream_ptr(stream1):
    try:
        expected_ptr = int(stream1.handle.value)
    except AttributeError:
        expected_ptr = int(stream1.handle)
    assert _numba_stream_ptr(stream1) == expected_ptr


@pytest.mark.nocudasim
@pytest.mark.cupy
def test_cupy_stream_wrapper(stream1, stream2):
    """Verify current_cupy_stream always forwards a Numba stream.

    Pool allocations are ordered on the Numba integration stream, so the
    forwarding context manager wraps every non-default stream it is given
    (via ``Stream.from_external``).
    """
    import cupy as cp

    with current_cupy_stream(stream1) as cupy_stream:
        assert isinstance(cupy_stream.cupy_ext_stream, cp.cuda.Stream)
        assert cupy_stream.cupy_ext_stream.ptr == _numba_stream_ptr(stream1)
        assert cp.cuda.get_current_stream().ptr == _numba_stream_ptr(stream1)

    with current_cupy_stream(stream2) as cupy_stream:
        assert isinstance(cupy_stream.cupy_ext_stream, cp.cuda.Stream)
        assert cupy_stream.cupy_ext_stream.ptr == _numba_stream_ptr(stream2)
        assert cp.cuda.get_current_stream().ptr == _numba_stream_ptr(stream2)

    # Check that the default current stream is untouched
    assert cp.cuda.get_current_stream().ptr != _numba_stream_ptr(stream1)
    assert cp.cuda.get_current_stream().ptr != _numba_stream_ptr(stream2)


def test_placeholder_hooks_are_noop():
    """Test the default hook placeholders perform no operations."""
    assert placeholder_invalidate() is None
    assert placeholder_dataready(ArrayResponse()) is None


def test_numba_stream_ptr_none_stream():
    """Test _numba_stream_ptr returns None for a None stream."""
    assert _numba_stream_ptr(None) is None


def test_numba_stream_ptr_ctypes_void_p():
    """Test pointer extraction from a ctypes.c_void_p handle."""
    stream = DummyStream(ctypes.c_void_p(1234))
    assert _numba_stream_ptr(stream) == 1234


def test_numba_stream_ptr_ctypes_void_p_null():
    """Test a null ctypes.c_void_p handle yields None."""
    stream = DummyStream(ctypes.c_void_p(None))
    assert _numba_stream_ptr(stream) is None


def test_numba_stream_ptr_unconvertible_handle():
    """Test a handle that cannot be converted to int yields None."""
    stream = DummyStream("not-a-number")
    assert _numba_stream_ptr(stream) is None


@pytest.mark.nocudasim
def test_empty_pinned_real_gpu():
    """empty_pinned draws from CuPy's pinned pool on a real GPU."""
    arr = empty_pinned((4, 3), np.float32)
    assert arr.shape == (4, 3)
    assert arr.dtype == np.float32
    assert is_pinned_array(arr)


def test_instance_memory_settings_free_missing_key_warns():
    """Test free warns when the key is not in the allocations dict."""
    settings = InstanceMemorySettings()
    with pytest.warns(UserWarning, match="not found in the allocations"):
        settings.free("missing")


def test_attrs_post_init_cuda_less_environment():
    """A failed probe builds a manager holding no size."""

    class NoCudaMemoryManager(MemoryManager):
        def get_memory_info(self):
            raise ValueError("not enough values to unpack (expected 2, got 0)")

    mgr = NoCudaMemoryManager()
    assert mgr.totalmem is None
    assert mgr.pinned_max_bytes is None


def test_sizing_raises_from_the_stored_probe_error():
    """Every sizing decision re-raises the driver's own error."""

    probe_error = CudaSupportError("Error at driver init")

    class NoCudaMemoryManager(MemoryManager):
        def get_memory_info(self):
            raise probe_error

    mgr = NoCudaMemoryManager()
    with pytest.raises(NoCudaDeviceError) as excinfo:
        mgr.pinned_budget_bytes
    assert excinfo.value.__cause__ is probe_error
    with pytest.raises(NoCudaDeviceError):
        mgr.choose_host_memory_type(1024)
    with pytest.raises(NoCudaDeviceError):
        mgr.allocate_pinned_array((8,), np.float64)
    with pytest.raises(NoCudaDeviceError):
        mgr.get_available_memory("default")


def test_probe_device_sizes_a_manager_built_without_one():
    """A later device answer sizes an unsized manager."""
    total = 24 * 1024**3
    devices = []

    class LateDeviceMemoryManager(MemoryManager):
        def get_memory_info(self):
            if not devices:
                raise CudaSupportError("Error at driver init")
            return total // 2, total

    mgr = LateDeviceMemoryManager()
    assert mgr.totalmem is None

    devices.append("device")
    mgr.probe_device()

    assert mgr.totalmem == total
    assert mgr.pinned_max_bytes == total
    assert mgr._device_probe_error is None


def test_sizing_recovers_when_a_device_appears():
    """A sizing decision reprobes an unsized manager by itself."""
    total = 24 * 1024**3
    devices = []

    class LateDeviceMemoryManager(MemoryManager):
        def get_memory_info(self):
            if not devices:
                raise CudaSupportError("Error at driver init")
            return total // 2, total

    mgr = LateDeviceMemoryManager()
    assert mgr.totalmem is None

    devices.append("device")
    assert mgr.get_available_memory("default") == total // 2
    assert mgr.totalmem == total


def test_register_proportion_out_of_range(mgr, memory_client):
    """Test register raises ValueError for an out-of-range proportion."""
    inst = memory_client
    with pytest.raises(ValueError, match="Proportion must be between"):
        mgr.register(inst, proportion=1.5)


def test_set_manual_proportion_out_of_range(mgr, memory_client):
    """Test set_manual_proportion raises ValueError when out of range."""
    inst = memory_client
    with pytest.raises(ValueError, match="Proportion must be between"):
        mgr.set_manual_proportion(inst, 1.5)


def test_set_manual_limit_mode_noop_when_already_manual(mgr, memory_client):
    """Test set_manual_limit_mode is a no-op for an already-manual
    instance."""
    inst = memory_client
    mgr.register(inst)
    mgr.set_manual_limit_mode(inst, 0.3)
    assert abs(mgr.proportion(inst) - 0.3) < 1e-6
    # Second call should return early, leaving the proportion unchanged.
    mgr.set_manual_limit_mode(inst, 0.9)
    assert abs(mgr.proportion(inst) - 0.3) < 1e-6


def test_set_auto_limit_mode_converts_manual_to_auto(mgr, memory_client):
    """Test set_auto_limit_mode moves a manual instance to the auto
    pool."""
    inst = memory_client
    mgr.register(inst, proportion=0.5)
    instance_id = id(inst)
    assert instance_id in mgr._manual_pool
    mgr.set_auto_limit_mode(inst)
    assert instance_id not in mgr._manual_pool
    assert instance_id in mgr._auto_pool
    assert abs(mgr.proportion(inst) - 1.0) < 1e-6


@pytest.mark.parametrize("memory_clients", [2], indirect=True)
def test_add_manual_proportion_exceeds_total(mgr, memory_clients):
    """Test manual proportions summing above 1.0 raise ValueError."""
    inst1, inst2 = memory_clients
    mgr.register(inst1, proportion=0.6)
    with pytest.raises(ValueError, match="exceed total available memory"):
        mgr.register(inst2, proportion=0.6)


def test_add_manual_proportion_warns_low_autopool(mgr, memory_client):
    """Test a manual proportion leaving <5% free warns when the auto
    pool is empty."""
    inst = memory_client
    with pytest.warns(UserWarning, match="less than 5%"):
        mgr.register(inst, proportion=0.97)


@pytest.mark.parametrize("memory_clients", [2], indirect=True)
def test_add_auto_proportion_raises_when_pool_too_small(mgr, memory_clients):
    """Test registering an auto instance raises ValueError once the
    manual pool leaves less than the minimum auto pool size."""
    manual_inst, auto_inst = memory_clients
    with pytest.warns(UserWarning, match="less than 5%"):
        mgr.register(manual_inst, proportion=0.97)
    with pytest.raises(ValueError, match="less than"):
        mgr.register(auto_inst)


def test_get_available_memory_active_mode_warns_low_headroom(
    mgr, memory_client
):
    """Test get_available_memory warns when a group has used more
    than 95% of its allotted memory in active mode."""
    inst = memory_client
    mgr.register(inst, proportion=0.5)
    mgr.set_limit_mode("active")
    settings = mgr.registry[id(inst)]
    cap = settings.cap
    settings.add_allocation("fake", FakeAllocation(int(cap * 0.99)))
    with pytest.warns(UserWarning, match="more than 95%"):
        available = mgr.get_available_memory("default")
    free, _ = mgr.get_memory_info()
    headroom = cap - settings.allocated_bytes
    assert available == min(headroom, free)


def test_get_available_memory_active_mode_normal(mgr, memory_client):
    """Test get_available_memory returns headroom in active mode when
    below the 95% warning threshold."""
    inst = memory_client
    mgr.register(inst, proportion=0.5)
    mgr.set_limit_mode("active")
    available = mgr.get_available_memory("default")
    settings = mgr.registry[id(inst)]
    free, _ = mgr.get_memory_info()
    headroom = settings.cap - settings.allocated_bytes
    assert available == min(headroom, free)


@pytest.mark.parametrize(
    "fixed_mem_override", [{"free": 1024, "total": 1024 * 100}], indirect=True
)
def test_get_chunk_parameters_warns_and_raises_when_unfittable(
    mgr, memory_client
):
    """Test get_chunk_parameters warns when a request exceeds 20x
    VRAM and then raises when even one run cannot fit."""
    inst = memory_client
    mgr.register(inst, stream_group="test")
    requests = {
        id(inst): {
            "huge": ArrayRequest(
                shape=(1000, 1000, 1000),
                dtype=np.float32,
                memory="device",
                unchunkable=False,
                chunk_axis_index=0,
                total_runs=1000,
            ),
        }
    }
    with pytest.warns(UserWarning, match="exceeds available VRAM"):
        with pytest.raises(ValueError, match="single run"):
            mgr.get_chunk_parameters(
                requests=requests,
                axis_length=1000,
                stream_group="test",
            )


def test_is_request_chunkable_empty_shape():
    """Test is_request_chunkable returns False for a scalar shape."""
    request = ArrayRequest(shape=(), dtype=np.float32)
    assert is_request_chunkable(request) is False


def test_is_request_chunkable_no_chunk_axis():
    """Test is_request_chunkable returns False when chunk_axis_index
    is None."""
    request = ArrayRequest(
        shape=(4, 4), dtype=np.float32, chunk_axis_index=None
    )
    assert is_request_chunkable(request) is False


def test_is_request_chunkable_axis_out_of_bounds():
    """Test is_request_chunkable returns False when chunk_axis_index
    is outside the shape."""
    request = ArrayRequest(
        shape=(4, 4), dtype=np.float32, chunk_axis_index=5
    )
    assert is_request_chunkable(request) is False


def test_is_request_chunkable_degenerate_axis():
    """Test is_request_chunkable returns False when the run axis has
    length 1."""
    request = ArrayRequest(
        shape=(4, 1), dtype=np.float32, chunk_axis_index=1
    )
    assert is_request_chunkable(request) is False


def test_get_portioned_request_size():
    """Test get_portioned_request_size splits chunkable and
    unchunkable byte totals across instances."""
    requests = {
        1: {
            "chunkable": ArrayRequest(
                shape=(2, 3, 4),
                dtype=np.float32,
                chunk_axis_index=2,
                unchunkable=False,
            ),
            "unchunkable": ArrayRequest(
                shape=(2, 3, 4),
                dtype=np.float32,
                unchunkable=True,
            ),
        },
    }
    chunkable, unchunkable = get_portioned_request_size(requests)
    expected = 2 * 3 * 4 * np.dtype(np.float32).itemsize
    assert chunkable == expected
    assert unchunkable == expected


def test_replace_with_chunked_size():
    """Test replace_with_chunked_size swaps only the run axis."""
    shape = (2, 3, 4)
    newshape = replace_with_chunked_size(shape, axis_index=1, chunked_size=9)
    assert newshape == (2, 9, 4)


def test_attrs_post_init_unexpected_exception():
    """A probe failure that is not device absence stays visible."""

    class BrokenMemoryManager(MemoryManager):
        def get_memory_info(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        BrokenMemoryManager()


def test_totalmem_is_not_a_constructor_argument():
    """A caller cannot claim a device size the device did not report."""
    with pytest.raises(TypeError):
        MemoryManager(totalmem=1024)


def test_set_auto_limit_mode_noop_when_already_auto(mgr, memory_client):
    """Test set_auto_limit_mode is a no-op for an already-auto
    instance."""
    inst = memory_client
    mgr.register(inst)
    instance_id = id(inst)
    proportion_before = mgr.proportion(inst)
    mgr.set_auto_limit_mode(inst)
    assert instance_id in mgr._auto_pool
    assert mgr.proportion(inst) == proportion_before


def test_add_allocation_overwrites_existing_key():
    """Test add_allocation frees a previous allocation before
    overwriting it with a new one under the same key."""
    settings = InstanceMemorySettings()
    arr1 = np.zeros((4,), dtype=np.float32)
    arr2 = np.ones((4,), dtype=np.float32)
    settings.add_allocation("foo", arr1)
    settings.add_allocation("foo", arr2)
    assert settings.allocations["foo"] is arr2


def test_create_host_array_with_like(mgr):
    """Test create_host_array copies data from the like source."""
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    arr = mgr.create_host_array(shape=(2, 3), dtype=np.float32, like=source)
    np.testing.assert_array_equal(arr, source)


def test_sync_stream(registered_mgr, registered_instance):
    """Test sync_stream synchronizes the instance's CUDA stream."""
    mgr = registered_mgr
    instance = registered_instance
    mgr.sync_stream(instance)  # Should not raise


@pytest.mark.parametrize("memory_clients", [2], indirect=True)
def test_allocate_queue_notifies_notaries(mgr, memory_clients):
    """Test allocate_queue notifies peers that never queued a request
    (notaries) with an empty ArrayResponse."""
    inst1, inst2 = memory_clients
    responses = {}

    def hook1(response):
        responses["inst1"] = response

    def hook2(response):
        responses["inst2"] = response

    mgr.register(
        inst1,
        allocation_ready_hook=hook1,
        invalidate_cache_hook=inst1.notice_invalidate,
        stream_group="grp",
    )
    mgr.register(
        inst2,
        allocation_ready_hook=hook2,
        invalidate_cache_hook=inst2.notice_invalidate,
        stream_group="grp",
    )

    requests = {
        "arr1": ArrayRequest(
            shape=(2, 2, 2),
            dtype=np.float32,
            memory="device",
            total_runs=2,
        )
    }
    mgr.queue_request(inst1, requests)
    mgr.allocate_queue(inst1)

    assert "inst2" in responses
    assert responses["inst2"].arr == {}
    assert responses["inst2"].chunks == responses["inst1"].chunks


@pytest.mark.parametrize(
    "fixed_mem_override",
    [{"free": 100 * 1024**2, "total": 8 * 1024**3}],
    indirect=True,
)
def test_get_chunk_parameters_unchunkable_alone_exceeds_memory(
    mgr, memory_client
):
    """Test get_chunk_parameters raises when unchunkable arrays alone
    exceed available memory, distinct from the all-unchunkable case
    (a genuinely chunkable array is also present)."""
    inst = memory_client
    mgr.register(inst, stream_group="test")

    requests = {
        id(inst): {
            "big_unchunkable": ArrayRequest(
                shape=(512, 512, 256),
                dtype=np.float32,
                memory="device",
                unchunkable=True,
                total_runs=1,
            ),
            "small_chunkable": ArrayRequest(
                shape=(4, 4, 10),
                dtype=np.float32,
                memory="device",
                unchunkable=False,
                chunk_axis_index=2,
                total_runs=10,
            ),
        }
    }

    with pytest.raises(ValueError, match="Unchunkable arrays require"):
        mgr.get_chunk_parameters(
            requests=requests,
            axis_length=10,
            stream_group="test",
        )


@pytest.mark.parametrize(
    "fixed_mem_override",
    [{"free": 10 * 1024**2, "total": 8 * 1024**3}],
    indirect=True,
)
def test_get_chunk_parameters_computes_chunk_size_when_eligible(
    mgr, memory_client
):
    """Test get_chunk_parameters returns a positive chunk size and
    chunk count when the request exceeds available memory but each
    run still fits."""
    inst = memory_client
    mgr.register(inst, stream_group="test")

    requests = {
        id(inst): {
            "arr": ArrayRequest(
                shape=(2, 5_000_000),
                dtype=np.float32,
                memory="device",
                unchunkable=False,
                chunk_axis_index=1,
                total_runs=5_000_000,
            ),
        }
    }

    chunk_length, num_chunks = mgr.get_chunk_parameters(
        requests=requests,
        axis_length=5_000_000,
        stream_group="test",
    )
    assert 0 < chunk_length < 5_000_000
    assert num_chunks > 1


@pytest.mark.parametrize(
    "fixed_mem_override",
    [{"free": 1024**3, "total": 8 * 1024**3}],
    indirect=True,
)
def test_active_cap_chunks_with_free_device_memory(mgr, memory_client):
    """An active cap chunks a request that fits physical VRAM."""
    inst = memory_client
    mgr.register(inst, proportion=0.01, stream_group="capped")
    mgr.set_limit_mode("active")
    requests = {
        id(inst): {
            "arr": ArrayRequest(
                shape=(2, 20_000_000),
                dtype=np.float32,
                chunk_axis_index=1,
                total_runs=20_000_000,
            )
        }
    }

    chunk_length, chunks = mgr.get_chunk_parameters(
        requests, 20_000_000, "capped"
    )

    assert 0 < chunk_length < 20_000_000
    assert chunks > 1


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "fixed_mem_override",
    [{"free": 1024**3, "total": 8 * 1024**3}],
    indirect=True,
)
def test_active_cap_chunking_leaves_unrelated_stream_running(
    mgr, memory_client, start_cuda_busy_work
):
    """Cap-driven chunking does not synchronize another stream."""
    inst = memory_client
    mgr.register(inst, proportion=0.01, stream_group="capped")
    mgr.set_limit_mode("active")
    requests = {
        id(inst): {
            "arr": ArrayRequest(
                shape=(2, 20_000_000),
                dtype=np.float32,
                chunk_axis_index=1,
                total_runs=20_000_000,
            )
        }
    }
    work, stream, done, release = start_cuda_busy_work()

    try:
        _, chunks = mgr.get_chunk_parameters(
            requests, 20_000_000, "capped"
        )

        assert chunks > 1
        assert not done.query()
    finally:
        release()
        stream.synchronize()
        assert work.copy_to_host()[0] > 0


@pytest.mark.parametrize(
    "fixed_mem_override",
    [{"free": 1024, "total": 8 * 1024**3}],
    indirect=True,
)
def test_allocation_pressure_does_not_run_cyclic_gc(mgr, memory_client):
    """Allocation pressure chunks without collecting client cycles."""

    class CyclicClient:
        def __init__(self):
            self._cycle = self

        def notice_invalidate(self):
            pass

    gc.disable()
    try:
        holder = CyclicClient()
        mgr.register(
            holder,
            stream_group="pressure_holder",
            invalidate_cache_hook=holder.notice_invalidate,
        )
        mgr.queue_request(
            holder,
            {
                "buf": ArrayRequest(
                    shape=(4, 4, 4),
                    dtype=np.float32,
                    memory="device",
                    total_runs=4,
                ),
            },
        )
        mgr.allocate_queue(holder)
        dead_id = id(holder)
        assert mgr.registry[dead_id].allocated_bytes > 0

        del holder
        assert dead_id in mgr.registry

        requester = memory_client
        mgr.register(
            requester,
            stream_group="pressure_requester",
            invalidate_cache_hook=requester.notice_invalidate,
        )
        mgr.queue_request(
            requester,
            {
                "big": ArrayRequest(
                    shape=(2, 2, 512),
                    dtype=np.float32,
                    memory="device",
                    chunk_axis_index=2,
                    total_runs=512,
                ),
            },
        )
        mgr.allocate_queue(requester)

        assert dead_id in mgr.registry
        cache_key = (
            "pressure_requester",
            mgr.registry[id(requester)].owner_id,
        )
        _, chunks = mgr._group_chunk_parameters[cache_key]
        assert chunks > 1
    finally:
        gc.enable()


def test_owner_work_state_tracks_submission(mgr, memory_client):
    """Owner work stays active until its completion event finishes."""
    inst = memory_client
    mgr.register(inst, stream_group="flags")
    settings = mgr.registry[id(inst)]
    assert settings.work_complete
    mgr.begin_work(inst)
    assert settings.submitting
    assert not settings.work_complete
    mgr.end_work(inst, mgr.get_stream(inst))
    assert not settings.submitting
    mgr.get_stream(inst).synchronize()
    assert settings.work_complete


@pytest.mark.parametrize(
    "fixed_mem_override",
    [{"free": 2048, "total": 8 * 1024**3}],
    indirect=True,
)
@pytest.mark.parametrize("memory_clients", [3], indirect=True)
def test_pressure_evicts_idle_complete_owner(mgr, memory_clients):
    """Physical pressure evicts idle completed owners.

    The idle client's allocations are freed and its invalidate hook
    runs (so it reallocates on its next solve); a client marked
    active keeps its allocations.
    """
    idle, busy, requester = memory_clients
    mgr.register(
        idle,
        invalidate_cache_hook=idle.notice_invalidate,
        stream_group="evict_idle",
    )
    mgr.register(
        busy,
        invalidate_cache_hook=busy.notice_invalidate,
        stream_group="evict_busy",
    )
    for instance in (idle, busy):
        mgr.queue_request(
            instance,
            {
                "buf": ArrayRequest(
                    shape=(4, 4, 4),
                    dtype=np.float32,
                    memory="device",
                    total_runs=4,
                ),
            },
        )
        mgr.allocate_queue(instance)
        assert mgr.registry[id(instance)].allocated_bytes > 0

    # Sentinel values: notice_invalidate resets proportion to None.
    idle.proportion = "armed"
    busy.proportion = "armed"
    mgr.begin_work(busy)

    mgr.register(
        requester,
        stream_group="evict_requester",
        invalidate_cache_hook=requester.notice_invalidate,
    )
    mgr.queue_request(
        requester,
        {
            "big": ArrayRequest(
                shape=(2, 2, 1024),
                dtype=np.float32,
                memory="device",
                chunk_axis_index=2,
                total_runs=1024,
            ),
        },
    )
    mgr.allocate_queue(requester)

    assert mgr.registry[id(idle)].allocated_bytes == 0
    assert idle.proportion is None
    assert mgr.registry[id(busy)].allocated_bytes > 0
    assert busy.proportion == "armed"


@pytest.mark.parametrize(
    "fixed_mem_override",
    [{"free": 2048, "total": 8 * 1024**3}],
    indirect=True,
)
@pytest.mark.parametrize("memory_clients", [2], indirect=True)
def test_inflight_owner_is_not_evicted(mgr, memory_clients):
    """Pressure keeps allocations whose CUDA work is still in flight."""
    inflight, requester = memory_clients
    mgr.register(
        inflight,
        invalidate_cache_hook=inflight.notice_invalidate,
        stream_group="inflight",
    )
    mgr.queue_request(
        inflight,
        {
            "buf": ArrayRequest(
                shape=(4, 4, 4),
                dtype=np.float32,
                memory="device",
                total_runs=4,
            )
        },
    )
    mgr.allocate_queue(inflight)
    mgr.begin_work(inflight)

    mgr.register(
        requester,
        stream_group="protected_requester",
        invalidate_cache_hook=requester.notice_invalidate,
    )
    mgr.queue_request(
        requester,
        {
            "big": ArrayRequest(
                shape=(2, 2, 1024),
                dtype=np.float32,
                memory="device",
                total_runs=1024,
            )
        },
    )
    mgr.allocate_queue(requester)

    assert mgr.registry[id(inflight)].allocated_bytes > 0


def test_queue_request_requires_live_invalidate_hook(mgr, memory_client):
    """Requesting allocations without an invalidate hook is an error.

    Allocation holders are eviction candidates, so a registration
    that cannot drop its device handles must be rejected before it
    ever holds any.
    """
    mgr.register(memory_client, stream_group="no_hook")
    with pytest.raises(ValueError, match="invalidate_cache_hook"):
        mgr.queue_request(
            memory_client,
            {
                "buf": ArrayRequest(
                    shape=(4, 4, 4),
                    dtype=np.float32,
                    memory="device",
                    total_runs=4,
                )
            },
        )


@pytest.mark.parametrize(
    "fixed_mem_override",
    [{"free": 2048, "total": 8 * 1024**3}],
    indirect=True,
)
@pytest.mark.parametrize("memory_clients", [3], indirect=True)
def test_pressure_evicts_only_required_lru_owner(mgr, memory_clients):
    """Eviction stops after the oldest owner covers the shortage."""
    owner_a, owner_b, requester = memory_clients
    owners = (owner_a, owner_b)
    for index, owner in enumerate(owners):
        mgr.register(
            owner,
            stream_group=f"lru_{index}",
            invalidate_cache_hook=owner.notice_invalidate,
        )
        mgr.queue_request(
            owner,
            {
                "buf": ArrayRequest(
                    shape=(4, 4, 4),
                    dtype=np.float32,
                    memory="device",
                    total_runs=4,
                )
            },
        )
        mgr.allocate_queue(owner)
        mgr.begin_work(owner)
        stream = mgr.get_stream(owner)
        mgr.end_work(owner, stream)
        stream.synchronize()

    mgr.register(
        requester,
        stream_group="lru_requester",
        invalidate_cache_hook=requester.notice_invalidate,
    )
    mgr.queue_request(
        requester,
        {
            "big": ArrayRequest(
                shape=(1, 1, 574),
                dtype=np.float32,
                memory="device",
                total_runs=574,
            )
        },
    )
    mgr.allocate_queue(requester)

    assert mgr.registry[id(owners[0])].allocated_bytes == 0
    assert mgr.registry[id(owners[1])].allocated_bytes > 0


def test_total_system_ram_reports_positive():
    """The RAM probe returns a positive byte count on CI platforms."""
    ram_total = total_system_ram()
    assert ram_total is not None
    assert ram_total > 0


def test_available_system_ram_reports_reclaimable_memory():
    """The availability probe reports a plausible fraction of RAM."""
    available = available_system_ram()
    assert 0 < available <= total_system_ram()


def test_choose_host_memory_type_applies_policy(mgr, tmp_path):
    """The chooser maps sizes to pinned, pageable, and disk backing."""
    mgr.pinned_max_bytes = 256

    assert mgr.choose_host_memory_type(128, 1024) == "pinned"
    assert mgr.choose_host_memory_type(512, 1024) == "host"
    assert mgr.choose_host_memory_type(4096, 1024) == "memmap"
    assert (
        mgr.choose_host_memory_type(128, 1024, allow_pinned=False)
        == "host"
    )

    small = mgr.create_host_array((4, 4), np.float64, "host")
    assert not isinstance(small, np.memmap)

    big_type = mgr.choose_host_memory_type(64 * 64 * 8, 1024)
    big = mgr.create_host_array(
        (64, 64), np.float64, big_type, spill_directory=str(tmp_path)
    )
    assert isinstance(big, np.memmap)
    assert (np.asarray(big) == 0.0).all()
    assert len(list(tmp_path.iterdir())) == 1

    source = np.arange(16, dtype=np.float64).reshape(4, 4)
    explicit = mgr.create_host_array(
        (4, 4),
        np.float64,
        "memmap",
        like=source,
        spill_directory=str(tmp_path),
    )
    assert isinstance(explicit, np.memmap)
    assert np.array_equal(np.asarray(explicit), source)

    del big, explicit
    gc.collect()
    assert len(list(tmp_path.iterdir())) == 0


def test_choose_host_memory_type_default_threshold(mgr):
    """An unset threshold spills only above the RAM fraction."""
    ram_threshold = int(HOST_SPILL_FRACTION * total_system_ram())
    assert mgr.choose_host_memory_type(ram_threshold + 1) == "memmap"
    mgr.pinned_max_bytes = 256
    assert mgr.choose_host_memory_type(128) == "pinned"


def test_choose_host_memory_type_host_only_without_device():
    """Disk and pageable backing decide without touching a device."""

    class NoCudaMemoryManager(MemoryManager):
        def get_memory_info(self):
            raise CudaSupportError("Error at driver init")

    mgr = NoCudaMemoryManager()
    assert mgr.choose_host_memory_type(2048, 1024) == "memmap"
    assert (
        mgr.choose_host_memory_type(128, 1024, allow_pinned=False)
        == "host"
    )


def test_pinned_ceiling_defaults_to_vram(mgr):
    """An unset pinned ceiling resolves to total device memory."""
    assert mgr.pinned_max_bytes == mgr.totalmem


def test_pinned_budget_counts_all_live_allocations(mgr):
    """Reservations that individually fit are refused cumulatively."""
    mgr.pinned_max_bytes = 1024
    first = mgr.allocate_pinned_array((96,), np.float64)
    assert first is not None
    assert mgr.pinned_live_bytes == 768
    second = mgr.allocate_pinned_array((96,), np.float64)
    assert second is None
    assert mgr.pinned_live_bytes == 768
    fallback = mgr.create_host_array((96,), np.float64, "pinned")
    assert fallback.shape == (96,)
    assert mgr.pinned_live_bytes == 768
    small = mgr.allocate_pinned_array((16,), np.float64)
    assert small is not None
    assert mgr.pinned_live_bytes == 768 + 128


def test_pinned_release_returns_budget(mgr):
    """Collected arrays retire to retained; pressure reclaims them."""
    mgr.pinned_max_bytes = 1024
    array = mgr.allocate_pinned_array((96,), np.float64)
    assert array is not None
    del array
    gc.collect()
    assert mgr.pinned_live_bytes == 0
    assert mgr.pinned_retained_bytes == 768
    again = mgr.allocate_pinned_array((96,), np.float64)
    assert again is not None
    assert mgr.pinned_live_bytes == 768
    assert mgr.pinned_retained_bytes == 0


def test_pinned_view_keeps_reservation(mgr):
    """A surviving view keeps the owning array's bytes reserved."""
    mgr.pinned_max_bytes = 1024
    array = mgr.allocate_pinned_array((96,), np.float64)
    view = array[10:20]
    del array
    gc.collect()
    assert mgr.pinned_live_bytes == 768
    del view
    gc.collect()
    assert mgr.pinned_live_bytes == 0


def test_concurrent_pinned_reservations_respect_budget(mgr):
    """Racing reservations never overshoot the cumulative budget."""
    from threading import Barrier, Thread

    mgr.pinned_max_bytes = 8 * 64
    granted = []
    barrier = Barrier(16)

    def worker():
        barrier.wait()
        array = mgr.allocate_pinned_array((8,), np.float64)
        if array is not None:
            granted.append(array)

    threads = [Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(granted) == 8
    assert mgr.pinned_live_bytes == 8 * 64


def test_forced_pinned_reservation_exceeds_budget(mgr):
    """A forced reservation records past the budget for pool liveness."""
    mgr.pinned_max_bytes = 64
    array = mgr.allocate_pinned_array((96,), np.float64, force=True)
    assert array is not None
    assert mgr.pinned_live_bytes == 768


def test_pinned_budget_capped_by_ram_fraction(mgr):
    """The enforced budget never exceeds the spill fraction of RAM."""
    mgr.pinned_max_bytes = 2**62
    assert mgr.pinned_budget_bytes <= int(
        HOST_SPILL_FRACTION * total_system_ram()
    )


def test_release_host_array_reports_unlink_failure(mgr, tmp_path):
    """Explicit spill cleanup reports a filesystem failure."""
    array = mgr.create_host_array(
        (4,), np.float64, "memmap", spill_directory=str(tmp_path)
    )
    path = Path(array._cubie_spill_path)
    cleanup = array._cubie_spill_cleanup
    array._mmap.close()
    path.unlink()
    path.mkdir()
    try:
        with pytest.raises(OSError, match="Could not remove spill file"):
            mgr.release_host_array(array)
    finally:
        cleanup.detach()
        path.rmdir()


@pytest.mark.nocudasim
def test_host_spill_does_not_wait_for_unrelated_stream(
    tmp_path, start_cuda_busy_work
):
    """Host spill setup leaves unrelated CUDA work running."""
    work, stream, done, release = start_cuda_busy_work()
    manager = MemoryManager()
    try:
        memory_type = manager.choose_host_memory_type(32 * 4, 1)
        arr = manager.create_host_array(
            (32,), np.float32, memory_type, spill_directory=tmp_path
        )
        assert isinstance(arr, np.memmap)
        assert not done.query()
    finally:
        release()
        stream.synchronize()
        assert work.copy_to_host()[0] > 0


@pytest.mark.nocudasim
def test_legacy_default_stream_sync_is_stream_local(
    memory_client, start_cuda_busy_work
):
    """Legacy stream zero does not wait for unrelated CUDA work."""
    manager = MemoryManager()
    owner = memory_client
    manager.register(owner)
    ready = cuda.event()
    ready.record(manager.get_stream(owner))
    ready.synchronize()
    work, stream, done, release = start_cuda_busy_work()
    try:
        manager.sync_stream(owner, stream=0)
        assert not done.query()
    finally:
        release()
        stream.synchronize()
        assert work.copy_to_host()[0] > 0


class TestDeadInstanceRelease:
    """The registry releases instances once they are collected.

    Clients are constructed inline rather than taken from fixtures:
    pytest's fixture cache holds a strong reference for the test's
    lifetime, which would prevent the collection these tests assert.
    """

    def test_registry_does_not_pin_instances(self, mgr):
        """Bound-method hooks leave the instance collectable."""
        inst = MemoryClient()
        mgr.register(
            inst,
            proportion=0.4,
            invalidate_cache_hook=inst.notice_invalidate,
            allocation_ready_hook=inst.notice_allocation,
        )
        instance_id = id(inst)
        ref = weakref.ref(inst)
        del inst
        gc.collect()
        assert ref() is None
        assert mgr.manual_pool_proportion == 0.0
        assert instance_id not in mgr.registry
        assert instance_id not in mgr._manual_pool

    def test_dead_manual_instances_release_proportion(self, mgr):
        """Collected manual instances free their reservations.

        Three successive 0.5 reservations exceed the whole memory if
        the dead instances' proportions are retained.
        """
        keeper = MemoryClient()
        mgr.register(keeper)
        for _ in range(3):
            inst = MemoryClient()
            mgr.register(inst, proportion=0.5)
            del inst
            gc.collect()
        assert mgr.manual_pool_proportion == 0.0
        survivor = MemoryClient()
        mgr.register(survivor, proportion=0.9)
        assert abs(mgr.manual_pool_proportion - 0.9) < 1e-9
        assert id(survivor) in mgr._manual_pool

    def test_dead_auto_instance_leaves_pool(self, mgr):
        """A collected auto instance stops diluting the auto pool."""
        keeper = MemoryClient()
        mgr.register(keeper)
        inst = MemoryClient()
        mgr.register(inst)
        del inst
        gc.collect()
        assert abs(mgr.auto_pool_proportion - 1.0) < 1e-9
        assert mgr.registry[id(keeper)].proportion == 1.0

    @pytest.mark.parametrize(
        "memory_client_override", [{"proportion": 0.3}], indirect=True
    )
    def test_bound_method_hooks_compare_equal(self, mgr, memory_client):
        """Registered hooks compare equal to the source bound methods."""
        inst = memory_client
        mgr.register(
            inst,
            invalidate_cache_hook=inst.notice_invalidate,
            allocation_ready_hook=inst.notice_allocation,
        )
        settings = mgr.registry[id(inst)]
        assert settings.invalidate_hook == inst.notice_invalidate
        assert settings.allocation_ready_hook == inst.notice_allocation
        settings.invalidate_hook()
        assert inst.proportion is None

    def test_dead_hooks_are_noops(self, mgr):
        """Hooks of a collected instance do nothing when invoked."""
        inst = MemoryClient()
        mgr.register(inst, invalidate_cache_hook=inst.notice_invalidate)
        del inst
        gc.collect()
        mgr.invalidate_all()

    def test_non_weakrefable_instance_registers(self, mgr):
        """Objects without weakref support register and persist.

        Such instances carry no instance_ref, so the purge treats
        them as permanently alive.
        """

        class Slotted:
            __slots__ = ()

        inst = Slotted()
        mgr.register(inst, proportion=0.2)
        assert mgr.registry[id(inst)].instance_ref is None
        assert abs(mgr.manual_pool_proportion - 0.2) < 1e-9
        assert id(inst) in mgr.registry
