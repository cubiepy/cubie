from os import environ
import pytest

import attrs
import numpy as np

from cubie.batchsolving.arrays.BaseArrayManager import (
    BaseArrayManager,
    ArrayContainer,
    ManagedArray,
)
from cubie.memory.array_requests import ArrayResponse, ArrayRequest
from cubie.memory.mem_manager import MemoryManager
from cubie.outputhandling.output_sizes import BatchOutputSizes
from cubie.cuda_simsafe import DeviceNDArray
from numpy import float32 as np_float32

if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from numpy import zeros as pinned_array
    from numpy import zeros as device_array
else:
    from cubie.cuda_simsafe import cuda

    pinned_array = cuda.pinned_array
    device_array = cuda.device_array


@attrs.define
class ConcreteArrayManager(BaseArrayManager):
    """Concrete implementation of BaseArrayManager"""

    def finalise(self, chunk_index):
        return chunk_index

    def initialise(self, chunk_index):
        return chunk_index

    def update(self):
        return


@attrs.define(slots=False)
class TestArrays(ArrayContainer):
    state: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            default_shape=(1, 1, 1),
            memory_type="host",
        )
    )
    observables: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            default_shape=(1, 1, 1),
            memory_type="host",
        )
    )
    state_summaries: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            default_shape=(1, 1, 1),
            memory_type="host",
        )
    )
    observable_summaries: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            default_shape=(1, 1, 1),
            memory_type="host",
        )
    )


@attrs.define(slots=False)
class TestArraysSimple(ArrayContainer):
    arr1: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            default_shape=(1, 1, 1),
            memory_type="host",
        )
    )
    arr2: ManagedArray = attrs.field(
        factory=lambda: ManagedArray(
            dtype=np.float32,
            default_shape=(1, 1, 1),
            memory_type="host",
        )
    )


@pytest.fixture(scope="function")
def arraytest_overrides(request):
    if hasattr(request, "param"):
        return request.param
    return {}


@pytest.fixture(scope="function")
def arraytest_settings(arraytest_overrides):
    stride_template = ("time", "variable", "run")
    settings = {
        "hostshape1": (4, 3, 4),
        "hostshape2": (4, 3, 4),
        "hostshape3": (2, 3, 4),
        "hostshape4": (2, 3, 4),
        "devshape1": (4, 3, 4),
        "devshape2": (4, 3, 4),
        "devshape3": (2, 3, 4),
        "devshape4": (2, 3, 4),
        "chunks": 2,
        "dtype": np.float32,
        "memory": "device",
        "stride_tuple": stride_template,
        "_stride_order": {
            "arr1": stride_template,
            "arr2": stride_template,
            "state": stride_template,
            "observables": stride_template,
            "state_summaries": stride_template,
            "observable_summaries": stride_template,
        },
        "stream_group": "default",
        "memory_proportion": None,
    }
    settings.update(arraytest_overrides)

    # Calculate derived shapes for summary arrays (time_dim - 2)
    settings["hostshape3"] = (
        max(0, settings["hostshape1"][0] - 2),
        settings["hostshape1"][1],
        settings["hostshape1"][2],
    )
    settings["hostshape4"] = (
        max(0, settings["hostshape2"][0] - 2),
        settings["hostshape2"][1],
        settings["hostshape2"][2],
    )
    settings["devshape3"] = (
        max(0, settings["devshape1"][0] - 2),
        settings["devshape1"][1],
        settings["devshape1"][2],
    )
    settings["devshape4"] = (
        max(0, settings["devshape2"][0] - 2),
        settings["devshape2"][1],
        settings["devshape2"][2],
    )

    return settings


@pytest.fixture(scope="function")
def hostarrays(arraytest_settings):
    if arraytest_settings["memory"] == "pinned":
        empty = pinned_array
    else:
        empty = np.zeros
    host_memory = (
        "pinned" if arraytest_settings["memory"] == "pinned" else "host"
    )
    host = TestArraysSimple(
        arr1=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type=host_memory,
            stride_order=("", "", "run"),
            num_runs=arraytest_settings["hostshape1"][2],
        ),
        arr2=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type=host_memory,
            stride_order=("", "", "run"),
            num_runs=arraytest_settings["hostshape2"][2],
        ),
    )
    host.arr1.array = empty(
        arraytest_settings["hostshape1"], dtype=arraytest_settings["dtype"]
    )
    host.arr2.array = empty(
        arraytest_settings["hostshape2"], dtype=arraytest_settings["dtype"]
    )
    return host


@pytest.fixture(scope="function")
def devarrays(arraytest_settings, hostarrays):
    if arraytest_settings["memory"] == "pinned":
        devarr1 = DeviceNDArray(hostarrays.arr1, arraytest_settings["dtype"])
        devarr2 = DeviceNDArray(hostarrays.arr2, arraytest_settings["dtype"])
    else:
        devarr1 = device_array(
            arraytest_settings["devshape1"], arraytest_settings["dtype"]
        )
        devarr2 = device_array(
            arraytest_settings["devshape2"], arraytest_settings["dtype"]
        )
    device_memory = arraytest_settings["memory"]
    device = TestArraysSimple(
        arr1=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type=device_memory,
            stride_order=("", "", "run"),
            num_runs=devarr1.shape[2],
        ),
        arr2=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type=device_memory,
            stride_order=("", "", "run"),
            num_runs=devarr1.shape[2],
        ),
    )
    device.arr1.array = devarr1
    device.arr2.array = devarr2
    return device


@pytest.fixture(scope="function")
def test_memory_manager():
    """Create an actual MemoryManager instance for testing"""
    # Create a memory manager in passive mode to avoid CUDA context issues
    mem_mgr = MemoryManager(mode="passive")
    return mem_mgr


@pytest.fixture(scope="function")
def test_arrmgr(
    hostarrays,
    devarrays,
    precision,
    arraytest_settings,
    test_memory_manager,
    batch_output_sizes,
):
    return ConcreteArrayManager(
        precision=precision,
        sizes=batch_output_sizes,
        host=hostarrays,
        device=devarrays,
        stream_group=arraytest_settings["stream_group"],
        memory_proportion=arraytest_settings["memory_proportion"],
        memory_manager=test_memory_manager,
    )


@pytest.fixture(scope="function")
def allocation_response(arraytest_settings, precision):
    """Create a test ArrayResponse based on arraytest_settings for consistent
    parametrized testing
    """
    # Create arrays that match the settings
    if arraytest_settings["memory"] == "pinned":
        create_func = pinned_array
    else:
        create_func = device_array

    mock_arrays = {
        "arr1": create_func(arraytest_settings["devshape1"], dtype=precision),
        "arr2": create_func(arraytest_settings["devshape2"], dtype=precision),
    }
    return ArrayResponse(arr=mock_arrays, chunks=1)


@pytest.fixture(scope="function")
def array_requests(arraytest_settings, precision):
    """Create ArrayRequest objects based on arraytest_settings"""
    return {
        "arr1": ArrayRequest(
            shape=arraytest_settings["devshape1"],
            dtype=precision,
            memory=arraytest_settings["memory"],
            chunk_axis_index=2,
            total_runs=arraytest_settings["devshape2"][2],
        ),
        "arr2": ArrayRequest(
            shape=arraytest_settings["devshape2"],
            dtype=precision,
            memory=arraytest_settings["memory"],
            chunk_axis_index=2,
            total_runs=arraytest_settings["devshape2"][2],
        ),
    }


@pytest.fixture(scope="function")
def array_requests_sized(arraytest_settings, precision):
    """Create ArrayRequest objects based on arraytest_settings"""
    return {
        "state": ArrayRequest(
            shape=arraytest_settings["devshape1"],
            dtype=precision,
            memory=arraytest_settings["memory"],
            chunk_axis_index=2,
            total_runs=arraytest_settings["devshape2"][2],
        ),
        "observables": ArrayRequest(
            shape=arraytest_settings["devshape2"],
            dtype=precision,
            memory=arraytest_settings["memory"],
            chunk_axis_index=2,
            total_runs=arraytest_settings["devshape2"][2],
        ),
    }


@pytest.fixture(scope="function")
def second_arrmgr(
    test_memory_manager, precision, arraytest_settings, batch_output_sizes
):
    """Create a second array manager for testing grouping behavior"""
    # Create simple arrays for the second manager
    host_arrays = TestArraysSimple(
        arr1=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type="host",
            stride_order=("", "", "run"),
            num_runs=arraytest_settings["hostshape1"][2],
        ),
        arr2=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type="host",
            stride_order=("", "", "run"),
            num_runs=arraytest_settings["hostshape2"][2],
        ),
    )
    host_arrays.arr1.array = np.zeros(
        arraytest_settings["hostshape1"], dtype=arraytest_settings["dtype"]
    )
    host_arrays.arr2.array = np.zeros(
        arraytest_settings["hostshape2"], dtype=arraytest_settings["dtype"]
    )
    dev_arrays = TestArraysSimple(
        arr1=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type=arraytest_settings["memory"],
            stride_order=("", "", "run"),
            num_runs=arraytest_settings["devshape1"][2],
        ),
        arr2=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type=arraytest_settings["memory"],
            stride_order=("", "", "run"),
            num_runs=arraytest_settings["devshape2"][2],
        ),
    )
    dev_arrays.arr1.array = device_array(
        arraytest_settings["devshape1"], arraytest_settings["dtype"]
    )
    dev_arrays.arr2.array = device_array(
        arraytest_settings["devshape2"], arraytest_settings["dtype"]
    )

    return ConcreteArrayManager(
        precision=precision,
        sizes=batch_output_sizes,
        host=host_arrays,
        device=dev_arrays,
        stream_group=arraytest_settings["stream_group"],
        memory_proportion=arraytest_settings["memory_proportion"],
        memory_manager=test_memory_manager,
    )


class TestArrayContainer:
    """Test the ArrayContainer base class"""

    def test_attach_existing_label(self):
        """Test attaching an array to an existing label"""
        container = TestArraysSimple()
        test_array = np.array([1, 2, 3])

        container.attach("arr1", test_array)
        assert container.arr1.array is test_array

    def test_attach_nonexistent_label(self):
        """Test attaching an array to a non-existent label raises warning"""
        container = TestArraysSimple()
        test_array = np.array([1, 2, 3])

        with pytest.warns(
            UserWarning,
            match="Device array with label 'nonexistent' does not exist",
        ):
            container.attach("nonexistent", test_array)

    def test_delete_all(self):
        """Test deleting all arrays"""
        test_array1 = np.array([1, 2, 3])
        test_array2 = np.array([4, 5, 6])
        container = TestArraysSimple()
        container.arr1.array = test_array1
        container.arr2.array = test_array2

        container.delete_all()
        assert container.arr1.array is None
        assert container.arr2.array is None


class TestBaseArrayManager:
    """Test the BaseArrayManager class"""

    def test_initialization(
        self, test_arrmgr, precision, arraytest_settings, test_memory_manager
    ):
        """Test proper initialization of BaseArrayManager"""
        assert test_arrmgr._precision == precision
        assert isinstance(test_arrmgr.device, ArrayContainer)
        assert isinstance(test_arrmgr.host, ArrayContainer)
        assert test_arrmgr._stream_group == arraytest_settings["stream_group"]
        assert (
            test_arrmgr._memory_proportion
            == arraytest_settings["memory_proportion"]
        )
        assert test_arrmgr._needs_reallocation
        assert test_arrmgr._needs_overwrite == []
        assert test_arrmgr._memory_manager is test_memory_manager

        # Check that the instance was registered with the memory manager
        instance_id = id(test_arrmgr)
        assert instance_id in test_memory_manager.registry

    def test_register_with_memory_manager(
        self, test_arrmgr, test_memory_manager
    ):
        """Test registration with memory manager using the test_arrmgr fixture
        """
        instance_id = id(test_arrmgr)
        assert instance_id in test_memory_manager.registry

        settings = test_memory_manager.registry[instance_id]
        assert settings.invalidate_hook == test_arrmgr._invalidate_hook
        assert (
            settings.allocation_ready_hook
            == test_arrmgr._on_allocation_complete
        )

        # Check stream group assignment
        assert (
            test_memory_manager.get_stream_group(test_arrmgr)
            == test_arrmgr._stream_group
        )

    def test_on_allocation_complete(self, test_arrmgr, allocation_response):
        """Test allocation completion callback"""
        # Setup test data - arrays need to be in reallocation list to be
        # attached
        test_arrmgr._needs_reallocation = ["arr1", "arr2"]
        test_arrmgr._needs_overwrite = ["arr1"]

        test_arrmgr._on_allocation_complete(allocation_response)

        # Check that arrays were attached (should be different from originals)
        assert test_arrmgr.device.arr1.array is allocation_response.arr["arr1"]
        assert test_arrmgr.device.arr2.array is allocation_response.arr["arr2"]

        # Check that lists were cleared
        assert test_arrmgr._needs_reallocation == []

    def test_request_allocation_auto(
        self, test_arrmgr, second_arrmgr, array_requests
    ):
        """Test automatic allocation request when grouped"""
        # Change both managers to the same non-default group to make them
        # grouped
        test_arrmgr._memory_manager.change_stream_group(
            test_arrmgr, "test_group"
        )
        test_arrmgr._memory_manager.change_stream_group(
            second_arrmgr, "test_group"
        )

        # Now they should be detected as grouped
        assert test_arrmgr._memory_manager.is_grouped(test_arrmgr) is True

        # Request allocation - should be queued
        test_arrmgr.request_allocation(array_requests)

        # Trigger group allocation
        test_arrmgr._memory_manager.allocate_queue(test_arrmgr)

        # Verify allocation occurred
        assert test_arrmgr.device.arr1.array is not None
        assert test_arrmgr.device.arr2.array is not None

    def test_invalidate_hook_functionality(self, test_arrmgr):
        """Test cache invalidation hook functionality"""

        test_arrmgr._needs_reallocation = ["old_item"]
        test_arrmgr._needs_overwrite = ["old_overwrite"]
        test_arrmgr.device.arr1.array = np.array([1, 2, 3])
        test_arrmgr.device.arr2.array = np.array([4, 5, 6])

        # Call the invalidate hook
        test_arrmgr._invalidate_hook()

        # Check that lists were cleared and arrays were set for reallocation
        assert test_arrmgr._needs_reallocation == ["arr1", "arr2"]
        assert test_arrmgr._needs_overwrite == []
        # Arrays should be None after delete_all
        assert test_arrmgr.device.arr1.array is None
        assert test_arrmgr.device.arr2.array is None

    def test_arrays_equal_both_none(self, test_arrmgr):
        """Test arrays_equal with both arrays None"""
        assert test_arrmgr._arrays_equal(None, None) is True

    def test_arrays_equal_one_none(self, test_arrmgr):
        """Test arrays_equal with one array None"""
        arr = np.array([1, 2, 3])
        assert test_arrmgr._arrays_equal(None, arr) is False
        assert test_arrmgr._arrays_equal(arr, None) is False

    def test_arrays_equal_same_arrays(self, test_arrmgr):
        """Test arrays_equal with identical arrays"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        assert test_arrmgr._arrays_equal(arr1, arr2) is True

    def test_arrays_equal_different_arrays(self, test_arrmgr):
        """Test arrays_equal with different arrays"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        assert test_arrmgr._arrays_equal(arr1, arr2) is False

    def test_arrays_equal_shape_only_same_shape(self, test_arrmgr):
        """Test arrays_equal with shape_only=True for same shape arrays"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        assert test_arrmgr._arrays_equal(arr1, arr2, shape_only=True) is True

    def test_arrays_equal_shape_only_different_shape(self, test_arrmgr):
        """Test arrays_equal with shape_only=True for different shape arrays"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5])
        assert test_arrmgr._arrays_equal(arr1, arr2, shape_only=True) is False

    def test_arrays_equal_shape_only_3d(self, test_arrmgr):
        """Test arrays_equal with shape_only=True for 3D arrays"""
        arr1 = np.zeros((10, 5, 3))
        arr2 = np.ones((10, 5, 3))
        assert test_arrmgr._arrays_equal(arr1, arr2) is False
        assert test_arrmgr._arrays_equal(arr1, arr2, shape_only=True) is True

    def test_arrays_equal_shape_only_different_dtype(self, test_arrmgr):
        """Test arrays_equal with shape_only=True but different dtype"""
        arr1 = np.zeros((10, 5, 3), dtype=np.float32)
        arr2 = np.zeros((10, 5, 3), dtype=np.float64)
        assert test_arrmgr._arrays_equal(arr1, arr2, shape_only=True) is False

    def test_update_host_array_no_change(self, test_arrmgr):
        """Equal-valued resubmission still queues a device copy."""
        test_arrmgr._needs_reallocation = []
        dtype = test_arrmgr.host.arr1.dtype
        current = np.array([1, 2, 3], dtype=dtype)
        new = np.array([1, 2, 3], dtype=dtype)

        test_arrmgr._update_host_array(new, current, "arr1")
        test_arrmgr.allocate()
        assert "arr1" not in test_arrmgr._needs_reallocation
        assert "arr1" in test_arrmgr._needs_overwrite

    def test_update_host_array_shape_change(self, test_arrmgr):
        """Test update_host_array when array shape changes"""
        dtype = test_arrmgr.host.arr1.dtype
        current = np.array([[1, 2], [3, 4]], dtype=dtype)
        new = np.array([1, 2, 3], dtype=dtype)
        test_arrmgr._update_host_array(new, current, "arr1")

        assert "arr1" in test_arrmgr._needs_reallocation
        assert "arr1" in test_arrmgr._needs_overwrite
        # The array is attached verbatim, not copied
        assert test_arrmgr.host.arr1.array is new

    def test_update_host_array_dtype_mismatch_casts_once(self, test_arrmgr):
        """A mismatched dtype is cast into a fresh buffer on attach."""
        dtype = test_arrmgr.host.arr1.dtype
        current = np.array([1, 2, 3], dtype=dtype)
        new = np.array([4, 5, 6], dtype=np.int64)
        test_arrmgr._update_host_array(new, current, "arr1")

        attached = test_arrmgr.host.arr1.array
        assert attached is not new
        assert attached.dtype == dtype
        np.testing.assert_array_equal(attached, [4, 5, 6])

    def test_update_host_array_zero_shape(self, test_arrmgr):
        """Test update_host_array when new array has zero in shape"""
        current = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
        new = np.zeros((3, 5, 0))

        test_arrmgr._update_host_array(new, current, "arr1")

        # Should attach a minimal (1,1,1) array when zero in shape
        assert test_arrmgr.host.arr1.shape == (1, 1, 1)
        assert test_arrmgr.host.arr1.dtype == test_arrmgr._precision
        assert "arr1" in test_arrmgr._needs_reallocation

    def test_update_host_array_value_change(self, test_arrmgr):
        """A same-size resubmission attaches the new array verbatim."""
        test_arrmgr._needs_reallocation = []
        dtype = test_arrmgr.host.arr1.dtype
        current = np.array([1, 2, 3], dtype=dtype)
        new = np.array([4, 5, 6], dtype=dtype)
        test_arrmgr.allocate()
        test_arrmgr._update_host_array(new, current, "arr1")

        assert "arr1" not in test_arrmgr._needs_reallocation
        assert "arr1" in test_arrmgr._needs_overwrite
        # The new array is attached verbatim; nothing is copied
        assert test_arrmgr.host.arr1.array is new
        np.testing.assert_array_equal(current, [1, 2, 3])

    def test_chunk_placeholders(self, test_arrmgr):
        """Test next_chunk method (placeholder implementation)"""
        # This is a placeholder method, so just ensure it exists and is
        # callable
        assert test_arrmgr.initialise("test1") == ("test1")
        assert test_arrmgr.finalise("test1") == ("test1")


@pytest.fixture(scope="function")
def batch_output_sizes(arraytest_settings):
    """Create a real BatchOutputSizes instance using arraytest_settings"""
    return BatchOutputSizes(
        state=arraytest_settings["hostshape1"],
        observables=arraytest_settings["hostshape2"],
        state_summaries=arraytest_settings["hostshape3"],
        observable_summaries=arraytest_settings["hostshape4"],
    )


@pytest.fixture(scope="function")
def test_arrays_with_stride_order(arraytest_settings):
    """Create test arrays with proper stride order set using arraytest_settings
    shapes
    """
    host_arrays = TestArrays(
        state=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type="host",
        ),
        observables=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type="host",
        ),
        state_summaries=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type="host",
        ),
        observable_summaries=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type="host",
        ),
    )
    host_arrays.state.array = np.zeros(
        arraytest_settings["hostshape1"], dtype=arraytest_settings["dtype"]
    )
    host_arrays.observables.array = np.zeros(
        arraytest_settings["hostshape2"], dtype=arraytest_settings["dtype"]
    )
    host_arrays.state_summaries.array = np.zeros(
        arraytest_settings["hostshape3"], dtype=arraytest_settings["dtype"]
    )
    host_arrays.observable_summaries.array = np.zeros(
        arraytest_settings["hostshape4"], dtype=arraytest_settings["dtype"]
    )

    device_arrays = TestArrays(
        state=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type=arraytest_settings["memory"],
        ),
        observables=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type=arraytest_settings["memory"],
        ),
        state_summaries=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type=arraytest_settings["memory"],
        ),
        observable_summaries=ManagedArray(
            dtype=arraytest_settings["dtype"],
            memory_type=arraytest_settings["memory"],
        ),
    )
    device_arrays.state.array = device_array(
        arraytest_settings["devshape1"], dtype=arraytest_settings["dtype"]
    )
    device_arrays.observables.array = device_array(
        arraytest_settings["devshape2"], dtype=arraytest_settings["dtype"]
    )
    device_arrays.state_summaries.array = device_array(
        arraytest_settings["devshape3"], dtype=arraytest_settings["dtype"]
    )
    device_arrays.observable_summaries.array = device_array(
        arraytest_settings["devshape4"], dtype=arraytest_settings["dtype"]
    )

    return host_arrays, device_arrays


@pytest.fixture(scope="function")
def test_manager_with_sizing(
    test_memory_manager, test_arrays_with_stride_order, batch_output_sizes
):
    """Create array manager with proper arrays for size testing"""
    host_arrays, device_arrays = test_arrays_with_stride_order
    return ConcreteArrayManager(
        precision=np.float32,
        sizes=batch_output_sizes,
        host=host_arrays,
        device=device_arrays,
        stream_group="default",
        memory_proportion=None,
        memory_manager=test_memory_manager,
    )


class TestCheckSizesAndTypes:
    """Test the updated check_sizes and check_type methods that return
    dictionaries
    """

    def test_check_type_returns_dict(self, test_manager_with_sizing):
        """Test that check_type returns a dictionary of results"""
        arrays = {
            "state": np.zeros((10, 5, 3), dtype=np.float32),
            "observables": np.zeros(
                (10, 5, 2), dtype=np.float64
            ),  # Wrong dtype
        }

        result = test_manager_with_sizing.check_type(arrays)

        assert isinstance(result, dict)
        assert result["state"] is True
        assert result["observables"] is False

    def test_check_type_all_correct(self, test_manager_with_sizing):
        """Test check_type when all arrays have correct dtype"""
        arrays = {
            "state": np.zeros((10, 5, 3), dtype=np.float32),
            "observables": np.zeros((10, 5, 2), dtype=np.float32),
        }

        result = test_manager_with_sizing.check_type(arrays)

        assert all(result.values())
        assert result["state"] is True
        assert result["observables"] is True

    def test_check_type_with_none_arrays(self, test_manager_with_sizing):
        """Test check_type with None arrays"""
        arrays = {
            "state": None,
            "observables": np.zeros((10, 5, 2), dtype=np.float32),
        }

        result = test_manager_with_sizing.check_type(arrays)

        assert result["state"] is True  # None arrays should pass type check
        assert result["observables"] is True

    def test_check_sizes_returns_dict(
        self, arraytest_settings, test_manager_with_sizing
    ):
        """Test that check_sizes returns a dictionary of results"""
        arrays = {
            "state": np.zeros(
                arraytest_settings["hostshape1"], dtype=np.float32
            ),
            "observables": np.zeros(
                arraytest_settings["hostshape2"], dtype=np.float32
            ),
            "state_summaries": np.zeros(
                arraytest_settings["hostshape3"], dtype=np.float32
            ),
            "observable_summaries": np.zeros(
                arraytest_settings["hostshape4"], dtype=np.float32
            ),
        }

        result = test_manager_with_sizing.check_sizes(arrays, location="host")
        assert isinstance(result, dict)
        assert all([key in result for key in arrays.keys()])

    def test_check_sizes_correct_shapes(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test check_sizes with correctly shaped arrays"""
        arrays = {
            "state": np.zeros(
                arraytest_settings["hostshape1"], dtype=np.float32
            ),
            "observables": np.zeros(
                arraytest_settings["hostshape2"], dtype=np.float32
            ),
        }

        result = test_manager_with_sizing.check_sizes(arrays, location="host")

        assert result["state"] is True
        assert result["observables"] is True

    def test_check_sizes_wrong_shapes(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test check_sizes with incorrectly shaped arrays"""
        # Create wrong shapes by adding 1 to each dimension
        wrong_shape1 = tuple(
            dim + 1 for dim in arraytest_settings["hostshape1"]
        )
        wrong_shape2 = tuple(
            dim + 1 for dim in arraytest_settings["hostshape2"]
        )

        arrays = {
            "state": np.zeros(wrong_shape1, dtype=np.float32),
            "observables": np.zeros(wrong_shape2, dtype=np.float32),
        }

        result = test_manager_with_sizing.check_sizes(arrays, location="host")

        assert result["state"] is False
        assert result["observables"] is False

    def test_check_sizes_invalid_location(self, test_manager_with_sizing):
        """Test check_sizes with invalid location raises AttributeError"""
        arrays = {"state": np.zeros((10, 5, 3), dtype=np.float32)}

        with pytest.raises(AttributeError, match="Invalid location: invalid"):
            test_manager_with_sizing.check_sizes(arrays, location="invalid")


class TestCheckIncomingArrays:
    """Test the check_incoming_arrays method"""

    def test_check_incoming_arrays_all_pass(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test check_incoming_arrays when all arrays pass both checks"""
        arrays = {
            "state": np.zeros(
                arraytest_settings["hostshape1"], dtype=np.float32
            ),
            "observables": np.zeros(
                arraytest_settings["hostshape2"], dtype=np.float32
            ),
        }

        result = test_manager_with_sizing.check_incoming_arrays(
            arrays, location="host"
        )

        assert isinstance(result, dict)
        assert result["state"] is True
        assert result["observables"] is True

    def test_check_incoming_arrays_size_fail(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test check_incoming_arrays when size check fails"""
        # Create wrong shapes by adding 1 to each dimension
        wrong_shape1 = tuple(
            dim + 1 for dim in arraytest_settings["hostshape1"]
        )

        arrays = {
            "state": np.zeros(wrong_shape1, dtype=np.float32),  # Wrong shape
            "observables": np.zeros(
                arraytest_settings["hostshape2"], dtype=np.float32
            ),
        }

        result = test_manager_with_sizing.check_incoming_arrays(
            arrays, location="host"
        )

        assert result["state"] is False  # Should fail due to size
        assert result["observables"] is True

    def test_check_incoming_arrays_type_fail(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test check_incoming_arrays when type check fails"""
        arrays = {
            "state": np.zeros(
                arraytest_settings["hostshape1"], dtype=np.float64
            ),  # Wrong dtype
            "observables": np.zeros(
                arraytest_settings["hostshape2"], dtype=np.float32
            ),
        }

        result = test_manager_with_sizing.check_incoming_arrays(
            arrays, location="host"
        )

        assert result["state"] is False  # Should fail due to type
        assert result["observables"] is True


class TestUpdateHostArrays:
    """Test the new update_host_arrays method"""

    def test_update_host_arrays_success(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test update_host_arrays with valid arrays"""
        # Set up initial arrays in the host container
        test_manager_with_sizing.host.state.array = np.zeros(
            arraytest_settings["hostshape1"], dtype=np.float32
        )
        test_manager_with_sizing.host.observables.array = np.zeros(
            arraytest_settings["hostshape2"], dtype=np.float32
        )

        new_arrays = {
            "state": np.ones(
                arraytest_settings["hostshape1"], dtype=np.float32
            ),
            "observables": np.ones(
                arraytest_settings["hostshape2"], dtype=np.float32
            ),
        }

        test_manager_with_sizing.update_host_arrays(new_arrays)

        # Arrays should be updated and marked for overwrite
        # Note: stride conversion may create new array objects
        np.testing.assert_array_equal(
            test_manager_with_sizing.host.state.array, new_arrays["state"]
        )
        np.testing.assert_array_equal(
            test_manager_with_sizing.host.observables.array,
            new_arrays["observables"],
        )
        assert "state" in test_manager_with_sizing._needs_overwrite
        assert "observables" in test_manager_with_sizing._needs_overwrite

    def test_update_host_arrays_shape_change(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test update_host_arrays when array shapes change"""
        # Set up initial arrays in the host container
        test_manager_with_sizing.host.state.array = np.zeros(
            arraytest_settings["hostshape1"], dtype=np.float32
        )

        # Create a different shape by modifying the first dimension
        new_shape = (
            arraytest_settings["hostshape1"][0] - 1,
        ) + arraytest_settings["hostshape1"][1:]
        new_arrays = {
            "state": np.ones(new_shape, dtype=np.float32),  # Different shape
        }

        test_manager_with_sizing.update_host_arrays(new_arrays)

        # Array should be updated and marked for reallocation
        # Note: stride conversion may create new array objects
        np.testing.assert_array_equal(
            test_manager_with_sizing.host.state.array, new_arrays["state"]
        )
        assert "state" in test_manager_with_sizing._needs_reallocation

    def test_update_host_arrays_size_mismatch(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test update_host_arrays with size mismatch"""
        test_manager_with_sizing.host.state = np.zeros(
            arraytest_settings["hostshape1"], dtype=np.float32
        )

        # Create wrong shape by adding 1 to each dimension
        wrong_shape = tuple(
            dim + 1 for dim in arraytest_settings["hostshape1"]
        )
        new_arrays = {
            "state": np.ones(wrong_shape, dtype=np.float32),  # Wrong shape
        }

        with pytest.warns(
            UserWarning, match="do not match the expected system sizes"
        ):
            test_manager_with_sizing.update_host_arrays(new_arrays)

    def test_update_host_arrays_nonexistent_array(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test update_host_arrays with non-existent array"""
        new_arrays = {
            "nonexistent": np.zeros(
                arraytest_settings["hostshape1"], dtype=np.float32
            ),
        }

        with pytest.warns(UserWarning, match="does not exist"):
            test_manager_with_sizing.update_host_arrays(new_arrays)

    def test_update_host_arrays_no_change(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test update_host_arrays when arrays are unchanged"""
        initial_array = np.ones(
            arraytest_settings["hostshape1"], dtype=np.float32
        )
        test_manager_with_sizing.host.state = initial_array

        new_arrays = {
            "state": np.ones(
                arraytest_settings["hostshape1"], dtype=np.float32
            ),  # Same values
        }

        initial_needs_reallocation = (
            test_manager_with_sizing._needs_reallocation.copy()
        )
        initial_needs_overwrite = (
            test_manager_with_sizing._needs_overwrite.copy()
        )

        test_manager_with_sizing.update_host_arrays(new_arrays)

        # Should not add to reallocation or overwrite lists since arrays are
        # equal
        assert (
            test_manager_with_sizing._needs_reallocation
            == initial_needs_reallocation
        )
        assert (
            test_manager_with_sizing._needs_overwrite
            == initial_needs_overwrite
        )


class TestUpdateSizes:
    """Test the update_sizes method"""

    def test_update_sizes_success(
        self, test_manager_with_sizing, arraytest_settings
    ):
        """Test successful update of sizes"""
        # Create new sizes with different dimensions
        new_sizes = BatchOutputSizes(
            state=(12, 6, 4),  # Different from original
            observables=(12, 6, 3),
            state_summaries=(10, 6, 4),
            observable_summaries=(10, 6, 3),
        )

        test_manager_with_sizing.update_sizes(new_sizes)

        assert test_manager_with_sizing._sizes is new_sizes

    def test_update_sizes_wrong_type(self, test_manager_with_sizing):
        """Test update_sizes with wrong type raises TypeError"""
        # Try to update with a different type
        wrong_type_sizes = {"state": (12, 6, 4)}

        with pytest.raises(
            TypeError,
            match="Expected the new sizes object to be the same size",
        ):
            test_manager_with_sizing.update_sizes(wrong_type_sizes)


# Add missing precision fixture
@pytest.fixture(scope="function")
def precision():
    """Provide precision for tests"""
    return np.float32


class TestMemoryManagerIntegration:
    """Test integration between BaseArrayManager and MemoryManager"""

    def test_memory_manager_registration(
        self, test_arrmgr, test_memory_manager, arraytest_settings
    ):
        """Test that ArrayManager properly registers with MemoryManager using
        parametrized fixture
        """
        instance_id = id(test_arrmgr)
        assert instance_id in test_memory_manager.registry

        # Check registration details
        settings = test_memory_manager.registry[instance_id]
        assert settings.invalidate_hook == test_arrmgr._invalidate_hook
        assert (
            settings.allocation_ready_hook
            == test_arrmgr._on_allocation_complete
        )

    @pytest.mark.parametrize(
        "arraytest_overrides",
        [
            {"memory_proportion": 0.5},
            {"memory_proportion": 0.3},
            {"memory_proportion": 0.7},
        ],
        indirect=True,
    )
    def test_memory_manager_proportion_handling(
        self, test_arrmgr, test_memory_manager, arraytest_settings
    ):
        """Test that memory proportion is handled correctly using parametrized
        proportions
        """
        expected_proportion = arraytest_settings["memory_proportion"]

        # The proportion should be registered with the memory manager
        if expected_proportion is not None:
            assert (
                test_memory_manager.proportion(test_arrmgr)
                == expected_proportion
            )

    @pytest.mark.parametrize(
        "arraytest_overrides",
        [
            {"stream_group": "group1"},
            {"stream_group": "group2"},
        ],
        indirect=True,
    )
    def test_stream_group_management(
        self,
        test_arrmgr,
        second_arrmgr,
        test_memory_manager,
        arraytest_settings,
    ):
        """Test stream group functionality using parametrized groups"""
        # Both managers should be in the same group from arraytest_settings
        group_name = arraytest_settings["stream_group"]

        # Check that managers in same group are detected as grouped
        if group_name != "default":
            assert test_memory_manager.is_grouped(test_arrmgr) is True
            assert test_memory_manager.is_grouped(second_arrmgr) is True

            # Get instances in group
            group_instances = (
                test_memory_manager.stream_groups.get_instances_in_group(
                    group_name
                )
            )
            assert len(group_instances) == 2
            assert id(test_arrmgr) in group_instances
            assert id(second_arrmgr) in group_instances
        else:
            # Default group behavior - not grouped unless multiple instances
            assert (
                test_memory_manager.get_stream_group(test_arrmgr) == group_name
            )
            assert (
                test_memory_manager.get_stream_group(second_arrmgr)
                == group_name
            )

    def test_allocation_with_settings(
        self,
        test_arrmgr,
        test_memory_manager,
        array_requests,
        arraytest_settings,
    ):
        """Test that allocation respects arraytest_settings parameters"""
        # Request allocation
        test_arrmgr.request_allocation(array_requests)
        test_arrmgr.set_array_runs(arraytest_settings["devshape1"][2])
        test_memory_manager.allocate_queue(test_arrmgr)
        # Check that allocated arrays match the settings
        assert test_arrmgr.device.arr1.shape == arraytest_settings["devshape1"]
        assert test_arrmgr.device.arr2.shape == arraytest_settings["devshape2"]
        assert test_arrmgr.device.arr1.dtype == arraytest_settings["dtype"]
        assert test_arrmgr.device.arr2.dtype == arraytest_settings["dtype"]


# Parametrized tests for different array configurations
@pytest.mark.parametrize(
    "arraytest_overrides",
    [
        {"memory": "device", "stream_group": "test_group"},
        {"memory": "device", "memory_proportion": 0.5},
        {"hostshape1": (5, 5, 2), "devshape1": (5, 5, 2)},
        {"dtype": np.float64},
    ],
    indirect=True,
)
def test_array_manager_with_different_configs(
    test_arrmgr, arraytest_overrides, arraytest_settings
):
    """Test BaseArrayManager with different configurations"""
    # Check that overrides were applied correctly
    if "stream_group" in arraytest_overrides:
        assert test_arrmgr._stream_group == arraytest_settings["stream_group"]
    if "memory_proportion" in arraytest_overrides:
        assert (
            test_arrmgr._memory_proportion
            == arraytest_settings["memory_proportion"]
        )
    if "hostshape1" in arraytest_overrides:
        assert test_arrmgr.host.arr1.shape == arraytest_settings["hostshape1"]
    if "dtype" in arraytest_overrides:
        assert test_arrmgr.host.arr1.dtype == arraytest_settings["dtype"]

    # Should still be properly initialized
    assert isinstance(test_arrmgr.device, ArrayContainer)
    assert isinstance(test_arrmgr.host, ArrayContainer)


@pytest.mark.parametrize(
    "arraytest_overrides",
    [
        {
            "memory": "device",
            "devshape1": (10, 20, 3),
            "devshape2": (15, 25, 2),
        },
        {"memory": "device", "devshape1": (3, 4, 5), "devshape2": (6, 7, 8)},
    ],
    indirect=True,
)
def test_allocation_response_matches_settings(
    allocation_response, arraytest_settings
):
    """Test that allocation_response fixture uses arraytest_settings correctly
    """
    assert (
        allocation_response.arr["arr1"].shape
        == arraytest_settings["devshape1"]
    )
    assert (
        allocation_response.arr["arr2"].shape
        == arraytest_settings["devshape2"]
    )
    assert allocation_response.chunks == 1


@pytest.mark.parametrize(
    "arraytest_overrides",
    [
        {
            "memory": "device",
            "stream_group": "custom_group",
            "memory_proportion": 0.3,
        },
        {
            "memory": "device",
            "stream_group": "another_group",
            "memory_proportion": 0.7,
        },
    ],
    indirect=True,
)
def test_memory_integration_with_settings(
    test_arrmgr, test_memory_manager, arraytest_settings
):
    """Test that memory manager integration respects arraytest_settings"""
    # Check stream group from settings
    assert (
        test_memory_manager.get_stream_group(test_arrmgr)
        == arraytest_settings["stream_group"]
    )

    # Check proportion from settings
    if arraytest_settings["memory_proportion"] is not None:
        assert (
            test_memory_manager.proportion(test_arrmgr)
            == arraytest_settings["memory_proportion"]
        )


class TestChunkedHostSliceTransfers:
    """Test contiguous slice handling for chunked transfers."""

    def test_noncontiguous_host_slice_detected(self):
        """Verify that slicing a host array creates non-contiguous views."""
        host_array = np.zeros((100, 10, 50), dtype=np_float32)
        # Slice on run axis (last axis)
        host_slice = host_array[:, :, 0:10]
        assert not host_slice.flags["C_CONTIGUOUS"]

    def test_contiguous_copy_matches_shape(self):
        """Verify ascontiguousarray creates matching-shape contiguous array."""
        host_array = np.arange(100 * 10 * 50, dtype=np_float32).reshape(
            100, 10, 50
        )
        host_slice = host_array[:, :, 0:10]
        contiguous = np.ascontiguousarray(host_slice)

        assert contiguous.flags["C_CONTIGUOUS"]
        assert contiguous.shape == (100, 10, 10)
        np.testing.assert_array_equal(contiguous, host_slice)


class TestManagedArrayChunkedShape:
    """Test ManagedArray chunked_shape field and needs_chunked_transfer."""

    def test_managed_array_has_chunked_shape_field(self):
        """Verify ManagedArray has chunked_shape field defaulting to None."""
        managed = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 5, 100),
            memory_type="device",
        )
        assert managed.chunked_shape is None

    def test_managed_array_needs_chunked_transfer_false_when_none(self):
        """Verify needs_chunked_transfer returns False when chunked_shape is
        None.
        """
        managed = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 5, 100),
            memory_type="device",
        )
        assert managed.chunked_shape is None
        assert managed.needs_chunked_transfer is False

    def test_managed_array_needs_chunked_transfer_false_when_equal(self):
        """Verify needs_chunked_transfer returns False when shapes match."""
        managed = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 5, 25),
            memory_type="device",
        )
        managed.chunked_shape = (10, 5, 25)
        assert managed.needs_chunked_transfer is False

    def test_managed_array_needs_chunked_transfer_true_when_different(self):
        """Verify needs_chunked_transfer returns True when shapes differ."""
        managed = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 5, 100),
            memory_type="device",
        )
        # Full shape is (10, 5, 100), chunked shape is (10, 5, 25)
        managed.chunked_shape = (10, 5, 25)
        assert managed.needs_chunked_transfer is True


class TestChunkedShapePropagation:
    """Test chunked_shape propagation from allocation response."""

    def test_on_allocation_complete_stores_chunked_shape(
        self, arraytest_settings, precision, test_memory_manager
    ):
        """Verify _on_allocation_complete populates ManagedArray.chunked_shape.
        """
        # Create host and device arrays
        host_arrays = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="host",
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="host",
            ),
        )
        host_arrays.arr1.array = np.zeros(
            arraytest_settings["hostshape1"], dtype=precision
        )
        host_arrays.arr2.array = np.zeros(
            arraytest_settings["hostshape2"], dtype=precision
        )

        device_arrays = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="device",
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="device",
            ),
        )

        manager = ConcreteArrayManager(
            precision=precision,
            sizes=None,
            host=host_arrays,
            device=device_arrays,
            stream_group="default",
            memory_proportion=None,
            memory_manager=test_memory_manager,
        )

        # Create mock arrays for the response
        arr1 = device_array(arraytest_settings["devshape1"], dtype=precision)
        arr2 = device_array(arraytest_settings["devshape2"], dtype=precision)

        # Expected chunked shapes (smaller than full shapes)
        chunked_shapes = {
            "arr1": (4, 3, 2),  # Chunked version of devshape1
            "arr2": (4, 3, 2),  # Chunked version of devshape2
        }

        response = ArrayResponse(
            arr={"arr1": arr1, "arr2": arr2},
            chunks=2,
            chunked_shapes=chunked_shapes,
        )

        # Set up the arrays that need reallocation
        manager._needs_reallocation = ["arr1", "arr2"]

        # Call the method under test
        manager._on_allocation_complete(response)

        # Verify chunked_shape was stored in the ManagedArrays
        assert manager.device.arr1.chunked_shape == (4, 3, 2)
        assert manager.device.arr2.chunked_shape == (4, 3, 2)


def test_chunked_shape_propagates_through_allocation(test_memory_manager):
    """Test chunked_shape flows from MemoryManager to ManagedArray.

    This integration test verifies:
    1. MemoryManager.compute_chunked_shapes calculates correct shapes
    2. ArrayResponse includes chunked_shapes
    3. ManagedArray.chunked_shape is populated after allocation
    4. needs_chunked_transfer returns correct value based on shapes
    """
    precision = np_float32

    # Create arrays that will require chunking (5 runs with 2 chunks)
    host_shape = (10, 3, 5)  # 5 runs
    expected_chunked_shape = (10, 3, 2)  # 5 // 2 = 2 runs per chunk

    # Create host arrays
    host_arrays = TestArraysSimple(
        arr1=ManagedArray(
            dtype=precision,
            default_shape=host_shape,
            memory_type="host",
        ),
        arr2=ManagedArray(
            dtype=precision,
            default_shape=host_shape,
            memory_type="host",
        ),
    )
    host_arrays.arr1.array = np.zeros(host_shape, dtype=precision)
    host_arrays.arr2.array = np.zeros(host_shape, dtype=precision)

    # Create device arrays
    device_arrays = TestArraysSimple(
        arr1=ManagedArray(
            dtype=precision,
            default_shape=host_shape,  # Full shape before chunking
            memory_type="device",
        ),
        arr2=ManagedArray(
            dtype=precision,
            default_shape=host_shape,
            memory_type="device",
        ),
    )

    # Create manager
    manager = ConcreteArrayManager(
        precision=precision,
        sizes=None,
        host=host_arrays,
        device=device_arrays,
        stream_group="default",
        memory_proportion=None,
        memory_manager=test_memory_manager,
    )

    # Simulate allocation response with chunked shapes
    arr1 = device_array(expected_chunked_shape, dtype=precision)
    arr2 = device_array(expected_chunked_shape, dtype=precision)

    chunked_shapes = {
        "arr1": expected_chunked_shape,
        "arr2": expected_chunked_shape,
    }

    # Slice function for chunking on run axis (index 2)
    def make_slice_fn(run_axis_idx, chunk_size):
        def slice_fn(chunk_idx):
            slices = [slice(None)] * 3
            start = chunk_idx * chunk_size
            end = start + chunk_size
            slices[run_axis_idx] = slice(start, end)
            return tuple(slices)

        return slice_fn

    response = ArrayResponse(
        arr={"arr1": arr1, "arr2": arr2},
        chunks=2,
        chunked_shapes=chunked_shapes,
    )

    # Mark arrays as needing reallocation
    manager._needs_reallocation = ["arr1", "arr2"]

    # Call allocation complete callback
    manager._on_allocation_complete(response)

    # Verify chunked_shape is stored
    assert manager.device.arr1.chunked_shape == expected_chunked_shape
    assert manager.device.arr2.chunked_shape == expected_chunked_shape

    # Verify needs_chunked_transfer returns True since full shape != chunked
    # Full shape is (10, 3, 5), chunked shape is (10, 3, 2)
    assert manager.host.arr1.needs_chunked_transfer is True
    assert manager.host.arr2.needs_chunked_transfer is True

    # Verify the chunks were stored
    assert manager._chunks == 2


class TestAllocationCallbackSimplifiedResponse:
    """Test allocation callback works with simplified ArrayResponse."""

    def test_allocation_callback_works_with_simplified_response(
        self, test_memory_manager
    ):
        """Verify array managers handle ArrayResponse without removed fields.

        This test ensures that _on_allocation_complete works correctlyhe array
        manager should only access:
        - response.arr (dict of allocated arrays)
        - response.chunks (int, number of chunks)
        - response.chunked_shapes (dict of per-chunk shapes)
        - response.chunked_slices (dict of slicing functions)
        """
        precision = np_float32
        host_shape = (10, 3, 100)
        chunked_shape = (10, 3, 25)

        # Create host arrays
        host_arrays = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                default_shape=host_shape,
                memory_type="host",
            ),
            arr2=ManagedArray(
                dtype=precision,
                default_shape=host_shape,
                memory_type="host",
            ),
        )
        host_arrays.arr1.array = np.zeros(host_shape, dtype=precision)
        host_arrays.arr2.array = np.zeros(host_shape, dtype=precision)

        # Create device arrays
        device_arrays = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                default_shape=host_shape,
                memory_type="device",
            ),
            arr2=ManagedArray(
                dtype=precision,
                default_shape=host_shape,
                memory_type="device",
            ),
        )

        # Create manager
        manager = ConcreteArrayManager(
            precision=precision,
            sizes=None,
            host=host_arrays,
            device=device_arrays,
            stream_group="default",
            memory_proportion=None,
            memory_manager=test_memory_manager,
        )

        arr1 = device_array(chunked_shape, dtype=precision)
        arr2 = device_array(chunked_shape, dtype=precision)

        # Create slice function for chunking
        def make_slice_fn(chunk_size):
            def slice_fn(chunk_idx):
                start = chunk_idx * chunk_size
                end = start + chunk_size
                return (slice(None), slice(None), slice(start, end))

            return slice_fn

        response = ArrayResponse(
            arr={"arr1": arr1, "arr2": arr2},
            chunks=4,  # 100 runs / 25 per chunk = 4 chunks
            chunked_shapes={"arr1": chunked_shape, "arr2": chunked_shape},
        )

        # Verify response does NOT have removed fields
        assert not hasattr(response, "axis_length")
        assert not hasattr(response, "dangling_chunk_length")

        # Mark arrays for reallocation
        manager._needs_reallocation = ["arr1", "arr2"]

        # Call allocation complete - should work without accessing removed
        # fields
        manager._on_allocation_complete(response)

        # Verify allocation completed successfully
        assert manager.device.arr1.array is arr1
        assert manager.device.arr2.array is arr2
        assert manager._chunks == 4
        assert manager.device.arr1.chunked_shape == chunked_shape
        assert manager.device.arr2.chunked_shape == chunked_shape
        assert manager._needs_reallocation == []

        # Verify chunked transfer detection works
        assert manager.host.arr1.needs_chunked_transfer is True
        assert manager.host.arr2.needs_chunked_transfer is True


class TestChunkSliceMethod:
    """Test the enhanced chunk_slice() method with chunk_index parameter."""

    def test_chunk_slice_no_chunking_returns_full_array(self):
        """Verify chunk_slice returns full array when is_chunked=False."""
        # Create ManagedArray with chunking disabled
        managed = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 5, 100),
            memory_type="host",
            is_chunked=False,
        )
        managed.array = np.arange(5000, dtype=np_float32).reshape(10, 5, 100)

        # Set chunk parameters (even though chunking is disabled)
        managed.num_chunks = 4
        managed.chunk_length = 25

        # chunk_slice should return the full array
        result = managed.chunk_slice(0)
        assert result.shape == (10, 5, 100)
        np.testing.assert_array_equal(result, managed.array)

    def test_chunk_slice_computes_correct_slices(self):
        """Verify chunk_slice computes correct start/end for each chunk."""
        # Create ManagedArray with known data pattern
        managed = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 5, 100),
            memory_type="host",
            stride_order=("", "", "run"),
            is_chunked=True,
        )
        # Fill array with values that make chunks easy to identify
        # Each "run" (last axis) position has its own value
        test_array = np.zeros((10, 5, 100), dtype=np_float32)
        for i in range(100):
            test_array[:, :, i] = i
        managed.array = test_array

        # Set chunk parameters: 100 runs, 25 per chunk, 4 chunks
        managed.num_chunks = 4
        managed.chunk_length = 25

        # Test each chunk
        for chunk_idx in range(4):
            result = managed.chunk_slice(chunk_idx)
            expected_start = chunk_idx * 25
            expected_end = expected_start + 25

            # Shape should be (10, 5, 25) for each chunk
            assert result.shape == (10, 5, 25)

            # Verify data corresponds to correct slice
            expected = test_array[:, :, expected_start:expected_end]
            np.testing.assert_array_equal(result, expected)

            # Check that first position has correct value
            assert result[0, 0, 0] == expected_start

    def test_chunk_slice_handles_final_chunk_dynamically(self):
        """Verify final chunk is handled dynamically without
        dangling_chunk_length.
        """
        # Create ManagedArray with 105 runs (4 chunks of 25, last chunk has 5)
        managed = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 5, 105),
            memory_type="host",
            stride_order=("", "", "run"),
            is_chunked=True,
        )
        test_array = np.arange(5250, dtype=np_float32).reshape(10, 5, 105)
        managed.array = test_array

        # Set chunk parameters - no dangling_chunk_length needed
        managed.num_chunks = 5
        managed.chunk_length = 25

        # Test regular chunks (0-3)
        for chunk_idx in range(4):
            result = managed.chunk_slice(chunk_idx)
            assert result.shape == (10, 5, 25)

        # Test final chunk - should be shorter (dynamically calculated)
        final_chunk = managed.chunk_slice(4)
        assert final_chunk.shape == (10, 5, 5)

        # Verify data is from correct slice (runs 100-105)
        expected = test_array[:, :, 100:105]
        np.testing.assert_array_equal(final_chunk, expected)

    def test_chunk_slice_none_chunk_axis_returns_full_array(self):
        """Verify chunk_slice returns full array when _chunk_axis_index is
        None."""
        # Create ManagedArray without "run" in stride_order
        managed = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 5),
            memory_type="host",
            is_chunked=True,
        )
        managed.array = np.arange(50, dtype=np_float32).reshape(10, 5)

        # _chunk_axis_index should be None (no "run" axis)
        assert managed._chunk_axis_index is None

        # chunk_slice should return full array
        result = managed.chunk_slice(0)
        assert result.shape == (10, 5)
        np.testing.assert_array_equal(result, managed.array)

    def test_chunk_slice_none_parameters_returns_full_array(self):
        """Verify chunk_slice returns full array when chunk parameters are
        None."""
        managed = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 5, 100),
            memory_type="host",
            stride_order=("", "", "run"),
            is_chunked=True,
        )
        managed.array = np.arange(5000, dtype=np_float32).reshape(10, 5, 100)

        # Leave chunk parameters as None
        assert managed.num_chunks == 1
        assert managed.chunk_length is None

        # chunk_slice should return full array (fast path)
        result = managed.chunk_slice(0)
        assert result.shape == (10, 5, 100)
        np.testing.assert_array_equal(result, managed.array)

    def test_chunk_slice_different_axis_indices(self):
        """Verify chunk_slice works with chunk axis at different
        positions."""
        # Test with "run" at axis 0
        managed_axis0 = ManagedArray(
            dtype=np_float32,
            default_shape=(100, 5, 10),
            memory_type="host",
            stride_order=("run", "", ""),
            is_chunked=True,
        )
        test_array = np.arange(5000, dtype=np_float32).reshape(100, 5, 10)
        managed_axis0.array = test_array
        managed_axis0.num_chunks = 4
        managed_axis0.chunk_length = 25
        assert managed_axis0._chunk_axis_index == 0

        # Chunk 0 should be runs 0-25 on first axis
        result = managed_axis0.chunk_slice(0)
        assert result.shape == (25, 5, 10)
        np.testing.assert_array_equal(result, test_array[0:25, :, :])

        # Test with "run" at axis 1
        managed_axis1 = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 100, 5),
            memory_type="host",
            stride_order=("", "run", ""),
            is_chunked=True,
        )
        test_array2 = np.arange(5000, dtype=np_float32).reshape(10, 100, 5)
        managed_axis1.array = test_array2
        managed_axis1.num_chunks = 4
        managed_axis1.chunk_length = 25
        assert managed_axis1._chunk_axis_index == 1

        # Chunk 0 should be runs 0-25 on middle axis
        result2 = managed_axis1.chunk_slice(0)
        assert result2.shape == (10, 25, 5)
        np.testing.assert_array_equal(result2, test_array2[:, 0:25, :])


def test_on_allocation_complete_stores_chunk_parameters(test_memory_manager):
    """Verify _on_allocation_complete stores chunk parameters in ManagedArray.

    Tests that chunk_length and num_chunks are extracted from ArrayResponse
    and stored in both host and device ManagedArray objects.
    """
    precision = np_float32
    host_shape = (10, 3, 100)
    chunked_shape = (10, 3, 25)

    # Create host arrays
    host_arrays = TestArraysSimple(
        arr1=ManagedArray(
            dtype=precision,
            default_shape=host_shape,
            stride_order=("", "", "run"),
            memory_type="host",
        ),
        arr2=ManagedArray(
            dtype=precision,
            default_shape=host_shape,
            stride_order=("", "", "run"),
            memory_type="host",
        ),
    )
    host_arrays.arr1.array = np.zeros(host_shape, dtype=precision)
    host_arrays.arr2.array = np.zeros(host_shape, dtype=precision)

    # Create device arrays
    device_arrays = TestArraysSimple(
        arr1=ManagedArray(
            dtype=precision,
            default_shape=host_shape,
            stride_order=("", "", "run"),
            memory_type="device",
        ),
        arr2=ManagedArray(
            dtype=precision,
            default_shape=host_shape,
            stride_order=("", "", "run"),
            memory_type="device",
        ),
    )

    # Create manager
    manager = ConcreteArrayManager(
        precision=precision,
        sizes=None,
        host=host_arrays,
        device=device_arrays,
        stream_group="default",
        memory_proportion=None,
        memory_manager=test_memory_manager,
    )

    # Create ArrayResponse with chunk parameters
    arr1 = device_array(chunked_shape, dtype=precision)
    arr2 = device_array(chunked_shape, dtype=precision)

    # Chunk parameters: 100 runs divided into 4 chunks of 25 each
    chunks = 4
    chunk_length = 25

    response = ArrayResponse(
        arr={"arr1": arr1, "arr2": arr2},
        chunks=chunks,
        chunk_length=chunk_length,
        chunked_shapes={"arr1": chunked_shape, "arr2": chunked_shape},
    )

    # Mark arrays for reallocation
    manager._needs_reallocation = ["arr1", "arr2"]

    # Call allocation complete callback
    manager._on_allocation_complete(response)

    # Verify chunk parameters are stored in both device and host arrays
    # Device arrays
    assert manager.device.arr1.chunk_length == chunk_length
    assert manager.device.arr1.num_chunks == chunks
    assert manager.device.arr2.chunk_length == chunk_length
    assert manager.device.arr2.num_chunks == chunks

    # Host arrays
    assert manager.host.arr1.chunk_length == chunk_length
    assert manager.host.arr1.num_chunks == chunks
    assert manager.host.arr2.chunk_length == chunk_length
    assert manager.host.arr2.num_chunks == chunks

    # Verify chunked_shape is still set (existing functionality)
    assert manager.device.arr1.chunked_shape == chunked_shape
    assert manager.device.arr2.chunked_shape == chunked_shape
    assert manager.host.arr1.chunked_shape == chunked_shape
    assert manager.host.arr2.chunked_shape == chunked_shape

    # Verify chunks was stored in manager
    assert manager._chunks == chunks


def test_managed_array_no_chunked_slice_fn_field():
    """Verify ManagedArray does not have chunked_slice_fn attribute.

    This test ensures that the obsolete chunked_slice_fn field has been
    removed from ManagedArray after the refactoring that replaced
    callable-based slicing with parameter-based slicing in chunk_slice().
    """
    # Create a fresh ManagedArray instance
    managed = ManagedArray(
        dtype=np_float32,
        default_shape=(10, 5, 100),
        memory_type="host",
        stride_order=("", "", "run"),
        is_chunked=True,
    )

    # Verify chunked_slice_fn attribute does not exist
    assert not hasattr(managed, "chunked_slice_fn"), (
        "ManagedArray should not have chunked_slice_fn field after "
        "refactoring to parameter-based chunk_slice() method"
    )

    # Verify the new chunk parameter fields exist instead
    assert hasattr(managed, "chunk_length")
    assert hasattr(managed, "num_chunks")

    # Verify chunk_slice method exists
    assert hasattr(managed, "chunk_slice")
    assert callable(managed.chunk_slice)


class TestNumRunsAttribute:
    """Test the num_runs attribute and set_array_runs method."""

    def test_set_array_runs_sets_attribute(
        self, test_memory_manager, precision
    ):
        """Verify set_array_runs sets the num_runs attribute."""
        host_arrays = TestArraysSimple()
        device_arrays = TestArraysSimple()

        manager = ConcreteArrayManager(
            precision=precision,
            sizes=None,
            host=host_arrays,
            device=device_arrays,
            stream_group="default",
            memory_proportion=None,
            memory_manager=test_memory_manager,
        )

        # Initially num_runs should be None
        assert manager.num_runs == 1

        # Set num_runs
        manager.set_array_runs(100)

        # Verify it's set correctly
        assert manager.num_runs == 100

    def test_set_array_runs_validates_type(
        self, test_memory_manager, precision
    ):
        """Verify set_array_runs validates type."""
        host_arrays = TestArraysSimple()
        device_arrays = TestArraysSimple()

        manager = ConcreteArrayManager(
            precision=precision,
            sizes=None,
            host=host_arrays,
            device=device_arrays,
            stream_group="default",
            memory_proportion=None,
            memory_manager=test_memory_manager,
        )

        # Test with float (invalid)
        with pytest.raises(TypeError, match="must be"):
            manager.set_array_runs(100.5)

        # Test with string (invalid)
        with pytest.raises(TypeError, match="must be"):
            manager.set_array_runs("100")

    def test_set_array_runs_validates_range(
        self, test_memory_manager, precision
    ):
        """Verify set_array_runs validates range."""
        host_arrays = TestArraysSimple()
        device_arrays = TestArraysSimple()

        manager = ConcreteArrayManager(
            precision=precision,
            sizes=None,
            host=host_arrays,
            device=device_arrays,
            stream_group="default",
            memory_proportion=None,
            memory_manager=test_memory_manager,
        )

        # Test with 0 (invalid)
        with pytest.raises(ValueError, match="must be >= 1"):
            manager.set_array_runs(0)

        # Test with negative value (invalid)
        with pytest.raises(ValueError, match="must be >= 1"):
            manager.set_array_runs(-1)

        # Test with 1 (valid minimum)
        manager.set_array_runs(1)
        assert manager.num_runs == 1


class TestChunkMetadataFlow2:
    """Test chunk metadata propagation from allocation to slicing."""

    def test_allocation_complete_sets_chunk_metadata(
        self, test_memory_manager, precision
    ):
        """Verify _on_allocation_complete sets chunk_length and num_chunks.

        This test verifies that chunk metadata from ArrayResponse is properly
        propagated to ManagedArray objects for both host and device containers.
        """
        # Create host and device arrays with known shapes
        host_shape = (10, 5, 100)
        chunked_shape = (10, 5, 25)

        host_arrays = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="host",
                stride_order=("", "", "run"),
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="host",
                stride_order=("", "", "run"),
            ),
        )
        host_arrays.arr1.array = np.zeros(host_shape, dtype=precision)
        host_arrays.arr2.array = np.zeros(host_shape, dtype=precision)

        device_arrays = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="device",
                stride_order=("", "", "run"),
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="device",
                stride_order=("", "", "run"),
            ),
        )

        manager = ConcreteArrayManager(
            precision=precision,
            sizes=None,
            host=host_arrays,
            device=device_arrays,
            stream_group="default",
            memory_proportion=None,
            memory_manager=test_memory_manager,
        )

        # Create ArrayResponse with chunk metadata
        arr1 = device_array(chunked_shape, dtype=precision)
        arr2 = device_array(chunked_shape, dtype=precision)

        # Define chunk parameters: 100 runs / 25 per chunk = 4 chunks
        chunks = 4
        chunk_length = 25

        response = ArrayResponse(
            arr={"arr1": arr1, "arr2": arr2},
            chunks=chunks,
            chunk_length=chunk_length,
            chunked_shapes={"arr1": chunked_shape, "arr2": chunked_shape},
        )

        # Mark arrays for reallocation
        manager._needs_reallocation = ["arr1", "arr2"]

        # Call allocation complete - this should set chunk metadata
        manager._on_allocation_complete(response)

        # Verify chunk_length is set on device arrays
        assert manager.device.arr1.chunk_length == chunk_length
        assert manager.device.arr2.chunk_length == chunk_length

        # Verify num_chunks is set on device arrays
        assert manager.device.arr1.num_chunks == chunks
        assert manager.device.arr2.num_chunks == chunks

        # Verify chunk_length is set on host arrays
        assert manager.host.arr1.chunk_length == chunk_length
        assert manager.host.arr2.chunk_length == chunk_length

        # Verify num_chunks is set on host arrays
        assert manager.host.arr1.num_chunks == chunks
        assert manager.host.arr2.num_chunks == chunks

        # Verify chunked_shape is also set (existing functionality)
        assert manager.device.arr1.chunked_shape == chunked_shape
        assert manager.device.arr2.chunked_shape == chunked_shape
        assert manager.host.arr1.chunked_shape == chunked_shape
        assert manager.host.arr2.chunked_shape == chunked_shape

    def test_chunk_slice_uses_chunk_metadata_from_response(
        self, test_memory_manager, precision
    ):
        """Verify chunk_slice returns correct shapes using chunk metadata.

        This test verifies that chunk_slice method uses chunk_length and
        num_chunks that were set by _on_allocation_complete to compute
        correct chunk slices.
        """
        # Create host arrays with known data pattern
        host_shape = (10, 5, 100)
        chunked_shape = (10, 5, 25)

        host_arrays = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="host",
                stride_order=("", "", "run"),
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="host",
                stride_order=("", "", "run"),
            ),
        )
        # Fill with identifiable pattern - each run position has its value
        test_data = np.zeros(host_shape, dtype=precision)
        for i in range(100):
            test_data[:, :, i] = i
        host_arrays.arr1.array = test_data.copy()
        host_arrays.arr2.array = test_data.copy()

        device_arrays = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="device",
                stride_order=("", "", "run"),
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="device",
                stride_order=("", "", "run"),
            ),
        )

        manager = ConcreteArrayManager(
            precision=precision,
            sizes=None,
            host=host_arrays,
            device=device_arrays,
            stream_group="default",
            memory_proportion=None,
            memory_manager=test_memory_manager,
        )

        # Create ArrayResponse with chunk metadata
        arr1 = device_array(chunked_shape, dtype=precision)
        arr2 = device_array(chunked_shape, dtype=precision)

        chunks = 4
        chunk_length = 25

        response = ArrayResponse(
            arr={"arr1": arr1, "arr2": arr2},
            chunks=chunks,
            chunk_length=chunk_length,
            chunked_shapes={"arr1": chunked_shape, "arr2": chunked_shape},
        )

        manager._needs_reallocation = ["arr1", "arr2"]
        manager._on_allocation_complete(response)

        # Now test chunk_slice with the set metadata
        # Chunk 0 should be runs 0-25
        chunk0 = manager.host.arr1.chunk_slice(0)
        assert chunk0.shape == (10, 5, 25)
        assert chunk0[0, 0, 0] == 0  # First run
        assert chunk0[0, 0, 24] == 24  # Last run of chunk 0

        # Chunk 1 should be runs 25-50
        chunk1 = manager.host.arr1.chunk_slice(1)
        assert chunk1.shape == (10, 5, 25)
        assert chunk1[0, 0, 0] == 25  # First run of chunk 1
        assert chunk1[0, 0, 24] == 49  # Last run of chunk 1

        # Chunk 2 should be runs 50-75
        chunk2 = manager.host.arr1.chunk_slice(2)
        assert chunk2.shape == (10, 5, 25)
        assert chunk2[0, 0, 0] == 50
        assert chunk2[0, 0, 24] == 74

        # Chunk 3 (final) should be runs 75-100
        chunk3 = manager.host.arr1.chunk_slice(3)
        assert chunk3.shape == (10, 5, 25)
        assert chunk3[0, 0, 0] == 75
        assert chunk3[0, 0, 24] == 99  # Last run

    def test_chunk_metadata_flow_integration(
        self, test_memory_manager, precision
    ):
        """End-to-end test of allocation → propagation → slicing.

        This integration test verifies the complete flow:
        1. Create manager with _sizes containing runs
        2. Call allocate() to create requests with total_runs
        3. Simulate allocation response with chunk_length and chunks
        4. Verify ManagedArray has correct chunk_length and num_chunks
        5. Verify chunk_slice returns correct shapes

        Tests multiple scenarios:
        - num_chunks=1 (no chunking)
        - num_chunks=4 (even chunking)
        - num_chunks=5 (dangling chunk)
        """
        from cubie.outputhandling.output_sizes import BatchOutputSizes

        # Test case 1: No chunking (num_chunks=1)
        host_shape_1 = (10, 5, 100)
        chunked_shape_1 = (10, 5, 100)  # Same as full shape

        sizes_1 = BatchOutputSizes(
            state=host_shape_1,
            observables=host_shape_1,
        )

        host_arrays_1 = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="host",
                stride_order=("", "", "run"),
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="host",
                stride_order=("", "", "run"),
            ),
        )
        host_arrays_1.arr1.array = np.zeros(host_shape_1, dtype=precision)
        host_arrays_1.arr2.array = np.zeros(host_shape_1, dtype=precision)

        device_arrays_1 = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="device",
                stride_order=("", "", "run"),
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="device",
                stride_order=("", "", "run"),
            ),
        )

        manager_1 = ConcreteArrayManager(
            precision=precision,
            sizes=sizes_1,
            host=host_arrays_1,
            device=device_arrays_1,
            stream_group="default",
            memory_proportion=None,
            memory_manager=test_memory_manager,
        )

        # Simulate allocation response with no chunking
        arr1_dev = device_array(chunked_shape_1, dtype=precision)
        arr2_dev = device_array(chunked_shape_1, dtype=precision)

        response_1 = ArrayResponse(
            arr={"arr1": arr1_dev, "arr2": arr2_dev},
            chunks=1,
            chunk_length=100,
            chunked_shapes={"arr1": chunked_shape_1, "arr2": chunked_shape_1},
        )

        manager_1._needs_reallocation = ["arr1", "arr2"]
        manager_1._on_allocation_complete(response_1)

        # Verify chunk metadata
        assert manager_1.host.arr1.chunk_length == 100
        assert manager_1.host.arr1.num_chunks == 1

        # Verify chunk_slice returns full array for single chunk
        chunk = manager_1.host.arr1.chunk_slice(0)
        assert chunk.shape == host_shape_1

        # Test case 2: Even chunking (num_chunks=4)
        host_shape_2 = (10, 5, 100)
        chunked_shape_2 = (10, 5, 25)

        sizes_2 = BatchOutputSizes(
            state=host_shape_2,
            observables=host_shape_2,
        )

        host_arrays_2 = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="host",
                stride_order=("", "", "run"),
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="host",
                stride_order=("", "", "run"),
            ),
        )
        test_data = np.arange(5000, dtype=precision).reshape(host_shape_2)
        host_arrays_2.arr1.array = test_data.copy()
        host_arrays_2.arr2.array = test_data.copy()

        device_arrays_2 = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="device",
                stride_order=("", "", "run"),
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="device",
                stride_order=("", "", "run"),
            ),
        )

        manager_2 = ConcreteArrayManager(
            precision=precision,
            sizes=sizes_2,
            host=host_arrays_2,
            device=device_arrays_2,
            stream_group="default",
            memory_proportion=None,
            memory_manager=test_memory_manager,
        )

        # Simulate allocation response with even chunking
        arr1_dev_2 = device_array(
            chunked_shape_2,
            dtype=precision,
        )
        arr2_dev_2 = device_array(
            chunked_shape_2,
            dtype=precision,
        )

        response_2 = ArrayResponse(
            arr={"arr1": arr1_dev_2, "arr2": arr2_dev_2},
            chunks=4,
            chunk_length=25,
            chunked_shapes={"arr1": chunked_shape_2, "arr2": chunked_shape_2},
        )

        manager_2._needs_reallocation = ["arr1", "arr2"]
        manager_2._on_allocation_complete(response_2)

        # Verify chunk metadata
        assert manager_2.host.arr1.chunk_length == 25
        assert manager_2.host.arr1.num_chunks == 4

        # Verify all chunks have correct shape
        for chunk_idx in range(4):
            chunk = manager_2.host.arr1.chunk_slice(chunk_idx)
            assert chunk.shape == (10, 5, 25)

        # Test case 3: Dangling chunk (num_chunks=5, last chunk smaller)
        host_shape_3 = (10, 5, 105)  # 105 runs
        chunked_shape_3 = (10, 5, 25)  # 25 per chunk

        sizes_3 = BatchOutputSizes(
            state=host_shape_3,
            observables=host_shape_3,
        )

        host_arrays_3 = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="host",
                stride_order=("", "", "run"),
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="host",
                stride_order=("", "", "run"),
            ),
        )
        test_data_3 = np.arange(5250, dtype=precision).reshape(host_shape_3)
        host_arrays_3.arr1.array = test_data_3.copy()
        host_arrays_3.arr2.array = test_data_3.copy()

        device_arrays_3 = TestArraysSimple(
            arr1=ManagedArray(
                dtype=precision,
                memory_type="device",
                stride_order=("", "", "run"),
            ),
            arr2=ManagedArray(
                dtype=precision,
                memory_type="device",
                stride_order=("", "", "run"),
            ),
        )

        manager_3 = ConcreteArrayManager(
            precision=precision,
            sizes=sizes_3,
            host=host_arrays_3,
            device=device_arrays_3,
            stream_group="default",
            memory_proportion=None,
            memory_manager=test_memory_manager,
        )

        # Simulate allocation response with dangling chunk
        # Last chunk will be smaller (5 runs instead of 25)
        arr1_dev_3 = device_array(
            chunked_shape_3,
            dtype=precision,
        )
        arr2_dev_3 = device_array(
            chunked_shape_3,
            dtype=precision,
        )

        response_3 = ArrayResponse(
            arr={"arr1": arr1_dev_3, "arr2": arr2_dev_3},
            chunks=5,  # 4 full chunks + 1 dangling chunk
            chunk_length=25,
            chunked_shapes={"arr1": chunked_shape_3, "arr2": chunked_shape_3},
        )

        manager_3._needs_reallocation = ["arr1", "arr2"]
        manager_3._on_allocation_complete(response_3)

        # Verify chunk metadata
        assert manager_3.host.arr1.chunk_length == 25
        assert manager_3.host.arr1.num_chunks == 5

        # Verify first 4 chunks have full size
        for chunk_idx in range(4):
            chunk = manager_3.host.arr1.chunk_slice(chunk_idx)
            assert chunk.shape == (10, 5, 25)

        # Verify last chunk has correct smaller size (105 - 100 = 5)
        final_chunk = manager_3.host.arr1.chunk_slice(4)
        assert final_chunk.shape == (10, 5, 5)

        # Verify data integrity for final chunk
        expected_final = test_data_3[:, :, 100:105]
        np.testing.assert_array_equal(final_chunk, expected_final)


class TestChunkSliceTypeValidation:
    """Test chunk_slice() rejects non-int chunk_index."""

    def test_chunk_slice_non_int_raises_type_error(self):
        """Verify chunk_slice raises TypeError for a non-int index."""
        managed = ManagedArray(
            dtype=np_float32,
            default_shape=(10, 5, 100),
            memory_type="host",
            stride_order=("", "", "run"),
        )
        with pytest.raises(TypeError, match="chunk_index must be int"):
            managed.chunk_slice("0")
        with pytest.raises(TypeError, match="chunk_index must be int"):
            managed.chunk_slice(1.5)


class TestArrayContainerMemoryType:
    """Test ArrayContainer.memory_type property."""

    def test_memory_type_returns_first_managed_array_type(self):
        """Returns the memory_type of the first managed array."""
        container = TestArraysSimple(
            arr1=ManagedArray(dtype=np_float32, memory_type="pinned"),
            arr2=ManagedArray(dtype=np_float32, memory_type="device"),
        )
        assert container.memory_type == "pinned"

    def test_memory_type_no_arrays_managed_fallback(self):
        """Returns the fallback string when no arrays are managed."""

        @attrs.define(slots=False)
        class EmptyArrayContainer(ArrayContainer):
            """Container with no ManagedArray fields."""

        container = EmptyArrayContainer()
        assert container.memory_type == "No arrays managed"


class TestCheckSizesRemainingBranches:
    """Test check_sizes branches not covered by the happy-path tests."""

    def test_check_sizes_array_name_not_in_container(
        self, test_manager_with_sizing
    ):
        """An array name absent from the container is False, not a
        KeyError."""
        arrays = {"not_a_real_array": np.zeros((1, 1, 1))}
        result = test_manager_with_sizing.check_sizes(
            arrays, location="host"
        )
        assert result["not_a_real_array"] is False

    def test_check_sizes_ndim_mismatch_is_false(
        self, test_manager_with_sizing
    ):
        """An array whose ndim differs from the expected size tuple's
        length is False rather than raising."""
        arrays = {"state": np.zeros((5, 5))}  # 2D instead of 3D
        result = test_manager_with_sizing.check_sizes(
            arrays, location="host"
        )
        assert result["state"] is False


class TestUpdateHostArrayRemainingBranches:
    """Test _update_host_array branches not covered elsewhere."""

    def test_update_host_array_none_raises(self, test_arrmgr):
        """Passing None as the new array raises ValueError."""
        with pytest.raises(ValueError, match="New array is None"):
            test_arrmgr._update_host_array(None, np.array([1, 2]), "arr1")

    def test_update_host_array_current_none_marks_new(self, test_arrmgr):
        """When current_array is None, the label is queued for both
        reallocation and overwrite and the array is attached."""
        test_arrmgr._needs_reallocation = []
        test_arrmgr._needs_overwrite = []
        new = np.array(
            [7.0, 8.0, 9.0], dtype=test_arrmgr.host.arr1.dtype
        )

        test_arrmgr._update_host_array(new, None, "arr1")

        assert "arr1" in test_arrmgr._needs_reallocation
        assert "arr1" in test_arrmgr._needs_overwrite
        assert test_arrmgr.host.arr1.array is new


class TestAllocateSkipsUnattachedArrays:
    """Test allocate() skips reallocation entries with no host array."""

    def test_allocate_skips_array_with_none_host_array(self, test_arrmgr):
        """An array marked for reallocation whose host array is None is
        skipped instead of raising an AttributeError."""
        test_arrmgr.host.arr1.array = None
        test_arrmgr._needs_reallocation = ["arr1"]

        # Should not raise despite the None host array.
        test_arrmgr.allocate()
