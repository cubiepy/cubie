import pytest
from cubie.cuda_simsafe import cuda
from cubie.memory.array_requests import ArrayRequest
from cubie.memory.mem_manager import InstanceMemorySettings, MemoryManager
import numpy as np


@pytest.fixture(scope="session")
def array_request_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="session")
def array_request_settings(array_request_override):
    """Fixture to provide settings for ArrayRequest."""
    defaults = {
        "shape": (1, 1, 1),
        "dtype": np.float32,
        "memory": "device",
        "total_runs": 1,
    }
    if array_request_override:
        for key, value in array_request_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


@pytest.fixture(scope="session")
def array_request(array_request_settings):
    return ArrayRequest(**array_request_settings)


@pytest.fixture(scope="session")
def expected_single_array(array_request_settings):
    arr_request = array_request_settings
    if arr_request["memory"] == "device":
        arr = cuda.device_array(
            array_request_settings["shape"],
            dtype=array_request_settings["dtype"],
        )
    elif arr_request["memory"] == "pinned":
        arr = cuda.pinned_array(
            array_request_settings["shape"],
            dtype=array_request_settings["dtype"],
        )
    else:
        raise ValueError(f"Invalid memory type: {arr_request['memory']}")
    return arr


class MemoryClient:
    """Minimal registrant standing in for an array manager.

    Exposes the attributes and hooks that ``MemoryManager.register``
    expects from a caller: a proportion, an invalidate-all hook, and
    notification callbacks that record how the manager called back
    into it.
    """

    def __init__(self, proportion=None, invalidate_all_hook=None):
        self.proportion = proportion
        self.invalidate_all_hook = invalidate_all_hook
        self.last_response = None

    def notice_invalidate(self):
        self.proportion = None

    def notice_allocation(self, response):
        self.last_response = response


class DummyStream:
    """Stand-in for a CUDA stream exposing only ``handle``."""

    def __init__(self, handle):
        self.handle = handle


class FakeAllocation:
    """Stand-in allocation exposing only ``nbytes``."""

    def __init__(self, nbytes):
        self.nbytes = nbytes


@pytest.fixture(scope="function")
def memory_client_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def memory_client_settings(memory_client_override):
    defaults = {
        "proportion": None,
        "invalidate_all_hook": None,
    }
    if memory_client_override:
        for key, value in memory_client_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


@pytest.fixture(scope="function")
def memory_client(memory_client_settings):
    """Return a single fresh MemoryClient instance."""
    return MemoryClient(**memory_client_settings)


@pytest.fixture(scope="function")
def memory_clients(request):
    """List of fresh registrants; parametrize the count indirectly."""
    count = request.param if hasattr(request, "param") else 2
    return [MemoryClient() for _ in range(count)]


@pytest.fixture(scope="function")
def instance_settings_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def instance_settings(instance_settings_override):
    defaults = {
        "proportion": 0.5,
        "invalidate_hook": lambda: None,
    }
    if instance_settings_override:
        for key, value in instance_settings_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


@pytest.fixture(scope="function")
def instance_settings_obj(instance_settings):
    return InstanceMemorySettings(**instance_settings)


# ========================= Memory Manager Class =========================== #
@pytest.fixture
def fixed_mem_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def fixed_mem_settings(fixed_mem_override):
    defaults = {"free": 1 * 1024**3, "total": 8 * 1024**3}
    if fixed_mem_override:
        for key, value in fixed_mem_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


@pytest.fixture(scope="function")
def mem_manager_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def mem_manager_settings(mem_manager_override):
    # Reported sizes are synthetic, so the device pool's reserve
    # granule is not part of the fake.
    defaults = {"mode": "passive", "allocation_granule_bytes": 0}
    if mem_manager_override:
        for key, value in mem_manager_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


# Override memory info for consistent tests
@pytest.fixture(scope="function")
def mgr(fixed_mem_settings, mem_manager_settings):
    class TestMemoryManager(MemoryManager):
        def get_memory_info(self):
            free = fixed_mem_settings["free"]
            total = fixed_mem_settings["total"]
            return free, total

    # Create an instance of the TestMemoryManager with the provided settings
    return TestMemoryManager(**mem_manager_settings)


@pytest.fixture(scope="function")
def registered_instance_override(request):
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def registered_instance_settings(registered_instance_override):
    defaults = {
        "proportion": 0.5,
        "invalidate_all_hook": lambda: None,
    }
    if registered_instance_override:
        for key, value in registered_instance_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


@pytest.fixture(scope="function")
def registered_instance(registered_instance_settings):
    return MemoryClient(**registered_instance_settings)


@pytest.fixture(scope="function")
def registered_mgr(mgr, registered_instance):
    mgr.register(
        registered_instance,
        proportion=registered_instance.proportion,
        invalidate_cache_hook=registered_instance.invalidate_all_hook,
    )
    return mgr


def registered_mgr_context_safe(
    mem_manager_settings, registered_instance_settings
):
    fixed_mem_settings_closure = {"free": 1 * 1024**3, "total": 8 * 1024**3}

    class TestMemoryManager(MemoryManager):
        def get_memory_info(self):
            free = fixed_mem_settings_closure["free"]
            total = fixed_mem_settings_closure["total"]
            return free, total

    manager = TestMemoryManager(**mem_manager_settings)
    registeree = MemoryClient(**registered_instance_settings)

    manager.register(
        registeree,
        proportion=registeree.proportion,
        invalidate_cache_hook=registeree.invalidate_all_hook,
    )

    return manager, registeree
