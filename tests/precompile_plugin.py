"""Populate and enforce the shared CUDA test-kernel cache."""
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from cuda.core._device import ComputeCapability
import numpy as np
from numba_cuda_mlir import cuda as backend_cuda
from numba_cuda_mlir import tools as mlir_tools
from numba_cuda_mlir.descriptor import MLIRDispatcher
from numba_cuda_mlir.numba_cuda import types, typing
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof as _mlir_typeof

from tests._precompile_hashing import (
    _function_key,
    _stable_value_hash,
)


def _parse_target_cc():
    raw_cc = os.environ.get("CUBIE_TARGET_CC", "").strip()
    if not raw_cc:
        raise RuntimeError(
            "tests.precompile_plugin requires CUBIE_TARGET_CC, "
            "for example '8.9'."
        )
    try:
        target_cc = tuple(
            int(part) for part in raw_cc.replace(",", ".").split(".")
        )
    except ValueError as exc:
        raise RuntimeError(
            f"CUBIE_TARGET_CC must look like '8.9', got {raw_cc!r}."
        ) from exc
    if len(target_cc) != 2 or any(part < 0 for part in target_cc):
        raise RuntimeError(
            f"CUBIE_TARGET_CC must look like '8.9', got {raw_cc!r}."
        )
    return target_cc


POPULATION = bool(os.environ.get("CUBIE_TARGET_CC", "").strip())
TARGET_CC = _parse_target_cc() if POPULATION else None

if not os.environ.get("CUBIE_KERNEL_CACHE_DIR", "").strip():
    raise RuntimeError(
        "tests.precompile_plugin requires CUBIE_KERNEL_CACHE_DIR."
    )
CACHE_DIR = Path(os.environ["CUBIE_KERNEL_CACHE_DIR"]).resolve()

# Keep every kernel in the uploaded artifact.
os.environ.setdefault("CUBIE_MAX_CACHE_ENTRIES", "0")


if POPULATION:
    # No-op the numpy comparisons so tests reach every launch.
    import numpy.testing as _np_testing

    def _population_no_op_assert(*args, **kwargs):
        return None

    for _assert_name in (
        "assert_allclose",
        "assert_almost_equal",
        "assert_approx_equal",
        "assert_array_almost_equal",
        "assert_array_almost_equal_nulp",
        "assert_array_equal",
        "assert_array_less",
        "assert_array_max_ulp",
        "assert_equal",
        "assert_string_equal",
    ):
        setattr(_np_testing, _assert_name, _population_no_op_assert)


if POPULATION:
    mlir_tools._cached_cc = ComputeCapability(*TARGET_CC)


class _FakeStream:
    handle = None

    def synchronize(self):
        pass


class _FakeDeviceArray(np.ndarray):
    """Host array with the device-copy methods cubie uses."""

    def copy_to_host(self, ary=None, stream=0):
        if ary is None:
            return np.array(self)
        np.copyto(ary, self)
        return ary

    def copy_to_device(self, ary, stream=0):
        np.copyto(self, np.asarray(ary).reshape(self.shape))

    def get(self, stream=None, order="C", out=None):
        if out is not None:
            np.copyto(out, np.array(self))
            return out
        return np.array(self)


def _fake_device_array(array):
    return array.view(_FakeDeviceArray)


def _fake_to_device(ary, stream=0, copy=True, to=None):
    array = np.array(ary, copy=True)
    if to is not None:
        np.copyto(to, array.reshape(np.shape(to)))
        return to
    return _fake_device_array(array)


class _FakeEvent:
    def record(self, stream=0):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 0.0


_fake_stream = _FakeStream()


def _fake_device_zeros(shape, dtype=np.float64, *args, **kwargs):
    return _fake_device_array(np.zeros(shape, dtype=dtype))


def _host_zeros(shape, dtype=np.float64, *args, **kwargs):
    return np.zeros(shape, dtype=dtype)


def _fake_new_event(timing=True):
    return _FakeEvent()


def _fake_new_stream(*args):
    return _fake_stream


if POPULATION:
    backend_cuda.to_device = _fake_to_device
    backend_cuda.device_array = _fake_device_zeros
    backend_cuda.pinned_array = _host_zeros
    backend_cuda.event = _fake_new_event
    backend_cuda.stream = _fake_new_stream
    backend_cuda.external_stream = _fake_new_stream


if POPULATION:
    # Driverless stand-in for the device cubie queries outside launches.
    _fake_device = SimpleNamespace(
        compute_capability=ComputeCapability(*TARGET_CC),
        id=0,
        name=b"cubie-precompile",
        MAX_SHARED_MEMORY_PER_BLOCK=49152,
        MAX_SHARED_MEMORY_PER_BLOCK_OPTIN=49152,
        MAX_SHARED_MEMORY_PER_MULTIPROCESSOR=65536,
        RESERVED_SHARED_MEMORY_PER_BLOCK=1024,
        MULTIPROCESSOR_COUNT=1,
        L2_CACHE_SIZE=4 << 20,
        WARP_SIZE=32,
        MAX_REGISTERS_PER_MULTIPROCESSOR=65536,
        MAX_THREADS_PER_MULTIPROCESSOR=1024,
        MAX_BLOCKS_PER_MULTIPROCESSOR=16,
    )
    _fake_context = SimpleNamespace(device=_fake_device)

    def _fake_current_device():
        return _fake_device

    def _fake_get_context(*args, **kwargs):
        return _fake_context

    backend_cuda.get_current_device = _fake_current_device


if POPULATION:
    from numba_cuda_mlir.numba_cuda.cudadrv import (  # noqa: E402
        devices as mlir_devices,
    )

    mlir_devices.get_context = _fake_get_context


def _host_copy(self, instance, from_arrays, to_arrays, stream=None):
    for source, destination in zip(from_arrays, to_arrays):
        if getattr(source, "size", 0):
            np.copyto(
                destination,
                np.asarray(source).reshape(np.shape(destination)),
            )


STATS = {"cache_hits": 0, "compilations_completed": 0}
MISSED_KERNELS = []
_WORKER_STATS = []
_WORKER_MISSES = []
_PENDING_DISPATCHERS = []
_CACHE_STATS_INSTALLED = False


def _combined_stats():
    combined = STATS.copy()
    for worker_stats in _WORKER_STATS:
        for key in combined:
            combined[key] += worker_stats.get(key, 0)
    return combined


def _combined_misses():
    combined = list(MISSED_KERNELS)
    for worker_misses in _WORKER_MISSES:
        combined.extend(worker_misses)
    return combined


def _describe_compilation(cache, sig):
    """Name a compiled kernel so cache misses are diagnosable.

    Production caches are named by their semantic key components
    (system name, system hash, compile-settings hash); test-kernel
    caches by their function identity. No alternate identity is
    constructed here — the description quotes the components the
    cache actually keys on.
    """
    function_key = getattr(cache, "_function_key", None)
    if function_key is not None and len(function_key) >= 5:
        identity = (
            f"{function_key[0]}.{function_key[1]} "
            f"closure={function_key[2][:10]} "
            f"code={function_key[3][:10]} "
            f"defaults={function_key[4][:10]}"
        )
    else:
        system_name = getattr(cache, "_system_name", None)
        if system_name is not None:
            identity = (
                f"{system_name} "
                f"system={cache._system_hash[:10]} "
                f"config={cache._compile_settings_hash[:10]}"
            )
        else:
            identity = getattr(cache, "_name", repr(cache))
    return f"{identity} sig={sig}"


def _install_cache_stats():
    global _CACHE_STATS_INSTALLED
    if POPULATION or _CACHE_STATS_INSTALLED:
        return
    from cubie.cubie_cache import CUBIECache

    original_load = CUBIECache.load_overload
    original_save = CUBIECache.save_overload

    def load_overload(self, sig, target_context):
        result = original_load(self, sig, target_context)
        if result is not None:
            STATS["cache_hits"] += 1
        return result

    def save_overload(self, sig, data):
        result = original_save(self, sig, data)
        STATS["compilations_completed"] += 1
        # Name the test whose launch compiled the kernel.
        current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
        MISSED_KERNELS.append(
            f"{_describe_compilation(self, sig)} test={current_test}"
        )
        return result

    CUBIECache.load_overload = load_overload
    CUBIECache.save_overload = save_overload
    _CACHE_STATS_INSTALLED = True


_TEST_KERNEL_CACHE_CLASS = None


def _test_kernel_cache_class():
    """Return the test-kernel cache class, defined on first use.

    Deferred because :mod:`cubie.cubie_cache` finishes importing only
    after the plugin's backend patches are installed. This cache
    exists only for dispatchers with no owning production factory
    (kernels defined inline in test files); every production
    dispatcher keeps the ``CUBIECache`` its factory attaches, keyed
    by the production system and configuration identity.
    """
    global _TEST_KERNEL_CACHE_CLASS
    if _TEST_KERNEL_CACHE_CLASS is not None:
        return _TEST_KERNEL_CACHE_CLASS

    from cubie._utils import package_source_hash
    from cubie.cubie_cache import CUBIECache

    class _TestKernelCache(CUBIECache):
        """Shared-artifact cache for factory-less test kernels."""

        def __init__(self, py_func, options_hash):
            super().__init__(
                system_name="pytest_kernels",
                system_hash="pytest_kernels",
                config_hash=options_hash,
                max_entries=0,
                custom_cache_dir=CACHE_DIR,
            )
            self._function_key = _function_key(py_func)

        def _index_key(self, sig, codegen):
            return (
                sig,
                codegen.magic_tuple(),
                self._system_hash,
                self._compile_settings_hash,
                package_source_hash(),
                self._function_key,
            )

    _TEST_KERNEL_CACHE_CLASS = _TestKernelCache
    return _TestKernelCache


def _attach_cache(dispatcher):
    _install_cache_stats()
    from cubie.cubie_cache import CUBIECache

    # A dispatcher with a CUBIECache got it from its production
    # factory; only factory-less dispatchers (inline test kernels)
    # take the function-keyed test cache.
    if isinstance(dispatcher._cache, CUBIECache):
        return
    dispatcher._cache = _test_kernel_cache_class()(
        dispatcher.py_func,
        _stable_value_hash(dispatcher.targetoptions),
    )


def _attach_or_queue(dispatcher):
    cache_module = sys.modules.get("cubie.cubie_cache")
    if not hasattr(cache_module, "CUBIECache"):
        _PENDING_DISPATCHERS.append(dispatcher)
        return
    _attach_cache(dispatcher)


def _attach_pending():
    while _PENDING_DISPATCHERS:
        _attach_cache(_PENDING_DISPATCHERS.pop())


_dispatcher_init = MLIRDispatcher.__init__
_dispatcher_getitem = MLIRDispatcher.__getitem__


def _marshal_launch_arg(value):
    """Normalize a launch argument the way the real launch does.

    Mirrors ``_ArgMarshaller._maybe_copy_to_device_item``'s scalar
    rules in the installed cubie-numba-cuda-mlir wheel: numpy
    integer scalars stay at their exact width (typing an
    ``np.int32`` argument as ``int32``), float64 scalars become
    Python floats, and numpy bools become Python bools before
    typing. Population signatures must match, or GPU consumers
    recompile with differently-typed scalar signatures.
    """
    if isinstance(value, (tuple, list)):
        processed = [_marshal_launch_arg(item) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*processed)
        return type(value)(processed)
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return value
    if isinstance(value, np.integer):
        return value
    if isinstance(value, (np.float16, np.float32)):
        return value
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _init_dispatcher(self, *args, **kwargs):
    _dispatcher_init(self, *args, **kwargs)
    _attach_or_queue(self)


MLIRDispatcher.__init__ = _init_dispatcher


def _precompile_mlir(dispatcher, args):
    _attach_pending()
    _attach_cache(dispatcher)
    argtypes = tuple(
        _mlir_typeof(_marshal_launch_arg(arg)) for arg in args
    )
    if argtypes in dispatcher.overloads:
        return None

    dispatcher.targetoptions["chip"] = (
        f"sm_{TARGET_CC[0]}{TARGET_CC[1]}"
    )
    signature = typing.signature(types.none, *argtypes)
    dispatcher.compile(signature)
    return None


def _mlir_getitem(self, config):
    dispatcher = self

    class _Shim:
        def __call__(self, *args):
            return _precompile_mlir(dispatcher, args)

    return _Shim()


if POPULATION:
    MLIRDispatcher.__getitem__ = _mlir_getitem
else:

    def _cached_mlir_getitem(self, config):
        _attach_pending()
        _attach_cache(self)
        return _dispatcher_getitem(self, config)

    MLIRDispatcher.__getitem__ = _cached_mlir_getitem


if POPULATION:
    # Replace eager allocations with zero-filled host arrays.
    import cubie.memory.mem_manager as mem_manager  # noqa: E402

    def _no_op(*args, **kwargs):
        return None

    def _always_idle(stream):
        return True

    def _never_pinned(array):
        return False

    mem_manager._ensure_cuda_context = _no_op
    # No CUDA driver: host buffers stand in for pinned blocks.
    mem_manager.page_locked_block = bytearray
    mem_manager.stream_idle = _always_idle
    mem_manager.flush_deferred_frees = _no_op

    # Compile the launch specialization; stand in for driver queries.
    _backend_utils = importlib.import_module("cubie.backend.utils")
    _production_compile = _backend_utils._compile

    def _population_compile_kernel_specialization(dispatcher, args):
        _attach_pending()
        _attach_cache(dispatcher)
        return _production_compile(dispatcher, args)

    def _population_kernel_resources(dispatcher, signature=None):
        return _backend_utils.KernelResources(0, 0, 0)

    def _population_active_blocks(
        dispatcher, blocksize, dynamic_shared, signature=None
    ):
        return 1

    _backend_utils.compile_kernel_specialization = (
        _population_compile_kernel_specialization
    )
    _backend_utils.kernel_resources = _population_kernel_resources
    _backend_utils.active_blocks_per_multiprocessor = (
        _population_active_blocks
    )
    for _module_name in (
        "cubie.batchsolving.BatchSolverKernel",
        "cubie.batchsolving.calibration",
        "cubie.batchsolving.optimize",
    ):
        _module = importlib.import_module(_module_name)
        for _helper_name in (
            "active_blocks_per_multiprocessor",
            "compile_kernel_specialization",
            "kernel_resources",
        ):
            if hasattr(_module, _helper_name):
                setattr(
                    _module,
                    _helper_name,
                    getattr(_backend_utils, _helper_name),
                )

    def _population_allocate(self, shape, dtype, memory_type, stream=0):
        return _fake_device_zeros(shape, dtype)

    def _population_device_view(self, settings, key, request, stream):
        return _fake_device_zeros(request.shape, request.dtype)

    def _population_available_memory(self, group):
        return 8 << 30

    def _population_memory_info(self):
        return 8 << 30, 24 << 30

    _MemoryManager = mem_manager.MemoryManager
    _MemoryManager.allocate = _population_allocate
    _MemoryManager._device_view = _population_device_view
    _MemoryManager.to_device = _host_copy
    _MemoryManager.from_device = _host_copy
    _MemoryManager.get_available_memory = _population_available_memory
    _MemoryManager.get_memory_info = _population_memory_info

    # Read the patched figures into the already-built shared manager.
    from cubie.memory import default_memmgr as _default_memmgr  # noqa: E402

    _default_memmgr.probe_device()

    # The busy-kernel fixture creates its stream through the driver.
    import cuda.bindings.driver as _cuda_driver  # noqa: E402

    def _population_stream_create(flags):
        return _cuda_driver.CUresult.CUDA_SUCCESS, 0

    def _population_stream_destroy(handle):
        return (_cuda_driver.CUresult.CUDA_SUCCESS,)

    _cuda_driver.cuStreamCreate = _population_stream_create
    _cuda_driver.cuStreamDestroy = _population_stream_destroy

    # Placeholder host buffers are never page-locked.
    for _module_name in (
        "cubie.memory.driver_memory",
        "cubie.batchsolving.BatchInputHandler",
        "cubie.batchsolving.arrays.BaseArrayManager",
    ):
        importlib.import_module(_module_name).is_pinned_array = _never_pinned

    import cubie._utils as _cubie_utils  # noqa: E402

    # Fake device arrays take the device-input path, keeping their layout.
    _real_is_device_array = _cubie_utils.is_device_array

    def _population_is_device_array(value):
        if isinstance(value, _FakeDeviceArray):
            return True
        return _real_is_device_array(value)

    for _module_name in (
        "cubie._utils",
        "cubie.batchsolving.BatchInputHandler",
        "cubie.batchsolving.arrays.BatchInputArrays",
    ):
        importlib.import_module(_module_name).is_device_array = (
            _population_is_device_array
        )


# CuBIE creates a few dispatchers while importing its cache module. Finish
# that import, then replace the temporary NullCache objects they received.
import cubie  # noqa: E402, F401

_attach_pending()

# A pool that never pays keeps candidate compiles in this process.
import cubie.batchsolving.comparison as _comparison  # noqa: E402

_comparison.WORKER_STARTUP_SECONDS = float("inf")


def pytest_configure(config):
    _attach_pending()


def pytest_testnodedown(node, error):
    if POPULATION:
        return
    workeroutput = getattr(node, "workeroutput", {})
    worker_stats = workeroutput.get("cubie_kernel_cache_stats")
    if worker_stats:
        _WORKER_STATS.append(worker_stats)
    worker_misses = workeroutput.get("cubie_kernel_cache_misses")
    if worker_misses:
        _WORKER_MISSES.append(worker_misses)


def pytest_sessionfinish(session, exitstatus):
    if POPULATION:
        return
    if hasattr(session.config, "workeroutput"):
        session.config.workeroutput["cubie_kernel_cache_stats"] = STATS.copy()
        session.config.workeroutput["cubie_kernel_cache_misses"] = list(
            MISSED_KERNELS
        )
        return
    if _combined_stats()["compilations_completed"]:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter):
    if POPULATION:
        return
    stats = _combined_stats()
    terminalreporter.write_line(
        f"KERNEL_CACHE cache_hits={stats['cache_hits']} "
        "compilations_completed="
        f"{stats['compilations_completed']}"
    )
    for description in sorted(_combined_misses()):
        terminalreporter.write_line(f"KERNEL_CACHE MISS {description}")
