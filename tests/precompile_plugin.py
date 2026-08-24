"""Populate and enforce the shared CUDA test-kernel cache."""
import contextlib
import importlib
from importlib.util import find_spec
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

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


def _select_backend():
    requested = os.environ.get("CUBIE_CUDA_BACKEND", "").strip().lower()
    if requested == "mlir":
        return "mlir"
    if requested == "numba-cuda":
        return "numba-cuda"
    if requested:
        raise RuntimeError(f"Unsupported CUBIE_CUDA_BACKEND: {requested!r}")
    return "mlir" if find_spec("numba_cuda_mlir") else "numba-cuda"


BACKEND = _select_backend()

if BACKEND == "mlir":
    from cuda.core._device import ComputeCapability  # noqa: E402
    from numba_cuda_mlir import cuda as backend_cuda  # noqa: E402
    from numba_cuda_mlir import tools as mlir_tools  # noqa: E402

    if POPULATION:
        mlir_tools._cached_cc = ComputeCapability(*TARGET_CC)
else:
    import numba.cuda as backend_cuda  # noqa: E402


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
if POPULATION:
    backend_cuda.to_device = _fake_to_device
    backend_cuda.device_array = (
        lambda shape, dtype=np.float64, *args, **kwargs: _fake_device_array(
            np.zeros(shape, dtype=dtype)
        )
    )
    backend_cuda.device_array_like = (
        lambda ary, stream=0: _fake_device_array(np.zeros_like(ary))
    )
    backend_cuda.pinned_array = (
        lambda shape, dtype=np.float64, *args, **kwargs: np.zeros(
            shape, dtype=dtype
        )
    )
    backend_cuda.event = lambda timing=True: _FakeEvent()
    backend_cuda.synchronize = lambda: None
    backend_cuda.stream = lambda: _fake_stream
    backend_cuda.default_stream = lambda: _fake_stream
    backend_cuda.external_stream = lambda ptr: _fake_stream

if POPULATION and BACKEND == "numba-cuda":
    from cuda.core._device import ComputeCapability  # noqa: E402
    import numba.cuda.dispatcher as nb_dispatcher  # noqa: E402
    from numba.cuda.cudadrv import devices as nb_devices  # noqa: E402

    _fake_device = SimpleNamespace(
        compute_capability=ComputeCapability(*TARGET_CC),
        id=0,
        name=b"cubie-precompile",
        MAX_SHARED_MEMORY_PER_BLOCK=49152,
        WARP_SIZE=32,
    )
    _fake_context = SimpleNamespace(device=_fake_device)
    nb_dispatcher.get_current_device = lambda: _fake_device
    nb_devices.get_context = lambda *args, **kwargs: _fake_context
    backend_cuda.get_current_device = lambda: _fake_device

if POPULATION and BACKEND == "mlir":
    from numba_cuda_mlir.numba_cuda.cudadrv import (  # noqa: E402
        devices as mlir_devices,
    )

    # cubie queries the live device outside any launch: the
    # memory-placement heuristics call
    # ``cuda.get_current_device().compute_capability`` during Solver
    # construction (``resolve_thresholds``), which initialises the
    # real CUDA driver and raises on the driverless population
    # runner, erroring every affected fixture before its kernels
    # reach the cache. Mirror the numba-cuda fakes above.
    _fake_device = SimpleNamespace(
        compute_capability=ComputeCapability(*TARGET_CC),
        id=0,
        name=b"cubie-precompile",
        MAX_SHARED_MEMORY_PER_BLOCK=49152,
        WARP_SIZE=32,
    )
    _fake_context = SimpleNamespace(device=_fake_device)
    mlir_devices.get_context = lambda *args, **kwargs: _fake_context
    backend_cuda.get_current_device = lambda: _fake_device


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
        MISSED_KERNELS.append(_describe_compilation(self, sig))
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
            key = (
                sig,
                codegen.magic_tuple(),
                self._system_hash,
                self._compile_settings_hash,
                package_source_hash(),
                self._function_key,
            )
            if self._launch_config_key is not None:
                key += (("launch_config", self._launch_config_key),)
            return key

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


if BACKEND == "numba-cuda":
    from numba.cuda.cext import _dispatcher  # noqa: E402
    from numba.cuda.dispatcher import (  # noqa: E402
        CUDADispatcher,
        _LAUNCH_CONFIG_KW,
        _Kernel,
    )

    if POPULATION:
        _Kernel.bind = (
            lambda self: self._codelibrary.get_cubin(cc=TARGET_CC)
        )

    _dispatcher_init = CUDADispatcher.__init__
    _dispatcher_compile = CUDADispatcher.compile

    def _init_dispatcher(self, *args, **kwargs):
        _dispatcher_init(self, *args, **kwargs)
        _attach_or_queue(self)

    def _compile_dispatcher(self, *args, **kwargs):
        _attach_pending()
        _attach_cache(self)
        return _dispatcher_compile(self, *args, **kwargs)

    CUDADispatcher.__init__ = _init_dispatcher
    CUDADispatcher.compile = _compile_dispatcher

    def _precompile_numba(dispatcher, args, launch_config):
        previous_args = launch_config._push_args(args)
        try:
            selected = dispatcher._select_launch_config_dispatcher(
                launch_config
            )
            if selected is not dispatcher:
                selected._cache = dispatcher._cache

            _attach_pending()
            _attach_cache(selected)
            _dispatcher.Dispatcher._cuda_call(
                selected,
                *args,
                **{_LAUNCH_CONFIG_KW: launch_config},
            )
            return None
        finally:
            launch_config._pop_args(previous_args)

    def _numba_getitem(self, config):
        dispatcher = self
        config = list(config)
        if len(config) >= 3 and isinstance(config[2], _FakeStream):
            config[2] = 0
        launch_config = dispatcher.configure(*config)

        class _Shim:
            def __call__(self, *args):
                return _precompile_numba(
                    dispatcher, args, launch_config
                )

        return _Shim()

    if POPULATION:
        CUDADispatcher.__getitem__ = _numba_getitem
        CUDADispatcher.call = _precompile_numba
else:
    from numba_cuda_mlir.descriptor import MLIRDispatcher  # noqa: E402
    from numba_cuda_mlir.numba_cuda import types, typing  # noqa: E402
    from numba_cuda_mlir.numba_cuda.typing.typeof import (  # noqa: E402
        typeof as _mlir_typeof,
    )

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
    import cubie.memory.stream_groups as stream_groups  # noqa: E402

    _batch_solver_kernel = importlib.import_module(
        "cubie.batchsolving.BatchSolverKernel"
    )
    stream_groups.cuda = SimpleNamespace(stream=lambda: _fake_stream)
    mem_manager._ensure_cuda_context = lambda: None
    mem_manager.empty_pinned = (
        lambda shape, dtype: np.zeros(shape, dtype=dtype)
    )
    # No CUDA driver here, so the pool flush must not touch cupy.
    mem_manager.free_all_pinned_blocks = lambda: None
    _batch_solver_kernel.max_shared_memory_per_block = lambda: 49152

    _MemoryManager = mem_manager.MemoryManager
    _MemoryManager.allocate = (
        lambda self, shape, dtype, memory_type, stream=0: _fake_device_array(
            np.zeros(shape, dtype=dtype)
        )
    )
    _MemoryManager.to_device = _host_copy
    _MemoryManager.from_device = _host_copy
    _MemoryManager.get_available_memory = lambda self, group: 8 << 30
    _MemoryManager.get_memory_info = lambda self: (8 << 30, 24 << 30)

    # Read the patched figures into the already-built shared manager.
    from cubie.memory import default_memmgr as _default_memmgr  # noqa: E402

    _default_memmgr.probe_device()

    # ArrayInterpolator.get_interpolated stages its kernel arguments
    # through cupy directly; without a CUDA driver those calls raise
    # before the launch, so the kernel would never reach the cache.
    import cubie.array_interpolator as _array_interpolator  # noqa: E402

    @contextlib.contextmanager
    def _fake_cupy_stream(stream):
        yield stream

    # cupy.empty is zero-filled: nothing launches to overwrite it.
    _array_interpolator.cupy = SimpleNamespace(
        asarray=lambda a: _fake_device_array(np.array(a, copy=True)),
        empty=lambda shape, dtype=np.float64: _fake_device_array(
            np.zeros(shape, dtype=dtype)
        ),
    )
    _array_interpolator.current_cupy_stream = _fake_cupy_stream

    # Input/output chunk staging draws pinned buffers from the
    # ChunkBufferPool, which allocates through ``cupyx.empty_pinned``;
    # without a CUDA driver that raises inside every solver run's
    # fixture setup, so no batch-solver kernel would reach the cache.
    import cubie.memory.chunk_buffer_pool as _chunk_buffer_pool  # noqa: E402

    _chunk_buffer_pool.cupyx = SimpleNamespace(
        empty_pinned=lambda shape, dtype=np.float64: np.zeros(
            shape, dtype=dtype
        ),
    )

    # The busy-kernel canary fixture (tests.conftest.start_cuda_busy_work)
    # builds its non-blocking stream through ``cupy.cuda.Stream``,
    # imported from cuda_simsafe at fixture call time. Stub the stream
    # constructor while keeping the real pinned-pointer class so
    # ``is_pinned_array`` still answers correctly.
    import cubie.cuda_simsafe as _cuda_simsafe  # noqa: E402

    _real_pinned_pointer = _cuda_simsafe.cupy.cuda.PinnedMemoryPointer
    _cuda_simsafe.cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            Stream=lambda non_blocking=False: SimpleNamespace(ptr=0),
            PinnedMemoryPointer=_real_pinned_pointer,
        ),
    )


# CuBIE creates a few dispatchers while importing its cache module. Finish
# that import, then replace the temporary NullCache objects they received.
import cubie  # noqa: E402, F401

_attach_pending()


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
