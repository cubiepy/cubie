"""Phase-time a fresh process's first CUDA launch at one stack layer."""
import ctypes
import subprocess
import sys
from time import perf_counter


def _fail(name, detail):
    print(f"{name} failed: {detail}", file=sys.stderr)
    sys.exit(1)


def _check(return_code, name):
    if return_code != 0:
        _fail(name, f"CUresult {return_code}")


def probe_driver():
    """Time the raw driver path with ctypes; print per-phase seconds."""
    t0 = perf_counter()
    if sys.platform == "win32":
        lib = ctypes.WinDLL("nvcuda.dll")
    else:
        lib = ctypes.CDLL("libcuda.so.1")
    t1 = perf_counter()

    _check(lib.cuInit(0), "cuInit")
    t2 = perf_counter()

    device = ctypes.c_int()
    get_device = lib.cuDeviceGet
    get_device.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    _check(get_device(ctypes.byref(device), 0), "cuDeviceGet")
    t3 = perf_counter()

    context = ctypes.c_void_p()
    retain = lib.cuDevicePrimaryCtxRetain
    retain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    _check(retain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
    set_current = lib.cuCtxSetCurrent
    set_current.argtypes = [ctypes.c_void_p]
    _check(set_current(context), "cuCtxSetCurrent")
    t4 = perf_counter()

    n_bytes = 4096
    device_pointer = ctypes.c_uint64()
    mem_alloc = lib.cuMemAlloc_v2
    mem_alloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    _check(mem_alloc(ctypes.byref(device_pointer), n_bytes), "cuMemAlloc")
    t5 = perf_counter()

    host_buffer = (ctypes.c_ubyte * n_bytes)()
    host_to_device = lib.cuMemcpyHtoD_v2
    host_to_device.argtypes = [
        ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t
    ]
    _check(
        host_to_device(device_pointer, host_buffer, n_bytes),
        "cuMemcpyHtoD",
    )
    t6 = perf_counter()

    device_to_host = lib.cuMemcpyDtoH_v2
    device_to_host.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t
    ]
    _check(
        device_to_host(host_buffer, device_pointer, n_bytes),
        "cuMemcpyDtoH",
    )
    t7 = perf_counter()

    print(
        f"driver: lib_load={t1 - t0:.3f}s cuinit={t2 - t1:.3f}s "
        f"device_get={t3 - t2:.3f}s primary_ctx={t4 - t3:.3f}s "
        f"first_alloc={t5 - t4:.3f}s h2d={t6 - t5:.3f}s "
        f"d2h={t7 - t6:.3f}s total={t7 - t0:.3f}s"
    )


def probe_numba():
    """Time numba.cuda import, context creation, and first transfer."""
    from numpy import zeros
    host_array = zeros(8)
    t0 = perf_counter()
    from numba import cuda
    t1 = perf_counter()
    cuda.current_context()
    t2 = perf_counter()
    device_array = cuda.to_device(host_array)
    t3 = perf_counter()
    device_array.copy_to_host()
    t4 = perf_counter()
    print(
        f"numba: import={t1 - t0:.3f}s context={t2 - t1:.3f}s "
        f"to_device={t3 - t2:.3f}s copy_back={t4 - t3:.3f}s "
        f"total={t4 - t0:.3f}s"
    )


def probe_cupy():
    """Time cupy import, runtime query, and first pool allocation."""
    import numpy  # noqa: F401  (untimed in every mode)
    t0 = perf_counter()
    import cupy
    t1 = perf_counter()
    cupy.cuda.runtime.getDeviceCount()
    t2 = perf_counter()
    array = cupy.zeros(8)
    cupy.cuda.Device().synchronize()
    t3 = perf_counter()
    array.get()
    t4 = perf_counter()
    print(
        f"cupy: import={t1 - t0:.3f}s device_count={t2 - t1:.3f}s "
        f"first_alloc={t3 - t2:.3f}s copy_back={t4 - t3:.3f}s "
        f"total={t4 - t0:.3f}s"
    )


def probe_cubie():
    """Time cubie import (creates the context) and first transfer."""
    from numpy import zeros
    host_array = zeros(8)
    t0 = perf_counter()
    import cubie  # noqa: F401
    t1 = perf_counter()
    from cubie.cuda_simsafe import cuda
    device_array = cuda.to_device(host_array)
    t2 = perf_counter()
    device_array.copy_to_host()
    t3 = perf_counter()
    print(
        f"cubie: import={t1 - t0:.3f}s to_device={t2 - t1:.3f}s "
        f"copy_back={t3 - t2:.3f}s total={t3 - t0:.3f}s"
    )


def probe_kernel():
    """Time context, eager kernel compile, first launch, and sync."""
    from numpy import zeros
    host_array = zeros(8)
    t0 = perf_counter()
    from numba import cuda
    t1 = perf_counter()
    cuda.current_context()
    device_array = cuda.to_device(host_array)
    t2 = perf_counter()

    @cuda.jit("void(float64[::1])")
    def _bump(array):  # pragma: no cover
        index = cuda.grid(1)
        if index < array.size:
            array[index] += 1.0

    t3 = perf_counter()
    _bump[1, 32](device_array)
    t4 = perf_counter()
    cuda.synchronize()
    device_array.copy_to_host()
    t5 = perf_counter()
    print(
        f"kernel: import={t1 - t0:.3f}s context={t2 - t1:.3f}s "
        f"compile={t3 - t2:.3f}s launch={t4 - t3:.3f}s "
        f"sync={t5 - t4:.3f}s total={t5 - t0:.3f}s"
    )


def fanout(mode, count):
    """Start count fresh probe processes at once; print each output."""
    t0 = perf_counter()
    children = [
        subprocess.Popen(
            [sys.executable, __file__, mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(count)
    ]
    status = 0
    for index, child in enumerate(children):
        output, _ = child.communicate()
        elapsed = perf_counter() - t0
        status |= child.returncode
        print(f"[{index}] done_at={elapsed:.3f}s {output.strip()}")
    sys.exit(status)


def probe_solver():
    """Time cubie import, system build, first solve, second solve."""
    t0 = perf_counter()
    from cubie import create_ODE_system, solve_ivp
    t1 = perf_counter()
    system = create_ODE_system(
        dxdt=["dx = -k * x"],
        states={"x": 1.0},
        parameters={"k": 0.5},
        name="first_launch_probe",
    )
    t2 = perf_counter()
    solve_ivp(system, y0={"x": 1.0}, duration=0.1, dt=0.01)
    t3 = perf_counter()
    solve_ivp(system, y0={"x": 2.0}, duration=0.1, dt=0.01)
    t4 = perf_counter()
    print(
        f"solver: import={t1 - t0:.3f}s build={t2 - t1:.3f}s "
        f"first_solve={t3 - t2:.3f}s second_solve={t4 - t3:.3f}s "
        f"total={t4 - t0:.3f}s"
    )


PROBES = {
    "driver": probe_driver,
    "numba": probe_numba,
    "cupy": probe_cupy,
    "cubie": probe_cubie,
    "kernel": probe_kernel,
    "solver": probe_solver,
}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "fanout":
        fanout(sys.argv[2], int(sys.argv[3]))
    if len(sys.argv) != 2 or sys.argv[1] not in PROBES:
        _fail(
            "usage",
            f"first_launch_split.py {{{'|'.join(PROBES)}}} "
            "| fanout <mode> <n>",
        )
    PROBES[sys.argv[1]]()
