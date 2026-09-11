"""Driver and compiled-kernel queries with one form on every backend.

Published Objects
-----------------
:func:`compile_kernel_specialization`
    Compile a launch's specialization and raise its dynamic shared
    limit to the opt-in maximum.
:class:`DeviceHardware` / :func:`device_hardware`
    Driver-reported quantities of the current device.
:class:`KernelResources` / :func:`kernel_resources`
    Register, local-memory and SASS size of a compiled kernel.
:func:`active_blocks_per_multiprocessor`
    Resident blocks per SM at a launch geometry.
:func:`register_limited_threads` / :func:`shared_limited_threads`
    Resident threads per SM under the register or shared-memory limit.
:func:`shared_keeps_occupancy`
    Whether a shared buffer keeps the register-limited thread count.
:func:`max_shared_memory_per_block`
    Opt-in dynamic shared-memory limit per block.
"""

from struct import unpack_from
from typing import Any, Tuple

from attrs import frozen

from cubie.cuda_backend import IS_MLIR
from cubie.cuda_simsafe import CUDA_SIMULATION, cuda

SASS_INSTRUCTION_BYTES = 16
"""Bytes per SASS instruction."""

INSTRUCTION_CACHE_BYTES = {(7, 5): 65536, (8, 9): 131072}
"""Measured instruction-cache capacity by compute capability."""

DEFAULT_INSTRUCTION_CACHE_BYTES = 131072
"""Capacity assumed for an unmeasured compute capability."""

MAX_REGISTERS_PER_THREAD = 255
"""Registers a thread can address; register-capped kernels sit here."""

REGISTER_ALLOCATION_GRANULARITY = 256
"""Registers are allocated to a warp in units of this size."""

LAUNCH_BLOCKSIZES = (32, 64, 128, 256)
"""Block sizes the automatic launch chooses between."""


@frozen
class DeviceHardware:
    """Driver-reported quantities of the current device.

    Attributes
    ----------
    compute_capability
        ``(major, minor)``.
    multiprocessor_count
        SMs on the device.
    l2_cache_bytes
        L2 size in bytes.
    shared_memory_per_multiprocessor
        Shared memory per SM in bytes.
    reserved_shared_memory_per_block
        Driver reserve per resident block in bytes.
    max_dynamic_shared_memory_per_block
        Opt-in dynamic shared limit per block in bytes.
    instruction_cache_bytes
        Instruction-cache capacity from :data:`INSTRUCTION_CACHE_BYTES`.
    registers_per_multiprocessor
        32-bit registers per SM.
    max_threads_per_multiprocessor
        Resident thread limit per SM.
    max_blocks_per_multiprocessor
        Resident block limit per SM.
    warp_size
        Threads per warp.
    """

    compute_capability: Tuple[int, int]
    multiprocessor_count: int
    l2_cache_bytes: int
    shared_memory_per_multiprocessor: int
    reserved_shared_memory_per_block: int
    max_dynamic_shared_memory_per_block: int
    instruction_cache_bytes: int
    registers_per_multiprocessor: int
    max_threads_per_multiprocessor: int
    max_blocks_per_multiprocessor: int
    warp_size: int


@frozen
class KernelResources:
    """Per-thread resources of a compiled kernel.

    Attributes
    ----------
    registers_per_thread
        Registers allocated per thread.
    local_bytes_per_thread
        Local-memory frame per thread in bytes.
    sass_bytes
        Machine-code size of the kernel's ``.text`` sections.
    """

    registers_per_thread: int
    local_bytes_per_thread: int
    sass_bytes: int


def device_hardware() -> DeviceHardware:
    """Return the current device's quantities; a 48 KiB stand-in in CUDASIM."""
    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return DeviceHardware(
            compute_capability=(0, 0),
            multiprocessor_count=1,
            l2_cache_bytes=0,
            shared_memory_per_multiprocessor=49152,
            reserved_shared_memory_per_block=0,
            max_dynamic_shared_memory_per_block=49152,
            instruction_cache_bytes=DEFAULT_INSTRUCTION_CACHE_BYTES,
            registers_per_multiprocessor=65536,
            max_threads_per_multiprocessor=1024,
            max_blocks_per_multiprocessor=16,
            warp_size=32,
        )
    device = cuda.get_current_device()
    major, minor = device.compute_capability
    capability = (int(major), int(minor))
    return DeviceHardware(
        compute_capability=capability,
        multiprocessor_count=int(device.MULTIPROCESSOR_COUNT),
        l2_cache_bytes=int(device.L2_CACHE_SIZE),
        shared_memory_per_multiprocessor=int(
            device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        ),
        reserved_shared_memory_per_block=int(
            device.RESERVED_SHARED_MEMORY_PER_BLOCK
        ),
        max_dynamic_shared_memory_per_block=int(
            device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        ),
        instruction_cache_bytes=INSTRUCTION_CACHE_BYTES.get(
            capability, DEFAULT_INSTRUCTION_CACHE_BYTES
        ),
        registers_per_multiprocessor=int(
            device.MAX_REGISTERS_PER_MULTIPROCESSOR
        ),
        max_threads_per_multiprocessor=int(
            device.MAX_THREADS_PER_MULTIPROCESSOR
        ),
        max_blocks_per_multiprocessor=int(
            device.MAX_BLOCKS_PER_MULTIPROCESSOR
        ),
        warp_size=int(device.WARP_SIZE),
    )


def register_sub_partitions(hardware: DeviceHardware) -> int:
    """Return the sub-partitions an SM's register file is split into."""
    if hardware.compute_capability == (6, 0):
        return 2
    return 4


def register_limited_threads(
    hardware: DeviceHardware, registers_per_thread: int
) -> int:
    """Return the resident threads per SM the register file allows, with
    registers allocated per warp from each sub-partition's share."""
    unit = REGISTER_ALLOCATION_GRANULARITY
    warp_registers = registers_per_thread * hardware.warp_size
    warp_registers = -(-warp_registers // unit) * unit
    partitions = register_sub_partitions(hardware)
    warps_per_partition = (
        hardware.registers_per_multiprocessor // partitions
    ) // warp_registers
    threads = warps_per_partition * partitions * hardware.warp_size
    return min(threads, hardware.max_threads_per_multiprocessor)


def shared_keeps_occupancy(
    hardware: DeviceHardware, bytes_per_run: int, fraction: int = 1
) -> bool:
    """Whether ``bytes_per_run`` of shared memory per thread keeps at
    least ``1 / fraction`` of the register-limited threads resident."""
    shared_threads = shared_limited_threads(hardware, bytes_per_run)
    register_threads = register_limited_threads(
        hardware, MAX_REGISTERS_PER_THREAD
    )
    return fraction * shared_threads >= register_threads


def shared_limited_threads(
    hardware: DeviceHardware, bytes_per_run: int
) -> int:
    """Return the most resident threads per SM ``bytes_per_run`` allows."""
    best = 0
    for blocksize in LAUNCH_BLOCKSIZES:
        block_bytes = (
            bytes_per_run * blocksize
            + hardware.reserved_shared_memory_per_block
        )
        if block_bytes > hardware.max_dynamic_shared_memory_per_block:
            continue
        blocks = min(
            hardware.shared_memory_per_multiprocessor // block_bytes,
            hardware.max_threads_per_multiprocessor // blocksize,
            hardware.max_blocks_per_multiprocessor,
        )
        best = max(best, blocks * blocksize)
    return best


def max_shared_memory_per_block() -> int:
    """Return the opt-in dynamic shared-memory limit per block in bytes."""
    return device_hardware().max_dynamic_shared_memory_per_block


def _compiled_kernel_function(dispatcher: Any) -> Any:
    """Return the driver function of a dispatcher's one compiled kernel."""
    (kernel,) = dispatcher.overloads.values()
    return kernel._codelibrary.get_cufunc()


def _compiled_cubin(dispatcher: Any) -> bytes:
    """Return the cubin of a dispatcher's one compiled kernel."""
    (definition,) = dispatcher.overloads.values()
    library = definition._codelibrary
    if hasattr(library, "get_cubin"):
        return bytes(library.get_cubin().code)
    return bytes(library._cubin)


def sass_bytes_from_cubin(cubin: bytes) -> int:
    """Return the summed size of the ``.text`` sections of an ELF cubin."""
    if cubin[:4] != b"\x7fELF":
        raise ValueError("cubin is not an ELF image")
    if cubin[4] == 2:
        (section_offset,) = unpack_from("<Q", cubin, 0x28)
        entry_size, count, names_index = unpack_from("<HHH", cubin, 0x3A)
        header = "<IIQQQQ"
    else:
        (section_offset,) = unpack_from("<I", cubin, 0x20)
        entry_size, count, names_index = unpack_from("<HHH", cubin, 0x2E)
        header = "<IIIIII"

    def section(index):
        fields = unpack_from(
            header, cubin, section_offset + index * entry_size
        )
        name, _, _, _, offset, size = fields
        return name, offset, size

    _, names_offset, _ = section(names_index)
    total = 0
    for index in range(count):
        name, _, size = section(index)
        start = names_offset + name
        end = cubin.index(b"\0", start)
        if cubin[start:end].startswith(b".text."):
            total += size
    return total


def kernel_resources(dispatcher: Any) -> KernelResources:
    """Return the register, local-memory and SASS size of a compiled kernel."""
    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return KernelResources(0, 0, 0)
    (registers,) = dispatcher.get_regs_per_thread().values()
    (local_bytes,) = dispatcher.get_local_mem_per_thread().values()
    sass_bytes = sass_bytes_from_cubin(_compiled_cubin(dispatcher))
    return KernelResources(int(registers), int(local_bytes), sass_bytes)


def active_blocks_per_multiprocessor(
    dispatcher: Any, blocksize: int, dynamic_shared: int
) -> int:
    """Return the driver's resident blocks per SM at a launch geometry."""
    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return 1
    return int(
        cuda.current_context().get_active_blocks_per_multiprocessor(
            _compiled_kernel_function(dispatcher),
            int(blocksize),
            int(dynamic_shared),
        )
    )


if CUDA_SIMULATION:

    def compile_kernel_specialization(dispatcher: Any, args: Tuple) -> None:
        """No-op: the simulator interprets kernels without compiling."""

else:  # pragma: no cover - exercised in GPU environments
    if IS_MLIR:
        from cuda.bindings import driver as _cuda_binding

        def _set_function_attribute(cufunc, attribute, value) -> None:
            """Set one driver attribute on a loaded kernel function."""
            (err,) = _cuda_binding.cuFuncSetAttribute(
                cufunc.handle, attribute, int(value)
            )
            if err != _cuda_binding.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuFuncSetAttribute failed with error {err}"
                )

        def _compile(dispatcher: Any, args: Tuple) -> None:
            dispatcher.compile_for(*args)

    else:
        from numba.cuda.cudadrv.driver import (  # type: ignore
            binding as _cuda_binding,
            driver as _numba_driver,
        )

        def _set_function_attribute(cufunc, attribute, value) -> None:
            """Set one driver attribute on a loaded kernel function."""
            _numba_driver.cuKernelSetAttribute(
                attribute, int(value), cufunc.handle, cufunc.device.id
            )

        def _compile(dispatcher: Any, args: Tuple) -> None:
            argtypes = tuple(dispatcher.typeof_pyval(arg) for arg in args)
            dispatcher.compile(argtypes)

    _MAX_DYNAMIC_SHARED_ATTRIBUTE = (
        _cuda_binding.CUfunction_attribute
        .CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    )

    def compile_kernel_specialization(dispatcher: Any, args: Tuple) -> None:
        """Compile the specialization a launch with ``args`` reuses."""
        _compile(dispatcher, args)
        _set_function_attribute(
            _compiled_kernel_function(dispatcher),
            _MAX_DYNAMIC_SHARED_ATTRIBUTE,
            max_shared_memory_per_block(),
        )


__all__ = [
    "DEFAULT_INSTRUCTION_CACHE_BYTES",
    "INSTRUCTION_CACHE_BYTES",
    "LAUNCH_BLOCKSIZES",
    "MAX_REGISTERS_PER_THREAD",
    "REGISTER_ALLOCATION_GRANULARITY",
    "SASS_INSTRUCTION_BYTES",
    "DeviceHardware",
    "KernelResources",
    "active_blocks_per_multiprocessor",
    "compile_kernel_specialization",
    "device_hardware",
    "kernel_resources",
    "max_shared_memory_per_block",
    "register_limited_threads",
    "register_sub_partitions",
    "sass_bytes_from_cubin",
    "shared_keeps_occupancy",
    "shared_limited_threads",
]
