"""
GPU memory management subsystem for cubie.

This module provides GPU memory management capabilities including:
- CuPy async memory pool backing native Numba device arrays via an EMM plugin
- Stream group management for asynchronous CUDA operations
- Array request/response system for structured memory allocation
- Manual or automatic allocation of VRAM to different processes
- Automatic chunking for large allocations that exceed available memory

The main components are:

- :class:`MemoryManager`: Singleton interface for managing all memory operations
- :class:`ArrayRequest`: Specification for array allocation requests
- :class:`ArrayResponse`: Results of array allocation operations
- :class:`StreamGroups`: Management of CUDA stream groups for coordination
- :class:`current_cupy_stream`: Context manager for CuPy stream integration

The default memory manager instance is available as `default_memmgr`.
Without a device it builds anyway and raises `NoCudaDeviceError` from
any sizing decision.
"""

from cubie.memory.cupy_emm import CuPyAsyncNumbaManager, install_async_emm
from cubie.memory.mem_manager import (
    MemoryManager,
    NoCudaDeviceError,
    current_cupy_stream,
)

# Install the CuPy async pool as Numba's EMM before the first CUDA context is
# created (below, when the manager queries device memory).
install_async_emm()

default_memmgr = MemoryManager()

__all__ = [
    "current_cupy_stream",
    "CuPyAsyncNumbaManager",
    "default_memmgr",
    "MemoryManager",
    "NoCudaDeviceError",
]
