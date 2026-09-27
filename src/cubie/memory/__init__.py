"""
GPU memory management subsystem for cubie.

This module provides GPU memory management capabilities including:
- Stream-ordered device buffers reused by same-size or smaller
  requests
- Stream group management for asynchronous CUDA operations
- Array request/response system for structured memory allocation
- Manual or automatic allocation of VRAM to different processes
- Automatic chunking for large allocations that exceed available memory

The main components are:

- :class:`MemoryManager`: Singleton interface for managing all memory
  operations
- :class:`ArrayRequest`: Specification for array allocation requests
- :class:`ArrayResponse`: Results of array allocation operations
- :class:`StreamGroups`: Management of CUDA stream groups for coordination

The default memory manager instance is available as `default_memmgr`.
Without a device it builds anyway and raises `NoCudaDeviceError` from
any sizing decision.
"""

from cubie.memory.mem_manager import MemoryManager, NoCudaDeviceError

default_memmgr = MemoryManager()

__all__ = [
    "default_memmgr",
    "MemoryManager",
    "NoCudaDeviceError",
]
