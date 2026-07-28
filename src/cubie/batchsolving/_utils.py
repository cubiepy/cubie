"""Shared helpers for batch solver components.

See Also
--------
:mod:`cubie.batchsolving`
    Parent package exporting batch solving infrastructure.
"""

from typing import Any, Callable, Dict

from cubie.cuda_simsafe import cuda

__all__ = ["name_and_compile_kernel"]


def name_and_compile_kernel(
    kernel_function: Callable,
    kernel_name: str,
    jit_kwargs: Dict[str, Any],
) -> Callable:
    """Rename a kernel function and compile it for the device.

    Parameters
    ----------
    kernel_function
        Undecorated kernel function to rename and compile.
    kernel_name
        Name the compiled kernel appears under in profiler and
        disassembly output.
    jit_kwargs
        Keyword arguments forwarded to :func:`numba.cuda.jit`.

    Returns
    -------
    Callable
        Compiled kernel dispatcher carrying ``kernel_name``.

    Notes
    -----
    numba mangles the device symbol from ``__qualname__``, so its last
    segment is renamed alongside ``__name__``.
    """
    qualname_parts = kernel_function.__qualname__.rsplit(".", 1)
    qualname_parts[-1] = kernel_name
    kernel_function.__name__ = kernel_name
    kernel_function.__qualname__ = ".".join(qualname_parts)
    return cuda.jit(**jit_kwargs)(kernel_function)
