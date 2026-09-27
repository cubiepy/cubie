"""Simulator stand-ins for ``numba_cuda_mlir.cuda.experimental``."""


def consteval(value):
    """Return ``value``; the simulator has no compile-time pass."""
    return value
