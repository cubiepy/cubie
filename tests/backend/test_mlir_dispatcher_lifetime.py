"""Dropped MLIR kernels are collected once nothing references them."""

import gc
import weakref

import numpy as np
import pytest

from cubie.cuda_simsafe import cuda

pytestmark = [pytest.mark.mlir_only, pytest.mark.nocudasim]


def _make_kernel():
    @cuda.jit(device=True)
    def helper(x):
        return x + np.float32(1.0)

    @cuda.jit
    def kernel(values):
        values[0] = helper(values[0])

    return kernel


def test_dropped_kernel_is_collected_after_launch():
    kernel = _make_kernel()
    values = cuda.to_device(np.zeros(1, dtype=np.float32))
    kernel[1, 1](values)
    cuda.synchronize()
    assert values.copy_to_host()[0] == np.float32(1.0)

    kernel_ref = weakref.ref(kernel)
    del kernel
    gc.collect()

    assert kernel_ref() is None
