"""Tests for the cuda module and its simulator extensions."""
import numpy as np

from cubie.backend.jit import compile_kwargs
from cubie._cudasim_extensions import cuda
from cubie.memory import default_memmgr


def test_warp_intrinsics_in_a_kernel():
    """selp, activemask, syncwarp, all_sync and any_sync run in a kernel."""

    @cuda.jit(**compile_kwargs)
    def kernel(out, flag):
        mask = cuda.activemask()
        cuda.syncwarp(mask)
        out[0] = cuda.selp(True, 5.0, 3.0)
        out[1] = cuda.selp(False, 5.0, 3.0)
        out[2] = 1.0 if mask == 0xFFFFFFFF else 0.0
        out[3] = 1.0 if cuda.all_sync(mask, flag > 0) else 0.0
        out[4] = 1.0 if cuda.any_sync(mask, flag > 0) else 0.0

    stream = default_memmgr.get_group_stream()
    device_out = cuda.to_device(
        np.zeros(5, dtype=np.float32), stream=stream
    )
    kernel[1, 32, stream](device_out, np.int32(1))
    out = device_out.copy_to_host(stream=stream)
    stream.synchronize()
    np.testing.assert_array_equal(out, [5.0, 3.0, 1.0, 1.0, 1.0])

