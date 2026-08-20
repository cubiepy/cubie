"""Tests for cuda_simsafe module functionality."""
import pytest


@pytest.mark.sim_only
def test_compile_kwargs_in_cudasim_mode():
    """Test that compile_kwargs is empty in CUDASIM mode."""
    from cubie.cuda_simsafe import compile_kwargs, CUDA_SIMULATION

    assert CUDA_SIMULATION is True
    assert compile_kwargs == {}


@pytest.mark.nocudasim
def test_compile_kwargs_without_cudasim():
    """Test that compile_kwargs contains lineinfo when CUDASIM is disabled."""
    from cubie.cuda_simsafe import CUDA_SIMULATION, compile_kwargs
    assert CUDA_SIMULATION is False
    assert compile_kwargs != {}


@pytest.mark.nocudasim
def test_jit_flags_render_over_live_defaults():
    """Overrides render without mutating the live default flag set."""
    from cubie.cuda_simsafe import JITFlags, compile_kwargs, get_jit_kwargs

    kwargs = get_jit_kwargs(JITFlags(afn=False, lto=False))

    expected = set(compile_kwargs["fastmath"]) - {"afn"}
    assert kwargs["fastmath"] == expected
    assert "afn" in compile_kwargs["fastmath"]
    assert kwargs["lineinfo"] == compile_kwargs["lineinfo"]
    assert kwargs["lto"] is False
    assert compile_kwargs["lto"] is True


@pytest.mark.sim_only
def test_selp_function_in_cudasim():
    """Test that selp function works in CUDASIM mode."""
    from cubie.cuda_simsafe import selp

    # Test predicated selection
    assert selp(True, 5.0, 3.0) == 5.0
    assert selp(False, 5.0, 3.0) == 3.0


@pytest.mark.sim_only
def test_activemask_function_in_cudasim():
    """Test that activemask function works in CUDASIM mode."""
    from cubie.cuda_simsafe import activemask

    # In CUDASIM mode, activemask always returns 0xFFFFFFFF
    assert activemask() == 0xFFFFFFFF


@pytest.mark.sim_only
def test_all_sync_function_in_cudasim():
    """Test that all_sync function works in CUDASIM mode."""
    from cubie.cuda_simsafe import all_sync

    # In CUDASIM mode, all_sync just returns the predicate
    assert all_sync(0xFFFFFFFF, True) is True
    assert all_sync(0xFFFFFFFF, False) is False


@pytest.mark.nocudasim
def test_narrow_f64_unflushed_under_ftz():
    """narrow_f64 keeps subnormal results where the plain cast flushes."""
    import numpy as np
    from cubie.cuda_backend import IS_MLIR
    if not IS_MLIR:
        pytest.skip("unflushed narrowing is MLIR-backend behaviour")
    from cubie.cuda_simsafe import cuda, float32, narrow_f64

    @cuda.jit(fastmath={"ftz", "contract", "nsz", "arcp", "afn"})
    def kernel(out, x):
        out[0] = narrow_f64(x)
        out[1] = float32(x)

    out = np.zeros(2, dtype=np.float32)
    kernel[1, 1](out, 1e-40)
    cuda.synchronize()
    assert out[0] == np.float32(1e-40)
    assert out[0] != 0.0
    assert out[1] == 0.0
