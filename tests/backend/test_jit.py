"""Tests for the cuda.jit keyword arguments."""
import pytest

from cubie.CUDAFactory import JITFlags
from cubie.backend.jit import compile_kwargs, get_jit_kwargs
from cubie.cubie_cudasim_extensions import CUDA_SIMULATION


@pytest.mark.nocudasim
def test_compile_kwargs_without_cudasim():
    """Test that compile_kwargs contains lineinfo when CUDASIM is disabled."""
    assert CUDA_SIMULATION is False
    assert compile_kwargs != {}


@pytest.mark.nocudasim
def test_jit_flags_render_over_live_defaults():
    """Overrides render without mutating the live default flag set."""

    kwargs = get_jit_kwargs(JITFlags(afn=False, lto=False))

    expected = set(compile_kwargs["fastmath"]) - {"afn"}
    assert kwargs["fastmath"] == expected
    assert "afn" in compile_kwargs["fastmath"]
    assert kwargs["lineinfo"] == compile_kwargs["lineinfo"]
    assert kwargs["lto"] is False
    assert compile_kwargs["lto"] is True


def test_jit_kwargs_carry_ast_transform_flag():
    """Every build requests the MLIR AST transforms."""

    kwargs = get_jit_kwargs()
    assert kwargs["experimental_ast_transforms"] is True
    assert compile_kwargs["experimental_ast_transforms"] is True
