"""Tests for cubie device intrinsics and the unroll_if pass."""
import ast

import numpy as np
import pytest
from numba_cuda_mlir import cuda as ncm_cuda
from numba_cuda_mlir.ast_transforms import apply_ast_transforms
from numba_cuda_mlir.types import float32, int32

from cubie.CUDAFactory import UnrollFlags
from cubie.backend.intrinsics import narrow_f64, unroll_if
from cubie.backend.jit import compile_kwargs
from cubie.cubie_cudasim_extensions import cuda
from cubie.memory import default_memmgr

consteval = cuda.experimental.consteval


def test_consteval_loop_in_inlined_device_function():
    """A consteval loop inside an inline device function runs."""

    width = int32(4)

    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def fill(out):
        for i in consteval(range(width)):
            out[i] = consteval(i * 10)

    @cuda.jit(**compile_kwargs)
    def kernel(out):
        fill(out)

    stream = default_memmgr.get_group_stream()
    device_out = cuda.to_device(
        np.zeros(4, dtype=np.float32), stream=stream
    )
    kernel[1, 1, stream](device_out)
    out = device_out.copy_to_host(stream=stream)
    stream.synchronize()
    np.testing.assert_array_equal(out, [0.0, 10.0, 20.0, 30.0])


def test_zero_trip_consteval_loop_alone_in_if_body():
    """An if whose only statement is a zero-trip consteval loop compiles."""

    width = int32(0)
    guard = True

    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def fill(out):
        if guard:
            for i in consteval(range(width)):
                out[i] = 1.0
        out[0] = 2.0

    @cuda.jit(**compile_kwargs)
    def kernel(out):
        fill(out)

    stream = default_memmgr.get_group_stream()
    device_out = cuda.to_device(
        np.zeros(1, dtype=np.float32), stream=stream
    )
    kernel[1, 1, stream](device_out)
    out = device_out.copy_to_host(stream=stream)
    stream.synchronize()
    np.testing.assert_array_equal(out, [2.0])


@pytest.mark.nocudasim
def test_narrow_f64_unflushed_under_ftz():
    """narrow_f64 keeps subnormal results where the plain cast flushes."""

    @cuda.jit(fastmath={"ftz", "contract", "nsz", "arcp", "afn"})
    def kernel(out, x):
        out[0] = narrow_f64(x)
        out[1] = float32(x)

    out = np.zeros(2, dtype=np.float32)
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](out, 1e-40)
    stream.synchronize()
    assert out[0] == np.float32(1e-40)
    assert out[0] != 0.0
    assert out[1] == 0.0


@pytest.mark.sim_only
def test_unroll_if_passes_iterable_through_in_cudasim():
    """unroll_if returns its iterable unchanged under the simulator."""

    assert list(unroll_if(range(3), True)) == [0, 1, 2]
    assert list(unroll_if(range(3), False)) == [0, 1, 2]
    assert list(unroll_if(range(3), (True, 2))) == [0, 1, 2]
    assert list(unroll_if(range(3), True, 2)) == [0, 1, 2]


def test_unroll_if_loop_runs_under_both_flag_values():
    """A kernel with unroll_if loops computes the same either way."""

    width = int32(4)
    results = {}
    for flag_value in (True, False, (True, 2), (True, 1)):
        unroll_flag = flag_value

        @cuda.jit(device=True, inline=True, **compile_kwargs)
        def fill(out):
            for i in unroll_if(range(width), unroll_flag):
                out[i] = i * 10
            for j in unroll_if(range(width), unroll_flag, 2):
                out[j] += 1

        @cuda.jit(**compile_kwargs)
        def kernel(out):
            fill(out)

        stream = default_memmgr.get_group_stream()
        device_out = cuda.to_device(
            np.zeros(4, dtype=np.float32), stream=stream
        )
        kernel[1, 1, stream](device_out)
        out = device_out.copy_to_host(stream=stream)
        stream.synchronize()
        results[flag_value] = out
    expected = [1.0, 11.0, 21.0, 31.0]
    np.testing.assert_array_equal(results[True], expected)
    np.testing.assert_array_equal(results[False], expected)
    np.testing.assert_array_equal(results[(True, 2)], expected)
    np.testing.assert_array_equal(results[(True, 1)], expected)


def _transformed_loops(func):
    """Return (source, for-loop iterator sources) after the AST passes."""

    transformed, src = apply_ast_transforms(
        func, {"experimental_ast_transforms": True}
    )
    loops = [
        ast.unparse(node.iter)
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.For)
    ]
    return transformed, src, loops


@pytest.mark.nocudasim
def test_unroll_if_pass_resolves_closure_flags():
    """True emits the full hint, False a plain loop, count 1 the count."""

    width = 3

    def make(flag_value):
        do_unroll = flag_value

        def body(out):
            for i in unroll_if(range(width), do_unroll):
                out[i] = consteval(i * 10)

        return body

    unrolled, unrolled_src, loops = _transformed_loops(make(True))
    assert loops == ["_cubie_unroll(range(width))"]
    assert "consteval" not in unrolled_src
    assert "unroll_if" not in unrolled_src
    assert unrolled.__globals__["_cubie_unroll"] is ncm_cuda.unroll

    _, plain_src, loops = _transformed_loops(make(False))
    assert loops == ["range(width)"]
    assert "consteval" not in plain_src
    assert "unroll_if" not in plain_src

    _, rolled_src, loops = _transformed_loops(make((True, 1)))
    assert loops == ["_cubie_unroll(range(width), 1)"]
    assert "consteval" not in rolled_src


@pytest.mark.nocudasim
def test_unroll_if_pass_emits_count_hints():
    """Pair flags and explicit counts reach the count-unroll hint."""

    width = 8
    by_four = (True, 4)
    full = True
    plain = False
    depth = 2
    one = 1

    def body(out):
        for a in unroll_if(range(width), by_four):
            out[a] = a
        for b in unroll_if(range(width), full, 3):
            out[b] += b
        for c in unroll_if(range(width), by_four, depth):
            out[c] += c
        for d in unroll_if(range(width), plain, 3):
            out[d] += d
        for e in unroll_if(range(width), by_four, None):
            out[e] += e
        for f in unroll_if(range(width), full, 1):
            out[f] += f
        for g in unroll_if(range(width), by_four, one):
            out[g] += g

    _, _, loops = _transformed_loops(body)
    assert loops == [
        "_cubie_unroll(range(width), 4)",
        "_cubie_unroll(range(width), 3)",
        "_cubie_unroll(range(width), 2)",
        "range(width)",
        "_cubie_unroll(range(width), 4)",
        "_cubie_unroll(range(width), 1)",
        "_cubie_unroll(range(width), 1)",
    ]


@pytest.mark.nocudasim
def test_unroll_if_pass_rejects_zero_count():
    """A count below 1 raises at transform time."""

    full = True

    def zero_count(out):
        for i in unroll_if(range(3), full, 0):
            out[i] = i

    with pytest.raises(ValueError):
        _transformed_loops(zero_count)


@pytest.mark.nocudasim
def test_unroll_if_flag_must_be_a_closed_over_name():
    """A non-name or unresolvable flag raises at transform time."""

    def literal_flag(out):
        for i in unroll_if(range(3), True):
            out[i] = i

    with pytest.raises(TypeError):
        apply_ast_transforms(
            literal_flag, {"experimental_ast_transforms": True}
        )


@pytest.mark.nocudasim
def test_unroll_if_pass_resolves_attribute_flags():
    """A ``name.attr`` flag on a closure object resolves per attribute."""

    width = 3
    unroll = UnrollFlags(
        unroll_stage=True,
        unroll_norms=False,
        unroll_accumulator=(True, 2),
        unroll_other_small=(True, 1),
    )

    def body(out):
        for i in unroll_if(range(width), unroll.unroll_stage):
            out[i] = i
        for j in unroll_if(range(width), unroll.unroll_norms):
            out[j] = j
        for k in unroll_if(range(width), unroll.unroll_accumulator):
            out[k] = k
        for m in unroll_if(range(width), unroll.unroll_other_small):
            out[m] = m

    _, _, loops = _transformed_loops(body)
    assert loops == [
        "_cubie_unroll(range(width))",
        "range(width)",
        "_cubie_unroll(range(width), 2)",
        "_cubie_unroll(range(width), 1)",
    ]
