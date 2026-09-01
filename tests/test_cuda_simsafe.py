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


@pytest.mark.sim_only
def test_consteval_passes_iterable_through_in_cudasim():
    """consteval returns its argument unchanged under the simulator."""
    from cubie.cuda_simsafe import consteval

    assert list(consteval(range(3))) == [0, 1, 2]
    assert consteval(7) == 7


@pytest.mark.nocudasim
def test_jit_kwargs_carry_backend_ast_transform_flag():
    """MLIR builds request the AST transforms; numba-cuda builds do not."""
    from cubie.cuda_backend import IS_MLIR
    from cubie.cuda_simsafe import compile_kwargs, get_jit_kwargs

    kwargs = get_jit_kwargs()
    if IS_MLIR:
        assert kwargs["experimental_ast_transforms"] is True
        assert compile_kwargs["experimental_ast_transforms"] is True
    else:
        assert set(kwargs) == {"fastmath", "lineinfo", "lto"}
        assert set(compile_kwargs) == {"fastmath", "lineinfo", "lto"}


def test_consteval_loop_in_inlined_device_function():
    """A consteval loop inside an inline device function runs."""
    import numpy as np
    from cubie.cuda_simsafe import (
        compile_kwargs,
        consteval,
        cuda,
        int32,
    )
    from cubie.memory import default_memmgr

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
    import numpy as np
    from cubie.cuda_simsafe import (
        compile_kwargs,
        consteval,
        cuda,
        int32,
    )
    from cubie.memory import default_memmgr

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
    """narrow_f64 keeps subnormal results where the plain cast flushes."""
    import numpy as np
    from cubie.cuda_backend import IS_MLIR
    if not IS_MLIR:
        pytest.skip("unflushed narrowing is MLIR-backend behaviour")
    from cubie.cuda_simsafe import cuda, float32, narrow_f64
    from cubie.memory import default_memmgr

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
    """unroll_if returns its iterable unchanged under the simulator."""
    from cubie.cuda_simsafe import unroll_if

    assert list(unroll_if(range(3), True)) == [0, 1, 2]
    assert list(unroll_if(range(3), False)) == [0, 1, 2]
    assert list(unroll_if(range(3), (True, 2))) == [0, 1, 2]
    assert list(unroll_if(range(3), True, 2)) == [0, 1, 2]


def test_unroll_if_loop_runs_under_both_flag_values():
    """A kernel with unroll_if loops computes the same either way."""
    import numpy as np
    from cubie.cuda_simsafe import (
        compile_kwargs,
        cuda,
        int32,
        unroll_if,
    )
    from cubie.memory import default_memmgr

    width = int32(4)
    results = {}
    for flag_value in (True, False, (True, 2)):
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


def _transformed_loops(func):
    """Return (source, for-loop iterator sources) after the AST passes."""
    import ast
    from numba_cuda_mlir.ast_transforms import apply_ast_transforms

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
    """A true flag emits the full-unroll hint, false the disable hint."""
    from cubie.cuda_backend import IS_MLIR
    if not IS_MLIR:
        pytest.skip("the UnrollIf pass is MLIR-backend behaviour")
    from numba_cuda_mlir import cuda as ncm_cuda
    from cubie.cuda_simsafe import consteval, unroll_if

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
    assert unrolled.__globals__["_cubie_nounroll"] is ncm_cuda.nounroll

    _, rolled_src, loops = _transformed_loops(make(False))
    assert loops == ["_cubie_nounroll(range(width))"]
    assert "consteval" not in rolled_src
    assert "unroll_if" not in rolled_src


@pytest.mark.nocudasim
def test_unroll_if_pass_emits_count_hints():
    """Pair flags and explicit counts reach the count-unroll hint."""
    from cubie.cuda_backend import IS_MLIR
    if not IS_MLIR:
        pytest.skip("the UnrollIf pass is MLIR-backend behaviour")
    from cubie.cuda_simsafe import unroll_if

    width = 8
    by_four = (True, 4)
    full = True
    rolled = False
    depth = 2

    def body(out):
        for a in unroll_if(range(width), by_four):
            out[a] = a
        for b in unroll_if(range(width), full, 3):
            out[b] += b
        for c in unroll_if(range(width), by_four, depth):
            out[c] += c
        for d in unroll_if(range(width), rolled, 3):
            out[d] += d
        for e in unroll_if(range(width), by_four, None):
            out[e] += e

    _, _, loops = _transformed_loops(body)
    assert loops == [
        "_cubie_unroll(range(width), 4)",
        "_cubie_unroll(range(width), 3)",
        "_cubie_unroll(range(width), 2)",
        "_cubie_nounroll(range(width))",
        "_cubie_unroll(range(width), 4)",
    ]


@pytest.mark.nocudasim
def test_unroll_if_pass_rejects_bad_counts():
    """A count below 1 or of the wrong type raises at transform time."""
    from cubie.cuda_backend import IS_MLIR
    if not IS_MLIR:
        pytest.skip("the UnrollIf pass is MLIR-backend behaviour")
    from cubie.cuda_simsafe import unroll_if

    full = True

    def zero_count(out):
        for i in unroll_if(range(3), full, 0):
            out[i] = i

    def float_count(out):
        for i in unroll_if(range(3), full, 2.0):
            out[i] = i

    with pytest.raises(ValueError):
        _transformed_loops(zero_count)
    with pytest.raises(TypeError):
        _transformed_loops(float_count)


@pytest.mark.nocudasim
def test_unroll_if_flag_must_be_a_closed_over_name():
    """A non-name or unresolvable flag raises at transform time."""
    from cubie.cuda_backend import IS_MLIR
    if not IS_MLIR:
        pytest.skip("the UnrollIf pass is MLIR-backend behaviour")
    from numba_cuda_mlir.ast_transforms import apply_ast_transforms
    from cubie.cuda_simsafe import unroll_if

    def literal_flag(out):
        for i in unroll_if(range(3), True):
            out[i] = i

    with pytest.raises(TypeError):
        apply_ast_transforms(
            literal_flag, {"experimental_ast_transforms": True}
        )


def test_unroll_flags_update_reads_prefixed_keys():
    """``unroll_*`` keys derive a replacement; other keys are ignored."""
    from cubie.cuda_simsafe import ALL_UNROLL_PARAMETERS, UnrollFlags

    flags = UnrollFlags()
    replacement, recognised, changed = flags.update(
        {"unroll_norms": False, "unroll_stage": True, "lineinfo": True}
    )
    assert recognised == {"unroll_norms", "unroll_stage"}
    assert changed == {"unroll_norms"}
    assert replacement.norms == (False, None)
    assert replacement.stage == (True, None)
    assert flags.norms == (True, None)
    assert "unroll_other_small" in ALL_UNROLL_PARAMETERS
    assert UnrollFlags.from_loose(
        {"unroll_accumulator": False}
    ).accumulator == (False, None)

    counted, recognised, changed = replacement.update(
        {"unroll_stage": (True, 4), "unroll_norms": (False, None)}
    )
    assert recognised == {"unroll_stage", "unroll_norms"}
    assert changed == {"unroll_stage"}
    assert counted.stage == (True, 4)
    assert counted.norms == (False, None)
    assert counted.update({"unroll_stage": [True, 4]})[2] == set()


def test_normalise_unroll_flag_forms():
    """Bools and pairs normalise to ``(unroll, count)``; bad forms raise."""
    import numpy as np
    from cubie.cuda_simsafe import UnrollFlags, normalise_unroll_flag

    assert normalise_unroll_flag(True) == (True, None)
    assert normalise_unroll_flag(False) == (False, None)
    assert normalise_unroll_flag((True, None)) == (True, None)
    assert normalise_unroll_flag((True, 4)) == (True, 4)
    assert normalise_unroll_flag([True, 4]) == (True, 4)
    assert normalise_unroll_flag((True, np.int64(3))) == (True, 3)
    assert normalise_unroll_flag((False, None)) == (False, None)
    with pytest.raises(ValueError):
        normalise_unroll_flag((False, 4))
    with pytest.raises(ValueError):
        normalise_unroll_flag((True, 0))
    with pytest.raises(TypeError):
        normalise_unroll_flag((True, True))
    with pytest.raises(TypeError):
        normalise_unroll_flag((True, 2.0))
    with pytest.raises(TypeError):
        normalise_unroll_flag((1, None))
    with pytest.raises(TypeError):
        normalise_unroll_flag("yes")
    with pytest.raises(TypeError):
        normalise_unroll_flag((True, 2, 3))
    with pytest.raises(ValueError):
        UnrollFlags(stage=(False, 2))
    assert UnrollFlags(stage=(True, 2)) == UnrollFlags(stage=[True, 2])
    assert UnrollFlags(stage=True) == UnrollFlags(stage=(True, None))


@pytest.mark.nocudasim
def test_unroll_if_pass_resolves_attribute_flags():
    """A ``name.attr`` flag on a closure object resolves per attribute."""
    from cubie.cuda_backend import IS_MLIR
    if not IS_MLIR:
        pytest.skip("the UnrollIf pass is MLIR-backend behaviour")
    from cubie.cuda_simsafe import UnrollFlags, unroll_if

    width = 3
    unroll = UnrollFlags(stage=True, norms=False, accumulator=(True, 2))

    def body(out):
        for i in unroll_if(range(width), unroll.stage):
            out[i] = i
        for j in unroll_if(range(width), unroll.norms):
            out[j] = j
        for k in unroll_if(range(width), unroll.accumulator):
            out[k] = k

    _, _, loops = _transformed_loops(body)
    assert loops == [
        "_cubie_unroll(range(width))",
        "_cubie_nounroll(range(width))",
        "_cubie_unroll(range(width), 2)",
    ]
