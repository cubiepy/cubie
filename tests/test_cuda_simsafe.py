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


def _plain_loops_fixture_step():
    from cubie.cuda_simsafe import consteval

    stages = 2
    width = 3

    def step(out, flags, table):
        for prev_idx in consteval(range(stages)):
            if consteval(flags[prev_idx + 1]):
                for idx in consteval(range(width)):
                    out[prev_idx * width + idx] = table[prev_idx]

    return step


@pytest.mark.nocudasim
def test_plain_loop_class_rules():
    """Loop classes follow module, qualname, loop variable and nesting."""
    pytest.importorskip("numba_cuda_mlir")
    from cubie.backend._mlir_compat import plain_loop_class

    algorithms = "cubie.integrators.algorithms."
    norms = "cubie.integrators.norms"
    cases = [
        (algorithms + "generic_dirk", "step", "prev_idx", True,
         "stage_outer"),
        (algorithms + "generic_dirk", "step", "idx", False, "step_vectors"),
        (algorithms + "generic_firk", "step", "stage_idx", True,
         "firk_assembly"),
        (algorithms + "generic_firk", "step", "stage_idx", False,
         "step_vectors"),
        (norms, "DIRKCorrectionNorm.build.<locals>.correction_norm", "i",
         True, "solver_norms"),
        (norms, "TwoRefMaskedScaledNorm.build.<locals>.scaled_norm", "i",
         True, "controller_norm"),
        ("jacobi_preconditioner_plain_s0abc", "apply", "i", True,
         "krylov_body"),
        ("cubie.integrators.loops.ode_loop", "loop_fn", "i", False, "fills"),
        ("cubie.batchsolving.other", "f", "i", True, None),
    ]
    for module, qualname, loop_var, outer, expected in cases:
        assert plain_loop_class(module, qualname, loop_var, outer) == expected


@pytest.mark.nocudasim
def test_plain_loops_pass_rewrites_active_class_only():
    """Active classes lose consteval on the loop and its dependent tests."""
    pytest.importorskip("numba_cuda_mlir")
    import ast
    from numba_cuda_mlir.ast_transforms import (
        TransformContext,
        create_default_pipeline,
    )
    from numba_cuda_mlir.ast_transforms.common import get_function_ast
    from cubie._env import active_plain_loops, set_active_plain_loops

    func = _plain_loops_fixture_step()
    func.__module__ = "cubie.integrators.algorithms.generic_dirk"
    saved = active_plain_loops()
    try:
        set_active_plain_loops({"stage_outer"})
        tree = get_function_ast(func)
        pipeline = create_default_pipeline()
        first_pass = pipeline._passes[0]
        tree, modified = first_pass.transform(
            tree, TransformContext(func=func)
        )
        source = ast.unparse(tree)
        assert modified
        assert "for prev_idx in range(stages):" in source
        assert "if flags[prev_idx + 1]:" in source
        assert "for idx in consteval(range(width)):" in source

        set_active_plain_loops(set())
        tree = get_function_ast(func)
        tree, modified = first_pass.transform(
            tree, TransformContext(func=func)
        )
        assert not modified
        assert "for prev_idx in consteval(range(stages)):" in ast.unparse(tree)
    finally:
        set_active_plain_loops(saved)


def test_plain_loops_setting_parses_classes(monkeypatch):
    """CUBIE_PLAIN_LOOPS names classes, accepts all, rejects unknown."""
    from cubie._env import PLAIN_LOOP_CLASSES, plain_loops_default

    monkeypatch.setenv("CUBIE_PLAIN_LOOPS", "newton_body, krylov_body")
    assert plain_loops_default() == {"newton_body", "krylov_body"}
    monkeypatch.setenv("CUBIE_PLAIN_LOOPS", "all")
    assert plain_loops_default() == set(PLAIN_LOOP_CLASSES)
    monkeypatch.setenv("CUBIE_PLAIN_LOOPS", "")
    assert plain_loops_default() == frozenset()
    monkeypatch.setenv("CUBIE_PLAIN_LOOPS", "stage_inner")
    with pytest.raises(ValueError):
        plain_loops_default()


def test_plain_loops_enter_cache_fingerprint():
    """The active plain-loop classes are part of the ABI fingerprint."""
    from cubie._env import active_plain_loops, set_active_plain_loops
    from cubie.cubie_cache import _abi_fingerprint_entries

    saved = active_plain_loops()
    try:
        set_active_plain_loops({"fills", "newton_body"})
        assert "plain-loops=fills,newton_body" in _abi_fingerprint_entries()
    finally:
        set_active_plain_loops(saved)


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
