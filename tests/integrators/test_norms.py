"""Tests for cubie.integrators.norms."""

from __future__ import annotations

import numpy as np
import pytest
from cubie.cuda_simsafe import cuda
from cubie.memory import default_memmgr
from numpy.testing import assert_allclose

from cubie.integrators.norms import (
    DIRKCorrectionNorm,
    FIRKCorrectionNorm,
    FIRKCorrectionNormConfig,
    ScaledNorm,
    ScaledNormConfig,
    TiledScaledNorm,
    TiledScaledNormConfig,
)


# ── ScaledNormConfig construction ────────────────────────── #


def test_config_defaults():
    """Default solver_width=1, n=1, atol=[1e-6], rtol=[1e-6]."""
    cfg = ScaledNormConfig(precision=np.float64)
    assert cfg.solver_width == 1
    assert cfg.atol.shape == (1,)
    assert cfg.rtol.shape == (1,)
    assert_allclose(cfg.atol, [1e-6])
    assert_allclose(cfg.rtol, [1e-6])


def test_config_n_validated_minimum():
    """n must be >= 1."""
    with pytest.raises((ValueError, TypeError)):
        ScaledNormConfig(precision=np.float64, solver_width=0, n=0)


def test_config_custom_tolerances():
    """Custom atol/rtol arrays are stored correctly."""
    atol = np.array([1e-4, 1e-5, 1e-6], dtype=np.float64)
    rtol = np.array([1e-3, 1e-4, 1e-5], dtype=np.float64)
    cfg = ScaledNormConfig(
        precision=np.float64, solver_width=3, n=3, atol=atol, rtol=rtol
    )
    assert_allclose(cfg.atol, atol)
    assert_allclose(cfg.rtol, rtol)
    assert cfg.atol.shape == (3,)


def test_config_tolerance_arrays_sealed_after_hashing():
    """Stored tolerances cannot change under a memoized hash."""
    caller_atol = np.array([1e-4, 1e-5, 1e-6], dtype=np.float32)
    cfg = ScaledNormConfig(
        precision=np.float32, solver_width=3, n=3, atol=caller_atol,
        rtol=1e-4,
    )
    hash_before = cfg.values_hash

    assert cfg.atol is not caller_atol
    with pytest.raises(ValueError):
        cfg.atol[0] = 5.0
    with pytest.raises(ValueError):
        cfg.rtol[0] = 5.0

    caller_atol[0] = 5.0
    assert cfg.atol[0] == np.float32(1e-4)
    assert cfg.values_hash == hash_before


def test_config_scalar_tolerance_broadcast():
    """Scalar tolerance is broadcast to array of length n."""
    cfg = ScaledNormConfig(
        precision=np.float64, solver_width=4, n=4, atol=1e-5, rtol=1e-4
    )
    assert cfg.atol.shape == (4,)
    assert cfg.rtol.shape == (4,)
    assert_allclose(cfg.atol, np.full(4, 1e-5))
    assert_allclose(cfg.rtol, np.full(4, 1e-4))


def test_config_negative_atol_rejected():
    """atol rejects arrays containing negative values."""
    with pytest.raises(ValueError):
        ScaledNormConfig(
            precision=np.float64, solver_width=2, n=2, atol=-1e-6
        )


def test_config_negative_rtol_rejected():
    """rtol rejects arrays containing negative elements."""
    rtol = np.array([1e-4, -1e-4], dtype=np.float64)
    with pytest.raises(ValueError):
        ScaledNormConfig(
            precision=np.float64, solver_width=2, n=2, rtol=rtol
        )


def test_config_zero_tolerances_accepted():
    """Zero tolerances are valid; tol_floor guards the division."""
    cfg = ScaledNormConfig(
        precision=np.float64, solver_width=2, n=2, atol=0.0, rtol=0.0
    )
    assert_allclose(cfg.atol, np.zeros(2))
    assert_allclose(cfg.rtol, np.zeros(2))


def test_config_inv_n():
    """inv_n returns precision(1.0/n)."""
    cfg = ScaledNormConfig(precision=np.float32, solver_width=5, n=5)
    expected = np.float32(1.0 / 5)
    assert cfg.inv_n == pytest.approx(float(expected), rel=1e-6)


def test_config_tol_floor():
    """tol_floor returns precision(1e-16)."""
    cfg = ScaledNormConfig(precision=np.float64, solver_width=2, n=2)
    assert cfg.tol_floor == pytest.approx(1e-16)


def test_config_atol_prefixed_metadata():
    """atol field has prefixed=True metadata."""
    for f in ScaledNormConfig.__attrs_attrs__:
        if f.name == "atol":
            assert f.metadata.get("prefixed") is True
            break


def test_config_rtol_prefixed_metadata():
    """rtol field has prefixed=True metadata."""
    for f in ScaledNormConfig.__attrs_attrs__:
        if f.name == "rtol":
            assert f.metadata.get("prefixed") is True
            break


# ── tolerance sizing through the update path ─────────────── #


def test_resize_uniform_tolerances_on_n_change():
    """Uniform tolerance arrays expand when n changes."""
    cfg = ScaledNormConfig(
        precision=np.float64, solver_width=2, n=2, atol=1e-5, rtol=1e-4
    )
    assert cfg.atol.shape == (2,)
    replacement, _, changed = cfg.update({"solver_width": 5, "n": 5})
    assert "solver_width" in changed
    assert replacement.atol.shape == (5,)
    assert replacement.rtol.shape == (5,)
    assert_allclose(replacement.atol, np.full(5, 1e-5))
    assert_allclose(replacement.rtol, np.full(5, 1e-4))
    # Original snapshot untouched
    assert cfg.atol.shape == (2,)


def test_resize_skips_matching_length():
    """Tolerances already matching n are not modified."""
    atol = np.array([1e-4, 1e-5, 1e-6], dtype=np.float64)
    cfg = ScaledNormConfig(
        precision=np.float64, solver_width=3, n=3, atol=atol, rtol=1e-3
    )
    replacement, _, changed = cfg.update({"solver_width": 3})  # same size
    assert changed == set()
    assert_allclose(replacement.atol, atol)


def test_resize_nonuniform_wrong_length_raises():
    """A non-uniform tolerance of the wrong length fails the update.

    Update ``n`` and the tolerance arrays together in one call to
    change both.
    """
    atol = np.array([1e-4, 1e-5], dtype=np.float64)
    cfg = ScaledNormConfig(
        precision=np.float64, solver_width=2, n=2, atol=atol, rtol=1e-3
    )
    with pytest.raises(ValueError, match="shape"):
        cfg.update({"solver_width": 5, "n": 5})
    # A combined update supplies consistent values in one snapshot.
    new_atol = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8], dtype=np.float64)
    replacement, _, changed = cfg.update(
        {"solver_width": 5, "n": 5, "atol": new_atol}
    )
    assert replacement.atol.shape == (5,)
    assert_allclose(replacement.atol, new_atol)


# ── stage-tiled tolerance sizing ─────────────────────────── #


def test_whole_vector_config_rejects_smaller_n():
    """A whole-vector norm cannot hold fewer tolerances than values."""
    with pytest.raises(ValueError, match="whole-vector"):
        ScaledNormConfig(precision=np.float64, solver_width=6, n=3)


@pytest.mark.parametrize(
    "config_class, extra",
    [
        (TiledScaledNormConfig, {}),
        (
            FIRKCorrectionNormConfig,
            {"stage_coefficients": tuple(np.arange(4, dtype=float))},
        ),
    ],
)
def test_tiled_config_tolerances_are_per_state(config_class, extra):
    """Stage-tiled configs size tolerances by n, not solver_width."""
    atol = np.array([1e-7, 1e-6], dtype=np.float64)
    cfg = config_class(
        precision=np.float64,
        solver_width=4,
        n=2,
        atol=atol,
        rtol=1e-4,
        **extra,
    )
    assert cfg.tol_length == 2
    assert cfg.atol.shape == (2,)
    assert cfg.rtol.shape == (2,)
    assert_allclose(cfg.atol, atol)


def test_tiled_config_rejects_solver_width_length_tolerance():
    """A stage-stacked tolerance vector is the wrong length."""
    atol = np.array([1e-7, 1e-6, 1e-5, 1e-4], dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        TiledScaledNormConfig(
            precision=np.float64, solver_width=4, n=2, atol=atol
        )


def test_tiled_norm_tiles_tolerances_across_stages():
    """Each stage block scales by its physical state's tolerance."""
    n = 2
    stages = 3
    width = n * stages
    atol = np.array([1e-3, 1e-5], dtype=np.float64)
    rtol = np.array([1e-2, 1e-4], dtype=np.float64)
    factory = TiledScaledNorm(
        precision=np.float64, solver_width=width, n=n, atol=atol, rtol=rtol
    )
    fn = factory.device_function

    @cuda.jit
    def kernel(values, reference, result):
        result[0] = fn(values, reference)

    values_host = np.arange(1.0, width + 1.0, dtype=np.float64)
    reference_host = np.array([0.5, -2.0], dtype=np.float64)
    values = cuda.to_device(values_host)
    reference = cuda.to_device(reference_host)
    result = cuda.to_device(np.zeros(1, dtype=np.float64))

    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](values, reference, result)
    stream.synchronize()

    expected = 0.0
    for index in range(width):
        state = index % n
        tol = atol[state] + rtol[state] * abs(reference_host[state])
        expected += (values_host[index] / tol) ** 2
    expected /= width
    assert_allclose(result.copy_to_host()[0], expected, rtol=1e-10)


def test_firk_correction_norm_tiles_tolerances_across_stages():
    """The coupled Newton norm scales stage blocks per state."""
    n = 2
    stages = 2
    width = n * stages
    a = np.array([[0.5, 0.0], [0.5, 0.5]], dtype=np.float64)
    atol = np.array([1.0, 0.25], dtype=np.float64)
    rtol = np.array([0.1, 0.4], dtype=np.float64)
    factory = FIRKCorrectionNorm(
        precision=np.float64,
        solver_width=width,
        n=n,
        stage_coefficients=tuple(a.ravel().tolist()),
        atol=atol,
        rtol=rtol,
    )
    fn = factory.device_function

    @cuda.jit
    def kernel(delta, increment, stage_base, step_start, a_ij, result):
        result[0] = fn(delta, increment, stage_base, step_start, a_ij)

    delta_host = np.array([0.21, 0.3, 0.46, 0.15], dtype=np.float64)
    increment_host = np.array([2.0, -1.0, 4.0, 3.0], dtype=np.float64)
    base_host = np.array([10.0, -4.0], dtype=np.float64)
    start_host = np.array([8.0, -5.0], dtype=np.float64)
    result = cuda.to_device(np.zeros(1, dtype=np.float64))

    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](
        cuda.to_device(delta_host),
        cuda.to_device(increment_host),
        cuda.to_device(base_host),
        cuda.to_device(start_host),
        np.float64(0.0),
        result,
    )
    stream.synchronize()

    expected = 0.0
    for index in range(width):
        stage = index // n
        state = index - stage * n
        stage_value = base_host[state]
        for contribution in range(stages):
            stage_value += (
                a[stage, contribution]
                * increment_host[contribution * n + state]
            )
        reference = max(abs(stage_value), abs(start_host[state]))
        tol = atol[state] + rtol[state] * reference
        expected += (delta_host[index] / tol) ** 2
    expected /= width
    assert_allclose(result.copy_to_host()[0], expected, rtol=1e-10)


# ── ScaledNormCache ──────────────────────────────────────── #


def test_cache_from_build():
    """Build returns ScaledNormCache with scaled_norm field."""
    factory = ScaledNorm(precision=np.float64, solver_width=3, n=3)
    _ = factory.device_function
    cache = factory._cache
    # Cache holds the same function as device_function property
    assert cache.scaled_norm is factory.device_function


# ── ScaledNorm __init__ ──────────────────────────────────── #


def test_init_sets_compile_settings():
    """__init__ creates config and sets up compile_settings."""
    factory = ScaledNorm(
        precision=np.float64,
        solver_width=4,
        n=4,
        atol=1e-5,
        rtol=1e-4,
    )
    cs = factory.compile_settings
    assert cs.solver_width == 4
    assert cs.precision == np.float64
    assert_allclose(cs.atol, np.full(4, 1e-5))
    assert_allclose(cs.rtol, np.full(4, 1e-4))


def test_init_with_instance_label():
    """Prefixed kwargs are stripped and applied with instance_label."""
    atol = np.array([1e-10, 1e-9, 1e-8], dtype=np.float64)
    rtol = np.array([1e-5, 1e-4, 1e-3], dtype=np.float64)
    factory = ScaledNorm(
        precision=np.float64,
        solver_width=3, n=3,
        instance_label="krylov",
        krylov_atol=atol,
        krylov_rtol=rtol,
    )
    assert_allclose(factory.atol, atol)
    assert_allclose(factory.rtol, rtol)


def test_init_empty_instance_label():
    """Empty instance_label uses unprefixed kwargs."""
    atol = np.array([1e-8, 1e-7], dtype=np.float64)
    factory = ScaledNorm(
        precision=np.float64, solver_width=2, n=2, instance_label="", atol=atol
    )
    assert_allclose(factory.atol, atol)


# ── ScaledNorm.build (device function) ───────────────────── #


def test_build_converged_norm():
    """Norm <= 1.0 when errors are within tolerance."""
    factory = ScaledNorm(
        precision=np.float64,
        solver_width=3,
        n=3,
        atol=1e-3,
        rtol=1e-3,
    )
    fn = factory.device_function

    @cuda.jit
    def kernel(values, reference, result):
        result[0] = fn(values, reference)

    vals = cuda.to_device(np.array([1e-5, 1e-5, 1e-5], dtype=np.float64))
    refs = cuda.to_device(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    res = cuda.to_device(np.array([0.0], dtype=np.float64))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](vals, refs, res)
    stream.synchronize()
    result = res.copy_to_host()[0]

    # tol_i = 1e-3 + 1e-3*1.0 = 2e-3; ratio = 1e-5/2e-3 = 5e-3
    # nrm2 = (5e-3)^2 * (1/3) * 3 = 2.5e-5
    expected = (1e-5 / 2e-3) ** 2
    assert_allclose(result, expected, rtol=1e-10)


def test_build_exceeds_tolerance():
    """Norm > 1.0 when errors exceed tolerance."""
    factory = ScaledNorm(
        precision=np.float64,
        solver_width=3,
        n=3,
        atol=1e-6,
        rtol=1e-6,
    )
    fn = factory.device_function

    @cuda.jit
    def kernel(values, reference, result):
        result[0] = fn(values, reference)

    vals = cuda.to_device(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    refs = cuda.to_device(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    res = cuda.to_device(np.array([0.0], dtype=np.float64))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](vals, refs, res)
    stream.synchronize()
    result = res.copy_to_host()[0]

    # tol_i = 2e-6; ratio = 1.0/2e-6 = 5e5; nrm2 = (5e5)^2 = 2.5e11
    expected = (1.0 / 2e-6) ** 2
    assert_allclose(result, expected, rtol=1e-6)
    assert result > 1.0


def test_build_tol_floor_prevents_division_by_zero():
    """When atol and rtol*ref are near zero, floor of 1e-16 applies."""
    factory = ScaledNorm(
        precision=np.float64, solver_width=1, n=1, atol=0.0, rtol=0.0
    )
    fn = factory.device_function

    @cuda.jit
    def kernel(values, reference, result):
        result[0] = fn(values, reference)

    vals = cuda.to_device(np.array([1e-10], dtype=np.float64))
    refs = cuda.to_device(np.array([0.0], dtype=np.float64))
    res = cuda.to_device(np.array([0.0], dtype=np.float64))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](vals, refs, res)
    stream.synchronize()
    result = res.copy_to_host()[0]

    # tol_i = max(0 + 0*0, 1e-16) = 1e-16
    # ratio = 1e-10 / 1e-16 = 1e6
    # nrm2 = (1e6)^2 = 1e12
    expected = (1e-10 / 1e-16) ** 2
    assert_allclose(result, expected, rtol=1e-6)


def test_build_mean_squared_norm():
    """Norm is mean of squared ratios (divided by n)."""
    atol = np.array([1e-3, 1e-4], dtype=np.float64)
    rtol = np.array([0.0, 0.0], dtype=np.float64)
    factory = ScaledNorm(
        precision=np.float64,
        solver_width=2,
        n=2,
        atol=atol,
        rtol=rtol,
    )
    fn = factory.device_function

    @cuda.jit
    def kernel(values, reference, result):
        result[0] = fn(values, reference)

    vals = cuda.to_device(np.array([1e-3, 1e-4], dtype=np.float64))
    refs = cuda.to_device(np.array([0.0, 0.0], dtype=np.float64))
    res = cuda.to_device(np.array([0.0], dtype=np.float64))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](vals, refs, res)
    stream.synchronize()
    result = res.copy_to_host()[0]

    # tol0 = 1e-3, ratio0 = 1e-3/1e-3 = 1.0
    # tol1 = 1e-4, ratio1 = 1e-4/1e-4 = 1.0
    # nrm2 = (1^2 + 1^2) / 2 = 1.0
    assert_allclose(result, 1.0, rtol=1e-10)


# The scaling reference is the physical stage state, not the iterate;
# commit 7d381c2 fixed the increment-referenced scaling these cases pin.
_CORRECTION_NORM_CASES = {
    "dirk": dict(
        factory=DIRKCorrectionNorm,
        factory_kwargs=dict(solver_width=2, n=2),
        a_ij=0.5,
        delta=(0.21, 0.3),
        increment=(2.0, -1.0),
        stage_base=(10.0, -4.0),
        step_start=(8.0, -5.0),
        expected=0.025,
    ),
    "firk": dict(
        factory=FIRKCorrectionNorm,
        factory_kwargs=dict(
            solver_width=4,
            n=2,
            stage_coefficients=(0.5, 0.0, 0.5, 0.5),
        ),
        a_ij=0.0,
        delta=(0.21, 0.3, 0.46, 0.15),
        increment=(2.0, -1.0, 4.0, 3.0),
        stage_base=(10.0, -4.0),
        step_start=(8.0, -5.0),
        expected=0.025,
    ),
}


@pytest.fixture(scope="session")
def correction_norm_case(request):
    """Return one named correction-norm scaling case."""
    return _CORRECTION_NORM_CASES[request.param]


@pytest.fixture(scope="session")
def correction_norm_kernel(correction_norm_case, precision):
    """Compile the correction-norm kernel once per parameter set."""
    factory = correction_norm_case["factory"](
        precision=precision,
        atol=1.0,
        rtol=0.1,
        **correction_norm_case["factory_kwargs"],
    )
    correction_norm = factory.device_function

    @cuda.jit
    def kernel(delta, increment, stage_base, step_start, a_ij, result):
        result[0] = correction_norm(
            delta,
            increment,
            stage_base,
            step_start,
            a_ij,
        )

    return kernel


@pytest.mark.parametrize(
    "correction_norm_case", ["dirk", "firk"], indirect=True
)
def test_correction_norm_scales_by_physical_stage_state(
    correction_norm_case, correction_norm_kernel, precision
):
    """Correction norms scale by the stage state the update produces."""
    case = correction_norm_case
    delta = cuda.to_device(np.array(case["delta"], dtype=precision))
    increment = cuda.to_device(
        np.array(case["increment"], dtype=precision)
    )
    stage_base = cuda.to_device(
        np.array(case["stage_base"], dtype=precision)
    )
    step_start = cuda.to_device(
        np.array(case["step_start"], dtype=precision)
    )
    result = cuda.to_device(np.zeros(1, dtype=precision))

    stream = default_memmgr.get_group_stream()
    correction_norm_kernel[1, 1, stream](
        delta,
        increment,
        stage_base,
        step_start,
        precision(case["a_ij"]),
        result,
    )
    stream.synchronize()

    assert_allclose(result.copy_to_host()[0], case["expected"], rtol=1e-5)


# ── ScaledNorm.update ────────────────────────────────────── #


def test_update_invalidates_cache():
    """update() invalidates the cache when settings change."""
    factory = ScaledNorm(
        precision=np.float64,
        solver_width=3,
        n=3,
        atol=1e-6,
        rtol=1e-6,
    )
    _ = factory.device_function
    assert factory.cache_valid
    new_atol = np.array([1e-3, 1e-3, 1e-3], dtype=np.float64)
    factory.update(atol=new_atol)
    assert not factory.cache_valid
    assert_allclose(factory.atol, new_atol)


def test_update_empty_returns_empty_set():
    """Empty update returns empty set without cache invalidation."""
    factory = ScaledNorm(precision=np.float64, solver_width=2, n=2)
    _ = factory.device_function
    result = factory.update()
    assert result == set()
    assert factory.cache_valid


def test_update_merges_dict_and_kwargs():
    """update() merges updates_dict and kwargs."""
    factory = ScaledNorm(
        precision=np.float64,
        solver_width=2,
        n=2,
        atol=1e-6,
        rtol=1e-6,
    )
    new_atol = np.full(2, 1e-3, dtype=np.float64)
    new_rtol = np.full(2, 1e-4, dtype=np.float64)
    recognized = factory.update({"atol": new_atol}, rtol=new_rtol)
    assert "atol" in recognized
    assert "rtol" in recognized
    assert_allclose(factory.atol, new_atol)
    assert_allclose(factory.rtol, new_rtol)


# ── Forwarding properties ────────────────────────────────── #


@pytest.mark.parametrize(
    "prop, child_attr",
    [
        ("precision", "precision"),
        ("solver_width", "solver_width"),
    ],
)
def test_forwarding_scalar_properties(prop, child_attr):
    """Scalar forwarding properties delegate to compile_settings."""
    factory = ScaledNorm(
        precision=np.float64,
        solver_width=4,
        n=4,
        atol=1e-5,
        rtol=1e-4,
    )
    assert getattr(factory, prop) == getattr(
        factory.compile_settings, child_attr
    )


@pytest.mark.parametrize(
    "prop, child_attr",
    [
        ("atol", "atol"),
        ("rtol", "rtol"),
    ],
)
def test_forwarding_array_properties(prop, child_attr):
    """Array forwarding properties delegate to compile_settings."""
    factory = ScaledNorm(
        precision=np.float64,
        solver_width=3,
        n=3,
        atol=1e-5,
        rtol=1e-4,
    )
    result = getattr(factory, prop)
    expected = getattr(factory.compile_settings, child_attr)
    assert_allclose(result, expected)


def test_device_function_forwards_cache():
    """device_function returns get_cached_output('scaled_norm')."""
    factory = ScaledNorm(precision=np.float64, solver_width=2, n=2)
    fn = factory.device_function
    assert fn is factory.get_cached_output("scaled_norm")


# ── correction-norm rtol precision floor ─────────────────── #


def test_correction_rtol_below_noise_floor_raised_silently():
    """Nonzero correction rtol below 4 ULPs floors per component."""
    import warnings as warnings_module

    from cubie.integrators.norms import CorrectionNormConfig

    floor32 = 4.0 * np.finfo(np.float32).eps
    rtol = np.array([1e-15, 0.0, 1e-3], dtype=np.float32)
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        cfg = CorrectionNormConfig(
            precision=np.float32, solver_width=3, n=3, rtol=rtol
        )
    assert_allclose(
        cfg.rtol, np.array([floor32, 0.0, 1e-3], dtype=np.float32)
    )


def test_correction_rtol_at_or_above_floor_unchanged():
    """rtol at or above the floor passes through without warning."""
    import warnings as warnings_module

    from cubie.integrators.norms import CorrectionNormConfig

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        cfg = CorrectionNormConfig(
            precision=np.float64, solver_width=2, n=2, rtol=1e-9
        )
    assert_allclose(cfg.rtol, [1e-9, 1e-9])


def test_residual_norm_rtol_not_floored():
    """Plain scaled norms keep sub-ULP rtol requests unchanged."""
    import warnings as warnings_module

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        cfg = ScaledNormConfig(
            precision=np.float32, solver_width=2, n=2, rtol=1e-15
        )
    assert_allclose(cfg.rtol, np.array([1e-15, 1e-15], np.float32))
