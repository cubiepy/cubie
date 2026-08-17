"""Tests for cubie.CUDAFactory."""

import numpy as np
import pytest
import attrs
from attrs.exceptions import FrozenInstanceError

from cubie.CUDAFactory import (
    CUDAFactory,
    CUDADispatcherCache,
    CUDAFactoryConfig,
    MultipleInstanceCUDAFactory,
    MultipleInstanceCUDAFactoryConfig,
    _CubieConfigBase,
    _config_field_map,
    _nested_config_fields,
    attribute_is_hashable,
)
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie.cuda_simsafe import numba_from_dtype as from_dtype
from numpy import dtype as np_dtype


# ── Test helper classes (abstract base requires concrete subclass) ── #


@attrs.define
class _TestCache(CUDADispatcherCache):
    """Minimal cache for testing CUDAFactory build/cache mechanics."""

    device_function: object = attrs.field(default=-1, eq=False)


def _make_factory(build_fn=None):
    """Return a concrete CUDAFactory subclass instance.

    Inline construction justified: CUDAFactory is abstract; no fixture
    hierarchy can produce it. These tests verify the base class itself.
    """
    class _ConcreteFactory(CUDAFactory):
        def __init__(self):
            super().__init__()

        def build(self):
            if build_fn is not None:
                return build_fn()
            return _TestCache(device_function=lambda: 20.0)

    return _ConcreteFactory()


def _make_config(**overrides):
    """Return a minimal frozen _CubieConfigBase subclass instance."""
    defaults = {"value1": 10, "value2": "test"}
    defaults.update(overrides)
    keys = list(defaults.keys())
    cls = attrs.make_class(
        "_TestConfig",
        keys,
        bases=(_CubieConfigBase,),
        frozen=True,
        slots=True,
    )
    return cls(**defaults)


def _make_factory_with_settings(precision=np.float32):
    """Return a factory with compile settings already attached."""
    factory = _make_factory()
    cfg = _make_config(precision=precision, flag=False)
    factory.setup_compile_settings(cfg)
    return factory


# ── attribute_is_hashable ──────────────────────────────────── #


def test_attribute_is_hashable_eq_false():
    """Returns False when attribute.eq is False."""
    @attrs.frozen
    class _C:
        x: int = attrs.field(default=1, eq=False)

    fld = attrs.fields(_C).x
    assert attribute_is_hashable(fld, 1) is False


def test_attribute_is_hashable_eq_true():
    """Returns True for normal attributes."""
    @attrs.frozen
    class _C:
        x: int = 1

    fld = attrs.fields(_C).x
    assert attribute_is_hashable(fld, 1) is True


# ── Immutability ───────────────────────────────────────────── #


def test_direct_assignment_raises():
    """Direct field assignment on a config snapshot raises."""
    c = _make_config()
    with pytest.raises(FrozenInstanceError):
        c.value1 = 99


def test_factory_settings_assignment_raises():
    """Assignment through a factory's compile_settings raises."""
    f = _make_factory_with_settings()
    with pytest.raises(FrozenInstanceError):
        f.compile_settings.flag = True


# ── class-level field maps ─────────────────────────────────── #


def test_field_map_contains_name_and_alias():
    """The class field map contains both field names and aliases."""
    @attrs.frozen
    class _C(_CubieConfigBase):
        _val: int = attrs.field(default=1, alias="val")

    field_map = _config_field_map(_C)
    assert "_val" in field_map
    assert "val" in field_map
    assert field_map["_val"] is field_map["val"]


def test_nested_config_fields_identifies_attrs_fields():
    """Nested attrs-class fields are recognised for recursion."""
    @attrs.frozen
    class _Inner(_CubieConfigBase):
        x: int = 1

    @attrs.frozen
    class _Outer(_CubieConfigBase):
        inner: _Inner = attrs.Factory(_Inner)
        plain: int = 2

    nested_names = {fld.name for fld in _nested_config_fields(_Outer)}
    assert "inner" in nested_names
    assert "plain" not in nested_names


def test_post_init_generates_hash():
    """values_hash is a 64-char hex string after construction."""
    c = _make_config()
    assert len(c.values_hash) == 64
    assert all(ch in "0123456789abcdef" for ch in c.values_hash)


def test_post_init_raises_for_dict_field():
    """TypeError raised for dict-type fields not marked eq=False."""
    @attrs.frozen
    class _Bad(_CubieConfigBase):
        d: dict = attrs.Factory(dict)

    with pytest.raises(TypeError, match="dict"):
        _Bad()


# ── _CubieConfigBase.update ───────────────────────────────── #


def test_update_empty_returns_self_and_empty_sets():
    """Returns (self, empty, empty) for empty updates."""
    c = _make_config()
    replacement, recognized, changed = c.update({})
    assert replacement is c
    assert recognized == set()
    assert changed == set()


def test_update_never_mutates_self():
    """The original snapshot is untouched by an update."""
    c = _make_config(value1=10)
    old_hash = c.values_hash
    replacement, recognized, changed = c.update({"value1": 99})
    assert c.value1 == 10
    assert c.values_hash == old_hash
    assert replacement.value1 == 99


def test_update_recognizes_by_name_and_alias():
    """Recognizes fields by name or alias."""
    @attrs.frozen
    class _C(_CubieConfigBase):
        _val: int = attrs.field(default=1, alias="val")

    c = _C()
    replacement, recognized, changed = c.update({"val": 2})
    assert "val" in recognized
    assert replacement._val == 2


def test_update_eq_false_identity_change():
    """A replaced eq=False object counts as a change."""
    fn_a = lambda: 1  # noqa: E731
    fn_b = lambda: 2  # noqa: E731

    @attrs.frozen
    class _C(_CubieConfigBase):
        fn: object = attrs.field(default=fn_a, eq=False)

    c = _C()
    replacement, recognized, changed = c.update({"fn": fn_b})
    assert "fn" in changed
    assert replacement.fn is fn_b
    # Hash is unaffected: eq=False fields are not semantic identity.
    assert replacement.values_hash == c.values_hash


def test_update_eq_false_same_identity_no_change():
    """Passing the identical eq=False object back is not a change."""
    fn = lambda: 1  # noqa: E731

    @attrs.frozen
    class _C(_CubieConfigBase):
        fn: object = attrs.field(default=None, eq=False)

    c = _C(fn=fn)
    replacement, recognized, changed = c.update({"fn": fn})
    assert replacement is c
    assert changed == set()


def test_update_ndarray_comparison():
    """Semantic array fields compare elementwise."""
    @attrs.frozen
    class _C(_CubieConfigBase):
        arr: np.ndarray = attrs.field(
            factory=lambda: np.array([1.0, 2.0]),
            eq=attrs.cmp_using(eq=np.array_equal),
        )

    c = _C()
    replacement, _, changed = c.update({"arr": np.array([1.0, 2.0])})
    assert changed == set()
    assert replacement is c
    replacement, _, changed = c.update({"arr": np.array([3.0, 4.0])})
    assert "arr" in changed


def test_update_no_change_when_same_value():
    """Only reports change when a value actually changed."""
    c = _make_config(value1=10)
    replacement, recognized, changed = c.update({"value1": 10})
    assert "value1" in recognized
    assert "value1" not in changed
    assert replacement is c


def test_update_delegates_to_nested():
    """Delegates to nested attrs objects, evolving parent and child."""
    @attrs.frozen
    class _Inner(_CubieConfigBase):
        x: int = 1

    @attrs.frozen
    class _Outer(_CubieConfigBase):
        inner: _Inner = attrs.Factory(_Inner)

    c = _Outer()
    replacement, recognized, changed = c.update({"x": 42})
    assert "x" in recognized
    assert "x" in changed
    assert replacement.inner.x == 42
    # Original snapshot and its nested child are untouched.
    assert c.inner.x == 1


def test_update_delegates_to_optional_nested():
    """Delegates to a nested config behind an Optional annotation."""
    from typing import Optional

    @attrs.frozen
    class _Inner(_CubieConfigBase):
        x: int = 1

    @attrs.frozen
    class _Outer(_CubieConfigBase):
        inner: Optional[_Inner] = attrs.Factory(_Inner)

    c = _Outer()
    replacement, recognized, changed = c.update({"x": 42})
    assert "x" in recognized
    assert "x" in changed
    assert replacement.inner.x == 42
    assert c.inner.x == 1
    # A None-valued optional nested config is skipped, not an error.
    empty = _Outer(inner=None)
    replacement, recognized, changed = empty.update(
        {"x": 42},
    )
    assert recognized == set()
    assert changed == set()
    assert replacement is empty


def test_update_changes_hash_on_replacement():
    """The replacement's hash differs after a semantic change."""
    c = _make_config(value1=10)
    replacement, _, _ = c.update({"value1": 99})
    assert replacement.values_hash != c.values_hash


def test_update_equal_values_preserve_hash():
    """An update carrying only equal values preserves the hash."""
    c = _make_config(value1=10, value2="test")
    replacement, _, changed = c.update({"value1": 10, "value2": "test"})
    assert changed == set()
    assert replacement.values_hash == c.values_hash


def test_update_returns_recognized_and_changed():
    """Returns correct recognized and changed sets."""
    c = _make_config(value1=10, value2="test")
    _, recognized, changed = c.update({"value1": 99, "value2": "test"})
    assert recognized == {"value1", "value2"}
    assert changed == {"value1"}


def test_update_kwargs_only():
    """Update works when called with only kwargs, no positional dict."""
    c = _make_config(value1=10)
    replacement, recognized, _ = c.update(value1=99)
    assert "value1" in recognized
    assert replacement.value1 == 99


def test_update_reruns_converters_and_validators():
    """Converters and validators run on the replacement snapshot."""
    @attrs.frozen
    class _C(_CubieConfigBase):
        x: int = attrs.field(default=1, converter=int)

    c = _C()
    replacement, _, changed = c.update({"x": "7"})
    assert replacement.x == 7
    assert "x" in changed


# ── _CubieConfigBase properties ───────────────────────────── #


def test_cache_dict_excludes_eq_false():
    """cache_dict returns dict without eq=False fields."""
    @attrs.frozen
    class _C(_CubieConfigBase):
        a: int = 1
        b: object = attrs.field(default=None, eq=False)

    c = _C()
    d = c.cache_dict
    assert "a" in d
    assert "b" not in d


def test_values_hash_memoized_and_stable():
    """values_hash is stable across repeated access."""
    c = _make_config()
    assert c.values_hash == c.values_hash
    # An equal snapshot from an equal construction shares the hash.
    c2 = _make_config()
    assert c.values_hash == c2.values_hash


# ── CUDAFactoryConfig ─────────────────────────────────────── #


@pytest.mark.parametrize("prec_in,prec_out", [
    (np.float32, np.float32),
    (np.float64, np.float64),
])
def test_config_precision_validation_and_conversion(prec_in, prec_out):
    """Construction validates and converts precision."""
    @attrs.frozen
    class _C(CUDAFactoryConfig):
        pass

    c = _C(precision=prec_in)
    assert c.precision == prec_out


def test_config_numba_precision():
    """numba_precision returns from_dtype(np_dtype(precision))."""
    @attrs.frozen
    class _C(CUDAFactoryConfig):
        pass

    c = _C(precision=np.float32)
    expected = from_dtype(np_dtype(np.float32))
    assert c.numba_precision == expected


def test_config_simsafe_precision():
    """simsafe_precision returns simsafe_dtype(np_dtype(precision))."""
    @attrs.frozen
    class _C(CUDAFactoryConfig):
        pass

    c = _C(precision=np.float64)
    expected = simsafe_dtype(np_dtype(np.float64))
    assert c.simsafe_precision == expected


# ── CUDAFactory __init__ / setup / properties ──────────────── #


def test_factory_init_defaults():
    """__init__ sets settings=None, cache_valid=True, cache=None."""
    f = _make_factory()
    assert f._compile_settings is None
    assert f._cache_valid is True
    assert f._cache is None


def test_setup_raises_for_non_attrs():
    """setup_compile_settings raises TypeError for non-attrs."""
    f = _make_factory()
    with pytest.raises(TypeError, match="attrs class"):
        f.setup_compile_settings({"not": "attrs"})


def test_setup_stores_settings_and_invalidates():
    """setup_compile_settings stores settings and invalidates cache."""
    f = _make_factory()
    cfg = _make_config()
    f.setup_compile_settings(cfg)
    assert f._compile_settings is cfg
    assert f._cache_valid is False


def test_cache_valid_property():
    """cache_valid returns _cache_valid."""
    f = _make_factory()
    assert f.cache_valid is True
    f._cache_valid = False
    assert f.cache_valid is False


def test_device_function_calls_get_cached_output():
    """device_function delegates to get_cached_output('device_function')."""
    f = _make_factory_with_settings()
    fn = f.device_function
    # The build returns a lambda returning 20.0
    assert fn() == 20.0


def test_compile_settings_returns_stored():
    """compile_settings returns _compile_settings."""
    f = _make_factory()
    cfg = _make_config()
    f.setup_compile_settings(cfg)
    assert f.compile_settings is cfg


# ── CUDAFactory.update_compile_settings ────────────────────── #


def test_update_settings_empty_returns_empty():
    """Returns empty set for empty updates."""
    f = _make_factory_with_settings()
    result = f.update_compile_settings({})
    assert result == set()


def test_update_settings_raises_when_not_set():
    """Raises ValueError when settings not set up."""
    f = _make_factory()
    with pytest.raises(ValueError, match="set up"):
        f.update_compile_settings({"x": 1})


def test_update_settings_swaps_replacement_snapshot():
    """A changed update swaps in a replacement snapshot."""
    f = _make_factory_with_settings()
    old_settings = f.compile_settings
    f.update_compile_settings(flag=True)
    assert f.compile_settings is not old_settings
    assert f.compile_settings.flag is True
    # The old snapshot is untouched.
    assert old_settings.flag is False


def test_update_settings_unchanged_keeps_snapshot():
    """An update carrying equal values keeps the current snapshot."""
    f = _make_factory_with_settings()
    old_settings = f.compile_settings
    _ = f.device_function
    f.update_compile_settings(flag=False)
    assert f.compile_settings is old_settings
    assert f.cache_valid is True


def test_update_settings_raises_for_unrecognized():
    """Raises KeyError for unrecognized params when silent=False."""
    f = _make_factory_with_settings()
    with pytest.raises(KeyError, match="bogus"):
        f.update_compile_settings(bogus=42)


def test_update_settings_silent_suppresses_error():
    """Suppresses errors when silent=True."""
    f = _make_factory_with_settings()
    # Should not raise
    result = f.update_compile_settings({"bogus": 42}, silent=True)
    assert "bogus" not in result


def test_update_settings_invalidates_cache():
    """Invalidates cache when any field changed."""
    f = _make_factory_with_settings()
    _ = f.device_function  # build cache
    assert f.cache_valid is True
    f.update_compile_settings(flag=True)
    assert f.cache_valid is False


def test_update_settings_eq_false_replacement_invalidates():
    """A replaced eq=False derived callable invalidates the build."""
    factory = _make_factory()

    @attrs.frozen
    class _C(_CubieConfigBase):
        fn: object = attrs.field(default=None, eq=False)

    factory.setup_compile_settings(_C(fn=lambda: 1))
    _ = factory.device_function
    old_hash = factory.compile_settings.values_hash
    factory.update_compile_settings(fn=lambda: 2)
    assert factory.cache_valid is False
    # Semantic hash is unchanged: the callable's identity lives with
    # its owning child factory, not this snapshot.
    assert factory.compile_settings.values_hash == old_hash


def test_update_settings_returns_recognized():
    """Returns recognized set."""
    f = _make_factory_with_settings()
    result = f.update_compile_settings(flag=True)
    assert "flag" in result


# ── CUDAFactory._build / _invalidate_cache ─────────────────── #


def test_invalidate_cache_sets_false():
    """_invalidate_cache sets _cache_valid to False."""
    f = _make_factory()
    f._cache_valid = True
    f._invalidate_cache()
    assert f._cache_valid is False


def test_build_raises_for_non_cache_return():
    """_build raises TypeError if build() doesn't return CUDADispatcherCache."""
    def bad_build():
        return {"not": "a cache"}

    f = _make_factory(build_fn=bad_build)
    cfg = _make_config()
    f.setup_compile_settings(cfg)
    with pytest.raises(TypeError, match="CUDADispatcherCache"):
        f._build()


def test_build_stores_result_and_validates():
    """_build stores result and sets cache_valid=True."""
    f = _make_factory_with_settings()
    f._build()
    assert f._cache is not None
    assert f._cache_valid is True


# ── CUDAFactory.get_cached_output ──────────────────────────── #


def test_get_cached_triggers_build_when_invalid():
    """Triggers _build when cache invalid."""
    f = _make_factory_with_settings()
    assert f._cache_valid is False
    result = f.get_cached_output("device_function")
    assert f._cache_valid is True
    assert result() == 20.0


def test_get_cached_raises_runtime_when_cache_none():
    """Raises RuntimeError when cache is None after build."""
    def null_build():
        # Return valid cache but we'll set it to None after
        return _TestCache()

    f = _make_factory(build_fn=null_build)
    cfg = _make_config()
    f.setup_compile_settings(cfg)
    f._cache_valid = True  # pretend valid
    f._cache = None  # but no cache
    with pytest.raises(RuntimeError, match="not been initialized"):
        f.get_cached_output("device_function")


def test_get_cached_raises_key_for_missing_output():
    """Raises KeyError when output_name not in cache."""
    f = _make_factory_with_settings()
    with pytest.raises(KeyError, match="nonexistent"):
        f.get_cached_output("nonexistent")


def test_get_cached_returns_valid_output():
    """Returns cached value for valid output."""
    f = _make_factory_with_settings()
    fn = f.get_cached_output("device_function")
    assert fn() == 20.0


# ── CUDAFactory.config_hash ───────────────────────────────── #


def test_config_hash_no_children():
    """Returns own hash when no child factories."""
    f = _make_factory_with_settings()
    assert f.config_hash == f.compile_settings.values_hash
    assert len(f.config_hash) == 64


def test_config_hash_with_children():
    """Combines own hash with child hashes when children exist."""
    class _Parent(CUDAFactory):
        def __init__(self):
            super().__init__()
            self._child = _make_factory_with_settings()

        def build(self):
            return _TestCache()

    p = _Parent()
    p.setup_compile_settings(_make_config(x=1))
    # Combined hash differs from own hash
    assert p.config_hash != p.compile_settings.values_hash
    assert len(p.config_hash) == 64


def test_config_hash_stable_across_equal_states():
    """Equal snapshots produce equal config hashes."""
    f1 = _make_factory_with_settings()
    f2 = _make_factory_with_settings()
    assert f1.config_hash == f2.config_hash
    # Updating one to a new value and back restores the hash.
    f1.update_compile_settings(flag=True)
    assert f1.config_hash != f2.config_hash
    f1.update_compile_settings(flag=False)
    assert f1.config_hash == f2.config_hash


# ── CUDAFactory._iter_child_factories ──────────────────────── #


def test_iter_child_yields_factory_instances():
    """Yields CUDAFactory instances from direct attributes."""
    child = _make_factory()

    class _Parent(CUDAFactory):
        def __init__(self):
            super().__init__()
            self._child = child
            self._plain = 42

        def build(self):
            return _TestCache()

    p = _Parent()
    children = list(p._iter_child_factories())
    assert len(children) == 1
    assert children[0] is child


def test_iter_child_alphabetical_order():
    """Alphabetical ordering by attribute name."""
    child_z = _make_factory()
    child_a = _make_factory()

    class _Parent(CUDAFactory):
        def __init__(self):
            super().__init__()
            self._z_child = child_z
            self._a_child = child_a

        def build(self):
            return _TestCache()

    p = _Parent()
    children = list(p._iter_child_factories())
    assert children[0] is child_a
    assert children[1] is child_z


def test_iter_child_deduplicates_by_id():
    """Deduplicates by id."""
    shared = _make_factory()

    class _Parent(CUDAFactory):
        def __init__(self):
            super().__init__()
            self._ref1 = shared
            self._ref2 = shared

        def build(self):
            return _TestCache()

    p = _Parent()
    children = list(p._iter_child_factories())
    assert len(children) == 1


# ── Forwarding properties (table-driven) ───────────────────── #


def test_factory_precision_forwarding():
    """precision forwards to compile_settings.precision."""
    f = _make_factory()
    cfg = CUDAFactoryConfig(precision=np.float32)
    f.setup_compile_settings(cfg)
    assert f.precision == f.compile_settings.precision
    assert f.precision == np.float32


def test_factory_numba_precision_forwarding():
    """numba_precision forwards to compile_settings.numba_precision."""
    f = _make_factory()
    cfg = CUDAFactoryConfig(precision=np.float64)
    f.setup_compile_settings(cfg)
    assert f.numba_precision == f.compile_settings.numba_precision


def test_factory_simsafe_precision_forwarding():
    """simsafe_precision forwards to compile_settings.simsafe_precision."""
    f = _make_factory()
    cfg = CUDAFactoryConfig(precision=np.float32)
    f.setup_compile_settings(cfg)
    assert f.simsafe_precision == f.compile_settings.simsafe_precision


def test_factory_shared_buffer_size(single_integrator_run):
    """shared_buffer_size delegates to buffer_registry."""
    expected = buffer_registry.shared_buffer_size(single_integrator_run)
    assert single_integrator_run.shared_buffer_size == expected


def test_factory_local_buffer_size(single_integrator_run):
    """local_buffer_size delegates to buffer_registry."""
    expected = buffer_registry.local_buffer_size(single_integrator_run)
    assert single_integrator_run.local_buffer_size == expected


def test_factory_persistent_local_buffer_size(single_integrator_run):
    """persistent_local_buffer_size delegates to buffer_registry."""
    expected = buffer_registry.persistent_local_buffer_size(
        single_integrator_run
    )
    assert single_integrator_run.persistent_local_buffer_size == expected


# ── MultipleInstanceCUDAFactoryConfig ──────────────────────── #


def test_get_prefixed_attributes_names():
    """get_prefixed_attributes(aliases=False) returns field names."""
    @attrs.frozen
    class _C(MultipleInstanceCUDAFactoryConfig):
        _atol: float = attrs.field(
            default=1e-6, metadata={"prefixed": True}
        )
        plain: int = 10

    names = _C.get_prefixed_attributes(aliases=False)
    assert "_atol" in names
    assert "plain" not in names


def test_get_prefixed_attributes_aliases():
    """get_prefixed_attributes(aliases=True) returns aliases."""
    @attrs.frozen
    class _C(MultipleInstanceCUDAFactoryConfig):
        _atol: float = attrs.field(
            default=1e-6, alias="atol", metadata={"prefixed": True}
        )

    aliases = _C.get_prefixed_attributes(aliases=True)
    assert "atol" in aliases
    assert "_atol" not in aliases


def test_prefix_property():
    """prefix property returns instance_label."""
    @attrs.frozen
    class _C(MultipleInstanceCUDAFactoryConfig):
        pass

    c = _C(precision=np.float32, instance_label="krylov")
    assert c.prefix == "krylov"
    assert c.prefix == c.instance_label


def test_post_init_sets_prefixed_attributes_when_label_nonempty():
    """__attrs_post_init__ sets prefixed_attributes when label non-empty."""
    @attrs.frozen
    class _C(MultipleInstanceCUDAFactoryConfig):
        _atol: float = attrs.field(
            default=1e-6, metadata={"prefixed": True}
        )
        plain: int = 10

    c = _C(precision=np.float32, instance_label="krylov")
    assert "atol" in c.prefixed_attributes
    assert "plain" not in c.prefixed_attributes

    # Empty label -> empty set
    c_empty = _C(precision=np.float32, instance_label="")
    assert c_empty.prefixed_attributes == frozenset()


# ── MultipleInstanceCUDAFactoryConfig.update ───────────────── #


def test_mi_update_removes_non_prefixed_for_prefixed_attrs():
    """Removes non-prefixed keys for prefixed attributes."""
    @attrs.frozen
    class _C(MultipleInstanceCUDAFactoryConfig):
        _atol: float = attrs.field(
            default=1e-6, alias="atol", metadata={"prefixed": True}
        )

    c = _C(precision=np.float32, instance_label="krylov")
    # Passing unprefixed key for a prefixed attribute: should be removed
    replacement, recognized, changed = c.update({"atol": 1e-8})
    # atol is stripped, krylov_atol not present -> no change
    assert "atol" not in changed
    assert replacement._atol == 1e-6


def test_mi_update_maps_prefixed_to_unprefixed():
    """Maps prefixed keys (e.g. krylov_atol) to unprefixed (atol)."""
    @attrs.frozen
    class _C(MultipleInstanceCUDAFactoryConfig):
        _atol: float = attrs.field(
            default=1e-6, alias="atol", metadata={"prefixed": True}
        )

    c = _C(precision=np.float32, instance_label="krylov")
    replacement, recognized, changed = c.update({"krylov_atol": 1e-10})
    assert "krylov_atol" in recognized
    assert replacement._atol == 1e-10
    # Original untouched
    assert c._atol == 1e-6


def test_mi_update_returns_prefixed_key_names():
    """Returns recognized/changed with prefixed key names restored."""
    @attrs.frozen
    class _C(MultipleInstanceCUDAFactoryConfig):
        _atol: float = attrs.field(
            default=1e-6, alias="atol", metadata={"prefixed": True}
        )

    c = _C(precision=np.float32, instance_label="test")
    _, recognized, changed = c.update({"test_atol": 1e-10})
    assert "test_atol" in recognized
    assert "test_atol" in changed


# ── MultipleInstanceCUDAFactory ────────────────────────────── #


def test_mi_factory_init_stores_label():
    """__init__ stores _instance_label and calls super().__init__."""
    class _F(MultipleInstanceCUDAFactory):
        def build(self):
            return _TestCache()

    f = _F(instance_label="newton")
    assert f._instance_label == "newton"
    assert f._compile_settings is None
    assert f._cache_valid is True


def test_mi_factory_instance_label_property():
    """instance_label property returns _instance_label."""
    class _F(MultipleInstanceCUDAFactory):
        def build(self):
            return _TestCache()

    f = _F(instance_label="krylov")
    assert f.instance_label == "krylov"
