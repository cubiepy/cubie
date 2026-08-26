"""Tests for cubie.buffer_registry."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from cubie.buffer_registry import (
    BufferGroup,
    BufferRegistry,
    CUDABuffer,
    buffer_registry,
)
from tests._utils import MID_RUN_PARAMS, merge_dicts


# ── Fixtures ─────────────────────────────────────────────── #
# BufferRegistry is a library singleton. Per Rule 9, a session-
# scoped fixture provides a fresh instance shared across tests.
# Each test that needs isolation registers under a unique parent
# (BufferRegistry treats each parent as an independent group).


@pytest.fixture(scope="session")
def fresh_registry():
    """Fresh BufferRegistry instance (not the module singleton)."""
    return BufferRegistry()


# ── CUDABuffer construction and type properties ──────────── #


@pytest.mark.parametrize(
    "location, persistent, expected_shared, expected_local, "
    "expected_persistent",
    [
        pytest.param(
            "shared", False, True, False, False, id="shared",
        ),
        pytest.param(
            "local", False, False, True, False, id="local",
        ),
        pytest.param(
            "local", True, False, False, True, id="persistent",
        ),
    ],
)
def test_cuda_buffer_type_properties(
    location, persistent, expected_shared, expected_local,
    expected_persistent,
):
    """CUDABuffer type properties reflect location and persistent.

    Inline construction justified: testing CUDABuffer __init__ and
    derived boolean properties directly.
    """
    buf = CUDABuffer(
        name="buf", size=10, location=location,
        persistent=persistent,
    )
    assert buf.is_shared is expected_shared
    assert buf.is_local is expected_local
    assert buf.is_persistent_local is expected_persistent


def test_cuda_buffer_construction_stores_fields():
    """CUDABuffer stores all constructor arguments.

    Inline construction justified: testing __init__ field storage.
    """
    buf = CUDABuffer(
        name="test", size=42, location="shared",
        persistent=False, aliases="parent", dtype=np.float64,
    )
    assert buf.name == "test"
    assert buf.size == 42
    assert buf.location == "shared"
    assert buf.persistent is False
    assert buf.aliases == "parent"
    assert buf.dtype == np.float64


def test_cuda_buffer_defaults():
    """CUDABuffer defaults: persistent=False, aliases=None, dtype=float32.

    Inline construction justified: testing __init__ defaults.
    """
    buf = CUDABuffer(name="d", size=5, location="local")
    assert buf.persistent is False
    assert buf.aliases is None
    assert buf.dtype == np.float32


def test_cuda_buffer_invalid_location_raises():
    """CUDABuffer raises ValueError for invalid location.

    Inline construction justified: testing __init__ validation.
    """
    with pytest.raises(ValueError):
        CUDABuffer(name="x", size=1, location="invalid")


def test_cuda_buffer_invalid_dtype_raises():
    """CUDABuffer raises ValueError for an unsupported dtype.

    Inline construction justified: testing __init__ validation.
    """
    with pytest.raises(ValueError, match="float16, float32, float64"):
        CUDABuffer(
            name="x", size=1, location="shared",
            dtype=np.complex64,
        )


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float64, id="float64"),
        pytest.param(np.float16, id="float16"),
        pytest.param(np.int32, id="int32"),
        pytest.param(np.int64, id="int64"),
    ],
)
def test_cuda_buffer_valid_dtypes(dtype):
    """CUDABuffer accepts all supported buffer dtypes.

    Inline construction justified: testing __init__ dtype validation.
    """
    buf = CUDABuffer(
        name="x", size=1, location="shared", dtype=dtype,
    )
    assert buf.dtype == dtype


# ── Buffer dtype vs parent dtype ───────────────────────── #


def test_cuda_buffer_parent_elements_and_view():
    """Sizes and view flags follow the dtype/parent_dtype pair.

    Inline construction justified: testing derived dtype properties.
    """
    narrow = CUDABuffer(
        name="n", size=3, location="shared",
        dtype=np.int32, parent_dtype=np.float64,
    )
    assert narrow.needs_view is True
    assert narrow.parent_elements == 2
    assert narrow.aligned_offset(5) == 5

    wide = CUDABuffer(
        name="w", size=3, location="shared",
        dtype=np.float64, parent_dtype=np.float32,
    )
    assert wide.needs_view is True
    assert wide.parent_elements == 6
    assert wide.aligned_offset(5) == 6

    same = CUDABuffer(
        name="s", size=3, location="shared",
        dtype=np.float32, parent_dtype=np.float32,
    )
    assert same.needs_view is False
    assert same.parent_elements == 3


def test_group_register_defaults_dtype_to_parent():
    """A buffer registered without a dtype takes the group's."""
    group = BufferGroup(parent_dtype=np.float64)
    group.register("buf", 4, "shared")
    group.register("ints", 4, "shared", dtype=np.int32)
    assert group.entries["buf"].dtype == np.float64
    assert group.entries["buf"].needs_view is False
    assert group.entries["ints"].dtype == np.int32
    assert group.entries["ints"].parent_dtype == np.float64
    assert group.shared_layout["buf"] == slice(0, 4)
    assert group.shared_layout["ints"] == slice(4, 6)


def test_group_set_parent_dtype_restamps_and_invalidates():
    """A parent dtype change updates entries and rebuilds layouts."""
    group = BufferGroup()
    group.register("ints", 4, "shared", dtype=np.int32)
    assert group.shared_layout["ints"] == slice(0, 4)

    group.set_parent_dtype(np.float64)
    assert group._shared_layout is None
    assert group.entries["ints"].parent_dtype == np.float64
    assert group.shared_layout["ints"] == slice(0, 2)


def test_registry_register_stamps_parent_precision(fresh_registry):
    """The group's parent dtype comes from the parent's precision."""

    class _DoubleOwner:
        precision = np.float64

    owner = _DoubleOwner()
    fresh_registry.register("buf", owner, 4, "shared")
    fresh_registry.register(
        "ints", owner, 4, "shared", dtype=np.int32,
    )
    group = fresh_registry._groups[owner]
    assert group.parent_dtype == np.float64
    assert group.entries["buf"].dtype == np.float64
    assert fresh_registry.shared_buffer_size(owner) == 6
    fresh_registry.clear_parent(owner)


# ── CUDABuffer.build_allocator ────────────────────────────── #


def test_build_allocator_shared_slice():
    """build_allocator with shared_slice returns callable allocator.

    Inline construction justified: testing CUDABuffer.build_allocator
    directly; no fixture provides bare CUDABuffer instances.
    """
    buf = CUDABuffer(name="s", size=10, location="shared")
    alloc = buf.build_allocator(
        shared_slice=slice(0, 10),
        persistent_slice=None,
        local_size=None,
        zero=False,
    )
    assert callable(alloc)
    assert alloc.__name__ == "allocate_buffer"


def test_build_allocator_persistent_slice():
    """build_allocator with persistent_slice (no shared) returns callable.

    Inline construction justified: testing CUDABuffer.build_allocator.
    """
    buf = CUDABuffer(
        name="p", size=5, location="local", persistent=True,
    )
    alloc = buf.build_allocator(
        shared_slice=None,
        persistent_slice=slice(0, 5),
        local_size=None,
        zero=False,
    )
    assert callable(alloc)
    assert alloc.__name__ == "allocate_buffer"


def test_build_allocator_local_size():
    """build_allocator with local_size (no shared/persistent) returns callable.

    Inline construction justified: testing CUDABuffer.build_allocator.
    """
    buf = CUDABuffer(name="l", size=3, location="local")
    alloc = buf.build_allocator(
        shared_slice=None,
        persistent_slice=None,
        local_size=3,
        zero=False,
    )
    assert callable(alloc)
    assert alloc.__name__ == "allocate_buffer"


def test_build_allocator_zero_flag():
    """build_allocator with zero=True produces a different closure.

    Inline construction justified: testing CUDABuffer.build_allocator
    zero-flag branch.
    """
    buf = CUDABuffer(name="z", size=4, location="shared")
    alloc_no_zero = buf.build_allocator(
        shared_slice=slice(0, 4),
        persistent_slice=None,
        local_size=None,
        zero=False,
    )
    alloc_zero = buf.build_allocator(
        shared_slice=slice(0, 4),
        persistent_slice=None,
        local_size=None,
        zero=True,
    )
    assert callable(alloc_no_zero)
    assert callable(alloc_zero)
    assert alloc_no_zero is not alloc_zero


# ── BufferGroup.register validation ──────────────────────── #


def test_register_empty_name_raises():
    """BufferGroup.register raises ValueError for empty name."""
    group = BufferGroup()
    with pytest.raises(ValueError, match="cannot be empty"):
        group.register("", 10, "shared")


def test_register_self_alias_raises():
    """BufferGroup.register raises ValueError for self-aliasing."""
    group = BufferGroup()
    with pytest.raises(ValueError, match="cannot alias itself"):
        group.register("buf", 10, "shared", aliases="buf")


def test_register_missing_alias_target_raises():
    """BufferGroup.register raises ValueError for missing alias target."""
    group = BufferGroup()
    with pytest.raises(ValueError, match="not registered"):
        group.register("child", 10, "shared", aliases="parent")


def test_register_adds_entry_and_invalidates():
    """Registration adds entry to group and invalidates layouts."""
    group = BufferGroup()
    group.register("buf", 10, "shared")
    group.build_layouts()
    assert group._shared_layout is not None

    group.register("buf2", 5, "shared")
    assert "buf2" in group.entries
    assert group.entries["buf2"].size == 5
    assert group.entries["buf2"].location == "shared"
    assert group._shared_layout is None


# ── BufferGroup.update_buffer ─────────────────────────────── #


def test_update_buffer_unregistered_returns_false_false():
    """update_buffer returns (False, False) for unknown buffer."""
    group = BufferGroup()
    recognized, changed = group.update_buffer("missing", size=10)
    assert recognized is False
    assert changed is False


def test_update_buffer_no_change_returns_true_false():
    """update_buffer returns (True, False) when values unchanged."""
    group = BufferGroup()
    group.register("buf", 10, "shared")
    recognized, changed = group.update_buffer("buf", size=10)
    assert recognized is True
    assert changed is False


def test_update_buffer_changed_returns_true_true():
    """update_buffer returns (True, True) and invalidates on change."""
    group = BufferGroup()
    group.register("buf", 10, "shared")
    group.build_layouts()
    assert group._shared_layout is not None

    recognized, changed = group.update_buffer("buf", size=20)
    assert recognized is True
    assert changed is True
    assert group.entries["buf"].size == 20
    assert group._shared_layout is None


# ── BufferGroup.invalidate_layouts ────────────────────────── #


def test_invalidate_layouts_clears_all():
    """invalidate_layouts sets all caches to None."""
    group = BufferGroup()
    group.register("s", 10, "shared")
    group.register("p", 5, "local", persistent=True)
    group.register("l", 3, "local")
    group.build_layouts()

    group.invalidate_layouts()
    assert group._shared_layout is None
    assert group._persistent_layout is None
    assert group._local_sizes is None
    assert group._alias_consumption == {}


# ── BufferGroup.build_layouts ─────────────────────────────── #


def test_build_layouts_shared_sequential_offsets():
    """build_layouts assigns sequential shared slices."""
    group = BufferGroup()
    group.register("a", 10, "shared")
    group.register("b", 20, "shared")
    group.build_layouts()

    assert group.shared_layout["a"] == slice(0, 10)
    assert group.shared_layout["b"] == slice(10, 30)


def test_build_layouts_persistent_sequential_offsets():
    """build_layouts assigns sequential persistent slices."""
    group = BufferGroup()
    group.register("a", 15, "local", persistent=True)
    group.register("b", 25, "local", persistent=True)
    group.build_layouts()

    assert group.persistent_layout["a"] == slice(0, 15)
    assert group.persistent_layout["b"] == slice(15, 40)


def test_build_layouts_local_sizes_min_one():
    """build_layouts uses max(size, 1) for local buffers."""
    group = BufferGroup()
    group.register("zero", 0, "local")
    group.register("nonzero", 7, "local")
    group.build_layouts()

    assert group.local_sizes["zero"] == 1
    assert group.local_sizes["nonzero"] == 7


def test_build_layouts_short_circuits_when_populated():
    """build_layouts returns early when all caches already built."""
    group = BufferGroup()
    group.register("s", 10, "shared")
    group.build_layouts()
    original = group._shared_layout

    group.build_layouts()
    assert group._shared_layout is original


# ── BufferGroup.layout_aliases ────────────────────────────── #


def test_alias_overlaps_shared_parent():
    """Aliased buffer overlaps within shared parent when space."""
    group = BufferGroup()
    group.register("parent", 100, "shared")
    group.register("child", 30, "shared", aliases="parent")
    group.build_layouts()

    assert group.shared_layout["parent"] == slice(0, 100)
    assert group.shared_layout["child"] == slice(0, 30)


def test_alias_exceeds_parent_falls_back():
    """Aliased buffer exceeding parent gets own shared allocation."""
    group = BufferGroup()
    group.register("parent", 50, "shared")
    group.register("child", 80, "shared", aliases="parent")
    group.build_layouts()

    assert group.shared_layout["parent"] == slice(0, 50)
    assert group.shared_layout["child"] == slice(50, 130)
    assert group.shared_buffer_size() == 130


def test_alias_fallback_persistent():
    """Persistent aliased buffer falls back to persistent layout."""
    group = BufferGroup()
    group.register("parent", 10, "local")
    group.register(
        "child", 5, "local", persistent=True, aliases="parent",
    )
    group.build_layouts()

    assert group.persistent_layout["child"] == slice(0, 5)
    assert group.persistent_local_buffer_size() == 5


def test_alias_fallback_local():
    """Local aliased buffer falls back to local pile."""
    group = BufferGroup()
    group.register("parent", 10, "local", persistent=True)
    group.register("child", 5, "local", aliases="parent")
    group.build_layouts()

    assert group.local_sizes["child"] == 5


def test_alias_local_child_of_shared_parent_overlaps():
    """Local child aliasing shared parent overlaps in shared."""
    group = BufferGroup()
    group.register("parent", 100, "shared")
    group.register("child", 30, "local", aliases="parent")
    group.build_layouts()

    assert group.shared_layout["child"] == slice(0, 30)
    assert group.local_buffer_size() == 0


def test_multiple_aliases_sequential_consumption():
    """Multiple aliases consume parent space sequentially."""
    group = BufferGroup()
    group.register("parent", 100, "shared")
    group.register("c1", 40, "shared", aliases="parent")
    group.register("c2", 40, "shared", aliases="parent")
    group.register("c3", 40, "shared", aliases="parent")
    group.build_layouts()

    assert group.shared_layout["c1"] == slice(0, 40)
    assert group.shared_layout["c2"] == slice(40, 80)
    # c3 doesn't fit (only 20 left), gets own allocation
    assert group.shared_layout["c3"] == slice(100, 140)
    assert group.shared_buffer_size() == 140


def test_alias_never_overlaps_persistent_shared_parent():
    """A persistent shared parent keeps its window to itself."""
    group = BufferGroup()
    group.register("parent", 100, "shared", persistent=True)
    group.register("child", 30, "shared", aliases="parent")
    group.build_layouts()

    assert group.shared_layout["parent"] == slice(0, 100)
    assert group.shared_layout["child"] == slice(100, 130)
    assert group.protected_shared_ranges() == ((0, 100),)


def test_persistent_alias_never_overlaps_scratch_parent():
    """A persistent shared alias gets its own window."""
    group = BufferGroup()
    group.register("parent", 100, "shared")
    group.register(
        "child", 30, "shared", persistent=True, aliases="parent",
    )
    group.build_layouts()

    assert group.shared_layout["child"] == slice(100, 130)
    assert group.protected_shared_ranges() == ((100, 130),)


def test_alias_skips_protected_range_of_rollup():
    """An alias binds only where the parent holds no persistent range."""
    group = BufferGroup()
    group.register("rollup", 40, "shared", protected=((20, 21),))
    group.register("fits", 15, "shared", aliases="rollup")
    group.register("clobbers", 25, "shared", aliases="rollup")
    group.build_layouts()

    assert group.shared_layout["fits"] == slice(0, 15)
    assert group.shared_layout["clobbers"] == slice(40, 65)
    assert group.protected_shared_ranges() == ((20, 21),)


def test_update_buffer_keeps_protected_tuple():
    """Location updates carry the protected ranges through unchanged."""
    group = BufferGroup()
    group.register("rollup", 40, "shared", protected=((20, 21),))
    recognized, changed = group.update_buffer("rollup", size=50)

    assert (recognized, changed) == (True, True)
    assert group.entries["rollup"].protected == ((20, 21),)


def test_register_child_stamps_nested_persistent_ranges(fresh_registry):
    """Child roll-ups carry persistent ranges up through every level."""

    class _Owner:
        precision = np.float32

    inner, mid, outer = _Owner(), _Owner(), _Owner()
    fresh_registry.register("delta", inner, 6, "shared")
    fresh_registry.register(
        "prev_theta", inner, 1, "shared", persistent=True,
    )
    fresh_registry.register("mid_scratch", mid, 4, "shared")
    fresh_registry.register_child(mid, inner, name="solver")
    fresh_registry.register("outer_scratch", outer, 2, "shared")
    fresh_registry.register_child(outer, mid, name="algorithm")
    fresh_registry.register(
        "error_solver_shared", outer, 11, "shared",
        aliases="algorithm_shared",
    )
    fresh_registry.register(
        "small_scratch", outer, 4, "shared", aliases="algorithm_shared",
    )

    mid_group = fresh_registry._groups[mid]
    outer_group = fresh_registry._groups[outer]
    assert mid_group.entries["solver_shared"].protected == ((6, 7),)
    assert mid_group.protected_shared_ranges() == ((10, 11),)
    assert outer_group.entries["algorithm_shared"].protected == ((10, 11),)
    algorithm_window = outer_group.shared_layout["algorithm_shared"]
    assert outer_group.shared_layout["small_scratch"] == slice(
        algorithm_window.start, algorithm_window.start + 4
    )
    assert outer_group.shared_layout["error_solver_shared"] == slice(
        algorithm_window.stop, algorithm_window.stop + 11
    )
    for owner in (outer, mid, inner):
        fresh_registry.clear_parent(owner)


# ── BufferGroup lazy property triggers ────────────────────── #


@pytest.mark.parametrize(
    "prop",
    [
        pytest.param("shared_layout", id="shared"),
        pytest.param("persistent_layout", id="persistent"),
        pytest.param("local_sizes", id="local"),
    ],
)
def test_layout_property_triggers_build(prop):
    """Accessing layout property triggers build when None."""
    group = BufferGroup()
    group.register("s", 5, "shared")
    group.register("p", 3, "local", persistent=True)
    group.register("l", 2, "local")

    assert group._shared_layout is None
    _ = getattr(group, prop)
    assert group._shared_layout is not None
    assert group._persistent_layout is not None
    assert group._local_sizes is not None


# ── BufferGroup size methods ──────────────────────────────── #


def test_shared_buffer_size_empty():
    """shared_buffer_size returns 0 for empty layout."""
    group = BufferGroup()
    assert group.shared_buffer_size() == 0


def test_shared_buffer_size_returns_max_stop():
    """shared_buffer_size returns max slice stop."""
    group = BufferGroup()
    group.register("a", 10, "shared")
    group.register("b", 20, "shared")
    assert group.shared_buffer_size() == 30


def test_local_buffer_size_returns_sum():
    """local_buffer_size returns sum of local sizes."""
    group = BufferGroup()
    group.register("a", 5, "local")
    group.register("b", 8, "local")
    assert group.local_buffer_size() == 13


def test_persistent_buffer_size_empty():
    """persistent_local_buffer_size returns 0 for empty layout."""
    group = BufferGroup()
    assert group.persistent_local_buffer_size() == 0


def test_persistent_buffer_size_returns_max_stop():
    """persistent_local_buffer_size returns max slice stop."""
    group = BufferGroup()
    group.register("a", 30, "local", persistent=True)
    group.register("b", 40, "local", persistent=True)
    assert group.persistent_local_buffer_size() == 70


# ── BufferGroup.get_allocator ─────────────────────────────── #


def test_get_allocator_unregistered_raises():
    """get_allocator raises KeyError for unregistered buffer."""
    group = BufferGroup()
    with pytest.raises(KeyError, match="not registered"):
        group.get_allocator("missing")


def test_get_allocator_returns_allocator_for_registered():
    """get_allocator returns allocator with correct name."""
    group = BufferGroup()
    group.register("buf", 10, "shared")
    alloc = group.get_allocator("buf")
    assert callable(alloc)
    assert alloc.__name__ == "allocate_buffer"


# ── BufferRegistry central registry ──────────────────────── #


def test_registry_register_creates_group(
    fresh_registry, step_controller,
):
    """register creates new BufferGroup for unknown parent."""
    fresh_registry.register("buf", step_controller, 10, "shared")
    assert step_controller in fresh_registry._groups
    entry = fresh_registry._groups[step_controller].entries["buf"]
    assert entry.size == 10
    assert entry.location == "shared"
    # Cleanup: remove group so other tests start clean
    fresh_registry.clear_parent(step_controller)


def test_registry_register_reuses_group(
    fresh_registry, step_controller,
):
    """register reuses existing group for known parent."""
    fresh_registry.register("a", step_controller, 10, "shared")
    fresh_registry.register("b", step_controller, 5, "shared")
    assert len(fresh_registry._groups) >= 1
    entries = fresh_registry._groups[step_controller].entries
    assert "a" in entries
    assert "b" in entries
    fresh_registry.clear_parent(step_controller)


def test_registry_update_buffer_unknown_parent(
    fresh_registry, output_functions,
):
    """update_buffer returns (False, False) for unknown parent."""
    # output_functions is not registered, so it's genuinely unknown
    recognized, changed = fresh_registry.update_buffer(
        "buf", output_functions,
    )
    assert recognized is False
    assert changed is False


def test_registry_update_buffer_delegates(
    fresh_registry, step_controller,
):
    """update_buffer delegates to group for known parent."""
    fresh_registry.register("buf", step_controller, 10, "shared")
    recognized, changed = fresh_registry.update_buffer(
        "buf", step_controller, size=20,
    )
    assert recognized is True
    assert changed is True
    assert (
        fresh_registry._groups[step_controller].entries["buf"].size
        == 20
    )
    fresh_registry.clear_parent(step_controller)


def test_registry_clear_layout_known_parent(
    fresh_registry, step_controller,
):
    """clear_layout invalidates layouts for known parent."""
    fresh_registry.register("buf", step_controller, 10, "shared")
    _ = fresh_registry.shared_buffer_size(step_controller)
    group = fresh_registry._groups[step_controller]
    assert group._shared_layout is not None

    fresh_registry.clear_layout(step_controller)
    assert group._shared_layout is None
    fresh_registry.clear_parent(step_controller)


def test_registry_clear_layout_unknown_parent_noop(
    fresh_registry, output_functions,
):
    """clear_layout is a no-op for unknown parent."""
    fresh_registry.clear_layout(output_functions)  # should not raise


def test_registry_clear_parent_removes_group(
    fresh_registry, step_controller,
):
    """clear_parent removes group for known parent."""
    fresh_registry.register("buf", step_controller, 10, "shared")
    fresh_registry.clear_parent(step_controller)
    assert step_controller not in fresh_registry._groups


def test_registry_clear_parent_unknown_noop(
    fresh_registry, output_functions,
):
    """clear_parent is a no-op for unknown parent."""
    fresh_registry.clear_parent(output_functions)  # should not raise


def test_registry_clear_parent_cascades_through_children(
    fresh_registry, single_integrator_run, step_controller,
    output_functions,
):
    """clear_parent removes recorded children recursively.

    Chain: single_integrator_run hosts step_controller, which hosts
    output_functions; clearing the root clears all three groups.
    """
    fresh_registry.register("inner", output_functions, 4, "shared")
    fresh_registry.register("mid", step_controller, 6, "shared")
    fresh_registry.register_child(
        step_controller, output_functions, name="inner_child",
    )
    fresh_registry.register(
        "outer", single_integrator_run, 8, "shared",
    )
    fresh_registry.register_child(
        single_integrator_run, step_controller, name="mid_child",
    )

    fresh_registry.clear_parent(single_integrator_run)
    assert single_integrator_run not in fresh_registry._groups
    assert step_controller not in fresh_registry._groups
    assert output_functions not in fresh_registry._groups


def test_registry_child_reregistration_replaces_recorded_child(
    fresh_registry, single_integrator_run, step_controller,
    output_functions,
):
    """Re-registering a child name replaces the recorded child.

    After the same base name is registered with a new child, a
    cascade from the parent clears the new child only; the replaced
    child's group survives.
    """
    fresh_registry.register("a", step_controller, 6, "shared")
    fresh_registry.get_child_allocators(
        single_integrator_run, step_controller, name="component",
    )
    fresh_registry.register("b", output_functions, 4, "shared")
    fresh_registry.get_child_allocators(
        single_integrator_run, output_functions, name="component",
    )

    fresh_registry.clear_parent(single_integrator_run)
    assert output_functions not in fresh_registry._groups
    assert step_controller in fresh_registry._groups
    fresh_registry.clear_parent(step_controller)


def test_registry_clear_parent_terminates_on_cycle(
    fresh_registry, single_integrator_run, step_controller,
):
    """A registration cycle clears both groups without recursing."""
    fresh_registry.register("a", single_integrator_run, 4, "shared")
    fresh_registry.register("b", step_controller, 4, "shared")
    fresh_registry.register_child(
        single_integrator_run, step_controller, name="down",
    )
    fresh_registry.register_child(
        step_controller, single_integrator_run, name="up",
    )

    fresh_registry.clear_parent(single_integrator_run)
    assert single_integrator_run not in fresh_registry._groups
    assert step_controller not in fresh_registry._groups


def test_registry_reset_clears_all(
    fresh_registry, step_controller, output_functions,
):
    """reset clears all groups."""
    fresh_registry.register("a", step_controller, 10, "shared")
    fresh_registry.register("b", output_functions, 5, "local")
    fresh_registry.reset()
    assert len(fresh_registry._groups) == 0


# ── BufferRegistry.update ─────────────────────────────────── #


def test_registry_update_empty_returns_empty(
    fresh_registry, step_controller,
):
    """update returns empty set for empty updates."""
    fresh_registry.register("buf", step_controller, 10, "local")
    assert fresh_registry.update(step_controller) == set()
    fresh_registry.clear_parent(step_controller)


def test_registry_update_unknown_parent_returns_empty(
    fresh_registry, output_functions,
):
    """update returns empty set for unknown parent."""
    result = fresh_registry.update(
        output_functions, buf_location="shared",
    )
    assert result == set()


def test_registry_update_recognizes_location_keys(
    fresh_registry, step_controller,
):
    """update recognizes keys ending in _location."""
    fresh_registry.register("buf", step_controller, 10, "local")
    recognized = fresh_registry.update(
        step_controller, buf_location="shared",
    )
    assert "buf_location" in recognized
    fresh_registry.clear_parent(step_controller)


def test_registry_update_invalid_location_raises(
    fresh_registry, step_controller,
):
    """update raises ValueError for invalid location value."""
    fresh_registry.register("buf", step_controller, 10, "local")
    try:
        with pytest.raises(ValueError, match="Invalid location"):
            fresh_registry.update(
                step_controller, buf_location="invalid",
            )
    finally:
        fresh_registry.clear_parent(step_controller)


def test_registry_update_changes_location_and_invalidates(
    fresh_registry, step_controller,
):
    """update changes location and invalidates layouts."""
    fresh_registry.register("buf", step_controller, 10, "local")
    _ = fresh_registry.local_buffer_size(step_controller)
    group = fresh_registry._groups[step_controller]
    assert group._local_sizes is not None

    fresh_registry.update(step_controller, buf_location="shared")
    assert group.entries["buf"].location == "shared"
    assert group._local_sizes is None
    fresh_registry.clear_parent(step_controller)


def test_registry_update_returns_all_recognized(
    fresh_registry, step_controller,
):
    """update returns set of all recognized keys."""
    fresh_registry.register("a", step_controller, 10, "local")
    fresh_registry.register("b", step_controller, 5, "local")
    recognized = fresh_registry.update(
        step_controller,
        updates_dict={"a_location": "shared"},
        b_location="shared",
    )
    assert recognized == {"a_location", "b_location"}
    fresh_registry.clear_parent(step_controller)


def test_registry_update_ignores_non_location_keys(
    fresh_registry, step_controller,
):
    """update ignores params not ending in _location."""
    fresh_registry.register("buf", step_controller, 10, "local")
    recognized = fresh_registry.update(
        step_controller, other_param="value",
    )
    assert recognized == set()
    fresh_registry.clear_parent(step_controller)


def test_registry_update_no_change_preserves_layout(
    fresh_registry, step_controller,
):
    """update preserves layout when location unchanged."""
    fresh_registry.register("buf", step_controller, 10, "local")
    _ = fresh_registry.local_buffer_size(step_controller)
    group = fresh_registry._groups[step_controller]
    assert group._local_sizes is not None

    fresh_registry.update(step_controller, buf_location="local")
    assert group._local_sizes is not None
    fresh_registry.clear_parent(step_controller)


# ── BufferRegistry size delegation ────────────────────────── #


@pytest.mark.parametrize(
    "method",
    [
        pytest.param("shared_buffer_size", id="shared"),
        pytest.param("local_buffer_size", id="local"),
        pytest.param("persistent_local_buffer_size", id="persistent"),
    ],
)
def test_registry_size_unknown_parent_returns_zero(
    fresh_registry, output_functions, method,
):
    """Size methods return 0 for unknown parent."""
    assert getattr(fresh_registry, method)(output_functions) == 0


def test_registry_size_delegates_to_group(
    fresh_registry, step_controller,
):
    """Size methods delegate to group methods for known parent."""
    fresh_registry.register("s", step_controller, 10, "shared")
    fresh_registry.register("l", step_controller, 5, "local")
    fresh_registry.register(
        "p", step_controller, 3, "local", persistent=True,
    )

    assert fresh_registry.shared_buffer_size(step_controller) == 10
    assert fresh_registry.local_buffer_size(step_controller) == 5
    assert (
        fresh_registry.persistent_local_buffer_size(step_controller)
        == 3
    )
    fresh_registry.clear_parent(step_controller)


# ── BufferRegistry.get_allocator ──────────────────────────── #


def test_registry_get_allocator_unknown_parent_raises(
    fresh_registry, output_functions,
):
    """get_allocator raises KeyError for unknown parent."""
    with pytest.raises(KeyError, match="no registered"):
        fresh_registry.get_allocator("buf", output_functions)


def test_registry_get_allocator_delegates(
    fresh_registry, step_controller,
):
    """get_allocator delegates to group for known parent."""
    fresh_registry.register("buf", step_controller, 10, "shared")
    alloc = fresh_registry.get_allocator("buf", step_controller)
    assert callable(alloc)
    assert alloc.__name__ == "allocate_buffer"
    fresh_registry.clear_parent(step_controller)


# ── BufferRegistry separate parent contexts ───────────────── #


def test_separate_parents_independent(
    fresh_registry, step_controller, output_functions,
):
    """Different parents have independent buffer groups."""
    fresh_registry.register("buf", step_controller, 100, "shared")
    fresh_registry.register("buf", output_functions, 50, "shared")
    assert (
        fresh_registry.shared_buffer_size(step_controller) == 100
    )
    assert (
        fresh_registry.shared_buffer_size(output_functions) == 50
    )
    fresh_registry.clear_parent(step_controller)
    fresh_registry.clear_parent(output_functions)


def test_clear_one_parent_preserves_others(
    fresh_registry, step_controller, output_functions,
):
    """Clearing one parent does not affect others."""
    fresh_registry.register("buf", step_controller, 100, "shared")
    fresh_registry.register("buf", output_functions, 50, "shared")
    fresh_registry.clear_parent(step_controller)
    assert step_controller not in fresh_registry._groups
    assert (
        fresh_registry.shared_buffer_size(output_functions) == 50
    )
    fresh_registry.clear_parent(output_functions)


# ── BufferRegistry.get_child_allocators ───────────────────── #


def test_get_child_allocators_registers_buffers(
    single_integrator_run,
):
    """get_child_allocators registers child shared/persistent."""
    reg = BufferRegistry()
    parent = single_integrator_run
    child = single_integrator_run._loop

    child_shared = buffer_registry.shared_buffer_size(child)
    child_persistent = buffer_registry.persistent_local_buffer_size(
        child,
    )

    alloc_s, alloc_p = reg.get_child_allocators(
        parent=parent, child=child, name="loop",
    )
    entries = reg._groups[parent].entries
    assert entries["loop_shared"].size == child_shared
    assert entries["loop_persistent"].size == child_persistent
    assert callable(alloc_s)
    assert callable(alloc_p)
    assert alloc_s.__name__ == "allocate_buffer"
    assert alloc_p.__name__ == "allocate_buffer"


def test_get_child_allocators_with_name(single_integrator_run):
    """get_child_allocators uses provided name for buffer names."""
    reg = BufferRegistry()
    parent = single_integrator_run
    child = single_integrator_run._algo_step

    reg.get_child_allocators(
        parent=parent, child=child, name="solver",
    )
    assert "solver_shared" in reg._groups[parent].entries
    assert "solver_persistent" in reg._groups[parent].entries


def test_get_child_allocators_default_name(single_integrator_run):
    """get_child_allocators uses child_{id} when name=None."""
    reg = BufferRegistry()
    parent = single_integrator_run
    child = single_integrator_run._algo_step

    reg.get_child_allocators(
        parent=parent, child=child, name=None,
    )
    child_id = id(child)
    expected_shared = f"child_{child_id}_shared"
    expected_persistent = f"child_{child_id}_persistent"
    assert expected_shared in reg._groups[parent].entries
    assert expected_persistent in reg._groups[parent].entries


# ── BufferRegistry.get_toplevel_allocators ────────────────── #


def test_get_toplevel_allocators_returns_callables(solverkernel):
    """get_toplevel_allocators returns (alloc_shared, alloc_persistent)."""
    reg = BufferRegistry()
    alloc_shared, alloc_persistent = reg.get_toplevel_allocators(
        solverkernel,
    )
    assert callable(alloc_shared)
    assert callable(alloc_persistent)
    assert alloc_shared.__name__ == "alloc_shared"
    assert alloc_persistent.__name__ == "alloc_persistent"


# ── Deterministic layout order ────────────────────────────── #


def test_layout_deterministic_regardless_of_access_order(
    single_integrator_run,
):
    """Layout is deterministic regardless of property access order."""
    group = BufferGroup()
    group.register("parent", 100, "shared")
    group.register("child", 30, "shared", aliases="parent")
    group.register("local", 20, "local")
    group.register("persist", 10, "local", persistent=True)

    group.build_layouts()
    shared1 = dict(group.shared_layout)
    persistent1 = dict(group.persistent_layout)
    local1 = dict(group.local_sizes)

    group.invalidate_layouts()

    # Access in different order
    local2 = dict(group.local_sizes)
    persistent2 = dict(group.persistent_layout)
    shared2 = dict(group.shared_layout)

    assert shared1 == shared2
    assert persistent1 == persistent2
    assert local1 == local2


# ── get_child_allocators size snapshots ───────────────────── #


def test_child_allocators_snapshot_unregistered_child_is_zero(
    fresh_registry, step_controller, output_functions,
):
    """A child with no buffer group registers zero-size entries."""
    fresh_registry.get_child_allocators(
        step_controller, output_functions, name="solver"
    )
    entries = fresh_registry._groups[step_controller].entries
    assert entries["solver_shared"].size == 0
    assert entries["solver_persistent"].size == 0
    fresh_registry.clear_parent(step_controller)


def test_child_allocators_reregistration_refreshes_sizes(
    fresh_registry, step_controller, output_functions,
):
    """Re-registering under the same name picks up late child buffers.

    Parents snapshot child sizes at call time, so buffers a child
    registers later (e.g. its own grandchildren during build) are
    missed until the parent re-registers under the same name.
    """
    fresh_registry.register("x", output_functions, 4, "shared")
    fresh_registry.get_child_allocators(
        step_controller, output_functions, name="solver"
    )
    parent_entries = fresh_registry._groups[step_controller].entries
    assert parent_entries["solver_shared"].size == 4

    # Late grandchild-style registration on the child
    fresh_registry.register("y", output_functions, 6, "shared")
    assert parent_entries["solver_shared"].size == 4

    fresh_registry.get_child_allocators(
        step_controller, output_functions, name="solver"
    )
    parent_entries = fresh_registry._groups[step_controller].entries
    assert parent_entries["solver_shared"].size == 10

    fresh_registry.clear_parent(step_controller)
    fresh_registry.clear_parent(output_functions)


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        merge_dicts(
            MID_RUN_PARAMS,
            {
                "algorithm": "firk",
                "step_controller": "fixed",
                "preconditioned_vec_location": "shared",
            },
        )
    ],
    indirect=True,
)
def test_nested_shared_solver_buffers_sized_through_chain(
    single_integrator_run,
):
    """Loop pool sizing sees shared buffers of deeply nested children.

    With a linear-solver buffer placed in shared memory, the size
    chain loop -> algorithm -> newton_krylov -> linear_solver must be
    consistent after build; stale snapshots undersize the shared pool
    and the kernel indexes past its end (issue #520).
    """
    _ = single_integrator_run.device_function

    loop = single_integrator_run._loop
    algo = single_integrator_run._algo_step
    newton = algo.solver
    linear = newton.linear_solver

    linear_entries = buffer_registry._groups[linear].entries
    assert linear_entries["preconditioned_vec"].location == "shared"
    linear_shared = buffer_registry.shared_buffer_size(linear)
    assert linear_shared > 0

    newton_entries = buffer_registry._groups[newton].entries
    assert (
        newton_entries["linear_solver_shared"].size == linear_shared
    )

    algo_entries = buffer_registry._groups[algo].entries
    assert (
        algo_entries["solver_shared"].size
        == buffer_registry.shared_buffer_size(newton)
    )

    loop_entries = buffer_registry._groups[loop].entries
    assert (
        loop_entries["algorithm_shared"].size
        == buffer_registry.shared_buffer_size(algo)
    )


# ── Dead-parent release ───────────────────────────────────── #


class _Owner:
    """Weakref-able stand-in for a buffer-owning component."""

    precision = np.float32


def test_dead_parent_group_is_released(fresh_registry):
    """A parent's buffer group disappears when the parent dies."""
    owner = _Owner()
    fresh_registry.register("buf", owner, 8, "shared")
    assert owner in fresh_registry._groups
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is None
    assert "buf" not in [
        name
        for group in fresh_registry._groups.values()
        for name in group.entries
    ]


def test_child_released_with_its_parent(fresh_registry):
    """A recorded child is released once its parent dies."""
    parent = _Owner()
    child = _Owner()
    fresh_registry.register("outer", parent, 8, "shared")
    fresh_registry.register("inner", child, 4, "shared")
    fresh_registry.register_child(parent, child, name="component")
    child_ref = weakref.ref(child)
    del child
    gc.collect()
    # The ownership edge keeps the child alive with its parent.
    assert child_ref() is not None
    del parent
    gc.collect()
    assert child_ref() is None
