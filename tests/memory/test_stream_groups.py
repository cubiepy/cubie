"""Tests for cubie.memory.stream_groups."""

from __future__ import annotations

import pytest

from cubie.memory.stream_groups import StreamGroups


# ── __attrs_post_init__ ───────────────────────────────────────── #

def test_default_group_created_when_none():
    """Passing groups=None triggers post_init to create default group."""
    sg = StreamGroups(groups=None)
    assert "default" in sg.groups


def test_empty_dict_default_no_post_init():
    """Default Factory(dict) produces empty dict; post_init skips it.

    This is a source defect: __attrs_post_init__ checks ``is None``
    but the default is an empty dict, so the "default" group is NOT
    auto-created when using default construction.
    """
    sg = StreamGroups()
    # Source defect: default group NOT created with empty-dict default
    assert "default" not in sg.groups


# ── add_instance ──────────────────────────────────────────────── #

def test_add_instance_with_int():
    """add_instance accepts int identifiers."""
    sg = StreamGroups()
    sg.add_instance(42, "g1")
    assert 42 in sg.groups["g1"]


def test_add_instance_with_object():
    """add_instance uses id() for non-int objects."""
    sg = StreamGroups()
    obj = object()
    sg.add_instance(obj, "g1")
    assert id(obj) in sg.groups["g1"]


def test_add_instance_raises_if_already_registered():
    """add_instance raises ValueError for duplicate instance."""
    sg = StreamGroups()
    sg.add_instance(99, "g1")
    with pytest.raises(ValueError, match="already in a stream group"):
        sg.add_instance(99, "g2")


def test_add_instance_creates_new_group():
    """add_instance creates a new group with stream when group missing."""
    sg = StreamGroups()
    sg.add_instance(10, "new_group")
    assert "new_group" in sg.groups
    assert "new_group" in sg.streams


def test_add_instance_appends_to_existing():
    """add_instance appends to existing group list."""
    sg = StreamGroups()
    sg.add_instance(1, "g1")
    sg.add_instance(2, "g1")
    assert sg.groups["g1"] == [1, 2]


# ── get_group ─────────────────────────────────────────────────── #

def test_get_group_returns_correct_group():
    """get_group returns the name of the group containing the instance."""
    sg = StreamGroups()
    sg.add_instance(7, "alpha")
    assert sg.get_group(7) == "alpha"


def test_get_group_raises_for_unknown():
    """get_group raises ValueError for unregistered instance."""
    sg = StreamGroups()
    with pytest.raises(ValueError, match="not in any stream groups"):
        sg.get_group(999)


def test_get_group_handles_object():
    """get_group resolves non-int instances via id()."""
    sg = StreamGroups()
    obj = object()
    sg.add_instance(obj, "beta")
    assert sg.get_group(obj) == "beta"


# ── get_group_stream ──────────────────────────────────────────── #

def test_get_group_stream_creates_dedicated_stream():
    """get_group_stream lazily creates a dedicated per-group stream."""
    sg = StreamGroups()
    stream = sg.get_group_stream("default")
    assert "default" in sg.groups
    assert stream is not None
    assert stream != 0
    assert sg.get_group_stream("default") is stream
    assert sg.get_group_stream("other") is not stream


def test_get_group_stream_backfills_group_without_stream():
    """A group created without a stream receives one on first access."""
    sg = StreamGroups(groups=None)
    assert "default" not in sg.streams
    stream = sg.get_group_stream("default")
    sg.add_instance(1, "default")
    assert sg.get_stream(1) is stream


def test_add_instance_shares_group_stream():
    """Instances added after get_group_stream share the same stream."""
    sg = StreamGroups()
    stream = sg.get_group_stream("default")
    sg.add_instance(11, "default")
    assert sg.get_stream(11) is stream


def test_memory_manager_get_group_stream_passthrough(mgr):
    """MemoryManager.get_group_stream returns the group's stream."""
    stream = mgr.get_group_stream()
    assert stream is not None
    assert stream != 0
    mgr.stream_groups.add_instance(21, "default")
    assert mgr.get_stream(21) is stream


# ── get_stream ────────────────────────────────────────────────── #

def test_get_stream_returns_group_stream():
    """get_stream returns the stream associated with the instance's group."""
    sg = StreamGroups()
    sg.add_instance(5, "g1")
    stream = sg.get_stream(5)
    assert stream is sg.streams["g1"]


# ── get_instances_in_group ────────────────────────────────────── #

def test_get_instances_in_group_existing():
    """get_instances_in_group returns instance ids for existing group."""
    sg = StreamGroups()
    sg.add_instance(1, "g1")
    sg.add_instance(2, "g1")
    result = sg.get_instances_in_group("g1")
    assert result == [1, 2]


def test_get_instances_in_group_nonexistent():
    """get_instances_in_group returns empty list for unknown group."""
    sg = StreamGroups()
    assert sg.get_instances_in_group("no_such") == []


# ── change_group ──────────────────────────────────────────────── #

def test_change_group_moves_instance():
    """change_group removes from old group and adds to new."""
    sg = StreamGroups()
    sg.add_instance(10, "src")
    sg.change_group(10, "dst")
    assert 10 not in sg.groups["src"]
    assert 10 in sg.groups["dst"]


def test_change_group_creates_target_if_missing():
    """change_group creates new group with stream when target missing."""
    sg = StreamGroups()
    sg.add_instance(10, "src")
    sg.change_group(10, "brand_new")
    assert "brand_new" in sg.groups
    assert "brand_new" in sg.streams
    assert 10 in sg.groups["brand_new"]


def test_change_group_handles_object():
    """change_group works with non-int instances."""
    sg = StreamGroups()
    obj = object()
    sg.add_instance(obj, "a")
    sg.change_group(obj, "b")
    assert id(obj) in sg.groups["b"]
    assert id(obj) not in sg.groups["a"]


# ── reinit_streams ────────────────────────────────────────────── #

def test_reinit_streams_replaces_all():
    """reinit_streams creates fresh streams for every group."""
    sg = StreamGroups()
    sg.add_instance(1, "g1")
    sg.add_instance(2, "g2")
    old_g1 = sg.streams["g1"]
    old_g2 = sg.streams["g2"]
    sg.reinit_streams()
    # All streams should be replaced (different objects)
    assert sg.streams["g1"] is not old_g1
    assert sg.streams["g2"] is not old_g2
    # Groups should still exist
    assert 1 in sg.groups["g1"]
    assert 2 in sg.groups["g2"]
