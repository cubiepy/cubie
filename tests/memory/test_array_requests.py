"""Tests for cubie.memory.array_requests."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.memory.array_requests import ArrayRequest, ArrayResponse


# ── ArrayRequest: dtype validation ──────────────────── #

@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.float64, id="float64"),
        pytest.param(np.float32, id="float32"),
        pytest.param(np.int32, id="int32"),
    ],
)
def test_array_request_accepts_valid_dtypes(dtype):
    """ArrayRequest accepts float64, float32, and int32."""
    req = ArrayRequest(dtype=dtype)
    assert req.dtype is dtype


def test_array_request_rejects_invalid_dtype():
    """ArrayRequest rejects dtypes not in the allowed set."""
    with pytest.raises(ValueError):
        ArrayRequest(dtype=np.int64)


# ── ArrayRequest: defaults ──────────────────────────── #

def test_array_request_defaults():
    """Default shape, memory, chunk_axis_index, unchunkable, total_runs."""
    req = ArrayRequest(dtype=np.float64)
    assert req.shape == (1, 1, 1)
    assert req.memory == "device"
    assert req.chunk_axis_index == 2
    assert req.unchunkable is False
    assert req.total_runs == 1


# ── ArrayRequest: shape validation ──────────────────── #

def test_array_request_accepts_custom_shape():
    """ArrayRequest stores a custom shape tuple."""
    req = ArrayRequest(dtype=np.float32, shape=(100, 4, 50))
    assert req.shape == (100, 4, 50)


def test_array_request_rejects_non_tuple_shape():
    """Shape must be a tuple, not a list."""
    with pytest.raises((TypeError, ValueError)):
        ArrayRequest(dtype=np.float64, shape=[10, 20])


# ── ArrayRequest: memory validation ─────────────────── #

@pytest.mark.parametrize(
    "mem",
    [
        pytest.param("device", id="device"),
        pytest.param("pinned", id="pinned"),
    ],
)
def test_array_request_accepts_valid_memory(mem):
    """ArrayRequest accepts the supported memory placements."""
    req = ArrayRequest(dtype=np.float64, memory=mem)
    assert req.memory == mem


@pytest.mark.parametrize(
    "mem",
    [
        pytest.param("host", id="host"),
        pytest.param("mapped", id="mapped"),
        pytest.param("managed", id="managed"),
    ],
)
def test_array_request_rejects_invalid_memory(mem):
    """ArrayRequest rejects unsupported memory placements."""
    with pytest.raises(ValueError):
        ArrayRequest(dtype=np.float64, memory=mem)


# ── ArrayRequest: chunk_axis_index ──────────────────── #

def test_array_request_chunk_axis_none():
    """chunk_axis_index accepts None."""
    req = ArrayRequest(dtype=np.float64, chunk_axis_index=None)
    assert req.chunk_axis_index is None


def test_array_request_chunk_axis_rejects_negative():
    """chunk_axis_index rejects negative values."""
    with pytest.raises(ValueError, match="must be >= 0"):
        ArrayRequest(dtype=np.float64, chunk_axis_index=-1)


# ── ArrayRequest: total_runs ────────────────────────── #

def test_array_request_total_runs_custom():
    """total_runs stores a positive integer."""
    req = ArrayRequest(dtype=np.float64, total_runs=100)
    assert req.total_runs == 100


def test_array_request_total_runs_rejects_zero():
    """total_runs rejects zero."""
    with pytest.raises(ValueError, match="must be >= 1"):
        ArrayRequest(dtype=np.float64, total_runs=0)


def test_array_request_total_runs_rejects_none():
    """total_runs rejects None (not optional)."""
    with pytest.raises((TypeError, ValueError)):
        ArrayRequest(dtype=np.float64, total_runs=None)


# ── ArrayResponse: defaults ─────────────────────────── #

def test_array_response_defaults():
    """ArrayResponse defaults: arr={}, chunks=1, chunk_length=1,
    chunked_shapes={}.
    """
    resp = ArrayResponse()
    assert resp.arr == {}
    assert resp.chunks == 1
    assert resp.chunk_length == 1
    assert resp.chunked_shapes == {}


def test_array_response_stores_chunked_shapes():
    """ArrayResponse stores provided chunked_shapes dict."""
    shapes = {"output": (100, 3, 50), "state": (3, 50)}
    resp = ArrayResponse(chunks=2, chunk_length=50, chunked_shapes=shapes)
    assert resp.chunks == 2
    assert resp.chunk_length == 50
    assert resp.chunked_shapes["output"] == (100, 3, 50)
    assert resp.chunked_shapes["state"] == (3, 50)
