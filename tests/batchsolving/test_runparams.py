"""Tests for RunParams class."""

import pytest
from attrs import evolve

from cubie.batchsolving.BatchSolverKernel import RunParams
from cubie.memory.array_requests import ArrayResponse


def test_runparams_creation():
    """Verify RunParams can be created with valid parameters."""
    # Create with all parameters specified
    params = RunParams(
        duration=1.0,
        warmup=0.1,
        t0=0.0,
        runs=100,
        num_chunks=4,
        chunk_length=25,
    )

    assert params.duration == 1.0
    assert params.warmup == 0.1
    assert params.t0 == 0.0
    assert params.runs == 100
    assert params.num_chunks == 4
    assert params.chunk_length == 25

    # Create with default chunking (num_chunks=1, chunk_length=0)
    params_default = RunParams(
        duration=2.0,
        warmup=0.5,
        t0=1.0,
        runs=50,
    )

    assert params_default.duration == 2.0
    assert params_default.warmup == 0.5
    assert params_default.t0 == 1.0
    assert params_default.runs == 50
    assert params_default.num_chunks == 1
    assert params_default.chunk_length == 0


def test_runparams_creation_validates_duration():
    """Verify duration validation rejects negative values."""
    with pytest.raises(ValueError):
        RunParams(duration=-1.0, warmup=0.0, t0=0.0, runs=10)


def test_runparams_creation_validates_warmup():
    """Verify warmup validation rejects negative values."""
    with pytest.raises(ValueError):
        RunParams(duration=1.0, warmup=-0.1, t0=0.0, runs=10)


def test_runparams_creation_validates_runs():
    """Verify runs validation rejects zero and negative values."""
    with pytest.raises(ValueError):
        RunParams(duration=1.0, warmup=0.0, t0=0.0, runs=0)

    with pytest.raises(ValueError):
        RunParams(duration=1.0, warmup=0.0, t0=0.0, runs=-5)


def test_runparams_creation_validates_num_chunks():
    """Verify num_chunks validation rejects zero and negative values."""
    with pytest.raises(ValueError):
        RunParams(duration=1.0, warmup=0.0, t0=0.0, runs=10, num_chunks=0)

    with pytest.raises(ValueError):
        RunParams(duration=1.0, warmup=0.0, t0=0.0, runs=10, num_chunks=-1)


def test_runparams_getitem_single_chunk():
    """Verify __getitem__(0) returns full runs when num_chunks=1."""
    params = RunParams(
        duration=1.0,
        warmup=0.1,
        t0=0.0,
        runs=100,
        num_chunks=1,
        chunk_length=100,
    )

    chunk_params = params[0]

    # Chunk parameters should have same runs as original
    assert chunk_params.runs == 100
    assert chunk_params.duration == 1.0
    assert chunk_params.warmup == 0.1
    assert chunk_params.t0 == 0.0


def test_runparams_getitem_multiple_chunks():
    """Verify __getitem__ returns correct runs for each chunk."""
    # 100 runs divided into 4 chunks: ceil(100/4) = 25 runs per chunk
    params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=100,
        num_chunks=4,
        chunk_length=25,
    )

    # First three chunks should have chunk_length runs
    for i in range(3):
        chunk_params = params[i]
        assert chunk_params.runs == 25
        assert chunk_params.duration == 1.0
        assert chunk_params.warmup == 0.0
        assert chunk_params.t0 == 0.0

    # Last chunk should have remaining runs: 100 - (3 * 25) = 25
    last_chunk = params[3]
    assert last_chunk.runs == 25


def test_runparams_getitem_dangling_chunk():
    """Verify last chunk gets correct remaining runs."""
    # 100 runs divided into 3 chunks: ceil(100/3) = 34 runs per chunk
    # First 2 chunks: 34 runs each
    # Last chunk: 100 - (2 * 34) = 32 runs (dangling)
    params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=100,
        num_chunks=3,
        chunk_length=34,
    )

    # First two chunks should have chunk_length runs
    chunk0 = params[0]
    assert chunk0.runs == 34

    chunk1 = params[1]
    assert chunk1.runs == 34

    # Last chunk should have remaining runs: 100 - (2 * 34) = 32
    chunk2 = params[2]
    assert chunk2.runs == 32


def test_runparams_getitem_exact_division():
    """Verify all chunks equal when runs % chunk_length == 0."""
    # 100 runs divided into 4 chunks: 100/4 = 25 runs per chunk (exact)
    params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=100,
        num_chunks=4,
        chunk_length=25,
    )

    # All chunks should have exactly chunk_length runs
    for i in range(4):
        chunk_params = params[i]
        assert chunk_params.runs == 25

    # 1000 runs divided into 10 chunks: 1000/10 = 100 runs per chunk
    params2 = RunParams(
        duration=2.0,
        warmup=0.1,
        t0=0.0,
        runs=1000,
        num_chunks=10,
        chunk_length=100,
    )

    for i in range(10):
        chunk_params = params2[i]
        assert chunk_params.runs == 100


def test_runparams_getitem_out_of_bounds():
    """Verify IndexError raised for invalid indices."""
    params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=100,
        num_chunks=4,
        chunk_length=25,
    )

    # Negative index should raise IndexError
    with pytest.raises(IndexError, match="out of range"):
        _ = params[-1]

    # Index >= num_chunks should raise IndexError
    with pytest.raises(IndexError, match="out of range"):
        _ = params[4]

    with pytest.raises(IndexError, match="out of range"):
        _ = params[10]


def test_runparams_update_from_allocation():
    """Verify update_from_allocation sets num_chunks and chunk_length."""
    # Create initial params without chunking
    params = RunParams(
        duration=1.0,
        warmup=0.1,
        t0=0.0,
        runs=100,
    )

    # Create mock allocation response with chunking
    response = ArrayResponse(
        arr={},
        chunks=4,
        chunk_length=25,
    )

    # Update params from allocation response
    updated_params = params.update_from_allocation(response)

    # Check that chunking metadata was updated
    assert updated_params.num_chunks == 4
    assert updated_params.chunk_length == 25

    # Original time parameters should be preserved
    assert updated_params.duration == 1.0
    assert updated_params.warmup == 0.1
    assert updated_params.t0 == 0.0
    assert updated_params.runs == 100


def test_runparams_update_from_allocation_single_chunk():
    """Verify update_from_allocation handles single chunk case."""
    params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=50,
    )

    # Allocation response with no chunking
    response = ArrayResponse(
        arr={},
        chunks=1,
        chunk_length=50,
    )

    updated_params = params.update_from_allocation(response)

    # When num_chunks=1, chunk_length should equal runs
    assert updated_params.num_chunks == 1
    assert updated_params.chunk_length == 50


def test_runparams_update_from_allocation_dangling_chunk():
    """Verify update_from_allocation calculates chunk_length correctly."""
    params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=100,
    )

    # Allocation response requiring 3 chunks
    response = ArrayResponse(
        arr={},
        chunks=3,
        chunk_length=34,
    )

    updated_params = params.update_from_allocation(response)

    # chunk_length should be ceil(100/3) = 34
    assert updated_params.num_chunks == 3
    assert updated_params.chunk_length == 34

    # Verify the last chunk calculation
    last_chunk = updated_params[2]
    # Last chunk: 100 - (2 * 34) = 32
    assert last_chunk.runs == 32


def test_runparams_update_from_allocation_rejects_short_partition():
    """A partition holding fewer runs than the batch raises."""
    params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=100,
    )
    response = ArrayResponse(
        arr={},
        chunks=1,
        chunk_length=1,
    )

    with pytest.raises(ValueError, match="fewer than the batch's 100"):
        params.update_from_allocation(response)


def test_runparams_immutability():
    """Verify RunParams is immutable (frozen)."""
    params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=100,
    )

    # Attempting to modify should raise FrozenInstanceError
    with pytest.raises(Exception):  # attrs.exceptions.FrozenInstanceError
        params.runs = 200

    with pytest.raises(Exception):
        params.num_chunks = 5


def test_runparams_evolve_pattern():
    """Verify RunParams can be modified using evolve()."""
    params = RunParams(
        duration=1.0,
        warmup=0.0,
        t0=0.0,
        runs=100,
    )

    # Use evolve to create modified copy
    updated = evolve(params, runs=200, num_chunks=5)

    # Original should be unchanged
    assert params.runs == 100
    assert params.num_chunks == 1

    # Updated should have new values
    assert updated.runs == 200
    assert updated.num_chunks == 5
