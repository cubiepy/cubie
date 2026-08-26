"""allocate_queue processes one owner's requests at a time."""

import numpy as np
import pytest

from cubie.memory.mem_manager import ArrayRequest, partition_covers


def _request(total_runs):
    return {
        "buf": ArrayRequest(
            shape=(2, 2, total_runs),
            dtype=np.float32,
            memory="device",
            chunk_axis_index=2,
            total_runs=total_runs,
        )
    }


def _register_owner(mgr, inputs, outputs, group):
    """Register two registrants as one owner, recording responses."""
    responses = {"inputs": [], "outputs": []}
    mgr.register(
        inputs,
        allocation_ready_hook=lambda r: responses["inputs"].append(r),
        invalidate_cache_hook=inputs.notice_invalidate,
        stream_group=group,
    )
    mgr.register(
        outputs,
        allocation_ready_hook=lambda r: responses["outputs"].append(r),
        invalidate_cache_hook=outputs.notice_invalidate,
        stream_group=group,
        owner=inputs,
    )
    return responses


@pytest.mark.parametrize("memory_clients", [3], indirect=True)
def test_peer_owner_requests_stay_queued(mgr, memory_clients):
    """Another owner's queued requests wait for that owner's trigger."""
    inputs, outputs, peer = memory_clients
    responses = _register_owner(mgr, inputs, outputs, "scoped")
    peer_responses = []
    mgr.register(
        peer,
        allocation_ready_hook=lambda r: peer_responses.append(r),
        invalidate_cache_hook=peer.notice_invalidate,
        stream_group="scoped",
    )
    mgr.queue_request(outputs, _request(8))
    mgr.queue_request(peer, _request(1))

    mgr.allocate_queue(outputs)

    owner_key = ("scoped", mgr.registry[id(inputs)].owner_id)
    assert mgr._group_chunk_parameters[owner_key] == (8, 1)
    assert responses["outputs"][0].chunk_length == 8
    assert responses["inputs"][0].arr == {}
    assert responses["inputs"][0].chunk_length == 8
    assert peer_responses == []
    assert id(peer) in mgr._queued_allocations["scoped"]
    assert mgr.registry[id(peer)].allocations == {}

    mgr.allocate_queue(peer)

    assert peer_responses[0].chunk_length == 1
    peer_key = ("scoped", mgr.registry[id(peer)].owner_id)
    assert mgr._group_chunk_parameters[peer_key] == (1, 1)
    assert mgr._group_chunk_parameters[owner_key] == (8, 1)
    assert "scoped" not in mgr._queued_allocations


@pytest.mark.parametrize("memory_clients", [3], indirect=True)
def test_peer_placeholder_does_not_size_owner_partition(
    mgr, memory_clients
):
    """An owner with nothing queued keeps its partition when a peer's
    one-run request is the only thing in the group queue."""
    inputs, outputs, peer = memory_clients
    responses = _register_owner(mgr, inputs, outputs, "placeholder")
    mgr.register(
        peer,
        allocation_ready_hook=peer.notice_allocation,
        invalidate_cache_hook=peer.notice_invalidate,
        stream_group="placeholder",
    )
    mgr.queue_request(inputs, _request(8))
    mgr.queue_request(outputs, _request(8))
    mgr.allocate_queue(outputs)
    owner_key = ("placeholder", mgr.registry[id(inputs)].owner_id)
    assert mgr._group_chunk_parameters[owner_key] == (8, 1)

    mgr.queue_request(peer, _request(1))
    mgr.allocate_queue(inputs)

    assert mgr._group_chunk_parameters[owner_key] == (8, 1)
    assert responses["inputs"][-1].chunk_length == 8
    assert responses["outputs"][-1].chunk_length == 8
    assert id(peer) in mgr._queued_allocations["placeholder"]

    mgr.queue_request(outputs, _request(8))
    mgr.allocate_queue(outputs)

    assert responses["outputs"][-1].arr["buf"].shape == (2, 2, 8)
    assert responses["outputs"][-1].chunk_length == 8


@pytest.mark.parametrize("memory_clients", [2], indirect=True)
def test_cached_partition_not_covering_batch_is_recomputed(
    mgr, memory_clients
):
    """A stored partition holding fewer runs than the batch is replaced."""
    inputs, outputs = memory_clients
    responses = _register_owner(mgr, inputs, outputs, "coverage")
    mgr.queue_request(inputs, _request(8))
    mgr.allocate_queue(inputs)
    owner_key = ("coverage", mgr.registry[id(inputs)].owner_id)
    mgr._group_chunk_parameters[owner_key] = (1, 1)

    mgr.queue_request(outputs, _request(8))
    mgr.allocate_queue(outputs)

    assert mgr._group_chunk_parameters[owner_key] == (8, 1)
    assert responses["outputs"][-1].arr["buf"].shape == (2, 2, 8)
    assert responses["outputs"][-1].chunk_length == 8


def test_partition_covers():
    """A partition covers a batch when its even chunks hold every run."""
    assert partition_covers((8, 1), 8)
    assert partition_covers((34, 3), 100)
    assert not partition_covers((1, 1), 8)
    assert not partition_covers((33, 3), 100)
    assert not partition_covers((0, 1), 8)


def test_release_instance_drops_owner_partition(mgr, memory_client):
    """Deregistering an owner discards its cached run partition."""
    instance = memory_client
    mgr.register(
        instance,
        allocation_ready_hook=instance.notice_allocation,
        invalidate_cache_hook=instance.notice_invalidate,
        stream_group="dropped",
    )
    mgr.queue_request(instance, _request(8))
    mgr.allocate_queue(instance)
    settings = mgr.registry[id(instance)]
    owner_key = ("dropped", settings.owner_id)
    assert owner_key in mgr._group_chunk_parameters

    mgr.release_instance(id(instance), settings)

    assert owner_key not in mgr._group_chunk_parameters
