"""Tests for measured auto buffer-placement heuristics."""

import pytest

from cubie.buffer_registry import buffer_registry
from cubie.integrators.memory_heuristics import (
    DEFAULT_ARCH,
    THRESHOLDS_BY_ARCH,
    auto_memory_locations,
    resolve_thresholds,
)
from tests._utils import (
    ALGORITHM_CHAIN_SETS,
    LARGE_BACKWARDS_EULER,
    LARGE_DIRK,
    LARGE_FIRK,
    LARGE_TSIT5,
)


def loop_and_algo_shared_buffers(solver):
    """Return non-child shared buffer names for the loop and step."""
    run = solver.kernel.single_integrator
    names = set()
    for parent in (run._loop, run._algo_step):
        group = buffer_registry._groups.get(parent)
        if group is None:
            continue
        for name in buffer_registry.relocatable_buffer_names(parent):
            entry = group.entries[name]
            if entry.location == "shared" and entry.size > 0:
                names.add(name)
    return names


def test_small_default_system_keeps_all_buffers_local(solver):
    """The default euler chain stays all-local below the spill gate."""
    assert loop_and_algo_shared_buffers(solver) == set()


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        ALGORITHM_CHAIN_SETS["erk"],
        ALGORITHM_CHAIN_SETS["backwards_euler"],
    ],
    indirect=True,
)
def test_small_system_keeps_all_buffers_local(solver):
    """No placement fires below the spill gate: small systems stay
    all-local, where shared placements measured neutral-to-slower."""
    assert loop_and_algo_shared_buffers(solver) == set()


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        LARGE_TSIT5,
        LARGE_DIRK,
    ],
    indirect=True,
)
def test_large_system_moves_state_pair_to_shared(solver):
    """Heavily spilled kernels with a sub-1-KiB state pair get the
    measured state/proposed_state shared placement."""
    assert loop_and_algo_shared_buffers(solver) == {
        "state",
        "proposed_state",
    }


@pytest.mark.parametrize(
    "solver_settings_override",
    [LARGE_BACKWARDS_EULER],
    indirect=True,
)
def test_deep_implicit_moves_state_pair_to_shared(solver):
    """Deeply spilled narrow-width implicit runs share the state pair."""
    assert loop_and_algo_shared_buffers(solver) == {
        "state",
        "proposed_state",
    }


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**LARGE_TSIT5, "state_location": "local"}],
    indirect=True,
)
def test_user_location_key_blocks_whole_group(solver):
    """Pinning one key of a placement group keeps the whole group
    local: partially relocated groups were never benchmarked."""
    assert loop_and_algo_shared_buffers(solver) == set()


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**LARGE_TSIT5, "auto_memory": False}],
    indirect=True,
)
def test_auto_memory_false_keeps_all_buffers_local(solver):
    """auto_memory=False disables every heuristic placement."""
    assert loop_and_algo_shared_buffers(solver) == set()


@pytest.mark.parametrize(
    "solver_settings_override",
    [LARGE_FIRK],
    indirect=True,
)
def test_stacked_width_solve_stays_local(solver):
    """Width-coupled implicit solves get no placement."""
    assert loop_and_algo_shared_buffers(solver) == set()


def test_resolver_skips_unmeasured_families(solver):
    """The resolver returns nothing for the default euler config and
    respects explicitly supplied keys."""
    run = solver.kernel.single_integrator
    assert auto_memory_locations(run) == {}


def test_unknown_architecture_falls_back_to_default():
    """Cards without a calibrated entry receive the default
    architecture's thresholds."""
    default = THRESHOLDS_BY_ARCH[DEFAULT_ARCH]
    assert resolve_thresholds("0.0") == default
    assert resolve_thresholds(DEFAULT_ARCH) == default
