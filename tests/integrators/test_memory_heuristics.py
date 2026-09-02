"""Tests for measured auto buffer-placement heuristics."""

import pytest

from cubie.buffer_registry import buffer_registry
from cubie.integrators.memory_heuristics import (
    DEFAULT_ARCH,
    STATE_PAIR_KEYS,
    THRESHOLDS_BY_ARCH,
    DeclaredSizes,
    auto_memory_locations,
    declared_sizes,
    placement_candidates,
    resolve_thresholds,
)
from tests._utils import (
    ALGORITHM_CHAIN_SETS,
    LARGE_DIRK,
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


def _implicit_sizes(**overrides):
    """Return implicit-step sizes for a narrow single-stage solve."""
    fields = dict(
        itemsize=4,
        is_implicit=True,
        is_linear=False,
        stacked_width=False,
        stage_count=1,
        footprint_bytes=THRESHOLDS_BY_ARCH[DEFAULT_ARCH].implicit_deep_bytes,
        state_pair_bytes=800,
        work_group_bytes=0,
        work_location_keys=(),
    )
    fields.update(overrides)
    return DeclaredSizes(**fields)


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


def test_deep_implicit_moves_state_pair_to_shared():
    """Deeply spilled narrow-width implicit runs share the state pair."""
    thresholds = THRESHOLDS_BY_ARCH[DEFAULT_ARCH]
    assert placement_candidates(_implicit_sizes(), thresholds) == [
        STATE_PAIR_KEYS
    ]


def test_implicit_below_deep_gate_keeps_state_pair_local():
    """An implicit footprint under the deep gate gets no placement."""
    thresholds = THRESHOLDS_BY_ARCH[DEFAULT_ARCH]
    sizes = _implicit_sizes(
        footprint_bytes=thresholds.implicit_deep_bytes - 1
    )
    assert placement_candidates(sizes, thresholds) == []


def test_stacked_width_solve_stays_local():
    """Width-coupled implicit solves get no placement."""
    thresholds = THRESHOLDS_BY_ARCH[DEFAULT_ARCH]
    sizes = _implicit_sizes(stacked_width=True, stage_count=3)
    assert placement_candidates(sizes, thresholds) == []


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["firk"]],
    indirect=True,
)
def test_firk_run_declares_a_stacked_width(single_integrator_run):
    """A FIRK run's declared sizes report the coupled solve width."""
    sizes = declared_sizes(single_integrator_run)
    tableau = single_integrator_run._algo_step.tableau
    assert sizes.is_implicit
    assert sizes.stacked_width
    assert sizes.stage_count == tableau.stage_count


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
