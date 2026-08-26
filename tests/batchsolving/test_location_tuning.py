"""Tests for measured per-buffer placement tuning."""

import pickle

import pytest

from cubie import Solver
from cubie.batchsolving.location_tuning import (
    _tuning_dir,
    candidate_buffers,
)
from cubie.buffer_registry import buffer_registry
from tests._utils import ALGORITHM_CHAIN_SETS


def registered_buffers(solver):
    """Yield (parent, name, entry) for every registered non-rollup buffer."""
    for parent, group in buffer_registry._groups.items():
        for name in group.relocatable_names():
            yield parent, name, group.entries[name]


def test_symbolic_system_pickles_with_current_values(system):
    """A system round-trips through pickle with its identity and values."""
    rebuilt = pickle.loads(pickle.dumps(system))
    assert rebuilt.config_hash == system.config_hash
    assert rebuilt.parameters.as_float_dict == system.parameters.as_float_dict
    assert (
        rebuilt.initial_values.as_float_dict
        == system.initial_values.as_float_dict
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["dirk"], ALGORITHM_CHAIN_SETS["firk"]],
    indirect=True,
)
def test_every_registered_buffer_has_a_location_setting(solver):
    """Each registered buffer's owner exposes a ``{name}_location``."""
    for parent, name, _ in registered_buffers(solver):
        assert hasattr(parent.compile_settings, f"{name}_location")


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_CHAIN_SETS["dirk"]],
    indirect=True,
)
def test_candidate_buffers_are_nonempty_local_settings(solver):
    """Candidates are the nonempty local buffers with a location setting."""
    candidates = candidate_buffers(solver)
    names = {candidate.name for candidate in candidates}
    expected = {
        name
        for parent, name, entry in registered_buffers(solver)
        if entry.size > 0 and entry.location == "local"
    }
    assert names == expected
    for candidate in candidates:
        assert candidate.key == f"{candidate.name}_location"
        assert candidate.bytes_per_run == candidate.elements * candidate.itemsize


def test_construction_record_captures_constructor_inputs(solver):
    """The solver records the system and kwargs it was built from."""
    record = solver._construction_record
    assert record["system"] is solver.kernel.system
    assert (
        record["kwargs"]["algorithm"]
        == solver.kernel.single_integrator.compile_settings.algorithm
    )


def test_auto_memory_rejects_unknown_mode(system):
    with pytest.raises(ValueError):
        Solver(system, auto_memory="fast")


@pytest.mark.sim_only
def test_tune_locations_unavailable_under_simulator(system):
    solver = Solver(system, auto_memory="tune")
    with pytest.raises(RuntimeError):
        solver.tune_locations(
            system.initial_values.as_float_dict,
            system.parameters.as_float_dict,
            duration=0.01,
        )


@pytest.mark.nocudasim
@pytest.mark.slow
def test_tune_locations_applies_and_persists(system):
    """Tuning applies the measured placement and reuses it from disk."""
    initial_values = system.initial_values.as_float_dict
    parameters = system.parameters.as_float_dict
    solver = Solver(
        system, algorithm="dirk", auto_memory="tune", dt=1e-3,
        step_controller="fixed", duration=0.01,
    )
    with pytest.warns(UserWarning, match="waves"):
        result = solver.tune_locations(
            initial_values, parameters, duration=0.01, workers=2, force=True
        )
    assert result.cached is False
    assert result.baseline.regs > 0
    assert len(result.trials) >= len(
        [t for t in result.trials if t.local_delta is not None and t.local_delta < 0]
    )
    for trial in result.trials:
        assert trial.resources is not None
        if trial.local_delta is not None and trial.local_delta < 0:
            assert trial.ratio is not None
    assert set(result.chosen) == set(solver.tuned_placement)
    for key in result.chosen:
        name = key[: -len("_location")]
        entries = {n: e for _, n, e in registered_buffers(solver)}
        assert entries[name].location == "shared"
    assert any(_tuning_dir().iterdir())

    reused = Solver(
        system, algorithm="dirk", auto_memory="tune", dt=1e-3,
        step_controller="fixed", duration=0.01,
    )
    outcome = reused.solve(initial_values, parameters, duration=0.01)
    assert reused.tuned_placement == result.chosen
    assert int(outcome.status_codes.max()) == 0
