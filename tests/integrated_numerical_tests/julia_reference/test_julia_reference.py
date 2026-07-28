"""Golden-reference gate: cubie vs DifferentialEquations.jl.

Single-solve Float32 checks on the Lorenz ensemble (N=1024, rho in
[0, 21], t in [0, 1]): per algorithm, one fixed-step solve at a pinned
difficult dt and one adaptive solve under Julia-matched control at a
pinned tight tolerance. Cubie integrates the same bit-identical
Float32 inputs as the vendored DifferentialEquations.jl sweeps; final
states are judged against the Float64 golden reference (Vern9, tol
1e-13) and against the Julia implementation at the pin. At the pinned
dt a dropped order inflates the truncation error by ``1/dt`` per lost
power, so the single solve catches drift from the textbook scheme.
Protocol, pin selection, thresholds, and provenance: ``ne_gate.py``
and ``data/README.md``.
"""

import numpy as np
import pytest

from cubie.integrators.algorithms import algorithm_is_adaptive

from tests.integrated_numerical_tests.julia_reference.ne_gate import (
    ATOL_FIXED_NE,
    DT0_NE,
    DT_MIN_NE,
    DT_MAX_NE,
    RTOL_FIXED_NE,
    adaptive_pin,
    fixed_pin,
    julia_adaptive_finals,
    julia_fixed_finals,
    load_algorithms,
    load_controller_constants,
    load_reference,
    matched_controller_settings,
    point_matches_julia,
)

pytestmark = pytest.mark.nocudasim

ALGORITHMS = {row["cubie_alias"]: row for row in load_algorithms()}

_ADAPTIVE_CONSTANTS = load_controller_constants()


def _collect_pins():
    """Per-alias pinned dt and tolerance from the vendored data."""
    fixed_pins = {}
    adaptive_pins = {}
    with load_reference() as archive:
        golden = archive["golden_states"].astype(np.float64)
        scale = float(np.sqrt(np.mean(golden ** 2)))
        keys = set(archive.files)
        for alias, row in ALGORITHMS.items():
            fixed_pins[alias] = fixed_pin(archive, alias, golden, scale)
            if not algorithm_is_adaptive(alias):
                continue
            if "adaptive_{0}_tols".format(alias) not in keys:
                continue
            if matched_controller_settings(
                    _ADAPTIVE_CONSTANTS, alias, row["order"]) is None:
                continue
            adaptive_pins[alias] = adaptive_pin(
                archive, alias, golden, scale)
    return fixed_pins, adaptive_pins


FIXED_PINS, ADAPTIVE_PINS = _collect_pins()

# Shared solve pins for every gate run: the protocol saves only the
# final state of all three Lorenz states at t = 1.0, and leaves the
# inner Newton/Krylov tolerances unset so they derive from atol/rtol
# exactly as they do for a library user.
_GATE_BASE_OVERRIDE = {
    "system_type": "lorenz_julia",
    "duration": 1.0,
    "save_every": 1.0,
    "summarise_every": None,
    "sample_summaries_every": None,
    "output_types": ["state"],
    "saved_state_indices": [0, 1, 2],
    "saved_observable_indices": [],
    "summarised_state_indices": [],
    "summarised_observable_indices": [],
    "blocksize": 64,
    "krylov_atol": None,
    "krylov_rtol": None,
    "newton_atol": None,
    "newton_rtol": None,
}


def _fixed_override(alias):
    """Solver settings for one algorithm's pinned fixed-step solve."""
    override = dict(_GATE_BASE_OVERRIDE)
    override.update(
        algorithm=alias,
        step_controller="fixed",
        dt=FIXED_PINS[alias],
        atol=ATOL_FIXED_NE,
        rtol=RTOL_FIXED_NE,
    )
    return override


def _adaptive_override(alias, matched):
    """Solver settings for one algorithm's pinned adaptive solve."""
    override = dict(_GATE_BASE_OVERRIDE)
    override.update(
        algorithm=alias,
        dt=DT0_NE,
        dt_min=DT_MIN_NE,
        dt_max=DT_MAX_NE,
        atol=ADAPTIVE_PINS[alias],
        rtol=ADAPTIVE_PINS[alias],
    )
    override.update(matched)
    return override


# Aliases that resolve to a tableau-free step class or to their
# family's default tableau. Every other alias carries
# ``specific_algos``, the same mark as ``SPECIFIC_ALGORITHM_COMBOS``
# in ``tests/_utils.py``: each alias keys its own session fixture
# chain and its own compiled kernel, so both CI legs deselect the
# per-tableau sweep and run the gate on the defaults only; the full
# sweep runs on demand without the marker filter.
DEFAULT_TABLEAU_ALIASES = frozenset({
    "euler",
    "backwards_euler",
    "crank_nicolson",
    "dormand-prince-54",
    "ros3p",
})

FIXED_PARAMS = [
    pytest.param(
        _fixed_override(alias),
        id=alias,
        marks=()
        if alias in DEFAULT_TABLEAU_ALIASES
        else pytest.mark.specific_algos,
    )
    for alias in ALGORITHMS
]

ADAPTIVE_PARAMS = [
    pytest.param(
        _adaptive_override(
            alias,
            matched_controller_settings(
                _ADAPTIVE_CONSTANTS, alias, ALGORITHMS[alias]["order"]),
        ),
        id=alias,
        marks=()
        if alias in DEFAULT_TABLEAU_ALIASES
        else pytest.mark.specific_algos,
    )
    for alias in ADAPTIVE_PINS
]


@pytest.mark.parametrize(
    "solver_settings_override", FIXED_PARAMS, indirect=True)
def test_fixed_step_matches_julia(gate_final, julia_reference,
                                  golden_ensemble):
    """Pinned fixed-step solve agrees with the Julia side."""
    alias, cubie_final = gate_final
    _, golden_states, _ = golden_ensemble
    pin = FIXED_PINS[alias]
    julia_final = julia_fixed_finals(julia_reference, alias)[pin]

    ok, report = point_matches_julia(
        cubie_final, julia_final, golden_states)

    assert ok, (
        "{0} (order {1}, {2}) deviates from Julia at dt={3:g}: "
        "{4}".format(
            alias, ALGORITHMS[alias]["order"],
            ALGORITHMS[alias]["julia_expr"], pin, report))


@pytest.mark.parametrize(
    "solver_settings_override", ADAPTIVE_PARAMS, indirect=True)
def test_adaptive_matched_controller_tracks_julia(
        gate_final, julia_reference, golden_ensemble):
    """Pinned adaptive solve under Julia-matched control tracks Julia."""
    alias, cubie_final = gate_final
    _, golden_states, _ = golden_ensemble
    pin = ADAPTIVE_PINS[alias]
    julia_final = julia_adaptive_finals(julia_reference, alias)[pin]

    ok, report = point_matches_julia(
        cubie_final, julia_final, golden_states)

    assert ok, (
        "{0} diverges from Julia under matched control at tol={1:g}: "
        "{2}".format(alias, pin, report))
