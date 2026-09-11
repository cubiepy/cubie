"""Tests for cubie.integrators.algorithms.__init__."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.integrators.algorithms import (
    _ALGORITHM_REGISTRY,
    _TABLEAU_REGISTRY_BY_ALGORITHM,
    ExplicitEulerStep,
    BackwardsEulerStep,
    BackwardsEulerPCStep,
    CrankNicolsonStep,
    DIRKStep,
    FIRKStep,
    ERKStep,
    GenericRosenbrockWStep,
    ERK_TABLEAU_REGISTRY,
    DIRK_TABLEAU_REGISTRY,
    FIRK_TABLEAU_REGISTRY,
    ROSENBROCK_TABLEAUS,
    resolve_alias,
    resolve_supplied_tableau,
    get_algorithm_step,
)


# ── _ALGORITHM_REGISTRY (items 1-2) ──────────────────────── #

_EXPECTED_REGISTRY = {
    "euler": ExplicitEulerStep,
    "backwards_euler": BackwardsEulerStep,
    "backwards_euler_pc": BackwardsEulerPCStep,
    "crank_nicolson": CrankNicolsonStep,
    "dirk": DIRKStep,
    "firk": FIRKStep,
    "erk": ERKStep,
    "rosenbrock": GenericRosenbrockWStep,
}


@pytest.mark.parametrize(
    "key, expected_class",
    list(_EXPECTED_REGISTRY.items()),
)
def test_algorithm_registry_entries(key, expected_class):
    """_ALGORITHM_REGISTRY maps each key to the correct step class."""
    assert _ALGORITHM_REGISTRY[key] is expected_class


def test_algorithm_registry_size():
    """_ALGORITHM_REGISTRY contains exactly 8 entries."""
    assert len(_ALGORITHM_REGISTRY) == 8


# ── _TABLEAU_REGISTRY_BY_ALGORITHM (items 2-6) ───────────── #

def test_base_algorithms_have_none_tableau():
    """Base algorithm entries have None tableau."""
    for key in _ALGORITHM_REGISTRY:
        constructor, tableau = _TABLEAU_REGISTRY_BY_ALGORITHM[key]
        assert constructor is _ALGORITHM_REGISTRY[key]
        assert tableau is None


@pytest.mark.parametrize("alias", list(ERK_TABLEAU_REGISTRY.keys()))
def test_erk_aliases_registered(alias):
    """ERK tableau aliases map to ERKStep with correct tableau."""
    constructor, tableau = _TABLEAU_REGISTRY_BY_ALGORITHM[alias]
    assert constructor is ERKStep
    assert tableau is ERK_TABLEAU_REGISTRY[alias]


@pytest.mark.parametrize("alias", list(DIRK_TABLEAU_REGISTRY.keys()))
def test_dirk_aliases_registered(alias):
    """DIRK tableau aliases map to DIRKStep with correct tableau."""
    constructor, tableau = _TABLEAU_REGISTRY_BY_ALGORITHM[alias]
    assert constructor is DIRKStep
    assert tableau is DIRK_TABLEAU_REGISTRY[alias]


@pytest.mark.parametrize("alias", list(FIRK_TABLEAU_REGISTRY.keys()))
def test_firk_aliases_registered(alias):
    """FIRK tableau aliases map to FIRKStep with correct tableau."""
    constructor, tableau = _TABLEAU_REGISTRY_BY_ALGORITHM[alias]
    assert constructor is FIRKStep
    assert tableau is FIRK_TABLEAU_REGISTRY[alias]


@pytest.mark.parametrize("alias", list(ROSENBROCK_TABLEAUS.keys()))
def test_rosenbrock_aliases_registered(alias):
    """Rosenbrock aliases map to GenericRosenbrockWStep with tableau."""
    constructor, tableau = _TABLEAU_REGISTRY_BY_ALGORITHM[alias]
    assert constructor is GenericRosenbrockWStep
    assert tableau is ROSENBROCK_TABLEAUS[alias]


# ── resolve_alias (items 7-9) ────────────────────────────── #

def test_resolve_alias_lowercases():
    """resolve_alias lowercases the input before lookup."""
    cls, tab = resolve_alias("EULER")
    assert cls is ExplicitEulerStep
    assert tab is None


@pytest.mark.parametrize("alias", ["Kvaerno3", "KvAeRnO5"])
def test_resolve_kvaerno_alias_is_case_insensitive(alias):
    """Kvaerno aliases resolve independently of input case."""

    canonical_alias = alias.lower()
    cls, tab = resolve_alias(alias)
    assert cls is DIRKStep
    assert tab is DIRK_TABLEAU_REGISTRY[canonical_alias]


def test_resolve_alias_returns_tuple():
    """resolve_alias returns (class, tableau) for known alias."""
    cls, tab = resolve_alias("dormand-prince-54")
    assert cls is ERKStep
    assert tab is ERK_TABLEAU_REGISTRY["dormand-prince-54"]


def test_resolve_alias_unknown_raises():
    """resolve_alias raises KeyError for unknown alias."""
    with pytest.raises(KeyError):
        resolve_alias("nonexistent_algorithm")


# ── resolve_supplied_tableau (items 10-14) ────────────────── #

def test_resolve_supplied_tableau_erk():
    """resolve_supplied_tableau returns ERKStep for ERKTableau."""
    tab = ERK_TABLEAU_REGISTRY["dormand-prince-54"]
    cls, returned_tab = resolve_supplied_tableau(tab)
    assert cls is ERKStep
    assert returned_tab is tab


def test_resolve_supplied_tableau_dirk():
    """resolve_supplied_tableau returns DIRKStep for DIRKTableau."""
    tab = list(DIRK_TABLEAU_REGISTRY.values())[0]
    cls, returned_tab = resolve_supplied_tableau(tab)
    assert cls is DIRKStep
    assert returned_tab is tab


def test_resolve_supplied_tableau_firk():
    """resolve_supplied_tableau returns FIRKStep for FIRKTableau."""
    tab = list(FIRK_TABLEAU_REGISTRY.values())[0]
    cls, returned_tab = resolve_supplied_tableau(tab)
    assert cls is FIRKStep
    assert returned_tab is tab


def test_resolve_supplied_tableau_rosenbrock():
    """resolve_supplied_tableau returns GenericRosenbrockWStep."""
    tab = list(ROSENBROCK_TABLEAUS.values())[0]
    cls, returned_tab = resolve_supplied_tableau(tab)
    assert cls is GenericRosenbrockWStep
    assert returned_tab is tab


def test_resolve_supplied_tableau_unknown_raises():
    """resolve_supplied_tableau raises TypeError for unknown type."""
    with pytest.raises(TypeError, match="does not match known"):
        resolve_supplied_tableau("not_a_tableau")


# ── get_algorithm_step (items 15-25) ─────────────────────── #

def test_get_algorithm_step_missing_algorithm():
    """get_algorithm_step raises ValueError when algorithm key missing."""
    with pytest.raises(ValueError, match="must include 'algorithm'"):
        get_algorithm_step(np.float32, settings={})


def test_get_algorithm_step_unknown_string():
    """get_algorithm_step raises ValueError for unknown algorithm name."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        get_algorithm_step(
            np.float32, settings={"algorithm": "bogus_algo"}
        )


def test_get_algorithm_step_invalid_type():
    """get_algorithm_step raises TypeError for non-str non-tableau."""
    with pytest.raises(TypeError, match="Expected algorithm name"):
        get_algorithm_step(np.float32, settings={"algorithm": 42})


def test_get_algorithm_step_kwargs_override_settings():
    """kwargs override settings dict entries."""
    # If kwargs override works, algorithm from kwargs wins
    step = get_algorithm_step(
        np.float32,
        settings={"algorithm": "euler", "n_states": 3, "n_drivers": 0},
        algorithm="euler",
        n_states=5,
    )
    assert step.compile_settings.n_states == 5


def test_get_algorithm_step_string_returns_instance(step_object):
    """get_algorithm_step with string returns correct step type."""
    # step_object is built from fixture using 'euler' default
    assert step_object.compile_settings.precision == np.float32


def test_get_algorithm_step_tableau_injects_tableau():
    """get_algorithm_step injects tableau when resolved is not None."""
    tab = ERK_TABLEAU_REGISTRY["dormand-prince-54"]
    step = get_algorithm_step(
        np.float32,
        settings={
            "algorithm": tab,
            "n_states": 3,
            "n_drivers": 0,
        },
    )
    assert step.tableau is tab


def test_get_algorithm_step_no_tableau_for_base():
    """get_algorithm_step does not inject tableau for base algorithms."""
    step = get_algorithm_step(
        np.float32,
        settings={
            "algorithm": "euler",
            "n_states": 3,
            "n_drivers": 0,
        },
    )
    assert not hasattr(step, "tableau") or step.tableau is None


def test_get_algorithm_step_injects_precision():
    """get_algorithm_step injects precision into the step instance."""
    step = get_algorithm_step(
        np.float64,
        settings={
            "algorithm": "euler",
            "n_states": 3,
            "n_drivers": 0,
        },
    )
    assert step.compile_settings.precision == np.float64
