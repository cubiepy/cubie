"""Lightweight checks for the explicit Runge--Kutta tableau registry."""

import numpy as np
import pytest

from cubie.integrators.algorithms.generic_erk import (
    ERKStep,
)
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DEFAULT_ERK_TABLEAU,
    ERK_TABLEAU_REGISTRY,
)
EXPECTED_TABLEAU_NAMES = (
    "heun-21",
    "ralston-33",
    "bogacki-shampine-32",
    "dormand-prince-54",
    "classical-rk4",
    "cash-karp-54",
    "fehlberg-45",
)


@pytest.fixture(scope="module")
def erk_registry():
    """Expose the ERK tableau registry for tests."""

    return ERK_TABLEAU_REGISTRY


@pytest.fixture(scope="module")
def expected_tableau_names():
    """Return tableau identifiers expected in the registry."""

    return EXPECTED_TABLEAU_NAMES


@pytest.fixture()
def erk_step_settings_override(request):
    """Optional overrides for ERK step constructor settings."""

    return getattr(request, "param", None)


@pytest.fixture()
def erk_step_settings(erk_step_settings_override):
    """Return baseline ERK step constructor settings merged with overrides."""

    settings = {
        "precision": np.float64,
        "n_states": 2,
    }
    if erk_step_settings_override:
        settings.update(erk_step_settings_override)
    return settings


def test_registry_contains_expected_tableaus(
    erk_registry, expected_tableau_names
):
    """Named tableaus should be available through the registry."""

    for name in expected_tableau_names:
        assert name in erk_registry


def test_default_tableau_is_registered(erk_registry):
    """The default tableau should be available via the registry mapping."""

    assert erk_registry["dormand-prince-54"] is DEFAULT_ERK_TABLEAU


@pytest.mark.parametrize(
    "erk_step_settings_override",
    [{"n_states": 3}],
    indirect=True,
)
def test_step_accepts_registered_tableau(erk_step_settings, erk_registry):
    """Any registered tableau should configure the ERK step."""

    tableau = erk_registry["cash-karp-54"]
    step = ERKStep(tableau=tableau, **erk_step_settings)
    assert step.tableau is tableau
    assert step.order == tableau.order
