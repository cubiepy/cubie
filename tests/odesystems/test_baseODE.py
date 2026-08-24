"""Tests for the BaseODE management interface via SymbolicODE."""

import numpy as np
import pytest

from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


@pytest.fixture
def tiny_system():
    """Return a minimal symbolic system with one folded value."""
    system = create_ODE_system(
        dxdt=["dx = -k * x + c0"],
        states={"x": 1.0},
        parameters={"k": 0.5, "c0": 1.0},
        observables=[],
        precision=np.float32,
        strict=True,
        name="tiny_base_ode",
    )
    system.set_categories(
        parameters={"k": 0.5}, constants={"c0": 1.0}
    )
    return system


class TestUpdate:
    """Cover the BaseODE.update dispatch branches."""

    def test_update_none_dict_with_kwargs(self, tiny_system):
        """A None dict plus kwargs updates recognised constants."""
        recognised = tiny_system.update(None, c0=2.0)
        assert recognised == {"c0"}

    def test_update_empty_returns_empty_set(self, tiny_system):
        """An empty update returns an empty set without side effects."""
        assert tiny_system.update({}) == set()

    def test_update_unrecognised_key_raises(self, tiny_system):
        """An unrecognised key raises KeyError when not silent."""
        with pytest.raises(KeyError, match="Unrecognized parameters"):
            tiny_system.update({"not_a_key": 1.0})


class TestSetConstants:
    """Cover the BaseODE.set_constants branches directly.

    ``SymbolicODE`` overrides ``set_constants``, so the base-class
    branches are exercised against ``BaseODE`` directly.
    """

    def test_none_dict_returns_empty_set(self, tiny_system):
        """A None dict with no kwargs returns an empty set."""
        assert BaseODE.set_constants(tiny_system, None) == set()

    def test_kwargs_only_updates_constant(self, tiny_system):
        """Base-class set_constants applies kwargs-only updates."""
        recognised = BaseODE.set_constants(tiny_system, None, c0=7.0)
        assert recognised == {"c0"}
        assert tiny_system.constants.values_dict["c0"] == 7.0

    def test_mixed_recognised_and_unknown_raises(self, tiny_system):
        """A recognised key beside an unknown key raises KeyError."""
        with pytest.raises(KeyError, match="Unrecognized parameters"):
            BaseODE.set_constants(
                tiny_system, {"c0": 1.0, "not_a_key": 1.0}
            )


class TestNumConstants:
    """Cover the num_constants property."""

    def test_num_constants(self, tiny_system):
        """num_constants reports the declared constant count."""
        assert tiny_system.num_constants == 1


class TestGetSolverHelper:
    """Cover the base-class solver-helper contract.

    ``SymbolicODE`` overrides ``get_solver_helper`` with generated
    helpers; the abstract base provides none.
    """

    def test_get_solver_helper_raises_on_base(self, system):
        """Base-class get_solver_helper raises for any request."""
        with pytest.raises(NotImplementedError, match="symbolic"):
            BaseODE.get_solver_helper(system, "linear_operator")
