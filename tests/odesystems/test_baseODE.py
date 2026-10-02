"""Tests for the BaseODE management interface via SymbolicODE."""

import numpy as np
import pytest

from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


@pytest.fixture
def tiny_system():
    """Return a minimal symbolic system with two parameters (no compile)."""
    return create_ODE_system(
        dxdt=["dx = -k * x + c0"],
        states={"x": 1.0},
        parameters={"k": 0.5, "c0": 1.0},
        observables=[],
        precision=np.float32,
        strict=True,
        name="tiny_base_ode",
    )


def test_copy_is_an_independent_unbuilt_system(tiny_system):
    """A copy hashes the same, owns its values and holds no build."""
    tiny_system.dxdt_fn
    twin = tiny_system.copy()
    assert twin is not tiny_system
    assert twin.config_hash == tiny_system.config_hash
    assert twin.cache_valid is False
    assert tiny_system.cache_valid is True
    twin.update(c0=3.0)
    assert tiny_system.parameters.values_dict["c0"] == 1.0
    assert twin.parameters.values_dict["c0"] == 3.0


class TestUpdate:
    """Cover the BaseODE.update dispatch branches."""

    def test_update_none_dict_with_kwargs(self, tiny_system):
        """A None dict plus kwargs updates recognised parameters."""
        recognised = tiny_system.update(None, c0=2.0)
        assert recognised == {"c0"}

    def test_update_empty_returns_empty_set(self, tiny_system):
        """An empty update returns an empty set without side effects."""
        assert tiny_system.update({}) == set()

    def test_update_unrecognised_key_raises(self, tiny_system):
        """An unrecognised key raises KeyError when not silent."""
        with pytest.raises(KeyError, match="Unrecognized parameters"):
            tiny_system.update({"not_a_key": 1.0})

    def test_update_does_not_take_swept_names(self, tiny_system):
        """Swept names are set through set_swept_parameters."""
        with pytest.raises(KeyError, match="swept_parameters"):
            tiny_system.update(swept_parameters=["k"])


class TestSetParameterValues:
    """Cover BaseODE.set_parameter_values."""

    def test_set_value_is_compiled_in(self, tiny_system):
        """A parameter's new value is compiled in."""
        recognised = tiny_system.set_parameter_values({"c0": 7.0})
        assert recognised == {"c0"}
        assert tiny_system.parameters.values_dict["c0"] == 7.0
        assert tiny_system.fixed_parameter_values["c0"] == 7.0

    def test_setting_a_swept_value_fixes_it(self, tiny_system):
        """A value set for a swept parameter compiles it in."""
        tiny_system.set_swept_parameters(["k", "c0"])
        tiny_system.set_parameter_values({"k": 3.0})
        assert tiny_system.swept_parameters == ("c0",)
        assert tiny_system.fixed_parameter_values == {"k": 3.0}

    def test_unknown_name_raises(self, tiny_system):
        """A name outside the system's parameters raises KeyError."""
        with pytest.raises(KeyError, match="not_a_key"):
            tiny_system.set_parameter_values({"not_a_key": 1.0})


class TestSetSweptParameters:
    """Cover BaseODE.set_swept_parameters."""

    def test_sweeps_names_in_order_and_fixes_the_rest(self, tiny_system):
        """The names are swept in the given order; the rest fix."""
        tiny_system.update(c0=2.0)
        changed = tiny_system.set_swept_parameters(["k", "c0"])
        assert changed is True
        assert tiny_system.swept_parameters == ("k", "c0")
        assert tiny_system.indices.parameter_names == ["k", "c0"]
        tiny_system.set_swept_parameters(["k"])
        assert tiny_system.fixed_parameter_values == {"c0": 2.0}

    def test_same_names_report_no_change(self, tiny_system):
        """Sweeping the swept names again changes nothing."""
        tiny_system.set_swept_parameters(["c0"])
        assert tiny_system.set_swept_parameters(["c0"]) is False

    def test_unknown_name_raises(self, tiny_system):
        """A name outside the system's parameters raises KeyError."""
        with pytest.raises(KeyError, match="not_a_key"):
            tiny_system.set_swept_parameters(["not_a_key"])


class TestBind:
    """Cover BaseODE.bind."""

    def test_sweeps_and_stores_values_together(self, tiny_system):
        """Swept names and values apply together; values are stored."""
        changed = tiny_system.bind(swept=["k"], values={"c0": 4.0})
        assert changed is True
        assert tiny_system.sizes.parameters == 1
        assert tiny_system.swept_parameters == ("k",)
        assert tiny_system.parameters.values_dict["c0"] == 4.0
        assert tiny_system.fixed_parameter_values == {"c0": 4.0}

    def test_same_state_reports_no_change(self, tiny_system):
        """Binding the current names and values changes nothing."""
        values = tiny_system.fixed_parameter_values
        assert tiny_system.bind(swept=(), values=values) is False

    def test_unknown_name_raises(self, tiny_system):
        """A name outside the system's parameters raises KeyError."""
        with pytest.raises(KeyError, match="nope"):
            tiny_system.bind(swept=["nope"])


class TestGetSolverHelper:
    """Cover the base-class solver-helper contract.

    ``SymbolicODE`` overrides ``get_solver_helper`` with generated
    helpers; the abstract base provides none.
    """

    def test_get_solver_helper_raises_on_base(self, system):
        """Base-class get_solver_helper raises for any request."""
        with pytest.raises(NotImplementedError, match="symbolic"):
            BaseODE.get_solver_helper(system, "linear_operator")
