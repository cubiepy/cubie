"""Tests for the BaseODE management interface via SymbolicODE."""

import numpy as np
import pytest

from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.ODEData import FixedParameterValues
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
    twin.set_default_parameters({"c0": 3.0})
    assert tiny_system.parameters.values_dict["c0"] == 1.0
    assert twin.parameters.values_dict["c0"] == 3.0


class TestUpdate:
    """Cover the BaseODE.update dispatch branches."""

    def test_update_none_dict_with_kwargs(self, tiny_system):
        """A None dict plus kwargs updates recognised settings."""
        recognised = tiny_system.update(None, operation_ordering="kahn")
        assert recognised == {"operation_ordering"}

    def test_update_empty_returns_empty_set(self, tiny_system):
        """An empty update returns an empty set without side effects."""
        assert tiny_system.update({}) == set()

    def test_update_unrecognised_key_raises(self, tiny_system):
        """An unrecognised key raises KeyError when not silent."""
        with pytest.raises(KeyError, match="Unrecognized parameters"):
            tiny_system.update({"not_a_key": 1.0})


class TestSweptParameters:
    """Cover swept parameters and fixed values set through update."""

    def test_swept_names_set_the_parameters_array_rows(self, tiny_system):
        """Swept names become the parameters array rows, in order."""
        tiny_system.update(swept_parameters=("c0", "k"))
        assert tiny_system.swept_parameters == ("c0", "k")
        assert tiny_system.sizes.swept_parameters == 2
        assert tiny_system.indices.parameter_names == ["c0", "k"]

    def test_fixed_values_are_compiled_in(self, tiny_system):
        """Fixed values are the values the system compiles in."""
        tiny_system.update(
            swept_parameters=("k",),
            fixed_values=FixedParameterValues({"c0": 4.0}),
        )
        assert tiny_system.fixed_parameter_values == {"c0": 4.0}
        assert tiny_system.parameters.values_dict["c0"] == 1.0

    def test_unchanged_parameters_keep_the_build(self, tiny_system):
        """Repeating the swept names and fixed values keeps the build."""
        tiny_system.update(
            swept_parameters=("k",),
            fixed_values=FixedParameterValues({"c0": 4.0}),
        )
        tiny_system.dxdt_fn
        tiny_system.update(
            swept_parameters=("k",),
            fixed_values=FixedParameterValues({"c0": 4.0}),
        )
        assert tiny_system.cache_valid is True

    def test_fixed_values_change_the_identity(self, tiny_system):
        """Systems fixed at different values hash differently."""
        tiny_system.set_fixed_values({"c0": 4.0})
        first = tiny_system.config_hash
        tiny_system.set_fixed_values({"c0": 5.0})
        assert tiny_system.config_hash != first

    def test_fixed_value_of_a_swept_name_raises(self, tiny_system):
        """A fixed value for a swept parameter raises."""
        with pytest.raises(ValueError, match="not swept"):
            tiny_system.update(
                swept_parameters=("k",),
                fixed_values=FixedParameterValues({"k": 4.0}),
            )


class TestSetSweptParameters:
    """Cover BaseODE.set_swept_parameters."""

    def test_unnamed_parameters_compile_in_at_their_defaults(
        self, tiny_system
    ):
        """Parameters not swept and not fixed compile in at defaults."""
        tiny_system.set_swept_parameters(["k"])
        assert tiny_system.swept_parameters == ("k",)
        assert tiny_system.fixed_parameter_values == {"c0": 1.0}

    def test_sweep_keeps_the_other_fixed_values(self, tiny_system):
        """Changing the sweep leaves other parameters' fixed values."""
        tiny_system.set_fixed_values({"c0": 4.0})
        tiny_system.set_swept_parameters(["k"])
        assert tiny_system.fixed_values == {"c0": 4.0}

    def test_sweeping_a_fixed_parameter_discards_its_value(
        self, tiny_system
    ):
        """A parameter swept after being fixed has no fixed value."""
        tiny_system.set_fixed_values({"c0": 4.0, "k": 2.0})
        tiny_system.set_swept_parameters(["k"])
        assert tiny_system.fixed_values == {"c0": 4.0}

    def test_unchanged_sweep_keeps_the_build(self, tiny_system):
        """Repeating the current sweep keeps the build."""
        tiny_system.set_swept_parameters(["k"])
        tiny_system.dxdt_fn
        tiny_system.set_swept_parameters(["k"])
        assert tiny_system.cache_valid is True

    def test_unknown_name_raises(self, tiny_system):
        """A swept name outside the parameters raises KeyError."""
        with pytest.raises(KeyError, match="not_a_key"):
            tiny_system.set_swept_parameters(["not_a_key"])

    def test_swept_values_follow_the_sweep_in_row_order(self, tiny_system):
        """Swept values hold each swept default in row order."""
        tiny_system.set_swept_parameters(["c0", "k"])
        assert list(tiny_system.swept_values.values_dict.items()) == [
            ("c0", 1.0),
            ("k", 0.5),
        ]
        tiny_system.set_swept_parameters(["k"])
        assert list(tiny_system.swept_values.values_dict) == ["k"]

    def test_swept_values_are_reused_until_the_settings_change(
        self, tiny_system
    ):
        """Unchanged settings return the same swept values object."""
        first = tiny_system.swept_values
        assert tiny_system.swept_values is first


class TestSetFixedValues:
    """Cover BaseODE.set_fixed_values and set_batch_parameters."""

    def test_fixed_value_compiles_in_and_keeps_the_default(
        self, tiny_system
    ):
        """A fixed value is compiled in and the default is unchanged."""
        tiny_system.set_swept_parameters(["k"])
        tiny_system.set_fixed_values({"c0": 4.0})
        assert tiny_system.swept_parameters == ("k",)
        assert tiny_system.fixed_parameter_values == {"c0": 4.0}
        assert tiny_system.parameters.values_dict["c0"] == 1.0

    def test_empty_values_return_to_the_defaults(self, tiny_system):
        """An empty mapping compiles every parameter in at its default."""
        tiny_system.set_fixed_values({"c0": 4.0})
        tiny_system.set_fixed_values({})
        assert tiny_system.fixed_values == {}
        assert tiny_system.fixed_parameter_values == {"c0": 1.0, "k": 0.5}

    def test_unchanged_values_keep_the_build(self, tiny_system):
        """Repeating the current fixed values keeps the build."""
        tiny_system.set_fixed_values({"c0": 4.0})
        tiny_system.dxdt_fn
        tiny_system.set_fixed_values({"c0": 4.0})
        assert tiny_system.cache_valid is True

    def test_unknown_name_raises(self, tiny_system):
        """A fixed name outside the parameters raises KeyError."""
        with pytest.raises(KeyError, match="not_a_key"):
            tiny_system.set_fixed_values({"not_a_key": 1.0})

    def test_swept_name_raises(self, tiny_system):
        """A fixed value for a swept parameter raises ValueError."""
        tiny_system.set_swept_parameters(["k"])
        with pytest.raises(ValueError, match="not swept"):
            tiny_system.set_fixed_values({"k": 1.0})

    def test_batch_parameters_set_both(self, tiny_system):
        """set_batch_parameters sets the sweep and the fixed values."""
        tiny_system.set_fixed_values({"k": 2.0})
        tiny_system.set_batch_parameters(["k"], {"c0": 4.0})
        assert tiny_system.swept_parameters == ("k",)
        assert tiny_system.fixed_values == {"c0": 4.0}

    def test_default_change_leaves_a_fixed_value(self, tiny_system):
        """A new default does not replace a parameter's fixed value."""
        tiny_system.set_fixed_values({"c0": 4.0})
        tiny_system.set_default_parameters({"c0": 7.0})
        assert tiny_system.fixed_parameter_values["c0"] == 4.0


class TestSetDefaultParameters:
    """Cover BaseODE.set_default_parameters."""

    def test_unfixed_parameter_compiles_in_its_new_default(
        self, tiny_system
    ):
        """A parameter without a fixed value compiles in its default."""
        tiny_system.set_default_parameters({"c0": 7.0})
        assert tiny_system.parameters.values_dict["c0"] == 7.0
        assert tiny_system.fixed_parameter_values["c0"] == 7.0

    def test_swept_parameter_stays_swept(self, tiny_system):
        """A swept parameter's new default leaves it swept."""
        tiny_system.set_swept_parameters(["k"])
        tiny_system.set_default_parameters({"k": 3.0})
        assert tiny_system.swept_parameters == ("k",)
        assert tiny_system.parameters.values_dict["k"] == 3.0

    def test_unknown_name_raises(self, tiny_system):
        """A name outside the system's parameters raises KeyError."""
        with pytest.raises(KeyError, match="not_a_key"):
            tiny_system.set_default_parameters({"not_a_key": 1.0})


class TestGetSolverHelper:
    """Cover the base-class solver-helper contract.

    ``SymbolicODE`` overrides ``get_solver_helper`` with generated
    helpers; the abstract base provides none.
    """

    def test_get_solver_helper_raises_on_base(self, system):
        """Base-class get_solver_helper raises for any request."""
        with pytest.raises(NotImplementedError, match="symbolic"):
            BaseODE.get_solver_helper(system, "linear_operator")
