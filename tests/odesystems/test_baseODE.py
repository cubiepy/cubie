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
    twin.set_default_parameters({"c0": 3.0})
    assert tiny_system.parameters.defaults["c0"] == 1.0
    assert twin.parameters.defaults["c0"] == 3.0


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
    """Cover the sweep and the values a batch compiles in."""

    def test_swept_names_set_the_parameters_array_rows(self, tiny_system):
        """Swept names become the parameters array rows, in order."""
        tiny_system.set_swept_parameters(("k", "c0"))
        assert tiny_system.swept_parameters == ("c0", "k")
        assert tiny_system.sizes.swept_parameters == 2
        assert tiny_system.layout.swept_names == ("c0", "k")

    def test_a_sweep_given_at_creation_reads_rows_in_system_order(self):
        """The device function reads the rows the solver fills."""
        system = create_ODE_system(
            dxdt=["dx = -k * x + c0"],
            states={"x": 1.0},
            parameters={"k": 0.5, "c0": 1.0},
            precision=np.float32,
            swept_parameters=("k", "c0"),
        )
        assert system.swept_parameters == ("c0", "k")
        assert system.layout.swept_names == ("c0", "k")

    def test_fixed_values_are_compiled_in(self, tiny_system):
        """Compile a specialisation's fixed values in without changing
        the defaults."""
        tiny_system.update(
            specialisation=tiny_system.specialise(("k",), {"c0": 4.0})
        )
        assert tiny_system.fixed_parameter_values == {"c0": 4.0}
        assert tiny_system.parameters.defaults["c0"] == 1.0

    def test_the_compiled_specialisation_keeps_the_build(self, tiny_system):
        """Keep the build when compiling the current specialisation."""
        tiny_system.dxdt_fn
        tiny_system.update(specialisation=tiny_system.specialise())
        assert tiny_system.cache_valid is True

    def test_unknown_swept_name_raises(self, tiny_system):
        """Raise KeyError for a swept name the system does not have."""
        with pytest.raises(KeyError, match="not_a_key"):
            tiny_system.specialise(("not_a_key",))


class TestSetDefaultParameters:
    """Cover BaseODE.set_default_parameters."""

    def test_fixed_parameter_compiles_in_its_new_default(self, tiny_system):
        """A fixed parameter's new default is compiled in."""
        tiny_system.set_default_parameters({"c0": 7.0})
        assert tiny_system.parameters.defaults["c0"] == 7.0
        assert tiny_system.fixed_parameter_values["c0"] == 7.0

    def test_swept_parameter_keeps_the_build(self, tiny_system):
        """A swept parameter's new default leaves it swept and rebuilds
        nothing."""
        tiny_system.set_swept_parameters(("k",))
        tiny_system.dxdt_fn
        tiny_system.set_default_parameters({"k": 3.0})
        assert tiny_system.swept_parameters == ("k",)
        assert tiny_system.parameters.defaults["k"] == 3.0
        assert tiny_system.cache_valid is True

    def test_new_defaults_drop_the_fixed_values(self, tiny_system):
        """Every fixed parameter compiles in at its default afterwards."""
        tiny_system.update(
            specialisation=tiny_system.specialise(("k",), {"c0": 4.0})
        )
        tiny_system.set_default_parameters({"k": 3.0})
        assert tiny_system.fixed_parameter_values == {"c0": 1.0}

    def test_unknown_name_raises(self, tiny_system):
        """A name outside the system's parameters raises KeyError."""
        with pytest.raises(KeyError, match="not_a_key"):
            tiny_system.set_default_parameters({"not_a_key": 1.0})


class TestSpecialise:
    """Cover the memoised specialisations and variable sets."""

    def test_a_repeated_request_returns_the_same_specialisation(
        self, tiny_system
    ):
        """Return the same object for the same request."""
        specialisation = tiny_system.specialise(("k",), {"c0": 2.0})
        assert tiny_system.specialise(("k",), {"c0": 2.0}) is specialisation

    def test_fixing_a_default_returns_the_default_specialisation(
        self, tiny_system
    ):
        """Return the default specialisation for a parameter fixed at
        its default."""
        specialisation = tiny_system.specialise(("k",))
        assert tiny_system.specialise(("k",), {"c0": 1.0}) is specialisation

    def test_specialising_leaves_the_compiled_system(self, tiny_system):
        """Return a specialisation without compiling it."""
        compiled = tiny_system.specialisation
        tiny_system.dxdt_fn
        tiny_system.specialise(("k",), {"c0": 2.0})
        assert tiny_system.specialisation is compiled
        assert tiny_system.cache_valid is True

    def test_a_specialisation_outlives_a_new_default(self, tiny_system):
        """Keep a specialisation keyed on its values through a default
        change."""
        specialisation = tiny_system.specialise(("k",), {"c0": 2.0})
        tiny_system.set_default_parameters({"c0": 3.0})
        assert tiny_system.specialise(("k",), {"c0": 2.0}) is specialisation

    def test_new_initial_values_give_a_new_state_set(self, tiny_system):
        """The state set is reused until an initial value changes."""
        states = tiny_system.states
        assert tiny_system.states is states
        tiny_system.set_initial_values({"x": 4.0})
        assert tiny_system.states.defaults == {"x": 4.0}


class TestSetInitialValues:
    """Cover BaseODE.set_initial_values."""

    def test_new_initial_values_keep_the_build(self, tiny_system):
        """Initial values are runtime data and rebuild nothing."""
        tiny_system.dxdt_fn
        hash_before = tiny_system.config_hash
        tiny_system.set_initial_values({"x": 4.0})
        assert tiny_system.states.defaults == {"x": 4.0}
        assert tiny_system.cache_valid is True
        assert tiny_system.config_hash == hash_before

    def test_unknown_name_raises(self, tiny_system):
        """A name outside the system's states raises KeyError."""
        with pytest.raises(KeyError, match="not_a_key"):
            tiny_system.set_initial_values({"not_a_key": 1.0})


class TestGetSolverHelper:
    """Cover the base-class solver-helper contract.

    ``SymbolicODE`` overrides ``get_solver_helper`` with generated
    helpers; the abstract base provides none.
    """

    def test_get_solver_helper_raises_on_base(self, system):
        """Base-class get_solver_helper raises for any request."""
        with pytest.raises(NotImplementedError, match="symbolic"):
            BaseODE.get_solver_helper(system, "linear_operator")
