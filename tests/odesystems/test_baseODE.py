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
    assert tiny_system.parameters.values_dict["c0"] == 1.0
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
    """Cover swept and fixed parameters set through update."""

    def test_swept_names_set_the_parameters_array_rows(self, tiny_system):
        """Swept names become the parameters array rows, in order."""
        tiny_system.update(
            swept_parameters=("c0", "k"), fixed_parameters=()
        )
        assert tiny_system.swept_parameters == ("c0", "k")
        assert tiny_system.sizes.swept_parameters == 2
        assert tiny_system.layout.swept_names == ("c0", "k")

    def test_an_iterator_of_names_sets_the_sweep(self, tiny_system):
        tiny_system.set_swept_parameters(iter(["k"]))
        assert tiny_system.swept_parameters == ("k",)

    def test_a_sweep_given_at_creation_reads_rows_in_system_order(self):
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
        """Fixed values are the values the system compiles in."""
        tiny_system.update(
            swept_parameters=("k",), fixed_parameters=(("c0", 4.0),)
        )
        assert tiny_system.fixed_parameter_values == {"c0": 4.0}
        assert tiny_system.parameters.values_dict["c0"] == 1.0

    def test_unchanged_parameters_keep_the_build(self, tiny_system):
        """Repeating the swept and fixed parameters keeps the build."""
        tiny_system.dxdt_fn
        fixed = tuple(tiny_system.fixed_parameter_values.items())
        tiny_system.update(swept_parameters=(), fixed_parameters=fixed)
        assert tiny_system.cache_valid is True

    def test_every_parameter_needs_one_role(self, tiny_system):
        """Swept and fixed names must cover each parameter once."""
        with pytest.raises(ValueError, match="once"):
            tiny_system.update(
                swept_parameters=("k",), fixed_parameters=()
            )


class TestSetDefaultParameters:
    """Cover BaseODE.set_default_parameters."""

    def test_fixed_parameter_compiles_in_its_new_default(self, tiny_system):
        """A fixed parameter's new default is compiled in."""
        tiny_system.set_default_parameters({"c0": 7.0})
        assert tiny_system.parameters.defaults["c0"] == 7.0
        assert tiny_system.fixed_parameter_values["c0"] == 7.0

    def test_swept_parameter_stays_swept(self, tiny_system):
        """A swept parameter's new default leaves it swept."""
        tiny_system.update(
            swept_parameters=("k",), fixed_parameters=(("c0", 1.0),)
        )
        tiny_system.set_default_parameters({"k": 3.0})
        assert tiny_system.swept_parameters == ("k",)
        assert tiny_system.parameters.values_dict["k"] == 3.0
        assert tiny_system.cache_valid is True

    def test_new_defaults_drop_the_fixed_values(self, tiny_system):
        tiny_system.set_swept_parameters(("k",), {"c0": 4.0})
        tiny_system.set_default_parameters({"k": 3.0})
        assert tiny_system.fixed_parameter_values == {
            "c0": tiny_system.parameters.defaults["c0"]
        }

    def test_unknown_name_raises(self, tiny_system):
        """A name outside the system's parameters raises KeyError."""
        with pytest.raises(KeyError, match="not_a_key"):
            tiny_system.set_default_parameters({"not_a_key": 1.0})


class TestCompiledParameters:
    def test_fixing_a_default_compiles_the_same_parameters(
        self, tiny_system
    ):
        tiny_system.set_swept_parameters(("k",))
        tiny_system.dxdt_fn
        compiled = tiny_system.compiled_parameters
        tiny_system.set_swept_parameters(("k",), {"c0": 1.0})
        assert tiny_system.compiled_parameters is compiled
        assert tiny_system.cache_valid is True

    def test_the_sweep_reaches_the_config_hash(self, tiny_system):
        hash_default = tiny_system.config_hash
        tiny_system.set_swept_parameters(("k",))
        hash_swept = tiny_system.config_hash
        tiny_system.set_swept_parameters(("k",), {"c0": 2.0})
        assert hash_swept != hash_default
        assert tiny_system.config_hash != hash_swept

    def test_the_layout_assembles_without_compiling(self, tiny_system):
        tiny_system.dxdt_fn
        tiny_system.set_swept_parameters(("k",), {"c0": 2.0})
        assert tiny_system.layout.swept_names == ("k",)
        assert tiny_system.sizes.swept_parameters == 1
        assert tiny_system.cache_valid is False

    def test_sizes_are_a_product_of_the_specialiser(self, tiny_system):
        first = tiny_system.sizes
        assert tiny_system.sizes is first
        tiny_system.set_swept_parameters(("k",))
        assert tiny_system.sizes is not first
        assert tiny_system.sizes.swept_parameters == 1

    def test_the_swept_set_follows_the_sweep(self, tiny_system):
        tiny_system.set_swept_parameters(("c0", "k"))
        defaults = tiny_system.parameters.defaults
        assert tiny_system.swept_set.defaults == {
            name: defaults[name] for name in ("c0", "k")
        }


class TestSetInitialValues:
    def test_new_initial_values_keep_the_build(self, tiny_system):
        tiny_system.dxdt_fn
        hash_before = tiny_system.config_hash
        tiny_system.set_initial_values({"x": 4.0})
        assert tiny_system.states.defaults == {"x": 4.0}
        assert tiny_system.cache_valid is True
        assert tiny_system.config_hash == hash_before

    def test_unknown_name_raises(self, tiny_system):
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
