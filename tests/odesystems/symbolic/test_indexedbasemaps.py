"""Tests for IndexedBaseMap and IndexedBases classes."""

import pytest
import sympy as sp

from cubie.odesystems.symbolic.indexedbasemaps import (
    IndexedBaseMap,
    IndexedBases,
)


class TestIndexedBaseMap:
    """Test cases for IndexedBaseMap class."""

    def test_init_basic(self):
        """Test basic initialization of IndexedBaseMap."""
        symbol_labels = ["x", "y", "z"]
        base_map = IndexedBaseMap("test", symbol_labels)

        # Create the expected symbols
        x, y, z = (
            sp.Symbol("x", real=True),
            sp.Symbol("y", real=True),
            sp.Symbol("z", real=True),
        )

        assert base_map.base_name == "test"
        assert base_map.length == 3
        assert base_map.real is True
        assert isinstance(base_map.base, sp.IndexedBase)
        assert len(base_map.index_map) == 3
        assert len(base_map.ref_map) == 3
        assert len(base_map.symbol_map) == 3

        # Check index mappings (keyed by symbol)
        assert base_map.index_map[x] == 0
        assert base_map.index_map[y] == 1
        assert base_map.index_map[z] == 2

        # Check references (keyed by symbol)
        assert str(base_map.ref_map[x]) == "test[0]"
        assert str(base_map.ref_map[y]) == "test[1]"
        assert str(base_map.ref_map[z]) == "test[2]"

        # Check symbol mappings (keyed by string)
        assert base_map.symbol_map["x"] == x
        assert base_map.symbol_map["y"] == y
        assert base_map.symbol_map["z"] == z

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        symbol_labels = ["a", "b"]
        defaults = [1.0, 2.0]
        base_map = IndexedBaseMap(
            "test", symbol_labels, input_defaults=defaults
        )

        # Create the expected symbols
        a, b = sp.Symbol("a", real=True), sp.Symbol("b", real=True)

        assert base_map.default_values == {a: 1.0, b: 2.0}

    def test_init_with_custom_length(self):
        """Test initialization with custom length."""
        symbol_labels = ["x", "y"]
        base_map = IndexedBaseMap("test", symbol_labels, length=5)

        assert base_map.length == 5
        assert base_map.base.shape == (5,)

    def test_init_not_real(self):
        """Test initialization with complex numbers."""
        symbol_labels = ["x"]
        base_map = IndexedBaseMap("test", symbol_labels, real=False)

        assert not base_map.real
        assert not base_map.base.assumptions0.get("real")

    def test_init_defaults_length_mismatch(self):
        """Test error when defaults length doesn't match symbols."""
        symbol_labels = ["x", "y"]
        defaults = [1.0]  # Wrong length

        with pytest.raises(
            ValueError, match="Input defaults must be the same length"
        ):
            IndexedBaseMap("test", symbol_labels, input_defaults=defaults)

    def test_pop_symbol(self):
        """Test removing a symbol from the map."""
        symbol_labels = ["x", "y", "z"]
        base_map = IndexedBaseMap("test", symbol_labels)

        # Create the expected symbols
        x, y, z = (
            sp.Symbol("x", real=True),
            sp.Symbol("y", real=True),
            sp.Symbol("z", real=True),
        )

        initial_length = base_map.length
        base_map.pop(y)

        assert y not in base_map.ref_map
        assert y not in base_map.index_map
        assert "y" not in base_map.symbol_map
        assert base_map.length == initial_length - 1
        assert base_map.base.shape == (2,)
        assert base_map.index_map == {x: 0, z: 1}
        assert str(base_map.ref_map[x]) == "test[0]"
        assert str(base_map.ref_map[z]) == "test[1]"

    def test_push_symbol(self):
        """Test adding a symbol to the map."""
        symbol_labels = ["x", "y"]
        base_map = IndexedBaseMap("test", symbol_labels)

        initial_length = base_map.length
        z = sp.Symbol("z", real=True)

        base_map.push(z)
        assert z in base_map.ref_map
        assert z in base_map.index_map
        assert base_map.symbol_map["z"] == z
        assert base_map.index_map[z] == initial_length
        assert base_map.length == initial_length + 1
        assert str(base_map.ref_map[z]) == f"test[{initial_length}]"

    def test_update_values_with_string_keys(self):
        """Test updating values using string keys."""
        symbol_labels = ["x", "y", "z"]
        defaults = [1.0, 2.0, 3.0]
        base_map = IndexedBaseMap(
            "test", symbol_labels, input_defaults=defaults
        )

        # Update with string keys
        base_map.update_values({"x": 10.0, "z": 30.0})

        x, y, z = (
            sp.Symbol("x", real=True),
            sp.Symbol("y", real=True),
            sp.Symbol("z", real=True),
        )

        assert base_map.default_values[x] == 10.0
        assert base_map.default_values[y] == 2.0  # unchanged
        assert base_map.default_values[z] == 30.0

    def test_update_values_with_symbol_keys(self):
        """Test updating values using Symbol keys."""
        symbol_labels = ["x", "y", "z"]
        defaults = [1.0, 2.0, 3.0]
        base_map = IndexedBaseMap(
            "test", symbol_labels, input_defaults=defaults
        )

        x, y, z = (
            sp.Symbol("x", real=True),
            sp.Symbol("y", real=True),
            sp.Symbol("z", real=True),
        )

        # Update with Symbol keys
        base_map.update_values({x: 100.0, z: 300.0})

        assert base_map.default_values[x] == 100.0
        assert base_map.default_values[y] == 2.0  # unchanged
        assert base_map.default_values[z] == 300.0

    def test_update_values_with_kwargs(self):
        """Test updating values using kwargs."""
        symbol_labels = ["x", "y", "z"]
        defaults = [1.0, 2.0, 3.0]
        base_map = IndexedBaseMap(
            "test", symbol_labels, input_defaults=defaults
        )

        # Update with kwargs
        base_map.update_values(x=50.0, y=60.0)

        x, y, z = (
            sp.Symbol("x", real=True),
            sp.Symbol("y", real=True),
            sp.Symbol("z", real=True),
        )

        assert base_map.default_values[x] == 50.0
        assert base_map.default_values[y] == 60.0
        assert base_map.default_values[z] == 3.0  # unchanged

    def test_update_values_kwargs_override_dict(self):
        """Test that kwargs override dictionary values."""
        symbol_labels = ["x", "y"]
        defaults = [1.0, 2.0]
        base_map = IndexedBaseMap(
            "test", symbol_labels, input_defaults=defaults
        )

        # kwargs should override dict values
        base_map.update_values({"x": 10.0, "y": 20.0}, x=100.0)

        x, y = sp.Symbol("x", real=True), sp.Symbol("y", real=True)

        assert base_map.default_values[x] == 100.0  # overridden by kwargs
        assert base_map.default_values[y] == 20.0  # from dict

    def test_update_values_ignores_unknown_keys(self):
        """Test that unknown keys are silently ignored."""
        symbol_labels = ["x", "y"]
        defaults = [1.0, 2.0]
        base_map = IndexedBaseMap(
            "test", symbol_labels, input_defaults=defaults
        )

        # Include unknown keys - should be ignored
        base_map.update_values(
            {"x": 10.0, "unknown": 999.0, "also_unknown": 888.0}
        )

        x, y = sp.Symbol("x", real=True), sp.Symbol("y", real=True)

        assert base_map.default_values[x] == 10.0
        assert base_map.default_values[y] == 2.0  # unchanged
        # Should not raise error and should not add unknown keys

    def test_update_values_empty_updates(self):
        """Test that empty updates don't change anything."""
        symbol_labels = ["x", "y"]
        defaults = [1.0, 2.0]
        base_map = IndexedBaseMap(
            "test", symbol_labels, input_defaults=defaults
        )

        original_values = base_map.default_values.copy()

        # Test empty dict
        base_map.update_values({})
        assert base_map.default_values == original_values

        # Test None
        base_map.update_values(None)
        assert base_map.default_values == original_values

        # Test no arguments
        base_map.update_values()
        assert base_map.default_values == original_values


class TestIndexedBases:
    """Test cases for IndexedBases class."""

    @pytest.fixture
    def sample_indexed_bases(self):
        """Create a sample IndexedBases instance for testing."""
        states = IndexedBaseMap("state", ["x", "y"])
        parameters = IndexedBaseMap("param", ["a", "b"])
        constants = IndexedBaseMap("const", ["c"])
        observables = IndexedBaseMap("obs", ["o1"])
        drivers = IndexedBaseMap("drv", ["d1"])
        dxdt = IndexedBaseMap("dxdt", ["dx", "dy"])

        return IndexedBases(
            states, parameters, constants, observables, drivers, dxdt
        )

    def test_init(self, sample_indexed_bases):
        """Test IndexedBases initialization."""
        ib = sample_indexed_bases

        assert isinstance(ib.states, IndexedBaseMap)
        assert isinstance(ib.parameters, IndexedBaseMap)
        assert isinstance(ib.constants, IndexedBaseMap)
        assert isinstance(ib.observables, IndexedBaseMap)
        assert isinstance(ib.drivers, IndexedBaseMap)
        assert isinstance(ib.dxdt, IndexedBaseMap)

        # Check all_indices combines all mappings except constants
        assert len(ib.all_indices) == 8  # 2+2+1+1+2 symbols

    def test_from_user_inputs(self):
        """Test creating IndexedBases from user inputs."""
        states = ["x", "y"]
        parameters = ["a", "b"]
        constants = ["c"]
        observables = ["o1"]
        drivers = ["d1"]

        ib = IndexedBases.from_user_inputs(
            states, parameters, constants, observables, drivers
        )

        assert list(ib.state_names) == ["x", "y"]
        assert list(ib.parameter_names) == ["a", "b"]
        assert list(ib.constant_names) == ["c"]
        assert list(ib.observable_names) == ["o1"]
        assert list(ib.driver_names) == ["d1"]
        assert list(ib.dxdt_names) == ["dx", "dy"]

    def test_from_user_inputs_with_defaults(self):
        """Test creating IndexedBases from user inputs with default values."""
        states = {"x": 1.5, "y": 2.3}
        parameters = {"a": 0.1, "b": 0.2}
        constants = {"c": 3.14}
        observables = ["o1"]
        drivers = ["d1"]

        ib = IndexedBases.from_user_inputs(
            states, parameters, constants, observables, drivers
        )

        # Check that default values are properly passed through
        assert ib.state_values[sp.Symbol("x", real=True)] == 1.5
        assert ib.state_values[sp.Symbol("y", real=True)] == 2.3
        assert ib.parameter_values[sp.Symbol("a", real=True)] == 0.1
        assert ib.parameter_values[sp.Symbol("b", real=True)] == 0.2
        assert ib.constant_values[sp.Symbol("c", real=True)] == 3.14

    def test_getters(self):
        """Test all getter properties for equality with symbols, strings, and
        values.
        """
        # Create test data with default values
        states_dict = {"x": 1.0, "y": 2.0}
        params_dict = {"a": 0.1, "b": 0.2}
        constants_dict = {"c": 3.14, "d": 2.71}
        observables_list = ["obs1", "obs2"]
        drivers_list = ["drv1"]

        ib = IndexedBases.from_user_inputs(
            states=states_dict,
            parameters=params_dict,
            constants=constants_dict,
            observables=observables_list,
            drivers=drivers_list,
        )

        # Test state getters
        state_names = ib.state_names
        state_values = ib.state_values

        assert set(state_names) == {"x", "y"}
        assert len(state_values) == len(state_names)
        assert state_values[sp.Symbol("x", real=True)] == 1.0
        assert state_values[sp.Symbol("y", real=True)] == 2.0

        # Test parameter getters
        param_names = ib.parameter_names
        param_symbols = ib.parameter_symbols
        param_values = ib.parameter_values

        assert set(param_names) == {"a", "b"}
        assert len(param_symbols) == len(param_names)
        assert all(isinstance(sym, sp.Symbol) for sym in param_symbols)
        assert set(str(sym) for sym in param_symbols) == set(param_names)
        assert len(param_values) == len(param_names)
        assert param_values[sp.Symbol("a", real=True)] == 0.1
        assert param_values[sp.Symbol("b", real=True)] == 0.2

        # Test constant getters
        const_names = ib.constant_names
        const_values = ib.constant_values

        assert set(const_names) == {"c", "d"}
        assert len(const_values) == len(const_names)
        assert const_values[sp.Symbol("c", real=True)] == 3.14
        assert const_values[sp.Symbol("d", real=True)] == 2.71

        # Test observable getters
        obs_names = ib.observable_names

        assert set(obs_names) == {"obs1", "obs2"}

        # Test driver getters
        drv_names = ib.driver_names

        assert set(drv_names) == {"drv1"}

        # Test dxdt getters
        dxdt_names = ib.dxdt_names

        assert set(dxdt_names) == {"dx", "dy"}

    def test_properties(self, sample_indexed_bases):
        """Test properties of IndexedBases."""
        ib = sample_indexed_bases

        assert "x" in ib.state_names
        assert "y" in ib.state_names
        assert "a" in ib.parameter_names
        assert "b" in ib.parameter_names
        assert "c" in ib.constant_names
        assert "o1" in ib.observable_names
        assert "d1" in ib.driver_names
        assert "dx" in ib.dxdt_names
        assert "dy" in ib.dxdt_names

    def test_getitem(self, sample_indexed_bases):
        """Test __getitem__ method."""
        ib = sample_indexed_bases

        # Should be able to access any symbol
        x_ref = ib[sp.Symbol("x", real=True)]
        assert str(x_ref) == "state[0]"

        a_ref = ib[sp.Symbol("a", real=True)]
        assert str(a_ref) == "param[0]"

    def test_all_symbols(self, sample_indexed_bases):
        """Test all_symbols property."""
        ib = sample_indexed_bases
        symbols = ib.all_symbols

        assert len(symbols) == 9
        assert "x" in symbols
        assert "a" in symbols
        assert "c" in symbols

    def test_update_constants_with_dict(self):
        """Test updating constants using dictionary."""
        states = {"x": 1.0, "y": 2.0}
        parameters = {"a": 0.1, "b": 0.2}
        constants = {"c": 3.14, "d": 2.71, "e": 1.0}
        observables = ["obs1"]
        drivers = ["drv1"]

        ib = IndexedBases.from_user_inputs(
            states, parameters, constants, observables, drivers
        )

        # Update constants using dictionary
        ib.update_constants({"c": 100.0, "d": 200.0})

        c, d, e = (
            sp.Symbol("c", real=True),
            sp.Symbol("d", real=True),
            sp.Symbol("e", real=True),
        )

        assert ib.constant_values[c] == 100.0
        assert ib.constant_values[d] == 200.0
        assert ib.constant_values[e] == 1.0  # unchanged

    def test_update_constants_with_kwargs(self):
        """Test updating constants using kwargs."""
        states = {"x": 1.0}
        parameters = {"a": 0.1}
        constants = {"c": 3.14, "d": 2.71}
        observables = ["obs1"]
        drivers = ["drv1"]

        ib = IndexedBases.from_user_inputs(
            states, parameters, constants, observables, drivers
        )

        # Update constants using kwargs
        ib.update_constants(c=999.0, d=888.0)

        c, d = sp.Symbol("c", real=True), sp.Symbol("d", real=True)

        assert ib.constant_values[c] == 999.0
        assert ib.constant_values[d] == 888.0

    def test_update_constants_kwargs_override_dict(self):
        """Test that kwargs override dictionary values for constants."""
        states = {"x": 1.0}
        parameters = {"a": 0.1}
        constants = {"c": 3.14, "d": 2.71}
        observables = ["obs1"]
        drivers = ["drv1"]

        ib = IndexedBases.from_user_inputs(
            states, parameters, constants, observables, drivers
        )

        # kwargs should override dict values
        ib.update_constants({"c": 10.0, "d": 20.0}, c=100.0)

        c, d = sp.Symbol("c", real=True), sp.Symbol("d", real=True)

        assert ib.constant_values[c] == 100.0  # overridden by kwargs
        assert ib.constant_values[d] == 20.0  # from dict

    def test_update_constants_ignores_unknown_keys(self):
        """Test that unknown constant keys are silently ignored."""
        states = {"x": 1.0}
        parameters = {"a": 0.1}
        constants = {"c": 3.14}
        observables = ["obs1"]
        drivers = ["drv1"]

        ib = IndexedBases.from_user_inputs(
            states, parameters, constants, observables, drivers
        )

        # Include unknown keys - should be ignored
        ib.update_constants({"c": 100.0, "unknown_const": 999.0})

        c = sp.Symbol("c", real=True)
        assert ib.constant_values[c] == 100.0
        # Should not raise error and should not add unknown keys

    def test_update_constants_empty_updates(self):
        """Test that empty constant updates don't change anything."""
        states = {"x": 1.0}
        parameters = {"a": 0.1}
        constants = {"c": 3.14, "d": 2.71}
        observables = ["obs1"]
        drivers = ["drv1"]

        ib = IndexedBases.from_user_inputs(
            states, parameters, constants, observables, drivers
        )

        original_values = ib.constant_values.copy()

        # Test empty dict
        ib.update_constants({})
        assert ib.constant_values == original_values

        # Test None
        ib.update_constants(None)
        assert ib.constant_values == original_values

        # Test no arguments
        ib.update_constants()
        assert ib.constant_values == original_values

    def test_update_constants_only_affects_constants(self):
        """Test that update_constants only affects constants, not other values.
        """
        states = {"x": 1.0, "y": 2.0}
        parameters = {"a": 0.1, "b": 0.2}
        constants = {"c": 3.14, "d": 2.71}
        observables = ["obs1"]
        drivers = ["drv1"]

        ib = IndexedBases.from_user_inputs(
            states, parameters, constants, observables, drivers
        )

        # Store original values
        original_state_values = ib.state_values.copy()
        original_param_values = ib.parameter_values.copy()

        # Update constants
        ib.update_constants({"c": 999.0})

        # Check that only constants changed
        assert ib.state_values == original_state_values
        assert ib.parameter_values == original_param_values
        assert ib.constant_values[sp.Symbol("c", real=True)] == 999.0


class TestIndexedBaseMapExtras:
    """Cover passthrough defaults and unit handling in IndexedBaseMap."""

    def test_dict_defaults_are_passthrough(self):
        """A dict of input defaults is used verbatim as passthrough."""
        base_map = IndexedBaseMap(
            "drv", ["d0", "d1"], input_defaults={"d0": 1.0, "d1": 2.0}
        )
        assert base_map._passthrough_defaults is True
        assert base_map.default_values == {"d0": 1.0, "d1": 2.0}
        assert base_map.defaults == {"d0": 1.0, "d1": 2.0}

    def test_units_as_sequence(self):
        """A sequence of unit strings maps to labels in order."""
        base_map = IndexedBaseMap(
            "vals", ["x", "y"], units=["metre", "second"]
        )
        assert base_map.units == {"x": "metre", "y": "second"}

    def test_units_sequence_length_mismatch_raises(self):
        """A unit sequence of the wrong length raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            IndexedBaseMap("vals", ["x", "y"], units=["metre"])

    def test_update_values_noop_when_passthrough(self):
        """update_values is a no-op when defaults are passthrough."""
        base_map = IndexedBaseMap(
            "drv", ["d0"], input_defaults={"d0": 1.0}
        )
        base_map.update_values({"d0": 5.0})
        assert base_map.default_values == {"d0": 1.0}
