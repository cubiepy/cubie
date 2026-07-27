from functools import lru_cache

import attrs
import numpy as np
import pytest
from cubie.cuda_simsafe import cuda
from cubie._utils import (
    _expand_dtype,
    build_config,
    clamp_factory,
    ensure_nonzero_size,
    float_array_validator,
    in_attr,
    is_device_validator,
    mass_equal,
    merge_kwargs_into_settings,
    nonnegative_float_array_validator,
    precision_validator,
    slice_variable_dimension,
    tol_converter,
    unpack_dict_values,
)
from cubie.cuda_simsafe import is_devfunc


@lru_cache(maxsize=None)
def _clamp_kernel(fn):
    """Compile one clamp kernel per clamp device function."""

    @cuda.jit()
    def clamp_test_kernel(d_value, d_low_clip, d_high_clip, dout):
        dout[0] = fn(d_value, d_low_clip, d_high_clip)

    return clamp_test_kernel


def clamp_tester(fn, value, low_clip, high_clip, precision):
    out = cuda.device_array(1, dtype=precision)
    d_out = cuda.to_device(out)
    _clamp_kernel(fn)[1, 1](value, low_clip, high_clip, d_out)
    n_out = d_out.copy_to_host()
    return n_out


def test_clamp_kernel_float64():
    precision = np.float64

    clamp_64 = clamp_factory(precision)
    out = clamp_tester(
        clamp_64,
        precision(-2.0),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == -1.0
    out = clamp_tester(
        clamp_64,
        precision(2.0),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == 1.0
    out = clamp_tester(
        clamp_64,
        precision(0.5),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == 0.5
    out = clamp_tester(
        clamp_64,
        precision(-0.5),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == -0.5


def test_clamp_kernel_float32():
    precision = np.float32
    clamp_32 = clamp_factory(precision)
    out = clamp_tester(
        clamp_32,
        precision(-2.0),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == -1.0
    out = clamp_tester(
        clamp_32,
        precision(2.0),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == 1.0
    out = clamp_tester(
        clamp_32,
        precision(0.5),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == 0.5
    out = clamp_tester(
        clamp_32,
        precision(-0.5),
        precision(-1.0),
        precision(1.0),
        precision,
    )
    assert out[0] == -0.5


def test_slice_variable_dimension():
    """Test slice_variable_dimension function."""
    # Test basic functionality
    result = slice_variable_dimension(slice(1, 3), 0, 3)
    expected = (slice(1, 3), slice(None), slice(None))
    assert result == expected

    # Test multiple slices and indices
    slices = [slice(1, 3), slice(0, 2)]
    indices = [0, 2]
    result = slice_variable_dimension(slices, indices, 4)
    expected = (slice(1, 3), slice(None), slice(0, 2), slice(None))
    assert result == expected

    # Test single values converted to lists
    result = slice_variable_dimension(slice(1, 3), [0], 2)
    expected = (slice(1, 3), slice(None))
    assert result == expected

    # Test error cases
    with pytest.raises(
        ValueError, match="slices and indices must have the same length"
    ):
        slice_variable_dimension([slice(1, 3)], [0, 1], 3)

    with pytest.raises(ValueError, match="indices must be less than ndim"):
        slice_variable_dimension(slice(1, 3), 3, 3)


@attrs.define
class AttrsClasstest:
    field1: int
    _field2: str


def test_in_attr():
    """Test in_attr function."""
    attrs_instance = AttrsClasstest(1, "test")

    # Test existing field
    assert in_attr("field1", attrs_instance) == True

    # Test existing private field (with underscore)
    assert in_attr("field2", attrs_instance) == True  # Should find _field2
    assert in_attr("_field2", attrs_instance) == True

    # Test non-existing field
    assert in_attr("nonexistent", attrs_instance) == False


def test_is_devfnc():
    """Test is_devfnc function."""

    @cuda.jit(device=True)
    def cuda_device_func(x, y):
        """A simple CUDA device function."""
        return x + y

    @cuda.jit(device=False)
    def cuda_kernel(x, y):
        """A regular Python function."""
        y = x

    def noncuda_func(x, y):
        """A regular Python function."""
        return x + y

    dev_is_device = is_devfunc(cuda_device_func)
    kernel_is_device = is_devfunc(cuda_kernel)
    noncuda_is_device = is_devfunc(noncuda_func)

    assert dev_is_device
    assert not kernel_is_device
    assert not noncuda_is_device


def test_unpack_dict_values_basic():
    """Test basic dict unpacking functionality."""
    # Basic unpacking: dict values are unpacked, regular values pass through
    input_dict = {
        'step_settings': {'dt_min': 0.01, 'dt_max': 1.0},
        'precision': np.float32
    }
    result, unpacked = unpack_dict_values(input_dict)
    
    assert result == {'dt_min': 0.01, 'dt_max': 1.0, 'precision': np.float32}
    assert unpacked == {'step_settings'}


def test_unpack_dict_values_mixed():
    """Test unpacking with mixed dict and non-dict values."""
    input_dict = {
        'controller': {'atol': 1e-5, 'rtol': 1e-3},
        'algorithm': 'rk4',
        'output': {'save_state': True}
    }
    result, unpacked = unpack_dict_values(input_dict)
    
    assert result == {
        'atol': 1e-5,
        'rtol': 1e-3,
        'algorithm': 'rk4',
        'save_state': True
    }
    assert unpacked == {'controller', 'output'}


def test_unpack_dict_values_empty():
    """Test unpacking with empty dict."""
    result, unpacked = unpack_dict_values({})
    assert result == {}
    assert unpacked == set()


def test_unpack_dict_values_empty_dict_value():
    """Test unpacking when a dict value is empty."""
    input_dict = {'settings': {}, 'value': 42}
    result, unpacked = unpack_dict_values(input_dict)
    
    assert result == {'value': 42}
    assert unpacked == {'settings'}


def test_unpack_dict_values_nested_dicts():
    """Test that only one level deep is unpacked."""
    # Nested dicts within dict values are NOT recursively unpacked
    input_dict = {
        'outer': {'inner': {'nested': 'value'}, 'regular': 5}
    }
    result, unpacked = unpack_dict_values(input_dict)
    
    # Should unpack outer, but leave inner as a dict
    assert result == {'inner': {'nested': 'value'}, 'regular': 5}
    assert unpacked == {'outer'}


def test_unpack_dict_values_no_dicts():
    """Test when there are no dict values to unpack."""
    input_dict = {'a': 1, 'b': 2, 'c': 'test'}
    result, unpacked = unpack_dict_values(input_dict)
    
    assert result == input_dict
    assert unpacked == set()


def test_unpack_dict_values_collision_regular_and_unpacked():
    """Test that key collision between regular entry and unpacked dict raises error."""
    # A key appears both as a regular entry and within an unpacked dict
    input_dict = {
        'dt_min': 0.001,
        'step_settings': {'dt_min': 0.01}
    }
    
    with pytest.raises(ValueError, match="Key collision detected.*dt_min"):
        unpack_dict_values(input_dict)


def test_unpack_dict_values_collision_multiple_unpacked():
    """Test that key collision between multiple unpacked dicts raises error."""
    # Same key appears in two different dict values
    input_dict = {
        'settings1': {'dt_min': 0.01},
        'settings2': {'dt_min': 0.02}
    }
    
    with pytest.raises(ValueError, match="Key collision detected.*dt_min"):
        unpack_dict_values(input_dict)


def test_unpack_dict_values_collision_duplicate_regular():
    """Test that duplicate regular keys raise error."""
    # This shouldn't happen in normal Python dict creation, but test the check
    # Note: Python dicts don't allow duplicate keys, so this tests the robustness
    # of our implementation. We'll test by processing in order.
    input_dict = {'a': 1, 'b': {'a': 2}}
    
    # Should raise error because 'a' appears in result, then 'b' unpacks 'a'
    with pytest.raises(ValueError, match="Key collision detected.*a"):
        unpack_dict_values(input_dict)


def test_unpack_dict_values_preserves_types():
    """Test that unpacking preserves various value types."""
    input_dict = {
        'settings': {
            'int_val': 42,
            'float_val': 3.14,
            'str_val': 'test',
            'bool_val': True,
            'none_val': None,
            'list_val': [1, 2, 3],
        }
    }
    result, unpacked = unpack_dict_values(input_dict)
    
    assert result['int_val'] == 42
    assert result['float_val'] == 3.14
    assert result['str_val'] == 'test'
    assert result['bool_val'] is True
    assert result['none_val'] is None
    assert result['list_val'] == [1, 2, 3]
    assert unpacked == {'settings'}


# =============================================================================
# Tests for build_config helper function
# =============================================================================


@attrs.define
class SimpleTestConfig:
    """Simple attrs config for testing build_config."""
    precision: type = attrs.field()
    n: int = attrs.field()
    optional_float: float = attrs.field(default=1.0)
    optional_str: str = attrs.field(default='default')


@attrs.define
class ConfigWithFactory:
    """Config with attrs.Factory default for testing build_config."""
    precision: type = attrs.field()
    data: dict = attrs.field(factory=dict)
    items: list = attrs.field(factory=list)


@attrs.define
class ConfigWithAlias:
    """Config with underscore-prefixed field (auto-aliased) for testing."""
    precision: type = attrs.field()
    _private_value: float = attrs.field(default=0.5, alias='private_value')


class TestBuildConfig:
    """Tests for build_config helper function."""

    def test_build_config_basic(self):
        """Verify basic config construction with required params only."""
        config = build_config(
            SimpleTestConfig,
            required={'precision': np.float32, 'n': 3},
        )
        assert config.precision == np.float32
        assert config.n == 3
        assert config.optional_float == 1.0
        assert config.optional_str == 'default'

    def test_build_config_optional_override(self):
        """Verify optional parameters override defaults."""
        config = build_config(
            SimpleTestConfig,
            required={'precision': np.float64, 'n': 5},
            optional_float=2.5,
            optional_str='custom',
        )
        assert config.precision == np.float64
        assert config.n == 5
        assert config.optional_float == 2.5
        assert config.optional_str == 'custom'

    def test_build_config_passes_values_directly(self):
        """Verify build_config passes all values directly to attrs.
        
        Note: None filtering happens when settings are merged.
        If None values reach build_config, they are passed through to attrs.
        """
        # This test verifies the pass-through behavior
        config = build_config(
            SimpleTestConfig,
            required={'precision': np.float32, 'n': 2},
            optional_float=3.14,
        )
        assert config.optional_float == 3.14
        assert config.optional_str == 'default'

    def test_build_config_attrs_handles_missing_required(self):
        """Verify attrs raises error on missing required fields."""
        with pytest.raises(TypeError):
            build_config(
                SimpleTestConfig,
                required={'precision': np.float32},
            )

    def test_build_config_extra_kwargs_ignored(self):
        """Verify extra kwargs are silently ignored."""
        config = build_config(
            SimpleTestConfig,
            required={'precision': np.float32, 'n': 4},
            extra_param='ignored',
            another_extra=123,
        )
        assert config.precision == np.float32
        assert config.n == 4
        assert not hasattr(config, 'extra_param')
        assert not hasattr(config, 'another_extra')

    def test_build_config_non_attrs_raises(self):
        """Verify TypeError for non-attrs class."""
        class RegularClass:
            def __init__(self, x):
                self.x = x

        with pytest.raises(TypeError, match="is not an attrs class"):
            build_config(
                RegularClass,
                required={'x': 1},
            )

    def test_build_config_factory_defaults(self):
        """Verify attrs.Factory defaults are handled correctly."""
        config = build_config(
            ConfigWithFactory,
            required={'precision': np.float32},
        )
        assert config.precision == np.float32
        assert config.data == {}
        assert config.items == []
        assert config.data is not ConfigWithFactory.__attrs_attrs__[1].default

    def test_build_config_factory_override(self):
        """Verify attrs.Factory defaults can be overridden."""
        config = build_config(
            ConfigWithFactory,
            required={'precision': np.float32},
            data={'key': 'value'},
            items=[1, 2, 3],
        )
        assert config.data == {'key': 'value'}
        assert config.items == [1, 2, 3]

    def test_build_config_alias_handling(self):
        """Verify underscore-prefixed fields with aliases work correctly."""
        config = build_config(
            ConfigWithAlias,
            required={'precision': np.float32},
            private_value=0.75,
        )
        assert config.precision == np.float32
        assert config._private_value == 0.75

    def test_build_config_alias_default(self):
        """Verify alias fields use defaults when not overridden."""
        config = build_config(
            ConfigWithAlias,
            required={'precision': np.float64},
        )
        assert config._private_value == 0.5

    def test_build_config_with_real_config_class(self):
        """Test build_config with actual cubie config class."""
        from cubie.integrators.step_control.fixed_step_controller import (
            FixedStepControlConfig
        )
        config = build_config(
            FixedStepControlConfig,
            required={'precision': np.float32, 'n': 3, 'dt': 0.01},
        )
        assert config.precision == np.float32
        assert config.n == 3

    def test_build_config_required_in_optional_overrides(self):
        """Verify required fields can also be in optional kwargs."""
        config = build_config(
            SimpleTestConfig,
            required={'precision': np.float32},
            n=7,
        )
        assert config.n == 7

    def test_build_config_empty_required(self):
        """Verify empty required dict works when all fields have defaults."""
        @attrs.define
        class AllOptionalConfig:
            value: int = attrs.field(default=42)
            name: str = attrs.field(default='test')

        config = build_config(
            AllOptionalConfig,
            required={},
        )
        assert config.value == 42
        assert config.name == 'test'


# =============================================================================
# Tests for tol_converter helper function
# =============================================================================


class MockConfig:
    """Mock configuration object for tol_converter tests."""

    def __init__(self, tol_length, precision):
        self.tol_length = tol_length
        self.precision = precision


def test_tol_converter_scalar_to_array():
    """Verify scalar input is broadcast to array of shape (n,)."""
    config = MockConfig(tol_length=5, precision=np.float32)
    result = tol_converter(1e-6, config)

    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.float32
    assert np.allclose(result, 1e-6)


def test_tol_converter_single_element_broadcast():
    """Verify single-element array is broadcast when n > 1."""
    config = MockConfig(tol_length=4, precision=np.float64)
    result = tol_converter(np.array([0.001]), config)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)
    assert result.dtype == np.float64
    assert np.allclose(result, 0.001)


def test_tol_converter_full_array_copied():
    """Verify full-length array is copied with dtype conversion."""
    config = MockConfig(tol_length=3, precision=np.float32)
    input_array = np.array([1e-3, 2e-3, 3e-3], dtype=np.float64)
    result = tol_converter(input_array, config)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert result.dtype == np.float32
    assert np.allclose(result, [1e-3, 2e-3, 3e-3])


def test_tol_converter_returns_owned_readonly_array():
    """Every branch returns an owned, read-only array."""
    config = MockConfig(tol_length=3, precision=np.float32)

    scalar_result = tol_converter(1e-6, config)
    assert not scalar_result.flags.writeable

    uniform = np.full(2, 1e-5, dtype=np.float32)
    uniform_result = tol_converter(uniform, config)
    assert not uniform_result.flags.writeable

    caller = np.array([1e-3, 2e-3, 3e-3], dtype=np.float32)
    result = tol_converter(caller, config)
    assert result is not caller
    assert not result.flags.writeable
    with pytest.raises(ValueError):
        result[0] = 9.0

    # Caller-side mutation does not reach the stored array.
    caller[0] = 9.0
    assert result[0] == np.float32(1e-3)

    # A stored read-only array converts again without error.
    rerun = tol_converter(result, config)
    assert rerun is not result
    assert not rerun.flags.writeable


def test_tol_converter_wrong_size_raises():
    """Verify ValueError raised for wrong size array."""
    config = MockConfig(tol_length=5, precision=np.float32)

    with pytest.raises(ValueError, match="tol must have shape"):
        tol_converter(np.array([1e-3, 2e-3]), config)


# =============================================================================
# Tests for ensure_nonzero_size helper function
# =============================================================================


class TestEnsureNonzeroSize:
    """Tests for ensure_nonzero_size utility function."""

    def test_single_zero_means_all_ones(self):
        """Test that single zero causes entire tuple to become all 1s."""
        result = ensure_nonzero_size((2, 0, 2))
        assert result == (1, 1, 1)

    def test_multiple_zeros_all_ones(self):
        """Test that multiple zeros cause entire tuple to become all 1s."""
        result = ensure_nonzero_size((0, 2, 0))
        assert result == (1, 1, 1)

    def test_all_zeros_replaced(self):
        """Test that all zeros are replaced with all 1s."""
        result = ensure_nonzero_size((0, 0, 0))
        assert result == (1, 1, 1)

    def test_no_zeros_unchanged(self):
        """Test that tuple with no zeros is unchanged."""
        result = ensure_nonzero_size((2, 3, 4))
        assert result == (2, 3, 4)

    def test_integer_zero(self):
        """Test that integer zero becomes 1."""
        result = ensure_nonzero_size(0)
        assert result == 1

    def test_integer_nonzero(self):
        """Test that nonzero integer is unchanged."""
        result = ensure_nonzero_size(5)
        assert result == 5

    def test_first_element_zero_all_ones(self):
        """Test zero in first position causes all 1s."""
        result = ensure_nonzero_size((0, 3, 4))
        assert result == (1, 1, 1)

    def test_last_element_zero_all_ones(self):
        """Test zero in last position causes all 1s."""
        result = ensure_nonzero_size((2, 3, 0))
        assert result == (1, 1, 1)

    def test_string_tuple_passthrough(self):
        """Test that tuple of strings is passed through unchanged."""
        result = ensure_nonzero_size(("time", "variable", "run"))
        assert result == ("time", "variable", "run")

    def test_mixed_type_tuple_with_zero(self):
        """Test tuple with mixed numeric and non-numeric values with zero."""
        result = ensure_nonzero_size((0, "label", 2))
        assert result == (1, 1, 1)

    def test_mixed_type_tuple_no_zero(self):
        """Test tuple with mixed numeric and non-numeric values, no zero."""
        result = ensure_nonzero_size((3, "label", 2))
        assert result == (3, "label", 2)

    def test_none_treated_as_zero(self):
        """Test that None values in tuple cause all 1s."""
        result = ensure_nonzero_size((5, None, 3))
        assert result == (1, 1, 1)

    def test_two_element_tuple_with_zero(self):
        """Test two-element tuple with zero becomes (1, 1)."""
        result = ensure_nonzero_size((0, 5))
        assert result == (1, 1)

    def test_two_element_tuple_no_zero(self):
        """Test two-element tuple without zero is unchanged."""
        result = ensure_nonzero_size((3, 5))
        assert result == (3, 5)

    def test_non_int_non_tuple_passthrough(self):
        """Test that values which are neither int nor tuple pass through."""
        assert ensure_nonzero_size("label") == "label"
        assert ensure_nonzero_size(None) is None
        assert ensure_nonzero_size(2.5) == 2.5


# =============================================================================
# Tests for precision_validator
# =============================================================================


def test_precision_validator_accepts_allowed_precision():
    """Verify precision_validator does not raise for a supported dtype."""
    precision_validator(None, None, np.float32)


def test_precision_validator_rejects_unsupported_precision():
    """Verify precision_validator raises ValueError for bad precision."""
    with pytest.raises(ValueError, match="precision must be one of"):
        precision_validator(None, None, np.int32)


# =============================================================================
# Tests for is_device_validator
# =============================================================================


def test_is_device_validator_accepts_device_function():
    """Verify is_device_validator does not raise for a device function."""
    @cuda.jit(device=True)
    def dev_func(x):
        return x

    is_device_validator(None, "attr", dev_func)


def test_is_device_validator_rejects_non_device_function():
    """Verify is_device_validator raises TypeError for a plain callable."""
    def plain_func(x):
        return x

    with pytest.raises(TypeError, match="must be a Numba CUDA device"):
        is_device_validator(None, "attr", plain_func)


# =============================================================================
# Tests for float_array_validator / nonnegative_float_array_validator
# =============================================================================


def test_float_array_validator_accepts_finite_float_array():
    """Verify float_array_validator does not raise for a valid array."""
    float_array_validator(None, "attr", np.array([1.0, 2.0], dtype=np.float64))


def test_float_array_validator_rejects_non_ndarray():
    """Verify float_array_validator raises TypeError for a non-array."""
    with pytest.raises(TypeError, match="must be a numpy array of floats"):
        float_array_validator(None, "attr", [1.0, 2.0])


def test_float_array_validator_rejects_non_float_dtype():
    """Verify float_array_validator raises TypeError for an int array."""
    with pytest.raises(TypeError, match="got dtype"):
        float_array_validator(None, "attr", np.array([1, 2], dtype=np.int32))


def test_float_array_validator_rejects_nan_or_inf():
    """Verify float_array_validator raises ValueError for NaN/inf values."""
    with pytest.raises(ValueError, match="must not contain NaNs"):
        float_array_validator(
            None, "attr", np.array([1.0, np.nan], dtype=np.float64)
        )


def test_nonnegative_float_array_validator_rejects_negative():
    """Verify nonnegative_float_array_validator raises for negatives."""
    with pytest.raises(ValueError, match="must not contain negative"):
        nonnegative_float_array_validator(
            None, "attr", np.array([1.0, -1.0], dtype=np.float64)
        )


def test_nonnegative_float_array_validator_accepts_nonnegative():
    """Verify nonnegative_float_array_validator accepts nonnegative array."""
    nonnegative_float_array_validator(
        None, "attr", np.array([0.0, 1.0], dtype=np.float64)
    )


# =============================================================================
# Tests for _expand_dtype
# =============================================================================


def test_expand_dtype_float_expands_to_np_floating():
    """Verify float expands to (float, np_floating)."""
    assert _expand_dtype(float) == (float, np.floating)


def test_expand_dtype_int_expands_to_np_integer():
    """Verify int expands to (int, np_integer)."""
    assert _expand_dtype(int) == (int, np.integer)


def test_expand_dtype_other_type_passthrough():
    """Verify a type that is neither float nor int passes through."""
    assert _expand_dtype(str) is str
    assert _expand_dtype(bool) is bool


# =============================================================================
# Tests for merge_kwargs_into_settings duplicate-key warning
# =============================================================================


def test_merge_kwargs_into_settings_warns_on_duplicate_keys():
    """Verify a UserWarning is raised when kwargs and user_settings share
    keys, and that the kwargs value takes precedence."""
    with pytest.warns(UserWarning, match="Duplicate settings"):
        merged, recognized = merge_kwargs_into_settings(
            {"dt_min": 0.5},
            valid_keys={"dt_min"},
            user_settings={"dt_min": 0.1},
        )
    assert merged["dt_min"] == 0.5
    assert recognized == {"dt_min"}


def test_merge_kwargs_into_settings_no_warning_without_duplicates():
    """Verify no warning is raised when keys don't overlap."""
    merged, recognized = merge_kwargs_into_settings(
        {"dt_min": 0.5},
        valid_keys={"dt_min"},
        user_settings={"dt_max": 1.0},
    )
    assert merged == {"dt_max": 1.0, "dt_min": 0.5}
    assert recognized == {"dt_min"}


def test_merge_kwargs_into_settings_ignores_none_values():
    """None values do not override component defaults."""
    merged, recognized = merge_kwargs_into_settings(
        {"dt_min": None, "dt_max": 1.0},
        valid_keys={"dt_min", "dt_max"},
        user_settings={"dt_min": 0.1, "dt_max": None},
    )
    assert merged == {"dt_max": 1.0, "dt_min": 0.1}
    assert recognized == {"dt_min", "dt_max"}


# =============================================================================
# Tests for mass_equal
# =============================================================================


def test_mass_equal_both_none():
    """Verify mass_equal returns True when both operands are None."""
    assert mass_equal(None, None) is True


def test_mass_equal_one_none_one_not():
    """Verify mass_equal returns False when only one operand is None."""
    assert mass_equal(None, np.array([1.0])) is False
    assert mass_equal(np.array([1.0]), None) is False


def test_mass_equal_ndarray_operands():
    """Verify mass_equal compares ndarrays elementwise."""
    left = np.array([1.0, 2.0])
    right = np.array([1.0, 2.0])
    assert mass_equal(left, right) is True
    assert mass_equal(left, np.array([1.0, 3.0])) is False


def test_mass_equal_non_ndarray_operands():
    """Verify mass_equal falls back to == for non-ndarray operands."""
    assert mass_equal(1, 1) is True
    assert mass_equal(1, 2) is False


# =============================================================================
# Tests for unpack_dict_values regular-key collision with an unpacked dict
# =============================================================================


def test_unpack_dict_values_collision_unpacked_then_regular():
    """Test collision when a regular key duplicates an already-unpacked
    dict key (the reverse order of the existing collision tests)."""
    input_dict = {"settings": {"dt_min": 0.01}, "dt_min": 0.5}

    with pytest.raises(ValueError, match="Key collision detected.*dt_min"):
        unpack_dict_values(input_dict)


# =============================================================================
# Tests for build_config instance_label validation
# =============================================================================


def test_build_config_instance_label_invalid_for_class_raises():
    """Verify ValueError when instance_label is used with a config class
    that has no instance_label attribute."""
    with pytest.raises(ValueError, match="is not valid for"):
        build_config(
            SimpleTestConfig,
            required={"precision": np.float32, "n": 3},
            instance_label="krylov",
        )
