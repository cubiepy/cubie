import numpy as np
import pytest
import sympy as sp
from numpy.testing import assert_array_equal

from cubie._utils import is_devfunc
from cubie.odesystems.solver_helpers import SolverHelperRequest
from cubie.odesystems.symbolic.codegen.linear_operators import (
    generate_linear_operator_code,
)
from cubie.odesystems.symbolic.helper_registry import helper_source_hash
from cubie.odesystems.symbolic.parsing.parser import parse_input
from cubie.odesystems.symbolic.symbolicODE import (
    SymbolicODE,
    create_ODE_system,
)


def _helper_fn(system, role, **kwargs):
    """Request a helper and return its device function."""
    return system.get_solver_helper(role=role, **kwargs).device_function


def test_create_with_driver_array_dict(precision):
    """Driver-array dicts pass interpolator validation at create time."""
    t_samples = np.linspace(0.0, 1.0, 11)
    ode = create_ODE_system(
        dxdt=["dx = -k * x + d1"],
        states={"x": 1.0},
        parameters={"k": 0.5},
        drivers={
            "d1": np.sin(t_samples),
            "time": t_samples,
            "wrap": False,
        },
        precision=precision,
        name="test_driver_array_create",
    )
    assert ode.num_drivers == 1


@pytest.fixture(scope="session")
def symbolic_input_simple():
    return {
        "observables": ["obs1", "obs2"],
        "parameters": {"k1": 0.32, "k2": 0.91},
        "constants": {"c1": 2.1, "c2": 1.8},
        "drivers": {"d1": 0.9, "d2": 0.8},
        "states": {"x1": 0.5, "x2": 2.0},
        "dxdt": [
            "obs1 = k1 * x1 * d2 + d1 * c1",
            "obs2 = c2 * c2 * k2 + x1 + x2 ** 2 + obs1",
            "dx1 = obs1 + c2",
            "dx2 = c1 + obs1 + obs2",
        ],
    }


@pytest.fixture(scope="session")
def simple_ode_strict(symbolic_input_simple, precision):
    return SymbolicODE.create(
        precision=precision,
        dxdt=symbolic_input_simple["dxdt"],
        states=symbolic_input_simple["states"],
        parameters=symbolic_input_simple["parameters"],
        constants=symbolic_input_simple["constants"],
        observables=symbolic_input_simple["observables"],
        drivers=symbolic_input_simple["drivers"],
        name="simpletest_strict",
        strict=True,
    )


@pytest.fixture(scope="session")
def simple_ode_nonstrict(symbolic_input_simple, precision):
    return SymbolicODE.create(
        precision=precision,
        dxdt=symbolic_input_simple["dxdt"],
        strict=False,
        name="simpletest_nonstrict",
    )


def test_create_ODE_system_strict(
    simple_ode_strict, symbolic_input_simple, precision
):
    sys1 = create_ODE_system(
        dxdt=symbolic_input_simple["dxdt"],
        precision=precision,
        states=symbolic_input_simple["states"],
        parameters=symbolic_input_simple["parameters"],
        constants=symbolic_input_simple["constants"],
        observables=symbolic_input_simple["observables"],
        drivers=symbolic_input_simple["drivers"],
        name="simpletest_strict",
        strict=True,
    )
    sys2 = simple_ode_strict
    assert_array_equal(
        sys1.constants.values_array, sys2.constants.values_array
    )
    assert_array_equal(
        sys1.parameters.values_array, sys2.parameters.values_array
    )
    assert_array_equal(
        sys1.initial_values.values_array, sys2.initial_values.values_array
    )
    assert_array_equal(
        sys1.observables.values_array, sys2.observables.values_array
    )
    assert_array_equal(sys1.num_drivers, sys2.num_drivers)


def test_create_ODE_system_nonstrict(
    simple_ode_nonstrict, symbolic_input_simple, precision
):
    sys1 = create_ODE_system(
        dxdt=symbolic_input_simple["dxdt"],
        precision=precision,
        name="simpletest_nonstrict",
    )
    sys2 = simple_ode_nonstrict
    assert_array_equal(
        sys1.constants.values_array, sys2.constants.values_array
    )
    assert_array_equal(
        sys1.parameters.values_array, sys2.parameters.values_array
    )
    assert_array_equal(
        sys1.initial_values.values_array, sys2.initial_values.values_array
    )
    assert_array_equal(
        sys1.observables.values_array, sys2.observables.values_array
    )
    assert_array_equal(sys1.num_drivers, sys2.num_drivers)


@pytest.mark.parametrize(
    "operation_ordering",
    ["greedy", "dfs", "liveness_auto"],
)
def test_operation_ordering_is_explicit_system_compile_setting(
    precision,
    operation_ordering,
):
    """System construction defaults to Kahn and accepts alternatives."""
    kwargs = {
        "dxdt": ["dx = -k * x"],
        "states": {"x": 1.0},
        "parameters": {"k": 0.5},
        "precision": precision,
        "strict": True,
    }
    kahn = create_ODE_system(
        **kwargs, name="ordering_kahn", operation_ordering="kahn"
    )
    alternative = SymbolicODE.create(
        **kwargs,
        name=f"ordering_{operation_ordering}",
        operation_ordering=operation_ordering,
    )

    assert kahn.operation_ordering == "kahn"
    assert alternative.operation_ordering == operation_ordering
    assert kahn.fn_hash == alternative.fn_hash
    assert kahn.gen_file.fn_hash != alternative.gen_file.fn_hash

    request = SolverHelperRequest(role="residual")
    assert helper_source_hash(kahn, request) != helper_source_hash(
        alternative, request
    )


@pytest.mark.parametrize("operation_ordering", ["liveness", "bogus"])
def test_create_ode_rejects_invalid_operation_ordering(
    precision,
    operation_ordering,
):
    """Direct symbolic construction validates the ordering policy."""
    with pytest.raises(ValueError, match="operation_ordering"):
        create_ODE_system(
            dxdt=["dx = -x"],
            states={"x": 1.0},
            precision=precision,
            operation_ordering=operation_ordering,
        )


def test_operation_ordering_update_rebuilds_source_and_jvp(precision):
    """Changing ordering rebuilds code without redefining fn_hash."""
    ode = create_ODE_system(
        dxdt=["dx = -k * x"],
        states={"x": 1.0},
        parameters={"k": 0.5},
        precision=precision,
        strict=True,
        name="ordering_update",
        operation_ordering="kahn",
    )
    first_function = ode.evaluate_f
    first_source_hash = ode.gen_file.fn_hash
    first_fn_hash = ode.fn_hash
    first_jvp = ode._get_jvp_exprs()

    recognised = ode.update(operation_ordering="liveness_auto")

    assert recognised == {"operation_ordering"}
    assert ode.operation_ordering == "liveness_auto"
    assert ode.fn_hash == first_fn_hash

    second_function = ode.evaluate_f
    assert second_function is not first_function
    assert ode.gen_file.fn_hash != first_source_hash
    assert ode.fn_hash == first_fn_hash
    assert ode._get_jvp_exprs() is not first_jvp


@pytest.fixture(scope="session")
def metadata_ode(precision):
    """Return a read-only system covering the metadata accessors."""
    return SymbolicODE.create(
        dxdt=["dx = -k * x + c + d1", "dy = k * x"],
        precision=precision,
        states={"x": 1.0, "y": 0.0},
        parameters={"k": 0.1},
        constants={"c": 0.5},
        drivers=["d1"],
        name="symbolicode_metadata",
    )


def test_simple_strict_builds(system):
    assert callable(_helper_fn(system, "linear_operator"))


def test_simple_nonstrict_builds(simple_ode_nonstrict):
    assert callable(_helper_fn(simple_ode_nonstrict, "linear_operator"))


def test_solver_helper_cached(system):
    func1 = _helper_fn(system, "linear_operator")
    assert callable(func1)
    func2 = _helper_fn(system, "linear_operator")
    assert func1 is func2


def test_observables_helper_available(system):
    """Symbolic systems should expose an observables-only helper."""

    func = system.evaluate_observables
    assert callable(func)
    cached = system.evaluate_observables
    assert func is cached


def test_time_derivative_helper_available(system):
    """Time-derivative helper should be compiled during system build."""

    helper = _helper_fn(system, "time_derivative_rhs")
    assert callable(helper)


@pytest.fixture(scope="session")
def sympy_string_pair(precision):
    """Return one definition parsed from sympy and from strings."""
    x, y, k, z = sp.symbols("x y k z")
    dx, dy = sp.symbols("dx dy")
    states = {"x": 1.0, "y": 0.0}
    parameters = {"k": 0.1}
    observables = ["z"]

    ode_sympy = SymbolicODE.create(
        dxdt=[sp.Eq(dx, -k * x), sp.Eq(dy, k * x), sp.Eq(z, x * k)],
        precision=precision,
        states=states,
        parameters=parameters,
        observables=observables,
        name="sympy_string_pair_sympy",
    )
    ode_string = SymbolicODE.create(
        dxdt=["dx = -k * x", "dy = k * x", "z = x * k"],
        precision=precision,
        states=states,
        parameters=parameters,
        observables=observables,
        name="sympy_string_pair_string",
    )
    return ode_sympy, ode_string


class TestSympyStringEquivalence:
    """Test equivalence of SymPy and string input pathways."""

    def test_generated_code_identical(self, sympy_string_pair):
        """Verify SymPy and string inputs generate identical code."""
        ode_sympy, ode_string = sympy_string_pair

        assert is_devfunc(ode_sympy.evaluate_f)
        assert is_devfunc(ode_string.evaluate_f)

        assert ode_sympy.num_states == ode_string.num_states
        assert ode_sympy.num_states == 2

    def test_hash_consistency(self):
        """Verify hash is consistent for equivalent definitions."""
        x, k = sp.symbols('x k')
        dx = sp.Symbol('dx')
        dxdt_sympy = [sp.Eq(dx, -k * x)]

        dxdt_string = "dx = -k * x"

        result_sympy = parse_input(
            dxdt=dxdt_sympy,
            states=['x'],
            parameters=['k'],
            constants={'c': 1.0}
        )

        result_string = parse_input(
            dxdt=dxdt_string,
            states=['x'],
            parameters=['k'],
            constants={'c': 1.0}
        )

        hash_sympy = result_sympy[4]
        hash_string = result_string[4]

        assert hash_sympy == hash_string

    def test_observables_equivalence(self, sympy_string_pair):
        """Verify observables work identically in both pathways."""
        ode_sympy, ode_string = sympy_string_pair

        assert len(ode_sympy.indices.observables.index_map) == 1
        assert len(ode_string.indices.observables.index_map) == 1


class TestSymbolicODEHash:
    """Test hash handling in SymbolicODE."""

    def test_symbolic_ode_hash_determinism(self, precision):
        """Verify identical SymbolicODE systems produce identical fn_hash."""
        dxdt = ["dx = -k * x", "dy = k * x"]
        states = {"x": 1.0, "y": 0.0}
        parameters = {"k": 0.1}

        ode1 = SymbolicODE.create(
            dxdt=dxdt,
            precision=precision,
            states=states,
            parameters=parameters,
            name="hash_test_1",
        )

        ode2 = SymbolicODE.create(
            dxdt=dxdt,
            precision=precision,
            states=states,
            parameters=parameters,
            name="hash_test_2",
        )

        assert ode1.fn_hash == ode2.fn_hash

    def test_symbolic_ode_hash_fallback(self, precision):
        """Verify __init__ correctly computes hash when fn_hash=None."""
        from cubie.odesystems.symbolic.parsing.parser import parse_input

        dxdt = ["dx = -k * x", "dy = k * x"]
        states = {"x": 1.0, "y": 0.0}
        parameters = {"k": 0.1}
        constants = {"c": 2.0}

        # Create via parse_input (provides fn_hash)
        parsed_result = parse_input(
            dxdt=dxdt,
            states=list(states.keys()),
            parameters=list(parameters.keys()),
            constants=constants,
        )
        expected_hash = parsed_result[4]

        # Create via SymbolicODE.create (computes hash internally)
        ode = SymbolicODE.create(
            dxdt=dxdt,
            precision=precision,
            states=states,
            parameters=parameters,
            constants=constants,
            name="hash_fallback_test",
        )

        assert ode.fn_hash == expected_hash


class TestCacheSkipsCodegen:
    """Tests verifying codegen is skipped when file cache exists."""

    def test_prepare_jac_aux_count_cached(self, precision):
        """Verify prepare_jac aux_count is retrieved from cached factory."""
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x", "dy = k * x"],
            precision=precision,
            states={"x": 1.0, "y": 0.0},
            parameters={"k": 0.1},
            name="cache_test_prepare_jac",
        )

        # First call generates and caches prepare_jac
        result1 = ode.get_solver_helper(role="prepare_jac", jacobian_at="step")
        assert callable(result1.device_function)
        aux_count_initial = result1.cached_auxiliary_count
        assert aux_count_initial is not None

        # Create a new ODE instance with the same definition and name
        # to exercise retrieval of prepare_jac from the file cache.
        ode_cached = SymbolicODE.create(
            dxdt=["dx = -k * x", "dy = k * x"],
            precision=precision,
            states={"x": 1.0, "y": 0.0},
            parameters={"k": 0.1},
            name="cache_test_prepare_jac",
        )

        # Second call should retrieve from file cache (no fresh codegen)
        # and restore aux_count from the cached factory attribute.
        result2 = ode_cached.get_solver_helper(
            role="prepare_jac",
            jacobian_at="step",
        )
        assert callable(result2.device_function)
        assert result2.cached_auxiliary_count == aux_count_initial

    def test_codegen_skipped_on_cache_hit(self, precision):
        """Verify that code generation is skipped when function is cached."""
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x", "dy = k * x + c"],
            precision=precision,
            states={"x": 1.0, "y": 0.0},
            parameters={"k": 0.1},
            constants={"c": 0.5},
            name="cache_skip_codegen_test",
        )

        # First call generates linear_operator
        request = SolverHelperRequest(role="linear_operator")
        helper1 = ode.get_solver_helper("linear_operator").device_function
        assert callable(helper1)

        # Verify function is marked as cached in file under its
        # source-suffixed name.
        source_hash = helper_source_hash(ode, request)
        factory_name = f"linear_operator_plain_s{source_hash}"
        assert ode.gen_file.function_is_cached(factory_name)

        # Create a new ODE instance with the same definition and name
        # to exercise retrieval from the file cache.
        ode_cached = SymbolicODE.create(
            dxdt=["dx = -k * x", "dy = k * x + c"],
            precision=precision,
            states={"x": 1.0, "y": 0.0},
            parameters={"k": 0.1},
            constants={"c": 0.5},
            name="cache_skip_codegen_test",
        )

        # Second call should skip codegen (uses file cache)
        helper2 = ode_cached.get_solver_helper(
            "linear_operator"
        ).device_function
        assert callable(helper2)

    def test_array_layout_replaces_same_name_disk_source(self, precision):
        """A changed array layout replaces source under the same name."""
        name = "cache_array_layout_replacement"
        equations = [
            "dx = a*x + b*y",
            "dy = b*x - a*y",
        ]
        first = SymbolicODE.create(
            dxdt=equations,
            precision=precision,
            states={"x": 1.0, "y": 2.0},
            parameters={"a": 3.0, "b": 4.0},
            name=name,
        )
        _ = first.evaluate_f
        first_source = first.gen_file.file_path.read_text()

        second = SymbolicODE.create(
            dxdt=equations,
            precision=precision,
            states={"x": 1.0, "y": 2.0},
            parameters={"a": 3.0},
            constants={"b": 4.0},
            name=name,
        )
        _ = second.evaluate_f
        second_source = second.gen_file.file_path.read_text()

        assert first.fn_hash != second.fn_hash
        assert first_source != second_source
        assert list(first.parameters.names) == ["a", "b"]
        assert list(second.parameters.names) == ["a"]

    def test_reordered_declaration_is_one_system(self, precision):
        """The same names in a different order are one system."""
        name = "cache_array_layout_reorder"
        equations = [
            "dx = a*x + b*y",
            "dy = b*x - a*y",
        ]
        first = SymbolicODE.create(
            dxdt=equations,
            precision=precision,
            states={"x": 1.0, "y": 2.0},
            parameters={"a": 3.0, "b": 4.0},
            name=name,
        )
        _ = first.evaluate_f
        first_source = first.gen_file.file_path.read_text()

        second = SymbolicODE.create(
            dxdt=equations,
            precision=precision,
            states={"y": 2.0, "x": 1.0},
            parameters={"b": 4.0, "a": 3.0},
            name=name,
        )
        _ = second.evaluate_f
        second_source = second.gen_file.file_path.read_text()

        assert first.fn_hash == second.fn_hash
        assert first_source == second_source
        assert list(first.parameters.names) == ["a", "b"]
        assert list(second.parameters.names) == ["a", "b"]
        assert list(second.states.names) == ["x", "y"]

    def test_derivative_helper_replaces_same_name_disk_source(
        self, precision
    ):
        """A changed derivative helper replaces cached source."""

        class DeviceFunction:
            targetoptions = {"device": True}

            def __call__(self, *args, **kwargs):
                return 0

        def grad_a(value, index):
            return 0

        def grad_b(value, index):
            return 0

        kwargs = {
            "dxdt": "dx = myfunc(x)",
            "precision": precision,
            "states": {"x": 1.0},
            "user_functions": {"myfunc": DeviceFunction()},
            "name": "cache_derivative_helper_replacement",
        }
        first = SymbolicODE.create(
            **kwargs,
            user_function_derivatives={"myfunc": grad_a},
        )
        first.gen_file.add_function(
            generate_linear_operator_code(
                first.equations,
                first.indices,
                jvp_equations=first._get_jvp_exprs(),
            )
        )
        assert "grad_a(" in first.gen_file.file_path.read_text()

        second = SymbolicODE.create(
            **kwargs,
            user_function_derivatives={"myfunc": grad_b},
        )
        second.gen_file.add_function(
            generate_linear_operator_code(
                second.equations,
                second.indices,
                jvp_equations=second._get_jvp_exprs(),
            )
        )
        second_source = second.gen_file.file_path.read_text()

        assert first.fn_hash != second.fn_hash
        assert "grad_b(" in second_source
        assert "grad_a(" not in second_source

    def test_firk_tableau_hot_swap_uses_distinct_caches(self, precision):
        """FIRK helpers are keyed by every tableau value."""
        ode = SymbolicODE.create(
            dxdt="dx = -x",
            precision=precision,
            states={"x": 1.0},
            name="cache_firk_tableau_hot_swap",
        )
        first_coefficients = ((0.5,),)
        first_nodes = (0.5,)
        second_coefficients = ((1.0,),)
        second_nodes = (1.0,)
        third_coefficients = ((0.25, 0.0), (0.5, 0.25))
        third_nodes = (0.25, 0.75)

        request_kwargs = [
            dict(
                role="residual",
                jacobian_at="stage",
                stacked=True,
                stage_coefficients=coefficients,
                stage_nodes=nodes,
            )
            for coefficients, nodes in (
                (first_coefficients, first_nodes),
                (second_coefficients, second_nodes),
                (third_coefficients, third_nodes),
            )
        ]
        first = ode.get_solver_helper(**request_kwargs[0])
        second = ode.get_solver_helper(**request_kwargs[1])
        second_again = ode.get_solver_helper(**request_kwargs[1])
        third = ode.get_solver_helper(**request_kwargs[2])

        assert first is not second
        assert second_again is second
        assert third is not second
        source = ode.gen_file.file_path.read_text()
        for kwargs in request_kwargs:
            source_hash = helper_source_hash(
                ode, SolverHelperRequest(**kwargs)
            )
            assert f"residual_stacked_stages_s{source_hash}(" in source


class TestConstantParameterConversion:
    """Tests for converting constants to parameters and vice versa."""

    def test_make_parameter_converts_constant(self, precision):
        """Verify make_parameter moves a constant to parameters."""
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x + c"],
            precision=precision,
            states={"x": 1.0},
            parameters={"k": 0.1},
            constants={"c": 0.5},
            name="test_make_param",
        )

        assert "c" in ode.indices.constant_names
        assert "c" not in ode.indices.parameter_names

        ode.make_parameter("c")

        assert "c" not in ode.indices.constant_names
        assert "c" in ode.indices.parameter_names
        assert ode.parameters["c"] == 0.5

    def test_make_constant_converts_parameter(self, precision):
        """Verify make_constant moves a parameter to constants."""
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x + c"],
            precision=precision,
            states={"x": 1.0},
            parameters={"k": 0.1, "c": 0.5},
            name="test_make_const",
        )

        assert "c" in ode.indices.parameter_names
        assert "c" not in ode.indices.constant_names

        ode.make_constant("c")

        assert "c" not in ode.indices.parameter_names
        assert "c" in ode.indices.constant_names
        assert ode.constants["c"] == 0.5

    def test_make_parameter_raises_for_unknown(self, metadata_ode):
        """Verify make_parameter raises KeyError for unknown name."""
        with pytest.raises(KeyError):
            metadata_ode.make_parameter("nonexistent")

    def test_make_constant_raises_for_unknown(self, metadata_ode):
        """Verify make_constant raises KeyError for unknown name."""
        with pytest.raises(KeyError):
            metadata_ode.make_constant("nonexistent")

    def test_roundtrip_conversion(self, precision):
        """Verify constant->parameter->constant preserves value."""
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x + c"],
            precision=precision,
            states={"x": 1.0},
            parameters={"k": 0.1},
            constants={"c": 0.5},
            name="test_roundtrip",
        )

        # Convert to parameter
        ode.make_parameter("c")
        assert ode.parameters["c"] == 0.5

        # Convert back to constant
        ode.make_constant("c")
        assert ode.constants["c"] == 0.5

    def test_make_parameter_regenerates_source(self, precision):
        """Generated source follows a constant-to-parameter move."""
        ode = SymbolicODE.create(
            dxdt="dx = -k*x + c",
            precision=precision,
            states={"x": 1.0},
            parameters={"k": 0.1},
            constants={"c": 0.5},
            name="constant_to_parameter_source",
        )
        _ = ode.evaluate_f

        ode.make_parameter("c")
        _ = ode.evaluate_f
        source = ode.gen_file.file_path.read_text()

        assert "parameters[1]" in source
        assert "precision(constants['c'])" not in source

    def test_make_constant_regenerates_source(self, precision):
        """Generated source follows a parameter-to-constant move."""
        ode = SymbolicODE.create(
            dxdt="dx = -k*x + c",
            precision=precision,
            states={"x": 1.0},
            parameters={"k": 0.1, "c": 0.5},
            name="parameter_to_constant_source",
        )
        _ = ode.evaluate_f

        ode.make_constant("c")
        _ = ode.evaluate_f
        source = ode.gen_file.file_path.read_text()

        # The parameter's value folds into the source as a literal.
        assert "precision(0.5)" in source
        assert "constants['c']" not in source
        assert "parameters[1]" not in source


class TestValueSetters:
    """Tests for value setting methods."""

    def test_set_parameter_value(self, precision):
        """Verify set_parameter_value updates parameter correctly."""
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x"],
            precision=precision,
            states={"x": 1.0},
            parameters={"k": 0.1},
            name="test_set_param",
        )

        ode.set_parameter_value("k", 0.5)
        assert ode.parameters["k"] == 0.5

    def test_set_constant_value(self, precision):
        """Verify set_constant_value updates constant correctly."""
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x + c"],
            precision=precision,
            states={"x": 1.0},
            parameters={"k": 0.1},
            constants={"c": 0.5},
            name="test_set_const",
        )

        ode.set_constant_value("c", 1.0)
        assert ode.constants["c"] == 1.0

    def test_constants_getter_is_sealed_against_bypass(self, precision):
        """In-place mutation through the public getter raises.

        The constants container held by the settings snapshot is
        fully sealed, so the ``update_compile_settings`` bypass that
        would leave the build cache valid with stale closure values
        cannot happen; ``set_constants`` remains the write path and
        changes ``config_hash``.
        """
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x + c"],
            precision=precision,
            states={"x": 1.0},
            parameters={"k": 0.1},
            constants={"c": 0.5},
            name="test_sealed_constants",
        )
        _ = ode.evaluate_f
        hash_before = ode.config_hash

        with pytest.raises(ValueError):
            ode.constants.update_from_dict({"c": 2.0})
        with pytest.raises(ValueError):
            ode.constants["c"] = 2.0
        with pytest.raises(ValueError):
            ode.constants.values_array[0] = 2.0
        with pytest.raises(TypeError):
            ode.constants.values_dict["c"] = 2.0

        assert ode.constants["c"] == precision(0.5)
        assert ode.config_hash == hash_before

        ode.set_constants({"c": 2.0})
        assert ode.constants["c"] == precision(2.0)
        assert ode.config_hash != hash_before

    def test_set_initial_value(self, precision):
        """Verify set_initial_value updates state correctly."""
        ode = SymbolicODE.create(
            dxdt=["dx = -k * x"],
            precision=precision,
            states={"x": 1.0},
            parameters={"k": 0.1},
            name="test_set_init",
        )

        ode.set_initial_value("x", 2.0)
        assert ode.initial_values["x"] == 2.0


class TestInfoGetters:
    """Tests for information getter methods."""

    def test_get_constants_info(self, metadata_ode):
        """Verify get_constants_info returns correct structure."""
        info = metadata_ode.get_constants_info()
        assert len(info) == 1
        assert info[0]["name"] == "c"
        assert info[0]["value"] == 0.5
        assert "unit" in info[0]

    def test_get_parameters_info(self, metadata_ode):
        """Verify get_parameters_info returns correct structure."""
        info = metadata_ode.get_parameters_info()
        assert len(info) == 1
        assert info[0]["name"] == "k"
        assert info[0]["value"] == 0.1
        assert "unit" in info[0]

    def test_get_states_info(self, metadata_ode):
        """Verify get_states_info returns correct structure."""
        info = metadata_ode.get_states_info()
        assert len(info) == 2
        names = [i["name"] for i in info]
        assert "x" in names
        assert "y" in names


class TestSymbolicODEConstructorDefaults:
    """Cover derivation of all_symbols and fn_hash in __init__."""

    def test_derives_symbols_and_hash_when_omitted(self):
        """A None all_symbols and fn_hash are derived from inputs."""
        index_map, _, _, equations, _, _, *_ = parse_input(
            dxdt=["dx = -k * x"],
            states={"x": 1.0},
            parameters={"k": 0.5},
            constants={},
            observables=[],
            strict=True,
        )
        ode = SymbolicODE(
            equations,
            np.float32,
            index_map,
            all_symbols=None,
            fn_hash=None,
            name=None,
        )
        assert "x" in ode.all_symbols
        assert isinstance(ode.fn_hash, str)
        # name defaults to the derived hash when omitted.
        assert ode.name == ode.fn_hash


class TestSymbolicODEUnitAccessors:
    """Cover unit accessors."""

    def test_driver_units_reports_declared_drivers(self, metadata_ode):
        """driver_units exposes units for declared drivers."""
        assert "d1" in metadata_ode.driver_units


class TestSetConstantsKwargs:
    """set_constants forwards keyword updates to the base class."""

    def test_kwargs_only_updates_compiled_constant(self):
        """A kwargs-only call updates the compiled constant value."""
        ode = create_ODE_system(
            dxdt=["dx = -k * x + c0"],
            states={"x": 1.0},
            parameters={"k": 0.5},
            constants={"c0": 1.0},
            precision=np.float32,
            strict=True,
            name="set_constants_kwargs_test",
        )
        recognised = ode.set_constants(None, c0=3.0)
        assert recognised == {"c0"}
        assert ode.constants.values_dict["c0"] == np.float32(3.0)


class TestPreconditionerTypeValidation:
    """Cover preconditioner-type validation in the algorithm layer."""

    def test_unknown_preconditioner_type_raises(self, system, precision):
        """An unknown preconditioner type raises at construction."""
        from cubie.integrators.algorithms import get_algorithm_step

        with pytest.raises(ValueError, match="preconditioner_type"):
            get_algorithm_step(
                precision=precision,
                settings={
                    "algorithm": "backwards_euler",
                    "n": system.sizes.states,
                    "preconditioner_type": "bogus",
                    "get_solver_helper_fn": system.get_solver_helper,
                },
            )
