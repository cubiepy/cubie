import math
import warnings

import pytest
import sympy as sp

from tests._utils import specialise_all_swept

from cubie.odesystems.symbolic.codegen import (
    generate_linear_operator_code,
    print_cuda_multiple,
)

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import (
    from_sympy,
    to_sympy,
)

from cubie.odesystems.symbolic.parsing.normalise import (
    _process_calls,
)
from cubie.odesystems.symbolic.parsing.parse_primitives import (
    EquationWarning,
    ParsedEquations,
    _replace_if,
    _sanitise_input_math,
)
from cubie.odesystems.symbolic.parsing.parser import (
    _detect_input_type,
    parse_system,
)
from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.symbolic.sym_utils import hash_system_definition


class TestInputCleaning:
    """Test input sanitization functions."""

    def test_sanitise_input_math_simple_if(self):
        """Test sanitizing simple if-else expressions."""
        expr = "a if x > 0 else b"
        result = _sanitise_input_math(expr)
        assert "Piecewise" in result
        assert "a" in result and "b" in result

    def test_sanitise_input_math_nested_if(self):
        """Test sanitizing nested if-else expressions."""
        expr = "a if x > 0 else (b if y > 0 else c)"
        result = _sanitise_input_math(expr)
        assert result.count("Piecewise") == 2

    def test_replace_if_simple(self):
        """Test _replace_if with simple expression."""
        expr = "a if x > 0 else b"
        result = _replace_if(expr)
        expected = "Piecewise((a, x > 0), (b, True))"
        assert result == expected

    def test_replace_if_complex_condition(self):
        """Test _replace_if with complex condition."""
        expr = "x + 1 if y < z and w > 0 else x - 1"
        result = _replace_if(expr)
        assert "Piecewise((x + 1, y < z and w > 0), (x - 1, True))" == result

    def test_replace_if_no_match(self):
        """Test _replace_if when no if-else pattern is found."""
        expr = "x + y"
        result = _replace_if(expr)
        assert result == expr


class TestProcessCalls:
    """Test function call processing."""

    def test_process_calls_sympy_functions(self):
        """Test processing known SymPy functions."""
        equations = ["dx = sin(x) + cos(y)", "dy = exp(z)"]
        funcs = _process_calls(equations)

        assert "sin" in funcs
        assert "cos" in funcs
        assert "exp" in funcs
        assert callable(funcs["sin"])

    def test_process_calls_user_functions(self):
        """Test processing user-defined functions."""
        equations = ["dx = custom_func(x)"]
        user_funcs = {"custom_func": lambda x: x**2}

        funcs = _process_calls(equations, user_funcs)
        assert "custom_func" in funcs
        assert funcs["custom_func"] == user_funcs["custom_func"]

    def test_process_calls_unknown_function(self):
        """Test error when unknown function is called."""
        equations = ["dx = unknown_func(x)"]

        with pytest.raises(ValueError, match="function unknown_func"):
            _process_calls(equations)

    def test_process_calls_no_functions(self):
        """Test when no function calls are present."""
        equations = ["dx = x + y"]
        funcs = _process_calls(equations)
        assert funcs == {}

    def test_process_calls_d_notation_exempt(self):
        """The d() derivative notation is not resolved as a call."""
        equations = ["d(x, t) = -x"]
        funcs = _process_calls(equations)
        assert funcs == {}


class TestDetectInputType:
    """Test input type detection for parse_system."""

    def test_detect_string_single_line(self):
        """Test detection of single-line string input."""
        dxdt = "dx = -k * x"
        result = _detect_input_type(dxdt)
        assert result == "string"

    def test_detect_string_list(self):
        """Test detection of string list input."""
        dxdt = ["dx = -k * x", "dy = k * x"]
        result = _detect_input_type(dxdt)
        assert result == "string"

    def test_detect_sympy_equality(self):
        """Test detection of sp.Equality input."""
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")
        dxdt = [sp.Eq(dx, -k * x)]
        result = _detect_input_type(dxdt)
        assert result == "sympy"

    def test_detect_sympy_tuple(self):
        """Test detection of (Symbol, Expr) tuple input."""
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")
        dxdt = [(dx, -k * x)]
        result = _detect_input_type(dxdt)
        assert result == "sympy"

    def test_detect_sympy_expression(self):
        """Test detection of bare sp.Expr input."""
        x, k = sp.symbols("x k")
        dxdt = [-k * x]
        result = _detect_input_type(dxdt)
        assert result == "sympy"

    def test_detect_bare_equality(self):
        """A single sp.Equality outside a list is SymPy input."""
        x = sp.Symbol("x")
        assert _detect_input_type(sp.Eq(sp.Symbol("dx"), -x)) == "sympy"

    def test_detect_none_input(self):
        """Test error on None input."""
        with pytest.raises(TypeError, match="cannot be None"):
            _detect_input_type(None)

    def test_detect_empty_list(self):
        """Test error on empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _detect_input_type([])

    def test_detect_invalid_type(self):
        """Test error on invalid type."""
        with pytest.raises(TypeError, match="must be string or iterable"):
            _detect_input_type(123)

    def test_detect_invalid_element_type(self):
        """Test error on invalid element type."""
        with pytest.raises(
            TypeError, match="must be strings or symbolic expressions"
        ):
            _detect_input_type([123, 456])

    def test_detect_tuple_with_unconvertible_member(self):
        """A (lhs, rhs) tuple with a non-expression member errors."""
        with pytest.raises(TypeError, match="lhs is"):
            _detect_input_type([(object(), sp.Symbol("x"))])
        with pytest.raises(TypeError, match="rhs is"):
            _detect_input_type([(sp.Symbol("dx"), {"a": 1})])

    def test_detect_tuple_with_string_members_accepted(self):
        """Strings and numbers are sympifiable tuple members."""
        assert _detect_input_type([("dx", "-k*x")]) == "sympy"
        assert _detect_input_type([(sp.Symbol("dx"), -1.5)]) == "sympy"


class TestSympyEquationErrors:
    """Error paths for SymPy equation input."""

    @pytest.mark.parametrize(
        "equation",
        [
            (ir.sym("dx"), -sp.Symbol("x")),
            (sp.Symbol("dx"), -ir.sym("x")),
        ],
    )
    def test_mixed_ir_sympy_tuple(self, equation):
        """Each tuple member converts independently."""

        specialised = parse_system(
            dxdt=[equation], states=["x"]
        ).specialise()
        equations = specialised.equations
        assert equations.ordered == ((ir.sym("dx"), -ir.sym("x")),)

    def test_bare_expression_rejected(self):
        """A bare sp.Expr element cannot infer an LHS."""
        x = sp.Symbol("x")
        with pytest.raises(TypeError, match="expected sp.Eq or a"):
            parse_system(dxdt=[x + 1], states=["x"]).specialise()

    def test_unconvertible_member_mid_list_names_equation(self):
        """Garbage after the first element errors with its index."""
        x = sp.Symbol("x")
        equations = [
            (sp.Symbol("dx"), -x),
            (sp.Symbol("dy"), object()),
        ]
        with pytest.raises(
            TypeError, match="Equation 1: could not convert"
        ):
            parse_system(dxdt=equations, states=["x", "y"]).specialise()

    def test_derivative_of_function_rejected(self):
        """Derivative of a function application is unsupported."""
        f = sp.Function("f")
        t = sp.Symbol("t")
        equations = [sp.Eq(sp.Derivative(f(t), t), sp.Symbol("x"))]

        with pytest.raises(
            ValueError, match="non-symbol expression"
        ):
            parse_system(dxdt=equations, states=["x"]).specialise()

    def test_undefined_symbol_strict_raises(self):
        """Strict mode rejects undefined RHS symbols."""
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")
        with pytest.raises(ValueError, match="undefined symbol"):
            parse_system(
                dxdt=[sp.Eq(dx, -k * x)], states=["x"], strict=True
            ).specialise()

    def test_immutable_assignment_rejected(self):
        """Assigning a parameter raises."""
        x, k = sp.symbols("x k")
        with pytest.raises(ValueError, match="immutable input"):
            parse_system(
                dxdt=[
                    sp.Eq(sp.Symbol("dx"), -k * x),
                    sp.Eq(k, x + 1),
                ],
                states=["x"],
                parameters=["k"],
            ).specialise()


class TestLhsSemantics:
    """Left-hand-side classification through the public parser."""

    def test_auxiliary_inference(self, simple_system_defaults):
        """Aux assignments become anonymous auxiliaries."""
        (
            states,
            parameters,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        specialised = parse_system(
            states=states,
            parameters=parameters,
            observables=observables,
            drivers=drivers,
            dxdt=dxdt_list,
        ).specialise()
        layout = specialised.layout
        equations = specialised.equations
        assert equations.mass_matrix is None
        assert "done" in layout.dxdt_names
        assert ir.sym("uninited") in equations.auxiliary_symbols

    def test_strict_unlisted_auxiliary(self):
        """Unlisted LHS assignments stay anonymous in strict mode."""
        specialised = specialise_all_swept(
            dxdt=["obs = x + a", "aux_val = obs", "dx = obs"],
            states=["x"],
            parameters=["a"],
            observables=["obs"],
            strict=True,
        )
        equations = specialised.equations
        assert ir.sym("aux_val") in equations.auxiliary_symbols

    def test_observable_with_derivative_becomes_state(self):
        """A derivative equation for an observable promotes it."""
        with pytest.warns(
            EquationWarning, match="selected as a\\s+solver state"
        ):
            specialised = parse_system(
                dxdt=["dy = x + a", "dx = y"],
                states=["x"],
                parameters=["a"],
                observables=["y"],
            ).specialise()
            layout = specialised.layout
            equations = specialised.equations
        assert equations is not None
        assert "y" in layout.state_names

    def test_state_assigned_algebraically_is_eliminated(self):
        """A declared state with an algebraic assignment reduces."""
        with pytest.warns(
            EquationWarning, match="eliminated by structural"
        ):
            specialised = parse_system(
                dxdt=["dx = y", "y = a + 1"],
                states={"x": 0.0, "y": 0.0},
                parameters=["a"],
            ).specialise()
            layout = specialised.layout
            equations = specialised.equations
        assert equations is not None
        assert list(layout.state_names) == ["x"]

    def test_immutable_assignment_rejected(self):
        """Assigning a parameter raises."""
        with pytest.raises(ValueError, match="immutable input"):
            parse_system(
                dxdt=["a = x + 1", "dx = x"],
                states=["x"],
                parameters=["a"],
                observables=["obs"],
            ).specialise()

    def test_unassigned_observable_rejected(self):
        """An observable with no defining equation is rejected."""
        with pytest.raises(
            ValueError, match="no\\s+defining equation"
        ):
            parse_system(
                dxdt=["dx = x + a"],
                states=["x"],
                parameters=["a"],
                observables=["obs"],
            ).specialise()


class TestRhsSemantics:
    """Right-hand-side parsing through the public parser."""

    def test_undefined_symbol_strict(self):
        """Strict mode rejects undefined RHS symbols."""
        with pytest.raises(ValueError, match="Undefined symbols"):
            parse_system(
                dxdt=["dx = x + undefined_var"],
                states=["x"],
                strict=True,
            ).specialise()

    def test_if_else_becomes_piecewise(self):
        """Inline conditionals parse to Piecewise."""
        specialised = specialise_all_swept(
            dxdt=["dx = a if x > 0 else b"],
            states=["x"],
            parameters=["a", "b"],
        )
        equations = specialised.equations
        x, a, b = sp.symbols("x a b", real=True)
        assert equations.ordered[0][1] is from_sympy(
            sp.Piecewise((a, x > 0), (b, True))
        )

    def test_time_symbol_available(self):
        """``t`` is available without declaration."""
        specialised = parse_system(
            dxdt=["dx = t"],
            states={"x": 0.0},
            strict=True,
        ).specialise()
        equations = specialised.equations
        assert equations.non_observable_equations() == [
            (ir.sym("dx"), ir.sym("t"))
        ]
        assert specialised.functions == {}

    def test_rhs_derivative_reference_binds_assignment(self):
        """RHS ``dX`` tokens bind to the derivative of ``X``."""
        specialised = parse_system(
            dxdt=["dx = -x", "speed = dx**2"],
            states=["x"],
        ).specialise()
        equations = specialised.equations
        assert equations.mass_matrix is None
        eqs = {lhs.name: rhs for lhs, rhs in equations.ordered}
        # The solved derivative substitutes into the reference.
        x = sp.Symbol("x", real=True)
        assert sp.simplify(to_sympy(eqs["speed"]) - x**2) == 0


class TestHashSystemDefinition:
    """Test system definition hashing function."""

    def test_hash_system_definition_tuple_input(self):
        """Test hashing (Symbol, Expr) tuple input."""
        x, y = sp.symbols("x y", real=True)
        dx, dy = sp.symbols("dx dy", real=True)
        dxdt = [(dx, x + y), (dy, x - y)]
        result = hash_system_definition(dxdt, {})
        assert isinstance(result, str)

    def test_hash_system_definition_list(self):
        """Test hashing list of (Symbol, Expr) tuples."""
        x, y = sp.symbols("x y", real=True)
        dx, dy = sp.symbols("dx dy", real=True)
        dxdt = [(dx, x + y), (dy, x - y)]
        result = hash_system_definition(dxdt, {})
        assert isinstance(result, str)

    def test_hash_system_definition_single_equation(self):
        """Test hashing single equation tuple."""
        x, y = sp.symbols("x y", real=True)
        dx = sp.Symbol("dx", real=True)
        dxdt = [(dx, x + y)]
        result = hash_system_definition(dxdt, {})
        assert isinstance(result, str)

    def test_hash_system_definition_consistency(self):
        """Test that equivalent inputs produce same hash."""
        x, y = sp.symbols("x y", real=True)
        dx, dy = sp.symbols("dx dy", real=True)

        # Same equations in different order
        dxdt1 = [(dx, x + y), (dy, x - y)]
        dxdt2 = [(dy, x - y), (dx, x + y)]

        hash1 = hash_system_definition(dxdt1, {})
        hash2 = hash_system_definition(dxdt2, {})

        # Hashes should be identical due to sorting by LHS
        assert hash1 == hash2

    def test_hash_system_definition_different_content(self):
        """Test that different content produces different hashes."""
        x, y = sp.symbols("x y", real=True)
        dx = sp.Symbol("dx", real=True)

        dxdt1 = [(dx, x + y)]
        dxdt2 = [(dx, x - y)]

        hash1 = hash_system_definition(dxdt1, {})
        hash2 = hash_system_definition(dxdt2, {})

        assert hash1 != hash2


class TestParseInput:
    """Test the main parse_system function."""

    def test_parse_system_basic(self, simple_system_defaults):
        """Test basic parsing functionality."""
        (
            states,
            parameters,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            specialised = specialise_all_swept(
                states=states,
                parameters=parameters,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt_str,
            )
            layout = specialised.layout
            equation_map = specialised.equations
            fn_hash = specialised.fn_hash

        assert isinstance(layout, SystemLayout)
        assert isinstance(equation_map, ParsedEquations)
        assert isinstance(fn_hash, str)

        # Check that we got the expected symbols
        assert "one" in layout.state_names
        assert "zebra" in layout.swept_names
        assert "apple" in layout.swept_names

    def test_parse_system_with_list(self, simple_system_defaults):
        """Test parsing with list input."""
        (
            states,
            parameters,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            specialised = parse_system(
                states=states,
                parameters=parameters,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt_list,
            ).specialise()
            equation_map = specialised.equations

        assert isinstance(equation_map, ParsedEquations)
        assert len(equation_map) > 0

    def test_parse_system_with_user_functions(self):
        """Test parsing with user-defined functions."""
        states = ["x"]
        parameters = ["a"]
        observables = ["y"]
        drivers = []
        dxdt = ["dx = custom_func(x)", "y = x"]
        user_functions = {"custom_func": lambda x: x**2}

        specialised = parse_system(
            states=states,
            parameters=parameters,
            observables=observables,
            drivers=drivers,
            user_functions=user_functions,
            dxdt=dxdt,
        ).specialise()
        funcs = specialised.functions

        assert funcs["custom_func"] == user_functions["custom_func"]

    def test_parse_system_includes_time_symbol(self):
        """Ensure ``t`` is always present without explicit declaration."""

        specialised = parse_system(
            dxdt=["dx = t"],
            states={"x": 0.0},
            observables=[],
            parameters={},
            drivers=[],
            strict=True,
        ).specialise()
        funcs = specialised.functions
        equation_map = specialised.equations

        assert equation_map.non_observable_equations() == [
            (ir.sym("dx"), ir.sym("t"))
        ]
        assert funcs == {}

    def test_parse_system_invalid_dxdt_type(self):
        """Test error with invalid dxdt type."""
        states = ["x"]
        parameters = ["a"]
        observables = []
        drivers = []
        dxdt = 123  # Invalid type

        with pytest.raises(TypeError):
            parse_system(
                states=states,
                parameters=parameters,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt,
            ).specialise()

    def test_parse_system_empty_lines_filtered(self):
        """Test that empty lines are filtered out."""
        states = ["x"]
        parameters = ["a"]
        observables = ["y"]
        drivers = []
        dxdt = ["dx = x + a", "", "  ", "y = x"]  # Contains empty lines
        specialised = specialise_all_swept(
            states=states,
            parameters=parameters,
            observables=observables,
            drivers=drivers,
            dxdt=dxdt,
        )
        equation_map = specialised.equations

        dx_equations = equation_map.non_observable_equations()
        assert dx_equations[0][0] is ir.sym("dx")
        assert dx_equations[0][1] is ir.add(ir.sym("x"), ir.sym("a"))
        observable_equations = list(equation_map.observables)
        assert observable_equations[0][0] is ir.sym("y")
        assert observable_equations[0][1] is ir.sym("x")

    def test_parse_system_indexed_variable_equivalence(self):
        """Indexed tokens are parsed equivalently to scalar naming."""

        states = {"x0": 0.0, "x1": 0.0}
        parameters = {"parameters0": 1.0, "parameters1": 2.0}
        observables = []
        drivers = []
        indexed_dxdt = [
            "dx[0] = parameters[0] * x[1]",
            "dx[1] = parameters[1] * x[0]",
        ]
        scalar_dxdt = [
            "dx0 = parameters0 * x1",
            "dx1 = parameters1 * x0",
        ]

        indexed_equations = parse_system(
            states=states,
            parameters=parameters,
            observables=observables,
            drivers=drivers,
            dxdt=indexed_dxdt,
            strict=True,
        ).specialise().equations

        scalar_equations = parse_system(
            states=states,
            parameters=parameters,
            observables=observables,
            drivers=drivers,
            dxdt=scalar_dxdt,
            strict=True,
        ).specialise().equations

        assert indexed_equations.ordered == scalar_equations.ordered

    def test_parse_system_mixed_derivatives_and_auxiliaries(self):
        """Test system with derivatives and d-prefixed auxiliaries."""
        states = ["x", "y"]
        parameters = ["k"]
        observables = []
        drivers = []
        dxdt = [
            "dx = -k * x",
            "dy = k * x",
            "delta = x + y",  # auxiliary, not derivative of elta
            "damping = delta * k",  # auxiliary referencing delta
        ]

        specialised = parse_system(
            states=states,
            parameters=parameters,
            observables=observables,
            drivers=drivers,
            dxdt=dxdt,
            strict=True,
        ).specialise()
        layout = specialised.layout
        parsed_eqs = specialised.equations

        # Verify correct categorization
        assert "x" in layout.state_names
        assert "y" in layout.state_names
        assert "elta" not in layout.state_names  # NOT a state
        assigned = {lhs.name for lhs, _ in parsed_eqs.ordered}
        assert "delta" in assigned  # auxiliary
        assert "damping" in assigned  # auxiliary
        assert len(parsed_eqs.state_derivatives) == 2


class TestIntegrationWithFixtures:
    """Test integration with conftest fixtures."""

    def test_with_simple_system_defaults(self, simple_system_defaults):
        """Test parsing the fixture system."""
        (
            states,
            parameters,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # Test both string and list versions
            result1 = parse_system(
                states=states,
                parameters=parameters,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt_str,
            ).specialise()

            result2 = parse_system(
                states=states,
                parameters=parameters,
                observables=observables,
                drivers=drivers,
                dxdt=dxdt_list,
            ).specialise()

        # Results should be equivalent
        assert result1.fn_hash == result2.fn_hash
        assert result1.equations == result2.equations

    def test_equation_parsing_with_fixture(self, simple_system_defaults):
        """Test specific equation parsing behaviors with fixture."""
        (
            states,
            parameters,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        parsed = parse_system(
            states=states,
            parameters=parameters,
            observables=observables,
            drivers=drivers,
            dxdt=dxdt_str,
            strict=True,
        )
        specialised = parsed.specialise()
        equation_map = specialised.equations

        assigned_to = [expr[0] for expr in equation_map]
        # Check that equations were parsed correctly
        assert ir.sym("done") in assigned_to
        # The dfoo derivative equation keeps its observable-defined
        # form on the observables pass side.
        eqs = {lhs.name: rhs for lhs, rhs in equation_map.observables}
        assert ir.sym("zoo") in ir.free_atoms(eqs["safari"])


class TestNonStrictInput:
    """Test non-strict input parsing."""

    def test_simple(self, simple_system_defaults):
        (
            states,
            parameters,
            drivers,
            observables,
            dxdt_str,
            dxdt_list,
        ) = simple_system_defaults

        with pytest.raises(ValueError, match="strict"):
            parse_system(dxdt=dxdt_str, strict=True).specialise()
        specialised = specialise_all_swept(dxdt=dxdt_str, strict=False)
        layout = specialised.layout
        equation_map = specialised.equations
        assert "apple" in layout.swept_names
        assert "zebra" in layout.swept_names
        assert "driver1" in layout.swept_names
        assert "one" in layout.state_names
        assert "safari" not in layout.observable_names
        assert "uninited" not in layout.observable_names
        assigned = {lhs.name for lhs, _ in equation_map.ordered}
        assert "safari" in assigned
        assert "uninited" in assigned
        assert "dfoo" in layout.dxdt_names


class TestFunctions:
    """Test passing of functions in text in equations"""

    def test_sympyfuncs(self):
        """Add equations with some simple sympy-known functions in them and no
        user input
        """
        eqs = ("dx = sin(a) + exp(b)", "dy = min(c,d) + log(e)")
        specialised = specialise_all_swept(
            dxdt=eqs
        )
        eq_map = specialised.equations
        code = print_cuda_multiple(
            eq_map, function_aliases=eq_map.function_aliases
        )
        assert code == [
            "dx = math.exp(b) + math.sin(a)",
            "dy = min(c, d) + math.log(e)",
        ]

    def test_userfunc_priority(self):
        """Add equations with some simple user-defined functions in them
        that clobber sympy functions, test that the user-defined function
        is called"""

        def custom_func(x):
            return x**2

        userfuncs = {"ex_squared": custom_func, "exp": lambda x: math.exp(x)}

        eqs = ["dx = exp(a) + exp(b)", "dy = x"]
        specialised = specialise_all_swept(
            dxdt=eqs, user_functions=userfuncs
        )
        eq_map = specialised.equations
        code = print_cuda_multiple(
            eq_map, function_aliases=eq_map.function_aliases
        )

        assert code == [
            "dx = precision(exp(a)) + precision(exp(b))",
            "dy = x",
        ]

    def test_device_userfunc_derivative_mapping(self):
        """Ensure device-like user functions use provided derivative name in
        JVP code without CUDA runtime.
        """

        # Pseudo device function: has .targetoptions['device']=True and is
        # callable
        class MyFuncDevice:
            targetoptions = {"device": True}

            def __call__(self, *args, **kwargs):
                return 0

        def myfunc_grad(
            a, b, index
        ):  # derivative callable name should appear in code
            return 0

        userfuncs = {"myfunc": MyFuncDevice()}
        userfunc_grads = {"myfunc": myfunc_grad}
        eqs = ["dx = myfunc(x, y)", "dy = x"]
        specialised = parse_system(
            states=["x", "y"],
            parameters=[],
            observables=[],
            drivers=[],
            dxdt=eqs,
            user_functions=userfuncs,
            user_function_derivatives=userfunc_grads,
        ).specialise()
        layout = specialised.layout
        eq_map = specialised.equations
        # Base code should reference the function name
        lines = print_cuda_multiple(
            eq_map, function_aliases=eq_map.function_aliases
        )
        assert any("myfunc(x, y)" in ln for ln in lines)
        # Operator-apply code calls the provided derivative name
        code = generate_linear_operator_code(
            equations=eq_map, layout=layout
        )
        assert "myfunc_grad(" in code


class TestSympyInputPathway:
    """Integration tests for SymPy input pathway."""

    def test_simple_ode_sympy_equality(self):
        """Test simple ODE via SymPy Equality input."""
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")

        dxdt = [sp.Eq(dx, -k * x)]

        specialised = specialise_all_swept(
            dxdt=dxdt, states=["x"], parameters=["k"], strict=True
        )
        parsed_eqs = specialised.equations

        assert len(parsed_eqs.state_derivatives) == 1
        assert parsed_eqs.state_derivatives[0][0].name == "dx"
        rhs_syms = ir.free_atoms(parsed_eqs.state_derivatives[0][1])
        assert any(s.name == "k" for s in rhs_syms)
        assert any(s.name == "x" for s in rhs_syms)

    def test_simple_ode_sympy_tuple(self):
        """Test simple ODE via tuple input."""
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")

        dxdt = [(dx, -k * x)]

        specialised = parse_system(
            dxdt=dxdt, states=["x"], parameters=["k"], strict=True
        ).specialise()
        parsed_eqs = specialised.equations

        assert len(parsed_eqs.state_derivatives) == 1
        assert parsed_eqs.state_derivatives[0][0].name == "dx"

    def test_ode_with_observables_sympy(self):
        """Test ODE with observables via SymPy input."""
        x, y, k = sp.symbols("x y k")
        dx, dy, z = sp.symbols("dx dy z")

        dxdt = [sp.Eq(dx, -k * x), sp.Eq(dy, k * x), sp.Eq(z, x + y)]

        specialised = parse_system(
            dxdt=dxdt,
            states=["x", "y"],
            parameters=["k"],
            observables=["z"],
            strict=True,
        ).specialise()
        parsed_eqs = specialised.equations

        assert len(parsed_eqs.state_derivatives) == 2
        assert len(parsed_eqs.observables) == 1
        assert parsed_eqs.observables[0][0].name == "z"

    def test_ode_with_user_functions_sympy(self):
        """Test ODE with user functions via SymPy input."""
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")

        custom_func = sp.Function("custom_func")

        dxdt = [sp.Eq(dx, -k * custom_func(x))]

        def custom_impl(val):
            return val**2

        specialised = parse_system(
            dxdt=dxdt,
            states=["x"],
            parameters=["k"],
            user_functions={"custom_func": custom_impl},
            strict=True,
        ).specialise()
        funcs = specialised.functions
        parsed_eqs = specialised.equations

        assert "custom_func" in funcs
        assert len(parsed_eqs.state_derivatives) == 1

    def test_sympy_vs_string_equivalence(self):
        """Test that SymPy and string input produce same results."""
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")
        dxdt_sympy = [sp.Eq(dx, -k * x)]

        dxdt_string = "dx = -k * x"

        result_sympy = parse_system(
            dxdt=dxdt_sympy, states=["x"], parameters=["k"], strict=True
        ).specialise()

        result_string = parse_system(
            dxdt=dxdt_string, states=["x"], parameters=["k"], strict=True
        ).specialise()

        sympy_eq = result_sympy.equations.state_derivatives[0]
        string_eq = result_string.equations.state_derivatives[0]

        assert str(sympy_eq[0]) == str(string_eq[0])
        assert str(sympy_eq[1]) == str(string_eq[1])
        assert result_sympy.fn_hash == result_string.fn_hash

    def test_sympy_infer_states_non_strict(self):
        """Test state inference in non-strict mode."""
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")

        dxdt = [sp.Eq(dx, -k * x)]

        specialised = parse_system(
            dxdt=dxdt, states=[], parameters=["k"], strict=False
        ).specialise()
        layout = specialised.layout
        parsed_eqs = specialised.equations

        assert "x" in layout.state_names
        assert len(parsed_eqs.state_derivatives) == 1

    def test_sympy_infer_parameters_non_strict(self):
        """Test parameter inference from RHS symbols."""
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")

        dxdt = [sp.Eq(dx, -k * x)]

        specialised = specialise_all_swept(
            dxdt=dxdt, states=["x"], parameters=[], strict=False
        )
        layout = specialised.layout

        assert "k" in layout.swept_names

    def test_sympy_user_functions_symbols_dict(self):
        """Test user functions are properly added to symbols dict in SymPy
        pathway.
        """
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")

        custom_func = sp.Function("custom_func")
        dxdt = [sp.Eq(dx, -k * custom_func(x))]

        def custom_impl(val):
            return val**2

        specialised = parse_system(
            dxdt=dxdt,
            states=["x"],
            parameters=["k"],
            user_functions={"custom_func": custom_impl},
            strict=True,
        ).specialise()
        funcs = specialised.functions
        parsed_eqs = specialised.equations

        assert funcs["custom_func"] is custom_impl
        # Verify user function in symbols dict
        # Verify no alias map for SymPy input (aliases only for string pathway)
        assert parsed_eqs.function_aliases == {"custom_func": "custom_func"}

    def test_sympy_derivative_lhs_equality(self):
        """Test canonical SymPy form with sp.Eq(sp.Derivative(...), ...)."""
        x, k, t = sp.symbols("x k t")

        # Canonical SymPy form for ODEs
        dxdt = [sp.Eq(sp.Derivative(x, t), -k * x)]

        specialised = specialise_all_swept(
            dxdt=dxdt, states=["x"], parameters=["k"], strict=True
        )
        parsed_eqs = specialised.equations

        assert len(parsed_eqs.state_derivatives) == 1
        lhs, rhs = parsed_eqs.state_derivatives[0]
        assert lhs.name == "dx"
        assert rhs is from_sympy(-k * x)

    def test_sympy_derivative_lhs_tuple(self):
        """Test tuple form with sp.Derivative as LHS."""
        x, k, t = sp.symbols("x k t")

        # Tuple form with Derivative
        dxdt = [(sp.Derivative(x, t), -k * x)]

        specialised = specialise_all_swept(
            dxdt=dxdt, states=["x"], parameters=["k"], strict=True
        )
        parsed_eqs = specialised.equations

        assert len(parsed_eqs.state_derivatives) == 1
        lhs, rhs = parsed_eqs.state_derivatives[0]
        assert lhs.name == "dx"
        assert rhs is from_sympy(-k * x)

    def test_sympy_derivative_multiple_states(self):
        """Test multiple states with Derivative form."""
        x, y, k, t = sp.symbols("x y k t")
        z = sp.Symbol("z")

        dxdt = [
            sp.Eq(sp.Derivative(x, t), -k * x),
            sp.Eq(sp.Derivative(y, t), k * x),
            sp.Eq(z, x + y),
        ]

        specialised = parse_system(
            dxdt=dxdt,
            states=["x", "y"],
            parameters=["k"],
            observables=["z"],
            strict=True,
        ).specialise()
        parsed_eqs = specialised.equations

        assert len(parsed_eqs.state_derivatives) == 2
        assert len(parsed_eqs.observables) == 1

        # Check state derivatives
        state_lhs = [
            lhs.name for lhs, _ in parsed_eqs.state_derivatives
        ]
        assert "dx" in state_lhs
        assert "dy" in state_lhs

        # Check observable
        obs_lhs, obs_rhs = parsed_eqs.observables[0]
        assert obs_lhs.name == "z"
        assert obs_rhs is from_sympy(x + y)


class TestHashConsistency:
    """Test hash consistency across input pathways."""

    def test_hash_consistency_string_vs_sympy_order(self):
        """Verify string and SymPy inputs with different order produce same
        hash.

        The hash should be computed from the canonical ParsedEquations form,
        which sorts equations alphabetically by LHS symbol name. This test
        verifies that reordering equations does not affect the hash.
        """
        # String input: dx first, then dy
        result_string = parse_system(
            dxdt=["dx = -k*x", "dy = k*x"],
            states=["x", "y"],
            parameters=["k"],
            observables=[],
            drivers=[],
        ).specialise()

        # SymPy input with REVERSED order: dy first, then dx
        x, y, k = sp.symbols("x y k", real=True)
        dx, dy = sp.symbols("dx dy", real=True)
        result_sympy = parse_system(
            dxdt=[sp.Eq(dy, k * x), sp.Eq(dx, -k * x)],
            states=["x", "y"],
            parameters=["k"],
            observables=[],
            drivers=[],
        ).specialise()

        # Hash should be identical regardless of input order
        fn_hash_string = result_string.fn_hash
        fn_hash_sympy = result_sympy.fn_hash
        assert fn_hash_string == fn_hash_sympy

    def test_hash_computed_after_parsing(self):
        """Verify hash correctly reflects parsed structure, not raw input.

        The hash is computed from ParsedEquations, ensuring identical systems
        produce identical hashes even with different input ordering.
        """
        # Order A: dx first
        result_a = parse_system(
            dxdt=["dx = -x", "dy = x"],
            states=["x", "y"],
            parameters=[],
            observables=[],
            drivers=[],
        ).specialise()

        # Order B: dy first (reversed)
        result_b = parse_system(
            dxdt=["dy = x", "dx = -x"],
            states=["x", "y"],
            parameters=[],
            observables=[],
            drivers=[],
        ).specialise()

        # Hash should be identical since equations define the same system
        assert result_a.fn_hash == result_b.fn_hash

    def test_sympy_ambiguous_prefix_not_state_is_auxiliary(self):
        """SymPy: delta_i symbol without state elta_i is auxiliary."""
        x, k = sp.symbols("x k")
        dx = sp.Symbol("dx")
        delta_i = sp.Symbol("delta_i")

        dxdt = [sp.Eq(dx, -k * x), sp.Eq(delta_i, x + 1)]

        specialised = parse_system(
            dxdt=dxdt, states=["x"], parameters=["k"], strict=True
        ).specialise()
        layout = specialised.layout
        parsed_eqs = specialised.equations

        # delta_i should NOT create state elta_i
        assert "elta_i" not in layout.state_names
        # delta_i should be in auxiliaries
        assert "delta_i" in {lhs.name for lhs, _ in parsed_eqs.ordered}

    def test_sympy_ambiguous_prefix_is_state_is_derivative(self):
        """SymPy: delta symbol with state elta is derivative."""
        elta, k = sp.symbols("elta k")
        delta = sp.Symbol("delta")

        dxdt = [sp.Eq(delta, -k * elta)]

        specialised = parse_system(
            dxdt=dxdt, states=["elta"], parameters=["k"], strict=True
        ).specialise()
        parsed_eqs = specialised.equations

        assert len(parsed_eqs.state_derivatives) == 1
        assert parsed_eqs.state_derivatives[0][0].name == "delta"


class TestDerivativeNotation:
    """State-aware derivative detection through the public parser."""

    def test_basic_derivative_with_declared_state(self):
        """dx = ... with state x declared is treated as derivative."""
        specialised = parse_system(
            dxdt=["dx = -k * x"],
            states=["x"],
            parameters=["k"],
            strict=True,
        ).specialise()
        equations = specialised.equations
        assert equations.mass_matrix is None
        assert len(equations.state_derivatives) == 1
        assert not equations.auxiliaries

    def test_ambiguous_prefix_not_a_state_treated_as_auxiliary(self):
        """delta_i = ... with no state elta_i is auxiliary."""
        specialised = parse_system(
            dxdt=["dx = -k * x", "delta_i = x + 1"],
            states=["x"],
            parameters=["k"],
            strict=True,
        ).specialise()
        layout = specialised.layout
        equations = specialised.equations
        assert ir.sym("delta_i") in equations.auxiliary_symbols
        assert "elta_i" not in layout.state_names

    def test_ambiguous_prefix_is_a_state_treated_as_derivative(self):
        """delta = ... with state elta declared is derivative of elta."""
        specialised = parse_system(
            dxdt=["delta = -k * elta"],
            states=["elta"],
            parameters=["k"],
            strict=True,
        ).specialise()
        layout = specialised.layout
        equations = specialised.equations
        assert len(equations.state_derivatives) == 1
        assert "delta" in layout.dxdt_names

    def test_function_notation_with_declared_state(self):
        """d(x, t) = ... with state x declared is derivative."""
        specialised = parse_system(
            dxdt=["d(x, t) = -k * x"],
            states=["x"],
            parameters=["k"],
            strict=True,
        ).specialise()
        equations = specialised.equations
        assert len(equations.state_derivatives) == 1
        assert not equations.auxiliaries

    def test_function_notation_undeclared_state_strict_raises(self):
        """d(x, t) = ... with no state x in strict mode raises."""
        with pytest.raises(ValueError, match="No state called x"):
            parse_system(
                dxdt=["d(x, t) = -k * x"],
                states=[],
                parameters=["k"],
                strict=True,
            ).specialise()

    def test_function_notation_undeclared_state_non_strict_infers(self):
        """d(x, t) = ... with no state x in non-strict infers x."""
        specialised = parse_system(
            dxdt=["d(x, t) = -k * x"],
            states=[],
            parameters=["k"],
            strict=False,
        ).specialise()
        layout = specialised.layout
        assert "x" in layout.state_names

    def test_non_strict_state_inference_from_d_prefix(self):
        """dx = ... with no state x in non-strict infers x as state."""
        specialised = parse_system(
            dxdt=["dx = -k * x"],
            states=[],
            parameters=["k"],
            strict=False,
        ).specialise()
        layout = specialised.layout
        assert "x" in layout.state_names

    def test_non_strict_auxiliary_not_inferred_as_state(self):
        """delta = ... with no state elta in non-strict is auxiliary."""
        specialised = parse_system(
            dxdt=["dx = -k * x", "delta = x + 1"],
            states=["x"],
            parameters=["k"],
            strict=False,
        ).specialise()
        layout = specialised.layout
        equations = specialised.equations
        assert ir.sym("delta") in equations.auxiliary_symbols
        assert "elta" not in layout.state_names

    def test_function_notation_with_whitespace(self):
        """d( x , t ) = ... with extra whitespace works."""
        specialised = parse_system(
            dxdt=["d( x , t ) = -k * x"],
            states=["x"],
            parameters=["k"],
            strict=True,
        ).specialise()
        equations = specialised.equations
        assert len(equations.state_derivatives) == 1

    def test_single_letter_d_treated_as_auxiliary(self):
        """d = ... alone is treated as auxiliary, not derivative."""
        specialised = parse_system(
            dxdt=["dx = -k * x", "d = x + 1"],
            states=["x"],
            parameters=["k"],
            strict=True,
        ).specialise()
        equations = specialised.equations
        assert ir.sym("d") in equations.auxiliary_symbols


class TestParsedEquationsAccessors:
    """Cover ParsedEquations indexing and property accessors."""

    def _parsed(self):
        specialised = parse_system(
            dxdt=["dx = x + a", "obs = x"],
            states=["x"],
            parameters=["a"],
            observables=["obs"],
            strict=True,
        ).specialise()
        equations = specialised.equations
        return equations

    def test_getitem_returns_equation(self):
        """Indexing returns the equation in original order."""
        parsed = self._parsed()
        lhs, rhs = parsed[0]
        assert lhs.name == "dx"

    def test_state_symbols_property(self):
        """state_symbols exposes the derivative-output symbols."""
        parsed = self._parsed()
        assert ir.sym("dx") in parsed.state_symbols
