"""Tests for function_parser module and end-to-end callable integration."""

import ast

import pytest
import sympy as sp

from cubie.odesystems.symbolic.codegen.linear_operators import (
    generate_operator_apply_code,
)
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.expr import _children
from cubie.odesystems.symbolic.engine.from_sympy import to_sympy
from cubie.odesystems.symbolic.indexedbasemaps import (
    IndexedBaseMap,
    IndexedBases,
)
from cubie.odesystems.symbolic.parsing.function_inspector import (
    AstToSympyConverter,
)
from cubie.odesystems.symbolic.parsing.function_parser import (
    _resolve_called_functions,
    _unpack_return,
    infer_function_states,
    parse_function_input,
)
from cubie.odesystems.symbolic.parsing.parser import (
    EquationWarning,
    parse_input,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


def _walk(node):
    """Yield ``node`` and every descendant IR node via a stack."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(_children(current))


def _has_piecewise(node):
    """Return whether an IR expression contains a Piecewise node."""
    return any(isinstance(n, ir.Piecewise) for n in _walk(node))


class TestParseInput:
    """End-to-end tests through parse_input with callable dxdt."""

    def test_simple_exponential_decay(self):
        """Single-state exponential decay via function."""
        def f(t, y):
            return [-0.1 * y[0]]

        index_map, syms, fns, eqs, h, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )
        assert len(eqs.state_derivatives) == 1
        assert len(index_map.state_names) == 1

    def test_two_state_with_params(self):
        """Damped oscillator with parameter k."""
        def f(t, y, p):
            return [-p[0] * y[1], y[0]]

        index_map, syms, fns, eqs, h, _, *_ = parse_input(
            dxdt=f,
            states={"velocity": 1.0, "position": 0.0},
            parameters={"k": 0.5},
        )
        assert len(eqs.state_derivatives) == 2

    def test_named_states_positional(self):
        """Named states accessed by positional index."""
        def f(t, y):
            return [-y[0], y[0] - y[1]]

        index_map, syms, fns, eqs, h, _, *_ = parse_input(
            dxdt=f,
            states={"a": 0.0, "b": 1.0},
        )
        assert len(eqs.state_derivatives) == 2

    def test_string_indexed_states(self):
        """String subscript state access."""
        def f(t, y):
            return [-y["velocity"]]

        index_map, syms, fns, eqs, h, _, *_ = parse_input(
            dxdt=f,
            states={"velocity": 1.0},
        )
        assert len(eqs.state_derivatives) == 1

    def test_constants_attribute_access(self):
        """Constants accessed via attribute pattern."""
        def f(t, y, c):
            return [-c.damping * y[0]]

        index_map, syms, fns, eqs, h, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
            constants={"damping": 0.1},
        )
        assert len(eqs.state_derivatives) == 1

    def test_constants_string_subscript(self):
        """Constants accessed via string subscript."""
        def f(t, y, c):
            return [-c["damping"] * y[0]]

        index_map, syms, fns, eqs, h, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
            constants={"damping": 0.1},
        )
        assert len(eqs.state_derivatives) == 1

    def test_equivalence_with_string_input(self):
        """Function input produces equivalent equations to string input."""
        def f(t, y):
            return [-0.1 * y[0]]

        _, _, _, eqs_func, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )

        _, _, _, eqs_str, _, _, *_ = parse_input(
            dxdt="dx = -0.1 * x",
            states={"x": 1.0},
        )

        assert len(eqs_func.state_derivatives) == len(
            eqs_str.state_derivatives
        )
        # Both should have one derivative equation for dx
        func_rhs = to_sympy(eqs_func.state_derivatives[0][1])
        str_rhs = to_sympy(eqs_str.state_derivatives[0][1])
        assert sp.simplify(func_rhs - str_rhs) == 0

    def test_math_functions(self):
        """Math functions like sin are converted."""
        def f(t, y):
            from math import sin
            return [sin(y[0])]

        index_map, syms, fns, eqs, h, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )
        assert len(eqs.state_derivatives) == 1

    def test_local_alias(self):
        """Local variable aliasing a state does not create auxiliary."""
        def f(t, y):
            v = y[0]
            return [-0.1 * v]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )
        # Alias should not appear as auxiliary
        assert len(eqs.auxiliaries) == 0
        assert len(eqs.state_derivatives) == 1


    def test_augmented_assignment_correctness(self):
        """Augmented assignment produces correct summed expression."""
        def f(t, y):
            total = y[0]
            total += y[1]
            total += y[2]
            return [-total, total * 0.5, -total * 0.25]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"a": 1.0, "b": 0.5, "c": 0.0},
        )
        assert len(eqs.state_derivatives) == 3
        # ``total`` becomes an auxiliary: total = a + b + c
        # Derivatives reference the auxiliary symbol.
        assert len(eqs.auxiliaries) == 1
        aux_lhs, aux_rhs = eqs.auxiliaries[0]
        assert aux_lhs.name == "total"
        a = sp.Symbol("a", real=True)
        b = sp.Symbol("b", real=True)
        c = sp.Symbol("c", real=True)
        aux_rhs_sp = to_sympy(aux_rhs)
        assert sp.simplify(aux_rhs_sp - (a + b + c)) == 0, (
            f"Expected a + b + c, got {aux_rhs_sp}"
        )


    def test_for_loop_unrolling(self):
        """For-loop over range() unrolled to correct equations."""
        def f(t, y, p):
            total = 0.0
            for i in range(3):
                total += y[i] * p[i]
            return [-total, total * 0.5, -total * 0.25]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"a": 1.0, "b": 0.5, "c": 0.0},
            parameters={"p0": 0.1, "p1": 0.2, "p2": 0.3},
        )
        assert len(eqs.state_derivatives) == 3
        # ``total`` auxiliary should contain all three terms
        assert len(eqs.auxiliaries) == 1
        aux_rhs = to_sympy(eqs.auxiliaries[0][1])
        a = sp.Symbol("a", real=True)
        b = sp.Symbol("b", real=True)
        c = sp.Symbol("c", real=True)
        p0 = sp.Symbol("p0", real=True)
        p1 = sp.Symbol("p1", real=True)
        p2 = sp.Symbol("p2", real=True)
        expected = a * p0 + b * p1 + c * p2
        assert sp.simplify(aux_rhs - expected) == 0, (
            f"Expected {expected}, got {aux_rhs}"
        )

    def test_tuple_unpacking(self):
        """Tuple unpacking in assignment works."""
        def f(t, y):
            a, b = y[0], y[1]
            return [-a, b]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0, "v": 0.5},
        )
        assert len(eqs.state_derivatives) == 2

    def test_if_else_piecewise(self):
        """If/else produces Piecewise in auxiliary equations."""
        def f(t, y):
            if y[0] > 0:
                result = y[0]
            else:
                result = -y[0]
            return [-result]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )
        # The Piecewise appears in the auxiliary for 'result'
        assert len(eqs.auxiliaries) == 1
        _, aux_rhs = eqs.auxiliaries[0]
        assert _has_piecewise(aux_rhs)

    def test_if_elif_else_piecewise(self):
        """If/elif/else chain produces multi-branch Piecewise."""
        def f(t, y):
            if y[0] > 1.0:
                rate = 2.0 * y[0]
            elif y[0] > 0.0:
                rate = y[0]
            else:
                rate = 0.0
            return [-rate]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
        )
        assert len(eqs.auxiliaries) == 1
        _, aux_rhs = eqs.auxiliaries[0]
        assert _has_piecewise(aux_rhs)

    def test_elif_only_container_access(self):
        """Container access appearing only in an elif branch resolves."""
        def f(t, y, p):
            if t > 1.0:
                rate = p.k1 * y[0]
            elif t > 0.5:
                rate = p.k2 * y[0]
            else:
                rate = 0.0
            return [-rate]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
            parameters={"k1": 1.0, "k2": 2.0},
        )
        assert len(eqs.auxiliaries) == 1
        _, aux_rhs = eqs.auxiliaries[0]
        assert _has_piecewise(aux_rhs)
        assert ir.sym("k2") in ir.free_atoms(aux_rhs)

    def test_for_loop_inside_if_branch(self):
        """A for-loop inside an if-branch unrolls into the branch."""
        def f(t, y, p):
            if t > 0.0:
                total = 0.0
                for i in range(2):
                    total += p[i] * y[i]
            else:
                total = 1.0
            return [-total, total]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"a": 1.0, "b": 0.5},
            parameters={"k0": 0.1, "k1": 0.2},
        )
        assert len(eqs.auxiliaries) == 1
        _, aux_rhs = eqs.auxiliaries[0]
        assert _has_piecewise(aux_rhs)
        a = sp.Symbol("a", real=True)
        b = sp.Symbol("b", real=True)
        k0 = sp.Symbol("k0", real=True)
        k1 = sp.Symbol("k1", real=True)
        branch_expr = to_sympy(aux_rhs.pairs[0][0])
        expected = k0 * a + k1 * b
        assert sp.simplify(branch_expr - expected) == 0, (
            f"Expected {expected}, got {branch_expr}"
        )

    def test_return_inside_branch_raises(self):
        """A return inside an if-branch raises with guidance."""
        def f(t, y):
            if y[0] > 0:
                return [-y[0]]
            return [y[0]]

        with pytest.raises(NotImplementedError, match="return"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_while_inside_branch_raises(self):
        """A while-loop inside an if-branch raises."""
        def f(t, y):
            if y[0] > 0:
                a = y[0]
                while a > 1.0:
                    a = a - 1.0
            else:
                a = 0.0
            return [-a]

        with pytest.raises(NotImplementedError, match="While"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_for_else_raises(self):
        """A for/else loop raises."""
        def f(t, y):
            total = 0.0
            for i in range(2):
                total += y[i]
            else:
                total += 1.0
            return [-total, total]

        with pytest.raises(NotImplementedError, match="for/else"):
            parse_input(dxdt=f, states={"a": 1.0, "b": 0.5})

    def test_try_except_raises(self):
        """try/except raises instead of merging branch assignments."""
        def f(t, y):
            try:
                a = y[0]
            except ValueError:
                a = 0.0
            return [-a]

        with pytest.raises(NotImplementedError, match="try/except"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_match_statement_raises(self):
        """match statements raise instead of merging case bodies."""
        def f(t, y):
            match int(t):
                case 0:
                    a = y[0]
                case _:
                    a = -y[0]
            return [a]

        with pytest.raises(NotImplementedError, match="match"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_nested_def_raises(self):
        """Nested function definitions raise with guidance."""
        def f(t, y):
            def helper(v):
                return -v
            return [helper(y[0])]

        with pytest.raises(
            NotImplementedError, match="user_functions"
        ):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_annotated_assignment(self):
        """An annotated assignment is treated as an assignment."""
        def f(t, y):
            a: float = 2.0 * y[0]
            return [-a]

        _, _, _, eqs, _, _, *_ = parse_input(dxdt=f, states={"x": 1.0})
        assert len(eqs.auxiliaries) == 1
        _, aux_rhs = eqs.auxiliaries[0]
        x = sp.Symbol("x", real=True)
        assert sp.simplify(to_sympy(aux_rhs) - 2.0 * x) == 0


class TestCreateODESystem:
    """Test create_ODE_system with callable dxdt."""

    def test_simple(self):
        """Basic callable creates SymbolicODE."""
        def f(t, y):
            return [-y[0]]

        ode = create_ODE_system(
            dxdt=f,
            states={"x": 1.0},
        )
        assert ode.num_states == 1

    def test_with_parameters(self):
        """Callable with parameter argument."""
        def f(t, y, p):
            return [-p[0] * y[0]]

        ode = create_ODE_system(
            dxdt=f,
            states={"x": 1.0},
            parameters={"k": 0.5},
        )
        assert ode.num_states == 1


class TestDetectInputType:
    """Test that callable detection works in _detect_input_type."""

    def test_callable_detected(self):
        """Callable dxdt returns 'function' type."""
        from cubie.odesystems.symbolic.parsing.parser import (
            _detect_input_type,
        )

        def f(t, y):
            return [-y[0]]

        assert _detect_input_type(f) == "function"

    def test_string_still_works(self):
        """String dxdt still returns 'string' type."""
        from cubie.odesystems.symbolic.parsing.parser import (
            _detect_input_type,
        )
        assert _detect_input_type("dx = -x") == "string"


class TestDriverAccess:
    """Drivers resolve through container arguments (issue #564)."""

    def test_driver_via_parameter_container(self):
        """p.<driver> resolves to the declared driver symbol."""
        def f(t, y, p):
            dx = y.v
            dv = -y.x + p.forcing
            return [dx, dv]

        index_map, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0, "v": 0.0},
            drivers=["forcing"],
        )
        driver_sym = index_map.drivers.symbol_map["forcing"]
        rhs_symbols = set()
        for _, rhs in eqs.state_derivatives:
            rhs_symbols |= ir.free_atoms(rhs)
        assert ir.sym(driver_sym.name) in rhs_symbols
        assert "forcing" not in index_map.parameter_names

    def test_driver_via_dedicated_fourth_argument(self):
        """A dedicated d argument reaches drivers the same way."""
        def f(t, y, p, d):
            dx = y.v
            dv = -p.k * y.x + d.forcing
            return [dx, dv]

        index_map, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0, "v": 0.0},
            parameters={"k": 2.0},
            drivers=["forcing"],
        )
        driver_sym = index_map.drivers.symbol_map["forcing"]
        rhs_symbols = set()
        for _, rhs in eqs.state_derivatives:
            rhs_symbols |= ir.free_atoms(rhs)
        assert ir.sym(driver_sym.name) in rhs_symbols

    def test_driver_string_subscript(self):
        """String subscript access reaches drivers."""
        def f(t, y, p):
            return [-y["x"] + p["forcing"]]

        index_map, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
            drivers=["forcing"],
        )
        driver_sym = index_map.drivers.symbol_map["forcing"]
        rhs = eqs.state_derivatives[0][1]
        assert ir.sym(driver_sym.name) in ir.free_atoms(rhs)

    def test_bare_driver_name_raises_with_hint(self):
        """Bare-name driver references raise with the container hint."""
        def f(t, y, p):
            return [-y.x + forcing]  # noqa: F821

        with pytest.raises(ValueError, match=r"p\.forcing"):
            parse_input(
                dxdt=f,
                states={"x": 1.0},
                drivers=["forcing"],
            )


class TestStateInference:
    """states can be omitted in unambiguous cases (issue #565)."""

    def test_dict_return_infers_names_and_order(self):
        """Dict return keys supply state names and order."""
        def f(t, y, p):
            dx = -p.k * y.x
            dv = y.x
            return {"x": dx, "v": dv}

        index_map, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            parameters={"k": 0.5},
        )
        assert index_map.state_names == ["x", "v"]
        assert len(eqs.state_derivatives) == 2
        assert index_map.states.defaults == {"x": 0.0, "v": 0.0}

    def test_positional_access_synthesizes_names(self):
        """Pure positional access synthesizes indexed state names."""
        def f(t, y):
            return [-0.1 * y[0], y[0] - y[1]]

        index_map, _, _, eqs, _, _, *_ = parse_input(dxdt=f)
        assert index_map.state_names == ["y0", "y1"]
        assert len(eqs.state_derivatives) == 2

    def test_single_expression_return_infers_one_state(self):
        """A scalar return infers a single synthesized state."""
        def f(t, y):
            return -0.5 * y[0]

        index_map, _, _, eqs, _, _, *_ = parse_input(dxdt=f)
        assert index_map.state_names == ["y0"]
        assert len(eqs.state_derivatives) == 1

    def test_list_return_with_named_access_raises(self):
        """List return plus named access is ambiguous without states."""
        def f(t, y):
            return [-y.x, y.x - y.v]

        with pytest.raises(ValueError, match="return a dict"):
            parse_input(dxdt=f)

    def test_strict_without_states_still_raises(self):
        """strict=True keeps requiring explicit states."""
        def f(t, y):
            return [-0.1 * y[0]]

        with pytest.raises(ValueError, match="No state symbols"):
            parse_input(dxdt=f, strict=True)

    def test_inferred_states_match_explicit_equations(self):
        """Inferred dict-return system equals the explicit-states one."""
        def f(t, y, p):
            dx = -p.k * y.x
            return {"x": dx}

        _, _, _, eqs_inferred, _, _, *_ = parse_input(
            dxdt=f, parameters={"k": 0.5}
        )
        _, _, _, eqs_explicit, _, _, *_ = parse_input(
            dxdt=f, parameters={"k": 0.5}, states={"x": 0.0}
        )
        lhs_i, rhs_i = eqs_inferred.state_derivatives[0]
        lhs_e, rhs_e = eqs_explicit.state_derivatives[0]
        assert lhs_i == lhs_e
        assert sp.simplify(to_sympy(rhs_i) - to_sympy(rhs_e)) == 0


class TestUndeclaredSymbols:
    """Undeclared symbols die or infer at parse time (issue #563)."""

    def test_unknown_bare_name_raises(self):
        """An undeclared bare name raises at parse time."""
        def f(t, y, p):
            dx = -k_leak * y.x  # noqa: F821
            return [dx]

        with pytest.raises(ValueError, match="k_leak"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_bare_declared_parameter_raises_with_hint(self):
        """A declared parameter used bare names its container access."""
        def f(t, y, p):
            return [-mu * y.x]  # noqa: F821

        with pytest.raises(ValueError, match=r"p\.mu"):
            parse_input(
                dxdt=f, states={"x": 1.0}, parameters={"mu": 2.0}
            )

    def test_bare_driver_hints_dedicated_driver_argument(self):
        """A bare driver name hints the trailing container argument."""
        def f(t, y, p, d):
            return [forcing - p.k * y.x]  # noqa: F821

        with pytest.raises(ValueError, match=r"d\.forcing"):
            parse_input(
                dxdt=f,
                states={"x": 1.0},
                parameters={"k": 1.0},
                drivers=["forcing"],
            )

    def test_container_access_infers_parameter(self):
        """Undeclared container access infers a parameter and warns."""
        def f(t, y, p):
            return [-p.k_new * y.x]

        with pytest.warns(EquationWarning, match="k_new"):
            index_map, _, _, _, _, _, *_ = parse_input(
                dxdt=f, states={"x": 1.0}
            )
        assert "k_new" in index_map.parameter_names

    def test_container_access_strict_raises(self):
        """strict=True forbids container-access inference."""
        def f(t, y, p):
            return [-p.k_new * y.x]

        with pytest.raises(ValueError, match="strict"):
            parse_input(dxdt=f, states={"x": 1.0}, strict=True)

    def test_unknown_state_attribute_raises(self):
        """Accessing an undeclared state names the declared states."""
        def f(t, y):
            return [-y.z]

        with pytest.raises(ValueError, match="Declared states"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_state_index_out_of_range_raises(self):
        """Out-of-range positional state access raises."""
        def f(t, y):
            return [-y[3]]

        with pytest.raises(ValueError, match="out of range"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_container_index_out_of_range_raises(self):
        """Out-of-range positional container access raises."""
        def f(t, y, p):
            return [-p[4] * y[0]]

        with pytest.raises(ValueError, match="out of range"):
            parse_input(
                dxdt=f, states={"x": 1.0}, parameters={"k": 1.0}
            )

    def test_state_accessed_via_container_raises(self):
        """A state reached through p.* points back at the state arg."""
        def f(t, y, p):
            return [-p.x]

        with pytest.raises(ValueError, match=r"y\.x"):
            parse_input(dxdt=f, states={"x": 1.0})


class TestUserFunctions:
    """user_functions work in the callable form (issue #562)."""

    def test_nondevice_function_inlined(self):
        """Plain callables are inlined symbolically."""
        def hill(conc, km):
            return conc / (km + conc)

        def f(t, y, p):
            dx = -hill(y.x, p.km)
            return [dx]

        index_map, _, funcs, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
            parameters={"km": 0.5},
            user_functions={"hill": hill},
        )
        assert funcs["hill"] is hill
        x = sp.Symbol("x", real=True)
        km = sp.Symbol("km", real=True)
        rhs = to_sympy(eqs.state_derivatives[0][1])
        assert sp.simplify(rhs - (-x / (km + x))) == 0

    def test_user_function_overrides_known_function(self):
        """A user function shadows a KNOWN_FUNCTIONS entry."""
        def f(t, y):
            return [-exp(y[0])]  # noqa: F821

        def my_exp(v):
            return 2 * v

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
            user_functions={"exp": my_exp},
        )
        x = sp.Symbol("x", real=True)
        rhs = to_sympy(eqs.state_derivatives[0][1])
        assert sp.simplify(rhs + 2 * x) == 0

    def test_device_function_stays_symbolic(self):
        """Device-like callables stay as symbolic calls."""
        class MyFuncDevice:
            targetoptions = {"device": True}

            def __call__(self, *args, **kwargs):
                return 0

        def f(t, y):
            dx = -myfunc(y.x, y.v)  # noqa: F821
            return [dx, y.x]

        index_map, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0, "v": 0.0},
            user_functions={"myfunc": MyFuncDevice()},
        )
        rhs = eqs.state_derivatives[0][1]
        applied = [
            n for n in _walk(rhs)
            if isinstance(n, ir.Call) and n.name == "myfunc"
        ]
        assert len(applied) == 1

    def test_device_function_derivative_in_operator_code(self):
        """Derivative helpers appear in generated operator code."""
        class MyFuncDevice:
            targetoptions = {"device": True}

            def __call__(self, *args, **kwargs):
                return 0

        def myfunc_grad(a, b, index):
            return 0

        def f(t, y):
            dx = -myfunc(y.x, y.v)  # noqa: F821
            return [dx, y.x]

        index_map, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0, "v": 0.0},
            user_functions={"myfunc": MyFuncDevice()},
            user_function_derivatives={"myfunc": myfunc_grad},
        )
        code = generate_operator_apply_code(
            equations=eqs, index_map=index_map
        )
        assert "myfunc_grad(" in code

    def test_unknown_call_still_raises(self):
        """Calls that match nothing raise NotImplementedError."""
        def f(t, y):
            return [-mystery(y[0])]  # noqa: F821

        with pytest.raises(NotImplementedError, match="mystery"):
            parse_input(dxdt=f, states={"x": 1.0})


class TestScalarArguments:
    """Extra args used bare are scalars (SciPy args= convention)."""

    def test_scalar_arg_resolves_declared_parameter(self):
        """A bare extra arg binds to the like-named parameter."""
        def f(t, y, mu):
            return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

        index_map, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0, "v": 0.0},
            parameters={"mu": 1.5},
        )
        rhs_symbols = set()
        for _, rhs in eqs.state_derivatives:
            rhs_symbols |= ir.free_atoms(rhs)
        assert ir.sym("mu") in rhs_symbols

    def test_scalar_arg_resolves_declared_driver(self):
        """A bare extra arg binds to a like-named driver."""
        def f(t, y, k, forcing):
            return [-k * y[0] + forcing]

        index_map, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
            parameters={"k": 0.5},
            drivers=["forcing"],
        )
        driver_sym = index_map.drivers.symbol_map["forcing"]
        rhs = eqs.state_derivatives[0][1]
        assert ir.sym(driver_sym.name) in ir.free_atoms(rhs)

    def test_undeclared_scalar_arg_infers_parameter(self):
        """An undeclared scalar arg infers a parameter and warns."""
        def f(t, y, k_new):
            return [-k_new * y[0]]

        with pytest.warns(EquationWarning, match="k_new"):
            index_map, _, _, _, _, _, *_ = parse_input(
                dxdt=f, states={"x": 1.0}
            )
        assert "k_new" in index_map.parameter_names

    def test_undeclared_scalar_arg_strict_raises(self):
        """strict=True forbids scalar-argument inference."""
        def f(t, y, k_new):
            return [-k_new * y[0]]

        with pytest.raises(ValueError, match="strict"):
            parse_input(dxdt=f, states={"x": 1.0}, strict=True)

    def test_scalar_and_container_args_mix(self):
        """Scalar and container extra args coexist."""
        def f(t, y, mu, p):
            return [-mu * y[0] + p.k]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
            parameters={"mu": 1.0, "k": 2.0},
        )
        rhs = eqs.state_derivatives[0][1]
        assert {ir.sym("mu"), ir.sym("k")} <= ir.free_atoms(rhs)

    def test_unused_extra_arg_ignored(self):
        """An extra arg never referenced infers nothing."""
        def f(t, y, p):
            return [-0.5 * y[0]]

        index_map, _, _, _, _, _, *_ = parse_input(dxdt=f, states={"x": 1.0})
        assert "p" not in index_map.parameter_names

    def test_scalar_arg_matching_state_raises(self):
        """A scalar arg shadowing a state points at the state arg."""
        def f(t, y, x):
            return [-x * y[0]]

        with pytest.raises(ValueError, match=r"y\.x"):
            parse_input(dxdt=f, states={"x": 1.0})


class TestDerivativeAliasReference:
    """Locals named after derivative outputs inline everywhere."""

    def test_observable_referencing_dxdt_alias(self):
        """An observable can reference a dx-named local."""
        def f(t, y):
            dx = -0.5 * y.x
            flux = dx * 2.0  # noqa: F841
            return [dx]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f,
            states={"x": 1.0},
            observables=["flux"],
        )
        assert len(eqs.observables) == 1
        _, obs_rhs = eqs.observables[0]
        x = sp.Symbol("x", real=True)
        assert sp.simplify(to_sympy(obs_rhs) - (-1.0 * x)) == 0


def _single_state_index_map():
    """Return an ``IndexedBases`` with one state ``x`` and empty rest."""
    return IndexedBases.from_user_inputs(
        states={"x": 1.0},
        parameters={},
        constants={},
        observables=[],
        drivers=[],
    )


class TestParseFunctionInputDirect:
    """Direct calls into ``parse_function_input`` and helpers."""

    def test_observables_default_to_empty(self):
        """Omitting ``observables`` yields the standard equation triple."""
        def f(t, y):
            return [-y[0]]

        equation_map, funcs, new_params = parse_function_input(
            f, _single_state_index_map()
        )
        assert len(equation_map) == 1
        assert funcs == {}
        assert new_params == []

    def test_observable_symbol_falls_back_to_new_symbol(self):
        """An observable absent from the index map gets a fresh symbol."""
        def f(t, y):
            flux = 2.0 * y[0]  # noqa: F841
            return [-y[0]]

        equation_map, _, _ = parse_function_input(
            f, _single_state_index_map(), observables=["flux"]
        )
        flux_sym = sp.Symbol("flux", real=True)
        x = sp.Symbol("x", real=True)
        lhs, rhs = next(
            (lhs, rhs)
            for lhs, rhs in equation_map
            if lhs == flux_sym
        )
        assert sp.simplify(rhs - 2.0 * x) == 0

    def test_dict_return_synthesizes_missing_dxdt_symbol(self):
        """A dict return with no matching dxdt entry synthesizes one."""
        states = IndexedBaseMap("state", ["x"], input_defaults=[1.0])
        empty_params = IndexedBaseMap("parameters", [])
        empty_consts = IndexedBaseMap("constants", [])
        empty_obs = IndexedBaseMap("observables", [])
        empty_drivers = IndexedBaseMap("drivers", [])
        empty_dxdt = IndexedBaseMap("out", [])
        index_map = IndexedBases(
            states,
            empty_params,
            empty_consts,
            empty_obs,
            empty_drivers,
            empty_dxdt,
        )

        def f(t, y):
            return {"x": -y[0]}

        equation_map, _, _ = parse_function_input(f, index_map)
        dx_sym = sp.Symbol("dx", real=True)
        lhs, rhs = equation_map[-1]
        assert lhs == dx_sym
        x = sp.Symbol("x", real=True)
        assert sp.simplify(rhs + x) == 0

    def test_unpack_return_without_assignments(self):
        """``_unpack_return`` defaults missing assignments to empty."""
        x = sp.Symbol("x", real=True)
        converter = AstToSympyConverter({"y[0]": x})
        node = ast.parse("[-y[0]]", mode="eval").body
        result = _unpack_return(node, converter)
        assert len(result) == 1
        assert sp.simplify(result[0] + x) == 0

    def test_resolve_called_functions_bare_name(self):
        """A module-qualified call resolves through its bare name."""
        def dummy(a):
            return a

        resolved = _resolve_called_functions(
            {"np.dummy"}, {"dummy": dummy}
        )
        assert resolved["dummy"] is dummy

    def test_infer_states_dict_non_string_key_raises(self):
        """Non-string dict return keys raise during state inference."""
        def f(t, y):
            return {t: -y[0]}

        with pytest.raises(ValueError, match="string literals"):
            infer_function_states(f)


class TestParseFunctionErrors:
    """Error and edge paths reached through ``parse_input``."""

    def test_local_named_like_constant_arg_skipped(self):
        """A local reassigning a constant arg is not an auxiliary."""
        def f(t, y, k):
            k = 2.0  # noqa: F841
            return [-k * y[0]]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f, states={"x": 1.0}, parameters={"k": 1.0}
        )
        assert len(eqs.state_derivatives) == 1
        aux_names = [str(lhs) for lhs, _ in eqs.auxiliaries]
        assert "k" not in aux_names

    def test_local_named_like_state_arg_skipped(self):
        """A local reassigning the state argument is not an auxiliary."""
        def f(t, y):
            y = y[0]
            return [-y]

        _, _, _, eqs, _, _, *_ = parse_input(dxdt=f, states={"x": 1.0})
        assert len(eqs.state_derivatives) == 1
        assert len(eqs.auxiliaries) == 0

    def test_return_length_mismatch_raises(self):
        """A return longer than the state vector raises."""
        def f(t, y):
            return [-y[0], y[0]]

        with pytest.raises(ValueError, match="Return has"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_dict_return_non_constant_key_raises(self):
        """A dict return with a non-literal key raises."""
        def f(t, y):
            return {t: -y[0]}

        with pytest.raises(ValueError, match="string literals"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_dict_return_unknown_state_raises(self):
        """A dict return keyed by an undeclared state raises."""
        def f(t, y):
            return {"z": -y[0]}

        with pytest.raises(ValueError, match="not a declared state"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_unknown_string_state_raises(self):
        """A string subscript for an undeclared state raises."""
        def f(t, y):
            return [-y["z"]]

        with pytest.raises(ValueError, match="Unknown state 'z'"):
            parse_input(dxdt=f, states={"x": 1.0})

    def test_non_constant_container_subscript_raises(self):
        """A computed container subscript raises NotImplementedError."""
        def f(t, y, p):
            return [-p[0 + 0]]

        with pytest.raises(
            NotImplementedError, match="constant subscripts"
        ):
            parse_input(
                dxdt=f, states={"x": 1.0}, parameters={"k": 1.0}
            )

    def test_observable_via_container_raises(self):
        """Accessing an observable through a container argument raises."""
        def f(t, y, p):
            return [-p.flux * y[0]]

        with pytest.raises(ValueError, match="observable"):
            parse_input(
                dxdt=f, states={"x": 1.0}, observables=["flux"]
            )

    def test_repeated_container_inference_reuses_symbol(self):
        """A second access to an inferred container key reuses it."""
        def f(t, y, p):
            return [-p.k_new * y[0] + p.k_new]

        with pytest.warns(EquationWarning, match="k_new"):
            index_map, _, _, eqs, _, _, *_ = parse_input(
                dxdt=f, states={"x": 1.0}
            )
        assert index_map.parameter_names.count("k_new") == 1

    def test_scalar_arg_reuses_inferred_container_symbol(self):
        """A scalar arg matching an inferred container key reuses it."""
        def f(t, y, p, k_new):
            return [-p.k_new * y[0] + k_new]

        with pytest.warns(EquationWarning, match="k_new"):
            index_map, _, _, eqs, _, _, *_ = parse_input(
                dxdt=f, states={"x": 1.0}
            )
        assert index_map.parameter_names.count("k_new") == 1

    def test_string_subscript_alias_resolves(self):
        """A local aliasing a string-subscript state resolves to it."""
        def f(t, y):
            v = y["x"]
            return [-v]

        _, _, _, eqs, _, _, *_ = parse_input(dxdt=f, states={"x": 1.0})
        assert len(eqs.state_derivatives) == 1
        assert len(eqs.auxiliaries) == 0
        x = sp.Symbol("x", real=True)
        rhs = to_sympy(eqs.state_derivatives[0][1])
        assert sp.simplify(rhs + x) == 0

    def test_attribute_access_alias_not_auxiliary(self):
        """A local aliasing ``p.k`` is not emitted as an auxiliary."""
        def f(t, y, p):
            a = p.k
            return [-a * y[0]]

        _, _, _, eqs, _, _, *_ = parse_input(
            dxdt=f, states={"x": 1.0}, parameters={"k": 1.0}
        )
        assert len(eqs.state_derivatives) == 1
        assert len(eqs.auxiliaries) == 0
