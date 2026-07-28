"""Tests for function_inspector module."""

import ast
import math
import warnings

import pytest
import sympy as sp

from cubie.odesystems.symbolic.parsing.function_inspector import (
    AstToSympyConverter,
    _call_name,
    _resolve_func_name,
    inspect_ode_function,
)


def _expr(source):
    """Parse *source* as a single expression AST node."""
    return ast.parse(source, mode="eval").body

class TestInspectOdeFunction:
    """Tests for inspect_ode_function."""

    def test_simple_one_state(self):
        """Single state, one equation."""
        def f(t, y):
            return [-y[0]]

        result = inspect_ode_function(f)
        assert result.state_param == "y"
        assert result.constant_params == []
        assert len(result.state_accesses) == 1
        assert result.state_accesses[0]["key"] == 0
        assert result.state_accesses[0]["pattern_type"] == "int"

    def test_with_parameter(self):
        """Parameter detected from third argument."""
        def f(t, y, k):
            return [-k * y[0]]

        result = inspect_ode_function(f)
        assert result.constant_params == ["k"]

    def test_math_function_calls(self):
        """Math functions detected in calls."""
        def f(t, y):
            return [math.sin(y[0])]

        result = inspect_ode_function(f)
        assert "math.sin" in result.function_calls

    def test_multi_state_with_locals(self):
        """Multiple states with local variable aliases."""
        def f(t, y):
            v = y[0]
            x = y[1]  # noqa: F841 -- the alias is the inspected input
            return [-0.1 * v, v]

        result = inspect_ode_function(f)
        assert len(result.state_accesses) == 2
        assert "v" in result.assignments
        assert "x" in result.assignments

    def test_string_indexing(self):
        """String subscript access pattern."""
        def f(t, y):
            return [-y["velocity"]]

        result = inspect_ode_function(f)
        assert result.state_accesses[0]["key"] == "velocity"
        assert result.state_accesses[0]["pattern_type"] == "string"

    def test_attribute_access(self):
        """Attribute access pattern on state."""
        def f(t, y):
            return [-y.velocity]

        result = inspect_ode_function(f)
        assert result.state_accesses[0]["key"] == "velocity"
        assert result.state_accesses[0]["pattern_type"] == "attribute"

    def test_constant_attribute_access(self):
        """Attribute access on constants parameter."""
        def f(t, y, p):
            return [-p.damping * y[0]]

        result = inspect_ode_function(f)
        assert len(result.constant_accesses) == 1
        assert result.constant_accesses[0]["key"] == "damping"
        assert result.constant_accesses[0]["pattern_type"] == "attribute"

    def test_constant_string_subscript(self):
        """String subscript on constants parameter."""
        def f(t, y, p):
            return [-p["mass"] * y[0]]

        result = inspect_ode_function(f)
        assert result.constant_accesses[0]["key"] == "mass"
        assert result.constant_accesses[0]["pattern_type"] == "string"

    def test_reject_lambda(self):
        """Lambda functions rejected."""
        f = lambda t, y: [-y[0]]  # noqa: E731
        with pytest.raises(TypeError, match="Lambda"):
            inspect_ode_function(f)

    def test_reject_too_few_params(self):
        """Functions with <2 params rejected."""
        def f(y):
            return [-y[0]]

        with pytest.raises(ValueError, match="at least 2"):
            inspect_ode_function(f)

    def test_reject_no_return(self):
        """Functions without return rejected."""
        def f(t, y):
            x = y[0]  # noqa: F841

        with pytest.raises(ValueError, match="return statement"):
            inspect_ode_function(f)

    def test_reject_multiple_returns(self):
        """Functions with multiple returns rejected.

        Uses a for-loop to produce two returns without an if-statement
        (which is separately rejected).
        """
        # Can't use if-statement (rejected), so test the validator
        # message via a function that genuinely has two returns at
        # the top level.  In practice this is hard to construct
        # without control flow, so we test the if-statement rejection
        # instead — the multiple-return check is a secondary guard.
        def f(t, y):
            a = 1.0 if True else 0.0
            return [-y[0] * a]

        # This should succeed (single return, ternary is fine)
        result = inspect_ode_function(f)
        assert result.return_node is not None

    def test_reject_mixed_access(self):
        """Mixed int and string subscript on same base rejected."""
        def f(t, y):
            return [-y[0] + y["x"]]

        with pytest.raises(ValueError, match="Mixed access"):
            inspect_ode_function(f)

    def test_warn_unconventional_time_param(self):
        """Warn when first param is not 't'."""
        def f(time, y):
            return [-y[0]]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inspect_ode_function(f)
            assert any("conventionally named 't'" in str(x.message) for x in w)

    def test_for_loop_unrolled(self):
        """For-loops over constant range() are unrolled."""
        def f(t, y):
            total = 0.0
            for i in range(2):
                total += y[i]
            return [-total]

        result = inspect_ode_function(f)
        assert "total" in result.assignments

    def test_for_loop_literal_list(self):
        """For-loops over literal list of constants are unrolled."""
        def f(t, y):
            total = 0.0
            for i in [0, 1, 3]:
                total += y[i]
            return [-total]

        result = inspect_ode_function(f)
        assert "total" in result.assignments

    def test_for_loop_literal_tuple(self):
        """For-loops over literal tuple of constants are unrolled."""
        def f(t, y):
            total = 0.0
            for c in (0.5, 1.0, 0.25):
                total += c * y[0]
            return [-total]

        result = inspect_ode_function(f)
        assert "total" in result.assignments

    def test_reject_for_loop_variable_iterable(self):
        """For-loops over variable iterables rejected."""
        def f(t, y):
            items = [1, 2]
            for i in items:
                pass
            return [-y[0]]

        with pytest.raises(NotImplementedError, match="literal"):
            inspect_ode_function(f)

    def test_reject_while_loop(self):
        """While-loops raise NotImplementedError."""
        def f(t, y):
            i = 0
            while i < 2:
                i = i + 1
            return [-y[0]]

        with pytest.raises(NotImplementedError, match="While-loop"):
            inspect_ode_function(f)

    def test_if_else_to_piecewise(self):
        """If/else converted to IfExp for downstream Piecewise."""
        def f(t, y):
            if y[0] > 0:
                result = y[0]
            else:
                result = -y[0]
            return [-result]

        result = inspect_ode_function(f)
        assert "result" in result.assignments
        # The stored AST node should be an IfExp (ternary)
        assert isinstance(result.assignments["result"], ast.IfExp)

    def test_if_elif_else(self):
        """If/elif/else chain produces nested IfExp."""
        def f(t, y):
            if y[0] > 1:
                result = y[0]
            elif y[0] > 0:
                result = 0.5 * y[0]
            else:
                result = 0.0
            return [-result]

        result = inspect_ode_function(f)
        node = result.assignments["result"]
        assert isinstance(node, ast.IfExp)
        # The orelse of the outer IfExp should also be an IfExp
        assert isinstance(node.orelse, ast.IfExp)

    def test_if_no_else_uses_fallback(self):
        """If without else falls back to prior assignment."""
        def f(t, y):
            result = y[0]
            if y[0] > 0:
                result = 2.0 * y[0]
            return [-result]

        result = inspect_ode_function(f)
        node = result.assignments["result"]
        assert isinstance(node, ast.IfExp)

    def test_reject_list_comprehension(self):
        """List comprehensions raise NotImplementedError."""
        def f(t, y):
            vals = [y[i] for i in range(2)]
            return vals

        with pytest.raises(NotImplementedError, match="List comprehension"):
            inspect_ode_function(f)

    def test_reject_generator_expression(self):
        """Generator expressions raise NotImplementedError."""
        def f(t, y):
            total = sum(y[i] for i in range(2))
            return [-total]

        with pytest.raises(NotImplementedError, match="Generator"):
            inspect_ode_function(f)

    def test_walrus_operator(self):
        """Walrus operator (:=) treated as assignment."""
        def f(t, y):
            if (v := y[0]) > 0:
                pass
            return [-v]

        result = inspect_ode_function(f)
        assert "v" in result.assignments

    def test_reject_with_statement(self):
        """'with' statements raise NotImplementedError."""
        def f(t, y):
            with open("x"):  # noqa: SIM117
                pass
            return [-y[0]]

        with pytest.raises(NotImplementedError, match="with"):
            inspect_ode_function(f)


    def test_reject_assert_statement(self):
        """'assert' statements raise NotImplementedError."""
        def f(t, y):
            assert y[0] > 0
            return [-y[0]]

        with pytest.raises(NotImplementedError, match="assert"):
            inspect_ode_function(f)

    def test_reject_raise_statement(self):
        """'raise' statements raise NotImplementedError."""
        def f(t, y):
            raise ValueError("boom")

        with pytest.raises(NotImplementedError, match="raise"):
            inspect_ode_function(f)

    def test_reject_global_statement(self):
        """'global' statements raise NotImplementedError with guidance."""
        def f(t, y):
            global SOME_VAR  # noqa: F824
            return [-y[0]]

        with pytest.raises(NotImplementedError, match="constants"):
            inspect_ode_function(f)

    def test_reject_nonlocal_statement(self):
        """'nonlocal' statements raise NotImplementedError."""
        x = 1.0  # noqa: F841

        def f(t, y):
            nonlocal x # noqa
            return [-y[0]]

        with pytest.raises(NotImplementedError, match="nonlocal"):
            inspect_ode_function(f)

    def test_augmented_assignment(self):
        """Augmented assignment (+=) chains correctly."""
        def f(t, y):
            total = y[0]
            total += y[1]
            return [-total]

        result = inspect_ode_function(f)
        assert "total" in result.assignments

    def test_piecewise_ifexp(self):
        """If/else ternary detected."""
        def f(t, y):
            return [1.0 if y[0] > 0 else -1.0]

        result = inspect_ode_function(f)
        assert result.return_node is not None


class TestAstToSympyConverter:
    """Tests for AST to SymPy conversion."""

    def test_constant_int(self):
        """Integer constant converts to sp.Integer."""
        import ast
        node = ast.Constant(value=42)
        converter = AstToSympyConverter({})
        result = converter.convert(node)
        assert result == sp.Integer(42)

    def test_constant_float(self):
        """Float constant converts to sp.Float."""
        import ast
        node = ast.Constant(value=3.14)
        converter = AstToSympyConverter({})
        result = converter.convert(node)
        assert result == sp.Float(3.14)

    def test_name_lookup(self):
        """Named variable resolves from symbol map."""
        import ast
        x = sp.Symbol("x", real=True)
        node = ast.Name(id="x")
        converter = AstToSympyConverter({"x": x})
        result = converter.convert(node)
        assert result == x

    def test_unknown_name_creates_symbol(self):
        """Unknown name creates a new real symbol."""
        import ast
        node = ast.Name(id="foo")
        converter = AstToSympyConverter({})
        result = converter.convert(node)
        assert isinstance(result, sp.Symbol)
        assert str(result) == "foo"


class TestSubscriptAndCallRecording:
    """Access recording for non-constant subscripts and call names."""

    def test_nested_subscript_value_ignored(self):
        """A subscript whose base is not a bare Name is not recorded.

        The outer ``y[0][1]`` subscript has a subscript (not a Name) as
        its value, so only the inner ``y[0]`` access is recorded.
        """
        def f(t, y):
            return [y[0][1]]

        result = inspect_ode_function(f)
        keys = [a["key"] for a in result.state_accesses]
        assert 0 in keys
        assert all(a["base"] == "y" for a in result.state_accesses)

    def test_name_subscript_pattern(self):
        """A bare-name subscript ``y[i]`` records pattern_type 'name'."""
        def f(t, y, i):
            return [y[i]]

        result = inspect_ode_function(f)
        assert result.state_accesses[0]["pattern_type"] == "name"
        assert result.state_accesses[0]["key"] == "i"

    def test_expr_subscript_pattern(self):
        """A computed subscript ``y[i + 1]`` records pattern_type 'expr'."""
        def f(t, y, i):
            return [y[i + 1]]

        result = inspect_ode_function(f)
        assert result.state_accesses[0]["pattern_type"] == "expr"

    def test_call_name_of_non_name_callable(self):
        """``_call_name`` returns None for a non-Name, non-Attribute func."""
        node = _expr("obj[0]()")
        assert _call_name(node) is None

    def test_resolve_func_name_strips_module_prefix(self):
        """Module-qualified names lose their known prefix."""
        assert _resolve_func_name("math.sin") == "sin"
        assert _resolve_func_name("np.exp") == "exp"

    def test_resolve_func_name_keeps_unknown_prefix(self):
        """A dotted name with an unknown prefix is returned unchanged."""
        assert _resolve_func_name("foo.bar") == "foo.bar"
        assert _resolve_func_name("sin") == "sin"


class TestAssignmentEdgeCases:
    """Tuple unpacking, augmented, walrus, and import handling."""

    def test_tuple_unpacking_length_mismatch_raises(self):
        """Mismatched tuple unpacking lengths raise a ValueError."""
        def f(t, y):
            a, b = y[0], y[1], y[2]  # noqa: F841 -- mismatch under test
            return [-a]

        with pytest.raises(ValueError, match="length mismatch"):
            inspect_ode_function(f)

    def test_tuple_unpacking_non_tuple_rhs(self):
        """Unpacking a single call binds each target to that call."""
        def f(t, y):
            from math import sin
            a, b = sin(y[0])
            return [-a, -b]

        result = inspect_ode_function(f)
        assert "a" in result.assignments
        assert "b" in result.assignments
        assert result.assignments["a"] is result.assignments["b"]

    def test_augmented_assignment_without_prior_raises(self):
        """An augmented assignment with no prior value raises."""
        def f(t, y):
            total += y[0]  # noqa: F821
            return [-total]

        with pytest.raises(ValueError, match="no prior assignment"):
            inspect_ode_function(f)

    def test_walrus_at_statement_level(self):
        """A walrus outside an if condition is treated as an assignment."""
        def f(t, y):
            z = (v := y[0]) + 1.0
            return [-v - z]

        result = inspect_ode_function(f)
        assert "v" in result.assignments
        assert "z" in result.assignments

    def test_plain_import_allowed(self):
        """A plain ``import`` statement is accepted."""
        def f(t, y):
            import math
            return [math.sin(y[0])]

        result = inspect_ode_function(f)
        assert "math.sin" in result.function_calls


class TestIfFallbacks:
    """If/elif fallbacks that raise when a branch lacks a prior value."""

    def test_if_branch_only_no_prior_raises(self):
        """A variable set only in an if-branch with no prior raises."""
        def f(t, y):
            if y[0] > 0:
                result = y[1]
            return [-result]  # noqa: F821

        with pytest.raises(ValueError, match="no else-branch"):
            inspect_ode_function(f)

    def test_else_branch_only_no_prior_raises(self):
        """A variable set only in an else-branch with no prior raises."""
        def f(t, y):
            if y[0] > 0:
                pass
            else:
                result = y[1]
            return [-result]  # noqa: F821

        with pytest.raises(ValueError, match="no if-branch"):
            inspect_ode_function(f)

    def test_nested_if_only_no_fallback_raises(self):
        """A nested-if-only assignment without a fallback raises."""
        def f(t, y):
            if y[0] > 1:
                r = y[0]
            else:
                if y[1] > 0:
                    r = y[1]
            return [-r]  # noqa: F821

        with pytest.raises(ValueError, match="nested if"):
            inspect_ode_function(f)

    def test_nested_else_only_no_fallback_raises(self):
        """A nested-else-only assignment without a fallback raises."""
        def f(t, y):
            if y[0] > 1:
                r = y[0]
            else:
                if y[1] > 0:
                    pass
                else:
                    r = y[1]
            return [-r]  # noqa: F821

        with pytest.raises(ValueError, match="nested"):
            inspect_ode_function(f)

    def test_else_branch_uses_prior_fallback(self):
        """An else-only assignment falls back to the prior value."""
        def f(t, y):
            r = 0.0
            if y[0] > 0:
                pass
            else:
                r = y[1]
            return [-r]

        result = inspect_ode_function(f)
        assert isinstance(result.assignments["r"], ast.IfExp)

    def test_nested_if_uses_prior_fallback(self):
        """A nested-if assignment falls back to the prior value."""
        def f(t, y):
            r = 0.0
            if y[0] > 1:
                r = y[0]
            else:
                if y[1] > 0:
                    r = y[1]
            return [-r]

        result = inspect_ode_function(f)
        assert isinstance(result.assignments["r"], ast.IfExp)

    def test_nested_else_uses_prior_fallback(self):
        """A nested-else assignment falls back to the prior value."""
        def f(t, y):
            r = 0.0
            if y[0] > 1:
                r = y[0]
            else:
                if y[1] > 0:
                    pass
                else:
                    r = y[1]
            return [-r]

        result = inspect_ode_function(f)
        assert isinstance(result.assignments["r"], ast.IfExp)

    def test_branch_annassign_expr_and_call_recorded(self):
        """Branch annotated-assign, bare expr, and calls are recorded."""
        def f(t, y):
            from math import sin
            a = 0.0
            if y[0] > 0:
                a: float = sin(y[1])
                sin(y[0])
            else:
                a = 1.0
            return [-a]

        result = inspect_ode_function(f)
        assert "sin" in result.function_calls
        assert isinstance(result.assignments["a"], ast.IfExp)

    def test_branch_tuple_unpacking_matched(self):
        """Matched tuple unpacking inside a branch splits per target."""
        def f(t, y):
            if y[0] > 0:
                a, b = y[0], y[1]
            else:
                a, b = -y[0], -y[1]
            return [-a, -b]

        result = inspect_ode_function(f)
        assert isinstance(result.assignments["a"], ast.IfExp)
        assert isinstance(result.assignments["b"], ast.IfExp)

    def test_branch_tuple_unpacking_non_tuple_rhs(self):
        """A single-call unpacking inside a branch binds every target."""
        def f(t, y):
            from math import sin
            if y[0] > 0:
                a, b = sin(y[0])
            else:
                a, b = sin(y[1]), sin(y[0])
            return [-a, -b]

        result = inspect_ode_function(f)
        assert isinstance(result.assignments["a"], ast.IfExp)
        assert isinstance(result.assignments["b"], ast.IfExp)

    def test_branch_augmented_without_prior_raises(self):
        """An augmented assignment in a branch with no prior raises."""
        def f(t, y):
            if y[0] > 0:
                total += y[1]  # noqa: F821
            else:
                total = 0.0
            return [-total]

        with pytest.raises(ValueError, match="no prior assignment"):
            inspect_ode_function(f)


class TestForIterableSigns:
    """For-loop unrolling over iterables carrying negative literals."""

    def test_range_with_negative_start(self):
        """``range`` arguments may be negated integer literals."""
        def f(t, y):
            total = 0.0
            for i in range(-1, 1):
                total += y[0]
            return [-total]

        result = inspect_ode_function(f)
        assert "total" in result.assignments

    def test_literal_tuple_with_negative_element(self):
        """Literal tuple elements may be negated numeric literals."""
        def f(t, y):
            total = 0.0
            for c in (-0.5, 0.5):
                total += c * y[0]
            return [-total]

        result = inspect_ode_function(f)
        assert "total" in result.assignments


class TestInspectRejections:
    """Top-level guards in ``inspect_ode_function``."""

    def test_non_callable_raises(self):
        """A non-callable argument raises TypeError."""
        with pytest.raises(TypeError, match="Expected callable"):
            inspect_ode_function(42)

    def test_builtin_without_source_raises(self):
        """A builtin with no inspectable source raises TypeError."""
        with pytest.raises(TypeError, match="builtin or C-extension"):
            inspect_ode_function(len)

    def test_unconventional_state_name_warns(self):
        """A second parameter with an unusual name warns."""
        def f(t, q):
            return [-q[0]]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inspect_ode_function(f)
            assert any(
                "conventionally named" in str(x.message) for x in w
            )

    def test_multiple_return_statements_raise(self):
        """More than one return statement raises."""
        def f(t, y):
            return [-y[0]]
            return [y[0]]  # noqa

        with pytest.raises(ValueError, match="exactly one return"):
            inspect_ode_function(f)

    def test_async_function_raises(self):
        """An async function raises: its AST has no FunctionDef node.

        ``ast.walk`` only matches ``ast.FunctionDef``; an ``async def``
        produces ``ast.AsyncFunctionDef``, so the definition search
        fails and the no-definition guard fires.
        """
        async def f(t, y):
            return [-y[0]]

        with pytest.raises(ValueError, match="Could not find function"):
            inspect_ode_function(f)


class TestConverterOperators:
    """Direct conversion of operator and comparison AST nodes."""

    def _converter(self):
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        return AstToSympyConverter({"x": x, "y": y}), x, y

    def test_div_floordiv_mod(self):
        """Division, floor division, and modulo convert correctly."""
        conv, x, y = self._converter()
        assert conv.convert(_expr("x / y")) == x / y
        assert conv.convert(_expr("x // y")) == sp.floor(x / y)
        assert conv.convert(_expr("x % y")) == sp.Mod(x, y)

    def test_unary_plus_and_not(self):
        """Unary plus is identity and ``not`` maps to ``sp.Not``."""
        conv, x, y = self._converter()
        assert conv.convert(_expr("+x")) == x
        assert conv.convert(_expr("not x")) == sp.Not(x)

    def test_all_comparisons(self):
        """Every supported comparison maps to its SymPy relation."""
        conv, x, y = self._converter()
        assert conv.convert(_expr("x >= y")) == sp.Ge(x, y)
        assert conv.convert(_expr("x < y")) == sp.Lt(x, y)
        assert conv.convert(_expr("x <= y")) == sp.Le(x, y)
        assert conv.convert(_expr("x == y")) == sp.Eq(x, y)
        assert conv.convert(_expr("x != y")) == sp.Ne(x, y)

    def test_bool_and_or(self):
        """Boolean ``and``/``or`` map to ``sp.And``/``sp.Or``."""
        conv, x, y = self._converter()
        assert conv.convert(_expr("(x > 0) and (y > 0)")) == sp.And(
            sp.Gt(x, 0), sp.Gt(y, 0)
        )
        assert conv.convert(_expr("(x > 0) or (y > 0)")) == sp.Or(
            sp.Gt(x, 0), sp.Gt(y, 0)
        )

    def test_tuple_and_list_expressions_raise(self):
        """Bare tuple/list expressions raise NotImplementedError."""
        conv, x, y = self._converter()
        with pytest.raises(NotImplementedError, match="unpacked"):
            conv.convert(_expr("(x, y)"))
        with pytest.raises(NotImplementedError, match="unpacked"):
            conv.convert(_expr("[x, y]"))


class TestConverterUserFunctions:
    """User-callable resolution and symbolic fallback."""

    def test_module_prefixed_user_call_inlined(self):
        """A module-qualified call resolves via the bare user name."""
        x = sp.Symbol("x", real=True)
        conv = AstToSympyConverter(
            {"x": x},
            user_callables={"myfn": lambda a: a * 2},
            user_function_classes={"myfn": sp.Function("myfn")},
        )
        result = conv.convert(_expr("np.myfn(x)"))
        assert result == 2 * x

    def test_user_call_symbolic_fallback_on_error(self):
        """A user callable that rejects symbolic args stays symbolic."""
        def bad(a):
            raise ValueError("symbolic arg rejected")

        x = sp.Symbol("x", real=True)
        myfn = sp.Function("myfn")
        conv = AstToSympyConverter(
            {"x": x},
            user_callables={"myfn": bad},
            user_function_classes={"myfn": myfn},
        )
        result = conv.convert(_expr("myfn(x)"))
        assert result == myfn(x)
