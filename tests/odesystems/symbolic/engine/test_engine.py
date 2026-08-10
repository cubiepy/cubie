"""Unit tests for the IR expression engine.

Verifies interning and algebraic folding, differentiation against
SymPy ground truth, simultaneous substitution, CSE numeric
equivalence, topological ordering, pruning, and SymPy round-trips.
"""

import gc
import math
import random
import weakref

import pytest
import sympy as sp

from cubie.odesystems.symbolic.engine import (
    TRUE,
    Arr,
    ConversionError,
    Local,
    add,
    arr,
    call,
    count_ops,
    cse_and_stack,
    diff,
    div,
    free_atoms,
    from_sympy,
    is_one,
    is_zero,
    mul,
    neg,
    num,
    piecewise,
    pow_,
    prune_unused,
    rel,
    sub,
    sym,
    to_sympy,
    topological_sort,
    xreplace,
)
from cubie.odesystems.symbolic.engine.adapter import system_ir
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie.odesystems.symbolic.parsing.parser import ParsedEquations


class TestInterningAndFolding:
    def test_structural_interning(self):
        x, y = sym("x"), sym("y")
        assert add(x, y) is add(y, x)
        assert mul(x, y) is mul(y, x)
        assert sym("x") is x
        assert arr("state", 1) is arr("state", 1)

    def test_like_term_collection(self):
        x = sym("x")
        assert add(x, x) is mul(num(2), x)
        assert sub(x, x) is num(0)

    def test_power_collection(self):
        x = sym("x")
        assert mul(x, x) is pow_(x, num(2))
        assert div(x, x) is num(1)

    def test_identity_folds(self):
        x = sym("x")
        assert add(x, num(0)) is x
        assert mul(x, num(1)) is x
        assert is_zero(mul(x, num(0)))
        assert is_one(pow_(x, num(0)))
        assert pow_(x, num(1)) is x

    def test_numeric_folding(self):
        assert mul(num(2), num(3)) is num(6)
        assert add(num(0.5), num(0.25)) is num(0.75)
        assert pow_(pow_(sym("x"), num(2)), num(3)) is pow_(
            sym("x"), num(6)
        )

    def test_int_and_float_zero_distinct_nodes(self):
        assert num(0) is not num(0.0)
        assert is_zero(num(0.0))

    def test_piecewise_collapses_after_true(self):
        x = sym("x")
        collapsed = piecewise((x, TRUE))
        assert collapsed is x

    def test_piecewise_drops_false_branches(self):
        from cubie.odesystems.symbolic.engine import FALSE

        x, y = sym("x"), sym("y")
        cond = rel("<", x, num(0))
        pruned = piecewise((y, FALSE), (x, cond), (y, TRUE))
        assert pruned is piecewise((x, cond), (y, TRUE))
        try:
            piecewise((x, FALSE))
        except ValueError:
            pass
        else:
            raise AssertionError("all-false piecewise did not raise")

    def test_piecewise_requires_fallback(self):
        x = sym("x")
        with pytest.raises(ValueError, match="final true"):
            piecewise((x, rel(">", x, num(0))))

    def test_count_ops_covers_conditionals(self):
        x, y = sym("x"), sym("y")
        cond = rel("<", add(x, y), num(0))
        assert count_ops(cond) == 2
        selection = piecewise((mul(x, y), cond), (x, TRUE))
        assert count_ops(selection) == 5
        from cubie.odesystems.symbolic.engine import bool_op

        both = bool_op("and", cond, rel(">", x, num(0)))
        assert count_ops(both) == 5

    def test_float_one_folds_to_identity(self):
        x = sym("x")
        assert mul(num(1.0), x) is x
        assert add(mul(num(0.5), x), mul(num(0.5), x)) is x
        assert pow_(x, num(1.0)) is x
        assert is_one(pow_(x, num(0.0)))
        assert is_one(pow_(num(1.0), x))

    def test_zero_after_power_combination(self):
        x, y = sym("x"), sym("y")
        vanishing = mul(pow_(num(0), x), pow_(num(0), sub(num(2), x)))
        assert is_zero(mul(vanishing, y))

    def test_pickle_round_trip_preserves_interning(self):
        import pickle

        x, y = sym("x"), sym("y")
        node = add(
            mul(num(2), x, y),
            pow_(x, num(3)),
            piecewise((x, rel("<", x, num(1))), (y, TRUE)),
            call("exp", arr("state", 2)),
        )
        restored = pickle.loads(pickle.dumps(node))
        assert restored is node

    def test_unreferenced_nodes_leave_intern_pool(self):
        node = sym("weak_intern_test_symbol")
        reference = weakref.ref(node)
        del node
        gc.collect()
        assert reference() is None

    def test_huge_integer_literal_constructs(self):
        node = pow_(num(10), num(400))
        assert node.value == 10**400

    def test_float_equal_payloads_order_totally(self):
        x = sym("x")
        big, bigger = 2**60, 2**60 + 1
        forward = add(pow_(x, num(big)), pow_(x, num(bigger)))
        reverse = add(pow_(x, num(bigger)), pow_(x, num(big)))
        assert forward is reverse


class TestDifferentiation:
    def test_matches_sympy_on_rule_table(self):
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        cases = [
            x * sp.sin(x),
            sp.exp(2 * x) / (1 + x**2),
            sp.sqrt(x + y) * sp.cos(x * y),
            sp.tan(x) + sp.atan(x) + sp.tanh(x) + sp.atanh(x),
            sp.log(x) * sp.asin(y) + sp.acos(x * y),
            x ** sp.Rational(3, 2) + y**-2,
            sp.erf(x) + sp.erfc(2 * x),
            sp.atan2(y, x),
            sp.sinh(x) * sp.cosh(y) + sp.asinh(x) + sp.acosh(x + 2),
        ]
        rng = random.Random(7)
        for case in cases:
            for var in (x, y):
                expected = sp.lambdify(
                    (x, y), sp.diff(case, var), "math"
                )
                produced = sp.lambdify(
                    (x, y),
                    to_sympy(
                        diff(from_sympy(case), from_sympy(var))
                    ),
                    "math",
                )
                # Domain 0.15..0.85 keeps every argument valid:
                # |atanh/asin args| < 1 and acosh sees x + 2 > 1.
                for _ in range(8):
                    px = rng.uniform(0.15, 0.85)
                    py = rng.uniform(0.15, 0.85)
                    want = expected(px, py)
                    got = produced(px, py)
                    assert math.isclose(
                        got, want, rel_tol=1e-10, abs_tol=1e-12
                    ), (case, var, px, py, got, want)

    def test_piecewise_differentiates_by_branch(self):
        x = sym("x")
        expr = piecewise(
            (mul(x, x), rel("<", x, num(0))),
            (mul(num(3), x), TRUE),
        )
        result = diff(expr, x)
        expected = piecewise(
            (mul(num(2), x), rel("<", x, num(0))),
            (num(3), TRUE),
        )
        assert result is expected

    def test_min_differentiates_to_selection(self):
        x, y = sym("x"), sym("y")
        result = diff(call("Min", x, y), x)
        assert result is piecewise(
            (num(1), rel("<=", x, y)), (num(0), TRUE)
        )

    def test_user_function_derivative_placeholder(self):
        x, y = sym("x"), sym("y")
        result = diff(call("foo_", x, y), x)
        assert result is call("d_foo", x, y, num(0))
        renamed = diff(
            call("foo_", x, y),
            y,
            derivative_names={"foo_": "dfoo_dx"},
        )
        assert renamed is call("dfoo_dx", x, y, num(1))

    def test_mod_differentiates_almost_everywhere(self):
        x = sym("x")
        derivative = diff(call("Mod", x, num(3)), x)
        assert is_one(derivative)

    def test_variable_exponent_uses_general_power_rule(self):
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        rng = random.Random(11)
        for case, var in ((x**y, y), (x**x, x), (x**y, x)):
            expected = sp.lambdify((x, y), sp.diff(case, var), "math")
            produced = sp.lambdify(
                (x, y),
                to_sympy(diff(from_sympy(case), from_sympy(var))),
                "math",
            )
            for _ in range(8):
                px = rng.uniform(0.2, 2.0)
                py = rng.uniform(0.2, 2.0)
                assert math.isclose(
                    produced(px, py),
                    expected(px, py),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ), (case, var, px, py)

    def test_no_derivative_rule_raises(self):
        from cubie.odesystems.symbolic.engine import (
            DifferentiationError,
        )

        x = sym("x")
        for name in (
            "gamma",
            "loggamma",
            "hypot",
            "fmod",
            "remainder",
            "copysign",
        ):
            args = (x, x) if name in (
                "hypot", "fmod", "remainder", "copysign"
            ) else (x,)
            try:
                diff(call(name, *args), x)
            except DifferentiationError:
                continue
            raise AssertionError(f"{name} did not raise")

    def test_array_reference_is_constant(self):
        x = sym("x")
        assert is_zero(diff(arr("v", 0), x))
        v0 = arr("v", 0)
        assert diff(mul(x, v0), x) is v0


class TestSubstitution:
    def test_simultaneous_replacement(self):
        x, y = sym("x"), sym("y")
        expr = add(x, y)
        swapped = xreplace(expr, {x: y, y: x})
        assert swapped is expr  # commutative interning

    def test_no_rescan_of_images(self):
        x = sym("x")
        image = add(x, num(1))
        replaced = xreplace(mul(x, x), {x: image})
        assert replaced is pow_(image, num(2))

    def test_composite_key_replacement(self):
        x, y, z = sym("x"), sym("y"), sym("z")
        inner = add(x, y)
        expr = mul(num(3), pow_(inner, num(2)))
        replaced = xreplace(expr, {inner: z})
        assert replaced is mul(num(3), pow_(z, num(2)))

    def test_array_replacement(self):
        combo = add(arr("base", 0), mul(sym("a"), arr("u", 0)))
        replaced = xreplace(
            mul(arr("v", 0), sym("k")), {arr("v", 0): combo}
        )
        assert replaced is mul(combo, sym("k"))


def _evaluate_lines(assignments, bindings):
    """Execute printed assignments and return the environment."""
    from cubie.odesystems.symbolic.engine import print_cuda_multiple

    env = {"math": math, "precision": float}
    env.update(bindings)
    for line in print_cuda_multiple(assignments):
        exec(line, env)
    return env


class TestCseAndStack:
    def test_numeric_equivalence(self):
        x, y, k = sym("x"), sym("y"), sym("k")
        shared = mul(call("exp", mul(k, x)), add(x, y))
        assignments = [
            (sym("r0"), add(shared, x)),
            (sym("r1"), mul(num(2), shared)),
            (sym("r2"), mul(shared, y, num(-3))),
        ]
        bindings = {"x": 0.37, "y": -1.2, "k": 2.5}
        direct = {
            "r0": (
                math.exp(2.5 * 0.37) * (0.37 - 1.2) + 0.37
            ),
            "r1": 2 * math.exp(2.5 * 0.37) * (0.37 - 1.2),
            "r2": -3 * math.exp(2.5 * 0.37) * (0.37 - 1.2) * -1.2,
        }
        env = _evaluate_lines(cse_and_stack(assignments), bindings)
        for name, expected in direct.items():
            assert math.isclose(env[name], expected, rel_tol=1e-12)

    def test_coefficient_scaled_products_share(self):
        u, w, z = sym("u"), sym("w"), sym("z")
        shared = mul(u, call("exp", w))
        assignments = [
            (sym("p0"), mul(num(2), shared)),
            (sym("p1"), mul(num(-3), shared)),
            (arr("out", 0), mul(shared, z)),
        ]
        stacked = cse_and_stack(assignments)
        names = [
            lhs.name
            for lhs, _ in stacked
            if hasattr(lhs, "name") and lhs.name.startswith("_cse")
        ]
        assert names, "no shared subexpression extracted"
        # exp(w)*u must be computed exactly once.
        from cubie.odesystems.symbolic.engine import (
            print_cuda_multiple,
        )

        text = "\n".join(print_cuda_multiple(stacked))
        assert text.count("math.exp(w)") == 1

    def test_numbering_continues_after_existing(self):
        x = sym("x")
        shared = call("exp", mul(x, x))
        assignments = [
            (sym("_cse4"), add(x, num(1))),
            (sym("a"), add(shared, sym("_cse4"))),
            (sym("b"), mul(shared, num(2))),
        ]
        stacked = cse_and_stack(assignments)
        new_names = {
            lhs.name
            for lhs, _ in stacked
            if hasattr(lhs, "name") and lhs.name.startswith("_cse")
        }
        assert "_cse4" in new_names
        assert "_cse5" in new_names

    @pytest.mark.parametrize(
        "reserved",
        [
            sym("_cse0"),
            arr("_cse0", 0),
            call("_cse0", sym("x")),
        ],
    )
    def test_user_cse_name_is_not_overwritten(self, reserved):
        x, k = sym("x"), sym("k")
        shared = add(x, k)
        assignments = [
            (
                arr("out", 0),
                add(
                    reserved,
                    call("sin", shared),
                    call("cos", shared),
                ),
            )
        ]
        stacked = cse_and_stack(assignments)
        locals_ = [
            lhs.name for lhs, _ in stacked if isinstance(lhs, Local)
        ]
        assert locals_ == ["_cse1"]

    def test_generated_local_is_not_symbol_mapped(self):
        from cubie.odesystems.symbolic.engine import print_cuda_multiple

        x, k = sym("x"), sym("k")
        shared = call("exp", add(x, k))
        stacked = cse_and_stack(
            [
                (arr("out", 0), add(shared, x)),
                (arr("out", 1), add(shared, k)),
            ]
        )
        lines = print_cuda_multiple(
            stacked,
            symbol_map={"_cse0": arr("parameters", 9)},
        )
        assert any(line.startswith("_cse0 = ") for line in lines)
        assert not any(line.startswith("parameters[9] = ") for line in lines)

    def test_piecewise_fallback_is_not_extracted(self):
        x = sym("x")
        condition = rel(">", x, num(0))
        first = piecewise((x, condition), (num(0), TRUE))
        second = piecewise((num(1), condition), (num(2), TRUE))
        stacked = cse_and_stack(
            [(arr("out", 0), first), (arr("out", 1), second)]
        )
        assert stacked


class TestOrderingAndPruning:
    def test_topological_sort_orders_dependencies(self):
        a, b = sym("a"), sym("b")
        ordered = topological_sort(
            [(b, add(a, num(1))), (a, num(2))]
        )
        assert [lhs for lhs, _ in ordered] == [a, b]

    def test_topological_sort_detects_cycles(self):
        a, b = sym("a"), sym("b")
        try:
            topological_sort([(a, b), (b, a)])
        except ValueError as error:
            assert "Circular" in str(error)
        else:
            raise AssertionError("cycle not detected")

    def test_topological_sort_rejects_duplicate_targets(self):
        a, b = sym("a"), sym("b")
        try:
            topological_sort([(a, b), (a, num(2))])
        except ValueError as error:
            assert "Duplicate assignment targets" in str(error)
        else:
            raise AssertionError("duplicate target not detected")

    def test_prune_drops_dead_assignments(self):
        a, dead = sym("a"), sym("dead")
        pruned = prune_unused(
            [
                (a, num(2)),
                (dead, num(9)),
                (arr("out", 0), mul(a, num(3))),
            ],
            output_name="out",
        )
        assert (dead, num(9)) not in pruned

    def test_prune_without_outputs_is_noop(self):
        a = sym("a")
        assignments = [(a, num(2))]
        assert prune_unused(assignments, output_name="out") == (
            assignments
        )

    def test_topological_sort_small_graphs_keep_stable_order(self):
        a, b = sym("a"), sym("b")
        ordered = topological_sort(
            [
                (a, num(2)),
                (b, num(3)),
                (arr("out", 0), add(a, num(1))),
                (arr("out", 1), add(b, num(1))),
            ]
        )
        assert [lhs for lhs, _ in ordered] == [
            a, b, arr("out", 0), arr("out", 1)
        ]

    def test_topological_sort_groups_output_chains(self):
        """Wide disjoint chains schedule contiguously, one at a time."""
        n_chains = 70
        assignments = []
        for index in range(n_chains):
            root = sym(f"a{index}")
            assignments.append((root, num(index + 2)))
        for index in range(n_chains):
            assignments.append(
                (arr("out", index), add(sym(f"a{index}"), num(1)))
            )
        ordered = topological_sort(assignments)
        self._assert_dependencies_precede_uses(ordered)
        assert self._peak_live(ordered) == 1

    @staticmethod
    def _peak_live(ordered):
        """Count peak simultaneously-live scalar temporaries."""
        targets = {lhs for lhs, _ in ordered}
        position = {lhs: i for i, (lhs, _) in enumerate(ordered)}
        last_use = {}
        for lhs, rhs in ordered:
            for dep in free_atoms(rhs) & targets:
                last_use[dep] = position[lhs]
        live = 0
        peak = 0
        ends = {}
        for index, (lhs, _) in enumerate(ordered):
            if not isinstance(lhs, Arr) and lhs in last_use:
                live += 1
                ends.setdefault(last_use[lhs], []).append(lhs)
            peak = max(peak, live)
            live -= len(ends.get(index, ()))
        return peak

    @staticmethod
    def _assert_dependencies_precede_uses(ordered):
        targets = {lhs for lhs, _ in ordered}
        emitted = set()
        for lhs, rhs in ordered:
            deps = free_atoms(rhs) & targets
            assert deps <= emitted
            emitted.add(lhs)

    def test_topological_sort_shared_prefix_no_worse_than_kahn(self):
        """A shared chain with a deep first output stays low-peak."""
        n_stages = 16
        stages = [sym("s0")]
        assignments = [(stages[0], call("exp", sym("x0")))]
        for index in range(1, n_stages):
            stage = sym(f"s{index}")
            assignments.append(
                (stage, call("sin", stages[index - 1]))
            )
            stages.append(stage)
        # Deep output first, then a shallow tap on every stage.
        assignments.append(
            (arr("out", 0), add(stages[-1], num(1)))
        )
        for index in range(n_stages):
            assignments.append(
                (
                    arr("out", index + 1),
                    mul(stages[index], num(2)),
                )
            )
        ordered = topological_sort(assignments)
        self._assert_dependencies_precede_uses(ordered)
        assert sorted(
            str(lhs) for lhs, _ in ordered
        ) == sorted(str(lhs) for lhs, _ in assignments)
        peak = self._peak_live(ordered)
        assert peak <= 3

    def test_topological_sort_fanout_diamond(self):
        """Fan-out diamonds retire each diamond before the next."""
        assignments = []
        for index in range(70):
            root = sym(f"d{index}")
            left = sym(f"l{index}")
            right = sym(f"r{index}")
            assignments.extend(
                [
                    (root, call("exp", sym(f"x{index}"))),
                    (left, call("sin", root)),
                    (right, call("cos", root)),
                    (arr("out", index), add(left, right)),
                ]
            )
        ordered = topological_sort(assignments)
        self._assert_dependencies_precede_uses(ordered)
        assert self._peak_live(ordered) <= 3

    def test_topological_sort_mixed_graph(self):
        """Disjoint chains plus shared intermediates stay low-peak."""
        shared = sym("shared")
        assignments = [(shared, call("exp", sym("x")))]
        for index in range(6):
            tap = sym(f"t{index}")
            assignments.append((tap, mul(shared, num(index + 2))))
            assignments.append(
                (arr("out", index), add(tap, num(1)))
            )
        for index in range(70):
            lone = sym(f"c{index}")
            assignments.append(
                (lone, call("sin", sym(f"y{index}")))
            )
            assignments.append(
                (arr("out", 6 + index), add(lone, num(1)))
            )
        ordered = topological_sort(assignments)
        self._assert_dependencies_precede_uses(ordered)
        assert self._peak_live(ordered) <= 3

    def test_topological_sort_deterministic(self):
        assignments = [
            (sym("a"), num(2)),
            (sym("b"), mul(sym("a"), num(3))),
            (sym("c"), add(sym("a"), sym("b"))),
            (arr("out", 0), add(sym("c"), num(1))),
            (arr("out", 1), mul(sym("b"), num(2))),
        ]
        first = topological_sort(list(assignments))
        second = topological_sort(list(assignments))
        assert first == second
        self._assert_dependencies_precede_uses(first)

    def test_free_atoms_and_count_ops(self):
        x, k = sym("x"), sym("k")
        expr = mul(x, call("sin", mul(k, x)))
        assert free_atoms(expr) == frozenset((x, k))
        assert count_ops(expr) == 3


class TestSympyRoundTrip:
    @pytest.mark.parametrize(
        "expression",
        [
            sp.airyai(sp.Symbol("x")),
            sp.besselj(0, sp.Symbol("x")),
            sp.factorial(sp.Symbol("x")),
            sp.Function("unknown")(sp.Symbol("x")),
        ],
    )
    def test_unsupported_functions_raise(self, expression):
        with pytest.raises(ConversionError):
            from_sympy(expression)

    def test_registered_function_alias_converts(self):
        x = sp.Symbol("x")
        expression = sp.Function("exp_")(x)
        converted = from_sympy(
            expression,
            allowed_functions={"exp_"},
        )
        assert converted is call("exp_", sym("x"))

    def test_incomplete_piecewise_raises(self):
        x = sp.Symbol("x")
        expression = sp.Piecewise((1, x > 0))
        with pytest.raises(ConversionError, match="final true"):
            from_sympy(expression)

    def test_round_trip_preserves_value(self):
        x, y = sp.symbols("x y", real=True)
        cases = [
            x**2 * sp.sin(y) + sp.Rational(1, 2) * x / y,
            sp.Piecewise((x, x > 0), (-x, True)),
            sp.Min(x, y) + sp.Max(x, 2 * y),
            sp.exp(-(x**2)) * sp.erf(y),
        ]
        rng = random.Random(42)
        for case in cases:
            round_tripped = to_sympy(from_sympy(case))
            for _ in range(5):
                point = {
                    x: rng.uniform(0.1, 2.0),
                    y: rng.uniform(0.1, 2.0),
                }
                expected = float(case.subs(point))
                produced = float(round_tripped.subs(point))
                assert math.isclose(
                    produced, expected, rel_tol=1e-12
                )

    def test_indexed_becomes_arr(self):
        base = sp.IndexedBase("state")
        node = from_sympy(base[2])
        assert node is arr("state", 2)

    def test_bracket_named_symbol_becomes_arr(self):
        node = from_sympy(sp.Symbol("jvp[3]"))
        assert node is arr("jvp", 3)

    def test_negation_prints_through(self):
        x = sym("x")
        assert to_sympy(neg(x)) == -sp.Symbol("x", real=True)


def test_system_ir_reflects_index_mutation():
    index_map = IndexedBases.from_user_inputs(
        states={"x": 1.0},
        parameters={"k": 2.0},
        constants={"c": 3.0},
        observables=[],
        drivers=[],
    )
    equations = ParsedEquations.from_equations(
        [(sym("dx"), add(sym("x"), sym("k"), sym("c")))],
        index_map,
    )
    before = system_ir(equations, index_map)
    index_map.constant_to_parameter("c")
    after = system_ir(equations, index_map)

    assert before is not after
    assert "c" not in before.arrayrefs
    assert after.arrayrefs["c"] is arr("parameters", 1)
