"""Unit tests for the greedy auxiliary cache planner.

Each test builds a real JVPEquations instance from small assignment
lists whose operation counts steer the planner into a specific
selection outcome. Expressions are built directly on the engine IR.
"""

import time

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.parsing import JVPEquations
from cubie.odesystems.symbolic.parsing.auxiliary_caching import (
    plan_auxiliary_cache,
)


def _chain(symbol, functions, extra=None):
    """Sum of ``functions`` applied to ``symbol`` plus an optional term."""
    terms = [ir.call(name, symbol) for name in functions]
    if extra is not None:
        terms.append(extra)
    return ir.add(*terms)


def _assert_partition_consistent(equations, selection):
    """Assert the structural invariants every cache plan must hold."""
    all_nodes = set(equations.non_jvp_order)
    cached = set(selection.cached_leaf_order)
    removed = set(selection.removal_nodes)
    runtime = set(selection.runtime_nodes)
    prepare = set(selection.prepare_nodes)

    assert cached <= removed
    assert cached <= prepare
    assert removed | runtime == all_nodes
    assert removed & runtime == set()
    assert len(selection.cached_leaf_order) <= equations.cache_slot_limit
    for group in selection.groups:
        assert group.saved >= equations.min_ops_threshold
    # Runtime assignments may read runtime or cached values only.
    for lhs in runtime:
        for dep in equations.dependencies.get(lhs, set()):
            assert dep in runtime or dep in cached
    # Prepare assignments are self-contained.
    for lhs in prepare:
        for dep in equations.dependencies.get(lhs, set()):
            assert dep in prepare
    # No cached value may depend on the direction vector.
    assert not (cached & equations.v_dependent_nodes)


def test_zero_slot_limit_selects_nothing():
    """max_cached_terms=0 disables caching and keeps all nodes runtime."""
    x0 = ir.sym("x0")
    aux = ir.sym("aux_a")
    exprs = [
        (aux, _chain(x0, ("sin", "cos", "exp", "tan"))),
        (ir.arr("jvp", 0), ir.mul(aux, ir.arr("v", 0))),
    ]
    equations = JVPEquations(
        exprs, max_cached_terms=0, min_ops_threshold=1
    )
    selection = plan_auxiliary_cache(equations)
    assert selection.groups == ()
    assert selection.cached_leaves == ()
    assert selection.runtime_nodes == tuple(equations.non_jvp_order)


def test_dead_auxiliary_is_never_a_seed():
    """An assignment feeding no jvp term stays runtime and uncached."""
    x0 = ir.sym("x0")
    dead = ir.sym("aux_dead")
    live = ir.sym("aux_live")
    exprs = [
        (dead, _chain(x0, ("sin", "cos", "exp", "tan"))),
        (live, _chain(x0, ("sinh", "cosh", "tanh", "log"))),
        (ir.arr("jvp", 0), ir.mul(live, ir.arr("v", 0))),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=1)
    selection = plan_auxiliary_cache(equations)
    assert dead not in selection.cached_leaves
    assert dead in selection.runtime_nodes
    assert live in selection.cached_leaves
    _assert_partition_consistent(equations, selection)


def test_cheap_leaves_below_threshold_select_nothing():
    """Leaves cheaper than the per-slot threshold stay runtime."""
    x0, x1 = ir.sym("x0"), ir.sym("x1")
    aux = ir.sym("aux_a")
    exprs = [
        (aux, ir.add(x0, x1)),
        (ir.arr("jvp", 0), ir.mul(aux, ir.arr("v", 0))),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=10)
    selection = plan_auxiliary_cache(equations)
    assert selection.groups == ()
    assert selection.cached_leaves == ()
    assert aux in selection.runtime_nodes


def test_cse_locals_are_cacheable():
    """A ``_cse`` local shared by live consumers is a valid cache leaf."""
    x0, x1 = ir.sym("x0"), ir.sym("x1")
    cse0 = ir.sym("_cse0")
    aux_a = ir.sym("aux_a")
    aux_b = ir.sym("aux_b")
    exprs = [
        (cse0, _chain(x0, ("sin", "cos", "exp", "tan"), extra=x1)),
        (aux_a, ir.add(cse0, x0)),
        (aux_b, ir.mul(cse0, x1)),
        (ir.arr("jvp", 0), ir.mul(aux_a, ir.arr("v", 0))),
        (ir.arr("jvp", 1), ir.mul(aux_b, ir.arr("v", 1))),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=8)
    selection = plan_auxiliary_cache(equations)
    # The shared transcendental-heavy _cse local is the only node
    # worth a slot; its cheap consumers stay runtime and read it
    # back from the cache buffer.
    assert cse0 in selection.cached_leaves
    assert aux_a in selection.runtime_nodes
    assert aux_b in selection.runtime_nodes
    assert selection.saved == equations.ops_cost[cse0]
    _assert_partition_consistent(equations, selection)


def test_node_with_live_dependents_is_cacheable():
    """Caching an intermediate keeps its consumers live in runtime."""
    x0 = ir.sym("x0")
    inner = ir.sym("aux_inner")
    outer = ir.sym("aux_outer")
    exprs = [
        (inner, _chain(x0, ("sin", "cos", "exp", "tan"))),
        (outer, ir.add(inner, x0)),
        (ir.arr("jvp", 0), ir.mul(outer, ir.arr("v", 0))),
        (ir.arr("jvp", 1), ir.mul(inner, ir.arr("v", 1))),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=8)
    selection = plan_auxiliary_cache(equations)
    assert inner in selection.cached_leaves
    assert outer in selection.runtime_nodes
    assert selection.saved == equations.ops_cost[inner]
    _assert_partition_consistent(equations, selection)


def test_caching_sole_consumer_cascades_into_dead_dependencies():
    """A dependency left without live consumers leaves runtime too."""
    x0 = ir.sym("x0")
    dep = ir.sym("aux_dep")
    top = ir.sym("aux_top")
    exprs = [
        (dep, _chain(x0, ("sin", "cos", "exp", "tan"))),
        (top, _chain(dep, ("sinh", "cosh", "tanh", "log"))),
        (ir.arr("jvp", 0), ir.mul(top, ir.arr("v", 0))),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=1)
    selection = plan_auxiliary_cache(equations)
    # Caching top removes dep as well: nothing else reads it.
    assert selection.cached_leaves == (top,)
    assert dep in selection.removal_nodes
    assert dep not in selection.runtime_nodes
    assert dep in selection.prepare_nodes
    assert selection.saved == (
        equations.ops_cost[top] + equations.ops_cost[dep]
    )
    _assert_partition_consistent(equations, selection)


def test_v_dependent_nodes_are_never_cached():
    """A node reading the direction vector is excluded from caching."""
    x0 = ir.sym("x0")
    mixes_v = ir.sym("aux_v")
    downstream = ir.sym("aux_down")
    exprs = [
        (
            mixes_v,
            _chain(x0, ("sin", "cos", "exp"), extra=ir.arr("v", 0)),
        ),
        (downstream, _chain(mixes_v, ("sinh", "cosh", "tanh"))),
        (ir.arr("jvp", 0), downstream),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=1)
    assert mixes_v in equations.v_dependent_nodes
    assert downstream in equations.v_dependent_nodes
    selection = plan_auxiliary_cache(equations)
    assert selection.cached_leaves == ()
    assert mixes_v in selection.runtime_nodes
    assert downstream in selection.runtime_nodes


def test_device_weighting_prefers_transcendental_candidates():
    """A transcendental-heavy node beats a larger add-only tree."""
    x0, x1 = ir.sym("x0"), ir.sym("x1")
    heavy = ir.sym("aux_heavy")
    wide = ir.sym("aux_wide")
    add_terms = [ir.mul(x0, ir.num(float(k + 2))) for k in range(6)]
    exprs = [
        (heavy, ir.add(ir.call("exp", x0), ir.call("sin", x1))),
        (wide, ir.add(*add_terms)),
        (ir.arr("jvp", 0), ir.mul(heavy, ir.arr("v", 0))),
        (ir.arr("jvp", 1), ir.mul(wide, ir.arr("v", 1))),
    ]
    equations = JVPEquations(
        exprs, max_cached_terms=1, min_ops_threshold=1
    )
    assert equations.ops_cost[heavy] > equations.ops_cost[wide]
    selection = plan_auxiliary_cache(equations)
    assert selection.cached_leaves == (heavy,)
    assert wide in selection.runtime_nodes


def test_greedy_records_marginal_savings_in_order():
    """Groups record greedy order and per-slot marginal savings."""
    x0, x1 = ir.sym("x0"), ir.sym("x1")
    aux_a = ir.sym("aux_a")
    aux_b = ir.sym("aux_b")
    aux_c = ir.sym("aux_c")
    exprs = [
        (
            aux_a,
            _chain(
                x0,
                ("sin", "cos", "exp", "tan", "sinh", "cosh"),
                extra=x1,
            ),
        ),
        (
            aux_b,
            _chain(x0, ("tanh", "log", "asin", "acos"), extra=x1),
        ),
        (aux_c, _chain(x0, ("atan", "asinh", "acosh"))),
        (ir.arr("jvp", 0), ir.mul(aux_a, ir.arr("v", 0))),
        (ir.arr("jvp", 1), ir.mul(aux_b, ir.arr("v", 1))),
        (ir.arr("jvp", 2), ir.mul(aux_c, ir.arr("v", 2))),
    ]
    equations = JVPEquations(exprs, min_ops_threshold=5)
    selection = plan_auxiliary_cache(equations)
    assert set(selection.cached_leaves) == {aux_a, aux_b, aux_c}
    assert [group.seed for group in selection.groups] == [
        aux_a,
        aux_b,
        aux_c,
    ]
    assert [group.saved for group in selection.groups] == [
        equations.ops_cost[aux_a],
        equations.ops_cost[aux_b],
        equations.ops_cost[aux_c],
    ]
    assert selection.saved == sum(
        group.saved for group in selection.groups
    )
    # Slot order is evaluation order regardless of greedy order.
    assert selection.cached_leaf_order == (aux_a, aux_b, aux_c)
    _assert_partition_consistent(equations, selection)


def test_slot_limit_caps_selection():
    """Only the most valuable candidates fit within the slot cap."""
    x0 = ir.sym("x0")
    leaves = []
    exprs = []
    functions = ("sin", "cos", "exp", "tan", "sinh")
    for i in range(4):
        aux = ir.sym(f"aux_{i}")
        leaves.append(aux)
        exprs.append((aux, _chain(x0, functions[: i + 2])))
        exprs.append((ir.arr("jvp", i), ir.mul(aux, ir.arr("v", i))))
    equations = JVPEquations(
        exprs, max_cached_terms=2, min_ops_threshold=1
    )
    selection = plan_auxiliary_cache(equations)
    assert len(selection.cached_leaves) == 2
    # The two most expensive chains win the slots.
    assert set(selection.cached_leaves) == {leaves[3], leaves[2]}
    _assert_partition_consistent(equations, selection)


def test_planner_terminates_on_wide_systems():
    """Planning stays fast when many auxiliaries feed many outputs.

    The subset-enumeration planner this replaces did not terminate
    for systems of this width (issue 603).
    """
    n = 48
    exprs = []
    for i in range(n):
        aux = ir.sym(f"aux_{i}")
        exprs.append(
            (
                aux,
                _chain(
                    ir.sym(f"x{i}"),
                    ("sin", "cos", "exp", "tan"),
                ),
            )
        )
        exprs.append(
            (
                ir.arr("jvp", i),
                ir.mul(aux, ir.arr("v", i)),
            )
        )
    equations = JVPEquations(exprs, min_ops_threshold=1)
    started = time.perf_counter()
    selection = plan_auxiliary_cache(equations)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0
    assert len(selection.cached_leaves) == len(
        selection.cached_leaf_order
    )
    assert selection.saved > 0
    _assert_partition_consistent(equations, selection)
