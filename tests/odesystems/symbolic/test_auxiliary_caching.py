"""Unit tests for the min-cut auxiliary cache planner."""

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
    assert selection.read_price >= equations.read_price
    # The plan is worth its slots at the price it was solved at.
    assert selection.saved >= selection.read_price * len(cached)
    # Runtime assignments may read runtime or cached values only.
    for lhs in runtime:
        for dep in equations.dependencies.get(lhs, set()):
            assert dep in runtime or dep in cached
    # A removed-but-uncached node's consumers must all be removed.
    for lhs in removed - cached:
        for consumer in equations.dependents.get(lhs, set()):
            assert consumer in removed
        assert equations.jvp_usage.get(lhs, 0) == 0
    # Prepare assignments are self-contained.
    for lhs in prepare:
        for dep in equations.dependencies.get(lhs, set()):
            assert dep in prepare
    # Duplicate cost covers exactly the fill work that stays runtime.
    assert selection.duplicate_cost == sum(
        equations.ops_cost.get(lhs, 0) for lhs in prepare & runtime
    )
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
    equations = JVPEquations(exprs, max_cached_terms=0, read_price=1)
    selection = plan_auxiliary_cache(equations)
    assert selection.cached_leaf_order == ()
    assert selection.runtime_nodes == tuple(equations.non_jvp_order)


def test_dead_auxiliary_is_never_cached():
    """An assignment feeding no jvp term stays runtime and uncached."""
    x0 = ir.sym("x0")
    dead = ir.sym("aux_dead")
    live = ir.sym("aux_live")
    exprs = [
        (dead, _chain(x0, ("sin", "cos", "exp", "tan"))),
        (live, _chain(x0, ("sinh", "cosh", "tanh", "log"))),
        (ir.arr("jvp", 0), ir.mul(live, ir.arr("v", 0))),
    ]
    equations = JVPEquations(exprs, read_price=1)
    selection = plan_auxiliary_cache(equations)
    assert dead not in selection.cached_leaf_order
    assert dead in selection.runtime_nodes
    assert live in selection.cached_leaf_order
    _assert_partition_consistent(equations, selection)


def test_cheap_leaves_below_read_price_select_nothing():
    """Leaves cheaper than the read price stay runtime."""
    x0, x1 = ir.sym("x0"), ir.sym("x1")
    aux = ir.sym("aux_a")
    exprs = [
        (aux, ir.add(x0, x1)),
        (ir.arr("jvp", 0), ir.mul(aux, ir.arr("v", 0))),
    ]
    equations = JVPEquations(exprs, read_price=10)
    selection = plan_auxiliary_cache(equations)
    assert selection.cached_leaf_order == ()
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
    equations = JVPEquations(exprs, read_price=8)
    selection = plan_auxiliary_cache(equations)
    assert cse0 in selection.cached_leaf_order
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
    equations = JVPEquations(exprs, read_price=8)
    selection = plan_auxiliary_cache(equations)
    assert inner in selection.cached_leaf_order
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
    equations = JVPEquations(exprs, read_price=1)
    selection = plan_auxiliary_cache(equations)
    # Caching top removes dep as well: nothing else reads it.
    assert selection.cached_leaf_order == (top,)
    assert dep in selection.removal_nodes
    assert dep not in selection.runtime_nodes
    assert dep in selection.prepare_nodes
    assert selection.saved == (
        equations.ops_cost[top] + equations.ops_cost[dep]
    )
    _assert_partition_consistent(equations, selection)


def test_cheap_frontier_retires_shared_producer_network():
    """Caching a cheap frontier wins when it removes its producers."""
    x0, x1 = ir.sym("x0"), ir.sym("x1")
    prod_a = ir.sym("aux_prod_a")
    prod_b = ir.sym("aux_prod_b")
    front_a = ir.sym("aux_front_a")
    front_b = ir.sym("aux_front_b")
    exprs = [
        (prod_a, _chain(x0, ("sin", "cos", "exp"))),
        (prod_b, _chain(x1, ("tan", "sinh", "cosh"))),
        (front_a, ir.add(prod_a, prod_b)),
        (front_b, ir.mul(prod_a, prod_b)),
        (ir.arr("jvp", 0), ir.mul(front_a, ir.arr("v", 0))),
        (ir.arr("jvp", 1), ir.mul(front_b, ir.arr("v", 1))),
    ]
    equations = JVPEquations(exprs, read_price=8)
    assert equations.ops_cost[front_a] < equations.read_price
    assert equations.ops_cost[front_b] < equations.read_price
    selection = plan_auxiliary_cache(equations)
    assert set(selection.cached_leaf_order) == {front_a, front_b}
    assert prod_a in selection.removal_nodes
    assert prod_b in selection.removal_nodes
    assert selection.saved == sum(
        equations.ops_cost[node]
        for node in (prod_a, prod_b, front_a, front_b)
    )
    assert selection.duplicate_cost == 0
    _assert_partition_consistent(equations, selection)


def test_duplicate_cost_counts_shared_producers():
    """A producer with a live runtime consumer is computed twice."""
    x0 = ir.sym("x0")
    producer = ir.sym("aux_producer")
    heavy = ir.sym("aux_heavy")
    light = ir.sym("aux_light")
    exprs = [
        (producer, _chain(x0, ("sin", "cos", "exp"))),
        (heavy, _chain(producer, ("tan", "sinh", "cosh", "log"))),
        (light, ir.add(producer, x0)),
        (ir.arr("jvp", 0), ir.mul(heavy, ir.arr("v", 0))),
        (ir.arr("jvp", 1), ir.mul(light, ir.arr("v", 1))),
    ]
    equations = JVPEquations(exprs, max_cached_terms=1, read_price=8)
    selection = plan_auxiliary_cache(equations)
    assert selection.cached_leaf_order == (heavy,)
    assert producer in selection.runtime_nodes
    assert producer in selection.prepare_nodes
    assert selection.duplicate_cost == equations.ops_cost[producer]
    assert selection.fill_cost == (
        equations.ops_cost[producer] + equations.ops_cost[heavy]
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
    equations = JVPEquations(exprs, read_price=1)
    assert mixes_v in equations.v_dependent_nodes
    assert downstream in equations.v_dependent_nodes
    selection = plan_auxiliary_cache(equations)
    assert selection.cached_leaf_order == ()
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
    equations = JVPEquations(exprs, max_cached_terms=1, read_price=1)
    assert equations.ops_cost[heavy] > equations.ops_cost[wide]
    selection = plan_auxiliary_cache(equations)
    assert selection.cached_leaf_order == (heavy,)
    assert wide in selection.runtime_nodes
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
    equations = JVPEquations(exprs, max_cached_terms=2, read_price=1)
    selection = plan_auxiliary_cache(equations)
    assert len(selection.cached_leaf_order) == 2
    # The two most expensive chains win the slots.
    assert set(selection.cached_leaf_order) == {leaves[3], leaves[2]}
    # The recorded price is the bisected one that fits the cap.
    assert selection.read_price > equations.read_price
    _assert_partition_consistent(equations, selection)


def test_slot_order_is_evaluation_order():
    """Slot indices bind to evaluation order, not selection value."""
    x0, x1 = ir.sym("x0"), ir.sym("x1")
    aux_a = ir.sym("aux_a")
    aux_b = ir.sym("aux_b")
    exprs = [
        (aux_a, _chain(x0, ("tanh", "log", "asin"))),
        (aux_b, _chain(x1, ("sin", "cos", "exp", "tan", "sinh"))),
        (ir.arr("jvp", 0), ir.mul(aux_a, ir.arr("v", 0))),
        (ir.arr("jvp", 1), ir.mul(aux_b, ir.arr("v", 1))),
    ]
    equations = JVPEquations(exprs, read_price=5)
    selection = plan_auxiliary_cache(equations)
    assert selection.cached_leaf_order == (aux_a, aux_b)
    assert selection.saved == (
        equations.ops_cost[aux_a] + equations.ops_cost[aux_b]
    )
    _assert_partition_consistent(equations, selection)


def test_planner_terminates_on_wide_systems():
    """Planning stays fast when many auxiliaries feed many outputs."""
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
    equations = JVPEquations(exprs, read_price=1)
    started = time.perf_counter()
    selection = plan_auxiliary_cache(equations)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0
    assert selection.saved > 0
    _assert_partition_consistent(equations, selection)
