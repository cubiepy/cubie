"""Structural simplification pipeline tests."""

import pytest
import sympy as sp

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import to_sympy
from cubie.odesystems.symbolic.structural.alias_elimination import (
    eliminate_perfect_aliases,
    trivial_tearing,
)
from cubie.odesystems.symbolic.structural.derivative_block import (
    eliminate_singular_derivative_blocks,
)
from cubie.odesystems.symbolic.structural.errors import (
    ExtraEquationsSystemError,
    ExtraVariablesSystemError,
)
from cubie.odesystems.symbolic.structural.simplify import (
    structural_simplify,
)
from cubie.odesystems.symbolic.structural.symbolics import (
    DerivativeRegistry,
    fixpoint_sub,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)

T = ir.sym("t")


def syms(names):
    return tuple(ir.sym(name) for name in names.split())


def same_equations(left, right):
    return len(left) == len(right) and all(
        a[0] is b[0] and a[1] is b[1] for a, b in zip(left, right)
    )


def printed(eq):
    return f"{eq[0]} ~ {eq[1]}"


def residual(eq):
    return ir.sub(eq[1], eq[0])


def free_symbols(eq):
    return ir.free_atoms(eq[0]) | ir.free_atoms(eq[1])


def derivative_edges(state):
    return {
        (v, state.derivative_of(v))
        for v in range(len(state.fullvars))
        if state.derivative_of(v) is not None
    }


def states(result):
    return result.differential_states + result.algebraic_states


def edge_set(graph):
    return {
        (e, v)
        for e in range(graph.nsrcs())
        for v in graph.s_neighbors(e)
    }


def make_state(eqs, unknowns, knowns=(), priorities=None,
               irreducibles=None):
    names = {s.name for s in unknowns} | {s.name for s in knowns}
    registry = DerivativeRegistry(names | {"t"})
    return registry, lambda: StructuralState(
        eqs,
        registry,
        set(knowns),
        T,
        state_priorities=priorities or {},
        irreducibles=irreducibles or (),
    )


class TestEquationOrder:
    def _states(self):
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        eqs = [
            (registry.derivative(x), -k * x + z),
            (y, 2 * x),
            (z, y + 1),
        ]
        return [
            StructuralState(order, registry, {k}, T)
            for order in (eqs, eqs[::-1], [eqs[1], eqs[0], eqs[2]])
        ]

    def test_equations_sorted_by_printed_form(self):
        for state in self._states():
            texts = [printed(eq) for eq in state.eqs]
            assert texts == sorted(texts)

    def test_order_independent_of_input_order(self):
        states = self._states()
        reference = [printed(eq) for eq in states[0].original_eqs]
        for state in states[1:]:
            assert [printed(eq) for eq in state.original_eqs] == reference

    def test_original_equations_follow_sorted_equations(self):
        for state in self._states():
            for eq, original in zip(state.eqs, state.original_eqs):
                difference = to_sympy(residual(eq) - residual(original))
                assert sp.simplify(difference) == 0


class TestIndexCompaction:
    def test_rm_eqs_vars_renumbers_consistently(self):
        x, y, z, w, k = syms("x y z w k")
        registry = DerivativeRegistry({"x", "y", "z", "w", "k", "t"})
        state = StructuralState(
            [
                (registry.derivative(x), -k * x + z),
                (y, 2 * x),
                (z, y + 1),
                (w, x - z),
            ],
            registry,
            {k},
            T,
        )
        state.linear_subsys_adjmat()
        s = state
        nvars = len(state.fullvars)
        # Distinct per-variable markers to follow through renumbering.
        s.state_priorities[:] = list(range(nvars))
        s.canonical_ranks[:] = list(range(nvars, 0, -1))
        state.always_present[:] = [v % 2 == 0 for v in range(nvars)]

        old_vars = list(state.fullvars)
        old_priorities = list(s.state_priorities)
        old_ranks = list(s.canonical_ranks)
        old_present = list(state.always_present)
        old_eqs = list(state.eqs)
        old_original = list(state.original_eqs)
        old_edges = edge_set(s.graph)
        old_solvable = edge_set(s.solvable_graph)
        old_diff = derivative_edges(state)

        rm_var = state.var2idx[w]
        rm_eq = next(
            i for i, eq in enumerate(state.original_eqs) if eq[0] is y
        )
        old_to_new_eq, old_to_new_var = state.rm_eqs_vars(
            [rm_eq], [rm_var, rm_var]
        )
        assert old_to_new_eq[rm_eq] == -1
        assert old_to_new_var[rm_var] == -1

        kept_vars = [v for v in range(nvars) if old_to_new_var[v] >= 0]
        assert len(state.fullvars) == len(kept_vars)
        assert len(s.state_priorities) == len(kept_vars)
        assert len(s.canonical_ranks) == len(kept_vars)
        assert len(state.always_present) == len(kept_vars)
        for old in kept_vars:
            new = old_to_new_var[old]
            assert state.fullvars[new] is old_vars[old]
            assert state.var2idx[old_vars[old]] == new
            assert s.state_priorities[new] == old_priorities[old]
            assert s.canonical_ranks[new] == old_ranks[old]
            assert state.always_present[new] == old_present[old]

        kept_eqs = [
            e for e in range(len(old_eqs)) if old_to_new_eq[e] >= 0
        ]
        assert len(state.eqs) == len(kept_eqs)
        assert len(state.original_eqs) == len(kept_eqs)
        for old in kept_eqs:
            new = old_to_new_eq[old]
            assert state.eqs[new] is old_eqs[old]
            assert state.original_eqs[new] is old_original[old]

        for graph, edges in (
            (s.graph, old_edges),
            (s.solvable_graph, old_solvable),
        ):
            expected = {
                (old_to_new_eq[e], old_to_new_var[v])
                for e, v in edges
                if old_to_new_eq[e] >= 0 and old_to_new_var[v] >= 0
            }
            assert edge_set(graph) == expected
            assert graph.nsrcs() == len(kept_eqs)
            assert graph.ndsts() == len(kept_vars)
            for v in range(graph.ndsts()):
                assert graph.d_neighbors(v) == sorted(
                    e for e, dst in expected if dst == v
                )
        assert derivative_edges(state) == {
            (old_to_new_var[a], old_to_new_var[b])
            for a, b in old_diff
            if old_to_new_var[a] >= 0 and old_to_new_var[b] >= 0
        }


class TestIntegerMatrixDifferentiation:
    def test_integer_linear_derivative_row_added(self):
        x, y = syms("x y")
        registry = DerivativeRegistry({"x", "y", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [(dx, -x), (ir.ZERO, x - 2 * y)],
            registry,
            set(),
            T,
        )
        state.mm = state.linear_subsys_adjmat()
        dy_index = state.var_derivative(state.var2idx[y])
        constraint = next(
            i for i, eq in enumerate(state.eqs) if dx not in free_symbols(eq)
        )
        derivative = state.eq_derivative(constraint)
        mm = state.mm
        assert mm.nparentrows == state.graph.nsrcs()
        assert mm.nzrows[-1] == derivative
        row = {
            state.fullvars[v]: c
            for v, c in zip(mm.row_cols[-1], mm.row_vals[-1])
        }
        assert row == {dx: 1, state.fullvars[dy_index]: -2}
        assert mm.row_cols[-1] == state.graph.s_neighbors(
            derivative
        )


class TestVariableRanks:
    def _states(self):
        x, y, z = syms("x y z")
        registry = DerivativeRegistry({"x", "y", "z", "t"})
        dx = registry.derivative(x)
        ddx = registry.derivative(dx)
        dz = registry.derivative(z)
        eqs = [
            (ddx, -x + y),
            (dz, -z),
            (ir.ZERO, y - z),
        ]
        return registry, [
            StructuralState(order, registry, set(), T)
            for order in (eqs, eqs[::-1])
        ]

    def test_ranks_order_by_base_name_then_derivative_order(self):
        registry, states = self._states()
        reference = None
        for state in states:
            ranks = state.canonical_ranks
            by_rank = sorted(
                range(len(state.fullvars)), key=lambda i: ranks[i]
            )
            keys = [
                (base.name, order)
                for base, order in (
                    registry.base_and_order(state.fullvars[i])
                    for i in by_rank
                )
            ]
            assert keys == sorted(keys)
            assert sorted(ranks) == list(range(len(state.fullvars)))
            ranked = {v: ranks[i] for i, v in enumerate(state.fullvars)}
            if reference is None:
                reference = ranked
            assert ranked == reference

    def test_new_derivative_takes_source_rank(self):
        _, states = self._states()
        state = states[0]
        state.linear_subsys_adjmat()
        ranks = state.canonical_ranks
        y = state.var2idx[ir.sym("y")]
        dy = state.var_derivative(y)
        assert ranks[dy] == ranks[y]
        dz = state.derivative_of(state.var2idx[ir.sym("z")])
        z_t = state.add_variable(ir.sym("z_t_"), dz)
        assert ranks[z_t] == ranks[dz]
        assert len(ranks) == len(state.fullvars)


class TestVariableOrder:
    def _states(self):
        x, y, z, w = syms("x y z w")
        registry = DerivativeRegistry({"x", "y", "z", "w", "t"})
        ddx = registry.derivative(registry.derivative(x))
        dz = registry.derivative(z)
        eqs = [
            (ddx, -x + y),
            (dz, -z + w),
            (ir.ZERO, y - z),
            (ir.ZERO, w - x),
        ]
        orders = [eqs, eqs[::-1], [eqs[2], eqs[0], eqs[3], eqs[1]]]
        return registry, [
            StructuralState(order, registry, set(), T) for order in orders
        ]

    def test_order_independent_of_input_order(self):
        _, states = self._states()
        for state in states[1:]:
            assert state.fullvars == states[0].fullvars

    def test_derivatives_then_chains_then_other_unknowns(self):
        registry, states = self._states()
        fullvars = states[0].fullvars
        derivatives = {v for v in fullvars if registry.is_derivative(v)}
        occurring = set()
        for eq in states[0].eqs:
            occurring |= free_symbols(eq)
        lower_orders = set()
        for v in derivatives & occurring:
            v = registry.lower_order(v)
            while v is not None:
                lower_orders.add(v)
                v = registry.lower_order(v)
        lower_orders -= derivatives & occurring
        n_head = len(derivatives & occurring)
        n_chain = n_head + len(lower_orders)
        head = fullvars[:n_head]
        chain = fullvars[n_head:n_chain]
        tail = fullvars[n_chain:]
        assert set(head) == derivatives & occurring
        assert set(chain) == lower_orders

        def keys(group):
            return [
                (base.name, order)
                for base, order in map(registry.base_and_order, group)
            ]

        assert keys(head) == sorted(keys(head))
        assert keys(chain) == sorted(
            keys(chain), key=lambda key: (key[0], -key[1])
        )
        assert keys(tail) == sorted(keys(tail))


class TestCoefficientAdmission:
    def _state(self, coefficient):
        x, y = syms("x y")
        registry = DerivativeRegistry({"x", "y", "t"})
        return StructuralState(
            [(ir.ZERO, coefficient * x - y)],
            registry,
            set(),
            T,
        )

    @pytest.mark.parametrize("coefficient", [127, -127, 127.0])
    def test_coefficient_within_limit_enters_row(self, coefficient):
        state = self._state(coefficient)
        mm = state.linear_subsys_adjmat()
        assert mm.nzrows == [0]
        row = {
            state.fullvars[v]: c
            for v, c in zip(mm.row_cols[0], mm.row_vals[0])
        }
        x, y = syms("x y")
        assert row == {x: int(coefficient), y: -1}

    @pytest.mark.parametrize("coefficient", [128, -128, 2.5])
    def test_coefficient_beyond_limit_stays_solvable(self, coefficient):
        state = self._state(coefficient)
        state.linear_subsys_adjmat()
        all_int_vars, _ = state.find_eq_solvables(0)
        assert all_int_vars is False
        assert state.solvable_graph.s_neighbors(0) == sorted(
            state.var2idx[v] for v in syms("x y")
        )

    @pytest.mark.parametrize("conservative", [False, True])
    def test_cancelled_coefficient_drops_incidence(self, conservative):
        x, y, z = syms("x y z")
        registry = DerivativeRegistry({"x", "y", "z", "t"})
        state = StructuralState(
            [(ir.ZERO, (x + 1) * y - x * y - z)],
            registry,
            set(),
            T,
        )
        mm = state.linear_subsys_adjmat(conservative=conservative)
        y_z = [state.var2idx[y], state.var2idx[z]]
        assert state.graph.s_neighbors(0) == y_z
        assert mm.nzrows == [0]
        assert mm.row_cols[0] == y_z
        assert mm.row_vals[0] == [1, -1]

    def test_conservative_admits_unit_coefficients_only(self):
        x, y = syms("x y")
        state = self._state(2)
        state.linear_subsys_adjmat(conservative=True)
        all_int_vars, _ = state.find_eq_solvables(0, conservative=True)
        assert all_int_vars is False
        assert state.solvable_graph.s_neighbors(0) == [
            state.var2idx[y]
        ]


class TestTrivialTearing:
    def test_explicit_chain_torn_to_observed(self):
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        dx = registry.derivative(x)
        y_rhs = 2 * x
        z_rhs = y + 1
        state = StructuralState(
            [
                (dx, -k * x),
                (y, y_rhs),
                (z, z_rhs),
            ],
            registry,
            {k},
            T,
        )
        trivial_tearing(state)
        assert set(state.fullvars) == {x, dx}
        assert [eq[0] for eq in state.eqs] == [dx]
        assert len(state.additional_observed) == 2
        observed = dict(state.additional_observed)
        assert observed[y] is y_rhs
        assert observed[z] is z_rhs
        assert state.graph.nsrcs() == len(state.eqs)
        assert state.graph.ndsts() == len(state.fullvars)

    def test_irreducible_and_self_referencing_not_torn(self):
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                (dx, -k * x),
                (y, 2 * x),
                (z, z * x + 1),
            ],
            registry,
            {k},
            T,
            irreducibles=[y],
        )
        n_eqs = len(state.eqs)
        trivial_tearing(state)
        assert set(state.fullvars) == {x, dx, y, z}
        assert len(state.eqs) == n_eqs


class TestExplicitSystems:
    def test_removed_expression_simplify_option_raises(self):
        """The removed expression option fails explicitly."""

        x = ir.sym("x")
        registry = DerivativeRegistry({"x", "t"})
        state = StructuralState(
            [(registry.derivative(x), -x)],
            registry,
            set(),
            T,
        )
        with pytest.raises(TypeError, match="unexpected keyword"):
            structural_simplify(state, simplify=True)

    def test_plain_ode_passthrough(self):
        x, k = syms("x k")
        registry = DerivativeRegistry({"x", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [(dx, -k * x)], registry, {k}, T
        )
        result = structural_simplify(state)
        assert states(result) == [x]
        assert sp.simplify(to_sympy(result.dxdt[x] + k * x)) == 0
        assert result.residuals == []

    def test_observed_chain_extracted(self):
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                (dx, -k * x + z),
                (y, 2 * x),
                (z, y + 1),
            ],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        assert states(result) == [x]
        obs = dict(result.observed)
        assert set(obs) == {y, z}
        # Substituting observed into dxdt reproduces the dynamics.
        rhs = result.dxdt[x]
        full = rhs
        for _ in range(3):
            full = ir.xreplace(full, obs)
        expected = -k * x + 2 * x + 1
        assert sp.simplify(to_sympy(full - expected)) == 0

    def test_perfect_alias_eliminated(self):
        x, y, k = syms("x y k")
        registry = DerivativeRegistry({"x", "y", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                (dx, -k * y),
                (ir.ZERO, x - y),
            ],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        assert states(result) == [x]
        obs = dict(result.observed)
        assert obs[y] is x
        assert sp.simplify(to_sympy(result.dxdt[x] + k * x)) == 0

    def test_negated_alias_sign(self):
        x, y, k = syms("x y k")
        registry = DerivativeRegistry({"x", "y", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                (dx, -k * y),
                (ir.ZERO, x + y),
            ],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        obs = dict(result.observed)
        assert obs[y] is -x
        assert sp.simplify(to_sympy(result.dxdt[x] - k * x)) == 0

    def test_alias_target_prefers_priority(self):
        x, y = syms("x y")
        registry = DerivativeRegistry({"x", "y", "t"})
        dx = registry.derivative(x)
        dy = registry.derivative(y)
        # x and y are aliased differentiated states with
        # integer-linear dynamics; after aliasing, one of the now
        # duplicate differential equations reduces away exactly, and
        # priority keeps y as the surviving state.
        state = StructuralState(
            [
                (dx, -3 * x),
                (dy, -3 * y),
                (ir.ZERO, x - y),
            ],
            registry,
            set(),
            T,
            state_priorities={y: 5},
        )
        result = structural_simplify(state)
        assert states(result) == [y]
        assert dict(result.observed)[x] is y
        assert sp.simplify(to_sympy(result.dxdt[y] + 3 * y)) == 0

    def test_derivative_priority_reaches_base(self):
        x, y = syms("x y")
        registry = DerivativeRegistry({"x", "y", "t"})
        dx = registry.derivative(x)
        dy = registry.derivative(y)
        # The priority set on dx outranks y's and carries down to x,
        # so x survives the alias.
        state = StructuralState(
            [
                (dx, -3 * x),
                (dy, -3 * y),
                (ir.ZERO, x - y),
            ],
            registry,
            set(),
            T,
            state_priorities={dx: 10, y: 5},
        )
        priorities = state.state_priorities
        assert priorities[state.var2idx[x]] == 10
        assert priorities[state.var2idx[dx]] == 10
        result = structural_simplify(state)
        assert states(result) == [x]
        assert dict(result.observed)[y] is x
        assert sp.simplify(to_sympy(result.dxdt[x] + 3 * x)) == 0

    def test_irreducible_not_eliminated(self):
        x, y, k = syms("x y k")
        registry = DerivativeRegistry({"x", "y", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                (dx, -k * y),
                (y, x),
            ],
            registry,
            {k},
            T,
            irreducibles=[y],
        )
        result = structural_simplify(state)
        # y must survive as a solver unknown (algebraic state).
        assert y in states(result)


class TestAlgebraicSystems:
    def test_nonlinear_algebraic_solvable_becomes_observed(self):
        x, k = syms("x k")
        z = ir.sym("z")
        registry = DerivativeRegistry({"x", "z", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                (dx, -k * x + z),
                (ir.ZERO, z - x**2),
            ],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        assert states(result) == [x]
        obs = dict(result.observed)
        assert sp.simplify(to_sympy(obs[z] - x**2)) == 0

    def test_unsolvable_algebraic_torn_to_residual(self):
        x, z = syms("x z")
        registry = DerivativeRegistry({"x", "z", "t"})
        dx = registry.derivative(x)
        # z**5 + z - x = 0 is not explicitly solvable for z.
        state = StructuralState(
            [
                (dx, -z),
                (ir.ZERO, z**5 + z - x),
            ],
            registry,
            set(),
            T,
        )
        result = structural_simplify(state)
        assert result.differential_states == [x]
        assert result.algebraic_states == [z]
        assert len(result.residuals) == 1
        expected = z**5 + z - x
        assert sp.simplify(
            to_sympy(result.residuals[0] - expected)
        ) == 0


class TestAliasEdgeCases:
    def test_conflicting_aliases_force_zero(self):
        # x = y and x = -y can only hold together at zero: the
        # conflict group pins both variables to 0.
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        dz = registry.derivative(z)
        state = StructuralState(
            [
                (dz, -k * z + x),
                (ir.ZERO, x - y),
                (ir.ZERO, x + y),
            ],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        obs = dict(result.observed)
        assert obs[x] is ir.ZERO
        assert obs[y] is ir.ZERO
        assert states(result) == [z]
        assert sp.simplify(to_sympy(result.dxdt[z] + k * z)) == 0

    def test_sign_chain_three_variables(self):
        # x = -y, y = -z: signs compose along the chain.
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                (dx, -k * x),
                (ir.ZERO, x + y),
                (ir.ZERO, y + z),
            ],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        assert states(result) == [x]
        obs = dict(result.observed)
        assert obs[y] is -x
        assert obs[z] is x

    def test_derivative_chain_sign_propagation(self):
        # y = -x where x carries a second-order derivative chain:
        # the alias must propagate through the derivative chain.
        x, y = syms("x y")
        registry = DerivativeRegistry({"x", "y", "t"})
        d1 = registry.derivative(x)
        d2 = registry.derivative(d1)
        state = StructuralState(
            [
                (d2, -x),
                (ir.ZERO, y + x),
            ],
            registry,
            set(),
            T,
        )
        result = structural_simplify(state)
        assert len(result.differential_states) == 2
        obs = dict(result.observed)
        assert obs[y] is -x

    def _second_order_alias_state(self, registry):
        # x carries a second derivative and is aliased to the
        # higher-priority y, which has no derivative of its own.
        x, y = syms("x y")
        dx = registry.derivative(x)
        return StructuralState(
            [
                (registry.derivative(dx), -x),
                (ir.ZERO, y - x),
            ],
            registry,
            set(),
            T,
            state_priorities={y: 5},
        )

    def test_removed_derivative_chain_maps_to_target_chain(self):
        x, y = syms("x y")
        registry = DerivativeRegistry({"x", "y", "t"})
        dy = registry.derivative(y)
        ddy = registry.derivative(dy)
        state = self._second_order_alias_state(registry)
        eliminate_perfect_aliases(state)
        assert set(state.fullvars) == {y, dy, ddy}
        observed = dict(state.additional_observed)
        assert observed == {x: y}
        y_index = state.var2idx[y]
        dy_index = state.derivative_of(y_index)
        assert state.fullvars[dy_index] is dy
        assert state.fullvars[state.derivative_of(dy_index)] is ddy
        assert any(
            same_equations([eq], [(ddy, -y)]) for eq in state.eqs
        )

    def test_second_order_alias_reduces_to_oscillator(self):
        y = ir.sym("y")
        registry = DerivativeRegistry({"x", "y", "t"})
        state = self._second_order_alias_state(registry)
        result = structural_simplify(state)
        assert len(result.differential_states) == 2
        velocity = result.dxdt[y]
        assert velocity in states(result)
        assert sp.simplify(to_sympy(result.dxdt[velocity] + y)) == 0

    def test_priority_tie_warns(self):
        x, y, z = syms("x y z")
        registry = DerivativeRegistry({"x", "y", "z", "t"})
        dz = registry.derivative(z)
        state = StructuralState(
            [
                (dz, x + y),
                (ir.ZERO, x - y),
                (ir.ZERO, x + y - z),
            ],
            registry,
            set(),
            T,
            state_priorities={x: 100, y: 100},
        )
        with pytest.warns(UserWarning, match="state_priority"):
            structural_simplify(state)


class TestSingularIntegerSCC:
    def _singular_state(self):
        x, y, z, w = syms("x y z w")
        registry = DerivativeRegistry({"x", "y", "z", "w", "t"})
        dz = registry.derivative(z)
        return StructuralState(
            [
                (dz, w),
                (ir.ZERO, x + y + w),
                (ir.ZERO, 2 * x + 2 * y - w),
                (ir.ZERO, w**5 + w - z),
            ],
            registry,
            set(),
            T,
        )

    def test_singular_integer_block_raises(self):
        # 2*eq1 - eq2 pins w = 0, leaving x, y underdetermined:
        # singularity removal exposes the deficiency and the
        # consistency check reports a structurally singular system.
        from cubie.odesystems.symbolic.structural.errors import (
            InvalidSystemError,
        )

        with pytest.raises(InvalidSystemError):
            structural_simplify(self._singular_state())

    def test_conservative_excludes_nonunit_rows(self):
        # Conservative mode admits only unit coefficients into the
        # integer subsystem; the 2x + 2y - w row must leave mm
        # entirely rather than desync its coefficient row, and the
        # system tears structurally: the unit row x + y + w = 0 is
        # solved for one of x, y and the 2x + 2y - w row stays a
        # residual unchanged.
        x, y, w = syms("x y w")
        result = structural_simplify(
            self._singular_state(), conservative=True
        )
        assert len(result.residuals) == len(result.algebraic_states)
        assert len(result.observed) == 1
        solved, expression = result.observed[0]
        assert solved in (x, y)
        assert sp.simplify(
            to_sympy(solved - expression - (x + y + w))
        ) == 0
        nonunit_row = 2 * x + 2 * y - w
        assert any(
            sp.simplify(to_sympy(r - nonunit_row)) == 0
            or sp.simplify(to_sympy(r + nonunit_row)) == 0
            for r in result.residuals
        )


class TestPantelidesAndDummyDerivatives:
    def make_pendulum(self, priorities=None):
        x, y, vx, vy, Tn, g, L = syms("x y vx vy T g L")
        registry = DerivativeRegistry(
            {"x", "y", "vx", "vy", "T", "g", "L", "t"}
        )
        eqs = [
            (registry.derivative(x), vx),
            (registry.derivative(y), vy),
            (registry.derivative(vx), Tn * x),
            (registry.derivative(vy), Tn * y - g),
            (ir.ZERO, x**2 + y**2 - L**2),
        ]
        state = StructuralState(
            eqs,
            registry,
            {g, L},
            T,
            state_priorities=priorities or {},
        )
        return state, (x, y, vx, vy, Tn, g, L)

    def test_pendulum_balanced_reduction(self):
        state, symbols = self.make_pendulum()
        result = structural_simplify(state)
        # 2 differential + 3 algebraic states, 3 residuals: the
        # constraint and its two time derivatives.
        assert len(result.differential_states) == 2
        assert len(result.algebraic_states) == 3
        assert len(result.residuals) == 3
        x, y = symbols[0], symbols[1]
        L = symbols[6]
        # The original constraint survives as a residual.
        constraint = x**2 + y**2 - L**2
        assert any(
            sp.simplify(to_sympy(r - constraint)) == 0
            for r in result.residuals
        )

    def test_pendulum_priorities_select_states(self):
        # Without priorities y and vy are kept; priority keeps x and vx.
        x, vx = syms("x vx")
        state, _ = self.make_pendulum(priorities={x: 10, vx: 10})
        result = structural_simplify(state)
        assert x in result.differential_states
        assert vx in result.differential_states

    def test_pendulum_bare_index_reduction(self):
        # dummy_derivative=False runs bare Pantelides index
        # reduction: the matched (differentiated) position
        # equations are kept, so first-order lowering introduces
        # velocity aliases x_t/y_t alongside vx/vy, and only the
        # acceleration-level constraint survives as the residual.
        state, symbols = self.make_pendulum()
        result = structural_simplify(state, dummy_derivative=False)
        x, y, vx, vy, Tn, g, L = symbols
        assert set(result.algebraic_states) == {Tn}
        assert len(result.residuals) == 1
        assert len(result.differential_states) == 6
        for sym in (x, y, vx, vy):
            assert sym in result.differential_states
        names = {s.name: s for s in result.differential_states}
        x_t, y_t = names["x_t"], names["y_t"]
        accel = (
            2 * x_t**2
            + 2 * Tn * x**2
            + 2 * y_t**2
            + 2 * y * (Tn * y - g)
        )
        assert sp.simplify(
            to_sympy(result.residuals[0] - accel)
        ) == 0

    def test_cancelled_unknown_needs_no_derivative(self):
        # w's coefficient in the constraint cancels, so differentiating
        # the constraint needs no derivative of w.
        x, y, w, c = syms("x y w c")
        resolved = []
        for extra in ((w + 1) * c - w * c, c):
            registry = DerivativeRegistry({"x", "y", "w", "c", "t"})
            state = StructuralState(
                [
                    (registry.derivative(x), y),
                    (ir.ZERO, x + extra - T),
                    (ir.ZERO, w - y),
                ],
                registry,
                {c},
                T,
            )
            observed = dict(structural_simplify(state).observed)
            resolved.append(
                {
                    sym: fixpoint_sub(rhs, observed)
                    for sym, rhs in observed.items()
                }
            )
        cancelled, plain = resolved
        assert set(cancelled) == set(plain)
        for sym, rhs in plain.items():
            assert sp.simplify(to_sympy(cancelled[sym] - rhs)) == 0

    def test_second_order_pendulum_bare_index_reduction(self):
        # Bare reduction of the pendulum given in second-order form
        # integrates the given accelerations and keeps the
        # acceleration-level constraint.
        x, y, lam, g = syms("x y lam g")
        registry = DerivativeRegistry({"x", "y", "lam", "g", "t"})
        dx, dy = registry.derivative(x), registry.derivative(y)
        state = StructuralState(
            [
                (registry.derivative(dx), lam * x),
                (registry.derivative(dy), lam * y - g),
                (ir.ZERO, x**2 + y**2 - 1),
            ],
            registry,
            {g},
            T,
        )
        result = structural_simplify(state, dummy_derivative=False)
        assert result.algebraic_states == [lam]
        x_t, y_t = result.dxdt[x], result.dxdt[y]
        assert set(result.differential_states) == {x, y, x_t, y_t}
        assert sp.simplify(to_sympy(result.dxdt[x_t] - lam * x)) == 0
        assert sp.simplify(to_sympy(result.dxdt[y_t] - (lam * y - g))) == 0
        accel = (
            2 * x_t**2
            + 2 * lam * x**2
            + 2 * y_t**2
            + 2 * y * (lam * y - g)
        )
        assert len(result.residuals) == 1
        assert sp.simplify(to_sympy(result.residuals[0] - accel)) == 0

    def test_higher_order_input_lowered(self):
        x, w = syms("x w")
        registry = DerivativeRegistry({"x", "w", "t"})
        d1 = registry.derivative(x)
        d2 = registry.derivative(d1)
        # x'' = -x (harmonic oscillator given as second order).
        state = StructuralState(
            [(d2, -x)],
            registry,
            set(),
            T,
        )
        result = structural_simplify(state)
        assert len(result.differential_states) == 2
        assert not result.residuals
        assert x in result.differential_states
        # The generated companion state x_t satisfies d(x) = x_t and
        # d(x_t) = -x.
        other = [
            s for s in result.differential_states if s != x
        ][0]
        assert result.dxdt[x] is other
        assert sp.simplify(to_sympy(result.dxdt[other] + x)) == 0


class TestConsistencyErrors:
    def test_overdetermined_raises(self):
        x, k = syms("x k")
        registry = DerivativeRegistry({"x", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                (dx, -k * x),
                (ir.ZERO, x - 1),
                (ir.ZERO, x - 2),
            ],
            registry,
            {k},
            T,
        )
        with pytest.raises(ExtraEquationsSystemError):
            structural_simplify(state)

    def test_underdetermined_raises(self):
        x, z = syms("x z")
        registry = DerivativeRegistry({"x", "z", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [(dx, -x + z)],
            registry,
            set(),
            T,
        )
        with pytest.raises(ExtraVariablesSystemError):
            structural_simplify(state)

    def test_alias_only_group_counts_as_unknown(self):
        # Alias elimination removes y ~ z, leaving its target in no
        # equation; the target still counts as an unknown.
        x, y, z = syms("x y z")
        registry = DerivativeRegistry({"x", "y", "z", "t"})
        state = StructuralState(
            [(registry.derivative(x), -x), (y, z)],
            registry,
            set(),
            T,
        )
        with pytest.raises(ExtraVariablesSystemError):
            structural_simplify(state)

    def test_underdetermined_tearing_only_mode_rejected(self):
        x, z = syms("x z")
        registry = DerivativeRegistry({"x", "z", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [(dx, -x + z)],
            registry,
            set(),
            T,
        )
        with pytest.raises(
            ExtraVariablesSystemError, match=r"Sym\(z\)"
        ):
            structural_simplify(state, fully_determined=False)


def _mentions_derivative(result, registry):
    """Whether any output expression reads a state's derivative symbol."""

    expressions = list(result.dxdt.values()) + list(result.residuals)
    expressions.extend(expr for _, expr in result.observed)
    return any(
        registry.is_derivative(atom)
        for expr in expressions
        for atom in ir.free_atoms(expr)
    )


class TestSingularDerivativeBlocks:
    """Dependent derivative rows become constraints before analysis."""

    def make_pair(self, coefficient):
        x, y, z = syms("x y z")
        registry = DerivativeRegistry({"x", "y", "z", "c", "t"})
        dx, dy, dz = (registry.derivative(s) for s in (x, y, z))
        c = coefficient
        eqs = [
            (-c * dx + c * dy, -x + z),
            (c * dx - c * dy, ir.call("exp", y) - x**3 - 1),
            (dz, -z + x),
        ]
        return registry, eqs, (x, y, z)

    def test_numeric_block_rewritten_to_constraint(self):
        registry, eqs, (x, y, z) = self.make_pair(ir.num(1e-6))
        state = StructuralState(eqs, registry, set(), T)
        rewritten = eliminate_singular_derivative_blocks(state)
        assert len(rewritten) == 1
        constraint = state.eqs[rewritten[0]]
        assert ir.is_zero(constraint[0])
        # The rows sum to the derivative-free node constraint.
        expected = (-x + z) + (ir.call("exp", y) - x**3 - 1)
        assert sp.simplify(to_sympy(constraint[1] - expected)) == 0
        incidence = state.graph.s_neighbors(rewritten[0])
        assert {state.fullvars[v] for v in incidence} == {x, y, z}

    @pytest.mark.parametrize(
        "coefficient",
        ["numeric", "parameter", "sum"],
        ids=["float", "param", "sum"],
    )
    def test_pair_reduces_to_index_one(self, coefficient):
        knowns = set()
        if coefficient == "numeric":
            c = ir.num(1e-6)
        elif coefficient == "parameter":
            c = ir.sym("c")
            knowns = {c}
        else:
            c = ir.add(ir.sym("c1a"), ir.sym("c1b"))
            knowns = {ir.sym("c1a"), ir.sym("c1b")}
        registry, eqs, (x, y, z) = self.make_pair(c)
        state = StructuralState(eqs, registry, knowns, T)
        result = structural_simplify(state)
        # Constraint and its derivative, both reading algebraic states.
        assert len(result.residuals) == 2
        assert len(result.algebraic_states) == 2
        assert z in result.differential_states
        read = frozenset().union(
            *(ir.free_atoms(r) for r in result.residuals)
        )
        assert set(result.algebraic_states) <= read
        assert not _mentions_derivative(result, registry)
        constraint = (-x + z) + (ir.call("exp", y) - x**3 - 1)
        assert any(
            sp.simplify(to_sympy(r + constraint)) == 0
            or sp.simplify(to_sympy(r - constraint)) == 0
            for r in result.residuals
        )

    def test_three_row_dependence_solves_explicitly(self):
        x, y, z, a, b = syms("x y z a b")
        registry = DerivativeRegistry({"x", "y", "z", "a", "b", "t"})
        dx, dy, dz = (registry.derivative(s) for s in (x, y, z))
        eqs = [
            (a * dx + b * dy, -x),
            (b * dy + a * dz, -y),
            (a * dx + 2 * b * dy + a * dz, -z),
        ]
        state = StructuralState(eqs, registry, {a, b}, T)
        assert eliminate_singular_derivative_blocks(state) == [2]
        assert sp.simplify(
            to_sympy(state.eqs[2][1]) - to_sympy(x + y - z)
        ) == 0 or sp.simplify(
            to_sympy(state.eqs[2][1]) + to_sympy(x + y - z)
        ) == 0

    @pytest.mark.parametrize("allow_parameter", [True, False])
    def test_parameter_pivot_never_divides(self, allow_parameter):
        x, y, p = syms("x y p")
        registry = DerivativeRegistry({"x", "y", "p", "t"})
        dx = registry.derivative(x)
        eqs = [(p * dx, -x), (dx, -y)]
        state = StructuralState(eqs, registry, {p}, T)
        assert eliminate_singular_derivative_blocks(
            state, allow_parameter=allow_parameter
        ) == [0]
        constraint = state.eqs[0]
        assert ir.is_zero(constraint[0])
        expected = x - p * y
        assert (
            sp.simplify(to_sympy(constraint[1] - expected)) == 0
            or sp.simplify(to_sympy(constraint[1] + expected)) == 0
        )
        assert same_equations([state.eqs[1]], [eqs[1]])

    def test_rejected_parameter_pivot_leaves_rows(self):
        x, y, p, q = syms("x y p q")
        registry = DerivativeRegistry({"x", "y", "p", "q", "t"})
        dx = registry.derivative(x)
        eqs = [(p * dx, -x), (q * dx, -y)]
        state = StructuralState(eqs, registry, {p, q}, T)
        assert eliminate_singular_derivative_blocks(
            state, allow_parameter=False
        ) == []
        assert same_equations(state.eqs, eqs)

    def test_independent_block_untouched(self):
        x, y = syms("x y")
        registry = DerivativeRegistry({"x", "y", "t"})
        dx, dy = registry.derivative(x), registry.derivative(y)
        eqs = [
            (2 * dx + 3 * dy, -x),
            (3 * dy - 2 * dx, -y),
        ]
        state = StructuralState(eqs, registry, set(), T)
        before = list(state.eqs)
        assert eliminate_singular_derivative_blocks(state) == []
        assert same_equations(state.eqs, before)

    @pytest.mark.parametrize("output", ["a", "z"])
    @pytest.mark.parametrize("dynamics", ["derivative_lhs", "state_lhs"])
    def test_state_only_row_keeps_derivative(self, output, dynamics):
        x, d, o = syms(f"x d {output}")
        registry = DerivativeRegistry({"x", "d", output, "t"})
        dx = registry.derivative(x)
        own = (dx, -x) if dynamics == "derivative_lhs" else (x, -dx)
        eqs = [own, (o, dx + d + ir.num(0.5))]
        state = StructuralState(eqs, registry, {d}, T)
        rewritten = eliminate_singular_derivative_blocks(state)
        assert len(rewritten) == 1
        kept = [
            eq for index, eq in enumerate(state.eqs)
            if index not in rewritten
        ]
        assert same_equations(kept, [own])

    def test_unknown_coefficient_excluded(self):
        x, y, w = syms("x y w")
        registry = DerivativeRegistry({"x", "y", "w", "t"})
        dx, dy, dw = (registry.derivative(s) for s in (x, y, w))
        eqs = [
            (w * dx - w * dy, -x),
            (-w * dx + w * dy, -y + 1),
            (dw, -w),
        ]
        state = StructuralState(eqs, registry, set(), T)
        assert eliminate_singular_derivative_blocks(state) == []


class TestExactLinearSCCRewrite:
    def test_integer_constraint_block_solves_explicitly(self):
        # The integer-linear SCC solves explicitly; x and x_t observed.
        x, y, z = syms("x y z")
        registry = DerivativeRegistry({"x", "y", "z", "t"})
        dx, dy, dz = (registry.derivative(s) for s in (x, y, z))
        eqs = [
            (dx + dy, -x),
            (dy + dz, -y),
            (ir.ZERO, x + y - z),
        ]
        state = StructuralState(eqs, registry, set(), T)
        result = structural_simplify(state)
        assert result.residuals == []
        assert set(states(result)) == {y, z}
        assert not _mentions_derivative(result, registry)
        obs = dict(result.observed)
        x_t = ir.sym("x_t")
        assert set(obs) == {x, x_t}
        full = {}
        for sym, rhs in result.dxdt.items():
            for _ in range(3):
                rhs = ir.xreplace(rhs, obs)
            full[sym] = rhs
        # x = z - y, x_t = y - 2x: dy = x - y = z - 2y, dz = -x = y - z.
        assert sp.simplify(to_sympy(full[y] - (z - 2 * y))) == 0
        assert sp.simplify(to_sympy(full[z] - (y - z))) == 0

    def test_singular_integer_block_warns_and_tears(self):
        # x + 2y + z = 0 differentiates to the sum of the two
        # derivative rows, so the derivative block is singular.
        x, y, z = syms("x y z")
        registry = DerivativeRegistry({"x", "y", "z", "t"})
        dx, dy, dz = (registry.derivative(s) for s in (x, y, z))
        eqs = [
            (dx + dy, -x),
            (dy + dz, -y),
            (ir.ZERO, x + 2 * y + z),
        ]
        state = StructuralState(eqs, registry, set(), T)
        with pytest.warns(UserWarning, match="Integer-linear"):
            result = structural_simplify(state)
        assert len(result.residuals) == len(result.algebraic_states)
        assert result.residuals
