"""Structural simplification pipeline tests."""

import pytest
import sympy as sp

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import to_sympy
from cubie.odesystems.symbolic.structural.alias_elimination import (
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
)
from cubie.odesystems.symbolic.structural.system_structure import (
    Equation,
    StructuralState,
)

T = ir.sym("t")


def syms(names):
    return tuple(ir.sym(name) for name in names.split())


def make_state(eqs, unknowns, knowns=(), priorities=None,
               irreducibles=None):
    names = {s.name for s in unknowns} | {s.name for s in knowns}
    registry = DerivativeRegistry(names | {"t"})
    return registry, lambda: StructuralState(
        eqs,
        unknowns,
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
            Equation(registry.derivative(x), -k * x + z),
            Equation(y, 2 * x),
            Equation(z, y + 1),
        ]
        return [
            StructuralState(order, [x, y, z], registry, {k}, T)
            for order in (eqs, eqs[::-1], [eqs[1], eqs[0], eqs[2]])
        ]

    def test_equations_sorted_by_printed_form(self):
        for state in self._states():
            printed = [str(eq) for eq in state.eqs]
            assert printed == sorted(printed)

    def test_order_independent_of_input_order(self):
        states = self._states()
        reference = [str(eq) for eq in states[0].original_eqs]
        for state in states[1:]:
            assert [str(eq) for eq in state.original_eqs] == reference

    def test_original_equations_follow_sorted_equations(self):
        for state in self._states():
            for eq, original in zip(state.eqs, state.original_eqs):
                difference = to_sympy(eq.residual() - original.residual())
                assert sp.simplify(difference) == 0


class TestIndexCompaction:
    def test_rm_eqs_vars_renumbers_consistently(self):
        x, y, z, w, k = syms("x y z w k")
        registry = DerivativeRegistry({"x", "y", "z", "w", "k", "t"})
        state = StructuralState(
            [
                Equation(registry.derivative(x), -k * x + z),
                Equation(y, 2 * x),
                Equation(z, y + 1),
                Equation(w, x - z),
            ],
            [x, y, z, w],
            registry,
            {k},
            T,
        )
        state.find_solvables()
        s = state.structure
        s.complete()
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
        old_edges = set(s.graph.edges())
        old_solvable = set(s.solvable_graph.edges())
        old_diff = list(s.var_to_diff.edges())

        rm_var = state.var2idx[w]
        rm_eq = next(
            i for i, eq in enumerate(state.original_eqs) if eq.lhs is y
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
            assert set(graph.edges()) == expected
            assert graph.nsrcs() == len(kept_eqs)
            assert graph.ndsts() == len(kept_vars)
            for v in range(graph.ndsts()):
                assert graph.d_neighbors(v) == sorted(
                    e for e, dst in expected if dst == v
                )
        assert set(s.var_to_diff.edges()) == {
            (old_to_new_var[a], old_to_new_var[b])
            for a, b in old_diff
            if old_to_new_var[a] >= 0 and old_to_new_var[b] >= 0
        }


class TestTrivialTearing:
    def test_explicit_chain_torn_to_observed(self):
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        dx = registry.derivative(x)
        y_rhs = 2 * x
        z_rhs = y + 1
        state = StructuralState(
            [
                Equation(dx, -k * x),
                Equation(y, y_rhs),
                Equation(z, z_rhs),
            ],
            [x, y, z],
            registry,
            {k},
            T,
        )
        trivial_tearing(state)
        assert set(state.fullvars) == {x, dx}
        assert [eq.lhs for eq in state.eqs] == [dx]
        assert len(state.additional_observed) == 2
        assert Equation(y, y_rhs) in state.additional_observed
        assert Equation(z, z_rhs) in state.additional_observed
        assert state.structure.graph.nsrcs() == len(state.eqs)
        assert state.structure.graph.ndsts() == len(state.fullvars)

    def test_irreducible_and_self_referencing_not_torn(self):
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                Equation(dx, -k * x),
                Equation(y, 2 * x),
                Equation(z, z * x + 1),
            ],
            [x, y, z],
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
            [Equation(registry.derivative(x), -x)],
            [x],
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
            [Equation(dx, -k * x)], [x], registry, {k}, T
        )
        result = structural_simplify(state)
        assert result.states == [x]
        assert sp.simplify(to_sympy(result.dxdt[x] + k * x)) == 0
        assert result.residuals == []
        assert result.mass_matrix is None

    def test_observed_chain_extracted(self):
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                Equation(dx, -k * x + z),
                Equation(y, 2 * x),
                Equation(z, y + 1),
            ],
            [x, y, z],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        assert result.states == [x]
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
                Equation(dx, -k * y),
                Equation(ir.ZERO, x - y),
            ],
            [x, y],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        assert result.states == [x]
        obs = dict(result.observed)
        assert obs[y] is x
        assert sp.simplify(to_sympy(result.dxdt[x] + k * x)) == 0

    def test_negated_alias_sign(self):
        x, y, k = syms("x y k")
        registry = DerivativeRegistry({"x", "y", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                Equation(dx, -k * y),
                Equation(ir.ZERO, x + y),
            ],
            [x, y],
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
                Equation(dx, -3 * x),
                Equation(dy, -3 * y),
                Equation(ir.ZERO, x - y),
            ],
            [x, y],
            registry,
            set(),
            T,
            state_priorities={y: 5},
        )
        result = structural_simplify(state)
        assert result.states == [y]
        assert dict(result.observed)[x] is y
        assert sp.simplify(to_sympy(result.dxdt[y] + 3 * y)) == 0

    def test_irreducible_not_eliminated(self):
        x, y, k = syms("x y k")
        registry = DerivativeRegistry({"x", "y", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                Equation(dx, -k * y),
                Equation(y, x),
            ],
            [x, y],
            registry,
            {k},
            T,
            irreducibles=[y],
        )
        result = structural_simplify(state)
        # y must survive as a solver unknown (algebraic state).
        assert y in result.states


class TestAlgebraicSystems:
    def test_nonlinear_algebraic_solvable_becomes_observed(self):
        x, k = syms("x k")
        z = ir.sym("z")
        registry = DerivativeRegistry({"x", "z", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                Equation(dx, -k * x + z),
                Equation(ir.ZERO, z - x**2),
            ],
            [x, z],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        assert result.states == [x]
        obs = dict(result.observed)
        assert sp.simplify(to_sympy(obs[z] - x**2)) == 0

    def test_unsolvable_algebraic_torn_to_residual(self):
        x, z = syms("x z")
        registry = DerivativeRegistry({"x", "z", "t"})
        dx = registry.derivative(x)
        # z**5 + z - x = 0 is not explicitly solvable for z.
        state = StructuralState(
            [
                Equation(dx, -z),
                Equation(ir.ZERO, z**5 + z - x),
            ],
            [x, z],
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
        mass = result.mass_matrix
        assert mass is not None
        assert mass[0][0] == 1 and mass[1][1] == 0


class TestAliasEdgeCases:
    def test_conflicting_aliases_force_zero(self):
        # x = y and x = -y can only hold together at zero: the
        # conflict group pins both variables to 0.
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        dz = registry.derivative(z)
        state = StructuralState(
            [
                Equation(dz, -k * z + x),
                Equation(ir.ZERO, x - y),
                Equation(ir.ZERO, x + y),
            ],
            [x, y, z],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        obs = dict(result.observed)
        assert obs[x] is ir.ZERO
        assert obs[y] is ir.ZERO
        assert result.states == [z]
        assert sp.simplify(to_sympy(result.dxdt[z] + k * z)) == 0

    def test_sign_chain_three_variables(self):
        # x = -y, y = -z: signs compose along the chain.
        x, y, z, k = syms("x y z k")
        registry = DerivativeRegistry({"x", "y", "z", "k", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [
                Equation(dx, -k * x),
                Equation(ir.ZERO, x + y),
                Equation(ir.ZERO, y + z),
            ],
            [x, y, z],
            registry,
            {k},
            T,
        )
        result = structural_simplify(state)
        assert result.states == [x]
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
                Equation(d2, -x),
                Equation(ir.ZERO, y + x),
            ],
            [x, y],
            registry,
            set(),
            T,
        )
        result = structural_simplify(state)
        assert len(result.differential_states) == 2
        obs = dict(result.observed)
        assert obs[y] is -x

    def test_priority_tie_warns(self):
        x, y, z = syms("x y z")
        registry = DerivativeRegistry({"x", "y", "z", "t"})
        dz = registry.derivative(z)
        state = StructuralState(
            [
                Equation(dz, x + y),
                Equation(ir.ZERO, x - y),
                Equation(ir.ZERO, x + y - z),
            ],
            [x, y, z],
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
                Equation(dz, w),
                Equation(ir.ZERO, x + y + w),
                Equation(ir.ZERO, 2 * x + 2 * y - w),
                Equation(ir.ZERO, w**5 + w - z),
            ],
            [x, y, z, w],
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
        # system tears structurally.
        x, y = syms("x y")
        result = structural_simplify(
            self._singular_state(), conservative=True
        )
        assert len(result.residuals) == len(result.algebraic_states)
        obs = dict(result.observed)
        assert x in ir.free_atoms(obs[y])


class TestPantelidesAndDummyDerivatives:
    def make_pendulum(self, priorities=None):
        x, y, vx, vy, Tn, g, L = syms("x y vx vy T g L")
        registry = DerivativeRegistry(
            {"x", "y", "vx", "vy", "T", "g", "L", "t"}
        )
        eqs = [
            Equation(registry.derivative(x), vx),
            Equation(registry.derivative(y), vy),
            Equation(registry.derivative(vx), Tn * x),
            Equation(registry.derivative(vy), Tn * y - g),
            Equation(ir.ZERO, x**2 + y**2 - L**2),
        ]
        state = StructuralState(
            eqs,
            [x, y, vx, vy, Tn],
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
        state, symbols = self.make_pendulum()
        y, vy = symbols[1], symbols[3]
        state.structure.state_priorities = [
            10 if state.fullvars[i] in (y, vy) else 0
            for i in range(len(state.fullvars))
        ]
        result = structural_simplify(state)
        assert y in result.differential_states
        assert vy in result.differential_states

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

    def test_higher_order_input_lowered(self):
        x, w = syms("x w")
        registry = DerivativeRegistry({"x", "w", "t"})
        d1 = registry.derivative(x)
        d2 = registry.derivative(d1)
        # x'' = -x (harmonic oscillator given as second order).
        state = StructuralState(
            [Equation(d2, -x)],
            [x],
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
                Equation(dx, -k * x),
                Equation(ir.ZERO, x - 1),
                Equation(ir.ZERO, x - 2),
            ],
            [x],
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
            [Equation(dx, -x + z)],
            [x, z],
            registry,
            set(),
            T,
        )
        with pytest.raises(ExtraVariablesSystemError):
            structural_simplify(state)

    def test_underdetermined_tearing_only_mode(self):
        x, z = syms("x z")
        registry = DerivativeRegistry({"x", "z", "t"})
        dx = registry.derivative(x)
        state = StructuralState(
            [Equation(dx, -x + z)],
            [x, z],
            registry,
            set(),
            T,
        )
        with pytest.warns(UserWarning):
            result = structural_simplify(
                state, fully_determined=False
            )
        assert x in result.states


def _mentions_internal_derivative(result):
    """Whether any output expression reads a registry-internal symbol."""

    expressions = list(result.dxdt.values()) + list(result.residuals)
    expressions.extend(expr for _, expr in result.observed)
    return any(
        atom.name.startswith("_cubie_D")
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
            Equation(-c * dx + c * dy, -x + z),
            Equation(c * dx - c * dy, ir.call("exp", y) - x**3 - 1),
            Equation(dz, -z + x),
        ]
        return registry, eqs, (x, y, z)

    def test_numeric_block_rewritten_to_constraint(self):
        registry, eqs, (x, y, z) = self.make_pair(ir.num(1e-6))
        state = StructuralState(eqs, [x, y, z], registry, set(), T)
        rewritten = eliminate_singular_derivative_blocks(state)
        assert len(rewritten) == 1
        constraint = state.eqs[rewritten[0]]
        assert ir.is_zero(constraint.lhs)
        # The rows sum to the derivative-free node constraint.
        expected = (-x + z) + (ir.call("exp", y) - x**3 - 1)
        assert sp.simplify(to_sympy(constraint.rhs - expected)) == 0
        incidence = state.structure.graph.s_neighbors(rewritten[0])
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
        state = StructuralState(eqs, [x, y, z], registry, knowns, T)
        result = structural_simplify(state)
        # Constraint and its derivative, both reading algebraic states.
        assert len(result.residuals) == 2
        assert len(result.algebraic_states) == 2
        assert z in result.differential_states
        read = frozenset().union(
            *(ir.free_atoms(r) for r in result.residuals)
        )
        assert set(result.algebraic_states) <= read
        assert not _mentions_internal_derivative(result)
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
            Equation(a * dx + b * dy, -x),
            Equation(b * dy + a * dz, -y),
            Equation(a * dx + 2 * b * dy + a * dz, -z),
        ]
        state = StructuralState(eqs, [x, y, z], registry, {a, b}, T)
        assert eliminate_singular_derivative_blocks(state) == [2]
        assert sp.simplify(
            to_sympy(state.eqs[2].rhs) - to_sympy(x + y - z)
        ) == 0 or sp.simplify(
            to_sympy(state.eqs[2].rhs) + to_sympy(x + y - z)
        ) == 0

    @pytest.mark.parametrize("allow_parameter", [True, False])
    def test_parameter_pivot_never_divides(self, allow_parameter):
        x, y, p = syms("x y p")
        registry = DerivativeRegistry({"x", "y", "p", "t"})
        dx = registry.derivative(x)
        eqs = [Equation(p * dx, -x), Equation(dx, -y)]
        state = StructuralState(eqs, [x, y], registry, {p}, T)
        assert eliminate_singular_derivative_blocks(
            state, allow_parameter=allow_parameter
        ) == [0]
        constraint = state.eqs[0]
        assert ir.is_zero(constraint.lhs)
        expected = x - p * y
        assert (
            sp.simplify(to_sympy(constraint.rhs - expected)) == 0
            or sp.simplify(to_sympy(constraint.rhs + expected)) == 0
        )
        assert state.eqs[1] == eqs[1]

    def test_rejected_parameter_pivot_leaves_rows(self):
        x, y, p, q = syms("x y p q")
        registry = DerivativeRegistry({"x", "y", "p", "q", "t"})
        dx = registry.derivative(x)
        eqs = [Equation(p * dx, -x), Equation(q * dx, -y)]
        state = StructuralState(eqs, [x, y], registry, {p, q}, T)
        assert eliminate_singular_derivative_blocks(
            state, allow_parameter=False
        ) == []
        assert state.eqs == eqs

    def test_independent_block_untouched(self):
        x, y = syms("x y")
        registry = DerivativeRegistry({"x", "y", "t"})
        dx, dy = registry.derivative(x), registry.derivative(y)
        eqs = [
            Equation(2 * dx + 3 * dy, -x),
            Equation(3 * dy - 2 * dx, -y),
        ]
        state = StructuralState(eqs, [x, y], registry, set(), T)
        before = list(state.eqs)
        assert eliminate_singular_derivative_blocks(state) == []
        assert state.eqs == before

    def test_unknown_coefficient_excluded(self):
        x, y, w = syms("x y w")
        registry = DerivativeRegistry({"x", "y", "w", "t"})
        dx, dy, dw = (registry.derivative(s) for s in (x, y, w))
        eqs = [
            Equation(w * dx - w * dy, -x),
            Equation(-w * dx + w * dy, -y + 1),
            Equation(dw, -w),
        ]
        state = StructuralState(eqs, [x, y, w], registry, set(), T)
        assert eliminate_singular_derivative_blocks(state) == []


class TestExactLinearSCCRewrite:
    def test_integer_constraint_block_solves_explicitly(self):
        # The integer-linear SCC solves explicitly; x and x_t observed.
        x, y, z = syms("x y z")
        registry = DerivativeRegistry({"x", "y", "z", "t"})
        dx, dy, dz = (registry.derivative(s) for s in (x, y, z))
        eqs = [
            Equation(dx + dy, -x),
            Equation(dy + dz, -y),
            Equation(ir.ZERO, x + y - z),
        ]
        state = StructuralState(eqs, [x, y, z], registry, set(), T)
        result = structural_simplify(state)
        assert result.residuals == []
        assert set(result.states) == {y, z}
        assert not _mentions_internal_derivative(result)
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
