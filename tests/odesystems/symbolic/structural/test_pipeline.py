"""Structural simplification pipeline tests."""

import pytest
import sympy as sp

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import to_sympy
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

    def test_inline_linear_scc_solves_analytically(self):
        x, u, v, p = syms("x u v p")
        registry = DerivativeRegistry({"x", "u", "v", "p", "t"})
        dx = registry.derivative(x)
        # Coupled linear block: u + 2v = x; 3u - v = 1 with
        # non-integer-friendly structure kept linear.

        def state_fn():
            return StructuralState(
                [
                    Equation(dx, -u),
                    Equation(ir.ZERO, u + 2 * v - x),
                    Equation(ir.ZERO, 3 * u - v - 1),
                ],
                [x, u, v],
                registry,
                {p},
                T,
            )

        result = structural_simplify(
            state_fn(), inline_linear_sccs=True
        )
        # The integer-linear exact SCC matching or the analytic
        # solve must fully determine u and v as observed.
        assert result.states == [x]
        obs = dict(result.observed)
        u_val = obs[u]
        for _ in range(3):
            u_val = ir.xreplace(u_val, obs)
        expected = (x + 2) / 7
        assert sp.simplify(to_sympy(u_val - expected)) == 0


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

    def test_exact_scc_matching_singular_warns(self):
        # Unit-level pin of the rank-deficient fallback: an SCC of
        # integer-linear rows that is exactly singular over its own
        # variables warns and reports no exact matching.
        from cubie.odesystems.symbolic.structural.bipartite import (
            Matching,
        )
        from cubie.odesystems.symbolic.structural.tearing import (
            exact_scc_matching,
        )

        x, y = syms("x y")
        registry = DerivativeRegistry({"x", "y", "t"})
        state = StructuralState(
            [
                Equation(ir.ZERO, x + y),
                Equation(ir.ZERO, 2 * x + 2 * y),
            ],
            [x, y],
            registry,
            set(),
            T,
        )
        mm = state.linear_subsys_adjmat()
        mm_row_of = {e: i for i, e in enumerate(mm.nzrows)}
        matching = Matching(2).complete(2)
        with pytest.warns(UserWarning, match="exactly singular"):
            exact = exact_scc_matching(
                state.structure,
                mm,
                mm_row_of,
                matching,
                [0, 1],
                [0, 1],
                None,
                [],
            )
        assert not exact


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
