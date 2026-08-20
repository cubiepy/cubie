"""Symbolic evaluation helpers for CPU reference integrators."""

from typing import Dict, List, Optional, Set

import numpy as np
import sympy as sp

from cubie import SymbolicODE
from cubie.odesystems.symbolic.codegen.jacobian import generate_jacobian
from cubie.odesystems.symbolic.engine import to_sympy
from cubie.odesystems.symbolic.sym_utils import topological_sort

from .cpu_utils import Array


TIME_SYMBOL = sp.Symbol("t", real=True)


class CPUODESystem:
    """Evaluator for symbolic systems using compiled numerical functions."""

    def __init__(self, system: SymbolicODE) -> None:
        self.system = system
        self.precision = system.precision
        self.n_states = system.sizes.states
        self.n_observables = system.sizes.observables
        self._state_template = np.zeros(self.n_states, dtype=self.precision)
        self._observable_template = np.zeros(
            self.n_observables, dtype=self.precision
        )

        indexed = system.indices
        self._state_index = indexed.states.index_map
        self._parameter_index = indexed.parameters.index_map
        self._constant_index = indexed.constants.index_map
        self._driver_index = indexed.drivers.index_map
        self._observable_index = indexed.observables.index_map

        self._dx_index = indexed.dxdt.index_map
        # The parser emits engine-IR pairs; the CPU reference
        # lambdifies SymPy expressions, so convert once here.
        sympy_equations = [
            (to_sympy(lhs), to_sympy(rhs))
            for lhs, rhs in system.equations.to_equation_list()
        ]
        ordered_equations = topological_sort(sympy_equations)
        self._equations = ordered_equations
        jacobian_rows = generate_jacobian(
            system.equations,
            self._state_index,
            self._dx_index,
        )
        # The engine returns IR rows; the CPU reference lambdifies
        # SymPy expressions, so convert once here.
        self._jacobian_expr = sp.Matrix(
            [
                [to_sympy(entry) for entry in row]
                for row in jacobian_rows
            ]
        )

        self._base_symbols: Set[sp.Symbol] = set().union(
            self._state_index.keys(),
            self._parameter_index.keys(),
            self._constant_index.keys(),
            self._driver_index.keys(),
            {TIME_SYMBOL},
        )
        self._observable_symbols: Set[sp.Symbol] = set(
            self._observable_index.keys()
        )
        self._dx_symbols: Set[sp.Symbol] = set(self._dx_index.keys())

        self._compile_expressions()
        # Prepare fast-path caches (symbol slots, evaluation plans, jacobian
        # slots)
        self._prepare_fast_paths()
        self._prepare_time_derivative_terms()

    def _compile_expressions(self) -> None:
        """Compile symbolic expressions into fast numerical functions."""

        self._compiled_equations = {}
        self._equation_symbols = {}

        for lhs, rhs in self._equations:
            free_vars = list(rhs.free_symbols)
            if free_vars:
                compiled_fn = sp.lambdify(free_vars, rhs, modules=["numpy"])
            else:
                compiled_fn = self.precision(rhs)
            self._equation_symbols[lhs] = free_vars
            self._compiled_equations[lhs] = compiled_fn

        if self._jacobian_expr.shape[0] > 0:
            jacobian_entries = []
            jacobian_symbols = []
            jac_rows, jac_cols = self._jacobian_expr.shape

            for i in range(jac_rows):
                row_entries = []
                row_symbols = []
                for j in range(jac_cols):
                    expr = self._jacobian_expr[i, j]
                    expr_syms = list(expr.free_symbols)
                    if expr_syms:
                        compiled_entry = sp.lambdify(
                            expr_syms, expr, modules=["numpy"]
                        )
                    else:
                        compiled_entry = self.precision(expr)

                    row_entries.append(compiled_entry)
                    row_symbols.append(expr_syms)
                jacobian_entries.append(row_entries)
                jacobian_symbols.append(row_symbols)

            self._compiled_jacobian = jacobian_entries
            self._jacobian_symbols = jacobian_symbols
        else:
            self._compiled_jacobian = []
            self._jacobian_symbols = []

        self._observable_eval_order = self._resolve_dependencies(
            self._observable_symbols
        )
        self._dx_eval_order = self._resolve_dependencies(
            self._dx_symbols, skip=self._observable_symbols
        )

    def _prepare_fast_paths(self) -> None:
        """Precompute symbol slots, constant base buffer, and evaluation plans.

        This avoids per-call dictionary construction and expensive symbol
        lookups.
        """
        # Build a dense symbol slot mapping covering base symbols and all
        # equation LHS. Order categories by their integer index for stable,
        # deterministic layout.
        def sorted_symbols(index_map: Dict[sp.Symbol, int]) -> list[sp.Symbol]:
            return [
                sym
                for sym, _ in sorted(index_map.items(), key=lambda kv: kv[1])
            ]

        symbol_order: list[sp.Symbol] = []
        symbol_to_slot: Dict[sp.Symbol, int] = {}

        def add_symbols(symbols: list[sp.Symbol]) -> None:
            for sym in symbols:
                if sym not in symbol_to_slot:
                    symbol_to_slot[sym] = len(symbol_order)
                    symbol_order.append(sym)

        add_symbols(sorted_symbols(self._state_index))
        add_symbols(sorted_symbols(self._parameter_index))
        # Constants are immutable; include them early so their slots are fixed
        # and prefilled.
        add_symbols(sorted_symbols(self._constant_index))
        add_symbols(sorted_symbols(self._driver_index))
        # Ensure all equation LHS (observables, dx, and intermediates) have
        # slots.
        add_symbols([lhs for lhs, _ in self._equations])
        # Include time symbol at the end.
        add_symbols([TIME_SYMBOL])

        self._sym_order = symbol_order
        self._sym_slots = symbol_to_slot
        nslots = len(symbol_order)

        # Map array indices (state/param/driver/observable) -> slots for fast
        # vectorized fill
        def slots_for(index_map: Dict[sp.Symbol, int]) -> np.ndarray:
            # Build array where position is the category index and value is the
            # slot index
            arr = np.empty(len(index_map), dtype=np.int64)
            for sym, idx in index_map.items():
                arr[idx] = symbol_to_slot[sym]
            return arr

        self._state_slots = slots_for(self._state_index)
        self._param_slots = slots_for(self._parameter_index)
        self._driver_slots = slots_for(self._driver_index)
        self._observable_slots = (
            slots_for(self._observable_index)
            if self._observable_index
            else np.empty(0, dtype=np.int64)
        )
        # LHS slots for quick assignment during equation evaluation
        self._lhs_slots = {
            lhs: symbol_to_slot[lhs] for lhs, _ in self._equations
        }

        # Build a base buffer with constants prefilled; other entries are
        # zero-initialized.
        base = np.zeros(nslots, dtype=self.precision)
        const_values = self.system.constants.values_dict
        for sym in self._constant_index.keys():
            base[symbol_to_slot[sym]] = self.precision(const_values[str(sym)])
        self._base_value_buffer = base

        # Precompute evaluation plans: specialized for arity 0/1/2/Many to
        # avoid tuple handling. Plan entry: (lhs_slot:int,
        # target_idx:Optional[int], fn:callable|number, kind:int, s0:int,
        # s1:int, sN:tuple[int,...]|None)
        def build_plan(
            eval_order: list[sp.Symbol],
            target_index: Dict[sp.Symbol, int],
        ):
            plan = []
            for lhs in eval_order:
                lhs_slot = symbol_to_slot[lhs]
                t_idx = target_index.get(lhs)
                compiled_fn = self._compiled_equations[lhs]
                arg_syms = self._equation_symbols[lhs]
                if not arg_syms:
                    plan.append(
                        (lhs_slot, t_idx, compiled_fn, 0, -1, -1, None)
                    )
                elif len(arg_syms) == 1:
                    plan.append((
                        lhs_slot, t_idx, compiled_fn, 1,
                        symbol_to_slot[arg_syms[0]], -1, None,
                    ))
                elif len(arg_syms) == 2:
                    plan.append((
                        lhs_slot, t_idx, compiled_fn, 2,
                        symbol_to_slot[arg_syms[0]],
                        symbol_to_slot[arg_syms[1]], None,
                    ))
                else:
                    # store as a small Python tuple of slots to avoid creating
                    # temp arrays per-call
                    slots = tuple(symbol_to_slot[s] for s in arg_syms)
                    plan.append(
                        (lhs_slot, t_idx, compiled_fn, 3, -1, -1, slots)
                    )
            return plan

        self._obs_eval_plan = build_plan(
            self._observable_eval_order, self._observable_index
        )
        self._dx_eval_plan = build_plan(self._dx_eval_order, self._dx_index)

        # Cache jacobian eval plan mirroring _jacobian_symbols with arity
        # specialization. Jacobian entry: (kind:int, fn:callable|number,
        # s0:int, s1:int, sN:tuple[int,...]|None)
        self._jacobian_plan = []
        for row_syms, row_fns in zip(
            self._jacobian_symbols, self._compiled_jacobian
        ):
            row_plan = []
            for expr_syms, compiled_entry in zip(row_syms, row_fns):
                if not expr_syms:
                    row_plan.append((0, compiled_entry, -1, -1, None))
                elif len(expr_syms) == 1:
                    row_plan.append((
                        1, compiled_entry,
                        self._sym_slots[expr_syms[0]], -1, None,
                    ))
                elif len(expr_syms) == 2:
                    row_plan.append((
                        2, compiled_entry,
                        self._sym_slots[expr_syms[0]],
                        self._sym_slots[expr_syms[1]], None,
                    ))
                else:
                    slots = tuple(self._sym_slots[s] for s in expr_syms)
                    row_plan.append((3, compiled_entry, -1, -1, slots))
            self._jacobian_plan.append(row_plan)

        # Precompute symbol-values reconstruction layouts (with and without
        # observables)
        def sv_layout(include_observables: bool):
            syms: list[sp.Symbol] = []
            # States, Parameters, Constants, Drivers in index order for
            # determinism
            syms.extend(sorted_symbols(self._state_index))
            syms.extend(sorted_symbols(self._parameter_index))
            syms.extend(sorted_symbols(self._constant_index))
            syms.extend(sorted_symbols(self._driver_index))
            if include_observables and self._observable_index:
                syms.extend(sorted_symbols(self._observable_index))
            syms.append(TIME_SYMBOL)
            slots = [symbol_to_slot[s] for s in syms]
            return syms, slots

        self._sv_syms_with_obs, self._sv_slots_with_obs = sv_layout(True)
        self._sv_syms_no_obs, self._sv_slots_no_obs = sv_layout(False)

    def _prepare_time_derivative_terms(self) -> None:
        """Precompute ∂F/∂t and driver-partial expressions for ``dx/dt``."""

        driver_symbols = list(self._driver_index.keys())
        terms: List[
            Optional[tuple[Optional[sp.Expr], tuple[tuple[int, sp.Expr], ...]]]
        ]
        terms = [None] * self.n_states

        for lhs, rhs in self._equations:
            if lhs not in self._dx_index:
                continue
            state_idx = self._dx_index[lhs]
            time_expr = sp.diff(rhs, TIME_SYMBOL)
            time_term = None if time_expr == 0 else time_expr
            driver_terms: List[tuple[int, sp.Expr]] = []
            for driver_symbol in driver_symbols:
                partial = sp.diff(rhs, driver_symbol)
                if partial == 0:
                    continue
                driver_terms.append(
                    (self._driver_index[driver_symbol], partial)
                )
            terms[state_idx] = (time_term, tuple(driver_terms))

        self._time_derivative_terms = terms

    def _resolve_dependencies(
        self,
        targets: Set[sp.Symbol],
        *,
        skip: Optional[Set[sp.Symbol]] = None,
    ) -> list[sp.Symbol]:
        """Return ordered symbols needed to evaluate ``targets``."""

        if skip is None:
            skip = set()

        closure: Set[sp.Symbol] = set()

        def visit(symbol: sp.Symbol) -> None:
            if (
                symbol in skip
                or symbol in closure
                or symbol not in self._equation_symbols
            ):
                return

            closure.add(symbol)
            for dependency in self._equation_symbols[symbol]:
                if dependency in skip or dependency in self._base_symbols:
                    continue
                visit(dependency)

        for target in targets:
            visit(target)

        ordered = [lhs for lhs, _ in self._equations if lhs in closure]
        return ordered

    def _alloc_buffer(self) -> np.ndarray:
        """Return a fresh working buffer seeded with constants."""
        return self._base_value_buffer.copy()

    def _fill_value_buffer(
        self,
        buffer: np.ndarray,
        state: Array,
        params: Array,
        drivers: Array,
        time_scalar: float,
        observables: Optional[Array] = None,
    ) -> None:
        """Fill dynamic entries of the buffer from input arrays and time."""
        precision = self.precision
        # Vectorized assignment with dtype conversion only when needed
        if self._state_slots.size:
            state_arr = (
                state
                if isinstance(state, np.ndarray) and state.dtype == precision
                else np.asarray(state, dtype=precision)
            )
            buffer[self._state_slots] = state_arr
        if self._param_slots.size:
            params_arr = (
                params
                if isinstance(params, np.ndarray) and params.dtype == precision
                else np.asarray(params, dtype=precision)
            )
            buffer[self._param_slots] = params_arr
        if self._driver_slots.size:
            drivers_arr = (
                drivers
                if isinstance(drivers, np.ndarray)
                and drivers.dtype == precision
                else np.asarray(drivers, dtype=precision)
            )
            buffer[self._driver_slots] = drivers_arr
        if observables is not None and self._observable_slots.size:
            obs_arr = (
                observables
                if isinstance(observables, np.ndarray)
                and observables.dtype == precision
                else np.asarray(observables, dtype=precision)
            )
            buffer[self._observable_slots] = obs_arr
        # Time scalar
        buffer[self._sym_slots[TIME_SYMBOL]] = (
            time_scalar
            if isinstance(time_scalar, precision)
            else precision(time_scalar)
        )

    def _build_symbol_values_dict(
        self,
        buffer: np.ndarray,
        *,
        include_observables: bool,
    ) -> Dict[sp.Symbol, float]:
        """Reconstruct the symbol->value dict matching previous semantics.

        Includes base symbols (states, params, constants, drivers, time),
        optionally observables, and any evaluated dx symbols will be added by
        the callers after equation evaluation.
        """
        # Fast path using precomputed layouts
        if include_observables:
            syms, slots = self._sv_syms_with_obs, self._sv_slots_with_obs
        else:
            syms, slots = self._sv_syms_no_obs, self._sv_slots_no_obs
        return {s: buffer[i] for s, i in zip(syms, slots)}

    def observables(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        time_scalar: float,
    ) -> Array:
        """Evaluate the observable expressions for the current state."""

        # Prepare working buffer seeded with constants; fill dynamic inputs.
        work = self._alloc_buffer()
        self._fill_value_buffer(
            work,
            state,
            params,
            drivers,
            time_scalar,
        )

        observables = self._observable_template.copy()
        w = work  # local alias for faster access in hot loops
        # Evaluate observables and any dependencies in a precomputed order
        for lhs_slot, target_idx, fn, kind, s0, s1, sN in self._obs_eval_plan:
            if kind == 0:
                value = fn  # constant numeric
            elif kind == 1:
                value = fn(w[s0])
            elif kind == 2:
                value = fn(w[s0], w[s1])
            else:
                # kind 3: variadic via direct indexing, avoiding a temp array
                # allocation
                value = fn(*(w[idx] for idx in sN))
            w[lhs_slot] = value
            if target_idx is not None:
                observables[target_idx] = value

        return observables

    def _compute_rhs_buffer(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        observables: Array,
        time_scalar: float,
    ) -> tuple[Array, np.ndarray]:
        """Compute dxdt and return the working buffer used."""
        work = self._alloc_buffer()
        self._fill_value_buffer(
            work,
            state,
            params,
            drivers,
            time_scalar,
            observables=observables,
        )
        dxdt = self._state_template.copy()
        w = work  # local alias for faster access in hot loops
        for lhs_slot, target_idx, fn, kind, s0, s1, sN in self._dx_eval_plan:
            if kind == 0:
                value = fn
            elif kind == 1:
                value = fn(w[s0])
            elif kind == 2:
                value = fn(w[s0], w[s1])
            else:
                value = fn(*(w[idx] for idx in sN))
            w[lhs_slot] = value
            if target_idx is not None:
                dxdt[target_idx] = value
        return dxdt, work

    def rhs(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        observables: Array,
        time_scalar: float,
    ) -> tuple[Array, Dict[sp.Symbol, float]]:
        """Evaluate ``dx/dt`` using pre-computed observables."""

        dxdt, work = self._compute_rhs_buffer(
            state,
            params,
            drivers,
            observables,
            time_scalar,
        )

        # Rebuild symbol_values dict to preserve previous external behavior
        symbol_values = self._build_symbol_values_dict(
            work, include_observables=True
        )
        # Add evaluated dx/intermediate symbols encountered in the dx plan
        for lhs_slot, _, _, _, _, _, _ in self._dx_eval_plan:
            sym = self._sym_order[lhs_slot]
            symbol_values[sym] = work[lhs_slot]

        return dxdt, symbol_values

    def time_derivative(
        self,
        symbol_values: Dict[sp.Symbol, float],
        driver_dt: Array,
    ) -> Array:
        """Evaluate the time-derivative contribution of the RHS."""

        result = np.zeros(self.n_states, dtype=self.precision)
        if not getattr(self, "_time_derivative_terms", None):
            return result

        driver_rates = (
            np.asarray(driver_dt, dtype=self.precision)
            if len(driver_dt) > 0
            else np.zeros(0, dtype=self.precision)
        )
        zero = self.precision(0.0)
        for idx, terms in enumerate(self._time_derivative_terms):
            if terms is None:
                result[idx] = zero
                continue
            time_term, driver_terms = terms
            value = zero
            if time_term is not None:
                value = self.precision(time_term.subs(symbol_values))
            for driver_index, expr in driver_terms:
                if driver_index >= driver_rates.shape[0]:
                    continue
                partial_value = self.precision(expr.subs(symbol_values))
                value = value + partial_value * driver_rates[driver_index]
            result[idx] = value
        return result

    def jacobian(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        observables: Array,
        time_scalar: float,
    ) -> Array:
        if not self._compiled_jacobian:
            return np.zeros(
                (self.n_states, self.n_states), dtype=self.precision
            )

        # Reuse the fast buffer path and avoid building a symbol dict
        _, work = self._compute_rhs_buffer(
            state,
            params,
            drivers,
            observables,
            time_scalar,
        )
        jac_rows = len(self._compiled_jacobian)
        jac_cols = len(self._compiled_jacobian[0]) if jac_rows > 0 else 0
        jacobian = np.zeros((jac_rows, jac_cols), dtype=self.precision)

        w = work  # local alias for speed
        for i, row_plan in enumerate(self._jacobian_plan):
            for j, (kind, fn, s0, s1, sN) in enumerate(row_plan):
                if kind == 0:
                    jacobian[i, j] = fn
                elif kind == 1:
                    jacobian[i, j] = fn(w[s0])
                elif kind == 2:
                    jacobian[i, j] = fn(w[s0], w[s1])
                else:
                    jacobian[i, j] = fn(*(w[idx] for idx in sN))
        return jacobian


__all__ = ["CPUODESystem", "TIME_SYMBOL"]
