<!-- Parent: ../AGENTS.md -->

# structural

## Purpose
Structural simplification and tearing for DAE systems, following the continuous-system
`mtkcompile` pipeline of ModelingToolkit.jl. Takes a general DAE (implicit equations,
higher-order derivatives, algebraic unknowns), runs singular derivative-block removal,
perfect-alias elimination, trivial tearing, exact integer-linear singularity removal,
Pantelides index reduction, dummy-derivative state selection, exact matching of
integer-linear SCCs, and Modia tearing, and reassembles an explicit ODE — or, when
algebraic loops cannot be torn symbolically, a semi-explicit index-1 system with residual
rows paired with algebraic states.
Entry point: `structural_simplify(StructuralState) -> SimplifiedSystem`; the parsing front
end (`parsing/normalise.py` + `parsing/assemble.py`) builds the state and consumes the
result.

## Key Files
| File | Description |
|------|-------------|
| `simplify.py` | Pipeline driver `structural_simplify` (the `mtkcompile!` equivalent) and the residual/algebraic-state pairing check of unbalanced results. |
| `system_structure.py` | `StructuralState` (MTK's `TearingState` and `SystemStructure` in one): equations as `(lhs, rhs)` IR pairs, `fullvars`, incidence and solvable graphs, derivative chains read from the registry (`derivative_of`, `primal_of`, `is_algebraic`), solvability analysis via linear expansion, integer-linear subsystem matrix, `add_variable`/`add_equation`/`rewrite_from_row`, symbolic equation/variable differentiation, removal/reindexing (`rm_eqs_vars`), ranks and priorities. |
| `bipartite.py` | `BipartiteGraph` (sorted forward and backward adjacency, equations x variables), `Matching` with inverse view, augmenting-path `maximal_matching`, `UNASSIGNED`/`SELECTED_STATE` sentinels. |
| `digraph.py` | Matching-oriented directed views (`DiCMOBiGraphT`/`F`), iterative Tarjan SCC, `find_var_sccs` (BLT ordering), `toposort_equations`, and the `IncrementalCycleTracker` used to keep tearing assignments acyclic. |
| `diffgraph.py` | `DiffGraph`: equation differentiation chains with inverse view. |
| `clil.py` | `SparseMatrixCLIL` integer matrix and the one fraction-free Bareiss elimination (`bareiss`, pivoting through a list of column masks; CLIL-specialised update; `find_masked_pivot`). |
| `symbolics.py` | Engine-IR primitives: structural `linear_expansion`, `fixpoint_sub` (with a cycle guard), `total_derivative`, `linear_dependencies` (fraction-free elimination with numeric-first pivots gated by a caller predicate; returns the rows the pivot rows span, with multipliers), and `DerivativeRegistry` (plain-symbol stand-in for MTK `Differential` terms and the record of every derivative chain; keys are interned `ir.Sym` nodes). |
| `derivative_block.py` | `eliminate_singular_derivative_blocks`: replaces each equation whose derivative terms are an exact combination of the pivot equations' derivative terms with the derivative-free equation that combination implies; symbolic pivots follow the `allow_symbolic`/`allow_parameter` division policy, and equations with an unknown in a derivative coefficient are left alone. |
| `alias_elimination.py` | Perfect-alias elimination (sign-tracking union-find, conflict groups force zeros), trivial tearing, and the integer-linear `alias_elimination` driver. |
| `singularity_removal.py` | `structural_singularity_removal` over the integer-linear subsystem and `get_new_mm`. |
| `pantelides.py` | Pantelides index reduction and `computed_highest_diff_variables`. |
| `dummy_derivatives.py` | Dummy-derivative state selection (`dummy_derivative_graph`, integer-Jacobian rank via Bareiss elimination one column at a time, with structural-rank fallback). |
| `exact_matching.py` | `match_linear_sccs`: before Modia tearing, reduces each square SCC of homogeneous integer-linear equations with Bareiss elimination over its own variables (derivative pivots first) into equations that solve their variables explicitly in sequence, rewriting the equations, `mm` rows, incidence and solvable graphs; singular SCCs warn and are torn. |
| `tearing.py` | `ModiaTearing` (per-SCC greedy tearing of an incidence graph along its solvable graph, kept acyclic by the incremental cycle tracker, then a final tear of free equations on overdetermined systems), `TearingResult` (carrying the free equations), `free_equations`. |
| `reassemble.py` | `default_reassemble` -> `SimplifiedSystem`: cuts dummy derivatives from their base, first-order lowering (an unsolved derivative `x_t` becomes a state and `x` takes a new derivative solved by `0 ~ D(x) - x_t`), per-SCC equation generation (differential/observed/residual) with BLT sorting, and the unknowns read off the unsolved equations. |
| `consistency.py` | Balance and structural-singularity checks with best-effort offender reporting. |
| `errors.py` | `InvalidSystemError`, `ExtraVariablesSystemError`, `ExtraEquationsSystemError`, and `raise_unmatched`, the one reporter of unmatched equations and variables. |

## Sources
Ported code is translated to Python with 0-based indices; each module docstring names the
source (package, commit, file, function) of every ported part; code written for cubie in place
of a ported design is marked "Not ported".

| Source | Licence | Used in |
|--------|---------|---------|
| ModelingToolkit.jl c4177c335 | MIT | `tearing.py`, `reassemble.py`, `dummy_derivatives.py`, `consistency.py`, `system_structure.py`, `alias_elimination.py` (`trivial_tearing`), `singularity_removal.py` (`get_new_mm`), `digraph.py` (`find_var_sccs`, `toposort_equations`), `pantelides.py`, `bipartite.py` (`SelectedState`) |
| ModelingToolkit.jl a2b6dc56 | MIT | `simplify.py` (pipeline order, `_pantelides_reassemble_state`, `_integer_jacobian`), `alias_elimination.py` (perfect-alias and integer-linear alias elimination), `system_structure.py` (`always_present`) |
| StateSelection.jl 74df007e | MIT | `clil.py`, `diffgraph.py`, `pantelides.py`, `singularity_removal.py`, `consistency.py`, `errors.py` |
| BipartiteGraphs.jl 647b6a42 (v0.1.14) | MIT | `bipartite.py`, `digraph.py` (`DiCMOBiGraphT`/`F`) |
| Graphs.jl dffc7a64 (v1.15.0) | BSD-2-Clause | `digraph.py` (`tarjan_scc`, `IncrementalCycleTracker`) |
| Modia.jl, via ModelingToolkit.jl c4177c335 | MIT | `tearing.py` (Modia tearing) |

Licence texts are in the repository's `THIRD_PARTY_LICENSES`.

## Conventions
- Equations are `(lhs, rhs)` engine IR pairs; use `ir.sub`, `ir.free_atoms` and
  `ir.xreplace` on the sides.
- Derivative terms are plain registered symbols (`DerivativeRegistry`), not
  `Differential` wrappers, named `x_t`, `x_tt`, ... from the moment they are minted
  (an underscore is appended while the name is taken; the parser reserves every name in
  the input). `dX` stays the name of the dxdt output of state `X`.
- The registry is the one record of derivative chains: `StructuralState.derivative_of`
  and `primal_of` look a variable's chain up through it and `var2idx`. Only
  `eq_to_diff` is a `DiffGraph`. `registry.cut` makes a dummy derivative an ordinary
  variable; `registry.move_derivative` rebases a chain when lowering finds an existing
  `0 ~ x_t - y`.
- Per-variable lists (`fullvars`, `state_priorities`, `canonical_ranks`,
  `always_present`) grow only through `add_variable` and shrink only through
  `rm_eqs_vars`; equations grow only through `add_equation`.
- `find_eq_solvables` recomputes an equation's solvable edges from scratch.
- `StructuralState(derivative_names=...)` carries the user-function derivative helper
  names; equation differentiation and the integer Jacobian pass them to `ir.diff`.
- `bareiss` returns the pivot columns in elimination order; their count is the rank.
- Discrete systems, state machines, hierarchical connections and SDE tearing are not
  supported.

## Determinism
`StructuralState.fullvars` lists the occurring derivative symbols by
base name and derivative order, then the rest of their chains down to the base unknowns
by base name and descending order, then the remaining unknowns by base name; ties
between equal candidates go to the lower index. Equations are ordered by their printed
form `f"{lhs} ~ {rhs}"`.
`find_var_sccs` (BLT order of the condensation) and `toposort_equations` (evaluation
order of an SCC's solved equations) both use the engine's depth-first `dfs_order`:
roots (nodes nothing depends on) are visited in index order for SCCs and in the given
equation order within an SCC, and dependencies are followed in ascending index order,
each emitted before its dependents. Orders are fixed by indices and list order, never
by hash order. Never iterate a Python `set` where it feeds an order.

## Mutation
Every pass mutates `StructuralState` in place. `Matching.__setitem__` maintains the
inverse (assigning an equation unassigns its previous variable, which
`generate_derivative_variables` relies on), `BipartiteGraph` inverse views alias storage,
and `rm_eqs_vars` renumbers everything: use its `old_to_new` maps and rebuild `mm` via
`get_new_mm`.

## Output contract
`SimplifiedSystem.differential_states` and `algebraic_states` are in BLT order; `dxdt`
maps differential states to explicit RHS; `residuals[i]` pairs with
`algebraic_states[i]`, the torn variable reached from the residual's equation by
alternating pre-tearing and tearing matches (`torn_partner`), so each residual row
depends on its own state. The mass matrix is built by `parsing/assemble.py` from the
differential states when residuals exist. Observed assignments are topologically sorted.
Balanced inputs always pair residuals with algebraic states. `fully_determined=False`
results must match residuals to algebraic states one to one through the states each
residual reads (directly or via observed assignments); otherwise
`ExtraEquationsSystemError`, `ExtraVariablesSystemError` or, when both are left over,
`InvalidSystemError` names the unpaired residuals and states.

## Dependencies
### Internal
- `cubie.odesystems.symbolic.engine` (all expression algebra: IR nodes, `xreplace`,
  `diff`, `free_atoms`, and the `topological_sort` used for observed sorting). The
  parsing front end (`parsing/normalise.py`, `parsing/assemble.py`) consumes this package;
  nothing here imports upward, and nothing here imports SymPy — expressions arrive as
  engine IR from the parse boundary.
### External
- Stdlib `bisect`, `heapq`, `warnings`.
