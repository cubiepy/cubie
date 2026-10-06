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
rows under a singular diagonal mass matrix.
Entry point: `structural_simplify(StructuralState) -> SimplifiedSystem`; the parsing front
end (`parsing/normalise.py` + `parsing/assemble.py`) builds the state and consumes the
result.

## Key Files
| File | Description |
|------|-------------|
| `simplify.py` | Pipeline driver `structural_simplify` (the `mtkcompile!` equivalent) and the `SimplifiedSystem` result (states, `dxdt`, residuals, observed, mass matrix). |
| `system_structure.py` | `StructuralState`/`SystemStructure` (the `TearingState` equivalent): incidence graph construction, solvability analysis via linear expansion, integer-linear subsystem matrix, symbolic equation/variable differentiation, removal/reindexing, deterministic ranks and priorities. |
| `bipartite.py` | `BipartiteGraph` (sorted adjacency, equations x variables), `Matching` with inverse view, augmenting-path `maximal_matching`, `UNASSIGNED`/`SELECTED_STATE` sentinels. |
| `digraph.py` | Matching-oriented directed views (`DiCMOBiGraphT`/`F`), iterative Tarjan SCC, `find_var_sccs` (BLT ordering), `toposort_equations`, `neighborhood_in`, and the `IncrementalCycleTracker` used to keep tearing assignments acyclic. |
| `diffgraph.py` | `DiffGraph`: variable/equation differentiation chains with inverse view. |
| `clil.py` | `SparseMatrixCLIL` integer matrix and fraction-free Bareiss elimination (`bareiss`, CLIL-specialised update, `nullspace_rank`). |
| `symbolics.py` | Engine-IR primitives: structural `linear_expansion`, `fixpoint_sub`, `total_derivative`, `linear_dependencies` (fraction-free elimination with numeric-first pivots gated by a caller predicate; returns the rows the pivot rows span, with multipliers), and `DerivativeRegistry` (plain-symbol stand-in for MTK `Differential` terms, `x_t` dummy naming; keys are interned `ir.Sym` nodes). |
| `derivative_block.py` | `eliminate_singular_derivative_blocks`: replaces each equation whose derivative terms are an exact combination of the pivot equations' derivative terms with the derivative-free equation that combination implies; symbolic pivots follow the `allow_symbolic`/`allow_parameter` division policy, and equations with an unknown in a derivative coefficient are left alone. |
| `alias_elimination.py` | Perfect-alias elimination (sign-tracking union-find, conflict groups force zeros) and the integer-linear `alias_elimination` driver. |
| `singularity_removal.py` | `structural_singularity_removal` over the integer-linear subsystem. |
| `pantelides.py` | Pantelides index reduction and `computed_highest_diff_variables`. |
| `dummy_derivatives.py` | Dummy-derivative state selection (`dummy_derivative_graph`, integer-Jacobian rank via Bareiss nullspace with structural-rank fallback). |
| `exact_matching.py` | `match_linear_sccs`: before Modia tearing, reduces each square SCC of homogeneous integer-linear equations with fraction-free Bareiss elimination over its own variables (derivative pivots first) into equations that solve their variables explicitly in sequence, rewriting the equations, `mm` rows, incidence and solvable graphs; singular SCCs warn and are torn. |
| `tearing.py` | `ModiaTearing` (per-SCC greedy tearing kept acyclic by the incremental cycle tracker, then a final tear of free equations on overdetermined systems), `TearingResult`, `free_equations`, `contract_variables`. |
| `reassemble.py` | `default_reassemble`: dummy-derivative renaming, first-order lowering (`0 ~ D(x) - x_t`), per-SCC equation generation (differential/observed/residual) with BLT sorting, final reordering. |
| `consistency.py` | Balance and structural-singularity checks with best-effort offender reporting. |
| `errors.py` | `InvalidSystemError`, `ExtraVariablesSystemError`, `ExtraEquationsSystemError`. |

## Sources
Ported code is translated to Python with 0-based indices; each module docstring names the
source (package, commit, file, function) of every ported part; code written for cubie in place
of a ported design is marked "Not ported".

| Source | Licence | Used in |
|--------|---------|---------|
| ModelingToolkit.jl c4177c335 | MIT | `tearing.py`, `reassemble.py`, `dummy_derivatives.py`, `consistency.py`, `system_structure.py`, `alias_elimination.py` (`trivial_tearing`), `singularity_removal.py` (`get_new_mm`), `digraph.py` (`find_var_sccs`, `toposort_equations`), `pantelides.py`, `bipartite.py` (`SelectedState`) |
| ModelingToolkit.jl a2b6dc56 | MIT | `simplify.py` (pipeline order, `_pantelides_reassemble_state`, `_integer_jacobian`), `alias_elimination.py` (perfect-alias and integer-linear alias elimination), `system_structure.py` (`always_present`) |
| StateSelection.jl 74df007e | MIT | `clil.py`, `diffgraph.py`, `pantelides.py`, `singularity_removal.py`, `consistency.py`, `errors.py`, `system_structure.py` (derivative-graph hooks) |
| BipartiteGraphs.jl 647b6a42 (v0.1.14) | MIT | `bipartite.py`, `digraph.py` (`DiCMOBiGraphT`/`F`), `system_structure.py` (`rm_eqs_vars`) |
| Graphs.jl dffc7a64 (v1.15.0) | BSD-2-Clause | `digraph.py` (`tarjan_scc`, `neighborhood_in`, `IncrementalCycleTracker`) |
| Modia.jl, via ModelingToolkit.jl c4177c335 | MIT | `tearing.py` (Modia tearing) |

Licence texts are in the repository's `THIRD_PARTY_LICENSES`.

## Conventions
- Derivative terms are plain registered symbols (`DerivativeRegistry`), not
  `Differential` wrappers. Internal derivative symbols are mangled
  (`_cubie_D<order>_<base>`) and never user-visible; state selection renames dummies to
  `x_t`-style names (`lower_varname`). `registry.rename` cuts the link to the base
  variable.
- `diff_eq_states` follows the graph chain after `find_duplicate_dd` rewires
  `diff_to_primal`.
- Bareiss returns rank and pivot order.
- Discrete systems, state machines, hierarchical connections and SDE tearing are not
  supported.

## Determinism
`StructuralState.fullvars` lists the occurring derivative symbols by
base name and derivative order, then the rest of their chains down to the base unknowns
by base name and descending order, then the remaining unknowns by base name; ties
between equal candidates go to the lower index.
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
`SimplifiedSystem.states` = differential states (BLT order) + torn algebraic states;
`dxdt` maps differential states to explicit RHS; `residuals[i]` pairs with
`algebraic_states[i]`, the torn variable reached from the residual's equation by
alternating pre-tearing and tearing matches (`torn_partner`), so each residual row
depends on its own state; `mass_matrix` is `None` for fully torn systems, else the singular
diagonal as nested float lists. Observed assignments are topologically sorted. Balanced
inputs always pair residuals with algebraic states; `fully_determined=False` outputs may
not.

## Dependencies
### Internal
- `cubie.odesystems.symbolic.engine` (all expression algebra: IR nodes, `xreplace`,
  `diff`, `free_atoms`, and the `topological_sort` used for observed sorting). The
  parsing front end (`parsing/normalise.py`, `parsing/assemble.py`) consumes this package;
  nothing here imports upward, and nothing here imports SymPy — expressions arrive as
  engine IR from the parse boundary.
### External
- Stdlib `bisect`, `heapq`, `warnings`.
