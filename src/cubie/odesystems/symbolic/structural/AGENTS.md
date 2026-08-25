<!-- Parent: ../AGENTS.md -->

# structural

## Purpose
MTK-style structural simplification and tearing for DAE systems: a Python port of the
continuous-system `mtkcompile` pipeline from ModelingToolkit.jl v11 and its factored
algorithm packages (BipartiteGraphs.jl, StateSelection.jl, ModelingToolkitTearing). Takes a
general DAE (implicit equations, higher-order derivatives, algebraic unknowns), runs
singular derivative-block removal, perfect-alias elimination, trivial tearing, exact
integer-linear singularity removal,
Pantelides index reduction, dummy-derivative state selection, and Carpanzano/Modia tearing,
and reassembles an explicit ODE — or, when algebraic loops cannot be torn symbolically, a
semi-explicit index-1 system with residual rows under a singular diagonal mass matrix.
Entry point: `structural_simplify(StructuralState) -> SimplifiedSystem`; the parsing front
end (`parsing/normalise.py` + `parsing/assemble.py`) builds the state and consumes the
result.

## Key Files
| File | Description |
|------|-------------|
| `simplify.py` | Pipeline driver `structural_simplify` (the `mtkcompile!` equivalent) and the `SimplifiedSystem` result (states, `dxdt`, residuals, observed, mass matrix, BLT blocks). |
| `system_structure.py` | `StructuralState`/`SystemStructure` (the `TearingState` equivalent): incidence graph construction, solvability analysis via linear expansion, integer-linear subsystem matrix, symbolic equation/variable differentiation, removal/reindexing, deterministic ranks and priorities. |
| `bipartite.py` | `BipartiteGraph` (sorted adjacency, equations x variables), `Matching` with inverse view, augmenting-path `maximal_matching`, `UNASSIGNED`/`SELECTED_STATE` sentinels. |
| `digraph.py` | Matching-oriented directed views (`DiCMOBiGraphT`/`F`), iterative Tarjan SCC, `find_var_sccs` (BLT ordering), BFS `neighborhood_in`, and the BFGT Algorithm-N `IncrementalCycleTracker` used to keep tearing assignments acyclic. |
| `diffgraph.py` | `DiffGraph`: variable/equation differentiation chains with inverse view. |
| `clil.py` | `SparseMatrixCLIL` integer matrix and fraction-free Bareiss elimination (`bareiss`, CLIL-specialised update, `nullspace_rank`). |
| `symbolics.py` | Engine-IR primitives: structural `linear_expansion`, `solve_linear`, `fixpoint_sub`, `total_derivative`, small-int gate, `solve_linear_system` (dense symbolic Gaussian elimination for small linear SCCs), and `DerivativeRegistry` (plain-symbol stand-in for MTK `Differential` terms, `x_t` dummy naming; keys are interned `ir.Sym` nodes). |
| `derivative_block.py` | `eliminate_singular_derivative_blocks`: exact rank test of the derivative-coefficient rows (rational weights over symbolic monomials, so folded constants and parameter coefficients both cancel); a dependent row is replaced in place by the derivative-free combination it implies, making a numerically singular mass block a constraint Pantelides can see. Rows with unknowns in a derivative coefficient are left alone. |
| `alias_elimination.py` | Perfect-alias elimination (sign-tracking union-find, conflict groups force zeros), `trivial_tearing` (preemptive observed extraction), and the integer-linear `alias_elimination` driver. |
| `singularity_removal.py` | Tiered-pivot Bareiss over the integer-linear subsystem (`structural_singularity_removal`, `aag_bareiss`), per-connected-component elimination, `get_new_mm`, `RestrictedBareissContext` for exact SCC matching. |
| `pantelides.py` | Pantelides index reduction and `computed_highest_diff_variables`. |
| `dummy_derivatives.py` | Dummy-derivative state selection (`dummy_derivative_graph`, integer-Jacobian rank via Bareiss nullspace with structural-rank fallback) and level-based partial state selection. |
| `tearing.py` | `ModiaTearing` and `CarpanzanoTearing` (default; exact integer-linear SCC matching), `TearingResult`, `contract_variables`, deterministic `OrderedSet`. Exact SCC matching returns the Bareiss-reduced rows as `extra["linear_rewrite"]`; `simplify._apply_linear_rewrites` writes them into the symbolic equations before reassembly. |
| `reassemble.py` | `default_reassemble`: dummy-derivative renaming, first-order lowering (`0 ~ D(x) - x_t`), per-SCC equation generation (differential/observed/residual) with BLT sorting, analytic small-N linear SCC solves, final reordering. |
| `consistency.py` | Balance and structural-singularity checks with best-effort offender reporting. |
| `errors.py` | `InvalidSystemError`, `ExtraVariablesSystemError`, `ExtraEquationsSystemError`. |

## For AI Agents

### Port constraints
The algorithms follow ModelingToolkit.jl v11, StateSelection.jl, and
BipartiteGraphs.jl with 0-based indices:
- Derivative terms are plain registered symbols (`DerivativeRegistry`), not `Differential`
  wrappers. Internal derivative symbols are mangled (`_cubie_D<order>_<base>`) and never
  user-visible; state selection renames dummies to `x_t`-style names (`lower_varname`).
  `registry.rename` cuts the base link (MTK `diff2term` semantics).
- `diff_eq_states` follows the graph chain after `find_duplicate_dd`
  rewires `diff_to_primal`.
- Bareiss returns rank and pivot order; Python integers do not overflow.
- Discrete systems, state machines, hierarchical connections, and SDE
  tearing are not supported.

### Determinism
Results must not depend on declaration or equation order: canonical ranks
(`_canonical_sort_key`), the structural equation sort key, `OrderedSet` in tearing, and
ascending tie-breaks in SCC/toposort exist for exactly this. Never swap a Python `set`
into an iteration that feeds a tie-break.

### Mutation discipline
`StructuralState` is mutated in place by every pass, and graph/matching invariants are
coupled: `Matching.__setitem__` maintains the inverse (assigning an equation unassigns its
previous variable — behaviour `generate_derivative_variables` relies on), `BipartiteGraph`
invviews alias storage, and `rm_eqs_vars` renumbers everything (old indices are invalid
afterwards; use the returned `old_to_new` maps, and rebuild `mm` via `get_new_mm`).

### Output contract
`SimplifiedSystem.states` = differential states (BLT order) + torn algebraic states;
`dxdt` maps differential states to explicit RHS; `residuals[i]` pairs with
`algebraic_states[i]`; `mass_matrix` is `None` for fully torn systems, else the singular
diagonal as nested float lists. Observed assignments are topologically sorted. Balanced inputs always pair
residuals with algebraic states; `fully_determined=False` outputs may not.

### Testing
`tests/odesystems/symbolic/structural/`: `test_graphs.py`, `test_clil_symbolics.py`
(pure algorithms), `test_pipeline.py` (pass-level and pipeline cases incl. the index-3
pendulum), `test_dae_parser.py` (front-end integration), `test_dae_solve.py` (numerical
solve of a torn DAE against a reference). Pure-Python except the solve test.

## Dependencies
### Internal
- `cubie.odesystems.symbolic.engine` (all expression algebra: IR nodes, `xreplace`,
  `diff`, `free_atoms`, and the `topological_sort` used for observed sorting). The
  parsing front end (`parsing/normalise.py`, `parsing/assemble.py`) consumes this package;
  nothing here imports upward, and nothing here imports SymPy — expressions arrive as
  engine IR from the parse boundary.
### External
- Stdlib `bisect`, `heapq`, `warnings`, `os`, `re`, `fractions`.
