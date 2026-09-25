<!-- Parent: ../AGENTS.md -->

# codegen

## Purpose
IR-to-CUDA **source generators**. Each public `generate_*` function takes parsed equations
(`ParsedEquations`/`JVPEquations`) plus an index map (`IndexedBases`), converts them once to
the `engine/` IR through `engine.adapter.system_ir`, and returns a
**Python source string** defining a `*_factory(precision, ...)` function. Nothing here
compiles CUDA: `symbolicODE.get_solver_helper()` writes the string to disk via `ODEFile`, imports
it, and calls the factory to JIT a Numba CUDA device function. This module is the source of the
explicit `dxdt`/observables RHS, the analytic Jacobian and Jacobian-vector products (JVP), and the
three matrix-free device callbacks the implicit solvers need:

- **Linear operator** — `out = β·M·v − γ·a_ij·h·(J·v)`
- **Nonlinear residual** — `out = β·M·u − γ·h·f(base_state + a_ij·u)`
- **Neumann preconditioner** — truncated series approximating `(β·I − γ·a_ij·h·J)⁻¹·v` (Horner form)

These three are algebraically locked together: the operator is the exact Jacobian of the residual
with respect to the stage increment `u`. The `a_ij` placement differs between them *by design*
(see the sign-convention note below) and must not be "symmetrised".

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Star-imports `linear_operators`, `nonlinear_residuals`, `preconditioners` and re-exports the engine printer (`print_cuda`, `print_cuda_multiple`, `CUDA_FUNCTIONS`). `dxdt`, `time_derivative`, `jacobian`, `_stage_utils` are imported by full path, not star-imported. |
| `_matrix_utils.py` | `mass_diagonal_flags` — normalises a mass matrix (`None` or a 0/1 diagonal) to per-row boolean flags, raising on anything else; `mass_matrix_is_identity` detects `None` or a literal identity. |
| `dxdt.py` | `generate_dxdt_fac_code` (emits `dxdt(state, parameters, drivers, observables, out, t)`), `generate_observables_fac_code` (emits `get_observables(...)`), and `generate_evaluate_inv_mass_f_code` (emits `evaluate_inv_mass_f(...)` = `M**-1 @ f`, same ABI as `dxdt`; the identity is the only invertible derived mass, so it emits the plain `dxdt` body and raises on any zero mass row). The explicit RHS used by all algorithms. |
| `time_derivative.py` | `generate_time_derivative_fac_code`: emits `time_derivative_rhs(state, parameters, drivers, driver_dt, observables, out, t)` computing ∂RHS/∂t = direct ∂t + driver chain (via `driver_dt`) + chain rule through intermediates. Used by Rosenbrock-W. |
| `jacobian.py` | Pure-symbolic (emits no CUDA): `generate_jacobian` (full analytic Jacobian via chain rule over auxiliary assignments; returns row-major lists of IR expressions) and `generate_analytical_jvp` (returns `JVPEquations` with pinned `j_ij` entry symbols and `Arr("jvp", i)` product terms). Memoised in a module-level `_cache` keyed by `get_cache_key` over interned IR nodes. |
| `linear_operators.py` | Emits the matrix-free linear operator and JVP cache helpers: `generate_linear_operator_code(..., variant)` (one entry covering the plain, cached, at-state, and flattened-FIRK forms), `generate_prepare_jac_code` (populates `cached_aux`, returns `(code, aux_count)`), `generate_apply_mass_code` (`apply_mass(v, out)` = `M @ v`). All take an optional prebuilt `JVPEquations`. |
| `preconditioners.py` | Emits the Neumann-series and diagonal-Jacobi preconditioners: `generate_neumann_preconditioner_code(..., variant)` and `generate_jacobi_preconditioner_code(..., variant)`, each covering all four variants through one template. Emitted signature ends `..., v, out, jvp`. Both run a truncated series of `order` terms (a factory binding, not a source input); Jacobi order 0 is the plain diagonal solve. |
| `lu_solver.py` | Emits the direct sparse LU solves for `LUSolver`: `generate_lu_solve_code(..., variant)` returning `(code, lu_nnz)`: Markowitz-ordered symbolic factorisation of `W = beta*M - gamma*a_ij*h*J` emitted as straight-line elimination and substitution with literal indices. `beta`/`gamma` (and `a_ij` when the step's diagonals are uniform) fold in as numeric literals; pivots are static row/column choices over structural nonzeros (mass-carrying diagonals preferred; on stacked FIRK matrices, same-stage-block entries preferred over cross-stage ones); generation raises on a structurally singular pattern and the solve always returns `int32(0)`. Factor slots hold inverse pivots, and prefactored `cached_aux` layouts are compacted (details in the module docstring). |
| `nonlinear_residuals.py` | Emits the Newton residual: `generate_residual_code(..., variant)` (single stage for SDIRK/ESDIRK, flattened FIRK under `STACKED_STAGES`). |
| `_stage_utils.py` | Shared FIRK helpers: `prepare_stage_data` (Butcher `A`/`c` → IR rows, nodes, stage count) and `build_stage_metadata` (emit `_cubie_codegen_c_<i>`, `_cubie_codegen_a_<i>_<j>` symbol assignments). Used by every `STACKED_STAGES` body builder. |

## Generator variants
Each Jacobian-facing generator takes a `HelperVariant`, the internal product of the request axes `jacobian_at`/`prefactored`/`stacked`:

- **`PLAIN`** (single-stage Newton): the `state` argument is the stage increment; the
  generator substitutes `state_sym → base_state[i] + a_ij*state[i]` inline.
- **`AT_STATE`** (error smoothing): same signature but no increment
  substitution — J is evaluated at the `state` argument, `base_state` is unused, and
  `a_ij` scales the matrix only.
- **`STACKED_STAGES`** (FIRK): one flattened system of `s·n` unknowns; stage coupling (`A⊗J`) and
  per-stage time nodes are baked in via `_stage_utils`.
- **`CACHED_STACKED`** (FIRK simplified Newton): the flattened form on a Jacobian
  frozen at the step-start state — one shared v-independent auxiliary chain
  (`cached_shared_assignments`, cached slots bound) serves every stage, and only
  v-dependent assignments are stage-instantiated
  (`build_stage_cached_jvp_assignments`).
- **`PREFACTORED`** (`lu_solve`/`lu_prepare_blocks` only): substitution against
  step-start per-diagonal LU factors read from `cached_aux`.
- **`PREFACTORED_STACKED`** (lu family): the eigenvalue block-transform substitution against step-start block factors.
- **`CACHED` / `prepare_jac`** (Rosenbrock-W): `state` is the actual state (no substitution);
  selected auxiliaries are precomputed once per step into `cached_aux` by `prepare_jac` and read
  back by the operator/preconditioner. `GenericRosenbrockWStep` requests the cached
  operator and preconditioner — each `HelperResult` carries the `prepare_jac`
  companion and `cached_auxiliary_count` — and runs
  `prepare_jacobian` once per step. *Which* auxiliaries get cached is chosen by the planner
  (`parsing/auxiliary_caching.plan_auxiliary_cache`): every v-independent assignment in the
  JVP graph (named auxiliaries, Jacobian entries, `_cse` locals) is a candidate. Selection is
  a maximum-weight closure solved by min-cut over device-weighted costs
  (`engine.count_device_ops`); each cached slot charges `JVPEquations.read_price` (default 8)
  per operator call, capped at `cache_slot_limit` slots (default `2*len(jvp_terms)`,
  overridable via `max_cached_terms`). Consumers of a cached value stay in the runtime body
  and read the buffer slot; removed nodes whose consumers are all removed move into the
  prepare fill uncached.

**Consumers:** dispatch and identities are owned by `../AGENTS.md`
(get_solver_helper section) and `../helper_registry.py`. Treat the
device-function signatures in each template docstring as the contract;
factory-binding signatures are declared in the registry.

## Generators emit strings
Every `generate_*` builds Python source from a module-level `*_TEMPLATE`; wiring one in
takes a `SolverHelperRole` subclass in `../helper_registry.py`. Templates are
indentation-sensitive: bodies from `print_cuda_multiple(...)` are joined with explicit
leading spaces (8 inside a factory body, 12 inside the preconditioner's
`for _ in unroll_if(...)` loop). Emitted loops take their `unroll_if` flag from the
factory's `unroll_solver_element`/`unroll_other_small` arguments.

## Sign and coefficient convention
Operator `β·M·v − γ·a_ij·h·(J·v)` (explicit `a_ij`); residual `β·M·u − γ·h·f(base +
a_ij·u)` (`a_ij` only inside the evaluation point); the preconditioner approximates
`(β·I − γ·a_ij·h·J)⁻¹` with `h_eff = (γ·a_ij/β)·h`. The operator is `∂residual/∂u`, which
is where the operator's explicit `a_ij` comes from; change all three forms together.
Non-cached paths substitute `state → base_state + a_ij*state`; cached paths read
`cached_aux` with no substitution (`_build_operator_body`'s `use_cached_aux`).

## The `_cubie_codegen_` namespace
User constants fold into the equations as literals and bind no name (the LU family also
folds `operator_beta`/`operator_gamma`, keyed through `folded_args`). Every name the
generators bind lives under `_cubie_codegen_`: scalings (`_cubie_codegen_beta`,
`_cubie_codegen_gamma`), scalar device arguments (`_cubie_codegen_h`,
`_cubie_codegen_a_ij`), factory locals (`_cubie_codegen_n`, `_cubie_codegen_order`,
`_cubie_codegen_total_n`, ...), tableau metadata (`_cubie_codegen_c_<i>`,
`_cubie_codegen_a_<i>_<j>`) and IR locals (`_cubie_codegen_dx_*`,
`_cubie_codegen_aux_*`, `_cubie_codegen_j_*`, `_cubie_codegen_diag_*`, stage renames
`_cubie_codegen_s<i>_*`). `IndexedBases.from_user_inputs` rejects user names with that
prefix. A new generator binds nothing outside the namespace except the template's
positional argument names (`t` stays bare). Factory signatures expose `precision` (plus
`order` for preconditioners).

## Mass matrix and order
`M` is `None` (identity) or the 0/1 diagonal structural simplification derives for torn
systems, consumed as per-row flags: an identity row emits the plain form, a zero row the
residual form. `order` is in every factory signature; only the preconditioner uses it
(Neumann truncation degree, default `1`).

## Helper pipeline
Parsed equations → JVP graph → cache selection → per-helper emission.
`generate_analytical_jvp` is the expensive step; operator, preconditioner and cached-JVP
generators take a prebuilt `JVPEquations` via `jvp_equations=`. Cached bodies read the
graph's views (`cached_runtime_assignments()`, `jacobian_entry(i, j)`,
`prepare_fill_assignments()`, `cached_slot_order`); stacked consumers reach them through
`build_stage_jvp_assignments` renames. Only cached variants may depend on the cache
selection.

## Printing
The printer is `engine/printer.py`: numeric literals wrapped in `precision(...)` (array
indices stay plain integers), integer powers as multiplication chains up to
`_POW_CHAIN_LIMIT`, `CUDA_FUNCTIONS`, user-function aliases, Piecewise as branchless
`selp`, and scalar-to-array remapping through a name-keyed symbol map (generators pass
`sysir.arrayrefs`).

## Codegen hygiene
- `engine.prune_unused(..., output_name=...)` runs last in every `_build_*`, dropping
  intermediates that don't feed the named output (`'out'`, `'jvp'`, `'cached_aux'`, …).
- Stage builders apply one combined substitution map per stage in a single
  `engine.xreplace`; `build_stage_substitutions` renames every non-dx LHS per stage.
- `cse` selects `cse_and_stack` or `topological_sort`; the Jacobian/JVP cache key
  includes the CSE flag.
- The Jacobian `_cache` is process-global and unbounded, keyed by equation tuple,
  input/output orders and CSE flag; the Jacobian and JVP share an entry.
- Each generator module registers `default_timelogger` events at import and brackets
  its work with `start_event`/`stop_event`.
- Validate emitted operators against finite-difference Jacobians and the
  operator/residual/preconditioner consistency, not just that the source imports.

## Dependencies
### Internal
- `cubie.odesystems.symbolic.parsing` (`ParsedEquations`, `IndexedBases`, `JVPEquations`,
  `TIME_SYMBOL`); `cubie.time_logger`
  (`default_timelogger` codegen timing). Consumed by `symbolicODE` and, downstream,
  `cubie.integrators.matrix_free_solvers` and the implicit algorithms.
### External
- `sympy` only at the conversion boundary (via `engine.from_sympy`). `numba` (CUDA) is the
  target of the emitted source, invoked downstream, not here.
