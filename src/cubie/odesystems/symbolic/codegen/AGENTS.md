<!-- Parent: ../AGENTS.md -->

# codegen

## Purpose
IR-to-CUDA **source generators**. Each public `generate_*` function takes parsed equations
(`ParsedEquations`/`JVPEquations`) plus an index map (`IndexedBases`), converts them once to
the `engine/` IR through `engine.adapter.system_ir`, and returns a
**Python source string** defining a `*_factory(constants, precision, ...)` function. Nothing here
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
| `jacobian.py` | Pure-symbolic (emits no CUDA): `generate_jacobian` (full analytic Jacobian via chain rule over auxiliary assignments; returns row-major lists of IR expressions) and `generate_analytical_jvp` (returns `JVPEquations` with `j_ij` entry symbols and `Arr("jvp", i)` product terms). Memoised in a module-level `_cache` keyed by `get_cache_key` over interned IR nodes. |
| `linear_operators.py` | Emits the matrix-free linear operator and JVP cache helpers: `generate_operator_apply_code`, `generate_operator_apply_at_state_code` (error smoothing; no increment substitution), `generate_cached_operator_apply_code`, `generate_prepare_jac_code` (populates `cached_aux`, returns `(code, aux_count)`), `generate_cached_jvp_code`, `generate_n_stage_linear_operator_code` (flattened FIRK), `generate_apply_mass_code` (`apply_mass(v, out)` = `M @ v`). `*_from_jvp` variants take a prebuilt `JVPEquations`. |
| `preconditioners.py` | Emits the Neumann-series and diagonal-Jacobi preconditioners (`generate_neumann_preconditioner_code` + cached/n-stage/at-state variants, `generate_jacobi_preconditioner_code` + cached/n-stage/at-state variants). Emitted signature ends `..., v, out, jvp`. The Neumann family is the only generator whose `order` argument is live (truncation degree, factory default 1). |
| `nonlinear_residuals.py` | Emits the Newton residual: `generate_residual_code` / `generate_stage_residual_code` (single stage, SDIRK/ESDIRK) and `generate_n_stage_residual_code` (flattened FIRK). |
| `neumann_convergence.py` | Neumann-series convergence diagnostic: `NeumannRHSEvaluator` (a `CUDAFactory` building a finite-difference Jacobian kernel over the compiled `dxdt`; cache policy fixed at construction, ownership in `../AGENTS.md`), `neumann_spectral_radius`, `check_neumann_convergence`. Runs as the registry validation hook on `neumann_*` requests; reports the initial-state radius per unit `h` (FIRK) or per unit `a_ij*h`. |
| `_stage_utils.py` | Shared FIRK helpers: `prepare_stage_data` (Butcher `A`/`c` → IR rows, nodes, stage count) and `build_stage_metadata` (emit `_cubie_codegen_c_<i>`, `_cubie_codegen_a_<i>_<j>` symbol assignments). Used by every `n_stage_*` generator. |

## Generator variants
Most callbacks come in up to three forms, differing only in how state/auxiliaries are supplied —
this is a property of the emitted code, not separate subsystems:

- **single-stage** (`generate_operator_apply_code`, `generate_residual_code`,
  `generate_neumann_preconditioner_code`): the `state` argument is the stage increment; the
  generator substitutes `state_sym → base_state[i] + a_ij*state[i]` inline.
- **`*_at_state`** (error smoothing): same signature as single-stage but no increment
  substitution — J is evaluated at the `state` argument, `base_state` is unused, and
  `a_ij` scales the matrix only.
- **`n_stage_*`** (FIRK): one flattened system of `s·n` unknowns; stage coupling (`A⊗J`) and
  per-stage time nodes are baked in via `_stage_utils`.
- **`*_cached` / `prepare_jac`** (Rosenbrock-W): `state` is the actual state (no substitution);
  selected auxiliaries are precomputed once per step into `cached_aux` by `prepare_jac` and read
  back by the operator/JVP/preconditioner. `GenericRosenbrockWStep` requests these
  (`prepare_jac` — whose `HelperResult` carries `cached_auxiliary_count` —
  `linear_operator_cached`, `neumann_preconditioner_cached`) and runs
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

## For AI Agents

### Generators emit strings, not functions
Every public `generate_*` builds Python source from a module-level `*_TEMPLATE`. To wire in a
new one, add a `SolverHelperKind` and a registry entry in `../helper_registry.py` —
generating the string alone does nothing. **Templates are
indentation-sensitive:** bodies come from `print_cuda_multiple(...)` then joined with explicit
leading spaces (8 inside a factory body, 12 inside the preconditioner's `for _ in range(order)`
loop). Preserve the exact counts or the generated source won't parse.

### Sign and coefficient convention
Operator `β·M·v − γ·a_ij·h·(J·v)` (explicit `a_ij`); residual `β·M·u − γ·h·f(base + a_ij·u)`
(`a_ij` only inside the eval point, not multiplying `f`); preconditioner approximates
`(β·I − γ·a_ij·h·J)⁻¹` with `h_eff = (γ·a_ij/β)·h`. The operator equals `∂residual/∂u` —
differentiating the residual's eval point pulls the `a_ij` down via the chain rule, which is why
it appears explicitly in the operator but implicitly in the residual. **The `a_ij` asymmetry is
deliberate; changing one form without the others breaks Newton convergence.** State evaluation
also differs by variant (see Generator variants): non-cached paths substitute
`state → base_state + a_ij*state`; cached paths read `cached_aux` and apply no substitution
(`_build_operator_body`'s `use_cached_aux` flag gates this).

### The `_cubie_codegen_` reserved namespace (#373 and successors)
User constants never bind a name at all — their values fold into the
equations as literals before generation. Every name the generators do bind — solver
scalings (`_cubie_codegen_beta`/`_cubie_codegen_gamma`), the scalar device arguments
`_cubie_codegen_h`/`_cubie_codegen_a_ij`, factory locals (`_cubie_codegen_n`,
`_cubie_codegen_order`, `_cubie_codegen_total_n`, ...), tableau metadata
(`_cubie_codegen_c_<i>`, `_cubie_codegen_a_<i>_<j>`), and builder-internal IR locals
(`_cubie_codegen_dx_*`, `_cubie_codegen_aux_*`, `_cubie_codegen_j_*`,
`_cubie_codegen_diag_*`, stage renames `_cubie_codegen_s<i>_*`) — lives in the
`_cubie_codegen_` namespace. `IndexedBases.from_user_inputs` rejects user names with
that prefix, so a user symbol can never alias a generated binding (in either
direction) at IR-merge time, factory scope, or device-body scope. When adding a
generator, bind nothing outside this namespace except the template's positional
argument names (`t` is parse-reserved and stays bare). Factory signatures still
expose plain `beta=1.0, gamma=1.0, order=1`.

### Mass matrix & order
`M` defaults to identity (`None`); the only other legal value is the 0/1 diagonal structural
simplification derives for torn systems. Codegen consumes it as per-row flags — an identity row
emits the plain form, a zero row the residual form — and no matrix values enter generated source.
`order` is in every factory signature but only the preconditioner uses it (Neumann
truncation degree, factory default `1`); operators/residuals accept and ignore it.

### Reuse jvp_equations
`generate_analytical_jvp` is the expensive step; operator/preconditioner/cached-JVP generators all
accept a prebuilt `JVPEquations` via `jvp_equations=` so one differentiation feeds the whole helper
set.

### Printer (engine)
The printer lives in `engine/printer.py` (see `engine/AGENTS.md`). Emission rules:
`precision(...)` wrapping of numeric literals (array indices stay plain integers),
integer-power multiplication chains up to `_POW_CHAIN_LIMIT` via structural Pow
rules, `CUDA_FUNCTIONS`, explicit
user-function aliases, Piecewise as branchless `selp` selections, and
scalar-to-array remapping via a name-keyed symbol map (generators pass `sysir.arrayrefs`).
Constant values arrive as `Num` literals; the printer never names a constant.

### Codegen hygiene
- `engine.prune_unused(..., output_name=...)` runs last in every `_build_*`, dropping
  intermediates that don't feed the named output array (`'out'`, `'jvp'`, `'cached_aux'`, …).
  Omitting it bloats output and leaves dangling symbols.
- Stage builders construct **one combined substitution map per stage** and apply it with a
  single `engine.xreplace` pass; all non-dx equation left-hand sides are stage-renamed by
  `build_stage_substitutions` so repeated assignment targets never collapse across stages.
- `cse` toggles the engine's `cse_and_stack` vs `topological_sort` (either emits in dependency
  order); the Jacobian/JVP cache normalises the CSE flag into its key.
- Jacobian `_cache` is process-global and unbounded, keyed by equation tuple + input/output orders
  + CSE flag; the Jacobian matrix and JVP share one entry (`"jac"`/`"jvp"`).
- Each generator module registers `default_timelogger` events at import (the functions return
  strings, not cacheable objects) and brackets work with `start_event`/`stop_event`.

### Testing
`tests/odesystems/symbolic/` (`test_dxdt`, `test_time_derivative`, `test_jacobian`,
`test_cuda_printer`, `test_solver_helpers`) + `tests/odesystems/symbolic/codegen/`
(`test_stage_utils`). These are numerically critical — validate emitted operators against
finite-difference Jacobians and the trio's algebraic consistency, not just that the source imports.
See root for CUDASIM/real-CUDA commands.

## Dependencies
### Internal
- `cubie.odesystems.symbolic.parsing` (`ParsedEquations`, `IndexedBases`, `JVPEquations`,
  `TIME_SYMBOL`); `cubie.time_logger`
  (`default_timelogger` codegen timing). Consumed by `symbolicODE` and, downstream,
  `cubie.integrators.matrix_free_solvers` and the implicit algorithms.
### External
- `sympy` only at the conversion boundary (via `engine.from_sympy`). `numba` (CUDA) is the
  target of the emitted source, invoked downstream, not here.
