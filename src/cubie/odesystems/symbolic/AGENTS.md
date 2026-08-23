<!-- Parent: ../AGENTS.md -->

# symbolic

## Purpose
CUDA codegen pipeline that turns symbolic ODE definitions into JIT-compiled Numba-CUDA
device functions. SymPy is a parse-boundary translation layer only: string input parses
through `sympy.parse_expr` and SymPy input is accepted directly, but every expression
converts to the hash-consed IR in `engine/` inside `parsing/normalise.py`, and all
compute (structural simplification, differentiation, substitution, CSE,
hashing, printing) runs on IR nodes. This top level holds the user-facing system class
(`SymbolicODE`), the disk-backed source cache (`ODEFile`), the symbol-to-device-index maps
(`IndexedBaseMap`/`IndexedBases`, SymPy-facing for GUIs and `SystemValues`), and shared
utilities (hashing, the reserved codegen prefix). Equation parsing lives in `parsing/`,
and every parsed system runs through the structural simplification in `structural/`;
the CUDA source emitters live in `codegen/`. `SymbolicODE` orchestrates both: parse via
`parsing.parse_input`, generate
`dxdt`/`observables`/solver-helper factories via `codegen`, write them to a per-system module on
disk, and reload the compiled factories. As the sole concrete `BaseODE` subclass it is the main
entry point for defining systems — users construct one via `create_ODE_system()` (string / SymPy
/ callable) or `load_cellml_model()`.

See `CUDAFactory` (root) for the build/cache/`update` contract, closure capture, config, and
attrs conventions; `BaseODE` (parent, `../AGENTS.md`) for `ODECache`/`config_hash`/`set_constants`.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Star-imports `codegen`, `parsing`, `indexedbasemaps`, `odefile`, `symbolicODE`, `sym_utils`; declares `__all__ = ["SymbolicODE", "create_ODE_system", "load_cellml_model"]`. |
| `symbolicODE.py` | `SymbolicODE(BaseODE)` plus `create_ODE_system()`. Owns parsing, codegen caching, the swept/folded repartition (`set_categories`), units, optional Qt GUIs, and `get_solver_helper(request)` which resolves requests through `helper_registry`. |
| `helper_registry.py` | Concrete solver-helper roles: one `SolverHelperRole` subclass per role (`LinearOperator`, `NeumannPreconditioner`, `JacobiPreconditioner`, `LuSolve`, `LuPrepareBlocks`, `LuSmoothingSolve`, `Residual`, `InitResidual`, `InitLuSolve` (consistent-initialisation forms, PLAIN-only), `ApplyMass`, `EvaluateInvMassF`, `TimeDerivativeRHS`, internal `PrepareJac`), each declaring capabilities and implementing `generate`; Neumann also implements `validate`. Defines `helper_source_hash` and `helper_member_hash`. |
| `odefile.py` | `ODEFile` disk cache. Writes generated factory source to `<cache root>/<name>/<name>_<hash10>.py` (root from `cubie.cache_root`; one file per source identity, so alternating constant sets keep their cached source), hash-guards staleness, checks per-function caching, and imports factories via `importlib`. |
| `indexedbasemaps.py` | `IndexedBaseMap` (named scalar symbols → fixed-size `sympy.IndexedBase`, held in sorted name order) and `IndexedBases` (bundle of state/parameter/constant/observable/driver/dxdt maps). Provides `from_user_inputs`, constant↔parameter conversion, units, ref/index/symbol maps. |
| `sym_utils.py` | Shared helpers: `hash_system_definition` (SHA-256, order-independent, over the IR pairs' reprs), `RESERVED_CODEGEN_PREFIX`, plus SymPy `topological_sort`/`cse_and_stack`/`prune_unused_assignments` retained for the CPU reference tests (production code uses the IR equivalents in `engine/`). |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `engine/` | Hash-consed expression IR and its compute passes: SymPy conversion, differentiation, substitution, CSE, ordering, pruning, and the CUDA printer (see `engine/AGENTS.md`). |
| `codegen/` | CUDA source emitters for dxdt, observables, Jacobian/JVP, linear operators, preconditioners, residuals, and time derivatives, all computing on the `engine/` IR (see `codegen/AGENTS.md`). |
| `parsing/` | Converts string / SymPy / callable / CellML input into `ParsedEquations` + `IndexedBases`, plus `JVPEquations` and auxiliary-caching heuristics; one normalised front end feeds every system through `structural/` (see `parsing/AGENTS.md`). |
| `structural/` | MTK-style structural simplification and tearing (alias elimination, Pantelides index reduction, dummy derivatives, Carpanzano/Modia tearing); runs on every parsed system (see `structural/AGENTS.md`). |

## For AI Agents

### get_solver_helper — the single helper entry point
`build()` compiles only `dxdt` and `observables`; every other device function comes from `get_solver_helper(role, cache_policy=None, **request_kwargs)`, where `role` is a role name or preconditioner type string and the getter assembles the immutable `SolverHelperRequest`.
Two identities per request, both from the canonical serializer:
- `helper_source_hash` (role + variant + `fn_hash` + stage spec and
  cache selection where the variant applies) names the generated
  factory `<role>_<variant>_s<full source hash>` in the `ODEFile`.
- `helper_member_hash` (source hash + the binding arguments the role
  declares) keys the bound member in `ODECache.helpers`. Different
  bindings reuse one generated factory.
Adding a helper means one `SolverHelperRole` subclass in
`helper_registry.py` (capabilities + `generate`) and a generator entry in
`codegen/`; registration is automatic. The algorithm layer passes its
`preconditioner_type` string as the role name; the request converter
resolves it through `PRECONDITIONER_ROLES`. The `no_preconditioner`
role answers `preconditioner_type="none"` with an identity
preconditioner (`out = v`) at the request's solver width. Validation hooks
(`Role.validate`) run per
request, including cache hits: the Neumann hook rejects mass-matrix
systems before its convergence diagnostic; the hook resolves the
consumer's own evaluator from `cache_policy` — `SymbolicODE` keys one
`NeumannRHSEvaluator` per policy. Members whose variant reads
`cached_aux` (`cached`, `cached_stacked`, `prefactored`) are served
with their role-declared prepare companion
(`Role.prepare_request_kwargs`: `prepare_jac` for the iterative
helpers, `lu_prepare_blocks` for the prefactored LU variants) and its
buffer size on `HelperResult.cached_auxiliary_count`; the `lu_solve`
role's per-call factor-buffer length travels on
`HelperResult.lu_nnz` (each source stamps `aux_count`/`lu_nnz`,
`None` when unsized). Every
beta/gamma-consuming helper folds those values (the LU solve also a
baked `a_ij`) into the source as literals, keyed into the source
hash through the role's `folded_args` instead of the factory
binding; factories bind `precision` (plus `order` for the
preconditioners), and constant values always key the bound member.
Mass-consuming helpers read the
system's own `compile_settings.mass` — always `None` or a 0/1 diagonal,
consumed by codegen as per-row flags (a zero row selects the residual
form, an identity row the plain form).

### Constant specialisation — values are source identity
Constant values substitute into the equations as IR literals at the head of
the codegen pipeline (`parsing/parsed_system.py`); generated source never
names a constant and device functions capture no constant closures. The
checkpoint on `SymbolicODE._parsed_system` (a `ParsedSystem` for every input
pathway) drives re-specialisation on every constant-value change:
substitution, constructor folding, structural simplification, and tearing,
updating the state layout and mass matrix to match the values.
`set_constants` derives the new products and pushes every changed compile
setting through `update_compile_settings` in one call. Live solvers receive
changes through `Solver.update`; a direct `set_constants` on a
solver-attached system raises at the next solve.

### build() and system identity
`build()` compiles `dxdt`+`observables` into the `ODECache`, first recomputing the system hash —
swapping `self.gen_file` to a fresh `ODEFile` when the specialised source identity changed.
The identity is `fn_hash` from `hash_system_definition`: equations (with constant values folded
as literals), name-sorted state/dxdt/parameter/driver/observable layouts, constant labels,
derivative helpers, and function aliases. Each source identity keeps its own
`ODEFile`. Equations sort by
LHS name, so string and SymPy input hit the same cache.

### Swept/folded repartition
`set_categories(parameters, constants)` takes a full partition of the
system's named values, evolves the checkpoint's category maps, and
re-specialises once for any number of moves: a promoted name returns to
the equations as a symbol reading the parameters array, and a folded
name's value bakes into the source as a literal. The two dicts must
cover exactly the named-value pool. An unchanged partition with changed
folded values routes through `set_constants`, which re-specialises on
any value change. `update()` forwards updates to every existing Neumann
diagnostic evaluator. `Solver` calls `set_categories` per solve, deriving
the partition from the batch grid.

### Codegen cache gotchas (`ODEFile`)
- `function_is_cached` parses the generated file textually: it needs a top-level `def <name>(`
  with a `return` one indent level in. A generator that emits a factory without a `return` is
  treated as uncached forever.
- Output lands under `cubie.cache_root.get_cache_root()` — by default
  `<cwd>/generated`, evaluated at `ODEFile` construction, relocatable with
  `set_cache_root()` — not under the package.

### IndexedBaseMap rebuilds on structural change
`push` inserts at the sorted position and `pop` removes, both rebuilding the
`sympy.IndexedBase` and reindexing every entry, so any `ref_map` array reference
captured before a repartition goes stale — re-read it after `set_categories`.

### Qt GUIs are lazily imported
`constants_gui`/`states_gui` import the `cubie.gui` editors *inside* the method (Qt is optional).
Never import Qt or `cubie.gui` at module top level.

### Testing
`tests/odesystems/symbolic/` (`test_symbolicode.py`, `test_odefile.py`, `test_indexedbasemaps.py`,
`test_sym_utils.py`, `test_solver_helpers.py`); codegen tests under `.../codegen/`. Prefer real
`SymbolicODE` fixtures (`conftest.py`). See root for CUDASIM/real-CUDA commands.

## Dependencies
### Internal
- `cubie.odesystems.baseODE` (`BaseODE`, `ODECache`); `cubie.odesystems.symbolic.codegen` (all
  source emitters); `cubie.odesystems.symbolic.parsing` (`parse_input`, `IndexedBases`,
  `ParsedEquations`, `JVPEquations`); `cubie.array_interpolator.ArrayInterpolator`
  (driver-array setup); `cubie._utils` (`PrecisionDType`), `cubie.time_logger.default_timelogger`,
  `cubie.cuda_simsafe` (in the generated module header), `cubie.gui.*` (lazy, optional).
### External
- `sympy`; `numpy` (`float32`, `ndarray`); `numba`/`numba.cuda` (generated header + precision types).
