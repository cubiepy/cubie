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
| `symbolicODE.py` | `SymbolicODE(BaseODE)` plus `create_ODE_system()`. Owns parsing, codegen caching, constant/parameter conversion, units, optional Qt GUIs, and `get_solver_helper(request)` which resolves requests through `helper_registry`. |
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
| `structural/` | MTK-style structural simplification and tearing (singular derivative-block removal, alias elimination, Pantelides index reduction, dummy derivatives, Carpanzano/Modia tearing); runs on every parsed system (see `structural/AGENTS.md`). |

## get_solver_helper
`build()` compiles only `dxdt` and `observables`; every other device function comes from
`get_solver_helper(role, **request_kwargs)`, `role` a role name or preconditioner type,
which assembles an immutable `SolverHelperRequest`. Each request has two canonical
identities:
- `helper_source_hash` (role, variant, `fn_hash`, and the stage spec and cache
  selection where the variant uses them) names the generated factory
  `<role>_<variant>_s<source hash>` in the `ODEFile`.
- `helper_member_hash` (source hash plus the role's binding arguments) keys the bound
  member in `ODECache.helpers`; different bindings share one generated factory.

Adding a helper is one `SolverHelperRole` subclass in `helper_registry.py`
(capabilities + `generate`) and a generator in `codegen/`; registration is automatic.
`preconditioner_type` resolves through `PRECONDITIONER_ROLES`; `no_preconditioner`
answers `"none"` with an identity (`out = v`) at the solver width. `Role.validate` runs
on every request, cache hits included: Neumann rejects mass-matrix systems, Jacobi
rejects series orders on stacked multi-stage operators. Variants reading `cached_aux`
(`cached`, `cached_stacked`, `prefactored`) come with their role's prepare companion
(`Role.prepare_request_kwargs`: `prepare_jac` for iterative helpers,
`lu_prepare_blocks` for prefactored LU) and its buffer size on
`HelperResult.cached_auxiliary_count`; `lu_solve`'s factor-buffer length is
`HelperResult.lu_nnz` (`None` when unsized). `operator_beta`/`operator_gamma` (and the LU
solve's `a_ij`) fold into the source as literals, keyed into the source hash through the
role's `folded_args`; factories bind `precision` (plus `order` for preconditioners), and
constant values key the bound member. Mass-consuming helpers read
`compile_settings.mass`, `None` or a 0/1 diagonal: a zero row selects the residual form,
an identity row the plain form.

## Constant specialisation
Constant values substitute into the equations as IR literals at the head of the codegen
pipeline (`parsing/parsed_system.py`); generated source never names a constant and device
functions capture no constant closures. `SymbolicODE._parsed_system` (a `ParsedSystem`)
re-specialises on every constant-value change: substitution, constructor folding,
structural simplification and tearing, updating the state layout and mass matrix.
`set_constants` pushes every changed compile setting through one
`update_compile_settings`. Live solvers take changes through `Solver.update`; a direct
`set_constants` on a solver-attached system raises at the next solve.

`make_parameter`/`make_constant` evolve the checkpoint's category maps and re-specialise:
a freed constant returns to the equations as a parameter-array symbol, a new constant
folds in as a literal. The checkpoint is replaced only when specialisation succeeds.

## build() and system identity
`build()` recomputes the system hash first and switches `self.gen_file` to a fresh
`ODEFile` when the source identity changed. The identity is `fn_hash` from
`hash_system_definition`: equations (constants folded), name-sorted state, dxdt,
parameter, driver and observable layouts, constant labels, derivative helpers and
function aliases. Equations sort by LHS name, so string and SymPy input hit the same
cache.

## Gotchas
- `ODEFile.function_is_cached` parses the file textually: it needs a top-level
  `def <name>(` with a `return` one indent level in; a factory without one is never
  cached.
- Generated files land under `cubie.cache_root.get_cache_root()` (default
  `<cwd>/generated`, read at `ODEFile` construction, relocatable with
  `set_cache_root()`).
- `IndexedBaseMap.push`/`pop` rebuild the `sympy.IndexedBase` and reindex every entry;
  re-read `ref_map` array references after `make_parameter`/`make_constant`.
- `constants_gui`/`states_gui` import `cubie.gui` inside the method (Qt is optional).

## Dependencies
### Internal
- `cubie.odesystems.baseODE` (`BaseODE`, `ODECache`); `cubie.odesystems.symbolic.codegen` (all
  source emitters); `cubie.odesystems.symbolic.parsing` (`parse_input`, `IndexedBases`,
  `ParsedEquations`, `JVPEquations`); `cubie.array_interpolator.ArrayInterpolator`
  (driver-array setup); `cubie._utils` (`PrecisionDType`), `cubie.time_logger.default_timelogger`,
  `cubie.cuda_simsafe` (in the generated module header), `cubie.gui.*` (lazy, optional).
### External
- `sympy`; `numpy` (`float32`, `ndarray`); `numba`/`numba.cuda` (generated header + precision types).
