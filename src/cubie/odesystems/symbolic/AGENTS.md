<!-- Parent: ../AGENTS.md -->

# symbolic

## Purpose
CUDA codegen pipeline that turns symbolic ODE definitions into JIT-compiled Numba-CUDA
device functions. SymPy is a parse-boundary translation layer only: string input parses
through `sympy.parse_expr` and SymPy input is accepted directly, but every expression
converts to the hash-consed IR in `engine/` inside `parsing/normalise.py`, and all
compute (classification, structural simplification, differentiation, substitution, CSE,
hashing, printing) runs on IR nodes. This top level holds the user-facing system class
(`SymbolicODE`), the disk-backed source cache (`ODEFile`), the symbol-to-device-index maps
(`IndexedBaseMap`/`IndexedBases`, SymPy-facing for GUIs and `SystemValues`), and shared
utilities (hashing, constant-assignment rendering). Equation parsing lives in `parsing/`;
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
| `helper_registry.py` | Declarative registry of solver-helper generators: each `SolverHelperKind` maps to a `_RegistryEntry` (generator, declared source dependencies, exact factory-binding argument names — never introspected, aux-count metadata flag, optional validation hook). Defines `helper_source_hash` and `helper_member_hash` — the two canonical helper identities. |
| `odefile.py` | `ODEFile` disk cache. Writes generated factory source to `<cache root>/<name>/<name>.py` (root from `cubie.cache_root`), hash-guards staleness, checks per-function caching, and imports factories via `importlib`. |
| `indexedbasemaps.py` | `IndexedBaseMap` (named scalar symbols → fixed-size `sympy.IndexedBase`, held in sorted name order) and `IndexedBases` (bundle of state/parameter/constant/observable/driver/dxdt maps). Provides `from_user_inputs`, constant↔parameter conversion, units, ref/index/symbol maps. |
| `sym_utils.py` | Shared helpers: `hash_system_definition` (SHA-256, order-independent, over the IR pairs' reprs), `render_constant_assignments`, `EXPONENT_ALIAS_PREFIX`, plus SymPy `topological_sort`/`cse_and_stack`/`prune_unused_assignments` retained for the CPU reference tests (production code uses the IR equivalents in `engine/`). |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `engine/` | Hash-consed expression IR and its compute passes: SymPy conversion, differentiation, substitution, CSE, ordering, pruning, and the CUDA printer (see `engine/AGENTS.md`). |
| `codegen/` | CUDA source emitters for dxdt, observables, Jacobian/JVP, linear operators, preconditioners, residuals, and time derivatives, all computing on the `engine/` IR (see `codegen/AGENTS.md`). |
| `parsing/` | Converts string / SymPy / callable / CellML input into `ParsedEquations` + `IndexedBases`, plus `JVPEquations` and auxiliary-caching heuristics; one normalised front end classifies input as explicit or DAE and routes the latter through `structural/` (see `parsing/AGENTS.md`). |
| `structural/` | MTK-style structural simplification and tearing (alias elimination, Pantelides index reduction, dummy derivatives, Carpanzano/Modia tearing); enabled automatically for DAE-shaped input or forced via `create_ODE_system(..., simplify=True)` (see `structural/AGENTS.md`). |

## For AI Agents

### get_solver_helper — the single helper entry point
`build()` compiles only `dxdt` and `observables`; every other device function
comes from `get_solver_helper(request, cache_policy=None)` with an immutable
`SolverHelperRequest`. Two identities per request, both from the canonical
serializer:
- `helper_source_hash` (kind + `fn_hash` + mass, stage spec, composed stage
  kinds, and cache selection where the trait applies) names the generated
  factory `<kind>_s<full source hash>` in the `ODEFile`.
- `helper_member_hash` (source hash + the binding arguments the registry
  entry declares) keys the bound member in `ODECache.helpers`. Different
  bindings reuse one generated factory.
Adding a helper means a kind + trait entry in
`odesystems/solver_helpers.py`, a generator in `codegen/`, and a registry
entry. Kind-level traits live in `HELPER_KIND_TRAITS`; the algorithm layer
resolves `preconditioner_type` via
`resolve_preconditioner_kind`/`resolve_chained_kind`, and a multi-type
sequence becomes one chained-kind request fused into a single generated
source. Validation hooks (the Neumann convergence diagnostic) run per
request, including cache hits; the hook resolves the consumer's own
evaluator from `cache_policy` — `SymbolicODE` keys one `NeumannRHSEvaluator`
per policy. `prepare_jac`'s auxiliary count travels on
`HelperResult.cached_auxiliary_count`. Mass-consuming helpers read the
system's own `compile_settings.mass`.

### build() and system identity
`build()` compiles `dxdt`+`observables` into the `ODECache`, first recomputing the system hash —
swapping `self.gen_file` to a fresh `ODEFile` if constants↔parameters changed since construction.
The identity is `fn_hash` from `hash_system_definition`: equations, name-sorted state/dxdt/
parameter/driver/observable layouts, constant labels, derivative helpers, and function
aliases. Constant
values are compile settings, not source identity. Equations sort by LHS name, so string and SymPy
input hit the same cache. The mass matrix is **not** part of
`fn_hash` — it enters only the `source_hash` of mass-consuming helper kinds, whose generated
factory names carry it via their source suffix.

### Constant/parameter conversion
`make_parameter`/`make_constant` update `self.indices`
(`constant_to_parameter`/`parameter_to_constant`) and then derive **copies** of the
constants and parameters containers, apply `remove_entry`/`add_entry` to the copies, and
pass both copies through `update_compile_settings`. Keep the index map and the
containers in sync. `SymbolicODE` overrides `set_constants()` to update the index map
(`self.indices.update_constants(...)`) before delegating to `BaseODE.set_constants`, and
`update()` to forward updates to every existing Neumann diagnostic evaluator.

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
captured before a conversion goes stale — re-read it after `make_parameter`/`make_constant`.

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
