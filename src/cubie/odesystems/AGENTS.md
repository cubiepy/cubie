<!-- Parent: ../AGENTS.md -->

# odesystems

## Purpose
Defines the abstract and concrete representations of CUDA-backed ODE systems.
`BaseODE(CUDAFactory)` is the abstract root — **never instantiated directly** — and owns the
machinery every system needs: the `CUDAFactory` cache interaction, the `ODEData`
compile-settings container (states, parameters, constants, observables, precision), and the
`SystemValues` name↔value/array mappings. `SymbolicODE` (in `symbolic/`) is the sole concrete
subclass, adding programmatic generation of the `dxdt`, Jacobian, and matrix-free solver-helper
device functions. `ODECache` is the attrs cache `build()` returns (compiled `dxdt` + optional
solver helpers); `ODEData`/`SystemSizes` bundle the system metadata integrator factories read.

See `CUDAFactory` (root) for the build/cache/`update` contract, closure capture, config, and
attrs conventions.

## Key Files
| File | Description |
|------|-------------|
| `baseODE.py` | `BaseODE(CUDAFactory)` abstract base and `ODECache(CUDADispatcherCache)` — the cache `build()` returns: `dxdt`, `observables`, and a `helpers: SolverHelperCache` member map. |
| `ODEData.py` | `ODEData(CUDAFactoryConfig)` compile-settings bundle + `SystemSizes` (frozen per-category counts passed to kernels). Holds only ODE-system state — solver-helper request parameters live with the requesting algorithm. |
| `solver_helpers.py` | Solver-helper request/product containers and request identity: `SolverHelperKind`, `HELPER_KIND_TRAITS` (the single authority for kind-level traits — stage awareness, chained membership; `STAGE_AWARE_KINDS`/`CHAINED_KINDS` are derived views), frozen `SolverHelperRequest` (kind, beta, gamma, order, canonical stage spec, `chained_kinds` for composed preconditioners), `HelperResult` (device callable + `cached_auxiliary_count`), mutable `SolverHelperCache` (`factories[source_hash]`, `members[member_hash]`). |
| `SystemValues.py` | `SystemValues` — name↔value mapping with dict/array access, precision coercion, and sympy-key conversion. |
| `__init__.py` | Re-exports `BaseODE`, `ODECache`, `ODEData`, `SystemSizes`, `SystemValues`, and (from `symbolic/`) `SymbolicODE`, `create_ODE_system`, `load_cellml_model`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `symbolic/` | IR-based CUDA code generation for `SymbolicODE(BaseODE)` (see `symbolic/AGENTS.md`). |

## For AI Agents

### ODECache and the helper member map
`build()` (implemented by `SymbolicODE`, not `BaseODE`) returns an `ODECache` holding
`dxdt`, `observables`, and `helpers: SolverHelperCache`. There is no sentinel and no
`NotImplementedError`-as-cache-miss path: unknown helper kinds fail when a
`SolverHelperRequest` is constructed, and supported kinds always return a typed
`HelperResult`. A true ODE compile-setting change rebuilds the `ODECache` and
therefore starts a fresh member map. The helper identity/reuse protocol
(source and member hashes, factory naming, binding) is owned by
`symbolic/AGENTS.md` and `symbolic/helper_registry.py`.

### BaseODE.update() — additions over the base contract
On top of the standard `CUDAFactory` update (non-underscored keys, `KeyError` on unknown unless
`silent`, returns the recognised `set`), `BaseODE.update()` also routes constant-*value* changes
through `set_constants()`, which derives a **copy** of the constants container, applies the
values to the copy, and passes the copy through `update_compile_settings` (snapshot
discipline — never mutate the instance a snapshot holds). A `precision` change
re-materialises all four embedded `SystemValues` on the replacement snapshot through
`ODEData.update`.

### config_hash folds constant values
`BaseODE.config_hash` extends the parent hash with a canonical digest over the sorted
constant items, because constants are captured into CUDA closures and so affect compiled
output while `SystemValues`' canonical identity is structural (names + precision) only.

### get_solver_helper at the base level
`BaseODE.get_solver_helper(request, cache_policy=None)` raises
`NotImplementedError`: solver helpers are generated from symbolic systems, and only
`SymbolicODE` overrides it. `cache_policy` is per-request service context for
diagnostics run on the consumer's behalf; a consumer that owns a policy binds it
once via `solver_helper_getter(policy)` and passes the returned getter around.
No consumer policy is ever stored on the system.

### The mass matrix is system-owned
The mass matrix is part of the system definition, fixed at construction (`mass=` on
`BaseODE`/`create_ODE_system`, or derived by structural simplification). It is normalised
to a canonical float64 array in `ODEData._mass` (exposed as `BaseODE.mass`). It does
**not** enter `fn_hash`: it participates only in the `source_hash` of helper kinds whose
generators bake it into source, so base `dxdt`/observables source is never renamed by an
algorithm helper's matrix. Algorithms never supply or store an `M`: mass-consuming
solver helpers read the system's own matrix, and `SingleIntegratorRunCore` rejects
explicit algorithms (at construction and on hot-swap) whenever `system.mass` is not
`None`.

### SystemValues
- **A plain Python class, not attrs** — don't use `attrs.fields()`/`has()` on it.
- Accepts `sympy.Symbol` keys (auto-converted to strings by `_convert_symbol_keys`) and
  lists/tuples of names (expanded to `{name: 0.0}` — used to declare variables before values).
- **Precision is fixed at construction:** `__init__` calls `update_param_array_and_indices()`,
  materialising `values_array` at `precision`. Reassigning `.precision` later does *not* recast
  existing values — call `update_param_array_and_indices()` again to rebuild.
- `update_from_dict()` returns the **recognised** keys as `set[str]` (not the unrecognised);
  `add_entry()`/`remove_entry()` mutate in place and rebuild the index maps —
  on unfrozen instances only.
- **Snapshot freezing is enforced:** `ODEData`'s field converters call
  `freeze()` on every container a snapshot takes. Structure (names, precision,
  packed layout) seals on all four; the constants container seals fully
  because constant values are compile-critical, so
  `system.constants.update_from_dict(...)` raises — use
  `set_constants()`/`update()`, which derive a `copy()` (unfrozen) and pass it
  through the update boundary. Parameter/state/observable *values* stay
  writable in place: they are runtime data outside configuration identity.
  Value equality (`__eq__`) exists for change detection at the update
  boundary; instances are unhashable (`__hash__ = None`) because a mutable
  value-equal container cannot satisfy the hash contract. The canonical
  serialization identity (`_cubie_canonical_`) is structural — names and
  precision — because stored values are runtime data (constants fold into
  `config_hash` separately).

### `initial_values` is an alias for `states`
`BaseODE.initial_values` and `.states` both return `compile_settings.initial_states`. Don't
confuse the property with the `initial_values` constructor argument, which feeds
`ODEData.from_BaseODE_initargs` — the canonical `ODEData` builder (never construct `ODEData`
directly).

### Adding a component category
A fifth slot (beyond states/parameters/constants/observables) touches two files in several
places: in `ODEData.py`, add the field, the `SystemSizes` count, and extend the
precision-propagation list in `ODEData.update()` and `from_BaseODE_initargs()`; in
`baseODE.py`, add a property.

### Testing
`tests/odesystems/test_ODEData.py`, `test_SystemValues.py`; `BaseODE` is covered indirectly via
`SymbolicODE` fixtures. See root for the CUDASIM/real-CUDA commands.

## Dependencies
### Internal
- `cubie.CUDAFactory` (`CUDAFactory`, `CUDAFactoryConfig`, `CUDADispatcherCache`);
  `cubie._serialize` (`canonical_digest`); `cubie._utils` (`PrecisionDType`);
  `cubie.odesystems.symbolic` (re-exported).
### External
- `attrs`; `numpy`; `sympy` (`Symbol`); `numba`/`numba-cuda` (compilation in subclasses).
