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
| `baseODE.py` | `BaseODE(CUDAFactory)` abstract base and `ODECache(CUDADispatcherCache)` — the cache `build()` returns: `dxdt`, `observables`, their `operation_counts` (`BaseODE.operation_count`), and a `helpers: SolverHelperCache` member map. A `BaseODE` pickles and deep-copies without its build cache; `copy()` returns that deep copy. |
| `ODEData.py` | `ODEData(CUDAFactoryConfig)` compile-settings bundle + `SystemSizes` (frozen per-category counts passed to kernels). Holds only ODE-system state — solver-helper request parameters live with the requesting algorithm. |
| `solver_helpers.py` | Solver-helper contract: request axes `jacobian_at` (`stage`/`state`/`step`), `prefactored`, `stacked` map to the internal `HelperVariant`; declarative `SolverHelperRole` base with capability-derived `legal_variants()`; frozen `SolverHelperRequest`; `HelperResult` (device function, buffer sizes, `operation_count`); `OperationCounts` (per-role binary-operator counts, `total(names)`) and `device_function_operation_count`; mutable `SolverHelperCache`. `jacobian_at="step"` on a non-Jacobian role normalises to `"stage"`. |
| `SystemValues.py` | `SystemValues` — name↔value mapping with dict/array access, precision coercion, and sympy-key conversion. |
| `__init__.py` | Re-exports `BaseODE`, `ODECache`, `ODEData`, `SystemSizes`, `SystemValues`, and (from `symbolic/`) `SymbolicODE`, `create_ODE_system`, `load_cellml_model`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `symbolic/` | IR-based CUDA code generation for `SymbolicODE(BaseODE)` (see `symbolic/AGENTS.md`). |

## ODECache and helpers
`build()` (implemented by `SymbolicODE`) returns an `ODECache` holding `dxdt`,
`observables` and `helpers: SolverHelperCache`. Illegal role/variant combinations fail at
`SolverHelperRequest` construction; legal requests always return a `HelperResult`. A
compile-setting change rebuilds the `ODECache`, and with it the helper map; the helper
identity protocol lives in `symbolic/AGENTS.md`. `BaseODE.get_solver_helper` raises
`NotImplementedError`; `SymbolicODE` overrides it.

## BaseODE updates and identity
- `BaseODE._update()` routes constant-value changes through `set_constants()`, which
  updates a copy of the constants container and passes it through
  `update_compile_settings`. A `precision` change re-materialises all four
  `SystemValues` through `ODEData.update`.
- `BaseODE.config_hash` adds a digest of the sorted constant items; a `SystemValues`
  canonical identity is its names and precision only.
- The mass matrix is a float64 array in `ODEData._mass` (`BaseODE.mass`); codegen reads
  it as boolean diagonal flags. Explicit algorithms and Neumann preconditioners reject a
  non-identity mass matrix.
- `BaseODE.initial_values` and `.states` both return `compile_settings.initial_states`;
  build `ODEData` through `ODEData.from_BaseODE_initargs`.

## SystemValues
- A plain class, not attrs.
- Accepts `sympy.Symbol` keys (converted to strings) and lists or tuples of names
  (expanded to `{name: 0.0}`).
- Precision is fixed at construction; after reassigning `.precision`, call
  `update_param_array_and_indices()` to recast.
- `update_from_dict()` returns the recognised keys; `add_entry()`/`remove_entry()` mutate
  in place, on unfrozen instances only.
- `ODEData`'s converters `freeze()` every container a snapshot takes: structure seals on
  all four, and constants seal fully, so `system.constants.update_from_dict(...)` raises
  (use `set_constants()`/`update()`). Parameter, state and observable values stay
  writable. Instances compare by value and are unhashable.

## Adding a component category
A fifth category (beyond states/parameters/constants/observables) needs, in `ODEData.py`,
the field, a `SystemSizes` count and entries in the precision propagation of
`ODEData.update()` and `from_BaseODE_initargs()`; and in `baseODE.py`, a property.

## Dependencies
### Internal
- `cubie.CUDAFactory` (`CUDAFactory`, `CUDAFactoryConfig`, `CUDADispatcherCache`);
  `cubie._serialize` (`canonical_digest`); `cubie._utils` (`PrecisionDType`);
  `cubie.odesystems.symbolic` (re-exported).
### External
- `attrs`; `numpy`; `sympy` (`Symbol`); `numba`/`numba-cuda` (compilation in subclasses).
