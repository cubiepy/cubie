<!-- Parent: ../AGENTS.md -->

# odesystems

## Purpose
Defines the abstract and concrete representations of CUDA-backed ODE systems.
`BaseODE(CUDAFactory)` is the abstract root — **never instantiated directly** — and owns the
machinery every system needs: the `CUDAFactory` cache interaction and the `ODEData`
compile-settings container, which holds the system's variable sets (`variables.py`) and
precision. `SymbolicODE` (in `symbolic/`) is the sole concrete
subclass, adding programmatic generation of the `dxdt`, Jacobian, and matrix-free solver-helper
device functions. `ODECache` is the attrs cache `build()` returns (compiled `dxdt` + optional
solver helpers); `ODEData`/`SystemSizes` bundle the system metadata integrator factories read.

See `CUDAFactory` (root) for the build/cache/`update` contract, closure capture, config, and
attrs conventions.

## Key Files
| File | Description |
|------|-------------|
| `baseODE.py` | `BaseODE(CUDAFactory)` abstract base and `ODECache(CUDADispatcherCache)` — the cache `build()` returns: `dxdt`, `observables`, their `operation_counts` (`BaseODE.operation_count`), and a `helpers: SolverHelperCache` member map. A `BaseODE` pickles and deep-copies without its build cache; `copy()` returns that deep copy. |
| `ODEData.py` | `ODEData(CUDAFactoryConfig)` compile-settings bundle + `SystemSizes` (frozen per-category counts passed to kernels; `drivers` is the drivers-buffer length and `driver_derivatives` the number of driver derivatives read). Holds the `SystemLayout`, the operation ordering and the mass matrix; `parameters`, `states`, `observables`, `num_drivers` and `driver_derivatives` are read from the layout. Holds only ODE-system state — solver-helper request parameters live with the requesting algorithm. |
| `layout.py` | `SystemLayout`: the variables in each device buffer, in buffer order — the states, observables and drivers as name-and-unit `VariableSet`s, the batch's `ParameterSet` (whose swept parameters are the parameters array's rows) and the driver-derivative slots after the drivers. `state_names`, `swept_names`, `observable_names`, `driver_names`, `dxdt_names` and `num_drivers` derive from them. It holds no defaults and no IR; `symbolic.engine.adapter.buffer_maps` builds the IR positions and array references once per layout. |
| `solver_helpers.py` | Solver-helper contract: request axes `jacobian_at` (`stage`/`state`/`step`), `prefactored`, `stacked` map to the internal `HelperVariant`; declarative `SolverHelperRole` base with capability-derived `legal_variants()`; frozen `SolverHelperRequest`; `HelperResult` (device function, buffer sizes, `operation_count`); `OperationCounts` (per-role binary-operator counts, `total(names)`) and `device_function_operation_count`; mutable `SolverHelperCache`. `jacobian_at="step"` on a non-Jacobian role normalises to `"stage"`. |
| `variables.py` | Frozen `Variable(name, unit)`, `State` and `Parameter` (each with a `default`) records, and `CompiledParameter(name, unit, swept, value)`, a parameter as a batch compiles it; `VariableSet`, an ordered immutable set with `get_indices`/`get_labels`, `units`, `subset` and `without_values`; `ValuedSet`, a `VariableSet` of states or parameters that adds `defaults`, `defaults_array(precision)` and `with_defaults`; and `ParameterSet`, a set of `CompiledParameter` records built by `ParameterSet.build(parameters, swept, values)`, with `swept` and `fixed` read from its records. |
| `__init__.py` | Re-exports `BaseODE`, `ODECache`, `ODEData`, `SystemSizes`, the variable classes, and (from `symbolic/`) `SymbolicODE`, `create_ODE_system`, `load_cellml_model`. |

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
- `ODEData` holds only what changes the generated code: `parameters` (a `ParameterSet`),
  and the names and units of the `states` and `observables`. A set's order is the
  layout of every array holding one value per variable.
- We keep the declared `states` and `parameters`, with their defaults, on `BaseODE`,
  outside the compile settings. `set_initial_values()` replaces the declared states and
  never rebuilds. `set_default_parameters()` replaces the declared parameters and
  recompiles the current sweep from the new defaults, so it rebuilds only when a fixed
  value changes, and it drops any values a solve gave for its own batch.
- A `ParameterSet` holds one `CompiledParameter` per parameter, in system order: swept
  with no value, or fixed at the value we compile in. We fill the parameters array with
  the swept parameters in that order. Build one with `ParameterSet.build(declared,
  swept, values)` and apply it with `BaseODE.apply_batch` or `set_swept_parameters`;
  `SymbolicODE._update` re-derives the equations when it changes.
- We hash a `ParameterSet` by each parameter's name, sweep and compiled value, and any
  other set by its names alone, so a swept parameter's default never changes
  `config_hash`.
- The sets and records are frozen, hold no dicts, and are precision-free:
  `defaults_array(precision)` builds an array at the point of use.
- The mass matrix is a float64 array in `ODEData._mass` (`BaseODE.mass`); codegen reads
  it as boolean diagonal flags. Explicit algorithms and Neumann preconditioners reject a
  non-identity mass matrix.
- `BaseODE.initial_values` and `.states` both return the declared states;
  `BaseODE.compiled_parameters` returns `compile_settings.parameters`.

## Adding a component category
A fourth category (beyond states/parameters/observables) needs a `VariableSet` field and
a `SystemSizes` count in `ODEData.py`, a constructor argument and property in
`baseODE.py`, and the set built in `SymbolicODE.__init__`.

## Dependencies
### Internal
- `cubie.CUDAFactory` (`CUDAFactory`, `CUDAFactoryConfig`, `CUDADispatcherCache`);
  `cubie._serialize` (`canonical_digest`); `cubie._utils` (`PrecisionDType`);
  `cubie.odesystems.symbolic` (re-exported).
### External
- `attrs`; `numpy`; `sympy` (`Symbol`); numba-cuda-mlir (compilation in subclasses).
