<!-- Parent: ../AGENTS.md -->

# odesystems

## Purpose
Defines the abstract and concrete representations of CUDA-backed ODE systems.
`BaseODE(CUDAFactory)` is the abstract root — **never instantiated directly** — and owns the
machinery every system needs: the `CUDAFactory` cache interaction, the declared parameters
and initial values with their defaults, the `ODEData` compile settings (precision and
operation ordering) and a child factory, the specialiser, whose compile setting is the
`ParameterSet` the batch compiles and whose build products are the system's `layout`,
`mass` and equations. `SymbolicODE` (in `symbolic/`) is the sole concrete
subclass, adding programmatic generation of the `dxdt`, Jacobian, and matrix-free solver-helper
device functions. `ODECache` is the attrs cache `build()` returns (compiled `dxdt` + optional
solver helpers); `ODEData`/`SystemSizes` bundle the system metadata integrator factories read.

See `CUDAFactory` (root) for the build/cache/`update` contract, closure capture, config, and
attrs conventions.

## Key Files
| File | Description |
|------|-------------|
| `baseODE.py` | `BaseODE(CUDAFactory)` abstract base and `ODECache(CUDADispatcherCache)` — the cache `build()` returns: `dxdt`, `observables`, their `operation_counts` (`BaseODE.operation_count`), and a `helpers: SolverHelperCache` member map. `BaseODE(precision, parameters, specialiser, ...)` takes the declared parameters and the specialiser child; `layout`, `sizes`, `mass`, `states`, `observables`, `swept_parameters`, `fixed_parameter_values` and `compiled_parameters` read the child's settings and products. `set_swept_parameters(names, fixed)` builds the `ParameterSet` for a request and updates the child; `set_default_parameters` and `set_initial_values` replace the defaults. A `BaseODE` pickles and deep-copies without its build cache; `copy()` returns that deep copy. |
| `ODEData.py` | `ODEData(CUDAFactoryConfig)`: the precision and operation ordering the system generates code with, and nothing about the parameters, which the specialiser child holds. `SystemSizes` (frozen per-category counts passed to kernels; `drivers` is the drivers-buffer length and `driver_derivatives` the number of driver derivatives read); `SystemSizes.of_layout(layout)` counts a layout's buffers. Solver-helper request parameters live with the requesting algorithm. |
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
- The parameters a batch compiles are the compile setting `compiled_parameters` of the
  specialiser child (`symbolic/specialiser.py`), a `ParameterSet` holding one
  `CompiledParameter` per parameter, in system order: swept with no value, or fixed at
  the value we compile in. We fill the parameters array with the swept parameters in
  that order. The child's build assembles the parsed system for that set, and its
  products (`layout`, `equations`, `functions`, `fn_hash`, `mass`) are what
  `BaseODE.layout`, `sizes`, `mass`, `states` and `observables` read, so reading any of
  them assembles the system without generating or compiling code.
- `system.update(compiled_parameters=...)` reaches the child by name through
  `BaseODE._update`, which also invalidates the system's own build when the child's
  settings changed, because the generated code embeds the child's products.
  `config_hash` recurses into the child, so the sweep and fixed values reach the kernel
  cache key with no field on `ODEData`.
- `BaseODE.set_swept_parameters(names, fixed)` is where a request meets the defaults: it
  compares `ParameterSet.key_for(declared, names, fixed)` with the compiled set's key
  and builds a new `ParameterSet` only when they differ, so asking for the parameters
  already compiled constructs nothing. Fixing a parameter at its default gives an equal
  set, and nothing rebuilds.
- We keep the declared `states` and `parameters`, with their defaults, on `BaseODE`,
  outside the compile settings. `set_initial_values()` replaces the default initial
  values and never rebuilds. `set_default_parameters()` replaces the declared
  parameters and recompiles the current sweep from the new defaults, so it rebuilds only
  when a fixed value changes, and it drops any values a solve gave for its own batch.
- We hash a `ParameterSet` by each parameter's name, sweep and compiled value, and any
  other set by its names alone, so a swept parameter's default never changes
  `config_hash`.
- The sets and records are frozen, hold no dicts, and are precision-free:
  `defaults_array(precision)` builds an array at the point of use. `BaseODE.states`
  and `swept_set` build a `ValuedSet` from the layout and the defaults on every read.
- The mass matrix is the specialiser product `mass` (`BaseODE.mass`), a float64 array or
  `None`; codegen reads it as boolean diagonal flags. Explicit algorithms and Neumann
  preconditioners reject a non-identity mass matrix.
- `BaseODE.initial_values` and `.states` both return the layout's states with their
  default initial values.

## Adding a component category
A fourth category (beyond states/parameters/observables) needs a `VariableSet` field on
`SystemLayout`, a `SystemSizes` count in `ODEData.py`, a property in `baseODE.py`, and
the set built in `parsing/assemble.py`.

## Dependencies
### Internal
- `cubie.CUDAFactory` (`CUDAFactory`, `CUDAFactoryConfig`, `CUDADispatcherCache`);
  `cubie._serialize` (`canonical_digest`); `cubie._utils` (`PrecisionDType`);
  `cubie.odesystems.symbolic` (re-exported).
### External
- `attrs`; `numpy`; `sympy` (`Symbol`); numba-cuda-mlir (compilation in subclasses).
