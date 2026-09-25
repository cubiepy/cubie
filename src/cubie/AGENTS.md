# cubie

## Purpose
CuBIE (CUDA Batch Integration Engine) is a Numba-CUDA JIT batch ODE/SDE solver: it
compiles CUDA device functions on the fly to integrate large numbers of systems in
parallel on NVIDIA GPUs without the user writing CUDA. This package-root directory
holds the cross-cutting infrastructure the entire codebase depends on: the
`CUDAFactory` cached-compilation base class, the singleton `buffer_registry`, shared
validators/converters (`_utils.py`), the CUDA-simulator compatibility layer
(`cuda_simsafe.py`), file-based kernel caching (`cubie_cache.py`), and timing
(`time_logger.py`). `__init__.py` assembles the public API by star-importing the
subpackages.

## Public API
`__init__.py` re-exports from the subpackages and declares `__all__`:

| Symbol | Origin | Role |
|--------|--------|------|
| `Solver`, `solve_ivp` | `batchsolving` | User-facing batch solver class and convenience function. |
| `SymbolicODE`, `create_ODE_system`, `load_cellml_model` | `odesystems` | Build ODE systems from symbolic expressions or CellML. |
| `summary_metrics` | `outputhandling` | Singleton summary-metric registry. |
| `default_memmgr` | `memory` | Global `MemoryManager` singleton. |
| `ArrayTypes` | `batchsolving` | Array-type helper exported at package level. |
| `TimeLogger`, `default_timelogger` | `time_logger` | Timing/verbosity logger and its global singleton. |
| `CUBIE_RESULT_CODES` | `result_codes` | Bit-flag status codes for the per-run status word (device→solver). |

`__init__.py` also sets `NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS="0"` at import time and
resolves `__version__` via `importlib.metadata.version("cubie")`.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package entry point: star-imports subpackages, sets the Numba occupancy-warning env var, defines `__all__` and `__version__`. |
| `CUDAFactory.py` | Core cached-compilation framework: `CUDAFactory` (ABC; exposes `jit_kwargs`, the property every `build()` splats into `@cuda.jit`; `update` merges, flattens groups and raises, `_update` is the subclass hook), `FrozenSettings` (frozen attrs base; `update(updates)` returns `(replacement, recognised, changed)` with converters and validators run on the replacement), `values_differ`, `build_config` (builds a config from a settings mapping; keys without a field go to its `FrozenSettings`-typed fields, e.g. `unroll_*`/`lineinfo`) and `nested_config_fields`, `JITFlags` (defaults from `cuda_simsafe.JIT_FLAG_DEFAULTS`), `UnrollFlags`/`UnrollChoice`/`unroll_flag_converter`/`ALL_UNROLL_PARAMETERS`, `CUDAFactoryConfig`/`_CubieConfigBase` (`FrozenSettings` snapshots whose `update` recurses into nested settings fields; carry the `jit_flags: JITFlags` and `unroll: UnrollFlags` compile settings every factory honours, with a read-only `lineinfo` passthrough; `init_kwargs` returns the `__init__` fields by `__init__` name without device-function slots), `CUDADispatcherCache`, and the `MultipleInstance*` variants. `CUDAFactory.settings_dict` merges every child factory's `settings_dict` (the `config_hash` children), writes the config's `init_kwargs` over them and keeps the class's `settings_keys` (a factory's loose-key set; `None` keeps every key); a factory that derives values overrides it to return them only as given. `copy()` is `type(self)(**settings_dict)`; factories that take a system override it. Hashing derives from `_serialize`. |
| `_serialize.py` | Versioned typed canonical serializer: `canonical_bytes`/`canonical_digest` with explicit type tags and length prefixes over the compile-setting value domain (no `str()` fallback — unsupported values raise). Every semantic identity (values_hash, config_hash, helper source/member hashes, ODE constants fold) derives from it; `SCHEMA_VERSION` prefixes every digest. Value objects join via a `_cubie_canonical_()` method. |
| `_env.py` | `CUBIE_*` environment-variable registry: `env_bool`, `lineinfo_default` (`CUBIE_LINEINFO`), `cache_dir_default` (`CUBIE_CACHE_DIR`), `kernel_cache_dir_default` (`CUBIE_KERNEL_CACHE_DIR`), `max_cache_entries_default` (`CUBIE_MAX_CACHE_ENTRIES`), `operation_ordering_default` (`CUBIE_OPERATION_ORDERING`, the codegen ordering-policy default consumed by every `operation_ordering` signature default), `block_schedule_default`/`active_block_schedule`/`set_active_block_schedule` (`CUBIE_BLOCK_SCHEDULE`, the typed-IR scheduler policy, default `anchor_dfs`; the active value folds into the kernel-cache fingerprint), plus documentation of `CUBIE_CUDA_BACKEND`. Env values are defaults; explicit solver arguments always win. |
| `cuda_backend.py` | Resolves which CUDA backend cubie compiles against: `CUDA_BACKEND` (`"numba-cuda"` or `"mlir"`) and `IS_MLIR`. `CUBIE_CUDA_BACKEND` picks explicitly; otherwise the installed backend is used (mlir preferred when both are installed; numba-cuda preferred under CUDASIM). Consumed by `cuda_simsafe`, `cubie_cache`, and `__init__` (which imports `backend/_numba_cuda_compat` or `backend/_mlir_compat` accordingly). |
| `cache_root.py` | Single source of truth for the on-disk cache root (`get_cache_root`/`set_cache_root`/`get_cache_root_override`; precedence: `set_cache_root` override → `CUBIE_CACHE_DIR` → `<cwd>/generated`). The codegen, CellML parse, and compiled-kernel caches all resolve through it. |
| `buffer_registry.py` | Singleton `buffer_registry` (`BufferRegistry`) managing CUDA buffer metadata, layout, aliasing, and allocator generation; defines `CUDABuffer` and `BufferGroup`. |
| `_utils.py` | Shared helpers: `PrecisionDType`, precision/buffer validators + converters, attrs validator factories, `device_function_field`, `merge_kwargs_into_settings`, `ensure_nonzero_size`, `slice_variable_dimension`, `clamp_factory`. |
| `cuda_simsafe.py` | The CUDA import hub and CUDASIM compatibility layer. Re-exports the active backend's `cuda` module object, scalar types, `numba_from_dtype`, driver internals, cache base classes, and `INLINE_ALWAYS`; owns `CUDA_SIMULATION`, `JIT_FLAG_DEFAULTS`, `compile_kwargs`, `get_jit_kwargs` (renders a `JITFlags` via the `CUDAFactory.jit_kwargs` property, the single sanctioned route to `@cuda.jit` kwargs), `UnrollFlag` (the `(unroll, count)` pair `unroll_if` reads), `from_dtype`, `is_devfunc`/`is_cuda_array`, the warp intrinsics, `stwt`, `fmax`/`fmin` (NaN-dropping max/min on both device and simulator), and memory-manager/array stand-ins. Every other module imports CUDA symbols from here, never from a backend package; driver and compiled-kernel queries live in `backend/utils.py`. |
| `cubie_cache.py` | File-based persistence of compiled kernels: `CUBIECache*` and `toolchain_fingerprint`. `BatchSolverKernel.build_kernel()` constructs a `CUBIECache` from its config's hash-excluded `cache` settings plus `fn_hash`/`config_hash` and attaches it to the dispatcher. The source stamp folds a minimal ABI/toolchain fingerprint (schema version, Python ABI tag, backend id, backend serialization package versions), not a full package freeze. Built on the backend's cache bases from `cuda_simsafe` (numba-cuda `_Kernel` serialization or the MLIR compile-result scheme). |
| `time_logger.py` | `TimeLogger` (verbosity-gated timing; `"silent"` records CUDA events and prints nothing), `CUDAEvent` (GPU event pair with CUDASIM fallback), `TimingEvent`, `default_timelogger`. |
| `result_codes.py` | `CUBIE_RESULT_CODES(IntFlag)` — the package-central status vocabulary OR-combined into the per-run status word — plus `decode_status_codes` for host-side decoding. |
| `array_interpolator.py` | `ArrayInterpolator(CUDAFactory)`: builds piecewise-polynomial (spline) coefficients from sampled driver arrays and compiles `evaluate_all` (Horner evaluation of all drivers at `t`) and `evaluate_time_derivative`. Samples are the nested `drivers` compile setting, a `DriverSamples`: name-keyed columns with `t0` and `driver_sample_period`, equal by value, hashed by names, time base and sample count. `coefficients` is a cached build output. With no drivers the evaluators are `None` and the table is `(0, 0, order + 1)`. Owned by `BatchSolverKernel` as `driver_interpolator` (`Solver.driver_interpolator` is a passthrough); defines `ArrayInterpolatorConfig` (`boundary_condition=None` derives `periodic`/`clamped` from `wrap`), `InterpolatorCache`. `driver_sample_period` is the sample spacing; `dt` is the integrator timestep. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `backend/` | numba-cuda and numba-cuda-mlir compatibility shims, MLIR lowering for cubie device utilities, and the typed-IR block scheduler (see `backend/AGENTS.md`). |
| `batchsolving/` | High-level batch integration API: `Solver`, `solve_ivp`, `BatchSolverKernel`, grid building, system interface, result containers, host/device array managers (see `batchsolving/AGENTS.md`). |
| `integrators/` | Numerical integration components: `SingleIntegratorRun`, algorithm step factories, step controllers, matrix-free solvers, and CUDA loop builders (see `integrators/AGENTS.md`). |
| `memory/` | GPU memory subsystem: `MemoryManager` singleton (`default_memmgr`), array request/response containers, stream groups, CuPy-backed device/pinned allocation (see `memory/AGENTS.md`). |
| `odesystems/` | ODE system definitions and IR-based CUDA code generation (see `odesystems/AGENTS.md`). |
| `outputhandling/` | Output and summary-metric system (see `outputhandling/AGENTS.md`). |
| `gui/` | Optional Qt-based editors for `SymbolicODE` constants/parameters/states (see `gui/AGENTS.md`). |
| `vendored/` | Third-party code vendored as compatibility shims (see `vendored/AGENTS.md`). |

## CUDAFactory (cached compilation)
Subpackage `AGENTS.md` files describe only what they add to these conventions.
- **Subclasses override `build()`** to return a `CUDADispatcherCache` subclass
  instance (a bare callable raises `TypeError`) and **expose compiled device
  functions as named properties** (`device_function`, `dxdt_fn`); callers use those
  properties, never `build()` or `get_cached_output(name)`. A stored device-function
  reference goes stale when settings change: rebuild is lazy, on the next property
  access.
- **`build()` compiles by closure capture:** it bakes the current `compile_settings`,
  registry allocators and child device functions into the compiled function as
  closure constants, so any settings change needs a rebuild.
- **Three cache layers:**
  1. **Compiled-kernel cache** (`cubie_cache`), keyed by `config_hash` (each
     factory's `values_hash` re-hashed with its children's). An unchanged
     `config_hash` reuses the on-disk kernel. `BaseODE` folds constant *values*
     into its `config_hash`.
  2. **Object build cache** (`CUDAFactory._cache` + `_cache_valid`).
     `update_compile_settings` invalidates it only when a setting changed.
  3. **Codegen source cache** (`odesystems/symbolic`: `ODEFile`), keyed by
     `fn_hash`: one generated source file per source identity.
- **`update` / `update_compile_settings` contract:** keys are the non-underscored
  field names; an unrecognised key raises `KeyError` unless `silent=True`; returns a
  **`set`** of recognised labels. `CUDAFactory.update` merges `updates_dict` and
  kwargs, flattens dict values (group names count as recognised) and calls
  `_update(updates, silent)`, the only method subclasses override (default:
  `update_compile_settings`); `_update` may add entries to `updates` and returns the
  names any child took. The config-level `update` is pure: it returns
  `(replacement, recognised, changed)`; `update_compile_settings` swaps the
  replacement in and invalidates the build when any field changed. Change detection
  compares post-conversion values: device-function fields by identity, arrays
  elementwise, the rest by inequality. `values_hash` covers only eq-participating
  fields, but a replaced `eq=False` callable still rebuilds the consumer.
  Snapshots are sealed: assignment raises, array fields are owned read-only copies,
  and `SystemValues` containers freeze at the snapshot boundary (structure always;
  values too for constants). Updates modify a `copy()` and pass it through the
  boundary. A subclass `update` documents only its additions.
- **`config_hash` recurses into child `CUDAFactory` attributes** (direct attributes,
  alphabetical). Attribute names in `_excluded_child_factories` contribute nothing
  to identity (diagnostic services).
- **`MultipleInstanceCUDAFactory`** maps prefixed external keys (`krylov_atol`) to
  unprefixed fields via `instance_label`; build configs with `build_config(...)`.
  `products` and `settings_dict` carry the label (`krylov_linear_solver_fn`);
  `prefixed(name)` returns the labelled key. A consumer field a labelled child fills
  carries that label (`newton_nonlinear_solver_fn`, `error_linear_solver_fn`); a
  labelled consumer's own device slot is `device_function_field(prefixed=True)`,
  keyed `{label}_norm_fn`.

## Config classes (attrs convention)
- Compile settings are **frozen** attrs classes subclassing `CUDAFactoryConfig` /
  `MultipleInstanceCUDAFactoryConfig`. Variable- or float-typed members are stored
  underscore-prefixed and exposed, type-coerced, through a same-named property;
  `__init__` and `update` take the non-underscored names.
- Derived fields are recomputed in `__attrs_post_init__` (via `object.__setattr__`).
  Converters and validators re-run on every replacement, so collections normalise to
  immutable values (tuples, not lists).
- A system runs at **one precision** (`ALLOWED_PRECISIONS` = float16/32/64); float
  members are returned cast via `self.precision(...)`.
- `build()` writes the sizes, flags and device functions a parent reads into the
  cache; parents read children's `products` at update time to feed siblings.
- **Settings resolve at the `Solver`:** `batchsolving/solver_settings.py` records the
  given settings (`None` = not given), `batchsolving/resolve_defaults.py` resolves the
  rest, and the Solver passes the effective settings as one flat dict down `update`.
  A factory resolves nothing; a config `update` (or `build_config`) given `None` sets
  the field's declared default. `build_config` folds loose nested keys (`unroll_*`,
  jit flags, `cache_*`) through the config's `update`; every child's `ALL_*` key set
  includes `ALL_UNROLL_PARAMETERS` and `ALL_JIT_PARAMETERS`.
- **Device functions are named `<full words>_fn`** on both sides (`dxdt_fn`,
  `step_fn`, `loop_fn`). Counts are `n_states`, `n_observables`, `n_parameters`,
  `n_drivers`; `solver_width` is a solver's vector length.
- **Device-function fields use `device_function_field()`** (`_utils`): compared by
  identity, excluded from hashing. `CUDAFactory.products` returns a build's cache
  fields by name, building first if needed.
- **`eq=False`** excludes a field from equality and hashing (callables; array fields
  use a custom `eq`). Plain `dict` fields are rejected; wrap compile-critical data in
  an attrs class. Every eq-participating value must be canonically serializable
  (`_serialize.py`).

## buffer_registry (CUDA memory layout)
- A factory with managed buffers registers them with `register(name, parent, size,
  location, persistent=...)` in `register_buffers()` and allocates through
  `get_allocator(name, self)` / `get_child_allocators(parent, child, name)`; sizes
  come from the `*_buffer_size` properties. Locations: `'local'` or `'shared'`.
- `register_child(parent, child, name)` (called by `get_child_allocators`) records
  the ownership edge; `clear_parent` cascades through it when a component is
  swapped. Both take `aliases=` naming a shared entry the child's window overlaps;
  every re-registration repeats it. An alias touching persistent storage gets its
  own allocation instead.
- A buffer's `dtype` defaults to the parent's precision; others pass `dtype=`
  (`np_int32` counters) and get their slice through a `view` of the parent array.
- A child `AGENTS.md` lists the buffers it registers and nothing about the
  registry itself.

## Array sizing (`ArraySizingClass`)
Host-side shapes come from `ArraySizingClass` subclasses
(`outputhandling/output_sizes.py`). Call `.nonzero` (every dimension floored to 1)
before allocating.

## Device code
- Import every CUDA symbol from `cuda_simsafe`, never from a backend package
  (`numba.cuda`, `numba_cuda_mlir`).
- Device-function bodies are bracketed with `# no cover: start` / `# no cover: end`
  (coverage cannot see compiled code); keep the brackets when editing.
- Import NumPy scalar types with an `np_` prefix (`from numpy import float32 as
  np_float32`); prefer explicit imports over `import numpy as np`.
- Compile-time loops: `for i in unroll_if(range(n), flag)`, `flag` a closure local
  from `compile_settings.unroll`; warp-voted iteration loops take
  `unroll_newton_exits` (Newton, DAE initialiser) or `unroll_krylov_exits` (Krylov);
  runtime-bounded loops use plain `range`.
- Prefer `selp` over branches, except on closure constants, which the compiler prunes.
- All threads compute; gate their commits, not their participation.
- Iteration exits are warp-coherent: `all_sync` or `any_sync`.
- Tests never `xfail`, `importorskip` or otherwise conditionally skip.

## Gotchas
- `cubie_cache` uses numba-cuda internals (`_Kernel`, `IndexDataCacheFile`,
  `CUDACache`) that can change between numba-cuda versions; under CUDASIM it uses
  the vendored `CUDACache`.
- `default_timelogger` starts at `verbosity=None` (no timing); enable with
  `time_logging_level=` on `solve_ivp` or `Solver`.

## Dependencies
### Internal
This root infrastructure is depended on by every subpackage. Within the root, the
dependency order is roughly `cuda_simsafe` ← `_utils` ← `buffer_registry`,
`CUDAFactory`; `cubie_cache` depends on `CUDAFactory`, `_utils`, `cuda_simsafe`,
`time_logger`, `vendored.numba_cuda_cache`, and `cache_root`. All three disk
cache layers (codegen source, CellML parse, compiled kernels) resolve their
base directory through `cache_root.get_cache_root()`; `set_cache_root()`
relocates them together.
