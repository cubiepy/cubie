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
| `CUDAFactory.py` | Core cached-compilation framework: `CUDAFactory` (ABC; exposes `jit_kwargs`, the property every `build()` splats into `@cuda.jit`), `CUDAFactoryConfig`/`_CubieConfigBase` (frozen attrs snapshots; carry the `jit_flags: JITFlags` compile setting every factory honours, with a read-only `lineinfo` passthrough), `CUDADispatcherCache`, and the `MultipleInstance*` variants. Hashing derives from `_serialize`. |
| `_serialize.py` | Versioned typed canonical serializer: `canonical_bytes`/`canonical_digest` with explicit type tags and length prefixes over the compile-setting value domain (no `str()` fallback — unsupported values raise). Every semantic identity (values_hash, config_hash, helper source/member hashes, ODE constants fold) derives from it; `SCHEMA_VERSION` prefixes every digest. Value objects join via a `_cubie_canonical_()` method. |
| `_env.py` | `CUBIE_*` environment-variable registry: `env_bool`, `lineinfo_default` (`CUBIE_LINEINFO`), `cache_dir_default` (`CUBIE_CACHE_DIR`), `kernel_cache_dir_default` (`CUBIE_KERNEL_CACHE_DIR`), `max_cache_entries_default` (`CUBIE_MAX_CACHE_ENTRIES`), plus documentation of `CUBIE_CUDA_BACKEND`. Env values are defaults; explicit solver arguments always win. |
| `cuda_backend.py` | Resolves which CUDA backend cubie compiles against: `CUDA_BACKEND` (`"numba-cuda"` or `"mlir"`) and `IS_MLIR`. `CUBIE_CUDA_BACKEND` picks explicitly; otherwise the installed backend is used (mlir preferred when both are installed; numba-cuda preferred under CUDASIM). Consumed by `cuda_simsafe`, `cubie_cache`, and `__init__` (which imports `_numba_cuda_compat` or `_mlir_compat` accordingly). |
| `_mlir_compat.py` | numba-cuda-mlir compatibility shims, imported first thing from `__init__` on the MLIR backend: missing lowerings (Boolean bitwise/comparison ops, floored integer `%`//`//`, nested-tuple dynamic getitem, empty-slice anchoring), numpy-scalar constant handling, dynamic-shared-memory and array-literal fixes, memref pointer-offset routing, semantic local stack slots, float min/max semantics, zero-power folds, selective fastmath, and the compiler-frontend perf patches. Each shim feature-detects patched builds and no-ops there. |
| `cache_root.py` | Single source of truth for the on-disk cache root (`get_cache_root`/`set_cache_root`/`get_cache_root_override`; precedence: `set_cache_root` override → `CUBIE_CACHE_DIR` → `<cwd>/generated`). The codegen, CellML parse, and compiled-kernel caches all resolve through it. |
| `buffer_registry.py` | Singleton `buffer_registry` (`BufferRegistry`) managing CUDA buffer metadata, layout, aliasing, and allocator generation; defines `CUDABuffer` and `BufferGroup`. |
| `_utils.py` | Shared helpers: `PrecisionDType`, precision/buffer validators + converters, attrs validator factories, `build_config`, `merge_kwargs_into_settings`, `ensure_nonzero_size`, `slice_variable_dimension`, `clamp_factory`. |
| `cuda_simsafe.py` | The CUDA import hub and CUDASIM compatibility layer. Re-exports the active backend's `cuda` module object, scalar types, `numba_from_dtype`, driver internals, cache base classes, and `INLINE_ALWAYS`; owns `CUDA_SIMULATION`, `compile_kwargs`, `JITFlags`/`get_jit_kwargs` (rendered via the `CUDAFactory.jit_kwargs` property, the single sanctioned route to `@cuda.jit` kwargs), `from_dtype`, `is_devfunc`/`is_cuda_array`, the warp intrinsics, `stwt`, and memory-manager/array stand-ins. Every other module imports CUDA symbols from here, never from a backend package. |
| `cubie_cache.py` | File-based persistence of compiled kernels: `CUBIECache*`, `CachePolicy`, `CubieCacheHandler`, `ALL_CACHE_PARAMETERS`, `toolchain_fingerprint`. Cache policy (enabled/mode/limit/dir) is service configuration owned by the handler — never nested in compile settings, never hashed into identity; identity hashes are call-time arguments to `configured_cache()`. The source stamp folds a minimal ABI/toolchain fingerprint (schema version, Python ABI tag, backend id, backend serialization package versions), not a full package freeze. Built on the backend's cache bases from `cuda_simsafe` (numba-cuda `_Kernel` serialization or the MLIR compile-result scheme). |
| `time_logger.py` | `TimeLogger` (verbosity-gated timing), `CUDAEvent` (GPU event pair with CUDASIM fallback), `TimingEvent`, `default_timelogger`. |
| `result_codes.py` | `CUBIE_RESULT_CODES(IntFlag)` — the package-central status vocabulary OR-combined into the per-run status word — plus `decode_status_codes` for host-side decoding. |
| `array_interpolator.py` | `ArrayInterpolator(CUDAFactory)`: builds piecewise-polynomial (spline) coefficients from sampled driver arrays and compiles `evaluate_all` (Horner evaluation of all drivers at `t`) and `evaluate_time_derivative`. Owned by `BatchSolverKernel` as `driver_interpolator` (a direct child factory; `Solver.driver_interpolator` is a passthrough); defines `ArrayInterpolatorConfig`, `InterpolatorCache`. The sample spacing is `driver_sample_period` (input-dict key and config field) — never `dt`, which is the integrator timestep. |
| `writing_cuda_functions.md` | Working notes on CUDA device-function *optimisation* conventions (predicated commit, warp-coherent loops, …). Under discussion — consult before hand-optimising device code. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `batchsolving/` | High-level batch integration API: `Solver`, `solve_ivp`, `BatchSolverKernel`, grid building, system interface, result containers, host/device array managers (see `batchsolving/AGENTS.md`). |
| `integrators/` | Numerical integration components: `SingleIntegratorRun`, algorithm step factories, step controllers, matrix-free solvers, and CUDA loop builders (see `integrators/AGENTS.md`). |
| `memory/` | GPU memory subsystem: `MemoryManager` singleton (`default_memmgr`), array request/response containers, stream groups, CuPy-backed device/pinned allocation (see `memory/AGENTS.md`). |
| `odesystems/` | ODE system definitions and IR-based CUDA code generation (see `odesystems/AGENTS.md`). |
| `outputhandling/` | Output and summary-metric system (see `outputhandling/AGENTS.md`). |
| `gui/` | Optional Qt-based editors for `SymbolicODE` constants/parameters/states (see `gui/AGENTS.md`). |
| `vendored/` | Third-party code vendored as compatibility shims (see `vendored/AGENTS.md`). |

## For AI Agents

This directory is the **compilation spine**. The invariants below are uniform across
the codebase; subpackage `AGENTS.md` files describe only what they *add* and point
back here. CUDA-authoring **optimisation** conventions (predicated commit,
warp-coherent loops, …) live in `writing_cuda_functions.md`.

### CUDAFactory (cached compilation)
- **Subclasses override `build()`** to return a `CUDADispatcherCache` subclass
  instance (a bare callable raises `TypeError`). They **expose compiled device
  functions as named properties** (e.g. `device_function`, `evaluate_f`); callers use
  those properties. `get_cached_output(name)` is the internal plumbing the properties
  use, not the external interface. Never call `build()` directly — storing a
  device-function reference and then updating settings yields a stale reference
  (rebuild is lazy, on next property access).
- **`build()` compiles by closure capture.** A `build()` reads the current
  `compile_settings` (plus registry allocators and child device-function references)
  and bakes those values into the compiled device function as **closure constants** —
  fixed at compile time, not read at call time. This is why any settings change needs a
  rebuild (cache Layer B): the old values are frozen into the old closure. Capturing
  Python scalars/booleans as constants also lets Numba constant-fold and drop dead
  branches — see `writing_cuda_functions.md`.
- **Three cache layers — know which one you are touching:**
  1. **Compiled-kernel cache** (`cubie_cache`), keyed by `config_hash` = each
     factory's `values_hash` re-hashed together with its child factories'. An
     unchanged `config_hash` reuses the on-disk compiled kernel, so the dispatcher
     does not recompile. `BaseODE` folds constant *values* into its `config_hash`.
  2. **Object build cache** (`CUDAFactory._cache` + `_cache_valid`).
     `update_compile_settings` invalidates it **only if a setting actually changed**,
     re-running `build()` on the next property access.
  3. **Codegen source cache** (`odesystems/symbolic`: `ODEFile`),
     keyed by `fn_hash` — equations, ordered array layouts, constants,
     observables, derivative helpers, and function aliases. It caches
     generated CUDA source, separate from compilation.
- **`update` / `update_compile_settings` contract (uniform):** keys are the
  non-underscored field names; raises `KeyError` on an unrecognised key unless
  `silent=True`; returns a **`set`** of recognised/updated labels. The config-level
  `update` is **pure**: it returns a `(replacement, recognised, changed)` triple and
  never mutates the snapshot — `update_compile_settings` is the sole write boundary,
  swapping the replacement in and invalidating the build when **any** field changed.
  Change detection is per-field over post-conversion values: `eq=False` fields
  (derived callables) compare by identity, arrays elementwise, everything else by
  inequality. Hashing and invalidation are different predicates: `values_hash`
  covers only semantic (eq-participating) fields, but a replaced `eq=False`
  callable still rebuilds the consumer. Compile-settings snapshots are deeply
  sealed, not just top-level frozen: direct assignment raises, array-valued
  fields are stored by their converters as owned read-only copies (never
  aliases of caller arrays), and `SystemValues` containers freeze in place at
  the snapshot boundary — structure always; values too for constants, whose
  values are compile-critical. In-place mutation of a held value raises, which
  is what makes memoizing `values_hash` sound. Updates derive a copy
  (`copy()`), modify the copy, and pass it through the boundary.
  A subclass `update` documents **only its additions** over this contract.
- **`config_hash` recurses into child `CUDAFactory` attributes** (direct attributes
  only, discovered alphabetically), so a composite factory invalidates when any
  child's config changes. A subclass may exclude a *diagnostic service* factory
  from discovery by listing its attribute name in `_excluded_child_factories` —
  excluded factories deliberately contribute nothing to semantic identity.
- **`MultipleInstanceCUDAFactory`** maps prefixed external keys (e.g. `krylov_atol`)
  to unprefixed internal fields via `instance_label`; build configs with
  `build_config(...)`.

### Config classes (attrs convention)
- Compile settings are **frozen** attrs classes (`@attrs.frozen`) subclassing
  `CUDAFactoryConfig` / `MultipleInstanceCUDAFactoryConfig`. **Variable- or
  float-typed members are stored underscore-prefixed and exposed, type-coerced,
  through a same-named property**; attrs `__init__` and `update` take the
  **non-underscored** names, so the entire external interface is non-underscored.
  Never pass underscored names; never alias underscored fields.
- Derived fields are recomputed in `__attrs_post_init__` (via
  `object.__setattr__`) so every snapshot — construction or update-derived
  replacement — is self-consistent; converters and validators re-run on every
  replacement, so collections must normalize to canonical immutable values
  (tuples, not lists).
- A system runs at **one precision** (`ALLOWED_PRECISIONS` = float16/32/64); float
  members are returned cast to it via `self.precision(...)`.
- **`eq=False`** marks fields excluded from config equality/hashing (device-fn
  handles, callables; array fields use a custom `eq`) — a replaced value is still
  a change for invalidation. Plain `dict`-typed fields are rejected at
  construction — wrap compile-critical data in its own attrs class. Every
  eq-participating value must be canonically serializable (see `_serialize.py`);
  there is no fallback encoding.

### buffer_registry (CUDA memory layout)
- **Code requirement:** a factory with managed buffers must **register and allocate
  them through the registry** — `register(name, parent, size, location,
  persistent=...)` in `register_buffers()`, then `get_allocator(name, self)` /
  `get_child_allocators(parent, child, name)` for device-side allocation; sizes via
  the `*_buffer_size` properties. Locations: `'local'` (thread registers / persistent
  local) vs `'shared'` (block shared memory). The shared/persistent carve-out and
  buffer **aliasing** are registry-internal. `register_child(parent, child, name)`
  registers a child's buffer footprint with its parent and records the ownership edge
  (`get_child_allocators` calls it before returning allocators), and `clear_parent`
  cascades through recorded children — this is how hot-swap paths drop a replaced
  component's whole chain, so registering children through `register_child` /
  `get_child_allocators` is what keeps swap cleanup working. Both take an optional
  `aliases=` naming a shared entry the child's window overlaps; the caller owns the
  lifetime argument, and every re-registration must repeat it. A buffer's
  `dtype` defaults to the parent's run precision; buffers that differ pass
  `dtype=` (e.g. `np_int32` counters) and receive their shared/persistent
  slice through a `view` of the parent array.
- **Docs requirement:** a child `AGENTS.md` just **lists the buffers it registers**;
  it does not re-describe the registry mechanics or aliasing.

### Array sizing (`ArraySizingClass`)
Host-side array shapes are computed by small attrs helpers subclassing `ArraySizingClass`
(defined in `outputhandling/output_sizes.py`, also used by `batchsolving`). Each exposes a
**`.nonzero`** property returning a copy with every int/tuple dimension floored to a minimum
of 1 — call it before allocating host or device buffers to avoid zero-length allocations.

### Device-code conventions
- **Import every CUDA symbol from `cuda_simsafe`** — never import a CUDA
  backend package (`numba.cuda`, `numba_cuda_mlir`) directly, and **never set
  `NUMBA_ENABLE_CUDASIM` in source.**
- **`# no cover` on device functions:** coverage cannot see inside compiled
  `@cuda.jit` code, so device-function bodies/closures are wrapped with
  `# no cover: start` / `# no cover: end` (and `# pragma: no cover` where
  appropriate). Keep these brackets when editing device code.
- **Import aliasing:** import NumPy scalar types with an `np_` prefix
  (`from numpy import float32 as np_float32`) to disambiguate them from the
  same-named numba types. Prefer explicit symbol imports over `import numpy as np`.
- **Optimisation strategies** (predicated commit, warp-coherent loop exits, …) are
  under discussion in `writing_cuda_functions.md` — consult it before hand-optimising
  device code.

### Testing
See the repo-root `AGENTS.md` for the canonical simulator vs real-GPU commands, markers, and the
full-suite approval gate. Never `xfail`, `importorskip`, or otherwise conditionally skip
behaviour; use the shared `tests/conftest.py` fixtures rather than mocking cubie objects.

### Root-file gotchas
- **`cubie_cache` depends on numba-cuda internals** (`_Kernel`, `IndexDataCacheFile`,
  `CUDACache`) and may break across numba-cuda versions; under CUDASIM it uses the
  vendored `CUDACache`.
- **Timing is no-op by default:** `default_timelogger` starts at `verbosity=None`;
  enable via `solve_ivp(time_logging_level=...)` / `Solver(time_logging_level=...)`.

## Dependencies
### Internal
This root infrastructure is depended on by every subpackage. Within the root, the
dependency order is roughly `cuda_simsafe` ← `_utils` ← `buffer_registry`,
`CUDAFactory`; `cubie_cache` depends on `CUDAFactory`, `_utils`, `cuda_simsafe`,
`time_logger`, `vendored.numba_cuda_cache`, and `cache_root`. All three disk
cache layers (codegen source, CellML parse, compiled kernels) resolve their
base directory through `cache_root.get_cache_root()`; `set_cache_root()`
relocates them together.

### External
- **numba / numba-cuda** — CUDA JIT, device intrinsics, cache internals.
- **numpy** (`>=2.0`) — dtypes, array hashing/comparison, validators.
- **attrs** — all config/data containers.
- **sympy** — parses string and user-supplied symbolic input.
- Optional: **cupy** (memory pool, via `memory/`), **qtpy + a Qt backend** (`gui/`).
