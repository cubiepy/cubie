# backend/

## Purpose
Cubie's interference with its CUDA backends: compatibility shims
applied at `import cubie`, MLIR lowering registrations for cubie
device utilities, and the typed-IR block scheduler. Nothing here is
public API.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Docstring only; importing the package has no side effects. |
| `_mlir_compat.py` | numba-cuda-mlir compatibility shims, imported first thing from `cubie/__init__` on the MLIR backend: empty-slice anchoring, dynamic-shared-memory and array-literal fixes, semantic local stack slots, float min/max semantics, compiler-frontend perf patches, `consteval` AST transforms on inlined device-function callees (the inliner otherwise consumes the untransformed `py_func`), and `register_typed_block_scheduler` (registers `TypedBlockScheduler` when the wheel carries the typed-planner hook; warns and no-ops on hookless wheels with a non-source `CUBIE_BLOCK_SCHEDULE`). Each shim feature-detects patched builds and no-ops there. |
| `_numba_cuda_compat.py` | Compile-time performance and lineinfo patches for stock numba-cuda (no-op on the `cubie_patch` fork, under CUDASIM, and for patches already upstream), plus a numpy 2.5 `row_stack` stand-in. |
| `_mlir_intrinsics.py` | MLIR-backend typing and lowering for cubie device utilities: `narrow_f64` (float64→float32 narrowing without subnormal flush). Imported by `cuda_simsafe` on the MLIR backend. |
| `_block_schedule_policies.py` | Ordering policies for the typed-IR block scheduler (`ScheduleNode`, `order_nodes`, `modeled_peak`) — pure graph computations with no CUDA backend imports, unit-testable everywhere. |
| `_typed_block_scheduler.py` | `TypedBlockScheduler` builds a per-block dependency DAG (flow edges, name chains, per-element memory chains, barriers, Del pins) over the fully inlined typed Numba IR and reorders each block under the selected policy. Registers through `numba_cuda_mlir.extending.register_typed_planner` and declares `cache_safe = True`; the active policy folds into the kernel-cache fingerprint. Imports the backend at module import; only `_mlir_compat` imports it, after feature detection. |

## For AI Agents
- The compat modules run their patches at import time; `cubie/__init__`
  imports them before anything can compile a kernel. Never import them
  from anywhere else.
- Coverage omits `_mlir_compat.py` and `_numba_cuda_compat.py`
  (`pyproject.toml`).
- Scheduler knobs: `CUBIE_BLOCK_SCHEDULE` (policy; documented in
  `cubie/_env.py`), `CUBIE_BLOCK_SCHEDULE_DUMP` (gzip graph dump),
  `CUBIE_BLOCK_SCHEDULE_ORDER` (JSON orders for the `inject` policy).
  The active policy enters the kernel-cache ABI fingerprint via
  `cubie._env.active_block_schedule`.
- `unroll_if(range(n), flag[, count])` loops resolve in `_mlir_compat._UnrollIfPass` to `cuda.unroll(range(n))`, `cuda.unroll(range(n), count)` or `cuda.nounroll(range(n))` from the closure `flag` (bool or `(unroll, count)` pair; an explicit `count` wins while unrolling), bound as `_cubie_unroll`/`_cubie_nounroll` globals via `TransformContext.stored_values`; body `consteval(...)` reading the loop variable is stripped; a wheel without the hints raises at transform time. Identity fallback in `cuda_simsafe.unroll_if`.
