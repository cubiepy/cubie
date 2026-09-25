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
| `utils.py` | Driver and compiled-kernel queries with one form on every backend and a CUDASIM stand-in: `compile_kernel_specialization` (compiles a launch's specialization and raises its dynamic shared limit to the opt-in maximum), `device_hardware()`/`DeviceHardware` (SM count, L2 size, shared memory per SM, per-block reserve, opt-in dynamic-shared limit, instruction-cache capacity from `INSTRUCTION_CACHE_BYTES`), `kernel_resources(dispatcher)`, `active_blocks_per_multiprocessor`, `max_shared_memory_per_block`. |
| `_mlir_compat.py` | Imported first thing from `cubie/__init__` on the MLIR backend. One section per open numba-cuda-mlir pull request cubie uses, each reproducing that branch as merged onto the pinned `cubie-numba-cuda-mlir` wheel and applied unconditionally; then `register_typed_block_scheduler` registers `TypedBlockScheduler` with the wheel's typed-planner hook. |
| `_numba_cuda_compat.py` | Compile-time performance and lineinfo patches for stock numba-cuda (no-op on the `cubie_patch` fork, under CUDASIM, and for patches already upstream), plus a numpy 2.5 `row_stack` stand-in. |
| `_mlir_cubie_extensions.py` | The `unroll_if` AST pass: loops become `cuda.unroll` hints or plain loops from the closure flag `(unroll, count)`: `True` → `cuda.unroll(range(n))`, `(True, k)` → `cuda.unroll(range(n), k)`, `False` → plain `range(n)`; an explicit `count` argument overrides `k`. Imported from `cubie/__init__` after `_mlir_compat`. |
| `_mlir_intrinsics.py` | MLIR-backend typing and lowering for cubie device utilities: `narrow_f64` (float64→float32 narrowing without subnormal flush). Imported by `cuda_simsafe` on the MLIR backend. |
| `_block_schedule_policies.py` | Ordering policies for the typed-IR block scheduler (`ScheduleNode`, `order_nodes`, `modeled_peak`) — pure graph computations with no CUDA backend imports, unit-testable everywhere. |
| `_typed_block_scheduler.py` | `TypedBlockScheduler` builds a per-block dependency DAG (flow edges, name chains, per-element memory chains, barriers, Del pins) over the fully inlined typed Numba IR and reorders each block under the selected policy. Registers through `numba_cuda_mlir.extending.register_typed_planner` and declares `cache_safe = True`; the active policy folds into the kernel-cache fingerprint. Imports the backend at module import; only `_mlir_compat` imports it. |

