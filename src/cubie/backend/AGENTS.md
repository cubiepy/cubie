# backend/

## Purpose
Cubie's interference with numba-cuda-mlir: compatibility shims
applied at `import cubie`, `cuda.jit` keyword arguments, cubie device
intrinsics and their MLIR lowering, and the typed-IR block scheduler.
Nothing here is public API.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Docstring only; importing the package has no side effects. |
| `utils.py` | Driver and compiled-kernel queries: `compile_kernel_specialization` (compiles a launch's specialization and raises its dynamic shared limit to the opt-in maximum), `device_hardware()`/`DeviceHardware` (SM count, L2 size, shared memory per SM, per-block reserve, opt-in dynamic-shared limit, instruction-cache capacity from `INSTRUCTION_CACHE_BYTES`), `kernel_resources(dispatcher)`, `active_blocks_per_multiprocessor`, `max_shared_memory_per_block`. |
| `_mlir_compat.py` | Imported first thing from `cubie/__init__`. One section per open numba-cuda-mlir pull request cubie uses, each reproducing that branch as merged onto the pinned `cubie-numba-cuda-mlir` wheel and applied unconditionally; then `register_typed_block_scheduler` registers `TypedBlockScheduler` with the wheel's typed-planner hook. |
| `_mlir_cubie_extensions.py` | The `unroll_if` AST pass: loops become `cuda.unroll` hints or plain loops from the closure flag `(unroll, count)`: `True` → `cuda.unroll(range(n))`, `(True, k)` → `cuda.unroll(range(n), k)`, `False` → plain `range(n)`; an explicit `count` argument overrides `k`. Imported from `cubie/__init__` after `_mlir_compat`. |
| `intrinsics.py` | Cubie device intrinsics with numba-cuda-mlir typing and lowering: `narrow_f64` (float64→float32 narrowing without subnormal flush; its Python body is a plain `np.float32` call); `unroll_if(range(n), flag[, count])`, which returns its iterable and is consumed by the `_mlir_cubie_extensions` pass; `UnrollFlag` (the `(unroll, count)` flag type). |
| `jit.py` | `cuda.jit` keyword arguments: `JIT_FLAG_DEFAULTS`, `compile_kwargs` (import-time device functions) and `get_jit_kwargs` (renders a `JITFlags`; the `CUDAFactory.jit_kwargs` property is the single sanctioned route to build kwargs). Every set carries `experimental_ast_transforms=True`. |
| `_block_schedule_policies.py` | Ordering policies for the typed-IR block scheduler (`ScheduleNode`, `order_nodes`, `modeled_peak`) — pure graph computations with no CUDA backend imports, unit-testable everywhere. |
| `_typed_block_scheduler.py` | `TypedBlockScheduler` builds a per-block dependency DAG (flow edges, name chains, per-element memory chains, barriers, Del pins) over the fully inlined typed Numba IR and reorders each block under the selected policy. Registers through `numba_cuda_mlir.extending.register_typed_planner` and declares `cache_safe = True`; the active policy folds into the kernel-cache fingerprint. Imports the backend at module import; only `_mlir_compat` imports it. |

