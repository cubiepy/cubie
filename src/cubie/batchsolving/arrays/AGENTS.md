<!-- Parent: ../AGENTS.md -->

# arrays

## Purpose
Host/device array coordination for batch solves. Owns the NumPy host arrays and Numba CUDA
device arrays the batch kernel reads and writes, brokering their sizing, allocation,
host↔device transfer, and run-axis chunking through the shared `MemoryManager`.
`BatchSolverKernel` owns one `InputArrays` and one `OutputArrays` (both `BaseArrayManager`
subclasses); each holds a `host` and a `device` `ArrayContainer` of `ManagedArray` metadata
wrappers — one per logical array, matched across containers by field name.

See `CUDAFactory` (root) for the `ArraySizingClass`/`.nonzero` pattern and attrs conventions.
Allocation, streams, and chunk math live in `cubie.memory`.

## Key Files
| File | Description |
|------|-------------|
| `BaseArrayManager.py` | `ManagedArray` (per-array metadata: dtype, `stride_order`, shapes, chunk axis/length, backing array), `ArrayContainer` (ABC), `BaseArrayManager` (ABC: registration, size/dtype checks, host updates, allocation, chunk-aware transfer, pinned↔numpy conversion). |
| `BatchInputArrays.py` | `InputArrayContainer` (`initial_values`, `parameters`, `driver_coefficients`) + `InputArrays` — sizes from `BatchInputSizes`; `initialise` stages H2D. |
| `BatchOutputArrays.py` | `OutputArrayContainer` (`state`, `observables`, `state_summaries`, `observable_summaries`, `status_codes`, `iteration_counters`) + `OutputArrays` — sizes from `BatchOutputSizes`; `finalise` does D2H + async writeback. |
| `__init__.py` | Empty — managers are imported from their modules. |

## For AI Agents

### Two containers per manager
Each manager owns a `host` and a `device` container of the same type; arrays are matched by
field name. Iterate both via `_iter_managed_arrays` (device then host), one container via
`container.iter_managed_arrays()`. Host slot types describe the attached array's actual
backing (`pinned`/`host`/`memmap`); device is `"device"`.

### ManagedArray & chunking
- A `ManagedArray` always holds a real backing array — `__attrs_post_init__` allocates a
  `np_zeros` default, so `.array` is never `None`.
- Chunking is always along the `"run"` axis: the chunk axis index is `stride_order.index("run")`.
  Arrays without `"run"` in `stride_order` or with `is_chunked=False` (e.g.
  `driver_coefficients`) are never chunked; `needs_chunked_transfer` is true only when the full
  `shape` differs from `chunked_shape`.
- `status_codes` is `int32`/`("run",)`; `iteration_counters` is `int32`/`(time,variable,run)`
  default `(1,4,1)`. Float arrays get their dtype rebound to the solver precision in
  `update_from_solver`; integer arrays keep their dtype.

### Lifecycle
`from_solver(...)` builds a manager with sizes only (no allocation). `update(...)` refreshes
sizes/precision/run-count, sets host arrays (`update_host_arrays` — incoming arrays are
attached **verbatim** with their actual backing recorded on the slot; the only copy is a
dtype cast; same-shape attaches queue `_needs_overwrite`, shape changes also queue
reallocation), and calls `allocate()`, which queues `ArrayRequest`s with the memory manager.
The memory manager later drives `_on_allocation_complete(response)`: attach device arrays,
record `chunked_shape`/`chunk_length`/`num_chunks`, set `_chunks`, and re-back
kernel-written output slots (repin small non-chunked buffers, pageable for chunked — fresh
allocations, never copies, since the kernel overwrites them; input managers override both
conversions as no-ops). `_invalidate_hook` drops device refs and re-marks everything for
reallocation.

`InputArrays.update` detects device-array inputs (`cuda_simsafe.is_device_array`) and
routes them to `_attach_device_inputs`: validated (exact shape vs `_sizes`, exact dtype —
raise, never coerce) and attached directly as the kernel-facing device array, tracked in
`_device_inputs`, removed from `_needs_reallocation`/`_needs_overwrite` so no buffer is
allocated and no H2D runs. A slot that later reverts to host input is re-queued for
reallocation. Device inputs are single-chunk only, guarded in `BatchSolverKernel._execute_run` — the
only place that can: an attached slot queues no allocation, so `InputArrays` never learns
the run's chunk count. `has_device_inputs` exposes the state, and
the `initial_values`/`parameters` properties return the caller's device array while one is
attached.

### Teardown
Explicit close drains transfer watchers before clearing staging pools and
device registrations. Failures leave resources attached so close can be
retried. Finalizers use cleanup calls that do not capture the manager.

### Per-chunk hooks (called by `BatchSolverKernel.run` around each launch)
- `initialise(chunk_index)` — pre-launch. `InputArrays`: H2D. Pinned contiguous sources
  transfer directly; everything else (pageable, memmap, or any chunk slice) stages
  block-by-block through `ChunkBufferPool` pinned buffers, each handed to the transfer
  watcher with its own event. `OutputArrays`: no-op.
- `finalise(chunk_index)` — post-launch. `OutputArrays`: D2H for the outputs the compile
  flags enable (placeholder arrays are skipped); pinned slots transfer directly, everything
  else stages block-by-block, each block submitted to the `WritebackWatcher` with its own
  event for the trimmed copy into the host target and buffer release.
- Neither hook ever blocks the host on the stream: pacing comes from the pool's
  RAM-headroom bound, so the CPU stages chunk N+1 while kernel N runs and writebacks of
  chunk N drain during kernel N+1.

### Memory types
Output host arrays are created pageable (or `"memmap"` above the spill
threshold); after the chunk decision, non-chunked arrays at or below
the manager's `pinned_max_bytes` (default: total VRAM) are re-backed
pinned (fresh, no copy) for direct async transfer, when RAM headroom
allows. Input slots record the attached array's actual backing; the
grid handler materialises assembled inputs through the manager's host
backing policy, so eligible arrays transfer directly. Staging blocks
are capped by `HOST_STAGING_BYTES`. Full-size pinned allocations
above the ceiling never happen.

### Result buffer loans
After a solve, `loan_host_arrays(result)` empties every host slot into
the returned `SolveResult`. `reclaim_or_release_loan()` runs before
the next allocation and in `SolveResult.from_solver`: a collected
owner's buffers return to their slots (with their memory types and
size signature) for reuse; a live owner keeps them and fresh backing
is allocated.

### Async writeback
Transfer watchers release pinned buffers after their CUDA event completes.
Output tasks also copy staged data into the result arrays. Shutdown drains all
tasks before it clears the pool.

### Container mechanics
Containers use `@define(slots=False)` and discover their arrays by scanning `__dict__` for
`ManagedArray` instances (`_iter_field_items`), so fields are picked up dynamically. `attach`
warns (doesn't raise) on an unknown label.

### Sizes
`_sizes` is a `BatchInputSizes`/`BatchOutputSizes` (`ArraySizingClass`); `update_sizes` raises
`TypeError` if the replacement isn't the same subtype. `.nonzero` (floor empty/disabled dims to
1) is applied to `_sizes` before allocation, in `update_from_solver`. All dims are concrete:
`BatchInputSizes.driver_coefficients` comes from `kernel.driver_coefficients_shape` — a
`BatchSolverConfig` compile setting the Solver keeps aligned with
`ArrayInterpolator.coefficients_shape` — so no `None` wildcards exist in the sizing scheme. See root for the `ArraySizingClass`/`.nonzero` pattern.

### Testing
`tests/batchsolving/arrays/`. Pinned/async-writeback paths short-circuit under `CUDA_SIMULATION`
(events treated as complete). Prefer building via `InputArrays.from_solver`/
`OutputArrays.from_solver` against a `BatchSolverKernel` fixture.

## Dependencies
### Internal
- `cubie.memory` (`default_memmgr`, `MemoryManager`, `ArrayRequest`/`ArrayResponse`,
  `chunk_buffer_pool.ChunkBufferPool`/`PinnedBuffer`); `cubie.outputhandling.output_sizes`
  (`ArraySizingClass`, `BatchInputSizes`, `BatchOutputSizes`); `cubie.batchsolving`
  (`ArrayTypes`); `cubie.batchsolving.writeback_watcher` (`WritebackWatcher`);
  `cubie.cuda_simsafe` (`DeviceNDArrayBase`, `CUDA_SIMULATION`); `cubie._utils` (validators).
### External
- `numpy`; `attrs`; `numba.cuda` (events in `OutputArrays.finalise`).
