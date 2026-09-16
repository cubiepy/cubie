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
| `BaseArrayManager.py` | `ManagedArray` (per-array metadata: dtype, `stride_order`, shapes, chunk axis/length, backing array), `ArrayContainer` (ABC), `BaseArrayManager` (ABC: registration, size/dtype checks, host updates, allocation, chunk-aware transfer). |
| `BatchInputArrays.py` | `InputArrayContainer` (`initial_values`, `parameters`, `driver_coefficients`) + `InputArrays` — sizes from `BatchInputSizes`; `initialise` stages H2D for the queued slots, skipping zero-size host data; `update` with `driver_coefficients=None` leaves the attached table in place. |
| `BatchOutputArrays.py` | `OutputArrayContainer` (`state`, `observables`, `state_summaries`, `observable_summaries`, `status_codes`, `iteration_counters`) + `OutputArrays` — sizes from `BatchOutputSizes`; `finalise` does D2H + async writeback. |
| `__init__.py` | Empty — managers are imported from their modules. |

## For AI Agents

### Two containers per manager
Each manager owns a `host` and a `device` container of the same type; arrays are matched by
field name. Iterate both via `_iter_managed_arrays` (device then host), one container via
`container.iter_managed_arrays()`. Host slot types describe the attached array's actual
backing (`pinned`/`host`/`memmap`); device is `"device"`.

### ManagedArray & chunking
- A `ManagedArray` starts with a real backing array (`__attrs_post_init__` allocates a
  `np_zeros` default); an output host slot is `None` after a loan or a device-only run
  until the next transfer backs it.
- Chunking is always along the `"run"` axis: the chunk axis index is `stride_order.index("run")`.
  Arrays without `"run"` in `stride_order` or with `is_chunked=False` (e.g.
  `driver_coefficients`) are never chunked; `needs_chunked_transfer` is true only when the full
  `shape` differs from `chunked_shape`.
- `status_codes` is `int32`/`("run",)`; `iteration_counters` is `int32`/`(time,variable,run)`
  default `(1,4,1)`. Float arrays get their dtype rebound to the solver precision in
  `update_from_solver`; integer arrays keep their dtype.

### Lifecycle
`from_solver(...)` builds a manager with sizes only (no allocation). `InputArrays.update(...)`
refreshes sizes/precision/run-count, sets host arrays (`update_host_arrays` — incoming arrays
are attached **verbatim** with their actual backing recorded on the slot; the only copy is a
dtype cast; same-shape attaches queue `_needs_overwrite`, shape changes also queue
reallocation), and calls `allocate()`. `OutputArrays.update(solver, transfer_outputs)`
refreshes the sizes and, when they changed, drops host buffers of another shape and queues
every device output for reallocation. `allocate()` queues `ArrayRequest`s with the memory
manager, shaped by `_request_shape(label)` (the host array for inputs, `_sizes` for outputs),
and drops the device reference of every requested slot. The memory manager later drives
`_on_allocation_complete(response)`: attach device arrays, record
`chunked_shape`/`chunk_length`/`num_chunks`, set `_chunks`, then the `_after_allocation`
hook. `_invalidate_hook` drops device refs and re-marks everything for reallocation.

Output host buffers exist only for runs that transfer: `OutputArrays._ensure_host_arrays`
runs from `_after_allocation` when the run transfers and from `finalise(0)`, and keeps a
buffer only when it has the sized shape and dtype and a backing the partition accepts
(pinned only unchunked, pageable only when the policy would not pin it, memmap always).
A device-only run leaves every output host slot `None`.

`InputArrays.update` detects device-array inputs (`cuda_simsafe.is_device_array`); a slot's
own device buffer supplied back queues nothing, and any other device array is
routed to `_attach_device_inputs`: validated (exact shape vs `_sizes`, exact dtype —
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
- `initialise(chunk_index)` — pre-launch. `InputArrays`: H2D for the queued slots (zero-size sources skipped). Pinned contiguous sources
  transfer directly; everything else (pageable, memmap, or any chunk slice) stages
  block-by-block through `ChunkBufferPool` pinned buffers, each handed to the transfer
  watcher with its own event. `OutputArrays`: no-op.
- `finalise(chunk_index)` — post-launch. `OutputArrays`: D2H for the outputs the compile
  flags enable (placeholder arrays are skipped); pinned slots transfer directly, everything
  else stages block-by-block, each block submitted to the `WritebackWatcher` with its own
  event for the trimmed copy into the host target and buffer release.
- Neither hook ever blocks the host on the stream: pacing comes from the pool's
  depth, RAM-headroom, and pinned-budget bounds, so the CPU stages chunk N+1 while kernel
  N runs and writebacks of chunk N drain during kernel N+1.
- `staging_blocks(device_array, host_array, budget)` cuts both stagers' blocks along the
  leading axis, descending into rows over budget, so every block is a contiguous device
  region within `HOST_STAGING_BYTES`; blocks stop at the shorter host extent on every axis.

### Memory types
Output host arrays are created after the chunk decision with the
backing `choose_host_memory_type` picks: `"memmap"` above the spill
threshold, pinned for unchunked arrays the cumulative pinned budget
grants, else pageable, which stages through the pool. Input slots
record the attached array's actual backing; the grid handler
assembles inputs directly into buffers chosen by the kernel's
registered host backing policy. Staging blocks are capped by
`HOST_STAGING_BYTES` and charged to the same budget.

### Result buffer loans
After a solve, `loan_host_arrays(result)` empties every host slot into
the returned `SolveResult`. `reclaim_or_release_loan()` runs in
`update_from_solver` and in `SolveResult.from_solver`: a collected
owner's buffers return to their slots (with their memory types) for
reuse; a live owner keeps them and the next transfer builds fresh
host backing while the device outputs stay allocated.

### Async writeback
Transfer watchers release pinned buffers after their CUDA event completes.
Output tasks also copy staged data into the result arrays. Shutdown drains all
tasks before it clears the pool.

Every acquired buffer must reach a release. A failing writeback copy is stored
on the watcher and re-raised by `wait_all` once the queue drains. A stager
that raises drains the stream, then releases its buffer. An event whose query
raises retires its task, frees the buffer, and stores the error for
`wait_all`.

### Container mechanics
Containers use `@define(slots=False)` and discover their arrays by scanning `__dict__` for
`ManagedArray` instances (`_iter_field_items`), so fields are picked up dynamically. `attach`
warns (doesn't raise) on an unknown label.

### Sizes
`_sizes` is a `BatchInputSizes`/`BatchOutputSizes` (`ArraySizingClass`); `update_sizes` raises
`TypeError` if the replacement isn't the same subtype. `.nonzero` (floor empty/disabled dims to
1) is applied to `_sizes` before allocation, in `update_from_solver`. All dims are concrete:
`BatchInputSizes.driver_coefficients` comes from `kernel.coefficients_shape` — a
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
