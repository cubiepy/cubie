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

## Containers
Each manager owns a `host` and a `device` container of the same type; arrays match by
field name. `_iter_managed_arrays` iterates both (device then host);
`container.iter_managed_arrays()` iterates one. Containers are `@define(slots=False)` and
find their `ManagedArray`s by scanning `__dict__`; `attach` warns on an unknown label.
Host slot types record the attached array's backing (`pinned`/`host`/`memmap`); device
slots are `"device"`.

## ManagedArray and chunking
- A `ManagedArray` starts with a `np_zeros` backing; an output host slot is `None` after
  a loan or a device-only run until the next transfer backs it.
- Chunking is along the `"run"` axis (`stride_order.index("run")`). Arrays without
  `"run"` or with `is_chunked=False` (`driver_coefficients`) are never chunked;
  `needs_chunked_transfer` is true only when `shape` differs from `chunked_shape`.
- `status_codes` is `int32`/`("run",)`; `iteration_counters` is
  `int32`/`(time, variable, run)`. Float arrays take the solver precision in
  `update_from_solver`; integer arrays keep their dtype.

## Lifecycle
- `from_solver(...)` builds a manager with sizes only.
- `InputArrays.update(...)` refreshes sizes, precision and run count, attaches host
  arrays verbatim with their backing recorded (the only copy is a dtype cast; same
  shape queues `_needs_overwrite`, a new shape also queues reallocation) and calls
  `allocate()`.
- `OutputArrays.update(solver, transfer_outputs)` refreshes the sizes; when they
  changed it drops host buffers of another shape and queues every device output.
- `allocate()` queues `ArrayRequest`s shaped by `_request_shape(label)` and drops the
  device reference of every requested slot. The memory manager then calls
  `_on_allocation_complete(response)`: attach device arrays, record
  `chunked_shape`/`chunk_length`/`num_chunks`, then run `_after_allocation`.
  `_invalidate_hook` drops device references and queues everything again.
- Output host buffers exist only for transferring runs:
  `OutputArrays._ensure_host_arrays` (from `_after_allocation` and `finalise(0)`) keeps
  a buffer only with the sized shape and dtype and a backing the partition accepts
  (pinned only unchunked, pageable only when the policy would not pin it, memmap
  always).
- Device-array inputs (`cuda_simsafe.is_device_array`) go to `_attach_device_inputs`:
  exact shape and dtype or raise, attached as the kernel-facing array, tracked in
  `_device_inputs`, with no allocation or H2D. A slot's own device buffer supplied back
  queues nothing; a slot that reverts to host input is queued for reallocation.
  `BatchSolverKernel._execute_run` enforces single-chunk for device inputs.
  `has_device_inputs` reports the state; the `initial_values`/`parameters` properties
  return the attached device array.

## Per-chunk hooks
- `initialise(chunk_index)` (pre-launch, `InputArrays`): H2D for the queued slots.
  Pinned contiguous sources transfer directly; everything else stages block by block
  through `ChunkBufferPool` buffers, each handed to the transfer watcher with its own
  event.
- `finalise(chunk_index)` (post-launch, `OutputArrays`): D2H for the enabled outputs;
  pinned slots transfer directly, the rest stage block by block through the
  `WritebackWatcher`, which copies each block into the host target and releases it.
- Neither hook blocks the host on the stream. The pool's depth, RAM headroom and pinned
  budget pace the pipeline, so chunk N+1 stages while kernel N runs.
- `staging_blocks(device_array, host_array, budget)` cuts blocks along the leading axis
  (into rows when over budget), each a contiguous device region within
  `HOST_STAGING_BYTES`.

## Host memory types and loans
Output host arrays are created after the chunk decision with the backing
`choose_host_memory_type` picks: `"memmap"` above the spill threshold, pinned for
unchunked arrays within the cumulative pinned budget, else pageable (staged through the
pool). The grid handler writes inputs directly into buffers chosen by the kernel's host
backing policy. Staging blocks are capped by `HOST_STAGING_BYTES` and charged to the
pinned budget.

`loan_host_arrays(result)` empties every host slot into the returned `SolveResult`.
`reclaim_or_release_loan()` (in `update_from_solver` for a transferring run, and in
`SolveResult.from_solver`) returns a collected owner's buffers to their slots; a live
owner keeps them and the next transfer builds fresh host backing.

## Writeback and teardown
Transfer watchers release pinned buffers once their CUDA event completes; output tasks
also copy the staged data into the result arrays. Every acquired buffer reaches a
release: a failed writeback copy, a failed stager or a failed event query frees the
buffer and stores the error, which `wait_all` raises once the queue drains. Close
drains the watchers, then clears the staging pools and device registrations; a failed
close leaves resources attached for a retry. Finalizers do not capture the manager.

## Sizes
`_sizes` is a `BatchInputSizes`/`BatchOutputSizes`; `update_sizes` raises `TypeError` for
another subtype, and `.nonzero` is applied in `update_from_solver`.
`BatchInputSizes.driver_coefficients` comes from `kernel.coefficients_shape`, so every
dimension is concrete. Under `CUDA_SIMULATION`, pinned and async-writeback paths treat
events as complete.

## Dependencies
### Internal
- `cubie.memory` (`default_memmgr`, `MemoryManager`, `ArrayRequest`/`ArrayResponse`,
  `chunk_buffer_pool.ChunkBufferPool`/`PinnedBuffer`); `cubie.outputhandling.output_sizes`
  (`ArraySizingClass`, `BatchInputSizes`, `BatchOutputSizes`); `cubie.batchsolving`
  (`ArrayTypes`); `cubie.batchsolving.writeback_watcher` (`WritebackWatcher`);
  `cubie.cuda_simsafe` (`DeviceNDArrayBase`, `CUDA_SIMULATION`); `cubie._utils` (validators).
### External
- `numpy`; `attrs`; `numba.cuda` (events in `OutputArrays.finalise`).
