<!-- Parent: ../AGENTS.md -->

# memory

## Purpose
GPU memory management for CuBIE. `MemoryManager` is a proportion-based VRAM allocator that
registers caller instances, enforces per-instance memory caps, chunks batches along the run
axis when they exceed available VRAM, and routes host/device transfers to the right CUDA
stream. It runs as a process-wide instance (`default_memmgr`, created in `__init__.py` at
import) — a singleton **by convention**, not enforced via `__new__`. The device's
stream-ordered memory pool (`cudaMallocAsync`, through CuPy's `malloc_async`) is the single
device allocation provider on a real GPU, plugged into Numba as an External Memory Manager
(`cupy_emm.py`), so `cuda.device_array` returns **native** `DeviceNDArray` objects backed by
stream-ordered allocations. Every pinned host array, chunk staging buffers included, is
carved from the manager's `PinnedArena` (`pinned_arena.py`). The CUDA simulator never
touches CuPy — it keeps its own numpy-backed fakes. Supporting pieces: `StreamGroups` (CUDA
stream grouping), `ArrayRequest`/`ArrayResponse` (allocation metadata), `ChunkBufferPool`
(reusable pinned staging buffers).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Installs the device-pool EMM (`install_async_emm()`, before any CUDA context exists), instantiates `default_memmgr = MemoryManager()`; re-exports `MemoryManager`, `NoCudaDeviceError`, `current_cupy_stream`, `CuPyAsyncNumbaManager`. |
| `cupy_emm.py` | `CuPyAsyncNumbaManager` — Numba EMM plugin drawing device memory from the device's stream-ordered pool with a release threshold of zero; `install_async_emm()`. |
| `pinned_arena.py` | `PinnedArena` — page-locked slabs sub-allocated into host arrays of any shape; `PINNED_ALIGNMENT_BYTES`, `SLAB_GRANULE_BYTES`, `MIN_SLAB_BYTES`. |
| `mem_manager.py` | `MemoryManager` (central allocator); `NoCudaDeviceError`; `InstanceMemorySettings` (per-instance registry entry); `ALL_MEMORY_MANAGER_PARAMETERS`; `MIN_AUTOPOOL_SIZE`; `current_cupy_stream` (Numba→CuPy stream forwarding). |
| `array_requests.py` | `ArrayRequest` (shape/dtype/placement spec) and `ArrayResponse` (allocated arrays + chunk metadata). |
| `stream_groups.py` | `StreamGroups` — maps instance ids to named groups, each backed by a CUDA stream. |
| `chunk_buffer_pool.py` | `PinnedBuffer` + `ChunkBufferPool` — reusable pinned staging buffers. Not exported from `__init__.py`. |

## Registration and pools
- `register(instance, proportion=None, invalidate_cache_hook=…, allocation_ready_hook=…,
  stream_group="default")` once per object. `proportion=None` joins the auto pool (an
  equal share of the remaining VRAM); a float takes a manual pool. `MIN_AUTOPOOL_SIZE`
  reserves part of VRAM for the auto pool; a manual proportion that would crowd it raises
  `ValueError` when auto instances exist, else warns. `proportion(instance)` is the
  fraction the instance may use now; `manual_proportion(instance)` the fraction given at
  registration (`None` for the auto pool).
- The registry is keyed by `id(instance)`: keep a live reference to every registered
  object, or a new object can claim its slot.
- Limit mode (`set_limit_mode()`): `"passive"` (default) computes caps without enforcing
  them; `"active"` enforces per-instance caps.

## No device
- `probe_device()` reads `totalmem`. A device-absence failure (`CudaSupportError`, the
  unpacking `ValueError`) stores its error and leaves `totalmem` and `pinned_max_bytes`
  `None`; other failures propagate.
- Sizing decisions (`pinned_budget_bytes`, `allocate_pinned_array`,
  `get_available_memory`, `get_chunk_parameters`, a pinned choice in
  `choose_host_memory_type`) reprobe an unsized manager, then raise `NoCudaDeviceError`
  chained to the probe error. Disk and pageable choices need no device.
- `register` needs a group stream, so a driverless process fails in `stream_groups`.
  `_cap_bytes` is the only place a proportion becomes bytes. The precompile plugin
  patches `get_memory_info` after `import cubie` and calls `probe_device()` before
  registering.

## Deregistration and eviction
- Registry allocations keep device arrays alive until deregistration.
  `release_instance` removes one exact registry entry (an identity check guards against
  reused ids). Freed device blocks leave the pool at the next sync of their stream;
  `BatchSolverKernel.close` syncs its stream after releasing its arrays, then calls
  `trim_pinned_pool`.
- Explicit close reports cleanup failures and can be retried; finalizers are best
  effort and silent at interpreter shutdown.
- Allocation, copies, launch and release use the run's stream; memory caps chunk the
  batch without device-wide synchronization or garbage collection.
- Physical pressure evicts whole idle owners, oldest first, once their completion event
  (recorded by `end_work`) has fired; owners with work in flight are never evicted.
  Every registration is a candidate, so `queue_request` rejects an instance registered
  without a live `invalidate_cache_hook`. Evicted owners reallocate on their next solve.

## Host backing
- `choose_host_memory_type(nbytes, allow_pinned)`: memmap above `HOST_SPILL_FRACTION` of
  RAM, pinned up to `pinned_max_bytes` (default: total VRAM), else pageable.
- `allocate_pinned_array` takes the best-fitting free extent of any arena slab; a
  collected array and its views return the extent. A new slab is page-locked only when
  nothing fits, within `pinned_budget_bytes` = `min(pinned_max_bytes,
  HOST_SPILL_FRACTION × total RAM)` of slab bytes and within `host_headroom_bytes()`
  (`force` ignores both). A `cudaHostAlloc` that runs out of RAM can keep its partial
  commit and make later pinned allocations fail with `cudaErrorAlreadyMapped`. Idle slabs are
  freed before a new slab, by `trim_pinned_pool`, and by `retire_idle_pinned` (called
  after each host-result solve's sync) once idle at two calls running, only while every
  group stream is idle; `flush_pinned_pool` frees them unconditionally. Freeing a slab waits for the
  whole device and stalls launches on every thread.
  `pinned_live_bytes`/`pinned_reserved_bytes` report the arena.
- `create_host_array` allocates the requested type; a `"pinned"` request the budget or
  the driver refuses lands pageable; `"memmap"` arrays land in the cache root. Pageable
  and memmap transfers stage through the pinned staging pool, charged to the same budget
  (the first buffer per label may exceed it). Spill settings live on the solver kernel.
- Chunk parameters are cached per `(stream group, owner)`. Partial reallocations reuse
  the cached partition; a full reallocation of the owner's registrations picks a new
  one. A cached partition is reused only when it covers the batch exactly
  (`partition_covers`) and is dropped when its owner deregisters.
- `change_stream_group(instance, group)` moves every registration of the instance's
  owner, their queued requests and the cached partition.

## Allocation provider
The device's stream-ordered pool is the only device allocator, reached through the EMM
plugin; take `cupy`/`cupyx` from `cubie.cuda_simsafe`. The pool's release threshold is
zero. An out-of-memory allocation syncs the current stream and retries once.
`get_memory_info` reports device free memory plus the pool's reserved but unused bytes.
`allocate()` routes `"device"` requests through `cuda.device_array` inside
`current_cupy_stream` and `"pinned"` requests through `allocate_pinned_array`; any other
placement raises `ValueError`. `to_device`/
`from_device` issue streamed copies between pinned host buffers and native device
arrays; device arrays must be allocated through `allocate_queue` first.

## Queued and chunked allocation
- Each participating instance calls `queue_request(instance, {label:
  ArrayRequest(...)})`, then one `allocate_queue(triggering_instance)`. The manager sizes
  chunks across the queued requests of the trigger's owner in its stream group and calls
  each of those instances' `allocation_ready_hook(ArrayResponse)`; other owners' requests
  stay queued.
- Instances in the group and owner with nothing queued still get the hook, with an empty
  `arr` and the chunk parameters. With nothing queued for the owner, `allocate_queue`
  calls no hook; callers keep their last partition.
- Chunking replaces `shape[chunk_axis_index]` with `chunk_length`; `unchunkable=True`
  keeps the full shape.
- `get_chunk_parameters` offers `min((1 − CHUNK_HEADROOM_FRACTION) × available, physical
  free − allocation_granule_bytes)` bytes; `num_chunks` is what the largest fitting chunk
  needs and `chunk_length = ceil(runs / num_chunks)`. Tests faking `get_memory_info` at
  byte scale pass `allocation_granule_bytes=0`.
- `allocate_all` releases the entries a request replaces before allocating.
- `ArrayRequest.dtype` must be exactly `float64`/`float32`/`int32` and `memory` one of
  `device`/`pinned`; `chunk_axis_index` defaults to `2` (the run axis of the 3-D output
  layout); `total_runs ≥ 1` sizes the chunks. `ArrayResponse` carries `arr`, `chunks`,
  `chunk_length`, `chunked_shapes`.

## Stream groups and CuPy streams
- Groups map instance ids to a shared `cuda.stream()`. The `"default"` group is created
  on its first registration. `reinit_streams()` replaces every group's stream.
  `add_instance`/`get_group`/`get_stream`/`change_group` take an `int` id or an object.
- `current_cupy_stream` forwards a Numba stream into CuPy via
  `cupy.cuda.Stream.from_external`; Numba's default stream (handle `0`) stays CuPy's
  current stream. Allocation and release enter it; transfers use the Numba stream.

## ChunkBufferPool
Pinned staging buffers keyed by `array_name`. `acquire` returns the smallest idle buffer
that fits, its `array` viewed in the requested shape and dtype, replacing an idle buffer
too small; it grows while fewer than `STAGING_POOL_DEPTH` of the label are in flight and
headroom and budget allow (a label with nothing in flight always gets one), else blocks
until a release; this bound paces the pipeline. `release` frees a buffer and wakes waiters; `clear` frees all (use on error
paths). Buffers are charged to the pinned ledger.

## Dependencies
### Internal
- `cubie.cuda_simsafe` (`Stream`, `DeviceNDArrayBase`, `CUDA_SIMULATION`, `current_mem_info`);
  `cubie._utils` (validators in `array_requests.py`).
### External
- `numba`/`numba.cuda` (context/stream management, kernel launch, pinned arrays, driver
  copies); `attrs`; `numpy`; `cupy` (required on a real GPU — its async pool backs all device
  allocation through the EMM plugin, imported once through `cubie.cuda_simsafe`, which
  supplies `None` stand-ins and numpy-backed pinned-allocation equivalents under the CUDA
  simulator).
