# Conditional instruction delivery

`nominal_instruction_cache.py` adds instruction-fetch readiness to the
finite event scheduler. A request can proceed while its warp waits for
arithmetic operands. Other warps can issue while a fill is pending.
Execution waits for both instruction and operand readiness. There is
no additive instruction-cache penalty after scheduling.

The scheduler requests the next represented PC after the preceding
instruction dispatch. A BRA delays that eligibility until its supplied
branch-completion time. One request is kept for each current instruction;
there is no invented stream-prefetch depth. A contracted two-instruction
store/load motif requests both PCs before its atomic execution reservation.
This is an explicit conservative ordering approximation for that existing
atomic motif abstraction. The whole measured pair service is charged once.

The cache state persists across each actual common-work wave. Cold misses,
pending-line merges, arrived hits, LRU evictions, fill bytes, and request
resource reservations are retained. Pending fills have an explicitly named
unlimited-queue capacity ceiling. Pending lines consume resident capacity
when they arrive. Cache lines are immutable; no dirty-data cost is added.

## Scenario interface

`scenario["instruction_fetch"]` has `mode: "hierarchy"`, an `id`, physical
`provenance`, and an explicit `assumption`. It contains:

- `fetch_policy: "next_pc_demand"` and
  `outstanding_fills: "unlimited_capacity_ceiling"`.
- `levels`, ordered from the instruction front end toward backing memory.
  Each level supplies `name`, `capacity_bytes`, `line_bytes`, four
  `partition_domains`, `request_interval_cycles`, and `path_ready_cycles`.
- `backing`, with four `partition_domains`, `request_interval_cycles`, and
  `path_ready_cycles`.
- `initial_state: "cold"`, or `"explicit_seed_lines"` with `seed_lines`
  containing `level`, `domain`, and integer `line` index.
- `event_pcs`, an exact executable event-ID to synthetic 16-byte PC map.

Cycles are integers or exact rational dictionaries. Readiness is a whole
path from front-end request to available instruction, not one additive
latency per visited cache. The result uses exactly one selected whole-path
service plus waiting for separately reserved request resources. Logical
lookups occur together; stage lookup delays are included in the supplied
whole-path readiness. The model does not claim independent stage timing.
Absent readiness or initiation values return explicit missing-service
requests. Finite resource intervals must be positive. Zero extra L0 hit
readiness is admissible only as an explicitly qualified perfect-hit ceiling.

Each lower-level request reserves that level's initiation resource only
when the upper lookup reaches it. A pending match merges with the existing
fill. Its readiness is the later of the pending completion and the queued
request's minimum front-end hit readiness. A miss installs pending lines
in all missed upper levels. Arrival performs fully associative LRU
insertion. Nested line sizes must be multiples, starting at 16 bytes.

Domain labels are supplied for the four scheduler partitions. Four
different labels represent private instances; repeated labels share an
instance and request resource. A label such as `nominal_gpc` is a declared
sharing hypothesis. It does not assign physical GPC/TPC membership from
an SM number. The scheduler represents one SM's work; a larger shared
domain therefore needs an explicit effective capacity/interference
hypothesis. Assuming other SMs fetch the same PCs coherently is one such
qualified approximation. Independent other-SM traffic is not generated.

The instruction return granule is independent of the source instruction
width. For example, a supplied 256-byte line contains sixteen synthetic
16-byte instruction slots. The requested PC order, visited union, and
reserved address span remain distinct quantities. Source projection does
not identify physical cache set indexing or native compiler alignment.

## Actual candidate binding

`nominal_scenarios.bind_instruction_fetch` binds a verified source graph,
policy-plan wrapper and hierarchy hypothesis. It constructs or checks the
instruction projection by full source reconstruction, retaining graph,
wrapper, typed-plan, allocation
event and projection hashes. Its `projection` settings explicitly choose
`helper_lowering`, `false_lowering`, and `project_recurrent_caps`.
`build_scenario` invokes this binding when `evidence["instruction_fetch"]`
is supplied; `forecast` then schedules the actual candidate normally.

The typed-plan scheduler rejects an instruction mapping bound to another
allocation. Direct `schedule_events` callers supply their own complete
event map. `mode: "disabled"` preserves the prior scheduler's numerical
results exactly. It represents omitted fetch cost, not a hardware claim.

Current service values are caller-supplied alternatives. Target probe
composites require independent miss/sector and cache-state attribution
before being assigned as an instruction service. Published constant-cache
or other-architecture path latencies can be explicit sensitivity transfers;
they are not silently relabeled as measured Ada instruction latency.
No fitted prefetch depth, solver timing regression or per-miss penalty is
introduced by this interface.
