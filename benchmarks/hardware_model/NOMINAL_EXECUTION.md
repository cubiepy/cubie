# Conditional nominal finite execution

`nominal_execution.py` estimates a declared allocated event region using
the physical catalog and explicit scenario inputs. It schedules the
actual typed trace, including spills/reloads, and produces exact rational
wave cycles and cycles for common useful work. It is CPU-only and does
not consume compiled native labels, measured iteration counts or solver
timings. It does not predict omitted caller work. An explicit
`instruction_fetch` hierarchy gates instruction readiness using the
source-PC interface in `NOMINAL_INSTRUCTION_CACHE.md`; without that
scenario, instruction delivery cost is omitted.

The catalog's cross-architecture and instruction-form transfers remain
estimates. Supplying a service resolves a scheduling input; it does not
certify that service for an entire native kernel. The output binds the
events, catalog, scenario, and optional actual plan by normalized JSON
hash. These hashes identify inputs, not a new admission certificate.

## API and scenario inputs

`schedule_plan(plan, catalog, scenario, resident_warps, work, ...)` takes
an actual `conditional_typed_implicit_native_plan` and uses its allocated
events. `schedule_events(...)` accepts those events directly, including
the `release` and `free_home` bookkeeping entries. Those entries consume
no instruction service. Spill/reload offsets are already in the complete
local frame; they are normalized into explicit local-memory events.

The required scenario fields are `id` and `full_warp_coherent: true`.
The latter is an explicit workload assumption. Known nonfull active or
participating masks in events are rejected. The source allocation model
and semantic path remain the caller's responsibility.

Cycle values accept integers, fraction strings, or
`{"numerator": n, "denominator": d}`. Floats are rejected to keep
hand calculations and sensitivity comparisons exact. Service cycle
values must be positive. Unknown services never acquire a zero cost.

`service_overrides[opcode]` applies to that form; an
`event_overrides[str(event_id)]` record can refine a specific event.
Every override carries `provenance` and `assumption`. Supported fields
are:

| Field | Meaning |
|---|---|
| `result_latency_cycles` | Issue to available register result |
| `source_consumed_cycles` | Qualified load/other input-overwrite delay |
| `warp_block_cycles` | BRA issue to next eligible instruction in its warp |
| `store_source_consumed_cycles` | Issue to safe overwrite of store inputs |
| `store_visible_cycles` | Issue to readiness for a following aliasing access |
| `store_complete_cycles` | Issue to completion used for draining this region |
| `resource` | Explicit execution-port hypothesis |
| `initiation_cycles` | Aggregate SM reservation interval for that service |

An isolated store requires all three store fields from catalog/scenario,
ordered as input
consumption <= visibility <= completion. It has no register-result
latency. Loads use their selected complete path latency. A local L2-hit
scenario overrides the LDL result latency with 273 or 284.8 from the
catalog; it does not add either to the L1 latency.

VOTE.ALL receives the qualified epoch-2 motif estimate. Other VOTE forms
and ACTIVEMASK require result latency; BRA requires warp blocking.
Their undefined values are returned as precise service requests.
CAPTURED_LOOKUP requests a concrete allocated native expansion. The
caller must lower it and account for any new temporaries before sending
the trace; a single lookup latency cannot represent an arbitrary array
selection sequence.

Memory events require a qualified `memory_issue` record:

```json
{
  "kind": "scalar_full_warp_wavefronts",
  "default_wavefronts": 1,
  "event_wavefronts": {},
  "provenance": ["Address-layout or explicitly named issue hypothesis"],
  "assumption": "Conflict-free scalar access under this layout"
}
```

The optional event map overrides wavefront count by event ID. Demand is
the catalog's full-warp scalar interval times `bytes/4` times wavefronts.
This is an explicit issue estimate, not a claim that a masked access
costs active-lanes/32 of an instruction. Larger-width scalar requests
scale by words; vector instruction throughput requires a distinct form.

## Scheduling and capacity

Each warp stays on `warp_id % 4`. Each partition has one dispatch slot
per cycle, a 16-lane dedicated FP32 route, and a 16-lane route shared by
FP32 and integer-like work. A full-warp FP32 operation reserves either
route for two cycles. Integer-like work uses only the shared route,
normally for two cycles. This implements the published Ada capacity
coupling as a nominal routing model; it is not a claim about undocumented
per-instruction routing or exact half-warp execution order.

The earliest jointly available dispatch/resource reservation issues
first. Warp ID breaks ties; the dedicated FP32 route wins an equal-time
route tie. This is a deterministic greedy schedule, not an optimal
compiler schedule. It preserves in-order issue within each warp while
allowing independent results and other warps to overlap execution.

SFU, vote and LSU retain aggregate SM reservation intervals from the
catalog. Fractional SM issue times are a continuous capacity convention;
per-partition dispatch still reserves a full cycle. This simplification
does not assert cross-partition migration of a warp or exact physical
issue-clock phase.

Every resource reservation begins at the selected issue time. The next
instruction must respect register RAW and WAW completion, and WAR input
consumption. Input consumption uses its sourced catalog/scenario delay where
available, including load addresses. Other instruction inputs retain the
explicit capture-at-issue assumption. Stores keep their own input delay. Final wave time
includes outstanding results, declared store completion and reserved
resources. It does not add those overlapping durations together.

## Memory aliases and motifs

Aliases use actual byte intervals within the local/shared frame, not
logical cell labels. Two names with overlapping offsets alias. Disjoint
bytes do not. Read/read accesses can overlap; read-after-write waits for
visibility, write-after-read conservatively waits for the prior load
result, and write-after-write preserves visibility order.

By default, each warp owns distinct per-thread frame addresses. Explicit
`cross_warp_alias: true` shared accesses use block ownership and require
`warps_per_block`. Shared synchronization is not synthesized: cross-warp
history follows the chosen issue interleaving, so a source needing a
barrier must already have its synchronization represented in the input
region or remain outside this scheduler's supported semantic scope.
Thread-local frames cannot alias across warps. Initial memory and live-in
registers are assumed ready at region entry.

A `store_load_motifs` entry may contract adjacent STS/LDS or STL/LDL
instructions accessing exactly the same byte interval:

```json
{
  "event_ids": [10, 11],
  "pair_cycles": "7465/257",
  "store_complete_cycles": "EXPLICIT_SCENARIO_VALUE",
  "issue_model": "noninterleaved_reservation",
  "provenance": ["Exact matching probe and completion hypothesis"],
  "assumption": "Recorded same-cell motif transfer"
}
```

The completion placeholder above must be replaced by a positive sourced
or explicitly qualified scenario value. Pair readback does not identify
downstream store drain.

The macro reserves both LSU demands and both dispatches. The second
issue offset is `max(1, first_LSU_interval)`. It reserves its partition's
dispatch through that offset plus one, and LSU through the offset plus
the second LSU interval. Blocking the intermediate dispatch gap is an
explicit noninterleaving approximation. It can overestimate contention;
it does not pretend to know arbitrary-gap store/load readiness.

The pair's result is available after `pair_cycles`. The measured interval
includes recurring native administration, so individual load latency,
standalone store latency and probe administration are not added. Store
inputs remain protected through the pair result as a conservative
readback bound. The pair visibility bound is also that result time;
final completion is the maximum of pair result and declared store drain.
Other instructions in the caller's trace still retain their own costs;
the contraction contains only the two selected memory instructions and
the administration represented by the transferred pair interval.

## Common work and validation

Work must specify `kind: synchronized_full_waves` and
`warp_attempts_per_sm`. Each candidate must contain complete waves, with
at least two waves. For example, nine cycles at four resident warps and
96 common warp attempts gives 24 waves and 216 cycles. Comparing raw
resident-wave time alone would compare different quantities of work.
The caller must also align family, iteration regime and useful work
identity across candidates; the scheduler preserves those plan records.

The external author audit uses positive hand-computed fixtures for
dependent chains, independent FP32 routes, integer contention, register
reuse, store input hold, exact and partial memory aliases, independent
loads, contracted pair cost and common work. Synthetic fixture services
are labeled as logic checks and never added to the physical catalog.

An actual Kvaerno3/LU allocated prefix has a 153-cycle nominal wave at
four resident warps under catalog services. This is a prefix diagnostic,
not a whole-solver estimate. The complete actual plan returns requests
for its unresolved non-ALL collective/control and store services. A separately labeled
synthetic scenario verifies that the entire allocated trace can execute
once those fields are specified; its timing is not a hardware forecast.

The CLI reads a JSON request containing `plan` or `events`, `catalog`,
`scenario`, `resident_warps`, `common_work`, and optional
`warps_per_block`/`include_trace`. It writes a new output file, preserving
prior evidence. No CUDA import, compilation or GPU launch is performed.


## Immutable indexed LDC

Epoch 2 accepts only 32-bit read events with `kind: immutable_constant`,
`space: constant`, a bound `table_id`, affine address description,
`offset_is_execution_witness: true`, and
`broadcast_regime: uniform_indices_over_declared_active_warp`.
The lowerer supplies the actual IMAD address-register dependency and
independently derives index uniformity. The witness offset describes the
chosen source regime; it does not replace dynamic address computation.

A qualified `constant_cache` scenario with
`kind: immutable_broadcast_hit` is required. LDC is excluded from mutable
local/shared byte-alias history and produces no implicit data-cache
traffic. It reserves a `constant_broadcast:<partition>` resource for one
cycle alongside dispatch. This is a scheduler-only resource ceiling,
not measured constant-cache bandwidth. Miss paths or divergent indices
need a different explicit service model.

The indexed hit result and address-input hold are both 29 cycles under
an Ampere-to-Ada transfer. A 26-cycle Turing result sensitivity leaves
input hold at 29 unless a separate form hypothesis changes it. Region
completion respects both result availability and input consumption.
The generic input-hold field also represents LDS9 and named LDL11
address-space-transfer scenarios; isolated store defaults gain STS12 and
named STL14 source-hold scenarios, with visibility/drain still explicit.
# Optional source-local cache state

The `data_cache` scenario enables committed-issue cache path selection and
persistent state across actual synchronized waves. See
`NOMINAL_DATA_CACHE.md` for capacity, fill readiness and traffic assumptions.
Without that optional scenario, the existing stationary-wave calculation
is unchanged. Cache-enabled outputs report separate wave schedules and
their summed common-work cost.
