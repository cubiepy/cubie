# ERK NativePlan CPU predictions

These are predictions from explicit source/lowering/allocation scenarios,
not native measurements. The implementation and qualifications are in
[NATIVE_PLAN.md](NATIVE_PLAN.md). Independent review clears this bounded
ERK component for diagnostic native comparisons. It does not approve a
complete heuristic or predict whole-kernel runtime. The final release
receipt is
`verification/native_plan_independent_semantics_v2_20260905/final_freeze.json`,
SHA256 `0049228ce394ddd1b1a82bee37131939a862ff8b0a37884c9342620b8bce9b0e`.
The fourteen predictions remain frozen before native holdout labels.

Raw root:
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.
The final author receipt is
`verification/native_plan_cpu_v4_20260905/receipt.json`, SHA256
`c6fa73cc3f6c83fa615aa3b4d019059ae8e13110c5b63a0ec736364a895ae69e`.
It binds estimator SHA256
`f547ee91e5f3a390d68c8113e8eb438bde03438935ca8d4b294e148fb9480471`
and every prediction's exact JSON bytes. Construction inputs are under
`native_plan_cpu_v3`; the ten isolated workers all completed with zero
native overloads and separate generated-source caches.

The hardware inputs come from the existing
`instruction_sharing_ordinary_e2/manifest.json` query. No earlier sharing
timing, occupancy-query conclusion, or profiler interpretation is used.
The model applies its own conditional resource equations to those device
attributes. All rows use a fixed 64-thread modeled block.

| Source / scenario | Modeled no-spill words | Allocated peak words | Modeled spill bytes | Emitted modeled instructions per warp-step |
|---|---:|---:|---:|---:|
| Lorenz RK4, local promoted | 26 | 26 | 0 | 139 |
| Lorenz RK4, local addressable | 11 | 11 | 0 | 370 |
| Lorenz RK4, promoted, explicit budget 8 | 26 | 8 | 52 | 210 |
| Lorenz RK4, local promoted, optional contraction | 26 | 26 | 0 | 85 |
| Lorenz RK4, actual shared accumulator, other local cells promoted | 21 | 21 | 0 | 262 |
| chain32 Vern7, local promoted | 463 | 255 | 812 | 11,970 |
| chain32 Vern7, local addressable | 76 | 76 | 0 | 26,614 |
| chain32 Vern7, actual shared accumulator, other local cells promoted | 205 | 205 | 0 | 20,726 |
| chain16 Vern7, local promoted | 239 | 239 | 0 | 5,062 |
| chain17 Vern7, local promoted | 253 | 253 | 0 | 5,365 |
| chain18 Vern7, local promoted | 267 | 255 | 28 | 5,711 |
| chain16 Vern7, actual shared accumulator, other local cells promoted | 109 | 109 | 0 | 10,390 |
| chain17 Vern7, actual shared accumulator, other local cells promoted | 115 | 115 | 0 | 11,017 |
| chain18 Vern7, actual shared accumulator, other local cells promoted | 121 | 121 | 0 | 11,682 |

The table does not turn promotion into a user setting or equate a modeled
word count with reported native registers. The word demand includes the
specified constant/base materialization and instruction schedule. General
compiler CSE, different instruction forms, rematerialization and outer
caller values can change it. The optional contraction row is a separate
compiler scenario under the same actual enabled flag.

Every row passed replay of register identities, spill slots, named-cell
versions, source dependency order and final observable values. Four
uncontracted Lorenz rows additionally reproduced every emitted arithmetic
or load result bit for bit against the original FP32 source graph,
including the forced-spill trace. The chain reciprocal approximation and
optional fused arithmetic have no source-bit-equality claim.

Fifteen deliberately corrupted allocation traces were rejected, including
missing entry/exit values, empty exit locations, reloads without memory,
wrong reloaded value, invalid spill schemas and inconsistent extents.
The independent review's original three failing cases remain preserved
alongside the corrected receipts. Synthetic service values
exercised the event scheduler and scenario-separated ranking API; those
values are explicitly algorithm fixtures and are not attached to any
physical prediction. The saved physical predictions retain missing
service symbols and have no point estimate of elapsed kernel time.

The local-backing alternatives are retained separately. For example,
the forced-spill Lorenz trace has identical read/write sector demand in
both mappings, while the modeled write misses are 2,496 with resident-slot
reuse and 4,992 with disjoint later-wave backing. These counts cover one
modeled SM, two waves and one step per warp under the declared LRU/write
policy. They demonstrate mapping sensitivity, not measured cache misses.
Counts from candidates with different resident-warps counts must be
normalized to a common warp-step workload before comparison.

## Source-selected dimension holdouts

The fresh chain16/17/18 graphs contain 14,211/15,068/15,985 source nodes.
Their no-spill NativePlan word demands are 239/253/267. Applying the
hardware register-allocation quantum changes the allocated equivalent
from 240 words per thread at chain16 to 256 at chain17. The modeled
255-word limit then introduces seven four-byte spill slots at chain18.
All three have the same conditional four-block register-residency limit
at 64 threads, so these are allocation/spill boundaries, not an asserted
occupancy cliff.

Matched shared candidates were also constructed before native labels.
Their actual padded per-run strides are 580/612/652 bytes; the fixed
64-thread model allocates 38,144/40,192/42,752 shared bytes per block,
including the queried reservation and allocation quantum. Each has a
conditional two-block limit under the nominal 100 KiB carveout. The
public launch path would reduce these requested blocks, so an exact
64-thread native contrast requires reviewed pinning. These predictions
must not be joined to a 32-thread measurement as though geometry matched.

The proposed fresh native contrasts are chain16 versus chain17 for the
allocation quantum, and chain17 versus chain18 for modeled spill onset,
each with the same actual unroll flags and a matched local/shared
placement comparison. Their dimensions were selected from the generated
graphs and hardware equations before reading native or ordinary labels
for these configurations. Existing banks informed the mechanisms and are
not presented as untouched holdouts.

Native checks must preserve compiler/source identity and actual geometry,
including any public launch-size reduction. They should test lowering
categories, materialized arrays, registers and local traffic before
judging ordinary timing. A disagreement diagnoses the explicit compiler
or storage assumptions; it does not introduce a fitted register factor
or timing penalty. Source construction and CPU replay are not the
independent review or GPU validation gate.
