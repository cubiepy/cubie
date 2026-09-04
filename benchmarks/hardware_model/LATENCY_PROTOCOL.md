# Direct-address dependent-load protocol

This separate instrument measures the RTX 4070 SUPER's dependent load
service under recorded launch and cache conditions. It does not change
CuBIE, its backend, or the frozen earlier hardware probes. Preparation
is CPU-only; native compilation and measurement are explicit commands.

The earlier global probe executes descriptor stride/offset arithmetic
between every load, and its final clock has no last-load dependency.
Those concrete causes are documented in DEPENDENCY_EVIDENCE.md. Here one
side-effecting MLIR inline-PTX primitive contains initialization, priming,
the timed chain, a dependent endpoint guard, and direct output stores.
The installed `cubie-numba-cuda-mlir` wheel is the compiler being tested.

## Addresses and controlled work

The global ring stores uint64 **device addresses** and executes repeated
`ld.global.ca.u64 p,[p]` or `ld.global.cg.u64 p,[p]`. These are integer
pointers, not FP64 arithmetic. Shared memory stores uint32 addresses in
its own address space. Its host ring contains byte offsets; a parallel
initialization adds the actual shared symbol address once per entry.
Repeated `ld.shared.u32 p,[p]` then needs no shift, multiply or descriptor.
Every thread participates in the shared initialization and barrier;
only thread zero performs priming, timing and output in either space.

One randomized, single-cycle topology places one pointer in each
32-byte sector. The complete offset array and cycle order are retained
and checked. Global blocks share one read-only device ring. Shared
blocks have separate instances. This controls byte footprint and avoids
silently treating an 8-byte payload as a 32-byte cache request. Payload,
sector traffic and instruction counts remain distinct quantities.

A full ring traversal precedes the first clock, within each block's
active lane. A branch dependent on its final loaded pointer must resolve
before the starting clock; issue order alone does not prove completion.
Its one-load counted loop is independently checked against the native
node-count parameter. The measured body contains 33 or 257 direct loads, followed
by one runtime decrement, comparison and backedge. These body sizes
allow nonfull-cycle N/2N controls for the capacity-derived rings. The
repeat count exceeds one full ring traversal and is selected so both measured final pointers differ from
the starting pointer and from each other. Exactly doubling repeats
doubles the measured load count; it does not change body bytes, binary,
priming count, initialization or geometry. Every raw ring and final
pointer is validated, including pointer rebasing in fresh profile runs.
Starting offsets use evenly spaced positions in the exact randomized
cycle for each CTA, with their raw array saved and verified. Each sample
records unique visited nodes, full cycles, remainder loads and traversed
sector bytes per active thread. The grid still shares the global ring;
phase offsets reduce synchronized access without asserting independence
between SMs or eliminating shared L2 reuse.

Before the ending clock, a predicate consumes the final loaded pointer's
low word. All valid pointer addresses are aligned, so `0xffffffff` is an
invalid low word. The clock must be predicated on that result or occur
after the corresponding dependent forward branch. Native admission
rejects a clock hoisted ahead of this dependency. The comparison and
branch still contribute endpoint cost; they are not subtracted.

The native gate requires exactly two clock reads and one repeated chain
between them. Every direct load's address words must equal the preceding
load's result words, including the final-to-first loop-carried edge.
Register renaming is retained as an explicit per-PC witness. Shared
loads consume one uint32 register; global loads consume a uint64 pair.
The observed global loop carries that pair through two scalar copies
before its first load. Admission proves the low/high correspondence
from the final loaded pair to the first address pair, checks both copy
definitions dominate that load, and rejects clobbers or indexed operands.
One common load opcode and explicit decrement/test/backedge dataflow are
required. The counter may be scheduled between loads; its destination
must be disjoint from every pointer word. The observed global entry also
contains one unpredicated `YIELD` and one `ULDC.64 UR4,c[0][0x118]`.
Only that exact four-instruction entry inventory is admitted, entirely
before the first load. Its constant-bank read remains separate from
pointer-load work; neither it nor YIELD receives a zero-cost assumption.
All body instructions dominate the loop exit, excluding side
entries that skip a load or its counter administration.
`cuobjdump --dump-elf` must report the emitted five-scalar parameter
ABI: three uint64 addresses at byte offsets 0, 8 and 16, followed by
uint32 repeat/node counts at offsets 24 and 28. The total is 32 bytes.
The reaching native count definition must resolve through simple
copies to the runtime-count parameter's constant-bank offset. The exact
reachable control graph must prove these definitions dominate their
uses. A counter definition must be an explicit scalar destination;
the high word of a packed parameter load cannot stand for its low word.
The start clock must preserve the initial pointer and counter, and its
timestamp pair must survive every load and administrative instruction.
The end clock preserves that timestamp and the final pointer. All
64-bit implicit high-word writes are included. Shared loads require the
native 32-bit `LDS` form. No indexed address or descriptor work is allowed.

A conditional backedge or the observed terminal exit form is admitted:
an inverted count predicate controls `CALL.REL.NOINC` to the immediately
following tail, followed by an unconditional backedge on continuation.
Every target-tail path must move forward to one unconditional `EXIT`,
with no call, return, indirect edge, or path back to the measured body.
The resolved target is modeled as a terminal branch only after that
proof. This admits no arbitrary function call. Both clocks retain their
final-pointer-dependent guards, including the compiler's equivalent
NE comparison with inverted failure branch. An extra scalar repeat-count
copy is accepted only if it reaches the same ABI parameter and has one
post-clock use in the emitted `repeats * body_loads` output calculation.
The check follows that scalar value until its next write; the multiply
may reuse its register for the resulting uint64 product.
Different unproved forms fail and retain source/PTX/cubin/SASS before
any target launch.
Counter and endpoint proofs require independent inspection of the first
actual compiled artifact; source construction is not native validation.

The retained first shared8 compile is
`cubie-notes/hardware_unroll_placement/latency_shared8_compile_e1`.
It has 257 dependent `LDS` instructions at PCs 0x210 through 0x1230,
with the decrement/test at 0x220/0x230, tail exit at 0x1240, backedge at
0x1250, and ending pointer guard at 0x1260. R8 closes the recurrence;
R4/R5 hold the preserved starting clock. Its original failed admission
record is immutable; a separate reviewed reparse is not a GPU launch.

The retained global `.ca` compile is
`cubie-notes/hardware_unroll_placement/latency_l1_quarter_compile_e1`.
Its loop has 257 `LDG.E.64.STRONG.SM` instructions and eight administrative
instructions, for 265 total. R14/R15 carry the final pointer; entry copies
at 0x1c0/0x1e0 map them to R22/R23 before the first load at 0x200.
The 256 internal edges directly connect each prior result to the next
address. Priming has the analogous R14/R15 to R6/R7 transport, one
YIELD, one identical uniform constant-bank load, a count decrement/test,
one pointer load and one backedge. Its eight instructions are outside
the timed interval. R6/R7 hold the starting clock during the measured
loop; its final pointer guard precedes the ending R10/R11 clock read.
The original failed gate and source remain preserved. These concrete
entry costs distinguish this native chain from an intrinsic LDG latency
measurement; the N/2N and body-size controls retain them explicitly.

## Hardware-derived footprints and geometry

Preparation reads an existing hardware-query manifest, then the native
worker re-queries those capacity inputs. Ada's documented unified
L1/texture/shared pool is 128 KiB. Global windows are one quarter and
twice that nominal pool, and one quarter and twice the queried device
L2 capacity. On this device these are 32 KiB, 256 KiB, 12 MiB and 96 MiB.
They are candidate working sets, not asserted cache-residency classes.
Shared rings use 8, 16 and 32 KiB static windows drawn from supported
capacities, each below the queried static per-block limit. Actual shared
configuration also accommodates driver reservation and remains separately
profile-qualified. [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#unified-shared-memory-l1-texture-cache).

The block uses the queried maximum 1,024 threads. The queried maximum
1,536 resident threads/SM excludes a second such block by integer
capacity, independently of shared carveout. The driver occupancy query
must agree on one block. Grid size is at least twice SM count times
resident blocks, giving at least two full occupancy waves. This is
32 allocated resident warps/SM, with one warp executing one lane's
timed chain while other lanes wait at the barrier. This is not an
achieved active-warp occupancy count. SMIDs are recorded; no contiguous numbering
or physical cache topology is inferred. No carveout attribute is set.

Inactive lanes wait at a final uniform CTA barrier after the timed
interval. The active lane also reaches it after its outputs or the
invalid-pointer path. Native admission requires this barrier to dominate
every reachable exit, keeping all allocated warps resident while timing.
The observed `BAR.SYNC.DEFER_BLOCKING` is accepted as the retained
source's full-CTA `bar.sync 0` lowering; every BSSY token is paired with
its BSYNC join immediately before a barrier. No separate scheduling or
timing meaning is assigned to the undocumented native modifier.
The global lowering's unpredicated `WARPSYNC 0xffffffff` is accepted
only immediately before that final CTA barrier. It is a full-mask
warp join, not a substitute for the CTA residency barrier.
[PTX barrier synchronization semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
require waiting for participating warps; the retained source uses neither
`bar.arrive` nor a reduced participant count.
Early warp retirement is not assumed to preserve the thread occupancy
bound. NVIDIA describes release granularity as unspecified in this
[block scheduling clarification](https://forums.developer.nvidia.com/t/scheduling-of-blocks-does-every-thread-of-a-block-need-to-finish-before-a-new-block-launches/274531).

Actual LaunchStats must independently confirm block/grid dimensions,
resident resource limits, static/dynamic shared bytes, actual shared
configuration and waves. The native thread-capacity proof avoids the
carveout-dependent occupancy-search error discovered in another probe;
it does not replace the actual launch qualification.

## Ordinary timing and matched profiles

The first ordinary launch calibrates the repeat count. The timed chain
itself must reach 20 ms at the maximum before/after observed SM clock,
and CUDA event elapsed time must also reach 20 ms. The reader recomputes
duration from each validated raw cycle array, requires finite positive
event/clock values and matching GPU UUIDs with explicit MHz units, and
rejects a different retained duration. Clock snapshots are
not a continuous frequency trace; raw cycles are primary. This prevents
long untimed priming from satisfying the duration gate on its own.
Two measurement blocks collect six N and six 2N samples each, alternating
their order. All calibration and measurement outputs, events, clocks,
native identities and inputs remain in the bank. Profile events never
replace ordinary timing samples.

A matched profile runs exactly one N or 2N launch from a completed
ordinary bank, using its chosen repeat count. It requires exact source,
topology, installed compiler, native binary, resources, geometry, raw
ordinary sample membership, nonfull outputs and successful worker exit.
Device addresses may change across processes: the complete uint64 ring
must equal the retained byte-offset topology plus the newly recorded
device allocation base. Other input changes fail. The isolated worker
changes no device attributes; normal process exit releases its context
and allocations and is part of ordinary-bank admission.

Collect LaunchStats, Occupancy, SourceCounters and instruction counts.
For global cases collect global-load L1 sectors and lookup misses,
L2 read sectors and lookup misses, and DRAM read bytes. Keep warm-loop
PCs separate from timed-load PCs in SourceCounters. Priming contributes
to hardware aggregate traffic; its exact unchanged work must be retained
when comparing N and 2N. For shared cases collect shared-load instructions,
wavefronts and bank conflicts. All metric names/units must be queried
on AD104 before capture, and raw reports must be joined to this exact
source snapshot, cubin and one-launch output before interpretation.

PTX `.ca` and `.cg` are cache hints; `.cg` requests bypassing L1.
They do not certify measured hit rates. [PTX cache operators, Table 30](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators).
No latency is named for a cache level until actual hit/miss and capacity
counters qualify that path. Raw per-thread intervals include loop and
endpoint administration plus scheduling. N/2N differences and the
33/257-body contrast expose those contributions without assigning a
cycle cost to auxiliary instructions or fitting any solver timing.

## Initial discriminating sequence

1. Compile-only the 257-load shared8 and 32 KiB global `.ca` cases.
   Independently inspect direct address, native count and endpoint proofs.
2. Run ordinary N/2N controls and matched profiles for those two cases.
   Repeat with 33-load bodies if the measured administration needs a
   bound before reporting the dependent service interval.
3. Compare 32 KiB `.ca` against `.cg` with matched actual counters, then
   measure 12 MiB `.cg`. Use 256 KiB `.ca` and 96 MiB `.cg` only to
   discriminate the next cache-service transition indicated by those
   measurements. All footprints derive from capacities, not fitted
   thresholds. Larger rings retain full priming cost outside the clocks.

Example CPU preparation (use the original frozen-source PYTHONPATH):

```text
python benchmarks/hardware_model/latency_probe.py
  --hardware-manifest <rawroot>/icache_contrast_20260904/manifest.json
  --footprint shared8 --body-loads 257 --out <fresh-directory>
```

Add `--compile-only` for the serialized native gate, or `--execute` for
ordinary sampling after review. A profile instead adds
`--profile-multiplier 1` (or `2`) and `--ordinary-dir <accepted-bank>`.
All output directories are fresh. Nothing automatically invokes Nsight.

## External paper context

[Luo et al., Table 3](https://arxiv.org/html/2501.12084v2#S4.T3)
reports RTX 4090 L1 32.0 cycles, shared 30.1 and L2 273.0. Its §4.1
prose instead says approximately 284.8 for that L2 result. Table 5 also
distinguishes FP32 and wider-access throughput. Those values and the
internal L2 discrepancy are comparison context, not RTX 4070 SUPER
constants. This instrument measures this device and instruction width.
