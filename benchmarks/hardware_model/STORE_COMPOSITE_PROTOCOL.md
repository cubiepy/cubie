# Same-cell store/load composite

This separate instrument measures a retained same-thread store/load
sequence. It does not assign standalone store latency or equate memory
completion with a register-result latency. The frozen arithmetic,
latency, hardware and solver sources remain unchanged.

The three source forms are shared with ordinary same-thread ordering,
shared with a CTA memory fence between each pair, and local with ordinary
same-thread ordering. Each has 33- and 257-pair bodies. One binary accepts
either the first complete warp or all 32 warps in a 1,024-thread CTA.
Every thread reaches the final CTA barrier, including inactive and
invalid-result paths. The queried 1,536-thread SM limit excludes a second
1,024-thread block independently of shared carveout; the driver must also
report one resident block. The target is the retained 56-SM SM89 device,
and 112 blocks provide two full occupancy waves. Profiles must confirm
actual resource limits and shared configuration separately.

Each active lane owns one four-byte cell. Shared storage is a 4,096-byte
static window indexed by thread ID. Local storage is one four-byte PTX
local allocation per thread; its actual native frame and address origin
must be retained and independently verified. The two offsets are
different runtime scalar inputs, both zero in every admitted run. Their
native address operands must remain distinct and invariant through the
body. Source constants or array descriptors do not supply a substitute
for that proof.

The actual ABI has **nine scalar parameters**: four uint64 addresses
(output, lane seeds, expected endpoint, memory observations) and five
uint32 scalars (repeat count, selected warp count, reserved zero, write
offset, read offset). The reserved slot is always zero. The emitted
signature, parameter offsets and complete native parameter extent are
retained. It is not a CUDA array-descriptor ABI.

Lane seeds are unsigned words 1 through 32, repeated across warps. A
poison value `0xa5a5a5a5` is stored and read before timing. That actual
read is saved, and participates with seed, expected endpoint, count and
both offsets in a pre-clock checksum guard. Every pair stores `x`, then
loads it from the same cell into the next store's value. One integer
increment follows each complete body. After N bodies the endpoint is
`seed + N (mod 2^32)` and the last stored memory value is `seed + N - 1`.
Both are checked, as are the initial poison and inactive sentinels.
Odd N and 2N have distinct endpoints. The payload is a 32-bit memory word;
no FP64 arithmetic is introduced.

The final timed load, increment and expected-value guard must precede the
end clock. Output has eight uint64 words per lane: begin clock, end clock,
difference, incremented endpoint, initial SMID, final SMID, success code
and timed pair count. A separate two-word uint32 record holds the actual
poison read and final observed value. In the first shared native artifact,
the compiler removes the redundant source-level post-clock load and the
observation store consumes the unclobbered result of the last timed load.
Admission binds that store to the final trace address. This proves the
last loaded value, but it is not an independent post-clock reread.

For B pairs per body and R bodies, each active lane executes BR timed
stores, BR timed loads, R increments, and (for the fence form) BR fences.
The admitted shared native form also has one initialization store and one
initialization read per active lane. It therefore has B+1 total native
stores and B+1 total native loads; the removed observation load is not
counted. The exact final timed-load result must stay live through the end
clock into the single scalar observation store. These counts must be
joined to exact native PCs,
software SourceCounters and hardware counters. Preserved instructions do
not imply that hardware performed an equal number of physical memory
transactions.

PTX defines ordering for overlapping same-thread accesses. Volatile
instructions have additional compiler constraints, but hardware may
merge their memory operations; this probe does not use volatile as proof
of transaction count. The shared fence contrast records its actual
native ordering instruction and cost. It establishes no local-memory
drain, cache flush or standalone completion time.
[PTX ISA 9.3, §§8.4.2, 8.7 and 8.10.5](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).

The first release has two native-admission stages. The automated gate
proves exact scalar32 pair/fence cardinality, load-to-next-store edges,
loop-carried increment, invariant distinct address operands, count
origin, timestamp/address liveness, loop control and final CTA retention.
Its full inventory exposes the remaining independent-review obligations:
actual address origins, initial-load completion, final-clock guard,
every output word, the forwarded final observation and complete
convergence-stack paths. A retained
`independent_store_composite_native_review` certificate with PASS status,
reviewer identity, exact source/worker/native hashes, matching admission,
and PC witnesses plus reasoning for every obligation is mandatory before
ordinary or profiled execution. Each witness is a unique canonical
lowercase native address such as `0x1a0`; the loader requires every address
to occur in the exact retained SASS instruction inventory. It is produced
by reviewing the real
compile-only artifacts. No certificate or native result is fabricated by
CPU preparation, and no unsupported native form is admitted by a loose
opcode list.

Ordinary execution calibrates a common N until both populations meet
20 ms for the CUDA event and minimum in-kernel interval converted with
the retained clock snapshots. Those snapshots are not a continuous
frequency trace. It then records two mirrored blocks, six samples for
each population and N/2N: 48 measured launches. Every output array,
source operand, native identity before/after, event and clock sample is
retained. No profile event time replaces an ordinary sample. A matched
profile executes exactly one chosen population/count against the same
ordinary source, binary, inputs, certificate and geometry.

Example CPU preparation:

```text
python store_composite_probe.py --out <fresh> --space shared --fence none
  --body-operations 257 --hardware-manifest <retained manifest>
```

Add `--compile-only --nvdisasm <fixed tool> --cuobjdump <fixed tool>` for
the first explicitly scheduled native gate. Ordinary mode adds
`--execute --native-certificate <independent review>`. A profile uses
`--profile-multiplier 1|2 --profile-warps 1|32 --ordinary-dir <bank>` and
the same certificate. All output directories must be fresh.
The saved worker command, working directory, child exit code and forced-
cleanup flag are rederived by the ordinary loader.

The intended engine input is a conditional store-to-load RAW edge
service for this precise sequence and population. Store issue, operand
consumption, same-cell RAW readiness, external visibility and final
kernel completion are separate events. A composite edge must not also
charge the same load latency twice. Unconsumed final stores do not acquire
a fabricated load-like completion delay. This instrument does not modify
the NativePlan event engine or fit any solver timing.
