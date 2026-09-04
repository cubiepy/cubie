# Same-address reuse versus two instruction streams

This is a bounded proposed experiment, not an implemented probe or a
cache-domain conclusion. The existing 8-warp capacity transition cannot
alone distinguish an individual warp's working set from instruction
reuse among SMs. Two separately addressed streams can change aggregate
instruction demand while preserving each executing warp's body size.

## Primary contrast

Compile one kernel containing two disjoint approximately 80 KiB FFMA
regions with the same arithmetic recurrence, dependency spacing, opcode
mix and executed body length. One runtime mode selects stream A on all
SMs, another selects stream B on all SMs, and a mixed mode selects A/B
using an observed SMID bit. Both uniform modes are required: otherwise
stream alignment or address-index effects could masquerade as sharing.
All modes use the identical cubin and unchanged function attributes.

The selector runs outside the repeated region. Each thread keeps its
selected stream for every repeat. Use the existing observable, bounded
FP32 recurrence and independent-chain structure from the validated
instruction probe. Preserve runtime repeat count instead of flattening
the repetitions into the instruction stream. Static source duplication
is not evidence of two native streams.

First inspect the installed MLIR intrinsic path already used for
`clock64` in `hardware_probes.py:467-479` and a diagnostic `%smid` read.
The compiler may merge identical branches or duplicate helpers. The
compile gate must reject that outcome. Distinct source labels or function
names do not prove distinct SASS address regions. A controlled inline-PTX
body is a possible implementation mechanism only if the resulting native
regions satisfy every equality gate; no differing arithmetic, memory
traffic or padding penalty may be hidden as an anti-merging device.

Use one resident block per SM, established through a legitimate dynamic
shared reservation and driver occupancy queries, with a fixed block size
giving eight warps. Use at least `2*SM_count` blocks. The shared allocation,
actual profiled shared configuration, register allocation and geometry
must be identical in all modes. Grid size alone cannot impose residency.

## Coverage and equal-work requirements

Record block/warp identifiers, selected stream and entry/exit SMID, plus
the complete result array. The observed set must cover the queried
physical SM count and both streams in the mixed arm. Record per-SMID
block and warp counts rather than assuming the scheduler gives each SM
exactly two blocks. Require uniform selection within each warp. Reject
observed migration or incomplete coverage for the domain contrast.

[PTX ISA 10.8 and 10.9](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-smid)
state that SMIDs need not be contiguous, `%nsmid` may exceed the number of
physical SMs, and a thread's SMID can change after preemption. Entry/exit
agreement cannot prove no intermediate migration. Neither SMID parity
nor consecutive identifiers establishes TPC/GPC membership.

Native admission requires all of the following:

- One identical cubin/resource manifest for all runtime modes.
- Two nonoverlapping native body PC ranges, with exact executed FFMA and
  control counts and recurrence dependencies matched; no residual inner
  loop, call-path difference, unexpected memory traffic or merged tail.
- Same registers, block shape, dynamic shared bytes, one-block residency
  and at least two full occupancy waves at the actual compiled geometry.
- Raw results exactly equal across uniform A, uniform B and mixed modes
  for identical FP32 inputs and repeat count.
- Source counter work matches selected stream counts and repeat count;
  each warp repeatedly executes one body rather than both bodies.

Start with those three modes at one repeat count giving at least 20 ms.
Repeat the same three modes at twice that count to distinguish recurring
instruction-fetch pressure from the fixed selector and output cost.
Use mirrored ordinary samples with raw clocks/status/geometry retained.
This is six ordinary conditions, not a size-by-residency search.

## Counter interpretation

Profile one fixed-repeat instance of each mode after the ordinary gates.
Use the reviewed elevated session and preserve exact source/cubin/profile
receipts. Collect ICC lookup hit/miss counters in their exported `cycle`
units and GCC instruction lookup hit/miss counters in `request` units;
these are different denominators. Also retain instruction-related issue
stalls, eligible/active warps, executed instructions and LaunchStats.
Retain counter domains and units:
ICC and GCC names are not evidence that either cache is private per SM.
Profiled event times must not enter the ordinary timing comparison.

If both uniform controls agree but the mixed mode increases GCC misses
and instruction starvation at equal per-warp body/work, the result
supports sensitivity to instruction-address reuse outside an individual
warp. It does not uniquely identify physical cache size, associativity
or the sharing group. A negative result does not prove privacy: the
chosen SMID partition may separate the unobserved sharing groups.

Only if the first mixed assignment is non-discriminating should one
additional observed-SMID partition be considered, retaining the same
binary through a runtime mask. Its purpose would be to test assignment
sensitivity, not to declare a topology from bit positions. If stream
duplication, equal native work or coverage cannot be established, reject
the experiment and retain that concrete limitation. No fitted fetch
penalty, cache-capacity constant or application default follows from the
proposed experiment alone.
