# Same-address reuse versus two instruction streams

The source constructor and gated worker are implemented in
`instruction_sharing_probe.py` and `instruction_sharing_worker.py`.
CPU source/admission checks pass; native compilation and execution remain
unverified. This is not a cache-domain conclusion. The existing 8-warp
capacity transition cannot
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

## Instrument protocol and current validation

The controller's default writes `kernel.py`, an exact worker snapshot,
an exact controller snapshot and `request.json`. It uses only the Python
standard library. The generated intrinsic contains two literal PTX
regions, each with 5,120 FFMAs by default. It retains the validated eight
independent contracting FP32 recurrences and a runtime unsigned repeat
counter. Native instruction addresses, register roles and loop control
must pass admission; source/PTX duplication alone is insufficient.

Run from the research tree using the frozen runtime environment:

```powershell
python benchmarks/hardware_model/instruction_sharing_probe.py `
    --out <fresh-source-directory>

python benchmarks/hardware_model/instruction_sharing_probe.py `
    --out <fresh-compile-directory> --compile-only

python benchmarks/hardware_model/instruction_sharing_probe.py `
    --out <fresh-ordinary-directory> --execute

python benchmarks/hardware_model/instruction_sharing_probe.py `
    --out <fresh-profile-directory> --profile-mode all_a `
    --ordinary-dir <completed-ordinary-directory>
```

The profile command runs under the reviewed elevated session's fixed
`ncu` action. Repeat it separately for `all_b` and `mixed`, preserving
three distinct outputs. Profile mode loads the accepted ordinary repeat
count, validates all retained raw ordinary arrays, mirrored membership,
source/compiler identity and exact native binary, then performs one
target launch. It never runs ordinary calibration inside the profiler.
Each profile still needs its native executed-PC binding: address order
does not identify which admitted body is selected by A or B.

Ordinary calibration doubles N until all three controls exceed 20 ms.
Its rows and snapshots are retained separately from measurements. Two
measurement blocks at N and two at 2N each run the mirrored sequence
`A,B,mixed,mixed,B,A` three times, giving six samples per arm and block.
Any measurement below 20 ms fails the cohort. The same CUfunc, cubin,
single native overload, resources and driver occupancy are checked at
every setting; the shared reservation admits one eight-warp block/SM.
There are at least two full theoretical occupancy waves.

Each launch retains complete FP32 output and unsigned entry-SMID,
exit-SMID and selected-stream arrays in NPZ, hashes them, and records
per-SMID block counts, selected warp counts, raw event milliseconds and
before/after clocks. Every mode must cover the queried number of physical
SMs, both selections must occur in mixed mode, and selection must remain
uniform within each block/warp. Entry/exit differences, nonfinite output,
or unequal output for matching N invalidate the cohort. Different SMID
coverage patterns remain visible instead of being replaced by an assumed
topology or exact scheduling distribution.

The native gate requires exactly two disjoint repeated FFMA ranges,
eight independent accumulator registers in the same repeated order, and
identical complete body operand roles after register renaming. It rejects
predicated FFMAs, memory instructions, calls, interior branches and
unapproved loop instructions. Two native SMID reads must bracket the
regions. Fixed prologue/output instructions are retained and excluded
from the reported hot-warp-FFMA denominator; the N/2N contrast measures
whether their fixed cost matters. Native local frames are rejected.

Ordinary completion is labelled `ordinary_complete_counters_pending`;
profile completion is labelled `profile_complete_counters_pending`.
Neither status admits a physical cache claim. Independent counter review
must bind each selected stream to its executed PCs and verify, per hot
PC, `selected_warps * N` warp executions. It must also compare actual
profiled shared configuration, achieved warps, eligible/issue counters,
ICC cycle counters and GCC instruction-request counters at equal work.
The profile event durations remain excluded from performance samples.

CPU receipt:
`cubie-notes/hardware_unroll_placement/verification/instruction_sharing_cpu_validation_20260905_e2.json`.
It validates generated syntax/operand counts, equal synthetic native
streams, noncontiguous SMID coverage and rejection of merged/overlapping
regions, memory traffic, changed chain spacing, predication, missing SMID
brackets, migration, incomplete coverage, selection errors and nonfinite
outputs. These are admission fixtures, not hardware measurements. No
CUDA import, compilation or GPU launch occurred in that validation.
