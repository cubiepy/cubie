# Same-address reuse versus two instruction streams

The source constructor and gated worker are implemented in
`instruction_sharing_probe.py` and `instruction_sharing_worker.py`.
CPU source/admission checks pass against retained native artifacts;
ordinary execution and counter admission remain unverified. The two
original compile attempts remain failed records under their original
gate. This is not a cache-domain conclusion. The existing 8-warp
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
  repeated control/constant-operand work and recurrence dependencies
  matched. Fixed selection and exit-path differences are recorded
  separately; whole-kernel instruction counts need not be identical.
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

The native gate proves the exact SM89 countdown form described below.
It requires two disjoint repeated FFMA ranges, eight shared input/output
accumulator registers in the same repeated order, and identical complete
body operand roles. It rejects predicated FFMAs, different constant
operands, extra MOVs, interior control, altered decrement/test chains,
alternative entries and nonlinear or returning epilogues.
Two native SMID reads must bracket the regions. Fixed selection/output
instructions and the first stream's extra exit branch are retained and
excluded from the hot-warp-FFMA denominator. The N/2N contrast checks
whether fixed cost matters. Native local frames are rejected.

Ordinary completion is labelled `ordinary_complete_counters_pending`;
profile completion is labelled `profile_complete_counters_pending`.
Neither status admits a physical cache claim. Independent counter review
must bind each selected stream to its executed PCs and verify
`selected_warps * N` executions of each arithmetic, parameter-MOV,
decrement and zero-test PC. Backedge transfers are `N-1` per selected
warp; exit transfers occur once. The predicated CALL's guard is visited
N times, and its metric-specific issued/predicate-on counts must be
checked separately. It must also compare actual
profiled shared configuration, achieved warps, eligible/issue counters,
ICC cycle counters and GCC instruction-request counters at equal work.
The profile event durations remain excluded from performance samples.

The original source/synthetic admission CPU receipt is retained at:
`cubie-notes/hardware_unroll_placement/verification/instruction_sharing_cpu_validation_20260905_e2.json`.
It validates generated syntax/operand counts, equal synthetic native
streams, noncontiguous SMID coverage and rejection of merged/overlapping
regions, memory traffic, changed chain spacing, predication, missing SMID
brackets, migration, incomplete coverage, selection errors and nonfinite
outputs. These are admission fixtures, not hardware measurements. No
CUDA import, compilation or GPU launch occurred in that validation.

## Uniform runtime-count backedges

The first native compilation, retained in
`hardware_unroll_placement/instruction_sharing_compile_e1`, failed
admission before any launch. Each 82000-byte repeated region contained
5120 FFMAs, one constant-memory `MOV`, one `UIADD3`, one `ISETP`, a
predicated `CALL.REL.NOINC` exit and an unconditional backedge. The native
gate rejected the additional control and traffic. All e1 artifacts and
its failure status remain unchanged.

The two PTX loop backedges use `@again bra.uni stream_*`. Each loop
counter begins with the same positive scalar kernel argument and is
decremented identically once per iteration. Its predicate and fixed
target are uniform among active lanes independently of the selected
SMID stream. The selection branch retains its original form. PTX's
[uniform branch semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-bra)
require identical predicates and targets among the active warp lanes;
the modifier does not promise any particular SASS encoding.

The uniform-branch change left the original admission unchanged.
Its second native attempt, `instruction_sharing_compile_e2`, failed
that gate and produced exactly the same cubin and disassembly as e1.
The two PTX files differ, but the observed native MOV and CALL form did
not change. Neither failed record contains a performance measurement.

The e1 FFMAs also name `c[0x0][0x210]` as their increment operand;
the repeated MOV reads multiplier `c[0x0][0x20c]`. These are constant
memory operands, including 5120 increment references per body traversal,
and are not evidence of 5120 cache misses. Admission retains their exact
addresses and per-traversal occurrences; the FFMA result count alone
does not describe all traffic. No opcode admission was widened by the
uniform-branch source change.

## Exact countdown and nonreturning exit admission

The e1/e2 cubin SHA is
`495962af77fdd626996e2dd09b6c13b60e9abd398a217398a970b009bc96aa9e`;
their SASS SHA is
`4acb977a279dfde99a2e2ca1accc0c4ad7055f9e5b20ad387d79b208c04e4c9e`.
The admission revision recognizes this proven control form, without
accepting arbitrary CALLs or MOVs. It applies to the emitted kernel's
SM89 parameter layout: unsigned repeat count at `c[0][0x200]`, FP32
multiplier at `c[0][0x20c]`, and FP32 increment at `c[0][0x210]`.
These addresses are program operands, not model coefficients.

`ULDC UR4,c[0][0x200]` at PC `0x120` is the sole prefix counter load;
every feasible path to either stream passes through it. The dispatch
at `0x1c0` falls through to `0x1d0` or branches to `0x14230`. No other
prefix branch may enter either body or its interior. A wide load that
overlaps the counter, a counter clobber, or a bypass rejects admission.

Each 82,000-byte body begins with its one multiplier MOV and executes
5,120 plain unconditional FFMAs. The eight accumulator registers are
`R9,R11,R13,R15,R17,R19,R7,R5` in both bodies; each FFMA updates its own
chain using the same multiplier and increment operands. Exactly one
unconditional `UIADD3 UR4,UR4,-1,URZ` precedes exactly one
`ISETP.NE.U32.AND P0,PT,RZ,UR4,PT`. Those administrative operations may
be interleaved with FFMAs, but their dependency order is checked. No
other operation can write the count, predicate or multiplier in the body.

The body ends with `@!P0 CALL.REL.NOINC` followed by an unconditional
backedge to its own first MOV. The first exit at `0x14200` targets
`0x14220`, a single unconditional branch to `0x28280`; the second exit
at `0x28260` targets `0x28280` directly. That common epilogue is linear,
reduces eight values with seven FADDs, writes four outputs and terminates
at unconditional `EXIT` at `0x28510`. No RET or path back into either
body exists. The terminal self-branch/NOP footer is unreachable from
that EXIT. Different exit targets or epilogue control reject admission.
The gate checks the common linear control path and seven-FADD/four-STG
counts. The eight-value reduction is an observation of the saved operand
dataflow; the gate is not a general proof of reduction/store arithmetic.
Exact runtime output equality remains a separate required numerical gate.

For an initial positive unsigned N, the logical CFG therefore visits
each selected repeated arithmetic/MOV/decrement/test instruction N
times, transfers around its backedge N-1 times and transfers to the
epilogue once. Both repeated regions have identical control and
constant-operand work. **The full paths differ:** the first stream has
one extra exit BRA, and selection prologue paths depend on runtime mode.
The receipt retains those fixed paths, their PCs and instruction counts;
it does not claim whole-kernel equal work. Each body has one multiplier
reference and 5,120 increment-operand references per traversal, without
assigning cache requests, sectors, misses or latency to those references.

These are static CFG facts conditional on the captured kernel/ABI.
SourceCounters must establish actual warp/thread execution and
predication counts before any instruction-sharing interpretation.
The existing ordinary numerical, residency, coverage, N/2N and
same-cubin gates remain required. Earlier failed artifacts remain
unchanged; they are not retroactively marked successful.
