# Dependency probe evidence

Recorded 2026-09-04 from real-GPU runs on the RTX 4070 SUPER.
This is a CPU audit of the saved data. No probe code was changed and
no GPU work was launched by the author of this document. These are
measured instruction-chain intervals, not hardware latency constants.

## Initial 32-operation bodies

The [receipt JSON](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/dependency_pilot_receipt_20260904.json)
records the source paths and SHA256s, all five per-case sample files,
raw cycle extrema, medians, exact denominators, and output limitations.
Each sample file contains 112 unsigned 64-bit clock deltas, one per
active thread. The CPU audit matched each array's shape, minimum,
maximum and mean to its sample record and matched all event times to
the result rows. Every sample's before/after SM clock reading is
2670 MHz; those readings are not a continuous frequency trace.

All cases use 1,024 threads/block, only thread zero participating in
the measured region, 112 blocks, and 56 SMs. The driver occupancy
query gives one resident block per SM: two theoretical occupancy waves.
The 32 warps describe the initial block allocation. Inactive warps exit
before the timed chain, so this resource query does not establish at
most one simultaneously active timing warp per SM after partial block
retirement. The saved data does not establish whether overlap occurred.
The shared and local initialization precedes the inactive-lane exit
and clock read. NVIDIA describes resource-release granularity as
architecture-dependent and unspecified in this
[block scheduling clarification](https://forums.developer.nvidia.com/t/scheduling-of-blocks-does-every-thread-of-a-block-need-to-finish-before-a-new-block-launches/274531).
These historical intervals therefore retain a scheduling/residency
uncertainty as well as the endpoint and address-work qualifications below.

| Case | Repeats | Native operations per active thread | Minimum / median / maximum cycles per operation, across all 560 thread samples |
|---|---:|---:|---|
| FP32, one chain | 262,144 | 8,388,608 FFMAs | 4.375018 / 4.375030 / 4.375050 |
| FP32, eight chains | 524,288 | 16,777,216 FFMAs | 1.625009 / 1.625020 / 1.779528 |
| Shared, one chain | 32,768 | 1,048,576 LDS | 30.000031 / 30.000031 / 30.000031 |
| Local, one chain | 32,768 | 1,048,576 LDL | 35.562531 / 35.567851 / 35.592718 |
| Global, one chain | 16,384 | 524,288 LDG | 59.031639 / 59.033311 / 59.036037 |

The normalization is each thread's `clock64` difference divided by
its verified repeated operations, not event time divided by a chain
length. The latter would include two waves, initialization and launch
overhead. The eight-chain FP32 sample numbered one has a real tail up
to 1.779528 cycles/FFMA, corresponding to its slower 22.108736 ms
event. It is retained in the receipt rather than discarded.

Raw datasets:

- [FP32 results](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/fp32_dependency_pilot_20260904/results.jsonl)
- [Shared results](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_shared_dependency_pilot_20260904/results.jsonl)
- [Local results](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_local_dependency_pilot_20260904/results.jsonl)
- [Global results](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_global_dependency_pilot_20260904/results.jsonl)

NVIDIA defines `%clock64` as an unsigned 64-bit cycle counter in
[PTX ISA §10.24](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clock64).
The [CUDA Programming Guide, time function](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#time-function)
distinguishes elapsed thread intervals from cycles spent executing
its instructions because scheduling can intervene. The saved SASS
confirms two `CS2R ... SR_CLOCKLO` instructions and a 64-bit stored
delta. These measurements retain that scheduling qualification.

## Instructions inside and around the clock interval

The initial hot bodies contain 32 verified, unpredicated target instructions.
The auxiliary counts below are instruction counts, not cycle costs;
they must not be subtracted as one cycle each.

| Case | Hot address range, end exclusive | Hot instructions and auxiliary opcodes | Clock-read addresses |
|---|---|---|---|
| FP32, one chain | `0x150..0x3b0` | 38: FFMA32, MOV2, ISETP1, IADD3 2, BRA1 | `0xa0`, `0x3b0` |
| FP32, eight chains | `0x1f0..0x450` | 38: FFMA32, MOV2, ISETP1, IADD3 2, BRA1 | `0x110`, `0x450` |
| Shared | `0xf60..0x13b0` | 69: LDS32, IMAD31, SHF3, ISETP1, IADD3 1, BRA1 | `0xec0`, `0x13b0` |
| Local | `0x270..0x6c0` | 69: LDL32, IMAD19, LEA14, ISETP1, IADD3 2, BRA1 | `0x1d0`, `0x6c0` |
| Global | `0x130..0x11a0` | 263: LDG32, SHF32, IMAD114, LEA64, IADD3 18, ISETP1, CALL1, BRA1 | `0x80`, `0x11a0` |

The first clock read also precedes one-time loop setup: ten static
instructions for one-chain FP32/global, nine for local/shared, and
13 for eight-chain FP32, including three seed FFMAs in the latter.
These are outside the repeated operation denominator but inside the
clock interval. Large repeat counts amortize them; no numerical
cycle cost is assigned to them.

The ending clock instruction has no data operand consuming the final
loaded pointer or FP32 accumulator. In the shared case, the last LDS
is at `0x1390`, the backedge at `0x13a0`, and the clock at `0x13b0`.
The data-dependent output use occurs afterward. Thus the brackets
verify instruction placement but do not prove that the last target
operation has completed before the clock is read. The final operation
of each chain is an endpoint uncertainty. Do not call the normalized
number a strict intrinsic-latency upper bound.

The one-chain FP32 recurrence is visible as repeated
`FFMA.FTZ R13, R13, ...`; eight independent accumulators reduce the
average interval per FFMA. Both bodies also read the constant addend
`c[0x0][0x1b8]` and reload multiplier `c[0x0][0x1b4]` once per repeat.
The observed 4.375 and 1.625 values cannot be split into intrinsic
FFMA latency, issue throughput, and branch cost from this pair alone.
Eight chains in a single sparse warp do not measure the full SM's
128 FP32 results/cycle ceiling.

## Global addressing is a measured compiler workload

The [global SASS, line 36](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_global_dependency_pilot_20260904/memory_global_ops32_chains1_warps0/kernel.sass:36)
shows signed-high extraction and multiple multiply/add/LEA instructions
between consecutive LDG instructions. The
[saved PTX](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_global_dependency_pilot_20260904/memory_global_ops32_chains1_warps0/kernel.ptx)
makes the cause explicit: the loaded index is multiplied by a runtime
array stride, combined with an array offset, scaled by four, and added
to the base pointer. The signed index is widened. For each pointer
step, those address computations depend on the previous load.

Consequently, 59.03 cycles/step is a combined signed-index,
array-descriptor addressing and cached-load chain. Its extra 194 hot
instructions relative to local/shared are not 194 cycles of measured
overhead: instructions overlap and have different dependencies and
execution resources. Subtracting the shared result also would not
isolate global memory latency because the address chains and cache
states differ.

The logical ring is 64 int32 values, 256 bytes. Global uses one ring
shared across the grid; shared uses one per block; local reserves
256 bytes per thread and initializes all 1,024 threads' rings before
timing. The local initialization therefore writes 256 KiB of logical
thread-local payload per block, while the timed region accesses one
thread's ring. Logical bytes do not establish physical cache sectors
or traffic. The global experiment is a repeatedly accessed small
window, not an isolated DRAM-latency experiment. Carveout preference
zero is recorded, but actual carveout and cache hit levels require
counter evidence.

## Completed 256-operation and output controls

The [combined CPU audit receipt](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/dependency_counter_audit_20260904.json)
verifies eight additional cases. For each case it records the exact
source/cubin hashes, raw cycle files, clock brackets, opcode counts,
output hashes and denominator. CPU re-disassembly of every cubin exactly
reproduced the saved SASS. Recounting each hot address range confirmed
256 or 33 unpredicated target instructions as requested. The frozen
probe source SHA256 is
`18e049529d00a75fb0d369d878e41f95d87c0a8aa30f4652690a06ea12d11896`.
No kernel was compiled or launched for this audit.

The 256-operation bodies retain the pilot's 1,024-thread block,
112-block grid, two theoretical occupancy waves and one active lane.
All saved cycle arrays have shape `(112, 1)` and dtype `uint64`.
Every warmup, calibration and measured array matched its saved minimum,
maximum and mean; measured event arrays matched their result rows.
Each measured run's before/after snapshots showed SM 2670 MHz, memory
10251 MHz and P2. They do not establish continuous clock stability.

| Case | Recorded repeats | 32-body minimum cycles/op | 256-body minimum / median / maximum cycles/op |
|---|---:|---:|---|
| FP32, one chain | 32,768 | 4.375018 | 4.050800 / 4.050808 / 4.050829 |
| FP32, eight chains | 131,072 | 1.625009 | 1.074223 / 1.074225 / 1.074231 |
| Shared, one chain | 4,096 | 30.000031 | 30.000034 / 30.000034 / 30.000034 |
| Local, one chain | 4,096 | 35.562531 | 35.570349 / 35.572425 / 35.595054 |
| Global, one chain | 4,096 | 59.031639 | 59.004102 / 59.004940 / 59.006296 |

Each 256-body distribution contains five samples of 112 clock deltas.
The old and new cohorts are separate runs, not alternating paired
measurements. The larger FP32 body reduces administration per FFMA and
has a smaller measured interval; these runs do not isolate the cause.
The shared/local/global intervals remain close to their 32-body results.
This observation does not isolate branch latency or
justify subtracting auxiliary instruction counts as cycle costs.
The compiler also changes the repeated control sequence:

| 256-body case | Total hot instructions | Target count | Auxiliary instructions |
|---|---:|---:|---|
| FP32, either chain count | 263 | FFMA256 | MOV2, ISETP1, IADD3 2, CALL1, BRA1 |
| Shared | 518 | LDS256 | IMAD166, SHF91, ISETP1, IADD3 1, MOV1, CALL1, BRA1 |
| Local | 518 | LDL256 | IMAD131, LEA126, ISETP1, IADD3 2, CALL1, BRA1 |
| Global | 2,055 | LDG256 | SHF256, IMAD898, LEA512, IADD3 130, ISETP1, CALL1, BRA1 |

These larger bodies preserve the addressing dependency and endpoint
qualification described above. In particular, the global result remains
a signed-index/descriptor/load chain. One or eight active accumulators
in a sparse warp do not measure aggregate SM FP32 throughput.

Raw completed body results:

- [FP32](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/fp32_dependency_body256_20260904/results.jsonl)
- [Shared](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_shared_body256_20260904/results.jsonl)
- [Local](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_local_body256_20260904/results.jsonl)
- [Global](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_global_body256_20260904/results.jsonl)

For both the 32- and 256-body memory timings, every active thread begins
at `(1024 * block_index) & 63 = 0`, and the number of pointer advances
is a multiple of 64. Their correct zero output alone cannot detect an
omitted full cycle. The completed nonfull-ring controls address this
ambiguity with 33 operations and exactly 32,769 recorded repeats:
1,081,377 advances, or 33 modulo 64. The independent audit verified
the saved ring is a permutation with one 64-node cycle, then traversed
that actual ring to compute the expected final pointer. All 112 outputs
are exactly **36** in every memory space.

| Nonfull-ring control | Minimum cycles/step | Expected and actual pointer |
|---|---:|---:|
| Shared | 30.000030 | 36 |
| Local | 35.579180 | 36 |
| Global | 59.077624 | 36 |

These controls retain one measured profile each, not five timing samples.
They establish a nontrivial final-pointer control, while remaining
insensitive to omitted whole 64-step cycles. Finite FP32 outputs were
verified; no bit-exact CPU FMA recurrence comparison is claimed.
Actual shared carveout and cache hit levels remain unmeasured.

Raw output-control results:

- [Shared](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_shared_nonfull_ring_20260904/results.jsonl)
- [Local](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_local_nonfull_ring_20260904/results.jsonl)
- [Global](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/memory_global_nonfull_ring_20260904/results.jsonl)

## Dedicated latency instrument design

The separate implementation and its native admission gates are described
in [LATENCY_PROTOCOL.md](LATENCY_PROTOCOL.md). It holds inactive lanes at
a final uniform CTA barrier and verifies that barrier in native code.
Its fresh direct-address measurements do not inherit the historical
int32 intervals below. The alternatives explain the design distinctions.

Changing the ring payload and cursor to uint32 could remove signed
extension, but changing only the initial cursor is insufficient when
the subsequent array loads are int32. Even fully unsigned indices do
not remove the observed runtime descriptor stride/offset operations.
Compile-only inspection must establish which operations disappear
before treating this as a cleaner instrument.

A more controlled 32-bit global experiment can pass a base address
and an unsigned index to a PTX primitive performing explicit
`mad.wide.u32` byte addressing followed by `ld.global.ca.u32`.
This fixes the layout and removes descriptor operations while retaining
the payload width. It still measures address generation plus load
dependency. A `.cg` variant provides a cache-path control: cache hints
and subsequent SASS/counter evidence must agree before assigning a
level. PTX [load syntax and cache operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)
document explicit address spaces, load types and cache qualifiers.

A separate 64-bit address ring permits repeated
`ld.global.ca.u64 pointer, [pointer]` without index multiply/add between
loads. Those are integer addresses, not FP64 arithmetic. It changes
payload width, alignment and footprint, so it must remain a distinct
instrument with matched byte-window controls and exact stored-address
validation. Shared-memory byte-offset rings offer an analogous way
to remove repeated index-to-byte shifts from the LDS chain. Neither
instrument inherits a latency value from the current int32 results.

Any inline-PTX version must preserve compiler-visible side effects,
correct memory-space conversions and a data-dependent completion
guard before the ending clock. NVIDIA's
[Inline PTX Guide, §§1.2.2–1.2.3](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#pitfalls)
explains address-space and optimization hazards. A volatile or memory
clobber prevents certain compiler transformations; it does not itself
prove completion of an outstanding load. Validate the final SASS
dependency and clock ordering independently, retain full raw outputs,
and keep at least two compiled-occupancy waves for timing runs.

The historical evidence supports combined instruction-chain costs only.
The separate instrument requires its own ordinary and profile evidence.
