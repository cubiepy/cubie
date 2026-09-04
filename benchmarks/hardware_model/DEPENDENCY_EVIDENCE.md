# Dependency probe evidence

Recorded 2026-09-04 from real-GPU runs on the RTX 4070 SUPER.
This is a CPU audit of the saved data. No probe code was changed and
no GPU work was launched by the author of this document. These are
measured instruction-chain intervals, not hardware latency constants.

## Raw receipts and measured intervals

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
query gives one resident block per SM: two complete occupancy waves.
The 32 resident warps are a resource allocation; the measured region
has at most one active warp per SM, with one active lane. The shared
and local initialization precedes the active-lane exit and clock read.

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

All hot bodies contain 32 verified, unpredicated target instructions.
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

## Output-control limitation

For the current block size and active lane, every initial cursor is
`(1024 * block_index) & 63 = 0`. Every recorded memory advance is a
multiple of 64, so its correct final pointer is also zero. All saved
outputs are zero. The SASS and nonzero clock intervals demonstrate
executed dependent loads; this final-pointer equality alone cannot
detect omitted full cycles.

A discriminating existing-CLI correctness control uses 33 operations
and 32,769 repeats on the same 64-element ring. The intended advance
is 33 modulo 64, so a one-cycle permutation must finish away from its
starting node. Inspect the recorded repeat count: automatic duration
calibration doubles it when necessary and could restore a full-cycle
advance. Require the recorded `(operations * repeats) % elements`
to be nonzero before accepting this control. It is a validation
case, not the paired 32/256-body timing comparison.

## Next controls using the frozen CLI

These commands have not been executed in this audit. Run sequentially
in the orchestrator's GPU slot from the worktree root, with distinct
output directories. The first comparison changes body operations
from 32 to 256 while preserving the ring, chains and geometry.
Calibration keeps event duration at least 20 ms and records its final
repeat count. Compare per-thread clock distributions and SASS, not
unadjusted event durations.

```powershell
$env:PYTHONPATH = 'C:\local_working_projects\cubie-worktrees\hardware-unroll-placement\src'
& 'C:\local_working_projects\cubie\.venv\Scripts\python.exe' -m benchmarks.hardware_model.hardware_probes fp32 --operations 256 --chains 1,8 --block-size 1024 --active-lanes 1 --resident-warps 0 --carveout 0 --iterations 32768 --output 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\fp32_dependency_body256_20260904'
& 'C:\local_working_projects\cubie\.venv\Scripts\python.exe' -m benchmarks.hardware_model.hardware_probes memory --space shared --elements 64 --operations 256 --chains 1 --block-size 1024 --active-lanes 1 --resident-warps 0 --carveout 0 --iterations 4096 --output 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\memory_shared_dependency_body256_20260904'
& 'C:\local_working_projects\cubie\.venv\Scripts\python.exe' -m benchmarks.hardware_model.hardware_probes memory --space local --elements 64 --operations 256 --chains 1 --block-size 1024 --active-lanes 1 --resident-warps 0 --carveout 0 --iterations 4096 --output 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\memory_local_dependency_body256_20260904'
& 'C:\local_working_projects\cubie\.venv\Scripts\python.exe' -m benchmarks.hardware_model.hardware_probes memory --space global --elements 64 --operations 256 --chains 1 --block-size 1024 --active-lanes 1 --resident-warps 0 --carveout 0 --iterations 4096 --output 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\memory_global_dependency_body256_20260904'
```

The 256-body control reduces repeated loop administration per target
instruction, but it does not reduce per-load address arithmetic. The
compiler may change the branch encoding as the body grows; confirm
actual tail controls and the exact operation count. A converging
cycles/op sequence can constrain the measured chain's steady-state
cost without fitting a performance model. No difference is presumed
to equal pure branch cost, and no arbitrary coefficient is learned.

Run the output control separately for each memory space:

```powershell
foreach ($taskSpace in 'shared', 'local', 'global') {
    & 'C:\local_working_projects\cubie\.venv\Scripts\python.exe' -m benchmarks.hardware_model.hardware_probes memory --space $taskSpace --elements 64 --operations 33 --chains 1 --block-size 1024 --active-lanes 1 --resident-warps 0 --carveout 0 --iterations 32769 --output "C:\local_working_projects\cubie-notes\hardware_unroll_placement\memory_${taskSpace}_noncycle_control_20260904"
}
```

## Candidate dedicated latency instruments, not implemented

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

These candidates require an explicit implementation/review pass.
The present evidence supports combined instruction-chain costs only.
