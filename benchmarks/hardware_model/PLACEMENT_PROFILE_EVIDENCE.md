# Stage-base placement: native work and memory traffic

For chain32/Kvaerno3/LU with all eight loop groups fully unrolled and a
64-thread block, placing `stage_base` in shared memory is slower in the
accepted ordinary cohort. The matched profiles show several changes:
slightly more exact instruction work, a larger local frame, more local
stores and L2 traffic, and a larger shared-memory configuration. This
comparison does not isolate the cost of explicit shared loads/stores or
the shared/L1 capacity split.

## Verified artifacts and scope

Paths here are relative to
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.
Inputs are `stage_base_placement_e1` and its matched profiles
`profile_stage_base_placement_baseline_e1` and
`profile_stage_base_placement_shared_e1`. Each profile uses
`metrics.csv`, `source_counters.csv`, its saved `profile.ncu-rep`, and
`benchmark/result.json` with raw arrays and native artifacts.

The independent outputs are
`verification/placement_profile_analysis_independent_20260905/{baseline,shared}/analysis.json`
and `per_pc.jsonl`; the same directory's `receipt.json` records PASS.
Adapter SHA256 is
`d15d147d18a11c3a46f16d303e94fe0672e315b9a42b2a40c4da811b10350bb5`.
Its pure CSV/native/aggregation core remains
`ecd46638fb63c83540ead9a586f856d04856500bdeb83aba11f543a4e11731b3`.
The reviewer reran both full audits using only saved-report import and
saved-cubin disassembly; there was no CUDA compilation or GPU launch.

The audits verify the completed ordinary manifest, accepted attempt 5,
both mirrored blocks, all 36 measurement rows, all warm NPZs, input and
compiler identities, and each profiled solve's exact original warm
state/status. The historical profile script `c3ab3ecb...93b34c71` is
bound to its explicit independent source-review receipt; actual command
arguments select exactly one solve. Every source PC/opcode/operand
matches the saved original-role cubin after documented display
normalization. Source/compiler/configuration and requested/actual
launch geometry are matched; the buffer's resolved owner/layout and
allocator evidence are retained, including its alias relationship.

The baseline buffer resolves to local storage of 32 FP32 elements; the
shared buffer resolves to shared slice `[0,32)`. Physical registry and
allocator checks are independently recorded in
`verification/placement_physical_review_20260904` and
`verification/placement_probe_independent_review_20260904.json`.
This is one named-buffer placement contrast, not multiple independent
allocations inferred from alias names.

## Ordinary timings and compiled resources

Each ordinary role has 12 accepted measurements, six per mirrored
block, at duration 1.6 and 262,144 trajectories. Every sample is at
least 20 ms. The duplicate is a separately compiled baseline process;
its native binary matches the baseline exactly. All three roles have
exactly matching accepted warm state/status arrays.

| Ordinary role | Minimum, ms | Median, ms | Maximum, ms |
|---|---:|---:|---:|
| Baseline local | 21.576992 | 21.709072 | 21.766752 |
| Duplicate local | 21.577408 | 21.668112 | 21.856833 |
| Shared | 31.996544 | 32.013952 | 32.058720 |

These are descriptive statistics of retained ordinary measurements.
Profiled durations and the much larger CUDA-event time printed around
the instrumented/replayed solve do not replace these samples.

| Resource | Baseline local | Shared |
|---|---:|---:|
| Registers/thread | 255 | 255 |
| Allocated registers/thread | 256 | 256 |
| Local frame bytes/thread | 688 | 752 |
| Actual dynamic shared bytes/block | 4 | 8,448 |
| Resident blocks/SM | 4 | 4 |
| Resident threads/SM | 256 | 256 |
| Occupancy waves at compiled geometry | 18.285714 | 18.285714 |
| Actual profiled shared configuration, `Kbyte` | 8.192000 | 65.536000 |

Both profiles have grid `(4096,1,1)`, block `(1,64,1)`, and 56 SMs on
SM89. The shared dynamic allocation is exported as `8.448000 Kbyte/block`,
meaning 8,448 bytes/block. Similarly, the shared configurations mean
8,192 and 65,536 bytes, not 8.192 or 65.536 KiB. The actual configuration
is established for each profiled launch; the ordinary cohort did not
measure its per-launch carveout.

Ada has 128 KiB of unified L1 data/shared capacity. Subtracting the
observed configurations gives nominal remaining capacities of 120 and
64 KiB for these profiles. That arithmetic does not measure effective
data working-set capacity or instruction-cache capacity. [Ada unified
cache documentation](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#unified-shared-memory-l1-texture-cache).

## Exact native work and footprints

| Observable | Baseline local | Shared |
|---|---:|---:|
| Whole instruction addresses | 23,936 | 23,632 |
| Whole encoded instruction bytes | 382,976 | 378,112 |
| Addresses with positive exact execution | 8,088 | 8,107 |
| Encoded bytes at those addresses | 129,408 | 129,712 |
| Exact warp instructions | 5,107,818,264 | 5,164,436,169 |
| Exact thread instructions | 163,214,649,187 | 165,025,118,453 |
| Exact predicated-on thread instructions | 162,558,476,791 | 164,368,684,617 |

Exact warp work rises 1.10846%. The executed-address union changes by
only 304 bytes even though whole native bodies shrink. Neither quantity
is a temporal working set or a per-SM cache footprint. Full address
frequency distributions and all opcode/predicate counters are retained.

| Exact warp opcode executions | Baseline local | Shared |
|---|---:|---:|
| `FFMA.FTZ` | 2,172,158,109 | 2,174,898,489 |
| `FMUL.FTZ` | 1,132,534,129 | 1,102,978,243 |
| `FADD.FTZ` | 88,483,246 | 115,268,878 |
| `LDL` | 369,018,262 | 309,856,519 |
| `LDL.LU` | 113,776,484 | 155,210,822 |
| `LDL.64` | 28,425 | 28,425 |
| `LDL.LU.64` | 1,674,102 | 1,674,102 |
| `STL` | 174,676,927 | 224,202,565 |
| `STL.64` | 1,708,153 | 1,708,153 |
| `LDS` | 0 | 80,356,896 |
| `STS` | 0 | 134,190,304 |

Arithmetic and local-memory instructions both change. A source-level
placement does not correspond to replacing only that buffer's native
local operations with shared operations. The larger local frame and
additional local stores are observed compiler outcomes; these counters
alone cannot assign every local site to a specific buffer or distinguish
explicit locals, ABI storage, and register spills. No equal cycle cost
is assigned to different opcodes. Warp execution counts are independent
of participating-thread count; the separately retained predicated-on
thread counters matter for traffic interpretation. [Nsight source
counter definitions](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#source-page).

## Memory traffic, instruction delivery and sampled stalls

| Metric suffix (`.sum`) and exported unit | Baseline local | Shared |
|---|---:|---:|
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld` [sector] | 1,943,004,895 | 1,872,183,675 |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_st` [sector] | 712,138,649 | 909,591,670 |
| `lts__t_sectors_op_read` [sector] | 499,627,117 | 1,341,212,511 |
| `lts__t_sectors_op_write` [sector] | 713,762,339 | 911,695,521 |
| `lts__t_sectors_lookup_miss` [sector] | 1,127,264 | 1,152,096 |
| `sm__icc_requests` [cycle] | 751,653,925 | 698,265,410 |
| `sm__icc_requests_lookup_miss` [cycle] | 117,447,104 | 62,243,862 |
| `gcc__cache_requests_type_instruction` [request] | 151,799,895 | 171,040,371 |
| `gcc__cache_requests_type_instruction_lookup_miss` [request] | 31,793 | 177,649 |
| `smsp__warps_issue_stalled_no_instruction` [warp] | 3,463,470,222 | 1,110,199,852 |

Local-load sectors fall, while local stores and L2 reads/writes rise.
L2 totals do not identify a single source buffer and can include other
traffic sources. Instruction-delivery counters are mixed: ICC miss
cycles and no-instruction stalled warps fall, while GCC instruction
requests and misses rise. An ICC `cycle` value must not be relabeled a
miss request or multiplied by an invented miss latency.

Source exact instruction sums equal their software aggregate metrics
in both profiles. Hardware `smsp__inst_executed.sum` exceeds the source
warp totals by 51 instructions in each; the residual is retained without
an assigned cause or correction. Source and global sample totals agree
at 3,368,398 and 4,545,462. Source long-scoreboard All samples rise from
1,194,646 to 2,732,132; Not-issued samples rise from 958,429 to 2,347,694.
These are periodic observations, not exact executions or lost cycles;
All and Not-issued sampling can use separate replay passes. [Nsight
sampling definitions](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#source-page).

The physical model must admit simultaneous code-generation, local
storage, shared access and data-cache capacity changes. Same-binary
carveout controls can separate one part of that problem, but this
two-binary comparison cannot apportion its slowdown between them.
These measurements establish a rejected placement in this particular
family/LU/size/unroll configuration. They supply no universal shared
memory penalty, register formula, or new pre-compile default.
