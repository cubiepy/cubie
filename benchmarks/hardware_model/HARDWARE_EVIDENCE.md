# Hardware evidence and next instruction-cache contrast

Recorded 2026-09-04. This document records measurements and their
limits. No instruction-cache capacity or sharing domain is established
by these measurements, and no timing penalty is fitted here.

## Primary hardware provenance

| Quantity | Evidence | Use and limit |
|---|---|---|
| SM89 resources | NVIDIA [Ada Tuning Guide, §1.4.1.1](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#occupancy): 64K 32-bit registers/SM, 255 registers/thread, 48 warps/SM, 24 blocks/SM | Resource limits; use the occupancy API for compiled geometry. |
| Unified data L1/shared | [Ada guide, §1.4.2.2](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#unified-shared-memory-l1-texture-cache): 128 KiB combined, shared carveouts 0/8/16/32/64/100 KiB, 1 KiB reserved/block | The older placement model's tuple omits 32 KiB. This is not instruction-cache capacity. |
| Carveout preference | [CUDA Driver API, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html): preference can be overridden | Requested percentage is not measured L1 capacity. Retain LaunchStats. |
| Instruction size | NVIDIA [CUDA Binary Utilities, JSON-format notes](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html): 16 bytes, explicitly illustrated for SM89 | Measure repeated address ranges, not ELF size. |
| Instruction hierarchy | [Ada whitepaper, printed pp.10–11](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf): L0 instruction cache in each SM partition; four partitions/SM | No byte capacity is given. |
| ICC and GCC | [Nsight Compute Profiling Guide, §2.3.4](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): ICC is per TPC; misses go to per-GPC GCC, which caches instructions/constants and fetches misses from L2 | A metric beginning `sm__` does not establish per-SM physical ownership. |
| Relevant footprint | NVIDIA [instruction-cache investigation](https://developer.nvidia.com/blog/improving-gpu-performance-by-reducing-instruction-cache-misses-2/): aggregate hot regions and drift between warp instruction streams affect pressure | The example is Hopper; its sizes and optimal factors do not transfer. |
| Memory latencies | [Luo et al., Table 3, arXiv:2501.12084v2](https://arxiv.org/html/2501.12084v2): RTX4090 L1 32.0, shared 30.1, L2 273.0, global 571 cycles | Different SKU/driver/toolchain. The prose gives L2 284.8, inconsistent with the table. These are not 4070 SUPER constants. |
| Arithmetic throughput | [CUDA Best Practices, §12.1.1/Table 5](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions): SM89 FP32 add/mul/FMA 128 results/SM/cycle, approximate SFU operations 16 | These are throughput ceilings, not dependent instruction latencies or high-level function costs. |

The pilot's CUDA device queries report 56 SMs, 50,331,648 bytes L2,
65,536 registers/SM, 1,536 threads/SM, 24 blocks/SM, and 1,024 reserved
shared bytes/block. Its sampled SM clocks before and after every timed
sample are 2670 MHz. The device attribute `CLOCK_RATE=2505000` is not
the observed execution clock. Full provenance is in the
[pilot manifest](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/manifest.json).

## Existing probe compared with the corrected pilot

Old source:
[icache_probe.py](/C:/local_working_projects/cubie-notes/unroll_placement_model/tools/icache_probe.py).
Old records:
[icache_probe.jsonl](/C:/local_working_projects/cubie-notes/unroll_placement_model/features/icache_probe.jsonl).
New implementation:
[hardware_probes.py](/C:/local_working_projects/cubie-worktrees/hardware-unroll-placement/benchmarks/hardware_model/hardware_probes.py).
The old generated `tools/probe_kernels` directory is not present. The
old tool disassembles a temporary cubin, deletes it, and retains counts
only. Consequently the old exact hot addresses, register allocation,
instruction scheduling, and call targets cannot be reconstructed from
these records. Do not infer them from the new kernels.

| Axis | Old probe | Corrected pilot | Interpretation |
|---|---|---|---|
| Recurrence | Eight accumulators; initial `tid*1e-7 + k`; multiply by float32 `1.0000001`, add float32 `1e-9` | Eight accumulators; initial `k+1 + (tid&7)*0.001`; multiply by float32 `0.99999994`, add float32 `1e-7` | Same source chain width; different input values/prologue. New recurrence contracts toward a finite fixed point. Exact operand allocation/scheduling must come from SASS. |
| Inner stream | Eight updates in `unroll_if(range(m), flag)` | Eight updates in an explicitly full inner loop | The runnable pilot uses forced-loop expansion, not literal flattening into thousands of Python statements. Exact repeated FFMA counts are checked before timing. |
| Outer loop | Plain runtime `range(iters)` | Runtime `unroll_if(..., ROLLED)` | The new repeat loop is explicitly rolled and its compiled shape checked. |
| Recorded bytes | Total counted SASS ×16, including prologue/tail/padding | Actual repeated SASS start/end plus full opcounts | Total 129 KiB and hot 128 KiB are different quantities. |
| Block geometry | 32 threads; grid = label ×56 | 128 threads; grid = 2 ×56 ×driver resident blocks | Old labels 8/16/32 specify grid population, not resource-limited residency. A 32-thread block cannot exceed 24 resident warps/SM because SM89 permits 24 blocks/SM. |
| Residency control | No resource constraint or occupancy query | Driver query; 19,457 dynamic shared bytes requests four blocks/SM, yielding 16 theoretical warps/SM | Actual residency and partition assignment still need profiling. Neither probe records a warp-to-SMSP mapping. |
| Shared carveout | No setting or measurement in script | Preference 100%; exact dynamic allocation retained; actual carveout marked unknown | This is a deliberate control difference. The arithmetic hot region has no LD/ST opcodes but does read fixed constant-memory operands. |
| Grid/waves | No two-wave gate; the 8/16 cases are below one full architectural block-capacity wave; 32 is below two | 448 blocks/57,344 threads; exactly two driver-occupancy waves | Old/new per-thread timing denominators are not directly interchangeable. |
| Repeat count | `max(4, floor(4e6/(8*m)))`; warmup is two repeats | Starts at 128 repeats; doubles to >=20 ms; records calibration separately | Old work per thread is approximately fixed while runtime changes with residency. New duration is controlled independently of body size. |
| Timing records | Five events, their minimum, and `elapsed/(source_ops*repeats)` | All samples, work denominators, clocks before/after, resources and artifacts | New reported rate uses verified operations ×repeats ×participating threads. It is throughput, not latency. |
| Toolchain | No per-row package/driver manifest | MLIR wheel 0.5.1.1, CUDA13.3, driver610.62, Python3.14.3 | The old RESUME's wheel version is contextual; it does not establish each row's actual toolchain. |

The old 8-labelled-warp observations have a jump from 129 KiB total
SASS at 0.88652665 ns/source FFMA (line 9) to 145 KiB at 2.23579101
(line 32). The old 16-labelled-warp observations already have a
different shape: 129 KiB is 2.35133830 ns/source FFMA (line 41), while
161/193/225 KiB are approximately 2.26 (lines 42–44). Thus the pilot's
128-to-144 behavior does not contradict the old 16-labelled-warp
shape; it also does not validate a capacity model.

At 2048 old trips (line 11), static FFMA count drops to 8199 and a
fourth `BRA` appears despite 16384 source FFMA operations. The old
209 KiB plateau at larger trips is a different compiled loop shape.
Those rows cannot define a straight-line instruction-capacity curve.

## Pilot receipt

Command, executed without another GPU workload:

```powershell
$env:PYTHONPATH = 'C:\local_working_projects\cubie-worktrees\hardware-unroll-placement\src'
$env:CUBIE_CACHE_DIR = 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\icache_pilot_cache_20260904'
& 'C:\local_working_projects\cubie\.venv\Scripts\python.exe' -m benchmarks.hardware_model.hardware_probes icache --body-kib 1,128,144 --resident-warps 16 --output 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\icache_pilot_20260904'
```

| Requested FFMA bytes | Actual repeated range, end exclusive | Hot instructions / FFMA | Registers | Repeat count | Five raw event times, ms |
|---|---|---|---|---|---|
| 1 KiB | `0x1b0..0x610`, 1120 B | 70 /64 | 22 | 131072 | 28.904097, 28.852160, 28.897312, 28.826624, 28.868608 |
| 128 KiB | `0x1b0..0x20220`, 131184 B | 8199 /8192 | 21 | 1024 | 38.881279, 39.042049, 38.853634, 39.166977, 38.998016 |
| 144 KiB | `0x1b0..0x24220`, 147568 B | 9223 /9216 | 23 | 512 | 21.443584, 21.351423, 21.101568, 21.117952, 21.431295 |

All three have zero local allocation and no LD/ST opcodes in the
repeated body. FFMAs read an addend from `c[0x0][0x1b8]`, and each
repeat loads the multiplier from `c[0x0][0x1b4]` through `MOV`.
These fixed constant-memory operands must be acknowledged when
interpreting instruction/constant-cache counters. The first two
perform 481,036,337,152 scalar FMA results;
the third performs 270,582,939,648. Rates, rather than the unadjusted
event times, therefore compare them. Maximum sampled rates are
16.687, 12.381 and 12.823 trillion FMA results/second, respectively.
These measurements demonstrate a slowdown from the small stream, but
do not isolate its cause or locate a cache boundary.

The small loop ends in predicated `BRA`. Both large loops use
predicated `CALL.REL.NOINC` to the forward exit followed by unconditional
`BRA` to the loop head. The checker allows that tail-exit encoding
only when it follows every measured operation and targets outside the
body. It rejects interior control flow. This concrete encoding change
is another reason to collect branch-resolution and ICC/GCC counters;
it is not evidence that branch cost explains the slowdown.

Raw evidence, including generated source and cubins:

- [All result rows](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/results.jsonl)
- [Small source](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/icache_ops64_chains8_warps16/kernel.py), [SASS](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/icache_ops64_chains8_warps16/kernel.sass), [raw samples](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/icache_ops64_chains8_warps16/samples.jsonl)
- [128 KiB source](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/icache_ops8192_chains8_warps16/kernel.py), [SASS](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/icache_ops8192_chains8_warps16/kernel.sass), [raw samples](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/icache_ops8192_chains8_warps16/samples.jsonl)
- [144 KiB source](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/icache_ops9216_chains8_warps16/kernel.py), [SASS](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/icache_ops9216_chains8_warps16/kernel.sass), [raw samples](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_pilot_20260904/icache_ops9216_chains8_warps16/samples.jsonl)

## Verified ordinary contrast

The independent verifier passed source and saved-artifact review.
The subsequent ordinary run completed all 15 cases successfully.
All 75 sample records have five samples per case, exact agreement
between timing rows and sample receipts, and 2670 MHz SM clock
samples before and after each launch. The endpoint readings do not
constitute a continuous frequency trace.

| Requested FFMA body, KiB | Actual hot bytes | 8 warps | 16 warps | 32 warps |
|---|---:|---:|---:|---:|
| 64 | 65,648 | 18.664 | 18.836 | 18.991 |
| 120 | 122,992 | 16.180 | 13.458 | 14.666 |
| 128 | 131,184 | 16.389 | 12.542 | 14.101 |
| 144 | 147,568 | 6.489 | 12.794 | 13.026 |
| 192 | 196,720 | 6.493 | 12.799 | 12.603 |

Values are maximum sampled trillions of FP32 FMA results/second,
using verified operations times runtime repeats times participating
threads. Each FMA result is two floating-point operations. At the
sampled frequency, the documented arithmetic ceiling is
`56 * 2670e6 * 128 = 19.13856e12` FMA results/second. The 64 KiB
cases achieve 97.52%, 98.42%, and 99.23% of that ceiling.

The 128-to-144 KiB rate ratio is 0.39594 at eight warps, 1.02010 at
16, and 0.92377 at 32. Thus a sharp transition is reproduced at eight
theoretical resident warps and changes substantially with residency.
This contradicts a single residency-independent timing penalty. It
does not identify a physical cache capacity or domain, or establish
which part of the change is instruction reuse versus latency hiding.

For each footprint, the cubin and SASS hashes are identical across
all three residency settings. Every contrast footprint uses the same
forward `CALL.REL.NOINC` exit followed by the backedge `BRA`; there is
no tail-encoding transition within this contrast. Registers are 21
for 64/128/192 KiB and 23 for 120/144 KiB, with zero local allocation.
The verified resource constraint gives 2/4/8 blocks per SM, with
224/448/896 grid blocks and 33,025/19,457/10,241 dynamic shared bytes
per block respectively. All cases use block size 128 and two complete
driver-occupancy waves. Achieved occupancy and actual carveout remain
unmeasured because profiling failed.

The reproducible CPU receipt preserves source line numbers, all raw
sample times, SHA256 hashes, exact denominators, constant operand
addresses, and minimum/median/maximum rates:
[ordinary receipt JSON](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_contrast_ordinary_receipt_20260904.json).
Its source is the untouched
[ordinary results](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_contrast_20260904/results.jsonl)
and each case's linked `samples.jsonl` and `kernel.sass` artifacts.

## Contrast protocol and failed counter attempt

Use 64/120/128/144/192 KiB requested FFMA bodies at 8/16/32
theoretical resident warps, with block size 128 and eight chains fixed:
15 cases. The old sharp transition occurred in the rows labelled eight
warps; its rows labelled 16 already showed a broad plateau. Including
eight actual theoretical resident warps tests that regime with verified
occupancy and two waves. Each case queries its own occupancy. Retain
actual hot bytes and tail encoding because requested bytes omit loop
control. The pilot reports native capacity of 12 blocks/SM, so 48 warps
is feasible for its compiled resource counts; add that contrast only
if the first measurements leave a residency mechanism unresolved.

First collect ordinary timings to a fresh dataset:

```powershell
$env:PYTHONPATH = 'C:\local_working_projects\cubie-worktrees\hardware-unroll-placement\src'
$env:CUBIE_CACHE_DIR = 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\icache_contrast_cache_20260904'
& 'C:\local_working_projects\cubie\.venv\Scripts\python.exe' -m benchmarks.hardware_model.hardware_probes icache --body-kib 64,120,128,144,192 --resident-warps 8,16,32 --iterations 4096 --output 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\icache_contrast_20260904'
```

Counters require a separate invocation/dataset. This command
does not change clocks, execute production solvers, or modify Cubie
configuration. It profiles one launch per case; kernel replay may
execute that launch multiple times internally. Its event timings are
not the ordinary timing sample. `--clock-control none` preserves the
externally established clock setting; `--cache-control none` avoids
profiler-forced cache flushing between replay passes.

```powershell
$taskMetrics = 'sm__icc_requests.sum,sm__icc_requests_lookup_hit.sum,sm__icc_requests_lookup_miss.sum,sm__icc_requests_lookup_miss_tag_hit.sum,sm__icc_requests_lookup_miss_tag_miss.sum,gcc__cache_requests_type_instruction.sum,gcc__cache_requests_type_instruction_lookup_hit.sum,gcc__cache_requests_type_instruction_lookup_miss.sum,smsp__warps_issue_stalled_no_instruction.sum,smsp__warps_issue_stalled_branch_resolving.sum,smsp__warps_eligible.sum,smsp__inst_executed.sum,sm__cycles_elapsed.sum'
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\ncu.bat' --clock-control none --cache-control none --kernel-name regex:probe --section LaunchStats --section Occupancy --metrics $taskMetrics --csv --log-file 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\icache_contrast_ncu_20260904.csv' --export 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\icache_contrast_ncu_20260904' 'C:\local_working_projects\cubie\.venv\Scripts\python.exe' -m benchmarks.hardware_model.hardware_probes icache --body-kib 64,120,128,144,192 --resident-warps 8,16,32 --iterations 4096 --profile-once --output 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\icache_contrast_profiled_20260904'
```

The attempted invocation failed with `ERR_NVGPUCTRPERM` and exit 1.
The file named
[NCU CSV](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_contrast_ncu_20260904.csv)
contains the diagnostic, not counter rows. No successful report or
counter measurements were produced. Treat the entire profiled attempt
as failed regardless of any per-case status; its event times are not
ordinary performance samples.

Metric enumeration checks availability, not permission. Once the user
has enabled access through the supported NVIDIA controls, first run a
single 64 KiB/eight-warp profile with the same metric set to a fresh
output directory. Require process success, a readable report, and
nonempty requested counter rows before running the complete contrast.
This is a pre-sweep capability gate, not a timing measurement.
NVIDIA's [permission guidance, Windows section](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
identifies NVIDIA App's System > Advanced > Developer controls and
states that changing access requires administrative privileges. This
research lane did not change permissions or attempt a bypass.

Metric availability was queried without a GPU launch on 2026-09-04:

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\ncu.bat' --query-metrics --chip ad104 --query-metrics-mode base
```

The installed query reports the ICC bases as `Counter / cycle`, while
the GCC instruction bases are `Counter / request`. Its descriptions
identify ICC miss-tag-hit as covered by a pending miss and
miss-tag-miss as sent to GCC. Do not equate raw ICC totals with GCC
request totals. It also confirms the no-instruction, branch-resolving,
eligible-warp, executed-instruction and SM-cycle counter bases.

Interpret the contrast by mechanism, without fitting coefficients:

1. If a body-size transition coincides with ICC miss growth, inspect
   whether requests reaching GCC and GCC misses also grow. Their
   combination identifies the observed level of instruction supply
   pressure, not physical byte capacity by itself.
2. If large-body cost changes with residency and ICC behavior changes
   with it, warp phase overlap/reuse is relevant. If ICC misses stay
   comparable but eligible warps rise and time falls, latency hiding
   is a distinct candidate. Retain actual achieved occupancy.
3. If the tail-encoding change accompanies branch-resolution stalls
   without increased ICC/GCC pressure, investigate control flow.
   If both change, this contrast has not separated the causes.
4. Bisect only the interval that counters identify. `--ffmas` permits
   finer operation counts; eight-chain counts step by eight FFMAs,
   i.e.128 instruction bytes. Keep the successful residency fixed.

An eventual cache-sharing experiment requires interference between
distinct warmed code regions and recorded execution locations.
[Jia et al., §3.3/Figures3.3–3.5](https://arxiv.org/pdf/1804.06826)
provides an original experimental method. Its Volta capacities and
per-SM findings are not Ada constants.

## Verification and remaining limits

The icache pilot is run-verified. FP32 one/eight-chain probes and
shared/local/global one-chain probes compile with exact operation
counts and in-kernel clock reads; they have not been run in this lane.
Their clock intervals include loop/address instructions, scheduler
contention, and a possible one-operation endpoint effect. The global
indexing implementation has substantial address/index code, so those
intervals must not be entered as pure global-load latency constants.
Memory output validation compares every active thread with the exact
pointer-cycle advance. LLVM inline assembly supplies clock64 because
the installed NVVM path did not accept its special-register form.

CPU parser/ring checks, Python compilation, Ruff, 79-column checking
and the repository's blocking Flake8 selection passed. The independent
verifier passed the code and saved icache/FP32/memory artifacts,
including exact counts, true dependencies, clock brackets and two-wave
geometry. Runtime validation of the FP32/memory mechanisms remains
separate. This lane launched no GPU work after releasing the pilot
slot; the orchestrator executed the ordinary contrast and failed
counter attempt recorded above.

## Independently audited fine instruction-supply contrast, 2026-09-04

The successful elevated profiles and ordinary timings use the clean
frozen checkout `ff3a567f1646a63e70e04c1ab2ea999dc5ac1df4`, hardware-probe
source hash `18e049529d00a75fb0d369d878e41f95d87c0a8aa30f4652690a06ea12d11896`,
MLIR wheel 0.5.1.1 and the RTX 4070 SUPER, SM89, driver 610.62. Raw root:
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.
For each residency, ordinary/profile pairs are
`icache_bisect{8,16}_ordinary_20260904/results.jsonl` and
`icache_bisect{8,16}_elevated_20260904_artifacts/results.jsonl`;
wide counters are `icache_bisect{8,16}_elevated_20260904_metrics.csv`.
Each ordinary/profile pair has identical cubin bytes. Ordinary timing
uses five CUDA-event samples; rates below use their minimum and are
trillions of scalar FFMAs/s, not FLOPs/s. Profile replay times are separate.

| Requested resident warps/SM | Nominal FFMA KiB | Actual hot bytes | Ordinary T scalar FFMA/s | GCC instruction miss % | Measured active warps/SM |
|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 131184 | 16.359 | 3.220 | 8.000 |
| 8 | 132 | 135280 | 11.585 | 17.318 | 8.001 |
| 8 | 136 | 139376 | 8.674 | 30.345 | 7.998 |
| 8 | 140 | 143472 | 7.235 | 42.907 | 7.998 |
| 8 | 144 | 147568 | 6.482 | 50.079 | 8.001 |
| 16 | 128 | 131184 | 12.488 | 0.964 | 15.365 |
| 16 | 132 | 135280 | 13.063 | 4.512 | 14.900 |

The seven data rows are ordinary/profile JSONL lines 1–5 and 1–2;
their wide-counter CSV lines are 3–7 and 3–4 (line 2 contains units).
GCC miss percentage is instruction-lookup misses divided by instruction
requests, using the `.sum` counters. ICC counters have cycle units and
must not be substituted into that request ratio.

Every profile exactly matches the warp-instruction accounting
`grid_blocks * active_lanes / 32 * (hot_instructions * 4096 + 43)`.
The 43 outside-loop instructions are an observed exact accounting term,
not a fitted latency constant. Every hot loop contains the requested
FFMAs plus seven instructions, has the same CALL/BRA tail form, and runs
4,096 iterations. Geometry is two full queried occupancy waves throughout.
Local bytes are zero. Declared registers change from 21 at 128 KiB to 23
at 132–144 KiB; Nsight reports 24 allocated registers/thread throughout.
All profiles report 102,400 bytes of configured shared memory. Dynamic
reservations are 33,025 bytes for eight warps and 19,457 for sixteen;
the theoretical uncapped resident-block count stays 12.

At eight warps, increasing code size accompanies growing GCC instruction
misses, rising no-instruction stalls and fewer eligible warps. Dividing
the respective SMSP average warp counts by average elapsed SM cycles
gives no-instruction values 0.470→1.499 and eligible values 1.458→0.437
across 128→144 KiB. ICC miss-tag-miss remains about 688,184–688,285 cycle
counts. The continuing 132→144 ramp occurs with fixed declared registers
and unchanged tail form. Together these observations support instruction
supply pressure; they do not measure a pure miss latency.

Sixteen-warps behavior differs: 128→132 KiB slightly improves ordinary
throughput while GCC misses rise, and actual active warps remain below
the requested sixteen. ICC miss-tag-miss is 37,146,220→34,426,407 cycle
counts, much larger than at eight warps. The same cubins are used for
the matching eight-/sixteen-warp bodies. This dependence on residency
rules out treating these rows as a single deterministic code-byte
threshold or as a register-count-only model. Warp phase/reuse and
instruction-cache hierarchy are plausible explanations, not isolated
measurements of physical cache capacity or sharing domain. No constants
or penalty curve are fitted here.

The independent receipt records all hashes, line references, resources,
normalized counts and ordinary/profile pairs:
`C:/local_working_projects/cubie-notes/hardware_unroll_placement/verification/fresh_fabbri_icache_independent_20260904.json`.
