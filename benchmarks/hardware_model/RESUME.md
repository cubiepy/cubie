# Hardware-derived unroll and placement research

## Binding task and authorization

The user has assigned the weekend GPU window to finding a post-codegen,
pre-compilation heuristic for loop-group unrolling and individual buffer
placement, with algorithm-family and LU/iterative workload distinctions.
On 2026-09-04 the user explicitly authorized any changes in throwaway
worktrees and required keeping them off main. This authorization covers
research harness, model, and experimental source edits in this worktree;
do not ask again for routine edits. Do not merge into main.

Worktree: `C:/local_working_projects/cubie-worktrees/hardware-unroll-placement`.
Branch: `codex/hardware-unroll-placement`.
Python: `C:/local_working_projects/cubie/.venv/Scripts/python.exe`.
New raw artifacts: `C:/local_working_projects/cubie-notes/hardware_unroll_placement`.
Set `PYTHONPATH` to the worktree's `src`, backend to `mlir`, and CUDASIM off
for hardware measurements. Use fresh explicit output/cache directories.

The main checkout at `C:/local_working_projects/cubie` stays untouched.
Do not install or update its environment while a measurement epoch runs.
The installed compiler wheel, not upstream source, defines compilation.
Keep code committed/pushed and maintain a ready PR. Never edit changelog.md.
All code changes require an independent verifier pass. GPU jobs are serial;
agents can analyze or compile only when explicitly coordinated with root.

## Model contract

- No timing-bank regressors, fitted family/config penalties, or arbitrary
  thresholds. Family differences come from actual call/loop structure.
- Only source-operation to SASS translation may be calibrated, by operation
  category, with inspectable generated kernels and compiler provenance.
- Every numerical model constant has a hardware/compiler source or a
  dedicated, valid microbenchmark. Unknown values remain unknown.
- Final decision inputs exist after codegen and before compilation. A
  baseline compile, CUDASIM trace, observed iteration count, register report,
  or historical winning config is an experimental label, not a shipped input.
- Separate dynamically executed work from the hot instruction-address
  footprint. Whole-kernel SASS size is not the latter.
- Separate register capacity/occupancy, scalarized arrays, address-taken
  local storage, spill slots, cache sectors, reads, writes and dirty eviction.
- Individual buffers and launch geometry are considered jointly with
  unrolling. Shared-memory carveout changes L1 capacity and occupancy.
- Float32 only. Timed launches require at least two complete occupancy
  waves at compiled geometry. Use driver/NVIDIA occupancy calculations.
- Preserve numerical status/mask mismatches as explicit invalid or unresolved
  rows. Never let a failed solve count as a performance win.
- Preserve all raw samples, exact source/compiler epochs and alias identity.
  Aliases are not independent repetitions. Compare matched wave/block
  references; do not pool incompatible duration/protocol cohorts.

## Integrated provenance and verification

Starting main: `1391bf35b9ec7bef9e957ebe7874cfe17914a142`, containing PR #910
(split Newton/Krylov flags). PR #909's `d62d7748` counter fix was imported as
`bc573e3b`. The 15 harness commits after `c58d0f27` through `b87c1104` were
cherry-picked, ending at `264e63e6`. Range-diffs are identical.

At that integration state, the complete real-GPU MLIR suite passed 3,631
tests; the simulator suite passed 3,570. Blocking flake8 passed. The required
`benchmarks/ab_gate.py` passed on both numba-cuda and MLIR. Full logs are under
the new artifact root's `verification/`. Those results do not validate
subsequent source edits; use the appropriate new gate before publishing them.

Independent compatibility review then found omitted observer producers:
`placement_landscape.features_row` reads `_typed_block_scheduler.BLOCK_LOG`
and `assignments.LIVENESS_LOG`, neither present on main. Original commit
`c716be2c` added these producers with the old placement harness. The repair
ports only observer additions, preserving current scheduling. The touch
allocator wrapper must also forward the current `unroll` argument.

## Evidence recovered on 2026-09-04

Primary old banks:

- `cubie-notes/unroll_landscape/post882`: 40 completed configs; factorial,
  single-False and fixed-four followups. Six backup records files preserve
  partial earlier attempts, not clean independent complete sweeps.
- `cubie-notes/unroll_landscape/split_flags`: 24 measured configs; split,
  split-fill, and one chain32/radau5 retime. Forty copied feature rows do
  not mean forty measured configs.
- `cubie-notes/unroll_placement_model/{RESUME,PLACEMENT_MODEL,SPILL_MODEL}.md`
  and tools/features: historical mechanism evidence. Its all-consteval
  placement outcomes are not current-unroll placement defaults.

Keeping stage, step-element, accumulator, solver-element and norms full
contains a best measured cubin in 38/40 old configurations. Exceptions are
Fabbri/Radau-3 and Radau-5, costing 5.43% and 24.06% against the unrestricted
winner. In Radau-5 every individual rollback loses while the joint rollback
wins. Other-small is not universally full for LU: chain32/kvaerno5 favors
rolling other-small and Newton together.

Two split cells have no timed identical cubin: Lorenz/kvaerno3_bicgstab and
Lorenz/radau_iia_5_bicgstab, fully rolled Newton+Krylov (`u11111100`). Old
aliases point to untimed libnvvm. Current b87 harness fixes new runs; use a
fresh bank, not mutation of historical records. Newton factors 2/4 crossed
with Krylov full were also omitted from the split grid.

Registers alone fail: many outcomes reverse at the same 255-register cap;
Lorenz/ROS23 BiCGSTAB loses about 4.4% when Krylov rolling lowers registers
86 to 78. Use reversed outcomes as physical discriminants.

The old placement formula clips negative removed traffic, although 236/1388
placement rows increase LDL+STL. It omits instruction-fetch cost, uses
unproven latency/rate constants and an unjustified L2 overflow fraction,
and reconstructs FIRK/total Newton workloads incorrectly. Its CUDASIM
Belady trace is neither static nor an accurate frame predictor.

## Hardware facts and measurement gaps

Device query: RTX 4070 SUPER sm89; 56 SM; 50,331,648-byte L2; 65,536 32-bit
registers/SM; 1,536 threads/SM; warp 32; 102,400-byte shared/SM;
101,376-byte maximum user shared/block. Driver CUDA API 13030, CuPy runtime
13020. Installed packages: cubie-numba-cuda-mlir 0.5.1.1, numba-cuda 0.30.4,
Numba 0.67.0, NumPy 2.5.2, CuPy CUDA13x 14.1.1. nvidia-smi driver 610.62.
SM clock was locked at 2670 MHz when inspected; query actual clocks per run.

Official Ada documentation supplies 128 KiB unified shared/L1 capacity;
carveouts 0/8/16/32/64/100 KiB; 1 KiB shared reservation/block; max 24
resident blocks and 48 warps/SM; 255 registers/thread. The old model omits
the 32 KiB carveout. Carveout requests are hints, not measured allocation.

SASS instructions are 16 bytes. FP32 add/mul/FMA throughput is 128
results/SM/cycle; approximate SFU operations 16. These are throughput,
not dependent latency or high-level function instruction counts.

The old microprobe's observed 129-to-145 KiB jump is real data but is not
yet a precise per-SM 128 KiB capacity fact: one-warp blocks were launched
below two waves, the requested 32 resident warps hit the 24-block limit,
and compiler-retained loops change source-trip normalization. NVIDIA
describes L0 per SM partition, ICC per TPC, GCC per GPC. Measure hot-region
capacity/phase effects and cache counters before assigning a domain/penalty.

Primary sources:

- https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-unit
- https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
- https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions
- https://developer.nvidia.com/blog/improving-gpu-performance-by-reducing-instruction-cache-misses-2/
- https://arxiv.org/html/2501.12084v2 (4090, not exact-SKU constants; L2 table/prose discrepancy)

## Active work and experiment sequence

Root coordinates these disjoint implementation lanes, then sends a verifier:
observer/touch compatibility repair; static AST/DAG descriptors and
alias-aware bank analysis; corrected hardware probes. All write only here.

1. Commit and verify the observer repair, preserving generated code and
   scheduling with logging enabled/disabled. Freeze a source epoch before
   measurements; don't let compiler workers import evolving source files.
2. Close split alias holes on counter-corrected source with exact matched
   baselines. Include the omitted partial-Newton/full-Krylov cells.
3. Probe the instruction transition with verified hot SASS and actual
   occupancy, then separate dependent latency from sustainable throughput.
4. Compare Fabbri/Radau-3/-5 joint flags, L96 Kvaerno3/5, and Lorenz/L96/chain
   ROS BiCGSTAB. Record corrected counters, dynamic work, hot instruction
   regions, registers/spills, occupancy and relevant cache/issue counters.
5. Retake discriminating placements under the same unroll epoch:
   chain32/kvaerno3 stage_base@64; chain64/radau5 delta@32;
   chain32/vern7 stage_accumulator@32. Then expand across state size and
   family/solver workload only where the inferred mechanism needs support.
6. Compose family-specific static loop/call descriptions with typed source
   op-to-SASS estimates, register/liveness bounds and measured hardware
   resources. Keep numerical iteration regimes symbolic where not inferable.
   Validate candidate rankings and each failure against held-out mechanism
   cases and fresh measurements, not a fitted timing objective.

Update this checkpoint with commands, revisions, output paths, findings,
active process/session IDs and verifier results after each bounded stage.
The model is not complete merely because setup, probes or tests pass.

## Checkpoint at 2026-09-04 10:36 UTC

Verified observer/workload commit: `640a9b99`, pushed on the research branch.
No PR yet at this checkpoint; create a ready PR after the pending tooling
verification. Main was not changed. The observer source files are frozen.

Current-source validation:

- `verification/observers_gpu.log`: 3,631 passed, logging enabled.
- `verification/observers_sim.log`: 3,570 passed, logging enabled.
- `verification/observers_ab_gate.log`: both backends PASS; compile resource
  tables identical to main. Full stdout preserved. Logging is default off
  in this gate; the enabled-observer suites above exercised the other mode.
- Existing exact codegen ordering tests passed with logging on and off.
- `verification/observer_metadata/records.jsonl`: actual fresh GPU feature
  extraction captures 16 codegen and 439 block-liveness records. Scheduler
  summary: 212 blocks, 46 reordered, 2,505 moved, 7,107 statements, maximum
  block 508; source/scheduled peaks 65. Cached compiler results omit this
  custom metadata, so the harness preserves fresh compile metadata and
  recovers it only when baseline key, source hash, and cubin SHA all match.
  Independent review passed this repair and `workload.py`.

Pending independent-review fixes in `unroll_landscape.py`,
`static_descriptors.py`, and `bank_analysis.py`: compiler/backend identity;
complete compatible sample eligibility; ambiguous compile exclusion;
duplicate reference in every block; boolean math-call types; conservative
dependencies after a dynamic-index store. The executor is repairing them.
Do not launch a new solver bank until these fixes pass re-review. Existing
`recovered/*_audit.json` was generated before these stricter filters and
must be regenerated into distinct filenames before using its rankings.
The two Lorenz split holes remain historical evidence, not yet filled.

Corrected instruction pilot finished under
`icache_pilot_20260904/`: exact hot regions 1,120 / 131,184 / 147,568 bytes,
64 / 8,192 / 9,216 FFMA instructions, 21-23 registers, no local storage.
All cases use 128 threads, driver residency 4 blocks/SM = 16 warps/SM,
448 blocks = two waves, 19,457 dynamic shared bytes for residency control,
five samples of at least 20 ms. Throughput floors are approximately
16.687 / 12.381 / 12.823 trillion FMA results/s. At this residency the
large streams slow down, but 128-to-144 KiB does not add a cliff. This
does not establish an instruction-cache capacity or domain. The next
experiment varies verified residency and collects ICC/GCC counters.

`hardware_probes.py` is independently under review. Arithmetic and
shared/local/global clock64 paths compiled with inspected SASS but have
not yet been timed. No GPU job is active at this checkpoint. Root owns
the GPU scheduling slot; agents must obtain it before launching.

`MODEL_PROTOCOL.md` states the physical derivation and validation contract;
the physical predictor itself is still under investigation.

A thread heartbeat is active: `cubie-weekend-hardware-model`, every
30 minutes through Sunday 2026-09-06 23:59 Pacific/Auckland. It continues
productive work from this checkpoint and current messages, remains quiet
on unchanged state, and stops launching jobs after the weekend window.
Do not duplicate it. Record new active processes, measurements, commits,
verification, and PR links at each bounded stage.

## Checkpoint at 2026-09-04 10:47 UTC

Independent verifier cleared all new modules, targeted timing protocol,
strict bank eligibility, static memory-dependency frontiers, source math
types, workload descriptors, metadata recovery, and model protocol. Earlier
pending review items above are resolved. Strict audits are
`recovered/post882_audit_strict.json` and `split_flags_audit_strict.json`.
They retain 38/23 rankable configs, 2,578/327 eligible launch groups, and
161/11 rejected groups. Read `RECOVERED_EVIDENCE.md` for exact receipts.

The 15-case ordinary instruction contrast completed successfully under
`icache_contrast_20260904`, with 75 samples, two waves, and 2670 MHz
endpoint clocks. For requested 128-to-144 KiB bodies, throughput ratios
are 0.396 / 1.020 / 0.924 at 8/16/32 resident warps. Cubin hashes are
identical across residency choices. This reproduces the eight-warp cliff
and establishes its residency dependence. The compact independent receipt
is `icache_contrast_ordinary_receipt_20260904.json`.

Real-GPU dependency pilots also completed successfully:
`fp32_dependency_pilot_20260904` and
`memory_{shared,local,global}_dependency_pilot_20260904`. They report
clock64 intervals, verified SASS, and exact pointer outputs. Do not call
their combined loop/address costs intrinsic memory latency. A larger
straight-line body and non-full-cycle pointer control are needed before
extracting hardware constants. `DEPENDENCY_EVIDENCE.md` is being written.

Counter access: the user clarified that counters are enabled and explicitly
authorized using an elevated process. Do not change driver permissions.
The normal-user attempt `icache_contrast_ncu_20260904.csv` failed with
ERR_NVGPUCTRPERM. Its profiled timing rows are excluded.

An elevated one-case gate succeeded and produced real ICC/GCC counters:
`icache_gate_elevated_20260904.ncu-rep` and its `_metrics.csv` export.
The first helper status says failed only because Nsight with `--export`
did not print counters to its diagnostic log; offline `--import --page raw`
verified the report. The corrected contrast helper is
`profile_elevated_contrast_20260904.ps1` under the artifact root. It uses
normal Windows RunAs, a hidden window, and explicitly imports the report
to CSV. Its status file is `elevated_profile_contrast_20260904_status.json`.
Root owns the GPU slot while this helper is pending/running. Check its
completion before starting any other GPU job.

Next solver retake, after that helper completes: fresh targeted cohort for
Lorenz Kvaerno3/Radau5 BiCGSTAB, both rolled, each individually rolled,
Newton 2/4 with Krylov full, and Krylov False controls. Use the corrected
targeted harness and new output/cache paths; preserve duplicate reference
solvers in every block. Then retake Fabbri Radau joint flags. The physical
heuristic remains under investigation; the verified tools are its inputs.

## Checkpoint at 2026-09-04 11:09 UTC

Ready PR: https://github.com/cubiepy/cubie/pull/912, branch
`codex/hardware-unroll-placement`. Source and measurement tooling are
frozen at `ff3a567f1646a63e70e04c1ab2ea999dc5ac1df4` in the detached tree
`C:/local_working_projects/cubie-worktrees/hardware-epoch-ff3a567f`.
Do not edit that measurement tree. The research tree may receive reviewed
tooling and evidence. Main remains untouched.

The elevated 15-case instruction contrast completed successfully. Nsight
reports must be imported with `--import --page raw --csv`: diagnostic
logs alone do not contain counter rows when `--export` is used. Raw
reports and imported counters are `icache_contrast_elevated_20260904*`;
the independently checked receipt is
`icache_elevated_counter_receipt_20260904.json`. At eight resident warps,
128-to-144 KiB raises GCC instruction misses from 3.23 to 50.08 percent
and no-instruction observations per verified warp FFMA from 0.545 to
4.402, with achieved residency unchanged. At 16 warps the same footprint
change has different reuse/eligibility behavior. This supports an
instruction-supply mechanism, not a universal byte-only timing curve or
a physical cache-capacity claim. `INSTRUCTION_COUNTER_EVIDENCE.md`,
`DEPENDENCY_EVIDENCE.md`, and `SOLVER_EVIDENCE.md` passed independent review.

The Lorenz bridge completed under `lorenz_split_bridge_e1`, filling both
historical rolled-loop timing holes and the partial-Newton omissions.
Its complete strict audit is `lorenz_split_bridge_e1_complete_audit.json`:
16 eligible groups per configuration, six samples per group, no rejects.
In the measured BiCGSTAB/Jacobi/inexact-Newton/prefactored regime, full
unrolling remains fastest for Kvaerno3. Newton-only rolling gives the
best Radau5 candidate ratio, 0.8949, at unchanged theoretical residency.
Radau5 count-2/count-4 Newton requests produce byte-identical all-full
cubins despite fresh compilation. They are native aliases, not distinct
performance choices. The actual lowering cause is still being traced.

GPU queue, owned exclusively by root:

1. `fabbri_radau_interactions_e1` is running from the frozen tree, driver
   PIDs 61492/79908, root exec session 36261. Radau3 completed in 348 s;
   Radau5 remains active. The Radau3 partial audit is explicitly named
   `fabbri_radau_interactions_e1_radau3_audit.json`. Its joint rollback
   is about 23 percent faster than full, while several individual
   rollbacks lose; every candidate reports 255 registers. Complete and
   audit Radau5 before writing a whole-cohort conclusion.
2. Elevated helper PID 48528 waits for that driver, then runs ordinary
   and profiled instruction bodies 128/132/136/140/144 KiB at eight warps.
   It selects the adjacent largest observed drop as an experimental
   bracket and measures those two bodies at 16 warps. Script:
   `profile_icache_bisect_20260904.ps1`; status:
   `icache_bisect_20260904_status.json`. Outputs use
   `icache_bisect{8,16}_{ordinary,elevated}_20260904` prefixes. Do not
   start another GPU job until the driver and elevated helper finish.

CPU-only work in the research tree: `counter_probe.py` collects matched
state-only and instrumented iteration labels without timing; it awaits
independent review by hardware_provenance before GPU execution.
`expansion.py` describes requested source-loop expansion and explicit
unknowns; placement_audit is independently reviewing its constant/alias
invalidation. `operation_translation.py` is being constructed by
hardware_provenance to calibrate only source-operation-to-native
instruction categories. Peers must review it before CUDA compilation.
These files are not yet evidence for a completed physical predictor.

After the serial queue: execute the reviewed counter capture against
the completed Lorenz bank, increase dependency bodies from 32 to 256
operations, and run non-full-ring correctness controls. The dependency
pilot values combine loop/address/setup/scheduling costs and remain
unsuitable as intrinsic memory-latency constants. Then select individual
placement contrasts under fixed, recorded unroll directives. Keep raw
measurements, failed attempts, all identities, and numerical differences.

## Checkpoint at 2026-09-04 11:18 UTC

Evidence commit `4221cbb9` and independently reviewed counter-capture
commit `30645371` are pushed to ready PR #912. Counter peer-review receipt:
`verification/counter_probe_review_20260904.json`. This clears a controlled
GPU launch, not the resulting labels before runtime validation.

The user reports that every new RunAs process triggers UAC and requests
one persistent elevated session. A dedicated profiler worker is being
implemented and independently reviewed before its single normal RunAs
launch. Reuse that worker for subsequent elevated profiling. Do not spawn
a fresh elevated process per job or change driver counter permissions.
It will remain idle between structured profiling requests and stop at
the weekend deadline or on an explicit stop request.

The previous queue remains active: Fabbri Radau5, then elevated instruction
bisect helper PID 48528. One ordinary (unelevated) counter-label gate is
queued after the bisect completes successfully: PowerShell PID 56356,
exec session 44980, status `counter_gate_20260904_status.json`. It measures
all-full Lorenz Kvaerno3 BiCGSTAB then Radau5 BiCGSTAB, using frozen source
and harness imports, logging enabled, one exact label sample per measured
geometry. Outputs are `counter_gate_{kvaerno3_bicgstab,radau_iia_5_bicgstab}_20260904`.
It stops on a failed state/status/native-identity gate and retains raw
arrays. No other GPU work is authorized concurrently with this queue.

## Checkpoint at 2026-09-04 11:39 UTC

The Fabbri cohort and fine instruction bisect completed. Full audit:
`fabbri_radau_interactions_e1_complete_audit.json`; independent combined
receipt: `verification/fresh_fabbri_icache_independent_20260904.json`.
There are 46 eligible six-sample Fabbri groups, no failed statuses or
NaN-mask mismatches, and 19 nonzero warm state differences. Best matched
candidate ratios are 0.772852 for Radau3 `u11100000` and 0.812139 for
Radau5 `u00100000`; both use LU, inexact_newton=False,prefactored=False.
All candidates report 255 registers and 256 resident threads/SM, hiding
large differences in local storage and whole-kernel code. These findings
do not isolate one physical mechanism or establish numerical equivalence.

Fine instruction bodies 128/132/136/140/144 KiB at eight warps give
16.359/11.585/8.674/7.235/6.482 T scalar FFMA/s. GCC instruction miss
fractions rise from 3.22 to 50.08 percent. The corresponding sixteen-warp
128-to-132 contrast improves 12.488 to 13.063 T FFMA/s. All seven pairs
match cubins and exact dynamic-instruction accounting. Declared registers
change from 21 to 23 across the first step; allocated registers remain
24 and measured residency remains stable in the eight-warp ramp. Read
the appended hardware evidence for units and physical-capacity limits.

The initial counter gate failed because the bank installs verbose linker
diagnostics and the counter tool did not. The complete disassembly diff
is only the link-options metadata string with/without `-v`, retained in
`counter_gate_kvaerno3_bicgstab_20260904/reference_sass.diff`. The repaired
tool installs the identical hook once during execution. It keeps the
strict cubin gate and does not weaken identity checks. Fresh v2 full-policy
gates pass for both Lorenz algorithms, as do eight distinct policy
contrasts under `counter_contrast_*_20260904`. Every label matches its own
state-only bank binary and exact finite states/status. Counter instrumentation
changes register allocation, so its printed raw diagnostic timings must
not be used as performance observations. Per-run totals are not warp
body counts. Raw audit and evidence prose are being independently checked.

Dependency controls completed at `fp32_dependency_body256_20260904` and
`memory_{shared,local,global}_{body256,nonfull_ring}_20260904`.
The body-256 minimum clock intervals per operation are approximately
4.050800/1.074223 for one/eight FFMA chains and
30.000034/35.570349/59.004102 for shared/local/global index-load chains.
Non-full-ring controls execute 1,081,377 transitions, remainder 33 modulo
64, and all 112 active threads return the independently checked nonzero
pointer index 36. These controls address the old full-cycle output hole;
the memory intervals still include address-dependency instructions.

Persistent elevation is now owned by `profiler_host.ps1`, host PID 48004,
under `_profiler_sessions/weekend_20260904`. Do not spawn another RunAs
per profile. The host starts only the fixed profiler worker and retains
elevation through worker restarts. A controlled idle restart changed
worker PID 8536 to 74332 with the same elevated host PID and no UAC:
`verification/profiler_host_restart_receipt_20260904.json`.
The earlier standalone workers exited after wrapper return-value and
Windows status-file sharing errors; both causes are repaired and raw
failures retained. The host provides recovery without another UAC.

Read `PROFILER_SESSION.md` before queue operations. The host and worker
are disarmed after every worker start until a fresh GPU release file is
written. Use structured queue files, fixed source hashes and fresh
outputs. `worker.restart` requests maintenance while idle; `host.stop`
ends the host. The fixed deadline is Sunday 2026-09-06 11:59 UTC.
At this checkpoint GPU jobs `root_profile_gate_002` and `_003` are queued
in that session. They validate real profiling/import through the same
worker. Check both receipts and disarm the worker before ordinary jobs.

Source-expansion tooling and counter-link repair are independently reviewed
and pushed as `c4d12fce`. Placement tooling has nine CPU-only worker
receipts and is under independent review. Operation-translation identity,
constant-memory and reachable-control-flow fixes are under re-review;
no translation kernel has been compiled yet. Main and frozen source
remain unchanged. The physical predictor is still under investigation.

## Checkpoint at 2026-09-04 12:01 UTC (Saturday 00:01 NZ)

Persistent elevated host PID 48004 is still alive. Its worker is PID
67168 after two reviewed maintenance restarts without UAC. Real hardware
gate jobs root_profile_gate_002/_003 both passed. Four actual solver
profiles then passed through this host, each with one kernel counter row,
profile/import exit zero, and successful exact state/status checks:

- `solver_profile_radau5_full_e1_v3`
- `profile_solver_radau5_newton_rolled_e1`
- `profile_solver_kvaerno3_full_e1`
- `profile_solver_kvaerno3_both_rolled_e1`

Each has `profile.ncu-rep`, `metrics.csv` and exported SASS source counters.
The first uses `source_counters_v2.csv`; its initial `source_counters.csv`
preserves a failed import with the wrong report filename. The others use
`source_counters.csv`. Per-PC instruction execution and sampled stalls
are different fields; preserve units from the raw metrics' units row.
placement_audit is constructing and independently validating the CPU
analysis. Profile event timings must not enter the ordinary timing bank.

The first two Radau profile attempts failed identity gates before launch.
The research source/harness files have different byte hashes from the
frozen tree despite equivalent text after newline normalization. New
`runtime_tree: epoch_ff3a567f` selects frozen source and harness imports
while running the new research probe. The remaining mismatch was four
CUDA toolkit environment overrides inherited by the elevated host but
absent from the frozen epoch. The fixed epoch child environment removes
only CUDA_HOME, CUDA_PATH, CUDA_PATH_V13_2 and CUDA_PATH_V13_3, recording
their inherited values in command.json. Strict source/compiler/cubin
checks remain unchanged. Independent source-review receipt:
`verification/profiler_epoch_independent_20260904.json`.

The 48-case operation translation compilation completed with zero kernel
launches at `operation_translation_compiled_20260904`. All artifacts and
native opcode deltas passed independent CPU re-disassembly; raw receipt
`operation_translation_audit_20260904.json`. See
OPERATION_TRANSLATION_EVIDENCE.md for five conditional instruction-vector
candidates and seven context-dependent cases. Uniform runtime parameters,
repeated denominators, contraction and output-address changes materially
affect lowering; these are not universal operation or register costs.

DEPENDENCY_EVIDENCE.md and COUNTER_EVIDENCE.md passed separate review:
`verification/dependency_counter_independent_20260904.json`. Radau5 full
versus Newton-rolled has identical four-channel totals for every run;
Kvaerno3 policy totals mostly differ per run despite close aggregate
means. Neither result gives event-level warp maxima.

The first fixed-unroll placement cohort `stage_base_placement_e1` is
complete: accepted attempt 5, duration 1.6, two paired blocks, six samples
per role per block, all accepted samples at least 20 ms, exact warm
states/statuses, and 18.2857 occupancy waves. Baseline/duplicate share
one cubin; shared has a distinct cubin. Both use 255 registers/thread
and 256 resident threads/SM; local storage grows from 688 to 752 bytes
per thread under shared placement. Indicative kernel times are 21.7 ms
versus 32.0 ms. Actual carveout/traffic await matched profiles; do not
infer them from the storage allocation. hardware_provenance is building
a separate placement_profile.py against this completed cohort, keeping
the verified placement_probe.py frozen.

CURRENT GPU OWNER: ordinary placement queue, exec session 4402. It runs
`delta_placement_e1` (chain64/radau5/delta, block32), then
`stage_accumulator_placement_e1` (chain32/vern7/stage_accumulator, block32),
both full eight-group unroll and two paired blocks, stopping on failure.
The elevated worker is DISARMED during this queue. Inspect completion and
clean workers before releasing profiling or starting any additional GPU
job. No other native compile is authorized concurrently.

unroll_evidence is inspecting retained optimized MLIR/LTO from the
Lorenz bank to localize counted-unroll aliases. Counts survive into
optimized MLIR; full/count2/count4 Radau5 cubin convergence occurs later.
Do not label this as ignored directives without the native-pass evidence.
The queued size/family unroll contrasts in MODEL_PROTOCOL.md have not
started. Main and the frozen measurement tree remain untouched; no
physical predictor has yet passed the full derivation/holdout gate.

## Checkpoint at 2026-09-04 12:17 UTC

Evidence/tools commit `8778dab7` is pushed to ready PR #912. Four Lorenz
solver profiles and the two stage_base placement profiles completed in
the same elevated host. Stage_base raw profile outputs are
`profile_stage_base_placement_{baseline,shared}_e1`, each with a successful
exact state-only solve and `source_counters.csv`. The observed shared
carveout changes 8192 to 65536 bytes. L2 read sectors change 499627117
to 1341212511; local load sectors 1943004895 to 1872183675; local store
sectors 712138649 to 909591670. Executed hardware instructions increase
about 1.1 percent. These are matched mechanism labels, not isolated
carveout causality. A same-cubin carveout control is being constructed.

The ordinary `stage_accumulator_placement_e1` cohort completed: exact
warm states/statuses, two blocks and twelve accepted samples per role,
duration 51.2, all samples above 20 ms. At block32, local frames change
616 to 1848 bytes/thread and resident threads/SM change 256 to 64;
indicative kernel times are 32.4 versus 143.6 ms. Its first two profile
attempts failed before compilation solely on generated module bytes.
The original reused generated module contains an unused residual helper
from a prior implicit solver, while the fresh ERK cache did not. The
reviewed profile tool preserves an exact content-addressed original
source snapshot and seeds a private cache with those full bytes; it
does not weaken the source/hash or native-identity gate. Both CPU role
replays pass. Review: `verification/placement_source_replay_review_20260905/receipt.json`.
Snapshot sidecars are added without editing original manifest/results.

`delta_placement_e1` stopped at exact warm equivalence; all three workers
closed cleanly. Baseline and duplicate are identical; shared differs in
10999247 state values, at most 6 ULP / 3.5763e-7. Inputs match and all
states are finite with success statuses. Layout inspection finds distinct,
disjoint delta storage, and arithmetic opcode counts already differ in
PTX. A unique cause or accuracy certificate is not established, and no
timing is accepted from this cohort. See
`verification/DELTA_PLACEMENT_AUDIT_20260905.md` and
`delta_placement_numerical_audit_20260905.json`. Do not loosen the gate.

CURRENT GPU OWNER: ordinary eight-config size/family unroll queue,
PowerShell PID 11956, exec session 80814. Status is
`size_family_unroll_20260905_status.json`; data is under
`size_family_unroll_e1/<system>__<algo>/`, with separate caches and logs
per config. It compares 14 explicit policies plus duplicate references,
workers=1, block-solvers=2, per-config timeout 21600 seconds, cohort
`size-family-e1`. Lorenz/ROS23 BiCGSTAB completed; its strict audit is
`size_family_unroll_e1/lorenz__rosenbrock23_bicgstab_complete_audit.json`.
At this checkpoint L96_20/ROS23 BiCGSTAB is active. Check the status file
for later progress. All source/harness imports use frozen ff3a567f.

The elevated worker PID 67168 is DISARMED. Its release file records
external_jobs_finished=false and blocked PID 11956. Two reviewed ERK
profile retries are queued: `stage_accumulator_placement_baseline_e1_v2`
and `stage_accumulator_placement_shared_e1_v2`. Their target SHA is
`bffac0f7f4028fe6cdee1a80b33d9bfb05d25ca2decbc2f29d806f2a6e51612b`.
Only release them after the ordinary queue and all children exit. Keep
host PID 48004 open; no new RunAs/UAC is needed.

CPU work: placement_audit's solver_profile_analysis.py passes the four
actual solver datasets and is receiving independent review from
unroll_evidence. Treat outputs as provisional until the receipt clears
command/launch identity and exact counter integrality. Initial exact
per-PC counts distinguish whole encoded code from executed addresses:
Kvaerno3 full is 421120 bytes encoded but only 63072 bytes executed;
Radau5 full is 974720 encoded / 959328 executed. These are address unions
over a solve, not a temporally reconstructed cache working set.

COUNTED_UNROLL_EVIDENCE.md passed root source/prose review. Extraction
receipt_v2.json distinguishes cached MLIR UTF-8 literal hashes from
CRLF export-file hashes; the original ambiguous receipt is retained.
Root rehashed all six cache, MLIR-export and LTO-export artifacts in
`verification/counted_unroll_independent_20260905.json`. No specific
late native optimization pass is yet identified.

unroll_evidence is implementing buffer_descriptors.py from the reviewed
workload/expansion graph design: allocation/view identity, interprocedural
read/write effects and conditional element SSA. Native register use,
spills and temporal cache reuse remain explicit unknowns. Its separate
verifier is hardware_provenance after the bounded carveout work. No
production source changed; main and frozen measurements remain intact.
