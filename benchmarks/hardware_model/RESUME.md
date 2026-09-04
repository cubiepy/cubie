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

## Checkpoint at 2026-09-04 12:40 UTC

Ready PR #912 includes committed four-solver analysis and independently
reviewed evidence through `35562c0b`. The core analyzer is frozen at
SHA `ecd46638fb63c83540ead9a586f856d04856500bdeb83aba11f543a4e11731b3`.
The separate placement adapter passed independent actual-data replay for
both stage_base profiles at SHA
`d15d147d18a11c3a46f16d303e94fe0672e315b9a42b2a40c4da811b10350bb5`:
`verification/placement_profile_analysis_independent_20260905/receipt.json`.
It retains the historical target-script review binding and exact decimal
Nsight units. It does not modify the frozen core or invent old snapshots.

The same elevated host PID 48004 has restarted its worker to PID 31904,
still elevated and DISARMED. No new RunAs/UAC occurred. The loaded worker
SHA is `ec5ec90e1fca25f931ade79d91c2271056ee67124d410ce79cb1ee0c573de782`.
New jobs preserve a locked, exact-byte `benchmark_source.py` and its hash
beside their reports, while executing the original verified path.
Independent seven-check CPU receipt:
`verification/profiler_source_snapshot_independent_20260905_e2/receipt.json`.
The first reviewer attempt is explicitly marked invalid; its incompatible
reader aborted checks and is not a PASS. Historical outputs stay intact.

CURRENT GPU OWNER remains ordinary queue PID 11956 / exec session 80814.
Five configurations completed: all three ROS23 BiCGSTAB sizes and both
Kvaerno3 BiCGSTAB sizes. L96_20/Kvaerno5 is active; chain32/Kvaerno5 and
L96_20/Radau5 BiCGSTAB follow. Strict completed-bank audits are saved for
the first four configurations. Check the status file for current state.
The fresh ROS results reverse with size: rolling Krylov is about 4.4%
slower for Lorenz, about 47% faster for L96_20, and about 20% faster for
chain32. L96_20/Kvaerno3 distinguishes the loop interaction: both rolled
is about 18% faster, but Krylov-only rolled about 13% slower; count four
is about 2.27 times full. These are ordinary paired descriptive ratios,
not heuristic thresholds, accuracy certificates or isolated causes.

The worker queue contains the two ERK source-replay retries, six original
ROS size profiles, and seven newly validated requests prefixed `z_`:
chain32 ROS full/rolled plus L96 Kvaerno3 full/both-rolled/Krylov-rolled/
count-two/count-four. All use frozen runtime and exact counter-probe SHA.
Validation receipts: `verification/size_profile_queue_additions_20260905`.
Release remains false, blocked by PID 11956. Only arm after the ordinary
queue and every owned child have exited; inspect real profile failures.

The same-cubin carveout probe passed independent CPU/source review at
SHA `d15eccfdf9f94ddd222003c98b123997cba7b279b3c4d2f074e56f5bb0274743`.
Receipt: `verification/carveout_independent_review_20260905.json`.
Run one ordinary stage_base baseline control after the current GPU queue,
then matched profiles if its exact native/state/timing gates pass. The
8/64 KiB preferences remain hints until per-launch NCU observation; no
ordinary actual carveout or causal conclusion is yet established.

Buffer descriptor CPU construction covers nine actual solver/system
cases without native overloads. Independent review found alias/control
issues, which are being repaired before release. No source liveness to
native-register conversion is accepted. Physical issue-capacity and
SMID instruction-domain work remain bounded investigations, with no
fitted coefficients or completed pre-compile decision model claimed.

## Checkpoint at 2026-09-04 13:07 UTC

All eight size/family configurations completed successfully. Ordinary
queue PID 11956 and its observed descendants exited; exec 80814 returned
zero. Every configuration has a strict `_complete_audit.json` alongside
its separate bank under `size_family_unroll_e1`. No failed warm statuses
or NaN-mask mismatches occur; nonzero warm differences remain recorded
and are not numerical-equivalence certificates. Source and compiler
epochs have not changed. Main and the frozen worktree are clean.

Pushed commits now include `664ff6ff` (source snapshots, carveout probe,
placement adapter), `80e0356c` (reviewed placement evidence), `def99d7c`
(buffer effects and instruction-sharing design), and `bf0e475b` (exact
physical dispatch capacity). PR #912 remains ready/open; its updated
body is saved as `research_pr_body_20260905_v2.md`, with full AB stdout.

The buffer observer passed final independent semantic review at SHA
`dfc6ddcc50cbf914e0f8ca81389d07694cfc8fe4e857c84b0a2fe648d23708c5`.
Its nine actual v4 cases and extra shared-alias case have zero native
overloads. Receipt: `verification/buffer_descriptor_independent_20260905.json`.
Its retention quantity includes lexical control unions and ordering
dependencies: it is not a native-register prediction or minimum.
The dispatch component passed root independent source/dimension checks
and six exact saved-profile bounds at SHA
`9e16c448a441ed177f2466befd314736f4f16ed33d6bc7bff30bac1f1fb9a479`.
Receipt: `verification/capacity_component_independent_20260905.json`.
It retains caller proof obligations, unknown symbolic native helper work
and warp visits. No source-operation or iteration mean is inserted.

The same-cubin ordinary carveout preference experiment completed at
`stage_base_carveout_ordinary_e1`, PID 48320 / exec 16911 (both exited).
All 36 measurements exceed 20 ms, native artifacts and warm arrays match,
and the original preference is restored with clean cleanup. The medians
for control8/shared64/control8_repeat are 21.653712/21.647456/21.643424 ms.
Its two completed profiles are `profile_a_carveout_control8_e1` and
`profile_b_carveout_shared64_e1`. Both report actual shared configuration
8.192000 Kbyte, despite setter/readback values 8/64. The desired physical
contrast was not established; do not use this to reject a cache effect.
The original script is preserved as `benchmark_source.py` in the ordinary
directory and both profiles. hardware_provenance is independently auditing
these raw results. placement_audit is constructing a separate unused
dynamic-shared-reservation control, keeping the same binary and register-
limited four blocks/SM; it must prove actual 8/64 KiB after profiling.

CURRENT GPU OWNER is elevated host 48004 / worker 31904. Release is true,
ordinary owners/children are gone, and the worker runs serial profiles.
There are 28 jobs in this batch: two hint-only carveout profiles, two ERK
source-replay retries, six ROS profiles, seven `z_` profiles, and eleven
`zz_` family/solver profiles. The latter cover chain32 Kvaerno3 BiCGSTAB,
both Kvaerno5/LU sizes and L96 Radau5 BiCGSTAB. Every requested policy has
a strictly eligible ordinary observation. See
`verification/family_solver_queue_additions_20260905` for exact requests.
Never start a native compile or ordinary GPU probe while this queue owns
the GPU. Read live status; no new elevation is required.

All queued profiles include exact local LD/ST L1 lookup-miss sums, each
in sectors, verified by an explicit AD104 metric-catalog query. Their
catalog and prior queued requests are preserved under `verification/`
`ad104_local_lookup_miss_metrics_20260905.txt` and
`local_miss_queue_revision_20260905`. The first release attempt included
unsupported metadata and was rejected without a child launch; it is
saved as `carveout_profile_release_rejected_20260905.json`. The corrected
release uses only external_jobs_finished/blocked_process_ids/
allow_profile_jobs. A stale last_error can retain that old schema error
while current jobs succeed; inspect individual result states.

The two ERK profile retries completed successfully with exact source,
native and own-reference state gates. CPU saved-report analyses are
running in exec 27926 under `verification/placement_profiles_e1_analysis`.
Four additional ROS saved-profile analyses are running in exec 65761
under `verification/size_profiles_e1_analysis`. The first two audited
L96 ROS profiles already pass: full/count-two exact warp work is
31,310,979,840 / 31,358,313,104, executed-address union 216,720 / 134,320
bytes, local LD warp work 714,329,172 / 57,177,557, local ST warp work
687,755,525 / 57,184,673. Their GCC instruction misses are
104,001,585 / 2,860,042 requests. Local-load L1 misses are
65,266,217 / 971,402 sectors. Exact software/hardware instruction totals
agree with zero residual here. This does not yet apportion the ordinary
speedup between reduced local traffic and instruction-fetch pressure.

CPU lanes: unroll_evidence is preparing a source-bound late-unroll
observer, then independently reviews hardware_provenance's dual-stream
probe. hardware_provenance reviews the late tool and reservation control.
The dual-stream source/worker are CPU-validated but await independent
review and a serial native admission gate. These probes are not yet
hardware findings. No final defaults or complete heuristic are claimed.

## Checkpoint at 2026-09-04 13:46 UTC

All 28 prior profiles completed with successful capture/import. All 26
solver/placement profiles have completed saved-report CPU analyses;
the two carveout-hint profiles remain the failed physical contrast
documented in CARVEOUT_HINT_EVIDENCE.md. Six more profiles completed:
`profile_a_reservation_{baseline,reserved}_e1` and
`profile_fabbri_{radau3,radau5}_{full,winner}_e1`. The latter four still
need SourceCounters exports and solver_profile_analysis.py audits.
Their exact staged requests and prerequisite audit are retained in
`verification/reservation_fabbri_requests_e1` and
`verification/fabbri_counter_prerequisite_audit_20260905.json`.

The same elevated host 48004 and worker 31904 remain alive. Root disarmed
the empty queue at 13:46 for the next reviewed native gate. Release is
false; no root ordinary/native owner remains active. Do not start new
elevation. Two CPU raw-report audits may still be completing in exec
56428; they only import saved NCU reports and disassemble saved cubins.

Pushed commits include 9dca34ca (compiler/reuse probes), 86ac3f26
(unused reservation control), and 8e888a96 (L96 Kvaerno3 evidence).
SIZE_FAMILY_EVIDENCE.md passed root review of 20 table rows and units:
`verification/l96_kvaerno3_prose_review_20260905.json`. Its final hash is
9a62e2dba1e4a9ec708e0056bcc07190c4a1fc10a0f224709a5e38932696a05f.
Both-rolled and Krylov-count4 have identical per-run counters and saved
states, but their cross-block baseline-normalized time ratio is 2.754476.
K4 executes 10.364595% more warp instructions and much greater local
traffic, while instruction-delivery counters improve versus full.
The exact scope, raw counts and residuals are retained in that document.

Unused reservation ordinary control completed with exact binary, state,
one CUfunc, 255 registers, 688-byte local frame and four blocks/SM.
Data: `stage_base_reservation_ordinary_e1`; exec 31699 exited zero.
All 36 samples exceed 20 ms. Baseline/reserved/repeated-baseline medians
are 21.622096/29.008528/21.648864 ms. Strict CPU ordinary admission is
`verification/reservation_ordinary_admission_e1/result.json`, with root
resource/membership recount `verification/reservation_ordinary_root_recount_e1.json`.
Reservation source e31119f8cacef5032b0ef185036169ccbc0dbc6e384ecce54b8f395117f073e6
passed independent review in `verification/reservation_independent_20260905`.
Shared allocation is 128 bytes on this Ada device, confirmed by both
NCU's architecture table and CUDA13.3 cuda_occupancy.h:620-642. The
256-unit allocation applies to registers per warp, not Ada shared bytes.
Thus baseline allocation is 1152 bytes, reserved allocation 9472 bytes.

Root raw reservation report audit source is
`verification/reservation_saved_profile_audit_e1.py`, final hash
24952440a28273c84015af89b4a2ad9a45ed9ef5f6a3b1d741be15d87d5123d0.
Its e3 reports found actual shared capacity 8192/65536 bytes and exactly
5,107,818,264 warp instructions in both profiles (hardware residual 51).
Final e4 adds explicit execution-root/report-hash/command/source gates;
hardware_provenance is independently reviewing it. Preserve the earlier
failed missing-constant and name-display audit snapshots. Do not treat
their aborted runs as PASS. The new e4 output is
`verification/reservation_saved_profiles_e4`; no causal model constant
has been fitted from this control.

Two native probe gates exposed concrete harness gaps; raw failures stay:

- `unroll_stages_native_e1/radau5_count2` failed both byte gates. Original
  LTO relink omitted the bank's forced verbose/no_cache options; its
  executable/data sections match, but .note.nv.tkinfo lacks `-v`.
  More importantly the bare compiler replay omitted frozen CuBIE's
  `_mlir_compat` external-shared pre-codegen hook. The replay leaves
  `__dynamic_shmem__0` internal. The diagnosis is in
  `verification/lto_replay_hook_audit_20260905`; this path defect still
  needs a controlled byte-identical retake. unroll_evidence prepared v5
  with exact frozen hook and harness provenance, awaiting independent
  hardware_provenance review. Prepared inputs: `unroll_stages_prepare_v5`.
  New external CUBIE_CACHE_DIR must point to fresh output/cubie_cache,
  plus the existing external natural NVVM-dump path and CUDASIM=0.
- `instruction_sharing_compile_e1` compiled without launches but failed
  unchanged native admission. Each 82000-byte stream has 5120 FFMAs,
  a MOV from a constant parameter, UIADD3/ISETP, a predicated exit CALL
  and unconditional backedge. Root diagnosis is retained alongside its
  artifacts. hardware_provenance changed only the two runtime-count
  backedges to `bra.uni`; independent unroll_evidence review is pending.
  The admission gate and worker remain unchanged. Run a fresh compile
  gate only after review, retain any failure, and do not widen the opcode
  allowlist without a valid control/traffic argument.

placement_audit is implementing a typed source-value graph and legal
schedule/frontier certificate on actual generated helpers and an ERK
step. Its proposal `verification/source_retention_mechanism_20260905`
records old unary/cast copy-identity and statement-dependency errors.
This component must separate value and ordering edges and is not a
native register estimator. No compiler wheel or production source has
been changed. Main and frozen source remain outside the edit scope.

## Checkpoint at 2026-09-04 14:27 UTC

The elevated host 48004 and worker 31904 remain alive. The queue is
empty and release is false after the completed capacity captures and a
failed sharing profile gate. Do not start another elevation. No native
or GPU work remains active at this checkpoint. Main is clean and
untouched. The frozen ff3a567f measurement tree and installed compiler
remain unchanged.

All six corrected compiler replays in `unroll_stages_native_e2` passed
both exact LTO/cubin gates. Independent audit:
`verification/stage_replay_independent_20260905/receipt.json`;
root prose/table review: `verification/stage_replay_prose_review_20260905.json`.
STAGE_REPLAY_EVIDENCE.md records the actual replay LLVM metadata and
diagnostic PTX boundary. Count2/count4 differ by one LLVM metadata digit
in each family. Radau5's three diagnostic PTX files and original cubins
are identical; the responsible optimization is still unresolved.

All four Fabbri profiles passed saved-report core analysis and an
independent raw-array/PC/opcode/timing-line audit. See
`verification/fabbri_profiles_independent_20260905` and
FABBRI_PROFILE_EVIDENCE.md. R3/R5 winners execute 74.712%/119.016% more
warp instructions. Cross-policy numerical workloads differ. The R5
winner's block-0 samples vary widely; its median ratio is about 0.868
against that block's full baseline. Its GCC instruction misses increase
despite reduced sampled no-instruction stalls. Preserve the R5 full
83,385-sample reconciliation residual.

The original 4/8448-byte unused reservation profiles passed final e4
review (`verification/reservation_saved_profile_independent_final_20260905`).
RESERVATION_EVIDENCE.md records exact work and actual 8/64 KiB capacities.
The new seven-slot ordinary capacity sweep completed in
`stage_base_capacity_ordinary_e1`: 140 solves, 84 measurements, 14 exact
warm snapshots, one unchanged binary and four resident blocks/SM.
Baseline, capacity16, capacity32, capacity64, original64, capacity100,
and repeated-baseline medians are respectively 21.636929, 22.537008,
25.091647, 29.077200, 29.058960, 31.867424 and 21.665904 ms.

All six `profile_capacity_<arm>_e1` captures completed. Root exported
their SourceCounters and ran the independently authored pure CPU audit
`verification/reservation_capacity_saved_profile_audit_e1.py`, SHA
ad5810ba72d0af9768153259082a08e672aa70012dcdcfb5cbcb6cb5c2ec62d0.
Output `verification/reservation_capacity_saved_profiles_e1/receipt.json`
is PASS_ALL_SIX. Actual capacities are 8/16/32/64/64/100 KiB, with
identical 5,107,818,264 exact warp instructions and 1,943,004,895 local
load sectors. Local-load L1 miss sectors rise from 493,809,615 to
1,860,539,696. The two 64 KiB reservations yield closely matching traffic.
These capacities were measured in profiles; ordinary timing remains
separate. No latency or replacement parameter is fitted.

Peer review found a saved-reader defect in capacity source 5b66e9bf:
its ordinary reader checked duration/manifest against its own rows but
omitted direct original-cohort joins. The actual e1 data passes explicit
external joins, and the saved-profile audit above includes them. The
reproduction and external qualification are in
`verification/reservation_capacity_independent_20260905`. Do not call
that source API fully verified. hardware_provenance is repairing the
reader's original duration, run count, manifest, construction, input
and compile/geometry joins. The old source is preserved byte-exact in
each e1 profile's `benchmark_source.py`. New execution must use reviewed
repaired source and fresh output directories.

Exact dual-stream CFG admission passed independent review at
`verification/instruction_sharing_cfg_independent_20260905`. Source is
21db7e0df15667715095673da635ed616caa7f0ebd90ca9270392a525bee060b.
The first two rejected native gates remain unchanged; fresh
`instruction_sharing_compile_e3` passed without launches and reproduces
their exact cubin 495962af77fdd626996e2dd09b6c13b60e9abd398a217398a970b009bc96aa9e.
`instruction_sharing_ordinary_e1` passed 72 mirrored measurements plus
three calibration launches with exact output/SMID/native gates. N=4096
medians A/B/mixed are 34.958687/34.942736/48.454607 ms; 2N medians are
69.598125/69.541553/96.685169 ms. The independent CPU ordinary receipt
is `verification/instruction_sharing_ordinary_independent_20260905`.
This is not yet a cache-domain result.

`profile_sharing_all_a_e1` failed before compilation/launch: its only
identity difference was serialization order of the fastmath set. The
unchanged hardware helper turns sets into unsorted lists. placement_audit
is repairing only the worker's normalization to canonicalize actual
sets while preserving ordered lists/tuples. It needs independent CPU
review, a fresh ordinary run and profiles; do not retry e1 directories
or weaken list identity generally. The existing requests in
`verification/sharing_profile_requests_e1` name the old worker epoch.

The source-value graph covers actual Lorenz/RK4 and chain32/Vern7 with
zero native overloads. Independent review confirmed all frontier/cut
witnesses and the fourth Lorenz RHS source optimum of 13. A Boolean
index was incorrectly accepted as an integer; the author is applying a
bounded rejection before final receipts. This graph remains a typed
source model, not a native register bound. No family default or final
pre-compile heuristic is claimed complete.

## Checkpoint 2026-09-04 14:46 UTC

The source-value graph's Boolean-index repair and both completed actual
source certificates passed independent verification. Final code SHA is
7209fcad89e1aa8e6e59f8cac992c5a1728456c97cdf284179c98ea3afa0d0da;
document SHA is
fcdb533f270fe89ced3917495f64f645d4d93862ba1e5bfe95378db705578854.
Receipt: `verification/source_value_graph_independent_v3_20260905`.
Commit ac20420b is pushed to the existing ready PR. It is a source
certificate component, not a native GPR predictor.

Capacity reader repair 7b692fc56d82dcfcc7ed16cb7f8d3fd73529c9dc2585347c25d333c752c0853b
passed independent CPU review in
`verification/reservation_capacity_repaired_independent_20260905`.
Fresh `stage_base_capacity_ordinary_e3` completed 140 solves with clean
restoration. The e2 directory retains a pre-native environment mismatch:
PowerShell SetEnvironmentVariable with null left empty CUDA path entries.
Removing the named process environment entries restored exact original
identity; no comparison gate was relaxed. The repaired-source baseline
profile `profile_capacity_baseline_e3` completed with one kernel and
successful capture/import through the same elevated host. Its separate
saved-report audit is pending. The six e1 profiles retain their already
qualified immutable source and are not overwritten or relabeled.

The sharing worker's set normalization repair passed independent CPU
checks and fresh ordinary execution in `instruction_sharing_ordinary_e2`.
The subsequent all-A profile exposed a different physical failure:
`profile_sharing_all_a_e2` reports 65536 shared bytes and capacity for
six blocks/SM, despite the ordinary driver's one-block query under a
zero-percent carveout preference. Its 112-block grid is only one third
of a theoretical full-capacity wave. Exact repeated-body counts and
arrays pass, but e1/e2 ordinary actual residency remains unobserved and
the asserted two-wave geometry is unqualified. No cache-sharing finding
is admitted from those timings. The other e2 profiles were not run.

The root cause is frozen hardware_probes._geometry's occupancy search
under a carveout preference that does not constrain the actual launch.
A separate worker repair derives a reservation excluding a second block
even at maximum SM shared capacity: with 102400 bytes, 128-byte allocation
units and 1024 driver-reserved bytes, minimum dynamic 50177 rounds to
51328 per block. Two blocks require 102656 bytes. This repair is awaiting
independent CPU review and fresh native/ordinary/profile evidence; it
does not alter the frozen helper or earlier records. Existing fine
instruction-supply profiles already reported 102400-byte capacity and
must be judged by their own retained geometry, not this failed contrast.
The independent maximum-capacity recheck is
`verification/icache_capacity_exclusion_peer_20260905.json`: all fourteen
retained ordinary/profile rows satisfy the two-wave bound. Allocations
34176 and 20608 bytes exclude a third and fifth block respectively at
102400 bytes, independently of the preference hint.

The elevated host remains PID 48004 with worker 31904, idle and disarmed
after the capacity baseline capture. No new elevation is needed. Root
owns the serial GPU lane; agents are preparing the corrected sharing
adapter, direct-pointer latency probe and mechanistic model design on
CPU. The model design must produce explicit hardware-based estimates
for supported regions, with stated compiler/cache/scheduling assumptions
and holdout checks. Rigorous lower bounds alone do not rank candidates.

## Checkpoint 2026-09-04 15:48 UTC

The repaired capacity reader completed its fresh end-to-end baseline
profile audit. Receipt
`verification/reservation_capacity_saved_profiles_e3/receipt.json` is
`PASS_REPAIRED_BASELINE_API_AND_PROFILE`: the complete 140-solve ordinary
bank and baseline's exact original joins pass, actual shared capacity is
8192 bytes, four blocks/SM, and source warp work is 5,107,818,264. Its
unchanged hardware/source residual is 51. The six e1 capacity contrasts
remain the already-qualified historical data under their original source.

The repaired sharing worker passed independent review and fresh ordinary
and all three profile arms. See `INSTRUCTION_SHARING_EVIDENCE.md` for
the exact work, actual one-block/two-wave resources and cache metrics.
Receipt `verification/instruction_sharing_profiles_e3/receipt.json`
confirms every native repeated instruction and fixed path. All three
profiles use the same binary and report 65536 shared bytes. N=4096
ordinary medians A/B/mixed are 38.380001/38.221935/97.396126 ms; mixed
GCC instruction misses are 155,765,178 versus 1,696/9,223 for uniform
selection. This supports instruction-delivery pressure, not an identified
physical cache domain, isolated miss latency or universal 128 KiB penalty.

Unchanged-instrument mask-2 and mask-32 ordinary banks completed and
passed the separately reviewed mask-aware CPU audit. Their mixed N
medians are 91.602322 and 91.265568 ms. At this checkpoint the mask-32
mixed profile completed through the existing elevated host, with actual
one-block/two-wave geometry, 68,102,624 GCC instruction misses and
907,884,374 ICC miss cycles; its full saved-source audit is pending.
Six exact requests for both masks are validated under
`verification/sharing_profile_requests_masks_e1`. The remaining five are
queued. Check actual session status/results before dispatching more.
Sharing controller and worker remain frozen at 21db7e0d and 1dde5b8c.

Direct-address latency source 9de6a49cd124268df63b06f6e2561c650c8803a07f9a6515f56b8c787ec6b6ac
passed independent source/native review and is committed/pushed as
ebae9546. Shared8 compilee2 exactly reproduces the preserved e1 cubin
fab351b2a44aeef8906a2dac42295fcb63f0ad05c7afc0c22a2b133c1797c2fc.
Its 257 dependent LDS edges include register renaming; counter, both
pointer-dependent clock guards and final uniform CTA barrier are proved.
Twenty registers, 8192 static shared bytes, 1024 threads and 112 blocks
give two waves while inactive warps remain at the barrier. The original
e1 failed native-admission record remains unchanged and has no launches.

`latency_shared8_ordinary_e1` completed 25 launches: one calibration and
24 measurements at N=32769 and 2N. Exact raw arrays, inputs, source,
clock identity/duration and shutdown pass the CPU audit in
`verification/latency_shared8_ordinary_audit_20260905`. The median of the
twelve paired median-cycle differences is 24 cycles per added dependent
load; four individual differences are lower because their N samples have
slower CTAs. Every measurement's minimum is `24 * loads + 22` cycles. The
two matched profiles `profile_latency_shared8_n_e1` and
`profile_latency_shared8_2n_e1` completed capture/import and source export
using the same elevated session. The first reports 16384 shared bytes,
one resident block, two waves, exactly 943,251,568 LDS and shared
wavefronts, and zero bank conflicts. Full N/2N saved-profile audit is
in progress; this is a measured chain interval, not a solver fit.

The first global direct-address compile,
`latency_l1_quarter_compile_e1`, is preserved as a no-launch rejection.
Its loop-carried pointer uses two native register copies before the
first LDG, plus YIELD and a uniform constant load per loop; consecutive
loads still form a direct chain. The author is preparing a bounded
native dataflow/control-flow admission repair. No old source snapshot
or failed result is edited. Fresh source review and fresh compile are
required before any global ordinary measurement.

`NATIVE_PLAN.md` and `native_plan.py` implement the first conditional ERK
lowering/allocation/cache/service component. Fresh chain16/17/18 source
graphs predict 239/253/267 no-spill words under the explicit promoted
source-order scenario, with modeled spill onset at chain18. These
dimensions were chosen before native labels. Fourteen actual local/shared
plans exist; independent review found and prompted repairs to exit/spill
certificate validation, without changing the computed plans. Final source
and receipt hashes are still being frozen; do not treat the component as
the complete model or release native holdout comparisons before that
review. Actual implicit-family LU/iterative and rolled-loop adapters
remain active work, with separate main/error solver widths and symbolic
iteration regimes required. No fitted timing or register multiplier is
introduced.

Main and the ff3a567f measurement tree remain untouched. Root owns the
serial GPU/native lane. The persistent elevated host PID 48004 and worker
31904 are reused; there has been no additional UAC prompt. The remaining
research objective is active and the existing weekend heartbeat remains
in place. Always inspect `gpu_release.json`, queue and worker status
before launching ordinary work; the profiler is currently armed for the
remaining mask controls.

## Checkpoint 2026-09-04 16:34 UTC

All nine sharing profiles, including masks 2 and 32, have passed their
full saved-source and physical-geometry audits. Each uses the same cubin,
one resident block and two waves. Mixed GCC instruction misses are
155,765,178 / 69,717,051 / 68,102,624 for masks 1 / 2 / 32; their ordinary
N medians are 97.396126 / 91.602322 / 91.265568 ms. Mask 32 selects 512
A and 384 B warps; the other masks select 448 each. This is a measured
instruction-delivery contrast, not a virtual-SMID-to-physical-domain map
or an additive fitted miss penalty. `INSTRUCTION_SHARING_EVIDENCE.md`
records the exact counters and receipts, including the imbalance.

Shared8 direct-chain N/2N full profile review passed. Its lower envelope
is `24 * loads + 22` cycles. Every paired minimum difference gives 24
cycles/load; only the median of the paired median differences is 24.
Four individual paired median differences are lower because the N rows
contain slower CTAs. This correction is retained in `LATENCY_EVIDENCE.md`
and the independent median-qualification addendum. No raw row changed.

Global32KiB `.ca` source bd6172f8 passed the pointer-copy/control repair
review, fresh compile, ordinary N/2N and both counter captures. Receipt
`verification/latency_l1_quarter_profile_independent_20260905/receipt.json`
has SHA a1295ae612c37afa05847ab3e56a79456cc7b9de4d869f9739d77dba4fcdb12f.
All 336 native PCs and 25 raw arrays are checked. Both captures report
8192 shared bytes, one block/SM and two waves. Timed global L1 lookup-hit
fractions are at least 99.9939085448% and 99.9969542724%, conservatively
assigning every whole-launch miss to the timed chain. Every ordinary
minimum and median is `8860 * repeats + 9` cycles. The 257-load body has
eight administrative instructions; its 8860/257 interval is not isolated
LDG latency. Hardware/source total-warp residuals 3,784,816 / 7,454,944
remain unresolved despite equaling the counted YIELD visits numerically.
Global pointers are uint64 and shared pointers uint32; comparing the two
intervals is not a controlled memory-space penalty.

The unchanged latency controller's 33-load global control passed compile
and independent native review in
`verification/latency_l1_quarter33_compile_independent_20260905`. Cubin
96a474ae3a1b07e145f2974cf6e94bfd6f8d37e430492d168636ddde82bdae4f
has 26 registers, 33 dependent LDG plus seven administrative instructions,
a GPR counter and direct conditional backedge. Ordinary e1 completed 26
launches, including two calibrations. N is 65539; all ordinary intervals
are `1230 * repeats + 23` cycles, giving 1230/33 per added load. Its
matched counter captures are pending. The control changes native loop
administration, so two body lengths do not identify an intrinsic latency
by blindly subtracting one fitted fixed overhead. Root derived separate
33-body readers and requests under
`verification/latency33_control_adapter_v2_20260905`; peer review checks
the specific PC map and preserved admission gates before capture.

The frozen NativePlan f547ee91 passed its final bounded ERK review. The
holdout observer committed as d7ce0ff4 freezes 48 predictions for six
chain16/17/18 local/shared constructions, both 32/64-thread geometries
and separate promotion/contraction hypotheses. Source preparation e2
manifest b5ffb329e1eb231f0f56ef1a835911e3f08d799fc218ab6ae98c581b2c43fad2
passed independent release; all predictions are byte-identical to e1.
The first native e1 attempt failed before compilation because the external
worker needed the research repository root on PYTHONPATH. The corrected
order starts with frozen `src`, then research root. Native e2 chain17
shared compiled successfully and passed postcompile source/config hashes,
then failed artifact extraction: installed MLIR CodeLibrary has no
`get_cubin`. No kernel launch occurred. The author is repairing this
observer against the installed API and preserving the 48 predictions;
fresh source/native receipts are required, and e1/e2 failures stay intact.
Implicit-family role/width/counter adapters remain active CPU work.

`ARITHMETIC_SERVICE_DESIGN.md` passed independent review and was committed
with the global evidence as c579b6e7. Its separate instrument uses runtime
one/full-warp populations, exact recurrences, retained clock guards and
final barriers, and 33/257 controls. It must distinguish achieved dense
throughput from architectural initiation rate. Implementation is active;
no arithmetic native or GPU result is admitted yet. This work must supply
complete, explicitly qualified service scenarios for candidate estimates,
not stop at hardware lower bounds.

Main and the detached measurement tree remain clean and untouched. The
same elevated host PID 48004 and worker 31904 are running; no new UAC is
needed. At this checkpoint ordinary work has exited and the profiler is
disarmed; inspect actual status/queue before changing ownership. PR #912
is ready/open with its exact AB gate suffix preserved. The weekend
objective and existing quiet heartbeat remain active; the full heuristic
and family defaults are not complete.
