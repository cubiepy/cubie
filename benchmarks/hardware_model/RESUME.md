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
