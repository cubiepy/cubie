# Unused dynamic shared reservation on one baseline binary

The hint-only control retained exact native/state identities and 36
ordinary samples of at least 20 ms. Its separate profiles
`profile_a_carveout_control8_e1` and
`profile_b_carveout_shared64_e1` both reported 8192 bytes of actual shared
configuration. Setter/getter preferences of 8 and 64 percent did not
establish a capacity contrast. Those results and `carveout_probe.py`
remain unchanged.

`reservation_probe.py` instead changes the dynamic shared byte argument
of the original local baseline launch. It uses one Solver, one native
specialization and one loaded CUfunction. It does not set a carveout
preference. The initial preference must survive every solve unchanged.

## Physical derivation and admission

The sole supported case is the completed
`stage_base_placement_e1` cohort: chain32, Kvaerno3, LU, policy
`u11111111`, 262144 runs, duration 1.6, and 64 threads per block. Its
accepted shared placement resolves `stage_base` to 32 FP32 elements.
The package applies a four-byte skew to a nonempty even shared extent.
Thus the accepted allocation requests `(32*4+4)*64 = 8448` dynamic bytes
per block. The original local baseline requests four dummy bytes. The
control adds 8444 unused bytes; it does not create a shared stage buffer.

The target is copied from that accepted layout and checked against the
reviewed shared profile's exact source/configuration/native identities.
No byte count is selected by optimizing a timing result.

`ncu_occupancy.get_gpu_data(8, 9)` supplies the architecture's 128-byte
shared allocation unit, supported shared capacities and register
granularity. The exact Python wrapper and native table library hashes
are retained. This is a CPU architecture-table query. The reviewed
profile records a 1024-byte driver reservation per block, subsequently
checked against a live device attribute during explicit execution.

These units must remain distinct. The installed CUDA 13.3
`include/cuda_occupancy.h:620-642` also selects 128 shared bytes for
compute-major 8; its 256-byte shared unit applies to earlier architectures.
Its separate register-allocation routine (`:679-701`) uses 256 registers
per warp. A prior summary attributing 256-byte shared rounding to the Ada
helper was mistaken. There is no disagreement between these current
calculators for CC8. The observed baseline allocation is 1028 rounded to
1152 bytes; rounding it to 1280 with a 256-byte shared unit would be wrong.
The target 9472 is divisible by both, so that error would not change this
particular reservation target.

| Quantity | Baseline | Reserved |
|---|---:|---:|
| Requested dynamic shared bytes/block | 4 | 8448 |
| Driver-reserved bytes/block | 1024 | 1024 |
| Rounded allocated bytes/block | 1152 | 9472 |
| Bytes for four resident blocks | 4608 | 37888 |
| Smallest supported compatible capacity | 8192 | 65536 |

The architecture calculator independently reproduces those rounded
allocations. For each capacity large enough to admit a block, its
integer block/resource results are retained. At 32768 bytes, the
reserved arm fits three blocks; at 65536 bytes it fits the original
four blocks, limited by registers. Both arms allocate 16384 registers
per block for the original 255-register/thread binary.

The 65536-byte value is the smallest capacity compatible with four
blocks. Reserving 8448 bytes alone does not prove that the driver chooses
four blocks or 65536 bytes. Driver occupancy queried at the requested
reservation must remain four blocks, and separate profiles must report
the actual capacity and launch occupancy limits. Ordinary receipts
deliberately retain `ordinary_actual_carveout_observed=false`.

## Native and launch proof

- Frozen `src/cubie/batchsolving/BatchSolverKernel.py:823-837` accepts the
  dynamic shared value from `limit_blocksize`; lines 855-872 pass it as
  the fourth dispatcher launch argument. Existing
  `placement_landscape.py:691-695` pins only that per-instance return.
  The reservation control uses this already approved instrument.
- `BatchSolverKernel.py:975-985` captures the shared extent and run
  stride; lines 1059-1068 construct the per-run shared view. The actual
  cached baseline kernel closure has `shared_elems_per_run=0` and
  `run_stride_f32=0`. Host shared bytes and skew are also zero. Closure
  inspection uses the cached property and must leave zero overloads.
- `BatchSolverKernel.py:1318-1342` defines the FP32 skew rule used in the
  target calculation. The accepted shared registry slice and allocator
  branch are retained in the original cohort, including the local
  accumulator alias parent.
- Installed `numba_cuda_mlir/descriptor.py:1933-2068` carries configured
  shared bytes to `LaunchConfiguration`; lines 2220-2224 expose tuple
  configuration. A launch-sensitive dispatcher could choose different
  code. The probe rejects `_launch_config_enabled`, which includes both
  required launch configuration and launch-sensitive extensions
  (`descriptor.py:1672-1673`). It never changes that compiler state.
- The entire saved baseline SASS text has 23936 instructions. The audit
  requires one complete contiguous text section, parses every addressed
  instruction, admits only the reviewed explicit non-shared opcode set,
  and resolves every local relative CALL target inside the same text.
  Its memory operations are global, local and constant operations;
  shared access count is zero throughout the binary, not just along an
  observed execution. Unknown opcodes or unresolved calls fail admission.

Fresh execution must reproduce exact original cubin, diagnostic PTX and
decoded SASS bytes, entry, registers, local frame, generated source,
compiler identity and input arrays. Fresh SASS undergoes the same whole
binary audit. Static shared bytes must remain zero. Before and after
each solve the loaded handle, one overload, native hash, registers,
frame, preference and pinned geometry are checked. Only the dynamic
byte field may differ. The planned launch is grid `(4096,1,1)`, block
`(1,64,1)` with four resident blocks/SM and more than two full waves.
Actual dimensions remain a mandatory profile check.

The relevant public hardware definitions are the
[Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#unified-shared-memory-l1-texture-cache),
which documents supported capacities and per-block driver reservation,
and [CUDA's shared/L1 configuration description](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#configuring-l1-shared-memory-balance),
which permits driver selection despite preferences. The installed
Nsight Python API documents `get_gpu_data` fields at
`extras/python/ncu_occupancy.py:37-96`. All numerical derivations use
retained queried values, rather than inferred cache capacity from timing.

## Ordinary and profile protocol

CPU preparation is the default. The explicit `--execute` gate compiles
the baseline through the unchanged placement helper and then collects
two blocks of six mirrored samples for each of
`baseline/reserved/baseline_repeat`. The two baseline slots share the
same loaded function. Every block begins with full warm state/status
NPZs for all three slots, required to match the original accepted warm
arrays exactly. Every solve must retain finite state, successful status,
one chunk, identical input hashes, duration and run count. All 36
measurement rows must reach 20 ms at the original fixed duration.
Short samples fail; duration is never extended to change the workload.

The original baseline reservation is restored and observed before
restoring the original per-instance launch method and closing the
Solver. The temporary generated-source cache override is restored.
Any cleanup failure invalidates the attempt.

Profile mode requires a completed ordinary reservation cohort from the
same source/helper identities, physical plan and original cohort. It
revalidates every raw sample, mirrored membership, all warm NPZs,
native snapshots, treatment sequence, artifacts and restoration. It
then performs exactly one snapshot solve for `baseline` or `reserved`.
Profile timings are diagnostics and cannot enter ordinary timing data.

Example, with the frozen source/harness already selected on `PYTHONPATH`:

```powershell
python benchmarks/hardware_model/reservation_probe.py --cohort-dir C:\local_working_projects\cubie-notes\hardware_unroll_placement\stage_base_placement_e1 --shared-analysis C:\local_working_projects\cubie-notes\hardware_unroll_placement\verification\placement_profile_analysis_independent_20260905\shared\analysis.json --shared-analysis-sha256 414de60dfa255051c40184c88b3ef9453a6a5682d3eeaab16e496b0207f3af8c --out FRESH_OUTPUT
```

The analysis SHA is an immutable reviewed input identity. Add
`--execute` only in the coordinated GPU lane. For each separate profile,
add `--ordinary-dir COMPLETED_ORDINARY --profile-arm baseline` or
`reserved`, with fresh output and the reviewed elevated NCU session.
This tool never starts Nsight Compute.

Require successful profile/import exits, exactly the intended one solve,
exact command/script/native/source/input/state joins, and preserved
exported units. `profile_metric_gate()` is a reusable CPU constraint
check for actual launch metrics after those joins. It does not itself
establish report identity. It requires exact dynamic/static/driver/
allocated bytes, capacity 8192 versus 65536, original grid/block/SM
count and registers, four-block register and total occupancy limits,
and at least two waves. For example, `8.192000 Kbyte` means 8192 bytes;
`65.536000 Kbyte` means 65536 bytes.

Even successful profiles establish actual configuration only for their
own launches. The ordinary run remains an unused-reservation experiment
with separate profile evidence. Any unmatched actual capacity or
occupancy leaves the intended cache-capacity contrast unresolved. No
pre-compile default, fitted slowdown or latency coefficient is supplied.
