# Same-binary shared-capacity sweep

`reservation_capacity_probe.py` extends the accepted unused-reservation
experiment through a separate instrument. The frozen
`reservation_probe.py` and its raw cohorts remain unchanged. The sole
case is chain32, Kvaerno3, LU, `u11111111`, 262144 runs, duration 1.6,
block `(1,64,1)`, grid `(4096,1,1)`. Every arm uses the same Solver,
CUfunction, cubin, input arrays, state workload and launch geometry.

## Derivation

The [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#unified-shared-memory-l1-texture-cache)
documents a 128 KiB combined L1/texture/shared pool, supported shared
capacities 0/8/16/32/64/100 KiB, and a 1 KiB per-block CUDA reservation.
The installed Nsight architecture table independently supplies those
capacities and a 128-byte shared allocation unit. These are queried
inputs checked against the bounded AD104 case, not timing parameters.
`cuda_occupancy.h:1414-1416` rounds driver, static and dynamic bytes
together; `:620-642` selects a 128-byte shared unit for compute-major 8.
The baseline has zero static shared bytes and four register-limited
resident blocks per SM.

Let `U` be the allocation unit, `R` the driver reservation, `B` the
resident blocks, and `P` the preceding supported capacity. The next
required allocation unit is `A = U*(floor(P/(B*U))+1)`. The smallest
integer dynamic request rounding to `A` is `d = A-U-R+1`. The instrument
checks both `B*round_up(d+R,U)>P` and the opposite inequality for `d-1`.
Odd byte requests are intentional: the launch argument is a byte count,
and the complete baseline binary never accesses shared memory.

| Arm | Dynamic bytes/block | Allocated bytes/block | Required for four blocks | Minimum capacity | Nominal L1/texture complement |
|---|---:|---:|---:|---:|---:|
| baseline | 4 | 1152 | 4608 | 8 KiB | 120 KiB |
| capacity16 | 1025 | 2176 | 8704 | 16 KiB | 112 KiB |
| capacity32 | 3073 | 4224 | 16896 | 32 KiB | 96 KiB |
| capacity64 | 7169 | 8320 | 33280 | 64 KiB | 64 KiB |
| original64 | 8448 | 9472 | 37888 | 64 KiB | 64 KiB |
| capacity100 | 15361 | 16512 | 66048 | 100 KiB | 28 KiB |
| baseline_repeat | 4 | 1152 | 4608 | 8 KiB | 120 KiB |

The two 64 KiB arms separate requested reservation amount from a common
actual capacity, conditional on matched profile admission. The nominal
complement is subtraction from the documented combined pool; it is not
a measured usable cache capacity or an assertion about replacement.

The Nsight CPU calculator agrees with four register-limited blocks at
every intended capacity. At several predecessor capacities it is
internally inconsistent for these minimal byte requests. For example,
1025 requested bytes at 8192 capacity produces allocation 2176 bytes
and `allocated_blocks=4`, although four allocations require 8704 bytes;
its `unallocated_blocks` is 4294967295. The Python wrapper passes integer
byte fields directly (`ncu_occupancy.py:957-960`); its compiled internal
cause is unresolved. These raw rows are retained alongside the rigorous
`floor(capacity/rounded_allocation)` bound. They are not used to assert
predecessor occupancy. A contradictory calculator result at an intended
capacity fails preparation. Failed initial CPU preparation is preserved
at `verification/reservation_capacity_cpu_e1/result.json`.

## Identity and execution gates

The input includes the accepted original 4/8448-byte ordinary cohort.
The frozen helper reopens its raw rows, six warm NPZs, native artifacts,
treatment chain and cleanup before preparation. The new tool binds that
evidence and all imported helper hashes. Fresh generated-source replay,
original manifest/source/compiler identity, input arrays and zero native
overloads follow the existing placement protocol. Default execution
stops after host construction and closes the Solver.

Explicit execution must reproduce original cubin, PTX, SASS, entry,
255 registers/thread, 688-byte local frame and pinned geometry. The
frozen complete-native proof covers all 23936 SASS instructions and
zero shared accesses, plus zero shared extent in the generated closure.
No carveout preference is set. Every treatment observes the unchanged
preference, handle, one native overload, binary, frame and resources
before and after pinning the new byte argument and after the solve.
Driver occupancy must remain four blocks and at least two full waves.
The live driver reservation and maximum shared-memory attributes must
match the plan. No kernel compilation option changes between levels.

Two ordinary blocks each contain all seven warm snapshots, settlement,
then six measurement passes in forward/reverse alternating arm order:
84 measurements in total, with a baseline at each end of the forward
ordering. Every measurement must reach 20 ms at the original duration;
short samples fail. All warm states and statuses must match the original
accepted arrays exactly. All solves require finite state, successful
status, one chunk and unchanged inputs. Raw settlement times are retained
but excluded from measurements. No duration extension changes the work.

Cleanup restores and observes the original four-byte launch, restores
the per-instance launch method, closes the Solver and restores the
generated-source cache override. Any cleanup error invalidates the run.

## Separate matched profiles

Profile preparation requires the completed new ordinary sweep, with
exact reference-bank evidence, plan, helpers, raw order/membership,
warm arrays, native treatment chain and restored launch. A profile runs
exactly one state-snapshot solve for one of the six distinct arms. Its
timings are diagnostics. Use the existing serial elevated session;
the instrument never launches Nsight itself.

Both the returned original reservation reference and the new ordinary
sweep are independently bound to the original completed placement
cohort before admission. Duration, run count, manifest hash, complete
construction fields, generated-source bytes/function hash, retained
input arrays, compilation fields and pinned geometry must agree with
that cohort. The fixed ordinary protocol is checked for each bank, and
the reference's verified workload receipt must match the one retained
by the new sweep. Self-consistent edits to a saved bank cannot redefine
the original workload.

The e1 capacity bank and six profiles retain the earlier measurement
source with SHA256 `5b66e9bfaf44c30db10c72ee3b539e40099cac5d5c49170c11a159316b96dddf`
in each profile's `benchmark_source.py`. That loader omitted the direct
original-workload joins. Its actual duration is 1.6 throughout; an
independent saved-report audit rederives those joins from the original
cohort while interpreting the immutable source snapshot. Repaired
measurement runs require fresh output and matching repaired-source
ordinary evidence; earlier artifacts retain their original identities.

`profile_metric_gate()` delegates the unchanged frozen physical checks
with the selected arm's exact derived bytes and capacity. It requires
actual dynamic/static/driver/rounded bytes, target capacity, dimensions,
registers, SM count, four-block resource limits and at least two waves.
The driver can choose a different capacity or fewer blocks; either
outcome rejects that intended contrast. Preferences or ordinary timing
alone never establish actual capacity. Separate command/source snapshot,
report/export, full SASS, input/state and executed-work joins remain
mandatory before counter comparison.

Collect local load/store sectors and their L1 lookup-miss sectors:
`l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum` and
`l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}_lookup_miss.sum`.
Retain L2 read sectors, instruction counters, source executed counts,
LaunchStats and all occupancy limits. Keep sector, request and cycle
units separate. Compare traffic to the nominal L1/texture complement
only after actual capacity is admitted; no penalty coefficient, fitted
replacement rule or timing model is introduced.

## Invocation

Use the frozen epoch's `src` and `benchmarks` first on `PYTHONPATH`, then
the research `benchmarks`, with the original backend environment. The
required options are:

```text
--cohort-dir <rawroot>/stage_base_placement_e1
--shared-analysis <rawroot>/verification/placement_profile_analysis_independent_20260905/shared/analysis.json
--shared-analysis-sha256 414de60dfa255051c40184c88b3ef9453a6a5682d3eeaab16e496b0207f3af8c
--reference-ordinary-dir <rawroot>/stage_base_reservation_ordinary_e1
--out <fresh-output>
```

Omitting `--execute` performs CPU preparation. Add it only in the
coordinated native/GPU lane. For separate profiles also supply
`--ordinary-dir <completed-new-sweep>` and `--profile-arm` with
`baseline`, `capacity16`, `capacity32`, `capacity64`, `original64`, or
`capacity100`. All outputs must be fresh; earlier failures are retained.
