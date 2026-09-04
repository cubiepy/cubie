# Matched solver retakes

## Lorenz split-loop bridge, 2026-09-04

Both configurations use BiCGSTAB linear correction with Jacobi
preconditioning, `inexact_newton=True`, and `prefactored=True`:
`kvaerno3_bicgstab` and `radau_iia_5_bicgstab`. The conclusions below apply
to this iterative-solver regime. They do not establish LU defaults.

Frozen measurement revision: `ff3a567f`, in
`C:/local_working_projects/cubie-worktrees/hardware-epoch-ff3a567f`.
It includes the iteration-counter accumulation change and split loop flags.
The targeted cohort is `lorenz-split-bridge-e1`; its manifest records source
hash `4899b5cb04523177ed3cd3f1aef566591829ed026e064b951fdfbf629cfcef6a`,
MLIR backend, installed wheel versions, and compiler/scheduler identity.

Raw bank:
`C:/local_working_projects/cubie-notes/hardware_unroll_placement/lorenz_split_bridge_e1`.
Complete strict audit:
`C:/local_working_projects/cubie-notes/hardware_unroll_placement/lorenz_split_bridge_e1_complete_audit.json`.
The earlier `*_kvaerno_audit.json` is an explicitly partial snapshot.

Both configurations completed. Each has 16 eligible launch groups with six
timings per group, and no rejected groups. Every memory block contains
separate all-full and duplicate all-full Solver instances. Ratios below
use the minimum candidate timing divided by the minimum all-full timing
in its own block. They are descriptive matched ratios, without an invented
significance threshold. Aliases and repeated binaries are not independent
evidence for different compiler behavior.

All runs use float32, 262,144 trajectories, block size 64, and duration 2.0
for Kvaerno3 and 8.0 for Radau5, as recorded in their cohort protocols.
Every compiled geometry exceeds two occupancy waves. All 32 warm checks
across the configurations report zero failed runs and matching NaN masks.
Thirteen warm checks have nonzero state differences from all-full; these
are retained, without claiming an independent trajectory-accuracy test.

The first six groups remain full in every named Newton/Krylov contrast.
`False` delegates the relevant loop choice to the backend. `libnvvm`
delegates all eight groups. A counted directive describes the request;
the inspected cubin determines the actual native result.

| Newton / Krylov request | Kvaerno3 BiCGSTAB ratio | Registers | Whole SASS instructions | Radau5 BiCGSTAB ratio | Registers | Whole SASS instructions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full / full | 1.0000 | 96 | 26,320 | 1.0000 | 167 | 60,920 |
| Rolled / rolled | 1.0939 | 96 | 1,480 | 0.9696 | 137 | 1,576 |
| Rolled / full | 1.0928 | 99 | 3,672 | 0.8949 | 161 | 8,816 |
| Full / rolled | 1.0294 | 95 | 8,768 | 0.9109 | 165 | 7,552 |
| Count 2 / full | 1.1500 | 98 | 6,912 | 0.9980 | 167 | 60,920 |
| Count 4 / full | 1.1038 | 96 | 13,480 | 0.9999 | 167 | 60,920 |
| Full / False | 1.0297 | 95 | 8,768 | 0.9126 | 165 | 7,552 |
| Rolled / False | 1.0935 | 96 | 1,480 | 0.9678 | 137 | 1,576 |
| libnvvm | 1.0935 | 96 | 1,480 | 0.9683 | 137 | 1,576 |

The previously unmeasured both-rolled kernels are now measured. They do
not improve Kvaerno3, and they are slower than Newton-only rolling for
Radau5. The omitted partial-Newton/full-Krylov requests are also covered.
For Radau5 those count-2/count-4 requests compile to the same cubin as
all-full. Their small timing differences do not establish distinct native
strategies. The cause of that alias is being traced through actual closure
settings and backend lowering.

Kvaerno3's all-full kernel is the fastest measured physical kernel in this
cohort despite its larger whole-kernel instruction count. Rolling both
loops retains 96 registers and the same theoretical residency but is about
9.4 percent slower. Rolling only Newton or requesting count 2 uses 99/98
registers and reduces theoretical resident threads from 640 to 512. Count
4 retains 640 resident threads and still loses. Register count alone does
not explain all these cases.

Radau5's Newton-only rolled variant has the best measured candidate ratio,
about 10.5 percent below all-full. All listed default geometries have 384
resident threads per SM. The both-rolled variant's lower register count
therefore does not increase residency at this geometry. Its much smaller
binary also does not make it the fastest candidate.

These results require a workload/hot-region explanation rather than a
whole-binary size rule. They do not establish which unrolled instruction
addresses are reached or how many Newton/Krylov bodies execute. The
unroll harness records numerical status but does not collect per-policy
iteration counters; its `features` row is a timing-protocol description.
Matched state-only and counters-enabled captures are being added as
separate experimental labels. Instrumented timings cannot replace this
uninstrumented timing bank, and measured iteration counts cannot become
inputs to a pre-compilation heuristic.

## Fabbri Radau interaction cohort

`fabbri-radau-interactions-e1` runs on the same frozen revision. It retakes
both Radau3 and Radau5 with the historical joint winners, matched individual
stage/step-element/solver-element/norm/other-small rollbacks, Newton rolling,
partial Newton requests, and libnvvm. Its raw bank is
`C:/local_working_projects/cubie-notes/hardware_unroll_placement/fabbri_radau_interactions_e1`.
Results must be added only after cohort completion and strict eligibility
checks. The historical interaction is a hypothesis for this retake, not
a result copied into the new epoch.
