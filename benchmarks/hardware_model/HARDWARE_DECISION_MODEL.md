# Hardware decision model

`decision_model.py` chooses the Newton loop level (full or count 1) for
one system and algorithm before compilation. Inputs are the projected
executed instruction bytes per step of each candidate, the resident warps
from the CUDA occupancy equations at the source allocator's register
estimate, and the measured instruction-delivery curve in
`INSTRUCTION_DELIVERY_CATALOG.json`. Stage, step element, accumulator,
solver element, norms and other-small loops stay full; Krylov loops on
iterative solvers stay at count 1 (measured rulings, 2026-09-04).

```powershell
python benchmarks/hardware_model/decision_model.py chain32 radau_iia_5 --regimes 1,2,4,cap
```

## Executed footprint

`footprint_calibration.py` rebuilds the source policy graph, lowers it to
typed opcodes and counts 16-byte slots (`instruction_footprint.forecast`).
The executed footprint at N Newton bodies per step is the covered body
plus (N minus visited loops) replicated bodies, divided by the
projected-slot factor. Over 279 post882 compiles the projection to SASS
ratio has median 1.443; rolled step bodies sit within 3 to 8% of SASS
and fully unrolled bodies 1.3 to 1.7 times over because the compiler
folds copies. Requested Krylov full on BiCGSTAB projects to megabytes
the compiler never emits, so Krylov is not decided by this model.

## Delivery rule

Service is interpolated on the measured curve for the closest resident
warp count. Both candidates under the curve's flat region (129 KiB at 8
warps) means full. A candidate over the region while the other is under
means the one under. Both over means the smaller executed footprint.

## Scores

Counts against the strict post882 and split-flags audits, Newton per
step from `cubie-notes/hardware_unroll_placement/iteration_counts_20260904`:

| bank | comparison | captured | missed | measured ties (all predicted tie) |
|---|---|---:|---:|---:|
| post882 LU | all full vs Newton count 1 | 10 | 1 | 3 |
| split flags | Newton count 1 with Krylov count 1 vs full | 4 | 0 | 0 |
| split flags | Newton count 1 with Krylov full vs full | 9 | 2 | 4 |

The post882 miss is fabbri/radau_iia_5, where both candidates exceed the
curve and the measured full win comes with fewer local loads; the split
misses are lorenz/radau_iia_5_bicgstab rows with equal footprints. Those
need the register and spill term, not delivery.

## Placement

`SHARED_PARTITION.md` gives the driver-selected partition; L1 data
capacity is 128 KiB minus it. The chain32/Kvaerno3/LU reservation curve
(`RESERVATION_EVIDENCE.md`) measures the L1 loss: 21.6 ms at 8 KiB,
25.1 at 32, 29.1 at 64, 31.9 at 100. The per-buffer bank
`cubie-notes/placement_landscape/post913` (19 configs at 20 or more
states, production unroll) has 45 wins, 122 losses and 64 ties over 231
shared rows; wins are stage_increment on chain32 Radau, accumulator on
chain32 Kvaerno3/BiCGSTAB, state on chain32 Vern7 and the BiCGSTAB work
buffers on lorenz96_20 Rosenbrock. `placement_score.py` and
`placement_mechanism.py` tabulate ratios against partition, frame and
spill deltas; `placement_traffic.py` extracts per-buffer local and shared
demand from `placement_decision`. A placement rule is not admitted: the
demand output has no validated service model, and the largest winner
fails its lowering. Setting the partition from the model requires the
backend to apply the preference to the launched function.
