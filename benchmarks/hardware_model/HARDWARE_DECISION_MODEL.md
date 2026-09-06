# Hardware decision model

`decision_model.py` picks the Newton loop level, full or count 1, before
compilation. Inputs: projected executed instruction bytes per step per
candidate, resident warps from the occupancy equations at the source
allocator's register estimate, and the measured delivery curve in
`INSTRUCTION_DELIVERY_CATALOG.json`. The other six loop groups stay full
and Krylov loops stay at count 1.

```powershell
python benchmarks/hardware_model/decision_model.py chain32 radau_iia_5 --regimes 1,2,4,cap
```

## Executed footprint

`footprint_calibration.py` writes each policy's typed-slot forecast beside
its post882 SASS count. Executed slots at N Newton bodies per step are
the covered body plus (N minus visited loops) replicated bodies, divided
by the projected-slot factor 1.443 (median projection/SASS over 279
rows). Krylov full on BiCGSTAB projects to megabytes the compiler never
emits and is outside this model.

## Delivery rule

Service is interpolated on the curve for the closest resident warp
count. Both candidates inside the flat region (129 KiB at 8 warps):
full. One inside: that one. Both outside: the smaller footprint.

## Scores

`fetch_rule_score.py` against the strict audits, with Newton per step
from `cubie-notes/hardware_unroll_placement/iteration_counts_20260904`:

| bank | comparison | captured | missed | ties |
|---|---|---:|---:|---:|
| post882 LU | all full vs Newton count 1 | 10 | 1 | 3 |
| split flags | Newton count 1, Krylov count 1, vs full | 4 | 0 | 0 |
| split flags | Newton count 1, Krylov full, vs full | 9 | 2 | 4 |

Misses: fabbri/radau_iia_5 (both outside the curve, full wins with fewer
local loads) and lorenz/radau_iia_5_bicgstab rows with equal footprints.
Both need the register and spill term.

## Placement

Partition: `SHARED_PARTITION.md`; L1 data capacity is 128 KiB minus it.
L1 loss: chain32/Kvaerno3/LU reservation curve in
`RESERVATION_EVIDENCE.md`. Per-buffer bank at production unroll:
`cubie-notes/placement_landscape/post913`, 19 configs, 231 shared rows,
45 wins, 122 losses, 64 ties; wins are stage_increment on chain32 Radau,
accumulator on chain32 Kvaerno3/BiCGSTAB, state on chain32 Vern7 and the
BiCGSTAB work buffers on lorenz96_20 Rosenbrock. `placement_score.py`,
`placement_mechanism.py` and `placement_traffic.py` tabulate the rows
against partition, frame, spill and source memory demand. No placement
rule is admitted; setting the partition from the model needs the backend
to apply the preference to the launched function.
