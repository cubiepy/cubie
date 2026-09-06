# Hardware decision model

Source-only choices: unroll level of the Newton and Krylov loops, and the
memory space of each relocatable buffer. Score = configurations whose
chosen candidate is within 5% of the fastest measured (win) or not
(loss), with the worst loss as chosen time over fastest time.

## Unroll choice

`decision_model.py` chooses; `unroll_choice_score.py` scores the rule.

```powershell
python benchmarks/hardware_model/decision_model.py chain32 radau_iia_5 --newton 2
python benchmarks/hardware_model/unroll_choice_score.py --bank lu
python benchmarks/hardware_model/unroll_choice_score.py --bank split --regime cap
```

Inputs: static slots per runtime region (`static_slots.py`, records in
`cubie-notes/hardware_unroll_placement/static_slots_e1`), resident
warps at the allocator's register count, the delivery curve
(`INSTRUCTION_DELIVERY_CATALOG.json`), iterations per step (measured
counts for scoring, the source cap before compilation).

Hot footprint: outside slots + Newton body per static copy (visits up
to the cap for full, one for count 1), each with one Krylov body per
Krylov copy, plus the error solve's Krylov copies; slots × 16 B / 1.417.
Choice: smallest service; default (Newton full, Krylov count 1) inside
2%; past the curve the smaller footprint, default inside 5%.

Scores (`recovered/post882_audit_strict.json`,
`recovered/split_flags_audit_strict.json`, counts from
`iteration_counts_20260904`):

| bank | candidates | iterations | wins | losses | worst loss |
|---|---|---|---:|---:|---:|
| post882 LU, 14 configs | Newton full / count 1 | measured | 12 | 2 | 10.3% fabbri/radau_iia_5 |
| post882 LU, 14 configs | Newton full / count 1 | cap | 11 | 3 | 10.3% fabbri/radau_iia_5 |
| split flags, 9 configs | four Newton/Krylov corners | measured | 9 | 0 | 4.4% |
| split flags, 9 configs | four Newton/Krylov corners | cap | 8 | 1 | 9.4% lorenz/kvaerno3_bicgstab |
| default Newton full, Krylov count 1 | | | 16 | 7 | 96.5% lorenz96_20/kvaerno5 |

Losses at measured iterations: fabbri/radau_iia_5 (both candidates past
the curve, full faster by 9.7%), lorenz96_20/radau_iia_3 (count 1 below
the 65 KiB service step, full above it, full faster by 6.3%).

Register and spill term: the allocator sees one trace per configuration,
so its spill and reload counts are equal across candidates. SASS LDL per
Newton body, rolled against unrolled: fabbri/radau_iia_5 4854/4190,
lorenz96_20/radau_iia_5 326/294, chain32/radau_iia_5 555/578,
lorenz96_20/radau_iia_3 98/104, chain32/radau_iia_3 227/226,
fabbri/radau_iia_3 1399/1444. No spill term is admitted.

## Placement choice

Partition: `SHARED_PARTITION.md`; L1 data capacity is 128 KiB minus it.
`placement_traffic.py` replays each buffer's step memory events local
under the all-local partition and shared under that buffer's partition;
`placement_traffic_score.py` scores "shared when removed L1 miss sectors
exceed added shared bank wavefronts" against
`cubie-notes/placement_landscape/post913`.

PLACEMENT_RESULTS

## Partition control

Fork PR 18 (`carveout-launched-function`) applies `shared_memory_carveout`
to the launched CUfunction; executed partitions per preference are in
`SHARED_PARTITION.md`.
