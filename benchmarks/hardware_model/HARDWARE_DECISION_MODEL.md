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

### Unseen-size holdout

`unroll_holdout.py` freezes the source-cap choice, then
`benchmarks/unroll_landscape.py --config --policies` times the
candidates. Systems chain20, chain64, chain32_c8, lorenz96_10,
lorenz96_40; algorithms kvaerno3, kvaerno5, radau_iia_3, radau_iia_5
and the two BiCGSTAB variants. Freeze
`cubie-notes/unroll_landscape/holdout_e1_freeze`, bank `holdout_e1`,
scores `holdout_e1_score.txt`. chain64/radau_iia_5 (LU and BiCGSTAB)
have no prediction: entry values exceed the allocator's bank budget.

| choice | wins | losses | worst loss |
|---|---:|---:|---:|
| frozen source-cap choice, 28 configs | 21 | 7 | 11.1% lorenz96_40/radau_iia_3 |
| default all full | 9 | 19 | 268% chain20/kvaerno5 |

Losses: radau_iia_3 on chain20, lorenz96_10, lorenz96_40 and kvaerno3 on
lorenz96_10 (count 1 chosen, full faster by 6 to 11%); the BiCGSTAB
Newton/Krylov corners on lorenz96_10/kvaerno3 and lorenz96_40 (both
corners chosen count 1, a mixed corner faster by 6 to 7%).

## Placement choice

Partition: `SHARED_PARTITION.md`; L1 data capacity is 128 KiB minus it.
`placement_traffic.py` replays each buffer's step memory events local
under the all-local partition and shared under that buffer's partition;
`placement_traffic_score.py` scores "shared when removed L1 miss sectors
exceed added shared bank wavefronts" against
`cubie-notes/placement_landscape/post913`.

Records: `cubie-notes/placement_landscape/post913_traffic/p1..p4`, per-row
scores in `score_promote.txt`; 116 rows scored, fabbri unscored (Exp and
Pow not admitted by the placement lowering).

| rule | wins | losses | worst loss |
|---|---:|---:|---:|
| removed misses > added wavefronts, promote | 96 | 20 | 127% chain32/kvaerno3_bicgstab accumulator |
| same, addressable | 87 | 29 | 161% lorenz96_20/radau_iia_5 cached_auxiliaries |
| always local | 100 | 16 | 127% chain32/kvaerno3_bicgstab accumulator |

The rule captures none of the 16 measured shared wins. Largest removed
misses: chain32/kvaerno5 accumulator (2.15), chain32/kvaerno3
cached_auxiliaries (1.85), chain32/radau_iia_3 cached_auxiliaries (1.42),
chain32/radau_iia_5 stage_increment (0.70). No placement rule is admitted.

## Partition control

Fork PR 18 (`carveout-launched-function`) applies `shared_memory_carveout`
to the launched CUfunction; executed partitions per preference are in
`SHARED_PARTITION.md`.
