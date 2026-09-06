# Hardware decision model

Two decisions, both made from source before compilation: the unroll
level of the Newton and Krylov iteration loops, and the memory space of
each relocatable buffer. Scores are counts of configurations where the
model's choice is within five percent of the fastest measured candidate
(wins) or not (losses), with the worst loss as the chosen time over the
fastest time.

## Unroll choice

`decision_model.py` chooses; `unroll_choice_score.py` scores the same
rule.

```powershell
python benchmarks/hardware_model/decision_model.py chain32 radau_iia_5 --newton 2
python benchmarks/hardware_model/unroll_choice_score.py --bank lu
python benchmarks/hardware_model/unroll_choice_score.py --bank split --regime cap
```

Inputs:

- Static instruction slots per runtime region of one policy graph
  (`static_slots.py`; records in
  `cubie-notes/hardware_unroll_placement/static_slots_e1`): code outside
  the iteration loops, the Newton body, the Krylov body inside it and
  the Krylov body of the error solve. Every candidate policy shares this
  trace; the candidates differ only in how many static copies of each
  body a step touches.
- Resident warps from the occupancy equations at the allocator's
  register count.
- The measured instruction-delivery curve
  (`INSTRUCTION_DELIVERY_CATALOG.json`), service per warp instruction
  against hot footprint.
- Iterations per step: measured counts for scoring, the source cap
  before compilation.

Hot footprint of a policy: outside slots, plus one Newton body per
static copy (the visits per step up to the cap for full, one for count
1), each carrying one Krylov body per Krylov copy, plus the error
solve's Krylov copies; slots times 16 bytes over the projected-slot
factor 1.417 (median projection/SASS over 345 calibration rows).
Choice: the smallest service on the curve, the default (Newton full,
Krylov count 1) inside a two percent band; when every candidate lies
past the measured curve the smaller footprint, since delivery cost does
not fall with footprint, the default inside a five percent footprint
band.

Scores against the strict audits (`recovered/post882_audit_strict.json`,
`recovered/split_flags_audit_strict.json`), measured iterations per step
from `iteration_counts_20260904`:

| bank | candidates | iterations | wins | losses | worst loss |
|---|---|---|---:|---:|---:|
| post882 LU, 14 configs | Newton full / count 1 | measured | 12 | 2 | 10.3% fabbri/radau_iia_5 |
| post882 LU, 14 configs | Newton full / count 1 | cap | 11 | 3 | 10.3% fabbri/radau_iia_5 |
| split flags, 9 configs | four Newton/Krylov corners | measured | 9 | 0 | 4.4% |
| split flags, 9 configs | four Newton/Krylov corners | cap | 8 | 1 | 9.4% lorenz/kvaerno3_bicgstab |
| default Newton full, Krylov count 1 | | | 16 | 7 | 96.5% lorenz96_20/kvaerno5 |

Losses at measured iterations: fabbri/radau_iia_5 (full wins by 9.7%
with both candidates past the curve) and lorenz96_20/radau_iia_3 (full
wins by 6.3% with the count-1 footprint below the 65 KiB service step
and the full footprint above it).

### Register and spill term

The allocator (`implicit_native_lowering.BankAllocation`) sees one trace
for every candidate, so its registers, spill and reload events per body
are identical across candidates and cannot separate them. In the SASS
of the post882 bank the rolled Newton body carries more LDL than the
unrolled body per iteration on fabbri/radau_iia_5 (4854 against 4190)
and lorenz96_20/radau_iia_5 (326 against 294), fewer on
chain32/radau_iia_5 (555 against 578) and lorenz96_20/radau_iia_3 (98
against 104), and the same on chain32/radau_iia_3 and fabbri/radau_iia_3;
the sign is not a property of rolling. No spill term is admitted. The
allocator's per-region spill and reload counts stay in the static-slot
records.

## Placement choice

Partition: `SHARED_PARTITION.md`; L1 data capacity is 128 KiB minus it.
`placement_traffic.py` replays the step's memory events for each
relocatable buffer twice, local under the all-local partition and shared
under the partition the driver selects for that buffer's dynamic bytes;
`placement_traffic_score.py` scores the rule "shared when the removed
L1 miss sectors exceed the added shared bank wavefronts per warp and
step" against `cubie-notes/placement_landscape/post913` (19 configs,
231 shared rows, 45 wins, 122 losses, 64 ties).

PLACEMENT_RESULTS

## Partition control

The fork branch `carveout-launched-function` (ccam80/numba-cuda-mlir PR
18) applies the `shared_memory_carveout` target option to the launched
CUfunction; CUPTI confirms the executed partition follows it
(`SHARED_PARTITION.md`). Production sets no preference; the placement
model can request the partition it evaluated once that wheel is
installed.
