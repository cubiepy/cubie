# Recovered unroll evidence — strict eligibility audit

Reconstructed on 2026-09-04. These are historical measurements among the
available eligible cubins, not exhaustive searches or globally optimal
policies. They motivate current-source experiments, not shipped defaults.

## Scope and receipts

- **P**: `C:/local_working_projects/cubie-notes/unroll_landscape/post882`.
  Compile source hash
  `e83c198d2c71f738e4b136c294572eeb74f8a617c3078f2dde59d3e65c28bbac`
  (`compiles.jsonl:1`). Seven policy positions are stage, step-element,
  accumulator, solver-element, norms, other-small, combined exits.
- **S**: `C:/local_working_projects/cubie-notes/unroll_landscape/split_flags`.
  Compile source hash
  `7e1cf932f2054d789310a5d063adf3103cdc23db0f8ff69a8b8159a4406dbb8f`
  (`compiles.jsonl:1`). Its eight positions replace combined exits with
  Newton then Krylov. Harness `b87c1104` on `chore/unroll-landscape`, based
  on PR #910; PR #909's counter correction was not present.
- `unroll_placement_model/UNROLL_SWEEP.md:7` records wheel
  `cubie-numba-cuda-mlir 0.5.1.1`. Historical compile rows have no complete
  compiler/backend/default-setting identity. The source hashes are known;
  per-row toolchain equivalence cannot be retrospectively certified.
- Levels: `1` full, `0` rolled, `2`/`4` partial, `n` False/libnvvm choice.
  Identical cubin aliases are inherited observations, not repetitions.

Strict artifacts, outside the repository:

- `C:/local_working_projects/cubie-notes/hardware_unroll_placement/recovered/post882_audit_strict.json`
- `C:/local_working_projects/cubie-notes/hardware_unroll_placement/recovered/split_flags_audit_strict.json`

They include input hashes, analyzer hash, sample receipts, rejected rows,
alias holes and eligible rankings. Earlier `*_audit.json` files remain
unchanged and predate the eligibility repairs. No raw bank is copied into
the repository.

## Eligibility and counts

Each ranking uses the minimum of a complete two-round, three-repeat
launch divided by the minimum complete all-full launch in its own
wave/block. Duration, run count, source/compiler identity and warm launch
geometry must be compatible. Warm snapshots must report zero failed runs,
matching NaN masks and at least two occupancy waves. Ambiguous compile
identities and incomplete/duplicate samples are rejected. Targeted current
cohorts additionally require their manifest and complete duplicate
reference. Historical toolchain fields remain unknown, as noted above.

Nonzero absolute differences alone do not prove invalidity for long
chaotic trajectories; they remain recorded diagnostics. Timings below
20 ms are flagged separately and remain a duration-protocol limitation.

| Bank | Configs with timings | Rankable configs | Eligible launches | Rejected launches | Five-full captures available best | Within 10% |
|---|---:|---:|---:|---:|---:|---:|
| P | 40 | 38 | 2,578 | 161 | 36/38 | 37/38 |
| S | 24 | 23 | 327 | 11 | 23/23 | 23/23 |

The five fixed groups are stage, step-element, accumulator,
solver-element and norms. S only explores that fixed region, so 23/23
does not test the universality of these five choices. The earlier raw
38/40 P count included numerically ineligible configurations.

P rejected-launch reasons overlap: 29 failed-status rows, 148 NaN-mask
mismatches, and 161 launches whose matching reference is ineligible.
S has five failed-status rows, six NaN mismatches, and 11 launches with an
ineligible reference. P excludes Fabbri/Kvaerno3 and Fabbri/Rodas3p from
rankings; S excludes Fabbri/Kvaerno3.

| Numerical receipt | Failed warm rows | Maximum failed runs | NaN mismatches | First failed / mismatched snapshot |
|---|---:|---:|---:|---|
| P Fabbri/Kvaerno3 | 8 | 13 | 133 | `records.jsonl:14698` / `14700` |
| P Fabbri/Rodas3p | 21 | 38 | 15 | `records.jsonl:17438` / `17440` |
| S Fabbri/Kvaerno3 | 5 | 5 | 6 | `records.jsonl:2069` / `2071` |

## Eligible counterexamples to five fixed groups

Both remaining exceptions are Fabbri/FIRK and pass the stored status/mask
checks. Their coupled changes beat the available five-full region:

| P configuration | Best observed policy | Time/all-full | Best five-full policy | Five-full penalty | Winning / five-full timing receipt |
|---|---|---:|---|---:|---|
| Fabbri/Radau IIA 3 | `u1110000` | 0.786046 | `u1111110` | 5.4346% | `records.jsonl:16628` / `16784` |
| Fabbri/Radau IIA 5 | `u0010000` | 0.801879 | `u1111111` | 24.0599% | `records.jsonl:17435` / `17242` |

The eight-position current-source equivalents are `u11100000` and
`u00100000`. They were outside S's fixed-five search and must be measured
explicitly. Whole-kernel SASS size does not establish the dynamic hot
instruction working set; these outcomes alone do not identify a cache
capacity or prove a register-only mechanism.

## Coverage holes and aliases

Two S physical cubins never received a settled full-duration timing:

| Configuration | Aliased labels | Recorded representative | S record lines |
|---|---|---|---|
| Lorenz/Kvaerno3 BiCGSTAB | `u11111100`, `u1111110n` | untimed `libnvvm` | 122, 123 |
| Lorenz/Radau IIA 5 BiCGSTAB | `u11111100`, `u1111110n` | untimed `libnvvm` | 467, 468 |

S's split-fill only timed single rolled deviations. Newton factors 2/4
with Krylov full were not in the split grid: the preserved b87 harness at
`C:/local_working_projects/cubie-worktrees/unroll-landscape/benchmarks/unroll_landscape.py:68`
sets Newton levels `1024`, and line 69 sets Krylov levels `024n`.
`u11111121`, `u11111141`, `u11111021`, `u11111041` each have zero compile
rows among S's 24 configs. This is a syntactic coverage gap; dead loop
groups in LU or ROS can make some candidates physically equivalent.

P has **43 alias events across 11 configs pointing to 18 cubins without
any settled timing**, even before numerical exclusions. Representative
receipts and complete config counts:

| P configuration | Events | Alias record lines |
|---|---:|---|
| Lorenz/BS32 | 2 | 17544, 17547 |
| Lorenz/ROS23 | 1 | 17860 |
| Lorenz/Rodas3p | 1 | 17897 |
| L96-20/Kvaerno5 | 4 | 20179, 20181, 20182, 20183 |
| Chain32/BS32 | 15 | 18589, 18591, 18593, 18594, 18596, 18598, 18599, 18601, 18603, 18605, 18606, 20435, 20437, 20438, 20439 |
| Chain32/Kvaerno3 | 3 | 20499, 20502, 20503 |
| Chain32/Kvaerno5 | 4 | 20531, 20533, 20534, 20535 |
| Chain32/ROS23 | 1 | 19088 |
| Fabbri/BS32 | 4 | 20787, 20789, 20790, 20791 |
| Fabbri/Vern7 | 4 | 20819, 20821, 20822, 20823 |
| Fabbri/Kvaerno5 | 4 | 20883, 20885, 20886, 20887 |

After numerical eligibility, P has 108 alias events without an eligible
representative and S has 32. These totals include the structural holes
above plus aliases whose only observations/reference are rejected.
P's fixed-four follow-up added no new cubins, but claiming that every
alias already had an eligible timing would be incorrect.

## Historical repetitions

P's current file retains one six-sample group per label/block. Six backup
snapshots recover 26,063 distinct timing records when keyed by stored key
plus timestamp, versus 16,434 in the current file. The backups include
partial runs, changed durations and changed memory-block protocol. They
are not multiple interchangeable full-bank repeats. The strict P artifact
keeps their reference warm cohorts separately without pooling timings.

Reproduce the strict files from this checkout with:

```powershell
$py = 'C:/local_working_projects/cubie/.venv/Scripts/python.exe'
$raw = 'C:/local_working_projects/cubie-notes/unroll_landscape'
$out = 'C:/local_working_projects/cubie-notes/hardware_unroll_placement/recovered'
& $py benchmarks/hardware_model/bank_analysis.py "$raw/post882" --history --observations --output "$out/post882_audit_strict.json"
& $py benchmarks/hardware_model/bank_analysis.py "$raw/split_flags" --observations --output "$out/split_flags_audit_strict.json"
```

## Fresh Fabbri Radau exception checks, 2026-09-04

The corrected-source cohort is separate from historical P/S. Raw files
are under
`C:/local_working_projects/cubie-notes/hardware_unroll_placement/fabbri_radau_interactions_e1`.
Its source hash is `4899b5cb04523177ed3cd3f1aef566591829ed026e064b951fdfbf629cfcef6a`;
compiler fingerprint is `68e2731a23c73a42669ec8e16a8cc22fd42f6b09432ea0a54854c2a15ea9ce24`,
MLIR with `anchor_dfs`, `liveness_auto`, and recorded default JIT flags.
Both methods use `inexact_newton=False`, `prefactored=False` and LU,
duration 0.2,
131,072 runs, block size 64, and at least 9.142857 occupancy waves.

Independent raw grouping verifies 46 complete launch groups: two rounds
of three timings each, 276 timings and 46 warm diagnostics. The all-full
and separately owned duplicate repeat in each of six blocks per method.
There are 22 candidate launch groups plus 24 reference/duplicate groups;
the references are not 24 different policies. The strict bank audit has
46 eligible groups, zero rejected groups and no coverage holes within
this selected cohort. Ratios below divide six-sample minima by the local
block's all-full minimum; they do not establish significance or a global
optimum beyond the candidates measured here.

| Method | Best observed policy | Min ms | Ratio to local full | Best five-full cubin / ratio | Five-full penalty |
|---|---|---:|---:|---|---:|
| Radau IIA 3 | `u11100000` | 83.0294 | 0.772852 | `u11111101` / 0.834081 | 1.079225 |
| Radau IIA 5 | `u00100000` | 178.0639 | 0.812139 | all-full identity / 0.995612 | 1.225915 |

Winner timing receipts are `records.jsonl:16` and `:197`; corresponding
warm rows are `:6` and `:173`. The winning cubins use 255 registers/thread
in both methods, with 3,104/8,296 local bytes and 9,512/23,112 total SASS
instructions. These are whole-kernel static counts, not hot working-set
sizes. Simultaneous changes in code, local accesses and potentially
iteration workload do not isolate one physical cause.

Byte identity is observed despite independently requested directives:
Radau3 all-full and Newton-count4 share cubin `62fca06e...`
(`compiles.jsonl:1`, `:12`); Radau5 all-full, Newton-count2 and count4 share
`dbcd28b8...` (`:13`, `:23`, `:24`). They were directly timed, not skipped
by alias events. Radau5's best five-full ratio inherits the identical
count2 cubin measurement in block 4; its small departure from 1 is local
timing scatter between byte-identical programs. These observations do
not identify the backend's counted-unroll lowering mechanism.

No warm row records failed runs or a NaN-mask mismatch. However, 19 warm
rows differ from the full reference (10 Radau3, nine Radau5); the largest
reported absolute difference is `9.918212890625e-05`. The winners differ
by `8.0108642578125e-05` and `7.62939453125e-05`, respectively, with
131,072 runs differing. These are diagnostics, not an accuracy-tolerance
or matched-iteration-work proof. All twelve duplicate warm comparisons
are exactly zero. Timing eligibility must not be called numerical
equivalence merely because the status flags are clear.

Independent source hashes, every minimum/reference line, aliases and
warm differences are retained in
`C:/local_working_projects/cubie-notes/hardware_unroll_placement/verification/fresh_fabbri_icache_independent_20260904.json`.
