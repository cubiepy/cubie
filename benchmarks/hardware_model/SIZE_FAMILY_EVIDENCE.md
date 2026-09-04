# Lorenz96-20: Kvaerno3 iteration-loop interaction

For this 20-state Lorenz96 workload, rolling both Newton and Krylov
loops is faster than full unrolling, while rolling only Krylov is
slower. Krylov count 4 is the slowest of the five profiled policies.
Its instruction-cache miss counts are lower than full, but its local
frame, local traffic and exact instruction work are larger. These
measurements require separate accounting for code delivery and local
memory effects; neither instruction footprint nor register count alone
orders these policies.

## Scope and receipts

The [independent CPU receipt](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/verification/l96_kvaerno3_independent_20260905/receipt.json)
recounts original SourceCounters CSV exact instruction totals, opcode
groups, sampled totals and every address-frequency bucket. It checks
raw metric values and units, hashes of profiles/CSVs/cubins/NPZs, all
per-run labels, state/status equality, and the
[strict ordinary bank audit](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/verification/l96_kvaerno3_independent_20260905/strict_bank_audit.json).
No CUDA import, native compilation or GPU execution occurred in this
audit. The existing profile analyses use the independently reviewed
`solver_profile_analysis.py`, SHA `ecd46638fb63c83540ead9a586f856d04856500bdeb83aba11f543a4e11731b3`.

The five input analyses are under
`verification/size_profiles_e1_analysis/profile_z_dirk_size_l96_<arm>_e1`,
where `<arm>` is `full`, `both_rolled`, `krylov_rolled`, `krylov_2`, or
`krylov_4`. All paths here are relative to
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.
Each profile targets its state-only reference solve; separately
instrumented iteration counters are validation labels, not timings.

All five use the same exact input NPZ, duration 1, 262,144 trajectories,
float32, Kvaerno3 with BiCGSTAB/Jacobi and `inexact_newton=True`, on the
RTX 4070 SUPER. Source SHA is
`4899b5cb04523177ed3cd3f1aef566591829ed026e064b951fdfbf629cfcef6a`.
Compiler identity is MLIR 0.5.1.1, `anchor_dfs`, `liveness_auto`, LTO,
`nsz/contract/arcp/afn/ftz`, with line information disabled. These are
five different cubins, not aliases counted as independent profiles.
The first six unroll groups remain full in every profiled arm.

## Ordinary timing evidence

The bank is `size_family_unroll_e1/lorenz96_20__kvaerno3_bicgstab`.
All 27 observed launch groups pass the strict cohort checks; each has
six distinct samples with compatible full and duplicate-full references.
The bank contains nine distinct compiled cubins. There are no failed-run
or NaN-mask-mismatch warm diagnostics; seven warm rows have nonzero
state differences. Eligibility does not imply cross-policy bitwise
equality or establish an error tolerance.

| Arm | Policy | Block | Minimum ms | Same-block full ms | Ratio to full | Minimum record line |
|---|---|---:|---:|---:|---:|---:|
| Full | `u11111111` | 0 | 212.5197 | 212.5197 | 1.000000 | 8 |
| Both rolled | `u11111100` | 0 | 174.8739 | 212.5197 | 0.822860 | 14 |
| Krylov rolled | `u11111110` | 1 | 239.1559 | 212.0934 | 1.127597 | 54 |
| Krylov 2 | `u11111112` | 2 | 295.1349 | 212.9472 | 1.385953 | 73 |
| Krylov 4 | `u11111114` | 3 | 481.6456 | 212.5019 | 2.266547 | 99 |

Lines refer to the bank's `records.jsonl`; the receipt retains all six
sample lines for every observation. The full row above shows block 0;
full and duplicate-full solvers recur in all seven bank blocks. Every
listed sample exceeds 20 ms and the compiled geometry provides
18.285714 theoretical occupancy waves.

K4/both-rolled is
`(481.6456 / 212.5019) / (174.8739 / 212.5197) = 2.754476`.
The arms occupy blocks 3 and 0. This ratio uses contemporaneous full
normalization, but is not a directly paired K4/both-rolled timing or a
confidence interval. The unnormalized minima ratio is 2.754245 and has
the same cross-block limitation. Krylov-2/Krylov-rolled similarly gives
1.229121 after normalization in blocks 2 and 1. Sample minima remain
descriptive and can select noise; no fitted significance threshold is
introduced. Profile durations are excluded from these comparisons.

## Exact work and matched numerical labels

| Observable | Full | Both rolled | Krylov rolled | Krylov 2 | Krylov 4 |
|---|---:|---:|---:|---:|---:|
| Exact warp instructions | 57,433,107,505 | 56,962,742,108 | 56,140,168,443 | 55,886,488,155 | 62,866,699,496 |
| Whole encoded bytes | 1,524,480 | 104,064 | 652,032 | 1,025,664 | 619,008 |
| Positive-execution encoded bytes | 200,720 | 102,352 | 154,240 | 231,600 | 140,112 |
| Exact warp `LDL` executions, all widths | 3,620,684,591 | 3,184,579,878 | 3,048,760,969 | 3,035,825,044 | 4,435,826,220 |
| Exact warp `STL` executions, all widths | 2,258,319,521 | 1,865,099,011 | 1,892,297,368 | 1,879,360,767 | 4,201,795,025 |

Encoded bytes are 16 times native instruction addresses. The executed
footprint is a distinct-PC union over the entire grid and solve, not a
temporal hot set or per-SM instruction-cache demand. Local instruction
counts include scalar, 64-bit and 128-bit variants; they are not equal
byte transfers. Exact thread and predicated-on thread work, each opcode
variant and all frequency buckets remain in the receipt.

The instrumented solve matches its own reference state and status
exactly for all five arms, with finite float32 state and 262,144 success
statuses. The four label totals are:

| Arm group | Newton iterations | Linear iterations | Attempted steps | Rejected steps |
|---|---:|---:|---:|---:|
| Full | 745,717,953 | 1,105,771,776 | 181,038,885 | 698 |
| Both rolled and K4 | 745,717,486 | 1,105,771,164 | 181,038,834 | 698 |
| Krylov rolled and K2 | 745,723,736 | 1,105,777,569 | 181,038,937 | 707 |

Within each two-arm group, **every per-run vector, every saved-interval
counter and every reference state element is identical**. The independent
receipt records dtype/shape and SHA-256 of the exact C-order array bytes.
For both-rolled and K4 these are:

- Per-run totals, int64 `[4,262144]`:
  `d65c7a8c76565a51698c0af50c3830fe26f26b6e821b1235d4edbb22e673c124`.
- Interval counters, int32 `[2,4,262144]`:
  `216391b667548ef79d06a5a839e0e8a468dcba53e5a06c13229ac264e89f7380`.
- Reference state, float32 `[2,20,262144]`:
  `75bcaac5241177a944132d7753005bc2ea9a26a7b8ef5526bd41c8583ca23d94`.

The corresponding Krylov-rolled/K2 hashes are `724cdb1e...f72d5`,
`fc6e29c8...89e69` and `e572b348...10d5`; their full hashes are retained
in the receipt. K4 still executes 10.364595% more warp instructions than
both-rolled. K2 executes 0.451869% fewer than Krylov-rolled. Differences
in these four measured solver-work vectors do not explain either pair's
timing separation. Equal totals do not supply per-step warp-voted
iteration histories; those histories were not recorded.

Full is not numerically identical to the other groups. Both-rolled/K4
differ from full on 55,445 Newton and 55,715 linear-iteration per-run
totals, despite aggregate differences of only -467 and -612. Their
attempt totals differ on 4,531 runs and rejection totals on 10 runs.
Krylov-rolled/K2 differ on 252,222 Newton, 252,286 linear, 31,177 attempt
and 35 rejection totals. Aggregate Newton/linear/attempt changes versus
full are below 0.000776%; rejection totals rise from 698 to 707 in the
latter group, or 1.2894%. Small aggregate changes are not evidence of
identical trajectories. Maximum absolute saved-state differences from
full are 0.00818253 and 0.01852417, respectively. This audit does not
relabel these differences as acceptable error or replace missing warp
maxima with means, aggregate totals, or maxima of per-run totals.

## Resources and local-memory observations

Every arm uses 255 reported registers/thread, 256 allocated, blocks of
64 threads, four resident blocks/SM, 4 dynamic shared bytes/block and
an actual shared configuration of 8,192 bytes (`8.192000 Kbyte`, decimal
units). The register occupancy bound does not change. Local frames do:

| Observable | Full | Both rolled | Krylov rolled | Krylov 2 | Krylov 4 |
|---|---:|---:|---:|---:|---:|
| Local bytes/thread | 544 | 408 | 496 | 496 | 976 |
| Local load sectors | 14,522,947,640 | 12,778,911,990 | 12,235,953,766 | 12,184,227,469 | 21,856,817,278 |
| Local store sectors | 9,053,037,360 | 7,480,744,781 | 7,593,120,206 | 7,540,654,231 | 21,920,062,489 |
| Local load lookup-miss sectors | 1,262,250,221 | 609,356,615 | 881,430,277 | 879,891,214 | 8,527,943,884 |
| Local store lookup-miss sectors | 2,233,205,868 | 829,239,540 | 1,645,004,934 | 1,596,241,984 | 14,582,572,820 |

These are exact exported `l1tex__t_sectors_pipe_lsu_mem_local_op_* .sum`
metrics in **sectors**, including the new `*_lookup_miss.sum` counters.
K4's load/store miss fractions are 39.02%/66.53%, versus full's
8.69%/24.67% and both-rolled's 4.77%/11.08%. Relative to both-rolled,
K4 issues 1.7104x load sectors and 2.9302x store sectors, with 13.9950x
and 17.5855x lookup-miss sectors. Its much greater local-memory demand
is directly observed at unchanged allocated-register occupancy and
shared configuration. Static frame bytes alone cannot predict these
traffic counts, identify the source buffer, or assign a latency cost.

## Instruction delivery and interpretation

| Metric `.sum` and exported unit | Full | Both rolled | Krylov rolled | Krylov 2 | Krylov 4 |
|---|---:|---:|---:|---:|---:|
| `sm__icc_requests` [cycle] | 7,866,336,097 | 7,839,714,536 | 6,526,020,641 | 6,710,344,131 | 8,705,359,794 |
| `sm__icc_requests_lookup_miss` [cycle] | 1,231,690,642 | 780,417,461 | 1,152,953,236 | 2,161,646,293 | 796,540,014 |
| `gcc__cache_requests_type_instruction` [request] | 1,098,845,570 | 1,083,728,645 | 485,452,253 | 515,005,497 | 1,984,388,761 |
| `gcc__cache_requests_type_instruction_lookup_miss` [request] | 86,457,906 | 31,885 | 166,040,504 | 274,143,310 | 108,954 |
| `smsp__warps_issue_stalled_no_instruction` [warp] | 41,913,830,915 | 20,357,025,082 | 55,613,683,346 | 120,301,729,815 | 12,012,234,343 |

ICC's exported unit is cycle despite the name containing "requests";
GCC uses requests. These quantities cannot be added or converted to
time with one miss penalty. Cache-domain topology is not inferred from
their names.

K4's slowdown combines lower measured instruction-delivery stalls with
increased local-memory traffic. Relative to full it has lower ICC
lookup-miss cycles, GCC miss requests and no-instruction stall observations,
while exact warp work rises 9.460731% and local traffic rises sharply.
Relative to the numerically identical both-rolled arm, the larger local
frame, traffic and work constrain the explanation further. The evidence
is consistent with a local-memory cost that code-size reduction does
not offset; it does not apportion elapsed time among spills, cache
misses, dependencies or other effects.

Rolling only Krylov lowers exact work by 2.251209% and local traffic
relative to full, yet increases GCC instruction misses 1.9205x and
no-instruction stalls 1.3269x and is 12.7597% slower. Rolling both loops
instead lowers those instruction-delivery counters and local traffic,
with 0.818980% less exact work and 17.7140% lower bank time. This is a
Newton/Krylov interaction for this workload, not an independently
additive benefit for each loop group. The matched Krylov-rolled/K2 pair
is another constraint: almost unchanged local traffic and slightly less
work coexist with 1.8749x ICC misses, 1.6511x GCC misses, 2.1632x
no-instruction stalls and 22.9121% higher normalized time. These
patterns support competing physical effects, not one monotone rule
based on source unroll count or whole-kernel bytes.

Every exact source total matches its software instruction metric.
Hardware warp totals exceed source totals by `[54,52,54,54,52]` in table
order; these residuals remain unresolved. K2 has 41,624,344 aggregate
samples versus 41,582,775 source samples, a residual of 41,569. The other
four aggregate/source sample residuals are zero. Sampled stalls remain
separate from exact software work and hardware issue counters; no
residual is silently corrected and no samples are converted to cycles.

The smaller three-state Lorenz Kvaerno3 contrast in
[SOLVER_PROFILE_EVIDENCE.md](SOLVER_PROFILE_EVIDENCE.md) has no observed
local LD/ST traffic and favors full over both-rolled. This 20-state
case therefore does not supply a universal Kvaerno3 default. These are
different systems and durations, not a controlled size-only experiment.
This contrast adds a
measured constraint for a workload-specific model of code expansion,
scalarization/address-taking and local-memory demand, with native
lowering uncertainty retained and no fitted constants.
