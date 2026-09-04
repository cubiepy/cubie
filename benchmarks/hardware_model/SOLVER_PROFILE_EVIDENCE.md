# Exact solver work and instruction footprints

The matched Lorenz profiles show different tradeoffs for Radau IIA 5
and Kvaerno 3 with BiCGSTAB. Rolling Radau's Newton loop increases exact
warp instruction work by 1.859%, while the matched bank minimum falls
10.512%. Its instruction delivery counters improve strongly. Rolling
both Kvaerno iteration loops increases exact work by 9.295%; its bank
minimum rises 9.386%, despite a smaller executed instruction footprint
and improved instruction delivery counters. These observations require
family-specific workload accounting. They do not identify a universal
cache penalty, opcode cycle weight, or register predictor.

## Provenance and validation

All paths below are relative to
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.
The four input directories, in table order, are:

1. `solver_profile_radau5_full_e1_v3`, using
   `source_counters_v2.csv`; the earlier failed export is retained.
2. `profile_solver_radau5_newton_rolled_e1`, using
   `source_counters.csv`.
3. `profile_solver_kvaerno3_full_e1`, using `source_counters.csv`.
4. `profile_solver_kvaerno3_both_rolled_e1`, using
   `source_counters.csv`.

The independently reproduced outputs are under
`verification/solver_profile_independent_final_20260905`, with one
`analysis.json` and `per_pc.jsonl` per input directory. Its `receipt.json`
independently recomputes exact source totals, opcode groups, and every
frequency-curve bucket from the CSV/PC data. The analysis tool
`solver_profile_analysis.py` is frozen at SHA256
`ecd46638fb63c83540ead9a586f856d04856500bdeb83aba11f543a4e11731b3`.
`verification/solver_profile_final_adversarial_20260905.json` verifies
rejection of changed actual launch skip, script, policy, duplicate flags,
and fractional values in integer hardware/sample counts.

Each audit imports the saved Nsight report again, compares both exported
CSV tables, and disassembles the saved reference cubin on the CPU.
Every source PC, opcode, predicate and operand matches the native binary
after documented display normalization and symbol relocation. Neither
audit compiles CUDA nor launches a kernel. The command/request identity
gate establishes that launch skip 0/count 1 selects the state-only
reference solve before the separate instrumented counter solve.

The checks join exact source, compiler, configuration, policy, input
arrays, reference cubin, compiled geometry, raw counter arrays, successful
finite state/status comparisons, and original bank timing rows. The
frozen package source digest is
`4899b5cb04523177ed3cd3f1aef566591829ed026e064b951fdfbf629cfcef6a`.
Compiler identity is MLIR with `anchor_dfs`, `liveness_auto`, LTO and
`nsz/contract/arcp/afn/ftz` enabled, and line information disabled.
All launches use 262,144 trajectories, blocks of 64 threads, grid
`(4096,1,1)`, block dimensions `(1,64,1)`, and 56 SMs on the RTX 4070
SUPER. Radau integrates duration 8; Kvaerno integrates duration 2.

## Exact work, footprints and original timings

The abbreviations R5 and K3 denote Radau IIA 5 and Kvaerno 3. Policies
are `u11111111`, `u11111101`, `u11111111`, and `u11111100`, respectively.
The first six groups retain their full policy in these contrasts.

| Observable | R5 full | R5 Newton rolled | K3 full | K3 both rolled |
|---|---:|---:|---:|---:|
| Whole native instruction addresses | 60,920 | 8,816 | 26,320 | 1,480 |
| Whole encoded bytes | 974,720 | 141,056 | 421,120 | 23,680 |
| Addresses with positive exact execution | 59,958 | 8,562 | 3,942 | 1,385 |
| Encoded bytes at those addresses | 959,328 | 136,992 | 63,072 | 22,160 |
| Addresses with zero execution | 962 | 254 | 22,378 | 95 |
| Exact warp instructions | 8,091,476,305 | 8,241,883,972 | 9,660,455,338 | 10,558,425,758 |
| Exact thread instructions | 258,400,719,202 | 263,206,340,008 | 308,492,396,190 | 337,192,967,248 |
| Exact predicated-on thread instructions | 254,411,451,583 | 259,127,749,729 | 300,482,452,585 | 328,265,929,198 |
| Original bank samples | 60 | 6 | 48 | 18 |
| Original bank minimum, ms | 19.9575 | 17.8596 | 21.0759 | 23.0541 |
| Original bank median, ms | 20.01945 | 17.88985 | 21.1026 | 23.0734 |

Every native instruction occupies 16 encoded bytes in these binaries.
The positive-execution footprint is the distinct-address union across
the entire grid and solve. It provides no temporal reuse distance,
per-SM working set, or cache-line fetch count. K3 full executes addresses
covering only 63,072 bytes of its 421,120-byte instruction body. Applying
a cache-size cliff to the complete body would count many unvisited
addresses. R5 Newton rolled still has a 136,992-byte executed union,
larger than 128 KiB; this observation alone cannot establish cache fit.

Each output retains the complete address-frequency rank curve, including
the zero-execution bucket. Positive per-address warp execution counts
range from 9 to 2,795,810 for R5 full; 77 to 8,372,801 for R5 rolled;
22 to 4,834,058 for K3 full; and 8,192 to 23,818,506 for K3 rolled.
These frequency distributions expose uneven use without assuming a
temporal instruction stream or fitting a hot/cold threshold.

The bank samples have unequal counts and include repeated baselines;
their minima and medians are descriptive, not a newly paired estimator.
Some original samples are below the later 20 ms per-sample protocol.
The profiles do not repair that limitation. Nsight duration is retained
as a diagnostic in the raw metrics, and instrumented counter timings
are never substituted for the uninstrumented bank samples.

## Native operation mix and iteration labels

These are exact warp-level opcode executions. A warp instruction is
counted independently of the participating thread count and predicate;
the separate predicated-on thread column must be used when that
distinction matters. In particular, an issued predicated call is not
automatically an executed callee invocation. [Nsight Source Page](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#source-page)

| Opcode | R5 full | R5 Newton rolled | K3 full | K3 both rolled |
|---|---:|---:|---:|---:|
| `FFMA.FTZ` | 2,374,171,553 | 2,416,262,183 | 1,677,706,524 | 1,774,016,597 |
| `FMUL.FTZ` | 1,676,268,050 | 1,681,816,178 | 1,898,863,045 | 1,902,452,549 |
| `FADD.FTZ` | 507,848,706 | 516,164,343 | 433,482,506 | 464,357,133 |
| `FMNMX.NAN` | 1,832,375,916 | 1,832,375,916 | 1,782,844,283 | 1,773,685,725 |
| `PLOP3.LUT` | 220,142,795 | 259,034,670 | 599,468,333 | 781,836,174 |
| `MUFU.RCP` | 282,526,638 | 285,300,702 | 476,270,412 | 490,566,492 |
| `FSEL` | 226,061,956 | 220,516,013 | 405,019,064 | 451,852,013 |
| `BRA` | 48,847,505 | 54,446,242 | 144,127,319 | 196,819,092 |

The complete vectors, including all predicates and all three exact
execution columns, are retained in the analysis artifacts. Different
opcodes are not assigned equal cycle costs by this audit. More exact
instructions therefore need not mean proportionately more execution
time; scheduling, dependencies, instruction delivery and operation
mix remain distinct physical quantities.

Both R5 policies have identical per-run totals for the four separately
instrumented labels, with aggregate totals
`[177330968, 381395202, 88629784, 241552]` in Newton iterations, linear
solver iterations, attempted steps, and rejected steps order. Their
state-only reference states also match exactly. Thus the observed R5
instruction-work difference does not require a difference in these
four measured iteration totals. K3 full and rolled label totals are
`[651408496, 933207502, 153045310, 33291]` and
`[651355123, 933139757, 153046643, 32470]`. Each instrumented solve
matches its own state-only reference exactly. The labels describe
individual trajectories; sums or lane maxima cannot reconstruct warp
loop execution counts. They validate workloads and are not inputs to
the shipped pre-compile heuristic. See `COUNTER_EVIDENCE.md`.

## Resources, instruction delivery and traffic units

| Resource | R5 full | R5 Newton rolled | K3 full | K3 both rolled |
|---|---:|---:|---:|---:|
| Registers/thread | 167 | 161 | 96 | 96 |
| Allocated registers/thread | 168 | 168 | 96 | 96 |
| Resident blocks/SM | 6 | 6 | 10 | 10 |
| Occupancy waves, from compiled geometry | 12.190476 | 12.190476 | 7.314286 | 7.314286 |
| Dynamic shared bytes/block | 4 | 4 | 4 | 4 |
| Shared configuration, exported `Kbyte` | 16.384000 | 16.384000 | 32.768000 | 32.768000 |

The shared configurations are 16,384 and 32,768 bytes respectively;
the exported decimal `Kbyte` unit is not KiB. None of these contrasts
changes allocated register occupancy or its pair's shared configuration.
There is no observed local load/store traffic in these four profiles.
This is a constraint on explanations of these cases, not a general rule
that iteration loops are insensitive to registers or buffer placement.

| Metric suffix (`.sum`) and exact exported unit | R5 full | R5 Newton rolled | K3 full | K3 both rolled |
|---|---:|---:|---:|---:|
| `sm__icc_requests` [cycle] | 1,064,696,353 | 985,118,026 | 1,281,483,754 | 1,180,646,905 |
| `sm__icc_requests_lookup_miss` [cycle] | 68,715,333 | 2,101,709 | 66,855,613 | 907 |
| `gcc__cache_requests_type_instruction` [request] | 60,759,213 | 4,110,862 | 122,454,115 | 5,012 |
| `gcc__cache_requests_type_instruction_lookup_miss` [request] | 2,282,878 | 12,421 | 3,332 | 0 |
| `smsp__warps_issue_stalled_no_instruction` [warp] | 6,836,196,809 | 803,196,639 | 4,389,871,440 | 1,494,060,312 |
| `lts__t_sectors_op_read` [sector] | 37,820,740 | 374,922 | 341,280 | 338,950 |
| `lts__t_sectors_op_write` [sector] | 293,867 | 295,375 | 373,102 | 367,659 |
| `lts__t_sectors_lookup_miss` [sector] | 0 | 64 | 8 | 22 |

The `sm__icc_*` values retain their actual `cycle` unit despite the
metric names containing "requests". No conversion to request count or
miss penalty is applied. L2 sector totals do not identify traffic as
buffer data: instruction traffic and other request sources must be
separated before interpreting them as buffer placement costs. The
instruction delivery counters decrease in both rolled contrasts;
K3's slower execution demonstrates that this improvement alone does
not determine the best policy.

## Exact counters and sampled stalls remain separate

All source sums equal their corresponding software execution metrics
exactly (`inst_executed`, `thread_inst_executed`, and
`thread_inst_executed_true`). The hardware metric
`smsp__inst_executed.sum` exceeds the exact source warp sum by 336, 336,
1,647 and 1,638 instructions respectively. These small residuals remain
unresolved in the artifacts; no offset is fitted or silently removed.

| Sample observable | R5 full | R5 Newton rolled | K3 full | K3 both rolled |
|---|---:|---:|---:|---:|
| Global `smsp__pcsamp_sample_count` | 2,881,601 | 2,559,279 | 3,033,914 | 3,323,835 |
| Sum of source `# Samples` | 2,866,034 | 2,559,279 | 3,033,914 | 3,323,835 |
| Source `stall_no_inst` | 570,358 | 67,965 | 225,951 | 78,760 |
| Source `stall_no_inst (Not Issued)` | 419,799 | 20,845 | 56,391 | 20,930 |

R5 full has a further unresolved 15,567-sample aggregate/source
difference. Its global no-instruction sampled counts are 570,712 and
420,151, while the source sums are as shown above. The other three
profiles' corresponding aggregate/source totals agree. Sampling
observations are not exact executions or a count of lost cycles.
Multiplying them by the sampling interval cannot establish a latency.
The exported unit of the aggregate sampled stall metrics is `inst`;
the sampling mechanism still governs their interpretation.

Nsight documents separate replay passes for software-patched execution
metrics, and notes that All and Not-issued samples can come from
different passes. This establishes why the measurement mechanisms must
remain distinct; it does not establish the cause of either unresolved
residual here. [Profiling overhead](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#overhead),
[Source Page sampling](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#source-page).

## Consequences for the physical model

The pre-compile model must describe reachable helper bodies and loop
expansion separately from symbolic iteration work. A complete cubin's
instruction bytes are not a sufficient proxy for the executed body,
as the K3 full case demonstrates. Even the exact executed-address union
is insufficient to infer a temporal cache working set.

For R5, unchanged iteration labels, increased exact instruction work,
unchanged allocated occupancy, and sharply improved instruction
delivery counters support investigating instruction delivery as a
source of the speedup. They do not isolate its cycle contribution.
For K3, rolling improves the same delivery metrics while increasing
issued work and runtime; a smaller instruction footprint alone would
select the slower contrast. Both outcomes must be represented before
proposing a family default. These measurements supply validation
targets and physical constraints, with no fitted timing coefficient,
register formula, cache penalty, or baseline compilation in the shipped
post-codegen heuristic.
