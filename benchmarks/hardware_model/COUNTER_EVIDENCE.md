# Matched Lorenz iteration-counter evidence

Recorded 2026-09-04. A CPU audit verified ten completed policy-specific
counter label cases against their saved arrays and original uninstrumented
bank. The [audit receipt](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/dependency_counter_audit_20260904.json)
contains every label/NPZ hash, source and compiler comparison, compiled
config/cubin identity, geometry, per-run sums/distributions and exact bank
timing record paths/line numbers. No GPU work was launched for this audit.

## What was verified

Each case retained one full-grid counter sample and its own state-only
reference: 262,144 runs, block size 64, duration 2 for Kvaerno3/BiCGSTAB
or 8 for Radau IIA5/BiCGSTAB. Both compiled kernels meet the two-wave
occupancy requirement. Every saved input/output array matched its recorded
shape, dtype and value hash. Every counter sample's complete state array
exactly equals its own reference, all states are finite, status arrays are
identical and zero, and each solve uses one chunk. This equality is within
each policy; it does not assert identical trajectories across policies.

The state-only reference cubin bytes match the original bank cubin exactly.
The source hash, compiler identity, explicit policy, duration and input
grid match the associated completed cohort. Both reference and instrumented
compiled config hashes and cubin hashes are retained. Counter-label runs
have no independent clock snapshots; their labels establish workload
observations, not a new timing comparison.

All raw counter arrays are `int32` with shape `(2, 4, 262144)` in
`(save_row, counter, run)` order. Their first save row is exactly zero.
Summing saved intervals in int64 reproduces every stored per-run total;
independently reconstructed values/counts, means and totals reproduce
every stored distribution. The channels are Newton iterations, total
linear-solver iterations, attempted steps and rejected steps. Attempted
steps include rejected attempts; the older output-size docstring's
"accepted steps" wording does not describe the increment implementation.
Counter stores, interval resets and exact multi-stage tests are cited in
[COUNTER_PROTOCOL.md](/C:/local_working_projects/cubie-worktrees/hardware-unroll-placement/benchmarks/hardware_model/COUNTER_PROTOCOL.md:63).

## Policy contrasts

The table gives per-run means of the summed counter intervals. `N/K` are
the last two unroll-policy characters; the other six groups are full.
`1` means full expansion, `0` is the explicit rolled `(True, 1)` directive,
and `2`/`4` are expansion counts. The last column is the minimum of the
matched **original uninstrumented bank** samples; it is descriptive,
not a statistic from the instrumented counter run or a fitted model.
The [bank records](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/lorenz_split_bridge_e1/records.jsonl)
and their exact per-policy joins are retained in the audit receipt.

| Family | N/K | Newton mean | Linear mean | Attempted mean | Rejected mean | Bank minimum ms |
|---|---|---:|---:|---:|---:|---:|
| Kvaerno3 | 1/1 | 2484.926208 | 3559.904106 | 583.821526 | 0.126995 | 21.0759 |
| Kvaerno3 | 0/0 | 2484.722607 | 3559.645679 | 583.826611 | 0.123863 | 23.0541 |
| Kvaerno3 | 0/1 | 2484.963261 | 3559.934036 | 583.819984 | 0.125713 | 23.0359 |
| Kvaerno3 | 1/0 | 2484.800613 | 3559.734356 | 583.825844 | 0.124573 | 21.7012 |
| Kvaerno3 | 2/1 | 2484.963261 | 3559.934036 | 583.819984 | 0.125713 | 24.2475 |
| Kvaerno3 | 4/1 | 2484.963322 | 3559.934113 | 583.819981 | 0.125713 | 23.2627 |
| Radau IIA5 | 1/1 | 676.463959 | 1454.907234 | 338.095795 | 0.921448 | 19.9575 |
| Radau IIA5 | 0/0 | 676.492004 | 1454.945675 | 338.097244 | 0.921410 | 19.3469 |
| Radau IIA5 | 0/1 | 676.463959 | 1454.907234 | 338.095795 | 0.921448 | 17.8596 |
| Radau IIA5 | 1/0 | 676.492004 | 1454.945675 | 338.097244 | 0.921410 | 18.2266 |

Relative to full expansion, the Kvaerno3 Newton, linear and attempted
means change by less than 0.01%. Its rejected-step mean changes by up to
2.466% from a small baseline of about 0.127 rejected attempts per run.
Its largest descriptive minimum-time increase is 15.048% for N/K=2/1.
Tiny aggregate changes do not mean identical per-run work: depending on
policy, 208,575 to 261,613 of the 262,144 Kvaerno3 per-run four-channel
vectors differ from full expansion.

For Radau IIA5, N/K=0/1 has exactly the same four-channel total vector
as full expansion for **every run**, alongside a 10.512% smaller original
bank minimum. N/K=0/0 and 1/0 share the same aggregate means, but each
differs from full in 139,117 per-run vectors. These observations make
total iteration volume alone an inadequate explanation of the bank
timing contrasts; they do not uniquely identify register pressure,
instruction-cache capacity or any other compiled-code mechanism.

## Interpretation limits

Counters sum work within each lane over save intervals. They do not retain
the individual stage/solve sequence, per-Newton Krylov counts, convergence
predicates or warp votes. Even exactly equal per-run totals do not establish
equal warp-coherent loop maxima at every dynamic call. No warp-max estimate
is fabricated from lane totals. These measured counters are validation
labels and are never inputs to a pre-compile heuristic.

Instrumentation changes compiled resources. For Kvaerno3/full, the
state-only reference has 96 registers/thread and 10 resident blocks/SM;
the counter kernel has 106 registers/thread and 8 resident blocks/SM.
Their grids have respectively 7.314 and 9.143 theoretical occupancy waves.
Thus structured instrumented timing is neither extracted nor substituted
for the bank. Raw Solver diagnostic logs are retained and can contain
printed instrumented timings; those lines are excluded from the model
and from every timing number in this document.

All ten cases contain one counter sample, so this audit establishes exact
within-sample checks rather than repeated-run counter stability. The
original bank's duration and measurement protocol remain attached to its
timings, including samples below 20 ms; label collection does not repair
or relabel the earlier timing protocol.

The two full-policy raw label receipts are
[Kvaerno3](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/counter_gate_kvaerno3_bicgstab_20260904_v2/u11111111-bs64/labels.json)
and [Radau IIA5](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/counter_gate_radau_iia_5_bicgstab_20260904_v2/u11111111-bs64/labels.json).
The eight contrast label/NPZ paths and all exact identities appear in the
audit receipt's `counter_cases` entries.
