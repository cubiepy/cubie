# Continuation: hardware policy model, 2026-09-05

The active development worktree is
`C:/local_working_projects/cubie-worktrees/hardware-policy-estimator`, on
`codex/hardware-policy-continuation`. The earlier continuation worktree is
detached at checkpoint 134822bf and preserves its measurement provenance.
The branch starts from `origin/main` 1391bf35
and carries the prior research through c36143b5 by fast-forward. PR #910's
Newton/Krylov split is on main; PR #909's counter correction is present as
bc573e3b. Main and the frozen ff3a567f measurement tree remain untouched.

The six recovered files were copied from the external handoff snapshot only
after checking every SHA256 in its manifest. Their prior unverified status
was retained. This continuation repairs and independently evaluates them.

## Verified component changes

The selector binds an actual policy graph, its complete typed plan, all eight
loop directives, every registry placement, and global shared stride. Its
comparison uses the same number of warp attempts for every occupancy.
Exposed fetch stalls must bind to the same execution wave; total fetch service
cannot be added as if it did not overlap execution. Subgroup allocations are
excluded from finite ranking until per-lane spill/merge conservation exists.

Placement constructs separate actual local/shared graphs and fresh allocation
for each materialization hypothesis. It compares identical explicit iteration
regimes and branch assumptions. Sector/bank demand uses actual issue masks;
local-store write-through and write-back remain explicit alternatives. These
cache alternatives are hypotheses, not inferred cache-hit labels.

Policy construction preserves stage ownership, complete Newton/linear-call
coverage, nested Krylov identity, dynamic induction witnesses and Boolean
source replay. Seven untouched actual source/plan pairs pass independent
review, including separate Newton/Krylov count2 for minimal residual. The
trust boundary is the extractor and hash-bound artifacts; invariant checks
are not a complete independent interpreter of arbitrary modified graphs.

The independent placement/selector receipt is
`verification/placement_selector_independent_20260905/receipt.json`, SHA256
`7518610d6b658f986d690f17c285b43d4503d7260f7cd1e4098c4f81ba16ee5a`.
It checks 36 actual arms across DIRK/FIRK/Rosenbrock and LU/MR/BiCGSTAB,
six selector policies, 8,160 occupancy cases against CUDA 13.3 equations,
common-work scheduling and exact minimax arithmetic.

Policy review is recorded under
`verification/cpu_continuation_independent_20260905/independent_physical_policy_e2`.
Deliberately corrupted-graph admission limitations remain in the adjacent
`independent_policy_e1` records, rather than being reported as failures of
the untouched constructors.

## GPU evidence

One elevated persistent profiler host was started through Windows RunAs.
Supervisor PID 23660 and worker PID 75936 serve session
`_profiler_sessions/continuation_20260905`. Two pings verify the same elevated
worker. The four previously prepared local257 requests completed with profile
and import exit zero. The queue is empty and disarmed while root runs other
GPU work; the host remains available without another elevation prompt.

The six pending instruction-cache captures independently pass exact native
and ordinary-counterpart audits. Eight-warp-capacity 131184 to 135280-byte
bodies decline from 16.359 to 11.585 trillion scalar FFMA/s; sixteen-warp-
capacity 139376 to 147568-byte bodies decline from 13.040 to 12.780.
These captures report residency capacity, not achieved occupancy. They do
not supply a universal miss penalty or decode a physical cache-sharing map.

All four local33 and all four local257 profiles pass native workload,
SourceCounters, certificate and output checks. L2 hit-plus-miss conservation
fails for local33 at 32 timed warps and for every local257 capture. Those
raw residuals remain recorded and prevent exact L2-service inference.
The local33 one-warp controls show downstream L2 writes despite L1 store
hits, so eliminating all downstream traffic for an L1 write hit is unjustified.

Fresh profile receipts and findings are under
`verification/continuation_profiles_independent_20260905`:

- `icache/receipt.json`: SHA256
  `4128ca6c18479d2922f5090ceba897628df2872a2c775e66e0ce1346eb12b8b2`.
- `local33/receipt.json`: SHA256
  `50ca73631fa27e28563627b2b1ecf40f8c4de7329b42317d181360d1b9553606`.
- `local257/receipt.json`: SHA256
  `d2962e63d795dcc11ee48fc6729636a147cd38e88a0c42500ac083434b235788`.

All relative raw paths above resolve under
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.

## Repository verification and active model work

The unchanged imported production source passes all complete suites:
3,570 simulator tests, 3,631 MLIR GPU tests, and 3,625 numba-cuda GPU tests.
Blocking Flake8 passes. The default A/B gate passes on both installed CUDA
backends, including the two-full-wave case. Full, untruncated logs are under
`verification/continuation_repository_20260905`.

Instruction-footprint forecasts and nominal hardware-service estimates are
active research. The complete joint heuristic, per-family defaults, and
frozen holdout timing validation are not complete. Component verification
does not establish native allocation accuracy or whole-kernel timing accuracy.
No solver timing regressions, fitted register multipliers or empirical family
winner tables have been introduced.

## Source-model continuation after checkpoint 134822bf

Surviving runtime addresses carry exact affine byte expressions into typed
IMAD and local/shared accesses. A single dynamic access prevents scalar
promotion of its whole aliased storage. Nine actual DIRK/FIRK/Rosenbrock
and LU/MR/BiCGSTAB combinations pass independent address checks under
`verification/dynamic_address_independent_20260905/cohort_e4`.

Captured NumPy table views use the installed backend's contiguous-copy
materialization. Their runtime indices produce IMAD/LDC operations before
allocation, with a separate immutable constant-memory identity. All nine
combinations pass independent allocation and address checks in
`verification/cpu_continuation_independent_20260905/captured_lookup_independent_e1`.
Content identity does not establish physical deduplication of separate
closure objects in constant memory.

Shared read forwarding is an explicit compiler alternative restricted to
proved straight-line source regions. Stores and control operations remain.
Fresh allocation captures longer retained-value live ranges and changed
spills. Independent actual-family checks are recorded in
`verification/shared_forwarding_independent_20260905`. This is not a claim
that the native backend performs every eligible forwarding operation.

The conditional execution scheduler separates dispatch, coupled FP32/INT32
capacity, register dependencies, memory visibility and source consumption.
Its independent hand-computed checks are in
`verification/nominal_execution_independent_20260905`. The sourced nominal
catalog and its architecture transfers remain estimates; scheduler logic
verification is not whole-solver prediction validation.

[Collective measurements](COLLECTIVE_SERVICE_EVIDENCE.md) add independently
checked all-vote and ballot/compare motifs on the target GPU. Both population
profiles use the persistent elevated session, with no additional prompt.

The larger source cohort at `verification/chain32_policy_cost_20260905/e2`
preserves three completed Kvaerno3/LU forecasts and one allocation failure.
The inline homogeneous-cap projections are 494064 bytes for full,
92208 bytes for Newton count1, and 188880 bytes for stage count1. These
covered-body forecasts exclude unmodeled caller and native code forms.
The failed accumulator-count1 case exposed 192 internal induction visits
represented as simultaneous entry registers. Correct loop initialization
and increment producers are required before ranking those allocations.
The cohort receipt records its partial status and source-snapshot coverage.

## Source and execution review checkpoint at 23:25 UTC

The induction repair independently passes under
`verification/cpu_continuation_independent_20260905/induction_independent_e2`.
The chain32 rolled accumulator has 73 actual entry values; its internal
iteration values have source-derived MOV/IADD producers and ordinary
lifetimes. The earlier 255-register entry failure is preserved as a model
defect, rather than used to exclude that policy.

ERK source coverage independently passes 60 retained cases and 12 fresh
constructions under
`verification/cpu_continuation_independent_20260905/erk_independent_e1`.
This includes fresh, accepted-cache and rejected-cache FSAL paths, exact
RHS counts, and source-proved fixed extents for counted stage slices.
The nominal scheduler's separate explicit-plan admission repair is authored
and undergoing independent review.

The stateful nominal cache scheduler independently passes ten direct
accounting checks and eight fresh source scenarios under
`verification/nominal_cache_independent_20260905`. Cache state changes only
after selected issue; pending fills have readiness times, and successive
waves share an explicit cache/backing state. Local/shared capacity follows
the source layout, register occupancy and shared carveout. LRU organization,
equal per-SM L2 capacity share, unlimited pending fills and store visibility
remain declared hypotheses. Downstream writeback timing is not identified.

Synthetic instruction-address projection independently passes 21 actual
cases and 26,723 allocated executable events under
`verification/instruction_addresses_independent_20260905`. Rolled visits
reuse addresses; cap copies occupy their lexical nested positions. For the
chain32 example, a 640,512-byte reserved envelope contains an accessed union
of only 129,824 bytes in the selected regime. Neither the envelope nor that
union replaces temporal fetch/cache modeling.

Fixed-loop comparisons and backedges were present in footprint supplements
but absent from execution schedules. The new loop-control author cohort
covers 21 actual cases, including 18/8/4 completed count1/count2/count4
accumulator chunks for Lorenz. Independent review found a missing control
edge on an internal address operation from a multi-instruction expansion.
The repaired implementation independently passes 27 cases under
`verification/cpu_continuation_independent_20260905/loop_control_independent_e2`.
The original missing-edge finding remains in the preceding review epoch.

Fabbri source extraction exposed missing exp/log/real-power primitives and
literal fractions such as `precision(1/60)`. Eleven isolated FP32 native
forms are retained under `math_forms_e1`, without kernel launches or timing
measurements. Default MLIR AFN/FTZ lowering uses FMUL/EX2 for exp,
LG2/FMUL for log, and LG2/scale/EX2 for real powers. Source coverage and
domain-respecting numerical replay remain under development; the native
form probe is awaiting independent review. Fabbri Radau3 extraction reaches
the complete graph in `verification/fabbri_source_20260905/radau3_full_e7`.
Its strict numerical check rejects the arbitrary signed live-in probes at
a real power. The unverified graph and exact failure are preserved; source
defaults and proven caller initialization will supply a valid replay point.

The complete joint heuristic, per-family defaults, instruction-delivery
service and frozen holdout validation remain active work. These component
reviews do not establish native allocation accuracy or completed defaults.
