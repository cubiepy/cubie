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

## Source math and instruction probes at 00:00 UTC

The ERK nominal admission repair independently passes six saved schedules
and six fresh actual constructions in `erk_nominal_admission_independent_e1`.
The fixed-loop frontend also removes source-proved empty loop bodies:
captured compile-time guards eliminate both the body and its control
administration. `loop_control_empty_independent_e1` verifies the repair;
runtime-dependent guards are not eligible for this elimination.

All eleven isolated math forms have independent native disassembly and
dataflow PASS. Math admission binds each operation to its actual captured
dispatcher, source function and exact calibrated flags. Independent
`math_owner_independent_cpu_e3` covers four fresh finite-math cases and
Fabbri Radau3, including all 250 Fabbri math forms.

`source_replay_point.py` binds actual system defaults and source-proved
caller initialization solely to the numerical certificate. It does not
turn dynamic prediction inputs into codegen constants. Fabbri Radau3
replays all 21,228 source nodes with 115 bound live-ins, eight explicit
infinity operations, no NaN and finite boundary values. A separately
written evaluator reproduces these results in
`verification/source_replay_independent_20260905`. At the source allocator's
255-GPR budget the typed plan has a 2,420-byte local frame; this is a model
allocation, not a measured native register/frame label.

The default zero ACh parameter produces positive infinity in a negative
power before a reciprocal returns a finite contribution. One isolated
native FP32 functional probe checks all five exposed intermediate/output
columns in 14,336 rows exactly, without timing. Independent fresh native
disassembly and array verification pass in `math_exception_independent_e1`.
The math catalog extension adds LG2 and EX2 using the existing published
generic MUFU 15/14-cycle transfers and NVIDIA SFU throughput. Independent
`math_catalog_independent_e1` confirms every pre-existing catalog entry is
unchanged.

`instruction_fill_probe.py` has five native-reviewed register-only images
in `instruction_fill_e3`: 32/192/256 KiB streaming payloads and a shared-PC
cold/warm victim behind 32/256 KiB aggressors. The first two preparation
epochs remain failed-native provenance: one had a PTX conversion typo;
the next let the compiler fold uniform arithmetic coefficients into
constant-memory operands. The current image loads runtime coefficients
and recurring pointers before the initial barrier. All hot arithmetic
operands are registers and the images have zero local spills.

Ordinary N/2N measurements cover one, four, eight, sixteen and thirty-two
active full warps, with 1024 allocated threads per CTA and two complete
occupancy waves. At one warp the 32 KiB stream is about 1.018 cycles per
FFMA; the 256 KiB stream is about 5.534. The victim cold-minus-warm
composite is 18 cycles behind the small aggressor and 300 behind the large
one. These remain inclusive native motifs, not intrinsic refill latency.

Fourteen initial Nsight captures complete. Their aggregate instruction
cache hit/miss counts conserve, but some L2 request totals disagree with
the separately collected hit/miss fields. No exact refill service is
assigned from those inconsistent aggregates. A matched capture with an
explicit unprofiled code warmup will test first-pass L2-state effects.
The persistent elevated profiler remains available without another login.

The joint finite candidate evaluator and instruction-fetch scheduling
integration are active author work. Family defaults and frozen native
holdout validation remain incomplete.

## Joint evaluation and refill controls at 01:45 UTC

Independent `nominal_instruction_independent_e2` passes the instruction
delivery integration: source-projected instruction addresses, exact
pending fills and finite LRU state, independent warp progress, branch
readiness, joint data dependencies and carried state across occupancy
waves. Disabled delivery reproduces the previous engine exactly. The
enabled service remains an explicit demand-fetch hypothesis using
qualified published transfers; automatic prefetch is not identified.

Independent `joint_policy_independent_e3` passes 221 saved costs, 113
graph/plan/address joins, exact rational minimax and all hardware register
plateaus from 1 through 255 registers. Five 65-wave cache cases agree
exactly with ordinary execution. Eight fresh cases compare different
residencies using the same 192 attempted steps. These are finite candidate
and scenario checks, not validation of family defaults. Equivalent event
schedules can share numerical execution while retaining distinct source
candidate and allocation identities; the memoized schedule's plan hash
identifies its representative.

Source-cap construction is feasible for the actual implicit endpoints,
including Radau3 BiCGSTAB with eight Newton bodies and nine Krylov bodies.
The cap cost cohort is running separately from the smaller regime bank.
The completed 32-state chain bank selects full/local among its tested
actions under both perfect and qualified demand-fetch scenarios. Shared
placement reduces modeled registers but loses occupancy. These are source
allocator and scheduler predictions, not native register observations.
Fabbri's first non-Lorenz epoch stopped because its copied source tree
omitted the CellML fixture. The failed epoch remains intact; a complete
source snapshot is required before resuming that case.

The persistent profiler supports bounded `cache_control` and `replay_mode`
enums. Independent `profiler_controls_independent_e1` checks both unchanged
defaults and every admitted combination. The reviewed script was mirrored
to the fixed research runtime, then the idle worker restarted inside its
existing elevated supervisor. Deployment hashes are retained under
`verification/profiler_controls_deployment_e1`; no new login was needed.

Controlled epochs e6/e7 reconcile GCC returned sectors with total L2
sectors in all eight captures. The small stream also reconciles L2 hits
and misses exactly: flushed replay has misses, while application replay
with the explicit warmup has zero misses. Large-stream hit/miss totals
remain sensitive to separate profiling passes, especially at 32 warps.
Independent `instruction_fill_e6_e7_independent` retains those residuals;
they do not establish an exact target refill service.

Independent `instruction_fill_occupancy_independent_e1` passes 64 raw
arrays from the same two streaming images at 256/512/768/1024 threads per
CTA. The first three geometries all admit 48 resident warps, but their
warp clock intervals differ. CTA grouping and execution overlap therefore
remain relevant; these medians are not whole-SM bandwidth measurements.

The new `instruction_gap_e1` contains 18 reviewed native images and 288
ordinary arrays. Actual clock-to-victim separations range from 224 to
16,608 bytes. The large-aggressor cold composite rises with separation,
but the closest image also has a slower warm reference. Cold-minus-warm
alone would conceal that anomaly. The timed control form changes at the
4,320-byte separation, so absolute comparisons must retain that distinction.
Independent raw/native review and twelve matched counter captures are in
progress. No prefetch depth or intrinsic latency is assigned from this
curve.

`native_policy_holdout.py` and its document have CPU author receipts for
sixteen constructors across four families, source prediction freezes,
geometry checks, numerical retention and paired timing arithmetic. The
runner is undergoing independent review before the first native bank.
Predictions must be frozen before compiling any corresponding candidates;
native resource labels and timings are validation outputs only.

The subsequent independent runner receipt is
`verification/native_holdout_independent_20260905/cohort_e1/receipt.json`.
It passes sixteen fresh source constructors, a complete CPU freeze,
strict numerical retention and independent timing arithmetic. The first
ten-workload Lorenz prediction freeze is now retained at
`native_holdout_pilot10_e1/frozen`, manifest SHA256
`ef4018ee0d7a6f6c30c325c8075f8989facf4e1beff84317f3192c5dcbe55c82`.
Its four actions per workload and independent full/local duplicates will
use the unchanged landscape batch and duration. Native execution has not
begun at this checkpoint.

The earlier snapshot directory named `research_checkpoint_20260906_0002utc`
was created on September 5 near 00:02 UTC. Its directory date is a naming
error; its bytes and manifest remain preserved under that original name.

The joint heuristic, verified per-family defaults and frozen native
holdout validation are still incomplete. No production defaults change in
this checkpoint.

## Frozen native bank and caller coverage at 02:38 UTC

The ten-workload Lorenz bank completes in
`native_holdout_pilot10_e1/native_e2`: fifty compiled entries, including
ten independent baseline duplicates, and 3,100 retained timing samples.
Every status reports success and every endpoint is finite. All duplicate
arrays match exactly. Independent `native_e2/bank_conservation_receipt.json`
under `verification/native_holdout_independent_20260905` passes source
bindings, native resources, compiled occupancy, two-wave admission, array
hashes and paired timing arithmetic.

The bank's strict numerical validation does **not** pass. Twenty-seven
of forty policy actions fail the frozen cross-policy agreement gate,
producing 1,674 retained failure records. The gate uses the configured
local tolerances; it is not a global trajectory-error proof. Failed
actions retain their raw timings but are ineligible for fastest-policy
selection. A zero regret against a baseline-only eligible set therefore
does not validate a default. `VALIDATION_DIAGNOSTICS.md` and its bound JSON
retain descriptive timing ratios and failure magnitudes separately from
selection eligibility. Tolerances and prediction inputs remain unchanged.

The initial native run stopped at an installed-backend API mismatch:
the MLIR CUFunc object has no `attrs` member. Its failed epoch and cubins
remain preserved; it produced no valid timing records. The independently
reviewed runner amendment uses the dispatcher's public native-resource
queries, backed by checked CUDA driver attribute calls. The amendment
explicitly binds the original prediction manifest and both runner hashes.
It permits only that runner-file change; every other frozen source asset
was checked before the successful run. Amendment receipts are under
`native_holdout_independent_20260905/amendment_e1`.

Forty separate counter builds complete after the counter-free bank in
`native_holdout_iteration_diagnostics_e1`. Independent
`iteration_diagnostics_e1/receipt.json` confirms each instrumented build
matches its own counter-free states and statuses byte for byte. Seventeen
builds have lower resident-block counts after instrumentation. Counts
are diagnostic outputs, excluded from prediction; endpoint identity does
not establish identical internal counter-free iteration histories.
Cross-policy aggregate Newton, linear and attempt totals differ by at
most 0.01411% of full/local totals. This does not isolate the cause of
timing differences. Relative changes in rejected-step counts can be
large when baseline counts are tiny. The counter named `krylov` also
counts LU calls and is interpreted as linear iterations in analysis.

The Fabbri ANS1 replay exposes an infinity-valued captured scalar and
valid nonfinite intermediates before finite output. The isolated repair
decodes exact scalar payloads, admits supported IEEE intermediate values
and retains rejection of NaN, undefined arithmetic and nonfinite boundary
values. Independent `scalar_snapshot_independent_e2` passes the actual
23,529-node graph and its 179 finite boundary/observable payloads. The
reviewed source bytes are integrated only after all native bank and
counter diagnostics finish. The earlier smaller Fabbri certificate did
not cover this exact ANS1 source snapshot.

The historical unroll label decoder is also explicit: `1` means full,
`0` means a counted loop with unroll count one, and `n` means False.
The first Fabbri candidate epoch incorrectly interpreted `0` as False;
it is preserved with `MISENCODED_HISTORICAL_LABEL.json` and is not used
as historical-policy evidence. The corrected cohort remains in progress.

Independent `repeated_step_independent_e1` passes one preceding identical
step with the same physical storage, cache objects, absolute time and
pending service state carried into at least two measured waves. Its
suffix agrees exactly with uninterrupted execution. This is a drained,
synchronized compute-boundary sensitivity; it does not reproduce the
actual asynchronous outer loop of each warp. Perfect instruction delivery
is a service baseline, not a proved lower bound for the greedy scheduler:
changed issue ties can improve a schedule with nonzero fetch service.

The caller-liveness extension is active author work. Actual folded outer
loop control and helper demand determine values used after the step call,
including FP64 time scalars occupying two words each and persistent
controller cells. Existing step aliases must join these identities.
Addressable versus promoted cells require fresh placement-specific
allocation; a fixed register increment is not an admissible repair.
An exact integer-tick execution engine is separately under review to
reduce rational-arithmetic CPU cost without changing event ordering,
hazards or numerical results. Neither unfinished component qualifies a
complete joint policy model.

## Complete-region instruction probes

Independent gap reviews cover all eighteen earlier images, 288 ordinary
arrays and twelve matched profiles. They verify executed paths and exact
endpoints while retaining three large-stream cold L2 counter residuals.
The nearest large-aggressor warm reference samples instruction-delivery
stalls in a checksum sector omitted by its victim-only prime. This is
consistent with incomplete warming, but does not prove a unique cause or
identify a refill latency.

`instruction_region_probe.py` primes the same complete native region,
including both clocks, victim arithmetic, checksum and guard. Warm mode
discards the first interval and retains the second. Native clock counts
are doubled in warm mode; retained timestamp counts remain unchanged.
The first optimized images move padding outside the timed gap, collapsing
all requested separations to 32 bytes. They remain failed-layout evidence
in `instruction_region_e1`; no GPU measurements use those images.

The separate optimization-level-zero epoch retains eighteen distinct
requested layouts, with actual separations from 320 through 16,832 bytes.
Independent `instruction_region_layout_20260905/native_o0_independent_e1`
passes complete native interpretation, identical timed PCs, exact clock
and endpoint counts, unexecuted padding, 34 registers and zero spills.
Ordinary measurements complete in `instruction_region_o0_e1` at two
complete occupancy waves. Independent `ordinary_independent_e1` verifies
288 records, 33,030,144 endpoint cells and 3,096,576 active timestamps.
Both aggressor sizes give zero cold-minus-warm cycles at the 320-byte
gap; larger small-aggressor gaps give nine cycles. Large-aggressor gaps
give nine, 59, 245--246, 280--281, 280--281, 265--266, 262--263 and 256
cycles in ascending separation order. Twelve matched profiles are queued
with application replay, explicit warmup and no cache flush. These
composites retain their native scheduling and control costs.
Extra native MOVs and different scheduling
distinguish this form from the optimized earlier probe; it is not an
exact causal replication of that warm-reference anomaly. No intrinsic
fill latency or fitted prefetch depth is assigned.
