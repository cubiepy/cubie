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

## Exact scheduling and native lowering gaps at 03:02 UTC

The fixed-service scheduler has an independently verified integer-clock
implementation. Its tick unit is the least common multiple of the exact
service denominators; no hardware parameter or time is rounded. Independent
`nominal_integer_independent_e1` covers all ten saved algorithm/inner
plans, exact issue traces, byte/register hazards, pair completion, large
coprime denominators and arbitrary-precision times. The profiled source
case retains all 16,968 issue rows and runs about four times faster on
the CPU. That speed measurement is excluded from hardware prediction.

Independent `nominal_integer_integration_independent_e1` passes the joint
harness dispatch. Twelve actual old/new scenarios preserve complete
results, errors and identities. Only fixed-service scenarios with admitted
disabled or omitted instruction fetch use integer execution. Stateful
data and instruction caches retain the original Fraction engine. The
integer source's later hash change is exactly trailing-newline cleanup;
its AST and all earlier bytes are independently checked. Existing frozen
source-cap and Fabbri epochs remain untouched.

The completed Fabbri exception epoch retains 24 finite costs and seven
candidate rejections. Only full/local and accumulator-count1/local are
admitted. The requested historical exceptions are not yet compared:
`norms.py:481` requires runtime integer floor division by the positive
source state count, and shared-stage lowering exposes a promoted-copy
dependency gap. The exact requests and failures remain in
`verification/joint_policy_author_20260905/fabbri_exception_design_e2`.
The source repairs are separate author/reviewer work, not changes to the
historical policy interpretation or numerical admission.

Offline native diagnosis of fifteen saved images is retained in
`verification/native_policy_code_diagnosis_e1`. RK23 stage-count1/local
performs indexed coefficient loads and local read/modify/write of its
stage accumulator; full expansion removes that storage and folds known
coefficients and zero terms. Its complete native images are about 5 KiB,
so an instruction-capacity explanation alone is insufficient.

Kvaerno3/LU stage-count1/local has no native local frame. Dynamic indexing
instead becomes scalar register selection: its 95-instruction successor
body contains three FFMAs plus predicate/move networks selecting and
updating the nine accumulator homes. This is a missing conditional native
form in the addressable-only dynamic-index lowerer. The register-selection
form must retain every reachable home, selection instruction, temporary
and source guard in fresh allocation. Three observed kernels do not
justify an empirical array-size cutoff for choosing the compiler form.

Radau3/BiCGSTAB also contains wider register-selection loops. Its 79
MATCH.ANY sites are traced to active-mask uniformity checks before votes,
not coefficient broadcasts. The full baseline uses different native
uniformity checks. Both complete local images exceed 400 KiB, but their
entire static spans are not hot instruction footprints. Dynamic per-PC
execution and stalls require separate solver captures. These observations
identify concrete lowering differences without assigning a fitted share
of solver runtime to them.

All twelve complete-region O0 profiles finish. Independent
`instruction_region_layout_20260905/profile_independent_e2` verifies 82
saved application-process arrays, ten reported NCU replay passes and two
PC-sampling passes per capture. Process counts and replay-pass counts are
distinct. Executed instructions, hot PCs, doubled warm clock counts,
unchanged timestamp stores and zero padding execution all agree exactly.
Ten L2 hit/miss partitions conserve. Cold gaps 576 and 16,832 bytes retain
residuals of 27,520 and four sectors respectively; the first also has an
opposite GCC-return/L2-total residual. Those captures cannot identify an
exact L2 service. Warm PC samples include both discarded and retained
intervals and cannot identify stalls in the second interval alone.

`verification/instruction_region_comparison_e1/comparison.png` and its PDF
show the two native epochs with exact underlying CSV data. Independent
`instruction_region_comparison_independent_e1` verifies all 36 rows,
native gaps, six paired cold/warm differences, medians and ranges. The
caption retains the different compiler levels and the old O3 control-form
change at a 4,320-byte gap. No curve is fitted. Six reviewed O0 images
also complete ordinary measurements at eight, sixteen and thirty-two
active warps; their allocation remains thirty-two warps per CTA.
Independent `ordinary_population_independent_e1` verifies all 288 records,
33,030,144 endpoint cells and 57,802,752 active timestamps. At thirty-two
active warps, the large-body 1,472-byte gap gives paired differences of
230--284 cycles and the 16,832-byte gap gives 279--295 cycles. The wider
range remains explicit; these are active-population sensitivities, not
measurements at eight or sixteen allocated resident warps.

## 2026-09-05 03:49 UTC continuation

The isolated caller revision passes independent review in
`verification/caller_independent_e2`: seventeen actual source/config
inventories, 132 allocations and 1,987 scalar CFG nodes. RK23 eliminates
the proved dead counter read. Rosenbrock retains the proof of its exact
pre-step zero and rematerializes that value; its adjacent linear counter
remains part of the step. These are conditional caller-state allocations,
not calibrated native register counts. Integration into the joint harness
is separate work and remains incomplete at this checkpoint.

`verification/constant_division_independent_e2` passes the isolated
source-bound integer forms with 151,667 exact arithmetic checks and
complete Fabbri/chain32 graph and allocation rebuilds. Complete source
bounds, rather than selected iteration witnesses, justify the reciprocal
multiply or power-of-two shift. The independent native functional audit
checks seven images and 868 exact outputs. Those one-thread launches are
functional checks, with no performance or latency claim. Generic native
integer service transfers remain explicit conditional assumptions.
Combining these forms with promoted-cell forwarding allows the historical
Fabbri Radau3 graph and allocation to build. Instruction-address projection
and joint caller integration are still under author/reviewer work.

The profiler wrapper has independent CPU admission with all six actual
constructors. Its production `UnrollFlags` object repairs the original
dictionary-constructor failure, which remains preserved. Independent
`verification/native_profile_preflight_independent_e1` verifies three
full-unroll native reproductions: Kvaerno3/LU, Radau3/BiCGSTAB and RK23.
Each matches the complete original cubin, SASS, resource allocation,
geometry and both warmup/capture final-state and status arrays. The
original at-least-two-wave launch requirement holds. Those three images
are queued through the existing authenticated profiler, using application
replay, cache control none and the original counter-free output ABI.

The three stage-count-one preflights stop before solving because their
whole cubin bytes differ. Independent ELF classification traces all three
pairs to internal constant-array symbol numbers assigned by the installed
backend's process-global counter. Code, constants, allocation metadata
and relocation payloads match exactly; string/symbol tables and physical
ELF layout differ. Whole-file and raw load-segment identity therefore
remain failed. Any narrower section-bound identity contract requires its
own author/reviewer receipt and fresh preflight; this checkpoint does not
admit those three images for profiling. The original native bank's
cross-policy numerical failures and frozen predictions remain unchanged.

The source-cap cohort completes six workloads with ninety-six finite costs
and is still running. Dynamic local register selection is a separate
conditional compiler-form implementation under review. The complete
joint heuristic, finite family defaults and fresh validation remain
unfinished. No native timing, register label or fitted parameter enters
the prediction changes described here.

## 2026-09-05 04:25 UTC continuation

The reviewed e16 modules are promoted into the active research tree.
`verification/joint_caller_independent_e1` verifies 388 source hashes,
fifty fresh complete plan/projection rebuilds, every scenario/residency/
common-work binding and independently recomputed minimax rankings.
Step-only diagnostics are excluded from the default ranking. Each caller
materialization and placement receives a fresh allocation. Eight Fabbri
historical local/shared/forwarding projections also pass, with seventy
or 105 source-bound division nodes mapped to sixteen-byte instruction
slots. `native_plan.py` remains unchanged. This verifies attempted-step
resources and conditional costs; caller rematerialization remains outside
that interval, and complete-kernel timing/default validation is unfinished.

The profiler's narrow ELF naming amendment passes independent review in
`verification/cubin_equivalence_independent_e2`. All six real image pairs
and exact-self checks pass their separate contracts; all twenty-five
independent code/data/metadata/layout/symbol mutations are rejected. The
full generated-symbol map is injective, including unchanged names. Raw
binary inequality remains explicit for the three rolled images. Fresh
rolled preflights then pass exact own-array and geometry checks in
`verification/native_profile_preflight_independent_e2`.

All six native policy captures complete through the persistent profiler.
`verification/native_policy_counters_independent_e1` checks twenty-two
full-policy and twenty-five rolled-policy application processes. Every
warmup/capture state and status snapshot equals its original candidate.
Hardware and source-PC instruction totals conserve for both Radau cases.
Kvaerno and RK23 retain small unresolved warp/thread/predicated-on
differences despite exact saved endpoints. These residuals are not
normalized away and profiler timings remain outside prediction.

The actual instruction unions are 13,104/11,280 bytes for Kvaerno,
28,320/48,704 for Radau and 4,288/4,672 for RK23, full/rolled respectively.
The corresponding 256-byte touched-line unions are 13,824/11,520,
29,184/49,408 and 4,352/4,864 bytes. Radau's whole static images exceed
400 KiB, but neither observed instruction union reaches 128 KiB. Its
rolled case issues about 5.62 times as many warp instructions as its full
baseline. Comparison and move instructions make up about 71.7 percent
of rolled warp issues. Seven of seventy-nine MATCH sites execute; the
remaining sites are cold in this capture. Static CALL mnemonics also
require native control inspection: Kvaerno's active rolled CALL targets
a local convergence block, not a returning vote helper.

The installed carveout route has a separate verified ownership problem.
`verification/carveout_handle_native_independent_e1` checks fifty-five
successful driver calls, distinct compatibility/actual modules and exact
FP32 endpoint bits. Setting the compatibility preference to 65 leaves
the actual launch function at -1. Independent setter changes leave the
other function's attribute unchanged. The complete source/binary bindings
and the single functional launch are retained. The diagnostic establishes
setter ownership, not physical partition or exact Solver-image identity.

Measured shared partitions are 32/16 KiB for Kvaerno, 16/8 KiB for Radau
and 32/32 KiB for RK23, full/rolled. The original requested 100 KiB is
therefore not an attained hardware-capacity control. Frozen model costs
retain their conditional physical geometries. The next executable
ranking separates requested preferences from legal per-action physical
partition uncertainty, including larger partitions required by shared
placements. No rule is fitted to these six observed partitions.

Dynamic register selection passes thirty-two independent plan/dataflow
rebuilds but still requires complete-source compiler-eligibility proof
for omitted branches and loop paths before joint admission. Pure-proof
caching and partition-envelope arithmetic are isolated author work. The
complete joint heuristic, validated family defaults and a new prediction
freeze remain unfinished.

## 2026-09-05 12:36 UTC continuation

The frozen e6 source-cap cohort completes all ten family/inner workloads
with 160 finite scalar costs, no rejected or missing cells and no changed
source hashes. Its conditional full-unroll/all-local winner agrees with
the earlier pilot for every workload. The comparison is retained in
`verification/joint_policy_author_20260905/`
`source_caps_joint_pilot_comparison_e1.json`. This remains the earlier
step-only, fixed-physical-geometry model; it does not validate defaults.

Reviewed e19/e20 files are promoted into the active research tree.
`verification/verification_cache_independent_e1` admits opt-in pure-proof
caching with complete typed-input/source revalidation and fresh ordinary
construction/allocation. Thirty-two cost/ranking/work results remain
exact. The small complete cohort remains slower with caching; no general
speedup is claimed. The optimization is CPU infrastructure, excluded from
hardware service values.

`verification/joint_partition_independent_e1` verifies 394 source hashes,
four actual plan rebuilds, forty-eight physical geometry calculations and
sixty-eight exact cost links at 6,720 common warp attempts per SM. Its
eight requested actions and two scenarios agree with exhaustive evaluation
of 125,000 joint partition assignments, including attaining witnesses.
Shared placements needing larger partitions remain feasible. Requested
preferences are separate from physical capacity; incomplete scenarios
remain explicit conditional exclusions. The nine-file promotion passes
`verification/e20_active_promotion_independent_e1`. The first corrected
Fabbri ranking stops before issuing costs because captured constant-array
lookup uniformity does not yet recognize source-proved FloorDiv. All four
graphs and unpressured forms are retained. A bounded uniformity repair
requires independent review before a fresh ranking run.

All six intermediate-IR captures pass artifact-retention review in
`verification/native_policy_ir_independent_e2`. Twenty-four inspector
artifacts retain the exact original cached LTOIR and native images.
Stored MLIR, lazy LLVM translation and the separate diagnostic LTO PTX
re-link remain distinct products. The separate author diagnosis reproduces
Kvaerno's exact 19,072 native code bytes and 1,192 PC/instruction texts
through standalone PTXAS from its diagnostic PTX. Its 36-byte dynamic
local depot becomes register selection, with zero local frame/spills.
Tool-note metadata differs, so the complete cubin is not byte-identical.
This establishes capability for that input, not a universal size rule.

RK23's same-sized depot remains addressable local storage. Its native
40-byte frame has local-array loads/stores but zero PTXAS spill counts;
local traffic alone is not evidence of allocator spilling. Offline
address canonicalization preserves its instruction text. A separately
labeled non-equivalent removal of three store sites still retains local
storage. Those counterfactual images are never launched or timed.

Complete-source register-selection e5 passes independent review in
`verification/register_selection_independent_e5`: thirty-two plans,
2,236 affine/home checks, eighteen source controls and 673,400 loop
positions. Earlier alias, receiver, count-tail and short-circuit findings
remain preserved. Unsupported alias containers, call-bearing conditional
evaluation and unproved escapes are explicitly refused. This admits a
conditional SSA SEL form; joint integration and actual compiler choice
remain separate questions.

The controlled actual-function carveout runner passes CPU review, but
its first native preflight fails the strict ELF gate before any launch.
Its failure receipt also encounters an unsupported target-option JSON
value; the complete log and partial binaries are retained under
`controlled_carveout_native_e1`. No controlled physical partition or
performance result is admitted. The persistent authenticated profiler
remains available. Production defaults, the shared backend and the
original failed numerical gates remain unchanged. The complete joint
heuristic and fresh frozen validation are still unfinished.

## 2026-09-05 14:07 UTC continuation

Reviewed e21/e22 components are combined and promoted as e23. The
400-file source epoch retains the exact independently reviewed register
source proof and adds FloorDiv uniformity to immutable table loads.
`verification/register_joint_independent_e1` verifies thirty-six actual
plans, caller-observable unions, fresh allocation and 240 common-work
costs across twelve compiler/forwarding/store scenarios. Register
selection remains a common compiler scenario, not a candidate-specific
choice of its cheapest lowering. The SSA selection form is conditional;
the installed backend's choice is still unproved.

The initial combined Radau interaction cohort has no eligible selected
extents. It remains valid addressable-storage coverage. A separate
stage-count-one/norms-count-one Radau cohort supplies the positive
interaction: six rebuilt plans retain six source divisions and twenty-four
division-to-table-load links each. Four selection plans have one eligible
extent; all six retain twenty-one caller words. Independent review in
`verification/combined_positive_independent_e1` checks the positive
selection/home arithmetic and exact projections. The ten-file active
promotion passes `verification/e23_active_promotion_independent_e1`.
The pinned native allocator and production source/test trees are unchanged.

The corrected Fabbri Radau3 four-action evaluation completes from the
separate immutable e22 epoch. Its source-proved FloorDiv table indices
pass independent admission. It remains a one-Newton-body, perfect-delivery
slice, with legal physical partitions and explicit incomplete promoted
caller-cell scenarios. The author's finite ranking retains full/local;
its ranking receipt awaits independent review. Cache-qualified costs and
the Radau5 comparison remain required before a Fabbri policy conclusion.

Controlled actual-function carveout preflights now pass for all six
preferences: 0, 8, 16, 32, 64 and 100 percent. Earlier PTX and empty-CUDA
callback carriers changed native code and remain failed epochs. Offline
single-input LTO links isolate the extra linked module as the source of
the e3 code change for this input. E4/e5 bind an identical retained kernel
IR callback input; the installed linker's exact-byte deduplication leaves
one input. Fresh IR equality and the original strict native identity gate
both remain required. The stored IR is validation-only provenance.

E4 reaches one exact native/output launch but fails its counter checker:
the raw kernel API exposes a disabled counter's ABI placeholder, whereas
SolveResult returns None according to output selection. E5 checks the
positive disabled source/output flags and logical zero-sized output,
retaining the allocated int32 rank-three placeholder. Activated counters
are refused. No native identity or state/status comparison is relaxed.

`verification/controlled_carveout_native_independent_e5` verifies all six
preflights, twelve exact FP32/SUCCESS snapshots, 156 driver calls and
CUlibrary/kernel/function/module ownership. The common image has 44
registers and a 40-byte addressable local frame. Actual-function occupancy
queries give 7/7/10/10/10/10 blocks per SM, separately from the original
compatibility handle's ten-block result. These queries do not establish
physical cache partition. Six matching application-replay NCU captures
complete under `controlled_carveout_e1_percent000` through `percent100`.
Independent review in `verification/controlled_carveout_profiles_independent_e1`
verifies 270 requested metric values and units, sixty replay processes,
120 exact own-candidate arrays and 1,560 driver calls. Each report records
fourteen replay passes, distinct from its ten saved application processes.
The observed shared partition is 8,192 bytes for preference zero and
102,400 bytes for every nonzero preference. Preference eight retains an
occupancy-API result of seven blocks versus ten in the NCU launch metrics.

Hardware warp, thread and predicated-on instruction totals match across
all six profiles. Exact software PC counters are unavailable: source
exports omit the execution columns and explicit software metric exports
are zero despite nonzero hardware totals. No exact executed path or hot
instruction union is admitted from these captures. Local load misses rise
from approximately 0.434 percent to 46.8 percent; local store misses rise
from approximately 1.52 percent to 31.8 percent. Local hit/miss balances
are exact. TEX and GCC residuals remain explicit and unassigned.

The persistent authenticated profiler is disarmed and remains available.
A separate ordinary event-timing run completes 901 launches on one actual
function and unchanged native image.
`verification/controlled_carveout_timing_native_independent_e1`
passes all 901 output links, 900 event samples, 5,241 driver calls and
the complete schedule. The median paired change against preference sixteen
is -2.335156 percent for zero; the other four comparisons are within
0.361 percent in magnitude. All twenty individual pairs remain reported,
without a significance threshold. A CUPTI activity-only collector is
being prepared to
measure the executed shared partition outside NCU; the profiled partition
is not assumed to transfer to ordinary timing. Profiling cycles and all
native timings remain validation diagnostics, excluded from prediction.

`numerical_contraction.py` separately reproduces the frozen RK23
full/rolled numerical disagreement and changes only the public contract
flag. Independent audit verifies eight exact own-candidate snapshots.
Removing the flag leaves each candidate's output unchanged: the original
cross-policy gate still fails on 122,822 elements in 67,806 trajectories.
Native FFMA remains in both intervention images. Installed fast-math
option translation omits the downstream FMA option when contract is
absent; this is not a verified contraction-elimination experiment.
`NUMERICAL_CONTRACTION.md` retains that qualification and failed gate.

The source-only policy-axis inventory identifies remaining coverage:
independent Newton/Krylov actions, MR split-loop workloads, source-size
and arithmetic-intensity boundaries, and partially converged warps.
Newton's body and commit masks differ: converged lanes still participate
in residual, linear-solve and norm work. The current full-warp scheduler
cannot be made mask-aware by scaling every instruction or memory access.
False loop directives also need a distinct late-unroll compiler alternative
before their retained-loop costs can cover backend choice. These model
extensions are under source investigation. The complete joint heuristic,
validated family defaults and fresh frozen validation remain unfinished.
