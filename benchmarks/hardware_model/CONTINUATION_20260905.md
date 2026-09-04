# Continuation: hardware policy model, 2026-09-05

The new worktree is
`C:/local_working_projects/cubie-worktrees/hardware-policy-continuation`, on
`codex/hardware-policy-continuation`. It starts from `origin/main` 1391bf35
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
