# Exact integer-clock scheduling

`nominal_integer_execution.py` accelerates the fixed-service branch of
`nominal_execution.py`. It provides the same `schedule_events` and
`schedule_plan` interfaces and returns the same exact rational result
schema, including every optional trace row. The ordinary engine remains
the reference implementation.

This changes arithmetic representation only. It introduces no hardware
constant, fitted parameter, approximation, rounding, or service choice.
It does not change a candidate's graph, allocation, placement, source
regime, common work, or scenario identity.

## Exact time representation

The existing service parser resolves physical catalog values and qualified
scenario overrides. The integer engine gathers every resolved result,
source-consumption, store-visibility, store-completion, branch-blocking,
and initiation interval, every contracted pair/completion interval, and
every resource reservation in every partition and route. It computes the
least common multiple of their exact rational denominators. One cycle
contains that many integer ticks. Integer-valued intervals are included;
missing service fields retain the ordinary missing-service response.

Each interval multiplied by this unit must have denominator one. Python
integers have arbitrary precision, so a large least common multiple does
not cause fixed-width overflow. No denominator cap or nearest-tick
conversion exists. Resource and dependency clocks start at zero and use
integer addition and maximum. These operations preserve the exact
rational ordering under multiplication by the common positive unit.

The scheduler minimizes `(issue_tick, warp_id, route_index)`, preserving
the ordinary warp tie and dedicated-FP32 route tie. The two FP32 routes,
coupled integer route, partition dispatch, and other resource reservations
are produced by the ordinary `reservations` function before conversion.
Byte-overlap RAW, WAR, and WAW memory dependencies retain local warp and
shared block ownership. Register result readiness and source-operand
consumption remain separate. A contracted store/load pair is charged once
and retains its declared completion.

Every reported time, resource demand, and trace reservation converts back
with `Fraction(ticks, ticks_per_cycle)`. Normalization to common work then
uses the ordinary exact rational function. IDs, counts, routes, and byte
addresses are never scaled. `prepare_ticks` exposes the computed unit for
audit receipts without adding fields to the ordinary result schema.

## Supported scope and admission

The accelerated path supports fixed local load paths, fixed shared memory
services and bank wavefront counts, and qualified immutable broadcast
loads. It accepts omitted instruction-fetch specifications or a qualified
explicit `mode="disabled"` specification. It rejects every supplied
stateful data-cache specification and every enabled instruction hierarchy.
Those scenarios require the ordinary scheduler, which models pending
fills, cache state, repeated waves, and fetch eligibility.

Executable events, qualified overrides, motif contracts, full-warp masks,
CUDA block completeness, nominal catalog identity, and common-work checks
use the ordinary admission helpers. Plan kind, prohibited native/timing
labels, plan hash, event hash, catalog hash, scenario hash, placement
identity, and iteration regime are preserved. As in the ordinary engine,
the caller's scenario recipe is responsible for reconstructing and binding
the graph to the policy plan. Scheduling historical saved typed events
does not recertify those events against subsequently changed lowering
sources.

The preparation/result wrapper follows the ordinary fixed-service branch;
its full-result equality is a maintenance requirement when admission or
the reference result schema changes. No runtime replacement, monkeypatch,
or alternate native lowering is involved. This module is separate from
the active joint candidate harness until independent verification passes.

## Validation

Author scripts and exact request/result artifacts live outside the
repository under
`hardware_unroll_placement/verification/nominal_integer_author_20260905`.
Checks compare complete results and full issue traces, not just totals.
They include dependent arithmetic, coupled route ties, register WAW,
constant-address WAR, partially overlapping byte memory, shared block
ownership, exact pair contracts, and coprime rational intervals. Saved
pilot algorithm families exercise actual typed events at multiple
resident-warp populations. Historical graphs and plans retain their
original hashes; these equivalence fixtures are not new solver forecasts.

CPU runtime measurements use identical input plans with trace recording
disabled and alternate engine order. They measure implementation cost,
remain separate from GPU service inputs, and do not train the predictor.
Independent verification must compare the original scheduling contract
and complete traces before integration.

Independent `verification/nominal_integer_independent_e1/receipt.json`
passes 25 checks against source hashes `634b4716` (integer engine) and
`d15cd5ac` (reference engine). These include all ten saved algorithm/inner
plans, 85,728 trace rows at 48 resident warps in FIRK/BiCGSTAB, the exact
16,968-row profiling fixture, hand-derived hazards and pair completion,
and explicit unsupported-state refusal. A coprime-denominator case uses
2,588,832,555,403,324,885,852 ticks per cycle; another exceeds `2**100`
cycles. Both agree exactly without rounding or fixed-width overflow.
