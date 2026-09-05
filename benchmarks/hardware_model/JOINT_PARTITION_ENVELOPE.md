# Source-bound partition selection

The joint evaluator's default request uses the complete supported
hardware shared-partition list. The requested carveout byte preference
is retained separately in `partition_selection.requested_carveout_bytes`.
The default request holds this preference fixed at the largest supported
value; it does not assert that this is the achieved partition.

This separation follows the exact-handle diagnosis in
`verification/carveout_handle_native_independent_e1`: the installed
compatibility setter can mutate a different function handle from the
function actually launched. Measured partition sizes are not used as
driver rules, fitted capacities, or inputs to policy selection.

Each allocated policy/placement/compiler/caller form enters a retained
plan inventory before physical geometry is enumerated. The adapter reads
the exact hash-bound graph and plan artifacts. It takes register demand
from the source allocation's `peak_resident.R`, shared stride from the
actual constructor's `candidate_construction.shared_stride_bytes`, and
reserves `max(4, block_threads * shared_stride_bytes)` dynamic bytes as
the launch path does. The existing hardware residency calculation then
checks every supported physical partition using its allocation quanta
and register/shared/block limits. Native resource labels are not inputs.

The graph/plan pair is the pair produced by the trusted source constructor
and fresh typed allocation, already admitted by the policy/scenario
pipeline. The adapter validates artifact byte hashes and recomputes the
resource projection; it does not claim an independent source interpreter.
The retained inventory and per-partition residency records make every
legality calculation reproducible in a separate review.

Actions contain policy, declared placement, block size and requested
preference. Physical partitions are excluded from action identity. For
each common compiler/service scenario, the adapter requires a finite
cost for every legal partition of every otherwise feasible action. A
shared-heavy action retains its larger legal partitions even when other
actions can use smaller capacities. Empty or incomplete rows are
reported explicitly; they are not filled from a different plan or
silently interpreted as an infinite runtime. Actions with no legal
geometry in any admitted compiler form are retained in the infeasible
inventory, with their physical proofs.

The independently reviewed `rank_partition_envelopes` arithmetic selects
the exact minimax action over independent per-action legal partitions.
The independent envelope can be conservative because coupling between
the driver's choices for different compiled functions is unknown. No
unproved common driver rule narrows it. Cross-regime family comparisons
use the same action identity and nested partition-cost matrices.

Schema-2 results retain original physical candidates, their cost links,
and their fixed-geometry conditional ranking under `physical_geometry`.
The main `candidates`, `costs`, `legal_partitions` and `ranking` describe
requested actions and achieved-capacity envelopes. Every physical cost
link has an explicit action/partition map, and all costs in a comparison
retain the same synchronized attempted-work count. Native holdout tools
must consume the requested action geometry under this schema instead of
treating a physical partition diagnostic as a launch preference.

The prediction interval remains an attempted step with source-derived
caller live-through resources. Outer-loop work and post-step descriptor
rematerialization are separately qualified. The Fabbri constant-division
form is the source-range-optimal IMAD.HI/SHF alternative; the installed
pipeline has not been shown to discover that range form. Shared
forwarding, memory materialization and instruction-delivery forms remain
explicit compiler/hardware alternatives. Finite results therefore do not
claim an unqualified prediction of installed native code.
