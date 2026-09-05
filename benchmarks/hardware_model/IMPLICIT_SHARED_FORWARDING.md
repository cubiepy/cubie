# Shared read forwarding in implicit policy graphs

`implicit_shared_forwarding.py` defines the explicit compiler alternative
`within_region_store_to_load_forwarding`. It removes a shared load only
when the actual source graph proves that an earlier constant-address store
supplies the same typed value within one straight-line control region.
Every store, runtime branch, vote and source memory exit value remains.
The retained value enters a fresh typed allocation, including its longer
live range and any resulting spills. No register multiplier or fixed
forwarding benefit is used.

## Eligibility

The constructor must describe the actual private shared slice for each run,
including its FP32 byte stride. Accesses identify aligned four-byte FP32,
int32 or uint32 cells. Both forwarded endpoints must have source-constant
addresses; a replayed runtime index is insufficient. Every dynamic shared
write invalidates all available shared versions because its witnessed cell
does not prove disjointness from other cells.

The scan records every boundary and the current helper invocation, active
source branch IDs, recurrent or rolled loop execution indices, and issue
mask. It clears the version map at helper entry/return, runtime branch
decisions, or changes in those control identities. Fully expanded fixed
loops need no runtime merge and may share a region. An opaque call fails
the alternative. Full-warp issue is required; scalar allocation does not
prove register and spill merges across lane subsets.

A load with no eligible reaching store remains a load. The model retains
shared live-in reads, including a repeated read unless a suitable in-region
store supplies its value. It does not infer register inputs from a shared
boundary cell.

## Dataflow and identity

At each retained shared store, the lowerer records its actual typed data
operand. A proved read reuses that exact value, carries the source branch
dependencies and value producer, and incurs no shared-store-completion
dependency. Later consumers therefore wait for their data, not for an
unnecessary shared round trip. All source stores remain in their original
order and retain their original data values.

`make_policy_plan(..., shared_forwarding=True)` selects the alternative.
The plan records `compiler_materialization` with separate local and shared
choices. Its shared choice is either `retained_loads_stores` or
`within_region_store_to_load_forwarding`. Verification reconstructs the
selected choice before comparing the entire allocation and proof.

The `shared_forwarding` record contains exact reaching-store/read pairs,
their control-region signatures, every version-map boundary, the actual
shared stride and the complete source-order retained-store list. The
original ERK forwarding module and its stricter complete-step admission
rules are unchanged.

## Model boundary

This is a conditional finite compiler alternative, not a statement that
the backend always performs every eligible forwarding operation. Cross-call
retention, recurrent phi values, branch merge values, lane-specific merges,
intermediate-store elimination and final-store motion require different
proofs. They cannot be inferred from a single selected execution path.
The source branches and loop directives remain runtime operations wherever
the policy graph records them as such. Native ABI, scheduling and complete
kernel instruction coverage retain their existing model limitations.

## Source-only author evidence

The actual Lorenz/Kvaerno3/LU shared-accumulator construction forwards 30
loads under full unrolling, zero with count-1 stage loops, and 21 with count-1
accumulator loops. A two-body Newton scenario retains the same 30 eligible
forwards; recurrent boundaries remain explicit. Local placement has zero
shared forwards. Each case was allocated with both 32- and 64-word GPR
budgets and both shared compiler alternatives.

The full shared case extends 12 source-value live ranges. Its 32-word
allocation frame increases from 60 to 64 bytes; at 64 words, both frames
are eight bytes. Every retained typed operation has the same source identity,
input and output semantics, and memory effect as the retained-load plan.
Only the proved shared loads disappear. This is allocation evidence for an
explicit compiler alternative, not measured native register or timing data.
The author receipt requires separate independent review:
`verification/implicit_shared_forwarding_author_20260905/cohort_e1/receipt.json`
in the external raw-evidence directory.
