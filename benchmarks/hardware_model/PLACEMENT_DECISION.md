# Placement-specific lowering and physical memory demand

`placement_source.py` constructs a fresh actual Solver for every local/shared
combination of owner-qualified registry buffers. It captures generated source
without requesting a kernel, native overload, or device launch. Each target
pairs its owner and buffer name with the public placement setting. This retains
aliases, persistent storage, the full registry, and the actual kernel's padded
shared stride. Shared allocation uses the entire run stride; overlapping aliases
and nested owner windows are not summed twice.

`workload_identity.py` identifies the actual system by `fn_hash` and its typed
compile-settings digest. It also binds the complete tableau coefficient bytes
and every recursive step/solver factory's semantic configuration. Only unroll,
placement-location fields, and fields declared `eq=False` are excluded. Private
codegen directory names do not identify the workload. The actual precision must
be FP32. All other solver settings and typed constants retain their identity.
The shared candidate contract is `graph.candidate_construction`, containing
`workload_identity`, `shared_stride_bytes`, and `precision`. The full registry's
owner-qualified placement map is `graph.placement_identity`.

`placement_decision.py` accepts schema version 2 source-graph requests. It no
longer accepts external allocation plans. Every placement is lowered anew by
`implicit_native_lowering.make_plan`, including fresh base-address values,
LDS/STS or LDL/STL instructions, predicate spills, and register allocation.
Both promoted-local and addressable-local compiler alternatives are required;
they are uncertainty scenarios, not user-selectable settings. Shared placement
can raise or lower register demand. Source liveness is never multiplied by a
coefficient or subtracted from a reused local register count.

Every row retains its complete conditional NativePlan. Hardware residency uses
published register and shared allocation quanta, capacities and launch limits.
The cache scenario supplies a supported shared carveout. L1 capacity is the
unified data capacity minus that carveout, not an inferred hit rate. The source
request must contain the complete local/shared target product; full registry
placement, constructor, shared-padding and graph identities are checked. Every
arm must use the identical declared iteration/mask regime, branch choices and
source semantic contract; N=1 and N=2 are different comparison scenarios.

The physical demand pass replays actual lowered named accesses and allocator
spill accesses through two full resident waves. Its explicit hypotheses are:

- Aligned FP32/int32/uint32 local words coalesce into four 32-byte sectors per
  full warp. Subgroup entries use their actual source runtime issue mask; only
  sectors touched by entered lanes count. Spill/reload instructions inherit
  the issue mask of their allocation source position. Ambiguous fused masks
  reject admission. Modeled local frames are disjoint per resident warp slot; a
  `trajectory_unique` alternative gives subsequent waves fresh slots.
- Shared addresses use the captured padded run stride and 32 four-byte banks.
  Distinct words sharing a bank serialize; equal-address reads broadcast.
- L1 and L2 use cold, fully associative sector LRU and write allocation. L2
  available capacity per modeled SM is an explicit partition scenario bounded
  by published total L2, not a fit to observed hit rates. Other SM activity is
  represented by that available-capacity hypothesis.
- `local_store_policy` must explicitly choose `l1_write_through_l2_write_back`
  or `l1_write_back_l2_write_back`. The first sends every local-store sector to
  L2 even on L1 hits. This distinction matters: the retained local33 profiles
  show L2 write requests growing with repeated local stores despite warm L1
  loads. Writeback-only traffic is not asserted as hardware behavior.

The output reports register/frame demand, resident warps, shared bytes, bank
wavefronts, cache requests/hits/misses, dirty evictions and retained dirty
sectors. Cold-start downstream requests and retained dirty data are separate;
there is no artificial end-of-step flush. Zero-capacity cache scenarios forward
accesses without trying to evict from an empty cache. Counts normalized per
completed warp are emitted as exact rational physical coefficients so changing
residency does not compare different amounts of work.

These are conditional physical demand predictions, not additive runtime terms.
A timing comparison still requires independently qualified memory services,
instruction delivery and dependency-aware scheduling. No arbitrary cache miss
penalty, native register observation, solve-time regression, or winner table
enters this component. In particular it does not infer that a lower cold miss
count compensates for more instructions or lower occupancy. The complete plans
and normalized demands support the joint selector's separate qualified service
scenarios.

Example source construction request:

```json
{"system":"lorenz","algo":"kvaerno3","linear_solver":"lu",
 "newton_bodies":1,"krylov_bodies":1,
 "branch_choices":{"generic_dirk.py:744":false},
 "targets":[{"owner":"step","name":"stage_base",
             "setting":"stage_base_location"}]}
```

Construction additionally accepts `solver_settings` for exact public settings.
Use the intended worktree's `src` on `PYTHONPATH` to avoid an editable install
silently selecting a different source checkout. The generated `graphs.json`
contains the placement-to-source bindings used as `source_graphs` in the v2
request. That request also supplies `named_buffers` (`owner:name`), hardware,
architecture bank-budget scenario, compiler alternative, block size, static
shared bytes, both materialization scenarios, and the explicit cache scenario.

```text
python -m benchmarks.hardware_model.placement_source --request construction.json --output fresh-source-dir
python -m benchmarks.hardware_model.placement_decision --request request.json --out fresh-result.json
```

The constructor refuses an existing output directory; the decision CLI creates
its output exclusively. CPU construction and conservation checks are author
validation only. Independent verifier approval and native holdouts remain
separate gates before interpreting these hypotheses as a production heuristic.


Scalar-tag allocation proves one coherent full-warp trace. A subgroup issue mask
can be counted exactly by the memory model, but a spill performed inside that
subgroup does not establish conservation for lanes that rejoin afterward.
`allocation_regime_support` therefore records the complete observed source-mask
set and a positive `supported_coherent_full_warp` status only when every source
operation issues on all 32 lanes. Any subgroup produces
`unsupported_subgroup_allocation` and `finite_ranking_supported=false`.
Subgroup physical counts remain inspectable conditional demands; a finite joint
ranking must reject them until per-lane allocation and merge are proved.
