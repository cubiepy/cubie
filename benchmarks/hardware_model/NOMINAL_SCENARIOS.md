# Finite attempted-step service scenarios

`nominal_scenarios.py` builds explicit physical sensitivity scenarios for
the frozen epoch-2 scheduler and catalog. Its objective is one attempted
algorithm step with caller-visible state. The omitted outer caller, native
ABI registers, instruction delivery, and kernel writeback drain remain
outside the estimate. These schedules are conditional estimates, not
certified bounds on complete kernel execution.

`build_scenario(graph, plan, catalog, hardware, geometry, evidence)` accepts
the complete actual policy plan. It reconstructs and verifies the plan
against that graph before using its allocation, and records both hashes.
Old plans require their preserved lowerer epoch or a fresh reconstruction.
Registers come from allocation peak R; the
selector applies published register allocation quanta and occupancy limits.
The geometry record supplies a qualified block size, static shared bytes,
and carveout. Its block size is the actual source-selected size or an
explicit geometry hypothesis; this builder does not reproduce the caller's
block-halving logic. Dynamic shared memory is the larger of four bytes and
captured per-run stride times block threads, following
`BatchSolverKernel.py:837`. No compiled register labels enter prediction.

Shared issue multiplicity follows the actual lane address pattern:
`lane * shared_stride_bytes + offset`. Distinct four-byte words in each of
32 banks serialize; equal-address reads broadcast. Local scalar same-slot
accesses use the verified four-sector warp pattern. Their resident footprint
is `frame_bytes * 32 * resident_warps`. An optional qualified capacity record
reports unified L1/shared capacity minus carveout and device-wide local
footprint versus L2. Capacity fit alone never supplies a miss fraction.

The branch hypothesis transfers the median of six verified recurring-loop
branch-resolving counter motifs to a typed BRA. This extrapolates beyond
the catalog's measured recurring-loop scope and is explicitly a scenario,
not an intrinsic branch latency. ACTIVEMASK transfers the independently
verified single-active-warp ballot-plus-compare motif median, including
comparison and loop administration. No arithmetic latency is subtracted.
The published vote issue rate remains a distinct shared resource demand.

Standalone STS and STL visibility have two sensitivity envelopes. The
`pair_readback` hypothesis uses the complete same-cell store/load interval,
7465/257 or 9012/257 cycles respectively. `input_consumption` uses measured
source WAR, 12 or 14 cycles, with the catalog's explicit address-space
transfer for STL. Both assign caller-region completion at that hypothesized
visibility; neither establishes kernel drain or a physical runtime bound.
An independently scheduled consumer still incurs its own load service: the
whole pair is a deliberately conservative visibility proxy, not an exact
pair decomposition. When explicit exact pair contracts are supplied, the
scheduler instead charges the pair once through its motif interface.

Local load scenarios select one complete path: 32 cycles L1, 273 or 284.8
cycles L2, or 571 cycles DRAM. Both conflicting published L2 figures remain
separate. L1 and L2 costs are never added. The shared LSU calendar represents
front-end pressure; downstream L2/DRAM bandwidth is not identified by these
latency scenarios.

Immutable LDC needs an explicit broadcast-cache-hit assumption. Constant
footprint reports both identical-payload dedup and duplicated-table per
captured-owner hypotheses. Payload identity is not physical allocation
identity: the installed backend caches globals by object identity. Counting
every owner separately exposes duplication without pretending that repeated
source invocations necessarily own different Python objects.

`forecast(..., work=...)` schedules common complete-wave attempted-step work
with the source-derived occupancy. The engine rejects incomplete or fewer
than two waves. Comparisons must use the same work count and iteration
regime; scenario IDs do not make different workloads comparable.

The optional `evidence.data_cache` enables capacity-derived per-event
paths through `bind_data_cache`. It supplies published unified and L2
capacities, one explicit L2 latency alternative, initial cache state,
frame backing, write policy and the unlimited-fill capacity hypothesis.
The builder derives available L1, nominal per-SM L2 share and frame bytes;
the scheduler then determines each load path at committed issue time.
See `NOMINAL_DATA_CACHE.md` for pending fills and repeated-wave semantics.

The author receipt lives outside the repository at
`verification/nominal_scenarios_author_20260905/receipt.json`. Independent
review is required before treating this new builder as a verified component.
Fresh plan-identity and cache integration author evidence is at
`verification/nominal_data_cache_author_20260905/actual_e1/receipt.json`.
The earlier 32-scenario receipt and matching source bytes are preserved
in the external `nominal_before_cache_snapshot_20260905` directory.
