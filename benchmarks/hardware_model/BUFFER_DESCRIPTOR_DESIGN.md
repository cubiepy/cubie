# Buffer identity, effects and scalar-replacement opportunities

The component is `buffer_descriptors.py`: an interprocedural source
descriptor built from the actual uncompiled helper graph and resolved
buffer registry. Its output identifies which element accesses can be
represented as scalar values, what prevents that proof, and which
conditions depend on unroll lowering. It does not predict native
register counts, spills or timing.

## Why this component is needed

`workload.py` connects captured helper instances and call arguments;
`expansion.py` describes requested index visibility; `static_descriptors.py`
provides local source DAGs. None proves that differently named arrays in
caller and callee refer to the same allocation, resolves overlapping
views throughout the call graph, or versions buffer elements across calls.
Those gaps prevent a justified scalarization or address-taking analysis.

Declared placement alone is insufficient. The
[allocator](/C:/local_working_projects/cubie-worktrees/hardware-epoch-ff3a567f/src/cubie/buffer_registry.py:699)
uses resolved shared/persistent slices or a local allocation; an alias
can override an entry's declared location. The existing
[placement observer](/C:/local_working_projects/cubie-worktrees/hardware-unroll-placement/benchmarks/hardware_model/placement_probe.py:265)
shows how to inspect the layout and allocator closure without compiling.

## Inputs and output contract

Inputs are a constructed solver, explicit group directives, generated
Python source, captured immutable configuration and published hardware
limits. Record source/closure/extractor/compiler identities and verify
zero native overloads before and after. Main and smoothing solver
instances remain separate; bindings must honor the complete signature,
including keywords, without silently truncating arguments.

The JSON contains allocation identities, byte views, effect summaries,
per-element source versions/dependencies, and proof conditions:

- Resolve each allocation from owner, registry name, memory space, dtype,
  shape, shared/persistent offset and allocator closure. Physical overlap
  joins aliases; a merely declared alias is not assumed to overlap.
- Propagate allocation/view identities through assignments, slices and
  known helper calls. Track reads, writes, returned aliases, possible
  address escape, and live-in/live-out elements at each call boundary.
  Reinterpreted views use bytes; unresolved strides or overlap stay unknown.
- Literal accesses receive element identities. Affine fixed-loop accesses
  retain index expressions and finite ranges. Full/count 1/2/4/False remain
  distinct conditions on index visibility; canonical requested copies are
  never labeled guaranteed native copies. Newton/Krylov bodies remain
  recurrent summaries with symbolic invocation counts.
- Build element SSA for straight-line regions and known-call effects.
  A store creates a version; a read depends on the reaching version(s).
  Merge branches conservatively and represent loop-carried state explicitly.
  Dynamic stores can affect every overlapping element; dynamic reads retain
  all possible producers. Opaque calls receiving a buffer invalidate its
  possible writes and record escape uncertainty, including keyword/receiver
  arguments. A known inlined helper call is not automatically an escape.
- Report separate proof states: literal element opportunity, opportunity
  conditional on index specialization/callee treatment, or unresolved.
  Passing these proofs means scalar replacement is semantically possible
  under their conditions; it does not mean the compiler performs it.

Summaries attach to call instances and storage lifetimes. Repeated dynamic
calls reuse a summary; their frequency does not replicate source bodies or
allocate a new buffer simultaneously. Distinct live allocations and
overlapping byte views cannot be counted twice.

## Concrete family and solver differences

The [source receipt](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/verification/buffer_source_design_20260905.json)
recounts actual generated helpers and records their current file and
function-AST hashes. These are source accesses, not native instructions:

| Captured helper | Output width | Constant factor elements | Factor reads / writes |
|---|---:|---:|---:|
| Fabbri Radau5 main LU | 105 | 1,227 | 8,104 / 1,227 |
| Fabbri Radau5 smoothing LU | 35 | 137 | 344 / 137 |
| Fabbri Kvaerno3 main LU | 35 | 137 | 344 / 137 |
| Lorenz prefactored Radau5 | 9 | factor unused | 0 / 0 |

All observed factor/rhs/x subscripts in these helpers are literal. The
prefactored Lorenz helper instead makes 36 reads from 20 constant
`cached_aux` elements at
[generated line 111](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/expansion_probe_codegen/Lorenz/Lorenz_c60e4ab194.py:111).
Its factors belong to a longer-lived prepared cache, so replacing it by
the non-prefactored factor-buffer model would change the workload.

FIRK's [main width is stages times n](/C:/local_working_projects/cubie-worktrees/hardware-epoch-ff3a567f/src/cubie/integrators/algorithms/generic_firk.py:169);
smoothing uses n. DIRK reuses an n-wide solve across implicit stages.
ERK stage storage depends on the actual tableau and accumulation scheme.
These call graphs, rather than family labels alone, determine storage
lifetimes and address-visible interfaces.

BiCGSTAB exposes a different mechanism: [five work buffers](/C:/local_working_projects/cubie-worktrees/hardware-epoch-ff3a567f/src/cubie/integrators/matrix_free_solvers/bicgstab_solver.py:172)
cross preconditioner/operator calls, while vector loops use
`unroll_solver_element` and the recurrence uses `unroll_krylov_exits`
at lines 280/308. Literal vector indices require analysis across those
calls and loops; they cannot inherit the generated LU proof.

## Hardware use without a fitted register multiplier

Resolved shared extents and per-run layout give allocation bytes at a
requested block geometry, including alias reuse. Combine them with the
recorded hardware reservation, allocation granularity and supported
carveouts to bound feasible residency. Keep actual selected carveout
unknown until established by a separate compiler/driver model.

For a stated source schedule, live scalar/element versions have exact
payload widths. Their sum in 32-bit words describes a hypothetical mapping
that retains each version once. Comparing that mapping with hardware
register budgets can reject that particular retention hypothesis. It is
not a lower bound or estimate for native registers: rematerialization,
folding, scheduling, packing and extra address temporaries remain unknown.
No scalar-liveness multiplier is introduced.

Source accesses also provide logical read/write payloads and byte ranges.
They do not establish native local/spill traffic, cache sectors or hit
levels. Preserve scalarized and addressable alternatives until compiler
evidence distinguishes them; no historical winner or timing score selects
an alternative. The counted-unroll aliases in
[COUNTED_UNROLL_EVIDENCE.md](COUNTED_UNROLL_EVIDENCE.md) make this distinction
necessary even when the requested count is explicit.

## Implemented observer and validation boundary

`describe_buffers(solver, unroll=None)` returns the registry tree,
allocation instances, call bindings, byte-view memory versions, source
retention hypothesis, expansion conditions and residual unknowns.
Known helper signatures are bound completely before interpretation.
Allocator identities come from actual closures and parent views;
compatible registry labels are retained as candidates when ambiguous.
Root external arrays share an unknown alias set. Their dtype/extent is
not inferred from the solver's floating-point precision.

Non-full and iterative inventories retain explicit recurrence
boundaries. Counted main/tail inventories share a conservative boundary;
their exact parts remain in `source_region`, and the boundary is not an
assertion that the backend executes one combined loop. Branch and
short-circuit paths are joined. Possible alias stores and opaque writes
use weak updates; returned and skipped paths retain their frontiers.
Calls expose source memory versions live across their boundaries.
Ordering dependencies and mutually exclusive paths can inflate the
retention hypothesis, so it is not a minimum physical storage demand.

The command requires a fresh `--cache-root`, installs it through the
public cache API, and restores the prior setting after construction.
The callable API expects an already constructed solver and does not
change its cache configuration. No production source is modified.

The nine-worker CPU matrix is saved in
`C:/local_working_projects/cubie-notes/hardware_unroll_placement/`
`buffer_descriptor_validation_v4`. All nine report zero overloads and
their own codegen directories. Five fixtures cover chain32 DIRK and ERK,
Fabbri FIRK LU, Lorenz prefactored FIRK and iterative FIRK; four additional
Lorenz fixtures request solver-element count 1/2/4 and False. The
`positive_invariants.json` receipt verifies source file hashes, forward
memory dependencies, recurrence metadata, and actual LU factor effects.
Fabbri main/smoothing factors bind to 4,908/548-byte local views, with
1,227/137 stores and 8,104/344 reads. The main output binds to 420 bytes;
the external smoothing output retains unknown physical extent.

Contiguous captured rows retain their rank and byte stride. Integer tuple
indices compose those views; mixed tuple slices retain unknown extents
because a column can be strided. Captured row selection is a view rather
than a scalar load. A separate chain32 DIRK shared-placement fixture
resolves `stage_base` to bytes `[0,128)` inside the accumulator's
`[0,384)` window, using one shared allocation identity.

The component does not resolve hidden mutable state outside captured
closures and passed aliases, arbitrary Python object protocols, or
native lifetime reuse of repeated allocation sites. Those contracts
must be supplied or remain unknown when extending beyond these actual
helper graphs. Independent semantic review is required before model use.
