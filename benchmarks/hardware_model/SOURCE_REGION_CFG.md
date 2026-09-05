# Source regions, call streams and common allocation

`declared_call_stream.bind_call_stream` consumes exact actual step-call IDs
and explicit `<step ID>.linear<body index>` entries. It checks each source
cap through the existing workload evaluator, requires every entered nested
call, and preserves the different counter semantics of Newton, MR and
BiCGSTAB. It never replaces the supplied map with one common N/K count.
Admission here does not assert that the existing full-step lowerer supports
partial masks.

`source_region_cfg.RegionCFG` is the executable first integration stage of
the approved masked-stream architecture. It compiles complete selected
statements from an actual captured function into a structured cyclic CFG.
The selection boundary is explicit in every program. Actual MR statements
294–302 cover the guarded alpha division and guarded x/rhs update. They
include real scalar and aliased-memory joins. This stage is not a complete
masked solver, a SASS program, or a timing predictor.

The compiler builds both runtime branch arms. Source closure constants may
fold a branch; replay values cannot. Full fixed loops produce separate
copies. Counted loops retain one induction header/backedge with the actual
counted copies and a guarded tail. False remains an explicit retained-loop
compiler alternative, with the original False source flag preserved; this
does not claim the installed compiler chooses that form. Late full unroll
is not admitted by this constructor.

SSA construction inserts scalar and memory phis on predecessor edges.
Loop header phis name the entry definition and actual carried definition.
A missing incoming scalar definition is an error, not an invented zero.
Memory SSA represents complete addressable storage aliases; each store
updates only the demanded cell of its lane's incoming memory version.
`captured_arrays` obtains exact extents from actual call formal bindings or
one invariant allocator alias in that call context. Its receipt binds the
source graph, workload identity, function and call. Explicit scratch-array
bindings remain available for focused semantic fixtures and are labeled
by the missing capture receipt.

Allocation solves backward SSA liveness to a fixed point, with phi uses on
their respective incoming edges. It colors one static interference graph
in separate predicate and general-register banks. Pointer values occupy
two contiguous 32-bit words. Interference conservatively retains inputs
through an operation's output and parallel phi edge copies. This is a
physical-word allocation for the source SSA stage. Native expansion,
spill/reload forms and allocation quanta must be incorporated before its
resource count can become a complete solver prediction. Exhausting the
declared hardware-valid budget currently refuses this stage explicitly.
The pinned native_plan.py is untouched.

Lane replay uses that single allocation for every declared execution. Each
physical register word carries its current source value identity per lane;
every read checks that identity. Writes and parallel phi copies preserve
other lanes. The divergent form skips an untaken arm. The alternative
`predicated_acyclic_arms` issues the instructions of both acyclic arms while
their execution masks remain disjoint. Thus a predicate-false instruction
can issue with zero memory lanes. Arms containing a retained loop keep a
structured branch in this alternative; it does not invent a loop-counter
execution rule for a completely predicated cyclic region.

The actual Newton commit loop is an important positive control: both old
and candidate values are evaluated and every entered lane performs the
store of its selected value. A false commit predicate does not mask away
the load/add/store body. Actual Newton and BiCGSTAB counter assignments
likewise preserve entry-mask issue while changing only their source-selected
lane values. MR's guarded stores instead execute on their branch lanes.

The e24 author cohort contains five actual MR closures (full/count1/count2/
count4/False), each using an explicit different-per-call Newton map and
exact actual allocator aliases. Forty lane replays cover one lane, eight
contiguous lanes, alternating lanes and all lanes under both branch forms.
Separate actual Newton/BiCGSTAB scalar commits and the Newton aliased array
commit check source-specific counter/store behavior. Numerical replay
payloads validate these programs; they never enter source folding,
allocation, service calibration or candidate selection.

The complete captured-step composition, recurrent convergence-control
phis/call effects, native lowering, masked cache traffic and per-warp
scheduler cursors are still required by the approved architecture. This
stage makes no admission or family-default claim for those paths.
