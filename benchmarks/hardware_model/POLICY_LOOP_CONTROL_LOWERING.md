# Fixed-loop execution control

`policy_loop_control_lowering.py` supplies a conditional native control
form for the actual fixed source loops retained by the policy frontend.
It runs through source-boundary callbacks in the research typed lowerer,
before register allocation. No allocated event trace is patched.

For a positive compile-time main-trip count with multiple runtime chunks,
the form is a bottom-tested loop. Its source-derived base starts with an
immediate MOV. Each completed chunk advances that same base using
IADD3, compares it with the main-loop terminal bound using signed ISETP,
and issues a predicate-dependent BRA. The updated SSA base is shared with
the body-index mapper. Counted-lane offsets retain their ordinary typed
IADD3 forms. Thus control and address calculation use one carried base;
the model does not charge a second body recurrence.

The complete captured source range determines start, stride, chunk width,
main-loop end and branch results. Every complete dynamic chunk receives
one predicate and branch, including the terminal false decision. A
positive source stride uses a less-than predicate; a negative source
stride uses greater-than. The terminal arithmetic and immediate chunk
increment must fit signed int32. A missing chunk, incomplete counted
lane set or unsupported arithmetic domain is rejected explicitly.

Full expansions, constant tails, zero trips and fixed single-chunk regions
need no additional dynamic control. Actual source Newton/Krylov exit
branches remain unchanged. Fixed loops nested inside their execution
regions still receive their own loop control. Loop operations retain the
loop constructor's participating mask and outer source scope rather than
a body-only selected subbranch. The normal source-order allocator sees
every new integer and predicate value, use, dependency and spill.

The verifier evaluates the emitted MOV/IADD3 operands, then the actual
ISETP input values and predicate-consuming branches. It checks the exact
Cartesian set of loop IDs, completed chunks and control opcodes and
preserves the original source BranchDecision count. Source-operation
mapping remains separate from synthetic native loop-control attachment.

The instruction forecast excludes matching typed initialization,
increments, comparisons and backedges from its supplementary forms.
`False` with a full-replication footprint is explicitly a static
replication sensitivity which retains typed forms. It does not prove that
a native optimizer retains their instructions after late full unrolling.
A physically distinct full-unroll execution/allocation candidate must use
its own fresh full-policy source graph and typed lowering.

This form has no timing fit or cache penalty. Its counts follow the source
range and directive; scheduler services remain separately qualified
hardware inputs. Native lowering, branch elimination, instruction
packing and temporal instruction residency remain compiler/hardware
questions rather than conclusions of the source graph.

## Captured empty-body specialization

A fixed loop with no extracted source nodes is eliminated only when its
actual loop AST proves an empty body under exact captured Boolean closure
values. The proof records the source/function identities, guard expressions,
selected branches and captured values. Runtime predicates, stores, calls,
unknown expressions and loop-else bodies do not qualify. Retained loops keep
complete chunk-control coverage. The typed model and instruction forecast
both retain the elimination receipt and charge no administration for a
proven empty loop.
