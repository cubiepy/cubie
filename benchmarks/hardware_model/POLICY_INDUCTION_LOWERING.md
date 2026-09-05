# Runtime induction native forms

`policy_induction_lowering.py` gives internal rolled/count-N loop indices
real typed producers. The source graph retains a runtime SSA identity for
each executed visit. These identities are not external kernel arguments
and must not all occupy registers at step entry.

The lowerer derives the range start and stride from the complete captured
source loop structure, checks every range position, and binds the counted
lane and chunk to the production directive. A chunk base starts with one
immediate `MOV`. Subsequent required chunks use `IADD3` with the previous
runtime base and the source-position difference times the range stride.
Counted lanes use `IADD3(base, lane * stride, 0)`. Constant tails and fully
unrolled indices retain their existing constant treatment.

The initial literal comes from the source range start. Execution witnesses
are used only to check the resulting integer values. They never replace
the operands of a runtime address instruction. If earlier visits do not
use the index, the initial base still uses the source range start and a
separate arithmetic jump covers the actual skipped source positions.
This is a conditional affine-induction form for the selected execution
regime; it does not establish that an optimizing compiler selects the same
recurrence or instruction schedule.

Pure induction arithmetic is emitted at its first required source use.
Each instruction retains the loop constructor's source/call ownership,
the consuming operation's runtime mask, and its exact source control
predecessors. Previous-chunk operands create explicit loop-carried data
dependencies. Counted-lane intermediates and chunk bases therefore receive
normal register liveness and allocation, including spills, instead of
being pinned in the kernel entry inventory. Reverse first-use order is
rejected because it needs an additional placement rule.

`verify_inductions` evaluates the emitted immediate MOV and actual IADD3
operands under signed 32-bit wrap, then checks every mapped source
induction witness. The normal allocation verifier conserves all these
typed operations and their operands. The graph's source replay remains
separate from this native-form check.

The footprint uses distinct source identities for initialization, carried
increments, and lane offsets. Only matching loop-ID/category pairs remove
supplementary initialization or induction charges. Header predicates and
backedges remain the separately declared control alternative; this mixin
does not claim they are represented in the dynamic execution schedule.
The typed form corresponds to rolled/count-N execution. A backend-choice
full-replication footprint remains a static compiler sensitivity; a late
full-unroll execution/allocation alternative requires its own source graph.

Full-unroll arithmetic and memory operations remain byte-for-byte equal to
the reused lowerer's output when the actual source graph has no dynamic
forms. The native compiler, hardware counters, measured iterations, and
solver timing labels do not participate in construction or verification.
