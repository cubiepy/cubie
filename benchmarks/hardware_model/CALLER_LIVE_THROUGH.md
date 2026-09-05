# Caller live-through state

The actual ODE-loop step call is analyzed after cached integrator wiring.
The extractor binds the exact step graph, semantic workload, full recursive
caller configuration (including unroll/placement), actual function sources
and helper formal demand. It does not execute device functions or inspect
native resource counts.

Scalar liveness uses a backward use/kill fixed point over retained runtime
branches and the outer loop. Only proved source/closure constants are
removed. Actual helper formal-demand summaries preserve native intrinsics
as operations; their Python placeholder bodies are not interpreted as
empty implementations. Source casts determine word widths. The actual
FP32-state caller retains binary64 times; these are represented as opaque
low/high int32 words, with two distinct identities and no new FP64 step
arithmetic. The existing caller:t_prec source value is reused at exit.

Cell liveness separately derives helper read-before-write and must-write
sets. Constant-trip loops are expanded for memory-set analysis regardless
of compiler policy, while surviving caller induction taint separately
prevents scalar promotion of the entire aliased storage. Actual runtime
branches remain in the memory CFG. The existing step graph already retains
all accessed boundary cells as exit observables; those cells join by exact
storage/start/end/dtype identity. The additional PI-controller cells are
its previous error norm and dt[0]. accept_out[0] is overwritten before its
first read and adds no old-value demand.

Two cell forms are declared: promoted constant cells, admitted only when
all relevant alias addresses are constant, and addressable whole storage.
The latter rebuilds storage extents, frame offsets and all affected step
loads/stores before allocation. It can change algorithm-persistent traffic
when controller history occupies the same root storage. Existing shared
memory remains addressable. The promoted-copy mixin preserves loaded typed
versions through subsequent promoted array copies without removing any
addressable access.

Two global-address forms are declared. Retained descriptor keeps a 64-bit
base and one 64-bit byte stride per source-demanded axis. The actual 2D
output view therefore has six words. Parameter rematerialization reloads
those descriptor fields after the attempted step. It excludes that reload
cost from this step-only interval; it does not claim that the complete
solver obtains those fields for free. Local/shared address forms reuse the
existing supplied base and literal view displacements.

All retained scalar/cell words become real entry live-ins and exit
observables before fresh typed bank allocation. They can spill under the
same physical R/P budgets as the algorithm. An opaque binary64 pair can
be reassembled for caller arithmetic outside this step boundary; this
adapter does not model those caller instructions or their pair-placement
constraints. This remains a conditional step-resource model, not an exact
whole-kernel register prediction. No fitted overhead, measured iterations,
solver timings or measured native register counts enter the model.

Author evidence: 17 actual source constructions and 132 admitted fresh
allocations at 64/255 GPR, covering all ten family/inner combinations,
counted stage/element policies and fixed/I/PI/PID/Gustafsson controllers.
Six hand-checkable memory CFGs exercise read/write ordering, branch joins,
loop recurrence and constant-trip writes. Independent review is required.


Caller memory demand first removes source-proven dead scalar assignments
whose expressions contain no calls and perform no writes. Scalar
use/kill reaches a fixed point before cell liveness is solved afresh;
recorded proofs retain the source line, cell reads, defined names and
live-out names. Calls and memory writes remain effects even when their
scalar result is unused.

An additional demanded caller cell can instead be a rematerialized
source constant. This form requires an exact typed constant store on the
unique straight-line predecessor segment of the step call. Control
joins, calls and unproved writes stop that proof. The selected step
graph must have no overlapping exact-cell access and no runtime address
that may alias the cell. Nonoverlapping accesses to another element of
the same allocation remain valid: Rosenbrock's Krylov counter may change
while its adjacent Newton counter retains the source-written zero.
These constant cells are recorded separately and add no live-through
entry words. Recreating them occurs after the step, outside the modeled
attempted-step interval. This proof does not specialize runtime data or
use an execution witness as a constant.

The scalar CFG currently admits a retained `for` loop only when its
induction target is dead on the zero-trip exit edge. A target live on
that edge requires edge-specific definitions and is explicitly rejected.
The admitted source cohort satisfies this condition.


The joint adapter composes CallerLiveThrough from caller_lowering_mixin
with the normal or shared-forwarding policy lowerer. The standalone
CallerTypedLowering remains a thin composition for source audits. The
retention algorithm is the reviewed caller pass; mixin extraction lets
policy verification reconstruct the same complete allocation without an
import cycle. Policy wrappers bind the full caller context and named
cell/pointer forms. Inventories are captured before Solver.close.
