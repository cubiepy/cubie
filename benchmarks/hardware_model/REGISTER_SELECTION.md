# Conditional register selection for local arrays

The typed source model previously gave every surviving dynamic local
address an addressable frame. Saved validation SASS contains another
form: a local array's values occupy scalar registers, with index
comparisons and conditional register moves selecting reads and writes.
The concrete Kvaerno3 stage-count-one example is documented in
`verification/native_policy_code_diagnosis_e1/REPORT.md`: its nine-element
accumulator uses scalar register selection rather than LDL/STL. Those
native resource counts and timings are validation evidence, not inputs
to this component or a size threshold.

`register_selection.py` implements an explicit **ISETP/SEL** alternative.
This differs from destructive predicated MOV: SEL consumes predicate,
true value, and old false value, and can write a fresh SSA register. The
ordinary allocator therefore models all operands and lifetimes without
pretending that it enforces native predicated-MOV destination ties.
IMAD computes relative byte addresses; comparisons select scalar homes;
MOV binds a read result to its source value identity. Every operation
uses the existing qualified IMAD, ISETP, SEL, and MOV catalog services.
There are no new numerical service constants.

The rule `all_complete_source_proved_dynamic_local_extents` is one common
compiler scenario across every family, placement, and unroll candidate.
It selects every eligible source extent, rather than choosing whichever
native form predicts the lowest cost for each candidate. An explicit
supplied storage list must equal that complete derived set. Addressable
storage remains a separate common compiler scenario. Whole-extent and
source-domain selection are also separate common scenarios, not actions
that individual candidates optimize independently.

Eligibility comes from the complete source function and call descriptors,
with numerical replay inputs treated as unknown. The separate
`register_selection_source.py` reads and hashes every admitted function.
It visits both runtime branch arms, full source loop ranges, and exact
array slices passed to source-bound helpers. Its compiler rule considers
every local allocation, including extents without represented dynamic
events in a particular execution regime.

Admission requires:

- Every alias must agree on one 32-bit scalar dtype and exactly cover its
  whole local storage extent. Shared placement keeps its address space.
- Every possible source path must initialize all homes before a dynamic
  access, including writes that require the old false-input SSA value.
  Definite initialization is intersected across runtime branches;
  zero-trip loops contribute no initialization. Runtime indexing follows
  source unroll directives and propagates through arithmetic and slices.
  Unsupported early-return, while, alias escape, or index forms receive
  explicit rejection reasons. No fabricated old value is introduced.
- Dynamic address expressions must derive from constants, complete source
  induction ranges, exact casts/add/subtract/multiply, or independently
  admitted floor division. Their finite Cartesian domains preserve shared
  occurrences of the same SSA induction identity within an expression.
  Distinct versions have separate conservative range variables; different
  loop visits are never equated by their shared lexical loop ID.
  Intermediate index and byte
  arithmetic must fit int32 and all possible offsets must stay in the
  complete aligned home set.

The index domain is explicitly evaluated **before guard filtering**. It
is a conservative typed-event domain when guards correlate indices. The model
does not label it as the exact set of addresses executed by an arbitrary
mask regime, and it rejects an extent if this domain leaves its bounds.
Original guard, loop, and runtime-region operations remain in the typed
stream. Replay witnesses are never used as address bounds or constants.

In `whole_extent` mode each read selects among all homes, starting with
the first home and applying equality/SEL stages for the others. Each
write applies equality/SEL to all homes, preserving each old value when
its predicate is false. In `source_domain` mode only the source-domain
homes participate. A singleton read is a source-bound MOV; a singleton
write remains a conditional update in the current compiler form.

The constructor tracks current SSA values for every home, including
constant-address accesses between dynamic accesses. A read's existing
source read mapping points to the selected value. A write retains the
exact typed store operand as each conditional true input. The home map
is synchronized with the separately reviewed promoted-copy map. Selected
caller-visible final homes become allocation observables; source boundary
values from the addressable representation do not replace them. Helper
calls remain the existing inline source model; integration with any
additional outer caller ABI is a separately bound compiler scenario.

`make_selection_plan` validates the source graph, applies the common
compiler rule, builds the typed stream, and performs fresh register and
predicate allocation. `verify_selection_plan` rebuilds the complete plan
from those same source-only inputs. The original scalar `native_plan.py`
is unchanged. Selection metadata records whole aliases, complete-source
access/control receipts, index domains, before/after home versions, exact typed nodes, source
hashes, and the common compiler rule. No native compilation or GPU work
occurs in this module.

The author cohort uses actual RK23, Kvaerno3/LU, Radau3/BiCGSTAB, and
Rosenbrock23/MR constructors, local and source-supported shared settings,
both common selection domains, and register budgets 64 and 255. It checks
every source alias operation and every reachable target's read/write
selection algebra, fresh allocation, source/control bindings, finite
services for emitted selection operations, and exact baseline allocation
when the common rule finds no eligible extent. Independent verification
remains a separate pass before model integration.

The complete-source repair's 32 fresh saved plans retain exact prior typed
nodes, values, alias home records, and allocation. The prior independent
2,236 arithmetic checks therefore remain applicable to those unchanged
native-form artifacts; source admission receives a fresh independent pass.
The module and its source-admission dependency are both hash-bound.
