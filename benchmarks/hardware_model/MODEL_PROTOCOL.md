# Physical model and discriminating experiments

The prediction boundary is generated source plus solver configuration,
before native specialization. A compiled baseline, measured iteration
counts, SASS, stack size, or a bank winner cannot be an input at that
boundary. Experiments can use all of those as labels to test mechanisms.

## Quantities with different meanings

Source operations, native instructions, executed instructions, and hot
instruction addresses are separate quantities. Unrolling changes source
replication and enables folding, scalarization, and scheduling. It need not
multiply dynamic work. Rolling an early-exit loop can reduce the instruction
working set without reducing the arithmetic performed by a converged solve.

`static_descriptors.py` records source operations and dependencies.
`workload.py` records the instantiated call graph, closure constants, and
loop directives. Neither currently estimates SASS or registers. In
particular, source scalar liveness is not a lower bound on native registers:
folding, rematerialization, address calculations, and scheduling can change
both the values and their lifetimes.

The dynamic workload is a step attempt, indexed by stage and solve call.
DIRK has a Newton solve for each implicit stage. FIRK has a coupled solve
whose state width is stages times system size. Rosenbrock has its actual
stage linear solves and optional error smoothing. The graph must count
each call exactly once. Total Newton counts must not be multiplied by the
stage count again. An iterative linear solve is indexed by its enclosing
Newton iteration when applicable.

Per-lane iteration counters also differ from warp body iterations. An
instruction still issues while any participating lane needs it. Prediction
therefore retains numerical iteration regimes as symbolic quantities and
reports which decisions change across those regimes.

## Derivation rules

1. Expand requested fixed loop copies using actual closure trip counts and
   coefficient values. Track guards and backend-controlled directives as
   unresolved when their transformation is not determined. Distinguish a
   partial unroll body from its tail and loop overhead.
2. Translate surviving source operations by instruction category. If this
   translation needs calibration, use dedicated generated fragments,
   compiler fingerprints, and inspected SASS. Do not calibrate it against
   solve timing or use a different coefficient for each system/family.
3. Preserve dependencies, buffer identity, memory space, access width, and
   active-lane structure. Build an executed-work description and a separate
   instruction-address reuse description from the same graph.
4. Apply published or measured hardware service rates and capacities. An
   arithmetic issue bound is executed work divided by the corresponding
   issue rate. A dependency bound uses dependent latency along a path.
   They can overlap; adding both as independent costs double counts work.
5. Derive register and shared-memory residency using allocation granularity,
   subpartition limits, block limits, and the chosen launch geometry.
   Native register estimates require explicit compiler lowering/scheduling
   reasoning. Do not multiply source liveness by a fitted constant.
6. Account for accesses at the hardware's sector/request granularity, with
   reads and writes separate. A named local buffer and compiler spill slots
   are different storage. Removing one local allocation can create spills
   elsewhere; the traffic change is signed.
7. Select shared carveout from supported hardware choices, including 32 KiB
   on this Ada device. Account for driver reservation, allocated shared
   bytes, occupancy, and the remaining L1 capacity. A requested preference
   alone does not establish the actual carveout.
8. Determine cache residency from accessed sectors and reuse, not allocated
   frame size times the total batch. A batch can exceed cache capacity
   while each simultaneously resident wave reuses a smaller working set.
   Conversely dirty eviction and competing instruction/data traffic can
   matter even when a named buffer appears to fit.
9. Keep instruction-cache domain, working set, and miss service separate.
   Do not attach a fitted penalty to whole-kernel SASS exceeding 128 KiB.
   The old transition motivates a controlled experiment; it does not by
   itself identify which cache's capacity or service rate caused it.

Unknown rates or transformations remain explicit. Where physical bounds
overlap, select an experiment that distinguishes the competing mechanisms;
do not invent a scalar score to force a ranking. Family defaults must be
explained by their workload graphs and validated over size/intensity
changes, with exceptions stated as physical conditions.

## First matched contrasts

Every cohort uses one frozen source/compiler epoch, numerical checks,
complete repeated samples, at least two occupancy waves, and contemporaneous
all-full references. Preserve duplicate baseline solvers in every block.
An alias supplies no additional independent measurement.

| Contrast | Question resolved |
| --- | --- |
| Lorenz Kvaerno3 and Radau5 BiCGSTAB: both rolled, each singly rolled, Newton 2/4 with Krylov full | Close the two historical physical-kernel holes and the omitted partial-Newton cells. |
| Fabbri Radau3/5: all full, old joint winners, and matched individual rollbacks | Determine why a joint rollback wins when individual rollbacks lose. |
| Lorenz/L96/chain Rosenbrock BiCGSTAB: full versus Krylov rolled | Separate size-dependent latency hiding, instruction demand, and memory traffic. |
| L96 Kvaerno5 and chain32 Kvaerno3 BiCGSTAB: Newton/Krylov contrasts | Resolve changes hidden by the common 255-register cap. |
| Chain32 Kvaerno3 stage_base at 64 threads | Test whether shared placement loses through carveout despite a smaller frame and unchanged historical residency. |
| Chain64 Radau5 delta at 32 threads | Test whether reduced local traffic/cache demand outweighs reduced residency. |
| Chain32 Vern7 stage_accumulator at 32 threads | Test the historical allocator/spill increase under current unroll semantics. |

For each contrast, inspect numerical status first, then counters and actual
generated code. The baseline can change after the counter and compiler
patches; do not pool its timing with an older epoch. Once a mechanism is
supported, vary system size or arithmetic intensity across the predicted
resource boundary and hold out those observations from derivation.

## Evidence needed before a default is called a heuristic

- Every decision input is available at the prediction boundary.
- Every numerical constant has hardware or instruction-translation
  provenance, with units and architecture/compiler scope.
- A family/solver decision is derived from its own graph, not copied from
  another family or selected by historical winner lookup.
- Predicted changes survive fresh matched measurements across the relevant
  size, intensity, iteration, and placement regimes.
- Numerical failures and unresolved identity/protocol cohorts cannot win.
- Prediction errors are diagnosed against generated code and hardware
  counters; they are not absorbed into fitted penalties.

This document defines the derivation and verification contract. It does
not claim that the physical predictor has been completed.
