# Implicit-family contraction diagnostic

This new epoch tests Kvaerno3/LU and Radau3/BiCGSTAB independently of the
RK23 result. Root owns all native compilation, linking and launches.
Each stage is frozen and independently CPU reviewed before root runs it;
native results are separately reviewed before the following stage.
There are no timing measurements, production changes, fitted constants,
modified solver tolerances or revised prediction inputs.

The exact frozen action IDs are workload_001 for Kvaerno3/LU and
workload_003 for Radau3/BiCGSTAB, each source_0002 full/local and
source_0000 stage-count-one/local, all b128_s102400. These identities
come from the reviewed six-case prepared manifest. Both systems are the
same frozen Lorenz grid of 262144 trajectories, duration one, FP32.
The original request supplies all solver settings, iteration limits,
unroll levels, locations and constants; these are not reconstructed from
labels. The old RK23 diagnostic package remains frozen.

## Stage one: independent own references

Capture constructs the original Solver through the reviewed constructor
recipe, applies the public `contract=False` update and checks every
actual factory's fast-math flags against its original set with only
`contract` removed. The native source image, original cached LTOIR,
actual signature and all public resource attributes are retained before
two ordinary Solver.solve executions. Each complete state/status NPZ is
preserved. The source image, IR and dispatcher must remain identical
across these solves, with one specialization.

The two own-reference outputs must be finite FP32, raw SUCCESS zero,
same shape/dtype as the original bank, counter-free and exactly
repeatable. They are not required to equal the original fast-math
outputs: removing source contraction may change these implicit
workloads. Every original-versus-new own-output comparison is retained
using the original workload tolerances. Both original and new full/rolled
comparisons are retained separately, including any failures. The new
reference never replaces the old bank.

Every solve preserves the original 128-thread block and source dynamic
shared allocation, minimum four bytes. It must use one chunk, the exact
original run count and at least two full occupancy waves. The capture
serializes all four Solver lifetimes to limit memory use. No NCU or
iteration-counter instrumentation is added.

## Stage two: exact singleton IR link

Each saved original IR is linked alone with the original physical
architecture and exact link options, once with final FMA true and once
false. All rendered options and input hashes are retained. The true
image must be byte-identical to its captured source image. The false
image must actually contain zero FFMA instructions before contraction
removal is claimed or an intervention is admitted. If an explicit fused
form or another native effect defeats this gate, that observation is
retained and the following intervention is refused. No inferred zero
from the option name is accepted.

## Stage three: direct-driver functional comparison

The extended harness uses the reviewed MLIR 79-parameter layout only
when the actual source signature and every driver-reported parameter
count, offset and size match it. Arrays remain actual source-owned
objects with recorded pointer, dtype, shape and element strides; scalar
values and bit patterns are retained. The default flag set is unchanged.
This is not the legacy Numba NRT ABI.

Each direct true image must reproduce its new independent own reference
twice before its false intervention is run twice. The same memory
manager, array initialization/finalization and stream synchronization
protocol is used, with separately owned modules/functions and at least
two full waves for each actual image. All finite/SUCCESS/repeat checks
remain strict. Each family's full/rolled comparison uses that family's
unchanged original tolerance. It does not borrow RK23's equality result.

The first planned capture is eight Solver.solve executions, two per
action. Later direct validation is sixteen launches, two per image,
after the separate link/native review and direct-harness admission.
Neither stage measures performance. Only the complete evidence can show
whether the RK23 contraction diagnosis transfers to these families.
