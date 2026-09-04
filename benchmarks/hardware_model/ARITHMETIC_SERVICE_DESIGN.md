# Direct arithmetic service experiment

The separate instrument supplies arithmetic service observations for
NativePlan without fitting solver timings. The existing sparse FP32
probe has concrete endpoint and early-warp-retirement ambiguities;
`DEPENDENCY_EVIDENCE.md` records its SASS and raw receipts. Its approximately
4.05-cycle FFMA interval is not an intrinsic SM89 latency input.

This design uses the installed MLIR backend and leaves the memory-latency
instrument, CuBIE source, and existing measurement banks unchanged.
Preparation is CPU-only. A fresh compiled native artifact requires an
independent review before any launch. Ordinary timing and matched saved
counter reports have separate admission gates.

## Two populations, one counted body

Each block contains 1,024 threads. Queried SM thread capacity, register
allocation and driver occupancy must permit exactly one such block.
The grid has at least two complete occupancy waves. An unconditional
final CTA barrier retains every warp until the timed work completes.

A runtime warp-limit argument selects either the first full warp or all
32 full warps. Both treatments use the same kernel, function handle,
cubin, body, registers, block/grid, seed array and coefficients. The
native branch must use the supplied limit; source intent alone does not
prove this. The one-warp treatment measures a serialized recurrence
under that population. The dense treatment exposes aggregate issue and
execution capacity with 32 independent warp recurrences. It does not
identify a physical pipeline or subpartition mapping by itself.

Every active lane receives a lane-dependent exact seed. This also keeps
integer target operations on general registers: a uniform integer
recurrence is not accepted as an IADD3/IMAD measurement. The loop counter
may use the uniform datapath, but belongs to the separate administration
inventory.

The primary body contains 257 target operations. A 33-operation body is
a separately compiled administration-amortization control, with its own
native/resource identity. Repetition count changes runtime rather than
flattening the body. N is odd and 2N is even; exact outputs and source
counts distinguish them. Both populations use the same N/2N pair within
each binary. Calibration increases N until every accepted measurement's
minimum thread interval and ordinary event duration are at least 20 ms.
Clock conversion uses qualified before/after MHz readings and preserves
the fact that these are snapshots, not a continuous clock trace.

Two mirrored blocks of six samples per population/count retain complete
per-lane clocks, output bits, SMIDs, native identities and input hashes.
The admission records expected warp instructions and predicated thread
instructions separately. Event time includes both occupancy waves;
dividing it by one thread's chain length cannot produce instruction
latency.

For the dense treatment, all lanes of a CTA must report the same entry
and exit SMID. Its interval envelope is `max(end) - min(begin)` across
the active lanes, using unsigned 64-bit SM clocks. The native clock
guards ensure this envelope contains every target operation in that
CTA. With one resident CTA, summing these envelopes over CTAs gives a
recorded SM-cycle denominator for their counted work; it does not mix
timestamps from different SMs or convert wall time with an assumed
clock. The envelope still contains scheduling and administrative work.
Its work/envelope ratio is an achieved rate for this workload, not proof
that a pipeline is saturated. Whole-device work per ordinary event
second is reported separately. Counter-report cycle totals remain
diagnostics of the profiled run and do not replace ordinary intervals.

## Exact target recurrences

Coefficients arrive as runtime bits. Explicit PTX operations determine
the intended arithmetic; every claimed native operation is checked in
SASS. Register allocation, operand modifiers, constant-bank operands and
any retained coefficient loads remain visible in the inventory.

| Target | Recurrence and exact oracle | Native requirement |
|---|---|---|
| FADD | `x = a - x`, `a = 1`; lane seed `(2*lane+1)/128` for lanes 0..31; alternates between the seed and its exact complement | FADD with the observed sign modifier and a result-to-next-input GPR dependency |
| FMUL | `x = m*x`, `m = -1`; normal, nonzero binary-rational lane seeds; sign alternates | FMUL, not an eliminated move or sign-only replacement |
| FFMA | `x = m*x+a`, `m = -1`, `a = 1`; the FADD seeds and complements are exact | One FFMA per intended step, not separate multiply/add or algebraic replacement |
| IADD3 | `x = x+s mod 2^32`, odd runtime `s`; exact unsigned modular output | GPR IADD3 target chain, distinct from its counter |
| IMAD | `x = 5*x+1 mod 2^32`, runtime coefficients; exact modular affine-composition oracle | GPR IMAD low-word chain with the actual multiplier/addend operands |

The floating-point values stay normal and finite. Their two-state cycles
are exact FP32 identities for the chosen binary-rational inputs, not
stabilization assumptions about a general solve. Integer address
arithmetic uses integer oracles; it is not FP64 computation. The IMAD
oracle uses modular composition rather than iterating millions of host
steps. IMAD.WIDE is a distinct operation and receives no low-word IMAD
service value by substitution.

## Approximate reciprocal and move boundaries

`rcp.approx.f32` permits an error up to one ulp. A presumed exact
`1.5 -> 2/3 -> 1.5` cycle is therefore not an adequate oracle.
[PTX ISA 9.3, section 9.7.3.13](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp).

The reciprocal case first records an untimed sequence of actual native
MUFU.RCP results for normal finite seeds. Each step is checked against
an exact integer-rational reciprocal and the stated FP32 error bound.
A repeated bit pattern must establish a closed transition cycle before
that seed qualifies for long counted timing. The trace is retained as
functional evidence, with its source, native opcode, input bits and
device identity; it is not a fitted timing coefficient. A bounded trace
that does not close does not manufacture a cycle or a latency result.
N/2N expected bits are computed from the verified cycle and must differ.
Special zero/infinity paths cannot stand for normal reciprocal service.
The exact relation between the functional trace and counted native
instruction form requires explicit admission and independent review.

A PTX move chain may disappear through register coalescing. Its absence
is evidence of that lowering, not a zero-cycle native MOV. A retained
move experiment must prove a real result dependency and count its
actual opcode form, including IMAD.MOV when emitted. If the compiler
cannot retain a standalone chain with a valid observable endpoint, the
instrument reports that capability result and preserves the artifact;
the model receives no fabricated MOV service constant. The same rule
applies to a wide-integer chain whose high word becomes dead.

## Clock and native proof

One side-effecting inline-PTX primitive encloses each timed body and its
clocks. Initial operand loads and a guard precede the start clock. A
guard consuming the final arithmetic result precedes the end clock.
The exact expected result is available before timing when needed for
an integer guard. The native proof must show these dependencies and
domination, not merely instruction order. The guard and administrative
work remain part of the measured interval.

The native gate derives the parameter ABI from the emitted signature
and ELF record. It requires the counted body to have the declared
number of target instructions, a complete per-register-word dependency
witness, a runtime counter with a dominating reaching definition, and
only explicitly proved loop/control/operand administration. Starting
and ending timestamp pairs, coefficients and result words receive
full implicit-high-word clobber checks. The starting timestamp remains
live across every body instruction. No added memory or call work is
silently discarded. Terminal call-shaped exits require the same
forward, nonreturning control proof used by the reviewed memory probe.
Every reachable kernel exit must pass the final CTA barrier.

Profiles capture exactly one admitted population/count treatment and
retain its benchmark source snapshot, full command/environment, report,
raw arrays, cubin and SASS. Saved-report audits check every native PC,
full-lane masks, target and administrative counts, driver/resource
limits, actual block population and two-wave geometry. Raw timing from
a profiled launch is excluded from the ordinary bank.

## Catalog interpretation

The primary outputs are the complete dependent-interval distributions
and dense ordinary throughput, each with exact work and population.
N/2N differences and the 33/257 contrast retain their full distribution;
the median of paired values does not imply every pair is equal. No
unknown administrative instruction is assigned zero cost, and no
baseline is blindly subtracted when native allocation or scheduling
changes.

NativePlan currently requires latency, initiation interval, pipeline
name and SM/subpartition scope together (`native_plan.py:1264`). A
dependent result alone does not fill this record. Dense measurements
and published operation capacity can support an explicitly conditional
resource-sharing scenario, while unmeasured physical mapping remains
an assumption in that scenario. Aggregate throughput alone cannot
identify whether two opcode families share a physical execution path.

NVIDIA's table defines theoretical rates as results per SM clock and
distinguishes those rates from latency. Its CC8.9 FP32 add/multiply/FMA
capacity is 128 results/SM clock; one full warp instruction comprises
32 results, with an FMA still counted as one result in this convention.
These are capacity inputs, not measurements of a four-cycle recurrence.
[CUDA Best Practices Guide 13.3, section 12.1.1, Table 5](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions).

The same guide's approximately four-cycle latency statement explicitly
concerns CC7.0. Other-architecture microbenchmarks may be retained as
labeled proxy scenarios, but cannot become measured SM89 values. No
service observation is fitted against solver runtimes or used to tune
a family weight or liveness multiplier.
