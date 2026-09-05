# Source-bound constant integer division

`policy_integer_division.py` admits the actual FIRK correction norm's
`index // state_n` when `index` is an internal int32 induction and the
complete captured source range is nonnegative. The divisor must be an
actual positive int32 source constant. The selected execution witness
resolves addresses and numerical replay but never supplies the bound or
becomes a compiler constant.

The source adapter records the owning loop, full affine range, source
numerator identity, and algebraic form. The typed adapter reconstructs
that proof from the graph, emits the selected native form, and verifies
its exact source-to-typed operand and result mapping. Existing source
order, branch, and runtime-region metadata pass through the ordinary
emitter. Fresh bank allocation accounts for every materialized constant,
operand, output, and spill. The pinned scalar `native_plan.py` is unchanged.

For divisor one the form is a move. For a power of two it is an unsigned
right shift with RZ as the zero-fill operand. The saved native
form is `SHF.R.U32.HI destination, RZ, shift, numerator`;
the model records this exact operand spelling and HI flag. Otherwise define
`W=2^32`, `M=ceil(W/d)`, and `e=M*d-W`. Admission requires
`maximum_n*e < W`; the result is the unsigned product's high 32-bit word.
All constants are algebraic, without solver timing inputs or fitted
parameters.

For `n=q*d+r`, `0<=r<d`,

```
n*M/W = q + r/d + n*e/(d*W).
```

The final two terms are nonnegative and their sum is strictly less than
one because `r<=d-1` and `n*e<W`. Thus the high word is exactly `q`.
The admitted numerator and quotient are nonnegative int32 values; signed
and unsigned result bits agree. Negative dynamic numerators, variable
divisors, non-induction expressions without a range proof, and domains
requiring reciprocal correction are explicitly rejected. Genuine source
constants use Python integer floor semantics, including negative values.
No dynamic signed truncation is substituted for floor division.

The actual corrected Fabbri Radau3 `u11100000` source has 35 states and a
70-element norm loop. Its full domain is 0 through 69. Therefore
`M=122713352`, `e=24`, and `69*24=1656 < 4294967296`. The constructor retains
70 source divisions. The complete graph and fresh allocation, including
the separately reviewed promoted-copy mixin, reach a typed policy plan.
The current author receipt records the exact graph, plan, and source
hashes under `verification/constant_division_author_e1`.

This is a conditional range-aware compiler alternative. It does not
assert that the installed Python/MLIR pipeline discovers this optimal
range simplification. Offline CUDA 13.3 PTXAS images establish that both
immediate and register `mul.hi.u32` forms compile to `IMAD.HI.U32` with RZ
as its addend. The model's selected materialized-GPR literal policy keeps
the multiplier's physical lifetime and load/move cost; the native
immediate alternative does not establish that those costs disappear in
the selected model. An explicit `shr.u32` produces SHF; a power-of-two
`mul.hi.u32` was retained as IMAD.HI by this PTXAS. Generic unsigned and
signed PTX division images retain longer reciprocal/correction sequences
and are preserved separately. Their signed PTX division is truncation,
not an admitted general Python-floor implementation.

The native arithmetic meaning comes from NVIDIA's
[PTX integer multiplication specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-mul).
`division_catalog(catalog)` explicitly adds the distinct HI and shift
opcodes. HI uses the existing primary-source IMAD 4-cycle result latency
and 5-cycle sensitivity with the named assumption of a cross-form and
cross-architecture transfer. Its integer route reserves 32 lanes against
the published 64-result/SM-cycle capacity. Shift uses the catalog's
4-cycle integer ALU form transfer. Neither is claimed to be a local
measurement of the exact opcode. The catalogue provenance and uncertainty
remain visible; the arithmetic operation never has zero or symbolic-only
cost in a fully supplied finite scenario.

Validation uses the whole-source snapshot `execution_source_e14`, the
reviewed promoted-copy module, exact arithmetic boundary cases and actual
source replay, seven offline native images, and a rebuilt actual Fabbri
graph/plan. CPU author scripts are outside the repository; no model mocks,
GPU launches, solver timings, or measured native resource labels are used.
Independent review and parent-owned native functional checks are separate
receipts and remain prerequisites for integration.
