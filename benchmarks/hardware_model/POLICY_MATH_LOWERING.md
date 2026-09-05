# Source-bound FP32 math forms

`policy_math_lowering.py` expands generated exponential, logarithm and
power operations into typed conditional native instruction forms before
fresh register allocation. The source graph retains all temporary values
and producer edges. It does not consume solver timings or a compiled
solver's register count.

The calibration is eleven isolated operations compiled through the
installed MLIR backend with the actual CuBIE default JIT flags. Their
source, PTX, cubin and disassembly are retained at
`C:/local_working_projects/cubie-notes/hardware_unroll_placement/math_forms_e1`.
Independent fresh disassembly and dataflow review is recorded in
`verification/cpu_continuation_independent_20260905/math_forms_native_independent_e1`.

| Source operation | Conditional native sequence |
| --- | --- |
| `exp(x)` | FP32 multiply by rounded log2(e), MUFU.EX2 |
| `log(x)` | MUFU.LG2, FP32 multiply by rounded ln(2) |
| `log2(x)` | MUFU.LG2 |
| `x ** exponent` | MUFU.LG2, FP32 exponent multiply, MUFU.EX2 |
| `x ** float32(2)` | MUFU.LG2, doubled logarithm with FADD, MUFU.EX2 |
| `x ** float32(-1)` | MUFU.LG2, negated logarithm with FADD, MUFU.EX2 |

Even FP32 integer exponents use this path in the isolated calibration.
The lowerer does not substitute repeated multiplication merely because a
source literal happens to be integral. Native immediate constants can
avoid GPR materialization; the selected materialized-GPR model remains an
explicit compiler alternative.

Each math node joins its source call and hot-template identity to the
actual owning captured dispatcher. Admission requires the exact five
fastmath flags (`afn`, `arcp`, `contract`, `ftz`, `nsz`), LTO, AST transform
and line-information settings used by calibration, with default compiler
optimization overrides. An undecorated helper cannot inherit a presumed
flag set. The serialized contract binds the complete source function
record, owner coverage and native evidence hash. Typed lowering verifies
that contract again. No native specialization is requested by this capture.

`math_owner_independent_cpu_e3/receipt.json` independently checks four
finite RK23/Kvaerno3 workloads and Fabbri Radau3. The latter has 250 source
math operations: 216 exponential, 22 power and 12 logarithm forms.

The numerical certificate follows the separate source replay contract.
Fabbri's default ACh parameter is zero: its negative power is positive
infinity, and the enclosing reciprocal expression returns a finite
value. The graph retains that intermediate rather than replacing the
parameter or dropping the operation. `math_exception_probe.py` verifies
this native FP32 path with exact output bits and no timing measurement.
All 14,336 rows passed; independent native/data review is retained at
`math_exception_independent_e1/receipt.json`. This establishes the isolated
exceptional path, not whole-solver numerical equivalence.

The service catalog assigns LG2 and EX2 the published generic Turing MUFU
15-cycle dependency latency, with the published Volta 14-cycle value as
an architecture-transfer alternative. NVIDIA's 16 results per SM cycle
gives two aggregate SM cycles per full-warp SFU operation. These are
qualified transfers; no operation-specific Ada latency was measured.
