# NativePlan FP32 operand forms

This component projects proved physical operand forms into a separate
NativePlan alternative. It does not change the frozen base allocator,
forwarding model, or their forecasts.

The retained whole-entry SM89 disassemblies contain FP32 numeric literals
only in operand 2 of `FADD` and `FMUL`, and operands 2 or 3 of `FFMA`.
Every observed instruction has at most one numeric literal. The source
constants at the original no-spill peak have matching native immediate bit
magnitudes except zero and one. These observations establish an available
instruction grammar. They do not identify every source use that the installed
compiler selects for that form.

The retained census is
`verification/native_operand_form_census_20260905/receipt.json`, SHA256
`6430f7685f42747493f4d518da3945d9252fcd2f39eaaf80ee99714d2acc936e`.
For chain 21 it records 1,449 `FFMA` operand-2 literals, 21 operand-3
literals, and 489 or 491 `FMUL` operand-2 literals. Chain 22 records 1,518,
22, and 491 or 493 respectively. A separate signed-magnitude set check finds
45 of 47 peak constants for chain 21 and 47 of 49 for chain 22 in the saved
native immediates; zero and unit magnitude are the two exceptions in every
case. Its receipt is
`verification/native_operand_signed_magnitude_20260905/receipt.json`. This is
set membership rather than a source-use map.

The projection therefore exposes a conditional compiler alternative. An exact
constant `FNEG` toggles the FP32 sign bit. A finite, nonzero, nonunit FP32
constant may occupy one admitted arithmetic slot. `FADD` and `FMUL` operands,
or the two multiplicands of an `FFMA`, may commute under the declared finite
input domain so the constant reaches that slot. The plan retains the original
logical operand identity and order beside the projected register operands.
An independent verifier reconstructs every logical input, checks the literal
payload bit for bit, and replays deterministic finite FP32 traces.

Zero-register and unit-operand elimination are excluded because the saved
native census does not yet bind those forms to exact source uses. General CSE,
reassociation, nonconstant sign modifiers, memory literals, and integer or
predicate encodings are also excluded.

Two schedules remain explicit. `source_topological` is the frozen source-ID
tie order. `register_release` chooses among ready nodes by the exact number of
nonconstant register words whose final use the node releases minus result
words it keeps live, then by source node ID. This uses value widths and SM89
register capacity rather than a fitted timing weight. The reported resource
choice first minimizes exact modeled spill bytes, then maximizes occupancy
from the hardware allocation equations, then minimizes retained instruction
events. It is a conditional compiler schedule and not a user-facing setting.

The output is useful before native compilation: it separates logical values
from register operands, exposes which constants can be embedded, and carries
the resulting register, spill, occupancy, and optional service scenarios. It
does not claim the installed compiler chooses every legal literal, reproduce
its final register allocation, or model caller liveness beyond the captured
region.
