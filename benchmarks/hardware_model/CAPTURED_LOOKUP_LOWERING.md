# Captured constant-table lowering

`CapturedLookupLowering` expands a retained `CapturedIndexRead` into
source-index byte arithmetic and an immutable `LDC` before register
allocation. The form is a compiler hypothesis supported by the installed
MLIR implementation, not an observed native instruction sequence.

The installed `mlir_lowering.py:1312` calls `np.ascontiguousarray(value)`.
Lines 1323–1331 emit an internal constant global in address space 4;
lines 1335–1343 construct its descriptor using the copy's shape and
strides. The component binds this implementation's exact source hash.
Original `closure_array_views` root identities, shapes, strides, offsets,
and snapshots remain in the table provenance. They are not substituted
for the contiguous copy's physical strides. Each captured NumPy view is
materialized as its own logical value; immutable identical payloads have
content identities without asserting native allocation coalescing.

Each dynamic scalar coordinate contributes one 32-bit `IMAD`, using the
materialized byte stride. Literal coordinates contribute to the initial
constant displacement. The final address feeds a conditional indexed
constant-bank `LDC`. Kvaerno3's four-by-four FP32 table therefore has byte
address `16*i + 4*j`. Source dependency IDs may be sorted independently
of axis order; the index template retains the axis-to-value mapping.

Immutable loads carry table identity, the complete typed payload,
source-index affine terms, and an execution-witness cell and result.
Witness offsets verify the selected source execution but never replace
dynamic address operands. Allocation verification checks table bytes,
result semantics, and the actual address def-use chain. These loads do
not participate in mutable local/shared alias histories.

Broadcast is admitted only when source index dependencies reduce to
constants and declared coherent loop-induction values through supported
integer operations or immutable table reads. An unproved index remains
explicitly unqualified for broadcast. This qualification does not provide
a constant-cache residency or throughput claim; the scheduler needs a
separate hardware service and cache hypothesis.

The admitted payload types are four-byte float32, int32, and uint32
NumPy arrays with scalar, in-bounds selected coordinates. Tuple sources,
boolean byte arrays, slice-valued lookups, and unresolved index templates
raise an explicit unsupported-form error. They are not silently assigned
a four-byte constant load. Native conversion to selection trees, literal
folding, or outlining remains compiler sensitivity.
