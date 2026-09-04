# Covered instruction-body forecast

`instruction_footprint.py` forecasts a numerical native-byte contribution
from an actual policy graph and its typed plan before native compilation.
It does not consume timing banks, native labels, measured iteration counts,
or fitted parameters. Forecasts retain the exact graph and plan hashes.

NVIDIA's [Binary Utilities JSON documentation](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#json-format)
defines 16-byte instructions and illustrates SM89. The explicit compiler
alternative maps each retained typed arithmetic, predicate, control, or
memory opcode in the module's inventory to one such slot. This is an
operation-to-native-form hypothesis, not a claim that every operation
survives optimization. Unknown forms, including `CAPTURED_LOOKUP`, retain
their counts and contribute an unresolved byte term.

## Static replication

The counter uses source call sites, captured parameter specializations,
source expression identities, and each canonical loop's part/lane pair.
Repeated runtime visits to one rolled lane count once. Counted loops retain
their separate main lanes and constant-tail lanes. Full directives retain
each source copy. `False` has two explicit compiler scenarios: rolled and
fully replicated; neither is asserted to be libNVVM's universal choice.

Two numbers are returned for each scenario:

- **Covered selected templates:** slots represented by source operations
  visited in the declared semantic replay, with repeated visits collapsed.
- **Homogeneous recurrent-cap projection:** the same visited typed body
  forms extrapolated to every static lane implied by each recurrent source
  cap and directive, including nested Newton/Krylov replication.

The second is conditional on unvisited iteration lanes preserving the
visited forms and captured specialization. Both body and loop-top-exit
operations can appear in the trace; matching source/opcode identities share
their static lane. When visits lower differently, the counter uses the
maximum observed opcode multiplicity at that site. This envelope is an
explicit compiler alternative, not a proved upper or lower bound.

Helper code has two scenarios: all helpers inline at their lexical call
sites, or helpers with identical captured parameter specialization share
one body. Sharing removes inherited caller-loop copies but retains loops
owned by the helper itself. Neither scenario estimates call/return code.

## Canonical control and address alternatives

The supplementary forecast counts a specific positive-trip do-while
lowering: one `MOV` initializes the induction, one `IADD3` advances it, one
`ISETP` tests the bound, and one predicated `BRA` returns to the body. These
are reported separately by function. A single-trip main needs none of
these forms; a constant remainder has no separate runtime control. The
same source call/loop replication rules apply to this administration code.

An independent address alternative uses one 32-bit `IMAD` to form the byte
address of each dynamic FP32 cell access. Existing typed arithmetic for
the logical index remains in the body contribution. A row-major captured
table lookup uses one `IMAD` per dynamic coordinate and one `LDC`; fixed
coordinates fold into the constant byte displacement. NVIDIA's instruction
tables identify the integer-add, integer-multiply-add, predicate-compare,
branch, and constant-load forms. Their use here is an explicit lowering
hypothesis; the documentation does not guarantee compiler selection.

Address terms do not assert that promoted source aliases are valid native
register accesses, or supply a corresponding register-allocation plan.
Captured lookup terms are an alternative replacement for the unresolved
lookup operation, not an additional charge on a separately counted load.
No flat cycle or byte penalty is used. Supplementary byte contributions
remain separate so a downstream model cannot silently treat their form
and allocation assumptions as jointly established.

## Coverage boundary

The core finite numbers cover typed body instructions only. The output names
unvisited branch arms, unvisited iteration-specific code, loop induction and
backedges, dynamic address arithmetic, captured-table materialization,
literal moves, ABI work, spills/reloads, reconvergence, caller/controller
code, and alignment as unresolved additions. A separate allocation-trace
slot count is diagnostic; it is not added to static bytes.

The footprint is not a temporal instruction working set. No union of
selected program counters, instruction-cache residency claim, 128-KiB
threshold decision, or runtime winner is produced. Cache decisions require
a complete covered execution region and an instruction-delivery model.

The retained Kvaerno3/LU accumulator-only cohort illustrates a concrete
limitation: full/count1/count2/count4/False have identical typed arithmetic
footprints because initialization stores become source aliases under
promotion. Their native loop-control, addressing, and store differences
remain unresolved. Stage-count1 does change the covered arithmetic body
and introduces dynamic captured lookups. The report exposes these missing
forms rather than attributing the difference to an empirical cache penalty.

## Use

```text
python -m benchmarks.hardware_model.instruction_footprint \
  --graph graph.json --plan plan.json --out forecast.json
```

The CLI refuses to overwrite a forecast. The caller supplies an already
verified graph/plan pair; the counter checks its policy and prediction-input
boundary, then hashes both inputs. Independent graph/plan verification and
native comparison remain separate passes.
