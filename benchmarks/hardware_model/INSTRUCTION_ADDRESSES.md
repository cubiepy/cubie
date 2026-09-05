# Synthetic source instruction addresses

`instruction_addresses.project_instruction_addresses(graph, wrapper)`
projects a source-constructed typed body or allocated plan into synthetic
16-byte instruction slots. It returns the slots, an allocation
`event_to_pc` map, a `typed_node_to_pc` map, the reserved `span_bytes`, and
the selected execution's `accessed_pc_union`. These are source projections,
not measured SASS addresses or a complete compiled program.

The nominal helper alternative is `inline`. The optional
`helper_lowering="identical_specialization_shared"` gives a separate
compiler sensitivity. `false_lowering` selects `rolled` or `full` for
backend-choice directives; it does not assert which libNVVM chooses.
`project_recurrent_caps=True` reserves the complete source-cap copy space
under the same homogeneous-body assumption as `instruction_footprint`.

## Identity and layout

Static identities reuse `Footprint.operation_keys`: captured function
specialization, lexical helper call sites, source expression sites, native
opcode, and each actual counted/full loop-copy identity. Repeated rolled
visits share a static identity. Fully expanded lanes have distinct copies;
counted main lanes and constant tails remain separate. A local multiplicity
ordinal distinguishes repeated native forms at the same source site.

All projected copies are constructed before assigning PCs. Sorting walks
the nested source call and loop-copy path, then the source expression and
its lowering order. Consequently, unvisited Newton copies within stage
zero reserve their span before stage one begins. They are not appended to
the end of the visited trace. Source initialization MOVs precede their
loop's projected body. The layout is a lexical compiler alternative and
does not claim native scheduling or physical branch layout.

Allocated literal MOVs, spills, reloads, and predicate conversions attach
to the typed source instruction whose allocation requires them. Their
identity uses opcode, operand-position/type roles, literal payload,
memory width, and actual spill offset. It never uses a dynamic value ID or
a physical register number. Execution-witness named-array offsets do not
become static instruction operands; their explicit displacement and affine
stride form remain represented.

Allocation over a selected dynamic trace can choose different spills,
constant materializations, or forms on different rolled visits. The
projection therefore reserves the union of concrete form variants and the
maximum observed multiplicity of each form at each static site. A changed
spill offset is an explicit variant. This maximal static-form envelope is
not a proof that those visits belong to one realizable native program.
It is the declared conditional code image supplied to a delivery model.

## Coverage and execution

Every executable allocation event maps to exactly one synthetic PC.
Every typed source node has a corresponding mapped source event.
`release` and `free_home` have no instruction PC. Unknown native forms are
rejected instead of being silently assigned one instruction. The reused
allocation verifier checks the supplied event stream; source-constructor
and typed-form validation remain the caller's responsibility.

Represented typed loop administration receives ordinary instruction slots
automatically. This module does not synthesize unresolved predicates,
backedges, caller work, ABI code, omitted branch arms, or native alignment.
A body without an allocation has only its typed instructions and records
`allocation_constructed: false`.

`span_bytes` is the reserved conditional code span. `accessed_pc_union`
contains only PCs visited by supplied dynamic events and can be smaller.
Neither number is a temporal cache working set: a delivery simulation must
consume the ordered event-to-PC stream with each warp's readiness and
source execution regime. Instruction fetch completion should constrain
warp eligibility while arithmetic, memory, and other warps continue.

The output binds the graph, plan, and exact event stream by normalized
JSON hashes. It contains no native labels, timing measurements, fitted
cache costs, or inferred physical instruction-cache ownership.
