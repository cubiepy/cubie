"""Project source-derived static instruction identities onto dynamic events."""

from collections import Counter, defaultdict
from itertools import product
import json

from benchmarks.hardware_model import implicit_native_lowering as native
from benchmarks.hardware_model.instruction_footprint import (
    Footprint,
    ONE_SLOT_OPCODES,
    WIDTH_BYTES,
    WIDTH_SOURCE,
    canonical,
    digest,
)


BOOKKEEPING = frozenset({"release", "free_home"})
ALLOCATED_FORMS = frozenset({
    "source", "constant", "spill", "reload", "predicate_to_word",
    "word_to_predicate",
})
PART_ORDER = {
    "fully_expanded": 0, "counted_main": 0,
    "backend_choice_template": 0, "backend_full": 0, "constant_tail": 1,
}


def form_signature(event, node, values):
    """Describe concrete forms without dynamic IDs or register numbers."""
    result = dict(kind=event["kind"], opcode=event["opcode"])
    if event["kind"] == "source":
        result["operand_types"] = [values[key]["dtype"]
                                   for key in node["inputs"]]
        result["result_types"] = [values[key]["dtype"]
                                  for key in node["outputs"]]
        semantics = node.get("semantics", {})
        result["native_semantics"] = {
            name: semantics[name] for name in (
                "relation", "operand_dtype", "signed", "wrap_bits",
                "vote_operation", "immediate_payload", "immediate_mask",
                "fp32_flush_subnormals", "approximate", "refinement",
                "native_form", "induction_category",
            ) if name in semantics
        }
        memory = node.get("memory")
        if memory:
            result["memory"] = {key: memory[key] for key in (
                "kind", "space", "access", "bytes", "dtype", "table_id",
                "displacement_bytes", "broadcast_regime",
            ) if key in memory}
            if memory.get("offset_is_execution_witness") is not True:
                for key in ("offset", "byte_offset"):
                    if key in memory:
                        result["memory"][key] = memory[key]
            affine = memory.get("address_affine")
            if affine:
                result["memory"]["affine_term_strides"] = [
                    term["stride_bytes"] for term in affine["terms"]
                ]
    else:
        result["operand_roles"] = [
            dict(direction=direction, position=position,
                 dtype=values[ref["value"]]["dtype"], bank=ref["bank"])
            for direction in ("reads", "writes")
            for position, ref in enumerate(event[direction])
        ]
        for key in ("payload", "offset", "bytes", "canonical_words",
                    "relation"):
            if key in event:
                result[key] = event[key]
    return result


class AddressProjection:
    """Build a maximal static-form envelope in nested lexical order."""

    def __init__(self, graph, typed, sharing, false_lowering, project_caps):
        self.graph = graph
        self.typed = typed
        self.lowered = typed["lowering"]
        self.observed = Footprint(graph, typed, sharing, false_lowering, False)
        self.projected = Footprint(
            graph, typed, sharing, false_lowering, project_caps
        )
        self.sharing = sharing
        self.slots = {}
        self.event_keys = {}
        self.node_keys = {}
        self.variants = defaultdict(set)
        self.owner_ordinals = {}
        self.lexical_ordinals = {}
        semantic_counts = Counter()
        lexical_counts = Counter()
        for node in self.lowered["nodes"]:
            base, _, execution = self.observed.operation_keys(node)
            key = canonical((base, execution))
            self.owner_ordinals[node["id"]] = semantic_counts[key]
            semantic_counts[key] += 1
            lexical = canonical((base[:-1], execution))
            self.lexical_ordinals[node["id"]] = lexical_counts[lexical]
            lexical_counts[lexical] += 1

    def lexical_key(self, node, lanes):
        """Order copied loop bodies inside their actual inline call sites."""
        context = node["source_contexts"][0]
        footprint = self.projected
        selected = {canonical(owner): slot for owner, slot in lanes}
        records = context.get("execution_loop_instances", [])
        if self.sharing == "inline":
            chain = []
            call = footprint.calls[context["context"]]
            while True:
                chain.append(call)
                site = call["call_site"]
                if site is None:
                    break
                call = footprint.calls[site["context"]]
            chain.reverse()
            segment = "inline_root"
        else:
            chain = [footprint.calls[context["context"]]]
            segment = digest(context["hot_template"]["function"])
        tokens = []
        for index, call in enumerate(chain):
            for record in records:
                control = footprint.loops[record["policy_loop_id"]]
                if control["source"]["context"] != call["context"]:
                    continue
                owner_path = (footprint.call_path(call["context"])
                              if self.sharing == "inline" else ())
                owner = (control["source"]["path"],
                         control["source"]["line"], owner_path)
                slot = selected[canonical(owner)]
                tokens.append((int(control["source"]["line"]), 0,
                               PART_ORDER[slot[0]], int(slot[1]), "loop"))
            if index + 1 < len(chain):
                site = chain[index + 1]["call_site"]
                tokens.append((int(site["line"]), 1, 0, 0, site["syntax"]))
        initialization = node.get("semantics", {}).get(
            "induction_category"
        ) == "initialization"
        tokens.append((int(context["line"]), -1 if initialization else 1,
                       0, 0, context["syntax"]))
        return segment, tuple(tokens), self.lexical_ordinals[node["id"]]

    def add_event(self, event, ordinal, sequence):
        """Bind one execution event to its observed and projected copies."""
        node = self.lowered["nodes"][event["source_position"]]
        base, dimensions, execution = self.projected.operation_keys(node)
        _, actual_dimensions, _ = self.observed.operation_keys(node)
        actual_lanes = list(product(*actual_dimensions))
        if len(actual_lanes) != 1:
            raise ValueError("One dynamic event needs exactly one static copy")
        signature = form_signature(event, node, self.lowered["values"])
        owner_ordinal = self.owner_ordinals[node["id"]]
        family = (base, owner_ordinal, event["kind"], ordinal)
        self.variants[canonical((family, actual_lanes[0]))].add(
            canonical(signature)
        )
        actual_key = None
        for lanes in product(*dimensions):
            identity = (family, lanes, signature)
            key = canonical(identity)
            if key not in self.slots:
                self.slots[key] = dict(
                    identity=digest(identity), opcode=event["opcode"],
                    form=signature, source_site=base[:-1],
                    copy_identity=lanes, form_ordinal=ordinal,
                    owner_ordinal=owner_ordinal, observed_events=[],
                    lexical_key=self.lexical_key(node, lanes),
                    allocation_sequence=sequence,
                )
            slot = self.slots[key]
            slot["allocation_sequence"] = min(
                slot["allocation_sequence"], sequence
            )
            if lanes == actual_lanes[0]:
                actual_key = key
                slot["observed_events"].append(event["id"])
        if actual_key is None:
            raise ValueError("Observed source copy is outside projected span")
        self.event_keys[event["id"]] = actual_key
        if event["kind"] == "source":
            self.node_keys[node["id"]] = actual_key

    def build(self, events):
        """Reserve lexical cap copies before assigning synthetic addresses."""
        multiplicities = Counter()
        sequence_by_node = Counter()
        for event in events:
            if event["kind"] in BOOKKEEPING:
                if event.get("opcode") is not None:
                    raise ValueError("Bookkeeping has an instruction opcode")
                continue
            if event["kind"] not in ALLOCATED_FORMS or not event.get("opcode"):
                raise ValueError("Allocation event needs an explicit form")
            if event["opcode"] not in ONE_SLOT_OPCODES:
                raise ValueError("Instruction lacks a one-slot native form")
            position = event["source_position"]
            if not 0 <= position < len(self.lowered["nodes"]):
                raise ValueError("Allocation event lacks typed source owner")
            node = self.lowered["nodes"][position]
            if event["kind"] == "source" and (
                    event["node"] != position
                    or event["opcode"] != node["opcode"]):
                raise ValueError("Allocated source instruction differs")
            signature = canonical(form_signature(
                event, node, self.lowered["values"]
            ))
            key = (position, event["kind"], signature)
            self.add_event(event, multiplicities[key], sequence_by_node[position])
            multiplicities[key] += 1
            sequence_by_node[position] += 1
        ordered = sorted(self.slots.items(), key=lambda pair: (
            pair[1]["lexical_key"], pair[1]["allocation_sequence"], pair[0]
        ))
        addresses = {}
        output = []
        for index, (key, slot) in enumerate(ordered):
            pc = index * WIDTH_BYTES
            addresses[key] = pc
            output.append(dict(
                pc=pc, **{name: value for name, value in slot.items()
                          if name not in ("lexical_key",)},
                lexical_order=slot["lexical_key"],
                observed=bool(slot["observed_events"]),
            ))
        event_map = [addresses.get(self.event_keys.get(event["id"]))
                     for event in events]
        node_map = [addresses.get(self.node_keys.get(node["id"]))
                    for node in self.lowered["nodes"]]
        if any(pc is None for pc in node_map):
            raise ValueError("A typed source operation lacks a synthetic PC")
        return dict(
            slots=output, event_to_pc=event_map, typed_node_to_pc=node_map,
            span_bytes=len(output) * WIDTH_BYTES,
            accessed_pc_union=sorted({pc for pc in event_map if pc is not None}),
            static_form_variants=[dict(
                source_family=digest(family), variants=[
                    json.loads(item) for item in sorted(variants)
                ]
            ) for family, variants in sorted(self.variants.items())
                if len(variants) > 1],
        )


def project_instruction_addresses(graph, wrapper, helper_lowering="inline",
                                  false_lowering="rolled",
                                  project_recurrent_caps=True):
    """Return synthetic instruction addresses for a declared source plan."""
    if helper_lowering not in ("inline", "identical_specialization_shared"):
        raise ValueError("Unknown helper-code alternative")
    if false_lowering not in ("rolled", "full"):
        raise ValueError("Unknown backend-choice replication alternative")
    if type(project_recurrent_caps) is not bool:
        raise ValueError("Cap projection must be an explicit bool")
    if graph.get("policy") != wrapper.get("policy"):
        raise ValueError("Source and plan policy identities differ")
    if wrapper.get("native_labels_consumed") is not False:
        raise ValueError("Projection requires a source-only plan")
    typed = wrapper["typed_plan"]
    lowered = typed["lowering"]
    allocation = typed.get("allocation")
    if allocation is not None:
        allocation_check = native.verify_allocation(lowered, allocation)
        events = allocation["events"]
    else:
        allocation_check = {"status": "typed_body_without_allocation"}
        events = [dict(
            id=node["id"], kind="source", opcode=node["opcode"],
            source_position=node["id"], node=node["id"], reads=[], writes=[],
        ) for node in lowered["nodes"]]
    if [event["id"] for event in events] != list(range(len(events))):
        raise ValueError("Event IDs must be contiguous in execution order")
    projection = AddressProjection(
        graph, typed, helper_lowering, false_lowering, project_recurrent_caps
    ).build(events)
    projection.update(
        schema=1, kind="conditional_source_instruction_address_projection",
        graph_sha256=digest(graph), plan_sha256=digest(wrapper),
        event_sha256=digest(events), instruction_width_bytes=WIDTH_BYTES,
        instruction_width_provenance=WIDTH_SOURCE,
        helper_lowering=helper_lowering, false_lowering=false_lowering,
        project_recurrent_caps=project_recurrent_caps,
        materialization=wrapper.get("compiler_materialization"),
        allocation_constructed=allocation is not None,
        allocation_verification=allocation_check,
        assumptions=[
            "Synthetic 16-byte instruction slots, not native program counters",
            "Maximal per-visit static-form envelope with explicit form variants",
            "Nested lexical call and loop-copy order; no native code scheduling",
            "Unvisited recurrent copies preserve visited source/native forms",
            "Numeric literals and spill/reload forms follow supplied allocation",
            "Input source and typed plan require their own constructor validation",
        ],
        coverage=[
            "Every executable supplied allocation event maps to one synthetic PC",
            "Bookkeeping has no PC; represented typed loop administration included",
            "Reserved cap span differs from selected dynamic accessed-PC union",
            "Unrepresented branch arms, caller work, ABI and alignment not added",
            "No fetch latency, cache residency or timing claim from this projection",
        ],
        native_labels_consumed=False, timings_consumed=False,
        fitted_parameters=False,
    )
    return projection
