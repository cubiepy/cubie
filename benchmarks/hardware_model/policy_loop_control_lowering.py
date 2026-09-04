"""Lower fixed source-loop chunk boundaries before register allocation."""

from copy import deepcopy

from benchmarks.hardware_model import implicit_native_lowering as native


def fixed_loop_form(control):
    """Derive a finite, positive-trip signed32 bottom-tested loop form."""
    if control["kind"] != "policy_fixed_execution_trace":
        return None
    structure = control["structure"]
    instances = structure["execution_instances"]
    dynamic = [item for item in instances if not item["codegen_constant"]]
    if not dynamic:
        return None
    chunks = sorted({item["chunk"] for item in dynamic})
    if len(chunks) <= 1:
        return None
    if chunks != list(range(len(chunks))):
        raise native.Unresolved("Fixed loop chunks do not form a full range")
    width = (structure["directive"][1]
             if structure["mode"] == "counted" else 1)
    start = instances[0]["value"]
    step = instances[1]["value"] - start
    if step == 0 or any(item["value"] != start + step * item["position"]
                        for item in instances):
        raise native.Unresolved("Fixed loop range is not affine")
    if any(item["position"] != item["chunk"] * width + item["lane"]
           for item in dynamic):
        raise native.Unresolved("Fixed loop lanes differ from chunk bounds")
    for chunk in chunks:
        lanes = [item["lane"] for item in dynamic if item["chunk"] == chunk]
        if lanes != list(range(width)):
            raise native.Unresolved("Fixed loop chunk is incomplete")
    stop = start + step * width * len(chunks)
    if any(not -(2**31) <= value < 2**31
           for value in (start, stop, step * width)):
        raise native.Unresolved("Fixed-loop terminal arithmetic exceeds int32")
    return dict(
        policy_loop_id=control["policy_loop_id"],
        start=start, step=step, chunk_width=width, chunks=len(chunks),
        main_stop=stop, relation="Lt" if step > 0 else "Gt",
        source_kind=control["kind"], source_directive=control["directive"],
    )


class PolicyLoopControlLowering:
    """Emit carried bases, predicates and branches at source boundaries."""

    def __init__(self, *args, **kwargs):
        self.fixed_loop_stack = []
        self.fixed_loop_records = []
        self.fixed_loop_branch = None
        self.fixed_last_source = None
        super().__init__(*args, **kwargs)
        self.fixed_loop_forms = {
            item["policy_loop_id"]: form
            for item in self.graph["policy_loops"]
            if (form := fixed_loop_form(item)) is not None
        }

    def emit(self, *args, **kwargs):
        first = len(self.nodes)
        predecessor = self.fixed_loop_branch
        identifier = super().emit(*args, **kwargs)
        if predecessor is not None:
            # Nested native expansions can return only their final node.
            # Every emitted operation still belongs after this backedge.
            for node in self.nodes[first:]:
                node["order_predecessors"] = sorted(set(
                    node["order_predecessors"] + [predecessor]
                ))
        return identifier

    def emit_loop_induction(self, frame, node, initialization=False):
        """Share the exact carried SSA base with the body-index mapper."""
        identifier = frame["policy_loop_id"]
        control = self.graph["policy_loops"][identifier]
        form = self.fixed_loop_forms[identifier]
        position = frame["chunk"] * form["chunk_width"]
        instance = control["structure"]["execution_instances"][position]
        source_mapping = list(self.source_nodes.get(node["id"], []))
        if initialization:
            result = self.value("int32", "expression",
                                f"induction_base:{identifier}:0")
            emitted = self.emit_induction(
                "MOV", [], result, node, control, instance, "initialization",
                immediate_payload=dict(dtype="int32", value=form["start"]),
                source_range_start=form["start"],
                source_range_step=form["step"], source_position=0,
            )
            self.induction_bases[identifier] = (0, result)
        else:
            current, base = self.induction_bases[identifier]
            if current != position:
                raise native.Unresolved("Body and control induction differ")
            next_position = position + form["chunk_width"]
            result = self.value("int32", "expression",
                                f"induction_base:{identifier}:{next_position}")
            increment = form["step"] * form["chunk_width"]
            amount = self.literal(dict(dtype="int32", value=increment))
            zero = self.literal(dict(dtype="int32", value=0))
            emitted = self.emit_induction(
                "IADD3", [base, amount, zero], result, node, control,
                instance, "induction", source_range_step=form["step"],
                previous_source_position=position,
                source_position=next_position, increment=increment,
                lane_offset=0,
            )
            self.induction_bases[identifier] = (next_position, result)
        self.source_nodes[node["id"]] = source_mapping
        # A pure loop counter belongs to the loop's participating scope.
        # It does not inherit a body-only selected subbranch.
        context = self.nodes[emitted]["source_contexts"][0]
        for key in ("runtime_region", "branch_scopes"):
            if key in control["source"]:
                context[key] = deepcopy(control["source"][key])
            else:
                context.pop(key, None)
        self.nodes[emitted]["semantics"]["fixed_loop_control_base"] = True
        return result

    def emit_loop_control(self, frame, opcode, inputs, output, category):
        """Retain source-loop ownership without changing body mappings."""
        identifier = frame["policy_loop_id"]
        control = self.graph["policy_loops"][identifier]
        form = self.fixed_loop_forms[identifier]
        source_node = self.fixed_last_source
        original_mapping = list(self.source_nodes.get(source_node["id"], []))
        selected = frame["chunk"] + 1 < form["chunks"]
        semantics = dict(
            source_operation="runtime_fixed_loop_control",
            policy_loop_id=identifier, control_category=category,
            chunk_index=frame["chunk"], **{
                key: form[key] for key in (
                    "start", "step", "chunk_width", "chunks", "main_stop"
                )
            },
            relation=form["relation"], operand_dtype="int32",
            signed=True, selected_path=selected,
            explicit_selected_path=True,
            selection_reason="source-derived fixed-loop terminal bound",
            native_form_is_conditional=True,
        )
        emitted = self.emit(opcode, inputs, output, [source_node["id"]],
                            semantics=semantics)
        self.source_nodes[source_node["id"]] = original_mapping
        context = deepcopy(control["source"])
        position = frame["chunk"] * form["chunk_width"]
        instance = control["structure"]["execution_instances"][position]
        context.setdefault("execution_loop_instances", []).append(dict(
            policy_loop_id=identifier, line=control["source"]["line"],
            group=control["group"], part=instance["part"], lane=0,
            chunk=frame["chunk"], execution_index=instance["value"],
            codegen_constant=False, recurrent=False,
            directive=control["directive"],
        ))
        context["syntax"] = control["source"]["syntax"] + " :: " + category
        self.nodes[emitted]["source_contexts"] = [context]
        self.fixed_loop_records.append(emitted)
        return emitted

    def close_fixed_chunk(self, frame):
        next_base = self.emit_loop_induction(frame, self.fixed_last_source)
        form = self.fixed_loop_forms[frame["policy_loop_id"]]
        bound = self.literal(dict(dtype="int32", value=form["main_stop"]))
        predicate = self.value("bool", "expression",
                               "fixed_loop_predicate:"
                               f"{frame['policy_loop_id']}:{frame['chunk']}")
        self.emit_loop_control(frame, "ISETP", [next_base, bound], predicate,
                               "header_predicate")
        branch = self.emit_loop_control(frame, "BRA", [predicate], None,
                                       "backedge")
        self.fixed_loop_branch = branch

    def before_source_node(self, node):
        desired = [dict(policy_loop_id=item["policy_loop_id"],
                        chunk=item["chunk"])
                   for item in node["source"].get(
                       "execution_loop_instances", [])
                   if item["policy_loop_id"] in self.fixed_loop_forms
                   and not item["codegen_constant"]]
        common = 0
        for old, new in zip(self.fixed_loop_stack, desired):
            if old != new:
                break
            common += 1
        for frame in reversed(self.fixed_loop_stack[common:]):
            self.close_fixed_chunk(frame)
        self.fixed_loop_stack = self.fixed_loop_stack[:common]
        for frame in desired[common:]:
            identifier = frame["policy_loop_id"]
            form = self.fixed_loop_forms[identifier]
            if identifier not in self.induction_bases:
                if frame["chunk"] != 0:
                    raise native.Unresolved("Fixed loop skips its first chunk")
                self.emit_loop_induction(frame, node, initialization=True)
            expected = frame["chunk"] * form["chunk_width"]
            if self.induction_bases[identifier][0] != expected:
                raise native.Unresolved("Fixed loop skips a source chunk")
            self.fixed_loop_stack.append(frame)
        self.fixed_last_source = node
        return super().before_source_node(node)

    def finish_source_nodes(self):
        for frame in reversed(self.fixed_loop_stack):
            self.close_fixed_chunk(frame)
        self.fixed_loop_stack = []
        return super().finish_source_nodes()

    def build(self):
        result = super().build()
        result["fixed_loop_control_model"] = dict(
            compiler_form="positive_trip_bottom_tested_shared_carried_base",
            controls=list(self.fixed_loop_forms.values()),
            typed_control_nodes=self.fixed_loop_records,
            recurrent_exit_branches="retained_original_source_BranchDecision",
            constant_tail="source_constant_no_dynamic_control",
            false_full_replication=(
                "replication sensitivity retains these forms; a distinct "
                "native full-unroll candidate requires fresh full lowering"
            ),
        )
        result["induction_model"]["loop_predicate_and_backedge"] = (
            "typed_fixed_loop_control; recurrent_exit_control_from_source"
        )
        result["fixed_loop_control_verification"] = verify_fixed_controls(
            self.graph, result
        )
        return result


def verify_fixed_controls(graph, lowered):
    """Replay actual integer/predicate operands and complete chunk ends."""
    numbers = {value["id"]: value["constant"]["value"]
               for value in lowered["values"]
               if value["kind"] == "constant" and value["dtype"] == "int32"}
    expected = {item["policy_loop_id"]: fixed_loop_form(item)
                for item in graph["policy_loops"]}
    observed = {}
    for node in lowered["nodes"]:
        semantics = node.get("semantics", {})
        kind = semantics.get("source_operation")
        if kind == "runtime_loop_induction":
            number = (semantics["immediate_payload"]["value"]
                      if node["opcode"] == "MOV"
                      else sum(numbers[key] for key in node["inputs"]))
            numbers[node["outputs"][0]] = (number + 2**31) % 2**32 - 2**31
        if kind != "runtime_fixed_loop_control":
            continue
        identifier = semantics["policy_loop_id"]
        form = expected[identifier]
        chunk = semantics["chunk_index"]
        if form is None or not 0 <= chunk < form["chunks"]:
            raise native.Unresolved("Native loop control has no source chunk")
        key = (identifier, chunk, node["opcode"])
        if key in observed:
            raise native.Unresolved("A fixed-loop control was duplicated")
        observed[key] = node["id"]
        selected = chunk + 1 < form["chunks"]
        if node["opcode"] == "ISETP":
            left, right = (numbers[value] for value in node["inputs"])
            wanted = form["start"] + (chunk + 1) * (
                form["step"] * form["chunk_width"])
            if left != wanted or right != form["main_stop"]:
                raise native.Unresolved("Loop predicate operands differ")
            result = left < right if form["relation"] == "Lt" else left > right
            numbers[node["outputs"][0]] = result
        elif node["opcode"] == "BRA":
            result = numbers[node["inputs"][0]]
        else:
            raise native.Unresolved("Unknown fixed-loop native control")
        if result != selected or semantics["selected_path"] != selected:
            raise native.Unresolved("Loop branch decision differs from range")
    wanted = {(identifier, chunk, opcode)
              for identifier, form in expected.items() if form is not None
              for chunk in range(form["chunks"])
              for opcode in ("ISETP", "BRA")}
    if set(observed) != wanted:
        raise native.Unresolved("Fixed-loop chunk control coverage differs")
    original = [node for node in lowered["nodes"] if node["opcode"] == "BRA"
                and node.get("semantics", {}).get("source_operation")
                != "runtime_fixed_loop_control"]
    if len(original) != sum(node["kind"] == "BranchDecision"
                            for node in graph["nodes"]):
        raise native.Unresolved("Original runtime branch count changed")
    return dict(status="PASS", completed_chunks=len(observed) // 2,
                original_runtime_branches=len(original))
