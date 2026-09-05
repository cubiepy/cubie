"""Give surviving runtime loop indices source-derived native producers."""

from copy import deepcopy

from benchmarks.hardware_model import implicit_native_lowering as native


class PolicyInductionLowering:
    """Materialize loop-carried chunk bases and counted-lane offsets."""

    def __init__(self, *args, **kwargs):
        self.induction_bases = {}
        self.induction_records = []
        super().__init__(*args, **kwargs)

    def induction_form(self, source_value):
        """Derive range arithmetic from the complete captured source loop."""
        value = self.source[source_value]
        control = self.graph["policy_loops"][value["policy_loop_id"]]
        structure = control["structure"]
        instances = structure["execution_instances"]
        position = value["execution_position"]
        instance = instances[position]
        if (instance["codegen_constant"]
                or instance["position"] != position
                or instance["lane"] != value["template_lane"]
                or instance["part"] != value["loop_part"]):
            raise native.Unresolved("Induction differs from its source loop")
        start = instances[0]["value"]
        step = (instances[1]["value"] - start if len(instances) > 1 else 1)
        if any(item["value"] != start + step * item["position"]
               for item in instances):
            raise native.Unresolved("Captured induction is not affine range")
        count = (structure["directive"][1]
                 if structure["mode"] == "counted" else 1)
        chunk_position = position - instance["lane"]
        if chunk_position != instance["chunk"] * count:
            raise native.Unresolved("Counted induction chunk differs")
        if value["declared_trace_value"]["value"] != start + step * position:
            raise native.Unresolved("Induction witness differs from range")
        return control, instance, start, step, chunk_position

    def emit_induction(self, opcode, inputs, output, node, control,
                       instance, category, **details):
        """Keep source ownership while naming a distinct native form."""
        semantics = dict(
            source_operation="runtime_loop_induction",
            policy_loop_id=control["policy_loop_id"],
            induction_category=category,
            source_use_node=node["id"],
            source_control_predecessors=list(node["order_predecessors"]),
            native_form_is_conditional=True,
            wrap_bits=32,
            **details,
        )
        identifier = self.emit(
            opcode, inputs, output, [node["id"]], semantics=semantics
        )
        context = deepcopy(control["source"])
        # The loop constructor owns outer scope; the consuming operation
        # supplies the selected branch/mask under which pure work is needed.
        for key in ("runtime_region", "branch_scopes"):
            if key in node["source"]:
                context[key] = deepcopy(node["source"][key])
        if category != "initialization":
            records = context.setdefault("execution_loop_instances", [])
            records.append(dict(
                policy_loop_id=control["policy_loop_id"],
                line=control["source"]["line"], group=control["group"],
                part=instance["part"],
                lane=(instance["lane"] if category == "lane_offset" else 0),
                chunk=instance["chunk"],
                execution_index=instance["value"] - (
                    details.get("lane_offset", 0)
                    if category != "lane_offset" else 0),
                codegen_constant=False,
                recurrent=control["kind"] == "recurrent_execution_trace",
                directive=control["directive"],
            ))
        context["syntax"] = (
            f"{control['source']['syntax']} :: induction_{category}"
        )
        self.nodes[identifier]["source_contexts"] = [context]
        self.induction_records.append(identifier)
        return identifier

    def mapped(self, source_value, node=None):
        """Create internal SSA producers when a runtime index is used."""
        value = self.source[source_value]
        if value.get("source_origin") != "runtime_loop_induction":
            return super().mapped(source_value, node)
        if source_value in self.source_values:
            return self.source_values[source_value]
        if node is None:
            raise native.Unresolved("Internal induction needs a source use")
        control, instance, start, step, position = self.induction_form(
            source_value
        )
        loop_id = control["policy_loop_id"]
        state = self.induction_bases.get(loop_id)
        if state is None or state[0] != position:
            semantic = (f"source:{source_value}" if instance["lane"] == 0
                        else f"induction_base:{loop_id}:{position}")
            if state is None:
                initial_semantic = (semantic if position == 0 else
                                    f"induction_base:{loop_id}:0")
                base = self.value("int32", "expression", initial_semantic)
                self.emit_induction(
                    "MOV", [], base, node, control, instance,
                    "initialization", immediate_payload=dict(
                        dtype="int32", value=start),
                    source_range_start=start, source_range_step=step,
                    source_position=0,
                )
                state = (0, base)
            if state[0] != position:
                if position < state[0]:
                    raise native.Unresolved("Induction uses reverse time")
                base = self.value("int32", "expression", semantic)
                increment = step * (position - state[0])
                amount = self.literal(dict(dtype="int32", value=increment))
                zero = self.literal(dict(dtype="int32", value=0))
                self.emit_induction(
                    "IADD3", [state[1], amount, zero], base, node, control,
                    instance, "induction", source_range_step=step,
                    previous_source_position=state[0],
                    source_position=position, increment=increment,
                    lane_offset=instance["lane"] * step,
                )
            state = (position, base)
            self.induction_bases[loop_id] = state
        result = state[1]
        if instance["lane"]:
            offset = instance["lane"] * step
            amount = self.literal(dict(dtype="int32", value=offset))
            zero = self.literal(dict(dtype="int32", value=0))
            result = self.value(
                "int32", "expression", f"source:{source_value}"
            )
            self.emit_induction(
                "IADD3", [state[1], amount, zero], result, node, control,
                instance, "lane_offset", lane_offset=offset,
                source_position=value["execution_position"],
            )
        elif self.values[result]["semantic"] != f"source:{source_value}":
            # A later use of lane zero can reuse the previously materialized
            # chunk base: it is the same mathematical runtime SSA value.
            self.values[result]["semantic"] = f"source:{source_value}"
        self.source_values[source_value] = result
        return result

    def build(self):
        """Bind the explicit induction form and positive value replay."""
        result = super().build()
        result["induction_model"] = dict(
            form="MOV_chunk_initialization_IADD3_carried_base_and_lane",
            source_nodes=self.induction_records,
            initialization="immediate_MOV_at_first_required_source_use",
            skipped_unused_visits="source_position_difference_times_step",
            external_kernel_inputs=False,
            loop_predicate_and_backedge="separate_control_alternative",
        )
        result["induction_verification"] = verify_inductions(
            self.graph, result
        )
        return result


def verify_inductions(graph, lowered):
    """Replay actual MOV/IADD operands against source range witnesses."""
    numbers = {}
    for value in lowered["values"]:
        if value["kind"] == "constant" and value["dtype"] == "int32":
            numbers[value["id"]] = value["constant"]["value"]
    forms = []
    for node in lowered["nodes"]:
        semantics = node.get("semantics", {})
        if semantics.get("source_operation") != "runtime_loop_induction":
            continue
        if node["opcode"] == "MOV" and not node["inputs"]:
            number = semantics["immediate_payload"]["value"]
        elif node["opcode"] == "IADD3" and len(node["inputs"]) == 3:
            number = sum(numbers[value] for value in node["inputs"])
        else:
            raise native.Unresolved("Unknown induction native form")
        number = (number + 2**31) % 2**32 - 2**31
        numbers[node["outputs"][0]] = number
        forms.append(node["id"])
    checked = []
    mapping = lowered["source_value_mapping"]
    for value in graph["values"]:
        if value.get("source_origin") != "runtime_loop_induction":
            continue
        mapped = mapping[value["id"]]
        if mapped is None:
            continue
        typed = lowered["values"][mapped]
        if (typed["kind"] != "expression" or "producer" not in typed
                or numbers[mapped] != value["declared_trace_value"]["value"]):
            raise native.Unresolved("Internal induction lacks exact producer")
        checked.append(value["id"])
    return dict(status="PASS", source_values=checked, typed_forms=forms)
