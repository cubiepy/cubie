"""Forward exact shared reads within proved straight-line source regions."""

from benchmarks.hardware_model import implicit_native_lowering as native


STORAGE = "caller:shared_scratch"
ALTERNATIVE = "within_region_store_to_load_forwarding"


def shared_read_regions(graph):
    """Bind reaching stores to constant reads without crossing runtime joins."""
    nodes = graph["nodes"]
    full = 2**32 - 1
    if graph["regime"]["step_entry_mask"] != full:
        raise native.Unresolved("Shared forwarding needs full-warp allocation")
    if any(call["kind"] not in ("allocator", "source_call")
           for call in graph["calls"]):
        raise native.Unresolved("Unknown call prevents shared forwarding")
    construction = graph["candidate_construction"]
    stride = construction["shared_stride_bytes"]
    if construction["precision"] != "float32" or type(stride) is not int:
        raise native.Unresolved("Shared forwarding needs actual FP32 stride")
    cuts = {}
    for call in graph["calls"]:
        if call["kind"] == "source_call":
            for name, index in (("helper_entry", call["first_node"]),
                                ("helper_return", call["end_node"])):
                cuts.setdefault(index, []).append(dict(
                    reason=name, context=call["context"],
                ))
    available = {}
    previous_signature = None
    forwards = []
    boundaries = []
    region_number = -1
    for node in nodes:
        source = node["source"]
        mask = source.get("runtime_region", {}).get("entry_mask", full)
        if mask != full:
            raise native.Unresolved(
                "Shared forwarding lacks per-lane allocation merges"
            )
        branches = [key for key in node["order_predecessors"]
                    if nodes[key]["kind"] == "BranchDecision"]
        iterations = [dict(
            policy_loop_id=loop["policy_loop_id"],
            execution_index=loop["execution_index"],
        ) for loop in source.get("execution_loop_instances", [])
            if loop["recurrent"] or not loop["codegen_constant"]]
        signature = dict(
            helper=source["context"], branches=branches,
            runtime_iterations=iterations, issue_mask=mask,
        )
        reasons = list(cuts.get(node["id"], []))
        if signature != previous_signature:
            reasons.append(dict(reason="source_control_region_change"))
        if node["kind"] == "BranchDecision":
            reasons.append(dict(reason="retained_runtime_branch"))
        if reasons:
            available.clear()
            region_number += 1
            boundaries.append(dict(
                before_node=node["id"], reasons=reasons,
                region=region_number, signature=signature,
            ))
        previous_signature = signature
        cell = node.get("cell")
        if cell is None or cell[0] != STORAGE:
            continue
        _, low, high, dtype = cell
        if (dtype not in ("float32", "int32", "uint32")
                or high - low != 4 or low < 0 or low % 4 or high > stride
                or len(node["inputs"]) != 1):
            raise native.Unresolved("Shared forwarding lacks exact typed cell")
        if node.get("address_value_ids"):
            if node["kind"] == "element_write_alias":
                available.clear()
                boundaries.append(dict(
                    after_node=node["id"], region=region_number,
                    reasons=[dict(reason="dynamic_shared_alias_write")],
                    invalidated_storage=STORAGE,
                ))
            continue
        key = tuple(cell)
        if node["kind"] == "element_write_alias":
            available[key] = node
        elif node["kind"] == "element_read_alias":
            previous = available.get(key)
            if previous is not None:
                if previous["inputs"] != node["inputs"]:
                    raise native.Unresolved("Shared reaching version differs")
                forwards.append(dict(
                    read=node["id"], reaching_write=previous["id"],
                    cell=cell, source_value=node["inputs"][0],
                    region=region_number, signature=signature,
                ))
        else:
            raise native.Unresolved("Unknown shared operation")
    return dict(
        alternative=ALTERNATIVE,
        forwards=forwards,
        boundaries=boundaries,
        private_slice=dict(
            storage=STORAGE, stride_bytes=stride,
            ownership="captured caller private per-run shared slice",
        ),
        retained_shared_stores=[node["id"] for node in nodes
                                if node["kind"] == "element_write_alias"
                                and node["cell"][0] == STORAGE],
        conditions=dict(
            issue_mask=full,
            source_constant_addresses=True,
            call_boundaries_preserved=True,
            runtime_control_boundaries_preserved=True,
            dynamic_write_invalidates_entire_shared_storage=True,
            all_stores_retained=True,
            cross_iteration_phi_forwarding=False,
            cross_branch_phi_forwarding=False,
            shared_live_in_loads_retained=True,
        ),
    )


class SharedReadForwarding:
    """Reuse stored typed values while retaining every observable store."""

    def __init__(self, graph, compiler, materialization="promote"):
        self.forwarding_proof = shared_read_regions(graph)
        self.forwarded_reads = {
            item["read"]: item for item in self.forwarding_proof["forwards"]
        }
        self.stored_values = {}
        super().__init__(graph, compiler, materialization)

    def memory(self, node):
        """Replace only a proved read with its reaching store's data value."""
        witness = self.forwarded_reads.get(node["id"])
        if witness is not None:
            value = self.stored_values[witness["reaching_write"]]
            if self.values[value]["semantic"] != self.semantic(
                    node["inputs"][0]):
                raise native.Unresolved("Forwarded shared semantic differs")
            self.read_values[node["id"]] = value
            parents = {
                parent for before in witness["signature"]["branches"]
                for parent in self.source_nodes.get(before, [])
            }
            producer = self.values[value].get("producer")
            if producer is not None:
                parents.add(producer)
            self.source_nodes[node["id"]] = sorted(parents)
            self.rewrites.append(dict(
                rule="forward_within_region_shared_version",
                source_nodes=[node["id"]],
                reaching_write=witness["reaching_write"],
                retained_value=value, cell=node["cell"],
                retained_source_branch_nodes=witness["signature"]["branches"],
                store_completion_is_not_a_data_dependency=True,
            ))
            return
        super().memory(node)
        if (node["kind"] == "element_write_alias"
                and node["cell"][0] == STORAGE):
            access = self.nodes[self.source_nodes[node["id"]][-1]]
            if access["opcode"] != "STS":
                raise native.Unresolved("Shared store disappeared")
            self.stored_values[node["id"]] = access["inputs"][1]

    def build(self):
        """Bind the region proof to every retained shared store and read."""
        result = super().build()
        actual_stores = [node["source_nodes"][0] for node in result["nodes"]
                         if node["opcode"] == "STS"]
        if actual_stores != self.forwarding_proof["retained_shared_stores"]:
            raise native.Unresolved("Shared stores differ from source order")
        result["shared_forwarding"] = self.forwarding_proof
        return result
