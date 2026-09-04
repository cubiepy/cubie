"""Lower surviving source addresses without scalarizing trace witnesses."""

from benchmarks.hardware_model import implicit_native_lowering as native


class PolicyAddressLowering:
    """Require constant addresses throughout a promoted storage extent."""

    def bind_layouts(self):
        """Bind whole storage aliases before selecting local promotion."""
        super().bind_layouts()
        dynamic = {}
        for node in self.graph["nodes"]:
            if node.get("address_value_ids"):
                storage = node["cell"][0]
                dynamic.setdefault(storage, []).append(node["id"])
        self.promotion_eligibility = []
        cursor = 0
        for storage, layout in self.layouts.items():
            local = layout["space"] == native.base.LOCAL
            if storage in dynamic:
                self.promoted.discard(storage)
            self.promotion_eligibility.append(dict(
                storage=storage,
                bytes=layout["bytes"],
                dynamic_access_nodes=dynamic.get(storage, []),
                all_aliases_constant=storage not in dynamic,
                promoted=storage in self.promoted,
                reason=("shared_address_space" if not local else
                        "surviving_runtime_address" if storage in dynamic
                        else "all_captured_alias_addresses_constant"),
            ))
            if storage in self.promoted:
                layout["frame_offset"] = None
            elif local:
                cursor = native.base.round_up(cursor, 4)
                layout["frame_offset"] = cursor
                cursor += layout["bytes"]
        self.named_frame_bytes = native.base.round_up(cursor, 4)

    def memory(self, node):
        """Consume a symbolic byte address before the physical access."""
        if not node.get("address_value_ids"):
            return super().memory(node)
        storage, low, _, _ = node["cell"]
        affine = node.get("address_affine")
        if not affine or storage in self.promoted:
            raise native.Unresolved("Dynamic memory needs an addressable form")
        if sorted(set(term["value"] for term in affine["terms"])) != (
            node["address_value_ids"]
        ):
            raise native.Unresolved("Dynamic byte-address terms are incomplete")
        witnessed = affine["constant_bytes"] + sum(
            self.source[term["value"]]["declared_trace_value"]["value"]
            * term["stride_bytes"] for term in affine["terms"]
        )
        if witnessed != low:
            raise native.Unresolved("Symbolic address differs from trace cell")
        space = self.spaces[storage]
        base = self.bases[space]
        address = base
        address_nodes = []
        for term in affine["terms"]:
            index = self.mapped(term["value"], node)
            if self.values[index]["dtype"] != "int32":
                raise native.Unresolved("Dynamic address index must be int32")
            stride = self.literal(dict(
                dtype="int32", value=term["stride_bytes"]
            ))
            output = self.value(
                "uint32", "address", f"address:{node['id']}:{len(address_nodes)}"
            )
            address_nodes.append(self.emit(
                "IMAD", [index, stride, address], output, [node["id"]],
                semantics=dict(
                    source_operation="dynamic_byte_address",
                    logical_operation="index * stride_bytes + base",
                    wrap_bits=32,
                    signed_index=True,
                    stride_bytes=term["stride_bytes"],
                    source_index_value=term["value"],
                    native_form="IMAD_32bit_address_plus_memory_displacement",
                    native_form_is_conditional=True,
                ),
            ))
            address = output
        self.bases[space] = address
        try:
            super().memory(node)
        finally:
            self.bases[space] = base
        access = self.nodes[-1]
        access["memory"].update(
            address_affine=affine,
            displacement_bytes=(
                self.layouts[storage]["frame_offset"]
                + affine["constant_bytes"]
            ),
            offset_is_execution_witness=True,
        )
        self.source_nodes[node["id"]] = address_nodes + [access["id"]]

    def build(self):
        """Expose the physical materialization condition with its trace."""
        result = super().build()
        result["promotion_eligibility"] = self.promotion_eligibility
        result["dynamic_address_model"] = dict(
            form="32_bit_IMAD_terms_plus_memory_displacement",
            eligibility_scope="whole_aliased_storage_extent",
            zero_fill_scalarization="not_assumed",
            late_full_unroll="requires_separate_source_policy_graph",
        )
        return result
