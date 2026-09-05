"""Retain source-derived caller values during exact step allocation."""

import hashlib
import json

from benchmarks.hardware_model import implicit_native_lowering as native


class CallerLiveThrough:
    """Bind whole-allocation caller materialization before native lowering."""

    def __init__(
        self, graph, compiler, inventory, cells, cell_form, pointer_form
    ):
        graph_hash = hashlib.sha256(json.dumps(
            graph, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if inventory["step_graph_sha256"] != graph_hash or (
            cells["step_graph_sha256"] != graph_hash
            or cells["caller_configuration"] != inventory["caller_configuration"]
        ):
            raise ValueError("Caller context differs from exact step construction")
        if cell_form not in ("promoted_constant_cells", "addressable_storage"):
            raise ValueError("Unknown caller cell compiler form")
        if pointer_form not in (
            "parameter_rematerialization",
            "retained_descriptor",
        ):
            raise ValueError("Unknown caller pointer compiler form")
        if cells["unresolved"] and cell_form == "promoted_constant_cells":
            raise ValueError(
                "Caller local promotion requires complete constant aliases"
            )
        self.caller_inventory = inventory
        self.caller_cells = cells
        self.caller_cell_form = cell_form
        self.caller_pointer_form = pointer_form
        super().__init__(graph, compiler, "promote")

    def bind_layouts(self):
        super().bind_layouts()
        additional = [
            row
            for row in self.caller_cells["cells"]
            if row["additional_caller_value"]
        ]
        dynamic = {
            node["cell"][0]
            for node in self.graph["nodes"]
            if node.get("address_value_ids")
        }
        caller_dynamic = {
            row["storage"]
            for row in self.caller_cells.get("dynamic_accesses", [])
        }
        dynamic |= caller_dynamic
        for storage in caller_dynamic:
            self.promoted.discard(storage)
        for row in additional:
            storage, _, end, _ = row["cell"]
            if storage not in self.layouts:
                raise ValueError(
                    "Caller cell has no captured allocator layout"
                )
            self.layouts[storage]["bytes"] = max(
                end, self.layouts[storage]["bytes"]
            )
            if self.caller_cell_form == "addressable_storage":
                self.promoted.discard(storage)
            elif storage in dynamic:
                raise ValueError(
                    "Caller cell aliases surviving step runtime address"
                )
        cursor = 0
        for storage, layout in self.layouts.items():
            if storage in self.promoted:
                layout["frame_offset"] = None
            elif layout["space"] == native.base.LOCAL:
                cursor = native.base.round_up(cursor, 4)
                layout["frame_offset"] = cursor
                cursor += layout["bytes"]
            else:
                layout["frame_offset"] = 0
        self.named_frame_bytes = native.base.round_up(cursor, 4)
        for entry in self.promotion_eligibility:
            entry["bytes"] = self.layouts[entry["storage"]]["bytes"]
            entry["promoted"] = entry["storage"] in self.promoted
            if any(row["cell"][0] == entry["storage"] for row in additional):
                entry["caller_cell_form"] = self.caller_cell_form

    def build(self):
        result = super().build()
        added, joined = [], []
        observable = set(result["observable_values"])

        def retain(name, dtype, count=1, **provenance):
            values = []
            for word in range(count):
                lowered_dtype = "int32" if dtype == "float64" else dtype
                value = self.value(
                    lowered_dtype,
                    "live_in",
                    "caller_live_through:" + name + ":" + str(word),
                    caller_source_dtype=dtype,
                    caller_word_index=word,
                    caller_value=name,
                    **provenance,
                )
                observable.add(value)
                values.append(value)
            added.append(
                dict(
                    name=name,
                    source_dtype=dtype,
                    typed_values=values,
                    **provenance,
                )
            )

        for row in self.caller_inventory["scalar_live_through"]:
            if len(row["dtypes"]) != 1:
                raise ValueError(
                    "Caller scalar has unresolved or mixed type: "
                    + row["name"]
                )
            dtype = row["dtypes"][0]
            if row["step_source_values"]:
                if len(row["step_source_values"]) != 1:
                    raise ValueError(
                        "Ambiguous existing caller scalar identity"
                    )
                source_value = row["step_source_values"][0]
                value = self.mapped(source_value)
                if self.values[value]["dtype"] != dtype:
                    raise ValueError("Caller scalar alias changes its type")
                observable.add(value)
                joined.append(
                    dict(
                        name=row["name"],
                        source_value=source_value,
                        typed_value=value,
                    )
                )
            elif dtype == "float64":
                retain(
                    row["name"],
                    dtype,
                    2,
                    representation="opaque_binary64_low_high_words",
                )
            elif dtype in ("float32", "int32", "uint32", "bool"):
                retain(row["name"], dtype)
            else:
                raise ValueError("Unmodeled caller scalar type " + dtype)
        for row in self.caller_cells["cells"]:
            if not row["additional_caller_value"]:
                continue
            storage, low, high, dtype = row["cell"]
            if storage not in self.promoted:
                continue
            if high - low != 4:
                raise ValueError(
                    "Caller promoted element width must be 32 bits"
                )
            retain(
                "cell:" + repr(row["cell"]),
                dtype,
                cell=row["cell"],
                representation="constant_cell_scalar_replacement",
            )
        for row in self.caller_inventory["pointer_live_through"]:
            reference = row["reference"]
            if reference.get("address_space") != "global":
                continue
            if self.caller_pointer_form == "retained_descriptor":
                # The demanded view has a 64-bit base and one 64-bit byte
                # stride per source axis; shapes have no bounds-check use.
                fields = ["base"] + [
                    "stride" + str(axis)
                    for axis in range(reference["source_index_rank"])
                ]
                for field in fields:
                    retain(
                        row["name"] + ":" + field,
                        "uint32",
                        2,
                        representation="opaque_global_descriptor_word_pair",
                        source_pointer=reference,
                    )
        result["observable_values"] = sorted(observable)
        result["caller_live_through"] = dict(
            scalar_inventory=self.caller_inventory,
            cell_inventory=self.caller_cells,
            cell_form=self.caller_cell_form,
            pointer_form=self.caller_pointer_form,
            added_values=added,
            existing_scalar_joins=joined,
            local_address_form="Existing local/shared base plus source-literal view displacement",
            rematerialization_scope="Global descriptor reload after the step; those caller instructions are outside the attempted-step interval",
            binary64_scope="Opaque live-through bits, no FP64 state or added FP64 arithmetic",
        )
        return result

