"""Lower immutable captured NumPy tables into conditional indexed LDC."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from benchmarks.hardware_model.policy_integer_division import source_form


BACKEND_LOWERING_SHA = (
    "d12039a3e788a0644667845fd5f5cc3cc904be2f01bc630fb19173c3a2f7e701"
)


def materialization_source():
    """Bind the installed constant-copy lowering supporting this form."""
    spec = importlib.util.find_spec("numba_cuda_mlir")
    if spec is None or spec.origin is None:
        raise ValueError("Captured constant form requires installed MLIR")
    path = Path(spec.origin).parent / "mlir_lowering.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BACKEND_LOWERING_SHA:
        raise ValueError("Installed captured-array lowering needs review")
    return dict(path=str(path), sha256=digest, line=1312,
                constant_global_line=1323, descriptor_line=1335)


def payload_identity(payload):
    """Name a typed immutable payload independently of trace indices."""
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def captured_array(payload):
    """Reconstruct and validate an exact contiguous FP32 or integer table."""
    if (payload.get("kind") != "array"
            or payload.get("dtype") not in ("float32", "int32", "uint32")):
        raise ValueError("Indexed LDC requires a captured four-byte array")
    array = np.ascontiguousarray(np.asarray(
        payload["values"], dtype=payload["dtype"]))
    if (list(array.shape) != payload["shape"]
            or hashlib.sha256(array.tobytes()).hexdigest()
            != payload["sha256"]):
        raise ValueError("Captured constant payload bytes differ")
    return array


def uniform_index_roots(graph, identifier, active=None):
    """Prove index uniformity from constants and coherent loop induction."""
    active = set() if active is None else active
    if identifier in active:
        return None
    value = graph["values"][identifier]
    if value["kind"] == "constant":
        return set()
    if value.get("source_origin") == "runtime_loop_induction":
        return {identifier}
    producer = value.get("producer")
    if producer is None:
        return None
    node = graph["nodes"][producer]
    if node["kind"] == "FloorDiv":
        try:
            source_form(graph, node)
        except ValueError:
            return None
    if node["kind"] not in (
            "Add", "Sub", "Mult", "BitAnd", "BitOr", "cast",
            "CapturedIndexRead", "FloorDiv"):
        return None
    results = [uniform_index_roots(graph, item, active | {identifier})
               for item in node["inputs"]]
    if any(item is None for item in results):
        return None
    return set().union(*results)


class CapturedLookupLowering:
    """Model a contiguous constant copy and explicit byte-address chain."""

    def bind_layouts(self):
        """Initialize immutable tables alongside private buffer layouts."""
        super().bind_layouts()
        self.immutable_tables = {}
        self.captured_lookup_forms = []

    def constant_table(self, node):
        """Bind the captured payload to its actual owning closure view."""
        payload = node["captured"]
        array = captured_array(payload)
        call = next(item for item in self.graph["calls"]
                    if item.get("context") == node["source"]["context"])
        matches = [dict(name=name, source_view=view)
                   for name, view in call["closure_array_views"].items()
                   if call["closure_constants"].get(name) == payload]
        if not matches:
            raise ValueError("Captured table lacks an owning NumPy closure")
        identifier = payload_identity(payload)
        table = self.immutable_tables.setdefault(identifier, dict(
            table_id=identifier,
            payload=payload,
            materialized_shape=list(array.shape),
            materialized_strides=list(array.strides),
            materialized_bytes=array.nbytes,
            address_space=4,
            immutable=True,
            identity_scope="identical_typed_payload_content",
            materialization="numpy_ascontiguousarray_captured_value",
            source_views=[],
        ))
        for match in matches:
            record = dict(owner_context=call["context"], **match)
            if record not in table["source_views"]:
                table["source_views"].append(record)
        return table, array

    def emit(self, opcode, inputs, output, source_ids, memory=None,
             semantics=None):
        """Expand the source lookup before allocating any register banks."""
        if opcode != "CAPTURED_LOOKUP":
            return super().emit(opcode, inputs, output, source_ids, memory,
                                semantics)
        if len(source_ids) != 1:
            raise ValueError("Captured lookup needs one exact source node")
        source_id = source_ids[0]
        node = self.graph["nodes"][source_id]
        table, array = self.constant_table(node)
        template = node["index_template"]
        if len(template) != array.ndim:
            raise ValueError("Captured table index rank is unresolved")
        constant = 0
        terms = []
        for axis, (index, stride) in enumerate(zip(template, array.strides)):
            if set(index) == {"literal"} and type(index["literal"]) is int:
                literal = index["literal"]
                if not 0 <= literal < array.shape[axis]:
                    raise ValueError("Constant table coordinate is outside bounds")
                constant += literal * stride
            elif (set(index) == {"dynamic_values"}
                  and len(index["dynamic_values"]) == 1):
                terms.append(dict(value=index["dynamic_values"][0],
                                  stride_bytes=stride, axis=axis))
            else:
                raise ValueError("Captured table needs scalar axis indices")
        address = self.literal(dict(dtype="uint32", value=constant))
        emitted = []
        roots = set()
        uniform = True
        for term in terms:
            source_value = term["value"]
            index = self.source_values[source_value]
            if self.values[index]["dtype"] != "int32":
                raise ValueError("Constant table address requires int32 indices")
            stride = self.literal(dict(dtype="int32", value=term["stride_bytes"]))
            result = self.value("uint32", "address",
                                f"constant_address:{source_id}:{term['axis']}")
            emitted.append(super().emit(
                "IMAD", [index, stride, address], result, source_ids,
                semantics=dict(
                    source_operation="captured_constant_byte_address",
                    native_form="32bit_indexed_constant_bank_offset",
                    table_id=table["table_id"],
                    source_index_value=source_value,
                    stride_bytes=term["stride_bytes"],
                    wrap_bits=32,
                ),
            ))
            address = result
            proof = uniform_index_roots(self.graph, source_value)
            if proof is None:
                uniform = False
            else:
                roots.update(proof)
        witness = node["selected_execution_index"]
        if any(type(index) is not int or not 0 <= index < size
               for index, size in zip(witness, array.shape)):
            raise ValueError("Declared table execution index is outside bounds")
        offset = sum(index * stride for index, stride in zip(
            witness, array.strides))
        selected = array[tuple(witness)]
        detail = dict(
            kind="immutable_constant", space="constant", access="read",
            bytes=4, table_id=table["table_id"], offset=offset,
            cell=["constant:" + table["table_id"], offset, offset + 4,
                  array.dtype.name],
            expected_semantic=self.values[output]["semantic"],
            resolved_value=dict(dtype=array.dtype.name, value=selected.item()),
            address_affine=dict(constant_bytes=constant, terms=terms),
            source_index_values=sorted(term["value"] for term in terms),
            uniform_induction_roots=sorted(roots) if uniform else [],
            broadcast_regime=("uniform_indices_over_declared_active_warp"
                              if uniform else "unproved_index_uniformity"),
            offset_is_execution_witness=True,
            source_node=source_id,
        )
        identifier = super().emit(
            "LDC", [address], output, source_ids, memory=detail,
            semantics=dict(source_operation="CapturedIndexRead",
                           native_form="LDC_register_indexed_constant_bank",
                           native_form_is_conditional=True,
                           table_id=table["table_id"]),
        )
        emitted.append(identifier)
        self.source_nodes[source_id] = emitted
        self.captured_lookup_forms.append(dict(
            source_node=source_id, typed_nodes=emitted,
            table_id=table["table_id"],
        ))
        return identifier

    def build(self):
        """Expose immutable storage and source-derived native forms."""
        lowered = super().build()
        lowered["immutable_tables"] = list(self.immutable_tables.values())
        lowered["captured_lookup_forms"] = self.captured_lookup_forms
        lowered["captured_lookup_lowering"] = dict(
            source_path=str(Path(__file__).resolve()),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            form="contiguous_constant_copy_IMAD_LDC",
            tuple_lookup="requires_separate_source_lowering",
            register_allocation="performed_after_all_lookup_operations",
            backend_materialization=materialization_source(),
        )
        return lowered


def verify_constant_load(lowered, node):
    """Verify an immutable load's storage, witness, and address operands."""
    memory = node["memory"]
    tables = {item["table_id"]: item for item in lowered["immutable_tables"]}
    table = tables[memory["table_id"]]
    array = captured_array(table["payload"])
    if (payload_identity(table["payload"]) != table["table_id"]
            or table["address_space"] != 4 or table["immutable"] is not True
            or memory["space"] != "constant" or memory["access"] != "read"
            or memory["bytes"] != 4 or node["opcode"] != "LDC"
            or memory["offset_is_execution_witness"] is not True):
        raise ValueError("Immutable constant-load contract differs")
    offset = memory["offset"]
    if type(offset) is not int or offset % 4 or not 0 <= offset < array.nbytes:
        raise ValueError("Immutable constant offset is outside its table")
    dtype = array.dtype.name
    if memory["cell"] != ["constant:" + table["table_id"], offset,
                          offset + 4, dtype]:
        raise ValueError("Immutable constant cell differs")
    if memory["resolved_value"] != dict(
            dtype=dtype, value=array.ravel()[offset // 4].item()):
        raise ValueError("Immutable constant witness value differs")
    values = lowered["values"]
    if (len(node["outputs"]) != 1
            or values[node["outputs"][0]]["semantic"]
            != memory["expected_semantic"]):
        raise ValueError("Immutable load output semantic differs")
    address = node["inputs"][0]
    for term in reversed(memory["address_affine"]["terms"]):
        producer = lowered["nodes"][values[address]["producer"]]
        if (producer["opcode"] != "IMAD"
                or producer["inputs"][0]
                != lowered["source_value_mapping"][term["value"]]
                or values[producer["inputs"][1]]["constant"]["value"]
                != term["stride_bytes"]):
            raise ValueError("Immutable load address lost a source index")
        address = producer["inputs"][2]
    if values[address]["constant"]["value"] != (
            memory["address_affine"]["constant_bytes"]):
        raise ValueError("Immutable load constant displacement differs")
