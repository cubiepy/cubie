"""Bind numerical source replay inputs without specializing workload data."""

import ast
import hashlib
import inspect
import json
import struct

import numpy as np

from benchmarks.hardware_model.expansion import source_receipt
from benchmarks.hardware_model.workload import python_function, source_function


IEEE_DOMAIN = (
    "FP32 source operations; finite inputs and boundary results; "
    "IEEE infinity intermediates are recorded and NaN is rejected"
)


def digest(value):
    """Hash a complete JSON-compatible source-input contract."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def input_contract(graph):
    """Bind exact source live-ins, workload, policy and caller identity."""
    return dict(
        live_ins=[
            value for value in graph["values"] if value["kind"] == "live_in"
        ],
        workload=graph["candidate_construction"]["workload_identity"],
        policy=graph["policy"],
        caller=graph["caller"],
    )


def scalar_bits(dtype, value, finite=True):
    """Serialize actual FP32/int32/Boolean bits, explicitly admitting infinity."""
    converted = np.asarray(value, dtype=dtype).reshape(())
    if dtype == "float32" and (
        np.isnan(converted) or (finite and not np.isfinite(converted))
    ):
        raise ValueError("Source replay input/boundary must be finite")
    return dict(dtype=dtype, bits=converted.tobytes().hex())


def scalar_from_bits(item):
    """Recover one exact typed scalar from a retained input payload."""
    data = bytes.fromhex(item["bits"])
    dtype = np.dtype(item["dtype"])
    if len(data) != dtype.itemsize:
        raise ValueError("Replay scalar payload has the wrong width")
    return np.frombuffer(data, dtype=dtype)[0]


def first_step_point(solver, graph, bound, request):
    """Capture actual defaults and proved caller initialization for replay.

    The request declares a runtime start time and the untruncated captured
    initial timestep. It supplies no convergence labels or fitted values.
    Only callers with a proved no-op initialiser and no driver/observable
    evaluation are admitted by this constructor.
    """
    if (
        set(request) != {"kind", "time", "effective_dt"}
        or request["kind"] != "source_default_first_step"
        or request["effective_dt"] != "untruncated_captured_initial_dt"
    ):
        raise ValueError("Replay point needs an explicit first-step contract")
    loop = python_function(
        solver.kernel.single_integrator._loop.device_function
    )
    closure = inspect.getclosurevars(loop)
    raw = dict(closure.builtins, **closure.globals, **closure.nonlocals)
    tree = source_function(loop)
    initialiser = python_function(raw["initialise_state"])
    init = source_function(initialiser)
    body = [
        statement
        for statement in init.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]
    if len(body) != 1 or ast.unparse(body[0]) != "return int32(0)":
        raise ValueError(
            "First-step replay requires a proved no-op initialiser"
        )
    if int(raw["n_drivers"]) or int(raw["n_observables"]):
        raise ValueError(
            "Driver/observable initialization needs source replay"
        )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "step_function"
    ]
    if len(calls) != 1 or any(
        not isinstance(arg, ast.Name) for arg in calls[0].args
    ):
        raise ValueError("First-step caller arguments are not direct bindings")
    step = python_function(
        solver.kernel.single_integrator._algo_step.step_function
    )
    names = dict(
        zip(
            inspect.signature(step).parameters,
            (arg.id for arg in calls[0].args),
            strict=True,
        )
    )
    statements = {
        ast.unparse(node): node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
    }
    proofs = []

    def assignment(syntax):
        if syntax not in statements:
            raise ValueError(
                "Caller initialization assignment differs: " + syntax
            )
        proofs.append(dict(line=statements[syntax], syntax=syntax))

    for parameter, origin in (
        ("state", "initial_states"),
        ("parameters", "parameters"),
    ):
        assignment(f"{names[parameter]}[k] = {origin}[k]")
    for parameter in ("first_step_flag", "accepted_flag"):
        assignment(f"{names[parameter]} = True")
    for index in (0, 1):
        assignment(f"{names['counters']}[{index}] = int32(0)")
    scalar = {
        bound["dt_scalar"]["identity"]: np.float32(raw["initial_dt"]),
        bound["time_scalar"]["identity"]: np.float32(request["time"]),
        bound["first_step_flag"]["identity"]: np.bool_(True),
        bound["accepted_flag"]["identity"]: np.bool_(True),
    }
    arrays = {}
    for name, values in (
        ("state", solver.system.initial_values.values_array),
        ("parameters", solver.system.parameters.values_array),
    ):
        array = np.asarray(values)
        if array.dtype != np.dtype("float32") or array.ndim != 1:
            raise ValueError(
                "Actual source defaults must be one-dimensional FP32"
            )
        arrays[name] = array
    zeroed = [
        call
        for call in graph["calls"]
        if call["kind"] == "allocator"
        and call.get("boundary_binding")
        and call["zero"]
    ]

    def within(label, view):
        return (
            label[0] == view["storage"]
            and label[3] == view["dtype"]
            and view["bytes"] is not None
            and view["offset"]
            <= label[1]
            < label[2]
            <= view["offset"] + view["bytes"]
        )

    entries = []
    for value in graph["values"]:
        if value["kind"] != "live_in":
            continue
        key = value["id"]
        if value.get("source_origin") == "runtime_loop_induction":
            number = value["declared_trace_value"]["value"]
            origin = "internal_source_loop_witness_not_external_input"
        elif key in scalar:
            number, origin = scalar[key], "declared_first_step_scalar"
        else:
            label = value["label"]
            if not isinstance(label, list) or len(label) != 4:
                raise ValueError("Unbound first-step live-in: " + str(label))
            number = None
            for name, array in arrays.items():
                view = bound[name]["reference"]
                if within(label, view):
                    offset = label[1] - view["offset"]
                    if offset % 4 or label[2] - label[1] != 4:
                        raise ValueError("Source-default cell is not one FP32")
                    number, origin = (
                        array[offset // 4],
                        "system_default:" + name,
                    )
                    break
            if number is None and within(
                label, bound["counters"]["reference"]
            ):
                number, origin = 0, "source_reset_proposed_counter"
            if number is None:
                matches = [
                    call for call in zeroed if within(label, call["view"])
                ]
                if not matches:
                    raise ValueError(
                        "No source initial value for " + str(label)
                    )
                number = 0
                origin = "source_zero_allocator:" + str(
                    matches[0]["source"]["line"]
                )
            scalar[key] = number
        entries.append(
            dict(
                value_id=key,
                label=value["label"],
                origin=origin,
                **scalar_bits(value["dtype"], number),
            )
        )
    result = dict(
        kind=request["kind"],
        request=request,
        input_contract_sha256=digest(input_contract(graph)),
        inputs=entries,
        source=source_receipt(loop),
        initialiser=source_receipt(initialiser),
        caller_initialization=proofs,
        zero_allocator_proofs=zeroed,
        defaults={
            name: dict(
                dtype="float32",
                shape=list(array.shape),
                bytes_hex=array.tobytes().hex(),
            )
            for name, array in arrays.items()
        },
        prediction_inputs=False,
        scope=(
            "Numerical replay of the declared source path at actual system "
            "defaults and first-step initialization. Runtime convergence "
            "and branch regimes remain separately declared hypotheses; "
            "this is not a measured or inferred iteration trace."
        ),
    )
    result["point_sha256"] = digest(result)
    return result


def point_values(graph):
    """Validate complete, exact retained live-in bits for numerical replay."""
    point = graph["numerical_replay_point"]
    content = {
        key: value for key, value in point.items() if key != "point_sha256"
    }
    if point["point_sha256"] != digest(content) or point[
        "input_contract_sha256"
    ] != digest(input_contract(graph)):
        raise ValueError("Numerical replay point/source identity differs")
    expected = {
        value["id"]: value
        for value in graph["values"]
        if value["kind"] == "live_in"
    }
    entries = point["inputs"]
    if len(entries) != len(expected) or {
        entry["value_id"] for entry in entries
    } != set(expected):
        raise ValueError("Numerical replay point must bind every live-in once")
    result = {}
    for entry in entries:
        value = expected[entry["value_id"]]
        if (
            entry["dtype"] != value["dtype"]
            or entry["label"] != value["label"]
        ):
            raise ValueError("Replay input differs from source value identity")
        number = scalar_from_bits(entry)
        scalar_bits(entry["dtype"], number)
        result[entry["value_id"]] = number
    return result


def infinity_record(node, inputs, output):
    """Record each source operation that creates or consumes infinity."""
    values = list(inputs) + ([] if output is None else [output])
    if not any(
        isinstance(value, np.float32) and np.isinf(value) for value in values
    ):
        return None

    def describe(value):
        if isinstance(value, np.float32):
            return dict(
                dtype="float32",
                bits=struct.pack("<f", value).hex(),
                category=(
                    "negative_infinity" if value < 0 else "positive_infinity"
                )
                if np.isinf(value)
                else "finite",
            )
        return dict(dtype=str(np.asarray(value).dtype), value=int(value))

    return dict(
        node_id=node["id"],
        kind=node["kind"],
        source=node["source"],
        inputs=[describe(value) for value in inputs],
        output=None if output is None else describe(output),
        reason="IEEE FP32 infinity-producing or consuming operation",
    )
