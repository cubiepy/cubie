"""Bind explicit iteration streams to actual source call identities."""

import copy
import hashlib
import json

from benchmarks.hardware_model import implicit_workload as workload


def digest(value):
    """Hash a complete JSON source or scenario value."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def bind_call_stream(descriptor, calls, step_entry_mask):
    """Resolve every explicit top-level and nested source call.

    Parameters
    ----------
    descriptor : dict
        Actual post-codegen workload descriptor.
    calls : dict
        Exact source call IDs mapped to entry, active and terminal masks.
        A Newton body's linear ID is ``<step ID>.linear<body index>``.
    step_entry_mask : int
        Declared participating step lanes; no convergence is predicted.

    Returns
    -------
    dict
        Exact nested scenarios and source-specific counter consequences.
        This admission does not assert masked lowering support.
    """
    if not isinstance(calls, dict):
        raise ValueError("An explicit source call map is required")
    workload.mask(step_entry_mask, "step entry mask")
    if not step_entry_mask:
        raise ValueError("A stream needs at least one entered lane")
    top = descriptor["step_calls"]
    identifiers = [row["id"] for row in top]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Actual source call IDs are not distinct")
    used = set()
    rows = []

    def take(identifier, role_name, parent=None, body=None):
        if identifier in used or identifier not in calls:
            raise ValueError("Missing or colliding actual call " + identifier)
        declaration = calls[identifier]
        if not isinstance(declaration, dict) or set(declaration) != {
            "entry_mask",
            "active_masks",
            "terminal_active_mask",
        }:
            raise ValueError("A call needs exact mask fields: " + identifier)
        if not isinstance(declaration["active_masks"], list):
            raise ValueError("Active masks require an ordered body list")
        for value in [
            declaration["entry_mask"],
            *declaration["active_masks"],
            declaration["terminal_active_mask"],
        ]:
            workload.mask(value, "declared source mask")
        role = descriptor["roles"][role_name]
        if role["solver_type"] == "lu" and (
            declaration["active_masks"] or declaration["terminal_active_mask"]
        ):
            raise ValueError("A direct solve cannot have an iteration exit")
        used.add(identifier)
        rows.append(
            dict(
                id=identifier,
                role=role_name,
                parent=parent,
                body=body,
                source_function=role["function"],
                source_cap=role["cap"],
                body_iterations=len(declaration["active_masks"]),
            )
        )
        return copy.deepcopy(declaration)

    scenarios = {}
    for row in top:
        identifier = row["id"]
        role = descriptor["roles"][row["role"]]
        declared = take(identifier, row["role"])
        if role["solver_type"] == "newton":
            declared["linear_calls"] = [
                take(
                    identifier + f".linear{body}",
                    "main_linear",
                    identifier,
                    body,
                )
                for body in range(len(declared["active_masks"]))
            ]
        scenarios[identifier] = declared
    if used != set(calls):
        raise ValueError(
            "Unused source call IDs: " + repr(sorted(set(calls) - used))
        )
    evaluated = workload.evaluate_regime(
        descriptor,
        scenarios,
        step_entry_mask=step_entry_mask,
    )
    return dict(
        schema=1,
        kind="actual_source_call_stream",
        descriptor_sha256=digest(descriptor),
        declaration_sha256=digest(
            dict(calls=calls, step_entry_mask=step_entry_mask)
        ),
        step_entry_mask=step_entry_mask,
        calls=copy.deepcopy(calls),
        coverage=rows,
        scenarios=scenarios,
        evaluated=evaluated,
        assumption="Declared runtime outcomes, never numerical constants",
        lowering_admission="Requires one common source CFG allocation",
    )


def flatten_scenarios(descriptor, scenarios):
    """Expose existing exact nested scenarios through the explicit call API."""
    result = {}
    for call in descriptor["step_calls"]:
        identifier = call["id"]
        value = copy.deepcopy(scenarios[identifier])
        nested = value.pop("linear_calls", [])
        result[identifier] = value
        for body, child in enumerate(nested):
            result[identifier + f".linear{body}"] = child
    return result
