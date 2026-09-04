"""Exact, scoped SM89 dispatch bounds from certified native work.

This module reads no GPU state and imports no compiler. The public API
is dispatch_capacity(request), combine_lower_bounds(bounds), and
profile_example(path, expected_sha256). The CLI reads --input JSON and
writes --out JSON. Counts must describe logical native warp work.
Source operation counts and unproved work estimates remain unresolved.
"""

import argparse
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


SCRIPT = Path(__file__).resolve()
IMPORTED_SHA256 = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
ARCHITECTURE = {
    "compute_capability": [8, 9],
    "warp_size": 32,
    "schedulers_per_sm": 4,
    "maximum_warp_issues_per_scheduler_cycle": 1,
    "provenance": [
        {
            "url": "https://images.nvidia.com/aem-dam/Solutions/Data-Center/"
            "l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf#page=11",
            "locator": "Figure 5, printed page 11",
        },
        {
            "url": "https://docs.nvidia.com/cuda/archive/13.0.0/"
            "cuda-c-programming-guide/index.html#compute-capability-8-x",
            "locator": "20.7.1 and 8.2.3",
        },
    ],
}
QUALIFICATIONS = (
    "certified_native_work",
    "saved_native_profile_diagnostic",
    "symbolic_native_definition",
)
SYMBOL_KINDS = (
    "native_instructions_per_visit",
    "warp_visits",
    "native_warp_work",
)


def rational(value):
    """Read a nonnegative exact integer or [numerator, denominator]."""
    if type(value) is int:
        result = Fraction(value)
    elif (
        isinstance(value, list)
        and len(value) == 2
        and all(type(part) is int for part in value)
        and value[1] > 0
    ):
        result = Fraction(*value)
    else:
        raise ValueError("Expected exact integer or rational pair")
    if result < 0:
        raise ValueError(
            "Work, coefficients and physical rates are nonnegative"
        )
    return result


def pair(value):
    """Serialize an exact rational without decimal rounding."""
    return [value.numerator, value.denominator]


def polynomial(value, symbols):
    """Read a nonnegative polynomial over explicitly declared symbols.

    The JSON form is {"terms": [{"coefficient": [1, 1],
    "powers": {"warp_visits": 1, "native_body": 1}}]}. Constants also
    accept an integer or rational pair. Repeated monomials are combined.
    """
    if not isinstance(value, dict):
        number = rational(value)
        return {(): number} if number else {}
    if set(value) != {"terms"} or not isinstance(value["terms"], list):
        raise ValueError("Unknown polynomial expression schema")
    result = {}
    for term in value["terms"]:
        if set(term) != {"coefficient", "powers"} or not isinstance(
            term["powers"], dict
        ):
            raise ValueError("Unknown native-work monomial schema")
        powers = []
        for name, exponent in term["powers"].items():
            if (
                name not in symbols
                or type(exponent) is not int
                or exponent < 1
            ):
                raise ValueError("Unknown symbol or invalid exponent")
            powers.append((name, exponent))
        key = tuple(sorted(powers))
        result[key] = result.get(key, Fraction(0)) + rational(
            term["coefficient"]
        )
    return {key: value for key, value in result.items() if value}


def expression(value):
    """Serialize a canonical polynomial, folding rational constants."""
    if not value or set(value) == {()}:
        return pair(value.get((), Fraction(0)))
    return {
        "terms": [
            {"coefficient": pair(coefficient), "powers": dict(powers)}
            for powers, coefficient in sorted(value.items())
        ]
    }


def scale(value, denominator):
    """Scale a work expression by a known positive physical divisor."""
    if denominator <= 0:
        raise ValueError("Physical divisor must be positive")
    return {
        powers: coefficient / denominator
        for powers, coefficient in value.items()
    }


def evidence(value, label):
    """Require explicit provenance rather than inventing a source."""
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) or not item for item in value)
    ):
        raise ValueError(f"Missing provenance for {label}")


def unresolved(request, reason):
    """Retain unsupported input without emitting a numeric capacity."""
    return dict(
        schema_version=1,
        kind="physical_dispatch_capacity",
        status="unresolved",
        reason=reason,
        request=deepcopy(request),
        component_sha256=IMPORTED_SHA256,
    )


def dispatch_capacity(request):
    """Return exact dispatch-work intervals and runtime lower bounds.

    Work interval endpoints must be certified logical native warp counts
    across all participating SMs. Nonnegative symbols are native definitions,
    not guessed source-operation translations. An interval is admitted only
    when coefficient-wise ordering proves lower <= upper for all symbols.
    The upper endpoint is never a runtime upper bound.
    """
    if hashlib.sha256(SCRIPT.read_bytes()).hexdigest() != IMPORTED_SHA256:
        raise ValueError("Capacity source changed after import")
    if request.get("schema_version") != 1:
        return unresolved(request, "Unsupported request schema")
    if request.get("compute_capability") != [8, 9]:
        return unresolved(request, "Only SM89 issue capacity is established")
    work = request.get("work", {})
    if work.get("kind") != "logical_native_warp_instructions" or (
        work.get("scope") != "all_participating_sms"
        or work.get("qualification") not in QUALIFICATIONS
    ):
        return unresolved(request, "Native counting contract is unproved")
    evidence(work.get("provenance"), "native work")
    domain = request.get("execution_domain")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("An explicit execution domain is required")
    symbols = request.get("symbols", {})
    for name, definition in symbols.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or (
            definition.get("kind") not in SYMBOL_KINDS
            or definition.get("nonnegative") is not True
        ):
            return unresolved(request, "Unproved symbol counting semantics")
        evidence(definition.get("provenance"), f"symbol {name}")
        if "value" in definition or "mean" in definition:
            return unresolved(
                request, "Symbol bindings require a separate scenario"
            )
    lower = polynomial(work["lower"], symbols)
    upper = (
        None
        if work.get("upper") is None
        else polynomial(work["upper"], symbols)
    )
    if upper is not None and any(
        upper.get(key, Fraction(0)) < coefficient
        for key, coefficient in lower.items()
    ):
        return unresolved(request, "Work interval ordering is not proved")
    divisor = Fraction(ARCHITECTURE["schedulers_per_sm"])
    scenarios = [("aggregate_sm_cycles", "SM_cycle", divisor, None)]
    device = request.get("device_scenario")
    if device is not None:
        if (
            type(device.get("sm_count")) is not int
            or device["sm_count"] <= 0
            or (device.get("common_sm_cycle") is not True)
        ):
            return unresolved(
                request, "Device-cycle scenario is not established"
            )
        evidence(
            device.get("provenance"), "participating SM count/common cycle"
        )
        divisor *= device["sm_count"]
        scenarios.append(("device_cycles", "cycle", divisor, device))
    clock = request.get("clock_scenario")
    if clock is not None:
        if device is None or clock.get("kind") != "explicit_common_clock":
            return unresolved(
                request, "Explicit common clock and device required"
            )
        evidence(clock.get("provenance"), "common clock scenario")
        frequency = rational(clock["hertz"])
        if frequency <= 0:
            raise ValueError("Common clock frequency must be positive")
        scenarios.append(
            (
                "wall_seconds",
                "second",
                divisor * frequency,
                dict(device=device, clock=clock),
            )
        )
    bounds = {}
    for scope, unit, denominator, scenario in scenarios:
        low = expression(scale(lower, denominator))
        high = None if upper is None else expression(scale(upper, denominator))
        bounds[scope] = dict(
            kind="runtime_lower_bound",
            execution_domain=domain,
            scope=scope,
            unit=unit,
            compute_capability=[8, 9],
            scenario=deepcopy(scenario),
            expression=low,
            dispatch_work_interval=dict(lower=low, upper=high),
            upper_interpretation=(
                "Upper dispatch-work endpoint is not a runtime upper bound"
            ),
            provenance=deepcopy(work["provenance"]),
            symbols=deepcopy(symbols),
            qualification=work["qualification"],
        )
    symbolic = any(key for key in lower) or (
        upper is not None and any(key for key in upper)
    )
    return dict(
        schema_version=1,
        kind="physical_dispatch_capacity",
        status="symbolic" if symbolic else "exact_rational",
        completeness="lower_only" if upper is None else "interval",
        architecture=deepcopy(ARCHITECTURE),
        bounds=bounds,
        request=deepcopy(request),
        component_sha256=IMPORTED_SHA256,
        limitations=[
            "A counting contract is an explicit caller proof obligation.",
            "Symbolic native definitions do not establish source lowering.",
            "No measured iteration mean or clock is supplied implicitly.",
            "Dispatch capacity is a lower bound, "
            "not attainable opcode latency.",
            "Pipeline/dependency/memory bounds need "
            "compatible units and scope.",
        ],
    )


def combine_lower_bounds(bounds):
    """Take a maximum only across compatible, proven lower bounds.

    All bounds must cover the same execution domain, cycle/time scope,
    architecture, symbol definitions and explicit scenario. Pipeline or
    dependency bounds must already be expressed in that same scope. This
    function supplies no pipeline rate or memory/dependency constants.
    """
    if not bounds:
        raise ValueError("At least one lower bound is required")
    fields = (
        "execution_domain",
        "scope",
        "unit",
        "compute_capability",
        "scenario",
        "symbols",
        "qualification",
    )
    first = bounds[0]
    allowed = {
        "aggregate_sm_cycles": "SM_cycle",
        "device_cycles": "cycle",
        "wall_seconds": "second",
    }
    if (
        first.get("scope") not in allowed
        or first.get("unit") != allowed[first["scope"]]
        or first.get("compute_capability") != [8, 9]
        or not isinstance(first.get("execution_domain"), str)
        or not first["execution_domain"].strip()
    ):
        raise ValueError("Unsupported or incompatible physical units")
    expressions = []
    provenance = []
    for bound in bounds:
        if bound.get("kind") != "runtime_lower_bound" or (
            any(
                field not in bound or bound[field] != first.get(field)
                for field in fields
            )
            or bound["qualification"] not in QUALIFICATIONS
        ):
            raise ValueError(
                "Lower bounds differ in domain/scope/qualification"
            )
        evidence(bound.get("provenance"), "lower bound")
        for definition in bound["symbols"].values():
            if definition.get("nonnegative") is not True or (
                definition.get("kind") not in SYMBOL_KINDS
            ):
                raise ValueError("Unproved lower-bound symbol semantics")
            evidence(definition.get("provenance"), "lower-bound symbol")
        normalized = expression(
            polynomial(bound["expression"], bound["symbols"])
        )
        expressions.append(normalized)
        provenance.extend(deepcopy(bound["provenance"]))
    value = (
        pair(max(rational(item) for item in expressions))
        if all(isinstance(item, list) for item in expressions)
        else dict(operation="max", operands=expressions)
    )
    return dict(
        kind="combined_runtime_lower_bound",
        **{field: deepcopy(first[field]) for field in fields},
        expression=value,
        provenance=provenance,
        combination="maximum; no additive latency assumption",
    )


def profile_example(path, expected_sha256):
    """Build a diagnostic request from one independently reviewed analysis.

    The caller supplies its expected SHA from the independent review.
    No profile is launched. Native counts and software aggregate counters
    must agree exactly; raw hardware residuals remain separate.
    """
    path = Path(path).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Reviewed analysis digest differs")
    data = json.loads(path.read_text())
    if data["status"] != "ok" or data["kind"] not in (
        "saved_solver_profile_analysis",
        "saved_placement_profile_analysis",
    ):
        raise ValueError("Unsupported saved native analysis")
    value = data["exact_totals"]["Instructions Executed"]
    if type(value) is not int or value < 0:
        raise ValueError(
            "Native warp count is not an exact nonnegative integer"
        )
    metric = data["metrics"]["inst_executed"]
    if metric["unit"] != "inst" or Decimal(metric["decimal"]) != value:
        raise ValueError(
            "Exact source/aggregate native instruction counts differ"
        )
    resources = data.get("reference_resources", data.get("resources"))
    return dict(
        schema_version=1,
        compute_capability=resources["compute_capability"],
        execution_domain=data["profile_directory"],
        symbols={},
        work=dict(
            kind="logical_native_warp_instructions",
            scope="all_participating_sms",
            qualification="saved_native_profile_diagnostic",
            lower=value,
            upper=value,
            provenance=[
                dict(
                    path=str(path),
                    sha256=actual,
                    field="exact_totals.Instructions Executed",
                )
            ],
        ),
        diagnostic_only=dict(
            hardware_source_residual=data["reconciliation"][
                "hardware_minus_source"
            ],
            elapsed_sm_cycles=data["metrics"]["sm__cycles_elapsed.sum"],
        ),
    )


def main():
    """Evaluate a JSON counting contract without device or compiler access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = dispatch_capacity(json.loads(args.input.read_text()))
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(status=value["status"], output=str(args.out.resolve()))
        )
    )


if __name__ == "__main__":
    main()
