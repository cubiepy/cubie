"""Audit saved solver source counters without CUDA imports or execution.

Use --profile-dir, --source-counters, --nvdisasm and a fresh --out.
External tools only import an existing Nsight report or disassemble a
saved cubin. Exact opcode work is separate from sampled stalls.
"""

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


SCRIPT = Path(__file__).resolve()
EXACT_COLUMNS = {
    "Instructions Executed": "inst_executed",
    "Thread Instructions Executed": "thread_inst_executed",
    "Predicated-On Thread Instructions Executed": "thread_inst_executed_true",
}
SAMPLE_COLUMNS = (
    "Warp Stall Sampling (All Samples)",
    "Warp Stall Sampling (Not-issued Samples)",
    "# Samples",
)
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s*(.*?);\s*$")
LABEL = re.compile(r"^\s*([\w.$]+):\s*$")
SECTION = re.compile(r"^\s*//-+\s*\.text\.(\S+)")
TARGET = re.compile(r"`\(([\w.$]+)\)")


def digest(path):
    """Hash exact file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    """Hash the persisted cohort manifest's canonical representation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    """Preserve exact integer counts and reject nonfinite JSON numbers."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def integer(value):
    """Read a nonnegative integer counter without rounding."""
    number = Decimal(value.replace(",", ""))
    if not number.is_finite() or number < 0 or number != int(number):
        raise ValueError(f"Invalid exact counter: {value}")
    return int(number)


def read_metrics(path):
    """Retain the CSV unit row and require exactly one profiled launch."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 3 or len(set(map(len, rows))) != 1:
        raise ValueError("Expected one metric header, unit row and launch")
    names, units, values = rows
    if len(set(names)) != len(names) or names[0] != "ID":
        raise ValueError("Invalid or duplicate metric names")
    result = {}
    for name, unit, value in zip(names, units, values):
        try:
            parsed = Decimal(value.replace(",", ""))
            numeric = str(parsed) if parsed.is_finite() else None
        except InvalidOperation:
            numeric = None
        result[name] = dict(raw=value, unit=unit, decimal=numeric)
    return result


def quantity(metrics, name, unit):
    """Require the exact exported metric unit before arithmetic."""
    row = metrics[name]
    if row["unit"] != unit or row["decimal"] is None:
        raise ValueError(f"Unknown metric value/unit for {name}: {row}")
    return Decimal(row["decimal"])


def source_rows(path):
    """Read the kernel header and every exact source-PC counter row."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        kernel = next(reader)
        names = next(reader)
        if kernel[0] != "Kernel Name" or names[:2] != ["Address", "Source"]:
            raise ValueError("Expected exported SASS SourceCounters headers")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate source counter columns")
        required = set(EXACT_COLUMNS) | set(SAMPLE_COLUMNS)
        if not required.issubset(names):
            raise ValueError("Missing exact-execution or sampled columns")
        samples = list(SAMPLE_COLUMNS) + [
            name for name in names if name.startswith("stall_")
        ]
        result = []
        for line, values in enumerate(reader, 3):
            if len(values) != len(names):
                raise ValueError(f"Incomplete source CSV row {line}")
            row = dict(zip(names, values))
            address = int(row["Address"], 16)
            result.append(
                dict(
                    csv_line=line,
                    address=address,
                    source=row["Source"].strip(),
                    exact={name: integer(row[name]) for name in EXACT_COLUMNS},
                    sampled={name: integer(row[name]) for name in samples},
                )
            )
    if not result or len({row["address"] for row in result}) != len(result):
        raise ValueError("Source-PC inventory is empty or contains duplicates")
    return kernel[1], result


def validate_exports(profile, source, output):
    """Re-export the saved report on CPU and bind both CSVs to it."""
    command = json.loads((profile / "command.json").read_text())
    executable = Path(command["executable"])
    receipts = []
    for path, arguments in (
        (profile / "metrics.csv", ["--page", "raw"]),
        (
            source,
            [
                "--page",
                "source",
                "--print-source",
                "sass",
                "--section",
                "SourceCounters",
            ],
        ),
    ):
        args = [str(executable), "--import", str(profile / "profile.ncu-rep")]
        args += arguments + ["--csv"]
        result = subprocess.run(
            args, capture_output=True, text=True, check=True
        )
        with path.open(newline="", encoding="utf-8-sig") as handle:
            saved = list(csv.reader(handle))
        if list(csv.reader(io.StringIO(result.stdout))) != saved:
            raise ValueError(
                f"Saved CSV differs from report re-export: {path}"
            )
        receipts.append(
            dict(
                command=args,
                returncode=result.returncode,
                stderr=result.stderr,
                saved_csv_sha256=digest(path),
                csv_equal=True,
            )
        )
    receipt = dict(executable_sha256=digest(executable), exports=receipts)
    write_json(output / "report_export_identity.json", receipt)
    return receipt


def disassemble(cubin, executable, output, entry):
    """Read one saved native section with all local/helper symbols."""
    command = [str(executable), "-c", str(cubin)]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=True
    )
    (output / "reference.sass").write_text(completed.stdout, encoding="utf-8")
    write_json(
        output / "disassembly_command.json",
        dict(
            command=command,
            returncode=completed.returncode,
            stderr=completed.stderr,
            executable_sha256=digest(executable),
        ),
    )
    sections = []
    labels = {}
    instructions = []
    for line in completed.stdout.splitlines():
        match = SECTION.match(line)
        if match:
            sections.append(match[1])
        match = LABEL.match(line)
        if match:
            labels[match[1]] = len(instructions)
        match = INSTRUCTION.match(line)
        if match:
            instructions.append((int(match[1], 16), match[2].strip()))
    if sections != [entry] or not instructions:
        raise ValueError("Unsupported native section topology or entry")
    if [item[0] for item in instructions] != list(
        range(0, len(instructions) * 16, 16)
    ):
        raise ValueError("Noncontiguous SM89 native instruction addresses")
    return instructions, labels


def match_native(rows, instructions, labels):
    """Match every relocated PC, opcode, operand and branch target."""
    if len(rows) != len(instructions):
        raise ValueError("Profile and cubin instruction inventories differ")
    base = rows[0]["address"]
    for row, (offset, native) in zip(rows, instructions):
        if row["address"] != base + offset:
            raise ValueError("Source and native instruction offsets differ")

        def relocate(match):
            return hex(base + instructions[labels[match[1]]][0])

        normalized = TARGET.sub(relocate, native).replace(".reuse", "")
        display = re.sub(
            r"`\([\w.$]+\)\s+(?=0x[0-9a-fA-F]+)", "", row["source"]
        )
        if (
            normalized.replace(",", " ").split()
            != display.replace(",", " ").split()
        ):
            raise ValueError(
                f"Native/source mismatch at {offset:#x}: "
                f"{native!r} versus {row['source']!r}"
            )
        words = native.split()
        predicate = words.pop(0) if words[0].startswith("@") else ""
        row.update(
            offset=offset,
            native=native,
            opcode=words[0],
            predicate=predicate,
            encoded_bytes=16,
        )
    return base


def array_identity(value):
    """Hash array shape, dtype and ordered value bytes."""
    return dict(
        shape=list(value.shape),
        dtype=value.dtype.str,
        sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    )


def recorded_row(receipt):
    """Resolve an exact original bank line and record key."""
    path = Path(receipt["path"])
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if number == receipt["line"]:
                row = json.loads(line)
                if row["key"] != receipt["record_key"]:
                    raise ValueError("Bank line/key identity differs")
                return row
    raise ValueError("Missing original bank record")


def validate_labels(profile):
    """Verify raw counter/state labels and original state-only binary."""
    paths = list((profile / "benchmark").rglob("labels.json"))
    if len(paths) != 1:
        raise ValueError("Expected one completed counter-label case")
    path = paths[0]
    data = json.loads(path.read_text())
    if data["status"] != "ok" or data["kind"] != "policy_iteration_labels":
        raise ValueError("Profile benchmark labels did not complete")
    cohort = data["cohort"]
    manifest = cohort["manifest"]
    if json_digest(manifest) != cohort["manifest_sha256"]:
        raise ValueError("Original cohort manifest digest differs")
    if data["source_hash"] != manifest["source_hash"] or (
        data["compiler_identity"] != manifest["compiler_identity"]
    ):
        raise ValueError("Profile and original source/compiler differ")
    if data["duration"] != cohort["protocol"]["duration"] or (
        data["n_runs"] != cohort["protocol"]["n_runs"]
    ):
        raise ValueError("Profile duration/grid differs from completed bank")
    plan = data["plan"]
    original = recorded_row(plan["bank_compile"]["_receipt"])
    reference = data["compiled"]["reference"]
    if (
        original["source_hash"] != data["source_hash"]
        or original["policy"] != plan["policy"]
        or original["cubin_sha256"] != reference["cubin_sha256"]
    ):
        raise ValueError("Profile reference binary/policy differs from bank")
    for compiled in data["compiled"].values():
        if digest(compiled["cubin_path"]) != compiled["cubin_sha256"]:
            raise ValueError("Saved reference/instrumented cubin hash differs")
        if compiled["compute_capability"] != [8, 9]:
            raise ValueError("Only the recorded SM89 encoding is supported")
        if compiled["geometry"]["waves"] < 2:
            raise ValueError(
                "Original compiled occupancy has fewer than 2 waves"
            )
    with np.load(data["input_artifact"], allow_pickle=False) as arrays:
        identities = {
            name: array_identity(arrays[name]) for name in arrays.files
        }
        if identities != data["preparation"]["input_arrays"]:
            raise ValueError("Raw input grid identity differs")
    if len(data["samples"]) != 1:
        raise ValueError(
            "Only the single reference-first profile is supported"
        )
    sample = data["samples"][0]
    if (
        not all(sample["checks"].values())
        or not sample["eligible_validation_label"]
    ):
        raise ValueError("Counter label numerical checks did not pass")
    with np.load(sample["artifact"], allow_pickle=False) as arrays:
        actual = {name: array_identity(arrays[name]) for name in arrays.files}
        if actual != sample["arrays"]:
            raise ValueError("Raw label array identity differs")
        if not np.array_equal(arrays["state"], arrays["reference_state"]):
            raise ValueError("Profiled reference and counter state differ")
        if not np.all(np.isfinite(arrays["state"])) or any(
            np.any(arrays[name])
            for name in ("status_codes", "reference_status_codes")
        ):
            raise ValueError("Nonfinite state or nonzero status")
        if not np.array_equal(
            arrays["iteration_counters"].sum(axis=0, dtype=np.int64),
            arrays["per_run_totals"],
        ):
            raise ValueError("Raw per-run counter totals differ")
        counter_totals = arrays["per_run_totals"].sum(axis=1, dtype=np.int64)
    timings = []
    for receipt in plan["timing_receipts"]:
        row = recorded_row(receipt)
        if (
            row["warm"]
            or row["duration"] != data["duration"]
            or row["n_runs"] != data["n_runs"]
            or row["source_hash"] != data["source_hash"]
            or row["compiler_identity_sha256"]
            != json_digest(data["compiler_identity"])
            or row["policy"] not in plan["timing_policies"]
            or row["blocksize"] != reference["geometry"]["blocksize"]
            or row["dynshared"] != reference["geometry"]["dynshared"]
        ):
            raise ValueError(
                "Original timing join differs from profile workload"
            )
        timings.append(dict(receipt=receipt, observation=row))
    return data, dict(
        path=str(path),
        sha256=digest(path),
        sample_path=sample["artifact"],
        sample_sha256=digest(sample["artifact"]),
        inputs_sha256=digest(data["input_artifact"]),
        counter_channels=data["counter_columns"],
        counter_totals=[int(value) for value in counter_totals],
        reference_state_sha256=sample["arrays"]["reference_state"]["sha256"],
        counter_array_sha256=sample["arrays"]["iteration_counters"]["sha256"],
        original_uninstrumented_timings=timings,
    )


def validate_launch(profile, metrics, data):
    """Bind the first reference solve to exported launch/resources."""
    request = json.loads((profile / "request.json").read_text())
    command = json.loads((profile / "command.json").read_text())
    if (
        request["action"] != "profile"
        or request["target"] != "script"
        or Path(request["script"]).resolve()
        != SCRIPT.with_name("counter_probe.py")
    ):
        raise ValueError("Expected the reviewed counter_probe profile target")
    if request["launch_skip"] != 0 or request["launch_count"] != 1:
        raise ValueError("Only the first reference solve may be profiled")
    if request["sha256"].lower() != command["source_sha256"].lower():
        raise ValueError("Requested and executed probe source digests differ")
    if digest(request["script"]) != request["sha256"].lower():
        raise ValueError("Current probe source cannot verify recorded order")
    arguments = request["arguments"]
    options = [value for value in arguments if value.startswith("--")]
    if len(options) != len(set(options)) or arguments.count("--execute") != 1:
        raise ValueError("Duplicate options or missing execution flag")
    expected_command = [
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--kernel-name-base",
        "function",
        "--kernel-name",
        request["kernel_filter"],
        "--launch-skip",
        str(request["launch_skip"]),
        "--launch-count",
        str(request["launch_count"]),
    ]
    for section in request["sections"]:
        expected_command += ["--section", section]
    if request["metrics"]:
        expected_command += ["--metrics", ",".join(request["metrics"])]
    expected_command += [
        "--csv",
        "--log-file",
        str(profile / "diagnostic.csv"),
        "--export",
        str(profile / "profile"),
        sys.executable,
        request["script"],
    ]
    expected_command += arguments + [
        request["output_flag"],
        str(profile / "benchmark"),
    ]
    if command["arguments"] != expected_command:
        raise ValueError(
            "Executed NCU/Python/script arguments differ from request"
        )
    if command["working_directory"] != request["_runtime_tree"] or (
        command["pythonpath"] != request["_pythonpath"]
    ):
        raise ValueError("Executed source/runtime import roots differ")
    expected = {
        "--bank": data["cohort"]["bank"],
        "--system": data["cohort"]["manifest"]["system"],
        "--algo": data["cohort"]["manifest"]["algo"],
        "--policy": data["plan"]["policy"],
        "--cohort": data["cohort"]["manifest"]["cohort"],
    }
    if any(
        arguments[arguments.index(key) + 1] != value
        for key, value in expected.items()
    ):
        raise ValueError("Profile request and completed benchmark differ")
    reference = data["compiled"]["reference"]
    geometry = reference["geometry"]
    fixed = {
        "launch__block_dim_x": ("block", reference["threads_per_run"]),
        "launch__block_dim_y": (
            "block",
            geometry["blocksize"] // reference["threads_per_run"],
        ),
        "launch__block_dim_z": ("block", 1),
        "launch__block_size": ("", geometry["blocksize"]),
        "launch__grid_dim_x": (
            "",
            data["n_runs"]
            // (geometry["blocksize"] // reference["threads_per_run"]),
        ),
        "launch__grid_dim_y": ("", 1),
        "launch__grid_dim_z": ("", 1),
        "launch__thread_count": (
            "thread",
            data["n_runs"] * reference["threads_per_run"],
        ),
        "launch__registers_per_thread": (
            "register/thread",
            reference["registers_per_thread"],
        ),
        "launch__shared_mem_per_block_dynamic": (
            "byte/block",
            geometry["dynshared"],
        ),
        "launch__sm_count": ("SM", reference["multiprocessor_count"]),
        "launch__occupancy_limit_registers": (
            "block",
            geometry["blocks_per_sm"],
        ),
    }
    for name, (unit, value) in fixed.items():
        if quantity(metrics, name, unit) != value:
            raise ValueError(f"Profile launch differs at {name}")
    if metrics["CC"]["raw"] != "8.9":
        raise ValueError("Profile architecture differs")
    return dict(
        source_order="counter_probe executes reference before instrumented",
        request_sha256=digest(profile / "request.json"),
        command_sha256=digest(profile / "command.json"),
        profile_report_sha256=digest(profile / "profile.ncu-rep"),
        source_sha256=command["source_sha256"].lower(),
        exact_geometry_fields=list(fixed),
        rounded_waves_metric=metrics["launch__waves_per_multiprocessor"],
        exact_waves_from_compiled_geometry=geometry["waves"],
    )


def aggregate(rows, metrics):
    """Aggregate frequency and opcode work without a hotness threshold."""
    exact = Counter()
    sampled = Counter()
    opcodes = defaultdict(
        lambda: dict(
            static_addresses=0,
            executed_addresses=0,
            exact=Counter(),
            sampled=Counter(),
        )
    )
    frequencies = Counter()
    for row in rows:
        exact.update(row["exact"])
        sampled.update(row["sampled"])
        work = row["exact"]["Instructions Executed"]
        if (
            row["exact"]["Predicated-On Thread Instructions Executed"]
            > row["exact"]["Thread Instructions Executed"]
        ):
            raise ValueError("Predicated-on work exceeds active-thread work")
        if row["exact"]["Thread Instructions Executed"] > 32 * work:
            raise ValueError(
                "Thread work exceeds 32 lanes per warp instruction"
            )
        frequencies[work] += 1
        item = opcodes[row["opcode"]]
        item["static_addresses"] += 1
        item["executed_addresses"] += int(work > 0)
        item["exact"].update(row["exact"])
        item["sampled"].update(row["sampled"])
    for column, metric in EXACT_COLUMNS.items():
        if quantity(metrics, metric, "inst") != exact[column]:
            raise ValueError(f"Source instruction total differs from {metric}")
    curve = []
    addresses = work_sum = 0
    for frequency, count in sorted(frequencies.items(), reverse=True):
        addresses += count
        work_sum += count * frequency
        curve.append(
            dict(
                warp_executions_per_address=frequency,
                address_count=count,
                encoded_bytes=count * 16,
                warp_instruction_work=count * frequency,
                cumulative_addresses=addresses,
                cumulative_bytes=addresses * 16,
                cumulative_warp_instruction_work=work_sum,
            )
        )
    hardware = integer(
        str(quantity(metrics, "smsp__inst_executed.sum", "inst"))
    )
    sample_global = integer(
        str(quantity(metrics, "smsp__pcsamp_sample_count", ""))
    )
    return dict(
        exact_totals=dict(exact),
        sampled_totals=dict(sampled),
        opcode_work=dict(sorted(opcodes.items())),
        frequency_rank_curve=curve,
        footprint=dict(
            encoding_bytes_per_instruction=16,
            whole_addresses=len(rows),
            whole_encoded_bytes=len(rows) * 16,
            executed_addresses=len(rows) - frequencies[0],
            executed_encoded_bytes=(len(rows) - frequencies[0]) * 16,
            zero_execution_addresses=frequencies[0],
            interpretation=(
                "Global distinct-PC union; not temporal reuse distance "
                "or per-SM hot working set"
            ),
        ),
        reconciliation=dict(
            source_matches_software_metrics=True,
            hardware_warp_instructions=int(hardware),
            hardware_minus_source=int(hardware)
            - exact["Instructions Executed"],
            hardware_difference_attribution=(
                "Unresolved; no correction applied"
            ),
            global_samples=int(sample_global),
            global_minus_source_samples=int(sample_global)
            - sampled["# Samples"],
            sample_difference_attribution=(
                "Unresolved; source and aggregate sample counts "
                "retained separately"
            ),
        ),
    )


def analyze(args):
    """Validate one completed profile and write reusable CPU-only evidence."""
    profile = args.profile_dir.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    exports = validate_exports(profile, args.source_counters, output)
    metrics = read_metrics(profile / "metrics.csv")
    data, labels = validate_labels(profile)
    launch = validate_launch(profile, metrics, data)
    source_name, rows = source_rows(args.source_counters)
    reference = data["compiled"]["reference"]
    native, symbols = disassemble(
        Path(reference["cubin_path"]),
        args.nvdisasm,
        output,
        reference["entry_name"],
    )
    base = match_native(rows, native, symbols)
    result = aggregate(rows, metrics)
    result.update(
        schema_version=1,
        kind="saved_solver_profile_analysis",
        status="ok",
        tool_sha256=digest(SCRIPT),
        profile_directory=str(profile),
        profile_source_counters=str(args.source_counters.resolve()),
        source_counters_sha256=digest(args.source_counters),
        metrics_sha256=digest(profile / "metrics.csv"),
        policy=data["plan"]["policy"],
        algo=data["cohort"]["manifest"]["algo"],
        duration=data["duration"],
        n_runs=data["n_runs"],
        cubin_sha256=reference["cubin_sha256"],
        config_hash=reference["config_hash"],
        source_hash=data["source_hash"],
        compiler_identity=data["compiler_identity"],
        reference_resources=reference,
        labels=labels,
        launch_identity=launch,
        report_export_identity=exports,
        source_kernel_name=source_name,
        runtime_base_address=base,
        native_match=(
            "Every PC/opcode/operand agrees after symbol relocation, "
            "comma normalization, NCU label annotation removal and "
            "removal of display-omitted .reuse"
        ),
        metrics=metrics,
        limitations=[
            "Exact source execution counters and periodic sampled stalls "
            "are separate observables.",
            "Sampled counts do not estimate cycles by multiplication "
            "or predict cache miss cost.",
            "Distinct executed bytes aggregate the entire grid and solve; "
            "no temporal order is observed.",
            "Frequency ranks describe repeated address execution, "
            "not fetch bytes or physical cache traffic.",
            "Profile duration is diagnostic; original uninstrumented "
            "bank samples retain their timing role.",
            "Iteration counters are separate instrumented labels, "
            "not pre-compile model inputs.",
            "Hardware/source and aggregate/source sample residuals "
            "are not silently corrected.",
        ],
    )
    with (output / "per_pc.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(output / "analysis.json", result)
    print(
        json.dumps(
            dict(
                status="ok",
                output=str(output),
                footprint=result["footprint"],
                exact=result["exact_totals"],
                reconciliation=result["reconciliation"],
            )
        )
    )


def main():
    """Analyze saved source exports without invoking CUDA or the profiler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--source-counters", type=Path, required=True)
    parser.add_argument("--nvdisasm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
