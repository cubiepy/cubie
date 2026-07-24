#!/usr/bin/env python
"""Profile CuBIE adaptive algorithms with NVIDIA Nsight Compute.

The public mode launches one Nsight Compute process for every selected
problem/backend pair.  Each process calibrates a separate trajectory count
for Tsit5, Kvaerno3, Radau IIA 5, and Rosenbrock23 (``ode23s``), warms all
four kernels, and then profiles exactly one hot launch per algorithm.

Trajectory counts double until either the kernel reaches ``--target-ms`` or
normalised kernel time stops improving.  The latter is the throughput floor:
two consecutive doublings whose milliseconds per trajectory improve by less
than ``--floor-improvement``.

Examples
--------
Profile the complete two-problem, two-backend matrix::

    python benchmarks/ncu_algorithm_comparison.py --problem all \
        --backend all

Profile one combination::

    python benchmarks/ncu_algorithm_comparison.py --problem lorenz \
        --backend mlir

Run a combination directly for attachment from the NCU GUI::

    python benchmarks/ncu_algorithm_comparison.py --problem lorenz \
        --backend mlir --no-ncu

Results are written below ``generated/ncu_algorithm_comparison`` by default.
Every NCU-captured combination produces a report, raw-metric CSV,
SASS/source dump, worker manifest, and Markdown comparison. ``--no-ncu``
writes the worker manifests only; configure the GUI to profile child
processes.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "benchmarks" / "ncu_algorithm_worker.py"
DEFAULT_OUTPUT = REPO_ROOT / "generated" / "ncu_algorithm_comparison"

ALGORITHMS = ("tsit5", "kvaerno3", "radau", "ode23s")
PROBLEMS = ("lorenz", "very-stiff")
BACKENDS = ("numba-cuda", "mlir")
SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SpeedOfLight",
    "SchedulerStats",
    "WarpStateStats",
    "SourceCounters",
    "InstructionStats",
)

SUMMARY_METRICS = (
    ("NCU duration (ms)", "gpu__time_duration.sum"),
    (
        "Achieved occupancy (%)",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ),
    ("Theoretical occupancy (%)", "__theoretical_occupancy__"),
    ("Registers / thread", "launch__registers_per_thread"),
    ("Static SASS size (bytes)", "sass__size"),
    ("Branch instructions", "smsp__inst_executed_op_branch.sum"),
    (
        "Branch instructions (%)",
        "derived__smsp__inst_executed_op_branch_pct",
    ),
    (
        "Branch efficiency (%)",
        "smsp__sass_average_branch_targets_threads_uniform.pct",
    ),
    (
        "Divergent branch targets",
        "smsp__sass_branch_targets_threads_divergent.avg",
    ),
    (
        "Active threads / warp",
        "smsp__thread_inst_executed_per_inst_executed.ratio",
    ),
    (
        "Predicated-on threads / warp",
        "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio",
    ),
    (
        "Warp cycles / issued instruction",
        "smsp__average_warp_latency_per_inst_issued.ratio",
    ),
)

STALL_METRICS = (
    (
        "Stall barrier",
        "smsp__average_warps_issue_stalled_barrier"
        "_per_issue_active.ratio",
    ),
    (
        "Stall branch resolving",
        "smsp__average_warps_issue_stalled_branch_resolving"
        "_per_issue_active.ratio",
    ),
    (
        "Stall dispatch",
        "smsp__average_warps_issue_stalled_dispatch_stall"
        "_per_issue_active.ratio",
    ),
    (
        "Stall drain",
        "smsp__average_warps_issue_stalled_drain"
        "_per_issue_active.ratio",
    ),
    (
        "Stall IMC miss",
        "smsp__average_warps_issue_stalled_imc_miss"
        "_per_issue_active.ratio",
    ),
    (
        "Stall LG throttle",
        "smsp__average_warps_issue_stalled_lg_throttle"
        "_per_issue_active.ratio",
    ),
    (
        "Stall long scoreboard",
        "smsp__average_warps_issue_stalled_long_scoreboard"
        "_per_issue_active.ratio",
    ),
    (
        "Stall math pipe throttle",
        "smsp__average_warps_issue_stalled_math_pipe_throttle"
        "_per_issue_active.ratio",
    ),
    (
        "Stall memory barrier",
        "smsp__average_warps_issue_stalled_membar"
        "_per_issue_active.ratio",
    ),
    (
        "Stall MIO throttle",
        "smsp__average_warps_issue_stalled_mio_throttle"
        "_per_issue_active.ratio",
    ),
    (
        "Stall miscellaneous",
        "smsp__average_warps_issue_stalled_misc"
        "_per_issue_active.ratio",
    ),
    (
        "Stall no instruction",
        "smsp__average_warps_issue_stalled_no_instruction"
        "_per_issue_active.ratio",
    ),
    (
        "Stall not selected",
        "smsp__average_warps_issue_stalled_not_selected"
        "_per_issue_active.ratio",
    ),
    (
        "Stall selected",
        "smsp__average_warps_issue_stalled_selected"
        "_per_issue_active.ratio",
    ),
    (
        "Stall short scoreboard",
        "smsp__average_warps_issue_stalled_short_scoreboard"
        "_per_issue_active.ratio",
    ),
    (
        "Stall sleeping",
        "smsp__average_warps_issue_stalled_sleeping"
        "_per_issue_active.ratio",
    ),
    (
        "Stall texture throttle",
        "smsp__average_warps_issue_stalled_tex_throttle"
        "_per_issue_active.ratio",
    ),
    (
        "Stall wait",
        "smsp__average_warps_issue_stalled_wait"
        "_per_issue_active.ratio",
    ),
)

PAIR_METRICS = SUMMARY_METRICS + STALL_METRICS


def _native_command(
    executable: Path,
    arguments: Sequence[str],
) -> list[str]:
    """Return argv using NCU's native executable, never its batch shell."""

    if executable.suffix.lower() != ".bat":
        return [str(executable), *arguments]
    native = (
        executable.parent
        / "target"
        / "windows-desktop-win7-x64"
        / f"{executable.stem}.exe"
    )
    if not native.is_file():
        raise FileNotFoundError(
            f"{executable} is a batch launcher, but its native "
            f"executable was not found at {native}"
        )
    return [str(native), *arguments]


def executable_command(command: Sequence[str]) -> list[str]:
    """Return a directly executable argv list."""

    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(f"could not find executable {command[0]!r}")
    return _native_command(Path(executable), command[1:])


def selected_values(value: str, values: Sequence[str]) -> tuple[str, ...]:
    """Expand ``all`` or return one validated selection."""

    if value == "all":
        return tuple(values)
    if value not in values:
        allowed = ", ".join(("all", *values))
        raise ValueError(f"expected one of {allowed}; received {value!r}")
    return (value,)


def worker_command(
    python: str,
    problem: str,
    backend: str,
    output_dir: Path,
    target_ms: float,
    floor_improvement: float,
    max_n: int,
) -> list[str]:
    """Return the direct worker command for one problem/backend pair."""

    return [
        python,
        str(WORKER),
        "--problem",
        problem,
        "--backend",
        backend,
        "--output-dir",
        str(output_dir),
        "--target-ms",
        str(target_ms),
        "--floor-improvement",
        str(floor_improvement),
        "--max-n",
        str(max_n),
    ]


def ncu_command(
    python: str,
    problem: str,
    backend: str,
    output_dir: Path,
    target_ms: float,
    floor_improvement: float,
    max_n: int,
) -> list[str]:
    """Return the NCU command for one problem/backend combination."""

    report_base = output_dir / f"{problem}_{backend}"
    command = [
        "ncu",
        "--force-overwrite",
        "--export",
        str(report_base),
        "--profile-from-start",
        "off",
        "--kernel-name",
        "regex:integration_kernel",
        "--launch-count",
        str(len(ALGORITHMS)),
        "--import-source",
        "yes",
    ]
    for section in SECTIONS:
        command.extend(("--section", section))
    command.extend(
        worker_command(
            python,
            problem,
            backend,
            output_dir,
            target_ms,
            floor_improvement,
            max_n,
        )
    )
    return command


def _csv_rows(text: str) -> list[dict[str, str]]:
    """Return metric rows from NCU's raw CSV console output."""

    rows = list(csv.reader(text.splitlines()))
    header_index = None
    for index, row in enumerate(rows):
        if "ID" in row and (
            "Metric Name" in row or "gpu__time_duration.sum" in row
        ):
            header_index = index
            break
    if header_index is None:
        raise ValueError("NCU raw output has no metric header")
    header = rows[header_index]
    parsed = []
    for row in rows[header_index + 1:]:
        if len(row) != len(header):
            continue
        parsed_row = dict(zip(header, row))
        if parsed_row.get("ID", ""):
            parsed.append(parsed_row)
    return parsed


def parse_raw_metrics(
    text: str,
    algorithms: Sequence[str] = ALGORITHMS,
) -> dict[str, dict[str, str]]:
    """Map NCU raw metrics to algorithms in launch order."""

    rows = _csv_rows(text)
    if "Metric Name" not in rows[0]:
        if len(rows) != len(algorithms):
            raise ValueError(
                f"expected {len(algorithms)} profiled launches, "
                f"found {len(rows)}"
            )
        return {
            algorithm: row
            for algorithm, row in zip(algorithms, rows)
        }
    id_column = "ID"
    if id_column not in rows[0]:
        matches = [
            key for key in rows[0]
            if key.lower() in {"id", "launch id", "result id"}
        ]
        if not matches:
            raise ValueError("NCU raw output has no launch identifier")
        id_column = matches[0]
    launch_ids = []
    for row in rows:
        launch_id = row[id_column]
        if launch_id not in launch_ids:
            launch_ids.append(launch_id)
    if len(launch_ids) != len(algorithms):
        raise ValueError(
            f"expected {len(algorithms)} profiled launches, "
            f"found {len(launch_ids)}"
        )
    algorithm_for = dict(zip(launch_ids, algorithms))
    metrics = {algorithm: {} for algorithm in algorithms}
    for row in rows:
        algorithm = algorithm_for[row[id_column]]
        metrics[algorithm][row["Metric Name"]] = row["Metric Value"]
    return metrics


def _format_value(value: object) -> str:
    """Format a manifest or NCU table value."""

    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _numeric(value: str) -> float:
    """Return an NCU numeric value written with grouping commas."""

    return float(value.replace(",", ""))


def _metric_value(
    algorithm_metrics: dict[str, str],
    metric_name: str,
) -> object:
    """Return one display-scaled or derived NCU metric."""

    if metric_name == "gpu__time_duration.sum":
        value = algorithm_metrics.get(metric_name)
        return None if value is None else _numeric(value) / 1e6
    if metric_name != "__theoretical_occupancy__":
        return algorithm_metrics.get(metric_name)
    limit_names = (
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
    )
    try:
        resident_blocks = min(
            _numeric(algorithm_metrics[name])
            for name in limit_names
        )
        block_size = _numeric(
            algorithm_metrics["launch__block_size"]
        )
        max_threads = _numeric(
            algorithm_metrics[
                "device__attribute_max_threads_per_multiprocessor"
            ]
        )
    except KeyError:
        return None
    return 100.0 * resident_blocks * block_size / max_threads


def comparison_markdown(
    problem: str,
    backend: str,
    manifest: dict[str, object],
    metrics: dict[str, dict[str, str]],
) -> str:
    """Return a concise runtime, work, occupancy, and stalls table."""

    lines = [
        f"# NCU comparison: {problem} / {backend}",
        "",
        "| Metric | " + " | ".join(ALGORITHMS) + " |",
        "|---|" + "|".join("---:" for _ in ALGORITHMS) + "|",
    ]
    records = manifest["algorithms"]
    native_rows = (
        ("Trajectories", "n"),
        ("Native kernel (ms)", "kernel_ms"),
        ("Native ns / trajectory", "ns_per_trajectory"),
        ("Newton iterations / trajectory", "newton_per_trajectory"),
        ("Krylov iterations / trajectory", "krylov_per_trajectory"),
        ("Attempted steps / trajectory", "attempted_per_trajectory"),
        ("Accepted steps / trajectory", "accepted_per_trajectory"),
        ("Rejected steps / trajectory", "rejected_per_trajectory"),
    )
    for label, key in native_rows:
        values = [_format_value(records[name].get(key)) for name in ALGORITHMS]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    for label, metric_name in PAIR_METRICS:
        values = [
            _format_value(_metric_value(metrics[name], metric_name))
            for name in ALGORITHMS
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend(
        (
            "",
            "Branch counts are dynamic and should be compared together with "
            "the trajectory count and accepted-step count. The SASS dump "
            "beside this file contains the static generated code and "
            "per-instruction source counters.",
            "",
        )
    )
    return "\n".join(lines)


def matrix_summary_markdown(
    output_dir: Path,
    pairs: Sequence[tuple[str, str]],
) -> str:
    """Return a cross-problem/backend comparison table."""

    columns = (
        "problem",
        "backend",
        "algorithm",
        "n",
        "native ms",
        "NCU ms",
        "occupancy %",
        "registers",
        "branch eff. %",
        "branch inst. %",
        "branches / trajectory",
        "SASS bytes",
        "branch stall",
        "wait stall",
        "warp latency",
        "Newton / trajectory",
        "Krylov / trajectory",
        "attempted / trajectory",
        "accepted / trajectory",
    )
    lines = [
        "# NCU algorithm comparison matrix",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for problem, backend in pairs:
        prefix = f"{problem}_{backend}"
        manifest = json.loads(
            (
                output_dir / f"{prefix}_manifest.json"
            ).read_text(encoding="utf-8")
        )
        raw = (
            output_dir / f"{prefix}_raw.csv"
        ).read_text(encoding="utf-8")
        metrics = parse_raw_metrics(raw)
        for algorithm in ALGORITHMS:
            record = manifest["algorithms"][algorithm]
            n = record["n"]
            algorithm_metrics = metrics[algorithm]
            branches = _numeric(
                algorithm_metrics[
                    "smsp__inst_executed_op_branch.sum"
                ]
            )
            values = (
                problem,
                backend,
                algorithm,
                n,
                record["kernel_ms"],
                _metric_value(
                    algorithm_metrics, "gpu__time_duration.sum"
                ),
                algorithm_metrics.get(
                    "sm__warps_active.avg."
                    "pct_of_peak_sustained_active"
                ),
                algorithm_metrics.get(
                    "launch__registers_per_thread"
                ),
                algorithm_metrics.get(
                    "smsp__sass_average_branch_targets_threads_"
                    "uniform.pct"
                ),
                algorithm_metrics.get(
                    "derived__smsp__inst_executed_op_branch_pct"
                ),
                branches / n,
                algorithm_metrics.get("sass__size"),
                algorithm_metrics.get(
                    "smsp__average_warps_issue_stalled_branch_"
                    "resolving_per_issue_active.ratio"
                ),
                algorithm_metrics.get(
                    "smsp__average_warps_issue_stalled_wait_"
                    "per_issue_active.ratio"
                ),
                algorithm_metrics.get(
                    "smsp__average_warp_latency_per_inst_issued.ratio"
                ),
                record.get("newton_per_trajectory"),
                record.get("krylov_per_trajectory"),
                record.get("attempted_per_trajectory"),
                record.get("accepted_per_trajectory"),
            )
            lines.append(
                "| " + " | ".join(
                    _format_value(value) for value in values
                ) + " |"
            )
    stall_columns = (
        "problem",
        "backend",
        "algorithm",
        *(label.removeprefix("Stall ") for label, _ in STALL_METRICS),
    )
    lines.extend(
        (
            "",
            "## Warp stalls",
            "",
            "| " + " | ".join(stall_columns) + " |",
            "|" + "|".join("---" for _ in stall_columns) + "|",
        )
    )
    for problem, backend in pairs:
        prefix = f"{problem}_{backend}"
        raw = (
            output_dir / f"{prefix}_raw.csv"
        ).read_text(encoding="utf-8")
        metrics = parse_raw_metrics(raw)
        for algorithm in ALGORITHMS:
            values = (
                problem,
                backend,
                algorithm,
                *(
                    metrics[algorithm].get(metric_name)
                    for _, metric_name in STALL_METRICS
                ),
            )
            lines.append(
                "| " + " | ".join(
                    _format_value(value) for value in values
                ) + " |"
            )
    lines.append("")
    return "\n".join(lines)


def import_report(report: Path, output_dir: Path) -> None:
    """Export raw CSV, SASS/source text, and the Markdown comparison."""

    prefix = report.stem
    raw = subprocess.run(
        executable_command((
            "ncu",
            "--import",
            str(report),
            "--csv",
            "--page",
            "raw",
            "--print-units",
            "base",
        )),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    raw_path = output_dir / f"{prefix}_raw.csv"
    raw_path.write_text(raw, encoding="utf-8")
    source = subprocess.run(
        executable_command((
            "ncu",
            "--import",
            str(report),
            "--page",
            "source",
            "--print-source",
            "sass",
        )),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (output_dir / f"{prefix}_sass.txt").write_text(
        source, encoding="utf-8"
    )
    problem, backend = prefix.rsplit("_", 1)
    manifest_path = output_dir / f"{prefix}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = parse_raw_metrics(raw)
    comparison = comparison_markdown(
        problem, backend, manifest, metrics
    )
    (output_dir / f"{prefix}_comparison.md").write_text(
        comparison, encoding="utf-8"
    )


def run_matrix(args: argparse.Namespace) -> None:
    """Run selected workers, with or without NCU CLI capture."""

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = selected_values(args.problem, PROBLEMS)
    backends = selected_values(args.backend, BACKENDS)
    pairs = []
    for problem in problems:
        for backend in backends:
            pairs.append((problem, backend))
            report = output_dir / f"{problem}_{backend}.ncu-rep"
            manifest = output_dir / (
                f"{problem}_{backend}_manifest.json"
            )
            if (
                not args.no_ncu
                and args.reuse_existing
                and report.exists()
                and manifest.exists()
            ):
                print(
                    f"\n=== reusing {problem} / {backend} ===",
                    flush=True,
                )
                import_report(report, output_dir)
                continue
            command_builder = (
                worker_command if args.no_ncu else ncu_command
            )
            command = command_builder(
                sys.executable,
                problem,
                backend,
                output_dir,
                args.target_ms,
                args.floor_improvement,
                args.max_n,
            )
            environment = os.environ.copy()
            environment["CUBIE_CUDA_BACKEND"] = backend
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(REPO_ROOT / "src"), str(REPO_ROOT))
            )
            action = "running" if args.no_ncu else "profiling"
            print(
                f"\n=== {action} {problem} / {backend} ===",
                flush=True,
            )
            subprocess.run(
                executable_command(command),
                check=True,
                cwd=REPO_ROOT,
                env=environment,
            )
            if not args.no_ncu:
                import_report(report, output_dir)
    if args.no_ncu:
        return
    summary = matrix_summary_markdown(output_dir, pairs)
    (output_dir / "matrix_summary.md").write_text(
        summary, encoding="utf-8"
    )


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse the public profiler CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        default="all",
        choices=("all", *PROBLEMS),
    )
    parser.add_argument(
        "--backend",
        default="all",
        choices=("all", *BACKENDS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--target-ms", type=float, default=50.0)
    parser.add_argument(
        "--floor-improvement",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=2**27,
        help=(
            "hard failure guard for trajectory tuning; increase it if "
            "neither the target nor throughput floor is reached"
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Import complete existing reports and profile only missing pairs.",
    )
    parser.add_argument(
        "--no-ncu",
        action="store_true",
        help=(
            "run workers directly without NCU CLI capture or report "
            "import; intended for NCU GUI child-process profiling"
        ),
    )
    args = parser.parse_args(argv)
    if args.target_ms <= 0:
        parser.error("--target-ms must be positive")
    if not 0 < args.floor_improvement < 1:
        parser.error("--floor-improvement must be between 0 and 1")
    if args.max_n < 32:
        parser.error("--max-n must be at least 32")
    if args.no_ncu and args.reuse_existing:
        parser.error("--no-ncu cannot be combined with --reuse-existing")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the selected profiler matrix."""

    run_matrix(parse_args(argv))


if __name__ == "__main__":
    main()
