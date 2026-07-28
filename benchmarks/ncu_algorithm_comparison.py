#!/usr/bin/env python
"""Profile CuBIE adaptive algorithms with NVIDIA Nsight Compute.

The public mode launches one Nsight Compute process for every selected
problem/backend pair. Each process compiles the selected algorithms
(Tsit5, Kvaerno3, Radau IIA 5, and Rosenbrock23 (``ode23s``) by
default), estimates the trajectories required to fill the requested
number of occupancy-limited CUDA waves (ten by default), warms every
kernel, and then profiles exactly one hot launch per algorithm with the
``full`` metric set.

Profiled kernels compile with ``lineinfo`` on and LTO off: the MLIR
backend's LTO link strips line tables, and per-line source attribution
is the point of this harness. ``--import-source yes`` embeds the
correlated Python source (including the generated system module) in the
report at capture time.

Examples
--------
Profile the complete two-problem, two-backend matrix::

    python benchmarks/ncu_algorithm_comparison.py --problem all \
        --backend all

Profile one combination, one algorithm::

    python benchmarks/ncu_algorithm_comparison.py --problem lorenz \
        --backend mlir --algorithm radau

Run a combination directly for attachment from the NCU GUI::

    python benchmarks/ncu_algorithm_comparison.py --problem lorenz \
        --backend mlir --no-ncu

NCU GUI mode
------------
Set the GUI's application to the ``--no-ncu`` command above (one
problem/backend pair per session) and enable three settings:

- child-process profiling (the workers are subprocesses),
- "Profile from start" off — the worker brackets the hot launches with
  ``cuda.profile_start()``/``cuda.profile_stop()`` after compilation,
  sizing, and warmup,
- "Import Source" yes, for per-line attribution in the Source page.

Results are written below ``generated/ncu_algorithm_comparison`` by
default. Every NCU-captured combination produces a report, raw-metric
CSV, correlated source/SASS dump, worker manifest, and Markdown
comparison. ``--no-ncu`` writes the worker manifests only.
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

DEFAULT_WAVES = 10
ALGORITHMS = ("tsit5", "kvaerno3", "radau", "ode23s")
PROBLEMS = ("lorenz", "very-stiff")
BACKENDS = ("numba-cuda", "mlir")

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


def run_prefix(
    problem: str,
    backend: str,
    algorithms: Sequence[str],
) -> str:
    """Return the output-file prefix for one profiled combination."""

    if tuple(algorithms) == ALGORITHMS:
        return f"{problem}_{backend}"
    return f"{problem}_{backend}_" + "-".join(algorithms)


def worker_command(
    python: str,
    problem: str,
    backend: str,
    output_dir: Path,
    waves: int,
    algorithms: Sequence[str],
    prefix: str,
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
        "--waves",
        str(waves),
        "--algorithms",
        ",".join(algorithms),
        "--prefix",
        prefix,
    ]


def ncu_command(
    python: str,
    problem: str,
    backend: str,
    output_dir: Path,
    waves: int,
    algorithms: Sequence[str],
    prefix: str,
) -> list[str]:
    """Return the NCU command for one problem/backend combination."""

    command = [
        "ncu",
        "--force-overwrite",
        "--export",
        str(output_dir / prefix),
        "--profile-from-start",
        "off",
        "--kernel-name",
        "regex:integration_kernel",
        "--launch-count",
        str(len(algorithms)),
        "--set",
        "full",
        "--import-source",
        "yes",
    ]
    command.extend(
        worker_command(
            python,
            problem,
            backend,
            output_dir,
            waves,
            algorithms,
            prefix,
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
    algorithms: Sequence[str],
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

    algorithms = manifest["launch_order"]
    lines = [
        f"# NCU comparison: {problem} / {backend}",
        "",
        "| Metric | " + " | ".join(algorithms) + " |",
        "|---|" + "|".join("---:" for _ in algorithms) + "|",
    ]
    records = manifest["algorithms"]
    native_rows = (
        ("Trajectories", "n"),
        ("Target occupancy waves", "waves"),
        ("Estimated grid blocks", "grid_blocks"),
        (
            "Resident blocks / SM",
            "blocks_per_multiprocessor",
        ),
        ("Trajectories / block", "trajectories_per_block"),
        ("Native kernel (ms)", "kernel_ms"),
        ("Native ns / trajectory", "ns_per_trajectory"),
        ("Newton iterations / trajectory", "newton_per_trajectory"),
        ("Krylov iterations / trajectory", "krylov_per_trajectory"),
        ("Attempted steps / trajectory", "attempted_per_trajectory"),
        ("Accepted steps / trajectory", "accepted_per_trajectory"),
        ("Rejected steps / trajectory", "rejected_per_trajectory"),
    )
    for label, key in native_rows:
        values = [
            _format_value(records[name].get(key)) for name in algorithms
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    for label, metric_name in PAIR_METRICS:
        values = [
            _format_value(_metric_value(metrics[name], metric_name))
            for name in algorithms
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend(
        (
            "",
            "Branch counts are dynamic and should be compared together with "
            "the trajectory count and accepted-step count. The correlated "
            "source/SASS dump beside this file interleaves the Python "
            "source with the generated code and per-instruction source "
            "counters.",
            "",
        )
    )
    return "\n".join(lines)


def matrix_summary_markdown(
    output_dir: Path,
    entries: Sequence[tuple[str, str, str]],
) -> str:
    """Return a cross-problem/backend comparison table."""

    columns = (
        "problem",
        "backend",
        "algorithm",
        "n",
        "waves",
        "grid blocks",
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
    for problem, backend, prefix in entries:
        manifest = json.loads(
            (
                output_dir / f"{prefix}_manifest.json"
            ).read_text(encoding="utf-8")
        )
        raw = (
            output_dir / f"{prefix}_raw.csv"
        ).read_text(encoding="utf-8")
        algorithms = manifest["launch_order"]
        metrics = parse_raw_metrics(raw, algorithms)
        for algorithm in algorithms:
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
                record.get("waves"),
                record.get("grid_blocks"),
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
    for problem, backend, prefix in entries:
        manifest = json.loads(
            (
                output_dir / f"{prefix}_manifest.json"
            ).read_text(encoding="utf-8")
        )
        raw = (
            output_dir / f"{prefix}_raw.csv"
        ).read_text(encoding="utf-8")
        algorithms = manifest["launch_order"]
        metrics = parse_raw_metrics(raw, algorithms)
        for algorithm in algorithms:
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


def import_report(
    report: Path,
    output_dir: Path,
    problem: str,
    backend: str,
) -> None:
    """Export raw CSV, correlated source/SASS, and the comparison."""

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
            "cuda,sass",
        )),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (output_dir / f"{prefix}_source_sass.txt").write_text(
        source, encoding="utf-8"
    )
    manifest_path = output_dir / f"{prefix}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = parse_raw_metrics(raw, manifest["launch_order"])
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
    algorithms = selected_values(args.algorithm, ALGORITHMS)
    entries = []
    for problem in problems:
        for backend in backends:
            prefix = run_prefix(problem, backend, algorithms)
            entries.append((problem, backend, prefix))
            command_builder = (
                worker_command if args.no_ncu else ncu_command
            )
            command = command_builder(
                sys.executable,
                problem,
                backend,
                output_dir,
                args.waves,
                algorithms,
                prefix,
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
                import_report(
                    output_dir / f"{prefix}.ncu-rep",
                    output_dir,
                    problem,
                    backend,
                )
    if args.no_ncu:
        return
    summary = matrix_summary_markdown(output_dir, entries)
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
        "--algorithm",
        default="all",
        choices=("all", *ALGORITHMS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--waves",
        type=int,
        default=DEFAULT_WAVES,
        help=(
            "occupancy-limited CUDA waves per algorithm "
            f"(default: {DEFAULT_WAVES})"
        ),
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
    if args.waves < 1:
        parser.error("--waves must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the selected profiler matrix."""

    run_matrix(parse_args(argv))


if __name__ == "__main__":
    main()
