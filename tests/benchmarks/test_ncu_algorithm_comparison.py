"""Tests for the persistent Nsight Compute comparison harness."""

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import ncu_algorithm_comparison as comparison  # noqa: E402


def test_selected_values_expands_all_and_one():
    assert comparison.selected_values(
        "all", comparison.PROBLEMS
    ) == comparison.PROBLEMS
    assert comparison.selected_values(
        "lorenz", comparison.PROBLEMS
    ) == ("lorenz",)


def test_selected_values_rejects_unknown_value():
    with pytest.raises(ValueError, match="received 'other'"):
        comparison.selected_values("other", comparison.PROBLEMS)


def test_ncu_command_profiles_exactly_four_hot_launches(tmp_path):
    command = comparison.ncu_command(
        "python",
        "lorenz",
        "mlir",
        tmp_path,
        50.0,
        0.05,
        1024,
    )

    count_index = command.index("--launch-count")
    assert command[count_index + 1] == "4"
    assert command[command.index("--profile-from-start") + 1] == "off"
    for section in comparison.SECTIONS:
        assert section in command


def test_worker_command_contains_no_ncu_cli_options(tmp_path):
    command = comparison.worker_command(
        "python",
        "very-stiff",
        "numba-cuda",
        tmp_path,
        50.0,
        0.05,
        1024,
    )

    assert command[0] == "python"
    assert command[1] == str(comparison.WORKER)
    assert command[command.index("--problem") + 1] == "very-stiff"
    assert command[command.index("--backend") + 1] == "numba-cuda"
    assert "--export" not in command
    assert "--section" not in command
    assert "--import" not in command


def test_parse_args_rejects_no_ncu_with_reuse_existing(capsys):
    with pytest.raises(SystemExit):
        comparison.parse_args(("--no-ncu", "--reuse-existing"))

    assert (
        "--no-ncu cannot be combined"
        in capsys.readouterr().err
    )


def test_executable_command_resolves_python():
    command = comparison.executable_command(
        (sys.executable, "-c", "print('ok')")
    )

    assert Path(command[0]).resolve() == Path(sys.executable).resolve()


def test_native_command_bypasses_batch_shell(tmp_path):
    batch = tmp_path / "ncu.bat"
    batch.touch()
    native = (
        tmp_path
        / "target"
        / "windows-desktop-win7-x64"
        / "ncu.exe"
    )
    native.parent.mkdir(parents=True)
    native.touch()
    output_path = r"C:\profile & data\100% complete"

    command = comparison._native_command(
        batch, ("--output", output_path)
    )

    assert command == [str(native), "--output", output_path]


def test_parse_raw_metrics_maps_launch_order():
    text = "\n".join(
        (
            "==PROF== report",
            '"ID","Metric Name","Metric Value"',
            '"7","gpu__time_duration.sum","1.0"',
            '"7","launch__registers_per_thread","20"',
            '"9","gpu__time_duration.sum","2.0"',
            '"11","gpu__time_duration.sum","3.0"',
            '"13","gpu__time_duration.sum","4.0"',
        )
    )

    metrics = comparison.parse_raw_metrics(text)

    assert metrics["tsit5"]["gpu__time_duration.sum"] == "1.0"
    assert metrics["kvaerno3"]["gpu__time_duration.sum"] == "2.0"
    assert metrics["radau"]["gpu__time_duration.sum"] == "3.0"
    assert metrics["ode23s"]["gpu__time_duration.sum"] == "4.0"


def test_parse_raw_metrics_requires_four_launches():
    text = "\n".join(
        (
            '"ID","Metric Name","Metric Value"',
            '"7","gpu__time_duration.sum","1.0"',
        )
    )

    with pytest.raises(ValueError, match="expected 4"):
        comparison.parse_raw_metrics(text)


def test_parse_wide_raw_metrics_maps_rows():
    text = "\n".join(
        (
            '"ID","gpu__time_duration.sum","launch__registers_per_thread"',
            '"","nsecond","register/thread"',
            '"0","1.0","20"',
            '"1","2.0","30"',
            '"2","3.0","40"',
            '"3","4.0","50"',
        )
    )

    metrics = comparison.parse_raw_metrics(text)

    assert metrics["tsit5"]["launch__registers_per_thread"] == "20"
    assert metrics["ode23s"]["gpu__time_duration.sum"] == "4.0"


def test_summary_exposes_all_stalls_and_code_metrics():
    metric_names = {
        metric_name
        for _, metric_name in comparison.PAIR_METRICS
    }

    assert len(comparison.STALL_METRICS) == 18
    assert len({
        metric_name for _, metric_name in comparison.STALL_METRICS
    }) == 18
    assert "derived__smsp__inst_executed_op_branch_pct" in metric_names
    assert "sass__size" in metric_names


def test_pair_comparison_renders_every_stall():
    record = {
        "n": 32,
        "kernel_ms": 1.0,
        "ns_per_trajectory": 1.0,
        "newton_per_trajectory": 1.0,
        "krylov_per_trajectory": 1.0,
        "attempted_per_trajectory": 1.0,
        "accepted_per_trajectory": 1.0,
        "rejected_per_trajectory": 0.0,
    }
    manifest = {
        "algorithms": {
            name: record for name in comparison.ALGORITHMS
        },
    }
    metric_values = {
        metric_name: "1"
        for _, metric_name in comparison.PAIR_METRICS
        if not metric_name.startswith("__")
    }
    metrics = {
        name: metric_values for name in comparison.ALGORITHMS
    }

    rendered = comparison.comparison_markdown(
        "lorenz", "mlir", manifest, metrics
    )

    for label, _ in comparison.STALL_METRICS:
        assert f"| {label} |" in rendered
