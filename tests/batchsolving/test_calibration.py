"""Tests for the staged solver calibration tournament."""

import pytest

from cubie.batchsolving.calibration import (
    FAMILY_ORDER_PANELS,
    FAMILY_REPRESENTATIVES,
    CalibrationResult,
    CandidateResult,
    CandidateSpec,
    complete_apply_settings,
    erk_stage_specs,
    order_stage_specs,
    preconditioner_stage_specs,
    solver_stage_specs,
    toggle_stage_specs,
    _alias_is_adaptive,
    _load_calibration_file,
    _screen_rungs,
    _supports_prediction,
    _supports_smoothing,
    _write_calibration_file,
)


class TestPanelConstruction:
    """Structural pruning of the candidate panels."""

    def test_dirk_preconditioner_panel_covers_all_pairs(self):
        specs = preconditioner_stage_specs("dirk", "kvaerno3")
        pairs = {
            (
                spec.settings_dict["preconditioner_type"],
                spec.settings_dict["preconditioner_order"],
            )
            for spec in specs
        }
        assert pairs == {
            ("jacobi", 0),
            ("jacobi", 1),
            ("jacobi", 2),
            ("neumann", 2),
            ("none", 0),
        }
        for spec in specs:
            assert (
                spec.settings_dict["linear_correction_type"]
                == "bicgstab"
            )

    def test_firk_preconditioner_panel_prunes_series_orders(self):
        specs = preconditioner_stage_specs("firk", "radau_iia_5")
        pairs = {
            (
                spec.settings_dict["preconditioner_type"],
                spec.settings_dict["preconditioner_order"],
            )
            for spec in specs
        }
        assert pairs == {
            ("jacobi", 0),
            ("neumann", 2),
            ("none", 0),
        }

    def test_mass_matrix_panel_prunes_neumann(self):
        specs = preconditioner_stage_specs(
            "dirk", "kvaerno3", has_mass=True
        )
        pairs = {
            (
                spec.settings_dict["preconditioner_type"],
                spec.settings_dict["preconditioner_order"],
            )
            for spec in specs
        }
        assert pairs == {
            ("jacobi", 0),
            ("jacobi", 1),
            ("jacobi", 2),
            ("none", 0),
        }

    def test_neumann_winner_materialises_series_order(self):
        spec = CandidateSpec(
            label="bicgstab neumann-2",
            family="dirk",
            algorithm="kvaerno3",
            settings=(
                ("linear_correction_type", "bicgstab"),
                ("preconditioner_type", "neumann"),
                ("preconditioner_order", 2),
            ),
        )
        updates = complete_apply_settings(spec)
        assert updates["preconditioner_type"] == "neumann"
        assert updates["preconditioner_order"] == 2

    def test_rosenbrock_solver_panel_is_correction_types_only(self):
        specs = solver_stage_specs(
            "rosenbrock", "ros3p", ("jacobi", 1)
        )
        settings = [spec.settings_dict for spec in specs]
        assert settings == [
            {
                "linear_correction_type": "bicgstab",
                "preconditioner_type": "jacobi",
                "preconditioner_order": 1,
            },
            {
                "linear_correction_type": "minimal_residual",
                "preconditioner_type": "jacobi",
                "preconditioner_order": 1,
            },
            {"linear_correction_type": "lu"},
        ]

    def test_dirk_solver_panel_crosses_newton_variants(self):
        specs = solver_stage_specs(
            "dirk", "kvaerno3", ("none", 0)
        )
        settings = [spec.settings_dict for spec in specs]
        iterative = {
            "preconditioner_type": "none",
            "preconditioner_order": 0,
        }
        assert settings == [
            {"linear_correction_type": "bicgstab", **iterative},
            {
                "linear_correction_type": "bicgstab",
                "inexact_newton": True,
                **iterative,
            },
            {
                "linear_correction_type": "minimal_residual",
                **iterative,
            },
            {
                "linear_correction_type": "minimal_residual",
                "inexact_newton": True,
                **iterative,
            },
            {"linear_correction_type": "lu"},
            {
                "linear_correction_type": "lu",
                "inexact_newton": True,
                "prefactored": False,
            },
            {
                "linear_correction_type": "lu",
                "inexact_newton": True,
                "prefactored": True,
            },
        ]

    def test_firk_solver_panel_has_single_frozen_direct_solve(self):
        specs = solver_stage_specs(
            "firk", "radau_iia_5", ("jacobi", 0)
        )
        lu_settings = [
            spec.settings_dict
            for spec in specs
            if spec.settings_dict["linear_correction_type"] == "lu"
        ]
        assert lu_settings == [
            {"linear_correction_type": "lu"},
            {
                "linear_correction_type": "lu",
                "inexact_newton": True,
                "prefactored": True,
            },
        ]

    def test_erk_panel_is_adaptive_only(self):
        specs = erk_stage_specs()
        assert len(specs) == len(FAMILY_ORDER_PANELS["erk"])
        for spec in specs:
            assert _alias_is_adaptive(spec.algorithm)
            assert spec.settings == ()

    def test_toggle_panel_crosses_supported_toggles(self):
        base = (("linear_correction_type", "lu"),)
        specs = toggle_stage_specs("firk", "radau_iia_5", base)
        combos = {
            (
                spec.settings_dict["use_smoothed_error"],
                spec.settings_dict["attempt_dense_prediction"],
            )
            for spec in specs
        }
        assert combos == {
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        }
        for spec in specs:
            assert (
                spec.settings_dict["linear_correction_type"] == "lu"
            )

    def test_toggle_panel_rosenbrock_smoothing_only(self):
        specs = toggle_stage_specs("rosenbrock", "ros3p", ())
        settings = [spec.settings_dict for spec in specs]
        assert settings == [
            {"use_smoothed_error": True},
            {"use_smoothed_error": False},
        ]

    def test_order_panel_leads_with_incumbent(self):
        winner = (
            ("linear_correction_type", "bicgstab"),
            ("preconditioner_type", "jacobi"),
            ("preconditioner_order", 0),
        )
        specs = order_stage_specs("firk", winner, "radau_iia_5")
        assert specs[0].algorithm == "radau_iia_5"
        aliases = [spec.algorithm for spec in specs]
        for alias in FAMILY_ORDER_PANELS["firk"]:
            if _alias_is_adaptive(alias):
                assert alias in aliases
        for spec in specs:
            assert spec.settings_dict[
                "linear_correction_type"
            ] == "bicgstab"

    def test_order_panel_regates_toggles_per_tableau(self):
        winner = (
            ("linear_correction_type", "lu"),
            ("use_smoothed_error", True),
            ("attempt_dense_prediction", True),
        )
        specs = order_stage_specs(
            "dirk", winner, FAMILY_REPRESENTATIVES["dirk"]
        )
        for spec in specs:
            settings = spec.settings_dict
            if "use_smoothed_error" in settings:
                assert _supports_smoothing(spec.algorithm)
            if "attempt_dense_prediction" in settings:
                assert _supports_prediction(spec.algorithm, "dirk")


class TestApplySettings:
    """Materialisation of the winner's settings for application."""

    def test_erk_winner_applies_algorithm_only(self):
        spec = CandidateSpec(
            label="tsit5", family="erk", algorithm="tsit5"
        )
        assert complete_apply_settings(spec) == {
            "algorithm": "tsit5"
        }

    def test_lu_winner_materialises_newton_axes(self):
        spec = CandidateSpec(
            label="lu prefactored",
            family="dirk",
            algorithm="kvaerno3",
            settings=(
                ("linear_correction_type", "lu"),
                ("inexact_newton", True),
                ("prefactored", True),
            ),
        )
        updates = complete_apply_settings(spec)
        assert updates["algorithm"] == "kvaerno3"
        assert updates["linear_correction_type"] == "lu"
        assert updates["inexact_newton"] is True
        assert updates["prefactored"] is True
        assert updates["attempt_dense_prediction"] is True

    def test_iterative_winner_materialises_preconditioner(self):
        spec = CandidateSpec(
            label="bicgstab jacobi-1",
            family="firk",
            algorithm="radau_iia_5",
            settings=(
                ("linear_correction_type", "bicgstab"),
                ("preconditioner_type", "jacobi"),
                ("preconditioner_order", 1),
            ),
        )
        updates = complete_apply_settings(spec)
        assert updates["preconditioner_type"] == "jacobi"
        assert updates["preconditioner_order"] == 1
        assert updates["inexact_newton"] is False
        assert updates["prefactored"] is True
        assert updates["use_smoothed_error"] is True


class TestScreenRungs:
    """Screening-ladder construction."""

    def test_rungs_ascend_probe_then_screen(self):
        rungs = _screen_rungs({}, 16.0, 4.0, 0.25)
        assert rungs == ((1.0, 0.25), (4.0, 1.0))

    def test_rungs_collapse_when_intervals_clamp(self):
        rungs = _screen_rungs({"save_every": 8.0}, 16.0, 0.0, 0.25)
        assert rungs == ((8.0, 0.0),)


class TestResultFile:
    """Markdown persistence round-trip."""

    def test_report_round_trips_through_file(self, tmp_path):
        spec_a = CandidateSpec(
            label="tsit5", family="erk", algorithm="tsit5"
        )
        spec_b = CandidateSpec(
            label="kvaerno3 lu",
            family="dirk",
            algorithm="kvaerno3",
            settings=(
                ("linear_correction_type", "lu"),
                ("inexact_newton", True),
            ),
        )
        result_a = CandidateResult(
            spec=spec_a,
            stage="final",
            times_ms=(4.25, 4.5),
            screen_ms=0.5,
            failures=0,
            runs=64,
        )
        result_b = CandidateResult(
            spec=spec_b,
            stage="dirk:solver",
            runs=64,
            failures=64,
            dropped=True,
            reason="screen failures 64/64",
        )
        report = CalibrationResult(
            candidates=[result_a, result_b],
            winner=result_a,
            equivalent=[result_a],
            features={"n_states": 3, "spectral_radius": 12.5},
            applied_settings={"algorithm": "tsit5"},
        )
        conditions = {"fn_hash": "abcdef012345", "duration": 1.0}
        path = tmp_path / "calibration_test.md"
        _write_calibration_file(path, report, conditions)

        loaded = _load_calibration_file(path)
        assert loaded is not None
        assert loaded.winner.spec.key == spec_a.key
        assert loaded.winner.times_ms == (4.25, 4.5)
        assert loaded.winner.screen_ms == 0.5
        assert [r.spec.key for r in loaded.equivalent] == [
            spec_a.key
        ]
        assert loaded.candidates[1].dropped
        assert loaded.candidates[1].reason == "screen failures 64/64"
        assert (
            loaded.candidates[1].spec.settings
            == spec_b.settings
        )
        assert loaded.features["n_states"] == 3
        assert loaded.applied_settings == {}


class TestCalibrateGuards:
    """Input validation on the shared three-state system."""

    @pytest.mark.nocudasim
    def test_calibrate_requires_drivers_for_driver_systems(
        self, solver_mutable
    ):
        with pytest.raises(ValueError, match="drivers"):
            solver_mutable.calibrate(
                {"x0": [0.5], "x1": [-0.25], "x2": [1.2]},
                {"p0": [0.7], "p1": [0.9], "p2": [1.1]},
                duration=0.2,
                families=["erk"],
                verbose=False,
            )
