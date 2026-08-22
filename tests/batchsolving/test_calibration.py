"""Tests for the solver-configuration calibration race."""

import numpy as np
import pytest

from cubie.batchsolving.calibration import (
    FAMILY_ORDERS,
    FAMILY_REPRESENTATIVES,
    CandidateSpec,
    erk_specs,
    error_option_specs,
    linear_solver_specs,
    order_specs,
    preconditioner_specs,
    _trial_durations,
)


class TestCandidateSpecs:
    """Construction of the candidate configuration lists."""

    def test_dirk_preconditioners_cover_all_pairs(self):
        specs = preconditioner_specs("dirk", "kvaerno3")
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
            ("neumann", 1),
            ("neumann", 2),
            ("none", 0),
        }
        for spec in specs:
            assert (
                spec.settings_dict["linear_correction_type"]
                == "bicgstab"
            )
            assert spec.settings_dict["inexact_newton"] is False

    def test_rosenbrock_preconditioners_carry_no_newton_flag(self):
        specs = preconditioner_specs("rosenbrock", "ros3p")
        settings = [spec.settings_dict for spec in specs]
        assert settings == [
            {
                "linear_correction_type": "bicgstab",
                "preconditioner_type": p_type,
                "preconditioner_order": p_order,
            }
            for p_type, p_order in (
                ("jacobi", 0),
                ("jacobi", 1),
                ("jacobi", 2),
                ("neumann", 1),
                ("neumann", 2),
                ("none", 0),
            )
        ]

    def test_rosenbrock_solvers_are_correction_types_only(self):
        specs = linear_solver_specs(
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

    def test_dirk_solvers_cross_newton_variants_explicitly(self):
        specs = linear_solver_specs(
            "dirk", "kvaerno3", ("none", 0)
        )
        settings = [spec.settings_dict for spec in specs]
        iterative = {
            "preconditioner_type": "none",
            "preconditioner_order": 0,
        }
        assert settings == [
            {
                "linear_correction_type": "bicgstab",
                "inexact_newton": False,
                **iterative,
            },
            {
                "linear_correction_type": "bicgstab",
                "inexact_newton": True,
                **iterative,
            },
            {
                "linear_correction_type": "minimal_residual",
                "inexact_newton": False,
                **iterative,
            },
            {
                "linear_correction_type": "minimal_residual",
                "inexact_newton": True,
                **iterative,
            },
            {
                "linear_correction_type": "lu",
                "inexact_newton": False,
            },
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

    def test_firk_solvers_have_single_frozen_direct_solve(self):
        specs = linear_solver_specs(
            "firk", "radau_iia_5", ("jacobi", 0)
        )
        lu_settings = [
            spec.settings_dict
            for spec in specs
            if spec.settings_dict["linear_correction_type"] == "lu"
        ]
        assert lu_settings == [
            {
                "linear_correction_type": "lu",
                "inexact_newton": False,
            },
            {
                "linear_correction_type": "lu",
                "inexact_newton": True,
                "prefactored": True,
            },
        ]

    def test_winning_preconditioner_shares_bicgstab_key(self):
        precond = preconditioner_specs("dirk", "kvaerno3")
        solvers = linear_solver_specs(
            "dirk", "kvaerno3", ("jacobi", 1)
        )
        winning = next(
            spec
            for spec in precond
            if spec.settings_dict["preconditioner_type"] == "jacobi"
            and spec.settings_dict["preconditioner_order"] == 1
        )
        repeated = next(
            spec
            for spec in solvers
            if spec.settings_dict["linear_correction_type"]
            == "bicgstab"
            and spec.settings_dict["inexact_newton"] is False
        )
        assert winning.key == repeated.key

    def test_erk_specs_cover_the_order_list(self):
        specs = erk_specs()
        assert [spec.algorithm for spec in specs] == list(
            FAMILY_ORDERS["erk"]
        )
        for spec in specs:
            assert spec.settings == ()

    def test_firk_error_options_cross_both_toggles(self):
        base = (
            ("linear_correction_type", "lu"),
            ("inexact_newton", False),
        )
        specs = error_option_specs(
            "firk", "radau_iia_5", base, np.float32
        )
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

    def test_rosenbrock_error_options_smoothing_only(self):
        specs = error_option_specs(
            "rosenbrock", "ros3p", (), np.float32
        )
        settings = [spec.settings_dict for spec in specs]
        assert settings == [
            {"use_smoothed_error": True},
            {"use_smoothed_error": False},
        ]

    def test_order_specs_lead_with_incumbent(self):
        winner = (
            ("linear_correction_type", "bicgstab"),
            ("inexact_newton", False),
            ("preconditioner_type", "jacobi"),
            ("preconditioner_order", 0),
        )
        specs = order_specs("firk", winner, "radau_iia_5")
        assert specs[0].algorithm == "radau_iia_5"
        aliases = [spec.algorithm for spec in specs]
        for alias in FAMILY_ORDERS["firk"]:
            assert alias in aliases
        for spec in specs:
            assert spec.settings == winner

    def test_incumbent_order_spec_shares_the_winning_key(self):
        winner = CandidateSpec(
            label="kvaerno3 lu prefactored",
            family="dirk",
            algorithm="kvaerno3",
            settings=(
                ("linear_correction_type", "lu"),
                ("inexact_newton", True),
                ("prefactored", True),
            ),
        )
        specs = order_specs(
            "dirk",
            winner.settings,
            FAMILY_REPRESENTATIVES["dirk"],
        )
        assert specs[0].key == winner.key


class TestTrialDurations:
    """Ascending trial-length construction."""

    def test_trials_ascend_short_then_long(self):
        trials = _trial_durations({}, 16.0, 4.0)
        assert trials == ((0.0625, 0.015625), (1.0, 0.25))

    def test_trials_collapse_when_intervals_clamp(self):
        trials = _trial_durations({"save_every": 8.0}, 16.0, 0.0)
        assert trials == ((8.0, 0.0),)


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
                verbose=False,
            )
