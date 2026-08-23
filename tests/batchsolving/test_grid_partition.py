"""Solve-time swept/folded partitioning from batch grids."""

import numpy as np
import pytest

from cubie.batchsolving.solver import solve_ivp

PARTITION_SETTINGS = {
    "algorithm": "euler",
    "step_controller": "fixed",
    "dt": 0.01,
    "save_every": 0.05,
    "output_types": ["state", "time"],
}


@pytest.mark.parametrize(
    "solver_settings_override", [PARTITION_SETTINGS], indirect=True
)
class TestGridPartition:
    """Dict grids drive the swept/folded partition per solve."""

    def test_uniform_column_folds(
        self, solver_mutable, system, driver_settings
    ):
        solver_mutable.solve(
            {"x0": [0.1, 0.2]},
            {"p0": [0.5, 0.9], "p1": np.ones(2)},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        )
        assert list(system.parameters.names) == ["p0"]
        folded = system.constants.values_dict
        assert folded["p1"] == 1.0
        assert "p2" in folded

    def test_scalar_entry_folds(
        self, solver_mutable, system, driver_settings
    ):
        solver_mutable.solve(
            {"x0": [0.1, 0.2]},
            {"p0": [0.5, 0.9], "p1": 4.0},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        )
        assert list(system.parameters.names) == ["p0"]
        assert system.constants.values_dict["p1"] == 4.0

    def test_swept_set_follows_each_solve(
        self, solver_mutable, system, driver_settings
    ):
        solver_mutable.solve(
            {"x0": [0.1, 0.2]},
            {"p0": [0.5, 0.9]},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        )
        assert list(system.parameters.names) == ["p0"]
        solver_mutable.solve(
            {"x0": [0.1, 0.2]},
            {"p1": [0.5, 0.9], "p2": [0.7, 0.8]},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        )
        assert list(system.parameters.names) == ["p1", "p2"]

    def test_all_uniform_verbatim_keeps_run_count(
        self, solver_mutable, system, driver_settings
    ):
        result = solver_mutable.solve(
            None,
            {"p0": [0.7, 0.7, 0.7]},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        )
        assert list(system.parameters.names) == []
        assert solver_mutable.num_runs == 3
        assert result.time_domain_array.shape[-1] == 3

    def test_swept_params_conflict_raises(
        self, solver_mutable, driver_settings
    ):
        with pytest.raises(ValueError, match="swept_params"):
            solver_mutable.solve(
                {"x0": [0.1]},
                {"p1": [0.5, 0.9]},
                drivers=driver_settings,
                duration=0.05,
                grid_type="verbatim",
                swept_params=["p0"],
            )

    def test_swept_params_keeps_uniform_column_live(
        self, solver_mutable, system, driver_settings
    ):
        solver_mutable.solve(
            {"x0": [0.1, 0.2]},
            {"p0": np.ones(2)},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
            swept_params=["p0", "p1"],
        )
        assert sorted(system.parameters.names) == ["p0", "p1"]
        assert solver_mutable.swept_params == ("p0", "p1")

    def test_clearing_swept_params_returns_to_auto(
        self, solver_mutable, system, driver_settings
    ):
        solver_mutable.set_swept_params(["p0", "p1"])
        solver_mutable.set_swept_params(None)
        assert solver_mutable.swept_params is None
        solver_mutable.solve(
            {"x0": [0.1, 0.2]},
            {"p1": [0.5, 0.9]},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        )
        assert list(system.parameters.names) == ["p1"]

    def test_array_input_binds_current_layout(
        self, solver_mutable, system, precision, driver_settings
    ):
        solver_mutable.set_swept_params(["p0", "p2"])
        params = np.array(
            [[0.5, 0.9], [0.7, 0.7]], dtype=precision
        )
        inits = np.tile(
            system.initial_values.values_array[:, None], (1, 2)
        ).astype(precision)
        solver_mutable.solve(
            inits,
            params,
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        )
        # Arrays never repartition: the uniform row stays live.
        assert sorted(system.parameters.names) == ["p0", "p2"]

    def test_array_without_swept_layout_raises(
        self, solver_mutable, system, precision, driver_settings
    ):
        solver_mutable.solve(
            {"x0": [0.1]},
            {"p0": 0.5, "p1": 0.9, "p2": 1.1},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        )
        assert list(system.parameters.names) == []
        params = np.array([[0.5, 0.9]], dtype=precision)
        with pytest.raises(ValueError, match="set_swept_params"):
            solver_mutable.solve(
                np.tile(
                    system.initial_values.values_array[:, None],
                    (1, 2),
                ).astype(precision),
                params,
                drivers=driver_settings,
                duration=0.05,
                grid_type="verbatim",
            )

    def test_folded_solve_matches_live_solve(
        self, solver_mutable, fresh_solver_factory, system,
        system_restored, driver_settings,
    ):
        folded = solver_mutable.solve(
            {"x0": [0.1, 0.2]},
            {"p0": [0.5, 0.9], "p1": np.full(2, 0.9)},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        ).time_domain_array.copy()

        live = solver_mutable.solve(
            {"x0": [0.1, 0.2]},
            {"p0": [0.5, 0.9], "p1": [0.9, 0.90001]},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        ).time_domain_array

        # Run 0 is identical between the folded and live solves.
        np.testing.assert_allclose(
            folded[..., 0],
            live[..., 0],
            rtol=5e-5,
            atol=5e-6,
        )

    def test_update_reaches_folded_value(
        self, solver_mutable, system, system_restored, driver_settings
    ):
        solver_mutable.solve(
            {"x0": [0.1, 0.2]},
            {"p0": [0.5, 0.9]},
            drivers=driver_settings,
            duration=0.05,
            grid_type="verbatim",
        )
        assert "p1" in system.constants.values_dict
        solver_mutable.update({"p1": 7.5})
        assert system.constants.values_dict["p1"] == pytest.approx(
            7.5
        )

    def test_combinatorial_uniform_column_run_count(
        self, solver_mutable, system, driver_settings
    ):
        result = solver_mutable.solve(
            {"x0": [0.1, 0.2]},
            {"p0": [0.5, 0.9], "p1": [3.0, 3.0]},
            drivers=driver_settings,
            duration=0.05,
            grid_type="combinatorial",
        )
        # Duplicate values deduplicate in combinatorial grids, so
        # folding the uniform column leaves the run count unchanged.
        assert solver_mutable.num_runs == 4
        assert system.constants.values_dict["p1"] == 3.0
        assert result is not None


@pytest.mark.parametrize(
    "solver_settings_override", [PARTITION_SETTINGS], indirect=True
)
def test_solve_ivp_partitions_dict_grid(
    system_restored, precision, driver_settings
):
    result = solve_ivp(
        system_restored,
        y0={"x0": [0.1, 0.2]},
        parameters={"p0": [0.5, 0.9], "p1": np.ones(2)},
        drivers=driver_settings,
        method="euler",
        dt=0.01,
        save_every=0.05,
        duration=0.05,
        grid_type="verbatim",
    )
    assert list(system_restored.parameters.names) == ["p0"]
    assert result is not None
