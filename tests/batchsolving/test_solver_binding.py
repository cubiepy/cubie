"""The Solver binds its system to the parameters each batch sweeps."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cubie.odesystems.ODEData import ParameterBinding


def _solve(solver, inits, params, driver_settings, **kwargs):
    """Run a short solve and return its time-domain array."""
    return solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=0.05,
        **kwargs,
    ).time_domain_array.copy()


def test_varying_dict_rows_are_swept(
    solver_mutable, system_restored, driver_settings
):
    """A dict sweep binds its varying names; the rest are fixed."""
    system = system_restored
    names = list(system.parameters.names)
    _solve(
        solver_mutable,
        None,
        {names[0]: [1.0, 2.0], names[1]: [0.5, 0.5]},
        driver_settings,
        grid_type="verbatim",
    )
    assert solver_mutable.swept_parameters == (names[0],)
    assert solver_mutable.binding.fixed_values[names[1]] == 0.5
    assert_array_equal(solver_mutable.parameters, [[1.0, 2.0]])
    assert not solver_mutable.kernel.system_config_stale


def test_uniform_verbatim_grid_keeps_its_runs(
    solver_mutable, system_restored, driver_settings
):
    """Every parameter fixed still solves one run per grid column."""
    names = list(system_restored.parameters.names)
    result = _solve(
        solver_mutable,
        None,
        {names[0]: [1.0, 1.0, 1.0]},
        driver_settings,
        grid_type="verbatim",
    )
    assert solver_mutable.swept_parameters == ()
    assert result.shape[2] == 3


def test_build_grid_binds_the_solver(
    solver_mutable, system_restored, simple_parameters, driver_settings
):
    """build_grid binds the system; its arrays solve as built."""
    inits, params = solver_mutable.build_grid(
        None, simple_parameters, grid_type="verbatim"
    )
    binding = solver_mutable.binding
    assert binding.swept == tuple(sorted(simple_parameters))
    assert params.shape[0] == len(binding.swept)
    _solve(solver_mutable, inits, params, driver_settings)
    assert solver_mutable.binding == binding


def test_compile_binds_like_solve(
    solver_mutable, system_restored, simple_parameters, driver_settings
):
    """compile takes the parameters a solve would and binds the same."""
    solver_mutable.compile(
        parameters=simple_parameters,
        grid_type="verbatim",
        drivers=driver_settings,
    )
    compiled = solver_mutable.binding
    compiled_kernel = solver_mutable.kernel.kernel
    _solve(
        solver_mutable,
        None,
        simple_parameters,
        driver_settings,
        grid_type="verbatim",
    )
    assert solver_mutable.binding == compiled
    assert solver_mutable.kernel.kernel is compiled_kernel


def test_found_constant_rows_match_swept_rows(
    solver_mutable, system_restored, driver_settings, tolerance
):
    """Fixing uniform rows solves the same batch as sweeping them."""
    system = system_restored
    names = list(system.parameters.names)
    params = np.tile(
        system.parameters.values_array[:, np.newaxis], (1, 3)
    )
    params[0] = params[0] * np.array([0.5, 1.0, 1.5])
    swept = _solve(
        solver_mutable, None, params, driver_settings, grid_type="verbatim"
    )
    assert solver_mutable.binding == ParameterBinding(swept=names)

    fixed = _solve(
        solver_mutable,
        None,
        params,
        driver_settings,
        grid_type="verbatim",
        find_constant_params=True,
    )
    assert solver_mutable.swept_parameters == (names[0],)
    assert_allclose(
        fixed, swept, rtol=tolerance.rel_tight, atol=tolerance.abs_tight
    )


def test_parameter_value_update_recompiles_fixed_value(
    solver_mutable, system_restored, driver_settings
):
    """A new value for a fixed parameter reaches the next solve."""
    names = list(system_restored.parameters.names)
    first = _solve(solver_mutable, None, None, driver_settings)
    value = float(system_restored.parameters.values_dict[names[0]])
    solver_mutable.update({names[0]: 2.0 * value + 1.0})
    second = _solve(solver_mutable, None, None, driver_settings)
    assert solver_mutable.binding.fixed_values[names[0]] == (
        pytest.approx(2.0 * value + 1.0)
    )
    assert not np.array_equal(first, second)
