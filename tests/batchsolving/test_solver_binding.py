"""The Solver binds its system to the parameters each batch sweeps."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cubie._cudasim_extensions import cuda


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
    assert solver_mutable.fixed_parameter_values[names[1]] == 0.5
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
    swept = solver_mutable.swept_parameters
    assert swept == tuple(sorted(simple_parameters))
    assert params.shape[0] == len(swept)
    _solve(solver_mutable, inits, params, driver_settings)
    assert solver_mutable.swept_parameters == swept


def test_compile_binds_like_solve(
    solver_mutable, system_restored, simple_parameters, driver_settings
):
    """compile takes the parameters a solve would and binds the same."""
    solver_mutable.compile(
        parameters=simple_parameters,
        grid_type="verbatim",
        drivers=driver_settings,
    )
    compiled = solver_mutable.swept_parameters
    compiled_kernel = solver_mutable.kernel.kernel
    _solve(
        solver_mutable,
        None,
        simple_parameters,
        driver_settings,
        grid_type="verbatim",
    )
    assert solver_mutable.swept_parameters == compiled
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
    assert solver_mutable.swept_parameters == tuple(names)

    fixed = _solve(
        solver_mutable,
        None,
        params,
        driver_settings,
        grid_type="verbatim",
        fix_constant_parameters=True,
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
    solver_mutable.set_swept_parameters([])
    first = _solve(solver_mutable, None, None, driver_settings)
    value = float(system_restored.parameters.values_dict[names[0]])
    solver_mutable.update({names[0]: 2.0 * value + 1.0})
    second = _solve(solver_mutable, None, None, driver_settings)
    assert solver_mutable.fixed_parameter_values[names[0]] == (
        pytest.approx(2.0 * value + 1.0)
    )
    assert not np.array_equal(first, second)


def test_set_swept_parameters_orders_the_table_rows(
    solver_mutable, system_restored, driver_settings
):
    """Swept-height arrays follow set_swept_parameters' order."""
    names = list(system_restored.parameters.names)
    grid = {names[0]: [0.5, 1.0, 1.5], names[1]: [0.2, 0.4, 0.6]}
    from_dict = _solve(
        solver_mutable, None, grid, driver_settings, grid_type="verbatim"
    )
    solver_mutable.set_swept_parameters([names[1], names[0]])
    assert solver_mutable.swept_parameters == (names[1], names[0])
    compiled_kernel = solver_mutable.kernel.kernel
    table = np.array(
        [grid[names[1]], grid[names[0]]], dtype=system_restored.precision
    )
    from_array = _solve(solver_mutable, None, table, driver_settings)
    assert solver_mutable.swept_parameters == (names[1], names[0])
    assert solver_mutable.kernel.kernel is compiled_kernel
    assert_array_equal(from_array, from_dict)

    inits = np.tile(
        system_restored.initial_values.values_array[:, np.newaxis],
        (1, 3),
    ).astype(system_restored.precision)
    from_device = _solve(
        solver_mutable,
        cuda.to_device(inits),
        cuda.to_device(table),
        driver_settings,
    )
    assert solver_mutable.swept_parameters == (names[1], names[0])
    assert_array_equal(from_device, from_dict)


def test_no_parameters_fix_every_parameter(
    solver_mutable, system_restored, driver_settings
):
    """A solve without parameters compiles every parameter in."""
    names = list(system_restored.parameters.names)
    solver_mutable.set_swept_parameters([names[0]])
    _solve(solver_mutable, None, None, driver_settings)
    assert solver_mutable.swept_parameters == ()


def test_solve_rejects_parameter_values_as_options(
    solver_mutable, system_restored, driver_settings
):
    """Parameter values go in the parameters input, not options."""
    name = system_restored.parameters.names[0]
    with pytest.raises(ValueError, match="parameters input"):
        _solve(
            solver_mutable, None, None, driver_settings, **{name: 1.0}
        )


def test_single_values_become_stored_values(
    solver_mutable, system_restored, driver_settings
):
    """Values given once are the system's stored values after a solve."""
    system = system_restored
    names = list(system.parameters.names)
    state = system.initial_values.names[0]
    _solve(
        solver_mutable,
        {state: 0.25},
        {names[0]: [1.0, 2.0], names[1]: 0.75},
        driver_settings,
        grid_type="verbatim",
    )
    assert system.parameters.values_dict[names[1]] == 0.75
    assert system.initial_values.values_dict[state] == 0.25
    solver_mutable.set_swept_parameters([names[0]])
    assert solver_mutable.fixed_parameter_values[names[1]] == 0.75
