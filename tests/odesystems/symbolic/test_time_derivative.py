import numpy as np
import pytest
from cubie.odesystems.solver_helpers import SolverHelperRequest
from cubie.cuda_simsafe import cuda

from cubie.odesystems.symbolic.codegen.time_derivative import (
    generate_time_derivative_fac_code,
)
from cubie.odesystems.symbolic.symbolicODE import SymbolicODE


def test_time_derivative_factory_structure(simple_equations, indexed_bases):
    """Generated factory should expose the expected CUDA signature."""

    code = generate_time_derivative_fac_code(simple_equations, indexed_bases)
    assert "def time_derivative_rhs(" in code
    assert "driver_dt" in code
    assert "return time_derivative_rhs" in code


@pytest.fixture(scope="session")
def time_derivative_system(precision):
    """Return a symbolic system with explicit time and driver dependence."""

    dxdt = [
        "aux = t * drive",
        "dx = aux + drive",
    ]
    system = SymbolicODE.create(
        dxdt=dxdt,
        states={"x": precision(0.0)},
        drivers={"drive": precision(0.0)},
        precision=precision,
        strict=True,
        name="time_derivative_system",
    )
    return system

@pytest.mark.parametrize("time", [0.0, 1.0, 2.0, 10.0])
def test_time_derivative_helper_matches_reference(time_derivative_system,
                                                  time,
                                                  precision):
    """Helper should compute ∂ₜF + Σ∂₍driver₎F·driver_dt."""

    del_t = time_derivative_system.get_solver_helper(
        SolverHelperRequest(role="time_derivative_rhs")
    ).device_function
    numba_precision = time_derivative_system.numba_precision
    state_len = time_derivative_system.sizes.states
    driver_len = time_derivative_system.num_drivers
    out_len = state_len
    param_len = 1
    obs_len = 1

    @cuda.jit
    def kernel(time_value, driver_value, driver_rate, out_array):
        state = cuda.local.array(state_len, numba_precision)
        parameters = cuda.local.array(param_len, numba_precision)
        drivers = cuda.local.array(driver_len, numba_precision)
        driver_dt = cuda.local.array(driver_len, numba_precision)
        observables = cuda.local.array(obs_len, numba_precision)
        result = cuda.local.array(out_len, numba_precision)

        if driver_len > 0:
            drivers[0] = driver_value[0]
            driver_dt[0] = driver_rate[0]

        del_t(
            state,
            parameters,
            drivers,
            driver_dt,
            observables,
            result,
            time_value[0],
        )
        for idx in range(out_len):
            out_array[idx] = result[idx]

    time_host = np.array([precision(time)], dtype=precision)
    driver_host = np.array([precision(time**2)], dtype=precision)
    driver_rate_host = np.array([precision(2*time)], dtype=precision)
    out_host = np.zeros(out_len, dtype=precision)

    time_dev = cuda.to_device(time_host)
    driver_dev = cuda.to_device(driver_host)
    driver_rate_dev = cuda.to_device(driver_rate_host)
    out_dev = cuda.to_device(out_host)

    kernel[1, 1](time_dev, driver_dev, driver_rate_dev, out_dev)
    out_dev.copy_to_host(out_host)

    expected = driver_host[0] + (time_host[0] + precision(1.0)) * driver_rate_host[0]
    np.testing.assert_allclose(out_host[0], expected, rtol=1e-6, atol=1e-6)
