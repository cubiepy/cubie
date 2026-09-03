"""End-to-end tests for user device functions in generated systems.

The generated module is imported standalone, so user device callables
must be injected into its namespace before the factory compiles. These
tests build systems whose ``dxdt`` calls a real
``@cuda.jit(device=True)`` function and solve them, comparing against
an identical system written without user functions.
"""

import numpy as np
import pytest
from cubie.cuda_simsafe import cuda, INLINE_ALWAYS

from cubie import create_ODE_system, solve_ivp
from cubie.odesystems.symbolic.codegen.dxdt import (
    generate_dxdt_fac_code,
)


@pytest.fixture(scope="module")
def cubed():
    @cuda.jit(device=True, inline=INLINE_ALWAYS)
    def cubed(x):
        return x * x * x

    return cubed


@pytest.fixture(scope="module")
def d_cubed():
    @cuda.jit(device=True, inline=INLINE_ALWAYS)
    def d_cubed(x, index):
        return 3.0 * x * x

    return d_cubed


def _solve(system, method):
    result = solve_ivp(
        system,
        y0={"x": 2.0},
        method=method,
        duration=0.5,
        dt=0.01,
        save_every=0.05,
    )
    assert not np.any(result.status_codes)
    return result.time_domain_array


@pytest.fixture(scope="module")
def reference_system(precision):
    """Return the user-function-free twin of ``dx = -cubed(x)``."""
    return create_ODE_system(
        "dx = -x*x*x",
        states={"x": 2.0},
        precision=precision,
        name="userfunc_reference",
    )


@pytest.fixture(scope="module")
def derivative_system(cubed, d_cubed, precision):
    """Return the user-function system carrying a derivative."""
    return create_ODE_system(
        "dx = -cubed(x)",
        states={"x": 2.0},
        user_functions={"cubed": cubed},
        user_function_derivatives={"cubed": d_cubed},
        precision=precision,
        name="userfunc_derivative",
    )


@pytest.fixture(scope="module")
def reference_explicit(reference_system):
    return _solve(reference_system, "euler")


def test_string_system_device_function_solves(cubed, precision,
                                              reference_explicit):
    """String-form dxdt calling a device function compiles and solves."""
    system = create_ODE_system(
        "dx = -cubed(x)",
        states={"x": 2.0},
        user_functions={"cubed": cubed},
        precision=precision,
        name="userfunc_string_explicit",
    )
    state = _solve(system, "euler")
    np.testing.assert_allclose(state, reference_explicit, rtol=1e-6)


def test_callable_system_device_function_solves(cubed, precision,
                                                reference_explicit):
    """Callable-form dxdt calling a device function compiles and solves."""

    def rhs(t, y):
        dx = -cubed(y.x)  # noqa: F821
        return [dx]

    system = create_ODE_system(
        rhs,
        states={"x": 2.0},
        user_functions={"cubed": cubed},
        precision=precision,
        name="userfunc_callable_explicit",
    )
    state = _solve(system, "euler")
    np.testing.assert_allclose(state, reference_explicit, rtol=1e-6)


def test_device_function_with_derivative_implicit_solve(
    derivative_system, reference_system
):
    """Jacobian-based helpers resolve the derivative device function."""
    state = _solve(derivative_system, "backwards_euler")
    expected = _solve(reference_system, "backwards_euler")
    np.testing.assert_allclose(state, expected, rtol=1e-5)


@pytest.fixture(scope="module")
def region():
    @cuda.jit("int32(float32)", device=True, inline=INLINE_ALWAYS)
    def region(x):
        return 1

    return region


@pytest.mark.nocudasim
def test_nonfloat_function_call_prints_bare(
    cubed, region, precision, reference_explicit
):
    """Integer-return calls print bare; region(x) - 1 leaves -x**3."""
    system = create_ODE_system(
        "dx = -cubed(x) + region(x) - 1.0",
        states={"x": 2.0},
        user_functions={"cubed": cubed, "region": region},
        precision=precision,
        name="userfunc_nonfloat",
    )
    assert system.equations.nonfloat_functions == frozenset(
        {"region", "region_"}
    )
    code = generate_dxdt_fac_code(
        system.equations, system.indices, "dxdt_factory"
    )
    assert "precision(cubed(state[0]))" in code
    assert "+ region(state[0])" in code
    state = _solve(system, "euler")
    np.testing.assert_allclose(state, reference_explicit, rtol=1e-6)
