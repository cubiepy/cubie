"""Tests for CUDA input-array interpolation helpers."""

from typing import Tuple

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from cubie.odesystems.solver_helpers import SolverHelperRequest
from cubie.cuda_simsafe import cuda

from cubie.array_interpolator import ArrayInterpolator
from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
from tests.integrators.cpu_reference.cpu_utils import DriverEvaluator


@pytest.fixture(scope="session")
def quadratic_input(precision) -> ArrayInterpolator:
    """input array sampling ``f(t) = t^2`` on unit spacing."""

    times = np.arange(0.0, 6.0, 1.0, dtype=precision)
    values = times**2
    input_dict = {"values": values, "time": times, "order": 2, "wrap": False}
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )
    return input


@pytest.fixture(scope="session")
def cubic_inputs(precision) -> ArrayInterpolator:
    """Two input signals that are exactly represented by cubic polynomials."""

    times = np.linspace(0.0, 5.0, 11, dtype=precision)
    t = times
    input_dict = {
        "cubic1": t**3 - 2.0 * t,
        "cubic2": 0.5 * t**3 + 2 * t**2 + t,
        "time": times,
        "order": 3,
        "boundary_condition": "not-a-knot",
        "wrap": False,
    }
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )
    # Evaluation times avoid the final endpoint to prevent wrap behaviour.
    return input


@pytest.fixture(scope="session")
def wrapping_inputs(precision) -> Tuple[ArrayInterpolator, ArrayInterpolator]:
    """Return clamp and wrap input arrays sharing identical samples."""

    times = np.linspace(0.0, 4.0, 6, dtype=precision)
    values = np.array([0.0, 1.5, -0.75, 2.25, -3.0, 0.0], dtype=precision)
    clamp_input_dict = {"values": values, "time": times, "order": 3,
                        'wrap': False, 'boundary_condition': 'clamped'}
    wrap_input_dict = {"values": values, "time": times, "order": 3, 'wrap':
        True,
                       'boundary_condition': 'periodic'}
    clamp = ArrayInterpolator(
        precision=precision,
        input_dict=clamp_input_dict,
    )
    wrap = ArrayInterpolator(
        precision=precision,
        input_dict=wrap_input_dict,
    )
    return clamp, wrap


def _cpu_evaluate(
    time: float,
    coefficients: np.ndarray,
    start: float,
    resolution: float,
    wrap: bool,
    boundary_condition: str,
    precision,
) -> np.ndarray:
    """Evaluate all input polynomials on the CPU.

    Parameters
    ----------
    time : float
        Query time to evaluate.
    coefficients : numpy.ndarray
        Segment-major polynomial coefficients.
    start : float
        Start time of the input samples.
    resolution : float
        Temporal resolution between consecutive samples.
    wrap : bool
        Whether the input wraps beyond the final segment.

    Returns
    -------
    numpy.ndarray
        Evaluated input values at ``time``.
    """

    pad_clamped = (not wrap) and (boundary_condition == "clamped")
    resolution = precision(resolution)
    start = precision(start)
    time = precision(time)
    evaluation_start = start - (
        resolution if pad_clamped else precision(0.0)
    )
    inv_res = precision(precision(1.0) / resolution)
    num_segments = coefficients.shape[0]
    scaled = precision((time - evaluation_start) * inv_res)
    scaled_floor = precision(np.floor(scaled))
    idx = np.int32(scaled_floor)

    if wrap:
        segment = idx % num_segments
        if segment < 0:
            segment += num_segments
        tau = precision(scaled - scaled_floor)
        in_range = True
    else:
        in_range = (
            scaled >= precision(0.0)
            and scaled <= precision(float(num_segments))
        )
        segment = idx if idx >= 0 else 0
        if segment >= num_segments:
            segment = num_segments - 1
        tau = precision(scaled - precision(float(segment)))
    zero_value = precision(0.0)
    out = np.empty(coefficients.shape[1], dtype=precision)
    for input_index in range(coefficients.shape[1]):
        coeffs = coefficients[segment, input_index]
        acc = zero_value
        for k in range(np.int32(coeffs.size - 1), np.int32(-1), np.int32(-1)):
            acc = acc * tau + coeffs[k]
        if wrap or in_range:
            out[input_index] = acc
        else:
            out[input_index] = zero_value
    return out


def _run_evaluate(
    device_fn, coefficients: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    """Execute the device evaluation function across supplied samples.

    Parameters
    ----------
    device_fn : callable
        Device function to execute.
    coefficients : numpy.ndarray
        Segment-major polynomial coefficients.
    query_times : numpy.ndarray
        Time samples to evaluate on the device.

    Returns
    -------
    numpy.ndarray
        Evaluated input values for each query time.
    """

    n_times = query_times.size
    n_inputs = coefficients.shape[1]
    out_host = np.empty((n_times, n_inputs), dtype=coefficients.dtype)

    @cuda.jit
    def kernel(times, coeffs, out):
        idx = cuda.grid(1)
        if idx < times.size:
            device_fn(times[idx], coeffs, out[idx])

    d_times = cuda.to_device(query_times)
    d_coeffs = cuda.to_device(coefficients)
    d_out = cuda.to_device(out_host)
    threads_per_block = 64
    blocks = (n_times + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](d_times, d_coeffs, d_out)
    d_out.copy_to_host(out_host)
    return out_host


def test_driver_del_t_matches_cubic_reference(cubic_inputs,
                                              tolerance,
                                              precision):
    """Driver time derivatives should match the cubic analytic derivative."""

    derivative_fn = cubic_inputs.driver_del_t
    coefficients = cubic_inputs.coefficients
    query_times = np.linspace(
        cubic_inputs.t0,
        cubic_inputs.t0
        + cubic_inputs.driver_sample_period * (cubic_inputs.num_segments - 1),
        num=17,
        dtype=cubic_inputs.precision,
    )
    evaluated = _run_evaluate(derivative_fn, coefficients, query_times)
    times = query_times.astype(precision, copy=False)
    expected = np.column_stack(
        (
            3.0 * times**2 - 2.0,
            1.5 * times**2 + 4.0 * times + 1.0,
        )
    ).astype(cubic_inputs.precision, copy=False)
    np.testing.assert_allclose(
        evaluated,
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
        err_msg=(
            "driver_del_t evaluation diverged from analytic derivative\n"
            f"evaluated:\n{np.array2string(evaluated)}\n"
            f"expected:\n{np.array2string(expected)}"
        ),
    )


def test_cpu_driver_derivative_matches_gpu_reference(
    cubic_inputs, tolerance, precision
) -> None:
    """Driver derivatives from the CPU reference should match the GPU."""

    derivative_fn = cubic_inputs.driver_del_t
    coefficients = cubic_inputs.coefficients
    query_times = np.linspace(
        cubic_inputs.t0,
        cubic_inputs.t0
        + cubic_inputs.driver_sample_period * (cubic_inputs.num_segments - 1),
        num=17,
        dtype=cubic_inputs.precision,
    )
    evaluator = DriverEvaluator(
        coefficients=coefficients,
        dt=precision(cubic_inputs.driver_sample_period),
        t0=precision(cubic_inputs.t0),
        wrap=cubic_inputs.wrap,
        precision=precision,
        boundary_condition=cubic_inputs.boundary_condition,
    )
    gpu_values = _run_evaluate(derivative_fn, coefficients, query_times)
    cpu_values = np.stack(
        [evaluator.derivative(float(time)) for time in query_times]
    )
    np.testing.assert_allclose(
        cpu_values,
        gpu_values,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "CPU derivative diverged from ArrayInterpolator derivative\n"
            f"cpu:\n{np.array2string(cpu_values)}\n"
            f"gpu:\n{np.array2string(gpu_values)}"
        ),
    )


def _run_time_derivative(del_t, system, query_times):
    """Execute the symbolic time-derivative helper over supplied samples."""

    numba_precision = system.numba_precision
    n_state = system.sizes.states
    n_params = system.sizes.parameters
    n_drivers = system.num_drivers
    n_obs = system.sizes.observables
    n_out = n_state

    state_len = max(n_state, 1)
    param_len = max(n_params, 1)
    driver_len = max(n_drivers, 1)
    obs_len = max(n_obs, 1)
    out_len = max(n_out, 1)

    out_host = np.zeros(
        (query_times.size, out_len), dtype=system.precision
    )

    @cuda.jit
    def kernel(times, out):
        idx = cuda.grid(1)
        if idx < times.size:
            state = cuda.local.array(state_len, numba_precision)
            parameters = cuda.local.array(param_len, numba_precision)
            drivers = cuda.local.array(driver_len, numba_precision)
            driver_dt = cuda.local.array(driver_len, numba_precision)
            observables = cuda.local.array(obs_len, numba_precision)
            result = cuda.local.array(out_len, numba_precision)

            zero = numba_precision(0.0)
            for i in range(state_len):
                state[i] = zero
            for i in range(param_len):
                parameters[i] = zero
            for i in range(driver_len):
                drivers[i] = zero
                driver_dt[i] = zero
            for i in range(obs_len):
                observables[i] = zero
            for i in range(out_len):
                result[i] = zero

            del_t(
                state,
                parameters,
                drivers,
                driver_dt,
                observables,
                result,
                times[idx],
            )

            for j in range(n_out):
                out[idx, j] = result[j]

    d_times = cuda.to_device(
        query_times.astype(system.precision, copy=False)
    )
    d_out = cuda.to_device(out_host)
    threads_per_block = 64
    blocks = (query_times.size + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](d_times, d_out)
    d_out.copy_to_host(out_host)
    return out_host[:, :n_out]


def test_symbolic_time_derivative_matches_interpolated(cubic_inputs, precision):
    """Symbolic and interpolated derivatives of a cubic should agree."""

    system = SymbolicODE.create(
        dxdt=["dx = t**3 - 2.0 * t"],
        states={"x": precision(0.0)},
        precision=precision,
        strict=True,
        name="cubic_time_derivative",
    )

    helper = system.get_solver_helper(
        SolverHelperRequest(kind="time_derivative_rhs")
    ).device_function

    query_times = np.array([0.75, 2.25], dtype=precision)

    symbolic = _run_time_derivative(helper, system, query_times)[:, 0]

    derivative_fn = cubic_inputs.driver_del_t
    coefficients = cubic_inputs.coefficients
    interpolated = _run_evaluate(
        derivative_fn, coefficients, query_times
    )[:, 0]

    np.testing.assert_allclose(
        interpolated,
        symbolic,
        rtol=1e-7,
        atol=1e-7,
        err_msg=(
            "interpolated derivative diverged from symbolic reference\n"
            f"times: {np.array2string(query_times)}\n"
            f"interpolated: {np.array2string(interpolated)}\n"
            f"symbolic: {np.array2string(symbolic)}"
        ),
    )


def test_build_coefficients_matches_polynomial(quadratic_input, tolerance):
    """Segment-wise coefficients should reproduce the quadratic exactly."""

    coefficients = quadratic_input.coefficients[:, 0, :]
    interior = coefficients[1:-1]
    segment_starts = (
        np.arange(interior.shape[0], dtype=quadratic_input.precision)
        * quadratic_input.driver_sample_period
    )
    expected = np.column_stack(
        (
            segment_starts ** 2,
            2.0 * segment_starts,
            np.ones_like(segment_starts),
        )
    )
    np.testing.assert_allclose(
        interior,
        expected,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "interior coefficients diverged from quadratic reference\n"
            f"device:\n{np.array2string(interior)}\n"
            f"expected:\n{np.array2string(expected)}"
        ),
    )

    def _horner(coeffs, tau):
        value = 0.0
        for coef in coeffs[::-1]:
            value = value * tau + coef
        return value

    leading = coefficients[0]
    trailing = coefficients[-1]
    np.testing.assert_allclose(
        [_horner(leading, tau) for tau in (0.0, 1.0)],
        [0.0, 0.0],
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg="leading ghost segment should start and end at zero",
    )
    last_sample = quadratic_input.input_array[-1, 0]
    np.testing.assert_allclose(
        [_horner(trailing, tau) for tau in (0.0, 1.0)],
        [last_sample, 0.0],
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg="trailing ghost segment should decay to zero",
    )


def test_device_interpolation_matches_cpu(
    cubic_inputs, tolerance, precision
) -> None:
    """Device evaluation must agree with a CPU Horner evaluation."""

    input = cubic_inputs
    query_times = np.arange(input.num_samples + 2, dtype=np.int32)
    query_times = query_times.astype(input.precision) * input.driver_sample_period

    coefficients = input.coefficients
    device_fn = input.evaluation_function

    gpu = _run_evaluate(device_fn, coefficients, query_times)

    cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                coefficients,
                input.t0,
                input.driver_sample_period,
                wrap=False,
                boundary_condition=input.boundary_condition,
                precision=precision,
            )
            for t in query_times
        ]
    )

    np.testing.assert_allclose(
        gpu,
        cpu,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "device vs cpu Horner evaluation mismatch\n"
            f"gpu:\n{np.array2string(gpu)}\n"
            f"cpu:\n{np.array2string(cpu)}"
        ),
    )


def test_get_interpolated_matches_kernel_output(cubic_inputs):
    """Host helper should agree with explicit device kernel launches."""

    query_times = np.linspace(
        cubic_inputs.t0,
        cubic_inputs.t0 + cubic_inputs.driver_sample_period * (cubic_inputs.num_segments - 1),
        17,
        dtype=cubic_inputs.precision,
    )

    expected = _run_evaluate(
        cubic_inputs.evaluation_function,
        cubic_inputs.coefficients,
        query_times,
    )
    observed = cubic_inputs.get_interpolated(query_times)

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg=(
            "get_interpolated diverged from device reference\n"
            f"observed:\n{np.array2string(observed)}\n"
            f"expected:\n{np.array2string(expected)}"
        ),
    )


def test_get_interpolated_requires_one_dimensional_input(cubic_inputs):
    """Supplying multidimensional evaluation grids should fail."""

    bad_times = np.zeros((2, 2), dtype=cubic_inputs.precision)
    with pytest.raises(ValueError):
        cubic_inputs.get_interpolated(bad_times)


def test_wrap_vs_clamp_evaluation(
    wrapping_inputs, tolerance, precision
) -> None:
    """Wrapping alters extrapolation compared to clamping semantics."""

    clamp_input, wrap_input = wrapping_inputs

    query_times = np.array(
        [-0.5, 0.0, 1.5, 4.0, 4.5, 6.2], dtype=clamp_input.precision
    )

    clamp_gpu = _run_evaluate(
        clamp_input.evaluation_function, clamp_input.coefficients, query_times
    )
    wrap_gpu = _run_evaluate(
        wrap_input.evaluation_function, wrap_input.coefficients, query_times
    )

    clamp_cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                clamp_input.coefficients,
                clamp_input.t0,
                clamp_input.driver_sample_period,
                wrap=False,
                boundary_condition=clamp_input.boundary_condition,
                precision=precision,
            )
            for t in query_times
        ]
    )
    wrap_cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                wrap_input.coefficients,
                wrap_input.t0,
                wrap_input.driver_sample_period,
                wrap=True,
                boundary_condition=wrap_input.boundary_condition,
                precision=precision,
            )
            for t in query_times
        ]
    )

    # There's 2-3 ULP of accumulated difference between the two,
    # attributable to FMA operations in the accumulation, so we go for 5e-7
    # tolerance (2.5ULP)
    np.testing.assert_allclose(
        clamp_gpu,
        clamp_cpu,
        rtol=tolerance.rel_tight * 5,
        atol=tolerance.abs_tight * 5,
        err_msg=(
            "clamp device evaluation diverged from CPU reference\n"
            f"gpu:\n{np.array2string(clamp_gpu)}\n"
            f"cpu:\n{np.array2string(clamp_cpu)}"
        ),
    )



    np.testing.assert_allclose(
        wrap_gpu,
        wrap_cpu,
        rtol=tolerance.rel_tight*5,
        atol=tolerance.abs_tight*5,
        err_msg=(
            "wrap device evaluation diverged from CPU reference\n"
            f"gpu:\n{np.array2string(wrap_gpu)}\n"
            f"cpu:\n{np.array2string(wrap_cpu)}"
        ),
    )
    assert not np.allclose(
        clamp_gpu,
        wrap_gpu,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )


def test_non_wrap_returns_zero_outside_range(quadratic_input, tolerance) -> None:
    """Non-wrapping inputs must output zeros beyond sampled times."""

    input = quadratic_input
    coefficients = input.coefficients
    device_fn = input.evaluation_function
    dtype = input.precision
    end_time = input.t0 + (input.num_samples - 1) * input.driver_sample_period
    query_times = np.array(
        [
            input.t0 - input.driver_sample_period,
            input.t0 + 0.5 * input.driver_sample_period,
            end_time + input.driver_sample_period,
        ],
        dtype=dtype,
    )

    gpu = _run_evaluate(device_fn, coefficients, query_times)
    np.testing.assert_allclose(
        gpu[0],
        0.0,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "leading extrapolation should clamp to zero\n"
            f"gpu:\n{np.array2string(gpu[0])}"
        ),
    )
    np.testing.assert_allclose(
        gpu[2],
        0.0,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "trailing extrapolation should clamp to zero\n"
            f"gpu:\n{np.array2string(gpu[2])}"
        ),
    )
    assert not np.allclose(
        gpu[1],
        0.0,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )


def test_non_wrap_defaults_to_clamped_boundary(quadratic_input) -> None:
    """Non-wrapping inputs should apply clamped spline boundaries."""

    assert quadratic_input.boundary_condition == "clamped"


def test_wrap_repeats_periodically(wrapping_inputs, tolerance) -> None:
    """Wrapping inputs must repeat their samples beyond the final index."""

    _, wrap_input = wrapping_inputs
    coefficients = wrap_input.coefficients
    device_fn = wrap_input.evaluation_function
    period = wrap_input.num_segments * wrap_input.driver_sample_period
    dtype = wrap_input.precision
    query_times = np.array(
        [
            wrap_input.t0 + 0.25 * period,
            wrap_input.t0 + 1.25 * period,
            wrap_input.t0 - 0.75 * period,
        ],
        dtype=dtype,
    )

    gpu = _run_evaluate(device_fn, coefficients, query_times)
    np.testing.assert_allclose(
        gpu[0],
        gpu[1],
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
        err_msg=(
            "wrap evaluation should repeat forward period\n"
            f"gpu0:\n{np.array2string(gpu[0])}\n"
            f"gpu1:\n{np.array2string(gpu[1])}"
        ),
    )
    np.testing.assert_allclose(
        gpu[0],
        gpu[2],
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
        err_msg=(
            "wrap evaluation should repeat backward period\n"
            f"gpu0:\n{np.array2string(gpu[0])}\n"
            f"gpu2:\n{np.array2string(gpu[2])}"
        ),
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_polynomial_samples_are_reproduced(order, precision, tolerance) -> None:
    """Spline evaluation must match supplied samples for polynomial data."""

    times = np.linspace(0.0, 6.0, 25, dtype=precision)
    coeffs = np.arange(order + 1, dtype=precision) + 1.0
    values = np.zeros_like(times)
    for power, coef in enumerate(coeffs):
        values += coef * times**power
    input_dict = {"values": values, "time": times, "order": order, "wrap": False}
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )
    gpu_samples = _run_evaluate(
        input.evaluation_function,
        input.coefficients,
        times,
    )
    np.testing.assert_allclose(
        gpu_samples[:, 0],
        values,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "polynomial samples were not reproduced\n"
            f"gpu:\n{np.array2string(gpu_samples[:, 0])}\n"
            f"reference:\n{np.array2string(values)}"
        ),
    )


@pytest.mark.parametrize(
    "bc",
    ["natural", "periodic", "clamped", "not-a-knot"],
)
def test_order_three_matches_scipy_reference(precision, bc, tolerance) -> None:
    """Order-three interpolation should match SciPy's cubic spline."""

    rng = np.random.default_rng(1234)
    times = np.linspace(0.0, 2.0, 21, dtype=precision)
    samples = np.sin(2.3 * times) + 0.3 * np.cos(4.1 * times)
    samples += 0.05 * rng.standard_normal(times.size).astype(precision)
    wrap = bc == "periodic"
    if wrap:
        samples[-1] = samples[0]
    input_dict = {
        "drive": samples,
        "time": times,
        "order": 3,
        "wrap": wrap,
        "boundary_condition": bc,
    }
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )
    query = np.linspace(times[0], times[-1], 257, dtype=precision)
    gpu = _run_evaluate(input.evaluation_function, input.coefficients, query)
    scipy_samples = samples.copy()
    scipy_times = times.copy()
    if not wrap and bc == "clamped":
        dt = np.diff(times)[0]
        scipy_samples = np.hstack([precision(0), samples, precision(0)])
        scipy_times = np.hstack([times[0] - dt, times, times[-1] + dt])
    spline = CubicSpline(scipy_times, scipy_samples, bc_type=bc)
    scipy_values = np.asarray(spline(query), dtype=precision)

    np.testing.assert_allclose(
        gpu[:, 0],
        scipy_values,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
        err_msg=(
            "cubic interpolation diverged from SciPy reference\n"
            f"gpu:\n{np.array2string(gpu[:, 0])}\n"
            f"scipy:\n{np.array2string(scipy_values)}"
        ),
    )


def _falling_factorial(power: int, derivative: int) -> int:
    """Return the falling factorial ``power! / (power - derivative)!``."""

    result = 1
    for step in range(derivative):
        result *= power - step
    return result


def _evaluate_derivative(
    coefficients: np.ndarray,
    segment: int,
    input_index: int,
    tau: float,
    derivative: int,
    dt: float,
) -> float:
    """Evaluate a spline derivative at ``tau`` in the supplied segment."""

    total = 0.0
    for power in range(derivative, coefficients.shape[-1]):
        factor = _falling_factorial(power, derivative)
        total += (
            coefficients[segment, input_index, power]
            * factor
            * (tau ** (power - derivative))
        )
    return total / (dt ** derivative)


def test_natural_boundary_supports_higher_orders(precision, tolerance) -> None:
    """Natural constraints should zero high derivatives for any order."""

    order = 4
    times = np.linspace(0.0, 3.0, 9, dtype=precision)
    samples = np.sin(times) + 0.25 * times**2
    input_dict = {"drive": samples, "time": times, "order": order, "wrap": False, "boundary_condition": "natural"}
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )

    coefficients = input.coefficients
    dt = input.driver_sample_period
    expected_zero = []
    remaining = order - 1
    derivative = 2
    while remaining > 0 and derivative <= order:
        expected_zero.append((0, derivative))
        remaining -= 1
        if remaining == 0:
            break
        expected_zero.append((input.num_segments - 1, derivative))
        remaining -= 1
        derivative += 1

    for segment, derivative in expected_zero:
        tau = 0.0 if segment == 0 else 1.0
        value = _evaluate_derivative(coefficients, segment, 0, tau, derivative, dt)
        np.testing.assert_allclose(
            value,
            0.0,
            rtol=tolerance.rel_tight * 2, # 1 ULP @0.0
            atol=tolerance.abs_tight * 2, # 1 ULP @1.0
            err_msg=(
                "natural boundary derivative failed to vanish\n"
                f"segment={segment} derivative={derivative} value={value}"
            ),
        )

    gpu = _run_evaluate(input.evaluation_function, input.coefficients, times)
    reference = samples
    np.testing.assert_allclose(
        gpu[:, 0],
        reference,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg="natural boundary spline failed to reproduce samples",
    )


def test_periodic_boundary_respects_general_order(precision, tolerance) -> None:
    """Periodic constraints should hold for arbitrary spline order."""

    order = 5
    num_samples = 12
    times = np.linspace(0.0, 2.0 * np.pi, num_samples, dtype=precision)
    values = np.column_stack(
        (
            np.sin(times),
            0.75 * np.cos(2.0 * times),
        )
    )
    values = values.astype(precision)
    values[0] = values[-1]
    input_dict = {"s": values[:, 0], "c": values[:, 1], "time": times, "order": order, "wrap": True, "boundary_condition": "periodic"}
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )

    coefficients = input.coefficients
    dt = input.driver_sample_period
    last_segment = input.num_segments - 1
    for derivative in range(1, order):
        left = _evaluate_derivative(coefficients, 0, 0, 0.0, derivative, dt)
        right = _evaluate_derivative(
            coefficients,
            last_segment,
            0,
            1.0,
            derivative,
            dt,
        )
        np.testing.assert_allclose(
            left,
            right,
            rtol=tolerance.rel_loose,
            atol=tolerance.abs_loose,
            err_msg=(
                "periodic derivative mismatch for sine input\n"
                f"derivative={derivative} left={left} right={right}"
            ),
        )

    for derivative in range(1, order):
        left = _evaluate_derivative(coefficients, 0, 1, 0.0, derivative, dt)
        right = _evaluate_derivative(
            coefficients,
            last_segment,
            1,
            1.0,
            derivative,
            dt,
        )
        np.testing.assert_allclose(
            left,
            right,
            rtol=tolerance.rel_loose,
            atol=tolerance.abs_loose,
            err_msg=(
                "periodic derivative mismatch for cosine input\n"
                f"derivative={derivative} left={left} right={right}"
            ),
        )

    #A sample exactly on 2*np.pi gets slightly different results than numpy,
    # I believe as it gets shunted into the wrong polynomial by
    # floating-point rounding. As the value is so close to zero,
    # the relative error is large. Rather than soften relative tolerance for
    # all, we just fetch all but the exactly 2*pi sample.
    gpu = _run_evaluate(input.evaluation_function, input.coefficients,
                        times[:-1])
    reference = values[:-1,:]
    np.testing.assert_allclose(
        gpu,
        reference,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=("periodic spline failed to reproduce samples\n"
                 f"device={gpu}\nref={reference}\ndelta={gpu-reference}"),
    )


def test_cubic_interpolation_matches_analytic(cubic_inputs, precision, tolerance) -> None:
    """Cubic interpolation should reproduce analytic cubic polynomials at arbitrary query times."""

    inp = cubic_inputs
    # choose query times strictly inside the sampled range to avoid ghost/wrap handling
    start = inp.t0 + 0.1 * inp.driver_sample_period
    end = inp.t0 + (inp.num_samples - 2) * inp.driver_sample_period - 0.1 * inp.driver_sample_period
    query_times = np.linspace(start, end, 25, dtype=inp.precision)

    observed = inp.get_interpolated(query_times)

    times = query_times.astype(precision, copy=False)
    expected = np.column_stack(
        (
            times ** 3 - 2.0 * times,
            0.5 * times ** 3 + 2.0 * times ** 2 + times,
        )
    ).astype(inp.precision, copy=False)

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
        err_msg=(
            "cubic interpolation diverged from analytic cubic reference\n"
            f"times: {np.array2string(times)}\n"
            f"observed: {np.array2string(observed)}\n"
            f"expected: {np.array2string(expected)}")
        )


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "two_driver"}],
    indirect=True,
)
def test_check_against_system_drivers_orders_by_declared_order(
    system, precision
) -> None:
    """Driver entries are reordered to the system's declared order."""

    samples_a = np.full(6, 2.0, dtype=precision)
    samples_b = np.full(6, 5.0, dtype=precision)
    shuffled = {
        "d_b": samples_b,
        "d_a": samples_a,
        "driver_sample_period": precision(0.1),
        "wrap": False,
    }

    ordered = ArrayInterpolator.check_against_system_drivers(shuffled, system)

    driver_keys = [key for key in ordered if key in ("d_a", "d_b")]
    assert driver_keys == list(system.indices.driver_names)
    # Non-driver configuration/timing entries are preserved.
    assert ordered["driver_sample_period"] == precision(0.1)
    assert ordered["wrap"] is False


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "two_driver"}],
    indirect=True,
)
def test_interpolator_columns_track_declared_driver_order(
    system, precision
) -> None:
    """Coefficient columns align with declared drivers for any key order."""

    samples_a = np.full(6, 2.0, dtype=precision)
    samples_b = np.full(6, 5.0, dtype=precision)

    forward = {
        "d_a": samples_a,
        "d_b": samples_b,
        "driver_sample_period": precision(0.1),
    }
    reversed_dict = {
        "d_b": samples_b,
        "d_a": samples_a,
        "driver_sample_period": precision(0.1),
    }

    forward_interp = ArrayInterpolator(
        precision=precision,
        input_dict=ArrayInterpolator.check_against_system_drivers(
            forward, system
        ),
    )
    reversed_interp = ArrayInterpolator(
        precision=precision,
        input_dict=ArrayInterpolator.check_against_system_drivers(
            reversed_dict, system
        ),
    )

    # Column 0 holds d_a (constant 2.0), column 1 holds d_b (constant 5.0)
    # regardless of the caller's insertion order.
    np.testing.assert_array_equal(
        forward_interp.coefficients[0, :, 0],
        np.array([2.0, 5.0], dtype=precision),
    )
    np.testing.assert_array_equal(
        reversed_interp.coefficients[0, :, 0],
        forward_interp.coefficients[0, :, 0],
    )


# ── _normalise_input_array ──────────────────────────────────────────── #


def test_construction_rejects_non_convertible_array(precision):
    """An array that cannot be cast to a NumPy float array raises

    ValueError naming the offending key.
    """
    with pytest.raises(ValueError, match="could not be converted"):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "values": ["a", "b", "c"],
                "driver_sample_period": precision(0.1),
            },
        )


def test_construction_rejects_multidimensional_input(precision):
    """A two-dimensional input array raises ValueError."""
    with pytest.raises(ValueError, match="must be one-dimensional"):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "values": np.zeros((3, 2), dtype=precision),
                "driver_sample_period": precision(0.1),
            },
        )


def test_construction_rejects_mismatched_input_lengths(precision):
    """Input vectors of differing lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "a": np.zeros(5, dtype=precision),
                "b": np.zeros(4, dtype=precision),
                "driver_sample_period": precision(0.1),
            },
        )


def test_construction_rejects_too_few_samples(precision):
    """Fewer than order + 1 samples raise ValueError."""
    with pytest.raises(ValueError, match="At least order \\+ 1 samples"):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "values": np.array([1.0, 2.0], dtype=precision),
                "order": 3,
                "driver_sample_period": precision(0.1),
            },
        )


# ── _validate_time_inputs ───────────────────────────────────────────── #


def test_construction_rejects_both_dt_and_time(precision):
    """Providing both dt and time raises ValueError."""
    with pytest.raises(
        ValueError, match="Only one of driver_sample_period or time"
    ):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "values": np.arange(4, dtype=precision),
                "driver_sample_period": precision(0.1),
                "time": np.arange(4, dtype=precision),
            },
        )


def test_construction_rejects_multidimensional_time(precision):
    """A two-dimensional time array raises ValueError."""
    with pytest.raises(ValueError, match="Time array must be"):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "values": np.arange(4, dtype=precision),
                "time": np.zeros((4, 1), dtype=precision),
            },
        )


def test_construction_rejects_time_length_mismatch(precision):
    """A time array whose length differs from the samples raises

    ValueError.
    """
    with pytest.raises(ValueError, match="must match the number"):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "values": np.arange(4, dtype=precision),
                "time": np.arange(5, dtype=precision),
            },
        )


def test_construction_rejects_non_increasing_time(precision):
    """A non-strictly-increasing time array raises ValueError."""
    with pytest.raises(ValueError, match="strictly increasing"):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "values": np.array([1.0, 2.0, 3.0, 4.0], dtype=precision),
                "time": np.array([0.0, 1.0, 1.0, 2.0], dtype=precision),
            },
        )


def test_construction_rejects_non_uniform_time(precision):
    """A non-uniformly-spaced time array raises ValueError."""
    with pytest.raises(ValueError, match="uniformly spaced"):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "values": np.array([1.0, 2.0, 3.0, 4.0], dtype=precision),
                "time": np.array([0.0, 1.0, 3.0, 4.0], dtype=precision),
            },
        )


def test_construction_rejects_neither_dt_nor_time(precision):
    """Providing neither dt nor time raises ValueError."""
    with pytest.raises(
        ValueError, match="time array or driver_sample_period"
    ):
        ArrayInterpolator(
            precision=precision,
            input_dict={"values": np.arange(4, dtype=precision)},
        )


# ── update() ─────────────────────────────────────────────────────────── #


def test_update_with_no_changes_returns_empty_set(quadratic_input):
    """update() with no arguments returns an empty set without error."""
    assert quadratic_input.update() == set()
    assert quadratic_input.update(updates_dict={}) == set()


def test_update_accepts_kwargs():
    """kwargs passed to update() are merged and recognised."""
    interp = ArrayInterpolator(
        precision=np.float32,
        input_dict={
            "values": np.arange(6, dtype=np.float32),
            "driver_sample_period": np.float32(0.1),
            "order": 2,
            "wrap": False,
        },
    )
    recognised = interp.update(order=1)
    assert "order" in recognised
    assert interp.order == 1


def test_update_raises_on_unrecognised_parameter(quadratic_input):
    """An unrecognised update key raises KeyError."""
    with pytest.raises(KeyError, match="Unrecognized parameters"):
        quadratic_input.update(not_a_real_parameter=1)


# ── get_input_array / get_interpolated ──────────────────────────────── #


def test_get_input_array_returns_normalised_array(quadratic_input):
    """get_input_array returns the stored normalised input array."""
    array = quadratic_input.get_input_array()
    assert array is quadratic_input.input_array


def test_get_interpolated_empty_times_returns_empty_array(quadratic_input):
    """An empty eval_times array short-circuits to an empty result."""
    result = quadratic_input.get_interpolated(np.array([], dtype=np.float64))
    assert result.shape == (0, quadratic_input.num_inputs)


def test_get_interpolated_requires_coefficients(quadratic_input):
    """get_interpolated raises RuntimeError if coefficients are missing.

    Coefficients are always populated by construction; this exercises
    the documented defensive guard by clearing the cached array
    directly, the only way to reach the un-set state.
    """
    original = quadratic_input._coefficients
    try:
        quadratic_input._coefficients = None
        with pytest.raises(RuntimeError, match="have not been generated"):
            quadratic_input.get_interpolated(
                np.array([0.5], dtype=np.float64)
            )
    finally:
        quadratic_input._coefficients = original


# ── check_against_system_drivers ────────────────────────────────────── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "two_driver"}],
    indirect=True,
)
def test_check_against_system_drivers_rejects_wrong_count(system, precision):
    """A driver-count mismatch raises ValueError."""
    with pytest.raises(ValueError, match="does not match number of"):
        ArrayInterpolator.check_against_system_drivers(
            {"d_a": np.zeros(4, dtype=precision), "driver_sample_period": precision(0.1)},
            system,
        )


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"system_type": "two_driver"}],
    indirect=True,
)
def test_check_against_system_drivers_rejects_wrong_symbols(
    system, precision,
):
    """A driver-name mismatch raises ValueError."""
    with pytest.raises(ValueError, match="do not match drivers"):
        ArrayInterpolator.check_against_system_drivers(
            {
                "d_a": np.zeros(4, dtype=precision),
                "not_a_driver": np.zeros(4, dtype=precision),
                "driver_sample_period": precision(0.1),
            },
            system,
        )


# ── _compute_coefficients: periodic-boundary guards ─────────────────── #


def test_periodic_boundary_requires_wrap(precision):
    """An explicit periodic boundary condition with wrap=False raises

    ValueError.
    """
    with pytest.raises(ValueError, match="require wrap=True"):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "values": np.arange(6, dtype=precision),
                "driver_sample_period": precision(0.1),
                "wrap": False,
                "boundary_condition": "periodic",
            },
        )


def test_periodic_boundary_requires_matching_endpoints(precision):
    """Periodic boundary conditions with mismatched endpoints raise

    ValueError.
    """
    values = np.array([0.0, 1.0, 2.0, 3.0, 5.0], dtype=precision)
    with pytest.raises(ValueError, match="first and last samples"):
        ArrayInterpolator(
            precision=precision,
            input_dict={
                "values": values,
                "driver_sample_period": precision(0.1),
                "wrap": True,
            },
        )


# ── _compute_coefficients: not-a-knot single-constraint order ───────── #


def test_not_a_knot_order_two_uses_single_start_constraint(precision):
    """order=2 not-a-knot needs only one constraint, exercising the

    early-exit branch before the mirrored end-of-grid constraint is
    added.
    """
    times = np.arange(0.0, 5.0, 1.0, dtype=precision)
    values = times**2
    interp = ArrayInterpolator(
        precision=precision,
        input_dict={
            "values": values,
            "time": times,
            "order": 2,
            "wrap": False,
            "boundary_condition": "not-a-knot",
        },
    )
    assert interp.coefficients is not None


# ── update: coefficients track the compiled configuration ───────────── #


def test_settings_only_update_recomputes_coefficients(precision):
    """``update`` keeps coefficients matched to the compiled layout."""
    times = np.arange(0.0, 6.0, 1.0, dtype=precision)
    interp = ArrayInterpolator(
        precision=precision,
        input_dict={
            "values": times**2,
            "time": times,
            "order": 2,
            "wrap": False,
            "boundary_condition": "clamped",
        },
    )
    base_segments = interp.num_samples - 1
    assert interp.num_segments == base_segments + 2
    assert interp.coefficients.shape == interp.coefficients_shape

    interp.update({"boundary_condition": "natural"})
    assert interp.num_segments == base_segments
    assert interp.coefficients.shape == interp.coefficients_shape

    interp.update({"order": 4})
    assert interp.boundary_condition == "natural"
    assert interp.coefficients_shape == (base_segments, 1, 5)
    assert interp.coefficients.shape == interp.coefficients_shape

    evaluated = interp.get_interpolated(
        np.linspace(0.5, 4.5, 9, dtype=precision)
    )
    assert evaluated.shape == (9, 1)
    assert np.all(np.isfinite(evaluated))


def test_update_from_dict_applies_config_change_with_equal_arrays(
    precision,
):
    """Equal arrays with new settings still refresh the evaluator."""
    times = np.arange(0.0, 6.0, 1.0, dtype=precision)
    input_dict = {
        "values": times**2,
        "time": times,
        "order": 2,
        "wrap": False,
        "boundary_condition": "clamped",
    }
    interp = ArrayInterpolator(precision=precision, input_dict=input_dict)
    assert interp.update_from_dict(dict(input_dict)) is False

    changed_dict = dict(input_dict)
    changed_dict["order"] = 3
    assert interp.update_from_dict(changed_dict) is True
    assert interp.order == 3
    assert interp.coefficients_shape[2] == 4
    assert interp.coefficients.shape == interp.coefficients_shape

