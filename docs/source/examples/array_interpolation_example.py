"""Plot CUDA driver array interpolation against SciPy references.

The example builds a richly featured driver array with several harmonic and
modulated components, compiles the :class:`~cubie.integrators.driver_array.
ArrayInterpolator` device function, and then compares GPU-evaluated samples
against SciPy's :class:`scipy.interpolate.CubicSpline`. The script requires
SciPy, NumPy, Matplotlib, and a CUDA-capable device (or Numba's CUDA
simulator).
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from cubie.cuda_simsafe import cuda
from scipy.interpolate import CubicSpline

from cubie.array_interpolator import ArrayInterpolator


def build_wiggly_driver(
    num_samples: int = 32, duration: float = 1.0, precision=np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a long driver array with rapidly changing structure.

    Parameters
    ----------
    num_samples : int, optional
        Number of temporal samples to generate.
    duration : float, optional
        Total duration of the driver timeline in seconds.
    precision : numpy.dtype, optional
        Floating point precision to use, default np.float64

    Returns
    -------
    tuple of numpy.ndarray
        Pair of time samples and driver values.
    """

    times = np.linspace(0.0, duration, num_samples, dtype=precision)
    base = np.sin(0.6 * times) + 0.2 * np.sin(4.2 * times + 0.7)
    chirp = 0.35 * np.sin(0.15 * times ** 1.4)
    ripples = 0.5 * np.sin(21.0 * times) + 0.05 * np.cos(39.0 * times)
    envelope = 1.0 + 0.3 * np.sin(0.25 * times)
    values = envelope * (base + chirp + ripples)
    values[-1] = values[0]
    return times, values.astype(precision)


def evaluate_on_device(
    device_fn: Callable[[float, np.ndarray, np.ndarray], None],
    coefficients: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Evaluate the driver array device function on supplied samples.

    Parameters
    ----------
    device_fn : callable
        CUDA device function produced by :class:`ArrayInterpolator`.
    coefficients : numpy.ndarray
        Segment-major polynomial coefficients returned by ``build``.
    query_times : numpy.ndarray
        Times at which to evaluate the driver array.

    Returns
    -------
    numpy.ndarray
        Device-evaluated driver values for each query time.
    """

    out_host = np.empty((query_times.size, coefficients.shape[1]))

    @cuda.jit
    def kernel(times, coeffs, out):
        idx = cuda.grid(1)
        if idx < times.size:
            device_fn(times[idx], coeffs, out[idx])

    d_times = cuda.to_device(query_times)
    d_coeffs = cuda.to_device(coefficients)
    d_out = cuda.device_array_like(out_host)
    threads_per_block = 128
    blocks = (query_times.size + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](d_times, d_coeffs, d_out)
    return d_out.copy_to_host(out_host)


def main() -> None:
    """Generate plots comparing CUDA and SciPy spline interpolation."""
    precision = np.float64
    boundary_conditions = ["periodic", "natural", "clamped", "not-a-knot"]
    scipy_values = {}
    device_values = {}
    times, samples = build_wiggly_driver(precision=precision)
    dense_times = np.linspace(times[0], times[-1]*2, 8192, dtype=np.float64)

    for boundary_condition in boundary_conditions:
        interp = ArrayInterpolator(
            precision=precision,
            input_dict={
                'wiggler': samples,
                'time': times,
                'order': 3,
                'wrap': True,
                'boundary_condition': boundary_condition
            },
        )

        device_values[boundary_condition] = evaluate_on_device(
            interp.device_function, interp.coefficients, dense_times
        )

        spline = CubicSpline(times, samples,
                             bc_type=boundary_condition)
        scipy_values[boundary_condition] = spline(dense_times)

    plt.figure(figsize=(12, 6))
    plt.plot(times, samples, label="samples", alpha=0.5, linewidth=1.0)

    for boundary_condition in boundary_conditions:
        plt.plot(dense_times, device_values[boundary_condition][:, 0],
                 label=None, linewidth=1.5)
        plt.plot(
            dense_times,
            scipy_values[boundary_condition],
            label=None,
            linewidth=1.0,
        )
        plt.plot(
            dense_times[::32],
            device_values[boundary_condition][::32, 0], 'o',
            label=f"CUDA {boundary_condition}"
        )
        plt.plot(dense_times[::32], scipy_values[boundary_condition][::32],
                 'x',
                 label=f"SciPy {boundary_condition}", linewidth=1.0)
    plt.title("Driver array interpolation comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Driver value")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
