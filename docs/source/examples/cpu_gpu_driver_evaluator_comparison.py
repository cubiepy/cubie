"""Compare CPU and GPU driver evaluators on shared input samples.

This example constructs a compact driver sequence sampled from ``t = 0.5`` to
``t = 1.5`` seconds and evaluates it with both the CUDA
:class:`cubie.array_interpolator.ArrayInterpolator` factory and the
CPU reference
:class:`tests.integrators.cpu_reference.cpu_utils.DriverEvaluator`.

Each subplot contrasts the GPU spline evaluation with its CPU counterpart for a
single end-point handling strategy while the sampled driver values are shown as
markers.  The script requires NumPy, Matplotlib, and a CUDA-capable device (or
Numba's CUDA simulator).
"""

import attrs
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from cubie.array_interpolator import ArrayInterpolator
from tests.integrators.cpu_reference.cpu_utils import DriverEvaluator


Array = NDArray[np.floating[Any]]


@attrs.define
class HandlingConfig:
    """Describe how samples should be extrapolated beyond their support."""

    label: str
    settings: Dict[str, object]


def build_sample_series(
    num_samples: int = 33,
    start: float = 0.5,
    stop: float = 1.5,
    precision: type[np.floating[Any]] = np.float64,
) -> Tuple[Array, Array]:
    """Return regularly spaced driver samples.

    Parameters
    ----------
    num_samples
        Number of values to sample between ``start`` and ``stop``.
    start
        Lower bound of the sampling interval in seconds.
    stop
        Upper bound of the sampling interval in seconds.
    precision
        Floating point dtype used for the samples.

    Returns
    -------
    tuple of numpy.ndarray
        Pair of time samples and associated driver values.
    """

    times = np.linspace(start, stop, num_samples, dtype=precision)
    base = np.sin(3 * np.pi * (times - 0.5))
    modulation = 0.25 * np.cos(3.0 * np.pi * times + 0.4)
    values = (base + modulation).astype(precision)
    values[-1] = values[0]
    return times, values


def evaluate_gpu(
    interpolator: ArrayInterpolator, query_times: Array
) -> Array:
    """Evaluate the CUDA driver interpolator on ``query_times``."""

    return interpolator.get_interpolated(query_times)


def evaluate_cpu(
    evaluator: DriverEvaluator,
    query_times: Array,
    precision: type[np.floating[Any]],
) -> Array:
    """Evaluate the CPU driver evaluator on ``query_times``."""

    out = np.empty(
        (query_times.size, evaluator.coefficients.shape[1]),
        dtype=precision,
    )
    for idx, time in enumerate(query_times):
        out[idx] = evaluator(time)
    return out


def build_driver_evaluator(
    interpolator: ArrayInterpolator,
    precision: type[np.floating[Any]],
) -> DriverEvaluator:
    """Construct a :class:`DriverEvaluator` mirroring ``interpolator``."""

    return DriverEvaluator(
        coefficients=np.array(
            interpolator.coefficients,
            copy=True,
            dtype=precision,
        ),
        dt=precision(interpolator.driver_sample_period),
        t0=precision(interpolator.t0),
        wrap=interpolator.wrap,
        precision=precision,
        boundary_condition=interpolator.boundary_condition,
    )


def plot_comparisons(
    handling_configs: Iterable[HandlingConfig],
    sample_times: Array,
    sample_values: Array,
    evaluation_times: Array,
    precision: type[np.floating[Any]],
) -> None:
    """Generate CPU versus GPU comparison plots for each handling mode."""

    handling_list = list(handling_configs)
    num_configs = len(handling_list)
    fig, axes = plt.subplots(
        num_configs,
        1,
        sharex=True,
        figsize=(10, 3 * num_configs),
    )
    if num_configs == 1:
        axes = [axes]

    for axis, config in zip(axes, handling_list):
        settings = {
            "time": sample_times,
            "driver": sample_values,
            "order": 3,
        }
        settings.update(config.settings)
        interpolator = ArrayInterpolator(
            precision=precision,
            input_dict=settings,
        )
        evaluator = build_driver_evaluator(interpolator, precision)
        gpu_values = evaluate_gpu(interpolator, evaluation_times)[:, 0]
        cpu_values = evaluate_cpu(evaluator, evaluation_times, precision)[:, 0]

        axis.plot(evaluation_times, gpu_values, label="GPU", linewidth=2.0)
        axis.plot(
            evaluation_times,
            cpu_values,
            label="CPU",
            linestyle="--",
            linewidth=1.5,
        )
        axis.plot(
            sample_times,
            sample_values,
            "o",
            label="Samples",
            markersize=4,
        )
        axis.set_title(config.label)
        axis.set_ylabel("Driver value")
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right")
    fig.suptitle("CPU and GPU driver evaluator comparison")
    fig.tight_layout()
    plt.show()


def main() -> None:
    """Run the comparison script."""

    precision = np.float64
    sample_times, sample_values = build_sample_series(precision=precision)
    evaluation_times = np.linspace(0.0, 2.0, 801, dtype=precision)

    handling_configs = (
        HandlingConfig(
            label="Clamped end points",
            settings={"wrap": False, "boundary_condition": "clamped"},
        ),
        HandlingConfig(
            label="Natural end points",
            settings={"wrap": False, "boundary_condition": "natural"},
        ),
        HandlingConfig(
            label="not-a-knot end points",
            settings={"wrap": False, "boundary_condition": "not-a-knot"},
        ),

        HandlingConfig(
            label="Wrapped end points",
            settings={"wrap": True, "boundary_condition": "periodic"},
        ),
    )

    plot_comparisons(
        handling_configs=handling_configs,
        sample_times=sample_times,
        sample_values=sample_values,
        evaluation_times=evaluation_times,
        precision=precision,
    )


if __name__ == "__main__":
    main()
