"""Piecewise polynomial interpolation of array-driven forcing terms.

Published Classes
-----------------
:class:`DriverSamples`
    Uniformly sampled driver series with their time base; the
    ``drivers`` setting of a Solver.

:class:`ArrayInterpolatorConfig`
    Configuration container describing an input-array interpolation
    problem (order, wrap, boundary condition, samples).

:class:`ArrayInterpolator`
    CUDAFactory that computes spline coefficients from sampled driver
    arrays and compiles CUDA device functions for on-device evaluation.

    >>> from numpy import float64, linspace, sin
    >>> times = linspace(0, 1, 11)
    >>> drivers = DriverSamples({"driver_0": sin(times)}, time=times)
    >>> interp = ArrayInterpolator(precision=float64, drivers=drivers)
    >>> interp.num_inputs
    1

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactory`
    Parent factory class.
:class:`~cubie.batchsolving.solver.Solver`
    Primary consumer that owns an ArrayInterpolator as
    ``driver_interpolator``.
"""

import math
from types import MappingProxyType
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
    TYPE_CHECKING,
    Any,
    Tuple,
)

from numpy import (
    allclose,
    any as np_any,
    arange,
    array_equal,
    asarray,
    column_stack,
    concatenate,
    diff,
    empty,
    floating,
    full_like,
    transpose as np_transpose,
    zeros,
    vstack,
)
from numpy.linalg import solve as np_solve
from attrs import cmp_using, define, field, validators, frozen
from cubie.cuda_simsafe import cuda, int32
from cubie.cuda_simsafe import unroll_if
from numpy.typing import NDArray

from cubie.cuda_simsafe import CUDA_SIMULATION, cupy, selp
from cubie.CUDAFactory import (
    CUDAFactory,
    CUDAFactoryConfig,
    CUDADispatcherCache,
    FrozenSettings,
)
from cubie._utils import (
    PrecisionDType,
    gttype_validator,
)
from cubie.memory import current_cupy_stream, default_memmgr

if TYPE_CHECKING:
    from cubie.memory.mem_manager import MemoryManager


FloatArray = NDArray[floating]


@define
class InterpolatorCache(CUDADispatcherCache):
    """Build outputs of :class:`ArrayInterpolator`.

    Attributes
    ----------
    drivers_fn
        Device function evaluating every input at a time.
    driver_derivative_fn
        Device function evaluating every input's time derivative.
    coefficients
        Host ``(num_segments, num_inputs, order + 1)`` table.
    coefficients_shape
        That table's layout.
    """

    drivers_fn: Optional[Callable] = field(default=None)
    driver_derivative_fn: Optional[Callable] = field(default=None)
    coefficients: Optional[FloatArray] = field(default=None)
    coefficients_shape: Tuple[int, int, int] = field(default=(0, 0, 0))


ALL_INTERPOLATOR_PARAMETERS = frozenset(
    {"drivers", "order", "wrap", "boundary_condition"}
)
"""Keyword args for the driver samples and interpolation settings."""


def _sample_converter(
    samples: Mapping[str, Any],
) -> Mapping[str, FloatArray]:
    """Return ``samples`` as a read-only mapping of owned 1-D arrays."""
    owned = {}
    for name, values in samples.items():
        try:
            array = asarray(values, dtype=float).copy()
        except (TypeError, ValueError):
            raise ValueError(
                f"Forcing array {name} could not be converted to a NumPy "
                "array."
            )
        if array.ndim != 1:
            raise ValueError(f"Forcing array {name} must be one-dimensional.")
        array.setflags(write=False)
        owned[str(name)] = array
    return MappingProxyType(owned)


def _optional_time_converter(value: Any) -> Optional[FloatArray]:
    """Return ``value`` as an owned 1-D float array, or ``None``."""
    if value is None:
        return None
    array = asarray(value, dtype=float).copy()
    if array.ndim != 1:
        raise ValueError("Time array must be one-dimensional.")
    array.setflags(write=False)
    return array


@frozen
class DriverSamples(FrozenSettings):
    """Uniformly sampled driver series and their time base.

    Parameters
    ----------
    samples
        Driver name to its 1-D sample array; every array has the same
        length.
    time
        Sample times, strictly increasing and uniformly spaced. Give
        this or ``driver_sample_period``.
    driver_sample_period
        Spacing between consecutive samples.
    t0
        Time of the first sample; ``0.0`` when only the period is
        given.

    Attributes
    ----------
    names
        The driver names in column order.
    input_array
        Sample columns ``(num_samples, num_inputs)``.
    """

    samples: Mapping[str, FloatArray] = field(
        converter=_sample_converter, eq=False
    )
    time: Optional[FloatArray] = field(
        default=None, converter=_optional_time_converter, eq=False
    )
    driver_sample_period: Optional[float] = field(default=None)
    t0: Optional[float] = field(default=None)
    names: Tuple[str, ...] = field(init=False)
    input_array: FloatArray = field(
        init=False, eq=cmp_using(eq=array_equal)
    )

    def __attrs_post_init__(self):
        if not self.samples:
            raise ValueError("DriverSamples needs at least one driver.")
        columns = list(self.samples.values())
        if any(col.shape[0] != columns[0].shape[0] for col in columns):
            raise ValueError(
                "All forcing vectors must have the same length / be "
                "sampled on the same grid"
            )
        input_array = column_stack(columns)
        input_array.setflags(write=False)
        object.__setattr__(self, "names", tuple(self.samples))
        object.__setattr__(self, "input_array", input_array)
        period, t0 = self._resolve_time_base(input_array.shape[0])
        object.__setattr__(self, "driver_sample_period", period)
        object.__setattr__(self, "t0", t0)

    def _resolve_time_base(self, num_samples: int) -> Tuple[float, float]:
        """Return ``(driver_sample_period, t0)`` from the given time base."""
        if self.time is None:
            if self.driver_sample_period is None:
                raise ValueError(
                    "Either a time array or driver_sample_period must be "
                    "provided."
                )
            period = float(self.driver_sample_period)
            t0 = 0.0 if self.t0 is None else float(self.t0)
        else:
            if self.driver_sample_period is not None or self.t0 is not None:
                raise ValueError(
                    "Only one of driver_sample_period or time should be "
                    "provided."
                )
            if self.time.shape[0] != num_samples:
                raise ValueError(
                    "Time array length must match the number of samples "
                    "in provided input vectors."
                )
            differences = diff(self.time)
            if np_any(differences <= 0.0):
                raise ValueError("Time array must be strictly increasing.")
            if not allclose(
                differences,
                full_like(differences, differences[0]),
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError("Time array must be uniformly spaced.")
            period = float(differences[0])
            t0 = float(self.time[0])
        if period <= 0.0:
            raise ValueError("driver_sample_period must be positive.")
        return period, t0

    def _cubie_canonical_(self) -> Tuple[Any, ...]:
        """Identity by names, time base and sample count, not values."""
        return (
            self.names,
            self.t0,
            self.driver_sample_period,
            self.num_samples,
        )

    @property
    def num_samples(self) -> int:
        """Samples per driver."""
        return int(self.input_array.shape[0])

    @property
    def num_inputs(self) -> int:
        """Number of drivers."""
        return int(self.input_array.shape[1])

    def ordered(self, names: Iterable[str]) -> "DriverSamples":
        """Return these samples with columns in the order of ``names``.

        Raises
        ------
        ValueError
            ``names`` and the sampled drivers differ in count or
            spelling.
        """
        names = tuple(names)
        if len(names) != len(self.names):
            raise ValueError(
                f"Number of sampled drivers ({len(self.names)}) does not "
                f"match number of drivers in system ({len(names)})."
            )
        if set(names) != set(self.names):
            raise ValueError(
                f"Sampled driver names ({set(self.names)}) do not match "
                f"drivers symbols in system ({set(names)})."
            )
        if names == self.names:
            return self
        # The time base was given one way; hand it on the same way.
        timed = self.time is not None
        return DriverSamples(
            {name: self.samples[name] for name in names},
            time=self.time,
            driver_sample_period=None if timed else self.driver_sample_period,
            t0=None if timed else self.t0,
        )


@frozen
class ArrayInterpolatorConfig(CUDAFactoryConfig):
    """Configuration describing an input-array interpolation problem.

    Attributes
    ----------
    precision : numpy.dtype
        Precision to be used when generating polynomial coefficients.
    order : int
        Polynomial order for the interpolation over each segment.
    wrap : bool
        Whether the vector should repeat or provide zero values
        outside of the sampled range.
    boundary_condition : {"natural", "periodic", "clamped", "not-a-knot"}
        Boundary condition for the spline interpolation; ``None``
        selects ``"periodic"`` when wrapping, else ``"clamped"``.
    drivers : DriverSamples, optional
        The sampled drivers; ``None`` interpolates nothing.
    num_inputs : int
        Column count of the sample table.
    num_segments : int
        Polynomial segments in the table: samples minus one, plus two
        ghost segments for clamped non-wrapping inputs, zero with no
        inputs.
    """

    order: int = field(
        default=3,
        validator=gttype_validator(int, 0),
    )
    wrap: bool = field(
        default=True,
        validator=validators.instance_of(bool),
    )
    _boundary_condition: Optional[str] = field(
        default=None,
        alias="boundary_condition",
        validator=validators.optional(
            validators.in_({"natural", "periodic", "not-a-knot", "clamped"})
        ),
    )
    drivers: Optional[DriverSamples] = field(
        default=None,
        validator=validators.optional(
            validators.instance_of(DriverSamples)
        ),
    )
    num_inputs: int = field(default=0, init=False)
    num_segments: int = field(default=0, init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        num_samples, num_inputs = self.input_array.shape
        if num_inputs and num_samples < self.order + 1:
            raise ValueError(
                "At least order + 1 samples are required to construct"
                " splines.",
            )
        self._check_periodic(num_samples)
        object.__setattr__(self, "num_inputs", int(num_inputs))
        object.__setattr__(
            self, "num_segments", self._segment_count(num_samples)
        )

    def _check_periodic(self, num_samples: int) -> None:
        """Reject periodic settings without wrap or matching ends."""
        if self.boundary_condition != "periodic":
            return
        if not self.wrap:
            raise ValueError(
                "Periodic boundary conditions require wrap=True so that "
                "the input repeats after the final segment."
            )
        if num_samples and not allclose(
            self.input_array[0], self.input_array[-1]
        ):
            raise ValueError(
                "Periodic boundary conditions require the first and "
                "last samples to match."
            )

    def _segment_count(self, num_samples: int) -> int:
        """Return the segment count the coefficient table holds."""
        if num_samples == 0:
            return 0
        pad_clamped = (not self.wrap) and (
            self.boundary_condition == "clamped"
        )
        return num_samples - 1 + (2 if pad_clamped else 0)

    @property
    def boundary_condition(self) -> str:
        """The boundary condition; derived from ``wrap`` when not given."""
        if self._boundary_condition is None:
            return "periodic" if self.wrap else "clamped"
        return self._boundary_condition

    @property
    def input_array(self) -> FloatArray:
        """Sample columns ``(num_samples, num_inputs)``; empty with none."""
        if self.drivers is None:
            return empty((0, 0))
        return self.drivers.input_array

    @property
    def num_samples(self) -> int:
        """Number of samples per input."""
        return int(self.input_array.shape[0])

    @property
    def t0(self) -> float:
        """Time of the first sample."""
        return 0.0 if self.drivers is None else self.drivers.t0

    @property
    def driver_sample_period(self) -> float:
        """Spacing between consecutive samples."""
        if self.drivers is None:
            return 1e-16
        return self.drivers.driver_sample_period


class ArrayInterpolator(CUDAFactory):
    """Factory emitting CUDA device functions for interpolating array-driven
    forcing terms."""

    def __init__(
        self,
        precision: PrecisionDType,
        drivers: Optional[DriverSamples] = None,
        memory_manager: "MemoryManager" = default_memmgr,
        **settings: Any,
    ) -> None:
        """Initialize the array interpolator factory.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for coefficients and evaluation.
        drivers : DriverSamples, optional
            The sampled drivers; ``None`` interpolates nothing.
        memory_manager : MemoryManager
            Manager whose policy sizes the pinned coefficients buffer.
        **settings
            ``order``, ``wrap`` and ``boundary_condition``.
        """
        super().__init__()
        self.setup_compile_settings(
            ArrayInterpolatorConfig(
                precision=precision, drivers=drivers, **settings
            )
        )
        self._memory_manager = memory_manager

    # ---------------------------------------------------------------------- #
    # Evaluation function machinery
    # ---------------------------------------------------------------------- #
    def build(self) -> InterpolatorCache:
        """Compute the coefficient table and compile its device evaluators.

        Returns
        -------
        InterpolatorCache
            Coefficients and evaluators; evaluators ``None`` without inputs.
        """
        coefficients = self._compute_coefficients()
        if self.num_inputs == 0:
            return InterpolatorCache(
                coefficients=coefficients,
                coefficients_shape=self.coefficients_shape,
            )
        precision = self.precision

        order = self.order
        num_inputs = self.num_inputs
        resolution = precision(self.driver_sample_period)
        inv_resolution = precision(precision(1.0) / resolution)
        start_time = precision(self.t0)
        num_segments = int32(self.num_segments)
        wrap = self.wrap
        boundary_condition = self.boundary_condition
        pad_clamped = (not wrap) and (boundary_condition == "clamped")
        unroll_other_small = self.compile_settings.unroll.unroll_other_small
        zero_value = precision(0.0)
        evaluation_start = precision(
            start_time - (resolution if pad_clamped else precision(0.0))
        )

        # no cover: start
        @cuda.jit(
            # (numba_precision,
            #  numba_precision[:,:,::1],
            #  numba_precision[::1]),
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def evaluate_all(time, coefficients, out) -> None:
            """Evaluate all input polynomials at ``time`` on the device.

            Parameters
            ----------
            time : float
                Query time for evaluation.
            coefficients : device array
                Segment-major coefficients with trailing polynomial degrees.
            out : device array
                Output array to populate with evaluated input values.
            """
            # Just in case, should no-op if input is precision-type
            time = precision(time)
            scaled = (time - evaluation_start) * inv_resolution
            scaled_floor = precision(math.floor(scaled))
            idx = int32(scaled_floor)

            if wrap:
                seg = int32(idx % num_segments)
                tau = precision(scaled - scaled_floor)
                in_range = True
            else:
                in_range = (scaled >= precision(0.0)) and (
                    scaled <= num_segments
                )
                seg = selp(idx < int32(0), int32(0), idx)
                seg = selp(seg >= num_segments, int32(num_segments - 1), seg)
                tau = precision(scaled - precision(seg))

            # Evaluate polynomials using Horner's rule
            for input_index in unroll_if(
                range(num_inputs), unroll_other_small
            ):
                acc = zero_value
                for k in unroll_if(
                    range(int32(order), int32(-1), int32(-1)),
                    unroll_other_small,
                ):
                    acc = acc * tau + coefficients[seg, input_index, k]
                out[input_index] = acc if in_range else zero_value

        # no cover: end

        # no cover: start
        @cuda.jit(
            # [(numba_precision,
            #   numba_precision[:,:,::1],
            #   numba_precision[::1])],
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def evaluate_time_derivative(
            time,
            coefficients,
            out,
        ) -> None:
            """Evaluate the derivative of each driver polynomial."""
            time = precision(time)
            scaled = (time - evaluation_start) * inv_resolution
            scaled_floor = precision(math.floor(scaled))
            idx = int32(scaled_floor)

            if wrap:
                seg = int32(idx % num_segments)
                tau = precision(scaled - scaled_floor)
                in_range = True
            else:
                in_range = (scaled >= precision(0.0)) and (
                    scaled <= num_segments
                )
                seg = selp(idx < int32(0), int32(0), idx)
                seg = selp(seg >= num_segments, int32(num_segments - 1), seg)
                tau = precision(scaled - precision(seg))

            for input_index in unroll_if(
                range(int32(num_inputs)), unroll_other_small
            ):
                acc = zero_value
                for k in unroll_if(
                    range(int32(order), int32(0), int32(-1)),
                    unroll_other_small,
                ):
                    acc = (
                        acc * tau
                        + precision(k) * (coefficients[seg, input_index, k])
                    )
                out[input_index] = (
                    acc * inv_resolution if in_range else zero_value
                )

        # no cover: end
        cache = InterpolatorCache(
            drivers_fn=evaluate_all,
            driver_derivative_fn=evaluate_time_derivative,
            coefficients=coefficients,
            coefficients_shape=self.coefficients_shape,
        )
        return cache

    @property
    def drivers_fn(self) -> Optional[Callable]:
        """Device function evaluating all inputs; ``None`` without inputs."""
        return self.get_cached_output("drivers_fn")

    @property
    def driver_derivative_fn(self) -> Optional[Callable]:
        """Driver time-derivative device function; ``None`` without inputs."""
        return self.get_cached_output("driver_derivative_fn")

    @property
    def coefficients(self) -> FloatArray:
        """Host coefficient table matching the compiled evaluators."""
        return self.get_cached_output("coefficients")

    @property
    def coefficients_shape(self) -> Tuple[int, int, int]:
        """Exact coefficient layout captured by the device evaluators."""
        return (self.num_segments, self.num_inputs, self.order + 1)

    # ---------------------------------------------------------------------- #
    # Inspection interface
    # ---------------------------------------------------------------------- #
    def get_interpolated(
        self,
        eval_times: NDArray[floating],
    ) -> NDArray[floating]:
        """Evaluate the interpolated drivers on the device.

        Parameters
        ----------
        eval_times
            One-dimensional array of query times.

        Returns
        -------
        numpy.ndarray
            Interpolated driver values with shape ``(len(eval_times),
            num_inputs)``.

        Raises
        ------
        ValueError
            Raised when ``eval_times`` is not one-dimensional.
        """

        times = asarray(eval_times, dtype=self.precision)
        if times.ndim != 1:
            raise ValueError("eval_times must be one-dimensional.")

        num_points = times.size
        if num_points == 0 or self.num_inputs == 0:
            return empty((num_points, self.num_inputs), dtype=self.precision)

        coefficients = self.coefficients
        device_eval = self.drivers_fn

        # no cover: start
        @cuda.jit(**self.jit_kwargs)
        def _evaluate_kernel(times_device, coefficients_device, out_device):
            idx = cuda.grid(1)
            if idx < times_device.shape[0]:
                device_eval(
                    times_device[idx],
                    coefficients_device,
                    out_device[idx],
                )

        # no cover: end

        stream = default_memmgr.get_group_stream()
        if CUDA_SIMULATION:  # pragma: no cover - simulated
            # The simulator runs kernels on host memory: NumPy arrays
            # pass straight in and the kernel writes the output array
            # in place, so there is nothing to stage or copy back.
            times_device = asarray(times)
            coefficients_device = coefficients
            out_device = empty(
                (num_points, self.num_inputs),
                dtype=self.precision,
            )
        else:
            with current_cupy_stream(stream):
                times_device = cupy.asarray(times)
                coefficients_device = cupy.asarray(coefficients)
                out_device = cupy.empty(
                    (num_points, self.num_inputs),
                    dtype=self.precision,
                )

        threads_per_block = 128
        blocks_per_grid = (num_points + threads_per_block - 1) // (
            threads_per_block
        )
        _evaluate_kernel[blocks_per_grid, threads_per_block, stream](
            times_device,
            coefficients_device,
            out_device,
        )
        stream.synchronize()

        if CUDA_SIMULATION:  # pragma: no cover - simulated
            return out_device
        return out_device.get()

    def plot_interpolated(
        self,
        eval_times: NDArray[floating],
    ) -> Tuple[Any, Any]:  # pragma: no cover - optional dependency
        """Plot interpolated drivers against the sampled input data.

        Parameters
        ----------
        eval_times
            One-dimensional array of times at which to evaluate the
            interpolated drivers.

        Returns
        -------
        tuple
            Matplotlib figure and axes containing the plot.

        Raises
        ------
        ImportError
            Raised when :mod:`matplotlib` is not installed.
        ValueError
            Raised when ``eval_times`` is not one-dimensional.
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Optional dependency matplotlib is required for plotting."
            ) from exc

        times = asarray(eval_times, dtype=self.precision)
        if times.ndim != 1:
            raise ValueError("eval_times must be one-dimensional.")

        interpolated = self.get_interpolated(times)

        sample_times = self.t0 + self.driver_sample_period * arange(
            self.num_samples,
            dtype=self.precision,
        )
        sample_values = self.input_array.astype(self.precision, copy=False)

        if self.wrap and times.size:
            period = self.driver_sample_period * self.num_samples
            min_eval = times.min()
            max_eval = times.max()
            repeats_before = int(
                math.ceil(max(0.0, (sample_times[0] - min_eval) / period))
            )
            repeats_after = int(
                math.ceil(max(0.0, (max_eval - sample_times[-1]) / period))
            )
            time_tiles = []
            value_tiles = []
            for step in range(repeats_before, 0, -1):
                time_tiles.append(sample_times - step * period)
                value_tiles.append(sample_values)
            time_tiles.append(sample_times)
            value_tiles.append(sample_values)
            for step in range(1, repeats_after + 1):
                time_tiles.append(sample_times + step * period)
                value_tiles.append(sample_values)
            marker_times = concatenate(time_tiles)
            marker_values = vstack(value_tiles)
        else:
            marker_times = sample_times
            marker_values = sample_values

        fig, ax = plt.subplots()
        for input_index in range(self.num_inputs):
            ax.plot(
                times,
                interpolated[:, input_index],
                label=f"Input {input_index}",
            )
            ax.plot(
                marker_times,
                marker_values[:, input_index],
                linestyle="None",
                marker="x",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Driver value")
        if self.num_inputs > 1:
            ax.legend()
        plt.show()
        return fig, ax

    # ---------------------------------------------------------------------- #
    # Spline coefficient generation
    # ---------------------------------------------------------------------- #

    def _compute_coefficients(self) -> FloatArray:
        """Return spline coefficients respecting the requested boundary.

        Returns
        -------
        numpy.ndarray
            Fresh pinned ``(num_segments, num_inputs, order + 1)``
            array; zero-sized with no inputs.

        Raises
        ------
        ValueError
            Raised when the constraints do not form a square system.
        """
        boundary_condition = self.boundary_condition

        precision = self.precision
        num_inputs = self.num_inputs
        order = self.order
        if num_inputs == 0:
            return zeros((0, 0, order + 1), dtype=precision)
        base_inputs = self.input_array.astype(precision, copy=False)

        pad_with_zeros = (not self.wrap) and boundary_condition == "clamped"
        if pad_with_zeros:
            zero_row = zeros((1, num_inputs), dtype=precision)
            inputs = vstack((zero_row, base_inputs, zero_row))
        else:
            inputs = base_inputs

        num_segments = inputs.shape[0] - 1

        num_coeffs = num_segments * (order + 1)
        matrix = zeros((num_coeffs, num_coeffs), dtype=precision)
        rhs = zeros((num_coeffs, num_inputs), dtype=precision)
        row_index = 0

        def coeff_index(segment: int, power: int) -> int:
            """Return the flattened coefficient index for ``segment``."""
            return segment * (order + 1) + power

        falling = zeros((order + 1, order + 1), dtype=precision)
        falling[:, 0] = precision(1.0)
        for derivative in range(1, order + 1):
            for power in range(derivative, order + 1):
                falling[power, derivative] = falling[
                    power, derivative - 1
                ] * precision(power - (derivative - 1))

        # Function value constraints at the left edge of each segment.
        for segment in range(num_segments):
            matrix[row_index, coeff_index(segment, 0)] = precision(1.0)
            rhs[row_index] = inputs[segment]
            row_index += 1

        # Function value constraints at the right edge of each segment.
        for segment in range(num_segments):
            base = coeff_index(segment, 0)
            for power in range(order + 1):
                matrix[row_index, base + power] = precision(1.0)
            rhs[row_index] = inputs[segment + 1]
            row_index += 1

        # Continuity of derivatives across interior knots.
        for segment in range(num_segments - 1):
            for derivative in range(1, order):
                base = coeff_index(segment, 0)
                for power in range(derivative, order + 1):
                    matrix[row_index, base + power] = falling[
                        power, derivative
                    ]
                next_index = coeff_index(segment + 1, derivative)
                matrix[row_index, next_index] -= falling[
                    derivative, derivative
                ]
                row_index += 1

        if boundary_condition == "natural":
            remaining = order - 1
            derivative = 2
            while remaining > 0 and derivative <= order:
                base_start = coeff_index(0, 0)
                matrix[row_index, base_start + derivative] = falling[
                    derivative, derivative
                ]
                row_index += 1
                remaining -= 1
                if remaining == 0:
                    break
                base_end = coeff_index(num_segments - 1, 0)
                for power in range(derivative, order + 1):
                    matrix[row_index, base_end + power] = falling[
                        power, derivative
                    ]
                row_index += 1
                remaining -= 1
                derivative += 1

        elif boundary_condition == "periodic":
            for derivative in range(1, order):
                base_last = coeff_index(num_segments - 1, 0)
                for power in range(derivative, order + 1):
                    matrix[row_index, base_last + power] = falling[
                        power, derivative
                    ]
                base_first = coeff_index(0, derivative)
                matrix[row_index, base_first] -= falling[
                    derivative, derivative
                ]
                row_index += 1

        elif boundary_condition == "clamped":
            remaining = order - 1
            derivative = 1
            while remaining > 0 and derivative <= order:
                base_start = coeff_index(0, 0)
                matrix[row_index, base_start + derivative] = falling[
                    derivative, derivative
                ]
                row_index += 1
                remaining -= 1
                if remaining == 0:
                    break
                base_end = coeff_index(num_segments - 1, 0)
                for power in range(derivative, order + 1):
                    matrix[row_index, base_end + power] = falling[
                        power, derivative
                    ]
                row_index += 1
                remaining -= 1
                derivative += 1

        elif boundary_condition == "not-a-knot":
            constraints_needed = order - 1
            constraints_added = 0
            highest_power = order
            for difference_order in range(1, order):
                if constraints_added >= constraints_needed:
                    break

                # Enforce vanishing forward difference at the start of the
                # grid.
                start_row = row_index
                for offset in range(difference_order + 1):
                    coefficient = (-1) ** (difference_order - offset)
                    coefficient *= math.comb(difference_order, offset)
                    segment = offset
                    matrix[start_row, coeff_index(segment, highest_power)] = (
                        precision(coefficient)
                    )
                row_index += 1
                constraints_added += 1
                if constraints_added >= constraints_needed:
                    break

                # Mirror the same finite-difference constraint at the end.
                end_row = row_index
                for offset in range(difference_order + 1):
                    coefficient = (-1) ** (difference_order - offset)
                    coefficient *= math.comb(difference_order, offset)
                    segment = num_segments - 1 - (difference_order - offset)
                    matrix[end_row, coeff_index(segment, highest_power)] = (
                        precision(coefficient)
                    )
                row_index += 1
                constraints_added += 1

        if row_index != num_coeffs:
            raise ValueError(
                "Failed to assemble a square spline system; "
                "please verify boundary condition handling."
            )

        solution = np_solve(matrix, rhs)
        coefficients = solution.reshape(num_segments, order + 1, num_inputs)
        coefficients = np_transpose(coefficients, (0, 2, 1))
        # Fresh pinned buffer per build.
        buffer = self._memory_manager.create_host_array(
            coefficients.shape, precision, "pinned"
        )
        buffer[...] = coefficients
        return buffer

    # ---------------------------------------------------------------------- #
    # Getters and pass-through
    # ---------------------------------------------------------------------- #

    @property
    def num_inputs(self) -> int:
        """Return the number of input signals."""
        return self.compile_settings.num_inputs

    @property
    def num_samples(self) -> int:
        """Number of samples available for interpolation."""
        return self.compile_settings.num_samples

    @property
    def input_array(self) -> FloatArray:
        """Return the normalised, read-only input array."""
        return self.compile_settings.input_array

    @property
    def order(self) -> int:
        """Return the interpolating polynomial order."""
        return self.compile_settings.order

    @property
    def wrap(self) -> bool:
        """Return whether the input should wrap past the final sample."""
        return self.compile_settings.wrap

    @property
    def boundary_condition(self) -> Optional[str]:
        """Return the spline boundary condition to enforce, if any."""
        return self.compile_settings.boundary_condition

    @property
    def num_segments(self) -> int:
        """Return the number of polynomial segments."""
        return self.compile_settings.num_segments

    @property
    def t0(self) -> float:
        """Return the start time of the input samples."""
        return self.compile_settings.t0

    @property
    def driver_sample_period(self) -> float:
        """Return the sample spacing."""
        return self.compile_settings.driver_sample_period
