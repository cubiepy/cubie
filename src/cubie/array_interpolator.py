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
    Dict,
    Iterable,
    Mapping,
    Optional,
    TYPE_CHECKING,
    Any,
    Tuple,
)

from numpy import (
    full as np_full,
    int32 as np_int32,
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
from attrs import cmp_using, define, field, fields, validators, frozen
from numba_cuda_mlir.types import int32
from numba_cuda_mlir.numba_cuda.np.numpy_support import from_dtype
from cubie._cudasim_extensions import cuda
from cubie.backend.intrinsics import unroll_if
from numpy.typing import NDArray

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
from cubie.memory import default_memmgr

if TYPE_CHECKING:
    from cubie.memory.mem_manager import MemoryManager


FloatArray = NDArray[floating]


@define
class InterpolatorCache(CUDADispatcherCache):
    """Build outputs of :class:`ArrayInterpolator`.

    Attributes
    ----------
    drivers_fn
        Device function evaluating every column at a time.
    driver_derivative_fn
        Device function evaluating every column's time derivative.
    coefficients
        Host ``(num_segments, num_columns, order + 1)`` table.
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
        Dict mapping each driver's name to its 1-D sample array.
    time
        Uniformly spaced sample times, or give the period instead.
    driver_sample_period
        Spacing between samples.
    t0
        Time of the first sample, set to ``0.0`` when not given.

    Attributes
    ----------
    names
        Driver names in column order.
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

    def __getstate__(self) -> dict:
        """Return the pickled state with ``samples`` as a plain dict."""
        state = {
            fld.name: getattr(self, fld.name)
            for fld in fields(DriverSamples)
        }
        state["samples"] = dict(self.samples)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state and rebuild the read-only samples view."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "samples", MappingProxyType(state["samples"])
        )

    def _cubie_canonical_(self) -> Tuple[Any, ...]:
        """Identity: names, time base and sample count."""
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
            ``names`` differ from the sampled drivers.
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
        # Pass the time base on as it was given.
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
    derivative_columns : tuple of (int, int)
        ``(input, order)`` of each derivative column; set by the system.
    num_inputs : int
        Column count of the sample table.
    num_columns : int
        Inputs plus derivative columns; zero with no inputs.
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
    derivative_columns: Tuple[Tuple[int, int], ...] = field(
        default=(),
        converter=lambda columns: tuple(
            (int(column), int(order)) for column, order in columns
        ),
        validator=validators.instance_of(tuple),
    )
    driver_evaluation: str = field(
        default="columns",
        validator=validators.in_(
            {"columns", "combined", "combined_selp", "two_loop", "two_loop_mul", "two_loop_guard"}
        ),
    )
    num_inputs: int = field(default=0, init=False)
    num_columns: int = field(default=0, init=False)
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
        self._check_derivative_columns(num_inputs)
        object.__setattr__(self, "num_inputs", int(num_inputs))
        object.__setattr__(
            self,
            "num_columns",
            int(num_inputs + len(self.derivative_columns))
            if num_inputs and self.driver_evaluation == "columns"
            else int(num_inputs),
        )
        object.__setattr__(
            self, "num_segments", self._segment_count(num_samples)
        )

    def _check_derivative_columns(self, num_inputs: int) -> None:
        """Reject derivative columns the spline cannot supply."""
        for column, order in self.derivative_columns:
            if order < 1 or order > self.order:
                raise ValueError(
                    f"The system reads the order-{order} time derivative "
                    f"of a driver, but an order-{self.order} spline "
                    f"supplies derivatives up to order {self.order}; "
                    f"set the interpolation order to at least {order}."
                )
            if num_inputs and not 0 <= column < num_inputs:
                raise ValueError(
                    f"Derivative column reads input {column}, but the "
                    f"samples hold {num_inputs} inputs."
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

    @classmethod
    def system_inputs(cls, system: Any) -> Dict[str, Any]:
        """Return interpolator settings from a system object."""
        return dict(derivative_columns=system.driver_derivative_columns)

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
        num_columns = self.num_columns
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
        evaluation = self.compile_settings.driver_evaluation
        if evaluation != "columns":
            return self._build_slot_evaluators(
                coefficients,
                evaluation,
                order,
                inv_resolution,
                num_segments,
                wrap,
                unroll_other_small,
                zero_value,
                evaluation_start,
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
                seg = cuda.selp(idx < int32(0), int32(0), idx)
                seg = cuda.selp(
                    seg >= num_segments, int32(num_segments - 1), seg
                )
                tau = precision(scaled - precision(seg))

            # Evaluate polynomials using Horner's rule
            for input_index in unroll_if(
                range(num_columns), unroll_other_small
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
                seg = cuda.selp(idx < int32(0), int32(0), idx)
                seg = cuda.selp(
                    seg >= num_segments, int32(num_segments - 1), seg
                )
                tau = precision(scaled - precision(seg))

            for input_index in unroll_if(
                range(int32(num_columns)), unroll_other_small
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
        return (self.num_segments, self.num_columns, self.order + 1)

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
        times_device = cuda.to_device(times, stream=stream)
        coefficients_device = cuda.to_device(coefficients, stream=stream)
        out_device = cuda.device_array(
            (num_points, self.num_columns),
            dtype=self.precision,
            stream=stream,
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
        return out_device.copy_to_host()[:, : self.num_inputs]

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
            Fresh pinned ``(num_segments, num_columns, order + 1)``
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
        if self.compile_settings.driver_evaluation == "columns":
            derived = self._derivative_coefficients(coefficients, falling)
            coefficients = concatenate((coefficients, derived), axis=1)
        # Fresh pinned buffer per build.
        buffer = self._memory_manager.create_host_array(
            coefficients.shape, precision, "pinned"
        )
        buffer[...] = coefficients
        return buffer

    def _build_slot_evaluators(
        self,
        coefficients,
        evaluation,
        order,
        inv_resolution,
        num_segments,
        wrap,
        unroll_other_small,
        zero_value,
        evaluation_start,
    ) -> InterpolatorCache:
        """Compile the experimental derivative-slot evaluators."""
        precision = self.precision
        numba_precision = from_dtype(precision)
        num_inputs = self.num_inputs
        n_inputs = int32(num_inputs)
        slots = self.compile_settings.derivative_columns
        n_slots = int32(len(slots))
        has_slots = len(slots) > 0
        max_order = max((k for _, k in slots), default=0)
        inv_res = float(inv_resolution)

        def falling(p, k):
            return math.factorial(p) // math.factorial(p - k)

        # Combined: one Horner pass per input with derivative accumulators.
        n_value_terms = int32(max_order + 1)
        n_rate_terms = int32(max_order + 2)
        value_size = max_order + 1
        rate_size = max_order + 2
        driver_orders = zeros(max(num_inputs, 1), dtype=np_int32)
        slot_index = np_full(
            (max(num_inputs, 1), max_order + 2), -1, dtype=np_int32
        )
        for s, (i, k) in enumerate(slots):
            driver_orders[i] = max(driver_orders[i], k)
            slot_index[i, k] = num_inputs + s
        derivative_scale = asarray(
            [math.factorial(k) * inv_res**k for k in range(max_order + 2)],
            dtype=precision,
        )

        # Two-loop: each slot from its input's coefficients, shifted.
        n_rows = max(len(slots), 1)
        slot_inputs = asarray([i for i, _ in slots] or [0], dtype=np_int32)
        slot_powers = zeros((n_rows, order + 1), dtype=np_int32)
        slot_scales = zeros((n_rows, order + 1), dtype=precision)
        rate_powers = zeros((n_rows, order + 1), dtype=np_int32)
        rate_scales = zeros((n_rows, order + 1), dtype=precision)
        for s, (_, k) in enumerate(slots):
            for q in range(order + 1):
                slot_powers[s, q] = min(q + k, order)
                rate_powers[s, q] = min(q + k + 1, order)
                if q + k <= order:
                    slot_scales[s, q] = falling(q + k, k) * inv_res**k
                if q + k + 1 <= order:
                    rate_scales[s, q] = (
                        falling(q + k + 1, k + 1) * inv_res ** (k + 1)
                    )
        selp_commit = evaluation == "combined_selp"
        always_unroll = (True, None)
        top = int32(order)
        # Active, first and continuing powers of each slot and its rate.
        slot_on = zeros((n_rows, order + 1), dtype=bool)
        slot_start = zeros((n_rows, order + 1), dtype=bool)
        slot_continue = zeros((n_rows, order + 1), dtype=bool)
        rate_on = zeros((n_rows, order + 1), dtype=bool)
        rate_start = zeros((n_rows, order + 1), dtype=bool)
        rate_continue = zeros((n_rows, order + 1), dtype=bool)
        for s, (_, k) in enumerate(slots):
            for q in range(order + 1):
                slot_on[s, q] = q + k <= order
                slot_start[s, q] = q == order - k
                slot_continue[s, q] = q < order - k
                rate_on[s, q] = q + k + 1 <= order
                rate_start[s, q] = q == order - k - 1
                rate_continue[s, q] = q < order - k - 1

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def locate(time):
            """Return the segment, local position and in-range flag."""
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
                seg = cuda.selp(idx < int32(0), int32(0), idx)
                seg = cuda.selp(
                    seg >= num_segments, int32(num_segments - 1), seg
                )
                tau = precision(scaled - precision(seg))
            return seg, tau, in_range

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def combined_all(time, coefficients, out) -> None:
            """Evaluate inputs and their derivative slots in one pass."""
            seg, tau, in_range = locate(time)
            for input_index in unroll_if(
                range(n_inputs), unroll_other_small
            ):
                acc = cuda.local.array(value_size, numba_precision)
                for j in unroll_if(range(n_value_terms), unroll_other_small):
                    acc[j] = zero_value
                top = driver_orders[input_index]
                for p in unroll_if(
                    range(int32(order), int32(-1), int32(-1)),
                    unroll_other_small,
                ):
                    if has_slots:
                        for j in unroll_if(
                            range(
                                n_value_terms - int32(1),
                                int32(0),
                                int32(-1),
                            ),
                            unroll_other_small,
                        ):
                            if selp_commit:
                                acc[j] = cuda.selp(
                                    j <= top,
                                    acc[j] * tau + acc[j - 1],
                                    acc[j],
                                )
                            else:
                                if j <= top:
                                    acc[j] = acc[j] * tau + acc[j - 1]
                    acc[0] = (
                        acc[0] * tau + coefficients[seg, input_index, p]
                    )
                out[input_index] = acc[0] if in_range else zero_value
                if has_slots:
                    for k in unroll_if(
                        range(int32(1), n_value_terms), unroll_other_small
                    ):
                        index = slot_index[input_index, k]
                        if index >= int32(0):
                            out[index] = (
                                acc[k] * derivative_scale[k]
                                if in_range
                                else zero_value
                            )

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def combined_rate(time, coefficients, out) -> None:
            """Evaluate every buffer slot's time derivative in one pass."""
            seg, tau, in_range = locate(time)
            for input_index in unroll_if(
                range(n_inputs), unroll_other_small
            ):
                acc = cuda.local.array(rate_size, numba_precision)
                for j in unroll_if(range(n_rate_terms), unroll_other_small):
                    acc[j] = zero_value
                top = driver_orders[input_index] + int32(1)
                for p in unroll_if(
                    range(int32(order), int32(-1), int32(-1)),
                    unroll_other_small,
                ):
                    for j in unroll_if(
                        range(n_rate_terms - int32(1), int32(0), int32(-1)),
                        unroll_other_small,
                    ):
                        if selp_commit:
                            acc[j] = cuda.selp(
                                j <= top, acc[j] * tau + acc[j - 1], acc[j]
                            )
                        else:
                            if j <= top:
                                acc[j] = acc[j] * tau + acc[j - 1]
                    acc[0] = (
                        acc[0] * tau + coefficients[seg, input_index, p]
                    )
                out[input_index] = (
                    acc[1] * derivative_scale[1] if in_range else zero_value
                )
                if has_slots:
                    for k in unroll_if(
                        range(int32(1), n_value_terms), unroll_other_small
                    ):
                        index = slot_index[input_index, k]
                        if index >= int32(0):
                            out[index] = (
                                acc[k + 1] * derivative_scale[k + 1]
                                if in_range
                                else zero_value
                            )

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def two_loop_all(time, coefficients, out) -> None:
            """Evaluate inputs, then each slot from its input."""
            seg, tau, in_range = locate(time)
            for input_index in unroll_if(
                range(n_inputs), unroll_other_small
            ):
                acc = zero_value
                for p in unroll_if(
                    range(int32(order), int32(-1), int32(-1)),
                    unroll_other_small,
                ):
                    acc = acc * tau + coefficients[seg, input_index, p]
                out[input_index] = acc if in_range else zero_value
            if has_slots:
                for slot in unroll_if(range(n_slots), unroll_other_small):
                    column = slot_inputs[slot]
                    acc = zero_value
                    for q in unroll_if(
                        range(int32(order), int32(-1), int32(-1)),
                        unroll_other_small,
                    ):
                        acc = acc * tau + (
                            coefficients[seg, column, slot_powers[slot, q]]
                            * slot_scales[slot, q]
                        )
                    out[n_inputs + slot] = acc if in_range else zero_value

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def two_loop_rate(time, coefficients, out) -> None:
            """Evaluate input rates, then each slot's rate."""
            seg, tau, in_range = locate(time)
            for input_index in unroll_if(
                range(n_inputs), unroll_other_small
            ):
                acc = zero_value
                for p in unroll_if(
                    range(int32(order), int32(0), int32(-1)),
                    unroll_other_small,
                ):
                    acc = acc * tau + precision(p) * (
                        coefficients[seg, input_index, p]
                    )
                out[input_index] = (
                    acc * inv_resolution if in_range else zero_value
                )
            if has_slots:
                for slot in unroll_if(range(n_slots), unroll_other_small):
                    column = slot_inputs[slot]
                    acc = zero_value
                    for q in unroll_if(
                        range(int32(order), int32(-1), int32(-1)),
                        unroll_other_small,
                    ):
                        acc = acc * tau + (
                            coefficients[seg, column, rate_powers[slot, q]]
                            * rate_scales[slot, q]
                        )
                    out[n_inputs + slot] = acc if in_range else zero_value

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def peeled_inputs(seg, tau, in_range, coefficients, out):
            """Evaluate every input from its top coefficient down."""
            for input_index in unroll_if(
                range(n_inputs), unroll_other_small
            ):
                acc = coefficients[seg, input_index, top]
                for p in unroll_if(
                    range(top - int32(1), int32(-1), int32(-1)),
                    unroll_other_small,
                ):
                    acc = acc * tau + coefficients[seg, input_index, p]
                out[input_index] = acc if in_range else zero_value

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def peeled_input_rates(seg, tau, in_range, coefficients, out):
            """Evaluate every input's rate from its top coefficient down."""
            for input_index in unroll_if(
                range(n_inputs), unroll_other_small
            ):
                acc = precision(order) * coefficients[seg, input_index, top]
                for p in unroll_if(
                    range(top - int32(1), int32(0), int32(-1)),
                    unroll_other_small,
                ):
                    acc = acc * tau + precision(p) * (
                        coefficients[seg, input_index, p]
                    )
                out[input_index] = (
                    acc * inv_resolution if in_range else zero_value
                )

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def mul_all(time, coefficients, out) -> None:
            """Peeled inputs; slots switched by multiplying a bool."""
            seg, tau, in_range = locate(time)
            peeled_inputs(seg, tau, in_range, coefficients, out)
            if has_slots:
                for slot in unroll_if(range(n_slots), always_unroll):
                    column = slot_inputs[slot]
                    acc = (
                        coefficients[seg, column, slot_powers[slot, top]]
                        * slot_scales[slot, top]
                        * slot_on[slot, top]
                    )
                    for q in unroll_if(
                        range(top - int32(1), int32(-1), int32(-1)),
                        always_unroll,
                    ):
                        acc = acc * tau + (
                            coefficients[seg, column, slot_powers[slot, q]]
                            * slot_scales[slot, q]
                            * slot_on[slot, q]
                        )
                    out[n_inputs + slot] = acc if in_range else zero_value

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def mul_rate(time, coefficients, out) -> None:
            """Peeled input rates; slot rates switched by a bool."""
            seg, tau, in_range = locate(time)
            peeled_input_rates(seg, tau, in_range, coefficients, out)
            if has_slots:
                for slot in unroll_if(range(n_slots), always_unroll):
                    column = slot_inputs[slot]
                    acc = (
                        coefficients[seg, column, rate_powers[slot, top]]
                        * rate_scales[slot, top]
                        * rate_on[slot, top]
                    )
                    for q in unroll_if(
                        range(top - int32(1), int32(-1), int32(-1)),
                        always_unroll,
                    ):
                        acc = acc * tau + (
                            coefficients[seg, column, rate_powers[slot, q]]
                            * rate_scales[slot, q]
                            * rate_on[slot, q]
                        )
                    out[n_inputs + slot] = acc if in_range else zero_value

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def guard_all(time, coefficients, out) -> None:
            """Peeled inputs; each slot starts at its top real power."""
            seg, tau, in_range = locate(time)
            peeled_inputs(seg, tau, in_range, coefficients, out)
            if has_slots:
                for slot in unroll_if(range(n_slots), always_unroll):
                    column = slot_inputs[slot]
                    acc = zero_value
                    for q in unroll_if(
                        range(top, int32(-1), int32(-1)), always_unroll
                    ):
                        term = (
                            coefficients[seg, column, slot_powers[slot, q]]
                            * slot_scales[slot, q]
                        )
                        if slot_start[slot, q]:
                            acc = term
                        elif slot_continue[slot, q]:
                            acc = acc * tau + term
                    out[n_inputs + slot] = acc if in_range else zero_value

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def guard_rate(time, coefficients, out) -> None:
            """Peeled input rates; each slot rate starts at its top power."""
            seg, tau, in_range = locate(time)
            peeled_input_rates(seg, tau, in_range, coefficients, out)
            if has_slots:
                for slot in unroll_if(range(n_slots), always_unroll):
                    column = slot_inputs[slot]
                    acc = zero_value
                    for q in unroll_if(
                        range(top, int32(-1), int32(-1)), always_unroll
                    ):
                        term = (
                            coefficients[seg, column, rate_powers[slot, q]]
                            * rate_scales[slot, q]
                        )
                        if rate_start[slot, q]:
                            acc = term
                        elif rate_continue[slot, q]:
                            acc = acc * tau + term
                    out[n_inputs + slot] = acc if in_range else zero_value

        # no cover: end
        if evaluation == "two_loop_mul":
            drivers_fn, rate_fn = mul_all, mul_rate
        elif evaluation == "two_loop_guard":
            drivers_fn, rate_fn = guard_all, guard_rate
        elif evaluation == "two_loop":
            drivers_fn, rate_fn = two_loop_all, two_loop_rate
        else:
            drivers_fn, rate_fn = combined_all, combined_rate
        return InterpolatorCache(
            drivers_fn=drivers_fn,
            driver_derivative_fn=rate_fn,
            coefficients=coefficients,
            coefficients_shape=self.coefficients_shape,
        )

    def _derivative_coefficients(
        self, coefficients: FloatArray, falling: FloatArray
    ) -> FloatArray:
        """Return the polynomial columns of ``derivative_columns``.

        Parameters
        ----------
        coefficients
            ``(num_segments, num_inputs, order + 1)`` input polynomials.
        falling
            Falling factorials, ``falling[p, k] = p! / (p - k)!``.

        Returns
        -------
        numpy.ndarray
            ``(num_segments, len(derivative_columns), order + 1)``.
        """
        precision = self.precision
        order = self.order
        inv_resolution = precision(1.0) / precision(
            self.driver_sample_period
        )
        columns = self.compile_settings.derivative_columns
        derived = zeros(
            (coefficients.shape[0], len(columns), order + 1),
            dtype=precision,
        )
        for index, (column, derivative) in enumerate(columns):
            scale = inv_resolution**derivative
            for power in range(derivative, order + 1):
                derived[:, index, power - derivative] = (
                    coefficients[:, column, power]
                    * falling[power, derivative]
                    * scale
                )
        return derived

    # ---------------------------------------------------------------------- #
    # Getters and pass-through
    # ---------------------------------------------------------------------- #

    @property
    def num_inputs(self) -> int:
        """Return the number of input signals."""
        return self.compile_settings.num_inputs

    @property
    def num_columns(self) -> int:
        """Return the coefficient columns: inputs then derivatives."""
        return self.compile_settings.num_columns

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
