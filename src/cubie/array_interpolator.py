"""Piecewise polynomial interpolation of array-driven forcing terms.

Published Classes
-----------------
:class:`ArrayInterpolatorConfig`
    Configuration container describing an input-array interpolation
    problem (order, wrap, boundary condition, timing).

:class:`ArrayInterpolator`
    CUDAFactory that computes spline coefficients from sampled driver
    arrays and compiles CUDA device functions for on-device evaluation.

    >>> from numpy import float64, linspace, sin
    >>> times = linspace(0, 1, 11)
    >>> inputs = {"driver_0": sin(times)}
    >>> inputs["driver_sample_period"] = times[1] - times[0]
    >>> interp = ArrayInterpolator(precision=float64, input_dict=inputs)
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
from typing import (
    Callable,
    Dict,
    Optional,
    Set,
    TYPE_CHECKING,
    Union,
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
from attrs import define, field, validators, frozen
from cubie.cuda_simsafe import cuda, int32
from numpy.typing import NDArray

from cubie.cuda_simsafe import CUDA_SIMULATION, cupy, selp
from cubie.CUDAFactory import (
    CUDAFactory,
    CUDAFactoryConfig,
    CUDADispatcherCache,
)
from cubie._utils import (
    PrecisionDType,
    getype_validator,
    gttype_validator,
)
from cubie.memory import current_cupy_stream, default_memmgr

if TYPE_CHECKING:
    from cubie.memory.mem_manager import MemoryManager
    from cubie.odesystems.symbolic.symbolicODE import SymbolicODE


FloatArray = NDArray[floating]


@define
class InterpolatorCache(CUDADispatcherCache):
    """Cached device helpers emitted by :class:`ArrayInterpolator`."""

    evaluation_function: Optional[Callable] = field(default=None)
    driver_del_t: Optional[Callable] = field(default=None)


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
    boundary_condition : {"natural", "periodic", "clamped", "not-a-knot"},
        optional boundary condition for the spline interpolation.
        defaults to 'not-a-knot' to match Scipy's CubicSpline.
    t0 : float
        start time of input samples
    driver_sample_period : float
        Temporal spacing between consecutive driver samples.
    num_inputs : int
        Number of separate input vectors
    num_segments : int
        Number of polynomial segments in the coefficient table. For
        clamped, non-wrapping inputs this includes two ghost segments that
        transition from and to zero-valued padding samples.
    """

    order: int = field(
        default=3,
        validator=gttype_validator(int, 0),
    )
    wrap: bool = field(
        default=True,
        validator=validators.instance_of(bool),
    )
    boundary_condition: str = field(
        default="not-a-knot",
        validator=validators.optional(
            validators.in_({"natural", "periodic", "not-a-knot", "clamped"})
        ),
    )
    driver_sample_period: float = field(
        default=1e-16, validator=getype_validator(float, 0)
    )
    t0: float = field(default=0.0, validator=getype_validator(float, 0))
    num_inputs: int = field(
        default=0,
        validator=validators.instance_of(int),
    )
    num_segments: int = field(
        default=0,
        validator=validators.instance_of(int),
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()


class ArrayInterpolator(CUDAFactory):
    """Factory emitting CUDA device functions for interpolating array-driven
    forcing terms."""

    config_keys = ("wrap", "order", "boundary_condition")
    time_info = ("time", "driver_sample_period", "t0")

    def __init__(
        self,
        precision: PrecisionDType,
        input_dict: Dict[str, FloatArray],
        memory_manager: "MemoryManager" = default_memmgr,
    ) -> None:
        """Initialize the array interpolator factory.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for coefficients and evaluation.
        input_dict : dict
            Dictionary containing input arrays and configuration. See
            :meth:`update_from_dict` for required and optional fields.
        memory_manager : MemoryManager
            Manager whose policy sizes the pinned coefficients buffer.
        """
        super().__init__()
        config = ArrayInterpolatorConfig(
            precision=precision,
        )
        self.setup_compile_settings(config)
        self._memory_manager = memory_manager
        self._coefficients = None
        self._input_array = None
        self.update_from_dict(input_dict)

    def update_from_dict(self, input_dict: Dict[str, Any]) -> bool:
        """Update the factory configuration from a user-supplied dictionary.

        Parameters
        ----------
        input_dict
            Dictionary containing input arrays and configuration options
            for the interpolated inputs.

        Returns
        -------
        bool
            ``True`` when the compiled evaluator configuration changed
            and consumers must refresh their device-function handles.

        Notes
        -----
        ## Input dictionary
        input_dict fields must include:

            - ``"time"``: 1D float array of sample times corresponding to
            input array values, or
                - ``"driver_sample_period"``: uniform spacing between samples, and
                - ``"t0"``: starting time of the input samples.
            - ``[input_name]``: one-dimensional float array of samples for
            each input, where ``input_name`` is the name of the input signal
            as entered in the system definition.

            Fields may optionally include:

            - ``"order"``: polynomial order for spline interpolation,
            default 3.
            - ``"wrap"``: whether the input should wrap past the final
            value when the last time index is exceeded. When False the
            interpolator clamps to zero before ``t0`` and after the final
            sample.
            - ``"boundary_condition"``: boundary condition for splines.
            Defaults to ``"clamped"`` when ``"wrap"`` is False and to
            ``"periodic"`` when wrapping is enabled.

        The input arrays must all be one-dimensional and of the same length.

        The final interpolation result is an array of polynomial
        coefficients with shape (num_segments, num_inputs, order + 1),
        where num_segments is one less than the number of samples provided.

        ## Interpolation behaviour
        If ``"boundary_condition"`` is None, then spline coefficients are
        calculated in segments, with no continuity constraints. Otherwise,
        the spline coefficients are fit simultaneously for all segments,
        and end conditions are enforced according to the boundary condition:

        - ``"natural"``: second derivative at the ends of the curve is set
        to zero.
        - ``"periodic"``: the first and last segments are identical. For
        this condition, the first and last samples must match. This is the
        default when "wrap" is True, to avoid introducing a discontinuity on
        wrap.

        These boundary conditions are identical to those in [SciPy's
        CubicSpline interpolator]<https://docs.scipy.org/doc/scipy/reference
        /generated/scipy.interpolate.CubicSpline.html>
        """

        config = {k: v for k, v in input_dict.items() if k in self.config_keys}
        inputs = {
            k: v
            for k, v in input_dict.items()
            if k not in self.config_keys and k not in self.time_info
        }
        time = {k: v for k, v in input_dict.items() if k in self.time_info}

        # Update order first, for checks  in _normalise_input_array
        initial_hash = self.compile_settings.values_hash
        self.update_compile_settings(config)

        input_array = self._normalise_input_array(inputs)
        arrays_changed = not array_equal(input_array, self.input_array)
        if arrays_changed:
            self._input_array = input_array

        sample_period, t0 = self._validate_time_inputs(time)
        config.update(
            {
                "t0": t0,
                "driver_sample_period": sample_period,
                "num_inputs": self.num_inputs,
            }
        )

        # Final update; invalidates cache if settings have changed.
        self._derive_segment_settings(config)
        self.update_compile_settings(config)
        fn_changed = self.compile_settings.values_hash != initial_hash
        if fn_changed or arrays_changed:
            self._coefficients = self._compute_coefficients()

        return fn_changed

    def _derive_segment_settings(self, config: Dict[str, Any]) -> None:
        """Fill derived boundary-condition and segment-count settings.

        Parameters
        ----------
        config
            Pending driver configuration updates, modified in place.
            ``boundary_condition`` defaults from the wrap setting when
            absent, and ``num_segments`` is set from the sample count
            and boundary condition.
        """
        base_segments = self.num_samples - 1
        wrap_setting = config.get("wrap", self.wrap)
        if wrap_setting:
            if "boundary_condition" not in config:
                config["boundary_condition"] = "periodic"
            num_segments = base_segments
        elif "boundary_condition" not in config:
            config["boundary_condition"] = "clamped"
            num_segments = base_segments + 2
        else:
            boundary = config["boundary_condition"]
            if boundary == "clamped":
                num_segments = base_segments + 2
            else:
                num_segments = base_segments
        config["num_segments"] = num_segments

    def _normalise_input_array(
        self, input_dict: Dict[str, FloatArray]
    ) -> FloatArray:
        """Construct inputs array and check sizes.

        Parameters
        ----------
        input_dict
            Dictionary mapping input names to 1d arrays of samples.

        Returns
        -------
        np.ndarray of floats
            Input vectors stacked into a single array.

        Raises
        ------
        ValueError
            Raised when the input array is the wrong shape, type,
            or multiple arrays have different lengths.
        """

        for key, array in input_dict.items():
            try:
                array = asarray(array, dtype=self.precision)
            except ValueError:
                raise ValueError(
                    f"Forcing array {key} could not be converted "
                    f"to a NumPy array."
                )
            if array.ndim != 1:
                raise ValueError(
                    f"Forcing array {key} must be one-dimensional."
                )
            input_dict[key] = array
        input_vectors = list(input_dict.values())
        if not all(
            array.shape[0] == input_vectors[0].shape[0]
            for array in input_vectors
        ):
            raise ValueError(
                "All forcing vectors must have the same length / be sampled "
                "on the same grid",
            )
        input_array = column_stack(input_vectors)
        if input_array.shape[0] < self.order + 1:
            raise ValueError(
                "At least order + 1 samples are required to construct"
                " splines.",
            )
        return input_array

    def _validate_time_inputs(
        self, time_dict: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Process and check time inputs.

        Parameters
        ----------
        time_dict
            Dictionary of time-related user inputs. If
            "driver_sample_period" is provided, then this will be used
            and "t0" will be fetched from the dict or default to 0.0.
            If "time" is provided, the sample period will be calculated
            as the difference between samples, and t0 as
            time_dict['time'][0].
        Returns
        -------
        tuple (float, float)
            Sample period and t0, either obtained directly from
            time_dict or computed from a "time" array.
        Raises
        ------
        ValueError
            Raised if the time array is not strictly increasing or the
            spacing between samples is non-uniform.
        """

        if ("driver_sample_period" in time_dict) and (
            "time" in time_dict
        ):
            raise ValueError(
                "Only one of driver_sample_period or time should be "
                "provided."
            )
        if "driver_sample_period" in time_dict:
            sample_period = time_dict["driver_sample_period"]
            t0 = time_dict.get("t0", 0.0)
        elif "time" in time_dict:
            timeArray = time_dict["time"]
            if timeArray.ndim != 1:
                raise ValueError("Time array must be one-dimensional.")
            if timeArray.shape[0] != self.num_samples:
                raise ValueError(
                    "Time array length must match the number of"
                    " samples in provided input vectors."
                )
            t0 = timeArray[0]
            time_differences = diff(timeArray)
            if np_any(time_differences <= 0.0):
                raise ValueError("Time array must be strictly increasing.")
            if not allclose(
                time_differences,
                full_like(time_differences, time_differences[0]),
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError("Time array must be uniformly spaced.")
            sample_period = time_differences[0]
        else:
            raise ValueError(
                "Either a time array or driver_sample_period must be "
                "provided."
            )

        return sample_period, t0

    # ---------------------------------------------------------------------- #
    # Evaluation function machinery
    # ---------------------------------------------------------------------- #
    def build(self) -> Callable:
        """Compile device helpers and return them alongside host coefficients.

        Returns
        -------
        Callable
            Device function which evaluates input polynomials at a given time.
        """
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
            for input_index in range(num_inputs):
                acc = zero_value
                for k in range(int32(order), int32(-1), int32(-1)):
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

            for input_index in range(int32(num_inputs)):
                acc = zero_value
                for k in range(int32(order), int32(0), int32(-1)):
                    acc = (
                        acc * tau
                        + precision(k) * (coefficients[seg, input_index, k])
                    )
                out[input_index] = (
                    acc * inv_resolution if in_range else zero_value
                )

        # no cover: end
        cache = InterpolatorCache(
            evaluation_function=evaluate_all,
            driver_del_t=evaluate_time_derivative,
        )
        return cache

    def update(
        self,
        updates_dict: Optional[Dict[str, object]] = None,
        silent: bool = False,
        **kwargs: object,
    ) -> Set[str]:
        """Apply configuration updates and invalidate caches when needed.

        Parameters
        ----------
        updates_dict
            Mapping of configuration keys to their new values.
        silent
            When ``True``, suppress warnings about inapplicable keys.
        **kwargs
            Additional configuration updates supplied inline.

        Returns
        -------
        set
            Set of configuration keys that were recognized and updated.

        Raises
        ------
        KeyError
            Raised when an unknown key is provided while ``silent`` is False.

        Notes
        -----
        Updates naming a key in ``config_keys`` rederive
        ``num_segments`` from the resulting wrap and boundary
        condition; keys absent from the update keep their current
        values. Changed settings recompute the host coefficients so
        they always match the configuration captured by the compiled
        device evaluators.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        initial_hash = self.compile_settings.values_hash
        recognised = self.update_compile_settings(updates_dict, silent=True)
        unrecognised = set(updates_dict.keys()) - recognised

        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )

        driver_config = {
            key: updates_dict[key]
            for key in recognised
            if key in self.config_keys
        }
        if driver_config:
            driver_config.setdefault(
                "boundary_condition", self.boundary_condition
            )
            self._derive_segment_settings(driver_config)
            self.update_compile_settings(driver_config, silent=True)
        if self.compile_settings.values_hash != initial_hash:
            self._coefficients = self._compute_coefficients()

        return recognised

    @property
    def evaluation_function(self) -> Callable:
        """Device function for evaluating all inputs."""
        return self.get_cached_output("evaluation_function")

    @property
    def driver_del_t(self) -> Callable:
        """Device function returning the interpolated driver time derivative."""

        return self.get_cached_output("driver_del_t")

    @property
    def coefficients(self) -> FloatArray:
        """Return the host-side coefficients array."""
        return self._coefficients

    @property
    def coefficients_shape(self) -> Tuple[int, int, int]:
        """Exact coefficient layout captured by the device evaluators."""
        return (self.num_segments, self.num_inputs, self.order + 1)

    # ---------------------------------------------------------------------- #
    # Inspection interface
    # ---------------------------------------------------------------------- #
    def get_input_array(self) -> FloatArray:
        """Return the input array."""
        return self._input_array

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
        RuntimeError
            Raised when interpolation coefficients are unavailable.
        """

        times = asarray(eval_times, dtype=self.precision)
        if times.ndim != 1:
            raise ValueError("eval_times must be one-dimensional.")

        num_points = times.size
        if num_points == 0:
            return empty((0, self.num_inputs), dtype=self.precision)

        coefficients = self.coefficients
        if coefficients is None:
            raise RuntimeError(
                "Interpolation coefficients have not been generated."
            )

        device_eval = self.evaluation_function

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
    # System-specific interface
    # ---------------------------------------------------------------------- #
    @staticmethod
    def check_against_system_drivers(
        inputs_dict: Dict[str, Union[float, bool, FloatArray]],
        system: "SymbolicODE",
    ) -> Dict[str, Union[float, bool, FloatArray]]:
        """Validate input keys and order driver columns by declared order.

        The interpolator stacks driver arrays into columns in dictionary
        insertion order and the compiled kernel reads those columns
        positionally against the system's declared driver order, so the
        driver entries are reordered to match the system before the
        interpolator consumes them.

        Parameters
        ----------
        inputs_dict
            Dictionary of input arrays to validate against the system.
        system
            SymbolicODE instance defining the expected driver symbols.

        Returns
        -------
        dict
            Copy of ``inputs_dict`` with driver entries reordered to the
            system's declared driver order, followed by the remaining
            configuration and timing entries in their original order.

        Raises
        ------
        ValueError
            Raised when the number of inputs does not match the number of
            drivers, or when input symbols do not match driver symbols.
        """
        input_keys = [
            key
            for key in inputs_dict
            if key
            not in (
                ArrayInterpolator.config_keys + ArrayInterpolator.time_info
            )
        ]
        driver_order = list(system.indices.driver_names)
        system_driver_keys = set(driver_order)
        if len(input_keys) != system.num_drivers:
            raise ValueError(
                f"Number of inputs in inputs_dict "
                f"({len(input_keys)}) does not match number of "
                f"drivers in system ({system.num_drivers})."
            )
        if set(input_keys) != system_driver_keys:
            raise ValueError(
                f"input symbols in inputs_dict ("
                f"{set(input_keys)}) do not match drivers "
                f"symbols in system ({system_driver_keys})."
            )

        ordered = {name: inputs_dict[name] for name in driver_order}
        for key, value in inputs_dict.items():
            if key not in ordered:
                ordered[key] = value
        return ordered

    # ---------------------------------------------------------------------- #
    # Spline coefficient generation
    # ---------------------------------------------------------------------- #

    def _compute_coefficients(self) -> FloatArray:
        """Return spline coefficients respecting the requested boundary.

        Returns
        -------
        numpy.ndarray
            Segment-major coefficient array of shape ``(num_segments,
            num_inputs, order + 1)``.

        Raises
        ------
        ValueError
            Raised when periodic constraints are incompatible with the
            input configuration.
        """
        boundary_condition = self.boundary_condition

        precision = self.precision
        base_inputs = self.input_array.astype(precision, copy=False)
        num_inputs = self.num_inputs
        order = self.order

        pad_with_zeros = (not self.wrap) and boundary_condition == "clamped"
        if pad_with_zeros:
            zero_row = zeros((1, num_inputs), dtype=precision)
            inputs = vstack((zero_row, base_inputs, zero_row))
        else:
            inputs = base_inputs

        num_segments = inputs.shape[0] - 1

        if boundary_condition == "periodic":
            if not self.wrap:
                raise ValueError(
                    "Periodic boundary conditions require wrap=True so that "
                    "the input repeats after the final segment."
                )
            if not allclose(inputs[0], inputs[-1]):
                raise ValueError(
                    "Periodic boundary conditions require the first and "
                    "last samples to match."
                )

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

                # Enforce vanishing forward difference at the start of the grid.
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
        return self._land_coefficients(coefficients)

    def _land_coefficients(self, coefficients: FloatArray) -> FloatArray:
        """Copy coefficients into a reused pinned-or-pageable buffer."""
        buffer = self._coefficients
        if (
            buffer is None
            or buffer.shape != coefficients.shape
            or buffer.dtype != self.precision
        ):
            buffer = self._memory_manager.create_host_array(
                coefficients.shape, self.precision, "pinned"
            )
        buffer[...] = coefficients
        return buffer

    # ---------------------------------------------------------------------- #
    # Getters and pass-through
    # ---------------------------------------------------------------------- #

    @property
    def num_inputs(self) -> int:
        """Return the number of input signals."""
        return self.input_array.shape[1]

    @property
    def num_samples(self) -> int:
        """Number of samples available for interpolation."""
        return self.input_array.shape[0]

    @property
    def input_array(self) -> FloatArray:
        """Return the normalised input array."""
        return self._input_array

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
