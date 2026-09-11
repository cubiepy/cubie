"""Sizing helpers for output arrays.

The classes in this module compute output array shapes needed for CUDA
batch solving host-visible output layouts. Each class inherits from
:class:`ArraySizingClass`, which offers a utility for coercing zero-sized
buffers to a minimum of one element for safe allocation.

Published Classes
-----------------
:class:`ArraySizingClass`
    Abstract base providing a :pyattr:`nonzero` property for safe
    allocation.

:class:`OutputArrayHeights`
    Heights of time-series and summary outputs.

    >>> heights = OutputArrayHeights(state=10, observables=5)
    >>> heights.state
    10

:class:`SingleRunOutputSizes`
    Output array sizes for a single integration run.

:class:`BatchInputSizes`
    Input array sizes for batch integration runs.

:class:`BatchOutputSizes`
    Output array sizes for batch integration runs.

See Also
--------
:class:`~cubie.outputhandling.output_functions.OutputFunctions`
    Factory that produces :class:`OutputArrayHeights` from its
    configuration.
:class:`~cubie.integrators.loops.ode_loop.IVPLoop`
    Internal loop buffer sizing.
"""

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
    from cubie.outputhandling.output_functions import OutputFunctions

from abc import ABC

import attrs

from cubie._utils import ensure_nonzero_size


@attrs.define
class ArraySizingClass(ABC):
    """Base class for output sizing helpers.

    Notes
    -----
    All subclasses inherit the :pyattr:`nonzero` property, which ensures that
    every integer or tuple size has a minimum length of one so host-side
    allocation code can safely request buffers.
    """

    @property
    def nonzero(self) -> "ArraySizingClass":
        """Return a copy with all sizes expanded to at least one element.

        Returns
        -------
        ArraySizingClass
            A new object with every integer and tuple size coerced to a
            minimum of one.

        Notes
        -----
        CUDA allocators cannot handle zero-length buffers, so callers should
        use this property before preallocating device or host memory from
        sizing data.
        """
        new_obj = attrs.evolve(self)
        for field in attrs.fields(self.__class__):
            value = getattr(new_obj, field.name)
            if isinstance(value, (int, tuple)):
                setattr(new_obj, field.name, ensure_nonzero_size(value))
        return new_obj


@attrs.define
class OutputArrayHeights(ArraySizingClass):
    """Heights of time-series and summary outputs.

    Attributes
    ----------
    state : int, default 1
        Height of state output arrays, including a slot for time stamps when
        requested.
    observables : int, default 1
        Height of observable output arrays.
    state_summaries : int, default 1
        Height of state summary outputs.
    observable_summaries : int, default 1
        Height of observable summary outputs.
    per_variable : int, default 1
        Height reserved per tracked variable for summary outputs.
    """

    state: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    observables: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    state_summaries: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    observable_summaries: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )
    per_variable: int = attrs.field(
        default=1, validator=attrs.validators.instance_of(int)
    )

    @classmethod
    def from_output_fns(
        cls, output_fns: "OutputFunctions"
    ) -> "OutputArrayHeights":
        """Compute array heights from configured output functions.

        Parameters
        ----------
        output_fns
            Output function factory describing which values are saved and how
            summaries are aggregated.

        Returns
        -------
        OutputArrayHeights
            Array heights derived from the output configuration.

        Notes
        -----
        The state output height reserves an extra row when time saving is
        enabled so that timestamps align with the saved states.
        """
        state = output_fns.n_saved_states + 1 * output_fns.save_time
        observables = output_fns.n_saved_observables
        state_summaries = output_fns.state_summaries_output_height
        observable_summaries = output_fns.observable_summaries_output_height
        per_variable = output_fns.summaries_output_height_per_var
        obj = cls(
            state,
            observables,
            state_summaries,
            observable_summaries,
            per_variable,
        )
        return obj


@attrs.define
class SingleRunOutputSizes(ArraySizingClass):
    """Output array sizes for a single integration run.

    This class provides 2D array sizes (time × variable) for output arrays
    from a single integration run.

    Attributes
    ----------
    state : tuple[int, int], default (1, 1)
        Shape of state output array as (time_samples, n_variables).
    observables : tuple[int, int], default (1, 1)
        Shape of observable output array as (time_samples, n_variables).
    state_summaries : tuple[int, int], default (1, 1)
        Shape of state summary array as (summary_samples, n_summaries).
    observable_summaries : tuple[int, int], default (1, 1)
        Shape of observable summary array as (summary_samples, n_summaries).
    """

    state: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    observables: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    state_summaries: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    observable_summaries: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "SingleRunOutputSizes":
        """Transform solver metadata into single-run output shapes.

        Parameters
        ----------
        solver_instance
            Batch solver kernel exposing ``output_array_heights``,
            ``output_length``, and ``summaries_length`` attributes.

        Returns
        -------
        SingleRunOutputSizes
            Array shapes for one simulation run.
        """
        heights = solver_instance.output_array_heights
        output_samples = solver_instance.output_length
        summarise_samples = solver_instance.summaries_length

        state = (output_samples, heights.state)
        observables = (output_samples, heights.observables)
        state_summaries = (summarise_samples, heights.state_summaries)
        observable_summaries = (
            summarise_samples,
            heights.observable_summaries,
        )
        obj = cls(
            state,
            observables,
            state_summaries,
            observable_summaries,
        )

        return obj


@attrs.define
class BatchInputSizes(ArraySizingClass):
    """Input array sizes for batch integration runs.

    This class specifies the sizes of input arrays needed for batch
    processing, including initial conditions, parameters, and forcing terms.

    Attributes
    ----------
    initial_values : tuple[int, int], default (1, 1)
        Shape of initial values array as (n_states, n_runs).
    parameters : tuple[int, int], default (1, 1)
        Shape of parameters array as (n_parameters, n_runs).
    driver_coefficients : tuple[int, int, int], default (1, 1, 1)
        Shape of the driver coefficient array as
        (num_segments, num_drivers, order + 1), pinned from the
        driver interpolator's compiled layout.
    """

    initial_values: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    parameters: Tuple[int, int] = attrs.field(
        default=(1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    driver_coefficients: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple)
    )

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "BatchInputSizes":
        """Create batch input shapes based on solver metadata.

        Parameters
        ----------
        solver_instance
            Batch solver kernel exposing ``num_runs`` and system sizes.

        Returns
        -------
        BatchInputSizes
            Input array dimensions for the batch run.
        """
        system_sizes = solver_instance.system_sizes
        num_runs = solver_instance.num_runs
        initial_values = (system_sizes.states, num_runs)
        parameters = (system_sizes.parameters, num_runs)
        driver_coefficients = solver_instance.coefficients_shape
        obj = cls(initial_values, parameters, driver_coefficients)
        return obj


@attrs.define
class BatchOutputSizes(ArraySizingClass):
    """Output array sizes for batch integration runs.

    This class provides 3D array sizes (time × variable × run) for output
    arrays from batch integration runs.

    Attributes
    ----------
    state : tuple[int, int, int], default (1, 1, 1)
        Shape of state output array as (time_samples, n_variables, n_runs).
    observables : tuple[int, int, int], default (1, 1, 1)
        Shape of observable output array as (time_samples, n_variables,
        n_runs).
    state_summaries : tuple[int, int, int], default (1, 1, 1)
        Shape of state summary array as (summary_samples, n_summaries,
        n_runs).
    observable_summaries : tuple[int, int, int], default (1, 1, 1)
        Shape of observable summary array as (summary_samples, n_summaries,
        n_runs).
    status_codes : tuple[int], default (1,)
        Shape of the status code output array indexed by run.
    iteration_counters : tuple[int, int, int], default (1, 4, 1)
        Shape of the iteration counter array as
        (time_samples, counter_channels, n_runs). The four channels
        are Newton iterations, Krylov iterations, accepted steps,
        and rejected steps.
    """

    state: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    observables: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    state_summaries: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    observable_summaries: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple)
    )
    status_codes: Tuple[int] = attrs.field(
        default=(1,), validator=attrs.validators.instance_of(Tuple)
    )
    iteration_counters: Tuple[int, int, int] = attrs.field(
        default=(1, 4, 1), validator=attrs.validators.instance_of(Tuple)
    )

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "BatchOutputSizes":
        """Lift single-run shapes to batched output arrays.

        Parameters
        ----------
        solver_instance
            Batch solver kernel exposing ``num_runs`` and single-run sizing
            helpers.

        Returns
        -------
        BatchOutputSizes
            Output array dimensions for the batch run.

        Notes
        -----
        Builds 3D arrays by pairing the number of runs with single-run heights
        for each data category.
        """
        single_run_sizes = SingleRunOutputSizes.from_solver(solver_instance)
        num_runs = solver_instance.num_runs
        state = (
            single_run_sizes.state[0],
            single_run_sizes.state[1],
            num_runs,
        )
        observables = (
            single_run_sizes.observables[0],
            single_run_sizes.observables[1],
            num_runs,
        )
        state_summaries = (
            single_run_sizes.state_summaries[0],
            single_run_sizes.state_summaries[1],
            num_runs,
        )
        observable_summaries = (
            single_run_sizes.observable_summaries[0],
            single_run_sizes.observable_summaries[1],
            num_runs,
        )
        status_codes = (num_runs,)

        # Iteration counters have shape (n_saves, 4, n_runs)
        # where 4 is for [Newton, Krylov, steps, rejections].
        # When counters are not requested the size collapses to a
        # placeholder, like every other inactive output.
        if solver_instance.save_counters:
            iteration_counters = (
                single_run_sizes.state[0],  # n_saves
                4,
                num_runs,
            )
        else:
            iteration_counters = (0, 0, 0)

        obj = cls(
            state,
            observables,
            state_summaries,
            observable_summaries,
            status_codes,
            iteration_counters,
        )
        return obj
