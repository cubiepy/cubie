"""Factory for CUDA device functions that save solver state.

Published Functions
-------------------
:func:`save_state_factory`
    Build a CUDA device function that copies selected state, observable,
    time, and counter values into output buffers during integration.

See Also
--------
:class:`~cubie.outputhandling.output_functions.OutputFunctions`
    Caller that invokes this factory during compilation.
:func:`~cubie.outputhandling.update_summaries.update_summary_factory`
    Companion factory for accumulating summary metrics.
"""

from typing import Callable, Optional, Sequence, Union

from cubie.cuda_simsafe import cuda, int32
from cubie.cuda_simsafe import UnrollFlag, unroll_if
from numpy.typing import ArrayLike

from cubie.cuda_simsafe import get_jit_kwargs, stwt


def save_state_factory(
    saved_state_indices: Union[Sequence[int], ArrayLike],
    saved_observable_indices: Union[Sequence[int], ArrayLike],
    save_state: bool,
    save_observables: bool,
    save_time: bool,
    save_counters: bool = False,
    lineinfo: Optional[bool] = None,
    unroll: UnrollFlag = (False, None),
) -> Callable:
    """Build a CUDA device function that stores solver state and observables.

    Parameters
    ----------
    saved_state_indices
        Sequence of state indices to write into the state output window.
    saved_observable_indices
        Sequence of observable indices to write into the observable output
        window.
    save_state
        When ``True`` the generated function copies the current state slice.
    save_observables
        When ``True`` the generated function copies the current observable
        slice.
    save_time
        When ``True`` the generated function appends the current step to the
        end of the state output window.
    save_counters
        When ``True`` the generated function writes iteration counters to
        the output.
    lineinfo
        Compile with source-line correlation data. ``None`` defers to the
        ``CUBIE_LINEINFO`` environment variable.
    unroll
        Save loop flag.

    Returns
    -------
    Callable
        CUDA device function that writes state, observable, and optional time
        values into contiguous output buffers.

    Notes
    -----
    The generated device function expects ``current_state``,
    ``current_observables``, ``output_states_slice``,
    ``output_observables_slice``, ``current_step``, ``output_counters_slice``,
    and ``counters_array`` arguments. The function mutates
    ``output_states_slice``, ``output_observables_slice``, and
    ``output_counters_slice`` in place.
    """
    # Extract sizes from heights object
    nobs = int32(len(saved_observable_indices))
    nstates = int32(len(saved_state_indices))
    ncounters = int32(4)

    # no cover: start
    @cuda.jit(
        device=True,
        inline=True,
        **get_jit_kwargs(lineinfo),
    )
    def save_state_func(
        current_state,
        current_observables,
        current_counters,
        current_step,
        output_states_slice,
        output_observables_slice,
        output_counters_slice,
    ):
        """Write selected state, observable, and time values to device buffers.

        Parameters
        ----------
        current_state
            device array containing the latest integrator state values.
        current_observables
            device array containing the latest observable values.
        current_counters
            device array containing iteration counter values to save.
        current_step
            Scalar step or time value associated with the current sample.
        output_states_slice
            device array window that receives saved state (and optional time)
            values in place.
        output_observables_slice
            device array window that receives saved observable values in
            place.
        output_counters_slice
            device array window that receives iteration counter values in
            place.

        Returns
        -------
        None
            The device function mutates the provided output buffers in place.

        Notes
        -----
        When ``save_time`` is ``True`` the current step value is stored at the
        first slot immediately after the copied state values.
        """
        if save_state:
            for k in unroll_if(range(nstates), unroll):
                stwt(
                    output_states_slice,
                    k,
                    current_state[saved_state_indices[k]],
                )
        if save_time:
            # Append time at the end of the state output
            stwt(output_states_slice, nstates, current_step)
        if save_observables:
            for m in unroll_if(range(nobs), unroll):
                stwt(
                    output_observables_slice,
                    m,
                    current_observables[saved_observable_indices[m]],
                )
        if save_counters:
            for i in unroll_if(range(ncounters), unroll):
                stwt(output_counters_slice, i, current_counters[i])
        # no cover: stop

    return save_state_func
