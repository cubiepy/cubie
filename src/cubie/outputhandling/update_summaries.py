"""Factory for CUDA device functions that accumulate summary metrics.

Published Functions
-------------------
:func:`update_summary_factory`
    Build a CUDA device function that accumulates summary metrics
    from the current state sample into working buffers.

:func:`chain_metrics`
    Recursively compose metric update functions into a single
    callable (internal).

See Also
--------
:func:`~cubie.outputhandling.save_summaries.save_summary_factory`
    Companion factory that persists accumulated metrics.
:class:`~cubie.outputhandling.output_functions.OutputFunctions`
    Caller that invokes this factory during compilation.

Notes
-----
Based on the recursive chain approach by sklam
(https://github.com/numba/numba/issues/3405) for composing
JIT-compiled functions without passing them as an iterable.
"""

from typing import Callable, Optional, Sequence, Union

from cubie.cuda_simsafe import cuda, int32
from cubie.cuda_simsafe import UnrollFlag, unroll_if
from numpy.typing import ArrayLike

from cubie.cuda_simsafe import compile_kwargs, get_jit_kwargs
from cubie.outputhandling.summarymetrics import summary_metrics


# no cover: start
@cuda.jit(
    device=True,
    inline=True,
    **compile_kwargs,
)
def do_nothing(
    values,
    buffer,
    current_step,
):
    """Provide a no-op device function for empty metric chains.

    Parameters
    ----------
    values
        device array containing the current scalar value (unused).
    buffer
        device array slice reserved for summary accumulation (unused).
    current_step
        Integer or scalar step identifier (unused).

    Notes
    -----
    Base case for the recursive chain when no summary metrics are
    configured.
    """
    pass
# no cover: end


def chain_metrics(
    metric_functions: Sequence[Callable],
    buffer_offsets: Sequence[int],
    buffer_sizes: Sequence[int],
    function_params: Sequence[object],
    inner_chain: Callable = do_nothing,
    lineinfo: Optional[bool] = None,
) -> Callable:
    """
    Recursively chain summary metric update functions for CUDA execution.

    This function builds a recursive chain of summary metric update functions,
    where each function in the sequence is wrapped with the previous
    functions to create a single callable that updates all metrics.

    Parameters
    ----------
    metric_functions
        Sequence of CUDA device functions for updating summary metrics.
    buffer_offsets
        Sequence of offsets into the metric buffer for each function.
    buffer_sizes
        Sequence of per-metric buffer lengths.
    function_params
        Sequence of parameter payloads passed to each metric function.
    inner_chain
        Callable executed before the current metric; defaults to
        ``do_nothing``.

    Returns
    -------
    Callable
        CUDA device function that executes all chained metric updates.

    Notes
    -----
    The function uses recursion to build a chain where each level executes
    the inner chain first, then the current metric update function. This
    ensures all requested metrics are updated in the correct order during
    each integration step.
    """
    if len(metric_functions) == 0:
        return do_nothing

    current_fn = metric_functions[0]
    current_offset = buffer_offsets[0]
    current_size = buffer_sizes[0]
    current_param = function_params[0]

    remaining_functions = metric_functions[1:]
    remaining_offsets = buffer_offsets[1:]
    remaining_sizes = buffer_sizes[1:]
    remaining_params = function_params[1:]

    # no cover: start
    @cuda.jit(
        device=True,
        inline=True,
        **get_jit_kwargs(lineinfo),
    )
    def wrapper(
        value,
        buffer,
        current_step,
    ):
        """Apply the accumulated metric chain before invoking the current
        metric.

        Parameters
        ----------
        value
            device array element being summarised.
        buffer
            device array slice containing the metric working storage.
        current_step
            Integer or scalar step identifier passed through the chain.

        Returns
        -------
        None
            The device function mutates the metric buffer in place.
        """
        inner_chain(value, buffer, current_step)
        current_fn(
            value,
            buffer[current_offset : current_offset + current_size],
            current_step,
            current_param,
        )

    if remaining_functions:
        return chain_metrics(
            remaining_functions,
            remaining_offsets,
            remaining_sizes,
            remaining_params,
            wrapper,
            lineinfo=lineinfo,
        )
    else:
        return wrapper
    # no cover: stop


def update_summary_factory(
    summaries_buffer_height_per_var: int,
    summarised_state_indices: Union[Sequence[int], ArrayLike],
    summarised_observable_indices: Union[Sequence[int], ArrayLike],
    summaries_list: Sequence[str],
    lineinfo: Optional[bool] = None,
    unroll: UnrollFlag = (False, None),
) -> Callable:
    """
    Factory function for creating CUDA device functions to update summary
    metrics.

    This factory generates an optimized CUDA device function that applies
    chained summary metric updates to all requested state and observable
    variables during each integration step.

    Parameters
    ----------
    summaries_buffer_height_per_var
        Number of buffer slots required per tracked variable.
    summarised_state_indices
        Sequence of state indices to include in summary calculations.
    summarised_observable_indices
        Sequence of observable indices to include in summary calculations.
    summaries_list
        Ordered list of summary metric identifiers registered with
        :mod:`cubie.outputhandling.summarymetrics`.
    lineinfo
        Compile with source-line correlation data. ``None`` defers to the
        ``CUBIE_LINEINFO`` environment variable.
    unroll
        ``(unroll, count)`` flag of the per-variable loops.

    Returns
    -------
    Callable
        CUDA device function for updating summary metrics.

    Notes
    -----
    The generated function iterates through all specified state and observable
    variables, applying the chained summary metric updates to accumulate data
    in the appropriate buffer locations during each integration step.
    """
    unroll_other_small = unroll
    num_summarised_states = int32(len(summarised_state_indices))
    num_summarised_observables = int32(len(summarised_observable_indices))
    buff_per_var = summaries_buffer_height_per_var
    total_buffer_size = int32(buff_per_var)
    buffer_offsets = summary_metrics.buffer_offsets(summaries_list)
    num_metrics = len(buffer_offsets)

    summarise_states = (num_summarised_states > 0) and (num_metrics > 0)
    summarise_observables = (num_summarised_observables > 0) and (
        num_metrics > 0
    )

    update_fns = summary_metrics.update_functions(summaries_list)
    buffer_sizes_list = summary_metrics.buffer_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)
    chain_fn = chain_metrics(
        update_fns, buffer_offsets, buffer_sizes_list, params,
        lineinfo=lineinfo,
    )

    # no cover: start
    @cuda.jit(
        device=True,
        inline=True,
        **get_jit_kwargs(lineinfo),
    )
    def update_summary_metrics_func(
        current_state,
        current_observables,
        state_summary_buffer,
        observable_summary_buffer,
        current_step,
    ):
        """Accumulate summary metrics from the current state sample.

        Parameters
        ----------
        current_state
            device array holding the latest integrator state values.
        current_observables
            device array holding the latest observable values.
        state_summary_buffer
            device array slice used to accumulate state summary data.
        observable_summary_buffer
            device array slice used to accumulate observable summary data.
        current_step
            Integer or scalar step identifier associated with the sample.

        Returns
        -------
        None
            The device function mutates the supplied summary buffers in place.

        Notes
        -----
        The chained metric function is executed for each selected state or
        observable entry, writing into the contiguous buffer segment assigned
        to that variable.
        """
        if summarise_states:
            for idx in unroll_if(range(num_summarised_states), unroll_other_small):
                start = idx * total_buffer_size
                end = start + total_buffer_size
                chain_fn(
                    current_state[summarised_state_indices[idx]],
                    state_summary_buffer[start:end],
                    current_step,
                )

        if summarise_observables:
            for idx in unroll_if(range(num_summarised_observables), unroll_other_small):
                start = idx * total_buffer_size
                end = start + total_buffer_size
                chain_fn(
                    current_observables[summarised_observable_indices[idx]],
                    observable_summary_buffer[start:end],
                    current_step,
                )

    # no cover: stop
    return update_summary_metrics_func
