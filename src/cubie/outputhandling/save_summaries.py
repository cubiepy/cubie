"""Factory for CUDA device functions that persist summary metrics.

Published Functions
-------------------
:func:`save_summary_factory`
    Build a CUDA device function that exports accumulated summary
    metrics from working buffers to output arrays.

:func:`chain_metrics`
    Recursively compose metric save functions into a single callable
    (internal).

See Also
--------
:func:`~cubie.outputhandling.update_summaries.update_summary_factory`
    Companion factory that accumulates metrics during integration.
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
from cubie.cuda_simsafe import unroll_if
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
    buffer,
    output,
    summarise_every,
):
    """Provide a no-op device function for empty metric chains.

    Parameters
    ----------
    buffer
        device array slice containing accumulated metric values (unused).
    output
        device array slice that would receive saved results (unused).
    summarise_every
        Integer interval between summary exports (unused).

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
    output_offsets: Sequence[int],
    output_sizes: Sequence[int],
    function_params: Sequence[object],
    inner_chain: Callable = do_nothing,
    lineinfo: Optional[bool] = None,
) -> Callable:
    """
    Recursively chain summary metric functions for CUDA execution.

    This function builds a recursive chain of summary metric functions,
    where each function in the sequence is wrapped with the previous
    functions to create a single callable that executes all metrics.

    Parameters
    ----------
    metric_functions
        Sequence of CUDA device functions that save summary metrics.
    buffer_offsets
        Sequence of offsets into the accumulation buffer for each metric.
    buffer_sizes
        Sequence of per-metric buffer lengths.
    output_offsets
        Sequence of offsets into the output window for each metric.
    output_sizes
        Sequence of per-metric output lengths.
    function_params
        Sequence of parameter payloads passed to each metric function.
    inner_chain
        Callable executed before the current metric; defaults to
        ``do_nothing``.

    Returns
    -------
    Callable
        CUDA device function that executes all chained metrics.

    Notes
    -----
    The function uses recursion to build a chain where each level executes
    the inner chain first, then the current metric function. This ensures
    all requested metrics are computed in the correct order.
    """
    if len(metric_functions) == 0:
        return do_nothing
    current_metric_fn = metric_functions[0]
    current_buffer_offset = buffer_offsets[0]
    current_buffer_size = buffer_sizes[0]
    current_output_offset = output_offsets[0]
    current_output_size = output_sizes[0]
    current_metric_param = function_params[0]

    remaining_metric_fns = metric_functions[1:]
    remaining_buffer_offsets = buffer_offsets[1:]
    remaining_buffer_sizes = buffer_sizes[1:]
    remaining_output_offsets = output_offsets[1:]
    remaining_output_sizes = output_sizes[1:]
    remaining_metric_params = function_params[1:]

    # no cover: start
    @cuda.jit(
        device=True,
        inline=True,
        **get_jit_kwargs(lineinfo),
    )
    def wrapper(
        buffer,
        output,
        summarise_every,
    ):
        """Apply the accumulated metric chain before invoking the current
        metric.

        Parameters
        ----------
        buffer
            device array slice holding accumulated metric state.
        output
            device array slice that receives exported summary values.
        summarise_every
            Integer interval between summary exports passed along the chain.

        Returns
        -------
        None
            The device function mutates the provided output window in place.
        """
        inner_chain(
            buffer,
            output,
            summarise_every,
        )
        current_metric_fn(
            buffer[
                current_buffer_offset : current_buffer_offset
                + current_buffer_size
            ],
            output[
                current_output_offset : current_output_offset
                + current_output_size
            ],
            summarise_every,
            current_metric_param,
        )

    if remaining_metric_fns:
        return chain_metrics(
            remaining_metric_fns,
            remaining_buffer_offsets,
            remaining_buffer_sizes,
            remaining_output_offsets,
            remaining_output_sizes,
            remaining_metric_params,
            wrapper,
            lineinfo=lineinfo,
        )
    else:
        return wrapper
    # no cover: stop


def save_summary_factory(
    summaries_buffer_height_per_var: int,
    summarised_state_indices: Union[Sequence[int], ArrayLike],
    summarised_observable_indices: Union[Sequence[int], ArrayLike],
    summaries_list: Sequence[str],
    lineinfo: Optional[bool] = None,
    unroll_other_small: bool = True,
) -> Callable:
    """
    Factory function for creating CUDA device functions to save summary
    metrics.

    This factory generates a CUDA device function that applies chained
    summary metric calculations to all requested state and observable
    variables.

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

    Returns
    -------
    Callable
        CUDA device function for saving summary metrics.

    Notes
    -----
    The generated function iterates through all specified state and observable
    variables, applying the chained summary metrics to each variable's buffer
    and saving results to the appropriate output arrays.
    """
    num_summarised_states = int32(len(summarised_state_indices))
    num_summarised_observables = int32(len(summarised_observable_indices))

    save_functions = summary_metrics.save_functions(summaries_list)

    buff_per_var = summaries_buffer_height_per_var
    total_buffer_size = int32(buff_per_var)
    total_output_size = int32(summary_metrics.summaries_output_height(
        summaries_list))

    buffer_offsets = summary_metrics.buffer_offsets(summaries_list)
    buffer_sizes_list = summary_metrics.buffer_sizes(summaries_list)
    output_offsets = summary_metrics.output_offsets(summaries_list)
    output_sizes = summary_metrics.output_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)
    num_summary_metrics = len(output_offsets)

    summarise_states = (num_summarised_states > 0) and (
        num_summary_metrics > 0
    )
    summarise_observables = (num_summarised_observables > 0) and (
        num_summary_metrics > 0
    )

    summary_metric_chain = chain_metrics(
        save_functions,
        buffer_offsets,
        buffer_sizes_list,
        output_offsets,
        output_sizes,
        params,
        lineinfo=lineinfo,
    )

    # no cover: start
    @cuda.jit(
        device=True,
        inline=True,
        **get_jit_kwargs(lineinfo),
    )
    def save_summary_metrics_func(
        buffer_state_summaries,
        buffer_observable_summaries,
        output_state_summaries_window,
        output_observable_summaries_window,
        summarise_every,
    ):
        """Export summary metrics from accumulation buffers to output windows.

        Parameters
        ----------
        buffer_state_summaries
            device array slice holding accumulated state summary data.
        buffer_observable_summaries
            device array slice holding accumulated observable summary data.
        output_state_summaries_window
            device array slice that receives state summary results.
        output_observable_summaries_window
            device array slice that receives observable summary results.
        summarise_every
            Integer interval between summary exports.

        Returns
        -------
        None
            The device function mutates the provided output windows in place.

        Notes
        -----
        The chained metric function is executed for each selected state or
        observable entry, writing the requested metric results into contiguous
        regions of the output arrays.
        """
        if summarise_states:
            for state_index in unroll_if(
                range(num_summarised_states), unroll_other_small
            ):
                buffer_array_slice_start = state_index * total_buffer_size
                out_array_slice_start = state_index * total_output_size

                summary_metric_chain(
                    buffer_state_summaries[
                        buffer_array_slice_start : buffer_array_slice_start
                        + total_buffer_size
                    ],
                    output_state_summaries_window[
                        out_array_slice_start : out_array_slice_start
                        + total_output_size
                    ],
                    summarise_every,
                )

        if summarise_observables:
            for observable_index in unroll_if(
                range(num_summarised_observables), unroll_other_small
            ):
                buffer_array_slice_start = observable_index * total_buffer_size
                out_array_slice_start = observable_index * total_output_size

                summary_metric_chain(
                    buffer_observable_summaries[
                        buffer_array_slice_start : buffer_array_slice_start
                        + total_buffer_size
                    ],
                    output_observable_summaries_window[
                        out_array_slice_start : out_array_slice_start
                        + total_output_size
                    ],
                    summarise_every,
                )

    # no cover: stop
    return save_summary_metrics_func
