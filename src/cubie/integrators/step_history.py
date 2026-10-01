"""CUDA factory for the accepted-state history of BDF steps.

The history commits a proposal at the start of the next step, once the
loop reports it accepted.

Published Classes
-----------------
:class:`StepHistoryConfig`
    Attrs compile settings for the history factory.

:class:`StepHistory`
    CUDAFactory building the history device function.

See Also
--------
:class:`~cubie.integrators.algorithms.generic_bdf.BDFStep`
    Owns a history and solves the corrector it describes.
"""

from typing import Any, Callable, Dict, Set

from attrs import define, field, frozen, validators
from numpy import asarray as np_asarray
from numpy import int32 as np_int32
from numpy import ndarray as np_ndarray

from cubie._utils import PrecisionDType, getype_validator, is_device_validator
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import (
    CUDADispatcherCache,
    CUDAFactory,
    CUDAFactoryConfig,
    build_config,
)
from numba_cuda_mlir.types import int32
from cubie._cudasim_extensions import cuda
from cubie.backend.intrinsics import unroll_if
from cubie.integrators.algorithms.generic_bdf_tableaus import BDFTableau


@frozen
class StepHistoryConfig(CUDAFactoryConfig):
    """Compile settings for :class:`StepHistory`.

    Attributes
    ----------
    n_states : int
        Number of state variables per stored state.
    tableau : BDFTableau
        Formula whose order sizes the history.
    history_values_location : str
        Memory location of the stored states.
    history_intervals_location : str
        Memory location of the stored step sizes.
    """

    n_states: int = field(default=1, validator=getype_validator(int, 1))
    tableau: BDFTableau = field(
        default=None,
        validator=validators.instance_of(BDFTableau),
    )
    history_values_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    history_intervals_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )

    @property
    def max_order(self) -> int:
        """Return the highest order the history runs."""
        return self.tableau.order

    @property
    def history_length(self) -> int:
        """Return the number of stored states."""
        return self.tableau.history_length

    @property
    def ratio_limits(self) -> np_ndarray:
        """Return each order's step-ratio limit in precision."""
        return np_asarray(self.tableau.ratio_limits, dtype=self.precision)


@define
class StepHistoryCache(CUDADispatcherCache):
    """Hold the compiled history device function."""

    history_fn: Callable = field(validator=is_device_validator)


class StepHistory(CUDAFactory):
    """Compile the history update and BDF coefficient device function.

    The compiled device function has signature::

        history(state, step_size, first_step, accepted, base_state,
                prediction, shared, persistent_local)
            -> (corrector_step, error_scale, restart)

    Slot ``j`` holds the state ``j`` accepted steps back; interval
    ``j`` ends at slot ``j - 1``; interval 0 is the last attempt. A step
    longer than ``ratio_limits[0]`` times interval 1 drops slot 1.

    The order is the largest with ``order + 1`` stored states and every
    step ratio across them within ``ratio_limits``. The new state ``y``
    solves ``M (y - base_state) = corrector_step * f(y)``;
    ``error_scale * (y - prediction)`` is the local error. With no
    qualifying order, ``restart`` is set, the order is one,
    ``prediction`` is the start state (the caller adds an explicit
    Euler step) and ``error_scale`` is one half.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n_states: int,
        tableau: BDFTableau,
        **kwargs,
    ) -> None:
        """Initialise the history factory.

        Parameters
        ----------
        precision
            Floating-point precision of the stored states.
        n_states
            Number of state variables per stored state.
        tableau
            Formula whose order sizes the history.
        **kwargs
            Optional overrides for other compile settings. None
            values are ignored.
        """

        super().__init__()
        config = build_config(
            StepHistoryConfig,
            required={
                "precision": precision,
                "n_states": n_states,
                "tableau": tableau,
            },
            **kwargs,
        )
        self.setup_compile_settings(config)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register the history's buffers with the registry."""

        config = self.compile_settings
        history_length = int(config.history_length)
        buffer_registry.register(
            "history_values",
            self,
            history_length * int(config.n_states),
            config.history_values_location,
            persistent=True,
        )
        buffer_registry.register(
            "history_intervals",
            self,
            history_length,
            config.history_intervals_location,
            persistent=True,
        )
        buffer_registry.register(
            "history_count",
            self,
            1,
            "local",
            persistent=True,
            dtype=np_int32,
        )

    def _update(self, updates: Dict[str, Any], silent: bool) -> Set[str]:
        """Apply the settings and re-register the buffers.

        Parameters
        ----------
        updates
            Setting names to new values.
        silent
            Whether :meth:`update` ignores unrecognised names.

        Returns
        -------
        set[str]
            Names the history settings recognised.
        """
        recognised = self.update_compile_settings(updates, silent=True)
        if recognised:
            self.register_buffers()
        return recognised

    @property
    def device_function(self) -> Callable:
        """Return the compiled history device function."""

        return self.get_cached_output("history_fn")

    def build(self) -> StepHistoryCache:
        """Compile the history device function."""

        config = self.compile_settings
        numba_precision = config.numba_precision
        typed_zero = numba_precision(0.0)
        typed_half = numba_precision(0.5)
        typed_one = numba_precision(1.0)
        n = int32(config.n_states)
        max_order = int32(config.max_order)
        history_length = int32(config.history_length)
        ratio_limits = config.ratio_limits
        unroll_stage = config.unroll.unroll_stage
        unroll_step_element = config.unroll.unroll_step_element
        unroll_other_small = config.unroll.unroll_other_small

        getalloc = buffer_registry.get_allocator
        alloc_values = getalloc("history_values", self)
        alloc_intervals = getalloc("history_intervals", self)
        alloc_count = getalloc("history_count", self)

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def history(
            state,
            step_size,
            first_step,
            accepted,
            base_state,
            prediction,
            shared,
            persistent_local,
        ):
            values = alloc_values(shared, persistent_local)
            intervals = alloc_intervals(shared, persistent_local)
            count = alloc_count(shared, persistent_local)

            # ------------------------------------------------------- #
            #        Commit the previous step if it was accepted      #
            # ------------------------------------------------------- #
            first = first_step != int32(0)
            shift = (accepted != int32(0)) and not first
            take_state = first or shift

            for offset in unroll_if(
                range(history_length - int32(1)), unroll_stage
            ):
                slot = history_length - int32(1) - offset
                for i in unroll_if(range(n), unroll_step_element):
                    values[slot * n + i] = cuda.selp(
                        shift,
                        values[(slot - int32(1)) * n + i],
                        values[slot * n + i],
                    )
            for i in unroll_if(range(n), unroll_step_element):
                values[i] = cuda.selp(take_state, state[i], values[i])

            for offset in unroll_if(
                range(history_length - int32(1)), unroll_other_small
            ):
                slot = history_length - int32(1) - offset
                intervals[slot] = cuda.selp(
                    shift, intervals[slot - int32(1)], intervals[slot]
                )
            stored = count[0]
            stored = cuda.selp(
                shift, min(stored + int32(1), history_length), stored
            )
            stored = cuda.selp(first, int32(1), stored)

            # A jump past order one's limit joins the last two steps.
            merge = (stored > int32(2)) and not (
                step_size <= ratio_limits[0] * intervals[1]
            )
            for slot in unroll_if(
                range(int32(1), history_length - int32(1)), unroll_stage
            ):
                for i in unroll_if(range(n), unroll_step_element):
                    values[slot * n + i] = cuda.selp(
                        merge,
                        values[(slot + int32(1)) * n + i],
                        values[slot * n + i],
                    )
            intervals[1] = cuda.selp(
                merge, intervals[1] + intervals[2], intervals[1]
            )
            for slot in unroll_if(
                range(int32(2), history_length - int32(1)),
                unroll_other_small,
            ):
                intervals[slot] = cuda.selp(
                    merge, intervals[slot + int32(1)], intervals[slot]
                )
            stored = cuda.selp(merge, stored - int32(1), stored)
            count[0] = stored
            intervals[0] = step_size

            # ------------------------------------------------------- #
            #       Inverse distances from the step end to each state #
            # ------------------------------------------------------- #
            # rho[m] = step_size / (t_end - t_m); rho[0] is one.
            rho = cuda.local.array(history_length, numba_precision)
            rho[0] = typed_one
            elapsed = step_size
            for m in unroll_if(
                range(int32(1), history_length), unroll_other_small
            ):
                elapsed = elapsed + intervals[m]
                reached = elapsed > typed_zero
                rho[m] = cuda.selp(
                    reached,
                    step_size / cuda.selp(reached, elapsed, typed_one),
                    typed_zero,
                )

            # ------------------------------------------------------- #
            #     Largest order the history and step ratios support   #
            # ------------------------------------------------------- #
            # Order q needs q + 1 states and each step within its limit.
            positive_step = step_size > typed_zero
            order = int32(0)
            supported = True
            for order_index in unroll_if(
                range(max_order), unroll_other_small
            ):
                limit = ratio_limits[order_index]
                within = True
                for step_index in unroll_if(
                    range(max_order), unroll_other_small
                ):
                    if step_index == int32(0):
                        newer = step_size
                    else:
                        newer = intervals[step_index]
                    older = intervals[step_index + int32(1)]
                    within = within and (
                        step_index > order_index or newer <= limit * older
                    )
                candidate = order_index + int32(1)
                supported = (
                    supported
                    and candidate < stored
                    and within
                    and (order_index == int32(0) or positive_step)
                )
                order = cuda.selp(supported, candidate, order)
            restart = order == int32(0)
            corrector_order = max(order, int32(1))

            # ------------------------------------------------------- #
            #              Corrector and predictor weights            #
            # ------------------------------------------------------- #
            leading = typed_zero
            for m in unroll_if(range(max_order), unroll_other_small):
                leading += cuda.selp(m < corrector_order, rho[m], typed_zero)

            corrector = cuda.local.array(max_order, numba_precision)
            for j in unroll_if(range(max_order), unroll_other_small):
                power = typed_one
                spread = typed_one
                for m in unroll_if(range(max_order), unroll_other_small):
                    in_formula = m < corrector_order
                    power *= cuda.selp(in_formula, rho[j], typed_one)
                    spread *= cuda.selp(
                        in_formula and m != j, rho[j] - rho[m], typed_one
                    )
                in_formula = j < corrector_order
                scale = cuda.selp(in_formula, spread * leading, typed_one)
                corrector[j] = cuda.selp(
                    in_formula, power / scale, typed_zero
                )

            predictor = cuda.local.array(history_length, numba_precision)
            for j in unroll_if(range(history_length), unroll_other_small):
                numerator = typed_one
                denominator = typed_one
                for m in unroll_if(
                    range(history_length), unroll_other_small
                ):
                    in_fit = m <= corrector_order and m != j
                    numerator *= cuda.selp(in_fit, rho[j], typed_one)
                    denominator *= cuda.selp(
                        in_fit, rho[j] - rho[m], typed_one
                    )
                # Stale nodes outside the fit can coincide.
                usable = (j <= corrector_order) and (
                    denominator != typed_zero
                )
                denominator = cuda.selp(usable, denominator, typed_one)
                fitted = cuda.selp(
                    usable, numerator / denominator, typed_zero
                )
                start_only = cuda.selp(j == int32(0), typed_one, typed_zero)
                predictor[j] = cuda.selp(restart, start_only, fitted)

            next_node = typed_zero
            for m in unroll_if(range(history_length), unroll_other_small):
                next_node += cuda.selp(
                    m == corrector_order, rho[m], typed_zero
                )
            error_scale = cuda.selp(
                restart, typed_half, next_node / leading
            )
            corrector_step = step_size / leading

            # ------------------------------------------------------- #
            #           Corrector base state and prediction           #
            # ------------------------------------------------------- #
            for i in unroll_if(range(n), unroll_step_element):
                base = typed_zero
                predicted = typed_zero
                for j in unroll_if(range(history_length), unroll_stage):
                    value = values[j * n + i]
                    if j < max_order:
                        base += corrector[j] * value
                    predicted += predictor[j] * value
                base_state[i] = base
                prediction[i] = predicted

            return corrector_step, error_scale, restart

        # no cover: end
        return StepHistoryCache(history_fn=history)
