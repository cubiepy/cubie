"""CUDA factories for scaled norms."""

from typing import Callable, Optional, Tuple
from warnings import warn

from numpy import asarray, finfo, int32 as np_int32, ndarray
from numpy import uint32 as np_uint32
from numba_cuda_mlir.types import int32, uint32
from cubie._cudasim_extensions import cuda
from cubie.backend.intrinsics import (
    bf16x2_fma,
    bf16x2_high,
    bf16x2_low,
    bf16x2_pack,
    unroll_if,
)
from cubie.backend.jit import compile_kwargs
from attrs import define, field, Converter, frozen, validators

from cubie._utils import (
    PrecisionDType,
    getype_validator,
    nonnegative_float_array_validator,
    is_device_validator,
    optional_tuple_converter,
    tol_converter,
)
from cubie.CUDAFactory import (
    CUDADispatcherCache,
    MultipleInstanceCUDAFactoryConfig,
    MultipleInstanceCUDAFactory,
    build_config,
)

ATOL_FLOOR = 1e-16
"""Smallest absolute tolerance any norm divides by."""

PACKED_ABS_MASK = 0x7FFF7FFF
"""Mask clearing both sign bits of a bfloat16 pair."""

PACKED_RECIPROCAL_MAGIC = 0x7EF37EF3
"""Bits that, less a positive bfloat16 pair, estimate its reciprocals.

Hardware-independent: the constant minimises the largest relative
error of the estimate over every positive normal bfloat16, found by
exhaustive search; that error is 5.1%.
"""

PACKED_NEGATIVE_ZERO = 0x80008000
"""A bfloat16 pair of negative zeros, the addend of a packed product."""


def pack_pairs(values: ndarray) -> ndarray:
    """Return ``values`` as bfloat16 pairs, a trailing odd entry dropped."""
    pairs = [
        bf16x2_pack(values[2 * k], values[2 * k + 1])
        for k in range(len(values) // 2)
    ]
    return asarray(pairs if pairs else [0], dtype=np_uint32)


# no cover: start
@cuda.jit(device=True, inline=True, **compile_kwargs)
def packed_squared_ratios(
    values_pair, reference_pair, atol_pair, rtol_pair, accumulator
):
    """Return ``accumulator`` plus both lanes' squared scaled ratios.

    Every argument is a bfloat16 pair; ``reference_pair`` holds the
    magnitudes the tolerances scale with.
    """
    tolerance = bf16x2_fma(rtol_pair, reference_pair, atol_pair)
    inverse = uint32(uint32(PACKED_RECIPROCAL_MAGIC) - tolerance)
    ratio = bf16x2_fma(
        uint32(values_pair & uint32(PACKED_ABS_MASK)),
        inverse,
        uint32(PACKED_NEGATIVE_ZERO),
    )
    return bf16x2_fma(ratio, ratio, accumulator)


@cuda.jit(device=True, inline=True, **compile_kwargs)
def packed_sum(accumulator):
    """Return the sum of a bfloat16 pair as a float32."""
    return bf16x2_high(accumulator) + bf16x2_low(accumulator)


# no cover: end


def _floored(tolerance: ndarray, floor: float, mask: ndarray) -> ndarray:
    """Return ``tolerance`` with the masked entries raised to ``floor``."""
    if mask.any():
        tolerance = tolerance.copy()
        tolerance[mask] = floor
        tolerance.setflags(write=False)
    return tolerance


def atol_floor_converter(value, self_) -> ndarray:
    """Convert atol, flooring every component at ``ATOL_FLOOR``."""
    tolerance = tol_converter(value, self_)
    below = (tolerance >= 0.0) & (tolerance < ATOL_FLOOR)
    if below.any():
        warn(
            f"atol entries below {ATOL_FLOOR:g} raised to that floor: "
            f"{tolerance[below].tolist()}",
            UserWarning,
            stacklevel=2,
        )
    return _floored(tolerance, ATOL_FLOOR, below)


def rtol_floor_converter(value, self_) -> ndarray:
    """Convert rtol, flooring nonzero components at 4 ULPs."""
    tolerance = tol_converter(value, self_)
    floor = 4.0 * float(finfo(self_.precision).eps)
    below = (tolerance > 0.0) & (tolerance < floor)
    return _floored(tolerance, floor, below)


@frozen
class ScaledNormConfig(MultipleInstanceCUDAFactoryConfig):
    """Configure a scaled norm.

    Attributes
    ----------
    solver_width : int
        Length of the solver vectors the norm reduces over.
    n : int
        Number of physical states per stage.
    atol : ndarray
        Absolute tolerance array of shape (n,), every entry floored at
        ``ATOL_FLOOR``.
    rtol : ndarray
        Relative tolerance array of shape (n,).

    Notes
    -----
    Every snapshot re-runs :func:`cubie._utils.tol_converter`, which
    broadcasts scalar or uniform-array tolerances to shape ``(n,)``.
    A non-uniform array of another length raises, so change ``n`` and
    the tolerance arrays in one call. ``n`` is declared before the
    tolerance fields because the converter reads it.
    """

    solver_width: int = field(
        default=1,
        validator=getype_validator(int, 1),
    )
    n_states: int = field(
        default=1,
        validator=getype_validator(int, 1),
    )
    atol: ndarray = field(
        default=asarray([1e-6]),
        validator=nonnegative_float_array_validator,
        converter=Converter(atol_floor_converter, takes_self=True),
        metadata={"prefixed": True},
    )
    rtol: ndarray = field(
        default=asarray([1e-6]),
        validator=nonnegative_float_array_validator,
        converter=Converter(tol_converter, takes_self=True),
        metadata={"prefixed": True},
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self._check_widths()

    def _check_widths(self) -> None:
        """Require one tolerance per solver-vector entry."""
        if self.n_states != self.solver_width:
            raise ValueError(
                "n_states must equal solver_width for a whole-vector norm; "
                "use a tiled norm config for stage-blocked tolerances"
            )

    @property
    def inv_n(self) -> float:
        """Return 1/solver_width in configured precision."""
        return self.precision(1.0 / self.solver_width)

    @property
    def tol_length(self) -> int:
        """Return the tolerance-array length for tol_converter."""
        return self.n_states


@frozen
class CorrectionNormConfig(ScaledNormConfig):
    """Newton correction norm config; ``rtol`` floors at 4 ULPs."""

    rtol: ndarray = field(
        default=asarray([1e-6]),
        validator=nonnegative_float_array_validator,
        converter=Converter(rtol_floor_converter, takes_self=True),
        metadata={"prefixed": True},
    )


@frozen
class FIRKCorrectionNormConfig(CorrectionNormConfig):
    """Configure a coupled FIRK correction norm.

    Attributes
    ----------
    n : int
        Number of physical states per stage. The tolerance arrays hold
        one entry per physical state, reused for every stage block.
    tableau : ButcherTableau
        The tableau whose ``a`` matrix weights the stage increments.
    """

    tableau: Optional[object] = field(default=None)

    def _check_widths(self) -> None:
        """Require whole stage blocks of ``n_states`` physical states."""
        if self.solver_width % self.n_states != 0:
            raise ValueError(
                "solver_width must be a multiple of n_states"
            )

    @property
    def stage_count(self) -> int:
        """Return the number of coupled stages."""
        return self.solver_width // self.n_states


@define
class ScaledNormCache(CUDADispatcherCache):
    """Hold a scaled norm device function."""

    norm_fn: Callable = field(validator=is_device_validator)


class ScaledNorm(MultipleInstanceCUDAFactory):
    """Compile a mean squared scaled norm."""

    config_type = ScaledNormConfig

    def __init__(
        self,
        precision: PrecisionDType,
        solver_width: int,
        n_states: int,
        instance_label: str = "",
        **kwargs,
    ) -> None:
        """Initialize ScaledNorm factory.

        Parameters
        ----------
        precision : PrecisionDType
            Numerical precision for computations.
        solver_width : int
            Length of the solver vectors the norm reduces over.
        n_states : int
            Number of physical states per stage.
        instance_label : str, optional
            Prefix label for parameter names when used as a nested factory.
        **kwargs
            Optional parameters passed to ScaledNormConfig including
            atol and rtol. None values are ignored. ``atol`` and
            ``rtol`` hold one entry per physical state.
        """
        super().__init__(instance_label=instance_label)

        config = build_config(
            self.config_type,
            required={
                "precision": precision,
                "solver_width": solver_width,
                "n_states": n_states,
            },
            instance_label=instance_label,
            **kwargs,
        )

        self.setup_compile_settings(config)

    def build(self) -> ScaledNormCache:
        """Compile the whole-vector norm."""
        config = self.compile_settings

        n = config.solver_width
        atol = config.atol
        rtol = config.rtol
        numba_precision = config.numba_precision
        inv_n = config.inv_n

        typed_zero = numba_precision(0.0)
        n_val = int32(n)
        unroll_norms = config.unroll.unroll_norms

        if config.uses_reduced_precision("low_precision_norms"):
            return ScaledNormCache(
                norm_fn=self._packed_norm(n, atol, rtol, inv_n)
            )

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def norm_fn(values, reference):
            """Return the mean squared scaled norm."""
            nrm2 = typed_zero
            for i in unroll_if(range(n_val), unroll_norms):
                tol_i = atol[i] + rtol[i] * abs(reference[i])
                ratio = abs(values[i]) / tol_i
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(norm_fn=norm_fn)

    def _packed_norm(self, n, atol, rtol, inv_n) -> Callable:
        """Compile the whole-vector norm over bfloat16 pairs."""
        unroll_norms = self.compile_settings.unroll.unroll_norms
        pair_count = int32(n // 2)
        last = int32(n - 1)
        odd = n % 2 == 1
        atol_pairs = pack_pairs(atol)
        rtol_pairs = pack_pairs(rtol)
        typed_zero = self.compile_settings.numba_precision(0.0)

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def norm_fn(values, reference):
            """Return the mean squared scaled norm."""
            accumulator = uint32(0)
            for k in unroll_if(range(pair_count), unroll_norms):
                i = int32(2) * k
                accumulator = packed_squared_ratios(
                    bf16x2_pack(values[i], values[i + int32(1)]),
                    bf16x2_pack(
                        abs(reference[i]), abs(reference[i + int32(1)])
                    ),
                    atol_pairs[k],
                    rtol_pairs[k],
                    accumulator,
                )
            nrm2 = packed_sum(accumulator)
            if odd:
                ratio = abs(values[last]) / (
                    atol[last] + rtol[last] * abs(reference[last])
                )
                nrm2 += ratio * ratio
            else:
                nrm2 += typed_zero
            return nrm2 * inv_n

        # no cover: end
        return norm_fn

    @property
    def device_function(self) -> Callable:
        """Return cached scaled norm device function."""
        return self.get_cached_output("norm_fn")

    @property
    def precision(self) -> PrecisionDType:
        """Return configured precision."""
        return self.compile_settings.precision

    @property
    def solver_width(self) -> int:
        """Return the solver vector length."""
        return self.compile_settings.solver_width

    @property
    def atol(self) -> ndarray:
        """Return absolute tolerance array."""
        return self.compile_settings.atol

    @property
    def rtol(self) -> ndarray:
        """Return relative tolerance array."""
        return self.compile_settings.rtol


@frozen
class TiledScaledNormConfig(ScaledNormConfig):
    """Configure a scaled norm with a stage-tiled reference.

    Attributes
    ----------
    n : int
        Number of physical states per stage. The reference and
        tolerance vectors hold one entry per physical state, reused
        for every stage block.
    """

    def _check_widths(self) -> None:
        """Require whole stage blocks of ``n_states`` physical states."""
        if self.solver_width % self.n_states != 0:
            raise ValueError(
                "solver_width must be a multiple of n_states"
            )


class TiledScaledNorm(ScaledNorm):
    """Compile a scaled norm whose reference tiles across stages.

    Coupled FIRK solves stack ``solver_width = stage_count * n``
    values. The compiled function reads the reference and tolerance
    entries for value ``i`` at ``i mod n``, so callers pass the
    single-stage reference and the per-state tolerances directly.
    """

    config_type = TiledScaledNormConfig

    def build(self) -> ScaledNormCache:
        """Compile the stage-tiled norm."""
        config = self.compile_settings

        atol = config.atol
        rtol = config.rtol
        numba_precision = config.numba_precision
        inv_n = config.inv_n
        n_val = int32(config.solver_width)
        unroll_norms = config.unroll.unroll_norms
        state_n = int32(config.n_states)

        typed_zero = numba_precision(0.0)

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def norm_fn(values, reference):
            """Return the mean squared scaled norm."""
            nrm2 = typed_zero
            for index in unroll_if(range(n_val), unroll_norms):
                stage_index = index // state_n
                state_index = index - stage_index * state_n
                tol_i = (
                    atol[state_index]
                    + rtol[state_index] * abs(reference[state_index])
                )
                ratio = abs(values[index]) / tol_i
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(norm_fn=norm_fn)


class CorrectionNorm(ScaledNorm):
    """Base factory for Newton correction norms.

    Correction norms scale the Newton update against the physical
    stage state and the step-start state, matching the reference
    scaling ``atol + rtol * max(|stage_value|, |step_start|)``. The
    compiled function takes ``(values, stage_increment, stage_base,
    step_start, a_ij)`` in place of the two-argument scaled norm.
    """

    config_type = CorrectionNormConfig


class DIRKCorrectionNorm(CorrectionNorm):
    """Compile a DIRK correction norm."""

    def build(self) -> ScaledNormCache:
        """Compile the correction norm function."""
        config = self.compile_settings
        atol = config.atol
        rtol = config.rtol
        inv_n = config.inv_n
        numba_precision = config.numba_precision
        n_val = int32(config.solver_width)
        unroll_norms = config.unroll.unroll_norms
        typed_zero = numba_precision(0.0)

        if config.uses_reduced_precision("low_precision_norms"):
            return ScaledNormCache(
                norm_fn=self._packed_correction_norm(atol, rtol, inv_n)
            )

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def correction_norm(
            values,
            stage_increment,
            stage_base,
            step_start,
            a_ij,
        ):
            """Return the mean squared scaled correction norm."""
            nrm2 = typed_zero
            for i in unroll_if(range(n_val), unroll_norms):
                stage_value = (
                    stage_base[i] + a_ij * stage_increment[i]
                )
                reference = max(abs(stage_value), abs(step_start[i]))
                tolerance = atol[i] + rtol[i] * reference
                ratio = values[i] / tolerance
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(norm_fn=correction_norm)

    def _packed_correction_norm(self, atol, rtol, inv_n) -> Callable:
        """Compile the correction norm over bfloat16 pairs."""
        config = self.compile_settings
        n = config.solver_width
        unroll_norms = config.unroll.unroll_norms
        pair_count = int32(n // 2)
        last = int32(n - 1)
        odd = n % 2 == 1
        atol_pairs = pack_pairs(atol)
        rtol_pairs = pack_pairs(rtol)
        typed_zero = config.numba_precision(0.0)

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def reference_value(stage_increment, stage_base, step_start, a_ij, i):
            """Return the magnitude the tolerance of entry ``i`` scales."""
            stage_value = stage_base[i] + a_ij * stage_increment[i]
            return max(abs(stage_value), abs(step_start[i]))

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def correction_norm(
            values,
            stage_increment,
            stage_base,
            step_start,
            a_ij,
        ):
            """Return the mean squared scaled correction norm."""
            accumulator = uint32(0)
            for k in unroll_if(range(pair_count), unroll_norms):
                i = int32(2) * k
                j = i + int32(1)
                accumulator = packed_squared_ratios(
                    bf16x2_pack(values[i], values[j]),
                    bf16x2_pack(
                        reference_value(
                            stage_increment, stage_base, step_start, a_ij, i
                        ),
                        reference_value(
                            stage_increment, stage_base, step_start, a_ij, j
                        ),
                    ),
                    atol_pairs[k],
                    rtol_pairs[k],
                    accumulator,
                )
            nrm2 = packed_sum(accumulator)
            if odd:
                ratio = values[last] / (
                    atol[last]
                    + rtol[last]
                    * reference_value(
                        stage_increment, stage_base, step_start, a_ij, last
                    )
                )
                nrm2 += ratio * ratio
            else:
                nrm2 += typed_zero
            return nrm2 * inv_n

        # no cover: end
        return correction_norm


class FIRKCorrectionNorm(CorrectionNorm):
    """Compile a coupled FIRK correction norm."""

    config_type = FIRKCorrectionNormConfig

    def build(self) -> ScaledNormCache:
        """Compile the correction norm function."""
        config = self.compile_settings
        atol = config.atol
        rtol = config.rtol
        inv_n = config.inv_n
        numba_precision = config.numba_precision
        n_val = int32(config.solver_width)
        unroll_norms = config.unroll.unroll_norms
        state_n = int32(config.n_states)
        stage_count = int32(config.stage_count)
        stage_coefficients = config.tableau.a_flat(config.precision)
        typed_zero = numba_precision(0.0)

        if config.uses_reduced_precision("low_precision_norms"):
            return ScaledNormCache(norm_fn=self._packed_stage_norm(inv_n))

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def correction_norm(
            values,
            stage_increment,
            stage_base,
            step_start,
            a_ij,
        ):
            """Return the mean squared scaled correction norm."""
            nrm2 = typed_zero
            for index in unroll_if(range(n_val), unroll_norms):
                stage_index = index // state_n
                state_index = index - stage_index * state_n
                stage_value = stage_base[state_index]
                for contribution_index in unroll_if(range(stage_count), unroll_norms):
                    coefficient_index = (
                        stage_index * stage_count + contribution_index
                    )
                    increment_index = (
                        contribution_index * state_n + state_index
                    )
                    stage_value += (
                        stage_coefficients[coefficient_index]
                        * stage_increment[increment_index]
                    )

                reference = max(
                    abs(stage_value), abs(step_start[state_index])
                )
                tolerance = (
                    atol[state_index] + rtol[state_index] * reference
                )
                ratio = values[index] / tolerance
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(norm_fn=correction_norm)

    def _packed_stage_norm(self, inv_n) -> Callable:
        """Compile the coupled correction norm over bfloat16 pairs.

        Pairs run over the stage-major solver vector, so each pair
        takes its tolerances from the states its two entries hold.
        """
        config = self.compile_settings
        n = config.solver_width
        state_count = config.n_states
        unroll_norms = config.unroll.unroll_norms
        pair_count = int32(n // 2)
        last = int32(n - 1)
        odd = n % 2 == 1
        atol_full = asarray(
            [config.atol[i % state_count] for i in range(n)]
        )
        rtol_full = asarray(
            [config.rtol[i % state_count] for i in range(n)]
        )
        atol_pairs = pack_pairs(atol_full)
        rtol_pairs = pack_pairs(rtol_full)
        atol_last = config.precision(atol_full[-1])
        rtol_last = config.precision(rtol_full[-1])
        typed_zero = config.numba_precision(0.0)
        stage_count = int32(config.stage_count)
        state_n = int32(config.n_states)
        stage_coefficients = config.tableau.a_flat(config.precision)

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def reference_value(stage_increment, stage_base, step_start, index):
            """Return the magnitude the tolerance of entry ``index`` scales."""
            stage_index = index // state_n
            state_index = index - stage_index * state_n
            stage_value = stage_base[state_index]
            for contribution_index in unroll_if(
                range(stage_count), unroll_norms
            ):
                coefficient_index = (
                    stage_index * stage_count + contribution_index
                )
                increment_index = contribution_index * state_n + state_index
                stage_value += (
                    stage_coefficients[coefficient_index]
                    * stage_increment[increment_index]
                )
            return max(abs(stage_value), abs(step_start[state_index]))

        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def correction_norm(
            values,
            stage_increment,
            stage_base,
            step_start,
            a_ij,
        ):
            """Return the mean squared scaled correction norm."""
            accumulator = uint32(0)
            for k in unroll_if(range(pair_count), unroll_norms):
                i = int32(2) * k
                j = i + int32(1)
                accumulator = packed_squared_ratios(
                    bf16x2_pack(values[i], values[j]),
                    bf16x2_pack(
                        reference_value(
                            stage_increment, stage_base, step_start, i
                        ),
                        reference_value(
                            stage_increment, stage_base, step_start, j
                        ),
                    ),
                    atol_pairs[k],
                    rtol_pairs[k],
                    accumulator,
                )
            nrm2 = packed_sum(accumulator)
            if odd:
                ratio = values[last] / (
                    atol_last
                    + rtol_last
                    * reference_value(
                        stage_increment, stage_base, step_start, last
                    )
                )
                nrm2 += ratio * ratio
            else:
                nrm2 += typed_zero
            return nrm2 * inv_n

        # no cover: end
        return correction_norm


@frozen
class TwoRefMaskedScaledNormConfig(ScaledNormConfig):
    """Scaled-norm config with one ``mass_flags`` entry per state."""

    _mass_flags: Optional[Tuple[bool, ...]] = field(
        default=None,
        converter=optional_tuple_converter,
        validator=validators.optional(
            validators.deep_iterable(
                validators.instance_of(bool),
                validators.instance_of(tuple),
            )
        ),
        metadata={"prefixed": True},
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if len(self.mass_flags) != self.n_states:
            raise ValueError(
                "mass_flags must carry one flag per state: got "
                f"{len(self.mass_flags)} flags for n_states={self.n_states}."
            )

    @property
    def mass_flags(self) -> Tuple[bool, ...]:
        """Return the per-row mass flags; every row when unset."""
        if self._mass_flags is None:
            return (True,) * self.n_states
        return self._mass_flags

    @property
    def flagged_indices(self) -> Tuple[int, ...]:
        """Return the indices of the rows whose flag is set."""
        return tuple(
            index for index, flag in enumerate(self.mass_flags) if flag
        )

    @property
    def inv_n(self) -> float:
        """Return 1/(number of flagged rows) in configured precision."""
        return self.precision(1.0 / max(1, len(self.flagged_indices)))


class TwoRefMaskedScaledNorm(ScaledNorm):
    """Compile ``norm(values, ref_a, ref_b)`` over the flagged rows."""

    config_type = TwoRefMaskedScaledNormConfig

    def build(self) -> ScaledNormCache:
        """Compile the masked two-reference norm."""
        config = self.compile_settings
        atol = config.atol
        rtol = config.rtol
        inv_n = config.inv_n
        numba_precision = config.numba_precision
        typed_zero = numba_precision(0.0)
        jit_kwargs = self.jit_kwargs
        unroll_norms = config.unroll.unroll_norms

        if all(config.mass_flags) and config.uses_reduced_precision(
            "low_precision_norms"
        ):
            return ScaledNormCache(
                norm_fn=self._packed_two_reference_norm(atol, rtol, inv_n)
            )

        if all(config.mass_flags):
            n_val = int32(config.n_states)

            # no cover: start
            @cuda.jit(device=True, inline=True, **jit_kwargs)
            def norm_fn(values, reference_a, reference_b):
                """Return the mean squared scaled norm."""
                nrm2 = typed_zero
                for i in unroll_if(range(n_val), unroll_norms):
                    tol_i = atol[i] + rtol[i] * max(
                        abs(reference_a[i]), abs(reference_b[i])
                    )
                    ratio = abs(values[i]) / tol_i
                    nrm2 += ratio * ratio
                return nrm2 * inv_n

            # no cover: end
            return ScaledNormCache(norm_fn=norm_fn)

        flagged_indices = asarray(config.flagged_indices, dtype=np_int32)
        flagged_count = int32(len(config.flagged_indices))

        # no cover: start
        @cuda.jit(device=True, inline=True, **jit_kwargs)
        def norm_fn(values, reference_a, reference_b):
            """Return the mean squared scaled norm over the flagged rows."""
            nrm2 = typed_zero
            for k in unroll_if(range(flagged_count), unroll_norms):
                i = flagged_indices[k]
                tol_i = atol[i] + rtol[i] * max(
                    abs(reference_a[i]), abs(reference_b[i])
                )
                ratio = abs(values[i]) / tol_i
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(norm_fn=norm_fn)

    def _packed_two_reference_norm(self, atol, rtol, inv_n) -> Callable:
        """Compile the all-rows two-reference norm over bfloat16 pairs."""
        config = self.compile_settings
        n = config.n_states
        unroll_norms = config.unroll.unroll_norms
        pair_count = int32(n // 2)
        last = int32(n - 1)
        odd = n % 2 == 1
        atol_pairs = pack_pairs(atol)
        rtol_pairs = pack_pairs(rtol)
        typed_zero = config.numba_precision(0.0)

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def norm_fn(values, reference_a, reference_b):
            """Return the mean squared scaled norm."""
            accumulator = uint32(0)
            for k in unroll_if(range(pair_count), unroll_norms):
                i = int32(2) * k
                j = i + int32(1)
                accumulator = packed_squared_ratios(
                    bf16x2_pack(values[i], values[j]),
                    bf16x2_pack(
                        max(abs(reference_a[i]), abs(reference_b[i])),
                        max(abs(reference_a[j]), abs(reference_b[j])),
                    ),
                    atol_pairs[k],
                    rtol_pairs[k],
                    accumulator,
                )
            nrm2 = packed_sum(accumulator)
            if odd:
                tolerance = atol[last] + rtol[last] * max(
                    abs(reference_a[last]), abs(reference_b[last])
                )
                ratio = abs(values[last]) / tolerance
                nrm2 += ratio * ratio
            else:
                nrm2 += typed_zero
            return nrm2 * inv_n

        # no cover: end
        return norm_fn

    @property
    def mass_flags(self) -> Tuple[bool, ...]:
        """Return the per-row mass flags."""
        return self.compile_settings.mass_flags
