"""CUDA factories for scaled norms."""

from typing import Callable, Optional, Tuple
from warnings import warn

from numpy import asarray, finfo, int32 as np_int32, ndarray
from cubie.cuda_simsafe import cuda, int32
from cubie.cuda_simsafe import unroll_if
from attrs import define, field, Converter, frozen, validators

from cubie._utils import (
    PrecisionDType,
    build_config,
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
)

ATOL_FLOOR = 1e-16
"""Smallest absolute tolerance any norm divides by."""


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
    n: int = field(
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
        if self.n != self.solver_width:
            raise ValueError(
                "n must equal solver_width for a whole-vector norm; "
                "use a tiled norm config for stage-blocked tolerances"
            )

    @property
    def inv_n(self) -> float:
        """Return 1/solver_width in configured precision."""
        return self.precision(1.0 / self.solver_width)

    @property
    def tol_length(self) -> int:
        """Return the tolerance-array length for tol_converter."""
        return self.n


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
    stage_coefficients : tuple
        Row-major flattened Butcher ``a`` matrix, as produced by
        ``tableau.a_flat``.
    """

    stage_coefficients: tuple = field(default=(1.0,))

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        stage_count = self.stage_count
        if len(self.stage_coefficients) != stage_count * stage_count:
            raise ValueError(
                "stage_coefficients must hold stage_count**2 values"
            )

    def _check_widths(self) -> None:
        """Require whole stage blocks of ``n`` physical states."""
        if self.solver_width % self.n != 0:
            raise ValueError(
                "solver_width must be a multiple of n"
            )

    @property
    def stage_count(self) -> int:
        """Return the number of coupled stages."""
        return self.solver_width // self.n


@define
class ScaledNormCache(CUDADispatcherCache):
    """Hold a scaled norm device function."""

    scaled_norm: Callable = field(validator=is_device_validator)


class ScaledNorm(MultipleInstanceCUDAFactory):
    """Compile a mean squared scaled norm."""

    config_type = ScaledNormConfig

    def __init__(
        self,
        precision: PrecisionDType,
        solver_width: int,
        n: int,
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
        n : int
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
                "n": n,
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
        unroll = config.unroll

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def scaled_norm(values, reference):
            """Return the mean squared scaled norm."""
            nrm2 = typed_zero
            for i in unroll_if(range(n_val), unroll.norms):
                tol_i = atol[i] + rtol[i] * abs(reference[i])
                ratio = abs(values[i]) / tol_i
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(scaled_norm=scaled_norm)

    def update(self, updates_dict=None, silent=False, **kwargs):
        """Update compile settings and invalidate cache if changed.

        Parameters
        ----------
        updates_dict : dict, optional
            Dictionary of settings to update.
        silent : bool, default False
            If True, suppress warnings about unrecognized keys.
        **kwargs
            Additional settings as keyword arguments.

        Returns
        -------
        set
            Set of recognized parameter names that were updated.
        """
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)

        if not all_updates:
            return set()

        return self.update_compile_settings(
            updates_dict=all_updates, silent=silent
        )

    @property
    def device_function(self) -> Callable:
        """Return cached scaled norm device function."""
        return self.get_cached_output("scaled_norm")

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
        """Require whole stage blocks of ``n`` physical states."""
        if self.solver_width % self.n != 0:
            raise ValueError(
                "solver_width must be a multiple of n"
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
        unroll = config.unroll
        state_n = int32(config.n)

        typed_zero = numba_precision(0.0)

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def scaled_norm(values, reference):
            """Return the mean squared scaled norm."""
            nrm2 = typed_zero
            for index in unroll_if(range(n_val), unroll.norms):
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
        return ScaledNormCache(scaled_norm=scaled_norm)


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
        unroll = config.unroll
        typed_zero = numba_precision(0.0)

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
            for i in unroll_if(range(n_val), unroll.norms):
                stage_value = (
                    stage_base[i] + a_ij * stage_increment[i]
                )
                reference = max(abs(stage_value), abs(step_start[i]))
                tolerance = atol[i] + rtol[i] * reference
                ratio = values[i] / tolerance
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(scaled_norm=correction_norm)


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
        unroll = config.unroll
        state_n = int32(config.n)
        stage_count = int32(config.stage_count)
        stage_coefficients = tuple(
            numba_precision(value) for value in config.stage_coefficients
        )
        typed_zero = numba_precision(0.0)

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
            for index in unroll_if(range(n_val), unroll.norms):
                stage_index = index // state_n
                state_index = index - stage_index * state_n
                stage_value = stage_base[state_index]
                for contribution_index in unroll_if(range(stage_count), unroll.norms):
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
        return ScaledNormCache(scaled_norm=correction_norm)


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
        if len(self.mass_flags) != self.n:
            raise ValueError(
                "mass_flags must carry one flag per state: got "
                f"{len(self.mass_flags)} flags for n={self.n}."
            )

    @property
    def mass_flags(self) -> Tuple[bool, ...]:
        """Return the per-row mass flags; every row when unset."""
        if self._mass_flags is None:
            return (True,) * self.n
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
        unroll = config.unroll

        if all(config.mass_flags):
            n_val = int32(config.n)

            # no cover: start
            @cuda.jit(device=True, inline=True, **jit_kwargs)
            def scaled_norm(values, reference_a, reference_b):
                """Return the mean squared scaled norm."""
                nrm2 = typed_zero
                for i in unroll_if(range(n_val), unroll.norms):
                    tol_i = atol[i] + rtol[i] * max(
                        abs(reference_a[i]), abs(reference_b[i])
                    )
                    ratio = abs(values[i]) / tol_i
                    nrm2 += ratio * ratio
                return nrm2 * inv_n

            # no cover: end
            return ScaledNormCache(scaled_norm=scaled_norm)

        flagged_indices = asarray(config.flagged_indices, dtype=np_int32)
        flagged_count = int32(len(config.flagged_indices))

        # no cover: start
        @cuda.jit(device=True, inline=True, **jit_kwargs)
        def scaled_norm(values, reference_a, reference_b):
            """Return the mean squared scaled norm over the flagged rows."""
            nrm2 = typed_zero
            for k in unroll_if(range(flagged_count), unroll.norms):
                i = flagged_indices[k]
                tol_i = atol[i] + rtol[i] * max(
                    abs(reference_a[i]), abs(reference_b[i])
                )
                ratio = abs(values[i]) / tol_i
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(scaled_norm=scaled_norm)

    @property
    def mass_flags(self) -> Tuple[bool, ...]:
        """Return the per-row mass flags."""
        return self.compile_settings.mass_flags
