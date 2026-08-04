"""CUDA factories for scaled norms."""

import warnings
from typing import Callable

from numpy import asarray, finfo, ndarray
from cubie.cuda_simsafe import cuda, int32
from attrs import define, field, Converter, frozen

from cubie._utils import (
    PrecisionDType,
    build_config,
    getype_validator,
    nonnegative_float_array_validator,
    is_device_validator,
    tol_converter,
)
from cubie.CUDAFactory import (
    CUDADispatcherCache,
    MultipleInstanceCUDAFactoryConfig,
    MultipleInstanceCUDAFactory,
)


def rtol_floor_converter(value, self_) -> ndarray:
    """Convert rtol and raise sub-noise components to a usable floor.

    A relative tolerance below four ULPs of the working precision
    asks the scaled norm to resolve corrections smaller than rounding
    noise, which no iteration can achieve. Nonzero components below
    ``4 * eps`` are normalized up to that floor with a warning; zero
    components (relative control disabled) pass through unchanged.
    """
    tolerance = tol_converter(value, self_)
    floor = 4.0 * float(finfo(self_.precision).eps)
    below = (tolerance > 0.0) & (tolerance < floor)
    if below.any():
        label = self_.instance_label or "norm"
        warnings.warn(
            f"{label} rtol components below {floor:.3e} (4 ULPs at "
            "the working precision) cannot be met; they were raised "
            "to that floor.",
            UserWarning,
            stacklevel=2,
        )
        tolerance = tolerance.copy()
        tolerance[below] = floor
        tolerance.setflags(write=False)
    return tolerance


@frozen
class ScaledNormConfig(MultipleInstanceCUDAFactoryConfig):
    """Configure a scaled norm.

    Attributes
    ----------
    solver_width : int
        Length of the solver vectors the norm reduces over.
    atol : ndarray
        Absolute tolerance array of shape (solver_width,).
    rtol : ndarray
        Relative tolerance array of shape (solver_width,).

    Notes
    -----
    Tolerance sizing follows ``solver_width`` through the
    converter: every snapshot (construction or update-derived
    replacement) re-runs :func:`cubie._utils.tol_converter`, which
    broadcasts scalar or uniform-array specifications to shape
    ``(solver_width,)``. A non-uniform array of the wrong length
    raises at the write boundary; update ``solver_width`` and the
    tolerance arrays together in one call. Nonzero ``rtol``
    components below four ULPs of the working precision are raised
    to that floor with a warning (see
    :func:`rtol_floor_converter`).
    """

    solver_width: int = field(
        default=1,
        validator=getype_validator(int, 1),
    )
    atol: ndarray = field(
        default=asarray([1e-6]),
        validator=nonnegative_float_array_validator,
        converter=Converter(tol_converter, takes_self=True),
        metadata={"prefixed": True},
    )
    rtol: ndarray = field(
        default=asarray([1e-6]),
        validator=nonnegative_float_array_validator,
        converter=Converter(rtol_floor_converter, takes_self=True),
        metadata={"prefixed": True},
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def inv_n(self) -> float:
        """Return 1/solver_width in configured precision."""
        return self.precision(1.0 / self.solver_width)

    @property
    def tol_length(self) -> int:
        """Return the tolerance-array length for tol_converter."""
        return self.solver_width

    @property
    def tol_floor(self) -> float:
        """Return minimum tolerance floor to avoid division by zero."""
        return self.precision(1e-16)


@frozen
class FIRKCorrectionNormConfig(ScaledNormConfig):
    """Configure a coupled FIRK correction norm.

    Attributes
    ----------
    n : int
        Number of physical states per stage.
    stage_coefficients : tuple
        Row-major flattened Butcher ``a`` matrix, as produced by
        ``tableau.a_flat``.
    """

    n: int = field(default=1, validator=getype_validator(int, 1))
    stage_coefficients: tuple = field(default=(1.0,))

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.solver_width % self.n != 0:
            raise ValueError(
                "solver_width must be a multiple of n"
            )
        stage_count = self.solver_width // self.n
        if len(self.stage_coefficients) != stage_count * stage_count:
            raise ValueError(
                "stage_coefficients must hold stage_count**2 values"
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
        instance_label : str, optional
            Prefix label for parameter names when used as a nested factory.
        **kwargs
            Optional parameters passed to ScaledNormConfig including
            atol and rtol. None values are ignored.
        """
        super().__init__(instance_label=instance_label)

        config = build_config(
            self.config_type,
            required={
                "precision": precision,
                "solver_width": solver_width,
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
        tol_floor = config.tol_floor

        typed_zero = numba_precision(0.0)
        n_val = int32(n)

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def scaled_norm(values, reference):
            """Return the mean squared scaled norm."""
            nrm2 = typed_zero
            for i in range(n_val):
                tol_i = atol[i] + rtol[i] * abs(reference[i])
                tol_i = max(tol_i, tol_floor)
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
        Number of physical states per stage. The reference vector
        holds one entry per physical state and is reused for every
        stage block of the ``solver_width``-element value vector.
    """

    n: int = field(default=1, validator=getype_validator(int, 1))

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.solver_width % self.n != 0:
            raise ValueError(
                "solver_width must be a multiple of n"
            )


class TiledScaledNorm(ScaledNorm):
    """Compile a scaled norm whose reference tiles across stages.

    Coupled FIRK solves stack ``solver_width = stage_count * n``
    values, but the physical reference vector holds only ``n``
    entries. The compiled function reads the reference entry for
    value ``i`` at ``i mod n`` so callers pass the single-stage
    reference directly.
    """

    config_type = TiledScaledNormConfig

    def build(self) -> ScaledNormCache:
        """Compile the stage-tiled norm."""
        config = self.compile_settings

        atol = config.atol
        rtol = config.rtol
        numba_precision = config.numba_precision
        inv_n = config.inv_n
        tol_floor = config.tol_floor
        n_val = int32(config.solver_width)
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
            for index in range(n_val):
                stage_index = index // state_n
                state_index = index - stage_index * state_n
                tol_i = (
                    atol[index]
                    + rtol[index] * abs(reference[state_index])
                )
                tol_i = max(tol_i, tol_floor)
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


class DIRKCorrectionNorm(CorrectionNorm):
    """Compile a DIRK correction norm."""

    def build(self) -> ScaledNormCache:
        """Compile the correction norm function."""
        config = self.compile_settings
        atol = config.atol
        rtol = config.rtol
        inv_n = config.inv_n
        tol_floor = config.tol_floor
        numba_precision = config.numba_precision
        n_val = int32(config.solver_width)
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
            for i in range(n_val):
                stage_value = (
                    stage_base[i] + a_ij * stage_increment[i]
                )
                reference = max(abs(stage_value), abs(step_start[i]))
                tolerance = atol[i] + rtol[i] * reference
                tolerance = max(tolerance, tol_floor)
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
        tol_floor = config.tol_floor
        numba_precision = config.numba_precision
        n_val = int32(config.solver_width)
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
            for index in range(n_val):
                stage_index = index // state_n
                state_index = index - stage_index * state_n
                stage_value = stage_base[state_index]
                for contribution_index in range(stage_count):
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
                tolerance = atol[index] + rtol[index] * reference
                tolerance = max(tolerance, tol_floor)
                ratio = values[index] / tolerance
                nrm2 += ratio * ratio
            return nrm2 * inv_n

        # no cover: end
        return ScaledNormCache(scaled_norm=correction_norm)
