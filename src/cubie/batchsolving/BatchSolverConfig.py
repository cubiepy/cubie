"""Compile-time configuration for batch solver kernels.

Published Classes
-----------------
:class:`ActiveOutputs`
    Boolean flags indicating which output types are enabled.

:class:`BatchSolverConfig`
    Compile-critical settings that trigger kernel recompilation when changed.

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactoryConfig`
    Parent class for compile-critical configuration.
:class:`~cubie.outputhandling.output_config.OutputCompileFlags`
    Source flags from which ``ActiveOutputs`` is derived.
:class:`~cubie.batchsolving.BatchSolverKernel.BatchSolverKernel`
    Consumer of this configuration.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import attrs
from attrs import converters
from attrs import validators as val

from cubie._env import kernel_cache_dir_default, max_cache_entries_default
from cubie._utils import (
    device_function_field,
    getype_validator,
)
from cubie.backend.utils import SASS_INSTRUCTION_BYTES
from cubie.CUDAFactory import CUDAFactoryConfig, _CubieConfigBase
from cubie.cuda_simsafe import (
    ALL_UNROLL_PARAMETERS,
    UnrollChoice,
    UnrollFlag,
    UnrollFlags,
    unroll_flag_converter,
)
from cubie.outputhandling.output_config import OutputCompileFlags


@attrs.frozen
class ActiveOutputs(_CubieConfigBase):
    """
    Track which output arrays are configured to be produced.

    This class provides boolean flags indicating which output types are
    enabled according to compile-time configuration flags, for example
    values derived from ``OutputCompileFlags``.

    Parameters
    ----------
    state
        Whether state output is active.
    observables
        Whether observables output is active.
    state_summaries
        Whether state summaries output is active.
    observable_summaries
        Whether observable summaries output is active.
    status_codes
        Whether status code output is active.
    iteration_counters
        Whether iteration counter output is active.
    """

    state: bool = attrs.field(default=False, validator=val.instance_of(bool))
    observables: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    state_summaries: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    observable_summaries: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    status_codes: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    iteration_counters: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @classmethod
    def from_compile_flags(cls, flags: OutputCompileFlags) -> "ActiveOutputs":
        """
        Create ActiveOutputs from compile flags.

        Parameters
        ----------
        flags
            The compile flags determining which outputs are active.

        Returns
        -------
        ActiveOutputs
            Instance with flags derived from compile flags.

        Notes
        -----
        Maps OutputCompileFlags to ActiveOutputs:
        - save_state → state
        - save_observables → observables
        - summarise_state → state_summaries
        - summarise_observables → observable_summaries
        - save_counters → iteration_counters
        - status_codes is always True (always written during execution)
        """
        return cls(
            state=flags.save_state,
            observables=flags.save_observables,
            state_summaries=flags.summarise_state,
            observable_summaries=flags.summarise_observables,
            status_codes=True,
            iteration_counters=flags.save_counters,
        )


@attrs.frozen
class CacheSettings(_CubieConfigBase):
    """Disk-cache settings for the compiled kernel.

    Attributes
    ----------
    cache_enabled
        Whether the compiled kernel persists to a disk cache.
    cache_mode
        ``"hash"`` keeps every configuration's entry;
        ``"flush_on_change"`` clears the directory on a settings change.
    max_cache_entries
        Entries kept per cache directory before LRU eviction; ``0``
        disables eviction.
    cache_dir
        Cache directory; ``None`` uses the shared cache root.
    """

    cache_enabled: bool = attrs.field(
        default=True, validator=val.instance_of(bool)
    )
    cache_mode: str = attrs.field(
        default="hash", validator=val.in_(("hash", "flush_on_change"))
    )
    max_cache_entries: int = attrs.field(
        factory=max_cache_entries_default,
        validator=getype_validator(int, 0),
    )
    cache_dir: Optional[Path] = attrs.field(
        factory=kernel_cache_dir_default,
        validator=val.optional(val.instance_of((str, Path))),
        converter=attrs.converters.optional(Path),
    )


ALL_CACHE_PARAMETERS = frozenset(
    fld.name for fld in attrs.fields(CacheSettings) if fld.init
)
"""Loose keyword names of the :class:`CacheSettings` fields."""

# Kernel-level kwargs the Solver routes to the kernel.
ALL_KERNEL_PARAMETERS = (
    frozenset(
        {
            "max_registers",
            "kernel_name",
            "cache",
            "blocksize",
        }
    )
    | ALL_CACHE_PARAMETERS
)

KERNEL_PLACEMENT_PARAMETERS = frozenset(
    {"stage_increment_location", "accumulator_location", "state_location"}
)
"""Buffer placements the kernel owns and resolves."""

KERNEL_PERFORMANCE_PARAMETERS = (
    ALL_UNROLL_PARAMETERS | KERNEL_PLACEMENT_PARAMETERS | {"blocksize"}
)
"""Keys ``auto_performance`` and ``Solver.optimize`` may set."""

DEFAULT_BLOCKSIZE = 64
"""Threads per block when ``blocksize`` is not given."""

SHARED_STAGE_INCREMENT_MIN_STATES = 20
"""A FIRK ``stage_increment`` goes to shared memory above this state
count."""


def _location_field():
    return attrs.field(
        default=None,
        validator=val.optional(val.in_(("local", "shared"))),
    )


def _unroll_field():
    return attrs.field(
        default=None, converter=converters.optional(unroll_flag_converter)
    )


def cache_settings_converter(
    value: Union[CacheSettings, bool, str, Path, None],
) -> CacheSettings:
    """Accept a CacheSettings or the ``cache`` shorthand."""
    if isinstance(value, CacheSettings):
        return value
    if value in (False, None):
        return CacheSettings(cache_enabled=False)
    if value is True:
        return CacheSettings()
    if value == "flush_on_change":
        return CacheSettings(cache_mode="flush_on_change")
    return CacheSettings(cache_dir=Path(value))


def _as_int_tuple(value: Tuple) -> Tuple[int, ...]:
    """Coerce an iterable of dimension sizes to a tuple of ints."""
    return tuple(int(dim) for dim in value)


def _three_dims(instance, attribute, value) -> None:
    """Validate that a shape tuple has exactly three dimensions."""
    if len(value) != 3:
        raise ValueError(
            f"{attribute.name} must have three dimensions, got {value}."
        )


@attrs.frozen
class BatchSolverConfig(CUDAFactoryConfig):
    """Compile-critical settings for the batch solver kernel.

    Attributes
    ----------
    precision
        NumPy floating-point data type used for host and device arrays.
    loop_fn
        CUDA device loop function generated by :class:`SingleIntegratorRun`.
    compile_flags
        Boolean compile-time controls for output features.
    max_registers
        Per-thread register cap passed to ``cuda.jit``. ``None`` leaves
        allocation to ptxas (currently 255 for large systems, limiting
        occupancy to one block per SM); capping trades spill traffic
        for more resident warps.
    coefficients_shape
        Driver-coefficient layout ``(num_segments, num_drivers,
        order + 1)`` baked into the compiled driver evaluators as
        closure constants. The Solver keeps it aligned with
        ``ArrayInterpolator.coefficients_shape``; input sizing and
        device-array validation check supplied coefficient arrays
        against it. The zero default marks kernels never given driver
        metadata (sizing floors it to a unit placeholder).
    kernel_name
        Name of the compiled kernel function, shown in profiler and
        disassembly output. ``None`` derives
        ``{algorithm}_{system name}``; the LTO state is appended as
        ``_ltoon``/``_ltooff`` either way.
    cache
        :class:`CacheSettings`; accepts the ``cache`` shorthand and
        loose ``cache_*`` keys through ``update``.
    auto_performance
        Resolve the unset performance keys from the system's size and
        the GPU.
    blocksize
        Threads per block for every launch, as given.
    unroll_stage, unroll_step_element, unroll_accumulator,
    unroll_solver_element, unroll_norms, unroll_other_small,
    unroll_newton_exits, unroll_krylov_exits
        The unroll flags as given; ``unroll`` holds the flags in effect.
    stage_increment_location, accumulator_location, state_location
        Buffer placements as given.
    instruction_cache_bytes
        The GPU's instruction cache size.
    n_states, is_implicit, algorithm_family, newton_solves_per_step,
    step_operation_count, system_operation_count
        The run's sizes, flags and operator counts the performance
        rules read.
    """

    loop_fn: Optional[Callable] = device_function_field()
    compile_flags: Optional[OutputCompileFlags] = attrs.field(
        factory=OutputCompileFlags,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(OutputCompileFlags)
        ),
    )
    max_registers: Optional[int] = attrs.field(
        default=None,
        validator=attrs.validators.optional(getype_validator(int, 1)),
    )
    coefficients_shape: Tuple[int, int, int] = attrs.field(
        default=(0, 0, 0),
        converter=_as_int_tuple,
        validator=[
            val.deep_iterable(val.instance_of(int), val.instance_of(tuple)),
            _three_dims,
        ],
    )
    kernel_name: Optional[str] = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(str)
        ),
    )
    cache: CacheSettings = attrs.field(
        factory=CacheSettings,
        converter=cache_settings_converter,
        validator=val.instance_of(CacheSettings),
        eq=False,
    )
    auto_performance: bool = attrs.field(
        default=True, validator=val.instance_of(bool), eq=False
    )
    _blocksize: Optional[int] = attrs.field(
        default=None,
        validator=val.optional(getype_validator(int, 1)),
        eq=False,
    )
    _unroll_stage: Optional[UnrollFlag] = _unroll_field()
    _unroll_step_element: Optional[UnrollFlag] = _unroll_field()
    _unroll_accumulator: Optional[UnrollFlag] = _unroll_field()
    _unroll_solver_element: Optional[UnrollFlag] = _unroll_field()
    _unroll_norms: Optional[UnrollFlag] = _unroll_field()
    _unroll_other_small: Optional[UnrollFlag] = _unroll_field()
    _unroll_newton_exits: Optional[UnrollFlag] = _unroll_field()
    _unroll_krylov_exits: Optional[UnrollFlag] = _unroll_field()
    _stage_increment_location: Optional[str] = _location_field()
    _accumulator_location: Optional[str] = _location_field()
    _state_location: Optional[str] = _location_field()
    instruction_cache_bytes: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )
    n_states: int = attrs.field(default=1, validator=getype_validator(int, 1))
    is_implicit: bool = attrs.field(
        default=False, validator=val.instance_of(bool)
    )
    algorithm_family: str = attrs.field(
        default="", validator=val.instance_of(str)
    )
    newton_solves_per_step: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )
    step_operation_count: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )
    system_operation_count: int = attrs.field(
        default=0, validator=getype_validator(int, 0)
    )

    def __attrs_post_init__(self):
        """Resolve the unroll flags in effect from the given ones."""
        super().__attrs_post_init__()
        flags = {
            name: self._resolved_unroll(name) for name in ALL_UNROLL_PARAMETERS
        }
        object.__setattr__(self, "unroll", UnrollFlags(**flags))

    def _resolved_unroll(self, name: str) -> UnrollFlag:
        """Given, else the Newton rule, else the flag's default."""
        given = getattr(self, f"_{name}")
        if given is not None:
            return given
        if name == "unroll_newton_exits" and self.newton_rule_applies:
            unrolled = self.system_operation_count + self.step_operation_count
            capacity = self.instruction_cache_bytes // SASS_INSTRUCTION_BYTES
            if unrolled > capacity:
                return unroll_flag_converter(UnrollChoice.ROLLED)
            return unroll_flag_converter(UnrollChoice.FULL)
        return getattr(UnrollFlags(), name)

    @property
    def newton_rule_applies(self) -> bool:
        """Whether Newton unrolling follows the instruction cache."""
        return (
            self.auto_performance
            and self.is_implicit
            and self.newton_solves_per_step > 0
        )

    @property
    def blocksize(self) -> int:
        """Return the threads per block in effect."""
        if self._blocksize is None:
            return DEFAULT_BLOCKSIZE
        return self._blocksize

    @property
    def blocksize_given(self) -> bool:
        """Return whether ``blocksize`` was given."""
        return self._blocksize is not None

    @property
    def stage_increment_location(self) -> str:
        """Return the ``stage_increment`` placement: given, else shared
        for a FIRK step above the state-count cut."""
        if self._stage_increment_location is not None:
            return self._stage_increment_location
        shared = (
            self.auto_performance
            and self.algorithm_family == "firk"
            and self.n_states > SHARED_STAGE_INCREMENT_MIN_STATES
        )
        return "shared" if shared else "local"

    @property
    def accumulator_location(self) -> str:
        """Return the ``accumulator`` placement: given, else local."""
        if self._accumulator_location is not None:
            return self._accumulator_location
        return "local"

    @property
    def state_location(self) -> str:
        """Return the loop's ``state`` placement: given, else local."""
        if self._state_location is not None:
            return self._state_location
        return "local"

    @property
    def performance_settings(self) -> Dict[str, Any]:
        """Flags and placements in effect, keyed as the children take them."""
        return {
            "unroll": self.unroll,
            "stage_increment_location": self.stage_increment_location,
            "accumulator_location": self.accumulator_location,
            "state_location": self.state_location,
        }

    @property
    def performance_given(self) -> Set[str]:
        """Return the performance keys that were given."""
        return {
            key
            for key in KERNEL_PERFORMANCE_PARAMETERS
            if getattr(self, f"_{key}") is not None
        }

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Derive ActiveOutputs from compile_flags."""
        return ActiveOutputs.from_compile_flags(self.compile_flags)
