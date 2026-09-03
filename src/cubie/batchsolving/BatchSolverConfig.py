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
from typing import Any, Callable, Dict, Optional, Tuple, Union

import attrs
from attrs import validators as val

from cubie._env import kernel_cache_dir_default, max_cache_entries_default
from cubie._utils import (
    getype_validator,
    is_device_validator,
)
from cubie.CUDAFactory import CUDAFactoryConfig, _CubieConfigBase
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
class CacheSettings:
    """Disk-cache settings: enabled, mode, entry limit, directory."""

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

    def update(
        self, updates_dict: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Tuple["CacheSettings", set, set]:
        """Derive a replacement from loose ``cache_*`` keys."""
        if updates_dict is None:
            updates_dict = {}
        updates_dict = {**updates_dict, **kwargs}
        recognized = set()
        changed = set()
        replacements = {}
        for key, value in updates_dict.items():
            if key not in ALL_CACHE_PARAMETERS:
                continue
            recognized.add(key)
            replacements[key] = value
        if not replacements:
            return self, recognized, changed
        candidate = attrs.evolve(self, **replacements)
        for key in replacements:
            if getattr(self, key) != getattr(candidate, key):
                changed.add(key)
        if not changed:
            return self, recognized, changed
        return candidate, recognized, changed


ALL_CACHE_PARAMETERS = frozenset(
    fld.name for fld in attrs.fields(CacheSettings)
)
"""Loose keyword names of the :class:`CacheSettings` fields."""

# Kernel-level kwargs the Solver routes to BatchSolverConfig.
ALL_KERNEL_PARAMETERS = (
    frozenset({"max_registers", "kernel_name", "cache"})
    | ALL_CACHE_PARAMETERS
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
    driver_coefficients_shape
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
        Hash-excluded :class:`CacheSettings`; accepts the ``cache``
        shorthand and loose ``cache_*`` keys through ``update``.
    """

    loop_fn: Optional[Callable] = attrs.field(
        default=None,
        validator=attrs.validators.optional(is_device_validator),
        eq=False,
    )
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
    driver_coefficients_shape: Tuple[int, int, int] = attrs.field(
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

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Derive ActiveOutputs from compile_flags."""
        return ActiveOutputs.from_compile_flags(self.compile_flags)
