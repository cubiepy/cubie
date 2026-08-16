"""Base classes for constructing cached CUDA device functions with Numba.

Published Classes
-----------------
:class:`CUDAFactoryConfig`
    Base attrs config with ``precision`` field and Numba type
    conversions.

    >>> from numpy import float64
    >>> cfg = CUDAFactoryConfig(precision=float64)
    >>> cfg.numba_precision  # doctest: +SKIP
    float64

:class:`CUDADispatcherCache`
    Base class for cache containers holding compiled device functions.

:class:`CUDAFactory`
    Abstract factory base; manages compile settings and cache
    invalidation.

:class:`MultipleInstanceCUDAFactoryConfig`
    Config subclass for components with prefixed parameters (e.g.,
    ``krylov_atol``).

:class:`MultipleInstanceCUDAFactory`
    Factory subclass that maps prefixed configuration keys.

Notes
-----
Compile-settings objects are immutable snapshots. The only write path
is :meth:`CUDAFactory.update_compile_settings`, which derives a
replacement snapshot through the config's pure :meth:`update`, swaps it
in, and invalidates the factory's build cache when any field — semantic
or ``eq=False`` derived — actually changed. ``values_hash`` is a pure
derivation of the snapshot's semantic fields through the canonical
serializer in :mod:`cubie._serialize`; invalidation and hashing are
deliberately different predicates, because a replaced ``eq=False``
device callable must rebuild the consumer while its semantic identity
is carried by the owning child factory's ``config_hash``.

See Also
--------
:mod:`cubie._serialize`
    Canonical serialization used for all configuration hashing.
:mod:`cubie.buffer_registry`
    Buffer registry used by factories for memory management.
:mod:`cubie._utils`
    Validator and converter helpers used by config classes.
"""

from abc import ABC, abstractmethod
from functools import cache
from itertools import count
from typing import Any, Dict, Optional, Set, Tuple

from attrs import (
    Attribute,
    asdict,
    define,
    evolve,
    field,
    fields,
    frozen,
    has,
)
from attrs import validators as attrs_validators
from numpy import (
    array_equal,
    asarray,
    ndarray,
    dtype as np_dtype,
)
from cubie.cuda_simsafe import numba_from_dtype as from_dtype

from cubie._serialize import canonical_digest
from cubie._utils import (
    in_attr,
    PrecisionDType,
    precision_validator,
    precision_converter,
)
from cubie.cuda_simsafe import JITFlags, get_jit_kwargs
from cubie.cuda_simsafe import from_dtype as simsafe_dtype
from cubie.buffer_registry import buffer_registry


_factory_uid_counter = count()
"""Monotonic ids for factories; keys child-invalidation snapshots."""


def attribute_is_hashable(attribute: Attribute, value: Any) -> bool:
    """Check if an attribute participates in semantic equality.

    Parameters
    ----------
    attribute
        An attrs field attribute.
    value
        Value of the attribute.

    Returns
    -------
    bool
        True if the field participates in equality and hashing,
        False for ``eq=False`` fields.
    """
    return attribute.eq is not False


@cache
def _config_field_map(cls: type) -> Dict[str, Attribute]:
    """Return the name-and-alias lookup table for a config class.

    Also validates, once per class, that no equality-participating
    field is a plain ``dict`` — compile-critical mappings must be
    wrapped in their own attrs class.
    """
    from typing import get_origin

    field_map: Dict[str, Attribute] = {}
    for fld in fields(cls):
        field_map[fld.name] = fld
        if fld.alias is not None:
            field_map[fld.alias] = fld
        if fld.eq is not False and (
            (get_origin(fld.type) is dict) or fld.type is dict
        ):
            raise TypeError(
                "Fields of type 'dict' are not supported in "
                "CUDAFactoryConfig subclasses, as they're not hashable, "
                "cacheable, and their entries are not easily updated by "
                "the update() method. Please create an attrs class for "
                "the compile-critical data you're adding."
            )
    return field_map


@cache
def _nested_config_fields(cls: type) -> Tuple[Attribute, ...]:
    """Return fields whose declared type is an attrs class.

    ``Optional``/``Union`` annotations are unwrapped so an optional
    nested config still participates in recursive updates.
    """
    from typing import Union, get_args, get_origin

    nested = []
    for fld in fields(cls):
        candidates = (fld.type,)
        if get_origin(fld.type) is Union:
            candidates = get_args(fld.type)
        for candidate in candidates:
            if isinstance(candidate, type) and has(candidate):
                nested.append(fld)
                break
    return tuple(nested)


def _values_differ(fld: Attribute, old: Any, new: Any) -> bool:
    """Return whether a field's value changed across a snapshot.

    ``eq=False`` fields (derived callables, device-function handles)
    are compared by identity: a replaced object is a change even
    though it never participates in semantic equality or hashing.
    Arrays are compared elementwise; everything else by ``!=``.
    """
    if fld.eq is False:
        return old is not new
    if isinstance(old, ndarray) or isinstance(new, ndarray):
        return not array_equal(asarray(old), asarray(new))
    return bool(old != new)


@frozen
class _CubieConfigBase:
    """Immutable base for configuration containers with session state.

    Instances are frozen snapshots: fields change only by deriving a
    replacement through :meth:`update`, never by assignment. The
    semantic hash is a pure derivation of the snapshot, memoized on
    first access.

    Memoizing the hash is only sound because nested values seal at
    the snapshot boundary: converters on array-valued fields store
    owned read-only copies (never aliases of caller arrays), and
    container values such as ``SystemValues`` freeze in place when a
    snapshot takes them. A converter that stores a mutable alias
    would silently break this contract — copy and seal instead.
    """

    _values_hash_memo: Optional[str] = field(
        default=None, init=False, repr=False, eq=False
    )

    def __attrs_post_init__(self):
        """Trigger the per-class field-map validation."""
        _config_field_map(type(self))

    def update(
        self, updates_dict: dict = None, **kwargs
    ) -> Tuple["_CubieConfigBase", Set[str], Set[str]]:
        """Derive a replacement snapshot with new field values.

        Parameters
        ----------
        updates_dict
            Mapping of setting names to new values. Keys should be
            non-underscored field names (e.g., ``"precision"`` not
            ``"_precision"``).
        **kwargs
            Additional settings to update.

        Returns
        -------
        tuple[_CubieConfigBase, set[str], set[str]]
            replacement: A new snapshot carrying the updates, or
            ``self`` when nothing changed.
            recognized: Names of settings that matched known fields.
            changed: Names of settings whose values differ in the
            replacement.

        Notes
        -----
        This method never mutates ``self``. Field converters and
        validators run on the replacement snapshot, and change
        detection compares post-conversion values — ``eq=False``
        fields by identity, arrays elementwise, everything else by
        inequality. Fields tagged ``metadata={"constructor_only":
        True}`` are settable only at construction: update treats
        their keys as unrecognised. Nested attrs-class fields are
        updated recursively:
        the nested object derives its own replacement, which is folded
        into this snapshot's replacement.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        updates_dict.update(kwargs)
        if not updates_dict:
            return self, set(), set()

        cls = type(self)
        field_map = _config_field_map(cls)

        recognized = set()
        direct = {}
        for key, value in updates_dict.items():
            fld = field_map.get(key)
            if fld is None or not fld.init:
                continue
            if fld.metadata.get("constructor_only"):
                continue
            recognized.add(key)
            direct[key] = fld

        evolve_kwargs = {}
        for key, fld in direct.items():
            evolve_kwargs[fld.alias or fld.name] = updates_dict[key]

        changed = set()
        for fld in _nested_config_fields(cls):
            nested_obj = getattr(self, fld.name)
            if nested_obj is None:
                continue
            new_nested, nested_recognized, nested_changed = nested_obj.update(
                updates_dict
            )
            recognized.update(nested_recognized)
            if nested_changed:
                evolve_kwargs[fld.alias or fld.name] = new_nested
                changed.update(nested_changed)

        if not evolve_kwargs:
            return self, recognized, set()

        candidate = evolve(self, **evolve_kwargs)

        for key, fld in direct.items():
            old_value = getattr(self, fld.name)
            new_value = getattr(candidate, fld.name)
            if _values_differ(fld, old_value, new_value):
                changed.add(key)

        if not changed:
            return self, recognized, set()
        return candidate, recognized, changed

    @property
    def cache_dict(self):
        """Return a dict of all attrs fields without eq=False, for saving
        and loading complete state."""
        return asdict(self, recurse=True, filter=attribute_is_hashable)

    @property
    def values_hash(self) -> str:
        """Canonical digest of the snapshot's semantic fields.

        Returns
        -------
        str
            64-character hex string identifying the configuration
            state, derived through
            :func:`cubie._serialize.canonical_digest` and memoized on
            the immutable snapshot.
        """
        memo = self._values_hash_memo
        if memo is None:
            memo = canonical_digest(self)
            object.__setattr__(self, "_values_hash_memo", memo)
        return memo


@frozen
class CUDAFactoryConfig(_CubieConfigBase):
    """Base class for CUDAFactory compile settings containers.

    Provides infrastructure for tracking configuration values and
    computing stable hashes for cache key generation. Subclasses are
    defined with the ``@attrs.frozen`` decorator.

    Notes
    -----
    Instances are immutable: direct attribute assignment raises
    :class:`attrs.exceptions.FrozenInstanceError`. All changes flow
    through :meth:`CUDAFactory.update_compile_settings`, which derives
    a replacement snapshot and invalidates the factory when any field
    changed. Fields with ``eq=False`` are excluded from hashing
    (typically callables or device functions) but a replaced value
    still counts as a change for invalidation.
    """

    precision: PrecisionDType = field(
        validator=precision_validator,
        converter=precision_converter,
    )
    jit_flags: JITFlags = field(
        factory=JITFlags,
        validator=attrs_validators.instance_of(JITFlags),
        kw_only=True,
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def lineinfo(self) -> bool:
        """Return the lineinfo flag from the jit compile flags."""
        return self.jit_flags.lineinfo

    @property
    def numba_precision(self) -> type:
        """Return the Numba dtype associated with ``precision``."""

        return from_dtype(np_dtype(self.precision))

    @property
    def simsafe_precision(self) -> type:
        """Return the CUDA-simulator-safe dtype for ``precision``."""

        return simsafe_dtype(np_dtype(self.precision))


@define
class CUDADispatcherCache:
    """Base class for CUDAFactory device function Dispatchers.

    Cache containers are build products, not compile settings: they
    may be mutable (invariant: settings snapshots are immutable, cache
    result containers need not be).
    """

    pass


class CUDAFactory(ABC):
    """Factory for creating and caching CUDA device functions.

    Subclasses implement :meth:`build` to construct Numba CUDA device functions
    or other cached outputs. Compile settings are stored as immutable attrs
    snapshots and any change invalidates the cache to ensure functions are
    rebuilt when needed.

    Attributes
    ----------
    _compile_settings : attrs class or None
        Current compile-settings snapshot.
    _cache_valid : bool
        Indicates whether cached outputs are valid.
    _cache : attrs class or None
        Container for cached outputs (CUDADispatcherCache subclass).

    Notes
    -----
    There is potential for a cache mismatch when doing the following:

    ```python
    device_function = self.device_function  # calls build if settings updated
    self.update_compile_settings(new_setting=value)  # updates settings but
    does not rebuild

    device_function(argument_derived_from_new_setting)  # this will use the
    old device function, not the new one
    ```

    The lesson is: Always use CUDAFactory.device_function at the point of
    use, otherwise you'll defeat the cache invalidation logic.

    If your build function returns multiple cached items, create a cache
    class decorated with @attrs.define. For example:
    ```python
    @attrs.define
    class MyCache:
        device_function: callable
        other_output: int
    ```
    Then, in your build method, return an instance of this class:
    ```python
    def build(self):
        return MyCache(device_function=my_device_function, other_output=42)
    ```

    The current cache validity can be checked using the `cache_valid` property,
    which will return True if the cache
    is valid and False otherwise.
    """

    def __init__(self):
        """Initialize the CUDA factory.

        Notes
        -----
        Uses the global default time logger from cubie.time_logger.
        Configure timing via solve_ivp(time_logging_level=...) or
        Solver(time_logging_level=...).
        """
        self._compile_settings = None
        self._cache_valid = True
        self._cache = None
        self._factory_uid = next(_factory_uid_counter)
        self._invalidation_count = 0
        self._built_child_invalidations = {}
        self._child_names_memo = None

    @abstractmethod
    def build(self):
        """Build and return the CUDA device function.

        This method must be overridden by subclasses.

        Returns
        -------
        callable or attrs class
            Compiled CUDA function or container of cached outputs.
        """
        return None

    def setup_compile_settings(self, compile_settings):
        """Attach a container of compile-critical settings to the object.

        Parameters
        ----------
        compile_settings : attrs class
            Settings snapshot used to configure the CUDA function.

        Notes
        -----
        Any existing settings are replaced.
        """
        if not has(type(compile_settings)):
            raise TypeError(
                "Compile settings must be an attrs class instance."
            )
        self._compile_settings = compile_settings
        self._invalidate_cache()

    @property
    def cache_valid(self):
        """bool: ``True`` if cached outputs are up to date.

        A factory's outputs are stale when its own settings changed or
        when any owned child factory was invalidated after this
        factory's last build. Observing a stale child marks this
        factory invalid, so staleness propagates to every ancestor
        that checks its cache — the route by which a change made
        directly on a nested factory (for example
        ``system.set_constants``) reaches a live solver's kernel.
        """
        if not self._cache_valid:
            return False
        snapshot = self._built_child_invalidations
        for child in self._iter_child_factories():
            # Consulting the child lets it observe its own children
            # and advance its counter; the staleness criterion is the
            # counter alone, so a never-built child (a service whose
            # product this factory never consumed) does not read as
            # stale forever.
            child.cache_valid
            if (
                snapshot.get(child._factory_uid)
                != child._invalidation_count
            ):
                self._invalidate_cache()
                return False
        return True

    @property
    def device_function(self):
        """Return the compiled CUDA device function.

        Returns
        -------
        callable
            Compiled CUDA device function.
        """
        return self.get_cached_output("device_function")

    @property
    def compile_settings(self):
        """Return the current compile-settings snapshot."""
        return self._compile_settings

    @property
    def jit_kwargs(self) -> dict:
        """Return ``cuda.jit`` keyword arguments for this factory.

        Renders the compile settings' :class:`JITFlags` through
        :func:`cubie.cuda_simsafe.get_jit_kwargs` — the single route
        by which jit arguments reach ``@cuda.jit`` decorators in
        ``build()`` implementations.

        Returns
        -------
        dict
            Keyword arguments to splat into ``cuda.jit``.
        """
        return get_jit_kwargs(self.compile_settings.jit_flags)

    def update_compile_settings(
        self, updates_dict=None, silent=False, **kwargs
    ) -> Set[str]:
        """Update compile settings with new values.

        This is the sole write boundary for compile settings: it
        derives a replacement snapshot through the config's pure
        :meth:`update`, swaps it in, and invalidates the build cache
        when any field changed.

        Parameters
        ----------
        updates_dict : dict, optional
            Mapping of setting names to new values.
        silent : bool, default=False
            Suppress errors for unrecognised parameters.
        **kwargs
            Additional settings to update.

        Returns
        -------
        set[str]
            Names of settings that were successfully updated.

        Raises
        ------
        ValueError
            If compile settings have not been set up.
        KeyError
            If an unrecognised parameter is supplied and ``silent`` is
            ``False``.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        if self._compile_settings is None:
            raise ValueError(
                "Compile settings must be set up using "
                "self.setup_compile_settings before updating."
            )
        replacement, recognized, changed = self._compile_settings.update(
            updates_dict
        )

        unrecognised = set(updates_dict.keys()) - recognized
        if unrecognised and not silent:
            invalid = ", ".join(sorted(unrecognised))
            raise KeyError(
                f"'{invalid}' is not a valid compile setting for this "
                "object, and so was not updated.",
            )
        if changed:
            self._compile_settings = replacement
            self._invalidate_cache()

        return recognized

    def _invalidate_cache(self):
        """Mark cached Dispatchers as invalid.

        Also advances the invalidation counter parents snapshot at
        build time, so a parent that built against the previous state
        of this factory reads as stale through :attr:`cache_valid`.
        """
        self._cache_valid = False
        self._invalidation_count += 1

    def _build(self):
        """Rebuild cached outputs if they are invalid."""
        build_result = self.build()

        if not isinstance(build_result, CUDADispatcherCache):
            raise TypeError(
                "build() must return an attrs class (CUDADispatcherCache "
                "subclass)"
            )

        self._cache = build_result
        self._cache_valid = True
        # Snapshot children's invalidation counters: rebuilding does
        # not advance a counter, so products fetched from children
        # during build() stay matched to the recorded state.
        self._built_child_invalidations = {
            child._factory_uid: child._invalidation_count
            for child in self._iter_child_factories()
        }

    def get_cached_output(self, output_name):
        """Return a named cached output.

        Parameters
        ----------
        output_name : str
            Name of the cached item to retrieve.

        Returns
        -------
        Any
            Cached value associated with ``output_name``.

        Raises
        ------
        KeyError
            If ``output_name`` is not present in the cache.
        """
        if not self.cache_valid:
            self._build()
        if self._cache is None:
            raise RuntimeError("Cache has not been initialized by build().")
        if not in_attr(output_name, self._cache):
            raise KeyError(
                f"Output '{output_name}' not found in cached outputs."
            )
        return getattr(self._cache, output_name)

    @property
    def config_hash(self):
        """Returns the hash of the current compile settings of the factory.
        If the factory has child factories, their hashes will be combined
        with this factory's own hash through the canonical serializer.
        """
        own_hash = self.compile_settings.values_hash
        child_hashes = tuple(
            child_factory.config_hash
            for child_factory in self._iter_child_factories()
        )
        if child_hashes:
            return canonical_digest(
                ("cubie-config-hash", own_hash, child_hashes)
            )
        return own_hash

    _excluded_child_factories: frozenset = frozenset()
    """Attribute names excluded from child-factory discovery.

    Owned child factories must be direct attributes so
    :meth:`config_hash` recurses into them. A subclass lists an
    attribute here only when the factory it holds is a diagnostic
    service whose configuration deliberately does not shape this
    factory's built products — excluded factories contribute nothing
    to semantic identity.
    """

    def __setattr__(self, name, value):
        """Set an attribute, invalidating the child-name memo.

        Assigning a factory (or overwriting a memoized child name)
        drops the memoized discovery so :meth:`_iter_child_factories`
        re-scans on its next call.
        """
        memo = self.__dict__.get("_child_names_memo")
        if memo is not None and (
            isinstance(value, CUDAFactory) or name in memo
        ):
            self.__dict__["_child_names_memo"] = None
        object.__setattr__(self, name, value)

    def _iter_child_factories(self):
        """Yield direct attribute values that are CUDAFactory instances.

        Only inspects immediate attributes (no nested attrs/dicts/iterables).
        Each child is yielded once (uniqueness by id). Attributes are sorted
        alphabetically by name for deterministic ordering. Names in
        :attr:`_excluded_child_factories` are skipped. Discovery is
        memoized per instance; :meth:`__setattr__` drops the memo when
        a factory-valued attribute is assigned.
        """
        names = self.__dict__.get("_child_names_memo")
        if names is None:
            excluded = self._excluded_child_factories
            names = tuple(
                name
                for name in sorted(vars(self).keys())
                if name not in excluded
                and isinstance(
                    self.__dict__.get(name), CUDAFactory
                )
            )
            self.__dict__["_child_names_memo"] = names
        seen = set()
        for name in names:
            val = self.__dict__.get(name)
            if isinstance(val, CUDAFactory):
                oid = id(val)
                if oid not in seen:
                    seen.add(oid)
                    yield val

    @property
    def precision(self) -> type:
        """Return the precision dtype used by compiled device functions."""
        return self.compile_settings.precision

    @property
    def numba_precision(self) -> type:
        """Return the Numba dtype used by compiled device functions."""

        return self.compile_settings.numba_precision

    @property
    def simsafe_precision(self) -> type:
        """Return the CUDA-simulator-safe dtype for the functions."""

        return self.compile_settings.simsafe_precision

    @property
    def shared_buffer_size(self) -> int:
        """Return total shared memory elements registered for this factory.

        Returns
        -------
        int
            Total shared memory elements from buffer_registry.
        """
        return buffer_registry.shared_buffer_size(self)

    @property
    def local_buffer_size(self) -> int:
        """Return total local memory elements registered for this factory.

        Returns
        -------
        int
            Total local memory elements from buffer_registry.
        """
        return buffer_registry.local_buffer_size(self)

    @property
    def persistent_local_buffer_size(self) -> int:
        """Return total persistent local elements registered for this factory.

        Returns
        -------
        int
            Total persistent local elements from buffer_registry.
        """
        return buffer_registry.persistent_local_buffer_size(self)


@frozen
class MultipleInstanceCUDAFactoryConfig(CUDAFactoryConfig):
    """Extends CUDAFactoryConfig for instances which have multiple
    concurrent configurations - e.g. a newton and krylov solver. Provides a
    set of prefixed keys for non-shared parameters, for example `newton_atol`
    or `krylov_atol`. Performs the substitution internally on update,
    however requires non-prefixed keys for init. See `build_config` for a
    utility function that initialises this class from prefixed keys.
    """

    instance_label: str = field(default="", repr=False, eq=False)
    prefixed_attributes: frozenset = field(
        factory=frozenset,
        converter=frozenset,
        repr=False,
        eq=False,
    )

    @classmethod
    def get_prefixed_attributes(cls, aliases: bool = False) -> Set[str]:
        """Return names of attributes that use instance-specific prefixes.

        Parameters
        ----------
        aliases
            If True, return field aliases instead of names. Use this when
            using the prefixed attributes for initialization, as it provides
            the non-underscored names which the generated __init__ function
            accepts.
        Returns
        -------
        set[str]
            Names or aliases of attributes that are prefixed based on
            ``instance_label``.
        """
        prefixed = set()

        for fld in fields(cls):
            if getattr(fld, "metadata", None) is not None:
                if fld.metadata.get("prefixed", False):
                    if aliases:
                        prefixed.add(
                            fld.alias if fld.alias is not None else fld.name
                        )
                    else:
                        prefixed.add(fld.name)

        return prefixed

    @property
    def prefix(self) -> str:
        """Return the prefix string for this instance.

        Returns
        -------
        str
            The instance_label value (prefix without trailing underscore).
        """
        return self.instance_label

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.instance_label != "":
            # Aliases keep prefixed keys well-formed for fields with
            # underscored names (e.g. ``_residual_floor`` maps to
            # ``krylov_residual_floor``, not ``krylov__residual_floor``).
            prefixed_attributes = type(self).get_prefixed_attributes(
                aliases=True
            )
            object.__setattr__(
                self, "prefixed_attributes", frozenset(prefixed_attributes)
            )

    def update(
        self, updates_dict: dict = None, **kwargs
    ) -> Tuple["MultipleInstanceCUDAFactoryConfig", Set[str], Set[str]]:
        """Derive a replacement snapshot, handling prefixed keys.

        Parameters
        ----------
        updates_dict
            Mapping of setting names to new values. Keys matching
            ``{prefix}_*`` are mapped to unprefixed equivalents.
        kwargs
            Additional settings to update.

        Returns
        -------
        tuple[MultipleInstanceCUDAFactoryConfig, set[str], set[str]]
            replacement: New snapshot, or ``self`` when unchanged.
            recognized: Names of settings that matched known fields.
            changed: Names of settings whose values were updated.
        """
        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)

        # Get rid of non-prefixed keys; write de-prefixed values in their place
        for key in self.prefixed_attributes:
            prefixed_key = f"{self.prefix}_{key}"
            has_prefixed = prefixed_key in all_updates

            _ = all_updates.pop(key, None)

            if has_prefixed:
                all_updates[key] = all_updates.pop(prefixed_key)

        replacement, recognized_base, changed_base = super().update(
            all_updates
        )

        # Transform recognised keys back into prefixed versions to make as seen
        recognized = set()
        for key in recognized_base:
            if key in self.prefixed_attributes:
                recognized.add(f"{self.prefix}_{key}")
            else:
                recognized.add(key)

        changed = set()
        for key in changed_base:
            if key in self.prefixed_attributes:
                changed.add(f"{self.prefix}_{key}")
            else:
                changed.add(key)

        return replacement, recognized, changed


class MultipleInstanceCUDAFactory(CUDAFactory):
    """Factory for CUDA device functions with instance-specific prefixes.

    Extends CUDAFactory to automatically map prefixed configuration
    keys (e.g., ``krylov_atol``) to unprefixed internal keys (e.g.,
    ``atol``) during settings updates. Subclasses use ``instance_label``
    to differentiate configuration parameters when multiple instances
    coexist.

    Attributes
    ----------
    instance_label : str
        Prefix used to identify settings for this instance
        (e.g., "krylov", "newton"). Keys in update dicts matching
        ``{instance_label}_*`` are mapped to unprefixed equivalents.

    Notes
    -----
    The transformation occurs in the MultipleInstanceCUDAFactoryConfig
    object's update() method. When update_compile_settings() is called
    on the factory, it delegates to the config's update() method which
    handles prefix mapping automatically.
    """

    def __init__(
        self,
        instance_label: str,
    ) -> None:
        """Initialize with instance label for prefix mapping and arguments
        for config.

        Parameters
        ----------
        instance_label : str
            Prefix for external configuration keys. Should NOT
            include trailing underscore (added automatically).
        """
        self._instance_label = instance_label
        super().__init__()

    @property
    def instance_label(self) -> str:
        """Return the instance label for this factory."""
        return self._instance_label
