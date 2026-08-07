"""Centralized buffer registry for CUDA memory management.

This module provides a package-wide singleton registry that manages
buffer metadata for all CUDA factories. Factories register their
buffer requirements during initialization, and the registry provides
CUDA-compatible allocator device functions during build.

The registry uses a lazy cached build pattern - slice layouts are
computed on demand and invalidated when any buffer is modified.

Each buffer has a ``dtype``, which defaults to the run precision of
the parent that registers it. Shared and persistent memory arrays are
typed at the parent's precision, so a buffer whose dtype differs
(e.g. ``np_int32`` iteration counters) receives its slice through a
``view`` of the parent array.
"""

from typing import Callable, Dict, Optional, Tuple, Any, Set
from weakref import WeakKeyDictionary

from attrs import (
    asdict as attrs_asdict,
    define,
    evolve as attrs_evolve,
    field,
)
from attrs.validators import (
    in_ as attrsval_in,
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
)
from numpy import dtype as np_dtype, float32 as np_float32

from cubie.cuda_simsafe import cuda
from cubie.cuda_simsafe import int32
from cubie.cuda_simsafe import float32

from cubie._utils import getype_validator, buffer_dtype_validator
from cubie.cuda_simsafe import compile_kwargs, from_dtype


@define
class CUDABuffer:
    """Immutable record describing a single buffer's requirements.

    Attributes
    ----------
    name : str
        Unique buffer name within parent context.
    size : int
        Buffer size in elements.
    location : str
        Memory location for the buffer: 'shared' or 'local'.
    persistent : bool
        If True and location='local', use persistent_local.
    aliases : str or None
        Name of buffer to alias (must exist in same context).
    dtype : type
        NumPy element type of the buffer.
    parent_dtype : type
        NumPy element type of the parent's shared and persistent
        memory arrays. Stored by the owning group at registration.
    """

    name: str = field(validator=attrsval_instance_of(str))
    size: int = field(validator=getype_validator(int, 0))
    location: str = field(
        validator=attrsval_in(["shared", "local"])
    )
    persistent: bool = field(
        default=False, validator=attrsval_instance_of(bool)
    )
    aliases: Optional[str] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(str))
    )
    dtype: type = field(
        default=np_float32,
        validator=buffer_dtype_validator
    )
    parent_dtype: type = field(
        default=np_float32,
        validator=buffer_dtype_validator
    )

    @property
    def is_shared(self) -> bool:
        """Return True if buffer uses shared memory."""
        return self.location == "shared"

    @property
    def is_persistent_local(self) -> bool:
        """Return True if buffer uses persistent local memory."""
        return self.location == 'local' and self.persistent

    @property
    def is_local(self) -> bool:
        """Return True if buffer uses local (non-persistent) memory."""
        return self.location == "local" and not self.persistent

    @property
    def itemsize(self) -> int:
        """Return the bytes per element of this buffer's dtype."""
        return np_dtype(self.dtype).itemsize

    @property
    def parent_itemsize(self) -> int:
        """Return the bytes per element of the parent's dtype."""
        return np_dtype(self.parent_dtype).itemsize

    @property
    def needs_view(self) -> bool:
        """Return True when this buffer's dtype differs from the parent's."""
        return np_dtype(self.dtype) != np_dtype(self.parent_dtype)

    @property
    def parent_elements(self) -> int:
        """Return the number of parent elements this buffer requires.

        When the parent's dtype has a different itemsize, the buffer
        occupies enough parent elements to cover its bytes.
        """
        own_bytes = self.size * self.itemsize
        return (own_bytes + self.parent_itemsize - 1) // self.parent_itemsize

    def aligned_offset(self, offset: int) -> int:
        """Return ``offset`` rounded up to this buffer's alignment.

        A view to a dtype wider than the parent's is only valid when
        the slice starts on a whole-element boundary of the wider
        dtype.

        Parameters
        ----------
        offset
            Candidate start offset in parent elements.

        Returns
        -------
        int
            Aligned start offset in parent elements.
        """
        if self.itemsize <= self.parent_itemsize:
            return offset
        ratio = self.itemsize // self.parent_itemsize
        return (offset + ratio - 1) // ratio * ratio

    def build_allocator(
        self,
        shared_slice: Optional[slice],
        persistent_slice: Optional[slice],
        local_size: Optional[int],
        zero: bool = False,
    ) -> Callable:
        """Compile CUDA device function for buffer allocation.

        Generates an inlined device function that allocates this buffer
        from the appropriate memory region based on which slice parameters
        are provided.

        Parameters
        ----------
        shared_slice
            Slice into shared memory, or None if not using shared.
        persistent_slice
            Slice into persistent local memory, or None if not using
            persistent.
        local_size
            Size for local array allocation, or None if not local.
        zero
            If True, initialize all elements to zero after allocation.

        Returns
        -------
        Callable
            CUDA device function: (shared, persistent) -> array

            The device function accepts shared and persistent memory
            arrays and returns a view/slice into the appropriate memory
            region, or allocates a fresh local array.

        Notes
        -----
        Shared and persistent slices index parent memory typed at the
        parent's dtype; when this buffer's dtype differs, the slice is
        reinterpreted with ``view`` so the returned array carries the
        registered dtype.

        When a buffer aliases another buffer and aliasing succeeds, both
        buffers receive slices in the same memory region (shared or
        persistent) that overlap. The allocator transparently provides
        the correct view without needing a separate parent reference.
        """
        # Compile-time constants captured in closure
        _use_shared = shared_slice is not None
        _use_persistent = persistent_slice is not None
        _shared_slice = shared_slice if _use_shared else slice(0, 0)
        _persistent_slice = (
            persistent_slice if _use_persistent else slice(0, 0)
        )
        _local_size = local_size if local_size is not None else 1
        _dtype = self.dtype
        _needs_view = self.needs_view
        _zero = zero
        elements = int32(self.size)

        # no cover: start
        @cuda.jit(device=True, inline=True, **compile_kwargs)
        def allocate_buffer(shared, persistent):
            """Allocate buffer from appropriate memory region."""
            if _use_shared:
                if _needs_view:
                    array = shared[_shared_slice].view(_dtype)
                else:
                    array = shared[_shared_slice]
            elif _use_persistent:
                if _needs_view:
                    array = persistent[_persistent_slice].view(_dtype)
                else:
                    array = persistent[_persistent_slice]
            else:
                array = cuda.local.array(_local_size, _dtype)
            if _zero:
                for i in range(elements):
                    array[i] = _dtype(0.0)
            return array

        # no cover: end
        return allocate_buffer


@define
class BufferGroup:
    """Groups all buffer entries for a single parent object.

    Attributes
    ----------
    entries : Dict[str, CUDABuffer]
        Registered buffers by name.
    children : Dict[str, object]
        Child instances whose buffers this parent hosts, keyed by
        the registration base name, recorded by
        :meth:`BufferRegistry.register_child`. Re-registering a
        name replaces the recorded child.
    parent_dtype : type
        NumPy element type of the parent's shared and persistent
        memory arrays. Shared and persistent layouts are measured
        in elements of this type. Set from the parent's
        ``precision`` on every registration.
    _shared_layout : Dict[str, slice] or None
        Cached unified shared memory layout (None when invalid).
    _persistent_layout : Dict[str, slice] or None
        Cached persistent_local slices (None when invalid).
    _local_sizes : Dict[str, int] or None
        Cached local sizes (None when invalid).
    _alias_consumption : Dict[str, int]
        Tracks consumed space in aliased buffers for layout
        computation.
    """

    entries: Dict[str, CUDABuffer] = field(factory=dict)
    children: Dict[str, object] = field(factory=dict, init=False)
    parent_dtype: type = field(default=np_float32)
    _shared_layout: Optional[Dict[str, slice]] = field(
        default=None, init=False
    )
    _persistent_layout: Optional[Dict[str, slice]] = field(
        default=None, init=False
    )
    _local_sizes: Optional[Dict[str, int]] = field(
        default=None, init=False
    )
    _alias_consumption: Dict[str, int] = field(
        factory=dict, init=False
    )

    def invalidate_layouts(self) -> None:
        """Set all cached layouts to None and clear alias consumption."""
        self._shared_layout = None
        self._persistent_layout = None
        self._local_sizes = None
        self._alias_consumption.clear()

    def set_parent_dtype(self, dtype: type) -> None:
        """Record the element type of the parent's memory arrays.

        A change updates every entry and invalidates cached layouts,
        which are measured in parent elements.

        Parameters
        ----------
        dtype
            NumPy element type of the parent's shared and persistent
            memory arrays.
        """
        if np_dtype(dtype) == np_dtype(self.parent_dtype):
            return
        self.parent_dtype = dtype
        for name, entry in self.entries.items():
            self.entries[name] = attrs_evolve(entry, parent_dtype=dtype)
        self.invalidate_layouts()

    def build_layouts(self) -> None:
        """Build all buffer layouts in deterministic order.

        Orchestrates layout building to ensure consistent results
        regardless of which property is accessed first:

        1. Build non-aliased shared buffers into _shared_layout
        2. Build non-aliased persistent buffers into _persistent_layout
        3. Build non-aliased local buffers into _local_sizes
        4. Call layout_aliases() to handle all aliased entries

        All three layout caches are fully populated after this
        method completes.
        """
        # If all layouts already built, nothing to do
        if (self._shared_layout is not None
                and self._persistent_layout is not None
                and self._local_sizes is not None):
            return

        # Clear state for fresh build
        self._alias_consumption.clear()
        self._shared_layout = {}
        self._persistent_layout = {}
        self._local_sizes = {}

        # Phase 1: Non-aliased shared buffers
        shared_offset = 0
        for name, entry in self.entries.items():
            if entry.is_shared and entry.aliases is None:
                shared_offset = entry.aligned_offset(shared_offset)
                extent = entry.parent_elements
                self._shared_layout[name] = slice(
                    shared_offset, shared_offset + extent
                )
                self._alias_consumption[name] = 0
                shared_offset += extent

        # Phase 2: Non-aliased persistent buffers
        persistent_offset = 0
        for name, entry in self.entries.items():
            if entry.is_persistent_local and entry.aliases is None:
                persistent_offset = entry.aligned_offset(persistent_offset)
                extent = entry.parent_elements
                self._persistent_layout[name] = slice(
                    persistent_offset, persistent_offset + extent
                )
                self._alias_consumption[name] = 0
                persistent_offset += extent

        # Phase 3: Non-aliased local buffers
        for name, entry in self.entries.items():
            if entry.is_local and entry.aliases is None:
                self._local_sizes[name] = max(entry.size, 1)

        # Phase 4: Process all aliased entries
        self.layout_aliases()

    def layout_aliases(self) -> None:
        """Process all aliased entries and assign to appropriate layouts.

        For each entry with aliases is not None:
        - If parent is shared with available space: overlap within parent
        - Else fallback based on entry's own type:
          - is_shared: allocate in _shared_layout
          - is_persistent_local: allocate in _persistent_layout
          - is_local: add to local pile (processed at end)

        Local pile entries are added to _local_sizes after all
        aliasing decisions are made.
        """
        local_pile = []

        # Compute current offsets from existing layouts
        shared_offset = 0
        if self._shared_layout:
            shared_offset = max(s.stop for s in self._shared_layout.values())

        persistent_offset = 0
        if self._persistent_layout:
            persistent_offset = max(
                s.stop for s in self._persistent_layout.values()
            )

        # Process aliased entries
        for name, entry in self.entries.items():
            if entry.aliases is None:
                continue

            parent_entry = self.entries[entry.aliases]
            aliased = False
            extent = entry.parent_elements

            # Check if parent is in shared layout and has space
            if entry.aliases in self._shared_layout:
                consumed = self._alias_consumption.get(entry.aliases, 0)
                available = parent_entry.parent_elements - consumed

                if extent <= available:
                    # Overlap within parent's shared memory
                    parent_slice = self._shared_layout[entry.aliases]
                    start = entry.aligned_offset(
                        parent_slice.start + consumed
                    )
                    if start + extent <= parent_slice.stop:
                        self._shared_layout[name] = slice(
                            start, start + extent
                        )
                        self._alias_consumption[entry.aliases] = (
                            start + extent - parent_slice.start
                        )
                        aliased = True

            # Fallback based on entry's own type
            if not aliased:
                if entry.is_shared:
                    shared_offset = entry.aligned_offset(shared_offset)
                    self._shared_layout[name] = slice(
                        shared_offset, shared_offset + extent
                    )
                    shared_offset += extent
                elif entry.is_persistent_local:
                    persistent_offset = entry.aligned_offset(
                        persistent_offset
                    )
                    self._persistent_layout[name] = slice(
                        persistent_offset, persistent_offset + extent
                    )
                    persistent_offset += extent
                else:
                    # is_local: collect for batch processing
                    local_pile.append(entry)

        # Process local pile
        for entry in local_pile:
            self._local_sizes[entry.name] = max(entry.size, 1)

    def register(
        self,
        name: str,
        size: int,
        location: str,
        persistent: bool = False,
        aliases: Optional[str] = None,
        dtype: Optional[type] = None,
    ) -> None:
        """Register a buffer with this group.

        Parameters
        ----------
        name
            Unique buffer name within this group.
        size
            Buffer size in elements.
        location
            Memory location: 'shared' or 'local'.
        persistent
            If True and location='local', use persistent_local.
        aliases
            Name of buffer to alias (must exist in same context).
        dtype
            NumPy element type of the buffer. Defaults to the
            parent's dtype.

        Raises
        ------
        ValueError
            If aliases references non-existent buffer.
        ValueError
            If name is empty string.
        ValueError
            If buffer attempts to alias itself.

        Notes
        -----
        Registering a name that already exists silently replaces the
        entry and invalidates the layouts. Parents rely on this to
        refresh child buffer sizes by re-registering after children
        have built.
        """
        if not name:
            raise ValueError("Buffer name cannot be empty.")
        if aliases is not None and aliases == name:
            raise ValueError(f"Buffer '{name}' cannot alias itself.")
        if aliases is not None and aliases not in self.entries:
            raise ValueError(
                f"Alias target '{aliases}' not registered. "
                f"Register '{aliases}' before '{name}'."
            )

        if dtype is None:
            dtype = self.parent_dtype
        entry = CUDABuffer(
            name=name,
            size=size,
            location=location,
            persistent=persistent,
            aliases=aliases,
            dtype=dtype,
            parent_dtype=self.parent_dtype,
        )
        self.entries[name] = entry
        self.invalidate_layouts()

    def update_buffer(self, name: str, **kwargs: object) -> Tuple[bool, bool]:
        """Update an existing buffer's properties.

        Parameters
        ----------
        name
            Buffer name to update.
        **kwargs
            Properties to update (size, location, persistent, aliases).

        Returns
        -------
        bool, bool
            Whether the buffer was recognized and updated, respectively.

        Notes
        -----
        Silently ignores updates for buffers not registered.
        """
        recognized = False
        changed = False

        if name not in self.entries:
            recognized = False
            changed = False
        else:
            recognized = True
            old_entry = self.entries[name]
            new_values = attrs_asdict(old_entry)
            new_values.update(kwargs)

            if new_values != attrs_asdict(old_entry):
                changed = True
                self.entries[name] = CUDABuffer(**new_values)
                self.invalidate_layouts()

        return recognized, changed

    @property
    def shared_layout(self) -> Dict[str, slice]:
        """Return shared memory layout.

        Returns
        -------
        Dict[str, slice]
            Mapping of buffer names to shared memory slices.
        """
        if self._shared_layout is None:
            self.build_layouts()
        return self._shared_layout

    @property
    def persistent_layout(self) -> Dict[str, slice]:
        """Return persistent local memory layout.

        Returns
        -------
        Dict[str, slice]
            Mapping of buffer names to persistent local slices.
        """
        if self._persistent_layout is None:
            self.build_layouts()
        return self._persistent_layout

    @property
    def local_sizes(self) -> Dict[str, int]:
        """Return local buffer sizes.

        Returns
        -------
        Dict[str, int]
            Mapping of buffer names to local array sizes.
        """
        if self._local_sizes is None:
            self.build_layouts()
        return self._local_sizes

    def shared_buffer_size(self) -> int:
        """Return total shared memory elements.

        Returns
        -------
        int
            Total shared memory elements needed (end of last slice).
        """
        layout = self.shared_layout
        if not layout:
            return 0
        return max(s.stop for s in layout.values())

    def local_buffer_size(self) -> int:
        """Return total local memory elements.

        Returns
        -------
        int
            Total local memory elements (max(size, 1) for each).
        """
        return sum(self.local_sizes.values())

    def persistent_local_buffer_size(self) -> int:
        """Return total persistent local elements.

        Returns
        -------
        int
            Total persistent_local elements needed (end of last slice).
        """
        layout = self.persistent_layout
        if not layout:
            return 0
        return max(s.stop for s in layout.values())

    def relocatable_names(self) -> Tuple[str, ...]:
        """Return buffer names registered directly on this group.

        Excludes the ``{child}_shared`` and ``{child}_persistent``
        roll-up entries created by
        :meth:`BufferRegistry.register_child`, leaving only buffers
        whose location is set through a ``{name}_location`` setting.
        """
        rollups = set()
        for base in self.children:
            rollups.add(f"{base}_shared")
            rollups.add(f"{base}_persistent")
        return tuple(
            name for name in self.entries if name not in rollups
        )

    def nonaliased_elements(self, names: Tuple[str, ...]) -> int:
        """Return total parent elements the named buffers would allocate.

        Aliased buffers overlap their parent's allocation, so only
        non-aliased entries contribute. Unknown names count zero.
        """
        total = 0
        for name in names:
            entry = self.entries.get(name)
            if entry is not None and entry.aliases is None:
                total += entry.parent_elements
        return total

    def get_allocator(self, name: str, zero: bool = False) -> Callable:
        """Generate CUDA device function for buffer allocation.

        Retrieves the pre-computed memory slice for this buffer from the
        appropriate layout (shared, persistent, or local) and generates
        an allocator that uses that slice.

        Parameters
        ----------
        name
            Buffer name to generate allocator for.
        zero
            If True, initialize all elements to zero after allocation.

        Returns
        -------
        Callable
            CUDA device function that allocates the buffer.
            Signature: (shared, persistent) -> array

        Raises
        ------
        KeyError
            If buffer name not registered.

        Notes
        -----
        The layout building phase (build_layouts) determines which memory
        region each buffer uses and assigns slices accordingly. This method
        simply retrieves those pre-computed slices and creates an allocator
        that uses them.

        For aliased buffers, the layout builder assigns slices that
        overlap the parent buffer, implementing aliasing transparently
        at the slice level.
        """
        if name not in self.entries:
            raise KeyError(f"Buffer '{name}' not registered for parent.")

        entry = self.entries[name]

        # Get slice from appropriate layout (properties trigger build)
        shared_slice = self.shared_layout.get(name)
        persistent_slice = self.persistent_layout.get(name)
        local_size = self.local_sizes.get(name)

        return entry.build_allocator(
            shared_slice,
            persistent_slice,
            local_size,
            zero,
        )


@define
class BufferRegistry:
    """Central registry managing all buffer metadata for CUDA parents.

    The registry is a package-level singleton that tracks buffer
    requirements for all parent objects. It uses lazy cached builds for
    slice/layout computation - layouts are set to None on any change
    and regenerated on access.

    Registration is the only point where a parent's ``precision`` is
    read: every parent re-runs its ``register_buffers()`` when its
    settings change, so a precision change always re-registers before
    any layout or allocator is consumed.

    Attributes
    ----------
    _groups : WeakKeyDictionary
        Maps parent instances to their buffer groups. Parents are
        held weakly: a group disappears with its parent, so dead
        components do not accumulate in the registry across a
        session.
    """

    _groups: WeakKeyDictionary = field(
        factory=WeakKeyDictionary, init=False
    )

    def register(
        self,
        name: str,
        parent: object,
        size: int,
        location: str,
        persistent: bool = False,
        aliases: Optional[str] = None,
        dtype: Optional[type] = None,
    ) -> None:
        """Register a buffer with the central registry.

        Parameters
        ----------
        name
            Unique buffer name within parent context.
        parent
            Parent instance that owns this buffer.
        size
            Buffer size in elements.
        location
            Memory location: 'shared' or 'local'.
        persistent
            If True and location='local', use persistent_local.
        aliases
            Name of buffer to alias (must exist in same context).
        dtype
            NumPy element type of the buffer. Defaults to the
            parent's ``precision``; pass a dtype only for buffers
            that differ from it (e.g. ``np_int32`` counters).

        Raises
        ------
        ValueError
            If aliases references non-existent buffer.
        ValueError
            If name is empty string.
        ValueError
            If buffer attempts to alias itself.
        ValueError
            If dtype is not a supported NumPy type.

        Notes
        -----
        Registering a name that already exists for the parent silently
        replaces the entry; parents refresh child buffer sizes by
        re-registering after children have built.
        """
        if parent not in self._groups:
            self._groups[parent] = BufferGroup()
        group = self._groups[parent]
        group.set_parent_dtype(parent.precision)
        group.register(name, size, location, persistent, aliases, dtype)

    def update_buffer(
        self,
        name: str,
        parent: object,
        **kwargs: object,
    ) -> Tuple[bool, bool]:
        """Update an existing buffer's properties.

        Parameters
        ----------
        name
            Buffer name to update.
        parent
            Parent instance that owns this buffer.
        **kwargs
            Properties to update (size, location, persistent, aliases).

         Returns
        -------
        bool, bool
            Whether the buffer was recognized and updated, respectively.

        Notes
        -----
        Silently ignores updates for parents with no registered group.
        """
        if parent not in self._groups:
            return False, False

        recognized, changed = self._groups[parent].update_buffer(name,
                                                                 **kwargs)
        return recognized, changed

    def clear_layout(self, parent: object) -> None:
        """Invalidate cached slices for a parent.

        Parameters
        ----------
        parent
            Parent instance whose layouts should be cleared.
        """
        if parent in self._groups:
            self._groups[parent].invalidate_layouts()

    def clear_own(self, parent: object) -> None:
        """Remove a parent's own registrations, sparing its children.

        Unlike :meth:`clear_parent` this does not cascade: children
        keep their registrations, so a parent refreshing its own
        entries (re-running its ``register_buffers``) does not wipe
        the still-valid declarations of the components it hosts.
        The parent is expected to re-record its children via
        :meth:`register_child` afterwards.

        Parameters
        ----------
        parent
            Parent instance whose own registrations are removed.
        """
        self._groups.pop(parent, None)

    def clear_parent(self, parent: object) -> None:
        """Remove a parent's buffer registrations and its children's.

        Cascades through the children recorded by
        :meth:`register_child`, so clearing a component also
        clears every component whose buffers it hosts (e.g. an
        implicit step's nonlinear solver and that solver's inner
        linear solver). Unknown parents are ignored.

        Parameters
        ----------
        parent
            Parent instance to remove.
        """
        group = self._groups.pop(parent, None)
        if group is None:
            return
        # Popping before recursing makes a registration cycle
        # terminate: a revisited parent has no group and returns.
        for child in group.children.values():
            self.clear_parent(child)

    def reset(self) -> None:
        """Clear every parent's buffer registrations from the registry."""
        allparents = list(self._groups.keys())
        for parent in allparents:
            self.clear_parent(parent)

    def update(
        self,
        parent: object,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> Set[str]:
        """Update buffer locations from keyword arguments.

        For each key of the form '[buffer_name]_location', finds the
        corresponding buffer and updates its location. Mirrors the pattern
        of CUDAFactory.update_compile_settings().

        Parameters
        ----------
        parent
            Parent instance that owns the buffers to update.
        updates_dict
            Mapping of parameter names to new values.
        silent
            Suppress errors for unrecognized parameters.
        **kwargs
            Additional parameters merged into updates_dict.

        Returns
        -------
        Set[str]
            Names of parameters that were successfully recognized.

        Raises
        ------
        ValueError
            If a location value is not 'shared' or 'local'.

        Notes
        -----
        A parameter is recognized if it matches '[buffer_name]_location'
        where buffer_name is registered for the parent. The method
        silently ignores unrecognized parameters when silent=True.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if not updates_dict:
            return set()

        if parent not in self._groups:
            return set()

        recognized = set()

        for key, value in updates_dict.items():
            if not key.endswith('_location'):
                continue

            buffer_name = key.removesuffix("_location")

            if value not in ('shared', 'local'):
                raise ValueError(
                    f"Invalid location '{value}' for buffer "
                    f"'{buffer_name}'. Must be 'shared' or 'local'."
                )

            buffer_recognized, buffer_changed = self.update_buffer(
                buffer_name, parent, location=value
            )
            if buffer_recognized:
                recognized.add(key)
            if buffer_changed:
                self.clear_layout(parent)

        return recognized

    def shared_buffer_size(self, parent: object) -> int:
        """Return total shared memory elements for a parent.

        Parameters
        ----------
        parent
            Parent instance to query.

        Returns
        -------
        int
            Total shared memory elements (excludes aliased buffers).
        """
        if parent not in self._groups:
            return 0
        return self._groups[parent].shared_buffer_size()

    def local_buffer_size(self, parent: object) -> int:
        """Return total local memory elements for a parent.

        Parameters
        ----------
        parent
            Parent instance to query.

        Returns
        -------
        int
            Total local memory elements (max(size, 1) for each).
        """
        if parent not in self._groups:
            return 0
        return self._groups[parent].local_buffer_size()

    def persistent_local_buffer_size(self, parent: object) -> int:
        """Return total persistent local elements for a parent.

        Parameters
        ----------
        parent
            Parent instance to query.

        Returns
        -------
        int
            Total persistent_local elements (excludes aliased buffers).
        """
        if parent not in self._groups:
            return 0
        return self._groups[parent].persistent_local_buffer_size()

    def relocatable_buffer_names(self, parent: object) -> Tuple[str, ...]:
        """Return buffer names registered directly on a parent.

        Excludes child roll-up entries, leaving the buffers whose
        location is set through a ``{name}_location`` setting.
        Parents with no registered group return an empty tuple.

        Parameters
        ----------
        parent
            Parent instance to query.

        Returns
        -------
        Tuple[str, ...]
            Names of the parent's own registered buffers.
        """
        if parent not in self._groups:
            return ()
        return self._groups[parent].relocatable_names()

    def nonaliased_elements(
        self,
        parent: object,
        names: Tuple[str, ...],
    ) -> int:
        """Return total parent elements the named buffers would allocate.

        Aliased buffers overlap their parent's allocation, so only
        non-aliased entries contribute. Unknown parents or names
        count zero.

        Parameters
        ----------
        parent
            Parent instance to query.
        names
            Buffer names registered on the parent.

        Returns
        -------
        int
            Total non-aliased elements across the named buffers.
        """
        if parent not in self._groups:
            return 0
        return self._groups[parent].nonaliased_elements(names)

    def declared_local_elements(self, parent: object) -> int:
        """Return the declared per-thread local footprint in elements.

        Sums plain-local buffer sizes across the parent and every
        child recorded by :meth:`register_child`, recursively, plus
        the parent's persistent-local total (children's persistent
        totals already roll up into the parent's child entries).

        Parameters
        ----------
        parent
            Parent instance to query.

        Returns
        -------
        int
            Declared local plus persistent elements per thread.
        """
        group = self._groups.get(parent)
        if group is None:
            return 0

        def plain_local(current: BufferGroup) -> int:
            total = current.local_buffer_size()
            for child in current.children.values():
                child_group = self._groups.get(child)
                if child_group is not None:
                    total += plain_local(child_group)
            return total

        return plain_local(group) + group.persistent_local_buffer_size()

    def get_allocator(
        self,
        name: str,
        parent: object,
        zero: bool = False,
    ) -> Callable:
        """Generate CUDA device function for buffer allocation.

        Parameters
        ----------
        name
            Buffer name to generate allocator for.
        parent
            Parent instance that owns the buffer.
        zero
            If True, initialize all elements to zero after allocation.

        Returns
        -------
        Callable
            CUDA device function that allocates the buffer.

        Raises
        ------
        KeyError
            If parent or buffer name not registered.
        """
        if parent not in self._groups:
            raise KeyError(
                f"Parent {parent} has no registered buffer group."
            )
        return self._groups[parent].get_allocator(name, zero)

    def register_child(
        self,
        parent: object,
        child: object,
        name: Optional[str] = None,
        aliases: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Register a child's buffer footprint with its parent.

        Registers '{name}_shared' and '{name}_persistent' entries in
        the parent's group, sized from the child's current buffer
        registrations, and records the parent-to-child ownership edge
        that :meth:`clear_parent` cascades through. Idempotent:
        repeated calls with the same name refresh the entry sizes and
        replace the recorded child.

        Parameters
        ----------
        parent
            Parent instance that will allocate memory for the child.
        child
            Child instance whose buffer requirements should be
            registered.
        name
            Optional base name for the buffer registrations. If not
            provided, uses 'child_{id(child)}' as the base name.
        aliases
            Optional shared entry the child's shared window overlaps.
            The caller owns the lifetime argument: the two children
            must never hold live data at the same time.

        Returns
        -------
        str
            Name of the child's shared-memory entry.
        str
            Name of the child's persistent-local entry.
        """
        child_shared_size = self.shared_buffer_size(child)
        child_persistent_size = self.persistent_local_buffer_size(child)

        if name is None:
            base_name = f'child_{id(child)}'
        else:
            base_name = name

        shared_name = f'{base_name}_shared'
        persistent_name = f'{base_name}_persistent'

        self.register(
            shared_name,
            parent,
            child_shared_size,
            'shared',
            aliases=aliases,
        )
        self.register(
            persistent_name,
            parent,
            child_persistent_size,
            'local',
            persistent=True,
        )
        self._groups[parent].children[base_name] = child

        return shared_name, persistent_name

    def get_child_allocators(
        self,
        parent: object,
        child: object,
        name: Optional[str] = None,
        aliases: Optional[str] = None,
    ) -> Tuple[Callable, Callable]:
        """Register child buffers and return shared and persistent allocators.

        Delegates registration and ownership recording to
        :meth:`register_child`, then returns allocators that provide
        slices into the parent's shared and persistent memory regions.

        Parameters
        ----------
        parent
            Parent instance that will allocate memory for the child.
        child
            Child instance whose buffer requirements should be registered.
        name
            Optional base name for the buffer registrations. If not provided,
            uses 'child_{id}' as the base name.
        aliases
            Optional shared entry the child's shared window overlaps;
            see :meth:`register_child`.

        Returns
        -------
        Callable
            Allocator for child's shared memory (returns slice).
        Callable
            Allocator for child's persistent memory (returns slice).
        """
        shared_name, persistent_name = self.register_child(
            parent, child, name, aliases
        )
        alloc_shared = self.get_allocator(shared_name, parent)
        alloc_persistent = self.get_allocator(persistent_name, parent)

        return alloc_shared, alloc_persistent

    def get_toplevel_allocators(
        self,
        kernel: object,
    ) -> Tuple[Callable, Callable]:
        """Create allocators for top-level kernel shared and persistent memory.

        Returns a tuple of two device functions for use in CUDA kernels:
        - A shared memory allocator that returns cuda.shared.array(0, ...)
        - A persistent local allocator that handles CUDASIM compatibility

        Parameters
        ----------
        kernel
            Kernel instance with `persistent_local_elements` and
            `precision` properties.

        Returns
        -------
        Tuple[Callable, Callable]
            (alloc_shared, alloc_persistent) device functions where:
            - alloc_shared: () -> shared memory array
            - alloc_persistent: (shared) -> persistent local array
        """

        persistent_size = max(1, kernel.persistent_local_elements)
        precision = kernel.precision
        numba_precision = from_dtype(precision)

        # no cover: start
        @cuda.jit(device=True, inline=True, **compile_kwargs)
        def alloc_shared():
            return cuda.shared.array(0,
                                     dtype=float32)

        @cuda.jit(device=True, inline=True, **compile_kwargs)
        def alloc_persistent():
                return cuda.local.array(persistent_size,
                                        dtype=numba_precision)

        # no cover: end
        return alloc_shared, alloc_persistent


# Module-level singleton instance
buffer_registry = BufferRegistry()
