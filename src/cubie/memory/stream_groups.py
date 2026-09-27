"""Stream group management for coordinating CUDA work queues.

This module groups host-side objects and identifiers under shared CUDA
streams so that related kernels, transfers, and memory operations
execute together. The default group always exists and receives a fresh
stream after CUDA context resets.

Published Classes
-----------------
:class:`StreamGroups`
    Container for organising instances into groups with shared streams.

    >>> groups = StreamGroups()
    >>> groups.add_instance(42, "default")
    >>> groups.get_group(42)
    'default'

See Also
--------
:class:`~cubie.memory.mem_manager.MemoryManager`
    Owns a :class:`StreamGroups` instance and delegates stream
    operations to it.
"""

from typing import Any, Optional, Union
from cubie._cudasim_extensions import cuda, Stream
import attrs
import attrs.validators as val


@attrs.define
class StreamGroups:
    """Container for organizing instances into groups with shared streams.

    Parameters
    ----------
    groups
        Dictionary mapping group names to lists of instance identifiers. When
        omitted, an empty mapping is created and populated with the "default"
        group.
    streams
        Dictionary mapping group names to CUDA streams. When omitted, each
        group, including "default", receives a dedicated stream from
        :func:`numba_cuda_mlir.cuda.stream` on first use. No group is
        ever backed by the device-wide default stream, so work in one
        process never orders against the CUDA null stream.

    Attributes
    ----------
    groups
        Dictionary mapping group names to lists of instance identifiers.
    streams
        Dictionary mapping group names to CUDA streams.

    Notes
    -----
    Each group has an associated CUDA stream that all instances in the group
    share for coordinated operations. The "default" group is created
    automatically.
    """

    groups: Optional[dict[str, list[int]]] = attrs.field(
        default=attrs.Factory(dict),
        validator=val.optional(val.instance_of(dict)),
    )
    streams: dict[str, Union[Stream, int]] = attrs.field(
        default=attrs.Factory(dict), validator=val.instance_of(dict)
    )

    def __attrs_post_init__(self) -> None:
        """Initialize default group and stream if not provided."""
        if self.groups is None:
            self.groups = {"default": []}

    def add_instance(self, instance: Any, group: str) -> None:
        """Add an instance to a stream group.

        Parameters
        ----------
        instance
            Host object or integer identifier to register with a
            stream group.
        group
            Name of the destination group.

        Raises
        ------
        ValueError
            If the instance is already in a stream group.

        Notes
        -----
        If the group does not exist, it is created with a new CUDA
        stream.
        """
        if isinstance(instance, int):
            instance_id = instance
        else:
            instance_id = id(instance)
        if any(instance_id in group for group in self.groups.values()):
            raise ValueError(
                "Instance already in a stream group. Call change_group instead"
            )
        self.get_group_stream(group)
        self.groups[group].append(instance_id)

    def get_group(self, instance: Any) -> str:
        """
        Get the stream group associated with an instance.

        Parameters
        ----------
        instance
            Host object or integer identifier whose group is requested.

        Returns
        -------
        str
            Name of the group containing the instance.

        Raises
        ------
        ValueError
            If the instance is not in any stream groups.
        """
        if isinstance(instance, int):
            instance_id = instance
        else:
            instance_id = id(instance)
        try:
            return [
                key
                for key, value in self.groups.items()
                if instance_id in value
            ][0]
        except IndexError:
            raise ValueError("Instance not in any stream groups")

    def get_group_stream(self, group: str = "default") -> Union[Stream, int]:
        """
        Get the dedicated stream for a named group, creating it if needed.

        Parameters
        ----------
        group
            Name of the stream group.

        Returns
        -------
        Stream
            The group's dedicated CUDA stream. A missing group (or a
            group created without a stream) receives a fresh stream
            from :func:`numba_cuda_mlir.cuda.stream`, never the device-wide
            default stream.
        """
        if group not in self.groups:
            self.groups[group] = []
        if group not in self.streams:
            self.streams[group] = cuda.stream()
        return self.streams[group]

    def get_stream(self, instance: Any) -> Union[Stream, int]:
        """
        Get the CUDA stream associated with an instance.

        Parameters
        ----------
        instance
            Host object or integer identifier whose stream is requested.

        Returns
        -------
        Stream or int
            CUDA stream associated with the instance's group.
        """
        return self.streams[self.get_group(instance)]

    def get_instances_in_group(self, group: str) -> list[int]:
        """
        Get all instances in a stream group.

        Parameters
        ----------
        group
            Name of the group to inspect.

        Returns
        -------
        list of int
            List of instance identifiers associated with the group, or an empty
            list when the group has not been created.
        """
        if group not in self.groups:
            return []

        return self.groups[group]

    def remove_instance(self, instance: Any) -> None:
        """Remove an instance from its stream group, if it has one.

        Parameters
        ----------
        instance
            Host object or integer identifier to remove.

        Notes
        -----
        Removing an instance that is in no group is a no-op; the
        group and its stream are kept for the remaining members.
        """
        if isinstance(instance, int):
            instance_id = instance
        else:
            instance_id = id(instance)
        for members in self.groups.values():
            if instance_id in members:
                members.remove(instance_id)
                return

    def change_group(self, instance: Any, new_group: str) -> None:
        """Move an instance to another stream group.

        Parameters
        ----------
        instance
            Host object or integer identifier to move.
        new_group
            Name of the destination group.

        Notes
        -----
        If the new group does not exist, it is created with a new
        CUDA stream.
        """
        if isinstance(instance, int):
            instance_id = instance
        else:
            instance_id = id(instance)

        # Remove from current group
        current_group = self.get_group(instance)
        self.groups[current_group].remove(instance_id)

        # Add to new group
        self.get_group_stream(new_group)
        self.groups[new_group].append(instance_id)

    def reinit_streams(self) -> None:
        """Reinitialize all streams after a context reset.

        Notes
        -----
        Called after CUDA context reset to create fresh streams for
        all groups.
        """
        for group in self.streams:
            self.streams[group] = cuda.stream()
