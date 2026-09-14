"""Frozen attrs settings base with a replacement-deriving ``update``."""

from typing import Any, Set, Tuple

from attrs import Attribute, evolve, fields, frozen
from numpy import array_equal, asarray, ndarray


def values_differ(fld: Attribute, old: Any, new: Any) -> bool:
    """Compare by identity (device fns), value (arrays), else !=."""
    if fld.metadata.get("device_function"):
        return old is not new
    if isinstance(old, ndarray) or isinstance(new, ndarray):
        return not array_equal(asarray(old), asarray(new))
    return bool(old != new)


@frozen
class FrozenSettings:
    """Frozen attrs settings; :meth:`update` derives a replacement."""

    def update(
        self, updates_dict: dict = None, **kwargs
    ) -> Tuple["FrozenSettings", Set[str], Set[str]]:
        """Derive a replacement snapshot with new field values.

        Parameters
        ----------
        updates_dict
            Init names to new values; unknown keys are ignored.
        **kwargs
            Additional settings to update.

        Returns
        -------
        tuple[FrozenSettings, set[str], set[str]]
            Replacement (``self`` when unchanged), recognised names,
            and the names whose converted value changed.
        """
        updates = {**(updates_dict or {}), **kwargs}
        by_handle = {
            (fld.alias or fld.name): fld
            for fld in fields(type(self))
            if fld.init
        }
        given = {
            key: by_handle[key] for key in updates if key in by_handle
        }
        if not given:
            return self, set(), set()
        candidate = evolve(self, **{key: updates[key] for key in given})
        changed = {
            key
            for key, fld in given.items()
            if values_differ(
                fld, getattr(self, fld.name), getattr(candidate, fld.name)
            )
        }
        if not changed:
            return self, set(given), set()
        return candidate, set(given), changed
