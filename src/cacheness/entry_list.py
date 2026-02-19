"""
EntryList — Rich result wrapper for cache entry listings.
==========================================================

Extends ``list`` so existing code that iterates, indexes, or checks
``len()`` continues to work unchanged.  Adds convenience methods for
common post-processing tasks:

- ``.to_dataframe()`` — convert to pandas DataFrame (optional dep)
- ``.to_json()`` / ``.to_json(path)`` — JSON serialization
- ``.keys()`` — extract all cache_key values
- ``.first()`` / ``.last()`` — safe single-element access
- ``.sort_by()`` — chainable sorting by any field
- ``.filter()`` — chainable predicate filtering
- Pretty ``__repr__`` with tabular summary

Backward compatible: ``isinstance(result, list)`` is ``True``.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

__all__ = ["EntryList"]


class EntryList(list):
    """A list of cache entry dicts with convenience methods.

    Returned by :meth:`UnifiedCache.list_entries`,
    :meth:`UnifiedCache.query_meta`, and backend ``list_entries()`` calls.

    Extends ``list`` directly — all standard list operations (iteration,
    indexing, slicing, ``len()``, ``bool()``, ``in``, etc.) work exactly
    as before.  Code that treats the result as ``List[Dict]`` is
    unaffected.

    Example::

        entries = cache.list_entries()
        # Old code still works:
        for entry in entries:
            print(entry["cache_key"])

        # New convenience methods:
        df = entries.to_dataframe()
        keys = entries.keys()
        recent = entries.sort_by("created_at", reverse=True).first()
    """

    # ── Constructors ──────────────────────────────────────────────

    def __new__(cls, data: List[Dict[str, Any]] | None = None) -> EntryList:
        """Create a new EntryList from an iterable of dicts."""
        return super().__new__(cls, data or [])

    def __init__(self, data: List[Dict[str, Any]] | None = None) -> None:
        super().__init__(data or [])

    # ── Convenience accessors ─────────────────────────────────────

    def keys(self) -> list[str]:
        """Extract ``cache_key`` from each entry.

        Returns:
            List of cache key strings.
        """
        return [entry.get("cache_key", "") for entry in self]

    def first(self) -> Dict[str, Any] | None:
        """Return the first entry, or ``None`` if the list is empty."""
        return self[0] if self else None

    def last(self) -> Dict[str, Any] | None:
        """Return the last entry, or ``None`` if the list is empty."""
        return self[-1] if self else None

    # ── Chainable operations ──────────────────────────────────────

    def sort_by(
        self,
        field: str,
        *,
        reverse: bool = False,
        default: Any = None,
    ) -> EntryList:
        """Return a new :class:`EntryList` sorted by *field*.

        Args:
            field: Dict key to sort by (e.g. ``"created_at"``, ``"file_size"``).
            reverse: Sort descending when ``True``.
            default: Value to use when *field* is missing from an entry.

        Returns:
            A new sorted EntryList (does not mutate the original).
        """
        return EntryList(
            sorted(self, key=lambda e: e.get(field, default), reverse=reverse)
        )

    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> EntryList:
        """Return a new :class:`EntryList` with only entries matching *predicate*.

        Args:
            predicate: A callable that takes an entry dict and returns bool.

        Returns:
            A new filtered EntryList.

        Example::

            large = entries.filter(lambda e: e.get("file_size", 0) > 1_000_000)
        """
        return EntryList([entry for entry in self if predicate(entry)])

    # ── Serialization ─────────────────────────────────────────────

    def to_json(
        self,
        path: Optional[str] = None,
        *,
        indent: int = 2,
        default: Callable[..., Any] | None = str,
    ) -> str:
        """Serialize entries to a JSON string.

        Args:
            path: If provided, also write JSON to this file path.
            indent: JSON indentation level.
            default: Fallback serializer for non-JSON-native types
                     (defaults to ``str``).

        Returns:
            JSON string representation of entries.
        """
        text = json.dumps(list(self), indent=indent, default=default)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_dataframe(self):
        """Convert entries to a :class:`pandas.DataFrame`.

        Requires ``pandas`` to be installed.

        Returns:
            pandas.DataFrame with one row per cache entry.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            ) from None
        return pd.DataFrame(list(self))

    # ── Representation ────────────────────────────────────────────

    def __repr__(self) -> str:
        n = len(self)
        if n == 0:
            return "EntryList([])"

        # Show compact summary
        keys_preview = ", ".join(repr(e.get("cache_key", "?")) for e in self[:3])
        if n > 3:
            keys_preview += f", ... (+{n - 3} more)"
        return f"EntryList([{keys_preview}], len={n})"

    # ── Ensure chainable ops return EntryList ─────────────────────

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if isinstance(index, slice):
            return EntryList(result)
        return result

    def __add__(self, other):
        return EntryList(super().__add__(other))

    def __radd__(self, other):
        if isinstance(other, list):
            return EntryList(other + list(self))
        return NotImplemented

    def copy(self) -> EntryList:
        """Return a shallow copy as an EntryList."""
        return EntryList(list(self))
