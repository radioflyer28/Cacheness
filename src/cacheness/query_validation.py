"""Validation helpers for safe SQLite cache metadata field queries."""

from collections.abc import Iterable, Mapping
import math
import re
from typing import Any

from .error_handling import CacheQueryValidationError, CacheReason


MAX_QUERY_FIELD_LENGTH = 255
MAX_QUERY_FIELD_SEGMENTS = 16
_QUERY_FIELD_SEGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _invalid_query_field(field: object) -> CacheQueryValidationError:
    """Build the stable public error used for an untrusted query field."""
    return CacheQueryValidationError(
        f"Invalid cache metadata query field: {field!r}",
        context={"field": field},
        reason=CacheReason.INVALID_QUERY_FIELD,
    )


def _invalid_query_value(field: str, value: object) -> CacheQueryValidationError:
    """Build the stable public error used for an invalid query value."""
    return CacheQueryValidationError(
        f"Invalid cache metadata query value for {field!r}: {value!r}",
        context={"field": field, "value": repr(value)},
        reason=CacheReason.INVALID_QUERY_VALUE,
    )


def validate_query_fields(fields: Mapping[str, Any] | Iterable[str]) -> tuple[str, ...]:
    """Validate and preserve ordered dotted identifier paths from query fields.

    Each field is constrained to ASCII identifiers separated by dots. This
    prevents caller-controlled JSON path syntax from reaching SQLite SQL.
    """
    field_names = tuple(fields.keys()) if isinstance(fields, Mapping) else tuple(fields)

    for field in field_names:
        if not isinstance(field, str):
            raise _invalid_query_field(field)

        segments = field.split(".")
        if (
            not field
            or len(field) > MAX_QUERY_FIELD_LENGTH
            or len(segments) > MAX_QUERY_FIELD_SEGMENTS
            or any(not _QUERY_FIELD_SEGMENT.fullmatch(segment) for segment in segments)
        ):
            raise _invalid_query_field(field)

    return field_names


def validate_query_numeric_filters(filters: Mapping[str, Any]) -> None:
    """Reject numeric thresholds that cannot have ordered comparison semantics.

    SQLite accepts permissive casts for non-finite spellings such as ``nan`` and
    ``inf``. Metadata numeric filters deliberately use threshold comparisons, so
    accepting those values would make their result backend-dependent.
    """
    for field, value in filters.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise _invalid_query_value(field, value)


def to_sqlite_json_path(field: str) -> str:
    """Return a SQLite JSON path after validating its caller-supplied field."""
    validate_query_fields((field,))
    return f"$.{field}"
