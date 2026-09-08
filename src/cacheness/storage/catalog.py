"""Native, bounded catalog schema and portable query contracts.

The catalog describes application metadata attached to a BlobStore descriptor.
It is intentionally independent from a persistence adapter: schemas define
validation and portable query meaning, while the lifecycle authority remains
the sole visibility and transactional boundary.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from cacheness.error_handling import (
    CacheCatalogCursorError,
    CacheCatalogQueryValidationError,
    CacheCatalogStaleCursorError,
    CacheCatalogValidationError,
    CacheMigrationOrRebuildRequiredError,
)


STORE_FORMAT_VERSION: Final[int] = 2
MIN_SIGNED_64: Final[int] = -(2**63)
MAX_SIGNED_64: Final[int] = 2**63 - 1
MAX_CATALOG_FIELDS: Final[int] = 64
MAX_CATALOG_NAME_BYTES: Final[int] = 128
MAX_METADATA_DEPTH: Final[int] = 8
MAX_METADATA_ENTRIES: Final[int] = 256
MAX_METADATA_BYTES: Final[int] = 65_536
MAX_QUERY_PREDICATES: Final[int] = 32
MAX_MEMBERSHIP_VALUES: Final[int] = 128
DEFAULT_PAGE_SIZE: Final[int] = 100
MAX_PAGE_SIZE: Final[int] = 256

_SUPPORTED_FIELD_KINDS: Final[frozenset[str]] = frozenset(
    {"string", "integer", "boolean"}
)
_QUERY_OPERATORS: Final[frozenset[str]] = frozenset(
    {"eq", "lt", "lte", "gt", "gte", "in", "exists"}
)


class _Missing:
    """Internal marker separating an absent stored field from a null value."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


MISSING: Final = _Missing()


CatalogValidationError = CacheCatalogValidationError
CatalogQueryValidationError = CacheCatalogQueryValidationError
CatalogCursorError = CacheCatalogCursorError
CatalogStaleCursorError = CacheCatalogStaleCursorError
CatalogMigrationRequiredError = CacheMigrationOrRebuildRequiredError


def _canonical_json(value: Any) -> bytes:
    """Encode a validated native catalog value deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _freeze_value(value: Any) -> Any:
    """Recursively freeze metadata retained by an immutable catalog result."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _validate_metadata_value(value: Any, *, depth: int, entries: list[int]) -> Any:
    """Validate the finite JSON-like scalar and container vocabulary."""
    if depth > MAX_METADATA_DEPTH:
        raise CatalogValidationError("Catalog metadata nesting limit exceeded")
    entries[0] += 1
    if entries[0] > MAX_METADATA_ENTRIES:
        raise CatalogValidationError("Catalog metadata entry limit exceeded")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not MIN_SIGNED_64 <= value <= MAX_SIGNED_64:
            raise CatalogValidationError("Catalog integer exceeds the signed-64 range")
        return value
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_METADATA_BYTES:
            raise CatalogValidationError("Catalog string byte limit exceeded")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_METADATA_ENTRIES:
            raise CatalogValidationError("Catalog metadata entry limit exceeded")
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CatalogValidationError("Catalog metadata keys must be non-empty strings")
            if len(key.encode("utf-8")) > MAX_CATALOG_NAME_BYTES:
                raise CatalogValidationError("Catalog metadata key byte limit exceeded")
            copied[key] = _validate_metadata_value(
                item, depth=depth + 1, entries=entries
            )
        return copied
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_METADATA_ENTRIES:
            raise CatalogValidationError("Catalog metadata entry limit exceeded")
        return [
            _validate_metadata_value(item, depth=depth + 1, entries=entries)
            for item in value
        ]
    raise CatalogValidationError("Catalog metadata is not serializable")


def validate_catalog_mapping(
    values: Mapping[str, Any], *, schema: CatalogSchema | None
) -> dict[str, Any]:
    """Validate a metadata mapping with an optional native declaration."""
    if schema is not None:
        return schema.validate_mapping(values)
    if not isinstance(values, Mapping):
        raise CatalogValidationError("Catalog metadata must be a mapping")
    validated = _validate_metadata_value(values, depth=1, entries=[0])
    if not isinstance(validated, dict):  # Defensive: the top-level input was a Mapping.
        raise CatalogValidationError("Catalog metadata must be a mapping")
    if len(_canonical_json(validated)) > MAX_METADATA_BYTES:
        raise CatalogValidationError("Catalog metadata byte limit exceeded")
    return validated


def inspect_store_layout(root: Any) -> Any:
    """Classify an existing layout without creating or changing any artifact."""
    # The catalog exposes the classification boundary while manifest owns its
    # implementation. Deferring this import avoids a format-constant cycle.
    from .manifest import inspect_store_layout as inspect

    return inspect(root)


@dataclass(frozen=True, slots=True)
class CatalogField:
    """One declared scalar field in a native catalog schema."""

    name: str
    kind: str
    required: bool = False
    default: Any = MISSING
    nullable: bool = False
    queryable: bool = False
    indexed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise CatalogValidationError("Catalog field names must be non-empty strings")
        if len(self.name.encode("utf-8")) > MAX_CATALOG_NAME_BYTES:
            raise CatalogValidationError("Catalog field name byte limit exceeded")
        if self.kind not in _SUPPORTED_FIELD_KINDS:
            raise CatalogValidationError("Catalog field kind is unsupported")
        if not all(
            isinstance(value, bool)
            for value in (self.required, self.nullable, self.queryable, self.indexed)
        ):
            raise CatalogValidationError("Catalog field flags must be booleans")
        if self.default is not MISSING:
            self.validate(self.default, field_name=self.name)

    def validate(self, value: Any, *, field_name: str | None = None) -> Any:
        """Validate one exact declared scalar without coercion."""
        name = field_name or self.name
        if value is None:
            if self.nullable:
                return value
            raise CatalogValidationError(f"Catalog field {name!r} cannot be null")
        if self.kind == "string" and isinstance(value, str):
            if len(value.encode("utf-8")) > MAX_METADATA_BYTES:
                raise CatalogValidationError("Catalog string byte limit exceeded")
            return value
        if self.kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            if MIN_SIGNED_64 <= value <= MAX_SIGNED_64:
                return value
            raise CatalogValidationError("Catalog integer exceeds the signed-64 range")
        if self.kind == "boolean" and isinstance(value, bool):
            return value
        raise CatalogValidationError(
            f"Catalog field {name!r} must be an exact {self.kind} value"
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical declaration used for schema fingerprinting."""
        result: dict[str, Any] = {
            "indexed": self.indexed,
            "kind": self.kind,
            "name": self.name,
            "nullable": self.nullable,
            "queryable": self.queryable,
            "required": self.required,
        }
        if self.default is not MISSING:
            result["default"] = self.default
        return result


@dataclass(frozen=True, slots=True)
class CatalogSchema:
    """Immutable native declaration for portable catalog values and queries."""

    fields: tuple[CatalogField, ...]
    schema_id: str = "catalog"
    revision: int = 1
    _by_name: Mapping[str, CatalogField] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple) or not self.fields:
            raise CatalogValidationError("Catalog schemas require a non-empty field tuple")
        if len(self.fields) > MAX_CATALOG_FIELDS:
            raise CatalogValidationError("Catalog schema field limit exceeded")
        if not isinstance(self.schema_id, str) or not self.schema_id:
            raise CatalogValidationError("Catalog schema ID must be a non-empty string")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise CatalogValidationError("Catalog schema revision must be a positive integer")
        by_name = {item.name: item for item in self.fields}
        if len(by_name) != len(self.fields):
            raise CatalogValidationError("Catalog schemas cannot repeat field names")
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))

    @property
    def fingerprint(self) -> str:
        """Return the stable fingerprint bound into current catalog manifests."""
        record = {
            "fields": [item.to_mapping() for item in self.fields],
            "revision": self.revision,
            "schema_id": self.schema_id,
        }
        return hashlib.sha256(_canonical_json(record)).hexdigest()

    @property
    def field_map(self) -> Mapping[str, CatalogField]:
        """Expose an immutable lookup for declared field semantics."""
        return self._by_name

    def validate_mapping(
        self, values: Mapping[str, Any], *, materialize_defaults: bool = True
    ) -> dict[str, Any]:
        """Validate declared fields and preserve bounded undeclared metadata."""
        if not isinstance(values, Mapping):
            raise CatalogValidationError("Catalog metadata must be a mapping")
        candidate = _validate_metadata_value(values, depth=1, entries=[0])
        if not isinstance(candidate, dict):
            raise CatalogValidationError("Catalog metadata must be a mapping")
        validated: dict[str, Any] = {}
        for name, value in candidate.items():
            declared = self._by_name.get(name)
            validated[name] = declared.validate(value) if declared else value
        for declared in self.fields:
            if declared.name in validated:
                continue
            if materialize_defaults and declared.default is not MISSING:
                validated[declared.name] = declared.default
                continue
            if declared.required:
                raise CatalogValidationError(
                    f"Catalog required field {declared.name!r} is missing"
                )
        if len(_canonical_json(validated)) > MAX_METADATA_BYTES:
            raise CatalogValidationError("Catalog metadata byte limit exceeded")
        return validated

    def read_mapping(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Read an older stored mapping without rewriting it for additive defaults."""
        result = self.validate_mapping(values, materialize_defaults=False)
        for declared in self.fields:
            if declared.name not in result and declared.default is not MISSING:
                result[declared.name] = declared.default
        return result


@dataclass(frozen=True, slots=True)
class CatalogPredicate:
    """One finite portable predicate; schemas validate its semantic details."""

    field: str
    operator: str
    value: Any = MISSING


@dataclass(frozen=True, slots=True)
class CatalogQuery:
    """An AND-only, bounded portable query over declared catalog fields."""

    predicates: tuple[CatalogPredicate, ...] = ()
    page_size: int = DEFAULT_PAGE_SIZE
    cursor: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.predicates, tuple):
            raise CatalogQueryValidationError("Catalog predicates must be an immutable tuple")
        if len(self.predicates) > MAX_QUERY_PREDICATES:
            raise CatalogQueryValidationError("Catalog predicate limit exceeded")
        if any(not isinstance(item, CatalogPredicate) for item in self.predicates):
            raise CatalogQueryValidationError("Catalog predicates must be native values")
        if (
            not isinstance(self.page_size, int)
            or isinstance(self.page_size, bool)
            or not 1 <= self.page_size <= MAX_PAGE_SIZE
        ):
            raise CatalogQueryValidationError("Catalog query page size is out of bounds")
        if self.cursor is not None and (not isinstance(self.cursor, str) or not self.cursor):
            raise CatalogQueryValidationError("Catalog query cursor must be an opaque string")

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint excluding pagination resume state."""
        predicates = [
            {"field": item.field, "operator": item.operator, "value": item.value}
            for item in self.predicates
        ]
        return hashlib.sha256(_canonical_json({"predicates": predicates})).hexdigest()


def _validate_predicate(predicate: CatalogPredicate, schema: CatalogSchema) -> None:
    if not isinstance(predicate.field, str) or not predicate.field:
        raise CatalogQueryValidationError("Catalog predicate field must be a non-empty string")
    if predicate.operator not in _QUERY_OPERATORS:
        raise CatalogQueryValidationError("Catalog predicate operator is unsupported")
    declared = schema.field_map.get(predicate.field)
    if declared is None or not declared.queryable:
        raise CatalogQueryValidationError("Catalog predicate field is not queryable")
    if predicate.operator == "exists":
        if not isinstance(predicate.value, bool):
            raise CatalogQueryValidationError("Catalog existence predicate needs a boolean")
        return
    if predicate.value is MISSING:
        raise CatalogQueryValidationError("Catalog predicate requires a value")
    if predicate.operator == "in":
        if isinstance(predicate.value, (str, bytes)) or not isinstance(
            predicate.value, Sequence
        ):
            raise CatalogQueryValidationError("Catalog membership predicate needs a sequence")
        if not 1 <= len(predicate.value) <= MAX_MEMBERSHIP_VALUES:
            raise CatalogQueryValidationError("Catalog membership predicate is out of bounds")
        try:
            for value in predicate.value:
                declared.validate(value)
        except CatalogValidationError as exc:
            raise CatalogQueryValidationError("Catalog membership value is invalid") from exc
        return
    if predicate.operator in {"lt", "lte", "gt", "gte"}:
        if declared.kind == "boolean" or predicate.value is None:
            raise CatalogQueryValidationError("Catalog comparison predicate is invalid")
    try:
        declared.validate(predicate.value)
    except CatalogValidationError as exc:
        raise CatalogQueryValidationError("Catalog predicate value is invalid") from exc


def validate_catalog_query(
    query: CatalogQuery, *, schema: CatalogSchema, authority: Any | None = None
) -> CatalogQuery:
    """Validate every query input before an authority is eligible for dispatch."""
    del authority  # Validation deliberately performs no storage operation.
    if not isinstance(query, CatalogQuery):
        raise CatalogQueryValidationError("Catalog query must be a native value")
    if not isinstance(schema, CatalogSchema):
        raise CatalogQueryValidationError("Catalog query requires a native schema")
    for predicate in query.predicates:
        _validate_predicate(predicate, schema)
    return query


def evaluate_predicates(
    predicates: tuple[CatalogPredicate, ...],
    stored_values: Mapping[str, Any],
    *,
    schema: CatalogSchema,
) -> bool:
    """Evaluate AND-composed predicates using raw stored presence only."""
    query = CatalogQuery(predicates=predicates)
    validate_catalog_query(query, schema=schema)
    if not isinstance(stored_values, Mapping):
        raise CatalogQueryValidationError("Stored catalog values must be a mapping")
    for predicate in predicates:
        present = predicate.field in stored_values
        value = stored_values.get(predicate.field, MISSING)
        if predicate.operator == "exists":
            if present is not predicate.value:
                return False
        elif not present:
            return False
        elif predicate.operator == "eq" and value != predicate.value:
            return False
        elif predicate.operator == "lt" and not value < predicate.value:
            return False
        elif predicate.operator == "lte" and not value <= predicate.value:
            return False
        elif predicate.operator == "gt" and not value > predicate.value:
            return False
        elif predicate.operator == "gte" and not value >= predicate.value:
            return False
        elif predicate.operator == "in" and value not in predicate.value:
            return False
    return True


def _require_signing_key(signing_key: bytes) -> None:
    if not isinstance(signing_key, bytes) or len(signing_key) != 32:
        raise CatalogCursorError("Catalog cursor signing key must be exactly 32 bytes")


@dataclass(frozen=True, slots=True)
class CatalogCursor:
    """Authenticated opaque keyset resume state for one catalog snapshot."""

    @staticmethod
    def create(
        *,
        store_id: str,
        format_version: int,
        schema_id: str,
        query_fingerprint: str,
        revision: int,
        last_identity: tuple[str, str],
        signing_key: bytes,
    ) -> str:
        """Create an opaque cursor bound to every semantic snapshot dimension."""
        _require_signing_key(signing_key)
        record = {
            "format_version": format_version,
            "last_identity": list(last_identity),
            "query_fingerprint": query_fingerprint,
            "revision": revision,
            "schema_id": schema_id,
            "store_id": store_id,
        }
        _validate_cursor_record(record)
        unsigned = _canonical_json(record)
        record["signature"] = hmac.new(signing_key, unsigned, hashlib.sha256).hexdigest()
        return base64.urlsafe_b64encode(_canonical_json(record)).decode("ascii").rstrip("=")

    @staticmethod
    def parse(
        cursor: str,
        *,
        store_id: str,
        format_version: int,
        schema_id: str,
        query_fingerprint: str,
        revision: int,
        signing_key: bytes,
    ) -> tuple[str, str]:
        """Authenticate a cursor and reject every mismatched snapshot binding."""
        _require_signing_key(signing_key)
        if not isinstance(cursor, str) or not cursor:
            raise CatalogCursorError("Catalog cursor must be a non-empty string")
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise CatalogCursorError("Catalog cursor is malformed") from exc
        if not isinstance(record, dict):
            raise CatalogCursorError("Catalog cursor is malformed")
        signature = record.pop("signature", None)
        if not isinstance(signature, str):
            raise CatalogCursorError("Catalog cursor signature is malformed")
        _validate_cursor_record(record)
        expected = hmac.new(signing_key, _canonical_json(record), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise CatalogCursorError("Catalog cursor authentication failed")
        expected_context = {
            "store_id": store_id,
            "format_version": format_version,
            "schema_id": schema_id,
            "query_fingerprint": query_fingerprint,
            "revision": revision,
        }
        if any(record[name] != value for name, value in expected_context.items()):
            raise CatalogCursorError("Catalog cursor does not match this snapshot")
        identity = record["last_identity"]
        return identity[0], identity[1]


def _validate_cursor_record(record: Mapping[str, Any]) -> None:
    required = {
        "format_version",
        "last_identity",
        "query_fingerprint",
        "revision",
        "schema_id",
        "store_id",
    }
    if set(record) != required:
        raise CatalogCursorError("Catalog cursor has an invalid shape")
    if (
        not isinstance(record["format_version"], int)
        or isinstance(record["format_version"], bool)
        or not isinstance(record["revision"], int)
        or isinstance(record["revision"], bool)
        or record["revision"] < 0
    ):
        raise CatalogCursorError("Catalog cursor numeric fields are invalid")
    if any(
        not isinstance(record[name], str) or not record[name]
        for name in ("store_id", "schema_id", "query_fingerprint")
    ):
        raise CatalogCursorError("Catalog cursor string fields are invalid")
    identity = record["last_identity"]
    if (
        not isinstance(identity, list)
        or len(identity) != 2
        or any(not isinstance(value, str) or not value for value in identity)
    ):
        raise CatalogCursorError("Catalog cursor identity is invalid")


def require_current_revision(*, cursor_revision: int, authority_revision: int) -> None:
    """Raise a retryable outcome rather than return a partial stale page."""
    if cursor_revision != authority_revision:
        raise CatalogStaleCursorError("Catalog cursor revision is stale")


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One immutable catalog entry in the portable key/generation ordering."""

    key: str
    generation: str
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (self.key, self.generation)):
            raise CatalogValidationError("Catalog entry identity must be non-empty strings")
        validated = validate_catalog_mapping(self.values, schema=None)
        object.__setattr__(self, "values", _freeze_value(validated))


@dataclass(frozen=True, slots=True)
class CatalogPage:
    """A bounded, revision-complete page ordered by key then generation."""

    entries: tuple[CatalogEntry, ...]
    revision: int
    cursor: str | None
    exhausted: bool
    examined_identity: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or len(self.entries) > MAX_PAGE_SIZE:
            raise CatalogQueryValidationError("Catalog page exceeds the portable bound")
        if any(not isinstance(entry, CatalogEntry) for entry in self.entries):
            raise CatalogQueryValidationError("Catalog pages require native entries")
        identities = [(entry.key, entry.generation) for entry in self.entries]
        if identities != sorted(identities) or len(set(identities)) != len(identities):
            raise CatalogQueryValidationError("Catalog pages must be ordered by key and generation")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 0:
            raise CatalogQueryValidationError("Catalog page revision is invalid")
        if not isinstance(self.exhausted, bool):
            raise CatalogQueryValidationError("Catalog page exhaustion flag is invalid")
        if self.exhausted:
            if self.cursor is not None:
                raise CatalogQueryValidationError("Exhausted catalog pages cannot carry a cursor")
        elif not isinstance(self.cursor, str) or not self.cursor:
            raise CatalogQueryValidationError("Incomplete catalog pages need an opaque cursor")
        if self.examined_identity is not None and (
            not isinstance(self.examined_identity, tuple)
            or len(self.examined_identity) != 2
            or any(not isinstance(value, str) or not value for value in self.examined_identity)
        ):
            raise CatalogQueryValidationError("Catalog examined identity is invalid")


__all__ = [
    "CatalogCursor",
    "CatalogCursorError",
    "CatalogEntry",
    "CatalogField",
    "CatalogMigrationRequiredError",
    "CatalogPage",
    "CatalogPredicate",
    "CatalogQuery",
    "CatalogQueryValidationError",
    "CatalogSchema",
    "CatalogStaleCursorError",
    "CatalogValidationError",
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "STORE_FORMAT_VERSION",
    "evaluate_predicates",
    "inspect_store_layout",
    "require_current_revision",
    "validate_catalog_mapping",
    "validate_catalog_query",
]
