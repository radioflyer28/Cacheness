"""Canonical, backend-neutral manifest records for BlobStore payloads.

The manifest describes a handler-owned payload.  It deliberately does not add
another container around native NumPy, parquet, pickle, or dill bytes.
"""

from __future__ import annotations

import json
import re
import hmac
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cacheness.error_handling import (
    CacheManifestIntegrityError,
    CacheManifestUnsupportedVersionError,
    CacheMigrationOrRebuildRequiredError,
    CacheReason,
)

from .catalog import (
    MAX_SIGNED_64 as CATALOG_MAX_SIGNED_64,
    STORE_FORMAT_VERSION,
    validate_catalog_mapping,
)


MANIFEST_SCHEMA_VERSION = 1
PAYLOAD_FORMAT_VERSION = 1
MANIFEST_SIGNATURE_ALGORITHM = "hmac-sha256"
PAYLOAD_DIGEST_ALGORITHM = "sha256"
MAX_MANIFEST_BYTES = 1_048_576
MAX_NESTING_DEPTH = 16
MAX_COLLECTION_ITEMS = 4_096
MAX_TOTAL_NODES = 16_384
MAX_STRING_UTF8_BYTES = 262_144
MIN_SIGNED_64 = -(2**63)
MAX_SIGNED_64 = 2**63 - 1
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANONICAL_FIELDS = frozenset(
    {
        "byte_size",
        "created_at",
        "digest",
        "digest_algorithm",
        "generation",
        "handler_metadata",
        "handler_type",
        "key",
        "locator",
        "payload_format",
        "payload_format_version",
        "schema_version",
        "signature",
        "signature_algorithm",
        "state",
        "user_metadata",
    }
)


ManifestDecodeError = CacheManifestIntegrityError
UnsupportedManifestVersionError = CacheManifestUnsupportedVersionError
MigrationOrRebuildRequired = CacheMigrationOrRebuildRequiredError


# The development V1 names below remain implementation evidence until the
# phase-wide cutover rewires the lifecycle. They are not the current public
# format contract. These dimensions intentionally vary independently.
CURRENT_MANIFEST_SCHEMA_VERSION = 3
CURRENT_SQLITE_USER_VERSION = 8
CURRENT_STORE_EPOCH = 1
_CURRENT_MANIFEST_FIELDS = frozenset(
    {
        "byte_size",
        "catalog_presence",
        "catalog_schema_fingerprint",
        "catalog_schema_id",
        "catalog_schema_revision",
        "catalog_values",
        "created_at",
        "digest",
        "digest_algorithm",
        "generation",
        "handler_metadata",
        "handler_type",
        "key",
        "locator",
        "manifest_schema_version",
        "payload_format",
        "payload_format_version",
        "signature",
        "signature_algorithm",
        "sqlite_user_version",
        "state",
        "store_epoch",
        "store_format_version",
        "user_metadata",
    }
)


def _require_current_key(signing_key: bytes) -> None:
    """Keep current manifest authentication on the fixed HMAC key contract."""
    if not isinstance(signing_key, bytes) or len(signing_key) != 32:
        raise CacheManifestIntegrityError("Manifest signing key must be exactly 32 bytes")


def _validate_current_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise CacheManifestIntegrityError(f"Current manifest {name} must be a non-empty string")
    if len(value.encode("utf-8")) > MAX_STRING_UTF8_BYTES:
        raise CacheManifestIntegrityError("Manifest string byte limit exceeded")
    return value


@dataclass(frozen=True)
class StoreVersionDimensions:
    """Independent identifiers for a format-2 store and one payload contract."""

    store_epoch: int = CURRENT_STORE_EPOCH
    manifest_schema_version: int = CURRENT_MANIFEST_SCHEMA_VERSION
    sqlite_user_version: int = CURRENT_SQLITE_USER_VERSION
    payload_format_version: int = PAYLOAD_FORMAT_VERSION
    store_format_version: int = STORE_FORMAT_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("store_epoch", self.store_epoch),
            ("manifest_schema_version", self.manifest_schema_version),
            ("sqlite_user_version", self.sqlite_user_version),
            ("payload_format_version", self.payload_format_version),
            ("store_format_version", self.store_format_version),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= CATALOG_MAX_SIGNED_64
            ):
                raise CacheManifestIntegrityError(
                    f"Current manifest {name} must be a positive signed-64 integer"
                )
        if self.store_format_version != STORE_FORMAT_VERSION:
            raise MigrationOrRebuildRequired(
                "Unsupported store format requires offline migration or rebuild",
                context={"store_format_version": self.store_format_version},
            )
        if self.manifest_schema_version != CURRENT_MANIFEST_SCHEMA_VERSION:
            raise MigrationOrRebuildRequired(
                "Unsupported manifest schema requires offline migration or rebuild",
                context={"manifest_schema_version": self.manifest_schema_version},
            )
        if self.sqlite_user_version != CURRENT_SQLITE_USER_VERSION:
            raise MigrationOrRebuildRequired(
                "Unsupported SQLite catalog schema requires offline migration or rebuild",
                context={"sqlite_user_version": self.sqlite_user_version},
            )

    def to_mapping(self) -> dict[str, int]:
        """Return explicit independent dimensions for canonical serialization."""
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "payload_format_version": self.payload_format_version,
            "sqlite_user_version": self.sqlite_user_version,
            "store_epoch": self.store_epoch,
            "store_format_version": self.store_format_version,
        }


@dataclass(frozen=True)
class BlobManifest:
    """The sole format-2 descriptor for one authenticated BlobStore generation."""

    versions: StoreVersionDimensions
    key: str
    generation: str
    locator: str
    handler_type: str
    payload_format: str
    digest: str
    byte_size: int
    created_at: str
    catalog_schema_id: str
    catalog_schema_revision: int
    catalog_schema_fingerprint: str
    catalog_values: Mapping[str, Any]
    catalog_presence: tuple[str, ...]
    user_metadata: Mapping[str, Any]
    handler_metadata: Mapping[str, Any]
    state: str = "committed"
    digest_algorithm: str = PAYLOAD_DIGEST_ALGORITHM
    signature_algorithm: str = MANIFEST_SIGNATURE_ALGORITHM
    signature: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.versions, StoreVersionDimensions):
            raise CacheManifestIntegrityError("Current manifest requires version dimensions")
        for name, value in (
            ("key", self.key),
            ("generation", self.generation),
            ("locator", self.locator),
            ("handler_type", self.handler_type),
            ("payload_format", self.payload_format),
            ("created_at", self.created_at),
            ("catalog_schema_id", self.catalog_schema_id),
        ):
            _validate_current_string(name, value)
        if self.state not in {"committed", "tombstoned"}:
            raise CacheManifestIntegrityError("Current manifest lifecycle state is invalid")
        if self.digest_algorithm != PAYLOAD_DIGEST_ALGORITHM:
            raise CacheManifestIntegrityError("Current manifest digest algorithm is invalid")
        if self.signature_algorithm != MANIFEST_SIGNATURE_ALGORITHM:
            raise CacheManifestIntegrityError("Current manifest signature algorithm is invalid")
        if not _HEX_SHA256.fullmatch(self.digest):
            raise CacheManifestIntegrityError("Current manifest digest is invalid")
        if self.signature and not _HEX_SHA256.fullmatch(self.signature):
            raise CacheManifestIntegrityError("Current manifest signature is invalid")
        if (
            not isinstance(self.byte_size, int)
            or isinstance(self.byte_size, bool)
            or not 0 <= self.byte_size <= CATALOG_MAX_SIGNED_64
        ):
            raise CacheManifestIntegrityError("Current manifest byte size is invalid")
        if (
            not isinstance(self.catalog_schema_revision, int)
            or isinstance(self.catalog_schema_revision, bool)
            or not 1 <= self.catalog_schema_revision <= CATALOG_MAX_SIGNED_64
        ):
            raise CacheManifestIntegrityError("Current manifest catalog schema revision is invalid")
        if not _HEX_SHA256.fullmatch(self.catalog_schema_fingerprint):
            raise CacheManifestIntegrityError("Current manifest catalog fingerprint is invalid")
        if not isinstance(self.catalog_presence, tuple) or (
            tuple(sorted(set(self.catalog_presence))) != self.catalog_presence
        ):
            raise CacheManifestIntegrityError("Current manifest catalog presence is invalid")
        if any(not isinstance(item, str) or not item for item in self.catalog_presence):
            raise CacheManifestIntegrityError("Current manifest catalog presence is invalid")
        catalog_values = validate_catalog_mapping(self.catalog_values, schema=None)
        if set(catalog_values) != set(self.catalog_presence):
            raise CacheManifestIntegrityError(
                "Current manifest catalog values and stored presence disagree"
            )
        user_metadata = validate_catalog_mapping(self.user_metadata, schema=None)
        handler_metadata = validate_catalog_mapping(self.handler_metadata, schema=None)
        object.__setattr__(self, "catalog_values", _freeze_json_value(catalog_values))
        object.__setattr__(self, "user_metadata", _freeze_json_value(user_metadata))
        object.__setattr__(self, "handler_metadata", _freeze_json_value(handler_metadata))

    @property
    def store_format_version(self) -> int:
        """Expose the format identity without conflating the other dimensions."""
        return self.versions.store_format_version

    @property
    def payload_format_version(self) -> int:
        """Expose the handler-owned payload format independently from the store."""
        return self.versions.payload_format_version

    def to_mapping(self, *, include_signature: bool = True) -> dict[str, Any]:
        """Build the explicit wire record without serializing the dataclass shape."""
        record: dict[str, Any] = {
            "byte_size": self.byte_size,
            "catalog_presence": list(self.catalog_presence),
            "catalog_schema_fingerprint": self.catalog_schema_fingerprint,
            "catalog_schema_id": self.catalog_schema_id,
            "catalog_schema_revision": self.catalog_schema_revision,
            "catalog_values": _thaw_json_value(self.catalog_values),
            "created_at": self.created_at,
            "digest": self.digest,
            "digest_algorithm": self.digest_algorithm,
            "generation": self.generation,
            "handler_metadata": _thaw_json_value(self.handler_metadata),
            "handler_type": self.handler_type,
            "key": self.key,
            "locator": self.locator,
            "payload_format": self.payload_format,
            "signature_algorithm": self.signature_algorithm,
            "state": self.state,
            "user_metadata": _thaw_json_value(self.user_metadata),
            **self.versions.to_mapping(),
        }
        if include_signature:
            record["signature"] = self.signature
        return record

    def canonical_bytes(self, *, include_signature: bool = True) -> bytes:
        """Encode one format-2 record deterministically with explicit fields."""
        return _canonical_encode(self.to_mapping(include_signature=include_signature))

    def signing_bytes(self) -> bytes:
        """Return the complete authenticated projection, excluding signature only."""
        return self.canonical_bytes(include_signature=False)

    def with_signature(self, signature: str) -> "BlobManifest":
        """Return a distinct immutable descriptor with an HMAC authentication tag."""
        if not isinstance(signature, str) or not _HEX_SHA256.fullmatch(signature):
            raise CacheManifestIntegrityError("Current manifest signature is invalid")
        return replace(self, signature=signature)

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "BlobManifest":
        """Decode only the exact format-2 wire shape; no legacy reader exists."""
        if set(record) != _CURRENT_MANIFEST_FIELDS:
            raise MigrationOrRebuildRequired(
                "Incomplete, mixed, or foreign manifest requires offline migration or rebuild"
            )
        versions = StoreVersionDimensions(
            store_epoch=record["store_epoch"],
            manifest_schema_version=record["manifest_schema_version"],
            sqlite_user_version=record["sqlite_user_version"],
            payload_format_version=record["payload_format_version"],
            store_format_version=record["store_format_version"],
        )
        values = dict(record)
        for name in versions.to_mapping():
            del values[name]
        values["catalog_presence"] = tuple(values["catalog_presence"])
        return cls(versions=versions, **values)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "BlobManifest":
        """Decode a current manifest after bounded JSON framing validation."""
        return cls.from_mapping(decode_current_manifest_record(raw))


def sign_current_manifest(manifest: BlobManifest, signing_key: bytes) -> BlobManifest:
    """Authenticate the exact current descriptor with the repository HMAC contract."""
    _require_current_key(signing_key)
    if not isinstance(manifest, BlobManifest):
        raise CacheManifestIntegrityError("Only a current manifest can be signed")
    signature = hmac.new(signing_key, manifest.signing_bytes(), "sha256").hexdigest()
    return manifest.with_signature(signature)


def verify_current_manifest(manifest: BlobManifest, signing_key: bytes) -> None:
    """Fail closed unless the current descriptor authenticates exactly."""
    _require_current_key(signing_key)
    if not isinstance(manifest, BlobManifest) or not manifest.signature:
        raise CacheManifestIntegrityError("Current manifest signature is missing")
    expected = hmac.new(signing_key, manifest.signing_bytes(), "sha256").hexdigest()
    if not hmac.compare_digest(manifest.signature, expected):
        raise CacheManifestIntegrityError("Current manifest authentication failed")


def decode_current_manifest_record(raw: bytes) -> dict[str, Any]:
    """Read bounded format dispatch data without interpreting a legacy descriptor."""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_MANIFEST_BYTES:
        raise CacheManifestIntegrityError("Current manifest bytes are invalid")
    try:
        record = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_int=_parse_canonical_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, CacheManifestIntegrityError):
            raise
        raise CacheManifestIntegrityError("Current manifest is not valid canonical JSON") from exc
    if not isinstance(record, dict):
        raise CacheManifestIntegrityError("Current manifest must be a JSON object")
    _validate_canonical_value(record, depth=1, nodes=[0])
    return record


@dataclass(frozen=True)
class StoreLayout:
    """Read-only result for an empty or already-current store root."""

    state: str
    versions: StoreVersionDimensions | None = None


def _layout_error(root: Path, layout: str) -> None:
    raise MigrationOrRebuildRequired(
        "Unsupported store layout requires explicit offline migration or rebuild",
        context={"root": str(root), "layout": layout},
    )


def _load_layout_marker(marker: Path, root: Path) -> StoreVersionDimensions:
    try:
        raw = marker.read_bytes()
    except OSError as exc:
        raise CacheManifestIntegrityError("Store format marker cannot be read") from exc
    if not raw or len(raw) > 4096:
        _layout_error(root, "invalid-format-marker")
    try:
        record = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_int=_parse_canonical_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        _layout_error(root, "invalid-format-marker")
    if not isinstance(record, dict):
        _layout_error(root, "invalid-format-marker")
    required = {
        "store_format_version",
        "store_epoch",
        "manifest_schema_version",
        "sqlite_user_version",
    }
    if set(record) != required:
        _layout_error(root, "incomplete-or-mixed-format-marker")
    try:
        return StoreVersionDimensions(
            store_epoch=record["store_epoch"],
            manifest_schema_version=record["manifest_schema_version"],
            sqlite_user_version=record["sqlite_user_version"],
            payload_format_version=PAYLOAD_FORMAT_VERSION,
            store_format_version=record["store_format_version"],
        )
    except MigrationOrRebuildRequired:
        raise
    except CacheManifestIntegrityError:
        _layout_error(root, "foreign-format-marker")


def inspect_store_layout(root: str | Path) -> StoreLayout:
    """Classify a root using reads only, before initialization or staging occurs."""
    candidate = Path(root)
    if not candidate.exists():
        return StoreLayout("empty")
    if not candidate.is_dir():
        _layout_error(candidate, "not-a-directory")
    current_marker = candidate / ".cacheness" / "store-format.json"
    root_marker = candidate / "store-format.json"
    marker_paths = [path for path in (current_marker, root_marker) if path.is_file()]
    if len(marker_paths) > 1:
        _layout_error(candidate, "mixed-format-markers")
    legacy_paths = (
        candidate / "manifest.json",
        candidate / "metadata.json",
        candidate / "metadata.sqlite3",
        candidate / ".cacheness" / "lifecycle-authority-v1.sqlite3",
    )
    if any(path.exists() for path in legacy_paths):
        _layout_error(candidate, "development-format-1")
    if marker_paths:
        return StoreLayout("current", _load_layout_marker(marker_paths[0], candidate))
    try:
        has_artifacts = any(candidate.iterdir())
    except OSError as exc:
        raise CacheManifestIntegrityError("Store layout cannot be inspected") from exc
    if has_artifacts:
        _layout_error(candidate, "foreign-or-incomplete")
    return StoreLayout("empty")


def _freeze_json_value(value: Any) -> Any:
    """Freeze JSON-compatible metadata so a frozen manifest stays immutable."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CacheManifestIntegrityError("Manifest metadata keys must be strings")
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    """Return normal JSON-compatible containers for canonical serialization."""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object while rejecting ambiguity from duplicate JSON keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CacheManifestIntegrityError(
                f"Duplicate canonical manifest key: {key}"
            )
        result[key] = value
    return result


def _parse_canonical_int(value: str) -> int:
    """Parse an integer token without accepting out-of-range Python integers."""
    try:
        number = int(value)
    except ValueError as exc:
        raise CacheManifestIntegrityError("Manifest integer is invalid") from exc
    if not MIN_SIGNED_64 <= number <= MAX_SIGNED_64:
        raise CacheManifestIntegrityError(
            "Manifest integer exceeds the signed-64 range",
            reason=CacheReason.MANIFEST_BOUNDS,
        )
    return number


def _reject_float(_value: str) -> None:
    """Reject floats and non-finite values from canonical JSON altogether."""
    raise CacheManifestIntegrityError("Manifest values cannot be floating point")


def _reject_constant(_value: str) -> None:
    """Reject JSON's NaN and infinity extensions from canonical JSON."""
    raise CacheManifestIntegrityError("Manifest values cannot be non-finite")


def _validate_canonical_value(value: Any, *, depth: int, nodes: list[int]) -> None:
    """Enforce bounded JSON-compatible metadata before model construction."""
    if depth > MAX_NESTING_DEPTH:
        raise CacheManifestIntegrityError(
            "Manifest nesting limit exceeded", reason=CacheReason.MANIFEST_BOUNDS
        )
    nodes[0] += 1
    if nodes[0] > MAX_TOTAL_NODES:
        raise CacheManifestIntegrityError(
            "Manifest total nodes limit exceeded", reason=CacheReason.MANIFEST_BOUNDS
        )
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if not MIN_SIGNED_64 <= value <= MAX_SIGNED_64:
            raise CacheManifestIntegrityError(
                "Manifest integer exceeds the signed-64 range",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_STRING_UTF8_BYTES:
            raise CacheManifestIntegrityError(
                "Manifest string byte limit exceeded", reason=CacheReason.MANIFEST_BOUNDS
            )
        return
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise CacheManifestIntegrityError(
                "Manifest collection item limit exceeded",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        for item in value:
            _validate_canonical_value(item, depth=depth + 1, nodes=nodes)
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise CacheManifestIntegrityError(
                "Manifest collection item limit exceeded",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        for key, item in value.items():
            if not isinstance(key, str):
                raise CacheManifestIntegrityError("Manifest metadata keys must be strings")
            _validate_canonical_value(key, depth=depth + 1, nodes=nodes)
            _validate_canonical_value(item, depth=depth + 1, nodes=nodes)
        return
    raise CacheManifestIntegrityError("Manifest metadata is not JSON-compatible")


def _canonical_encode(record: Mapping[str, Any]) -> bytes:
    """Encode one bounded raw record without applying semantic manifest rules."""
    _validate_canonical_value(record, depth=1, nodes=[0])
    try:
        text = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CacheManifestIntegrityError("Manifest metadata is not JSON-compatible") from exc
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise CacheManifestIntegrityError(
            "Canonical manifest byte limit exceeded",
            reason=CacheReason.MANIFEST_BOUNDS,
        )
    return encoded


def decode_canonical_manifest_record(raw: bytes) -> dict[str, Any]:
    """Boundedly decode only the framing needed before authentication.

    The schema version is needed to dispatch the wire format.  Every other
    structural claim remains raw until its complete signed projection has been
    authenticated by the BlobStore read boundary.
    """
    if not isinstance(raw, bytes) or not raw:
        raise CacheManifestIntegrityError("Canonical manifest bytes must be non-empty")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise CacheManifestIntegrityError(
            "Canonical manifest byte limit exceeded",
            reason=CacheReason.MANIFEST_BOUNDS,
        )
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CacheManifestIntegrityError("Canonical manifest is not valid UTF-8") from exc
    try:
        record = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_int=_parse_canonical_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, CacheManifestIntegrityError):
            raise
        raise CacheManifestIntegrityError("Canonical manifest is not valid JSON") from exc
    if not isinstance(record, dict):
        raise CacheManifestIntegrityError("Canonical manifest must be a JSON object")
    _validate_canonical_value(record, depth=1, nodes=[0])
    schema_version = record.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise CacheManifestIntegrityError("Manifest schema version must be an integer")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise CacheManifestUnsupportedVersionError(
            f"Unsupported manifest schema version: {schema_version}"
        )
    return record


def canonical_signing_bytes_from_record(record: Mapping[str, Any]) -> bytes:
    """Return the fixed unsigned projection of a bounded raw manifest record."""
    unsigned = dict(record)
    unsigned.pop("signature", None)
    return _canonical_encode(unsigned)


@dataclass(frozen=True)
class BlobManifestV1:
    """The schema-1 signed description of one committed BlobStore payload."""

    schema_version: int
    key: str
    generation: str
    state: str
    locator: str
    handler_type: str
    payload_format: str
    payload_format_version: int
    digest_algorithm: str
    digest: str
    byte_size: int
    created_at: str
    handler_metadata: Mapping[str, Any]
    user_metadata: Mapping[str, Any]
    signature_algorithm: str = MANIFEST_SIGNATURE_ALGORITHM
    signature: str = ""

    def __post_init__(self) -> None:
        """Validate the immutable schema-1 shape before it reaches storage."""
        required_strings = {
            "key": self.key,
            "generation": self.generation,
            "state": self.state,
            "locator": self.locator,
            "handler_type": self.handler_type,
            "payload_format": self.payload_format,
            "digest_algorithm": self.digest_algorithm,
            "digest": self.digest,
            "created_at": self.created_at,
            "signature_algorithm": self.signature_algorithm,
        }
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise CacheManifestIntegrityError("Manifest schema version must be an integer")
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise CacheManifestUnsupportedVersionError(
                f"Unsupported manifest schema version: {self.schema_version}"
            )
        if not isinstance(self.payload_format_version, int) or isinstance(
            self.payload_format_version, bool
        ):
            raise CacheManifestIntegrityError(
                "Payload format version must be an integer"
            )
        if self.payload_format_version != PAYLOAD_FORMAT_VERSION:
            raise CacheManifestUnsupportedVersionError(
                "Unsupported payload format version: "
                f"{self.payload_format_version}"
            )
        if any(not isinstance(value, str) or not value for value in required_strings.values()):
            raise CacheManifestIntegrityError(
                "Canonical manifest requires non-empty string fields"
            )
        if any(
            len(value.encode("utf-8")) > MAX_STRING_UTF8_BYTES
            for value in required_strings.values()
        ):
            raise CacheManifestIntegrityError(
                "Manifest string byte limit exceeded",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        if self.signature_algorithm != MANIFEST_SIGNATURE_ALGORITHM:
            raise CacheManifestIntegrityError("Unsupported manifest signature algorithm")
        if self.state not in {
            "prepared",
            "committed",
            "replacing",
            "tombstoned",
            "conflicted",
        }:
            raise CacheManifestIntegrityError(
                f"Unsupported manifest lifecycle state: {self.state}"
            )
        if self.digest_algorithm != PAYLOAD_DIGEST_ALGORITHM:
            raise CacheManifestIntegrityError(
                f"Unsupported payload digest algorithm: {self.digest_algorithm}"
            )
        if not isinstance(self.byte_size, int) or isinstance(self.byte_size, bool):
            raise CacheManifestIntegrityError("Canonical manifest byte_size must be an integer")
        if not 0 <= self.byte_size <= MAX_SIGNED_64:
            raise CacheManifestIntegrityError(
                "Canonical manifest byte_size must be a non-negative signed-64 integer"
            )
        if not isinstance(self.handler_metadata, Mapping) or not isinstance(
            self.user_metadata, Mapping
        ):
            raise CacheManifestIntegrityError(
                "Canonical manifest metadata must be string-keyed maps"
            )
        if not _HEX_SHA256.fullmatch(self.digest):
            raise CacheManifestIntegrityError("Canonical manifest digest is invalid")
        if self.signature and not _HEX_SHA256.fullmatch(self.signature):
            raise CacheManifestIntegrityError("Canonical manifest signature is invalid")
        _validate_canonical_value(self.handler_metadata, depth=2, nodes=[0])
        _validate_canonical_value(self.user_metadata, depth=2, nodes=[0])
        object.__setattr__(
            self, "handler_metadata", _freeze_json_value(self.handler_metadata)
        )
        object.__setattr__(self, "user_metadata", _freeze_json_value(self.user_metadata))

    def to_mapping(self, *, include_signature: bool = True) -> dict[str, Any]:
        """Return a normal mapping suitable for canonical JSON serialization."""
        record = {
            "byte_size": self.byte_size,
            "created_at": self.created_at,
            "digest": self.digest,
            "digest_algorithm": self.digest_algorithm,
            "generation": self.generation,
            "handler_metadata": _thaw_json_value(self.handler_metadata),
            "handler_type": self.handler_type,
            "key": self.key,
            "locator": self.locator,
            "payload_format": self.payload_format,
            "payload_format_version": self.payload_format_version,
            "schema_version": self.schema_version,
            "signature_algorithm": self.signature_algorithm,
            "state": self.state,
            "user_metadata": _thaw_json_value(self.user_metadata),
        }
        if include_signature:
            record["signature"] = self.signature
        return record

    def canonical_bytes(self, *, include_signature: bool = True) -> bytes:
        """Encode this exact field set into deterministic UTF-8 JSON bytes."""
        record = self.to_mapping(include_signature=include_signature)
        # Outgoing records must obey the same aggregate structural limits as
        # untrusted records on the read path.  Validating the two metadata
        # maps independently is insufficient because their combined node
        # count, plus the enclosing schema fields, can exceed the wire limit.
        return _canonical_encode(record)

    def signing_bytes(self) -> bytes:
        """Return the complete signed projection, excluding only the signature."""
        return self.canonical_bytes(include_signature=False)

    def with_signature(self, signature: str) -> "BlobManifestV1":
        """Return a new manifest with an authenticated HMAC value."""
        if not isinstance(signature, str) or not signature:
            raise CacheManifestIntegrityError("Canonical manifest signature must be non-empty")
        return replace(self, signature=signature)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "BlobManifestV1":
        """Decode a schema-1 record without interpreting any payload bytes."""
        return cls.from_mapping(decode_canonical_manifest_record(raw))

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "BlobManifestV1":
        """Validate a previously bounded and, when required, authenticated map."""
        if set(record) != _CANONICAL_FIELDS:
            raise CacheManifestIntegrityError(
                "Canonical manifest has an unknown or missing field"
            )
        return cls(**record)


__all__ = [
    "BlobManifest",
    "CURRENT_MANIFEST_SCHEMA_VERSION",
    "CURRENT_SQLITE_USER_VERSION",
    "CURRENT_STORE_EPOCH",
    "MigrationOrRebuildRequired",
    "STORE_FORMAT_VERSION",
    "StoreLayout",
    "StoreVersionDimensions",
    "decode_current_manifest_record",
    "inspect_store_layout",
    "sign_current_manifest",
    "verify_current_manifest",
    # Historical implementation evidence remains internal until the planned
    # phase-wide API cutover rewires every lifecycle consumer.
    "BlobManifestV1",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_SIGNATURE_ALGORITHM",
    "PAYLOAD_DIGEST_ALGORITHM",
    "PAYLOAD_FORMAT_VERSION",
    "MAX_MANIFEST_BYTES",
    "MAX_NESTING_DEPTH",
    "MAX_COLLECTION_ITEMS",
    "MAX_TOTAL_NODES",
    "MAX_STRING_UTF8_BYTES",
    "canonical_signing_bytes_from_record",
    "decode_canonical_manifest_record",
    "ManifestDecodeError",
    "UnsupportedManifestVersionError",
]
