"""Exact, read-only recognition for the Phase 1 compatibility evidence.

The canonical manifest repository intentionally has no fallback to these
historical layouts.  This module is an inspect-only adapter for the bounded
fixture identities retained by the compatibility window.  It never discovers
formats by traversal, opens a payload, writes a manifest, or performs a
migration.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..error_handling import (
    CacheBlobManifestMalformedError,
    CacheBlobMigrationRequiredError,
)


_MAX_LEGACY_DOCUMENT_BYTES = 64 * 1024
_SPLIT_JSON_KEYS = frozenset(
    {
        "entries",
        "access_times",
        "creation_times",
        "file_sizes",
        "data_types",
        "cache_key_params",
        "cache_hits",
        "cache_misses",
    }
)
_SIGNED_SPLIT_JSON_KEYS = _SPLIT_JSON_KEYS | {"entry_signature"}
_NESTED_JSON_KEYS = frozenset({"entries", "cache_hits", "cache_misses"})
_LEGACY_SQLITE_COLUMNS = (
    "cache_key",
    "description",
    "data_type",
    "prefix",
    "created_at",
    "accessed_at",
    "file_size",
    "file_hash",
    "entry_signature",
    "cache_key_params",
    "metadata_json",
)
_CURRENT_SQLITE_COLUMNS = (
    "cache_key",
    "description",
    "data_type",
    "prefix",
    "created_at",
    "accessed_at",
    "file_size",
    "file_hash",
    "entry_signature",
    "object_type",
    "storage_format",
    "serializer",
    "compression_codec",
    "actual_path",
    "cache_key_params",
)


@dataclass(frozen=True)
class _LegacyLayout:
    fixture_id: str
    source_version: str
    schema_identity: str
    payload_identity: str
    required_files: tuple[str, ...]
    metadata_kind: str


_EXACT_LAYOUTS = {
    layout.fixture_id: layout
    for layout in (
        _LegacyLayout(
            "array-raw-v035-compress",
            "0.3.5",
            "legacy-raw-array-v035",
            "blosc2-compress",
            ("payload.b2nd", "provenance.json"),
            "raw-array",
        ),
        _LegacyLayout(
            "array-raw-v037-compress2",
            "0.3.7",
            "legacy-raw-array-v037",
            "blosc2-compress2",
            ("payload.b2nd", "provenance.json"),
            "raw-array",
        ),
        _LegacyLayout(
            "json-split-unsigned-v037",
            "0.3.7",
            "legacy-json-split-v037",
            "npz-v1",
            ("metadata.json", "payload.npz", "provenance.json"),
            "split-json",
        ),
        _LegacyLayout(
            "json-split-signed-v038",
            "0.3.8",
            "legacy-json-split-v038-signed",
            "npz-v1",
            ("metadata.json", "payload.npz", "provenance.json"),
            "signed-split-json",
        ),
        _LegacyLayout(
            "sqlite-metadata-json-v039",
            "0.3.9",
            "legacy-sqlite-metadata-json-v039",
            "npz-v1",
            ("metadata.sqlite3", "payload.npz", "provenance.json"),
            "legacy-sqlite",
        ),
        _LegacyLayout(
            "decorator-key-v0313",
            "0.3.13",
            "legacy-decorator-key-v0313",
            "npz-v1",
            ("metadata.json", "payload.npz", "provenance.json"),
            "nested-json",
        ),
        _LegacyLayout(
            "json-nested-v0314",
            "0.3.14",
            "legacy-json-nested-v0314",
            "npz-v1",
            ("metadata.json", "payload.npz", "provenance.json"),
            "nested-json",
        ),
        _LegacyLayout(
            "sqlite-columns-v0314",
            "0.3.14",
            "legacy-sqlite-columns-v0314",
            "npz-v1",
            ("metadata.sqlite3", "payload.npz", "provenance.json"),
            "current-sqlite",
        ),
    )
}


class LegacyManifestRecognitionError(CacheBlobManifestMalformedError):
    """Raised when evidence is not one exact supported legacy layout."""


@dataclass(frozen=True)
class LegacyManifestIdentity:
    """An in-memory identity for one supported historical layout."""

    fixture_id: str
    source_version: str
    schema_identity: str
    payload_identity: str
    root: Path
    read_only: bool = True

    def require_explicit_migration(self) -> None:
        """Report that conversion is owned by the Phase 7 migration workflow."""
        raise CacheBlobMigrationRequiredError(
            "Legacy BlobStore evidence requires explicit migration",
            context={
                "fixture_id": self.fixture_id,
                "legacy_schema": self.schema_identity,
                "legacy_payload": self.payload_identity,
            },
        )


def _reject(message: str) -> None:
    raise LegacyManifestRecognitionError(message)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_LEGACY_DOCUMENT_BYTES:
            _reject("Legacy metadata document exceeds the bounded compatibility limit")
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LegacyManifestRecognitionError(
            "Legacy metadata document cannot be read as bounded JSON"
        ) from exc
    if not isinstance(parsed, dict):
        _reject("Legacy metadata document must be a JSON object")
    return parsed


def _require_exact_files(root: Path, layout: _LegacyLayout) -> None:
    """Check only known names; compatibility recognition never traverses a tree."""
    for name in layout.required_files:
        path = root / name
        if not path.is_file():
            _reject(f"Legacy layout {layout.fixture_id!r} is missing {name!r}")


def _validate_json_layout(root: Path, layout: _LegacyLayout) -> None:
    metadata = _read_json_object(root / "metadata.json")
    keys = frozenset(metadata)
    if layout.metadata_kind == "split-json":
        expected = _SPLIT_JSON_KEYS
    elif layout.metadata_kind == "signed-split-json":
        expected = _SIGNED_SPLIT_JSON_KEYS
    else:
        expected = _NESTED_JSON_KEYS
    if keys != expected:
        _reject("Legacy JSON metadata keys do not match one exact supported layout")
    if not isinstance(metadata.get("entries"), dict):
        _reject("Legacy JSON metadata entries must be an object")
    if layout.metadata_kind == "signed-split-json":
        signatures = metadata.get("entry_signature")
        if not isinstance(signatures, dict) or not signatures:
            _reject("Legacy signed split metadata has no exact signature map")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in signatures.values()
        ):
            _reject("Legacy signed split metadata contains a malformed signature")


def _validate_sqlite_layout(root: Path, layout: _LegacyLayout) -> None:
    database = root / "metadata.sqlite3"
    try:
        # ``immutable=1`` prevents SQLite from creating WAL sidecars while
        # inspecting immutable compatibility evidence.  The adapter accepts
        # only copied/read-only historical stores, so it must never attempt
        # journal recovery or any write-adjacent bookkeeping.
        connection = sqlite3.connect(
            f"{database.as_uri()}?mode=ro&immutable=1", uri=True
        )
        try:
            before = connection.execute("PRAGMA data_version").fetchone()[0]
            columns = tuple(
                row[1]
                for row in connection.execute("PRAGMA table_info(cache_entries)")
            )
            after = connection.execute("PRAGMA data_version").fetchone()[0]
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise LegacyManifestRecognitionError(
            "Legacy SQLite metadata cannot be inspected read-only"
        ) from exc
    expected = (
        _LEGACY_SQLITE_COLUMNS
        if layout.metadata_kind == "legacy-sqlite"
        else _CURRENT_SQLITE_COLUMNS
    )
    if before != after or columns != expected:
        _reject("Legacy SQLite metadata does not match one exact supported layout")


def recognize_legacy_fixture_tree(root: str | Path) -> LegacyManifestIdentity:
    """Recognize one exact Phase 1 compatibility fixture without persistence.

    This function is deliberately not a generic format probe.  The identity is
    selected only from a bounded provenance document and exact historical
    metadata shape.  Any mismatch is a typed outcome and never falls through
    to canonical decoding or another compatibility reader.
    """
    candidate = Path(root)
    if not candidate.is_dir():
        _reject("Legacy fixture root is not a directory")
    provenance = _read_json_object(candidate / "provenance.json")
    fixture_id = provenance.get("fixture_id")
    source_version = provenance.get("source_version")
    if not isinstance(fixture_id, str) or not isinstance(source_version, str):
        _reject("Legacy provenance lacks an exact fixture identity")
    layout = _EXACT_LAYOUTS.get(fixture_id)
    if layout is None or source_version != layout.source_version:
        _reject("Legacy provenance does not name a supported compatibility layout")
    _require_exact_files(candidate, layout)
    if layout.metadata_kind in {"split-json", "signed-split-json", "nested-json"}:
        _validate_json_layout(candidate, layout)
    elif layout.metadata_kind in {"legacy-sqlite", "current-sqlite"}:
        _validate_sqlite_layout(candidate, layout)
    elif (candidate / "payload.b2nd").stat().st_size == 0:
        _reject("Legacy raw-array payload is empty")
    return LegacyManifestIdentity(
        fixture_id=layout.fixture_id,
        source_version=layout.source_version,
        schema_identity=layout.schema_identity,
        payload_identity=layout.payload_identity,
        root=candidate,
    )
