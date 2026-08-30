"""Exact, read-only recognition for the Phase 1 compatibility evidence.

The canonical manifest repository intentionally has no fallback to these
historical layouts.  This module is an inspect-only adapter for the bounded
fixture identities retained by the compatibility window.  It never discovers
formats by traversal, opens a payload, writes a manifest, or performs a
migration.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..error_handling import (
    CacheBlobManifestMalformedError,
    CacheBlobMigrationRequiredError,
)
from ..security import verify_legacy_v038_entry


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
_LEGACY_V038_FIXTURE_KEY = bytes.fromhex(
    "00112233445566778899aabbccddeefffedcba98765432100123456789abcdef"
)

# This is an inspect-only compatibility seam, not a general legacy-store
# detector.  The hashes pin the deliberately small Phase 1 evidence corpus so
# a self-declared fixture id or a merely shape-compatible document cannot be
# reported as exact migration-ready evidence.
_EXACT_EVIDENCE_SHA256 = {
    "array-raw-v035-compress": {
        "payload.b2nd": "b7abe25fe6e6e192d0c6596fd9c30493f9f0a3e58412cc59c8913ccac37ecb46",
        "provenance.json": "5d0589d6ceae1f5d493898efa947d9a95f9b7fcf25c2632b00e71ed46657f1d9",
    },
    "array-raw-v037-compress2": {
        "payload.b2nd": "b7abe25fe6e6e192d0c6596fd9c30493f9f0a3e58412cc59c8913ccac37ecb46",
        "provenance.json": "44f84c2044677ef254f2577c7324390f66af3f0b2b4f823ce8d725bda0d14ed1",
    },
    "json-split-unsigned-v037": {
        "metadata.json": "0f1c7c772b0768c6287e63e82185cf0803f7e10e83243e685ed3bf537b9fd315",
        "payload.npz": "400fa42d786ab1e6924e051bbaa61238627909af333de87aab176c280c71fe2f",
        "provenance.json": "08091f60fcf1d57e3fc0388740c0f23e9a986e02cc82e879ffd99e2960dc89cf",
    },
    "json-split-signed-v038": {
        "metadata.json": "b8ebfb9bcc52b72a90e333dc2093273863cdad3c83fe5b3e50a9e5e032a2cb30",
        "payload.npz": "400fa42d786ab1e6924e051bbaa61238627909af333de87aab176c280c71fe2f",
        "provenance.json": "870f24839abc70f7d265cf9ea84aea48949d6403fcc3f593458f3a2bf458f172",
    },
    "sqlite-metadata-json-v039": {
        "metadata.sqlite3": "913ce4d06825544637ff5f44985518ef5cc8cb07a63505f3e76831ec71572734",
        "payload.npz": "400fa42d786ab1e6924e051bbaa61238627909af333de87aab176c280c71fe2f",
        "provenance.json": "225a337377a1d9e64249e46b205feaa42d5885a26473ce31f77cd41a4488bfb2",
    },
    "decorator-key-v0313": {
        "metadata.json": "73a9bfb8e36c1f846d6dd7d3a26e74131a4175112ea9ef202f80757adefc35fb",
        "payload.npz": "400fa42d786ab1e6924e051bbaa61238627909af333de87aab176c280c71fe2f",
        "provenance.json": "eb1319074752d15b117bc49cbe8ee2fea38253ef257dc74a25be1edd774812d6",
    },
    "json-nested-v0314": {
        "metadata.json": "3358f00e27b58a34d03e46ff33bbafe881f4e1dbf2ebf236a9a95272502888dd",
        "payload.npz": "400fa42d786ab1e6924e051bbaa61238627909af333de87aab176c280c71fe2f",
        "provenance.json": "da496e309fa3ae5d6b347459cbce5fd6d4a04cf14634ca437970f7ecc6b18474",
    },
    "sqlite-columns-v0314": {
        "metadata.sqlite3": "a236b3ad7d00f5121bc5d12a170dc0af26516b5126c7f4bd238c7c57d27f0379",
        "payload.npz": "400fa42d786ab1e6924e051bbaa61238627909af333de87aab176c280c71fe2f",
        "provenance.json": "ee95c6288298a2b55c58858d0eed39518713408580dbeee615ce39bcdb6cda75",
    },
}


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


class _OpenedLegacyRoot:
    """A retained root descriptor for one bounded legacy fixture inspection."""

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self.descriptor = descriptor

    @classmethod
    def open(cls, path: Path) -> "_OpenedLegacyRoot":
        try:
            root_stat = os.lstat(path)
            if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
                _reject("Legacy fixture root is not a non-symlink directory")
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            opened_stat = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened_stat.st_mode)
                or (opened_stat.st_dev, opened_stat.st_ino)
                != (root_stat.st_dev, root_stat.st_ino)
            ):
                os.close(descriptor)
                _reject("Legacy fixture root changed during containment setup")
            return cls(path, descriptor)
        except OSError as exc:
            raise LegacyManifestRecognitionError(
                "Legacy fixture root cannot be opened without following links"
            ) from exc

    def close(self) -> None:
        os.close(self.descriptor)

    def __enter__(self) -> "_OpenedLegacyRoot":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _open_regular(self, name: str) -> int:
        if Path(name).name != name:
            _reject("Legacy evidence name is not a direct child of its root")
        try:
            expected = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
            if not stat.S_ISREG(expected.st_mode):
                _reject(f"Legacy evidence {name!r} is not a regular file")
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self.descriptor,
            )
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
                os.close(descriptor)
                _reject(f"Legacy evidence {name!r} changed during contained open")
            return descriptor
        except OSError as exc:
            raise LegacyManifestRecognitionError(
                f"Legacy evidence {name!r} cannot be opened without following links"
            ) from exc

    def open_regular(self, name: str) -> int:
        """Open a direct child through the retained root without following links."""
        return self._open_regular(name)

    def read_bytes(self, name: str) -> bytes:
        descriptor = self._open_regular(name)
        try:
            chunks = []
            while chunk := os.read(descriptor, _MAX_LEGACY_DOCUMENT_BYTES + 1):
                chunks.append(chunk)
                if sum(map(len, chunks)) > _MAX_LEGACY_DOCUMENT_BYTES:
                    _reject("Legacy evidence exceeds the bounded compatibility limit")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def require_exact_names(self, names: tuple[str, ...]) -> None:
        try:
            found = os.listdir(self.descriptor)
        except OSError as exc:
            raise LegacyManifestRecognitionError(
                "Legacy fixture root cannot be listed through its descriptor"
            ) from exc
        if len(found) != len(names) or set(found) != set(names):
            _reject("Legacy fixture has unrecognized or missing evidence files")
        for name in names:
            descriptor = self._open_regular(name)
            os.close(descriptor)


def _read_json_object(raw: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LegacyManifestRecognitionError(
            "Legacy metadata document cannot be read as bounded JSON"
        ) from exc
    if not isinstance(parsed, dict):
        _reject("Legacy metadata document must be a JSON object")
    return parsed


def _verify_exact_evidence(
    root: _OpenedLegacyRoot, layout: _LegacyLayout
) -> dict[str, bytes]:
    """Read only named, contained evidence and match it to the pinned corpus."""
    root.require_exact_names(layout.required_files)
    expected = _EXACT_EVIDENCE_SHA256[layout.fixture_id]
    if set(expected) != set(layout.required_files):
        _reject("Legacy compatibility evidence allowlist is internally inconsistent")
    evidence = {name: root.read_bytes(name) for name in layout.required_files}
    for name, raw in evidence.items():
        if hashlib.sha256(raw).hexdigest() != expected[name]:
            _reject("Legacy evidence does not match a pinned compatibility fixture")
    return evidence


def _validate_json_layout(evidence: dict[str, bytes], layout: _LegacyLayout) -> None:
    metadata = _read_json_object(evidence["metadata.json"])
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
        entries = metadata["entries"]
        created_at = metadata["creation_times"]
        data_types = metadata["data_types"]
        file_sizes = metadata["file_sizes"]
        if not all(
            isinstance(value, dict)
            for value in (entries, created_at, data_types, file_sizes, signatures)
        ) or not (
            set(entries)
            == set(created_at)
            == set(data_types)
            == set(file_sizes)
            == set(signatures)
        ):
            _reject("Legacy signed split metadata has inconsistent entry evidence")
        for cache_key, entry in entries.items():
            if not isinstance(cache_key, str) or not isinstance(entry, dict):
                _reject("Legacy signed split metadata has an invalid entry")
            signature = signatures[cache_key]
            if (
                not isinstance(signature, str)
                or entry.get("entry_signature") != signature
                or not verify_legacy_v038_entry(
                    _LEGACY_V038_FIXTURE_KEY,
                    {
                        "cache_key": cache_key,
                        "created_at": created_at[cache_key],
                        "data_type": data_types[cache_key],
                        "file_hash": entry.get("file_hash"),
                        "file_size": file_sizes[cache_key],
                        "prefix": entry.get("prefix"),
                    },
                    signature,
                )
            ):
                _reject("Legacy signed split metadata failed historical HMAC authentication")


def _validate_sqlite_layout(root: _OpenedLegacyRoot, layout: _LegacyLayout) -> None:
    descriptor = root.open_regular("metadata.sqlite3")
    try:
        # SQLite receives the already opened no-follow descriptor, not a path
        # resolved beneath the caller-controlled root. ``immutable=1`` also
        # prevents journal recovery or creation of sidecars during inspection.
        connection = sqlite3.connect(
            f"file:/dev/fd/{descriptor}?mode=ro&immutable=1", uri=True
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
    finally:
        os.close(descriptor)
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
    with _OpenedLegacyRoot.open(candidate) as opened_root:
        provenance = _read_json_object(opened_root.read_bytes("provenance.json"))
        fixture_id = provenance.get("fixture_id")
        source_version = provenance.get("source_version")
        if not isinstance(fixture_id, str) or not isinstance(source_version, str):
            _reject("Legacy provenance lacks an exact fixture identity")
        layout = _EXACT_LAYOUTS.get(fixture_id)
        if layout is None or source_version != layout.source_version:
            _reject("Legacy provenance does not name a supported compatibility layout")
        evidence = _verify_exact_evidence(opened_root, layout)
        if layout.metadata_kind in {"split-json", "signed-split-json", "nested-json"}:
            _validate_json_layout(evidence, layout)
        elif layout.metadata_kind in {"legacy-sqlite", "current-sqlite"}:
            _validate_sqlite_layout(opened_root, layout)
        elif not evidence["payload.b2nd"]:
            _reject("Legacy raw-array payload is empty")
        return LegacyManifestIdentity(
            fixture_id=layout.fixture_id,
            source_version=layout.source_version,
            schema_identity=layout.schema_identity,
            payload_identity=layout.payload_identity,
            root=candidate,
        )
