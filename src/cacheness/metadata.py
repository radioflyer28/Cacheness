"""Derived JSON catalog projection state.

Canonical catalog membership remains in the selected ``LifecycleAuthority``.
``JsonProjection`` only persists a replay-safe local view that a
``ProjectionController`` can refresh from authenticated catalog pages.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .storage.catalog import CatalogEntry
from .storage.projections import ProjectionBatch, ProjectionCheckpoint


class JsonProjectionError(ValueError):
    """Raised when derived JSON projection state is malformed or incompatible."""


class JsonProjection:
    """A local, derived-only ``ProjectionSink`` backed by one JSON document.

    A batch is durably recorded as pending with its derived entry changes before
    the controller advances its checkpoint. Replaying that same pending batch is
    therefore a no-op after an interruption; a different batch is rejected
    until the matching checkpoint is saved. This object intentionally exposes
    no canonical authority, read, cleanup, repair, or lifecycle operations.
    """

    projection_name = "json"
    topology_capabilities = {
        "projection_refresh": True,
        "projection_rebuild": False,
        "online_rebuild": False,
        "offline_rebuild": False,
    }
    _FORMAT_VERSION = 1

    def __init__(self, metadata_file: str | Path, **options: Any) -> None:
        if not isinstance(metadata_file, (str, Path)):
            raise TypeError("JSON projection metadata_file must be a path")
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"Unsupported JSON projection options: {unknown}")
        self.metadata_file = Path(metadata_file)

    def apply_projection_batch(self, batch: object) -> None:
        """Persist one bounded derived batch without advancing the checkpoint."""
        if not isinstance(batch, ProjectionBatch):
            raise JsonProjectionError("JSON projection requires a ProjectionBatch")
        document = self._load_document()
        source = _source_from_checkpoint(batch.checkpoint)
        if document is None:
            document = _empty_document(source)
        elif document["source"] != source:
            if document["pending_batch"] is not None:
                raise JsonProjectionError(
                    "JSON projection cannot replace a snapshot with pending work"
                )
            # A new canonical revision starts a new derived snapshot. Its old
            # entries are derived state, never input to canonical decisions.
            document = _empty_document(source)

        pending = document["pending_batch"]
        if pending is not None:
            if pending["batch_id"] == batch.batch_id:
                return
            raise JsonProjectionError(
                "JSON projection has unresolved pending work for another batch"
            )

        records = _entry_records(batch.entries)
        entries = {
            (record["key"], record["generation"]): record
            for record in document["entries"]
        }
        entries.update(
            {(record["key"], record["generation"]): record for record in records}
        )
        next_checkpoint = _checkpoint_record(
            ProjectionCheckpoint(
                source_store_id=batch.checkpoint.source_store_id,
                store_epoch=batch.checkpoint.store_epoch,
                schema_id=batch.checkpoint.schema_id,
                schema_fingerprint=batch.checkpoint.schema_fingerprint,
                query_fingerprint=batch.checkpoint.query_fingerprint,
                revision=batch.checkpoint.revision,
                cursor=batch.next_cursor,
                complete=batch.exhausted,
            )
        )
        document["entries"] = [entries[key] for key in sorted(entries)]
        document["pending_batch"] = {
            "batch_id": batch.batch_id,
            "next_checkpoint": next_checkpoint,
        }
        self._write_document(document)

    def save_projection_checkpoint(self, checkpoint: object) -> None:
        """Advance only the checkpoint paired with the pending applied batch."""
        if not isinstance(checkpoint, ProjectionCheckpoint):
            raise JsonProjectionError("JSON projection requires a ProjectionCheckpoint")
        document = self._require_document()
        source = _source_from_checkpoint(checkpoint)
        if document["source"] != source:
            raise JsonProjectionError("JSON projection checkpoint belongs to another snapshot")
        pending = document["pending_batch"]
        if pending is None:
            raise JsonProjectionError("JSON projection has no applied batch to checkpoint")
        checkpoint_record = _checkpoint_record(checkpoint)
        if pending["next_checkpoint"] != checkpoint_record:
            raise JsonProjectionError("JSON projection checkpoint does not match pending work")
        document["checkpoint"] = checkpoint_record
        document["pending_batch"] = None
        self._write_document(document)

    def load_projection_checkpoint(self) -> ProjectionCheckpoint | None:
        """Load the last validated checkpoint without inferring one from entries."""
        document = self._load_document()
        if document is None or document["checkpoint"] is None:
            return None
        return _checkpoint_from_record(document["checkpoint"])

    def projected_entries(self) -> tuple[CatalogEntry, ...]:
        """Return validated derived records for diagnostics and projection tests."""
        document = self._load_document()
        if document is None:
            return ()
        return tuple(_entry_from_record(record) for record in document["entries"])

    def close(self) -> None:
        """Release no resources; document publication opens files per operation."""

    def _require_document(self) -> dict[str, Any]:
        document = self._load_document()
        if document is None:
            raise JsonProjectionError("JSON projection has no derived state")
        return document

    def _load_document(self) -> dict[str, Any] | None:
        if not self.metadata_file.exists():
            return None
        try:
            raw = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise JsonProjectionError("JSON projection document cannot be parsed") from error
        return _validate_document(raw)

    def _write_document(self, document: Mapping[str, Any]) -> None:
        validated = _validate_document(document)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.metadata_file.parent,
                prefix=f".{self.metadata_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(
                    validated,
                    handle,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.metadata_file)
        except (OSError, TypeError, ValueError) as error:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            raise JsonProjectionError("JSON projection document cannot be published") from error


def _empty_document(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": JsonProjection._FORMAT_VERSION,
        "source": source,
        "entries": [],
        "pending_batch": None,
        "checkpoint": None,
    }


def _validate_document(raw: object) -> dict[str, Any]:
    expected_keys = {
        "format_version",
        "source",
        "entries",
        "pending_batch",
        "checkpoint",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise JsonProjectionError("JSON projection document has an incompatible shape")
    if raw["format_version"] != JsonProjection._FORMAT_VERSION:
        raise JsonProjectionError("JSON projection document version is unsupported")
    source = _validate_source(raw["source"])
    entries_raw = raw["entries"]
    if not isinstance(entries_raw, list):
        raise JsonProjectionError("JSON projection entries must be a list")
    entries = [_entry_record(_entry_from_record(record)) for record in entries_raw]
    identities = [(record["key"], record["generation"]) for record in entries]
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise JsonProjectionError("JSON projection entries are not uniquely ordered")
    checkpoint_raw = raw["checkpoint"]
    checkpoint = None
    if checkpoint_raw is not None:
        checkpoint = _checkpoint_record(_checkpoint_from_record(checkpoint_raw))
        if _source_from_checkpoint(_checkpoint_from_record(checkpoint)) != source:
            raise JsonProjectionError("JSON projection checkpoint source does not match document")
    pending_raw = raw["pending_batch"]
    pending: dict[str, Any] | None
    if pending_raw is None:
        pending = None
    elif isinstance(pending_raw, Mapping) and set(pending_raw) == {
        "batch_id",
        "next_checkpoint",
    }:
        batch_id = pending_raw["batch_id"]
        if not isinstance(batch_id, str) or len(batch_id) != 64:
            raise JsonProjectionError("JSON projection pending batch identity is invalid")
        next_checkpoint = _checkpoint_record(
            _checkpoint_from_record(pending_raw["next_checkpoint"])
        )
        if _source_from_checkpoint(_checkpoint_from_record(next_checkpoint)) != source:
            raise JsonProjectionError("JSON projection pending source does not match document")
        pending = {"batch_id": batch_id, "next_checkpoint": next_checkpoint}
    else:
        raise JsonProjectionError("JSON projection pending batch has an incompatible shape")
    return {
        "format_version": JsonProjection._FORMAT_VERSION,
        "source": source,
        "entries": entries,
        "pending_batch": pending,
        "checkpoint": checkpoint,
    }


def _validate_source(raw: object) -> dict[str, Any]:
    expected_keys = {
        "source_store_id",
        "store_epoch",
        "schema_id",
        "schema_fingerprint",
        "query_fingerprint",
        "revision",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise JsonProjectionError("JSON projection source has an incompatible shape")
    try:
        checkpoint = ProjectionCheckpoint(cursor=None, complete=False, **dict(raw))
    except (TypeError, ValueError) as error:
        raise JsonProjectionError("JSON projection source is invalid") from error
    return _source_from_checkpoint(checkpoint)


def _source_from_checkpoint(checkpoint: ProjectionCheckpoint) -> dict[str, Any]:
    return {
        "source_store_id": checkpoint.source_store_id,
        "store_epoch": checkpoint.store_epoch,
        "schema_id": checkpoint.schema_id,
        "schema_fingerprint": checkpoint.schema_fingerprint,
        "query_fingerprint": checkpoint.query_fingerprint,
        "revision": checkpoint.revision,
    }


def _checkpoint_record(checkpoint: ProjectionCheckpoint) -> dict[str, Any]:
    return {
        **_source_from_checkpoint(checkpoint),
        "cursor": checkpoint.cursor,
        "complete": checkpoint.complete,
    }


def _checkpoint_from_record(raw: object) -> ProjectionCheckpoint:
    expected_keys = {
        "source_store_id",
        "store_epoch",
        "schema_id",
        "schema_fingerprint",
        "query_fingerprint",
        "revision",
        "cursor",
        "complete",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise JsonProjectionError("JSON projection checkpoint has an incompatible shape")
    try:
        return ProjectionCheckpoint(**dict(raw))
    except (TypeError, ValueError) as error:
        raise JsonProjectionError("JSON projection checkpoint is invalid") from error


def _entry_records(entries: tuple[object, ...]) -> list[dict[str, Any]]:
    return [_entry_record(_require_catalog_entry(entry)) for entry in entries]


def _require_catalog_entry(entry: object) -> CatalogEntry:
    if not isinstance(entry, CatalogEntry):
        raise JsonProjectionError("JSON projection batches require CatalogEntry records")
    return entry


def _entry_record(entry: CatalogEntry) -> dict[str, Any]:
    record = {
        "key": entry.key,
        "generation": entry.generation,
        "values": _json_safe(entry.values),
    }
    try:
        json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise JsonProjectionError("JSON projection values must be JSON-safe") from error
    return record


def _entry_from_record(raw: object) -> CatalogEntry:
    if not isinstance(raw, Mapping) or set(raw) != {"key", "generation", "values"}:
        raise JsonProjectionError("JSON projection entry has an incompatible shape")
    if not isinstance(raw["values"], Mapping):
        raise JsonProjectionError("JSON projection entry values must be a mapping")
    if not isinstance(raw["key"], str) or not isinstance(raw["generation"], str):
        raise JsonProjectionError("JSON projection entry identity is invalid")
    try:
        json.dumps(raw, allow_nan=False, sort_keys=True, separators=(",", ":"))
        return CatalogEntry(raw["key"], raw["generation"], dict(raw["values"]))
    except (TypeError, ValueError) as error:
        raise JsonProjectionError("JSON projection entry is invalid") from error


def _json_safe(value: object) -> object:
    """Return a lossless JSON-compatible copy or reject a derived value."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise JsonProjectionError("JSON projection mappings require string keys")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value
