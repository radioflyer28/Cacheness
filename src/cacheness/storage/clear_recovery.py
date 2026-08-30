"""Bounded local recovery for a single BlobStore clear operation.

This is intentionally a narrow, local-filesystem primitive.  It records one
prepared or committed clear journal for an exact JSON metadata file and one
anchored payload root; it is not a general lifecycle or reconciliation engine.
"""

from __future__ import annotations

import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from cacheness.error_handling import CacheStorageError
from cacheness.json_utils import dumps as json_dumps, loads as json_loads
from cacheness.metadata import JsonBackend

from .path_security import ManagedFileOps


_JOURNAL_NAME = ".cacheness-clear-journal-v1.json"
_JOURNAL_OWNER = "cacheness.clear_recovery"
_JOURNAL_VERSION = 1


class ClearRecoveryCoordinator:
    """Coordinate one durable JSON-backed payload clear and startup recovery."""

    def __init__(self, file_ops: ManagedFileOps, backend: JsonBackend) -> None:
        self.file_ops = file_ops
        self.backend = backend
        self.journal_path = file_ops.root / _JOURNAL_NAME

    def clear(self, mappings: Sequence[tuple[str, Path]]) -> int:
        """Clear validated payload mappings with prepared/committed recovery evidence."""
        self._require_json_backend()
        journal = self._new_prepared_journal(mappings)
        self._create_journal(journal)

        try:
            for mapping in journal["mappings"]:
                self._stage_mapping(mapping)
            cleared_count = self.backend.clear_all()
        except BaseException as exc:
            if isinstance(exc, Exception):
                try:
                    self._rollback_prepared(journal)
                except Exception as rollback_exc:
                    raise CacheStorageError(
                        "BlobStore clear failed and prepared recovery could not complete",
                        context={"operation": "clear", "backend": "json"},
                    ) from rollback_exc
            raise exc

        journal["state"] = "committed"
        self._replace_journal(journal)
        try:
            self._roll_forward_committed(journal, wrap_errors=False)
        except BaseException:
            # Committed evidence is authoritative.  Startup recovery will retry
            # terminal payload erasure instead of reviving cleared metadata.
            raise
        return cleared_count

    def recover(self) -> None:
        """Converge one prior prepared or committed journal before serving data."""
        self._require_json_backend()
        if not self.file_ops.exists(self.journal_path):
            return

        journal = self._read_journal()
        self._validate_journal(journal)
        if journal["state"] == "prepared":
            self._rollback_prepared(journal)
            return
        self._roll_forward_committed(journal)

    def _require_json_backend(self) -> None:
        """Keep the tracer truthful until the explicit backend matrix is added."""
        if type(self.backend) is not JsonBackend:
            raise CacheStorageError(
                "Local clear recovery currently requires an exact JsonBackend",
                context={
                    "operation": "clear",
                    "backend": type(self.backend).__name__,
                },
            )

    def _new_prepared_journal(
        self, mappings: Sequence[tuple[str, Path]]
    ) -> dict[str, Any]:
        """Create a complete local journal before the first payload mutation."""
        operation_id = uuid.uuid4().hex
        serialized_mappings = []
        for index, (cache_key, original) in enumerate(mappings):
            serialized_mappings.append(
                {
                    "cache_key": cache_key,
                    "original": self._relative_locator(original),
                    "tombstone": f"clear-tombstone-{operation_id}-{index}",
                }
            )
        return {
            "version": _JOURNAL_VERSION,
            "owner": _JOURNAL_OWNER,
            "operation_id": operation_id,
            "state": "prepared",
            "topology": self._topology(),
            "metadata_snapshot": deepcopy(self.backend.load_metadata()),
            "mappings": serialized_mappings,
        }

    def _topology(self) -> dict[str, Any]:
        """Bind recovery to the exact local payload root and JSON metadata file."""
        root_stat = os.stat(self.file_ops.root)
        return {
            "root": str(self.file_ops.root),
            "root_device": root_stat.st_dev,
            "root_inode": root_stat.st_ino,
            "backend": "json",
            "metadata_file": str(self.backend.metadata_file.resolve()),
        }

    def _relative_locator(self, locator: Path) -> str:
        """Encode only root-relative journal paths for ManagedFileOps validation."""
        try:
            return str(locator.relative_to(self.file_ops.root))
        except ValueError as exc:
            raise CacheStorageError(
                "Clear mapping is outside the managed payload root",
                context={"operation": "clear", "backend": "json"},
            ) from exc

    def _create_journal(self, journal: dict[str, Any]) -> None:
        """Exclusively publish durable prepared evidence without overwriting residue."""
        try:
            self.file_ops.create_bytes_durable_exclusive(
                self.journal_path, self._encode_journal(journal)
            )
        except FileExistsError as exc:
            raise CacheStorageError(
                "A prior BlobStore clear journal requires recovery",
                context={"operation": "clear", "backend": "json"},
            ) from exc
        except Exception as exc:
            raise CacheStorageError(
                "Unable to create durable BlobStore clear journal",
                context={"operation": "clear", "backend": "json"},
            ) from exc

    def _replace_journal(self, journal: dict[str, Any]) -> None:
        """Durably publish a later prepared/committed journal state."""
        try:
            self.file_ops.write_bytes_durable(self.journal_path, self._encode_journal(journal))
        except Exception as exc:
            raise CacheStorageError(
                "Unable to durably publish BlobStore clear journal state",
                context={"operation": "clear", "backend": "json"},
            ) from exc

    def _read_journal(self) -> dict[str, Any]:
        """Read one bounded-format journal before selecting a recovery action."""
        try:
            document = json_loads(self.file_ops.read_bytes(self.journal_path))
        except Exception as exc:
            raise CacheStorageError(
                "Unable to read BlobStore clear recovery journal",
                context={"operation": "clear", "backend": "json"},
            ) from exc
        if not isinstance(document, dict):
            raise CacheStorageError(
                "BlobStore clear recovery journal is malformed",
                context={"operation": "clear", "backend": "json"},
            )
        return document

    def _validate_journal(self, journal: dict[str, Any]) -> None:
        """Fail closed unless journal state exactly matches this local topology."""
        required_fields = {
            "version",
            "owner",
            "operation_id",
            "state",
            "topology",
            "metadata_snapshot",
            "mappings",
        }
        if set(journal) != required_fields:
            self._invalid_journal()
        if (
            journal["version"] != _JOURNAL_VERSION
            or journal["owner"] != _JOURNAL_OWNER
            or not isinstance(journal["operation_id"], str)
            or journal["state"] not in {"prepared", "committed"}
            or journal["topology"] != self._topology()
            or not isinstance(journal["metadata_snapshot"], dict)
            or not isinstance(journal["mappings"], list)
        ):
            self._invalid_journal()
        for mapping in journal["mappings"]:
            if (
                not isinstance(mapping, dict)
                or set(mapping) != {"cache_key", "original", "tombstone"}
                or not all(isinstance(mapping[name], str) for name in mapping)
            ):
                self._invalid_journal()
            self._journal_locator(mapping["original"])
            self._journal_locator(mapping["tombstone"])

    def _invalid_journal(self) -> None:
        raise CacheStorageError(
            "BlobStore clear recovery journal is unsupported for this topology",
            context={"operation": "clear", "backend": "json"},
        )

    def _journal_locator(self, relative_path: str) -> Path:
        """Resolve every persisted recovery locator through ManagedFileOps' boundary."""
        return self.file_ops.root / relative_path

    def _stage_mapping(self, mapping: dict[str, str]) -> None:
        """Durably copy one live payload before its original is removed."""
        original = self._journal_locator(mapping["original"])
        tombstone = self._journal_locator(mapping["tombstone"])
        if not self.file_ops.exists(original):
            return
        try:
            with self.file_ops.open_read(original) as source:
                self.file_ops.write_stream_to_locator(tombstone, source)
        except Exception:
            # A stream publication can fail after creating a tombstone but
            # before this method has removed the original.  Do not route that
            # case through generic rollback: the original remains authoritative
            # and the same write seam may still be faulted.
            if self.file_ops.exists(original) and self.file_ops.exists(tombstone):
                self.file_ops.delete_durable(tombstone)
            raise
        self.file_ops._fsync_containing_directory(tombstone)
        if not self.file_ops.delete_durable(original):
            raise FileNotFoundError("Blob payload disappeared while staging clear")

    def _rollback_prepared(self, journal: dict[str, Any]) -> None:
        """Restore payloads, exact metadata, tombstones, then prepared evidence."""
        try:
            for mapping in reversed(journal["mappings"]):
                original = self._journal_locator(mapping["original"])
                tombstone = self._journal_locator(mapping["tombstone"])
                if self.file_ops.exists(tombstone):
                    with self.file_ops.open_read(tombstone) as source:
                        self.file_ops.write_stream_to_locator(original, source)
                    self.file_ops._fsync_containing_directory(original)
            self.backend.save_metadata(deepcopy(journal["metadata_snapshot"]))
            for mapping in journal["mappings"]:
                self.file_ops.delete_durable(self._journal_locator(mapping["tombstone"]))
            self.file_ops.delete_durable(self.journal_path)
        except Exception as exc:
            raise CacheStorageError(
                "Prepared BlobStore clear recovery could not be completed",
                context={"operation": "clear", "backend": "json"},
            ) from exc

    def _roll_forward_committed(
        self, journal: dict[str, Any], *, wrap_errors: bool = True
    ) -> None:
        """Erase terminal payload copies after the metadata clear is committed."""
        try:
            for mapping in journal["mappings"]:
                self.file_ops.delete_durable(self._journal_locator(mapping["original"]))
                self.file_ops.delete_durable(self._journal_locator(mapping["tombstone"]))
            self.file_ops.delete_durable(self.journal_path)
        except Exception as exc:
            if not wrap_errors:
                raise
            raise CacheStorageError(
                "Committed BlobStore clear recovery could not be completed",
                context={"operation": "clear", "backend": "json"},
            ) from exc

    @staticmethod
    def _encode_journal(journal: dict[str, Any]) -> bytes:
        """Serialize a journal before it crosses the managed-file boundary."""
        return json_dumps(journal, default=str).encode("utf-8")
