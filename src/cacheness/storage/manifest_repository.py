"""Revision-bound JSON projection compatibility for lifecycle authority state.

``LifecycleAuthority`` owns canonical manifests. This module deliberately
contains no repository, conditional-write, import, or recovery implementation:
the JSON file is compatible, replaceable derived output for existing callers.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationRequiredError,
    CacheError,
)
from cacheness.json_utils import dumps as json_dumps

from .lifecycle_authority import LifecycleAuthority, ProjectionRevision
from .manifest import BlobManifestV1


_PROJECTION_PUBLISH_LOCK = RLock()


class JsonProjectionExporter:
    """Rebuild JSON metadata from a revision-bound authority snapshot.

    JSON is derived output: a failed or stale export is debt on authority
    state, never cause to roll back a committed transition or recover rows.
    """

    def __init__(
        self,
        authority: LifecycleAuthority,
        projection_path: Path | str,
        *,
        page_size: int = 128,
    ) -> None:
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("projection page_size must be a positive integer")
        self.authority = authority
        self.projection_path = Path(projection_path)
        self.page_size = page_size

    @staticmethod
    def _entry_data(key: str, raw_manifest: bytes) -> dict[str, Any]:
        """Render the established public entry shape from one committed row."""
        manifest = BlobManifestV1.from_canonical_bytes(raw_manifest)
        if manifest.key != key or manifest.state != "committed":
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority projection entry is not committed"
            )
        return {
            "cache_key": manifest.key,
            "data_type": manifest.handler_type,
            "file_size": manifest.byte_size,
            "created_at": manifest.created_at,
            "metadata": {
                **dict(manifest.user_metadata),
                **dict(manifest.handler_metadata),
                "actual_path": manifest.locator,
            },
        }

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Acknowledge the compatible JSON directory entry durably."""
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_snapshot(self, snapshot_path: Path, revision: ProjectionRevision) -> Path:
        """Keyset-stream one private authority backup into a JSON candidate."""
        if not self.projection_path.parent.is_dir():
            raise CacheBlobBackendError(
                "JSON projection directory is unavailable",
                context={"operation": "export_json_projection"},
            )
        descriptor, candidate_name = tempfile.mkstemp(
            prefix=f".{self.projection_path.name}.",
            suffix=".tmp",
            dir=self.projection_path.parent,
        )
        candidate = Path(candidate_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                destination.write('{"entries":{')
                first = True
                last_key = ""
                connection = sqlite3.connect(
                    f"{snapshot_path.as_uri()}?mode=ro&immutable=1",
                    uri=True,
                    isolation_level=None,
                )
                try:
                    while True:
                        rows = connection.execute(
                            "SELECT key, manifest FROM entries WHERE key > ? "
                            "ORDER BY key LIMIT ?",
                            (last_key, self.page_size),
                        ).fetchall()
                        if not rows:
                            break
                        for key, raw_manifest in rows:
                            if not isinstance(key, str) or not isinstance(raw_manifest, bytes):
                                raise CacheBlobMigrationRequiredError(
                                    "Lifecycle authority projection row is incompatible"
                                )
                            if not first:
                                destination.write(",")
                            destination.write(json_dumps(key))
                            destination.write(":")
                            destination.write(json_dumps(self._entry_data(key, raw_manifest)))
                            first = False
                            last_key = key
                finally:
                    connection.close()
                destination.write("}")
                destination.write(
                    f',"cache_hits":0,"cache_misses":0,'
                    f'"_cacheness_authority_revision":{revision.value}}}'
                )
                destination.flush()
                os.fsync(destination.fileno())
            return candidate
        except BaseException:
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
            raise

    @contextmanager
    def _publish_lock(self):
        """Order same-process projection writes outside authority transactions."""
        with _PROJECTION_PUBLISH_LOCK:
            yield

    def export(self) -> ProjectionRevision:
        """Atomically publish a snapshot and clean only its exact revision."""
        backup = getattr(self.authority, "projection_backup", None)
        if not callable(backup):
            raise CacheBlobBackendError(
                "Lifecycle authority cannot create a compatible JSON projection",
                context={"operation": "export_json_projection"},
            )
        candidate: Path | None = None
        try:
            with backup() as snapshot:
                candidate = self._write_snapshot(snapshot.path, snapshot.revision)
            with self._publish_lock():
                current_revision = self.authority.snapshot_state().revision
                if current_revision != snapshot.revision.value:
                    raise CacheBlobLifecycleConflictError("Projection revision changed")
                os.replace(candidate, self.projection_path)
                self._fsync_directory(self.projection_path)
                candidate = None
                return self.authority.compare_and_mark_projection(snapshot.revision)
        except (CacheBlobBackendError, CacheBlobLifecycleConflictError):
            raise
        except (CacheError, OSError, sqlite3.Error, ValueError) as error:
            raise CacheBlobBackendError(
                "Lifecycle authority JSON projection export failed",
                context={"operation": "export_json_projection"},
            ) from error
        finally:
            if candidate is not None:
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass
