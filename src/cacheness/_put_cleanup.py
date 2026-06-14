"""Rollback guard for put() / update_data() write operations."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class _PutCleanup:
    """Tracks resources written during ``put()`` so they can be rolled back.

    Usage::

        cleanup = _PutCleanup()
        try:
            # ... write blob ...
            cleanup.blob_path = Path(actual_path)
            # ... upload to S3 ...
            cleanup.set_remote(blob_backend, blob_uri)
            # ... write metadata ...
            cleanup.commit()   # disarm — nothing will be rolled back
        except Exception:  # intentionally broad — docstring example
            cleanup.rollback()
            raise

    On ``rollback()``, every tracked resource is deleted (local file **and**
    remote object).  ``commit()`` disarms the cleanup so ``rollback()`` is a
    no-op.
    """

    __slots__ = (
        "_committed",
        "blob_path",
        "_blob_backend",
        "_blob_uri",
        "previous_blob_path",
        "previous_snapshot_path",
    )

    def __init__(self) -> None:
        self._committed = False
        self.blob_path: Optional[Path] = None
        self._blob_backend: Any = None
        self._blob_uri: Optional[str] = None
        self.previous_blob_path: Optional[Path] = None
        self.previous_snapshot_path: Optional[Path] = None

    def set_remote(self, blob_backend: Any, blob_uri: str) -> None:
        """Register a remote blob (e.g. S3 object) for rollback."""
        self._blob_backend = blob_backend
        self._blob_uri = blob_uri

    def snapshot_previous_blob(self, path: Path) -> None:
        """Move an existing local blob aside so rollback can restore it."""
        if not path.exists():
            return

        snapshot_path = Path(str(path) + ".prev")
        self.previous_blob_path = path
        self.previous_snapshot_path = snapshot_path
        os.replace(path, snapshot_path)
        logger.debug(f"Snapshotted previous blob: {path} -> {snapshot_path}")

    def commit(self) -> None:
        """Disarm — a subsequent ``rollback()`` will be a no-op."""
        if self.previous_snapshot_path is not None:
            try:
                if self.previous_snapshot_path.exists():
                    self.previous_snapshot_path.unlink()
                    logger.debug(
                        f"Removed previous blob snapshot: {self.previous_snapshot_path}"
                    )
            except OSError:
                pass
        self._committed = True

    def rollback(self) -> None:
        """Delete every tracked resource.  Safe to call multiple times."""
        if self._committed:
            return

        # Clean up local blob file
        if self.blob_path is not None:
            try:
                if self.blob_path.exists():
                    self.blob_path.unlink()
                    logger.debug(f"Cleaned up orphaned local blob: {self.blob_path}")
            except OSError:
                pass

        # Clean up remote blob (e.g. S3 object)
        if self._blob_backend is not None and self._blob_uri is not None:
            try:
                self._blob_backend.delete_blob(self._blob_uri)
                logger.debug(f"Cleaned up orphaned remote blob: {self._blob_uri}")
            except Exception:  # intentionally broad — cleanup must not raise
                logger.warning(
                    f"Failed to clean up orphaned remote blob: {self._blob_uri}"
                )

        if (
            self.previous_blob_path is not None
            and self.previous_snapshot_path is not None
        ):
            try:
                if self.previous_snapshot_path.exists():
                    os.replace(
                        self.previous_snapshot_path,
                        self.previous_blob_path,
                    )
                    logger.debug(
                        f"Restored previous blob snapshot: {self.previous_blob_path}"
                    )
            except OSError:
                pass

        self._committed = True  # prevent double-rollback
