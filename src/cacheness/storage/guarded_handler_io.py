"""Private staging adapter for handler payload I/O.

Handlers retain their public ``put(data, Path, config)`` and
``get(Path, metadata)`` interfaces, but this module ensures those paths are
never managed storage paths. Managed bytes cross the boundary only through
``ManagedFileOps`` and reads are supplied as one context-owned private copy.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from cacheness.error_handling import CacheReason, CacheUnsafePathError
from cacheness.interfaces import GuardedReadSnapshot, GuardedWriteResult

from .path_security import (
    ManagedFileOps,
    resolve_managed_locator,
    resolve_storage_root,
    validate_blob_id,
)


_SAFE_SUFFIX = re.compile(r"(?:\.[A-Za-z0-9_-]+){0,4}\Z")
_MAX_SUFFIX_LENGTH = 96


def _raise_invalid_stage_artifact() -> None:
    """Fail closed without exposing a handler-controlled path in messages."""
    raise CacheUnsafePathError(
        "Handler produced an unsafe staging artifact",
        reason=CacheReason.INVALID_IDENTIFIER,
    )


class GuardedHandlerIO:
    """Publish handler output and snapshot handler input through managed I/O.

    A separate instance is retained by each high-level store/cache so the
    resolved root descriptor and deterministic test hook have one owner.
    """

    def __init__(self, root: Path | str):
        self.root = resolve_storage_root(root)
        self.file_ops = ManagedFileOps(self.root)

    def close(self) -> None:
        """Release the managed root descriptor held by the adapter."""
        self.file_ops.close()

    @contextmanager
    def _private_stage(self) -> Iterator[Path]:
        """Yield a mode-restricted temporary directory outside managed storage."""
        with tempfile.TemporaryDirectory(prefix="cacheness-handler-") as temporary:
            stage_root = Path(temporary)
            stage_root.chmod(0o700)
            try:
                stage_root.resolve().relative_to(self.root)
            except ValueError:
                yield stage_root
            else:
                raise RuntimeError("Private handler staging directory overlaps storage root")

    @staticmethod
    def _safe_suffix(stage_base: Path, artifact: Path) -> str:
        """Return the handler-declared suffix only when it is bounded and opaque."""
        if not artifact.name.startswith(stage_base.name):
            _raise_invalid_stage_artifact()
        suffix = artifact.name[len(stage_base.name) :]
        if len(suffix) > _MAX_SUFFIX_LENGTH or not _SAFE_SUFFIX.fullmatch(suffix):
            _raise_invalid_stage_artifact()
        return suffix

    @staticmethod
    def _staged_artifact(stage_root: Path, stage_base: Path, result: Dict[str, Any]) -> Path:
        """Validate that a handler result identifies one ordinary stage file."""
        actual_path = result.get("actual_path")
        if not isinstance(actual_path, (str, Path)):
            _raise_invalid_stage_artifact()
        artifact = Path(actual_path)
        if not artifact.is_absolute():
            artifact = stage_root / artifact
        try:
            artifact.relative_to(stage_root)
        except ValueError:
            _raise_invalid_stage_artifact()
        try:
            artifact_stat = os.lstat(artifact)
        except OSError as exc:
            raise CacheUnsafePathError(
                "Handler staging artifact is unavailable",
                reason=CacheReason.PATH_RACE,
            ) from exc
        if not stat.S_ISREG(artifact_stat.st_mode):
            _raise_invalid_stage_artifact()
        return artifact

    def put(
        self,
        handler: Any,
        data: Any,
        storage_id: str,
        config: Any,
    ) -> GuardedWriteResult:
        """Serialize in a private stage and publish through ``ManagedFileOps``."""
        safe_storage_id = validate_blob_id(storage_id)
        with self._private_stage() as stage_root:
            stage_base = stage_root / "payload"
            raw_result = handler.put(data, stage_base, config)
            if not isinstance(raw_result, dict):
                _raise_invalid_stage_artifact()
            artifact = self._staged_artifact(stage_root, stage_base, raw_result)
            suffix = self._safe_suffix(stage_base, artifact)
            final_id = validate_blob_id(f"{safe_storage_id}{suffix}")

            with artifact.open("rb") as source:
                final_path = self.file_ops.write_stream(final_id, source, shard_chars=0)

            result: GuardedWriteResult = dict(raw_result)
            result["actual_path"] = str(final_path)
            result["file_size"] = artifact.stat().st_size
            metadata = result.get("metadata")
            result["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
            return result

    @contextmanager
    def open_snapshot(
        self,
        locator: Path | str,
        metadata: Dict[str, Any],
    ) -> Iterator[GuardedReadSnapshot]:
        """Yield one private no-follow snapshot without deserializing it.

        The managed file is opened exactly once by ``copy_to_stream``. Callers
        must finish hashing, signature checks, and ``handler.get`` before this
        context exits and deletes the snapshot.
        """
        managed_locator = resolve_managed_locator(
            self.root,
            locator,
            operation="snapshot",
        )
        suffix = "".join(managed_locator.suffixes)
        if len(suffix) > _MAX_SUFFIX_LENGTH or not _SAFE_SUFFIX.fullmatch(suffix):
            _raise_invalid_stage_artifact()

        with self._private_stage() as stage_root:
            snapshot_path = stage_root / f"snapshot{suffix}"
            with snapshot_path.open("xb") as destination:
                snapshot_path.chmod(0o600)
                self.file_ops.copy_to_stream(managed_locator, destination)
                destination.flush()
                os.fsync(destination.fileno())

            snapshot_metadata = dict(metadata)
            snapshot_metadata["actual_path"] = str(snapshot_path)
            yield GuardedReadSnapshot(snapshot_path, snapshot_metadata)


__all__ = ["GuardedHandlerIO", "GuardedReadSnapshot"]
