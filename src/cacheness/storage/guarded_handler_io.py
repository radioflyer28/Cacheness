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
from dataclasses import dataclass
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


def _raise_stage_path_race() -> None:
    """Fail closed when the validated private-stage identity changes."""
    raise CacheUnsafePathError(
        "Handler staging artifact changed before publication",
        reason=CacheReason.PATH_RACE,
    )


@dataclass
class _StageArtifactRecord:
    """Identity captured from the validated staged regular-file descriptor."""

    st_dev: int
    st_ino: int
    st_mode: int
    st_nlink: int
    descriptor: int | None

    @classmethod
    def from_stat(cls, descriptor: int, artifact_stat: os.stat_result) -> _StageArtifactRecord:
        """Capture the file identity that publication must reopen exactly."""
        return cls(
            st_dev=artifact_stat.st_dev,
            st_ino=artifact_stat.st_ino,
            st_mode=stat.S_IFMT(artifact_stat.st_mode),
            st_nlink=artifact_stat.st_nlink,
            descriptor=descriptor,
        )

    def matches(self, artifact_stat: os.stat_result) -> bool:
        """Return whether an opened descriptor is the validated stage file."""
        return (
            artifact_stat.st_dev == self.st_dev
            and artifact_stat.st_ino == self.st_ino
            and stat.S_IFMT(artifact_stat.st_mode) == self.st_mode
            and artifact_stat.st_nlink == self.st_nlink
        )

    def close(self) -> None:
        """Release the retained validation descriptor exactly once."""
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


class _ValidatedStageArtifact(type(Path())):
    """A Path retaining the descriptor identity validated for publication."""

    record: _StageArtifactRecord


@dataclass
class GuardedStagedArtifact:
    """One validated private handler artifact held live for publication.

    The enclosing :meth:`GuardedHandlerIO.stage` context owns the temporary
    directory and retained descriptor.  Consumers can publish its bytes only
    while that context is live, preserving the validation identity captured
    before the lifecycle creates durable operation evidence.
    """

    stage_root: Path
    stage_base: Path
    artifact: _ValidatedStageArtifact
    raw_result: Dict[str, Any]

    @property
    def suffix(self) -> str:
        """Return the validated native handler suffix for a managed locator."""
        return GuardedHandlerIO._safe_suffix(self.stage_base, self.artifact)

    @contextmanager
    def open(self) -> Iterator[tuple[Any, int]]:
        """Yield the still-validated staged payload stream exactly once."""
        with GuardedHandlerIO._open_staged_artifact(
            self.stage_root, self.artifact, self.artifact.record
        ) as opened:
            yield opened

    def result_for(self, final_path: Path, file_size: int) -> GuardedWriteResult:
        """Return a compatible handler result naming a managed payload path."""
        result: GuardedWriteResult = dict(self.raw_result)
        result["actual_path"] = str(final_path)
        result["file_size"] = file_size
        metadata = result.get("metadata")
        result["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
        return result


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
    def stage(
        self,
        handler: Any,
        data: Any,
        config: Any,
    ) -> Iterator[GuardedStagedArtifact]:
        """Serialize a handler payload privately without managed side effects.

        Callers must create durable lifecycle evidence before calling
        :meth:`publish_generation`.  A serialization failure therefore leaves
        neither an operation record nor a managed candidate.
        """
        with self._private_stage() as stage_root:
            stage_base = stage_root / "payload"
            raw_result = handler.put(data, stage_base, config)
            if not isinstance(raw_result, dict):
                _raise_invalid_stage_artifact()
            artifact = self._staged_artifact(stage_root, stage_base, raw_result)
            try:
                yield GuardedStagedArtifact(stage_root, stage_base, artifact, raw_result)
            finally:
                artifact.record.close()

    def publish_generation(
        self,
        staged: GuardedStagedArtifact,
        locator: Path | str,
    ) -> GuardedWriteResult:
        """Exclusively publish a staged native payload at an immutable locator."""
        with staged.open() as (source, file_size):
            final_path = self.file_ops.create_stream_durable_exclusive(locator, source)
        result = staged.result_for(final_path, file_size)
        # This internal identity binds lifecycle verification to the exact
        # inode the managed exclusive publication installed. It never becomes
        # handler metadata or a persisted manifest field.
        result["_managed_generation_identity"] = self.file_ops.file_identity(final_path)
        return result

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
    def _staged_artifact(
        stage_root: Path, stage_base: Path, result: Dict[str, Any]
    ) -> _ValidatedStageArtifact:
        """Validate that a handler result identifies one ordinary stage file."""
        actual_path = result.get("actual_path")
        if not isinstance(actual_path, (str, Path)):
            _raise_invalid_stage_artifact()
        artifact = Path(actual_path)
        if not artifact.is_absolute():
            artifact = stage_root / artifact
        try:
            relative_parts = artifact.relative_to(stage_root).parts
        except ValueError:
            _raise_invalid_stage_artifact()
        if ".." in relative_parts:
            _raise_invalid_stage_artifact()
        current = stage_root
        for component in relative_parts:
            current = current / component
            try:
                component_stat = os.lstat(current)
            except OSError as exc:
                raise CacheUnsafePathError(
                    "Handler staging artifact is unavailable",
                    reason=CacheReason.PATH_RACE,
                ) from exc
            if stat.S_ISLNK(component_stat.st_mode):
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
        try:
            resolved_root = stage_root.resolve(strict=True)
            resolved_artifact = artifact.resolve(strict=True)
            resolved_artifact.relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise CacheUnsafePathError(
                "Handler staging artifact is unavailable",
                reason=CacheReason.PATH_RACE,
            ) from exc
        descriptor: int | None = None
        try:
            descriptor = os.open(
                resolved_artifact,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            descriptor_stat = os.fstat(descriptor)
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise CacheUnsafePathError(
                "Handler staging artifact is unavailable",
                reason=CacheReason.PATH_RACE,
            ) from exc
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
        ):
            os.close(descriptor)
            _raise_invalid_stage_artifact()
        if (
            descriptor_stat.st_dev != artifact_stat.st_dev
            or descriptor_stat.st_ino != artifact_stat.st_ino
        ):
            os.close(descriptor)
            _raise_stage_path_race()

        validated_artifact = _ValidatedStageArtifact(resolved_artifact)
        validated_artifact.record = _StageArtifactRecord.from_stat(
            descriptor, descriptor_stat
        )
        return validated_artifact

    @staticmethod
    @contextmanager
    def _open_staged_artifact(
        stage_root: Path, artifact: Path, record: _StageArtifactRecord
    ) -> Iterator[tuple[Any, int]]:
        """Yield one regular stage descriptor anchored below the private root."""
        descriptor: int | None = None
        source = None
        try:
            try:
                resolved_root = stage_root.resolve(strict=True)
                relative_parts = artifact.relative_to(resolved_root).parts
                if not relative_parts or ".." in relative_parts:
                    _raise_invalid_stage_artifact()

                supports_descriptor_walk = (
                    os.open in os.supports_dir_fd
                    and hasattr(os, "O_DIRECTORY")
                    and hasattr(os, "O_NOFOLLOW")
                )
                if supports_descriptor_walk:
                    directory_flags = (
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    )
                    root_descriptor = os.open(resolved_root, directory_flags)
                    parent_descriptor = root_descriptor
                    try:
                        for component in relative_parts[:-1]:
                            next_descriptor = os.open(
                                component,
                                directory_flags,
                                dir_fd=parent_descriptor,
                            )
                            if parent_descriptor != root_descriptor:
                                os.close(parent_descriptor)
                            parent_descriptor = next_descriptor
                        descriptor = os.open(
                            relative_parts[-1],
                            os.O_RDONLY | os.O_NOFOLLOW,
                            dir_fd=parent_descriptor,
                        )
                    finally:
                        if parent_descriptor != root_descriptor:
                            os.close(parent_descriptor)
                        os.close(root_descriptor)
                else:
                    descriptor = os.open(
                        artifact,
                        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    )
            except OSError as exc:
                raise CacheUnsafePathError(
                    "Handler staging artifact is unavailable",
                    reason=CacheReason.PATH_RACE,
                ) from exc

            descriptor_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(descriptor_stat.st_mode)
                or descriptor_stat.st_nlink != 1
            ):
                _raise_invalid_stage_artifact()
            if not record.matches(descriptor_stat):
                _raise_stage_path_race()
            if not supports_descriptor_walk:
                try:
                    pathname_stat = os.lstat(artifact)
                except OSError as exc:
                    raise CacheUnsafePathError(
                        "Handler staging artifact is unavailable",
                        reason=CacheReason.PATH_RACE,
                    ) from exc
                if (
                    stat.S_ISLNK(pathname_stat.st_mode)
                    or pathname_stat.st_dev != descriptor_stat.st_dev
                    or pathname_stat.st_ino != descriptor_stat.st_ino
                ):
                    _raise_stage_path_race()

            source = os.fdopen(descriptor, "rb")
            descriptor = None
            yield source, descriptor_stat.st_size
        finally:
            if source is not None:
                source.close()
            elif descriptor is not None:
                os.close(descriptor)

    def put(
        self,
        handler: Any,
        data: Any,
        storage_id: str,
        config: Any,
    ) -> GuardedWriteResult:
        """Serialize in a private stage and publish through ``ManagedFileOps``."""
        safe_storage_id = validate_blob_id(storage_id)
        with self.stage(handler, data, config) as staged:
            final_id = validate_blob_id(f"{safe_storage_id}{staged.suffix}")
            final_path = self.file_ops.blob_locator(final_id, shard_chars=0)
            with staged.open() as (source, file_size):
                published = self.file_ops.write_stream_to_locator(final_path, source)
            return staged.result_for(published, file_size)

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


__all__ = ["GuardedHandlerIO", "GuardedReadSnapshot", "GuardedStagedArtifact"]
