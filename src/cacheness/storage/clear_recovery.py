"""Bounded local recovery for one topology-bound BlobStore clear operation.

The coordinator intentionally supports only exact local JSON, SQLite, and
process-local in-memory metadata identities. It is not a general lifecycle or
reconciliation engine: remote, wrapped, and custom backends are refused before
any journal, payload, or metadata mutation can occur.
"""

from __future__ import annotations

import errno
import os
import re
import stat
import threading
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from cacheness.error_handling import CacheStorageError
from cacheness.json_utils import dumps as json_dumps, loads as json_loads
from cacheness.metadata import InMemoryBackend, JsonBackend, SqliteBackend

from .path_security import (
    ManagedFileOps,
    encode_physical_name,
    resolve_managed_locator,
    validate_blob_id,
)


MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_JOURNAL_ENTRIES = 100_000
MAX_JOURNAL_FIELD_BYTES = 8192

_JOURNAL_NAME = ".cacheness-clear-journal-v1.json"
_LOCK_NAME = ".cacheness-clear-admission.lock"
_JOURNAL_OWNER = "cacheness.clear_recovery"
_JOURNAL_VERSION = 1
_SNAPSHOT_FIELDS = {"entries", "cache_hits", "cache_misses"}
_JOURNAL_FIELDS = {
    "version",
    "owner",
    "operation_id",
    "state",
    "topology",
    "metadata_snapshot",
    "mappings",
}
_MAPPING_FIELDS = {"cache_key", "original", "tombstone"}
_SNAPSHOT_ENTRY_FIELDS = {
    "description",
    "data_type",
    "prefix",
    "created_at",
    "accessed_at",
    "file_size",
    "metadata",
}
_HANDLER_SUFFIX = re.compile(r"(?:\.[A-Za-z0-9_-]+){0,4}\Z")
_MAX_HANDLER_SUFFIX_LENGTH = 96
_CANDIDATE_PREFIX = re.compile(r"-candidate-[0-9a-f]{32}")
_GENERATION_PREFIX = re.compile(r"-generation-[0-9a-f]{32}-[0-9a-f]{32}")
_FAILURE_CONTEXT_KEY = "clear_recovery_failure"
_BACKEND_FAILURE = "backend_failure"
_LIFECYCLE_CONFLICT = "lifecycle_conflict"

_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[tuple[str, int, int], threading.Lock] = {}
_PROCESS_LOCK_OWNERS: dict[tuple[str, int, int], int] = {}


def _advisory_lock_available(root_path: Path) -> bool:
    """Return whether this platform offers the required POSIX advisory lock."""
    del root_path
    if os.name != "posix":
        return False
    try:
        import fcntl  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False
    return hasattr(fcntl, "flock")


class ClearRecoveryCoordinator:
    """Coordinate a bounded clear for one exact local metadata topology."""

    supported_local_kinds = ["json", "sqlite", "memory"]

    def __init__(
        self,
        file_ops: ManagedFileOps,
        backend: object,
        physical_name: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> None:
        self.file_ops = file_ops
        self.backend = backend
        self.kind = self._backend_kind(backend)
        self._physical_name = physical_name or self._blob_store_physical_name
        self.journal_path = file_ops.root / _JOURNAL_NAME
        self.lock_path = file_ops.root / _LOCK_NAME
        # A failed committed-journal publication may leave the durable state
        # unknown. Normal operations must not continue until recovery reaches
        # a terminal prepared rollback or committed roll-forward outcome.
        self._poisoned = False

    @staticmethod
    def _blob_store_physical_name(cache_key: str, _entry: dict[str, Any]) -> str:
        """Preserve BlobStore's canonical payload identity by default."""
        return encode_physical_name(cache_key, namespace="blob-store")

    @classmethod
    def can_coordinate(cls, backend: object) -> bool:
        """Return whether a concrete backend can prove a local clear boundary."""
        if type(backend) is JsonBackend:
            return backend._legacy_layout is None
        if type(backend) is SqliteBackend:
            return backend._legacy_layout is None and backend.db_file != ":memory:"
        return type(backend) is InMemoryBackend

    @classmethod
    def unsupported_error(cls, backend: object) -> CacheStorageError:
        """Build the common fail-before-mutation topology refusal."""
        return CacheStorageError(
            "BlobStore clear recovery requires an exact supported local metadata backend",
            context={
                "operation": "clear",
                "backend": type(backend).__name__,
                "supported_local_kinds": cls.supported_local_kinds,
            },
        )

    @classmethod
    def is_lifecycle_conflict(cls, error: CacheStorageError) -> bool:
        """Return whether tagged recovery evidence blocks normal lifecycle work."""
        return error.context.get(_FAILURE_CONTEXT_KEY) == _LIFECYCLE_CONFLICT

    @staticmethod
    def _backend_kind(backend: object) -> str:
        if type(backend) is JsonBackend:
            return "json"
        if type(backend) is SqliteBackend:
            return "sqlite"
        if type(backend) is InMemoryBackend:
            return "memory"
        raise ClearRecoveryCoordinator.unsupported_error(backend)

    @contextmanager
    def admission(self, *, blocking: bool = False) -> Iterator[None]:
        """Hold process and OS admission locks across recovery or one full clear."""
        if self._poisoned:
            raise self._poisoned_error()
        if not _advisory_lock_available(self.file_ops.root):
            raise self._admission_error("Reliable advisory locking is unavailable")

        try:
            process_lock, identity = self._process_lock()
        except OSError as exc:
            raise self._admission_error(
                "Unable to inspect the clear admission root"
            ) from exc
        owner_thread = threading.get_ident()
        with _PROCESS_LOCKS_GUARD:
            if _PROCESS_LOCK_OWNERS.get(identity) == owner_thread:
                raise self._admission_error(
                    "A same-root clear owner is already active",
                    lifecycle_conflict=True,
                )
        if not process_lock.acquire(blocking=blocking):
            raise self._admission_error(
                "A same-root clear owner is already active",
                lifecycle_conflict=True,
            )

        lock_descriptor: int | None = None
        try:
            with _PROCESS_LOCKS_GUARD:
                _PROCESS_LOCK_OWNERS[identity] = owner_thread
            lock_descriptor = self._acquire_advisory_lock(blocking=blocking)
            yield
        finally:
            release_error: OSError | None = None
            if lock_descriptor is not None:
                try:
                    self._release_advisory_lock(lock_descriptor)
                except OSError as exc:
                    release_error = exc
            with _PROCESS_LOCKS_GUARD:
                _PROCESS_LOCK_OWNERS.pop(identity, None)
            process_lock.release()
            if release_error is not None:
                raise self._admission_error(
                    "Unable to release the clear admission lock"
                ) from release_error

    @contextmanager
    def mutation_admission(self) -> Iterator[None]:
        """Exclude clear/recovery, then reconcile retained evidence before mutation.

        Phase 1 intentionally uses exclusive admission for all lifecycle
        mutations. This is narrower than a general reader/writer protocol but
        makes the clear snapshot and ordinary publication boundary linearizable.
        """
        with self.admission(blocking=True):
            try:
                self.recover()
                self._refresh_backend_view()
            except CacheStorageError as exc:
                self._raise_tagged_backend_failure(exc)
            except OSError as exc:
                raise self._admission_error(
                    "Clear recovery admission could not refresh backend state"
                ) from exc
            yield

    @contextmanager
    def read_admission(self) -> Iterator[None]:
        """Exclude a live clear while allowing an already-committed empty view.

        Reads cannot inspect a prepared rollback because its metadata and
        payloads may be between states. A committed journal, however, has
        already crossed the metadata authority boundary; readers may observe
        the cleared view while terminal tombstone reclamation is retried.
        """
        with self.admission(blocking=True):
            try:
                if self.file_ops.exists(self.journal_path):
                    journal = self._read_journal()
                    self._validate_journal(journal, allow_memory_nonce_mismatch=True)
                    if journal["state"] == "prepared":
                        raise self._admission_error(
                            "A prepared clear journal requires recovery before reads",
                            lifecycle_conflict=True,
                        )
                self._refresh_backend_view()
            except CacheStorageError as exc:
                self._raise_tagged_backend_failure(exc)
            except OSError as exc:
                raise self._admission_error(
                    "Clear recovery admission could not refresh backend state"
                ) from exc
            yield

    def _refresh_backend_view(self) -> None:
        """Prevent a second JSON instance from publishing a stale document view."""
        if self.kind == "json":
            self.backend._refresh_from_disk_for_clear_admission()

    def _process_lock(self) -> tuple[threading.Lock, tuple[str, int, int]]:
        root_stat = os.stat(self.file_ops.root)
        identity = (str(self.file_ops.root), root_stat.st_dev, root_stat.st_ino)
        with _PROCESS_LOCKS_GUARD:
            return _PROCESS_LOCKS.setdefault(identity, threading.Lock()), identity

    def _acquire_advisory_lock(self, *, blocking: bool) -> int:
        """Acquire the fixed lock file without treating contention as a retry."""
        import fcntl  # pylint: disable=import-outside-toplevel

        prepared = resolve_managed_locator(
            self.file_ops.root,
            _LOCK_NAME,
            operation="clear_admission",
            allow_missing_leaf=True,
        )
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(prepared, flags, 0o600)
        except OSError as exc:
            raise self._admission_error("Unable to open the clear admission lock") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise self._admission_error("Clear admission lock is not a regular file")
            lock_flags = fcntl.LOCK_EX
            if not blocking:
                lock_flags |= fcntl.LOCK_NB
            fcntl.flock(descriptor, lock_flags)
            return descriptor
        except OSError as exc:
            os.close(descriptor)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise self._admission_error("A process already owns the clear admission lock") from exc
            raise self._admission_error("Unable to acquire the clear admission lock") from exc
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _release_advisory_lock(descriptor: int) -> None:
        """Release the held advisory descriptor during every normal/error exit."""
        import fcntl  # pylint: disable=import-outside-toplevel

        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _admission_error(
        self,
        message: str,
        *,
        lifecycle_conflict: bool = False,
        context: dict[str, Any] | None = None,
    ) -> CacheStorageError:
        return CacheStorageError(
            message,
            context={
                "operation": "clear",
                "backend": self.kind,
                **(context or {}),
                _FAILURE_CONTEXT_KEY: (
                    _LIFECYCLE_CONFLICT if lifecycle_conflict else _BACKEND_FAILURE
                ),
            },
        )

    def _poisoned_error(self) -> CacheStorageError:
        return self._admission_error(
            "BlobStore clear recovery requires terminal reconciliation before normal operations",
            lifecycle_conflict=True,
        )

    def _raise_tagged_backend_failure(self, error: CacheStorageError) -> None:
        """Mark an admission-time recovery failure without changing its cause."""
        if error.context.get(_FAILURE_CONTEXT_KEY) is not None:
            raise error
        raise CacheStorageError(
            str(error),
            context={**error.context, _FAILURE_CONTEXT_KEY: _BACKEND_FAILURE},
        ) from error

    def _poison(self) -> None:
        """Fail closed after evidence cannot prove one terminal outcome."""
        self._poisoned = True

    def clear(self, mappings: Sequence[tuple[str, Path]]) -> int:
        """Clear preflighted payload mappings with prepared/committed evidence."""
        journal = self._new_prepared_journal(mappings)
        self._validate_journal(journal)
        self._create_journal(journal)
        # Retain the exact durable pre-commit record before the first staged
        # payload or metadata mutation. No allocation/copying boundary remains
        # between successful metadata clear and committed publication.
        prepared_journal = deepcopy(journal)

        try:
            for mapping in journal["mappings"]:
                self._stage_mapping(mapping)
            cleared_count = self._clear_backend()
        except BaseException as exc:
            if isinstance(exc, Exception):
                try:
                    self._rollback_prepared(journal)
                except Exception as rollback_exc:
                    self._poison()
                    raise self._admission_error(
                        "BlobStore clear failed and prepared recovery could not complete",
                        lifecycle_conflict=True,
                    ) from rollback_exc
            raise exc

        journal["state"] = "committed"
        try:
            self._replace_journal(journal)
        except BaseException as exc:
            # A failure before replacement is recoverable only when the live
            # evidence remains exactly the prepared journal. If replacement or
            # its directory acknowledgement might have happened, restoring the
            # snapshot could erase a successful clear, so retain evidence and
            # reject normal operations until a recovery pass can decide.
            if not isinstance(exc, Exception):
                # A BaseException models loss of control at this boundary in
                # the recovery tests. If application code catches it instead
                # of exiting, the live coordinator still cannot accept a
                # normal operation until a terminal recovery is performed.
                self._poison()
                raise
            publication_state = self._publication_state_after_failure(
                prepared_journal,
                journal,
            )
            if publication_state == "prepared":
                try:
                    self._rollback_prepared(prepared_journal)
                except Exception as rollback_exc:
                    self._poison()
                    raise self._admission_error(
                        "Committed clear journal publication failed and rollback could not complete",
                        lifecycle_conflict=True,
                    ) from rollback_exc
                raise

            self._poison()
            raise self._admission_error(
                "Committed clear journal publication has an unresolved recovery outcome",
                lifecycle_conflict=True,
                context={"publication_state": publication_state},
            ) from exc
        try:
            self._roll_forward_committed(journal, wrap_errors=False)
        except BaseException:
            # Committed evidence is authoritative. Startup recovery retries
            # terminal payload erasure instead of restoring cleared metadata.
            raise
        return cleared_count

    def recover(self) -> None:
        """Converge one previous clear before the store becomes usable."""
        if not self.file_ops.exists(self.journal_path):
            return

        journal = self._read_journal()
        self._validate_journal(journal, allow_memory_nonce_mismatch=True)
        if self.kind == "memory" and self._memory_nonce_mismatch(journal):
            # A new process has no trustworthy in-memory metadata. It must
            # erase residual payloads and never replay a stale disk snapshot.
            self._clear_backend()
            self._roll_forward_committed(journal)
            self._poisoned = False
            return
        if journal["state"] == "prepared":
            self._rollback_prepared(journal)
            self._poisoned = False
            return
        self._roll_forward_committed(journal)
        self._poisoned = False

    def _publication_state_after_failure(
        self,
        prepared_journal: dict[str, Any],
        committed_journal: dict[str, Any],
    ) -> str:
        """Classify the live journal without guessing past a failed barrier."""
        try:
            observed = self._read_journal()
            self._validate_journal(observed)
        except Exception:
            return "uncertain"
        if observed == prepared_journal:
            return "prepared"
        if observed == committed_journal:
            # Visible committed bytes do not prove that a failed directory
            # acknowledgement reached durable storage. Leave them intact for
            # restart recovery rather than attempting an unsafe rollback.
            return "committed"
        return "uncertain"

    def _new_prepared_journal(
        self, mappings: Sequence[tuple[str, Path]]
    ) -> dict[str, Any]:
        """Build a complete, bounded journal before the first payload mutation."""
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
            "metadata_snapshot": self._snapshot_backend(),
            "mappings": serialized_mappings,
        }

    def _topology(self) -> dict[str, Any]:
        """Bind recovery to root identity and the concrete metadata identity."""
        root_stat = os.stat(self.file_ops.root)
        topology = {
            "root": str(self.file_ops.root),
            "root_device": root_stat.st_dev,
            "root_inode": root_stat.st_ino,
            "backend": self.kind,
        }
        if self.kind == "json":
            topology["metadata_file"] = str(self.backend.metadata_file.resolve())
        elif self.kind == "sqlite":
            topology["database_file"] = str(Path(self.backend.db_file).resolve())
        else:
            topology["memory_nonce"] = self.backend._clear_recovery_nonce
        return topology

    def _memory_nonce_mismatch(self, journal: dict[str, Any]) -> bool:
        return journal["topology"]["memory_nonce"] != self._topology()["memory_nonce"]

    def _snapshot_backend(self) -> dict[str, Any]:
        return self.backend._snapshot_clear_state()

    def _restore_backend(self, snapshot: dict[str, Any]) -> None:
        self.backend._restore_clear_state(deepcopy(snapshot))

    def _clear_backend(self) -> int:
        return self.backend.clear_all()

    def _relative_locator(self, locator: Path) -> str:
        """Encode only root-relative journal paths through the containment boundary."""
        try:
            relative = locator.relative_to(self.file_ops.root)
        except ValueError as exc:
            raise CacheStorageError(
                "Clear mapping is outside the managed payload root",
                context={"operation": "clear", "backend": self.kind},
            ) from exc
        return str(relative)

    def _create_journal(self, journal: dict[str, Any]) -> None:
        """Exclusively publish prepared evidence without overwriting residue."""
        try:
            self.file_ops.create_bytes_durable_exclusive(
                self.journal_path, self._encode_journal(journal)
            )
        except FileExistsError as exc:
            raise self._admission_error(
                "A prior BlobStore clear journal requires recovery",
                lifecycle_conflict=True,
            ) from exc
        except CacheStorageError:
            raise
        except Exception as exc:
            raise CacheStorageError(
                "Unable to create durable BlobStore clear journal",
                context={"operation": "clear", "backend": self.kind},
            ) from exc

    def _replace_journal(self, journal: dict[str, Any]) -> None:
        """Durably publish the committed journal state."""
        try:
            self.file_ops.write_bytes_durable(self.journal_path, self._encode_journal(journal))
        except CacheStorageError:
            raise
        except Exception as exc:
            raise CacheStorageError(
                "Unable to durably publish BlobStore clear journal state",
                context={"operation": "clear", "backend": self.kind},
            ) from exc

    def _read_journal(self) -> dict[str, Any]:
        """Reject oversize evidence before loading JSON or invoking a callback."""
        try:
            journal_size = self.file_ops.get_size(self.journal_path)
            if journal_size < 0 or journal_size > MAX_JOURNAL_BYTES:
                self._invalid_journal()
            raw_journal = self.file_ops.read_bytes(self.journal_path)
            if len(raw_journal) > MAX_JOURNAL_BYTES:
                self._invalid_journal()
            document = json_loads(raw_journal)
        except CacheStorageError:
            raise
        except Exception as exc:
            raise CacheStorageError(
                "Unable to read BlobStore clear recovery journal",
                context={"operation": "clear", "backend": self.kind},
            ) from exc
        if not isinstance(document, dict):
            self._invalid_journal()
        return document

    def _validate_journal(
        self,
        journal: dict[str, Any],
        *,
        allow_memory_nonce_mismatch: bool = False,
    ) -> None:
        """Fail closed unless the full strict schema matches this local topology."""
        if not isinstance(journal, dict) or set(journal) != _JOURNAL_FIELDS:
            self._invalid_journal()
        if (
            type(journal["version"]) is not int
            or journal["version"] != _JOURNAL_VERSION
            or journal["owner"] != _JOURNAL_OWNER
            or not self._valid_operation_id(journal["operation_id"])
            or journal["state"] not in {"prepared", "committed"}
            or not isinstance(journal["topology"], dict)
            or not isinstance(journal["metadata_snapshot"], dict)
            or not isinstance(journal["mappings"], list)
            or len(journal["mappings"]) > MAX_JOURNAL_ENTRIES
        ):
            self._invalid_journal()

        self._validate_topology(
            journal["topology"],
            allow_memory_nonce_mismatch=allow_memory_nonce_mismatch,
        )
        self._validate_snapshot(journal["metadata_snapshot"])
        snapshot_entries = journal["metadata_snapshot"]["entries"]
        snapshot_keys = set(snapshot_entries)

        cache_keys = set()
        originals = set()
        tombstones = set()
        for index, mapping in enumerate(journal["mappings"]):
            if not isinstance(mapping, dict) or set(mapping) != _MAPPING_FIELDS:
                self._invalid_journal()
            cache_key = mapping["cache_key"]
            original = mapping["original"]
            tombstone = mapping["tombstone"]
            if not all(isinstance(value, str) for value in mapping.values()):
                self._invalid_journal()
            if not all(
                self._within_field_bound(value)
                for value in (cache_key, original, tombstone)
            ):
                self._invalid_journal()
            if (
                cache_key in cache_keys
                or original in originals
                or tombstone in tombstones
            ):
                self._invalid_journal()
            cache_keys.add(cache_key)
            originals.add(original)
            tombstones.add(tombstone)
            if cache_key not in snapshot_entries:
                self._invalid_journal()
            self._validate_original_locator(
                cache_key,
                original,
                snapshot_entries[cache_key],
            )
            self._validate_tombstone_locator(
                operation_id=journal["operation_id"],
                index=index,
                tombstone=tombstone,
            )
            original_locator = self._journal_locator(original)
            self._journal_locator(tombstone)
            self._validate_snapshot_entry(
                snapshot_entries[cache_key], original_locator
            )

        if (
            len(snapshot_keys) != len(snapshot_entries)
            or snapshot_keys != cache_keys
            or len(cache_keys) != len(journal["mappings"])
        ):
            self._invalid_journal()

    def _validate_topology(
        self,
        topology: dict[str, Any],
        *,
        allow_memory_nonce_mismatch: bool,
    ) -> None:
        expected = self._topology()
        if self.kind != "memory" or not allow_memory_nonce_mismatch:
            if topology != expected:
                self._invalid_journal()
            return

        if set(topology) != set(expected):
            self._invalid_journal()
        for name, value in expected.items():
            if name == "memory_nonce":
                if not isinstance(topology[name], str) or not topology[name]:
                    self._invalid_journal()
            elif topology[name] != value:
                self._invalid_journal()

    def _validate_snapshot(self, snapshot: dict[str, Any]) -> None:
        if set(snapshot) != _SNAPSHOT_FIELDS or not isinstance(snapshot["entries"], dict):
            self._invalid_journal()
        if not all(isinstance(key, str) for key in snapshot["entries"]):
            self._invalid_journal()
        if not all(isinstance(entry, dict) for entry in snapshot["entries"].values()):
            self._invalid_journal()
        if any(
            type(snapshot[counter]) is not int or snapshot[counter] < 0
            for counter in ("cache_hits", "cache_misses")
        ):
            self._invalid_journal()

    def _validate_original_locator(
        self,
        cache_key: str,
        original: str,
        snapshot_entry: dict[str, Any],
    ) -> None:
        """Accept only a direct canonical payload or one exact candidate form."""
        if Path(original).parent != Path("."):
            self._invalid_journal()
        try:
            physical_name = self._physical_name(cache_key, snapshot_entry)
        except Exception:
            self._invalid_journal()
        if not isinstance(physical_name, str) or not physical_name:
            self._invalid_journal()
        if not original.startswith(physical_name):
            self._invalid_journal()
        locator_suffix = original[len(physical_name) :]
        suffix = locator_suffix
        candidate_match = _CANDIDATE_PREFIX.match(locator_suffix)
        if candidate_match is not None:
            suffix = locator_suffix[candidate_match.end() :]
        else:
            generation_match = _GENERATION_PREFIX.match(locator_suffix)
            if generation_match is not None:
                suffix = locator_suffix[generation_match.end() :]
        if (
            len(suffix) > _MAX_HANDLER_SUFFIX_LENGTH
            or not _HANDLER_SUFFIX.fullmatch(suffix)
        ):
            self._invalid_journal()
        try:
            validate_blob_id(original)
        except Exception:
            self._invalid_journal()

    def _validate_tombstone_locator(
        self, *, operation_id: str, index: int, tombstone: str
    ) -> None:
        """Bind tombstones to one journal position and exclude control files."""
        expected = f"clear-tombstone-{operation_id}-{index}"
        if tombstone != expected or Path(tombstone).parent != Path("."):
            self._invalid_journal()

    def _validate_snapshot_entry(
        self, entry: object, original_locator: Path
    ) -> None:
        """Validate one complete standard entry before any restoration callback."""
        if not isinstance(entry, dict) or set(entry) != _SNAPSHOT_ENTRY_FIELDS:
            self._invalid_journal()
        if not all(
            isinstance(entry[field], str)
            for field in (
                "description",
                "data_type",
                "prefix",
                "created_at",
                "accessed_at",
            )
        ):
            self._invalid_journal()
        if type(entry["file_size"]) is not int or entry["file_size"] < 0:
            self._invalid_journal()
        metadata = entry["metadata"]
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("actual_path"), str
        ):
            self._invalid_journal()
        try:
            snapshot_locator = resolve_managed_locator(
                self.file_ops.root,
                metadata["actual_path"],
                operation="clear_recovery_snapshot",
            )
        except Exception:
            self._invalid_journal()
        if snapshot_locator != original_locator:
            self._invalid_journal()

    @staticmethod
    def _valid_operation_id(operation_id: object) -> bool:
        if not isinstance(operation_id, str) or len(operation_id) != 32:
            return False
        return all(character in "0123456789abcdef" for character in operation_id)

    @staticmethod
    def _within_field_bound(value: str) -> bool:
        try:
            return len(value.encode("utf-8")) <= MAX_JOURNAL_FIELD_BYTES
        except UnicodeEncodeError:
            return False

    def _invalid_journal(self) -> None:
        raise CacheStorageError(
            "BlobStore clear recovery journal is unsupported for this topology",
            context={"operation": "clear", "backend": self.kind},
        )

    def _journal_locator(self, relative_path: str) -> Path:
        """Resolve every persisted path through the managed containment boundary."""
        try:
            locator = Path(relative_path)
            if locator.is_absolute():
                self._invalid_journal()
            return resolve_managed_locator(
                self.file_ops.root,
                relative_path,
                operation="clear_recovery",
                allow_missing_leaf=True,
            )
        except CacheStorageError:
            raise
        except Exception:
            self._invalid_journal()

    def _stage_mapping(self, mapping: dict[str, str]) -> None:
        """Durably copy a live payload before unlinking its original."""
        original = self._journal_locator(mapping["original"])
        tombstone = self._journal_locator(mapping["tombstone"])
        if not self.file_ops.exists(original):
            return
        try:
            with self.file_ops.open_read(original) as source:
                self.file_ops.write_stream_to_locator(tombstone, source)
        except Exception:
            if self.file_ops.exists(original) and self.file_ops.exists(tombstone):
                self.file_ops.delete_durable(tombstone)
            raise
        self.file_ops._fsync_containing_directory(tombstone)
        if not self.file_ops.delete_durable(original):
            raise FileNotFoundError("Blob payload disappeared while staging clear")

    def _rollback_prepared(self, journal: dict[str, Any]) -> None:
        """Restore payloads and complete metadata before retiring prepared evidence."""
        try:
            for mapping in reversed(journal["mappings"]):
                original = self._journal_locator(mapping["original"])
                tombstone = self._journal_locator(mapping["tombstone"])
                if self.file_ops.exists(tombstone):
                    with self.file_ops.open_read(tombstone) as source:
                        self.file_ops.write_stream_to_locator(original, source)
                    self.file_ops._fsync_containing_directory(original)
            self._restore_backend(journal["metadata_snapshot"])
            for mapping in journal["mappings"]:
                self.file_ops.delete_durable(self._journal_locator(mapping["tombstone"]))
            self.file_ops.delete_durable(self.journal_path)
        except Exception as exc:
            raise CacheStorageError(
                "Prepared BlobStore clear recovery could not be completed",
                context={"operation": "clear", "backend": self.kind},
            ) from exc

    def _roll_forward_committed(
        self, journal: dict[str, Any], *, wrap_errors: bool = True
    ) -> None:
        """Erase terminal payloads and evidence without restoring metadata."""
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
                context={"operation": "clear", "backend": self.kind},
            ) from exc

    @staticmethod
    def _encode_journal(journal: dict[str, Any]) -> bytes:
        """Serialize a journal and apply the total encoded-byte bound."""
        encoded = json_dumps(journal).encode("utf-8")
        if len(encoded) > MAX_JOURNAL_BYTES:
            raise CacheStorageError(
                "BlobStore clear recovery journal exceeds its byte bound",
                context={"operation": "clear"},
            )
        return encoded


class LegacyClearEvidenceAdapter:
    """Reopen exact predecessor clear evidence without starting new work.

    Phase 1 clear journals remain an intentionally narrow compatibility input.
    This adapter delegates their established prepared/committed convergence to
    the frozen coordinator, but exposes no method for creating, replacing, or
    otherwise publishing predecessor evidence. New BlobStore clears are
    exclusively lifecycle operations and never call this adapter.
    """

    def __init__(self, file_ops: ManagedFileOps, backend: object) -> None:
        self._coordinator = ClearRecoveryCoordinator(file_ops, backend)

    @property
    def journal_path(self) -> Path:
        """Return the fixed predecessor-evidence locator for inspection only."""
        return self._coordinator.journal_path

    @property
    def kind(self) -> str:
        """Expose the validated predecessor backend identity for translation."""
        return self._coordinator.kind

    def has_evidence(self) -> bool:
        """Return whether the fixed predecessor evidence file already exists."""
        return self._coordinator.file_ops.exists(self.journal_path)

    def recover(self) -> bool:
        """Converge only evidence that was already present at reopen time.

        The coordinator validates the complete bounded predecessor schema and
        exact local topology before it restores or reclaims anything. Invalid
        bytes deliberately remain untouched for diagnosis.
        """
        if not self.has_evidence():
            return False
        self._coordinator.recover()
        return True


__all__ = ["ClearRecoveryCoordinator", "LegacyClearEvidenceAdapter"]
