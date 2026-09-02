"""Durable exact-byte persistence and bounded paging for lifecycle evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from threading import RLock, local
from typing import BinaryIO, Callable, Protocol

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationRequiredError,
    CacheManifestIntegrityError,
    CacheReason,
)

from .coordination import interprocess_open_file_lock, lock_stripe_index
from .integrity import sign_hmac_sha256, verify_hmac_sha256
from .manifest import MAX_MANIFEST_BYTES
from .operation_record import (
    MAX_CLEAR_TARGET_REFERENCE_CHUNKS,
    LifecycleOperationRecord,
)
from .path_security import ManagedFileOps, resolve_managed_locator, validate_blob_id


_CONDITIONAL_LOCK_STRIPES = tuple(RLock() for _ in range(64))
_OPERATION_LEASES = local()
_INVENTORY_SCHEMA_VERSION = 2
_INVENTORY_HEAD_MAX_BYTES = 4_096
_INVENTORY_EVENT_MIN_BYTES = 32 * 1024
_INVENTORY_FAMILIES = ("primary", "sidecar", "pending")
_INVENTORY_INITIALIZATION_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class OperationCursor:
    """Opaque generation-bound position after one operation-record page."""

    operation_id: str
    snapshot_high_water: int | None = None
    next_sequence: int | None = None

    def __post_init__(self) -> None:
        validate_blob_id(self.operation_id)
        if (self.snapshot_high_water is None) != (self.next_sequence is None):
            raise ValueError("operation cursor snapshot fields must be paired")
        for value in (self.snapshot_high_water, self.next_sequence):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("operation cursor snapshot field is invalid")

    @classmethod
    def before_first(cls, high_water: int) -> "OperationCursor":
        """Return the v2 position before sequence one of one fixed snapshot.

        ``operation_id`` is a compatibility projection only once a cursor has
        generation-bound sequence fields.  A reserved syntactically valid
        value lets reconciliation retain the first source member without
        overloading ``None``, which is the sole terminal representation.
        """
        return cls("0" * 32, snapshot_high_water=high_water, next_sequence=1)


@dataclass(frozen=True)
class OperationPage:
    """One bounded, lexically ordered page of opaque evidence bytes."""

    entries: tuple[tuple[str, bytes], ...]
    next_cursor: OperationCursor | None
    # Exact source positions immediately after each returned entry.  A
    # reconciler can consume only a prefix of a page without rebuilding a
    # mutable lexical cursor or replaying that prefix on resume.
    entry_next_cursors: tuple[OperationCursor, ...] = ()
    # Inventory members are work even when compacted or stale and therefore
    # yield no public evidence record.  Recovery uses this count to bound a
    # sparse historical scan independently of its action allowance.
    inspected_positions: int = 0

    def __post_init__(self) -> None:
        if self.entry_next_cursors and len(self.entry_next_cursors) != len(self.entries):
            raise ValueError("operation entry cursors must align with page entries")
        if type(self.inspected_positions) is not int or self.inspected_positions < 0:
            raise ValueError("operation inspected positions must be non-negative")


@dataclass(frozen=True)
class ReconciliationCheckpointCursor:
    """Opaque stable position after one reconciliation-sidecar page."""

    operation_id: str
    snapshot_high_water: int | None = None
    next_sequence: int | None = None

    def __post_init__(self) -> None:
        # This is an encrypted, opaque directory position rather than an
        # evidence identifier.  It may therefore represent a malformed name
        # that was inspected and skipped on a bounded page.
        if (
            not isinstance(self.operation_id, str)
            or not self.operation_id
            or len(self.operation_id.encode("utf-8")) > MAX_MANIFEST_BYTES
            or "/" in self.operation_id
            or "\\" in self.operation_id
            or "\x00" in self.operation_id
        ):
            raise ValueError("reconciliation checkpoint cursor is invalid")
        if (self.snapshot_high_water is None) != (self.next_sequence is None):
            raise ValueError("reconciliation checkpoint cursor snapshot fields must be paired")
        for value in (self.snapshot_high_water, self.next_sequence):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("reconciliation checkpoint cursor snapshot field is invalid")

    @classmethod
    def before_first(cls, high_water: int) -> "ReconciliationCheckpointCursor":
        """Return the v2 position before sequence one of one fixed snapshot."""
        # A sidecar cursor is an opaque scheduling name, never a direct path
        # or evidence identifier.  ``~`` cannot collide with the canonical
        # 32-hex operation IDs published by this repository.
        return cls("~", snapshot_high_water=high_water, next_sequence=1)


@dataclass(frozen=True)
class ReconciliationCheckpointPage:
    """Bounded opaque inventory of private reconciliation sidecars."""

    entries: tuple[tuple[str, bytes | None], ...]
    next_cursor: ReconciliationCheckpointCursor | None
    entry_next_cursors: tuple[ReconciliationCheckpointCursor, ...] = ()
    inspected_positions: int = 0

    def __post_init__(self) -> None:
        if self.entry_next_cursors and len(self.entry_next_cursors) != len(self.entries):
            raise ValueError("sidecar entry cursors must align with page entries")
        if type(self.inspected_positions) is not int or self.inspected_positions < 0:
            raise ValueError("sidecar inspected positions must be non-negative")


@dataclass(frozen=True)
class PendingControlCursor:
    """Opaque durable scheduling position for digest-bound pending controls."""

    name: str
    snapshot_high_water: int | None = None
    next_sequence: int | None = None

    def __post_init__(self) -> None:
        # Pending paging must advance across malformed prefixes too.  The
        # cursor is encrypted scheduling state, never a filename authority.
        if (
            not isinstance(self.name, str)
            or not self.name
            or len(self.name.encode("utf-8")) > MAX_MANIFEST_BYTES
            or "/" in self.name
            or "\\" in self.name
            or "\x00" in self.name
        ):
            raise ValueError("pending control cursor is invalid")
        if (self.snapshot_high_water is None) != (self.next_sequence is None):
            raise ValueError("pending control cursor snapshot fields must be paired")
        for value in (self.snapshot_high_water, self.next_sequence):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("pending control cursor snapshot field is invalid")


@dataclass(frozen=True)
class PendingControlPage:
    """A bounded page of exact pending control candidates and their bytes."""

    entries: tuple[tuple[str, bytes | None], ...]
    next_cursor: PendingControlCursor | None
    # Recovery may spend its action allowance on only a prefix of this page.
    # Keep the source-sequence continuation for every yielded member so that a
    # later call cannot rebuild a mutable lexical position and replay it.
    entry_next_cursors: tuple[PendingControlCursor, ...] = ()
    inspected_positions: int = 0

    def __post_init__(self) -> None:
        if self.entry_next_cursors and len(self.entry_next_cursors) != len(self.entries):
            raise ValueError("pending entry cursors must align with page entries")
        if type(self.inspected_positions) is not int or self.inspected_positions < 0:
            raise ValueError("pending inspected positions must be non-negative")


class OperationRecordRepository(Protocol):
    """Exact-byte evidence storage used by lifecycle and recovery code."""

    lifecycle_limits: LifecycleLimits

    def create_exclusive(
        self, record: LifecycleOperationRecord, raw_record: bytes
    ) -> Path:
        """Durably create evidence before a managed candidate side effect."""

    def get_raw(self, operation_id: str) -> bytes | None:
        """Return exact evidence bytes, or ``None`` only for an absent record."""

    def checkpoint_if_exact(
        self,
        record: LifecycleOperationRecord,
        *,
        expected_raw: bytes,
        raw_record: bytes,
    ) -> None:
        """Checkpoint only if the exact previously observed bytes still exist."""

    def retire_if_exact(
        self, record: LifecycleOperationRecord, *, expected_raw: bytes
    ) -> None:
        """Retire evidence only after the exact terminal record remains current."""

    def list_page(
        self,
        cursor: OperationCursor | None = None,
        *,
        page_size: int | None = None,
        max_inspections: int | None = None,
    ) -> OperationPage:
        """Return a stable, bounded inventory page without interpreting evidence."""

    def list_reconciliation_checkpoint_page(
        self,
        cursor: ReconciliationCheckpointCursor | None = None,
        *,
        page_size: int | None = None,
        max_inspections: int | None = None,
    ) -> ReconciliationCheckpointPage:
        """Return bounded private-sidecar bytes without assigning authority."""

    def create_clear_target_page_exclusive(
        self, operation_id: str, page_id: str, raw_record: bytes
    ) -> Path:
        """Persist one exact clear target page before advancing its cursor."""

    def get_clear_target_page_raw(
        self, operation_id: str, page_id: str
    ) -> bytes | None:
        """Return one exact target page without assigning it authority."""

    def create_clear_target_checkpoint_exclusive(
        self, operation_id: str, page_id: str, raw_record: bytes
    ) -> Path:
        """Persist zero progress before destructive target work begins."""

    def get_clear_target_checkpoint_raw(
        self, operation_id: str, page_id: str
    ) -> bytes | None:
        """Return one exact clear checkpoint without interpreting progress."""

    def checkpoint_clear_target_if_exact(
        self,
        operation_id: str,
        page_id: str,
        *,
        expected_raw: bytes,
        raw_record: bytes,
    ) -> None:
        """Advance target/page progress only from the exact observed bytes."""


class FileOperationRecordRepository:
    """Local operation evidence repository inside the managed store root."""

    def __init__(
        self,
        file_ops: ManagedFileOps,
        *,
        lifecycle_limits: LifecycleLimits,
        initialization_key_provider: Callable[[], bytes] | None = None,
    ):
        self.file_ops = file_ops
        # Retain the one caller-owned policy object; no field copies are used.
        self.lifecycle_limits = lifecycle_limits
        self._lock_handle_guard = RLock()
        self._lock_handles: dict[str, tuple[Path, BinaryIO, tuple[int, int]]] = {}
        self._pending_inventory_observer_depth = 0
        # This narrowly scoped provider is called only after the bounded
        # all-family compatibility proof for a fresh store, or to verify an
        # already-present initialization record.  It must not silently create
        # a replacement trust root while reading an existing store.
        self._initialization_key_provider = initialization_key_provider
        self.file_ops.pending_control_observer = self._record_pending_control

    def close(self) -> None:
        """Release retained bounded evidence-lock descriptors on store close."""
        first_error: Exception | None = None
        with self._lock_handle_guard:
            # Do not discard a descriptor from retry bookkeeping until its
            # close has a known successful outcome.  A failed close may leave
            # a process-scoped advisory lock live, so a later BlobStore.close
            # must retry rather than reporting a fictional terminal state.
            for lock_identity, (_locator, handle, _identity) in tuple(
                self._lock_handles.items()
            ):
                try:
                    handle.close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                    continue
                self._lock_handles.pop(lock_identity, None)
        if first_error is not None:
            raise first_error

    def _conditional_lock_for(self, operation_id: str) -> RLock:
        """Return a shared bounded in-process stripe for one evidence ID.

        The stripe is deliberately module-wide so independently constructed
        repositories in the same process cannot bypass their common advisory
        file lock.  It only orders evidence transitions; it is never an
        authority substitute for the cross-process compare-and-mutate lock.
        """
        stripe = lock_stripe_index(self.file_ops.root, operation_id)
        return _CONDITIONAL_LOCK_STRIPES[stripe]

    def _operation_lease_key(self, operation_id: str) -> tuple[tuple[int, int], str]:
        """Identify one descriptor-root-scoped lease in this calling thread."""
        return self.file_ops.root_identity, operation_id

    @staticmethod
    def _held_operation_leases() -> dict[tuple[tuple[int, int], str], int]:
        """Return the current thread's reentrant operation-lease depths."""
        leases = getattr(_OPERATION_LEASES, "leases", None)
        if leases is None:
            leases = {}
            _OPERATION_LEASES.leases = leases
        return leases

    def _has_held_operation_lease(self, operation_id: str) -> bool:
        """Return whether this exact operation already owns its lease.

        Reentrancy is safe only for the same record and the same retained
        authority lock.  A parent clear record is not an authority lease for a
        child delete record: treating it as one loses the child's cross-process
        exact-CAS exclusion.
        """
        return self._held_operation_leases().get(self._operation_lease_key(operation_id), 0) > 0

    def _conditional_lock_locator(self, operation_id: str) -> Path:
        """Return one of a fixed number of durable evidence-CAS lock stripes."""
        stripe = lock_stripe_index(self.file_ops.root, operation_id)
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".conditional-locks" / f"{stripe:02x}.lock",
            operation="operation_record_conditional_lock",
            allow_missing_leaf=True,
        )

    def _retained_conditional_lock(
        self, lock_identity: str
    ) -> tuple[Path, BinaryIO, tuple[int, int]]:
        """Return one retained descriptor for a fixed bounded lock stripe."""
        return self._retained_lock(
            lock_identity, self._conditional_lock_locator(lock_identity)
        )

    def _retained_lock(
        self, lock_identity: str, locator: Path
    ) -> tuple[Path, BinaryIO, tuple[int, int]]:
        """Create and retain one single-linked managed control-lock descriptor."""
        # Operation IDs intentionally share a fixed number of lock stripes.
        # Cache by the contained stripe pathname, not the caller's identity, so
        # a long-lived store retains only the bounded stripe set.
        cache_identity = str(locator.relative_to(self.file_ops.root))
        with self._lock_handle_guard:
            cached = self._lock_handles.get(cache_identity)
            if cached is not None:
                return cached
            expected = self.file_ops.ensure_lifecycle_lock(locator)
            handle = self.file_ops.open_verified_regular_file(locator)
            try:
                observed = (os.fstat(handle.fileno()).st_dev, os.fstat(handle.fileno()).st_ino)
                if observed != expected:
                    raise CacheBlobLifecycleConflictError(
                        "Lifecycle evidence lock changed during descriptor retention",
                        context={"operation": "conditional_evidence"},
                    )
            except BaseException:
                handle.close()
                raise
            cached = (locator, handle, expected)
            self._lock_handles[cache_identity] = cached
            return cached

    def _clear_resume_lock_locator(self) -> Path:
        """Return the one store-wide clear-resume lease, distinct from CAS stripes."""
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".clear-resume.lock",
            operation="clear_resume_lock",
            allow_missing_leaf=True,
        )

    @contextmanager
    def clear_operation_transition(self, operation_id: str) -> Iterator[None]:
        """Serialize clear continuation without inheriting child record authority.

        Clear is the one aggregate operation allowed to use a store-wide lease.
        It guards resume/recovery of the clear control flow only; each target's
        delete/checkpoint still takes its own exact evidence lease.
        """
        del operation_id
        lock_identity = "clear-resume"
        with self._conditional_lock_for(lock_identity):
            locator, handle, expected_identity = self._retained_lock(
                lock_identity, self._clear_resume_lock_locator()
            )
            self.file_ops.assert_retained_lock_identity(locator, expected_identity)
            with interprocess_open_file_lock(
                handle, exclusive=True, operation="clear_resume"
            ):
                self.file_ops.assert_retained_lock_identity(locator, expected_identity)
                yield

    @contextmanager
    def _conditional_transition(
        self, transition_id: str, *, operation_id: str | None = None
    ) -> Iterator[None]:
        """Serialize one exact evidence CAS across repository objects/processes.

        ``flock`` is acquired only around read/compare/replace-or-delete.  It
        never spans handler serialization, manifest publication, or unrelated
        operation IDs, preserving the phase's no-global-normal-lock contract.
        """
        if operation_id is not None and self._has_held_operation_lease(operation_id):
            # This is true reentrancy for the exact record whose advisory lock
            # is already held.  No other operation ID can bypass the file lock.
            yield
            return

        lock_identity = (
            f"operation:{operation_id}" if operation_id is not None else transition_id
        )
        with self._conditional_lock_for(lock_identity):
            lock_locator, handle, expected_identity = self._retained_conditional_lock(
                lock_identity
            )
            self.file_ops.assert_retained_lock_identity(lock_locator, expected_identity)
            with interprocess_open_file_lock(
                handle,
                exclusive=True,
                operation="conditional_evidence",
            ):
                self.file_ops.assert_retained_lock_identity(
                    lock_locator, expected_identity
                )
                yield

    @contextmanager
    def operation_transition(self, operation_id: str) -> Iterator[None]:
        """Hold a narrow cross-process lease for one resumable operation."""
        key = self._operation_lease_key(operation_id)
        leases = self._held_operation_leases()
        if leases.get(key, 0):
            leases[key] += 1
            try:
                yield
            finally:
                leases[key] -= 1
            return

        if any(
            leased_root == self.file_ops.root_identity and depth > 0
            for (leased_root, _leased_operation_id), depth in leases.items()
        ):
            raise CacheBlobLifecycleConflictError(
                "Nested lifecycle operation leases require releasing the parent first",
                context={"operation_id": operation_id, "operation": "operation_transition"},
            )

        with self._conditional_transition(
            f"operation:{operation_id}", operation_id=operation_id
        ):
            leases[key] = 1
            try:
                yield
            finally:
                del leases[key]

    def _read_bounded(self, locator: Path, *, operation: str) -> bytes | None:
        """Read exact evidence only after enforced descriptor-bounded limits."""
        maximum = self.lifecycle_limits.max_operation_record_bytes
        try:
            size = self.file_ops.get_size(locator)
            if size < 0:
                return None
            if size > maximum:
                raise CacheManifestIntegrityError(
                    "Lifecycle control evidence exceeds the configured byte limit",
                    reason=CacheReason.MANIFEST_BOUNDS,
                )
            return self.file_ops.read_bytes_bounded(locator, max_bytes=maximum)
        except ValueError as exc:
            raise CacheManifestIntegrityError(
                "Lifecycle control evidence exceeds the configured byte limit",
                reason=CacheReason.MANIFEST_BOUNDS,
            ) from exc
        except FileNotFoundError:
            return None
        except CacheManifestIntegrityError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle control evidence could not be read",
                context={"operation": operation},
            ) from exc

    def _legacy_inventory_locator(self, family: str) -> Path:
        """Return the retired whole-history inventory only for detection."""
        if family not in {"primary", "sidecar", "pending"}:
            raise ValueError("unknown lifecycle inventory family")
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".cacheness-inventory-v1" / f"{family}.json",
            operation="lifecycle_inventory",
            allow_missing_leaf=True,
        )

    def _inventory_head_locator(self, family: str) -> Path:
        """Return the bounded durable sequence head for one evidence family."""
        if family not in {"primary", "sidecar", "pending"}:
            raise ValueError("unknown lifecycle inventory family")
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".cacheness-inventory-v2" / family / "head.json",
            operation="lifecycle_inventory",
            allow_missing_leaf=True,
        )

    def _inventory_initialization_locator(self) -> Path:
        """Return the signed all-family v3 initialization provenance record."""
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".cacheness-inventory-v2" / "initialized-v3.json",
            operation="lifecycle_inventory",
            allow_missing_leaf=True,
        )

    def _legacy_inventory_initialization_locator(self) -> Path:
        """Return the unsigned v2 marker solely so it can fail closed.

        A v2 marker/head pair cannot prove it was published after one
        all-family compatibility decision.  It is deliberately migration
        evidence, not a compatibility shortcut for a missing sibling family.
        """
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".cacheness-inventory-v2" / "initialized",
            operation="lifecycle_inventory",
            allow_missing_leaf=True,
        )

    @staticmethod
    def _initialization_signing_bytes() -> bytes:
        """Return the stable HMAC preimage for all-family provenance."""
        return json.dumps(
            {
                "families": list(_INVENTORY_FAMILIES),
                "version": _INVENTORY_INITIALIZATION_SCHEMA_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _initialization_key(self) -> bytes:
        """Read the caller-owned trust root needed to verify v3 provenance."""
        if self._initialization_key_provider is None:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle inventory initialization requires a signing-key provider",
                context={"operation": "inventory_migration"},
            )
        key = self._initialization_key_provider()
        if type(key) is not bytes or len(key) != 32:
            raise CacheBlobBackendError(
                "Lifecycle inventory initialization key is invalid",
                context={"operation": "inventory"},
            )
        return key

    def _inventory_event_locator(self, family: str, sequence: int) -> Path:
        """Return one immutable event in the family-local monotonic sequence."""
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("inventory sequence is invalid")
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations")
            / ".cacheness-inventory-v2"
            / family
            / f"event-{sequence:020d}.json",
            operation="lifecycle_inventory",
            allow_missing_leaf=True,
        )

    def _inventory_event_max_bytes(self) -> int:
        """Bound one index event independently from one lifecycle record."""
        return max(
            _INVENTORY_EVENT_MIN_BYTES,
            self.lifecycle_limits.max_operation_field_bytes * 2 + 512,
        )

    @staticmethod
    def _empty_inventory_head() -> dict[str, int]:
        return {
            "version": _INVENTORY_SCHEMA_VERSION,
            "next_sequence": 1,
            "compact_next_sequence": 1,
            # The lowest sequence that can still name live evidence. Sparse
            # gaps below this floor were exact-revalidated as stale, so every
            # high-water cursor may safely skip them without changing its
            # snapshot membership.
            "first_live_sequence": 1,
            # A non-zero target is an exact durable upper bound for stale
            # scheduling work charged by a completed lifecycle boundary.
            # Recovery consumes it in caller-bounded windows before a later
            # aggregate clear can build an inventory snapshot.
            "maintenance_target_sequence": 0,
        }

    def _has_preindex_evidence(self, family: str) -> bool:
        """Fail closed if bounded legacy inspection finds known live evidence.

        A missing v2 head is only an empty inventory when no direct primary or
        sidecar evidence exists.  We deliberately do not rebuild a mutable
        directory order: an upgrade must use an explicit migration rather than
        silently omitting old recovery debt from a claimed high-water snapshot.
        """
        if family == "primary":
            name_filter = self._is_primary_filename
        elif family == "sidecar":
            name_filter = self._is_sidecar_filename
        else:
            name_filter = self._is_eligible_pending_name
        operations_directory = resolve_managed_locator(
            self.file_ops.root,
            "operations",
            operation="detect_legacy_lifecycle_inventory",
            allow_missing_leaf=True,
        )
        try:
            names, next_name = self.file_ops.list_directory_names_bounded(
                operations_directory,
                cursor=None,
                max_names=1,
                # Legacy discovery is deliberately bounded.  If its namespace
                # is too large to prove empty, callers receive the same typed
                # migration-required outcome instead of an unbounded scan.
                max_inventory_names=self.lifecycle_limits.max_inventory_items,
                max_scanned_names=self.lifecycle_limits.max_inventory_items,
                operation="detect_legacy_lifecycle_inventory",
                name_filter=name_filter,
            )
        except CacheBlobBackendError as exc:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle evidence inventory requires an explicit migration",
                context={"family": family, "operation": "inventory_migration"},
            ) from exc
        if next_name is not None:
            # A compatibility proof may not turn a bounded prefix of an old
            # namespace into a claim that the whole namespace is empty.
            raise CacheBlobMigrationRequiredError(
                "Lifecycle evidence inventory requires an explicit migration",
                context={"family": family, "operation": "inventory_migration"},
            )
        return bool(names)

    def _assert_fresh_inventory_namespace(self) -> None:
        """Prove every legacy family absent before publishing v2 provenance.

        This check lives under the same inventory transition as the marker and
        heads. A missing manifest key is deliberately irrelevant: it is not
        evidence that a pre-index operations directory is fresh.
        """
        for family in _INVENTORY_FAMILIES:
            try:
                legacy_size = self.file_ops.get_size(self._legacy_inventory_locator(family))
            except (OSError, ValueError) as exc:
                raise CacheBlobMigrationRequiredError(
                    "Lifecycle evidence inventory requires an explicit migration",
                    context={"family": family, "operation": "inventory_migration"},
                ) from exc
            if legacy_size >= 0 or self._has_preindex_evidence(family):
                raise CacheBlobMigrationRequiredError(
                    "Lifecycle evidence predates its durable inventory",
                    context={"family": family, "operation": "inventory_migration"},
                )

    def _record_pending_control(self, locator: Path, name: str, raw: bytes) -> None:
        """Index one future digest-bound pending control before it is created.

        The observer is private to ``ManagedFileOps`` and ignores the index's
        own control files.  Re-entrant event publication therefore cannot
        recursively invent pending work, while every real lifecycle candidate
        gets a monotonic pending-family membership slot before its first write.
        """
        if self._pending_inventory_observer_depth:
            return
        try:
            relative = locator.relative_to(self.file_ops.root)
        except ValueError:
            return
        if not relative.parts or relative.parts[0] != "operations":
            return
        if len(relative.parts) > 1 and relative.parts[1] == ".cacheness-inventory-v2":
            return
        self._pending_inventory_observer_depth += 1
        try:
            self._append_inventory_event("pending", name, raw)
        finally:
            self._pending_inventory_observer_depth -= 1

    def _decode_inventory_head(self, family: str, raw: bytes) -> dict[str, int]:
        """Validate one present head without inferring absence or migration."""
        try:
            state = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CacheBlobBackendError(
                "Lifecycle inventory is invalid",
                context={"operation": "inventory", "family": family},
            ) from exc
        if not isinstance(state, dict) or not (
            set(state) == {"version", "next_sequence", "compact_next_sequence"}
            or set(state)
            == {
                "version",
                "next_sequence",
                "compact_next_sequence",
                "first_live_sequence",
            }
            or set(state)
            == {
                "version",
                "next_sequence",
                "compact_next_sequence",
                "first_live_sequence",
                "maintenance_target_sequence",
            }
        ):
            raise CacheBlobBackendError(
                "Lifecycle inventory schema is invalid",
                context={"operation": "inventory", "family": family},
            )
        if (
            state["version"] != _INVENTORY_SCHEMA_VERSION
            or type(state["next_sequence"]) is not int
            or type(state["compact_next_sequence"]) is not int
            or state["next_sequence"] <= 0
            or not 1 <= state["compact_next_sequence"] <= state["next_sequence"]
        ):
            raise CacheBlobBackendError(
                "Lifecycle inventory version is invalid",
                context={"operation": "inventory", "family": family},
            )
        if "first_live_sequence" not in state:
            # Existing v2 heads remain valid.  Their history has not been
            # compacted through this floor yet, so start conservatively.
            state["first_live_sequence"] = 1
        if "maintenance_target_sequence" not in state:
            # Existing heads predate an exact debt boundary.  Treat their
            # already-allocated range as recovery work rather than allowing a
            # clear to re-scan a lifetime of stale slots as target pages.
            state["maintenance_target_sequence"] = state["next_sequence"] - 1
        if (
            type(state["first_live_sequence"]) is not int
            or not 1 <= state["first_live_sequence"] <= state["next_sequence"]
        ):
            raise CacheBlobBackendError(
                "Lifecycle inventory live-sequence floor is invalid",
                context={"operation": "inventory", "family": family},
            )
        if (
            type(state["maintenance_target_sequence"]) is not int
            or not 0
            <= state["maintenance_target_sequence"]
            < state["next_sequence"]
        ):
            raise CacheBlobBackendError(
                "Lifecycle inventory maintenance target is invalid",
                context={"operation": "inventory", "family": family},
            )
        return state

    def _read_present_inventory_head(self, family: str) -> dict[str, int] | None:
        """Return a validated present head, keeping a missing head distinct."""
        try:
            raw = self.file_ops.read_bytes_bounded(
                self._inventory_head_locator(family),
                max_bytes=_INVENTORY_HEAD_MAX_BYTES,
            )
        except FileNotFoundError:
            return None
        return self._decode_inventory_head(family, raw)

    def _has_current_inventory_initialization(self) -> bool:
        """Verify authenticated v3 provenance for one all-family decision.

        The record is the sole proof that a missing family head belongs to a
        store which was proven empty as a whole.  Older unsigned markers are
        intentionally not upgraded in place: their publication order did not
        serialize against every family evidence transition.
        """
        try:
            raw = self.file_ops.read_bytes_bounded(
                self._inventory_initialization_locator(),
                max_bytes=_INVENTORY_HEAD_MAX_BYTES,
            )
        except FileNotFoundError:
            return False
        except ValueError as exc:
            raise CacheBlobBackendError(
                "Lifecycle inventory initialization marker exceeds its bound",
                context={"operation": "inventory"},
            ) from exc
        try:
            record = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CacheBlobBackendError(
                "Lifecycle inventory initialization marker is invalid",
                context={"operation": "inventory"},
            ) from exc
        if (
            not isinstance(record, dict)
            or set(record) != {"families", "signature", "version"}
            or record["version"] != _INVENTORY_INITIALIZATION_SCHEMA_VERSION
            or record["families"] != list(_INVENTORY_FAMILIES)
            or not isinstance(record["signature"], str)
        ):
            raise CacheBlobBackendError(
                "Lifecycle inventory initialization marker is invalid",
                context={"operation": "inventory"},
            )
        if not verify_hmac_sha256(
            self._initialization_signing_bytes(),
            record["signature"],
            self._initialization_key(),
        ):
            raise CacheManifestIntegrityError(
                "Lifecycle inventory initialization provenance is unauthenticated",
                reason=CacheReason.MANIFEST_SIGNATURE_INVALID,
            )
        return True

    def _has_legacy_initialization_marker(self) -> bool:
        """Return whether the retired unsigned v2 marker is present."""
        try:
            self.file_ops.read_bytes_bounded(
                self._legacy_inventory_initialization_locator(),
                max_bytes=_INVENTORY_HEAD_MAX_BYTES,
            )
        except FileNotFoundError:
            return False
        return True

    def _write_current_inventory_initialization(self) -> None:
        """Publish signed v3 provenance before creating any empty heads."""
        record = {
            "families": list(_INVENTORY_FAMILIES),
            "signature": sign_hmac_sha256(
                self._initialization_signing_bytes(), self._initialization_key()
            ),
            "version": _INVENTORY_INITIALIZATION_SCHEMA_VERSION,
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        try:
            self.file_ops.create_bytes_durable_exclusive(
                self._inventory_initialization_locator(), encoded
            )
        except FileExistsError:
            if not self._has_current_inventory_initialization():
                raise CacheBlobBackendError(
                    "Lifecycle inventory initialization changed during creation",
                    context={"operation": "inventory"},
                )

    def initialize_new_store(self) -> None:
        """Durably establish all empty v2 family heads before first evidence.

        A signed v3 provenance record is written only after the compatibility
        proof and before any empty head.  Every evidence append takes this
        same store-level transition first, so no raw/v1 member can race the
        proof or become hidden behind a sibling head.
        """
        with self._conditional_transition("inventory:initialize"):
            initialized = self._has_current_inventory_initialization()
            present_heads = {
                family: self._read_present_inventory_head(family)
                for family in _INVENTORY_FAMILIES
            }
            if not initialized:
                # There is no safe interpretation for old marker-only or
                # lazy-head states.  They did not bind one all-family proof;
                # retain them for the explicit migration workflow instead of
                # inferring the missing family's emptiness from a sibling.
                if self._has_legacy_initialization_marker() or any(
                    present_heads.values()
                ):
                    raise CacheBlobMigrationRequiredError(
                        "Lifecycle inventory initialization requires explicit migration",
                        context={"operation": "inventory_migration"},
                    )
                self._assert_fresh_inventory_namespace()
                self._write_current_inventory_initialization()
            for family in _INVENTORY_FAMILIES:
                if present_heads[family] is not None:
                    continue
                state = self._empty_inventory_head()
                encoded = json.dumps(
                    state, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                try:
                    self.file_ops.create_bytes_durable_exclusive(
                        self._inventory_head_locator(family), encoded
                    )
                except FileExistsError:
                    if self._read_present_inventory_head(family) is None:
                        raise CacheBlobBackendError(
                            "Lifecycle inventory head disappeared during initialization",
                            context={"operation": "inventory", "family": family},
                        )

    def _read_inventory(self, family: str) -> dict[str, int]:
        """Read one bounded v2 sequence head, never the event history."""
        state = self._read_present_inventory_head(family)
        if state is not None:
            return state
        if self._has_current_inventory_initialization():
            return self._empty_inventory_head()
        if self._has_legacy_initialization_marker() or any(
            self._read_present_inventory_head(sibling) is not None
            for sibling in _INVENTORY_FAMILIES
            if sibling != family
        ):
            raise CacheBlobMigrationRequiredError(
                "Lifecycle inventory requires explicit migration before a missing family can be read",
                context={"family": family, "operation": "inventory_migration"},
            )
        # A v1 history cannot be safely reinterpreted as a v2 sparse sequence,
        # and a raw legacy record must never look like no debt.  Only a store
        # with neither the fresh marker nor another validated v2 head reaches
        # this bounded compatibility proof.
        try:
            legacy_size = self.file_ops.get_size(self._legacy_inventory_locator(family))
        except FileNotFoundError:
            legacy_size = -1
        if legacy_size < 0:
            if self._has_preindex_evidence(family):
                raise CacheBlobMigrationRequiredError(
                    "Lifecycle evidence predates its durable inventory",
                    context={"family": family, "operation": "inventory_migration"},
                )
            return self._empty_inventory_head()
        raise CacheBlobMigrationRequiredError(
            "Lifecycle inventory v1 requires an explicit migration",
            context={"family": family, "operation": "inventory_migration"},
        )

    def _bootstrap_pending_inventory(self) -> dict[str, int]:
        """Safely index one bounded legacy pending namespace exactly once.

        Pending candidates are non-authoritative scheduling residue, but their
        digest-bound names and exact bytes can be indexed without guessing
        ownership.  A namespace larger than the explicit inspection budget is
        intentionally migration-required rather than scanned piecemeal through
        a mutable lexical cursor.
        """
        operations_directory = resolve_managed_locator(
            self.file_ops.root,
            "operations",
            operation="bootstrap_pending_inventory",
            allow_missing_leaf=True,
        )
        with self._conditional_transition("inventory:pending"):
            try:
                existing = self.file_ops.read_bytes_bounded(
                    self._inventory_head_locator("pending"),
                    max_bytes=_INVENTORY_HEAD_MAX_BYTES,
                )
            except FileNotFoundError:
                existing = None
            if existing is not None:
                return self._read_inventory("pending")
            try:
                names, next_name = self.file_ops.list_directory_names_bounded(
                    operations_directory,
                    cursor=None,
                    max_names=self.lifecycle_limits.max_inventory_items,
                    max_inventory_names=self.lifecycle_limits.max_inventory_items,
                    max_scanned_names=self.lifecycle_limits.max_inventory_items,
                    operation="bootstrap_pending_inventory",
                    name_filter=self._is_eligible_pending_name,
                )
            except CacheBlobBackendError as exc:
                raise CacheBlobMigrationRequiredError(
                    "Legacy pending controls exceed the safe bootstrap bound",
                    context={"family": "pending", "operation": "inventory_migration"},
                ) from exc
            if next_name is not None:
                raise CacheBlobMigrationRequiredError(
                    "Legacy pending controls require an explicit migration",
                    context={"family": "pending", "operation": "inventory_migration"},
                )
            state = self._empty_inventory_head()
            for name in names:
                raw = self._get_pending_control_raw(name)
                if raw is None:
                    continue
                sequence = state["next_sequence"]
                encoded = json.dumps(
                    {
                        "version": _INVENTORY_SCHEMA_VERSION,
                        "name": name,
                        "digest": hashlib.sha256(raw).hexdigest(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                self.file_ops.create_bytes_durable_exclusive(
                    self._inventory_event_locator("pending", sequence), encoded
                )
                state["next_sequence"] = sequence + 1
            self._write_inventory_head("pending", state)
            return state

    def _write_inventory_head(self, family: str, state: dict[str, int]) -> None:
        """Durably publish the small sequence head after one event transition."""
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > _INVENTORY_HEAD_MAX_BYTES:
            raise AssertionError("lifecycle inventory head unexpectedly exceeds bound")
        self.file_ops.write_bytes_durable(self._inventory_head_locator(family), encoded)

    def _read_inventory_event(self, family: str, sequence: int) -> tuple[str, str] | None:
        """Return one exact immutable scheduling event, or a compacted gap."""
        try:
            raw = self.file_ops.read_bytes_bounded(
                self._inventory_event_locator(family, sequence),
                max_bytes=self._inventory_event_max_bytes(),
            )
        except FileNotFoundError:
            return None
        except ValueError as exc:
            # A bounded event cannot be silently treated as a sparse gap:
            # that would make remaining lifecycle debt look terminal.  Use
            # the same stable bounds reason as other control-evidence reads,
            # while retaining enough inventory provenance for callers to
            # remediate the exact family/sequence.
            raise CacheManifestIntegrityError(
                "Lifecycle inventory event exceeds the configured byte limit",
                context={
                    "operation": "read_inventory_event",
                    "family": family,
                    "sequence": sequence,
                },
                reason=CacheReason.MANIFEST_BOUNDS,
            ) from exc
        except OSError as exc:
            # Only FileNotFoundError is an intentional compacted sparse gap.
            # Other filesystem failures are observable storage faults, not
            # evidence that the immutable scheduling member is absent.
            raise CacheBlobBackendError(
                "Lifecycle inventory event could not be read",
                context={
                    "operation": "read_inventory_event",
                    "family": family,
                    "sequence": sequence,
                },
            ) from exc
        try:
            event = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CacheBlobBackendError(
                "Lifecycle inventory event is invalid",
                context={"operation": "inventory", "family": family, "sequence": sequence},
            ) from exc
        if (
            not isinstance(event, dict)
            or set(event) != {"version", "name", "digest"}
            or event["version"] != _INVENTORY_SCHEMA_VERSION
            or not isinstance(event["name"], str)
            or not event["name"]
            or len(event["name"].encode("utf-8")) > self.lifecycle_limits.max_operation_field_bytes
            or not isinstance(event["digest"], str)
            or len(event["digest"]) != 64
            or any(character not in "0123456789abcdef" for character in event["digest"])
        ):
            raise CacheBlobBackendError(
                "Lifecycle inventory event is invalid",
                context={"operation": "inventory", "family": family, "sequence": sequence},
            )
        return event["name"], event["digest"]

    @staticmethod
    def _schedule_inventory_maintenance(state: dict[str, int]) -> None:
        """Charge all allocated positions through one completed lifecycle boundary."""
        target = state["next_sequence"] - 1
        if target == 0:
            return
        if state["maintenance_target_sequence"] == 0:
            state["compact_next_sequence"] = state["first_live_sequence"]
        state["maintenance_target_sequence"] = max(
            state["maintenance_target_sequence"], target
        )

    def _compact_inventory_window(self, family: str, state: dict[str, int]) -> bool:
        """Advance one exact caller-bounded compaction continuation.

        Removing a stale event leaves a sparse sequence gap.  Resume cursors
        retain original sequence numbers, so old authenticated tokens simply
        inspect the gap and move forward; no live snapshot member is skipped.
        """
        target = state["maintenance_target_sequence"]
        if target == 0:
            return True
        # A durable empty-floor proof is terminal maintenance state.
        if state["first_live_sequence"] == state["next_sequence"]:
            state["compact_next_sequence"] = state["next_sequence"]
            state["maintenance_target_sequence"] = 0
            return True
        start = max(state["compact_next_sequence"], state["first_live_sequence"])
        stop = min(
            target + 1,
            start + self.lifecycle_limits.max_inventory_items,
        )
        reader: Callable[[str], bytes | None]
        if family == "primary":
            reader = self.get_raw
        elif family == "sidecar":
            reader = self.get_reconciliation_checkpoint_raw
        else:
            reader = self._get_pending_control_raw
        first_remaining: int | None = None
        for sequence in range(start, stop):
            event = self._read_inventory_event(family, sequence)
            if event is None:
                continue
            name, digest = event
            current = reader(name)
            if current is None or hashlib.sha256(current).hexdigest() != digest:
                try:
                    self.file_ops.delete_durable(self._inventory_event_locator(family, sequence))
                except FileNotFoundError:
                    pass
            elif first_remaining is None:
                first_remaining = sequence
        # The compacting cursor can wrap before a previously established live
        # floor.  Advancing only when ``start == floor`` strands that floor
        # after it is retired in a later window.  Any window that covers the
        # floor has exactly revalidated the contiguous subrange beginning
        # there, so it can safely move the monotonic lower bound forward.
        if start <= state["first_live_sequence"] < stop:
            if first_remaining is not None:
                state["first_live_sequence"] = first_remaining
            else:
                # All positions through ``stop - 1`` are durable sparse gaps.
                # A cursor may start at ``stop`` even when the next event has
                # not been inspected yet; it cannot skip a live member.
                state["first_live_sequence"] = stop
        state["compact_next_sequence"] = stop
        complete = stop > target
        if complete:
            state["maintenance_target_sequence"] = 0
        return complete

    def _append_inventory_event(self, family: str, name: str, raw: bytes) -> None:
        """Record membership before publishing the corresponding control file.

        A crash can leave a harmless index-only stale position, but can never
        leave a published evidence record absent from a snapshot index.  The
        index is not authority: every page re-reads and compares the recorded
        digest before yielding bytes to lifecycle code.
        """
        if not isinstance(raw, bytes) or not raw:
            raise TypeError("Lifecycle inventory events require non-empty bytes")
        # Publish the signed all-family provenance and every empty head before
        # accepting the first member of any one family.  Direct repository
        # callers use this path too; without it a lone sidecar/pending event
        # could create an unauthenticated lazy head that later readers must
        # treat as migration evidence.
        self.initialize_new_store()
        # The outer store-level lease is intentionally held across the
        # proof/provenance/head transition as well as all future family
        # publication.  Without it, a legacy raw member could appear between
        # the all-family scan and marker publication and be hidden forever by
        # an otherwise-valid sibling head.
        with self._conditional_transition("inventory:initialize"):
            with self._conditional_transition(f"inventory:{family}"):
                while True:
                    state = self._read_inventory(family)
                    sequence = state["next_sequence"]
                    encoded = json.dumps(
                        {
                            "version": _INVENTORY_SCHEMA_VERSION,
                            "name": name,
                            "digest": hashlib.sha256(raw).hexdigest(),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    if len(encoded) > self._inventory_event_max_bytes():
                        raise CacheBlobBackendError(
                            "Lifecycle inventory event exceeds its bounded field policy",
                            context={"operation": "inventory", "family": family},
                        )
                    try:
                        self.file_ops.create_bytes_durable_exclusive(
                            self._inventory_event_locator(family, sequence), encoded
                        )
                    except FileExistsError:
                        # A process loss can leave a durable event before its head
                        # acknowledgement.  It is safe non-authoritative stale
                        # scheduling membership; acknowledge the position and
                        # allocate the next one without reading/re-writing history.
                        if self._read_inventory_event(family, sequence) is None:
                            raise CacheBlobBackendError(
                                "Lifecycle inventory event disappeared during recovery",
                                context={"operation": "inventory", "family": family},
                            )
                        state["next_sequence"] = sequence + 1
                        self._write_inventory_head(family, state)
                        continue
                    state["next_sequence"] = sequence + 1
                    self._write_inventory_head(family, state)
                    return

    def _compact_inventory_after_retirement(self, family: str) -> bool:
        """Charge and advance bounded maintenance after a completed action.

        Starting a reconciliation action must not delete unrelated scheduling
        entries before its destructive payload transition.  Retiring an exact
        primary/sidecar is a completed lifecycle boundary, so it is safe to
        make one bounded compaction step there without changing normal reads
        or pre-action fault ordering.
        """
        with self._conditional_transition(f"inventory:{family}"):
            state = self._read_inventory(family)
            self._schedule_inventory_maintenance(state)
            complete = self._compact_inventory_window(family, state)
            self._write_inventory_head(family, state)
            return complete

    def compact_inventory_for_recovery(self) -> bool:
        """Advance one persisted family continuation and report clear readiness.

        This path is called only by lifecycle recovery or aggregate clear
        admission.  It inspects no more than one ``max_inventory_items``
        window per invocation, and reads remain entirely non-mutating.
        """
        # Constructor recovery can run before any lifecycle mutation has
        # established signed all-family provenance.  Maintenance must neither
        # invoke the key provider nor classify a missing sibling head in that
        # state: ordinary recovery retains responsibility for the later typed
        # legacy-evidence decision.  A present marker is then authenticated by
        # the normal head reads below.
        try:
            initialized_size = self.file_ops.get_size(
                self._inventory_initialization_locator()
            )
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle inventory initialization could not be inspected",
                context={"operation": "inventory"},
            ) from exc
        if initialized_size < 0:
            return True

        advanced = False
        for family in _INVENTORY_FAMILIES:
            with self._conditional_transition(f"inventory:{family}"):
                state = self._read_inventory(family)
                if state["maintenance_target_sequence"] == 0:
                    continue
                if advanced:
                    # Head reads are bounded; event inspection is not.  Leave
                    # the next family for a later recovery call rather than
                    # multiplying the caller's hard work limit by family.
                    return False
                complete = self._compact_inventory_window(family, state)
                self._write_inventory_head(family, state)
                advanced = True
                if not complete:
                    return False
        return True

    def _inventory_page(
        self,
        family: str,
        cursor: OperationCursor | ReconciliationCheckpointCursor | PendingControlCursor | None,
        *,
        page_size: int,
        max_inspections: int | None = None,
        read_current: Callable[[str], bytes | None],
        cursor_type: type[OperationCursor] | type[ReconciliationCheckpointCursor] | type[PendingControlCursor],
    ) -> tuple[
        tuple[tuple[str, bytes | None], ...], object | None, tuple[object, ...], int
    ]:
        """Read at most one stable high-water page from one family index."""
        state = self._read_inventory(family)
        high_water = (
            state["next_sequence"] - 1
            if cursor is None or cursor.snapshot_high_water is None
            else cursor.snapshot_high_water
        )
        requested_position = (
            1 if cursor is None or cursor.next_sequence is None else cursor.next_sequence
        )
        # Compacting a stale event never changes the immutable high-water or
        # an authenticated cursor's next position.  It only records a proven
        # lower bound, which prevents a fresh recovery from paying lifetime
        # cost for empty scheduling history.
        inspection_limit = (
            self.lifecycle_limits.max_inventory_items
            if max_inspections is None
            else max_inspections
        )
        if (
            type(inspection_limit) is not int
            or inspection_limit <= 0
            or inspection_limit > self.lifecycle_limits.max_inventory_items
        ):
            raise ValueError("inventory inspection limit exceeds configured lifecycle limit")
        position = max(requested_position, state["first_live_sequence"])
        inspected = 0
        entries: list[tuple[str, bytes | None]] = []
        entry_next_cursors: list[object] = []
        if cursor is not None and hasattr(cursor, "operation_id"):
            last_name = cursor.operation_id
        elif cursor is not None:
            last_name = cursor.name
        elif cursor_type is OperationCursor:
            # Primary cursors validate their compatibility projection as a
            # canonical operation ID.  An empty sparse window can be
            # nonterminal without having inspected a real name, so use the
            # valid before-first projection rather than the sidecar/pending
            # ``~`` scheduling sentinel.
            last_name = OperationCursor.before_first(high_water).operation_id
        else:
            last_name = "~"
        while (
            position <= high_water
            and inspected < inspection_limit
            and len(entries) < page_size
        ):
            event = self._read_inventory_event(family, position)
            position += 1
            inspected += 1
            if event is None:
                continue
            name, digest = event
            last_name = name
            # A missing record or digest mismatch makes this immutable
            # scheduling event stale.  A typed bounded-read, decode, or
            # integrity failure is not absence and must never be silently
            # converted into a clean-looking terminal page.
            current = read_current(name)
            if current is not None and hashlib.sha256(current).hexdigest() == digest:
                entries.append((name, current))
                # ``position`` already includes all stale events inspected
                # before and including this member.  Bind that exact source
                # position to the yielded entry, rather than deriving a
                # lexical name later in the reconciler.
                entry_next_cursors.append(
                    cursor_type(
                        name,
                        snapshot_high_water=high_water,
                        next_sequence=position,
                    )
                )
        next_cursor = None
        if position <= high_water:
            next_cursor = cursor_type(
                last_name,
                snapshot_high_water=high_water,
                next_sequence=position,
            )
        return tuple(entries), next_cursor, tuple(entry_next_cursors), inspected

    def locator_for(self, operation_id: str) -> Path:
        """Derive a contained locator from an opaque operation identifier."""
        safe_operation_id = validate_blob_id(operation_id)
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / f"{safe_operation_id}.json",
            operation="operation_record",
            allow_missing_leaf=True,
        )

    def _clear_target_locator(
        self, operation_id: str, page_id: str, *, checkpoint: bool
    ) -> Path:
        """Derive one contained deterministic clear page/checkpoint locator."""
        safe_operation_id = validate_blob_id(operation_id)
        safe_page_id = validate_blob_id(page_id)
        kind = "checkpoint" if checkpoint else "page"
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations")
            / f"clear-target-{kind}-{safe_operation_id}-{safe_page_id}.json",
            operation=f"clear_target_{kind}",
            allow_missing_leaf=True,
        )

    def _clear_target_reference_locator(
        self, operation_id: str, reference: str
    ) -> Path:
        """Derive a contained immutable exact-manifest sidecar locator."""
        safe_operation_id = validate_blob_id(operation_id)
        safe_reference = validate_blob_id(reference)
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations")
            / f"clear-target-reference-{safe_operation_id}-{safe_reference}.json",
            operation="clear_target_reference",
            allow_missing_leaf=True,
        )

    def _clear_target_reference_chunk_locator(
        self, operation_id: str, reference: str, chunk_index: int
    ) -> Path:
        """Derive one ordered bounded chunk of exact manifest evidence."""
        safe_operation_id = validate_blob_id(operation_id)
        safe_reference = validate_blob_id(reference)
        if type(chunk_index) is not int or not 0 <= chunk_index < MAX_CLEAR_TARGET_REFERENCE_CHUNKS:
            raise ValueError("clear target reference chunk index is invalid")
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations")
            / (
                "clear-target-reference-"
                f"{safe_operation_id}-{safe_reference}-part-{chunk_index:04x}.json"
            ),
            operation="clear_target_reference_chunk",
            allow_missing_leaf=True,
        )

    def clear_target_page_locator(self, operation_id: str, page_id: str) -> Path:
        """Return the exact contained locator for one target page."""
        return self._clear_target_locator(operation_id, page_id, checkpoint=False)

    def clear_target_checkpoint_locator(self, operation_id: str, page_id: str) -> Path:
        """Return the exact contained locator for one target checkpoint."""
        return self._clear_target_locator(operation_id, page_id, checkpoint=True)

    def clear_target_reference_locator(self, operation_id: str, reference: str) -> Path:
        """Return the exact contained locator for one referenced manifest copy."""
        return self._clear_target_reference_locator(operation_id, reference)

    def clear_target_reference_chunk_locator(
        self, operation_id: str, reference: str, chunk_index: int
    ) -> Path:
        """Return the deterministic contained locator for one reference chunk."""
        return self._clear_target_reference_chunk_locator(
            operation_id, reference, chunk_index
        )

    def reconciliation_checkpoint_locator(self, operation_id: str) -> Path:
        """Return a private sidecar used to resume one reconciliation action.

        The name deliberately cannot satisfy the 32-hex operation inventory
        grammar, so it never becomes lifecycle evidence or consumes a normal
        recovery page slot.
        """
        safe_operation_id = validate_blob_id(operation_id)
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / f"reconcile-action-{safe_operation_id}.json",
            operation="reconciliation_checkpoint",
            allow_missing_leaf=True,
        )

    def _pending_recovery_cursor_locator(self) -> Path:
        """Return private, non-authoritative progress for pending control scans."""
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".pending-recovery.cursor",
            operation="pending_recovery_cursor",
            allow_missing_leaf=True,
        )

    def _get_pending_control_raw(self, name: str) -> bytes | None:
        """Read one indexed pending candidate without pathname discovery."""
        if not self._is_eligible_pending_name(name):
            return None
        locator = resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / name,
            operation="read_pending_control",
            allow_missing_leaf=True,
        )
        try:
            return self.file_ops.read_bytes_bounded(
                locator, max_bytes=self.lifecycle_limits.max_operation_record_bytes
            )
        except FileNotFoundError:
            return None
        except ValueError as exc:
            raise CacheManifestIntegrityError(
                "Pending lifecycle control exceeds the configured byte limit",
                reason=CacheReason.MANIFEST_BOUNDS,
            ) from exc
        except OSError as exc:
            raise CacheBlobBackendError(
                "Pending lifecycle control could not be read",
                context={"operation": "read_pending_control"},
            ) from exc

    def _compact_consumed_pending_event(
        self, name: str, source_cursor: PendingControlCursor | None
    ) -> None:
        """Retire one exactly indexed consumed pending candidate when stale.

        ``source_cursor`` is the page's authenticated post-entry position, so
        its preceding sequence is the only event this method can touch.  The
        inventory lock serializes that compare/delete with concurrent appends;
        sparse sequence numbers remain stable for every existing cursor.
        """
        if (
            source_cursor is None
            or source_cursor.snapshot_high_water is None
            or source_cursor.next_sequence is None
            or source_cursor.next_sequence <= 1
        ):
            return
        sequence = source_cursor.next_sequence - 1
        with self._conditional_transition("inventory:pending"):
            state = self._read_inventory("pending")
            event = self._read_inventory_event("pending", sequence)
            if event is None or event[0] != name:
                return
            event_name, digest = event
            current = self._get_pending_control_raw(event_name)
            if current is not None and hashlib.sha256(current).hexdigest() == digest:
                return
            try:
                self.file_ops.delete_durable(
                    self._inventory_event_locator("pending", sequence)
                )
            except FileNotFoundError:
                return
            if state["first_live_sequence"] == sequence:
                # Only the verified sequence is skipped; the following event
                # remains subject to ordinary exact revalidation on its page.
                state["first_live_sequence"] = sequence + 1
            self._write_inventory_head("pending", state)

    def _pending_recovery_cursor(self) -> PendingControlCursor | None:
        """Read bounded scheduler progress; malformed progress safely restarts."""
        try:
            raw = self.file_ops.read_bytes_bounded(
                self._pending_recovery_cursor_locator(),
                max_bytes=self.lifecycle_limits.max_operation_field_bytes,
            )
        except (FileNotFoundError, OSError, ValueError):
            return None
        try:
            value = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            # v1 persisted only a lexical name. It remains scheduling-only and
            # is conservatively readable, but new checkpoints retain the exact
            # high-water sequence position.
            try:
                return PendingControlCursor(raw.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                return None
        try:
            if not isinstance(value, dict) or set(value) != {
                "version", "name", "snapshot_high_water", "next_sequence"
            } or value["version"] != _INVENTORY_SCHEMA_VERSION:
                return None
            return PendingControlCursor(
                value["name"],
                snapshot_high_water=value["snapshot_high_water"],
                next_sequence=value["next_sequence"],
            )
        except (TypeError, ValueError):
            return None

    def _checkpoint_pending_recovery_cursor(
        self, cursor: PendingControlCursor | None
    ) -> None:
        """Durably advance opaque scan scheduling without granting authority."""
        locator = self._pending_recovery_cursor_locator()
        if cursor is None:
            try:
                self.file_ops.delete_durable(locator)
            except FileNotFoundError:
                pass
            return
        encoded = json.dumps(
            {
                "version": _INVENTORY_SCHEMA_VERSION,
                "name": cursor.name,
                "snapshot_high_water": cursor.snapshot_high_water,
                "next_sequence": cursor.next_sequence,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.file_ops.write_bytes_durable(locator, encoded)

    def get_reconciliation_checkpoint_raw(self, operation_id: str) -> bytes | None:
        """Read opaque reconciliation progress without granting it authority."""
        return self._read_bounded(
            self.reconciliation_checkpoint_locator(operation_id),
            operation="get_reconcile",
        )

    def list_reconciliation_checkpoint_page(
        self,
        cursor: ReconciliationCheckpointCursor | None = None,
        *,
        page_size: int | None = None,
        max_inspections: int | None = None,
    ) -> ReconciliationCheckpointPage:
        """Return one stable bounded page of reconciliation sidecar bytes.

        Sidecars are opaque scheduling evidence until the reconciler validates
        their exact canonical bytes.  The repository therefore advances past
        malformed candidates instead of repeatedly reading an invalid lexical
        prefix, and reports an unreadable/oversized candidate as ``None``.
        """
        if cursor is not None and not isinstance(cursor, ReconciliationCheckpointCursor):
            raise TypeError("reconciliation checkpoint cursor is invalid")
        limit = self._page_size(page_size)
        records, next_cursor, entry_next_cursors, inspected_positions = self._inventory_page(
            "sidecar",
            cursor,
            page_size=limit,
            max_inspections=max_inspections,
            read_current=self.get_reconciliation_checkpoint_raw,
            cursor_type=ReconciliationCheckpointCursor,
        )
        return ReconciliationCheckpointPage(
            entries=records,
            next_cursor=next_cursor,
            entry_next_cursors=entry_next_cursors,
            inspected_positions=inspected_positions,
        )

    @staticmethod
    def _checkpoint_cursor_name(value: str) -> str:
        """Map legacy bare IDs to the one full-filename cursor namespace."""
        if FileOperationRecordRepository._is_hex_identifier(value):
            return f"reconcile-action-{value}.json"
        return value

    def list_reconciliation_checkpoint_raws(self) -> tuple[tuple[str, bytes], ...]:
        """Compatibility iterator composed from bounded sidecar pages."""
        cursor: ReconciliationCheckpointCursor | None = None
        records: list[tuple[str, bytes]] = []
        while True:
            page = self.list_reconciliation_checkpoint_page(cursor)
            records.extend(
                (operation_id, raw)
                for operation_id, raw in page.entries
                if raw is not None
            )
            if page.next_cursor is None:
                return tuple(records)
            cursor = page.next_cursor

    def create_reconciliation_checkpoint_exclusive(
        self, operation_id: str, raw_record: bytes
    ) -> Path:
        """Durably persist action intent before a reconciler mutates storage."""
        if not isinstance(raw_record, bytes) or not raw_record:
            raise TypeError("Reconciliation checkpoints require non-empty bytes")
        try:
            self._append_inventory_event("sidecar", operation_id, raw_record)
            return self.file_ops.create_bytes_durable_exclusive(
                self.reconciliation_checkpoint_locator(operation_id), raw_record
            )
        except FileExistsError as exc:
            raise CacheBlobLifecycleConflictError(
                "Reconciliation checkpoint already exists",
                context={"operation_id": operation_id, "operation": "create_reconcile"},
            ) from exc
        except OSError as exc:
            raise CacheBlobBackendError(
                "Reconciliation checkpoint could not be created",
                context={"operation_id": operation_id, "operation": "create_reconcile"},
            ) from exc

    def checkpoint_reconciliation_if_exact(
        self,
        operation_id: str,
        *,
        expected_raw: bytes,
        raw_record: bytes,
    ) -> None:
        """Advance one reconciliation checkpoint only from exact bytes."""
        if not isinstance(expected_raw, bytes) or not isinstance(raw_record, bytes):
            raise TypeError("Reconciliation checkpoints require exact bytes")
        try:
            with self._conditional_transition(
                f"reconcile:{operation_id}", operation_id=operation_id
            ):
                current = self.get_reconciliation_checkpoint_raw(operation_id)
                if current != expected_raw:
                    raise CacheBlobLifecycleConflictError(
                        "Reconciliation checkpoint no longer matches",
                        context={
                            "operation_id": operation_id,
                            "operation": "checkpoint_reconcile",
                        },
                    )
                self._append_inventory_event("sidecar", operation_id, raw_record)
                self.file_ops.write_bytes_durable(
                    self.reconciliation_checkpoint_locator(operation_id), raw_record
                )
        except CacheBlobLifecycleConflictError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Reconciliation checkpoint could not be persisted",
                context={"operation_id": operation_id, "operation": "checkpoint_reconcile"},
            ) from exc

    def retire_reconciliation_checkpoint_if_exact(
        self, operation_id: str, *, expected_raw: bytes
    ) -> None:
        """Remove private progress only after the exact completed bytes remain."""
        try:
            with self._conditional_transition(
                f"reconcile:{operation_id}", operation_id=operation_id
            ):
                if self.get_reconciliation_checkpoint_raw(operation_id) != expected_raw:
                    raise CacheBlobLifecycleConflictError(
                        "Reconciliation checkpoint no longer matches",
                        context={
                            "operation_id": operation_id,
                            "operation": "retire_reconcile",
                        },
                    )
                self.file_ops.delete_durable(
                    self.reconciliation_checkpoint_locator(operation_id)
                )
            self._compact_inventory_after_retirement("sidecar")
            # The durable write protocol records its private pending control
            # before every rename.  At this completed sidecar boundary those
            # candidates are no longer live authority, so amortize their
            # exact revalidation here as well instead of deferring ordinary
            # successful traffic to a later recovery call.
            self._compact_inventory_after_retirement("pending")
        except CacheBlobLifecycleConflictError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Reconciliation checkpoint could not be retired",
                context={"operation_id": operation_id, "operation": "retire_reconcile"},
            ) from exc

    def _create_clear_target_exclusive(
        self,
        operation_id: str,
        page_id: str,
        raw_record: bytes,
        *,
        checkpoint: bool,
    ) -> Path:
        """Create one durable clear control record without replacing evidence."""
        if not isinstance(raw_record, bytes) or not raw_record:
            raise TypeError("Clear target evidence must be non-empty bytes")
        locator = self._clear_target_locator(
            operation_id, page_id, checkpoint=checkpoint
        )
        try:
            return self.file_ops.create_bytes_durable_exclusive(locator, raw_record)
        except FileExistsError as exc:
            raise CacheBlobLifecycleConflictError(
                "Clear target evidence already exists",
                context={
                    "operation_id": operation_id,
                    "page_id": page_id,
                    "operation": "create_clear_target",
                },
            ) from exc
        except OSError as exc:
            raise CacheBlobBackendError(
                "Clear target evidence could not be created",
                context={
                    "operation_id": operation_id,
                    "page_id": page_id,
                    "operation": "create_clear_target",
                },
            ) from exc

    def create_clear_target_page_exclusive(
        self, operation_id: str, page_id: str, raw_record: bytes
    ) -> Path:
        """Durably save exact page targets before their source cursor advances."""
        return self._create_clear_target_exclusive(
            operation_id, page_id, raw_record, checkpoint=False
        )

    def create_clear_target_checkpoint_exclusive(
        self, operation_id: str, page_id: str, raw_record: bytes
    ) -> Path:
        """Durably save zero-progress state before a page can delete targets."""
        return self._create_clear_target_exclusive(
            operation_id, page_id, raw_record, checkpoint=True
        )

    def create_clear_target_reference_exclusive(
        self, operation_id: str, reference: str, raw_record: bytes
    ) -> Path:
        """Persist exact oversized target evidence without overwriting it."""
        if not isinstance(raw_record, bytes) or not raw_record:
            raise TypeError("Clear target reference requires non-empty bytes")
        if len(raw_record) > self.lifecycle_limits.max_operation_record_bytes:
            raise CacheManifestIntegrityError(
                "Clear target reference exceeds the configured byte limit",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        try:
            return self.file_ops.create_bytes_durable_exclusive(
                self._clear_target_reference_locator(operation_id, reference), raw_record
            )
        except FileExistsError as exc:
            raise CacheBlobLifecycleConflictError(
                "Clear target reference already exists",
                context={
                    "operation_id": operation_id,
                    "reference": reference,
                    "operation": "create_clear_target_reference",
                },
            ) from exc
        except OSError as exc:
            raise CacheBlobBackendError(
                "Clear target reference could not be created",
                context={
                    "operation_id": operation_id,
                    "reference": reference,
                    "operation": "create_clear_target_reference",
                },
            ) from exc

    def create_clear_target_reference_chunks_exclusive(
        self, operation_id: str, reference: str, raw_record: bytes
    ) -> int:
        """Persist bounded immutable chunks before a signed page resolves them.

        The caller must first prove the eventual page/control contract fits. A
        crash after a page is durable but before every chunk is written remains
        recoverable: pre-authority recovery authenticates and retires the page
        without using any target as deletion authority.
        """
        if not isinstance(raw_record, bytes) or not raw_record:
            raise TypeError("Clear target reference requires non-empty bytes")
        if len(raw_record) > MAX_MANIFEST_BYTES:
            raise CacheManifestIntegrityError(
                "Clear target reference exceeds the manifest byte limit",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        chunk_size = self.lifecycle_limits.max_operation_record_bytes
        chunk_count = (len(raw_record) + chunk_size - 1) // chunk_size
        if chunk_count > MAX_CLEAR_TARGET_REFERENCE_CHUNKS:
            raise CacheManifestIntegrityError(
                "Clear target reference needs too many bounded chunks",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        for chunk_index in range(chunk_count):
            chunk = raw_record[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
            locator = self._clear_target_reference_chunk_locator(
                operation_id, reference, chunk_index
            )
            try:
                self.file_ops.create_bytes_durable_exclusive(locator, chunk)
            except FileExistsError:
                existing = self._read_bounded(
                    locator, operation="create_clear_target_reference_chunk"
                )
                if existing != chunk:
                    raise CacheBlobLifecycleConflictError(
                        "Clear target reference chunk already exists",
                        context={
                            "operation_id": operation_id,
                            "reference": reference,
                            "chunk_index": str(chunk_index),
                            "operation": "create_clear_target_reference_chunk",
                        },
                    )
            except OSError as exc:
                raise CacheBlobBackendError(
                    "Clear target reference chunk could not be created",
                    context={
                        "operation_id": operation_id,
                        "reference": reference,
                        "chunk_index": str(chunk_index),
                        "operation": "create_clear_target_reference_chunk",
                    },
                ) from exc
        return chunk_count

    def _get_clear_target_raw(
        self, operation_id: str, page_id: str, *, checkpoint: bool
    ) -> bytes | None:
        """Load opaque exact clear evidence without declaring it authoritative."""
        return self._read_bounded(
            self._clear_target_locator(operation_id, page_id, checkpoint=checkpoint),
            operation="get_clear_target",
        )

    def get_clear_target_page_raw(
        self, operation_id: str, page_id: str
    ) -> bytes | None:
        """Read one exact target page without parsing or authenticating it."""
        return self._get_clear_target_raw(operation_id, page_id, checkpoint=False)

    def get_clear_target_checkpoint_raw(
        self, operation_id: str, page_id: str
    ) -> bytes | None:
        """Read one exact target checkpoint without parsing or authenticating it."""
        return self._get_clear_target_raw(operation_id, page_id, checkpoint=True)

    def get_clear_target_reference_raw(
        self,
        operation_id: str,
        reference: str,
        *,
        chunk_count: int | None = None,
        byte_length: int | None = None,
        chunk_digests: tuple[str, ...] | None = None,
    ) -> bytes | None:
        """Read exact target evidence under its signed bounded chunk contract.

        Legacy single-record references remain readable under the caller's
        limit. New references supply both count and length in the signed page;
        those values are validated before allocation or any chunk read.
        """
        if chunk_count is None and byte_length is None and chunk_digests is None:
            return self._read_bounded(
                self._clear_target_reference_locator(operation_id, reference),
                operation="get_clear_target_reference",
            )
        chunk_count, byte_length, chunk_digests = self._reference_chunk_contract(
            chunk_count, byte_length, chunk_digests
        )

        resolved = bytearray()
        for chunk_index in range(chunk_count):
            chunk = self._read_bounded(
                self._clear_target_reference_chunk_locator(
                    operation_id, reference, chunk_index
                ),
                operation="get_clear_target_reference_chunk",
            )
            if chunk is None:
                return None
            if (
                not chunk
                or hashlib.sha256(chunk).hexdigest() != chunk_digests[chunk_index]
                or len(resolved) + len(chunk) > byte_length
            ):
                raise CacheManifestIntegrityError(
                    "Clear target reference chunks do not match their signed length",
                    reason=CacheReason.MANIFEST_BOUNDS,
                )
            resolved.extend(chunk)
        if len(resolved) != byte_length:
            raise CacheManifestIntegrityError(
                "Clear target reference chunks do not match their signed length",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        return bytes(resolved)

    def _reference_chunk_contract(
        self,
        chunk_count: int | None,
        byte_length: int | None,
        chunk_digests: tuple[str, ...] | None,
    ) -> tuple[int, int, tuple[str, ...]]:
        """Validate signed reference bounds before reading or allocating chunks."""
        if (
            type(chunk_count) is not int
            or type(byte_length) is not int
            or not isinstance(chunk_digests, tuple)
            or not 0 < chunk_count <= MAX_CLEAR_TARGET_REFERENCE_CHUNKS
            or not 0 < byte_length <= MAX_MANIFEST_BYTES
            or len(chunk_digests) != chunk_count
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in chunk_digests
            )
        ):
            raise CacheManifestIntegrityError(
                "Clear target reference chunk contract is invalid",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        maximum = self.lifecycle_limits.max_operation_record_bytes
        if byte_length < chunk_count or byte_length > chunk_count * maximum:
            raise CacheManifestIntegrityError(
                "Clear target reference chunk contract exceeds configured bounds",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        return chunk_count, byte_length, chunk_digests

    def retire_clear_target_reference_chunks_if_bound(
        self,
        operation_id: str,
        reference: str,
        *,
        chunk_count: int | None,
        byte_length: int | None,
        chunk_digests: tuple[str, ...] | None,
    ) -> None:
        """Retire each page-bound chunk that still matches its signed digest.

        This is reserved for aborting an unauthoritative prepared inventory.
        A missing chunk is normal after an interrupted write; a mismatching
        chunk remains untouched rather than being inferred from its filename.
        """
        chunk_count, _byte_length, chunk_digests = self._reference_chunk_contract(
            chunk_count, byte_length, chunk_digests
        )
        with self._conditional_transition(
            f"clear-reference:{operation_id}:{reference}", operation_id=operation_id
        ):
            for chunk_index in range(chunk_count):
                locator = self._clear_target_reference_chunk_locator(
                    operation_id, reference, chunk_index
                )
                current = self._read_bounded(
                    locator, operation="abort_clear_target_reference_chunk"
                )
                if current is None:
                    continue
                if hashlib.sha256(current).hexdigest() != chunk_digests[chunk_index]:
                    raise CacheManifestIntegrityError(
                        "Clear target reference chunk no longer matches signed evidence"
                    )
                self.file_ops.delete_durable(locator)

    def _retire_locator_if_exact(
        self,
        locator: Path,
        *,
        transition_id: str,
        expected_raw: bytes,
        context: dict[str, str],
    ) -> bool:
        """Delete one control artifact only after exact-byte revalidation.

        A missing artifact is an idempotent completed retirement.  Any other
        bytes remain a typed conflict, never an invitation to delete by name.
        """
        if not isinstance(expected_raw, bytes):
            raise TypeError("Control artifact retirement requires exact bytes")
        with self._conditional_transition(
            transition_id, operation_id=context["operation_id"]
        ):
            current = self._read_bounded(locator, operation=context["operation"])
            if current is None:
                return False
            if current != expected_raw:
                raise CacheBlobLifecycleConflictError(
                    "Lifecycle control evidence no longer matches", context=context
                )
            self.file_ops.delete_durable(locator)
            return True

    def retire_clear_target_checkpoint_if_exact(
        self, operation_id: str, page_id: str, *, expected_raw: bytes
    ) -> bool:
        """Retire completed checkpoint evidence with exact CAS semantics."""
        return self._retire_locator_if_exact(
            self.clear_target_checkpoint_locator(operation_id, page_id),
            transition_id=f"clear:{operation_id}:{page_id}",
            expected_raw=expected_raw,
            context={
                "operation_id": operation_id,
                "page_id": page_id,
                "operation": "retire_clear_target_checkpoint",
            },
        )

    def retire_clear_target_page_if_exact(
        self, operation_id: str, page_id: str, *, expected_raw: bytes
    ) -> bool:
        """Retire completed target-page evidence with exact CAS semantics."""
        return self._retire_locator_if_exact(
            self.clear_target_page_locator(operation_id, page_id),
            transition_id=f"clear-page:{operation_id}:{page_id}",
            expected_raw=expected_raw,
            context={
                "operation_id": operation_id,
                "page_id": page_id,
                "operation": "retire_clear_target_page",
            },
        )

    def retire_clear_target_reference_if_exact(
        self,
        operation_id: str,
        reference: str,
        *,
        expected_raw: bytes,
        chunk_count: int | None = None,
        byte_length: int | None = None,
        chunk_digests: tuple[str, ...] | None = None,
    ) -> bool:
        """Retire authenticated reference bytes only after exact revalidation."""
        if chunk_count is None and byte_length is None and chunk_digests is None:
            return self._retire_locator_if_exact(
                self.clear_target_reference_locator(operation_id, reference),
                transition_id=f"clear-reference:{operation_id}:{reference}",
                expected_raw=expected_raw,
                context={
                    "operation_id": operation_id,
                    "reference": reference,
                    "operation": "retire_clear_target_reference",
                },
            )
        with self._conditional_transition(
            f"clear-reference:{operation_id}:{reference}", operation_id=operation_id
        ):
            current = self.get_clear_target_reference_raw(
                operation_id,
                reference,
                chunk_count=chunk_count,
                byte_length=byte_length,
                chunk_digests=chunk_digests,
            )
            if current is None:
                return False
            if current != expected_raw:
                raise CacheBlobLifecycleConflictError(
                    "Clear target reference no longer matches",
                    context={
                        "operation_id": operation_id,
                        "reference": reference,
                        "operation": "retire_clear_target_reference",
                    },
                )
            assert chunk_count is not None
            for chunk_index in range(chunk_count):
                self.file_ops.delete_durable(
                    self._clear_target_reference_chunk_locator(
                        operation_id, reference, chunk_index
                    )
                )
            return True

    def iter_clear_target_page_ids(self, operation_id: str) -> Iterator[str]:
        """Yield only this operation's syntactically exact page identifiers.

        Membership is never authority: callers must authenticate every bytes
        record before using the yielded ID to retire anything.  Streaming the
        directory avoids keeping a whole clear snapshot in memory during
        terminal cleanup and makes post-crash retirement resumable.
        """
        safe_operation_id = validate_blob_id(operation_id)
        prefix = f"clear-target-page-{safe_operation_id}-"
        operations_directory = resolve_managed_locator(
            self.file_ops.root,
            "operations",
            operation="list_clear_target_pages",
            allow_missing_leaf=True,
        )
        try:
            for path in operations_directory.iterdir():
                name = path.name
                if not name.startswith(prefix) or not name.endswith(".json"):
                    continue
                page_id = name[len(prefix) : -len(".json")]
                if len(page_id) != 32 or any(
                    character not in "0123456789abcdef" for character in page_id
                ):
                    continue
                yield page_id
        except FileNotFoundError:
            return
        except OSError as exc:
            raise CacheBlobBackendError(
                "Clear target evidence directory could not be listed",
                context={"operation_id": operation_id, "operation": "list_clear_pages"},
            ) from exc

    def checkpoint_clear_target_if_exact(
        self,
        operation_id: str,
        page_id: str,
        *,
        expected_raw: bytes,
        raw_record: bytes,
    ) -> None:
        """Conditionally replace exact progress so stale writers cannot regress it."""
        if not isinstance(expected_raw, bytes) or not isinstance(raw_record, bytes):
            raise TypeError("Clear target checkpoints require exact bytes")
        locator = self.clear_target_checkpoint_locator(operation_id, page_id)
        try:
            with self._conditional_transition(
                f"clear:{operation_id}:{page_id}", operation_id=operation_id
            ):
                current = self.get_clear_target_checkpoint_raw(operation_id, page_id)
                if current != expected_raw:
                    raise CacheBlobLifecycleConflictError(
                        "Clear target checkpoint no longer matches",
                        context={
                            "operation_id": operation_id,
                            "page_id": page_id,
                            "operation": "checkpoint_clear_target_if_exact",
                        },
                    )
                self.file_ops.write_bytes_durable(locator, raw_record)
        except CacheBlobLifecycleConflictError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Clear target checkpoint could not be persisted",
                context={
                    "operation_id": operation_id,
                    "page_id": page_id,
                    "operation": "checkpoint_clear_target_if_exact",
                },
            ) from exc

    def create_exclusive(
        self, record: LifecycleOperationRecord, raw_record: bytes
    ) -> Path:
        """Create evidence exclusively and durably before payload publication."""
        try:
            self._append_inventory_event("primary", record.operation_id, raw_record)
            return self.file_ops.create_bytes_durable_exclusive(
                self.locator_for(record.operation_id), raw_record
            )
        except FileExistsError as exc:
            raise CacheBlobLifecycleConflictError(
                "Lifecycle operation evidence already exists",
                context={"operation_id": record.operation_id, "operation": "create"},
            ) from exc
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence could not be created",
                context={"operation_id": record.operation_id, "operation": "create"},
            ) from exc

    def create(self, record: LifecycleOperationRecord, raw_record: bytes) -> Path:
        """Compatibility alias for explicit exclusive evidence creation."""
        return self.create_exclusive(record, raw_record)

    def get_raw(self, operation_id: str) -> bytes | None:
        """Read exact bytes without assigning evidence any read authority."""
        return self._read_bounded(
            self.locator_for(operation_id), operation="get_raw"
        )

    @staticmethod
    def _is_hex_identifier(value: str) -> bool:
        """Return whether ``value`` is the fixed opaque evidence identifier."""
        return len(value) == 32 and all(character in "0123456789abcdef" for character in value)

    @classmethod
    def _is_primary_filename(cls, name: str) -> bool:
        """Return whether one name exactly matches :meth:`locator_for`."""
        return name.endswith(".json") and cls._is_hex_identifier(name[:-5])

    @classmethod
    def _is_sidecar_filename(cls, name: str) -> bool:
        """Return whether one name is a direct reconciliation sidecar."""
        prefix = "reconcile-action-"
        return (
            name.startswith(prefix)
            and name.endswith(".json")
            and cls._is_hex_identifier(name.removeprefix(prefix)[:-5])
        )

    def _recoverable_pending_final(self, name: str) -> tuple[Path, str | None] | None:
        """Resolve one strictly named lifecycle control final without guessing.

        Recovery only promotes candidates for the finite set of direct control
        records this repository creates.  Their payloads remain opaque here;
        digest-bound provenance and later authenticated lifecycle parsing supply
        the separate integrity boundaries.
        """
        if not name.endswith(".json"):
            return None
        base = name[:-5]
        if self._is_hex_identifier(base):
            return self.locator_for(base), base
        if base.startswith("reconcile-action-"):
            operation_id = base.removeprefix("reconcile-action-")
            if self._is_hex_identifier(operation_id):
                return self.reconciliation_checkpoint_locator(operation_id), None
            return None
        for kind in ("page", "checkpoint"):
            prefix = f"clear-target-{kind}-"
            if base.startswith(prefix):
                parts = base.removeprefix(prefix).split("-")
                if len(parts) == 2 and all(self._is_hex_identifier(part) for part in parts):
                    return (
                        self._clear_target_locator(
                            parts[0], parts[1], checkpoint=kind == "checkpoint"
                        ),
                        None,
                    )
                return None
        prefix = "clear-target-reference-"
        if not base.startswith(prefix):
            return None
        parts = base.removeprefix(prefix).split("-")
        if len(parts) == 2 and all(self._is_hex_identifier(part) for part in parts):
            return self._clear_target_reference_locator(parts[0], parts[1]), None
        if (
            len(parts) == 4
            and all(self._is_hex_identifier(part) for part in parts[:2])
            and parts[2] == "part"
            and len(parts[3]) == 4
            and all(character in "0123456789abcdef" for character in parts[3])
        ):
            chunk_index = int(parts[3], 16)
            if chunk_index < MAX_CLEAR_TARGET_REFERENCE_CHUNKS:
                return self._clear_target_reference_chunk_locator(
                    parts[0], parts[1], chunk_index
                ), None
        return None

    def list_pending_control_page(
        self,
        cursor: PendingControlCursor | None,
        *,
        max_inspections: int | None = None,
    ) -> PendingControlPage:
        """Read an indexed high-water page of pending controls.

        This deliberately performs no directory enumeration: candidate names
        were appended before their first durable write, and exact candidate
        bytes are still revalidated before any recovery promotion.
        """
        if cursor is not None and not isinstance(cursor, PendingControlCursor):
            raise TypeError("pending control cursor is invalid")
        limit = self.lifecycle_limits.operation_page_size
        entries, next_cursor, entry_next_cursors, inspected_positions = self._inventory_page(
            "pending",
            cursor,
            page_size=limit,
            max_inspections=max_inspections,
            read_current=self._get_pending_control_raw,
            cursor_type=PendingControlCursor,
        )
        return PendingControlPage(
            entries=entries,
            next_cursor=next_cursor,
            entry_next_cursors=entry_next_cursors,
            inspected_positions=inspected_positions,
        )

    def recover_pending_operation_records(
        self, *, max_inspections: int | None = None
    ) -> tuple[str, ...]:
        """Promote digest-bound interrupted lifecycle control candidates.

        A candidate name binds one exact final control name and SHA-256 of its
        contents. Malformed names, oversized bytes, and digest mismatches remain
        untouched for reconciliation reporting; no broad temporary-file sweep
        is ever used as ownership evidence.
        """
        cursor = self._pending_recovery_cursor()
        pending_history_exists = self._read_inventory("pending")["next_sequence"] > 1
        page = self.list_pending_control_page(cursor, max_inspections=max_inspections)
        recovered: list[str] = []
        eligible_actions = 0
        last_processed_index: int | None = None
        for entry_index, (name, raw) in enumerate(page.entries):
            if not (name.startswith(".") and name.endswith(".tmp")):
                last_processed_index = entry_index
                continue
            pending_parts = name[1:-4].rsplit(".pending.", 1)
            if len(pending_parts) != 2:
                last_processed_index = entry_index
                continue
            base, digest_and_token = pending_parts
            resolved_final = self._recoverable_pending_final(base)
            if resolved_final is None:
                last_processed_index = entry_index
                continue
            final_locator, operation_id = resolved_final
            digest_parts = digest_and_token.rsplit(".", 1)
            if len(digest_parts) != 2:
                last_processed_index = entry_index
                continue
            digest, token = digest_parts
            if (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or len(token) != 32
                or any(character not in "0123456789abcdef" for character in token)
            ):
                last_processed_index = entry_index
                continue
            if raw is None:
                last_processed_index = entry_index
                continue
            if hashlib.sha256(raw).hexdigest() != digest:
                last_processed_index = entry_index
                continue
            if eligible_actions >= self.lifecycle_limits.max_reconcile_actions:
                break
            try:
                promoted = self.file_ops.promote_durable_pending_control(
                    final_locator, raw, pending_name=name
                )
            except OSError as exc:
                raise CacheBlobBackendError(
                    "Interrupted lifecycle control evidence could not be recovered",
                    context={"operation_id": operation_id or "sidecar", "operation": "recover_pending"},
                ) from exc
            if promoted and operation_id is not None:
                recovered.append(operation_id)
            if entry_index < len(page.entry_next_cursors):
                self._compact_consumed_pending_event(
                    name, page.entry_next_cursors[entry_index]
                )
            # A digest-valid candidate is real bounded recovery work even if
            # a concurrent winner already installed the same final record.
            # Syntax-valid bytes with the wrong digest deliberately do not
            # consume this budget and remain untouched/reportable.
            eligible_actions += 1
            last_processed_index = entry_index
        # A pending candidate is scheduling evidence, never lifecycle
        # authority.  Persisting its opaque name lets later invocations move
        # past invalid or Windows-blocked prefixes without deleting them.
        if (
            last_processed_index is not None
            and last_processed_index < len(page.entries) - 1
        ):
            next_cursor = page.entry_next_cursors[last_processed_index]
        else:
            next_cursor = page.next_cursor
        self._checkpoint_pending_recovery_cursor(next_cursor)
        if pending_history_exists:
            # A crash after a normal control rename can leave only a stale
            # pending event.  Make bounded maintenance during recovery even
            # when that event did not yield an actionable candidate page.
            self._compact_inventory_after_retirement("pending")
        return tuple(recovered)

    def _is_eligible_pending_name(self, name: str) -> bool:
        """Recognize only exact digest-bound pending evidence candidates.

        Inventory bounds apply after this grammar filter.  Ordinary files,
        malformed controls, and clear/reconciliation sidecars must never
        consume a lifecycle recovery action slot indefinitely.
        """
        if not (name.startswith(".") and name.endswith(".tmp")):
            return False
        pending_parts = name[1:-4].rsplit(".pending.", 1)
        if len(pending_parts) != 2:
            return False
        base, digest_and_token = pending_parts
        if self._recoverable_pending_final(base) is None:
            return False
        digest_parts = digest_and_token.rsplit(".", 1)
        if len(digest_parts) != 2:
            return False
        digest, token = digest_parts
        return (
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and len(token) == 32
            and all(character in "0123456789abcdef" for character in token)
        )

    def _require_exact_current(
        self, record: LifecycleOperationRecord, expected_raw: bytes, *, operation: str
    ) -> None:
        """Reject stale evidence without parsing or interpreting its bytes."""
        if self.get_raw(record.operation_id) != expected_raw:
            raise CacheBlobLifecycleConflictError(
                "Lifecycle operation evidence no longer matches",
                context={"operation_id": record.operation_id, "operation": operation},
            )

    def checkpoint_if_exact(
        self,
        record: LifecycleOperationRecord,
        *,
        expected_raw: bytes,
        raw_record: bytes,
    ) -> None:
        """Replace evidence only when the observed bytes remain current."""
        if not isinstance(expected_raw, bytes) or not isinstance(raw_record, bytes):
            raise TypeError("Operation evidence transitions require exact bytes")
        try:
            with self._conditional_transition(
                f"operation:{record.operation_id}", operation_id=record.operation_id
            ):
                self._require_exact_current(
                    record, expected_raw, operation="checkpoint_if_exact"
                )
                self._append_inventory_event(
                    "primary", record.operation_id, raw_record
                )
                self.file_ops.write_bytes_durable(
                    self.locator_for(record.operation_id), raw_record
                )
        except CacheBlobLifecycleConflictError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence could not be checkpointed",
                context={
                    "operation_id": record.operation_id,
                    "operation": "checkpoint_if_exact",
                },
            ) from exc

    def checkpoint(self, record: LifecycleOperationRecord, raw_record: bytes) -> None:
        """Compatibility checkpoint that still verifies current evidence."""
        previous = self.get_raw(record.operation_id)
        if previous is None:
            raise CacheBlobLifecycleConflictError(
                "Lifecycle operation evidence is absent",
                context={"operation_id": record.operation_id, "operation": "checkpoint"},
            )
        self.checkpoint_if_exact(record, expected_raw=previous, raw_record=raw_record)

    def retire_if_exact(
        self, record: LifecycleOperationRecord, *, expected_raw: bytes
    ) -> None:
        """Retire evidence only when the exact terminal bytes remain current."""
        if not isinstance(expected_raw, bytes):
            raise TypeError("Operation evidence retirement requires exact bytes")
        try:
            with self._conditional_transition(
                f"operation:{record.operation_id}", operation_id=record.operation_id
            ):
                self._require_exact_current(
                    record, expected_raw, operation="retire_if_exact"
                )
                if not self.file_ops.delete_durable(self.locator_for(record.operation_id)):
                    raise CacheBlobLifecycleConflictError(
                        "Lifecycle operation evidence is absent",
                        context={
                            "operation_id": record.operation_id,
                            "operation": "retire_if_exact",
                        },
                    )
            self._compact_inventory_after_retirement("primary")
            # Primary lifecycle retirement is the normal completion boundary
            # for create/replace/delete.  Drain its durable pending-control
            # residue here; reads remain strictly non-mutating.
            self._compact_inventory_after_retirement("pending")
        except CacheBlobLifecycleConflictError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence could not be retired",
                context={
                    "operation_id": record.operation_id,
                    "operation": "retire_if_exact",
                },
            ) from exc

    def retire(self, record: LifecycleOperationRecord) -> None:
        """Compatibility retirement that still compares current evidence."""
        previous = self.get_raw(record.operation_id)
        if previous is not None:
            self.retire_if_exact(record, expected_raw=previous)

    def _page_size(self, page_size: int | None) -> int:
        """Allow smaller caller pages but never bypass configured resource bounds."""
        resolved = (
            self.lifecycle_limits.operation_page_size
            if page_size is None
            else page_size
        )
        if type(resolved) is not int or resolved <= 0:
            raise ValueError("operation page size must be a positive integer")
        if resolved > self.lifecycle_limits.operation_page_size:
            raise ValueError("operation page size exceeds configured lifecycle limit")
        return resolved

    def list_page(
        self,
        cursor: OperationCursor | None = None,
        *,
        page_size: int | None = None,
        max_inspections: int | None = None,
    ) -> OperationPage:
        """Return one bounded page using only ``page_size + 1`` ID slots.

        Directory membership, cursors, and raw evidence remain opaque here. The
        lifecycle layer authenticates the bytes before assigning any authority.
        """
        if cursor is not None and not isinstance(cursor, OperationCursor):
            raise TypeError("operation cursor must be an OperationCursor or None")
        limit = self._page_size(page_size)
        records, next_cursor, entry_next_cursors, inspected_positions = self._inventory_page(
            "primary",
            cursor,
            page_size=limit,
            max_inspections=max_inspections,
            read_current=self.get_raw,
            cursor_type=OperationCursor,
        )
        return OperationPage(
            entries=tuple(
                (operation_id, raw)
                for operation_id, raw in records
                if raw is not None
            ),
            next_cursor=next_cursor,
            entry_next_cursors=tuple(
                cursor
                for (_operation_id, raw), cursor in zip(records, entry_next_cursors)
                if raw is not None
            ),
            inspected_positions=inspected_positions,
        )

    def iter_raw(self) -> Iterator[tuple[str, bytes]]:
        """Compatibility iterator composed from bounded cursor pages."""
        cursor: OperationCursor | None = None
        while True:
            page = self.list_page(cursor)
            yield from page.entries
            if page.next_cursor is None:
                return
            cursor = page.next_cursor


__all__ = [
    "FileOperationRecordRepository",
    "OperationCursor",
    "OperationPage",
    "OperationRecordRepository",
    "PendingControlCursor",
    "PendingControlPage",
    "ReconciliationCheckpointCursor",
    "ReconciliationCheckpointPage",
]
