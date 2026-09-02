"""Bounded, evidence-only BlobStore reconciliation reports.

The reconciler intentionally inventories canonical control records, not the
payload directory.  A candidate filename is never provenance: only a signed
manifest plus a signed lifecycle record can make a payload eligible for a
later repair action.  The default path is a pure dry run.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobReconciliationCheckpointError,
    CacheBlobReconciliationConflictError,
    CacheBlobManifestUnauthenticatedError,
    CacheManifestIntegrityError,
    CacheManifestUnsupportedVersionError,
    CacheStorageError,
)

from .integrity import verify_hmac_sha256
from .manifest import BlobManifestV1
from .manifest_repository import ManifestCursor, ManifestPage
from .operation_record import OperationCheckpoint, OperationTransition
from .operation_repository import (
    OperationCursor,
    OperationPage,
    PendingControlCursor,
    PendingControlPage,
    ReconciliationCheckpointCursor,
    ReconciliationCheckpointPage,
)
from .path_security import resolve_managed_locator


logger = logging.getLogger(__name__)


class ReconciliationStatus(str, Enum):
    """Safety disposition for one reported lifecycle discrepancy."""

    SAFE = "safe"
    BLOCKED = "blocked"
    REQUIRES_CONFIRMATION = "requires_confirmation"


class ReconciliationAction(str, Enum):
    """The only actions a report can propose without exposing raw locators."""

    DELETE_CANDIDATE = "delete_candidate"
    RECLAIM_PREVIOUS = "reclaim_previous"
    COMPLETE_TOMBSTONE = "complete_tombstone"
    RETIRE_EVIDENCE = "retire_evidence"
    REPORT_ONLY = "report_only"


def _fingerprint(value: str | bytes) -> str:
    """Return a stable identifier without reflecting key or locator material."""
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class ReconciliationFinding:
    """One stable, redacted reconciliation conclusion."""

    status: ReconciliationStatus
    action: ReconciliationAction
    reason: str
    evidence_id: str | None = None
    evidence_digest: str | None = None
    manifest_digest: str | None = None
    key_fingerprint: str | None = None
    locator_fingerprint: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-compatible redacted representation."""
        return {
            "status": self.status.value,
            "action": self.action.value,
            "reason": self.reason,
            "evidence_id": self.evidence_id,
            "evidence_digest": self.evidence_digest,
            "manifest_digest": self.manifest_digest,
            "key_fingerprint": self.key_fingerprint,
            "locator_fingerprint": self.locator_fingerprint,
        }


@dataclass(frozen=True)
class ReconciliationReport:
    """Frozen report for one bounded reconciliation inspection or apply run."""

    findings: tuple[ReconciliationFinding, ...]
    resume_token: str | None
    applied: bool
    manifest_records_seen: int
    operation_records_seen: int

    @property
    def human_summary(self) -> str:
        """Return a bounded operator-oriented result summary."""
        counts = {status: 0 for status in ReconciliationStatus}
        for finding in self.findings:
            counts[finding.status] += 1
        return (
            "Reconciliation "
            f"({'apply' if self.applied else 'dry-run'}): "
            f"{len(self.findings)} finding(s); "
            f"safe={counts[ReconciliationStatus.SAFE]}, "
            f"blocked={counts[ReconciliationStatus.BLOCKED]}, "
            f"requires_confirmation={counts[ReconciliationStatus.REQUIRES_CONFIRMATION]}, "
            f"resume={'yes' if self.resume_token else 'no'}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic machine-readable report data."""
        return {
            "applied": self.applied,
            "findings": [finding.to_dict() for finding in self.findings],
            "manifest_records_seen": self.manifest_records_seen,
            "operation_records_seen": self.operation_records_seen,
            "resume_token": self.resume_token,
            "human_summary": self.human_summary,
        }


@dataclass(frozen=True)
class _AuthenticatedManifest:
    key: str
    raw: bytes
    manifest: BlobManifestV1
    locator: Path


@dataclass(frozen=True)
class _ActionCheckpoint:
    """Private signed marker bracketing one destructive reconciliation action."""

    operation_id: str
    evidence_digest: str
    action: ReconciliationAction
    state: str
    signature: str

    _DOMAIN = b"cacheness.reconciliation.action.v1\x00"

    def signing_bytes(self) -> bytes:
        return self._DOMAIN + json.dumps(
            {
                "action": self.action.value,
                "evidence_digest": self.evidence_digest,
                "operation_id": self.operation_id,
                "state": self.state,
                "version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def canonical_bytes(self, *, lifecycle_limits: LifecycleLimits | None = None) -> bytes:
        raw = json.dumps(
            {
                "action": self.action.value,
                "evidence_digest": self.evidence_digest,
                "operation_id": self.operation_id,
                "signature": self.signature,
                "state": self.state,
                "version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if (
            lifecycle_limits is not None
            and len(raw) > lifecycle_limits.max_operation_record_bytes
        ):
            raise CacheManifestIntegrityError(
                "Reconciliation checkpoint exceeds the configured byte limit"
            )
        return raw

    @classmethod
    def from_canonical_bytes(
        cls,
        raw: bytes,
        key: bytes,
        *,
        lifecycle_limits: LifecycleLimits | None = None,
    ) -> "_ActionCheckpoint":
        """Reject malformed, non-canonical, or unauthenticated progress bytes."""
        try:
            if (
                lifecycle_limits is not None
                and len(raw) > lifecycle_limits.max_operation_record_bytes
            ):
                raise ValueError
            mapping = json.loads(raw)
            if not isinstance(mapping, dict) or set(mapping) != {
                "action",
                "evidence_digest",
                "operation_id",
                "signature",
                "state",
                "version",
            }:
                raise ValueError
            action = ReconciliationAction(mapping["action"])
            checkpoint = cls(
                operation_id=mapping["operation_id"],
                evidence_digest=mapping["evidence_digest"],
                action=action,
                state=mapping["state"],
                signature=mapping["signature"],
            )
            if (
                mapping["version"] != 1
                or checkpoint.state
                not in {
                    "prepared",
                    "payload_deleted",
                    "tombstone_removed",
                    "completed",
                }
                or len(checkpoint.operation_id) != 32
                or any(char not in "0123456789abcdef" for char in checkpoint.operation_id)
                or len(checkpoint.evidence_digest) != 64
                or any(char not in "0123456789abcdef" for char in checkpoint.evidence_digest)
                or not isinstance(checkpoint.signature, str)
                or (
                    lifecycle_limits is not None
                    and any(
                        isinstance(value, str)
                        and len(value.encode("utf-8"))
                        > lifecycle_limits.max_operation_field_bytes
                        for value in mapping.values()
                    )
                )
                or checkpoint.canonical_bytes(lifecycle_limits=lifecycle_limits) != raw
                or not verify_hmac_sha256(
                    checkpoint.signing_bytes(), checkpoint.signature, key
                )
            ):
                raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CacheBlobReconciliationCheckpointError(
                "Reconciliation checkpoint is invalid"
            ) from exc
        return checkpoint

    @classmethod
    def new(
        cls,
        operation_id: str,
        evidence_digest: str,
        action: ReconciliationAction,
        state: str,
        key: bytes,
    ) -> "_ActionCheckpoint":
        unsigned = cls(operation_id, evidence_digest, action, state, signature="")
        signature = hmac.new(key, unsigned.signing_bytes(), hashlib.sha256).hexdigest()
        return cls(operation_id, evidence_digest, action, state, signature)


class _Reconciler:
    """Private coordinator for bounded, non-deserializing evidence analysis."""

    _TOKEN_DOMAIN = b"cacheness.reconciliation.cursor.v1\x00"
    _TOKEN_VERSION = 2
    _TOKEN_NONCE_BYTES = 12
    _TOKEN_TAG_BYTES = 16

    def __init__(self, store: Any, *, lifecycle_limits: LifecycleLimits):
        self.store = store
        # Keep the exact caller-owned policy object; reconciliation must not
        # silently widen a paging, grace, or action budget.
        self.lifecycle_limits = lifecycle_limits

    def reconcile(
        self,
        *,
        apply: bool = False,
        resume_token: str | None = None,
        now: datetime | None = None,
    ) -> ReconciliationReport:
        """Inspect bounded evidence, applying nothing unless explicitly requested."""
        if type(apply) is not bool:
            raise TypeError("apply must be a boolean")
        observed_at = datetime.now(timezone.utc) if now is None else now
        if observed_at.tzinfo is None:
            raise ValueError("reconciliation requires a timezone-aware clock")

        (
            manifest_cursor,
            operation_cursor,
            sidecar_cursor,
            pending_cursor,
            priority,
        ) = self._decode_resume_token(
            resume_token
        )
        manifest_page = self._manifest_page(manifest_cursor)
        operation_page = self._operation_page(operation_cursor)
        sidecar_page = self._sidecar_page(sidecar_cursor)
        pending_page = self._pending_control_page(pending_cursor)
        findings: list[ReconciliationFinding] = []
        remaining = self.lifecycle_limits.max_reconcile_actions
        manifest_seen = 0
        operation_seen = 0
        consumed_manifest = 0
        consumed_operation = 0
        consumed_sidecar = 0
        # Pending controls are non-authoritative scheduling residue.  Inventory
        # them on every report page, but do not let malformed or blocked bytes
        # consume an unrelated reconciliation action slot.
        findings.extend(
            self._classify_pending_control(name, raw)
            for name, raw in pending_page.entries
        )

        # Alternate the source that receives the first bounded slot.  This
        # preserves deterministic reports while preventing a long manifest
        # inventory from starving lifecycle-operation evidence forever.
        source_pages = {
            "manifest": manifest_page.entries,
            "operation": operation_page.entries,
            "sidecar": sidecar_page.entries,
        }
        source_order = ("manifest", "operation", "sidecar")
        first_index = source_order.index(priority)
        sources = tuple(
            (source, source_pages[source])
            for source in source_order[first_index:] + source_order[:first_index]
        )
        for source, entries in sources:
            for identifier, raw in entries:
                if remaining == 0:
                    break
                if source == "manifest":
                    findings.append(self._classify_manifest(identifier, raw))
                    manifest_seen += 1
                    consumed_manifest += 1
                else:
                    if source == "operation":
                        operation_finding = self._classify_operation(
                            identifier, raw, observed_at
                        )
                        operation_finding = self._block_primary_for_mismatched_sidecar(
                            operation_finding, raw
                        )
                        findings.append(operation_finding)
                        operation_seen += 1
                        consumed_operation += 1
                        if (
                            operation_finding.status is ReconciliationStatus.BLOCKED
                            and operation_finding.reason
                            == "reconciliation_checkpoint_primary_mismatch"
                        ):
                            # An authenticated but foreign sidecar is visible
                            # conflict evidence, not a global denial-of-service
                            # budget consumer.  Continue to later independent
                            # safe records in this one bounded apply pass.
                            continue
                    else:
                        sidecar_finding = self._classify_sidecar(
                            identifier, raw, observed_at
                        )
                        findings.append(sidecar_finding)
                        consumed_sidecar += 1
                        # A blocked private sidecar is reportable scan evidence,
                        # not an authorized mutation.  It must not consume the
                        # one shared action budget and starve a later completed
                        # orphan on the same bounded sidecar page.
                        if sidecar_finding.status is ReconciliationStatus.BLOCKED:
                            continue
                remaining -= 1
            if remaining == 0:
                break

        next_manifest = self._next_manifest_cursor(
            manifest_page, manifest_cursor, consumed_manifest
        )
        next_operation = self._next_operation_cursor(
            operation_page, operation_cursor, consumed_operation
        )
        next_sidecar = self._next_sidecar_cursor(
            sidecar_page, sidecar_cursor, consumed_sidecar
        )
        next_pending = self._next_pending_cursor(pending_page, pending_cursor)
        pending_manifest = (
            consumed_manifest < len(manifest_page.entries)
            or next_manifest is not None
        )
        pending_operation = (
            consumed_operation < len(operation_page.entries)
            or next_operation is not None
        )
        pending_sidecar = (
            consumed_sidecar < len(sidecar_page.entries) or next_sidecar is not None
        )
        next_priority = None
        if pending_manifest or pending_operation or pending_sidecar or next_pending is not None:
            pending_sources = {
                "manifest": pending_manifest,
                "operation": pending_operation,
                "sidecar": pending_sidecar,
            }
            for offset in range(1, len(source_order) + 1):
                candidate = source_order[(first_index + offset) % len(source_order)]
                if pending_sources[candidate]:
                    next_priority = candidate
                    break
            # Pending controls are a first-class recovery source even though
            # they do not consume a primary action slot.  The v2 token carries
            # their independent opaque cursor; use the stable sidecar starting
            # priority when they are the only remaining page so a truthful
            # token is emitted instead of silently dropping their work.
            if next_priority is None and next_pending is not None:
                next_priority = "sidecar"
        token = self._encode_resume_token(
            next_manifest, next_operation, next_sidecar, next_pending, next_priority
        )
        if apply:
            findings.extend(
                self._apply_findings(findings, observed_at)
            )
        return ReconciliationReport(
            findings=tuple(findings),
            resume_token=token,
            applied=apply,
            manifest_records_seen=manifest_seen,
            operation_records_seen=operation_seen,
        )

    def _apply_findings(
        self,
        findings: list[ReconciliationFinding],
        observed_at: datetime,
    ) -> tuple[ReconciliationFinding, ...]:
        """Apply one normalized, revalidated action stream under one snapshot gate.

        A dry-run finding is not itself a mutation budget charge.  The shared
        budget advances only when this pass reaches the one repository action
        that can make progress for that finding.  In particular, a completed
        orphan sidecar is a sidecar-retirement action, never a pre-pass plus a
        second no-op primary action.
        """
        dispositions: list[ReconciliationFinding] = []
        # Reconciliation uses aggregate admission only while it reloads exact
        # authority and applies a bounded report. Ordinary operations retain
        # their per-key lifecycle/CAS concurrency contract.
        with self.store._admission_barrier.aggregate_admission():
            remaining = self.lifecycle_limits.max_reconcile_actions
            for finding in findings:
                if remaining <= 0:
                    break
                if (
                    finding.status is not ReconciliationStatus.SAFE
                    or finding.evidence_id is None
                    or finding.evidence_digest is None
                ):
                    continue
                try:
                    if finding.reason == "completed_reconciliation_checkpoint_orphan":
                        attempted = self._retire_orphaned_completed_checkpoint(
                            finding
                        )
                    else:
                        attempted = self._apply_finding(finding, observed_at)
                except (
                    CacheBlobReconciliationCheckpointError,
                    CacheBlobReconciliationConflictError,
                    CacheBlobLifecycleConflictError,
                ):
                    # Exact primary/sidecar CAS races are local to this action.
                    # Preserve current bytes, make the conflict visible, and
                    # continue independently authenticated later work.
                    logger.warning(
                        "BlobStore reconciliation action conflicted",
                        extra={"operation_id": finding.evidence_id, "operation": "reconcile"},
                    )
                    dispositions.append(
                        ReconciliationFinding(
                            ReconciliationStatus.BLOCKED,
                            ReconciliationAction.REPORT_ONLY,
                            "reconciliation_action_conflict",
                            evidence_id=finding.evidence_id,
                            evidence_digest=finding.evidence_digest,
                            key_fingerprint=finding.key_fingerprint,
                            locator_fingerprint=finding.locator_fingerprint,
                        )
                    )
                    attempted = True
                if attempted:
                    remaining -= 1
        return tuple(dispositions)

    def _apply_finding(
        self,
        finding: ReconciliationFinding,
        observed_at: datetime,
    ) -> bool:
        """Perform one exact revalidated action, never using an old report as proof."""
        repository = self.store.lifecycle.operation_repository
        raw = repository.get_raw(finding.evidence_id)
        if raw is None or hashlib.sha256(raw).hexdigest() != finding.evidence_digest:
            return False
        refreshed = self._classify_operation(finding.evidence_id, raw, observed_at)
        if (
            refreshed.status is not ReconciliationStatus.SAFE
            or refreshed.action is not finding.action
            or refreshed.evidence_digest != finding.evidence_digest
        ):
            return False
        recovered = self.store.lifecycle._recoverable_record(finding.evidence_id, raw)
        if recovered is None:
            return False
        record, candidate, _previous = recovered
        checkpoint, checkpoint_raw = self._prepare_action_checkpoint(
            record.operation_id,
            finding.evidence_digest,
            finding.action,
        )
        if checkpoint.state == "completed":
            self._finish_completed_checkpoint(
                record, candidate, checkpoint, checkpoint_raw
            )
            return True
        if finding.action is ReconciliationAction.DELETE_CANDIDATE:
            self._apply_candidate_delete(record, candidate, checkpoint, checkpoint_raw)
        elif finding.action is ReconciliationAction.RETIRE_EVIDENCE:
            completed, completed_raw = self._complete_action_checkpoint(
                checkpoint, checkpoint_raw
            )
            self._finish_completed_checkpoint(
                record, candidate, completed, completed_raw
            )
        elif finding.action is ReconciliationAction.COMPLETE_TOMBSTONE:
            self._apply_complete_tombstone(
                record, candidate, checkpoint, checkpoint_raw
            )
        return True

    def _retire_orphaned_completed_checkpoint(
        self, finding: ReconciliationFinding
    ) -> bool:
        """Retire exactly one still-orphaned completed sidecar.

        This is deliberately an action in the same stream as primary
        reconciliation work.  It revalidates exact bytes and orphan status
        immediately before the one destructive sidecar retirement, so a
        concurrent primary creation or exact sidecar update cannot consume a
        second budget slot or abort later actions.
        """
        assert finding.evidence_id is not None
        assert finding.evidence_digest is not None
        repository = self.store.lifecycle.operation_repository
        raw = repository.get_reconciliation_checkpoint_raw(finding.evidence_id)
        if raw is None or hashlib.sha256(raw).hexdigest() != finding.evidence_digest:
            return False
        checkpoint = _ActionCheckpoint.from_canonical_bytes(
            raw,
            self.store._manifest_key(),
            lifecycle_limits=self.lifecycle_limits,
        )
        if (
            checkpoint.operation_id != finding.evidence_id
            or checkpoint.state != "completed"
        ):
            return False
        if repository.get_raw(finding.evidence_id) is not None:
            raise CacheBlobLifecycleConflictError(
                "Reconciliation checkpoint is no longer orphaned",
                context={
                    "operation_id": finding.evidence_id,
                    "operation": "retire_reconcile",
                },
            )
        self._retire_completed_checkpoint(checkpoint, raw)
        return True

    def _apply_candidate_delete(
        self,
        record: Any,
        candidate: Path,
        checkpoint: _ActionCheckpoint,
        checkpoint_raw: bytes,
    ) -> None:
        """Delete one exact candidate and make post-delete cancellation resumable."""
        if not self.store.guarded_handler_io.file_ops.exists(candidate):
            completed, completed_raw = self._complete_action_checkpoint(
                checkpoint, checkpoint_raw
            )
            self._finish_completed_checkpoint(
                record, candidate, completed, completed_raw
            )
            return
        try:
            self.store._delete_or_prove_absent(candidate)
        except BaseException:
            # A signal can arrive after the unlink reaches the filesystem.  If
            # absence is now proven, finish durable progress before preserving
            # the interruption; a fresh store will not call delete again.
            if not self.store.guarded_handler_io.file_ops.exists(candidate):
                completed, completed_raw = self._complete_action_checkpoint(
                    checkpoint, checkpoint_raw
                )
                self._finish_completed_checkpoint(
                    record, candidate, completed, completed_raw
                )
            raise
        completed, completed_raw = self._complete_action_checkpoint(
            checkpoint, checkpoint_raw
        )
        self._finish_completed_checkpoint(record, candidate, completed, completed_raw)

    def _finish_completed_checkpoint(
        self,
        record: Any,
        candidate: Path,
        checkpoint: _ActionCheckpoint,
        checkpoint_raw: bytes,
    ) -> None:
        """Finish only the non-destructive terminal steps after a checkpoint.

        A completed checkpoint means its destructive action must never repeat
        after a crash.  It stays durable until the primary evidence is safely
        retired, then its exact bytes are removed as the final sidecar step.
        """
        self.store.lifecycle._fault(
            "reconcile_tombstone_before_primary_retire", record
        )
        if checkpoint.action is ReconciliationAction.DELETE_CANDIDATE:
            if self.store.guarded_handler_io.file_ops.exists(candidate):
                raise CacheBlobReconciliationConflictError(
                    "Completed reconciliation checkpoint still has its candidate",
                    context={"operation_id": checkpoint.operation_id},
                )
            self.store.lifecycle._retire(record)
        elif checkpoint.action is ReconciliationAction.RETIRE_EVIDENCE:
            self.store.lifecycle._retire(record)
        elif checkpoint.action is ReconciliationAction.COMPLETE_TOMBSTONE:
            # A completed tombstone checkpoint is the durable proof that both
            # destructive stages have finished.  Only now may its primary
            # evidence retire; a crash before this point retains the exact
            # record needed to resume without replaying the payload delete.
            if self.store.lifecycle.operation_repository.get_raw(record.operation_id) is not None:
                self.store.lifecycle._fault(
                    "reconcile_tombstone_inside_primary_retire", record
                )
                self.store.lifecycle._retire(record)
        self.store.lifecycle._fault(
            "reconcile_tombstone_after_primary_retire", record
        )
        self.store.lifecycle._fault(
            "reconcile_tombstone_before_sidecar_retire", record
        )
        self.store.lifecycle._fault(
            "reconcile_tombstone_inside_sidecar_retire", record
        )
        self._retire_completed_checkpoint(checkpoint, checkpoint_raw)
        self.store.lifecycle._fault(
            "reconcile_tombstone_after_sidecar_retire", record
        )

    def _retire_completed_checkpoint(
        self, checkpoint: _ActionCheckpoint, checkpoint_raw: bytes
    ) -> None:
        """Retire one exact completed checkpoint after terminal ordering."""
        if checkpoint.state != "completed":
            raise CacheBlobReconciliationCheckpointError(
                "Only completed reconciliation checkpoints may be retired"
            )
        self.store.lifecycle.operation_repository.retire_reconciliation_checkpoint_if_exact(
            checkpoint.operation_id, expected_raw=checkpoint_raw
        )

    def _advance_action_checkpoint(
        self,
        checkpoint: _ActionCheckpoint,
        expected_raw: bytes,
        state: str,
    ) -> tuple[_ActionCheckpoint, bytes]:
        """Persist one monotonic reconciliation stage from exact sidecar bytes."""
        if checkpoint.state == state:
            return checkpoint, expected_raw
        updated = _ActionCheckpoint.new(
            checkpoint.operation_id,
            checkpoint.evidence_digest,
            checkpoint.action,
            state,
            self.store._manifest_key(),
        )
        raw = updated.canonical_bytes(lifecycle_limits=self.lifecycle_limits)
        self.store.lifecycle.operation_repository.checkpoint_reconciliation_if_exact(
            checkpoint.operation_id,
            expected_raw=expected_raw,
            raw_record=raw,
        )
        return updated, raw

    def _apply_complete_tombstone(
        self,
        record: Any,
        candidate: Path,
        checkpoint: _ActionCheckpoint,
        checkpoint_raw: bytes,
    ) -> None:
        """Resume tombstone cleanup without retiring its only primary evidence.

        The action sidecar is an explicit durable state machine.  It records
        each destructive effect before the primary lifecycle record can retire:
        payload removal, tombstone retirement, and finally completion.  Thus a
        ``BaseException`` at any seam leaves enough authenticated evidence for
        an apply/reopen to continue forward without deleting the payload twice.
        """
        if checkpoint.state == "prepared":
            try:
                # Reopen may reach a prepared sidecar after a process loss
                # immediately after the filesystem effect.  Observe absence
                # *before* issuing any destructive call, then checkpoint it.
                if self.store.guarded_handler_io.file_ops.exists(candidate):
                    self.store.lifecycle._fault(
                        "reconcile_tombstone_before_payload_delete", record
                    )
                    self.store.lifecycle._fault(
                        "reconcile_tombstone_inside_payload_delete", record
                    )
                    self.store._delete_or_prove_absent(candidate)
                    self.store.lifecycle._fault(
                        "reconcile_tombstone_after_payload_delete", record
                    )
                checkpoint, checkpoint_raw = self._advance_tombstone_checkpoint(
                    record, checkpoint, checkpoint_raw, "payload_deleted"
                )
            except BaseException:
                # Every checkpoint seam can interrupt after the delete.  Try
                # the monotonic CAS directly (without replaying its test hook)
                # before preserving the original signal.  If durable storage
                # is itself unavailable, the next reopen observes absence
                # before it can call delete again.
                if not self.store.guarded_handler_io.file_ops.exists(candidate):
                    try:
                        checkpoint, checkpoint_raw = self._advance_action_checkpoint(
                            checkpoint, checkpoint_raw, "payload_deleted"
                        )
                    except BaseException:
                        logger.warning(
                            "BlobStore could not durably acknowledge payload deletion",
                            extra={
                                "operation_id": record.operation_id,
                                "operation": "reconcile",
                            },
                        )
                raise

        if checkpoint.state == "payload_deleted":
            self.store.lifecycle._fault(
                "reconcile_tombstone_before_manifest_remove", record
            )
            try:
                self.store.lifecycle._fault(
                    "reconcile_tombstone_inside_manifest_remove", record
                )
                current = self.store._load_authenticated_manifest_with_raw(
                    record.key,
                    operation="reconcile_tombstone",
                    require_locator=True,
                    allowed_states=frozenset({"committed", "tombstoned"}),
                )
                if current is not None:
                    manifest, raw_manifest, _handler, locator = current
                    if (
                        manifest.state == "tombstoned"
                        and manifest.generation == record.generation
                        and locator == candidate
                    ):
                        expectation = self.store.lifecycle._tombstone_expectation(
                            manifest, raw_manifest
                        )
                        try:
                            self.store.manifest_repository.remove_if_expected(
                                record.key, expectation
                            )
                        except CacheBlobLifecycleConflictError:
                            # A later winner is already authoritative.  This
                            # action must not retry its removal; the sidecar still
                            # records that this tombstone's removal step is no
                            # longer required before primary retirement.
                            pass
                    elif manifest.generation == record.generation:
                        raise CacheBlobReconciliationConflictError(
                            "Tombstone reconciliation locator changed before retirement",
                            context={"operation_id": record.operation_id},
                        )
                self.store.lifecycle._fault(
                    "reconcile_tombstone_after_manifest_remove", record
                )
            except BaseException:
                # A matching tombstone may already be gone (or a later winner
                # may have won) even though the post-effect seam interrupted.
                current = self.store._load_authenticated_manifest_with_raw(
                    record.key,
                    operation="reconcile_tombstone_effect_probe",
                    require_locator=True,
                    allowed_states=frozenset({"committed", "tombstoned"}),
                )
                if current is None or current[0].generation != record.generation:
                    checkpoint, checkpoint_raw = self._advance_tombstone_checkpoint(
                        record, checkpoint, checkpoint_raw, "tombstone_removed"
                    )
                raise
            checkpoint, checkpoint_raw = self._advance_tombstone_checkpoint(
                record, checkpoint, checkpoint_raw, "tombstone_removed"
            )

        if checkpoint.state == "tombstone_removed":
            self.store.lifecycle._fault(
                "reconcile_tombstone_before_checkpoint_complete", record
            )
            self.store.lifecycle._fault(
                "reconcile_tombstone_inside_checkpoint_complete", record
            )
            checkpoint, checkpoint_raw = self._complete_action_checkpoint(
                checkpoint, checkpoint_raw
            )
            self.store.lifecycle._fault(
                "reconcile_tombstone_after_checkpoint_complete", record
            )

        if checkpoint.state == "completed":
            self._finish_completed_checkpoint(
                record, candidate, checkpoint, checkpoint_raw
            )

    def _advance_tombstone_checkpoint(
        self,
        record: Any,
        checkpoint: _ActionCheckpoint,
        checkpoint_raw: bytes,
        state: str,
    ) -> tuple[_ActionCheckpoint, bytes]:
        """Advance one tombstone stage with explicit pre/inside/post seams."""
        self.store.lifecycle._fault(
            f"reconcile_tombstone_before_{state}_checkpoint", record
        )
        self.store.lifecycle._fault(
            f"reconcile_tombstone_inside_{state}_checkpoint", record
        )
        advanced, advanced_raw = self._advance_action_checkpoint(
            checkpoint, checkpoint_raw, state
        )
        self.store.lifecycle._fault(
            f"reconcile_tombstone_after_{state}_checkpoint", record
        )
        return advanced, advanced_raw

    def _prepare_action_checkpoint(
        self,
        operation_id: str,
        evidence_digest: str,
        action: ReconciliationAction,
    ) -> tuple[_ActionCheckpoint, bytes]:
        """Durably record exact intent before any destructive reconciliation call."""
        repository = self.store.lifecycle.operation_repository
        key = self.store._manifest_key()
        raw = repository.get_reconciliation_checkpoint_raw(operation_id)
        if raw is None:
            checkpoint = _ActionCheckpoint.new(
                operation_id, evidence_digest, action, "prepared", key
            )
            raw = checkpoint.canonical_bytes(lifecycle_limits=self.lifecycle_limits)
            try:
                repository.create_reconciliation_checkpoint_exclusive(operation_id, raw)
            except CacheBlobLifecycleConflictError:
                raw = repository.get_reconciliation_checkpoint_raw(operation_id)
                if raw is None:
                    raise
        checkpoint = _ActionCheckpoint.from_canonical_bytes(
            raw, key, lifecycle_limits=self.lifecycle_limits
        )
        if (
            checkpoint.evidence_digest != evidence_digest
            or checkpoint.action is not action
        ):
            raise CacheBlobReconciliationConflictError(
                "Reconciliation checkpoint is bound to different evidence",
                context={"operation_id": operation_id, "operation": "reconcile"},
            )
        return checkpoint, raw

    def _complete_action_checkpoint(
        self, checkpoint: _ActionCheckpoint, expected_raw: bytes
    ) -> tuple[_ActionCheckpoint, bytes]:
        """Advance durable progress after an action, preserving exact CAS semantics."""
        if checkpoint.state == "completed":
            return checkpoint, expected_raw
        key = self.store._manifest_key()
        completed = _ActionCheckpoint.new(
            checkpoint.operation_id,
            checkpoint.evidence_digest,
            checkpoint.action,
            "completed",
            key,
        )
        raw = completed.canonical_bytes(lifecycle_limits=self.lifecycle_limits)
        self.store.lifecycle.operation_repository.checkpoint_reconciliation_if_exact(
            checkpoint.operation_id,
            expected_raw=expected_raw,
            raw_record=raw,
        )
        return completed, raw

    def _manifest_page(self, cursor: ManifestCursor | None) -> ManifestPage:
        return self.store.manifest_repository.list_page(
            cursor,
            page_size=self.lifecycle_limits.manifest_page_size,
        )

    def _operation_page(self, cursor: OperationCursor | None) -> OperationPage:
        return self.store.lifecycle.operation_repository.list_page(
            cursor,
            page_size=self.lifecycle_limits.operation_page_size,
        )

    def _sidecar_page(
        self, cursor: ReconciliationCheckpointCursor | None
    ) -> ReconciliationCheckpointPage:
        """Page sidecars independently so malformed prefixes cannot starve debt."""
        return self.store.lifecycle.operation_repository.list_reconciliation_checkpoint_page(
            cursor,
            page_size=self.lifecycle_limits.operation_page_size,
        )

    def _pending_control_page(
        self, cursor: PendingControlCursor | None
    ) -> PendingControlPage:
        """Page pending scheduling residue without treating it as authority."""
        return self.store.lifecycle.operation_repository.list_pending_control_page(cursor)

    @staticmethod
    def _next_manifest_cursor(
        page: ManifestPage, current: ManifestCursor | None, consumed: int
    ) -> ManifestCursor | None:
        if consumed == 0:
            return current
        if consumed < len(page.entries):
            return ManifestCursor(page.entries[consumed - 1][0])
        return page.next_cursor

    @staticmethod
    def _next_operation_cursor(
        page: OperationPage, current: OperationCursor | None, consumed: int
    ) -> OperationCursor | None:
        if consumed == 0:
            return current
        if consumed < len(page.entries):
            return OperationCursor(page.entries[consumed - 1][0])
        return page.next_cursor

    @staticmethod
    def _next_sidecar_cursor(
        page: ReconciliationCheckpointPage,
        current: ReconciliationCheckpointCursor | None,
        consumed: int,
    ) -> ReconciliationCheckpointCursor | None:
        if consumed == 0:
            return current
        if consumed < len(page.entries):
            return ReconciliationCheckpointCursor(
                f"reconcile-action-{page.entries[consumed - 1][0]}.json"
            )
        return page.next_cursor

    @staticmethod
    def _next_pending_cursor(
        page: PendingControlPage, current: PendingControlCursor | None
    ) -> PendingControlCursor | None:
        """Every pending page entry is reportable, so its page cursor advances."""
        if not page.entries:
            return page.next_cursor
        return page.next_cursor

    def _classify_manifest(self, key: str, raw: bytes) -> ReconciliationFinding:
        """Report manifest-only contradictions without reading any payload bytes."""
        digest = hashlib.sha256(raw).hexdigest()
        try:
            authenticated = self._authenticate_manifest(key, raw)
        except CacheManifestUnsupportedVersionError:
            return self._manifest_finding(
                ReconciliationStatus.BLOCKED,
                "manifest_unsupported_version",
                key,
                digest,
            )
        except (CacheManifestIntegrityError, CacheStorageError):
            return self._manifest_finding(
                ReconciliationStatus.BLOCKED,
                "manifest_untrusted",
                key,
                digest,
            )
        if authenticated.manifest.state == "tombstoned":
            return self._manifest_finding(
                ReconciliationStatus.BLOCKED,
                "tombstone_without_validated_operation",
                key,
                digest,
            )
        if not self.store.guarded_handler_io.file_ops.exists(authenticated.locator):
            return self._manifest_finding(
                ReconciliationStatus.BLOCKED,
                "manifest_payload_missing",
                key,
                digest,
                locator=authenticated.locator,
            )
        return self._manifest_finding(
            ReconciliationStatus.REQUIRES_CONFIRMATION,
            "manifest_has_no_reconciliation_debt",
            key,
            digest,
        )

    def _manifest_finding(
        self,
        status: ReconciliationStatus,
        reason: str,
        key: str,
        digest: str,
        *,
        locator: Path | None = None,
    ) -> ReconciliationFinding:
        return ReconciliationFinding(
            status=status,
            action=ReconciliationAction.REPORT_ONLY,
            reason=reason,
            manifest_digest=digest,
            key_fingerprint=_fingerprint(key),
            locator_fingerprint=None if locator is None else _fingerprint(str(locator)),
        )

    def _classify_operation(
        self,
        operation_id: str,
        raw: bytes,
        observed_at: datetime,
    ) -> ReconciliationFinding:
        """Classify one signed lifecycle record without treating its path as proof."""
        digest = hashlib.sha256(raw).hexdigest()
        recovered = self.store.lifecycle._recoverable_record(operation_id, raw)
        if recovered is None:
            return ReconciliationFinding(
                status=ReconciliationStatus.BLOCKED,
                action=ReconciliationAction.REPORT_ONLY,
                reason="operation_evidence_untrusted",
                evidence_id=operation_id,
                evidence_digest=digest,
            )
        record, candidate, previous = recovered
        raw_manifest = self.store.manifest_repository.get_raw(record.key)
        current = self._try_authenticated_manifest(record.key, raw_manifest)
        base = {
            "evidence_id": operation_id,
            "evidence_digest": digest,
            "key_fingerprint": _fingerprint(record.key),
            "locator_fingerprint": _fingerprint(str(candidate)),
        }
        if record.transition is OperationTransition.CLEAR:
            return ReconciliationFinding(
                ReconciliationStatus.BLOCKED,
                ReconciliationAction.REPORT_ONLY,
                "clear_reconciliation_requires_authenticated_snapshot",
                **base,
            )
        if raw_manifest is not None and current is None:
            return ReconciliationFinding(
                ReconciliationStatus.BLOCKED,
                ReconciliationAction.REPORT_ONLY,
                "manifest_reference_untrusted",
                **base,
            )
        if record.transition is OperationTransition.TOMBSTONE:
            if (
                current is not None
                and current.manifest.state == "tombstoned"
                and current.manifest.generation == record.generation
                and current.locator == candidate
            ):
                return ReconciliationFinding(
                    ReconciliationStatus.SAFE,
                    ReconciliationAction.COMPLETE_TOMBSTONE,
                    "authenticated_tombstone_cleanup_debt",
                    manifest_digest=hashlib.sha256(current.raw).hexdigest(),
                    **base,
                )
            if current is not None and current.manifest.state == "committed":
                return ReconciliationFinding(
                    ReconciliationStatus.SAFE,
                    ReconciliationAction.RETIRE_EVIDENCE,
                    "superseded_tombstone_evidence",
                    manifest_digest=hashlib.sha256(current.raw).hexdigest(),
                    **base,
                )
            if (
                current is None
                and self._tombstone_checkpoint_has_durable_effect(record, digest)
            ):
                # A post-remove interruption can leave the primary tombstone
                # behind after its matching manifest is gone.  Only a signed
                # sidecar that binds these exact primary bytes and has already
                # crossed a destructive-effect checkpoint may resume the
                # non-destructive terminal stages.  A merely named or
                # ``prepared`` sidecar remains blocked below.
                return ReconciliationFinding(
                    ReconciliationStatus.SAFE,
                    ReconciliationAction.COMPLETE_TOMBSTONE,
                    "authenticated_tombstone_checkpoint_debt",
                    **base,
                )
            return ReconciliationFinding(
                ReconciliationStatus.BLOCKED,
                ReconciliationAction.REPORT_ONLY,
                "tombstone_authority_ambiguous",
                **base,
            )
        if record.checkpoint in {
            OperationCheckpoint.PREPARED,
            OperationCheckpoint.CANDIDATE_PUBLISHED,
        } and current is None:
            if record.expected_generation is not None:
                return ReconciliationFinding(
                    ReconciliationStatus.BLOCKED,
                    ReconciliationAction.REPORT_ONLY,
                    "missing_authority_for_replacement_candidate",
                    **base,
                )
            if not self.store.lifecycle.is_pre_authority_candidate_eligible(
                record, now=observed_at
            ):
                return ReconciliationFinding(
                    ReconciliationStatus.REQUIRES_CONFIRMATION,
                    ReconciliationAction.REPORT_ONLY,
                    "orphan_grace_not_elapsed",
                    **base,
                )
            return ReconciliationFinding(
                ReconciliationStatus.SAFE,
                ReconciliationAction.DELETE_CANDIDATE,
                "authenticated_uncommitted_candidate",
                **base,
            )
        if (
            current is not None
            and current.manifest.generation == record.generation
            and current.locator == candidate
        ):
            if previous is not None and previous != candidate:
                return ReconciliationFinding(
                    ReconciliationStatus.REQUIRES_CONFIRMATION,
                    ReconciliationAction.RECLAIM_PREVIOUS,
                    "authenticated_previous_payload_cleanup_debt",
                    manifest_digest=hashlib.sha256(current.raw).hexdigest(),
                    **base,
                )
            return ReconciliationFinding(
                ReconciliationStatus.SAFE,
                ReconciliationAction.RETIRE_EVIDENCE,
                "authenticated_terminal_evidence",
                manifest_digest=hashlib.sha256(current.raw).hexdigest(),
                **base,
            )
        if (
            current is not None
            and record.expected_generation is not None
            and current.manifest.generation == record.expected_generation
            and candidate != current.locator
        ):
            return ReconciliationFinding(
                ReconciliationStatus.SAFE,
                ReconciliationAction.DELETE_CANDIDATE,
                "authenticated_stale_candidate",
                manifest_digest=hashlib.sha256(current.raw).hexdigest(),
                **base,
            )
        if (
            current is not None
            and current.manifest.generation != record.generation
            and candidate != current.locator
        ):
            # A later authority at a distinct locator proves this signed,
            # immutable candidate lost before CAS.  It is safe to converge
            # only this operation's own residue; the same-locator case stays
            # blocked because it may be a later generation's payload.
            return ReconciliationFinding(
                ReconciliationStatus.SAFE,
                ReconciliationAction.DELETE_CANDIDATE,
                "authenticated_superseded_candidate",
                manifest_digest=hashlib.sha256(current.raw).hexdigest(),
                **base,
            )
        return ReconciliationFinding(
            ReconciliationStatus.BLOCKED,
            ReconciliationAction.REPORT_ONLY,
            "operation_authority_ambiguous",
            **base,
        )

    def _tombstone_checkpoint_has_durable_effect(
        self, record: Any, evidence_digest: str
    ) -> bool:
        """Accept only a bound post-effect tombstone sidecar for no-manifest resume."""
        raw = self.store.lifecycle.operation_repository.get_reconciliation_checkpoint_raw(
            record.operation_id
        )
        if raw is None:
            return False
        try:
            checkpoint = _ActionCheckpoint.from_canonical_bytes(
                raw,
                self.store._manifest_key(),
                lifecycle_limits=self.lifecycle_limits,
            )
        except (CacheBlobReconciliationCheckpointError, CacheStorageError):
            return False
        return (
            checkpoint.operation_id == record.operation_id
            and checkpoint.evidence_digest == evidence_digest
            and checkpoint.action is ReconciliationAction.COMPLETE_TOMBSTONE
            and checkpoint.state
            in {"payload_deleted", "tombstone_removed", "completed"}
        )

    def _classify_sidecar(
        self, operation_id: str, raw: bytes | None, observed_at: datetime
    ) -> ReconciliationFinding:
        """Report private checkpoint state without letting it become authority."""
        if raw is None:
            return ReconciliationFinding(
                ReconciliationStatus.BLOCKED,
                ReconciliationAction.REPORT_ONLY,
                "reconciliation_checkpoint_untrusted",
                evidence_id=operation_id,
            )
        try:
            checkpoint = _ActionCheckpoint.from_canonical_bytes(
                raw,
                self.store._manifest_key(),
                lifecycle_limits=self.lifecycle_limits,
            )
        except (CacheBlobReconciliationCheckpointError, CacheStorageError):
            return ReconciliationFinding(
                ReconciliationStatus.BLOCKED,
                ReconciliationAction.REPORT_ONLY,
                "reconciliation_checkpoint_untrusted",
                evidence_id=operation_id,
                evidence_digest=hashlib.sha256(raw).hexdigest(),
            )
        if checkpoint.operation_id != operation_id:
            return ReconciliationFinding(
                ReconciliationStatus.BLOCKED,
                ReconciliationAction.REPORT_ONLY,
                "reconciliation_checkpoint_untrusted",
                evidence_id=operation_id,
                evidence_digest=hashlib.sha256(raw).hexdigest(),
            )
        primary_raw = self.store.lifecycle.operation_repository.get_raw(operation_id)
        if primary_raw is None and checkpoint.state == "completed":
            return ReconciliationFinding(
                ReconciliationStatus.SAFE,
                ReconciliationAction.RETIRE_EVIDENCE,
                "completed_reconciliation_checkpoint_orphan",
                evidence_id=operation_id,
                evidence_digest=hashlib.sha256(raw).hexdigest(),
            )
        if primary_raw is None:
            return ReconciliationFinding(
                ReconciliationStatus.BLOCKED,
                ReconciliationAction.REPORT_ONLY,
                "reconciliation_checkpoint_orphan_incomplete",
                evidence_id=operation_id,
                evidence_digest=hashlib.sha256(raw).hexdigest(),
            )
        primary_finding = self._classify_operation(
            operation_id, primary_raw, observed_at
        )
        if not self._checkpoint_binds_primary(checkpoint, primary_raw, primary_finding):
            return ReconciliationFinding(
                ReconciliationStatus.BLOCKED,
                ReconciliationAction.REPORT_ONLY,
                "reconciliation_checkpoint_primary_mismatch",
                evidence_id=operation_id,
                evidence_digest=hashlib.sha256(raw).hexdigest(),
            )
        return ReconciliationFinding(
            ReconciliationStatus.REQUIRES_CONFIRMATION,
            ReconciliationAction.REPORT_ONLY,
            "reconciliation_checkpoint_attached",
            evidence_id=operation_id,
            evidence_digest=hashlib.sha256(raw).hexdigest(),
        )

    def _block_primary_for_mismatched_sidecar(
        self, finding: ReconciliationFinding, primary_raw: bytes
    ) -> ReconciliationFinding:
        """Keep an attached sidecar from proposing authority for other bytes."""
        if (
            finding.status is not ReconciliationStatus.SAFE
            or finding.evidence_id is None
            or finding.action is ReconciliationAction.REPORT_ONLY
        ):
            return finding
        raw = self.store.lifecycle.operation_repository.get_reconciliation_checkpoint_raw(
            finding.evidence_id
        )
        if raw is None:
            return finding
        try:
            checkpoint = _ActionCheckpoint.from_canonical_bytes(
                raw,
                self.store._manifest_key(),
                lifecycle_limits=self.lifecycle_limits,
            )
        except (CacheBlobReconciliationCheckpointError, CacheStorageError):
            checkpoint = None
        if checkpoint is None or not self._checkpoint_binds_primary(
            checkpoint, primary_raw, finding
        ):
            return ReconciliationFinding(
                ReconciliationStatus.BLOCKED,
                ReconciliationAction.REPORT_ONLY,
                "reconciliation_checkpoint_primary_mismatch",
                evidence_id=finding.evidence_id,
                evidence_digest=finding.evidence_digest,
                key_fingerprint=finding.key_fingerprint,
                locator_fingerprint=finding.locator_fingerprint,
            )
        return finding

    @staticmethod
    def _checkpoint_binds_primary(
        checkpoint: _ActionCheckpoint,
        primary_raw: bytes,
        primary_finding: ReconciliationFinding,
    ) -> bool:
        """Require exact primary bytes, compatible action, and legal stage."""
        if checkpoint.evidence_digest != hashlib.sha256(primary_raw).hexdigest():
            return False
        if (
            primary_finding.status is not ReconciliationStatus.SAFE
            or checkpoint.action is not primary_finding.action
        ):
            return False
        if checkpoint.action is ReconciliationAction.COMPLETE_TOMBSTONE:
            return checkpoint.state in {
                "prepared",
                "payload_deleted",
                "tombstone_removed",
                "completed",
            }
        return checkpoint.state in {"prepared", "completed"}

    @staticmethod
    def _classify_pending_control(
        name: str, raw: bytes | None
    ) -> ReconciliationFinding:
        """Expose pending residue without granting a filename destructive authority."""
        digest = name[1:-4].rsplit(".pending.", 1)[1].rsplit(".", 1)[0]
        if raw is None or hashlib.sha256(raw).hexdigest() != digest:
            return ReconciliationFinding(
                ReconciliationStatus.BLOCKED,
                ReconciliationAction.REPORT_ONLY,
                "pending_control_untrusted",
                locator_fingerprint=_fingerprint(name),
            )
        return ReconciliationFinding(
            ReconciliationStatus.SAFE,
            ReconciliationAction.REPORT_ONLY,
            "pending_control_recovery_scheduled",
            evidence_digest=hashlib.sha256(raw).hexdigest(),
            locator_fingerprint=_fingerprint(name),
        )

    def _try_authenticated_manifest(
        self, key: str, raw: bytes | None
    ) -> _AuthenticatedManifest | None:
        if raw is None:
            return None
        try:
            return self._authenticate_manifest(key, raw)
        except (CacheManifestIntegrityError, CacheManifestUnsupportedVersionError, CacheStorageError):
            return None

    def _authenticate_manifest(self, key: str, raw: bytes) -> _AuthenticatedManifest:
        manifest = BlobManifestV1.from_canonical_bytes(raw)
        if manifest.canonical_bytes() != raw or manifest.key != key:
            raise CacheManifestIntegrityError("Manifest bytes are not canonical")
        if not verify_hmac_sha256(
            manifest.signing_bytes(), manifest.signature, self.store._manifest_key()
        ):
            raise CacheBlobManifestUnauthenticatedError("Manifest signature is invalid")
        locator = resolve_managed_locator(
            self.store.guarded_handler_io.root,
            manifest.locator,
            operation="reconcile_manifest",
        )
        return _AuthenticatedManifest(key=key, raw=raw, manifest=manifest, locator=locator)

    def _encode_resume_token(
        self,
        manifest_cursor: ManifestCursor | None,
        operation_cursor: OperationCursor | None,
        sidecar_cursor: ReconciliationCheckpointCursor | None,
        pending_cursor: PendingControlCursor | None,
        priority: str | None,
    ) -> str | None:
        if priority is None:
            return None
        # The token is AEAD-protected with a domain-separated key derived from
        # the existing manifest key. A fresh nonce is mandatory: cursor tokens
        # must not reveal relations between independent logical-key cursors.
        key = self._token_key()
        payload = json.dumps(
            {
                "manifest": None if manifest_cursor is None else manifest_cursor.key,
                "operation": (
                    None if operation_cursor is None else operation_cursor.operation_id
                ),
                "sidecar": (
                    None if sidecar_cursor is None else sidecar_cursor.operation_id
                ),
                "pending": None if pending_cursor is None else pending_cursor.name,
                "priority": priority,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > self.lifecycle_limits.max_operation_field_bytes:
            raise ValueError("reconciliation resume token payload exceeds the byte limit")
        nonce = os.urandom(self._TOKEN_NONCE_BYTES)
        encrypted = ChaCha20Poly1305(key).encrypt(nonce, payload, self._TOKEN_DOMAIN)
        return base64.urlsafe_b64encode(
            bytes((self._TOKEN_VERSION,)) + nonce + encrypted
        ).decode("ascii")

    def _decode_resume_token(
        self, token: str | None
    ) -> tuple[
        ManifestCursor | None,
        OperationCursor | None,
        ReconciliationCheckpointCursor | None,
        PendingControlCursor | None,
        str,
    ]:
        if token is None:
            # Sidecars are inspected first.  Empty stores therefore avoid
            # loading a key, while malformed sidecars become visible in the
            # first dry run instead of being hidden behind primary paging.
            return None, None, None, None, "sidecar"
        if not isinstance(token, str) or not token:
            raise ValueError("reconciliation resume token must be a non-empty string")
        maximum = self.lifecycle_limits.max_operation_field_bytes
        max_encoded = 4 * ((1 + self._TOKEN_NONCE_BYTES + maximum + self._TOKEN_TAG_BYTES + 2) // 3)
        if len(token) > max_encoded:
            raise ValueError("reconciliation resume token exceeds the byte limit")
        try:
            packed = base64.urlsafe_b64decode(token.encode("ascii"))
            minimum = 1 + self._TOKEN_NONCE_BYTES + self._TOKEN_TAG_BYTES
            if len(packed) < minimum or len(packed) > minimum + maximum:
                raise ValueError("reconciliation resume token is invalid")
            version = packed[0]
            if version not in {1, self._TOKEN_VERSION}:
                raise ValueError("reconciliation resume token is invalid")
            nonce_end = 1 + self._TOKEN_NONCE_BYTES
            nonce = packed[1:nonce_end]
            payload = ChaCha20Poly1305(self._token_key()).decrypt(
                nonce, packed[nonce_end:], self._TOKEN_DOMAIN
            )
            if len(payload) > maximum:
                raise ValueError("reconciliation resume token is invalid")
            decoded = json.loads(payload)
            expected_fields = (
                {"manifest", "operation", "priority"}
                if version == 1
                else {"manifest", "operation", "sidecar", "pending", "priority"}
            )
            if set(decoded) != expected_fields:
                raise ValueError("reconciliation resume token is malformed")
            manifest = decoded["manifest"]
            operation = decoded["operation"]
            sidecar = None if version == 1 else decoded["sidecar"]
            pending = None if version == 1 else decoded["pending"]
            priority = decoded["priority"]
            valid_priorities = (
                {"manifest", "operation"}
                if version == 1
                else {"manifest", "operation", "sidecar"}
            )
            if priority not in valid_priorities:
                raise ValueError("reconciliation resume token is malformed")
            return (
                None if manifest is None else ManifestCursor(manifest),
                None if operation is None else OperationCursor(operation),
                None if sidecar is None else ReconciliationCheckpointCursor(sidecar),
                None if pending is None else PendingControlCursor(pending),
                priority,
            )
        except (InvalidTag, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("reconciliation resume token is invalid") from exc

    def _token_key(self) -> bytes:
        """Derive the token-only AEAD key without reusing manifest signatures."""
        return hmac.new(
            self.store._manifest_key(),
            self._TOKEN_DOMAIN + b"chacha20-poly1305-key",
            hashlib.sha256,
        ).digest()
