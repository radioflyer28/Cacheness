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
import time
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
from .lifecycle_authority import (
    ReconciliationSnapshot,
    ReconciliationWork,
)
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
    finding_id: str | None = None
    authority_revision: int | None = None
    run_revision: int | None = None
    operation_provenance: str | None = None
    expected_generation: str | None = None
    authoritative_generation: str | None = None
    residue_type: str | None = None
    residue_role: str | None = None
    applied_state: str = "not_applied"
    checkpoint_state: str = "pending"

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

    def machine_dict(self) -> dict[str, Any]:
        """Return the canonical D-13 v2 projection without raw locators or keys."""
        disposition = {
            ReconciliationStatus.SAFE: "safe",
            ReconciliationStatus.BLOCKED: "blocked",
            ReconciliationStatus.REQUIRES_CONFIRMATION: "confirmation_required",
        }[self.status]
        return {
            "finding_id": self.finding_id
            or _fingerprint("|".join(filter(None, (self.reason, self.evidence_id)))),
            "authority_revision": self.authority_revision,
            "run_revision": self.run_revision,
            "operation_provenance": self.operation_provenance,
            "key_fingerprint": self.key_fingerprint,
            "expected_generation": self.expected_generation,
            "authoritative_generation": self.authoritative_generation,
            "residue_type": self.residue_type,
            "residue_role": self.residue_role,
            "proposed_action": self.action.value,
            "reason_code": self.reason,
            "disposition": disposition,
            "applied_state": self.applied_state,
            "checkpoint_state": self.checkpoint_state,
        }


@dataclass(frozen=True)
class _ApplyOutcome:
    """Private revalidation result used to derive a truthful continuation.

    A report finding describes a scan-time conclusion.  It cannot by itself
    advance durable scheduling: exact evidence can change between scan and
    apply.  Keeping this small outcome separate avoids treating a stale
    finding as either a completed effect or a spent action budget slot.
    """

    finding: ReconciliationFinding
    source: str
    attempted: bool
    completed: bool
    conflicted: bool = False

    @property
    def stale_or_unapplied(self) -> bool:
        # An attempted exact CAS is a budget charge, not source progression.
        # A conflicting action has not discharged the current source member;
        # advancing its authenticated cursor would make a terminal-looking
        # report hide still-present safe or unresolved debt.
        return not self.completed


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
        """Return the characterized legacy-v1 dictionary exactly."""
        return {
            "applied": self.applied,
            "findings": [finding.to_dict() for finding in self.findings],
            "manifest_records_seen": self.manifest_records_seen,
            "operation_records_seen": self.operation_records_seen,
            "resume_token": self.resume_token,
            "human_summary": self.human_summary,
        }

    def machine_view(self, *, version: int = 2) -> dict[str, Any]:
        """Return the explicit versioned operator machine view.

        Zero-argument :meth:`to_dict` deliberately stays a pure legacy-v1
        adapter.  New consumers must select this schema-versioned surface so
        a compatibility projection can never become a second report model.
        """
        if version != 2:
            raise ValueError(f"Unsupported reconciliation report version: {version}")
        return {
            "schema_version": 2,
            "findings": [finding.machine_dict() for finding in self.findings],
            "resume_token": self.resume_token,
        }


class _AuthorityReconciler:
    """Bounded authority-only reconciliation for the lifecycle-authority path."""

    _TOKEN_DOMAIN = b"cacheness.authority-reconciliation.resume.v1\x00"

    def __init__(self, store: Any, *, lifecycle_limits: LifecycleLimits) -> None:
        self.store = store
        self.authority = store.lifecycle_authority
        self.lifecycle_limits = lifecycle_limits

    def reconcile(
        self,
        *,
        apply: bool = False,
        resume_token: str | None = None,
        now: datetime | None = None,
    ) -> ReconciliationReport:
        """Report or apply bounded authority work without opening payload bytes."""
        if type(apply) is not bool:
            raise TypeError("apply must be a boolean")
        if now is not None and now.tzinfo is None:
            raise ValueError("reconciliation requires a timezone-aware clock")
        run_token, snapshot, mutation_cursor, debt_cursor = self._resume_state(
            apply=apply, resume_token=resume_token
        )
        deadline = time.monotonic() + self.lifecycle_limits.authority_busy_timeout_seconds
        row_budget = self.lifecycle_limits.operation_page_size
        action_budget = self.lifecycle_limits.max_reconcile_actions
        byte_budget = self.lifecycle_limits.max_operation_record_bytes
        inspected_rows = 0
        applied_actions = 0
        inspected_bytes = 0
        findings: list[ReconciliationFinding] = []

        while (
            inspected_rows < row_budget
            and applied_actions < action_budget
            and inspected_bytes < byte_budget
            and time.monotonic() < deadline
        ):
            page = self.authority.page_reconciliation_work(
                snapshot,
                mutation_cursor=mutation_cursor,
                debt_cursor=debt_cursor,
            )
            if not page.works:
                mutation_cursor = page.mutation_cursor
                debt_cursor = page.debt_cursor
                break
            if not any(work.source == "mutation" for work in page.works):
                mutation_cursor = page.mutation_cursor
            if not any(work.source == "debt" for work in page.works):
                debt_cursor = page.debt_cursor
            for work in page.works:
                if (
                    inspected_rows >= row_budget
                    or inspected_bytes >= byte_budget
                    or time.monotonic() >= deadline
                ):
                    break
                finding, evidence_bytes = self._classify(work, snapshot)
                if inspected_bytes + evidence_bytes > byte_budget:
                    break
                inspected_rows += 1
                inspected_bytes += evidence_bytes
                if apply and finding.status is ReconciliationStatus.SAFE:
                    applied = self._apply(work, finding)
                    applied_actions += int(applied)
                    finding = self._with_apply_outcome(finding, applied)
                    if run_token is not None:
                        self.authority.checkpoint_reconciliation(
                            run_token,
                            work,
                            state="completed" if applied else "blocked",
                        )
                findings.append(finding)
                if work.source == "mutation":
                    mutation_cursor = work.row_id
                else:
                    debt_cursor = work.row_id
                if applied_actions >= action_budget:
                    break
            else:
                mutation_cursor = max(mutation_cursor, page.mutation_cursor)
                debt_cursor = max(debt_cursor, page.debt_cursor)
                continue
            break

        complete = (
            mutation_cursor >= snapshot.mutation_high_water
            and debt_cursor >= snapshot.debt_high_water
        )
        if apply and complete and run_token is not None:
            self.authority.checkpoint_reconciliation(run_token)
        return ReconciliationReport(
            findings=tuple(findings),
            resume_token=None
            if complete
            else self._encode_resume_token(
                run_token, snapshot, mutation_cursor, debt_cursor
            ),
            applied=apply,
            manifest_records_seen=0,
            operation_records_seen=inspected_rows,
        )

    def _resume_state(
        self, *, apply: bool, resume_token: str | None
    ) -> tuple[Any | None, ReconciliationSnapshot, int, int]:
        if resume_token is not None:
            payload = self._decode_resume_token(resume_token)
            run_id = payload.get("run_id")
            run_token = None if run_id is None else self._page_token(run_id)
            snapshot = (
                self.authority.reconciliation_snapshot(run_token)
                if run_token is not None
                else ReconciliationSnapshot(
                    payload["authority_revision"],
                    payload["mutation_high_water"],
                    payload["debt_high_water"],
                )
            )
            return run_token, snapshot, payload["mutation_cursor"], payload["debt_cursor"]
        if apply:
            run_token = self.authority.begin_reconciliation()
            return run_token, self.authority.reconciliation_snapshot(run_token), 0, 0
        return None, self.authority.reconciliation_snapshot(), 0, 0

    @staticmethod
    def _page_token(value: str):
        from .lifecycle_authority import PageToken

        return PageToken(value)

    def _classify(
        self, work: ReconciliationWork, snapshot: ReconciliationSnapshot
    ) -> tuple[ReconciliationFinding, int]:
        if work.source == "debt":
            assert work.debt is not None
            return self._classify_debt(work, snapshot)
        assert work.mutation is not None
        return self._classify_mutation(work, snapshot)

    def _classify_debt(
        self, work: ReconciliationWork, snapshot: ReconciliationSnapshot
    ) -> tuple[ReconciliationFinding, int]:
        debt = work.debt
        assert debt is not None
        current = self.authority.read_entry(debt.key)
        authoritative_generation = None
        try:
            if current is not None:
                manifest = self.store._authenticated_authority_manifest(current.manifest)
                authoritative_generation = manifest.generation
                if manifest.locator == debt.locator:
                    raise CacheBlobLifecycleConflictError(
                        "Cleanup debt is still current authority ownership"
                    )
        except CacheStorageError:
            return (
                self._finding(
                    work,
                    snapshot,
                    status=ReconciliationStatus.BLOCKED,
                    action=ReconciliationAction.REPORT_ONLY,
                    reason="authority_evidence_ambiguous",
                    key=debt.key,
                    expected_generation=debt.generation,
                    authoritative_generation=authoritative_generation,
                    residue_type="cleanup_debt",
                    residue_role=debt.role,
                    operation_id=debt.operation_id,
                ),
                len(debt.locator.encode("utf-8")),
            )
        return (
            self._finding(
                work,
                snapshot,
                status=ReconciliationStatus.SAFE,
                action=ReconciliationAction.RECLAIM_PREVIOUS,
                reason="authenticated_cleanup_debt",
                key=debt.key,
                expected_generation=debt.generation,
                authoritative_generation=authoritative_generation,
                residue_type="cleanup_debt",
                residue_role=debt.role,
                operation_id=debt.operation_id,
            ),
            len(debt.locator.encode("utf-8")),
        )

    def _classify_mutation(
        self, work: ReconciliationWork, snapshot: ReconciliationSnapshot
    ) -> tuple[ReconciliationFinding, int]:
        prepared = work.mutation
        assert prepared is not None
        spec = prepared.spec
        current = self.authority.read_entry(spec.key)
        authoritative_generation = None
        status = ReconciliationStatus.SAFE
        action = ReconciliationAction.DELETE_CANDIDATE
        reason = "authenticated_prepared_mutation"
        try:
            if work.state != "prepared":
                status = ReconciliationStatus.BLOCKED
                action = ReconciliationAction.REPORT_ONLY
                reason = "mutation_not_pending"
            elif current is not None:
                manifest = self.store._authenticated_authority_manifest(current.manifest)
                authoritative_generation = manifest.generation
                if manifest.locator == spec.candidate_locator:
                    status = ReconciliationStatus.BLOCKED
                    action = ReconciliationAction.REPORT_ONLY
                    reason = "candidate_still_authoritatively_owned"
        except CacheStorageError:
            status = ReconciliationStatus.BLOCKED
            action = ReconciliationAction.REPORT_ONLY
            reason = "authority_evidence_ambiguous"
        return (
            self._finding(
                work,
                snapshot,
                status=status,
                action=action,
                reason=reason,
                key=spec.key,
                expected_generation=spec.generation,
                authoritative_generation=authoritative_generation,
                residue_type="mutation",
                residue_role="candidate",
                operation_id=prepared.operation_id,
            ),
            len(spec.manifest) + len(spec.candidate_locator.encode("utf-8")),
        )

    def _finding(
        self,
        work: ReconciliationWork,
        snapshot: ReconciliationSnapshot,
        *,
        status: ReconciliationStatus,
        action: ReconciliationAction,
        reason: str,
        key: str,
        expected_generation: str | None,
        authoritative_generation: str | None,
        residue_type: str,
        residue_role: str,
        operation_id: str,
    ) -> ReconciliationFinding:
        provenance = _fingerprint(operation_id)
        finding_id = _fingerprint(
            f"{work.source}:{work.row_id}:{provenance}:{expected_generation or ''}"
        )
        return ReconciliationFinding(
            status,
            action,
            reason,
            evidence_id=provenance,
            evidence_digest=None,
            key_fingerprint=_fingerprint(key),
            finding_id=finding_id,
            authority_revision=snapshot.authority_revision,
            run_revision=snapshot.authority_revision,
            operation_provenance=provenance,
            expected_generation=expected_generation,
            authoritative_generation=authoritative_generation,
            residue_type=residue_type,
            residue_role=residue_role,
        )

    def _apply(self, work: ReconciliationWork, finding: ReconciliationFinding) -> bool:
        """Reload/revalidate exact authority evidence immediately before cleanup."""
        try:
            if work.source == "debt":
                debt = work.debt
                assert debt is not None
                current = self.authority.pending_cleanup_debts(
                    operation_id=debt.operation_id
                )
                if not any(
                    candidate.locator == debt.locator
                    and candidate.key == debt.key
                    and candidate.generation == debt.generation
                    and candidate.role == debt.role
                    for candidate in current
                ):
                    return True
                self.store._authority_lifecycle._settle_debts((debt,))
                return True
            prepared = work.mutation
            assert prepared is not None
            if prepared not in self.authority.pending_mutations():
                return True
            self.store._authority_lifecycle._abort(prepared, candidate_persisted=True)
            return True
        except (CacheBlobLifecycleConflictError, CacheStorageError):
            return False

    @staticmethod
    def _with_apply_outcome(
        finding: ReconciliationFinding, applied: bool
    ) -> ReconciliationFinding:
        return ReconciliationFinding(
            **{
                **finding.__dict__,
                "applied_state": "applied" if applied else "blocked",
                "checkpoint_state": "completed" if applied else "blocked",
            }
        )

    def _encode_resume_token(
        self,
        run_token: Any | None,
        snapshot: ReconciliationSnapshot,
        mutation_cursor: int,
        debt_cursor: int,
    ) -> str:
        payload = {
            "authority_revision": snapshot.authority_revision,
            "debt_cursor": debt_cursor,
            "debt_high_water": snapshot.debt_high_water,
            "mutation_cursor": mutation_cursor,
            "mutation_high_water": snapshot.mutation_high_water,
            "run_id": None if run_token is None else run_token.value,
            "version": 1,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        signature = hmac.new(
            self.store._authority_manifest_key(initialize_new_store=False),
            self._TOKEN_DOMAIN + encoded,
            hashlib.sha256,
        ).hexdigest()
        return base64.urlsafe_b64encode(encoded + b"." + signature.encode("ascii")).decode("ascii")

    def _decode_resume_token(self, token: str) -> dict[str, Any]:
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii"))
            encoded, signature = raw.rsplit(b".", 1)
            expected = hmac.new(
                self.store._authority_manifest_key(initialize_new_store=False),
                self._TOKEN_DOMAIN + encoded,
                hashlib.sha256,
            ).hexdigest().encode("ascii")
            payload = json.loads(encoded)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Reconciliation resume token is malformed") from exc
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Reconciliation resume token is unauthenticated")
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or any(
                type(payload.get(field)) is not int or payload[field] < 0
                for field in (
                    "authority_revision",
                    "mutation_cursor",
                    "mutation_high_water",
                    "debt_cursor",
                    "debt_high_water",
                )
            )
            or payload.get("run_id") is not None
            and (not isinstance(payload["run_id"], str) or not payload["run_id"])
        ):
            raise ValueError("Reconciliation resume token is malformed")
        return payload


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
    _TOKEN_VERSION = 4
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
        apply_dispositions: tuple[ReconciliationFinding, ...] = ()
        if apply:
            apply_dispositions, outcomes = self._apply_findings(findings, observed_at)
            findings.extend(apply_dispositions)
            # Derive progression *after* exact revalidation.  A stale primary
            # or sidecar is deliberately not an action attempt; returning the
            # scan-time terminal cursor in that case would hide the current,
            # independently actionable record from a caller that follows the
            # authenticated continuation.
            next_operation = self._retain_earliest_operation_source(
                operation_page,
                operation_cursor,
                next_operation,
                outcomes,
            )
            next_sidecar = self._retain_earliest_sidecar_source(
                sidecar_page,
                sidecar_cursor,
                next_sidecar,
                outcomes,
            )
            pending_operation = (
                consumed_operation < len(operation_page.entries)
                or next_operation is not None
            )
            pending_sidecar = (
                consumed_sidecar < len(sidecar_page.entries)
                or next_sidecar is not None
            )
            if next_priority is None and (pending_operation or pending_sidecar):
                next_priority = "operation" if pending_operation else "sidecar"
        token = self._encode_resume_token(
            next_manifest, next_operation, next_sidecar, next_pending, next_priority
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
    ) -> tuple[tuple[ReconciliationFinding, ...], tuple[_ApplyOutcome, ...]]:
        """Apply one normalized, revalidated action stream under one snapshot gate.

        A dry-run finding is not itself a mutation budget charge.  The shared
        budget advances only when this pass reaches the one repository action
        that can make progress for that finding.  In particular, a completed
        orphan sidecar is a sidecar-retirement action, never a pre-pass plus a
        second no-op primary action.
        """
        dispositions: list[ReconciliationFinding] = []
        outcomes: list[_ApplyOutcome] = []
        # Reconciliation uses aggregate admission only while it reloads exact
        # authority and applies a bounded report. Ordinary operations retain
        # their per-key lifecycle/CAS concurrency contract.
        with self.store._admission_barrier.aggregate_admission():
            remaining = self.lifecycle_limits.max_reconcile_actions
            for finding in findings:
                if remaining <= 0:
                    # Keep every safe finding after the action budget as an
                    # explicit unapplied outcome so cursor derivation cannot
                    # turn a bounded apply into a fictional terminal run.
                    if (
                        finding.status is ReconciliationStatus.SAFE
                        and finding.evidence_id is not None
                        and finding.evidence_digest is not None
                    ):
                        outcomes.append(
                            _ApplyOutcome(
                                finding,
                                self._finding_source(finding),
                                attempted=False,
                                completed=False,
                            )
                        )
                    continue
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
                    # A repository CAS was reached, so it consumes exactly
                    # one apply budget, but the current bytes are still
                    # reportable as a conflict rather than being silently
                    # classified as completion.
                    requires_retry, refreshed = self._reclassify_after_conflict(
                        finding, observed_at
                    )
                    if refreshed is not None:
                        dispositions.append(refreshed)
                    outcomes.append(
                        _ApplyOutcome(
                            finding,
                            self._finding_source(finding),
                            attempted=True,
                            completed=not requires_retry,
                            conflicted=True,
                        )
                    )
                    attempted = True
                else:
                    outcomes.append(
                        _ApplyOutcome(
                            finding,
                            self._finding_source(finding),
                            attempted=attempted,
                            completed=attempted,
                        )
                    )
                if attempted:
                    remaining -= 1
        return tuple(dispositions), tuple(outcomes)

    def _reclassify_after_conflict(
        self, finding: ReconciliationFinding, observed_at: datetime
    ) -> tuple[bool, ReconciliationFinding | None]:
        """Re-read exact current bytes after a charged CAS conflict.

        The old scan finding is not authority after a failed exact transition.
        A disappeared source may progress; any still-present primary or sidecar
        is classified from its current bytes and retained in the resume chain.
        This deliberately favors a bounded truthful replay over skipping a
        concurrent replacement that still carries recoverable debt.
        """
        assert finding.evidence_id is not None
        repository = self.store.lifecycle.operation_repository
        source = self._finding_source(finding)
        if source == "sidecar":
            raw = repository.get_reconciliation_checkpoint_raw(finding.evidence_id)
            if raw is None:
                return False, None
            return True, self._classify_sidecar(
                finding.evidence_id, raw, observed_at
            )
        raw = repository.get_raw(finding.evidence_id)
        if raw is None:
            return False, None
        current = self._classify_operation(finding.evidence_id, raw, observed_at)
        return True, self._block_primary_for_mismatched_sidecar(current, raw)

    @staticmethod
    def _finding_source(finding: ReconciliationFinding) -> str:
        """Classify the two evidence families that can own an apply action."""
        if finding.reason == "completed_reconciliation_checkpoint_orphan":
            return "sidecar"
        return "operation"

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

    @staticmethod
    def _first_unapplied_identifier(
        outcomes: tuple[_ApplyOutcome, ...],
        identifiers: tuple[str, ...],
        *,
        source: str,
    ) -> str | None:
        """Return the earliest scanned source item not advanced by apply."""
        outcome_by_id = {
            outcome.finding.evidence_id: outcome
            for outcome in outcomes
            if outcome.finding.evidence_id is not None and outcome.source == source
        }
        for identifier in identifiers:
            outcome = outcome_by_id.get(identifier)
            if outcome is not None and outcome.stale_or_unapplied:
                return identifier
        return None

    def _retain_earliest_operation_source(
        self,
        page: OperationPage,
        current: OperationCursor | None,
        computed: OperationCursor | None,
        outcomes: tuple[_ApplyOutcome, ...],
    ) -> OperationCursor | None:
        """Do not advance past stale primary evidence from an apply report."""
        identifiers = tuple(identifier for identifier, _raw in page.entries)
        retained = self._first_unapplied_identifier(
            outcomes, identifiers, source="operation"
        )
        if retained is None:
            return computed
        index = identifiers.index(retained)
        if index == 0:
            # ``None`` is terminal in an authenticated resume token.  A
            # conflict on the first item of a newly captured high-water page
            # must instead retain an explicit position before sequence one.
            if current is not None or not page.entry_next_cursors:
                return current
            first = page.entry_next_cursors[0]
            if (
                first.snapshot_high_water is None
                or first.next_sequence is None
            ):
                # Legacy lexical pages have no safe pre-first representation;
                # retain the caller's legacy cursor and conservatively replay.
                return current
            return OperationCursor.before_first(first.snapshot_high_water)
        if not page.entry_next_cursors:
            return current
        return page.entry_next_cursors[index - 1]

    def _retain_earliest_sidecar_source(
        self,
        page: ReconciliationCheckpointPage,
        current: ReconciliationCheckpointCursor | None,
        computed: ReconciliationCheckpointCursor | None,
        outcomes: tuple[_ApplyOutcome, ...],
    ) -> ReconciliationCheckpointCursor | None:
        """Do not retire a sidecar cursor until its exact apply outcome exists."""
        identifiers = tuple(identifier for identifier, _raw in page.entries)
        retained = self._first_unapplied_identifier(
            outcomes, identifiers, source="sidecar"
        )
        if retained is None:
            return computed
        index = identifiers.index(retained)
        if index == 0:
            # See the primary equivalent above: a first-source conflict must
            # never collapse to the terminal ``None`` token state.
            if current is not None or not page.entry_next_cursors:
                return current
            first = page.entry_next_cursors[0]
            if (
                first.snapshot_high_water is None
                or first.next_sequence is None
            ):
                return current
            return ReconciliationCheckpointCursor.before_first(
                first.snapshot_high_water
            )
        if not page.entry_next_cursors:
            return current
        return page.entry_next_cursors[index - 1]

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
        if not page.entries:
            return page.next_cursor
        if consumed == 0:
            return current
        if consumed < len(page.entries):
            if page.entry_next_cursors:
                return page.entry_next_cursors[consumed - 1]
            return current
        return page.next_cursor

    @staticmethod
    def _next_operation_cursor(
        page: OperationPage, current: OperationCursor | None, consumed: int
    ) -> OperationCursor | None:
        if not page.entries:
            return page.next_cursor
        if consumed == 0:
            return current
        if consumed < len(page.entries):
            if page.entry_next_cursors:
                return page.entry_next_cursors[consumed - 1]
            return current
        return page.next_cursor

    @staticmethod
    def _next_sidecar_cursor(
        page: ReconciliationCheckpointPage,
        current: ReconciliationCheckpointCursor | None,
        consumed: int,
    ) -> ReconciliationCheckpointCursor | None:
        if not page.entries:
            return page.next_cursor
        if consumed == 0:
            return current
        if consumed < len(page.entries):
            if page.entry_next_cursors:
                return page.entry_next_cursors[consumed - 1]
            return current
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
                "manifest": self._encode_manifest_cursor(manifest_cursor),
                "operation": (
                    self._encode_sequence_cursor(operation_cursor, "operation_id")
                ),
                "sidecar": (
                    self._encode_sequence_cursor(sidecar_cursor, "operation_id")
                ),
                "pending": self._encode_sequence_cursor(pending_cursor, "name"),
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
                self._decode_manifest_cursor(manifest, version=version),
                self._decode_sequence_cursor(
                    operation, OperationCursor, "operation_id", version=version
                ),
                self._decode_sequence_cursor(
                    sidecar,
                    ReconciliationCheckpointCursor,
                    "operation_id",
                    version=version,
                ),
                self._decode_sequence_cursor(
                    pending, PendingControlCursor, "name", version=version
                ),
                priority,
            )
        except (InvalidTag, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("reconciliation resume token is invalid") from exc

    @staticmethod
    def _encode_manifest_cursor(cursor: ManifestCursor | None) -> dict[str, int | str] | None:
        """Serialize the versioned generation-bound manifest cursor."""
        if cursor is None:
            return None
        if cursor.snapshot_high_water is None or cursor.next_sequence is None:
            return {"key": cursor.key, "restart": 1}
        return {
            "key": cursor.key,
            "high_water": cursor.snapshot_high_water,
            "next_sequence": cursor.next_sequence,
        }

    @staticmethod
    def _decode_manifest_cursor(
        value: object, *, version: int
    ) -> ManifestCursor | None:
        """Map v1/v2 lexical state into an explicit safe snapshot restart."""
        if value is None:
            return None
        if version in {1, 2}:
            if not isinstance(value, str):
                raise ValueError("legacy manifest cursor is invalid")
            # An authenticated lexical token cannot prove membership under a
            # new high-water index.  Restarting this source from a current
            # snapshot is intentionally conservative: it can repeat a report
            # but never skips pre-cursor insertion.
            return ManifestCursor(value)
        if not isinstance(value, dict):
            raise ValueError("manifest cursor is invalid")
        if set(value) == {"key", "restart"} and value["restart"] == 1:
            return ManifestCursor(value["key"])
        if set(value) != {"key", "high_water", "next_sequence"}:
            raise ValueError("manifest cursor is invalid")
        return ManifestCursor(
            value["key"],
            snapshot_high_water=value["high_water"],
            next_sequence=value["next_sequence"],
        )

    @staticmethod
    def _encode_sequence_cursor(
        cursor: OperationCursor | ReconciliationCheckpointCursor | PendingControlCursor | None,
        field: str,
    ) -> dict[str, int | str] | None:
        """Serialize a high-water evidence cursor without exposing its order."""
        if cursor is None:
            return None
        value = getattr(cursor, field)
        if cursor.snapshot_high_water is None or cursor.next_sequence is None:
            return {field: value, "restart": 1}
        return {
            field: value,
            "high_water": cursor.snapshot_high_water,
            "next_sequence": cursor.next_sequence,
        }

    @staticmethod
    def _decode_sequence_cursor(
        value: object,
        cursor_type: type[OperationCursor]
        | type[ReconciliationCheckpointCursor]
        | type[PendingControlCursor],
        field: str,
        *,
        version: int,
    ) -> OperationCursor | ReconciliationCheckpointCursor | PendingControlCursor | None:
        """Restart legacy lexical sources conservatively under their index."""
        if value is None:
            return None
        if version in {1, 2, 3}:
            if not isinstance(value, str):
                raise ValueError("legacy reconciliation cursor is invalid")
            return cursor_type(value)
        if not isinstance(value, dict):
            raise ValueError("reconciliation cursor is invalid")
        if set(value) == {field, "restart"} and value["restart"] == 1:
            return cursor_type(value[field])
        if set(value) != {field, "high_water", "next_sequence"}:
            raise ValueError("reconciliation cursor is invalid")
        return cursor_type(
            value[field],
            snapshot_high_water=value["high_water"],
            next_sequence=value["next_sequence"],
        )

    def _token_key(self) -> bytes:
        """Derive the token-only AEAD key without reusing manifest signatures."""
        return hmac.new(
            self.store._manifest_key(),
            self._TOKEN_DOMAIN + b"chacha20-poly1305-key",
            hashlib.sha256,
        ).digest()
