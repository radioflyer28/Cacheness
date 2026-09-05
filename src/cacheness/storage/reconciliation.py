"""Bounded, authority-only BlobStore reconciliation reports.

The reconciler inventories durable LifecycleAuthority evidence; payload names
are never provenance and remain unavailable to repair decisions.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheStorageError,
)

from .lifecycle_authority import ReconciliationSnapshot, ReconciliationWork


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
            "requires_confirmation="
            f"{counts[ReconciliationStatus.REQUIRES_CONFIRMATION]}, "
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
        """Return the explicit versioned operator machine view."""
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
                    completed = self._apply(work)
                    applied_actions += int(completed)
                    finding = self._with_apply_outcome(finding, completed)
                    if run_token is not None:
                        self.authority.checkpoint_reconciliation(
                            run_token,
                            work,
                            state="completed" if completed else "blocked",
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

    def _apply(self, work: ReconciliationWork) -> bool:
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
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        signature = hmac.new(
            self.store._authority_manifest_key(initialize_new_store=False),
            self._TOKEN_DOMAIN + encoded,
            hashlib.sha256,
        ).hexdigest()
        return base64.urlsafe_b64encode(
            encoded + b"." + signature.encode("ascii")
        ).decode("ascii")

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
