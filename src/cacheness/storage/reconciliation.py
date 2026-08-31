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
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobManifestUnauthenticatedError,
    CacheManifestIntegrityError,
    CacheManifestUnsupportedVersionError,
    CacheStorageError,
)

from .integrity import verify_hmac_sha256
from .manifest import BlobManifestV1
from .manifest_repository import ManifestCursor, ManifestPage
from .operation_record import OperationCheckpoint, OperationTransition
from .operation_repository import OperationCursor, OperationPage
from .path_security import resolve_managed_locator


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


class _Reconciler:
    """Private coordinator for bounded, non-deserializing evidence analysis."""

    _TOKEN_DOMAIN = b"cacheness.reconciliation.cursor.v1\x00"

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

        manifest_cursor, operation_cursor = self._decode_resume_token(resume_token)
        manifest_page = self._manifest_page(manifest_cursor)
        operation_page = self._operation_page(operation_cursor)
        findings: list[ReconciliationFinding] = []
        remaining = self.lifecycle_limits.max_reconcile_actions
        manifest_seen = 0
        operation_seen = 0
        consumed_manifest = 0
        consumed_operation = 0

        # A fixed source order makes reports stable.  Each repository page is
        # independently bounded; a combined action budget controls reporting
        # and any later apply work.
        for key, raw in manifest_page.entries:
            if remaining == 0:
                break
            findings.append(self._classify_manifest(key, raw))
            remaining -= 1
            manifest_seen += 1
            consumed_manifest += 1
        for operation_id, raw in operation_page.entries:
            if remaining == 0:
                break
            findings.append(self._classify_operation(operation_id, raw, observed_at))
            remaining -= 1
            operation_seen += 1
            consumed_operation += 1

        next_manifest = self._next_manifest_cursor(
            manifest_page, consumed_manifest
        )
        next_operation = self._next_operation_cursor(
            operation_page, consumed_operation
        )
        token = self._encode_resume_token(next_manifest, next_operation)
        return ReconciliationReport(
            findings=tuple(findings),
            resume_token=token,
            applied=apply,
            manifest_records_seen=manifest_seen,
            operation_records_seen=operation_seen,
        )

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

    @staticmethod
    def _next_manifest_cursor(
        page: ManifestPage, consumed: int
    ) -> ManifestCursor | None:
        if consumed < len(page.entries):
            return ManifestCursor(page.entries[consumed - 1][0]) if consumed else None
        return page.next_cursor

    @staticmethod
    def _next_operation_cursor(
        page: OperationPage, consumed: int
    ) -> OperationCursor | None:
        if consumed < len(page.entries):
            return OperationCursor(page.entries[consumed - 1][0]) if consumed else None
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
        return ReconciliationFinding(
            ReconciliationStatus.BLOCKED,
            ReconciliationAction.REPORT_ONLY,
            "operation_authority_ambiguous",
            **base,
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
    ) -> str | None:
        if manifest_cursor is None and operation_cursor is None:
            return None
        # The token is encrypted-and-authenticated with the existing manifest
        # key so reports do not disclose backend logical-key cursors.
        key = self.store._manifest_key()
        payload = json.dumps(
            {
                "manifest": None if manifest_cursor is None else manifest_cursor.key,
                "operation": (
                    None if operation_cursor is None else operation_cursor.operation_id
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        mask = self._mask(key, len(payload))
        encrypted = bytes(left ^ right for left, right in zip(payload, mask))
        signature = hmac.new(key, self._TOKEN_DOMAIN + encrypted, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(signature + encrypted).decode("ascii")

    def _decode_resume_token(
        self, token: str | None
    ) -> tuple[ManifestCursor | None, OperationCursor | None]:
        if token is None:
            return None, None
        if not isinstance(token, str) or not token:
            raise ValueError("reconciliation resume token must be a non-empty string")
        try:
            packed = base64.urlsafe_b64decode(token.encode("ascii"))
            signature, encrypted = packed[:32], packed[32:]
            key = self.store._manifest_key()
            expected = hmac.new(
                key, self._TOKEN_DOMAIN + encrypted, hashlib.sha256
            ).digest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("reconciliation resume token is invalid")
            mask = self._mask(key, len(encrypted))
            payload = bytes(left ^ right for left, right in zip(encrypted, mask))
            decoded = json.loads(payload)
            if set(decoded) != {"manifest", "operation"}:
                raise ValueError("reconciliation resume token is malformed")
            manifest = decoded["manifest"]
            operation = decoded["operation"]
            return (
                None if manifest is None else ManifestCursor(manifest),
                None if operation is None else OperationCursor(operation),
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("reconciliation resume token is invalid") from exc

    @classmethod
    def _mask(cls, key: bytes, length: int) -> bytes:
        blocks = []
        counter = 0
        while sum(len(block) for block in blocks) < length:
            blocks.append(
                hashlib.sha256(cls._TOKEN_DOMAIN + key + counter.to_bytes(4)).digest()
            )
            counter += 1
        return b"".join(blocks)[:length]
