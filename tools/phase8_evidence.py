#!/usr/bin/env python3
"""Strict, bounded evidence envelopes shared by Phase 8 qualification tools.

Evidence records describe one independently produced qualification class.  They
are deliberately not a release decision: only the final release verifier may
combine separately validated envelopes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


EVIDENCE_SCHEMA = "cacheness-phase8-evidence-v1"
EVIDENCE_CLASSES = (
    "deterministic",
    "packaging",
    "platform",
    "coverage",
    "structural",
    "controlled_performance",
    "live_services",
)
TERMINAL_STATUSES = frozenset({"PASS", "UNAVAILABLE", "NOT_QUALIFIED"})
CLAIM_CATEGORIES = ("integrity", "recovery", "progress", "performance")
CLAIM_STATES = frozenset(
    {"EVIDENCED", "NOT_QUALIFIED", "UNAVAILABLE", "DIAGNOSTIC"}
)
QUALIFIED_SUBJECTS = (
    "AuthorityLifecycleEngine",
    "BlobStore",
    "ObstoreGenerationIO",
    "UnifiedCache",
)

MAX_EVIDENCE_BYTES = 256 * 1024
MAX_TEXT_LENGTH = 4_096
MAX_COLLECTION_ITEMS = 32
MAX_OBJECT_KEYS = 16

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_TEXT = re.compile(
    r"://|access[_-]?key|credential|password|secret|token|private[_-]?key",
    re.IGNORECASE,
)
_ALLOWED_ENVELOPE_KEYS = frozenset(
    {
        "schema",
        "evidence_class",
        "status",
        "revision",
        "source_digest",
        "generated_at_utc",
        "payload",
    }
)
_COMMON_PAYLOAD_KEYS = frozenset(
    {"result", "claim_categories", "non_qualifying_classes", "subjects"}
)
_DETERMINISTIC_PAYLOAD_KEYS = _COMMON_PAYLOAD_KEYS | {"command"}
_RESULTS_BY_STATUS = {
    "PASS": frozenset({"passed"}),
    "UNAVAILABLE": frozenset({"unavailable"}),
    "NOT_QUALIFIED": frozenset(
        {
            "failed",
            "incomplete",
            "skipped",
            "source_dirty",
            "source_changed",
            "timed_out",
        }
    ),
}


class EvidenceValidationError(ValueError):
    """Raised when untrusted qualification evidence is not an exact safe shape."""


@dataclass(frozen=True)
class EvidenceEnvelope:
    """One immutable, class-scoped Phase 8 evidence result."""

    evidence_class: str
    status: str
    revision: str
    source_digest: str
    generated_at_utc: str
    payload: Mapping[str, object]

    def to_mapping(self) -> dict[str, object]:
        """Return the exact canonical JSON-ready envelope shape."""
        return {
            "schema": EVIDENCE_SCHEMA,
            "evidence_class": self.evidence_class,
            "status": self.status,
            "revision": self.revision,
            "source_digest": self.source_digest,
            "generated_at_utc": self.generated_at_utc,
            "payload": dict(self.payload),
        }


def _validate_safe_text(value: object, *, field: str) -> str:
    """Return safe bounded text or reject the whole evidence document."""
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT_LENGTH:
        raise EvidenceValidationError(f"invalid {field}")
    if any(ord(character) < 32 for character in value):
        raise EvidenceValidationError(f"unsafe {field}")
    if _FORBIDDEN_TEXT.search(value):
        raise EvidenceValidationError(f"unsafe {field}")
    return value


def _validate_timestamp(value: object) -> str:
    timestamp = _validate_safe_text(value, field="generated_at_utc")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise EvidenceValidationError("invalid generated_at_utc") from error
    if parsed.tzinfo is None:
        raise EvidenceValidationError("invalid generated_at_utc")
    return timestamp


def _validate_text_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > MAX_COLLECTION_ITEMS:
        raise EvidenceValidationError(f"invalid {field}")
    return [_validate_safe_text(item, field=field) for item in value]


def _validate_claim_categories(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(CLAIM_CATEGORIES):
        raise EvidenceValidationError("invalid claim_categories")
    claims: dict[str, str] = {}
    for category in CLAIM_CATEGORIES:
        state = value[category]
        if not isinstance(state, str) or state not in CLAIM_STATES:
            raise EvidenceValidationError("invalid claim_categories")
        claims[category] = state
    return claims


def _validate_payload(
    evidence_class: str, status: str, payload: object
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or len(payload) > MAX_OBJECT_KEYS:
        raise EvidenceValidationError("invalid payload")
    allowed = (
        _DETERMINISTIC_PAYLOAD_KEYS
        if evidence_class == "deterministic"
        else _COMMON_PAYLOAD_KEYS
    )
    if set(payload) != allowed:
        raise EvidenceValidationError("payload violates the exact allow-list")

    result = payload.get("result")
    if not isinstance(result, str) or result not in _RESULTS_BY_STATUS[status]:
        raise EvidenceValidationError("terminal status contradicts payload result")

    claims = _validate_claim_categories(payload.get("claim_categories"))
    non_qualifying_classes = _validate_text_list(
        payload.get("non_qualifying_classes"), field="non_qualifying_classes"
    )
    expected_non_qualifying = [
        candidate for candidate in EVIDENCE_CLASSES if candidate != evidence_class
    ]
    if non_qualifying_classes != expected_non_qualifying:
        raise EvidenceValidationError("invalid non_qualifying_classes")

    subjects = _validate_text_list(payload.get("subjects"), field="subjects")
    if sorted(subjects) != sorted(QUALIFIED_SUBJECTS) or len(set(subjects)) != len(
        subjects
    ):
        raise EvidenceValidationError("invalid qualified subjects")

    validated: dict[str, object] = {
        "result": result,
        "claim_categories": claims,
        "non_qualifying_classes": non_qualifying_classes,
        "subjects": subjects,
    }
    if evidence_class == "deterministic":
        command = _validate_text_list(payload.get("command"), field="command")
        if command != ["tools/verify_phase071_contracts.py", "--all"]:
            raise EvidenceValidationError("invalid deterministic command")
        validated["command"] = command

    if status == "PASS":
        if claims["performance"] != "NOT_QUALIFIED":
            raise EvidenceValidationError("performance cannot be promoted by this evidence")
        if evidence_class != "deterministic":
            raise EvidenceValidationError("only implemented evidence producers may pass")
    return validated


def envelope_from_mapping(value: Mapping[str, object]) -> EvidenceEnvelope:
    """Validate one decoded envelope without accepting extension keys or aliases."""
    if set(value) != _ALLOWED_ENVELOPE_KEYS:
        raise EvidenceValidationError("envelope violates the exact allow-list")
    if value.get("schema") != EVIDENCE_SCHEMA:
        raise EvidenceValidationError("invalid evidence schema")

    evidence_class = value.get("evidence_class")
    if not isinstance(evidence_class, str) or evidence_class not in EVIDENCE_CLASSES:
        raise EvidenceValidationError("invalid evidence_class")
    status = value.get("status")
    if not isinstance(status, str) or status not in TERMINAL_STATUSES:
        raise EvidenceValidationError("invalid terminal status")

    revision = value.get("revision")
    source_digest = value.get("source_digest")
    if not isinstance(revision, str) or not _REVISION_PATTERN.fullmatch(revision):
        raise EvidenceValidationError("invalid revision")
    if not isinstance(source_digest, str) or not _DIGEST_PATTERN.fullmatch(source_digest):
        raise EvidenceValidationError("invalid source_digest")

    generated_at_utc = _validate_timestamp(value.get("generated_at_utc"))
    payload = _validate_payload(evidence_class, status, value.get("payload"))
    return EvidenceEnvelope(
        evidence_class=evidence_class,
        status=status,
        revision=revision,
        source_digest=source_digest,
        generated_at_utc=generated_at_utc,
        payload=payload,
    )


def make_envelope(
    *,
    evidence_class: str,
    status: str,
    revision: str,
    source_digest: str,
    payload: Mapping[str, object],
    generated_at_utc: str | None = None,
) -> EvidenceEnvelope:
    """Build a validated envelope with a UTC generation time."""
    return envelope_from_mapping(
        {
            "schema": EVIDENCE_SCHEMA,
            "evidence_class": evidence_class,
            "status": status,
            "revision": revision,
            "source_digest": source_digest,
            "generated_at_utc": generated_at_utc
            if generated_at_utc is not None
            else datetime.now(UTC).isoformat(),
            "payload": dict(payload),
        }
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceValidationError("duplicate JSON key")
        result[key] = value
    return result


def load_envelope(path: Path) -> EvidenceEnvelope:
    """Read one bounded canonical JSON envelope and validate its full shape."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise EvidenceValidationError("evidence cannot be read") from error
    if not raw or len(raw) > MAX_EVIDENCE_BYTES:
        raise EvidenceValidationError("evidence exceeds byte bound")
    try:
        decoded = raw.decode("utf-8")
        parsed = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, EvidenceValidationError) as error:
        raise EvidenceValidationError("evidence is not safe canonical JSON") from error
    if not isinstance(parsed, Mapping):
        raise EvidenceValidationError("evidence root must be an object")
    envelope = envelope_from_mapping(parsed)
    if _canonical_json(envelope.to_mapping()).encode("utf-8") != raw:
        raise EvidenceValidationError("evidence is not canonical JSON")
    return envelope


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"


def write_envelope(path: Path, envelope: EvidenceEnvelope) -> None:
    """Atomically replace a destination only with validated canonical evidence."""
    validated = envelope_from_mapping(envelope.to_mapping())
    serialized = _canonical_json(validated.to_mapping()).encode("utf-8")
    if len(serialized) > MAX_EVIDENCE_BYTES:
        raise EvidenceValidationError("evidence exceeds byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        raise EvidenceValidationError("evidence cannot be written atomically") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def relevant_source_digest(root: Path, paths: Sequence[str]) -> str:
    """Hash a reviewed source inventory with paths and contents in stable order."""
    digest = hashlib.sha256()
    root = root.resolve()
    seen: set[Path] = set()
    for raw_path in paths:
        relative_path = Path(raw_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise EvidenceValidationError("invalid relevant source path")
        candidate = root / relative_path
        if candidate.is_symlink() or not candidate.exists():
            raise EvidenceValidationError("invalid relevant source path")
        files = [candidate] if candidate.is_file() else sorted(candidate.rglob("*"))
        for path in files:
            if path.is_dir():
                continue
            if path.is_symlink() or not path.is_file():
                raise EvidenceValidationError("invalid relevant source path")
            resolved = path.resolve()
            if root not in (resolved, *resolved.parents) or resolved in seen:
                continue
            seen.add(resolved)
            relative = resolved.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            with resolved.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    digest.update(chunk)
    if not seen:
        raise EvidenceValidationError("relevant source inventory is empty")
    return digest.hexdigest()
