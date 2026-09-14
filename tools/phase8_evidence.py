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
import hmac
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
CLAIM_STATES = frozenset({"EVIDENCED", "NOT_QUALIFIED", "UNAVAILABLE", "DIAGNOSTIC"})
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
_PLATFORM_PAYLOAD_KEYS = _COMMON_PAYLOAD_KEYS | {
    "expected_os",
    "actual_os",
    "expected_python_minor",
    "actual_python_minor",
    "feature_profile",
    "role",
    "advisory",
    "command_profile",
    "backlog_phase",
    "reason",
}
_PACKAGING_PAYLOAD_KEYS = _COMMON_PAYLOAD_KEYS | {
    "wheel_sha256",
    "python",
    "platform",
    "probes",
    "optional_groups",
    "compatibility",
    "non_live_groups",
}
_STRUCTURAL_PAYLOAD_KEYS = _COMMON_PAYLOAD_KEYS | {"environment", "observations"}
_STRUCTURAL_COUNTER_KEYS = frozenset(
    {
        "authority_pages",
        "authority_reads",
        "authority_writes",
        "participant_head",
        "participant_open",
        "participant_delete",
        "participant_list",
    }
)
_STRUCTURAL_OBSERVATION_KEYS = frozenset(
    {
        "operation",
        "seeded_entries",
        "page_size",
        "work_cap",
        "selected_entries",
        "workload_bytes",
        "peak_rss_bytes",
        "counters",
    }
)
_STRUCTURAL_ENVIRONMENT_KEYS = frozenset({"os", "rss_unit"})
_MAX_STRUCTURAL_OBSERVATIONS = 64
_PACKAGING_OPTIONAL_GROUPS = (
    "recommended",
    "dataframes",
    "tensorflow",
    "s3",
    "postgresql",
    "cloud",
)
_PACKAGING_NON_LIVE_GROUPS = ("s3", "postgresql", "cloud")
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
_CLAIM_STATES_BY_STATUS = {
    "PASS": {
        "integrity": "EVIDENCED",
        "recovery": "EVIDENCED",
        "progress": "EVIDENCED",
        "performance": "NOT_QUALIFIED",
    },
    "UNAVAILABLE": dict.fromkeys(CLAIM_CATEGORIES, "UNAVAILABLE"),
    "NOT_QUALIFIED": dict.fromkeys(CLAIM_CATEGORIES, "NOT_QUALIFIED"),
}
_PACKAGING_PASS_CLAIMS = dict.fromkeys(CLAIM_CATEGORIES, "NOT_QUALIFIED")


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


def is_qualification_evidence(
    envelope: EvidenceEnvelope, required_evidence_class: str
) -> bool:
    """Return whether one envelope can satisfy one matching required class.

    This deliberately evaluates one class only.  Release tooling, not a class
    producer, remains responsible for combining independently validated input.
    """
    if required_evidence_class not in EVIDENCE_CLASSES:
        raise EvidenceValidationError("invalid required evidence_class")
    return (
        envelope.evidence_class == required_evidence_class and envelope.status == "PASS"
    )


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


def _expected_claim_categories(evidence_class: str, status: str) -> dict[str, str]:
    """Return the exact claim boundary for one implemented evidence producer."""
    if status == "PASS" and evidence_class == "packaging":
        return _PACKAGING_PASS_CLAIMS
    return _CLAIM_STATES_BY_STATUS[status]


def _validate_packaging_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate bounded package evidence without accepting resolver diagnostics."""
    wheel_sha256 = payload.get("wheel_sha256")
    if not isinstance(wheel_sha256, str) or not _DIGEST_PATTERN.fullmatch(wheel_sha256):
        raise EvidenceValidationError("invalid wheel_sha256")
    python = _validate_safe_text(payload.get("python"), field="python")
    platform = _validate_safe_text(payload.get("platform"), field="platform")
    probes = _validate_text_list(payload.get("probes"), field="probes")
    if len(set(probes)) != len(probes):
        raise EvidenceValidationError("invalid probes")
    optional_groups = _validate_text_list(
        payload.get("optional_groups"), field="optional_groups"
    )
    if optional_groups != list(_PACKAGING_OPTIONAL_GROUPS):
        raise EvidenceValidationError("invalid optional_groups")
    compatibility = _validate_text_list(
        payload.get("compatibility"), field="compatibility"
    )
    expected_compatibility_prefixes = [
        f"{group}:" for group in _PACKAGING_OPTIONAL_GROUPS
    ]
    if len(compatibility) != len(expected_compatibility_prefixes) or any(
        not value.startswith(prefix)
        or value.removeprefix(prefix) not in {"COMPATIBLE", "INCOMPATIBLE"}
        for value, prefix in zip(compatibility, expected_compatibility_prefixes)
    ):
        raise EvidenceValidationError("invalid compatibility")
    non_live_groups = _validate_text_list(
        payload.get("non_live_groups"), field="non_live_groups"
    )
    if non_live_groups != list(_PACKAGING_NON_LIVE_GROUPS):
        raise EvidenceValidationError("invalid non_live_groups")
    return {
        "wheel_sha256": wheel_sha256,
        "python": python,
        "platform": platform,
        "probes": probes,
        "optional_groups": optional_groups,
        "compatibility": compatibility,
        "non_live_groups": non_live_groups,
    }


def _validate_structural_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate finite call-count and RSS facts without accepting timing data."""

    environment = payload.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != _STRUCTURAL_ENVIRONMENT_KEYS
    ):
        raise EvidenceValidationError("invalid structural environment")
    operating_system = _validate_safe_text(environment.get("os"), field="os")
    if environment.get("rss_unit") != "bytes":
        raise EvidenceValidationError("invalid structural RSS unit")

    observations = payload.get("observations")
    if (
        not isinstance(observations, list)
        or not observations
        or len(observations) > _MAX_STRUCTURAL_OBSERVATIONS
    ):
        raise EvidenceValidationError("invalid structural observations")
    validated_observations: list[dict[str, object]] = []
    for observation in observations:
        if (
            not isinstance(observation, Mapping)
            or set(observation) != _STRUCTURAL_OBSERVATION_KEYS
        ):
            raise EvidenceValidationError("invalid structural observation")
        operation = _validate_safe_text(observation.get("operation"), field="operation")
        integers: dict[str, int] = {}
        for field_name in (
            "seeded_entries",
            "page_size",
            "work_cap",
            "selected_entries",
            "workload_bytes",
            "peak_rss_bytes",
        ):
            value = observation.get(field_name)
            if type(value) is not int or value < 0:
                raise EvidenceValidationError("invalid structural observation")
            integers[field_name] = value
        if (
            integers["page_size"] <= 0
            or integers["work_cap"] <= 0
            or integers["page_size"] > integers["work_cap"]
            or integers["selected_entries"]
            > min(
                integers["seeded_entries"],
                integers["page_size"],
                integers["work_cap"],
            )
        ):
            raise EvidenceValidationError("invalid structural work bounds")
        counters = observation.get("counters")
        if (
            not isinstance(counters, Mapping)
            or set(counters) != _STRUCTURAL_COUNTER_KEYS
        ):
            raise EvidenceValidationError("invalid structural counters")
        validated_counters: dict[str, int] = {}
        for counter_name in sorted(_STRUCTURAL_COUNTER_KEYS):
            value = counters[counter_name]
            if type(value) is not int or value < 0 or value > 4_096:
                raise EvidenceValidationError("invalid structural counters")
            validated_counters[counter_name] = value
        validated_observations.append(
            {"operation": operation, **integers, "counters": validated_counters}
        )
    return {
        "environment": {"os": operating_system, "rss_unit": "bytes"},
        "observations": validated_observations,
    }


def _validate_platform_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate one bounded, runtime-bound platform qualification row."""
    expected_os = _validate_safe_text(payload.get("expected_os"), field="expected_os")
    actual_os = _validate_safe_text(payload.get("actual_os"), field="actual_os")
    expected_python_minor = _validate_safe_text(
        payload.get("expected_python_minor"), field="expected_python_minor"
    )
    actual_python_minor = _validate_safe_text(
        payload.get("actual_python_minor"), field="actual_python_minor"
    )
    feature_profile = _validate_safe_text(
        payload.get("feature_profile"), field="feature_profile"
    )
    role = _validate_safe_text(payload.get("role"), field="role")
    command_profile = _validate_safe_text(
        payload.get("command_profile"), field="command_profile"
    )
    backlog_phase = _validate_safe_text(
        payload.get("backlog_phase"), field="backlog_phase"
    )
    reason = _validate_safe_text(payload.get("reason"), field="reason")
    advisory = payload.get("advisory")
    if not isinstance(advisory, bool):
        raise EvidenceValidationError("invalid advisory")
    return {
        "expected_os": expected_os,
        "actual_os": actual_os,
        "expected_python_minor": expected_python_minor,
        "actual_python_minor": actual_python_minor,
        "feature_profile": feature_profile,
        "role": role,
        "advisory": advisory,
        "command_profile": command_profile,
        "backlog_phase": backlog_phase,
        "reason": reason,
    }


def _validate_payload(
    evidence_class: str, status: str, payload: object
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or len(payload) > MAX_OBJECT_KEYS:
        raise EvidenceValidationError("invalid payload")
    allowed = {
        "deterministic": _DETERMINISTIC_PAYLOAD_KEYS,
        "platform": _PLATFORM_PAYLOAD_KEYS,
        "packaging": _PACKAGING_PAYLOAD_KEYS,
        "structural": _STRUCTURAL_PAYLOAD_KEYS,
    }.get(evidence_class, _COMMON_PAYLOAD_KEYS)
    if set(payload) != allowed:
        raise EvidenceValidationError("payload violates the exact allow-list")

    result = payload.get("result")
    if not isinstance(result, str) or result not in _RESULTS_BY_STATUS[status]:
        raise EvidenceValidationError("terminal status contradicts payload result")

    claims = _validate_claim_categories(payload.get("claim_categories"))
    if claims != _expected_claim_categories(evidence_class, status):
        raise EvidenceValidationError("terminal status contradicts claim_categories")
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

    if evidence_class == "packaging":
        validated.update(_validate_packaging_payload(payload))

    if evidence_class == "platform":
        validated.update(_validate_platform_payload(payload))

    if evidence_class == "structural":
        validated.update(_validate_structural_payload(payload))

    if status == "PASS" and evidence_class not in {
        "deterministic",
        "packaging",
        "platform",
        "coverage",
        "structural",
    }:
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
    if not isinstance(source_digest, str) or not _DIGEST_PATTERN.fullmatch(
        source_digest
    ):
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
        with path.open("rb") as evidence_file:
            raw = evidence_file.read(MAX_EVIDENCE_BYTES + 1)
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
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


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
        raise EvidenceValidationError(
            "evidence cannot be written atomically"
        ) from error
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


def validate_source_identity(
    envelope: EvidenceEnvelope,
    *,
    revision: str,
    root: Path,
    paths: Sequence[str],
) -> None:
    """Reject evidence that does not match one exact reviewed source identity."""
    if not _REVISION_PATTERN.fullmatch(revision):
        raise EvidenceValidationError("invalid required revision")
    if not hmac.compare_digest(envelope.revision, revision):
        raise EvidenceValidationError(
            "evidence revision does not match required revision"
        )
    expected_digest = relevant_source_digest(root, paths)
    if not hmac.compare_digest(envelope.source_digest, expected_digest):
        raise EvidenceValidationError(
            "evidence source digest does not match reviewed sources"
        )
