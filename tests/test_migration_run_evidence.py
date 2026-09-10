"""Security and recovery contracts for explicit offline migration evidence."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cacheness.error_handling import (
    CacheBlobMigrationEvidenceError,
    CacheBlobMigrationEvidenceMismatchError,
)
from cacheness.storage.migration_authority import AuthorityIdentitySnapshot
from cacheness.storage.migration_evidence import (
    MaintenanceEvidenceState,
    MaintenanceEvidenceStore,
    MaintenanceRunEvidence,
    StoppedWorkerAcknowledgement,
    decode_maintenance_evidence,
    encode_maintenance_evidence,
)


class _SentinelKeyProvider:
    """Return a test-only key whose bytes must never be rendered or logged."""

    key = b"migration-evidence-secret-sentinel"

    def get_key(self) -> bytes:
        return self.key


def _identity(*, revision: int = 4, store_id: str = "source-store") -> AuthorityIdentitySnapshot:
    return AuthorityIdentitySnapshot(
        store_id=store_id,
        revision=revision,
        authority_kind="memory",
        capability="same_process",
        schema_version=1,
    )


def _evidence(*, state: MaintenanceEvidenceState = MaintenanceEvidenceState.INSPECTED) -> MaintenanceRunEvidence:
    source = _identity()
    return MaintenanceRunEvidence(
        evidence_version=1,
        run_id="maintenance-run-001",
        plan_digest="a" * 64,
        source_identity=source,
        source_revision=source.revision,
        destination_identity=_identity(store_id="destination-store"),
        destination_revision=4,
        acknowledgement=StoppedWorkerAcknowledgement(
            run_id="maintenance-run-001",
            plan_digest="a" * 64,
            source_identity=source,
            source_revision=source.revision,
        ),
        state=state,
        completed_steps=("inspect",),
    )


def test_evidence_store_round_trips_canonically_and_rejects_forgery(tmp_path: Path) -> None:
    """A run is read only after canonical bytes authenticate with its provider."""
    provider = _SentinelKeyProvider()
    evidence = _evidence()
    store = MaintenanceEvidenceStore(
        work_directory=tmp_path / "maintenance",
        run_id=evidence.run_id,
        key_provider=provider,
    )

    created = store.create(evidence)
    assert created == evidence
    assert store.load() == evidence

    raw = store.read_bytes()
    assert decode_maintenance_evidence(raw, provider.get_key()) == evidence
    assert encode_maintenance_evidence(evidence, provider.get_key()) == raw

    store.evidence_path.write_bytes(raw.replace(b"inspected", b"inSpected"))
    with pytest.raises(CacheBlobMigrationEvidenceError, match="authentication"):
        store.load()


def test_evidence_store_rejects_unsafe_run_paths_and_symlink_escapes(tmp_path: Path) -> None:
    """Evidence remains underneath one separately controlled operator directory."""
    provider = _SentinelKeyProvider()
    evidence = _evidence()

    with pytest.raises(CacheBlobMigrationEvidenceError, match="run_id"):
        MaintenanceEvidenceStore(tmp_path / "work", "../escape", provider)

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked-work"
    linked.symlink_to(outside, target_is_directory=True)
    store = MaintenanceEvidenceStore(linked, evidence.run_id, provider)
    with pytest.raises(CacheBlobMigrationEvidenceError, match="symlink"):
        store.create(evidence)
    assert not list(outside.iterdir())


def test_checkpoint_requires_exact_previous_bytes_and_legal_state_transition(tmp_path: Path) -> None:
    """Maintenance evidence cannot silently overwrite an unknown or illegal state."""
    provider = _SentinelKeyProvider()
    evidence = _evidence()
    store = MaintenanceEvidenceStore(tmp_path / "maintenance", evidence.run_id, provider)
    store.create(evidence)

    planned = _evidence(state=MaintenanceEvidenceState.PLANNED)
    with pytest.raises(CacheBlobMigrationEvidenceMismatchError, match="exact previous"):
        store.checkpoint(b"not-the-current-evidence", planned)

    with pytest.raises(CacheBlobMigrationEvidenceMismatchError, match="transition"):
        store.checkpoint(store.read_bytes(), _evidence(state=MaintenanceEvidenceState.ACTIVATED))

    store.checkpoint(store.read_bytes(), planned)
    assert store.load().state is MaintenanceEvidenceState.PLANNED


def test_evidence_never_renders_or_logs_key_material(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Error channels preserve only a non-secret signing identity fingerprint."""
    provider = _SentinelKeyProvider()
    evidence = _evidence()
    raw = encode_maintenance_evidence(evidence, provider.get_key())
    rendered = evidence.render_report()
    assert provider.key.decode("ascii") not in raw.decode("utf-8")
    assert provider.key.decode("ascii") not in rendered

    store = MaintenanceEvidenceStore(tmp_path / "maintenance", evidence.run_id, provider)
    with caplog.at_level(logging.ERROR):
        store.create(evidence)
        store.evidence_path.write_bytes(b"{")
        with pytest.raises(CacheBlobMigrationEvidenceError) as raised:
            store.load()
    assert provider.key.decode("ascii") not in str(raised.value)
    assert provider.key.decode("ascii") not in str(raised.value.context)
    assert provider.key.decode("ascii") not in caplog.text
