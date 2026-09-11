"""Security and recovery contracts for explicit offline migration evidence."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import logging
from pathlib import Path

import pytest

from cacheness.error_handling import (
    CacheBlobMigrationEvidenceError,
    CacheBlobMigrationEvidenceMismatchError,
    CacheBlobMigrationOfflineDecisionRequiredError,
    CacheBlobMigrationPlanStaleError,
)
from cacheness.storage import BackendRef, BlobStore, StoreTopology
from cacheness.storage.migration import (
    MigrationCompatibilityEdge,
    MigrationPlan,
    OfflineMigrationService,
)
from cacheness.storage.migration_authority import AuthorityIdentitySnapshot
from cacheness.storage.manifest import BlobManifest
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

    key = b"migration-evidence-key-sentinel0"

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
        evidence_version=2,
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
    assert created.signing_key_fingerprint
    assert store.load() == created

    raw = store.read_bytes()
    assert decode_maintenance_evidence(raw, provider.get_key()) == created
    assert encode_maintenance_evidence(created, provider.get_key()) == raw

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
    with pytest.raises(CacheBlobMigrationEvidenceError, match="symlink"):
        MaintenanceEvidenceStore(linked, evidence.run_id, provider)
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


def test_evidence_store_requires_inspection_first_and_rejects_hostile_raw_bytes(
    tmp_path: Path,
) -> None:
    """No malformed or pre-advanced evidence can create a resumable run."""
    provider = _SentinelKeyProvider()
    evidence = _evidence()
    store = MaintenanceEvidenceStore(tmp_path / "maintenance", evidence.run_id, provider)

    with pytest.raises(CacheBlobMigrationEvidenceMismatchError, match="inspection"):
        store.create(_evidence(state=MaintenanceEvidenceState.PLANNED))
    assert not store.evidence_path.exists()

    store.create(evidence)
    hostile_records = (
        b'{"evidence":{},"evidence":{},"signature":"0"}',
        b'{"evidence":' + b"[" * 20 + b"]" * 20 + b',"signature":"0"}',
        b"x" * 65_537,
    )
    for hostile in hostile_records:
        store.evidence_path.write_bytes(hostile)
        with pytest.raises(CacheBlobMigrationEvidenceError):
            store.load()


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


def _memory_store(root: Path, key_provider: _SentinelKeyProvider) -> BlobStore:
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
        ),
        cache_dir=root,
        manifest_key_provider=key_provider,
    )
    store.initialize()
    return store


def _service(
    source: BlobStore,
    destination: BlobStore,
    work_directory: Path,
    *,
    run_id: str = "resumable-maintenance-run",
) -> OfflineMigrationService:
    return OfflineMigrationService(
        source=source,
        destination=destination,
        work_directory=work_directory,
        run_id=run_id,
        stopped_workers_acknowledged=True,
        compatibility_edges=(MigrationCompatibilityEdge.current_to_current_for_test(),),
    )


def test_resume_requires_exact_run_and_evidence_then_revalidates_next_step(tmp_path: Path) -> None:
    """A fresh process resumes only the explicit authenticated staged run."""
    provider = _SentinelKeyProvider()
    source = _memory_store(tmp_path / "source", provider)
    destination = _memory_store(tmp_path / "destination", provider)
    try:
        source.put_entry({"answer": 42}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.plan(service.inspect())
        service.stage(plan)
        destination_revision = destination.lifecycle_authority.identity_snapshot().revision

        restarted = _service(source, destination, tmp_path / "maintenance")
        with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError):
            restarted.resume(plan, run_id=None, evidence_path=service.evidence_path)
        with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError):
            restarted.resume(plan, run_id=service.run_id, evidence_path=None)
        with pytest.raises(CacheBlobMigrationEvidenceMismatchError):
            restarted.resume(plan, run_id=service.run_id, evidence_path=tmp_path / "wrong.json")

        result = restarted.resume(
            plan, run_id=service.run_id, evidence_path=service.evidence_path
        )
        assert result.state is MaintenanceEvidenceState.VERIFIED
        assert destination.lifecycle_authority.identity_snapshot().revision == destination_revision
        assert destination.get("entry") is None
    finally:
        source.close()
        destination.close()


def test_resume_refuses_mismatched_output_or_stale_source_without_adoption(tmp_path: Path) -> None:
    """Recorded output validation precedes any next state or authority mutation."""
    provider = _SentinelKeyProvider()
    source = _memory_store(tmp_path / "source", provider)
    destination = _memory_store(tmp_path / "destination", provider)
    try:
        source.put_entry({"answer": 42}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.plan(service.inspect())
        service.stage(plan)
        destination_revision = destination.lifecycle_authority.identity_snapshot().revision
        candidate = service._candidate_entries[0]
        destination.payload_backend.write_blob(candidate.locator, b"forged-candidate")

        restarted = _service(source, destination, tmp_path / "maintenance")
        with pytest.raises(CacheBlobMigrationEvidenceMismatchError, match="candidate"):
            restarted.resume(
                plan, run_id=service.run_id, evidence_path=service.evidence_path
            )
        assert destination.lifecycle_authority.identity_snapshot().revision == destination_revision
        assert destination.get("entry") is None

        source.put_entry({"new": "source-revision"}, key="new-entry")
        with pytest.raises(CacheBlobMigrationPlanStaleError):
            restarted.resume(
                plan, run_id=service.run_id, evidence_path=service.evidence_path
            )
        assert destination.lifecycle_authority.identity_snapshot().revision == destination_revision
    finally:
        source.close()
        destination.close()


def test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shared plans authorize only fresh authenticated source state before writes."""
    provider = _SentinelKeyProvider()
    source = _memory_store(tmp_path / "source", provider)
    destination = _memory_store(tmp_path / "destination", provider)
    try:
        source.put_entry(
            {"answer": 42},
            key="entry",
            catalog_values={"unknown_authenticated_attribute": "preserve-me"},
        )
        service = _service(source, destination, tmp_path / "maintenance")
        shared_plan = MigrationPlan.from_canonical_bytes(
            service.plan(service.inspect()).to_canonical_bytes()
        )

        service.stage(shared_plan)
        candidate = service._candidate_entries[0]
        candidate_manifest = BlobManifest.from_canonical_bytes(candidate.manifest)
        assert candidate_manifest.catalog_values["unknown_authenticated_attribute"] == "preserve-me"

        source.close()
        destination.close()
        source = _memory_store(tmp_path / "drift-source", provider)
        destination = _memory_store(tmp_path / "drift-destination", provider)
        service = _service(source, destination, tmp_path / "drift-maintenance")
        source.put_entry(
            {"answer": 42},
            key="entry",
            catalog_values={"unknown_authenticated_attribute": "before"},
        )
        stale_plan = MigrationPlan.from_canonical_bytes(
            service.plan(service.inspect()).to_canonical_bytes()
        )
        source.put_entry(
            {"answer": 42},
            key="entry",
            catalog_values={"unknown_authenticated_attribute": "after"},
        )

        writes: list[object] = []

        def unexpected_candidate_write(*args: object, **kwargs: object) -> str:
            writes.append((args, kwargs))
            raise AssertionError("candidate write must follow source-state validation")

        monkeypatch.setattr(destination.payload_backend, "write_blob", unexpected_candidate_write)
        with pytest.raises(CacheBlobMigrationPlanStaleError, match="source_state_drift"):
            service.stage(stale_plan)

        assert writes == []
        assert destination.get("entry") is None

        source.close()
        destination.close()
        source = _memory_store(tmp_path / "forged-source", provider)
        destination = _memory_store(tmp_path / "forged-destination", provider)
        service = _service(source, destination, tmp_path / "forged-maintenance")
        source.put_entry({"answer": 42}, key="entry")
        forged_plan = MigrationPlan.from_canonical_bytes(
            service.plan(service.inspect()).to_canonical_bytes()
        )
        authority = source.lifecycle_authority
        authenticated_entry = authority._entries["entry"]
        forged_manifest = authenticated_entry.manifest[:-1] + b"X"
        authority._entries["entry"] = replace(
            authenticated_entry,
            manifest=forged_manifest,
            expectation=replace(
                authenticated_entry.expectation,
                manifest_digest=hashlib.sha256(forged_manifest).hexdigest(),
            ),
        )

        with pytest.raises(CacheBlobMigrationPlanStaleError, match="source_state_drift"):
            service.stage(forged_plan)
        assert destination.get("entry") is None
    finally:
        source.close()
        destination.close()
