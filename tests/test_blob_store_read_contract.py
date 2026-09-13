"""Current direct-read contracts for the composed BlobStore topology."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobStoreClosedError,
    CacheManifestIntegrityError,
)
from cacheness.storage import (
    BlobReceipt,
    BlobStore,
    PayloadTransportComparison,
    PayloadTransportComparisonStatus,
)
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.legacy_manifest import LegacyManifestRecognitionError
from cacheness.storage.manifest import BlobManifest, verify_current_manifest
from cacheness.storage.transport_evidence import PayloadTransportObservation


def _topology(root: Path) -> StoreTopology:
    """Build the supported local filesystem/SQLite authority pairing."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _store(root: Path) -> BlobStore:
    return BlobStore(_topology(root), cache_dir=root)


def _put_with_transport_evidence(
    store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    key: str,
) -> tuple[BlobReceipt, object, PayloadTransportObservation]:
    """Commit one local entry with a test-only signed transport observation."""
    participant = store._materialize_authority_store()
    publish_generation = participant.publish_generation
    observation: PayloadTransportObservation | None = None

    def publish_with_transport_evidence(staged, locator):
        nonlocal observation
        published = publish_generation(staged, locator)
        observation = PayloadTransportObservation(
            e_tag='"opaque-etag"',
            byte_size=published["file_size"],
            version="version-1",
        )
        published["transport_observation"] = observation
        return published

    monkeypatch.setattr(participant, "publish_generation", publish_with_transport_evidence)
    receipt = store.put_entry("value", key=key)
    assert observation is not None
    return receipt, participant, observation


def test_composed_store_reopens_one_authenticated_canonical_generation(tmp_path: Path) -> None:
    """A current descriptor, rather than a metadata backend, determines a read."""
    root = tmp_path / "reopen"
    first = _store(root)
    try:
        receipt = first.put_entry(
            {"answer": 42}, key="read-key", metadata={"label": "answer"}
        )
        assert isinstance(receipt, BlobReceipt)
    finally:
        first.close()

    reopened = _store(root)
    try:
        committed = reopened.lifecycle_authority.read_entry(receipt.key)
        assert committed is not None
        descriptor = BlobManifest.from_canonical_bytes(committed.manifest)
        verify_current_manifest(descriptor, reopened._authority_manifest_key())
        assert descriptor.generation == receipt.generation
        assert reopened.get(receipt.key) == {"answer": 42}
    finally:
        reopened.close()


def test_absence_is_distinct_from_a_present_none_payload(tmp_path: Path) -> None:
    """The direct store still distinguishes absence before later cache policy."""
    store = _store(tmp_path / "absence")
    try:
        receipt = store.put_entry(None, key="present-none")

        assert store.get("missing") is None
        assert store.exists("missing") is False
        assert store.get(receipt.key) is None
        assert store.exists(receipt.key) is True
    finally:
        store.close()


def test_transport_comparison_matches_one_committed_generation_without_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One exact head corroborates signed evidence without payload verification."""
    store = _store(tmp_path / "transport-match")
    try:
        receipt, participant, expected_observation = _put_with_transport_evidence(
            store, monkeypatch, key="transport-match"
        )
        calls: list[str] = []

        def observe_transport(locator: str) -> PayloadTransportObservation:
            calls.append(locator)
            return expected_observation

        def reject_payload_read(*_args, **_kwargs):
            raise AssertionError("transport comparison downloaded a payload")

        monkeypatch.setattr(participant, "observe_transport", observe_transport, raising=False)
        monkeypatch.setattr(participant, "open_snapshot", reject_payload_read)

        result = store.compare_transport_evidence(receipt.key)

        assert result is not None
        assert result.status is PayloadTransportComparisonStatus.MATCH
        assert (result.key, result.generation, result.locator) == (
            receipt.key,
            receipt.generation,
            receipt.locator,
        )
        assert result.canonical_payload_integrity_verified is False
        assert calls == [receipt.locator]
    finally:
        store.close()


def test_transport_comparison_reports_mismatch_without_mutating_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed opaque transport values remain a read-only mismatch."""
    store = _store(tmp_path / "transport-mismatch")
    try:
        receipt, participant, expected_observation = _put_with_transport_evidence(
            store, monkeypatch, key="transport-mismatch"
        )
        before = store.lifecycle_authority.read_entry(receipt.key)
        assert before is not None
        observed = PayloadTransportObservation(
            e_tag='"changed-opaque-etag"',
            byte_size=expected_observation.byte_size,
            version=expected_observation.version,
        )
        monkeypatch.setattr(
            participant, "observe_transport", lambda _locator: observed, raising=False
        )

        result = store.compare_transport_evidence(receipt.key)

        assert result is not None
        assert result.status is PayloadTransportComparisonStatus.MISMATCH
        assert store.lifecycle_authority.read_entry(receipt.key) == before
    finally:
        store.close()


def test_transport_comparison_reports_unavailable_without_evidence_or_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence and exact observation are both optional participant capabilities."""
    store = _store(tmp_path / "transport-unavailable")
    try:
        plain = store.put_entry("plain", key="transport-no-evidence")
        participant = store._materialize_authority_store()

        def reject_observation(*_args, **_kwargs):
            raise AssertionError("evidence-free entry requested an observation")

        monkeypatch.setattr(
            participant, "observe_transport", reject_observation, raising=False
        )
        no_evidence = store.compare_transport_evidence(plain.key)
        assert no_evidence is not None
        assert no_evidence.status is PayloadTransportComparisonStatus.UNAVAILABLE

        receipt, participant, _ = _put_with_transport_evidence(
            store, monkeypatch, key="transport-no-provider"
        )
        monkeypatch.delattr(participant, "observe_transport", raising=False)
        no_provider = store.compare_transport_evidence(receipt.key)
        assert no_provider is not None
        assert no_provider.status is PayloadTransportComparisonStatus.UNAVAILABLE
    finally:
        store.close()


def test_transport_comparison_reports_exact_absence_and_preserves_backend_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only typed exact absence is normalized; other transport errors stay causal."""
    store = _store(tmp_path / "transport-errors")
    try:
        receipt, participant, _ = _put_with_transport_evidence(
            store, monkeypatch, key="transport-errors"
        )

        def absent(_locator: str) -> PayloadTransportObservation:
            raise FileNotFoundError("exact generation is absent")

        monkeypatch.setattr(participant, "observe_transport", absent, raising=False)
        absent_result = store.compare_transport_evidence(receipt.key)
        assert absent_result is not None
        assert absent_result.status is PayloadTransportComparisonStatus.ABSENT

        error = PermissionError("head denied")

        def denied(_locator: str) -> PayloadTransportObservation:
            raise error

        monkeypatch.setattr(participant, "observe_transport", denied)
        with pytest.raises(CacheBlobBackendError) as raised:
            store.compare_transport_evidence(receipt.key)
        assert raised.value.__cause__ is error
    finally:
        store.close()


def test_transport_comparison_rejects_bad_evidence_before_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed or transplanted evidence fails closed before any object call."""
    store = _store(tmp_path / "transport-tamper")
    try:
        receipt, participant, _ = _put_with_transport_evidence(
            store, monkeypatch, key="transport-tamper"
        )
        read_entry = store.lifecycle_authority.read_entry

        def forged_entry(key: str):
            entry = read_entry(key)
            if entry is None:
                return None
            return replace(entry, transport_evidence=b'{"forged":true}')

        def reject_observation(*_args, **_kwargs):
            raise AssertionError("unauthenticated evidence reached the participant")

        monkeypatch.setattr(store.lifecycle_authority, "read_entry", forged_entry)
        monkeypatch.setattr(
            participant, "observe_transport", reject_observation, raising=False
        )
        with pytest.raises(CacheManifestIntegrityError):
            store.compare_transport_evidence(receipt.key)
    finally:
        store.close()


def test_metadata_cas_preserves_transport_identity_for_later_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata-only CAS never observes payloads but retains signed evidence."""
    store = _store(tmp_path / "transport-cas")
    try:
        receipt, participant, expected_observation = _put_with_transport_evidence(
            store, monkeypatch, key="transport-cas"
        )
        calls: list[str] = []

        def observe_transport(locator: str) -> PayloadTransportObservation:
            calls.append(locator)
            return expected_observation

        monkeypatch.setattr(participant, "observe_transport", observe_transport, raising=False)
        assert store.update_metadata(receipt.key, {"phase": 2})
        assert calls == []

        result = store.compare_transport_evidence(receipt.key)

        assert result is not None
        assert result.status is PayloadTransportComparisonStatus.MATCH
        assert calls == [receipt.locator]
    finally:
        store.close()


def test_transport_comparison_match_never_bypasses_canonical_read_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cheap transport match cannot authorize handler deserialization alone."""
    from cacheness.storage import lifecycle as lifecycle_module

    store = _store(tmp_path / "transport-canonical-read")
    try:
        receipt, participant, expected_observation = _put_with_transport_evidence(
            store, monkeypatch, key="transport-canonical-read"
        )
        monkeypatch.setattr(
            participant,
            "observe_transport",
            lambda _locator: expected_observation,
            raising=False,
        )
        matched = store.compare_transport_evidence(receipt.key)
        assert matched is not None
        assert matched.status is PayloadTransportComparisonStatus.MATCH

        sha256_and_size = lifecycle_module.sha256_and_size
        calls: list[Path] = []

        def record_canonical_verification(path: Path):
            calls.append(path)
            return sha256_and_size(path)

        monkeypatch.setattr(
            lifecycle_module, "sha256_and_size", record_canonical_verification
        )
        assert store.get(receipt.key) == "value"
        assert calls
    finally:
        store.close()


def test_metadata_update_reauthenticates_mutable_descriptor_without_payload_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A metadata patch changes only signed mutable facts through the authority CAS."""
    store = _store(tmp_path / "metadata")
    try:
        receipt = store.put_entry("value", key="metadata-key", metadata={"phase": 1})
        before = store.lifecycle_authority.read_entry(receipt.key)
        assert before is not None
        before_descriptor = BlobManifest.from_canonical_bytes(before.manifest)

        participant = store._materialize_authority_store()
        original_open_snapshot = participant.open_snapshot

        def reject_payload_lifecycle(*_args, **_kwargs):
            raise AssertionError("metadata-only update touched the payload participant")

        monkeypatch.setattr(participant, "publish_generation", reject_payload_lifecycle)
        monkeypatch.setattr(participant, "open_snapshot", reject_payload_lifecycle)
        monkeypatch.setattr(participant, "delete_or_prove_absent", reject_payload_lifecycle)

        assert store.update_metadata(receipt.key, {"phase": 2, "label": "current"})

        committed = store.lifecycle_authority.read_entry(receipt.key)
        assert committed is not None
        descriptor = BlobManifest.from_canonical_bytes(committed.manifest)
        verify_current_manifest(descriptor, store._authority_manifest_key())
        assert committed.manifest != before.manifest
        assert committed.expectation != before.expectation
        assert (
            committed.generation,
            committed.locator,
            descriptor.generation,
            descriptor.locator,
            descriptor.digest,
            descriptor.byte_size,
            committed.transport_evidence,
        ) == (
            before.generation,
            before.locator,
            before_descriptor.generation,
            before_descriptor.locator,
            before_descriptor.digest,
            before_descriptor.byte_size,
            before.transport_evidence,
        )
        assert descriptor.generation == receipt.generation
        assert descriptor.user_metadata == {"phase": 2, "label": "current"}
        monkeypatch.setattr(participant, "open_snapshot", original_open_snapshot)
        assert store.get(receipt.key) == "value"
    finally:
        store.close()


def test_unsupported_legacy_layout_is_rejected_without_mutation(tmp_path: Path) -> None:
    """Layout inspection never opens, upgrades, or repairs a legacy root."""
    root = tmp_path / "legacy-layout"
    root.mkdir()
    legacy = root / "provenance.json"
    legacy.write_text('{"source":"development-layout"}', encoding="utf-8")
    before = legacy.read_bytes()

    with pytest.raises(LegacyManifestRecognitionError):
        _store(root)

    assert legacy.read_bytes() == before
    assert not (root / ".cacheness" / "lifecycle-authority-v1.sqlite3").exists()


def test_close_prevents_later_direct_reads_without_reopening_authority(tmp_path: Path) -> None:
    """Close remains an admission boundary for the composed store."""
    store = _store(tmp_path / "closed")
    store.close()

    with pytest.raises(CacheBlobStoreClosedError):
        store.get("after-close")
