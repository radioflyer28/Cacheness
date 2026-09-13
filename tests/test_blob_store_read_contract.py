"""Current direct-read contracts for the composed BlobStore topology."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobStoreClosedError
from cacheness.storage import BlobReceipt, BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.legacy_manifest import LegacyManifestRecognitionError
from cacheness.storage.manifest import BlobManifest, verify_current_manifest
from cacheness.storage.read_contract import PayloadTransportComparisonStatus
from cacheness.storage.transport_evidence import PayloadTransportObservation


def _topology(root: Path) -> StoreTopology:
    """Build the supported local filesystem/SQLite authority pairing."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _store(root: Path) -> BlobStore:
    return BlobStore(_topology(root), cache_dir=root)


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
        participant = store._materialize_authority_store()
        publish_generation = participant.publish_generation
        expected_observation: PayloadTransportObservation | None = None

        def publish_with_transport_evidence(staged, locator):
            nonlocal expected_observation
            published = publish_generation(staged, locator)
            expected_observation = PayloadTransportObservation(
                e_tag='"opaque-etag"',
                byte_size=published["file_size"],
                version="version-1",
            )
            published["transport_observation"] = expected_observation
            return published

        calls: list[str] = []

        def observe_transport(locator: str) -> PayloadTransportObservation:
            calls.append(locator)
            assert expected_observation is not None
            return expected_observation

        def reject_payload_read(*_args, **_kwargs):
            raise AssertionError("transport comparison downloaded a payload")

        monkeypatch.setattr(participant, "publish_generation", publish_with_transport_evidence)
        monkeypatch.setattr(participant, "observe_transport", observe_transport, raising=False)
        receipt = store.put_entry("value", key="transport-match")
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
