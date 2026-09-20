"""Public contracts for authority-only catalog and user-metadata changes."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobLifecycleConflictError
from cacheness.storage import BlobStore
from cacheness.storage.catalog import CatalogField, CatalogSchema
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
from cacheness.storage.transport_evidence import PayloadTransportObservation


class _MetadataOnlyAuthority(InMemoryLifecycleAuthority):
    """Fail if a metadata operation uses a payload lifecycle transition."""

    qualification_identity = "memory"

    def __init__(self) -> None:
        super().__init__()
        self.metadata_replacements = []
        self.reject_lifecycle_mutations = False

    def _reject(self, operation: str) -> None:
        if self.reject_lifecycle_mutations:
            raise AssertionError(f"metadata-only update called {operation}")

    def prepare_mutation(self, spec):
        self._reject("prepare_mutation")
        return super().prepare_mutation(spec)

    def record_verification(self, prepared, proof) -> None:
        self._reject("record_verification")
        super().record_verification(prepared, proof)

    def promote_mutation(self, prepared):
        self._reject("promote_mutation")
        return super().promote_mutation(prepared)

    def abort_mutation(self, prepared, *, candidate_persisted=False) -> None:
        self._reject("abort_mutation")
        super().abort_mutation(prepared, candidate_persisted=candidate_persisted)

    def replace_committed_metadata(self, entry, *, expected, manifest):
        self.metadata_replacements.append((entry, expected, manifest))
        return super().replace_committed_metadata(
            entry, expected=expected, manifest=manifest
        )


def _schema() -> CatalogSchema:
    """Declare the small public catalog used by the CAS contract."""
    return CatalogSchema(
        schema_id="metadata-only-contract",
        fields=(CatalogField("region", "string", queryable=True),),
    )


def _store(root: Path) -> tuple[BlobStore, _MetadataOnlyAuthority]:
    """Seed an entry that has signed transport evidence to preserve exactly."""
    authority = _MetadataOnlyAuthority()
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(instance=authority),
        ),
        cache_dir=root,
    )
    participant = store._materialize_authority_store()
    publish_generation = participant.publish_generation

    def publish_with_observation(staged, locator):
        published = publish_generation(staged, locator)
        published["transport_observation"] = PayloadTransportObservation(
            e_tag='"opaque-seeded-etag"',
            byte_size=published["file_size"],
            version="seeded-version",
        )
        return published

    participant.publish_generation = publish_with_observation
    store.put(
        {"payload": "seeded"},
        key="metadata-only-key",
        metadata={"owner": "before"},
        catalog_schema=_schema(),
        catalog_values={"region": "before"},
    )
    authority.reject_lifecycle_mutations = True

    def reject_participant(*_args, **_kwargs):
        raise AssertionError("metadata-only update touched the payload participant")

    participant.publish_generation = reject_participant
    participant.open_snapshot = reject_participant
    participant.head_generation = reject_participant
    participant.delete_or_prove_absent = reject_participant
    participant.inventory_page = reject_participant
    return store, authority


def test_catalog_and_metadata_updates_use_one_authority_cas_without_payload_io(
    tmp_path: Path,
) -> None:
    """Mutable metadata changes preserve every immutable payload/evidence fact."""
    store, authority = _store(tmp_path / "metadata-only")
    try:
        before = store.lifecycle_authority.read_entry("metadata-only-key")
        assert before is not None
        assert before.transport_evidence is not None

        assert store.update_metadata("metadata-only-key", {"owner": "after"}) is True
        metadata_updated = store.lifecycle_authority.read_entry("metadata-only-key")
        assert metadata_updated is not None
        assert (
            metadata_updated.generation,
            metadata_updated.locator,
            metadata_updated.transport_evidence,
        ) == (before.generation, before.locator, before.transport_evidence)

        receipt = store.update_catalog(
            "metadata-only-key",
            catalog_schema=_schema(),
            catalog_values={"region": "after"},
        )
        assert receipt is not None
        catalog_updated = store.lifecycle_authority.read_entry("metadata-only-key")
        assert catalog_updated is not None
        assert (
            catalog_updated.generation,
            catalog_updated.locator,
            catalog_updated.transport_evidence,
        ) == (before.generation, before.locator, before.transport_evidence)
        assert len(authority.metadata_replacements) == 2

        with pytest.raises(CacheBlobLifecycleConflictError):
            store.update_catalog(
                "metadata-only-key",
                catalog_schema=_schema(),
                catalog_values={"region": "stale"},
                expected=before.expectation,
            )
        assert len(authority.metadata_replacements) == 2
    finally:
        store.close()
