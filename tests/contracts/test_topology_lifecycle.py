"""Common lifecycle contracts for each exact supported topology profile."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobBackendError
from cacheness.storage import BlobStore
from cacheness.storage.backends.blob_backends import InMemoryBlobBackend
from cacheness.storage.backends.s3_backend import S3InventoryPage, S3ObjectEvidence
from cacheness.storage.catalog import CatalogField, CatalogSchema
from cacheness.storage.composition import BackendRef, RoleRegistry, StoreTopology
from cacheness.storage.lifecycle import AuthorityLifecycleEngine
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
from cacheness.storage.reconciliation import ReconciliationAction


class _StaticRemoteManifestKey:
    """Application-owned shared key material for deterministic remote clients."""

    def get_key(self) -> bytes:
        return b"r" * 32

    def get_or_initialize_new_store(self) -> bytes:
        return self.get_key()

    def initialize_new_store(self) -> bytes:
        return self.get_key()


class _InventoryGenerationIO:
    """Delegate native payload mechanics while exposing one bounded S3 evidence page."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.inventory_calls: list[str | None] = []

    @property
    def root(self):
        return self._delegate.root

    def stage(self, *args: object, **kwargs: object):
        return self._delegate.stage(*args, **kwargs)

    def publish_generation(self, *args: object, **kwargs: object):
        return self._delegate.publish_generation(*args, **kwargs)

    def open_snapshot(self, *args: object, **kwargs: object):
        return self._delegate.open_snapshot(*args, **kwargs)

    def delete_or_prove_absent(self, *args: object, **kwargs: object):
        return self._delegate.delete_or_prove_absent(*args, **kwargs)

    def close(self) -> None:
        self._delegate.close()

    def inventory_page(self, continuation_token: str | None = None) -> S3InventoryPage:
        self.inventory_calls.append(continuation_token)
        if continuation_token is not None:
            return S3InventoryPage((), None)
        return S3InventoryPage(
            (S3ObjectEvidence("unattributed/immutable-generation", 17),),
            "remote-page-2",
        )


class _RemotePayload(InMemoryBlobBackend):
    """S3-shaped participant boundary without claiming live service evidence."""

    qualification_identity = "s3"
    topology_capabilities = {
        "durable": True,
        "process_scope": "multi_host",
        "host_scope": "multi_host",
        "immutable_generations": True,
        "streaming": True,
        "listing": True,
    }

    def __init__(self) -> None:
        super().__init__()
        self.inventory_io: _InventoryGenerationIO | None = None

    def materialize_handler_io(self) -> _InventoryGenerationIO:
        if self.inventory_io is None:
            self.inventory_io = _InventoryGenerationIO(super().materialize_handler_io())
        return self.inventory_io


class _NoUnboundedRemoteAuthority(InMemoryLifecycleAuthority):
    """Remote authority spy whose legacy whole-catalog path is forbidden."""

    qualification_identity = "postgresql"
    topology_capabilities = {
        "durable": True,
        "process_scope": "multi_host",
        "host_scope": "multi_host",
        "transaction_scope": "authority",
        "exact_cas": True,
        "portable_query": True,
        "canonical_scan": True,
        "index_acceleration": True,
    }

    def __init__(self) -> None:
        super().__init__()
        self.unbounded_list_calls = 0

    def list_entries(self):
        self.unbounded_list_calls += 1
        raise AssertionError("remote workflows must use bounded authority pages")


def _remote_store(tmp_path: Path) -> BlobStore:
    registry = RoleRegistry()
    registry.register(
        "payload",
        "s3",
        _RemotePayload,
        capabilities=_RemotePayload.topology_capabilities,
        replace=True,
    )
    registry.register(
        "authority",
        "postgresql",
        _NoUnboundedRemoteAuthority,
        capabilities=_NoUnboundedRemoteAuthority.topology_capabilities,
        replace=True,
    )
    return BlobStore(
        StoreTopology(
            payload=BackendRef(name="s3"),
            authority=BackendRef(name="postgresql"),
            role_registry=registry,
        ),
        cache_dir=tmp_path / "remote-store",
        manifest_key_provider=_StaticRemoteManifestKey(),
    )


def test_remote_profile_uses_one_engine_and_bounded_authority_pages(tmp_path: Path) -> None:
    """Remote public workflows never materialize a legacy authority catalog."""
    schema = CatalogSchema((CatalogField("tenant", "string", queryable=True),))
    store = _remote_store(tmp_path)
    try:
        assert type(store.lifecycle) is AuthorityLifecycleEngine
        assert store.put_entry(
            {"generation": "one"},
            key="remote-key",
            catalog_schema=schema,
            catalog_values={"tenant": "alpha"},
        ).key == "remote-key"

        page = store.list_page(schema=schema, limit=1, work_cap=2)
        assert [entry.key for entry in page.entries] == ["remote-key"]
        assert page.exhausted is True
        with pytest.raises(CacheBlobBackendError, match="list_page"):
            store.list()

        report = store.reconcile()
        assert report.inventory_cursor == "remote-page-2"
        assert len(report.findings) == 1
        assert report.findings[0].action is ReconciliationAction.REPORT_ONLY
        assert report.findings[0].reason == "unattributed_payload_inventory"
        assert store.payload_backend.inventory_io.inventory_calls == [None]

        assert store.clear() == 1
        assert store.lifecycle_authority.unbounded_list_calls == 0
    finally:
        store.close()


def test_remote_inventory_continuation_is_signed_with_the_authority_resume_token(
    tmp_path: Path,
) -> None:
    """One inventory page is resumable evidence, never a cleanup authorization."""
    store = _remote_store(tmp_path)
    try:
        first = store.reconcile()
        assert first.resume_token is not None
        assert first.inventory_cursor == "remote-page-2"

        second = store.reconcile(resume_token=first.resume_token)
        assert second.inventory_cursor is None
        assert store.payload_backend.inventory_io.inventory_calls == [None, "remote-page-2"]
        assert all(
            finding.action is ReconciliationAction.REPORT_ONLY
            for finding in (*first.findings, *second.findings)
        )
        assert store.lifecycle_authority.unbounded_list_calls == 0
    finally:
        store.close()
