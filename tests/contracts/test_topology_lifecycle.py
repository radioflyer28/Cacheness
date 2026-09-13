"""Common lifecycle contracts for each exact supported topology profile."""

from __future__ import annotations

from pathlib import Path

from obstore.store import MemoryStore
import pytest

from cacheness.error_handling import CacheBlobBackendError
from cacheness.storage import BlobStore
from cacheness.storage.catalog import CatalogField, CatalogSchema
from cacheness.storage.composition import BackendRef, RoleRegistry, StoreTopology
from cacheness.storage.guarded_handler_io import GuardedHandlerIO
from cacheness.storage.lifecycle import AuthorityLifecycleEngine
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
from cacheness.storage.obstore_generation_io import (
    ObstoreGenerationIO,
    ObstoreInventoryPage,
    ObstoreObjectEvidence,
)
from cacheness.storage.reconciliation import ReconciliationAction


class _StaticRemoteManifestKey:
    """Application-owned shared key material for deterministic remote clients."""

    def get_key(self) -> bytes:
        return b"r" * 32

    def get_or_initialize_new_store(self) -> bytes:
        return self.get_key()

    def initialize_new_store(self) -> bytes:
        return self.get_key()


class _RemotePayload(ObstoreGenerationIO):
    """S3-shaped unified participant without a live-service qualification claim."""

    qualification_identity = "s3"
    topology_capabilities = {
        "durable": True,
        "process_scope": "multi_host",
        "host_scope": "multi_host",
        "immutable_generations": True,
        "streaming": True,
        "listing": True,
    }

    def __init__(self, *, handler_root: Path) -> None:
        handler_root.mkdir(parents=True)
        super().__init__(
            MemoryStore(), GuardedHandlerIO(handler_root), qualification_identity="s3"
        )
        self.inventory_calls: list[str | None] = []

    def inventory_page(
        self, continuation: str | None = None, *, max_objects: int | None = None
    ) -> ObstoreInventoryPage:
        """Return fixed opaque pages; inventory stays diagnostic-only in this fake."""
        del max_objects
        self.inventory_calls.append(continuation)
        if continuation == "remote-page-2":
            return ObstoreInventoryPage(
                (ObstoreObjectEvidence("unknown/continued-generation", 17, None, None),),
                None,
            )
        if continuation is not None:
            return ObstoreInventoryPage((), None)
        return ObstoreInventoryPage(
            (
                ObstoreObjectEvidence("committed/immutable-generation", 17, None, None),
                ObstoreObjectEvidence("unknown/immutable-generation", 17, None, None),
            ),
            "remote-page-2",
        )


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
        self.inventory_attributed_locators: set[str] = set()
        self.inventory_attribution_available = True
        self.inventory_attribution_calls: list[tuple[int, tuple[str, ...]]] = []

    def list_entries(self):
        self.unbounded_list_calls += 1
        raise AssertionError("remote workflows must use bounded authority pages")

    def inventory_locator_attribution(self, snapshot, locators):
        """Test double for the narrow current-page authority capability."""
        self.inventory_attribution_calls.append((snapshot.authority_revision, locators))
        if (
            not self.inventory_attribution_available
            or snapshot.authority_revision != self._revision
        ):
            return None
        return frozenset(
            locator for locator in locators if locator in self.inventory_attributed_locators
        )


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
            payload=BackendRef(
                name="s3", options={"handler_root": tmp_path / "remote-handler"}
            ),
            authority=BackendRef(name="postgresql"),
            role_registry=registry,
        ),
        cache_dir=tmp_path / "remote-store",
        manifest_key_provider=_StaticRemoteManifestKey(),
    )


@pytest.mark.parametrize("profile", ("memory", "sqlite-filesystem"))
def test_local_reference_profiles_share_the_same_engine(
    tmp_path: Path, profile: str
) -> None:
    """Each exact local pairing reaches the same promotion/deletion sequence."""
    if profile == "memory":
        topology = StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        )
    else:
        root = tmp_path / "sqlite-filesystem"
        topology = StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        )
    store = BlobStore(topology, cache_dir=tmp_path / profile)
    try:
        assert type(store.lifecycle) is AuthorityLifecycleEngine
        assert type(store.payload_backend) is ObstoreGenerationIO
        assert store.put({"generation": "one"}, key="profile-key") == "profile-key"
        assert store.get("profile-key") == {"generation": "one"}
        assert store.put({"generation": "two"}, key="profile-key") == "profile-key"
        assert store.get("profile-key") == {"generation": "two"}
        assert store.delete("profile-key") is True
        assert store.get("profile-key") is None
    finally:
        store.close()


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
        store.lifecycle_authority.inventory_attributed_locators.add(
            "committed/immutable-generation"
        )

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
        assert store.lifecycle_authority.inventory_attribution_calls == [
            (
                store.lifecycle_authority._revision,
                (
                    "committed/immutable-generation",
                    "unknown/immutable-generation",
                ),
            )
        ]
        assert store.payload_backend.inventory_calls == [None]

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
        assert store.payload_backend.inventory_calls == [None, "remote-page-2"]
        assert [finding.reason for finding in (*first.findings, *second.findings)] == [
            "unattributed_payload_inventory",
            "unattributed_payload_inventory",
            "unattributed_payload_inventory",
        ]
        assert all(
            finding.action is ReconciliationAction.REPORT_ONLY
            for finding in (*first.findings, *second.findings)
        )
        assert store.lifecycle_authority.unbounded_list_calls == 0
    finally:
        store.close()


def test_remote_inventory_without_snapshot_attribution_is_indeterminate(tmp_path: Path) -> None:
    """A stale authority revision cannot turn remote inventory into residue."""
    store = _remote_store(tmp_path)
    try:
        store.lifecycle_authority.inventory_attribution_available = False

        report = store.reconcile()

        assert {finding.reason for finding in report.findings} == {
            "payload_inventory_attribution_indeterminate"
        }
        assert {finding.status.value for finding in report.findings} == {
            "requires_confirmation"
        }
        assert {finding.residue_role for finding in report.findings} == {"indeterminate"}
        machine_view = report.machine_view()
        assert "unknown/immutable-generation" not in repr(machine_view)
        assert "unknown/immutable-generation" not in repr(report.to_dict())
        assert all(
            finding.locator_fingerprint is not None for finding in report.findings
        )
    finally:
        store.close()
