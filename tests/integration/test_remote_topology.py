"""Real two-client PostgreSQL/Amazon-S3 topology qualification coverage."""

from __future__ import annotations

from pathlib import Path
from threading import Barrier, Thread
from typing import Any
from uuid import uuid4

import pytest

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobRecoverableCleanupError,
)
from cacheness.storage import BlobStore
from cacheness.storage.catalog import CatalogField, CatalogSchema
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.lifecycle import AuthorityLifecycleEngine


pytestmark = pytest.mark.live_remote


def _connection_factory(resources: Any):
    """Return a distinct direct driver connection for every authority operation."""
    import psycopg

    return lambda: psycopg.connect(resources.config.postgres_dsn)


def _remote_store(
    resources: Any,
    signer: Any,
    cache_dir: Path,
) -> BlobStore:
    """Build one independently owned client over the shared run namespace."""
    import boto3

    client = boto3.session.Session().client(
        "s3", region_name=resources.config.region or "us-east-1"
    )
    topology = StoreTopology(
        payload=BackendRef(
            name="s3",
            options={
                "bucket": resources.config.s3_bucket,
                "prefix": resources.namespace.prefix,
                "region": resources.config.region or "us-east-1",
                "client": client,
                "expected_bucket_owner": resources.config.expected_bucket_owner,
                "staging_root": cache_dir / "s3-private-stage",
                "max_upload_bytes": 8 * 1024 * 1024,
                "max_download_bytes": 8 * 1024 * 1024,
                "max_download_work": 256,
                "max_inventory_objects": 64,
                "max_inventory_bytes": 8 * 1024 * 1024,
                "max_inventory_work": 1,
            },
        ),
        authority=BackendRef(
            name="postgresql",
            options={
                "connection_factory": _connection_factory(resources),
                "schema": resources.namespace.schema,
                "store_identity": f"phase5-{resources.namespace.run_id[-32:]}",
                "statement_timeout_ms": 250,
                "lock_timeout_ms": 100,
            },
        ),
    )
    store = BlobStore(
        topology,
        cache_dir=cache_dir,
        manifest_key_provider=signer,
    )
    store.initialize()
    return store


def _recreated_signer(resources: Any) -> Any:
    """Create a third provider from external bytes without persisting key material."""
    from tests.qualification.conftest import InMemoryManifestSigner

    return InMemoryManifestSigner(resources.config.manifest_key)


def test_live_remote_clients_share_only_services_for_cross_client_read_and_paging(
    live_qualification_resources: Any,
    live_manifest_signers: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    """A newly constructed client reads committed values and follows catalog cursors."""
    resources = live_qualification_resources
    first_signer, second_signer = live_manifest_signers
    first = _remote_store(resources, first_signer, tmp_path / "client-a")
    second = _remote_store(resources, second_signer, tmp_path / "client-b")
    schema = CatalogSchema((CatalogField("suite", "string", queryable=True),))
    first_key = f"cross-client-{uuid4().hex}"
    second_key = f"cross-client-{uuid4().hex}"
    try:
        assert first is not second
        assert first.lifecycle_authority is not second.lifecycle_authority
        assert first.payload_backend is not second.payload_backend
        assert type(first.lifecycle) is AuthorityLifecycleEngine
        assert type(second.lifecycle) is AuthorityLifecycleEngine

        first.put(
            {"writer": "a", "value": 1},
            key=first_key,
            catalog_schema=schema,
            catalog_values={"suite": "remote"},
        )
        first.put(
            {"writer": "a", "value": 2},
            key=second_key,
            catalog_schema=schema,
            catalog_values={"suite": "remote"},
        )
        assert second.get(first_key) == {"writer": "a", "value": 1}

        page_one = second.list_page(schema=schema, limit=1, work_cap=2)
        assert len(page_one.entries) == 1
        assert page_one.next_cursor is not None
        page_two = second.list_page(
            schema=schema, cursor=page_one.next_cursor, limit=1, work_cap=2
        )
        assert {entry.key for entry in (*page_one.entries, *page_two.entries)} == {
            first_key,
            second_key,
        }

        # S3 inventory may report evidence, but it never revokes committed
        # authority entries.  Cross-client deletion and clear remain public
        # lifecycle operations through PostgreSQL promotion/debt handling.
        second.reconcile()
        assert first.get(first_key) == {"writer": "a", "value": 1}
        assert second.delete(first_key) is True
        assert first.get(first_key) is None
        assert second.clear() >= 1
        assert first.get(second_key) is None
    finally:
        first.close()
        second.close()


def test_live_remote_same_key_contention_preserves_one_complete_generation(
    live_qualification_resources: Any,
    live_manifest_signers: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    """Concurrent clients expose success/conflict/typed-retryable progress only."""
    resources = live_qualification_resources
    first_signer, second_signer = live_manifest_signers
    first = _remote_store(resources, first_signer, tmp_path / "contender-a")
    second = _remote_store(resources, second_signer, tmp_path / "contender-b")
    key = f"same-key-{uuid4().hex}"
    barrier = Barrier(2)
    outcomes: list[object] = []

    def contend(store: BlobStore, value: str) -> None:
        barrier.wait()
        try:
            outcomes.append(store.put({"winner": value}, key=key))
        except (
            CacheBlobLifecycleConflictError,
            CacheBlobLifecycleTimeoutError,
            CacheBlobRecoverableCleanupError,
        ) as error:
            outcomes.append(error)

    try:
        first_worker = Thread(target=contend, args=(first, "a"), daemon=True)
        second_worker = Thread(target=contend, args=(second, "b"), daemon=True)
        first_worker.start()
        second_worker.start()
        first_worker.join(timeout=15)
        second_worker.join(timeout=15)
        assert not first_worker.is_alive()
        assert not second_worker.is_alive()
        assert len(outcomes) == 2
        assert all(
            outcome == key
            or isinstance(
                outcome,
                (
                    CacheBlobLifecycleConflictError,
                    CacheBlobLifecycleTimeoutError,
                    CacheBlobRecoverableCleanupError,
                ),
            )
            for outcome in outcomes
        )
        committed = second.get(key)
        assert committed in ({"winner": "a"}, {"winner": "b"}, None)
        if committed is not None:
            assert first.get(key) == committed
    finally:
        first.close()
        second.close()


def test_live_remote_recreated_client_reconciles_interruption_and_cleanup_debt(
    live_qualification_resources: Any,
    live_manifest_signers: tuple[Any, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only durable intent/debt crosses reconstructed clients; no cross-resource ACID is claimed."""
    from tests._lifecycle_test_support import InjectedLifecycleFault

    resources = live_qualification_resources
    first_signer, second_signer = live_manifest_signers
    original = _remote_store(resources, first_signer, tmp_path / "interrupted")
    key = f"recovery-{uuid4().hex}"
    recreated: BlobStore | None = None
    try:
        def interrupt_before_promotion(boundary: str) -> None:
            if boundary == "put.before_promotion":
                raise InjectedLifecycleFault("real remote interruption")

        original.lifecycle.fault_hook = interrupt_before_promotion
        with pytest.raises(InjectedLifecycleFault):
            original.put({"state": "candidate"}, key=key)
        original.lifecycle.fault_hook = None
        assert original.get(key) is None
        candidate_debt = original.lifecycle_authority.pending_cleanup_debts(key=key)
        assert len(candidate_debt) == 1
        assert candidate_debt[0].role == "candidate"
        original.close()

        recreated = _remote_store(
            resources, _recreated_signer(resources), tmp_path / "recreated"
        )
        report = recreated.reconcile(apply=True)
        assert report.applied is True
        assert recreated.lifecycle_authority.pending_cleanup_debts(key=key) == ()

        recreated.put({"state": "old"}, key=key)
        original_delete = recreated._delete_or_prove_absent

        def fail_one_post_promotion_delete(locator: Path) -> None:
            monkeypatch.setattr(recreated, "_delete_or_prove_absent", original_delete)
            raise CacheBlobRecoverableCleanupError(
                "test-only post-promotion deletion interruption"
            )

        monkeypatch.setattr(recreated, "_delete_or_prove_absent", fail_one_post_promotion_delete)
        with pytest.raises(CacheBlobRecoverableCleanupError):
            recreated.put({"state": "committed"}, key=key)
        assert recreated.get(key) == {"state": "committed"}
        assert recreated.lifecycle_authority.pending_cleanup_debts(key=key)
        resumed = recreated.reconcile(apply=True)
        assert resumed.applied is True
        assert recreated.get(key) == {"state": "committed"}
        assert recreated.lifecycle_authority.pending_cleanup_debts(key=key) == ()
    finally:
        original.close()
        if recreated is not None:
            recreated.close()
