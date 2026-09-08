"""Red contracts for the one-root BlobStore composition and ownership model."""

from __future__ import annotations

import ast
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import sqlite3

import pytest


COMPOSITION_MODULE = "cacheness.storage.composition"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _composition():
    assert find_spec(COMPOSITION_MODULE) is not None, (
        "Phase 4 must provide StoreTopology as the only BlobStore composition root"
    )
    return import_module(COMPOSITION_MODULE)


class _ClosableParticipant:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _Payload(_ClosableParticipant):
    capabilities = {"immutable_generations": True, "streaming": True, "listing": True}

    def materialize_handler_io(self) -> object:
        """Satisfy the structural payload role in pure composition tests."""
        return object()


class _Authority(_ClosableParticipant):
    capabilities = {"transactional": True, "compare_and_swap": True, "portable_query": True}


def test_one_topology_keeps_exact_injected_instances_and_caller_ownership() -> None:
    composition = _composition()
    payload = _Payload()
    authority = _Authority()
    topology = composition.StoreTopology(payload=payload, authority=authority)

    resolved = topology.resolve()
    resolved.close()

    assert resolved.payload is payload
    assert resolved.authority is authority
    assert payload.close_calls == 0
    assert authority.close_calls == 0


def test_name_and_instance_or_options_and_instance_fail_before_initialization() -> None:
    composition = _composition()
    payload = _Payload()

    with pytest.raises(composition.CompositionValidationError):
        composition.BackendRef(instance=payload, name="memory-payload")
    with pytest.raises(composition.CompositionValidationError):
        composition.BackendRef(instance=payload, options={"root": "unused"})


def test_registered_names_and_builtins_use_the_same_role_registry_path() -> None:
    composition = _composition()
    registry = composition.RoleRegistry()
    registry.register("payload", "fake", _Payload)
    registry.register("authority", "fake", _Authority)

    resolved = composition.StoreTopology(
        payload=composition.BackendRef(name="fake"),
        authority=composition.BackendRef(name="fake"),
    ).resolve(registry)

    assert isinstance(resolved.payload, _Payload)
    assert isinstance(resolved.authority, _Authority)


def test_named_options_are_isolated_per_construction() -> None:
    composition = _composition()
    first = composition.BackendRef(name="memory", options={"namespace": "one"})
    second = composition.BackendRef(name="memory", options={"namespace": "two"})

    assert first.options == {"namespace": "one"}
    assert second.options == {"namespace": "two"}
    assert first.options is not second.options


def test_backend_ref_makes_caller_and_store_ownership_explicit() -> None:
    composition = _composition()
    injected = composition.BackendRef(instance=_Payload())
    registered = composition.BackendRef(name="memory")

    assert injected.ownership is composition.Ownership.CALLER
    assert registered.ownership is composition.Ownership.STORE


def test_explicit_ownership_transfer_closes_constructed_or_transferred_resources_once() -> None:
    composition = _composition()
    payload = _Payload()
    authority = _Authority()
    resolved = composition.StoreTopology(
        payload=composition.BackendRef(instance=payload, transfer_ownership=True),
        authority=composition.BackendRef(instance=authority, transfer_ownership=True),
    ).resolve()

    resolved.close()
    resolved.close()

    assert payload.close_calls == 1
    assert authority.close_calls == 1


def test_failed_construction_closes_only_resources_owned_by_the_store() -> None:
    composition = _composition()
    payload = _Payload()
    authority = _Authority()

    with pytest.raises(composition.CompositionValidationError):
        composition.StoreTopology(
            payload=composition.BackendRef(instance=payload, transfer_ownership=True),
            authority=composition.BackendRef(instance=authority),
            minimum_capabilities={"durable": True},
        ).resolve()

    assert payload.close_calls == 1
    assert authority.close_calls == 0


def test_legacy_selectors_factories_and_constructor_overload_are_absent() -> None:
    blob_store = ast.parse(
        (REPOSITORY_ROOT / "src/cacheness/storage/blob_store.py").read_text(encoding="utf-8")
    )
    class_node = next(node for node in blob_store.body if isinstance(node, ast.ClassDef) and node.name == "BlobStore")
    method_names = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}
    constructor = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    parameters = {argument.arg for argument in constructor.args.args + constructor.args.kwonlyargs}

    assert "_select_projection_backend" not in method_names
    assert "_create_lifecycle_authority" not in method_names
    assert "backend" not in parameters
    assert "metadata_backend" not in parameters


def test_direct_blob_store_uses_no_legacy_metadata_selector() -> None:
    source = (REPOSITORY_ROOT / "src/cacheness/storage/blob_store.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "_select_projection_backend",
        "_create_lifecycle_authority",
        "get_metadata_backend",
        "create_metadata_backend",
    ):
        assert forbidden not in source


def test_memory_tracer_uses_registered_same_process_participants(tmp_path: Path) -> None:
    """The first direct BlobStore slice keeps committed memory bytes in-process."""
    from cacheness.storage.blob_store import BlobStore

    composition = _composition()
    store_root = tmp_path / "memory-only-store"
    topology = composition.StoreTopology(
        payload=composition.BackendRef(name="memory"),
        authority=composition.BackendRef(name="memory"),
    )
    store = BlobStore(topology, cache_dir=store_root)
    try:
        written = store.put_entry({"answer": 42}, key="memory-tracer")
        observed = store.get_entry_info("memory-tracer")

        assert observed is not None
        assert observed.key == written.key == "memory-tracer"
        assert store.capabilities.durable is False
        assert store.capabilities.process_scope == "process"
        assert store.capabilities.host_scope == "process"
        assert store.capabilities.canonical_scan is True
        assert store.capabilities.index_acceleration is False
        assert not store_root.exists()
    finally:
        store.close()


def test_memory_tracer_keeps_exact_injected_instances_caller_owned(tmp_path: Path) -> None:
    """Injected memory participants survive the store close unless ownership transfers."""
    from cacheness.storage.backends.blob_backends import InMemoryBlobBackend
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority

    composition = _composition()
    payload = InMemoryBlobBackend()
    authority = InMemoryLifecycleAuthority()
    store = BlobStore(
        composition.StoreTopology(payload=payload, authority=authority),
        cache_dir=tmp_path / "injected-memory-store",
    )
    try:
        store.put_entry("injected", key="memory-injected")
        assert store.payload_backend is payload
        assert store.lifecycle_authority is authority
    finally:
        store.close()

    assert authority.read_entry("memory-injected") is not None
    assert payload.exists("memory://generations") is False


def test_sqlite_tracer_reopens_one_signed_canonical_descriptor(tmp_path: Path) -> None:
    """A format-2 filesystem/SQLite store persists no catalog mirror."""
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.composition import BackendRef, StoreTopology
    from cacheness.storage.manifest import BlobManifest, verify_current_manifest
    from cacheness.storage.sqlite_lifecycle_authority import AUTHORITY_RELATIVE_PATH

    root = tmp_path / "persistent-sqlite-store"

    def topology() -> StoreTopology:
        return StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        )

    with BlobStore(topology(), cache_dir=root) as first:
        first.initialize()
        receipt = first.put_entry(
            {"answer": 42}, key="persistent-tracer", metadata={"label": "answer"}
        )
        committed = first.lifecycle_authority.read_entry("persistent-tracer")
        assert committed is not None
        descriptor = BlobManifest.from_canonical_bytes(committed.manifest)
        verify_current_manifest(descriptor, first._authority_manifest_key())
        assert descriptor.key == receipt.key
        assert descriptor.user_metadata == {"label": "answer"}

    with sqlite3.connect(root / AUTHORITY_RELATIVE_PATH) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert not {name for name in tables if name.startswith("catalog_")}

    with BlobStore(topology(), cache_dir=root) as reopened:
        reopened.initialize()
        assert reopened.get("persistent-tracer") == {"answer": 42}
        observed = reopened.get_entry_info("persistent-tracer")
        assert observed is not None
        assert observed.key == receipt.key
        assert observed.generation == receipt.generation
        assert observed.metadata["metadata"]["label"] == "answer"


def test_selected_filesystem_participant_supplies_generation_io_at_its_own_root(
    tmp_path: Path,
) -> None:
    """An injected filesystem participant, not BlobStore.cache_dir, owns bytes."""
    from cacheness.storage.backends.blob_backends import FilesystemBlobBackend
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.catalog import CatalogField, CatalogSchema
    from cacheness.storage.composition import BackendRef, StoreTopology

    root_a = tmp_path / "selected-payload"
    root_b = tmp_path / "unselected-cache-dir"
    payload = FilesystemBlobBackend(root_a)
    schema = CatalogSchema(
        fields=(CatalogField("rank", "integer", default=0, queryable=True),),
        schema_id="participant-root",
    )
    topology = StoreTopology(
        payload=BackendRef(instance=payload),
        authority=BackendRef(name="sqlite", options={"root": root_a}),
    )
    store = BlobStore(topology, cache_dir=root_b)
    try:
        receipt = store.put_entry(
            {"payload": "selected"},
            key="selected-root",
            catalog_schema=schema,
            catalog_values={},
        )
        updated = store.update_catalog(
            receipt.key,
            catalog_schema=schema,
            catalog_values={"rank": 2},
            expected=receipt.expectation,
        )
        assert updated is not None
        assert store.guarded_handler_io.root == payload.base_dir
        assert any((root_a / "generations").rglob("*"))
        assert not (root_b / "generations").exists()
        assert store.get(receipt.key) == {"payload": "selected"}
    finally:
        store.close()
        payload.close()
