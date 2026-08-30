"""Contract tests for local canonical manifest repositories."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobBackendError, CacheReason
from cacheness.metadata import InMemoryBackend, JsonBackend, SqliteBackend
from cacheness.storage.integrity import sign_hmac_sha256
from cacheness.storage.manifest import BlobManifestV1
from cacheness.storage.manifest_repository import (
    InMemoryManifestRepository,
    JsonManifestRepository,
    SqliteManifestRepository,
)


_MANIFEST_KEY = b"manifest-backend-contract-key-000"


def _canonical_record(*, label: str = "first") -> bytes:
    """Build canonical bytes with nested Unicode metadata for persistence checks."""
    manifest = BlobManifestV1(
        schema_version=1,
        key=f"key-{label}",
        generation=f"generation-{label}",
        state="committed",
        locator=f"payload-{label}.bin",
        handler_type="object",
        payload_format="pickle",
        payload_format_version=1,
        digest_algorithm="sha256",
        digest="0" * 64,
        byte_size=7,
        created_at="2026-08-30T00:00:00+00:00",
        handler_metadata={"nested": {"label": "café", "values": [1, True]}},
        user_metadata={"owner": "测试", "nested": {"label": label}},
    )
    return manifest.with_signature(
        sign_hmac_sha256(manifest.signing_bytes(), _MANIFEST_KEY)
    ).canonical_bytes()


@pytest.fixture(params=("memory", "json", "sqlite"))
def local_repository(tmp_path: Path, request: pytest.FixtureRequest):
    """Construct each Phase 2 local adapter with its supporting backend."""
    if request.param == "memory":
        backend = InMemoryBackend()
        repository = InMemoryManifestRepository(backend)
    elif request.param == "json":
        backend = JsonBackend(tmp_path / "cache_metadata.json")
        repository = JsonManifestRepository(backend)
    else:
        backend = SqliteBackend(tmp_path / "cache_metadata.db")
        repository = SqliteManifestRepository(backend)

    try:
        yield repository, backend
    finally:
        backend.close()


def test_local_repositories_round_trip_exact_bytes_and_crud(local_repository):
    """Every local adapter preserves raw bytes and identical CRUD semantics."""
    repository, _ = local_repository
    first = _canonical_record(label="first")
    replacement = _canonical_record(label="replacement")

    assert repository.get_raw("alpha") is None
    repository.put_raw("alpha", first)
    assert repository.get_raw("alpha") == first
    assert repository.list_keys() == ["alpha"]

    repository.put_raw("alpha", replacement)
    assert repository.get_raw("alpha") == replacement
    assert repository.list_keys() == ["alpha"]

    repository.remove("alpha")
    assert repository.get_raw("alpha") is None
    assert repository.list_keys() == []


@pytest.mark.parametrize("backend_kind", ("json", "sqlite"))
def test_persistent_repositories_reopen_without_changing_canonical_bytes(
    tmp_path: Path, backend_kind: str
):
    """Persistent adapters retain Unicode and nested metadata without field projection."""
    record = _canonical_record(label=backend_kind)
    path = tmp_path / (
        "cache_metadata.json" if backend_kind == "json" else "cache_metadata.db"
    )

    if backend_kind == "json":
        backend = JsonBackend(path)
        repository = JsonManifestRepository(backend)
    else:
        backend = SqliteBackend(path)
        repository = SqliteManifestRepository(backend)
    try:
        repository.put_raw("persistent", record)
    finally:
        backend.close()

    if backend_kind == "json":
        reopened_backend = JsonBackend(path)
        reopened_repository = JsonManifestRepository(reopened_backend)
    else:
        reopened_backend = SqliteBackend(path)
        reopened_repository = SqliteManifestRepository(reopened_backend)
    try:
        assert reopened_repository.get_raw("persistent") == record
    finally:
        reopened_backend.close()


def test_sqlite_repository_uses_dedicated_blob_table_without_touching_legacy_rows(
    tmp_path: Path,
):
    """Canonical bytes stay outside the legacy fixed-column cache_entries table."""
    database_path = tmp_path / "cache_metadata.db"
    backend = SqliteBackend(database_path)
    backend.put_entry(
        "legacy",
        {
            "description": "legacy record",
            "data_type": "object",
            "file_size": 1,
            "metadata": {"actual_path": "legacy.bin"},
        },
    )
    try:
        with sqlite3.connect(database_path) as connection:
            before_schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cache_entries'"
            ).fetchone()
            before_rows = connection.execute(
                "SELECT cache_key, data_type, file_size, actual_path FROM cache_entries"
            ).fetchall()

        repository = SqliteManifestRepository(backend)
        record = _canonical_record(label="sqlite-isolation")
        repository.put_raw("canonical", record)

        with sqlite3.connect(database_path) as connection:
            after_schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cache_entries'"
            ).fetchone()
            after_rows = connection.execute(
                "SELECT cache_key, data_type, file_size, actual_path FROM cache_entries"
            ).fetchall()
            stored_record = connection.execute(
                "SELECT canonical_bytes FROM cacheness_manifest_records_v1 "
                "WHERE logical_key = ?",
                ("canonical",),
            ).fetchone()

        assert after_schema == before_schema
        assert after_rows == before_rows
        assert stored_record == (record,)
    finally:
        backend.close()


def test_repository_backend_failures_are_typed_and_preserve_their_cause(
    local_repository, monkeypatch: pytest.MonkeyPatch
):
    """Repository operational failures cannot be misreported as ordinary absence."""
    repository, backend = local_repository
    failure = OSError("backend is unavailable")

    if type(backend) is SqliteBackend:
        def fail_begin():
            raise failure

        monkeypatch.setattr(backend.engine, "begin", fail_begin)
    else:
        def fail_get_entry(_key: str):
            raise failure

        monkeypatch.setattr(backend, "get_entry", fail_get_entry)

    with pytest.raises(CacheBlobBackendError) as error:
        repository.get_raw("missing")

    assert error.value.context["reason"] == CacheReason.BLOB_BACKEND_FAILURE.value
    assert error.value.__cause__ is failure
