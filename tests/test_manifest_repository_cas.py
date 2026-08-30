"""Exact-record conditional publication contracts for local manifests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.config import LifecycleLimits
from cacheness.error_handling import CacheBlobBackendError, CacheBlobLifecycleConflictError
from cacheness.metadata import InMemoryBackend, JsonBackend, SqliteBackend
from cacheness.storage.manifest_repository import (
    InMemoryManifestRepository,
    JsonManifestRepository,
    ManifestCursor,
    ManifestExpectation,
    SqliteManifestRepository,
)


def _record(label: str) -> bytes:
    """Return deliberately opaque canonical-record stand-ins for repository tests."""
    return f"canonical-manifest-record::{label}".encode("utf-8")


def _repository_pair(tmp_path: Path, backend_name: str):
    """Build independently constructed repositories over one local topology."""
    if backend_name == "memory":
        backend = InMemoryBackend()
        return (
            InMemoryManifestRepository(backend),
            InMemoryManifestRepository(backend),
            (backend,),
        )
    if backend_name == "json":
        metadata_path = tmp_path / "metadata.json"
        first_backend = JsonBackend(metadata_path)
        second_backend = JsonBackend(metadata_path)
        return (
            JsonManifestRepository(first_backend),
            JsonManifestRepository(second_backend),
            (first_backend, second_backend),
        )
    metadata_path = tmp_path / "metadata.db"
    first_backend = SqliteBackend(metadata_path)
    second_backend = SqliteBackend(metadata_path)
    return (
        SqliteManifestRepository(first_backend),
        SqliteManifestRepository(second_backend),
        (first_backend, second_backend),
    )


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_create_if_absent_has_one_independent_repository_winner(
    tmp_path: Path, backend_name: str
) -> None:
    """A second absence contender loses without altering the exact winner bytes."""
    first_repo, second_repo, backends = _repository_pair(tmp_path, backend_name)
    winner = _record("winner")
    loser = _record("loser")
    try:
        first_repo.publish_if_expected(
            "key", ManifestExpectation.absent(), winner
        )

        with pytest.raises(CacheBlobLifecycleConflictError):
            second_repo.publish_if_expected(
                "key", ManifestExpectation.absent(), loser
            )

        assert first_repo.get_raw("key") == winner
        assert second_repo.get_raw("key") == winner
    finally:
        for backend in backends:
            backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_exact_record_cas_rejects_same_generation_stale_patch(
    tmp_path: Path, backend_name: str
) -> None:
    """A record digest prevents last-write-wins updates within one generation."""
    first_repo, second_repo, backends = _repository_pair(tmp_path, backend_name)
    original = _record("generation-one-metadata-one")
    winner = _record("generation-one-metadata-two")
    stale_loser = _record("generation-one-metadata-three")
    expectation = ManifestExpectation.from_authenticated_record("generation-one", original)
    try:
        first_repo.put_raw("key", original)
        first_repo.publish_if_expected("key", expectation, winner)

        with pytest.raises(CacheBlobLifecycleConflictError):
            second_repo.publish_if_expected("key", expectation, stale_loser)

        assert first_repo.get_raw("key") == winner
        assert second_repo.get_raw("key") == winner
    finally:
        for backend in backends:
            backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_remove_if_expected_cannot_retire_a_replaced_record(
    tmp_path: Path, backend_name: str
) -> None:
    """Stale reclamation preserves a record published after the observed bytes."""
    first_repo, second_repo, backends = _repository_pair(tmp_path, backend_name)
    original = _record("old")
    replacement = _record("new")
    old_expectation = ManifestExpectation.from_authenticated_record("old", original)
    new_expectation = ManifestExpectation.from_authenticated_record("new", replacement)
    try:
        first_repo.put_raw("key", original)
        first_repo.publish_if_expected("key", old_expectation, replacement)

        with pytest.raises(CacheBlobLifecycleConflictError):
            second_repo.remove_if_expected("key", old_expectation)

        assert first_repo.get_raw("key") == replacement
        second_repo.remove_if_expected("key", new_expectation)
        assert first_repo.get_raw("key") is None
    finally:
        for backend in backends:
            backend.close()


def test_sqlite_cas_rolls_back_compatibility_projection_with_raw_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw-row failure leaves both SQLite projections at their prior winner."""
    backend = SqliteBackend(tmp_path / "metadata.db")
    repository = SqliteManifestRepository(backend)
    original = _record("original")
    replacement = _record("replacement")
    original_entry = {"description": "original", "data_type": "object"}
    replacement_entry = {"description": "replacement", "data_type": "object"}
    repository.put_raw("key", original, entry_data=original_entry)
    expectation = ManifestExpectation.from_authenticated_record("generation", original)

    def fail_raw_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected raw record failure")

    monkeypatch.setattr(repository, "_write_raw_row", fail_raw_write)
    try:
        with pytest.raises(CacheBlobBackendError):
            repository.publish_if_expected(
                "key", expectation, replacement, entry_data=replacement_entry
            )

        assert repository.get_raw("key") == original
        assert backend.get_entry("key")["description"] == "original"
    finally:
        backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_manifest_pages_are_stable_bounded_and_retain_the_supplied_limits(
    tmp_path: Path, backend_name: str
) -> None:
    """Local adapters expose two-item opaque pages without a local policy copy."""
    repository, _other, backends = _repository_pair(tmp_path, backend_name)
    limits = LifecycleLimits(manifest_page_size=2)
    repository = type(repository)(repository.backend, lifecycle_limits=limits)
    records = {key: _record(key) for key in ("delta", "alpha", "charlie", "bravo")}
    try:
        for key, record in records.items():
            repository.put_raw(key, record)

        first = repository.list_page()
        assert repository.lifecycle_limits is limits
        assert [key for key, _raw in first.entries] == ["alpha", "bravo"]
        assert [raw for _key, raw in first.entries] == [records["alpha"], records["bravo"]]
        assert first.next_cursor == ManifestCursor("bravo")

        second = repository.list_page(first.next_cursor)
        assert [key for key, _raw in second.entries] == ["charlie", "delta"]
        assert second.next_cursor is None
    finally:
        for backend in backends:
            backend.close()
