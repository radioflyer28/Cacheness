"""Production compatibility tests for immutable stored-format evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cacheness.config import CacheConfig, CacheMetadataConfig, CompressionConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import CacheLegacyFormatError, CacheReason
from cacheness.metadata import JsonBackend, SqliteBackend


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "compat"


def _sha256(candidate: Path) -> str:
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def _assert_legacy_reason(error: pytest.ExceptionInfo[CacheLegacyFormatError]) -> None:
    assert error.value.context["reason"] == CacheReason.READ_ONLY_LEGACY_STORE.value


def _copy_split_map_fixture(tmp_path: Path, fixture_id: str) -> tuple[Path, str]:
    source = FIXTURE_ROOT / fixture_id
    target = tmp_path / "cache"
    target.mkdir()
    metadata_source = source / "metadata.json"
    payload_source = source / "payload.npz"
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    cache_key = next(iter(metadata["entries"]))
    payload_name = Path(metadata["entries"][cache_key]["actual_path"]).name
    shutil.copy2(metadata_source, target / "cache_metadata.json")
    shutil.copy2(payload_source, target / payload_name)
    return target, cache_key


def _sqlite_snapshot(database: Path) -> tuple[str, list[str], list[tuple[object, ...]], int]:
    source_digest = _sha256(database)
    with tempfile.TemporaryDirectory(prefix="cacheness-compat-") as temporary_dir:
        inspection_copy = Path(temporary_dir) / "metadata.sqlite3"
        shutil.copy2(database, inspection_copy)
        connection = sqlite3.connect(
            f"file:{inspection_copy.as_posix()}?mode=ro", uri=True
        )
        try:
            data_version_before = connection.execute("PRAGMA data_version").fetchone()[0]
            columns = [
                row[1]
                for row in connection.execute("PRAGMA table_info(cache_entries)").fetchall()
            ]
            rows = connection.execute("SELECT * FROM cache_entries").fetchall()
            data_version_after = connection.execute("PRAGMA data_version").fetchone()[0]
        finally:
            connection.close()
    assert data_version_before == data_version_after
    return source_digest, columns, rows, data_version_before


def _copy_metadata_json_fixture(tmp_path: Path) -> tuple[Path, str, Path]:
    source = FIXTURE_ROOT / "sqlite-metadata-json-v039"
    target = tmp_path / "cache"
    target.mkdir()
    database = target / "metadata.sqlite3"
    with tempfile.TemporaryDirectory(prefix="cacheness-compat-") as temporary_dir:
        inspection_copy = Path(temporary_dir) / "metadata.sqlite3"
        shutil.copy2(source / "metadata.sqlite3", inspection_copy)
        connection = sqlite3.connect(
            f"file:{inspection_copy.as_posix()}?mode=ro", uri=True
        )
        try:
            cache_key, metadata_json = connection.execute(
                "SELECT cache_key, metadata_json FROM cache_entries"
            ).fetchone()
        finally:
            connection.close()
    shutil.copy2(source / "metadata.sqlite3", database)
    payload_name = Path(json.loads(metadata_json)["actual_path"]).name
    shutil.copy2(source / "payload.npz", target / payload_name)
    return target, cache_key, database


def test_split_map_json_normalizes_exact_entry_and_rejects_mutation(tmp_path: Path) -> None:
    """Only the complete split-map layout is read through a deprecated adapter."""
    cache_dir, cache_key = _copy_split_map_fixture(tmp_path, "json-split-unsigned-v037")
    metadata_path = cache_dir / "cache_metadata.json"
    before = _sha256(metadata_path)

    with pytest.warns(DeprecationWarning, match="legacy"):
        backend = JsonBackend(metadata_path)

    entry = backend.get_entry(cache_key)
    assert entry == {
        "description": "0.3.7 unsigned split-map compatibility fixture",
        "data_type": "array",
        "prefix": "compat",
        "created_at": "2026-08-29T21:11:39.293100+00:00",
        "accessed_at": "2026-08-29T21:11:39.293100+00:00",
        "file_size": 219,
        "metadata": {
            "shape": [2, 3],
            "dtype": "int32",
            "storage_format": "npz",
            "compression": "zlib",
            "prefix": "compat",
            "actual_path": str(cache_dir / f"compat_{cache_key}.npz"),
            "file_hash": "cbc8742be97aa09e",
            "cache_key_params": {
                "fixture_id": "str:json-split-unsigned-v037",
            },
        },
    }
    with pytest.raises(CacheLegacyFormatError) as error:
        backend.put_entry(cache_key, {"data_type": "array"})
    _assert_legacy_reason(error)
    backend.record_legacy_read(cache_key)
    assert backend.legacy_compat_hits == 1
    assert backend.legacy_compat_access_times[cache_key]
    backend.close()
    assert _sha256(metadata_path) == before


@pytest.mark.parametrize(
    "operation",
    [
        lambda backend, cache_key: backend.put_entry(cache_key, {"data_type": "array"}),
        lambda backend, cache_key: backend.remove_entry(cache_key),
        lambda backend, cache_key: backend.clear_all(),
        lambda backend, cache_key: backend.cleanup_expired(1),
        lambda backend, cache_key: backend.cleanup_by_size(1),
        lambda backend, cache_key: backend.increment_hits(),
        lambda backend, cache_key: backend.increment_misses(),
        lambda backend, cache_key: backend.update_access_time(cache_key),
    ],
    ids=(
        "put",
        "remove",
        "clear",
        "expiry",
        "size",
        "hit",
        "miss",
        "access-time",
    ),
)
@pytest.mark.parametrize("backend_kind", ["json", "sqlite"])
def test_split_map_and_metadata_json_backends_reject_every_mutating_operation(
    tmp_path: Path,
    operation,
    backend_kind: str,
) -> None:
    """Every legacy persistence operation fails before changing evidence."""
    if backend_kind == "json":
        cache_dir, cache_key = _copy_split_map_fixture(
            tmp_path, "json-split-unsigned-v037"
        )
        metadata_path = cache_dir / "cache_metadata.json"
        before = _sha256(metadata_path)
        with pytest.warns(DeprecationWarning, match="legacy"):
            backend = JsonBackend(metadata_path)
        unchanged = lambda: _sha256(metadata_path) == before
    else:
        cache_dir, cache_key, database = _copy_metadata_json_fixture(tmp_path)
        before = _sqlite_snapshot(database)
        with pytest.warns(DeprecationWarning, match="legacy"):
            backend = SqliteBackend(str(database))
        unchanged = lambda: _sqlite_snapshot(database) == before

    try:
        with pytest.raises(CacheLegacyFormatError) as error:
            operation(backend, cache_key)
    finally:
        backend.close()

    _assert_legacy_reason(error)
    assert unchanged()


def test_split_map_mixed_layout_fails_typed_without_rewriting(tmp_path: Path) -> None:
    """Partial split maps are not guessed or silently converted."""
    cache_dir, _ = _copy_split_map_fixture(tmp_path, "json-split-unsigned-v037")
    metadata_path = cache_dir / "cache_metadata.json"
    mutated = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutated.pop("file_sizes")
    metadata_path.write_text(json.dumps(mutated), encoding="utf-8")
    before = _sha256(metadata_path)

    with pytest.raises(CacheLegacyFormatError) as error:
        JsonBackend(metadata_path)

    assert error.value.context["reason"] == CacheReason.UNSUPPORTED_LEGACY_LAYOUT.value
    assert _sha256(metadata_path) == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document, cache_key: document["entries"][cache_key].__setitem__(
            "shape", "not-a-list"
        ),
        lambda document, cache_key: document["file_sizes"].__setitem__(
            cache_key, "219"
        ),
        lambda document, cache_key: document["access_times"].__setitem__(
            cache_key, 0
        ),
    ],
    ids=("field-type", "map-type", "timestamp-map-type"),
)
def test_split_map_complete_but_malformed_values_fail_typed_without_rewriting(
    tmp_path: Path,
    mutate,
) -> None:
    """Matching keys alone must not make a split map a compatibility candidate."""
    cache_dir, cache_key = _copy_split_map_fixture(tmp_path, "json-split-unsigned-v037")
    metadata_path = cache_dir / "cache_metadata.json"
    malformed = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(malformed, cache_key)
    metadata_path.write_text(json.dumps(malformed), encoding="utf-8")
    before = _sha256(metadata_path)

    with pytest.raises(CacheLegacyFormatError) as error:
        JsonBackend(metadata_path)

    assert error.value.context["reason"] == CacheReason.UNSUPPORTED_LEGACY_LAYOUT.value
    assert _sha256(metadata_path) == before


def test_default_cleanup_reads_split_map_without_persisting_bookkeeping(
    tmp_path: Path,
) -> None:
    """A public legacy JSON read is read-only despite default startup cleanup."""
    cache_dir, cache_key = _copy_split_map_fixture(tmp_path, "json-split-unsigned-v037")
    metadata_path = cache_dir / "cache_metadata.json"
    source_path = FIXTURE_ROOT / "json-split-unsigned-v037" / "metadata.json"
    source_before = _sha256(source_path)
    copy_before = _sha256(metadata_path)

    with pytest.warns(DeprecationWarning, match="legacy"):
        cache = UnifiedCache(
            CacheConfig(
                cache_dir=str(cache_dir),
                metadata_backend="json",
                compression=CompressionConfig(use_blosc2_arrays=False),
            )
        )
    try:
        restored = cache.get(cache_key=cache_key, ttl_hours=None)
        np.testing.assert_array_equal(restored, np.arange(6, dtype=np.int32).reshape(2, 3))
        assert cache.metadata_backend.legacy_compat_hits == 1
        assert cache.metadata_backend.legacy_compat_access_times[cache_key]
    finally:
        cache.close()

    assert _sha256(source_path) == source_before
    assert _sha256(metadata_path) == copy_before


def test_default_cleanup_reads_metadata_json_sqlite_without_mutation(
    tmp_path: Path,
) -> None:
    """Legacy SQLite is detected before ORM table creation or statistics writes."""
    cache_dir, cache_key, database = _copy_metadata_json_fixture(tmp_path)
    source_path = FIXTURE_ROOT / "sqlite-metadata-json-v039" / "metadata.sqlite3"
    source_before = _sqlite_snapshot(source_path)
    copy_before = _sqlite_snapshot(database)

    with pytest.warns(DeprecationWarning, match="legacy"):
        cache = UnifiedCache(
            CacheConfig(
                cache_dir=str(cache_dir),
                metadata=CacheMetadataConfig(
                    metadata_backend="sqlite",
                    sqlite_db_file="metadata.sqlite3",
                ),
                compression=CompressionConfig(use_blosc2_arrays=False),
            )
        )
    try:
        restored = cache.get(cache_key=cache_key, ttl_hours=None)
        np.testing.assert_array_equal(restored, np.arange(6, dtype=np.int32).reshape(2, 3))
        assert cache.metadata_backend.legacy_compat_hits == 1
        assert cache.metadata_backend.legacy_compat_access_times[cache_key]
    finally:
        cache.close()

    assert _sqlite_snapshot(source_path) == source_before
    assert _sqlite_snapshot(database) == copy_before


@pytest.mark.parametrize(
    "reason",
    [
        CacheReason.UNSUPPORTED_LEGACY_LAYOUT,
        CacheReason.INVALID_LEGACY_SIGNATURE,
    ],
)
def test_default_cleanup_propagates_non_read_only_legacy_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: CacheReason,
) -> None:
    """Only the recognized read-only cleanup failure is suppressible."""
    cache_dir, _ = _copy_split_map_fixture(tmp_path, "json-split-unsigned-v037")
    metadata_path = cache_dir / "cache_metadata.json"
    before = _sha256(metadata_path)

    def fail_cleanup(_self, _ttl_hours: int) -> int:
        raise CacheLegacyFormatError("injected cleanup failure", reason=reason)

    monkeypatch.setattr(JsonBackend, "cleanup_expired", fail_cleanup)
    with pytest.raises(CacheLegacyFormatError) as error:
        UnifiedCache(CacheConfig(cache_dir=str(cache_dir), metadata_backend="json"))

    assert error.value.context["reason"] == reason.value
    assert _sha256(metadata_path) == before


def test_default_cleanup_propagates_untyped_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy compatibility never hides an ordinary startup cleanup failure."""
    cache_dir, _ = _copy_split_map_fixture(tmp_path, "json-split-unsigned-v037")
    metadata_path = cache_dir / "cache_metadata.json"
    before = _sha256(metadata_path)

    def fail_cleanup(_self, _ttl_hours: int) -> int:
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(JsonBackend, "cleanup_expired", fail_cleanup)
    with pytest.raises(OSError, match="injected cleanup failure"):
        UnifiedCache(CacheConfig(cache_dir=str(cache_dir), metadata_backend="json"))

    assert _sha256(metadata_path) == before


@pytest.mark.parametrize("backend_kind", ["json", "sqlite"])
def test_current_backend_persists_normal_access_bookkeeping(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    """Current JSON and denormalized SQLite retain ordinary persistence."""
    cache_dir = tmp_path / backend_kind
    metadata_config = (
        CacheMetadataConfig(metadata_backend="sqlite", sqlite_db_file="metadata.sqlite3")
        if backend_kind == "sqlite"
        else None
    )
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend=backend_kind,
            metadata=metadata_config,
            compression=CompressionConfig(use_blosc2_arrays=False),
        )
    )
    try:
        cache_key = cache.put(np.arange(6, dtype=np.int32).reshape(2, 3), fixture="current")
        if backend_kind == "json":
            before = json.loads(
                (cache_dir / "cache_metadata.json").read_text(encoding="utf-8")
            )
        else:
            before = (
                cache.metadata_backend.get_entry(cache_key),
                cache.metadata_backend.get_stats()["cache_hits"],
            )
        np.testing.assert_array_equal(cache.get(cache_key=cache_key, ttl_hours=None), np.arange(6, dtype=np.int32).reshape(2, 3))
        if backend_kind == "json":
            after = json.loads(
                (cache_dir / "cache_metadata.json").read_text(encoding="utf-8")
            )
        else:
            after = (
                cache.metadata_backend.get_entry(cache_key),
                cache.metadata_backend.get_stats()["cache_hits"],
            )
    finally:
        cache.close()

    if backend_kind == "json":
        assert after["cache_hits"] == before["cache_hits"] + 1
        assert (
            after["entries"][cache_key]["accessed_at"]
            != before["entries"][cache_key]["accessed_at"]
        )
    else:
        assert after[1] == before[1] + 1
        assert after[0]["accessed_at"] != before[0]["accessed_at"]
