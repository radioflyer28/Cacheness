"""Production compatibility tests for immutable stored-format evidence."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import shutil
import sqlite3
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from cacheness.config import (
    CacheConfig,
    CacheMetadataConfig,
    CompressionConfig,
    SecurityConfig,
)
from cacheness.core import UnifiedCache
from cacheness.decorators import cached
from cacheness.error_handling import CacheLegacyFormatError, CacheReason
from cacheness.handlers import ArrayHandler
from cacheness.metadata import JsonBackend, SqliteBackend


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "compat"
_SIGNED_FIXTURE_KEY = bytes.fromhex(
    "00112233445566778899aabbccddeefffedcba98765432100123456789abcdef"
)
_WRONG_SIGNED_FIXTURE_KEY = bytes.fromhex(
    "ffeeddccbbaa998877665544332211000123456789abcdeffedcba9876543210"
)


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


def _copy_nested_json_fixture(tmp_path: Path, fixture_id: str) -> tuple[Path, str]:
    """Copy nested JSON evidence while retaining its original bytes."""
    source = FIXTURE_ROOT / fixture_id
    target = tmp_path / "cache"
    target.mkdir()
    document = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    cache_key = next(iter(document["entries"]))
    payload_name = Path(document["entries"][cache_key]["metadata"]["actual_path"]).name
    shutil.copy2(source / "metadata.json", target / "cache_metadata.json")
    shutil.copy2(source / "payload.npz", target / payload_name)
    return target, cache_key


def _copy_current_sqlite_fixture(tmp_path: Path) -> tuple[Path, str, Path]:
    """Copy current denormalized SQLite evidence after inspecting only a temp copy."""
    source = FIXTURE_ROOT / "sqlite-columns-v0314"
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
            cache_key, actual_path = connection.execute(
                "SELECT cache_key, actual_path FROM cache_entries"
            ).fetchone()
        finally:
            connection.close()
    shutil.copy2(source / "metadata.sqlite3", database)
    shutil.copy2(source / "payload.npz", target / Path(actual_path).name)
    return target, cache_key, database


def _signed_legacy_cache(
    tmp_path: Path, signing_key: bytes, *, delete_invalid_signatures: bool = True
) -> tuple[UnifiedCache, str, Path]:
    """Construct the public signed-legacy read flow with a fixture key."""
    cache_dir, cache_key = _copy_split_map_fixture(tmp_path, "json-split-signed-v038")
    metadata_path = cache_dir / "cache_metadata.json"
    with pytest.warns(DeprecationWarning, match="legacy"):
        cache = UnifiedCache(
            CacheConfig(
                cache_dir=str(cache_dir),
                metadata_backend="json",
                compression=CompressionConfig(use_blosc2_arrays=False),
                security=SecurityConfig(
                    use_in_memory_key=True,
                    allow_unsigned_entries=False,
                    delete_invalid_signatures=delete_invalid_signatures,
                ),
            )
        )
    assert cache.signer is not None
    cache.signer.secret_key = signing_key
    return cache, cache_key, metadata_path


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

        def unchanged() -> bool:
            return _sha256(metadata_path) == before

    else:
        cache_dir, cache_key, database = _copy_metadata_json_fixture(tmp_path)
        before = _sqlite_snapshot(database)
        with pytest.warns(DeprecationWarning, match="legacy"):
            backend = SqliteBackend(str(database))

        def unchanged() -> bool:
            return _sqlite_snapshot(database) == before

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


def test_current_json_control_fixture_reads_through_production_backend_and_handler(
    tmp_path: Path,
) -> None:
    """The current JSON control keeps its source bytes while its entry is readable."""
    source_metadata = FIXTURE_ROOT / "json-nested-v0314" / "metadata.json"
    source_payload = FIXTURE_ROOT / "json-nested-v0314" / "payload.npz"
    source_before = (_sha256(source_metadata), _sha256(source_payload))
    cache_dir, cache_key = _copy_nested_json_fixture(tmp_path, "json-nested-v0314")
    metadata_path = cache_dir / "cache_metadata.json"
    copied_document = json.loads(metadata_path.read_text(encoding="utf-8"))
    copied_payload = cache_dir / Path(
        copied_document["entries"][cache_key]["metadata"]["actual_path"]
    ).name
    copy_before = (_sha256(metadata_path), _sha256(copied_payload))

    backend = JsonBackend(metadata_path)
    entry = backend.get_entry(cache_key)
    assert entry is not None
    restored = ArrayHandler().get(
        cache_dir / Path(entry["metadata"]["actual_path"]).name,
        entry["metadata"],
    )

    np.testing.assert_array_equal(restored, np.arange(6, dtype=np.int32).reshape(2, 3))
    assert (_sha256(source_metadata), _sha256(source_payload)) == source_before
    assert (_sha256(metadata_path), _sha256(copied_payload)) == copy_before


def test_current_sqlite_control_fixture_reads_through_production_backend_and_handler(
    tmp_path: Path,
) -> None:
    """The current SQLite control uses its real ORM backend and array reader."""
    source_database = FIXTURE_ROOT / "sqlite-columns-v0314" / "metadata.sqlite3"
    source_payload = FIXTURE_ROOT / "sqlite-columns-v0314" / "payload.npz"
    source_before = (_sqlite_snapshot(source_database), _sha256(source_payload))
    cache_dir, cache_key, database = _copy_current_sqlite_fixture(tmp_path)

    backend = SqliteBackend(str(database))
    try:
        entry = backend.get_entry(cache_key)
        assert entry is not None
        restored = ArrayHandler().get(
            cache_dir / Path(entry["metadata"]["actual_path"]).name,
            entry["metadata"],
        )
    finally:
        backend.close()

    np.testing.assert_array_equal(restored, np.arange(6, dtype=np.int32).reshape(2, 3))
    assert (_sqlite_snapshot(source_database), _sha256(source_payload)) == source_before


def test_signed_split_map_verifies_current_then_exact_legacy_on_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The six-field historical signature is checked before one handler read."""
    cache, cache_key, metadata_path = _signed_legacy_cache(tmp_path, _SIGNED_FIXTURE_KEY)
    before = _sha256(metadata_path)
    events: list[str] = []

    original_snapshot = cache.guarded_handler_io.open_snapshot

    @contextmanager
    def snapshot_spy(*args, **kwargs):
        events.append("snapshot")
        with original_snapshot(*args, **kwargs) as snapshot:
            yield snapshot

    original_hash = cache._calculate_file_hash

    def hash_spy(path: Path):
        events.append("digest")
        return original_hash(path)

    original_current_verify = cache.signer.verify_entry

    def current_verify_spy(*args, **kwargs):
        events.append("current")
        return original_current_verify(*args, **kwargs)

    original_legacy_verify = cache._verify_legacy_entry_signature

    def legacy_verify_spy(*args, **kwargs):
        events.append("legacy")
        return original_legacy_verify(*args, **kwargs)

    handler = cache.handlers.get_handler_by_type("array")
    original_handler_get = handler.get

    def handler_get_spy(*args, **kwargs):
        events.append("handler")
        return original_handler_get(*args, **kwargs)

    monkeypatch.setattr(cache.guarded_handler_io, "open_snapshot", snapshot_spy)
    monkeypatch.setattr(cache, "_calculate_file_hash", hash_spy)
    monkeypatch.setattr(cache.signer, "verify_entry", current_verify_spy)
    monkeypatch.setattr(cache, "_verify_legacy_entry_signature", legacy_verify_spy)
    monkeypatch.setattr(handler, "get", handler_get_spy)

    try:
        with warnings.catch_warnings(record=True) as emitted_warnings:
            warnings.simplefilter("always")
            restored = cache.get(cache_key=cache_key, ttl_hours=None)
        np.testing.assert_array_equal(restored, np.arange(6, dtype=np.int32).reshape(2, 3))
    finally:
        cache.close()

    assert any(
        warning.category is DeprecationWarning and "legacy" in str(warning.message)
        for warning in emitted_warnings
    )
    assert events == ["snapshot", "digest", "current", "legacy", "handler"]
    assert events.count("snapshot") == 1
    assert _sha256(metadata_path) == before


@pytest.mark.parametrize("delete_invalid_signatures", [True, False])
def test_wrong_signed_split_key_fails_typed_before_handler_or_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delete_invalid_signatures: bool,
) -> None:
    """A failed historical candidate is neither deserialized nor deleted."""
    cache, cache_key, metadata_path = _signed_legacy_cache(
        tmp_path,
        _WRONG_SIGNED_FIXTURE_KEY,
        delete_invalid_signatures=delete_invalid_signatures,
    )
    before = _sha256(metadata_path)
    payload_path = metadata_path.parent / f"compat_{cache_key}.npz"
    payload_before = _sha256(payload_path)
    events: list[str] = []

    original_snapshot = cache.guarded_handler_io.open_snapshot

    @contextmanager
    def snapshot_spy(*args, **kwargs):
        events.append("snapshot")
        with original_snapshot(*args, **kwargs) as snapshot:
            yield snapshot

    original_hash = cache._calculate_file_hash

    def hash_spy(path: Path):
        events.append("digest")
        return original_hash(path)

    original_current_verify = cache.signer.verify_entry

    def current_verify_spy(*args, **kwargs):
        events.append("current")
        return original_current_verify(*args, **kwargs)

    original_legacy_verify = cache._verify_legacy_entry_signature

    def legacy_verify_spy(*args, **kwargs):
        events.append("legacy")
        return original_legacy_verify(*args, **kwargs)

    handler = cache.handlers.get_handler_by_type("array")

    def fail_if_called(*_args, **_kwargs):
        events.append("handler")
        raise AssertionError("handler must not run after a failed signature")

    monkeypatch.setattr(cache.guarded_handler_io, "open_snapshot", snapshot_spy)
    monkeypatch.setattr(cache, "_calculate_file_hash", hash_spy)
    monkeypatch.setattr(cache.signer, "verify_entry", current_verify_spy)
    monkeypatch.setattr(cache, "_verify_legacy_entry_signature", legacy_verify_spy)
    monkeypatch.setattr(handler, "get", fail_if_called)

    try:
        with pytest.raises(CacheLegacyFormatError) as error:
            cache.get(cache_key=cache_key, ttl_hours=None)
        assert error.value.context["reason"] == CacheReason.INVALID_LEGACY_SIGNATURE.value
        assert cache.metadata_backend.get_entry(cache_key) is not None
    finally:
        cache.close()

    assert events == ["snapshot", "digest", "current", "legacy"]
    assert events.count("snapshot") == 1
    assert _sha256(metadata_path) == before
    assert payload_path.exists()
    assert _sha256(payload_path) == payload_before


def test_unrecognized_legacy_signature_metadata_is_not_treated_as_unsigned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rewrite_authority_manifest
) -> None:
    """Only the exact signed split-map discriminator can select legacy HMAC."""
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="memory",
            cleanup_on_init=False,
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(
                use_in_memory_key=True,
                allow_unsigned_entries=True,
                delete_invalid_signatures=False,
            ),
        )
    )
    cache_key = cache.put(np.arange(6, dtype=np.int32).reshape(2, 3))
    entry = cache.metadata_backend.get_entry(cache_key)
    assert entry is not None
    metadata = entry["metadata"]
    metadata.pop("entry_signature")
    metadata["legacy_entry_signature"] = "unrecognized-legacy-signature"

    def change(fields):
        fields["user_metadata"].pop("entry_signature")
        fields["user_metadata"]["legacy_entry_signature"] = "unrecognized-legacy-signature"

    rewrite_authority_manifest(cache._cache_blob_store, cache_key, change)

    handler = cache.handlers.get_handler_by_type("array")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("handler must not run for an unknown legacy signature")

    monkeypatch.setattr(handler, "get", fail_if_called)
    try:
        assert cache.get(cache_key=cache_key, ttl_hours=None) is None
    finally:
        cache.close()


def test_decorator_tries_one_historical_candidate_after_current_miss_without_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-unified decorator key is a single deprecated fallback."""
    cache_dir, legacy_storage_key = _copy_nested_json_fixture(
        tmp_path, "decorator-key-v0313"
    )
    metadata_path = cache_dir / "cache_metadata.json"
    before = _sha256(metadata_path)
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="json",
            cleanup_on_init=False,
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(enable_entry_signing=False),
        )
    )
    requested_keys: list[str] = []
    original_get_entry = cache.metadata_backend.get_entry

    def get_entry_spy(cache_key: str):
        requested_keys.append(cache_key)
        return original_get_entry(cache_key)

    def reject_scan():
        raise AssertionError("decorator fallback must not enumerate cache entries")

    monkeypatch.setattr(cache.metadata_backend, "get_entry", get_entry_spy)
    monkeypatch.setattr(cache.metadata_backend, "list_entries", reject_scan)
    function_calls: list[tuple[int, int]] = []

    def fixture_array(size: int, offset: int = 0) -> np.ndarray:
        function_calls.append((size, offset))
        return np.full((2, 3), -1, dtype=np.int32)

    fixture_array.__module__ = "compat_fixture_v0313"
    fixture_array.__qualname__ = "fixture_array"
    decorated = cached(
        cache_instance=cache,
        key_prefix="compat-v0313",
        ignore_errors=False,
    )(fixture_array)

    try:
        with warnings.catch_warnings(record=True) as emitted_warnings:
            warnings.simplefilter("always")
            restored = decorated(6, offset=0)
        np.testing.assert_array_equal(restored, np.arange(6, dtype=np.int32).reshape(2, 3))
    finally:
        cache.close()

    assert any(
        warning.category is DeprecationWarning and "legacy" in str(warning.message)
        for warning in emitted_warnings
    )
    assert function_calls == []
    assert len(requested_keys) == 2
    assert requested_keys[-1] == legacy_storage_key
    assert _sha256(metadata_path) == before


@pytest.mark.parametrize(
    "failure_factory",
    [
        lambda: CacheLegacyFormatError(
            "unexpected legacy state",
            reason=CacheReason.UNSUPPORTED_LEGACY_LAYOUT,
        ),
        lambda: OSError("unexpected bookkeeping failure"),
    ],
    ids=("other-typed", "ordinary"),
)
def test_post_success_bookkeeping_propagates_non_read_only_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_factory
) -> None:
    """Only the recognized post-success read-only condition may be suppressed."""
    cache, cache_key, metadata_path = _signed_legacy_cache(tmp_path, _SIGNED_FIXTURE_KEY)
    before = _sha256(metadata_path)
    failure = failure_factory()

    def fail_access(_cache_key: str) -> None:
        raise failure

    monkeypatch.setattr(cache.metadata_backend, "update_access_time", fail_access)
    try:
        with pytest.raises(type(failure), match=str(failure)):
            cache.get(cache_key=cache_key, ttl_hours=None)
    finally:
        cache.close()

    assert _sha256(metadata_path) == before


def test_pre_success_read_only_failure_is_never_suppressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only suppression begins only after reconstruction succeeds."""
    cache, cache_key, metadata_path = _signed_legacy_cache(tmp_path, _SIGNED_FIXTURE_KEY)
    before = _sha256(metadata_path)
    handler = cache.handlers.get_handler_by_type("array")

    def fail_before_success(*_args, **_kwargs):
        raise CacheLegacyFormatError(
            "injected before handler success",
            reason=CacheReason.READ_ONLY_LEGACY_STORE,
        )

    monkeypatch.setattr(cache, "_verify_legacy_entry_signature", lambda *_args: True)
    monkeypatch.setattr(handler, "get", fail_before_success)
    try:
        with pytest.raises(CacheLegacyFormatError) as error:
            cache.get(cache_key=cache_key, ttl_hours=None)
        _assert_legacy_reason(error)
    finally:
        cache.close()

    assert _sha256(metadata_path) == before
