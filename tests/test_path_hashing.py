"""Path-key serialization coverage for the canonical configuration model."""

from __future__ import annotations

from cacheness.config import CacheConfig, SerializationConfig
from cacheness.serialization import serialize_for_cache_key


def test_same_content_at_different_paths_has_one_content_identity(tmp_path) -> None:
    """Content hashing intentionally does not encode an incidental filename."""

    first = tmp_path / "one.bin"
    second = tmp_path / "two.bin"
    first.write_bytes(b"same")
    second.write_bytes(b"same")

    assert serialize_for_cache_key(first) == serialize_for_cache_key(second)


def test_file_modification_changes_path_content_identity(tmp_path) -> None:
    """A content-hashed file cannot retain its old cache identity after mutation."""

    path = tmp_path / "payload.bin"
    path.write_bytes(b"old")
    before = serialize_for_cache_key(path)
    path.write_bytes(b"new")

    assert serialize_for_cache_key(path) != before


def test_directory_content_identity_tracks_nested_files(tmp_path) -> None:
    """Directory hashing includes its managed recursive contents."""

    directory = tmp_path / "tree"
    directory.mkdir()
    nested = directory / "payload.txt"
    nested.write_text("old", encoding="utf-8")
    before = serialize_for_cache_key(directory)
    nested.write_text("new", encoding="utf-8")

    assert serialize_for_cache_key(directory) != before


def test_missing_path_has_stable_non_content_identity(tmp_path) -> None:
    """Absent paths are represented without attempting unsafe file access."""

    missing = tmp_path / "missing"
    assert serialize_for_cache_key(missing) == serialize_for_cache_key(missing)


def test_path_string_mode_uses_path_not_file_bytes(tmp_path) -> None:
    """The explicit path mode preserves its documented non-content behavior."""

    path = tmp_path / "payload.bin"
    path.write_bytes(b"old")
    config = CacheConfig(serialization=SerializationConfig(hash_path_content=False))
    before = serialize_for_cache_key(path, config)
    path.write_bytes(b"new")

    assert serialize_for_cache_key(path, config) == before
