"""Configuration options retained by the canonical key-serialization surface."""

from __future__ import annotations

from cacheness.config import CacheConfig, SerializationConfig
from cacheness.serialization import serialize_for_cache_key


def test_path_content_hashing_is_enabled_by_default(tmp_path) -> None:
    """Equivalent file bytes produce the same key evidence by default."""

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("identical", encoding="utf-8")
    second.write_text("identical", encoding="utf-8")

    assert serialize_for_cache_key(first, CacheConfig()) == serialize_for_cache_key(
        second, CacheConfig()
    )


def test_path_string_mode_preserves_distinct_paths(tmp_path) -> None:
    """Disabling content hashing makes the caller's path identity explicit."""

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("identical", encoding="utf-8")
    second.write_text("identical", encoding="utf-8")
    config = CacheConfig(serialization=SerializationConfig(hash_path_content=False))

    assert serialize_for_cache_key(first, config) != serialize_for_cache_key(
        second, config
    )


def test_content_change_changes_a_content_hashed_path_key(tmp_path) -> None:
    """Path-derived cache identity follows the current file content."""

    path = tmp_path / "payload.txt"
    path.write_text("before", encoding="utf-8")
    before = serialize_for_cache_key(path, CacheConfig())
    path.write_text("after", encoding="utf-8")

    assert serialize_for_cache_key(path, CacheConfig()) != before
