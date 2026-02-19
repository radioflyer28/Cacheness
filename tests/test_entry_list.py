"""
Tests for EntryList — the rich result wrapper for list_entries/query returns.
"""

from pathlib import Path

import pytest

from cacheness.entry_list import EntryList


# ── Sample data ───────────────────────────────────────────────────────

SAMPLE_ENTRIES = [
    {
        "cache_key": "alpha",
        "data_type": "object",
        "description": "first entry",
        "file_size": 1024,
        "created_at": "2026-01-01T00:00:00",
        "metadata": {"model": "xgboost"},
    },
    {
        "cache_key": "beta",
        "data_type": "dataframe",
        "description": "second entry",
        "file_size": 2048,
        "created_at": "2026-01-02T00:00:00",
        "metadata": {"model": "cnn"},
    },
    {
        "cache_key": "gamma",
        "data_type": "array",
        "description": "third entry",
        "file_size": 512,
        "created_at": "2026-01-03T00:00:00",
        "metadata": {"model": "rf"},
    },
]


# ── Construction & backward compat ────────────────────────────────────


class TestEntryListBackwardCompat:
    """EntryList must behave exactly like a plain list."""

    def test_isinstance_list(self):
        el = EntryList(SAMPLE_ENTRIES)
        assert isinstance(el, list)

    def test_len(self):
        el = EntryList(SAMPLE_ENTRIES)
        assert len(el) == 3

    def test_bool_empty(self):
        assert not EntryList([])
        assert EntryList(SAMPLE_ENTRIES)

    def test_iteration(self):
        el = EntryList(SAMPLE_ENTRIES)
        keys = [e["cache_key"] for e in el]
        assert keys == ["alpha", "beta", "gamma"]

    def test_indexing(self):
        el = EntryList(SAMPLE_ENTRIES)
        assert el[0]["cache_key"] == "alpha"
        assert el[-1]["cache_key"] == "gamma"

    def test_slicing_returns_entry_list(self):
        el = EntryList(SAMPLE_ENTRIES)
        sliced = el[:2]
        assert isinstance(sliced, EntryList)
        assert len(sliced) == 2

    def test_contains(self):
        el = EntryList(SAMPLE_ENTRIES)
        assert SAMPLE_ENTRIES[0] in el

    def test_equality_with_plain_list(self):
        el = EntryList(SAMPLE_ENTRIES)
        assert el == SAMPLE_ENTRIES

    def test_add_returns_entry_list(self):
        el1 = EntryList(SAMPLE_ENTRIES[:1])
        el2 = EntryList(SAMPLE_ENTRIES[1:])
        combined = el1 + el2
        assert isinstance(combined, EntryList)
        assert len(combined) == 3

    def test_radd_with_plain_list(self):
        el = EntryList(SAMPLE_ENTRIES[:1])
        combined = SAMPLE_ENTRIES[1:] + el
        assert isinstance(combined, EntryList)
        assert len(combined) == 3

    def test_empty_constructor(self):
        el = EntryList()
        assert len(el) == 0
        assert isinstance(el, list)

    def test_none_constructor(self):
        el = EntryList(None)
        assert len(el) == 0

    def test_copy(self):
        el = EntryList(SAMPLE_ENTRIES)
        cp = el.copy()
        assert isinstance(cp, EntryList)
        assert cp == el
        assert cp is not el


# ── Convenience accessors ─────────────────────────────────────────────


class TestEntryListAccessors:
    def test_keys(self):
        el = EntryList(SAMPLE_ENTRIES)
        assert el.keys() == ["alpha", "beta", "gamma"]

    def test_keys_empty(self):
        assert EntryList().keys() == []

    def test_first(self):
        el = EntryList(SAMPLE_ENTRIES)
        assert el.first()["cache_key"] == "alpha"

    def test_first_empty(self):
        assert EntryList().first() is None

    def test_last(self):
        el = EntryList(SAMPLE_ENTRIES)
        assert el.last()["cache_key"] == "gamma"

    def test_last_empty(self):
        assert EntryList().last() is None


# ── Chainable operations ─────────────────────────────────────────────


class TestEntryListChaining:
    def test_sort_by_ascending(self):
        el = EntryList(SAMPLE_ENTRIES)
        sorted_el = el.sort_by("file_size")
        assert isinstance(sorted_el, EntryList)
        assert sorted_el.keys() == ["gamma", "alpha", "beta"]

    def test_sort_by_descending(self):
        el = EntryList(SAMPLE_ENTRIES)
        sorted_el = el.sort_by("file_size", reverse=True)
        assert sorted_el.keys() == ["beta", "alpha", "gamma"]

    def test_sort_by_does_not_mutate(self):
        el = EntryList(SAMPLE_ENTRIES)
        _ = el.sort_by("file_size")
        assert el.keys() == ["alpha", "beta", "gamma"]

    def test_sort_by_missing_field(self):
        el = EntryList(SAMPLE_ENTRIES)
        # Should not raise; uses default=None
        sorted_el = el.sort_by("nonexistent", default=0)
        assert len(sorted_el) == 3

    def test_filter(self):
        el = EntryList(SAMPLE_ENTRIES)
        filtered = el.filter(lambda e: e["file_size"] > 1000)
        assert isinstance(filtered, EntryList)
        assert filtered.keys() == ["alpha", "beta"]

    def test_filter_empty_result(self):
        el = EntryList(SAMPLE_ENTRIES)
        filtered = el.filter(lambda e: e["data_type"] == "nonexistent")
        assert len(filtered) == 0
        assert isinstance(filtered, EntryList)

    def test_chained_operations(self):
        el = EntryList(SAMPLE_ENTRIES)
        result = (
            el.filter(lambda e: e["file_size"] >= 512)
            .sort_by("file_size", reverse=True)
            .first()
        )
        assert result["cache_key"] == "beta"


# ── Serialization ─────────────────────────────────────────────────────


class TestEntryListSerialization:
    def test_to_json_string(self):
        el = EntryList(SAMPLE_ENTRIES[:1])
        json_str = el.to_json()
        assert '"cache_key": "alpha"' in json_str

    def test_to_json_file(self, tmp_path):
        el = EntryList(SAMPLE_ENTRIES)
        path = str(tmp_path / "entries.json")
        result = el.to_json(path)
        assert Path(path).exists()
        assert result == Path(path).read_text(encoding="utf-8")

    def test_to_json_empty(self):
        assert EntryList().to_json() == "[]"

    def test_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        el = EntryList(SAMPLE_ENTRIES)
        df = el.to_dataframe()
        assert len(df) == 3
        assert list(df["cache_key"]) == ["alpha", "beta", "gamma"]

    def test_to_dataframe_empty(self):
        pd = pytest.importorskip("pandas")
        df = EntryList().to_dataframe()
        assert len(df) == 0


# ── Representation ────────────────────────────────────────────────────


class TestEntryListRepr:
    def test_repr_empty(self):
        assert repr(EntryList()) == "EntryList([])"

    def test_repr_small(self):
        el = EntryList(SAMPLE_ENTRIES[:2])
        r = repr(el)
        assert "EntryList([" in r
        assert "'alpha'" in r
        assert "'beta'" in r
        assert "len=2" in r

    def test_repr_truncated(self):
        entries = [{"cache_key": f"key_{i}"} for i in range(10)]
        el = EntryList(entries)
        r = repr(el)
        assert "+7 more" in r
        assert "len=10" in r


# ── Integration with UnifiedCache ─────────────────────────────────────


class TestEntryListIntegration:
    def test_list_entries_returns_entry_list(self, tmp_path):
        from cacheness.core import UnifiedCache
        from cacheness.config import CacheConfig, CacheStorageConfig

        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path / "cache"))
        )
        cache = UnifiedCache(config)
        cache.put("value1", cache_key="key1", description="first")
        cache.put("value2", cache_key="key2", description="second")

        entries = cache.list_entries()
        assert isinstance(entries, EntryList)
        assert len(entries) == 2

        # Convenience methods work
        keys = entries.keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_query_meta_returns_entry_list(self, tmp_path):
        from cacheness.core import UnifiedCache
        from cacheness.config import (
            CacheConfig,
            CacheStorageConfig,
            CacheMetadataConfig,
        )

        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path / "cache")),
            metadata=CacheMetadataConfig(
                metadata_backend="sqlite",
                store_full_metadata=True,
            ),
        )
        cache = UnifiedCache(config)
        cache.put("val", cache_key="qm1", model_type="xgboost")
        cache.put("val", cache_key="qm2", model_type="cnn")

        result = cache.query_meta(model_type="xgboost")
        assert isinstance(result, EntryList)
        assert len(result) == 1
        assert result.first()["cache_key"] == "qm1"

    def test_list_entries_sort_by_integration(self, tmp_path):
        from cacheness.core import UnifiedCache
        from cacheness.config import CacheConfig, CacheStorageConfig

        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path / "cache"))
        )
        cache = UnifiedCache(config)
        cache.put("small", cache_key="s")
        cache.put({"a": 1, "b": 2, "c": list(range(1000))}, cache_key="large")

        entries = cache.list_entries()
        sorted_entries = entries.sort_by("size_mb", reverse=True)
        assert isinstance(sorted_entries, EntryList)
        assert sorted_entries.first()["cache_key"] == "large"
