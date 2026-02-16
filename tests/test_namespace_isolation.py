"""Tests for namespace isolation: metadata, queries, and blob storage.

Verifies that two UnifiedCache instances sharing the same cache_dir but with
different namespaces are fully isolated — data stored in one namespace must
not be visible, queryable, or deletable from the other.

Related: CACHE-2o9
"""

import pytest
import numpy as np
from cacheness import cacheness
from cacheness.config import CacheConfig


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "shared_cache"


def _make_cache(cache_dir, namespace, backend="sqlite", store_full_metadata=False):
    """Create a cache instance with signing disabled for clean testing."""
    config = CacheConfig(
        cache_dir=str(cache_dir),
        namespace=namespace,
        metadata_backend=backend,
        cleanup_on_init=False,
    )
    config.security.enable_signing = False
    config.metadata.store_full_metadata = store_full_metadata
    return cacheness(config)


class TestNamespaceDataIsolation:
    """put/get/invalidate respect namespace boundaries."""

    def test_same_params_different_namespaces_are_isolated(self, cache_dir):
        """Two namespaces with identical params store separate data."""
        ns_a = _make_cache(cache_dir, "alpha")
        ns_b = _make_cache(cache_dir, "beta")

        ns_a.put({"source": "alpha"}, experiment="exp1")
        ns_b.put({"source": "beta"}, experiment="exp1")

        assert ns_a.get(experiment="exp1") == {"source": "alpha"}
        assert ns_b.get(experiment="exp1") == {"source": "beta"}

        ns_a.clear_all()
        ns_b.clear_all()

    def test_invalidate_in_one_namespace_does_not_affect_other(self, cache_dir):
        """Deleting data from ns_a leaves ns_b untouched."""
        ns_a = _make_cache(cache_dir, "alpha")
        ns_b = _make_cache(cache_dir, "beta")

        ns_a.put({"val": 1}, key="shared_key")
        ns_b.put({"val": 2}, key="shared_key")

        ns_a.invalidate(key="shared_key")

        assert ns_a.get(key="shared_key") is None
        assert ns_b.get(key="shared_key") == {"val": 2}

        ns_a.clear_all()
        ns_b.clear_all()

    def test_clear_all_scoped_to_namespace(self, cache_dir):
        """clear_all() only removes entries in the calling namespace."""
        ns_a = _make_cache(cache_dir, "alpha")
        ns_b = _make_cache(cache_dir, "beta")

        ns_a.put("a_data", key="k1")
        ns_b.put("b_data", key="k1")

        ns_a.clear_all()

        assert ns_a.get(key="k1") is None
        assert ns_b.get(key="k1") == "b_data"

        ns_b.clear_all()

    def test_list_entries_scoped_to_namespace(self, cache_dir):
        """list_entries() only shows entries from the calling namespace."""
        ns_a = _make_cache(cache_dir, "alpha")
        ns_b = _make_cache(cache_dir, "beta")

        ns_a.put("a1", key="k1")
        ns_a.put("a2", key="k2")
        ns_b.put("b1", key="k3")

        a_entries = ns_a.list_entries()
        b_entries = ns_b.list_entries()

        assert len(a_entries) == 2
        assert len(b_entries) == 1

        # Verify the data is actually from the correct namespace
        a_keys = {e.get("cache_key") for e in a_entries}
        b_keys = {e.get("cache_key") for e in b_entries}
        assert a_keys.isdisjoint(b_keys), "Namespaces should not share cache keys"

        ns_a.clear_all()
        ns_b.clear_all()


class TestNamespaceQueryMetaIsolation:
    """query_meta() respects namespace boundaries — the core CACHE-2o9 fix."""

    def test_query_meta_sqlite_respects_namespace(self, cache_dir):
        """query_meta() on SQLite must only return results from its namespace."""
        ns_a = _make_cache(
            cache_dir, "alpha", backend="sqlite", store_full_metadata=True
        )
        ns_b = _make_cache(
            cache_dir, "beta", backend="sqlite", store_full_metadata=True
        )

        # Use kwargs (not custom_metadata) — these go into metadata_dict
        ns_a.put(
            np.array([1, 2, 3]),
            prefix="Alpha result",
            project="proj_x",
            source="alpha",
        )
        ns_b.put(
            np.array([4, 5, 6]),
            prefix="Beta result",
            project="proj_x",
            source="beta",
        )

        # query_meta in ns_a should only see alpha's entry
        results_a = ns_a.query_meta(project="proj_x")
        assert results_a is not None
        assert len(results_a) == 1
        meta_a = results_a[0].get("metadata_dict", {})
        assert meta_a.get("source") == "alpha"

        # query_meta in ns_b should only see beta's entry
        results_b = ns_b.query_meta(project="proj_x")
        assert results_b is not None
        assert len(results_b) == 1
        meta_b = results_b[0].get("metadata_dict", {})
        assert meta_b.get("source") == "beta"

        ns_a.clear_all()
        ns_b.clear_all()

    def test_query_meta_without_filters_respects_namespace(self, cache_dir):
        """query_meta() with no filters still scoped to namespace."""
        ns_a = _make_cache(
            cache_dir, "alpha", backend="sqlite", store_full_metadata=True
        )
        ns_b = _make_cache(
            cache_dir, "beta", backend="sqlite", store_full_metadata=True
        )

        ns_a.put("a1", x=1)
        ns_a.put("a2", x=2)
        ns_b.put("b1", x=3)

        all_a = ns_a.query_meta()
        all_b = ns_b.query_meta()

        assert all_a is not None
        assert all_b is not None
        assert len(all_a) == 2
        assert len(all_b) == 1

        ns_a.clear_all()
        ns_b.clear_all()

    def test_query_meta_json_backend_respects_namespace(self, cache_dir):
        """query_meta() with JSON backend also respects namespace isolation."""
        ns_a = _make_cache(cache_dir, "alpha", backend="json", store_full_metadata=True)
        ns_b = _make_cache(cache_dir, "beta", backend="json", store_full_metadata=True)

        ns_a.put({"val": 1}, team="red", side="a")
        ns_b.put({"val": 2}, team="red", side="b")

        results_a = ns_a.query_meta(team="red")
        results_b = ns_b.query_meta(team="red")

        assert results_a is not None and len(results_a) == 1
        assert results_b is not None and len(results_b) == 1

        ns_a.clear_all()
        ns_b.clear_all()


class TestNamespaceDeleteMatchingIsolation:
    """delete_matching() / delete_where() respect namespace boundaries."""

    def test_delete_matching_scoped_to_namespace(self, cache_dir):
        """delete_matching() must only delete entries in its own namespace."""
        ns_a = _make_cache(
            cache_dir, "alpha", backend="sqlite", store_full_metadata=True
        )
        ns_b = _make_cache(
            cache_dir, "beta", backend="sqlite", store_full_metadata=True
        )

        ns_a.put("a_data", project="shared", side="a")
        ns_b.put("b_data", project="shared", side="b")

        deleted = ns_a.delete_matching(project="shared")
        assert deleted == 1

        # ns_a's entry is gone
        assert ns_a.get(project="shared", side="a") is None
        # ns_b's entry untouched
        assert ns_b.get(project="shared", side="b") == "b_data"

        ns_a.clear_all()
        ns_b.clear_all()


class TestNamespaceBlobIsolation:
    """Blob files are stored in separate directories per namespace."""

    def test_blob_files_in_separate_directories(self, cache_dir):
        """Each namespace's blobs live in cache_dir/{namespace}/."""
        from pathlib import Path

        ns_a = _make_cache(cache_dir, "alpha")
        ns_b = _make_cache(cache_dir, "beta")

        ns_a.put(np.array([1, 2, 3]), key="k1")
        ns_b.put(np.array([4, 5, 6]), key="k1")

        alpha_dir = Path(cache_dir) / "alpha"
        beta_dir = Path(cache_dir) / "beta"

        alpha_blobs = list(alpha_dir.rglob("*"))
        beta_blobs = list(beta_dir.rglob("*"))

        assert len([f for f in alpha_blobs if f.is_file()]) >= 1
        assert len([f for f in beta_blobs if f.is_file()]) >= 1

        ns_a.clear_all()
        ns_b.clear_all()
