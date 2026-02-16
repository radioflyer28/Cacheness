"""Tests for custom metadata namespace isolation.

Verifies that custom metadata stored in one namespace is invisible,
unqueryable, and undeletable from another namespace — even when both
namespaces share the same SQLite database file.

Key: each namespace gets its own custom metadata table
(``custom_<schema>_<namespace_id>``) with FK pointing to the
namespace-specific ``cache_entries_<namespace_id>`` table.  The
default namespace uses the original unsuffixed table names.

Related: CACHE-oow
"""

import pytest
import tempfile
import shutil
from sqlalchemy import Column, String, Float, Integer

from cacheness import cacheness, CacheConfig
from cacheness.config import CacheMetadataConfig
from cacheness.custom_metadata import (
    custom_metadata_model,
    CustomMetadataBase,
    get_namespace_custom_model,
    _custom_metadata_registry,
)
from cacheness.metadata import Base, DEFAULT_NAMESPACE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_cache_dir():
    d = tempfile.mkdtemp()
    yield d
    try:
        shutil.rmtree(d)
    except (PermissionError, OSError):
        pass


@pytest.fixture(autouse=True)
def ensure_model_registered(sample_model):
    """Ensure the session-scoped template is in the registry for each test.

    Other test modules may call ``_reset_registry()`` which clears the
    global ``_custom_metadata_registry``.  Re-register the model here
    so that every test in *this* module starts with a clean, valid state.
    """
    if "ns_test" not in _custom_metadata_registry:
        _custom_metadata_registry["ns_test"] = sample_model
    yield


@pytest.fixture(scope="session")
def sample_model():
    """Register a simple custom metadata model for testing."""

    @custom_metadata_model("ns_test")
    class NsTestMeta(Base, CustomMetadataBase):
        __tablename__ = "custom_ns_test"
        __table_args__ = {"extend_existing": True}

        label = Column(String(100), nullable=False, index=True)
        score = Column(Float, nullable=False)
        version = Column(Integer, nullable=False, default=1)

    return NsTestMeta


def _make_cache(cache_dir, namespace="default"):
    config = CacheConfig(
        cache_dir=str(cache_dir),
        namespace=namespace,
        metadata_backend="sqlite",
        cleanup_on_init=False,
    )
    config.security.enable_signing = False
    return cacheness(config)


# ---------------------------------------------------------------------------
# Tests: EntityName pattern for custom metadata models
# ---------------------------------------------------------------------------


class TestNamespaceCustomModelFactory:
    """Verify get_namespace_custom_model creates correct ORM classes."""

    def test_default_namespace_returns_template(self, sample_model):
        """Default namespace should return the original template class."""
        ns_model = get_namespace_custom_model("ns_test", DEFAULT_NAMESPACE)
        assert ns_model is sample_model

    def test_non_default_namespace_creates_distinct_class(self, sample_model):
        """Non-default namespace should produce a different class."""
        ns_model = get_namespace_custom_model("ns_test", "analytics")
        assert ns_model is not sample_model
        assert ns_model.__tablename__ == "custom_ns_test_analytics"

    def test_non_default_namespace_fk_target(self, sample_model):
        """FK should point to the namespace-specific entries table."""
        ns_model = get_namespace_custom_model("ns_test", "analytics")
        fk_targets = {
            fk.target_fullname for fk in ns_model.__table__.c.cache_key.foreign_keys
        }
        assert "cache_entries_analytics.cache_key" in fk_targets

    def test_default_namespace_fk_target(self, sample_model):
        """Default ns FK should point to cache_entries (unsuffixed)."""
        ns_model = get_namespace_custom_model("ns_test", DEFAULT_NAMESPACE)
        fk_targets = {
            fk.target_fullname for fk in ns_model.__table__.c.cache_key.foreign_keys
        }
        assert "cache_entries.cache_key" in fk_targets

    def test_user_columns_preserved(self, sample_model):
        """All user-defined columns should exist on the namespace model."""
        ns_model = get_namespace_custom_model("ns_test", "analytics")
        col_names = {c.name for c in ns_model.__table__.columns}
        assert {"label", "score", "version"}.issubset(col_names)

    def test_namespace_model_cached(self, sample_model):
        """Calling again with same args should return the same class object."""
        m1 = get_namespace_custom_model("ns_test", "analytics")
        m2 = get_namespace_custom_model("ns_test", "analytics")
        assert m1 is m2

    def test_different_namespaces_produce_different_classes(self, sample_model):
        """Different namespace IDs should produce distinct classes."""
        m_a = get_namespace_custom_model("ns_test", "alpha")
        m_b = get_namespace_custom_model("ns_test", "beta")
        assert m_a is not m_b
        assert m_a.__tablename__ != m_b.__tablename__


# ---------------------------------------------------------------------------
# Tests: End-to-end custom metadata namespace isolation
# ---------------------------------------------------------------------------


class TestCustomMetadataNamespaceIsolation:
    """put/get custom metadata respects namespace boundaries."""

    def test_store_and_retrieve_in_non_default_namespace(
        self, temp_cache_dir, sample_model
    ):
        """Custom metadata works in a non-default namespace."""
        cache = _make_cache(temp_cache_dir, "analytics")

        meta = sample_model(label="run_1", score=0.95, version=1)
        cache.put({"data": 1}, key="k1", custom_metadata=meta)

        result = cache.get_custom_metadata_for_entry(key="k1")
        assert "ns_test" in result
        assert result["ns_test"].label == "run_1"
        assert result["ns_test"].score == 0.95

        cache.clear_all()

    def test_custom_metadata_isolated_between_namespaces(
        self, temp_cache_dir, sample_model
    ):
        """Custom metadata in ns_a is invisible from ns_b."""
        ns_a = _make_cache(temp_cache_dir, "alpha")
        ns_b = _make_cache(temp_cache_dir, "beta")

        meta_a = sample_model(label="from_alpha", score=0.9, version=1)
        ns_a.put({"val": "a"}, key="shared", custom_metadata=meta_a)

        meta_b = sample_model(label="from_beta", score=0.8, version=2)
        ns_b.put({"val": "b"}, key="shared", custom_metadata=meta_b)

        # Each namespace sees only its own custom metadata
        a_result = ns_a.get_custom_metadata_for_entry(key="shared")
        b_result = ns_b.get_custom_metadata_for_entry(key="shared")

        assert a_result["ns_test"].label == "from_alpha"
        assert b_result["ns_test"].label == "from_beta"

        ns_a.clear_all()
        ns_b.clear_all()

    def test_default_namespace_custom_metadata_unchanged(
        self, temp_cache_dir, sample_model
    ):
        """Default namespace uses the original (unsuffixed) table name."""
        cache = _make_cache(temp_cache_dir, "default")

        meta = sample_model(label="default_run", score=0.99, version=1)
        cache.put({"d": 1}, key="dk1", custom_metadata=meta)

        result = cache.get_custom_metadata_for_entry(key="dk1")
        assert "ns_test" in result
        assert result["ns_test"].label == "default_run"

        cache.clear_all()


# ---------------------------------------------------------------------------
# Tests: query_custom / query_custom_session namespace isolation
# ---------------------------------------------------------------------------


class TestCustomMetadataQueryIsolation:
    """query_custom() and query_custom_session() respect namespace."""

    def test_query_custom_scoped_to_namespace(self, temp_cache_dir, sample_model):
        """query_custom() returns only results from the calling namespace."""
        ns_a = _make_cache(temp_cache_dir, "alpha")
        ns_b = _make_cache(temp_cache_dir, "beta")

        meta_a = sample_model(label="alpha_exp", score=0.9, version=1)
        ns_a.put({"a": 1}, key="k1", custom_metadata=meta_a)

        meta_b = sample_model(label="beta_exp", score=0.8, version=1)
        ns_b.put({"b": 1}, key="k2", custom_metadata=meta_b)

        a_results = ns_a.query_custom("ns_test")
        b_results = ns_b.query_custom("ns_test")

        assert len(a_results) == 1
        assert a_results[0].label == "alpha_exp"

        assert len(b_results) == 1
        assert b_results[0].label == "beta_exp"

        ns_a.clear_all()
        ns_b.clear_all()

    def test_query_custom_with_filters_respects_namespace(
        self, temp_cache_dir, sample_model
    ):
        """Filters work correctly within namespace scope."""
        ns_a = _make_cache(temp_cache_dir, "alpha")
        ns_b = _make_cache(temp_cache_dir, "beta")

        for i in range(3):
            m = sample_model(label=f"a_run_{i}", score=0.5 + i * 0.1, version=1)
            ns_a.put({"x": i}, key=f"a_{i}", custom_metadata=m)

        for i in range(2):
            m = sample_model(label=f"b_run_{i}", score=0.6 + i * 0.1, version=1)
            ns_b.put({"y": i}, key=f"b_{i}", custom_metadata=m)

        # ns_a has 3 records, ns_b has 2
        assert len(ns_a.query_custom("ns_test")) == 3
        assert len(ns_b.query_custom("ns_test")) == 2

        # Filter by version in ns_a — should still only see ns_a's records
        a_filtered = ns_a.query_custom("ns_test", {"version": 1})
        assert len(a_filtered) == 3

        ns_a.clear_all()
        ns_b.clear_all()

    def test_query_custom_session_scoped_to_namespace(
        self, temp_cache_dir, sample_model
    ):
        """query_custom_session() context manager respects namespace."""
        ns_a = _make_cache(temp_cache_dir, "alpha")
        ns_b = _make_cache(temp_cache_dir, "beta")

        meta_a = sample_model(label="session_a", score=0.95, version=1)
        ns_a.put({"a": 1}, key="sk1", custom_metadata=meta_a)

        meta_b = sample_model(label="session_b", score=0.85, version=1)
        ns_b.put({"b": 1}, key="sk2", custom_metadata=meta_b)

        with ns_a.query_custom_session("ns_test") as q:
            a_results = q.all()
        with ns_b.query_custom_session("ns_test") as q:
            b_results = q.all()

        assert len(a_results) == 1
        assert a_results[0].label == "session_a"

        assert len(b_results) == 1
        assert b_results[0].label == "session_b"

        ns_a.clear_all()
        ns_b.clear_all()


# ---------------------------------------------------------------------------
# Tests: FK cascade delete works for non-default namespaces
# ---------------------------------------------------------------------------


class TestCustomMetadataNamespaceCascade:
    """FK CASCADE delete works for namespace-specific custom metadata tables."""

    def test_invalidate_cascades_custom_metadata(self, temp_cache_dir, sample_model):
        """Deleting a cache entry should cascade-delete its custom metadata."""
        cache = _make_cache(temp_cache_dir, "analytics")

        meta = sample_model(label="to_delete", score=0.7, version=1)
        cache.put({"d": 1}, key="del_k", custom_metadata=meta)

        # Verify metadata exists
        result = cache.get_custom_metadata_for_entry(key="del_k")
        assert "ns_test" in result

        # Delete the cache entry
        cache.invalidate(key="del_k")

        # Custom metadata should be gone (FK cascade)
        result_after = cache.get_custom_metadata_for_entry(key="del_k")
        assert result_after == {} or "ns_test" not in result_after

        cache.clear_all()

    def test_clear_all_cascades_custom_metadata(self, temp_cache_dir, sample_model):
        """clear_all() should cascade-delete all custom metadata in namespace."""
        cache = _make_cache(temp_cache_dir, "analytics")

        for i in range(3):
            m = sample_model(label=f"run_{i}", score=0.5 + i * 0.1, version=1)
            cache.put({"x": i}, key=f"clr_{i}", custom_metadata=m)

        assert len(cache.query_custom("ns_test")) == 3

        cache.clear_all()

        # After clear_all, no custom metadata should remain
        assert len(cache.query_custom("ns_test")) == 0


# ---------------------------------------------------------------------------
# Tests: Custom metadata doesn't leak across default + non-default
# ---------------------------------------------------------------------------


class TestDefaultVsNonDefaultIsolation:
    """Ensure default and non-default namespaces are isolated."""

    def test_default_and_custom_namespace_fully_isolated(
        self, temp_cache_dir, sample_model
    ):
        """Default and non-default namespaces must not share custom metadata."""
        default_cache = _make_cache(temp_cache_dir, "default")
        custom_cache = _make_cache(temp_cache_dir, "analytics")

        meta_default = sample_model(label="default_label", score=1.0, version=1)
        default_cache.put({"v": 1}, key="iso_k", custom_metadata=meta_default)

        meta_custom = sample_model(label="analytics_label", score=2.0, version=2)
        custom_cache.put({"v": 2}, key="iso_k", custom_metadata=meta_custom)

        # Each sees only its own
        d_result = default_cache.get_custom_metadata_for_entry(key="iso_k")
        c_result = custom_cache.get_custom_metadata_for_entry(key="iso_k")

        assert d_result["ns_test"].label == "default_label"
        assert c_result["ns_test"].label == "analytics_label"

        # query_custom is also isolated
        d_all = default_cache.query_custom("ns_test")
        c_all = custom_cache.query_custom("ns_test")

        assert len(d_all) == 1
        assert d_all[0].score == 1.0

        assert len(c_all) == 1
        assert c_all[0].score == 2.0

        default_cache.clear_all()
        custom_cache.clear_all()
