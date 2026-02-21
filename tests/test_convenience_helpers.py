"""Tests for convenience helpers: put_with_meta, get_with_meta,
put_with_model, get_with_model, query_with_meta, query_with_model.

CACHE-dmr: Auto-populate metadata from cache key params.
"""

import uuid
import tempfile
import shutil
import pytest
from pathlib import Path
from sqlalchemy import Column, String, Float

from cacheness.core import UnifiedCache
from cacheness import CacheConfig
from cacheness.custom_metadata import (
    custom_metadata_model,
    CustomMetadataBase,
    _reset_registry,
)
from cacheness.metadata import Base


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset custom metadata registry before each test for isolation."""
    _reset_registry()
    yield
    _reset_registry()


@pytest.fixture
def temp_cache_dir(request):
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()

    def cleanup():
        import time
        import gc

        gc.collect()
        time.sleep(0.2)
        if Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                time.sleep(0.5)
                shutil.rmtree(temp_dir)

    request.addfinalizer(cleanup)
    return temp_dir


@pytest.fixture
def meta_cache(temp_cache_dir):
    """Cache with store_full_metadata=True (JSON backend — no ORM)."""
    config = CacheConfig(
        cache_dir=temp_cache_dir,
        metadata_backend="sqlite",
        store_full_metadata=True,
    )
    cache = UnifiedCache(config)
    yield cache
    cache.close()


@pytest.fixture
def meta_cache_no_store(temp_cache_dir):
    """Cache with store_full_metadata=False."""
    config = CacheConfig(
        cache_dir=temp_cache_dir,
        metadata_backend="sqlite",
        store_full_metadata=False,
    )
    cache = UnifiedCache(config)
    yield cache
    cache.close()


def _make_model():
    """Create a unique ORM model class for test isolation."""
    suffix = uuid.uuid4().hex[:8]
    schema_name = f"exp_{suffix}"
    table_name = f"custom_exp_{suffix}"

    @custom_metadata_model(schema_name)
    class _Model(Base, CustomMetadataBase):
        __tablename__ = table_name
        __table_args__ = {"extend_existing": True}

        experiment_id = Column(String(100), nullable=False, index=True)
        model_type = Column(String(50), nullable=False)
        accuracy = Column(Float, nullable=True)

    return _Model, schema_name


# ── put_with_meta / get_with_meta ───────────────────────────────────


class TestPutWithMeta:
    def test_basic_round_trip(self, meta_cache):
        """kwargs become both cache key and metadata_dict."""
        key = meta_cache.put_with_meta(
            {"val": 42}, experiment="exp_001", model="xgboost"
        )
        assert isinstance(key, str) and len(key) == 16

        result = meta_cache.get_with_meta(experiment="exp_001", model="xgboost")
        assert result is not None
        data, meta = result
        assert data == {"val": 42}
        assert meta["experiment"] == "exp_001"
        assert meta["model"] == "xgboost"

    def test_description_not_in_key(self, meta_cache):
        """description goes to the entry but not the metadata_dict."""
        key = meta_cache.put_with_meta("hello", description="my desc", tag="a")
        result = meta_cache.get_with_meta(tag="a")
        assert result is not None
        data, meta = result
        assert data == "hello"
        assert "description" not in meta  # description is not a metadata key
        assert meta["tag"] == "a"

    def test_overwrite_same_key(self, meta_cache):
        """Same kwargs overwrite the previous entry."""
        meta_cache.put_with_meta(100, x="1")
        meta_cache.put_with_meta(200, x="1")
        result = meta_cache.get_with_meta(x="1")
        assert result is not None
        assert result[0] == 200

    def test_different_kwargs_different_keys(self, meta_cache):
        """Different kwargs → different cache keys."""
        meta_cache.put_with_meta("a", k="1")
        meta_cache.put_with_meta("b", k="2")
        assert meta_cache.get_with_meta(k="1")[0] == "a"
        assert meta_cache.get_with_meta(k="2")[0] == "b"

    def test_no_kwargs_raises(self, meta_cache):
        with pytest.raises(ValueError, match="at least one keyword"):
            meta_cache.put_with_meta("data")

    def test_store_full_metadata_required(self, meta_cache_no_store):
        with pytest.raises(ValueError, match="store_full_metadata"):
            meta_cache_no_store.put_with_meta("data", x="1")


class TestGetWithMeta:
    def test_miss_returns_none(self, meta_cache):
        assert meta_cache.get_with_meta(x="nope") is None

    def test_no_kwargs_raises(self, meta_cache):
        with pytest.raises(ValueError, match="at least one keyword"):
            meta_cache.get_with_meta()

    def test_numeric_metadata_round_trip(self, meta_cache):
        meta_cache.put_with_meta([1, 2, 3], epoch=5, lr=0.001)
        result = meta_cache.get_with_meta(epoch=5, lr=0.001)
        assert result is not None
        data, meta = result
        assert data == [1, 2, 3]
        assert meta["epoch"] == 5
        assert meta["lr"] == 0.001


# ── put_with_model / get_with_model ─────────────────────────────────


class TestPutWithModel:
    def test_basic_round_trip(self, meta_cache):
        Model, schema = _make_model()
        key = meta_cache.put_with_model(
            {"val": 42},
            Model,
            experiment_id="exp_001",
            model_type="xgboost",
            accuracy=0.95,
        )
        assert isinstance(key, str) and len(key) == 16

        result = meta_cache.get_with_model(
            Model, experiment_id="exp_001", model_type="xgboost", accuracy=0.95
        )
        assert result is not None
        data, instance = result
        assert data == {"val": 42}
        assert instance.experiment_id == "exp_001"
        assert instance.model_type == "xgboost"
        assert instance.accuracy == 0.95

    def test_no_kwargs_raises(self, meta_cache):
        Model, _ = _make_model()
        with pytest.raises(ValueError, match="at least one keyword"):
            meta_cache.put_with_model("data", Model)

    def test_bad_kwargs_raises_type_error(self, meta_cache):
        Model, _ = _make_model()
        with pytest.raises(TypeError):
            meta_cache.put_with_model("data", Model, nonexistent_col="val")

    def test_overwrite_replaces_orm_row(self, meta_cache):
        Model, schema = _make_model()
        meta_cache.put_with_model(
            "v1", Model, experiment_id="e1", model_type="rf", accuracy=0.8
        )
        meta_cache.put_with_model(
            "v2", Model, experiment_id="e1", model_type="rf", accuracy=0.9
        )
        result = meta_cache.get_with_model(
            Model, experiment_id="e1", model_type="rf", accuracy=0.9
        )
        assert result is not None
        data, inst = result
        assert data == "v2"
        assert inst.accuracy == 0.9

    def test_requires_custom_metadata_backend(self, temp_cache_dir):
        """JSON backend doesn't support custom metadata."""
        config = CacheConfig(cache_dir=temp_cache_dir, metadata_backend="json")
        cache = UnifiedCache(config)
        Model, _ = _make_model()
        try:
            with pytest.raises(ValueError, match="SQLite or PostgreSQL"):
                cache.put_with_model("data", Model, experiment_id="e1", model_type="x")
        finally:
            cache.close()


class TestGetWithModel:
    def test_miss_returns_none(self, meta_cache):
        Model, _ = _make_model()
        assert (
            meta_cache.get_with_model(Model, experiment_id="nope", model_type="x")
            is None
        )

    def test_no_kwargs_raises(self, meta_cache):
        Model, _ = _make_model()
        with pytest.raises(ValueError, match="at least one keyword"):
            meta_cache.get_with_model(Model)

    def test_unregistered_model_returns_none(self, meta_cache):
        """A model class not decorated with @custom_metadata_model."""

        class FakeModel:
            pass

        meta_cache.put({"x": 1}, experiment="e1")
        result = meta_cache.get_with_model(FakeModel, experiment="e1")
        assert result is None


# ── query_with_meta ─────────────────────────────────────────────────


class TestQueryWithMeta:
    def test_basic_query(self, meta_cache):
        meta_cache.put_with_meta(10, color="red", size="large")
        meta_cache.put_with_meta(20, color="red", size="small")
        meta_cache.put_with_meta(30, color="blue", size="large")

        results = list(meta_cache.query_with_meta(color="red"))
        assert len(results) == 2
        values = {r[0] for r in results}
        assert values == {10, 20}

    def test_multi_filter(self, meta_cache):
        meta_cache.put_with_meta("hit", color="red", size="large")
        meta_cache.put_with_meta("miss", color="red", size="small")

        results = list(meta_cache.query_with_meta(color="red", size="large"))
        assert len(results) == 1
        assert results[0][0] == "hit"
        assert results[0][1]["size"] == "large"

    def test_empty_result(self, meta_cache):
        meta_cache.put_with_meta("data", tag="a")
        results = list(meta_cache.query_with_meta(tag="nope"))
        assert results == []

    def test_no_filters_returns_all(self, meta_cache):
        """query_with_meta with no kwargs passes through to query_meta."""
        meta_cache.put_with_meta("a", x="1")
        meta_cache.put_with_meta("b", x="2")
        results = list(meta_cache.query_with_meta())
        assert len(results) == 2

    def test_metadata_dict_in_results(self, meta_cache):
        meta_cache.put_with_meta(99, city="NYC", pop=8_000_000)
        results = list(meta_cache.query_with_meta(city="NYC"))
        assert len(results) == 1
        _, meta = results[0]
        assert meta["city"] == "NYC"
        assert meta["pop"] == 8_000_000

    def test_generator_is_lazy(self, meta_cache):
        """query_with_meta returns a generator, not a list."""
        meta_cache.put_with_meta("a", k="1")
        gen = meta_cache.query_with_meta(k="1")
        import types

        assert isinstance(gen, types.GeneratorType)


# ── query_with_model ────────────────────────────────────────────────


class TestQueryWithModel:
    def test_basic_query(self, meta_cache):
        Model, schema = _make_model()
        meta_cache.put_with_model(
            "a", Model, experiment_id="e1", model_type="xgb", accuracy=0.9
        )
        meta_cache.put_with_model(
            "b", Model, experiment_id="e2", model_type="rf", accuracy=0.8
        )
        meta_cache.put_with_model(
            "c", Model, experiment_id="e3", model_type="xgb", accuracy=0.7
        )

        results = list(meta_cache.query_with_model(Model, model_type="xgb"))
        assert len(results) == 2
        values = {r[0] for r in results}
        assert values == {"a", "c"}

    def test_query_returns_orm_instances(self, meta_cache):
        Model, _ = _make_model()
        meta_cache.put_with_model(
            "d", Model, experiment_id="e1", model_type="rf", accuracy=0.85
        )

        results = list(meta_cache.query_with_model(Model, model_type="rf"))
        assert len(results) == 1
        _, inst = results[0]
        assert inst.experiment_id == "e1"
        assert inst.accuracy == 0.85

    def test_no_match(self, meta_cache):
        Model, _ = _make_model()
        meta_cache.put_with_model(
            "d", Model, experiment_id="e1", model_type="rf", accuracy=0.5
        )

        results = list(meta_cache.query_with_model(Model, model_type="cnn"))
        assert results == []

    def test_unknown_column_raises(self, meta_cache):
        Model, _ = _make_model()
        with pytest.raises(ValueError, match="Unknown column"):
            list(meta_cache.query_with_model(Model, nonexistent="val"))

    def test_unregistered_model_raises(self, meta_cache):
        class NotRegistered:
            pass

        with pytest.raises(ValueError, match="not registered"):
            list(meta_cache.query_with_model(NotRegistered, x="1"))

    def test_generator_is_lazy(self, meta_cache):
        Model, _ = _make_model()
        meta_cache.put_with_model(
            "a", Model, experiment_id="e1", model_type="xgb", accuracy=0.9
        )
        gen = meta_cache.query_with_model(Model, model_type="xgb")
        import types

        assert isinstance(gen, types.GeneratorType)

    def test_no_filters_returns_all(self, meta_cache):
        Model, _ = _make_model()
        meta_cache.put_with_model(
            "a", Model, experiment_id="e1", model_type="xgb", accuracy=0.9
        )
        meta_cache.put_with_model(
            "b", Model, experiment_id="e2", model_type="rf", accuracy=0.8
        )
        results = list(meta_cache.query_with_model(Model))
        assert len(results) == 2


# ── Cross-API interop ───────────────────────────────────────────────


class TestInterop:
    """put_with_meta and put_with_model share the same key derivation,
    so identical kwargs should produce the same cache key."""

    def test_meta_and_model_same_key(self, meta_cache):
        """put_with_meta and put_with_model with identical kwargs hit same key."""
        Model, schema = _make_model()
        kw = {"experiment_id": "e1", "model_type": "xgb", "accuracy": 0.9}

        key1 = meta_cache.put_with_meta("via_meta", **kw)
        key2 = meta_cache.put_with_model("via_model", Model, **kw)
        assert key1 == key2

        # Last write wins
        result = meta_cache.get_with_meta(**kw)
        assert result is not None
        assert result[0] == "via_model"

    def test_get_after_model_put_with_plain_get(self, meta_cache):
        """Data stored via put_with_model is retrievable with plain get(on=...)."""
        Model, _ = _make_model()
        kw = {"experiment_id": "e1", "model_type": "xgb", "accuracy": 0.9}
        meta_cache.put_with_model("data", Model, **kw)

        data = meta_cache.get(on=kw)
        assert data == "data"

    def test_query_meta_finds_put_with_meta_entries(self, meta_cache):
        """query_meta() can find entries stored via put_with_meta."""
        meta_cache.put_with_meta("x", flavor="vanilla")
        results = meta_cache.query_meta(flavor="vanilla")
        assert results is not None and len(results) == 1
        assert results[0]["metadata_dict"]["flavor"] == "vanilla"


# ── on= key discriminator (CACHE-a2t) ──────────────────────────────


class TestOnParamPutWithMeta:
    """put_with_meta with on= extra key discriminator."""

    def test_on_changes_cache_key(self, meta_cache):
        """Same kwargs, different on= → different cache keys."""
        k1 = meta_cache.put_with_meta("v1", on={"epoch": 1}, model="xgb", lr=0.01)
        k2 = meta_cache.put_with_meta("v2", on={"epoch": 2}, model="xgb", lr=0.01)
        assert k1 != k2

    def test_on_not_in_metadata_dict(self, meta_cache):
        """on= params should NOT appear in stored metadata_dict."""
        meta_cache.put_with_meta("v1", on={"epoch": 5}, model="xgb", lr=0.01)
        result = meta_cache.get_with_meta(on={"epoch": 5}, model="xgb", lr=0.01)
        assert result is not None
        _, meta = result
        assert "epoch" not in meta  # on= key is NOT stored
        assert meta["model"] == "xgb"
        assert meta["lr"] == 0.01

    def test_on_round_trip(self, meta_cache):
        """Data stored with on= can be retrieved with matching on=."""
        meta_cache.put_with_meta(42, on={"run": "A"}, tag="x")
        result = meta_cache.get_with_meta(on={"run": "A"}, tag="x")
        assert result is not None
        assert result[0] == 42

    def test_on_miss_without_on(self, meta_cache):
        """Stored with on=, retrieved without on= → miss (different key)."""
        meta_cache.put_with_meta(42, on={"run": "A"}, tag="x")
        result = meta_cache.get_with_meta(tag="x")
        # Key without on= doesn't match key with on=
        assert result is None or result[0] != 42

    def test_on_none_is_default(self, meta_cache):
        """on=None is the same as not passing on= at all."""
        k1 = meta_cache.put_with_meta("a", on=None, tag="x")
        k2 = meta_cache.put_with_meta("b", tag="x")
        assert k1 == k2

    def test_overlap_raises(self, meta_cache):
        """on= keys overlapping with kwargs raises ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            meta_cache.put_with_meta("data", on={"tag": "x"}, tag="y")

    def test_empty_on_dict_is_noop(self, meta_cache):
        """on={} is the same as not passing on= at all."""
        k1 = meta_cache.put_with_meta("a", on={}, tag="x")
        k2 = meta_cache.put_with_meta("b", tag="x")
        assert k1 == k2

    def test_multiple_on_keys(self, meta_cache):
        """Multiple on= keys all participate in key derivation."""
        k1 = meta_cache.put_with_meta("a", on={"epoch": 1, "fold": 0}, model="xgb")
        k2 = meta_cache.put_with_meta("b", on={"epoch": 1, "fold": 1}, model="xgb")
        k3 = meta_cache.put_with_meta("c", on={"epoch": 2, "fold": 0}, model="xgb")
        assert len({k1, k2, k3}) == 3  # all distinct

    def test_query_with_meta_still_works_after_on(self, meta_cache):
        """Entries stored with on= are still queryable by metadata kwargs."""
        meta_cache.put_with_meta("v1", on={"epoch": 1}, model="xgb")
        meta_cache.put_with_meta("v2", on={"epoch": 2}, model="xgb")
        meta_cache.put_with_meta("v3", on={"epoch": 1}, model="rf")

        results = list(meta_cache.query_with_meta(model="xgb"))
        assert len(results) == 2
        values = {r[0] for r in results}
        assert values == {"v1", "v2"}


class TestOnParamGetWithMeta:
    """get_with_meta with on= extra key discriminator."""

    def test_overlap_raises(self, meta_cache):
        with pytest.raises(ValueError, match="overlap"):
            meta_cache.get_with_meta(on={"tag": "x"}, tag="y")

    def test_miss_with_wrong_on(self, meta_cache):
        """Wrong on= value → cache miss."""
        meta_cache.put_with_meta(99, on={"run": "A"}, tag="t")
        result = meta_cache.get_with_meta(on={"run": "B"}, tag="t")
        assert result is None


class TestOnParamPutWithModel:
    """put_with_model with on= extra key discriminator."""

    def test_on_changes_cache_key(self, meta_cache):
        Model, _ = _make_model()
        k1 = meta_cache.put_with_model(
            "v1", Model, on={"run": "A"}, experiment_id="e1", model_type="xgb"
        )
        k2 = meta_cache.put_with_model(
            "v2", Model, on={"run": "B"}, experiment_id="e1", model_type="xgb"
        )
        assert k1 != k2

    def test_on_not_in_orm_columns(self, meta_cache):
        """on= params should not be passed to the ORM constructor."""
        Model, _ = _make_model()
        meta_cache.put_with_model(
            "v1",
            Model,
            on={"run_id": "r1"},
            experiment_id="e1",
            model_type="xgb",
            accuracy=0.9,
        )
        result = meta_cache.get_with_model(
            Model,
            on={"run_id": "r1"},
            experiment_id="e1",
            model_type="xgb",
            accuracy=0.9,
        )
        assert result is not None
        _, inst = result
        assert inst.experiment_id == "e1"
        assert not hasattr(inst, "run_id")  # on= key not in ORM

    def test_overlap_raises(self, meta_cache):
        Model, _ = _make_model()
        with pytest.raises(ValueError, match="overlap"):
            meta_cache.put_with_model(
                "data",
                Model,
                on={"experiment_id": "e1"},
                experiment_id="e2",
                model_type="x",
            )

    def test_on_round_trip(self, meta_cache):
        Model, _ = _make_model()
        meta_cache.put_with_model(
            "data", Model, on={"fold": 3}, experiment_id="e1", model_type="xgb"
        )
        result = meta_cache.get_with_model(
            Model, on={"fold": 3}, experiment_id="e1", model_type="xgb"
        )
        assert result is not None
        assert result[0] == "data"


class TestOnParamGetWithModel:
    """get_with_model with on= extra key discriminator."""

    def test_overlap_raises(self, meta_cache):
        Model, _ = _make_model()
        with pytest.raises(ValueError, match="overlap"):
            meta_cache.get_with_model(
                Model, on={"experiment_id": "e1"}, experiment_id="e2", model_type="x"
            )

    def test_miss_with_wrong_on(self, meta_cache):
        Model, _ = _make_model()
        meta_cache.put_with_model(
            "v1", Model, on={"fold": 1}, experiment_id="e1", model_type="xgb"
        )
        result = meta_cache.get_with_model(
            Model, on={"fold": 999}, experiment_id="e1", model_type="xgb"
        )
        assert result is None
