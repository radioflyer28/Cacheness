"""Adapter-level contracts for generation-specific projection mutations."""

from __future__ import annotations

from pathlib import Path
from threading import Event, RLock, Thread
from types import SimpleNamespace

import pytest
from sqlalchemy import Column, String, create_engine, select
from sqlalchemy.orm import sessionmaker

from cacheness.custom_metadata import CacheMetadataLink, CustomMetadataBase, custom_metadata_model
from cacheness.core import UnifiedCache
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheReason,
)
from cacheness.metadata import (
    Base,
    CachedMetadataBackend,
    InMemoryBackend,
    JsonBackend,
    SqliteBackend,
)
from cacheness.storage.backends.postgresql_backend import (
    PgCacheStats,
    PostgresBackend,
    PostgresBase,
)


@custom_metadata_model("projection_contract")
class ProjectionContractMetadata(Base, CustomMetadataBase):
    """Metadata records used to prove promoted-token link ownership."""

    __tablename__ = "custom_projection_contract_metadata"

    label = Column(String(100), nullable=False)


def _entry(locator: str, *, payload: str = "value") -> dict:
    """Build the smallest projection row with one immutable locator token."""
    return {
        "data_type": "object",
        "description": payload,
        "file_size": len(payload),
        "metadata": {"actual_path": locator},
    }


@pytest.fixture(params=["memory", "json", "sqlite"])
def projection_backend(request: pytest.FixtureRequest, tmp_path: Path):
    """Provide every locally usable projection adapter."""
    if request.param == "memory":
        return InMemoryBackend()
    if request.param == "json":
        return JsonBackend(tmp_path / "projection.json")
    return SqliteBackend(str(tmp_path / "projection.db"))


def test_local_projection_adapters_share_exact_mutation_outcomes(projection_backend):
    """Absent/create, exact replacement/removal, convergence, and mismatch agree."""
    key = "0123456789abcdef"
    m1 = _entry("/projection/m1")
    m2 = _entry("/projection/m2")

    assert projection_backend.conditional_projection_mutation(
        key, expected_locator=None, replacement=m1
    ).status == "applied"
    assert projection_backend.conditional_projection_mutation(
        key, expected_locator="/projection/m1", replacement=m1
    ).status == "converged"
    assert projection_backend.conditional_projection_mutation(
        key, expected_locator="/projection/m1", replacement=m2
    ).status == "applied"
    assert projection_backend.conditional_projection_mutation(
        key, expected_locator="/projection/m1", replacement=None
    ).status == "mismatch"
    assert projection_backend.get_entry(key)["metadata"]["actual_path"] == "/projection/m2"
    assert projection_backend.conditional_projection_mutation(
        key, expected_locator="/projection/m2", replacement=None
    ).status == "applied"
    assert projection_backend.conditional_projection_mutation(
        key, expected_locator=None, replacement=None
    ).status == "converged"


def test_cached_projection_mismatch_evicts_an_m1_memory_entry(tmp_path: Path):
    """A cache wrapper never returns M1 after its backend reports M2 mismatch."""
    backend = InMemoryBackend()
    config = SimpleNamespace(
        enable_memory_cache=True,
        memory_cache_type="lru",
        memory_cache_maxsize=16,
        memory_cache_ttl_seconds=60,
        memory_cache_stats=False,
    )
    cached = CachedMetadataBackend(backend, config)
    key = "0123456789abcdef"
    backend.put_entry(key, _entry("/projection/m1"))
    assert cached.get_entry(key)["metadata"]["actual_path"] == "/projection/m1"

    assert backend.conditional_projection_mutation(
        key,
        expected_locator="/projection/m1",
        replacement=_entry("/projection/m2"),
    ).status == "applied"
    assert cached.conditional_projection_mutation(
        key,
        expected_locator="/projection/m1",
        replacement=None,
    ).status == "mismatch"
    assert cached.get_entry(key)["metadata"]["actual_path"] == "/projection/m2"


def test_json_cross_instance_stale_writer_reloads_inside_projection_commit_guard(
    tmp_path: Path,
) -> None:
    """M2 wins when A waits before its short JSON compare-and-replace section."""
    first = JsonBackend(tmp_path / "projection.json")
    second = JsonBackend(tmp_path / "projection.json")
    key = "0123456789abcdef"
    assert first.conditional_projection_mutation(
        key, expected_locator=None, replacement=_entry("/projection/m1")
    ).status == "applied"
    waiting = Event()
    release = Event()
    outcomes: list[str] = []

    def pause_before_lock(boundary: str) -> None:
        if boundary == "projection_commit.before_descriptor_lock":
            waiting.set()
            assert release.wait(timeout=5)

    def stale_writer() -> None:
        result = first.conditional_projection_mutation(
            key,
            expected_locator="/projection/m1",
            replacement=_entry("/projection/a"),
        )
        outcomes.append(result.status)

    first._projection_commit_hook = pause_before_lock
    writer = Thread(target=stale_writer)
    writer.start()
    assert waiting.wait(timeout=5)
    assert second.conditional_projection_mutation(
        key,
        expected_locator="/projection/m1",
        replacement=_entry("/projection/m2"),
    ).status == "applied"
    release.set()
    writer.join(timeout=5)

    assert outcomes == ["mismatch"]
    assert first.get_entry(key)["metadata"]["actual_path"] == "/projection/m2"


def _mocked_postgres_backend() -> PostgresBackend:
    """Construct a PostgreSQL mapper against SQLite without a live service."""
    custom_metadata_model("projection_contract")(ProjectionContractMetadata)
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._lock = RLock()
    backend.engine = create_engine("sqlite://")
    backend.SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=backend.engine
    )
    PostgresBase.metadata.create_all(backend.engine)
    Base.metadata.create_all(backend.engine)
    with backend.SessionLocal() as session:
        session.add(PgCacheStats(id=1))
        session.commit()
    return backend


def test_mocked_postgres_projection_mutation_has_exact_row_count_semantics():
    """PostgreSQL parity uses the same applied, converged, and mismatch results."""
    backend = _mocked_postgres_backend()
    key = "0123456789abcdef"
    assert backend.conditional_projection_mutation(
        key, expected_locator=None, replacement=_entry("/projection/m1")
    ).status == "applied"
    assert backend.conditional_projection_mutation(
        key,
        expected_locator="/projection/m1",
        replacement=_entry("/projection/m1", payload="refresh"),
    ).status == "converged"
    assert backend.conditional_projection_mutation(
        key, expected_locator="/projection/other", replacement=None
    ).status == "mismatch"
    assert backend.conditional_projection_mutation(
        key, expected_locator="/projection/m1", replacement=None
    ).status == "applied"


def test_mocked_postgres_rechecks_promoted_locator_before_link_insertion():
    """Core-to-Postgres delayed M1 linking rolls back after peer M2 wins."""
    backend = _mocked_postgres_backend()
    key = "0123456789abcdef"
    cache = object.__new__(UnifiedCache)
    cache.actual_backend = "postgresql"
    cache._custom_metadata_enabled = True
    cache.metadata_backend = backend
    assert backend.conditional_projection_mutation(
        key, expected_locator=None, replacement=_entry("/projection/m1")
    ).status == "applied"
    cache._store_custom_metadata(
        key, ProjectionContractMetadata(label="m1"), expected_locator="/projection/m1"
    )
    assert backend.conditional_projection_mutation(
        key,
        expected_locator="/projection/m1",
        replacement=_entry("/projection/m2"),
    ).status == "applied"
    cache._store_custom_metadata(
        key, ProjectionContractMetadata(label="m2"), expected_locator="/projection/m2"
    )

    with pytest.raises(CacheBlobLifecycleConflictError):
        cache._store_custom_metadata(
            key,
            ProjectionContractMetadata(label="stale-a"),
            expected_locator="/projection/m1",
        )

    with backend.SessionLocal() as session:
        links = session.execute(select(CacheMetadataLink)).scalars().all()
        assert len(links) == 1
        assert session.execute(
            select(ProjectionContractMetadata).where(
                ProjectionContractMetadata.label == "stale-a"
            )
        ).scalars().all() == []
        assert session.execute(
            select(ProjectionContractMetadata).where(
                ProjectionContractMetadata.label == "m2"
            )
        ).scalars().one().id == links[0].metadata_id


def test_core_dispatches_promoted_token_to_postgres_link_transaction():
    """UnifiedCache forwards its exact promoted locator without re-observing it."""
    calls: list[tuple[str, str, list[object]]] = []

    class ConditionalStore:
        def store_custom_metadata_if_current(self, key, token, metadata_objects):
            calls.append((key, token, metadata_objects))

    cache = object.__new__(UnifiedCache)
    cache.actual_backend = "postgresql"
    cache._custom_metadata_enabled = True
    cache.metadata_backend = ConditionalStore()
    metadata = ProjectionContractMetadata(label="promoted")

    cache._store_custom_metadata(
        "0123456789abcdef", metadata, expected_locator="/projection/m1"
    )

    assert calls == [("0123456789abcdef", "/projection/m1", [metadata])]


def test_core_reports_missing_conditional_projection_capability_precisely():
    """A custom adapter without the contract fails closed with its capability name."""
    cache = object.__new__(UnifiedCache)
    cache.metadata_backend = object()

    with pytest.raises(CacheBlobBackendError) as error:
        cache._conditional_projection_mutation(
            "0123456789abcdef", expected_locator=None, replacement=None
        )

    assert error.value.context["capability"] == "conditional_projection_mutation"
    assert error.value.context["reason"] == CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED.value
