"""Security contract for validated cache metadata query fields."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import CacheQueryValidationError, CacheReason
from cacheness.query_validation import to_sqlite_json_path, validate_query_fields


@pytest.mark.parametrize(
    "field",
    [
        "experiment",
        "_private",
        "model_2.version",
        ".".join(["segment"] * 16),
        "a" * 255,
    ],
)
def test_validated_query_fields_accept_bounded_identifier_paths(field: str) -> None:
    """Supported identifiers retain exact nested-field compatibility."""
    assert validate_query_fields({field: "value"}) == (field,)
    assert to_sqlite_json_path(field) == f"$.{field}"


@pytest.mark.parametrize(
    "field",
    [
        "",
        "a..b",
        ".a",
        "a.",
        "a[0]",
        "$.a",
        "a\"b",
        "a b",
        "a;drop",
        "a--comment",
        "a/b",
        "a\x00b",
        ".".join(["a"] * 17),
        "a" * 256,
    ],
)
def test_validated_query_fields_reject_hostile_or_overbound_paths(field: str) -> None:
    """Every caller-supplied field fails closed with its exact stable reason."""
    with pytest.raises(CacheQueryValidationError) as error:
        validate_query_fields({field: "value"})

    assert error.value.context == {
        "field": field,
        "reason": CacheReason.INVALID_QUERY_FIELD.value,
    }


def test_empty_filter_mapping_is_distinct_from_an_empty_filter_key() -> None:
    """Query-all is supported, but a supplied empty key is never silently ignored."""
    assert validate_query_fields({}) == ()

    with pytest.raises(CacheQueryValidationError) as error:
        validate_query_fields({"": "value"})

    assert error.value.context["field"] == ""


class _SessionSpy:
    """Record any database boundary access during field validation."""

    def __init__(self) -> None:
        self.session_calls = 0
        self.execute_calls = 0

    def __call__(self) -> _SessionSpy:
        self.session_calls += 1
        return self

    def __enter__(self) -> _SessionSpy:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def execute(self, *args: object, **kwargs: object) -> list[object]:
        self.execute_calls += 1
        return []


def _sqlite_cache_with_session_spy(session_spy: _SessionSpy) -> UnifiedCache:
    """Build only the query_meta dependencies needed to assert call ordering."""
    cache = object.__new__(UnifiedCache)
    cache.actual_backend = "sqlite"
    cache.config = SimpleNamespace(
        metadata=SimpleNamespace(store_cache_key_params=True)
    )
    cache.metadata_backend = SimpleNamespace(SessionLocal=session_spy)
    return cache


@pytest.fixture
def sqlite_query_cache(tmp_path):
    """Create a SQLite cache whose metadata filter values are persisted."""
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="sqlite",
            store_cache_key_params=True,
        )
    )
    try:
        yield cache
    finally:
        cache.close()


class _RecordingSession:
    """Delegate session execution while retaining the SQLAlchemy statement."""

    def __init__(self, session: object, statements: list[object]) -> None:
        self._session = session
        self._statements = statements

    def __enter__(self) -> _RecordingSession:
        self._session.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return self._session.__exit__(exc_type, exc_value, traceback)

    def execute(self, statement: object, *args: object, **kwargs: object):
        self._statements.append(statement)
        return self._session.execute(statement, *args, **kwargs)


def test_query_meta_binds_validated_paths_and_values(
    sqlite_query_cache, monkeypatch
) -> None:
    """Caller paths and values stay in bound parameters, never SQL text."""
    sqlite_query_cache.put("record", experiment="bound-value", score=1.5)
    statements: list[object] = []
    session_factory = sqlite_query_cache.metadata_backend.SessionLocal

    def recording_session_factory() -> _RecordingSession:
        return _RecordingSession(session_factory(), statements)

    monkeypatch.setattr(
        sqlite_query_cache.metadata_backend,
        "SessionLocal",
        recording_session_factory,
    )

    entries = sqlite_query_cache.query_meta(experiment="bound-value", score=1.5)

    assert len(entries) == 1
    statement = statements[-1]
    compiled = statement.compile()
    assert "$.experiment" not in str(compiled)
    assert "$.score" not in str(compiled)
    assert "bound-value" not in str(compiled)
    assert compiled.params["query_meta_path_0"] == "$.experiment"
    assert compiled.params["query_meta_value_0"] == "str:bound-value"
    assert compiled.params["query_meta_path_1"] == "$.score"
    assert compiled.params["query_meta_value_1"] == 1.5


@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_invalid_fields_fail_before_session_or_execute_at_every_mapping_position(
    position: str,
) -> None:
    """All field names are validated before the SQLite session boundary opens."""
    session_spy = _SessionSpy()
    cache = _sqlite_cache_with_session_spy(session_spy)
    invalid_field = "unsafe[0]"
    safe_fields = [("first_safe", "first"), ("middle_safe", "middle")]

    filters = dict(safe_fields)
    insertion_index = {"first": 0, "middle": 1, "last": 2}[position]
    filter_items = list(filters.items())
    filter_items.insert(insertion_index, (invalid_field, "blocked"))

    with pytest.raises(CacheQueryValidationError) as error:
        cache.query_meta(**dict(filter_items))

    assert error.value.context == {
        "field": invalid_field,
        "reason": CacheReason.INVALID_QUERY_FIELD.value,
    }
    assert session_spy.session_calls == 0
    assert session_spy.execute_calls == 0


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_nonfinite_numeric_filters_fail_before_session_or_execute(value: float) -> None:
    """Non-finite thresholds are rejected before reaching SQLite casts."""
    session_spy = _SessionSpy()
    cache = _sqlite_cache_with_session_spy(session_spy)

    with pytest.raises(CacheQueryValidationError) as error:
        cache.query_meta(score=value)

    assert error.value.context == {
        "field": "score",
        "value": repr(value),
        "reason": CacheReason.INVALID_QUERY_VALUE.value,
    }
    assert session_spy.session_calls == 0
    assert session_spy.execute_calls == 0
