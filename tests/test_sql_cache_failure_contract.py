"""Failure-contract tests for the independent SQL pull-through cache."""

from dataclasses import FrozenInstanceError
from datetime import date
import logging

import pandas as pd
import pytest
from sqlalchemy import Column, Date, Float, MetaData, String, Table, func, select

from cacheness.sql_cache import (
    SqlCache,
    SqlCacheAdapter,
    SqlCacheFailure,
    SqlCacheFetchError,
    SqlCacheGapDetectionError,
    SqlCacheResult,
    SqlCacheWriteError,
)


class DeterministicAdapter(SqlCacheAdapter):
    """Adapter whose range outcomes make pull-through failure behavior reproducible."""

    def __init__(self, table, outcomes, *, parse_error=None):
        self.table = table
        self.outcomes = outcomes
        self.parse_error = parse_error
        self.fetch_calls = []

    def get_table_definition(self):
        return self.table

    def parse_query_params(self, **kwargs):
        if self.parse_error is not None:
            raise self.parse_error
        return kwargs

    def fetch_data(self, **kwargs):
        range_id = kwargs.get("range_id")
        self.fetch_calls.append(range_id)
        outcome = self.outcomes.get(range_id, "success")
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "empty":
            return pd.DataFrame(columns=["id", "date", "value"])
        return pd.DataFrame(
            [
                {
                    "id": f"range-{range_id}",
                    "date": date(2024, 1, range_id or 1),
                    "value": float(range_id or 1),
                }
            ]
        )


def _make_cache(outcomes, range_ids, *, gap_detector=None, parse_error=None):
    """Create a fresh SQLite cache whose custom detector returns known ranges."""
    metadata = MetaData()
    table = Table(
        "failure_contract_data",
        metadata,
        Column("id", String(50), primary_key=True),
        Column("date", Date, primary_key=True),
        Column("value", Float),
    )
    adapter = DeterministicAdapter(table, outcomes, parse_error=parse_error)

    if gap_detector is None:

        def gap_detector(query_params, cached_data, cache):
            return [{"range_id": range_id} for range_id in range_ids]

    cache = SqlCache.with_sqlite(
        ":memory:", table, adapter, gap_detector=gap_detector
    )
    return cache, adapter


def _cached_row_count(cache):
    with cache.Session() as session:
        return session.execute(select(func.count()).select_from(cache.table)).scalar_one()


@pytest.mark.parametrize(
    ("range_ids", "failed_ranges"),
    [
        ([1], [1]),
        ([1, 2, 3], [1]),
        ([1, 2, 3], [2]),
        ([1, 2, 3], [3]),
        ([1, 2, 3, 4], [1, 3]),
    ],
)
def test_strict_fetch_failures_stage_every_range_and_roll_back(
    range_ids, failed_ranges
):
    """Strict mode neither persists nor returns successful fragments."""
    causes = {range_id: RuntimeError(f"range {range_id} failed") for range_id in failed_ranges}
    cache, adapter = _make_cache(causes, range_ids)

    try:
        with pytest.raises(SqlCacheFetchError) as exc_info:
            cache.get_data(query_name="failure-contract")

        error = exc_info.value
        assert adapter.fetch_calls == range_ids
        assert [dict(failure.range_params)["range_id"] for failure in error.failures] == failed_ranges
        assert [failure.cause for failure in error.failures] == [causes[range_id] for range_id in failed_ranges]
        assert error.__cause__ is causes[failed_ranges[0]]
        assert _cached_row_count(cache) == 0
    finally:
        cache.close()


def test_strict_empty_fetch_is_a_completeness_error_and_rolls_back():
    """A successful call that yields no rows is not a complete missing range."""
    cache, adapter = _make_cache({2: "empty"}, [1, 2, 3])

    try:
        with pytest.raises(SqlCacheFetchError) as exc_info:
            cache.get_data(query_name="empty-contract")

        assert adapter.fetch_calls == [1, 2, 3]
        failure = exc_info.value.failures[0]
        assert failure.operation == "fetch"
        assert failure.error_type == "empty_response"
        assert dict(failure.range_params) == {"range_id": 2}
        assert _cached_row_count(cache) == 0
    finally:
        cache.close()


def test_best_effort_returns_an_inspectable_partial_result_and_structured_logs(caplog):
    """Best effort is opt-in and reports every missing-range failure in order."""
    first_error = RuntimeError("first failure")
    third_error = RuntimeError("third failure")
    cache, adapter = _make_cache({1: first_error, 2: "empty", 3: third_error, 4: "success"}, [1, 2, 3, 4])

    try:
        with caplog.at_level(logging.WARNING, logger="cacheness.sql_cache"):
            result = cache.get_data(
                query_name="best-effort-contract", failure_mode="best_effort"
            )

        assert isinstance(result, SqlCacheResult)
        assert result.is_partial is True
        assert isinstance(result.failures, tuple)
        assert [failure.error_type for failure in result.failures] == [
            type(first_error).__name__,
            "empty_response",
            type(third_error).__name__,
        ]
        assert [dict(failure.range_params)["range_id"] for failure in result.failures] == [1, 2, 3]
        assert result.data["id"].tolist() == ["range-4"]
        with pytest.raises(FrozenInstanceError):
            result.is_partial = False

        failure_logs = [
            record
            for record in caplog.records
            if getattr(record, "operation", None) == "fetch"
        ]
        assert len(failure_logs) == 3
        assert [record.range_params["range_id"] for record in failure_logs] == [1, 2, 3]
        assert all(record.failure_mode == "best_effort" for record in failure_logs)
    finally:
        cache.close()


def test_adapter_parse_failures_preserve_the_original_cause():
    """Caller adapter parsing cannot be obscured by a generic cache error."""
    parse_error = ValueError("invalid query")
    cache, _adapter = _make_cache({}, [], parse_error=parse_error)

    try:
        with pytest.raises(SqlCacheFetchError) as exc_info:
            cache.get_data(query_name="parse-contract")

        assert exc_info.value.__cause__ is parse_error
        assert exc_info.value.failures[0].operation == "parse_query_params"
        assert _cached_row_count(cache) == 0
    finally:
        cache.close()


def test_gap_detector_error_is_typed_by_default_and_uses_builtin_only_in_best_effort(caplog):
    """A caller-controlled gap detector gets no silent fallback in strict mode."""
    detector_error = RuntimeError("detector failed")

    def failing_detector(query_params, cached_data, cache):
        raise detector_error

    strict_cache, strict_adapter = _make_cache({}, [], gap_detector=failing_detector)
    best_effort_cache, best_effort_adapter = _make_cache(
        {None: "success"}, [], gap_detector=failing_detector
    )

    try:
        with pytest.raises(SqlCacheGapDetectionError) as exc_info:
            strict_cache.get_data(query_name="gap-contract")
        assert exc_info.value.__cause__ is detector_error
        assert strict_adapter.fetch_calls == []

        with caplog.at_level(logging.WARNING, logger="cacheness.sql_cache"):
            result = best_effort_cache.get_data(
                query_name="gap-contract", failure_mode="best_effort"
            )
        assert isinstance(result, SqlCacheResult)
        assert result.is_partial is True
        assert result.failures[0].operation == "gap_detection"
        assert result.failures[0].cause is detector_error
        assert best_effort_adapter.fetch_calls == [None]
        gap_logs = [
            record
            for record in caplog.records
            if getattr(record, "operation", None) == "gap_detection"
        ]
        assert len(gap_logs) == 1
        assert gap_logs[0].fallback == "built_in_gap_detection"
    finally:
        strict_cache.close()
        best_effort_cache.close()


def test_bulk_upsert_fallback_is_logged_and_double_failure_is_typed(monkeypatch, caplog):
    """The equivalent row fallback stays usable while double failure preserves both causes."""
    bulk_error = RuntimeError("bulk upsert failed")
    cache, _adapter = _make_cache({}, [1])

    try:
        monkeypatch.setattr(
            cache,
            "_upsert_records",
            lambda session, records: (_ for _ in ()).throw(bulk_error),
        )
        with caplog.at_level(logging.WARNING, logger="cacheness.sql_cache"):
            result = cache.get_data(query_name="upsert-contract")
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert any(
            getattr(record, "operation", None) == "upsert_fallback"
            for record in caplog.records
        )
    finally:
        cache.close()

    row_error = RuntimeError("row upsert failed")
    failing_cache, _adapter = _make_cache({}, [1])
    try:
        monkeypatch.setattr(
            failing_cache,
            "_upsert_records",
            lambda session, records: (_ for _ in ()).throw(bulk_error),
        )
        monkeypatch.setattr(
            failing_cache,
            "_fallback_upsert",
            lambda session, records: (_ for _ in ()).throw(row_error),
        )

        with pytest.raises(SqlCacheWriteError) as exc_info:
            failing_cache.get_data(query_name="upsert-contract")

        assert exc_info.value.__cause__ is row_error
        assert exc_info.value.bulk_error is bulk_error
        assert exc_info.value.row_error is row_error
        assert _cached_row_count(failing_cache) == 0
    finally:
        failing_cache.close()
