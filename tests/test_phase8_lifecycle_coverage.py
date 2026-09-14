"""Phase 8 contracts for lifecycle integrity, recovery, and PostgreSQL boundaries."""

from __future__ import annotations

import pytest

from cacheness.error_handling import CacheBlobLifecycleTimeoutError


def test_postgresql_error_classification_preserves_typed_progress_outcomes() -> None:
    """Declared PostgreSQL SQLSTATEs remain retryable progress, not corruption."""

    authority = _phase8_postgresql_authority(sqlstate="40001")

    with pytest.raises(CacheBlobLifecycleTimeoutError) as raised:
        authority.inventory_page(limit=1, work_cap=1024)

    assert raised.value.context["progress_outcome"] == "serialization"
