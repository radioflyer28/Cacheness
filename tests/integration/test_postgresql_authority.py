"""Real PostgreSQL coverage for the Phase 5 lifecycle authority boundary.

These tests intentionally use the externally provisioned qualification schema.
They prove database semantics only; payload effects stay outside the authority
transaction and are exercised by the remote-topology suite.
"""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import uuid4

import pytest

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobMigrationRequiredError,
)
from cacheness.storage.backends.postgresql_lifecycle_authority import (
    PostgresqlLifecycleAuthority,
)
from cacheness.storage.lifecycle_authority import (
    EntryExpectation,
    MutationSpec,
    VerificationProof,
)


pytestmark = pytest.mark.live_postgresql


def _authority_for_resources(
    resources: Any, *, store_identity: str | None = None, lock_timeout_ms: int = 100
) -> PostgresqlLifecycleAuthority:
    """Construct a new authority with its own connection factory and no shared lease."""
    import psycopg

    return PostgresqlLifecycleAuthority(
        lambda: psycopg.connect(resources.config.postgres_dsn),
        schema=resources.namespace.schema,
        store_identity=(
            store_identity
            if store_identity is not None
            else f"phase5-{resources.namespace.run_id[-32:]}"
        ),
        statement_timeout_ms=100,
        lock_timeout_ms=lock_timeout_ms,
    )


def _prepared_mutation(
    authority: PostgresqlLifecycleAuthority,
    *,
    key: str,
    expected: EntryExpectation,
    generation: str,
    locator: str,
) -> tuple[object, bytes]:
    """Persist one verified, synthetic canonical descriptor through public authority calls."""
    manifest = f'{{"key":"{key}","generation":"{generation}"}}'.encode("utf-8")
    prepared = authority.prepare_mutation(
        MutationSpec.create(
            operation_id=uuid4().hex,
            key=key,
            generation=generation,
            candidate_locator=locator,
            expected=expected,
            manifest=manifest,
        )
    )
    authority.record_verification(
        prepared,
        VerificationProof(
            digest=hashlib.sha256(manifest).hexdigest(),
            byte_size=len(manifest),
            manifest=manifest,
        ),
    )
    return prepared, manifest


def test_live_authority_requires_explicit_initialization_and_exact_store_identity(
    live_qualification_resources: Any,
) -> None:
    """Reopen validates current schema while a foreign identity fails without repair."""
    resources = live_qualification_resources
    reopened = _authority_for_resources(resources)
    foreign = _authority_for_resources(resources, store_identity="foreign-phase5-store")
    try:
        # The fixture already performed the only permitted initialization.  A
        # repeat is validation-only and does not create a compatibility path.
        reopened.initialize()
        assert reopened.open() is reopened
        with pytest.raises(CacheBlobMigrationRequiredError):
            foreign.open()
    finally:
        reopened.close()
        foreign.close()


def test_live_authority_uses_exact_cas_and_retains_cleanup_debt_atomically(
    live_qualification_resources: Any,
) -> None:
    """Independent connections observe one promoted lineage and durable exact debt."""
    resources = live_qualification_resources
    writer = _authority_for_resources(resources)
    reader = _authority_for_resources(resources)
    key = f"authority-{uuid4().hex}"
    try:
        first, _ = _prepared_mutation(
            writer,
            key=key,
            expected=EntryExpectation.absent(),
            generation="generation-one",
            locator=f"generations/{key}/one.native",
        )
        first_result = writer.promote_mutation(first)
        assert reader.read_entry(key) == first_result.entry

        stale, _ = _prepared_mutation(
            reader,
            key=key,
            expected=EntryExpectation.absent(),
            generation="generation-stale",
            locator=f"generations/{key}/stale.native",
        )
        replacement, _ = _prepared_mutation(
            writer,
            key=key,
            expected=first_result.entry.expectation,
            generation="generation-two",
            locator=f"generations/{key}/two.native",
        )
        replacement_result = writer.promote_mutation(replacement)

        with pytest.raises(CacheBlobLifecycleConflictError):
            reader.promote_mutation(stale)
        assert reader.read_entry(key) == replacement_result.entry
        # An idempotent replay belongs to its original mutation, not the
        # mutable current entry selected by the later replacement.
        replayed_first = writer.promote_mutation(first)
        assert replayed_first.entry == first_result.entry
        assert replayed_first.entry != replacement_result.entry
        assert len(replacement_result.cleanup_debt) == 1
        debt = replacement_result.cleanup_debt[0]
        assert debt.locator == first_result.entry.locator
        assert writer.pending_cleanup_debts(key=key) == (debt,)

        # Candidate cleanup after an interruption is durable authority work,
        # not an attempted payload rollback inside this transaction.
        abandoned, _ = _prepared_mutation(
            writer,
            key=f"abandoned-{uuid4().hex}",
            expected=EntryExpectation.absent(),
            generation="candidate-generation",
            locator=f"generations/{key}/candidate.native",
        )
        writer.abort_mutation(abandoned, candidate_persisted=True)
        assert writer.pending_cleanup_debts(operation_id=abandoned.operation_id)
    finally:
        writer.close()
        reader.close()


def test_live_authority_maps_real_cross_connection_lock_contention_and_pages_work(
    live_qualification_resources: Any,
) -> None:
    """A real row lock becomes a typed progress result without promising success."""
    import psycopg
    from psycopg import sql

    resources = live_qualification_resources
    seed = _authority_for_resources(resources)
    contender = _authority_for_resources(resources, lock_timeout_ms=25)
    key = f"contention-{uuid4().hex}"
    blocking_connection = None
    try:
        initial, _ = _prepared_mutation(
            seed,
            key=key,
            expected=EntryExpectation.absent(),
            generation="generation-one",
            locator=f"generations/{key}/one.native",
        )
        current = seed.promote_mutation(initial).entry

        # This is database test setup, not a production advisory-lock protocol.
        blocking_connection = psycopg.connect(resources.config.postgres_dsn)
        with blocking_connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT key FROM {}.{} WHERE key = %s FOR UPDATE").format(
                    sql.Identifier(resources.namespace.schema), sql.Identifier("entries")
                ),
                (key,),
            )
            with pytest.raises(CacheBlobLifecycleTimeoutError) as blocked:
                contender.delete_entry(key, expected=current.expectation)
        assert blocked.value.__cause__ is not None
        assert contender.read_entry(key) == current

        # The portable reconciliation page remains explicitly bounded even
        # after contention; it exposes authority residue only.
        snapshot = seed.reconciliation_snapshot()
        page = seed.page_reconciliation_work(
            snapshot, mutation_cursor=0, debt_cursor=0
        )
        assert len(page.works) <= seed.lifecycle_limits.operation_page_size
        assert page.mutation_cursor <= snapshot.mutation_high_water
        assert page.debt_cursor <= snapshot.debt_high_water
        assert PostgresqlLifecycleAuthority.progress_outcome_for_sqlstate("55P03") == (
            "retryable_lock_timeout"
        )
    finally:
        if blocking_connection is not None:
            blocking_connection.rollback()
            blocking_connection.close()
        seed.close()
        contender.close()
