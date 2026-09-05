"""Deterministic same-process LifecycleAuthority reference implementation."""

from __future__ import annotations

from threading import RLock
from uuid import uuid4

from cacheness.error_handling import CacheBlobLifecycleConflictError

from .lifecycle_authority import (
    AuthorityCapabilities,
    CleanupDebt,
    EntryExpectation,
    EntrySnapshot,
    MutationSpec,
    PageToken,
    PreparedMutation,
    ProjectionRevision,
    PromotionResult,
    VerificationProof,
)


class InMemoryLifecycleAuthority:
    """Copy-on-read authority suitable only for one process and test stores."""

    capabilities = AuthorityCapabilities(durable=False, multiprocess=False)

    def __init__(self) -> None:
        self._lock = RLock()
        self._entries: dict[str, EntrySnapshot] = {}
        self._lineages: dict[str, int] = {}
        self._mutations: dict[str, tuple[MutationSpec, VerificationProof | None]] = {}
        self._closed = False
        self.open_write_transactions = 0

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Lifecycle authority is closed")

    def _transition(self, callback):
        self._require_open()
        with self._lock:
            self.open_write_transactions += 1
            try:
                return callback()
            finally:
                self.open_write_transactions -= 1

    def _expectation(self, key: str) -> EntryExpectation:
        entry = self._entries.get(key)
        if entry is not None:
            return entry.expectation
        lineage = self._lineages.get(key)
        return EntryExpectation(lineage=lineage, revision=None)

    @staticmethod
    def _copy(entry: EntrySnapshot) -> EntrySnapshot:
        return EntrySnapshot(
            key=entry.key,
            generation=entry.generation,
            locator=entry.locator,
            manifest=bytes(entry.manifest),
            expectation=EntryExpectation(entry.expectation.lineage, entry.expectation.revision),
        )

    def read_entry(self, key: str) -> EntrySnapshot | None:
        self._require_open()
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else self._copy(entry)

    def prepare_mutation(self, spec: MutationSpec) -> PreparedMutation:
        def prepare() -> PreparedMutation:
            existing = self._mutations.get(spec.operation_id)
            if existing is not None:
                if existing[0] == spec:
                    return PreparedMutation(spec.operation_id, spec)
                raise CacheBlobLifecycleConflictError("Operation identifier is not reusable")
            if self._expectation(spec.key) != spec.expected:
                raise CacheBlobLifecycleConflictError("Mutation expectation no longer matches authority")
            self._mutations[spec.operation_id] = (spec, None)
            return PreparedMutation(spec.operation_id, spec)
        return self._transition(prepare)

    def record_verification(self, prepared: PreparedMutation, proof: VerificationProof) -> None:
        def record() -> None:
            mutation = self._mutations.get(prepared.operation_id)
            if mutation is None or mutation[0] != prepared.spec:
                raise CacheBlobLifecycleConflictError("Prepared mutation cannot accept verification")
            self._mutations[prepared.operation_id] = (mutation[0], proof)
        self._transition(record)

    def record_verification_for_test(self, prepared: PreparedMutation) -> None:
        self.record_verification(prepared, VerificationProof("0" * 64, 0))

    def promote_mutation(self, prepared: PreparedMutation) -> PromotionResult:
        def promote() -> PromotionResult:
            mutation = self._mutations.get(prepared.operation_id)
            if mutation is None or mutation[1] is None:
                raise CacheBlobLifecycleConflictError("Mutation is not verified and prepared")
            spec, proof = mutation
            if self._expectation(spec.key) != spec.expected:
                raise CacheBlobLifecycleConflictError("Mutation lineage changed before promotion")
            lineage = self._lineages.get(spec.key, 0) + 1
            self._lineages[spec.key] = lineage
            entry = EntrySnapshot(
                key=spec.key,
                generation=spec.generation,
                locator=spec.candidate_locator,
                manifest=proof.manifest or spec.manifest,
                expectation=EntryExpectation(lineage=lineage, revision=lineage),
            )
            self._entries[spec.key] = entry
            self._mutations.pop(prepared.operation_id)
            return PromotionResult(self._copy(entry))
        return self._transition(promote)

    def abort_mutation(self, prepared: PreparedMutation) -> None:
        self._transition(lambda: self._mutations.pop(prepared.operation_id, None))

    def delete_entry(self, key: str, *, expected: EntryExpectation) -> None:
        def delete() -> None:
            if self._expectation(key) != expected:
                raise CacheBlobLifecycleConflictError("Delete expectation no longer matches authority")
            self._lineages[key] = self._lineages.get(key, 0) + 1
            self._entries.pop(key, None)
        self._transition(delete)

    def retire_tombstone(self, key: str, *, expected: EntryExpectation) -> None:
        self.delete_entry(key, expected=expected)

    def begin_clear(self) -> PageToken: return PageToken(uuid4().hex)
    def page_clear(self, token: PageToken) -> tuple[EntrySnapshot, ...]: return ()
    def checkpoint_clear(self, token: PageToken) -> None: return None
    def begin_reconciliation(self) -> PageToken: return PageToken(uuid4().hex)
    def page_reconciliation(self, token: PageToken) -> tuple[CleanupDebt, ...]: return ()
    def checkpoint_reconciliation(self, token: PageToken) -> None: return None
    def compare_and_mark_projection(self, expected: ProjectionRevision | None) -> ProjectionRevision: return ProjectionRevision(0)

    def close(self) -> None:
        self._closed = True


__all__ = ["InMemoryLifecycleAuthority"]
