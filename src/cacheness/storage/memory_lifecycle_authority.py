"""Deterministic same-process LifecycleAuthority reference implementation."""

from __future__ import annotations

import hashlib
from threading import RLock
from uuid import uuid4

from cacheness.error_handling import CacheBlobLifecycleConflictError, CacheBlobStoreClosedError

from .lifecycle_authority import (
    AuthorityCapabilities,
    AuthorityStateSnapshot,
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
        self._mutations: dict[str, tuple[MutationSpec, VerificationProof | None, str]] = {}
        self._debts: list[CleanupDebt] = []
        self._clear_targets: dict[str, tuple[EntrySnapshot, ...]] = {}
        self._clear_states: dict[str, str] = {}
        self._reconciliation_states: dict[str, str] = {}
        self._revision = 0
        self._projection_dirty = False
        self._closed = False
        self.open_write_transactions = 0

    def _require_open(self) -> None:
        if self._closed:
            raise CacheBlobStoreClosedError("Lifecycle authority is closed")

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
            expectation=EntryExpectation(
                entry.expectation.lineage,
                entry.expectation.revision,
                entry.expectation.generation,
                entry.expectation.manifest_digest,
            ),
        )

    def read_entry(self, key: str) -> EntrySnapshot | None:
        self._require_open()
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else self._copy(entry)

    def snapshot_state(self) -> AuthorityStateSnapshot:
        self._require_open()
        with self._lock:
            return AuthorityStateSnapshot(
                self._revision,
                self._projection_dirty,
                tuple((operation_id, row[2]) for operation_id, row in sorted(self._mutations.items())),
                tuple(self._debts),
            )

    def prepare_mutation(self, spec: MutationSpec) -> PreparedMutation:
        def prepare() -> PreparedMutation:
            existing = self._mutations.get(spec.operation_id)
            if existing is not None:
                if existing[0] == spec:
                    return PreparedMutation(spec.operation_id, spec)
                raise CacheBlobLifecycleConflictError("Operation identifier is not reusable")
            if self._expectation(spec.key) != spec.expected:
                raise CacheBlobLifecycleConflictError("Mutation expectation no longer matches authority")
            self._mutations[spec.operation_id] = (spec, None, "prepared")
            return PreparedMutation(spec.operation_id, spec)

        return self._transition(prepare)

    def record_verification(self, prepared: PreparedMutation, proof: VerificationProof) -> None:
        def record() -> None:
            mutation = self._mutations.get(prepared.operation_id)
            if mutation is None or mutation[0] != prepared.spec:
                raise CacheBlobLifecycleConflictError("Prepared mutation cannot accept verification")
            if mutation[2] == "promoted":
                return
            self._mutations[prepared.operation_id] = (mutation[0], proof, "prepared")

        self._transition(record)

    def record_verification_for_test(self, prepared: PreparedMutation) -> None:
        self.record_verification(prepared, VerificationProof("0" * 64, 0))

    def _promoted_result(self, operation_id: str) -> PromotionResult:
        mutation = self._mutations.get(operation_id)
        if mutation is None:
            raise CacheBlobLifecycleConflictError("Mutation does not exist")
        entry = self._entries.get(mutation[0].key)
        if entry is None:
            raise CacheBlobLifecycleConflictError("Promoted mutation has no committed entry")
        debts = tuple(debt for debt in self._debts if debt.operation_id == operation_id)
        return PromotionResult(self._copy(entry), debts)

    def promote_mutation(self, prepared: PreparedMutation) -> PromotionResult:
        def promote() -> PromotionResult:
            mutation = self._mutations.get(prepared.operation_id)
            if mutation is None or mutation[0] != prepared.spec:
                raise CacheBlobLifecycleConflictError("Mutation does not exist")
            spec, proof, state = mutation
            if state == "promoted":
                return self._promoted_result(prepared.operation_id)
            if proof is None:
                raise CacheBlobLifecycleConflictError("Mutation is not verified and prepared")
            if self._expectation(spec.key) != spec.expected:
                raise CacheBlobLifecycleConflictError("Mutation lineage changed before promotion")
            previous = self._entries.get(spec.key)
            lineage = self._lineages.get(spec.key, 0) + 1
            self._lineages[spec.key] = lineage
            self._revision += 1
            manifest = proof.manifest or spec.manifest
            entry = EntrySnapshot(
                key=spec.key,
                generation=spec.generation,
                locator=spec.candidate_locator,
                manifest=manifest,
                expectation=EntryExpectation(
                    lineage=lineage,
                    revision=self._revision,
                    generation=spec.generation,
                    manifest_digest=hashlib.sha256(manifest).hexdigest(),
                ),
            )
            self._entries[spec.key] = entry
            self._mutations[prepared.operation_id] = (spec, proof, "promoted")
            if previous is not None and previous.locator != spec.candidate_locator:
                self._debts.append(
                    CleanupDebt(
                        prepared.operation_id,
                        previous.locator,
                        previous.key,
                        previous.generation,
                    )
                )
            self._projection_dirty = True
            return self._promoted_result(prepared.operation_id)

        return self._transition(promote)

    def abort_mutation(self, prepared: PreparedMutation) -> None:
        def abort() -> None:
            mutation = self._mutations.get(prepared.operation_id)
            if mutation is not None and mutation[0] == prepared.spec and mutation[2] != "promoted":
                del self._mutations[prepared.operation_id]

        self._transition(abort)

    def delete_entry(self, key: str, *, expected: EntryExpectation) -> None:
        def delete() -> None:
            if self._expectation(key) != expected:
                raise CacheBlobLifecycleConflictError("Delete expectation no longer matches authority")
            self._lineages[key] = self._lineages.get(key, 0) + 1
            self._entries.pop(key, None)
            self._revision += 1
            self._projection_dirty = True

        self._transition(delete)

    def retire_tombstone(self, key: str, *, expected: EntryExpectation) -> None:
        self.delete_entry(key, expected=expected)

    def begin_clear(self) -> PageToken:
        token = PageToken(uuid4().hex)

        def begin() -> PageToken:
            self._clear_targets[token.value] = tuple(
                self._copy(entry) for _, entry in sorted(self._entries.items())
            )
            self._clear_states[token.value] = "active"
            return token

        return self._transition(begin)

    def page_clear(self, token: PageToken) -> tuple[EntrySnapshot, ...]:
        self._require_open()
        with self._lock:
            return self._clear_targets.get(token.value, ())

    def checkpoint_clear(self, token: PageToken) -> None:
        def checkpoint() -> None:
            if self._clear_states.get(token.value) != "active":
                raise CacheBlobLifecycleConflictError("Clear run cannot accept checkpoint")
            self._clear_states[token.value] = "checkpointed"

        self._transition(checkpoint)

    def begin_reconciliation(self) -> PageToken:
        token = PageToken(uuid4().hex)

        def begin() -> PageToken:
            self._reconciliation_states[token.value] = "active"
            return token

        return self._transition(begin)

    def page_reconciliation(self, token: PageToken) -> tuple[CleanupDebt, ...]:
        self._require_open()
        with self._lock:
            if token.value not in self._reconciliation_states:
                raise CacheBlobLifecycleConflictError("Reconciliation run does not exist")
            return tuple(self._debts)

    def checkpoint_reconciliation(self, token: PageToken) -> None:
        def checkpoint() -> None:
            if self._reconciliation_states.get(token.value) != "active":
                raise CacheBlobLifecycleConflictError(
                    "Reconciliation run cannot accept checkpoint"
                )
            self._reconciliation_states[token.value] = "checkpointed"

        self._transition(checkpoint)

    def compare_and_mark_projection(
        self, expected: ProjectionRevision | None
    ) -> ProjectionRevision:
        def mark() -> ProjectionRevision:
            if expected is not None and expected.value != self._revision:
                raise CacheBlobLifecycleConflictError("Projection revision changed")
            self._projection_dirty = False
            return ProjectionRevision(self._revision)

        return self._transition(mark)

    def close(self) -> None:
        with self._lock:
            self._closed = True


__all__ = ["InMemoryLifecycleAuthority"]
