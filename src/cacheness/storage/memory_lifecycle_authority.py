"""Deterministic same-process LifecycleAuthority reference implementation."""

from __future__ import annotations

import hashlib
from threading import RLock
from uuid import uuid4

from cacheness.config import LifecycleLimits
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
    ReconciliationPage,
    ReconciliationSnapshot,
    ReconciliationWork,
    VerificationProof,
)


class InMemoryLifecycleAuthority:
    """Copy-on-read authority suitable only for one process and test stores."""

    capabilities = AuthorityCapabilities(
        durable=False,
        multiprocess=False,
        exact_cas=True,
        indexed_paging=True,
        projection=False,
    )

    def __init__(self, *, lifecycle_limits: LifecycleLimits | None = None) -> None:
        self._lock = RLock()
        self.lifecycle_limits = (
            LifecycleLimits() if lifecycle_limits is None else lifecycle_limits
        )
        if not isinstance(self.lifecycle_limits, LifecycleLimits):
            raise TypeError("lifecycle_limits must be a LifecycleLimits instance")
        self._entries: dict[str, EntrySnapshot] = {}
        self._lineages: dict[str, int] = {}
        self._mutations: dict[str, tuple[MutationSpec, VerificationProof | None, str]] = {}
        self._mutation_order: list[str] = []
        self._debts: list[CleanupDebt] = []
        self._clear_targets: dict[str, dict[str, tuple[EntrySnapshot, str]]] = {}
        self._clear_cursors: dict[str, str] = {}
        self._clear_states: dict[str, str] = {}
        self._reconciliation_states: dict[str, str] = {}
        self._reconciliation_snapshots: dict[str, ReconciliationSnapshot] = {}
        self._revision = 0
        self._projection_dirty = False
        self._closed = False
        self.open_write_transactions = 0

    def _require_open(self) -> None:
        if self._closed:
            raise CacheBlobStoreClosedError("Lifecycle authority is closed")

    def preflight_mutation(self) -> None:
        """Verify this ephemeral authority is open without allocating state."""
        self._require_open()

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

    def read_expectation(self, key: str) -> EntryExpectation:
        """Return exact present or absent lineage for one later mutation CAS."""
        self._require_open()
        with self._lock:
            return self._expectation(key)

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
            self._mutation_order.append(spec.operation_id)
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

    def abort_mutation(
        self, prepared: PreparedMutation, *, candidate_persisted: bool = False
    ) -> None:
        def abort() -> None:
            mutation = self._mutations.get(prepared.operation_id)
            if mutation is None or mutation[0] != prepared.spec:
                return
            if mutation[2] == "promoted":
                return
            if candidate_persisted:
                debt = CleanupDebt(
                    prepared.operation_id,
                    prepared.spec.candidate_locator,
                    prepared.spec.key,
                    prepared.spec.generation,
                    "candidate",
                )
                if debt not in self._debts:
                    self._debts.append(debt)
                self._mutations[prepared.operation_id] = (
                    mutation[0],
                    mutation[1],
                    "aborted",
                )
                return
            del self._mutations[prepared.operation_id]

        self._transition(abort)

    def list_entries(self) -> tuple[EntrySnapshot, ...]:
        """Return immutable committed/tombstone entry snapshots by key."""
        self._require_open()
        with self._lock:
            return tuple(self._copy(entry) for _, entry in sorted(self._entries.items()))

    def pending_cleanup_debts(
        self,
        *,
        key: str | None = None,
        operation_id: str | None = None,
    ) -> tuple[CleanupDebt, ...]:
        """Return only exact still-pending cleanup work from authority state."""
        self._require_open()
        with self._lock:
            return tuple(
                debt
                for debt in self._debts
                if (key is None or debt.key == key)
                and (operation_id is None or debt.operation_id == operation_id)
            )

    def pending_mutations(self) -> tuple[PreparedMutation, ...]:
        """Return indexed, pre-promotion operations for exact recovery only."""
        self._require_open()
        with self._lock:
            return tuple(
                PreparedMutation(operation_id, spec)
                for operation_id, (spec, _proof, state) in sorted(self._mutations.items())
                if state == "prepared"
            )

    def retire_cleanup_debt(self, debt: CleanupDebt) -> None:
        """Retire one exact debt idempotently after external reclamation."""
        def retire() -> None:
            try:
                self._debts.remove(debt)
            except ValueError:
                return

        self._transition(retire)

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
        def begin() -> PageToken:
            active = next(
                (
                    run_id
                    for run_id, state in self._clear_states.items()
                    if state == "active"
                ),
                None,
            )
            if active is not None:
                return PageToken(active)
            token = PageToken(uuid4().hex)
            self._clear_targets[token.value] = {
                key: (self._copy(entry), "pending")
                for key, entry in self._entries.items()
            }
            self._clear_cursors[token.value] = ""
            self._clear_states[token.value] = "active"
            return token

        return self._transition(begin)

    def page_clear(self, token: PageToken) -> tuple[EntrySnapshot, ...]:
        self._require_open()
        with self._lock:
            if token.value not in self._clear_states:
                raise CacheBlobLifecycleConflictError("Clear run does not exist")
            cursor = self._clear_cursors[token.value]
            targets = self._clear_targets[token.value]
            return tuple(
                self._copy(entry)
                for key, (entry, state) in sorted(targets.items())
                if state == "pending" and key > cursor
            )[: self.lifecycle_limits.manifest_page_size]

    def checkpoint_clear(
        self,
        token: PageToken,
        target: EntrySnapshot | None = None,
        *,
        state: str = "completed",
    ) -> None:
        def checkpoint() -> None:
            current_state = self._clear_states.get(token.value)
            if target is None and current_state == "completed":
                return
            if current_state != "active":
                raise CacheBlobLifecycleConflictError("Clear run cannot accept checkpoint")
            if target is None:
                self._clear_states[token.value] = "completed"
                return
            if state not in {"completed", "conflicted", "blocked"}:
                raise ValueError("Clear target state is unsupported")
            stored = self._clear_targets[token.value].get(target.key)
            if stored is None or stored[0] != target or stored[1] != "pending":
                raise CacheBlobLifecycleConflictError("Clear target is no longer pending")
            current = self._entries.get(target.key)
            if state == "completed":
                lineage = self._lineages.get(target.key, 0)
                if current is not None or lineage <= (target.expectation.lineage or 0):
                    raise CacheBlobLifecycleConflictError(
                        "Clear target completion lacks exact absence proof"
                    )
            elif (
                state == "conflicted"
                and current is not None
                and current.expectation == target.expectation
            ):
                raise CacheBlobLifecycleConflictError("Clear target has not changed")
            self._clear_targets[token.value][target.key] = (stored[0], state)
            self._clear_cursors[token.value] = target.key
            if not any(
                target_state == "pending"
                for _, target_state in self._clear_targets[token.value].values()
            ):
                self._clear_states[token.value] = "completed"

        self._transition(checkpoint)

    def begin_reconciliation(self) -> PageToken:
        def begin() -> PageToken:
            active = next(
                (
                    run_id
                    for run_id, state in self._reconciliation_states.items()
                    if state == "active"
                ),
                None,
            )
            if active is not None:
                return PageToken(active)
            token = PageToken(uuid4().hex)
            self._reconciliation_states[token.value] = "active"
            self._reconciliation_snapshots[token.value] = ReconciliationSnapshot(
                self._revision,
                len(self._mutation_order),
                len(self._debts),
                token.value,
            )
            return token

        return self._transition(begin)

    def reconciliation_snapshot(
        self, token: PageToken | None = None
    ) -> ReconciliationSnapshot:
        self._require_open()
        with self._lock:
            if token is not None:
                snapshot = self._reconciliation_snapshots.get(token.value)
                if snapshot is None:
                    raise CacheBlobLifecycleConflictError("Reconciliation run does not exist")
                return snapshot
            return ReconciliationSnapshot(
                self._revision,
                len(self._mutation_order),
                len(self._debts),
            )

    def page_reconciliation_work(
        self,
        snapshot: ReconciliationSnapshot,
        *,
        mutation_cursor: int,
        debt_cursor: int,
    ) -> ReconciliationPage:
        self._require_open()
        with self._lock:
            page_size = max(1, self.lifecycle_limits.operation_page_size // 2)
            mutation_stop = min(snapshot.mutation_high_water, mutation_cursor + page_size)
            debt_stop = min(snapshot.debt_high_water, debt_cursor + page_size)
            works: list[ReconciliationWork] = []
            for row_id in range(mutation_cursor + 1, mutation_stop + 1):
                operation_id = self._mutation_order[row_id - 1]
                spec, _proof, state = self._mutations[operation_id]
                if state == "prepared":
                    works.append(
                        ReconciliationWork(
                            "mutation",
                            row_id,
                            state,
                            mutation=PreparedMutation(operation_id, spec),
                        )
                    )
            for row_id in range(debt_cursor + 1, debt_stop + 1):
                debt = self._debts[row_id - 1]
                works.append(
                    ReconciliationWork("debt", row_id, "pending", debt=debt)
                )
            return ReconciliationPage(tuple(works), mutation_stop, debt_stop)

    def page_reconciliation(self, token: PageToken) -> tuple[CleanupDebt, ...]:
        self._require_open()
        with self._lock:
            if token.value not in self._reconciliation_states:
                raise CacheBlobLifecycleConflictError("Reconciliation run does not exist")
            return tuple(self._debts)

    def checkpoint_reconciliation(
        self,
        token: PageToken,
        work: ReconciliationWork | None = None,
        *,
        state: str = "completed",
    ) -> None:
        def checkpoint() -> None:
            if self._reconciliation_states.get(token.value) != "active":
                raise CacheBlobLifecycleConflictError(
                    "Reconciliation run cannot accept checkpoint"
                )
            if work is None:
                self._reconciliation_states[token.value] = "completed"
            elif state not in {"completed", "blocked", "conflicted"}:
                raise ValueError("Reconciliation checkpoint state is unsupported")

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
