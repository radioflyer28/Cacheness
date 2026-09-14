"""Deterministic same-process LifecycleAuthority reference implementation."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from threading import RLock
from typing import Any, Callable, Iterator
from uuid import uuid4

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationOfflineDecisionRequiredError,
    CacheBlobStoreClosedError,
    CacheManifestIntegrityError,
)

from .lifecycle_authority import (
    AuthorityCapabilities,
    AuthorityStateSnapshot,
    CleanupDebt,
    EntryExpectation,
    EntrySnapshot,
    MutationReplay,
    MutationSpec,
    PageToken,
    PreparedMutation,
    ProjectionBackup,
    ProjectionRevision,
    PromotionResult,
    ReconciliationPage,
    ReconciliationSnapshot,
    ReconciliationWork,
    VerificationProof,
)
from .migration_authority import (
    ActivationReceipt,
    AuthorityInventoryCursor,
    AuthorityIdentitySnapshot,
    AuthorityInventoryEntry,
    AuthorityInventoryPage,
    AuthorityPublicationState,
    FinalizeReceipt,
    PriorStoreReceipt,
    RollbackReceipt,
    VerifiedCandidateReceipt,
    candidate_digest,
    validate_inventory_page_request,
)
from .manifest import BlobManifest
from .catalog import (
    CatalogCursor,
    CatalogPage,
    CatalogQuery,
    CatalogSchema,
    page_from_canonical_scan,
    validate_catalog_page_request,
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
    topology_capabilities = {
        "durable": False,
        "process_scope": "process",
        "host_scope": "process",
        "transaction_scope": "authority",
        "exact_cas": True,
        "portable_query": True,
        "canonical_scan": True,
        "index_acceleration": False,
    }

    def __init__(self, *, lifecycle_limits: LifecycleLimits | None = None) -> None:
        self._lock = RLock()
        self.lifecycle_limits = (
            LifecycleLimits() if lifecycle_limits is None else lifecycle_limits
        )
        if not isinstance(self.lifecycle_limits, LifecycleLimits):
            raise TypeError("lifecycle_limits must be a LifecycleLimits instance")
        self._entries: dict[str, EntrySnapshot] = {}
        self._lineages: dict[str, int] = {}
        self._mutations: dict[
            str,
            tuple[MutationSpec, VerificationProof | None, str, EntrySnapshot | None],
        ] = {}
        self._mutation_order: dict[str, int] = {}
        self._mutation_high_water = 0
        self._debts: dict[int, CleanupDebt] = {}
        self._debt_high_water = 0
        self._clear_targets: dict[str, dict[str, tuple[EntrySnapshot, str]]] = {}
        self._clear_cursors: dict[str, str] = {}
        self._clear_states: dict[str, str] = {}
        self._reconciliation_states: dict[str, str] = {}
        self._reconciliation_snapshots: dict[str, ReconciliationSnapshot] = {}
        self._revision = 0
        self._catalog_store_id = uuid4().hex
        self._projection_dirty = False
        self._migration_state = AuthorityPublicationState.IDLE
        self._migration_receipt: VerifiedCandidateReceipt | None = None
        self._migration_candidates: tuple[AuthorityInventoryEntry, ...] = ()
        self._migration_prior_entries: dict[str, EntrySnapshot] = {}
        self._migration_prior_revision = 0
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
            transport_evidence=(
                None
                if entry.transport_evidence is None
                else bytes(entry.transport_evidence)
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
                tuple(
                    (operation_id, row[2])
                    for operation_id, row in sorted(self._mutations.items())
                ),
                tuple(self._debts.values()),
            )

    def prepare_mutation(self, spec: MutationSpec) -> PreparedMutation:
        def prepare() -> PreparedMutation:
            existing = self._mutations.get(spec.operation_id)
            if existing is not None:
                if existing[0] == spec:
                    return PreparedMutation(spec.operation_id, spec)
                raise CacheBlobLifecycleConflictError(
                    "Operation identifier is not reusable"
                )
            if self._expectation(spec.key) != spec.expected:
                raise CacheBlobLifecycleConflictError(
                    "Mutation expectation no longer matches authority"
                )
            self._mutations[spec.operation_id] = (spec, None, "prepared", None)
            self._mutation_high_water += 1
            self._mutation_order[spec.operation_id] = self._mutation_high_water
            return PreparedMutation(spec.operation_id, spec)

        return self._transition(prepare)

    def read_mutation(self, operation_id: str) -> MutationReplay | None:
        """Read one exact operation record without changing authority state."""
        MutationSpec.validate_operation_id(operation_id)
        self._require_open()
        with self._lock:
            mutation = self._mutations.get(operation_id)
            if mutation is None:
                return None
            spec, proof, state, _promoted_entry = mutation
            prepared = PreparedMutation(operation_id, spec)
            if state == "prepared":
                return MutationReplay(prepared, state, proof)
            if state == "promoted":
                return MutationReplay(
                    prepared,
                    state,
                    proof,
                    self._promoted_result(operation_id),
                )
            raise CacheBlobLifecycleConflictError(
                "Operation record is not replayable",
                context={"operation_id": operation_id, "state": state},
            )

    def record_verification(
        self, prepared: PreparedMutation, proof: VerificationProof
    ) -> None:
        def record() -> None:
            mutation = self._mutations.get(prepared.operation_id)
            if mutation is None or mutation[0] != prepared.spec:
                raise CacheBlobLifecycleConflictError(
                    "Prepared mutation cannot accept verification"
                )
            if mutation[2] == "promoted":
                return
            if mutation[1] is not None:
                if mutation[1] != proof:
                    raise CacheBlobLifecycleConflictError(
                        "Prepared mutation verification cannot be replaced"
                    )
                return
            self._mutations[prepared.operation_id] = (
                mutation[0],
                proof,
                "prepared",
                mutation[3],
            )

        self._transition(record)

    def record_verification_for_test(self, prepared: PreparedMutation) -> None:
        self.record_verification(prepared, VerificationProof("0" * 64, 0))

    def _promoted_result(self, operation_id: str) -> PromotionResult:
        mutation = self._mutations.get(operation_id)
        if mutation is None:
            raise CacheBlobLifecycleConflictError("Mutation does not exist")
        entry = mutation[3]
        if entry is None:
            raise CacheBlobLifecycleConflictError(
                "Promoted mutation has no committed entry"
            )
        debts = tuple(
            debt for debt in self._debts.values() if debt.operation_id == operation_id
        )
        return PromotionResult(self._copy(entry), debts)

    def promote_mutation(self, prepared: PreparedMutation) -> PromotionResult:
        def promote() -> PromotionResult:
            mutation = self._mutations.get(prepared.operation_id)
            if mutation is None or mutation[0] != prepared.spec:
                raise CacheBlobLifecycleConflictError("Mutation does not exist")
            spec, proof, state, _promoted_entry = mutation
            if state == "promoted":
                return self._promoted_result(prepared.operation_id)
            if proof is None:
                raise CacheBlobLifecycleConflictError(
                    "Mutation is not verified and prepared"
                )
            if spec.manifest and proof.manifest and spec.manifest != proof.manifest:
                raise CacheBlobLifecycleConflictError(
                    "Verification descriptor differs from prepared descriptor"
                )
            if self._expectation(spec.key) != spec.expected:
                raise CacheBlobLifecycleConflictError(
                    "Mutation lineage changed before promotion"
                )
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
                transport_evidence=proof.transport_evidence,
            )
            self._entries[spec.key] = entry
            self._mutations[prepared.operation_id] = (
                spec,
                proof,
                "promoted",
                self._copy(entry),
            )
            if previous is not None and previous.locator != spec.candidate_locator:
                self._add_debt(
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

    def replace_committed_metadata(
        self,
        entry: EntrySnapshot,
        *,
        expected: EntryExpectation,
        manifest: bytes,
    ) -> EntrySnapshot:
        """Atomically replace only signed mutable descriptor fields for one entry."""

        def replace_metadata() -> EntrySnapshot:
            stored = self._entries.get(entry.key)
            if (
                stored is None
                or stored != entry
                or stored.expectation != expected
                or self._expectation(entry.key) != expected
            ):
                raise CacheBlobLifecycleConflictError(
                    "Metadata replacement expectation no longer matches authority"
                )
            try:
                current_manifest = BlobManifest.from_canonical_bytes(stored.manifest)
                replacement_manifest = BlobManifest.from_canonical_bytes(manifest)
            except (CacheManifestIntegrityError, TypeError, ValueError) as error:
                raise CacheBlobLifecycleConflictError(
                    "Metadata replacement descriptor is malformed"
                ) from error
            if (
                replacement_manifest.key != current_manifest.key
                or replacement_manifest.generation != current_manifest.generation
                or replacement_manifest.locator != current_manifest.locator
                or replacement_manifest.digest != current_manifest.digest
                or replacement_manifest.byte_size != current_manifest.byte_size
                or replacement_manifest.state != current_manifest.state
            ):
                raise CacheBlobLifecycleConflictError(
                    "Metadata replacement changed immutable payload identity"
                )
            current_record = current_manifest.to_mapping()
            replacement_record = replacement_manifest.to_mapping()
            for field_name in (
                "catalog_values",
                "catalog_presence",
                "user_metadata",
                "signature",
            ):
                current_record.pop(field_name, None)
                replacement_record.pop(field_name, None)
            if current_record != replacement_record:
                raise CacheBlobLifecycleConflictError(
                    "Metadata replacement changed immutable descriptor fields"
                )
            self._revision += 1
            replacement = EntrySnapshot(
                key=stored.key,
                generation=stored.generation,
                locator=stored.locator,
                manifest=bytes(manifest),
                expectation=EntryExpectation(
                    lineage=stored.expectation.lineage,
                    revision=self._revision,
                    generation=stored.generation,
                    manifest_digest=hashlib.sha256(manifest).hexdigest(),
                ),
                transport_evidence=(
                    None
                    if stored.transport_evidence is None
                    else bytes(stored.transport_evidence)
                ),
            )
            self._entries[stored.key] = replacement
            self._projection_dirty = True
            return self._copy(replacement)

        return self._transition(replace_metadata)

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
                if debt not in self._debts.values():
                    self._add_debt(debt)
                self._mutations[prepared.operation_id] = (
                    mutation[0],
                    mutation[1],
                    "aborted",
                    mutation[3],
                )
                return
            del self._mutations[prepared.operation_id]
            self._mutation_order.pop(prepared.operation_id, None)

        self._transition(abort)

    def list_entries(self) -> tuple[EntrySnapshot, ...]:
        """Return immutable committed/tombstone entry snapshots by key."""
        self._require_open()
        with self._lock:
            return tuple(
                self._copy(entry) for _, entry in sorted(self._entries.items())
            )

    # The methods below are intentionally not part of LifecycleAuthority.
    # They are an explicit, read-only maintenance seam used only by the
    # offline migration service. Normal reads/writes cannot reach them.
    def identity_snapshot(self) -> AuthorityIdentitySnapshot:
        """Return the current memory authority identity and exact revision."""
        self._require_open()
        with self._lock:
            return AuthorityIdentitySnapshot(
                store_id=self._catalog_store_id,
                revision=self._revision,
                authority_kind="memory",
                capability="memory-lifecycle-authority-v1",
                schema_version=0,
            )

    def inventory_page(
        self,
        cursor: AuthorityInventoryCursor | None = None,
        *,
        limit: int | None = None,
        work_cap: int | None = None,
    ) -> AuthorityInventoryPage:
        """Return one raw keyset page at one exact same-process revision."""
        effective_limit, effective_work_cap = validate_inventory_page_request(
            limit=limit,
            work_cap=work_cap,
            default_limit=self.lifecycle_limits.manifest_page_size,
            default_work_cap=self.lifecycle_limits.max_operation_record_bytes,
        )
        self._require_open()
        with self._lock:
            identity = self.identity_snapshot()
            if cursor is not None and (
                cursor.store_id != identity.store_id
                or cursor.revision != identity.revision
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration inventory changed; reinspection is required"
                )
            start_after = (
                (cursor.last_key, cursor.last_generation)
                if cursor is not None and cursor.last_key is not None
                else None
            )
            remaining = sorted(
                (
                    entry
                    for entry in self._entries.values()
                    if start_after is None
                    or (entry.key, entry.generation) > start_after
                ),
                key=lambda entry: (entry.key, entry.generation),
            )
            entries: list[EntrySnapshot] = []
            work_seen = 0
            for entry in remaining:
                if len(entries) == effective_limit:
                    break
                entry_bytes = len(entry.manifest)
                if work_seen + entry_bytes > effective_work_cap:
                    if not entries:
                        raise CacheBlobBackendError(
                            "Migration inventory entry exceeds the configured work bound",
                            context={"operation": "migration_inventory"},
                        )
                    break
                entries.append(self._copy(entry))
                work_seen += entry_bytes
            exhausted = len(entries) == len(remaining)
            next_cursor = None
            if not exhausted:
                last = entries[-1]
                next_cursor = AuthorityInventoryCursor(
                    store_id=identity.store_id,
                    revision=identity.revision,
                    last_key=last.key,
                    last_generation=last.generation,
                )
            return AuthorityInventoryPage(
                identity=identity,
                entries=tuple(entries),
                next_cursor=next_cursor,
                exhausted=exhausted,
            )

    def activate_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> ActivationReceipt:
        """Publish one complete verified candidate as the only visibility change.

        The memory authority has one in-process transactional boundary: either
        every candidate descriptor becomes current together, or no descriptor
        changes. Candidate bytes and maintenance evidence remain external,
        non-authoritative effects until this method returns.
        """

        def activate() -> ActivationReceipt:
            identity = self.identity_snapshot()
            if receipt.destination_identity != identity:
                raise CacheBlobLifecycleConflictError(
                    "Migration destination identity or revision changed before activation"
                )
            if receipt.destination_revision != self._revision:
                raise CacheBlobLifecycleConflictError(
                    "Migration destination revision changed before activation"
                )
            if len(entries) != receipt.entry_count:
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate entry count disagrees with verified receipt"
                )
            if sum(entry.byte_size for entry in entries) != receipt.byte_count:
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate byte count disagrees with verified receipt"
                )
            if candidate_digest(entries) != receipt.candidate_digest:
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate digest disagrees with verified receipt"
                )
            if self._migration_state is AuthorityPublicationState.ACTIVATED_OFFLINE:
                if self._migration_receipt == receipt:
                    return ActivationReceipt(
                        candidate_receipt=receipt,
                        activation_revision=self._revision,
                        prior_store=PriorStoreReceipt(
                            run_id=receipt.run_id,
                            revision=self._migration_prior_revision,
                            entry_count=len(self._migration_prior_entries),
                        ),
                    )
                raise CacheBlobLifecycleConflictError(
                    "A different migration activation is already selected"
                )
            if (
                self._migration_state is not AuthorityPublicationState.CANDIDATE
                or self._migration_receipt != receipt
                or self._migration_candidates != entries
            ):
                raise CacheBlobLifecycleConflictError(
                    "Verified migration candidate is not recorded by this authority"
                )
            if len({entry.key for entry in entries}) != len(entries):
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate contains duplicate keys"
                )

            next_revision = self._revision + 1
            prior_entries = {
                key: self._copy(entry) for key, entry in self._entries.items()
            }
            activated: dict[str, EntrySnapshot] = {}
            for candidate in entries:
                manifest = BlobManifest.from_canonical_bytes(candidate.manifest)
                if (
                    manifest.key != candidate.key
                    or manifest.generation != candidate.generation
                    or manifest.locator != candidate.locator
                    or manifest.digest != candidate.payload_digest
                    or manifest.byte_size != candidate.byte_size
                ):
                    raise CacheBlobLifecycleConflictError(
                        "Migration candidate descriptor does not corroborate its receipt"
                    )
                lineage = self._lineages.get(candidate.key, 0) + 1
                self._lineages[candidate.key] = lineage
                activated[candidate.key] = EntrySnapshot(
                    key=candidate.key,
                    generation=candidate.generation,
                    locator=candidate.locator,
                    manifest=bytes(candidate.manifest),
                    expectation=EntryExpectation(
                        lineage=lineage,
                        revision=next_revision,
                        generation=candidate.generation,
                        manifest_digest=hashlib.sha256(candidate.manifest).hexdigest(),
                    ),
                    transport_evidence=candidate.transport_evidence,
                )
            self._entries = activated
            self._revision = next_revision
            self._projection_dirty = True
            self._migration_prior_entries = prior_entries
            self._migration_prior_revision = next_revision - 1
            self._migration_state = AuthorityPublicationState.ACTIVATED_OFFLINE
            return ActivationReceipt(
                candidate_receipt=receipt,
                activation_revision=self._revision,
                prior_store=PriorStoreReceipt(
                    run_id=receipt.run_id,
                    revision=self._migration_prior_revision,
                    entry_count=len(prior_entries),
                ),
            )

        return self._transition(activate)

    def record_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> VerifiedCandidateReceipt:
        """Record a complete in-process candidate without selecting it for readers."""
        if not isinstance(receipt, VerifiedCandidateReceipt) or not isinstance(
            entries, tuple
        ):
            raise TypeError(
                "migration candidate receipt and entries must be immutable values"
            )

        def record() -> VerifiedCandidateReceipt:
            identity = self.identity_snapshot()
            if (
                receipt.destination_identity != identity
                or receipt.destination_revision != self._revision
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration destination identity or revision changed before candidate recording"
                )
            if (
                len(entries) != receipt.entry_count
                or sum(entry.byte_size for entry in entries) != receipt.byte_count
                or candidate_digest(entries) != receipt.candidate_digest
                or len({entry.key for entry in entries}) != len(entries)
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate does not match its verified receipt"
                )
            if self._migration_state is AuthorityPublicationState.CANDIDATE:
                if (
                    self._migration_receipt == receipt
                    and self._migration_candidates == entries
                ):
                    return receipt
                if (
                    self._migration_receipt is not None
                    and self._migration_receipt.run_id == receipt.run_id
                    and self._migration_receipt.plan_digest == receipt.plan_digest
                    and self._migration_receipt.source_identity
                    == receipt.source_identity
                    and self._migration_receipt.source_revision
                    == receipt.source_revision
                    and self._migration_candidates
                    == entries[: len(self._migration_candidates)]
                ):
                    self._migration_receipt = receipt
                    self._migration_candidates = entries
                    return receipt
                raise CacheBlobLifecycleConflictError(
                    "A different verified migration candidate is already recorded"
                )
            if self._migration_state is not AuthorityPublicationState.IDLE:
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate recording requires an idle authority"
                )
            for candidate in entries:
                try:
                    manifest = BlobManifest.from_canonical_bytes(candidate.manifest)
                except (TypeError, ValueError) as error:
                    raise CacheBlobLifecycleConflictError(
                        "Migration candidate manifest is malformed"
                    ) from error
                if (
                    manifest.key != candidate.key
                    or manifest.generation != candidate.generation
                    or manifest.locator != candidate.locator
                    or manifest.digest != candidate.payload_digest
                    or manifest.byte_size != candidate.byte_size
                    or manifest.canonical_bytes() != candidate.manifest
                ):
                    raise CacheBlobLifecycleConflictError(
                        "Migration candidate descriptor does not corroborate its receipt"
                    )
            self._migration_receipt = receipt
            self._migration_candidates = entries
            self._migration_state = AuthorityPublicationState.CANDIDATE
            return receipt

        return self._transition(record)

    def record_candidate_verification(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> None:
        """Attach only freshly verified destination evidence to one candidate.

        This changes no visibility state and preserves the immutable candidate
        receipt.  The authority accepts the update only when every descriptor
        remains the exact candidate it already attributed.
        """
        if not isinstance(receipt, VerifiedCandidateReceipt) or not isinstance(
            entries, tuple
        ):
            raise TypeError(
                "migration candidate receipt and entries must be immutable values"
            )

        def record() -> None:
            if (
                self._migration_state is not AuthorityPublicationState.CANDIDATE
                or self._migration_receipt != receipt
                or len(self._migration_candidates) != len(entries)
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate verification requires its recorded candidate"
                )
            for stored, verified in zip(
                self._migration_candidates, entries, strict=True
            ):
                if (
                    stored.key != verified.key
                    or stored.generation != verified.generation
                    or stored.locator != verified.locator
                    or stored.manifest != verified.manifest
                    or stored.payload_digest != verified.payload_digest
                    or stored.byte_size != verified.byte_size
                    or stored.transport_evidence
                    not in {None, verified.transport_evidence}
                ):
                    raise CacheBlobLifecycleConflictError(
                        "Migration verification does not match the recorded candidate"
                    )
            self._migration_candidates = entries

        self._transition(record)

    def candidate_entries_for_run(
        self, *, run_id: str
    ) -> tuple[AuthorityInventoryEntry, ...]:
        """Return only exact candidate descriptors already held by this authority."""
        self._require_open()
        with self._lock:
            if (
                self._migration_receipt is None
                or self._migration_receipt.run_id != run_id
                or self._migration_state
                not in {
                    AuthorityPublicationState.CANDIDATE,
                    AuthorityPublicationState.ACTIVATED_OFFLINE,
                    AuthorityPublicationState.ACTIVE,
                    AuthorityPublicationState.ROLLED_BACK,
                }
            ):
                return ()
            return self._migration_candidates

    def discard_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> None:
        """Clear one exact unactivated candidate after external retirement succeeds."""
        if not isinstance(receipt, VerifiedCandidateReceipt) or not isinstance(
            entries, tuple
        ):
            raise TypeError(
                "migration candidate receipt and entries must be immutable values"
            )

        def discard() -> None:
            if (
                self._migration_state is not AuthorityPublicationState.CANDIDATE
                or self._migration_receipt != receipt
                or self._migration_candidates != entries
            ):
                raise CacheBlobLifecycleConflictError(
                    "Only the exact unactivated migration candidate may be discarded"
                )
            self._migration_receipt = None
            self._migration_candidates = ()
            self._migration_state = AuthorityPublicationState.IDLE

        self._transition(discard)

    def publication_state(self) -> AuthorityPublicationState:
        """Return narrow maintenance status without admitting ordinary workers."""
        self._require_open()
        with self._lock:
            return self._migration_state

    def activation_receipt_for_candidate(
        self, receipt: VerifiedCandidateReceipt
    ) -> ActivationReceipt | None:
        """Classify a lost activation response from exact authority-owned state."""
        if not isinstance(receipt, VerifiedCandidateReceipt):
            raise TypeError("receipt must be a VerifiedCandidateReceipt")
        self._require_open()
        with self._lock:
            if (
                self._migration_state is not AuthorityPublicationState.ACTIVATED_OFFLINE
                or self._migration_receipt != receipt
            ):
                return None
            return ActivationReceipt(
                candidate_receipt=receipt,
                activation_revision=self._revision,
                prior_store=PriorStoreReceipt(
                    run_id=receipt.run_id,
                    revision=self._migration_prior_revision,
                    entry_count=len(self._migration_prior_entries),
                ),
            )

    def require_ordinary_worker_access(self) -> None:
        """Reject ordinary reads, queries, and mutations during the rollback window."""
        if self.publication_state() is AuthorityPublicationState.ACTIVATED_OFFLINE:
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "Offline migration activation requires explicit rollback or finalize before workers restart",
                context={"operation": "migration.worker_access"},
            )

    def rollback_verified_candidate(self, *, run_id: str) -> RollbackReceipt:
        """Restore the retained prior selection before ordinary worker restart."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")

        def rollback() -> RollbackReceipt:
            if (
                self._migration_state is not AuthorityPublicationState.ACTIVATED_OFFLINE
                or self._migration_receipt is None
                or self._migration_receipt.run_id != run_id
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration rollback requires the selected offline activation"
                )
            self._revision += 1
            self._entries = {
                key: self._copy(entry)
                for key, entry in self._migration_prior_entries.items()
            }
            self._projection_dirty = True
            self._migration_state = AuthorityPublicationState.ROLLED_BACK
            return RollbackReceipt(run_id=run_id, rollback_revision=self._revision)

        return self._transition(rollback)

    def finalize_verified_candidate(self, *, run_id: str) -> FinalizeReceipt:
        """End rollback eligibility without deleting the retained prior material."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")

        def finalize() -> FinalizeReceipt:
            if (
                self._migration_state is not AuthorityPublicationState.ACTIVATED_OFFLINE
                or self._migration_receipt is None
                or self._migration_receipt.run_id != run_id
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration finalize requires the selected offline activation"
                )
            self._migration_state = AuthorityPublicationState.ACTIVE
            return FinalizeReceipt(run_id=run_id, finalized_revision=self._revision)

        return self._transition(finalize)

    def retained_prior_entries(self, *, run_id: str) -> tuple[EntrySnapshot, ...]:
        """Return exact retained prior rows only after this run is finalized."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        self._require_open()
        with self._lock:
            if (
                self._migration_state is not AuthorityPublicationState.ACTIVE
                or self._migration_receipt is None
                or self._migration_receipt.run_id != run_id
            ):
                raise CacheBlobLifecycleConflictError(
                    "Retained prior entries require the finalized selected migration"
                )
            return tuple(
                self._copy(entry)
                for entry in sorted(
                    self._migration_prior_entries.values(),
                    key=lambda entry: (entry.key, entry.generation),
                )
            )

    def catalog_page(
        self,
        query: CatalogQuery,
        cursor: str | None,
        *,
        schema: CatalogSchema,
        limit: int,
        work_cap: int,
        signing_key: bytes,
        manifest_loader: Callable[[bytes], Any],
    ) -> CatalogPage:
        """Scan authenticated current descriptors in portable keyset order."""
        validate_catalog_page_request(
            query,
            schema=schema,
            cursor=cursor,
            limit=limit,
            work_cap=work_cap,
        )
        if cursor is not None:
            CatalogCursor.inspect(cursor, signing_key=signing_key)
        self._require_open()
        with self._lock:
            cursor_identity = (
                None
                if cursor is None
                else CatalogCursor.parse(
                    cursor,
                    store_id=self._catalog_store_id,
                    format_version=2,
                    schema_id=schema.schema_id,
                    schema_fingerprint=schema.fingerprint,
                    query_fingerprint=query.fingerprint,
                    revision=self._revision,
                    signing_key=signing_key,
                )
            )
            snapshots = tuple(
                self._copy(entry)
                for entry in self._entries.values()
                if cursor_identity is None
                or (entry.key, entry.generation) > cursor_identity
            )
            ordered = tuple(
                sorted(snapshots, key=lambda entry: (entry.key, entry.generation))
            )
            return page_from_canonical_scan(
                ordered[: work_cap + 1],
                query=query,
                schema=schema,
                revision=self._revision,
                store_id=self._catalog_store_id,
                cursor_identity=cursor_identity,
                limit=limit,
                work_cap=work_cap,
                signing_key=signing_key,
                manifest_loader=manifest_loader,
            )

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
                for debt in self._debts.values()
                if (key is None or debt.key == key)
                and (operation_id is None or debt.operation_id == operation_id)
            )

    def pending_mutations(self) -> tuple[PreparedMutation, ...]:
        """Return indexed, pre-promotion operations for exact recovery only."""
        self._require_open()
        with self._lock:
            return tuple(
                PreparedMutation(operation_id, spec)
                for operation_id, (spec, _proof, state, _promoted_entry) in sorted(
                    self._mutations.items()
                )
                if state == "prepared"
            )

    def retire_cleanup_debt(self, debt: CleanupDebt) -> None:
        """Retire one exact debt idempotently after external reclamation."""

        def retire() -> None:
            for row_id, pending in self._debts.items():
                if pending == debt:
                    del self._debts[row_id]
                    break

        self._transition(retire)

    def _add_debt(self, debt: CleanupDebt) -> None:
        """Allocate an identity that remains stable after earlier rows retire."""
        self._debt_high_water += 1
        self._debts[self._debt_high_water] = debt

    def delete_entry(self, key: str, *, expected: EntryExpectation) -> None:
        def delete() -> None:
            if self._expectation(key) != expected:
                raise CacheBlobLifecycleConflictError(
                    "Delete expectation no longer matches authority"
                )
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
                raise CacheBlobLifecycleConflictError(
                    "Clear run cannot accept checkpoint"
                )
            if target is None:
                self._clear_states[token.value] = "completed"
                return
            if state not in {"completed", "conflicted", "blocked"}:
                raise ValueError("Clear target state is unsupported")
            stored = self._clear_targets[token.value].get(target.key)
            if stored is None or stored[0] != target or stored[1] != "pending":
                raise CacheBlobLifecycleConflictError(
                    "Clear target is no longer pending"
                )
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
                self._mutation_high_water,
                self._debt_high_water,
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
                    raise CacheBlobLifecycleConflictError(
                        "Reconciliation run does not exist"
                    )
                return snapshot
            return ReconciliationSnapshot(
                self._revision,
                self._mutation_high_water,
                self._debt_high_water,
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
            mutations = sorted(
                (row_id, operation_id)
                for operation_id, row_id in self._mutation_order.items()
                if mutation_cursor < row_id <= snapshot.mutation_high_water
            )[:page_size]
            debts = [
                (row_id, debt)
                for row_id, debt in self._debts.items()
                if debt_cursor < row_id <= snapshot.debt_high_water
            ][:page_size]
            mutation_stop = (
                mutations[-1][0] if mutations else snapshot.mutation_high_water
            )
            debt_stop = debts[-1][0] if debts else snapshot.debt_high_water
            works: list[ReconciliationWork] = []
            for row_id, operation_id in mutations:
                spec, _proof, state, _promoted_entry = self._mutations[operation_id]
                if state == "prepared":
                    works.append(
                        ReconciliationWork(
                            "mutation",
                            row_id,
                            state,
                            mutation=PreparedMutation(operation_id, spec),
                        )
                    )
            for row_id, debt in debts:
                works.append(ReconciliationWork("debt", row_id, "pending", debt=debt))
            return ReconciliationPage(tuple(works), mutation_stop, debt_stop)

    def page_reconciliation(self, token: PageToken) -> tuple[CleanupDebt, ...]:
        self._require_open()
        with self._lock:
            if token.value not in self._reconciliation_states:
                raise CacheBlobLifecycleConflictError(
                    "Reconciliation run does not exist"
                )
            return tuple(self._debts.values())

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

    @contextmanager
    def projection_backup(self) -> Iterator[ProjectionBackup]:
        """Reject projection exports for this explicitly non-projecting authority.

        The in-memory authority fulfills the lifecycle interface so composition
        can validate it structurally, while its false projection capability
        continues to prevent callers from inferring an isolated export
        guarantee that this same-process implementation cannot provide.
        """
        self._require_open()
        raise CacheBlobLifecycleConflictError(
            "In-memory lifecycle authority does not support projection backups"
        )
        yield  # pragma: no cover - required for the context-manager type.

    def close(self) -> None:
        with self._lock:
            self._closed = True


__all__ = ["InMemoryLifecycleAuthority"]
