"""Authority-backed BlobStore mutation orchestration.

The authority is the only committed truth. Payload work stays outside every
authority transaction and native handler bytes are never wrapped.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobManifestMalformedError,
    CacheBlobManifestUnauthenticatedError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobPayloadMissingError,
    CacheBlobPayloadTamperedError,
    CacheBlobRecoverableCleanupError,
)

from .integrity import sha256_and_size, sign_hmac_sha256
from .lifecycle_authority import (
    CleanupDebt,
    EntryExpectation,
    EntrySnapshot,
    LifecycleAuthority,
    MutationSpec,
    PageToken,
    PreparedMutation,
    VerificationProof,
)
from .manifest import BlobManifestV1
from .path_security import resolve_managed_locator


_TOMBSTONE_OPERATION_ID_FIELD = "_cacheness_tombstone_operation_id"


@dataclass(frozen=True)
class _LifecycleHookContext:
    """Non-durable test timing context retained across the scheduler removal."""

    key: str | None


@dataclass(frozen=True)
class LifecyclePutResult:
    """Private immutable context for one completed authority-backed put."""

    key: str
    expected: EntryExpectation
    promoted: EntrySnapshot | None
    previous: EntrySnapshot | None
    expected_projection_locator: str | None = None


class AuthorityLifecycleEngine:
    """Coordinate immutable native payloads through complete authority calls."""

    def __init__(self, store: Any, authority: LifecycleAuthority):
        self.store = store
        self.authority = authority
        self.lifecycle_limits = authority.lifecycle_limits
        self.test_hook: Callable[[str], None] | None = None
        self.fault_hook: Callable[[str], None] | None = None

    def _reach(self, boundary: str, *, key: str | None = None) -> None:
        for hook in (self.test_hook, self.fault_hook):
            if hook is None:
                continue
            try:
                hook(boundary)
            except TypeError:
                # Phase 2 test hooks received a transient record as a second
                # argument. Keep that non-persistent timing seam compatible
                # while the scheduler record itself stays retired.
                legacy = {
                    "put.intent_prepared": "evidence_created",
                    "put.candidate_published": "candidate_published",
                    "put.candidate_verified": "candidate_verified",
                    "put.before_promotion": "manifest_publish",
                    "delete.before_tombstone_promotion": "tombstone_publish",
                    "cleanup.after_payload_delete": "cleanup_completed",
                    "put.cleanup_retired": "evidence_retired",
                }.get(boundary, boundary)
                hook(legacy, _LifecycleHookContext(key))

    def _entry_manifest(
        self, entry: EntrySnapshot, *, allow_tombstone: bool = False
    ) -> BlobManifestV1:
        manifest = self.store._authenticated_authority_manifest(entry.manifest)
        if manifest.key != entry.key:
            raise CacheBlobLifecycleConflictError(
                "Authority manifest key conflicts with its entry"
            )
        if manifest.generation != entry.generation:
            raise CacheBlobLifecycleConflictError(
                "Authority manifest generation conflicts with its entry"
            )
        if manifest.locator != entry.locator:
            raise CacheBlobLifecycleConflictError(
                "Authority manifest locator conflicts with its entry"
            )
        allowed_states = (
            {"committed", "tombstoned"} if allow_tombstone else {"committed"}
        )
        if manifest.state not in allowed_states:
            raise CacheBlobLifecycleConflictError(
                "Authority manifest state conflicts with the operation"
            )
        resolve_managed_locator(
            self.store._materialize_authority_store().root,
            manifest.locator,
            operation="authority_manifest",
        )
        return manifest

    def _sign(
        self, manifest: BlobManifestV1, *, initialize_new_store: bool = False
    ) -> BlobManifestV1:
        return manifest.with_signature(
            sign_hmac_sha256(
                manifest.signing_bytes(),
                self.store._authority_manifest_key(
                    initialize_new_store=initialize_new_store
                ),
            )
        )

    def _candidate_locator(self, key: str, generation: str, suffix: str) -> Path:
        return (
            Path("generations")
            / self.store._storage_id_for_key(key)
            / f"{generation}{suffix}"
        )

    def _settle_debts(self, debts: tuple[CleanupDebt, ...]) -> None:
        for debt in debts:
            try:
                current = self.authority.read_entry(debt.key) if debt.key else None
                if current is not None:
                    current_manifest = self._entry_manifest(
                        current, allow_tombstone=True
                    )
                    if current_manifest.locator == debt.locator:
                        raise CacheBlobLifecycleConflictError(
                            "Lifecycle cleanup debt still belongs to the current generation",
                            context={"operation": "cleanup", "generation": debt.generation},
                        )
                self._reach("cleanup.before_payload_delete", key=debt.key)
                self.store._delete_or_prove_absent(Path(debt.locator))
                self._reach("cleanup.after_payload_delete", key=debt.key)
                self.authority.retire_cleanup_debt(debt)
            except CacheBlobLifecycleConflictError:
                raise
            except Exception as exc:
                raise CacheBlobRecoverableCleanupError(
                    "Committed lifecycle state is durable but cleanup needs recovery",
                    context={"operation_id": debt.operation_id, "generation": debt.generation},
                ) from exc

    def _abort(self, prepared: PreparedMutation, *, candidate_persisted: bool) -> None:
        self.authority.abort_mutation(
            prepared, candidate_persisted=candidate_persisted
        )
        if candidate_persisted:
            self._settle_debts(
                self.authority.pending_cleanup_debts(
                    operation_id=prepared.operation_id
                )
            )

    def put(
        self,
        data: Any,
        *,
        key: str,
        metadata: dict[str, Any] | None,
        projection_context: str | None = None,
    ) -> LifecyclePutResult:
        """Prepare, publish, verify, promote, then reclaim exact old debt."""
        handler = self.store.handlers.get_handler(data)
        # A Windows authority root is a deployment-provisioned security
        # boundary. Validate it before read_entry can bootstrap SQLite or
        # guarded handler I/O can create the managed payload root.
        self.authority.preflight_mutation()
        previous = self.authority.read_entry(key)
        if previous is not None:
            self._entry_manifest(previous, allow_tombstone=True)
        expected = (
            previous.expectation
            if previous is not None
            else self.authority.read_expectation(key)
        )
        guarded_io = self.store._materialize_authority_store()
        with guarded_io.stage(handler, data, self.store.config) as staged:
            generation = uuid4().hex
            locator = self._candidate_locator(key, generation, staged.suffix)
            prepared = self.authority.prepare_mutation(
                MutationSpec.create(
                    operation_id=uuid4().hex,
                    key=key,
                    generation=generation,
                    candidate_locator=locator.as_posix(),
                    expected=expected,
                )
            )
            candidate_persisted = False
            try:
                self._reach("put.intent_prepared", key=key)
                self._reach("put.before_candidate_publish", key=key)
                published = guarded_io.publish_generation(staged, locator)
                candidate_persisted = True
                self._reach("put.candidate_published", key=key)
                with guarded_io.open_snapshot(
                    locator, dict(published.get("metadata", {}))
                ) as snapshot:
                    digest, byte_size = sha256_and_size(snapshot.path)
                payload_format = str(
                    published.get(
                        "payload_format",
                        getattr(
                            handler,
                            "payload_format",
                            published.get("storage_format", "native"),
                        ),
                    )
                )
                payload_format_version = int(
                    published.get(
                        "payload_format_version",
                        getattr(handler, "payload_format_version", 1),
                    )
                )
                manifest = BlobManifestV1(
                    schema_version=1,
                    key=key,
                    generation=generation,
                    state="committed",
                    locator=locator.as_posix(),
                    handler_type=handler.data_type,
                    payload_format=payload_format,
                    payload_format_version=payload_format_version,
                    digest_algorithm="sha256",
                    digest=digest,
                    byte_size=byte_size,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    handler_metadata={
                        **dict(published.get("metadata", {})),
                        "storage_format": payload_format,
                    },
                    user_metadata=dict(metadata or {}),
                )
                before_promotion = getattr(
                    self.store, "_before_authority_promotion", None
                )
                if callable(before_promotion):
                    manifest = before_promotion(
                        manifest,
                        LifecyclePutResult(
                            key=key,
                            expected=expected,
                            promoted=None,
                            previous=previous,
                            expected_projection_locator=projection_context,
                        ),
                    )
                    if not isinstance(manifest, BlobManifestV1):
                        raise CacheBlobLifecycleConflictError(
                            "Lifecycle projection hook must return a BlobManifestV1"
                        )
                manifest = self._sign(
                    manifest, initialize_new_store=previous is None
                )
                self._reach("put.candidate_verified", key=key)
                self.authority.record_verification(
                    prepared,
                    VerificationProof(
                        digest=digest,
                        byte_size=byte_size,
                        manifest=manifest.canonical_bytes(),
                    ),
                )
                self._reach("put.before_promotion", key=key)
                promoted = self.authority.promote_mutation(prepared)
            except BaseException as error:
                try:
                    self._abort(prepared, candidate_persisted=candidate_persisted)
                except Exception as cleanup_error:
                    # Preserve the pre-promotion failure as the direct cause.
                    # The cleanup failure is the surfaced recoverable state,
                    # but callers need the original serialization, signing, or
                    # projection error to diagnose why the candidate existed.
                    raise cleanup_error from error
                raise
        self._reach("put.promoted", key=key)
        self._settle_debts(promoted.cleanup_debt)
        self._reach("put.cleanup_retired", key=key)
        return LifecyclePutResult(
            key=key,
            expected=expected,
            promoted=promoted.entry,
            previous=previous,
            expected_projection_locator=projection_context,
        )

    def update_metadata(self, key: str, metadata: dict[str, Any]) -> bool:
        """Promote a new signed metadata revision without changing payload bytes."""
        if not isinstance(metadata, dict):
            raise CacheBlobLifecycleConflictError("BlobStore metadata patches must be dictionaries")
        entry = self.authority.read_entry(key)
        if entry is None:
            return False
        manifest = self._entry_manifest(entry)
        immutable = self.store._immutable_metadata_patch_fields.intersection(metadata)
        if immutable:
            raise CacheBlobLifecycleConflictError(
                "BlobStore metadata patches cannot change canonical structural fields",
                context={"fields": sorted(immutable)},
            )
        updated = self._sign(
            replace(
                manifest,
                generation=uuid4().hex,
                user_metadata={**dict(manifest.user_metadata), **metadata},
                signature="",
            )
        )
        prepared = self.authority.prepare_mutation(
            MutationSpec.create(
                operation_id=uuid4().hex,
                key=key,
                generation=updated.generation,
                candidate_locator=manifest.locator,
                expected=entry.expectation,
            )
        )
        try:
            self._reach("metadata.intent_prepared", key=key)
            with self.store._materialize_authority_store().open_snapshot(
                manifest.locator, self.store._handler_metadata(manifest)
            ) as snapshot:
                digest, byte_size = sha256_and_size(snapshot.path)
            if digest != manifest.digest or byte_size != manifest.byte_size:
                raise CacheBlobPayloadTamperedError(
                    "Authority lifecycle payload verification failed"
                )
            self.authority.record_verification(
                prepared,
                VerificationProof(
                    digest=updated.digest,
                    byte_size=updated.byte_size,
                    manifest=updated.canonical_bytes(),
                ),
            )
            self._reach("metadata.before_promotion", key=key)
            self.authority.promote_mutation(prepared)
        except Exception:
            self._abort(prepared, candidate_persisted=False)
            raise
        return True

    def delete(self, key: str, *, expected: EntryExpectation | None = None) -> bool:
        """Promote a signed tombstone before reclaiming its previous payload."""
        entry = self.authority.read_entry(key)
        if entry is None:
            return False
        if expected is not None and expected != entry.expectation:
            raise CacheBlobLifecycleConflictError("Delete expectation no longer matches authority")
        manifest = self._entry_manifest(entry, allow_tombstone=True)
        if manifest.state == "tombstoned":
            operation_id = manifest.handler_metadata.get(_TOMBSTONE_OPERATION_ID_FIELD)
            if not isinstance(operation_id, str) or not operation_id:
                raise CacheBlobLifecycleConflictError("Lifecycle tombstone lacks operation provenance")
            self._settle_debts(
                self.authority.pending_cleanup_debts(operation_id=operation_id)
            )
            self.authority.retire_tombstone(key, expected=entry.expectation)
            return True
        operation_id = uuid4().hex
        generation = uuid4().hex
        tombstone_metadata = dict(manifest.handler_metadata)
        tombstone_metadata[_TOMBSTONE_OPERATION_ID_FIELD] = operation_id
        tombstone = self._sign(
            replace(
                manifest,
                generation=generation,
                state="tombstoned",
                locator=(
                    Path("tombstones") / self.store._storage_id_for_key(key) / generation
                ).as_posix(),
                handler_metadata=tombstone_metadata,
                signature="",
            )
        )
        prepared = self.authority.prepare_mutation(
            MutationSpec.create(
                operation_id=operation_id,
                key=key,
                generation=generation,
                candidate_locator=tombstone.locator,
                expected=entry.expectation,
            )
        )
        try:
            self._reach("delete.intent_prepared", key=key)
            self.authority.record_verification(
                prepared,
                VerificationProof(
                    digest=tombstone.digest,
                    byte_size=tombstone.byte_size,
                    manifest=tombstone.canonical_bytes(),
                ),
            )
            self._reach("delete.before_tombstone_promotion", key=key)
            promoted = self.authority.promote_mutation(prepared)
        except Exception:
            self._abort(prepared, candidate_persisted=False)
            raise
        self._settle_debts(promoted.cleanup_debt)
        self.authority.retire_tombstone(key, expected=promoted.entry.expectation)
        return True

    def _read(self, key: str, *, deserialize: bool) -> Any | None:
        for attempt in range(2):
            entry = self.authority.read_entry(key)
            if entry is None:
                return None
            manifest = self._entry_manifest(entry, allow_tombstone=True)
            if manifest.state == "tombstoned":
                return None
            handler = self.store._resolve_payload_handler(manifest)
            try:
                with self.store._materialize_authority_store().open_snapshot(
                    manifest.locator, self.store._handler_metadata(manifest)
                ) as snapshot:
                    second = self.authority.read_entry(key)
                    if second is None or second.expectation != entry.expectation:
                        if attempt == 0:
                            continue
                        raise CacheBlobLifecycleConflictError(
                            "Authority changed during both read attempts",
                            context={"key": key, "operation": "read"},
                        )
                    second_manifest = self._entry_manifest(second, allow_tombstone=True)
                    if second_manifest.state != "committed":
                        if attempt == 0:
                            continue
                        raise CacheBlobLifecycleConflictError(
                            "Authority changed during both read attempts",
                            context={"key": key, "operation": "read"},
                        )
                    digest, byte_size = sha256_and_size(snapshot.path)
                    if digest != manifest.digest or byte_size != manifest.byte_size:
                        raise CacheBlobPayloadTamperedError(
                            "Canonical BlobStore payload integrity check failed"
                        )
                    return handler.get(snapshot.path, snapshot.metadata) if deserialize else True
            except FileNotFoundError as exc:
                if attempt == 0:
                    continue
                raise CacheBlobPayloadMissingError("Canonical BlobStore payload is missing") from exc
        raise CacheBlobLifecycleConflictError(
            "Authority read exhausted its bounded generation retry",
            context={"key": key, "operation": "read"},
        )

    def get(self, key: str) -> Any | None:
        return self._read(key, deserialize=True)

    def exists(self, key: str) -> bool:
        return bool(self._read(key, deserialize=False))

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        entry = self.authority.read_entry(key)
        if entry is None:
            return None
        manifest = self._entry_manifest(entry, allow_tombstone=True)
        return None if manifest.state == "tombstoned" else self.store._manifest_entry_data(manifest)

    def list(self, prefix: str | None, metadata_filter: dict[str, Any] | None) -> list[str]:
        keys: list[str] = []
        for entry in self.authority.list_entries():
            manifest = self._entry_manifest(entry, allow_tombstone=True)
            if manifest.state != "committed" or (prefix and not manifest.key.startswith(prefix)):
                continue
            if metadata_filter and any(
                manifest.user_metadata.get(name) != value
                for name, value in metadata_filter.items()
            ):
                continue
            keys.append(manifest.key)
        return keys

    def clear(self) -> int:
        """Remove one bounded, authority-owned clear snapshot safely.

        ``begin_clear`` commits immutable target membership before this method
        reaches any payload operation.  A restart returns the active run rather
        than creating a second snapshot, allowing a crash between deletion and
        checkpointing to converge without ever targeting post-snapshot state.
        """
        token = self.begin_clear()
        return self.complete_clear(token)

    def begin_clear(self) -> PageToken:
        """Persist the finite clear membership inside the short admission window."""
        token = self.authority.begin_clear()
        self._reach("clear.snapshot_committed")
        return token

    def complete_clear(self, token: PageToken) -> int:
        """Reclaim one already-persisted clear snapshot outside admission gating."""
        deadline = time.monotonic() + self.lifecycle_limits.authority_busy_timeout_seconds
        byte_budget = self.lifecycle_limits.max_operation_record_bytes
        action_budget = self.lifecycle_limits.max_reconcile_actions
        removed = 0
        consumed_bytes = 0
        consumed_actions = 0

        while (
            consumed_actions < action_budget
            and consumed_bytes < byte_budget
            and time.monotonic() < deadline
        ):
            targets = self.authority.page_clear(token)
            if not targets:
                self.authority.checkpoint_clear(token)
                return removed
            # A bounded authority page still needs a fail-closed current-entry
            # locator check before it mutates any member.  This is intentionally
            # page-local: clear never materializes every target or enumerates
            # payload names, and each destructive action reloads the current
            # entry below for its exact revalidation.
            for target in targets:
                current = self.authority.read_entry(target.key)
                if current is None:
                    continue
                try:
                    self._entry_manifest(current, allow_tombstone=True)
                except (
                    CacheBlobLifecycleConflictError,
                    CacheBlobManifestMalformedError,
                    CacheBlobManifestUnauthenticatedError,
                    CacheBlobManifestUnsupportedVersionError,
                ):
                    continue
            for target in targets:
                if (
                    consumed_actions >= action_budget
                    or consumed_bytes + len(target.manifest) > byte_budget
                    or time.monotonic() >= deadline
                ):
                    return removed
                # Authenticate the snapshot's own canonical bytes before it
                # participates in an exact-delete attempt.  No handler or
                # payload bytes are opened for this decision.  Ambiguous
                # evidence stays blocked and cannot revoke another target.
                try:
                    self._entry_manifest(target, allow_tombstone=True)
                except (
                    CacheBlobLifecycleConflictError,
                    CacheBlobManifestMalformedError,
                    CacheBlobManifestUnauthenticatedError,
                    CacheBlobManifestUnsupportedVersionError,
                ):
                    self.authority.checkpoint_clear(token, target, state="blocked")
                    consumed_actions += 1
                    consumed_bytes += len(target.manifest)
                    continue
                current = self.authority.read_entry(target.key)
                checkpoint_state = "conflicted"
                try:
                    if current is None:
                        # The authority checkpoint verifies lineage advanced
                        # beyond the captured target before accepting absence.
                        checkpoint_state = "completed"
                    elif current.expectation == target.expectation:
                        self._reach("clear.before_target_delete", key=target.key)
                        removed += int(self.delete(target.key, expected=target.expectation))
                        self._reach("clear.after_target_delete", key=target.key)
                        checkpoint_state = "completed"
                    else:
                        current_manifest = self._entry_manifest(
                            current, allow_tombstone=True
                        )
                        debts = self.authority.pending_cleanup_debts(key=target.key)
                        owns_target = any(
                            debt.generation == target.generation
                            and debt.locator == target.locator
                            for debt in debts
                        )
                        if current_manifest.state == "tombstoned" and owns_target:
                            self.delete(target.key, expected=current.expectation)
                            checkpoint_state = "completed"
                except CacheBlobLifecycleConflictError:
                    checkpoint_state = "conflicted"
                except (
                    CacheBlobManifestMalformedError,
                    CacheBlobManifestUnauthenticatedError,
                    CacheBlobManifestUnsupportedVersionError,
                ):
                    checkpoint_state = "blocked"
                self.authority.checkpoint_clear(
                    token, target, state=checkpoint_state
                )
                consumed_actions += 1
                consumed_bytes += len(target.manifest)
        return removed

    def reconcile(self) -> int:
        prepared = self.authority.pending_mutations()
        for mutation in prepared:
            self._abort(mutation, candidate_persisted=True)
        debts = self.authority.pending_cleanup_debts()
        self._settle_debts(debts)
        return len(prepared) + len(debts)


LifecycleEngine = AuthorityLifecycleEngine


__all__ = ["AuthorityLifecycleEngine", "LifecycleEngine"]
