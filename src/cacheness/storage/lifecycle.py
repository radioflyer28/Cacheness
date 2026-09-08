"""Authority-backed BlobStore mutation orchestration.

The authority is the only committed truth. Payload work stays outside every
authority transaction and native handler bytes are never wrapped.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobManifestMalformedError,
    CacheBlobManifestUnauthenticatedError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobPayloadMissingError,
    CacheBlobPayloadTamperedError,
    CacheBlobRecoverableCleanupError,
)

from .integrity import sha256_and_size
from .catalog import CatalogSchema
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
from .manifest import BlobManifest, StoreVersionDimensions, sign_current_manifest
from .path_security import resolve_managed_locator
from .read_contract import BlobEntry


_TOMBSTONE_OPERATION_ID_FIELD = "_cacheness_tombstone_operation_id"
_DEFAULT_CATALOG_SCHEMA_ID = "cacheness.default"
_DEFAULT_CATALOG_SCHEMA_REVISION = 1
_DEFAULT_CATALOG_SCHEMA_FINGERPRINT = hashlib.sha256(
    b"cacheness.catalog.default.v1"
).hexdigest()


@dataclass(frozen=True)
class _LifecycleHookContext:
    """Non-durable test timing context retained across the scheduler removal."""

    key: str | None


@dataclass(frozen=True)
class LifecyclePutResult:
    """Private immutable context for one completed authority-backed put."""

    operation_id: str
    key: str
    expected: EntryExpectation
    promoted: EntrySnapshot | None
    previous: EntrySnapshot | None
    cleanup_debt: tuple[CleanupDebt, ...] = ()


class AuthorityLifecycleEngine:
    """Coordinate immutable native payloads through complete authority calls."""

    def __init__(self, store: Any, authority: LifecycleAuthority):
        self.store = store
        self.authority = authority
        self.lifecycle_limits = authority.lifecycle_limits
        self.test_hook: Callable[[str], None] | None = None
        self.fault_hook: Callable[[str], None] | None = None

    def _reach(
        self,
        boundary: str,
        *,
        key: str | None = None,
        include_timing_hook: bool = True,
    ) -> None:
        """Reach one deterministic test boundary without changing lifecycle state.

        Stage boundaries are fault-only so existing timing observers retain
        their stable public sequence.  They are intentionally a test seam,
        not a lifecycle coordination mechanism or a second visibility point.
        """
        hooks = (
            (self.test_hook, self.fault_hook)
            if include_timing_hook
            else (None, self.fault_hook)
        )
        for hook in hooks:
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
    ) -> BlobManifest:
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
        self, manifest: BlobManifest, *, initialize_new_store: bool = False
    ) -> BlobManifest:
        return sign_current_manifest(
            manifest,
            self.store._authority_manifest_key(
                initialize_new_store=initialize_new_store
            ),
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
        catalog_schema_id: str,
        catalog_schema_revision: int,
        catalog_schema_fingerprint: str,
        catalog_values: dict[str, Any],
    ) -> LifecyclePutResult:
        """Prepare, publish, verify, promote, then reclaim exact old debt."""
        handler = self.store.handlers.get_handler(data)
        # A Windows authority root is a deployment-provisioned security
        # boundary. Validate it before read_entry can bootstrap SQLite or
        # guarded handler I/O can create the managed payload root.
        self.authority.preflight_mutation()
        # The same explicit startup path serves single-process convenience.
        # Existing catalogs validate only; they are never implicitly migrated.
        self.store.initialize()
        previous = self.authority.read_entry(key)
        if previous is not None:
            self._entry_manifest(previous, allow_tombstone=True)
        expected = (
            previous.expectation
            if previous is not None
            else self.authority.read_expectation(key)
        )
        guarded_io = self.store._materialize_authority_store()
        self._reach("put.before_stage", key=key, include_timing_hook=False)
        with guarded_io.stage(handler, data, self.store.config) as staged:
            self._reach("put.after_stage", key=key, include_timing_hook=False)
            generation = uuid4().hex
            locator = self._candidate_locator(key, generation, staged.suffix)
            raw_result = staged.raw_result
            with staged.open() as (source, byte_size):
                digest_builder = hashlib.sha256()
                while chunk := source.read(64 * 1024):
                    digest_builder.update(chunk)
                digest = digest_builder.hexdigest()
            payload_format = str(
                raw_result.get(
                    "payload_format",
                    getattr(
                        handler,
                        "payload_format",
                        raw_result.get("storage_format", "native"),
                    ),
                )
            )
            payload_format_version = int(
                raw_result.get(
                    "payload_format_version",
                    getattr(handler, "payload_format_version", 1),
                )
            )
            handler_metadata = raw_result.get("metadata", {})
            if not isinstance(handler_metadata, dict):
                raise CacheBlobLifecycleConflictError(
                    "Handler staging metadata must be a mapping"
                )
            manifest = BlobManifest(
                versions=StoreVersionDimensions(
                    payload_format_version=payload_format_version
                ),
                key=key,
                generation=generation,
                locator=locator.as_posix(),
                handler_type=handler.data_type,
                payload_format=payload_format,
                digest=digest,
                byte_size=byte_size,
                created_at=datetime.now(timezone.utc).isoformat(),
                catalog_schema_id=catalog_schema_id,
                catalog_schema_revision=catalog_schema_revision,
                catalog_schema_fingerprint=catalog_schema_fingerprint,
                catalog_values=catalog_values,
                catalog_presence=tuple(sorted(catalog_values)),
                user_metadata=dict(metadata or {}),
                handler_metadata={
                    **handler_metadata,
                    "storage_format": payload_format,
                },
            )
            before_promotion = getattr(self.store, "_before_authority_promotion", None)
            if callable(before_promotion):
                unsigned_manifest = manifest
                manifest = before_promotion(
                    manifest,
                    LifecyclePutResult(
                        operation_id="pending",
                        key=key,
                        expected=expected,
                        promoted=None,
                        previous=previous,
                    ),
                )
                if not isinstance(manifest, BlobManifest):
                    raise CacheBlobLifecycleConflictError(
                        "Lifecycle hook must return a format-2 BlobManifest"
                    )
                if (
                    replace(manifest, user_metadata=unsigned_manifest.user_metadata)
                    != unsigned_manifest
                ):
                    raise CacheBlobLifecycleConflictError(
                        "Compatibility metadata cannot change blob identity or integrity fields"
                    )
            manifest = self._sign(
                manifest, initialize_new_store=previous is None
            )
            prepared = self.authority.prepare_mutation(
                MutationSpec.create(
                    operation_id=uuid4().hex,
                    key=key,
                    generation=generation,
                    candidate_locator=locator.as_posix(),
                    expected=expected,
                    manifest=manifest.canonical_bytes(),
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
                # The staged digest is carried into the signed descriptor
                # before durable intent is recorded. Verify the published
                # immutable generation independently before promotion.
                if digest != manifest.digest or byte_size != manifest.byte_size:
                    raise CacheBlobLifecycleConflictError(
                        "Published payload disagrees with its prepared descriptor"
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
        try:
            self._settle_debts(promoted.cleanup_debt)
        except CacheBlobRecoverableCleanupError as error:
            error.context.update(
                committed=True, key=key, generation=promoted.entry.generation,
                expectation=promoted.entry.expectation,
            )
            raise
        self._reach("put.cleanup_retired", key=key)
        return LifecyclePutResult(
            operation_id=prepared.operation_id,
            key=key,
            expected=expected,
            promoted=promoted.entry,
            previous=previous,
            cleanup_debt=tuple(promoted.cleanup_debt),
        )

    def update_catalog(
        self,
        key: str,
        *,
        catalog_schema: CatalogSchema,
        catalog_values: dict[str, Any],
        expected: EntryExpectation | None,
        replace_values: bool,
    ) -> LifecyclePutResult | None:
        """Promote catalog attributes against the exact current record only."""
        entry = self.authority.read_entry(key)
        if entry is None:
            return None
        if expected is not None and expected != entry.expectation:
            raise CacheBlobLifecycleConflictError(
                "Catalog update expectation no longer matches authority"
            )
        manifest = self._entry_manifest(entry)
        if (
            manifest.catalog_schema_id != catalog_schema.schema_id
            or manifest.catalog_schema_revision != catalog_schema.revision
            or manifest.catalog_schema_fingerprint != catalog_schema.fingerprint
        ):
            raise CacheBlobLifecycleConflictError(
                "Catalog update schema does not match the committed descriptor"
            )
        candidate_values = (
            dict(catalog_values)
            if replace_values
            else {**dict(manifest.catalog_values), **catalog_values}
        )
        validated_values = catalog_schema.validate_mapping(
            candidate_values, materialize_defaults=False
        )
        updated = self._sign(
            replace(
                manifest,
                catalog_values=validated_values,
                catalog_presence=tuple(sorted(validated_values)),
                signature="",
            )
        )
        prepared = self.authority.prepare_mutation(
            MutationSpec.create(
                operation_id=uuid4().hex,
                key=key,
                generation=manifest.generation,
                candidate_locator=manifest.locator,
                expected=entry.expectation,
                manifest=updated.canonical_bytes(),
            )
        )
        try:
            self._reach("catalog.intent_prepared", key=key)
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
            self._reach("catalog.before_promotion", key=key)
            promoted = self.authority.promote_mutation(prepared)
        except Exception:
            self._abort(prepared, candidate_persisted=False)
            raise
        return LifecyclePutResult(
            operation_id=prepared.operation_id,
            key=key,
            expected=entry.expectation,
            promoted=promoted.entry,
            previous=entry,
            cleanup_debt=tuple(promoted.cleanup_debt),
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
                manifest=updated.canonical_bytes(),
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
                manifest=tombstone.canonical_bytes(),
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

    def entry_info(self, entry: EntrySnapshot) -> BlobEntry:
        """Render metadata from this exact authenticated authority observation."""
        manifest = self._entry_manifest(entry, allow_tombstone=True)
        return self._entry_info(entry, manifest)

    def _entry_info(self, entry: EntrySnapshot, manifest: BlobManifest) -> BlobEntry:
        """Render an already authenticated manifest without another observation."""
        return BlobEntry(
            entry.key,
            entry.generation,
            entry.locator,
            entry.expectation,
            self.store._manifest_entry_data(manifest),
        )

    def get_entry_info(self, key: str) -> BlobEntry | None:
        entry = self.authority.read_entry(key)
        if entry is None:
            return None
        manifest = self._entry_manifest(entry, allow_tombstone=True)
        return None if manifest.state == "tombstoned" else self._entry_info(entry, manifest)

    @contextmanager
    def open_entry(self, key: str):
        """Acquire once; never catch/retry exceptions raised by the caller."""
        resources = ExitStack()
        try:
            for attempt in range(2):
                entry = self.authority.read_entry(key)
                if entry is None:
                    yield None
                    return
                manifest = self._entry_manifest(entry, allow_tombstone=True)
                if manifest.state == "tombstoned":
                    yield None
                    return
                handler = self.store._resolve_payload_handler(manifest)
                try:
                    snapshot = resources.enter_context(
                        self.store._materialize_authority_store().open_snapshot(
                            manifest.locator, self.store._handler_metadata(manifest)
                        )
                    )
                except FileNotFoundError as exc:
                    resources.close()
                    if attempt == 0:
                        continue
                    raise CacheBlobPayloadMissingError(
                        "Canonical BlobStore payload is missing"
                    ) from exc
                second = self.authority.read_entry(key)
                if second is None or second.expectation != entry.expectation:
                    resources.close()
                    if attempt == 0:
                        continue
                    raise CacheBlobLifecycleConflictError(
                        "Authority changed during both read attempts",
                        context={"key": key, "operation": "read"},
                    )
                second_manifest = self._entry_manifest(second, allow_tombstone=True)
                if second_manifest.state != "committed":
                    resources.close()
                    if attempt == 0:
                        continue
                    raise CacheBlobLifecycleConflictError("Entry is no longer committed")
                digest, byte_size = sha256_and_size(snapshot.path)
                if digest != manifest.digest or byte_size != manifest.byte_size:
                    raise CacheBlobPayloadTamperedError(
                        "Canonical BlobStore payload integrity check failed"
                    )
                result = BlobEntry(
                    entry.key,
                    entry.generation,
                    entry.locator,
                    entry.expectation,
                    self.store._manifest_entry_data(manifest),
                    lambda: handler.get(snapshot.path, snapshot.metadata),
                )
                try:
                    yield result
                finally:
                    result._release()
                return
            raise CacheBlobLifecycleConflictError(
                "Authority read exhausted its bounded generation retry",
                context={"key": key, "operation": "read"},
            )
        finally:
            resources.close()

    def get(self, key: str) -> Any | None:
        with self.open_entry(key) as entry:
            return None if entry is None else entry.read()

    def exists(self, key: str) -> bool:
        with self.open_entry(key) as entry:
            return entry is not None

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        entry = self.authority.read_entry(key)
        if entry is None:
            return None
        manifest = self._entry_manifest(entry, allow_tombstone=True)
        return None if manifest.state == "tombstoned" else self.store._manifest_entry_data(manifest)

    def list(self, prefix: str | None = None) -> list[str]:
        if (
            self.store.topology.qualified_profile.requirements.coordination_scope
            == "multiple_hosts"
        ):
            raise CacheBlobBackendError(
                "Remote BlobStore listing requires list_page with an authority cursor",
                context={"operation": "list", "stage": "catalog_page"},
            )
        keys: list[str] = []
        for entry in self.authority.list_entries():
            manifest = self._entry_manifest(entry, allow_tombstone=True)
            if manifest.state != "committed" or (prefix and not manifest.key.startswith(prefix)):
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
