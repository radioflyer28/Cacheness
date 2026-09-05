"""Authority-backed BlobStore mutation orchestration.

The authority is the only committed truth. Payload work stays outside every
authority transaction and native handler bytes are never wrapped.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
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
    PreparedMutation,
    VerificationProof,
)
from .manifest import BlobManifestV1


_TOMBSTONE_OPERATION_ID_FIELD = "_cacheness_tombstone_operation_id"


@dataclass(frozen=True)
class _LifecycleHookContext:
    """Non-durable test timing context retained across the scheduler removal."""

    key: str | None


class AuthorityLifecycleEngine:
    """Coordinate immutable native payloads through complete authority calls."""

    def __init__(self, store: Any, authority: LifecycleAuthority):
        self.store = store
        self.authority = authority
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
        return self.store._authenticated_authority_manifest(
            entry.manifest, allow_tombstone=allow_tombstone
        )

    def _sign(self, manifest: BlobManifestV1) -> BlobManifestV1:
        return manifest.with_signature(
            sign_hmac_sha256(
                manifest.signing_bytes(), self.store._authority_manifest_key()
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

    def put(self, data: Any, *, key: str, metadata: dict[str, Any] | None) -> str:
        """Prepare, publish, verify, promote, then reclaim exact old debt."""
        handler = self.store.handlers.get_handler(data)
        guarded_io = self.store._materialize_authority_store()
        with guarded_io.stage(handler, data, self.store.config) as staged:
            previous = self.authority.read_entry(key)
            expected = (
                previous.expectation if previous is not None else EntryExpectation.absent()
            )
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
                with guarded_io.open_snapshot(locator, dict(published.get("metadata", {}))) as snapshot:
                    digest, byte_size = sha256_and_size(snapshot.path)
                manifest = BlobManifestV1(
                    schema_version=1,
                    key=key,
                    generation=generation,
                    state="committed",
                    locator=locator.as_posix(),
                    handler_type=handler.data_type,
                    payload_format=str(
                        published.get(
                            "storage_format", getattr(handler, "payload_format", "native")
                        )
                    ),
                    payload_format_version=int(
                        getattr(handler, "payload_format_version", 1)
                    ),
                    digest_algorithm="sha256",
                    digest=digest,
                    byte_size=byte_size,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    handler_metadata=dict(published.get("metadata", {})),
                    user_metadata=dict(metadata or {}),
                )
                manifest = self._sign(manifest)
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
            except Exception:
                self._abort(prepared, candidate_persisted=candidate_persisted)
                raise
        self._reach("put.promoted", key=key)
        self._settle_debts(promoted.cleanup_debt)
        self._reach("put.cleanup_retired", key=key)
        return key

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
        removed = 0
        for entry in self.authority.list_entries():
            try:
                removed += int(self.delete(entry.key, expected=entry.expectation))
            except CacheBlobLifecycleConflictError:
                continue
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
