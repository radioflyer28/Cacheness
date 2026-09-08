"""Bounded, derived-only catalog projection delivery.

Projection state is deliberately outside ``LifecycleAuthority``. A controller
copies authenticated, revision-bound catalog pages into a sink, records a
checkpoint only after apply, and reports any remaining work without changing a
committed BlobStore generation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from typing import Any

from .catalog import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, MAX_WORK_CAP, STORE_EPOCH
from .composition import (
    BackendRole,
    CapabilityRequirementError,
    ParticipantCapabilities,
    ProjectionSink,
)


class ProjectionCheckpointError(ValueError):
    """Raised when a derived checkpoint is malformed or belongs to another run."""


class ProjectionCapabilityError(CapabilityRequirementError):
    """Raised when an explicit projection operation lacks a declared capability."""


class ProjectionRebuildError(RuntimeError):
    """Raised when an isolated rebuild cannot complete or be safely published."""


class ProjectionStatus(str, Enum):
    """Truthful status of one derived participant after a canonical commit."""

    CURRENT = "current"
    DIRTY = "dirty"
    PARTIAL = "partial"


@dataclass(frozen=True)
class ProjectionCheckpoint:
    """A bounded resumption token bound to one canonical catalog snapshot."""

    source_store_id: str
    store_epoch: int
    schema_id: str
    schema_fingerprint: str
    query_fingerprint: str
    revision: int
    cursor: str | None

    def __post_init__(self) -> None:
        for field_name in (
            "source_store_id",
            "schema_id",
            "schema_fingerprint",
            "query_fingerprint",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 512:
                raise ProjectionCheckpointError(
                    f"Projection checkpoint {field_name} must be a bounded string"
                )
        for field_name, minimum in (("store_epoch", 1), ("revision", 0)):
            value = getattr(self, field_name)
            if type(value) is not int or value < minimum:
                raise ProjectionCheckpointError(
                    f"Projection checkpoint {field_name} is invalid"
                )
        if self.cursor is not None and (
            not isinstance(self.cursor, str)
            or not self.cursor
            or len(self.cursor.encode("utf-8")) > 16 * 1024
        ):
            raise ProjectionCheckpointError("Projection checkpoint cursor is invalid")

    def assert_matches(
        self,
        *,
        source_store_id: str,
        store_epoch: int,
        schema_id: str,
        schema_fingerprint: str,
        query_fingerprint: str,
        revision: int,
    ) -> None:
        """Reject a resume attempt that is not the original canonical snapshot."""
        expected = {
            "source_store_id": source_store_id,
            "store_epoch": store_epoch,
            "schema_id": schema_id,
            "schema_fingerprint": schema_fingerprint,
            "query_fingerprint": query_fingerprint,
            "revision": revision,
        }
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise ProjectionCheckpointError(
                    f"Projection checkpoint is stale for {field_name}"
                )


@dataclass(frozen=True)
class ProjectionBatch:
    """One bounded idempotent apply unit from a canonical catalog page."""

    checkpoint: ProjectionCheckpoint
    entries: tuple[Any, ...]
    next_cursor: str | None
    exhausted: bool
    batch_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or len(self.entries) > MAX_PAGE_SIZE:
            raise ProjectionCheckpointError("Projection batch entries exceed the portable bound")
        if self.next_cursor is not None and (
            not isinstance(self.next_cursor, str) or not self.next_cursor
        ):
            raise ProjectionCheckpointError("Projection batch cursor is invalid")
        if type(self.exhausted) is not bool:
            raise ProjectionCheckpointError("Projection batch exhaustion flag is invalid")
        if self.exhausted != (self.next_cursor is None):
            raise ProjectionCheckpointError("Projection batch cursor and exhaustion disagree")
        if not isinstance(self.batch_id, str) or len(self.batch_id) != 64:
            raise ProjectionCheckpointError("Projection batch identity is invalid")


@dataclass(frozen=True)
class ProjectionOutcome:
    """Immutable attribution for one projection after a canonical operation."""

    name: str
    status: ProjectionStatus
    checkpoint: ProjectionCheckpoint | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Projection outcome name must be a non-empty string")
        if not isinstance(self.status, ProjectionStatus):
            raise ValueError("Projection outcome status must be a ProjectionStatus")
        if self.error_type is not None and (
            not isinstance(self.error_type, str) or not self.error_type
        ):
            raise ValueError("Projection outcome error type must be a non-empty string")


@dataclass(frozen=True)
class ProjectionResult:
    """The result of explicit or best-effort derived work."""

    exhausted: bool
    checkpoint: ProjectionCheckpoint | None
    batches: int
    receipt: object | None = None
    projection_status: str = ProjectionStatus.CURRENT.value
    outcome: ProjectionOutcome | None = None


def validate_rebuild_mode(
    capabilities: Mapping[str, object] | ParticipantCapabilities | object,
    *,
    requested: str,
) -> None:
    """Require an advertised online/offline publication mode explicitly.

    A caller cannot infer online rebuilding from a projection's existence. In
    particular, a SQLite publication must declare ``offline_rebuild`` and is
    therefore never silently treated as live-worker safe.
    """
    if requested not in {"online", "offline"}:
        raise ProjectionCapabilityError("Projection rebuild mode must be online or offline")
    if isinstance(capabilities, Mapping):
        supported = capabilities.get(f"{requested}_rebuild", False)
        rebuild_supported = capabilities.get("projection_rebuild", True)
    else:
        supported = getattr(capabilities, f"{requested}_rebuild", False)
        rebuild_supported = getattr(capabilities, "projection_rebuild", True)
    if type(supported) is not bool or type(rebuild_supported) is not bool:
        raise ProjectionCapabilityError("Projection rebuild capabilities must be bools")
    if not rebuild_supported or not supported:
        raise ProjectionCapabilityError(
            f"Projection does not advertise {requested} rebuild publication"
        )


class ProjectionController:
    """Pull a bounded revision from canonical catalog pages into one sink.

    The controller uses the shared ``BackendRole.PROJECTION`` vocabulary and
    ``ProjectionSink`` protocol from composition. It owns no lifecycle state:
    source membership and cursor validation remain in ``BlobStore`` and its
    authority, while checkpoint persistence remains with the derived sink.
    """

    role = BackendRole.PROJECTION

    def __init__(
        self,
        source: object,
        sink: ProjectionSink | object,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        work_cap: int | None = None,
        query: object | None = None,
        schema: object | None = None,
        source_store_id: str | None = None,
        store_epoch: int = STORE_EPOCH,
        schema_id: str | None = None,
        schema_fingerprint: str | None = None,
        query_fingerprint: str | None = None,
        capabilities: Mapping[str, object] | object | None = None,
    ) -> None:
        if type(page_size) is not int or not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ProjectionCapabilityError("Projection page size is outside the portable bound")
        effective_work_cap = page_size if work_cap is None else work_cap
        if (
            type(effective_work_cap) is not int
            or effective_work_cap < page_size
            or effective_work_cap > MAX_WORK_CAP
        ):
            raise ProjectionCapabilityError("Projection work cap is outside the portable bound")
        if (query is None) != (schema is None):
            raise ProjectionCapabilityError(
                "Canonical projection pulls require query and schema together"
            )
        self.source = source
        self.sink = sink
        self.page_size = page_size
        self.work_cap = effective_work_cap
        self.query = query
        self.schema = schema
        self.source_store_id = source_store_id or _source_store_id(source)
        self.store_epoch = store_epoch
        self.schema_id = schema_id or getattr(schema, "schema_id", "catalog")
        self.schema_fingerprint = schema_fingerprint or getattr(
            schema, "fingerprint", self.schema_id
        )
        self.query_fingerprint = query_fingerprint or getattr(query, "fingerprint", "all")
        self.capabilities = capabilities
        self._active_checkpoint: ProjectionCheckpoint | None = None
        self._last_page_cursor: str | None = None

    @property
    def projection_name(self) -> str:
        """Return a stable participant name for immutable receipt outcomes."""
        name = getattr(self.sink, "projection_name", None)
        if isinstance(name, str) and name:
            return name
        return type(self.sink).__name__

    def pull(
        self, checkpoint: ProjectionCheckpoint | None = None
    ) -> ProjectionResult:
        """Apply a complete bounded canonical revision with checkpoint-after-apply."""
        active = checkpoint or self._load_checkpoint()
        self._active_checkpoint = active
        batches = 0
        while True:
            page = self._fetch_page(None if active is None else active.cursor)
            current = self._checkpoint_for_page(page, active)
            self._last_page_cursor = _page_cursor(page)
            batch = self._batch_for_page(page, current)
            self._apply_batch(batch)
            active = replace(current, cursor=batch.next_cursor)
            self._save_checkpoint(active)
            self._active_checkpoint = active
            batches += 1
            if batch.exhausted:
                return ProjectionResult(True, active, batches)

    def apply_page(self, page: object) -> ProjectionResult:
        """Apply one supplied bounded page for deterministic interruption tests."""
        current = self._checkpoint_for_page(page, self._active_checkpoint)
        self._last_page_cursor = _page_cursor(page)
        batch = self._batch_for_page(page, current)
        self._apply_batch(batch)
        next_checkpoint = replace(current, cursor=batch.next_cursor)
        self._save_checkpoint(next_checkpoint)
        self._active_checkpoint = next_checkpoint
        return ProjectionResult(batch.exhausted, next_checkpoint, 1)

    def best_effort(self, receipt: object) -> ProjectionResult:
        """Attempt derived work without changing the already committed receipt."""
        try:
            result = self.pull()
        except Exception as error:
            outcome = ProjectionOutcome(
                self.projection_name,
                ProjectionStatus.DIRTY,
                self._active_checkpoint,
                type(error).__name__,
            )
            return ProjectionResult(
                False,
                self._active_checkpoint,
                0,
                receipt=receipt,
                projection_status=ProjectionStatus.DIRTY.value,
                outcome=outcome,
            )
        outcome = ProjectionOutcome(
            self.projection_name, ProjectionStatus.CURRENT, result.checkpoint
        )
        return ProjectionResult(
            result.exhausted,
            result.checkpoint,
            result.batches,
            receipt=receipt,
            projection_status=ProjectionStatus.CURRENT.value,
            outcome=outcome,
        )

    def refresh(self, receipt: object) -> ProjectionResult:
        """Run requested derived work or report a committed-partial outcome.

        The typed error itself is defined at the public storage-error boundary
        in Task 2. Deferring its import keeps the projection controller free of
        a lifecycle-error dependency during ordinary best-effort work.
        """
        try:
            return self.pull()
        except Exception as error:
            from cacheness.error_handling import CacheBlobCommittedPartialError

            remaining_cursor = self._last_page_cursor
            if remaining_cursor is None and self._active_checkpoint is not None:
                remaining_cursor = self._active_checkpoint.cursor
            raise CacheBlobCommittedPartialError(
                "Canonical commit succeeded but requested projection refresh failed",
                receipt=receipt,
                remaining_cursor=remaining_cursor,
                projection_name=self.projection_name,
            ) from error

    def rebuild(self, *, requested: str = "offline") -> ProjectionResult:
        """Build an isolated destination and publish it only after completion."""
        capability_source = self.capabilities
        if capability_source is None:
            capability_source = getattr(self.sink, "topology_capabilities", None)
        try:
            validate_rebuild_mode(capability_source or {}, requested=requested)
        except ProjectionCapabilityError as error:
            raise ProjectionRebuildError("Projection rebuild is not capability-qualified") from error
        begin = getattr(self.sink, "begin_isolated_rebuild", None)
        publish = getattr(self.sink, "publish_isolated_rebuild", None)
        discard = getattr(self.sink, "discard_isolated_rebuild", None)
        if not callable(begin) or not callable(publish):
            raise ProjectionRebuildError(
                "Projection sink does not provide isolated rebuild publication"
            )
        candidate = begin()
        if candidate is self.sink:
            raise ProjectionRebuildError("Projection rebuild destination must be isolated")
        controller = ProjectionController(
            self.source,
            candidate,
            page_size=self.page_size,
            work_cap=self.work_cap,
            query=self.query,
            schema=self.schema,
            source_store_id=self.source_store_id,
            store_epoch=self.store_epoch,
            schema_id=self.schema_id,
            schema_fingerprint=self.schema_fingerprint,
            query_fingerprint=self.query_fingerprint,
            capabilities=self.capabilities,
        )
        try:
            result = controller.pull()
            if not result.exhausted:
                raise ProjectionRebuildError("Projection rebuild did not reach completion")
            publish(candidate, result.checkpoint)
            return result
        except Exception as error:
            if callable(discard):
                discard(candidate)
            if isinstance(error, ProjectionRebuildError):
                raise
            raise ProjectionRebuildError("Isolated projection rebuild failed") from error

    def _fetch_page(self, cursor: str | None) -> object:
        query_catalog = getattr(self.source, "query_catalog", None)
        if not callable(query_catalog):
            raise ProjectionCapabilityError("Projection source lacks canonical catalog pages")
        if self.query is None:
            return query_catalog(cursor)
        return query_catalog(
            self.query,
            schema=self.schema,
            cursor=cursor,
            limit=self.page_size,
            work_cap=self.work_cap,
        )

    def _checkpoint_for_page(
        self, page: object, active: ProjectionCheckpoint | None
    ) -> ProjectionCheckpoint:
        revision = _page_revision(page)
        if active is not None:
            active.assert_matches(
                source_store_id=self.source_store_id,
                store_epoch=self.store_epoch,
                schema_id=self.schema_id,
                schema_fingerprint=self.schema_fingerprint,
                query_fingerprint=self.query_fingerprint,
                revision=revision,
            )
            return active
        return ProjectionCheckpoint(
            source_store_id=self.source_store_id,
            store_epoch=self.store_epoch,
            schema_id=self.schema_id,
            schema_fingerprint=self.schema_fingerprint,
            query_fingerprint=self.query_fingerprint,
            revision=revision,
            cursor=None,
        )

    def _batch_for_page(self, page: object, checkpoint: ProjectionCheckpoint) -> ProjectionBatch:
        entries = _page_entries(page)
        if len(entries) > self.page_size:
            raise ProjectionCheckpointError("Canonical projection page exceeds its configured bound")
        cursor = _page_cursor(page)
        exhausted = _page_exhausted(page)
        batch_id = _batch_identity(checkpoint, entries, cursor, exhausted)
        return ProjectionBatch(checkpoint, entries, cursor, exhausted, batch_id)

    def _apply_batch(self, batch: ProjectionBatch) -> None:
        if _has_applied_batch(self.sink, batch.batch_id):
            return
        apply = getattr(self.sink, "apply_projection_batch", None)
        if callable(apply):
            apply(batch)
        else:
            apply = getattr(self.sink, "apply", None)
            if not callable(apply):
                raise ProjectionCapabilityError("Projection sink cannot apply a derived batch")
            apply(batch.entries)
        _mark_applied_batch(self.sink, batch.batch_id)

    def _load_checkpoint(self) -> ProjectionCheckpoint | None:
        load = getattr(self.sink, "load_projection_checkpoint", None)
        if not callable(load):
            return None
        checkpoint = load()
        if checkpoint is None:
            return None
        if not isinstance(checkpoint, ProjectionCheckpoint):
            raise ProjectionCheckpointError("Projection sink returned an invalid checkpoint")
        return checkpoint

    def _save_checkpoint(self, checkpoint: ProjectionCheckpoint) -> None:
        save = getattr(self.sink, "save_projection_checkpoint", None)
        if callable(save):
            save(checkpoint)
            return
        legacy = getattr(self.sink, "checkpoint", None)
        if not callable(legacy):
            raise ProjectionCapabilityError("Projection sink cannot persist a checkpoint")
        legacy(checkpoint.cursor)


CatalogProjection = ProjectionController
CommittedPartialProjectionError = None


def _source_store_id(source: object) -> str:
    configured = getattr(source, "projection_store_id", None)
    if isinstance(configured, str) and configured:
        return configured
    return f"{type(source).__module__}.{type(source).__qualname__}"


def _page_entries(page: object) -> tuple[Any, ...]:
    entries = getattr(page, "entries", None)
    if not isinstance(entries, tuple):
        raise ProjectionCheckpointError("Canonical projection page entries must be a tuple")
    return entries


def _page_revision(page: object) -> int:
    revision = getattr(page, "revision", None)
    if type(revision) is not int or revision < 0:
        raise ProjectionCheckpointError("Canonical projection page revision is invalid")
    return revision


def _page_cursor(page: object) -> str | None:
    cursor = getattr(page, "cursor", None)
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        raise ProjectionCheckpointError("Canonical projection page cursor is invalid")
    return cursor


def _page_exhausted(page: object) -> bool:
    exhausted = getattr(page, "exhausted", None)
    if type(exhausted) is not bool:
        raise ProjectionCheckpointError("Canonical projection page exhaustion flag is invalid")
    if exhausted != (_page_cursor(page) is None):
        raise ProjectionCheckpointError("Canonical projection page cursor and exhaustion disagree")
    return exhausted


def _batch_identity(
    checkpoint: ProjectionCheckpoint,
    entries: tuple[Any, ...],
    cursor: str | None,
    exhausted: bool,
) -> str:
    """Return a deterministic sink-local idempotency identity for one page."""
    identity = {
        "checkpoint": {
            "source_store_id": checkpoint.source_store_id,
            "store_epoch": checkpoint.store_epoch,
            "schema_id": checkpoint.schema_id,
            "schema_fingerprint": checkpoint.schema_fingerprint,
            "query_fingerprint": checkpoint.query_fingerprint,
            "revision": checkpoint.revision,
            "cursor": checkpoint.cursor,
        },
        "entries": [_entry_identity(entry) for entry in entries],
        "next_cursor": cursor,
        "exhausted": exhausted,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _entry_identity(entry: Any) -> object:
    if hasattr(entry, "key") and hasattr(entry, "generation"):
        return [getattr(entry, "key"), getattr(entry, "generation")]
    if isinstance(entry, tuple):
        return list(entry)
    return repr(entry)


def _has_applied_batch(sink: object, batch_id: str) -> bool:
    applied = getattr(sink, "_cacheness_projection_batches", None)
    return isinstance(applied, set) and batch_id in applied


def _mark_applied_batch(sink: object, batch_id: str) -> None:
    applied = getattr(sink, "_cacheness_projection_batches", None)
    if not isinstance(applied, set):
        applied = set()
        setattr(sink, "_cacheness_projection_batches", applied)
    applied.add(batch_id)


__all__ = [
    "CatalogProjection",
    "CommittedPartialProjectionError",
    "ProjectionBatch",
    "ProjectionCapabilityError",
    "ProjectionCheckpoint",
    "ProjectionCheckpointError",
    "ProjectionController",
    "ProjectionOutcome",
    "ProjectionRebuildError",
    "ProjectionResult",
    "ProjectionStatus",
    "validate_rebuild_mode",
]
