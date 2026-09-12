"""Contracts for explicit include-all rebuild planning and confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cacheness.error_handling import (
    CacheBlobMigrationOfflineDecisionRequiredError,
    CacheBlobMigrationEvidenceMismatchError,
    CacheBlobMigrationPlanStaleError,
    CacheBlobPayloadTamperedError,
)
from cacheness.storage import BackendRef, BlobStore, StoreTopology
from cacheness.storage.catalog import CatalogField, CatalogQuery, CatalogSchema
from cacheness.storage.migration import (
    MigrationCompatibilityEdge,
    MigrationPlanKind,
    MigrationPlanState,
    OfflineMigrationService,
    RebuildExclusion,
)
from cacheness.storage.migration_evidence import (
    MaintenanceEvidenceState,
    MaintenanceRunEvidence,
)
from cacheness.storage.projections import ProjectionController


class _SharedMemoryKeyProvider:
    """Provide one test-only signing identity for both offline participants."""

    def get_key(self) -> bytes:
        return b"r" * 32


@dataclass(frozen=True)
class _McapValue:
    """Small third-party native value used to exercise store-local handlers."""

    value: str


class _McapHandler:
    """A registered MCAP-like contract with observable source deserialization."""

    def __init__(self, *, fail_for: str | None = None) -> None:
        self.fail_for = fail_for
        self.read_calls = 0

    @property
    def data_type(self) -> str:
        return "mcap"

    @property
    def payload_format(self) -> str:
        return "mcap"

    @property
    def payload_format_version(self) -> int:
        return 1

    def supports_payload_contract(
        self, payload_format: str, payload_format_version: int
    ) -> bool:
        return (payload_format, payload_format_version) == ("mcap", 1)

    def payload_transformation_edges(self) -> tuple:
        return ()

    def can_handle(self, data, config=None) -> bool:
        del config
        return isinstance(data, _McapValue)

    def put(self, data: _McapValue, file_path: Path, config):
        del config
        if data.value == self.fail_for:
            raise OSError("destination handler write failed")
        payload_path = file_path.with_suffix(".mcap")
        payload_path.write_text(data.value, encoding="utf-8")
        return {
            "actual_path": str(payload_path),
            "file_size": len(data.value.encode("utf-8")),
            "payload_format": "mcap",
            "payload_format_version": 1,
        }

    def get(self, file_path: Path, metadata):
        del metadata
        self.read_calls += 1
        return _McapValue(file_path.read_text(encoding="utf-8"))

    def get_file_extension(self, config) -> str:
        del config
        return ".mcap"


def _store(
    root: Path,
    provider: _SharedMemoryKeyProvider,
    handler: _McapHandler | None = None,
    projections: tuple[object, ...] = (),
) -> BlobStore:
    """Create an initialized same-process store for rebuild-plan contracts."""

    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
            projections=projections,
        ),
        cache_dir=root,
        manifest_key_provider=provider,
    )
    if handler is not None:
        store.handlers.register_handler(handler, priority=0)
    store.initialize()
    return store


def _service(
    source: BlobStore,
    destination: BlobStore,
    work_directory: Path,
    *,
    run_id: str = "rebuild-run",
) -> OfflineMigrationService:
    """Build one explicitly offline service without a migration fallback."""

    return OfflineMigrationService(
        source=source,
        destination=destination,
        work_directory=work_directory,
        run_id=run_id,
        stopped_workers_acknowledged=True,
        compatibility_edges=(MigrationCompatibilityEdge.current_to_current_for_test(),),
    )


class _SimulatedProcessLoss(BaseException):
    """Model a process boundary that skips in-process failure cleanup."""


class _ProjectionCandidate:
    """Separate derived sink used only by explicit offline projection rebuild."""

    def __init__(self, owner: "_CountingProjectionSink") -> None:
        self._owner = owner

    def apply_projection_batch(self, _batch: object) -> None:
        self._owner.attempts += 1

    def save_projection_checkpoint(self, _checkpoint: object) -> None:
        return None

    def load_projection_checkpoint(self) -> None:
        return None


class _CountingProjectionSink:
    """Record derived work separately from canonical rebuild publication."""

    projection_name = "rebuild-derived"
    projection_query = CatalogQuery()
    projection_schema = CatalogSchema(
        (CatalogField("kind", "string"),), schema_id="rebuild-derived"
    )
    topology_capabilities = {"projection_rebuild": True, "offline_rebuild": True}

    def __init__(self) -> None:
        self.attempts = 0

    def apply_projection_batch(self, _batch: object) -> None:
        self.attempts += 1

    def save_projection_checkpoint(self, _checkpoint: object) -> None:
        return None

    def load_projection_checkpoint(self) -> None:
        return None

    def begin_isolated_rebuild(self) -> _ProjectionCandidate:
        return _ProjectionCandidate(self)

    def publish_isolated_rebuild(
        self, _candidate: _ProjectionCandidate, _checkpoint: object
    ) -> None:
        return None

    def discard_isolated_rebuild(self, _candidate: _ProjectionCandidate) -> None:
        return None


def test_rebuild_plan_defaults_to_every_entry_and_cannot_use_migration_stage(
    tmp_path: Path,
) -> None:
    """A separately created rebuild plan includes the complete source by default."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "first"}, key="first")
        source.put_entry({"value": "second"}, key="second")
        service = _service(source, destination, tmp_path / "maintenance")

        plan = service.create_rebuild_plan(service.inspect())

        assert plan.plan_kind is MigrationPlanKind.REBUILD
        assert plan.state is MigrationPlanState.PLANNED
        assert plan.exclusions == ()
        assert tuple(item.entry.key for item in plan.included_entries) == ("first", "second")
        assert plan.excluded_entries == ()
        assert plan.included_totals.total_entries == 2
        assert plan.excluded_totals.total_entries == 0
        with pytest.raises(CacheBlobMigrationEvidenceMismatchError, match="rebuild"):
            service.stage(plan)
        assert destination.get("first") is None
    finally:
        source.close()
        destination.close()


def test_rebuild_exclusion_regenerates_exact_plan_and_requires_its_confirmation(
    tmp_path: Path,
) -> None:
    """An omission is accepted only through a newly bound exact rebuild plan."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "retain"}, key="retain")
        source.put_entry({"value": "omit"}, key="omit")
        service = _service(source, destination, tmp_path / "maintenance")
        inspection = service.inspect()
        include_all = service.create_rebuild_plan(inspection)
        excluded = service.create_rebuild_plan(
            inspection,
            exclusions=(
                RebuildExclusion.exact_keys(
                    ("omit",), reason="operator_confirmed_omission"
                ),
            ),
        )

        assert excluded.plan_id != include_all.plan_id
        assert excluded.digest != include_all.digest
        assert tuple(item.entry.key for item in excluded.included_entries) == ("retain",)
        assert tuple(item.entry.key for item in excluded.excluded_entries) == ("omit",)
        assert excluded.excluded_totals.total_bytes > 0
        assert excluded.exclusions[0]["reason"] == "operator_confirmed_omission"
        assert excluded.exclusions[0]["keys"] == ("omit",)

        with pytest.raises(CacheBlobMigrationEvidenceMismatchError, match="confirmation"):
            service.confirm_rebuild(
                excluded,
                confirmation=service.rebuild_confirmation(include_all),
            )
        assert service.read_evidence().state is MaintenanceEvidenceState.INSPECTED

        service.confirm_rebuild(
            excluded,
            confirmation=service.rebuild_confirmation(excluded),
        )
        evidence = service.read_evidence()
        assert evidence.state is MaintenanceEvidenceState.PLANNED
        assert evidence.plan_digest == excluded.digest
        assert evidence.acknowledgement.plan_digest == excluded.digest
        assert evidence.completed_output_digests["confirm_rebuild"] == service.rebuild_confirmation(
            excluded
        )
        assert destination.get("retain") is None
    finally:
        source.close()
        destination.close()


def test_rebuild_confirmation_rejects_source_revision_drift_before_payload_io(
    tmp_path: Path,
) -> None:
    """An inspected rebuild cannot be confirmed after its source has changed."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "before"}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        source.put_entry({"value": "after"}, key="entry")

        with pytest.raises(CacheBlobMigrationPlanStaleError):
            service.confirm_rebuild(
                plan,
                confirmation=service.rebuild_confirmation(plan),
            )
        assert destination.get("entry") is None
    finally:
        source.close()
        destination.close()


def test_rebuild_uses_registered_source_handler_and_destination_blobstore_lifecycle(
    tmp_path: Path,
) -> None:
    """Exact custom contracts rebuild through verified reads and normal destination puts."""

    provider = _SharedMemoryKeyProvider()
    source_handler = _McapHandler()
    destination_handler = _McapHandler()
    source = _store(tmp_path / "source", provider, source_handler)
    destination = _store(tmp_path / "destination", provider, destination_handler)
    try:
        source.put_entry(
            _McapValue("preserved"),
            key="mcap-entry",
            catalog_values={"unknown_authenticated_attribute": "kept"},
        )
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))

        staged = service.stage_rebuild(plan)
        assert staged.state is MaintenanceEvidenceState.REBUILD_STAGED
        assert source_handler.read_calls == 1
        verified = service.verify_rebuild(plan)
        assert verified.state is MaintenanceEvidenceState.REBUILD_VERIFIED
        accepted = service.accept_rebuild(plan)
        assert accepted.state is MaintenanceEvidenceState.REBUILD_ACCEPTED

        entry = destination.get_entry_info("mcap-entry")
        assert entry is not None
        assert entry.metadata["catalog"]["values"] == {
            "unknown_authenticated_attribute": "kept"
        }
        assert destination.get("mcap-entry") == _McapValue("preserved")
        assert source.get("mcap-entry") == _McapValue("preserved")
    finally:
        source.close()
        destination.close()


def test_rebuild_integrity_failure_runs_before_custom_handler_and_discards_candidates(
    tmp_path: Path,
) -> None:
    """Tampered source bytes never reach the custom reader or a destination candidate."""

    provider = _SharedMemoryKeyProvider()
    source_handler = _McapHandler()
    source = _store(tmp_path / "source", provider, source_handler)
    destination = _store(tmp_path / "destination", provider, _McapHandler())
    try:
        source.put_entry(_McapValue("trusted"), key="mcap-entry")
        snapshot = source.lifecycle_authority.read_entry("mcap-entry")
        assert snapshot is not None
        source.payload_backend._storage[f"memory://{snapshot.locator}"] = b"tampered"
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))

        with pytest.raises(CacheBlobPayloadTamperedError):
            service.stage_rebuild(plan)

        assert source_handler.read_calls == 0
        assert destination.get("mcap-entry") is None
        assert service.read_evidence().state is MaintenanceEvidenceState.ABORTED
    finally:
        source.close()
        destination.close()


def test_rebuild_discards_only_its_run_owned_destination_entries_after_batch_failure(
    tmp_path: Path,
) -> None:
    """A failed write cleans up prior run-owned candidates without deleting source data."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider, _McapHandler())
    destination = _store(
        tmp_path / "destination", provider, _McapHandler(fail_for="second")
    )
    try:
        source.put_entry(_McapValue("first"), key="first")
        source.put_entry(_McapValue("second"), key="second")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))

        with pytest.raises(OSError, match="destination handler write failed"):
            service.stage_rebuild(plan)

        assert destination.get("first") is None
        assert destination.get("second") is None
        assert source.get("first") == _McapValue("first")
        assert source.get("second") == _McapValue("second")
        assert service.read_evidence().state is MaintenanceEvidenceState.ABORTED
    finally:
        source.close()
        destination.close()


def test_rebuild_response_loss_rederives_operation_id_and_replays_authority_receipt(
    tmp_path: Path,
) -> None:
    """Rebuild progress is durably bound to one canonical lifecycle receipt."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "recover"}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))

        def lose_process(boundary: str) -> None:
            if boundary == "rebuild.destination_committed_before_receipt_checkpoint":
                raise _SimulatedProcessLoss("response lost after canonical rebuild commit")

        service._rebuild_fault_hook = lose_process
        with pytest.raises(_SimulatedProcessLoss, match="response lost"):
            service.stage_rebuild(plan)

        operation_id = service._rebuild_operation_id(plan, plan.included_entries[0], 0)
        committed = destination.lifecycle_authority.read_mutation(operation_id)
        assert committed is not None and committed.state == "promoted"
        assert service.read_evidence().rebuild_receipt_batches == ()

        restarted = _service(source, destination, tmp_path / "maintenance")
        staged = restarted.resume(
            plan, run_id=restarted.run_id, evidence_path=restarted.evidence_path
        )
        assert staged.state is MaintenanceEvidenceState.REBUILD_STAGED
        receipt = restarted.read_evidence().rebuild_receipt_batches[0].receipts[0]
        assert receipt.key == "entry"
        assert receipt.operation_id == operation_id
        assert committed.promotion is not None
        assert receipt.generation == committed.promotion.entry.generation
        assert receipt.projections == {}
    finally:
        source.close()
        destination.close()


def test_projection_equipped_rebuild_replays_canonical_receipt_without_preacceptance_or_duplicate_derived_work(
    tmp_path: Path,
) -> None:
    """Canonical rebuild staging never runs a derived projection before acceptance."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    projection = _CountingProjectionSink()
    destination = _store(
        tmp_path / "destination", provider, projections=(projection,)
    )
    try:
        source.put_entry({"value": "canonical-only"}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))

        service.stage_rebuild(plan)
        evidence = service.read_evidence()
        receipt = evidence.rebuild_receipt_batches[0].receipts[0]
        assert receipt.projections == {}
        assert projection.attempts == 0

        service.verify_rebuild(plan)
        service.accept_rebuild(plan)
        assert projection.attempts == 0

        result = service.rebuild_projection(
            plan,
            ProjectionController(
                destination,
                projection,
                query=projection.projection_query,
                schema=projection.projection_schema,
            ),
        )
        assert result.exhausted is True
        assert projection.attempts == 1
    finally:
        source.close()
        destination.close()


def test_rebuild_checkpoints_exact_destination_receipts_and_resumes_each_rebuild_state(
    tmp_path: Path,
) -> None:
    """Every durable rebuild state resumes only from authenticated exact receipts."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "first"}, key="first")
        source.put_entry({"value": "second"}, key="second")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))

        def lose_after_first_receipt(boundary: str) -> None:
            if boundary == "rebuild.destination_committed_before_receipt_checkpoint":
                raise _SimulatedProcessLoss("restart while rebuilding")

        service._rebuild_fault_hook = lose_after_first_receipt
        with pytest.raises(_SimulatedProcessLoss):
            service.stage_rebuild(plan)
        assert service.read_evidence().state is MaintenanceEvidenceState.REBUILDING

        restarting = _service(source, destination, tmp_path / "maintenance")
        assert restarting.resume(
            plan, run_id=restarting.run_id, evidence_path=restarting.evidence_path
        ).state is MaintenanceEvidenceState.REBUILD_STAGED
        staged_evidence = restarting.read_evidence()
        assert len(staged_evidence.rebuild_receipt_batches) == 2
        assert all(
            receipt.operation_id == restarting._rebuild_operation_id(
                plan, plan.included_entries[index], index
            )
            for index, receipt in enumerate(
                restarting._flatten_rebuild_receipts(staged_evidence)
            )
        )

        staged_source = _store(tmp_path / "staged-resume-source", provider)
        staged_destination = _store(tmp_path / "staged-resume-destination", provider)
        try:
            staged_source.put_entry({"value": "staged"}, key="staged")
            staged_service = _service(
                staged_source, staged_destination, tmp_path / "staged-resume-maintenance"
            )
            staged_plan = staged_service.create_rebuild_plan(staged_service.inspect())
            staged_service.confirm_rebuild(
                staged_plan, confirmation=staged_service.rebuild_confirmation(staged_plan)
            )
            assert staged_service.stage_rebuild(staged_plan).state is MaintenanceEvidenceState.REBUILD_STAGED

            staged_restart = _service(
                staged_source, staged_destination, tmp_path / "staged-resume-maintenance"
            )
            assert staged_restart.resume(
                staged_plan,
                run_id=staged_restart.run_id,
                evidence_path=staged_restart.evidence_path,
            ).state is MaintenanceEvidenceState.REBUILD_VERIFIED
        finally:
            staged_source.close()
            staged_destination.close()

        verifying = restarting._write_evidence(
            restarting._new_evidence(
                state=MaintenanceEvidenceState.REBUILD_VERIFYING,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=staged_evidence.completed_steps,
                authority_receipts=staged_evidence.authority_receipts,
                cleanup_debt=staged_evidence.cleanup_debt,
                **restarting._rebuild_progress(staged_evidence),
                completed_output_digests=staged_evidence.completed_output_digests,
            )
        )
        assert verifying.state is MaintenanceEvidenceState.REBUILD_VERIFYING

        assert restarting.resume(
            plan, run_id=restarting.run_id, evidence_path=restarting.evidence_path
        ).state is MaintenanceEvidenceState.REBUILD_VERIFIED
        assert restarting.resume(
            plan, run_id=restarting.run_id, evidence_path=restarting.evidence_path
        ).state is MaintenanceEvidenceState.REBUILD_VERIFIED
    finally:
        source.close()
        destination.close()


def test_rebuild_verification_failure_cleans_only_exact_receipts_and_persists_debt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changed ownership is retained and recorded as retryable exact cleanup debt."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "first"}, key="first")
        source.put_entry({"value": "second"}, key="second")
        source.put_entry({"value": "third"}, key="third")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))
        original_stage_entry = service._stage_rebuild_entry
        original_delete = destination.delete

        def conflict_then_fail(rebuild_plan, assessment, entry_ordinal):
            if entry_ordinal == 2:
                destination.put_entry({"value": "new-owner"}, key="first")
                raise OSError("verification staging failed")
            return original_stage_entry(rebuild_plan, assessment, entry_ordinal)

        def fail_exact_second_delete(key, *, expected=None):
            if key == "second":
                raise OSError("destination cleanup unavailable")
            return original_delete(key, expected=expected)

        monkeypatch.setattr(service, "_stage_rebuild_entry", conflict_then_fail)
        monkeypatch.setattr(destination, "delete", fail_exact_second_delete)
        with pytest.raises(OSError, match="verification staging failed"):
            service.stage_rebuild(plan)

        evidence = service.read_evidence()
        first, second = tuple(
            receipt
            for batch in evidence.rebuild_receipt_batches
            for receipt in batch.receipts
        )
        assert evidence.state is MaintenanceEvidenceState.REBUILDING
        assert evidence.retired_rebuild_operation_ids == (first.operation_id,)
        assert evidence.cleanup_debt == (
            f"rebuild:{second.operation_id}:second:{second.generation}:{second.locator}",
        )
        assert destination.get("first") == {"value": "new-owner"}
        assert destination.get("second") == {"value": "second"}
        assert source.get("first") == {"value": "first"}
        assert source.get("second") == {"value": "second"}
        assert source.get("third") == {"value": "third"}
    finally:
        source.close()
        destination.close()


def test_rebuild_cleanup_debt_stays_resumable_until_exact_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response-loss cleanup debt settles only through the recorded receipt."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "first"}, key="first")
        source.put_entry({"value": "second"}, key="second")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))
        original_stage_entry = service._stage_rebuild_entry
        original_delete = destination.delete

        def stage_then_fail(rebuild_plan, assessment, entry_ordinal):
            if entry_ordinal == 1:
                raise OSError("rebuild staging failed")
            return original_stage_entry(rebuild_plan, assessment, entry_ordinal)

        def delete_then_lose_response(key, *, expected=None):
            original_delete(key, expected=expected)
            raise OSError("cleanup delete response lost")

        monkeypatch.setattr(service, "_stage_rebuild_entry", stage_then_fail)
        monkeypatch.setattr(destination, "delete", delete_then_lose_response)
        with pytest.raises(OSError, match="rebuild staging failed"):
            service.stage_rebuild(plan)

        evidence = service.read_evidence()
        receipt = evidence.rebuild_receipt_batches[0].receipts[0]
        assert evidence.state is MaintenanceEvidenceState.REBUILDING
        assert evidence.retired_rebuild_operation_ids == ()
        assert evidence.cleanup_debt == (service._rebuild_cleanup_debt(receipt),)
        assert destination.get_entry_info("first") is None

        monkeypatch.setattr(destination, "delete", original_delete)
        resumed = service.resume(
            plan, run_id=service.run_id, evidence_path=service.evidence_path
        )

        settled = service.read_evidence()
        assert resumed.state is MaintenanceEvidenceState.ABORTED
        assert settled.state is MaintenanceEvidenceState.ABORTED
        assert settled.cleanup_debt == ()
        assert settled.retired_rebuild_operation_ids == (receipt.operation_id,)
        assert destination.get("first") is None
        assert source.get("first") == {"value": "first"}
        assert source.get("second") == {"value": "second"}
    finally:
        source.close()
        destination.close()


def test_rebuild_evidence_rejects_terminal_aborted_cleanup_debt(
    tmp_path: Path,
) -> None:
    """Terminal rebuild evidence cannot stand in for unsettled cleanup debt."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "entry"}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))
        service.stage_rebuild(plan)
        evidence = service.read_evidence()
        receipt = evidence.rebuild_receipt_batches[0].receipts[0]

        with pytest.raises(ValueError, match="ABORTED rebuild evidence"):
            service._new_evidence(
                state=MaintenanceEvidenceState.ABORTED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=(*evidence.completed_steps, "abort_rebuild"),
                authority_receipts=evidence.authority_receipts,
                cleanup_debt=(service._rebuild_cleanup_debt(receipt),),
                **service._rebuild_progress(evidence),
                completed_output_digests=evidence.completed_output_digests,
            )
    finally:
        source.close()
        destination.close()


@pytest.mark.parametrize("component", ("operation", "key", "generation", "locator"))
def test_rebuild_cleanup_retry_rejects_forged_debt_without_payload_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    """A signed-looking debt string cannot broaden cleanup ownership."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "first"}, key="first")
        source.put_entry({"value": "second"}, key="second")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))
        original_stage_entry = service._stage_rebuild_entry

        def stage_then_fail(rebuild_plan, assessment, entry_ordinal):
            if entry_ordinal == 1:
                raise OSError("rebuild staging failed")
            return original_stage_entry(rebuild_plan, assessment, entry_ordinal)

        def fail_cleanup(*_args, **_kwargs):
            raise OSError("cleanup participant unavailable")

        monkeypatch.setattr(service, "_stage_rebuild_entry", stage_then_fail)
        monkeypatch.setattr(destination, "delete", fail_cleanup)
        with pytest.raises(OSError, match="rebuild staging failed"):
            service.stage_rebuild(plan)

        evidence = service.read_evidence()
        receipt = evidence.rebuild_receipt_batches[0].receipts[0]
        debt_parts = {
            "operation": ("f" * 64, receipt.key, receipt.generation, receipt.locator),
            "key": (receipt.operation_id, "forged-key", receipt.generation, receipt.locator),
            "generation": (
                receipt.operation_id,
                receipt.key,
                "forged-generation",
                receipt.locator,
            ),
            "locator": (
                receipt.operation_id,
                receipt.key,
                receipt.generation,
                "forged-locator",
            ),
        }
        operation_id, key, generation, locator = debt_parts[component]
        forged_debt = f"rebuild:{operation_id}:{key}:{generation}:{locator}"
        service._write_evidence(
            service._new_evidence(
                state=evidence.state,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=evidence.completed_steps,
                authority_receipts=evidence.authority_receipts,
                cleanup_debt=(forged_debt,),
                **service._rebuild_progress(evidence),
                completed_output_digests=evidence.completed_output_digests,
            )
        )

        participant_accesses: list[str] = []

        def unexpected_participant_access(*_args, **_kwargs):
            participant_accesses.append(component)
            raise AssertionError("forged debt must fail before participant access")

        monkeypatch.setattr(destination, "get_entry_info", unexpected_participant_access)
        monkeypatch.setattr(destination, "delete", unexpected_participant_access)
        monkeypatch.setattr(
            destination, "delete_migration_payload", unexpected_participant_access
        )
        monkeypatch.setattr(destination, "list", unexpected_participant_access)
        monkeypatch.setattr(destination, "list_page", unexpected_participant_access)

        with pytest.raises(
            CacheBlobMigrationEvidenceMismatchError,
            match="exact authenticated receipt",
        ):
            service.resume(plan, run_id=service.run_id, evidence_path=service.evidence_path)
        assert participant_accesses == []
        assert source.get("first") == {"value": "first"}
        assert source.get("second") == {"value": "second"}
    finally:
        source.close()
        destination.close()


def test_rebuild_cleanup_retry_preserves_changed_current_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Receipt settlement removes an old locator without deleting a later owner."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "first"}, key="first")
        source.put_entry({"value": "second"}, key="second")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))
        original_stage_entry = service._stage_rebuild_entry
        original_delete_payload = destination.delete_migration_payload

        def replace_owner_then_fail(rebuild_plan, assessment, entry_ordinal):
            if entry_ordinal == 1:
                destination.put_entry({"value": "new-owner"}, key="first")
                raise OSError("rebuild staging failed")
            return original_stage_entry(rebuild_plan, assessment, entry_ordinal)

        def fail_old_locator_cleanup(_locator: str) -> None:
            raise OSError("retired payload cleanup unavailable")

        monkeypatch.setattr(service, "_stage_rebuild_entry", replace_owner_then_fail)
        monkeypatch.setattr(
            destination, "delete_migration_payload", fail_old_locator_cleanup
        )
        with pytest.raises(OSError, match="rebuild staging failed"):
            service.stage_rebuild(plan)

        evidence = service.read_evidence()
        receipt = evidence.rebuild_receipt_batches[0].receipts[0]
        assert evidence.cleanup_debt == (service._rebuild_cleanup_debt(receipt),)
        assert destination.get("first") == {"value": "new-owner"}

        settled_locators: list[str] = []

        def delete_only_recorded_locator(locator: str) -> None:
            settled_locators.append(locator)
            original_delete_payload(locator)

        def current_key_delete_is_forbidden(*_args, **_kwargs):
            raise AssertionError("settlement must not delete the current key owner")

        monkeypatch.setattr(
            destination, "delete_migration_payload", delete_only_recorded_locator
        )
        monkeypatch.setattr(destination, "delete", current_key_delete_is_forbidden)
        monkeypatch.setattr(destination, "list", current_key_delete_is_forbidden)
        monkeypatch.setattr(destination, "list_page", current_key_delete_is_forbidden)
        resumed = service.resume(
            plan, run_id=service.run_id, evidence_path=service.evidence_path
        )

        settled = service.read_evidence()
        assert resumed.state is MaintenanceEvidenceState.ABORTED
        assert settled.cleanup_debt == ()
        assert settled.retired_rebuild_operation_ids == (receipt.operation_id,)
        assert settled_locators == [receipt.locator]
        assert destination.get("first") == {"value": "new-owner"}
        assert source.get("first") == {"value": "first"}
        assert source.get("second") == {"value": "second"}
    finally:
        source.close()
        destination.close()


def test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only explicit resume may settle authentic nonterminal rebuild debt."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "first"}, key="first")
        source.put_entry({"value": "second"}, key="second")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))
        original_stage_entry = service._stage_rebuild_entry
        original_get_entry_info = destination.get_entry_info

        def stage_then_fail(rebuild_plan, assessment, entry_ordinal):
            if entry_ordinal == 1:
                raise OSError("rebuild staging failed")
            return original_stage_entry(rebuild_plan, assessment, entry_ordinal)

        def fail_cleanup(*_args, **_kwargs):
            raise OSError("cleanup participant unavailable")

        monkeypatch.setattr(service, "_stage_rebuild_entry", stage_then_fail)
        monkeypatch.setattr(destination, "delete", fail_cleanup)
        with pytest.raises(OSError, match="rebuild staging failed"):
            service.stage_rebuild(plan)

        evidence = service.read_evidence()
        receipt = evidence.rebuild_receipt_batches[0].receipts[0]
        assert evidence.state is MaintenanceEvidenceState.REBUILDING
        assert evidence.cleanup_debt == (service._rebuild_cleanup_debt(receipt),)
        original_entry = original_get_entry_info("first")
        assert original_entry is not None
        original_entry_identity = (
            original_entry.generation,
            original_entry.locator,
            original_entry.expectation,
        )
        original_authority_identity = service._destination_authority.identity_snapshot()
        original_source = (source.get("first"), source.get("second"))

        participant_accesses: list[str] = []

        def unexpected_participant_access(*_args, **_kwargs):
            participant_accesses.append("payload")
            raise AssertionError("forward methods must fence debt before payload access")

        def unexpected_revalidation(*_args, **_kwargs):
            participant_accesses.append("revalidation")
            raise AssertionError("forward methods must fence debt before revalidation")

        monkeypatch.setattr(
            service, "_revalidate_plan_source_state", unexpected_revalidation
        )
        monkeypatch.setattr(destination, "get_entry_info", unexpected_participant_access)
        monkeypatch.setattr(destination, "open_entry", unexpected_participant_access)
        monkeypatch.setattr(destination, "delete", unexpected_participant_access)
        monkeypatch.setattr(
            destination, "delete_migration_payload", unexpected_participant_access
        )

        def write_authenticated_debt(state: MaintenanceEvidenceState) -> None:
            nonlocal evidence
            evidence = service._write_evidence(
                service._new_evidence(
                    state=state,
                    plan_digest=plan.digest,
                    source_identity=plan.source_identity,
                    destination_identity=plan.destination_identity,
                    completed_steps=evidence.completed_steps,
                    authority_receipts=evidence.authority_receipts,
                    cleanup_debt=evidence.cleanup_debt,
                    **service._rebuild_progress(evidence),
                    completed_output_digests=evidence.completed_output_digests,
                )
            )

        for state, operation in (
            (MaintenanceEvidenceState.REBUILDING, service.stage_rebuild),
            (MaintenanceEvidenceState.REBUILD_STAGED, service.verify_rebuild),
            (MaintenanceEvidenceState.REBUILD_VERIFIED, service.accept_rebuild),
        ):
            if evidence.state is not state:
                if state is MaintenanceEvidenceState.REBUILD_VERIFIED:
                    write_authenticated_debt(MaintenanceEvidenceState.REBUILD_VERIFYING)
                write_authenticated_debt(state)
            original_bytes = service.evidence_path.read_bytes()
            participant_accesses.clear()

            with pytest.raises(
                CacheBlobMigrationOfflineDecisionRequiredError,
                match="resume",
            ):
                operation(plan)

            assert service.evidence_path.read_bytes() == original_bytes
            current_entry = original_get_entry_info("first")
            assert current_entry is not None
            assert (
                current_entry.generation,
                current_entry.locator,
                current_entry.expectation,
            ) == original_entry_identity
            assert service._destination_authority.identity_snapshot() == original_authority_identity
            assert (source.get("first"), source.get("second")) == original_source
            assert participant_accesses == []

        settlement_source = _store(tmp_path / "settlement-source", provider)
        settlement_destination = _store(tmp_path / "settlement-destination", provider)
        try:
            settlement_source.put_entry({"value": "first"}, key="first")
            settlement_source.put_entry({"value": "second"}, key="second")
            settlement_service = _service(
                settlement_source,
                settlement_destination,
                tmp_path / "settlement-maintenance",
                run_id="rebuild-settlement-run",
            )
            settlement_plan = settlement_service.create_rebuild_plan(
                settlement_service.inspect()
            )
            settlement_service.confirm_rebuild(
                settlement_plan,
                confirmation=settlement_service.rebuild_confirmation(settlement_plan),
            )
            settlement_stage_entry = settlement_service._stage_rebuild_entry
            settlement_delete = settlement_destination.delete

            def settlement_stage_then_fail(rebuild_plan, assessment, entry_ordinal):
                if entry_ordinal == 1:
                    raise OSError("rebuild staging failed")
                return settlement_stage_entry(rebuild_plan, assessment, entry_ordinal)

            with monkeypatch.context() as settlement_patch:
                settlement_patch.setattr(
                    settlement_service,
                    "_stage_rebuild_entry",
                    settlement_stage_then_fail,
                )
                settlement_patch.setattr(
                    settlement_destination,
                    "delete",
                    fail_cleanup,
                )
                with pytest.raises(OSError, match="rebuild staging failed"):
                    settlement_service.stage_rebuild(settlement_plan)

                settlement_evidence = settlement_service.read_evidence()
                settlement_receipt = settlement_evidence.rebuild_receipt_batches[0].receipts[0]
                assert settlement_evidence.state is MaintenanceEvidenceState.REBUILDING
                assert settlement_evidence.cleanup_debt == (
                    settlement_service._rebuild_cleanup_debt(settlement_receipt),
                )
                settlement_patch.setattr(
                    settlement_destination,
                    "delete",
                    settlement_delete,
                )
                resumed = settlement_service.resume(
                    settlement_plan,
                    run_id=settlement_service.run_id,
                    evidence_path=settlement_service.evidence_path,
                )

            settled = settlement_service.read_evidence()
            assert resumed.state is MaintenanceEvidenceState.ABORTED
            assert settled.cleanup_debt == ()
            assert settled.retired_rebuild_operation_ids == (
                settlement_receipt.operation_id,
            )
            assert settlement_destination.get_entry_info("first") is None
            assert settlement_source.get("first") == {"value": "first"}
            assert settlement_source.get("second") == {"value": "second"}
        finally:
            settlement_source.close()
            settlement_destination.close()
    finally:
        source.close()
        destination.close()


def test_rebuild_evidence_rejects_accepted_cleanup_debt(tmp_path: Path) -> None:
    """Accepted rebuild evidence cannot retain an authenticated cleanup receipt."""

    provider = _SharedMemoryKeyProvider()
    source = _store(tmp_path / "source", provider)
    destination = _store(tmp_path / "destination", provider)
    try:
        source.put_entry({"value": "entry"}, key="entry")
        service = _service(source, destination, tmp_path / "maintenance")
        plan = service.create_rebuild_plan(service.inspect())
        service.confirm_rebuild(plan, confirmation=service.rebuild_confirmation(plan))
        service.stage_rebuild(plan)
        evidence = service.read_evidence()
        receipt = evidence.rebuild_receipt_batches[0].receipts[0]
        cleanup_debt = (service._rebuild_cleanup_debt(receipt),)

        with pytest.raises(ValueError, match="REBUILD_ACCEPTED rebuild evidence"):
            service._new_evidence(
                state=MaintenanceEvidenceState.REBUILD_ACCEPTED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=evidence.completed_steps,
                authority_receipts=evidence.authority_receipts,
                cleanup_debt=cleanup_debt,
                **service._rebuild_progress(evidence),
                completed_output_digests=evidence.completed_output_digests,
            )

        record = evidence.to_record()
        record["state"] = MaintenanceEvidenceState.REBUILD_ACCEPTED.value
        record["cleanup_debt"] = list(cleanup_debt)
        with pytest.raises(ValueError, match="REBUILD_ACCEPTED rebuild evidence"):
            MaintenanceRunEvidence.from_record(record)
    finally:
        source.close()
        destination.close()
