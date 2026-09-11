"""Contract coverage for bounded offline migration plan compatibility."""

from __future__ import annotations

import json

import pytest

from cacheness.storage.migration import (
    CompatibilityDimension,
    CompatibilityIdentity,
    MigrationCompatibilityEdge,
    CompatibilityMatrix,
    CompatibilityOutcome,
    MigrationDisposition,
    MigrationEntryAssessment,
    MigrationPlan,
    MigrationPlanKind,
    MigrationPlanState,
    MigrationReason,
    ReleaseWindow,
    VersionEdge,
    render_migration_report,
)
from cacheness.storage.migration_authority import AuthorityIdentitySnapshot, AuthorityInventoryEntry
from cacheness.storage.manifest import StoreVersionDimensions


def test_machine_plan_digest_binds_sensitive_catalog_and_manifest_without_serializing_them() -> None:
    """Plans bind authenticated source state without becoming a metadata export."""
    sentinel = "source-state-secret-sentinel"
    source_contract = _identity("release-2")
    compatibility = CompatibilityMatrix(
        ReleaseWindow(current_release="release-2", immediately_previous_release="release-1")
    ).classify(source_contract, source_contract)

    def build_plan(*, catalog_value: str, manifest_value: str) -> MigrationPlan:
        return MigrationPlan.create(
            run_id="confidential-plan",
            source_identity=AuthorityIdentitySnapshot("source-store", 4, "sqlite"),
            destination_identity=AuthorityIdentitySnapshot("destination-store", 2, "sqlite"),
            entries=(
                MigrationEntryAssessment(
                    entry=AuthorityInventoryEntry(
                        key="entry-a",
                        generation="generation-a",
                        locator="generations/private-locator",
                        manifest=json.dumps(
                            {
                                "credential": sentinel,
                                "handler_attribute": manifest_value,
                                "signing_material": sentinel,
                            },
                            sort_keys=True,
                        ).encode("utf-8"),
                        payload_digest="a" * 64,
                        byte_size=7,
                    ),
                    disposition=MigrationDisposition.MIGRATABLE,
                    reason=MigrationReason.COMPATIBLE_EDGE,
                    catalog_values={
                        "application_metadata": catalog_value,
                        "credential": sentinel,
                    },
                ),
            ),
            release_window=ReleaseWindow("release-2", "release-1"),
            compatibility=compatibility,
        )

    plan = build_plan(catalog_value=sentinel, manifest_value=sentinel)
    changed_catalog = build_plan(catalog_value="changed-catalog", manifest_value=sentinel)
    changed_manifest = build_plan(catalog_value=sentinel, manifest_value="changed-manifest")
    encoded = plan.to_canonical_bytes()
    record = json.loads(encoded)
    entry = record["entries"][0]

    assert plan.digest != changed_catalog.digest
    assert plan.digest != changed_manifest.digest
    assert sentinel.encode("utf-8") not in encoded
    assert "catalog_values" not in entry
    assert "manifest" not in entry["entry"]
    assert set(entry) == {
        "disposition",
        "entry",
        "reason",
        "source_catalog_digest",
        "source_manifest_digest",
    }
    assert set(entry["entry"]) == {"byte_size", "generation", "key", "payload_digest"}
    assert MigrationPlan.from_canonical_bytes(encoded).to_canonical_bytes() == encoded


def _identity(
    release: str,
    *,
    payload_version: int = 1,
    catalog_revision: int = 1,
) -> CompatibilityIdentity:
    """Build a wholly synthetic persisted-contract identity for one test release."""
    return CompatibilityIdentity(
        release=release,
        store_layout=("store-format", "2", "epoch", "1"),
        authority=("sqlite", "sqlite-lifecycle-authority", "8"),
        manifest_schema=("manifest", "3"),
        payload=("object", "compressed_pickle", str(payload_version)),
        catalog=("application-catalog", str(catalog_revision), "fingerprint"),
    )


def test_release_window_accepts_only_current_and_immediately_previous_release() -> None:
    """Older releases require successive declared migration steps."""
    window = ReleaseWindow(current_release="release-2", immediately_previous_release="release-1")

    assert window.supports_direct_source("release-2") is True
    assert window.supports_direct_source("release-1") is True
    assert window.supports_direct_source("release-0") is False
    assert window.supports_direct_target("release-2") is True
    assert window.supports_direct_target("release-1") is False


def test_matrix_requires_one_exact_edge_per_changed_dimension() -> None:
    """Changing payload and catalog contracts cannot infer compatibility from layout."""
    source = _identity("release-1")
    payload_target = _identity("release-2", payload_version=2)
    catalog_target = _identity("release-2", catalog_revision=2)
    window = ReleaseWindow(current_release="release-2", immediately_previous_release="release-1")
    payload_edge = VersionEdge(
        source_release="release-1",
        destination_release="release-2",
        dimension=CompatibilityDimension.PAYLOAD,
        source_value=source.value_for(CompatibilityDimension.PAYLOAD),
        destination_value=payload_target.value_for(CompatibilityDimension.PAYLOAD),
    )
    matrix = CompatibilityMatrix(window, edges=(payload_edge,))

    payload_result = matrix.classify(source, payload_target)
    catalog_result = matrix.classify(source, catalog_target)

    assert payload_result.outcome is CompatibilityOutcome.SUPPORTED
    assert payload_result.dimension_outcomes[CompatibilityDimension.PAYLOAD] is CompatibilityOutcome.SUPPORTED
    assert catalog_result.outcome is CompatibilityOutcome.REBUILD_ONLY
    assert catalog_result.dimension_outcomes[CompatibilityDimension.CATALOG] is CompatibilityOutcome.REBUILD_ONLY


def test_compatibility_edge_requires_exact_destination_dimensions() -> None:
    """A source match alone never authorizes a different configured target."""
    source = StoreVersionDimensions()
    destination = StoreVersionDimensions(payload_format_version=2)
    edge = MigrationCompatibilityEdge(
        source=source,
        destination=destination,
        name="object-v1-to-v2",
    )

    assert edge.supports(source, destination) is True
    assert edge.supports(source, StoreVersionDimensions()) is False
    assert edge.supports(
        StoreVersionDimensions(store_epoch=2), destination
    ) is False


def test_matrix_rejects_ambiguous_or_out_of_window_edges() -> None:
    """The registry has no wildcard, duplicate, or older-release direct jump."""
    source = _identity("release-1")
    target = _identity("release-2", payload_version=2)
    edge = VersionEdge(
        source_release="release-1",
        destination_release="release-2",
        dimension=CompatibilityDimension.PAYLOAD,
        source_value=source.value_for(CompatibilityDimension.PAYLOAD),
        destination_value=target.value_for(CompatibilityDimension.PAYLOAD),
    )
    window = ReleaseWindow(current_release="release-2", immediately_previous_release="release-1")

    with pytest.raises(ValueError, match="duplicate"):
        CompatibilityMatrix(window, edges=(edge, edge))

    with pytest.raises(ValueError, match="outside"):
        CompatibilityMatrix(
            window,
            edges=(
                VersionEdge(
                    source_release="release-0",
                    destination_release="release-2",
                    dimension=CompatibilityDimension.PAYLOAD,
                    source_value=edge.source_value,
                    destination_value=edge.destination_value,
                ),
            ),
        )


def test_canonical_plan_round_trips_and_human_report_uses_the_same_model() -> None:
    """Machine and human migration output retain one identity, count, and reason set."""
    source_contract = _identity("release-1")
    destination_contract = _identity("release-2", payload_version=2)
    edge = VersionEdge(
        source_release="release-1",
        destination_release="release-2",
        dimension=CompatibilityDimension.PAYLOAD,
        source_value=source_contract.value_for(CompatibilityDimension.PAYLOAD),
        destination_value=destination_contract.value_for(CompatibilityDimension.PAYLOAD),
    )
    compatibility = CompatibilityMatrix(
        ReleaseWindow(current_release="release-2", immediately_previous_release="release-1"),
        edges=(edge,),
    ).classify(source_contract, destination_contract)
    entry = AuthorityInventoryEntry(
        key="entry-a",
        generation="generation-a",
        locator="generations/opaque-a",
        manifest=b"{}",
        payload_digest="a" * 64,
        byte_size=7,
    )
    plan = MigrationPlan.create(
        run_id="bounded-plan",
        source_identity=AuthorityIdentitySnapshot("source-store", 4, "sqlite"),
        destination_identity=AuthorityIdentitySnapshot("destination-store", 2, "sqlite"),
        entries=(
            MigrationEntryAssessment(
                entry=entry,
                disposition=MigrationDisposition.MIGRATABLE,
                reason=MigrationReason.DIRECTED_EDGE,
                catalog_values={"unrecognized_application_field": "preserved"},
            ),
        ),
        release_window=ReleaseWindow("release-2", "release-1"),
        compatibility=compatibility,
    )

    encoded = plan.to_canonical_bytes()
    decoded = MigrationPlan.from_canonical_bytes(encoded)
    report = render_migration_report(plan)

    assert decoded.to_canonical_bytes() == encoded
    assert plan.plan_kind is MigrationPlanKind.MIGRATION
    assert plan.state is MigrationPlanState.PLANNED
    assert plan.totals.total_entries == 1
    assert plan.totals.total_bytes == 7
    assert plan.stopped_worker_acknowledgement_required is True
    assert plan.digest in report
    assert "source-store" in report
    assert "directed_edge" in report


def test_plan_decoder_rejects_duplicate_json_keys() -> None:
    """A plan parser never permits JSON's last-key-wins reinterpretation."""
    with pytest.raises(ValueError, match="canonical JSON"):
        MigrationPlan.from_canonical_bytes(b'{"plan_version":1,"plan_version":1}')
