"""Contract coverage for bounded offline migration plan compatibility."""

from __future__ import annotations

import pytest

from cacheness.storage.migration import (
    CompatibilityDimension,
    CompatibilityIdentity,
    CompatibilityMatrix,
    CompatibilityOutcome,
    ReleaseWindow,
    VersionEdge,
)


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
