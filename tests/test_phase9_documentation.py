"""Contract tests for the Phase 9 adoption documentation surface."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
README = PROJECT_ROOT / "README.md"
DOCS_INDEX = PROJECT_ROOT / "docs" / "README.md"
QUALIFICATION_GUIDE = PROJECT_ROOT / "docs" / "RELEASE_QUALIFICATION.md"


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _section(source: str, heading: str) -> str:
    """Return a level-two Markdown section without its following peer."""
    start = source.index(heading)
    remainder = source[start + len(heading) :]
    next_heading = remainder.find("\n## ")
    return remainder if next_heading == -1 else remainder[:next_heading]


def test_readme_is_a_local_ready_blobstore_first_gateway() -> None:
    """The gateway contains exactly the two supported local quick starts."""
    source = README.read_text(encoding="utf-8")

    assert "local-ready development version" in source
    assert "NOT_PUBLISHED" in source
    assert "owns storage lifecycle" in source
    assert "adds cache policy" in source
    assert source.count("### Quick start:") == 2
    assert "### Quick start: store an object" in source
    assert "### Quick start: cache a function result" in source
    assert "from cacheness.storage import BackendRef, BlobStore, StoreTopology" in source
    assert "@cached(cache=cache)" in source

    retired_or_unqualified = (
        "SqlCache",
        "SQL pull-through",
        "cacheness()",
        "Windows",
        "PostgreSQL",
        "S3",
    )
    assert all(term not in source for term in retired_or_unqualified)


def test_base_checkout_install_is_frozen_and_has_no_extra_opt_in() -> None:
    """The primary install is the minimal checked-out repository workflow."""
    source = README.read_text(encoding="utf-8")
    installation = _section(source, "## Install from a checkout")

    assert "uv sync --frozen --no-default-groups" in installation
    assert "pip install cacheness" not in installation
    assert "--extra" not in installation
    assert "[recommended]" not in installation
    assert "[dataframes]" not in installation


def test_task_first_navigation_links_current_journeys_and_single_matrix() -> None:
    """Navigation leads with jobs to do instead of stale component history."""
    source = DOCS_INDEX.read_text(encoding="utf-8")

    assert source.index("## Store objects") < source.index("## Reference")
    for document in (
        "BLOB_STORE.md",
        "CACHE_POLICY.md",
        "PLUGIN_DEVELOPMENT.md",
        "STORAGE_INITIALIZATION.md",
        "STORAGE_MIGRATION.md",
        "RELEASE_QUALIFICATION.md",
    ):
        assert document in source

    assert "SqlCache" not in source
    assert "SQL pull-through" not in source
    assert "NOT_QUALIFIED" not in source


def test_task_guides_own_their_current_capabilities() -> None:
    """Guides bind store, cache, and maintenance work to current contracts."""
    blob_store = _read("docs/BLOB_STORE.md")
    cache_policy = _read("docs/CACHE_POLICY.md")
    initialization = _read("docs/STORAGE_INITIALIZATION.md")
    migration = _read("docs/STORAGE_MIGRATION.md")

    assert "from cacheness.storage import (" in blob_store
    assert "store.initialize()" in blob_store
    assert "put_entry" in blob_store
    assert "query_catalog" in blob_store
    assert "update_catalog" in blob_store
    assert "reopen" in blob_store.lower()
    assert "store.close()" in blob_store

    assert "from cacheness import CacheConfig, UnifiedCache, cached" in cache_policy
    assert "@cached(cache=cache)" in cache_policy
    assert "value can legitimately be `None`" in cache_policy
    for outcome in ("hit", "absent", "expired", "corrupt", "conflict", "backend_error"):
        assert f"`{outcome}`" in cache_policy
    assert "clear_all" in cache_policy
    assert "exact-generation" in cache_policy

    assert "ordinary opens" in initialization.lower()
    assert "do not silently" in initialization.lower()
    assert "initialize before sharing" in initialization.lower()
    assert "stopped-worker" in migration
    assert "offline" in migration.lower()
    assert "copy" in migration.lower()
    assert "verify" in migration.lower()
    assert "switch" in migration.lower()
    assert "rebuild" in migration.lower()

    for source in (blob_store, cache_policy, initialization, migration):
        assert "SqlCache" not in source
        assert "SQL pull-through" not in source


def test_qualification_guide_is_the_one_detailed_owner_of_current_nonclaims() -> None:
    """One evidence matrix distinguishes local readiness from every deferred boundary."""
    guide = QUALIFICATION_GUIDE.read_text(encoding="utf-8")
    topology_reference = _read("docs/CATALOG_AND_TOPOLOGY.md")

    for evidence_class in (
        "Deterministic/local",
        "Packaging",
        "Platform",
        "Coverage/quality",
        "Structural",
        "Controlled performance",
        "Live service",
        "Publication",
    ):
        assert evidence_class in guide

    for boundary in (
        "S3 and PostgreSQL remain `NOT_QUALIFIED`",
        "Windows remains `UNAVAILABLE` / `NOT_QUALIFIED`",
        "controlled Linux performance remains `DEFERRED` / `NOT_QUALIFIED`",
        "immutable publication remains `NOT_PUBLISHED`",
        "128 MiB",
        "private staging",
        "canonical SHA-256 plus size",
        "string, signed 64-bit integer, and boolean",
        "bounded keyset scans",
        "cross-resource ACID",
        "typed contention outcomes",
    ):
        assert boundary in guide

    assert "[Release qualification](RELEASE_QUALIFICATION.md)" in topology_reference
    assert "Phase 8 satisfies" not in topology_reference
    assert "## Evidence matrix" not in topology_reference


def test_mcap_extension_tutorial_uses_one_store_local_safe_format_path() -> None:
    """The generic format guide follows the executable MCAP-style example."""

    source = _read("docs/PLUGIN_DEVELOPMENT.md")

    for required in (
        "examples/custom_mcap_format.py",
        "FormatHandler",
        "data_type",
        "payload_format",
        "payload_format_version",
        "`.mcap`",
        "store.handlers.register_handler",
        "private staging",
        "contained regular file",
        "round trip",
        "persisted payload identities",
        '"actual_path": str(',
    ):
        assert required in source

    for retired_or_out_of_scope in (
        "CacheHandler",
        "from cacheness import register_handler",
        "register_blob_backend",
        "obstore locator",
        "conformance kit",
    ):
        assert retired_or_out_of_scope not in source


def test_navigation_has_no_pre_cutover_configuration_or_backend_branches() -> None:
    """Only current task guides remain on the supported documentation path."""

    for obsolete in (
        "docs/BACKEND_SELECTION.md",
        "docs/CONFIGURATION.md",
        "docs/DEVELOPMENT_PLANNING.md",
    ):
        assert not (PROJECT_ROOT / obsolete).exists()

    navigation = DOCS_INDEX.read_text(encoding="utf-8")
    for retired_guide in (
        "BACKEND_SELECTION.md",
        "CONFIGURATION.md",
        "DEVELOPMENT_PLANNING.md",
    ):
        assert retired_guide not in navigation

    for current_guide in (
        "BLOB_STORE.md",
        "CACHE_POLICY.md",
        "RELEASE_QUALIFICATION.md",
    ):
        assert current_guide in navigation
