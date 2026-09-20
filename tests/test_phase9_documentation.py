"""Contract tests for the Phase 9 adoption documentation surface."""

from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).parent.parent
README = PROJECT_ROOT / "README.md"
DOCS_INDEX = PROJECT_ROOT / "docs" / "README.md"
API_REFERENCE = PROJECT_ROOT / "docs" / "API_REFERENCE.md"
STORAGE_MIGRATION = PROJECT_ROOT / "docs" / "STORAGE_MIGRATION.md"
QUALIFICATION_GUIDE = PROJECT_ROOT / "docs" / "RELEASE_QUALIFICATION.md"
CUTOVER_NOTE = (
    "Cacheness no longer ships SqlCache or a range-aware SQL pull-through cache."
)
CUTOVER_NOTE_OWNERS = (API_REFERENCE, STORAGE_MIGRATION, DOCS_INDEX)
NON_OWNER_CURRENT_GUIDES = (
    README,
    PROJECT_ROOT / "docs" / "BLOB_STORE.md",
    PROJECT_ROOT / "docs" / "CACHE_POLICY.md",
    PROJECT_ROOT / "docs" / "PLUGIN_DEVELOPMENT.md",
    PROJECT_ROOT / "docs" / "STORAGE_INITIALIZATION.md",
    PROJECT_ROOT / "docs" / "PANDAS_API_AUDIT.md",
)
PLATFORM_AND_TENSORFLOW_SUPPLEMENTS = (
    "docs/CROSS_PLATFORM_GUIDE.md",
    "docs/WINDOWS_COMPATIBILITY.md",
    "docs/TENSORFLOW_TENSOR_GUIDE.md",
    "docs/TENSORFLOW_HANDLER_STATUS.md",
)
PANDAS_AND_CUSTOM_METADATA_SUPPLEMENTS = (
    "docs/PANDAS_COMPATIBILITY.md",
    "docs/CUSTOM_METADATA.md",
)
CANONICAL_EXECUTABLE_EXAMPLES = (
    "memory_blob_store.py",
    "durable_catalog_store.py",
    "unified_cache.py",
    "custom_mcap_format.py",
)


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
        "range-aware SQL",
        "cacheness()",
        "Windows",
        "PostgreSQL",
        "S3",
    )
    assert all(term not in source for term in retired_or_unqualified)
    for qualification_detail in (
        "128 MiB",
        "opaque transport evidence",
        '"actual_path"',
    ):
        assert qualification_detail not in source


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

    assert "NOT_QUALIFIED" not in source


def test_task_guides_own_their_current_capabilities() -> None:
    """Guides bind store, cache, and maintenance work to current contracts."""
    blob_store = _read("docs/BLOB_STORE.md")
    cache_policy = _read("docs/CACHE_POLICY.md")
    initialization = _read("docs/STORAGE_INITIALIZATION.md")
    migration = STORAGE_MIGRATION.read_text(encoding="utf-8")

    assert "from cacheness.storage import (" in blob_store
    assert "store.initialize()" in blob_store
    assert "put_entry" in blob_store
    assert "query_catalog" in blob_store
    assert "update_catalog" in blob_store
    assert "CacheBlobLifecycleConflictError" in blob_store
    assert "`None` is reserved for an entry that is genuinely absent" in blob_store
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

    # This runbook routes mutable qualification status to the sole detailed
    # owner. It deliberately does not recreate that evidence matrix here.
    migration_qualification = re.sub(
        r"\s+", " ", _section(migration, "## Topology guarantees and non-claims")
    ).strip()
    assert (
        "its mutable status and details live in "
        "[Release qualification](RELEASE_QUALIFICATION.md)"
    ) in migration_qualification
    assert (
        "[SEED-006](../.planning/seeds/"
        "SEED-006-qualify-controlled-linux-performance.md) "
        "owns controlled-Linux performance"
    ) in migration_qualification
    assert (
        "[SEED-007](../.planning/seeds/"
        "SEED-007-qualify-real-postgresql-s3-and-publish-release.md) "
        "owns real PostgreSQL/Amazon-S3 qualification and immutable publication"
    ) in migration_qualification
    assert (
        "[Phase 999.1](../.planning/ROADMAP.md#"
        "phase-9991-qualify-native-windows-lifecycle-authority-backlog) "
        "owns native Windows qualification"
    ) in migration_qualification

    phase_8_future_ownership = re.compile(
        r"""
        \bphase\s*8\b[^.]{0,180}\b
        (?:alone|sole(?:ly)?|only)?\s*
        (?:owns?|is\s+(?:the\s+)?(?:sole\s+)?owner\s+of|
        is\s+responsible\s+for|controls?|is\s+tasked\s+with)\b
        [^.]{0,180}\b(?:future\s+)?
        (?:qualification|remote|windows|performance|publication)\b
        |
        \b(?:future\s+)?(?:qualification|remote|windows|performance|publication)\b
        [^.]{0,180}\b(?:belongs\s+to|is\s+owned\s+by|
        is\s+the\s+responsibility\s+of|is\s+controlled\s+by)\b
        [^.]{0,180}\bphase\s*8\b
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    for stale_claim in (
        "Phase 8 owns future qualification",
        "Phase 8 alone owns Windows qualification",
        "Future remote qualification belongs to Phase 8",
    ):
        assert phase_8_future_ownership.search(stale_claim)
    assert not phase_8_future_ownership.search(migration_qualification)
    assert "## Evidence matrix" not in migration
    for status_token in (
        "PASS",
        "NOT_QUALIFIED",
        "NOT_PUBLISHED",
        "UNAVAILABLE",
        "DEFERRED",
    ):
        assert re.search(rf"\b{status_token}\b", migration) is None
    assert "remote payload effects remain verifiable external effects" in migration
    assert "This guide makes no promise of automatic or seamless migration" in migration

    for source in (blob_store, cache_policy, initialization):
        assert "SqlCache" not in source
        assert "SQL pull-through" not in source
        assert "range-aware SQL" not in source


def test_canonical_cutover_notes_have_exact_owners_and_boundaries() -> None:
    """Only canonical guidance may name the retired product and its limits."""
    sources = {
        owner.relative_to(PROJECT_ROOT).as_posix(): owner.read_text(encoding="utf-8")
        for owner in CUTOVER_NOTE_OWNERS
    }

    assert tuple(sources) == (
        "docs/API_REFERENCE.md",
        "docs/STORAGE_MIGRATION.md",
        "docs/README.md",
    )
    for source in sources.values():
        assert CUTOVER_NOTE in source
        assert "UnifiedCache" in source
        assert "object/function caching" in source
        assert "BlobStore" in source
        assert "object persistence" in source
        assert "no in-package replacement" in source
        assert "caller-owned SQL tables" in source
        assert "untouched" in source
        assert "unsupported" in source


def test_non_owner_current_guides_cannot_promote_the_retired_product() -> None:
    """Removal wording belongs only to the three bounded canonical owners."""
    retired_product_terms = ("SqlCache", "SQL pull-through", "range-aware SQL")

    for guide in NON_OWNER_CURRENT_GUIDES:
        source = guide.read_text(encoding="utf-8")
        assert all(term not in source for term in retired_product_terms), guide


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
        "SQL_CACHE.md",
        "CUSTOM_GAP_DETECTION.md",
        "ARBITRARY_TIME_INCREMENTS.md",
    ):
        assert retired_guide not in navigation

    for current_guide in (
        "BLOB_STORE.md",
        "CACHE_POLICY.md",
        "RELEASE_QUALIFICATION.md",
    ):
        assert current_guide in navigation


def test_all_relative_documentation_links_resolve() -> None:
    """Deleting obsolete guides cannot leave supported Markdown links dangling."""

    markdown_link = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    missing: list[str] = []
    for document in sorted((PROJECT_ROOT / "docs").glob("*.md")):
        for raw_target in markdown_link.findall(document.read_text(encoding="utf-8")):
            target = raw_target.split("#", 1)[0]
            if not target or target.startswith("#") or re.match(r"^[a-z]+:", target):
                continue
            resolved = (document.parent / unquote(target)).resolve()
            if not resolved.exists():
                missing.append(f"{document.name}: {raw_target}")
    assert missing == []


def test_platform_and_tensorflow_supplements_are_consolidated_or_deleted() -> None:
    """Platform/TensorFlow guides yield to one qualified nonclaim owner."""
    present = [
        relative_path
        for relative_path in PLATFORM_AND_TENSORFLOW_SUPPLEMENTS
        if (PROJECT_ROOT / relative_path).exists()
    ]
    qualification = QUALIFICATION_GUIDE.read_text(encoding="utf-8")

    assert present == []
    for nonclaim in (
        "Windows remains `UNAVAILABLE` / `NOT_QUALIFIED`",
        "S3 and PostgreSQL remain `NOT_QUALIFIED`",
        "controlled Linux performance remains `DEFERRED` / `NOT_QUALIFIED`",
        "immutable publication remains `NOT_PUBLISHED`",
    ):
        assert nonclaim in qualification
    assert "tensorflow" not in qualification.casefold()


def test_pandas_and_custom_metadata_supplements_are_consolidated_or_deleted() -> None:
    """Current navigation owns one concise, verified DataFrame/Parquet note."""
    present = [
        relative_path
        for relative_path in PANDAS_AND_CUSTOM_METADATA_SUPPLEMENTS
        if (PROJECT_ROOT / relative_path).exists()
    ]
    navigation = DOCS_INDEX.read_text(encoding="utf-8")
    api_reference = API_REFERENCE.read_text(encoding="utf-8")

    assert present == []
    assert "Pandas and Polars DataFrames use Parquet" in api_reference
    assert "`dataframes`" in api_reference
    assert "PANDAS_COMPATIBILITY.md" not in navigation
    assert "CUSTOM_METADATA.md" not in navigation

    linked_examples = tuple(
        re.findall(r"\]\(\.\./examples/([a-z_]+\.py)\)", navigation)
    )
    assert linked_examples == CANONICAL_EXECUTABLE_EXAMPLES
    assert all(
        (PROJECT_ROOT / "examples" / example).is_file()
        for example in CANONICAL_EXECUTABLE_EXAMPLES
    )


def test_supplemental_documentation_is_consolidated_or_deleted() -> None:
    """Both independent consolidation contracts finish with one exact deletion set."""
    deleted_supplements = (
        *PLATFORM_AND_TENSORFLOW_SUPPLEMENTS,
        *PANDAS_AND_CUSTOM_METADATA_SUPPLEMENTS,
    )

    present = [
        relative_path
        for relative_path in deleted_supplements
        if (PROJECT_ROOT / relative_path).exists()
    ]

    assert deleted_supplements == (
        "docs/CROSS_PLATFORM_GUIDE.md",
        "docs/WINDOWS_COMPATIBILITY.md",
        "docs/TENSORFLOW_TENSOR_GUIDE.md",
        "docs/TENSORFLOW_HANDLER_STATUS.md",
        "docs/PANDAS_COMPATIBILITY.md",
        "docs/CUSTOM_METADATA.md",
    )
    assert present == []


def test_current_guidance_has_no_supported_tensorflow_claim() -> None:
    """Current guides cannot retain a dormant handler as supported guidance."""
    current_documents = tuple(sorted((PROJECT_ROOT / "docs").glob("*.md")))

    tensorflow_claims = [
        document.relative_to(PROJECT_ROOT).as_posix()
        for document in current_documents
        if "tensorflow" in document.read_text(encoding="utf-8").casefold()
    ]

    assert tensorflow_claims == []
