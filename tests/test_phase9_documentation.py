"""Contract tests for the Phase 9 adoption documentation surface."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
README = PROJECT_ROOT / "README.md"
DOCS_INDEX = PROJECT_ROOT / "docs" / "README.md"


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
