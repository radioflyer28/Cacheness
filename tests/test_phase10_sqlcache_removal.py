"""Fail-closed contracts for the pre-production SqlCache product cut."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import cacheness


PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "cacheness"

RETIRED_PUBLIC_NAMES = ("SqlCache", "SqlCacheAdapter")
RETIRED_SOURCE_PATHS = (
    "src/cacheness/sql_cache.py",
    "tests/test_sql_cache.py",
    "tests/test_sql_cache_documentation.py",
    "tests/test_sql_cache_failure_contract.py",
)
RETIRED_DOCUMENTATION_PATHS = (
    "docs/SQL_CACHE.md",
    "docs/CUSTOM_GAP_DETECTION.md",
    "docs/ARBITRARY_TIME_INCREMENTS.md",
)
RETIRED_EXAMPLE_PATHS = (
    "examples/beginner_sql_cache.py",
    "examples/database_backend_comparison.py",
    "examples/intelligent_storage_demo.py",
    "examples/simple_backend_demo.py",
    "examples/simple_stock_cache.py",
    "examples/stock_cache_example.py",
)
EXPECTED_OPTIONAL_GROUPS = (
    "recommended",
    "dataframes",
    "tensorflow",
    "s3",
    "postgresql",
    "cloud",
)
EXPECTED_RECOMMENDED_DEPENDENCIES = (
    "numpy>=2.0.0",
    "blosc2>=3.5.1",
    "pandas>=2.0.0,<4.0.0",
    "pyarrow>=21.0.0",
    "sqlalchemy>=2.0.0",
    "orjson>=3.8.0",
    "dill>=0.4.0",
)
CURRENT_REFERENCE_ROOTS = (
    "src/cacheness",
    "tests",
    "tools",
    "examples",
    "docs",
    "README.md",
    "pyproject.toml",
    "AGENTS.md",
    ".planning/codebase",
)
HISTORICAL_REFERENCE_EXCLUSIONS = frozenset(
    {
        "docs/phase3-architecture-audit-2026-09-06.md",
    }
)
CUTOVER_NOTE = (
    "Cacheness no longer ships SqlCache or a range-aware SQL pull-through cache."
)
REFERENCE_ALLOWLIST = {
    "docs/API_REFERENCE.md": "canonical API cutover note",
    "docs/STORAGE_MIGRATION.md": "canonical migration cutover note",
    "docs/README.md": "canonical documentation-index cutover note",
    "tests/test_phase10_sqlcache_removal.py": "negative source and import contract",
    "tests/test_public_api_contract.py": "negative public-boundary contract",
    "tests/test_phase6_public_api_contract.py": "negative cache-over-store contract",
    "tests/test_phase9_documentation.py": "negative documentation contract",
    "tools/run_phase8_packaging.py": "negative wheel boundary contract",
    "tests/packaging/test_wheel_matrix.py": "negative wheel boundary contract",
}
RETIRED_REFERENCE_MARKERS = (
    "SqlCache",
    "SqlCacheAdapter",
    "cacheness.sql_cache",
    "sql_cache",
    "SQL pull-through",
    "range-aware SQL",
    "duckdb",
    "duckdb-engine",
)
CURRENT_REFERENCE_SUFFIXES = frozenset({".md", ".py", ".toml"})


def _current_reference_files() -> tuple[Path, ...]:
    """Return the explicit current-surface scan without historical records."""
    files: set[Path] = set()
    for relative_root in CURRENT_REFERENCE_ROOTS:
        root = PROJECT_ROOT / relative_root
        if root.is_file():
            files.add(root)
        else:
            files.update(path for path in root.rglob("*") if path.is_file())

    return tuple(
        path
        for path in sorted(files)
        if path.suffix in CURRENT_REFERENCE_SUFFIXES
        and path.relative_to(PROJECT_ROOT).as_posix()
        not in HISTORICAL_REFERENCE_EXCLUSIONS
    )


def _allowed_reference(relative_path: str, source: str) -> bool:
    """Keep each surviving reference tied to an exact reviewed purpose."""
    purpose = REFERENCE_ALLOWLIST.get(relative_path)
    if purpose is None:
        return False
    if relative_path.startswith("docs/"):
        return CUTOVER_NOTE in source
    return "SqlCache" in source and (
        "not in" in source
        or "not hasattr" in source
        or "RETIRED" in source
        or "retired" in source
    )


def _matching_references(source: str) -> tuple[str, ...]:
    """Return retired product markers found in one current-surface file."""
    lowered = source.casefold()
    return tuple(
        marker
        for marker in RETIRED_REFERENCE_MARKERS
        if marker.casefold() in lowered
    )


def test_removed_source_and_dedicated_tests_are_absent() -> None:
    """The implementation and its dedicated test modules are physically gone."""
    present = [path for path in RETIRED_SOURCE_PATHS if (PROJECT_ROOT / path).exists()]
    assert present == []


def test_dedicated_docs_and_examples_are_absent() -> None:
    """Dedicated SqlCache guidance and executable journeys do not survive the cut."""
    retired_paths = RETIRED_DOCUMENTATION_PATHS + RETIRED_EXAMPLE_PATHS
    present = [path for path in retired_paths if (PROJECT_ROOT / path).exists()]
    assert present == []


def test_public_names_and_module_are_naturally_absent(tmp_path: Path) -> None:
    """Old imports fail through ordinary Python absence, never a compatibility hook."""
    package_source = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")

    assert all(name not in cacheness.__all__ for name in RETIRED_PUBLIC_NAMES)
    assert all(not hasattr(cacheness, name) for name in RETIRED_PUBLIC_NAMES)
    assert not hasattr(cacheness, "__getattr__")
    assert "sql_cache" not in package_source
    assert importlib.util.find_spec("cacheness.sql_cache") is None

    probe = """
import importlib
import importlib.util
import cacheness

assert "SqlCache" not in cacheness.__all__
assert "SqlCacheAdapter" not in cacheness.__all__
assert not hasattr(cacheness, "SqlCache")
assert not hasattr(cacheness, "SqlCacheAdapter")
assert not hasattr(cacheness, "__getattr__")
assert importlib.util.find_spec("cacheness.sql_cache") is None

try:
    exec("from cacheness import SqlCache", {})
except ImportError:
    pass
else:
    raise AssertionError("SqlCache import remained available")

try:
    importlib.import_module("cacheness.sql_cache")
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("cacheness.sql_cache module remained available")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_package_version_remains_unchanged() -> None:
    """The removal is a pre-release surface cut, not a version bump."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == "0.3.14"
    assert cacheness.__version__ == "0.3.14"


def test_manifest_lock_dependency_contract() -> None:
    """DuckDB and the retired dependency group vanish without pruning retained extras."""
    project_path = PROJECT_ROOT / "pyproject.toml"
    project_source = project_path.read_text(encoding="utf-8")
    project = tomllib.loads(project_source)
    optional = project["project"]["optional-dependencies"]
    dependency_groups = project["dependency-groups"]
    lock_source = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")

    assert tuple(optional) == EXPECTED_OPTIONAL_GROUPS
    assert tuple(optional["recommended"]) == EXPECTED_RECOMMENDED_DEPENDENCIES
    assert "sql" not in dependency_groups
    assert "duckdb" not in project_source.casefold()
    assert "duckdb" not in lock_source.casefold()
    assert "sqlalchemy>=2.0.0" in optional["recommended"]
    assert "sqlalchemy>=2.0.0" in optional["postgresql"]
    assert "psycopg[binary]>=3.1.0" in optional["postgresql"]
    assert "pandas>=2.0.0,<4.0.0" in optional["dataframes"]
    assert "pyarrow>=21.0.0" in optional["dataframes"]


def test_current_facing_references_match_allowlist() -> None:
    """Current claims cannot retain the product while dated/planning history stays intact."""
    unexpected: dict[str, tuple[str, ...]] = {}
    missing_cutover_notes: list[str] = []
    for path in _current_reference_files():
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        markers = _matching_references(source)
        if markers and not _allowed_reference(relative_path, source):
            unexpected[relative_path] = markers

    for path in (
        "docs/API_REFERENCE.md",
        "docs/STORAGE_MIGRATION.md",
        "docs/README.md",
    ):
        if CUTOVER_NOTE not in (PROJECT_ROOT / path).read_text(encoding="utf-8"):
            missing_cutover_notes.append(path)

    assert unexpected == {}
    assert missing_cutover_notes == []


def test_phase10_has_no_caller_table_tooling() -> None:
    """Phase 10 leaves caller-owned SqlCache tables untouched and unsupported."""
    forbidden = {
        path.relative_to(PACKAGE_ROOT).as_posix(): _matching_references(
            path.read_text(encoding="utf-8")
        )
        for path in PACKAGE_ROOT.rglob("*.py")
        if _matching_references(path.read_text(encoding="utf-8"))
    }

    assert forbidden == {}
