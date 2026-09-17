"""Fail-closed contracts for the pre-production SqlCache product cut."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import cacheness
import cacheness.storage as storage
import pytest


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
CUTOVER_NOTE_BLOCK = """\
> **SqlCache cutover:** Cacheness no longer ships SqlCache or a range-aware SQL pull-through cache. `UnifiedCache` provides object/function caching over
> `BlobStore`, and `BlobStore` provides direct object persistence. There is no in-package replacement.
> caller-owned SQL tables are untouched and unsupported; they are outside migration/rebuild tooling."""
CUTOVER_NOTE_OWNERS = frozenset(
    {
        "docs/API_REFERENCE.md",
        "docs/STORAGE_MIGRATION.md",
        "docs/README.md",
    }
)
ALLOWED_NON_DOCUMENT_MARKER_COUNTS = {
    "tests/packaging/test_wheel_matrix.py": {
        "SqlCache": 4,
        "SqlCacheAdapter": 1,
        "cacheness.sql_cache": 2,
        "sql_cache": 11,
        "duckdb": 3,
        "duckdb-engine": 1,
    },
    "tests/test_phase10_sqlcache_removal.py": {
        "SqlCache": 31,
        "SqlCacheAdapter": 10,
        "cacheness.sql_cache": 10,
        "sql_cache": 25,
        "SQL pull-through": 6,
        "range-aware SQL": 5,
        "duckdb": 12,
        "duckdb-engine": 4,
    },
    "tests/test_phase071_contract_verifier.py": {"SqlCache": 2},
    "tests/test_phase4_cutover_verifier.py": {"SqlCache": 1, "sql_cache": 2},
    "tests/test_phase6_public_api_contract.py": {
        "SqlCache": 3,
        "SqlCacheAdapter": 1,
        "cacheness.sql_cache": 1,
        "sql_cache": 1,
        "SQL pull-through": 1,
    },
    "tests/test_phase9_documentation.py": {
        "SqlCache": 4,
        "sql_cache": 1,
        "SQL pull-through": 4,
        "range-aware SQL": 4,
    },
    "tests/test_phase9_quality_workflow.py": {"SqlCache": 1},
    "tests/test_public_api_contract.py": {
        "SqlCache": 9,
        "SqlCacheAdapter": 3,
        "cacheness.sql_cache": 2,
        "sql_cache": 2,
        "SQL pull-through": 1,
        "range-aware SQL": 1,
    },
    "tools/run_phase8_packaging.py": {
        "SqlCache": 2,
        "SqlCacheAdapter": 1,
        "cacheness.sql_cache": 1,
        "sql_cache": 2,
        "duckdb": 2,
    },
    "tools/verify_phase071_contracts.py": {"SqlCache": 1},
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
MAINTENANCE_SOURCE_PATHS = (
    "src/cacheness/storage/migration.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/projections.py",
)
FORBIDDEN_CALLER_TABLE_IMPORT_PREFIXES = ("sqlalchemy", "sqlite3", "duckdb")
FORBIDDEN_CALLER_TABLE_OPERATIONS = frozenset(
    {
        "autoload_with",
        "create_all",
        "create_engine",
        "drop_all",
        "executemany",
        "reflect",
    }
)


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


def _marker_counts(source: str) -> dict[str, int]:
    """Return exact retired-marker counts for a deliberately reviewed file."""
    lowered = source.casefold()
    return {
        marker: lowered.count(marker.casefold())
        for marker in RETIRED_REFERENCE_MARKERS
        if lowered.count(marker.casefold())
    }


def _allowed_reference(relative_path: str, source: str) -> bool:
    """Keep each surviving reference tied to an exact bounded purpose."""
    if relative_path in CUTOVER_NOTE_OWNERS:
        if source.count(CUTOVER_NOTE_BLOCK) != 1:
            return False
        return _matching_references(source.replace(CUTOVER_NOTE_BLOCK, "")) == ()

    return _marker_counts(source) == ALLOWED_NON_DOCUMENT_MARKER_COUNTS.get(
        relative_path
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
    malformed_cutover_notes: list[str] = []
    for path in _current_reference_files():
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        markers = _matching_references(source)
        if markers and not _allowed_reference(relative_path, source):
            unexpected[relative_path] = markers

    for path in CUTOVER_NOTE_OWNERS:
        if not _allowed_reference(
            path, (PROJECT_ROOT / path).read_text(encoding="utf-8")
        ):
            malformed_cutover_notes.append(path)

    assert unexpected == {}
    assert malformed_cutover_notes == []


def test_cutover_note_does_not_allow_positive_retired_guidance() -> None:
    """One required negative note cannot exempt a second positive reference."""
    hostile_source = f"{CUTOVER_NOTE_BLOCK}\n\nUse SqlCacheAdapter with duckdb-engine."

    assert not _allowed_reference("docs/API_REFERENCE.md", hostile_source)


def _imported_modules_and_called_names(source: str) -> tuple[set[str], set[str]]:
    """Inspect maintenance implementation for direct caller-table tooling."""
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
    return imported_modules, called_names


class _CallerTableProbe:
    """Record any accidental attempt to treat an unknown object as a table."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.lookups: list[str] = []

    def __getattr__(self, name: str):
        self.lookups.append(name)
        return None


def test_phase10_maintenance_tooling_is_blobstore_scoped() -> None:
    """Offline maintenance rejects caller tables before it can operate on them."""
    assert {"OfflineMigrationService", "render_migration_report"} <= set(storage.__all__)
    maintenance_constructor = inspect.signature(storage.OfflineMigrationService)
    assert tuple(maintenance_constructor.parameters) == (
        "source",
        "destination",
        "work_directory",
        "run_id",
        "stopped_workers_acknowledged",
        "compatibility_edges",
        "run_limits",
    )
    assert {
        "connection",
        "database_url",
        "engine",
        "metadata",
        "table",
        "tables",
    }.isdisjoint(maintenance_constructor.parameters)
    assert tuple(inspect.signature(storage.render_migration_report).parameters) == ("plan",)
    assert tuple(inspect.signature(storage.BlobStore.rebuild_projection).parameters) == (
        "self",
        "name",
        "requested",
        "workers_stopped",
    )
    assert "[project.scripts]" not in (PROJECT_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    imported_modules: set[str] = set()
    called_names: set[str] = set()
    for relative_path in MAINTENANCE_SOURCE_PATHS:
        imports, calls = _imported_modules_and_called_names(
            (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        )
        imported_modules.update(imports)
        called_names.update(calls)

    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported_modules
        for prefix in FORBIDDEN_CALLER_TABLE_IMPORT_PREFIXES
    )
    assert FORBIDDEN_CALLER_TABLE_OPERATIONS.isdisjoint(called_names)

    source = _CallerTableProbe(PROJECT_ROOT / "phase10-caller-table-source")
    destination = _CallerTableProbe(PROJECT_ROOT / "phase10-caller-table-destination")
    with pytest.raises(TypeError, match="source store does not provide explicit migration authority"):
        storage.OfflineMigrationService(
            source=source,
            destination=destination,
            work_directory=PROJECT_ROOT / ".planning" / "phase10-maintenance-probe",
            run_id="phase10-maintenance-probe",
            stopped_workers_acknowledged=True,
            compatibility_edges=(
                storage.MigrationCompatibilityEdge.current_to_current_for_test(),
            ),
        )

    assert source.lookups == ["lifecycle_authority"]
    assert destination.lookups == []
