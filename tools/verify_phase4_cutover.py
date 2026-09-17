#!/usr/bin/env python3
"""Verify the Phase 4 public-surface cutover with fixed lifecycle evidence.

The owned release matrix is parsed only from the bounded owned section of
``04-VALIDATION.md``. Historical validation records are not live verifier
inputs.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = REPOSITORY_ROOT / (
    ".planning/phases/04-metadata-composition-and-topology-contracts/"
    "04-VALIDATION.md"
)
OWNED_HEADING = "## Phase 4-Owned Matrix Paths"
OWNED_START = "<!-- phase4-owned-matrix:start -->"
OWNED_END = "<!-- phase4-owned-matrix:end -->"
REQUIRED_OWNED_PATHS = frozenset(
    {
        "tests/test_blob_backend_registry.py",
        "tests/test_blob_manifest.py",
        "tests/test_blob_manifest_backends.py",
        "tests/test_blob_store_atomic_lifecycle.py",
        "tests/test_blob_store_close_contract.py",
        "tests/test_blob_store_composition.py",
        "tests/test_blob_store_concurrency.py",
        "tests/test_blob_store_integrity.py",
        "tests/test_blob_store_legacy_contract.py",
        "tests/test_blob_store_read_contract.py",
        "tests/test_blob_store_reconciliation.py",
        "tests/test_cached_custom_metadata.py",
        "tests/test_cached_query_meta.py",
        "tests/test_catalog_projection.py",
        "tests/test_catalog_query_contract.py",
        "tests/test_catalog_schema.py",
        "tests/test_clear_recovery.py",
        "tests/test_config_validation.py",
        "tests/test_core.py",
        "tests/test_custom_metadata.py",
        "tests/test_filesystem_containment.py",
        "tests/test_lifecycle_authority_contract.py",
        "tests/test_manifest_repository_cas.py",
        "tests/test_metadata.py",
        "tests/test_metadata_backend_registry.py",
        "tests/test_metadata_role_contract.py",
        "tests/test_phase3_gap_acceptance.py",
        "tests/test_phase3_local_workflows.py",
        "tests/test_phase3_scheduler_retirement.py",
        "tests/test_phase3_windows_contract.py",
        "tests/test_postgresql_backend.py",
        "tests/test_projection_mutation_contract.py",
        "tests/test_projection_sql_atomicity.py",
        "tests/test_public_api_contract.py",
        "tests/test_s3_blob_backend.py",
        "tests/test_sqlite_bootstrap_concurrency.py",
        "tests/test_sqlite_metadata_bootstrap_atomicity.py",
        "tests/test_stored_compatibility.py",
        "tests/test_topology_capabilities.py",
        "tests/test_unified_cache_adversarial_lifecycle.py",
        "tests/test_unified_cache_lifecycle_authority.py",
    }
)
RETIRED_METADATA_SYMBOLS = frozenset(
    {
        "Base",
        "CacheEntry",
        "CachedMetadataBackend",
        "JsonBackend",
        "JsonMetadataBackend",
        "MetadataBackend",
        "SQLiteMetadataBackend",
        "SqliteBackend",
        "create_metadata_backend",
        "get_metadata_backend",
        "register_metadata_backend",
        "unregister_metadata_backend",
    }
)
RETIRED_BLOB_SELECTOR_SYMBOLS = frozenset(
    {
        "BlobBackendRegistry",
        "create_blob_backend",
        "get_blob_backend",
        "list_blob_backends",
        "register_blob_backend",
        "unregister_blob_backend",
    }
)
_PATH_BULLET = re.compile(r"^- (tests/test_[A-Za-z0-9_]+\.py)$")


def _normalise_test_path(candidate: str) -> str:
    """Validate a literal repository-relative test module path."""
    path = PurePosixPath(candidate)
    if (
        candidate != path.as_posix()
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "tests"
        or not path.name.startswith("test_")
        or path.suffix != ".py"
    ):
        raise ValueError(f"matrix path is not a normalized tests/test_*.py path: {candidate}")
    if not (REPOSITORY_ROOT / path).is_file():
        raise ValueError(f"matrix path does not exist: {candidate}")
    return path.as_posix()


def _parse_bounded_paths(
    text: str, *, heading: str, start: str, end: str
) -> tuple[str, ...]:
    """Parse one exact marker-bounded Markdown path list."""
    if text.count(heading) != 1:
        raise ValueError(f"validation must contain exactly one heading: {heading}")
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"validation must contain one start/end marker for {heading}")
    heading_index = text.index(heading)
    start_index = text.index(start)
    end_index = text.index(end)
    if not heading_index < start_index < end_index:
        raise ValueError(f"validation markers are misplaced for {heading}")
    body = text[start_index + len(start) : end_index]
    paths: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _PATH_BULLET.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed matrix line in {heading}: {raw_line!r}")
        paths.append(_normalise_test_path(match.group(1)))
    if not paths:
        raise ValueError(f"validation section is empty: {heading}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"validation section contains duplicate paths: {heading}")
    return tuple(paths)


def load_owned_matrix() -> tuple[str, ...]:
    """Load the retained bounded matrix and enforce its release invariants."""
    text = VALIDATION_PATH.read_text(encoding="utf-8")
    owned = _parse_bounded_paths(
        text, heading=OWNED_HEADING, start=OWNED_START, end=OWNED_END
    )
    owned_set = set(owned)
    missing = REQUIRED_OWNED_PATHS - owned_set
    if missing:
        raise ValueError(f"owned Phase 4 matrix is missing required paths: {sorted(missing)}")
    return owned


# This is the only source of matrix membership. Do not broaden it by scraping
# arbitrary test references from this document or the repository.
PHASE4_MATRIX = load_owned_matrix()


def _dotted_name(node: ast.AST) -> str | None:
    """Return a dotted attribute/name expression when it has no dynamic part."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return None if prefix is None else f"{prefix}.{node.attr}"
    return None


class _RetiredConsumerVisitor(ast.NodeVisitor):
    """Find executable retired API uses while leaving string absence checks alone."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.retired_aliases: set[str] = set()
        self.module_aliases: dict[str, str] = {}
        self._reported: set[tuple[int, int, str]] = set()

    def _report(self, node: ast.AST, category: str, detail: str) -> None:
        """Record one deterministic finding per source location and category."""
        location = (
            getattr(node, "lineno", -1),
            getattr(node, "col_offset", -1),
            category,
        )
        if location not in self._reported:
            self._reported.add(location)
            self.errors.append(f"{category} {detail}")

    def _retire_import(self, alias: ast.alias, module: str, symbol: str) -> None:
        local_name = alias.asname or alias.name
        self.retired_aliases.add(local_name)
        self._report(
            alias,
            "retired import",
            f"{module}.{symbol} as {local_name}",
        )

    def _retire_star_import(self, node: ast.ImportFrom, module: str) -> None:
        self._report(node, "retired star import", module)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if module == "cacheness":
            for alias in node.names:
                if alias.name in RETIRED_BLOB_SELECTOR_SYMBOLS:
                    self._retire_import(alias, module, alias.name)
                elif alias.name == "metadata":
                    self.module_aliases[alias.asname or alias.name] = "cacheness.metadata"
        elif module == "cacheness.metadata":
            for alias in node.names:
                if alias.name == "*":
                    self._retire_star_import(node, module)
                elif alias.name in RETIRED_METADATA_SYMBOLS:
                    self._retire_import(alias, module, alias.name)
        elif module == "cacheness.storage.backends":
            for alias in node.names:
                if alias.name == "*":
                    self._retire_star_import(node, module)
                elif alias.name == "blob_backends":
                    self.module_aliases[alias.asname or alias.name] = (
                        "cacheness.storage.backends.blob_backends"
                    )
                elif alias.name in RETIRED_BLOB_SELECTOR_SYMBOLS:
                    self._retire_import(alias, module, alias.name)
        elif module == "cacheness.storage.backends.blob_backends":
            for alias in node.names:
                if alias.name == "*":
                    self._retire_star_import(node, module)
                elif alias.name in RETIRED_BLOB_SELECTOR_SYMBOLS:
                    self._retire_import(alias, module, alias.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name in {
                "cacheness",
                "cacheness.metadata",
                "cacheness.storage.backends",
                "cacheness.storage.backends.blob_backends",
            } and alias.asname:
                self.module_aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id in self.retired_aliases:
            self._report(node, "retired bound alias use:", node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        dotted = _dotted_name(node)
        if dotted is not None:
            root, *tail = dotted.split(".")
            resolved = self.module_aliases.get(root)
            if resolved is not None and tail:
                dotted = ".".join((resolved, *tail))
            if dotted.startswith("cacheness.metadata."):
                symbol = dotted.rsplit(".", 1)[1]
                if symbol in RETIRED_METADATA_SYMBOLS:
                    self._report(node, "retired package attribute:", dotted)
            if dotted.startswith("cacheness."):
                symbol = dotted.rsplit(".", 1)[1]
                if symbol in RETIRED_BLOB_SELECTOR_SYMBOLS:
                    self._report(node, "retired package attribute:", dotted)
            if dotted.startswith("cacheness.storage.backends"):
                symbol = dotted.rsplit(".", 1)[1]
                if symbol in RETIRED_BLOB_SELECTOR_SYMBOLS:
                    self._report(node, "retired package attribute:", dotted)
        self.generic_visit(node)


def audit_source(source: str, filename: str) -> tuple[str, ...]:
    """Audit one source string with the exact visitor used for the repository tree."""
    tree = ast.parse(source, filename=filename)
    visitor = _RetiredConsumerVisitor()
    visitor.visit(tree)
    return tuple(visitor.errors)


def _tracked_python_consumers() -> Iterable[Path]:
    """Yield every tracked-or-present executable consumer in the agreed roots."""
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    roots = ("src/", "tests/", "tools/", "examples/", "benchmarks/")
    for raw_path in sorted(set(result.stdout.splitlines())):
        if not raw_path.endswith(".py"):
            continue
        if raw_path != "verify_platform.py" and not raw_path.startswith(roots):
            continue
        path = REPOSITORY_ROOT / raw_path
        if path.is_file():
            yield path


def run_consumer_audit() -> int:
    """Fail on executable consumers of retired authority or selector APIs."""
    findings: list[str] = []
    for path in _tracked_python_consumers():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        try:
            findings.extend(
                f"{relative}: {item}"
                for item in audit_source(path.read_text(encoding="utf-8"), relative)
            )
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            findings.append(f"{relative}: cannot parse executable consumer: {error}")
    if findings:
        print("Phase 4 consumer audit failed:", file=sys.stderr)
        print("\n".join(f"- {item}" for item in findings), file=sys.stderr)
        return 1
    print("Phase 4 consumer audit passed")
    return 0


def _pytest_summary(output: str) -> str:
    """Return the terminal pytest count line for reproducible evidence output."""
    for line in reversed(output.splitlines()):
        if re.search(r"\b(?:passed|failed|skipped|error)\b", line):
            return line.strip()
    return "pytest completed without a recognized summary line"


def run_owned_matrix() -> int:
    """Run exactly the owned matrix with no selection, ignore, or xfail switch."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-vv",
        "-o",
        "log_cli=false",
        *PHASE4_MATRIX,
    ]
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        print("Phase 4 owned matrix failed", file=sys.stderr)
        print(output, file=sys.stderr)
        return result.returncode or 1
    print(
        "Phase 4 owned matrix passed "
        f"({len(PHASE4_MATRIX)} modules; Python {sys.version.split()[0]}; "
        f"{_pytest_summary(output)})"
    )
    return 0


def run_collection_diagnostic() -> int:
    """Require full-tree collection to remain free of collection errors."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "log_cli=false"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode == 0:
        print(
            "Full-tree collection diagnostic completed without collection errors; "
            "this is not a full-suite execution claim."
        )
        return 0
    print("Full-tree collection diagnostic failed:", file=sys.stderr)
    print(output, file=sys.stderr)
    return result.returncode or 1


def main() -> int:
    """Run one independently selectable release-evidence check."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit", action="store_true", help="AST-audit executable consumers")
    modes.add_argument("--matrix", action="store_true", help="run only the owned matrix")
    modes.add_argument(
        "--diagnostic", action="store_true", help="run only the full-tree collection diagnostic"
    )
    modes.add_argument("--all", action="store_true", help="run audit, matrix, and diagnostic")
    arguments = parser.parse_args()
    try:
        if arguments.audit:
            return run_consumer_audit()
        if arguments.matrix:
            return run_owned_matrix()
        if arguments.diagnostic:
            return run_collection_diagnostic()
        for check in (run_consumer_audit, run_owned_matrix, run_collection_diagnostic):
            result = check()
            if result:
                return result
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Phase 4 cutover verifier failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
