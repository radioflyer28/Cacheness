#!/usr/bin/env python3
"""Run the Phase 6 local regression corpus without live qualification tests.

This runner is intentionally a narrow selection boundary.  It executes the
repository's normal ``tests`` testpath and excludes only the three Phase 8
modules that require live PostgreSQL/Amazon S3 qualification resources.  A
successful local run is not evidence that either live qualification, or native
Windows qualification, has passed.
"""

from __future__ import annotations

import argparse
import stat
import subprocess
import sys
from pathlib import Path
from typing import Sequence


LIVE_QUALIFICATION_MODULES = (
    "tests/integration/test_postgresql_authority.py",
    "tests/integration/test_s3_generation.py",
    "tests/integration/test_remote_topology.py",
)

_EXPECTED_LIVE_QUALIFICATION_MODULES = frozenset(
    {
        "tests/integration/test_postgresql_authority.py",
        "tests/integration/test_s3_generation.py",
        "tests/integration/test_remote_topology.py",
    }
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 6 non-live local pytest corpus."
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        type=Path,
        help="Repository root; it must name the actual Git worktree root.",
    )
    return parser.parse_args(argv)


def _repository_root(path: Path) -> Path:
    """Validate and return an actual Git repository root.

    Requiring the supplied path to be the worktree root prevents a caller from
    silently changing what the repository-relative exclusions select.
    """
    candidate = path.resolve()
    if not candidate.is_dir():
        raise ValueError(f"repository root is not a directory: {candidate}")

    result = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"repository root is not a Git repository: {candidate}")

    actual_root = Path(result.stdout.strip()).resolve()
    if actual_root != candidate:
        raise ValueError(
            "repository root must be the Git worktree root: "
            f"expected {actual_root}, received {candidate}"
        )
    return actual_root


def _validate_live_module_selection(repo_root: Path) -> tuple[Path, ...]:
    """Return validated exact live-module exclusions under ``repo_root``."""
    if len(LIVE_QUALIFICATION_MODULES) != len(set(LIVE_QUALIFICATION_MODULES)):
        raise ValueError("live qualification module selection contains duplicates")
    if set(LIVE_QUALIFICATION_MODULES) != _EXPECTED_LIVE_QUALIFICATION_MODULES:
        raise ValueError("live qualification module selection is not the exact Phase 8 set")

    validated: list[Path] = []
    for raw_path in LIVE_QUALIFICATION_MODULES:
        relative_path = Path(raw_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"live qualification path must be repository-relative: {raw_path}")
        candidate = repo_root / relative_path
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError:
            raise ValueError(f"live qualification module is missing: {raw_path}")
        if stat.S_ISLNK(candidate_stat.st_mode):
            raise ValueError(f"live qualification module must not be a symlink: {raw_path}")
        if not stat.S_ISREG(candidate_stat.st_mode):
            raise ValueError(f"live qualification module must be a regular file: {raw_path}")
        validated.append(candidate)
    return tuple(validated)


def build_pytest_argv(repo_root: Path) -> list[str]:
    """Build the fixed non-live pytest command for a validated root."""
    ignored_modules = _validate_live_module_selection(repo_root)
    return [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "-o",
        "log_cli=false",
        *[f"--ignore={path.relative_to(repo_root).as_posix()}" for path in ignored_modules],
    ]


def run_local_suite(repo_root: Path) -> int:
    """Run pytest once and return its unmodified nonzero status when it fails."""
    root = _repository_root(repo_root)
    command = build_pytest_argv(root)
    print("Phase 6 local suite excludes only these Phase 8 live modules:")
    for path in LIVE_QUALIFICATION_MODULES:
        print(f"  - {path}")
    print("PostgreSQL/Amazon-S3 remain UNAVAILABLE/NOT_QUALIFIED locally.")
    print("Native Windows remains UNAVAILABLE/NOT_QUALIFIED on this host.")

    try:
        completed = subprocess.run(command, cwd=root, check=False)
    except OSError as error:
        print(f"failed to start pytest: {error}", file=sys.stderr)
        return 1
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the deterministic local suite."""
    arguments = _parse_args(argv)
    try:
        return run_local_suite(arguments.repo_root)
    except ValueError as error:
        print(f"Phase 6 local-suite configuration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
