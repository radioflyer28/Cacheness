"""Fail-closed evidence that the retired scheduler was never released."""

from __future__ import annotations

from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTATION = REPOSITORY_ROOT / "docs" / "lifecycle-authority.md"
ORIGIN_MAIN = "a22f4b4"
SCHEDULER_INTRODUCTION = "571dfd6"
SCHEDULER_PATHS = (
    "src/cacheness/storage/operation_repository.py",
    "src/cacheness/storage/operation_record.py",
    "src/cacheness/storage/clear_recovery.py",
)


def _git(*arguments: str) -> str:
    """Run one read-only Git evidence query."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_scheduler_history_is_after_origin_main_and_absent_from_release_tags() -> None:
    """Any tag containing scheduler code is a release contradiction and must halt."""
    assert _git("rev-parse", "--verify", ORIGIN_MAIN) == _git(
        "rev-parse", "--verify", "origin/main"
    )
    assert _git("merge-base", "--is-ancestor", ORIGIN_MAIN, SCHEDULER_INTRODUCTION) == ""
    assert _git("for-each-ref", "--contains", SCHEDULER_INTRODUCTION, "--format=%(refname)", "refs/tags") == ""
    introduced_paths = _git(
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        SCHEDULER_INTRODUCTION,
    ).splitlines()
    assert "src/cacheness/storage/lifecycle.py" in introduced_paths


def test_no_released_fixture_or_built_artifact_claims_scheduler_support() -> None:
    """Tracked release surfaces may not silently make scheduler replay necessary."""
    tracked = set(_git("ls-files").splitlines())
    assert not set(SCHEDULER_PATHS).intersection(
        path for path in tracked if path.startswith("tests/fixtures/")
    )

    artifacts = [
        path
        for path in (REPOSITORY_ROOT / "dist").glob("*")
        if path.suffix in {".whl", ".zip", ".gz"}
    ] if (REPOSITORY_ROOT / "dist").is_dir() else []
    for artifact in artifacts:
        assert "scheduler" not in artifact.name.lower()


def test_documented_rebuild_boundary_is_explicit_and_non_replaying() -> None:
    """The local development scheduler has one documented disposal path only."""
    document = DOCUMENTATION.read_text(encoding="utf-8")

    assert "571dfd6" in document
    assert "a22f4b4" in document
    assert "explicit rebuild" in document
    assert "No scheduler reader, replay engine, or migration executor" in document
    assert "not a released stored format" in document
