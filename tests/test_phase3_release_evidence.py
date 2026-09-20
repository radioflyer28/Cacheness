"""Fail-closed evidence that the retired scheduler is absent from v1.0."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tarfile
import zipfile


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


def test_scheduler_history_is_after_base_but_absent_from_v1_tag_tree() -> None:
    """A history-bearing tag is safe only when its released tree omits the scheduler."""
    assert _git("rev-parse", "--verify", ORIGIN_MAIN)
    assert _git("merge-base", "--is-ancestor", ORIGIN_MAIN, SCHEDULER_INTRODUCTION) == ""
    assert _git("ls-tree", "-r", "--name-only", "v1.0", "--", *SCHEDULER_PATHS) == ""
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
    scheduler_names = {Path(path).name for path in SCHEDULER_PATHS}
    release_text_paths = [
        path
        for path in tracked
        if path == "README.md"
        or path == "pyproject.toml"
        or path.startswith("docs/")
        or path.startswith("tests/fixtures/compat/")
    ]
    for relative_path in release_text_paths:
        if relative_path == "docs/lifecycle-authority.md":
            continue
        contents = (REPOSITORY_ROOT / relative_path).read_text(
            encoding="utf-8", errors="ignore"
        )
        assert not any(name in contents for name in scheduler_names), relative_path

    artifacts = [
        path
        for path in (REPOSITORY_ROOT / "dist").glob("*")
        if path.suffix in {".whl", ".zip", ".gz"}
    ] if (REPOSITORY_ROOT / "dist").is_dir() else []
    for artifact in artifacts:
        if artifact.suffix in {".whl", ".zip"}:
            with zipfile.ZipFile(artifact) as archive:
                member_names = archive.namelist()
        else:
            with tarfile.open(artifact) as archive:
                member_names = archive.getnames()
        assert not any(name in member for member in member_names for name in scheduler_names)


def test_documented_rebuild_boundary_is_explicit_and_non_replaying() -> None:
    """The local development scheduler has one documented disposal path only."""
    document = DOCUMENTATION.read_text(encoding="utf-8")

    assert "571dfd6" in document
    assert "a22f4b4" in document
    assert "explicit rebuild" in document
    assert "No scheduler reader, replay engine, or migration executor" in document
    assert "not a released stored format" in document
