"""Fail-closed contracts for evidence normalization and audit maintenance."""

from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PHASE3_VERIFICATION = (
    ROOT
    / ".planning/phases/03-atomic-lifecycle-and-recovery-engine/03-VERIFICATION.md"
)
PHASE8_VALIDATION = (
    ROOT
    / ".planning/phases/08-production-gates-and-performance-stabilization/08-VALIDATION.md"
)
PHASE3_VALIDATION = (
    ROOT
    / ".planning/phases/03-atomic-lifecycle-and-recovery-engine/03-VALIDATION.md"
)
PHASE11_VALIDATION = (
    ROOT
    / ".planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/11-VALIDATION.md"
)
MILESTONE_AUDIT = ROOT / ".planning/v1.0-v1.0-MILESTONE-AUDIT.md"
SEED_005 = ROOT / ".planning/seeds/SEED-005-remove-native-tensorflow-support.md"
SEEDS_DIRECTORY = ROOT / ".planning/seeds"

CANONICAL_VALIDATIONS = (
    (
        "01",
        ROOT / ".planning/phases/01-compatibility-and-security-baseline/01-VALIDATION.md",
    ),
    ("03", PHASE3_VALIDATION),
    (
        "05",
        ROOT
        / ".planning/phases/05-payload-backends-and-supported-topology-qualification/05-VALIDATION.md",
    ),
    (
        "06",
        ROOT / ".planning/phases/06-unifiedcache-policy-composition/06-VALIDATION.md",
    ),
    (
        "07",
        ROOT / ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-VALIDATION.md",
    ),
    ("08", PHASE8_VALIDATION),
    (
        "09",
        ROOT / ".planning/phases/09-adoption-and-release-surface-closure/09-VALIDATION.md",
    ),
)
EXPLICIT_NONCLAIMS = (
    "PostgreSQL",
    "Amazon S3",
    "controlled-Linux",
    "Windows",
    "NOT_QUALIFIED",
    "NOT_PUBLISHED",
)
FROZEN_NON_LIVE_COMMAND = (
    "uv run --isolated --all-extras --group dev --frozen pytest -q -o "
    "log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'"
)
ORIGINAL_AUDITED = "2026-09-17T19:16:14Z"
ORIGINAL_AUDITED_HEAD = "6c8e235721151111b81c69ffe6446412dc0ef803"
PROTECTED_TREE_PATHS = (
    "README.md",
    "AGENTS.md",
    "docs/",
    ".github/workflows/",
    "src/",
    "tests/",
    "tools/",
    "examples/",
    "pyproject.toml",
    "uv.lock",
    ".planning/codebase/",
    ".planning/REQUIREMENTS.md",
)


def _git_output(repository: Path, *arguments: str) -> str:
    """Run a read-only-or-fixture Git command and return its standard output."""

    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _commit_fixture(repository: Path, message: str) -> str:
    """Commit current fixture changes and return its immutable Git identity."""

    _git_output(repository, "add", ".")
    _git_output(repository, "commit", "-qm", message)
    return _git_output(repository, "rev-parse", "HEAD")


def _qualified_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a clean protected tree and one later qualified source revision."""

    repository = tmp_path / "qualified-source"
    repository.mkdir()
    _git_output(repository, "init", "-q")
    _git_output(repository, "config", "user.email", "fixture@example.invalid")
    _git_output(repository, "config", "user.name", "Provenance Fixture")

    for relative_path in PROTECTED_TREE_PATHS:
        path = repository / relative_path.rstrip("/")
        if relative_path.endswith("/"):
            path /= "fixture.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("base\n", encoding="utf-8")
    base_revision = _commit_fixture(repository, "base protected tree")

    (repository / "README.md").write_text("qualified\n", encoding="utf-8")
    qualified_revision = _commit_fixture(repository, "qualified source tree")
    return repository, base_revision, qualified_revision


def _is_protected_tree_path(path: str) -> bool:
    """Identify source, test, package, and current-map paths a run qualifies."""

    normalized_path = path.removesuffix("/")
    protected_files = {
        "README.md",
        "AGENTS.md",
        "pyproject.toml",
        "uv.lock",
        ".planning/REQUIREMENTS.md",
    }
    protected_directories = (
        "docs/",
        ".github/workflows/",
        "src/",
        "tests/",
        "tools/",
        "examples/",
        ".planning/codebase/",
    )
    return normalized_path in protected_files or normalized_path.startswith(
        protected_directories
    )


def _assert_no_protected_paths(paths: list[str]) -> None:
    """Reject evidence if any qualified source/test path differs or is dirty."""

    protected_paths = [path for path in paths if _is_protected_tree_path(path)]
    assert not protected_paths, protected_paths


def _assert_qualified_source_tree(repository: Path, revision: str) -> None:
    """Require an ancestor commit and a clean protected tree for local evidence."""

    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert _git_output(repository, "rev-parse", f"{revision}^{{commit}}") == revision

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=repository,
        check=False,
        text=True,
        capture_output=True,
    )
    assert ancestor.returncode == 0, ancestor.stderr

    _assert_no_protected_paths(
        _git_output(repository, "diff", "--name-only", f"{revision}..HEAD").splitlines()
    )
    _assert_no_protected_paths(
        _git_output(repository, "diff", "--cached", "--name-only").splitlines()
    )
    _assert_no_protected_paths(
        _git_output(repository, "diff", "--name-only").splitlines()
    )
    _assert_no_protected_paths(
        _git_output(repository, "ls-files", "--others", "--exclude-standard").splitlines()
    )


def _assert_refreshed_audit_provenance(
    repository: Path,
    qualified_source_revision: str,
    audited_head: str,
) -> None:
    """Require a refreshed audit to identify exactly the qualified source tree."""

    _assert_qualified_source_tree(repository, qualified_source_revision)
    assert audited_head == qualified_source_revision


def test_qualified_source_provenance_accepts_artifact_only_commit_before_audit(
    tmp_path: Path,
) -> None:
    """Planning artifacts may follow qualified source while the audit remains stale."""

    repository, stale_audit_head, qualified_revision = _qualified_fixture(tmp_path)
    artifact = repository / ".planning/phases/11/11-VALIDATION.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("recorded later\n", encoding="utf-8")
    _commit_fixture(repository, "record validation artifact")

    _assert_qualified_source_tree(repository, qualified_revision)
    with pytest.raises(AssertionError):
        _assert_refreshed_audit_provenance(
            repository,
            qualified_revision,
            stale_audit_head,
        )
    _assert_refreshed_audit_provenance(
        repository,
        qualified_revision,
        qualified_revision,
    )


@pytest.mark.parametrize("revision", ("not-a-revision", "a" * 39, "a" * 41))
def test_qualified_source_provenance_rejects_malformed_revision(
    tmp_path: Path,
    revision: str,
) -> None:
    """A qualification record requires exactly one full Git commit identity."""

    repository, _base_revision, _qualified_revision = _qualified_fixture(tmp_path)

    with pytest.raises(AssertionError):
        _assert_qualified_source_tree(repository, revision)


def test_qualified_source_provenance_rejects_nonancestor_revision(
    tmp_path: Path,
) -> None:
    """A syntactically valid commit must still lead to the checked-out tree."""

    repository, _base_revision, _qualified_revision = _qualified_fixture(tmp_path)
    unrelated_revision = _git_output(
        repository,
        "commit-tree",
        _git_output(repository, "write-tree"),
        "-m",
        "unrelated root",
    )

    with pytest.raises(AssertionError):
        _assert_qualified_source_tree(repository, unrelated_revision)


@pytest.mark.parametrize("protected_path", PROTECTED_TREE_PATHS)
def test_qualified_source_provenance_rejects_protected_committed_drift(
    tmp_path: Path,
    protected_path: str,
) -> None:
    """Later docs, CI, source, tooling, test, and package drift invalidates a run."""

    repository, _base_revision, qualified_revision = _qualified_fixture(tmp_path)
    changed_path = repository / protected_path.rstrip("/")
    if protected_path.endswith("/"):
        changed_path /= "later.txt"
    changed_path.parent.mkdir(parents=True, exist_ok=True)
    changed_path.write_text("changed after qualification\n", encoding="utf-8")
    _commit_fixture(repository, "change protected path")

    with pytest.raises(AssertionError):
        _assert_qualified_source_tree(repository, qualified_revision)


@pytest.mark.parametrize("state", ("staged", "unstaged", "untracked"))
def test_qualified_source_provenance_rejects_dirty_protected_paths(
    tmp_path: Path,
    state: str,
) -> None:
    """Protected staged, unstaged, and untracked paths cannot piggyback a pass."""

    repository, _base_revision, qualified_revision = _qualified_fixture(tmp_path)
    dirty_path = repository / "docs/dirty.md"
    if state != "untracked":
        dirty_path = repository / "docs/fixture.txt"
    dirty_path.write_text("dirty\n", encoding="utf-8")
    if state == "staged":
        _git_output(repository, "add", str(dirty_path.relative_to(repository)))

    with pytest.raises(AssertionError):
        _assert_qualified_source_tree(repository, qualified_revision)


def _frontmatter(document: str) -> str:
    """Return the fail-closed YAML-shaped frontmatter without a parser dependency."""

    match = re.match(r"---\n(.*?)\n---\n", document, flags=re.DOTALL)
    assert match is not None
    return match.group(1)


def _frontmatter_value(frontmatter: str, name: str) -> str:
    """Read one scalar field while rejecting absent or malformed values."""

    match = re.search(rf"^{re.escape(name)}: (.+)$", frontmatter, flags=re.MULTILINE)
    assert match is not None
    return match.group(1).strip()


def _optional_frontmatter_value(frontmatter: str, name: str) -> str | None:
    """Read an optional scalar field without making pre-qualification invalid."""

    match = re.search(rf"^{re.escape(name)}: (.+)$", frontmatter, flags=re.MULTILINE)
    return None if match is None else match.group(1).strip()


def _task_row(document: str, task_id: str) -> str:
    """Return exactly one task-map row for a named evidence owner."""

    rows = [
        line
        for line in document.splitlines()
        if line.startswith(f"| {task_id} ")
    ]
    assert len(rows) == 1
    return rows[0]


def _assert_explicit_nonclaims(document: str) -> None:
    """Keep local evidence from silently promoting deferred external claims."""

    for marker in EXPLICIT_NONCLAIMS:
        assert marker in document


def _assert_canonical_validation(path: Path) -> None:
    """Require one normalized validation record with no controlling pending rows."""

    document = path.read_text(encoding="utf-8")
    frontmatter = _frontmatter(document)

    assert _frontmatter_value(frontmatter, "status") == "validated"
    assert _frontmatter_value(frontmatter, "nyquist_compliant") == "true"
    assert _frontmatter_value(frontmatter, "wave_0_complete") == "true"
    assert "## Per-Task Verification Map" in document
    assert "⬜ pending" not in document
    assert "✅ green" in document or "superseded" in document.casefold()
    _assert_explicit_nonclaims(document)


def test_phase3_verification_is_historical_with_named_local_closure_evidence():
    """The stale report must not become a reason to reopen lifecycle work."""
    report = PHASE3_VERIFICATION.read_text(encoding="utf-8")

    for marker in (
        "historical",
        "03-21 through 03-25",
        "5282dca",
        "phase3-direct-implementation-2026-09-06.md",
        "07.1",
        "Phase 8",
        "SQLite/local filesystem",
        "memory authority",
        "Windows remains UNAVAILABLE/NOT_QUALIFIED",
        "not a fresh independent verifier verdict",
    ):
        assert marker in report


def test_phase8_validation_records_completed_local_evidence_without_promoting_nonclaims():
    """Local evidence cannot qualify services, platforms, performance, or release."""
    validation = PHASE8_VALIDATION.read_text(encoding="utf-8")
    frontmatter = _frontmatter(validation)

    if _frontmatter_value(frontmatter, "status") == "complete":
        assert _frontmatter_value(frontmatter, "nyquist_compliant") == "true"
        assert _frontmatter_value(frontmatter, "wave_0_complete") == "true"
        for marker in (
            "completion_disposition: local_readiness_complete",
            "08-VERIFICATION.md",
            "08-LOCAL-READINESS.json",
            "Wave 0",
            "Nyquist",
            "LOCAL_READY",
            "08-18",
            "08-19",
            "08-16",
            "BACK-05",
            "QUAL-06",
            "live services",
            "publication",
        ):
            assert marker in validation
    elif _frontmatter_value(frontmatter, "status") == "validated":
        _assert_canonical_validation(PHASE8_VALIDATION)
        for marker in ("08-18", "08-19", "08-16", "BACK-05", "QUAL-06"):
            assert marker in validation
    else:
        raise AssertionError("Phase 8 validation has an unsupported mixed state")

    _assert_explicit_nonclaims(validation)


def test_phase11_phase3_validation_is_canonical() -> None:
    """Phase 3 becomes canonical without reopening its scoped lifecycle closure."""

    _assert_canonical_validation(PHASE3_VALIDATION)
    validation = PHASE3_VALIDATION.read_text(encoding="utf-8")
    for marker in (
        "03-25-SUMMARY.md",
        "phase3-direct-implementation-2026-09-06.md",
        "SQLite/local filesystem",
        "memory",
        "ADR 0001",
    ):
        assert marker in validation


def test_phase11_seed_resolution_is_canonical() -> None:
    """The removal seed remains historical rationale after its Phase 11 resolution."""

    seed = SEED_005.read_text(encoding="utf-8")
    frontmatter = _frontmatter(seed)

    assert _frontmatter_value(frontmatter, "id") == "SEED-005"
    assert _frontmatter_value(frontmatter, "status") == "fulfilled"
    assert "2026-09-13" in seed
    assert "Phase 8" in seed
    assert "Phase 11" in seed
    assert "11-03-SUMMARY.md" in seed
    assert "TensorFlow" in seed
    assert "trigger_when:" not in frontmatter
    assert "scope: unknown" not in frontmatter


def test_phase11_phase_1_5_6_validations_are_canonical() -> None:
    """Phase 1, 5, and 6 normalize their own evidence before combined discovery."""

    for _phase, path in CANONICAL_VALIDATIONS[0:1] + CANONICAL_VALIDATIONS[2:4]:
        _assert_canonical_validation(path)
    for path in (
        CANONICAL_VALIDATIONS[0][1],
        CANONICAL_VALIDATIONS[3][1],
    ):
        document = path.read_text(encoding="utf-8")
        assert "supersession" in document.casefold()
        assert "Phase 10" in document


def test_phase11_phase_7_8_9_validations_are_canonical() -> None:
    """Phase 7, 8, and 9 retain their boundaries under canonical discovery."""

    for _phase, path in CANONICAL_VALIDATIONS[4:7]:
        _assert_canonical_validation(path)
    phase7 = CANONICAL_VALIDATIONS[4][1].read_text(encoding="utf-8")
    assert "stopped-worker" in phase7
    phase8 = PHASE8_VALIDATION.read_text(encoding="utf-8")
    for marker in ("08-18", "08-19", "08-16", "BACK-05", "QUAL-06"):
        assert marker in phase8


def test_phase11_validation_discovery_and_seed_resolution_are_canonical() -> None:
    """Discovery accepts only the literal seven-record inventory after its owners pass."""

    assert tuple(phase for phase, _path in CANONICAL_VALIDATIONS) == (
        "01",
        "03",
        "05",
        "06",
        "07",
        "08",
        "09",
    )
    for _phase, path in CANONICAL_VALIDATIONS:
        _assert_canonical_validation(path)
    seed = SEED_005.read_text(encoding="utf-8")
    assert _frontmatter_value(_frontmatter(seed), "status") == "fulfilled"


def test_phase11_validation_record_matches_final_acceptance_evidence() -> None:
    """Phase 11 stays honestly draft until the one recorded final suite exists."""

    validation = PHASE11_VALIDATION.read_text(encoding="utf-8")
    frontmatter = _frontmatter(validation)
    status = _frontmatter_value(frontmatter, "status")

    if status == "draft":
        assert _frontmatter_value(frontmatter, "nyquist_compliant") == "false"
        assert _frontmatter_value(frontmatter, "wave_0_complete") == "false"
        assert "**Approval:** pending" in validation
        assert FROZEN_NON_LIVE_COMMAND in validation
        assert "✅ green" not in _task_row(validation, "11-FINAL-01")
        assert "11-FINAL-RECORD" in validation
    elif status == "validated":
        assert _frontmatter_value(frontmatter, "nyquist_compliant") == "true"
        assert _frontmatter_value(frontmatter, "wave_0_complete") == "true"
        assert FROZEN_NON_LIVE_COMMAND in validation
        assert "✅ green" in _task_row(validation, "11-FINAL-01")
        assert "**Approval:** approved" in validation
        assert "⬜ pending" not in validation
        _assert_explicit_nonclaims(validation)
        qualified_source_revision = _optional_frontmatter_value(
            frontmatter,
            "qualified_source_revision",
        )
        if qualified_source_revision is not None:
            _assert_qualified_source_tree(ROOT, qualified_source_revision)
    else:
        raise AssertionError("Phase 11 validation has an unsupported mixed state")


def test_phase11_refreshed_milestone_audit_is_evidence_derived() -> None:
    """The audit stays historical until a complete provenance transition occurs."""

    audit = MILESTONE_AUDIT.read_text(encoding="utf-8")
    frontmatter = _frontmatter(audit)
    audited = _frontmatter_value(frontmatter, "audited")
    audited_head = _frontmatter_value(frontmatter, "audited_head")

    if audited == ORIGINAL_AUDITED and audited_head == ORIGINAL_AUDITED_HEAD:
        assert _frontmatter_value(frontmatter, "status") == "tech_debt"
        for marker in (
                "Supplemental guides still contain",
                "Seven VALIDATION.md files",
                "overall: not_validated",
                "does not promote mocked or local evidence",
        ):
            assert marker in audit
        assert "Phase 11 audit closure" not in audit
    elif audited != ORIGINAL_AUDITED and audited_head != ORIGINAL_AUDITED_HEAD:
        validation = PHASE11_VALIDATION.read_text(encoding="utf-8")
        validation_frontmatter = _frontmatter(validation)
        assert _frontmatter_value(validation_frontmatter, "status") == "validated"
        assert _frontmatter_value(validation_frontmatter, "nyquist_compliant") == "true"
        qualified_source_revision = _optional_frontmatter_value(
            validation_frontmatter,
            "qualified_source_revision",
        )
        if qualified_source_revision is not None:
            _assert_refreshed_audit_provenance(
                ROOT,
                qualified_source_revision,
                audited_head,
            )
        assert FROZEN_NON_LIVE_COMMAND in audit
        assert "Phase 11" in audit
        assert "Supplemental guides still contain" not in audit
        assert "Seven VALIDATION.md files" not in audit
        assert _frontmatter_value(frontmatter, "status") in {
            "passed",
            "gaps_found",
            "tech_debt",
        }
        for phase in ("01", "03", "05", "06", "07", "08", "09"):
            assert phase in audit
        for marker in ("BACK-05", "QUAL-06", "Windows", "NOT_PUBLISHED"):
            assert marker in audit
    else:
        raise AssertionError("audit provenance changed only partially")


def test_one_dormant_narwhals_investigation_preserves_handler_owned_parquet():
    """A future compatibility investigation must not become a dependency decision."""
    seeds = list(SEEDS_DIRECTORY.glob("*narwhals*"))
    assert len(seeds) == 1

    seed = seeds[0].read_text(encoding="utf-8")
    for marker in (
        "id: SEED-008",
        "status: dormant",
        "pandas",
        "PyArrow",
        "Polars",
        "Parquet remains handler-owned",
        "dataframes",
        "D-08",
        "future developer-kit",
        "Do not install Narwhals",
        "Do not implement an adapter",
    ):
        assert marker in seed

    assert "narwhals" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "narwhals" not in (ROOT / "uv.lock").read_text(encoding="utf-8").lower()
