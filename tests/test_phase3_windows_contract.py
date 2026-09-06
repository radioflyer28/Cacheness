"""Platform-neutral D-22/D-31 contract tests for Windows lifecycle authority."""

from __future__ import annotations

import importlib.util
import json
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cacheness.error_handling import CacheBlobBackendError, CacheConfigurationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_RUNNER = REPOSITORY_ROOT / "verify_platform.py"


def _run_phase3_evidence(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the repository evidence command without relying on a machine path."""
    return subprocess.run(
        [sys.executable, str(EVIDENCE_RUNNER), "--phase3", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _load_evidence_runner():
    """Load the standalone runner without adding the repository root to sys.path."""
    specification = importlib.util.spec_from_file_location(
        "phase3_platform_evidence", EVIDENCE_RUNNER
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the Phase 3 evidence runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "kwargs",
    (
        {"filesystem": "network"},
        {"principal_scope": "cross_user"},
        {"principal_scope": "cross_session"},
        {"principal_scope": "service_plus_interactive"},
    ),
)
def test_windows_contract_rejects_unsupported_before_mutation(kwargs: dict[str, str]) -> None:
    """Unsupported sharing is invalid configuration, never a SQLite downgrade."""
    from cacheness.config import LifecycleAuthorityTopology

    with pytest.raises(CacheConfigurationError):
        LifecycleAuthorityTopology(**kwargs)


def test_windows_contract_absent_root_fails_before_database_or_root_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows root provisioning is an offline deployment responsibility."""
    import cacheness.storage.sqlite_lifecycle_authority as sqlite_authority
    from cacheness.storage.lifecycle_authority import EntryExpectation, MutationSpec

    root = tmp_path / "missing-windows-root"
    monkeypatch.setattr(sqlite_authority, "_platform_name", lambda: "nt")
    authority = sqlite_authority.SqliteLifecycleAuthority.for_root(root)

    with pytest.raises(CacheBlobBackendError) as captured:
        authority.prepare_mutation(
            MutationSpec.create(
                operation_id="windows-absent",
                key="key",
                generation="generation",
                candidate_locator="generations/native",
                expected=EntryExpectation.absent(),
            )
        )

    assert not root.exists()
    assert "whoami /groups" in str(captured.value)
    assert "icacls.exe" in str(captured.value)


def test_windows_blobstore_put_preflights_absent_root_before_payload_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public write boundary rejects an unprovisioned Windows root unchanged."""
    import cacheness.storage.sqlite_lifecycle_authority as sqlite_authority
    from cacheness.storage import BlobStore

    root = tmp_path / "missing-public-windows-root"
    monkeypatch.setattr(sqlite_authority, "_platform_name", lambda: "nt")
    store = BlobStore(root, backend="json")
    try:
        with pytest.raises(CacheBlobBackendError) as captured:
            store.put({"value": "must-not-materialize"}, key="windows-preflight")

        assert not root.exists()
        assert "offline before Cacheness starts" in str(captured.value)
    finally:
        store.close()


def test_windows_contract_shape_keeps_sqlite_as_the_only_commit_authority() -> None:
    """The adapter validates deployment scope without adding a custom lock protocol."""
    import cacheness.storage.sqlite_lifecycle_authority as sqlite_authority

    source = inspect.getsource(sqlite_authority)
    assert "BEGIN IMMEDIATE" in source
    assert "whoami" in source
    assert "icacls.exe" in source
    assert "named mutex" not in source.lower()
    assert "msvcrt.locking" not in source


def test_windows_contract_documentation_binds_the_current_logon_sid() -> None:
    """Deployment guidance distinguishes the session SID from an account SID."""
    documentation = Path("docs/lifecycle-authority.md").read_text(encoding="utf-8")

    assert "S-1-5-5-X-Y" in documentation
    assert "inheritance disabled" in documentation
    assert "whoami /groups" in documentation
    assert "Get-Acl" in documentation
    assert "icacls.exe" in documentation
    assert "different session" in documentation.lower()


def test_phase3_runner_reports_non_windows_evidence_as_unavailable() -> None:
    """A non-Windows host cannot silently pass the native security target."""
    if os.name == "nt":
        pytest.skip("native Windows has a runnable evidence target")

    result = _run_phase3_evidence(
        "--require-system",
        "Windows",
        "--require-python",
        f"{sys.version_info.major}.{sys.version_info.minor}",
    )

    assert result.returncode == 2
    evidence = json.loads(result.stdout)
    assert evidence["status"] == "UNAVAILABLE"
    assert evidence["host"]["system"] != "Windows"
    assert evidence["python"]["major_minor"] == (
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    assert evidence["sqlite"]["journal_mode"] == "delete"
    assert evidence["sqlite"]["synchronous"] == "extra"
    assert evidence["topology"]["commit_authority"] == "sqlite"
    assert evidence["test_target"]["native_windows"] == "UNAVAILABLE"
    assert "icacls.exe" in evidence["offline_provisioning"]


def test_phase3_runner_reports_a_missing_required_python_as_unavailable() -> None:
    """Interpreter selection failures are blockers instead of passing skips."""
    result = _run_phase3_evidence("--require-python", "9.9")

    assert result.returncode == 2
    evidence = json.loads(result.stdout)
    assert evidence["status"] == "UNAVAILABLE"
    assert evidence["requirements"]["python"] == "9.9"
    assert evidence["test_target"]["focused_suite"] == "NOT_RUN"


def test_phase3_runner_rejects_a_second_token_report_with_the_current_logon_sid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A denial claim is insufficient when its token is the current session."""
    verify_platform = _load_evidence_runner()

    report = {
        "status": "DENIED",
        "authority_open": "DENIED",
        "root_mutation": "DENIED",
        "token_scope": "different-logon-session",
        "logon_sid": "S-1-5-5-10-20",
    }
    monkeypatch.setattr(
        verify_platform.subprocess,
        "run",
        lambda *_arguments, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout=json.dumps(report), stderr=""
        ),
    )

    denied, reason = verify_platform._run_different_token_denial(
        ["different-token-probe"], current_logon_sid="S-1-5-5-10-20"
    )

    assert denied is False
    assert reason == "different_token_not_distinct"


@pytest.mark.skipif(os.name != "nt", reason="native Windows evidence target")
def test_native_windows_phase3_evidence_target_requires_complete_security_proof() -> None:
    """Native execution cannot pass without the configured cross-token proof."""
    root = os.environ.get("CACHENESS_PHASE3_WINDOWS_ROOT")
    if not root:
        pytest.fail("CACHENESS_PHASE3_WINDOWS_ROOT must name the pre-provisioned root")
    if not os.environ.get("CACHENESS_PHASE3_WINDOWS_SECOND_TOKEN_COMMAND_JSON"):
        pytest.fail(
            "CACHENESS_PHASE3_WINDOWS_SECOND_TOKEN_COMMAND_JSON must contain a "
            "different-logon-session or service-token command"
        )

    result = _run_phase3_evidence(
        "--require-system",
        "Windows",
        "--require-python",
        f"{sys.version_info.major}.{sys.version_info.minor}",
        "--phase3-root",
        root,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["status"] == "PASS"
    assert evidence["test_target"]["native_windows"] == "PASS"
    assert evidence["topology"]["different_token"] == "DENIED"
