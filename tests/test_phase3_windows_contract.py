"""Platform-neutral D-22/D-31 contract tests for Windows lifecycle authority."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobBackendError, CacheConfigurationError


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
