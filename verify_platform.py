#!/usr/bin/env python3
"""
Cross-Platform Verification Script

Verifies that Cacheness works correctly on the current platform.
Tests core functionality and platform-specific considerations.
"""

import argparse
import json
import multiprocessing
import os
import platform
from queue import Empty
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4


PHASE3_PASS = "PASS"
PHASE3_FAIL = "FAIL"
PHASE3_UNAVAILABLE = "UNAVAILABLE"
PHASE3_EXIT_CODES = {
    PHASE3_PASS: 0,
    PHASE3_FAIL: 1,
    PHASE3_UNAVAILABLE: 2,
}
PHASE3_NATIVE_ROOT_ENV = "CACHENESS_PHASE3_WINDOWS_ROOT"
PHASE3_SECOND_TOKEN_COMMAND_ENV = "CACHENESS_PHASE3_WINDOWS_SECOND_TOKEN_COMMAND_JSON"
PHASE3_DOCUMENTED_WINDOWS_ROOT = r"C:\CachenessAuthorityRoot"

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_success(text):
    """Print success message."""
    try:
        print(f"✅ {text}")
    except UnicodeEncodeError:
        print(f"[OK] {text}")

def print_warning(text):
    """Print warning message."""
    try:
        print(f"⚠️  {text}")
    except UnicodeEncodeError:
        print(f"[WARNING] {text}")

def print_error(text):
    """Print error message."""
    try:
        print(f"❌ {text}")
    except UnicodeEncodeError:
        print(f"[ERROR] {text}")

def check_platform_info():
    """Display platform information."""
    print_header("Platform Information")
    print(f"System: {platform.system()}")
    print(f"Release: {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Python Implementation: {platform.python_implementation()}")

def check_imports():
    """Verify all required imports work."""
    print_header("Checking Imports")
    
    try:
        from cacheness import cacheness, CacheConfig
        print_success("Core imports successful")
    except ImportError as e:
        print_error(f"Core import failed: {e}")
        return False
    
    # Check optional dependencies
    optional = {
        "numpy": "NumPy arrays",
        "pandas": "Pandas DataFrames",
        "polars": "Polars DataFrames",
        "sqlalchemy": "SQLite backend",
        "blosc2": "Blosc2 compression",
    }
    
    for module, description in optional.items():
        try:
            __import__(module)
            print_success(f"{description} available")
        except ImportError:
            print_warning(f"{description} not installed (optional)")
    
    return True

def test_basic_caching():
    """Test basic caching operations."""
    print_header("Testing Basic Caching")
    
    try:
        from cacheness import cacheness, CacheConfig
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Create cache instance
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            # Test put/get
            test_data = {"test": "data", "number": 42}
            cache.put(test_data, key="test_key", version=1)
            print_success("Put operation successful")
            
            retrieved = cache.get(key="test_key", version=1)
            assert retrieved == test_data, "Retrieved data doesn't match"
            print_success("Get operation successful")
            
            # Test list entries
            entries = cache.list_entries()
            assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
            print_success("List entries successful")
            
            # Clean up
            cache.close()
            print_success("Cache close successful")
            
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print_error(f"Basic caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sqlite_backend():
    """Test SQLite backend functionality."""
    print_header("Testing SQLite Backend")
    
    try:
        import sqlalchemy
        from cacheness import cacheness, CacheConfig
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite",
            )
            cache = cacheness(config)
            
            # Test operations
            cache.put({"data": "test"}, experiment="exp1")
            cache.put({"data": "test2"}, experiment="exp2")
            print_success("SQLite put operations successful")
            
            entries = cache.list_entries()
            assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
            print_success("SQLite query successful")
            
            cache.close()
            print_success("SQLite backend test passed")
            
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        
        return True
        
    except ImportError:
        print_warning("SQLAlchemy not installed, skipping SQLite tests")
        return True
    except Exception as e:
        print_error(f"SQLite backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_manager():
    """Test context manager functionality."""
    print_header("Testing Context Manager")
    
    try:
        from cacheness import cacheness, CacheConfig
        
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_dir=temp_dir)
            
            # Use context manager
            with cacheness(config) as cache:
                cache.put({"data": "test"}, key="ctx")
                result = cache.get(key="ctx")
                assert result == {"data": "test"}
            
            print_success("Context manager works correctly")
            
        finally:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print_error(f"Context manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def _phase3_offline_provisioning_command() -> str:
    """Return the authority's exact offline-only provisioning instruction."""
    documentation = (
        Path(__file__).resolve().parent / "docs" / "lifecycle-authority.md"
    ).read_text(encoding="utf-8")
    marker = "```powershell\n"
    start = documentation.index(marker) + len(marker)
    end = documentation.index("\n```", start)
    return documentation[start:end]


def _phase3_sqlite_evidence() -> dict[str, str]:
    """Report the authority's fixed SQLite durability contract."""
    return {
        "library_version": sqlite3.sqlite_version,
        "journal_mode": "delete",
        "synchronous": "extra",
    }


def _phase3_base_evidence(arguments: argparse.Namespace) -> dict[str, object]:
    """Build redacted, machine-readable facts common to every Phase 3 result."""
    return {
        "schema_version": 1,
        "status": PHASE3_UNAVAILABLE,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "sqlite": _phase3_sqlite_evidence(),
        "filesystem": {
            "class": "local-ntfs" if os.name == "nt" else "non-windows-local",
            "root_provisioned": None,
        },
        "topology": {
            "commit_authority": "sqlite",
            "custom_win32_lock_authority": False,
            "logon_sid_class": "S-1-5-5-X-Y",
            "different_token": "NOT_RUN",
        },
        "requirements": {
            "system": arguments.require_system,
            "python": arguments.require_python,
        },
        "test_target": {
            "focused_suite": "NOT_RUN",
            "native_windows": "NOT_RUN",
        },
        "offline_provisioning": _phase3_offline_provisioning_command(),
    }


def _phase3_requirement_failure(
    evidence: dict[str, object], arguments: argparse.Namespace
) -> str | None:
    """Return a redacted unavailability reason for a host/runtime mismatch."""
    required_system = arguments.require_system
    if required_system and platform.system().casefold() != required_system.casefold():
        evidence["test_target"]["native_windows"] = PHASE3_UNAVAILABLE
        return "required_system_unavailable"

    required_python = arguments.require_python
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if required_python and actual_python != required_python:
        return "required_python_unavailable"
    return None


def _parse_second_token_command() -> list[str] | None:
    """Read an operator-provided command as JSON, never through a shell."""
    encoded = os.environ.get(PHASE3_SECOND_TOKEN_COMMAND_ENV)
    if not encoded:
        return None
    try:
        command = json.loads(encoded)
    except json.JSONDecodeError:
        return None
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        return None
    return command


def _validate_native_windows_root(root: Path) -> tuple[bool, str]:
    """Validate the existing root through the production read-only contract."""
    from cacheness.error_handling import CacheBlobBackendError
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority

    if not root.is_dir():
        return False, "preprovisioned_root_unavailable"

    authority = SqliteLifecycleAuthority.for_root(root)
    before = root.stat().st_mode, root.stat().st_mtime_ns
    observed_commands: list[tuple[str, ...]] = []
    original_run = authority._run_windows_command

    def record_read_only_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        observed_commands.append(tuple(arguments))
        return original_run(arguments)

    authority._run_windows_command = record_read_only_command
    try:
        authority._validate_windows_root()
    except CacheBlobBackendError:
        return False, "protected_dacl_rejected"

    after = root.stat().st_mode, root.stat().st_mtime_ns
    allowed = all(
        command[:1] in {("whoami",), ("icacls.exe",), ("powershell.exe",)}
        for command in observed_commands
    )
    if before != after or not allowed:
        return False, "runtime_root_or_acl_mutation_detected"
    return True, "protected_dacl_validated"


def _run_different_token_denial(
    command: list[str], *, current_logon_sid: str
) -> tuple[bool, str]:
    """Require a second-token helper to prove both authority denials."""
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False, "different_token_evidence_malformed"
    if not isinstance(report, dict):
        return False, "different_token_evidence_malformed"
    reported_logon_sid = report.get("logon_sid")
    if (
        not isinstance(reported_logon_sid, str)
        or re.fullmatch(r"S-1-5-5-[0-9]+-[0-9]+", reported_logon_sid) is None
        or reported_logon_sid == current_logon_sid
    ):
        return False, "different_token_not_distinct"
    denied = (
        completed.returncode != 0
        and report.get("status") == "DENIED"
        and report.get("authority_open") == "DENIED"
        and report.get("root_mutation") == "DENIED"
        and report.get("token_scope")
        in {"different-logon-session", "service-token"}
    )
    return denied, "different_token_denied" if denied else "different_token_not_denied"


def _phase3_same_session_worker(
    root: str, key: str, admitted: object, results: object
) -> None:
    """Write through a fresh process-local authority after an event release."""
    try:
        from cacheness.storage import BlobStore

        if not admitted.wait(timeout=10):
            raise RuntimeError("same-session worker was not admitted")
        store = BlobStore(Path(root), backend="json")
        try:
            store.put({"phase3": "same-session"}, key=key)
        finally:
            store.close()
        results.put(PHASE3_PASS)
    except Exception:
        results.put(PHASE3_FAIL)


def _run_same_session_contention(root: Path) -> tuple[bool, str]:
    """Exercise two same-token SQLite writers without a timing-based race."""
    from cacheness.storage import BlobStore

    context = multiprocessing.get_context("spawn")
    admitted = context.Event()
    results = context.Queue()
    prefix = f"phase3-same-session-{uuid4().hex}"
    keys = [f"{prefix}-{index}" for index in range(2)]
    workers = [
        context.Process(
            target=_phase3_same_session_worker,
            args=(str(root), key, admitted, results),
        )
        for key in keys
    ]
    for worker in workers:
        worker.start()
    admitted.set()
    for worker in workers:
        worker.join(timeout=20)
    if any(worker.is_alive() or worker.exitcode != 0 for worker in workers):
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        return False, "same_session_contention_failed"

    outcomes: list[str] = []
    try:
        for _ in workers:
            outcomes.append(results.get(timeout=5))
    except Empty:
        return False, "same_session_contention_failed"
    if outcomes != [PHASE3_PASS] * len(workers):
        return False, "same_session_contention_failed"

    store = BlobStore(root, backend="json")
    try:
        for key in keys:
            store.delete(key)
    finally:
        store.close()
    return True, "same_session_contention_passed"


def _collect_native_windows_evidence(
    arguments: argparse.Namespace, evidence: dict[str, object]
) -> tuple[str, str]:
    """Collect the Windows-only proof without provisioning or editing a DACL."""
    if os.name != "nt":
        return PHASE3_UNAVAILABLE, "native_windows_unavailable"

    root_value = (
        arguments.phase3_root
        or os.environ.get(PHASE3_NATIVE_ROOT_ENV)
        or PHASE3_DOCUMENTED_WINDOWS_ROOT
    )
    root = Path(root_value)
    evidence["filesystem"]["root_provisioned"] = root.is_dir()

    valid_root, root_reason = _validate_native_windows_root(root)
    if not valid_root:
        return PHASE3_FAIL, root_reason

    same_session, same_session_reason = _run_same_session_contention(root)
    if not same_session:
        return PHASE3_FAIL, same_session_reason

    command = _parse_second_token_command()
    if command is None:
        return PHASE3_UNAVAILABLE, "different_token_unavailable"
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority

    current_logon_sid = SqliteLifecycleAuthority.for_root(root)._current_windows_logon_sid()
    denied, denial_reason = _run_different_token_denial(
        command, current_logon_sid=current_logon_sid
    )
    evidence["topology"]["different_token"] = "DENIED" if denied else "NOT_DENIED"
    if not denied:
        return PHASE3_FAIL, denial_reason
    return PHASE3_PASS, root_reason


def run_phase3_evidence(arguments: argparse.Namespace) -> tuple[dict[str, object], int]:
    """Return Phase 3 evidence with PASS, FAIL, and UNAVAILABLE kept distinct."""
    evidence = _phase3_base_evidence(arguments)
    requirement_reason = _phase3_requirement_failure(evidence, arguments)
    if requirement_reason is not None:
        evidence["reason"] = requirement_reason
        evidence["status"] = PHASE3_UNAVAILABLE
        return evidence, PHASE3_EXIT_CODES[PHASE3_UNAVAILABLE]

    native_status, native_reason = _collect_native_windows_evidence(arguments, evidence)
    evidence["test_target"]["native_windows"] = native_status
    evidence["test_target"]["focused_suite"] = native_status
    evidence["reason"] = native_reason
    evidence["status"] = native_status
    return evidence, PHASE3_EXIT_CODES[native_status]


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse optional Phase 3 evidence arguments without changing legacy output."""
    parser = argparse.ArgumentParser(description="Verify Cacheness platform support")
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="emit machine-readable Phase 3 compatibility evidence",
    )
    parser.add_argument("--require-system", help="require this host system")
    parser.add_argument("--require-python", help="require this Python major.minor")
    parser.add_argument(
        "--phase3-root",
        help="pre-provisioned Windows authority root; never included in the report",
    )
    return parser.parse_args(arguments)


def run_standard_verification():
    """Run all verification tests."""
    print_header("Cacheness Cross-Platform Verification")
    
    tests = [
        ("Platform Info", check_platform_info),
        ("Imports", check_imports),
        ("Basic Caching", test_basic_caching),
        ("SQLite Backend", test_sqlite_backend),
        ("Context Manager", test_context_manager),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            if name == "Platform Info":
                test_func()
                results[name] = True
            else:
                results[name] = test_func()
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print_header("Verification Summary")
    
    # Count only test results (exclude Platform Info)
    test_results = {k: v for k, v in results.items() if k != "Platform Info"}
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    
    for name, result in test_results.items():
        if result:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print_success(f"All tests passed on {platform.system()}!")
        print("\nCacheness is fully compatible with your platform.")
        return 0
    else:
        print_warning(f"Some tests failed on {platform.system()}")
        print("\nPlease report any issues at: https://github.com/radioflyer28/cacheness/issues")
        return 1

def main(arguments: list[str] | None = None) -> int:
    """Run the legacy interactive checks or the Phase 3 evidence command."""
    parsed = parse_arguments(arguments)
    if parsed.phase3:
        evidence, status = run_phase3_evidence(parsed)
        print(json.dumps(evidence, sort_keys=True))
        return status
    return run_standard_verification()


if __name__ == "__main__":
    sys.exit(main())
