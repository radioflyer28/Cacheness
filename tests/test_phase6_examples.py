"""Executable public examples for the Phase 6 canonical cache surface."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOC = ROOT / "docs" / "CACHE_POLICY.md"


def _blocked_network_module(temporary_directory: Path) -> Path:
    """Install a child-only socket guard without changing the parent process."""

    guard_directory = temporary_directory / "network_guard"
    guard_directory.mkdir()
    (guard_directory / "sitecustomize.py").write_text(
        "import socket\n"
        "def _blocked(*_args, **_kwargs):\n"
        "    raise AssertionError('network access is prohibited in examples')\n"
        "socket.create_connection = _blocked\n"
        "socket.socket.connect = _blocked\n",
        encoding="utf-8",
    )
    return guard_directory


def _run_example(name: str, temporary_directory: Path) -> subprocess.CompletedProcess[str]:
    """Run one example from a fresh directory with sockets disabled."""

    guard_directory = _blocked_network_module(temporary_directory)
    environment = os.environ.copy()
    environment.pop("CACHENESS_CACHE_DIR", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(guard_directory), environment.get("PYTHONPATH")))
    )
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        cwd=temporary_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("name", "markers"),
    (
        (
            "simple_object_caching.py",
            (
                "PUT_COMMITTED=",
                "LOOKUP_OUTCOME=hit",
                "INVALIDATION=attempted:1,removed:1,complete:True",
                "CANONICAL_OBJECT_CACHE_EXAMPLE_OK",
            ),
        ),
        (
            "configurable_serialization_demo.py",
            (
                "PRESENCE_NONE_OUTCOME=hit",
                "STATISTICS_LOOKUPS=",
                "CANONICAL_SERIALIZATION_EXAMPLE_OK",
            ),
        ),
        (
            "api_request_caching.py",
            (
                "STORED_NONE_HIT=outcome:hit,transport_calls:1",
                "DEFAULT_FAILURE_RECOMPUTED=False",
                "OPT_IN_FAILURE=outcome:backend_error,cause:CacheBlobBackendError",
                "FUNCTION_CACHE_CLEAR=attempted:1,removed:1,complete:True,retryable:0",
                "CANONICAL_FUNCTION_CACHE_EXAMPLE_OK",
            ),
        ),
    ),
)
def test_examples_are_isolated_network_free_and_repeatable(
    tmp_path: Path, name: str, markers: tuple[str, ...]
) -> None:
    """Every public example runs twice with no service or singleton dependency."""

    outputs: list[str] = []
    for iteration in range(2):
        run_directory = tmp_path / f"{Path(name).stem}-{iteration}"
        run_directory.mkdir()
        completed = _run_example(name, run_directory)

        assert completed.returncode == 0, completed.stderr
        for marker in markers:
            assert marker in completed.stdout
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]


def test_cache_policy_docs_cover_the_canonical_outcome_contract() -> None:
    """The guide names canonical APIs, limits, ownership, and all outcomes."""

    guide = DOC.read_text(encoding="utf-8")

    for outcome in (
        "`hit`",
        "`absent`",
        "`expired`",
        "`corrupt`",
        "`conflict`",
        "`backend_error`",
    ):
        assert outcome in guide
    for required_claim in (
        "BlobStore",
        "CacheMaintenanceState",
        "CacheRemovalReport",
        "cache_last_lookup",
        "Phase 7",
        "SqlCache",
        "cross-resource ACID",
        "offline migration or rebuild",
    ):
        assert required_claim.lower() in guide.lower()
    assert "never upgrades it" in guide
