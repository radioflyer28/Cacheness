"""Exact executable journeys for the Phase 9 public adoption surface."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]

# This is deliberately a literal allowlist: the files adopters run are the
# exact files CI exercises, rather than test-local copies of their logic.
EXAMPLE_MARKERS = (
    ("memory_blob_store.py", "MEMORY_BLOB_STORE_EXAMPLE_OK"),
    ("durable_catalog_store.py", "DURABLE_CATALOG_STORE_EXAMPLE_OK"),
    ("unified_cache.py", "UNIFIED_CACHE_EXAMPLE_OK"),
    ("custom_mcap_format.py", "CUSTOM_MCAP_FORMAT_EXAMPLE_OK"),
)


def _socket_guard(temporary_directory: Path) -> Path:
    """Create a child-only import hook that rejects socket connections."""

    guard_directory = temporary_directory / "network_guard"
    guard_directory.mkdir(parents=True)
    (guard_directory / "sitecustomize.py").write_text(
        "import socket\n"
        "def _blocked(*_args, **_kwargs):\n"
        "    raise AssertionError('network access is prohibited in examples')\n"
        "socket.create_connection = _blocked\n"
        "socket.socket.connect = _blocked\n",
        encoding="utf-8",
    )
    return guard_directory


def _run_example(
    name: str, run_directory: Path, guard_directory: Path
) -> subprocess.CompletedProcess[str]:
    """Run one unchanged public example in an isolated, network-free child."""

    environment = os.environ.copy()
    for variable in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "CACHENESS_CACHE_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ):
        environment.pop(variable, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(guard_directory), environment.get("PYTHONPATH")))
    )
    return subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        cwd=run_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(("name", "marker"), EXAMPLE_MARKERS)
def test_canonical_examples_are_exact_repeatable_and_network_free(
    tmp_path: Path, name: str, marker: str
) -> None:
    """Each published file self-verifies twice without persistent residue."""

    guard_directory = _socket_guard(tmp_path / name)
    outputs: list[str] = []
    for iteration in range(2):
        run_directory = tmp_path / name / f"run-{iteration}"
        run_directory.mkdir()
        completed = _run_example(name, run_directory, guard_directory)

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == marker
        assert not tuple(run_directory.iterdir())
        outputs.append(completed.stdout)

    assert outputs == [f"{marker}\n", f"{marker}\n"]
