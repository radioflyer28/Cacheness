"""Regression coverage for the deterministic Phase 6 local-suite boundary."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]


def _load_runner():
    """Load the standalone runner without treating ``tools`` as a package."""
    runner_path = REPOSITORY_ROOT / "tools" / "run_phase6_local_suite.py"
    spec = spec_from_file_location("phase6_local_suite_runner", runner_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_requires_the_actual_repository_root(tmp_path: Path) -> None:
    """A subdirectory or unrelated directory cannot change test selection."""
    runner = _load_runner()

    assert runner._repository_root(REPOSITORY_ROOT) == REPOSITORY_ROOT.resolve()
    with pytest.raises(ValueError, match="Git worktree root"):
        runner._repository_root(REPOSITORY_ROOT / "tests")
    with pytest.raises(ValueError, match="not a Git repository"):
        runner._repository_root(tmp_path)


def test_runner_selects_exactly_three_live_modules_without_broad_ignore() -> None:
    """The local gate excludes only Phase 8's three live qualification files."""
    runner = _load_runner()

    command = runner.build_pytest_argv(REPOSITORY_ROOT)
    ignored = tuple(
        argument.removeprefix("--ignore=")
        for argument in command
        if argument.startswith("--ignore=")
    )

    assert ignored == runner.LIVE_QUALIFICATION_MODULES
    assert len(ignored) == 3
    assert "--ignore=tests/integration" not in command
    assert command[:5] == [sys.executable, "-m", "pytest", "tests", "-q"]
    assert command[5:7] == ["-o", "log_cli=false"]


def test_runner_rejects_a_missing_live_module_path(tmp_path: Path) -> None:
    """A renamed or absent live module fails closed before pytest starts."""
    runner = _load_runner()
    for raw_path in runner.LIVE_QUALIFICATION_MODULES[:-1]:
        fixture_path = tmp_path / raw_path
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.touch()

    with pytest.raises(ValueError, match="live qualification module is missing"):
        runner._validate_live_module_selection(tmp_path)


def test_runner_invokes_current_python_with_fixed_pytest_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The runner starts one child pytest with the fixed cwd and selection."""
    runner = _load_runner()
    invocations: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        invocations.append((list(argv), kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner, "_repository_root", lambda _path: REPOSITORY_ROOT)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.run_local_suite(REPOSITORY_ROOT) == 0
    output = capsys.readouterr().out

    assert len(invocations) == 1
    command, options = invocations[0]
    assert command == runner.build_pytest_argv(REPOSITORY_ROOT)
    assert options == {"cwd": REPOSITORY_ROOT, "check": False}
    assert "PostgreSQL/Amazon-S3 remain UNAVAILABLE/NOT_QUALIFIED locally." in output
    assert "Native Windows remains UNAVAILABLE/NOT_QUALIFIED on this host." in output


@pytest.mark.parametrize("returncode", (7, -9))
def test_runner_propagates_any_pytest_failure_status(
    monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    """Collection, test, timeout, and signal statuses cannot be made green."""
    runner = _load_runner()

    monkeypatch.setattr(runner, "_repository_root", lambda _path: REPOSITORY_ROOT)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda _argv, **_kwargs: SimpleNamespace(returncode=returncode),
    )

    assert runner.run_local_suite(REPOSITORY_ROOT) == returncode


ORDER_ISOLATION_NODE_ORDERS = (
    (
        "tests/test_public_api_contract.py",
        "tests/test_phase6_public_api_contract.py",
        "tests/test_sql_cache.py",
    ),
    (
        "tests/test_sql_cache.py",
        "tests/test_phase6_public_api_contract.py",
        "tests/test_public_api_contract.py",
    ),
)


def test_canonical_public_and_sqlcache_pairs_run_in_both_orders() -> None:
    """Every public/SqlCache pair is exercised in each relative order."""

    for first, second in (
        ("tests/test_public_api_contract.py", "tests/test_phase6_public_api_contract.py"),
        ("tests/test_public_api_contract.py", "tests/test_sql_cache.py"),
        ("tests/test_phase6_public_api_contract.py", "tests/test_sql_cache.py"),
    ):
        relative_orders = {
            nodes.index(first) < nodes.index(second)
            for nodes in ORDER_ISOLATION_NODE_ORDERS
        }
        assert relative_orders == {False, True}


@pytest.mark.parametrize("nodes", ORDER_ISOLATION_NODE_ORDERS)
def test_canonical_public_and_sqlcache_nodes_are_order_isolated(
    nodes: tuple[str, ...],
) -> None:
    """Public optional-import checks cannot pollute Phase 6 or SqlCache nodes."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *nodes, "-o", "log_cli=false"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
