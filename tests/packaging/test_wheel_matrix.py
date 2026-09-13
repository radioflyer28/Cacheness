"""Integration contracts for the isolated Phase 8 wheel qualification."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).parents[2]
RUNNER_PATH = PROJECT_ROOT / "tools" / "run_phase8_packaging.py"


def _load_runner():
    """Load the standalone wheel runner without making ``tools`` a package."""
    specification = importlib.util.spec_from_file_location(
        "phase8_packaging_runner_test", RUNNER_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_base_wheel_qualification_uses_one_artifact_and_public_round_trips(
    tmp_path: Path,
) -> None:
    """A source-free wheel proves the complete base public surface and formats."""
    runner = _load_runner()

    artifact = runner.build_wheel(tmp_path / "dist")
    result = runner.run_base_probe(artifact, workspace=tmp_path / "probe")

    assert artifact.path.is_file()
    assert len(artifact.sha256) == 64
    assert result.name == "base"
    assert result.requirement == str(artifact.path)
    assert result.probes == (
        "public_exports",
        "blobstore_generic",
        "blobstore_numpy_pickle",
        "blobstore_numpy_npz",
        "unified_cache_generic",
    )


def test_base_probe_command_cannot_import_the_checkout_or_inherited_packages(
    tmp_path: Path,
) -> None:
    """The base runner fixes isolated ``uv`` flags and strips Python path state."""
    runner = _load_runner()
    artifact_path = tmp_path / "cacheness-0.3.14-py3-none-any.whl"
    artifact_path.write_bytes(b"wheel")
    artifact = runner.WheelArtifact(path=artifact_path, sha256="a" * 64)
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = tuple(command)
        observed["environment"] = kwargs["env"]
        return runner.subprocess.CompletedProcess(command, 0, "", "")

    result = runner.run_base_probe(
        artifact,
        workspace=tmp_path / "probe",
        run=fake_run,
        environment={
            "PATH": "/usr/bin",
            "PYTHONPATH": "/checkout",
            "PYTHONHOME": "/python-home",
            "VIRTUAL_ENV": "/venv",
        },
    )

    assert result.name == "base"
    assert observed["command"][:5] == (
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
    )
    assert str(artifact_path) in observed["command"]
    environment = observed["environment"]
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "VIRTUAL_ENV" not in environment


def test_optional_group_inventory_is_exact_and_rejects_metadata_drift(
    tmp_path: Path,
) -> None:
    """The runner cannot silently omit, rename, or add an advertised extra."""
    runner = _load_runner()

    assert runner.OPTIONAL_GROUPS == (
        "recommended",
        "dataframes",
        "tensorflow",
        "s3",
        "postgresql",
        "cloud",
    )
    assert runner.optional_groups_from_pyproject(PROJECT_ROOT / "pyproject.toml") == (
        runner.OPTIONAL_GROUPS
    )

    drifted = tmp_path / "pyproject.toml"
    drifted.write_text(
        "[project]\n[project.optional-dependencies]\nrecommended = []\ndataframes = []\n"
        "tensorflow = []\ns3 = []\npostgresql = []\nrenamed-cloud = []\n",
        encoding="utf-8",
    )
    with pytest.raises(runner.PackagingQualificationError, match="optional group"):
        runner.optional_groups_from_pyproject(drifted)


def test_optional_groups_get_fresh_wheel_requirements_and_non_live_probes(
    tmp_path: Path,
) -> None:
    """Every literal extra is isolated; S3/PostgreSQL probes make no service claim."""
    runner = _load_runner()
    artifact_path = tmp_path / "cacheness-0.3.14-py3-none-any.whl"
    artifact_path.write_bytes(b"wheel")
    artifact = runner.WheelArtifact(path=artifact_path, sha256="b" * 64)
    calls: list[tuple[tuple[str, ...], Path]] = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs["cwd"]))
        return runner.subprocess.CompletedProcess(command, 0, "", "")

    results = runner.run_optional_probes(
        artifact,
        workspace=tmp_path / "probes",
        run=fake_run,
        environment={"PATH": "/usr/bin"},
        tensorflow_compatible=False,
    )

    assert [result.name for result in results] == list(runner.OPTIONAL_GROUPS)
    assert [result.requirement for result in results] == [
        f"{artifact_path}[{group}]" for group in runner.OPTIONAL_GROUPS
    ]
    assert len({workspace for _command, workspace in calls}) == len(calls)
    assert len(calls) == len(runner.OPTIONAL_GROUPS) - 1
    assert all("--isolated" in command for command, _workspace in calls)
    assert all("--no-project" in command for command, _workspace in calls)
    assert all(result.non_live for result in results if result.name in {"s3", "postgresql", "cloud"})
    tensorflow = next(result for result in results if result.name == "tensorflow")
    assert tensorflow.compatibility == "INCOMPATIBLE"
    assert tensorflow.probes == ("tensorflow_incompatible_platform",)
