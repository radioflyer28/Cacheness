"""Integration contracts for the isolated Phase 8 wheel qualification."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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
