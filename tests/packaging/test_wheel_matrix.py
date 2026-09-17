"""Integration contracts for the isolated Phase 8 wheel qualification."""

from __future__ import annotations

from importlib import metadata
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import pytest


PROJECT_ROOT = Path(__file__).parents[2]
RUNNER_PATH = PROJECT_ROOT / "tools" / "run_phase8_packaging.py"
EVIDENCE_PATH = PROJECT_ROOT / "tools" / "phase8_evidence.py"


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


def _load_evidence():
    """Load the shared evidence validator without making ``tools`` a package."""
    specification = importlib.util.spec_from_file_location(
        "phase8_packaging_evidence_test", EVIDENCE_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _synthetic_artifact(runner, tmp_path: Path, *members: str):
    """Create one minimal valid wheel archive for runner-boundary tests."""
    path = tmp_path / "cacheness-0.3.14-py3-none-any.whl"
    with ZipFile(path, "w") as wheel:
        for member in members:
            wheel.writestr(member, "synthetic")
    return runner.WheelArtifact(path=path, sha256=runner._wheel_sha256(path))


def _execute_base_probe_prelude(
    runner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    requirements: list[str] | None = None,
    extras: list[str] | None = None,
    retired_exports: tuple[str, ...] = (),
) -> None:
    """Run the generated absence/metadata assertions without a full wheel install."""
    installed_root = tmp_path / "installed"
    cacheness_module = ModuleType("cacheness")
    cacheness_module.__file__ = str(installed_root / "cacheness" / "__init__.py")
    cacheness_module.__path__ = []
    storage_module = ModuleType("cacheness.storage")
    storage_module.__file__ = str(installed_root / "cacheness" / "storage" / "__init__.py")
    config_module = ModuleType("cacheness.config")

    for export in runner.BASE_PUBLIC_EXPORTS["cacheness"]:
        setattr(cacheness_module, export, object())
    for export in retired_exports:
        setattr(cacheness_module, export, object())
    for export in runner.BASE_PUBLIC_EXPORTS["cacheness.storage"]:
        setattr(storage_module, export, object())
    config_module.CacheStorageConfig = object()

    monkeypatch.setitem(sys.modules, "cacheness", cacheness_module)
    monkeypatch.setitem(sys.modules, "cacheness.storage", storage_module)
    monkeypatch.setitem(sys.modules, "cacheness.config", config_module)
    monkeypatch.delitem(sys.modules, "cacheness.sql_cache", raising=False)
    monkeypatch.setattr(
        metadata,
        "distribution",
        lambda _name: SimpleNamespace(
            requires=requirements or [],
            metadata=SimpleNamespace(
                get_all=lambda field, _default=None: extras or []
                if field == "Provides-Extra"
                else []
            ),
        ),
    )
    monkeypatch.setenv("CACHENESS_PHASE8_SOURCE_ROOT", str(PROJECT_ROOT))

    prelude, separator, _remainder = runner._base_probe_source().partition("\ntopology =")
    assert separator
    exec(prelude, {"__name__": "phase10_packaging_probe"})


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


def test_base_probe_freezes_the_alias_free_storage_surface_and_quiet_import():
    """The isolated wheel contract rejects retired handler names and import noise."""
    runner = _load_runner()

    storage_exports = runner.BASE_PUBLIC_EXPORTS["cacheness.storage"]
    assert "FormatHandler" in storage_exports
    assert "FormatHandlerError" in storage_exports
    assert "CacheHandler" not in storage_exports
    assert "CacheHandlerError" not in storage_exports
    assert runner.RETIRED_PUBLIC_EXPORTS == {
        "cacheness": ("SqlCache", "SqlCacheAdapter"),
        "cacheness.storage": ("CacheHandler", "CacheHandlerError")
    }
    assert runner.RETIRED_IMPORT_MODULES == ("cacheness.sql_cache",)
    assert runner.RETIRED_WHEEL_MEMBERS == {"cacheness/sql_cache.py"}

    source = runner._base_probe_source()
    assert "redirect_stdout" in source
    assert "redirect_stderr" in source
    assert "package-generated stdout" in source
    assert "package-generated stderr" in source
    assert "metadata.distribution" in source
    assert "DuckDB requirement" in source
    assert "sql extra" in source


def test_base_probe_rejects_a_retired_wheel_member_before_installation(
    tmp_path: Path,
) -> None:
    """A stale wheel member fails before the isolated package install runs."""
    runner = _load_runner()
    artifact = _synthetic_artifact(runner, tmp_path, "cacheness/sql_cache.py")
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        calls.append(tuple(command))
        return runner.subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(runner.PackagingQualificationError, match="retired members"):
        runner.run_base_probe(artifact, workspace=tmp_path / "probe", run=fake_run)

    assert calls == []


@pytest.mark.parametrize(
    ("requirements", "extras", "retired_exports", "message"),
    [
        (["duckdb-engine>=0.16.0"], [], (), "DuckDB requirement"),
        ([], ["sql"], (), "sql extra"),
        ([], [], ("SqlCache",), "retired public export: cacheness.SqlCache"),
    ],
)
def test_base_probe_rejects_retired_installed_metadata_and_exports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    requirements: list[str],
    extras: list[str],
    retired_exports: tuple[str, ...],
    message: str,
) -> None:
    """The generated installed-wheel probe rejects every retired surface edge."""
    runner = _load_runner()

    with pytest.raises(AssertionError, match=message):
        _execute_base_probe_prelude(
            runner,
            monkeypatch,
            tmp_path,
            requirements=requirements,
            extras=extras,
            retired_exports=retired_exports,
        )


def test_base_probe_command_cannot_import_the_checkout_or_inherited_packages(
    tmp_path: Path,
) -> None:
    """The base runner fixes isolated ``uv`` flags and strips Python path state."""
    runner = _load_runner()
    artifact = _synthetic_artifact(runner, tmp_path)
    artifact_path = artifact.path
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
    artifact = _synthetic_artifact(runner, tmp_path)
    artifact_path = artifact.path
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
    assert all(
        result.non_live
        for result in results
        if result.name in {"s3", "postgresql", "cloud"}
    )
    tensorflow = next(result for result in results if result.name == "tensorflow")
    assert tensorflow.compatibility == "INCOMPATIBLE"
    assert tensorflow.probes == ("tensorflow_incompatible_platform",)


def test_optional_group_wheel_qualification_runs_each_compatible_extra(
    tmp_path: Path,
) -> None:
    """Compatible extras install separately and prove their public behavior."""
    runner = _load_runner()
    artifact = runner.build_wheel(tmp_path / "dist")

    results = runner.run_optional_probes(
        artifact,
        workspace=tmp_path / "probes",
        tensorflow_compatible=False,
    )

    assert [result.name for result in results] == list(runner.OPTIONAL_GROUPS)
    assert all(result.requirement.startswith(str(artifact.path)) for result in results)
    assert all(
        result.non_live
        for result in results
        if result.name in {"s3", "postgresql", "cloud"}
    )
    tensorflow = next(result for result in results if result.name == "tensorflow")
    assert tensorflow.compatibility == "INCOMPATIBLE"


def test_packaging_evidence_allows_only_the_reviewed_sanitized_pass_shape() -> None:
    """A package matrix pass is class-scoped evidence, not a live-service claim."""
    evidence = _load_evidence()
    payload = {
        "result": "passed",
        "claim_categories": {
            "integrity": "NOT_QUALIFIED",
            "recovery": "NOT_QUALIFIED",
            "progress": "NOT_QUALIFIED",
            "performance": "NOT_QUALIFIED",
        },
        "non_qualifying_classes": [
            evidence_class
            for evidence_class in evidence.EVIDENCE_CLASSES
            if evidence_class != "packaging"
        ],
        "subjects": list(evidence.QUALIFIED_SUBJECTS),
        "wheel_sha256": "a" * 64,
        "python": "3.12.9",
        "platform": "Linux-x86_64",
        "probes": [
            "base:public_exports",
            "dataframes:pandas_polars_parquet_round_trip",
        ],
        "optional_groups": [
            "recommended",
            "dataframes",
            "tensorflow",
            "s3",
            "postgresql",
            "cloud",
        ],
        "compatibility": [
            "recommended:COMPATIBLE",
            "dataframes:COMPATIBLE",
            "tensorflow:COMPATIBLE",
            "s3:COMPATIBLE",
            "postgresql:COMPATIBLE",
            "cloud:COMPATIBLE",
        ],
        "non_live_groups": ["s3", "postgresql", "cloud"],
    }

    envelope = evidence.make_envelope(
        evidence_class="packaging",
        status="PASS",
        revision="b" * 40,
        source_digest="c" * 64,
        generated_at_utc="2026-09-13T00:00:00+00:00",
        payload=payload,
    )

    assert evidence.is_qualification_evidence(envelope, "packaging")


def test_runner_payload_preserves_the_reviewed_non_live_group_order(
    tmp_path: Path,
) -> None:
    """Runner-generated evidence remains canonical instead of set-order dependent."""
    runner = _load_runner()
    artifact = _synthetic_artifact(runner, tmp_path)
    artifact_path = artifact.path
    results = (
        runner.ProbeResult("base", str(artifact_path), ("public_exports",)),
        *(
            runner.ProbeResult(
                group,
                f"{artifact_path}[{group}]",
                ("public_exports",),
                non_live=group in {"s3", "postgresql", "cloud"},
            )
            for group in runner.OPTIONAL_GROUPS
        ),
    )

    payload = runner._packaging_payload(artifact, results, status="PASS")

    assert payload["non_live_groups"] == ["s3", "postgresql", "cloud"]
