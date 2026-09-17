#!/usr/bin/env python3
"""Build one wheel and probe its base public surface in an isolated process."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import tomllib
from typing import Literal, Sequence
from zipfile import BadZipFile, ZipFile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "cacheness"
BUILD_TIMEOUT_SECONDS = 180
PROBE_TIMEOUT_SECONDS = 180
EVIDENCE_PATH = REPOSITORY_ROOT / "tools" / "phase8_evidence.py"
OPTIONAL_GROUPS = (
    "recommended",
    "dataframes",
    "tensorflow",
    "s3",
    "postgresql",
    "cloud",
)
TENSORFLOW_COMPATIBLE_PYTHON_MINORS = frozenset({(3, 11), (3, 12)})
NON_LIVE_SERVICE_GROUPS = ("s3", "postgresql", "cloud")
RELEVANT_SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "src/cacheness",
    "tools/phase8_evidence.py",
    "tools/run_phase8_packaging.py",
)
_PYTHON_PATH_ENVIRONMENT = frozenset(
    {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"}
)

# These are the supported base barrels.  Keep this inventory literal: a probe
# must fail when a deliberately exported symbol vanishes, rather than adapting
# to the wheel's current ``__all__`` at runtime.
BASE_PUBLIC_EXPORTS: dict[str, tuple[str, ...]] = {
    "cacheness": (
        "UnifiedCache",
        "CacheConfig",
        "CachePolicyConfig",
        "cached",
        "CacheOutcome",
        "CacheLookupResult",
        "CacheStatistics",
        "CacheRemovalReport",
        "CacheMaintenanceState",
        "CacheMaintenanceResult",
        "CachePutResult",
        "BlobStore",
        "RoleRegistry",
        "StoreTopology",
    ),
    "cacheness.storage": (
        "BlobStore",
        "BackendRef",
        "BackendRole",
        "RoleRegistry",
        "StoreTopology",
        "ObstoreGenerationIO",
        "ObstoreInventoryPage",
        "ObstoreObjectEvidence",
        "BlobManifest",
        "BlobReceipt",
        "CatalogCursor",
        "CatalogCursorError",
        "CatalogEntry",
        "CatalogField",
        "CatalogPage",
        "CatalogPredicate",
        "CatalogQuery",
        "CatalogQueryValidationError",
        "CatalogSchema",
        "CatalogStaleCursorError",
        "CatalogValidationError",
        "STORE_FORMAT_VERSION",
        "evaluate_predicates",
        "require_current_revision",
        "validate_catalog_mapping",
        "validate_catalog_query",
        "BlobEntry",
        "PayloadTransportComparison",
        "PayloadTransportComparisonStatus",
        "CacheBlobManifestMalformedError",
        "CacheBlobManifestUnauthenticatedError",
        "CacheBlobPayloadMissingError",
        "CacheBlobPayloadTamperedError",
        "CacheBlobManifestUnsupportedVersionError",
        "CacheBlobPayloadUnsupportedVersionError",
        "CacheManifestIntegrityError",
        "CacheManifestUnsupportedVersionError",
        "CacheBlobLifecycleConflictError",
        "CacheBlobBackendError",
        "CacheBlobLockReleaseError",
        "CacheBlobCloseTimeoutError",
        "CacheBlobLifecycleTimeoutError",
        "CacheBlobStoreClosedError",
        "CacheBlobMigrationRequiredError",
        "CacheMigrationOrRebuildRequiredError",
        "CacheBlobMigrationPlanStaleError",
        "CacheBlobMigrationEvidenceError",
        "CacheBlobMigrationEvidenceMismatchError",
        "CacheBlobMigrationConfirmationError",
        "CacheBlobMigrationOfflineDecisionRequiredError",
        "CacheBlobMigrationCleanupError",
        "CacheBlobReconciliationError",
        "CacheBlobReconciliationConflictError",
        "CacheBlobReconciliationCheckpointError",
        "ReconciliationStatus",
        "ReconciliationAction",
        "ReconciliationFinding",
        "ReconciliationReport",
        "OfflineMigrationService",
        "ReleaseWindow",
        "CompatibilityDimension",
        "CompatibilityOutcome",
        "CompatibilityIdentity",
        "CompatibilityResult",
        "CompatibilityMatrix",
        "VersionEdge",
        "MigrationCompatibilityEdge",
        "MigrationDisposition",
        "MigrationReason",
        "MigrationEntryAssessment",
        "MigrationInspection",
        "RebuildExclusion",
        "MigrationPlanKind",
        "MigrationPlanState",
        "MigrationTotals",
        "MigrationPlan",
        "MigrationStepResult",
        "AbortReceipt",
        "PurgeReceipt",
        "AuthorityPublicationState",
        "AuthorityIdentitySnapshot",
        "VerifiedCandidateReceipt",
        "PriorStoreReceipt",
        "ActivationReceipt",
        "RollbackReceipt",
        "FinalizeReceipt",
        "MaintenanceEvidenceState",
        "StoppedWorkerAcknowledgement",
        "MaintenanceRunEvidence",
        "render_migration_report",
        "FormatHandler",
        "FormatHandlerError",
        "HandlerRegistry",
        "ArrayHandler",
        "ObjectHandler",
        "write_compressed",
        "read_compressed",
        "is_pickleable",
        "BLOSC_AVAILABLE",
        "DILL_AVAILABLE",
        "CacheEntrySigner",
    ),
}
RETIRED_PUBLIC_EXPORTS: dict[str, tuple[str, ...]] = {
    "cacheness": ("SqlCache", "SqlCacheAdapter"),
    "cacheness.storage": ("CacheHandler", "CacheHandlerError"),
}
RETIRED_IMPORT_MODULES = ("cacheness.sql_cache",)
RETIRED_WHEEL_MODULE_STEMS = frozenset({"cacheness/sql_cache"})
BASE_PROBE_NAMES = (
    "public_exports",
    "blobstore_generic",
    "blobstore_numpy_pickle",
    "blobstore_numpy_npz",
    "unified_cache_generic",
)


class PackagingQualificationError(RuntimeError):
    """Raised when a wheel cannot prove its required isolated behavior."""


@dataclass(frozen=True)
class WheelArtifact:
    """One immutable built wheel and the digest used to identify it."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class ProbeResult:
    """One successful isolated probe without retaining untrusted subprocess logs."""

    name: str
    requirement: str
    probes: tuple[str, ...]
    compatibility: Literal["COMPATIBLE", "INCOMPATIBLE"] = "COMPATIBLE"
    non_live: bool = False


def _wheel_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one locally built wheel."""
    digest = hashlib.sha256()
    with path.open("rb") as wheel_file:
        for chunk in iter(lambda: wheel_file.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_current_wheel_artifact(artifact: WheelArtifact) -> None:
    """Reject a missing or changed wheel before qualifying its contents."""
    if not artifact.path.is_file() or len(artifact.sha256) != 64:
        raise PackagingQualificationError("wheel artifact is unavailable")
    actual_sha256 = _wheel_sha256(artifact.path)
    if actual_sha256 != artifact.sha256:
        raise PackagingQualificationError("wheel artifact changed after its digest was recorded")


def _normalized_wheel_member(member: str) -> str:
    """Normalize one archive path without allowing it to escape its wheel root."""
    parts: list[str] = []
    for part in member.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _is_retired_wheel_member(member: str) -> bool:
    """Return whether one normalized wheel member occupies a retired module path."""
    normalized = _normalized_wheel_member(member).rstrip("/")
    for module_stem in RETIRED_WHEEL_MODULE_STEMS:
        if normalized == module_stem or normalized.startswith(f"{module_stem}/"):
            return True
        parent, separator, name = normalized.rpartition("/")
        if separator and f"{parent}/{name.split('.', maxsplit=1)[0]}" == module_stem:
            return True
    return False


def _assert_retired_wheel_members_are_absent(artifact: WheelArtifact) -> None:
    """Reject wheels that retain a retired module file or package namespace."""
    _require_current_wheel_artifact(artifact)
    try:
        with ZipFile(artifact.path) as wheel:
            members = frozenset(wheel.namelist())
    except (BadZipFile, OSError) as error:
        raise PackagingQualificationError("wheel members are unavailable") from error
    retired_members = sorted(member for member in members if _is_retired_wheel_member(member))
    if retired_members:
        raise PackagingQualificationError(
            f"wheel contains retired members: {', '.join(retired_members)}"
        )


def build_wheel(
    dist: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WheelArtifact:
    """Build exactly one wheel from the checkout and bind it to a SHA-256 digest."""
    dist.mkdir(parents=True, exist_ok=True)
    try:
        run(
            ["uv", "build", "--wheel", "--out-dir", str(dist)],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PackagingQualificationError("wheel build failed") from error
    wheels = tuple(sorted(dist.glob(f"{PACKAGE_NAME}-*.whl")))
    if len(wheels) != 1:
        raise PackagingQualificationError("wheel build did not produce exactly one artifact")
    return WheelArtifact(path=wheels[0], sha256=_wheel_sha256(wheels[0]))


def _isolated_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a clean child environment that cannot inherit a source import path."""
    source = os.environ if environment is None else environment
    return {
        name: value
        for name, value in source.items()
        if name not in _PYTHON_PATH_ENVIRONMENT
    }


def _base_probe_source() -> str:
    """Render the literal base-surface and public-composition qualification probe."""
    return f'''
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module, metadata
from importlib.util import find_spec
from io import StringIO
import os
from pathlib import Path

import numpy as np

import_stdout = StringIO()
import_stderr = StringIO()
with redirect_stdout(import_stdout), redirect_stderr(import_stderr):
    import cacheness
    import cacheness.storage as storage
    from cacheness import CacheConfig, UnifiedCache
    from cacheness.config import CacheStorageConfig
    from cacheness.storage import BackendRef, BlobStore, StoreTopology

assert not import_stdout.getvalue(), "package-generated stdout during base import"
assert not import_stderr.getvalue(), "package-generated stderr during base import"

SOURCE_ROOT = Path(os.environ["CACHENESS_PHASE8_SOURCE_ROOT"]).resolve()
for module in (cacheness, storage):
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(SOURCE_ROOT)
    except ValueError:
        pass
    else:
        raise AssertionError(f"wheel probe imported checkout module: {{module.__name__}}")

for module_name, exports in {BASE_PUBLIC_EXPORTS!r}.items():
    module = import_module(module_name)
    for export in exports:
        assert hasattr(module, export), f"missing public export: {{module_name}}.{{export}}"
    for retired in {RETIRED_PUBLIC_EXPORTS!r}.get(module_name, ()):
        assert not hasattr(module, retired), f"retired public export: {{module_name}}.{{retired}}"

for module_name in {RETIRED_IMPORT_MODULES!r}:
    assert find_spec(module_name) is None, f"retired import remains discoverable: {{module_name}}"
    try:
        import_module(module_name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f"retired import remains available: {{module_name}}")

for retired in {RETIRED_PUBLIC_EXPORTS["cacheness"]!r}:
    try:
        exec(f"from cacheness import {{retired}}", {{}})
    except ImportError:
        pass
    else:
        raise AssertionError(f"retired top-level import remains available: {{retired}}")

distribution = metadata.distribution({PACKAGE_NAME!r})
requirements = tuple(distribution.requires or ())
assert all("duckdb" not in requirement.casefold() for requirement in requirements), (
    "installed metadata retains a DuckDB requirement"
)
extras = {{extra.casefold() for extra in distribution.metadata.get_all("Provides-Extra", [])}}
assert "sql" not in extras, "installed metadata retains the sql extra"
expected_extras = set({OPTIONAL_GROUPS!r})
assert extras == expected_extras, "installed metadata optional extras do not match review"

topology = StoreTopology(
    payload=BackendRef(name="memory"),
    authority=BackendRef(name="memory"),
)
store = BlobStore(topology, cache_dir=Path.cwd() / "blobstore")
store.initialize()
try:
    generic_key = store.put({{"answer": 42, "labels": ["base", True]}}, key="generic")
    assert store.get(generic_key) == {{"answer": 42, "labels": ["base", True]}}

    pickled_array = np.arange(6, dtype=np.int64).reshape(2, 3)
    pickled_key = store.put(("pickle", pickled_array), key="numpy-pickle")
    restored_tuple = store.get(pickled_key)
    assert restored_tuple[0] == "pickle"
    assert np.array_equal(restored_tuple[1], pickled_array)

    native_array = np.arange(9, dtype=np.float32).reshape(3, 3)
    native_key = store.put(native_array, key="numpy-npz")
    assert np.array_equal(store.get(native_key), native_array)
    native_metadata = store.get_metadata(native_key)
    assert native_metadata is not None
    assert native_metadata["data_type"] == "array"
    assert native_metadata["metadata"]["storage_format"] == "npz"

    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=Path.cwd() / "cache")),
        store=store,
    )
    try:
        cache.initialize()
        written = cache.put({{"cache": "public"}}, request_id="base")
        lookup = cache.lookup(cache_key=written.receipt.key)
        assert lookup.value == {{"cache": "public"}}
    finally:
        cache.close()
finally:
    store.close()
'''


def run_base_probe(
    artifact: WheelArtifact,
    *,
    workspace: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> ProbeResult:
    """Run all required base behavior against one wheel outside the checkout."""
    _assert_retired_wheel_members_are_absent(artifact)
    workspace.mkdir(parents=True, exist_ok=True)
    child_environment = _isolated_environment(environment)
    child_environment["CACHENESS_PHASE8_SOURCE_ROOT"] = str(REPOSITORY_ROOT)
    command = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        str(artifact.path),
        "python",
        "-c",
        _base_probe_source(),
    ]
    try:
        completed = run(
            command,
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            env=child_environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PackagingQualificationError("base isolated probe did not complete") from error
    if completed.returncode != 0:
        raise PackagingQualificationError("base isolated probe failed")
    return ProbeResult(
        name="base",
        requirement=str(artifact.path),
        probes=BASE_PROBE_NAMES,
    )


def optional_groups_from_pyproject(path: Path) -> tuple[str, ...]:
    """Read and freeze the reviewed optional-dependency inventory exactly."""
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        optional = document["project"]["optional-dependencies"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise PackagingQualificationError("optional group metadata is unavailable") from error
    if not isinstance(optional, dict) or tuple(optional) != OPTIONAL_GROUPS:
        raise PackagingQualificationError("optional group inventory does not match review")
    if not all(isinstance(dependencies, list) for dependencies in optional.values()):
        raise PackagingQualificationError("optional group metadata is malformed")
    return tuple(optional)


def tensorflow_is_compatible() -> bool:
    """Return whether this interpreter has a reviewed TensorFlow probe contract."""
    return sys.version_info[:2] in TENSORFLOW_COMPATIBLE_PYTHON_MINORS


def _extra_probe_source(group: str) -> str:
    """Return one group-specific public round trip after the base probe succeeds."""
    if group == "recommended":
        extension = '''
import blosc2
from cacheness.config import CompressionConfig

recommended_store = BlobStore(
    StoreTopology(payload=BackendRef(name="memory"), authority=BackendRef(name="memory")),
    cache_dir=Path.cwd() / "recommended-store",
    config=CacheConfig(compression=CompressionConfig(compression_threshold_bytes=1)),
)
recommended_store.initialize()
try:
    recommended_key = recommended_store.put({"large": list(range(4096))}, key="blosc2")
    assert recommended_store.get(recommended_key) == {"large": list(range(4096))}
    recommended_metadata = recommended_store.get_metadata(recommended_key)
    assert recommended_metadata is not None
    assert recommended_metadata["metadata"]["storage_format"] == "compressed_pickle"
finally:
    recommended_store.close()
'''
    elif group == "dataframes":
        extension = '''
import pandas as pd
import polars as pl

dataframe_store = BlobStore(
    StoreTopology(payload=BackendRef(name="memory"), authority=BackendRef(name="memory")),
    cache_dir=Path.cwd() / "dataframes-store",
)
dataframe_store.initialize()
try:
    pandas_value = pd.DataFrame({"id": [1, 2], "name": ["one", "two"]})
    pandas_key = dataframe_store.put(pandas_value, key="pandas-parquet")
    assert dataframe_store.get(pandas_key).equals(pandas_value)

    polars_value = pl.DataFrame({"id": [1, 2], "name": ["one", "two"]})
    polars_key = dataframe_store.put(polars_value, key="polars-parquet")
    assert dataframe_store.get(polars_key).equals(polars_value)
finally:
    dataframe_store.close()
'''
    elif group == "tensorflow":
        extension = '''
import tensorflow as tf

tensorflow_store = BlobStore(
    StoreTopology(payload=BackendRef(name="memory"), authority=BackendRef(name="memory")),
    cache_dir=Path.cwd() / "tensorflow-store",
)
tensorflow_store.initialize()
try:
    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    tensor_key = tensorflow_store.put(tensor, key="tensorflow-tensor")
    restored_tensor = tensorflow_store.get(tensor_key)
    assert isinstance(restored_tensor, tf.Tensor)
    assert bool(tf.reduce_all(tf.equal(restored_tensor, tensor)))
finally:
    tensorflow_store.close()
'''
    elif group in NON_LIVE_SERVICE_GROUPS:
        dependency_import = {
            "s3": "from importlib.util import find_spec\nassert find_spec(\"boto3\") is None",
            "postgresql": "import psycopg\nimport sqlalchemy",
            "cloud": "import psycopg\nimport sqlalchemy",
        }[group]
        extension = f'''
{dependency_import}

service_store = BlobStore(
    StoreTopology(payload=BackendRef(name="memory"), authority=BackendRef(name="memory")),
    cache_dir=Path.cwd() / "{group}-store",
)
service_store.initialize()
try:
    service_key = service_store.put({{"group": "{group}"}}, key="{group}-memory")
    assert service_store.get(service_key) == {{"group": "{group}"}}
finally:
    service_store.close()
'''
    else:
        raise PackagingQualificationError("optional group is not reviewed")
    return _base_probe_source() + extension


def _run_optional_probe(
    artifact: WheelArtifact,
    *,
    group: str,
    workspace: Path,
    run: Callable[..., subprocess.CompletedProcess[str]],
    environment: Mapping[str, str] | None,
) -> ProbeResult:
    """Run one group against the wheel in an otherwise fresh process."""
    workspace.mkdir(parents=True, exist_ok=True)
    requirement = f"{artifact.path}[{group}]"
    child_environment = _isolated_environment(environment)
    child_environment["CACHENESS_PHASE8_SOURCE_ROOT"] = str(REPOSITORY_ROOT)
    try:
        completed = run(
            [
                "uv",
                "run",
                "--isolated",
                "--no-project",
                "--with",
                requirement,
                "python",
                "-c",
                _extra_probe_source(group),
            ],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            env=child_environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PackagingQualificationError(f"{group} isolated probe did not complete") from error
    if completed.returncode != 0:
        raise PackagingQualificationError(f"{group} isolated probe failed")
    probe_names = {
        "recommended": ("public_exports", "blosc2_object_round_trip"),
        "dataframes": ("public_exports", "pandas_polars_parquet_round_trip"),
        "tensorflow": ("public_exports", "tensorflow_tensor_round_trip"),
        "s3": ("public_exports", "memory_round_trip", "no_live_service"),
        "postgresql": ("public_exports", "memory_round_trip", "no_live_service"),
        "cloud": ("public_exports", "memory_round_trip", "no_live_service"),
    }[group]
    return ProbeResult(
        name=group,
        requirement=requirement,
        probes=probe_names,
        non_live=group in NON_LIVE_SERVICE_GROUPS,
    )


def run_optional_probes(
    artifact: WheelArtifact,
    *,
    workspace: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environment: Mapping[str, str] | None = None,
    tensorflow_compatible: bool | None = None,
) -> tuple[ProbeResult, ...]:
    """Qualify every exact extra in a new source-free wheel environment."""
    optional_groups_from_pyproject(REPOSITORY_ROOT / "pyproject.toml")
    _require_current_wheel_artifact(artifact)
    compatible = tensorflow_is_compatible() if tensorflow_compatible is None else tensorflow_compatible
    results: list[ProbeResult] = []
    for group in OPTIONAL_GROUPS:
        if group == "tensorflow" and not compatible:
            results.append(
                ProbeResult(
                    name=group,
                    requirement=f"{artifact.path}[{group}]",
                    probes=("tensorflow_incompatible_platform",),
                    compatibility="INCOMPATIBLE",
                )
            )
            continue
        results.append(
            _run_optional_probe(
                artifact,
                group=group,
                workspace=workspace / group,
                run=run,
                environment=environment,
            )
        )
    return tuple(results)


def _load_evidence_module():
    """Load the sibling evidence utility when invoked as a standalone script."""
    specification = importlib.util.spec_from_file_location("phase8_evidence", EVIDENCE_PATH)
    if specification is None or specification.loader is None:
        raise PackagingQualificationError("Phase 8 evidence utility is unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


phase8_evidence = _load_evidence_module()


def _relevant_sources_are_clean() -> bool:
    """Reject qualification evidence when reviewed packaging inputs are dirty."""
    try:
        completed = subprocess.run(
            (
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *RELEVANT_SOURCE_PATHS,
            ),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return not completed.stdout.strip()


def _source_identity() -> tuple[str, str] | None:
    """Return the exact revision and reviewed-source digest for one matrix run."""
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        revision = completed.stdout.strip()
        source_digest = phase8_evidence.relevant_source_digest(
            REPOSITORY_ROOT, RELEVANT_SOURCE_PATHS
        )
    except (OSError, subprocess.SubprocessError, phase8_evidence.EvidenceValidationError):
        return None
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        return None
    if not _relevant_sources_are_clean():
        return None
    return revision, source_digest


def _packaging_payload(
    artifact: WheelArtifact,
    results: Sequence[ProbeResult],
    *,
    status: Literal["PASS", "UNAVAILABLE"],
) -> dict[str, object]:
    """Render bounded package evidence without retaining subprocess diagnostics."""
    if len(results) != len(OPTIONAL_GROUPS) + 1:
        raise PackagingQualificationError("package matrix is incomplete")
    claim_state = "NOT_QUALIFIED" if status == "PASS" else "UNAVAILABLE"
    return {
        "result": "passed" if status == "PASS" else "unavailable",
        "claim_categories": {
            category: claim_state
            for category in phase8_evidence.CLAIM_CATEGORIES
        },
        "non_qualifying_classes": [
            evidence_class
            for evidence_class in phase8_evidence.EVIDENCE_CLASSES
            if evidence_class != "packaging"
        ],
        "subjects": list(phase8_evidence.QUALIFIED_SUBJECTS),
        "wheel_sha256": artifact.sha256,
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "probes": [
            f"{result.name}:{probe}"
            for result in results
            for probe in result.probes
        ],
        "optional_groups": list(OPTIONAL_GROUPS),
        "compatibility": [
            f"{result.name}:{result.compatibility}"
            for result in results
            if result.name != "base"
        ],
        "non_live_groups": list(NON_LIVE_SERVICE_GROUPS),
    }


def run_qualification(
    *,
    output: Path,
    workspace: Path,
    tensorflow_compatible: bool | None = None,
) -> int:
    """Build one wheel, qualify each extra, and write exact package evidence."""
    optional_groups_from_pyproject(REPOSITORY_ROOT / "pyproject.toml")
    before = _source_identity()
    if before is None:
        raise PackagingQualificationError("source identity is unavailable")
    artifact = build_wheel(workspace / "dist")
    base_result = run_base_probe(artifact, workspace=workspace / "base")
    extra_results = run_optional_probes(
        artifact,
        workspace=workspace / "extras",
        tensorflow_compatible=tensorflow_compatible,
    )
    after = _source_identity()
    if after != before:
        raise PackagingQualificationError("source identity changed during qualification")
    status: Literal["PASS", "UNAVAILABLE"] = (
        "PASS"
        if all(result.compatibility == "COMPATIBLE" for result in extra_results)
        else "UNAVAILABLE"
    )
    envelope = phase8_evidence.make_envelope(
        evidence_class="packaging",
        status=status,
        revision=before[0],
        source_digest=before[1],
        payload=_packaging_payload(artifact, (base_result, *extra_results), status=status),
    )
    phase8_evidence.write_envelope(output, envelope)
    return 0 if status == "PASS" else 2


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the fixed package qualification without service configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parsed = parser.parse_args(arguments)
    try:
        with tempfile.TemporaryDirectory(prefix="phase8-packaging-") as temporary:
            return run_qualification(
                output=parsed.output,
                workspace=Path(temporary),
            )
    except PackagingQualificationError:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
