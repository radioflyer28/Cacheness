#!/usr/bin/env python3
"""Build one wheel and probe its base public surface in an isolated process."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "cacheness"
BUILD_TIMEOUT_SECONDS = 180
PROBE_TIMEOUT_SECONDS = 180
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
        "SqlCache",
        "SqlCacheAdapter",
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
        "CacheHandler",
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


def _wheel_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one locally built wheel."""
    digest = hashlib.sha256()
    with path.open("rb") as wheel_file:
        for chunk in iter(lambda: wheel_file.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
from importlib import import_module
import os
from pathlib import Path

import numpy as np

import cacheness
import cacheness.storage as storage
from cacheness import CacheConfig, UnifiedCache
from cacheness.config import CacheStorageConfig
from cacheness.storage import BackendRef, BlobStore, StoreTopology

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
    if not artifact.path.is_file() or len(artifact.sha256) != 64:
        raise PackagingQualificationError("wheel artifact is unavailable")
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
