"""
Storage Layer
=============

Direct object storage is composed from one ``StoreTopology`` and owned by
``BlobStore``.  Catalog authority is not selected independently from the
payload lifecycle.

This layer is designed to be reusable beyond caching use cases, such as:
- ML model versioning
- Artifact storage
- Data pipeline checkpoints

Usage:
    # Direct storage layer access
    from cacheness.storage import BlobStore
    
    store = BlobStore(topology, compression="lz4")
    blob_id = store.put(data, metadata={"type": "model", "version": "1.0"})
    data = store.get(blob_id)
    
    # Access handlers directly  
    from cacheness.storage.handlers import HandlerRegistry, ArrayHandler, ObjectHandler
"""

# Import from backends subpackage
# Import from handlers subpackage
from importlib.util import find_spec

from .handlers import (
    CacheHandler,
    HandlerRegistry,
    ArrayHandler,
    ObjectHandler,
)

# Import compression utilities
from .compression import (
    write_file as write_compressed,
    read_file as read_compressed,
    is_pickleable,
    BLOSC_AVAILABLE,
    DILL_AVAILABLE,
)

# Import security
from .security import CacheEntrySigner

# Import BlobStore
from .blob_store import BlobStore
from .catalog import (
    CatalogCursor,
    CatalogCursorError,
    CatalogEntry,
    CatalogField,
    CatalogPage,
    CatalogPredicate,
    CatalogQuery,
    CatalogQueryValidationError,
    CatalogSchema,
    CatalogStaleCursorError,
    CatalogValidationError,
    STORE_FORMAT_VERSION,
    evaluate_predicates,
    require_current_revision,
    validate_catalog_mapping,
    validate_catalog_query,
)
from .read_contract import BlobEntry, BlobReceipt
from .manifest import BlobManifest
from .migration import (
    AbortReceipt,
    CompatibilityDimension,
    CompatibilityIdentity,
    CompatibilityMatrix,
    CompatibilityOutcome,
    CompatibilityResult,
    MigrationCompatibilityEdge,
    MigrationDisposition,
    MigrationEntryAssessment,
    MigrationInspection,
    MigrationPlan,
    MigrationPlanKind,
    MigrationPlanState,
    MigrationReason,
    MigrationStepResult,
    MigrationTotals,
    OfflineMigrationService,
    PurgeReceipt,
    RebuildExclusion,
    ReleaseWindow,
    VersionEdge,
    render_migration_report,
)
from .migration_authority import (
    ActivationReceipt,
    AuthorityIdentitySnapshot,
    AuthorityPublicationState,
    FinalizeReceipt,
    PriorStoreReceipt,
    RollbackReceipt,
    VerifiedCandidateReceipt,
)
from .migration_evidence import (
    MaintenanceEvidenceState,
    MaintenanceRunEvidence,
    StoppedWorkerAcknowledgement,
)
from .reconciliation import (
    ReconciliationAction,
    ReconciliationFinding,
    ReconciliationReport,
    ReconciliationStatus,
)
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobCloseTimeoutError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobLockReleaseError,
    CacheBlobManifestMalformedError,
    CacheBlobManifestUnauthenticatedError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationCleanupError,
    CacheBlobMigrationConfirmationError,
    CacheBlobMigrationEvidenceError,
    CacheBlobMigrationEvidenceMismatchError,
    CacheBlobMigrationOfflineDecisionRequiredError,
    CacheBlobMigrationPlanStaleError,
    CacheBlobMigrationRequiredError,
    CacheBlobPayloadMissingError,
    CacheBlobPayloadTamperedError,
    CacheBlobPayloadUnsupportedVersionError,
    CacheManifestIntegrityError,
    CacheManifestUnsupportedVersionError,
    CacheMigrationOrRebuildRequiredError,
    CacheBlobReconciliationCheckpointError,
    CacheBlobReconciliationConflictError,
    CacheBlobReconciliationError,
    CacheBlobStoreClosedError,
)

from .composition import BackendRef, BackendRole, RoleRegistry, StoreTopology

try:
    from .backends.s3_backend import BOTO3_AVAILABLE, S3BlobBackend
except ImportError:
    BOTO3_AVAILABLE = False

try:
    if find_spec("psycopg") is None:
        raise ImportError
    from .backends.postgresql_lifecycle_authority import PostgresqlLifecycleAuthority
except ImportError:
    POSTGRESQL_AVAILABLE = False
else:
    POSTGRESQL_AVAILABLE = True

__all__ = [
    # Main API
    "BlobStore",
    "BackendRef",
    "BackendRole",
    "RoleRegistry",
    "StoreTopology",
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
    # Explicit offline migration and rebuild API
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
    # Handlers
    "CacheHandler",
    "HandlerRegistry",
    "ArrayHandler",
    "ObjectHandler",
    # Compression
    "write_compressed",
    "read_compressed",
    "is_pickleable",
    "BLOSC_AVAILABLE",
    "DILL_AVAILABLE",
    # Security
    "CacheEntrySigner",
]

if BOTO3_AVAILABLE:
    __all__.append(S3BlobBackend.__name__)

if POSTGRESQL_AVAILABLE:
    __all__.append(PostgresqlLifecycleAuthority.__name__)
