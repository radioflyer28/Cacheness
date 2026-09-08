"""Payload backends and typed store-composition exports.

Canonical lifecycle authority construction belongs exclusively to
``StoreTopology``. JSON and PostgreSQL integrations are derived projections,
not metadata factories or lifecycle authorities.
"""

from ..composition import (
    BackendRef,
    BackendRole,
    CompositionValidationError,
    RoleRegistry,
    StoreTopology,
    resolve_metadata_role,
)
from .blob_backends import (
    BlobBackend,
    FilesystemBlobBackend,
    InMemoryBlobBackend,
)

try:
    from .s3_backend import BOTO3_AVAILABLE, S3BlobBackend
except ImportError:
    BOTO3_AVAILABLE = False


__all__ = [
    "BlobBackend",
    "FilesystemBlobBackend",
    "InMemoryBlobBackend",
    "BackendRef",
    "BackendRole",
    "CompositionValidationError",
    "RoleRegistry",
    "StoreTopology",
    "resolve_metadata_role",
]

if BOTO3_AVAILABLE:
    __all__.append(S3BlobBackend.__name__)
