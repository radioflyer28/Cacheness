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
    get_blob_backend,
    list_blob_backends,
    register_blob_backend,
    unregister_blob_backend,
)

try:
    from .s3_backend import BOTO3_AVAILABLE, S3BlobBackend
except ImportError:
    BOTO3_AVAILABLE = False


__all__ = [
    "BlobBackend",
    "FilesystemBlobBackend",
    "InMemoryBlobBackend",
    "register_blob_backend",
    "unregister_blob_backend",
    "get_blob_backend",
    "list_blob_backends",
    "BackendRef",
    "BackendRole",
    "CompositionValidationError",
    "RoleRegistry",
    "StoreTopology",
    "resolve_metadata_role",
]

if BOTO3_AVAILABLE:
    __all__.append(S3BlobBackend.__name__)
