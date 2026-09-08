"""Payload backends and typed store-composition exports.

Canonical lifecycle authority construction belongs exclusively to
``StoreTopology``. JSON is a derived projection; PostgreSQL is exported only
as the narrow lifecycle-authority role when its optional driver is present.
"""

from .blob_backends import (
    BlobBackend,
    FilesystemBlobBackend,
    InMemoryBlobBackend,
)
from importlib.util import find_spec

_COMPOSITION_EXPORTS = frozenset(
    {
        "BackendRef",
        "BackendRole",
        "CompositionValidationError",
        "RoleRegistry",
        "StoreTopology",
        "resolve_metadata_role",
    }
)


def __getattr__(name: str):
    """Lazily re-export composition types without import-cycle participation."""
    if name not in _COMPOSITION_EXPORTS:
        raise AttributeError(name)
    from .. import composition

    return getattr(composition, name)

try:
    from .s3_backend import BOTO3_AVAILABLE, S3BlobBackend
except ImportError:
    BOTO3_AVAILABLE = False

try:
    if find_spec("psycopg") is None:
        raise ImportError
    from .postgresql_lifecycle_authority import PostgresqlLifecycleAuthority
except ImportError:
    POSTGRESQL_AVAILABLE = False
else:
    POSTGRESQL_AVAILABLE = True


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

if POSTGRESQL_AVAILABLE:
    __all__.append(PostgresqlLifecycleAuthority.__name__)
