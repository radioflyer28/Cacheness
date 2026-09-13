"""Current payload-participant and lifecycle-authority exports.

``StoreTopology`` remains the only composition root. Payload object mechanics
are provided by :class:`ObstoreGenerationIO`; PostgreSQL remains the optional
narrow lifecycle-authority role. Retired byte CRUD backends deliberately have
no compatibility import path.
"""

from importlib.util import find_spec

from ..obstore_generation_io import (
    ObstoreGenerationIO,
    ObstoreInventoryPage,
    ObstoreObjectEvidence,
)

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
    if find_spec("psycopg") is None:
        raise ImportError
    from .postgresql_lifecycle_authority import PostgresqlLifecycleAuthority
except ImportError:
    POSTGRESQL_AVAILABLE = False
else:
    POSTGRESQL_AVAILABLE = True


__all__ = [
    "ObstoreGenerationIO",
    "ObstoreInventoryPage",
    "ObstoreObjectEvidence",
    "BackendRef",
    "BackendRole",
    "CompositionValidationError",
    "RoleRegistry",
    "StoreTopology",
    "resolve_metadata_role",
]

if POSTGRESQL_AVAILABLE:
    __all__.append(PostgresqlLifecycleAuthority.__name__)
