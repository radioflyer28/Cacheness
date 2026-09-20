"""Optional PostgreSQL derived-projection participant.

PostgreSQL is intentionally not a Phase 4 lifecycle authority.  A future
projection implementation may use this participant for its own checkpointed
read model, but it cannot expose payload/catalog mutation operations.
"""

from __future__ import annotations

from importlib.util import find_spec


SQLALCHEMY_AVAILABLE = find_spec("sqlalchemy") is not None
PSYCOPG_AVAILABLE = find_spec("psycopg") is not None


class PostgresBackend:
    """A capability-gated placeholder for an explicit derived projection.

    It deliberately has no lifecycle authority, metadata registry, session,
    or cache-row API.  Constructing it validates optional dependencies only;
    Phase 5 supplies the live projection delivery contract.
    """

    projection_name = "postgresql"
    topology_capabilities = {
        "projection_refresh": False,
        "projection_rebuild": False,
        "online_rebuild": False,
        "offline_rebuild": False,
    }

    def __init__(self, connection_url: str, **_options: object) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for a PostgreSQL projection. "
                "Install with: uv add sqlalchemy"
            )
        if not PSYCOPG_AVAILABLE:
            raise ImportError(
                "PostgreSQL driver is required for a PostgreSQL projection. "
                "Install with: uv add psycopg[binary]"
            )
        if not isinstance(connection_url, str) or not connection_url:
            raise ValueError("PostgreSQL projection connection_url must be a non-empty string")
        self.connection_url = connection_url
