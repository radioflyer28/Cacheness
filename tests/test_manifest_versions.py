"""Current store-version dimensions remain coupled to the durable SQLite layout."""

from cacheness.storage.manifest import (
    CURRENT_SQLITE_USER_VERSION,
    StoreVersionDimensions,
)


def test_current_manifest_declares_sqlite_authority_schema_nine() -> None:
    """New manifests truthfully name the current durable SQLite layout."""
    assert CURRENT_SQLITE_USER_VERSION == 9
    assert StoreVersionDimensions().sqlite_user_version == 9
