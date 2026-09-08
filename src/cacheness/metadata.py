"""Derived metadata projections.

The development-era metadata backend hierarchy was removed in the pre-release
cutover. Canonical catalog membership is owned by a ``BlobStore`` lifecycle
authority; JSON is available only as an explicitly composed, rebuildable
projection participant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class JsonProjection:
    """A named, caller-configured JSON projection destination.

    It deliberately has no entry CRUD, lifecycle, query-completeness, or
    session API. A future projection delivery implementation supplies its
    bounded schema/query contract before this participant receives catalog
    batches.
    """

    projection_name = "json"
    topology_capabilities = {
        "projection_refresh": False,
        "projection_rebuild": False,
        "online_rebuild": False,
        "offline_rebuild": False,
    }

    def __init__(self, metadata_file: str | Path, **options: Any) -> None:
        if not isinstance(metadata_file, (str, Path)):
            raise TypeError("JSON projection metadata_file must be a path")
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"Unsupported JSON projection options: {unknown}")
        self.metadata_file = Path(metadata_file)

    def close(self) -> None:
        """Release no resources; the projection has not opened its target."""
