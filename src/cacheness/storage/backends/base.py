"""
Abstract Base Class for Metadata Backends
========================================

Re-exports the canonical ``MetadataBackend`` ABC from
``cacheness.metadata`` so that both import paths resolve to the
same type:

    from cacheness.metadata import MetadataBackend          # canonical
    from cacheness.storage.backends.base import MetadataBackend  # alias
    from cacheness.storage.backends import MetadataBackend       # alias

The canonical definition lives in ``cacheness.metadata`` to avoid
circular imports (``storage.backends.__init__`` imports concrete
backends from ``cacheness.metadata``).
"""

from cacheness.metadata import MetadataBackend

__all__ = ["MetadataBackend"]
