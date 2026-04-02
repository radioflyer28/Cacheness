"""Cache statistics tracking mixin for UnifiedCache."""

from typing import Any, Dict


class StatsMixin:
    """Hit/miss recording and statistics reporting."""

    def _record_hit(self):
        """Record a cache hit if stats tracking is enabled."""
        if self.config.metadata.enable_cache_stats:
            self.metadata_backend.increment_hits()

    def _record_miss(self):
        """Record a cache miss if stats tracking is enabled."""
        if self.config.metadata.enable_cache_stats:
            self.metadata_backend.increment_misses()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.metadata_backend.get_stats()

        # Add cache-specific information
        stats.update(
            {
                "cache_dir": str(self.cache_dir),
                "max_size_bytes": self.config.storage.max_cache_size_bytes,
                "max_size_mb": self.config.storage.max_cache_size_mb,  # Backward compat
                "default_ttl_seconds": self.config.metadata.default_ttl_seconds,
                "backend_type": self.actual_backend,  # Report actual backend used
            }
        )

        return stats
