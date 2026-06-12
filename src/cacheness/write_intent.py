"""Write intent journal for crash-safe blob writes.

Records intent files before blob writes and removes them after metadata
commit succeeds.  On cache init, stale intents (from crashed writes) are
detected and their orphaned blobs deleted.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from .json_utils import dumps as json_dumps
from .json_utils import loads as json_loads

logger = logging.getLogger(__name__)


def _intent_filename(cache_key: str) -> str:
    """Derive a filesystem-safe filename from a cache key."""
    import xxhash

    return xxhash.xxh64(cache_key.encode()).hexdigest() + ".intent"


class WriteIntentJournal:
    """File-based write intent journal for crash recovery.

    Each pending blob write is recorded as a small JSON file in
    ``{cache_dir}/.intents/``.  On successful metadata commit the
    intent file is removed.  If the process crashes between blob write
    and metadata commit, the intent file survives and is cleaned up on
    the next cache init.
    """

    def __init__(self, cache_dir: Path, stale_threshold_seconds: float = 300.0):
        self._cache_dir = Path(cache_dir)
        self._intents_dir = self._cache_dir / ".intents"
        self._stale_threshold = stale_threshold_seconds

    def _ensure_dir(self) -> None:
        """Lazily create the intents directory on first write."""
        self._intents_dir.mkdir(parents=True, exist_ok=True)

    def record_intent(self, cache_key: str, blob_path: str) -> Path:
        """Record a pending write intent before blob write completes.

        Returns the path to the intent file (for debugging/testing).
        """
        self._ensure_dir()
        intent_path = self._intents_dir / _intent_filename(cache_key)
        payload = json_dumps(
            {
                "cache_key": cache_key,
                "blob_path": blob_path,
                "created_at": time.time(),
            }
        )
        intent_path.write_text(payload, encoding="utf-8")
        return intent_path

    def clear_intent(self, cache_key: str) -> None:
        """Remove the intent file for *cache_key*.  No-op if absent."""
        intent_path = self._intents_dir / _intent_filename(cache_key)
        try:
            intent_path.unlink(missing_ok=True)
        except OSError:
            pass

    def cleanup_stale_intents(
        self, entry_exists: Optional[Callable[[str], bool]] = None
    ) -> int:
        """Delete orphaned blobs from stale (crashed) writes.

        Scans the intents directory for files older than the configured
        threshold.  For each stale intent the orphaned blob is deleted
        (if it still exists on disk) and the intent file is removed.

        Returns the number of cleaned-up intents.
        """
        if not self._intents_dir.exists():
            return 0

        now = time.time()
        cleaned = 0

        for intent_path in list(self._intents_dir.glob("*.intent")):
            try:
                data = json_loads(intent_path.read_text(encoding="utf-8"))
                created_at = data.get("created_at", 0)

                if now - created_at < self._stale_threshold:
                    continue  # still fresh — skip

                cache_key = data.get("cache_key", "")
                if entry_exists is not None and cache_key:
                    try:
                        if entry_exists(cache_key):
                            intent_path.unlink(missing_ok=True)
                            logger.info(
                                "Removed stale write intent for committed key: "
                                f"{cache_key}"
                            )
                            cleaned += 1
                            continue
                    except Exception:
                        logger.warning(
                            f"Failed to check committed entry for stale intent: {cache_key}",
                            exc_info=True,
                        )
                        continue

                # Delete the orphaned blob if it exists
                blob_path = data.get("blob_path")
                if blob_path:
                    bp = Path(blob_path)
                    if not bp.is_absolute():
                        bp = self._cache_dir / bp
                    if bp.exists():
                        bp.unlink()
                        logger.info(
                            f"Deleted orphaned blob from stale write intent: {bp}"
                        )

                # Remove the intent file
                intent_path.unlink(missing_ok=True)
                logger.info(f"Cleaned stale write intent for key: {cache_key or '?'}")
                cleaned += 1

            except Exception:  # intentionally broad — cleanup must not crash init
                logger.warning(
                    f"Failed to process stale intent file: {intent_path}",
                    exc_info=True,
                )

        return cleaned
