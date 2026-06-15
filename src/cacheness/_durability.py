"""Local filesystem durability helpers."""

import logging
import os
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)


def flush_and_fsync(file_obj: IO[Any]) -> None:
    """Flush a local file descriptor and fsync it."""
    file_obj.flush()
    os.fsync(file_obj.fileno())


def fsync_parent_dir(path: str | Path) -> None:
    """Best-effort fsync for the directory containing *path*."""
    parent = Path(path).parent
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY

    try:
        fd = os.open(parent, flags)
    except OSError as exc:
        logger.debug("Parent directory fsync unavailable for %s: %s", parent, exc)
        return

    try:
        os.fsync(fd)
    except OSError as exc:
        logger.debug("Parent directory fsync failed for %s: %s", parent, exc)
    finally:
        os.close(fd)
