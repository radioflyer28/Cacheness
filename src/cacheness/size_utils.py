"""Size conversion utilities for Cacheness.

Convention
----------
- **Internal storage / calculations**: all sizes in **bytes** (``int``).
- **Config / public API**: accepts human-readable strings (``"2GB"``)
  or raw byte counts.
- **Display**: converted to MB only at output boundaries using
  :func:`format_size` or :func:`bytes_to_mb_display`.

These helpers centralise every byte ↔ human-readable conversion so that
rounding errors cannot sneak into cleanup or eviction logic.
"""

from __future__ import annotations

import re

_BYTES_PER_KB: int = 1024
_BYTES_PER_MB: int = 1024**2
_BYTES_PER_GB: int = 1024**3
_BYTES_PER_TB: int = 1024**4

_UNITS: dict[str, int] = {
    "B": 1,
    "KB": _BYTES_PER_KB,
    "MB": _BYTES_PER_MB,
    "GB": _BYTES_PER_GB,
    "TB": _BYTES_PER_TB,
}


def parse_size(value: str | int | float) -> int:
    """Parse a human-readable size string into bytes.

    Accepts:
    - An ``int`` or ``float`` (treated as raw bytes, truncated to int).
    - A string like ``"500MB"``, ``"2.5 GB"``, ``"1024"``, ``"100 KB"``.
      Unit matching is case-insensitive.  If no unit suffix is found the
      string is interpreted as bytes.

    Returns:
        int: Size in bytes.

    Raises:
        ValueError: If the string cannot be parsed.

    Examples:
        >>> parse_size("7MB")
        7340032
        >>> parse_size("2.5 GB")
        2684354560
        >>> parse_size(1048576)
        1048576
        >>> parse_size("1024")
        1024
    """
    if isinstance(value, (int, float)):
        return int(value)

    if not isinstance(value, str):
        raise TypeError(f"Expected str, int, or float, got {type(value).__name__}")

    value = value.strip()
    if not value:
        raise ValueError("Empty size string")

    # Try matching with unit suffix
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)$", value, re.IGNORECASE)
    if match:
        number = float(match.group(1))
        unit = match.group(2).upper()
        return int(number * _UNITS[unit])

    # Try plain numeric string (interpret as bytes)
    try:
        return int(float(value))
    except ValueError:
        raise ValueError(
            f"Cannot parse size: {value!r}.  "
            f"Expected a number with optional unit (B, KB, MB, GB, TB)."
        ) from None


def format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Chooses the largest unit that keeps the numeric part ≥ 1.

    Examples:
        >>> format_size(1048576)
        '1.00 MB'
        >>> format_size(500)
        '500 B'
        >>> format_size(2684354560)
        '2.50 GB'
    """
    for unit, threshold in (
        ("TB", _BYTES_PER_TB),
        ("GB", _BYTES_PER_GB),
        ("MB", _BYTES_PER_MB),
        ("KB", _BYTES_PER_KB),
    ):
        if size_bytes >= threshold:
            return f"{size_bytes / threshold:.2f} {unit}"
    return f"{size_bytes} B"


def bytes_to_mb_display(size_bytes: int, decimals: int = 3) -> float:
    """Convert bytes → megabytes, rounded for **display only**.

    Never use the return value in calculations that feed back into
    cleanup or eviction logic — use raw byte counts instead.

    Examples:
        >>> bytes_to_mb_display(5400)
        0.005
        >>> bytes_to_mb_display(1048576)
        1.0
    """
    return round(size_bytes / _BYTES_PER_MB, decimals)
