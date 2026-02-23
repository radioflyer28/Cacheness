"""Unit conversion utilities for Cacheness.

Size Convention
---------------
- **Internal storage / calculations**: all sizes in **bytes** (``int``).
- **Config / public API**: accepts human-readable strings (``"2GB"``)
  or raw byte counts.
- **Display**: converted to MB only at output boundaries using
  :func:`format_size` or :func:`bytes_to_mb_display`.

Duration Convention
-------------------
- **Internal**: all durations in **seconds** (``float``).
- **Config / public API**: accepts human-readable strings with
  shorthand suffixes (``s``, ``m``, ``h``, ``d``, ``w``, ``mo``, ``y``)
  or raw seconds.
- **Display**: converted via :func:`format_duration`.

These helpers centralise every unit conversion so that rounding errors
cannot sneak into cleanup, eviction, or TTL logic.
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


# =====================================================================
# Duration parsing / formatting
# =====================================================================

_SECONDS_PER_MINUTE: float = 60
_SECONDS_PER_HOUR: float = 3600
_SECONDS_PER_DAY: float = 86400
_SECONDS_PER_WEEK: float = 604800
_SECONDS_PER_MONTH: float = 86400 * 30  # 30 days
_SECONDS_PER_YEAR: float = 86400 * 365  # 365 days

# Shorthand-only suffixes → multiplier (seconds).
# "mo" must sort before "m" in the regex alternation (longest first).
_DURATION_UNITS: dict[str, float] = {
    "mo": _SECONDS_PER_MONTH,
    "s": 1,
    "m": _SECONDS_PER_MINUTE,
    "h": _SECONDS_PER_HOUR,
    "d": _SECONDS_PER_DAY,
    "w": _SECONDS_PER_WEEK,
    "y": _SECONDS_PER_YEAR,
}

# Build regex alternation from longest suffix first to avoid partial matches.
_DURATION_SUFFIX_RE: str = "|".join(
    sorted(list(_DURATION_UNITS.keys()), key=len, reverse=True)
)
_DURATION_RE = re.compile(
    rf"^(\d+(?:\.\d+)?)\s*({_DURATION_SUFFIX_RE})$",
    re.IGNORECASE,
)


def parse_duration(value: str | int | float) -> float:
    """Parse a human-readable duration string into seconds.

    Accepts:
    - An ``int`` or ``float`` (treated as seconds).
    - A string with a single value and shorthand unit, e.g. ``"30s"``,
      ``"5m"``, ``"6h"``, ``"7d"``, ``"2w"``, ``"3mo"``, ``"1y"``.
      Case-insensitive.  If no unit suffix is found the string is
      interpreted as seconds.

    Returns:
        float: Duration in seconds.

    Raises:
        ValueError: If the string cannot be parsed.
        TypeError: If the value is not str/int/float.

    Examples:
        >>> parse_duration("30s")
        30.0
        >>> parse_duration("5m")
        300.0
        >>> parse_duration("6h")
        21600.0
        >>> parse_duration("7d")
        604800.0
        >>> parse_duration("3mo")
        7776000.0
        >>> parse_duration("1y")
        31536000.0
        >>> parse_duration(3600)
        3600.0
        >>> parse_duration("1.5h")
        5400.0
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise TypeError(f"Expected str, int, or float, got {type(value).__name__}")

    value = value.strip()
    if not value:
        raise ValueError("Empty duration string")

    match = _DURATION_RE.match(value)
    if match:
        number = float(match.group(1))
        unit = match.group(2).lower()
        return number * _DURATION_UNITS[unit]

    # Try plain numeric string (interpret as seconds)
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"Cannot parse duration: {value!r}.  "
            f"Expected a number with optional unit "
            f"(s, m, h, d, w, mo, y)."
        ) from None


def resolve_ttl(
    ttl: "str | None" = None,
    ttl_seconds: "float | int | None" = None,
    *,
    _param_owner: str = "",
) -> "float | None":
    """Resolve ``ttl`` / ``ttl_seconds`` into a numeric seconds value.

    Rules:
    - ``ttl`` accepts a human-readable duration string (``"6h"``, ``"2d"``).
    - ``ttl_seconds`` accepts a numeric value (int or float) only.
    - Passing **both** raises ``ValueError``.
    - Passing a *string* to ``ttl_seconds`` raises ``TypeError`` with a
      helpful message telling the caller to use ``ttl=`` instead.
    - If neither is provided, returns ``None`` (meaning "use the default").

    Args:
        ttl: Human-readable duration string.
        ttl_seconds: Numeric TTL in seconds.
        _param_owner: Optional context string for error messages
            (e.g. ``"@cached"``).

    Returns:
        Resolved TTL in seconds, or ``None`` when both inputs are ``None``.
    """
    ctx = f" in {_param_owner}" if _param_owner else ""

    # Guard: ttl_seconds must be numeric if provided
    if ttl_seconds is not None and isinstance(ttl_seconds, str):
        raise TypeError(
            f"ttl_seconds{ctx} must be numeric (int/float). "
            f'Use ttl="{ttl_seconds}" for duration strings.'
        )

    # Guard: mutually exclusive
    if ttl is not None and ttl_seconds is not None:
        raise ValueError(
            f"Cannot specify both ttl and ttl_seconds{ctx}. "
            f"Use ttl for duration strings or ttl_seconds for numeric values."
        )

    if ttl is not None:
        return parse_duration(ttl)
    if ttl_seconds is not None:
        return float(ttl_seconds)
    return None


def format_duration(seconds: float) -> str:
    """Format seconds as a concise human-readable duration string.

    Chooses the largest single unit that keeps the numeric part ≥ 1.
    For exact multiples the result is an integer (``"7d"``); otherwise
    it uses up to two decimal places (``"1.50h"``).

    Examples:
        >>> format_duration(86400)
        '1d'
        >>> format_duration(3600)
        '1h'
        >>> format_duration(300)
        '5m'
        >>> format_duration(45)
        '45s'
        >>> format_duration(5400)
        '1.50h'
    """
    if seconds <= 0:
        return "0s"

    for suffix, divisor in (
        ("y", _SECONDS_PER_YEAR),
        ("mo", _SECONDS_PER_MONTH),
        ("w", _SECONDS_PER_WEEK),
        ("d", _SECONDS_PER_DAY),
        ("h", _SECONDS_PER_HOUR),
        ("m", _SECONDS_PER_MINUTE),
    ):
        if seconds >= divisor:
            val = seconds / divisor
            if val == int(val):
                return f"{int(val)}{suffix}"
            return f"{val:.2f}{suffix}"
    # Less than a minute — show seconds
    if seconds == int(seconds):
        return f"{int(seconds)}s"
    return f"{seconds:.2f}s"
