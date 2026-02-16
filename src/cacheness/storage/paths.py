"""Shared path helpers for normalizing cache entry paths.

Cacheness stores ``actual_path`` in metadata as a relative,
forward-slash string (e.g. ``default/abcdef01.pkl``).  These
helpers convert between the stored format and absolute filesystem
paths used at runtime.
"""

from __future__ import annotations

from pathlib import Path, PurePath


def resolve_actual_path(actual_path_str: str, cache_dir: Path) -> Path | str:
    """Resolve a stored ``actual_path`` to a usable path.

    * **URIs** (containing ``://``) are returned as a plain ``str``
      — wrapping in ``Path`` would mangle the scheme on Windows.
    * **Relative paths** (new format) are joined with *cache_dir*
      and returned as a ``Path``.
    * **Legacy absolute paths** are returned as a ``Path``.

    Callers that require a ``Path`` should check for URIs first, or
    call ``str()`` on the result when passing to backends.
    """
    if "://" in actual_path_str:
        return actual_path_str

    p = Path(actual_path_str)
    if p.is_absolute():
        return p

    return cache_dir / p


def to_relative_path(path_str: str, cache_dir: Path) -> str:
    """Convert an absolute path to a cache-dir-relative, forward-slash path.

    URIs (containing ``://``) are returned unchanged.  Absolute paths
    that start with *cache_dir* have the prefix stripped.  All
    backslashes are replaced with forward slashes so that metadata is
    portable across platforms.
    """
    if "://" in path_str:
        return path_str

    p = Path(path_str).resolve()
    cache_root = cache_dir.resolve()

    try:
        rel: PurePath = p.relative_to(cache_root)
    except ValueError:
        # Already relative or from a different root — normalise separators
        rel = PurePath(path_str)

    return rel.as_posix()
