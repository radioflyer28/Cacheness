"""File convenience operations mixin for UnifiedCache."""

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FileOpsMixin:
    """put_file, get_file, and supporting helpers."""

    def put_file(
        self,
        file_path: str | Path,
        *,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        description: str = "",
        custom_metadata=None,
        move: bool = False,
        **kwargs,
    ) -> str:
        """Store an arbitrary file in the cache.

        Reads the file into memory as raw bytes and delegates to
        :meth:`put`.  File metadata (original filename, MIME type,
        file size) is automatically recorded in ``metadata_dict``
        when ``store_full_metadata=True``.

        Args:
            file_path: Path to the source file (``str`` or ``pathlib.Path``).
            cache_key: Explicit cache key.  When provided, *on* and
                ``**kwargs`` are ignored for key derivation.
            on: Dictionary of key parameters for cache key derivation.
            description: Human-readable description.
            custom_metadata: Custom metadata for the cache entry.  Supports
                single ORM objects, lists/tuples of ORM objects, or dicts.
                Passed through to :meth:`put` unchanged.
            move: If ``True``, delete the source file after a successful
                store (move-in semantics).  Defaults to ``False`` (copy-in).
            **kwargs: Extra key-value pairs for key derivation and/or
                ``metadata_dict`` (when ``store_full_metadata=True``).

        Returns:
            The 16-character hex cache key.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            IsADirectoryError: If *file_path* is a directory.

        Example:
            key = cache.put_file("data/model.onnx",
                                  description="ONNX model v2")
            key = cache.put_file("output.csv", on={"run": "exp_01"})
        """
        import mimetypes

        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file does not exist: {src}")
        if src.is_dir():
            raise IsADirectoryError(f"Expected a file, got a directory: {src}")

        data = src.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(src))

        # File metadata must NOT participate in cache key derivation.
        # Resolve the key from user-supplied params first, then pass
        # everything (file meta + user kwargs) with an explicit cache_key
        # so that _resolve_cache_key returns immediately.
        file_meta = {
            "original_filename": src.name,
            "mime_type": mime_type or "application/octet-stream",
            "original_size": len(data),
        }
        merged_kwargs = {**file_meta, **kwargs}  # user kwargs win on conflict

        if cache_key is None:
            cache_key = self._resolve_cache_key(None, on, kwargs)

        result_key = self.put(
            data,
            cache_key=cache_key,
            description=description,
            custom_metadata=custom_metadata,
            **merged_kwargs,
        )

        if move:
            src.unlink()

        return result_key

    def get_file(
        self,
        cache_key: Optional[str] = None,
        *,
        dest: str | Path | None = None,
        on: Optional[Dict] = None,
        ttl: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        move: bool = False,
        overwrite: bool = True,
        **kwargs,
    ) -> Optional[bytes | Path]:
        """Retrieve cached file data, optionally writing it to disk.

        This is the read counterpart of :meth:`put_file`.  When *dest*
        is provided the raw bytes are written to that path and a
        ``pathlib.Path`` is returned.  Otherwise the raw ``bytes`` are
        returned directly.

        Args:
            cache_key: Explicit cache key.  When provided, *on* and
                ``**kwargs`` are ignored.
            dest: Optional destination path.  Parent directories are
                created automatically.  If *dest* is a directory, the
                original filename from metadata is used (falls back to
                ``<cache_key>.bin`` when unavailable).
            on: Dictionary of key parameters for key lookup.
            ttl: TTL as a human-readable duration string (e.g. ``"6h"``).
            ttl_seconds: TTL in seconds.  Mutually exclusive with *ttl*.
            move: If ``True``, delete the cache entry after a successful
                write to *dest* (move-out semantics).  Requires *dest*
                to be set — raises ``ValueError`` otherwise.  Defaults
                to ``False`` (copy-out).
            overwrite: If ``False``, raise ``FileExistsError`` when
                *dest* already exists on disk.  Defaults to ``True``
                (silently overwrite).
            **kwargs: Key-value pairs for key derivation (must match what
                was passed to :meth:`put_file`).

        Returns:
            * ``bytes`` — when *dest* is ``None`` and entry exists.
            * ``pathlib.Path`` — when *dest* is given and entry exists.
            * ``None`` — on a cache miss.

        Raises:
            ValueError: If *move* is ``True`` but *dest* is ``None``.
            FileExistsError: If *overwrite* is ``False`` and *dest*
                already exists.

        Example:
            raw = cache.get_file("abc123def4567890")
            path = cache.get_file("abc123def4567890",
                                   dest="output/model.onnx")
        """
        if move and dest is None:
            raise ValueError(
                "move=True requires dest to be set. "
                "Without a destination path, use get() + invalidate() instead."
            )

        data = self.get(
            cache_key=cache_key,
            on=on,
            ttl=ttl,
            ttl_seconds=ttl_seconds,
            **kwargs,
        )
        if data is None:
            return None

        # Ensure we have bytes (in case the entry was stored without put_file)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"Expected bytes from cache, got {type(data).__name__}. "
                "get_file() should only be used with entries stored via put_file()."
            )
        raw = bytes(data) if not isinstance(data, bytes) else data

        if dest is None:
            return raw

        dest_path = Path(dest)
        if dest_path.is_dir():
            # Resolve filename from metadata
            resolved_key = (
                cache_key
                if cache_key is not None
                else self._resolve_cache_key(None, on, kwargs)
            )
            filename = self._resolve_original_filename(resolved_key)
            dest_path = dest_path / filename

        if not overwrite and dest_path.exists():
            raise FileExistsError(
                f"Destination already exists: {dest_path}. "
                "Pass overwrite=True to overwrite."
            )

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(raw)

        if move:
            resolved_key = (
                cache_key
                if cache_key is not None
                else self._resolve_cache_key(None, on, kwargs)
            )
            self.invalidate(cache_key=resolved_key)

        return dest_path

    def _resolve_original_filename(self, cache_key: str) -> str:
        """Look up the original filename from metadata, with fallback."""
        entry = self.metadata_backend.get_entry(cache_key)
        if entry:
            meta = self._extract_metadata_dict(entry)
            name = meta.get("original_filename")
            if name:
                return name
        return f"{cache_key}.bin"
