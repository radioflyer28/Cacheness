"""Inline-blob helpers for UnifiedCache put/get flows."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from .interfaces import HandlerResult

if TYPE_CHECKING:
    from ._put_cleanup import _PutCleanup

logger = logging.getLogger(__name__)


class InlineBlobMixin:
    """_try_inline_blob, _try_direct_inline, _read_inline_blob, _build_metadata_dict, _cleanup_stale_blob."""

    def _try_inline_blob(
        self,
        result: HandlerResult,
        file_hash: Optional[str],
        cleanup: "_PutCleanup",
    ) -> Optional[Dict[str, Any]]:
        """Try to inline a small blob into the metadata row.

        If the blob is small enough (≤ ``max_inline_size`` from config),
        reads the file bytes, computes an xxhash from the bytes, and
        returns a dict with ``blob_data``, ``is_inline=1``, and
        ``file_hash``.  The blob file is deleted and the cleanup guard
        is disarmed so rollback won't try to delete it again.

        Returns ``None`` when inlining is disabled or the blob is too large.
        """
        max_inline = self.config.blob.max_inline_size
        if max_inline <= 0:
            return None
        if result.file_size > max_inline:
            return None

        # Resolve the blob path written by _write_blob
        actual_path_str = result.actual_path
        if "://" in actual_path_str:
            # Remote blobs (S3, etc.) are not inlined
            return None

        blob_path = self._resolve_actual_path(actual_path_str)
        if not isinstance(blob_path, Path) or not blob_path.exists():
            return None

        blob_bytes = blob_path.read_bytes()

        # Compute hash from the raw bytes (matches file-based hashing)
        computed_hash = file_hash
        if computed_hash is None and self.config.metadata.verify_cache_integrity:
            import xxhash

            computed_hash = xxhash.xxh3_64(blob_bytes).hexdigest()

        # Preserve the original file suffix so _read_inline_blob can
        # recreate a temp file that the handler recognises (handlers
        # derive the expected path from the suffix, e.g. .pkl.zstd).
        # Extract all suffixes after the hash portion of the filename.
        inline_ext = "".join(blob_path.suffixes) or ".bin"

        # Delete the blob file — data now lives in metadata
        try:
            blob_path.unlink()
        except OSError as exc:
            logger.warning(f"Failed to remove inlined blob file {blob_path}: {exc}")

        # Disarm the cleanup guard so rollback doesn't try to delete
        cleanup.blob_path = None

        return {
            "blob_data": blob_bytes,
            "is_inline": 1,
            "file_hash": computed_hash,
            "inline_ext": inline_ext,
        }

    def _try_direct_inline(
        self,
        data: Any,
        handler: Any,
    ) -> Optional[Dict[str, Any]]:
        """Try zero-disk in-memory serialization for inline blob storage.

        Invokes ``handler.put_bytes()`` to serialize *data* entirely in
        memory.  If the handler supports it and the serialized blob fits
        within ``max_inline_size``, returns a dict ready for embedding in
        the metadata row — **no file is ever written to disk**.

        Returns ``None`` when:
        * Inlining is disabled (``max_inline_size ≤ 0``).
        * The handler raises :class:`NotImplementedError`.
        * The serialized blob exceeds ``max_inline_size``.

        The returned dict contains:

        * ``blob_data`` – the raw bytes to store in the metadata row.
        * ``file_hash`` – xxhash digest (or ``None`` when integrity
          checking is disabled).
        * ``inline_ext`` – file extension hint for ``get_bytes``/fallback.
        * ``result`` – :class:`HandlerResult` with serialization metadata
          (``storage_format``, ``compression_codec``, etc.).
        * ``handler`` – the handler instance (for ``data_type``).
        """
        max_inline = self.config.blob.max_inline_size
        if max_inline <= 0:
            return None

        try:
            blob_bytes, result = handler.put_bytes(data, self.config)
        except (NotImplementedError, Exception) as exc:
            # NotImplementedError → handler doesn't support in-memory path.
            # Any other exception → safer to fall back to disk path.
            if not isinstance(exc, NotImplementedError):
                logger.debug(
                    "put_bytes failed for %s, falling back to disk: %s",
                    handler.data_type,
                    exc,
                )
            return None

        if len(blob_bytes) > max_inline:
            return None

        # Hash is computed below on blob_data (ciphertext when encrypted, plaintext otherwise)
        computed_hash: Optional[str] = None

        # Encrypt inline blob if encryption is enabled (D-01)
        encryption_meta: dict[str, str] = {}
        if (
            hasattr(self, "_blob_store")
            and self._blob_store is not None
            and self._blob_store._encryption_key is not None
        ):
            from .encryption import encrypt_blob

            ciphertext, iv, algo = encrypt_blob(
                blob_bytes, self._blob_store._encryption_key
            )
            blob_bytes = ciphertext
            encryption_meta["encryption_algorithm"] = algo.decode()
            encryption_meta["encryption_iv"] = iv.hex()

        # Hash what's stored in blob_data (ciphertext when encrypted,
        # plaintext otherwise) — consistent with file-backed entries
        # and the verification mixin which hashes blob_data directly.
        if self.config.metadata.verify_cache_integrity:
            import xxhash

            computed_hash = xxhash.xxh3_64(blob_bytes).hexdigest()

        inline_ext = handler.get_file_extension(self.config)

        return {
            "blob_data": blob_bytes,
            "file_hash": computed_hash,
            "inline_ext": inline_ext,
            "result": result,
            "handler": handler,
            **encryption_meta,
        }

    def _read_inline_blob(
        self,
        entry: Dict[str, Any],
        data_type: str,
        metadata: Dict[str, Any],
    ) -> Any:
        """Deserialize an inline blob without touching the blob backend.

        Tries the handler's ``get_bytes()`` first for zero-disk
        deserialization.  Falls back to writing the raw bytes to a
        temporary file and delegating to the handler's ``get()`` method.
        """
        blob_bytes: bytes = entry["blob_data"]

        # Decrypt inline blob if encryption metadata is present (D-02)
        enc_algo = metadata.get("encryption_algorithm")
        if enc_algo:
            if (
                not hasattr(self, "_blob_store")
                or self._blob_store is None
                or self._blob_store._encryption_key is None
            ):
                logger.warning(
                    "Encrypted inline blob for %s but no encryption key configured",
                    data_type,
                )
                return None
            from .encryption import decrypt_blob

            iv = bytes.fromhex(metadata["encryption_iv"])
            blob_bytes = decrypt_blob(blob_bytes, self._blob_store._encryption_key, iv)

        # Fast path — zero-disk deserialization via get_bytes()
        try:
            handler = self._blob_store.handlers.get_handler_by_type(data_type)
            return handler.get_bytes(blob_bytes, metadata)
        except NotImplementedError:
            pass  # Fall through to temp-file path
        except Exception as exc:  # intentionally broad — handler may raise anything
            logger.debug(
                "get_bytes failed for %s, falling back to temp file: %s",
                data_type,
                exc,
            )

        # Slow path — write to temp file, delegate to handler.get()
        import tempfile

        suffix = metadata.get("inline_ext", ".bin")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(blob_bytes)
            tmp_path = Path(tmp.name)

        try:
            return self._blob_store._read_blob(tmp_path, data_type, metadata)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def _build_metadata_dict(
        self, result: HandlerResult, file_hash: Optional[str]
    ) -> Dict[str, Any]:
        """Build the metadata dict from a :class:`HandlerResult`.

        Merges handler ``extra`` fields with top-level columns
        (``actual_path``, ``file_hash``, ``storage_format``, etc.)
        so the backend can extract them to dedicated columns.
        """
        metadata_dict: Dict[str, Any] = {
            **result.extra,
            "actual_path": result.actual_path,
            "file_hash": file_hash,
            "storage_format": result.storage_format,
        }
        if result.serializer:
            metadata_dict["serializer"] = result.serializer
        if result.compression_codec:
            metadata_dict["compression_codec"] = result.compression_codec
        if result.object_type:
            metadata_dict["object_type"] = result.object_type
        return metadata_dict

    def _cleanup_stale_blob(
        self,
        cache_key: str,
        old_blob_path: Optional[str],
        new_actual_path: str,
    ) -> None:
        """Remove old blob file when the actual path changed.

        This happens when a data type change causes a different file
        extension (e.g. ``.parquet`` → ``.pkl.lz4``).  Best-effort:
        failure just logs a warning and leaves an orphan for
        ``verify_integrity`` to clean up later.
        """
        if not old_blob_path or old_blob_path == new_actual_path:
            return
        try:
            if "://" in old_blob_path:
                self._blob_store.blob_backend.delete_blob(old_blob_path)
            else:
                old_resolved = self._resolve_actual_path(old_blob_path)
                if isinstance(old_resolved, Path) and old_resolved.exists():
                    old_resolved.unlink()
            logger.debug(f"Cleaned up old blob for {cache_key}: {old_blob_path}")
        except (OSError, IOError) as exc:
            logger.warning(
                f"Failed to clean up old blob for {cache_key} at {old_blob_path}: {exc}"
            )
